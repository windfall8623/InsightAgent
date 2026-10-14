"""Observability: spans, cost/latency accounting, and anomaly detection.

Implements the OTel-shaped idea (a trace is a tree of timed spans with
attributes) without the OpenTelemetry dependency, because a portfolio project
should be runnable with zero infrastructure. The span records are JSONL so any
tracing backend can ingest them later.

Layer-3 security monitoring lives here too: unusual tool-call rates and blocked
calls are surfaced as anomalies rather than swallowed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Span:
    """One timed unit of work within a trace."""

    name: str
    trace_id: str
    parent: str | None = None
    start: float = field(default_factory=time.time)
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.time()) - self.start) * 1000


# Deterministic cost model. Real deployments read live pricing from the
# provider response; here the numbers are fixed so cost metrics stay
# reproducible across runs.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model -> (input cost per 1k tokens, output cost per 1k tokens)
    "sim-small": (0.0008, 0.0024),
    "sim-large": (0.0040, 0.0120),
}


@dataclass(slots=True)
class UsageRecord:
    """Token/cost accounting for one model call."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    route_reason: str = ""

    @property
    def cost_usd(self) -> float:
        in_rate, out_rate = MODEL_PRICING.get(self.model, MODEL_PRICING["sim-large"])
        return (self.input_tokens / 1000) * in_rate + (self.output_tokens / 1000) * out_rate


class Tracer:
    """Collects spans and usage for a run, and can persist them as JSONL or export via OTLP."""

    def __init__(
        self,
        *,
        sink: Path | None = None,
        enabled: bool = True,
        exporter: OTLPHttpExporter | None = None,
        auto_otlp: bool = True,
    ) -> None:
        self.spans: list[Span] = []
        self.usage: list[UsageRecord] = []
        self._sink = sink
        self.enabled = enabled
        if exporter is not None:
            self.exporter: OTLPHttpExporter | None = exporter
        elif auto_otlp and (
            os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        ):
            self.exporter = OTLPHttpExporter()
        else:
            self.exporter = None

    def flush_otlp(self, *, exporter: OTLPHttpExporter | None = None) -> bool:
        """Export spans to the OpenTelemetry OTLP collector."""
        exp = exporter or self.exporter
        if exp and self.spans:
            return exp.export(self.spans)
        return False

    @contextmanager
    def span(
        self, name: str, trace_id: str, parent: str | None = None, **attrs: Any
    ) -> Iterator[Span]:
        record = Span(name=name, trace_id=trace_id, parent=parent, attributes=dict(attrs))
        if self.enabled:
            self.spans.append(record)
        try:
            yield record
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.end = time.time()

    def record_usage(self, record: UsageRecord) -> None:
        if self.enabled:
            self.usage.append(record)

    # -- aggregates -------------------------------------------------------

    @property
    def total_cost(self) -> float:
        return sum(u.cost_usd for u in self.usage)

    @property
    def total_tokens(self) -> int:
        return sum(u.input_tokens + u.output_tokens for u in self.usage)

    def summary(self) -> dict[str, Any]:
        by_model: dict[str, dict[str, Any]] = {}
        for record in self.usage:
            entry = by_model.setdefault(
                record.model,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_ms": 0.0,
                },
            )
            entry["calls"] += 1
            entry["input_tokens"] += record.input_tokens
            entry["output_tokens"] += record.output_tokens
            entry["cost_usd"] = round(entry["cost_usd"] + record.cost_usd, 6)
            entry["latency_ms"] = round(entry["latency_ms"] + record.latency_ms, 2)
        for entry in by_model.values():
            entry["avg_latency_ms"] = round(entry["latency_ms"] / max(entry["calls"], 1), 2)
        return {
            "spans": len(self.spans),
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "by_model": by_model,
        }

    def flush(self) -> Path | None:
        self.flush_otlp()
        if not self._sink:
            return None
        self._sink.parent.mkdir(parents=True, exist_ok=True)
        with self._sink.open("a", encoding="utf-8") as handle:
            for span in self.spans:
                handle.write(json.dumps(asdict(span), ensure_ascii=False, default=str) + "\n")
            for record in self.usage:
                handle.write(json.dumps({"usage": asdict(record)}, ensure_ascii=False) + "\n")
        return self._sink


class AnomalyDetector:
    """Layer-3 monitoring: flags run patterns worth a human's attention."""

    def __init__(self, *, max_tool_calls: int = 12, max_reflections: int = 3) -> None:
        self.max_tool_calls = max_tool_calls
        self.max_reflections = max_reflections
        self.alerts: list[dict[str, str]] = []

    def observe(self, *, tool_calls: int, blocked: int, reflections: int, injection: bool) -> None:
        if tool_calls > self.max_tool_calls:
            self._alert(
                "tool_call_volume", f"{tool_calls} tool calls exceeds budget {self.max_tool_calls}"
            )
        if blocked > 0:
            self._alert("blocked_calls", f"{blocked} tool call(s) blocked by policy")
        if reflections > self.max_reflections:
            self._alert("reflection_loop", f"{reflections} reflection rounds suggests a loop")
        if injection:
            self._alert("prompt_injection", "prompt injection detected in user input")

    def _alert(self, kind: str, message: str) -> None:
        entry = {"kind": kind, "message": message}
        if entry not in self.alerts:
            self.alerts.append(entry)


# --------------------------------------------------------------------------
# Trace replay
# --------------------------------------------------------------------------


@dataclass(slots=True)
class TraceRecord:
    """A complete, replayable record of one agent run.

    Spans say what happened; this says what was asked and what came back, which
    is what a replay needs in order to reproduce the run.
    """

    trace_id: str
    question: str
    sql: str = ""
    answer: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    blocked: bool = False
    model: str = ""
    strategy: str = ""
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    agents_used: list[str] = field(default_factory=list)
    recorded_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class ReplayDiff:
    """How a replayed run differed from the recording."""

    trace_id: str
    matched: bool
    differences: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "matched": self.matched,
            "differences": self.differences,
        }


class TraceStore:
    """Persists run records and replays them, comparing outcome to original.

    Replay is the self-hosted equivalent of LangGraph's "time travel": re-execute
    a recorded run against current code and see exactly what drifted. It is how
    a behaviour change gets attributed instead of guessed at.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._records: dict[str, TraceRecord] = {}
        if path and path.exists():
            self._load()

    def record(self, record: TraceRecord) -> None:
        self._records[record.trace_id] = record

    def get(self, trace_id: str) -> TraceRecord | None:
        return self._records.get(trace_id)

    def all(self) -> list[TraceRecord]:
        return sorted(self._records.values(), key=lambda r: r.recorded_at)

    async def replay(self, trace_id: str, rerun: Callable[[TraceRecord], Any]) -> ReplayDiff:
        """Re-run a recorded trace and diff the outcome against the recording.

        Async because every caller of a recorded run is the agent itself, which
        is async. A sync rerun callable still works — the result is inspected
        for awaitability rather than assumed.
        """

        original = self._records.get(trace_id)
        if original is None:
            raise KeyError(f"unknown trace_id: {trace_id}")

        fresh = rerun(original)
        if inspect.isawaitable(fresh):
            fresh = await fresh
        differences: list[str] = []

        def compare(label: str, before: Any, after: Any) -> None:
            if before != after:
                differences.append(f"{label}: {before!r} -> {after!r}")

        compare("sql", original.sql, getattr(fresh, "sql", None))
        compare("columns", original.columns, getattr(fresh, "columns", None))
        compare("rows", original.rows, getattr(fresh, "rows", None))
        compare("blocked", original.blocked, getattr(fresh, "blocked", None))

        return ReplayDiff(trace_id=trace_id, matched=not differences, differences=differences)

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                self._records[payload["trace_id"]] = TraceRecord(**payload)

    def save(self) -> Path | None:
        if not self.path:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for record in self._records.values():
                handle.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + chr(10))
        return self.path


# --------------------------------------------------------------------------
# OpenTelemetry export
# --------------------------------------------------------------------------

# OTLP/JSON trace envelope. Emitted by hand rather than through the OTel SDK so
# the project keeps zero runtime dependencies while still speaking the standard
# format that Jaeger, Tempo and other collectors accept.
OTLP_VERSION = "0.3.0"


def to_otlp(spans: Sequence[Span], *, service_name: str = "insight-agent") -> dict[str, Any]:
    """Render spans as an OTLP/JSON ``ExportTraceServiceRequest``."""

    def hex_id(trace_id: str) -> str:
        digest = hashlib.sha256(trace_id.encode("utf-8")).hexdigest()
        return digest[:32]  # 16 bytes, the OTLP trace-id width

    otlp_spans = []
    for span in spans:
        start_ns = int(span.start * 1_000_000_000)
        end_ns = int((span.end or time.time()) * 1_000_000_000)
        attributes = [
            {"key": k, "value": {"stringValue": str(v)}} for k, v in span.attributes.items()
        ]
        if span.error:
            attributes.append({"key": "error", "value": {"stringValue": span.error}})
        otlp_spans.append(
            {
                "traceId": hex_id(span.trace_id),
                "spanId": hashlib.sha256(f"{span.trace_id}:{span.name}".encode()).hexdigest()[:16],
                "parentSpanId": hashlib.sha256(
                    f"{span.trace_id}:{span.parent}".encode()
                ).hexdigest()[:16]
                if span.parent
                else "",
                "name": span.name,
                "kind": 1,
                "startTimeUnixNano": str(start_ns),
                "endTimeUnixNano": str(end_ns),
                "attributes": attributes,
                "status": {"code": 2 if span.error else 1, "message": span.error or ""},
            }
        )

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": service_name}}]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "insight_agent.app.obs", "version": "0.2.0"},
                        "spans": otlp_spans,
                    }
                ],
            }
        ]
    }


class OTLPHttpExporter:
    """Exports trace spans via standard OTLP/HTTP JSON protocol.

    Compatible with Jaeger, Tempo, SigNoz, and standard OpenTelemetry Collector
    HTTP receivers (defaulting to /v1/traces). Uses Python standard library
    urllib.request with zero third-party dependencies.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 3.0,
        service_name: str = "insight-agent",
        enabled: bool = True,
    ) -> None:
        raw_endpoint = (
            endpoint
            or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        )
        if raw_endpoint and not raw_endpoint.endswith("/v1/traces") and "/v1/" not in raw_endpoint:
            self.endpoint = raw_endpoint.rstrip("/") + "/v1/traces"
        else:
            self.endpoint = raw_endpoint

        self.headers = headers or {}
        self.timeout = timeout
        self.service_name = service_name
        self.enabled = enabled and bool(self.endpoint)
        self.last_export_status: int | None = None
        self.last_error: str | None = None

    def export(self, spans: Sequence[Span]) -> bool:
        """Export spans to the configured OTLP HTTP collector.

        Returns True if the collector responded with 2xx HTTP status.
        Never raises exceptions; errors are captured in self.last_error.
        """
        if not self.enabled or not self.endpoint or not spans:
            return False

        payload = to_otlp(spans, service_name=self.service_name)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers = {
            "Content-Type": "application/json",
            **self.headers,
        }
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers=req_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                self.last_export_status = resp.status
                self.last_error = None
                return 200 <= resp.status < 300
        except urllib.error.HTTPError as exc:
            self.last_export_status = exc.code
            self.last_error = f"HTTPError {exc.code}: {exc.reason}"
            return False
        except Exception as exc:
            self.last_export_status = None
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
