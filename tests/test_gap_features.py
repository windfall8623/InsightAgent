"""Tests for the four capability-gap items added from the JD analysis.

Covers: trace replay, OTLP export, context compression, and the BFCL adapter.
"""

from __future__ import annotations

import json

import pytest

from insight_agent.app.context import compress, extract_facts
from insight_agent.app.funccall import FunctionCall, FunctionSpec, decide
from insight_agent.app.obs import Span, TraceRecord, TraceStore, to_otlp
from insight_agent.core.llm import Message

# --------------------------------------------------------------------------
# Trace replay
# --------------------------------------------------------------------------


class _FakeAnswer:
    def __init__(self, sql: str, rows: list, blocked: bool = False) -> None:
        self.sql = sql
        self.rows = rows
        self.columns = ["n"]
        self.blocked = blocked


class TestTraceReplay:
    def _store(self) -> tuple[TraceStore, TraceRecord]:
        store = TraceStore()
        record = TraceRecord(
            trace_id="t1",
            question="总销售额是多少",
            sql="SELECT 1",
            rows=[{"n": 1}],
            columns=["n"],
        )
        store.record(record)
        return store, record

    async def test_identical_rerun_matches(self) -> None:
        store, _ = self._store()
        diff = await store.replay("t1", lambda r: _FakeAnswer("SELECT 1", [{"n": 1}]))
        assert diff.matched
        assert diff.differences == []

    async def test_changed_sql_is_reported(self) -> None:
        store, _ = self._store()
        diff = await store.replay("t1", lambda r: _FakeAnswer("SELECT 2", [{"n": 1}]))
        assert not diff.matched
        assert any("sql" in d for d in diff.differences)

    async def test_changed_rows_are_reported(self) -> None:
        store, _ = self._store()
        diff = await store.replay("t1", lambda r: _FakeAnswer("SELECT 1", [{"n": 2}]))
        assert not diff.matched

    async def test_async_rerun_is_awaited(self) -> None:
        store, _ = self._store()

        async def rerun(_):
            return _FakeAnswer("SELECT 1", [{"n": 1}])

        diff = await store.replay("t1", rerun)
        assert diff.matched

    async def test_unknown_trace_raises(self) -> None:
        store, _ = self._store()
        with pytest.raises(KeyError):
            await store.replay("nope", lambda r: _FakeAnswer("", []))

    def test_round_trips_through_disk(self, store_dir) -> None:
        path = store_dir / "traces.jsonl"
        store = TraceStore(path)
        store.record(TraceRecord(trace_id="t9", question="q", sql="SELECT 1"))
        store.save()

        reloaded = TraceStore(path)
        assert reloaded.get("t9") is not None
        assert reloaded.get("t9").sql == "SELECT 1"


# --------------------------------------------------------------------------
# OTLP export
# --------------------------------------------------------------------------


class TestOTLPExport:
    def _span(self, name: str = "sql", parent: str | None = None) -> Span:
        return Span(
            name=name,
            trace_id="trace-1",
            parent=parent,
            start=1_700_000_000.0,
            end=1_700_000_001.5,
            attributes={"sql": "SELECT 1"},
        )

    def test_envelope_shape(self) -> None:
        payload = to_otlp([self._span()])
        assert "resourceSpans" in payload
        scope = payload["resourceSpans"][0]["scopeSpans"][0]
        assert len(scope["spans"]) == 1

    def test_span_has_required_otlp_fields(self) -> None:
        span = to_otlp([self._span()])["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        for field in ("traceId", "spanId", "name", "startTimeUnixNano", "endTimeUnixNano"):
            assert field in span

    def test_trace_id_is_32_hex_chars(self) -> None:
        span = to_otlp([self._span()])["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert len(span["traceId"]) == 32
        int(span["traceId"], 16)  # valid hex

    def test_span_id_is_16_hex_chars(self) -> None:
        span = to_otlp([self._span()])["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert len(span["spanId"]) == 16

    def test_ids_are_stable_across_calls(self) -> None:
        first = to_otlp([self._span()])["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        second = to_otlp([self._span()])["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert first["traceId"] == second["traceId"]
        assert first["spanId"] == second["spanId"]

    def test_timestamps_are_nanoseconds(self) -> None:
        span = to_otlp([self._span()])["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"]) == 1_500_000_000

    def test_parent_linkage_is_emitted(self) -> None:
        child = to_otlp([self._span("plan", parent="root")])["resourceSpans"][0]["scopeSpans"][0][
            "spans"
        ][0]
        assert child["parentSpanId"]

    def test_root_span_has_no_parent(self) -> None:
        root = to_otlp([self._span()])["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert root["parentSpanId"] == ""

    def test_error_marks_status_and_attribute(self) -> None:
        span = Span(name="sql", trace_id="t", error="boom")
        payload = to_otlp([span])["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert payload["status"]["code"] == 2
        assert any(a["key"] == "error" for a in payload["attributes"])

    def test_service_name_is_carried(self) -> None:
        payload = to_otlp([self._span()], service_name="my-service")
        attrs = payload["resourceSpans"][0]["resource"]["attributes"]
        assert any(a["value"]["stringValue"] == "my-service" for a in attrs)

    def test_output_is_json_serialisable(self) -> None:
        json.dumps(to_otlp([self._span()]))


# --------------------------------------------------------------------------
# Context compression
# --------------------------------------------------------------------------


def _history(n: int = 20) -> list[Message]:
    msgs = [Message(role="system", content="SYSTEM")]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(
            Message(role=role, content=f"turn {i} SELECT * FROM orders WHERE city = 'Shanghai'")
        )
    return msgs


class TestCompression:
    def test_short_history_is_untouched(self) -> None:
        msgs = [Message(role="user", content="hi")]
        out, stats = compress(msgs, budget=8)
        assert out == msgs
        assert stats.dropped == 0

    def test_respects_the_budget(self) -> None:
        out, stats = compress(_history(), budget=6)
        assert len(out) <= 6
        assert stats.after <= 6

    def test_keeps_the_system_prompt(self) -> None:
        out, _ = compress(_history(), budget=4)
        assert out[0].role == "system"
        assert out[0].content == "SYSTEM"

    def test_keeps_the_most_recent_turn(self) -> None:
        out, _ = compress(_history(), budget=4)
        assert "turn 19" in out[-1].content

    def test_produces_a_summary_of_dropped_turns(self) -> None:
        out, stats = compress(_history(), budget=4)
        assert any("compressed history" in m.content for m in out)
        assert stats.summarised > 0

    def test_summary_retains_entity_values(self) -> None:
        _, stats = compress(_history(), budget=4)
        assert "Shanghai" in stats.kept_values

    def test_compression_ratio_is_reported(self) -> None:
        _, stats = compress(_history(), budget=6)
        assert 0 < stats.ratio < 1

    def test_ratio_is_one_when_nothing_dropped(self) -> None:
        _, stats = compress(_history(2), budget=10)
        assert stats.ratio == 1.0

    def test_is_deterministic(self) -> None:
        a, _ = compress(_history(), budget=6)
        b, _ = compress(_history(), budget=6)
        assert [m.content for m in a] == [m.content for m in b]

    def test_extract_finds_tool_names(self) -> None:
        msgs = [Message(role="assistant", content="ran run_sql successfully")]
        assert "run_sql" in extract_facts(msgs)

    def test_empty_history_is_safe(self) -> None:
        out, stats = compress([], budget=4)
        assert out == []
        assert stats.before == 0

    def test_stats_payload_is_serialisable(self) -> None:
        _, stats = compress(_history(), budget=4)
        json.dumps(stats.to_payload())


# --------------------------------------------------------------------------
# BFCL adapter
# --------------------------------------------------------------------------


class TestFunctionCallingEngine:
    def _spec(self, name: str = "get_weather", desc: str = "Get the weather") -> FunctionSpec:
        return FunctionSpec(
            name=name,
            description=desc,
            parameters={
                "type": "dict",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

    def test_selects_the_matching_function(self) -> None:
        spec = self._spec()
        call = decide("please get the weather for Paris", [spec])
        assert call.names() == ["get_weather"]

    def test_declines_when_nothing_matches(self) -> None:
        call = decide("what is the capital of France", [self._spec()], relevance_threshold=0.6)
        assert call.calls == []

    def test_extracts_a_numeric_argument(self) -> None:
        spec = FunctionSpec(
            name="add",
            description="Add two numbers",
            parameters={"type": "dict", "properties": {"a": {}, "b": {}}, "required": ["a", "b"]},
        )
        call = decide("add 3 and 4", [spec])
        assert call.calls[0]["arguments"]

    def test_parses_the_benchmark_function_format(self) -> None:
        raw = json.dumps(
            [
                {
                    "name": "f",
                    "description": "d",
                    "parameters": {"type": "dict", "properties": {}, "required": []},
                }
            ]
        )
        assert FunctionSpec.from_raw(raw).name == "f"

    def test_namespaced_names_are_handled(self) -> None:
        spec = FunctionSpec.from_raw([{"name": "math.factorial", "description": "factorial"}])
        assert spec.bare_name == "factorial"


class TestBFCLEvaluator:
    def test_exact_match_scores(self) -> None:
        from insight_agent.eval.bfcl import score_case

        case = _bfcl_case(ground_truth=[{"get_weather": {"city": ["Paris"]}}])
        prediction = FunctionCall(calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}])
        assert score_case(case, prediction)

    def test_wrong_function_scores_zero(self) -> None:
        from insight_agent.eval.bfcl import score_case

        case = _bfcl_case(ground_truth=[{"get_weather": {"city": ["Paris"]}}])
        prediction = FunctionCall(calls=[{"name": "get_time", "arguments": {"city": "Paris"}}])
        assert not score_case(case, prediction)

    def test_extra_argument_scores_zero(self) -> None:
        from insight_agent.eval.bfcl import score_case

        case = _bfcl_case(ground_truth=[{"get_weather": {"city": ["Paris"]}}])
        prediction = FunctionCall(
            calls=[{"name": "get_weather", "arguments": {"city": "Paris", "extra": 1}}]
        )
        assert not score_case(case, prediction)

    def test_optional_argument_may_be_omitted(self) -> None:
        """BFCL marks an optional argument with an empty-string acceptable value."""

        from insight_agent.eval.bfcl import score_case

        case = _bfcl_case(ground_truth=[{"f": {"city": ["Paris"], "unit": ["", "c"]}}])
        prediction = FunctionCall(calls=[{"name": "f", "arguments": {"city": "Paris"}}])
        assert score_case(case, prediction)

    def test_required_argument_may_not_be_omitted(self) -> None:
        from insight_agent.eval.bfcl import score_case

        case = _bfcl_case(ground_truth=[{"f": {"city": ["Paris"]}}])
        prediction = FunctionCall(calls=[{"name": "f", "arguments": {}}])
        assert not score_case(case, prediction)

    def test_irrelevance_requires_calling_nothing(self) -> None:
        from insight_agent.eval.bfcl import score_case

        case = _bfcl_case(ground_truth=None)
        assert score_case(case, FunctionCall(calls=[]))
        assert not score_case(case, FunctionCall(calls=[{"name": "f", "arguments": {}}]))

    def test_call_order_is_not_significant(self) -> None:
        from insight_agent.eval.bfcl import score_case

        case = _bfcl_case(ground_truth=[{"a": {}}, {"b": {}}])
        prediction = FunctionCall(
            calls=[{"name": "b", "arguments": {}}, {"name": "a", "arguments": {}}]
        )
        assert score_case(case, prediction)

    def test_split_truth_unpacks_the_nested_format(self) -> None:
        from insight_agent.eval.bfcl import split_truth

        name, args = split_truth({"get_weather": {"city": ["Paris"]}})
        assert name == "get_weather"
        assert args == {"city": ["Paris"]}

    def test_numeric_ground_truth_compares_by_value(self) -> None:
        from insight_agent.eval.bfcl import score_case

        case = _bfcl_case(ground_truth=[{"f": {"n": [10]}}])
        assert score_case(case, FunctionCall(calls=[{"name": "f", "arguments": {"n": 10.0}}]))


class TestBFCLDataset:
    def test_cached_categories_load(self) -> None:
        from insight_agent.eval.bfcl import load_category

        cases = load_category("simple", limit=5)
        assert len(cases) == 5
        assert all(c.request for c in cases)
        assert all(c.functions for c in cases)

    def test_ground_truth_is_attached_for_scored_categories(self) -> None:
        from insight_agent.eval.bfcl import load_category

        cases = load_category("simple", limit=10)
        assert all(c.ground_truth for c in cases)

    def test_irrelevance_has_no_ground_truth(self) -> None:
        from insight_agent.eval.bfcl import load_category

        cases = load_category("irrelevance", limit=5)
        assert all(c.expect_no_call for c in cases)

    def test_evaluation_reports_a_real_number(self) -> None:
        from insight_agent.eval.bfcl import evaluate

        report = evaluate(limit=20)
        assert report.total > 0
        assert 0.0 <= report.accuracy <= 1.0


def _bfcl_case(ground_truth):
    from insight_agent.eval.bfcl import BFCLCase

    return BFCLCase(
        case_id="t",
        request="q",
        functions=[FunctionSpec(name="f", description="d")],
        ground_truth=ground_truth,
        category="test",
    )


# --------------------------------------------------------------------------
# Compression experiment
# --------------------------------------------------------------------------


class TestCompressionExperiment:
    def test_reports_every_budget(self) -> None:
        from insight_agent.eval.compression_eval import run_compression_experiment

        payload = asyncio_run(run_compression_experiment(budgets=(0, 8), limit=5))
        assert len(payload["results"]) == 2
        assert payload["baseline_budget"] == 0

    def test_report_states_its_own_limitation(self) -> None:
        from insight_agent.eval.compression_eval import run_compression_experiment

        payload = asyncio_run(run_compression_experiment(budgets=(0,), limit=3))
        assert "not evidence that compression is free" in payload["finding"]

    def test_compression_actually_shrinks_the_list(self) -> None:
        from insight_agent.eval.compression_eval import run_compression_experiment

        payload = asyncio_run(run_compression_experiment(budgets=(4,), limit=3))
        assert payload["results"][0]["mean_compression_ratio"] < 1.0


def asyncio_run(coro):  # type: ignore[no-untyped-def]
    import asyncio

    return asyncio.run(coro)
