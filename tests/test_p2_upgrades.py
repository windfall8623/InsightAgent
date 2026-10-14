"""Tests for P2 Phase Upgrades:

1. OpenTelemetry OTLP HTTP Exporter & Tracer Integration
2. Local Infrastructure Docker Compose Stack & Configs
3. Qdrant & BGE Fallback and Remote Healthcheck Hooks
4. Architecture Governance: Standalone ReAct Agent Loop & BFCL Funccall Engine
"""

from __future__ import annotations

import http.server
import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from insight_agent.app.funccall import FunctionSpec, decide
from insight_agent.app.obs import OTLPHttpExporter, Span, Tracer, to_otlp
from insight_agent.app.qdrant_store import (
    QdrantVectorStore,
    compute_dense_embedding,
    compute_hash_dense_embedding,
)
from insight_agent.core.agent import Agent, AgentConfig
from insight_agent.core.llm import LLMClient, Message, ToolCall, Turn

# ==============================================================================
# 1. OpenTelemetry OTLP HTTP Exporter Tests
# ==============================================================================


class _MockOTLPCollector(http.server.BaseHTTPRequestHandler):
    received_payloads: list[dict[str, Any]] = []
    received_headers: list[dict[str, str]] = []
    response_code = 200

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        _MockOTLPCollector.received_payloads.append(json.loads(body.decode("utf-8")))
        _MockOTLPCollector.received_headers.append(dict(self.headers))
        self.send_response(_MockOTLPCollector.response_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress stdout logging during tests
        pass


@pytest.fixture
def mock_collector():
    _MockOTLPCollector.received_payloads.clear()
    _MockOTLPCollector.received_headers.clear()
    _MockOTLPCollector.response_code = 200

    server = http.server.HTTPServer(("127.0.0.1", 0), _MockOTLPCollector)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{port}/v1/traces"
    yield endpoint, _MockOTLPCollector
    server.shutdown()
    server.server_close()


class TestOpenTelemetryExport:
    def test_to_otlp_json_structure(self) -> None:
        spans = [
            Span(
                name="stage_test",
                trace_id="trace_abc123",
                attributes={"user_id": "u1", "row_count": 42},
            )
        ]
        payload = to_otlp(spans, service_name="test-service")

        assert "resourceSpans" in payload
        res_spans = payload["resourceSpans"]
        assert len(res_spans) == 1
        attrs = res_spans[0]["resource"]["attributes"]
        assert any(a["key"] == "service.name" and a["value"]["stringValue"] == "test-service" for a in attrs)

        scope_spans = res_spans[0]["scopeSpans"][0]["spans"]
        assert len(scope_spans) == 1
        span_data = scope_spans[0]
        assert span_data["name"] == "stage_test"
        assert len(span_data["traceId"]) == 32
        assert len(span_data["spanId"]) == 16

    def test_otlp_http_exporter_successful_post(self, mock_collector) -> None:
        endpoint, collector = mock_collector
        exporter = OTLPHttpExporter(
            endpoint=endpoint,
            headers={"X-Test-Header": "insight-trace"},
            service_name="insight-agent-test",
        )

        spans = [
            Span(name="sql_exec", trace_id="tr_1", attributes={"rows": 10}),
            Span(name="insight_analyst", trace_id="tr_1", parent="sql_exec"),
        ]

        ok = exporter.export(spans)
        assert ok is True
        assert exporter.last_export_status == 200
        assert exporter.last_error is None

        assert len(collector.received_payloads) == 1
        received = collector.received_payloads[0]
        assert received["resourceSpans"][0]["resource"]["attributes"][0]["value"]["stringValue"] == "insight-agent-test"
        assert len(received["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 2
        headers_lower = {k.lower(): v for k, v in collector.received_headers[0].items()}
        assert headers_lower.get("x-test-header") == "insight-trace"

    def test_otlp_http_exporter_graceful_error_handling(self) -> None:
        # Invalid / unreachable port
        exporter = OTLPHttpExporter(
            endpoint="http://127.0.0.1:59998/v1/traces",
            timeout=0.5,
        )
        spans = [Span(name="test", trace_id="tr_err")]
        ok = exporter.export(spans)
        assert ok is False
        assert exporter.last_export_status is None
        assert exporter.last_error is not None
        assert "URLError" in exporter.last_error or "ConnectionRefused" in exporter.last_error

    def test_tracer_flushes_otlp(self, mock_collector) -> None:
        endpoint, collector = mock_collector
        exporter = OTLPHttpExporter(endpoint=endpoint)
        tracer = Tracer(exporter=exporter)

        with tracer.span("orchestrator_run", trace_id="tr_live"):
            pass

        assert len(tracer.spans) == 1
        flushed = tracer.flush_otlp()
        assert flushed is True
        assert len(collector.received_payloads) == 1


# ==============================================================================
# 2. Local Infrastructure Docker Compose Stack & Configs
# ==============================================================================


class TestInfrastructureCompose:
    def test_docker_compose_infra_file_exists_and_valid(self) -> None:
        compose_path = Path("docker-compose.infra.yml")
        assert compose_path.exists(), "docker-compose.infra.yml must exist"
        data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

        assert "services" in data
        services = data["services"]
        assert "insight-redis" in services
        assert "insight-qdrant" in services
        assert "insight-jaeger" in services

        # Check port mappings
        assert "6379:6379" in services["insight-redis"]["ports"]
        assert "6333:6333" in services["insight-qdrant"]["ports"]
        assert "4318:4318" in services["insight-jaeger"]["ports"]
        assert "16686:16686" in services["insight-jaeger"]["ports"]

    def test_env_infra_example_exists_and_configured(self) -> None:
        env_path = Path(".env.infra.example")
        assert env_path.exists(), ".env.infra.example must exist"
        content = env_path.read_text(encoding="utf-8")
        assert "REDIS_URL=" in content
        assert "QDRANT_URL=" in content
        assert "OTEL_EXPORTER_OTLP_ENDPOINT=" in content

    def test_main_docker_compose_includes_jaeger(self) -> None:
        compose_path = Path("docker-compose.yml")
        assert compose_path.exists()
        data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

        assert "insight-agent-jaeger" in data["services"]
        app_env = data["services"]["insight-agent-app"]["environment"]
        assert any("OTEL_EXPORTER_OTLP_ENDPOINT" in e for e in app_env)


# ==============================================================================
# 3. Qdrant & BGE Vector Store & Embeddings
# ==============================================================================


class TestVectorStoreAndEmbeddings:
    def test_qdrant_fallback_to_memory_when_remote_unavailable(self) -> None:
        store = QdrantVectorStore(
            collection_name=f"test_coll_{os.getpid()}",
            dims=64,
            url="http://127.0.0.1:59999",
            force_memory=False,
        )
        assert store.is_memory is True

    def test_compute_dense_embedding_shape_and_normalization(self) -> None:
        vec = compute_dense_embedding("物流延误在分拨中心滞留", dims=128)
        assert len(vec) == 128
        norm = sum(v * v for v in vec) ** 0.5
        assert pytest.approx(norm, abs=1e-3) == 1.0

    def test_compute_hash_dense_embedding_deterministic(self) -> None:
        v1 = compute_hash_dense_embedding("生鲜保温泡沫箱破损", dims=64)
        v2 = compute_hash_dense_embedding("生鲜保温泡沫箱破损", dims=64)
        assert v1 == v2
        assert len(v1) == 64


# ==============================================================================
# 4. Architecture Governance: Standalone ReAct Agent Loop & BFCL
# ==============================================================================


class _MockLLM(LLMClient):
    def __init__(self, responses: list[Turn]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Turn:
        self.calls.append(list(messages))
        return self._responses.pop(0)


class TestStandaloneReActAgent:
    @pytest.mark.asyncio
    async def test_agent_react_tool_execution_loop(self) -> None:
        from insight_agent.core.tools import tool

        @tool(name="add_numbers", description="Add two integers")
        def add_numbers(a: int, b: int) -> str:
            return str(a + b)

        responses = [
            Turn(
                text="Calling add tool",
                tool_calls=[ToolCall(id="c1", name="add_numbers", arguments={"a": 10, "b": 25})],
            ),
            Turn(text="The result is 35", tool_calls=[]),
        ]
        client = _MockLLM(responses)
        agent = Agent(client=client, tools=[add_numbers], config=AgentConfig(max_turns=3))

        result = await agent.run("Please calculate 10 + 25")
        assert result.text == "The result is 35"
        assert result.turns == 2
        assert result.tool_calls == 1

    def test_agent_run_sync_convenience_wrapper(self) -> None:
        responses = [
            Turn(text="Hello from sync agent", tool_calls=[]),
        ]
        client = _MockLLM(responses)
        agent = Agent(client=client, tools=[], config=AgentConfig(max_turns=2))

        result = agent.run_sync("Hi")
        assert result.text == "Hello from sync agent"
        assert result.turns == 1


class TestBFCLFunctionCallEngine:
    def test_bfcl_decide_and_param_extraction(self) -> None:
        specs = [
            FunctionSpec(
                name="weather.get_temperature",
                description="Get current temperature forecast for given days",
                parameters={
                    "properties": {"days": {"type": "integer"}},
                    "required": ["days"],
                },
            ),
            FunctionSpec(
                name="stock.get_quote",
                description="Get ticker quote for a public company",
                parameters={
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                },
            ),
        ]

        call = decide("Get temperature forecast for 3 days", specs)
        assert len(call.calls) > 0
        assert call.names()[0] == "weather.get_temperature"
        args = call.calls[0]["arguments"]
        assert args.get("days") == 3

    def test_bfcl_decide_returns_none_when_irrelevant(self) -> None:
        specs = [
            FunctionSpec(
                name="crypto.mint_nft",
                description="Mint a non-fungible token on blockchain",
                parameters={
                    "properties": {"token_id": {"type": "string"}},
                    "required": ["token_id"],
                },
            )
        ]
        call = decide("What is the refund rate of sellers last month?", specs)
        assert len(call.calls) == 0
        assert call.names() == []
