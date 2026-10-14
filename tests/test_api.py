"""End-to-end tests for the HTTP surface."""

from __future__ import annotations

import pytest

from insight_agent.api import create_app
from insight_agent.app.orchestrator import InsightOrchestrator


@pytest.fixture()
def client(conn, monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    # The memory routes require a token once one is configured; tests exercise
    # the guarded path, so the fixture turns it on explicitly.
    monkeypatch.setenv("INSIGHT_API_TOKEN", "test-token")
    app = create_app(InsightOrchestrator(conn))
    return fastapi_testclient.TestClient(app)


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_returns_rows_and_sql(client) -> None:
    response = client.post("/ask", json={"question": "总销售额是多少", "user_id": "u1"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["row_count"] == 1
    assert "SELECT" in payload["sql"]
    assert payload["agents_used"][0] == "schema-explorer"


def test_ask_anomaly_routing_returns_real_sql_and_evidence(client) -> None:
    response = client.post(
        "/ask",
        json={"question": "为什么最近华东地区中通快递延迟率激增？", "user_id": "operator_test"},
    )
    assert response.status_code == 200
    payload = response.json()

    # 1. Strategy & Routing
    assert payload["strategy"] == "langgraph_state_machine"
    assert payload["route_reason"] == "diagnostic_anomaly"
    assert payload["blocked"] is False

    # 2. Genuine Executed SQL (P0 Gap check: no placeholder "...")
    sql = payload["sql"]
    assert "..." not in sql
    assert sql.strip().startswith("SELECT")
    assert "cn_logistics" in sql
    assert "cn_orders" in sql

    # 3. Dynamic Columns & Real Executed Rows
    assert payload["row_count"] >= 1
    assert len(payload["rows"]) == payload["row_count"]
    first_row = payload["rows"][0]
    assert "carrier" in first_row or "total_orders" in first_row
    assert any("carrier" in col for col in payload["columns"])

    # 4. Multi-Modal Evidence: Tables and Review Snippets
    evidence = payload["evidence"]
    assert len(evidence) >= 2
    table_evs = [e for e in evidence if e.get("kind") == "table"]
    review_evs = [e for e in evidence if e.get("kind") == "review"]
    assert len(table_evs) >= 1
    assert any(e["id"] in ("cn_logistics", "cn_orders") for e in table_evs)
    assert len(review_evs) >= 1
    assert "snippet" in review_evs[0]

    # 5. Tools & Agents
    assert "query_cn_retail_sql" in payload["tools_used"]
    assert "search_customer_reviews" in payload["tools_used"]
    assert "langgraph-investigator" in payload["agents_used"]
    assert "evidence-judge" in payload["agents_used"]
    assert "insight-analyst" in payload["agents_used"]

    # 6. Final Report & Root Cause
    assert "中通" in payload["answer"]
    assert "延误" in payload["answer"] or "超时" in payload["answer"]


def test_explain_returns_the_trail(client) -> None:
    response = client.post("/explain", json={"question": "各城市销售额", "user_id": "u1"})
    payload = response.json()
    assert payload["agents_used"][-1] == "insight-analyst"
    assert payload["evidence"]
    assert payload["model"]


def test_schema_endpoint(client) -> None:
    payload = client.get("/schema").json()
    assert "orders" in payload["tables"]
    assert any(c.startswith("order_date") for c in payload["tables"]["orders"])


def test_agent_card_discovery(client) -> None:
    cards = client.get("/.well-known/agent.json").json()
    assert len(cards) == 4


def test_mcp_over_http(client) -> None:
    tools = client.get("/mcp/tools").json()["tools"]
    assert {t["name"] for t in tools} == {
        "list_tables",
        "get_schema",
        "run_sql",
        "describe_result",
    }

    result = client.post(
        "/mcp/call",
        json={"name": "run_sql", "arguments": {"sql": "SELECT COUNT(*) AS n FROM orders"}},
    ).json()
    assert result["isError"] is False


def test_mcp_over_http_rejects_writes(client) -> None:
    result = client.post(
        "/mcp/call", json={"name": "run_sql", "arguments": {"sql": "DROP TABLE orders"}}
    ).json()
    assert result["isError"] is True
    assert "BLOCKED" in result["content"][0]["text"]


def test_mcp_requires_a_name(client) -> None:
    assert client.post("/mcp/call", json={"arguments": {}}).status_code == 400


def test_memory_endpoints(client) -> None:
    auth = {"Authorization": "Bearer test-token"}
    client.post("/ask", json={"question": "我只看上海的订单有多少", "user_id": "u7"})
    profile = client.get("/memory/u7", headers=auth).json()
    assert profile["filters"] == {"city": "'Shanghai'"}

    client.post("/memory/u7/forget", headers=auth)
    assert client.get("/memory/u7", headers=auth).json()["filters"] == {}


class TestMemoryEndpointAuth:
    """The memory routes expose and erase one user's profile by id in the path."""

    def _client(self, token: str | None):
        import sqlite3

        from fastapi.testclient import TestClient

        from insight_agent.api import create_app
        from insight_agent.app.orchestrator import InsightOrchestrator

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE mock (id INT)")
        app = create_app(InsightOrchestrator(conn))
        if token is None:
            import os
            os.environ.pop("INSIGHT_API_TOKEN", None)
        else:
            import os
            os.environ["INSIGHT_API_TOKEN"] = token
        return TestClient(app)

    def test_refused_when_no_token_configured(self):
        c = self._client(None)
        assert c.get("/memory/u1").status_code == 503
        assert c.post("/memory/u1/forget").status_code == 503

    def test_rejects_missing_and_wrong_token(self):
        c = self._client("s3cret")
        assert c.get("/memory/u1").status_code == 401
        assert c.get("/memory/u1", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert (
            c.post("/memory/u1/forget", headers={"Authorization": "Basic s3cret"}).status_code
            == 401
        )

    def test_accepts_the_configured_token(self):
        c = self._client("s3cret")
        r = c.get("/memory/u1", headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200
        assert "user_id" in r.json()
        assert (
            c.post("/memory/u1/forget", headers={"Authorization": "Bearer s3cret"}).status_code
            == 200
        )

    def test_public_endpoints_stay_open(self):
        c = self._client("s3cret")
        assert c.get("/health").status_code == 200
        assert c.get("/schema").status_code == 200
