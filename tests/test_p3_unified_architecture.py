"""Tests for P3 Phase: Unified LangGraph Architecture & /chat Entrypoint.

Validates:
1. Subsuming InsightOrchestrator into LangGraph adhoc_query_node (Fast-path Semantic Subgraph).
2. Compiled AST SQL, tabular row outputs, and Redis query caching via LangGraph.
3. Unified @app.post("/chat") entrypoint handling:
   - Adhoc NL2SQL queries
   - Schema explanations
   - Root-cause anomaly investigations with Secure-by-Default HITL approval.
4. Thread resumption via /investigation/{thread_id}/resume from /chat threads.
5. Backward compatibility for legacy /ask and /diagnose endpoints.
"""

from __future__ import annotations

import pytest
from insight_agent.api import create_app
from insight_agent.app.graph import InsightGraphEngine
from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
from insight_agent.app.orchestrator import InsightOrchestrator
from insight_agent.app.redis_store import RedisMemoryStore


@pytest.fixture(autouse=True)
def fast_test_env(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)


@pytest.fixture()
def client(conn, monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("INSIGHT_API_TOKEN", "test-token")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    orch = InsightOrchestrator(conn)
    app = create_app(orch)
    return fastapi_testclient.TestClient(app)


# ==============================================================================
# 1. Direct Engine Tests: adhoc_query_node & chat()
# ==============================================================================


def test_engine_adhoc_query_node_delegates_to_orchestrator(conn) -> None:
    """Test that adhoc_query_node runs semantic NL2SQL via orchestrator and returns real SQL & rows."""
    retriever = OlistMultiModalRetriever(conn)
    redis_store = RedisMemoryStore()
    orch = InsightOrchestrator(conn)
    engine = InsightGraphEngine(conn, retriever, redis_store, orchestrator=orch)

    # Clean query cache
    redis_store.delete_query_cache("总销售额是多少")

    # Diagnose an adhoc query
    state = engine.chat("总销售额是多少")
    assert state.get("intent") == "adhoc_query"
    assert "Query executed successfully" in state.get("final_report", "")
    assert state.get("sql") is not None
    assert "SELECT" in state.get("sql", "").upper()
    assert state.get("row_count", 0) >= 1
    assert len(state.get("rows", [])) >= 1

    # Second invocation should hit Redis query cache
    cached_state = engine.chat("总销售额是多少")
    assert "【缓存命中】" in cached_state.get("final_report", "")
    assert cached_state.get("sql") is not None


def test_engine_chat_schema_explain(conn) -> None:
    """Test that schema exploration queries are correctly routed to schema_explain_node."""
    retriever = OlistMultiModalRetriever(conn)
    redis_store = RedisMemoryStore()
    orch = InsightOrchestrator(conn)
    engine = InsightGraphEngine(conn, retriever, redis_store, orchestrator=orch)

    state = engine.chat("解释一下订单表的字段和表结构")
    assert state.get("intent") == "schema_explain"
    assert "olist_orders" in state.get("final_report", "").lower() or "orders" in state.get("final_report", "").lower()


# ==============================================================================
# 2. Unified HTTP Gateway: POST /chat Tests
# ==============================================================================


def test_chat_adhoc_query_flow(client) -> None:
    """Test POST /chat executes natural language adhoc queries through LangGraph statechart."""
    response = client.post(
        "/chat",
        json={"message": "总销售额是多少", "user_id": "test_user_p3"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["intent"] == "adhoc_query"
    assert data["sql"] is not None
    assert "SELECT" in data["sql"].upper()
    assert data["row_count"] >= 1
    assert len(data["rows"]) == data["row_count"]
    assert "Query executed successfully" in data["answer"]
    assert data["thread_id"] is not None


def test_chat_schema_explain_flow(client) -> None:
    """Test POST /chat routing to schema explain."""
    response = client.post(
        "/chat",
        json={"message": "查看数据表结构与字段含义", "user_id": "test_user_p3"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["intent"] == "schema_explain"
    assert "schema" in data["answer"].lower() or "表" in data["answer"]


def test_chat_diagnostic_anomaly_secure_by_default_hitl(client) -> None:
    """Test POST /chat with diagnostic anomaly stops for HITL approval by default."""
    query = "为什么最近华东地区中通快递延迟率激增？"
    response = client.post(
        "/chat",
        json={"message": query, "user_id": "operator_p3"},
    )
    assert response.status_code == 200
    data = response.json()

    # Must be pending HITL approval by default (Secure-by-Default guarantee)
    assert data["status"] == "pending_hitl_approval"
    assert data["intent"] == "diagnostic_anomaly"
    assert data["hitl_approved"] is False
    assert len(data["candidate_actions"]) > 0
    assert data["resume_url"] is not None

    tid = data["thread_id"]
    assert f"/investigation/{tid}/resume" == data["resume_url"]

    # Resume the investigation with HITL approval and custom parameter override
    resume_resp = client.post(
        data["resume_url"],
        json={
            "query": query,
            "override_arguments": {"voucher_amount": 35.0, "supplier_id": "ZTO_EXP_002"},
        },
    )
    assert resume_resp.status_code == 200
    resumed_data = resume_resp.json()
    assert resumed_data["status"] == "resumed"
    assert resumed_data["action_result"] is not None
    assert isinstance(resumed_data["action_result"], str)
    assert len(resumed_data["action_result"]) > 0


def test_chat_diagnostic_anomaly_preapproved_hitl(client) -> None:
    """Test POST /chat executes remediation when hitl_approved is explicitly set to True."""
    response = client.post(
        "/chat",
        json={
            "message": "为什么最近华东地区中通快递延迟率激增？",
            "user_id": "authorized_admin",
            "hitl_approved": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["intent"] == "diagnostic_anomaly"
    assert data["hitl_approved"] is True
    assert data["action_result"] is not None


# ==============================================================================
# 3. Backward Compatibility Tests
# ==============================================================================


def test_legacy_endpoints_backward_compatibility(client) -> None:
    """Ensure existing /ask and /diagnose endpoints continue to work without breaking changes."""
    # 1. Legacy /ask
    ask_resp = client.post("/ask", json={"question": "总销售额是多少", "user_id": "u1"})
    assert ask_resp.status_code == 200
    assert ask_resp.json()["row_count"] >= 1

    # 2. Legacy /diagnose default hitl_approved=False
    diag_resp = client.post(
        "/diagnose",
        json={"query": "为什么最近华东地区中通快递延迟率激增？", "user_id": "u2"},
    )
    assert diag_resp.status_code == 200
    diag_data = diag_resp.json()
    assert diag_data["status"] == "pending_hitl_approval"
    assert diag_data["hitl_approved"] is False
    assert diag_data["resume_url"] is not None
