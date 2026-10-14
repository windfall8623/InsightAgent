"""Tests for ModelRouter dynamic dispatching, multi-turn conversation, and context compression."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from insight_agent.app.memory import MemoryStore
from insight_agent.app.orchestrator import (
    InsightOrchestrator,
    InsightRequest,
    RunConfig,
)
from insight_agent.app.planner import LLMPlanner
from insight_agent.app.router import ModelRouter
from insight_agent.core.llm import LLMClient, Message, Turn


def test_model_router_configuration_and_dynamic_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Defaults
    router = ModelRouter()
    assert router.small == "sim-small"
    assert router.large == "sim-large"
    assert router.threshold == 2

    # 2. Dynamic registration
    router.register_models(small="qwen-2.5-7b", large="qwen-2.5-72b")
    assert router.small == "qwen-2.5-7b"
    assert router.large == "qwen-2.5-72b"

    # 3. Environment variable configuration
    monkeypatch.setenv("INSIGHT_MODEL_SMALL", "deepseek-r1-distill-8b")
    monkeypatch.setenv("INSIGHT_MODEL_LARGE", "deepseek-r1-671b")
    monkeypatch.setenv("INSIGHT_ROUTER_THRESHOLD", "3")

    env_router = ModelRouter()
    assert env_router.small == "deepseek-r1-distill-8b"
    assert env_router.large == "deepseek-r1-671b"
    assert env_router.threshold == 3


def test_model_router_scoring_and_escalation() -> None:
    router = ModelRouter(small="small-model", large="large-model", threshold=2)

    # Simple metric question -> 0 score -> simple tier
    simple_dec = router.route("总销售额是多少")
    assert simple_dec.tier == "simple"
    assert simple_dec.model == "small-model"
    assert not simple_dec.escalated
    assert simple_dec.score == 0

    # Complex question with join/grouping + time window -> score >= 2 -> complex tier
    complex_dec = router.route("按各省份对比2024年的退款率和订单量排行")
    assert complex_dec.tier == "complex"
    assert complex_dec.model == "large-model"
    assert complex_dec.escalated
    assert complex_dec.score >= 2
    assert "time window" in complex_dec.reason or "join/grouping" in complex_dec.reason


def test_memory_store_multi_turn_history_and_state(store_dir: Any) -> None:
    store = MemoryStore(store_dir / "test_memory.json")
    session_id = "test_sess_001"

    # Initial history is empty
    assert store.get_history_messages(session_id) == []
    assert store.get_session_state(session_id) == {}

    # Remember turns
    store.remember_turn(session_id, "user", "上海的订单量是多少？")
    store.remember_turn(session_id, "assistant", "上海的订单量共计 1,230 单。")

    msgs = store.get_history_messages(session_id)
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[0].content == "上海的订单量是多少？"
    assert msgs[1].role == "assistant"
    assert "1,230" in msgs[1].content

    # Set and retrieve session state
    store.set_session_state(session_id, "standing_filters", {"city": "上海市"})
    assert store.get_session_state(session_id)["standing_filters"] == {"city": "上海市"}

    # Clear session
    store.clear_session(session_id)
    assert store.get_history_messages(session_id) == []
    assert store.get_session_state(session_id) == {}


@pytest.mark.asyncio
async def test_llm_planner_multi_model_dispatch_and_context_injection() -> None:
    mock_small = AsyncMock(spec=LLMClient)
    mock_small.complete.return_value = Turn(text="SELECT count(*) FROM orders;", stop_reason="stop")
    mock_small.model = "mock-small-v1"

    mock_large = AsyncMock(spec=LLMClient)
    mock_large.complete.return_value = Turn(
        text="SELECT city, count(*) FROM orders GROUP BY city;", stop_reason="stop"
    )
    mock_large.model = "mock-large-v1"

    planner = LLMPlanner(
        mock_small,
        clients={"sim-small": mock_small, "sim-large": mock_large},
    )

    history = [
        Message(role="user", content="前置提问：有哪些城市？"),
        Message(role="assistant", content="包含北京、上海、广州等。"),
    ]

    # Dispatch to sim-large
    res = await planner.plan(
        "按城市分组统计订单量",
        schema_hint="orders(city, order_id)",
        compressed_history=history,
        model="sim-large",
    )

    assert res.model == "sim-large"
    assert "GROUP BY city" in res.sql
    assert mock_large.complete.called
    call_messages = mock_large.complete.call_args[0][0]
    # Check that system prompt, history, and user query were assembled in order
    assert call_messages[0].role == "system"
    assert call_messages[1].role == "user"
    assert "前置提问" in call_messages[1].content
    assert call_messages[2].role == "assistant"
    assert call_messages[3].role == "user"
    assert "按城市分组统计订单量" in call_messages[3].content


@pytest.mark.asyncio
async def test_orchestrator_multi_turn_filter_inheritance(conn: Any, store_dir: Any) -> None:
    store = MemoryStore(store_dir / "test_mem_filters.json")
    orchestrator = InsightOrchestrator(conn, memory=store)
    session_id = "multi_turn_session_filter"

    # Turn 1: Explicit geographic query
    req1 = InsightRequest(
        question="统计上海市已完成订单总数",
        user_id="user_test",
        session_id=session_id,
    )
    ans1 = await orchestrator.run(req1)
    assert not ans1.blocked
    assert "上海" in ans1.sql or "users" in ans1.sql
    assert ans1.row_count >= 1

    # Verify session state captured standing filter
    state = store.get_session_state(session_id)
    assert "standing_filters" in state
    assert state["standing_filters"].get("city") in ("上海市", "Shanghai") or "Shanghai" in str(state["standing_filters"]) or "上海" in str(state["standing_filters"])

    # Turn 2: Follow-up question without explicit city name
    req2 = InsightRequest(
        question="那其中退款率是多少？",
        user_id="user_test",
        session_id=session_id,
    )
    ans2 = await orchestrator.run(req2)
    assert not ans2.blocked
    # Inherited standing filter applies to the follow-up SQL
    assert "Shanghai" in ans2.sql or "city" in ans2.sql or "users" in ans2.sql or "上海" in ans2.sql


@pytest.mark.asyncio
async def test_orchestrator_multi_turn_context_compression(conn: Any, store_dir: Any) -> None:
    store = MemoryStore(store_dir / "test_mem_compress.json")
    # Set context_budget to 2 so compression triggers quickly after 2 turns
    config = RunConfig(use_memory=True, context_budget=2)
    orchestrator = InsightOrchestrator(conn, memory=store, config=config)
    session_id = "multi_turn_compress_sess"

    # Turn 1
    ans1 = await orchestrator.run(
        InsightRequest(question="总销售额是多少", user_id="u1", session_id=session_id)
    )
    assert not ans1.context_compressed

    # Turn 2
    ans2 = await orchestrator.run(
        InsightRequest(question="各城市的销售额", user_id="u1", session_id=session_id)
    )
    # Turn 2 has 2 prior messages, budget is 2 -> not yet exceeding
    assert not ans2.context_compressed

    # Turn 3: 4 prior messages now exceed budget of 2 -> compression must trigger!
    ans3 = await orchestrator.run(
        InsightRequest(question="哪个城市销售额最高", user_id="u1", session_id=session_id)
    )
    assert ans3.context_compressed is True
    assert "Compressed" in ans3.context_summary or ans3.context_summary != ""
    assert ans3.row_count >= 1


@pytest.mark.asyncio
async def test_orchestrator_dynamic_model_routing_metadata(conn: Any) -> None:
    orchestrator = InsightOrchestrator(conn)
    orchestrator.router.register_models(small="fast-small-model", large="heavy-large-model")

    # Simple query
    simple_ans = await orchestrator.run(
        InsightRequest(question="总订单量是多少", user_id="u_route")
    )
    assert simple_ans.model == "fast-small-model"
    assert simple_ans.route_score is not None

    # Complex query
    complex_ans = await orchestrator.run(
        InsightRequest(
            question="按各省份和城市分别对比最近一年的退款率和订单增长趋势，找出top3城市",
            user_id="u_route",
        )
    )
    assert complex_ans.model == "heavy-large-model"
    assert complex_ans.route_score >= orchestrator.router.threshold


def test_graph_engine_model_router_integration(conn: Any) -> None:
    from insight_agent.app.graph import InsightGraphEngine
    from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
    from insight_agent.app.redis_store import RedisMemoryStore

    retriever = OlistMultiModalRetriever(conn)
    redis_store = RedisMemoryStore()
    custom_router = ModelRouter(small="graph-small", large="graph-large", threshold=2)

    engine = InsightGraphEngine(conn, retriever, redis_store, router=custom_router)
    assert engine.router.small == "graph-small"
    assert engine.router.large == "graph-large"

    # Test intent router node with simple query
    state_simple = {
        "thread_id": "t1",
        "user_id": "u1",
        "query": "各状态订单总数是多少",
    }
    out_simple = engine.intent_router_node(state_simple)  # type: ignore[arg-type]
    assert out_simple["intent"] == "adhoc_query"
    assert out_simple["route_tier"] == "simple"
    assert out_simple["route_model"] == "graph-small"

    # Test intent router node with complex diagnostic query
    state_complex = {
        "thread_id": "t2",
        "user_id": "u1",
        "query": "为什么最近华东地区的生鲜食品退款率激增出现严重延误和差评客诉？",
    }
    out_complex = engine.intent_router_node(state_complex)  # type: ignore[arg-type]
    assert out_complex["intent"] == "diagnostic_anomaly"
    assert out_complex["route_tier"] == "complex"
    assert out_complex["route_model"] == "graph-large"


@pytest.mark.asyncio
async def test_cn_retail_multi_turn_filter_inheritance(store_dir: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from insight_agent.data.cn_retail_db import connect_cn_retail, ensure_cn_retail_database

    db_path = ensure_cn_retail_database()
    cn_conn = connect_cn_retail(db_path)
    try:
        monkeypatch.setenv("INSIGHT_DATASET", "cn_retail")
        store = MemoryStore(store_dir / "test_cn_mem.json")
        orchestrator = InsightOrchestrator(cn_conn, memory=store)
        session_id = "cn_sess_multi_001"

        # Turn 1: Explicit province and category
        req1 = InsightRequest(
            question="统计广东省美妆个护品类的退款率",
            user_id="cn_user",
            session_id=session_id,
        )
        ans1 = await orchestrator.run(req1)
        assert not ans1.blocked
        assert "广东省" in ans1.sql
        assert "美妆个护" in ans1.sql
        assert ans1.row_count >= 1

        # Turn 2: Follow-up question without explicit province or category
        req2 = InsightRequest(
            question="那销售额是多少？",
            user_id="cn_user",
            session_id=session_id,
        )
        ans2 = await orchestrator.run(req2)
        assert not ans2.blocked
        # Inherited province and category from Turn 1!
        assert "广东省" in ans2.sql
        assert "美妆个护" in ans2.sql
    finally:
        cn_conn.close()
