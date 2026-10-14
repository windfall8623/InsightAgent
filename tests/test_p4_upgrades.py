"""Tests for P4 Phase Upgrades:

1. Deterministic Coreference Resolution Engine (CoreferenceResolver)
2. RulesPlanner Multi-Turn Pronoun, Ellipsis & Entity Substitution Integration
3. InsightOrchestrator Multi-Turn Conversational Dialogue Flow
4. ONNX Runtime Lightweight Cross-Encoder Reranker (ONNXBGEReranker & BGEReranker)
"""

from __future__ import annotations

from typing import Any

import pytest

from insight_agent.app.bge_reranker import (
    BGEReranker,
    HeuristicCrossAttentionReranker,
    ONNXBGEReranker,
    RerankCandidate,
)
from insight_agent.app.coreference import CoreferenceResolver
from insight_agent.app.orchestrator import InsightOrchestrator, InsightRequest
from insight_agent.app.planner import RulesPlanner
from insight_agent.core.llm import Message


@pytest.fixture(autouse=True)
def fast_test_env(monkeypatch):
    """Disable remote Qdrant and Reranker timeouts in tests."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("INSIGHT_RERANKER_URL", raising=False)
    monkeypatch.delenv("INSIGHT_ONNX_RERANKER_PATH", raising=False)
    monkeypatch.delenv("INSIGHT_BGE_ENABLED", raising=False)


# ==============================================================================
# 1. CoreferenceResolver Unit Tests
# ==============================================================================


def test_coreference_entity_substitution() -> None:
    """Test explicit entity substitution ('把刚才那个物流商换成顺丰再查一遍')."""
    resolver = CoreferenceResolver()
    history = [
        Message(role="user", content="查看中通快递最近一个月的延误率"),
        Message(role="assistant", content="中通快递最近一个月的延误率为 4.2%"),
    ]

    res = resolver.resolve("把刚才那个物流商换成顺丰再查一遍", history=history)
    assert res.is_rewritten is True
    assert res.resolution_type == "entity_replacement"
    assert res.substitute_entity == "顺丰"
    assert res.antecedent_entity == "中通"
    assert res.entity_dimension == "carrier"
    assert "顺丰" in res.resolved_query
    assert "延误率" in res.resolved_query
    assert res.standing_filters.get("carrier") == "顺丰"


def test_coreference_pronoun_resolution() -> None:
    """Test pronoun coreference ('它们上一季度的退款率是多少')."""
    resolver = CoreferenceResolver()
    history = [
        Message(role="user", content="查看数码3C在华东地区的总销售额"),
        Message(role="assistant", content="销售额共计 1,200,000 元"),
    ]

    res = resolver.resolve("它们上一季度的退款率是多少", history=history)
    assert res.is_rewritten is True
    assert res.resolution_type == "pronoun_coreference"
    assert res.antecedent_entity == "数码3C"
    assert "数码3C" in res.resolved_query
    assert "退款率" in res.resolved_query


def test_coreference_ellipsis_continuation() -> None:
    """Test follow-up ellipsis ('那顺丰呢？' / '那华南呢？')."""
    resolver = CoreferenceResolver()
    history = [
        Message(role="user", content="查看中通快递最近一个月的延误率"),
        Message(role="assistant", content="中通快递最近一个月的延误率为 4.2%"),
    ]

    res = resolver.resolve("那顺丰呢？", history=history)
    assert res.is_rewritten is True
    assert res.resolution_type == "ellipsis_continuation"
    assert res.substitute_entity == "顺丰"
    assert "顺丰" in res.resolved_query
    assert "延误率" in res.resolved_query


def test_coreference_no_op_for_standalone_query() -> None:
    """Standalone queries without pronouns or ellipsis must not be mutated."""
    resolver = CoreferenceResolver()
    q = "按状态统计订单量"
    res = resolver.resolve(q)
    assert res.is_rewritten is False
    assert res.resolution_type == "none"
    assert res.resolved_query == q


# ==============================================================================
# 2. RulesPlanner Multi-Turn Coreference Integration
# ==============================================================================


@pytest.mark.asyncio
async def test_rules_planner_probe_with_coreference() -> None:
    """RulesPlanner.probe should return True for ellipsis queries when context is provided."""
    planner = RulesPlanner(dataset="cn_retail")
    history = [
        Message(role="user", content="查看中通快递最近一个月的延误率"),
    ]

    # Without history, '那顺丰呢？' cannot be understood by deterministic parser
    assert planner.probe("那顺丰呢？") is False

    # With history, coreference resolves it into '查看顺丰快递最近一个月的延误率', returning True!
    assert planner.probe("那顺丰呢？", history=history) is True


@pytest.mark.asyncio
async def test_rules_planner_plan_resolves_substitution() -> None:
    """RulesPlanner.plan generates genuine SQL with substituted entity."""
    planner = RulesPlanner(dataset="cn_retail")
    history = [
        Message(role="user", content="查看中通快递最近一个月的延误率"),
        Message(role="assistant", content="中通快递延误率为 4.2%"),
    ]

    result = await planner.plan(
        "把刚才那个物流商换成顺丰再查一遍",
        compressed_history=history,
    )
    assert result.understood is True
    assert "顺丰" in result.sql
    assert "中通" not in result.sql
    assert any("coreference: resolved" in n for n in result.notes)


@pytest.mark.asyncio
async def test_rules_planner_plan_resolves_pronoun() -> None:
    """RulesPlanner.plan generates genuine SQL with antecedent category entity."""
    planner = RulesPlanner(dataset="cn_retail")
    history = [
        Message(role="user", content="查看数码3C在华东地区的销售额"),
        Message(role="assistant", content="销售额为 500,000 元"),
    ]

    result = await planner.plan(
        "它们上一季度的退款率是多少",
        compressed_history=history,
    )
    assert result.understood is True
    assert "数码3C" in result.sql
    assert any("coreference: resolved" in n for n in result.notes)


# ==============================================================================
# 3. InsightOrchestrator Multi-Turn Conversational Dialogue Flow
# ==============================================================================


@pytest.mark.asyncio
async def test_orchestrator_multi_turn_coreference_dialogue() -> None:
    """Test full multi-turn conversational session through InsightOrchestrator."""
    from insight_agent.data.cn_retail_db import connect_cn_retail, ensure_cn_retail_database

    ensure_cn_retail_database()
    cn_conn = connect_cn_retail()
    orch = InsightOrchestrator(cn_conn)
    session_id = "test_coref_session_p4"

    # Turn 1: Initial query establishing context
    req1 = InsightRequest(
        question="查看中通快递最近一个月的延误率",
        user_id="user_p4",
        session_id=session_id,
    )
    ans1 = await orch.run(req1)
    assert ans1.blocked is False
    assert ans1.sql != ""
    assert "中通" in ans1.sql

    # Turn 2: Follow-up entity replacement query in the same session
    req2 = InsightRequest(
        question="把刚才那个物流商换成顺丰再查一遍",
        user_id="user_p4",
        session_id=session_id,
    )
    ans2 = await orch.run(req2)
    assert ans2.blocked is False
    assert ans2.sql != ""
    assert "顺丰" in ans2.sql
    assert "中通" not in ans2.sql

    # Turn 3: Follow-up ellipsis query in the same session
    req3 = InsightRequest(
        question="那韵达呢？",
        user_id="user_p4",
        session_id=session_id,
    )
    ans3 = await orch.run(req3)
    assert ans3.blocked is False
    assert ans3.sql != ""
    assert "韵达" in ans3.sql


# ==============================================================================
# 4. ONNX Runtime Cross-Encoder Reranker Tests
# ==============================================================================


def test_onnx_reranker_fallback_when_model_missing() -> None:
    """When ONNX model file does not exist, ONNXBGEReranker safely falls back to heuristic."""
    reranker = ONNXBGEReranker(model_path="/non/existent/model.onnx")
    assert reranker.is_available is False

    cands = [
        RerankCandidate(doc_id="d1", text="中通快递华东转运中心包裹延误", initial_score=0.8),
        RerankCandidate(doc_id="d2", text="上海市数码产品销售订单统计", initial_score=0.3),
    ]
    reranked = reranker.rerank("中通快递为什么延误？", cands, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].doc_id == "d1"
    assert reranked[0].rerank_score > reranked[1].rerank_score


def test_onnx_reranker_with_mock_session() -> None:
    """Test ONNXBGEReranker inference math and scoring with active mock session."""
    import numpy as np

    class MockInferenceSession:
        def get_inputs(self) -> list[Any]:
            class InputMeta:
                def __init__(self, name: str) -> None:
                    self.name = name

            return [InputMeta("input_ids"), InputMeta("attention_mask")]

        def run(self, output_names: Any, feed_dict: dict[str, Any]) -> list[Any]:
            # Mock logit computation: higher score if '中通' or '延误' tokens present
            input_ids = feed_dict["input_ids"]
            batch_size = input_ids.shape[0]
            # Emit mock logits where doc 0 has logit +2.0, doc 1 has logit -1.0
            mock_logits = np.array([[2.5], [-1.5]]) if batch_size == 2 else np.zeros((batch_size, 1))
            return [mock_logits]

    reranker = ONNXBGEReranker()
    reranker.session = MockInferenceSession()
    reranker.is_available = True

    cands = [
        RerankCandidate(doc_id="doc_delay", text="中通快递转运中心严重延误", initial_score=0.6),
        RerankCandidate(doc_id="doc_other", text="服饰鞋包类目销量增长", initial_score=0.5),
    ]

    reranked = reranker.rerank("中通延误原因", cands, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].doc_id == "doc_delay"
    assert reranked[0].rerank_score > reranked[1].rerank_score


def test_bge_reranker_hierarchy_and_properties() -> None:
    """Verify BGEReranker backend priority selection and properties."""
    # 1. Default zero-infra mode without env vars: heuristic
    bge_default = BGEReranker()
    assert bge_default.backend_type == "heuristic"
    assert bge_default.is_neural is False

    # 2. Remote URL takes precedence
    bge_remote = BGEReranker(url="http://mock-rerank.internal")
    assert bge_remote.backend_type == "remote"
    assert bge_remote.is_neural is True

    # 3. Heuristic direct test
    heuristic = HeuristicCrossAttentionReranker()
    score_match = heuristic.compute_score("顺丰 延误", "顺丰速运华南枢纽中心因暴雨出现配送延误")
    score_unrelated = heuristic.compute_score("顺丰 延误", "生鲜水果热卖排行榜")
    assert score_match > score_unrelated


@pytest.mark.asyncio
async def test_orchestrator_episodic_memory_recall_wiring() -> None:
    """Verify episodic memory recall injection into orchestrator planner context."""
    from insight_agent.app.memory import MemoryStore
    from insight_agent.app.orchestrator import InsightOrchestrator, InsightRequest, RunConfig
    from insight_agent.data.cn_retail_db import connect_cn_retail, ensure_cn_retail_database

    ensure_cn_retail_database()
    conn = connect_cn_retail()
    mem = MemoryStore()
    mem.remember(
        user_id="user_gold",
        question="查询上海市订单总数",
        sql="SELECT count(*) FROM cn_orders WHERE province='上海市'",
        summary="上海订单统计",
    )

    orch = InsightOrchestrator(conn, memory=mem, config=RunConfig(use_memory=True))

    recalled = orch.memory.recall("user_gold", "查询上海市订单", top_k=2)
    assert len(recalled) == 1
    assert "上海市" in recalled[0].sql

    # Execute request and verify planner receives and executes properly
    req = InsightRequest(
        question="查询上海市订单总数",
        user_id="user_gold",
        session_id="sess_gold",
    )
    ans = await orch.run(req)
    assert ans.blocked is False
    assert ans.sql != ""
    assert len(ans.rows) > 0


def test_orchestrator_prefix_cache_and_gateway_wiring() -> None:
    """Verify PrefixCacheTracker and MultiProviderGateway initialization in orchestrator and api."""
    from unittest.mock import AsyncMock

    from insight_agent.api import create_app
    from insight_agent.app.orchestrator import InsightOrchestrator
    from insight_agent.app.planner import LLMPlanner
    from insight_agent.core.llm import LLMClient
    from insight_agent.core.providers.gateway import PrefixCacheTracker
    from insight_agent.data.cn_retail_db import connect_cn_retail, ensure_cn_retail_database

    ensure_cn_retail_database()
    conn = connect_cn_retail()

    # 1. Orchestrator automatically sets up tracker and binds to LLMPlanner
    mock_client = AsyncMock(spec=LLMClient)
    planner = LLMPlanner(mock_client)
    assert planner.prefix_cache_tracker is None

    tracker = PrefixCacheTracker()
    orch = InsightOrchestrator(conn, planner=planner, prefix_cache_tracker=tracker)
    assert orch.prefix_cache_tracker is tracker
    assert planner.prefix_cache_tracker is tracker

    # 2. FastAPI app exposes tracker and gateway on app.state
    app = create_app(orch)
    assert app.state.prefix_cache_tracker is tracker
    assert app.state.orchestrator is orch

