"""Independent Attribution & HITL Security Benchmark Test Suite.

This suite is decoupled from `anomaly_spec.py`:
1. Tests adversarial distractor queries that force Hypothesis 1 to be falsified,
   authentically verifying the multi-turn reflection loop (reflections >= 1).
2. Tests natural language phrasing without template keyword-stuffing.
3. Tests control cases to guarantee 0% false positives.
4. Tests default-insecure mitigation: verifying HITL is secure by default (False)
   and requires explicit resumption to execute high-impact actions.
"""

from __future__ import annotations

import pytest

from insight_agent.api import create_app
from insight_agent.app.graph import InsightGraphEngine
from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
from insight_agent.app.redis_store import RedisMemoryStore
from insight_agent.data.cn_retail_db import (
    connect_cn_retail,
    default_cn_retail_db_path,
    ensure_cn_retail_database,
)


@pytest.fixture(scope="module")
def cn_engine() -> InsightGraphEngine:
    db_path = default_cn_retail_db_path()
    ensure_cn_retail_database(db_path)
    conn = connect_cn_retail(db_path)
    retriever = OlistMultiModalRetriever(conn)
    redis_store = RedisMemoryStore()
    return InsightGraphEngine(conn, retriever, redis_store)


@pytest.fixture(scope="module")
def api_client():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    db_path = default_cn_retail_db_path()
    ensure_cn_retail_database(db_path)
    app = create_app()
    return fastapi_testclient.TestClient(app)


# ==============================================================================
# 1. Adversarial Distractor Cases: Authentic Reflection Loop Verification
# ==============================================================================


class TestAdversarialReflectionAttribution:
    """Tests where the query's surface phrasing misleads toward a false hypothesis,

    verifying that the agent falsifies Hypothesis 1 and reflects to Hypothesis 2.
    """

    def test_distractor_freight_on_logistics_delay(self, cn_engine: InsightGraphEngine) -> None:
        """Query suspects freight surge, but actual root cause is ZTO logistics congestion."""
        query = "华东江浙沪地区中通快递包裹大量客诉，是不是运费加价异常？"
        state = cn_engine.diagnose(query, hitl_approved=False)

        assert state["reflections"] >= 1, "Expected reflection loop to be triggered"
        hyps = state["hypotheses"]
        assert len(hyps) >= 2
        # H1 was the suspected freight hypothesis, which must be falsified
        assert hyps[0]["category"] == "freight"
        assert hyps[0]["status"] == "falsified"
        # H2 was the actual logistics hypothesis, which must be confirmed
        assert hyps[1]["category"] == "logistics"
        assert hyps[1]["status"] == "confirmed"
        # Root cause correctly points to ZTO
        assert "中通" in state["root_cause"] or "华东" in state["root_cause"]

    def test_distractor_logistics_on_remote_freight(self, cn_engine: InsightGraphEngine) -> None:
        """Query suspects carrier delay, but actual root cause is Northwest freight rule misconfig."""
        query = "新疆西藏地区大量订单退单取消，是不是快递配送延误严重？"
        state = cn_engine.diagnose(query, hitl_approved=False)

        assert state["reflections"] >= 1, "Expected reflection loop to be triggered"
        hyps = state["hypotheses"]
        assert len(hyps) >= 2
        # H1 was the suspected logistics hypothesis, which must be falsified in Xinjiang/Tibet
        assert hyps[0]["category"] == "logistics"
        assert hyps[0]["status"] == "falsified"
        # H2 was freight, which must be confirmed
        assert hyps[1]["category"] == "freight"
        assert hyps[1]["status"] == "confirmed"
        assert "运费" in state["root_cause"] or "偏远" in state["root_cause"]

    def test_distractor_logistics_on_fresh_packaging_defect(self, cn_engine: InsightGraphEngine) -> None:
        """Query suspects carrier delay, but actual root cause is fresh food packaging defect."""
        query = "生鲜商户天府冷链专营店大量买家退款维权，是不是顺丰速运物流超时延误了？"
        state = cn_engine.diagnose(query, hitl_approved=False)

        assert state["reflections"] >= 1, "Expected reflection loop to be triggered"
        hyps = state["hypotheses"]
        assert len(hyps) >= 2
        # H1 was suspected logistics, which must be falsified for Tianfu Cold Chain
        assert hyps[0]["category"] == "logistics"
        assert hyps[0]["status"] == "falsified"
        # H2 was product quality, which must be confirmed
        assert hyps[1]["category"] in ("product_quality", "batch_defect")
        assert hyps[1]["status"] == "confirmed"
        assert "天府冷链" in state["root_cause"] or "seller_0028" in state["root_cause"] or "包装" in state["root_cause"]


# ==============================================================================
# 2. Natural Language Anomaly & Control Cases
# ==============================================================================


class TestNaturalLanguageCausalAttribution:
    """Natural phrasing queries without template keyword-stuffing."""

    def test_natural_zto_logistics_delay(self, cn_engine: InsightGraphEngine) -> None:
        query = "江浙沪这边好多买家都在抱怨快递卡在转运中心不动了，到底哪家快递出问题了？"
        state = cn_engine.diagnose(query, hitl_approved=False)
        assert state["blocked"] is False
        assert "中通" in state["root_cause"]
        assert "延误" in state["root_cause"] or "时效" in state["root_cause"] or "积压" in state["root_cause"]

    def test_natural_fresh_packaging_defect(self, cn_engine: InsightGraphEngine) -> None:
        query = "夏天买的冷冻大黄鱼很多客户开箱都发臭漏水了，帮我查查是哪个商家的问题"
        state = cn_engine.diagnose(query, hitl_approved=False)
        assert state["blocked"] is False
        assert "seller_0028" in state["root_cause"] or "天府冷链" in state["root_cause"]

    def test_natural_remote_freight_cancellation(self, cn_engine: InsightGraphEngine) -> None:
        query = "为什么发往乌鲁木齐和拉萨的包裹用户下单后没发货就批量退单了？"
        state = cn_engine.diagnose(query, hitl_approved=False)
        assert state["blocked"] is False
        assert "运费" in state["root_cause"] or "偏远" in state["root_cause"]

    def test_control_normal_baseline_no_false_positive(self, cn_engine: InsightGraphEngine) -> None:
        """Normal query must NOT falsely confirm an anomaly."""
        query = "请排查一下10月份上海地区的订单履约与配送是否正常？"
        state = cn_engine.diagnose(query, hitl_approved=False)
        assert state["blocked"] is False
        # Confirms baseline is normal
        normal_markers = ("正常历史基线", "未见系统性异常", "未见异常", "符合正常预期", "处于历史正常基线")
        assert any(m in state["root_cause"] for m in normal_markers) or not state.get("candidate_actions")
        assert state.get("action_result") is None


# ==============================================================================
# 3. P0 HITL Security by Default & Resume Workflow Tests
# ==============================================================================


class TestHITLSecurityByDefaultAndResume:
    """Verifies that high-impact actions default to unapproved and require explicit resume."""

    def test_engine_diagnose_defaults_to_unapproved(self, cn_engine: InsightGraphEngine) -> None:
        query = "为什么最近华东地区中通快递延迟率激增？"
        # diagnose() with default hitl_approved (which must be False)
        state = cn_engine.diagnose(query)
        assert state["hitl_approved"] is False, "Default hitl_approved must be False"
        assert len(state["candidate_actions"]) > 0, "Candidate action should be generated"
        assert state["action_result"] is None, "High-impact action must NOT execute without approval"
        assert any(
            marker in state["final_report"]
            for marker in ("待人机协同审批确认", "人机协同审批", "需人工审核确认")
        )

    def test_api_diagnose_defaults_to_pending_approval(self, api_client) -> None:
        resp = api_client.post(
            "/diagnose",
            json={"query": "为什么最近华东地区中通快递延迟率激增？", "user_id": "auditor"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "pending_hitl_approval"
        assert payload["hitl_approved"] is False
        assert payload["action_result"] is None
        assert payload["resume_url"] is not None
        assert "/investigation/" in payload["resume_url"]
        assert len(payload["candidate_actions"]) > 0

    def test_api_hitl_resume_executes_action_with_overrides(self, api_client) -> None:
        # Step 1: Diagnose (pending)
        resp1 = api_client.post(
            "/diagnose",
            json={"query": "华东地区中通快递包裹延误", "user_id": "auditor"},
        )
        payload1 = resp1.json()
        tid = payload1["thread_id"]
        assert payload1["status"] == "pending_hitl_approval"

        # Step 2: Explicitly resume via HITL endpoint with amount override (e.g. 18.0)
        resp2 = api_client.post(
            f"/investigation/{tid}/resume",
            json={
                "query": "华东地区中通快递包裹延误",
                "override_arguments": {"amount": 18.0},
            },
        )
        assert resp2.status_code == 200
        payload2 = resp2.json()
        assert payload2["status"] == "resumed"
        assert payload2["action_result"] is not None
        assert "18.00" in payload2["action_result"]
