"""Comprehensive test suite for P1 stage upgrades:

1. P1.1: MCP Dual-Dataset Adaptation & CLI Subcommand Unification
2. P1.2: EvolutionEngine Persistence & Autonomous Closed-Loop Self-Evolution
3. P1.3: A2A Stated Capabilities Grounding (sql.repair & plan.clarify)
4. P1.4: HITL Interactive Gate (TerminalApprover & Approval Loop)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from insight_agent.app.a2a import TaskState
from insight_agent.app.evolution import (
    EvolutionEngine,
    Outcome,
    default_evolution_path,
    default_strategies,
)
from insight_agent.app.mcp_server import MCPServer
from insight_agent.app.orchestrator import (
    InsightOrchestrator,
    InsightRequest,
)
from insight_agent.app.security import (
    SecurityDecision,
    TerminalApprover,
    ToolPolicy,
    Verdict,
)
from insight_agent.app.tools import build_active_mcp_tool_set
from insight_agent.cli import build_parser, main
from insight_agent.core.llm import ToolCall

# --------------------------------------------------------------------------
# P1.1: MCP Tests
# --------------------------------------------------------------------------


class TestP1_1_MCPUpgrades:
    def test_active_mcp_tool_set_cn_retail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INSIGHT_DATASET", "cn_retail")
        registry = build_active_mcp_tool_set()
        names = {s["name"] for s in registry.schemas()}
        assert "query_cn_retail_sql" in names
        assert "search_customer_reviews" in names
        assert "diagnose_anomaly" in names
        assert "issue_voucher" in names

    def test_active_mcp_tool_set_olist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INSIGHT_DATASET", "olist")
        registry = build_active_mcp_tool_set()
        names = {s["name"] for s in registry.schemas()}
        assert "query_olist_sql" in names
        assert "search_customer_reviews" in names
        assert "diagnose_anomaly" in names

    async def test_mcp_server_jsonrpc_execution_on_cn_retail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INSIGHT_DATASET", "cn_retail")
        server = MCPServer(build_active_mcp_tool_set())

        # 1. Initialize
        init_resp = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert init_resp is not None
        assert init_resp["result"]["serverInfo"]["name"] == "insight-analytics"

        # 2. Tools list
        tools_resp = await server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert tools_resp is not None
        tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
        assert "query_cn_retail_sql" in tool_names

        # 3. Tools call
        call_resp = await server.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "query_cn_retail_sql",
                    "arguments": {"sql": "SELECT COUNT(*) as n FROM cn_orders", "limit": 5},
                },
            }
        )
        assert call_resp is not None
        assert call_resp["result"]["isError"] is False
        content_json = json.loads(call_resp["result"]["content"][0]["text"])
        assert content_json["row_count"] == 1
        assert "n" in content_json["columns"]

    def test_cli_parser_and_mcp_dispatch(self) -> None:
        parser = build_parser()
        # Ensure --interactive flag exists
        parsed = parser.parse_args(["--interactive", "mcp"])
        assert parsed.interactive is True
        assert parsed.question == ["mcp"]

        # Ensure main(["mcp"]) calls serve_tools
        with patch("insight_agent.cli.serve_tools") as mock_serve:
            ret = main(["mcp"])
            assert ret == 0
            mock_serve.assert_called_once()


# --------------------------------------------------------------------------
# P1.2: Evolution Persistence & Closed-Loop Tests
# --------------------------------------------------------------------------


class TestP1_2_EvolutionPersistenceAndLoop:
    def test_default_evolution_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INSIGHT_EVOLUTION_PATH", "custom/evo.json")
        assert default_evolution_path() == Path("custom/evo.json")

    def test_evolution_auto_save_and_auto_reload(self, store_dir: Path) -> None:
        save_path = store_dir / "sub" / "evolution.json"
        if save_path.exists():
            save_path.unlink()
        engine = EvolutionEngine(default_strategies(), path=save_path, auto_save_interval=2)

        # 1. First observation: not yet auto_save_interval (2)
        engine.observe(
            Outcome(
                case_id="case_1",
                shape="metric",
                strategy="cheap_hybrid",
                passed=True,
                recall=1.0,
                latency_ms=10.0,
                cost_usd=0.001,
            )
        )
        assert not save_path.exists()

        # 2. Second observation: triggers auto-save
        engine.observe(
            Outcome(
                case_id="case_2",
                shape="metric",
                strategy="cheap_hybrid",
                passed=True,
                recall=1.0,
                latency_ms=12.0,
                cost_usd=0.001,
            )
        )
        assert save_path.exists()

        # 3. Reload in a new instance with same path -> __init__ calls self.load()
        reloaded = EvolutionEngine(default_strategies(), path=save_path)
        assert len(reloaded.outcomes) == 2
        assert reloaded.outcomes[0].shape == "metric"
        assert reloaded.outcomes[0].case_id == "case_1"

    def test_evolution_autonomous_trigger(self, store_dir: Path) -> None:
        save_path = store_dir / "evo_trigger.json"
        if save_path.exists():
            save_path.unlink()
        engine = EvolutionEngine(default_strategies(), path=save_path)

        for i in range(6):
            engine.observe(
                Outcome(
                    case_id=f"case_{i}",
                    shape="filtered_metric",
                    strategy="cheap_flat",
                    passed=True,
                    recall=1.0,
                    latency_ms=5.0,
                    cost_usd=0.0005,
                )
            )

        # Trigger auto evolve
        changed = engine.maybe_auto_evolve(min_observations=5)
        assert changed is True
        assert engine.policy.get("filtered_metric") == "cheap_flat"
        assert engine.cycles == 1
        assert save_path.exists()

    async def test_orchestrator_closed_loop_evolution_integration(
        self, conn, store_dir: Path
    ) -> None:
        save_path = store_dir / "orch_evo.json"
        if save_path.exists():
            save_path.unlink()
        engine = EvolutionEngine(default_strategies(), path=save_path, auto_save_interval=1)
        orch = InsightOrchestrator(conn, evolution=engine)

        await orch.run(InsightRequest(question="总销售额是多少", user_id="tester"))
        assert len(engine.outcomes) >= 1
        assert save_path.exists()


# --------------------------------------------------------------------------
# P1.3: A2A Stated Capabilities (sql.repair & plan.clarify) Tests
# --------------------------------------------------------------------------


class TestP1_3_A2ASkills:
    async def test_a2a_plan_clarify_delegation(self, conn) -> None:
        orch = InsightOrchestrator(conn)
        task = await orch.a2a.delegate("plan.clarify", {"question": "看下数据"})
        assert task.state == TaskState.COMPLETED
        assert task.output.get("clarification_needed") is True
        assert len(task.output.get("suggestions", [])) >= 2
        assert "看下数据" in task.output.get("message", "")

    async def test_a2a_sql_repair_delegation(self, conn) -> None:
        orch = InsightOrchestrator(conn)
        task = await orch.a2a.delegate(
            "sql.repair",
            {
                "question": "各城市订单",
                "problem": "column city requires grouping",
                "plan": None,
            },
        )
        assert task.state == TaskState.COMPLETED
        assert task.output.get("repaired") is True
        assert "column city requires grouping" in task.output.get("problem", "")

    async def test_orchestrator_handles_ambiguous_question_with_clarify(self, conn) -> None:
        orch = InsightOrchestrator(conn)
        res = await orch.run(InsightRequest(question="统计一下", user_id="tester"))
        assert res.sql == ""
        # Verifies clarification suggestions are present in alerts
        alert_kinds = [a["kind"] for a in res.alerts]
        assert "clarification_needed" in alert_kinds


# --------------------------------------------------------------------------
# P1.4: HITL Approval Tests
# --------------------------------------------------------------------------


class TestP1_4_HITLApprover:
    def test_terminal_approver_accepted(self) -> None:
        approver = TerminalApprover(prompt_fn=lambda _prompt: "y")
        decision = SecurityDecision(Verdict.CONFIRM, reason="table scan risk", detail={"sql": "SELECT *"})
        call = ToolCall(id="c1", name="run_sql", arguments={"sql": "SELECT *"})
        assert approver(call, decision) is True

    def test_terminal_approver_declined(self) -> None:
        approver = TerminalApprover(prompt_fn=lambda _prompt: "n")
        decision = SecurityDecision(Verdict.CONFIRM, reason="table scan risk", detail={"sql": "SELECT *"})
        call = ToolCall(id="c1", name="run_sql", arguments={"sql": "SELECT *"})
        assert approver(call, decision) is False

    async def test_policy_enforcing_registry_with_approver(self, conn) -> None:
        from insight_agent.app.rag import build_retriever
        from insight_agent.app.security import PolicyEnforcingRegistry
        from insight_agent.app.tools import ToolContext, build_tool_set

        ctx = ToolContext(conn=conn, retriever=build_retriever(conn))
        raw_tools = build_tool_set(ctx)

        # High-impact query matching _HIGH_IMPACT regex (cross join)
        high_impact_sql = "SELECT * FROM orders CROSS JOIN users LIMIT 1"

        # 1. Denied by approver
        denying_policy = ToolPolicy(confirm_on_high_impact=True)
        registry_denied = PolicyEnforcingRegistry(
            raw_tools,
            denying_policy,
            approver=lambda _c, _d: False,
        )
        res_denied = await registry_denied.run(
            ToolCall(id="1", name="run_sql", arguments={"sql": high_impact_sql})
        )
        assert res_denied.is_error is True
        assert "DECLINED" in res_denied.content

        # 2. Approved by approver
        approving_policy = ToolPolicy(confirm_on_high_impact=True)
        registry_approved = PolicyEnforcingRegistry(
            raw_tools,
            approving_policy,
            approver=lambda _c, _d: True,
        )
        res_approved = await registry_approved.run(
            ToolCall(id="2", name="run_sql", arguments={"sql": high_impact_sql})
        )
        assert res_approved.is_error is False
