"""Architectural contract tests for Planners.

These tests assert that:
1. RulesPlanner explicitly declares `consumes_schema_hint = False` and can produce
   deterministic SQL without schema hints (Fast-path).
2. LLMPlanner explicitly declares `consumes_schema_hint = True` and genuinely passes
   the provided schema_hint into the model prompt (Slow-path).
3. Orchestrator respects the protocol boundary, bypassing expensive schema introspection
   when the planner does not consume schema hints.
"""

from __future__ import annotations

import sqlite3

import pytest

from insight_agent.app.orchestrator import InsightOrchestrator, InsightRequest
from insight_agent.app.planner import LLMPlanner, Planner, RulesPlanner
from insight_agent.core.llm import Message, Turn


class _RecordingClient:
    """Stub LLMClient that records messages and returns predefined SQL."""

    def __init__(self, response_sql: str = "SELECT COUNT(*) FROM test_table") -> None:
        self.response_sql = response_sql
        self.recorded_messages: list[list[Message]] = []

    async def complete(self, messages: list[Message], tools: tuple) -> Turn:
        self.recorded_messages.append(messages)
        return Turn(
            text=f"```sql\n{self.response_sql}\n```",
            tool_calls=[],
            stop_reason="end_turn",
        )


def test_planner_protocol_conformance():
    """Verify that both planners declare the consumes_schema_hint contract."""
    rules_p = RulesPlanner()
    assert isinstance(rules_p, Planner)
    assert RulesPlanner.consumes_schema_hint is False

    llm_p = LLMPlanner(_RecordingClient())  # type: ignore[arg-type]
    assert isinstance(llm_p, Planner)
    assert LLMPlanner.consumes_schema_hint is True


@pytest.mark.asyncio
async def test_rules_planner_pure_fastpath_contract():
    """RulesPlanner must work identically with or without schema_hint."""
    planner = RulesPlanner(dataset="cn_retail")
    res_no_hint = await planner.plan("618期间销售总额是多少", schema_hint="")
    assert res_no_hint.understood
    assert "total_amount" in res_no_hint.sql
    assert res_no_hint.model == "rules-cn"
    assert any("fastpath" in note.lower() for note in res_no_hint.notes)

    # Providing a bogus schema_hint must not break the deterministic compiler
    res_with_hint = await planner.plan(
        "618期间销售总额是多少", schema_hint="BOGUS_TABLE (fake_col TEXT)"
    )
    assert res_with_hint.sql == res_no_hint.sql


@pytest.mark.asyncio
async def test_llm_planner_must_consume_schema_hint():
    """LLMPlanner must genuinely inject schema_hint into system prompt."""
    client = _RecordingClient()
    planner = LLMPlanner(client)  # type: ignore[arg-type]

    unique_schema_marker = "CREATE TABLE cn_special_inventory (sku_id TEXT, warehouse_qty INT);"
    res = await planner.plan(
        "查询仓库库存总数",
        schema_hint=unique_schema_marker,
        domain="cn_special_inventory.sku_id: A101, B202",
    )
    assert res.understood
    assert len(client.recorded_messages) == 1

    system_message = client.recorded_messages[0][0]
    assert system_message.role == "system"
    # Assert that the schema_hint actually reached the LLM prompt!
    assert unique_schema_marker in system_message.content
    assert "cn_special_inventory.sku_id: A101, B202" in system_message.content


@pytest.mark.asyncio
async def test_orchestrator_bypasses_schema_introspection_for_rules_planner():
    """When using RulesPlanner, Orchestrator skips PRAGMA schema introspection if not retrieved."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE mock_table (id INT, val TEXT)")

    orchestrator = InsightOrchestrator(conn, planner=RulesPlanner(dataset="cn_retail"))
    assert orchestrator.planner.consumes_schema_hint is False

    # Execute a query through orchestrator
    await orchestrator.run(InsightRequest(question="销售额是多少"))
    # _schema_hint should not be lazily populated because rules planner does not consume it!
    assert orchestrator._schema_hint is None


@pytest.mark.asyncio
async def test_router_runs_before_planning_and_selects_the_planner():
    """The routing decision must precede planning and actually pick a planner.

    Routing used to run after the planner had already been invoked, so the tier
    only labelled a model that was already spent. These assertions fail if the
    decision is moved back after planning, or if it stops influencing which
    planner runs.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE mock_table (id INT, val TEXT)")

    orchestrator = InsightOrchestrator(conn, planner=RulesPlanner(dataset="cn_retail"))

    calls: list[str] = []

    class _Spy(LLMPlanner):
        name = "llm-spy"

        async def plan(self, question, *, schema_hint="", domain="", retrieved=False, **kw):
            calls.append(question)
            return await super().plan(
                question, schema_hint=schema_hint, domain=domain, retrieved=retrieved, **kw
            )

    llm_planner = _Spy(_RecordingClient())  # type: ignore[arg-type]
    orchestrator.planner = llm_planner

    # Routing is decided from the question and the retrieved evidence, then it
    # picks the planner. Whatever the tier, a request that only the model can
    # satisfy must still reach it -- and the choice must be reported.
    simple = InsightOrchestrator(
        sqlite3.connect(":memory:"), planner=RulesPlanner(dataset="cn_retail")
    )
    simple.planner = llm_planner

    # A question inside the compiler's surface, scored simple, downgrades to the
    # deterministic planner -- that is the saving the decision exists to produce.
    trivial = "订单总数"
    trivial_route = simple.router.route(trivial, retrieved_tables=0)
    assert trivial_route.tier == "simple"
    chosen, downgraded = simple.planner_for(trivial, trivial_route)
    assert chosen is simple._rules_planner
    assert downgraded is True

    # A question the compiler cannot parse stays on the live planner: downgrading
    # is a cost optimisation, never a capability limit.
    hard = "帮我看看退货率为什么最近特别高"
    hard_route = simple.router.route(hard, retrieved_tables=4)
    chosen, downgraded = simple.planner_for(hard, hard_route)
    assert chosen is llm_planner
    assert downgraded is False

    ans = await orchestrator.run(InsightRequest(question="销售额是多少", user_id="u1"))
    assert ans.route_reason, "the route decision must be recorded on the answer"
    assert ans.model


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_share_routing_state():
    """Per-request routing decisions must not leak between concurrent callers.

    One orchestrator serves every request, so anything the router decides has to
    travel with the task rather than live on the instance.
    """
    import asyncio

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE mock_table (id INT, val TEXT)")
    orchestrator = InsightOrchestrator(conn)

    questions = ["销售额是多少", "各品类的退款率", "订单总数是多少", "最近7天各城市销售额"] * 4
    results = await asyncio.gather(
        *(orchestrator.run(InsightRequest(question=q, user_id=f"u{i}")) for i, q in enumerate(questions))
    )

    by_question: dict[str, set[str]] = {}
    for r in results:
        by_question.setdefault(r.question, set()).add(r.sql or "")
    assert all(len(v) == 1 for v in by_question.values()), {
        q: v for q, v in by_question.items() if len(v) > 1
    }
