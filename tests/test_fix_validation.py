import sqlite3
from insight_agent.app.rag import build_schema_docs, _TABLE_ALIASES, _COLUMN_ALIASES
from insight_agent.app.cn_nl2sql import parse_cn_question, compile_cn_sql, CNQueryPlan
from insight_agent.app.tools import build_olist_diagnostic_tool_set
from insight_agent.core.tools import ToolCall
from insight_agent.app.planner import RulesPlanner
from insight_agent.app.multimodal_rag import MultiModalRetriever, CNMultiModalRetriever, OlistMultiModalRetriever
import json
import pytest

def test_cn_table_and_column_aliases_present():
    """Issue 1: Ensure cn_* tables and domestic columns exist in rag.py aliases."""
    expected_tables = ["cn_users", "cn_sellers", "cn_products", "cn_orders", "cn_order_items", "cn_logistics", "cn_reviews", "cn_aftersales"]
    for tbl in expected_tables:
        assert tbl in _TABLE_ALIASES, f"Table {tbl} missing from _TABLE_ALIASES"
        assert len(_TABLE_ALIASES[tbl]) >= 4, f"Table {tbl} aliases insufficient"

    expected_cols = ["order_time", "shipped_time", "actual_delivery_time", "carrier", "route_status", "category_l1", "refund_type"]
    for col in expected_cols:
        assert col in _COLUMN_ALIASES, f"Column {col} missing from _COLUMN_ALIASES"

def test_multimodal_retriever_aliases():
    """Issue 1: Ensure MultiModalRetriever and CNMultiModalRetriever are available."""
    assert MultiModalRetriever is OlistMultiModalRetriever
    assert CNMultiModalRetriever is OlistMultiModalRetriever

@pytest.mark.asyncio
async def test_tools_voucher_currency_compatibility():
    """Issue 1: Ensure issue_voucher supports amount, amount_cny, and amount_brl."""
    class DummyStore:
        def acquire_action_lock(self, lock_id, ttl_seconds):
            from contextlib import contextmanager
            @contextmanager
            def ctx():
                yield True
            return ctx()
    
    registry = build_olist_diagnostic_tool_set(sqlite3.connect(":memory:"), DummyStore())
    
    # Test 1: CNY
    call_cny = ToolCall(id="t1", name="issue_voucher", arguments={"user_id": "u1", "amount_cny": 50.0, "reason": "补偿"})
    res_cny = await registry.run(call_cny)
    assert not res_cny.is_error
    data_cny = json.loads(res_cny.content)
    assert data_cny["status"] == "success"
    assert data_cny["amount_cny"] == 50.0
    assert data_cny["amount"] == 50.0

    # Test 2: BRL (legacy)
    call_brl = ToolCall(id="t2", name="issue_voucher", arguments={"user_id": "u2", "amount_brl": 30.0, "reason": "delay"})
    res_brl = await registry.run(call_brl)
    assert not res_brl.is_error
    data_brl = json.loads(res_brl.content)
    assert data_brl["status"] == "success"
    assert data_brl["amount_brl"] == 30.0

    # Test 3: Standard amount
    call_std = ToolCall(id="t3", name="issue_voucher", arguments={"user_id": "u3", "amount": 20.0})
    res_std = await registry.run(call_std)
    assert not res_std.is_error
    data_std = json.loads(res_std.content)
    assert data_std["status"] == "success"
    assert data_std["amount"] == 20.0

def test_cn_query_plan_time_dimensions():
    """Issue 2: Verify time dimensions extraction and SQL compilation in cn_nl2sql."""
    # 1. 618 promotion window
    p_618 = parse_cn_question("618期间全平台销售总额是多少")
    assert p_618 is not None
    assert p_618.time_window == "618大促"
    assert "2024-06-01" in p_618.start_time
    assert "2024-06-18" in p_618.end_time
    sql_618 = compile_cn_sql(p_618)
    assert "o.order_time >= '2024-06-01" in sql_618
    assert "o.order_time <= '2024-06-18" in sql_618

    # 2. Specific month
    p_mar = parse_cn_question("2024年3月份上海市的订单总量")
    assert p_mar is not None
    assert p_mar.city_filter == "上海市"
    assert "2024-03-01" in p_mar.start_time
    assert "2024-03-31" in p_mar.end_time
    sql_mar = compile_cn_sql(p_mar)
    assert "u.city = '上海市'" in sql_mar
    assert "o.order_time >= '2024-03-01" in sql_mar

    # 3. Monthly trend grouping
    p_trend = parse_cn_question("按月统计2024年各月份订单数走势")
    assert p_trend is not None
    assert p_trend.dimension == "month"
    sql_trend = compile_cn_sql(p_trend)
    assert "strftime('%Y-%m', o.order_time) AS dimension" in sql_trend
    assert "GROUP BY dimension ORDER BY dimension ASC" in sql_trend

    # 4. Logistics time window
    p_log = parse_cn_question("8月份包裹总量及各承运商履约耗时")
    assert p_log is not None
    assert "2024-08-01" in p_log.start_time
    assert "2024-08-31" in p_log.end_time

@pytest.mark.asyncio
async def test_rules_planner_fastpath_contract():
    """Issue 3: Verify RulesPlanner explicitly declares and returns fast-path metadata."""
    planner = RulesPlanner(dataset="cn_retail")
    res = await planner.plan("618期间销售总额是多少", schema_hint="cn_orders(order_id, total_amount)")
    assert res.understood
    assert res.model == "rules-cn"
    assert any("fastpath" in note.lower() for note in res.notes)
    assert "order_time" in res.sql
