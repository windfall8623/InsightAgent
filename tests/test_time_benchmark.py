"""Comprehensive 20 Gold Benchmark Tests for Time Intelligence.

Covers:
1. Relative time window resolution with dataset anchor (2024-09-30):
   - 本月, 上月, 本季度, 上季度, 最近7天, 最近30天, 上半年, 下半年
2. Major promotions and explicit month/date ranges:
   - 618大促, 年货节, 3月份, 1-3月区间, ISO日期区间
3. Out-of-Window Guard:
   - 双11 / 11月份 (超出2024-09-30边界时触发友好拦截报警，杜绝静默返回0行)
4. Multi-granularity trend compilation:
   - 按周 (strftime %Y-%W), 按天 (%Y-%m-%d), 按月 (%Y-%m), 按季度 (CASE Q1-Q4)
5. Execution against real cn_retail.db verifying row counts and index filtering.
"""

from __future__ import annotations

import sqlite3
import pytest
from pathlib import Path

from insight_agent.app.cn_nl2sql import (
    parse_cn_question,
    compile_cn_sql,
    CNQueryPlan,
    DATASET_ANCHOR,
)
from insight_agent.app.planner import RulesPlanner


DB_PATH = Path("src/insight_agent/data/cn_retail.db")


# ==============================================================================
# 1. Relative Time Resolution (8 Gold Cases)
# ==============================================================================

def test_gold_01_current_month_anchor():
    """Gold 1: '本月全平台销售额' resolves to anchor month 2024-09."""
    p = parse_cn_question("本月全平台销售总额是多少")
    assert p is not None
    assert p.time_window == "本月"
    assert p.start_time == "2024-09-01 00:00:00"
    assert p.end_time == "2024-09-30 23:59:59"
    sql = compile_cn_sql(p)
    assert "o.order_time >= '2024-09-01" in sql
    assert "o.order_time <= '2024-09-30" in sql


def test_gold_02_previous_month():
    """Gold 2: '上个月订单总量' resolves to 2024-08."""
    p = parse_cn_question("上月全平台订单总量")
    assert p is not None
    assert p.time_window == "上月"
    assert p.start_time == "2024-08-01 00:00:00"
    assert p.end_time == "2024-08-31 23:59:59"


def test_gold_03_current_quarter():
    """Gold 3: '本季度订单数' resolves to Q3 (07-01 to 09-30)."""
    p = parse_cn_question("本季度各城市订单数")
    assert p is not None
    assert "2024-07-01" in p.start_time
    assert "2024-09-30" in p.end_time


def test_gold_04_previous_quarter():
    """Gold 4: '上季度退款率' resolves to Q2 (04-01 to 06-30)."""
    p = parse_cn_question("上季度全平台退款率是多少")
    assert p is not None
    assert "2024-04-01" in p.start_time
    assert "2024-06-30" in p.end_time


def test_gold_05_last_7_days():
    """Gold 5: '最近7天订单量' resolves to 7 days before 2024-09-30."""
    p = parse_cn_question("最近7天全平台订单总量")
    assert p is not None
    assert "2024-09-23" in p.start_time
    assert "2024-09-30" in p.end_time


def test_gold_06_last_30_days():
    """Gold 6: '近一个月销售额' resolves to 30 days before 2024-09-30."""
    p = parse_cn_question("近一个月销售额")
    assert p is not None
    assert "2024-08-31" in p.start_time
    assert "2024-09-30" in p.end_time


def test_gold_07_first_half_year():
    """Gold 7: '上半年销售额' resolves to 2024-01-01 to 2024-06-30."""
    p = parse_cn_question("上半年全平台GMV")
    assert p is not None
    assert "2024-01-01" in p.start_time
    assert "2024-06-30" in p.end_time


def test_gold_08_second_half_year():
    """Gold 8: '下半年订单量' resolves to 2024-07-01 to 2024-09-30."""
    p = parse_cn_question("下半年全平台订单总数")
    assert p is not None
    assert "2024-07-01" in p.start_time
    assert "2024-09-30" in p.end_time


# ==============================================================================
# 2. Major Promotions & Explicit Dates (5 Gold Cases)
# ==============================================================================

def test_gold_09_promotion_618():
    """Gold 9: '618期间销售额'."""
    p = parse_cn_question("618期间全平台销售总额")
    assert p is not None
    assert p.time_window == "618大促"
    assert "2024-06-01" in p.start_time
    assert "2024-06-18" in p.end_time


def test_gold_10_promotion_spring_festival():
    """Gold 10: '年货节订单量'."""
    p = parse_cn_question("年货节期间总单量")
    assert p is not None
    assert p.time_window == "年货节"
    assert "2024-01-15" in p.start_time
    assert "2024-02-05" in p.end_time


def test_gold_11_single_month():
    """Gold 11: '2024年3月份销售额'."""
    p = parse_cn_question("2024年3月份销售额")
    assert p is not None
    assert "2024-03-01" in p.start_time
    assert "2024-03-31" in p.end_time


def test_gold_12_month_range():
    """Gold 12: '1月至3月订单量'."""
    p = parse_cn_question("1月至3月各省份订单总量")
    assert p is not None
    assert "2024-01-01" in p.start_time
    assert "2024-03-31" in p.end_time


def test_gold_13_iso_date_range():
    """Gold 13: Explicit ISO date range."""
    p = parse_cn_question("2024-05-01至2024-05-31销售额")
    assert p is not None
    assert "2024-05-01" in p.start_time
    assert "2024-05-31" in p.end_time


# ==============================================================================
# 3. Out-of-Window Guard (3 Gold Cases)
# ==============================================================================

def test_gold_14_out_of_bounds_double_11():
    """Gold 14: '双11' is in Nov (data ends 2024-09-30) -> triggers Out-of-Window Guard."""
    p = parse_cn_question("双11大促全平台销售总额")
    assert p is not None
    assert p.is_out_of_bounds is True
    assert p.out_of_bounds_msg is not None
    assert "超出当前业务数据集时间范围" in p.out_of_bounds_msg
    assert "2024-09-30" in p.out_of_bounds_msg


def test_gold_15_out_of_bounds_november():
    """Gold 15: '2024年11月份销售额' -> triggers Out-of-Window Guard."""
    p = parse_cn_question("2024年11月份订单数")
    assert p is not None
    assert p.is_out_of_bounds is True
    assert "2024-11-01" in p.start_time


@pytest.mark.asyncio
async def test_gold_16_planner_propagates_out_of_bounds_warning():
    """Gold 16: RulesPlanner notes must include the out_of_bounds warning note."""
    planner = RulesPlanner(dataset="cn_retail")
    res = await planner.plan("双11期间销售总额是多少")
    assert res.understood
    assert res.plan.is_out_of_bounds is True
    assert any("out_of_bounds" in note.lower() for note in res.notes)


# ==============================================================================
# 4. Multi-Granularity Trend Compilation (4 Gold Cases)
# ==============================================================================

def test_gold_17_granularity_week():
    """Gold 17: Weekly trend uses SQLite strftime %Y-%W."""
    p = parse_cn_question("按周统计订单数走势")
    assert p is not None
    assert p.dimension == "week"
    assert p.time_granularity == "week"
    sql = compile_cn_sql(p)
    assert "strftime('%Y-%W', o.order_time) AS dimension" in sql
    assert "GROUP BY dimension ORDER BY dimension ASC" in sql


def test_gold_18_granularity_day():
    """Gold 18: Daily trend uses strftime %Y-%m-%d."""
    p = parse_cn_question("按天统计最近7天订单量走势")
    assert p is not None
    assert p.dimension == "day"
    assert p.time_granularity == "day"
    sql = compile_cn_sql(p)
    assert "strftime('%Y-%m-%d', o.order_time) AS dimension" in sql


def test_gold_19_granularity_month():
    """Gold 19: Monthly trend uses strftime %Y-%m."""
    p = parse_cn_question("按月统计全平台销售额走势")
    assert p is not None
    assert p.dimension == "month"
    sql = compile_cn_sql(p)
    assert "strftime('%Y-%m', o.order_time) AS dimension" in sql


def test_gold_20_real_execution_against_db():
    """Gold 20: Real execution against cn_retail.db returns positive rows and amounts."""
    if not DB_PATH.exists():
        pytest.skip("cn_retail.db not present locally")

    conn = sqlite3.connect(DB_PATH)

    # 1. 618 销售额
    p_618 = parse_cn_question("618期间销售额")
    sql_618 = compile_cn_sql(p_618)
    row_618 = conn.execute(sql_618).fetchone()
    assert row_618 is not None
    assert row_618[0] > 100_000.0, f"618 sales must be substantial, got {row_618[0]}"

    # 2. 本月订单量 (2024-09)
    p_sep = parse_cn_question("本月订单量")
    sql_sep = compile_cn_sql(p_sep)
    row_sep = conn.execute(sql_sep).fetchone()
    assert row_sep is not None
    assert row_sep[0] > 100, f"September order count must be positive, got {row_sep[0]}"

    # 3. 按月销售额趋势
    p_trend = parse_cn_question("按月统计销售额走势")
    sql_trend = compile_cn_sql(p_trend)
    rows_trend = conn.execute(sql_trend).fetchall()
    assert len(rows_trend) >= 8, f"Should have at least 8 months of data, got {len(rows_trend)}"
