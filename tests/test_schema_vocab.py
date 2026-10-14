"""Unit tests for the unified schema_vocab and semantic metric layer."""

from __future__ import annotations

import sqlite3
import pytest
from insight_agent.app.schema_vocab import (
    get_all_table_aliases,
    get_all_column_aliases,
    mine_column_literals,
    match_metric,
    METRIC_DEFINITIONS,
)
from insight_agent.app.rag import build_schema_docs, build_retriever, _TABLE_ALIASES, _COLUMN_ALIASES
from insight_agent.app.multimodal_rag import TABLE_SYNONYMS


def test_schema_vocab_is_single_source_of_truth():
    """rag.py and multimodal_rag.py must both use schema_vocab tables."""
    assert _TABLE_ALIASES == get_all_table_aliases()
    assert TABLE_SYNONYMS == get_all_table_aliases()
    assert _COLUMN_ALIASES == get_all_column_aliases()


def test_metric_definitions_coverage():
    """Verify semantic metrics can be matched from natural language."""
    # 1. 履约时效
    m_speed = match_metric("分析各承运商的履约时效")
    assert m_speed is not None
    assert m_speed["name"] == "履约时效"
    assert "cn_logistics" in m_speed["tables"]

    # 2. 客单价
    m_arpu = match_metric("统计上海市用户的客单价")
    assert m_arpu is not None
    assert m_arpu["name"] == "客单价"
    assert "cn_orders" in m_arpu["tables"]

    # 3. 货损率
    m_loss = match_metric("排查大促期间的货损率")
    assert m_loss is not None
    assert m_loss["name"] == "货损率"
    assert "cn_aftersales" in m_loss["tables"]


def test_literal_mining_on_memory_db():
    """Literal mining extracts distinct values from low-cardinality columns."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE cn_logistics (carrier TEXT, route_status TEXT)")
    conn.execute("INSERT INTO cn_logistics VALUES ('顺丰速运', '已签收')")
    conn.execute("INSERT INTO cn_logistics VALUES ('中通快递', '正常在途')")
    conn.execute("INSERT INTO cn_logistics VALUES ('顺丰速运', '网点积压')")

    carriers = mine_column_literals(conn, "cn_logistics", "carrier")
    assert set(carriers) == {"顺丰速运", "中通快递"}

    status = mine_column_literals(conn, "cn_logistics", "route_status")
    assert set(status) == {"已签收", "正常在途", "网点积压"}


def test_retriever_recalls_tables_from_metric_synonyms():
    """Retriever must recall cn_logistics for '履约时效' and cn_orders for '客单价'."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE cn_logistics (carrier TEXT, actual_delivery_time TEXT, shipped_time TEXT)")
    conn.execute("CREATE TABLE cn_orders (order_id TEXT, total_amount REAL, user_id TEXT)")
    conn.execute("CREATE TABLE cn_aftersales (refund_id TEXT, order_id TEXT, refund_reason TEXT)")

    retriever = build_retriever(conn)

    # 1. '履约时效' should retrieve cn_logistics
    hits_speed = retriever.search_tables("履约时效", top_k=1)
    assert len(hits_speed) > 0
    assert hits_speed[0][0] == "cn_logistics"

    # 2. '客单价' should retrieve cn_orders
    hits_arpu = retriever.search_tables("客单价", top_k=1)
    assert len(hits_arpu) > 0
    assert hits_arpu[0][0] == "cn_orders"

    # 3. '货损率' should retrieve cn_aftersales
    hits_damage = retriever.search_tables("货损率", top_k=1)
    assert len(hits_damage) > 0
    assert hits_damage[0][0] == "cn_aftersales"
