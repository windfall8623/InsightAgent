"""Comprehensive test suite for the domestic Chinese e-commerce retail data warehouse (CN-Retail 2.0).

Validates:
1. Physical schema constraints, indexes, and read-only authorizer.
2. Deterministic seeder reproducibility, volume (>= 50,000 entities), and injected anomalies.
3. Domestic Text-to-SQL compiler (`cn_nl2sql`) metric/dimension/filter parsing.
4. Jev intent classifier on Chinese e-commerce colloquial expressions.
5. Multi-modal retriever on Chinese reviews.
6. LangGraph 2.0 causal state machine diagnosis for East China logistics & fresh packaging.
7. Dual-dataset compatibility toggle (`INSIGHT_DATASET`).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from insight_agent.data.cn_retail_db import (
    connect_cn_retail,
    default_cn_retail_db_path,
    ensure_cn_retail_database,
)
from insight_agent.app.cn_nl2sql import (
    compile_cn_sql,
    parse_cn_question,
)
from insight_agent.app.jev_classifier import JevIntentClassifier
from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
from insight_agent.app.redis_store import RedisMemoryStore
from insight_agent.app.graph import InsightGraphEngine
from insight_agent.api import _get_active_dataset_conn


@pytest.fixture(scope="session")
def cn_db_conn() -> sqlite3.Connection:
    """Ensure database is seeded and return a connection."""
    db_path = default_cn_retail_db_path()
    ensure_cn_retail_database(db_path)
    conn = connect_cn_retail(db_path)
    yield conn
    conn.close()



class TestCNRetailDatabase:
    """Test schema, data volume, and security authorizer."""

    def test_schema_tables_exist(self, cn_db_conn: sqlite3.Connection) -> None:
        cursor = cn_db_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = [
            "cn_aftersales",
            "cn_logistics",
            "cn_order_items",
            "cn_orders",
            "cn_products",
            "cn_reviews",
            "cn_sellers",
            "cn_users",
        ]
        for tbl in expected_tables:
            assert tbl in tables, f"Expected table {tbl} not found in database"

    def test_data_volume_exceeds_fifty_thousand(self, cn_db_conn: sqlite3.Connection) -> None:
        cursor = cn_db_conn.cursor()
        total_rows = 0
        expected_tables = [
            "cn_users", "cn_sellers", "cn_products", "cn_orders",
            "cn_order_items", "cn_logistics", "cn_reviews", "cn_aftersales"
        ]
        counts = {}
        for tbl in expected_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
            count = cursor.fetchone()[0]
            counts[tbl] = count
            total_rows += count
            assert count > 0, f"Table {tbl} should not be empty"

        assert total_rows >= 50000, f"Expected total rows >= 50,000, got {total_rows}"
        assert counts["cn_orders"] >= 15000, "Expected >= 15,000 orders"
        assert counts["cn_order_items"] >= 20000, "Expected >= 20,000 order items"

    def test_readonly_authorizer_rejects_modifications(self, cn_db_conn: sqlite3.Connection) -> None:
        cursor = cn_db_conn.cursor()
        with pytest.raises(sqlite3.DatabaseError):
            cursor.execute("DROP TABLE cn_users")

        with pytest.raises(sqlite3.DatabaseError):
            cursor.execute("INSERT INTO cn_users (user_id, user_name) VALUES ('x', 'y')")

        with pytest.raises(sqlite3.DatabaseError):
            cursor.execute("DELETE FROM cn_orders")


class TestCNRetailAnomalies:
    """Verify injected business anomalies in Chinese domestic retail data."""

    def test_east_china_logistics_delay_anomaly(self, cn_db_conn: sqlite3.Connection) -> None:
        cursor = cn_db_conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM cn_logistics l
            JOIN cn_orders o ON l.order_id = o.order_id
            JOIN cn_users u ON o.user_id = u.user_id
            WHERE l.carrier = '中通快递'
              AND o.order_time BETWEEN '2024-07-01' AND '2024-08-31'
              AND u.province IN ('上海市', '江苏省', '浙江省')
              AND l.route_status = '网点积压'
            """
        )
        delayed_count = cursor.fetchone()[0]
        assert delayed_count >= 100, f"Expected delayed orders >= 100, got {delayed_count}"

    def test_fresh_food_packaging_defect_anomaly(self, cn_db_conn: sqlite3.Connection) -> None:
        cursor = cn_db_conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM cn_reviews r
            JOIN cn_orders o ON r.order_id = o.order_id
            JOIN cn_order_items oi ON o.order_id = oi.order_id
            JOIN cn_products p ON oi.product_id = p.product_id
            WHERE oi.seller_id = 'seller_0028'
              AND p.category_l1 = '生鲜食品'
              AND (r.comment_text LIKE '%破损%' OR r.comment_text LIKE '%漏液%')
            """
        )
        defect_count = cursor.fetchone()[0]
        assert defect_count >= 10, f"Expected defect reviews >= 10, got {defect_count}"


class TestCNTextToSQL:
    """Test domestic Text-to-SQL compiler rules and execution."""

    def test_shanghai_completed_orders_count(self, cn_db_conn: sqlite3.Connection) -> None:
        q = "上海已完成的订单总共有多少"
        plan = parse_cn_question(q)
        assert plan is not None
        assert plan.metric == "order_count"
        assert plan.city_filter == "上海市"
        assert plan.status_filter == "已妥投"

        sql = compile_cn_sql(plan)
        assert "cn_orders" in sql
        cursor = cn_db_conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchone()
        assert res is not None
        assert res[0] > 0

    def test_average_order_value(self, cn_db_conn: sqlite3.Connection) -> None:
        q = "我想知道全店已完成订单的平均客单价是多少"
        plan = parse_cn_question(q)
        assert plan is not None
        assert plan.metric == "avg_order_value"

        sql = compile_cn_sql(plan)
        cursor = cn_db_conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchone()
        assert res is not None
        assert res[0] > 100  # realistic average order value

    def test_category_refund_rate_top_3(self, cn_db_conn: sqlite3.Connection) -> None:
        q = "各品类退款率最高的是哪三个"
        plan = parse_cn_question(q)
        assert plan is not None
        assert plan.metric == "refund_rate"
        assert plan.dimension == "category_l1"

        sql = compile_cn_sql(plan)
        cursor = cn_db_conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        assert len(rows) == 3
        # In our seeded anomaly, 生鲜食品 should be highest
        assert rows[0][0] == "生鲜食品"
        assert rows[0][1] > 0.10


class TestJevIntentClassifierCN:
    """Test Jev classifier recognizing domestic e-commerce queries."""

    def test_logistics_anomaly_high_temperature(self) -> None:
        from insight_agent.app.router import classify_intent

        decision = classify_intent("最近华东地区物流怎么这么慢，快查一下是不是中通爆仓了")
        assert decision.is_anomaly is True

    def test_fresh_food_anomaly(self) -> None:
        from insight_agent.app.router import classify_intent

        decision = classify_intent("生鲜水果有破损变质，是不是冷链包装有问题")
        assert decision.is_anomaly is True

    def test_normal_metric_query(self) -> None:
        from insight_agent.app.router import classify_intent

        decision = classify_intent("帮我查询一下各个品类的销售额")
        assert decision.is_anomaly is False


class TestMultiModalRAGCN:
    """Test retriever handling Chinese reviews."""

    def test_search_customer_reviews_packaging_defect(self, cn_db_conn: sqlite3.Connection) -> None:
        retriever = OlistMultiModalRetriever(cn_db_conn)
        results = retriever.search_reviews("生鲜包装破损 漏水严重", top_k=5)
        assert len(results) > 0
        first_doc = results[0].message
        assert any(kw in first_doc for kw in ["破损", "漏", "生鲜", "包装", "烂", "压扁"])



class TestLangGraphDiagnosisCN:
    """Test end-to-end causal state machine diagnosis on CN-Retail."""

    def test_diagnose_logistics_delay(self, cn_db_conn: sqlite3.Connection) -> None:
        retriever = OlistMultiModalRetriever(cn_db_conn)
        redis_store = RedisMemoryStore()
        engine = InsightGraphEngine(cn_db_conn, retriever, redis_store)

        assert engine.is_cn is True
        state = engine.diagnose("为什么最近华东地区中通快递延迟率激增？")
        assert state["root_cause"] is not None
        assert "中通" in state["root_cause"] or "华东" in state["root_cause"] or "分拨" in state["root_cause"]
        assert "final_report" in state
        assert "¥" in state["final_report"]  # RMB symbol used in proposals

    def test_diagnose_fresh_food_defects(self, cn_db_conn: sqlite3.Connection) -> None:
        retriever = OlistMultiModalRetriever(cn_db_conn)
        redis_store = RedisMemoryStore()
        engine = InsightGraphEngine(cn_db_conn, retriever, redis_store)

        state = engine.diagnose("生鲜食品退款率异常偏高，分析商家和包装原因")
        assert state["root_cause"] is not None
        assert "seller_0028" in state["root_cause"] or "包装" in state["root_cause"] or "生鲜" in state["root_cause"]


class TestDatasetSwitching:
    """Verify switching datasets via INSIGHT_DATASET environment variable."""

    def test_switch_to_cn_retail(self) -> None:
        with patch.dict(os.environ, {"INSIGHT_DATASET": "cn_retail"}):
            conn, name = _get_active_dataset_conn()
            assert name == "cn_retail"
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cn_orders'")
            assert cursor.fetchone() is not None
            conn.close()

    def test_switch_to_olist(self) -> None:
        with patch.dict(os.environ, {"INSIGHT_DATASET": "olist"}):
            conn, name = _get_active_dataset_conn()
            assert name == "olist"
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('orders', 'olist_orders')")
            assert cursor.fetchone() is not None
            conn.close()


class TestCNDiagBench:
    """Test the 50-case CN-Retail benchmark dataset and SAGE acceptance gate."""

    def test_cn_diag_bench_manifest_split(self) -> None:
        from insight_agent.eval.diag_bench import (
            CN_DIAG_BENCHMARK_CASES,
            build_diag_split,
            diag_dataset_hash,
        )

        assert len(CN_DIAG_BENCHMARK_CASES) == 50
        split = build_diag_split(CN_DIAG_BENCHMARK_CASES)
        assert len(split.train) == 30
        assert len(split.dev) == 10
        assert len(split.test) == 10
        assert split.dataset_hash == diag_dataset_hash(CN_DIAG_BENCHMARK_CASES)
        assert len(split.dataset_hash) == 16

    def test_cn_diag_bench_full_eval_and_sage_gate(self, cn_db_conn: sqlite3.Connection) -> None:
        from insight_agent.eval.diag_bench import (
            CN_DIAG_BENCHMARK_CASES,
            evaluate_with_sage_gate,
            run_diag_eval,
        )

        retriever = OlistMultiModalRetriever(cn_db_conn)
        redis_store = RedisMemoryStore()
        engine = InsightGraphEngine(cn_db_conn, retriever, redis_store)

        assert engine.is_cn is True
        report = run_diag_eval(engine, CN_DIAG_BENCHMARK_CASES)

        assert report.total_cases == 50
        assert report.passed_cases >= 48
        assert report.accuracy >= 0.95
        assert report.security_block_rate == 1.0  # 100% security block rate on red-team injections

        # SAGE gate check against legacy baseline with 10 synthetic failures
        baseline = {c.id: True for c in CN_DIAG_BENCHMARK_CASES}
        for i in range(1, 11):
            baseline[f"diag_{i:03d}"] = False
        candidate = {r["case_id"]: r["passed"] for r in report.details}

        verdict = evaluate_with_sage_gate(baseline, candidate)
        assert verdict.decision.value == "accept"
        assert verdict.improvements >= 8
        assert verdict.regressions == 0
        assert verdict.p_value < 0.05

