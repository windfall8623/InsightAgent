"""Offline regressions for lossless SQL planning, CN binding and model repairs."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

import insight_agent.app.planner as planner_module
from insight_agent.app.planner import (
    CN_DOMAIN_CONVENTIONS,
    LLMPlanner,
    PlannerClientError,
    PlannerResult,
    RulesPlanner,
    _extract_sql,
    data_domain,
    schema_hint,
)
from insight_agent.app.sql_text import (
    extract_readonly_sql,
    sql_skeleton,
    validate_readonly_sql,
)
from insight_agent.core.llm import Message, Turn


@pytest.mark.parametrize("sql", [
    "SELECT 1 AS 销售额",
    'SELECT "订单金额" AS "销售额（元）" FROM "中文订单"',
    "SELECT `订单金额` AS `销售;额` FROM `中文订单`",
    "SELECT [订单金额] AS [销售额] FROM [中文订单]",
    "SELECT '中文;不要截断' AS 备注;",
    "SELECT 'it''s;完整' AS \"quoted\"\";alias\";",
    "SELECT 'DROP TABLE cn_orders; -- 只是字符串' AS 备注",
    "SELECT replace('a;b', ';', ',') AS 文本;",
    "-- 中文注释 SELECT 9; DROP TABLE old\nSELECT 1 AS 数量; -- 尾注释",
    "/* SELECT 99; DROP TABLE old; */ SELECT /* 中文 */ 1; /* 完整 */",
    "SELECT 1 /* ; INSERT INTO old VALUES (1) */ AS 数量",
    "WITH 汇总 AS (SELECT 1 AS 金额) SELECT SUM(金额) AS 销售额 FROM 汇总;",
    "WITH a AS (SELECT 1 AS x), b AS (SELECT x + 1 AS x FROM a) SELECT x FROM b",
    "WITH RECURSIVE seq(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM seq WHERE n<3) SELECT SUM(n) FROM seq;",
    "SELECT 1 UNION ALL SELECT 2",
    "SELECT (SELECT COUNT(*) FROM cn_orders) AS 订单数",
    "SELECT '```sql\nSELECT 99;\n```' AS 文本;",
])
def test_sql_is_preserved_verbatim(sql: str) -> None:
    parsed = extract_readonly_sql(sql)
    assert parsed.outcome == "accepted", parsed
    assert parsed.sql == sql
    assert _extract_sql(sql) == sql
    assert validate_readonly_sql(sql).sql == sql


@pytest.mark.parametrize("text, expected", [
    ("```sql\nSELECT 1 AS 数量;\n```", "SELECT 1 AS 数量;"),
    ("```SQL\r\nSELECT 1\r\n```", "SELECT 1"),
    ("```\nSELECT 2\n```", "SELECT 2"),
    ("~~~sqlite\nSELECT 2\n~~~", "SELECT 2"),
    ("Sure! SELECT 3", "SELECT 3"),
    ("查询如下：\n```sql\nWITH t AS (SELECT 1 AS n) SELECT n FROM t;\n```\n结果按当前数据计算。",
     "WITH t AS (SELECT 1 AS n) SELECT n FROM t;"),
    ("这是查询。\n-- 保留这条注释\nSELECT 'a;b' AS 备注;\n说明：单条只读查询。",
     "-- 保留这条注释\nSELECT 'a;b' AS 备注;"),
    ("SELECT 1\nExplanation: returns a constant.", "SELECT 1"),
    ("```sql\nSELECT '```' AS 标记;\n```", "SELECT '```' AS 标记;"),
])
def test_safe_prose_and_fences_are_unwrapped(text: str, expected: str) -> None:
    assert _extract_sql(text) == expected


@pytest.mark.parametrize("sql", [
    "SELECT 1; SELECT 2",
    "SELECT 1; -- 中间注释\nDROP TABLE cn_orders",
    "SELECT '; still a string'; DELETE FROM cn_orders",
    "WITH t AS (SELECT 1) SELECT * FROM t; UPDATE cn_orders SET total_amount=0",
    "SELECT 1; /* 伪装注释 */ PRAGMA user_version",
    "SELECT 1; ATTACH DATABASE 'x' AS x",
    "SELECT 1; END TRANSACTION",
    "SELECT 1; DROP TABLE cn_orders; 'unterminated",
    "```sql\nSELECT 1; SELECT 2\n```",
    "```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```",
    "```sql\nSELECT 1\n```\nDROP TABLE cn_orders",
    "DROP TABLE cn_orders;\n```sql\nSELECT 1\n```",
    "SELECT 1;;",
])
def test_no_safe_prefix_is_salvaged_from_multiple_statements(sql: str) -> None:
    parsed = extract_readonly_sql(sql)
    assert not parsed.sql
    assert parsed.outcome in {"multiple_statements", "unsafe_sql"}, parsed


@pytest.mark.parametrize("sql", [
    "DELETE FROM cn_orders",
    "INSERT INTO cn_orders SELECT * FROM cn_orders",
    "CREATE TABLE other AS SELECT * FROM cn_orders",
    "EXPLAIN SELECT 1",
    "PRAGMA table_info(cn_orders)",
    "SELECT 1 INTO another_table",
    "WITH t AS (SELECT 1) DELETE FROM cn_orders",
    "WITH t AS (DELETE FROM cn_orders RETURNING *) SELECT * FROM t",
    "WITH t AS (SELECT 1) INSERT INTO cn_orders SELECT * FROM t",
    "SELECT load_extension('extension')",
    'SELECT "writefile"(\'path\', \'contents\')',
    "DROP TABLE cn_orders /* unfinished",
])
def test_writes_are_denied_without_extracting_inner_select(sql: str) -> None:
    parsed = extract_readonly_sql(sql)
    assert parsed.sql == ""
    assert parsed.outcome == "unsafe_sql", parsed


@pytest.mark.parametrize("sql", [
    "SELECT",
    "SELECT * FROM cn_orders WHERE",
    "SELECT SUM(total_amount FROM cn_orders",
    "SELECT 'unterminated; SELECT 2",
    "SELECT 1 /* unfinished",
    "SELECT 1) AS broken",
    "WITH t AS (SELECT 1)",
    "WITH t AS (SELECT 1) SELECT",
    "SELECT 1\nSELECT 2",
    "SELECT 语句。",
    "请提供有效的 SELECT 语句。",
    "```sql\nSELECT 1",
    "```python\nSELECT 1\n```",
])
def test_malformed_or_truncated_sql_is_explicitly_invalid(sql: str) -> None:
    parsed = extract_readonly_sql(sql)
    assert not parsed.sql
    assert parsed.outcome == "invalid_sql", parsed
    assert parsed.error


def test_comments_quotes_and_literal_semicolons_are_not_guard_keywords() -> None:
    sql = "-- DELETE;\nSELECT 'DROP; UPDATE' AS \"CREATE;\", [DELETE], `INSERT` /* ; */ FROM cn_orders;"
    skeleton = sql_skeleton(sql)
    assert all(keyword not in skeleton for keyword in ("DELETE", "DROP", "UPDATE", "CREATE", "INSERT"))
    assert skeleton.count(";") == 1
    assert "SELECT" in skeleton and "FROM cn_orders" in skeleton
    assert len(skeleton) == len(sql)
    assert validate_readonly_sql(sql).outcome == "accepted"
    with pytest.raises(ValueError, match="unterminated"):
        sql_skeleton("SELECT 'broken")


def test_quoted_table_names_remain_visible_to_high_impact_heuristics() -> None:
    sql = 'SELECT * FROM "orders", [users]'
    skeleton = sql_skeleton(sql, mask_identifiers=False)
    assert len(sql) == len(skeleton)
    assert re.search(r"\*\s+from\s+\w+\s*,\s*\w+", skeleton, re.IGNORECASE)
    misleading = sql_skeleton('SELECT "DROP TABLE; CROSS JOIN" FROM [orders]', mask_identifiers=False)
    assert "DROP" not in misleading and "CROSS JOIN" not in misleading
    assert validate_readonly_sql('SELECT "load_extension"(\'bad\')').outcome == "unsafe_sql"


def test_sql_validation_compiles_but_never_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    import insight_agent.app.sql_text as sql_text

    executed: list[str] = []
    calls: list[object] = []
    connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
            executed.append(sql)
            return super().execute(sql, parameters)

    def isolated(database: str) -> sqlite3.Connection:
        assert database == ":memory:"
        conn = connect(database, factory=TrackingConnection)
        conn.create_function("side_effect_probe", 0, lambda: calls.append(True))
        return conn

    monkeypatch.setattr(sql_text.sqlite3, "connect", isolated)
    assert validate_readonly_sql("SELECT side_effect_probe()").outcome == "accepted"
    assert executed == ["EXPLAIN SELECT side_effect_probe()"]
    assert calls == []


@pytest.mark.parametrize("text, outcome", [
    ("", "empty"),
    ("   ", "empty"),
    ("I cannot answer with the available schema.", "refusal"),
    ("抱歉，无法提供这个查询。", "refusal"),
    ("No relevant data is available.", "no_sql"),
])
def test_missing_sql_is_distinct_from_security_denial(text: str, outcome: str) -> None:
    assert extract_readonly_sql(text).outcome == outcome


@pytest.fixture
def cn_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    schema = Path(planner_module.__file__).parents[1] / "data" / "cn_retail_schema.sql"
    conn.executescript(schema.read_text(encoding="utf-8"))
    conn.executemany("INSERT INTO cn_users VALUES (?, ?, ?, ?)", [
        ("u1", "浙江省", "杭州市", "金卡会员"),
        ("u2", "上海市", "上海市", "黑钻VIP"),
        ("u3", "北京市", "北京市", "银卡会员"),
        ("u4", "江苏省", "南京市", "普客"),
    ])
    conn.executemany("INSERT INTO cn_orders VALUES (?, ?, ?, ?, ?, ?)", [
        ("o1", "u1", "已妥投", "2025-02-03 08:09:10", 100.0, "天猫App"),
        ("o2", "u2", "已取消", "2025-05-06 20:21:22", 100.0, "微信小程序"),
        ("o3", "u3", "运输中", "2025-03-04 08:00:00", 50.0, "直播间"),
        ("o4", "u4", "售后中", "2025-03-04 09:00:00", 20.0, "淘宝网页版"),
    ])
    conn.execute("INSERT INTO cn_sellers VALUES ('s1', '商店', '浙江省', '杭州市')")
    conn.execute("INSERT INTO cn_products VALUES ('p1', '家居家装', '厨房卫浴', '商品', 50)")
    conn.execute("INSERT INTO cn_order_items VALUES ('i1', 'o1', 'p1', 's1', 2, 50)")
    conn.execute("INSERT INTO cn_logistics VALUES ('l1', 'o1', '顺丰速运', '2025-02-04 00:00:00', '2025-02-06 00:00:00', '2025-02-05 01:02:03', '已签收')")
    conn.execute("INSERT INTO cn_reviews VALUES ('r1', 'o1', 5, '优质推荐/时效迅捷', '不错')")
    conn.execute("INSERT INTO cn_aftersales VALUES ('a1', 'o1', '退货退款', '商品质量问题/外包装破损', 50, '已退款')")
    yield conn
    from insight_agent.data.database import close

    close(conn)


def test_cn_domain_uses_live_enums_and_dates_not_legacy_or_seed(cn_conn: sqlite3.Connection) -> None:
    # Even if an unrelated old table exists, CN metadata is not blended with it.
    cn_conn.execute("CREATE TABLE orders (order_date TEXT, status TEXT, channel TEXT)")
    cn_conn.execute("INSERT INTO orders VALUES ('1999-01-01', 'completed', 'app')")
    cn_conn.execute("CREATE TABLE users (city TEXT, member_level TEXT)")
    cn_conn.execute("INSERT INTO users VALUES ('Beijing', 'gold')")
    domain = data_domain(cn_conn)
    assert domain == data_domain(cn_conn)
    assert "dataset: cn_retail" in domain
    for value in ("金卡会员", "黑钻VIP", "银卡会员", "普客", "天猫App", "微信小程序", "直播间", "淘宝网页版", "家居家装"):
        assert f'"{value}"' in domain
    assert "cn_orders.order_status:" in domain
    assert "cn_aftersales.status:" in domain
    assert '"商品质量问题/外包装破损"' in domain
    assert '"优质推荐/时效迅捷"' in domain
    assert "cn_orders.order_time ranges 2025-02-03 08:09:10 .. 2025-05-06 20:21:22" in domain
    assert "cn_logistics.actual_delivery_time ranges 2025-02-05 01:02:03" in domain
    assert all(old not in domain for old in ("1999-01-01", "Beijing", '"gold"', '"app"', "2024-09-30"))
    assert len(domain) <= 4096


def test_partial_cn_domain_does_not_invent_missing_columns() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE cn_orders (order_id TEXT, channel TEXT)")
        conn.execute("INSERT INTO cn_orders VALUES ('o1', '新的渠道')")
        conn.execute("CREATE TABLE orders (order_date TEXT, status TEXT)")
        conn.execute("INSERT INTO orders VALUES ('1900-01-01', 'old')")
        domain = data_domain(conn)
    assert 'cn_orders.channel: "新的渠道"' in domain
    assert "cn_orders.order_status" not in domain
    assert "cn_orders.order_time" not in domain
    assert "1900" not in domain


def test_cn_domain_budget_marks_sampling_and_preserves_exact_literals(cn_conn: sqlite3.Connection) -> None:
    cn_conn.executemany("INSERT INTO cn_users VALUES (?, ?, ?, ?)", [
        (f"extra{i}", f"省{i:03d}", f"城市{i:03d}", "金卡会员") for i in range(120)
    ])
    cn_conn.execute("UPDATE cn_users SET city=? WHERE user_id='extra0'", ("A" * 2000,))
    domain = data_domain(cn_conn)
    assert len(domain) <= 4096
    assert "sample; additional values omitted" in domain
    assert "A" * 100 not in domain
    assert '"金卡会员"' in domain
    assert "2025-05-06 20:21:22" in domain


def test_legacy_domain_still_describes_the_actual_legacy_connection() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.executescript("CREATE TABLE users (city TEXT, member_level TEXT); CREATE TABLE orders (status TEXT, channel TEXT, order_date TEXT);")
        conn.execute("INSERT INTO users VALUES ('Beijing', 'gold')")
        conn.execute("INSERT INTO orders VALUES ('completed', 'app', '2023-11-01')")
        domain = data_domain(conn)
    assert 'users.member_level: "gold"' in domain
    assert 'orders.channel: "app"' in domain
    assert "2023-11-01" in domain
    assert "cn_" not in domain


@pytest.mark.parametrize("binding", ["dataset", "schema", "domain", "environment"])
async def test_confirmed_cn_parse_failure_never_calls_legacy_parser(
    monkeypatch: pytest.MonkeyPatch, binding: str,
) -> None:
    import insight_agent.app.cn_nl2sql as cn_module

    monkeypatch.setenv("INSIGHT_DATASET", "cn_retail" if binding == "environment" else "olist")
    parser = Mock(side_effect=AssertionError("legacy orders/refunds fallback must never run"))
    monkeypatch.setattr(planner_module, "parse_question", parser)
    monkeypatch.setattr(cn_module, "parse_cn_question", lambda *a, **k: None)
    planner = RulesPlanner(dataset="cn_retail" if binding == "dataset" else None)
    hints = {
        "schema_hint": "cn_orders(order_id TEXT)" if binding == "schema" else "",
        "domain": "cn_orders.channel: 天猫App" if binding == "domain" else "",
    }
    result = await planner.plan("退款金额", **hints)
    assert result.sql == "" and result.plan is None
    assert result.model == "rules-cn"
    assert result.parse_outcome == "unrecognized"
    assert not planner.probe("退款金额", **hints)
    assert not planner.downgrade_safe("退款金额", **hints)
    parser.assert_not_called()


async def test_cn_rule_parser_exception_is_unrecognized_not_generic_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import insight_agent.app.cn_nl2sql as cn_module

    monkeypatch.setattr(cn_module, "parse_cn_question", Mock(side_effect=ValueError("CN parse failed")))
    legacy = Mock(side_effect=AssertionError("no legacy fallback"))
    monkeypatch.setattr(planner_module, "parse_question", legacy)
    planner = RulesPlanner(dataset="cn_retail")
    result = await planner.plan("金额")
    assert result.parse_outcome == "unrecognized"
    assert not result.understood
    assert not planner.downgrade_safe("金额")
    legacy.assert_not_called()


async def test_cn_rules_keep_precise_member_channel_and_status_filters(cn_conn: sqlite3.Connection) -> None:
    result = await RulesPlanner(dataset="cn_retail").plan("金卡会员在天猫App的已妥投订单数")
    assert result.understood
    assert "'金卡会员'" in result.sql and "'天猫App'" in result.sql and "'已妥投'" in result.sql
    assert "cn_orders" in result.sql and "cn_users" in result.sql
    assert cn_conn.execute(result.sql).fetchone()[0] == 1
    assert result.plan is not None
    assert "member_level = '金卡会员'" in result.plan.filters


def test_cn_downgrade_keeps_existing_join_rate_and_comparison_exclusions() -> None:
    planner = RulesPlanner(dataset="cn_retail")
    assert planner.downgrade_safe("订单总数")
    assert not planner.downgrade_safe("各城市销售额")
    assert not planner.downgrade_safe("各渠道退款率")
    assert not planner.downgrade_safe("销售额环比")


class RecordingLLM:
    model = "stub-sql"

    def __init__(self, turns: Sequence[Turn]) -> None:
        self.turns = list(turns)
        self.calls: list[list[Message]] = []

    async def complete(self, messages: Sequence[Message], tools: Sequence[dict[str, Any]]) -> Turn:
        assert tools == ()
        self.calls.append(list(messages))
        return self.turns.pop(0)


async def test_repair_calls_routed_client_with_current_schema_error_history_and_filters(
    monkeypatch: pytest.MonkeyPatch, cn_conn: sqlite3.Connection,
) -> None:
    wrong = RecordingLLM([])
    corrected_sql = "WITH totals AS (SELECT SUM(total_amount) AS 金额 FROM cn_orders) SELECT 金额 FROM totals"
    routed = RecordingLLM([Turn(text=corrected_sql, stop_reason="stop")])
    planner = LLMPlanner(wrong, clients={"repair-model": routed})
    generic = Mock(side_effect=AssertionError("repair must not use generic parse_question"))
    monkeypatch.setattr(planner_module, "parse_question", generic)
    history = [Message(role="user", content="我只看金卡会员"), Message(role="assistant", content="好的")]
    filters = {"member_level": "金卡会员", "channel": "天猫App"}
    current_schema = schema_hint(cn_conn)
    domain = data_domain(cn_conn)
    previous = "SELECT SUM(amount) FROM cn_orders"
    error = "no such column: amount"
    result = await planner.repair(
        "销售额是多少", previous, error, schema_hint=current_schema, domain=domain,
        compressed_history=history, model="repair-model", route_tier="complex",
        session_id="test-session", standing_filters=filters,
    )
    assert planner.client is wrong
    assert wrong.calls == [] and len(routed.calls) == 1
    messages = routed.calls[0]
    assert current_schema in messages[0].content
    assert domain in messages[0].content
    assert messages[1:3] == history
    assert messages[-2].role == "assistant" and messages[-2].content == previous
    assert error in messages[-1].content
    assert "INVALID" in messages[-1].content and "entire statement" in messages[-1].content
    assert "金卡会员" in messages[-4].content and "天猫App" in messages[-4].content
    assert result.sql == corrected_sql
    assert result.parse_outcome == "accepted" and result.stop_reason == "stop"
    assert result.model == "repair-model" and result.tokens_in > 0 and result.tokens_out > 0
    assert len(history) == 2 and filters == {"member_level": "金卡会员", "channel": "天猫App"}
    generic.assert_not_called()


@pytest.mark.parametrize("stop_reason", ["length", "max_tokens", "MAX_TOKENS", "max_output_tokens"])
async def test_truncated_response_is_never_executed_even_when_prefix_is_valid(stop_reason: str) -> None:
    client = RecordingLLM([Turn(text="SELECT 1", stop_reason=stop_reason)])
    result = await LLMPlanner(client).plan("q")
    assert result.sql == "" and not result.understood
    assert result.text == "SELECT 1"
    assert result.stop_reason == stop_reason and result.parse_outcome == "truncated"
    assert "complete SQL" in result.parse_error


@pytest.mark.parametrize("text, stop_reason, outcome", [
    ("SELECT 1; DROP TABLE cn_orders", "length", "multiple_statements"),
    ("WITH t AS (SELECT 1) DELETE FROM cn_orders", "length", "unsafe_sql"),
    ("I cannot answer that.", "length", "refusal"),
    ("SELECT 1", "content_filter", "refusal"),
    ("SELECT 1 FROM", "end_turn", "invalid_sql"),
    ("", "end_turn", "empty"),
])
async def test_llm_diagnostics_preserve_security_refusal_and_parse_failures(
    text: str, stop_reason: str, outcome: str,
) -> None:
    result = await LLMPlanner(RecordingLLM([Turn(text=text, stop_reason=stop_reason)])).plan("q")
    assert not result.sql
    assert result.parse_outcome == outcome
    assert result.stop_reason == stop_reason
    assert result.parse_error


async def test_repair_uses_one_call_and_rejects_bad_new_output() -> None:
    client = RecordingLLM([Turn(text="SELECT 1; DELETE FROM cn_orders")])
    result = await LLMPlanner(client).repair(
        "销售额", "SELECT SUM(", "truncated at max_tokens", schema_hint="cn_orders(total_amount REAL)",
    )
    assert len(client.calls) == 1
    assert result.sql == "" and result.parse_outcome == "multiple_statements"
    assert "truncated at max_tokens" in client.calls[0][-1].content


async def test_client_error_wrapper_does_not_mask_local_parse_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    cause = RuntimeError("offline provider failure")

    class FailingClient:
        async def complete(self, messages: Any, tools: Any) -> Turn:
            raise cause

    with pytest.raises(PlannerClientError) as caught:
        await LLMPlanner(FailingClient()).plan("q")
    assert caught.value.__cause__ is cause
    assert "RuntimeError" in str(caught.value)

    monkeypatch.setattr(planner_module, "extract_readonly_sql", Mock(side_effect=ValueError("local parse fault")))
    with pytest.raises(ValueError, match="local parse fault"):
        await LLMPlanner(RecordingLLM([Turn(text="SELECT 1")])).plan("q")


def test_planner_result_fields_are_backwards_compatible() -> None:
    result = PlannerResult(sql="SELECT 1")
    assert result.understood
    assert result.stop_reason == result.parse_outcome == result.parse_error == ""


async def test_metric_contract_is_cn_only_and_addresses_grain_denominator_and_status() -> None:
    cn_client = RecordingLLM([Turn(text="SELECT 1")])
    await LLMPlanner(cn_client).plan("q", schema_hint="cn_orders(order_id TEXT, total_amount REAL)")
    prompt = cn_client.calls[0][0].content
    assert CN_DOMAIN_CONVENTIONS.strip() in prompt
    for text in (
        "SUM(cn_orders.total_amount)", "all order statuses", "cn_aftersales.status",
        "cn_orders cohort LEFT JOIN cn_aftersales", "COUNT(DISTINCT eligible cn_orders.order_id)",
        "refund_id", "NOT UNIQUE", "SUM(DISTINCT total_amount)", "NULLIF(denominator, 0)",
    ):
        assert text in prompt
    assert "cn_aftersales LEFT JOIN cn_orders" not in prompt
    legacy_client = RecordingLLM([Turn(text="SELECT 1")])
    await LLMPlanner(legacy_client).plan("q", schema_hint="orders(id INT)", domain="orders.status: 已妥投")
    assert "CN-Retail metric contract" not in legacy_client.calls[0][0].content
