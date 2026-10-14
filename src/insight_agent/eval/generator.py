"""Generated evaluation dataset.

Why this exists
---------------
The original 51 hand-written cases were, in effect, the development set: every
defect fixed in the rules engine was found by a failing case from that set, so a
100% on it measures memorisation, not generalisation. Reporting that number
without saying so would be the same mistake the rest of this project exists to
avoid.

So the dataset is generated instead, and crucially each **intent** is rendered
into several different **phrasings**. A planner that pattern-matches on the
wording it was tuned against fails the paraphrases; one that generalises does
not. That difference is the whole measurement.

Two independent renderers
-------------------------
``_gold_sql`` and ``_question`` both read the same :class:`Intent`, but neither
calls the other. The SQL is written from the semantic fields directly, not by
running :func:`insight_agent.app.nl2sql.compile_sql` — otherwise the dataset
would inherit the very bug it exists to detect.

The 51 original cases are kept and marked ``source="original"``; everything here
is ``source="generated"`` and is never used to tune anything.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

NL = chr(10)

# --- semantic vocabulary ---------------------------------------------------

CITIES = ("Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou", "Chengdu")
CITY_ZH = {
    "Beijing": "北京",
    "Shanghai": "上海",
    "Guangzhou": "广州",
    "Shenzhen": "深圳",
    "Hangzhou": "杭州",
    "Chengdu": "成都",
}
STATUSES = ("completed", "paid", "shipped", "cancelled")
STATUS_ZH = {
    "completed": "已完成",
    "paid": "已支付",
    "shipped": "已发货",
    "cancelled": "已取消",
}
CHANNELS = ("app", "web", "mini_program")
CHANNEL_ZH = {"app": "APP端", "web": "网页端", "mini_program": "小程序端"}
LEVELS = ("bronze", "silver", "gold")
LEVEL_ZH = {"bronze": "铜牌", "silver": "银牌", "gold": "金牌"}
CATEGORIES = (
    "Electronics",
    "Clothing",
    "Home",
    "Beauty",
    "Sports",
    "Books",
    "Toys",
    "Grocery",
)
CATEGORY_ZH = {
    "Electronics": "电子产品",
    "Clothing": "服装",
    "Home": "家居",
    "Beauty": "美妆",
    "Sports": "运动",
    "Books": "图书",
    "Toys": "玩具",
    "Grocery": "食品",
}
PAY_METHODS = ("alipay", "wechat", "card")
PAY_ZH = {"alipay": "支付宝", "wechat": "微信", "card": "银行卡"}

METRIC_ZH = {
    "amount": "销售额",
    "order_count": "订单数",
    "user_count": "用户数",
    "avg_order_value": "客单价",
    "quantity": "销量",
    "stock": "库存",
    "refund_rate": "退款率",
    "refund_amount": "退款金额",
    "refund_count": "退款笔数",
    "pay_success_rate": "支付成功率",
}

DIMENSION_ZH = {
    "city": "城市",
    "category": "品类",
    "channel": "渠道",
    "status": "订单状态",
    "member_level": "会员等级",
    "pay_method": "支付方式",
    "month": "月份",
}


@dataclass(frozen=True, slots=True)
class Intent:
    """One semantic question, independent of how it is worded."""

    metric: str
    dimension: str | None = None
    city: str | None = None
    status: str | None = None
    channel: str | None = None
    level: str | None = None
    category: str | None = None
    pay_method: str | None = None
    start: str | None = None
    end: str | None = None
    time_phrase: str | None = None
    needs_order_items: bool = field(default=False)


# --- gold SQL, written from the semantic fields ---------------------------

_DIM_JOIN = {
    "city": ("u.city", "JOIN users u ON u.id = o.user_id"),
    "status": ("o.status", None),
    "channel": ("o.channel", None),
    "member_level": ("u.member_level", "JOIN users u ON u.id = o.user_id"),
    "month": ("strftime('%Y-%m', o.order_date)", None),
    "category": (
        "c.name",
        "JOIN order_items oi ON oi.order_id = o.id "
        "JOIN products p ON p.id = oi.product_id "
        "JOIN categories c ON c.id = p.category_id",
    ),
    "pay_method": (
        "p.pay_method",
        "JOIN payments p ON p.order_id = o.id",
    ),
}

_METRIC_SQL = {
    "amount": "ROUND(SUM(o.total_amount), 2)",
    "order_count": "COUNT(*)",
    "user_count": "COUNT(DISTINCT o.user_id)",
    "avg_order_value": "ROUND(AVG(o.total_amount), 2)",
    "quantity": "ROUND(SUM(oi.quantity), 2)",
    "stock": "SUM(p.stock)",
    "refund_rate": ("ROUND(COUNT(r.id) * 100.0 / NULLIF(COUNT(DISTINCT o.id), 0), 2)"),
    "refund_amount": "ROUND(SUM(r.amount), 2)",
    "refund_count": "COUNT(r.id)",
    "pay_success_rate": (
        "ROUND(SUM(CASE WHEN p.pay_status = 'success' THEN 1 ELSE 0 END) * 100.0 "
        "/ NULLIF(COUNT(*), 0), 2)"
    ),
}


def _gold_sql(intent: Intent) -> str:
    """Render the reference query for an intent.

    Independent of the planner under test by construction.
    """

    if intent.metric == "stock":
        base = "FROM products p"
        if intent.category:
            base += " JOIN categories c ON c.id = p.category_id"
        stock_filter = [f"c.name = '{intent.category}'"] if intent.category else []
        label = intent.dimension or "total_stock_label"
        sql = f"SELECT {label} AS dimension, SUM(p.stock) AS metric{NL}{base}"

        # Stock has no time axis.
        if stock_filter:
            sql += "\nWHERE " + " AND ".join(stock_filter)
        return sql + "\nGROUP BY dimension"

    dim_expr, join = _DIM_JOIN.get(intent.dimension or "", (None, None))
    from_clause = "FROM orders o"
    if intent.needs_order_items and not join:
        join = "JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id"
    if intent.metric == "quantity" and "order_items" not in (join or ""):
        join = (join or "") + " JOIN order_items oi ON oi.order_id = o.id"
    if intent.metric == "pay_success_rate" and "payments" not in (join or ""):
        join = (join or "") + " JOIN payments p ON p.order_id = o.id"
    if intent.metric in ("refund_rate", "refund_amount", "refund_count"):
        join = (join or "") + " LEFT JOIN refunds r ON r.order_id = o.id"

    sql = f"FROM orders o{chr(10)}{join}" if join else from_clause

    where: list[str] = []
    if intent.start:
        where.append(f"o.order_date >= '{intent.start}'")
    if intent.end:
        where.append(f"o.order_date <= '{intent.end}'")
    if intent.city:
        where.append(f"u.city = '{intent.city}'")
        if "users u" not in (join or ""):
            sql = f"FROM orders o{chr(10)}JOIN users u ON u.id = o.user_id"
    if intent.status:
        where.append(f"o.status = '{intent.status}'")
    if intent.channel:
        where.append(f"o.channel = '{intent.channel}'")
    if intent.level:
        where.append(f"u.member_level = '{intent.level}'")
        if "users u" not in (join or ""):
            sql = f"FROM orders o{chr(10)}JOIN users u ON u.id = o.user_id"
    if intent.category:
        where.append(f"c.name = '{intent.category}'")
    if intent.pay_method:
        where.append(f"p.pay_method = '{intent.pay_method}'")
        if "payments p" not in (join or ""):
            sql = f"FROM orders o{chr(10)}JOIN payments p ON p.order_id = o.id"

    # Amount aggregated at line-item grain once the join fans out.
    metric_sql = _METRIC_SQL[intent.metric]
    if intent.metric == "amount" and "order_items" in sql:
        metric_sql = "ROUND(SUM(oi.unit_price * oi.quantity), 2)"

    if dim_expr:
        sql = f"SELECT {dim_expr} AS dimension, {metric_sql} AS metric\n{sql}"
    else:
        sql = f"SELECT {metric_sql} AS metric\n{sql}"

    if where:
        sql += "\nWHERE " + " AND ".join(where)
    if dim_expr:
        sql += "\nGROUP BY dimension\nORDER BY metric DESC"
    return sql + "\nLIMIT 100"


# --- natural language, written independently of the SQL -------------------

_OPENERS = ("", "帮我查一下", "麻烦问下", "我想知道", "算一下", "统计一下", "查询")


def _question(intent: Intent, variant: int) -> str:
    """Render one phrasing of an intent.

    Different variants word the same request differently on purpose: a planner
    that matches the exact strings it was tuned against will drop most of these.
    """

    opener = _OPENERS[variant % len(_OPENERS)]
    metric = METRIC_ZH[intent.metric]

    dim = ""
    if intent.dimension:
        dim = DIMENSION_ZH[intent.dimension]

    # Synonym sets keyed by variant, so the same intent reads differently.
    # Three phrasings of the same metric. Guard against "订单数数": several
    # metric labels already end in 数.
    if variant % 3 == 0:
        metric_phrase = metric
    elif variant % 3 == 1:
        metric_phrase = f"{metric}情况" if not metric.endswith("数") else metric
    else:
        metric_phrase = f"{metric}是多少"

    if dim:
        lead = {0: f"各{dim}", 1: f"按{dim}拆", 2: f"分{dim}看"}.get(variant % 3, f"各{dim}")
        shape = f"{lead}的{metric_phrase}"
    else:
        shape = f"{metric_phrase}"

    filters: list[str] = []
    if intent.city:
        filters.append(CITY_ZH[intent.city])
    if intent.status:
        filters.append(STATUS_ZH[intent.status] + "的")
    if intent.channel:
        filters.append(CHANNEL_ZH[intent.channel] + "的")
    if intent.level:
        filters.append(LEVEL_ZH[intent.level] + "用户的")
    if intent.category:
        filters.append(CATEGORY_ZH[intent.category] + "的")
    if intent.pay_method:
        filters.append(PAY_ZH[intent.pay_method] + "的")
    if filters:
        shape = "".join(filters) + shape

    if intent.time_phrase:
        shape = f"{intent.time_phrase}的{shape}"

    return f"{opener}{shape}" if opener else shape


def _required_tables(intent: Intent) -> list[str]:
    tables = ["orders"] if intent.metric != "stock" else ["products"]
    if intent.city or intent.level:
        tables.append("users")
    if intent.needs_order_items or intent.metric == "quantity" or intent.dimension == "category":
        tables += ["order_items", "products"]
    if intent.pay_method or intent.metric == "pay_success_rate":
        tables.append("payments")
    if intent.metric.startswith("refund"):
        tables.append("refunds")
    return sorted(set(tables))


# --- intent enumeration ---------------------------------------------------


def _intents() -> list[Intent]:
    out: list[Intent] = []
    months = [
        ("2024-01-01", "2024-01-31", "一月"),
        ("2024-03-01", "2024-03-31", "三月"),
        ("2024-06-01", "2024-06-30", "六月"),
        ("2024-09-01", "2024-09-30", "九月"),
    ]

    # Every metric on its own.
    for metric in METRIC_ZH:
        out.append(Intent(metric=metric))

    # Metric x dimension.
    for metric in ("amount", "order_count", "avg_order_value", "quantity", "refund_rate"):
        for dim in ("city", "category", "channel", "status", "month", "member_level"):
            needs_items = dim == "category" or metric == "quantity"
            out.append(Intent(metric=metric, dimension=dim, needs_order_items=needs_items))

    # Filters.
    for city in CITIES:
        for metric in ("amount", "order_count"):
            out.append(Intent(metric=metric, city=city))
    for status in STATUSES:
        out.append(Intent(metric="order_count", status=status))
        out.append(Intent(metric="amount", status=status))
    for channel in CHANNELS:
        out.append(Intent(metric="order_count", channel=channel))
    for level in LEVELS:
        out.append(Intent(metric="order_count", level=level))
    for category in CATEGORIES:
        out.append(Intent(metric="amount", category=category, needs_order_items=True))
        out.append(Intent(metric="quantity", category=category, needs_order_items=True))
    for method in PAY_METHODS:
        out.append(Intent(metric="pay_success_rate", pay_method=method))
        out.append(Intent(metric="order_count", pay_method=method))

    # Time windows.
    for start, end, zh in months:
        out.append(Intent(metric="amount", start=start, end=end, time_phrase=zh))
        out.append(Intent(metric="order_count", start=start, end=end, time_phrase=zh))
    for days, phrase in ((7, "最近7天"), (30, "最近30天"), (90, "最近90天")):
        end = "2024-12-31"
        out.append(Intent(metric="amount", start=_minus(days), end=end, time_phrase=phrase))

    # Combinations: a filter and a breakdown together.
    for city in CITIES[:3]:
        out.append(Intent(metric="amount", dimension="channel", city=city))
        out.append(Intent(metric="order_count", city=city, status="completed"))
    for category in CATEGORIES[:3]:
        out.append(
            Intent(metric="quantity", category=category, dimension="month", needs_order_items=True)
        )
    return out


def _minus(days: int) -> str:
    from datetime import date, timedelta

    return (date(2024, 12, 31) - timedelta(days=days - 1)).isoformat()


def generate(variants_per_intent: int = 4) -> list[dict[str, Any]]:
    """Build the generated dataset.

    Each intent is rendered into ``variants_per_intent`` distinct phrasings;
    the SQL is identical across them, so a planner must handle wording rather
    than pattern-match one sentence.
    """

    rng = random.Random(20261004)
    cases: list[dict[str, Any]] = []

    # Never repeat a question the development set already contains: a duplicate
    # would let the original phrasing stand in for a paraphrase of it.
    from insight_agent.eval.dataset import CASES as DEVELOPMENT_CASES

    seen_questions: set[str] = {str(c["question"]) for c in DEVELOPMENT_CASES}

    for index, intent in enumerate(_intents()):
        for variant in range(variants_per_intent):
            question = _question(intent, variant)
            if question in seen_questions:
                continue
            seen_questions.add(question)
            cases.append(
                {
                    "id": f"gen-{index:03d}-{variant}",
                    "question": question,
                    "gold_sql": _gold_sql(intent),
                    "requires_tables": _required_tables(intent),
                    "category": "aggregate" if not intent.dimension else "breakdown",
                    "difficulty": "hard" if intent.dimension and intent.city else "medium",
                    "expectation": "rows",
                    "source": "generated",
                }
            )
    rng.shuffle(cases)
    return cases


if __name__ == "__main__":
    from insight_agent.eval.dataset import load_cases as original

    gen = generate()
    counts: dict[str, int] = {}
    for c in gen:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    print(f"original : {len(original())}")
    print(f"generated: {len(gen)}")
    print(f"total    : {len(original()) + len(gen)}")
    print(counts)
    for c in gen[:5]:
        print("---")
        print(c["question"])
        print(c["gold_sql"].replace(chr(10), " ")[:150])
