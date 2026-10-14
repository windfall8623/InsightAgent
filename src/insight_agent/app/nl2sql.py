"""Deterministic natural-language to SQL translation.

**Read this before judging the numbers.** This module stands in for an LLM so
the project runs with no API key. It is a transparent rule engine, not a
language model, and its accuracy is *not* representative of what an LLM would
produce. What the evaluation measures here is the *system* — retrieval,
security, routing, orchestration, and the eval harness — which is the part that
transfers unchanged when a real model is plugged in behind
:class:`insight_agent.core.llm.LLMClient`.

Translation runs in two stages so each is testable on its own:

1. :func:`parse_question` — question text to a structured :class:`QueryPlan`.
2. :func:`compile_sql` — plan to SQL.

Failures are explicit: an unrecognised question yields a plan with
``metric is None`` rather than a guess, so the agent can ask instead of
inventing a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from insight_agent.data.database import AS_OF

CITIES = ("Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou", "Chengdu")
CITIES_ZH = {
    "北京": "Beijing",
    "上海": "Shanghai",
    "广州": "Guangzhou",
    "深圳": "Shenzhen",
    "杭州": "Hangzhou",
    "成都": "Chengdu",
}
STATUSES = ("created", "paid", "shipped", "completed", "cancelled")
# "支付" alone is deliberately absent: it appears in 支付方式 and 支付成功率,
# where it means "payment", not "order status = paid". Only the explicit
# 已支付 form asserts a status.
STATUSES_ZH = {
    "已创建": "created",
    "已支付": "paid",
    "已发货": "shipped",
    "已完成": "completed",
    "已取消": "cancelled",
    "完成": "completed",
    "取消": "cancelled",
}
PAY_STATUS_ZH = {"支付成功": "success", "支付失败": "failed", "支付中": "pending"}
CHANNELS = ("app", "web", "mini_program")
CHANNELS_ZH = {
    "app": "app",
    "小程序": "mini_program",
    "网页": "web",
    "网站": "web",
    "移动端": "app",
}
MEMBER_LEVELS = ("bronze", "silver", "gold")
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
CATEGORIES_ZH = {
    "电子产品": "Electronics",
    "数码": "Electronics",
    "服装": "Clothing",
    "家居": "Home",
    "美妆": "Beauty",
    "运动": "Sports",
    "图书": "Books",
    "玩具": "Toys",
    "食品": "Grocery",
}
PAY_METHODS_ZH = {"支付宝": "alipay", "微信": "wechat", "银行卡": "card"}


@dataclass(slots=True)
class QueryPlan:
    """The structured intent extracted from a question."""

    metric: str | None = None
    dimension: str | None = None
    filters: list[str] = field(default_factory=list)
    start: str | None = None
    end: str | None = None
    time_window: str | None = None
    time_granularity: str | None = None
    # The CN compiler populates these before handing its plan back; declared so
    # the orchestrator can read them off either planner's result.
    start_time: str | None = None
    end_time: str | None = None
    city_filter: str | None = None
    province_filter: str | None = None
    category_filter: str | None = None
    carrier_filter: str | None = None
    unsupported_entities: list[str] = field(default_factory=list)
    is_out_of_bounds: bool = False
    limit: int = 50
    order_by: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def understood(self) -> bool:
        return self.metric is not None

    @property
    def tables(self) -> set[str]:
        tables = {"orders"}
        if self.dimension in {"city", "member_level"}:
            tables.add("users")
        elif self.dimension == "category":
            tables |= {"order_items", "products", "categories"}
        elif self.dimension == "product":
            tables |= {"order_items", "products"}
        elif self.dimension == "pay_method":
            tables.add("payments")
        if self.metric in {"refund_rate", "refund_amount", "refund_count"}:
            tables.add("refunds")
        elif self.metric == "quantity":
            tables |= {"order_items", "products"}
        elif self.metric == "stock":
            tables |= {"products", "categories"}

        # A filter can reference a table the metric and dimension do not.
        # "北京有多少订单" has no city dimension but still needs `users` joined,
        # because the filter says u.city.
        for condition in self.filters:
            if condition.startswith("u."):
                tables.add("users")
            elif condition.startswith("c."):
                tables |= {"order_items", "products", "categories"}
            elif condition.startswith("p."):
                tables.add("payments")
        return tables


# Metric detection. Ordered: the first pattern that matches wins, so more
# specific metrics (refund rate) must precede the generic ones they contain.
_METRIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("refund_rate", re.compile(r"退款率|退单率|refund\s*rate", re.IGNORECASE)),
    ("refund_amount", re.compile(r"退款.{0,4}金额|退货.{0,4}金额|refund\s*amount", re.IGNORECASE)),
    ("refund_count", re.compile(r"退款.{0,4}(数|笔|单)|退单数|refund\s*count", re.IGNORECASE)),
    ("pay_success_rate", re.compile(r"支付成功率|payment\s*success\s*rate", re.IGNORECASE)),
    (
        "avg_order_value",
        re.compile(r"客单价|平均订单金额|average\s+order\s+value|avg\b", re.IGNORECASE),
    ),
    (
        "amount",
        re.compile(r"销售额|销售金额|成交额|总金额|gmv|金额|sales|revenue|amount", re.IGNORECASE),
    ),
    (
        "order_count",
        re.compile(
            r"订单.{0,4}(数|量|笔)|单数|多少.{0,4}订单|多少.{0,4}笔|订单总数|总订单|"
            r"(?<!状态)订单(?:\s*|情况|分布)?$|(?:各|每|按).{0,4}订单|"
            r"order\s*count|how\s+many\s+orders",
            re.IGNORECASE,
        ),
    ),
    (
        "user_count",
        re.compile(
            r"用户.{0,3}(数|量)|买家数|客户数|多少用户|user\s*count|distinct\s+users", re.IGNORECASE
        ),
    ),
    ("quantity", re.compile(r"销量|销售数量|出货量|quantity|units\s+sold", re.IGNORECASE)),
    ("stock", re.compile(r"库存|stock|inventory", re.IGNORECASE)),
)

_DIMENSION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("city", re.compile(r"城市|地区|区域|地域|city|region", re.IGNORECASE)),
    ("category", re.compile(r"品类|类目|分类|categor", re.IGNORECASE)),
    ("product", re.compile(r"商品|产品|货品|product|item\s+name", re.IGNORECASE)),
    ("channel", re.compile(r"渠道|channel", re.IGNORECASE)),
    ("status", re.compile(r"订单状态|状态|status", re.IGNORECASE)),
    ("pay_method", re.compile(r"支付方式|付款方式|pay\s*method|payment\s*method", re.IGNORECASE)),
    ("pay_status", re.compile(r"支付状态|pay\s*status|payment\s*status", re.IGNORECASE)),
    ("member_level", re.compile(r"会员等级|会员级别|member\s*level|tier", re.IGNORECASE)),
    ("month", re.compile(r"按月|每月|月度|monthly|by\s+month", re.IGNORECASE)),
    ("weekday", re.compile(r"按(星期|周)|星期几|weekday|day\s+of\s+week", re.IGNORECASE)),
)

_DATE_RANGE = re.compile(r"(\d{4})-(\d{2})-(\d{2})(?:\s*(?:到|至|~|-|to)\s*(\d{4}-\d{2}-\d{2}))?")
# Chinese dates: "2024年3月", "2024年3月15日", and ranges like "2024年1月到2024年3月".
_DATE_RANGE_ZH = re.compile(
    r"(\d{4})\s*年\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?"
    r"(?:\s*(?:到|至|~|-)\s*(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?)?"
)
_LAST_N_DAYS = re.compile(r"(?:最近|近|过去)\s*(\d+)\s*天|last\s*(\d+)\s*days")
_THIS_YEAR = re.compile(r"(今年|本年度|this\s+year)")
_LAST_YEAR = re.compile(r"(去年|上一年|last\s+year)")
_TOP_N = re.compile(r"top\s*(\d+)|前\s*(\d+)\s*(?:名|个|大)")


# Which SQL column each dimension groups by, used to detect a filter that has
# already pinned the grouping column to a single value.
_DIMENSION_COLUMN: dict[str, str] = {
    "city": "u.city",
    "category": "c.name",
    "product": "pr.name",
    "channel": "o.channel",
    "status": "o.status",
    "member_level": "u.member_level",
    "pay_method": "p.pay_method",
}

# A "total" cue with no explicit grouping cue means the user wants one number,
# not a breakdown: "商品总库存" is all stock, not stock per product.
_TOTAL_CUE = re.compile(r"总(共|计|量)?|一共|合计")
_GROUPING_CUE = re.compile(r"各|每|分|按|top|前\s*\d+|排行|排名|per\s|by\b", re.IGNORECASE)


def parse_question(
    question: str,
    *,
    as_of: date = AS_OF,
    standing_filters: dict[str, str] | None = None,
) -> QueryPlan:
    """Extract a :class:`QueryPlan` from a natural-language question."""

    lowered = question.lower()
    plan = QueryPlan()

    metric_span: tuple[int, int] | None = None
    for name, pattern in _METRIC_PATTERNS:
        match = pattern.search(lowered)
        if match:
            plan.metric = name
            metric_span = match.span()
            break

    for name, pattern in _DIMENSION_PATTERNS:
        if pattern.search(lowered):
            plan.dimension = name
            break

    # A literal category name ("电子产品") contains "产品", which the product
    # dimension would otherwise claim. An explicit category value is stronger
    # evidence than the generic dimension keyword.
    if any(zh in question for zh in CATEGORIES_ZH):
        plan.dimension = "category"

    # "支付成功率和支付失败率分别是多少" asks for the breakdown across payment
    # statuses, not a single success rate.
    group_by_pay_status = ("成功" in question and "失败" in question) or "支付状态" in question
    if group_by_pay_status:
        plan.dimension = "pay_status"

    if _TOTAL_CUE.search(question) and not _GROUPING_CUE.search(question) and plan.dimension:
        plan.notes.append(f"dimension {plan.dimension} suppressed by total cue")
        plan.dimension = None

    # Filters are read from the question with the metric phrase masked out.
    # Without this, "支付成功率" also matches the 支付 -> status='paid' filter,
    # which silently restricts the answer to paid orders.
    scrubbed = question
    if metric_span is not None:
        lo, hi = metric_span
        scrubbed = question[:lo] + "　" * (hi - lo) + question[hi:]
    _extract_filters(scrubbed, scrubbed.lower(), plan)

    # The status words are also picked up as literal filters. When the question
    # enumerates several statuses, it wants the breakdown, so grouping wins:
    # filtering on the grouped column would return a single row.
    if group_by_pay_status:
        plan.filters = [f for f in plan.filters if not f.startswith("p.pay_status")]

    start, end = _extract_date_range(lowered, as_of=as_of)
    plan.start, plan.end = start, end

    top = _TOP_N.search(lowered)
    if top:
        plan.limit = int(top.group(1) or top.group(2))
        plan.order_by = "DESC"
    elif plan.dimension:
        plan.order_by = "DESC"

    # Grouping by a column the filters already pin to one value produces a
    # single row with an extra column, not a useful breakdown.
    if plan.dimension:
        column = _DIMENSION_COLUMN.get(plan.dimension)
        if column and any(f.startswith(f"{column} =") for f in plan.filters):
            plan.notes.append(f"dimension {plan.dimension} dropped: filtered to a single value")
            plan.dimension = None

    for column, raw in (standing_filters or {}).items():
        col_name = _DIMENSION_COLUMN.get(column, column)
        val_str = str(raw).strip()
        if not (val_str.startswith("'") or val_str.startswith('"') or val_str.replace(".", "", 1).isdigit()):
            val_str = f"'{val_str}'"
        condition = f"{col_name} = {val_str}"
        if condition not in plan.filters and not any(f.startswith(f"{col_name} =") for f in plan.filters):
            plan.filters.append(condition)
            plan.notes.append(f"standing filter: {condition}")

    return plan


def _literal(token: str, text: str) -> bool:
    """Match an ASCII token inside text that may contain CJK.

    ``\\b`` is useless here: Python treats CJK characters as word characters, so
    ``\\bgold\\b`` never matches in "gold会员". Use explicit identifier boundaries.
    """

    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
            text,
            re.IGNORECASE,
        )
    )


def _extract_filters(question: str, lowered: str, plan: QueryPlan) -> None:
    """Collect literal filter values mentioned in the question."""

    def add(column: str, value: str) -> None:
        condition = f"{column} = '{value}'"
        if condition not in plan.filters:
            plan.filters.append(condition)

    for zh, en in CITIES_ZH.items():
        if zh in question:
            add("u.city", en)
    for city in CITIES:
        if _literal(city, question):
            add("u.city", city)

    for zh, en in STATUSES_ZH.items():
        if zh in question:
            add("o.status", en)
    for status in STATUSES:
        if _literal(status, question):
            add("o.status", status)

    for zh, en in PAY_STATUS_ZH.items():
        if zh in question:
            add("p.pay_status", en)

    for zh, en in CHANNELS_ZH.items():
        if zh in question:
            add("o.channel", en)
    for zh, en in CATEGORIES_ZH.items():
        if zh in question:
            add("c.name", en)
    for category in CATEGORIES:
        if _literal(category, question):
            add("c.name", category)

    for zh, en in PAY_METHODS_ZH.items():
        if zh in question:
            add("p.pay_method", en)
    for level in MEMBER_LEVELS:
        if _literal(level, lowered):
            add("u.member_level", level)


def _end_of_month(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def _extract_date_range(lowered: str, *, as_of: date) -> tuple[str | None, str | None]:
    explicit = _DATE_RANGE.search(lowered)
    if explicit:
        start = f"{explicit.group(1)}-{explicit.group(2)}-{explicit.group(3)}"
        end = explicit.group(4) or start
        return start, end

    zh = _DATE_RANGE_ZH.search(lowered)
    if zh:
        year, month, day = int(zh.group(1)), int(zh.group(2)), zh.group(3)
        start_date = date(year, month, int(day) if day else 1)
        end_date = start_date if day else _end_of_month(year, month)
        # An explicit end month was given; honour it.
        if zh.group(5):
            end_year = int(zh.group(4)) if zh.group(4) else year
            end_month, end_day = int(zh.group(5)), zh.group(6)
            end_date = (
                date(end_year, end_month, int(end_day))
                if end_day
                else _end_of_month(end_year, end_month)
            )
        return start_date.isoformat(), end_date.isoformat()

    last_n = _LAST_N_DAYS.search(lowered)
    if last_n:
        days = int(last_n.group(1) or last_n.group(2))
        window_start = as_of - timedelta(days=days - 1)
        return window_start.isoformat(), as_of.isoformat()

    if _LAST_YEAR.search(lowered):
        return f"{as_of.year - 1}-01-01", f"{as_of.year - 1}-12-31"
    if _THIS_YEAR.search(lowered):
        return f"{as_of.year}-01-01", f"{as_of.year}-12-31"
    return None, None


# --------------------------------------------------------------------------
# SQL compilation
# --------------------------------------------------------------------------

_METRIC_SELECT: dict[str, str] = {
    "amount": "SUM(o.total_amount)",
    "order_count": "COUNT(*)",
    "user_count": "COUNT(DISTINCT o.user_id)",
    "avg_order_value": "AVG(o.total_amount)",
    "quantity": "SUM(oi.quantity)",
}

_DIMENSION_SELECT: dict[str, tuple[str, str]] = {
    "city": ("u.city", "city"),
    "category": ("c.name", "category"),
    "product": ("pr.name", "product"),
    "channel": ("o.channel", "channel"),
    "status": ("o.status", "status"),
    "member_level": ("u.member_level", "member_level"),
    "pay_method": ("p.pay_method", "pay_method"),
    "pay_status": ("p.pay_status", "pay_status"),
    "month": ("strftime('%Y-%m', o.order_date)", "month"),
    "weekday": ("o.order_date", "day"),
}


def compile_sql(plan: QueryPlan) -> str:
    """Render a :class:`QueryPlan` as a read-only SELECT."""

    if plan.metric is None:
        raise ValueError("cannot compile SQL for an unparsed question")

    # Inventory has no time axis, so a date dimension or range cannot apply.
    if plan.metric == "stock":
        if plan.dimension not in (None, "category", "product"):
            plan.notes.append(f"dimension {plan.dimension} dropped: stock has no time axis")
            plan.dimension = None
        plan.start = plan.end = None
        plan.filters = [f for f in plan.filters if not f.startswith("o.")]

    where = _build_where(plan)
    dimension_expr = _DIMENSION_SELECT.get(plan.dimension, ("", ""))[0] if plan.dimension else None

    # Inventory is a property of products, not of orders; starting from orders
    # would both multiply rows and restrict to products that happened to sell.
    if plan.metric == "stock":
        select = "SUM(pr.stock) AS total_stock"
        alias = "total_stock"
        from_clause = "products pr"
        joins = (
            "\nJOIN categories c ON c.id = pr.category_id" if plan.dimension == "category" else ""
        )
    else:
        select, alias = _metric_expression(plan)
        if _payments_only(plan):
            from_clause = "payments p"
            joins = ""
        else:
            from_clause = "orders o"
            joins = _build_joins(plan)

    if dimension_expr is None:
        sql = f"SELECT {select}\nFROM {from_clause}"
    else:
        sql = f"SELECT {dimension_expr} AS dimension,\n       {select}\nFROM {from_clause}"

    sql += joins

    if where:
        sql += f"\nWHERE {where}"

    if dimension_expr is not None:
        sql += "\nGROUP BY dimension"
        # Order by the real output alias; hardcoding "metric" breaks for
        # refund/stock/payment metrics, which use their own alias.
        direction = "DESC" if plan.order_by and "DESC" in plan.order_by.upper() else "ASC"
        sql += f"\nORDER BY {alias} {direction}"

    sql += f"\nLIMIT {plan.limit}"
    return sql


def _metric_expression(plan: QueryPlan) -> tuple[str, str]:
    """Return the aggregate expression and the alias it is selected under."""

    if plan.metric == "refund_rate":
        return (
            "ROUND(COUNT(r.id) * 100.0 / NULLIF(COUNT(DISTINCT o.id), 0), 2) AS refund_rate_pct",
            "refund_rate_pct",
        )
    if plan.metric == "refund_amount":
        return "ROUND(SUM(r.amount), 2) AS refund_amount", "refund_amount"
    if plan.metric == "refund_count":
        return "COUNT(r.id) AS refund_count", "refund_count"
    if plan.metric == "pay_success_rate":
        # Splitting by payment status and asking for the success *rate* per
        # status is degenerate (each bucket is 100% or 0%); users asking that
        # question want the counts per status.
        if plan.dimension == "pay_status":
            return "COUNT(*) AS metric", "metric"
        return (
            "ROUND(SUM(CASE WHEN p.pay_status = 'success' THEN 1 ELSE 0 END) * 100.0"
            " / NULLIF(COUNT(*), 0), 2) AS pay_success_rate_pct",
            "pay_success_rate_pct",
        )
    if plan.metric == "amount" and "order_items" in plan.tables:
        # Summing an order total after joining through order_items repeats the
        # order once per line item (measured: 148.8M vs the true 49.4M). Sum at
        # the grain the join actually produces instead.
        return "ROUND(SUM(oi.unit_price * oi.quantity), 2) AS metric", "metric"
    aggregate = _METRIC_SELECT.get(plan.metric or "")
    if aggregate is None:
        raise ValueError(f"unsupported metric: {plan.metric!r}")
    return f"ROUND({aggregate}, 2) AS metric", "metric"


def _payments_only(plan: QueryPlan) -> bool:
    """True when the question is about payments alone, not orders.

    "支付宝支付成功了多少笔" filters on payments but nothing references orders,
    so starting from ``orders`` would needlessly join the whole fact table.
    """

    if plan.metric != "order_count" or plan.dimension is not None:
        return False
    return any(f.startswith("p.") for f in plan.filters)


def _build_joins(plan: QueryPlan) -> str:
    """Emit only the joins the plan actually needs.

    Over-joining would multiply rows and silently inflate ``COUNT(*)``, which is
    the classic NL2SQL correctness bug, so joins are derived from the plan rather
    than from a fixed template.
    """

    lines: list[str] = []
    tables = plan.tables

    # Keyed off the derived table set, not the dimension: a filter on u.city
    # requires the join even when nothing groups by city.
    if "users" in tables:
        lines.append("\nJOIN users u ON u.id = o.user_id")

    if "order_items" in tables:
        lines.append("\nJOIN order_items oi ON oi.order_id = o.id")
    if "products" in tables:
        lines.append("\nJOIN products pr ON pr.id = oi.product_id")
    if "categories" in tables:
        lines.append("\nJOIN categories c ON c.id = pr.category_id")

    # LEFT JOIN so unmatched rows (e.g. no refund) are still counted.
    if plan.metric in {"refund_rate", "refund_amount", "refund_count"}:
        lines.append("\nLEFT JOIN refunds r ON r.order_id = o.id")
    if plan.dimension == "pay_method" or plan.metric == "pay_success_rate":
        lines.append("\nJOIN payments p ON p.order_id = o.id")

    return "".join(lines)


def _build_where(plan: QueryPlan) -> str:
    clauses: list[str] = []
    date_col = "p" if _payments_only(plan) else "o"
    if plan.start:
        clauses.append(f"{date_col}.order_date >= '{plan.start}'")
    if plan.end:
        clauses.append(f"{date_col}.order_date <= '{plan.end}'")
    clauses.extend(plan.filters)
    return " AND ".join(clauses)
