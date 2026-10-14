"""Unified Schema Vocabulary and Semantic Metric Layer.

Single Source of Truth for:
1. Canonical table and column aliases across CN-Retail, Olist, and legacy e-commerce schemas.
2. Layer 1: Literal mining from low-cardinality database columns (route_status, carrier, refund_type, etc.).
3. Layer 2: Metric definition layer (Semantic Layer) defining business metric formulas, required tables, and synonyms.
4. Layer 3: Rule-based fallback synonyms for derived and high-level concepts.
"""

from __future__ import annotations

import sqlite3
from typing import Any


# ==============================================================================
# Layer 2: Semantic Metric Definitions
# ==============================================================================

METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "履约时效": {
        "entity": "logistics",
        "metric": "delivery_speed",
        "tables": ["cn_logistics"],
        "synonyms": ["履约时效", "时效", "配送时效", "送达耗时", "配送耗时", "物流速度", "送达时间", "耗时"],
        "description": "实际送达时间与发货时间的差值（小时）",
        "formula": "ROUND(AVG(julianday(actual_delivery_time) - julianday(shipped_time)) * 24, 1)",
    },
    "客单价": {
        "entity": "orders",
        "metric": "arpu",
        "tables": ["cn_orders"],
        "synonyms": ["客单价", "笔单价", "人均消费", "客单", "平均订单额"],
        "description": "销售总额除以下单买家数",
        "formula": "ROUND(SUM(total_amount) * 1.0 / NULLIF(COUNT(DISTINCT user_id), 0), 2)",
    },
    "退款率": {
        "entity": "aftersales",
        "metric": "refund_rate",
        "tables": ["cn_orders", "cn_aftersales"],
        "synonyms": ["退款率", "售后率", "退货率"],
        "description": "售后退款订单占有效订单的比例",
        "formula": "ROUND(COUNT(DISTINCT a.refund_id) * 100.0 / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)",
    },
    "货损率": {
        "entity": "aftersales",
        "metric": "damage_rate",
        "tables": ["cn_orders", "cn_aftersales"],
        "synonyms": ["货损率", "破损率", "货损", "外包装破损率", "商品破损率"],
        "description": "因包装破损发起的售后退款占订单的比例",
        "formula": "ROUND(COUNT(CASE WHEN a.refund_reason LIKE '%破损%' THEN 1 END) * 100.0 / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)",
    },
    "签收率": {
        "entity": "logistics",
        "metric": "delivery_rate",
        "tables": ["cn_logistics"],
        "synonyms": ["签收率", "妥投率", "送达率"],
        "description": "已签收包裹占全部物流包裹的比例",
        "formula": "ROUND(COUNT(CASE WHEN route_status='已签收' THEN 1 END) * 100.0 / COUNT(*), 2)",
    },
}


def match_metric(query: str) -> dict[str, Any] | None:
    """Check if query mentions any defined business metric."""
    for metric_name, defn in METRIC_DEFINITIONS.items():
        if any(syn in query for syn in defn["synonyms"]):
            return {"name": metric_name, **defn}
    return None


# ==============================================================================
# Layer 1: Literal Mining Engine
# ==============================================================================

# High-value low-cardinality columns to mine literals from
_MINED_COLUMNS: list[tuple[str, str]] = [
    ("cn_logistics", "carrier"),
    ("cn_logistics", "route_status"),
    ("cn_aftersales", "refund_type"),
    ("cn_aftersales", "refund_reason"),
    ("cn_products", "category_l1"),
    ("cn_users", "province"),
    ("cn_users", "city"),
    ("olist_order_reviews", "review_score"),
    ("olist_orders", "order_status"),
]


def mine_column_literals(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    max_distinct: int = 60,
) -> list[str]:
    """Extract distinct string literals from a low-cardinality table column."""
    if not table.replace("_", "").isalnum() or not column.replace("_", "").isalnum():
        return []
    try:
        rows = conn.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {max_distinct}'
        ).fetchall()
        return [str(r[0]) for r in rows if r[0] is not None and str(r[0]).strip()]
    except Exception:
        return []


def mine_all_literals(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Mine high-frequency literals across all registered catalog columns."""
    result: dict[str, list[str]] = {}
    for tbl, col in _MINED_COLUMNS:
        key = f"{tbl}.{col}"
        lits = mine_column_literals(conn, tbl, col)
        if lits:
            result[key] = lits
    return result


# ==============================================================================
# Layer 3: Canonical Aliases (Single Source of Truth)
# ==============================================================================

CN_TABLE_ALIASES: dict[str, list[str]] = {
    "cn_users": ["用户", "买家", "客户", "会员", "消费者", "省份", "城市", "user", "member", "buyer", "customer", "city", "province"],
    "cn_sellers": ["卖家", "商家", "店铺", "专营店", "旗舰店", "seller", "shop", "merchant"],
    "cn_products": ["商品", "产品", "货品", "类目", "品类", "一级类目", "生鲜", "水果", "海鲜", "单价", "标价", "product", "goods", "category"],
    "cn_orders": ["订单", "购买", "总金额", "销售额", "销售总额", "GMV", "下单", "支付", "渠道", "order", "sales", "gmv", "amount", "revenue"],
    "cn_order_items": ["商品明细", "订单明细", "件数", "购买数量", "成交单价", "item", "order_item", "quantity", "unit_price"],
    "cn_logistics": ["物流", "快递", "承运商", "包裹", "运单", "顺丰", "中通", "京东", "圆通", "极兔", "时效", "履约", "发货", "妥投", "签收", "延误", "超时", "网点积压", "积压", "logistics", "carrier", "delay", "shipping"],
    "cn_reviews": ["评价", "差评", "好评", "追评", "吐槽", "评分", "星级", "满意度", "客诉", "投诉", "review", "rating", "comment", "feedback"],
    "cn_aftersales": ["售后", "退款", "仅退款", "退货退款", "退款原因", "破损", "退款金额", "退款率", "货损", "aftersales", "refund", "return"],
}

OLIST_TABLE_ALIASES: dict[str, list[str]] = {
    "olist_customers": ["客户", "用户", "买家", "customer", "customers", "client", "buyer", "user", "city", "state"],
    "olist_orders": ["订单", "购买", "交付", "状态", "order", "orders", "purchase", "purchases", "delivered", "date"],
    "olist_order_items": ["商品明细", "明细", "运费", "items", "order_items", "item", "freight", "price"],
    "olist_order_payments": ["支付", "付款", "信用卡", "分期", "payments", "payment", "installments", "value", "boleto", "voucher"],
    "olist_order_reviews": ["评价", "差评", "投诉", "评分", "reviews", "review", "score", "comment", "feedback"],
    "olist_products": ["产品", "商品", "类目", "重量", "products", "product", "category", "weight"],
    "olist_sellers": ["卖家", "商家", "店铺", "sellers", "seller", "merchant"],
    "olist_geolocation": ["地理", "经纬度", "坐标", "邮编", "geolocation", "zip", "lat", "lng"],
    "product_category_name_translation": ["翻译", "类目翻译", "translation"],
}

LEGACY_TABLE_ALIASES: dict[str, list[str]] = {
    "users": ["用户", "买家", "user", "users", "customer", "customers", "buyer", "member"],
    "orders": ["订单", "购买", "order", "orders", "purchase", "purchases", "sale", "sales"],
    "order_items": ["明细", "商品明细", "item", "items", "line item"],
    "products": ["产品", "商品", "product", "products", "item", "goods", "sku"],
    "categories": ["类目", "品类", "category", "categories"],
    "payments": ["支付", "付款", "payment", "payments"],
    "refunds": ["退款", "售后", "refund", "refunds", "return"],
}

CN_COLUMN_ALIASES: dict[str, list[str]] = {
    "order_time": ["下单时间", "订单时间", "购买时间", "下单日期", "创建时间"],
    "shipped_time": ["发货时间", "出库时间", "起运时间"],
    "actual_delivery_time": ["实际送达时间", "送达时间", "妥投时间", "签收时间", "到达时间"],
    "promise_delivery_time": ["承诺送达时间", "预计送达时间", "时效承诺"],
    "carrier": ["快递公司", "承运商", "物流公司", "物流服务商", "顺丰", "中通", "京东", "圆通", "极兔"],
    "route_status": ["路由状态", "物流状态", "包裹状态", "配送状态", "网点积压", "正常在途", "已签收"],
    "category_l1": ["一级类目", "主类目", "大类", "品类"],
    "refund_type": ["退款类型", "售后方式", "仅退款", "退货退款"],
    "refund_reason": ["退款原因", "售后原因", "申请退款原因", "外包装破损", "质量问题"],
    "user_id": ["买家ID", "用户ID", "会员编号"],
    "total_amount": ["订单总额", "支付金额", "销售额", "成交额"],
}

OLIST_COLUMN_ALIASES: dict[str, list[str]] = {
    "order_purchase_timestamp": ["下单时间", "purchase_time", "order_time"],
    "order_delivered_customer_date": ["交付时间", "delivery_time", "delivered_at"],
    "order_estimated_delivery_date": ["预计交付时间", "estimated_delivery"],
    "review_score": ["评分", "星级", "score", "rating"],
    "price": ["价格", "单价", "price"],
    "freight_value": ["运费", "freight"],
}


def get_all_table_aliases() -> dict[str, list[str]]:
    """Return unified table aliases across all supported database schemas."""
    merged: dict[str, list[str]] = {}
    merged.update(LEGACY_TABLE_ALIASES)
    merged.update(OLIST_TABLE_ALIASES)
    merged.update(CN_TABLE_ALIASES)
    return merged


def get_all_column_aliases() -> dict[str, list[str]]:
    """Return unified column aliases across all supported database schemas."""
    merged: dict[str, list[str]] = {}
    merged.update(OLIST_COLUMN_ALIASES)
    merged.update(CN_COLUMN_ALIASES)
    return merged
