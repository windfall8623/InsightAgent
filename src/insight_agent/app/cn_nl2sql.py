"""Natural Language to SQL translator for CN-Retail Data Warehouse.

Translates domestic e-commerce queries into read-only SQL over:
cn_users, cn_sellers, cn_products, cn_orders, cn_order_items,
cn_logistics, cn_reviews, cn_aftersales.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any


@dataclass(slots=True)
class CNQueryPlan:
    metric: str | None = None
    dimension: str | None = None
    entity: str = "orders"  # orders | users | products | sellers | logistics | reviews | aftersales
    start_time: str | None = None
    end_time: str | None = None
    time_window: str | None = None
    time_granularity: str | None = None
    is_out_of_bounds: bool = False
    out_of_bounds_msg: str | None = None
    is_comparison: bool = False
    comparison_type: str | None = None
    city_filter: str | None = None
    province_filter: str | None = None
    carrier_filter: str | None = None
    carriers_filter: list[str] | None = None
    unsupported_entities: list[str] = field(default_factory=list)
    category_filter: str | None = None
    category_l2_filter: str | None = None
    channel_filter: str | None = None
    member_level_filter: str | None = None
    status_filter: str | None = None
    refund_type_filter: str | None = None
    refund_reason_filter: str | None = None
    route_status_filter: str | None = None
    rating_filter: int | None = None
    min_amount: float | None = None
    max_amount: float | None = None
    seller_id_filter: str | None = None
    tags_filter: str | None = None
    limit: int = 50
    order_by: str = "DESC"
    is_cn_retail: bool = True
    custom_sql: str | None = None


DEFAULT_DATASET_START = "2024-01-01"
DEFAULT_DATASET_END = "2024-09-30"
DATASET_ANCHOR = date(2024, 9, 30)


def resolve_relative_time_window(q: str, anchor: date = DATASET_ANCHOR) -> tuple[str | None, str | None, str | None]:
    """Resolve relative time phrases (本月/上月/本季度/上季度/最近N天) using the dataset anchor."""
    # 1. 本月 / 当月
    if any(k in q for k in ("本月", "当月", "这个月")):
        last_d = calendar.monthrange(anchor.year, anchor.month)[1]
        return (f"{anchor.year}-{anchor.month:02d}-01 00:00:00", f"{anchor.year}-{anchor.month:02d}-{last_d:02d} 23:59:59", "本月")

    # 2. 上月 / 上个月
    if any(k in q for k in ("上月", "上个月", "前一个月")):
        prev_y = anchor.year if anchor.month > 1 else anchor.year - 1
        prev_m = anchor.month - 1 if anchor.month > 1 else 12
        last_d = calendar.monthrange(prev_y, prev_m)[1]
        return (f"{prev_y}-{prev_m:02d}-01 00:00:00", f"{prev_y}-{prev_m:02d}-{last_d:02d} 23:59:59", "上月")

    # 3. 本季度 / 当季 (for anchor 2024-09-30, Q3 is 07-01 to 09-30)
    if any(k in q for k in ("本季度", "当季", "这季度")):
        return (f"{anchor.year}-07-01 00:00:00", f"{anchor.year}-09-30 23:59:59", "本季度(Q3)")

    # 4. 上季度
    if any(k in q for k in ("上季度", "上季")):
        return (f"{anchor.year}-04-01 00:00:00", f"{anchor.year}-06-30 23:59:59", "上季度(Q2)")

    # 5. 上半年 / 下半年
    if "上半年" in q:
        return (f"{anchor.year}-01-01 00:00:00", f"{anchor.year}-06-30 23:59:59", "上半年")
    if "下半年" in q:
        return (f"{anchor.year}-07-01 00:00:00", f"{anchor.year}-09-30 23:59:59", "下半年")

    # 6. 最近N天 / 近N天
    last_days = re.search(r"(?:最近|近)\s*(\d+)\s*天", q)
    if last_days:
        days = int(last_days.group(1))
        start_d = anchor - timedelta(days=days)
        return (f"{start_d.isoformat()} 00:00:00", f"{anchor.isoformat()} 23:59:59", f"最近{days}天")

    # 7. 最近N周 / 近N周
    last_weeks = re.search(r"(?:最近|近)\s*(\d+)\s*周", q)
    if last_weeks:
        weeks = int(last_weeks.group(1))
        start_d = anchor - timedelta(days=weeks * 7)
        return (f"{start_d.isoformat()} 00:00:00", f"{anchor.isoformat()} 23:59:59", f"最近{weeks}周")

    # 8. 近一个月 / 最近一个月
    if "近一个月" in q or "最近一个月" in q:
        start_d = anchor - timedelta(days=30)
        return (f"{start_d.isoformat()} 00:00:00", f"{anchor.isoformat()} 23:59:59", "最近一个月")

    return (None, None, None)


def _is_city_grouping(q: str) -> bool:
    if any(d in q for d in ("各城市", "各个城市", "不同城市", "按城市", "分城市", "哪些城市", "城市排名", "城市分布", "城市维度", "各市")):
        return True
    return any(r in q for r in ("前", "top", "排名", "最高", "最低", "排行", "分别")) and any(c in q for c in ("城市", "各市"))


def _is_province_grouping(q: str) -> bool:
    if any(d in q for d in ("各省份", "各个省份", "不同省份", "按省份", "分省份", "哪些省份", "省份排名", "省份分布", "省份维度", "各省")):
        return True
    return any(r in q for r in ("前", "top", "排名", "最高", "最低", "排行", "分别")) and any(p in q for p in ("省份", "各省"))


def _is_category_grouping(q: str) -> bool:
    if any(d in q for d in ("各品类", "各个品类", "按品类", "分类目", "各类目", "各个类目", "一级类目", "品类排名", "类目排名", "各分类")):
        return True
    return any(r in q for r in ("前", "top", "排名", "最高", "最低", "排行", "分别")) and any(c in q for c in ("品类", "类目"))


def _is_channel_grouping(q: str) -> bool:
    return any(d in q for d in ("各渠道", "各个渠道", "按渠道", "分渠道", "渠道分布", "渠道排名"))


def _is_carrier_grouping(q: str) -> bool:
    return any(d in q for d in ("各承运商", "各快递", "各物流", "按承运商", "按快递", "快递排名", "承运商排名", "快递分布"))


def parse_cn_question(
    question: str,
    standing_filters: dict[str, Any] | None = None,
) -> CNQueryPlan | None:
    """Parse a Chinese retail business question into a structured CNQueryPlan."""
    q = question.strip()
    plan = CNQueryPlan()

    # 0. Time window & date range extraction
    # A. Relative time phrases (anchored at 2024-09-30)
    rel_start, rel_end, rel_label = resolve_relative_time_window(q)
    if rel_start and rel_end:
        plan.start_time = rel_start
        plan.end_time = rel_end
        plan.time_window = rel_label

    # B. Large E-commerce promotions
    if not plan.start_time:
        if "618" in q:
            plan.start_time = "2024-06-01 00:00:00"
            plan.end_time = "2024-06-18 23:59:59"
            plan.time_window = "618大促"
        elif "年货节" in q or "春节" in q:
            plan.start_time = "2024-01-15 00:00:00"
            plan.end_time = "2024-02-05 23:59:59"
            plan.time_window = "年货节"
        elif "双十一" in q or "双11" in q:
            plan.start_time = "2024-10-20 00:00:00"
            plan.end_time = "2024-11-11 23:59:59"
            plan.time_window = "双11"

    # C. Specific months (e.g. 2024年3月, 3月份, 8月, 2024年1月至3月, 1-3月)
    if not plan.start_time:
        range_m = re.search(r"(?:2024年)?([1-9]|1[0-2])月(?:\s*(?:至|到|~|-)\s*(?:2024年)?([1-9]|1[0-2])月)", q)
        if range_m:
            m1, m2 = int(range_m.group(1)), int(range_m.group(2))
            plan.start_time = f"2024-{m1:02d}-01 00:00:00"
            last_day = 31 if m2 in (1, 3, 5, 7, 8, 10, 12) else (29 if m2 == 2 else 30)
            plan.end_time = f"2024-{m2:02d}-{last_day:02d} 23:59:59"
            plan.time_window = f"{m1}月-{m2}月"
        else:
            single_m = re.search(r"(?:2024年)?([1-9]|1[0-2])月(?:份)?", q)
            if single_m and not any(k in q for k in ("各月", "每月", "按月", "分月", "月度走势", "各月份")):
                m = int(single_m.group(1))
                plan.start_time = f"2024-{m:02d}-01 00:00:00"
                last_day = 31 if m in (1, 3, 5, 7, 8, 10, 12) else (29 if m == 2 else 30)
                plan.end_time = f"2024-{m:02d}-{last_day:02d} 23:59:59"
                plan.time_window = f"2024-{m:02d}"

    # D. Date range with explicit YYYY-MM-DD
    if not plan.start_time:
        iso_range = re.search(r"(\d{4}-\d{2}-\d{2})\s*(?:至|到|~|-)\s*(\d{4}-\d{2}-\d{2})", q)
        if iso_range:
            plan.start_time = f"{iso_range.group(1)} 00:00:00"
            plan.end_time = f"{iso_range.group(2)} 23:59:59"
            plan.time_window = f"{iso_range.group(1)}~{iso_range.group(2)}"

    # Check for MoM / YoY comparison
    if "环比" in q:
        plan.is_comparison = True
        plan.comparison_type = "mom"
    elif "同比" in q:
        plan.is_comparison = True
        plan.comparison_type = "yoy"

    # Out-of-window check
    if plan.start_time and plan.start_time > f"{DEFAULT_DATASET_END} 23:59:59":
        plan.is_out_of_bounds = True
        plan.out_of_bounds_msg = (
            f"查询的时间（{plan.time_window or plan.start_time[:10]}）超出当前业务数据集时间范围"
            f"（数据时窗为 {DEFAULT_DATASET_START} 至 {DEFAULT_DATASET_END}）。"
        )

    # E. Trend grouping & granularity
    if any(k in q for k in ("按周", "分周", "各周", "每周", "周走势", "各周走势")):
        plan.dimension = "week"
        plan.time_granularity = "week"
    elif any(k in q for k in ("按天", "分天", "各天", "每日", "每天", "日走势", "逐日")):
        plan.dimension = "day"
        plan.time_granularity = "day"
    elif any(k in q for k in ("按月", "分月", "各月", "每月", "月度走势", "各月份")):
        plan.dimension = "month"
        plan.time_granularity = "month"
    elif any(k in q for k in ("按季度", "分季度", "各季度", "季度走势")):
        plan.dimension = "quarter"
        plan.time_granularity = "quarter"

    # 1. Cities & Provinces
    cities = ["上海", "杭州", "南京", "苏州", "宁波", "广州", "深圳", "北京", "成都", "青岛", "武汉"]
    for c in cities:
        if c in q:
            plan.city_filter = f"{c}市" if not c.endswith("市") else c
            break

    provinces = ["浙江", "江苏", "广东", "四川", "山东", "湖北", "安徽", "上海"]
    for p in provinces:
        if p in q:
            if p == "上海":
                plan.province_filter = "上海市"
            else:
                plan.province_filter = f"{p}省" if not p.endswith("省") else p
            break

    # 2. Channels
    channels = ["天猫App", "微信小程序", "直播间", "淘宝网页版"]
    for ch in channels:
        if ch in q:
            plan.channel_filter = ch
            break

    # 3. Member levels
    member_levels = ["黑钻VIP", "金卡会员", "银卡会员", "普客"]
    for ml in member_levels:
        if ml in q:
            plan.member_level_filter = ml
            break

    # 4. Categories (L1 & L2)
    cat_l1s = ["生鲜食品", "数码3C", "美妆个护", "母婴用品", "服饰鞋包", "家家居装"]
    for c1 in cat_l1s:
        if c1 in q:
            plan.category_filter = c1
            break

    # 5. Carriers
    carriers = ["顺丰速运", "中通快递", "京东物流", "圆通速递", "极兔速递"]
    matched_crs = [cr for cr in carriers if cr in q or cr[:2] in q]
    if len(matched_crs) == 1:
        plan.carrier_filter = matched_crs[0]
    elif len(matched_crs) > 1:
        plan.carriers_filter = matched_crs

    # Detect common carriers not in this database to proactively inform user
    unsupported_carrier_map = {
        "申通": "申通快递",
        "韵达": "韵达快递",
        "邮政": "中国邮政EMS",
        "EMS": "中国邮政EMS",
        "天天": "天天快递",
        "德邦": "德邦快递",
        "百世": "百世快递",
        "宅急送": "宅急送",
    }
    for alias, formal_name in unsupported_carrier_map.items():
        if alias in q and formal_name not in plan.unsupported_entities:
            plan.unsupported_entities.append(formal_name)
    if not plan.carrier_filter and not plan.carriers_filter:
        for alias in unsupported_carrier_map:
            if alias in q:
                plan.carrier_filter = alias
                break

    # 6. Statuses
    if "已妥投" in q or "已完成" in q or "妥投" in q:
        plan.status_filter = "已妥投"
    elif "售后中" in q:
        plan.status_filter = "售后中"
    elif "已取消" in q:
        plan.status_filter = "已取消"
    elif "运输中" in q:
        plan.status_filter = "运输中"
    elif "已退款" in q:
        plan.status_filter = "已退款"

    # Route status
    if "网点积压" in q:
        plan.route_status_filter = "网点积压"
    elif "正常在途" in q:
        plan.route_status_filter = "正常在途"

    # Ratings
    if "5星" in q or "好评" in q:
        plan.rating_filter = 5
    elif "1星" in q or "差评" in q:
        plan.rating_filter = 1

    # Tags
    if "包装破损" in q or "外包装挤压破损" in q:
        plan.tags_filter = "外包装挤压破损"

    # Refund types
    if "退货退款" in q:
        plan.refund_type_filter = "退货退款"
    elif "仅退款" in q:
        plan.refund_type_filter = "仅退款"

    # Refund reasons
    if "未按约定时间发货" in q:
        plan.refund_reason_filter = "未按约定时间发货"
    elif "外包装破损" in q:
        plan.refund_reason_filter = "外包装破损"
    elif "商品质量问题" in q:
        plan.refund_reason_filter = "商品质量问题"

    # Seller ID
    seller_m = re.search(r"seller_\d+", q)
    if seller_m:
        plan.seller_id_filter = seller_m.group(0)

    # Price bounds
    if "大于1000" in q or ">1000" in q:
        plan.min_amount = 1000.0
    elif "低于50" in q or "<50" in q:
        plan.max_amount = 50.0

    # Limit
    top_m = re.search(r"(?:top|前)\s*(\d+)", q, re.IGNORECASE)
    if top_m:
        plan.limit = int(top_m.group(1))
    elif any(k in q for k in ("前三", "哪三", "top3", "前3")):
        plan.limit = 3
    elif any(k in q for k in ("前二", "哪二", "top2", "前2")):
        plan.limit = 2
    elif any(k in q for k in ("前五", "哪五", "top5", "前5")):
        plan.limit = 5
    elif any(k in q for k in ("前十", "哪十", "top10", "前10")):
        plan.limit = 10

    # Multi-turn contextual standing filters inheritance
    if standing_filters:
        # If the query is grouping/ranking along city/province/carrier/channel, never inherit those as filtering constraints
        is_city_grouping = plan.dimension == "city" or any(k in q for k in ("各城市", "各个城市", "按城市", "分城市", "哪些城市", "城市排名", "城市分布"))
        is_prov_grouping = plan.dimension == "province" or any(k in q for k in ("各省份", "各个省份", "按省份", "分省份", "哪些省份", "省份排名", "省份分布"))
        is_cat_grouping = plan.dimension in ("category_l1", "category_l2") or any(k in q for k in ("各品类", "各个品类", "分类目", "各类目", "各个类目", "品类排名"))
        is_carrier_grouping = (
            plan.dimension == "carrier"
            or bool(plan.carriers_filter)
            or _is_carrier_grouping(q)
            or any(k in q for k in ("哪个快", "送得快", "更快", "时效", "快一点", "哪个送得", "快递排名", "各承运商", "各快递", "各物流", "按快递", "按承运商", "耗时", "快递速度"))
        )
        is_channel_grouping = (
            plan.dimension == "channel"
            or _is_channel_grouping(q)
            or any(k in q for k in ("各渠道", "各平台", "渠道分布", "渠道排名"))
        )
        is_broad_ranking = any(k in q for k in ("各个", "各城市", "各省份", "各渠道", "前五", "前5", "前十", "前10", "top5", "top10", "哪些城市"))

        if not is_city_grouping and not is_broad_ranking and not plan.city_filter and standing_filters.get("city"):
            plan.city_filter = str(standing_filters["city"]).strip("'\"")
        if not is_city_grouping and not is_prov_grouping and not is_broad_ranking and not plan.province_filter and standing_filters.get("province"):
            plan.province_filter = str(standing_filters["province"]).strip("'\"")
        if not is_cat_grouping and not is_broad_ranking and not plan.category_filter and standing_filters.get("category"):
            plan.category_filter = str(standing_filters["category"]).strip("'\"")
        if not is_cat_grouping and not is_broad_ranking and not plan.category_filter and standing_filters.get("category_l1"):
            plan.category_filter = str(standing_filters["category_l1"]).strip("'\"")
        if not is_carrier_grouping and not is_broad_ranking and not plan.carrier_filter and not plan.carriers_filter and standing_filters.get("carrier"):
            plan.carrier_filter = str(standing_filters["carrier"]).strip("'\"")
        if not is_channel_grouping and not is_broad_ranking and not plan.channel_filter and standing_filters.get("channel"):
            plan.channel_filter = str(standing_filters["channel"]).strip("'\"")
        if not plan.start_time and standing_filters.get("start_time"):
            plan.start_time = str(standing_filters["start_time"])
            plan.end_time = str(standing_filters.get("end_time", "")) or None
            plan.time_window = standing_filters.get("time_window")

    # Entity detection
    if any(k in q for k in ("包裹", "运单", "承运", "物流", "中通", "顺丰", "京东", "圆通", "极兔")) and "订单" not in q and "退款" not in q:
        plan.entity = "logistics"
    elif any(k in q for k in ("用户", "买家", "会员", "人", "注册")) and "订单" not in q and "销售额" not in q and "GMV" not in q and "评价" not in q:
        plan.entity = "users"
    elif any(k in q for k in ("商品库", "单品", "款商品", "类目商品", "特价商品")) and "销售" not in q and "订单" not in q:
        plan.entity = "products"
    elif any(k in q for k in ("店铺", "商家")) and "销售" not in q and "订单" not in q and "退款" not in q:
        plan.entity = "sellers"
    elif any(k in q for k in ("评价", "好评", "差评", "星级", "打分")) and "订单" not in q and "销售" not in q:
        plan.entity = "reviews"
    elif any(k in q for k in ("售后", "退款")) and "订单" not in q and "退款率" not in q:
        plan.entity = "aftersales"

    # Metric & Dimension detection
    if "客单价" in q and ("总额" in q or "销售额" in q or "订单数" in q or "总共" in q):
        plan.metric = "orders_sales_and_aov"
        return plan

    if "客单价" in q or "平均订单金额" in q or "平均每单" in q:
        plan.metric = "avg_order_value"
        if "城市" in q or "地区" in q:
            plan.dimension = "city"
        elif "省份" in q:
            plan.dimension = "province"
        elif "渠道" in q:
            plan.dimension = "channel"
        elif "承运商" in q or "快递" in q:
            plan.dimension = "carrier"
        return plan

    if "退款率" in q or "退货率" in q or "售后率" in q:
        plan.metric = "refund_rate"
        if "品类" in q or "类目" in q:
            plan.dimension = "category_l1"
            if "二级" in q:
                plan.dimension = "category_l2"
        elif "商家" in q or "店铺" in q:
            plan.dimension = "seller"
        elif "城市" in q or "地区" in q:
            plan.dimension = "city"
        elif "省份" in q:
            plan.dimension = "province"
        elif "承运商" in q or "快递" in q:
            plan.dimension = "carrier"
        return plan

    if any(k in q for k in ("时效对比", "物流时效", "送达时间", "耗时", "快递速度", "平均时效", "更快", "快一点", "哪个快", "送得快", "时效", "哪个送得更")):
        plan.metric = "delivery_speed"
        plan.dimension = "carrier"
        return plan

    if any(k in q for k in ("延误率", "延迟率", "延误占比", "超时率", "延误", "延迟")):
        plan.metric = "delay_rate"
        plan.entity = "logistics"
        if "承运商" in q or "快递" in q:
            plan.dimension = "carrier"
        elif "城市" in q or "地区" in q:
            plan.dimension = "city"
        elif "省份" in q:
            plan.dimension = "province"
        return plan

    if any(k in q.lower() for k in ("销售额", "销售总额", "营业额", "成交额", "成交总额", "gmv", "销售金额")):
        plan.metric = "sales_amount"
        if _is_city_grouping(q):
            plan.dimension = "city"
        elif _is_province_grouping(q):
            plan.dimension = "province"
        elif _is_category_grouping(q):
            plan.dimension = "category_l1"
        elif any(d in q for d in ("二级类目", "二级品类")):
            plan.dimension = "category_l2"
        elif _is_channel_grouping(q):
            plan.dimension = "channel"
        elif any(d in q for d in ("各会员", "会员等级", "会员级别")):
            plan.dimension = "member_level"
        elif _is_carrier_grouping(q):
            plan.dimension = "carrier"
        elif any(d in q for d in ("商家", "店铺")):
            plan.dimension = "seller"
        elif any(d in q for d in ("单品", "商品名称")):
            plan.dimension = "product"
        return plan

    if any(k in q for k in ("销量", "售出商品总件数", "商品总数量")):
        plan.metric = "sales_quantity"
        if _is_city_grouping(q):
            plan.dimension = "city"
        elif _is_province_grouping(q):
            plan.dimension = "province"
        elif _is_category_grouping(q):
            plan.dimension = "category_l1"
        elif any(d in q for d in ("二级类目", "二级品类")):
            plan.dimension = "category_l2"
        elif _is_channel_grouping(q):
            plan.dimension = "channel"
        elif any(d in q for d in ("单品", "商品名称")):
            plan.dimension = "product"
        return plan

    if any(k in q for k in ("订单数", "订单量", "多少笔", "多少订单", "订单总数", "订单总量", "总共有多少", "总单量", "单量")):
        plan.metric = "order_count"
        if _is_city_grouping(q):
            plan.dimension = "city"
        elif _is_province_grouping(q):
            plan.dimension = "province"
        elif _is_category_grouping(q):
            plan.dimension = "category_l1"
        elif any(d in q for d in ("各会员", "会员等级", "会员级别")):
            plan.dimension = "member_level"
        elif _is_channel_grouping(q):
            plan.dimension = "channel"
        elif _is_carrier_grouping(q):
            plan.dimension = "carrier"
        elif any(d in q for d in ("状态分布", "订单状态")):
            plan.dimension = "order_status"
        return plan

    if any(k in q for k in ("用户数", "总人数", "会员数", "多少人")):
        plan.metric = "user_count"
        plan.entity = "users"
        if _is_province_grouping(q):
            plan.dimension = "province"
        elif _is_city_grouping(q):
            plan.dimension = "city"
        elif any(d in q for d in ("会员等级", "级别")):
            plan.dimension = "member_level"
        return plan

    if any(k in q for k in ("店铺总数", "商家总数", "入驻商家")):
        plan.metric = "seller_count"
        plan.entity = "sellers"
        if _is_province_grouping(q):
            plan.dimension = "province"
        elif _is_city_grouping(q):
            plan.dimension = "city"
        return plan

    if any(k in q for k in ("包裹总量", "包裹总数", "包裹数量")):
        plan.metric = "package_count"
        plan.entity = "logistics"
        if any(d in q for d in ("承运商", "快递")):
            plan.dimension = "carrier"
        elif any(d in q for d in ("路由状态", "物流状态")):
            plan.dimension = "route_status"
        return plan

    if any(k in q for k in ("评价总数", "评价数量", "评价星级", "评价打分", "差评订单总量")):
        plan.metric = "review_count"
        plan.entity = "reviews"
        if any(d in q for d in ("星级", "评分")):
            plan.dimension = "rating"
        elif any(d in q for d in ("渠道", "各渠道")):
            plan.dimension = "channel"
        elif any(d in q for d in ("会员等级", "会员")):
            plan.dimension = "member_level"
        return plan

    if any(k in q for k in ("退款总金额", "售后申请总数", "退款单量", "退款总数")):
        plan.metric = "aftersales_metric"
        plan.entity = "aftersales"
        if any(d in q for d in ("退款类型", "类型")):
            plan.dimension = "refund_type"
        elif any(d in q for d in ("状态", "退款状态")):
            plan.dimension = "status"
        elif any(d in q for d in ("退款原因", "原因")):
            plan.dimension = "refund_reason"
        elif any(d in q for d in ("渠道", "各渠道")):
            plan.dimension = "channel"
        elif any(d in q for d in ("承运商", "快递")):
            plan.dimension = "carrier"
        elif any(d in q for d in ("商家", "店铺")):
            plan.dimension = "seller"
        return plan

    # Default fallback plan if keywords matched
    if "状态分布" in q:
        plan.metric = "order_count"
        plan.dimension = "order_status"
        plan.status_filter = None
        return plan

    return None


def _render_time_dimension(granularity: str | None, col: str = "o.order_time") -> str:
    """Render SQLite datetime formatting expression for a given granularity."""
    mapping = {
        "week": f"strftime('%Y-%W', {col})",
        "day": f"strftime('%Y-%m-%d', {col})",
        "quarter": (
            f"(CASE "
            f"WHEN strftime('%m', {col}) BETWEEN '01' AND '03' THEN 'Q1' "
            f"WHEN strftime('%m', {col}) BETWEEN '04' AND '06' THEN 'Q2' "
            f"WHEN strftime('%m', {col}) BETWEEN '07' AND '09' THEN 'Q3' "
            f"ELSE 'Q4' END)"
        ),
    }
    return mapping.get(granularity or "", f"strftime('%Y-%m', {col})")


def compile_cn_sql(plan: CNQueryPlan | None) -> str:
    """Compile a CNQueryPlan into a standard SQLite SELECT statement for cn_retail.db."""
    if plan is None:
        return ""
    if plan.custom_sql:
        return plan.custom_sql

    where_clauses: list[str] = []
    joins: set[str] = set()

    # Filters
    if plan.status_filter:
        where_clauses.append(f"o.order_status = '{plan.status_filter}'")

    if plan.city_filter:
        joins.add("JOIN cn_users u ON o.user_id = u.user_id")
        where_clauses.append(f"u.city = '{plan.city_filter}'")

    if plan.province_filter:
        if plan.entity == "sellers":
            where_clauses.append(f"s.province = '{plan.province_filter}'")
        elif plan.entity == "users":
            where_clauses.append(f"u.province = '{plan.province_filter}'")
        else:
            joins.add("JOIN cn_users u ON o.user_id = u.user_id")
            where_clauses.append(f"u.province = '{plan.province_filter}'")

    if plan.channel_filter:
        where_clauses.append(f"o.channel = '{plan.channel_filter}'")

    if plan.member_level_filter:
        if plan.entity == "users":
            where_clauses.append(f"u.member_level = '{plan.member_level_filter}'")
        else:
            joins.add("JOIN cn_users u ON o.user_id = u.user_id")
            where_clauses.append(f"u.member_level = '{plan.member_level_filter}'")

    if plan.carriers_filter:
        crs_in = ", ".join(f"'{c}'" for c in plan.carriers_filter)
        if plan.entity == "logistics":
            where_clauses.append(f"l.carrier IN ({crs_in})")
        else:
            joins.add("JOIN cn_logistics l ON o.order_id = l.order_id")
            where_clauses.append(f"l.carrier IN ({crs_in})")
    elif plan.carrier_filter:
        if plan.entity == "logistics":
            where_clauses.append(f"l.carrier = '{plan.carrier_filter}'")
        else:
            joins.add("JOIN cn_logistics l ON o.order_id = l.order_id")
            where_clauses.append(f"l.carrier = '{plan.carrier_filter}'")

    if plan.category_filter:
        if plan.entity == "products":
            where_clauses.append(f"p.category_l1 = '{plan.category_filter}'")
        else:
            joins.add("JOIN cn_order_items oi ON o.order_id = oi.order_id")
            joins.add("JOIN cn_products p ON oi.product_id = p.product_id")
            where_clauses.append(f"p.category_l1 = '{plan.category_filter}'")

    if plan.route_status_filter:
        if plan.entity == "logistics":
            where_clauses.append(f"l.route_status = '{plan.route_status_filter}'")
        else:
            joins.add("JOIN cn_logistics l ON o.order_id = l.order_id")
            where_clauses.append(f"l.route_status = '{plan.route_status_filter}'")

    if plan.rating_filter:
        if plan.entity == "reviews":
            if plan.rating_filter == 1:
                where_clauses.append("r.rating = 1")
            else:
                where_clauses.append(f"r.rating = {plan.rating_filter}")
        else:
            joins.add("JOIN cn_reviews r ON o.order_id = r.order_id")
            where_clauses.append(f"r.rating = {plan.rating_filter}")

    if plan.min_amount:
        where_clauses.append(f"o.total_amount > {plan.min_amount}")

    # Time bounds filtering (hits idx_cn_orders_time)
    if plan.start_time and plan.end_time:
        if plan.entity == "logistics":
            where_clauses.append(f"l.shipped_time >= '{plan.start_time}' AND l.shipped_time <= '{plan.end_time}'")
        else:
            where_clauses.append(f"o.order_time >= '{plan.start_time}' AND o.order_time <= '{plan.end_time}'")

    where_str = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    join_str = " " + " ".join(joins) if joins else ""

    # 1. Users entity queries
    if plan.entity == "users" or plan.metric == "user_count":
        u_where = []
        if plan.member_level_filter:
            u_where.append(f"member_level = '{plan.member_level_filter}'")
        if plan.province_filter:
            u_where.append(f"province = '{plan.province_filter}'")
        if plan.city_filter:
            u_where.append(f"city = '{plan.city_filter}'")
        uw_str = f" WHERE {' AND '.join(u_where)}" if u_where else ""
        if plan.dimension == "member_level":
            return f"SELECT member_level, COUNT(*) AS count FROM cn_users{uw_str} GROUP BY member_level"
        if plan.dimension == "province":
            return f"SELECT province, COUNT(*) AS cnt FROM cn_users{uw_str} GROUP BY province ORDER BY cnt DESC LIMIT {plan.limit}"
        if plan.dimension == "city":
            return f"SELECT city, COUNT(*) AS cnt FROM cn_users{uw_str} GROUP BY city ORDER BY cnt DESC LIMIT {plan.limit}"
        return f"SELECT COUNT(*) AS metric FROM cn_users{uw_str}"

    # 2. Products entity queries
    if plan.entity == "products":
        p_where = []
        if plan.category_filter:
            p_where.append(f"category_l1 = '{plan.category_filter}'")
        if plan.max_amount:
            p_where.append(f"base_price < {plan.max_amount}")
        pw_str = f" WHERE {' AND '.join(p_where)}" if p_where else ""
        if plan.dimension == "category_l1":
            return f"SELECT category_l1, COUNT(*) AS count FROM cn_products{pw_str} GROUP BY category_l1"
        if plan.dimension == "category_l2":
            return f"SELECT category_l2, COUNT(*) AS count FROM cn_products{pw_str} GROUP BY category_l2"
        return f"SELECT COUNT(*) AS metric FROM cn_products{pw_str}"

    # 3. Sellers entity queries
    if plan.entity == "sellers" or plan.metric == "seller_count":
        s_where = []
        if plan.province_filter:
            s_where.append(f"province = '{plan.province_filter}'")
        if plan.city_filter:
            s_where.append(f"city = '{plan.city_filter}'")
        sw_str = f" WHERE {' AND '.join(s_where)}" if s_where else ""
        if plan.dimension == "province":
            return f"SELECT province, COUNT(*) AS cnt FROM cn_sellers{sw_str} GROUP BY province ORDER BY cnt DESC LIMIT {plan.limit}"
        return f"SELECT COUNT(*) AS metric FROM cn_sellers{sw_str}"

    # 4. Logistics entity queries
    if (plan.entity == "logistics" or plan.metric == "package_count") and plan.metric not in ("delivery_speed", "delay_rate"):
        l_where = []
        if plan.carriers_filter:
            crs_in = ", ".join(f"'{c}'" for c in plan.carriers_filter)
            l_where.append(f"carrier IN ({crs_in})")
        elif plan.carrier_filter:
            l_where.append(f"carrier = '{plan.carrier_filter}'")
        if plan.route_status_filter:
            l_where.append(f"route_status = '{plan.route_status_filter}'")
        lw_str = f" WHERE {' AND '.join(l_where)}" if l_where else ""
        if plan.dimension == "carrier":
            return f"SELECT carrier, COUNT(*) AS count FROM cn_logistics{lw_str} GROUP BY carrier ORDER BY count DESC LIMIT {plan.limit}"
        if plan.dimension == "route_status":
            return f"SELECT route_status, COUNT(*) AS count FROM cn_logistics{lw_str} GROUP BY route_status"
        return f"SELECT COUNT(*) AS metric FROM cn_logistics{lw_str}"

    # 5. Reviews entity queries
    if plan.entity == "reviews" or plan.metric == "review_count":
        r_where = []
        if plan.rating_filter:
            r_where.append(f"rating = {plan.rating_filter}")
        if plan.tags_filter:
            r_where.append(f"tags LIKE '%{plan.tags_filter}%'")
        rw_str = f" WHERE {' AND '.join(r_where)}" if r_where else ""
        if plan.dimension == "rating":
            return f"SELECT rating, COUNT(*) AS count FROM cn_reviews{rw_str} GROUP BY rating ORDER BY rating DESC"
        if plan.dimension == "channel":
            return "SELECT o.channel, ROUND(AVG(r.rating), 2) AS avg_rating FROM cn_orders o JOIN cn_reviews r ON o.order_id = r.order_id GROUP BY o.channel ORDER BY avg_rating DESC"
        if plan.dimension == "member_level":
            return "SELECT u.member_level, ROUND(AVG(r.rating), 2) AS avg_rating FROM cn_orders o JOIN cn_users u ON o.user_id = u.user_id JOIN cn_reviews r ON o.order_id = r.order_id GROUP BY u.member_level ORDER BY avg_rating DESC"
        return f"SELECT COUNT(*) AS metric FROM cn_reviews{rw_str}"

    # 6. Aftersales entity queries
    if plan.entity == "aftersales" or plan.metric == "aftersales_metric":
        a_where = []
        if plan.refund_type_filter:
            a_where.append(f"refund_type = '{plan.refund_type_filter}'")
        if plan.refund_reason_filter:
            a_where.append(f"refund_reason = '{plan.refund_reason_filter}'")
        if plan.status_filter:
            a_where.append(f"status = '{plan.status_filter}'")
        aw_str = f" WHERE {' AND '.join(a_where)}" if a_where else ""
        if plan.dimension == "refund_type":
            return f"SELECT refund_type, COUNT(*) AS count FROM cn_aftersales{aw_str} GROUP BY refund_type"
        if plan.dimension == "status":
            return f"SELECT status, COUNT(*) AS count FROM cn_aftersales{aw_str} GROUP BY status"
        if plan.dimension == "refund_reason":
            return f"SELECT refund_reason, COUNT(*) AS cnt FROM cn_aftersales{aw_str} GROUP BY refund_reason ORDER BY cnt DESC LIMIT {plan.limit}"
        if plan.dimension == "channel":
            return "SELECT o.channel, COUNT(a.refund_id) AS refund_count FROM cn_orders o JOIN cn_aftersales a ON o.order_id = a.order_id GROUP BY o.channel ORDER BY refund_count DESC"
        if plan.dimension == "seller":
            return f"SELECT s.seller_name, ROUND(SUM(a.refund_amount), 2) AS total_refund FROM cn_sellers s JOIN cn_order_items oi ON s.seller_id = oi.seller_id JOIN cn_aftersales a ON oi.order_id = a.order_id GROUP BY s.seller_name ORDER BY total_refund DESC LIMIT {plan.limit}"
        return f"SELECT COUNT(*) AS metric FROM cn_aftersales{aw_str}"

    # 7. Order Count
    if plan.metric == "order_count":
        if plan.dimension == "order_status":
            return "SELECT order_status, COUNT(*) AS count FROM cn_orders GROUP BY order_status"
        if plan.dimension in ("month", "week", "day", "quarter"):
            dim_expr = _render_time_dimension(plan.time_granularity or plan.dimension)
            return f"SELECT {dim_expr} AS dimension, COUNT(o.order_id) AS order_count FROM cn_orders o{join_str}{where_str} GROUP BY dimension ORDER BY dimension ASC"
        if plan.dimension == "city":
            joins.add("JOIN cn_users u ON o.user_id = u.user_id")
            join_str = " " + " ".join(joins)
            return f"SELECT u.city AS dimension, COUNT(o.order_id) AS order_count FROM cn_orders o{join_str}{where_str} GROUP BY u.city ORDER BY order_count DESC LIMIT {plan.limit}"
        if plan.dimension == "province":
            joins.add("JOIN cn_users u ON o.user_id = u.user_id")
            join_str = " " + " ".join(joins)
            return f"SELECT u.province AS dimension, COUNT(o.order_id) AS order_count FROM cn_orders o{join_str}{where_str} GROUP BY u.province ORDER BY order_count DESC LIMIT {plan.limit}"
        if plan.dimension == "channel":
            return f"SELECT o.channel AS dimension, COUNT(o.order_id) AS order_count FROM cn_orders o{join_str}{where_str} GROUP BY o.channel ORDER BY order_count DESC LIMIT {plan.limit}"
        if plan.dimension == "carrier":
            joins.add("JOIN cn_logistics l ON o.order_id = l.order_id")
            join_str = " " + " ".join(joins)
            return f"SELECT l.carrier AS dimension, COUNT(o.order_id) AS order_count FROM cn_orders o{join_str}{where_str} GROUP BY l.carrier ORDER BY order_count DESC LIMIT {plan.limit}"
        return f"SELECT COUNT(o.order_id) AS order_count FROM cn_orders o{join_str}{where_str}"

    # 8. Avg Order Value
    if plan.metric == "avg_order_value":
        if plan.dimension == "city":
            joins.add("JOIN cn_users u ON o.user_id = u.user_id")
            join_str = " " + " ".join(joins)
            return f"SELECT u.city AS dimension, ROUND(AVG(o.total_amount), 2) AS avg_order_amount FROM cn_orders o{join_str}{where_str} GROUP BY u.city ORDER BY avg_order_amount DESC LIMIT {plan.limit}"
        if plan.dimension == "province":
            joins.add("JOIN cn_users u ON o.user_id = u.user_id")
            join_str = " " + " ".join(joins)
            return f"SELECT u.province AS dimension, ROUND(AVG(o.total_amount), 2) AS avg_order_amount FROM cn_orders o{join_str}{where_str} GROUP BY u.province ORDER BY avg_order_amount DESC LIMIT {plan.limit}"
        if plan.dimension == "carrier":
            joins.add("JOIN cn_logistics l ON o.order_id = l.order_id")
            join_str = " " + " ".join(joins)
            return f"SELECT l.carrier AS dimension, ROUND(AVG(o.total_amount), 2) AS avg_order_amount FROM cn_orders o{join_str}{where_str} GROUP BY l.carrier ORDER BY avg_order_amount DESC LIMIT {plan.limit}"
        return f"SELECT ROUND(AVG(o.total_amount), 2) AS avg_order_amount FROM cn_orders o{join_str}{where_str}"

    # 8.5 Delay Rate
    if plan.metric == "delay_rate":
        joins.add("JOIN cn_logistics l ON o.order_id = l.order_id")
        join_str = " " + " ".join(joins)
        delay_cond = "(l.actual_delivery_time > l.promise_delivery_time OR l.route_status = '网点积压')"
        if plan.dimension == "carrier":
            return f"SELECT l.carrier AS dimension, ROUND(AVG(CASE WHEN {delay_cond} THEN 1.0 ELSE 0.0 END) * 100, 2) AS delay_rate_pct, COUNT(o.order_id) AS total_orders FROM cn_orders o{join_str}{where_str} GROUP BY l.carrier ORDER BY delay_rate_pct DESC LIMIT {plan.limit}"
        if plan.dimension == "city":
            joins.add("JOIN cn_users u ON o.user_id = u.user_id")
            join_str = " " + " ".join(joins)
            return f"SELECT u.city AS dimension, ROUND(AVG(CASE WHEN {delay_cond} THEN 1.0 ELSE 0.0 END) * 100, 2) AS delay_rate_pct, COUNT(o.order_id) AS total_orders FROM cn_orders o{join_str}{where_str} GROUP BY u.city ORDER BY delay_rate_pct DESC LIMIT {plan.limit}"
        if plan.dimension == "province":
            joins.add("JOIN cn_users u ON o.user_id = u.user_id")
            join_str = " " + " ".join(joins)
            return f"SELECT u.province AS dimension, ROUND(AVG(CASE WHEN {delay_cond} THEN 1.0 ELSE 0.0 END) * 100, 2) AS delay_rate_pct, COUNT(o.order_id) AS total_orders FROM cn_orders o{join_str}{where_str} GROUP BY u.province ORDER BY delay_rate_pct DESC LIMIT {plan.limit}"
        return f"SELECT ROUND(AVG(CASE WHEN {delay_cond} THEN 1.0 ELSE 0.0 END) * 100, 2) AS delay_rate_pct, COUNT(o.order_id) AS total_orders FROM cn_orders o{join_str}{where_str}"
 
    # 8.6 Delivery Speed
    if plan.metric == "delivery_speed":
        joins.add("JOIN cn_logistics l ON o.order_id = l.order_id")
        join_str = " " + " ".join(joins)
        speed_clauses = list(where_clauses)
        speed_clauses.append("l.actual_delivery_time IS NOT NULL")
        sw_str = f" WHERE {' AND '.join(speed_clauses)}"
        if plan.dimension == "carrier":
            return (
                f"SELECT l.carrier AS dimension, "
                f"ROUND(AVG(julianday(l.actual_delivery_time) - julianday(l.shipped_time)), 2) AS avg_delivery_days, "
                f"ROUND(AVG(CASE WHEN (l.actual_delivery_time > l.promise_delivery_time OR l.route_status = '网点积压') THEN 1.0 ELSE 0.0 END) * 100, 2) AS delay_rate_pct, "
                f"COUNT(o.order_id) AS total_orders "
                f"FROM cn_orders o{join_str}{sw_str} "
                f"GROUP BY l.carrier ORDER BY avg_delivery_days ASC LIMIT {plan.limit}"
            )
        return (
            f"SELECT ROUND(AVG(julianday(l.actual_delivery_time) - julianday(l.shipped_time)), 2) AS avg_delivery_days "
            f"FROM cn_orders o{join_str}{sw_str}"
        )

    # 9. Sales Amount (GMV)
    if plan.metric == "sales_amount":
        if plan.dimension == "city":
            joins.add("JOIN cn_users u ON o.user_id = u.user_id")
            join_str = " " + " ".join(joins)
            return f"SELECT u.city AS dimension, ROUND(SUM(o.total_amount), 2) AS total_amount FROM cn_orders o{join_str}{where_str} GROUP BY u.city ORDER BY total_amount DESC LIMIT {plan.limit}"
        if plan.dimension in ("month", "week", "day", "quarter"):
            dim_expr = _render_time_dimension(plan.time_granularity or plan.dimension)
            return f"SELECT {dim_expr} AS dimension, ROUND(SUM(o.total_amount), 2) AS total_amount FROM cn_orders o{join_str}{where_str} GROUP BY dimension ORDER BY dimension ASC"
        if plan.dimension == "province":
            joins.add("JOIN cn_users u ON o.user_id = u.user_id")
            join_str = " " + " ".join(joins)
            return f"SELECT u.province AS dimension, ROUND(SUM(o.total_amount), 2) AS total_amount FROM cn_orders o{join_str}{where_str} GROUP BY u.province ORDER BY total_amount DESC LIMIT {plan.limit}"
        if plan.dimension == "category_l1":
            joins.add("JOIN cn_order_items oi ON o.order_id = oi.order_id")
            joins.add("JOIN cn_products p ON oi.product_id = p.product_id")
            join_str = " " + " ".join(joins)
            return f"SELECT p.category_l1 AS dimension, ROUND(SUM(oi.unit_price * oi.quantity), 2) AS total_sales FROM cn_orders o{join_str}{where_str} GROUP BY p.category_l1 ORDER BY total_sales DESC LIMIT {plan.limit}"
        if plan.dimension == "category_l2":
            joins.add("JOIN cn_order_items oi ON o.order_id = oi.order_id")
            joins.add("JOIN cn_products p ON oi.product_id = p.product_id")
            join_str = " " + " ".join(joins)
            return f"SELECT p.category_l2 AS dimension, ROUND(SUM(oi.unit_price * oi.quantity), 2) AS total_sales FROM cn_orders o{join_str}{where_str} GROUP BY p.category_l2 ORDER BY total_sales DESC LIMIT {plan.limit}"
        if plan.dimension == "channel":
            return f"SELECT o.channel AS dimension, ROUND(SUM(o.total_amount), 2) AS total_sales FROM cn_orders o{join_str}{where_str} GROUP BY o.channel ORDER BY total_sales DESC LIMIT {plan.limit}"
        if plan.dimension == "member_level":
            joins.add("JOIN cn_users u ON o.user_id = u.user_id")
            join_str = " " + " ".join(joins)
            return f"SELECT u.member_level AS dimension, ROUND(SUM(o.total_amount), 2) AS total_sales FROM cn_orders o{join_str}{where_str} GROUP BY u.member_level ORDER BY total_sales DESC LIMIT {plan.limit}"
        if plan.dimension == "seller":
            joins.add("JOIN cn_order_items oi ON o.order_id = oi.order_id")
            joins.add("JOIN cn_sellers s ON oi.seller_id = s.seller_id")
            join_str = " " + " ".join(joins)
            return f"SELECT s.seller_name AS dimension, ROUND(SUM(oi.unit_price * oi.quantity), 2) AS total_sales FROM cn_orders o{join_str}{where_str} GROUP BY s.seller_name ORDER BY total_sales DESC LIMIT {plan.limit}"
        if plan.dimension == "product":
            joins.add("JOIN cn_order_items oi ON o.order_id = oi.order_id")
            joins.add("JOIN cn_products p ON oi.product_id = p.product_id")
            join_str = " " + " ".join(joins)
            return f"SELECT p.product_name AS dimension, ROUND(SUM(oi.unit_price * oi.quantity), 2) AS total_sales FROM cn_orders o{join_str}{where_str} GROUP BY p.product_name ORDER BY total_sales DESC LIMIT {plan.limit}"
        return f"SELECT ROUND(SUM(o.total_amount), 2) AS total_sales FROM cn_orders o{join_str}{where_str}"

    # 10. Refund Rate
    if plan.metric == "refund_rate":
        rf_joins = set(joins)
        if plan.dimension in ("month", "week", "day", "quarter"):
            dim_expr = _render_time_dimension(plan.time_granularity or plan.dimension)
            rf_join_str = (" " + " ".join(rf_joins)) if rf_joins else ""
            return (
                f"SELECT {dim_expr} AS dimension, "
                f"ROUND(COUNT(DISTINCT a.refund_id) * 100.0 / NULLIF(COUNT(DISTINCT o.order_id), 0), 2) AS refund_rate_pct "
                f"FROM cn_orders o{rf_join_str} "
                f"LEFT JOIN cn_aftersales a ON a.order_id = o.order_id{where_str} "
                f"GROUP BY dimension ORDER BY dimension ASC"
            )
        if plan.dimension == "category_l1" or plan.dimension is None:
            rf_joins.add("JOIN cn_order_items oi ON o.order_id = oi.order_id")
            rf_joins.add("JOIN cn_products p ON oi.product_id = p.product_id")
            rf_join_str = " " + " ".join(rf_joins)
            return (
                f"SELECT p.category_l1 AS dimension, "
                f"ROUND(COUNT(DISTINCT a.refund_id) * 100.0 / NULLIF(COUNT(DISTINCT o.order_id), 0), 2) AS refund_rate_pct "
                f"FROM cn_orders o{rf_join_str} "
                f"LEFT JOIN cn_aftersales a ON a.order_id = o.order_id{where_str} "
                f"GROUP BY p.category_l1 ORDER BY refund_rate_pct DESC LIMIT {plan.limit}"
            )
        if plan.dimension == "carrier":
            rf_joins.add("JOIN cn_logistics l ON o.order_id = l.order_id")
            rf_join_str = " " + " ".join(rf_joins)
            return (
                f"SELECT l.carrier AS dimension, "
                f"ROUND(COUNT(DISTINCT a.refund_id) * 100.0 / NULLIF(COUNT(DISTINCT o.order_id), 0), 2) AS refund_rate_pct "
                f"FROM cn_orders o{rf_join_str} "
                f"LEFT JOIN cn_aftersales a ON a.order_id = o.order_id{where_str} "
                f"GROUP BY l.carrier ORDER BY refund_rate_pct DESC LIMIT {plan.limit}"
            )

    # 11. Delivery Speed by Carrier
    if plan.metric == "delivery_speed":
        deliv_where = ["l.actual_delivery_time IS NOT NULL"]
        if plan.start_time and plan.end_time:
            deliv_where.append(f"l.shipped_time >= '{plan.start_time}' AND l.shipped_time <= '{plan.end_time}'")
        if plan.carrier_filter:
            deliv_where.append(f"l.carrier = '{plan.carrier_filter}'")
        dw_str = f" WHERE {' AND '.join(deliv_where)}"
        return (
            f"SELECT l.carrier AS dimension, "
            f"ROUND(AVG(julianday(l.actual_delivery_time) - julianday(l.shipped_time)) * 24, 1) AS avg_delivery_hours "
            f"FROM cn_logistics l{dw_str} "
            f"GROUP BY l.carrier ORDER BY avg_delivery_hours ASC LIMIT {plan.limit}"
        )

    # Default fallback
    return "SELECT order_status, COUNT(*) AS count FROM cn_orders GROUP BY order_status"
