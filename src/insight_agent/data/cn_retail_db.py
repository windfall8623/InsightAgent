"""CN-Retail Domestic E-Commerce Sandbox Database.

Implements:
1. 8-table Chinese retail data warehouse modeling (cn_users, cn_sellers, cn_products,
   cn_orders, cn_order_items, cn_logistics, cn_reviews, cn_aftersales).
2. Deterministic pseudo-random seed generator producing ~90,000+ realistic rows (50,000+ milestone).
3. Ground-truth business anomaly injections:
   - Logistics Congestion: Zhongtong Express transfer center congestion in East China (July-August 2024).
   - Packaging Defect: Fresh food foam box breakage and leakage for specific cold-chain sellers.
4. Read-only SQLite connection guarantee with engine-level authorizer and thread-safe locking.
"""

from __future__ import annotations

import random
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

CN_RETAIL_SCHEMA_PATH = Path(__file__).with_name("cn_retail_schema.sql")

CN_RETAIL_TABLES: tuple[str, ...] = (
    "cn_users",
    "cn_sellers",
    "cn_products",
    "cn_orders",
    "cn_order_items",
    "cn_logistics",
    "cn_reviews",
    "cn_aftersales",
)

CN_DATA_START = datetime(2024, 1, 1)
CN_DATA_END = datetime(2024, 10, 1)
AS_OF_CN = CN_DATA_END

# Realistic Chinese Administrative Regions
CN_REGIONS: dict[str, list[str]] = {
    "上海市": ["上海市"],
    "浙江省": ["杭州市", "宁波市", "温州市", "金华市", "嘉兴市", "绍兴市"],
    "江苏省": ["南京市", "苏州市", "无锡市", "常州市", "南通市"],
    "广东省": ["广州市", "深圳市", "东莞市", "佛山市", "珠海市"],
    "北京市": ["北京市"],
    "四川省": ["成都市", "绵阳市"],
    "山东省": ["青岛市", "济南市", "烟台市"],
    "湖北省": ["武汉市", "宜昌市"],
}

CARRIERS = ("顺丰速运", "中通快递", "京东物流", "圆通速递", "极兔速递")
CHANNELS = ("天猫App", "微信小程序", "直播间", "淘宝网页版")
MEMBER_LEVELS = ("普客", "银卡会员", "金卡会员", "黑钻VIP")

CN_CATEGORIES: list[tuple[str, str, list[str], tuple[float, float]]] = [
    ("生鲜食品", "冷冻海鲜", ["舟山大黄鱼", "青岛大虾", "智利三文鱼排", "大连冷冻鲍鱼", "厄瓜多尔白虾"], (48.0, 328.0)),
    ("生鲜食品", "新鲜水果", ["烟台红富士苹果", "新疆西梅", "海南红心火龙果", "四川爱媛38号柑橘", "赣南脐橙"], (25.0, 118.0)),
    ("美妆个护", "面部精华", ["玻尿酸修护原液", "烟酰胺焕亮精华", "视黄醇抗皱精华", "寡肽控油修护原液"], (89.0, 499.0)),
    ("美妆个护", "洗护用品", ["氨基酸控油洗发水", "植物精油沐浴露", "头皮舒缓磨砂膏"], (38.0, 168.0)),
    ("数码3C", "手机配件", ["快充氮化镓充电头", "磁吸无线充电宝", "防窥钢化膜2片装", "编织Type-C数据线"], (19.9, 159.0)),
    ("数码3C", "智能影音", ["主动降噪无线耳机", "桌面低音炮音箱", "高清便携投影仪"], (129.0, 1299.0)),
    ("服饰鞋包", "休闲女装", ["纯棉宽松短袖T恤", "高腰显瘦阔腿裤", "法式复古收腰连衣裙"], (59.0, 320.0)),
    ("服饰鞋包", "潮流男装", ["重磅防风连帽卫衣", "工装休闲束脚裤", "保暖加厚羽绒服"], (79.0, 580.0)),
    ("母婴用品", "纸尿裤", ["超薄透气婴儿拉拉裤L码", "抑菌丝柔新生儿纸尿裤NB", "瞬吸干爽学步裤XL"], (69.0, 189.0)),
    ("家居家装", "厨房卫浴", ["食品级硅胶保鲜盖", "不锈钢旋转沥水架", "增压过滤花洒套装"], (28.0, 199.0)),
]


@dataclass(slots=True)
class CNRetailQueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[object, ...]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: float = 0.0

    def as_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]

    def to_payload(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": self.as_dicts(),
            "row_count": self.row_count,
            "truncated": self.truncated,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


def default_cn_retail_db_path() -> Path:
    import os

    override = os.environ.get("INSIGHT_CN_RETAIL_DB_PATH")
    if override:
        return Path(override)
    return Path(__file__).with_name("cn_retail.db")


def _authorizer_readonly(
    action: int,
    arg1: str | None,
    arg2: str | None,
    db_name: str | None,
    trigger: str | None,
) -> int:
    """Refuse any write or schema-modifying actions inside SQLite."""
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
    if action in allowed:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() in {
        "table_info",
        "index_list",
        "foreign_key_list",
    }:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


_CN_LOCKS: dict[int, threading.Lock] = {}


def _lock_for_cn(conn: sqlite3.Connection) -> threading.Lock:
    lock = _CN_LOCKS.get(id(conn))
    if lock is None:
        lock = threading.Lock()
        _CN_LOCKS[id(conn)] = lock
    return lock


def connect_cn_retail(db_path: Path | None = None) -> sqlite3.Connection:
    """Connect to the CN-Retail database in read-only mode with authorizer enabled."""
    path = ensure_cn_retail_database(db_path)
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.set_authorizer(_authorizer_readonly)
    return conn


def execute_cn_retail(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence[object] = (),
    *,
    limit: int = 500,
) -> CNRetailQueryResult:
    """Safely execute a query with per-connection thread mutex."""
    import time

    start = time.perf_counter()
    with _lock_for_cn(conn):
        cursor = conn.cursor()
        cursor.execute(sql, params)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(limit + 1)
        truncated = len(rows) > limit
        if truncated:
            rows = rows[:limit]
        elapsed = (time.perf_counter() - start) * 1000
        return CNRetailQueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=round(elapsed, 2),
        )


def ensure_cn_retail_database(db_path: Path | None = None, *, seed: int = 42) -> Path:
    """Create and seed the CN-Retail dataset with authentic domestic retail scenarios."""
    path = db_path or default_cn_retail_db_path()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(CN_RETAIL_SCHEMA_PATH.read_text(encoding="utf-8"))
        _seed_cn_retail(conn, seed=seed)
        conn.commit()
    from insight_agent.data.inject_anomaly_specs import inject_standard_anomalies
    inject_standard_anomalies(path)
    return path


def _seed_cn_retail(conn: sqlite3.Connection, *, seed: int) -> None:
    """Generate realistic CN-Retail tables with verified domestic anomaly scenarios."""
    rng = random.Random(seed)

    # 1. Users (1,500 domestic buyers)
    users = []
    user_tuples = []
    provinces = list(CN_REGIONS.keys())
    weights = [25, 22, 18, 15, 8, 5, 4, 3]  # East & South China major e-commerce belts

    for i in range(1, 1501):
        uid = f"user_{i:05d}"
        prov = rng.choices(provinces, weights=weights)[0]
        city = rng.choice(CN_REGIONS[prov])
        m_level = rng.choices(MEMBER_LEVELS, weights=[50, 30, 15, 5])[0]
        users.append((uid, prov, city, m_level))
        user_tuples.append((uid, prov, city, m_level))

    conn.executemany(
        "INSERT INTO cn_users (user_id, province, city, member_level) VALUES (?, ?, ?, ?)",
        users,
    )

    # 2. Sellers (120 merchant shops)
    sellers = []
    seller_ids = []
    seller_prefixes = ["华东", "江南", "天府", "南粤", "京华", "极速", "优品", "旗舰", "臻选", "冷链"]
    seller_suffixes = ["专营店", "官方旗舰店", "精品店", "供应链直营", "生鲜特卖馆"]

    for i in range(1, 121):
        sid = f"seller_{i:04d}"
        prov = rng.choice(provinces)
        city = rng.choice(CN_REGIONS[prov])
        sname = f"{rng.choice(seller_prefixes)}{rng.choice(seller_suffixes)}_{i:03d}"
        if i == 28:
            sname = "天府冷链专营店"  # Packaging defect anomaly seller
            prov = "四川省"
            city = "成都市"
        sellers.append((sid, sname, prov, city))
        seller_ids.append((sid, sname, prov, city))

    conn.executemany(
        "INSERT INTO cn_sellers (seller_id, seller_name, province, city) VALUES (?, ?, ?, ?)",
        sellers,
    )

    # 3. Products (300 products)
    products = []
    product_tuples = []
    pid_counter = 0

    for cat_l1, cat_l2, items, price_range in CN_CATEGORIES:
        for item_name in items:
            for spec in ["【标准装】", "【家庭量贩装】", "【礼盒款】"]:
                pid_counter += 1
                pid = f"prod_{pid_counter:04d}"
                full_name = f"{item_name} {spec}"
                price = round(rng.uniform(price_range[0], price_range[1]), 2)
                products.append((pid, cat_l1, cat_l2, full_name, price))
                product_tuples.append((pid, cat_l1, cat_l2, full_name, price))

    conn.executemany(
        "INSERT INTO cn_products (product_id, category_l1, category_l2, product_name, base_price) VALUES (?, ?, ?, ?, ?)",
        products,
    )

    # 4. Orders, Items, Logistics, Reviews, Aftersales
    orders, order_items, logistics, reviews, aftersales = [], [], [], [], []
    order_counter = 0
    item_counter = 0
    review_counter = 0
    refund_counter = 0

    total_days = (CN_DATA_END - CN_DATA_START).days
    for day_offset in range(total_days):
        cur_date = CN_DATA_START + timedelta(days=day_offset)
        # E-commerce promotion surges (e.g. 618 Mid-Year festival in June, Double 11 preview)
        daily_count = rng.randint(45, 80)
        if cur_date.month == 6 and 15 <= cur_date.day <= 20:
            daily_count += 60  # 618 promotion peak
        elif cur_date.month in (7, 8):
            daily_count += 20  # Summer surge

        for _ in range(daily_count):
            order_counter += 1
            oid = f"SO{cur_date.strftime('%Y%m%d')}{order_counter:05d}"
            user_uid, user_prov, user_city, _ = rng.choice(user_tuples)
            channel = rng.choices(CHANNELS, weights=[50, 25, 15, 10])[0]
            order_time = cur_date + timedelta(
                hours=rng.randint(8, 23),
                minutes=rng.randint(0, 59),
                seconds=rng.randint(0, 59),
            )

            # Anomaly 1: Zhongtong Express Congestion in East China (Zhejiang/Shanghai) during July - August 2024
            is_zt_anomaly = (
                user_prov in ("上海市", "浙江省")
                and cur_date.year == 2024
                and cur_date.month in (7, 8)
                and rng.random() < 0.60
            )

            # Pick carrier
            if is_zt_anomaly:
                carrier = "中通快递"
            else:
                carrier = rng.choices(CARRIERS, weights=[35, 25, 20, 12, 8])[0]

            # Generate order items (1 to 4 items per order)
            num_items = rng.choices([1, 2, 3, 4], weights=[65, 22, 10, 3])[0]
            order_total = 0.0
            order_has_fresh_anomaly = False

            for seq in range(1, num_items + 1):
                item_counter += 1
                iid = f"item_{item_counter:06d}"

                # Anomaly 2: Fresh Food packaging defect from seller_0028
                is_fresh_defect = (
                    rng.random() < 0.12
                    and cur_date.month in (6, 7, 8)
                )

                if is_fresh_defect and seq == 1:
                    seller_id = "seller_0028"  # 天府冷链专营店
                    # Pick fresh food product
                    fresh_prods = [p for p in product_tuples if p[1] == "生鲜食品"]
                    prod_id, _, _, _, base_price = rng.choice(fresh_prods)
                    order_has_fresh_anomaly = True
                else:
                    prod_id, _, _, _, base_price = rng.choice(product_tuples)
                    seller_id = rng.choice(seller_ids)[0]

                qty = rng.choices([1, 2, 3], weights=[80, 15, 5])[0]
                unit_price = base_price
                order_total += unit_price * qty

                order_items.append(
                    (iid, oid, prod_id, seller_id, qty, unit_price)
                )

            # Fulfillment & Logistics dates
            shipped_time = order_time + timedelta(hours=rng.randint(4, 24))
            promise_delivery = shipped_time + timedelta(days=2)  # 48h SLA standard

            order_status = "已妥投"
            route_status = "已签收"

            if is_zt_anomaly:
                # Carrier severe delivery delay (+4 to +8 days past promise delivery time!)
                delay_days = rng.randint(4, 8)
                actual_delivery = promise_delivery + timedelta(days=delay_days)
                route_status = "网点积压"
                order_status = "已妥投"
            elif order_has_fresh_anomaly and rng.random() < 0.65:
                actual_delivery = shipped_time + timedelta(days=1)
                order_status = "售后中"
            else:
                if rng.random() < 0.02:
                    order_status = "已取消"
                    actual_delivery = None
                elif rng.random() < 0.03:
                    order_status = "运输中"
                    actual_delivery = None
                    route_status = "正常在途"
                else:
                    # Normal fulfillment within SLA
                    actual_delivery = shipped_time + timedelta(hours=rng.randint(18, 36))
                    route_status = "已签收"

            orders.append(
                (
                    oid,
                    user_uid,
                    order_status,
                    order_time.strftime("%Y-%m-%d %H:%M:%S"),
                    round(order_total, 2),
                    channel,
                )
            )

            logistics.append(
                (
                    f"LOG{oid[2:]}",
                    oid,
                    carrier,
                    shipped_time.strftime("%Y-%m-%d %H:%M:%S"),
                    promise_delivery.strftime("%Y-%m-%d %H:%M:%S"),
                    actual_delivery.strftime("%Y-%m-%d %H:%M:%S") if actual_delivery else None,
                    route_status,
                )
            )

            # Reviews & Customer Feedback
            if order_status in ("已妥投", "售后中"):
                review_counter += 1
                rid = f"REV{order_counter:06d}"

                if is_zt_anomaly:
                    rating = rng.choices([1, 2], weights=[85, 15])[0]
                    tag = "时效迟缓/分拨中心积压"
                    comment = rng.choice([
                        f"中通快递太垃圾了！物流卡在转运中心整整{delay_days}天都不动，急用完全耽误事，客服一直机器人回复！",
                        f"货老是迟到买家很不爽，承诺48小时送到，结果拖了一周多才派件，包装都压脏了！",
                        "物流时效极慢，中通在分拨中心积压严重，给差评绝不姑息！",
                        "催了三次单都没人理，中通快递慢如蜗牛，以后再也不在这个平台买了！",
                    ])
                elif order_has_fresh_anomaly:
                    rating = 1
                    tag = "外包装挤压破损/漏液"
                    comment = rng.choice([
                        "收到的时候泡沫箱都压扁破损了，里面的冰袋化成水漏得满地都是，海鲜全变质发臭了，赶紧退款！",
                        "天府冷链专营店包装极其敷衍，生鲜箱子碎了，里面的水果摔烂了一半，必须退货退款！",
                        "外包装破损严重，生鲜完全变质，这包装工艺也有脸发货？差评！",
                        "包装箱漏水，货品破损发臭，要求商家立刻全额退款并补偿！",
                    ])
                else:
                    # Normal distribution
                    rating = rng.choices([5, 4, 3, 2, 1], weights=[72, 18, 5, 3, 2])[0]
                    if rating >= 4:
                        tag = "优质推荐/时效迅捷"
                        comment = rng.choice([
                            f"{carrier}次日就妥投送达了，包装严实完好，商品成色非常棒，很满意！",
                            "发货迅速，顺丰送货上门，品质正品，性价比超高，全五星好评！",
                            "大促期间物流速度依然给力，商品与描述完全一致，下次还会回购！",
                            "质量很好，做工精细，给家人买的非常喜欢，物流也非常快！",
                        ])
                    else:
                        tag = "常规建议"
                        comment = rng.choice([
                            "东西一般般吧，性价比不算太高，物流速度中规中矩。",
                            "稍微有点色差，不过勉强能用，希望能加强品控。",
                            "外包装稍微有点压痕，里面的商品还好，先用用看吧。",
                        ])

                reviews.append((rid, oid, rating, tag, comment))

            # Aftersales Refunds
            if order_has_fresh_anomaly or (is_zt_anomaly and rng.random() < 0.25) or (rng.random() < 0.04):
                refund_counter += 1
                rf_id = f"REF{refund_counter:06d}"
                if order_has_fresh_anomaly:
                    rf_type = "退货退款"
                    rf_reason = "商品质量问题/外包装破损"
                    rf_amt = round(order_total, 2)
                    rf_status = "已退款"
                elif is_zt_anomaly:
                    rf_type = "仅退款"
                    rf_reason = "未按约定时间发货/物流严重延误"
                    rf_amt = round(order_total * 0.3, 2)  # Delay compensation
                    rf_status = "已退款"
                else:
                    rf_type = rng.choice(["仅退款", "退货退款"])
                    rf_reason = rng.choice(["不喜欢/效果差", "商品质量问题", "多买/拍错"])
                    rf_amt = round(order_total * rng.uniform(0.5, 1.0), 2)
                    rf_status = rng.choices(["已退款", "审核中"], weights=[85, 15])[0]

                aftersales.append((rf_id, oid, rf_type, rf_reason, rf_amt, rf_status))

    # Bulk insert all generated rows
    conn.executemany(
        "INSERT INTO cn_orders (order_id, user_id, order_status, order_time, total_amount, channel) VALUES (?, ?, ?, ?, ?, ?)",
        orders,
    )
    conn.executemany(
        "INSERT INTO cn_order_items (item_id, order_id, product_id, seller_id, quantity, unit_price) VALUES (?, ?, ?, ?, ?, ?)",
        order_items,
    )
    conn.executemany(
        "INSERT INTO cn_logistics (logistics_id, order_id, carrier, shipped_time, promise_delivery_time, actual_delivery_time, route_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        logistics,
    )
    conn.executemany(
        "INSERT INTO cn_reviews (review_id, order_id, rating, tags, comment_text) VALUES (?, ?, ?, ?, ?)",
        reviews,
    )
    conn.executemany(
        "INSERT INTO cn_aftersales (refund_id, order_id, refund_type, refund_reason, refund_amount, status) VALUES (?, ?, ?, ?, ?, ?)",
        aftersales,
    )
