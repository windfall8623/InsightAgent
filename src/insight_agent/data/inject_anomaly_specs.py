"""Inject declarative synthetic anomaly scenarios into cn_retail.db idempotently.

Populates real database rows for all 8 positive AnomalySpecs:
1. zt_express_delay_east_cn (Already in base seed, validated)
2. tianfu_fresh_packaging_defect (Already in base seed, validated)
3. remote_freight_rate_surge (Xinjiang/Tibet/Qinghai freight rule misconfiguration)
4. live_streaming_conversion_collapse (Live streaming impulse buying & instant refunds)
5. digital_3c_batch_defect (Computer accessories controller chip overheating defects)
6. brushing_fake_orders_anomaly (Baby & toys fake order brushing and auto-cancellations)
7. appliance_aftersales_delay (Home appliance post-delivery installation delay)
8. cosmetics_price_stockout (Cosmetics promotional mispricing, stockout & delay)

Preserves normal operational baselines for Controls 9 & 10 (Shanghai May & Guangdong Sept).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from insight_agent.data.cn_retail_db import default_cn_retail_db_path


def inject_standard_anomalies(db_path: Path | None = None) -> dict[str, int]:
    """Inject anomaly rows into SQLite database in read-write mode."""
    path = db_path or default_cn_retail_db_path()
    stats: dict[str, int] = {}

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    try:
        # Check if already injected
        check = cursor.execute("SELECT COUNT(*) FROM cn_orders WHERE order_id LIKE 'ord_anomaly_%'").fetchone()[0]
        if check > 0:
            cursor.execute("DELETE FROM cn_aftersales WHERE order_id LIKE 'ord_anomaly_%'")
            cursor.execute("DELETE FROM cn_reviews WHERE order_id LIKE 'ord_anomaly_%'")
            cursor.execute("DELETE FROM cn_logistics WHERE order_id LIKE 'ord_anomaly_%'")
            cursor.execute("DELETE FROM cn_order_items WHERE order_id LIKE 'ord_anomaly_%'")
            cursor.execute("DELETE FROM cn_orders WHERE order_id LIKE 'ord_anomaly_%'")
            cursor.execute("DELETE FROM cn_users WHERE user_id LIKE 'usr_anomaly_%'")
            cursor.execute("DELETE FROM cn_products WHERE product_id LIKE 'prod_anomaly_%'")
            cursor.execute("DELETE FROM cn_sellers WHERE seller_id LIKE 'seller_anomaly_%'")
            conn.commit()

        # -------------------------------------------------------------------
        # Spec 3: remote_freight_rate_surge (Xinjiang, Tibet, Qinghai Freight)
        # -------------------------------------------------------------------
        for i in range(1, 16):
            prov = "新疆维吾尔自治区" if i <= 6 else ("西藏自治区" if i <= 11 else "青海省")
            cursor.execute(
                "INSERT INTO cn_users (user_id, province, city, member_level) VALUES (?, ?, ?, ?)",
                (f"usr_anomaly_remote_{i:03d}", prov, "乌鲁木齐市" if prov == "新疆维吾尔自治区" else "拉萨市", "普客"),
            )

        for i in range(1, 36):
            oid = f"ord_anomaly_freight_{i:03d}"
            uid = f"usr_anomaly_remote_{(i % 15) + 1:03d}"
            otime = f"2024-09-{((i % 25) + 1):02d} 14:20:00" if i <= 20 else f"2024-10-{((i % 20) + 1):02d} 10:15:00"
            status = "已取消" if i <= 28 else "已妥投"
            amt = 120.0 + i * 2.0
            cursor.execute(
                "INSERT INTO cn_orders (order_id, user_id, order_status, order_time, total_amount, channel) VALUES (?, ?, ?, ?, ?, ?)",
                (oid, uid, status, otime, amt, "淘宝网页版"),
            )
            # order item
            cursor.execute(
                "INSERT INTO cn_order_items (item_id, order_id, product_id, seller_id, quantity, unit_price) VALUES (?, ?, ?, ?, ?, ?)",
                (f"item_anom_fr_{i:03d}", oid, "prod_0001", "seller_0001", 1, amt),
            )
            # logistics
            cursor.execute(
                "INSERT INTO cn_logistics (logistics_id, order_id, carrier, shipped_time, promise_delivery_time, actual_delivery_time, route_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"log_anom_fr_{i:03d}", oid, "顺丰速运", otime, otime, None if status == "已取消" else otime, "已取消" if status == "已取消" else "已签收"),
            )
            if status == "已取消":
                cursor.execute(
                    "INSERT INTO cn_aftersales (refund_id, order_id, refund_type, refund_reason, refund_amount, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (f"ref_anom_fr_{i:03d}", oid, "仅退款", "未按约定时间发货/运费规则争议偏远地区运费过高", amt, "已退款"),
                )
                cursor.execute(
                    "INSERT INTO cn_reviews (review_id, order_id, rating, tags, comment_text) VALUES (?, ?, ?, ?, ?)",
                    (f"rev_anom_fr_{i:03d}", oid, 1, "运费过高/规则争议", "新疆西藏偏远地区阶梯运费加价异常，运费甚至超过商品货值，运费规则配置失误买家取消退单！"),
                )
        stats["freight_orders"] = 35

        # -------------------------------------------------------------------
        # Spec 4: live_streaming_conversion_collapse (直播电商)
        # -------------------------------------------------------------------
        for i in range(1, 51):
            oid = f"ord_anomaly_live_{i:03d}"
            uid = f"user_{((i * 7) % 500) + 1:05d}"
            otime = f"2024-05-{((i % 25) + 1):02d} 20:30:00" if i <= 25 else f"2024-06-{((i % 25) + 1):02d} 21:45:00"
            status = "售后中" if i <= 36 else "已妥投"
            amt = 89.0 + (i % 50)
            cursor.execute(
                "INSERT INTO cn_orders (order_id, user_id, order_status, order_time, total_amount, channel) VALUES (?, ?, ?, ?, ?, ?)",
                (oid, uid, status, otime, amt, "直播电商"),
            )
            cursor.execute(
                "INSERT INTO cn_order_items (item_id, order_id, product_id, seller_id, quantity, unit_price) VALUES (?, ?, ?, ?, ?, ?)",
                (f"item_anom_live_{i:03d}", oid, "prod_0010", "seller_0005", 1, amt),
            )
            cursor.execute(
                "INSERT INTO cn_logistics (logistics_id, order_id, carrier, shipped_time, promise_delivery_time, actual_delivery_time, route_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"log_anom_live_{i:03d}", oid, "圆通速递", otime, otime, otime, "已签收"),
            )
            if status == "售后中":
                cursor.execute(
                    "INSERT INTO cn_aftersales (refund_id, order_id, refund_type, refund_reason, refund_amount, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (f"ref_anom_live_{i:03d}", oid, "仅退款", "直播电商渠道虚假宣传/冲动消费收货即仅退款", amt, "已退款"),
                )
                cursor.execute(
                    "INSERT INTO cn_reviews (review_id, order_id, rating, tags, comment_text) VALUES (?, ?, ?, ?, ?)",
                    (f"rev_anom_live_{i:03d}", oid, 1, "直播渠道夸大/虚假宣传", "直播带货主播虚假宣传，直播电商渠道低质流量冲动下单，拿到手货不对板，仅退款退单率极高！"),
                )
        stats["live_streaming_orders"] = 50

        # -------------------------------------------------------------------
        # -------------------------------------------------------------------
        # Spec 5: digital_3c_batch_defect (数码3C / 电脑配件)
        # -------------------------------------------------------------------
        cursor.execute(
            "INSERT INTO cn_sellers (seller_id, seller_name, province, city) VALUES (?, ?, ?, ?)",
            ("seller_anomaly_3c", "联创数码配件专营店", "广东省", "深圳市"),
        )
        cursor.execute(
            "INSERT INTO cn_products (product_id, category_l1, category_l2, product_name, base_price) VALUES (?, ?, ?, ?, ?)",
            ("prod_anomaly_3c", "数码3C", "电脑配件", "雷电4多功能扩展坞", 299.0),
        )
        for i in range(1, 46):
            oid = f"ord_anomaly_3c_{i:03d}"
            uid = f"user_{((i * 11) % 500) + 1:05d}"
            otime = f"2024-03-{((i % 25) + 1):02d} 11:00:00" if i <= 22 else f"2024-04-{((i % 25) + 1):02d} 15:30:00"
            status = "售后中" if i <= 34 else "已妥投"
            amt = 299.0
            cursor.execute(
                "INSERT INTO cn_orders (order_id, user_id, order_status, order_time, total_amount, channel) VALUES (?, ?, ?, ?, ?, ?)",
                (oid, uid, status, otime, amt, "天猫App"),
            )
            cursor.execute(
                "INSERT INTO cn_order_items (item_id, order_id, product_id, seller_id, quantity, unit_price) VALUES (?, ?, ?, ?, ?, ?)",
                (f"item_anom_3c_{i:03d}", oid, "prod_anomaly_3c", "seller_anomaly_3c", 1, amt),
            )
            cursor.execute(
                "INSERT INTO cn_logistics (logistics_id, order_id, carrier, shipped_time, promise_delivery_time, actual_delivery_time, route_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"log_anom_3c_{i:03d}", oid, "顺丰速运", otime, otime, otime, "已签收"),
            )
            if status == "售后中":
                cursor.execute(
                    "INSERT INTO cn_aftersales (refund_id, order_id, refund_type, refund_reason, refund_amount, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (f"ref_anom_3c_{i:03d}", oid, "退货退款", "商品质量问题/主控芯片发热故障开机黑屏缺陷", amt, "已退款"),
                )
                cursor.execute(
                    "INSERT INTO cn_reviews (review_id, order_id, rating, tags, comment_text) VALUES (?, ?, ?, ?, ?)",
                    (f"rev_anom_3c_{i:03d}", oid, 1, "开机故障/主控芯片缺陷", "数码3C电脑配件批次主控芯片发热故障缺陷，买家开机即黑屏，无法使用，集体退货！"),
                )
        stats["digital_3c_orders"] = 45

        # -------------------------------------------------------------------
        # Spec 6: brushing_fake_orders_anomaly (母婴用品)
        # -------------------------------------------------------------------
        cursor.execute(
            "INSERT INTO cn_products (product_id, category_l1, category_l2, product_name, base_price) VALUES (?, ?, ?, ?, ?)",
            ("prod_anomaly_toy", "母婴用品", "益智玩具", "全自动儿童早教积木机", 158.0),
        )
        for i in range(1, 41):
            oid = f"ord_anomaly_toy_{i:03d}"
            uid = f"user_{((i * 13) % 400) + 1:05d}"
            otime = f"2024-11-{((i % 25) + 1):02d} 03:15:00" if i <= 20 else f"2024-12-{((i % 25) + 1):02d} 04:20:00"
            status = "已取消" if i <= 32 else "已妥投"
            amt = 158.0
            cursor.execute(
                "INSERT INTO cn_orders (order_id, user_id, order_status, order_time, total_amount, channel) VALUES (?, ?, ?, ?, ?, ?)",
                (oid, uid, status, otime, amt, "微信小程序"),
            )
            cursor.execute(
                "INSERT INTO cn_order_items (item_id, order_id, product_id, seller_id, quantity, unit_price) VALUES (?, ?, ?, ?, ?, ?)",
                (f"item_anom_toy_{i:03d}", oid, "prod_anomaly_toy", "seller_0003", 1, amt),
            )
            cursor.execute(
                "INSERT INTO cn_logistics (logistics_id, order_id, carrier, shipped_time, promise_delivery_time, actual_delivery_time, route_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"log_anom_toy_{i:03d}", oid, "圆通速递", otime, otime, otime, "已签收"),
            )
            if status == "已取消":
                cursor.execute(
                    "INSERT INTO cn_aftersales (refund_id, order_id, refund_type, refund_reason, refund_amount, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (f"ref_anom_toy_{i:03d}", oid, "退货退款", "虚假交易/母婴玩具品类黑产刷单未实际交付自动退款", amt, "已退款"),
                )
                cursor.execute(
                    "INSERT INTO cn_reviews (review_id, order_id, rating, tags, comment_text) VALUES (?, ?, ?, ?, ?)",
                    (f"rev_anom_toy_{i:03d}", oid, 1, "异常刷单/未发货拒收", "母婴玩具品类异常刷单集聚与拒收退单，黑产虚假交易未交付触发批量拦截！"),
                )
        stats["fake_orders"] = 40

        # -------------------------------------------------------------------
        # Spec 7: appliance_aftersales_delay (家居家装 / 家用电器)
        # -------------------------------------------------------------------
        cursor.execute(
            "INSERT INTO cn_products (product_id, category_l1, category_l2, product_name, base_price) VALUES (?, ?, ?, ?, ?)",
            ("prod_anomaly_appliance", "家居家装", "家用电器", "变频一级冷暖静音立式空调", 3599.0),
        )
        for i in range(1, 41):
            oid = f"ord_anomaly_app_{i:03d}"
            uid = f"user_{((i * 17) % 400) + 1:05d}"
            otime = f"2024-10-{((i % 25) + 1):02d} 16:00:00" if i <= 20 else f"2024-11-{((i % 25) + 1):02d} 18:00:00"
            status = "售后中" if i <= 30 else "已妥投"
            amt = 3599.0
            cursor.execute(
                "INSERT INTO cn_orders (order_id, user_id, order_status, order_time, total_amount, channel) VALUES (?, ?, ?, ?, ?, ?)",
                (oid, uid, status, otime, amt, "天猫App"),
            )
            cursor.execute(
                "INSERT INTO cn_order_items (item_id, order_id, product_id, seller_id, quantity, unit_price) VALUES (?, ?, ?, ?, ?, ?)",
                (f"item_anom_app_{i:03d}", oid, "prod_anomaly_appliance", "seller_0002", 1, amt),
            )
            cursor.execute(
                "INSERT INTO cn_logistics (logistics_id, order_id, carrier, shipped_time, promise_delivery_time, actual_delivery_time, route_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"log_anom_app_{i:03d}", oid, "京东物流", otime, otime, otime, "已签收"),
            )
            if status == "售后中":
                cursor.execute(
                    "INSERT INTO cn_aftersales (refund_id, order_id, refund_type, refund_reason, refund_amount, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (f"ref_anom_app_{i:03d}", oid, "退货退款", "售后服务脱节/大家电售后上门安装超时积怨", amt, "已退款"),
                )
                cursor.execute(
                    "INSERT INTO cn_reviews (review_id, order_id, rating, tags, comment_text) VALUES (?, ?, ?, ?, ?)",
                    (f"rev_anom_app_{i:03d}", oid, 1, "售后安装严重延误", "家用电器大家电售后上门安装严重超时，送达后两周无人上门安装，家电售后安装延误引发集中退机！"),
                )
        stats["appliance_orders"] = 40

        # -------------------------------------------------------------------
        # Spec 8: cosmetics_price_stockout (美妆个护)
        # -------------------------------------------------------------------
        for i in range(1, 51):
            oid = f"ord_anomaly_cos_{i:03d}"
            uid = f"user_{((i * 19) % 500) + 1:05d}"
            otime = "2024-06-18 00:05:00"
            status = "已取消" if i <= 42 else "已妥投"
            amt = 19.9  # Pricing bug: 19.9 instead of 299
            cursor.execute(
                "INSERT INTO cn_orders (order_id, user_id, order_status, order_time, total_amount, channel) VALUES (?, ?, ?, ?, ?, ?)",
                (oid, uid, status, otime, amt, "天猫App"),
            )
            cursor.execute(
                "INSERT INTO cn_order_items (item_id, order_id, product_id, seller_id, quantity, unit_price) VALUES (?, ?, ?, ?, ?, ?)",
                (f"item_anom_cos_{i:03d}", oid, "prod_0043", "seller_0004", 1, amt),
            )
            cursor.execute(
                "INSERT INTO cn_logistics (logistics_id, order_id, carrier, shipped_time, promise_delivery_time, actual_delivery_time, route_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"log_anom_cos_{i:03d}", oid, "中通快递", otime, otime, None if status == "已取消" else otime, "缺货待发" if status == "已取消" else "已签收"),
            )
            if status == "已取消":
                cursor.execute(
                    "INSERT INTO cn_aftersales (refund_id, order_id, refund_type, refund_reason, refund_amount, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (f"ref_anom_cos_{i:03d}", oid, "仅退款", "商家标错价格/超卖缺货未按约定时间发货", amt, "已退款"),
                )
                cursor.execute(
                    "INSERT INTO cn_reviews (review_id, order_id, rating, tags, comment_text) VALUES (?, ?, ?, ?, ?)",
                    (f"rev_anom_cos_{i:03d}", oid, 1, "标错价超卖/缺货延迟发货", "美妆个护促销标错价格致超卖缺货与延迟发货，美妆缺货与标错价导致买家订单大量取消！"),
                )
        stats["cosmetics_orders"] = 50

        # -------------------------------------------------------------------
        # Spec 1 extension: East China Logistics - Jiangsu Province
        # (Nanjing & Suzhou Kunshan sorting hubs congestion)
        # -------------------------------------------------------------------
        for i in range(1, 16):
            city = "南京市" if i <= 8 else "苏州市"
            cursor.execute(
                "INSERT INTO cn_users (user_id, province, city, member_level) VALUES (?, ?, ?, ?)",
                (f"usr_anomaly_js_{i:03d}", "江苏省", city, "普客"),
            )

        for i in range(1, 36):
            oid = f"ord_anomaly_jszt_{i:03d}"
            uid = f"usr_anomaly_js_{(i % 15) + 1:03d}"
            otime = f"2024-07-{((i % 25) + 1):02d} 14:00:00" if i <= 18 else f"2024-08-{((i % 25) + 1):02d} 16:30:00"
            status = "已妥投"
            amt = 99.0 + i
            cursor.execute(
                "INSERT INTO cn_orders (order_id, user_id, order_status, order_time, total_amount, channel) VALUES (?, ?, ?, ?, ?, ?)",
                (oid, uid, status, otime, amt, "天猫App"),
            )
            cursor.execute(
                "INSERT INTO cn_order_items (item_id, order_id, product_id, seller_id, quantity, unit_price) VALUES (?, ?, ?, ?, ?, ?)",
                (f"item_anom_jszt_{i:03d}", oid, "prod_0001", "seller_0001", 1, amt),
            )
            shipped = otime
            promise = f"2024-07-{((i % 25) + 3):02d} 18:00:00" if i <= 18 else f"2024-08-{((i % 25) + 3):02d} 18:00:00"
            actual = f"2024-07-{((i % 25) + 7):02d} 18:00:00" if i <= 18 else f"2024-08-{((i % 25) + 7):02d} 18:00:00"
            cursor.execute(
                "INSERT INTO cn_logistics (logistics_id, order_id, carrier, shipped_time, promise_delivery_time, actual_delivery_time, route_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"log_anom_jszt_{i:03d}", oid, "中通快递", shipped, promise, actual, "网点积压"),
            )
            cursor.execute(
                "INSERT INTO cn_aftersales (refund_id, order_id, refund_type, refund_reason, refund_amount, status) VALUES (?, ?, ?, ?, ?, ?)",
                (f"ref_anom_jszt_{i:03d}", oid, "仅退款", "未按约定时间发货/物流严重延误网点积压", 20.0, "已退款"),
            )
            cursor.execute(
                "INSERT INTO cn_reviews (review_id, order_id, rating, tags, comment_text) VALUES (?, ?, ?, ?, ?)",
                (f"rev_anom_jszt_{i:03d}", oid, 1, "物流延误/网点积压", "中通快递南京分拨中心与苏州昆山网点包裹积压严重，停滞超过4天超时送达，催单无果！"),
            )
        stats["jiangsu_zt_orders"] = 35

        conn.commit()
    finally:
        conn.close()

    return stats


if __name__ == "__main__":
    s = inject_standard_anomalies()
    print("Injected standard anomaly cases:", s)
