"""Declarative Anomaly Specifications for CN-Retail Causal Diagnosis.

Provides a unified single source of truth for:
1. Synthetic anomaly injection into CN-Retail database tables (orders, logistics, reviews, refunds).
2. Ground-truth assertions for the independent attribution evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AnomalySpec:
    """Declarative specification defining an anomaly scenario and its expected diagnosis."""

    id: str
    name: str
    category: str  # "logistics" | "product_quality" | "freight" | "channel_conversion" | "batch_defect" | "fake_orders" | "aftersales_delay" | "no_anomaly"
    description: str
    window_months: tuple[int, ...]  # Active months (1-12)
    year: int = 2024
    provinces: tuple[str, ...] = ()  # Target provinces (empty = nationwide)
    categories_l1: tuple[str, ...] = ()  # Target category L1 (empty = all)
    target_seller_id: str | None = None
    target_seller_name: str | None = None
    target_carrier: str | None = None
    target_channel: str | None = None
    target_product_id: str | None = None
    probability: float = 0.5  # Infection probability among candidate rows
    min_effect_ratio: float = 2.5  # Relative effect size over same-class baseline
    min_evidence_orders: int = 10
    expect_dimensions: dict[str, str] = field(default_factory=dict)
    expect_entity: str = ""  # Canonical entity name/ID expected in root cause
    entity_aliases: tuple[str, ...] = ()  # Aliases/substrings acceptable as match
    diagnostic_queries: tuple[str, ...] = ()  # Benchmark query prompts testing this anomaly
    is_control: bool = False  # If True, represents normal baseline (no anomaly)

    def matches_order(
        self,
        order_date_year: int,
        order_date_month: int,
        province: str,
        channel: str,
    ) -> bool:
        """Check if an order falls into the spatial-temporal window of this spec."""
        if self.is_control:
            return False
        if order_date_year != self.year:
            return False
        if order_date_month not in self.window_months:
            return False
        if self.provinces and province not in self.provinces:
            return False
        if self.target_channel and channel != self.target_channel:
            return False
        return True


# ---------------------------------------------------------------------------
# Declarative Standard Registry of 10 Anomaly & Control Specs
# ---------------------------------------------------------------------------

STANDARD_ANOMALY_SPECS: tuple[AnomalySpec, ...] = (
    # 1. Existing Anomaly 1: Zhongtong Express Congestion in East China
    AnomalySpec(
        id="zt_express_delay_east_cn",
        name="中通快递华东分拨转运中心积压延误",
        category="logistics",
        description="2024年7-8月江浙沪地区中通承运订单分拨网点严重积压，派送时效超出承诺4-8天，引发超时客诉差评潮。",
        window_months=(7, 8),
        year=2024,
        provinces=("上海市", "浙江省", "江苏省"),
        target_carrier="中通快递",
        probability=0.65,
        min_effect_ratio=3.0,
        min_evidence_orders=25,
        expect_dimensions={"category": "logistics", "carrier": "中通快递"},
        expect_entity="中通快递",
        entity_aliases=("中通", "ZTO", "承运商中通", "中通快递华东分拨"),
        diagnostic_queries=(
            "分析2024年7-8月华东江浙沪地区订单履约时效大幅延误与差评激增的原因",
            "排查7月份上海和浙江地区大量包裹超过预计送达时效的物流根因",
            "为什么三季度江浙沪买家针对配送慢、网点滞留积压的客诉出现异常激增",
        ),
    ),
    # 2. Existing Anomaly 2: Fresh Food packaging structural defect (Tianfu Cold Chain)
    AnomalySpec(
        id="tianfu_fresh_packaging_defect",
        name="天府冷链生鲜保温泡沫箱破损漏液",
        category="product_quality",
        description="2024年6-8月天府冷链专营店生鲜泡沫箱外包装强度不足，运输颠簸导致大面积破损漏液发臭，触发破损退款激增。",
        window_months=(6, 7, 8),
        year=2024,
        categories_l1=("生鲜食品",),
        target_seller_id="seller_0028",
        target_seller_name="天府冷链专营店",
        probability=0.70,
        min_effect_ratio=3.5,
        min_evidence_orders=20,
        expect_dimensions={"category": "product_quality", "seller": "seller_0028", "category_l1": "生鲜食品"},
        expect_entity="天府冷链专营店",
        entity_aliases=("天府冷链", "seller_0028", "天府冷链专营店生鲜", "泡沫箱破损"),
        diagnostic_queries=(
            "分析2024年夏季生鲜食品品类退款率大幅攀升的质量与包装原因",
            "排查天府冷链专营店在6至8月份开箱变质、漏液破损差评集中的异动根因",
            "生鲜食品商户天府冷链退款激增，到底是什么环节出现了严重缺陷",
        ),
    ),
    # 3. Existing Category: Remote provinces freight rate rule misconfiguration
    AnomalySpec(
        id="remote_freight_rate_surge",
        name="西北偏远省份阶梯运费加价规则配置失误",
        category="freight",
        description="2024年9-10月新疆、西藏等偏远省份阶梯运费规则设置过高，运费甚至超过商品货值，引发大批未发货退款取消。",
        window_months=(9, 10),
        year=2024,
        provinces=("新疆维吾尔自治区", "西藏自治区", "青海省"),
        probability=0.60,
        min_effect_ratio=3.2,
        min_evidence_orders=15,
        expect_dimensions={"category": "freight", "provinces": "新疆,西藏"},
        expect_entity="偏远省份运费规则",
        entity_aliases=("运费", "阶梯运费", "偏远地区运费", "freight", "新疆运费"),
        diagnostic_queries=(
            "排查2024年9月西北偏远省份订单取消退款率异常飙升的根因",
            "为什么新疆西藏地区在秋季出现大量买家因运费过高而退单的异常",
            "分析偏远地区订单履约成本与买家流失异动的运费规则原因",
        ),
    ),
    # 4. New Category: Live Streaming Channel Conversion Collapse
    AnomalySpec(
        id="live_streaming_conversion_collapse",
        name="直播电商渠道冲动消费引发仅退款与履约崩塌",
        category="channel_conversion",
        description="2024年5-6月直播电商渠道引流劣质冲动消费客群，主播虚假宣传导致收货即退款率攀升至同类4倍。",
        window_months=(5, 6),
        year=2024,
        target_channel="直播电商",
        probability=0.55,
        min_effect_ratio=2.8,
        min_evidence_orders=30,
        expect_dimensions={"category": "channel_conversion", "channel": "直播电商"},
        expect_entity="直播电商",
        entity_aliases=("直播渠道", "直播带货", "直播电商渠道", "live_stream"),
        diagnostic_queries=(
            "分析2024年5-6月直播电商渠道订单售后退款率显著高于传统货架的原因",
            "为什么直播带货渠道在初夏转化率骤降且买家仅退款申请暴增",
            "排查线上销售渠道中直播电商退货异动与低质流量问题",
        ),
    ),
    # 5. New Category: Digital 3C Batch Component Defect
    AnomalySpec(
        id="digital_3c_batch_defect",
        name="数码3C电脑配件批次主控芯片发热故障缺陷",
        category="batch_defect",
        description="2024年3-4月电脑办公类目某批次电脑外设电源与主控元件批次缺陷，买家开机即黑屏，引发集中退换货。",
        window_months=(3, 4),
        year=2024,
        categories_l1=("数码3C", "电脑办公"),
        probability=0.50,
        min_effect_ratio=3.0,
        min_evidence_orders=20,
        expect_dimensions={"category": "batch_defect", "category_l1": "数码3C"},
        expect_entity="电脑办公批次质量缺陷",
        entity_aliases=("电脑配件", "电脑办公", "数码3C", "数码外设", "芯片故障", "批次质量", "扩展坞"),
        diagnostic_queries=(
            "分析2024年春季电脑办公类目开机故障与硬件瑕疵退换率飙升的原因",
            "排查数码配件在3月集中出现的批量无法开机质量缺陷归因",
            "为什么数码3C分类中电脑外设产品收到大面积硬件损坏客诉",
        ),
    ),
    # 6. New Category: Brushing / Fake Orders in Baby & Toys
    AnomalySpec(
        id="brushing_fake_orders_anomaly",
        name="母婴玩具品类异常刷单集聚与拒收退单",
        category="fake_orders",
        description="2024年11-12月母婴玩具品类某网点刷单虚假下单激增，后因未实际交付引发批量系统自动退单及拒收。",
        window_months=(11, 12),
        year=2024,
        categories_l1=("母婴用品", "母婴玩具"),
        probability=0.45,
        min_effect_ratio=2.6,
        min_evidence_orders=20,
        expect_dimensions={"category": "fake_orders", "category_l1": "母婴用品"},
        expect_entity="虚假刷单异常",
        entity_aliases=("刷单", "虚假交易", "异常下单", "母婴刷单", "母婴用品", "母婴玩具", "积木机"),
        diagnostic_queries=(
            "排查2024年底母婴玩具分类异常高发拒收退货的黑产刷单特征",
            "分析母婴玩具类目在年末退款激增且订单履约地址高度集中的原因",
            "为什么母婴品类短期内出现大量批量撤单与异常虚假妥投",
        ),
    ),
    # 7. New Category: Large Appliance Aftersales Installation Delay
    AnomalySpec(
        id="appliance_aftersales_delay",
        name="家用电器大家电售后上门安装超时积怨",
        category="aftersales_delay",
        description="2024年10-11月家用电器大促后售后网点安装师傅严重紧缺，买家签收后数周无人上门安装引发大面积差评与退机。",
        window_months=(10, 11),
        year=2024,
        categories_l1=("家居家装", "家用电器"),
        probability=0.55,
        min_effect_ratio=2.9,
        min_evidence_orders=20,
        expect_dimensions={"category": "aftersales_delay", "category_l1": "家居家装"},
        expect_entity="家电售后安装延误",
        entity_aliases=("大家电", "家用电器", "售后安装", "家用电器售后", "上门慢", "家电售后", "家居家装"),
        diagnostic_queries=(
            "排查2024年10月至11月家用电器大家电类目签收后差评暴涨的售后根因",
            "分析大家电用户反馈送达后无人上门安装导致退货退款激增的异动",
            "家用电器类目近期客诉集中爆发，到底是物流还是售后服务脱节",
        ),
    ),
    # 8. New Category: Cosmetics Price Misconfiguration & Stockout
    AnomalySpec(
        id="cosmetics_price_stockout",
        name="美妆个护促销标错价格致超卖缺货与延迟发货",
        category="stockout_misprice",
        description="2024年6月美妆个护类目大促期间折扣叠加导致价格标错，触发巨量薅羊毛订单超出库存能力，引发大面积未按约定发货退款。",
        window_months=(6,),
        year=2024,
        categories_l1=("美妆个护",),
        probability=0.50,
        min_effect_ratio=3.1,
        min_evidence_orders=25,
        expect_dimensions={"category": "stockout_misprice", "category_l1": "美妆个护"},
        expect_entity="美妆缺货与标错价",
        entity_aliases=("标错价", "美妆超卖", "缺货", "未按时发货", "美妆个护"),
        diagnostic_queries=(
            "分析2024年618大促期间美妆个护类目延迟发货与仅退款激增的原因",
            "排查美妆分类在6月份因为缺货超卖导致的买家集中投诉",
            "美妆个护品类订单大量取消，是否存在标错价格或库存超卖异动",
        ),
    ),
    # 9. Control 1: Normal Retail Baseline in Shanghai (No Anomaly)
    AnomalySpec(
        id="control_normal_shanghai_may",
        name="正常基线对照组：上海5月平稳运营无异常",
        category="no_anomaly",
        description="2024年5月上海地区各项物流履约与品类退款率均处于历史正常P50分位数内，无任何系统性异常。",
        window_months=(5,),
        year=2024,
        provinces=("上海市",),
        probability=0.0,
        min_effect_ratio=1.0,
        min_evidence_orders=0,
        expect_dimensions={"status": "normal"},
        expect_entity="正常大盘基线",
        entity_aliases=("正常", "无异常", "正常范围内", "平稳", "基线内", "no_anomaly"),
        diagnostic_queries=(
            "排查2024年5月份上海地区整体零售订单履约与售后退款是否存在异常",
            "分析上海市5月份各项业务运营指标的健康状态及潜在异常原因",
            "请诊断5月上海订单履约时效波动是否超出正常历史基线",
        ),
        is_control=True,
    ),
    # 10. Control 2: Normal Category Baseline in Guangdong (No Anomaly)
    AnomalySpec(
        id="control_normal_guangdong_sep",
        name="正常基线对照组：广东9月数码大盘常规波动无异常",
        category="no_anomaly",
        description="2024年9月广东省数码3C品类订单与售后处于大盘常规月度波动范围，无任何批次质量或承运商事故。",
        window_months=(9,),
        year=2024,
        provinces=("广东省",),
        categories_l1=("电脑办公", "手机数码"),
        probability=0.0,
        min_effect_ratio=1.0,
        min_evidence_orders=0,
        expect_dimensions={"status": "normal"},
        expect_entity="正常大盘基线",
        entity_aliases=("正常", "无异常", "常规波动", "未见异常", "符合预期", "no_anomaly"),
        diagnostic_queries=(
            "分析2024年9月广东省电脑办公与数码产品退款率波动的原因",
            "排查9月广东数码3C品类是否存在特定承运商或供应商事故",
            "诊断广东省秋季电子数码品类的客诉率是否处于正常业务基线内",
        ),
        is_control=True,
    ),
)
