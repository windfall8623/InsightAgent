"""Coreference Resolution Engine for E-Commerce Analytical Dialogues.

Resolves multi-turn conversational ambiguities before semantic planning:
1. Explicit Entity Substitution:
   "把刚才那个物流商换成顺丰再查一遍" -> resolves to "[原问题框架]，承运商换为顺丰"
   "换成上海呢" -> resolves to "[原指标]，省市换为上海"
2. Pronoun & Anaphora Resolution:
   "它们上一季度的退款率是多少" -> resolves antecedent entity from dialogue history (e.g. "数码3C")
   "它的平均客单价呢" -> resolves "它" to the preceding focus entity
   "该商家的差评率" -> resolves to "seller_xxx 的差评率"
3. Elliptical Follow-ups:
   "那顺丰呢？" -> inherits metric, time window, and dimension from prior query
   "那华南呢？" -> inherits metric and filters from prior query
4. Cross-Dataset Support:
   Supports both CN-Retail (Chinese supply chain entities) and Olist (Portuguese/English e-commerce).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from insight_agent.core.llm import Message

logger = logging.getLogger(__name__)


# Standard entities lexicon for fast deterministic resolution
KNOWN_CARRIERS = {
    "中通": "中通",
    "顺丰": "顺丰",
    "京东": "京东",
    "圆通": "圆通",
    "极兔": "极兔",
    "韵达": "韵达",
    "申通": "申通",
    "邮政": "邮政",
    "ems": "EMS",
    "德邦": "德邦",
    "correios": "correios",
}

KNOWN_PROVINCES = {
    "上海": "上海",
    "北京": "北京",
    "广东": "广东",
    "浙江": "浙江",
    "江苏": "江苏",
    "山东": "山东",
    "河南": "河南",
    "四川": "四川",
    "湖北": "湖北",
    "福建": "福建",
    "湖南": "湖南",
    "安徽": "安徽",
    "河北": "河北",
    "陕西": "陕西",
    "江西": "江西",
    "辽宁": "辽宁",
    "云南": "云南",
    "广西": "广西",
    "山西": "山西",
    "重庆": "重庆",
    "黑龙江": "黑龙江",
    "吉林": "吉林",
    "天津": "天津",
    "内蒙古": "内蒙古",
    "贵州": "贵州",
    "新疆": "新疆",
    "甘肃": "甘肃",
    "海南": "海南",
    "宁夏": "宁夏",
    "青海": "青海",
    "西藏": "西藏",
    "sp": "SP",
    "sao paulo": "Sao Paulo",
    "rj": "RJ",
    "rio": "Rio de Janeiro",
    "rio de janeiro": "Rio de Janeiro",
    "mg": "MG",
}

KNOWN_CITIES = {
    "上海": "上海",
    "北京": "北京",
    "广州": "广州",
    "深圳": "深圳",
    "杭州": "杭州",
    "南京": "南京",
    "苏州": "苏州",
    "成都": "成都",
    "武汉": "武汉",
    "sao paulo": "sao paulo",
    "rio de janeiro": "rio de janeiro",
}

KNOWN_REGIONS = {
    "华东": "华东",
    "华南": "华南",
    "华北": "华北",
    "华中": "华中",
    "西南": "西南",
    "西北": "西北",
    "东北": "东北",
}

KNOWN_CATEGORIES = {
    "数码3c": "数码3C",
    "3c数码": "数码3C",
    "数码": "数码3C",
    "食品生鲜": "食品生鲜",
    "生鲜": "食品生鲜",
    "服饰鞋包": "服饰鞋包",
    "女装": "服饰鞋包",
    "美妆个护": "美妆个护",
    "美妆": "美妆个护",
    "家居日用": "家居日用",
    "家居": "家居日用",
    "母婴玩具": "母婴玩具",
    "母婴": "母婴玩具",
    "运动户外": "运动户外",
    "图书文娱": "图书文娱",
    "汽车用品": "汽车用品",
    "家用电器": "家用电器",
    "家电": "家用电器",
    "beleza_saude": "beleza_saude",
    "cama_mesa_banho": "cama_mesa_banho",
    "esporte_lazer": "esporte_lazer",
}

KNOWN_CHANNELS = {
    "淘宝": "淘宝",
    "天猫": "天猫",
    "京东": "京东",
    "拼多多": "拼多多",
    "抖音": "抖音",
    "快手": "快手",
    "小红书": "小红书",
    "微信小程序": "微信小程序",
    "线下门店": "线下门店",
}

KNOWN_METRICS = [
    "延误率",
    "延迟率",
    "退款率",
    "销售额",
    "gmv",
    "成交额",
    "销售总额",
    "订单量",
    "订单数",
    "客单价",
    "平均客单价",
    "差评率",
    "好评率",
    "妥投时效",
    "配送时长",
    "平均评分",
    "用户数",
    "买家数",
]

KNOWN_TIME_WINDOWS = [
    "最近一个月",
    "近一个月",
    "上月",
    "上个月",
    "本月",
    "当月",
    "这个月",
    "上季度",
    "上季",
    "本季度",
    "当季",
    "上半年",
    "下半年",
    "最近7天",
    "近7天",
    "最近30天",
    "近30天",
    "最近90天",
    "最近半年",
]


@dataclass(slots=True)
class CoreferenceResult:
    """Outcome of resolving multi-turn coreference and ellipsis."""

    resolved_query: str
    original_query: str
    is_rewritten: bool = False
    resolution_type: str = "none"  # "entity_replacement" | "pronoun_coreference" | "ellipsis_continuation" | "none"
    antecedent_entity: str | None = None
    substitute_entity: str | None = None
    entity_dimension: str | None = None  # "carrier" | "city" | "province" | "category" | "channel" | "seller"
    inherited_metrics: list[str] = field(default_factory=list)
    inherited_time_window: str | None = None
    standing_filters: dict[str, Any] = field(default_factory=dict)


class CoreferenceResolver:
    """Deterministic, sub-millisecond Coreference Resolution Engine.

    Empowers RulesPlanner with multi-turn pronoun, ellipsis, and entity substitution
    capabilities without requiring heavy LLMs or external networks.
    """

    def __init__(self) -> None:
        pass

    def _extract_entities_from_text(self, text: str) -> dict[str, Any]:
        """Detect entities, dimensions, metrics, and time windows from text."""
        t_low = text.lower().strip()
        entities: dict[str, Any] = {}

        # 1. Carrier
        for k, v in KNOWN_CARRIERS.items():
            if k in t_low:
                entities["carrier"] = v
                break

        # 2. Province / Region
        for k, v in KNOWN_PROVINCES.items():
            if k in t_low:
                entities["province"] = v
                break
        if "province" not in entities:
            for k, v in KNOWN_REGIONS.items():
                if k in t_low:
                    entities["region"] = v
                    break

        # 3. City
        for k, v in KNOWN_CITIES.items():
            if k in t_low:
                entities["city"] = v
                break

        # 4. Category
        for k, v in KNOWN_CATEGORIES.items():
            if k in t_low:
                entities["category"] = v
                break

        # 5. Channel
        for k, v in KNOWN_CHANNELS.items():
            if k in t_low:
                entities["channel"] = v
                break

        # 6. Seller
        seller_m = re.search(r"seller_\d+", text, re.IGNORECASE)
        if seller_m:
            entities["seller"] = seller_m.group(0)

        # 7. Metrics
        metrics: list[str] = []
        for m in KNOWN_METRICS:
            if m in t_low:
                metrics.append(m)
        if metrics:
            entities["metrics"] = metrics

        # 8. Time window
        for tw in KNOWN_TIME_WINDOWS:
            if tw in t_low:
                entities["time_window"] = tw
                break

        return entities

    def _find_antecedents(
        self,
        history: Sequence[Message] | list[dict[str, Any]] | None,
        standing_filters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Traverse dialogue history backwards to identify the latest active context."""
        context: dict[str, Any] = {}
        if standing_filters:
            context.update(standing_filters)

        if not history:
            return context

        # Iterate history backwards
        for item in reversed(history):
            if isinstance(item, Message):
                role = item.role
                content = item.content
            elif isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
            else:
                continue

            if role == "user" and content:
                extracted = self._extract_entities_from_text(content)
                for k, v in extracted.items():
                    if k not in context:
                        context[k] = v
                if "last_user_query" not in context:
                    context["last_user_query"] = content

            elif role == "assistant" and content:
                # Check for SQL or structured mention in assistant response
                if "last_assistant_answer" not in context:
                    context["last_assistant_answer"] = content
                # Extract SQL where conditions if present
                sql_match = re.search(r"WHERE\s+([^\n;]+)", content, re.IGNORECASE)
                if sql_match:
                    where_clause = sql_match.group(1)
                    if "carrier" not in context:
                        c_m = re.search(r"carrier\s*=\s*'([^']+)'", where_clause)
                        if c_m:
                            context["carrier"] = c_m.group(1)
                    if "province" not in context:
                        p_m = re.search(r"province\s*=\s*'([^']+)'", where_clause)
                        if p_m:
                            context["province"] = p_m.group(1)
                    if "category" not in context:
                        cat_m = re.search(r"category(?:_l1)?\s*=\s*'([^']+)'", where_clause)
                        if cat_m:
                            context["category"] = cat_m.group(1)

        return context

    def resolve(
        self,
        question: str,
        *,
        history: Sequence[Message] | list[dict[str, Any]] | None = None,
        standing_filters: dict[str, Any] | None = None,
    ) -> CoreferenceResult:
        """Resolve pronouns, ellipses, and substitutions in the user query."""
        raw_q = question.strip()

        # If empty query, return verbatim
        if not raw_q:
            return CoreferenceResult(resolved_query=raw_q, original_query=raw_q)

        # Retrieve antecedent context from history and standing filters
        antecedents = self._find_antecedents(history, standing_filters)
        current_entities = self._extract_entities_from_text(raw_q)

        # ------------------------------------------------------------------
        # Pattern 1: Explicit Entity Substitution
        # "把刚才那个物流商换成顺丰再查一遍"
        # "换成顺丰呢" / "改成顺丰" / "把城市换成上海"
        # ------------------------------------------------------------------
        subst_patterns = [
            # 把刚才那个[物流商/省份/类目]换成[新实体]
            r"(?:把|将)?\s*(?:刚才(?:那个)?|之前的?|上一次的?)?\s*(物流商|快递|承运商|城市|省份|地区|类目|品类|渠道|店铺|商家)?\s*(?:换成|改成|替换为|改为)\s*([^\s，。？！,\?!]+)(?:.*(?:再查|查一下|看看|分析))?",
            # 换成[新实体]呢 / 改成[新实体]查一下
            r"^(?:换成|改成|换做|换为)\s*([^\s，。？！,\?!]+)(?:呢|查一下|看看)?$",
        ]

        for pat in subst_patterns:
            m = re.search(pat, raw_q)
            if m:
                groups = m.groups()
                new_entity_text = groups[-1]
                dim_hint = groups[0] if len(groups) > 1 and groups[0] else ""

                new_extracted = self._extract_entities_from_text(new_entity_text)
                substitute_dim = None
                substitute_val = None

                for d in ("carrier", "province", "city", "region", "category", "channel", "seller"):
                    if d in new_extracted:
                        substitute_dim = d
                        substitute_val = new_extracted[d]
                        break

                if not substitute_val and dim_hint:
                    # Match dimension from hint
                    if any(w in dim_hint for w in ("物流", "快递", "承运")):
                        substitute_dim = "carrier"
                    elif any(w in dim_hint for w in ("城市", "省份", "地区")):
                        substitute_dim = "province" if "省" in dim_hint else "city"
                    elif any(w in dim_hint for w in ("类目", "品类")):
                        substitute_dim = "category"
                    substitute_val = new_entity_text

                if substitute_val and substitute_dim:
                    # Retrieve base query to substitute into
                    last_q = antecedents.get("last_user_query", "")
                    old_val = antecedents.get(substitute_dim)

                    rewritten = ""
                    if last_q and old_val and old_val in last_q:
                        # Direct template substitution
                        rewritten = last_q.replace(old_val, substitute_val)
                    else:
                        # Construct from inherited metrics & time
                        metrics_str = "、".join(antecedents.get("metrics", ["销售额和订单量"]))
                        time_str = antecedents.get("time_window", "")
                        rewritten = f"查看{time_str}{substitute_val}的{metrics_str}"

                    new_filters = dict(standing_filters or {})
                    new_filters[substitute_dim] = substitute_val

                    return CoreferenceResult(
                        resolved_query=rewritten.strip(),
                        original_query=raw_q,
                        is_rewritten=True,
                        resolution_type="entity_replacement",
                        antecedent_entity=old_val,
                        substitute_entity=substitute_val,
                        entity_dimension=substitute_dim,
                        inherited_metrics=antecedents.get("metrics", []),
                        inherited_time_window=antecedents.get("time_window"),
                        standing_filters=new_filters,
                    )

        # ------------------------------------------------------------------
        # Pattern 2: Elliptical Follow-ups
        # "那顺丰呢？" / "顺丰呢" / "那上海呢？" / "华南呢？"
        # ------------------------------------------------------------------
        ellipsis_patterns = [
            r"^(?:那|那么)?\s*([^\s，。？！,\?!]+)\s*呢[？\?]?$",
            r"^那\s*([^\s，。？！,\?!]+)[？\?]?$",
        ]
        for pat in ellipsis_patterns:
            m = re.match(pat, raw_q)
            if m:
                target_word = m.group(1).strip()
                extracted = self._extract_entities_from_text(target_word)

                target_dim = None
                target_val = None
                for d in ("carrier", "province", "city", "region", "category", "channel", "seller"):
                    if d in extracted:
                        target_dim = d
                        target_val = extracted[d]
                        break

                if target_val and target_dim:
                    last_q = antecedents.get("last_user_query", "")
                    old_val = antecedents.get(target_dim)

                    is_replacement_query = any(k in last_q for k in ("换成", "改成", "刚才", "换为", "替换", "再查"))
                    if last_q and old_val and old_val in last_q and not is_replacement_query:
                        rewritten = last_q.replace(old_val, target_val)
                    else:
                        metrics_str = "、".join(antecedents.get("metrics", ["数据统计"]))
                        time_str = antecedents.get("time_window", "")
                        rewritten = f"查看{time_str}{target_val}的{metrics_str}"

                    new_filters = dict(standing_filters or {})
                    new_filters[target_dim] = target_val

                    return CoreferenceResult(
                        resolved_query=rewritten.strip(),
                        original_query=raw_q,
                        is_rewritten=True,
                        resolution_type="ellipsis_continuation",
                        antecedent_entity=old_val,
                        substitute_entity=target_val,
                        entity_dimension=target_dim,
                        inherited_metrics=antecedents.get("metrics", []),
                        inherited_time_window=antecedents.get("time_window"),
                        standing_filters=new_filters,
                    )

        # ------------------------------------------------------------------
        # Pattern 3: Pronouns & Anaphoric Expressions
        # "它们上一季度的退款率是多少" / "它的平均客单价呢" / "该商家的退款率"
        # ------------------------------------------------------------------
        pronoun_match = re.search(
            r"(它们|他们|它|这几家|这些|该商[家户]|该物流商|该承运商|该省份|该城市|该类目|刚才那个(?:物流商|商家|类目|省份|城市)?)",
            raw_q,
        )
        if pronoun_match:
            pronoun = pronoun_match.group(1)

            # Determine which antecedent entity applies
            resolved_antecedent = None
            resolved_dim = None

            if any(w in pronoun for w in ("物流", "承运", "快递")):
                resolved_antecedent = antecedents.get("carrier")
                resolved_dim = "carrier"
            elif any(w in pronoun for w in ("商", "店铺")):
                resolved_antecedent = antecedents.get("seller")
                resolved_dim = "seller"
            elif any(w in pronoun for w in ("省", "市", "地区")):
                resolved_antecedent = antecedents.get("province") or antecedents.get("city")
                resolved_dim = "province" if antecedents.get("province") else "city"
            elif any(w in pronoun for w in ("类目", "品类")):
                resolved_antecedent = antecedents.get("category")
                resolved_dim = "category"
            else:
                # Generic pronouns ("它", "它们", "他们") -> pick the primary antecedent
                for d in ("category", "carrier", "province", "city", "seller", "channel"):
                    if d in antecedents:
                        resolved_antecedent = antecedents[d]
                        resolved_dim = d
                        break

            if resolved_antecedent:
                rewritten = raw_q.replace(pronoun, str(resolved_antecedent))
                new_filters = dict(standing_filters or {})
                if resolved_dim:
                    new_filters[resolved_dim] = resolved_antecedent

                return CoreferenceResult(
                    resolved_query=rewritten.strip(),
                    original_query=raw_q,
                    is_rewritten=True,
                    resolution_type="pronoun_coreference",
                    antecedent_entity=str(resolved_antecedent),
                    substitute_entity=str(resolved_antecedent),
                    entity_dimension=resolved_dim,
                    inherited_metrics=current_entities.get("metrics") or antecedents.get("metrics", []),
                    inherited_time_window=current_entities.get("time_window") or antecedents.get("time_window"),
                    standing_filters=new_filters,
                )

        # ------------------------------------------------------------------
        # Pattern 4: Metric query with omitted subject inheriting standing filters
        # e.g. "延误率是多少" when standing_filters has carrier="中通"
        # ------------------------------------------------------------------
        if (
            current_entities.get("metrics")
            and not any(k in current_entities for k in ("carrier", "province", "city", "category", "seller"))
            and any(k in antecedents for k in ("carrier", "province", "city", "category", "seller"))
        ):
            # Prepend active entity to question
            primary_entity = (
                antecedents.get("carrier")
                or antecedents.get("category")
                or antecedents.get("province")
                or antecedents.get("city")
                or antecedents.get("seller")
            )
            time_prefix = antecedents.get("time_window", "") if not current_entities.get("time_window") else ""
            rewritten = f"{time_prefix}{primary_entity}的{raw_q}".strip()
            return CoreferenceResult(
                resolved_query=rewritten,
                original_query=raw_q,
                is_rewritten=True,
                resolution_type="implicit_subject_inheritance",
                antecedent_entity=str(primary_entity),
                substitute_entity=str(primary_entity),
                inherited_metrics=current_entities.get("metrics", []),
                inherited_time_window=current_entities.get("time_window") or antecedents.get("time_window"),
                standing_filters=dict(standing_filters or {}),
            )

        # No rewrite needed
        return CoreferenceResult(
            resolved_query=raw_q,
            original_query=raw_q,
            is_rewritten=False,
            resolution_type="none",
            standing_filters=dict(standing_filters or {}),
        )
