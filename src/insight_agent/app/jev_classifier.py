"""Jev Decision Model Adapter for Intent Classification and Routing.

Implements:
1. TypeSafe AI 'Jev' System-1 structured decision primitives (Choice & Score).
2. Five-way intent classification:
   - diagnostic_anomaly: Cause diagnosis, carrier strikes, delay, bad reviews.
   - complex_analytic: Multi-table joins, trends, window aggregations, ratios.
   - simple_metric: High-frequency single-table KPI counts, revenue sums.
   - prompt_injection: Malicious system jailbreaks, SQL destructive injections.
   - chat_general: Chit-chat, greeting, ungrounded assistant queries.
3. Offline fallback: when the external endpoint is unavailable, use deterministic
   keyword scores and softmax. These heuristic scores are not calibrated semantic
   model probabilities.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from insight_agent.app.intent_signals import (
    contains_keyword,
    detect_safety_signals,
    has_aggregation_signal,
    has_business_signal,
    has_causal_signal,
    is_standalone_chat,
)

logger = logging.getLogger(__name__)

IntentChoice = Literal[
    "diagnostic_anomaly",
    "complex_analytic",
    "simple_metric",
    "prompt_injection",
    "chat_general",
]

DEFAULT_INTENT_OPTIONS: dict[IntentChoice, str] = {
    "diagnostic_anomaly": (
        "排查业务异常、物流延误迟到、客诉差评飙升、商品破损瑕疵及承运商根因因果归因"
    ),
    "complex_analytic": (
        "涉及多表关联JOIN、同环比趋势对比、转化率占比、分组排名等复杂分析计算"
    ),
    "simple_metric": (
        "查询今日订单数、总销售额、用户数等基础聚合单表数值统计"
    ),
    "prompt_injection": (
        "恶意注入指令、试图破坏或删除数据库、越权逃逸或无视安全规则"
    ),
    "chat_general": (
        "打招呼、问好、自我介绍或非电商数据业务的日常闲聊对话"
    ),
}

# Semantic prototype clusters for Jev-style local discriminative scoring
_SEMANTIC_PROTOTYPES: dict[IntentChoice, list[str]] = {
    "diagnostic_anomaly": [
        "延误", "迟到", "慢", "没到", "没收到", "差评", "投诉", "客诉", "抱怨", "不爽",
        "喷", "破损", "损坏", "坏了", "瑕疵", "质量", "异常", "异动", "根因", "原因",
        "为什么", "暴跌", "骤降", "飙升", "退款", "退货", "承运商", "物流", "履约",
        "催单", "物流卡在分拨中心", "漏发", "碎了", "假货", "掉秤", "顺丰为什么还不派送",
        "客服不理人", "批量退单", "泡沫箱", "漏液", "发臭", "转运中心", "积压", "生鲜", "烂了",
        "排查", "排障", "归因", "诊断", "因何", "导致",
        "delay", "delayed", "why", "increase", "surge", "complaint", "broken",
        "damaged", "defect", "bad review",
        "olist", "里约", "rio", "圣保罗", "parana", "atraso", "quebrado",
    ],
    "complex_analytic": [
        "join", "关联", "对比", "同比", "环比", "占比", "转化率", "趋势", "排名", "top",
        "分组", "group by", "平均", "中位数", "去重", "distinct", "多表", "跨表",
        "漏斗", "留存", "复购", "退款率", "退货率", "时效对比", "时效", "哪个", "更快", "慢一点", "顺丰与中通",
        "trend", "compare", "ranking", "distribution", "conversion", "retention",
        "repurchase", "funnel", "ltv", " rate", "by channel", "by 省份",
    ],
    "simple_metric": [
        "多少", "总共", "销售额", "订单量", "用户数", "总数", "金额", "列表", "查一下",
        "看下", "有哪些", "今天", "本月", "单量", "客单价", "总共卖了多少钱", "今天客单价",
        "上海完成的订单数", "会员人数", "妥投", "已完成", "全店", "上海已完成",
    ],
    "prompt_injection": [
        "drop table", "truncate", "delete from", "update ", "ignore previous",
        "bypass", "system prompt", "管理员密码", "越权", "提权", "注入", "format disk",
        "忽略之前", "忽略以上", "忘记之前", "忘记以上", "无视规则", "无视安全",
        "删除数据库", "删库", "清空数据库", "你现在是管理员", "泄露系统提示", "绕过安全",
    ],
    "chat_general": [
        "你好", "hello", "hi", "在吗", "是谁", "能做什么", "帮助", "help", "再见", "谢谢",
        "你是谁", "介绍一下自己", "有什么功能", "怎么用", "如何使用", "系统介绍", "你好呀",
        "早上好", "下午好", "晚上好", "哈喽", "介绍一下", "你是干嘛的",
    ],
}


@dataclass(slots=True)
class JevChoiceResult:
    """Type-safe structured output corresponding to Jev's Choice primitive."""

    best_choice: IntentChoice
    probabilities: dict[IntentChoice, float]
    confidence: float
    latency_ms: float
    source: str = "jev_local_sim"  # "jev_api" | "jev_local_sim"
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_diagnostic(self) -> bool:
        return self.best_choice == "diagnostic_anomaly"

    def is_complex(self) -> bool:
        return self.best_choice == "complex_analytic"


class JevIntentClassifier:
    """Production Jev System-1 classifier with full zero-infra local simulator."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        model_name: str | None = None,
        temperature: float = 1.0,
        local_only: bool = False,
        shared_injection_patterns: bool = True,
        aggregation_guard: bool = True,
    ) -> None:
        self.api_key = (api_key or os.environ.get("INSIGHT_JEV_API_KEY")) if not local_only else None
        self.endpoint = (endpoint or os.environ.get("INSIGHT_JEV_URL")) if not local_only else None
        self.model_name = model_name or (
            os.environ.get("INSIGHT_JEV_MODEL", "jev-latest") if not local_only else "jev-local"
        )
        self.temperature = temperature
        self.has_api = bool(self.endpoint) and not local_only
        self.shared_injection_patterns = shared_injection_patterns
        self.aggregation_guard = aggregation_guard

    def classify(self, question: str) -> JevChoiceResult:
        """Classify user question using Jev Choice primitive."""
        start_t = time.perf_counter()

        if self.has_api:
            try:
                return self._call_remote_jev(question, start_t)
            except Exception as exc:
                logger.warning(f"Jev remote API call failed ({exc}); falling back to local simulator")

        return self._local_discriminate(question, start_t)

    def _call_remote_jev(self, question: str, start_t: float) -> JevChoiceResult:
        """Execute HTTP call to the TypeSafe AI System One (Jev) endpoint.

        Request/response conform to the official POST /v1/systemone schema:
        the content to evaluate goes in ``state``; the routing decision is a
        ``choice`` question whose ``criteria`` map holds the intent labels.
        The answer is nested under ``answers.<question_name>``.
        """
        import json
        import urllib.request

        payload = {
            "model": self.model_name,
            "state": question,
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "判断该电商用户问题应路由到的意图类别",
                    "criteria": dict(DEFAULT_INTENT_OPTIONS),
                }
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            str(self.endpoint),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        elapsed = (time.perf_counter() - start_t) * 1000
        answer = data.get("answers", {}).get("intent", {})
        best = answer.get("choice", "simple_metric")
        probs = {k: float(v) for k, v in answer.get("probabilities", {}).items()}
        confidence = float(answer.get("confidence", probs.get(best, 0.0)))
        return JevChoiceResult(
            best_choice=best,
            probabilities=probs,
            confidence=confidence,
            latency_ms=round(elapsed, 2),
            source="jev_api",
        )

    def _local_discriminate(self, question: str, start_t: float) -> JevChoiceResult:
        """Compute deterministic keyword scores and an uncalibrated softmax."""
        q_lower = question.lower()
        logits: dict[IntentChoice, float] = {
            "diagnostic_anomaly": 0.0,
            "complex_analytic": 0.0,
            "simple_metric": 0.5,  # slight base prior for ordinary data queries
            "prompt_injection": 0.0,
            "chat_general": 0.0,
        }

        # Shared precision guard, with the old smaller inventory retained only
        # as an ablation. A safety mention never becomes an injection logit.
        safety = detect_safety_signals(question)
        legacy_hit = any(
            contains_keyword(question, pat) for pat in _SEMANTIC_PROTOTYPES["prompt_injection"]
        )
        if safety.is_dangerous and (self.shared_injection_patterns or legacy_hit):
            logits["prompt_injection"] += 12.0

        # A salutation cannot defeat the actual business instruction following it.
        if is_standalone_chat(question):
            logits["chat_general"] += 8.0
        elif not has_business_signal(question) and not has_aggregation_signal(question):
            for pat in _SEMANTIC_PROTOTYPES["chat_general"]:
                if contains_keyword(question, pat):
                    logits["chat_general"] += 2.0

        # 3. Diagnostic anomaly semantic matching
        for pat in _SEMANTIC_PROTOTYPES["diagnostic_anomaly"]:
            if contains_keyword(question, pat):
                logits["diagnostic_anomaly"] += 2.0
                # Specific high-conviction colloquial markers
                if pat in ("迟到", "不爽", "为什么", "延误", "差评", "暴跌", "异动", "根因", "瘫痪"):
                    logits["diagnostic_anomaly"] += 1.8

        # 4. Complex analytic indicators
        for pat in _SEMANTIC_PROTOTYPES["complex_analytic"]:
            if contains_keyword(question, pat):
                logits["complex_analytic"] += 2.2

        # 5. Question structure signals
        if any(c in q_lower for c in ("为何", "为啥", "怎么回事", "咋回事", "谁的锅", "谁的责任", "到底咋", "出问题")):
            logits["diagnostic_anomaly"] += 3.0

        # If length is very long and has diagnostic clues
        if len(question) > 35 and logits["diagnostic_anomaly"] > 1.0:
            logits["diagnostic_anomaly"] += 1.2

        # 6. Aggregation guard (precision fix for overloaded diagnostic vocab):
        # a question that reads as a plain aggregation and carries no causal/anomaly
        # cue is a metric query that merely *filters* on a word like 退款/物流/破损 —
        # it must not be dragged into diagnosis by that filter value alone.
        _AGG_MARKERS = (
            "多少", "总数", "总额", "总金额", "总量", "总计", "统计", "数量", "占比",
            "占", "比例", "比率", "分布", "平均", "计算", "各", "列表", "有哪些",
            "排名", "排行", "top", "销售额", "金额", "gmv", "单量", "客单价",
            "订单数", "订单量", "用户数", "会员数", "笔",
        )
        _CAUSAL_MARKERS = (
            "为什么", "为何", "怎么回事", "咋回事", "什么原因", "原因是", "根因",
            "归因", "异常", "异动", "暴跌", "骤降", "飙升", "激增", "突然", "下滑",
            "下跌", "诊断", "排查", "排障", "谁的锅", "谁的责任", "导致", "因何",
        )
        has_causal = has_causal_signal(question)
        has_agg = has_aggregation_signal(question)
        if self.aggregation_guard and has_agg and not has_causal and logits["diagnostic_anomaly"] > 0.0:
            logits["diagnostic_anomaly"] = 0.0

        # Softmax normalization
        max_logit = max(logits.values())
        exp_sum = sum(math.exp((v - max_logit) / self.temperature) for v in logits.values())
        probs: dict[IntentChoice, float] = {
            k: round(math.exp((v - max_logit) / self.temperature) / exp_sum, 4)
            for k, v in logits.items()
        }

        # Select best choice
        best_intent = max(probs.items(), key=lambda it: it[1])[0]
        elapsed = (time.perf_counter() - start_t) * 1000

        return JevChoiceResult(
            best_choice=best_intent,
            probabilities=probs,
            confidence=probs[best_intent],
            latency_ms=round(elapsed, 2),
            source="jev_local_sim",
        )
