"""Difficulty-based model routing.

Most analytics questions are trivial ("how many orders today") and do not need a
frontier model. Routing them to a small one is where the cost and latency win
comes from; complex multi-join analytical questions escalate.

The decision is made from *observable* signals — join count, filters, date
windows, question length — rather than by asking a model to grade itself,
because a self-assessment is exactly the signal that is unreliable when a model
is confused.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from insight_agent.app.intent_signals import (
    History,
    IntentContext,
    detect_safety_signals,
    has_business_signal,
    has_causal_signal,
    has_followup_signal,
    is_explicit_chat,
    is_standalone_chat,
)
from insight_agent.core.llm import LLMClient

_DEFAULT_ROUTING_MODELS = ("sim-small", "sim-large")

_JOIN_HINT = re.compile(
    r"(\bjoin\b|\bgroup\s+by\b|关联|对比|分别|按.{0,4}分组|各个|各渠道|各品类|各省|各城市|分布)",
    re.IGNORECASE,
)
_COMPLEX_HINT = re.compile(
    r"(同比|环比|占比|转化率|率|趋势|top|排名|前\d+|每.*平均|中位数|distinct|去重|多表)",
    re.IGNORECASE,
)
_DATE_HINT = re.compile(r"(\d{4}|去年|今年|本月|上月|最近|近\s*\d+|每周|每天|季度)", re.IGNORECASE)


@dataclass(slots=True)
class RouteDecision:
    """Which model to use, and why — the reason is logged for the ablation."""

    model: str
    tier: str
    score: int
    reason: str

    @property
    def escalated(self) -> bool:
        return self.tier == "complex"


class ModelRouter:
    """Scores a question's difficulty and picks a model tier."""

    def __init__(
        self,
        *,
        small: str | None = None,
        large: str | None = None,
        threshold: int | None = None,
        allowed_models: tuple[str, ...] | None = None,
    ) -> None:
        env_model = os.environ.get("INSIGHT_MODEL")
        self.small = small or os.environ.get("INSIGHT_MODEL_SMALL") or (env_model if env_model else "sim-small")
        self.large = large or os.environ.get("INSIGHT_MODEL_LARGE") or (env_model if env_model else "sim-large")
        env_th = os.environ.get("INSIGHT_ROUTER_THRESHOLD")
        self.threshold = threshold if threshold is not None else (int(env_th) if env_th else 2)
        self.allowed_models = allowed_models

        # Validate if explicit allowed_models were provided
        if self.allowed_models is not None and (
            self.small not in self.allowed_models or self.large not in self.allowed_models
        ):
            raise ValueError("unknown routing model")

    def register_models(self, *, small: str, large: str) -> None:
        """Dynamically register or update small and large model identifiers."""
        self.small = small
        self.large = large

    def route(self, question: str, *, retrieved_tables: int = 0) -> RouteDecision:
        score = 0
        reasons: list[str] = []

        if _JOIN_HINT.search(question):
            score += 1
            reasons.append("join/grouping")
        if _COMPLEX_HINT.search(question):
            score += 1
            reasons.append("aggregation metric")
        if _DATE_HINT.search(question):
            score += 1
            reasons.append("time window")
        if retrieved_tables > 2:
            score += 1
            reasons.append(f"{retrieved_tables} tables in context")
        if len(question) > 40:
            score += 1
            reasons.append("long question")

        if score >= self.threshold:
            return RouteDecision(self.large, "complex", score, "+".join(reasons) or "score")
        return RouteDecision(self.small, "simple", score, "+".join(reasons) or "trivial")


_DIAGNOSTIC_KEYWORDS: tuple[str, ...] = (
    "延误", "差评", "客诉", "异常", "根因", "投诉", "履约", "积压", "滞留", "破损", "退款激增", "排障", "归因", "质检"
)

# Cascade-only, precision-first split of the diagnostic signal.
# A bare anomaly noun ("破损", "差评", "退款") is just as likely to be a *filter*
# in an ordinary metric query ("破损订单数量") as a diagnosis request. The
# fast-path therefore fires only when such a noun co-occurs with a causal cue —
# otherwise the question falls through to the semantic layer, which is where the
# simple/complex/diagnostic call belongs. Dataset markers (里约/olist) are NOT
# here: those route a dataset, not an intent.
_DIAGNOSTIC_NOUNS: tuple[str, ...] = (
    "延误", "差评", "客诉", "投诉", "履约", "积压", "滞留", "破损", "退款激增", "退款率",
    "质检", "超时", "丢件", "瑕疵", "损坏", "变质", "爆仓", "脱销", "缺货", "包装",
)
_CAUSAL_CUES: tuple[str, ...] = (
    "为什么", "为何", "为啥", "怎么回事", "咋回事", "怎么会", "什么原因", "原因是", "根因", "归因",
    "异常", "异动", "诊断", "排查", "排障", "暴跌", "骤降", "飙升", "激增", "偏高", "偏低", "超标",
    "突然", "下滑", "下跌", "谁的锅", "谁的责任", "问题出在", "卡在", "导致",
)

# Investigate / root-cause verbs that mean "diagnosis" on their own — they do not
# collide with ordinary metric vocabulary, so they alone trip the fast-path.
# ("原因" is deliberately NOT here: it collides with the field "退款原因".)
_STRONG_DIAGNOSTIC_CUES: tuple[str, ...] = (
    "排查", "排障", "根因", "归因", "诊断", "异动",
)

def _detect_injection(question: str) -> bool:
    """Compatibility shim; new guards should use ``detect_safety_signals``."""
    return detect_safety_signals(question).is_dangerous


def _diagnostic_two_signal(question: str) -> bool:
    """Fast-path diagnosis on a strong investigate verb, or on an anomaly-noun +
    causal-cue co-occurrence. A bare filter value alone never trips it."""
    if any(kw in question for kw in _STRONG_DIAGNOSTIC_CUES):
        return True
    has_noun = any(kw in question for kw in _DIAGNOSTIC_NOUNS)
    has_cue = any(kw in question for kw in _CAUSAL_CUES)
    return has_noun and has_cue

_CHAT_GENERAL_KEYWORDS: tuple[str, ...] = (
    "你好", "您好", "hello", "hi", "嗨", "在吗", "你是谁", "介绍一下", "你能做什么",
    "有什么功能", "怎么用", "如何使用", "使用指南", "帮助", "help", "谢谢", "再见", "哈喽",
    "早上好", "下午好", "晚上好", "你是干嘛的",
)


@dataclass(slots=True)
class IntentDecision:
    """Outcome of intent classification, supporting comparison between rule, jev, and cascade."""

    intent: str
    confidence: float
    mode: str  # "rule" | "jev" | "cascade"
    probabilities: dict[str, float]
    reason: str
    is_anomaly: bool = False
    continuation: bool = False
    topic_switch: bool = False
    resolved_question: str | None = None
    uncertain: bool = False
    needs_clarification: bool = False


@dataclass(frozen=True, slots=True)
class IntentFeatures:
    """Independently selectable components of the intent cascade."""

    injection_precheck: bool = True
    diagnostic_fast_path: bool = True
    greeting_fast_path: bool = True
    non_business_shortcut: bool = True
    fallback: bool = True
    shared_injection_patterns: bool = True
    aggregation_guard: bool = True


def _classify_single_turn(
    question: str,
    *,
    mode: str | None = None,
    features: IntentFeatures | None = None,
) -> IntentDecision:
    """Offline lexical backend. Scores are heuristics, not semantic certainty."""
    from insight_agent.app.jev_classifier import JevIntentClassifier

    features = features or IntentFeatures()
    selected_mode = (mode or os.environ.get("INSIGHT_INTENT_MODE", "cascade")).lower()
    if selected_mode not in ("rule", "jev", "cascade"):
        raise ValueError("intent mode must be rule, jev, or cascade")
    q_lower = question.strip().lower()

    # Deliberately retain the old rule baseline for reproducible ablations.
    if selected_mode == "rule":
        is_diag = any(kw in question for kw in _DIAGNOSTIC_KEYWORDS)
        is_chat = any(kw in q_lower for kw in _CHAT_GENERAL_KEYWORDS) and not any(
            kw in question for kw in ("订单", "销售", "金额", "退款", "用户", "类目", "品类", "物流", "时效")
        )
        intent = "diagnostic_anomaly" if is_diag else "chat_general" if is_chat else "simple_metric"
        return IntentDecision(
            intent=intent,
            confidence=1.0 if (is_diag or is_chat) else 0.5,
            mode="rule",
            probabilities={intent: 1.0},
            reason="keyword_rule_matched" if (is_diag or is_chat) else "no_keyword_matched",
            is_anomaly=is_diag,
        )

    classifier = JevIntentClassifier(
        local_only=True,
        shared_injection_patterns=features.shared_injection_patterns,
        aggregation_guard=features.aggregation_guard,
    )
    safety = detect_safety_signals(question)
    if selected_mode == "cascade":
        if features.injection_precheck and safety.is_dangerous:
            return IntentDecision(
                "prompt_injection", 0.99, "cascade(injection_guard)",
                {"prompt_injection": 0.99}, "injection_pattern_matched",
            )
        # Safety mentions and elliptical requests belong to semantic resolution,
        # even if another lexical heuristic happens to have a high score.
        defer = safety.ambiguous or has_followup_signal(question)
        if (
            features.diagnostic_fast_path and not defer
            and has_business_signal(question) and has_causal_signal(question)
            and _diagnostic_two_signal(question)
        ):
            return IntentDecision(
                "diagnostic_anomaly", 0.9, "cascade(rule_fast_path)",
                {"diagnostic_anomaly": 0.9}, "diagnostic_cooccurrence", is_anomaly=True,
            )
        if features.greeting_fast_path and not defer and is_standalone_chat(question):
            return IntentDecision(
                "chat_general", 1.0, "cascade(chat_fast_path)",
                {"chat_general": 1.0}, "chat_fast_path",
            )
        # Missing vocabulary is NOT positive evidence for chat. This feature now
        # shortcuts only positively identified general conversation.
        if features.non_business_shortcut and not defer and is_explicit_chat(question):
            return IntentDecision(
                "chat_general", 0.95, "cascade(non_business_general)",
                {"chat_general": 0.95}, "explicit_general_conversation",
            )
        if not features.fallback:
            return IntentDecision(
                "simple_metric", 0.5, "cascade(no_fallback)",
                {"simple_metric": 0.5}, "fallback_disabled_default_analytics",
                uncertain=True,
            )

    result = classifier.classify(question)
    return IntentDecision(
        intent=result.best_choice,
        confidence=result.confidence,
        mode="jev" if selected_mode == "jev" else "cascade(jev_semantic)",
        probabilities={str(key): value for key, value in result.probabilities.items()},
        reason=f"jev_choice({result.source})" if selected_mode == "jev" else f"cascade_jev_fallback({result.source})",
        is_anomaly=result.is_diagnostic(),
        uncertain=safety.ambiguous,
    )


def classify_intent(
    question: str,
    *,
    mode: str | None = None,
    features: IntentFeatures | None = None,
    local_only: bool = False,
    history: History | None = None,
    context: IntentContext | None = None,
) -> IntentDecision:
    """Compatible, always-offline classifier; production should use async.

    ``local_only`` is retained for callers/ablations; synchronous classification
    never invokes a provider, even when a Jev endpoint is configured.
    """
    from insight_agent.app.intent_service import classify_intent_local

    return classify_intent_local(question, history=history, context=context, mode=mode, features=features)


async def classify_intent_async(
    question: str,
    *,
    history: History | None = None,
    context: IntentContext | None = None,
    client: LLMClient | None = None,
    mode: str | None = None,
    local_only: bool = False,
) -> IntentDecision:
    """Precision fast paths plus bounded, validated semantic resolution.

    Only the explicitly supplied existing client may be called. No client is
    constructed and no credentials are read. Clarification retains the five-way
    label contract and sets ``needs_clarification`` instead of executing a task.
    """
    from insight_agent.app.intent_service import classify_intent_async as classify_async

    return await classify_async(
        question, history=history, context=context, client=client, mode=mode, local_only=local_only,
    )



def is_diagnostic_query(question: str, *, mode: str | None = None) -> bool:
    """Helper to determine whether a query should route to LangGraph 2.0."""
    return classify_intent(question, mode=mode).is_anomaly


class ResponseCache:
    """Caches SQL results per (user, sql) so repeat questions cost nothing.

    Bounded LRU. A real deployment would key on a semantic hash of the question;
    keying on the resolved SQL is exact and cannot serve a wrong answer.
    """

    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max_entries
        self._store: dict[tuple[str, str], Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, user_id: str, sql: str) -> Any | None:
        key = (user_id, sql)
        if key in self._store:
            self.hits += 1
            self._store[key] = self._store.pop(key)  # LRU touch
            return self._store[key]
        self.misses += 1
        return None

    def put(self, user_id: str, sql: str, value: Any) -> None:
        key = (user_id, sql)
        self._store[key] = value
        while len(self._store) > self.max_entries:
            self._store.pop(next(iter(self._store)))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
