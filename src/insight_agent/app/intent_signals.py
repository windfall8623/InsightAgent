"""Shared, precision-first intent signals; none of these helpers performs I/O.

A safety *mention* is not an execution request.  Conversely, a disclaimer in one
clause must not neutralise an actual destructive/exfiltration request in another.
``ambiguous`` is a request for semantic review, never an authorisation to execute.
The original user text must be checked before any contextual rewriting.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from insight_agent.core.llm import Message

History = Sequence[Message | Mapping[str, Any]]
IntentContext = Mapping[str, Any]


def normalize_intent_context(context: IntentContext | None) -> dict[str, Any]:
    """Accept flat task slots as well as the session store's ``last_task``."""
    if not isinstance(context, Mapping):
        return {}
    task = context.get("last_task")
    task = task if isinstance(task, Mapping) else {}
    fields = (
        "dataset", "metric", "dimensions", "filters", "standing_filters", "time_window",
        "question", "effective_question", "last_intent", "last_status", "last_refusal",
    )
    normalized = {key: task[key] for key in fields if key in task}
    normalized.update({key: context[key] for key in fields if key in context})
    normalized.setdefault("last_intent", task.get("intent"))
    statuses = (context.get("last_status"), context.get("last_turn_status"))
    normalized["last_status"] = "blocked" if "blocked" in statuses else next(
        (status for status in statuses if status), task.get("status", normalized.get("last_status"))
    )
    return {key: value for key, value in normalized.items() if value is not None}


def contains_keyword(text: str, keyword: str) -> bool:
    """Substring matching for Chinese, ASCII word boundaries for English."""
    term = keyword.strip().casefold()
    if not term:
        return False
    if re.search(r"[a-z]", term):
        left = r"(?<![a-z0-9_])" if term[0].isascii() and term[0].isalnum() else ""
        right = r"(?![a-z0-9_])" if term[-1].isascii() and term[-1].isalnum() else ""
        return bool(re.search(left + re.escape(term) + right, text.casefold()))
    return term in text.casefold()


def recent_history(history: History | None) -> tuple[Message, ...]:
    """Keep at most two user/assistant rounds, excluding tool/system payloads."""
    if not history or isinstance(history, (str, bytes)):
        return ()
    messages: list[Message] = []
    for item in history:
        role: Any
        content: Any
        if isinstance(item, Message):
            role, content = item.role, item.content
        elif isinstance(item, Mapping):
            if item.get("qualified") is False:
                continue
            role, content = item.get("role"), item.get("content")
        else:
            continue
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append(Message(role=str(role), content=content))
    user_indexes = [i for i, item in enumerate(messages) if item.role == "user"]
    if not user_indexes:
        return ()
    start = user_indexes[-2] if len(user_indexes) > 1 else user_indexes[0]
    return tuple(messages[start:][-4:])


_BUSINESS_TERMS = (
    "订单", "销售", "销量", "金额", "营收", "营业额", "成交", "卖了", "卖出", "下单",
    "退款", "售后", "买家", "用户", "会员", "商家", "店铺", "商品", "品类", "类目",
    "物流", "快递", "承运商", "运费", "差评", "好评", "评价", "评分", "客诉", "延误",
    "超时", "破损", "积压", "滞留", "货款", "实付", "单量", "客单价", "退货", "转化",
    "复购", "妥投", "时效", "工单", "积分", "代金券", "优惠券", "余额", "报表", "包裹", "大促", "到达时间",
    "顺丰", "中通", "京东", "圆通", "极兔", "生鲜", "美妆", "数码", "百货", "食品",
    "男装", "母婴", "天猫", "小程序", "直播间", "olist", "gmv", "sku", "ltv", "arpu",
    "order", "orders", "sales", "revenue", "refund", "refunds", "return", "returns",
    "review", "reviews", "rating", "ratings", "carrier", "carriers", "logistics",
    "delivery", "deliveries", "shipment", "shipments", "shipping", "delay", "delays",
    "conversion", "retention", "repurchase", "churn", "category", "categories",
    "channel", "channels", "member", "members", "customer", "customers", "seller",
    "sellers", "aftersales", "voucher", "vouchers", "province", "city", "cities",
)
_METRIC_TERMS = (
    "订单数", "订单量", "单量", "销售额", "销售总额", "成交额", "销量", "营收", "营业额",
    "金额", "用户数", "会员数", "会员人数", "买家数", "客单价", "退款率", "退货率",
    "延误率", "延迟率", "差评率", "好评率", "转化率", "妥投时效", "配送时长", "平均评分",
    "gmv", "revenue", "sales", "count", "total", "rate", "ltv", "arpu",
)
_AGGREGATION = re.compile(
    r"多少|总数|总额|总量|总计|汇总|统计|数量|占比|比例|比率|分布|平均|计算|排名|排行|"
    r"按.{1,8}(?:分组|汇总|统计)|各(?:省|城市|品类|渠道|月)|"
    r"\b(?:count|total|sum|average|how many|how much|group by|by (?:city|channel|month|week)|top\s*\d+)\b",
    re.IGNORECASE,
)
_COMPLEX = re.compile(
    r"同比|环比|占比|比例|转化率|退款率|退货率|留存|复购|趋势|排名|排行|分布|中位数|"
    r"分组|多表|跨表|关联|对比|各(?:省|城市|品类|渠道|月)|按.{1,8}(?:分组|汇总)|"
    r"\b(?:join|group by|trend|trends|compare|comparison|ranking|distribution|conversion|"
    r"retention|repurchase|funnel|ltv|rate|by (?:city|channel|month|week)|top\s*\d+)\b",
    re.IGNORECASE,
)
_CAUSAL = re.compile(
    r"为什么|为何|为啥|怎么回事|咋回事|怎么会|什么原因|原因是|根因|归因|异动|异常|诊断|排查|排障|"
    r"暴跌|骤降|飙升|激增|偏高|偏低|超标|突然|下滑|下跌|谁的锅|谁的责任|问题出在|导致|因何|"
    r"(?:分析|查明).{0,6}(?:原因|根因)|"
    r"\b(?:why|root cause|diagnos\w*|investigat\w*|surge|surging|plummet\w*)\b",
    re.IGNORECASE,
)
_NEGATED_ACTION = re.compile(
    r"(?:不要|别|不许|禁止|勿|不应|无需|不必|不能|不会|拒绝|停止)"
    r"(?:(?:再|去|执行|运行|尝试|帮我|直接|实际|真正|真的|先|分析|解释|排查|诊断)|[\s:：\"'`]){0,6}$|"
    r"\b(?:do not|don't|doesn't|never|must not|should not|avoid|stop|without)"
    r"\s+(?:(?:actually|really|ever|please|execute|run)\s+){0,3}$",
    re.IGNORECASE,
)


def has_business_signal(question: str) -> bool:
    return any(contains_keyword(question, term) for term in _BUSINESS_TERMS)


def has_metric_signal(question: str) -> bool:
    return any(contains_keyword(question, term) for term in _METRIC_TERMS)


def has_aggregation_signal(question: str) -> bool:
    return bool(_AGGREGATION.search(question)) or has_metric_signal(question)


def has_complex_signal(question: str) -> bool:
    return bool(_COMPLEX.search(question))


def has_causal_signal(question: str) -> bool:
    return any(
        not _NEGATED_ACTION.search(question[max(0, match.start() - 40):match.start()])
        for match in _CAUSAL.finditer(question)
    )


_GREETING = re.compile(
    r"你好(?:呀|啊|吗)?|您好|嗨|哈喽|在吗|早上好|下午好|晚上好|谢谢(?:你)?|再见|"
    r"你是谁|介绍一下(?:你)?自己|你能(?:帮我)?做什么|你有什么功能|有什么功能|"
    r"你是干嘛的|怎么用|如何使用|使用指南|帮助|"
    r"\b(?:hello|hi(?: there)?|hey|good (?:morning|afternoon|evening)|thanks|thank you|"
    r"bye|goodbye|help|who are you|what can you do)\b",
    re.IGNORECASE,
)
_CHAT_REQUEST = re.compile(
    r"(?:继续)?(?:讲|说|来)(?:个|一个|点)?(?:笑话|故事)|聊(?:聊天|天|点别的)|"
    r"(?:今天|明天|最近)?(?:的)?天气(?:怎么样|如何|好吗)?|"
    r"\b(?:tell me a (?:joke|story)|let'?s chat|how are you|(?:what is|what about|how is) the weather)\b",
    re.IGNORECASE,
)
_CHAT_FILLER = re.compile(r"[\s，。！？、,.;!?：:]+|请问|请|帮我|给我|我们|呀|啊|吧|呢")
# Self-contained metadata requests: they name no metric, so a "which metric?"
# clarification is nonsensical.  They are grounded tasks, never ambiguity.
_SCHEMA_REQUEST = re.compile(
    r"表结构|字段(?:含义|说明|解释|列表|有哪些)?|数据字典|口径|"
    r"\b(?:schema|show\s+tables?|table\s+structure|describe\s+(?:the\s+)?table)\b",
    re.IGNORECASE,
)
_TOPIC_SWITCH = re.compile(
    r"换个话题|换一个话题|聊点别的|先不(?:查|看|分析|讨论|统计).{0,18}了|"
    r"(?:不|不用|别)(?:再)?(?:查|看|分析|统计).{0,18}了|"
    r"\b(?:new topic|change (?:the )?topic|instead|forget (?:the )?(?:analytics|analysis)|let'?s chat)\b",
    re.IGNORECASE,
)
_FOLLOWUP = re.compile(
    r"^(?:那(?:么)?|换成|改成|改为|换为|换做|继续|接着|然后呢|再来|再说|再查|再看|再按|还是|同样|按照前面|按之前|照刚才)|"
    r"它们?|他们|这些|这几个|该(?:商家|商户|物流商|承运商|省份|城市|类目)|"
    r"刚才|前面|上一次|之前的|呢[？?。.!！\s]*$|"
    r"\b(?:what about|how about|same as|as before|previous (?:request|question)|"
    r"earlier (?:request|question)|continue|do it|go ahead|those|their|that one|it instead)\b|"
    r"^and\s+(?:by|in|for|the)\b",
    re.IGNORECASE,
)


def is_standalone_chat(question: str) -> bool:
    """Only a complete greeting/capability request, never a greeting substring."""
    remainder = _GREETING.sub("", question.strip())
    return bool(_GREETING.search(question)) and not _CHAT_FILLER.sub("", remainder)


def is_explicit_chat(question: str) -> bool:
    if is_standalone_chat(question):
        return True
    # An explicit cancellation may mention the OLD business task. It is not an
    # instruction to perform it; safety is always checked on the unmodified text.
    current = _TOPIC_SWITCH.sub("", question)
    remainder = _GREETING.sub("", _CHAT_REQUEST.sub("", current))
    return bool(_CHAT_REQUEST.search(current)) and not _CHAT_FILLER.sub("", remainder)


def has_topic_switch_signal(question: str) -> bool:
    return bool(_TOPIC_SWITCH.search(question))


def has_schema_request_signal(question: str) -> bool:
    """A schema/dictionary/calibre request is a complete, self-contained task."""
    return bool(_SCHEMA_REQUEST.search(question))


def has_followup_signal(question: str) -> bool:
    return bool(_FOLLOWUP.search(question.strip())) and not is_standalone_chat(question)


@dataclass(frozen=True, slots=True)
class SafetySignals:
    """Public guard contract. ``ambiguous`` must defer, not silently allow."""

    is_dangerous: bool = False
    ambiguous: bool = False
    reasons: tuple[str, ...] = ()


_SECRET = (
    r"(?:数据库|管理员|生产|真实|实际|当前|系统|内部|root|admin)?密码|密钥|凭据|私钥|"
    r"系统提示词?|环境变量|秘密请求|"
    r"(?<![a-z0-9_])(?:api[\s_-]*keys?|access[\s_-]*tokens?|secret[\s_-]*keys?|private[\s_-]*keys?|"
    r"passwords?|credentials?|system prompts?|internal keys?|environment variables?)(?![a-z0-9_])"
)
_SQL_WRITE = re.compile(
    r"(?<![a-z0-9_])(?:drop\s+(?:table|database|schema)|truncate(?:\s+table)?|delete\s+from|"
    r"alter\s+table|update\s+[\w.`\"\[\]]+\s+set|insert\s+into|xp_cmdshell)(?![a-z0-9_])",
    re.IGNORECASE,
)
_DESTRUCTIVE = re.compile(
    r"删库|删表|(?:删除|删掉|删去|清空|销毁|擦除|抹掉|删)"
    r"(?:[\w\s的\-]{0,24}?)(?:数据库|数据表|记录|订单|数据|日志|评价|表)|"
    r"(?:数据库|数据表|记录|订单|数据|日志|评价)(?:全部|全都|都|所有|给我|给它|统统|的)?"
    r"(?:删掉|删除|清空|销毁)|"
    r"\b(?:delete|erase|wipe|destroy|remove)\s+(?:(?:all|the|production|our|these)\s+){0,3}"
    r"(?:databases?|tables?|records?|orders?|data|logs?|reviews?)\b|\bformat\s+(?:disk|drive)\b",
    re.IGNORECASE,
)
_EXTRACT = re.compile(
    rf"(?:输出|显示|打印|告诉我|给我|给出|导出|获取|读取|提供|列出|泄露|透露|复制|粘贴|"
    rf"\b(?:dump|print|show|reveal|disclose|exfiltrate|extract|fetch|give me|tell me|return)\b)"
    rf"[^，。！？,;!?\n]{{0,40}}?(?:{_SECRET})|"
    rf"(?:{_SECRET})\s*(?:是多少|是什么|发给我|告诉我|给我|给出|输出给我|提供给我|打印出来|输出出来)",
    re.IGNORECASE,
)
_OVERRIDE = re.compile(
    r"<\s*/?\s*(?:system|assistant|instructions?)\s*>|\[\s*(?:SYSTEM|INST)\s*\]|"
    r"(?:忽略|无视|忘记|绕过|覆盖|取消|关闭)[^，。！？,;!?\n]{0,20}"
    r"(?:规则|指令|规范|安全|权限|限制|系统提示|防护)|"
    r"(?:不要|别)(?:遵守|服从)[^，。！？,;!?\n]{0,15}(?:规则|指令|安全|限制)|"
    r"\b(?:ignore|disregard|forget|override|bypass|disable)\b[^,;.!?\n]{0,40}"
    r"\b(?:instructions?|rules?|polic(?:y|ies)|guardrails?|restrictions?|security|safety|system prompt)\b|"
    r"\b(?:do not|don't)\s+(?:follow|obey)\b[^,;.!?\n]{0,30}\b(?:instructions?|rules?|polic(?:y|ies))\b|"
    r"(?:开启|启用|进入)(?:开发者模式|上帝模式|无限制模式)|"
    r"\b(?:enable|activate|enter)\s+(?:developer|god|dan)\s+mode\b|"
    r"\byou are now\s+(?:unrestricted|without (?:rules|restrictions)|(?:an? )?(?:root|admin))\b|"
    r"你(?:现在)?是(?:无限制|不受限制|root|超级管理员|管理员|数据库管理员|admin|dba)|不受(?:任何)?(?:规则|安全)限制",
    re.IGNORECASE,
)
_MENTION = re.compile(
    r"解释|讲解|学习|讨论|说明|教我|教学|教程|示例|举例|演示|示范|含义|什么意思|是什么|为什么|为何|如何|怎么|防止|防范|检测|识别|"
    r"\b(?:explain|explanation|example|tutorial|teach|teaching|demo|demonstrate|documentation|"
    r"meaning|what (?:is|does)|why|how (?:to|do|can)|prevent|detect)\b",
    re.IGNORECASE,
)
_EXECUTE = re.compile(r"执行|运行|立刻|立即|马上|现在就|\b(?:execute|run|right now|actually)\b", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"占位|虚构|假(?:的|密码|密钥)?|示例|样例|\b(?:fake|dummy|placeholder|example)\b", re.IGNORECASE)
_REAL_SECRET = re.compile(r"真实|实际|生产|当前|内部|\b(?:real|actual|production|current|internal)\b", re.IGNORECASE)
_SECRET_TOPIC_SUFFIX = re.compile(
    r"^\s*(?:安全|强度|策略|轮换|管理|配置|保护|加密|政策|的(?:安全|轮换|管理|策略)|"
    r"rotation\b|security\b|safety\b|policy\b|policies\b|management\b|configuration\b|strength\b|protection\b)",
    re.IGNORECASE,
)
_QUOTED = re.compile(r'“[^”]*”|‘[^’]*’|"[^"\n]*"|`[^`]*`')
_SAFETY_TOPIC = re.compile(
    rf"{_SECRET}|{_SQL_WRITE.pattern}|{_DESTRUCTIVE.pattern}|{_OVERRIDE.pattern}|"
    r"注入|越权|越狱|提权|绕过|删记录|\b(?:jailbreak|injection|bypass|developer mode)\b",
    re.IGNORECASE,
)
_REPLAY = re.compile(
    r"(?:按(?:照)?|照着?)(?:我)?(?:前面|之前|刚才|上次)|继续(?:执行|做|操作)?|重试|照办|照做|再来一次|"
    r"\b(?:continue|do it|go ahead|proceed|repeat (?:it|that)|same (?:request|operation))\b",
    re.IGNORECASE,
)


def _softened(clause: str, match: re.Match[str], kind: str, *, quoted: bool = False) -> bool:
    before = clause[:match.start()]
    if _NEGATED_ACTION.search(before[-60:]):
        return True
    if kind == "secret_extraction" and _SECRET_TOPIC_SUFFIX.search(clause[match.end():]):
        return True
    execute = any(
        not _NEGATED_ACTION.search(before[max(0, cue.start() - 40):cue.start()])
        for cue in _EXECUTE.finditer(before)
    )
    quoted = quoted or any(q.start() <= match.start() < q.end() for q in _QUOTED.finditer(clause))
    # A tutorial wrapper does not excuse "execute ..." or "print the REAL key".
    real_extraction = kind == "secret_extraction" and bool(_REAL_SECRET.search(match.group()))
    explanatory_prefix = bool(re.search(r"解释|讲解|为什么|为何|\b(?:explain|why)\b", before, re.IGNORECASE))
    if explanatory_prefix and not execute:
        return True
    if (quoted or _MENTION.search(clause)) and not execute and not real_extraction:
        return True
    return bool(kind == "secret_extraction" and _PLACEHOLDER.search(clause) and not real_extraction)


def _text_safety(question: str) -> SafetySignals:
    reasons: list[str] = []
    softened = False
    # Scope negation to a clause/action, not to the entire message. A later
    # positive command wins over an earlier disclaimer or a negated operation.
    quotes = [(match.start(), match.end()) for match in _QUOTED.finditer(question)]
    separators = list(re.finditer(
        r"[，。！？,;!?\n]+|\b(?:but|then|however|and)\b|但是|但随后|然后|不过|但是?|并且|再把|并把",
        question, re.IGNORECASE,
    ))
    boundaries = [(0, 0), *((match.start(), match.end()) for match in separators), (len(question), len(question))]
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        start, end = left[1], right[0]
        clause = question[start:end]
        for kind, pattern in (
            ("destructive_operation", _SQL_WRITE),
            ("destructive_operation", _DESTRUCTIVE),
            ("secret_extraction", _EXTRACT),
            ("instruction_override", _OVERRIDE),
        ):
            for match in pattern.finditer(clause):
                quoted = any(a <= start + match.start() < b for a, b in quotes)
                if _softened(clause, match, kind, quoted=quoted):
                    softened = True
                elif kind not in reasons:
                    reasons.append(kind)
    if reasons:
        return SafetySignals(True, False, tuple(reasons))
    if softened or _SAFETY_TOPIC.search(question):
        return SafetySignals(False, True, ("safety_mention_requires_semantics",))
    return SafetySignals()


def detect_safety_signals(
    question: str,
    *,
    history: History | None = None,
    context: IntentContext | None = None,
) -> SafetySignals:
    """Detect explicit destructive, secret-extraction and rule-override intent.

    Call on the ORIGINAL request. ``last_status='blocked'`` is not a reusable
    analytics task. Referring back to a refused dangerous operation remains
    unsafe, while a genuinely new task does not inherit that old classification.
    """
    direct = _text_safety(question)
    if direct.is_dangerous:
        return direct
    ctx = normalize_intent_context(context)
    prior_danger = ctx.get("last_status") == "blocked" or ctx.get("last_intent") == "prompt_injection"
    latest_user = next((item for item in reversed(recent_history(history)) if item.role == "user"), None)
    prior_danger = prior_danger or bool(latest_user and _text_safety(latest_user.content).is_dangerous)
    replay = bool(_REPLAY.search(question))
    secret_reference = bool(replay and re.search(r"秘密请求|密码请求|密钥请求|被拒绝的请求|secret request", question, re.IGNORECASE))
    explanation = bool(_MENTION.search(question)) or bool(re.search(r"^(?:先)?(?:不要|别|不用)|\b(?:don't|do not)\b", question, re.IGNORECASE))
    reference_only = replay and not re.search(
        r"查询|统计|汇总|订单|销售|营收|退款|天气|笑话|故事|\b(?:query|count|sales|orders?|weather|joke)\b",
        question, re.IGNORECASE,
    )
    new_task = has_metric_signal(question) or has_causal_signal(question) or is_explicit_chat(question)
    if (
        secret_reference or (prior_danger and has_followup_signal(question) and (reference_only or not new_task))
    ) and not explanation:
        return SafetySignals(True, False, ("unsafe_context_continuation",))
    if prior_danger and replay and not new_task:
        return SafetySignals(False, True, ("blocked_context_requires_semantics",))
    return direct
