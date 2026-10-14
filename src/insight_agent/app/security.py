"""Three-layer agent security: input, tool, and output.

Layering matters because each layer fails differently:

* **InputGuard** — marks content as untrusted and detects injection attempts
  before anything else sees it. Cheap, pattern-based, high recall.
* **ToolPolicy** — the only thing allowed to grant execution permission. It
  combines static SQL analysis, per-tool privilege levels, and a human
  confirmation gate for high-impact actions.
* **OutputGuard** — redacts sensitive values on the way back to the caller.

None of these is a complete solution on its own; the sandbox's SQLite
authorizer in :mod:`insight_agent.data.database` is the independent backstop
that makes a bypass of all three non-destructive.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from insight_agent.core.llm import ToolCall, ToolResult


class Verdict(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    CONFIRM = "confirm"


@dataclass(slots=True)
class SecurityDecision:
    """The outcome of a guard check, with the reason attached."""

    verdict: Verdict
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.verdict is Verdict.BLOCK

    @property
    def needs_confirmation(self) -> bool:
        return self.verdict is Verdict.CONFIRM


ALLOW = SecurityDecision(Verdict.ALLOW)


# --------------------------------------------------------------------------
# Layer 1 — input
# --------------------------------------------------------------------------

# Patterns that indicate an attempt to override the system prompt or extract
# the agent's instructions. Kept broad on purpose: a false positive costs the
# user one clarification, a false negative costs a data breach.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "prompt_override",
        re.compile(
            r"(ignore|disregard|forget|override|bypass)\s+"
            r"(all\s+|any\s+|the\s+)?((previous|prior|above|earlier|system)\s+)?"
            r"(instruction|instructions|prompt|rule|rules|direction|message)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_injection",
        re.compile(
            r"(you\s+are\s+now|from\s+now\s+on\s+you|act\s+as\s+(a\s+)?(admin|root|dba|developer))",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_delimiter",
        re.compile(
            r"(<\s*/?\s*(system|assistant|instructions?)\s*>|\[\s*(SYSTEM|INST)\s*\])",
            re.IGNORECASE,
        ),
    ),
    (
        "exfiltration",
        re.compile(
            r"(reveal|show|print|output|repeat|disclose)\s+(me\s+)?"
            r"(your\s+|the\s+)?(system\s+prompt|instructions?|api[_\s-]?key|secret|password|credential)",
            re.IGNORECASE,
        ),
    ),
    (
        "destructive_intent",
        re.compile(
            r"(drop|truncate|delete|delete\s+from)\s+(all\s+)?(the\s+)?(table|tables|database|data|records?|\w+)",
            re.IGNORECASE,
        ),
    ),
)

# Chinese-language equivalents; the demo and its users are Chinese-speaking.
_INJECTION_PATTERNS_ZH: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "prompt_override_zh",
        re.compile(
            r"(忽略|无视|忘记|绕过|跳过).{0,8}(之前|上面|以上|所有).{0,8}(指令|提示|规则|设定)"
        ),
    ),
    (
        "role_injection_zh",
        re.compile(
            r"(你现在是|从现在开始你是|扮演).{0,6}(管理员|root|开发者|数据库管理员)", re.IGNORECASE
        ),
    ),
    ("instruction_delimiter_zh", re.compile(r"(系统提示|你的提示词|初始指令)")),
    (
        "exfiltration_zh",
        re.compile(
            r"(输出|告诉我|打印|泄露|给我看|显示).{0,20}(系统提示|提示词|密钥|密码|凭证|api\s*key|secret|token)",
            re.IGNORECASE,
        ),
    ),
    (
        "destructive_intent_zh",
        re.compile(r"(删除|清空|删掉|drop|truncate).{0,6}(所有|全部|表|数据|记录|数据库)"),
    ),
)


@dataclass(slots=True)
class InputGuardResult:
    verdict: SecurityDecision
    categories: list[str] = field(default_factory=list)
    sanitized: str = ""

    @property
    def is_trusted(self) -> bool:
        return not self.verdict.blocked


class InputGuard:
    """Marks untrusted content and detects prompt-injection attempts."""

    def __init__(self, *, block_on_injection: bool = True) -> None:
        self._block_on_injection = block_on_injection

    def inspect(
        self, text: str, *, history: Any = None, context: Any = None,
    ) -> InputGuardResult:
        from insight_agent.app.intent_signals import detect_safety_signals

        signals = detect_safety_signals(text, history=history, context=context)
        hits = list(signals.reasons)
        if signals.is_dangerous:
            decision = SecurityDecision(
                Verdict.BLOCK if self._block_on_injection else Verdict.ALLOW,
                reason=f"prompt injection detected: {', '.join(hits)}",
                detail={"categories": hits, "ambiguous": signals.ambiguous},
            )
        else:
            decision = SecurityDecision(
                Verdict.ALLOW,
                reason="security-sensitive discussion" if signals.ambiguous else "no explicit unsafe instruction",
                detail={"ambiguous": signals.ambiguous},
            )
        return InputGuardResult(decision, hits, text)


# --------------------------------------------------------------------------
# Layer 2 — tools
# --------------------------------------------------------------------------

_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|"
    r"pragma|vacuum|reindex|grant|revoke)\b",
    re.IGNORECASE,
)

# Multi-statement smuggling: "SELECT 1; DROP TABLE orders".
_MULTI_STATEMENT = re.compile(r";\s*\S")

_COMMENT_OBFUSCATION = re.compile(r"/\*.*?\*/|--[^\n]*", re.DOTALL)

# Note the leading `\b` must sit *inside* the alternation: `\b(` before a group
# that can start with `*` requires a word boundary next to a non-word character,
# which never holds, so the cartesian-product branch would be dead code.
_HIGH_IMPACT = re.compile(
    r"(without\s+limit\b|cross\s+join\b|cartesian\b|\*\s+from\s+\w+\s*,\s*\w+)",
    re.IGNORECASE,
)


class Privilege(StrEnum):
    """Per-tool privilege levels; least privilege by default."""

    READ_METADATA = "read_metadata"
    READ_DATA = "read_data"


@dataclass(slots=True)
class ToolPolicy:
    """The single authority for whether a tool call may proceed."""

    allowed_tools: dict[str, Privilege] = field(
        default_factory=lambda: {
            "list_tables": Privilege.READ_METADATA,
            "get_schema": Privilege.READ_METADATA,
            "run_sql": Privilege.READ_DATA,
            "describe_result": Privilege.READ_DATA,
        }
    )
    confirm_on_high_impact: bool = True
    max_rows: int = 500
    auto_approve: bool = False

    def check(self, call: ToolCall) -> SecurityDecision:
        privilege = self.allowed_tools.get(call.name)
        if privilege is None:
            return SecurityDecision(
                Verdict.BLOCK,
                reason=f"tool not permitted by policy: {call.name}",
                detail={"tool": call.name},
            )

        if call.name == "run_sql":
            return self._check_sql(call)

        return SecurityDecision(Verdict.ALLOW, reason=f"tool {call.name} allowed")

    def _check_sql(self, call: ToolCall) -> SecurityDecision:
        sql = str(call.arguments.get("sql", ""))

        from insight_agent.app.sql_text import sql_skeleton, validate_readonly_sql

        parsed = validate_readonly_sql(sql)
        if parsed.outcome != "accepted":
            return SecurityDecision(
                Verdict.BLOCK,
                reason=parsed.error or parsed.outcome,
                detail={"tool": call.name, "parse_outcome": parsed.outcome},
            )
        stripped = sql_skeleton(sql, mask_identifiers=False)

        if self.confirm_on_high_impact and _HIGH_IMPACT.search(stripped):
            return SecurityDecision(
                Verdict.CONFIRM,
                reason="high-impact query requires human confirmation",
                detail={"tool": call.name, "sql": sql},
            )

        return SecurityDecision(Verdict.ALLOW, reason="read-only query", detail={"sql": sql})


class ConfirmationRequired(Exception):
    """Raised when a call needs a human decision the caller must supply."""

    def __init__(self, call: ToolCall, decision: SecurityDecision) -> None:
        super().__init__(decision.reason)
        self.call = call
        self.decision = decision


class PolicyEnforcingRegistry:
    """Wraps a :class:`ToolRegistry` so every call passes through ToolPolicy."""

    def __init__(
        self,
        registry: Any,
        policy: ToolPolicy,
        *,
        approver: Any = None,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._approver = approver

    @property
    def registry(self) -> Any:
        return self._registry

    def schemas(self) -> list[dict[str, Any]]:
        return list(self._registry.schemas())

    async def run(self, call: ToolCall) -> ToolResult:
        decision = self._policy.check(call)

        if decision.blocked:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"BLOCKED: {decision.reason}",
                is_error=True,
            )

        if decision.needs_confirmation:
            approved = await self._ask(call, decision)
            if not approved:
                return ToolResult(
                    call_id=call.id,
                    name=call.name,
                    content="DECLINED: user rejected the high-impact query",
                    is_error=True,
                )

        if call.name == "run_sql":
            call = ToolCall(
                id=call.id,
                name=call.name,
                arguments={**call.arguments, "limit": self._policy.max_rows},
            )
        result: ToolResult = await self._registry.run(call)
        return result

    async def _ask(self, call: ToolCall, decision: SecurityDecision) -> bool:
        if self._policy.auto_approve:
            return True
        if self._approver is None:
            return False
        result = self._approver(call, decision)
        if hasattr(result, "__await__"):
            return bool(await result)
        return bool(result)


class TerminalApprover:
    """Console-based human approver for interactive CLI sessions."""

    def __init__(self, prompt_fn: Callable[[str], str] | None = None) -> None:
        self.prompt_fn = prompt_fn or input

    def __call__(self, call: ToolCall, decision: SecurityDecision) -> bool:
        sql = decision.detail.get("sql", "")
        prompt = (
            f"\n[HITL 审批确认] 检测到高危数据库操作！\n"
            f"  原因: {decision.reason}\n"
            f"  工具: {call.name}\n"
            f"  SQL: {sql}\n"
            f"是否批准执行该操作? [y/N]: "
        )
        try:
            choice = self.prompt_fn(prompt).strip().lower()
            return choice in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False


# --------------------------------------------------------------------------
# Layer 3 — output
# --------------------------------------------------------------------------

_REDACTIONS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[REDACTED_EMAIL]"),
    ("phone_cn", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[REDACTED_PHONE]"),
    ("id_card_cn", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "[REDACTED_ID]"),
    ("card", re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "[REDACTED_CARD]"),
    ("api_key", re.compile(r"\b(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,})\b"), "[REDACTED_KEY]"),
)


class OutputGuard:
    """Strips sensitive values from anything leaving the agent."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self.redactions = 0

    def scrub(self, text: str) -> str:
        if not self._enabled:
            return text
        for _name, pattern, replacement in _REDACTIONS:
            text, count = pattern.subn(replacement, text)
            self.redactions += count
        return text
