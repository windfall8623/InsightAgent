"""Context-window management: keeping long conversations inside a budget.

The problem is that an agent loop's message list grows without bound. Every
prior turn, every tool result, every cited schema fragment stays resident, so
latency and token cost grow with conversation length even when the recent turns
are the only ones that matter.

Strategies, cheapest first:

* **Sliding window** — drop the oldest turns. O(1) and lossless for the recent
  turns, but it discards facts the agent established earlier.
* **Summarised head** — replace older turns with one extractive summary that
  keeps tool names, SQL fragments and entity values. Those are the details a
  later answer is most likely to need verbatim.

Compression is not free, and pretending otherwise would be the mistake. The
accompanying experiment measures exactly how much accuracy is lost at each
budget, so the trade-off is a number rather than an assumption.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from insight_agent.core.llm import Message

# Facts worth carrying into a summary: tool names, SQL, entity literals.
_TOOL_HINT = re.compile(r"\b(run_sql|list_tables|get_schema|describe_result)\b")
_SQL_HINT = re.compile(r"\b(SELECT|FROM|JOIN|WHERE|GROUP BY|ORDER BY|LIMIT)\b", re.IGNORECASE)
_VALUE_HINT = re.compile(r"(?:=\s*'([^']+)'|\b(\d{4}-\d{2}-\d{2})\b)", re.IGNORECASE)


@dataclass(slots=True)
class CompressionStats:
    """What a compression pass did, for reporting."""

    before: int
    after: int
    dropped: int
    summarised: int
    kept_values: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.after / self.before if self.before else 1.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "before": self.before,
            "after": self.after,
            "dropped": self.dropped,
            "summarised": self.summarised,
            "ratio": round(self.ratio, 4),
            "kept_values": self.kept_values[:10],
        }


def extract_facts(messages: Sequence[Message]) -> list[str]:
    """Pull the details most worth preserving: tool names, SQL, entity values."""

    facts: list[str] = []
    for message in messages:
        blob = f"{message.content} {' '.join(c.name for c in message.tool_calls)}"
        for match in _TOOL_HINT.finditer(blob):
            facts.append(match.group(0))
        if _SQL_HINT.search(blob):
            for line in blob.splitlines():
                if _SQL_HINT.search(line):
                    facts.append(line.strip())
                    break
        for value_match in _VALUE_HINT.finditer(blob):
            facts.append(value_match.group(1) or value_match.group(2))
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(f for f in facts if f))


def compress(
    messages: Sequence[Message],
    *,
    budget: int = 8,
    keep_system: bool = True,
) -> tuple[list[Message], CompressionStats]:
    """Reduce ``messages`` to at most ``budget`` entries.

    The system prompt and the most recent turns are always kept verbatim; older
    ones collapse into a single summary message.
    """

    items = list(messages)
    if len(items) <= budget:
        stats = CompressionStats(before=len(items), after=len(items), dropped=0, summarised=0)
        return items, stats

    head: list[Message] = []
    body = items
    if keep_system and items and items[0].role == "system":
        head = [items[0]]
        body = items[1:]

    keep_recent = max(budget - len(head) - 1, 1)
    recent = body[-keep_recent:]
    older = body[:-keep_recent]

    facts = extract_facts(older)
    summary = Message(
        role="system",
        content="[compressed history] " + "; ".join(facts[:40])
        if facts
        else "[compressed history]",
    )

    result = head + ([summary] if older else []) + recent
    stats = CompressionStats(
        before=len(items),
        after=len(result),
        dropped=len(items) - len(result),
        summarised=len(older),
        kept_values=facts[:20],
    )
    return result, stats
