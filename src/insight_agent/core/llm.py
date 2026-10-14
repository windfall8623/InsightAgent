"""Model-facing types and the client interface the agent loop talks to.

The agent loop only depends on :class:`LLMClient`, so a live provider and a
scripted test double are interchangeable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

Role = str  # "user" | "assistant" | "tool"

SYSTEM_PROMPT: str = (
    "You are Space Bunny, a tool-using assistant. "
    "Call a tool when it is the only way to answer accurately, and otherwise reply in plain text."
)


class ToolCall(BaseModel):
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of running a tool, fed back to the model."""

    call_id: str
    name: str
    content: str
    is_error: bool = False


class Message(BaseModel):
    """One turn of conversation, normalised across providers."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(slots=True)
class Turn:
    """What the model produced for one turn."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface the agent loop requires of a model."""

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> Turn:
        """Return the model's next turn given the conversation so far."""
        ...


class ScriptedLLM:
    """A deterministic client that replays canned turns.

    Useful for tests and for exercising the loop without a network call.
    """

    def __init__(self, turns: Sequence[Turn]) -> None:
        self._turns = list(turns)
        self._index = 0
        self.calls: list[tuple[list[Message], list[dict[str, Any]]]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> Turn:
        self.calls.append((list(messages), list(tools)))
        if self._index >= len(self._turns):
            raise RuntimeError("ScriptedLLM ran out of turns")
        turn = self._turns[self._index]
        self._index += 1
        return turn


class EchoLLM:
    """A client that answers with a fixed string and never calls tools."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> Turn:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user" and m.content),
            "",
        )
        return Turn(text=f"{self.reply}: {last_user}".strip(": "))


def dumps(value: Any) -> str:
    """Serialise a tool result so it can travel back to the model as text."""

    return json.dumps(value, ensure_ascii=False, default=str)
