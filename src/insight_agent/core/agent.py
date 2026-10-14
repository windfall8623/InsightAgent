"""Standalone ReAct Agent loop for autonomous multi-turn tool execution.

Architecture Positioning:
-------------------------
- `core.agent.Agent` provides a general-purpose, autonomous ReAct loop
  (Thought -> Action -> Observation -> Final Answer).
- In InsightAgent's production workflow:
  * Domain-specific e-commerce NL2SQL is handled by `InsightOrchestrator`
    (linear 8-stage pipeline with AST compilation and A2A delegation).
  * Deep root-cause diagnosis is handled by `InsightGraphEngine`
    (LangGraph 13-node statechart with reflection loops).
- `core.agent.Agent` serves as a standalone autonomous tool-calling runner
  and general sub-agent worker.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from insight_agent.core.llm import SYSTEM_PROMPT, LLMClient, Message
from insight_agent.core.tools import Tool, ToolRegistry


class MaxTurnsExceeded(RuntimeError):
    """Raised when the loop hits its turn budget without a final answer."""


@dataclass(slots=True)
class AgentConfig:
    """Knobs for a run."""

    system_prompt: str = SYSTEM_PROMPT
    max_turns: int = 12
    # Surface every tool call on stdout as it happens.
    verbose: bool = True


@dataclass(slots=True)
class AgentResult:
    """The outcome of :meth:`Agent.run`."""

    text: str
    messages: list[Message] = field(default_factory=list)
    turns: int = 0
    tool_calls: int = 0

    @property
    def transcript(self) -> list[dict[str, Any]]:
        return [m.model_dump() for m in self.messages]


class Agent:
    """Drives a conversation until the model answers without asking for tools."""

    def __init__(
        self,
        client: LLMClient,
        tools: Sequence[Tool] | ToolRegistry = (),
        config: AgentConfig | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.client = client
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.config = config or AgentConfig()
        self._on_event = on_event

    def _emit(self, kind: str, **payload: Any) -> None:
        if self._on_event is not None:
            self._on_event(kind, payload)

    async def run(self, prompt: str, history: Sequence[Message] = ()) -> AgentResult:
        messages: list[Message] = [
            Message(role="system", content=self.config.system_prompt),
            *history,
            Message(role="user", content=prompt),
        ]
        schemas = self.tools.schemas()
        tool_call_count = 0

        for turn_index in range(1, self.config.max_turns + 1):
            turn = await self.client.complete(messages, schemas)
            messages.append(
                Message(
                    role="assistant",
                    content=turn.text,
                    tool_calls=turn.tool_calls,
                )
            )

            if not turn.wants_tools:
                return AgentResult(
                    text=turn.text,
                    messages=messages,
                    turns=turn_index,
                    tool_calls=tool_call_count,
                )

            for call in turn.tool_calls:
                tool_call_count += 1
                self._emit("tool_call", call=call)
                result = await self.tools.run(call)
                self._emit("tool_result", call=call, result=result)
                messages.append(
                    Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=result.call_id,
                    )
                )

        raise MaxTurnsExceeded(f"Agent did not finish within {self.config.max_turns} turns")

    def run_sync(self, prompt: str, history: Sequence[Message] = ()) -> AgentResult:
        """Synchronously execute the ReAct agent loop."""
        return asyncio.run(self.run(prompt, history=history))


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"
