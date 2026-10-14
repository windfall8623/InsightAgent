"""Anthropic-backed implementation of :class:`~insight_agent.core.llm.LLMClient`.

The SDK is an optional dependency, so it is imported lazily and a clear error
is raised if it is missing.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from insight_agent.core.agent import new_call_id
from insight_agent.core.llm import Message, ToolCall, Turn

DEFAULT_MODEL = "claude-sonnet-4-5"


class AnthropicClient:
    """Adapts the Anthropic Messages API to the agent's client interface."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("INSIGHT_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
        # Relay and gateway services proxy the Anthropic protocol unchanged, so
        # pointing at one is a base-url change rather than a provider change.
        self.base_url = (
            base_url
            or os.environ.get("INSIGHT_ANTHROPIC_BASE_URL")
            or os.environ.get("INSIGHT_BASE_URL")
            or None
        )
        self.max_tokens = max_tokens
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env, or export it."
            )
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "The anthropic package is required: uv sync --extra anthropic"
            ) from exc
        self._client = AsyncAnthropic(api_key=key, base_url=self.base_url)

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> Turn:
        system, wire_messages = _to_wire(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": wire_messages,
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["input_schema"],
                }
                for t in tools
            ]

        response = await self._client.messages.create(**request)
        return _from_wire(response)


def _to_wire(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Flatten the agent's message list into Anthropic's system + content form."""

    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []

    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
            continue

        if message.role == "tool":
            wire.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id or new_call_id(),
                            "content": message.content,
                        }
                    ],
                }
            )
            continue

        if message.role == "assistant" and message.tool_calls:
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            content.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in message.tool_calls
            )
            wire.append({"role": "assistant", "content": content})
            continue

        if message.content:
            wire.append({"role": message.role, "content": message.content})

    return "\n\n".join(p for p in system_parts if p), wire


def _from_wire(response: Any) -> Turn:
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.input or {}),
                )
            )
    return Turn(
        text="\n".join(text_parts).strip(),
        tool_calls=calls,
        stop_reason=getattr(response, "stop_reason", None) or "end_turn",
    )
