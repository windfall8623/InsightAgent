"""OpenAI-protocol implementation of :class:`~insight_agent.core.llm.LLMClient`.

Any endpoint speaking the OpenAI Chat Completions shape works: the official
API, Azure OpenAI, or a local server such as vLLM or Ollama's compatible mode.
``base_url`` is what makes that possible, and it is read from
``INSIGHT_BASE_URL`` so a self-hosted model needs no code change.

Note the two protocol differences this adapter absorbs, because they are the
usual source of silent bugs when swapping providers:

* **Role naming.** Anthropic takes a top-level ``system`` string; OpenAI takes a
  ``system``-role message inside ``messages``.
* **Tool-call shape.** Anthropic returns ``tool_use`` blocks with a JSON object
  in ``input``; OpenAI returns ``tool_calls`` with an **JSON string** in
  ``arguments`` that must be parsed before use.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from insight_agent.core.llm import Message, ToolCall, Turn

DEFAULT_MODEL = "gpt-4.1-mini"


class OpenAIClient:
    """Adapts the OpenAI Chat Completions API to the agent's client interface."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.environ.get("INSIGHT_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.temperature = temperature
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env.")
        self.base_url = base_url or os.environ.get("INSIGHT_BASE_URL") or None

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError("The openai package is required: uv sync --extra openai") from exc
        timeout = float(os.environ.get("INSIGHT_TIMEOUT", "120.0"))
        self._client = AsyncOpenAI(api_key=key, base_url=self.base_url, timeout=timeout)
        self.default_thinking = os.environ.get("INSIGHT_ENABLE_THINKING", "false").lower() in ("true", "1", "yes")

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
        thinking: bool | None = None,
    ) -> Turn:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": _to_wire(messages),
        }
        effective_thinking = self.default_thinking if thinking is None else thinking
        request["reasoning_effort"] = "high" if effective_thinking else "low"
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": _normalise_schema(t["input_schema"]),
                    },
                }
                for t in tools
            ]

        try:
            response = await self._client.chat.completions.create(**request)
        except Exception as exc:
            if "reasoning_effort" in request and getattr(exc, "status_code", None) == 400:
                request.pop("reasoning_effort", None)
                response = await self._client.chat.completions.create(**request)
            else:
                raise
        return _from_wire(response)

    async def complete_stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        thinking: bool | None = None,
    ):
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": _to_wire(messages),
            "stream": True,
        }
        effective_thinking = self.default_thinking if thinking is None else thinking
        request["reasoning_effort"] = "high" if effective_thinking else "low"
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": _normalise_schema(t["input_schema"]),
                    },
                }
                for t in tools
            ]
        try:
            response = await self._client.chat.completions.create(**request)
        except Exception as exc:
            if "reasoning_effort" in request and getattr(exc, "status_code", None) == 400:
                request.pop("reasoning_effort", None)
                response = await self._client.chat.completions.create(**request)
            else:
                raise
        async for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta:
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        yield {"type": "reasoning", "delta": delta.reasoning_content}
                    if delta.content:
                        yield {"type": "content", "delta": delta.content}


def _normalise_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert an internal schema to strict JSON Schema for OpenAI.

    ``strict`` requires every property to appear in ``required``, so optional
    fields are listed as nullable. Without this the API rejects the request.
    """

    properties = schema.get("properties", {})
    return {
        "type": "object",
        "properties": {
            name: {
                "type": (
                    spec["type"]
                    if spec["type"] in ("string", "number", "boolean", "array", "object")
                    else "string"
                ),
                **({"enum": spec["enum"]} if "enum" in spec else {}),
            }
            for name, spec in properties.items()
        },
        "required": sorted(properties),
        "additionalProperties": False,
    }


def _to_wire(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Flatten to Chat Completions roles.

    OpenAI expresses tool results as ``tool`` messages referencing the call id,
    where Anthropic nests them inside a ``user`` turn.
    """

    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            wire.append({"role": "system", "content": message.content})

        elif message.role == "tool":
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id or "",
                    "content": message.content,
                }
            )

        elif message.role == "assistant" and message.tool_calls:
            entry: dict[str, Any] = {"role": "assistant", "content": message.content or None}
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    # OpenAI requires arguments to be a JSON *string*.
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
            wire.append(entry)

        elif message.content:
            wire.append({"role": message.role, "content": message.content})

    return wire


def _from_wire(response: Any) -> Turn:
    """Read a chat completion back, tolerating a refused or empty response."""

    choices = getattr(response, "choices", None) or []
    if not choices:
        return Turn(text="", stop_reason="no_choices")

    message = choices[0].message
    text = getattr(message, "content", "") or ""

    calls: list[ToolCall] = []
    for raw in getattr(message, "tool_calls", None) or []:
        arguments: dict[str, Any] = {}
        raw_args = getattr(raw.function, "arguments", None)
        if raw_args:
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                # A malformed payload is the model's error, not ours; surface an
                # empty argument set rather than crashing the run.
                arguments = {}
        calls.append(
            ToolCall(id=getattr(raw, "id", "") or "", name=raw.function.name, arguments=arguments)
        )

    return Turn(
        text=text.strip(),
        tool_calls=calls,
        stop_reason=getattr(choices[0], "finish_reason", None) or "stop",
    )
