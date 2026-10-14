"""Protocol routing: infer the wire format from a single endpoint.

The problem this solves
-----------------------
A relay, gateway or self-hosted model server is reached with a base URL, a model
id and a key — but not with a protocol name. Asking the user to also state
``INSIGHT_PROVIDER`` makes them guess which of OpenAI Chat Completions or
Anthropic Messages the endpoint speaks, which is exactly the kind of
configuration trivia that produces "why do I get a 404".

So: probe, then remember.

Strategy
--------
The first real request is attempted with the preferred protocol. Only if it
fails with a *protocol-shaped* error does the router fall back and retry with
the other one. A normal API error (bad key, quota, timeout) is re-raised
immediately, because retrying it against the other protocol would just burn a
second request for the same failure.

The protocol that works is cached for the life of the client, so the cost is at
most one failed call per process.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from insight_agent.core.llm import Message, Turn


def looks_like_protocol_error(exc: BaseException) -> bool:
    """True when the failure suggests the endpoint speaks a different protocol.

    Only route on shape, not on message text alone: a 401 or a quota error means
    the protocol was right and the request was not.
    """

    status = getattr(exc, "status_code", None)
    if status in (404, 405, 501):
        return True

    name = type(exc).__name__.lower()
    if any(marker in name for marker in ("notfound", "methodnotallowed", "unsupported")):
        return True

    text = str(exc).lower()
    markers = (
        "not found",
        "no such endpoint",
        "unsupported route",
        "unknown path",
        "not a valid url",
        "invalid request url",
    )
    return any(marker in text for marker in markers)


class AutoClient:
    """Tries one protocol, falls back to the other, then sticks with the winner."""

    def __init__(
        self,
        *,
        clients: dict[str, Any],
        order: Sequence[str] = ("openai", "anthropic"),
    ) -> None:
        available = [name for name in order if name in clients]
        if not available:
            raise ValueError("no clients supplied")
        self._clients = clients
        self._order = available
        self._resolved: str | None = None

    @property
    def model(self) -> str:
        name = self._resolved or self._order[0]
        return getattr(self._clients[name], "model", "llm")

    @property
    def protocol(self) -> str | None:
        """The protocol that worked, once one has."""

        return self._resolved

    def describe(self) -> dict[str, Any]:
        return {
            "protocol": self._resolved,
            "order": list(self._order),
            "model": self.model,
        }

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
        thinking: bool | None = None,
    ) -> Turn:
        last: BaseException | None = None

        for name in self._order:
            if self._resolved and name != self._resolved:
                continue
            try:
                turn: Turn = await self._clients[name].complete(messages, tools, thinking=thinking)
            except TypeError:
                turn = await self._clients[name].complete(messages, tools)
            except Exception as exc:  # noqa: BLE001 - re-raised below
                if self._resolved or not looks_like_protocol_error(exc):
                    raise
                last = exc
                continue
            self._resolved = name
            return turn

        # Every candidate looked like the wrong protocol.
        raise RuntimeError(
            "could not speak to the configured endpoint; tried "
            f"{', '.join(self._order)}. Last error: {last}"
        )

    async def complete_stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        thinking: bool | None = None,
    ):
        name = self._resolved or self._order[0]
        client = self._clients[name]
        if hasattr(client, "complete_stream"):
            try:
                async for item in client.complete_stream(messages, tools, thinking=thinking):
                    yield item
            except TypeError:
                async for item in client.complete_stream(messages, tools):
                    yield item
        else:
            turn = await self.complete(messages, tools or (), thinking=thinking)
            if turn and turn.text:
                yield {"type": "content", "delta": turn.text}


def build_auto_client(config: Any) -> AutoClient:
    """Assemble both clients and let the router choose between them."""

    from insight_agent.core.providers.anthropic import AnthropicClient
    from insight_agent.core.providers.openai_client import OpenAIClient

    clients: dict[str, Any] = {}

    if config.anthropic_api_key:
        try:
            from insight_agent.core.providers.anthropic import AnthropicClient

            clients["anthropic"] = AnthropicClient(
                model=config.model or None,
                api_key=config.anthropic_api_key,
                max_tokens=config.max_tokens,
                base_url=config.anthropic_base_url or None,
            )
        except Exception:
            pass

    if config.openai_api_key:
        try:
            from insight_agent.core.providers.openai_client import OpenAIClient

            clients["openai"] = OpenAIClient(
                model=config.model or None,
                api_key=config.openai_api_key,
                base_url=config.openai_base_url or None,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        except Exception:
            pass

    if not clients:
        raise RuntimeError("no API key configured; cannot build a live client")

    return AutoClient(clients=clients)
