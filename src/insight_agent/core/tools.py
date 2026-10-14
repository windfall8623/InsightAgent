"""Tool registration, JSON-schema generation, and dispatch.

Tools are plain functions decorated with :func:`tool`. The signature, type
hints, and docstring are turned into a schema the model can call against, and
the same function is used to execute the call.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from insight_agent.core.llm import ToolCall, ToolResult, dumps

_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolError(Exception):
    """Raised by a tool to tell the model the call was invalid."""


R = TypeVar("R")


@dataclass(slots=True)
class Tool:
    """A callable exposed to the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    is_async: bool

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    async def invoke(self, call: ToolCall) -> ToolResult:
        try:
            result = self.func(**call.arguments)
            if inspect.isawaitable(result):
                result = await result
        except ToolError as exc:
            return ToolResult(call_id=call.id, name=self.name, content=str(exc), is_error=True)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            return ToolResult(
                call_id=call.id,
                name=self.name,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
        # A tool that already returns text is passed through verbatim; encoding
        # it again would double-quote the string and corrupt structured payloads.
        content = result if isinstance(result, str) else dumps(result)
        return ToolResult(call_id=call.id, name=self.name, content=content)


class ToolRegistry:
    """A name-keyed collection of tools, with parallel and batch dispatch."""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def register(self, tool_: Tool) -> None:
        self._tools[tool_.name] = tool_

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    async def run(self, call: ToolCall) -> ToolResult:
        found = self.get(call.name)
        if found is None:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"Unknown tool: {call.name}",
                is_error=True,
            )
        return await found.invoke(call)

    async def run_all(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        return [await self.run(call) for call in calls]


def _json_type(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "string"
    origin = typing.get_origin(annotation)
    if origin is not None:
        if origin in _JSON_TYPES:
            return _JSON_TYPES[origin]
        if origin is typing.Union or str(origin) == "types.UnionType":
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            return _json_type(args[0]) if args else "string"
        return "string"
    return _JSON_TYPES.get(annotation, "string")


def build_parameters(func: Callable[..., Any]) -> dict[str, Any]:
    """Derive a JSON Schema object for the callable's keyword arguments."""

    signature = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:  # noqa: BLE001 - unresolvable forward refs shouldn't break registration
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if name in {"self", "cls"} or param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        annotation = hints.get(name, param.annotation)
        properties[name] = {"type": _json_type(annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


@overload
def tool(func: Callable[..., R], /) -> Tool: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., R]], Tool]: ...


def tool(
    func: Callable[..., R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., R]], Tool]:
    """Turn a function into a :class:`Tool`.

    Usable bare (``@tool``) or with options (``@tool(name="now")``). The
    docstring's first line becomes the tool description unless overridden.
    """

    def decorate(target: Callable[..., R]) -> Tool:
        doc = inspect.getdoc(target) or ""
        return Tool(
            name=name or target.__name__,
            description=description or doc.split("\n\n")[0] or target.__name__,
            parameters=build_parameters(target),
            func=target,
            is_async=inspect.iscoroutinefunction(target),
        )

    if func is None:
        return decorate
    return decorate(func)


__all__ = ["Tool", "ToolError", "ToolRegistry", "build_parameters", "tool"]

# Re-exported for callers that annotate their own tool functions.
SyncOrAsync = Callable[..., Any] | Callable[..., Awaitable[Any]]
