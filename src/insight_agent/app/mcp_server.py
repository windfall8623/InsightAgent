"""A real MCP (Model Context Protocol) server over the same tool registry.

This speaks JSON-RPC 2.0 over stdio with the MCP method names — ``initialize``,
``tools/list``, ``tools/call`` — rather than exposing the tools only as Python
functions. The point is that a genuine MCP client can drive the analytics tools
without knowing anything about this codebase.

Wire format: newline-delimited JSON-RPC on stdin/stdout.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from insight_agent.core.llm import ToolCall
from insight_agent.core.tools import ToolRegistry

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "insight-analytics", "version": "0.2.0"}


class MCPServer:
    """Serves a :class:`ToolRegistry` over MCP."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._initialised = False

    # -- request handling -------------------------------------------------

    async def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC request. Returns ``None`` for notifications."""

        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}

        if request_id is None:
            # Notification: no response expected.
            return None

        result: dict[str, Any]
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                }
                self._initialised = True
            elif method == "tools/list":
                listed: list[dict[str, Any]] = [
                    {
                        "name": s["name"],
                        "description": s["description"],
                        "inputSchema": s["input_schema"],
                    }
                    for s in self.registry.schemas()
                ]
                result = {"tools": listed}
            elif method == "tools/call":
                result = await self._call_tool(params)
            elif method == "ping":
                result = {}
            else:
                return _error(request_id, -32601, f"method not found: {method}")
        except Exception as exc:  # noqa: BLE001 - protocol-level error report
            return _error(request_id, -32603, f"{type(exc).__name__}: {exc}")

        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not name:
            raise ValueError("tools/call requires a 'name'")
        call = ToolCall(
            id=str(params.get("id", "mcp-1")),
            name=name,
            arguments=params.get("arguments") or {},
        )
        outcome = await self.registry.run(call)
        return {
            "content": [{"type": "text", "text": outcome.content}],
            "isError": outcome.is_error,
        }

    # -- stdio loop -------------------------------------------------------

    def serve(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        """Run the newline-delimited JSON-RPC loop until stdin closes."""

        import asyncio

        asyncio.run(self._serve_loop(stdin or sys.stdin, stdout or sys.stdout))

    def __repr__(self) -> str:
        return f"MCPServer({len(self.registry)} tools)"

    async def _serve_loop(self, stdin: TextIO, stdout: TextIO) -> None:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                stdout.write(json.dumps(_error(None, -32700, str(exc))) + "\n")
                stdout.flush()
                continue
            response = await self.handle(message)
            if response is not None:
                stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                stdout.flush()


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def serve_tools(registry: ToolRegistry) -> None:
    """Serve an explicit registry over stdio.

    Synchronous on purpose: console scripts are invoked with no arguments and
    their return value discarded, so an ``async def`` here would return an
    un-awaited coroutine and the server would never start.
    """

    MCPServer(registry).serve()


def main() -> None:
    """Zero-argument console entry point for ``insight-mcp``.

    Builds the active database connection and comprehensive diagnostic tool registry
    (auto-detecting CN-Retail or Olist), then serves over standard I/O.
    """

    from insight_agent.app.tools import build_active_mcp_tool_set

    MCPServer(build_active_mcp_tool_set()).serve()
