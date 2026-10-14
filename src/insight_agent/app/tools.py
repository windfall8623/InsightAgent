"""The agent's tool surface.

Four tools, bound to a per-run context (connection, retriever, memory) so that
tool functions stay plain callables with an inspectable signature — which is
what lets the JSON Schema be derived automatically and exposed over MCP without
a second declaration.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from insight_agent.app.rag import HybridRetriever
from insight_agent.core.tools import Tool, ToolRegistry
from insight_agent.core.tools import tool as make_tool
from insight_agent.data.database import QueryResult, execute


@dataclass(slots=True)
class ToolContext:
    """Per-run state shared by the tool closures."""

    conn: sqlite3.Connection
    retriever: HybridRetriever
    user_id: str = "anonymous"
    session_id: str | None = None
    last_result: QueryResult | None = None
    max_rows: int = 500
    events: list[dict[str, Any]] = field(default_factory=list)


def build_tool_set(ctx: ToolContext) -> ToolRegistry:
    """Construct the registry of tools available to the agents."""

    def list_tables() -> str:
        """List every table in the e-commerce database with a one-line summary."""

        rows = ctx.conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        return ", ".join(r[0] for r in rows)

    def get_schema(table: str) -> str:
        """Return the column names and types for one table."""

        if not table.replace("_", "").isalnum():
            return f"invalid table name: {table!r}"
        rows = ctx.conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        if not rows:
            available = ", ".join(
                r[0]
                for r in ctx.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            )
            return f"unknown table {table!r}; available: {available}"
        return ", ".join(f"{r[1]}:{r[2]}" for r in rows)

    def run_sql(sql: str, limit: int = 500) -> str:
        """Execute one read-only SELECT and return the rows as JSON."""

        result = execute(ctx.conn, sql, limit=min(limit, ctx.max_rows))
        ctx.last_result = result
        return _dump(result)

    def describe_result() -> str:
        """Describe the columns and row count of the most recent query result."""

        if ctx.last_result is None:
            return "no query has been executed yet"
        result = ctx.last_result
        summary = ", ".join(f"{c}" for c in result.columns)
        return f"columns: [{summary}]; rows: {result.row_count}"

    return ToolRegistry(
        [
            make_tool(list_tables),
            make_tool(get_schema),
            make_tool(run_sql),
            make_tool(describe_result),
        ]
    )


def _dump(result: QueryResult) -> str:
    import json

    payload = result.to_payload()
    if result.truncated:
        payload["note"] = f"truncated to {result.row_count} rows"
    return json.dumps(payload, ensure_ascii=False, default=str)


def build_olist_diagnostic_tool_set(
    conn: sqlite3.Connection | None = None,
    redis_store: Any | None = None,
) -> ToolRegistry:
    """Build tool set including both Olist SQL/review tools and LangGraph diagnostics."""
    import json
    import time

    from insight_agent.app.graph import InsightGraphEngine
    from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
    from insight_agent.app.redis_store import RedisMemoryStore
    from insight_agent.data.olist_db import connect_olist, execute_olist

    db_conn = conn or connect_olist()
    retriever = OlistMultiModalRetriever(db_conn)
    store = redis_store or RedisMemoryStore()
    engine = InsightGraphEngine(db_conn, retriever, store)

    def list_tables() -> str:
        """List every table in the e-commerce database with a one-line summary."""
        rows = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return ", ".join(r[0] for r in rows)

    def get_schema(table: str) -> str:
        """Return the column names and types for one table."""
        if not table.replace("_", "").isalnum():
            return f"invalid table name: {table!r}"
        rows = db_conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        if not rows:
            return f"unknown table {table!r}"
        return ", ".join(f"{r[1]}:{r[2]}" for r in rows)

    def query_olist_sql(sql: str, limit: int = 100) -> str:
        """Execute a read-only query over the Olist 9-table schema and return JSON."""
        result = execute_olist(db_conn, sql, limit=limit)
        return json.dumps(result.to_payload(), ensure_ascii=False, default=str)

    def search_customer_reviews(query: str, top_k: int = 5) -> str:
        """Search authentic customer review complaints using BM25 and vector fusion."""
        docs = retriever.retrieve_reviews(query, top_k=top_k)
        payload = [
            {
                "score": round(d.score, 4),
                "snippet": d.text,
                "title": d.title,
                "rating": d.metadata.get("review_score"),
            }
            for d in docs
        ]
        return json.dumps(payload, ensure_ascii=False)

    def diagnose_anomaly(anomaly_description: str, hitl_approved: bool = False) -> str:
        """Run full 2.0 LangGraph StateGraph root-cause diagnostic on a business anomaly."""
        state = engine.diagnose(anomaly_description, hitl_approved=hitl_approved)
        return json.dumps(
            {
                "root_cause": state.get("root_cause"),
                "intent": state.get("intent"),
                "reflections": state.get("reflections"),
                "action_result": state.get("action_result"),
                "report": state.get("final_report"),
                "blocked": state.get("blocked"),
            },
            ensure_ascii=False,
        )

    def issue_voucher(
        user_id: str,
        amount: float = 0.0,
        reason: str = "",
        *,
        amount_brl: float | None = None,
        amount_cny: float | None = None,
    ) -> str:
        """Execute remediation voucher issuance protected by Redis distributed lock."""
        actual_amount = amount_brl if amount_brl is not None else (amount_cny if amount_cny is not None else amount)
        lock_id = f"voucher_{user_id}_{int(time.time())}"
        with store.acquire_action_lock(lock_id, ttl_seconds=15) as acquired:
            if not acquired:
                return json.dumps({"status": "blocked", "reason": "distributed lock collision"})
            return json.dumps(
                {
                    "status": "success",
                    "user_id": user_id,
                    "amount": actual_amount,
                    "amount_cny": actual_amount,
                    "amount_brl": actual_amount,
                    "reason": reason,
                    "lock_id": lock_id,
                }
            )

    return ToolRegistry(
        [
            make_tool(list_tables),
            make_tool(get_schema),
            make_tool(query_olist_sql),
            make_tool(search_customer_reviews),
            make_tool(diagnose_anomaly),
            make_tool(issue_voucher),
        ]
    )


def build_cn_retail_diagnostic_tool_set(
    conn: sqlite3.Connection | None = None,
    redis_store: Any | None = None,
) -> ToolRegistry:
    """Build tool set including both CN-Retail SQL/review tools and LangGraph diagnostics."""
    import json
    import time

    from insight_agent.app.graph import InsightGraphEngine
    from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
    from insight_agent.app.redis_store import RedisMemoryStore
    from insight_agent.data.cn_retail_db import (
        connect_cn_retail,
        ensure_cn_retail_database,
        execute_cn_retail,
    )

    ensure_cn_retail_database()
    db_conn = conn or connect_cn_retail()
    retriever = OlistMultiModalRetriever(db_conn)
    store = redis_store or RedisMemoryStore()
    engine = InsightGraphEngine(db_conn, retriever, store)

    def list_tables() -> str:
        """List every table in the e-commerce database with a one-line summary."""
        rows = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return ", ".join(r[0] for r in rows)

    def get_schema(table: str) -> str:
        """Return the column names and types for one table."""
        if not table.replace("_", "").isalnum():
            return f"invalid table name: {table!r}"
        rows = db_conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        if not rows:
            return f"unknown table {table!r}"
        return ", ".join(f"{r[1]}:{r[2]}" for r in rows)

    def query_cn_retail_sql(sql: str, limit: int = 100) -> str:
        """Execute a read-only query over the CN-Retail 8-table schema and return JSON."""
        result = execute_cn_retail(db_conn, sql, limit=limit)
        return json.dumps(result.to_payload(), ensure_ascii=False, default=str)

    def search_customer_reviews(query: str, top_k: int = 5) -> str:
        """Search authentic customer review complaints using BM25 and vector fusion."""
        docs = retriever.retrieve_reviews(query, top_k=top_k)
        payload = [
            {
                "score": round(d.score, 4),
                "snippet": d.text,
                "title": d.title,
                "rating": d.metadata.get("review_score"),
            }
            for d in docs
        ]
        return json.dumps(payload, ensure_ascii=False)

    def diagnose_anomaly(anomaly_description: str, hitl_approved: bool = False) -> str:
        """Run full 2.0 LangGraph StateGraph root-cause diagnostic on a business anomaly."""
        state = engine.diagnose(anomaly_description, hitl_approved=hitl_approved)
        return json.dumps(
            {
                "root_cause": state.get("root_cause"),
                "intent": state.get("intent"),
                "reflections": state.get("reflections"),
                "action_result": state.get("action_result"),
                "report": state.get("final_report"),
                "blocked": state.get("blocked"),
            },
            ensure_ascii=False,
        )

    def issue_voucher(
        user_id: str,
        amount: float = 0.0,
        reason: str = "",
        *,
        amount_brl: float | None = None,
        amount_cny: float | None = None,
    ) -> str:
        """Execute remediation voucher issuance protected by Redis distributed lock."""
        actual_amount = (
            amount_cny if amount_cny is not None else (amount_brl if amount_brl is not None else amount)
        )
        lock_id = f"voucher_{user_id}_{int(time.time())}"
        with store.acquire_action_lock(lock_id, ttl_seconds=15) as acquired:
            if not acquired:
                return json.dumps({"status": "blocked", "reason": "distributed lock collision"})
            return json.dumps(
                {
                    "status": "success",
                    "user_id": user_id,
                    "amount": actual_amount,
                    "amount_cny": actual_amount,
                    "amount_brl": actual_amount,
                    "reason": reason,
                    "lock_id": lock_id,
                }
            )

    return ToolRegistry(
        [
            make_tool(list_tables),
            make_tool(get_schema),
            make_tool(query_cn_retail_sql),
            make_tool(search_customer_reviews),
            make_tool(diagnose_anomaly),
            make_tool(issue_voucher),
        ]
    )


def build_active_mcp_tool_set(
    conn: sqlite3.Connection | None = None,
    redis_store: Any | None = None,
) -> ToolRegistry:
    """Build the appropriate MCP tool registry based on the active dataset or connection.

    Automatically inspects table structures to detect whether the dataset is
    CN-Retail, Olist, or standard OLTP.
    """
    from insight_agent.api import _get_active_dataset_conn

    target_conn = conn
    ds_name = ""
    if target_conn is None:
        target_conn, ds_name = _get_active_dataset_conn()

    table_names: list[str] = []
    with contextlib.suppress(Exception):
        table_names = [
            r[0]
            for r in target_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]

    if any(t.startswith("cn_") for t in table_names) or ds_name == "cn_retail":
        return build_cn_retail_diagnostic_tool_set(target_conn, redis_store)
    if any(t.startswith("olist_") for t in table_names) or ds_name == "olist":
        return build_olist_diagnostic_tool_set(target_conn, redis_store)

    from insight_agent.app.rag import build_retriever

    ctx = ToolContext(conn=target_conn, retriever=build_retriever(target_conn))
    return build_tool_set(ctx)


def registry_to_mcp_tools(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Project tool schemas into the MCP ``tools/list`` response shape."""

    return [
        {
            "name": s["name"],
            "description": s["description"],
            "inputSchema": s["input_schema"],
        }
        for s in registry.schemas()
    ]


__all__ = [
    "Tool",
    "ToolContext",
    "build_tool_set",
    "build_olist_diagnostic_tool_set",
    "build_cn_retail_diagnostic_tool_set",
    "build_active_mcp_tool_set",
    "registry_to_mcp_tools",
    "Callable",
]

