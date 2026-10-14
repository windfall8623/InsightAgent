"""Sandbox dataset and guarded database access."""

from insight_agent.data.database import (
    AS_OF,
    DATA_END,
    DATA_START,
    QueryResult,
    close,
    connect,
    default_db_path,
    ensure_database,
    execute,
)

__all__ = [
    "AS_OF",
    "DATA_END",
    "DATA_START",
    "QueryResult",
    "close",
    "connect",
    "default_db_path",
    "ensure_database",
    "execute",
]
