"""SQLite sandbox: schema loading, deterministic seeding, and guarded execution.

Execution safety is enforced at three independent levels so a bypass of any one
of them still cannot write:

1. SQL static analysis in :mod:`insight_agent.app.security` (catches intent).
2. A read-only connection URI (the file itself is opened ``mode=ro``).
3. SQLite's own ``set_authorizer`` callback, which denies non-read actions
   inside the engine. This is the backstop: even if a statement slipped past
   static analysis, SQLite refuses to execute it.
"""

from __future__ import annotations

import random
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# The sandbox dataset ends here. Anchoring "today" keeps relative-date queries
# ("last 7 days") deterministic, which the reproducibility requirement depends on.
DATA_START = date(2024, 1, 1)
DATA_END = date(2024, 12, 31)
AS_OF = DATA_END

CITIES = ("Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou", "Chengdu")
CHANNELS = ("app", "web", "mini_program")
PAY_METHODS = ("alipay", "wechat", "card")
ORDER_STATUSES = ("created", "paid", "shipped", "completed", "cancelled")
REFUND_REASONS = ("damaged", "wrong_item", "not_described", "late_delivery")

_CATEGORY_NAMES = (
    "Electronics",
    "Clothing",
    "Home",
    "Beauty",
    "Sports",
    "Books",
    "Toys",
    "Grocery",
)


@dataclass(slots=True)
class QueryResult:
    """A materialised result set plus the timing needed for eval metrics."""

    columns: list[str] = field(default_factory=list)
    rows: list[tuple[object, ...]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: float = 0.0

    def as_dicts(self) -> list[dict[str, object]]:
        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]

    def to_payload(self) -> dict[str, object]:
        return {
            "columns": self.columns,
            "rows": self.as_dicts(),
            "row_count": self.row_count,
            "truncated": self.truncated,
        }


class RowLimitExceeded(RuntimeError):
    """Raised when a query returns more rows than the sandbox allows."""


def default_db_path() -> Path:
    """Where the sandbox database lives, overridable via ``INSIGHT_DB_PATH``."""

    import os

    override = os.environ.get("INSIGHT_DB_PATH")
    if override:
        return Path(override)
    return Path(__file__).with_name("ecommerce.db")


def ensure_database(db_path: Path | None = None, *, seed: int = 42) -> Path:
    """Create and seed the sandbox database if it does not already exist."""

    path = db_path or default_db_path()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _seed(conn, seed=seed)
        conn.commit()
    return path


def _seed(conn: sqlite3.Connection, *, seed: int) -> None:
    """Populate a deterministic dataset.

    Determinism matters more than realism here: the evaluation compares result
    sets against gold queries, so the data must not drift between runs.
    """

    rng = random.Random(seed)

    conn.executemany(
        "INSERT INTO categories (id, name, parent_id) VALUES (?, ?, NULL)",
        [(i + 1, name) for i, name in enumerate(_CATEGORY_NAMES)],
    )

    users = []
    for uid in range(1, 401):
        register = DATA_START - timedelta(days=rng.randint(0, 500))
        users.append(
            (
                uid,
                f"user_{uid:04d}",
                rng.choice(CITIES),
                register.isoformat(),
                rng.choices(("bronze", "silver", "gold"), weights=(6, 3, 1))[0],
            )
        )
    conn.executemany(
        "INSERT INTO users (id, name, city, register_date, member_level) VALUES (?,?,?,?,?)",
        users,
    )

    products = []
    for pid in range(1, 121):
        category_id = rng.randint(1, len(_CATEGORY_NAMES))
        products.append(
            (
                pid,
                f"{_CATEGORY_NAMES[category_id - 1]}_item_{pid:03d}",
                category_id,
                round(rng.uniform(5, 2000), 2),
                rng.randint(0, 500),
            )
        )
    conn.executemany(
        "INSERT INTO products (id, name, category_id, price, stock) VALUES (?,?,?,?,?)",
        products,
    )

    total_days = (DATA_END - DATA_START).days + 1
    orders, items, payments, refunds = [], [], [], []
    order_id = item_id = pay_id = refund_id = 0

    for day_offset in range(total_days):
        day = DATA_START + timedelta(days=day_offset)
        # Weekends carry more traffic, so date-window questions have a signal.
        volume = rng.randint(18, 32) + (8 if day.weekday() >= 5 else 0)
        for _ in range(volume):
            order_id += 1
            user_id = rng.randint(1, 400)
            status = rng.choices(ORDER_STATUSES, weights=(5, 15, 15, 60, 5))[0]
            n_items = rng.randint(1, 4)
            total = 0.0
            order_rows = []
            for _ in range(n_items):
                item_id += 1
                product_id = rng.randint(1, 120)
                unit_price = products[product_id - 1][3]
                quantity = rng.randint(1, 3)
                total += unit_price * quantity
                order_rows.append((item_id, order_id, product_id, quantity, unit_price))
            items.extend(order_rows)
            orders.append(
                (
                    order_id,
                    user_id,
                    day.isoformat(),
                    status,
                    round(total, 2),
                    rng.choice(CHANNELS),
                )
            )

            pay_id += 1
            payments.append(
                (
                    pay_id,
                    order_id,
                    rng.choice(PAY_METHODS),
                    round(total, 2),
                    "success" if status in ("paid", "shipped", "completed") else "failed",
                )
            )

            if status == "completed" and rng.random() < 0.12:
                refund_id += 1
                refunds.append(
                    (
                        refund_id,
                        order_id,
                        round(total * rng.uniform(0.2, 1.0), 2),
                        rng.choice(REFUND_REASONS),
                        (day + timedelta(days=rng.randint(1, 14))).isoformat(),
                    )
                )

    conn.executemany(
        "INSERT INTO orders (id, user_id, order_date, status, total_amount, channel)"
        " VALUES (?,?,?,?,?,?)",
        orders,
    )
    conn.executemany(
        "INSERT INTO order_items (id, order_id, product_id, quantity, unit_price)"
        " VALUES (?,?,?,?,?)",
        items,
    )
    conn.executemany(
        "INSERT INTO payments (id, order_id, pay_method, pay_amount, pay_status)"
        " VALUES (?,?,?,?,?)",
        payments,
    )
    conn.executemany(
        "INSERT INTO refunds (id, order_id, amount, reason, refund_date) VALUES (?,?,?,?,?)",
        refunds,
    )


# --- Read-only execution -------------------------------------------------


def _authorizer(
    action: int,
    arg1: str | None,
    arg2: str | None,
    db_name: str | None,
    trigger: str | None,
) -> int:
    """Deny every non-read action. Returning anything but SQLITE_DENY blocks it."""

    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
    # Actions with no explicit allow entry are denied.
    if action in allowed:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() in {
        "table_info",
        "index_list",
        "foreign_key_list",
    }:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


# sqlite3.Connection does not support weak references, so this registry is keyed
# by identity and pruned by close().
_LOCKS: dict[int, threading.Lock] = {}


def _lock_for(conn: sqlite3.Connection) -> threading.Lock:
    """Return the mutex guarding ``conn``, creating it on first use."""

    lock = _LOCKS.get(id(conn))
    if lock is None:
        lock = threading.Lock()
        _LOCKS[id(conn)] = lock
    return lock


def locked(conn: sqlite3.Connection) -> Any:
    """Context manager serialising access to ``conn``.

    Any code that issues SQL directly must take this lock. ``execute`` already
    does, but reading the schema does not, and a reader that skips it deadlocks
    against a concurrent ``execute`` — which is exactly what happened when
    schema introspection was added to the request path.
    """

    return _lock_for(conn)


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open the sandbox read-only, with the engine-level write authorizer on.

    ``check_same_thread=False`` lets a threaded server (uvicorn's worker pool, or
    the test client) use a connection opened on another thread. That flag alone
    is *not* enough: it lifts the thread-affinity check without making the
    connection safe to use from two threads at once, which deadlocks — found by
    the concurrent stress test, not by inspection. :func:`execute` therefore
    serialises access behind a per-connection lock.

    Read-only status keeps that critical section safe: it never nests and never
    waits on anything the database itself might hold.
    """

    path = ensure_database(db_path)
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.set_authorizer(_authorizer)
    _lock_for(conn)
    return conn


def close(conn: sqlite3.Connection) -> None:
    """Close a connection and release its lock registry entry."""

    _LOCKS.pop(id(conn), None)
    conn.close()


def execute(
    conn: sqlite3.Connection,
    sql: str,
    *,
    limit: int = 500,
) -> QueryResult:
    """Run a single read statement and materialise its result set."""

    import time

    started = time.perf_counter()
    # One caller at a time per connection: concurrent use of a single
    # sqlite3.Connection from several threads deadlocks rather than erroring.
    with _lock_for(conn):
        cursor = conn.execute(sql)
        fetched: Sequence[sqlite3.Row] = cursor.fetchmany(limit + 1)
        description = cursor.description
    truncated = len(fetched) > limit
    rows = [tuple(row) for row in fetched[:limit]]
    columns = [d[0] for d in (description or [])]
    elapsed = (time.perf_counter() - started) * 1000

    return QueryResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        elapsed_ms=elapsed,
    )
