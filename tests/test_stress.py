"""Stress and resource-exhaustion tests.

These answer a different question from the functional suite: *what happens when
the system is pushed beyond its happy path*. The red-team file checks whether
attacks are refused; this file checks whether legitimate load degrades
gracefully and whether unbounded inputs are contained rather than crashing.

The threat model here is untrusted or careless *legitimate* usage — a huge
result set, a pathological query, a thousand concurrent callers — not an
attacker. Attacker cases live in ``test_security_redteam.py``.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from insight_agent.app.memory import MemoryStore
from insight_agent.app.orchestrator import InsightOrchestrator, InsightRequest
from insight_agent.app.rag import build_retriever
from insight_agent.app.router import ResponseCache
from insight_agent.app.tools import ToolContext, build_tool_set
from insight_agent.core.llm import ToolCall


class TestThroughput:
    async def test_concurrent_queries(self, conn) -> None:
        """32 concurrent questions must all succeed and agree on the answer."""

        orchestrator = InsightOrchestrator(conn, memory=MemoryStore())
        questions = [
            "总销售额是多少",
            "最近7天各城市销售额",
            "各品类的退款率",
            "订单总数是多少",
        ] * 8

        started = time.perf_counter()
        results = await asyncio.gather(
            *(
                orchestrator.run(InsightRequest(question=q, user_id=f"u{i}"))
                for i, q in enumerate(questions)
            )
        )
        elapsed = (time.perf_counter() - started) * 1000

        assert all(not r.blocked for r in results)
        # All four question texts must map to the same answer regardless of order.
        by_question: dict[str, set[str]] = {}
        for r in results:
            by_question.setdefault(r.question, set()).add(r.sql)
        assert all(len(v) == 1 for v in by_question.values())
        print(f"\n32 concurrent queries in {elapsed:.1f} ms ({32 / (elapsed / 1000):.0f} q/s)")

    async def test_sequential_throughput_is_stable(self, conn) -> None:
        """Repeated identical calls must not degrade (no cache/memory growth)."""

        orchestrator = InsightOrchestrator(conn, memory=MemoryStore())
        for _ in range(3):
            await orchestrator.run(InsightRequest(question="总销售额是多少", user_id="u"))

        started = time.perf_counter()
        for _ in range(60):
            await orchestrator.run(InsightRequest(question="各城市销售额", user_id="u"))
        per_call = (time.perf_counter() - started) / 60 * 1000

        # A degenerate growth curve (memory or cache leak) shows up here as a
        # per-call cost far above the single-call baseline.
        assert per_call < 50, f"per-call time degraded to {per_call:.1f} ms"


class TestResourceContainment:
    async def test_unbounded_result_is_capped(self, conn) -> None:
        """A query returning far more rows than the cap must be truncated."""

        ctx = ToolContext(conn=conn, retriever=build_retriever(conn), max_rows=100)
        registry = build_tool_set(ctx)
        outcome = await registry.run(
            ToolCall(id="1", name="run_sql", arguments={"sql": "SELECT * FROM order_items"})
        )
        import json

        payload = json.loads(outcome.content)
        assert payload["truncated"] is True
        assert payload["row_count"] == 100
        assert "truncated" in payload["note"]

    async def test_cartesian_product_is_gated_not_executed(self, conn) -> None:
        """A pathological query must not run before human approval."""

        orchestrator = InsightOrchestrator(conn)
        orchestrator.approver = lambda call, decision: False
        result = await orchestrator.tools.run(
            ToolCall(
                id="1",
                name="run_sql",
                arguments={"sql": "SELECT * FROM orders, users, order_items"},
            )
        )
        assert result.is_error
        assert "DECLINED" in result.content

    async def test_cache_is_bounded(self, conn) -> None:
        cache = ResponseCache(max_entries=16)
        for i in range(200):
            cache.put("u", f"SELECT {i}", i)
        assert len(cache._store) == 16
        # The most recent entries must survive eviction.
        assert cache.get("u", "SELECT 199") == 199

    def test_episodic_memory_is_bounded(self) -> None:
        store = MemoryStore(episodic_limit=10)
        for i in range(100):
            store.remember("u", f"question {i}", f"SELECT {i}")
        assert len(store.profile("u").episodes) == 10


class TestMalformedInput:
    async def test_very_long_question_is_handled(self, conn) -> None:
        orchestrator = InsightOrchestrator(conn)
        result = await orchestrator.run(
            InsightRequest(question="总销售额是多少 " + "无关内容 " * 5000, user_id="u")
        )
        assert result.row_count in (0, 1)
        assert isinstance(result.answer, str)

    async def test_unicode_and_control_characters(self, conn) -> None:
        orchestrator = InsightOrchestrator(conn)
        for text in (
            "总销售额\x00\x00是多少",
            "总销售额\n\n\t是多少",
            "销售额" * 500,
            "🙂🙃 销售额是多少",
        ):
            result = await orchestrator.run(InsightRequest(question=text, user_id="u"))
            assert isinstance(result.answer, str)

    async def test_sql_metacharacters_in_the_question_do_not_execute(self, conn) -> None:
        """Question text must never reach the SQL string unmediated."""

        orchestrator = InsightOrchestrator(conn)
        await orchestrator.run(
            InsightRequest(question="销售额'; DROP TABLE orders; --", user_id="u")
        )
        # Either it is understood and parameterised, or it is refused — but the
        # table must still be there either way.
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] > 0

    async def test_empty_and_whitespace_questions(self, conn) -> None:
        orchestrator = InsightOrchestrator(conn)
        for text in ("", "   ", "\n\t"):
            result = await orchestrator.run(InsightRequest(question=text, user_id="u"))
            assert result.sql == ""

    async def test_malformed_sql_is_reported_not_crashed(self, conn) -> None:
        ctx = ToolContext(conn=conn, retriever=build_retriever(conn))
        registry = build_tool_set(ctx)
        outcome = await registry.run(
            ToolCall(id="1", name="run_sql", arguments={"sql": "SELECT FROM WHERE"})
        )
        assert outcome.is_error
        assert outcome.content


class TestApiStress:
    def test_http_concurrent_requests(self, conn) -> None:
        """Drive real HTTP concurrency against a real server.

        This deliberately does not use ``TestClient``: that is an in-process
        shim whose anyio portal is bound to the creating thread, so sharing one
        across threads hangs on its own machinery and tells us nothing about the
        application. A real uvicorn server on a real socket is what the
        production claim needs, and it also keeps this test honest about what is
        being measured.
        """

        pytest.importorskip("uvicorn")

        import json
        import threading
        import urllib.request

        import uvicorn

        from insight_agent.api import create_app

        app = create_app(InsightOrchestrator(conn))
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        deadline = time.perf_counter() + 30
        while not server.started and time.perf_counter() < deadline:
            time.sleep(0.1)
        if not server.started:  # pragma: no cover - environment dependent
            pytest.skip("uvicorn did not start in this environment")

        port = server.servers[0].sockets[0].getsockname()[1]
        try:

            def call(_: int) -> int:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/ask",
                    data=json.dumps({"question": "总销售额是多少"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    return response.status

            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=16) as pool:
                codes = list(pool.map(call, range(32)))
            elapsed = (time.perf_counter() - started) * 1000
        finally:
            server.should_exit = True
            thread.join(timeout=10)

        assert all(code == 200 for code in codes)
        print(f"{chr(10)}32 concurrent HTTP requests in {elapsed:.1f} ms")


class TestThreadSafety:
    """Regression: concurrent threads sharing one connection deadlocked.

    ``check_same_thread=False`` removes the thread-affinity error but does not
    make a sqlite3 connection safe to use from several threads at once — two
    callers would interleave on the same connection and hang. ``execute`` now
    serialises access behind a per-connection lock; this test fails (by timeout)
    if that lock is ever removed.
    """

    def test_concurrent_threads_do_not_deadlock(self, conn) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from insight_agent.data.database import execute as run_sql

        def call(i: int) -> int:
            return run_sql(conn, "SELECT COUNT(*) AS n FROM orders").rows[0][0]

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(call, range(32)))

        assert len(results) == 32
        assert len(set(results)) == 1, "concurrent reads disagreed"
