"""Offline regression tests for independent benchmark attempts and their caches."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from insight_agent.app.evolution import EvolutionEngine, Outcome, default_strategies
from insight_agent.app.memory import MemoryStore
from insight_agent.app.orchestrator import InsightAnswer, InsightOrchestrator, RunConfig
from insight_agent.app.planner import LLMPlanner
from insight_agent.core.llm import Turn
from insight_agent.eval import run_cn_retail_200_eval as evaluation

CASES = [
    {
        "id": "case-a",
        "question": "查询订单指标，独立问题甲",
        "expectation": "rows",
        "gold_sql": "SELECT 1 AS n /* GOLD_ONLY_EXPECTATION */",
        "requires_tables": [],
    },
    {
        "id": "case-b",
        "question": "查询销售指标，独立问题乙",
        "expectation": "rows",
        "gold_sql": "SELECT 1 AS n /* GOLD_ONLY_EXPECTATION */",
        "requires_tables": [],
    },
]


@pytest.fixture(autouse=True)
def isolated_eval_env(monkeypatch, tmp_path):
    # No local credentials, service discovery, model downloads, or live probes.
    for key in (
        "REDIS_URL", "INSIGHT_REDIS_URL", "QDRANT_URL", "QDRANT_API_KEY",
        "INSIGHT_EMBEDDING_URL", "INSIGHT_RERANKER_URL", "INSIGHT_JEV_URL",
        "INSIGHT_ONNX_RERANKER_PATH", "INSIGHT_ONNX_TOKENIZER_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("INSIGHT_BGE_ENABLED", "0")
    monkeypatch.setattr(evaluation, "STRATEGY_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(evaluation, "CN_RETAIL_200_CASES", [dict(case) for case in CASES])
    monkeypatch.setattr(
        evaluation, "get_default_bge_adapter", lambda: SimpleNamespace(remote_fallback_count=0)
    )


@pytest.fixture
def case_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE eval_numbers (n INTEGER)")
    conn.execute("INSERT INTO eval_numbers VALUES (1)")
    yield conn
    conn.close()


def answer(*, sql="SELECT 1 AS n", diagnostics=None, blocked=False):
    return InsightAnswer(
        question="", answer="抱歉，未能识别指标" if not sql else "指标为 1",
        sql=sql, blocked=blocked, block_reason="policy denied" if blocked else "",
        model="spy-model", planner="llm", route_reason="spy-route", route_score=7,
        agents_used=["planner"], cost_usd=0.01,
        planning_diagnostics=diagnostics or {},
    )


class ScopeSpy:
    """Honor the public scope contract while recording every attempt's reads.

    Write every memory layer even before raising: retries must discard partially
    completed state, not just state from successful questions.
    """

    def __init__(self, conn, *, memory=None, config=None, retriever=None, planner=None, replies=()):
        self.config = config or RunConfig()
        self.memory = memory or MemoryStore(local_only=True)
        self.retriever = retriever
        self.planner = planner or SimpleNamespace(name="llm")
        self.replies = list(replies)
        self.requests = []
        self.starts = []
        self.memories = []

    @contextmanager
    def case_memory_scope(self, memory):
        self.memories.append(memory)

        async def run(request, *, on_event=None):
            self.requests.append(request)
            self.starts.append((
                memory.get_history_messages(request.session_id),
                dict(memory.get_session_state(request.session_id)),
                memory.filters(request.user_id),
                memory.recall(request.user_id, request.question),
                memory.build_context(request.user_id, request.session_id),
            ))
            assert memory.redis_store is None
            assert memory.path is None
            memory.remember_turn(request.session_id, "user", request.question)
            memory.set_session_state(request.session_id, "standing_filters", {"city": "old"})
            memory.set_filter(request.user_id, "city", "old")
            memory.add_fact(request.user_id, "prior attempt fact")
            memory.remember(request.user_id, request.question, "SELECT 'OLD_EXAMPLE'")
            if on_event is not None:
                on_event({"stage": "plan"})
            await asyncio.sleep(0)
            reply = self.replies.pop(0) if self.replies else answer()
            if isinstance(reply, Exception):
                raise reply
            return reply

        yield SimpleNamespace(run=run)


def assert_fresh_attempts(orch, count):
    assert len(orch.requests) == count
    assert len({request.user_id for request in orch.requests}) == count
    assert len({request.session_id for request in orch.requests}) == count
    assert all(request.user_id != request.session_id for request in orch.requests)
    assert len({id(memory) for memory in orch.memories}) == count
    assert orch.starts == [([], {}, {}, [], "")] * count


@pytest.mark.parametrize("strategy", default_strategies(), ids=lambda s: s.name)
async def test_sweep_and_retry_start_clean_without_changing_strategy(strategy, case_conn, monkeypatch):
    created = []
    before_config = asdict(strategy.config)

    def factory(conn, **kwargs):
        orch = ScopeSpy(conn, **kwargs, replies=[TimeoutError("transient"), answer(sql="SELECT 2 AS n"), answer()])
        created.append(orch)
        return orch

    monkeypatch.setattr(evaluation, "InsightOrchestrator", factory)
    monkeypatch.setattr(evaluation, "build_retriever", lambda conn: object())
    outcomes, _, _ = await evaluation.run_sweep(
        case_conn, [strategy], planner=SimpleNamespace(name="llm"),
        max_repair_rounds=2, repair_pause=0,
    )
    orch = created[0]
    assert_fresh_attempts(orch, 3)
    assert [r.question for r in orch.requests] == [CASES[0]["question"], CASES[1]["question"], CASES[0]["question"]]
    assert asdict(orch.config) == asdict(strategy.config) == before_config
    first, second = outcomes[strategy.name]
    assert first.passed and first.retry_count == 1
    assert first.attempt_history[0]["llm_error"] is True
    assert first.attempt_history[0]["execution_diagnostics"]["failure_stage"] == "plan"
    # A legitimate wrong answer is not re-rolled until it passes.
    assert not second.passed and not second.llm_error and second.retry_count == 0
    assert evaluation._cache_is_clean(strategy.name, live=True, config=strategy.config)


class MemorySpy(MemoryStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history_reads = []
        self.filter_reads = []
        self.episode_reads = []

    def get_history_messages(self, session_id):
        result = super().get_history_messages(session_id)
        self.history_reads.append((session_id, list(result)))
        return result

    def filters(self, user_id):
        result = super().filters(user_id)
        self.filter_reads.append((user_id, dict(result)))
        return result

    def recall(self, user_id, question, top_k=3):
        result = super().recall(user_id, question, top_k)
        self.episode_reads.append((user_id, list(result)))
        return result


class ClientSpy:
    model = "offline-spy"

    def __init__(self):
        self.planning_messages = []

    async def complete(self, messages, tools):
        if "You translate a business question" in messages[0].content:
            self.planning_messages.append(list(messages))
            if len(self.planning_messages) == 1:
                raise TimeoutError("simulated provider timeout")
            return Turn(text="SELECT 1 AS n", stop_reason="stop")
        return Turn(text="指标为 1")


async def test_real_scope_isolates_client_prompts_user_examples_and_external_memory(case_conn, monkeypatch):
    class ExternalRedisSpy:
        def __init__(self):
            self.calls = []

        def append_dialogue_turn(self, *args):
            self.calls.append(("append", args))

        def update_user_profile(self, *args):
            self.calls.append(("profile", args))

        def delete_user_profile(self, *args):
            self.calls.append(("delete", args))

        def get_user_profile(self, *args):
            self.calls.append(("get", args))
            return {"city": "REAL_USER_FILTER"}

    external = ExternalRedisSpy()
    parent_memory = MemoryStore(redis_store=external)
    parent_memory.remember_turn("eval_session", "user", "REAL_SESSION_HISTORY")
    parent_memory.set_filter("eval_user", "city", "REAL_USER_FILTER")
    parent_memory.remember("eval_user", CASES[0]["question"], "SELECT 'REAL_USER_EXAMPLE'")
    parent_memory.remember_turn("real-session", "assistant", "real response")
    original_calls = list(external.calls)
    original_history = parent_memory.session("eval_session")
    client = ClientSpy()
    config = RunConfig(use_rag=False, use_routing=False, use_memory=True)
    orchestrator = InsightOrchestrator(
        case_conn, memory=parent_memory, config=config,
        planner=LLMPlanner(client), retriever=object(),
    )
    orchestrator.cache.put("real-user", "SELECT 1", {"real": True})
    original_cache = dict(orchestrator.cache._store)
    memories = []

    def local_memory(**kwargs):
        memory = MemorySpy(**kwargs)
        memories.append(memory)
        return memory

    from insight_agent.app import redis_store
    redis_initializations = []
    monkeypatch.setattr(redis_store, "RedisMemoryStore", lambda: redis_initializations.append(True))
    monkeypatch.setattr(evaluation, "MemoryStore", local_memory)
    first = await evaluation.run_one_case(orchestrator, case_conn, CASES[0], "test", live=True)
    second = await evaluation.run_one_case(orchestrator, case_conn, CASES[1], "test", live=True)
    retry = await evaluation.run_one_case(orchestrator, case_conn, CASES[0], "test", previous=first, live=True)

    assert first.llm_error and not first.passed
    assert second.passed and retry.passed
    assert len(memories) == len(client.planning_messages) == 3
    assert len({id(memory) for memory in memories}) == 3
    assert len({memory.history_reads[0][0] for memory in memories}) == 3
    assert len({memory.filter_reads[0][0] for memory in memories}) == 3
    for memory in memories:
        assert memory.local_only and memory.redis_store is None and memory.path is None
        assert memory.history_reads[0][1] == []
        assert memory.filter_reads[0][1] == {}
        assert memory.episode_reads[0][1] == []
    # A successful attempt really does record history and episodes; isolation
    # is not achieved by silently changing use_memory=False.
    assert memories[-1].session(memories[-1].history_reads[0][0])
    assert memories[-1].profile(memories[-1].filter_reads[0][0]).episodes
    for messages, case in zip(client.planning_messages, [CASES[0], CASES[1], CASES[0]], strict=True):
        assert [m.role for m in messages] == ["system", "user"]
        assert messages[-1].content == case["question"]
        prompt = "\n".join(m.content for m in messages)
        for marker in ("REAL_SESSION_HISTORY", "REAL_USER_EXAMPLE", "REAL_USER_FILTER", "Recalled Past Exemplars", "GOLD_ONLY_EXPECTATION"):
            assert marker not in prompt
    assert external.calls == original_calls
    assert redis_initializations == []
    assert parent_memory.session("eval_session") == original_history
    assert parent_memory.session("real-session") == ["assistant", "real response"]
    assert orchestrator.memory is parent_memory
    assert orchestrator.config is config and config.use_memory
    assert orchestrator.cache._store == original_cache
    assert orchestrator.tracer.total_cost == 0
    assert retry.attempt_history[0]["planning_diagnostics"]["llm_error"] is True


async def test_failure_diagnostics_are_bounded_redacted_and_serializable(case_conn, monkeypatch, capsys):
    secret = "synthetic-test-credential-314159"
    monkeypatch.setenv("INSIGHT_API_KEY", secret)
    raw = f"API_KEY={secret}; Bearer bearer-test-credential; https://user:password@host.invalid/\n" + "x" * 7000
    response = answer(sql="", diagnostics={
        "stage": "repair", "parse_reason": "truncated", "stop_reason": "length",
        "model_text": raw, "raw_planner_text": raw,
        "notes": [f"provider returned {secret}", "no complete SQL"],
        "planner_notes": ["no complete SQL"], "repair_attempts": 2,
        "final_error": f"api_key={secret}", "clarification_needed": True,
        "tokens_in": 17, "tokens_out": 23, "llm_error": False,
        "headers": {"Authorization": "Basic c3ludGhldGlj", "api_key": secret},
    })
    orch = ScopeSpy(case_conn, replies=[response])
    outcome = await evaluation.run_one_case(orch, case_conn, CASES[0], "test")
    path = evaluation.write_strategy_cache("test", [outcome], live=True)
    persisted = path.read_text(encoding="utf-8")
    logged = capsys.readouterr().out
    for sensitive in (secret, "bearer-test-credential", "user:password", "c3ludGhldGlj"):
        assert sensitive not in persisted + logged
    assert "[REDACTED]" in persisted
    assert len(outcome.planning_diagnostics["model_text"]) <= evaluation.MAX_DIAGNOSTIC_TEXT_CHARS
    assert outcome.planning_diagnostics["stop_reason"] == "length"
    assert outcome.planning_diagnostics["repair_attempts"] == 2
    assert outcome.planning_diagnostics["tokens_in"] == 17
    assert outcome.planning_diagnostics["notes"][1] == "no complete SQL"
    assert outcome.execution_diagnostics["failure_stage"] == "repair"
    assert outcome.execution_diagnostics["clarification_needed"] is True
    assert outcome.execution_diagnostics["blocked"] is False
    assert outcome.execution_diagnostics["model"] == "spy-model"
    assert outcome.execution_diagnostics["planner"] == "llm"
    assert outcome.execution_diagnostics["route_score"] == 7
    assert not outcome.llm_error  # no SQL/truncation is not a provider exception
    assert evaluation.load_strategy_cache("test")[0].planning_diagnostics == outcome.planning_diagnostics


async def test_raised_error_and_retry_history_retain_actual_stage(case_conn, monkeypatch, capsys):
    monkeypatch.setenv("INSIGHT_API_KEY", "synthetic-secret-271828")
    failure = TimeoutError("provider key=synthetic-secret-271828")
    failure.planning_diagnostics = {"stop_reason": "", "stage": "plan", "notes": ["request failed"]}
    orch = ScopeSpy(case_conn, replies=[failure, answer()])
    first = await evaluation.run_one_case(orch, case_conn, CASES[0], "test")
    retry = await evaluation.run_one_case(orch, case_conn, CASES[0], "test", previous=first)
    assert_fresh_attempts(orch, 2)
    assert first.execution_diagnostics["failure_stage"] == "plan"
    assert first.execution_diagnostics["exception_type"] == "TimeoutError"
    assert "[REDACTED]" in first.execution_diagnostics["final_error"]
    assert retry.retry_count == 1 and len(retry.attempt_history) == 1
    assert retry.attempt_history[0]["planning_diagnostics"]["notes"] == ["request failed"]
    assert retry.attempt_history[0]["attempt_id"] != retry.attempt_id
    assert "synthetic-secret-271828" not in capsys.readouterr().out


@pytest.mark.parametrize("blocked,expectation", [(True, "blocked"), (False, "clarify")])
async def test_expected_block_or_clarification_does_not_become_transport_failure(case_conn, blocked, expectation):
    response = answer(sql="", blocked=blocked, diagnostics={"clarification_needed": not blocked})
    case = {**CASES[0], "expectation": expectation}
    result = await evaluation.execute_case(ScopeSpy(case_conn, replies=[response]), case_conn, case)
    assert result.passed and not result.llm_error
    assert result.execution_diagnostics["blocked"] is blocked
    assert result.execution_diagnostics["clarification_needed"] is not blocked


async def test_missing_diagnostics_and_sql_grading_failure_remain_readable(case_conn):
    response = SimpleNamespace(
        sql="SELECT n FROM missing_table", blocked=False, evidence=[], cost_usd=0,
        agents_used=[], answer="query failed",
    )
    result = await evaluation.execute_case(ScopeSpy(case_conn, replies=[response]), case_conn, CASES[0])
    assert result.planning_diagnostics == {}
    assert result.grade_error.startswith("OperationalError:")
    assert result.execution_diagnostics["grading_failure_stage"] == "grading_agent_sql"
    assert result.execution_diagnostics["failure_stage"] == "grading_agent_sql"
    assert not result.llm_error


def legacy_record(case, *, failed=False):
    return {
        "case_id": case["id"], "shape": "flat", "strategy": "cheap_hybrid",
        "passed": False, "recall": 1.0, "latency_ms": 1.0, "cost_usd": 0.0,
        "llm_error": failed, "gold_rows_json": '[{"n": 1}]', "agent_rows_json": '[{"n": 1}]',
    }


def write_legacy(records):
    path = evaluation.STRATEGY_CACHE_DIR / "cheap_hybrid.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"strategy": "cheap_hybrid", "mode": "live_llm", "outcomes": records}), encoding="utf-8")
    return path


def test_legacy_cache_regrade_and_evolution_load_do_not_forge_isolated_provenance(tmp_path):
    records = [legacy_record(case) for case in CASES]
    legacy = write_legacy(records)
    original = legacy.read_bytes()
    loaded = evaluation.load_strategy_cache("cheap_hybrid")
    assert all(o.planning_diagnostics == {} and o.attempt_history == [] for o in loaded)
    assert all(o.memory_scope_version == "" for o in loaded)
    assert not evaluation._cache_is_clean("cheap_hybrid")
    regraded = evaluation.regrade_strategy_caches(["cheap_hybrid"])["cheap_hybrid"]
    assert all(o.passed for o in regraded)
    assert legacy.read_bytes() == original
    current = evaluation.strategy_cache_path("cheap_hybrid")
    assert current != legacy
    assert json.loads(current.read_text(encoding="utf-8"))["memory_scope_version"] == "legacy_or_mixed"
    assert not evaluation._cache_is_clean("cheap_hybrid")
    state = tmp_path / "evolution.json"
    state.write_text(json.dumps({"outcomes": records}), encoding="utf-8")
    engine = EvolutionEngine(path=state)
    assert engine.outcomes[0].retry_count == 0
    engine.save()
    assert EvolutionEngine(path=state).outcomes == engine.outcomes


def test_clean_cache_requires_version_case_set_config_and_mode():
    strategy = default_strategies()[1]
    fingerprint = evaluation.evaluation_fingerprint(strategy.name, strategy.config, live=True)
    outcomes = [
        Outcome(
            **legacy_record(case), attempt_id=case["id"],
            memory_scope_version=evaluation.CASE_MEMORY_SCOPE_VERSION,
            evaluation_fingerprint=fingerprint,
        )
        for case in CASES
    ]
    path = evaluation.write_strategy_cache(strategy.name, outcomes, live=True)
    assert evaluation._cache_is_clean(strategy.name, live=True)
    assert not evaluation._cache_is_clean(strategy.name, live=False)
    assert not evaluation._cache_is_clean(strategy.name, config=replace(strategy.config, use_memory=False))
    # Header alone is insufficient, including after a partial legacy repair.
    outcomes[0].memory_scope_version = ""
    evaluation.write_strategy_cache(strategy.name, outcomes, live=True)
    assert not evaluation._cache_is_clean(strategy.name)
    outcomes[0].memory_scope_version = evaluation.CASE_MEMORY_SCOPE_VERSION
    outcomes[1].case_id = outcomes[0].case_id
    evaluation.write_strategy_cache(strategy.name, outcomes, live=True)
    assert not evaluation._cache_is_clean(strategy.name)
    path.write_text("{invalid", encoding="utf-8")
    assert not evaluation._cache_is_clean(strategy.name)


def test_offline_report_rebuild_labels_mixed_provenance_without_writing_reports(monkeypatch):
    from insight_agent.eval import rebuild_live_report

    cases = [{**case, "tier": 1} for case in CASES]
    monkeypatch.setattr(evaluation, "CN_RETAIL_200_CASES", cases)
    strategies = default_strategies()
    legacy_paths = []
    for strategy in strategies:
        records = [{**legacy_record(case), "strategy": strategy.name} for case in cases]
        path = evaluation.STRATEGY_CACHE_DIR / f"{strategy.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mode": "live_llm", "outcomes": records}), encoding="utf-8")
        legacy_paths.append((path, path.read_bytes()))
    captured = []

    def aggregate_without_report(outcomes, selected, *, live, out_file):
        report = evaluation.aggregate_and_report(outcomes, selected, live=False)
        captured.append(report)
        return report

    monkeypatch.setattr(rebuild_live_report, "aggregate_and_report", aggregate_without_report)
    assert rebuild_live_report.main() == 0
    assert captured[0]["benchmark_meta"]["memory_scope_versions_by_strategy"] == {
        strategy.name: ["legacy"] for strategy in strategies
    }
    assert all(path.read_bytes() == original for path, original in legacy_paths)


@pytest.mark.parametrize("live", [False, True])
def test_regrade_preserves_execution_mode_and_provenance(live):
    strategy = default_strategies()[1]
    fingerprint = evaluation.evaluation_fingerprint(strategy.name, strategy.config, live=live)
    outcomes = [
        Outcome(
            **legacy_record(case), attempt_id=case["id"],
            memory_scope_version=evaluation.CASE_MEMORY_SCOPE_VERSION,
            evaluation_fingerprint=fingerprint,
        )
        for case in CASES
    ]
    evaluation.write_strategy_cache(strategy.name, outcomes, live=live)
    evaluation.regrade_strategy_caches([strategy.name])
    assert evaluation._cache_is_clean(strategy.name, live=live)
    assert not evaluation._cache_is_clean(strategy.name, live=not live)


async def test_degraded_rerun_uses_fresh_scopes_and_preserves_old_cache(case_conn, monkeypatch):
    # Import under monkeypatch so rerun's live-mode import flag cannot leak to
    # other tests. No CLI entry point, health probe or real client is invoked.
    monkeypatch.setenv("INSIGHT_EVAL_LIVE", "0")
    from insight_agent.eval import rerun_degraded

    monkeypatch.setattr(rerun_degraded, "CN_RETAIL_200_CASES", CASES)
    records = [legacy_record(CASES[0], failed=True), legacy_record(CASES[1])]
    legacy = write_legacy(records)
    original = legacy.read_bytes()
    created = []

    def factory(conn, **kwargs):
        orch = ScopeSpy(conn, **kwargs, replies=[TimeoutError("retry again"), answer()])
        created.append(orch)
        return orch

    monkeypatch.setattr(rerun_degraded, "InsightOrchestrator", factory)
    strategy = default_strategies()[1]
    stats = await rerun_degraded.rerun_strategy(
        strategy, case_conn, object(), SimpleNamespace(name="llm"), max_rounds=2, round_pause=0,
    )
    assert stats == {"degraded": 1, "resolved": 1, "unresolved": 0}
    assert_fresh_attempts(created[0], 2)
    assert all(r.question == CASES[0]["question"] for r in created[0].requests)
    first, untouched = evaluation.load_strategy_cache(strategy.name)
    assert first.passed and first.retry_count == 2 and len(first.attempt_history) == 2
    assert untouched.memory_scope_version == "" and untouched.retry_count == 0
    assert legacy.read_bytes() == original
    assert not evaluation._cache_is_clean(strategy.name)


async def test_concurrent_attempts_and_bounded_prior_history(case_conn):
    orch = ScopeSpy(case_conn)
    first, second = await asyncio.gather(*(
        evaluation.run_one_case(orch, case_conn, case, "test") for case in CASES
    ))
    first.retry_count = 40
    first.attempt_history = [{"attempt_id": f"old-{i}"} for i in range(40)]
    latest = await evaluation.run_one_case(orch, case_conn, CASES[0], "test", previous=first)
    assert_fresh_attempts(orch, 3)
    assert latest.retry_count == 41
    assert len(latest.attempt_history) == evaluation.MAX_ATTEMPT_HISTORY
    assert latest.attempt_history[-1]["attempt_id"] == first.attempt_id
    assert second.attempt_history == []
