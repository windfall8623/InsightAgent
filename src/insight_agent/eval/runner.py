"""Evaluation runner: executes the dataset against a configuration and reports.

Run configurations differ only in :class:`~insight_agent.app.orchestrator.RunConfig`
flags, which is what makes the ablation study meaningful — each variant differs
from the baseline in exactly one capability.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from insight_agent.app.memory import MemoryStore
from insight_agent.app.orchestrator import (
    InsightOrchestrator,
    InsightRequest,
    RunConfig,
)
from insight_agent.app.planner import Planner, RulesPlanner
from insight_agent.app.rag import build_retriever
from insight_agent.data.database import connect, execute
from insight_agent.eval.dataset import load_cases
from insight_agent.eval.metrics import (
    CaseResult,
    Report,
    aggregate,
    faithfulness,
    retrieval_recall,
    row_overlap,
    rows_match,
)

# Identifiers that legitimately exist in the schema, for the hallucination metric.
_SQL_KEYWORDS = {
    "select",
    "from",
    "where",
    "group",
    "by",
    "order",
    "limit",
    "join",
    "left",
    "inner",
    "outer",
    "on",
    "as",
    "and",
    "or",
    "not",
    "null",
    "nullif",
    "count",
    "sum",
    "avg",
    "round",
    "distinct",
    "case",
    "when",
    "then",
    "else",
    "end",
    "strftime",
    "desc",
    "asc",
    "o",
    "oi",
    "p",
    "pr",
    "c",
    "u",
    "r",
    "id",
    "dimension",
    "metric",
    "over",
    "in",
    "is",
    "like",
    "between",
    "having",
}


@dataclass(slots=True)
class Variant:
    """A named configuration to evaluate."""

    name: str
    config: RunConfig
    description: str = ""


def default_variants() -> list[Variant]:
    """The ablation grid: full system plus one variant per disabled capability."""

    return [
        Variant("full", RunConfig(), "all capabilities enabled"),
        Variant(
            "no_rag",
            RunConfig(use_rag=False),
            "schema retrieval disabled; evidence is empty",
        ),
        Variant(
            "keyword_only",
            RunConfig(retrieval_mode="keyword"),
            "BM25 leg only; vector leg removed",
        ),
        Variant(
            "vector_only",
            RunConfig(retrieval_mode="vector"),
            "vector leg only; BM25 removed",
        ),
        Variant(
            "no_reflection",
            RunConfig(use_reflection=False),
            "first result accepted without validation",
        ),
        Variant(
            "no_routing",
            RunConfig(use_routing=False),
            "every question routed to the large model",
        ),
        Variant(
            "no_memory",
            RunConfig(use_memory=False),
            "memory, episode recall, and caching disabled",
        ),
    ]


def _schema_identifiers(conn: Any) -> tuple[set[str], set[str]]:
    tables: set[str] = set()
    columns: set[str] = set()
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ):
        tables.add(name)
        for col in conn.execute(f'PRAGMA table_info("{name}")'):
            columns.add(col[1])
    return tables, columns


async def run_variant(
    variant: Variant,
    *,
    conn: Any = None,
    planner: Planner | None = None,
    cases: list[dict[str, Any]] | None = None,
) -> Report:
    """Evaluate every case under one configuration.

    ``planner`` defaults to the deterministic engine. Pass the configured LLM
    planner to measure a live backend, and ``cases`` to evaluate a specific
    split.
    """

    conn = conn or connect()
    retriever = build_retriever(conn)
    planner = planner or RulesPlanner()
    items = cases if cases is not None else load_cases()
    tables, columns = _schema_identifiers(conn)
    results: list[CaseResult] = []

    for case in items:
        # A fresh orchestrator per case keeps memory and cache from leaking
        # between evaluation items, which would inflate the numbers.
        orchestrator = InsightOrchestrator(
            conn,
            memory=MemoryStore(),
            config=variant.config,
            retriever=retriever,
            planner=planner,
        )
        results.append(await _run_case_async(case, orchestrator, conn, tables, columns))

    return aggregate(variant.name, results)


async def _run_case_async(
    case: dict[str, Any],
    orchestrator: InsightOrchestrator,
    conn: Any,
    tables: set[str],
    columns: set[str],
) -> CaseResult:
    expectation = case["expectation"]
    started = time.perf_counter()
    answer = await orchestrator.run(InsightRequest(question=case["question"], user_id="eval"))
    latency = (time.perf_counter() - started) * 1000

    base = CaseResult(
        case_id=case["id"],
        question=case["question"],
        category=case["category"],
        difficulty=case["difficulty"],
        expectation=expectation,
        passed=False,
        blocked=answer.blocked,
        row_count=answer.row_count,
        latency_ms=latency,
        cost_usd=answer.cost_usd,
        tools_used=len(answer.tools_used),
        agents_used=len(answer.agents_used),
        reflections=answer.reflections,
        model=answer.model,
        got_sql=answer.sql,
        want_sql=case.get("gold_sql") or "",
        retrieval_recall=retrieval_recall(answer.evidence, case.get("requires_tables") or []),
        faithfulness=faithfulness(answer.sql, tables, columns) if answer.sql else 1.0,
    )

    if expectation == "blocked":
        base.passed = answer.blocked
        if not base.passed:
            base.error = f"not blocked (answered: {answer.answer[:80]})"
        return base

    if expectation == "clarify":
        clarified = not answer.sql and "could not" in answer.answer.lower()
        base.passed = clarified
        if not clarified:
            base.error = f"did not ask for clarification (sql={answer.sql[:60]})"
        return base

    if answer.blocked:
        base.error = f"blocked unexpectedly: {answer.block_reason}"
        return base

    gold_sql = case.get("gold_sql")
    if not gold_sql:
        base.error = "case has no gold_sql"
        return base

    try:
        expected_rows = execute(conn, gold_sql, limit=500).as_dicts()
    except Exception as exc:  # noqa: BLE001 - a bad gold query is a dataset bug
        base.error = f"gold_sql failed: {type(exc).__name__}: {exc}"
        return base

    base.passed = rows_match(expected_rows, answer.rows)
    base.overlap = row_overlap(expected_rows, answer.rows)
    if not base.passed:
        base.error = f"result mismatch (overlap={base.overlap:.2f})"
    return base


async def run_all(
    variants: list[Variant] | None = None,
    *,
    out_dir: Path | None = None,
    planner: Planner | None = None,
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the full ablation grid and write the report to disk."""

    conn = connect()
    variants = variants or default_variants()
    reports: dict[str, Report] = {}
    for variant in variants:
        reports[variant.name] = await run_variant(variant, conn=conn, planner=planner, cases=cases)

    payload = {name: report.to_payload() for name, report in reports.items()}
    planner_name = getattr(planner, "name", "rules")
    payload["_meta"] = {
        "cases": len(cases if cases is not None else load_cases()),
        "variants": {v.name: v.description for v in variants},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "planner": planner_name,
        "backend": (
            "deterministic rule engine (no LLM API key configured)"
            if planner_name == "rules"
            else f"live LLM backend ({planner_name})"
        ),
    }

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    return payload
