"""Run Full 200-Case CN-Retail 2.0 Evaluation & Self-Evolution Experiment.

This script executes a comprehensive benchmark on 200 ground-truth cases:
1. Validates all 200 cases across 4 tiers on cn_retail.db (88,002 rows, 8 tables).
2. Runs 4 candidate strategies across all 200 cases (800 total executions):
   - cheap_flat: No RAG, Fast Router (Cheapest)
   - cheap_hybrid: Hybrid RAG, Fast Router (Standard)
   - rich_vector: Dense Vector RAG, Fast Router
   - deep_hybrid: Hybrid RAG, Deep Frontier Model
3. Evaluates per-item outcomes, latency, token costs, and schema retrieval recall.
4. Feeds outcomes to EvolutionEngine to classify question shapes and propose an optimal policy.
5. Evaluates candidate vs baseline through the SAGE statistical acceptance gate (McNemar test).
6. Exports full results to docs/cn_retail_200_benchmark_report.json.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Live mode (--live) drives the deep-path through the configured LLM and keeps
# the real Jev router endpoint. The default rules-only run blanks Jev so routing
# stays deterministic and the authoritative report is reproducible offline.
# INSIGHT_EVAL_LIVE=1 lets importers (e.g. rerun_degraded) force live mode
# without injecting --live into argparse.
_LIVE = ("--live" in sys.argv) or os.environ.get("INSIGHT_EVAL_LIVE") == "1"
if not _LIVE:
    os.environ["INSIGHT_JEV_URL"] = ""

from insight_agent.app.evolution import (
    EvolutionEngine,
    Outcome,
    PolicyMetrics,
    Strategy,
    classify_shape,
    default_strategies,
    uniform_policy,
)
from insight_agent.app.memory import MemoryStore
from insight_agent.app.orchestrator import InsightOrchestrator, InsightRequest, RunConfig
from insight_agent.app.planner import Planner, build_planner
from insight_agent.app.rag import HybridRetriever, build_retriever
from insight_agent.app.qdrant_store import get_default_bge_adapter
from insight_agent.config import load_config
from insight_agent.data.cn_retail_db import connect_cn_retail, ensure_cn_retail_database
from insight_agent.eval.cn_retail_200_cases import CN_RETAIL_200_CASES
from insight_agent.eval.metrics import retrieval_recall, row_overlap, rows_match, rows_match_v2
from insight_agent.eval.sage_gate import GateConfig, analyse_dynamics, evaluate_gate

# Legacy caches stay readable for offline regrading/rebuilding. New executions
# and repaired/regraded copies go into a versioned directory, never over them.
STRATEGY_CACHE_DIR = Path("docs/_live_strategy_cache")
LIVE_REPORT_PATH = Path("docs/cn_retail_200_live_benchmark_report.json")
RULES_REPORT_PATH = Path("docs/cn_retail_200_benchmark_report.json")
CASE_MEMORY_SCOPE_VERSION = "isolated-case-v1"
STRATEGY_CACHE_VERSION = 2
MAX_DIAGNOSTIC_TEXT_CHARS = 4096
MAX_ATTEMPT_HISTORY = 25

_CREDENTIAL_KEY = re.compile(
    r"api[_-]?key|authorization|cookie|password|passwd|"
    r"(?:^|[_-])(?:secret|token|access[_-]?token|refresh[_-]?token|credentials?)(?:$|[_-])",
    re.IGNORECASE,
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r'''(?ix)(["']?(?:[\w-]*(?:api[_-]?key|password|passwd|secret|credential)[\w-]*
    |(?:[\w-]+[_-])?(?:access[_-]?token|refresh[_-]?token|auth[_-]?token|token)
    |authorization|proxy-authorization|cookie|set-cookie)["']?\s*[:=]\s*)
    (?:\[REDACTED\]|"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;&}\]]+)'''
)


def _diagnostic_text(value: str, *, limit: int = MAX_DIAGNOSTIC_TEXT_CHARS) -> str:
    """Redact before truncating, including known credentials already in env.

    This never loads .env or serialises client/config objects. Redacting first
    avoids leaving the beginning of a credential at the truncation boundary.
    """
    text = value
    for key, secret in os.environ.items():
        if _CREDENTIAL_KEY.search(key) and len(secret) >= 4:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(
        r"(?i)(\b[a-z][a-z0-9+.-]*://)[^/\s@]+@", r"\1[REDACTED]@", text
    )
    text = re.sub(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", r"\1 [REDACTED]", text)
    text = _CREDENTIAL_ASSIGNMENT.sub(r"\1[REDACTED]", text)
    text = re.sub(
        r"\b(?:sk-[A-Za-z0-9_-]{8,}|AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{16,})\b",
        "[REDACTED]",
        text,
    )
    text = re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b", "[REDACTED]", text)
    suffix = "...[truncated]"
    return text if len(text) <= limit else text[: max(0, limit - len(suffix))] + suffix[:limit]


def _diagnostics(value: Any) -> dict[str, Any]:
    """Copy JSON-shaped telemetry with bounded text, depth and total nodes.

    Unknown objects are not repr'd: they may be SDK clients carrying secrets.
    """
    nodes_left = 256
    chars_left = 32768

    def scrub(item: Any, depth: int = 0) -> Any:
        nonlocal nodes_left, chars_left
        nodes_left -= 1
        if depth > 6 or nodes_left < 0:
            return "[truncated]"
        if isinstance(item, str):
            rendered = _diagnostic_text(item, limit=min(MAX_DIAGNOSTIC_TEXT_CHARS, chars_left))
            chars_left = max(0, chars_left - len(rendered))
            return rendered
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for key, child in list(item.items())[:32]:
                if nodes_left <= 0:
                    result["_truncated"] = True
                    break
                if not isinstance(key, str):
                    continue
                safe_key = _diagnostic_text(key, limit=128)
                result[safe_key] = "[REDACTED]" if _CREDENTIAL_KEY.search(key) else scrub(child, depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            return [scrub(child, depth + 1) for child in item[:32] if nodes_left > 0]
        return f"<{type(item).__name__}>"

    return scrub(value) if isinstance(value, dict) else {}


def evaluation_fingerprint(strat_name: str, config: RunConfig, *, live: bool) -> str:
    """Cache identity only; none of this benchmark data is sent to the planner."""
    payload = {
        "cache_version": STRATEGY_CACHE_VERSION,
        "memory_scope_version": CASE_MEMORY_SCOPE_VERSION,
        "grader": "rows_match_v2",
        "mode": "live_llm" if live else "rules_only",
        "strategy": strat_name,
        "config": asdict(config),
        "cases": CN_RETAIL_200_CASES,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def strategy_cache_path(strat_name: str, *, for_write: bool = False) -> Path:
    current = STRATEGY_CACHE_DIR / CASE_MEMORY_SCOPE_VERSION / f"{strat_name}.json"
    if for_write or current.exists():
        return current
    return STRATEGY_CACHE_DIR / f"{strat_name}.json"


def write_strategy_cache(strat_name: str, outcomes: list[Outcome], *, live: bool) -> Path:
    """Persist without upgrading legacy/mixed outcomes to isolated provenance."""
    path = strategy_cache_path(strat_name, for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    isolated = bool(outcomes) and all(
        o.memory_scope_version == CASE_MEMORY_SCOPE_VERSION and o.attempt_id
        for o in outcomes
    )
    fingerprints = {o.evaluation_fingerprint for o in outcomes}
    fingerprint = next(iter(fingerprints)) if len(fingerprints) == 1 else ""
    records = []
    for outcome in outcomes:
        record = asdict(outcome)
        record["planning_diagnostics"] = _diagnostics(outcome.planning_diagnostics)
        record["execution_diagnostics"] = _diagnostics(outcome.execution_diagnostics)
        record["attempt_history"] = [_diagnostics(item) for item in outcome.attempt_history[-MAX_ATTEMPT_HISTORY:]]
        records.append(record)
    path.write_text(
        json.dumps(
            {
                "cache_version": STRATEGY_CACHE_VERSION,
                "memory_scope_version": CASE_MEMORY_SCOPE_VERSION if isolated else "legacy_or_mixed",
                "evaluation_fingerprint": fingerprint,
                "strategy": strat_name,
                "mode": "live_llm" if live else "rules_only",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "degraded_case_ids": [
                    o.case_id for o in outcomes
                    if o.rerank_degraded or o.llm_error or o.embedding_degraded
                ],
                "rerank_degraded_case_ids": [o.case_id for o in outcomes if o.rerank_degraded],
                "embedding_degraded_case_ids": [o.case_id for o in outcomes if o.embedding_degraded],
                "llm_error_case_ids": [o.case_id for o in outcomes if o.llm_error],
                "outcomes": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_strategy_cache(strat_name: str) -> list[Outcome]:
    path = strategy_cache_path(strat_name)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Outcome(**o) for o in data["outcomes"]]


def regrade_strategy_caches(strat_names: list[str], *, write: bool = True) -> dict[str, list[Outcome]]:
    """Re-score cached outcomes with the current grader — no model calls.

    Because every ``rows`` case cached its agent SQL and both result sets, a
    grader change can be applied offline: this recomputes ``passed`` (v2),
    ``passed_legacy`` and ``overlap`` from the stored rows and (optionally)
    rewrites a versioned copy of the caches. Legacy files are never overwritten.
    blocked/clarify cases and cases where the agent produced no executable SQL
    are left exactly as they were.
    """

    case_expect = {c["id"]: c.get("expectation", "rows") for c in CN_RETAIL_200_CASES}
    regraded: dict[str, list[Outcome]] = {}
    for name in strat_names:
        outcomes = load_strategy_cache(name)
        flips = 0
        for o in outcomes:
            if case_expect.get(o.case_id, "rows") != "rows":
                continue
            if not (o.gold_rows_json and o.agent_rows_json):
                continue
            gold = json.loads(o.gold_rows_json)
            agent = json.loads(o.agent_rows_json)
            new_passed = rows_match_v2(gold, agent)
            if new_passed != o.passed:
                flips += 1
            o.passed = new_passed
            o.passed_legacy = rows_match(gold, agent)
            o.overlap = row_overlap(gold, agent)
        regraded[name] = outcomes
        if write:
            source = json.loads(strategy_cache_path(name).read_text(encoding="utf-8"))
            write_strategy_cache(name, outcomes, live=source.get("mode") != "rules_only")
        print(f"[Regrade] {name:<14} {flips} verdict change(s) vs cached")
    return regraded


def metrics_from_outcomes(outcomes: list[Outcome]) -> PolicyMetrics:
    n = len(outcomes)
    if n == 0:
        return PolicyMetrics()
    return PolicyMetrics(
        success=sum(1 for o in outcomes if o.passed) / n,
        recall=sum(o.recall for o in outcomes) / n,
        cost=sum(o.cost_usd for o in outcomes),
        latency=sum(o.latency_ms for o in outcomes) / n,
        score=sum(o.score for o in outcomes) / n,
        cases=n,
    )


def aggregate_and_report(
    strategy_outcomes: dict[str, list[Outcome]],
    strategies: list[Strategy],
    *,
    live: bool,
    out_file: Path | None = None,
) -> dict[str, Any]:
    """Rebuild policy + SAGE gate + report payload from per-strategy outcomes.

    Shared by the full sweep and the rebuild/rerun paths so both produce an
    identical report structure from the same inputs.
    """
    engine = EvolutionEngine(strategies)
    strategy_per_item: dict[str, dict[str, bool]] = {}
    strategy_metrics: dict[str, PolicyMetrics] = {}
    for name, outcomes in strategy_outcomes.items():
        strategy_per_item[name] = {o.case_id: o.passed for o in outcomes}
        strategy_metrics[name] = metrics_from_outcomes(outcomes)
        for o in outcomes:
            engine.observe(o)

    proposed_policy = engine.propose()
    unique_shapes = sorted({o.shape for o in engine.outcomes})

    candidate_outcomes: list[Outcome] = []
    candidate_per_item: dict[str, bool] = {}
    for case in CN_RETAIL_200_CASES:
        shape = classify_shape(case["question"]).key
        chosen_strat_name = proposed_policy.get(shape, engine.baseline_name)
        chosen_outcome = next(o for o in strategy_outcomes[chosen_strat_name] if o.case_id == case["id"])
        candidate_outcomes.append(chosen_outcome)
        candidate_per_item[case["id"]] = chosen_outcome.passed

    n_c = len(candidate_outcomes)
    candidate_metrics = PolicyMetrics(
        success=sum(1 for o in candidate_outcomes if o.passed) / n_c,
        recall=sum(o.recall for o in candidate_outcomes) / n_c,
        cost=sum(o.cost_usd for o in candidate_outcomes),
        latency=sum(o.latency_ms for o in candidate_outcomes) / n_c,
        score=sum(o.score for o in candidate_outcomes) / n_c,
        cases=n_c,
    )

    baseline_name = engine.baseline_name
    baseline_metrics = strategy_metrics[baseline_name]
    baseline_per_item = strategy_per_item[baseline_name]

    gate_verdict = evaluate_gate(baseline_per_item, candidate_per_item, engine.gate_config)

    accepted = False
    rejected_reasons: list[str] = []
    if gate_verdict.decision.value != "accept":
        rejected_reasons.append(gate_verdict.reason)
    elif candidate_metrics.recall < baseline_metrics.recall - 1e-9:
        rejected_reasons.append(f"Recall regressed from {baseline_metrics.recall:.4f} to {candidate_metrics.recall:.4f}")
    else:
        accepted = True

    dynamics = analyse_dynamics([baseline_metrics.score, candidate_metrics.score])
    improvement_pct = (
        (candidate_metrics.score - baseline_metrics.score) / abs(baseline_metrics.score) * 100
        if baseline_metrics.score else 0.0
    )

    shape_distribution: dict[str, int] = {}
    for strat_name in proposed_policy.values():
        shape_distribution[strat_name] = shape_distribution.get(strat_name, 0) + 1

    tier_stats: dict[str, dict[str, Any]] = {}
    for t in [1, 2, 3, 4]:
        tier_cases = [c for c in CN_RETAIL_200_CASES if c["tier"] == t]
        tier_cids = {c["id"] for c in tier_cases}
        tier_outcomes = [o for o in candidate_outcomes if o.case_id in tier_cids]
        tier_stats[f"Tier_{t}"] = {
            "total_queries": len(tier_cases),
            "pass_rate": round(sum(1 for o in tier_outcomes if o.passed) / max(len(tier_outcomes), 1), 4),
            "avg_recall": round(sum(o.recall for o in tier_outcomes) / max(len(tier_outcomes), 1), 4),
            "avg_latency_ms": round(sum(o.latency_ms for o in tier_outcomes) / max(len(tier_outcomes), 1), 2),
        }

    rerank_degraded_by_strategy = {
        name: sum(1 for o in outs if o.rerank_degraded) for name, outs in strategy_outcomes.items()
    }
    embedding_degraded_by_strategy = {
        name: sum(1 for o in outs if o.embedding_degraded) for name, outs in strategy_outcomes.items()
    }
    llm_error_by_strategy = {
        name: sum(1 for o in outs if o.llm_error) for name, outs in strategy_outcomes.items()
    }
    degraded_ids_by_strategy = {
        name: [
            o.case_id for o in outs
            if o.rerank_degraded or o.embedding_degraded or o.llm_error
        ]
        for name, outs in strategy_outcomes.items()
    }

    # Grading delta: the authoritative success rate uses the v2 grader (passed);
    # passed_legacy keeps the old exact-multiset verdict. Report both plus the
    # cases the new grader flipped so the improvement is auditable, not asserted.
    grading_by_strategy: dict[str, dict[str, Any]] = {}
    for name, outs in strategy_outcomes.items():
        row_outs = [
            o for o in outs
            if next((c for c in CN_RETAIL_200_CASES if c["id"] == o.case_id), {}).get(
                "expectation", "rows"
            ) == "rows"
        ]
        recovered = sorted(o.case_id for o in row_outs if o.passed and not o.passed_legacy)
        regressed = sorted(o.case_id for o in row_outs if o.passed_legacy and not o.passed)
        n_row = max(len(row_outs), 1)
        grading_by_strategy[name] = {
            "row_cases": len(row_outs),
            "legacy_pass_rate": round(sum(1 for o in row_outs if o.passed_legacy) / n_row, 4),
            "v2_pass_rate": round(sum(1 for o in row_outs if o.passed) / n_row, 4),
            "recovered_by_v2": recovered,
            "regressed_under_v2": regressed,
        }

    report_payload = {
        "benchmark_meta": {
            "name": "CN-Retail 2.0 Comprehensive Multi-Tier Benchmark",
            "mode": "live_llm" if live else "rules_only",
            "llm_model": (getattr(load_config(), "model", "") if live else None),
            "total_cases": len(CN_RETAIL_200_CASES),
            "database_rows": 88002,
            "database_tables": 8,
            "strategies_tested": [s.name for s in strategies],
            "total_executions": len(CN_RETAIL_200_CASES) * len(strategies),
            "shapes_observed": len(unique_shapes),
            "baseline_strategy": baseline_name,
            "memory_scope_versions_by_strategy": {
                name: sorted({o.memory_scope_version or "legacy" for o in outcomes})
                for name, outcomes in strategy_outcomes.items()
            },
            "evaluation_fingerprints_by_strategy": {
                name: sorted({o.evaluation_fingerprint or "legacy" for o in outcomes})
                for name, outcomes in strategy_outcomes.items()
            },
        },
        "tier_distribution": {
            "Tier_1_Single_Table_Filters": 60,
            "Tier_2_Multi_Table_JOINs": 70,
            "Tier_3_Complex_Diagnostics": 50,
            "Tier_4_RedTeam_Clarify": 20,
        },
        "tier_performance": tier_stats,
        "uniform_policies": {k: v.to_payload() for k, v in strategy_metrics.items()},
        "candidate_policy": {
            "accepted": accepted,
            "reasons": rejected_reasons,
            "baseline": baseline_metrics.to_payload(),
            "candidate": candidate_metrics.to_payload(),
            "improvement_pct": round(improvement_pct, 4),
            "shape_distribution": shape_distribution,
            "proposed_policy": proposed_policy,
        },
        "sage_gate_verdict": gate_verdict.to_payload(),
        "improvement_dynamics": dynamics.to_payload(),
        "data_quality": {
            "note": (
                "A case is 'degraded' when a real remote model it should have used "
                "silently fell back: the reranker to the heuristic backend "
                "(rerank_degraded), the embedding to the hash vector on a 429/timeout "
                "(embedding_degraded), or the LLM call raised (llm_error). Every "
                "degraded instance is re-run in a healthy window until it completes "
                "cleanly, so the final numbers reflect the real full chain."
            ),
            "rerank_degraded_by_strategy": rerank_degraded_by_strategy,
            "embedding_degraded_by_strategy": embedding_degraded_by_strategy,
            "llm_error_by_strategy": llm_error_by_strategy,
            "total_rerank_degraded": sum(rerank_degraded_by_strategy.values()),
            "total_embedding_degraded": sum(embedding_degraded_by_strategy.values()),
            "total_llm_error": sum(llm_error_by_strategy.values()),
            "total_degraded": sum(len(v) for v in degraded_ids_by_strategy.values()),
            "degraded_case_ids_by_strategy": degraded_ids_by_strategy,
        },
        "grading": {
            "note": (
                "Authoritative success uses the v2 grader (alias/scale/extra-column "
                "tolerant, with a single consistent column mapping). legacy_pass_rate "
                "is the old exact value-multiset grader. recovered_by_v2 lists row-cases "
                "the v2 grader now scores correct that the legacy grader marked wrong; "
                "regressed_under_v2 lists any the v2 grader is stricter on. Per-case "
                "agent SQL and both row sets are cached, so re-grading is free offline."
            ),
            "by_strategy": grading_by_strategy,
        },
    }

    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Report Saved] Wrote full evaluation report to {out_file.resolve()}\n")

    # Console summary
    print("=" * 80)
    print("CN-RETAIL 2.0 (200 CASES) SELF-EVOLUTION & BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<16}{'Success Rate':>14}{'Recall':>10}{'Cost ($)':>12}{'Latency (ms)':>14}{'Score':>12}")
    print("-" * 80)
    for name, m in strategy_metrics.items():
        print(f"{name:<16}{m.success*100:>13.2f}%{m.recall*100:>9.2f}%{m.cost:>12.5f}{m.latency:>14.2f}{m.score:>12.2f}")
    print("-" * 80)
    print(f"{'Candidate Policy':<16}{candidate_metrics.success*100:>13.2f}%{candidate_metrics.recall*100:>9.2f}%{candidate_metrics.cost:>12.5f}{candidate_metrics.latency:>14.2f}{candidate_metrics.score:>12.2f}")
    print("=" * 80)
    print(f"SAGE Statistical Gate: Decision = {gate_verdict.decision.value.upper()} (p={gate_verdict.p_value:.6f})")
    print(f"Improvement %: {improvement_pct:+.4f}%")
    print(
        f"degraded total: {sum(len(v) for v in degraded_ids_by_strategy.values())} | "
        f"rerank {rerank_degraded_by_strategy} | embedding {embedding_degraded_by_strategy} | "
        f"llm {llm_error_by_strategy}"
    )
    print("=" * 80)

    return report_payload


def verify_cases_database(conn: sqlite3.Connection) -> None:
    """Ensure all gold SQL queries execute cleanly."""
    cur = conn.cursor()
    errors = []
    for c in CN_RETAIL_200_CASES:
        sql = c.get("gold_sql")
        if sql:
            try:
                cur.execute(sql)
                _ = cur.fetchall()
            except Exception as e:
                errors.append((c["id"], sql, str(e)))
    if errors:
        for cid, sql, err in errors:
            print(f"[ERROR] {cid}: {err} | {sql}")
        raise RuntimeError(f"Found {len(errors)} invalid gold_sql queries!")
    print(f"[Verification] All {len(CN_RETAIL_200_CASES)} cases verified successfully.")


@dataclass(slots=True)
class CaseExec:
    """Everything one case execution produced, including the grading audit trail."""

    passed: bool
    recall: float
    latency_ms: float
    cost_usd: float
    agents_used: list[str]
    rerank_degraded: bool
    llm_error: bool
    embedding_degraded: bool
    passed_legacy: bool = False
    overlap: float = 0.0
    got_sql: str = ""
    gold_rows_json: str = ""
    agent_rows_json: str = ""
    grade_error: str = ""
    planning_diagnostics: dict[str, Any] = field(default_factory=dict)
    execution_diagnostics: dict[str, Any] = field(default_factory=dict)
    attempt_id: str = ""


def _rows_json(rows: list[dict[str, Any]]) -> str:
    """Serialise result rows for offline re-grading; capped to bound cache size."""

    return json.dumps(rows[:500], ensure_ascii=False, default=str)


async def execute_case(
    orchestrator: InsightOrchestrator,
    conn: sqlite3.Connection,
    case: dict[str, Any],
) -> CaseExec:
    # Snapshot the shared reranker's fallback counter so we can attribute any
    # remote-reranker degradation to exactly this case. Execution is strictly
    # sequential and rich_vector/deep_hybrid issue one rerank call per case, so
    # a counter increment across this run belongs unambiguously to this case.
    reranker = getattr(getattr(orchestrator, "retriever", None), "bge_reranker", None)
    fb_before = getattr(reranker, "remote_fallback_count", 0)
    # Same idea for the embedding adapter. The vector store embeds through the
    # module-level singleton (compute_dense_embedding -> get_default_bge_adapter),
    # so snapshot that singleton's counter: any increment means a 429/timeout
    # forced this case onto the hash embedding.
    emb_adapter = get_default_bge_adapter()
    emb_before = getattr(emb_adapter, "remote_fallback_count", 0)

    # A new identity AND a fresh local MemoryStore are required on every call:
    # changing only the session would retain user profiles/episodic examples.
    # The orchestrator owns the scope so Redis user-context and tool callbacks
    # cannot accidentally keep referencing the parent memory or mutate it.
    attempt_id = uuid.uuid4().hex
    request = InsightRequest(
        question=case["question"],
        user_id=f"eval_user_{attempt_id}",
        session_id=f"eval_session_{attempt_id}",
    )
    last_stage = "case_memory_scope"

    def on_event(event: dict[str, Any]) -> None:
        nonlocal last_stage
        if isinstance(event.get("stage"), str):
            last_stage = event["stage"]

    t0 = time.perf_counter()
    try:
        with orchestrator.case_memory_scope(MemoryStore(local_only=True)) as scoped:
            last_stage = "run"
            answer = await scoped.run(request, on_event=on_event)
    except Exception as exc:  # a single live-model failure must not abort the sweep
        latency_ms = (time.perf_counter() - t0) * 1000
        planning = _diagnostics(getattr(exc, "planning_diagnostics", {}))
        diagnostics = _diagnostics({
            "user_id": request.user_id,
            "session_id": request.session_id,
            "failure_stage": planning.get("failure_stage") or planning.get("stage") or last_stage,
            "exception_type": type(exc).__name__,
            "final_error": f"{type(exc).__name__}: {exc}",
        })
        print(
            f"\n    [case {case['id']} failed] "
            + json.dumps({"planning": planning, "execution": diagnostics}, ensure_ascii=False),
            flush=True,
        )
        # Preserve the sweep's retry-on-raised-exception policy, but retain the
        # actual failure stage/type instead of labelling every cause as planning.
        return CaseExec(
            passed=False,
            recall=0.0,
            latency_ms=latency_ms,
            cost_usd=0.0,
            agents_used=[],
            rerank_degraded=getattr(reranker, "remote_fallback_count", 0) > fb_before,
            llm_error=True,
            embedding_degraded=getattr(emb_adapter, "remote_fallback_count", 0) > emb_before,
            planning_diagnostics=planning,
            execution_diagnostics=diagnostics,
            attempt_id=attempt_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    rerank_degraded = getattr(reranker, "remote_fallback_count", 0) > fb_before
    embedding_degraded = getattr(emb_adapter, "remote_fallback_count", 0) > emb_before
    planning = _diagnostics(getattr(answer, "planning_diagnostics", {}))
    diagnostics = {
        "user_id": request.user_id,
        "session_id": request.session_id,
    }
    # Only read public telemetry actually supplied by the response. Do not
    # inspect planner/client internals or turn missing fields into fake causes.
    for name in (
        "model", "planner", "strategy", "route_reason", "route_score", "trace_id", "intent_decision",
        "blocked", "block_reason", "reflections", "alerts", "tools_used", "agents_used",
        "context_compressed", "context_summary", "clarification_needed", "repair_attempts",
    ):
        if hasattr(answer, name):
            diagnostics[name] = getattr(answer, name)
    if "clarification_needed" in planning:
        diagnostics["clarification_needed"] = planning["clarification_needed"]
    elif any(
        isinstance(alert, dict) and alert.get("kind") == "clarification_needed"
        for alert in getattr(answer, "alerts", [])
    ):
        diagnostics["clarification_needed"] = True
    if planning.get("failure_stage"):
        diagnostics["failure_stage"] = planning["failure_stage"]
    elif planning.get("stage") and planning.get("final_error"):
        diagnostics["failure_stage"] = planning["stage"]
    if planning.get("final_error"):
        diagnostics["final_error"] = planning["final_error"]
    # A2A may convert a provider exception into a normal response. Rerun only
    # its explicit failure flag, never a legitimate wrong SQL/clarification.
    llm_error = planning.get("llm_error") is True

    # 1. Check Retrieval Recall
    requires = case.get("requires_tables", [])
    recall = 1.0
    if requires:
        if not orchestrator.config.use_rag:
            recall = 0.0
        else:
            retrieved_tables = set()
            for ev in answer.evidence:
                tbl = ev.get("table", "")
                if tbl:
                    retrieved_tables.add(tbl)
            recall = len(set(requires) & retrieved_tables) / len(requires) if requires else 1.0

    passed = False
    passed_legacy = False
    overlap = 0.0
    got_sql = answer.sql or ""
    gold_rows_json = ""
    agent_rows_json = ""
    grade_error = ""
    expectation = case.get("expectation", "rows")

    if expectation == "blocked":
        passed = passed_legacy = bool(answer.blocked)
    elif expectation == "clarify":
        passed = passed_legacy = not bool(answer.sql) and (
            "抱歉" in answer.answer or "could not" in answer.answer.lower() or "未能" in answer.answer
        )
    elif not answer.blocked and case.get("gold_sql"):
        # 2. Execute both queries and grade. ``passed`` uses the v2 grader
        # (alias/scale/extra-column tolerant); ``passed_legacy`` keeps the old
        # exact-multiset verdict so the report can quantify the delta. Both row
        # sets and the agent SQL are captured so any future grader change is a
        # free offline re-grade.
        if answer.sql:
            cur = conn.cursor()
            grade_stage = "grading_gold_sql"
            try:
                cur.execute(case["gold_sql"])
                gold_rows = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
                grade_stage = "grading_agent_sql"
                cur.execute(answer.sql)
                agent_rows = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
                grade_stage = "grading_rows"
                gold_rows_json = _rows_json(gold_rows)
                agent_rows_json = _rows_json(agent_rows)
                passed_legacy = rows_match(gold_rows, agent_rows)
                passed = rows_match_v2(gold_rows, agent_rows)
                overlap = row_overlap(gold_rows, agent_rows)
            except Exception as exc:
                passed = passed_legacy = False
                grade_error = _diagnostic_text(f"{type(exc).__name__}: {exc}")
                diagnostics["grading_failure_stage"] = grade_stage
                diagnostics.setdefault("failure_stage", grade_stage)
                diagnostics["grading_error"] = grade_error
            finally:
                cur.close()

    if not passed or llm_error:
        diagnostics["answer"] = getattr(answer, "answer", "")
        if not passed:
            diagnostics["grading_expectation"] = expectation
            diagnostics["grading_reason"] = (
                "grading_error" if grade_error else
                "blocked" if answer.blocked else
                "missing_sql" if expectation == "rows" and not answer.sql else
                "rows_mismatch" if expectation == "rows" else
                "expectation_not_met"
            )
        print(
            f"\n    [case {case['id']} diagnostics] "
            + json.dumps({"planning": planning, "execution": _diagnostics(diagnostics)}, ensure_ascii=False),
            flush=True,
        )

    # Calculate realistic cost & latency directly from orchestrator telemetry
    cost_usd = answer.cost_usd
    return CaseExec(
        passed=passed,
        recall=recall,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        agents_used=answer.agents_used,
        rerank_degraded=rerank_degraded,
        llm_error=llm_error,
        embedding_degraded=embedding_degraded,
        passed_legacy=passed_legacy,
        overlap=overlap,
        got_sql=got_sql,
        gold_rows_json=gold_rows_json,
        agent_rows_json=agent_rows_json,
        grade_error=grade_error,
        planning_diagnostics=planning,
        execution_diagnostics=_diagnostics(diagnostics),
        attempt_id=attempt_id,
    )


def _attempt_snapshot(outcome: Outcome) -> dict[str, Any]:
    """Keep failed-attempt evidence without recursively nesting prior history."""
    return _diagnostics({
        "attempt_id": outcome.attempt_id,
        "memory_scope_version": outcome.memory_scope_version,
        "evaluation_fingerprint": outcome.evaluation_fingerprint,
        "passed": outcome.passed,
        "llm_error": outcome.llm_error,
        "rerank_degraded": outcome.rerank_degraded,
        "embedding_degraded": outcome.embedding_degraded,
        "latency_ms": outcome.latency_ms,
        "cost_usd": outcome.cost_usd,
        "got_sql": outcome.got_sql,
        "grade_error": outcome.grade_error,
        "planning_diagnostics": outcome.planning_diagnostics,
        "execution_diagnostics": outcome.execution_diagnostics,
    })


async def run_one_case(
    orchestrator: InsightOrchestrator,
    conn: sqlite3.Connection,
    case: dict[str, Any],
    strat_name: str,
    *,
    previous: Outcome | None = None,
    live: bool | None = None,
) -> Outcome:
    """Run with fresh memory; prior outcomes are used only for the audit trail."""
    shape = classify_shape(case["question"]).key
    ex = await execute_case(orchestrator, conn, case)
    recall = ex.recall
    # In cheap_flat without RAG, retrieval recall is penalized heavily for multi-table queries
    if not orchestrator.config.use_rag and case.get("requires_tables"):
        recall = 0.0
    if live is None:
        live = getattr(getattr(orchestrator, "planner", None), "name", "rules") != "rules"
    history = []
    if previous is not None:
        history = [*previous.attempt_history, _attempt_snapshot(previous)][-MAX_ATTEMPT_HISTORY:]
    return Outcome(
        case_id=case["id"],
        shape=shape,
        strategy=strat_name,
        passed=ex.passed,
        recall=recall,
        latency_ms=ex.latency_ms,
        cost_usd=ex.cost_usd,
        rerank_degraded=ex.rerank_degraded,
        llm_error=ex.llm_error,
        embedding_degraded=ex.embedding_degraded,
        passed_legacy=ex.passed_legacy,
        overlap=ex.overlap,
        got_sql=ex.got_sql,
        gold_rows_json=ex.gold_rows_json,
        agent_rows_json=ex.agent_rows_json,
        grade_error=ex.grade_error,
        planning_diagnostics=ex.planning_diagnostics,
        execution_diagnostics=ex.execution_diagnostics,
        attempt_id=ex.attempt_id,
        memory_scope_version=CASE_MEMORY_SCOPE_VERSION,
        evaluation_fingerprint=evaluation_fingerprint(strat_name, orchestrator.config, live=live),
        retry_count=previous.retry_count + 1 if previous is not None else 0,
        attempt_history=history,
    )


def _cache_is_clean(
    name: str,
    *,
    live: bool = True,
    config: RunConfig | None = None,
) -> bool:
    """Only complete, current-protocol outcomes may skip a fresh execution.

    Regrading or repairing only part of a legacy cache must not certify its
    untouched outcomes as memory-isolated. Wrong answers still count as clean:
    this gate is about execution quality, not picking favourable model rolls.
    """
    path = strategy_cache_path(name)
    if not path.exists():
        return False
    if config is None:
        strat = next((s for s in default_strategies() if s.name == name), None)
        if strat is None:
            return False
        config = strat.config
    fingerprint = evaluation_fingerprint(name, config, live=live)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        outs = [Outcome(**o) for o in data["outcomes"]]
        ids = [o.case_id for o in outs]
        return (
            data.get("cache_version") == STRATEGY_CACHE_VERSION
            and data.get("memory_scope_version") == CASE_MEMORY_SCOPE_VERSION
            and data.get("evaluation_fingerprint") == fingerprint
            and data.get("mode") == ("live_llm" if live else "rules_only")
            and data.get("strategy") == name
            and len(ids) == len(CN_RETAIL_200_CASES)
            and set(ids) == {c["id"] for c in CN_RETAIL_200_CASES}
            and all(
                o.strategy == name
                and o.memory_scope_version == CASE_MEMORY_SCOPE_VERSION
                and bool(o.attempt_id)
                and o.evaluation_fingerprint == fingerprint
                for o in outs
            )
            and not any(o.rerank_degraded or o.llm_error or o.embedding_degraded for o in outs)
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


async def run_sweep(
    conn: sqlite3.Connection,
    strategies: list[Strategy],
    planner: Planner | None = None,
    *,
    max_repair_rounds: int = 0,
    repair_pause: float = 15.0,
    reuse_clean_cache: bool = False,
) -> tuple[dict[str, list[Outcome]], dict[str, dict[str, bool]], dict[str, PolicyMetrics]]:
    """Runs the strategies over the 200 cases.

    When ``planner`` is supplied (live mode) every strategy's orchestrator is
    driven by the configured LLM planner; the orchestrator still downgrades
    simple questions to the deterministic compiler on its own.

    ``max_repair_rounds`` (>0, live only) re-runs each strategy's failed
    instances — those whose reranker fell back or whose LLM call raised — in
    place until the strategy completes cleanly or the cap is hit, so a transient
    endpoint bad window never pollutes a strategy's final numbers.
    ``reuse_clean_cache`` skips strategies only when their complete,
    failure-free cache matches this isolation protocol, config and mode.
    """
    retriever = build_retriever(conn)
    strategy_outcomes: dict[str, list[Outcome]] = {}
    strategy_per_item: dict[str, dict[str, bool]] = {}
    strategy_metrics: dict[str, PolicyMetrics] = {}
    case_by_id = {c["id"]: c for c in CN_RETAIL_200_CASES}

    def _failed(o: Outcome) -> bool:
        # Any degraded instance worth re-running: the reranker fell back to the
        # heuristic, the embedding fell back to the hash (429/timeout), or the
        # LLM call raised. A legitimate wrong answer is NOT a failure here and is
        # kept as a real result.
        return o.rerank_degraded or o.llm_error or o.embedding_degraded

    mode = "LIVE LLM" if planner is not None else "rules-only"
    print(f"\n[Starting Sweep] {len(strategies)} Strategies x {len(CN_RETAIL_200_CASES)} Cases ({mode})...")
    if planner is not None and max_repair_rounds > 0:
        print(f"[Repair] per-strategy: re-run failed instances up to {max_repair_rounds} rounds "
              f"(pause {repair_pause:.0f}s) until all complete cleanly.")
    t_start = time.perf_counter()

    for strat in strategies:
        # Reuse a prior fully-clean cache so strategies unaffected by failures
        # are not re-run (per request: "沿用之前的记过").
        if reuse_clean_cache and _cache_is_clean(
            strat.name, live=planner is not None, config=strat.config
        ):
            outcomes = load_strategy_cache(strat.name)
            strategy_outcomes[strat.name] = outcomes
            strategy_per_item[strat.name] = {o.case_id: o.passed for o in outcomes}
            strategy_metrics[strat.name] = metrics_from_outcomes(outcomes)
            print(f"  -> Reusing clean cache for {strat.name:<14} "
                  f"({len(outcomes)} cases, 0 failed) — not re-run.")
            continue

        print(f"  -> Running Strategy: {strat.name:<14} ({strat.description})...", flush=True)
        t_strat = time.perf_counter()
        orchestrator = InsightOrchestrator(
            conn,
            memory=MemoryStore(local_only=True),
            config=strat.config,
            retriever=retriever,
            planner=planner,
        )

        by_id: dict[str, Outcome] = {}
        for idx, case in enumerate(CN_RETAIL_200_CASES, start=1):
            by_id[case["id"]] = await run_one_case(
                orchestrator, conn, case, strat.name, live=planner is not None
            )
            if planner is not None and idx % 25 == 0:
                print(f"       ...{idx}/{len(CN_RETAIL_200_CASES)} cases ({time.perf_counter() - t_strat:.0f}s elapsed)", flush=True)

        # Per-strategy repair loop: re-run every failed instance until the whole
        # strategy completed cleanly (real reranker + successful LLM) or the
        # round cap is hit. Clean instances are never touched, so they are not
        # re-rolled through LLM nondeterminism.
        if planner is not None and max_repair_rounds > 0:
            for rnd in range(1, max_repair_rounds + 1):
                failed = [cid for cid, o in by_id.items() if _failed(o)]
                if not failed:
                    break
                print(f"     [repair {strat.name}] round {rnd}/{max_repair_rounds}: "
                      f"re-running {len(failed)} failed instance(s)...", flush=True)
                for cid in failed:
                    by_id[cid] = await run_one_case(
                        orchestrator, conn, case_by_id[cid], strat.name,
                        previous=by_id[cid], live=True,
                    )
                if [c for c, o in by_id.items() if _failed(o)] and rnd < max_repair_rounds:
                    await asyncio.sleep(repair_pause)
            still = [cid for cid, o in by_id.items() if _failed(o)]
            if still:
                print(f"     [repair {strat.name}] WARNING: {len(still)} instance(s) still failing after "
                      f"{max_repair_rounds} rounds; finalizing best-effort: {still[:10]}"
                      f"{'...' if len(still) > 10 else ''}", flush=True)

        outcomes = [by_id[c["id"]] for c in CN_RETAIL_200_CASES]
        per_item = {o.case_id: o.passed for o in outcomes}

        n = len(outcomes)
        success_rate = sum(1 for o in outcomes if o.passed) / n
        avg_recall = sum(o.recall for o in outcomes) / n
        total_cost = sum(o.cost_usd for o in outcomes)
        avg_lat = sum(o.latency_ms for o in outcomes) / n
        avg_score = sum(o.score for o in outcomes) / n
        degraded = sum(1 for o in outcomes if o.rerank_degraded)
        llm_errs = sum(1 for o in outcomes if o.llm_error)
        emb_deg = sum(1 for o in outcomes if o.embedding_degraded)

        strategy_outcomes[strat.name] = outcomes
        strategy_per_item[strat.name] = per_item
        strategy_metrics[strat.name] = PolicyMetrics(
            success=success_rate,
            recall=avg_recall,
            cost=total_cost,
            latency=avg_lat,
            score=avg_score,
            cases=n,
        )
        # Persist this strategy immediately so a later crash/stop never loses a
        # finished strategy, and so degraded instances can be re-run in isolation.
        cache_path = write_strategy_cache(strat.name, outcomes, live=planner is not None)
        print(
            f"     Done in {time.perf_counter() - t_strat:.2f}s | Success: {success_rate*100:.1f}%, "
            f"Recall: {avg_recall*100:.1f}%, Score: {avg_score:.2f} | "
            f"rerank_degraded: {degraded}, llm_error: {llm_errs}, embedding_degraded: {emb_deg} | cache -> {cache_path}"
        )

    print(f"[Sweep Complete] Total time: {time.perf_counter() - t_start:.2f}s\n")
    return strategy_outcomes, strategy_per_item, strategy_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="CN-Retail 2.0 200-case benchmark")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Drive the deep-path through the configured LLM (real full-chain eval).",
    )
    parser.add_argument(
        "--strategies",
        default="",
        help="Comma-separated subset of strategies to sweep (default: all). "
             "With a strict subset, only per-strategy caches are written; run "
             "rebuild_live_report to re-aggregate once all four caches exist.",
    )
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=25,
        help="Live only: after each strategy, re-run its failed instances "
             "(reranker fell back OR LLM raised) up to this many rounds until "
             "all complete cleanly. 0 disables repair (default 25).",
    )
    parser.add_argument(
        "--repair-pause",
        type=float,
        default=15.0,
        help="Seconds to wait between repair rounds (default 15).",
    )
    parser.add_argument(
        "--reuse-clean-cache",
        action="store_true",
        help="Skip complete, failure-free caches matching the current isolated-case "
             "protocol, strategy config, benchmark and execution mode.",
    )
    args = parser.parse_args()

    live_planner: Planner | None = None
    if args.live:
        cfg = load_config()
        reason = cfg.missing_reason()
        if not cfg.is_live or reason:
            print(f"[FATAL] --live requested but no live LLM configured: {reason or 'not live'}")
            return 2
        live_planner = build_planner(cfg)
        if type(live_planner).__name__ != "LLMPlanner":
            print("[FATAL] --live requested but build_planner degraded to RulesPlanner; check .env")
            return 2
        print(f"[LIVE MODE] Deep-path planner = {type(live_planner).__name__} | model = {cfg.model} | base = {cfg.base_url}")

    ensure_cn_retail_database()
    conn = connect_cn_retail()
    verify_cases_database(conn)

    all_strategies = default_strategies()
    requested = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if requested:
        unknown = [r for r in requested if r not in {s.name for s in all_strategies}]
        if unknown:
            print(f"[FATAL] unknown strategies: {unknown}; valid = {[s.name for s in all_strategies]}")
            return 2
        strategies = [s for s in all_strategies if s.name in requested]
    else:
        strategies = all_strategies

    # 1. Sweep requested strategies. Each strategy repairs its own failed
    #    instances in place (live) and writes its per-case cache.
    repair_rounds = args.max_repair_rounds if live_planner is not None else 0
    strategy_outcomes, _per_item, _metrics = asyncio.run(
        run_sweep(
            conn,
            strategies,
            planner=live_planner,
            max_repair_rounds=repair_rounds,
            repair_pause=args.repair_pause,
            reuse_clean_cache=args.reuse_clean_cache,
        )
    )

    # A strict subset cannot produce the full candidate-policy / SAGE report,
    # which needs every strategy's per-case outcomes. Leave the caches for the
    # rebuild step and stop here.
    if len(strategies) < len(all_strategies):
        print(
            f"[Partial Sweep] Swept {[s.name for s in strategies]}; per-strategy caches written to "
            f"{STRATEGY_CACHE_DIR}. Run `python -m insight_agent.eval.rebuild_live_report` once all "
            f"{len(all_strategies)} caches exist to regenerate the full report."
        )
        return 0

    out_file = LIVE_REPORT_PATH if live_planner is not None else RULES_REPORT_PATH
    aggregate_and_report(strategy_outcomes, all_strategies, live=live_planner is not None, out_file=out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
