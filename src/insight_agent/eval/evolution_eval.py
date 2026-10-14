"""Evaluate the self-evolution loop against a uniform baseline.

The question this answers is narrow and falsifiable: *does choosing a strategy
per question shape beat running one strategy for everything?*

Method
------
1. Sweep every candidate strategy over the whole dataset, recording an
   :class:`Outcome` per case. This is the ``observe`` phase and it is what the
   loop learns from.
2. Score a uniform policy (one strategy everywhere) as the baseline.
3. Let the engine propose a shape-conditioned policy from those observations
   alone, then validate it against the same dataset.
4. Report both, plus the regression guard's verdict.

**Honesty note.** Step 3 is tuned and scored on the same dataset, so the result
is *not* a held-out generalisation claim. It shows that per-shape routing
recovers at least the uniform baseline's correctness while spending differently
per question type — nothing more. The report labels this explicitly, and the
guard rejects any candidate that loses success or recall.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from insight_agent.app.evolution import (
    EvolutionEngine,
    Outcome,
    PolicyMetrics,
    classify_shape,
    default_strategies,
    uniform_policy,
)
from insight_agent.app.memory import MemoryStore
from insight_agent.app.orchestrator import InsightOrchestrator, InsightRequest
from insight_agent.app.rag import build_retriever
from insight_agent.data.database import connect, execute
from insight_agent.eval.dataset import load_cases
from insight_agent.eval.metrics import retrieval_recall, rows_match
from insight_agent.eval.runner import _schema_identifiers


async def _sweep(engine: EvolutionEngine, conn: Any, *, limit: int | None = None) -> None:
    """Observe: run every strategy over every case and record the outcome."""

    retriever = build_retriever(conn)
    tables, columns = _schema_identifiers(conn)

    cases = load_cases()[:limit] if limit else load_cases()
    for strategy in engine.strategies:
        for case in cases:
            orchestrator = InsightOrchestrator(
                conn,
                memory=MemoryStore(),
                config=strategy.config,
                retriever=retriever,
            )
            started = time.perf_counter()
            answer = await orchestrator.run(
                InsightRequest(question=case["question"], user_id="eval")
            )
            latency = (time.perf_counter() - started) * 1000

            passed = False
            if case["expectation"] == "blocked":
                passed = answer.blocked
            elif case["expectation"] == "clarify":
                passed = not answer.sql and "could not" in answer.answer.lower()
            elif not answer.blocked and case.get("gold_sql"):
                try:
                    gold = execute(conn, case["gold_sql"], limit=500).as_dicts()
                    passed = rows_match(gold, answer.rows)
                except Exception:  # noqa: BLE001 - a bad gold query is a dataset bug
                    passed = False

            engine.observe(
                Outcome(
                    case_id=case["id"],
                    shape=classify_shape(case["question"]).key,
                    strategy=strategy.name,
                    passed=passed,
                    recall=retrieval_recall(answer.evidence, case.get("requires_tables") or []),
                    latency_ms=latency,
                    cost_usd=answer.cost_usd,
                )
            )


def _make_evaluator(engine: EvolutionEngine, conn: Any) -> Any:
    """Build the evaluator that scores a whole policy in one pass.

    The most recent evaluation's per-item outcomes are stored on ``engine`` so
    :meth:`EvolutionEngine.evolve` can hand them to the statistical gate. Without
    this the gate has no evidence and abstains, which is correct but useless.
    """

    async def _evaluate(policy: dict[str, str]) -> PolicyMetrics:
        retriever = build_retriever(conn)
        metrics = PolicyMetrics()
        outcomes: list[Outcome] = []

        for case in load_cases():
            shape = classify_shape(case["question"]).key
            name = policy.get(shape) or policy.get("*") or engine.baseline_name
            strategy = engine.strategy(name)
            assert strategy is not None, f"unknown strategy {name!r}"

            orchestrator = InsightOrchestrator(
                conn, memory=MemoryStore(), config=strategy.config, retriever=retriever
            )
            started = time.perf_counter()
            answer = await orchestrator.run(
                InsightRequest(question=case["question"], user_id="eval")
            )
            latency = (time.perf_counter() - started) * 1000

            passed = False
            if case["expectation"] == "blocked":
                passed = answer.blocked
            elif case["expectation"] == "clarify":
                passed = not answer.sql and "could not" in answer.answer.lower()
            elif not answer.blocked and case.get("gold_sql"):
                try:
                    gold = execute(conn, case["gold_sql"], limit=500).as_dicts()
                    passed = rows_match(gold, answer.rows)
                except Exception:  # noqa: BLE001
                    passed = False

            outcome = Outcome(
                case_id=case["id"],
                shape=shape,
                strategy=name,
                passed=passed,
                recall=retrieval_recall(answer.evidence, case.get("requires_tables") or []),
                latency_ms=latency,
                cost_usd=answer.cost_usd,
            )
            outcomes.append(outcome)

        n = max(len(outcomes), 1)
        metrics.cases = len(outcomes)
        metrics.success = sum(1 for o in outcomes if o.passed) / n
        metrics.recall = sum(o.recall for o in outcomes) / n
        metrics.cost = sum(o.cost_usd for o in outcomes)
        metrics.latency = sum(o.latency_ms for o in outcomes) / n
        metrics.score = sum(o.score for o in outcomes) / n
        engine.last_outcomes = {o.case_id: o.passed for o in outcomes}
        return metrics

    def sync(policy: dict[str, str]) -> PolicyMetrics:
        return asyncio.run(_evaluate(policy))

    return sync


def run_evolution_eval(*, out_dir: Path | None = None, limit: int | None = None) -> dict[str, Any]:
    """Run the full observe -> propose -> validate -> accept cycle.

    Synchronous entry point: the sweep is the only genuinely async phase, and
    the policy evaluator is sync by design (``EvolutionEngine.evolve`` is a
    sync method), so nesting ``asyncio.run`` inside a running loop would fail.
    """

    conn = connect()
    engine = EvolutionEngine(default_strategies())
    asyncio.run(_sweep(engine, conn, limit=limit))

    evaluator = _make_evaluator(engine, conn)

    # Baseline sweep: every uniform policy, so the comparison is honest about
    # which uniform choice is the one to beat.
    uniform_scores: dict[str, PolicyMetrics] = {}
    for strategy in engine.strategies:
        uniform_scores[strategy.name] = evaluator(uniform_policy(strategy.name))

    baseline_outcomes = dict(engine.last_outcomes)
    candidate_placeholder = engine.propose()
    if candidate_placeholder:
        evaluator(candidate_placeholder)
    candidate_outcomes = dict(engine.last_outcomes)

    report = engine.evolve(
        evaluator,
        baseline_name=engine.baseline_name,
        per_item=(baseline_outcomes, candidate_outcomes),
    )

    shapes_seen = sorted({o.shape for o in engine.outcomes})
    # Per-shape routing decisions, so a no-op policy is visibly distinguishable
    # from a policy that reroutes without changing the totals.
    distribution: dict[str, int] = {}
    for name in report.policy.values():
        distribution[name] = distribution.get(name, 0) + 1
    payload = {
        "meta": {
            "cases": len(load_cases()[:limit] if limit else load_cases()),
            "strategies": [s.name for s in engine.strategies],
            "shapes_observed": len(shapes_seen),
            "observations": len(engine.outcomes),
            "baseline_policy": engine.baseline_name,
            "note": (
                "The candidate policy is proposed and scored on the same dataset "
                "it was tuned against; this is not a held-out generalisation result."
            ),
        },
        "uniform_policies": {k: v.to_payload() for k, v in uniform_scores.items()},
        "best_uniform": min(uniform_scores.items(), key=lambda kv: (kv[1].cost, -kv[1].score))[0],
        "evolution": report.to_payload(),
        "evolved_policy": report.policy,
        "shape_distribution": dict(sorted(distribution.items())),
    }

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "evolution_report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return payload


def main() -> int:
    payload = run_evolution_eval(out_dir=Path("eval_results"))
    meta = payload["meta"]
    evo = payload["evolution"]

    print(
        f"observations: {meta['observations']} "
        f"({meta['cases']} cases x {len(meta['strategies'])} strategies)"
    )
    print(f"question shapes observed: {meta['shapes_observed']}\n")

    print("uniform policies (one strategy for everything):")
    header = f"{'strategy':<14}{'success':>9}{'recall':>8}{'cost':>11}{'latency':>10}"
    print(header)
    print("-" * len(header))
    for name, m in payload["uniform_policies"].items():
        print(
            f"{name:<14}{m['success']:>9.4f}{m['recall']:>8.4f}"
            f"{m['cost']:>11.5f}{m['latency']:>10.2f}"
        )

    b, c = evo["baseline"], evo["candidate"]
    print(f"\nevolved policy (accepted={evo['accepted']}):")
    print(
        f"{'  baseline':<14}{b['success']:>9.4f}{b['recall']:>8.4f}"
        f"{b['cost']:>11.5f}{b['latency']:>10.2f}"
    )
    print(
        f"{'  candidate':<14}{c['success']:>9.4f}{c['recall']:>8.4f}"
        f"{c['cost']:>11.5f}{c['latency']:>10.2f}"
    )
    print(f"\ncomposite score improvement: {evo['improvement_pct']:+.2f}%")
    if evo["changed_shapes"]:
        print(f"shapes rerouted: {len(evo['changed_shapes'])}")
    if evo["rejected_reasons"]:
        print(f"rejected: {evo['rejected_reasons']}")
    print("\nwritten to eval_results/evolution_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
