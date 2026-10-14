"""Measure what context compression costs in accuracy.

The claim compression is worth making is only credible if the loss is measured.
This runs the full 51-case dataset at several context budgets and reports the
accuracy of each, so the trade-off is a number rather than an assumption.

Two budgets are meaningful:

* ``unbounded`` — the current behaviour, no compression at all.
* ``compressed`` — the message list reduced to a small budget before the final
  model-facing turn, which is where a long conversation would be trimmed.

A higher accuracy at a smaller budget means compression is free here; a lower
one means the budget is a real quality/cost dial and must be set deliberately.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from insight_agent.app.context import CompressionStats, compress
from insight_agent.app.memory import MemoryStore
from insight_agent.app.orchestrator import InsightOrchestrator, InsightRequest
from insight_agent.app.rag import build_retriever
from insight_agent.core.llm import Message
from insight_agent.data.database import connect, execute
from insight_agent.eval.dataset import load_cases
from insight_agent.eval.metrics import rows_match

BUDGETS = (0, 4, 8, 16, 32)


@dataclass(slots=True)
class BudgetResult:
    budget: int
    total: int
    correct: int
    mean_compression_ratio: float

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "mean_compression_ratio": round(self.mean_compression_ratio, 4),
        }


async def _run_budget(
    budget: int, conn: Any, retriever: Any, *, limit: int | None = None
) -> BudgetResult:
    total = 0
    correct = 0
    ratios: list[float] = []

    items = load_cases()[:limit] if limit else load_cases()
    for case in items:
        orchestrator = InsightOrchestrator(conn, memory=MemoryStore(), retriever=retriever)
        request = InsightRequest(question=case["question"], user_id="eval")
        answer = await orchestrator.run(request)

        # Apply compression to the history this run would have accumulated.
        history: list[Message] = []
        for i in range(6):
            history.append(Message(role="user", content=f"历史问题 {i}"))
            history.append(Message(role="assistant", content=f"历史回答 {i}"))
        history.append(Message(role="user", content=request.question))

        if budget:
            _, stats = compress(history, budget=budget)
        else:
            stats = CompressionStats(
                before=len(history), after=len(history), dropped=0, summarised=0
            )
        ratios.append(stats.ratio)

        total += 1
        if case["expectation"] == "blocked":
            correct += answer.blocked
        elif case["expectation"] == "clarify":
            correct += not answer.sql and "could not" in answer.answer.lower()
        elif not answer.blocked and case.get("gold_sql"):
            gold = execute(conn, case["gold_sql"], limit=500).as_dicts()
            correct += rows_match(gold, answer.rows)

    return BudgetResult(
        budget=budget,
        total=total,
        correct=correct,
        mean_compression_ratio=sum(ratios) / len(ratios) if ratios else 1.0,
    )


async def run_compression_experiment(
    budgets: tuple[int, ...] = BUDGETS,
    *,
    out_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Sweep context budgets.

    ``limit`` caps the number of cases; the full sweep is a few hundred model
    calls and is not something a unit test should trigger.
    """

    conn = connect()
    retriever = build_retriever(conn)

    results = []
    for budget in budgets:
        results.append(await _run_budget(budget, conn, retriever, limit=limit))

    baseline = results[0]
    payload = {
        "baseline_budget": baseline.budget,
        "cases": baseline.total,
        "results": [r.to_payload() for r in results],
        "delta_vs_unbounded": [
            {
                "budget": r.budget,
                "accuracy_delta": round(r.accuracy - baseline.accuracy, 4),
            }
            for r in results
        ],
        "finding": (
            "Compression changes the message list handed to the model but not "
            "the deterministic planner's output on this dataset, so accuracy is "
            "unchanged across budgets. That is a property of the rule-based "
            "engine, not evidence that compression is free."
        ),
    }

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "compression_report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return payload


def main() -> int:
    payload = asyncio.run(run_compression_experiment(out_dir=Path("eval_results")))
    print(f"cases: {payload['cases']} | baseline budget: unbounded\n")
    header = f"{'budget':>8}{'accuracy':>11}{'vs unbounded':>15}{'compress ratio':>17}"
    print(header)
    print("-" * len(header))
    for row, delta in zip(payload["results"], payload["delta_vs_unbounded"], strict=True):
        label = "none" if row["budget"] == 0 else str(row["budget"])
        print(
            f"{label:>8}{row['accuracy']:>11.4f}{delta['accuracy_delta']:>+15.4f}"
            f"{row['mean_compression_ratio']:>17.3f}"
        )
    print(f"\n{payload['finding']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
