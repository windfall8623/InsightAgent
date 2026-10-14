"""Re-aggregate the live 200-case report from per-strategy outcome caches.

Each strategy sweep writes ``docs/_live_strategy_cache/<strategy>.json`` with its
per-case outcomes. This script reads all four, recomputes the candidate policy,
the SAGE statistical gate and the full report payload, and writes
``docs/cn_retail_200_live_benchmark_report.json`` — without re-running any model.

Use it after ``rerun_degraded`` has spliced clean outcomes back into one or more
strategy caches, to fold the repaired data into the headline report.

    uv run python -m insight_agent.eval.rebuild_live_report
"""

from __future__ import annotations

import sys

from insight_agent.app.evolution import default_strategies
from insight_agent.eval.run_cn_retail_200_eval import (
    LIVE_REPORT_PATH,
    STRATEGY_CACHE_DIR,
    aggregate_and_report,
    load_strategy_cache,
    regrade_strategy_caches,
    strategy_cache_path,
)


def main() -> int:
    strategies = default_strategies()
    regrade = "--regrade" in sys.argv
    missing = [s.name for s in strategies if not strategy_cache_path(s.name).exists()]
    if missing:
        print(f"[FATAL] missing strategy caches in {STRATEGY_CACHE_DIR}: {missing}")
        print("Run the full sweep (or rerun_degraded) until every strategy cache exists.")
        return 2

    if regrade:
        # Re-score the cached outcomes with the current grader (no model calls)
        # and persist the updated verdicts before aggregating.
        strategy_outcomes = regrade_strategy_caches([s.name for s in strategies], write=True)
    else:
        strategy_outcomes = {s.name: load_strategy_cache(s.name) for s in strategies}
    for name, outcomes in strategy_outcomes.items():
        if len(outcomes) != 200:
            print(f"[WARN] strategy '{name}' cache has {len(outcomes)} outcomes (expected 200).")

    aggregate_and_report(strategy_outcomes, strategies, live=True, out_file=LIVE_REPORT_PATH)
    print(f"[Rebuilt] {LIVE_REPORT_PATH.resolve()}")
    print("Next: uv run python -m insight_agent.eval.gen_live_report_md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
