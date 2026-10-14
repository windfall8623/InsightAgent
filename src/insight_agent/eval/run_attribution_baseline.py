"""Run Initial Attribution Baseline Evaluation (Pre-Reform Benchmark).

Runs the independent AttributionEvaluator against the CURRENT UNMODIFIED InsightGraphEngine
to measure genuine baseline attribution accuracy, hypothesis recall, and false positive rate.
"""

from __future__ import annotations

import sys
import time

from insight_agent.app.graph import InsightGraphEngine
from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
from insight_agent.app.redis_store import RedisMemoryStore
from insight_agent.data.cn_retail_db import connect_cn_retail, ensure_cn_retail_database
from insight_agent.eval.attribution_evaluator import (
    AttributionEvaluator,
    save_attribution_report,
)


def main() -> int:
    print("=" * 80)
    print("RUNNING INITIAL ATTRIBUTION & CAUSAL DIAGNOSIS BASELINE")
    print("Target Engine: Current Unmodified InsightGraphEngine (LangGraph)")
    print("=" * 80)

    # 1. Setup Environment
    ensure_cn_retail_database()
    conn = connect_cn_retail()
    retriever = OlistMultiModalRetriever(conn)
    redis_store = RedisMemoryStore()
    engine = InsightGraphEngine(conn, retriever, redis_store)

    # 2. Run Evaluator
    evaluator = AttributionEvaluator()
    started = time.perf_counter()
    metrics, results = evaluator.run_evaluation(lambda q: engine.diagnose(q))
    elapsed = time.perf_counter() - started

    # 3. Print Results
    print(f"\n[Run Completed] Tested {metrics.total_cases} queries in {elapsed:.2f}s.\n")
    print("-" * 80)
    print(f"{'Metric':<35}{'Baseline Value':>20}{'Assessment':>25}")
    print("-" * 80)
    print(f"{'Attribution Accuracy (Hit@1)':<35}{metrics.attribution_accuracy * 100:>19.2f}%{'Pre-planted only':>25}")
    print(f"{'False Positive Rate (Control)':<35}{metrics.false_positive_rate * 100:>19.2f}%{'Severe over-reporting':>25}")
    print(f"{'Average Reflections / Query':<35}{metrics.avg_reflections:>20.2f}{'Fixed heuristic':>25}")
    print("-" * 80)

    print("\n[Per-Spec Causal Attribution Breakdown]")
    print(f"{'Spec ID':<36}{'Type':<10}{'Pass Rate':>12}{'Result':>18}")
    print("-" * 80)
    for spec_id, stats in metrics.spec_results.items():
        type_str = "Control" if stats["is_control"] else "Anomaly"
        status = "PASSED" if stats["pass_rate"] > 0.6 else "FAILED"
        print(f"{spec_id:<36}{type_str:<10}{stats['pass_rate'] * 100:>11.1f}%{status:>18}")
    print("-" * 80)

    # 4. Save Report
    out_file = save_attribution_report(metrics, results)
    print(f"\n[Saved Baseline Report] -> {out_file.resolve()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
