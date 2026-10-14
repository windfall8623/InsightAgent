"""Dual-Loop Evaluation Harness & Benchmark Runner.

Executes both Inner-Loop (Deterministic CI/CD Gate) and Outer-Loop (Ragas-Style Narrative
& Unstructured Evaluator) across production CN-Retail scenarios.

Outputs:
- Console report with Dual-Loop scorecard.
- Persisted benchmark JSON report in `docs/outer_loop_ragas_benchmark_report.json`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from insight_agent.app.graph import InsightGraphEngine
from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
from insight_agent.app.redis_store import RedisMemoryStore
from insight_agent.data.cn_retail_db import connect_cn_retail, ensure_cn_retail_database
from insight_agent.eval.anomaly_spec import STANDARD_ANOMALY_SPECS
from insight_agent.eval.outer_loop_ragas import (
    OuterLoopCaseResult,
    OuterLoopRagasEvaluator,
)

logger = logging.getLogger(__name__)


def run_dual_loop_benchmark() -> dict[str, Any]:
    print("=" * 80)
    print("  InsightAgent 2.0 Dual-Loop (Inner Gate + Outer Ragas) Evaluation Benchmark")
    print("=" * 80)

    # 1. Initialize Engines
    ensure_cn_retail_database()
    conn = connect_cn_retail()
    retriever = OlistMultiModalRetriever(conn)
    redis_store = RedisMemoryStore()
    engine = InsightGraphEngine(conn, retriever, redis_store)

    ragas_eval = OuterLoopRagasEvaluator()

    outer_results: list[OuterLoopCaseResult] = []
    inner_passes = 0
    total_evaluated = 0

    started = time.perf_counter()

    # Collect diagnostic and analytical benchmark queries
    test_specs = [s for s in STANDARD_ANOMALY_SPECS if not s.is_control][:8]

    for spec in test_specs:
        for q in spec.diagnostic_queries[:2]:
            total_evaluated += 1
            print(f"\n[Case #{total_evaluated}] Evaluating: '{q}' (Spec: {spec.id})")

            # Execute Engine
            state = engine.diagnose(q, user_id="dual_loop_eval_user")
            final_report = str(state.get("final_report", ""))
            sql_findings = state.get("sql_findings", [])
            review_evidence = state.get("review_evidence", [])
            candidate_actions = state.get("candidate_actions", [])

            # Extract rows & reviews for evaluation
            all_rows: list[dict[str, Any]] = []
            for sf in sql_findings:
                if isinstance(sf, dict):
                    rows_list = sf.get("data") or sf.get("rows") or []
                    for r in rows_list:
                        if isinstance(r, dict):
                            all_rows.append(r)

            retrieved_texts: list[str] = []
            for rev in review_evidence:
                if isinstance(rev, dict):
                    txt = str(
                        rev.get("snippet")
                        or rev.get("message")
                        or rev.get("comment_message")
                        or rev.get("title")
                        or ""
                    )
                    if txt:
                        retrieved_texts.append(txt)
                elif isinstance(rev, str):
                    retrieved_texts.append(rev)

            action_str = candidate_actions[0].get("action") if candidate_actions else None

            # --- Inner Loop Check (Deterministic correctness: did engine produce evidence & findings?) ---
            inner_ok = bool(sql_findings and final_report.strip())
            if inner_ok:
                inner_passes += 1

            # --- Outer Loop Evaluation (Ragas-style Narrative & Context Quality) ---
            keywords = [spec.name, spec.category, spec.expect_entity] + list(spec.entity_aliases)
            case_eval = ragas_eval.evaluate_case(
                case_id=f"{spec.id}_{total_evaluated}",
                query=q,
                report_text=final_report,
                sql_rows=all_rows,
                retrieved_contexts=retrieved_texts,
                evidence=sql_findings,
                actions=candidate_actions,
                remediation_action=action_str,
                ground_truth_keywords=keywords,
            )

            outer_results.append(case_eval)

            print(
                f"  -> Inner Gate: {'PASS' if inner_ok else 'FAIL'} | "
                f"Outer Ragas Score: {case_eval.composite_ragas_score:.3f} "
                f"(Narrative: {case_eval.narrative_faithfulness.score:.2f}, "
                f"Precision: {case_eval.unstructured_context.context_precision:.2f}, "
                f"Relevance: {case_eval.answer_relevance.score:.2f})"
            )

    elapsed_s = time.perf_counter() - started
    report = ragas_eval.aggregate(outer_results)
    inner_pass_rate = inner_passes / total_evaluated if total_evaluated else 0.0

    print("\n" + "=" * 80)
    print("Dual-Loop Evaluation Benchmark Summary:")
    print("=" * 80)
    print(f"Total Evaluated Cases                : {total_evaluated}")
    print(f"Total Benchmark Time                 : {elapsed_s:.2f}s ({elapsed_s/total_evaluated*1000:.1f} ms/query)")
    print(f"Inner-Loop Gate Pass Rate (CI/CD)    : {inner_passes}/{total_evaluated} ({inner_pass_rate * 100:.1f}%)")
    print("-" * 80)
    print("Outer-Loop (Ragas-Style) Quality Metrics:")
    print(f"  -> Composite Ragas Score           : {report.mean_composite_score * 100:.2f} / 100")
    print(f"  -> Narrative Faithfulness (No Halo): {report.mean_narrative_faithfulness * 100:.2f}%")
    print(f"  -> Numeric Groundedness Ratio      : {report.mean_numeric_grounded_ratio * 100:.2f}%")
    print(f"  -> Unstructured Context Precision  : {report.mean_context_precision * 100:.2f}%")
    print(f"  -> Unstructured Context Relevance  : {report.mean_context_relevance * 100:.2f}%")
    print(f"  -> Answer & Action Relevance       : {report.mean_answer_relevance * 100:.2f}%")
    print(f"  -> Secondary Hallucination Rate    : {report.secondary_hallucination_rate * 100:.2f}% (Target: < 5%)")
    print("=" * 80)

    # Persist Report to docs
    out_payload = {
        "benchmark": "InsightAgent 2.0 Dual-Loop & Outer-Loop Ragas Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_cases": total_evaluated,
            "latency_ms_per_case": round(elapsed_s / total_evaluated * 1000, 2),
            "inner_loop_gate_pass_rate": round(inner_pass_rate, 4),
            "outer_loop_composite_ragas_score": round(report.mean_composite_score, 4),
            "narrative_faithfulness": round(report.mean_narrative_faithfulness, 4),
            "numeric_groundedness": round(report.mean_numeric_grounded_ratio, 4),
            "context_precision": round(report.mean_context_precision, 4),
            "context_relevance": round(report.mean_context_relevance, 4),
            "answer_relevance": round(report.mean_answer_relevance, 4),
            "secondary_hallucination_rate": round(report.secondary_hallucination_rate, 4),
        },
        "report": report.to_payload(),
    }

    report_path = Path("docs/outer_loop_ragas_benchmark_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Outer-Loop Ragas Benchmark report saved to: {report_path.resolve()}\n")

    return out_payload


if __name__ == "__main__":
    run_dual_loop_benchmark()
