"""Independent Attribution & Causal Diagnosis Evaluator.

Decoupled evaluator for causal diagnosis engines (e.g. LangGraph InsightGraphEngine).
Evaluates based on observable outputs (dimensions, confirmed entities, false positives)
against declarative AnomalySpec definitions without depending on internal graph state.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from insight_agent.eval.anomaly_spec import STANDARD_ANOMALY_SPECS, AnomalySpec

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CaseEvalResult:
    """Outcome for one evaluation query."""

    spec_id: str
    query: str
    is_control: bool
    is_confirmed: bool  # Engine claimed an anomaly was found
    hit_category: bool  # Matched expected category
    hit_entity: bool  # Output referenced expected entity/alias
    passed: bool  # True attribution hit (for anomaly) or True negative (for control)
    is_false_positive: bool  # Control falsely confirmed as anomaly
    reflections: int
    detected_root_cause: str
    failure_reason: str = ""


@dataclass(slots=True)
class AttributionMetrics:
    """Aggregated attribution performance metrics."""

    total_cases: int = 0
    anomaly_cases: int = 0
    control_cases: int = 0

    # Primary quality metrics
    attribution_accuracy: float = 0.0  # Accuracy on genuine anomaly cases (Hit@1)
    false_positive_rate: float = 0.0  # Proportion of control cases falsely confirmed
    avg_reflections: float = 0.0  # Average hypothesis test cycles

    # Breakdowns
    category_accuracies: dict[str, float] = field(default_factory=dict)
    spec_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "anomaly_cases": self.anomaly_cases,
            "control_cases": self.control_cases,
            "attribution_accuracy": round(self.attribution_accuracy, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "avg_reflections": round(self.avg_reflections, 2),
            "category_accuracies": {k: round(v, 4) for k, v in self.category_accuracies.items()},
            "spec_results": self.spec_results,
        }


class AttributionEvaluator:
    """Evaluates causal attribution accuracy and false positive rates."""

    def __init__(
        self,
        specs: tuple[AnomalySpec, ...] = STANDARD_ANOMALY_SPECS,
    ) -> None:
        self.specs = specs

    def evaluate_case(
        self,
        spec: AnomalySpec,
        query: str,
        state: dict[str, Any],
    ) -> CaseEvalResult:
        """Score one diagnostic execution against AnomalySpec."""
        root_cause = str(state.get("root_cause") or "")
        final_report = str(state.get("final_report") or "")
        combined_text = f"{root_cause} {final_report}"
        reflections = int(state.get("reflections", 0))

        hyps = state.get("hypotheses", [])
        confirmed_hyps = [h for h in hyps if h.get("status") == "confirmed"]

        # An anomaly is considered "confirmed" if at least one hypothesis was confirmed
        # and root cause does not explicitly state it is within normal baseline.
        is_normal_statement = any(
            phrase in root_cause
            for phrase in ("处于正常历史基线范围内", "未见异常", "符合正常预期", "无系统性异常", "未见系统性异常")
        )
        is_confirmed = bool(confirmed_hyps and not is_normal_statement and root_cause.strip())

        # Check entity hit
        aliases = list(spec.entity_aliases)
        if spec.expect_entity:
            aliases.append(spec.expect_entity)
        hit_entity = any(alias.lower() in combined_text.lower() for alias in aliases if alias)

        # Check category hit
        hit_category = any(
            h.get("category") == spec.category for h in confirmed_hyps
        )

        if spec.is_control:
            # For control/no_anomaly scenarios:
            # Success means the engine did NOT falsely confirm an anomaly.
            is_fp = is_confirmed
            passed = not is_fp
            fail_reason = f"False positive: falsely confirmed anomaly '{root_cause[:40]}...'" if is_fp else ""
            return CaseEvalResult(
                spec_id=spec.id,
                query=query,
                is_control=True,
                is_confirmed=is_confirmed,
                hit_category=False,
                hit_entity=hit_entity,
                passed=passed,
                is_false_positive=is_fp,
                reflections=reflections,
                detected_root_cause=root_cause,
                failure_reason=fail_reason,
            )

        # For genuine anomaly scenarios:
        # Success requires confirming the anomaly, matching category, and identifying the correct entity
        passed = is_confirmed and (hit_entity or hit_category)
        fail_reasons = []
        if not is_confirmed:
            fail_reasons.append("Anomaly not confirmed")
        if not hit_category:
            fail_reasons.append(f"Category mismatch (expected '{spec.category}')")
        if not hit_entity:
            fail_reasons.append(f"Entity missing (expected '{spec.expect_entity}')")

        return CaseEvalResult(
            spec_id=spec.id,
            query=query,
            is_control=False,
            is_confirmed=is_confirmed,
            hit_category=hit_category,
            hit_entity=hit_entity,
            passed=passed,
            is_false_positive=False,
            reflections=reflections,
            detected_root_cause=root_cause,
            failure_reason="; ".join(fail_reasons),
        )

    def run_evaluation(
        self,
        diagnose_fn: Callable[[str], dict[str, Any]],
    ) -> tuple[AttributionMetrics, list[CaseEvalResult]]:
        """Run all test queries across all specs against diagnose_fn."""
        results: list[CaseEvalResult] = []
        spec_summary: dict[str, dict[str, Any]] = {}
        category_counts: dict[str, int] = {}
        category_hits: dict[str, int] = {}

        for spec in self.specs:
            spec_passed = 0
            spec_total = len(spec.diagnostic_queries)
            for query in spec.diagnostic_queries:
                state = diagnose_fn(query)
                res = self.evaluate_case(spec, query, state)
                results.append(res)
                if res.passed:
                    spec_passed += 1

                if not spec.is_control:
                    category_counts[spec.category] = category_counts.get(spec.category, 0) + 1
                    if res.passed:
                        category_hits[spec.category] = category_hits.get(spec.category, 0) + 1

            spec_summary[spec.id] = {
                "name": spec.name,
                "category": spec.category,
                "is_control": spec.is_control,
                "passed": spec_passed,
                "total": spec_total,
                "pass_rate": round(spec_passed / max(spec_total, 1), 4),
            }

        anomaly_results = [r for r in results if not r.is_control]
        control_results = [r for r in results if r.is_control]

        accuracy = (
            sum(1 for r in anomaly_results if r.passed) / max(len(anomaly_results), 1)
        )
        fpr = (
            sum(1 for r in control_results if r.is_false_positive)
            / max(len(control_results), 1)
        )
        avg_reflections = (
            sum(r.reflections for r in results) / max(len(results), 1)
        )

        cat_accs = {
            cat: category_hits.get(cat, 0) / max(category_counts[cat], 1)
            for cat in category_counts
        }

        metrics = AttributionMetrics(
            total_cases=len(results),
            anomaly_cases=len(anomaly_results),
            control_cases=len(control_results),
            attribution_accuracy=accuracy,
            false_positive_rate=fpr,
            avg_reflections=avg_reflections,
            category_accuracies=cat_accs,
            spec_results=spec_summary,
        )

        return metrics, results


def save_attribution_report(
    metrics: AttributionMetrics,
    results: list[CaseEvalResult],
    output_path: Path | str = "docs/attribution_benchmark_baseline_report.json",
) -> Path:
    """Save full attribution evaluation report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics.to_payload(),
        "case_details": [
            {
                "spec_id": r.spec_id,
                "query": r.query,
                "is_control": r.is_control,
                "passed": r.passed,
                "is_false_positive": r.is_false_positive,
                "hit_category": r.hit_category,
                "hit_entity": r.hit_entity,
                "reflections": r.reflections,
                "detected_root_cause": r.detected_root_cause,
                "failure_reason": r.failure_reason,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
