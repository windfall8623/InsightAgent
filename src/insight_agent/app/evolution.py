"""Self-evolution: a closed loop that tunes strategy from its own outcomes.

**What problem this solves.** The orchestrator runs one fixed configuration for
every question. But questions are not alike: a single-table count needs no
retrieval and should not pay for a frontier model, while a multi-join question
with a time window wants both. A fixed policy therefore either over-spends on
easy questions or under-serves hard ones.

**The loop.** ``observe -> propose -> validate -> accept``:

1. **Observe** — run the dataset under each candidate strategy and record
   (question shape, strategy) -> outcome.
2. **Propose** — for each question shape, pick the best-measured strategy.
   Strategies with no observations are not eligible: proposing an untried
   configuration would be guessing, not learning.
3. **Validate** — score the candidate policy over the whole dataset.
4. **Accept** — only if it regresses on neither success rate nor retrieval
   recall. A change that trades correctness for cost is rejected outright.

**The honesty constraint that shapes this design.** Accuracy is already
saturated — the deterministic translator compiles correct SQL on the first try —
so there is no accuracy headroom to optimise. The loop targets the metrics that
*do* have headroom (retrieval recall, latency, cost) and refuses any candidate
that costs correctness. Reporting an accuracy improvement here would mean
inventing one.

**What this does and does not prove.** The candidate policy is scored on the
same dataset it was tuned against, so the loop can only demonstrate that a
shape-conditioned policy beats a uniform one *on this data* — not that it
generalises. The regression guard makes acceptance safe; it does not make the
result a held-out generalisation claim. `evolve()` reports both numbers and the
report says which is which.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from insight_agent.app.orchestrator import RunConfig
from insight_agent.eval.sage_gate import (
    Decision,
    GateConfig,
    evaluate_gate,
)

# --------------------------------------------------------------------------
# Question shape
# --------------------------------------------------------------------------

_DIMENSION_WORDS = ("各", "每", "按", "分", "top", "前", "breakdown", "by ", "per ")
_TIME_WORDS = ("最近", "近", "去年", "今年", "月", "日", "年", "last", "this year")
_FILTER_WORDS = ("已", "只", "限定", "只看", "的")


@dataclass(frozen=True, slots=True)
class QuestionShape:
    """Cheap structural features, computable before planning.

    Deliberately pre-semantic: the policy must choose a strategy *before* the
    planner runs, so it cannot depend on knowing the metric.
    """

    has_dimension: bool
    has_time_window: bool
    filter_count: int
    length_bucket: int

    @property
    def key(self) -> str:
        dimension = "dim" if self.has_dimension else "flat"
        window = "time" if self.has_time_window else "notime"
        filters = f"f{min(self.filter_count, 2)}"
        length = f"l{self.length_bucket}"
        return f"{dimension}-{window}-{filters}-{length}"


def classify_shape(question: str) -> QuestionShape:
    lowered = question.lower()
    return QuestionShape(
        has_dimension=any(w in lowered for w in _DIMENSION_WORDS),
        has_time_window=any(w in lowered for w in _TIME_WORDS),
        filter_count=sum(1 for w in _FILTER_WORDS if w in question),
        length_bucket=min(len(question) // 20, 2),
    )


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Strategy:
    """A named configuration the loop may choose between.

    ``default_strategies`` orders these cheapest-first so ties resolve toward
    the cheap option: a strategy that is not measurably better should not cost
    more.
    """

    name: str
    config: RunConfig
    description: str = ""


def default_strategies() -> list[Strategy]:
    return [
        Strategy(
            "cheap_flat",
            RunConfig(use_rag=False, use_routing=True, retrieval_mode="keyword"),
            "no retrieval, small model — cheapest",
        ),
        Strategy(
            "cheap_hybrid",
            RunConfig(use_rag=True, use_routing=True, retrieval_mode="hybrid"),
            "hybrid retrieval, small model",
        ),
        Strategy(
            "rich_vector",
            RunConfig(use_rag=True, use_routing=True, retrieval_mode="vector"),
            "vector retrieval, small model",
        ),
        Strategy(
            "deep_hybrid",
            RunConfig(use_rag=True, use_routing=False, retrieval_mode="hybrid"),
            "hybrid retrieval, always the large model",
        ),
    ]


def uniform_policy(strategy_name: str) -> dict[str, str]:
    """A shape-agnostic baseline policy: one strategy for every shape."""

    return {"*": strategy_name}


# --------------------------------------------------------------------------
# Outcomes and reports
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Outcome:
    """One measured run of one strategy against one question."""

    case_id: str
    shape: str
    strategy: str
    passed: bool
    recall: float
    latency_ms: float
    cost_usd: float
    # True when this run was supposed to use the real remote reranker but it
    # timed out / disconnected and silently degraded to the heuristic backend.
    # Defaulted so older persisted outcomes still load.
    rerank_degraded: bool = False
    # True when the LLM chat-completion call raised and the case was scored as a
    # failure for that reason (as opposed to a legitimate wrong-answer failure).
    llm_error: bool = False
    # True when this run was supposed to use the real remote embedding endpoint
    # but it timed out / 429'd and silently degraded to the hash embedding.
    # Defaulted so older persisted outcomes still load.
    embedding_degraded: bool = False
    # Grading audit trail. ``passed`` above is the authoritative (v2) verdict;
    # ``passed_legacy`` keeps the old exact-multiset verdict so a re-grade can
    # report how many cases the v2 grader recovered (or newly caught). The SQL
    # and both row sets are captured so any future grader change is a free,
    # offline re-grade with no model calls. All defaulted for backward-compat.
    passed_legacy: bool = False
    overlap: float = 0.0
    got_sql: str = ""
    gold_rows_json: str = ""
    agent_rows_json: str = ""
    grade_error: str = ""
    # Evaluation provenance and bounded, credential-redacted diagnostics. Empty
    # defaults deliberately identify legacy caches rather than upgrading them.
    planning_diagnostics: dict[str, Any] = field(default_factory=dict)
    execution_diagnostics: dict[str, Any] = field(default_factory=dict)
    attempt_id: str = ""
    memory_scope_version: str = ""
    evaluation_fingerprint: str = ""
    # Harness retries, distinct from the planner's within-run repair attempts.
    # History retains prior attempts (the current attempt is the Outcome itself).
    retry_count: int = 0
    attempt_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Composite objective.

        Success is weighted far above every other term so that no amount of
        speed or cost saving can buy a wrong answer. The regression guard below
        enforces the same rule as a hard constraint, independently of these
        weights.
        """

        return (
            (1000.0 if self.passed else 0.0)
            + 10.0 * self.recall
            - 0.001 * self.latency_ms
            - 500.0 * self.cost_usd
        )


@dataclass(slots=True)
class PolicyMetrics:
    """Aggregate performance of a policy over the dataset."""

    success: float = 0.0
    recall: float = 0.0
    cost: float = 0.0
    latency: float = 0.0
    score: float = 0.0
    cases: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "success": round(self.success, 4),
            "recall": round(self.recall, 4),
            "cost": round(self.cost, 6),
            "latency": round(self.latency, 3),
            "score": round(self.score, 4),
            "cases": self.cases,
        }


@dataclass(slots=True)
class EvolutionReport:
    """What one cycle changed, and whether the change survived validation."""

    cycle: int = 0
    accepted: bool = False
    changed_shapes: list[str] = field(default_factory=list)
    rejected_reasons: list[str] = field(default_factory=list)
    gate: dict[str, Any] = field(default_factory=dict)
    baseline: PolicyMetrics = field(default_factory=PolicyMetrics)
    candidate: PolicyMetrics = field(default_factory=PolicyMetrics)
    policy: dict[str, str] = field(default_factory=dict)

    @property
    def improvement_pct(self) -> float:
        if not self.baseline.score:
            return 0.0
        return (self.candidate.score - self.baseline.score) / abs(self.baseline.score) * 100

    def to_payload(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "accepted": self.accepted,
            "changed_shapes": self.changed_shapes,
            "rejected_reasons": self.rejected_reasons,
            "gate": self.gate,
            "baseline": self.baseline.to_payload(),
            "candidate": self.candidate.to_payload(),
            "improvement_pct": round(self.improvement_pct, 4),
            "policy": self.policy,
        }


Evaluator = Callable[[dict[str, str]], PolicyMetrics]

# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
# Persistence & Defaults
# --------------------------------------------------------------------------


def default_evolution_path() -> Path:
    """Default persistence path for evolution policy across CLI and API runs."""
    import os

    override = os.environ.get("INSIGHT_EVOLUTION_PATH")
    if override:
        return Path(override)
    testdata_dir = Path(".testdata")
    if testdata_dir.exists():
        return testdata_dir / "evolution_policy.json"
    return Path(".insight") / "evolution_policy.json"


class EvolutionEngine:
    """Closed-loop strategy tuner with a hard regression guard."""

    def __init__(
        self,
        strategies: Sequence[Strategy] | None = None,
        *,
        path: Path | None = None,
        success_tolerance: float = 0.0,
        auto_save_interval: int = 1,
    ) -> None:
        self.strategies = list(strategies or default_strategies())
        self.path = path
        self.success_tolerance = success_tolerance
        self.auto_save_interval = auto_save_interval
        self.outcomes: list[Outcome] = []
        self.policy: dict[str, str] = {}
        self.baseline_name: str = "cheap_hybrid"
        self.gate_config = GateConfig()
        self.baseline_outcomes: dict[str, bool] = {}
        self.candidate_outcomes: dict[str, bool] = {}
        # Per-item outcomes from the most recent evaluation, for the gate.
        self.last_outcomes: dict[str, bool] = {}
        self.cycles = 0

        # Auto-load existing policy and outcomes on startup if file exists
        if self.path and self.path.exists():
            self.load()

    # -- names ------------------------------------------------------------

    def strategy(self, name: str) -> Strategy | None:
        return next((s for s in self.strategies if s.name == name), None)

    def _order(self, name: str) -> int:
        """Position in the cheapest-first list; used only for tie-breaking."""

        for i, s in enumerate(self.strategies):
            if s.name == name:
                return i
        return len(self.strategies)

    # -- 1. observe -------------------------------------------------------

    def observe(self, outcome: Outcome) -> None:
        self.outcomes.append(outcome)
        if self.path and (len(self.outcomes) % self.auto_save_interval == 0):
            self.save()

    def maybe_auto_evolve(self, min_observations: int = 5) -> bool:
        """Autonomous closed-loop trigger: if sufficient observations have been recorded,

        proposes a policy update and adopts it ONLY IF approved by SAGE statistical gate.
        Protects runtime execution from adopting strategies that fail the McNemar test
        or introduce hidden regressions.
        """
        if len(self.outcomes) < min_observations:
            return False
        proposal = self.propose()
        if not proposal or proposal == self.policy:
            return False

        # Build per-item outcomes between current policy and proposed policy
        baseline_map: dict[str, bool] = dict(self.baseline_outcomes)
        candidate_map: dict[str, bool] = dict(self.candidate_outcomes)

        if not baseline_map or not candidate_map:
            for idx, o in enumerate(self.outcomes):
                k = f"item_{idx}"
                current_strat = self.policy.get(o.shape, self.baseline_name)
                prop_strat = proposal.get(o.shape, current_strat)
                baseline_map[k] = False
                candidate_map[k] = o.passed
                if o.strategy == current_strat:
                    baseline_map[k] = o.passed
                if o.strategy == prop_strat:
                    candidate_map[k] = o.passed

        # Evaluate through SAGE statistical acceptance gate
        verdict = evaluate_gate(baseline_map, candidate_map, self.gate_config)
        if verdict.decision is not Decision.ACCEPT:
            # Rejected or abstained by SAGE gate - do not adopt proposal
            return False

        self.policy.update(proposal)
        self.cycles += 1
        self.save()
        return True

    def _grouped(self, shape: str) -> dict[str, list[Outcome]]:
        grouped: dict[str, list[Outcome]] = {}
        for o in self.outcomes:
            if o.shape == shape:
                grouped.setdefault(o.strategy, []).append(o)
        return grouped

    # -- 2. propose -------------------------------------------------------

    def propose(self) -> dict[str, str]:
        """Choose the best-measured strategy per question shape."""

        shapes = sorted({o.shape for o in self.outcomes})
        proposal: dict[str, str] = {}
        for shape in shapes:
            grouped = self._grouped(shape)
            scored = {
                name: sum(o.score for o in group) / len(group)
                for name, group in grouped.items()
                if group
            }
            if not scored:
                continue

            baseline_group = grouped.get(self.baseline_name, [])
            base_success = (
                sum(1.0 if o.passed else 0.0 for o in baseline_group) / len(baseline_group)
                if baseline_group
                else 0.0
            )
            base_recall = (
                sum(o.recall for o in baseline_group) / len(baseline_group)
                if baseline_group
                else 0.0
            )

            eligible = {}
            for name, sc in scored.items():
                grp = grouped[name]
                s_rate = sum(1.0 if o.passed else 0.0 for o in grp) / len(grp)
                r_rate = sum(o.recall for o in grp) / len(grp)
                if s_rate >= base_success - 1e-9 and r_rate >= base_recall - 1e-9:
                    eligible[name] = sc

            candidates = eligible or scored
            # Ties break toward the cheaper (earlier-declared) strategy.
            proposal[shape] = max(
                candidates.items(), key=lambda item: (item[1], -self._order(item[0]))
            )[0]
        return proposal

    # -- 3 & 4. validate and accept ---------------------------------------

    def evolve(
        self,
        evaluator: Evaluator,
        *,
        baseline_name: str | None = None,
        per_item: tuple[dict[str, bool], dict[str, bool]] | None = None,
    ) -> EvolutionReport:
        """Run one full cycle and adopt the candidate only if the gate approves.

        ``per_item`` supplies (baseline, candidate) outcome maps for the
        statistical gate. When omitted the gate falls back to an abstain, since
        an aggregate-only comparison is exactly what it exists to replace.
        """
        """Run one full cycle and adopt the candidate only if it is safe."""

        baseline_name = baseline_name or self.baseline_name
        report = EvolutionReport(cycle=self.cycles + 1)
        report.baseline = evaluator(uniform_policy(baseline_name))

        if per_item is not None:
            self.baseline_outcomes, self.candidate_outcomes = per_item

        proposal = self.propose()
        if not proposal:
            report.policy = dict(self.policy)
            return report

        report.candidate = evaluator(proposal)
        report.policy = dict(proposal)
        report.changed_shapes = sorted(
            shape
            for shape, name in proposal.items()
            if self.strategy(name) and self.strategy(name) != self.strategy(baseline_name)
        )

        # Per-item outcomes, so a candidate that trades regressions for a
        # matching number of improvements cannot hide behind an unchanged mean.
        verdict = evaluate_gate(self.baseline_outcomes, self.candidate_outcomes, self.gate_config)
        report.gate = verdict.to_payload()

        # Recall is still checked in aggregate: the gate sees pass/fail, but a
        # regression in grounding quality is worth catching even when the
        # answers stay correct.
        if report.candidate.recall < report.baseline.recall - 1e-9:
            verdict.decision = Decision.REJECT
            verdict.reason += (
                f"; recall regressed {report.baseline.recall:.4f} -> {report.candidate.recall:.4f}"
            )

        # An empty proposal is not an improvement; report it as a rejection so
        # the cycle never claims a gain it did not measure.
        if not report.changed_shapes and verdict.decision is Decision.ACCEPT:
            verdict.decision = Decision.REJECT
            verdict.reason = "proposal identical to baseline; no change to accept"

        if verdict.decision is not Decision.ACCEPT:
            report.rejected_reasons.append(verdict.reason)
            self.cycles += 1
            return report

        self.policy = proposal
        report.accepted = True
        self.cycles += 1
        self.save()
        return report

    # -- runtime application ---------------------------------------------

    def config_for(self, question: str) -> tuple[RunConfig, str]:
        """Resolve the strategy to use for a live question.

        Falls back to the baseline when the shape is unseen, so a live system
        never runs an unvalidated configuration just because a new phrasing
        appeared.
        """

        shape = classify_shape(question).key
        name = self.policy.get(shape)
        found = self.strategy(name) if name else None
        if found is not None:
            return found.config, found.name
        found = self.strategy(self.baseline_name)
        assert found is not None
        return found.config, found.name

    # -- persistence ------------------------------------------------------

    def save(self) -> Path | None:
        if not self.path:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "policy": self.policy,
                    "baseline_name": self.baseline_name,
                    "cycles": self.cycles,
                    "outcomes": [asdict(o) for o in self.outcomes],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self.path

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.policy = data.get("policy", {})
        self.baseline_name = data.get("baseline_name", self.baseline_name)
        self.cycles = data.get("cycles", 0)
        self.outcomes = [Outcome(**o) for o in data.get("outcomes", [])]
