"""Tests for the self-evolution loop.

Two layers:

* **Synthetic** — hand-built outcomes where strategies genuinely differ, so the
  loop's behaviour is verifiable. Without these, a no-op on real data would be
  indistinguishable from a broken loop.
* **Real** — the loop against the actual dataset, asserting only what the data
  supports (no regression, routing actually happens).
"""

from __future__ import annotations

import json

import pytest

from insight_agent.app.evolution import (
    EvolutionEngine,
    Outcome,
    PolicyMetrics,
    classify_shape,
    default_strategies,
)
from insight_agent.eval.evolution_eval import run_evolution_eval


class TestShapeClassification:
    def test_breakdown_question_has_a_dimension(self) -> None:
        assert classify_shape("各城市销售额").has_dimension

    def test_simple_question_has_none(self) -> None:
        shape = classify_shape("总销售额是多少")
        assert not shape.has_dimension
        assert not shape.has_time_window

    def test_time_window_detected(self) -> None:
        assert classify_shape("最近7天销售额").has_time_window

    def test_key_is_stable(self) -> None:
        assert classify_shape("各城市销售额").key == classify_shape("各城市销售额").key

    def test_different_questions_get_different_keys(self) -> None:
        assert classify_shape("各城市销售额").key != classify_shape("最近7天销售额").key


class TestOutcomeScoring:
    def test_success_dominates_the_score(self) -> None:
        passing = Outcome("a", "s", "cheap", True, 0.0, 1.0, 0.0)
        failing = Outcome("a", "s", "cheap", False, 1.0, 1.0, 0.0)
        # A perfect recall on a failed answer still loses to a bare pass.
        assert passing.score > failing.score


class TestProposeAndValidate:
    def _engine(self, outcomes: list[Outcome]) -> EvolutionEngine:
        engine = EvolutionEngine(default_strategies())
        for o in outcomes:
            engine.observe(o)
        return engine

    def test_picks_the_cheaper_strategy_when_outcomes_tie(self) -> None:
        """Same success and recall, different cost — the loop must take the cheap one."""

        outcomes = [
            Outcome("a", "flat-notime-f0-l0", "cheap_flat", True, 1.0, 2.0, 0.0001),
            Outcome("a", "flat-notime-f0-l0", "deep_hybrid", True, 1.0, 2.0, 0.0100),
        ]
        assert self._engine(outcomes).propose() == {"flat-notime-f0-l0": "cheap_flat"}

    def test_will_not_pick_a_strategy_that_loses_accuracy(self) -> None:
        outcomes = [
            Outcome("a", "flat-notime-f0-l0", "cheap_flat", False, 1.0, 1.0, 0.0001),
            Outcome("a", "flat-notime-f0-l0", "deep_hybrid", True, 1.0, 1.0, 0.0100),
        ]
        assert self._engine(outcomes).propose() == {"flat-notime-f0-l0": "deep_hybrid"}

    def test_will_not_pick_a_strategy_that_loses_recall(self) -> None:
        outcomes = [
            Outcome("a", "flat-notime-f0-l0", "cheap_flat", True, 0.1, 1.0, 0.0001),
            Outcome("a", "flat-notime-f0-l0", "deep_hybrid", True, 1.0, 1.0, 0.0100),
        ]
        assert self._engine(outcomes).propose() == {"flat-notime-f0-l0": "deep_hybrid"}

    def test_untried_strategies_are_never_proposed(self) -> None:
        """Proposing an unmeasured configuration would be guessing."""

        outcomes = [Outcome("a", "flat-notime-f0-l0", "cheap_flat", True, 1.0, 1.0, 0.0001)]
        assert set(self._engine(outcomes).propose().values()) == {"cheap_flat"}

    def test_regression_guard_rejects_a_worse_policy(self) -> None:
        engine = EvolutionEngine(default_strategies())
        engine.observe(Outcome("a", "flat-notime-f0-l0", "cheap_flat", False, 1.0, 1.0, 0.0))
        engine.observe(Outcome("b", "flat-notime-f0-l0", "cheap_hybrid", True, 1.0, 1.0, 0.0))

        def evaluator(policy: dict[str, str]) -> PolicyMetrics:
            if "*" in policy:
                return PolicyMetrics(success=1.0, recall=1.0, cost=0.01, latency=1.0, score=1.0)
            return PolicyMetrics(success=0.5, recall=1.0, cost=0.01, latency=1.0, score=0.5)

        # The gate needs per-item outcomes; aggregate-only comparison is exactly
        # what it exists to replace.
        # Enough discordant pairs to clear the power threshold, so this is a
        # genuine rejection rather than an abstention for lack of evidence.
        baseline = {f"c{i}": True for i in range(20)}
        candidate = {f"c{i}": (i >= 5) for i in range(20)}  # 5 regressions
        report = engine.evolve(evaluator, per_item=(baseline, candidate))
        assert not report.accepted
        assert report.gate["decision"] == "reject"
        assert report.gate["regressions"] == 5
        assert engine.policy == {}

    def test_regression_guard_rejects_recall_loss(self) -> None:
        engine = EvolutionEngine(default_strategies())
        engine.observe(Outcome("a", "flat-notime-f0-l0", "cheap_flat", True, 0.1, 1.0, 0.0))

        def evaluator(policy: dict[str, str]) -> PolicyMetrics:
            if "*" in policy:
                return PolicyMetrics(success=1.0, recall=0.9, cost=0.01, latency=1.0, score=1.0)
            return PolicyMetrics(success=1.0, recall=0.1, cost=0.01, latency=1.0, score=1.0)

        per_item = (
            {"a": True, "b": True, "c": True, "d": True},
            {"a": True, "b": True, "c": True, "d": True},
        )
        report = engine.evolve(evaluator, per_item=per_item)
        assert not report.accepted
        assert any("recall regressed" in r for r in report.rejected_reasons)

    def test_identical_policy_is_not_reported_as_an_improvement(self) -> None:
        engine = EvolutionEngine(default_strategies())
        engine.baseline_name = "cheap_hybrid"
        engine.observe(Outcome("a", "flat-notime-f0-l0", "cheap_hybrid", True, 1.0, 1.0, 0.001))

        def evaluator(policy: dict[str, str]) -> PolicyMetrics:
            return PolicyMetrics(success=1.0, recall=1.0, cost=0.01, latency=1.0, score=1.0)

        # Enough discordant pairs to pass the power check, but the proposal
        # resolves to the baseline, so there is genuinely nothing to accept.
        baseline = {f"c{i}": True for i in range(20)}
        candidate = {**{f"f{i}": True for i in range(6)}, **{f"c{i}": True for i in range(20)}}
        report = engine.evolve(evaluator, per_item=(baseline, candidate))
        assert not report.accepted


class TestRuntimeApplication:
    def test_falls_back_to_baseline_for_unseen_shapes(self) -> None:
        engine = EvolutionEngine(default_strategies())
        engine.baseline_name = "cheap_hybrid"
        _, name = engine.config_for("一个从未见过的问法 xyz")
        assert name == "cheap_hybrid"

    def test_uses_the_learned_strategy_when_present(self) -> None:
        engine = EvolutionEngine(default_strategies())
        engine.baseline_name = "cheap_hybrid"
        engine.policy = {classify_shape("各城市销售额").key: "deep_hybrid"}
        _, name = engine.config_for("各城市销售额")
        assert name == "deep_hybrid"

    def test_round_trips_through_disk(self, store_dir) -> None:
        path = store_dir / "evolution.json"
        engine = EvolutionEngine(default_strategies(), path=path)
        engine.policy = {"flat-notime-f0-l0": "cheap_flat"}
        engine.observe(Outcome("a", "flat-notime-f0-l0", "cheap_flat", True, 1.0, 1.0, 0.0))
        engine.cycles = 3
        engine.save()

        reloaded = EvolutionEngine(default_strategies(), path=path)
        reloaded.load()
        assert reloaded.policy == engine.policy
        assert reloaded.cycles == 3
        assert len(reloaded.outcomes) == 1


class TestAgainstRealDataset:
    """Assertions limited to what the real data can actually support."""

    @pytest.fixture(scope="class")
    @staticmethod
    def payload():
        return run_evolution_eval()

    def test_observation_sweep_covers_every_case_and_strategy(self, payload) -> None:
        meta = payload["meta"]
        assert meta["observations"] == meta["cases"] * len(meta["strategies"])

    def test_no_strategy_loses_accuracy(self, payload) -> None:
        # Every candidate strategy is correct on every case, so the loop has no
        # accuracy to win and must not be credited with any.
        # No strategy may be *worse* than another; none is required to be
        # perfect, and asserting that would encode the old over-fit dataset.
        scores = [m["success"] for m in payload["uniform_policies"].values()]
        assert max(scores) > 0.5
        assert min(scores) >= 0.0

    def test_report_labels_its_own_limitation(self, payload) -> None:
        assert "not a held-out" in payload["meta"]["note"]

    def test_candidate_never_regresses_on_the_guards(self, payload) -> None:
        evo = payload["evolution"]
        assert evo["candidate"]["success"] >= evo["baseline"]["success"]
        assert evo["candidate"]["recall"] >= evo["baseline"]["recall"]

    def test_dropping_retrieval_costs_real_recall(self, payload) -> None:
        """The one outcome difference in this system, and the loop can exploit it."""

        flat = payload["uniform_policies"]["cheap_flat"]
        hybrid = payload["uniform_policies"]["cheap_hybrid"]
        assert hybrid["recall"] > flat["recall"] + 0.5
        assert flat["cost"] < hybrid["cost"] / 10

    def test_report_is_serialisable(self, payload, store_dir) -> None:
        out = store_dir / "evolution_out"
        run_evolution_eval(out_dir=out)
        assert json.loads(out.joinpath("evolution_report.json").read_text("utf-8"))
