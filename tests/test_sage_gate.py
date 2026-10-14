"""Tests for the SAGE-style acceptance gate, the dataset split, and dynamics."""

from __future__ import annotations

import pytest

from insight_agent.eval.sage_gate import (
    Decision,
    GateConfig,
    analyse_dynamics,
    compare_per_item,
    evaluate_gate,
    mcnemar_exact_p,
)


def _pool(n_pass: int = 20, n_fail: int = 20) -> dict[str, bool]:
    pool = {f"c{i}": True for i in range(n_pass)}
    pool.update({f"f{i}": False for i in range(n_fail)})
    return pool


class TestMcNemar:
    def test_no_discordant_pairs_is_not_significant(self) -> None:
        assert mcnemar_exact_p(0, 0) == 1.0

    def test_one_each_side_is_not_significant(self) -> None:
        assert mcnemar_exact_p(1, 1) == 1.0

    def test_large_one_sided_improvement_is_significant(self) -> None:
        assert mcnemar_exact_p(10, 0) < 0.05

    def test_p_value_is_symmetric(self) -> None:
        assert mcnemar_exact_p(8, 2) == mcnemar_exact_p(2, 8)

    def test_p_value_is_bounded(self) -> None:
        for b in range(0, 12):
            for w in range(0, 12):
                assert 0.0 <= mcnemar_exact_p(b, w) <= 1.0


class TestPerItemComparison:
    def test_identifies_each_direction(self) -> None:
        baseline = {"a": True, "b": False, "c": True, "d": False}
        candidate = {"a": False, "b": True, "c": True, "d": False}
        improved, regressed, unchanged = compare_per_item(baseline, candidate)
        assert improved == ["b"]
        assert regressed == ["a"]
        assert unchanged == ["c", "d"]

    def test_only_shared_items_are_compared(self) -> None:
        improved, regressed, unchanged = compare_per_item({"a": False}, {"a": True, "z": True})
        assert improved == ["a"]
        assert "z" not in unchanged


class TestGateDecisions:
    def test_accepts_a_clear_improvement(self) -> None:
        baseline = _pool()
        candidate = {**baseline, **{f"f{i}": True for i in range(6)}}
        verdict = evaluate_gate(baseline, candidate)
        assert verdict.decision is Decision.ACCEPT
        assert verdict.regressions == 0
        assert verdict.p_value < 0.05

    def test_rejects_when_regressions_outweigh_improvements(self) -> None:
        """The case a purely aggregate guard cannot see."""

        baseline = _pool()
        candidate = dict(baseline)
        for i in range(3):
            candidate[f"c{i}"] = False
        for i in range(2):
            candidate[f"f{i}"] = True
        verdict = evaluate_gate(baseline, candidate)
        assert verdict.decision is Decision.REJECT
        assert verdict.net < 0
        assert "c0" in verdict.regressions_by_id

    def test_rejects_even_when_the_mean_rises(self) -> None:
        """Many small fixes must not license breaking a previously-correct item."""

        baseline = _pool()
        candidate = dict(baseline)
        for i in range(12):
            candidate[f"f{i}"] = True
        candidate["c0"] = False
        verdict = evaluate_gate(baseline, candidate)
        assert verdict.decision is Decision.REJECT

    def test_abstains_when_underpowered(self) -> None:
        baseline = _pool()
        verdict = evaluate_gate(baseline, {**baseline, "f0": True})
        assert verdict.decision is Decision.ABSTAIN
        assert "discordant" in verdict.reason

    def test_abstains_when_identical(self) -> None:
        baseline = _pool()
        verdict = evaluate_gate(baseline, dict(baseline))
        assert verdict.decision is Decision.ABSTAIN
        assert verdict.improvements == verdict.regressions == 0

    def test_abstains_without_overlap(self) -> None:
        verdict = evaluate_gate({"a": True}, {"z": False})
        assert verdict.decision is Decision.ABSTAIN
        assert "overlapping" in verdict.reason

    def test_asymmetric_penalty_is_configurable(self) -> None:
        baseline = _pool()
        candidate = dict(baseline)
        candidate["c0"] = False
        for i in range(2):
            candidate[f"f{i}"] = True
        lenient = evaluate_gate(
            baseline, candidate, GateConfig(regression_penalty=1.0, min_discordant=1)
        )
        strict = evaluate_gate(
            baseline, candidate, GateConfig(regression_penalty=5.0, min_discordant=1)
        )
        assert lenient.net == 2 - 1
        assert strict.net == 2 - 5

    def test_verdict_payload_is_serialisable(self) -> None:
        import json

        baseline = _pool()
        verdict = evaluate_gate(baseline, {**baseline, **{f"f{i}": True for i in range(6)}})
        json.dumps(verdict.to_payload())


class TestImprovementDynamics:
    def test_counts_directions(self) -> None:
        dynamics = analyse_dynamics([0.5, 0.6, 0.55, 0.7])
        assert dynamics.steps == 3
        assert dynamics.improvements == 2
        assert dynamics.regressions == 1

    def test_regression_magnitude_is_positive(self) -> None:
        assert analyse_dynamics([0.5, 0.4]).mean_regression == pytest.approx(0.1)

    def test_flat_sequence_has_no_improvements(self) -> None:
        dynamics = analyse_dynamics([0.5, 0.5, 0.5])
        assert dynamics.improvements == 0
        assert dynamics.improvement_rate == 0.0

    def test_decay_is_detected(self) -> None:
        dynamics = analyse_dynamics([0.50, 0.62, 0.68, 0.71, 0.72, 0.725])
        assert dynamics.decays is True
        assert dynamics.half_life and dynamics.half_life > 0

    def test_decay_is_not_claimed_from_too_few_points(self) -> None:
        assert analyse_dynamics([0.5, 0.6, 0.65]).decays is None

    def test_single_step_is_safe(self) -> None:
        assert analyse_dynamics([0.5]).steps == 0

    def test_empty_is_safe(self) -> None:
        assert analyse_dynamics([]).improvement_rate == 0.0

    def test_payload_is_serialisable(self) -> None:
        import json

        json.dumps(analyse_dynamics([0.5, 0.6, 0.55]).to_payload())


class TestSplit:
    def test_split_partitions_every_case_exactly_once(self) -> None:
        from insight_agent.eval.dataset import load_cases
        from insight_agent.eval.split import SPLITS, build_split

        split = build_split()
        assigned = [i for name in SPLITS for i in split.ids(name)]
        assert len(assigned) == len(set(assigned))
        # The default split sizes sum to fewer cases than the dataset now holds,
        # so it covers a subset — every covered id must be unique and real.
        assert set(assigned) <= {c["id"] for c in load_cases()}

    def test_holdout_split_excludes_the_development_cases(self) -> None:
        """The 51 hand-written cases were used to tune the rules engine.

        A split containing them would report training accuracy as if it were
        generalisation, which is the mistake this whole exercise corrects.
        """

        from insight_agent.eval.dataset import load_cases
        from insight_agent.eval.split import holdout_split

        split = holdout_split()
        assert set(split.train + split.dev + split.test) == {
            c["id"] for c in load_cases() if c.get("source") == "generated"
        }

    def test_split_is_deterministic(self) -> None:
        from insight_agent.eval.split import build_split

        assert build_split().to_payload() == build_split().to_payload()

    def test_default_sizes_are_respected(self) -> None:
        from insight_agent.eval.split import build_split

        split = build_split()
        assert len(split.train) == 30
        assert len(split.dev) == 11
        assert len(split.test) == 10

    def test_unknown_split_name_raises(self) -> None:
        from insight_agent.eval.split import build_split

        with pytest.raises(ValueError):
            build_split().ids("validation")

    def test_dataset_hash_is_stable(self) -> None:
        from insight_agent.eval.split import dataset_hash

        assert dataset_hash() == dataset_hash()
        assert len(dataset_hash()) == 16

    def test_freeze_verifies_against_the_current_dataset(self) -> None:
        from insight_agent.eval.split import build_split, verify_frozen

        split = build_split()
        ok, message = verify_frozen(split.dataset_hash)
        assert ok
        assert "unchanged" in message

    def test_a_changed_dataset_invalidates_the_freeze(self) -> None:
        from insight_agent.eval.split import verify_frozen

        ok, message = verify_frozen("deadbeefdeadbeef")
        assert not ok
        assert "invalidates" in message

    def test_manifest_round_trips(self, store_dir) -> None:
        from insight_agent.eval.split import build_split, load_manifest, save_manifest

        split = build_split()
        save_manifest(split, store_dir)
        assert load_manifest(store_dir).to_payload() == split.to_payload()

    def test_loading_cases_for_a_split_filters_correctly(self) -> None:
        from insight_agent.eval.split import build_split, load_cases_for

        split = build_split()
        cases = load_cases_for(split, "test")
        assert len(cases) == 10
        assert {c["id"] for c in cases} == set(split.test)


class TestEvolutionUsesTheGate:
    def test_engine_carries_a_gate_config(self) -> None:
        from insight_agent.app.evolution import EvolutionEngine

        assert isinstance(EvolutionEngine().gate_config, GateConfig)

    def test_report_carries_a_gate_payload(self) -> None:
        from insight_agent.app.evolution import EvolutionEngine, EvolutionReport

        assert "gate" in EvolutionReport().to_payload()
        assert EvolutionEngine() is not None
