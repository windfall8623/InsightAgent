"""SAGE-style statistical acceptance gate for self-evolving agents.

Adapts the method described in *"SAGE: A Statistical Acceptance Gate for
Self-Evolving Agents"* (arXiv, 2026-09-28), which identifies two failure modes
that a naive guard — including the one this project originally shipped — has:

1. **Aggregate blindness.** Accepting an edit because the overall success rate
   rose can silently break items that previously passed. Two cases fixed, three
   broken, score "unchanged" — and the guard approves.
2. **The optimizer's curse.** Picking the best of several noisy candidates
   selects for luck, so the accepted change may not survive re-measurement.

The method here:

* compare outcomes **per item** on identical examples, not as aggregate means
* penalise regressions **asymmetrically** — breaking something costs more than
  fixing something earns
* require the improvement to be **statistically reliable** before accepting
* **abstain** when there is not enough evidence either way, instead of
  reporting "no change" as if it were a measurement

The reliability test is an exact McNemar test on the discordant pairs, which is
the right test here: the two policies are evaluated on the same items, so only
the items where they disagree carry information.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Decision(StrEnum):
    """The gate's three-way verdict."""

    ACCEPT = "accept"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass(slots=True)
class GateConfig:
    """Thresholds that make the gate's conservatism explicit."""

    # Breaking an item must cost this many times more than fixing one.
    regression_penalty: float = 2.0
    # Below this many discordant pairs, the comparison is underpowered.
    min_discordant: int = 5
    # Two-sided significance required to accept.
    alpha: float = 0.05
    # A net score at or below this is not worth the churn.
    min_net_improvement: int = 1


@dataclass(slots=True)
class GateVerdict:
    """What the gate decided, and on what evidence."""

    decision: Decision
    improvements: int = 0
    regressions: int = 0
    unchanged: int = 0
    net: int = 0
    p_value: float = 1.0
    reason: str = ""
    regressions_by_id: dict[str, str] = field(default_factory=dict)
    improvements_by_id: dict[str, str] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "improvements": self.improvements,
            "regressions": self.regressions,
            "unchanged": self.unchanged,
            "net": self.net,
            "p_value": round(self.p_value, 6),
            "reason": self.reason,
            "regressions_by_id": self.regressions_by_id,
            "improvements_by_id": self.improvements_by_id,
        }


def mcnemar_exact_p(better: int, worse: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs.

    Under the null that the two policies are equivalent, each discordant item is
    a coin flip, so the count that improved is Binomial(n, 0.5) with n =
    better + worse. Both tails are summed, which is why the binomial
    coefficient is evaluated from both ends rather than doubled.
    """

    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail: float = float(sum(math.comb(n, i) for i in range(0, k + 1)))
    p_value = 2.0 * tail / float(2**n)
    return min(1.0, p_value)


def compare_per_item(
    baseline: dict[str, bool],
    candidate: dict[str, bool],
) -> tuple[list[str], list[str], list[str]]:
    """Return (improved, regressed, unchanged) case ids, comparing per item."""

    shared = sorted(set(baseline) & set(candidate))
    improved = [k for k in shared if not baseline[k] and candidate[k]]
    regressed = [k for k in shared if baseline[k] and not candidate[k]]
    unchanged = [k for k in shared if baseline[k] == candidate[k]]
    return improved, regressed, unchanged


def evaluate_gate(
    baseline: dict[str, bool],
    candidate: dict[str, bool],
    config: GateConfig | None = None,
) -> GateVerdict:
    """Decide whether a candidate policy should replace the baseline."""

    config = config or GateConfig()

    if not set(baseline) & set(candidate):
        return GateVerdict(Decision.ABSTAIN, reason="no overlapping items to compare")

    improved, regressed, unchanged = compare_per_item(baseline, candidate)
    net = len(improved) - int(config.regression_penalty * len(regressed))
    p_value = mcnemar_exact_p(len(improved), len(regressed))

    verdict = GateVerdict(
        decision=Decision.ABSTAIN,
        improvements=len(improved),
        regressions=len(regressed),
        unchanged=len(unchanged),
        net=net,
        p_value=p_value,
        regressions_by_id={k: "baseline passed, candidate failed" for k in regressed},
        improvements_by_id={k: "baseline failed, candidate passed" for k in improved},
    )

    # Underpowered: abstaining is the honest answer, not "no change".
    if len(improved) + len(regressed) < config.min_discordant:
        verdict.reason = (
            f"only {len(improved) + len(regressed)} discordant item(s), "
            f"below the {config.min_discordant} needed to distinguish signal from noise"
        )
        return verdict

    if regressed:
        verdict.decision = Decision.REJECT
        verdict.reason = (
            f"{len(regressed)} regression(s) "
            f"({', '.join(regressed[:5])}) outweigh the "
            f"{len(improved)} improvement(s) at penalty "
            f"{config.regression_penalty:g}"
        )
        return verdict

    if len(improved) < config.min_net_improvement:
        verdict.reason = "no item improved; nothing to accept"
        return verdict

    if p_value > config.alpha:
        verdict.reason = (
            f"improvement is not statistically reliable "
            f"(p={p_value:.4f} > alpha={config.alpha}) — likely noise"
        )
        return verdict

    verdict.decision = Decision.ACCEPT
    verdict.reason = (
        f"{len(improved)} improvement(s), 0 regressions, p={p_value:.4f} <= alpha={config.alpha}"
    )
    return verdict


# --------------------------------------------------------------------------
# Improvement dynamics
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ImprovementDynamics:
    """How a sequence of candidate scores evolved.

    Borrowed from the way Frontier-Eng (arXiv 2604.12290) reports self-evolving
    agents: not one delta, but how often improvement happened and how large it
    was. A single before/after number hides a loop that improved twice then
    regressed three times.
    """

    steps: int = 0
    improvements: int = 0
    regressions: int = 0
    mean_improvement: float = 0.0
    mean_regression: float = 0.0
    improvement_rate: float = 0.0
    decays: bool | None = None
    half_life: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "improvements": self.improvements,
            "regressions": self.regressions,
            "improvement_rate": round(self.improvement_rate, 4),
            "mean_improvement": round(self.mean_improvement, 4),
            "mean_regression": round(self.mean_regression, 4),
            "improvement_magnitude_decays": self.decays,
            "improvement_half_life": (
                round(self.half_life, 2) if self.half_life is not None else None
            ),
        }


def analyse_dynamics(scores: Sequence[float]) -> ImprovementDynamics:
    """Summarise improvement frequency and magnitude across iterations.

    ``decays`` reports whether improvement size falls roughly geometrically,
    the pattern Frontier-Eng reports. ``None`` when there are too few
    improvements to tell decay from noise.
    """

    deltas = [b - a for a, b in zip(scores, scores[1:], strict=False)]
    gains = [d for d in deltas if d > 0]
    losses = [d for d in deltas if d < 0]

    dynamics = ImprovementDynamics(
        steps=len(deltas),
        improvements=len(gains),
        regressions=len(losses),
        mean_improvement=(sum(gains) / len(gains)) if gains else 0.0,
        mean_regression=(sum(abs(loss) for loss in losses) / len(losses)) if losses else 0.0,
        improvement_rate=(len(gains) / len(deltas)) if deltas else 0.0,
    )

    # At least three gains are needed before a decay claim means anything.
    if len(gains) >= 3:
        ordered = sorted(gains, reverse=True)
        first_half = sum(ordered[: len(ordered) // 2]) / max(len(ordered) // 2, 1)
        second_half = sum(ordered[len(ordered) // 2 :]) / max(len(ordered) - len(ordered) // 2, 1)
        dynamics.decays = second_half < first_half
        if second_half > 0 and first_half > 0:
            # Geometric ratio r with second_half/first_half ~= r^(n/2).
            ratio = second_half / first_half
            if 0 < ratio < 1:
                dynamics.half_life = math.log(0.5) / math.log(ratio)
    return dynamics
