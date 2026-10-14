"""BFCL adapter: score the function-calling engine on the canonical benchmark.

Why this exists
---------------
The project's own 51-case dataset is the obvious weak point in an interview —
"you wrote the questions, of course you pass them". This module removes that
objection by scoring on **Berkeley Function Calling Leaderboard (BFCL) v3**,
an external benchmark maintained by the Gorilla team (13k+ stars, Apache-2.0),
downloaded from HuggingFace and shipped in ``.bfcl_cache/``.

Scoring follows BFCL's own semantics
------------------------------------
Ground truth is ``{"func_name": {arg: [acceptable values]}}``. A prediction
counts only if:

* the function name matches exactly, and
* the predicted argument names match exactly, and
* every predicted value matches one of the acceptable values for that argument.

That is strict on purpose. A rules baseline will score low here, and reporting
that honestly is the entire point: an external number nobody tuned against is
worth more than a flattering self-made one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insight_agent.app.funccall import FunctionCall, FunctionSpec, decide

CACHE_DIR = Path(".bfcl_cache")

# Categories that carry a ground-truth file. `irrelevance` has none by design:
# the correct behaviour there is to call nothing.
SCORED_CATEGORIES = ("simple", "multiple", "parallel")
IRRELEVANCE_CATEGORY = "irrelevance"

DATASET_URL = (
    "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
)


@dataclass(slots=True)
class BFCLCase:
    case_id: str
    request: str
    functions: list[FunctionSpec]
    ground_truth: list[dict[str, Any]] | None
    category: str

    @property
    def expect_no_call(self) -> bool:
        return self.ground_truth is None


@dataclass(slots=True)
class CategoryScore:
    category: str
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
        }


@dataclass(slots=True)
class BFCLReport:
    categories: list[CategoryScore] = field(default_factory=list)
    total: int = 0
    correct: int = 0
    source: str = ""

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "categories": [c.to_payload() for c in self.categories],
        }


def cache_path(name: str, *, answer: bool = False) -> Path:
    suffix = "_answer" if answer else ""
    return CACHE_DIR / f"BFCL_v3_{name}{suffix}.jsonl"


def fetch(category: str, *, answer: bool = True, timeout: int = 120) -> Path:
    """Download one BFCL category into the local cache, if not already present."""

    import urllib.request

    target = cache_path(category, answer=answer)
    if target.exists():
        return target
    folder = "possible_answer/" if answer else ""
    url = f"{DATASET_URL}/{folder}BFCL_v3_{category}.json"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(request, timeout=timeout).read().decode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(raw, encoding="utf-8")
    return target


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_category(category: str, *, limit: int | None = None) -> list[BFCLCase]:
    """Load one BFCL category into cases with ground truth attached."""

    cases: list[BFCLCase] = []
    answers = {
        a["id"]: a.get("ground_truth") for a in _read_jsonl(cache_path(category, answer=True))
    }

    for record in _read_jsonl(cache_path(category)):
        turns = record.get("question") or []
        # Each entry is a list of turns; the first user message is the request.
        text = ""
        for turn in turns:
            for message in turn:
                if message.get("role") == "user":
                    text = message.get("content", "")
                    break
            if text:
                break

        raw_functions = record.get("function")
        specs = [FunctionSpec.from_raw(raw_functions)] if raw_functions else []
        cases.append(
            BFCLCase(
                case_id=record["id"],
                request=text,
                functions=specs,
                ground_truth=answers.get(record["id"]),
                category=category,
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def _normalise(value: Any) -> str:
    """Lowercase, trim, and collapse numeric formatting for comparison."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        as_float = float(value)
        return str(int(as_float)) if as_float.is_integer() else f"{as_float:g}"
    return str(value).strip().lower()


def split_truth(ground_truth: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Unpack BFCL's nested ground truth: ``{"fn": {arg: [values]}}``.

    The function name is the dict *key*, not a field — reading it as a field
    silently scores every case wrong.
    """

    name = next(iter(ground_truth), "")
    arguments = ground_truth[name]
    return name, arguments if isinstance(arguments, dict) else {}


def _matches_call(predicted: dict[str, Any], truth: dict[str, Any]) -> bool:
    """BFCL-style AST match for a single call.

    Follows BFCL's own tolerance rule: a predicted argument that does not exist
    in the ground truth is a mismatch, but a ground-truth argument the model
    omitted is accepted when the empty string is among its acceptable values
    (that is how BFCL marks an argument as genuinely optional).
    """

    truth_name, expected = split_truth(truth)
    if predicted.get("name") != truth_name:
        return False

    got: dict[str, Any] = predicted.get("arguments", {}) or {}
    if not set(got).issubset(expected):
        return False

    for name, value in got.items():
        acceptable = expected.get(name)
        if acceptable is None:
            return False
        if isinstance(acceptable, dict):  # nested-object form, not handled here
            return False
        if not isinstance(acceptable, (list, tuple)):
            acceptable = [acceptable]
        if _normalise(value) not in {_normalise(v) for v in acceptable}:
            return False

    # Everything the ground truth demands must have been supplied, unless it is
    # explicitly optional.
    for name, acceptable in expected.items():
        if name in got:
            continue
        values = acceptable if isinstance(acceptable, (list, tuple)) else [acceptable]
        if _normalise("") not in {_normalise(v) for v in values}:
            return False
    return True


def score_case(case: BFCLCase, prediction: FunctionCall) -> bool:
    if case.expect_no_call:
        # Irrelevance: calling anything is a failure.
        return not prediction.calls
    truth = case.ground_truth or []
    predicted = prediction.calls
    if len(predicted) != len(truth):
        return False
    # Order-insensitive: parallel calls may legitimately be emitted in any order.
    remaining = list(truth)
    for call in predicted:
        for index, candidate in enumerate(remaining):
            if _matches_call(call, candidate):
                remaining.pop(index)
                break
        else:
            return False
    return True


def evaluate(
    categories: Sequence[str] = SCORED_CATEGORIES + (IRRELEVANCE_CATEGORY,),
    *,
    limit: int | None = None,
    relevance_threshold: float = 0.45,
) -> BFCLReport:
    """Run the engine over BFCL and score it with BFCL's own criteria."""

    report = BFCLReport(source="gorilla-llm/Berkeley-Function-Calling-Leaderboard (BFCL v3)")

    for category in categories:
        cases = load_category(category, limit=limit)
        if not cases:
            continue
        score = CategoryScore(category=category)
        for case in cases:
            prediction = decide(
                case.request, case.functions, relevance_threshold=relevance_threshold
            )
            if score_case(case, prediction):
                score.correct += 1
            score.total += 1
        report.categories.append(score)
        report.correct += score.correct
        report.total += score.total

    return report


def main() -> int:
    report = evaluate()
    print(f"source : {report.source}")
    print(f"total  : {report.total} cases, {report.correct} correct ({report.accuracy:.1%})\n")
    header = f"{'category':<16}{'correct':>9}{'total':>8}{'accuracy':>11}"
    print(header)
    print("-" * len(header))
    for category in report.categories:
        print(
            f"{category.category:<16}{category.correct:>9}"
            f"{category.total:>8}{category.accuracy:>10.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
