"""Train / dev / test split for the evaluation dataset.

Why this exists
---------------
Three rounds of work went into removing the "you wrote the questions, of
course you pass them" objection — including scoring on external BFCL data. A
prompt optimiser such as DSPy would immediately undo all of that, because its
entire purpose is to grind a metric down. Fed the same 51 cases it both trains
and reports on, it will simply memorise them.

So the split is not a formality:

* **train** — the only data an optimiser may see
* **dev**  — choosing between candidate configurations
* **test**  — touched exactly once, at final report time

Freezing
--------
The test membership is written to a manifest with a content hash. If the
dataset changes, the hash changes and the freeze is visibly invalidated rather
than silently re-drawn — otherwise a "frozen" test set can quietly become
whatever the author found convenient.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from insight_agent.eval.dataset import load_cases

SPLITS = ("train", "dev", "test")
DEFAULT_SIZES = {"train": 30, "dev": 11, "test": 10}
MANIFEST_NAME = "split_manifest.json"


@dataclass(slots=True)
class Split:
    """Which case ids belong to each split."""

    train: list[str] = field(default_factory=list)
    dev: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    dataset_hash: str = ""

    def ids(self, name: str) -> list[str]:
        if name not in SPLITS:
            raise ValueError(f"unknown split: {name!r}")
        value: list[str] = getattr(self, name)
        return value

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def dataset_hash() -> str:
    """Stable hash over the dataset's identity and expectations."""

    parts = [f"{c['id']}:{c['expectation']}" for c in load_cases()]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def build_split(
    sizes: dict[str, int] | None = None,
    *,
    seed: int = 20261004,
    source: str = "all",
) -> Split:
    """Partition the dataset deterministically.

    Assignment is a stable hash of the case id, not an RNG draw, so adding a
    case later does not reshuffle the others' membership.

    ``source="generated"`` restricts the split to cases never used to tune
    anything, which is the only split whose numbers mean anything.
    """

    sizes = dict(sizes or DEFAULT_SIZES)
    cases = load_cases()
    if source == "generated":
        cases = [c for c in cases if c.get("source") == "generated"] or cases
    ordered = sorted(cases, key=lambda c: hashlib.sha256(f"{seed}:{c['id']}".encode()).hexdigest())

    buckets: dict[str, list[str]] = {name: [] for name in SPLITS}
    for case in ordered:
        for name in SPLITS:
            if len(buckets[name]) < sizes.get(name, 0):
                buckets[name].append(case["id"])
                break

    return Split(
        train=sorted(buckets["train"]),
        dev=sorted(buckets["dev"]),
        test=sorted(buckets["test"]),
        dataset_hash=dataset_hash(),
    )


def load_cases_for(split: Split, name: str) -> list[dict[str, Any]]:
    wanted = set(split.ids(name))
    return [c for c in load_cases() if c["id"] in wanted]


def holdout_split() -> Split:
    """The honest measurement split: generated cases only, 60/20/20."""

    return build_split({"train": 255, "dev": 85, "test": 86}, source="generated")


def save_manifest(split: Split, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(split.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_manifest(out_dir: Path) -> Split:
    return Split(**json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8")))


def verify_frozen(expected_hash: str) -> tuple[bool, str]:
    """Check the dataset has not changed since the split was frozen."""

    current = dataset_hash()
    if current == expected_hash:
        return True, "dataset unchanged since the split was frozen"
    return False, (
        f"dataset hash changed: frozen as {expected_hash}, now {current}. "
        "Re-splitting invalidates every number reported against the old test set."
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Split the evaluation dataset.")
    parser.add_argument("--train", type=int, default=DEFAULT_SIZES["train"])
    parser.add_argument("--dev", type=int, default=DEFAULT_SIZES["dev"])
    parser.add_argument("--test", type=int, default=DEFAULT_SIZES["test"])
    parser.add_argument("--out", default="eval_results")
    args = parser.parse_args(argv)

    split = build_split({"train": args.train, "dev": args.dev, "test": args.test})
    out = Path(args.out)
    path = save_manifest(split, out)

    print(f"dataset hash : {split.dataset_hash}")
    for name in SPLITS:
        print(f"{name:<6}: {len(split.ids(name)):>3} cases")
    ok, message = verify_frozen(split.dataset_hash)
    print(f"\nfreeze      : {'OK' if ok else 'BROKEN'} — {message}")
    print(f"manifest    : {path}")

    # Guard against a split that silently drops or double-assigns cases.
    assigned = split.train + split.dev + split.test
    if len(assigned) != len(set(assigned)):
        print("error: a case appears in more than one split")
        return 1
    print(f"coverage    : {len(assigned)}/{len(load_cases())} cases assigned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
