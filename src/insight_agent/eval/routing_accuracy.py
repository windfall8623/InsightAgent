"""Intent-recognition and routing accuracy harness for the CN-Retail 2.0 cases.

This measures the *intent cascade* (``classify_intent`` in router.py) — the layer
that /ask and /ask/stream use to decide whether a question is chit-chat, a causal
diagnosis (LangGraph), an injection, or an ordinary analytics query that falls
through to the orchestrator.

Two scorecards are produced on purpose:

1. ``raw 5-class`` — intent == gold intent. Reported for continuity, but note
   that in /ask the simple_metric vs complex_analytic split has *no downstream
   consequence*: both fall through to ``agent.run``. So raw accuracy over-counts
   "errors" that change nothing.

2. ``functional`` — the three decisions that actually change the request's path:
   the diagnostic gate (is_anomaly), the chat gate (chat_general), and injection
   capture. These are the numbers to optimise.

Run:  PYTHONPATH=src python -m insight_agent.eval.routing_accuracy
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Any

from insight_agent.app.router import classify_intent
from insight_agent.eval.cn_retail_200_cases import CN_RETAIL_200_CASES

INTENTS = ("diagnostic_anomaly", "complex_analytic", "simple_metric", "prompt_injection", "chat_general")


def gold_intent(case: dict[str, Any]) -> str:
    """Reference intent for a benchmark case, derived from its category.

    The simple/complex boundary is a proxy (requires_tables >= 2); the clear
    classes are security -> injection, diagnostic -> diagnostic, unanswerable ->
    chat (no data query to run).
    """
    cat = case["category"]
    if cat == "security":
        return "prompt_injection"
    if cat == "diagnostic":
        return "diagnostic_anomaly"
    if cat == "unanswerable":
        return "chat_general"
    return "complex_analytic" if len(case.get("requires_tables") or []) >= 2 else "simple_metric"


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def evaluate(mode: str | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in CN_RETAIL_200_CASES:
        gold = gold_intent(case)
        dec = classify_intent(case["question"], mode=mode)
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "gold": gold,
                "pred": dec.intent,
                "is_anomaly": dec.is_anomaly,
                "confidence": round(dec.confidence, 3),
                "route_mode": dec.mode,
            }
        )

    n = len(rows)
    raw_correct = sum(1 for r in rows if r["pred"] == r["gold"])

    # Functional metric 1: diagnostic routing (is_anomaly). gold positive = gold diagnostic.
    d_tp = sum(1 for r in rows if r["gold"] == "diagnostic_anomaly" and r["is_anomaly"])
    d_fn = sum(1 for r in rows if r["gold"] == "diagnostic_anomaly" and not r["is_anomaly"])
    d_fp = sum(1 for r in rows if r["gold"] != "diagnostic_anomaly" and r["is_anomaly"])
    diag = _prf(d_tp, d_fp, d_fn)

    # Functional metric 2: chat gate (chat_general).
    c_tp = sum(1 for r in rows if r["gold"] == "chat_general" and r["pred"] == "chat_general")
    c_fn = sum(1 for r in rows if r["gold"] == "chat_general" and r["pred"] != "chat_general")
    c_fp = sum(1 for r in rows if r["gold"] != "chat_general" and r["pred"] == "chat_general")
    chat = _prf(c_tp, c_fp, c_fn)

    # Functional metric 3: injection capture (intent label).
    i_tp = sum(1 for r in rows if r["gold"] == "prompt_injection" and r["pred"] == "prompt_injection")
    i_fn = sum(1 for r in rows if r["gold"] == "prompt_injection" and r["pred"] != "prompt_injection")
    i_fp = sum(1 for r in rows if r["gold"] != "prompt_injection" and r["pred"] == "prompt_injection")
    inj = _prf(i_tp, i_fp, i_fn)

    confusion: dict[str, Counter] = {g: Counter() for g in INTENTS}
    for r in rows:
        confusion[r["gold"]][r["pred"]] += 1

    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        per_cat[r["category"]][1] += 1
        if r["pred"] == r["gold"]:
            per_cat[r["category"]][0] += 1

    return {
        "n": n,
        "raw_accuracy": raw_correct / n,
        "diag": {"precision": diag[0], "recall": diag[1], "f1": diag[2], "tp": d_tp, "fp": d_fp, "fn": d_fn},
        "chat": {"precision": chat[0], "recall": chat[1], "f1": chat[2], "tp": c_tp, "fp": c_fp, "fn": c_fn},
        "injection": {"precision": inj[0], "recall": inj[1], "f1": inj[2], "tp": i_tp, "fp": i_fp, "fn": i_fn},
        "confusion": {g: dict(confusion[g]) for g in INTENTS},
        "per_category": {k: {"correct": v[0], "total": v[1]} for k, v in per_cat.items()},
        "rows": rows,
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def print_report(res: dict[str, Any]) -> None:
    print(f"\n=== Intent recognition & routing accuracy (n={res['n']}) ===")
    print(f"raw 5-class accuracy : {_fmt_pct(res['raw_accuracy'])}  (simple<->complex split is consequence-free in /ask)")
    print("\n--- functional gates (the decisions that change the request path) ---")
    for name, key in (("diagnostic gate", "diag"), ("chat gate", "chat"), ("injection capture", "injection")):
        m = res[key]
        print(
            f"{name:18s}: P={_fmt_pct(m['precision'])} R={_fmt_pct(m['recall'])} "
            f"F1={_fmt_pct(m['f1'])}  (tp={m['tp']} fp={m['fp']} fn={m['fn']})"
        )

    print("\n--- confusion matrix (gold rows x pred cols) ---")
    short = {"diagnostic_anomaly": "diag", "complex_analytic": "cplx", "simple_metric": "smpl",
             "prompt_injection": "inj", "chat_general": "chat"}
    header = "gold \\ pred    " + "".join(f"{short[i]:>7}" for i in INTENTS)
    print(header)
    for g in INTENTS:
        line = f"{short[g]:>7} ({sum(res['confusion'][g].values()):>3})".ljust(15)
        line += "".join(f"{res['confusion'][g].get(p, 0):>7}" for p in INTENTS)
        print(line)

    print("\n--- per category raw accuracy ---")
    for cat, v in sorted(res["per_category"].items()):
        print(f"{cat:14s}: {v['correct']:>3}/{v['total']:<3} = {_fmt_pct(v['correct'] / v['total'])}")


if __name__ == "__main__":
    mode = os.environ.get("INSIGHT_INTENT_MODE")
    res = evaluate(mode=mode)
    print_report(res)
