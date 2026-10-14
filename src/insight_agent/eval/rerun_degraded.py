"""Re-run only the rerank-degraded instances of one or more strategies.

Each case carries ``rerank_degraded=True`` when its real remote reranker timed
out and silently fell back to the heuristic backend. Because the sweep is
strictly sequential and rich_vector/deep_hybrid issue exactly one rerank call
per case, that flag identifies the precise instances whose results are
contaminated. This script re-runs ONLY those instances in the current (hopefully
healthy) window, verifies each one actually used the real reranker this time,
splices the clean outcomes back into the per-strategy cache, and re-aggregates
the headline report.

Re-running only the broken instances is both faster and cleaner than re-running
the whole strategy: the ~160 already-clean cases keep their outcomes instead of
being re-rolled through LLM nondeterminism.

    uv run python -m insight_agent.eval.rerun_degraded                       # all strategies with degraded cases
    uv run python -m insight_agent.eval.rerun_degraded --strategies rich_vector,deep_hybrid
    uv run python -m insight_agent.eval.rerun_degraded --no-health-probe --max-rounds 5
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# Force live semantics BEFORE importing the eval module, so it keeps the real
# Jev endpoint instead of blanking it (that module decides at import time).
os.environ["INSIGHT_EVAL_LIVE"] = "1"

from insight_agent.app.bge_reranker import RerankCandidate  # noqa: E402
from insight_agent.app.evolution import Outcome, Strategy, default_strategies  # noqa: E402
from insight_agent.app.memory import MemoryStore  # noqa: E402
from insight_agent.app.orchestrator import InsightOrchestrator  # noqa: E402
from insight_agent.app.planner import build_planner  # noqa: E402
from insight_agent.app.rag import build_retriever  # noqa: E402
from insight_agent.config import load_config  # noqa: E402
from insight_agent.data.cn_retail_db import connect_cn_retail, ensure_cn_retail_database  # noqa: E402
from insight_agent.eval.cn_retail_200_cases import CN_RETAIL_200_CASES  # noqa: E402
from insight_agent.eval.rebuild_live_report import main as rebuild_main  # noqa: E402
from insight_agent.eval.run_cn_retail_200_eval import (  # noqa: E402
    STRATEGY_CACHE_DIR,
    load_strategy_cache,
    run_one_case,
    strategy_cache_path,
    write_strategy_cache,
)


def probe_endpoint(reranker, *, calls: int, min_success: float) -> tuple[bool, float]:
    """Fire a few real rerank calls; return (healthy, success_rate).

    Reuses the exact production rerank path, so a healthy verdict here means the
    re-run has a real chance of landing clean rather than re-contaminating.
    """
    cands = [
        RerankCandidate(doc_id="probe_a", text="订单金额 按 月份 汇总", initial_score=0.5),
        RerankCandidate(doc_id="probe_b", text="退货率 与 类目 的关系", initial_score=0.4),
    ]
    ok = 0
    for i in range(calls):
        before = reranker.remote_fallback_count
        reranker.rerank(f"健康探活调用 {i} 近30天各类目销量", cands, top_k=2)
        if reranker.remote_fallback_count == before:
            ok += 1
        time.sleep(0.3)
    rate = ok / calls if calls else 0.0
    return rate >= min_success, rate


async def rerun_strategy(
    strat: Strategy,
    conn,
    retriever,
    planner,
    *,
    max_rounds: int,
    round_pause: float,
) -> dict[str, int]:
    """Re-run the degraded instances of one strategy and rewrite its cache."""
    outcomes = load_strategy_cache(strat.name)
    by_id: dict[str, Outcome] = {o.case_id: o for o in outcomes}
    case_by_id = {c["id"]: c for c in CN_RETAIL_200_CASES}
    degraded_ids = [
        o.case_id for o in outcomes
        if o.rerank_degraded or o.embedding_degraded or o.llm_error
    ]

    stats = {"degraded": len(degraded_ids), "resolved": 0, "unresolved": 0}
    if not degraded_ids:
        print(f"  [{strat.name}] no degraded instances; skipping.")
        return stats

    orchestrator = InsightOrchestrator(
        conn, memory=MemoryStore(local_only=True), config=strat.config, retriever=retriever, planner=planner
    )

    unresolved = list(degraded_ids)
    for rnd in range(1, max_rounds + 1):
        if not unresolved:
            break
        print(f"  [{strat.name}] round {rnd}/{max_rounds}: re-running {len(unresolved)} degraded case(s)...", flush=True)
        still: list[str] = []
        for cid in unresolved:
            case = case_by_id.get(cid)
            if case is None:  # case id no longer in the dataset
                continue
            new_outcome = await run_one_case(
                orchestrator, conn, case, strat.name, previous=by_id[cid], live=True
            )
            by_id[cid] = new_outcome  # always keep the freshest attempt
            if new_outcome.rerank_degraded or new_outcome.embedding_degraded or new_outcome.llm_error:
                still.append(cid)  # degraded again — needs another healthy window
        unresolved = still
        if unresolved and rnd < max_rounds:
            await asyncio.sleep(round_pause)

    stats["resolved"] = stats["degraded"] - len(unresolved)
    stats["unresolved"] = len(unresolved)

    merged = [by_id[o.case_id] for o in outcomes]  # preserve original case order
    path = write_strategy_cache(strat.name, merged, live=True)
    print(
        f"  [{strat.name}] resolved {stats['resolved']}/{stats['degraded']} "
        f"(still degraded: {stats['unresolved']}) -> cache rewritten {path}"
    )
    return stats


async def _amain(args: argparse.Namespace) -> int:
    cfg = load_config()
    reason = cfg.missing_reason()
    if not cfg.is_live or reason:
        print(f"[FATAL] rerun needs a live LLM but none configured: {reason or 'not live'}")
        return 2
    planner = build_planner(cfg)
    if type(planner).__name__ != "LLMPlanner":
        print("[FATAL] build_planner degraded to RulesPlanner; check .env")
        return 2
    print(f"[LIVE] planner={type(planner).__name__} model={cfg.model} base={cfg.base_url}")

    ensure_cn_retail_database()
    conn = connect_cn_retail()
    retriever = build_retriever(conn)

    all_strategies = default_strategies()
    name_to_strat = {s.name: s for s in all_strategies}

    requested = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if requested:
        unknown = [r for r in requested if r not in name_to_strat]
        if unknown:
            print(f"[FATAL] unknown strategies: {unknown}; valid={list(name_to_strat)}")
            return 2
        targets = [name_to_strat[r] for r in requested]
    else:
        # Default: every strategy whose cache exists and has degraded instances.
        targets = []
        for s in all_strategies:
            p = strategy_cache_path(s.name)
            if not p.exists():
                continue
            if any(
                o.rerank_degraded or o.embedding_degraded or o.llm_error
                for o in load_strategy_cache(s.name)
            ):
                targets.append(s)

    missing = [s.name for s in targets if not strategy_cache_path(s.name).exists()]
    if missing:
        print(f"[FATAL] missing caches for requested strategies: {missing} (run the sweep first)")
        return 2
    if not targets:
        print(f"[Done] No degraded instances found in any cache under {STRATEGY_CACHE_DIR}. Nothing to re-run.")
        return 0

    print(f"[Targets] {[s.name for s in targets]}")

    if args.health_probe:
        healthy, rate = probe_endpoint(
            retriever.bge_reranker, calls=args.probe_calls, min_success=args.probe_min_success
        )
        print(f"[Health Probe] reranker success rate = {rate*100:.0f}% over {args.probe_calls} calls "
              f"(threshold {args.probe_min_success*100:.0f}%)")
        if not healthy:
            print("[ABORT] endpoint still in a bad window; re-running now would just re-contaminate. "
                  "Try again later, or pass --no-health-probe to force.")
            return 3

    totals = {"degraded": 0, "resolved": 0, "unresolved": 0}
    for strat in targets:
        s = await rerun_strategy(
            strat, conn, retriever, planner,
            max_rounds=args.max_rounds, round_pause=args.round_pause,
        )
        for k in totals:
            totals[k] += s[k]

    print(f"\n[Rerun Summary] degraded={totals['degraded']} resolved={totals['resolved']} "
          f"still_degraded={totals['unresolved']}")

    # Fold repaired caches into the headline report if the full set is present.
    if all(strategy_cache_path(s.name).exists() for s in all_strategies):
        print("[Re-aggregate] regenerating full live report from caches...")
        rebuild_main()
    else:
        have = [s.name for s in all_strategies if strategy_cache_path(s.name).exists()]
        print(f"[Re-aggregate skipped] only {have} cached; need all 4. "
              f"Run rebuild_live_report once the rest exist.")

    if totals["unresolved"]:
        print(f"[Note] {totals['unresolved']} instance(s) still fell back to heuristic even now; "
              f"re-run again in a healthier window to finish cleaning them.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-run rerank-degraded instances and re-aggregate.")
    parser.add_argument("--strategies", default="",
                        help="Comma-separated strategies (default: all strategies with degraded cases).")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Max retry rounds for instances that keep falling back (default 3).")
    parser.add_argument("--round-pause", type=float, default=10.0,
                        help="Seconds to wait between retry rounds (default 10).")
    parser.add_argument("--probe-calls", type=int, default=8,
                        help="Health-probe rerank calls before starting (default 8).")
    parser.add_argument("--probe-min-success", type=float, default=0.6,
                        help="Min probe success rate to proceed (default 0.6).")
    parser.add_argument("--no-health-probe", dest="health_probe", action="store_false",
                        help="Skip the pre-flight endpoint health probe.")
    parser.set_defaults(health_probe=True)
    args = parser.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
