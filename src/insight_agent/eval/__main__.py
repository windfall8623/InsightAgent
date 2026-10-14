"""Run the evaluation suite from the command line: ``uv run eval``."""

from __future__ import annotations

import asyncio
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    import argparse

    from insight_agent.app.planner import build_planner
    from insight_agent.config import PROVIDERS, load_config
    from insight_agent.eval.runner import run_all
    from insight_agent.eval.split import SPLITS, build_split, load_cases_for

    parser = argparse.ArgumentParser(prog="eval", description="Run the evaluation suite.")
    parser.add_argument("--out", default="eval_results", help="Output directory.")
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=None,
        help="Planner backend. Default comes from INSIGHT_PROVIDER.",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default=None,
        help="Evaluate only one split of the frozen dataset.",
    )
    args = parser.parse_args(argv)

    config = load_config(provider=args.provider)
    planner = build_planner(config)
    reason = config.missing_reason()

    cases = None
    if args.split:
        cases = load_cases_for(build_split(), args.split)
        print(f"split: {args.split} ({len(cases)} cases)")
    print(f"planner: {planner.name}" + (f" ({config.model})" if config.is_live else ""))
    if reason:
        print(f"warning: {reason}; falling back to the rules planner")
    print()

    payload = asyncio.run(run_all(out_dir=Path(args.out), planner=planner, cases=cases))

    meta = payload.pop("_meta")
    print(f"dataset: {meta['cases']} cases | backend: {meta['backend']}")
    print()
    header = (
        f"{'variant':<15}{'success':>9}{'exec_acc':>10}{'recall':>8}{'p50_ms':>9}{'cost_usd':>11}"
    )
    print(header)
    print("-" * len(header))
    for name, report in payload.items():
        print(
            f"{name:<15}{report['task_success_rate']:>9.3f}"
            f"{report['execution_accuracy']:>10.3f}"
            f"{report['mean_retrieval_recall']:>8.3f}"
            f"{report['p50_latency_ms']:>9.2f}"
            f"{report['total_cost_usd']:>11.5f}"
        )

    failures = payload["full"]["failures"]
    if failures:
        print(f"{chr(10)}{len(failures)} failing case(s):")
        for failure in failures:
            print(f"  {failure['case_id']}: {failure['question']} -- {failure['error']}")

    print(f"{chr(10)}written to {Path(args.out) / 'eval_report.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
