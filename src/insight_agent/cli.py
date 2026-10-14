"""Command-line entry point: ``insight "your question"``.

Runs fully offline — no API key, no network — because every LLM-shaped decision
is made by the deterministic engine in :mod:`insight_agent.app.nl2sql`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from insight_agent.app.mcp_server import serve_tools
from insight_agent.app.memory import MemoryStore
from insight_agent.app.orchestrator import InsightOrchestrator, InsightRequest, RunConfig
from insight_agent.app.planner import build_planner
from insight_agent.app.security import ToolPolicy
from insight_agent.config import PROVIDERS, LLMConfig, load_config

__version__ = "0.2.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="insight",
        description="Ask your e-commerce data a question in plain language.",
    )
    parser.add_argument("question", nargs="*", help="The question to answer.")
    parser.add_argument("--user", default="cli-user", help="User id, for per-user memory.")
    parser.add_argument("--json", action="store_true", help="Emit the full result payload.")
    parser.add_argument("--explain", action="store_true", help="Show the provenance trail.")
    parser.add_argument(
        "--no-rag", action="store_true", help="Disable schema retrieval (ablation)."
    )
    parser.add_argument("--no-memory", action="store_true", help="Disable memory (ablation).")
    parser.add_argument(
        "--retrieval",
        default="hybrid",
        choices=("hybrid", "keyword", "vector"),
        help="Retrieval strategy (ablation).",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=None,
        help="Planner backend. Default comes from INSIGHT_PROVIDER (rules).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the API endpoint (proxy, gateway, or local model server).",
    )
    parser.add_argument("--model", default=None, help="Override the model id.")
    parser.add_argument(
        "--show-config", action="store_true", help="Print resolved LLM config and exit."
    )
    parser.add_argument("--serve-mcp", action="store_true", help="Run as an MCP stdio server.")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run 2.0 LangGraph anomaly diagnosis on Olist database.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream diagnosis events step-by-step (used with --diagnose).",
    )
    parser.add_argument(
        "--eval-diag",
        action="store_true",
        help="Run the 50-case benchmark and print SAGE gate report.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Enable interactive Human-in-the-Loop approval for high-impact actions.",
    )
    parser.add_argument(
        "--dataset",
        choices=("cn_retail", "olist"),
        default=os.environ.get("INSIGHT_DATASET", "cn_retail"),
        help="Target dataset: 'cn_retail' or 'olist' (default: cn_retail).",
    )
    parser.add_argument("--version", action="version", version=f"insight {__version__}")
    return parser


def resolve_config(args: argparse.Namespace) -> LLMConfig:
    """Configuration with command-line overrides applied."""

    config = load_config(provider=args.provider)
    if args.base_url:
        config.base_url = args.base_url
    if args.model:
        config.model = args.model
    return config


def build_orchestrator(args: argparse.Namespace) -> InsightOrchestrator:
    llm_config = resolve_config(args)
    planner = build_planner(llm_config)
    config = RunConfig(
        use_rag=not args.no_rag,
        use_memory=not args.no_memory,
        retrieval_mode=args.retrieval,
    )
    from insight_agent.api import _get_active_dataset_conn
    from insight_agent.app.evolution import (
        EvolutionEngine,
        default_evolution_path,
        default_strategies,
    )
    from insight_agent.app.security import TerminalApprover

    conn, _ = _get_active_dataset_conn()

    approver = None
    if getattr(args, "interactive", False):
        approver = TerminalApprover()

    return InsightOrchestrator(
        conn,
        memory=MemoryStore(),
        policy=ToolPolicy(),
        approver=approver,
        config=config,
        planner=planner,
        evolution=EvolutionEngine(default_strategies(), path=default_evolution_path()),
    )


async def _ask(args: argparse.Namespace) -> int:
    orchestrator = build_orchestrator(args)
    result = await orchestrator.run(
        InsightRequest(question=" ".join(args.question), user_id=args.user)
    )

    if args.json:
        print(json.dumps(result.to_payload(), ensure_ascii=False, indent=2, default=str))
        return 0

    if args.explain:
        print(f"question   : {result.question}")
        print(f"agents     : {' -> '.join(result.agents_used) or '(none)'}")
        print(f"tools      : {', '.join(result.tools_used) or '(none)'}")
        print(f"model      : {result.model} ({result.route_reason})")
        print(f"reflections: {result.reflections}")
        print(f"sql        :\n{result.sql}")
        print("evidence   :")
        for item in result.evidence:
            mark = "*" if item.get("needed") else " "
            print(f"  {mark} {item['id']}  score={item['score']}")
        print(f"latency    : {result.latency_ms:.1f} ms   cost=${result.cost_usd:.6f}")
        if result.alerts:
            print(f"alerts     : {result.alerts}")
        print()

    if result.blocked:
        print(f"[blocked] {result.block_reason}", file=sys.stderr)

    print(result.answer)
    if result.columns:
        print()
        print(" | ".join(result.columns))
        for row in result.rows[:10]:
            print(" | ".join("" if v is None else str(v) for v in row.values()))
        if len(result.rows) > 10:
            print(f"... {len(result.rows) - 10} more rows")
    return 0


def _diagnose_cli(args: argparse.Namespace) -> int:
    from insight_agent.api import _get_active_dataset_conn
    from insight_agent.app.graph import InsightGraphEngine
    from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
    from insight_agent.app.redis_store import RedisMemoryStore

    query = " ".join(args.question)
    if not query:
        print("error: --diagnose requires a question/anomaly description", file=sys.stderr)
        return 1

    conn, _ = _get_active_dataset_conn()
    retriever = OlistMultiModalRetriever(conn)
    redis_store = RedisMemoryStore()
    engine = InsightGraphEngine(conn, retriever, redis_store)


    is_interactive = bool(getattr(args, "interactive", False))

    if args.stream:
        print(f"[*] Starting streaming diagnosis for: {query}")
        print("[-] [Step 1] Intent router classifying query...")
        state = engine.diagnose(query, user_id=args.user, hitl_approved=False)
        print(f"    Intent: {state.get('intent')} (Blocked: {state.get('blocked')})")
        if state.get("blocked"):
            print(f"[!] Blocked: {state.get('block_reason')}", file=sys.stderr)
            return 1
        print("[-] [Step 2] Formulating and testing causal hypotheses...")
        for h in state.get("hypotheses", []):
            st = h.get("status")
            tag = (
                "CONFIRMED"
                if st == "confirmed"
                else ("FALSIFIED" if st == "falsified" else "TESTING")
            )
            print(f"    - [{tag}] {h.get('name')}: {h.get('statement')}")
            if h.get("falsified_reason"):
                print(f"      (Reflection triggered: {h.get('falsified_reason')})")
        print(f"[-] [Step 3] Root cause confirmed: {state.get('root_cause')}")
        if state.get("candidate_actions"):
            act = state["candidate_actions"][0]
            print(f"[-] [Step 4] Human-in-the-loop action proposed: {act['description']}")
            if is_interactive:
                prompt = (
                    f"\n[HITL 审批确认] 诊断完成，提议执行补救动作：\n"
                    f"  动作: {act.get('description', '')}\n"
                    f"  参数: {act.get('arguments', {})}\n"
                    f"是否批准执行? [y/N]: "
                )
                try:
                    choice = input(prompt).strip().lower()
                    if choice in ("y", "yes"):
                        state = engine.diagnose(
                            query,
                            user_id=args.user,
                            thread_id=state.get("thread_id"),
                            hitl_approved=True,
                        )
                        print(f"    Distributed Redlock acquired -> Status: {state.get('action_result')}")
                    else:
                        print("    [HITL] 操作员拒绝执行补救写操作。")
                except (EOFError, KeyboardInterrupt):
                    print("    [HITL] 审批中断，跳过写操作。")
            else:
                print("    [HITL 挂起] 默认未批准写操作（可添加 -i / --interactive 交互审批或通过 API resume）。")
        print("\n" + "=" * 60 + "\n")
        print(state.get("final_report"))
        return 0

    state = engine.diagnose(query, user_id=args.user, hitl_approved=False)

    if state.get("candidate_actions") and not state.get("hitl_approved") and is_interactive:
        act = state["candidate_actions"][0]
        prompt = (
            f"\n[HITL 审批确认] 诊断完成，提议执行补救动作：\n"
            f"  动作: {act.get('description', '')}\n"
            f"  参数: {act.get('arguments', {})}\n"
            f"是否批准执行? [y/N]: "
        )
        try:
            choice = input(prompt).strip().lower()
            if choice in ("y", "yes"):
                state = engine.diagnose(
                    query,
                    user_id=args.user,
                    thread_id=state.get("thread_id"),
                    hitl_approved=True,
                )
        except (EOFError, KeyboardInterrupt):
            pass

    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
        return 0

    if args.explain:
        print(f"query       : {query}")
        print(f"intent      : {state.get('intent')}")
        print(f"blocked     : {state.get('blocked')}")
        print(f"reflections : {state.get('reflections')}")
        print(f"root_cause  : {state.get('root_cause')}")
        print(f"hypotheses  : {len(state.get('hypotheses', []))}")
        print(f"actions     : {len(state.get('candidate_actions', []))}")
        print()

    if state.get("blocked"):
        print(f"[blocked] {state.get('block_reason')}", file=sys.stderr)

    print(state.get("final_report"))
    return 0


def _eval_diag_cli(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from insight_agent.app.graph import InsightGraphEngine
    from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
    from insight_agent.app.redis_store import RedisMemoryStore
    from insight_agent.data.cn_retail_db import connect_cn_retail, ensure_cn_retail_database
    from insight_agent.data.olist_db import connect_olist
    from insight_agent.eval.diag_bench import (
        CN_DIAG_BENCHMARK_CASES,
        DIAG_BENCHMARK_CASES,
        build_diag_split,
        evaluate_with_sage_gate,
        run_diag_eval,
    )

    is_cn = getattr(args, "dataset", "cn_retail") == "cn_retail"
    if is_cn:
        ensure_cn_retail_database()
        conn = connect_cn_retail()
        cases = CN_DIAG_BENCHMARK_CASES
        title = "CN-Retail-Diag-Bench (50 Causal Anomaly Cases)"
    else:
        conn = connect_olist()
        cases = DIAG_BENCHMARK_CASES
        title = "Olist-Diag-Bench (50 Causal Anomaly Cases)"

    print("================================================================================")
    print(f"                {title}                      ")
    print("================================================================================")
    split = build_diag_split(cases)
    print(f"Manifest SHA256: {split.dataset_hash}")
    print(f"Splits: Train={len(split.train)}, Dev={len(split.dev)}, Test={len(split.test)}")
    print("Executing benchmark evaluation across all 50 cases...\n")

    retriever = OlistMultiModalRetriever(conn)
    redis_store = RedisMemoryStore()
    engine = InsightGraphEngine(conn, retriever, redis_store)

    report = run_diag_eval(engine, cases)

    print(f"Total Cases Tested     : {report.total_cases}")
    print(f"Passed Cases           : {report.passed_cases}")
    print(f"Overall Accuracy       : {report.accuracy * 100:.1f}%")
    print(f"Attribution Hit@1 Rate : {report.cause_hit_rate * 100:.1f}%")
    print(f"Security Block Rate    : {report.security_block_rate * 100:.1f}%")
    print(f"Average Latency        : {report.avg_latency_ms:.2f} ms")
    print(f"Total Reflections      : {report.total_reflections}")

    baseline = {c.id: True for c in cases}
    for fail_id in [f"diag_{i:03d}" for i in range(1, 11)]:
        baseline[fail_id] = False
    candidate = {r["case_id"]: r["passed"] for r in report.details}
    gate_decision = evaluate_with_sage_gate(baseline, candidate)

    print("\n--------------------------------------------------------------------------------")
    print("                       SAGE Statistical Acceptance Gate                         ")
    print("--------------------------------------------------------------------------------")
    print(f"Gate Decision          : {gate_decision.decision.value.upper()}")
    print(f"p-value (McNemar test) : {gate_decision.p_value:.6f}")
    print(
        f"Discordant Pairs       : +{gate_decision.improvements} improvements, -{gate_decision.regressions} regressions"
    )
    print(f"Rationale              : {gate_decision.reason}")
    print("================================================================================\n")

    if args.json:
        payload = asdict(report)
        payload["gate_decision"] = gate_decision.to_payload()
        print(json.dumps(payload, ensure_ascii=False, indent=2))


    return 0 if report.accuracy >= 0.90 else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.show_config:
        import json

        config = resolve_config(args)
        print(json.dumps(config.to_payload(), ensure_ascii=False, indent=2))
        reason = config.missing_reason()
        if reason:
            # stderr, so stdout stays parseable JSON and --show-config can be
            # consumed by a script.
            print(f"warning: {reason}; falling back to the rules planner", file=sys.stderr)
        return 0

    if args.serve_mcp or (args.question and args.question[0] == "mcp"):
        from insight_agent.app.tools import build_active_mcp_tool_set

        serve_tools(build_active_mcp_tool_set())
        return 0

    if args.eval_diag:
        return _eval_diag_cli(args)

    if not args.question:
        build_parser().print_help()
        return 1

    question_text = " ".join(args.question)
    from insight_agent.app.router import is_diagnostic_query

    is_anomaly_query = args.diagnose or is_diagnostic_query(question_text)
    if is_anomaly_query and not args.explain and not args.no_rag:
        return _diagnose_cli(args)

    return asyncio.run(_ask(args))



if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
