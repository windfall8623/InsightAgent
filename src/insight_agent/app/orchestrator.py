"""The orchestrator: four agents, three paradigms, one traceable run.

Paradigms and where each one lives:

* **Plan-and-Execute** — :meth:`InsightOrchestrator.run` plans first, then
  executes the plan's stages, instead of letting the model improvise each step.
* **ReAct** — inside the execute stage, each agent observes a tool result and
  decides the next action.
* **Reflection** — :meth:`InsightOrchestrator._validate` re-plans when the
  result fails a check, with a bounded retry count so a bad loop cannot run away.
* **Human-in-the-loop** — surfaced through the injected approver, so a
  high-impact query pauses the run instead of auto-approving.

Every stage is wrapped in a span, so a failed run can be read back to the
specific agent that misbehaved.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from insight_agent.app import a2a as a2a_proto
from insight_agent.app.context import CompressionStats, compress
from insight_agent.app.memory import MemoryStore, extract_preferences
from insight_agent.app.nl2sql import QueryPlan, compile_sql
from insight_agent.app.obs import (
    AnomalyDetector,
    Tracer,
    TraceRecord,
    TraceStore,
    UsageRecord,
)
from insight_agent.app.planner import (
    Planner,
    PlannerClientError,
    PlannerResult,
    RulesPlanner,
    data_domain,
    schema_hint,
)
from insight_agent.app.rag import HybridRetriever, SchemaDoc, build_retriever
from insight_agent.app.router import (
    IntentDecision,
    ModelRouter,
    ResponseCache,
    RouteDecision,
    classify_intent,
    classify_intent_async,
)
from insight_agent.app.security import (
    InputGuard,
    OutputGuard,
    PolicyEnforcingRegistry,
    SecurityDecision,
    ToolPolicy,
)
from insight_agent.app.tools import ToolContext, build_tool_set
from insight_agent.core.llm import Message, ToolCall
from insight_agent.core.providers.gateway import MultiProviderGateway, PrefixCacheTracker


@dataclass(slots=True)
class InsightRequest:
    question: str
    user_id: str = "anonymous"
    session_id: str | None = None
    effective_question: str | None = None
    intent_decision: IntentDecision | None = None


@dataclass(slots=True)
class InsightAnswer:
    """The end-to-end result, including the provenance needed for traceability."""

    question: str
    answer: str
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    agents_used: list[str] = field(default_factory=list)
    model: str = ""
    route_reason: str = ""
    strategy: str = ""
    planner: str = ""
    blocked: bool = False
    block_reason: str = ""
    latency_ms: float = 0.0
    reflections: int = 0
    alerts: list[dict[str, str]] = field(default_factory=list)
    cost_usd: float = 0.0
    trace_id: str = ""
    context_compressed: bool = False
    context_summary: str = ""
    route_score: int = 0
    planning_diagnostics: dict[str, Any] = field(default_factory=dict)
    intent_decision: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "evidence": self.evidence,
            "tools_used": self.tools_used,
            "agents_used": self.agents_used,
            "model": self.model,
            "route_reason": self.route_reason,
            "route_score": self.route_score,
            "strategy": self.strategy,
            "planner": self.planner,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "latency_ms": round(self.latency_ms, 2),
            "reflections": self.reflections,
            "alerts": self.alerts,
            "cost_usd": round(self.cost_usd, 6),
            "trace_id": self.trace_id,
            "context_compressed": self.context_compressed,
            "context_summary": self.context_summary,
            "planning_diagnostics": self.planning_diagnostics,
            "intent_decision": self.intent_decision,
        }


@dataclass(slots=True)
class RunConfig:
    """Knobs the evaluation harness toggles for ablations."""

    use_rag: bool = True
    use_routing: bool = True
    use_reflection: bool = True
    use_memory: bool = True
    retrieval_mode: str = "hybrid"
    max_reflections: int = 2
    top_k: int = 5
    # Tables handed to the planner once a table is retrieved its whole
    # schema is emitted, so this is cheap. Kept below the 7-table sandbox
    # deliberately: filtering a tiny catalogue removes tables the planner
    # legitimately needs. See docs/12 for why retrieval cannot pay off here.
    table_top_k: int = 6
    context_budget: int = 6


class InsightOrchestrator:
    """Coordinates the four agents to answer a question end to end."""

    def __init__(
        self,
        conn: Any,
        *,
        memory: MemoryStore | None = None,
        tracer: Tracer | None = None,
        router: ModelRouter | None = None,
        cache: ResponseCache | None = None,
        input_guard: InputGuard | None = None,
        output_guard: OutputGuard | None = None,
        policy: ToolPolicy | None = None,
        approver: Callable[[ToolCall, SecurityDecision], Any] | None = None,
        config: RunConfig | None = None,
        trace_sink: Path | None = None,
        retriever: HybridRetriever | None = None,
        evolution: Any = None,
        trace_store: TraceStore | None = None,
        planner: Planner | None = None,
        gateway: MultiProviderGateway | None = None,
        prefix_cache_tracker: PrefixCacheTracker | None = None,
    ) -> None:
        self.conn = conn
        self.memory = memory or MemoryStore()
        self.tracer = tracer or Tracer(sink=trace_sink)
        self.router = router or ModelRouter()
        self.cache = cache or ResponseCache()
        self.input_guard = input_guard or InputGuard()
        self.output_guard = output_guard or OutputGuard()
        self.policy = policy or ToolPolicy()
        self.approver = approver
        self.config = config or RunConfig()
        self.gateway = gateway
        self.prefix_cache_tracker = prefix_cache_tracker or PrefixCacheTracker()
        # Optional self-evolution layer. When present it resolves a per-question
        # strategy at the start of each run instead of using one fixed config.
        self.evolution = evolution
        # Planner decides question -> SQL. Defaults to the deterministic
        # engine so the project runs with no LLM configured.
        self.planner: Planner = planner or RulesPlanner()
        if hasattr(self.planner, "prefix_cache_tracker") and getattr(self.planner, "prefix_cache_tracker", None) is None:
            self.planner.prefix_cache_tracker = self.prefix_cache_tracker
        dataset_name = getattr(self.planner, "_dataset", None)
        try:
            table_names = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if "cn_orders" in table_names:
                dataset_name = "cn_retail"
            elif table_names & {"orders", "olist_orders", "users"}:
                dataset_name = "olist"
        except Exception:
            pass
        self.dataset = dataset_name or "unknown"
        if isinstance(self.planner, RulesPlanner) and getattr(self.planner, "_dataset", None) is None:
            self.planner._dataset = dataset_name

        # The deterministic planner, used whenever the router decides a question
        # is simple enough that a live model would be wasted spend.
        self._rules_planner: Planner = RulesPlanner(dataset=dataset_name)

        self._schema_hint: str | None = None
        self._domain_hint: str = ""
        self._last_recall: float | None = None
        # Optional run recorder; enables replay and OTLP export downstream.
        self.trace_store = trace_store
        self.retriever = retriever or build_retriever(conn)
        self.anomalies = AnomalyDetector()
        self.tool_log: list[str] = []

        self.ctx = ToolContext(conn=conn, retriever=self.retriever, max_rows=self.policy.max_rows)
        self.tools = PolicyEnforcingRegistry(
            build_tool_set(self.ctx), self.policy, approver=approver
        )
        self.a2a = a2a_proto.AgentRegistry()
        self._publish_agents()

    # -- A2A publishing ---------------------------------------------------

    def _publish_agents(self) -> None:
        """Register each agent with its card so delegation is discoverable."""

        self.a2a.publish(a2a_proto.PLANNER_CARD, self.agent_plan)
        self.a2a.publish(a2a_proto.SCHEMA_CARD, self.agent_schema)
        self.a2a.publish(a2a_proto.SQL_CARD, self.agent_sql)
        self.a2a.publish(a2a_proto.INSIGHT_CARD, self.agent_insight)

    @property
    def agent_cards(self) -> list[dict[str, Any]]:
        """The A2A discovery document served at ``/.well-known/agent.json``."""

        return self.a2a.cards()

    @property
    def llm_client(self) -> Any:
        return getattr(self.planner, "client", None) or self.gateway

    @staticmethod
    def session_key(request: InsightRequest) -> str:
        return request.session_id or f"anon-{request.user_id}"

    @contextlib.contextmanager
    def case_memory_scope(self, memory: MemoryStore) -> Iterator[InsightOrchestrator]:
        """Give an independent benchmark attempt its own mutable execution state."""
        child = InsightOrchestrator(
            self.conn,
            memory=memory,
            config=replace(self.config),
            planner=self.planner,
            retriever=self.retriever,
            router=self.router,
            input_guard=self.input_guard,
            output_guard=self.output_guard,
            policy=self.policy,
            approver=self.approver,
            gateway=self.gateway,
            prefix_cache_tracker=self.prefix_cache_tracker,
        )
        yield child

    def routing_context(self, request: InsightRequest) -> tuple[list[Message], dict[str, Any]]:
        if not self.config.use_memory:
            return [], {"dataset": self.dataset}
        session = self.session_key(request)
        state = dict(self.memory.get_session_state(session))
        task = state.get("last_task", {})
        context = {
            **task,
            **state,
            "dataset": self.dataset,
            "last_intent": state.get("last_intent", task.get("intent", "")),
            "last_status": state.get("last_turn_status", task.get("status", "")),
            "filters": dict(state.get("standing_filters", {})),
        }
        return self.memory.get_task_history_messages(session), context

    async def prepare_request(self, request: InsightRequest) -> InsightRequest:
        """Make one context-aware intent decision before choosing a product path."""
        history, context = self.routing_context(request)
        guard = self.input_guard.inspect(request.question, history=history, context=context)
        if guard.verdict.blocked:
            decision = IntentDecision(
                intent="prompt_injection", confidence=1.0, mode="input_guard",
                probabilities={"prompt_injection": 1.0}, reason=guard.verdict.reason,
            )
        else:
            decision = await classify_intent_async(
                request.question, history=history, context=context, client=self.llm_client,
            )
        return replace(
            request, intent_decision=decision,
            effective_question=decision.resolved_question or request.question,
        )

    def request_rejection(self, request: InsightRequest) -> InsightAnswer | None:
        history, context = self.routing_context(request)
        guard = self.input_guard.inspect(request.question, history=history, context=context)
        if not guard.verdict.blocked and request.effective_question:
            guard = self.input_guard.inspect(request.effective_question)
        decision = request.intent_decision
        if guard.verdict.blocked or (decision and decision.intent == "prompt_injection"):
            reason = guard.verdict.reason if guard.verdict.blocked else decision.reason
            return InsightAnswer(
                question=request.question,
                answer="无法处理该请求：不能绕过安全规则、泄露凭据或执行破坏性数据库操作。",
                blocked=True, block_reason=reason,
                planning_diagnostics={"stage": "input_guard", "final_error": reason, "repair_attempts": 0, "llm_error": False},
                intent_decision=asdict(decision) if decision else {},
            )
        if decision and decision.needs_clarification:
            return InsightAnswer(
                question=request.question,
                answer="抱歉，未能确定您要延续的指标或分析范围。请明确要查询的指标、对象或时间范围。",
                alerts=[{"kind": "clarification_needed", "message": decision.reason}],
                planning_diagnostics={"stage": "intent", "clarification_needed": True, "repair_attempts": 0, "llm_error": False},
                intent_decision=asdict(decision),
            )
        return None

    def record_response(self, request: InsightRequest, payload: dict[str, Any]) -> None:
        if not self.config.use_memory:
            return
        diagnostics = payload.get("planning_diagnostics", {})
        status = "blocked" if payload.get("blocked") else (
            "clarify" if diagnostics.get("clarification_needed") else (
                "failed" if diagnostics.get("final_error") else "completed"
            )
        )
        decision = request.intent_decision
        intent = decision.intent if decision else "simple_metric"
        session = self.session_key(request)
        if decision and decision.topic_switch and status == "completed":
            self.memory.start_topic(session)
        self.memory.record_interaction(
            session, self.output_guard.scrub(request.question),
            self.output_guard.scrub(str(payload.get("answer", ""))),
            intent=intent, status=status,
            effective_question=self.output_guard.scrub(request.effective_question or request.question),
            sql=str(payload.get("sql") or ""),
            reason=str(payload.get("block_reason") or ""),
        )

    # -- main entry -------------------------------------------------------

    async def run(
        self,
        request: InsightRequest,
        *,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> InsightAnswer:
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()
        cost_before = self.tracer.total_cost
        original_request = request

        async def _emit(evt: dict[str, Any]) -> None:
            if on_event is not None:
                try:
                    res = on_event(evt)
                    if inspect.isawaitable(res):
                        await res
                except Exception:
                    pass

        history, context = self.routing_context(request)
        if request.intent_decision is None:
            decision = classify_intent(
                request.question, history=history, context=context, local_only=True,
            )
            original_request = replace(
                request, intent_decision=decision,
                effective_question=request.effective_question or decision.resolved_question or request.question,
            )
        rejection = self.request_rejection(original_request)
        if rejection is not None:
            rejection.trace_id = trace_id
            rejection.latency_ms = (time.perf_counter() - started) * 1000
            self.record_response(original_request, rejection.to_payload())
            await _emit({"stage": "done", "payload": rejection.to_payload()})
            return rejection
        request = replace(
            original_request,
            question=original_request.effective_question or original_request.question,
        )
        session = self.session_key(request)
        decision = original_request.intent_decision
        topic_switch = bool(decision and decision.topic_switch)
        self.tool_log = []
        self.ctx.user_id = request.user_id
        self.ctx.session_id = session

        config = self.config
        strategy_name = "fixed"
        if self.evolution is not None and self.config.use_rag and self.config.use_memory:
            config, strategy_name = self.evolution.config_for(request.question)

        prior_messages = [] if topic_switch or not config.use_memory else history
        compressed_history: list[Message] = []
        compression_stats: CompressionStats | None = None
        context_summary = ""
        if prior_messages:
            compressed_history, compression_stats = compress(
                prior_messages, budget=config.context_budget, keep_system=True,
            )
            if compression_stats.summarised > 0:
                context_summary = f"Compressed {compression_stats.summarised} turns into summary"

        answer = InsightAnswer(
            question=original_request.question, answer="", trace_id=trace_id,
            strategy=strategy_name,
            intent_decision=asdict(decision) if decision else {},
            planning_diagnostics={"stage": "plan", "repair_attempts": 0, "llm_error": False},
        )
        prefs = extract_preferences(original_request.question) if config.use_memory else {}
        if self.dataset == "cn_retail" and "city" in prefs:
            city_names = {"Shanghai": "上海市", "Beijing": "北京市", "Shenzhen": "深圳市", "Guangzhou": "广州市", "Hangzhou": "杭州市", "Chengdu": "成都市"}
            value = prefs["city"].strip("'\"")
            prefs["city"] = city_names.get(value, value)
        if prefs:
            for column, value in prefs.items():
                self.memory.set_filter(request.user_id, column, value)
            self.memory.save()

        # Stage 1 — retrieve first. Retrieval is what the planner reasons over,
        # so running it afterwards could never have informed the plan.
        await _emit({"stage": "schema", "message": "正在检索数仓表元数据与关联证据..."})
        retrieved_context: str | None = None
        if config.use_rag:
            with self.tracer.span("retrieve", trace_id):
                schema_task = await self.a2a.delegate(
                    "schema.retrieve", {"question": request.question, "config": config}
                )
            if schema_task.state is not a2a_proto.TaskState.COMPLETED:
                answer.alerts.append(
                    {"kind": "agent_failure", "message": f"schema-explorer: {schema_task.output}"}
                )
            answer.evidence = schema_task.output.get("evidence", [])
            retrieved_context = str(schema_task.output.get("context") or "") or None
            self._last_recall = schema_task.output.get("recall")
            if retrieved_context is None:
                answer.alerts.append(
                    {
                        "kind": "retrieval_empty",
                        "message": "retriever returned no usable context; "
                        "planner will see the full schema instead",
                    }
                )

        # Stage 2 — route BEFORE planning. Difficulty has to be known before the
        # planner runs, otherwise the decision only labels a model that was
        # already used: a cheap question burns a frontier planner and an easy one
        # pays for RAG context it never needed. `retrieved_tables` comes from the
        # evidence we just collected, so this is the same signal the old
        # post-hoc call used -- only now it is available in time to act on.
        tables_in_context = len(answer.evidence) if config.use_rag else 0
        route = (
            self.router.route(request.question, retrieved_tables=tables_in_context)
            if config.use_routing
            else _always_large()
        )
        answer.model = route.model
        answer.route_reason = route.reason
        answer.route_score = route.score
        answer.context_compressed = bool(compression_stats and compression_stats.summarised > 0)
        answer.context_summary = context_summary
        route_reason = "routing disabled" if not config.use_routing else route.reason

        # Multi-turn contextual standing filters from session
        session_context = self.memory.get_session_state(session) if config.use_memory and not topic_switch else {}
        inherited_filters = dict(session_context.get("standing_filters", {}))
        user_filters = self.memory.filters(request.user_id) if config.use_memory else {}
        all_standing_filters = user_filters | inherited_filters

        # The route decides which planner runs. Tier "simple" goes to the
        # deterministic compiler; only genuinely complex questions pay for the
        # model-backed planner.
        planner, downgraded = self.planner_for(
            request.question, route, history=compressed_history, standing_filters=all_standing_filters
        )
        self.tracer.record_usage(
            UsageRecord(
                model=route.model,
                input_tokens=_rough_tokens(request.question) + 400 * len(answer.evidence),
                output_tokens=0,
                latency_ms=0.0,
                route_reason=route_reason,
            )
        )

        # Stage 2 — Plan-and-Execute: plan over the retrieved context.
        await _emit({"stage": "plan", "message": f"由智能体规划业务分析路径 ({route.model})..."})
        with self.tracer.span("plan", trace_id):
            plan_task = await self.a2a.delegate(
                "plan.decompose",
                {
                    "question": request.question,
                    "user_id": request.user_id,
                    "config": config,
                    "retrieved_context": retrieved_context,
                    "route_tier": route.tier,
                    "route_model": route.model,
                    "downgraded": downgraded,
                    "compressed_history": compressed_history,
                    "standing_filters": all_standing_filters,
                    "session_id": session,
                },
            )
        if config.use_rag:
            answer.agents_used.append("schema-explorer")
        answer.agents_used.append("planner")
        plan_output = plan_task.output
        answer.planner = str(plan_output.get("planner") or planner.name)
        answer.planning_diagnostics.update(plan_output.get("planning_diagnostics", {}))
        if plan_task.state is not a2a_proto.TaskState.COMPLETED:
            answer.planning_diagnostics["final_error"] = str(plan_output.get("error") or "planner failed")

        reflections = 0

        async def repair_query(problem: str, previous_sql: str) -> dict[str, Any] | None:
            nonlocal reflections
            if (
                not config.use_reflection
                or reflections >= max(0, config.max_reflections)
                or not callable(getattr(self.planner, "repair", None))
                or answer.planning_diagnostics.get("llm_error")
            ):
                return None
            if "attempts" not in answer.planning_diagnostics:
                answer.planning_diagnostics["attempts"] = [{"attempt": 0, **answer.planning_diagnostics}]
            reflections += 1
            answer.reflections = reflections
            answer.planning_diagnostics["repair_attempts"] = reflections
            await _emit({"stage": "repair", "message": f"正在修复查询（第 {reflections} 次）..."})
            with self.tracer.span("reflect", trace_id) as span:
                span.attributes["problem"] = problem
                task = await self.a2a.delegate(
                    "sql.repair",
                    {
                        "question": request.question,
                        "sql": previous_sql,
                        "problem": problem,
                        "route_tier": route.tier,
                        "route_model": route.model,
                        "compressed_history": compressed_history,
                        "standing_filters": all_standing_filters,
                        "session_id": session,
                    },
                )
            diagnostics = task.output.get("planning_diagnostics", {})
            attempts = answer.planning_diagnostics.setdefault("attempts", [])
            attempts.append({"attempt": reflections, "problem": self.output_guard.scrub(problem)[:600], **diagnostics})
            answer.planning_diagnostics.update(diagnostics)
            answer.planning_diagnostics["repair_attempts"] = reflections
            answer.planner = str(task.output.get("planner") or answer.planner)
            if task.state is not a2a_proto.TaskState.COMPLETED or task.output.get("error"):
                answer.planning_diagnostics["final_error"] = str(task.output.get("error") or "repair failed")
                return None
            return task.output

        planner_sql = str(plan_output.get("planner_sql") or "")
        while not planner_sql:
            parse_outcome = str(answer.planning_diagnostics.get("parse_reason") or "")
            if parse_outcome in {"unsafe_sql", "multiple_statements"}:
                answer.blocked = True
                answer.block_reason = parse_outcome
                answer.answer = "该请求或生成的查询不符合只读安全要求，已停止执行。"
                break
            if parse_outcome == "refusal":
                answer.answer = "抱歉，模型未能根据当前数据仓库提供此查询。请明确指标或调整到已有数据范围。"
                answer.planning_diagnostics["clarification_needed"] = True
                break
            problem = str(answer.planning_diagnostics.get("final_error") or "No executable SQL was produced")
            repaired = await repair_query(problem, str(plan_output.get("raw_planner_text") or ""))
            if repaired is None:
                if answer.planning_diagnostics.get("llm_error"):
                    answer.answer = "模型服务暂时未能完成查询，请稍后重试。"
                else:
                    clarify_task = await self.a2a.delegate("plan.clarify", {"question": original_request.question})
                    answer.answer = str(clarify_task.output.get("message") or "抱歉，未能识别要查询的指标，请明确指标及分析范围。")
                    answer.alerts.append({"kind": "clarification_needed", "suggestions": clarify_task.output.get("suggestions", [])})
                    answer.planning_diagnostics["clarification_needed"] = True
                break
            plan_output = repaired
            planner_sql = str(repaired.get("sql") or "")

        if not planner_sql:
            answer.latency_ms = (time.perf_counter() - started) * 1000
            answer.planning_diagnostics.setdefault("final_error", "no_sql")
            self.record_response(original_request, answer.to_payload())
            await _emit({"stage": "done", "payload": answer.to_payload()})
            return answer

        plan: QueryPlan | None = plan_output.get("plan")
        if plan and getattr(plan, "unsupported_entities", None):
            answer.alerts.append({"kind": "unsupported_entity", "message": f"数据仓库未收录实体：{', '.join(plan.unsupported_entities)}"})
        self.tracer.record_usage(
            UsageRecord(
                model=route.model, input_tokens=0,
                output_tokens=_rough_tokens(repr(plan) if plan else planner_sql),
                latency_ms=0.0, route_reason=route_reason,
            )
        )

        result_payload: dict[str, Any] = {}
        execution_succeeded = False
        while True:
            await _emit({"stage": "sql", "message": "已生成分析查询，正在执行安全只读 SQL...", "sql": planner_sql})
            with self.tracer.span("sql", trace_id) as span:
                sql_task = await self.a2a.delegate(
                    "sql.compile",
                    {
                        "question": request.question,
                        "plan": plan,
                        "config": config,
                        "planner_sql": planner_sql,
                        "user_id": request.user_id,
                    },
                )
                result_payload = sql_task.output
                span.attributes["sql"] = result_payload.get("sql", "")
            answer.agents_used.append("sql-generator")
            answer.sql = str(result_payload.get("sql") or planner_sql)
            if result_payload.get("blocked"):
                answer.blocked = True
                answer.block_reason = str(result_payload.get("reason") or "query blocked")
                answer.answer = str(result_payload.get("message") or "Query blocked by security policy.")
                answer.planning_diagnostics.update(stage="sql_guard", final_error=answer.block_reason)
                break
            problem = str(result_payload.get("error") or "")
            if sql_task.state is not a2a_proto.TaskState.COMPLETED and not problem:
                problem = "query stage failed"
            if not problem:
                problem = self._validate(result_payload) or ""
            if not problem:
                execution_succeeded = True
                answer.planning_diagnostics.update(stage="completed", final_error="", clarification_needed=False)
                await _emit({"stage": "sql_done", "message": f"SQL 执行成功，返回 {len(result_payload.get('rows', []))} 条记录", "sql": answer.sql})
                break
            answer.planning_diagnostics.update(stage="sql_execution", final_error=self.output_guard.scrub(problem)[:1000])
            while True:
                repaired = await repair_query(problem, planner_sql)
                if repaired is None:
                    answer.answer = f"未能完成查询（已修复 {reflections} 次）：{self.output_guard.scrub(problem)}"
                    break
                parse_outcome = str(answer.planning_diagnostics.get("parse_reason") or "")
                if parse_outcome in {"unsafe_sql", "multiple_statements"}:
                    answer.blocked = True
                    answer.block_reason = parse_outcome
                    answer.answer = "修复后的查询未通过只读安全检查，已停止执行。"
                    break
                if parse_outcome == "refusal":
                    answer.sql = ""
                    answer.answer = "抱歉，未能根据当前数据仓库完成查询，请明确指标或调整分析范围。"
                    answer.planning_diagnostics["clarification_needed"] = True
                    break
                candidate = str(repaired.get("sql") or "")
                if candidate:
                    planner_sql = candidate
                    plan = repaired.get("plan")
                    break
                problem = str(answer.planning_diagnostics.get("final_error") or "repair produced no SQL")
            if answer.answer or answer.blocked:
                break

        # Stage 3 — summarise only a successfully executed, validated result.
        if execution_succeeded and not answer.blocked and result_payload.get("rows") is not None:
            await _emit({"stage": "insight", "message": "分析师正在基于真实数据提炼商业洞察..."})
            with self.tracer.span("insight", trace_id):
                insight_task = await self.a2a.delegate(
                    "insight.summarize",
                    {"question": request.question, "result": result_payload, "plan": plan},
                )
            if insight_task.state is not a2a_proto.TaskState.COMPLETED:
                answer.alerts.append(
                    {
                        "kind": "agent_failure",
                        "message": f"insight-analyst: {insight_task.output}",
                    }
                )
            answer.answer = self.output_guard.scrub(insight_task.output.get("text", ""))
            answer.agents_used.append("insight-analyst")
            answer.columns = result_payload.get("columns", [])
            answer.rows = result_payload.get("rows", [])
            answer.row_count = len(answer.rows)

            if config.use_memory and answer.sql:
                self.memory.remember(request.user_id, request.question, answer.sql)
                self.memory.save()

        answer.tools_used = list(self.tool_log)
        answer.latency_ms = (time.perf_counter() - started) * 1000
        answer.cost_usd = max(0.0, self.tracer.total_cost - cost_before)
        self.anomalies.observe(
            tool_calls=len(answer.tools_used),
            blocked=1 if answer.blocked else 0,
            reflections=reflections,
            injection=False,
        )
        self.record_response(original_request, answer.to_payload())

        # Only completed tasks supply filters for subsequent turns.
        if execution_succeeded and plan is not None and config.use_memory:
            new_filters: dict[str, Any] = {}
            if getattr(plan, "filters", None):
                for f in plan.filters:
                    if " = " in f:
                        col, val = f.split(" = ", 1)
                        col = col.strip()
                        raw_val = val.strip()
                        clean_val = raw_val.strip("'\"")
                        new_filters[col] = raw_val
                        if col.endswith(".city") or col == "city":
                            new_filters["city"] = clean_val
                        elif col.endswith(".province") or col == "province":
                            new_filters["province"] = clean_val
                        elif col.endswith(".category") or col == "category":
                            new_filters["category"] = clean_val
                        elif col.endswith(".carrier") or col == "carrier":
                            if getattr(plan, "dimension", None) != "carrier" and getattr(plan, "metric", None) not in ("delivery_speed", "delay_rate"):
                                new_filters["carrier"] = clean_val
            if getattr(plan, "city_filter", None):
                new_filters["city"] = plan.city_filter
                new_filters["u.city"] = f"'{plan.city_filter}'"
            if getattr(plan, "province_filter", None):
                new_filters["province"] = plan.province_filter
            if getattr(plan, "category_filter", None):
                new_filters["category"] = plan.category_filter
            if getattr(plan, "carrier_filter", None):
                if getattr(plan, "dimension", None) != "carrier" and getattr(plan, "metric", None) not in ("delivery_speed", "delay_rate"):
                    new_filters["carrier"] = plan.carrier_filter
            if getattr(plan, "dimension", None) == "carrier" or getattr(plan, "metric", None) in ("delivery_speed", "delay_rate"):
                new_filters.pop("carrier", None)
                new_filters.pop("l.carrier", None)
            if getattr(plan, "start_time", None):
                new_filters["start_time"] = plan.start_time
                new_filters["end_time"] = plan.end_time
                new_filters["time_window"] = getattr(plan, "time_window", None)
            elif getattr(plan, "start", None):
                new_filters["start_time"] = plan.start
                new_filters["end_time"] = plan.end
                new_filters["time_window"] = getattr(plan, "time_window", None)
            if new_filters:
                self.memory.set_session_state(session, "standing_filters", new_filters)

        # Online self-evolution: observe outcome to adapt future routing strategies
        if self.evolution is not None and not answer.blocked:
            try:
                from insight_agent.app.evolution import Outcome, classify_shape

                shape_key = classify_shape(request.question).key
                outcome = Outcome(
                    case_id=request.question[:40],
                    shape=shape_key,
                    strategy=strategy_name,
                    passed=execution_succeeded,
                    recall=self._last_recall if self._last_recall is not None else 1.0,
                    latency_ms=answer.latency_ms,
                    cost_usd=answer.cost_usd,
                )
                self.evolution.observe(outcome)
                self.evolution.maybe_auto_evolve()
            except Exception:
                pass

        if self.trace_store is not None:
            self.trace_store.record(
                TraceRecord(
                    trace_id=answer.trace_id,
                    question=answer.question,
                    sql=answer.sql,
                    answer=answer.answer,
                    rows=answer.rows,
                    columns=answer.columns,
                    blocked=answer.blocked,
                    model=answer.model,
                    strategy=answer.strategy,
                    cost_usd=answer.cost_usd,
                    latency_ms=answer.latency_ms,
                    tools_used=list(answer.tools_used),
                    agents_used=list(answer.agents_used),
                )
            )
        self.tracer.flush_otlp()
        await _emit({"stage": "done", "payload": answer.to_payload()})
        return answer

    def planner_for(
        self,
        question: str,
        route: RouteDecision,
        history: Sequence[Message] | None = None,
        standing_filters: dict[str, Any] | None = None,
    ) -> tuple[Planner, bool]:
        """Pick the planner this question's difficulty actually justifies.

        A frontier planner only earns its cost on genuinely complex questions.
        When the configured planner is already the deterministic compiler there
        is nothing to downgrade; when a live client is configured, simple
        questions fall back to the compiler instead of paying for the model.

        Returns the planner and whether a fallback is warranted. Both travel in
        the task payload rather than on ``self``: concurrent requests share one
        orchestrator, so per-request state kept on the instance lets one
        question's decision leak into another's.
        """

        if isinstance(self.planner, RulesPlanner):
            return self.planner, False
        if route.tier == "simple" and self._rules_understands(question, history=history, standing_filters=standing_filters):
            return self._rules_planner, True
        return self.planner, False

    def _rules_understands(
        self,
        question: str,
        history: Sequence[Message] | None = None,
        standing_filters: dict[str, Any] | None = None,
    ) -> bool:
        """Whether the deterministic compiler should claim this question.

        Downgrading is only a saving when the cheap path is capable *and right*.
        The gate's strictness is set by ``INSIGHT_DOWNGRADE_MODE``:

        - ``strict`` (default): downgrade only single-table, non-rate aggregates
          (``RulesPlanner.downgrade_safe``). Multi-table JOINs and rate metrics --
          the compiler's main wrong-answer classes -- stay on the configured LLM.
        - ``loose``: legacy behaviour -- downgrade anything the compiler can parse
          (``RulesPlanner.probe``).
        - ``off``: never downgrade; the configured planner answers everything.
        """

        mode = os.environ.get("INSIGHT_DOWNGRADE_MODE", "strict").strip().lower()
        if mode == "off":
            return False

        method_name = "probe" if mode == "loose" else "downgrade_safe"
        try:
            probe_fn = getattr(self._rules_planner, method_name, None)
            if probe_fn is None:
                probe_fn = getattr(self._rules_planner, "probe", None)
            if probe_fn is None:
                return False
            import inspect

            sig = inspect.signature(probe_fn)
            kwargs: dict[str, Any] = {}
            if "schema_hint" in sig.parameters:
                kwargs["schema_hint"] = schema_hint(self.conn)
            if "domain" in sig.parameters:
                kwargs["domain"] = data_domain(self.conn)
            if "history" in sig.parameters or len(sig.parameters) >= 3:
                return bool(probe_fn(question, history=history, standing_filters=standing_filters, **kwargs))
            return bool(probe_fn(question, **kwargs))
        except Exception:
            return False

    async def _run_planner(
        self,
        question: str,
        *,
        user_id: str = "anonymous",
        retrieved_context: str | None = None,
        planner: Planner | None = None,
        route_tier: str | None = None,
        route_model: str | None = None,
        compressed_history: Sequence[Message] | None = None,
        standing_filters: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> PlannerResult:
        """Invoke the planner the route selected.

        When retrieval produced context, *that* is what the planner is given --
        not the full schema. This is what makes the layer retrieval rather than
        decoration: the model can only reference what the retriever surfaced,
        so turning retrieval off genuinely changes what can be answered.
        """

        planner = planner or self.planner
        consumes = getattr(planner, "consumes_schema_hint", True)
        if consumes and retrieved_context is None:
            if self._schema_hint is None:
                self._schema_hint = schema_hint(self.conn)
                self._domain_hint = data_domain(self.conn)
            retrieved_context = self._schema_hint
            self._last_recall = None

        # Episodic memory recall: inject previously validated question->SQL pairs for this user
        if self.config.use_memory and self.memory is not None and user_id:
            recalled_episodes = self.memory.recall(user_id, question, top_k=2)
            if recalled_episodes:
                exemplar_lines = [
                    f"-- Question: {ep.question}\n{ep.sql}"
                    for ep in recalled_episodes
                    if ep.sql
                ]
                if exemplar_lines:
                    exemplars_block = "\n-- Recalled Past Exemplars:\n" + "\n\n".join(exemplar_lines)
                    retrieved_context = (retrieved_context or "") + "\n" + exemplars_block

        return await planner.plan(
            question,
            schema_hint=retrieved_context or "",
            domain=self._domain_hint or (data_domain(self.conn) if consumes else ""),
            retrieved=retrieved_context is not None and bool(retrieved_context),
            standing_filters=standing_filters,
            compressed_history=compressed_history,
            route_tier=route_tier,
            model=route_model,
            session_id=session_id,
        )

    # -- agents -----------------------------------------------------------

    async def agent_plan(self, task: a2a_proto.Task) -> dict[str, Any]:
        """Planner: turn the question into a structured plan or clarify ambiguity.

        When the router downgraded this question to the deterministic compiler
        and the compiler cannot parse it -- the long tail it does not model --
        the configured planner gets a second attempt. Downgrading is a cost
        optimisation, never a capability limit.
        """

        question = str(task.input.get("question", ""))
        if task.skill == "plan.clarify":
            suggestions = [
                "按月份或省份查看大盘总销售额趋势",
                "查看不同承运商物流时效与延误率分布",
                "查询商品类目退款率与售后原因排行",
            ]
            msg = (
                f"抱歉，未能精确识别您在“{question}”中要查询的核心指标 (could not identify target metrics)。建议尝试以下常见分析方向：\n"
                + "\n".join(f"- {s}" for s in suggestions)
            )
            return {
                "clarification_needed": True,
                "question": question,
                "suggestions": suggestions,
                "message": msg,
                "text": msg,
            }
        retrieved_context = task.input.get("retrieved_context")
        user_id = str(task.input.get("user_id", "anonymous"))
        route_tier = task.input.get("route_tier")
        route_model = task.input.get("route_model")
        compressed_history = task.input.get("compressed_history")
        standing_filters = task.input.get("standing_filters")
        session_id = task.input.get("session_id")
        downgraded = bool(task.input.get("downgraded"))
        chosen = self._rules_planner if downgraded else self.planner
        try:
            result = await self._run_planner(
                question, user_id=user_id, retrieved_context=retrieved_context,
                planner=chosen, route_tier=route_tier, route_model=route_model,
                compressed_history=compressed_history, standing_filters=standing_filters,
                session_id=session_id,
            )
            if not result.understood and downgraded:
                chosen = self.planner
                result = await self._run_planner(
                    question, user_id=user_id, retrieved_context=retrieved_context,
                    planner=chosen, route_tier=route_tier, route_model=route_model,
                    compressed_history=compressed_history, standing_filters=standing_filters,
                    session_id=session_id,
                )
        except PlannerClientError as exc:
            return self._planner_failure(exc, "plan", chosen.name)
        return self._planner_payload(result, chosen.name, "plan")

    def _planner_failure(self, exc: PlannerClientError, stage: str, name: str) -> dict[str, Any]:
        error = self.output_guard.scrub(str(exc))[:1000]
        return {
            "understood": False, "planner": name, "error": error,
            "planning_diagnostics": {
                "stage": stage, "llm_error": True,
                "exception_type": type(exc.__cause__ or exc).__name__,
                "final_error": error,
            },
        }

    def _planner_payload(self, result: PlannerResult, name: str, stage: str) -> dict[str, Any]:
        notes = [self.output_guard.scrub(str(note))[:500] for note in result.notes[:12]]
        model_text = self.output_guard.scrub(result.text)[:2000]
        parse_reason = result.parse_outcome or ("accepted" if result.sql else "unrecognized")
        return {
            "plan": result.plan, "planner_sql": result.sql, "sql": result.sql,
            "planner": name, "tokens_out": result.tokens_out,
            "understood": result.understood, "notes": notes,
            "raw_planner_text": result.text,
            "text": result.sql,
            "planning_diagnostics": {
                "stage": stage, "parse_reason": parse_reason,
                "stop_reason": result.stop_reason, "notes": notes,
                "model_text": model_text, "llm_error": False,
                "final_error": self.output_guard.scrub(result.parse_error)[:1000],
            },
        }

    def agent_schema(self, task: a2a_proto.Task) -> dict[str, Any]:
        """Schema explorer: retrieve evidence, flagging what the plan needs."""

        question = str(task.input.get("question", ""))
        plan: QueryPlan | None = task.input.get("plan")
        query = question
        if plan is not None:
            query = f"{question} {plan.metric or ''} {plan.dimension or ''}"

        config = task.input.get("config") or self.config
        # Rank tables, not columns, then keep the column evidence for display.
        tables = self.retriever.search_tables(
            query, top_k=config.table_top_k, mode=config.retrieval_mode
        )
        keep = {name for name, _ in tables}
        hits: list[tuple[SchemaDoc, float]] = [
            (doc, score)
            for doc, score in self.retriever.search(
                query, top_k=config.top_k * 4, mode=config.retrieval_mode
            )
            if doc.table in keep
        ]
        needed = set(plan.tables) if plan is not None else set()
        evidence = [
            {**doc.to_evidence(score), "needed": doc.table in needed} for doc, score in hits
        ]
        return {
            "evidence": evidence,
            "tables": sorted(needed),
            # Rendered schema. Without this the planner is handed the whole
            # database instead, and retrieval affects nothing it produces.
            "context": render_context(hits, self.conn),
            "retrieved": bool(hits),
            "recall": (len(needed & keep) / len(needed) if needed else 1.0),
            "text": f"{len(hits)} docs",
        }

    async def agent_sql(self, task: a2a_proto.Task) -> dict[str, Any]:
        """SQL generator: compile, execute under policy, and report."""

        if task.skill == "sql.repair":
            problem = str(task.input.get("problem", ""))
            question = str(task.input.get("question", ""))
            repair = getattr(self.planner, "repair", None)
            try:
                if callable(repair):
                    result = await repair(
                        question, str(task.input.get("sql") or ""), problem,
                        schema_hint=schema_hint(self.conn), domain=data_domain(self.conn),
                        route_tier=task.input.get("route_tier"), model=task.input.get("route_model"),
                        compressed_history=task.input.get("compressed_history"),
                        standing_filters=task.input.get("standing_filters"),
                        session_id=task.input.get("session_id"),
                    )
                else:
                    result = await self._rules_planner.plan(
                        question,
                        standing_filters=task.input.get("standing_filters"),
                        schema_hint=schema_hint(self.conn),
                        domain=data_domain(self.conn),
                    )
            except PlannerClientError as exc:
                return self._planner_failure(exc, "repair", self.planner.name)
            return {
                **self._planner_payload(result, self.planner.name, "repair"),
                "repaired": bool(result.sql), "problem": problem,
            }

        plan: QueryPlan | None = task.input.get("plan")
        question = str(task.input.get("question", ""))
        override = task.input.get("sql_override")
        sql = str(override) if override else str(task.input.get("planner_sql") or "")

        if not sql:
            try:
                if plan is None:
                    raise ValueError("no query plan was produced")
                sql = compile_sql(plan)
            except ValueError as exc:
                return {
                    "error": str(exc),
                    "message": "I could not work out which number you are asking for.",
                }

        from insight_agent.app.sql_text import validate_readonly_sql

        validation = validate_readonly_sql(sql)
        if validation.outcome != "accepted":
            security_block = validation.outcome in {"unsafe_sql", "multiple_statements"}
            return {
                "sql": sql, "blocked": security_block,
                "reason": validation.error, "message": validation.error,
                "error": None if security_block else validation.error,
            }
        config = task.input.get("config") or self.config
        user_id = str(task.input.get("user_id") or self.ctx.user_id)
        cached = self.cache.get(user_id, sql) if config.use_memory else None
        if cached is not None:
            self.tool_log.append("run_sql(cache)")
            return {**cached, "cached": True}

        call = ToolCall(id=f"sql_{uuid.uuid4().hex[:8]}", name="run_sql", arguments={"sql": sql})
        self.tool_log.append("run_sql")
        outcome = await self.tools.run(call)

        if outcome.is_error:
            policy_denied = outcome.content.startswith(("BLOCKED:", "DECLINED:"))
            return {
                "sql": sql,
                "blocked": policy_denied,
                "reason": outcome.content,
                "message": (
                    "That request was blocked by the security policy."
                    if policy_denied
                    else f"The database rejected the query: {outcome.content}"
                ),
                "error": None if policy_denied else outcome.content,
            }

        payload = json.loads(outcome.content)
        if config.use_memory:
            self.cache.put(user_id, sql, payload)
        return {"sql": sql, **payload, "question": question, "cached": False}

    async def agent_insight(self, task: a2a_proto.Task) -> dict[str, Any]:
        """Insight analyst: describe the result set in business language."""

        result = task.input.get("result", {})
        plan: QueryPlan | None = task.input.get("plan")
        rows = result.get("rows", [])
        columns = result.get("columns", [])
        question = str(task.input.get("question", ""))

        # Without a structured plan (an LLM planner emits SQL only) fall back to
        # the column the query actually selected, so the sentence names a real
        # quantity instead of the word "value".
        metric = (plan.metric if plan else None) or ""
        metric_label = _METRIC_LABELS.get(metric) or _humanise(
            metric or _value_column(result.get("columns", [])) or "result"
        )

        if not rows:
            return {"text": "The query ran successfully but matched no rows."}

        # 1. LLM-powered natural business insight if an LLM client is available
        llm_client = self.llm_client

        unsupported = getattr(plan, "unsupported_entities", []) if plan else []
        unsupported_note = f"\n4. 特别说明：数据仓库当前暂未收录【{'、'.join(unsupported)}】的数据，请在回答末尾简要向用户说明暂无其数据。" if unsupported else ""

        if llm_client is not None and rows:
            try:
                sample_rows = rows[:10]
                summary_prompt = (
                    f"用户提问：{question}\n"
                    f"查询指标：{metric_label}\n"
                    f"查询得到的数据结果集（共 {len(rows)} 条）：\n"
                    f"{json.dumps(sample_rows, ensure_ascii=False)}\n\n"
                    "请作为专业零售电商数据分析师，用简炼明了的1-2句话直接回答用户的问题。"
                    "要求：\n"
                    "1. 直接给出核心结论（如谁更快、谁更高、差距是多少），对比核心指标数值；\n"
                    "2. 如果涉及配送时效，注意天数越小越快，并结合延误率说明稳定程度；\n"
                    f"3. 语言专业自然，严禁输出SQL或技术术语，严禁把订单量等非目标字段误当作指标数值。{unsupported_note}"
                )
                from insight_agent.core.llm import Message
                resp = await llm_client.complete(
                    [
                        Message(role="system", content="你是一位零售电商高级业务数据分析师，擅长直接针对业务问题从数据表中提炼准确结论。"),
                        Message(role="user", content=summary_prompt),
                    ],
                    tools=[],
                )
                if resp and getattr(resp, "text", None) and resp.text.strip():
                    return {"text": resp.text.strip()}
            except Exception:
                pass  # Fall back to high quality deterministic template

        # 2. High-quality deterministic fallback template
        value_column = _value_column(columns)
        dimension_column = _dimension_column(columns, value_column)

        if value_column is None:
            return {"text": f"针对问题“{question}”，数据查询共返回 {len(rows)} 条记录。"}

        def value_of(row: dict[str, Any]) -> float | None:
            return _as_float(row.get(value_column))

        def label_of(row: dict[str, Any]) -> str:
            if dimension_column is None:
                return ""
            return str(row.get(dimension_column, ""))

        unsupported_suffix = f"（注：数据仓库暂无【{'、'.join(unsupported)}】数据）" if unsupported else ""

        # Domain-aware insight for delivery speed: smaller days = faster!
        if metric == "delivery_speed" or value_column == "avg_delivery_days":
            valid_rows = [r for r in rows if value_of(r) is not None]
            valid_rows.sort(key=lambda r: value_of(r) or 0)
            if len(valid_rows) >= 2:
                fastest = valid_rows[0]
                second = valid_rows[1]
                f_name = label_of(fastest)
                f_days = value_of(fastest)
                s_name = label_of(second)
                s_days = value_of(second)
                diff = round((s_days or 0) - (f_days or 0), 2)
                f_delay = fastest.get("delay_rate_pct", 0)
                s_delay = second.get("delay_rate_pct", 0)
                delay_str = ""
                if f_delay is not None and s_delay is not None:
                    delay_str = f"，延误率为 {f_delay}%（对比【{s_name}】{s_delay}%）"
                text = (
                    f"【{f_name}】送得更快，平均送达时效为 {f_days} 天{delay_str}，"
                    f"比【{s_name}】（平均 {s_days} 天）快 {diff} 天。"
                )
                if len(valid_rows) > 2:
                    listed = "、".join(f"{label_of(r)}: {value_of(r)}天" for r in valid_rows)
                    text += f" 所有对比快递时效：{listed}。"
                if unsupported_suffix:
                    text += f" {unsupported_suffix}"
                return {"text": text}
            elif len(valid_rows) == 1:
                r = valid_rows[0]
                delay_str = f"，延误率为 {r.get('delay_rate_pct')}%" if "delay_rate_pct" in r else ""
                orders_str = f"，统计单量 {int(r.get('total_orders', 0)):,} 单" if "total_orders" in r else ""
                suffix = f" {unsupported_suffix}" if unsupported_suffix else ""
                return {
                    "text": f"【{label_of(r)}】的平均配送时效为 {value_of(r)} 天{delay_str}{orders_str}。{suffix}"
                }

        # Domain-aware insight for delay rate: lower = better!
        if metric == "delay_rate" or value_column == "delay_rate_pct":
            valid_rows = [r for r in rows if value_of(r) is not None]
            valid_rows.sort(key=lambda r: value_of(r) or 0)
            if len(valid_rows) >= 2:
                best = valid_rows[0]
                worst = valid_rows[-1]
                listed = "、".join(f"{label_of(r)}: {value_of(r)}%" for r in valid_rows[:5])
                return {
                    "text": f"物流延误率分析：【{label_of(best)}】表现最好，延误率仅为 {value_of(best)}%；【{label_of(worst)}】延误率最高，达 {value_of(worst)}%。主要数据：{listed}。"
                }

        if dimension_column is not None:
            head = f"{metric_label}已按维度统计，共包含 {len(rows)} 组数据。"
            top = max(
                (r for r in rows if value_of(r) is not None),
                key=lambda r: value_of(r) or 0,
                default=None,
            )
            if top is not None:
                head += f" 其中最高的是【{label_of(top)}】，数值为 {_format_number(value_of(top))}。"
            listed = "、".join(
                f"{label_of(r)}: {_format_number(value_of(r))}"
                for r in sorted(rows, key=lambda r: (value_of(r) is None, -(value_of(r) or 0)))[:5]
            )
            return {"text": f"{head} 主要分组数据：{listed}。" if listed else head}

        return {
            "text": f"根据业务数据分析，您所查询的问题（“{question}”）对应的【{metric_label}】为 {_format_number(value_of(rows[0]))}。"
        }

    # -- reflection validator ---------------------------------------------

    def _validate(self, payload: dict[str, Any]) -> str | None:
        """Return a validation failure without relaxing legitimate user filters."""
        if payload.get("error"):
            return str(payload["error"])
        if not isinstance(payload.get("rows"), list):
            return "query did not return a result set"
        return None


# --- helpers --------------------------------------------------------------

_METRIC_LABELS = {
    "amount": "销售总金额",
    "order_count": "订单总总量",
    "user_count": "独立购买用户数",
    "avg_order_value": "平均客单价",
    "quantity": "商品售出件数",
    "stock": "库存总量",
    "refund_rate": "订单退款率(%)",
    "refund_amount": "退款总金额",
    "refund_count": "退款订单数",
    "pay_success_rate": "支付成功率(%)",
    "delivery_speed": "物流配送时效",
    "delay_rate": "物流延误率",
    "sales_amount": "销售总额",
    "sales_quantity": "商品销售总量",
    "aftersales_metric": "售后退款指标",
}


# Aliases the rules engine emits. A hint for the summariser, never a
# requirement: an LLM planner names its aggregate column whatever it likes.
_KNOWN_VALUE_COLUMNS = (
    "avg_delivery_days",
    "delay_rate_pct",
    "metric",
    "refund_rate_pct",
    "refund_amount",
    "refund_count",
    "pay_success_rate_pct",
    "total_stock",
    "total_amount",
    "total_sales",
    "order_count",
)


def render_context(hits: Sequence[tuple[Any, float]], conn: Any = None) -> str:
    """Render retrieved schema docs into text the planner can actually use.

    **Table granularity, not column granularity.** Emitting only the columns
    that happened to match is the wrong unit: it produced a 90% smaller schema
    that silently removed ``orders.order_date`` and ``categories.name``, so the
    model dropped a time filter and referenced a table it could not see. Once a
    table is retrieved, every column of that table is safe and cheap to include,
    and nothing outside the retrieved tables is offered.
    """

    tables: dict[str, list[str]] = {}
    for doc, _score in hits:
        tables.setdefault(doc.table, [])

    if not tables:
        return ""

    if conn is not None:
        for table in list(tables):
            try:
                cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{table}")')]
                if cols:
                    tables[table] = cols
                    continue
            except Exception:  # noqa: BLE001 - fall back to retrieved columns
                pass

    lines: list[str] = []
    for table, columns in tables.items():
        uniq = list(dict.fromkeys(columns))
        lines.append(f"{table}({', '.join(uniq)})" if uniq else table)
    return chr(10).join(lines)


def _humanise(column: str) -> str:
    """Turn a column alias into something readable: total_sales -> Total sales."""

    words = [w for w in column.split("_") if w]
    if not words:
        return "Result"
    return " ".join(words).capitalize()


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _value_column(columns: Sequence[str]) -> str | None:
    """Pick the column holding the answer, by shape rather than by name.

    Prefers a known alias, then the last column: a grouped query puts the
    grouping key first and the aggregate last by convention.
    """

    for candidate in _KNOWN_VALUE_COLUMNS:
        if candidate in columns:
            return candidate
    return columns[-1] if columns else None


def _dimension_column(columns: Sequence[str], value_column: str | None) -> str | None:
    """Pick the grouping column when the query grouped by one."""

    if "dimension" in columns:
        return "dimension"
    others = [c for c in columns if c != value_column]
    return others[0] if len(columns) == 2 and others else None


def _metric_value(row: dict[str, Any]) -> float | None:
    """Value of a row whose column is already known."""

    for key in _KNOWN_VALUE_COLUMNS:
        if key in row and row[key] is not None:
            return _as_float(row[key])
    return None


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _always_large() -> Any:
    from insight_agent.app.router import RouteDecision

    return RouteDecision(model="sim-large", tier="complex", score=99, reason="routing_disabled")
