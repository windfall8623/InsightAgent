"""FastAPI surface: ask, explain, schema, agent discovery, and MCP-over-HTTP.

Kept thin on purpose — every route delegates to the orchestrator so the HTTP
layer holds no business logic and the evaluation harness can bypass it entirely.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from insight_agent.app.orchestrator import InsightOrchestrator, InsightRequest
from insight_agent.core.llm import ToolCall

try:  # pydantic is a hard dependency, but keep this module importable without it
    from pydantic import BaseModel, Field

    class AskRequest(BaseModel):
        question: str = Field(min_length=1)
        user_id: str = "anonymous"
        session_id: str | None = None
        thinking: bool = False

    class DiagnoseRequest(BaseModel):
        query: str = Field(min_length=1)
        user_id: str = "anonymous"
        thread_id: str | None = None
        hitl_approved: bool = False
        override_arguments: dict[str, Any] | None = None

    class ChatRequest(BaseModel):
        message: str = Field(min_length=1)
        user_id: str = "anonymous"
        session_id: str | None = None
        hitl_approved: bool = False
        override_arguments: dict[str, Any] | None = None

except ImportError:  # pragma: no cover
    AskRequest = None  # type: ignore[assignment,misc]
    DiagnoseRequest = None  # type: ignore[assignment,misc]
    ChatRequest = None  # type: ignore[assignment,misc]



def _get_active_dataset_conn() -> tuple[Any, str]:
    import os

    ds = os.environ.get("INSIGHT_DATASET", "cn_retail").lower().strip()
    if ds == "olist":
        from insight_agent.data.olist_db import connect_olist, ensure_olist_database

        ensure_olist_database()
        return connect_olist(), "olist"

    from insight_agent.data.cn_retail_db import connect_cn_retail, ensure_cn_retail_database

    ensure_cn_retail_database()
    return connect_cn_retail(), "cn_retail"


_ACTIVE_INVESTIGATIONS: dict[str, dict[str, Any]] = {}


def _resolve_diag_conn(agent_conn: Any, default_conn: Any) -> Any:
    """Select the appropriate database connection for LangGraph diagnostics.

    Prefers `agent_conn` if it already contains diagnostic tables (cn_orders or olist_orders).
    Falls back to `default_conn` (seeded active dataset) to avoid SQLite table missing errors.
    """
    if agent_conn is not None:
        try:
            diag_tables = [
                r[0]
                for r in agent_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('cn_orders', 'olist_orders')"
                )
            ]
            if diag_tables:
                return agent_conn
        except Exception:
            pass
    return default_conn


def create_app(orchestrator: InsightOrchestrator | None = None) -> Any:
    """Build the FastAPI application.

    ``AskRequest`` is declared at module scope rather than inside this factory:
    FastAPI resolves request models through ``typing.get_type_hints`` against the
    route function's module globals, and a locally-defined class is invisible
    there, which makes every request fail validation with 422.
    """

    from fastapi import FastAPI, HTTPException

    conn_db, ds_name = _get_active_dataset_conn()
    if orchestrator is None:
        from insight_agent.app.evolution import (
            EvolutionEngine,
            default_evolution_path,
            default_strategies,
        )
        from insight_agent.app.planner import build_planner
        from insight_agent.config import load_config
        from insight_agent.core.providers.gateway import PrefixCacheTracker

        prefix_tracker = PrefixCacheTracker()
        cfg = load_config()
        try:
            planner = build_planner(cfg)
        except Exception:
            from insight_agent.app.planner import RulesPlanner
            planner = RulesPlanner()
        agent = InsightOrchestrator(
            conn_db,
            planner=planner,
            evolution=EvolutionEngine(default_strategies(), path=default_evolution_path()),
            prefix_cache_tracker=prefix_tracker,
        )
    else:
        agent = orchestrator
        if hasattr(agent, "conn") and agent.conn is not None:
            try:
                tbls = [
                    r[0]
                    for r in agent.conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                ]
                if any(t.startswith("cn_") for t in tbls):
                    ds_name = "cn_retail"
                elif any(t in ("orders", "olist_orders") for t in tbls):
                    ds_name = "olist"
            except Exception:
                pass

    app = FastAPI(title="InsightAgent", version="0.2.0")
    app.state.orchestrator = agent
    app.state.prefix_cache_tracker = getattr(agent, "prefix_cache_tracker", None)
    app.state.gateway = getattr(agent, "gateway", None)

    from fastapi import Header as _Header

    def _require_token(authorization: str | None) -> None:
        """Gate endpoints that read or erase a user's stored profile.

        The memory routes expose one user's durable data to whoever names them
        in the path, so they need a shared secret. With no token configured the
        service is explicitly in local-development mode and the routes are
        refused outright rather than left open by default.
        """

        import os

        from fastapi import HTTPException

        expected = os.environ.get("INSIGHT_API_TOKEN", "").strip()
        if not expected:
            raise HTTPException(
                status_code=503,
                detail="memory endpoints disabled: set INSIGHT_API_TOKEN to enable",
            )
        supplied = (authorization or "").strip()
        if not supplied.startswith("Bearer ") or supplied[7:].strip() != expected:
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": app.version, "dataset": ds_name}

    @app.get("/schema")
    async def schema() -> dict[str, Any]:
        """Return the database schema the agent reasons over."""

        conn = agent.conn if hasattr(agent, "conn") and agent.conn is not None else conn_db
        tables: dict[str, Any] = {}
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for name in names:
            tables[name] = [f"{c[1]}:{c[2]}" for c in conn.execute(f'PRAGMA table_info("{name}")')]
        return {"tables": tables}

    @app.get("/.well-known/agent.json")
    async def agent_card() -> list[dict[str, Any]]:
        """A2A discovery document: every agent and the skills it advertises."""

        return agent.agent_cards

    async def _execute_diagnostic_investigation(p: AskRequest, request: InsightRequest) -> dict[str, Any]:
        if not hasattr(app.state, "graph_engine"):
            from insight_agent.app.graph import InsightGraphEngine
            from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
            from insight_agent.app.redis_store import RedisMemoryStore

            conn_agent = agent.conn if hasattr(agent, "conn") else None
            conn_diag = _resolve_diag_conn(conn_agent, conn_db)
            retriever = OlistMultiModalRetriever(conn_diag)
            redis_store = agent.memory.redis_store
            if redis_store is None:
                redis_store = RedisMemoryStore(local_only=True)
            app.state.graph_engine = InsightGraphEngine(
                conn_diag, retriever, redis_store, llm_client=agent.llm_client, orchestrator=agent,
            )

        engine = app.state.graph_engine
        state = await engine.a_diagnose(
            request.effective_question or request.question,
            original_query=request.question,
            user_id=p.user_id, thread_id=agent.session_key(request), hitl_approved=False,
            intent_decision=request.intent_decision,
        )
        if state.get("thread_id"):
            _ACTIVE_INVESTIGATIONS[state["thread_id"]] = state
        is_cn = getattr(engine, "is_cn", False)

        sql_findings = state.get("sql_findings", [])
        real_sql = ""
        real_rows: list[Any] = []
        real_columns: list[str] = []

        if sql_findings:
            root_cause = str(state.get("root_cause", "")).lower()
            selected_finding = sql_findings[-1]
            for finding in sql_findings:
                hyp = str(finding.get("hypothesis", "")).lower()
                if hyp and root_cause and (hyp in root_cause or root_cause in hyp):
                    selected_finding = finding
                    break
            real_sql = str(selected_finding.get("sql", "")).strip()
            raw_data = selected_finding.get("data", [])
            if isinstance(raw_data, list) and raw_data:
                real_rows = raw_data
                if isinstance(raw_data[0], dict):
                    real_columns = list(raw_data[0].keys())

        known_tables = (
            [
                "cn_orders",
                "cn_order_items",
                "cn_products",
                "cn_users",
                "cn_sellers",
                "cn_logistics",
                "cn_reviews",
                "cn_aftersales",
            ]
            if is_cn
            else [
                "olist_orders",
                "olist_order_reviews",
                "olist_order_items",
                "olist_products",
                "olist_customers",
                "olist_sellers",
                "olist_order_payments",
                "olist_geolocation",
                "product_category_name_translation",
            ]
        )
        real_evidence: list[dict[str, Any]] = []
        all_sqls = " ".join(str(f.get("sql", "")) for f in sql_findings) if sql_findings else real_sql
        for tbl in known_tables:
            if tbl in all_sqls:
                real_evidence.append({"kind": "table", "id": tbl})
        if not real_evidence:
            real_evidence.append({"kind": "table", "id": "cn_orders" if is_cn else "olist_orders"})

        for rev in state.get("review_evidence", [])[:4]:
            if isinstance(rev, dict):
                real_evidence.append(rev)

        tools_used = [
            "query_cn_retail_sql" if is_cn else "query_olist_sql",
            "search_customer_reviews",
        ]
        if state.get("candidate_actions"):
            tools_used.append("propose_remediation")
        if state.get("action_result"):
            tools_used.append(
                "issue_voucher"
                if "voucher" in str(state.get("action_result")).lower()
                else "execute_remediation"
            )

        agents_used = [
            "planner",
            "langgraph-investigator",
            "evidence-judge",
            "insight-analyst",
        ]
        if state.get("candidate_actions"):
            agents_used.append("action-proposer")
        if state.get("action_result"):
            agents_used.append("action-executor")

        return {
            "question": p.question,
            "answer": state.get("final_report", ""),
            "sql": real_sql,
            "columns": real_columns,
            "rows": real_rows,
            "row_count": len(real_rows),
            "evidence": real_evidence,
            "tools_used": tools_used,
            "agents_used": agents_used,
            "model": state.get("route_model", "langgraph-2.0"),
            "route_reason": "diagnostic_anomaly",
            "route_score": 3,
            "strategy": "langgraph_state_machine",
            "blocked": state.get("blocked", False),
            "block_reason": state.get("block_reason", ""),
            "reflections": state.get("reflections", 0),
            "alerts": state.get("alerts", []),
            "cost_usd": state.get("cost_usd", 0.05),
            "latency_ms": state.get("latency_ms", 120.0),
            "thread_id": state.get("thread_id", ""),
            "hitl_pending": bool(state.get("candidate_actions") and not state.get("hitl_approved")),
            "candidate_actions": state.get("candidate_actions", []),
            "action_result": state.get("action_result"),
        }

    async def _handle_general_chat(p: AskRequest, request: InsightRequest, on_token: Any = None) -> dict[str, Any]:
        started = time.perf_counter()
        llm_client = agent.llm_client
        active_model = getattr(llm_client, "model", "insight-chat")
        history = []
        if agent.config.use_memory and not (request.intent_decision and request.intent_decision.topic_switch):
            history = agent.memory.get_history_messages(agent.session_key(request))[-6:]

        import datetime

        now = datetime.datetime.now()
        weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]
        now_str = f"{now.year}年{now.month}月{now.day}日 {weekday_cn}"

        reply_text = ""
        if llm_client is not None:
            try:
                from insight_agent.core.llm import Message

                sys_prompt = (
                    f"当前系统真实时间为：{now_str}。\n"
                    "你是由 InsightAgent 2.0 驱动的智能 AI 助手。\n"
                    "面对用户的通用问答、聊天、常识咨询（如时间、日期、数学、常识探讨）或平台能力询问：\n"
                    "1. 请以亲切、自然、真实的中文直接回答用户的问题；\n"
                    "2. 如果用户是在问候或询问你的身份与能力，请自然介绍你作为电商数据智能体的核心能力（1. Text-to-SQL 智能数据查询；2. LangGraph 业务异动因果诊断；3. HITL 止损预案人机审批），并提供 2~3 个具体业务问题示例；\n"
                    "3. 如果是日常通用问题（如'今天是周几'、'讲个笑话'等），请直接给出准确、自然的回答；\n"
                    "注意：切勿生成任何 SQL 代码块。"
                )
                messages = [Message(role="system", content=sys_prompt), *history, Message(role="user", content=p.question)]
                is_thinking = getattr(p, "thinking", False)
                method = llm_client.complete_stream if hasattr(llm_client, "complete_stream") and on_token is not None else llm_client.complete
                parameters = inspect.signature(method).parameters
                options = {"thinking": is_thinking} if "thinking" in parameters or any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()) else {}
                if hasattr(llm_client, "complete_stream") and on_token is not None:
                    async with asyncio.timeout(40.0):
                        async for chunk in method(messages, (), **options):
                            if chunk.get("type", "content") == "content":
                                reply_text += str(chunk.get("delta") or "")
                else:
                    turn = await asyncio.wait_for(method(messages, (), **options), timeout=40.0)
                    if turn and turn.text and turn.text.strip():
                        reply_text = turn.text.strip()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Live LLM general chat error: %s", e)

        if not reply_text:
            if any(w in p.question for w in ("周", "星期", "几号", "日期", "时间")):
                reply_text = (
                    f"今天是 **{now_str}**。\n\n"
                    "我是 **InsightAgent 2.0** 电商智能数据助手。如果您需要查询电商数仓指标（如订单量统计、各品类退款率）或排查业务异动根因，随时可以向我提问！"
                )
            else:
                reply_text = (
                    "您好！我是 **InsightAgent 2.0**，您的企业级多模态电商数据分析与异常因果诊断专家。\n\n"
                    "我基于 **LangGraph 2.0 状态机** 与 **Text-to-SQL + BGE-M3 双路向量检索** 构建，能够为您提供以下核心服务：\n\n"
                    "1. **📊 智能数据分析 (Text-to-SQL)**\n"
                    "   - 支持多表关联、时序聚合下钻与维度筛选，自动编译生成安全只读 SQL 并可视化呈现。\n"
                    "   - *示例*：“统计2024年各月份华东地区的订单总数与总销售额”、“查询退款率最高的 Top 5 商品品类”\n\n"
                    "2. **🔍 业务异动因果诊断 (LangGraph Multi-Agent)**\n"
                    "   - 当业务指标异常时，自动检索数仓证据并深度对齐真实买家追评客诉，定位根本原因。\n"
                    "   - *示例*：“分析2024年7-8月华东江浙沪地区订单履约时效大幅延误与差评激增的原因”、“排查生鲜冷链品类退款率暴增的根因”\n\n"
                    "3. **🛡️ 止损预案与人机协同审批 (HITL)**\n"
                    "   - 诊断完成后自动输出补偿提议（如延误补偿券发放、承运商罚款），经人工审批后由分布式 Redis Redlock 保护执行。\n\n"
                    "请随时输入您关心的业务问题或分析需求！"
                )
        # Buffer before emitting so a sensitive value split across chunks is redacted.
        reply_text = agent.output_guard.scrub(reply_text)
        if on_token is not None:
            for i in range(0, len(reply_text), 15):
                await on_token({"stage": "delta", "delta": reply_text[i:i + 15], "type": "content"})

        elapsed = time.perf_counter() - started
        is_thinking = getattr(p, "thinking", False)
        return {
            "question": p.question,
            "answer": reply_text,
            "sql": "",
            "columns": [],
            "rows": [],
            "row_count": 0,
            "evidence": [],
            "tools_used": ["intent_router", "general_assistant"],
            "agents_used": ["intent_router", "conversational_assistant"],
            "model": f"{active_model} ({'🧠 深度思考' if is_thinking else '⚡ 极速模式'})",
            "route_reason": "chat_general (通用问候与平台引导)",
            "strategy": "conversational",
            "planner": "conversational",
            "blocked": False,
            "latency_ms": round(elapsed * 1000, 2),
            "alerts": [],
        }

    async def _dispatch_ask(payload: AskRequest, on_event: Any = None) -> dict[str, Any]:
        request = await agent.prepare_request(InsightRequest(
            question=payload.question, user_id=payload.user_id, session_id=payload.session_id,
        ))
        rejection = agent.request_rejection(request)
        if rejection is not None:
            result = rejection.to_payload()
            agent.record_response(request, result)
        else:
            decision = request.intent_decision
            if decision and decision.intent == "chat_general":
                if on_event is not None:
                    await on_event({"stage": "chat", "message": "正在生成通用问答..."})
                result = await _handle_general_chat(payload, request)
                agent.record_response(request, result)
            elif decision and decision.is_anomaly:
                if on_event is not None:
                    await on_event({"stage": "diagnosing", "message": "正在启动 LangGraph 因果诊断..."})
                try:
                    result = await _execute_diagnostic_investigation(payload, request)
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning("Diagnostic graph failed: %s", type(exc).__name__)
                    result = {
                        "question": payload.question, "answer": "诊断暂时未能完成，请稍后重试。",
                        "sql": "", "rows": [], "columns": [], "row_count": 0,
                        "blocked": False, "route_reason": "diagnostic_anomaly",
                        "planning_diagnostics": {"stage": "diagnostic", "final_error": type(exc).__name__},
                    }
                    agent.record_response(request, result)
            else:
                async def forward_stage(event: dict[str, Any]) -> None:
                    if on_event is not None and event.get("stage") != "done":
                        await on_event(event)
                answer = await agent.run(request, on_event=forward_stage)
                result = answer.to_payload()
        result["answer"] = agent.output_guard.scrub(str(result.get("answer", "")))
        result["intent_decision"] = asdict(request.intent_decision) if request.intent_decision else {}
        if on_event is not None:
            await on_event({"stage": "done", "payload": result})
        return result

    @app.post("/ask")
    async def ask(payload: AskRequest) -> dict[str, Any]:
        return await _dispatch_ask(payload)

    @app.post("/ask/stream")
    async def ask_stream(payload: AskRequest):
        """Stream multi-agent stages, intermediate SQL, token deltas, and final payload via SSE."""
        from fastapi.responses import StreamingResponse

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def stream_token(chunk: dict[str, Any]) -> None:
            await queue.put(chunk)

        async def emit_answer_deltas(text: str) -> None:
            if not text:
                return
            chunk_size = 8
            for i in range(0, len(text), chunk_size):
                await queue.put({
                    "stage": "delta",
                    "delta": text[i : i + chunk_size],
                    "type": "content",
                })
                await asyncio.sleep(0.012)

        async def on_stage_event(evt: dict[str, Any]) -> None:
            if evt.get("stage") == "done" and "payload" in evt:
                ans = evt["payload"].get("answer", "")
                if ans:
                    await emit_answer_deltas(ans)
            await queue.put(evt)

        async def stream_worker() -> None:
            try:
                await _dispatch_ask(payload, on_event=on_stage_event)
            except Exception as exc:
                await queue.put({"stage": "error", "message": f"执行异常: {type(exc).__name__}"})
            finally:
                await queue.put(None)

        asyncio.create_task(stream_worker())

        async def sse_stream():
            while True:
                item = await queue.get()
                if item is None:
                    break
                event_type = item.get("stage", "message")
                data_str = json.dumps(item, ensure_ascii=False)
                yield f"event: {event_type}\ndata: {data_str}\n\n"

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/explain")
    async def explain(payload: AskRequest) -> dict[str, Any]:
        """Run a question and return the provenance trail behind the answer."""

        result = await agent.run(
            InsightRequest(
                question=payload.question,
                user_id=payload.user_id,
                session_id=payload.session_id,
            )
        )
        payload_out = result.to_payload()
        return {
            "agents_used": payload_out["agents_used"],
            "tools_used": payload_out["tools_used"],
            "evidence": payload_out["evidence"],
            "route_reason": payload_out["route_reason"],
            "model": payload_out["model"],
            "reflections": payload_out["reflections"],
            "alerts": payload_out["alerts"],
            "sql": payload_out["sql"],
        }

    @app.get("/mcp/tools")
    async def mcp_tools() -> dict[str, Any]:
        """MCP `tools/list` over HTTP, for clients that cannot use stdio."""

        return {"tools": agent.tools.schemas()}

    @app.post("/mcp/call")
    async def mcp_call(body: dict[str, Any]) -> dict[str, Any]:
        """MCP `tools/call` over HTTP, routed through the security policy."""

        name = body.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="'name' is required")
        outcome = await agent.tools.run(
            ToolCall(id="http-1", name=name, arguments=body.get("arguments") or {})
        )
        return {
            "content": [{"type": "text", "text": outcome.content}],
            "isError": outcome.is_error,
        }

    @app.get("/memory/{user_id}")
    async def get_memory(user_id: str, authorization: str | None = _Header(default=None)) -> dict[str, Any]:
        """Return the durable profile for a user."""
        from dataclasses import asdict

        _require_token(authorization)
        return asdict(agent.memory.profile(user_id))

    @app.post("/memory/{user_id}/forget")
    async def forget(user_id: str, authorization: str | None = _Header(default=None)) -> dict[str, str]:
        """US-5: let a user erase everything the agent remembers about them."""

        _require_token(authorization)
        agent.memory.forget(user_id)
        return {"status": "forgotten", "user_id": user_id}


    @app.post("/chat")
    async def chat(payload: ChatRequest) -> dict[str, Any]:
        """Unified conversational entrypoint routing through LangGraph state machine.

        Handles:
        - Natural language SQL statistics (Fast-path NL2SQL Subgraph via Orchestrator)
        - Schema exploration & field explanations (Schema Explain Subgraph)
        - Root-cause anomaly investigation (Deep-path Diagnostic Statechart with Secure-by-Default HITL)
        """
        from insight_agent.app.graph import InsightGraphEngine
        from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
        from insight_agent.app.redis_store import RedisMemoryStore

        conn_agent = agent.conn if hasattr(agent, "conn") else None
        conn_diag = _resolve_diag_conn(conn_agent, conn_db)
        retriever = OlistMultiModalRetriever(conn_diag)
        redis_store = agent.memory.redis_store or RedisMemoryStore(local_only=True)
        engine = InsightGraphEngine(
            conn_diag, retriever, redis_store, orchestrator=agent
        )

        state = await engine.a_diagnose(
            payload.message,
            user_id=payload.user_id,
            thread_id=payload.session_id,
            hitl_approved=payload.hitl_approved,
            override_arguments=payload.override_arguments,
        )

        has_pending_actions = bool(state.get("candidate_actions")) and not bool(state.get("hitl_approved"))
        status = (
            "pending_hitl_approval"
            if has_pending_actions
            else ("blocked" if state.get("blocked") else "completed")
        )

        return {
            "status": status,
            "thread_id": state.get("thread_id"),
            "intent": state.get("intent"),
            "answer": state.get("final_report"),
            "sql": state.get("sql"),
            "columns": state.get("columns", []),
            "rows": state.get("rows", []),
            "row_count": state.get("row_count", len(state.get("rows", []))),
            "root_cause": state.get("root_cause"),
            "hypotheses": state.get("hypotheses", []),
            "candidate_actions": state.get("candidate_actions", []),
            "action_result": state.get("action_result"),
            "hitl_approved": state.get("hitl_approved", False),
            "reflections": state.get("reflections", 0),
            "blocked": state.get("blocked", False),
            "block_reason": state.get("block_reason", ""),
            "resume_url": f"/investigation/{state.get('thread_id')}/resume" if has_pending_actions else None,
            "latency_ms": state.get("latency_ms", 0.0),
            "cost_usd": state.get("cost_usd", 0.0),
        }

    @app.post("/diagnose")
    async def diagnose(payload: DiagnoseRequest) -> dict[str, Any]:
        """Run full LangGraph diagnostic investigation."""
        from insight_agent.app.graph import InsightGraphEngine
        from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
        from insight_agent.app.redis_store import RedisMemoryStore

        conn_agent = agent.conn if hasattr(agent, "conn") else None
        conn_diag = _resolve_diag_conn(conn_agent, conn_db)
        retriever = OlistMultiModalRetriever(conn_diag)
        redis_store = agent.memory.redis_store or RedisMemoryStore(local_only=True)
        engine = InsightGraphEngine(
            conn_diag, retriever, redis_store, orchestrator=agent
        )

        state = await engine.a_diagnose(
            payload.query,
            user_id=payload.user_id,
            thread_id=payload.thread_id,
            hitl_approved=payload.hitl_approved,
            override_arguments=payload.override_arguments,
        )
        has_pending_actions = bool(state.get("candidate_actions")) and not bool(state.get("hitl_approved"))
        status = (
            "pending_hitl_approval"
            if has_pending_actions
            else ("blocked" if state.get("blocked") else "completed")
        )
        return {
            "status": status,
            "thread_id": state.get("thread_id"),
            "intent": state.get("intent"),
            "root_cause": state.get("root_cause"),
            "final_report": state.get("final_report"),
            "hypotheses": state.get("hypotheses"),
            "candidate_actions": state.get("candidate_actions"),
            "action_result": state.get("action_result"),
            "hitl_approved": state.get("hitl_approved", False),
            "reflections": state.get("reflections"),
            "blocked": state.get("blocked"),
            "resume_url": f"/investigation/{state.get('thread_id')}/resume" if has_pending_actions else None,
        }

    @app.post("/diagnose/stream")
    async def diagnose_stream(payload: DiagnoseRequest) -> Any:
        """Stream intermediate diagnostic events via SSE (Server-Sent Events)."""
        import asyncio
        import json

        from fastapi.responses import StreamingResponse

        from insight_agent.app.graph import InsightGraphEngine
        from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
        from insight_agent.app.redis_store import RedisMemoryStore

        conn_agent = agent.conn if hasattr(agent, "conn") else None
        conn_diag = _resolve_diag_conn(conn_agent, conn_db)
        retriever = OlistMultiModalRetriever(conn_diag)
        redis_store = agent.memory.redis_store or RedisMemoryStore(local_only=True)
        engine = InsightGraphEngine(
            conn_diag, retriever, redis_store, orchestrator=agent
        )

        async def event_generator() -> Any:
            yield f"data: {json.dumps({'event': 'started', 'query': payload.query}, ensure_ascii=False)}\n\n"

            await asyncio.sleep(0.01)
            state = await engine.a_diagnose(
                payload.query,
                user_id=payload.user_id,
                thread_id=payload.thread_id,
                hitl_approved=payload.hitl_approved,
                override_arguments=payload.override_arguments,
            )
            yield f"data: {json.dumps({'event': 'intent_classified', 'intent': state.get('intent')}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'event': 'hypotheses_evaluated', 'reflections': state.get('reflections')}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'event': 'root_cause_confirmed', 'root_cause': state.get('root_cause')}, ensure_ascii=False)}\n\n"

            has_pending = bool(state.get("candidate_actions")) and not bool(state.get("hitl_approved"))
            if has_pending:
                tid = state.get("thread_id")
                resume_url = f"/investigation/{tid}/resume"
                pending_payload = {
                    "event": "hitl_pending",
                    "thread_id": tid,
                    "candidate_actions": state.get("candidate_actions"),
                    "resume_url": resume_url,
                }
                yield f"data: {json.dumps(pending_payload, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'event': 'completed', 'report': state.get('final_report')}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/investigation/{thread_id}/resume")
    async def resume_investigation(thread_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """HITL resume with parameter overrides (e.g. override voucher amount)."""
        from insight_agent.app.graph import InsightGraphEngine
        from insight_agent.app.multimodal_rag import OlistMultiModalRetriever
        from insight_agent.app.redis_store import RedisMemoryStore

        conn_agent = agent.conn if hasattr(agent, "conn") else None
        conn_diag = _resolve_diag_conn(conn_agent, conn_db)
        retriever = OlistMultiModalRetriever(conn_diag)
        redis_store = agent.memory.redis_store or RedisMemoryStore(local_only=True)
        engine = InsightGraphEngine(
            conn_diag, retriever, redis_store, orchestrator=agent
        )

        overrides = body.get("override_arguments", {})
        saved_state = _ACTIVE_INVESTIGATIONS.get(thread_id)
        if saved_state:
            saved_state["hitl_approved"] = True
            if overrides:
                saved_state["override_arguments"] = overrides
                if "amount" in overrides and saved_state.get("candidate_actions"):
                    saved_state["candidate_actions"][0]["amount"] = float(overrides["amount"])
            res = engine.action_executor_node(saved_state)
            saved_state["action_result"] = res.get("action_result")
            report_out = engine.insight_analyst_node(saved_state)
            saved_state["final_report"] = report_out.get("final_report")
            return {
                "thread_id": thread_id,
                "status": "resumed",
                "action_result": saved_state.get("action_result"),
                "final_report": saved_state.get("final_report"),
            }

        query = body.get("query", "继续执行排障诊断")
        state = await engine.a_diagnose(
            query,
            thread_id=thread_id,
            hitl_approved=True,
            override_arguments=overrides,
        )
        return {
            "thread_id": thread_id,
            "status": "resumed",
            "action_result": state.get("action_result"),
            "final_report": state.get("final_report"),
        }

    # Mount modern React frontend SPA if built
    # Locate frontend dist directory (supports both Docker container /app/frontend/dist and local repo)
    dist_path = Path("/app/frontend/dist")
    if not dist_path.exists():
        dist_path = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

    if dist_path.exists():
        from starlette.responses import FileResponse
        from starlette.staticfiles import StaticFiles

        assets_path = dist_path / "assets"
        if assets_path.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

        @app.get("/", include_in_schema=False)
        async def root_spa() -> Any:
            return FileResponse(dist_path / "index.html")

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon_spa() -> Any:
            fav = dist_path / "favicon.svg"
            if fav.exists():
                return FileResponse(fav)
            return FileResponse(dist_path / "index.html")

    return app


app = None  # populated lazily by `get_app` so importing this module is cheap


def get_app() -> Any:
    """Return the ASGI app (used by uvicorn: ``insight_agent.api:get_app``)."""

    global app
    if app is None:
        app = create_app()
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("insight_agent.api:get_app", host="0.0.0.0", port=8000, factory=True, reload=True)
