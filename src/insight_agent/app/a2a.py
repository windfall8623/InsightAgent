"""Agent-to-Agent delegation over an A2A-shaped message/task protocol.

Implemented against the A2A concepts that matter — **Agent Cards** advertising
capabilities, **Tasks** with a lifecycle and artefacts, and **message parts** —
rather than by depending on the SDK, so the protocol round-trip is testable
without a network.

Layering note: MCP exposes *tools*; this exposes *agents*. A client that can
call `list_tables` over MCP still needs something to decide what to ask and how
to interpret the answer — that is the gap this module fills.
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class TaskState(StrEnum):
    """A2A task lifecycle states."""

    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class AgentCard:
    """A capability advertisement: what this agent can be asked to do."""

    name: str
    description: str
    skills: list[str]
    url: str = "local://insight-agent"
    version: str = "0.2.0"
    capabilities: dict[str, bool] = field(
        default_factory=lambda: {"streaming": False, "pushNotifications": False}
    )

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["@context"] = "https://a2a-protocol.org/specification"
        payload["protocolVersion"] = "0.3.0"
        return payload


@dataclass(slots=True)
class Task:
    """A unit of delegated work between agents."""

    id: str
    skill: str
    input: dict[str, Any]
    state: TaskState = TaskState.SUBMITTED
    output: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "skill": self.skill,
            "input": self.input,
            "state": self.state.value,
            "output": self.output,
            "artifacts": [{"parts": [{"text": self.output.get("text", "")}]}]
            if self.output
            else [],
            "history": self.history,
        }


class AgentRegistry:
    """Directory of agents addressable by skill."""

    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}
        self._handlers: dict[str, Callable[[Task], Any]] = {}

    def publish(self, card: AgentCard, handler: Callable[[Task], Any]) -> None:
        """Register an agent and the handler that runs each of its skills.

        Handlers may be sync or async; agents that need to call tools are
        async because tool execution is.
        """

        self._cards[card.name] = card
        for skill in card.skills:
            self._handlers[skill] = handler

    def discover(self, skill: str) -> AgentCard | None:
        """Find which agent advertises a skill — the A2A discovery step."""

        for card in self._cards.values():
            if skill in card.skills:
                return card
        return None

    def cards(self) -> list[dict[str, Any]]:
        return [card.to_json() for card in self._cards.values()]

    async def delegate(self, skill: str, payload: dict[str, Any]) -> Task:
        """Create and execute a task for ``skill``."""

        card = self.discover(skill)
        if card is None:
            task = Task(id=new_task_id(), skill=skill, input=payload, state=TaskState.FAILED)
            task.output = {"error": f"no agent advertises skill {skill!r}"}
            return task

        task = Task(id=new_task_id(), skill=skill, input=payload, state=TaskState.WORKING)
        task.history.append({"role": "user", "parts": [{"text": str(payload.get("question", ""))}]})
        try:
            handler = self._handlers[skill]
            result = handler(task)
            if inspect.isawaitable(result):
                result = await result
            task.output = result
            task.state = TaskState.COMPLETED
        except Exception as exc:
            task.state = TaskState.FAILED
            task.output = {"error": f"{type(exc).__name__}: {exc}"}
        return task


def new_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:12]}"


# --- The four agents ------------------------------------------------------

PLANNER_CARD = AgentCard(
    name="planner",
    description="Decomposes an analytics question into an executable plan.",
    skills=["plan.decompose", "plan.clarify"],
)

SCHEMA_CARD = AgentCard(
    name="schema-explorer",
    description="Retrieves the schema evidence a question needs, with provenance.",
    skills=["schema.retrieve"],
)

SQL_CARD = AgentCard(
    name="sql-generator",
    description="Compiles a plan into safe read-only SQL and repairs it on failure.",
    skills=["sql.compile", "sql.repair"],
)

INSIGHT_CARD = AgentCard(
    name="insight-analyst",
    description="Summarises a result set into a business-readable answer.",
    skills=["insight.summarize"],
)
