"""Three-layer memory: conversational, durable user profile, and episodic.

The layers answer different questions and expire differently:

* **Short-term** — what is being asked right now; scoped to one session.
* **Long-term profile** — durable facts about a user ("only East region"),
  stored as key/value preferences that are injected into every future query.
* **Episodic** — past successful queries, retrievable by similarity so the agent
  can reuse a known-good SQL shape instead of re-deriving it.

Durable state is JSON on disk rather than Postgres, so the project stays
dependency-free while still persisting across process restarts.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from insight_agent.app.rag import tokenize
from insight_agent.core.llm import Message


@dataclass(slots=True)
class Episode:
    """A remembered successful interaction."""

    question: str
    sql: str
    result_summary: str = ""
    created_at: float = field(default_factory=time.time)

    def text(self) -> str:
        return f"{self.question} {self.sql}"


@dataclass(slots=True)
class UserProfile:
    """Durable preferences learned from a user's phrasing over time."""

    user_id: str
    filters: dict[str, str] = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> UserProfile:
        data = json.loads(raw)
        data["episodes"] = [Episode(**e) for e in data.get("episodes", [])]
        return cls(**data)


class MemoryStore:
    """Owns all three layers and the persistence behind them."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        episodic_limit: int = 200,
        redis_store: Any | None = None,
        local_only: bool = False,
    ) -> None:
        self.path = path
        self.episodic_limit = episodic_limit
        self._profiles: dict[str, UserProfile] = {}
        self._sessions: dict[str, list[str]] = {}
        self._session_states: dict[str, dict[str, Any]] = {}
        self.local_only = local_only
        self.redis_store = None if local_only else redis_store
        if self.redis_store is None and not local_only:
            try:
                from insight_agent.app.redis_store import RedisMemoryStore

                self.redis_store = RedisMemoryStore()
            except Exception:
                self.redis_store = None
        if path and path.exists():
            self._load()

    # -- short term -------------------------------------------------------

    def remember_turn(self, session_id: str, role: str, text: str) -> None:
        self._sessions.setdefault(session_id, []).extend([role, text])
        if self.redis_store is not None:
            with contextlib.suppress(Exception):
                self.redis_store.append_dialogue_turn(session_id, role, text)

    def session(self, session_id: str) -> list[str]:
        return list(self._sessions.get(session_id, []))

    def get_history_messages(self, session_id: str) -> list[Message]:
        """Convert short-term session dialogue turns into structured Message objects."""
        from insight_agent.core.llm import Message

        raw = self.session(session_id)
        messages: list[Message] = []
        for i in range(0, len(raw) - 1, 2):
            role, content = raw[i], raw[i + 1]
            if role in ("user", "assistant", "system"):
                messages.append(Message(role=role, content=content))
        return messages

    def get_session_state(self, session_id: str) -> dict[str, Any]:
        """Retrieve transient contextual state for an active session."""
        return self._session_states.setdefault(session_id, {})

    def get_task_history_messages(self, session_id: str) -> list[Message]:
        start = int(self.get_session_state(session_id).get("task_history_start", 0))
        return self.get_history_messages(session_id)[start:]

    def start_topic(self, session_id: str) -> None:
        state = self.get_session_state(session_id)
        state["task_history_start"] = len(self.get_history_messages(session_id))
        state.pop("standing_filters", None)
        state.pop("last_task", None)

    def record_interaction(
        self,
        session_id: str,
        question: str,
        answer: str,
        *,
        intent: str,
        status: str = "completed",
        effective_question: str | None = None,
        sql: str = "",
        reason: str = "",
    ) -> None:
        state = self.get_session_state(session_id)
        state["last_turn_status"] = status
        state["last_status"] = status
        if status == "blocked":
            # Refused instructions must not become a template for the next plan.
            state["last_refusal"] = {"reason": reason}
            return
        state.pop("last_refusal", None)
        if status != "completed":
            return
        self.remember_turn(session_id, "user", question)
        self.remember_turn(session_id, "assistant", answer)
        if intent == "chat_general":
            self.start_topic(session_id)
            state["last_intent"] = intent
            return
        state["last_intent"] = intent
        state["last_task"] = {
            "question": question,
            "effective_question": effective_question or question,
            "intent": intent,
            "status": status,
            "sql": sql,
        }

    def set_session_state(self, session_id: str, key: str, value: Any) -> None:
        """Store transient contextual state (e.g. active filters, previous plan)."""
        state = self.get_session_state(session_id)
        state[key] = value

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        if hasattr(self, "_session_states"):
            self._session_states.pop(session_id, None)

    # -- long term profile ------------------------------------------------

    def profile(self, user_id: str) -> UserProfile:
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
        return self._profiles[user_id]

    def set_filter(self, user_id: str, column: str, value: str) -> None:
        """Persist a default filter, e.g. ``city = 'Shanghai'``."""

        self.profile(user_id).filters[column] = value
        if self.redis_store is not None:
            with contextlib.suppress(Exception):
                self.redis_store.update_user_profile(user_id, {column: value})

    def add_fact(self, user_id: str, fact: str) -> None:
        profile = self.profile(user_id)
        if fact not in profile.facts:
            profile.facts.append(fact)

    def filters(self, user_id: str) -> dict[str, str]:
        return dict(self.profile(user_id).filters)

    # -- episodic ---------------------------------------------------------

    def remember(self, user_id: str, question: str, sql: str, summary: str = "") -> None:
        """Store a successful exchange, evicting the oldest beyond the cap."""

        profile = self.profile(user_id)
        profile.episodes.append(Episode(question=question, sql=sql, result_summary=summary))
        if len(profile.episodes) > self.episodic_limit:
            del profile.episodes[: len(profile.episodes) - self.episodic_limit]

    def recall(self, user_id: str, question: str, top_k: int = 3) -> list[Episode]:
        """Token-overlap recall of similar past queries.

        Deliberately simple and deterministic — Jaccard overlap over tokens —
        so the ablation study can show what vector recall would have to beat.
        """

        profile = self.profile(user_id)
        if not profile.episodes:
            return []

        q = set(tokenize(question))
        if not q:
            return []
        scored: list[tuple[float, Episode]] = []
        for episode in profile.episodes:
            e = set(tokenize(episode.text()))
            if not e:
                continue
            jaccard = len(q & e) / len(q | e)
            scored.append((jaccard, episode))
        scored.sort(key=lambda item: (-item[0], item[1].created_at))
        return [ep for score, ep in scored[:top_k] if score > 0]

    # -- context assembly -------------------------------------------------

    def build_context(self, user_id: str, session_id: str | None = None) -> str:
        """Render durable context to inject into the agent prompt."""

        profile = self.profile(user_id)
        lines: list[str] = []
        if session_id and self._sessions.get(session_id):
            lines.append("Recent conversation:")
            lines.extend(self._sessions[session_id][-4:])
        if profile.filters:
            rendered = ", ".join(f"{k} = {v!r}" for k, v in sorted(profile.filters.items()))
            lines.append(f"User's standing filters: {rendered}")
        if profile.facts:
            lines.append("Known user facts: " + "; ".join(profile.facts[-5:]))
        return "\n".join(lines)

    def forget(self, user_id: str) -> None:
        """Drop everything known about a user (the US-5 'clear my memory' path)."""

        self._profiles.pop(user_id, None)
        if self.redis_store is not None:
            with contextlib.suppress(Exception):
                self.redis_store.delete_user_profile(user_id)
        self.save()

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        assert self.path is not None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for user_id, raw in data.items():
            self._profiles[user_id] = UserProfile.from_json(raw)

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {uid: p.to_json() for uid, p in self._profiles.items()}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Preference extraction
# --------------------------------------------------------------------------

# "只看华东" / "我一般看上海" -> remember to filter by city.
_FILTER_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "city",
        re.compile(r"(只看|只关注|默认看|我一般看|限定).{0,6}?(上海|北京|深圳|广州|杭州|成都)"),
    ),
    ("member_level", re.compile(r"(只看|只关注|默认看).{0,6}?(gold|silver|bronze)", re.IGNORECASE)),
)

# The phrase is Chinese but the column stores the English city name.
_CITY_LITERALS = {
    "上海": "Shanghai",
    "北京": "Beijing",
    "深圳": "Shenzhen",
    "广州": "Guangzhou",
    "杭州": "Hangzhou",
    "成都": "Chengdu",
}


def extract_preferences(question: str) -> dict[str, str]:
    """Pull standing filter preferences out of a user's phrasing.

    Only high-confidence, explicit cues are captured; a wrong persistent
    preference is worse than a missing one because it silently biases later
    queries.
    """

    lowered = question.lower()
    found: dict[str, str] = {}
    for key, pattern in _FILTER_CUES:
        match = pattern.search(lowered)
        if not match:
            continue
        groups = match.groups()
        value = groups[-1] if groups else ""
        if not value:
            continue
        value = _CITY_LITERALS.get(value, value) if key == "city" else value.lower()
        found[key] = f"'{value}'"
    return found
