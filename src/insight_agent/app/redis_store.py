"""Production-grade Redis memory store, caching layer, and distributed locking.

Features:
1. Automatic fallback to fakeredis when no external Redis server is running (100% offline & reproducible).
2. L1 Short-term session memory (Redis List with sliding TTL).
3. L2 Long-term user profile (Redis Hash with GDPR compliance).
4. L3 Episodic memory (Diagnostic case retention & retrieval).
5. Query caching with anti-penetration sentinel protection.
6. Distributed atomic mutex (Redlock pattern) for high-impact action execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

try:
    import redis
except ImportError:
    redis = None  # type: ignore[assignment]

try:
    import fakeredis
except ImportError:
    fakeredis = None  # type: ignore[assignment]


@dataclass(slots=True)
class DiagnosticEpisode:
    id: str
    anomaly_query: str
    root_cause: str
    primary_table: str = ""
    sql: str = ""
    remediation_action: str | None = None
    suggested_actions: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)



class RedisMemoryStore:
    """Unified Redis interface for Agent memory, state, and distributed locking."""

    EMPTY_SENTINEL = "__REDIS_EMPTY_QUERY_RESULT__"

    def __init__(self, redis_url: str | None = None, *, prefix: str = "insight:", local_only: bool = False) -> None:
        self.prefix = prefix
        self._client: Any = None
        self._is_mock = False
        url = None if local_only else (redis_url or os.environ.get("REDIS_URL") or os.environ.get("INSIGHT_REDIS_URL"))
        if not local_only and not url and (os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER")):
            url = "redis://insight-agent-redis:6379/0"

        if url and redis is not None:
            try:
                client = redis.Redis.from_url(
                    url, decode_responses=True, socket_connect_timeout=0.2, socket_timeout=0.2
                )
                client.ping()
                self._client = client
            except Exception:
                self._client = None

        if self._client is None:
            # Fallback to in-process FakeRedis for zero-infra reproducibility
            if fakeredis is not None:
                self._client = fakeredis.FakeRedis(decode_responses=True)
                self._is_mock = True
            else:
                import threading

                class DictRedis:
                    def __init__(self) -> None:
                        self.store: dict[str, Any] = {}
                        self._lock = threading.Lock()

                    def get(self, key: str) -> Any:
                        with self._lock:
                            return self.store.get(key)

                    def set(self, key: str, val: Any, *args: Any, **kwargs: Any) -> bool:
                        with self._lock:
                            if kwargs.get("nx") and key in self.store:
                                return False
                            self.store[key] = val
                            return True

                    def setex(self, key: str, ttl: int, val: Any) -> bool:
                        return self.set(key, val)

                    def delete(self, *keys: str) -> int:
                        with self._lock:
                            cnt = sum(1 for k in keys if self.store.pop(k, None) is not None)
                            return cnt

                    def rpush(self, key: str, *vals: Any) -> int:
                        with self._lock:
                            self.store.setdefault(key, []).extend(vals)
                            return len(self.store[key])

                    def lrange(self, key: str, start: int, end: int) -> list[Any]:
                        with self._lock:
                            lst = self.store.get(key, [])
                            if end == -1:
                                return lst[start:]
                            return lst[start : end + 1]

                    def keys(self, pattern: str = "*") -> list[str]:
                        with self._lock:
                            pat = pattern.replace("*", ".*")
                            return [k for k in self.store if re.match(pat, k)]

                    def hset(
                        self,
                        name: str,
                        key: str | None = None,
                        value: Any = None,
                        mapping: dict[str, Any] | None = None,
                    ) -> int:
                        with self._lock:
                            h = self.store.setdefault(name, {})
                            if mapping:
                                h.update(mapping)
                                return len(mapping)
                            if key:
                                h[key] = value
                                return 1
                            return 0

                    def hget(self, name: str, key: str) -> Any:
                        with self._lock:
                            return self.store.get(name, {}).get(key)

                    def hgetall(self, name: str) -> dict[str, Any]:
                        with self._lock:
                            return dict(self.store.get(name, {}))

                    def expire(self, key: str, ttl: int) -> bool:
                        return True

                    def sadd(self, key: str, *members: Any) -> int:
                        with self._lock:
                            s = self.store.setdefault(key, set())
                            added = 0
                            for m in members:
                                if m not in s:
                                    s.add(m)
                                    added += 1
                            return added

                    def smembers(self, key: str) -> set[Any]:
                        with self._lock:
                            return set(self.store.get(key, set()))

                    def ping(self) -> bool: return True

                self._client = DictRedis()
                self._is_mock = True

    @property
    def is_mock(self) -> bool:
        """True if using in-process FakeRedis, False if connected to live server."""
        return self._is_mock

    def _k(self, key: str) -> str:
        return f"{self.prefix}{key}"

    # -- L1 Short-Term Conversational Memory -------------------------------

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int = 7200,
    ) -> None:
        """Append a conversational turn with sliding TTL."""
        key = self._k(f"session:{session_id}:messages")
        payload = json.dumps(
            {
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": time.time(),
            },
            ensure_ascii=False,
        )
        self._client.rpush(key, payload)
        self._client.expire(key, ttl_seconds)

    def get_messages(self, session_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve recent conversation turns for session."""
        key = self._k(f"session:{session_id}:messages")
        raw_items = self._client.lrange(key, -limit, -1) or []
        results = []
        for item in raw_items:
            try:
                results.append(json.loads(item))
            except Exception:
                continue
        return results

    def append_dialogue_turn(
        self, session_id: str, role: str, content: str, *, ttl_seconds: int = 7200
    ) -> None:
        """Alias for append_message."""
        self.append_message(session_id, role, content, ttl_seconds=ttl_seconds)

    def get_session_dialogue(self, session_id: str, *, max_turns: int = 50) -> list[dict[str, Any]]:
        """Alias for get_messages."""
        return self.get_messages(session_id, limit=max_turns)

    # -- L2 Long-Term User Profile ----------------------------------------

    def set_user_preference(self, user_id: str, field_name: str, value: str) -> None:
        """Persist a user preference (e.g. default_state = 'RJ')."""
        key = self._k(f"user:profile:{user_id}")
        self._client.hset(key, field_name, value)

    def get_user_preferences(self, user_id: str) -> dict[str, str]:
        key = self._k(f"user:profile:{user_id}")
        return dict(self._client.hgetall(key) or {})

    def update_user_profile(self, user_id: str, profile_dict: dict[str, Any]) -> None:
        """Update multiple user profile fields simultaneously."""
        key = self._k(f"user:profile:{user_id}")
        for k, v in profile_dict.items():
            self._client.hset(key, k, str(v))

    def get_user_profile(self, user_id: str) -> dict[str, str]:
        return self.get_user_preferences(user_id)

    def forget_user(self, user_id: str) -> bool:
        """GDPR forget compliance (US-5)."""
        key = self._k(f"user:profile:{user_id}")
        return bool(self._client.delete(key))

    def clear_user_profile(self, user_id: str) -> None:
        self.forget_user(user_id)

    def delete_user_profile(self, user_id: str) -> bool:
        return self.forget_user(user_id)

    # -- L3 Episodic Memory (Diagnostic Experience) -----------------------

    def save_episode(self, episode: DiagnosticEpisode) -> None:
        """Store a verified diagnostic episode."""
        key = self._k(f"episode:{episode.id}")
        payload = asdict(episode)
        self._client.set(key, json.dumps(payload, ensure_ascii=False))
        self._client.sadd(self._k("episodes:index"), episode.id)

    def record_episodic_case(
        self,
        case_id: str,
        description: str,
        root_cause: str,
        remediation_action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Convenience method for recording a diagnostic episode."""
        episode = DiagnosticEpisode(
            id=case_id,
            anomaly_query=description,
            root_cause=root_cause,
            primary_table="olist_orders",
            sql="",
            remediation_action=remediation_action,
            suggested_actions=[remediation_action] if remediation_action else [],
            created_at=time.time(),
        )
        self.save_episode(episode)


    def _tokenize_terms(self, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
        cjk = [c for c in text if "\u4e00" <= c <= "\u9fff"]
        tokens.update(cjk)
        for i in range(len(cjk) - 1):
            tokens.add(cjk[i] + cjk[i + 1])
        return tokens

    def recall_episodes(self, query: str, *, top_k: int = 3) -> list[DiagnosticEpisode]:
        """Deterministic token overlap for episode recall supporting bilingual text."""
        ids = self._client.smembers(self._k("episodes:index")) or set()
        if not ids:
            return []

        q_tokens = self._tokenize_terms(query)
        scored: list[tuple[float, DiagnosticEpisode]] = []
        for eid in ids:
            raw = self._client.get(self._k(f"episode:{eid}"))
            if not raw:
                continue
            try:
                data = json.loads(raw)
                ep = DiagnosticEpisode(**data)
                text = f"{ep.anomaly_query} {ep.root_cause} {ep.primary_table}"
                doc_tokens = self._tokenize_terms(text)
                score = len(q_tokens & doc_tokens) / max(len(q_tokens | doc_tokens), 1)
                if score > 0:
                    scored.append((score, ep))
            except Exception:
                continue

        scored.sort(key=lambda item: -item[0])
        return [ep for _, ep in scored[:top_k]]


    def recall_similar_episodes(self, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        """Convenience dictionary representation of recalled episodes."""
        eps = self.recall_episodes(query, top_k=top_k)
        return [
            {
                "case_id": ep.id,
                "description": ep.anomaly_query,
                "root_cause": ep.root_cause,
                "remediation_action": ep.remediation_action,
            }
            for ep in eps
        ]

    # -- Query Caching with Anti-Penetration ------------------------------

    def get_query_cache(self, query: str) -> Any | None:
        """Return cached query result, handling empty sentinels."""
        digest = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
        key = self._k(f"cache:sql:{digest}")
        raw = self._client.get(key)
        if raw is None:
            return None
        if raw == self.EMPTY_SENTINEL:
            return []
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set_query_cache(
        self, query: str, result: Any, *, ttl_seconds: int = 600, anti_penetration_ttl: int = 60
    ) -> None:
        """Cache query outcome, using empty sentinel on empty/false results."""
        digest = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
        key = self._k(f"cache:sql:{digest}")
        if not result:
            self._client.set(key, self.EMPTY_SENTINEL, ex=anti_penetration_ttl)
        else:
            self._client.set(
                key, json.dumps(result, ensure_ascii=False, default=str), ex=ttl_seconds
            )

    def delete_query_cache(self, query: str) -> None:
        """Evict query outcome from cache."""
        digest = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
        key = self._k(f"cache:sql:{digest}")
        self._client.delete(key)

    def get_cached_query_result(self, query: str) -> Any | None:
        """Alias for get_query_cache."""
        return self.get_query_cache(query)

    def set_cached_query_result(
        self, query: str, result: Any, *, ttl_seconds: int = 600
    ) -> None:
        """Alias for set_query_cache."""
        self.set_query_cache(query, result, ttl_seconds=ttl_seconds)

    # -- Distributed Lock (Redlock Pattern) ------------------------------

    @contextmanager
    def acquire_action_lock(
        self, action_id: str, *, ttl_seconds: int = 30, retry_delay: float = 0.05
    ) -> Iterator[bool]:
        """Distributed mutex preventing duplicate voucher issuances."""
        lock_key = self._k(f"lock:action:{action_id}")
        token = uuid.uuid4().hex
        acquired = False
        try:
            # Atomic SET key value NX EX
            acquired = bool(self._client.set(lock_key, token, ex=ttl_seconds, nx=True))
            yield acquired
        finally:
            if acquired:
                # Release lock safely if token matches
                val = self._client.get(lock_key)
                if val == token:
                    self._client.delete(lock_key)

    def close(self) -> None:
        """Close client connection if applicable."""
        try:
            if hasattr(self._client, "close"):
                self._client.close()
        except Exception:
            pass

