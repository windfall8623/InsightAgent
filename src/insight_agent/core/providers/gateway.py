"""Multi-provider LLM gateway, circuit breaker, key pool, and prompt prefix caching.

Provides enterprise-grade resilience for LLM serving:
1. CircuitBreaker: State machine (CLOSED -> OPEN -> HALF_OPEN) preventing cascading failures.
2. ProviderEndpoint: Configurable LLM service endpoint with key rotation.
3. MultiProviderGateway: Automatic failover, retry, and latency tracking.
4. PromptCacheOptimizer: Deterministic canonicalization for KV cache prefix reuse.
5. PrefixCacheTracker: Tracks cache hits, token savings, and estimated cost reduction.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from insight_agent.core.llm import LLMClient, Message, Turn


class CircuitBreakerState(str, Enum):
    """Three-state finite automaton for circuit breakers."""

    CLOSED = "closed"        # Healthy: requests pass through normally
    OPEN = "open"            # Tripped: requests immediately rejected/skipped
    HALF_OPEN = "half_open"  # Probing: single canary request allowed to test recovery


class CircuitBreaker:
    """Enterprise-grade Circuit Breaker pattern implementation.

    Guards downstream LLM APIs (DeepSeek, Qwen, Claude, OpenAI) against
    cascading failures, excessive timeouts, and rate-limit loops.
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        recovery_time_s: float = 15.0,
        half_open_max_probes: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_time_s = recovery_time_s
        self.half_open_max_probes = half_open_max_probes

        self.state: CircuitBreakerState = CircuitBreakerState.CLOSED
        self.consecutive_failures: int = 0
        self.consecutive_successes: int = 0
        self.total_failures: int = 0
        self.total_successes: int = 0
        self.last_failure_time: float = 0.0
        self.last_state_change_time: float = time.time()
        self.trips_count: int = 0
        self.active_probes: int = 0

    def can_execute(self) -> bool:
        """Determine if a request is allowed to pass through the breaker."""
        now = time.time()
        if self.state == CircuitBreakerState.CLOSED:
            return True
        if self.state == CircuitBreakerState.OPEN:
            if (now - self.last_failure_time) >= self.recovery_time_s:
                self._transition_to(CircuitBreakerState.HALF_OPEN)
                self.active_probes = 1
                return True
            return False
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.active_probes < self.half_open_max_probes:
                self.active_probes += 1
                return True
            return False
        return False

    def record_success(self) -> None:
        """Record successful downstream response; recovers breaker if in HALF_OPEN."""
        self.total_successes += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._transition_to(CircuitBreakerState.CLOSED)
            self.active_probes = 0

    def record_failure(self, exc: Exception | None = None) -> None:
        """Record failed downstream response; trips breaker if threshold exceeded."""
        now = time.time()
        self.total_failures += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = now

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Canary probe failed: immediately revert to OPEN
            self._transition_to(CircuitBreakerState.OPEN)
            self.active_probes = 0
        elif (
            self.state == CircuitBreakerState.CLOSED
            and self.consecutive_failures >= self.failure_threshold
        ):
            self._transition_to(CircuitBreakerState.OPEN)

    def trip(self, reason: str = "manual") -> None:
        """Manually trip breaker into OPEN state."""
        self.last_failure_time = time.time()
        self._transition_to(CircuitBreakerState.OPEN)

    def reset(self) -> None:
        """Reset breaker to healthy CLOSED initial state."""
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.active_probes = 0
        self._transition_to(CircuitBreakerState.CLOSED)

    def _transition_to(self, new_state: CircuitBreakerState) -> None:
        if self.state != new_state:
            if new_state == CircuitBreakerState.OPEN:
                self.trips_count += 1
            self.state = new_state
            self.last_state_change_time = time.time()


@dataclass
class ProviderEndpoint:
    """Configured LLM service provider endpoint with key pooling and circuit breaker."""

    name: str
    client: LLMClient | None = None
    model: str = ""
    provider_type: str = "openai_compatible"  # "openai_compatible" | "anthropic" | "custom"
    base_url: str | None = None
    api_keys: list[str] = field(default_factory=list)
    weight: float = 1.0
    timeout_s: float = 30.0
    max_retries: int = 1
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    _key_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.circuit_breaker.name == "default":
            self.circuit_breaker.name = self.name

    def get_active_key(self) -> str | None:
        """Get currently active API key from the pool."""
        if not self.api_keys:
            return None
        return self.api_keys[self._key_index % len(self.api_keys)]

    def rotate_key(self) -> str | None:
        """Rotate to next API key in the pool (e.g. upon 429 quota exhaustion)."""
        if not self.api_keys:
            return None
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        return self.get_active_key()


class AllProvidersFailedError(RuntimeError):
    """Raised when every candidate provider in the gateway fails or is tripped."""

    pass


class MultiProviderGateway:
    """Enterprise multi-provider gateway with key pooling, circuit breaker, and automatic failover.

    Implements the standard ``LLMClient`` protocol so it can be dropped into
    ``LLMPlanner``, ``Agent``, or any other LLM-consuming component.
    """

    def __init__(
        self,
        endpoints: Sequence[ProviderEndpoint],
        fallback_client: LLMClient | None = None,
        routing_strategy: str = "priority",  # "priority" | "round_robin" | "least_failures"
    ) -> None:
        self.endpoints = list(endpoints)
        self.fallback_client = fallback_client
        self.routing_strategy = routing_strategy
        self._round_robin_idx: int = 0

        # Telemetry metrics
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failover_events: int = 0
        self.endpoint_call_counts: dict[str, int] = {ep.name: 0 for ep in self.endpoints}
        self.endpoint_latencies_ms: dict[str, list[float]] = {ep.name: [] for ep in self.endpoints}

    def _select_candidate_endpoints(self) -> list[ProviderEndpoint]:
        """Order available endpoints based on routing strategy and breaker state."""
        # 1. Filter healthy or probe-eligible endpoints
        candidates = [ep for ep in self.endpoints if ep.circuit_breaker.can_execute()]
        if not candidates:
            return []

        # 2. Apply routing strategy
        if self.routing_strategy == "round_robin":
            n = len(candidates)
            idx = self._round_robin_idx % n
            self._round_robin_idx += 1
            return candidates[idx:] + candidates[:idx]
        elif self.routing_strategy == "least_failures":
            return sorted(candidates, key=lambda ep: ep.circuit_breaker.total_failures)
        else:  # "priority" (default)
            return candidates

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> Turn:
        """Route completion through candidate providers with circuit breaker and failover."""
        self.total_requests += 1
        candidates = self._select_candidate_endpoints()

        errors: list[str] = []

        # If all candidate endpoints are tripped
        if not candidates:
            if self.fallback_client is not None:
                self.failover_events += 1
                return await self.fallback_client.complete(messages, tools)
            raise AllProvidersFailedError(
                f"All {len(self.endpoints)} providers are currently tripped by circuit breaker."
            )

        for ep in candidates:
            if ep.client is None:
                continue

            self.endpoint_call_counts[ep.name] = self.endpoint_call_counts.get(ep.name, 0) + 1
            start_t = time.perf_counter()

            try:
                # Execute with endpoint timeout
                turn: Turn = await asyncio.wait_for(
                    ep.client.complete(messages, tools),
                    timeout=ep.timeout_s,
                )
                latency_ms = (time.perf_counter() - start_t) * 1000
                self.endpoint_latencies_ms.setdefault(ep.name, []).append(latency_ms)

                # Record success to circuit breaker
                ep.circuit_breaker.record_success()
                self.successful_requests += 1
                return turn

            except Exception as exc:
                latency_ms = (time.perf_counter() - start_t) * 1000
                self.endpoint_latencies_ms.setdefault(ep.name, []).append(latency_ms)

                # Record failure & rotate key if pool is available
                ep.circuit_breaker.record_failure(exc)
                if len(ep.api_keys) > 1:
                    ep.rotate_key()

                self.failover_events += 1
                err_msg = f"{ep.name}: {type(exc).__name__}({exc})"
                errors.append(err_msg)
                continue

        # If all candidates failed in this turn, try fallback_client
        if self.fallback_client is not None:
            try:
                self.failover_events += 1
                return await self.fallback_client.complete(messages, tools)
            except Exception as fb_exc:
                errors.append(f"fallback_client: {type(fb_exc).__name__}({fb_exc})")

        raise AllProvidersFailedError(
            f"All candidate providers failed to complete request. Errors: {'; '.join(errors)}"
        )

    def get_telemetry(self) -> dict[str, Any]:
        """Return operational telemetry report for Prometheus/Jaeger observation."""
        endpoint_reports = {}
        for ep in self.endpoints:
            lats = self.endpoint_latencies_ms.get(ep.name, [])
            avg_lat = sum(lats) / len(lats) if lats else 0.0
            endpoint_reports[ep.name] = {
                "state": ep.circuit_breaker.state.value,
                "calls": self.endpoint_call_counts.get(ep.name, 0),
                "total_failures": ep.circuit_breaker.total_failures,
                "trips": ep.circuit_breaker.trips_count,
                "avg_latency_ms": round(avg_lat, 2),
                "active_key_prefix": (ep.get_active_key() or "")[:8] + "...",
            }

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failover_events": self.failover_events,
            "success_rate_pct": round(
                (self.successful_requests / self.total_requests * 100) if self.total_requests else 0.0,
                2,
            ),
            "endpoints": endpoint_reports,
        }


class PromptCacheOptimizer:
    """Stabilizes prompt prefixes for LLM KV cache reuse (e.g. DeepSeek / Claude / OpenAI)."""

    @staticmethod
    def canonicalize_schema_hint(schema_hint: str) -> str:
        """Deterministically format schema definitions and normalize whitespace."""
        if not schema_hint:
            return ""
        lines = [line.strip() for line in schema_hint.strip().splitlines() if line.strip()]
        return "\n".join(lines)

    @staticmethod
    def compute_prefix_hash(prefix_text: str) -> str:
        """Compute stable SHA-256 fingerprint for the static prefix."""
        return hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()

    @staticmethod
    def build_canonical_prompt(
        system_template: str,
        schema_hint: str = "",
        domain: str = "",
        metric_vocab: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Construct a byte-stable system prompt prefix and return (canonical_text, prefix_hash)."""
        clean_schema = PromptCacheOptimizer.canonicalize_schema_hint(schema_hint)
        parts = [system_template.format(schema=clean_schema, domain=domain).strip()]
        if metric_vocab:
            sorted_metrics = sorted(metric_vocab.items(), key=lambda x: x[0].lower())
            metrics_str = "Standard Metric Vocab:\n" + "\n".join(f"- {k}: {v}" for k, v in sorted_metrics)
            parts.append(metrics_str)
        canonical_text = "\n\n".join(parts)
        prefix_hash = PromptCacheOptimizer.compute_prefix_hash(canonical_text)
        return canonical_text, prefix_hash


@dataclass
class CacheTelemetry:
    """Aggregated prompt prefix cache telemetry."""

    total_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cached_tokens_saved: int = 0
    estimated_cost_saved_usd: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total) if total > 0 else 0.0


class PrefixCacheTracker:
    """Tracks prompt prefix cache hits, token savings, and cost reductions."""

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        token_cost_per_m: float = 0.50,
        cache_discount_rate: float = 0.90,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.token_cost_per_m = token_cost_per_m  # $0.50 per 1M input tokens
        self.cache_discount_rate = cache_discount_rate  # 90% discount on cached tokens
        self._seen_prefixes: dict[str, float] = {}  # prefix_hash -> last_seen_time
        self.telemetry = CacheTelemetry()

    def record_prefix(self, prefix_hash: str, prefix_token_count: int) -> bool:
        """Record prefix usage and determine whether it counts as a cache hit."""
        now = time.time()
        self.telemetry.total_calls += 1
        is_hit = False

        if prefix_hash in self._seen_prefixes:
            if (now - self._seen_prefixes[prefix_hash]) <= self.ttl_seconds:
                is_hit = True
                self.telemetry.cache_hits += 1
                self.telemetry.cached_tokens_saved += prefix_token_count
                saved_usd = (prefix_token_count / 1_000_000.0) * self.token_cost_per_m * self.cache_discount_rate
                self.telemetry.estimated_cost_saved_usd += saved_usd
            else:
                self.telemetry.cache_misses += 1
        else:
            self.telemetry.cache_misses += 1

        self._seen_prefixes[prefix_hash] = now
        return is_hit

    def get_stats(self) -> dict[str, Any]:
        """Return summary dictionary of cache telemetry."""
        return {
            "total_calls": self.telemetry.total_calls,
            "cache_hits": self.telemetry.cache_hits,
            "cache_misses": self.telemetry.cache_misses,
            "hit_rate_pct": round(self.telemetry.hit_rate * 100, 2),
            "cached_tokens_saved": self.telemetry.cached_tokens_saved,
            "estimated_cost_saved_usd": round(self.telemetry.estimated_cost_saved_usd, 4),
        }
