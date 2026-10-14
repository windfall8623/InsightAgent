"""Comprehensive test suite for P5 upgrades:
1. CircuitBreaker finite automaton (CLOSED -> OPEN -> HALF_OPEN -> CLOSED).
2. ProviderEndpoint API key rotation and pooling.
3. MultiProviderGateway failover, routing strategies, and telemetry.
4. Fallback client execution and AllProvidersFailedError handling.
5. PromptCacheOptimizer deterministic schema canonicalization and SHA-256 fingerprinting.
6. PrefixCacheTracker hit/miss lifecycle, token savings, and cost reduction.
7. LLMPlanner integration with prefix caching notes.
8. High-concurrency stress test with 64 parallel queries through the gateway.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest
from insight_agent.app.planner import LLMPlanner
from insight_agent.core.llm import Message, Turn
from insight_agent.core.providers.gateway import (
    AllProvidersFailedError,
    CircuitBreaker,
    CircuitBreakerState,
    MultiProviderGateway,
    PrefixCacheTracker,
    PromptCacheOptimizer,
    ProviderEndpoint,
)


class MockFailingClient:
    """Mock client that always fails with a configurable exception."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.call_count = 0

    async def complete(self, messages: Any, tools: Any) -> Turn:
        self.call_count += 1
        raise self.exc


class MockWorkingClient:
    """Mock client that succeeds and returns a predetermined Turn."""

    def __init__(self, reply: str = "SELECT 1") -> None:
        self.reply = reply
        self.call_count = 0

    async def complete(self, messages: Any, tools: Any) -> Turn:
        self.call_count += 1
        return Turn(text=self.reply)


def test_circuit_breaker_lifecycle() -> None:
    """Verify CircuitBreaker trips after threshold failures and recovers via canary probe."""
    cb = CircuitBreaker(name="test-breaker", failure_threshold=3, recovery_time_s=0.1)
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.can_execute() is True

    # 1. First two failures don't trip breaker
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.can_execute() is True

    # 2. Third failure trips breaker to OPEN
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.trips_count == 1
    assert cb.can_execute() is False

    # 3. Fast-forward time past recovery_time_s
    time.sleep(0.12)
    # can_execute() should now trigger HALF_OPEN transition
    assert cb.can_execute() is True
    assert cb.state == CircuitBreakerState.HALF_OPEN
    # Second probe in HALF_OPEN should be rejected
    assert cb.can_execute() is False

    # 4. Successful canary probe resets breaker to CLOSED
    cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.consecutive_failures == 0
    assert cb.can_execute() is True


def test_circuit_breaker_probe_failure_re_trips() -> None:
    """If the canary probe fails during HALF_OPEN, breaker must immediately re-trip to OPEN."""
    cb = CircuitBreaker(name="probe-test", failure_threshold=2, recovery_time_s=0.05)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN

    time.sleep(0.06)
    assert cb.can_execute() is True
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # Probe fails
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.can_execute() is False


def test_provider_endpoint_key_rotation() -> None:
    """Verify ProviderEndpoint rotates keys in a round-robin cycle."""
    ep = ProviderEndpoint(
        name="qwen-pool",
        model="qwen-2.5-72b",
        api_keys=["key-alpha", "key-beta", "key-gamma"],
    )
    assert ep.get_active_key() == "key-alpha"
    assert ep.rotate_key() == "key-beta"
    assert ep.rotate_key() == "key-gamma"
    assert ep.rotate_key() == "key-alpha"


@pytest.mark.asyncio
async def test_multi_provider_gateway_priority_failover() -> None:
    """Primary fails with 429 quota exhaustion -> gateway automatically fails over to secondary."""
    primary_client = MockFailingClient(RuntimeError("429 Rate limit exceeded"))
    secondary_client = MockWorkingClient("SELECT count(*) FROM orders")

    ep_primary = ProviderEndpoint(name="deepseek-primary", client=primary_client, timeout_s=5.0)
    ep_secondary = ProviderEndpoint(name="qwen-backup", client=secondary_client, timeout_s=5.0)

    gateway = MultiProviderGateway([ep_primary, ep_secondary], routing_strategy="priority")

    messages = [Message(role="user", content="count orders")]
    turn = await gateway.complete(messages, ())

    assert turn.text == "SELECT count(*) FROM orders"
    assert primary_client.call_count == 1
    assert secondary_client.call_count == 1
    assert gateway.total_requests == 1
    assert gateway.successful_requests == 1
    assert gateway.failover_events == 1

    telemetry = gateway.get_telemetry()
    assert telemetry["total_requests"] == 1
    assert telemetry["failover_events"] == 1
    assert telemetry["endpoints"]["deepseek-primary"]["calls"] == 1
    assert telemetry["endpoints"]["qwen-backup"]["calls"] == 1


@pytest.mark.asyncio
async def test_multi_provider_gateway_round_robin() -> None:
    """Round-robin strategy rotates evenly across active endpoints."""
    client_a = MockWorkingClient("A")
    client_b = MockWorkingClient("B")

    ep_a = ProviderEndpoint(name="ep-a", client=client_a)
    ep_b = ProviderEndpoint(name="ep-b", client=client_b)

    gateway = MultiProviderGateway([ep_a, ep_b], routing_strategy="round_robin")
    messages = [Message(role="user", content="ping")]

    turn_1 = await gateway.complete(messages, ())
    turn_2 = await gateway.complete(messages, ())
    turn_3 = await gateway.complete(messages, ())
    turn_4 = await gateway.complete(messages, ())

    assert turn_1.text == "A"
    assert turn_2.text == "B"
    assert turn_3.text == "A"
    assert turn_4.text == "B"
    assert client_a.call_count == 2
    assert client_b.call_count == 2


@pytest.mark.asyncio
async def test_multi_provider_gateway_fallback_and_error_handling() -> None:
    """When all candidate providers fail, gateway activates fallback client or raises error."""
    client_1 = MockFailingClient(TimeoutError("Endpoint 1 timeout"))
    client_2 = MockFailingClient(RuntimeError("Endpoint 2 500 error"))
    fallback = MockWorkingClient("FALLBACK_SQL")

    ep_1 = ProviderEndpoint(name="p1", client=client_1)
    ep_2 = ProviderEndpoint(name="p2", client=client_2)

    # 1. With fallback client
    gateway_with_fb = MultiProviderGateway([ep_1, ep_2], fallback_client=fallback)
    turn = await gateway_with_fb.complete([Message(role="user", content="test")], ())
    assert turn.text == "FALLBACK_SQL"
    assert fallback.call_count == 1

    # 2. Without fallback client
    gateway_no_fb = MultiProviderGateway([ep_1, ep_2])
    with pytest.raises(AllProvidersFailedError) as exc_info:
        await gateway_no_fb.complete([Message(role="user", content="test")], ())
    assert "All candidate providers failed" in str(exc_info.value)


def test_prompt_cache_optimizer_canonical_hash_stability() -> None:
    """Different formatting of schema hints produces identical canonical prefix and SHA-256 hash."""
    template = "System prompt template with {schema} and {domain}."

    schema_v1 = "orders (order_id, status)\nproducts (product_id, price)\n"
    schema_v2 = "  orders (order_id, status)  \n\n  products (product_id, price)  \n"

    p1, h1 = PromptCacheOptimizer.build_canonical_prompt(template, schema_hint=schema_v1, domain="ecommerce")
    p2, h2 = PromptCacheOptimizer.build_canonical_prompt(template, schema_hint=schema_v2, domain="ecommerce")

    assert p1 == p2
    assert h1 == h2
    assert len(h1) == 64  # SHA-256


def test_prefix_cache_tracker_lifecycle() -> None:
    """PrefixCacheTracker correctly tracks cache hits, token savings, and estimated cost reductions."""
    tracker = PrefixCacheTracker(ttl_seconds=10.0, token_cost_per_m=1.0, cache_discount_rate=0.90)

    # First turn: cache miss
    is_hit_1 = tracker.record_prefix("hash-12345", prefix_token_count=2000)
    assert is_hit_1 is False
    assert tracker.telemetry.cache_misses == 1
    assert tracker.telemetry.cache_hits == 0

    # Second turn: identical hash within TTL -> cache hit
    is_hit_2 = tracker.record_prefix("hash-12345", prefix_token_count=2000)
    assert is_hit_2 is True
    assert tracker.telemetry.cache_hits == 1
    assert tracker.telemetry.cached_tokens_saved == 2000
    # 2000 tokens * $1.0/1M * 0.90 = $0.0018
    assert round(tracker.telemetry.estimated_cost_saved_usd, 4) == 0.0018

    stats = tracker.get_stats()
    assert stats["total_calls"] == 2
    assert stats["hit_rate_pct"] == 50.0
    assert stats["cached_tokens_saved"] == 2000


@pytest.mark.asyncio
async def test_llm_planner_with_prefix_cache_notes() -> None:
    """LLMPlanner integrates with PrefixCacheTracker and appends cache notes to PlannerResult."""
    tracker = PrefixCacheTracker(ttl_seconds=30.0)
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(
        return_value=Turn(text="```sql\nSELECT count(*) FROM orders\n```")
    )

    planner = LLMPlanner(client=mock_client, prefix_cache_tracker=tracker)

    # Turn 1
    res1 = await planner.plan(
        "查询总订单数",
        schema_hint="orders(id, created_at)",
        domain="retail",
    )
    assert any("prefix_cache: miss" in note for note in res1.notes)

    # Turn 2: same schema & domain
    res2 = await planner.plan(
        "查询最近7天订单数",
        schema_hint="orders(id, created_at)",
        domain="retail",
    )
    assert any("prefix_cache: hit" in note for note in res2.notes)
    assert tracker.telemetry.cache_hits == 1
    assert tracker.telemetry.cache_misses == 1


@pytest.mark.asyncio
async def test_high_concurrency_stress_load_gateway() -> None:
    """Stress test: 64 concurrent requests across a MultiProviderGateway pool."""
    client_a = MockWorkingClient("SELECT 'A'")
    client_b = MockWorkingClient("SELECT 'B'")
    client_c = MockWorkingClient("SELECT 'C'")

    ep_a = ProviderEndpoint(name="worker-a", client=client_a)
    ep_b = ProviderEndpoint(name="worker-b", client=client_b)
    ep_c = ProviderEndpoint(name="worker-c", client=client_c)

    gateway = MultiProviderGateway([ep_a, ep_b, ep_c], routing_strategy="round_robin")

    concurrency = 64
    messages = [Message(role="user", content="concurrent test")]

    start_t = time.perf_counter()
    results = await asyncio.gather(
        *(gateway.complete(messages, ()) for _ in range(concurrency))
    )
    elapsed_ms = (time.perf_counter() - start_t) * 1000

    assert len(results) == concurrency
    assert all(r.text in ("SELECT 'A'", "SELECT 'B'", "SELECT 'C'") for r in results)
    assert gateway.successful_requests == concurrency
    assert gateway.failover_events == 0

    telemetry = gateway.get_telemetry()
    assert telemetry["total_requests"] == concurrency
    assert telemetry["successful_requests"] == concurrency
    total_calls = sum(ep["calls"] for ep in telemetry["endpoints"].values())
    assert total_calls == concurrency

    qps = concurrency / (elapsed_ms / 1000.0)
    print(f"\n[Stress Benchmark] 64 concurrent gateway requests finished in {elapsed_ms:.1f}ms ({qps:.0f} QPS)")
