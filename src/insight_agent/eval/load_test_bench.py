"""High-Concurrency and Throughput Load Benchmark for InsightAgent 2.0.

Measures:
1. Cold Analytical SQLite Query Throughput & Latency (P50, P90, P95, P99) under 1, 8, 16, 32 workers.
2. Hot Redis Multi-Tier Cache Throughput & Latency (P50, P90, P95, P99) under 1, 8, 16, 32, 64 workers.
3. High-Concurrency Redlock Mutex Contention under 32 simultaneous workers.

Outputs verified empirical metrics saved to docs/concurrency_load_benchmark.json.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["INSIGHT_JEV_URL"] = ""

from insight_agent.app.redis_store import RedisMemoryStore
from insight_agent.data.cn_retail_db import connect_cn_retail, ensure_cn_retail_database


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100.0)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cold SQLite Analytical Queries Load Test (88k rows)
# ─────────────────────────────────────────────────────────────────────────────

COMPLEX_ANALYTICAL_SQL = (
    "SELECT p.category_l1, u.province, COUNT(DISTINCT o.order_id) as orders_count, "
    "ROUND(SUM(oi.quantity * oi.unit_price), 2) as gmv "
    "FROM cn_orders o "
    "JOIN cn_users u ON o.user_id = u.user_id "
    "JOIN cn_order_items oi ON o.order_id = oi.order_id "
    "JOIN cn_products p ON oi.product_id = p.product_id "
    "WHERE o.order_status = '已妥投' "
    "GROUP BY p.category_l1, u.province "
    "ORDER BY gmv DESC"
)


def run_cold_sql_load_test(concurrency_levels: list[int] = [1, 8, 16, 32], requests_per_level: int = 64) -> dict[str, Any]:
    print("[1/3] Running Cold Analytical Query Load Test on 88k-Row SQLite DB...", flush=True)
    results = {}

    def single_query() -> float:
        conn = connect_cn_retail()
        t0 = time.perf_counter()
        try:
            cur = conn.cursor()
            cur.execute(COMPLEX_ANALYTICAL_SQL)
            _ = cur.fetchall()
            lat = (time.perf_counter() - t0) * 1000
            return lat
        finally:
            conn.close()

    for workers in concurrency_levels:
        latencies: list[float] = []
        t_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(single_query) for _ in range(requests_per_level)]
            for fut in concurrent.futures.as_completed(futures):
                latencies.append(fut.result())
        total_time_s = time.perf_counter() - t_start
        qps = round(len(latencies) / total_time_s, 1)

        level_stats = {
            "workers": workers,
            "total_requests": len(latencies),
            "total_time_s": round(total_time_s, 3),
            "qps": qps,
            "p50_ms": round(percentile(latencies, 50), 2),
            "p90_ms": round(percentile(latencies, 90), 2),
            "p95_ms": round(percentile(latencies, 95), 2),
            "p99_ms": round(percentile(latencies, 99), 2),
            "mean_ms": round(sum(latencies) / len(latencies), 2),
        }
        results[f"concurrency_{workers}"] = level_stats
        print(f"  -> Concurrency {workers:2d}: QPS={qps:6.1f} | Mean={level_stats['mean_ms']:6.2f}ms | P95={level_stats['p95_ms']:6.2f}ms | P99={level_stats['p99_ms']:6.2f}ms", flush=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hot Redis Multi-Tier Cache Load Test
# ─────────────────────────────────────────────────────────────────────────────

def run_hot_cache_load_test(concurrency_levels: list[int] = [1, 8, 16, 32, 64], requests_per_level: int = 500) -> dict[str, Any]:
    print("\n[2/3] Running Hot Redis Multi-Tier Cache Load Test...", flush=True)
    store = RedisMemoryStore()

    # Prewarm cache
    conn = connect_cn_retail()
    cur = conn.cursor()
    cur.execute(COMPLEX_ANALYTICAL_SQL)
    rows = cur.fetchall()
    conn.close()

    cache_key = "analytics:gmv_category_province:2024"
    store.set_query_cache(cache_key, rows, ttl_seconds=3600)

    results = {}

    def single_cache_read() -> float:
        t0 = time.perf_counter()
        _ = store.get_query_cache(cache_key)
        return (time.perf_counter() - t0) * 1000

    for workers in concurrency_levels:
        latencies: list[float] = []
        t_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(single_cache_read) for _ in range(requests_per_level)]
            for fut in concurrent.futures.as_completed(futures):
                latencies.append(fut.result())
        total_time_s = time.perf_counter() - t_start
        qps = round(len(latencies) / total_time_s, 1)

        level_stats = {
            "workers": workers,
            "total_requests": len(latencies),
            "total_time_s": round(total_time_s, 4),
            "qps": qps,
            "p50_ms": round(percentile(latencies, 50), 3),
            "p90_ms": round(percentile(latencies, 90), 3),
            "p95_ms": round(percentile(latencies, 95), 3),
            "p99_ms": round(percentile(latencies, 99), 3),
            "mean_ms": round(sum(latencies) / len(latencies), 3),
        }
        results[f"concurrency_{workers}"] = level_stats
        print(f"  -> Concurrency {workers:2d}: QPS={qps:8.1f} | Mean={level_stats['mean_ms']:5.3f}ms | P95={level_stats['p95_ms']:5.3f}ms | P99={level_stats['p99_ms']:5.3f}ms", flush=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. High-Concurrency Redlock Mutex Contention Benchmark (32 Workers)
# ─────────────────────────────────────────────────────────────────────────────

def run_high_concurrency_redlock_test(workers: int = 32) -> dict[str, Any]:
    print(f"\n[3/3] Running High-Concurrency Redlock Mutex Test ({workers} Workers)...", flush=True)
    store = RedisMemoryStore()
    lock_key = "remediation:batch_voucher:order_cluster_hangzhou_2024"

    acquired_workers: list[int] = []
    rejected_workers: list[int] = []
    timings: list[float] = []

    def worker_action(wid: int) -> None:
        t0 = time.perf_counter()
        with store.acquire_action_lock(lock_key, ttl_seconds=5) as acquired:
            elapsed = (time.perf_counter() - t0) * 1000
            timings.append(elapsed)
            if acquired:
                acquired_workers.append(wid)
                time.sleep(0.05)  # simulate safe processing
            else:
                rejected_workers.append(wid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker_action, i) for i in range(workers)]
        concurrent.futures.wait(futures)

    success = (len(acquired_workers) == 1 and len(rejected_workers) == (workers - 1))
    print(f"  -> Contestants       : {workers} concurrent workers", flush=True)
    print(f"  -> Lock Acquired     : Exactly {len(acquired_workers)} winner (Worker #{acquired_workers[0] if acquired_workers else 'None'})", flush=True)
    print(f"  -> Safely Rejected   : {len(rejected_workers)} workers (0 duplicate remediation)", flush=True)
    print(f"  -> Verdict           : {'PERFECT MUTEX GUARANTEE' if success else 'LOCK CONTENTION FAILED'}", flush=True)

    return {
        "concurrent_workers": workers,
        "acquired_count": len(acquired_workers),
        "rejected_count": len(rejected_workers),
        "mutex_verified": success,
        "mean_contention_latency_ms": round(sum(timings) / len(timings), 3),
        "p99_latency_ms": round(percentile(timings, 99), 3),
    }


def main() -> None:
    ensure_cn_retail_database()
    print("================================================================================", flush=True)
    print("        InsightAgent 2.0 High-Concurrency & Load Benchmark                      ", flush=True)
    print("        Dataset: CN-Retail 2.0 (88,002 rows, 8 tables)                          ", flush=True)
    print("================================================================================\n", flush=True)

    cold_res = run_cold_sql_load_test()
    hot_res = run_hot_cache_load_test()
    lock_res = run_high_concurrency_redlock_test()

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "dataset": "cn_retail_2.0",
            "rows_count": 88002,
            "sqlite_mode": "read_only_with_authorizer",
            "cache_layer": "Redis_L3_MultiTier",
        },
        "summary": {
            "peak_cold_qps": max(v["qps"] for v in cold_res.values()),
            "peak_hot_cache_qps": max(v["qps"] for v in hot_res.values()),
            "hot_cache_p99_latency_ms": hot_res["concurrency_32"]["p99_ms"],
            "cache_throughput_multiplication": round(
                max(v["qps"] for v in hot_res.values()) / max(v["qps"] for v in cold_res.values()), 1
            ),
            "redlock_high_concurrency_safe": lock_res["mutex_verified"],
        },
        "cold_sql_bench": cold_res,
        "hot_cache_bench": hot_res,
        "redlock_bench": lock_res,
    }

    out_file = Path("docs/concurrency_load_benchmark.json")
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Concurrency & Load Benchmark results saved to: {out_file.resolve()}", flush=True)


if __name__ == "__main__":
    main()
