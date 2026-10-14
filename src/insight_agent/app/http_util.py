"""Resilient HTTP helper for remote model endpoints.

The embedding and reranker gateways drop a large fraction of connections under
sustained load (observed: ~57% ``RemoteDisconnected`` and read-timeouts, while
successful calls stay fast at 1-2s). A single transient failure otherwise sends
the whole call straight to a local heuristic, which quietly degrades evaluation
caliber. This helper retries transient failures with exponential backoff and
jitter, while refusing to retry auth/billing/client errors (which never recover
on retry and should trip the caller's circuit breaker instead).

Stdlib only, on purpose: the rest of the app avoids a hard ``requests``/``httpx``
dependency so it can run in zero-infra CI.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from typing import Any

# Client-side status codes that will not recover on retry. 429 is excluded on
# purpose: it is throttling and backing off then retrying is the right response.
_NON_RETRYABLE = {400, 401, 402, 403, 404, 405, 422}

logger = logging.getLogger(__name__)


class RateLimiter:
    """Minimum-interval throttle shared across callers for one endpoint.

    The gateway rate-limits per key (observed: HTTP 429 on bge-m3 under bursts).
    Spacing calls at least ``min_interval`` seconds apart keeps us under the cap
    so the 429 never fires in the first place. Thread-safe: concurrent callers
    queue on the lock and are spaced strictly, which is the intended behaviour.
    """

    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0.0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0.0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self.min_interval


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float | None:
    """Parse a 429/503 ``Retry-After`` header (delta-seconds or HTTP-date)."""
    try:
        headers = getattr(exc, "headers", None)
        raw = headers.get("Retry-After") if headers else None
        if not raw:
            return None
        raw = raw.strip()
        if raw.isdigit():
            return float(raw)
        dt = parsedate_to_datetime(raw)
        if dt is None:
            return None
        import datetime

        now = datetime.datetime.now(dt.tzinfo) if dt.tzinfo else datetime.datetime.now()
        return max(0.0, (dt - now).total_seconds())
    except Exception:
        return None


def post_json_with_retry(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    backoff: float = 2.0,
    max_delay: float = 30.0,
    rate_limiter: "RateLimiter | None" = None,
) -> tuple[dict[str, Any] | None, Exception | None]:
    """POST JSON and retry transient failures with exponential backoff.

    Returns ``(data, None)`` on success, or ``(None, last_exception)`` once every
    attempt is exhausted or a non-retryable client error is hit. The caller
    inspects the exception to decide between a plain fallback and a circuit break.

    429 (throttling) is retried with the same exponential schedule, but the
    server's ``Retry-After`` header is honoured as a floor when present, and the
    per-attempt delay is capped at ``max_delay``. A ``rate_limiter``, when given,
    paces every attempt so bursts stay under the endpoint's rate cap.
    """
    data = json.dumps(payload).encode("utf-8")
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        if rate_limiter is not None:
            rate_limiter.acquire()
        retry_after: float | None = None
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8")), None
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in _NON_RETRYABLE:
                return None, exc  # never recovers; let caller circuit-break
            if exc.code == 429:
                retry_after = _retry_after_seconds(exc)
        except Exception as exc:  # URLError, RemoteDisconnected, timeout, etc.
            last_exc = exc

        if attempt < max_attempts:
            # Exponential backoff with jitter; never shorter than the server's
            # Retry-After, never longer than max_delay.
            delay = base_delay * (backoff ** (attempt - 1))
            if retry_after is not None:
                delay = max(delay, retry_after)
            delay = min(delay, max_delay) + random.uniform(0.0, 0.25)
            time.sleep(delay)

    return None, last_exc

