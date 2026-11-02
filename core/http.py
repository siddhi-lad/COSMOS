"""Shared HTTP plumbing: pooled session with light retries, a TTL cache, a rate
limiter and a circuit breaker. These four things are what keep latency and
errors down when several free APIs are chained together."""
from __future__ import annotations

import threading
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session: requests.Session | None = None
_session_lock = threading.Lock()


def session() -> requests.Session:
    global _session
    with _session_lock:
        if _session is None:
            s = requests.Session()
            retry = Retry(
                total=1,
                backoff_factor=0.3,
                status_forcelist=(500, 502, 503, 504),
                allowed_methods=frozenset(["GET", "POST"]),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            s.headers["User-Agent"] = "COSMOS/1.0 (cosmetic ingredient research app)"
            _session = s
        return _session


class TTLCache:
    """Tiny thread-safe cache. Only successful results should be stored."""

    def __init__(self, ttl: float = 3600, max_items: int = 512):
        self.ttl, self.max_items = ttl, max_items
        self._d: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        with self._lock:
            hit = self._d.get(key)
            if not hit:
                return None
            if time.time() - hit[0] > self.ttl:
                self._d.pop(key, None)
                return None
            return hit[1]

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            if len(self._d) >= self.max_items:
                self._d.pop(next(iter(self._d)))
            self._d[key] = (time.time(), value)


class RateLimiter:
    """Spaces calls at least `min_interval` seconds apart (thread-safe)."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.min_interval
        if delay:
            time.sleep(delay)


class CircuitBreaker:
    """Remember that a provider/model just failed so we skip it instead of
    paying its timeout again on every request."""

    def __init__(self) -> None:
        self._until: dict[str, float] = {}
        self._lock = threading.Lock()

    def trip(self, key: str, cooldown: float) -> None:
        with self._lock:
            self._until[key] = time.time() + cooldown

    def is_open(self, key: str) -> bool:
        with self._lock:
            return self._until.get(key, 0) > time.time()

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._until.clear()
            else:
                self._until.pop(key, None)


breaker = CircuitBreaker()


def get_json(url: str, params: dict | None = None, timeout: float = 12, limiter: RateLimiter | None = None) -> Any:
    if limiter:
        limiter.wait()
    r = session().get(url, params=params, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def get_text(url: str, params: dict | None = None, timeout: float = 12, limiter: RateLimiter | None = None) -> str:
    if limiter:
        limiter.wait()
    r = session().get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.text
