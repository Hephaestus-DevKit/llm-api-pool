"""Admin/API authentication and the per-caller rate limit for /v1 endpoints."""
from __future__ import annotations

import hashlib
import hmac
import time
from collections import OrderedDict, deque
from typing import Optional

from fastapi import Header, HTTPException, Request

from . import settings
from .diagnostics import record_diagnostic_event


def extract_bearer_token(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def safe_token_match(supplied: Optional[str], expected: str) -> bool:
    if not supplied or not expected:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


async def require_admin(
    request: Request,
    x_admin_token: Optional[str] = Header(default=None),
):
    if not settings.require_admin_token():
        return
    supplied = x_admin_token or extract_bearer_token(request)
    if not safe_token_match(supplied, settings.ADMIN_TOKEN):
        record_diagnostic_event("warn", "bad_admin_token", client=client_ip(request), path=str(request.url.path))
        raise HTTPException(401, "bad admin token")


async def require_api_token(request: Request, x_api_key: Optional[str] = Header(default=None)):
    if not settings.API_TOKEN:
        return
    supplied = x_api_key or extract_bearer_token(request)
    if not safe_token_match(supplied, settings.API_TOKEN):
        record_diagnostic_event("warn", "bad_api_token", client=client_ip(request), path=str(request.url.path))
        raise HTTPException(401, "bad api token")


class SlidingWindowRateLimiter:
    """Per-key sliding window with a bounded key table.

    The key includes the caller's token, so an unauthenticated peer could otherwise mint
    unlimited keys and grow the table without bound. Idle keys are swept periodically and
    the table falls back to LRU eviction, which keeps memory flat without ever refusing a
    request just because the table is full.
    """

    def __init__(self, limit_per_minute: int, max_keys: int = 20000, sweep_interval: float = 30.0):
        self.limit_per_minute = max(0, int(limit_per_minute))
        self.window_seconds = 60.0
        self.max_keys = max(1, int(max_keys))
        self.sweep_interval = sweep_interval
        self._hits: "OrderedDict[str, deque]" = OrderedDict()
        self._last_sweep = 0.0

    def _sweep(self, now: float) -> None:
        if now - self._last_sweep < self.sweep_interval:
            return
        self._last_sweep = now
        for key in [k for k, hits in self._hits.items() if not hits or now - hits[-1] > self.window_seconds]:
            self._hits.pop(key, None)

    def tracked_keys(self) -> int:
        return len(self._hits)

    def check(self, key: str) -> bool:
        if self.limit_per_minute <= 0:
            return True
        now = time.monotonic()
        self._sweep(now)
        hits = self._hits.get(key)
        if hits is None:
            hits = self._hits[key] = deque()
            while len(self._hits) > self.max_keys:
                self._hits.popitem(last=False)
        else:
            self._hits.move_to_end(key)
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.limit_per_minute:
            return False
        hits.append(now)
        return True


api_rate_limiter = SlidingWindowRateLimiter(settings.RATE_LIMIT_PER_MINUTE)


def client_ip(request: Request) -> str:
    """Peer address, or the originating client when explicitly told to trust a proxy.

    Off by default: with an untrusted network path any caller can spoof X-Forwarded-For and
    give itself a fresh rate-limit bucket per request.
    """
    if settings.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for") or ""
        first = forwarded.split(",")[0].strip()
        if first:
            return first
        real_ip = (request.headers.get("x-real-ip") or "").strip()
        if real_ip:
            return real_ip
    return request.client.host if request.client else "unknown"


async def require_api_access(request: Request, x_api_key: Optional[str] = Header(default=None)):
    await require_api_token(request, x_api_key)
    supplied = x_api_key or extract_bearer_token(request) or "anonymous"
    token_hash = hashlib.sha256(supplied.encode("utf-8")).hexdigest()[:16]
    if not api_rate_limiter.check(f"{client_ip(request)}:{token_hash}:{request.url.path}"):
        record_diagnostic_event("warn", "api_rate_limited", client=client_ip(request), path=str(request.url.path))
        raise HTTPException(429, "rate limit exceeded")
