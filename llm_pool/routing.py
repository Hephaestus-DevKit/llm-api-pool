"""SmartRouter: channel selection with real-time quota monitoring and failure cooldowns.

Intelligent selection to prevent stuck/dead channels. Real-time stats: official headers +
web estimates + health + cooldowns. Per-channel concurrency limit (critical for agent
high-freq + browser web channels).
"""
from __future__ import annotations

import asyncio
import random
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

from . import backends, settings, store
from .canonical import CanonicalRequest

# Provider quota headers, most specific first. Substring matching used to pick whichever
# header happened to iterate last, which mixed request budgets with token budgets.
# Token budgets first: they are the binding constraint, and mixing them with request budgets
# under one threshold would read "9 requests left" as an exhausted account.
QUOTA_REMAINING_HEADERS = (
    ("x-ratelimit-remaining-tokens", "tokens"),
    ("anthropic-ratelimit-tokens-remaining", "tokens"),
    ("anthropic-ratelimit-input-tokens-remaining", "tokens"),
    ("x-ratelimit-remaining-requests", "requests"),
    ("anthropic-ratelimit-requests-remaining", "requests"),
    ("ratelimit-remaining", "requests"),
)


def parse_quota_remaining(headers: Optional[dict]) -> Optional[Tuple[int, str]]:
    """Return (remaining, unit) from the most specific quota header present."""
    if not headers:
        return None
    lowered = {str(k).lower(): v for k, v in headers.items()}
    for name, unit in QUOTA_REMAINING_HEADERS:
        raw = lowered.get(name)
        if raw is None:
            continue
        try:
            return int(str(raw).strip()), unit
        except (TypeError, ValueError):
            continue
    return None


def quota_is_exhausted(remaining: Optional[int], unit: str) -> bool:
    if remaining is None:
        return False
    limit = settings.QUOTA_EXHAUSTED_THRESHOLD if unit == "tokens" else settings.QUOTA_EXHAUSTED_REQUESTS
    return remaining < limit


class SmartRouter:
    def __init__(self):
        self.stats: Dict[str, dict] = {}       # channel_id -> live stats
        self.sems: Dict[str, asyncio.Semaphore] = {}
        self.cooldowns: Dict[str, float] = {}

    def _get_stats(self, ch_id: str) -> dict:
        if ch_id not in self.stats:
            self.stats[ch_id] = {
                "health": 1.0,
                "avg_latency": 4.0,
                "calls": 0,
                "success": 0,
                "used_est_tokens": 0,
                "quota_est": 100000,           # overridable per channel via config["quota"]
                "in_flight": 0,
                "consec_fail": 0,
                "last_call": 0,
                "last_quota_remaining": None,  # from provider headers, official channels only
                "last_quota_unit": "tokens",
                "last_quota_at": 0.0,
                "quota_category": "general",   # "chat" | "codex" | "general"
            }
        return self.stats[ch_id]

    def fresh_quota_remaining(self, stats: dict) -> Optional[int]:
        """The provider's reported headroom, or None once the reading has gone stale."""
        if stats.get("last_quota_remaining") is None:
            return None
        if time.time() - stats.get("last_quota_at", 0.0) > settings.QUOTA_STALE_SECONDS:
            return None
        return stats["last_quota_remaining"]

    def _sync_stats_from_config(self, ch: dict) -> dict:
        s = self._get_stats(ch["id"])
        cfg = ch.get("config") or {}
        if "quota" in cfg:
            try:
                s["quota_est"] = max(1, int(cfg["quota"]))
            except (TypeError, ValueError):
                pass
        if cfg.get("quota_category"):
            s["quota_category"] = cfg["quota_category"]
        return s

    def concurrency_limit(self, ch: dict) -> int:
        default = 2 if ch["type"].startswith("web_") else 8
        raw = (ch.get("config") or {}).get("max_concurrent")
        if raw is None:
            return default
        try:
            # Clamped: an unbounded value would defeat the point of a per-channel limit.
            return min(64, max(1, int(raw)))
        except (TypeError, ValueError):
            return default

    def get_sem(self, ch: dict) -> asyncio.Semaphore:
        cid = ch["id"]
        sem = self.sems.get(cid)
        if sem is None:
            sem = self.sems[cid] = asyncio.BoundedSemaphore(self.concurrency_limit(ch))
        return sem

    def is_compatible(self, ch: dict, model: str) -> bool:
        ch_type = ch["type"]
        name = (model or "auto").lower()
        for alias_key, alias_val in (ch.get("config", {}).get("aliases") or {}).items():
            if name == str(alias_key).lower() or name == str(alias_val).lower():
                return True
        if "codex" in name or "copilot" in name:
            return "codex" in ch_type or "openai" in ch_type or "chatgpt" in ch_type
        if any(token in name for token in ("claude", "sonnet", "opus", "haiku", "fable")):
            return "claude" in ch_type
        if "gemini" in name:
            return "gemini" in ch_type
        if "gpt" in name or name.startswith(("o1", "o3", "o4")):
            return any(token in ch_type for token in ("openai", "chatgpt", "codex"))
        return True  # "auto" and unrecognised names may go anywhere

    def compute_score(self, ch: dict, req: CanonicalRequest) -> float:
        s = self._sync_stats_from_config(ch)
        if self.cooldowns.get(ch["id"], 0) > time.time():
            return 0.0
        if s["consec_fail"] >= 4:
            return 0.05

        remaining = self.fresh_quota_remaining(s)
        if remaining is None or s.get("last_quota_unit") != "tokens":
            remaining = s["quota_est"] - s["used_est_tokens"]
        quota_factor = max(0.05, min(1.0, remaining / max(1, s["quota_est"])))
        latency_factor = 1.0 / (1.0 + s["avg_latency"] / 8.0)
        # A saturated channel would make the caller queue on its semaphore with no timeout.
        # Scoring it down routes around the congestion instead, while still allowing the
        # queue when every channel is full.
        saturated = s["in_flight"] >= self.concurrency_limit(ch)
        load_factor = (0.1 if saturated else 1.0) / (1.0 + s["in_flight"] * 0.5)
        model_bonus = 1.8 if self.is_compatible(ch, req.model) else 0.6
        category_bonus = 1.5 if ("codex" in (req.model or "").lower() and s.get("quota_category") == "codex") else 1.0
        # A channel that cannot carry tool calls would silently drop them, so deprioritise it.
        backend = backends.BACKENDS.get(ch["type"])
        tool_bonus = 0.25 if (req.tools and backend is not None and not backend.supports_tools) else 1.0
        try:
            priority = float((ch.get("config") or {}).get("priority") or 1)
        except (TypeError, ValueError):
            priority = 1.0
        priority_bonus = max(0.5, min(3.0, priority))

        score = (s["health"] * quota_factor * latency_factor * load_factor
                 * model_bonus * category_bonus * tool_bonus * priority_bonus)
        return max(0.01, score)

    def candidates(self, req: CanonicalRequest, exclude: Optional[set] = None) -> List[dict]:
        exclude = exclude or set()
        pool = [c for c in store.CHANNELS if c["id"] not in exclude]
        compatible = [c for c in pool if self.is_compatible(c, req.model)]
        if not compatible:
            # Answering a claude-only request from an OpenAI channel silently returns the
            # wrong model, which is worse for an agent than a clear error. Opt in if wanted.
            compatible = pool if settings.CROSS_PROVIDER_FALLBACK else []
        if req.tools and compatible:
            # A text-only channel would drop the tools and answer in prose, which reads as a
            # model failure. Only fall back to one when the pool has nothing better.
            capable = [c for c in compatible if getattr(backends.BACKENDS.get(c["type"]), "supports_tools", False)]
            if capable:
                return capable
        return compatible

    def select(self, req: CanonicalRequest, exclude: Optional[set] = None) -> Optional[dict]:
        """Pick a channel at random, weighted by score.

        Weighting matters: picking uniformly among the top few would make priority, health,
        quota and latency purely decorative in a small pool, which is the common case.
        """
        candidates = self.candidates(req, exclude)
        if not candidates:
            return None
        scored = [(self.compute_score(ch, req), ch) for ch in candidates]
        live = [(score, ch) for score, ch in scored if score > 0]
        if not live:
            # Everything is cooling down: try the least bad rather than refuse outright.
            return max(scored, key=lambda entry: entry[0])[1]
        return random.choices([ch for _, ch in live], weights=[score for score, _ in live], k=1)[0]

    def resolve_model(self, ch: dict, requested_model: str) -> str:
        """Map the requested model onto this channel's own naming via aliases/defaults."""
        requested_model = requested_model or "auto"
        name = requested_model.lower()
        config = ch.get("config") or {}
        if name == "auto":
            return config.get("default_model") or settings.DEFAULT_MODELS.get(ch["type"], requested_model)
        for alias_key, alias_val in (config.get("aliases") or {}).items():
            if name == str(alias_key).lower() or name == str(alias_val).lower():
                return alias_val
        return requested_model

    async def acquire(self, ch: dict) -> asyncio.Semaphore:
        """Returns the semaphore actually acquired. Callers must hand it back to release():
        a concurrency change or channel delete swaps self.sems, and releasing the *new*
        semaphore would permanently leak a permit from the old one."""
        sem = self.get_sem(ch)
        await sem.acquire()
        self._get_stats(ch["id"])["in_flight"] += 1
        return sem

    def release(self, ch: dict, sem: Optional[asyncio.Semaphore] = None) -> None:
        sem = sem if sem is not None else self.sems.get(ch["id"])
        if sem is not None:
            try:
                sem.release()
            except ValueError:
                pass  # bounded semaphore already at its limit
        self._get_stats(ch["id"])["in_flight"] = max(0, self._get_stats(ch["id"])["in_flight"] - 1)

    @asynccontextmanager
    async def use_channel(self, ch: dict):
        sem = await self.acquire(ch)
        try:
            yield
        finally:
            self.release(ch, sem)

    def record_result(self, ch: dict, success: bool, latency: float, tokens_used: int = 0, headers: Optional[dict] = None):
        s = self._sync_stats_from_config(ch)
        s["calls"] += 1
        s["last_call"] = time.time()
        s["used_est_tokens"] += max(0, tokens_used)

        if success:
            # Latency is only meaningful for calls that produced an answer. Feeding failures
            # in would reward a channel for refusing connections quickly, partly cancelling
            # out the health penalty it just earned.
            s["avg_latency"] = (s["avg_latency"] * 0.7 + latency * 0.3) if s["success"] else latency
            s["success"] += 1
            s["consec_fail"] = 0
            s["health"] = min(1.0, s["health"] + 0.05)
        else:
            s["consec_fail"] += 1
            s["health"] = max(0.1, s["health"] * 0.6)

        parsed = parse_quota_remaining(headers)
        if parsed is not None:
            s["last_quota_remaining"], s["last_quota_unit"] = parsed
            s["last_quota_at"] = time.time()

        # Failure backoff grows with the failure streak. Exhausted quota needs its own floor:
        # it is reported on a *successful* call, where the streak is zero.
        cooldown = min(600.0, 30.0 * s["consec_fail"]) if s["consec_fail"] >= 3 else 0.0
        if quota_is_exhausted(self.fresh_quota_remaining(s), s.get("last_quota_unit", "tokens")):
            cooldown = max(cooldown, settings.QUOTA_COOLDOWN_SECONDS)
        if cooldown > 0:
            self.cooldowns[ch["id"]] = time.time() + cooldown

    def forget(self, cid: str) -> None:
        self.stats.pop(cid, None)
        self.cooldowns.pop(cid, None)
        self.sems.pop(cid, None)

    def get_status(self) -> list:
        out = []
        for ch in store.CHANNELS:
            s = self._sync_stats_from_config(ch)
            out.append({
                "id": ch["id"],
                "name": ch["name"],
                "type": ch["type"],
                "health": round(s["health"], 2),
                "used_est": s["used_est_tokens"],
                "quota_est": s["quota_est"],
                "remaining_est": max(0, s["quota_est"] - s["used_est_tokens"]),
                "in_flight": s["in_flight"],
                "consec_fail": s["consec_fail"],
                "avg_latency": round(s["avg_latency"], 1),
                "cooldown": self.cooldowns.get(ch["id"], 0) > time.time(),
                "cooldown_until": round(self.cooldowns.get(ch["id"], 0), 3) or None,
                "quota_category": s.get("quota_category", "general"),
                "last_quota_remaining": self.fresh_quota_remaining(s),
                "last_quota_unit": s.get("last_quota_unit", "tokens"),
                "calls": s["calls"],
                "success_rate": round(s["success"] / s["calls"], 3) if s["calls"] else None,
                "supports_tools": bool(getattr(backends.BACKENDS.get(ch["type"]), "supports_tools", False)),
            })
        return out


router = SmartRouter()
