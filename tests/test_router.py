"""Channel selection, quota accounting, and concurrency bookkeeping."""
from __future__ import annotations

import asyncio

import pytest

import main
from llm_pool import settings


def request_for(model="auto", **kwargs):
    return main.CanonicalRequest(model=model, messages=[], **kwargs)


def channel(cid, ch_type, **config):
    return {"id": cid, "type": ch_type, "name": cid, "config": config}


# ---------------------------------------------------------------- compatibility

@pytest.mark.parametrize("model, ch_type, expected", [
    ("claude-sonnet-5", "official_claude", True),
    ("claude-sonnet-5", "official_openai", False),
    ("claude-opus-5", "web_claude", True),
    ("gpt-4o", "official_openai", True),
    ("gpt-4o", "official_gemini", False),
    ("o3-mini", "official_openai", True),
    ("gemini-2.5-flash", "official_gemini", True),
    ("codex-mini", "web_codex", True),
    ("codex-mini", "official_openai", True),
    ("auto", "official_gemini", True),
])
def test_is_compatible(model, ch_type, expected):
    assert main.SmartRouter().is_compatible(channel("c", ch_type), model) is expected


def test_aliases_make_any_channel_compatible():
    ch = channel("c", "official_openai", aliases={"sonnet": "gpt-4o"})
    assert main.SmartRouter().is_compatible(ch, "sonnet") is True


# ---------------------------------------------------------------- selection

def test_select_prefers_a_compatible_channel(pool):
    pool.CHANNELS.extend([channel("openai", "official_openai"), channel("claude", "official_claude")])
    router = main.SmartRouter()
    assert router.select(request_for("claude-sonnet-5"))["id"] == "claude"


def test_select_refuses_the_wrong_provider_by_default(pool, monkeypatch):
    """Answering a claude request from an OpenAI key silently returns the wrong model."""
    monkeypatch.setattr(settings, "CROSS_PROVIDER_FALLBACK", False)
    pool.CHANNELS.append(channel("openai", "official_openai"))
    assert main.SmartRouter().select(request_for("claude-sonnet-5")) is None


def test_cross_provider_fallback_can_be_opted_into(pool, monkeypatch):
    monkeypatch.setattr(settings, "CROSS_PROVIDER_FALLBACK", True)
    pool.CHANNELS.append(channel("openai", "official_openai"))
    assert main.SmartRouter().select(request_for("claude-sonnet-5"))["id"] == "openai"


def test_select_honours_the_exclude_set_so_failover_advances(pool):
    pool.CHANNELS.extend([channel("a", "official_claude"), channel("b", "official_claude")])
    router = main.SmartRouter()
    assert router.select(request_for("claude-sonnet-5"), exclude={"a"})["id"] == "b"
    assert router.select(request_for("claude-sonnet-5"), exclude={"a", "b"}) is None


def test_empty_pool_selects_nothing(pool):
    assert main.SmartRouter().select(request_for()) is None


def test_a_cooling_channel_loses_to_a_healthy_one(pool):
    pool.CHANNELS.extend([channel("cold", "official_claude"), channel("warm", "official_claude")])
    router = main.SmartRouter()
    router.cooldowns["cold"] = main.time.time() + 300
    assert router.compute_score(pool.CHANNELS[0], request_for()) == 0.0
    assert all(router.select(request_for())["id"] == "warm" for _ in range(15))


def test_everything_cooling_still_returns_a_channel(pool):
    """Refusing every request would be worse than trying the least-bad account."""
    pool.CHANNELS.append(channel("only", "official_claude"))
    router = main.SmartRouter()
    router.cooldowns["only"] = main.time.time() + 300
    assert router.select(request_for())["id"] == "only"


def test_tool_requests_deprioritise_channels_that_cannot_carry_them(pool):
    tool = main.CanonicalTool(name="grep", parameters={})
    gemini = channel("g", "official_gemini")
    claude = channel("c", "official_claude")
    pool.CHANNELS.extend([gemini, claude])
    router = main.SmartRouter()
    with_tools = request_for(tools=[tool])
    assert router.compute_score(gemini, with_tools) < router.compute_score(gemini, request_for())


def test_a_tool_request_never_lands_on_a_text_only_channel(pool):
    """A text-only channel silently drops the tools and answers in prose, which reads to the
    caller as the model ignoring its instructions."""
    pool.CHANNELS.extend([channel("gem", "official_gemini"), channel("web", "web_chatgpt"),
                          channel("claude", "official_claude")])
    router = main.SmartRouter()
    with_tools = request_for(tools=[main.CanonicalTool(name="grep", parameters={})])
    assert {router.select(with_tools)["id"] for _ in range(40)} == {"claude"}


def test_a_text_only_channel_is_still_used_when_nothing_else_exists(pool):
    pool.CHANNELS.append(channel("gem", "official_gemini"))
    router = main.SmartRouter()
    with_tools = request_for("gemini-2.5-flash", tools=[main.CanonicalTool(name="grep", parameters={})])
    assert router.select(with_tools)["id"] == "gem"


def test_selection_is_weighted_by_score_not_uniform(pool):
    """Picking uniformly among the top few made priority, health and quota decorative."""
    high = channel("high", "official_claude", priority=3)
    low = channel("low", "official_claude", priority=1)
    pool.CHANNELS.extend([high, low])
    router = main.SmartRouter()
    picks = [router.select(request_for("claude-sonnet-5"))["id"] for _ in range(2000)]
    share = picks.count("high") / len(picks)
    assert 0.6 < share < 0.9, f"expected the high-priority channel to dominate, got {share:.2f}"


def test_an_unhealthy_channel_loses_most_traffic(pool):
    good = channel("good", "official_claude")
    bad = channel("bad", "official_claude")
    pool.CHANNELS.extend([good, bad])
    router = main.SmartRouter()
    for _ in range(3):
        router.record_result(bad, False, 0.1)
    router.cooldowns.clear()  # isolate health from cooldown
    req = request_for("claude-sonnet-5")
    assert router.compute_score(good, req) > router.compute_score(bad, req) * 2
    picks = [router.select(req)["id"] for _ in range(2000)]
    assert picks.count("good") / len(picks) > 0.65


def test_failures_do_not_earn_a_latency_bonus(pool):
    """A refused connection returns fast; counting that as low latency would partly cancel
    the health penalty the same failure just applied."""
    ch = channel("c", "official_claude")
    router = main.SmartRouter()
    baseline = router._get_stats("c")["avg_latency"]
    for _ in range(3):
        router.record_result(ch, False, 0.01)
    assert router.stats["c"]["avg_latency"] == baseline
    router.record_result(ch, True, 2.0)
    assert router.stats["c"]["avg_latency"] == 2.0


# ---------------------------------------------------------------- model resolution

def test_auto_resolves_to_the_channel_default(pool):
    router = main.SmartRouter()
    assert router.resolve_model(channel("c", "official_claude"), "auto") == main.DEFAULT_MODELS["official_claude"]
    assert router.resolve_model(channel("c", "official_claude", default_model="custom"), "auto") == "custom"


def test_alias_resolves_to_its_target(pool):
    ch = channel("c", "official_claude", aliases={"fast": "claude-haiku-4-5-20251001"})
    assert main.SmartRouter().resolve_model(ch, "fast") == "claude-haiku-4-5-20251001"


def test_unknown_model_passes_through(pool):
    assert main.SmartRouter().resolve_model(channel("c", "official_claude"), "x-1") == "x-1"


# ---------------------------------------------------------------- concurrency

def test_release_returns_the_semaphore_that_was_acquired(pool):
    """Changing max_concurrent swaps router.sems; releasing the new one would leak a permit
    from the old semaphore and eventually deadlock the channel."""
    ch = channel("c", "web_claude", max_concurrent=1)

    async def scenario():
        router = main.SmartRouter()
        sem = await router.acquire(ch)
        router.sems.pop("c", None)          # simulate a concurrency update mid-flight
        router.release(ch, sem)
        assert sem._value == 1              # the original permit came back
        assert router.stats["c"]["in_flight"] == 0

    asyncio.run(scenario())


def test_concurrency_limit_is_enforced(pool):
    ch = channel("c", "web_claude", max_concurrent=1)

    async def scenario():
        router = main.SmartRouter()
        first = await router.acquire(ch)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(router.acquire(ch), timeout=0.05)
        router.release(ch, first)
        second = await asyncio.wait_for(router.acquire(ch), timeout=0.5)
        router.release(ch, second)

    asyncio.run(scenario())


def test_use_channel_releases_on_exception(pool):
    ch = channel("c", "official_claude", max_concurrent=2)

    async def scenario():
        router = main.SmartRouter()
        with pytest.raises(RuntimeError):
            async with router.use_channel(ch):
                raise RuntimeError("upstream blew up")
        assert router.stats["c"]["in_flight"] == 0
        assert router.get_sem(ch)._value == 2

    asyncio.run(scenario())


def test_forget_clears_all_per_channel_state(pool):
    router = main.SmartRouter()
    router._get_stats("c")
    router.cooldowns["c"] = 1.0
    router.get_sem(channel("c", "official_claude"))
    router.forget("c")
    assert "c" not in router.stats and "c" not in router.cooldowns and "c" not in router.sems


def test_invalid_max_concurrent_falls_back_to_a_default(pool):
    sem = main.SmartRouter().get_sem(channel("c", "web_claude", max_concurrent="lots"))
    assert sem._value == 2


def test_concurrency_limit_is_clamped(pool):
    router = main.SmartRouter()
    assert router.concurrency_limit(channel("a", "web_claude", max_concurrent=10_000)) == 64
    assert router.concurrency_limit(channel("b", "web_claude", max_concurrent=0)) == 1
    assert router.concurrency_limit(channel("c", "web_claude")) == 2
    assert router.concurrency_limit(channel("d", "official_claude")) == 8


def test_a_saturated_channel_is_routed_around(pool):
    """Selecting a full channel makes the caller queue on its semaphore with no timeout."""
    busy = channel("busy", "official_claude", max_concurrent=1)
    free = channel("free", "official_claude", max_concurrent=1)
    pool.CHANNELS.extend([busy, free])
    router = main.SmartRouter()
    router._get_stats("busy")["in_flight"] = 1
    req = request_for("claude-sonnet-5")
    assert router.compute_score(busy, req) < router.compute_score(free, req) / 5
    picks = [router.select(req)["id"] for _ in range(500)]
    assert picks.count("free") / len(picks) > 0.9


def test_everything_saturated_still_returns_a_channel(pool):
    only = channel("only", "official_claude", max_concurrent=1)
    pool.CHANNELS.append(only)
    router = main.SmartRouter()
    router._get_stats("only")["in_flight"] = 1
    assert router.select(request_for("claude-sonnet-5"))["id"] == "only"


# ---------------------------------------------------------------- quota accounting

def test_token_budget_header_wins_over_the_request_budget():
    headers = {"x-ratelimit-remaining-requests": "5", "x-ratelimit-remaining-tokens": "90000"}
    assert main.parse_quota_remaining(headers) == (90000, "tokens")


def test_anthropic_quota_headers_are_understood():
    assert main.parse_quota_remaining({"anthropic-ratelimit-tokens-remaining": "12"}) == (12, "tokens")
    assert main.parse_quota_remaining(
        {"anthropic-ratelimit-requests-remaining": "7"}) == (7, "requests")


def test_unrelated_remaining_headers_are_ignored():
    """Substring matching used to latch onto anything containing "remaining"."""
    assert main.parse_quota_remaining({"x-pages-remaining": "3"}) is None
    assert main.parse_quota_remaining({}) is None
    assert main.parse_quota_remaining(None) is None


def test_a_low_request_budget_is_not_read_as_an_exhausted_account():
    """"50 requests left" is healthy; the 100-unit threshold only makes sense for tokens."""
    assert main.quota_is_exhausted(50, "requests") is False
    assert main.quota_is_exhausted(50, "tokens") is True
    assert main.quota_is_exhausted(0, "requests") is True
    assert main.quota_is_exhausted(None, "tokens") is False


def test_zero_remaining_is_respected_not_treated_as_missing(pool):
    ch = channel("c", "official_claude", quota=1000)
    router = main.SmartRouter()
    router.record_result(ch, True, 0.1, 0, {"x-ratelimit-remaining-tokens": "0"})
    assert router.stats["c"]["last_quota_remaining"] == 0
    assert router.cooldowns.get("c", 0) > main.time.time()  # exhausted, so cool it down


def test_a_stale_quota_reading_stops_parking_the_channel(pool, monkeypatch):
    """Streaming exposes no headers, so without an expiry one low reading would park a
    streaming-only channel (Claude Code, Cursor) forever."""
    ch = channel("c", "official_claude", quota=1000)
    router = main.SmartRouter()
    router.record_result(ch, True, 0.1, 0, {"x-ratelimit-remaining-tokens": "5"})
    stats = router.stats["c"]
    assert router.fresh_quota_remaining(stats) == 5

    stats["last_quota_at"] -= main.QUOTA_STALE_SECONDS + 1
    assert router.fresh_quota_remaining(stats) is None
    router.cooldowns.clear()
    router.record_result(ch, True, 0.1, 0, None)
    assert router.cooldowns.get("c", 0) <= main.time.time()
    assert router.compute_score(ch, request_for()) > 0


def test_repeated_failures_trigger_a_bounded_cooldown(pool):
    ch = channel("c", "official_claude")
    router = main.SmartRouter()
    for _ in range(3):
        router.record_result(ch, False, 0.1)
    assert router.cooldowns["c"] > main.time.time()
    for _ in range(50):
        router.record_result(ch, False, 0.1)
    assert router.cooldowns["c"] - main.time.time() <= 600


def test_success_restores_health(pool):
    ch = channel("c", "official_claude")
    router = main.SmartRouter()
    router.record_result(ch, False, 0.1)
    degraded = router.stats["c"]["health"]
    router.record_result(ch, True, 0.1)
    assert router.stats["c"]["health"] > degraded
    assert router.stats["c"]["consec_fail"] == 0


def test_status_does_not_leak_config(pool):
    pool.CHANNELS.append(channel("c", "official_claude", api_key="sk-secret", quota=10))
    row = main.SmartRouter().get_status()[0]
    assert "sk-secret" not in str(row)
    assert row["supports_tools"] is True
