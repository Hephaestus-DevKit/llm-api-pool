"""End-to-end endpoint behaviour with fake provider backends."""
from __future__ import annotations

import json

import main
from conftest import FakeBackend
from llm_pool import settings

CLAUDE_TOOLS = [{"name": "read_file", "description": "Read a file",
                 "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]


def anthropic_body(**overrides):
    body = {"model": "claude-sonnet-5", "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]}
    body.update(overrides)
    return body


# ---------------------------------------------------------------- the regression that mattered

def test_anthropic_request_with_tools_no_longer_500s(client, make_channel):
    """Claude Code sends tools on every request; this returned HTTP 500."""
    _ch, backend = make_channel("official_claude", backend=FakeBackend(text="done"))
    response = client.post("/v1/messages", json=anthropic_body(tools=CLAUDE_TOOLS))
    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "done"
    assert [t.name for t in backend.calls[0].tools] == ["read_file"]


def test_openai_request_with_tools_reaches_the_backend(client, make_channel):
    _ch, backend = make_channel("official_openai", backend=FakeBackend())
    response = client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "grep", "parameters": {}}}]})
    assert response.status_code == 200
    assert [t.name for t in backend.calls[0].tools] == ["grep"]


# ---------------------------------------------------------------- response shape

def test_anthropic_response_shape(client, make_channel):
    make_channel("official_claude", backend=FakeBackend(
        text="hello", usage={"prompt_tokens": 9, "completion_tokens": 3}))
    payload = client.post("/v1/messages", json=anthropic_body()).json()
    assert payload["type"] == "message"
    assert payload["role"] == "assistant"
    assert payload["stop_reason"] == "end_turn"
    assert payload["usage"] == {"input_tokens": 9, "output_tokens": 3}


def test_openai_response_reports_real_usage(client, make_channel):
    make_channel("official_openai", backend=FakeBackend(
        text="hello", usage={"prompt_tokens": 21, "completion_tokens": 4}))
    payload = client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}).json()
    assert payload["usage"] == {"prompt_tokens": 21, "completion_tokens": 4, "total_tokens": 25}
    assert payload["object"] == "chat.completion"


def test_tool_calls_reach_an_anthropic_client(client, make_channel):
    make_channel("official_claude", backend=FakeBackend(text="", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a"}'}}]))
    payload = client.post("/v1/messages", json=anthropic_body(tools=CLAUDE_TOOLS)).json()
    assert payload["stop_reason"] == "tool_use"
    assert payload["content"][0] == {"type": "tool_use", "id": "c1",
                                     "name": "read_file", "input": {"path": "a"}}


def test_tool_calls_reach_an_openai_client(client, make_channel):
    make_channel("official_openai", backend=FakeBackend(text="", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{}"}}]))
    payload = client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}).json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "grep"


# ---------------------------------------------------------------- routing and failover

def test_a_failing_channel_fails_over_to_the_next(client, make_channel):
    _bad, bad_backend = make_channel("official_claude", name="bad",
                                     backend=FakeBackend(error=RuntimeError("boom")))
    # Both channels share the type, so register the healthy backend as a second type
    # the same model is compatible with.
    good, good_backend = make_channel("web_claude", name="good", backend=FakeBackend(text="recovered"))
    good["config"]["priority"] = 3

    response = client.post("/v1/messages", json=anthropic_body())
    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "recovered"
    assert good_backend.calls, "healthy channel was never tried"


def test_every_channel_failing_reports_502_with_detail(client, make_channel):
    make_channel("official_claude", backend=FakeBackend(error=RuntimeError("upstream exploded")))
    response = client.post("/v1/messages", json=anthropic_body())
    assert response.status_code == 502
    assert "upstream exploded" in response.json()["detail"]


def test_failover_stops_after_max_attempts(client, make_channel, monkeypatch):
    monkeypatch.setattr(settings, "MAX_ROUTE_ATTEMPTS", 2)
    backends = []
    for ch_type in ("official_claude", "web_claude", "web_gemini"):
        _ch, backend = make_channel(ch_type, backend=FakeBackend(error=RuntimeError("no")))
        backends.append(backend)
    monkeypatch.setattr(settings, "CROSS_PROVIDER_FALLBACK", True)
    client.post("/v1/messages", json=anthropic_body())
    assert sum(len(b.calls) for b in backends) == 2


def test_no_matching_provider_returns_503_not_a_wrong_answer(client, make_channel):
    make_channel("official_openai", backend=FakeBackend(text="I am not Claude"))
    response = client.post("/v1/messages", json=anthropic_body())
    assert response.status_code == 503
    assert "claude-sonnet-5" in response.json()["detail"]


def test_empty_pool_returns_503(client):
    assert client.post("/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": "hi"}]}).status_code == 503


def test_a_web_rate_limit_surfaces_as_429(client, make_channel):
    make_channel("web_claude", backend=FakeBackend(error=RuntimeError("WEB_RATE_LIMIT:acct")))
    assert client.post("/v1/messages", json=anthropic_body()).status_code == 429


def test_channel_stats_record_the_failure(client, make_channel):
    make_channel("official_claude", backend=FakeBackend(error=RuntimeError("boom")))
    client.post("/v1/messages", json=anthropic_body())
    assert main.router.stats["ch0"]["consec_fail"] == 1
    assert main.router.stats["ch0"]["in_flight"] == 0


def test_a_successful_call_records_usage_against_the_channel(client, make_channel):
    make_channel("official_claude", backend=FakeBackend(
        text="ok", usage={"prompt_tokens": 10, "completion_tokens": 5}))
    client.post("/v1/messages", json=anthropic_body())
    assert main.router.stats["ch0"]["used_est_tokens"] == 15
    assert main.router.stats["ch0"]["in_flight"] == 0


# ---------------------------------------------------------------- misc endpoints

def test_health_is_public(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_models_list_is_openai_shaped(client, make_channel):
    make_channel("official_claude", aliases={"fast": "claude-haiku-4-5-20251001"})
    payload = client.get("/v1/models").json()
    assert payload["object"] == "list"
    ids = {entry["id"] for entry in payload["data"]}
    assert {"fast", "auto", main.DEFAULT_MODELS["official_claude"]} <= ids
    assert all(entry["object"] == "model" for entry in payload["data"])


def test_models_list_deduplicates_across_channels(client, make_channel):
    make_channel("official_claude", name="a")
    make_channel("web_claude", name="b", default_model=main.DEFAULT_MODELS["official_claude"])
    ids = [entry["id"] for entry in client.get("/v1/models").json()["data"]]
    assert len(ids) == len(set(ids))


def test_channels_json_never_stores_runtime_counters(pool, tmp_path):
    pool.CHANNELS.append({"id": "c", "type": "official_claude", "name": "c",
                          "config": {"api_key": "k"}, "stats": {"calls": 99}})
    pool.save_channels()
    saved = json.loads((tmp_path / "channels.json").read_text(encoding="utf-8"))
    assert "stats" not in saved[0]
