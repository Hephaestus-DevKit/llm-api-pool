"""Dialect conversion. These are the paths agents exercise on literally every request."""
from __future__ import annotations

import json

import main


# ---------------------------------------------------------------- tools

def test_anthropic_tools_are_accepted():
    """Claude Code sends {"name", "input_schema"}; this used to raise and return HTTP 500."""
    req = main.anthropic_to_canonical({
        "model": "claude-sonnet-5",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "read_file", "description": "Read a file",
                   "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
    })
    assert [t.name for t in req.tools] == ["read_file"]
    assert req.tools[0].parameters["properties"]["path"]["type"] == "string"


def test_openai_tools_are_accepted():
    req = main.openai_to_canonical({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {
            "name": "read_file", "description": "Read", "parameters": {"type": "object"}}}],
    })
    assert [t.name for t in req.tools] == ["read_file"]


def test_tool_round_trip_between_dialects():
    req = main.anthropic_to_canonical({
        "model": "m", "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "grep", "description": "search", "input_schema": {"type": "object"}}],
    })
    as_openai = main.canonical_tools_to_openai(req.tools)
    assert as_openai[0]["type"] == "function"
    assert as_openai[0]["function"]["name"] == "grep"
    assert main.canonical_tools_to_anthropic(req.tools)[0]["input_schema"] == {"type": "object"}


def test_malformed_tool_is_skipped_not_fatal():
    req = main.openai_to_canonical({
        "model": "m", "messages": [],
        "tools": ["garbage", {"no_name": 1}, {"name": "good", "input_schema": {}}],
    })
    assert [t.name for t in req.tools] == ["good"]


def test_tools_absent_stays_none():
    assert main.openai_to_canonical({"model": "m", "messages": []}).tools is None


def test_tool_choice_round_trips():
    assert main.tool_choice_to_canonical("auto") == {"mode": "auto"}
    assert main.tool_choice_to_canonical({"type": "any"}) == {"mode": "required"}
    assert main.tool_choice_to_canonical({"type": "tool", "name": "x"}) == {"mode": "tool", "name": "x"}
    assert main.tool_choice_to_canonical(
        {"type": "function", "function": {"name": "x"}}) == {"mode": "tool", "name": "x"}

    choice = {"mode": "tool", "name": "x"}
    assert main.canonical_tool_choice_to_openai(choice) == {"type": "function", "function": {"name": "x"}}
    assert main.canonical_tool_choice_to_anthropic(choice) == {"type": "tool", "name": "x"}
    assert main.canonical_tool_choice_to_anthropic({"mode": "required"}) == {"type": "any"}
    # Anthropic has no "none"; the backend drops the tools array instead.
    assert main.canonical_tool_choice_to_anthropic({"mode": "none"}) is None


# ---------------------------------------------------------------- content blocks

def test_tool_result_block_is_not_stringified_as_python_repr():
    req = main.anthropic_to_canonical({"model": "m", "messages": [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "72F"}]}]})
    block = req.messages[0].content[0]
    assert block == {"type": "tool_result", "tool_use_id": "tu_1", "content": "72F", "is_error": False}
    assert "'type':" not in json.dumps(req.messages[0].content)


def test_openai_tool_message_becomes_a_tool_result_block():
    req = main.openai_to_canonical({"model": "m", "messages": [
        {"role": "tool", "tool_call_id": "call_1", "content": "42"}]})
    assert req.messages[0].role == "user"
    assert req.messages[0].content[0]["tool_use_id"] == "call_1"


def test_openai_assistant_tool_calls_become_tool_use_blocks():
    req = main.openai_to_canonical({"model": "m", "messages": [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "grep", "arguments": '{"q": "x"}'}}]}]})
    block = req.messages[0].content[0]
    assert block["type"] == "tool_use"
    assert block["name"] == "grep"
    assert block["input"] == {"q": "x"}


def test_tool_use_survives_canonical_to_openai_messages():
    req = main.anthropic_to_canonical({"model": "m", "messages": [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_1", "name": "grep", "input": {"q": "x"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "found"}]},
    ]})
    messages = main.canonical_messages_to_openai(req)
    assert messages[0]["tool_calls"][0]["function"]["name"] == "grep"
    assert json.loads(messages[0]["tool_calls"][0]["function"]["arguments"]) == {"q": "x"}
    assert messages[1] == {"role": "tool", "tool_call_id": "tu_1", "content": "found"}


def test_anthropic_messages_merge_consecutive_same_role_turns():
    """Two OpenAI tool messages in a row become two user turns; Anthropic requires one."""
    req = main.openai_to_canonical({"model": "m", "messages": [
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "content": "2"},
    ]})
    messages = main.canonical_messages_to_anthropic(req)
    assert len(messages) == 1
    assert [b["tool_use_id"] for b in messages[0]["content"]] == ["a", "b"]


def test_anthropic_messages_drop_empty_turns():
    req = main.openai_to_canonical({"model": "m", "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]})
    assert [m["role"] for m in main.canonical_messages_to_anthropic(req)] == ["user"]


def test_thinking_blocks_are_dropped_not_stringified():
    """Claude Code replays extended-thinking turns; other providers must not see the raw
    JSON of a thinking block pasted into the prompt as prose."""
    req = main.anthropic_to_canonical({"model": "m", "messages": [
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "secret chain of thought", "signature": "sig"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "the answer"}]}]})
    assert req.messages[0].content == "the answer"
    assert "chain of thought" not in json.dumps(main.canonical_messages_to_openai(req))


def test_images_convert_between_dialects():
    req = main.anthropic_to_canonical({"model": "m", "messages": [
        {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
            {"type": "text", "text": "what is this"}]}]})
    openai_messages = main.canonical_messages_to_openai(req)
    assert openai_messages[0]["content"][0]["image_url"]["url"] == "data:image/png;base64,AAA"

    back = main.openai_to_canonical({"model": "m", "messages": openai_messages})
    assert main.canonical_messages_to_anthropic(back)[0]["content"][0]["source"]["data"] == "AAA"


def test_system_blocks_are_joined_not_repr():
    req = main.anthropic_to_canonical({"model": "m", "messages": [],
                                       "system": [{"type": "text", "text": "a"},
                                                  {"type": "text", "text": "b"}]})
    assert req.system == "a\n\nb"


def test_multiple_openai_system_messages_are_kept():
    req = main.openai_to_canonical({"model": "m", "messages": [
        {"role": "system", "content": "first"},
        {"role": "system", "content": "second"},
        {"role": "user", "content": "hi"}]})
    assert req.system == "first\n\nsecond"


def test_system_prompt_reaches_text_only_channels():
    req = main.openai_to_canonical({"model": "m", "messages": [
        {"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]})
    assert "System: be terse" in main.canonical_to_text_prompt(req)


# ---------------------------------------------------------------- responses

def test_usage_uses_real_provider_counts():
    resp = main.make_canonical_response("hello", usage={"prompt_tokens": 11, "completion_tokens": 7})
    assert resp.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def test_usage_reads_anthropic_and_gemini_field_names():
    assert main.normalize_usage({"input_tokens": 3, "output_tokens": 4})["total_tokens"] == 7
    assert main.normalize_usage(
        {"prompt_token_count": 5, "candidates_token_count": 6})["total_tokens"] == 11


def test_usage_falls_back_to_an_estimate_only_when_absent():
    assert main.make_canonical_response("x" * 400).usage["completion_tokens"] == 100


def test_stop_reason_maps_max_tokens():
    resp = main.make_canonical_response("hi", finish_reason="length")
    assert main.canonical_to_anthropic(resp)["stop_reason"] == "max_tokens"


def test_stop_reason_maps_tool_use():
    resp = main.make_canonical_response("", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}])
    payload = main.canonical_to_anthropic(resp)
    assert payload["stop_reason"] == "tool_use"
    assert payload["content"][0] == {"type": "tool_use", "id": "c1", "name": "f", "input": {}}


def test_openai_response_carries_tool_calls():
    resp = main.make_canonical_response("", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}])
    payload = main.canonical_to_openai(resp)
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "f"


# ---------------------------------------------------------------- input hardening

def test_garbage_scalars_do_not_raise():
    req = main.openai_to_canonical({"model": None, "messages": None,
                                    "temperature": "hot", "max_tokens": "many"})
    assert (req.model, req.temperature, req.max_tokens) == ("auto", None, None)


def test_an_unspecified_temperature_stays_unspecified():
    """Substituting a value would override every provider's own default, and reasoning
    models reject an explicit temperature entirely."""
    assert main.openai_to_canonical({"model": "m", "messages": []}).temperature is None
    assert main.anthropic_to_canonical({"model": "m", "messages": []}).temperature is None


def test_temperature_is_clamped_and_zero_is_preserved():
    assert main.openai_to_canonical({"temperature": 99}).temperature == 2.0
    assert main.openai_to_canonical({"temperature": -5}).temperature == 0.0
    assert main.openai_to_canonical({"temperature": 0}).temperature == 0.0


def test_reasoning_models_are_recognised():
    for name in ("o1", "o3-mini", "O4-preview"):
        assert main.is_reasoning_model(name) is True
    for name in ("gpt-4o", "claude-sonnet-5", "", None):
        assert main.is_reasoning_model(name) is False


def test_non_positive_max_tokens_is_dropped():
    assert main.openai_to_canonical({"max_tokens": 0}).max_tokens is None


def test_max_completion_tokens_is_accepted():
    """Newer OpenAI clients send max_completion_tokens instead of max_tokens."""
    assert main.openai_to_canonical({"max_completion_tokens": 77}).max_tokens == 77
    assert main.openai_to_canonical({"max_tokens": 5, "max_completion_tokens": 77}).max_tokens == 5


def test_bad_tool_arguments_json_does_not_raise():
    assert main.parse_json_object("{not json") == {}
    assert main.parse_json_object('"scalar"') == {"value": "scalar"}


def test_missing_gemini_sdk_yields_an_actionable_error(monkeypatch):
    """The native ARM64 build ships without google-genai; the failure must say so
    instead of surfacing a bare ModuleNotFoundError."""
    import asyncio
    import sys

    import pytest

    monkeypatch.setitem(sys.modules, "google", None)
    monkeypatch.setitem(sys.modules, "google.genai", None)
    backend = main.OfficialGeminiBackend()
    with pytest.raises(RuntimeError, match="web_gemini"):
        asyncio.run(backend._client({"config": {"api_key": "k"}}))
