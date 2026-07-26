"""What each backend actually sends to its SDK.

The provider SDKs are not installed in CI, so these tests inspect the kwargs the backends
build rather than the calls they make. That is where the request-shaping bugs live: a value
that the API rejects fails identically on every channel, so failover cannot recover from it.
"""
from __future__ import annotations


import main
import pytest


def request_for(dialect="openai", **body):
    base = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    base.update(body)
    return (main.openai_to_canonical if dialect == "openai" else main.anthropic_to_canonical)(base)


# ---------------------------------------------------------------- openai

def openai_kwargs(req, stream=False):
    return main.OfficialOpenAIBackend()._kwargs(req, stream=stream)


def test_temperature_is_omitted_when_the_caller_did_not_set_one():
    assert "temperature" not in openai_kwargs(request_for())


def test_temperature_is_forwarded_when_set():
    assert openai_kwargs(request_for(temperature=0.2))["temperature"] == 0.2


def test_reasoning_models_get_no_temperature_and_max_completion_tokens():
    """o-series models reject `temperature` and `max_tokens`; both would 400 on every
    channel, so failover would burn through the whole pool."""
    kwargs = openai_kwargs(request_for(model="o3-mini", temperature=0.7, max_tokens=500))
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs
    assert kwargs["max_completion_tokens"] == 500


def test_ordinary_models_keep_max_tokens():
    kwargs = openai_kwargs(request_for(max_tokens=500))
    assert kwargs["max_tokens"] == 500
    assert "max_completion_tokens" not in kwargs


def test_tools_are_forwarded_in_openai_shape():
    req = request_for(tools=[{"type": "function", "function": {"name": "grep", "parameters": {}}}],
                      tool_choice="required")
    kwargs = openai_kwargs(req)
    assert kwargs["tools"][0]["function"]["name"] == "grep"
    assert kwargs["tool_choice"] == "required"


def test_no_tools_means_no_tool_keys():
    kwargs = openai_kwargs(request_for())
    assert "tools" not in kwargs and "tool_choice" not in kwargs


def test_streaming_asks_for_usage():
    assert openai_kwargs(request_for(), stream=True)["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------- anthropic

def anthropic_kwargs(req):
    return main.OfficialAnthropicBackend()._kwargs(req)


def test_system_is_omitted_rather_than_sent_as_null():
    """The SDK distinguishes an omitted argument from an explicit None, and None is
    serialized as a literal JSON null that the API rejects."""
    assert "system" not in anthropic_kwargs(request_for())
    with_system = request_for(messages=[{"role": "system", "content": "be terse"},
                                        {"role": "user", "content": "hi"}])
    assert anthropic_kwargs(with_system)["system"] == "be terse"


def test_temperature_is_clamped_to_the_anthropic_range():
    """Canonical carries OpenAI's 0..2; Anthropic only accepts 0..1."""
    assert anthropic_kwargs(request_for(temperature=1.5))["temperature"] == 1.0
    assert anthropic_kwargs(request_for(temperature=0.3))["temperature"] == 0.3
    assert "temperature" not in anthropic_kwargs(request_for())


def test_max_tokens_is_required_so_it_gets_a_default():
    assert anthropic_kwargs(request_for())["max_tokens"] == main.ANTHROPIC_DEFAULT_MAX_TOKENS
    assert anthropic_kwargs(request_for(max_tokens=100))["max_tokens"] == 100


def test_a_system_only_request_still_produces_a_message():
    """Anthropic rejects an empty messages array, but system-only is valid in the OpenAI
    dialect, so the same request must not 502 just because it routed to a Claude channel."""
    req = request_for(messages=[{"role": "system", "content": "you are a haiku bot"}])
    kwargs = anthropic_kwargs(req)
    assert kwargs["messages"] == [{"role": "user", "content": "you are a haiku bot"}]
    assert "system" not in kwargs  # not duplicated


def test_a_request_with_no_content_at_all_is_rejected_clearly():
    with pytest.raises(ValueError, match="no prompt content"):
        anthropic_kwargs(request_for(messages=[]))


def test_tool_choice_none_drops_the_tools_array():
    """Anthropic has no "none" mode; the only way to express it is to send no tools."""
    req = request_for(tools=[{"type": "function", "function": {"name": "grep", "parameters": {}}}],
                      tool_choice="none")
    assert "tools" not in anthropic_kwargs(req)


def test_tools_are_forwarded_in_anthropic_shape():
    req = request_for(tools=[{"type": "function", "function": {
        "name": "grep", "description": "search", "parameters": {"type": "object"}}}])
    tool = anthropic_kwargs(req)["tools"][0]
    assert tool == {"name": "grep", "description": "search", "input_schema": {"type": "object"}}


# ---------------------------------------------------------------- conversation shape

def test_an_empty_assistant_turn_does_not_fuse_its_neighbours():
    """Dropping the empty turn must not present two separate questions as one."""
    req = main.openai_to_canonical({"model": "m", "messages": [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "second question"},
    ]})
    messages = main.canonical_messages_to_anthropic(req)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"][0]["text"] == main.EMPTY_TURN_PLACEHOLDER
    assert messages[0]["content"] == "first question"
    assert messages[2]["content"] == "second question"


def test_genuinely_adjacent_same_role_turns_still_merge():
    req = main.openai_to_canonical({"model": "m", "messages": [
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "content": "2"},
    ]})
    messages = main.canonical_messages_to_anthropic(req)
    assert len(messages) == 1
    assert [b["tool_use_id"] for b in messages[0]["content"]] == ["a", "b"]


def test_alternation_is_never_violated():
    req = main.openai_to_canonical({"model": "m", "messages": [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "c"},
    ]})
    roles = [m["role"] for m in main.canonical_messages_to_anthropic(req)]
    assert all(a != b for a, b in zip(roles, roles[1:], strict=False)), roles


# ---------------------------------------------------------------- structured tool results

def test_an_image_tool_result_survives_anthropic_to_anthropic():
    """Screenshot-returning tools (computer use, Playwright MCP) put image blocks here.
    Flattening them loses the image and injects a base64 blob as prompt text."""
    req = main.anthropic_to_canonical({"model": "m", "messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "text", "text": "screenshot:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
        ]}]}]})
    block = req.messages[0].content[0]
    assert isinstance(block["content"], list)
    assert block["content"][1] == {"type": "image", "media_type": "image/png", "data": "AAA"}

    rendered = main.canonical_messages_to_anthropic(req)[0]["content"][0]
    assert rendered["content"][1]["source"] == {
        "type": "base64", "media_type": "image/png", "data": "AAA"}


def test_a_text_only_tool_result_stays_a_plain_string():
    req = main.anthropic_to_canonical({"model": "m", "messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "42"}]}]}]})
    assert req.messages[0].content[0]["content"] == "42"


def test_structured_tool_results_flatten_for_openai():
    """OpenAI's tool message is text-only, so this is the one place flattening is correct."""
    req = main.anthropic_to_canonical({"model": "m", "messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "text", "text": "ok"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
        ]}]}]})
    message = main.canonical_messages_to_openai(req)[0]
    assert message["role"] == "tool"
    assert isinstance(message["content"], str)
    assert "ok" in message["content"]
