"""SSE encoding for both dialects, including streamed tool calls."""
from __future__ import annotations

import json

import main
from conftest import FakeBackend


def anthropic_body(**overrides):
    body = {"model": "claude-sonnet-5", "max_tokens": 64, "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}
    body.update(overrides)
    return body


def sse_blocks(raw: str):
    """Parse an SSE body into (event_name, payload) pairs, skipping [DONE]."""
    out = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        name = None
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if data in (None, "[DONE]"):
            continue
        out.append((name, json.loads(data)))
    return out


TEXT_STREAM = [
    {"choices": [{"delta": {"content": "Hel"}}]},
    {"choices": [{"delta": {"content": "lo"}}]},
    {"choices": [{"finish_reason": "stop"}]},
    {"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
]

# What OfficialAnthropicBackend actually emits: message_delta carries stop_reason and usage
# in the SAME event, so a consumer that only harvests usage from choice-less chunks drops it.
ANTHROPIC_SHAPED_STREAM = [
    {"choices": [{"delta": {"content": "Hello"}}]},
    {"choices": [{"finish_reason": "stop"}],
     "usage": {"prompt_tokens": 1234, "completion_tokens": 567, "total_tokens": 1801}},
]

TOOL_STREAM = [
    {"choices": [{"delta": {"content": "let me look"}}]},
    {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "c1", "type": "function",
         "function": {"name": "read_file", "arguments": ""}}]}}]},
    {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "type": "function", "function": {"arguments": '{"path"'}}]}}]},
    {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "type": "function", "function": {"arguments": ':"a.py"}'}}]}}]},
    {"choices": [{"finish_reason": "tool_calls"}]},
]


# ---------------------------------------------------------------- anthropic SSE

def test_anthropic_text_stream_has_a_well_formed_event_sequence(client, make_channel):
    make_channel("official_claude", backend=FakeBackend(stream_chunks=TEXT_STREAM))
    events = sse_blocks(client.post("/v1/messages", json=anthropic_body()).text)
    names = [name for name, _ in events]
    assert names[0] == "message_start"
    assert names[-2:] == ["message_delta", "message_stop"]
    assert names.count("content_block_start") == names.count("content_block_stop") == 1
    text = "".join(payload["delta"]["text"] for name, payload in events
                   if name == "content_block_delta")
    assert text == "Hello"


def test_anthropic_stream_reports_real_output_tokens(client, make_channel):
    make_channel("official_claude", backend=FakeBackend(stream_chunks=TEXT_STREAM))
    events = dict((name, payload) for name, payload in
                  sse_blocks(client.post("/v1/messages", json=anthropic_body()).text))
    assert events["message_delta"]["usage"]["output_tokens"] == 2


def test_usage_is_harvested_when_it_shares_a_chunk_with_the_stop_reason(client, make_channel):
    """Anthropic sends both in one message_delta; requiring a choice-less chunk lost it, so
    every Anthropic stream reported a character-count estimate instead of real tokens."""
    channel, _backend = make_channel("official_claude", backend=FakeBackend(
        stream_chunks=ANTHROPIC_SHAPED_STREAM))
    events = dict(sse_blocks(client.post("/v1/messages", json=anthropic_body()).text))
    assert events["message_delta"]["usage"]["output_tokens"] == 567
    # The same number must reach the router, or quota-based routing drifts by orders of magnitude.
    assert main.router.stats[channel["id"]]["used_est_tokens"] == 1801


def test_openai_stream_reports_usage_that_shared_a_chunk(client, make_channel):
    make_channel("official_openai", backend=FakeBackend(stream_chunks=ANTHROPIC_SHAPED_STREAM))
    frames = [p for _n, p in sse_blocks(client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).text)]
    assert any(frame.get("usage", {}).get("completion_tokens") == 567 for frame in frames)


def test_a_stream_error_closes_the_open_block_and_stops_the_message(client, make_channel):
    """Leaving a content block open makes SDK parsers choke on a half-finished message
    instead of surfacing the error."""
    class Exploding(FakeBackend):
        async def stream(self, req, ch):
            yield {"choices": [{"delta": {"content": "partial"}}]}
            raise RuntimeError("connection reset")

    make_channel("official_claude", backend=Exploding())
    names = [name for name, _ in sse_blocks(client.post("/v1/messages", json=anthropic_body()).text)]
    assert names.count("content_block_start") == names.count("content_block_stop") == 1
    assert names.index("content_block_stop") < names.index("error")
    assert names[-1] == "message_stop"


def test_anthropic_stream_emits_tool_use_blocks(client, make_channel):
    make_channel("official_claude", backend=FakeBackend(stream_chunks=TOOL_STREAM))
    events = sse_blocks(client.post("/v1/messages", json=anthropic_body()).text)

    starts = [p for n, p in events if n == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["text", "tool_use"]
    assert starts[1]["content_block"]["name"] == "read_file"
    # Blocks are opened and closed in order, one at a time.
    assert [s["index"] for s in starts] == [0, 1]
    assert [p["index"] for n, p in events if n == "content_block_stop"] == [0, 1]

    partial = "".join(p["delta"]["partial_json"] for n, p in events
                      if n == "content_block_delta" and p["delta"]["type"] == "input_json_delta")
    assert json.loads(partial) == {"path": "a.py"}

    stop = next(p for n, p in events if n == "message_delta")
    assert stop["delta"]["stop_reason"] == "tool_use"


def test_interleaved_tool_fragments_are_not_dropped(client, make_channel):
    """Anthropic never reopens a closed block, so a provider that interleaves slots cannot be
    streamed slot-by-slot. Buffering keeps every fragment instead of silently losing them."""
    interleaved = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c0", "type": "function", "function": {"name": "f", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 1, "id": "c1", "type": "function", "function": {"name": "g", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "type": "function", "function": {"arguments": '{"a":1}'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 1, "type": "function", "function": {"arguments": '{"b":2}'}}]}}]},
        {"choices": [{"finish_reason": "tool_calls"}]},
    ]
    make_channel("official_claude", backend=FakeBackend(stream_chunks=interleaved))
    events = sse_blocks(client.post("/v1/messages", json=anthropic_body()).text)

    starts = [p for n, p in events if n == "content_block_start"]
    assert [(s["content_block"]["id"], s["content_block"]["name"]) for s in starts] == [("c0", "f"), ("c1", "g")]
    assert [s["index"] for s in starts] == [0, 1]
    assert [p["index"] for n, p in events if n == "content_block_stop"] == [0, 1]

    by_index = {}
    for name, payload in events:
        if name == "content_block_delta" and payload["delta"]["type"] == "input_json_delta":
            by_index.setdefault(payload["index"], "")
            by_index[payload["index"]] += payload["delta"]["partial_json"]
    assert json.loads(by_index[0]) == {"a": 1}
    assert json.loads(by_index[1]) == {"b": 2}


def test_anthropic_stream_maps_max_tokens_stop_reason(client, make_channel):
    make_channel("official_claude", backend=FakeBackend(
        stream_chunks=[{"choices": [{"delta": {"content": "x"}, "finish_reason": "length"}]}]))
    events = sse_blocks(client.post("/v1/messages", json=anthropic_body()).text)
    stop = next(p for n, p in events if n == "message_delta")
    assert stop["delta"]["stop_reason"] == "max_tokens"


def test_anthropic_stream_error_is_reported_as_an_error_event(client, make_channel):
    """A mid-stream failure cannot be retried, so it must reach the client as an SSE error."""
    class Exploding(FakeBackend):
        async def stream(self, req, ch):
            yield {"choices": [{"delta": {"content": "partial"}}]}
            raise RuntimeError("connection reset")

    make_channel("official_claude", backend=Exploding())
    events = sse_blocks(client.post("/v1/messages", json=anthropic_body()).text)
    error = next(p for n, p in events if n == "error")
    assert "connection reset" in error["error"]["message"]


# ---------------------------------------------------------------- openai SSE

def test_openai_stream_chunks_carry_the_required_envelope(client, make_channel):
    """The OpenAI SDK rejects chunks without id/object/created/model."""
    make_channel("official_openai", backend=FakeBackend(stream_chunks=TEXT_STREAM))
    body = client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).text
    assert body.rstrip().endswith("data: [DONE]")
    frames = [payload for _name, payload in sse_blocks(body)]
    for frame in frames:
        assert frame.get("object") == "chat.completion.chunk"
        assert frame.get("id") and frame.get("created") and frame.get("model")
    text = "".join((f["choices"][0]["delta"].get("content") or "")
                   for f in frames if f.get("choices"))
    assert text == "Hello"


def test_openai_stream_opens_with_the_assistant_role(client, make_channel):
    make_channel("official_openai", backend=FakeBackend(stream_chunks=TEXT_STREAM))
    frames = [p for _n, p in sse_blocks(client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).text)]
    assert frames[0]["choices"][0]["delta"] == {"role": "assistant"}


def test_openai_stream_forwards_tool_call_deltas(client, make_channel):
    make_channel("official_openai", backend=FakeBackend(stream_chunks=TOOL_STREAM))
    frames = [p for _n, p in sse_blocks(client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).text)]
    arguments = "".join(
        call["function"].get("arguments") or ""
        for frame in frames for choice in frame.get("choices", [])
        for call in (choice.get("delta") or {}).get("tool_calls") or [])
    assert json.loads(arguments) == {"path": "a.py"}
    assert any(c.get("finish_reason") == "tool_calls"
               for f in frames for c in f.get("choices", []))


def test_openai_stream_reports_usage(client, make_channel):
    make_channel("official_openai", backend=FakeBackend(stream_chunks=TEXT_STREAM))
    frames = [p for _n, p in sse_blocks(client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).text)]
    assert frames[-1]["usage"]["total_tokens"] == 7


# ---------------------------------------------------------------- stream failover

def test_a_stream_that_fails_before_the_first_byte_fails_over(client, make_channel):
    make_channel("official_claude", name="bad",
                 backend=FakeBackend(stream_error=RuntimeError("handshake failed")))
    good, good_backend = make_channel("web_claude", name="good",
                                      backend=FakeBackend(stream_chunks=TEXT_STREAM))
    good["config"]["priority"] = 3
    events = sse_blocks(client.post("/v1/messages", json=anthropic_body()).text)
    text = "".join(p["delta"]["text"] for n, p in events if n == "content_block_delta")
    assert text == "Hello"
    assert good_backend.calls


def test_stream_failure_releases_the_channel(client, make_channel):
    class Exploding(FakeBackend):
        async def stream(self, req, ch):
            yield {"choices": [{"delta": {"content": "partial"}}]}
            raise RuntimeError("reset")

    make_channel("official_claude", backend=Exploding())
    client.post("/v1/messages", json=anthropic_body())
    assert main.router.stats["ch0"]["in_flight"] == 0
    assert main.router.stats["ch0"]["consec_fail"] == 1


def test_completed_stream_releases_the_channel(client, make_channel):
    make_channel("official_claude", backend=FakeBackend(stream_chunks=TEXT_STREAM))
    client.post("/v1/messages", json=anthropic_body())
    assert main.router.stats["ch0"]["in_flight"] == 0
    assert main.router.stats["ch0"]["success"] == 1
    assert main.router.stats["ch0"]["used_est_tokens"] == 7


# ---------------------------------------------------------------- default replay stream

def test_backends_without_native_streaming_replay_their_answer(client, make_channel):
    class NonStreaming(main.BaseBackend):
        supports_tools = True

        async def generate(self, req, ch):
            return main.make_canonical_response(
                "replayed answer", model=req.model,
                usage={"prompt_tokens": 2, "completion_tokens": 3}), {}

    make_channel("web_claude", backend=NonStreaming())
    events = sse_blocks(client.post("/v1/messages", json=anthropic_body()).text)
    text = "".join(p["delta"]["text"] for n, p in events if n == "content_block_delta")
    assert text == "replayed answer"


# ---------------------------------------------------------------- OpenAI SDK chunk mapping

def test_openai_backend_stream_preserves_an_explicit_index_of_zero():
    """`index or position` misfiled tool call 0's fragments whenever they arrived at a
    later list position, gluing its arguments onto a different call."""
    import asyncio
    from types import SimpleNamespace as NS

    call_a = NS(index=1, id="call_b", function=NS(name="second", arguments='{"b":'))
    call_b = NS(index=0, id=None, function=NS(name=None, arguments='1}'))
    chunk = NS(usage=None, choices=[NS(
        delta=NS(content=None, tool_calls=[call_a, call_b]),
        finish_reason=None,
    )])

    class FakeStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield chunk

    class FakeClient:
        chat = NS(completions=NS())

    async def fake_create(**_kwargs):
        return FakeStream()

    FakeClient.chat.completions.create = fake_create
    backend = main.OfficialOpenAIBackend()

    async def fake_client(_ch):
        return FakeClient()

    backend._client = fake_client

    async def collect():
        req = main.CanonicalRequest(model="gpt-4o", messages=[])
        return [c async for c in backend.stream(req, {"config": {"api_key": "k"}})]

    chunks = asyncio.run(collect())
    indices = [c["index"] for c in chunks[0]["choices"][0]["delta"]["tool_calls"]]
    assert indices == [1, 0]
