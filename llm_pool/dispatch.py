"""Turning a canonical request into an answer: failover, stream leases, SSE encoders.

A pool exists to survive a bad account, so both endpoints retry on the next-best channel.
Streams are retried only while nothing has been sent to the client yet: once the first
chunk is out, switching channels would splice two different answers together.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from . import backends, routing, settings
from .canonical import (
    CanonicalRequest,
    CanonicalResponse,
    anthropic_stop_reason,
    new_tool_call_id,
    normalize_usage,
)
from .diagnostics import record_diagnostic_event, redact_sensitive_text

SSE_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def routing_failure(canon: CanonicalRequest, endpoint: str, tried: set, errors: List[str]) -> HTTPException:
    if not tried:
        record_diagnostic_event("warn", "no_suitable_channel", endpoint=endpoint, model=canon.model)
        hint = "" if settings.CROSS_PROVIDER_FALLBACK else " Set CROSS_PROVIDER_FALLBACK=1 to allow any provider."
        return HTTPException(503, f"No channel in the pool can serve model '{canon.model}'.{hint}")
    detail = "; ".join(errors) or "unknown upstream error"
    record_diagnostic_event("error", "all_channels_failed", endpoint=endpoint, model=canon.model, attempts=len(tried))
    if any("WEB_RATE_LIMIT" in message for message in errors):
        return HTTPException(429, f"Every candidate channel is rate limited: {detail[:300]}")
    return HTTPException(502, f"All {len(tried)} candidate channel(s) failed: {detail[:300]}")


def scoped_request(canon: CanonicalRequest, ch: dict) -> CanonicalRequest:
    """A per-attempt copy carrying this channel's model name, leaving `canon` pristine so a
    later attempt still resolves aliases against what the caller originally asked for."""
    return canon.model_copy(update={"model": routing.router.resolve_model(ch, canon.model)})


async def aclose_agen(agen: Any) -> None:
    closer = getattr(agen, "aclose", None)
    if closer is None:
        return
    try:
        await closer()
    except Exception:
        pass


async def generate_with_failover(canon: CanonicalRequest, endpoint: str) -> Tuple[CanonicalResponse, Dict[str, Any]]:
    """Returns (response, the channel that produced it)."""
    router = routing.router
    tried: set = set()
    errors: List[str] = []
    for attempt in range(settings.MAX_ROUTE_ATTEMPTS):
        ch = router.select(canon, exclude=tried)
        if ch is None:
            break
        tried.add(ch["id"])
        backend = backends.BACKENDS.get(ch["type"])
        if backend is None:
            errors.append(f"{ch['name']}: no backend for {ch['type']}")
            continue
        started = time.time()
        sem = await router.acquire(ch)
        try:
            resp, headers = await backend.generate(scoped_request(canon, ch), ch)
        except Exception as exc:
            router.record_result(ch, False, time.time() - started, 0, None)
            message = redact_sensitive_text(str(exc))[:160]
            errors.append(f"{ch['name']}: {message}")
            record_diagnostic_event("error", "upstream_failed", endpoint=endpoint, channel_id=ch["id"],
                                    type=ch["type"], name=ch["name"], attempt=attempt + 1, error=str(exc))
            continue
        finally:
            router.release(ch, sem)
        router.record_result(ch, True, time.time() - started, resp.usage.get("total_tokens", 0), headers)
        return resp, ch
    raise routing_failure(canon, endpoint, tried, errors)


class StreamLease:
    """A channel held open for the duration of one streamed response."""

    def __init__(self, channel: dict, sem: asyncio.Semaphore, model: str, agen: Any, first: Any, started: float):
        self.channel = channel
        self.sem = sem
        self.model = model
        self.agen = agen
        self.first = first
        self.started = started
        self._closed = False

    async def chunks(self):
        if self.first is not None:
            yield self.first
        async for chunk in self.agen:
            yield chunk

    async def finish(self, success: bool, usage: Optional[dict], produced_chars: int) -> None:
        if self._closed:
            return
        self._closed = True
        await aclose_agen(self.agen)
        tokens = 0
        if usage:
            tokens = usage.get("total_tokens") or usage.get("completion_tokens") or 0
        routing.router.record_result(self.channel, success, time.time() - self.started, tokens or produced_chars // 4, None)
        routing.router.release(self.channel, self.sem)

    async def abandon(self) -> None:
        """Safety net for a response whose body was never consumed. No-op once finished."""
        if self._closed:
            return
        record_diagnostic_event("warn", "stream_abandoned", channel_id=self.channel.get("id"),
                                name=self.channel.get("name"))
        await self.finish(False, None, 0)


async def open_stream_with_failover(canon: CanonicalRequest, endpoint: str) -> StreamLease:
    router = routing.router
    tried: set = set()
    errors: List[str] = []
    for attempt in range(settings.MAX_ROUTE_ATTEMPTS):
        ch = router.select(canon, exclude=tried)
        if ch is None:
            break
        tried.add(ch["id"])
        backend = backends.BACKENDS.get(ch["type"])
        if backend is None:
            errors.append(f"{ch['name']}: no backend for {ch['type']}")
            continue
        scoped = scoped_request(canon, ch)
        started = time.time()
        sem = await router.acquire(ch)
        agen = backend.stream(scoped, ch)
        try:
            first = await agen.__anext__()
        except StopAsyncIteration:
            first = None
        except BaseException as exc:
            # BaseException, not Exception: a cancellation here would otherwise walk out of
            # the function still holding the permit, and nothing would ever give it back.
            await aclose_agen(agen)
            router.record_result(ch, False, time.time() - started, 0, None)
            router.release(ch, sem)
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
            errors.append(f"{ch['name']}: {redact_sensitive_text(str(exc))[:160]}")
            record_diagnostic_event("error", "upstream_stream_failed", endpoint=endpoint, channel_id=ch["id"],
                                    type=ch["type"], name=ch["name"], attempt=attempt + 1, error=str(exc))
            continue
        return StreamLease(ch, sem, scoped.model, agen, first, started)
    raise routing_failure(canon, endpoint, tried, errors)


def stream_with_lease(lease: StreamLease, encoder, endpoint: str) -> StreamingResponse:
    """Wire a lease to an SSE encoder.

    The release is also registered as a background task: Starlette runs it once the response
    finishes *or* is abandoned, which covers the case where a client disconnects before the
    body generator is ever started. Without it the generator's `finally` never runs and the
    channel loses a permit for good. StreamLease.finish is idempotent, so whichever path
    fires first wins.
    """
    return StreamingResponse(
        encoder(lease, endpoint),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
        background=BackgroundTask(lease.abandon),
    )


def sse_event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_data(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def openai_stream_response(lease: StreamLease, endpoint: str):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    produced = 0
    usage: Optional[dict] = None
    success = True

    def frame(choices: List[dict], extra: Optional[dict] = None) -> dict:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": lease.model,
            "choices": choices,
        }
        if extra:
            payload.update(extra)
        return payload

    try:
        yield sse_data(frame([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]))
        async for chunk in lease.chunks():
            # Anthropic sends stop_reason and usage in the same message_delta, so usage must
            # be harvested even when the chunk also carries choices.
            if chunk.get("usage") is not None:
                usage = normalize_usage(chunk["usage"])
                if not chunk.get("choices"):
                    continue
            entry = (chunk.get("choices") or [{}])[0]
            delta = entry.get("delta") or {}
            produced += len(delta.get("content") or "")
            yield sse_data(frame([{
                "index": 0,
                "delta": delta,
                "finish_reason": entry.get("finish_reason"),
            }]))
        if usage:
            yield sse_data(frame([], {"usage": usage}))
        yield "data: [DONE]\n\n"
    except Exception as exc:
        success = False
        record_diagnostic_event("error", "upstream_stream_failed", endpoint=endpoint,
                                channel_id=lease.channel.get("id"), name=lease.channel.get("name"), error=str(exc))
        yield sse_data({"error": {"message": redact_sensitive_text(str(exc))[:180], "type": "upstream_error"}})
        yield "data: [DONE]\n\n"
    finally:
        await lease.finish(success, usage, produced)


class AnthropicStreamEncoder:
    """Renders the pool's internal chunk stream as Anthropic SSE.

    Anthropic requires content blocks to be opened and closed in order, so exactly one block
    is kept open at a time and the encoder switches blocks when the backend moves from text
    to a tool call, or between tool calls.
    """

    def __init__(self, message_id: str, model: str):
        self.message_id = message_id
        self.model = model
        self.next_index = 0
        self.open_index = 0
        self.open_kind: Optional[str] = None
        self.open_slot: Optional[int] = None
        self.text_chars = 0
        self.finish_reason: Optional[str] = None
        self.pending_tools: Dict[int, dict] = {}  # slot -> accumulating tool call

    def start(self) -> str:
        return sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

    def close_open_block(self) -> List[str]:
        if self.open_kind is None:
            return []
        self.open_kind = None
        self.open_slot = None
        return [sse_event("content_block_stop", {"type": "content_block_stop", "index": self.open_index})]

    def _open_block(self, kind: str, content_block: dict, slot: Optional[int] = None) -> List[str]:
        events = self.close_open_block()
        self.open_index = self.next_index
        self.next_index += 1
        self.open_kind = kind
        self.open_slot = slot
        events.append(sse_event("content_block_start", {
            "type": "content_block_start",
            "index": self.open_index,
            "content_block": content_block,
        }))
        return events

    def _input_delta(self, partial: str) -> str:
        return sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": self.open_index,
            "delta": {"type": "input_json_delta", "partial_json": partial},
        })

    def _feed_tool_call(self, call: dict) -> None:
        """Accumulate a tool-call fragment. Nothing is emitted until finish().

        Anthropic allows exactly one open content block and never lets a closed one reopen,
        so a provider that interleaves fragments across slots cannot be streamed faithfully.
        Buffering by slot is order-independent and always produces a valid block sequence;
        the only cost is that tool arguments arrive in one delta instead of several, which
        clients handle identically. Text still streams incrementally.
        """
        slot = call.get("index") or 0
        function = call.get("function") or {}
        pending = self.pending_tools.get(slot)
        if pending is None:
            pending = self.pending_tools[slot] = {"id": "", "name": "", "arguments": ""}
        if call.get("id"):
            pending["id"] = call["id"]
        if function.get("name"):
            pending["name"] = function["name"]
        pending["arguments"] += function.get("arguments") or ""

    def feed(self, chunk: dict) -> List[str]:
        entry = (chunk.get("choices") or [{}])[0]
        delta = entry.get("delta") or {}
        events: List[str] = []
        text = delta.get("content") or ""
        if text:
            if self.open_kind != "text":
                events += self._open_block("text", {"type": "text", "text": ""})
            self.text_chars += len(text)
            events.append(sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": self.open_index,
                "delta": {"type": "text_delta", "text": text},
            }))
        for call in delta.get("tool_calls") or []:
            self._feed_tool_call(call)
        if entry.get("finish_reason"):
            self.finish_reason = entry["finish_reason"]
        return events

    def finish(self, usage: Optional[dict]) -> List[str]:
        events = self.close_open_block()
        for slot in sorted(self.pending_tools):
            pending = self.pending_tools[slot]
            events += self._open_block("tool", {
                "type": "tool_use",
                "id": pending["id"] or new_tool_call_id(),
                "name": pending["name"],
                "input": {},
            }, slot=slot)
            if pending["arguments"]:
                events.append(self._input_delta(pending["arguments"]))
            events += self.close_open_block()
        self.pending_tools.clear()

        output_tokens = (usage or {}).get("completion_tokens")
        if output_tokens is None:
            output_tokens = self.text_chars // 4
        events.append(sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": anthropic_stop_reason(self.finish_reason), "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }))
        events.append(sse_event("message_stop", {"type": "message_stop"}))
        return events


async def anthropic_stream_response(lease: StreamLease, endpoint: str):
    encoder = AnthropicStreamEncoder(f"msg_{uuid.uuid4().hex}", lease.model)
    usage: Optional[dict] = None
    success = True
    try:
        yield encoder.start()
        async for chunk in lease.chunks():
            # Anthropic sends stop_reason and usage in the same message_delta, so usage must
            # be harvested even when the chunk also carries choices.
            if chunk.get("usage") is not None:
                usage = normalize_usage(chunk["usage"])
                if not chunk.get("choices"):
                    continue
            for event in encoder.feed(chunk):
                yield event
        for event in encoder.finish(usage):
            yield event
    except Exception as exc:
        success = False
        record_diagnostic_event("error", "upstream_stream_failed", endpoint=endpoint,
                                channel_id=lease.channel.get("id"), name=lease.channel.get("name"), error=str(exc))
        # Close whatever block is open first: leaving one dangling makes SDK parsers choke on
        # a half-finished message instead of surfacing the error.
        for event in encoder.close_open_block():
            yield event
        yield sse_event("error", {
            "type": "error",
            "error": {"type": "api_error", "message": redact_sensitive_text(str(exc))[:180]},
        })
        yield sse_event("message_stop", {"type": "message_stop"})
    finally:
        await lease.finish(success, usage, encoder.text_chars)
