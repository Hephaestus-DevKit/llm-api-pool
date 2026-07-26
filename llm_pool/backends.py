"""Provider backends, unified via the canonical format.

Every backend implements generate() -> (CanonicalResponse, response_headers) and stream().

Streaming uses one internal chunk vocabulary, modelled on OpenAI's stream shape because
both dialects can be rendered from it:
  {"choices": [{"delta": {"content": "text fragment"}}]}
  {"choices": [{"delta": {"tool_calls": [{"index", "id", "function": {"name", "arguments"}}]}}]}
  {"choices": [{"finish_reason": "tool_calls"}]}
  {"usage": {...}}                      optional, terminal accounting
"arguments" fragments are concatenated by index, exactly like OpenAI's own protocol.
"""
from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from . import settings, webdrive
from .canonical import (
    OPENAI_FINISH_REASONS,
    CanonicalRequest,
    CanonicalResponse,
    canonical_messages_to_anthropic,
    canonical_messages_to_openai,
    canonical_to_text_prompt,
    canonical_tool_choice_to_anthropic,
    canonical_tool_choice_to_openai,
    canonical_tools_to_anthropic,
    canonical_tools_to_openai,
    estimate_tokens,
    latest_user_prompt,
    make_canonical_response,
    model_to_dict,
    new_tool_call_id,
    normalize_usage,
)

REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")


def is_reasoning_model(name: Optional[str]) -> bool:
    """OpenAI reasoning models reject `temperature` and `max_tokens` outright."""
    return bool(name) and name.lower().lstrip().startswith(REASONING_MODEL_PREFIXES)


async def aclose_client(client: Any) -> None:
    for name in ("aclose", "close"):
        closer = getattr(client, name, None)
        if closer is None:
            continue
        try:
            result = closer()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass
        return


class ProviderClientCache:
    """Bounded LRU cache of provider SDK clients, keyed by API key.

    Each client owns an HTTP connection pool, so caching them without a bound keeps sockets
    and TLS sessions alive for channels that were deleted long ago.
    """

    def __init__(self, max_size: int = 16):
        self.max_size = max(1, max_size)
        self._clients: "OrderedDict[str, Any]" = OrderedDict()

    async def get(self, key: str, factory) -> Any:
        client = self._clients.get(key)
        if client is not None:
            self._clients.move_to_end(key)
            return client
        client = factory()
        self._clients[key] = client
        while len(self._clients) > self.max_size:
            _, stale = self._clients.popitem(last=False)
            await aclose_client(stale)
        return client

    async def aclose(self) -> None:
        clients = list(self._clients.values())
        self._clients.clear()
        for client in clients:
            await aclose_client(client)


class BaseBackend:
    supports_tools = False

    def __init__(self):
        self._clients = ProviderClientCache()

    async def generate(self, req: CanonicalRequest, ch: dict) -> Tuple[CanonicalResponse, dict]:
        raise NotImplementedError

    async def stream(self, req: CanonicalRequest, ch: dict):
        """Replay a non-streaming result as a stream. Backends with real streaming override this."""
        resp, _headers = await self.generate(req, ch)
        choice = model_to_dict(resp.choices[0]) if resp.choices else {}
        message = choice.get("message") or {}
        text = message.get("content") or ""
        for start in range(0, len(text), 24):
            yield {"choices": [{"delta": {"content": text[start:start + 24]}}]}
            await asyncio.sleep(0)  # yield to the loop without inventing latency
        for index, call in enumerate(message.get("tool_calls") or []):
            yield {"choices": [{"delta": {"tool_calls": [{**call, "index": index}]}}]}
        if choice.get("finish_reason"):
            yield {"choices": [{"finish_reason": choice["finish_reason"]}]}
        yield {"usage": resp.usage}

    async def aclose(self) -> None:
        await self._clients.aclose()


class OfficialOpenAIBackend(BaseBackend):
    supports_tools = True

    async def _client(self, ch: dict):
        key = ch["config"]["api_key"]

        def factory():
            from openai import AsyncOpenAI
            return AsyncOpenAI(api_key=key)

        return await self._clients.get(key, factory)

    def _kwargs(self, req: CanonicalRequest, stream: bool) -> Dict[str, Any]:
        model = req.model or settings.DEFAULT_MODELS["official_openai"]
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": canonical_messages_to_openai(req),
            "stream": stream,
        }
        # Reasoning models reject an explicit temperature and require max_completion_tokens.
        reasoning = is_reasoning_model(model)
        if req.temperature is not None and not reasoning:
            kwargs["temperature"] = req.temperature
        if req.max_tokens:
            kwargs["max_completion_tokens" if reasoning else "max_tokens"] = req.max_tokens
        tools = canonical_tools_to_openai(req.tools)
        if tools:
            kwargs["tools"] = tools
            choice = canonical_tool_choice_to_openai(req.tool_choice)
            if choice is not None:
                kwargs["tool_choice"] = choice
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    async def generate(self, req: CanonicalRequest, ch: dict) -> Tuple[CanonicalResponse, dict]:
        client = await self._client(ch)
        raw = await client.chat.completions.with_raw_response.create(**self._kwargs(req, stream=False))
        resp = raw.parse()
        choice = resp.choices[0] if resp.choices else None
        message = getattr(choice, "message", None)
        tool_calls = [
            {
                "id": getattr(call, "id", None) or new_tool_call_id(),
                "type": "function",
                "function": {
                    "name": getattr(call.function, "name", "") or "",
                    "arguments": getattr(call.function, "arguments", "") or "{}",
                },
            }
            for call in (getattr(message, "tool_calls", None) or [])
        ]
        canon = make_canonical_response(
            getattr(message, "content", None) or "",
            model=getattr(resp, "model", None) or req.model,
            tool_calls=tool_calls or None,
            usage=getattr(resp, "usage", None),
            finish_reason=getattr(choice, "finish_reason", None),
        )
        return canon, dict(raw.headers)

    async def stream(self, req: CanonicalRequest, ch: dict):
        client = await self._client(ch)
        stream = await client.chat.completions.create(**self._kwargs(req, stream=True))
        try:
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                choices = getattr(chunk, "choices", None)
                if not choices:
                    if usage is not None:
                        yield {"usage": normalize_usage(usage)}
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                payload: Dict[str, Any] = {}
                if delta is not None:
                    if getattr(delta, "content", None):
                        payload["content"] = delta.content
                    calls = getattr(delta, "tool_calls", None) or []
                    if calls:
                        payload["tool_calls"] = [
                            {
                                # `or position` would misfile an explicit index of 0 whenever
                                # it arrives at a later list position, gluing its argument
                                # fragments onto a different tool call.
                                "index": position if getattr(call, "index", None) is None else call.index,
                                "id": getattr(call, "id", None),
                                "type": "function",
                                "function": {
                                    "name": getattr(getattr(call, "function", None), "name", None),
                                    "arguments": getattr(getattr(call, "function", None), "arguments", None) or "",
                                },
                            }
                            for position, call in enumerate(calls)
                        ]
                entry: Dict[str, Any] = {}
                if payload:
                    entry["delta"] = payload
                if getattr(choice, "finish_reason", None):
                    entry["finish_reason"] = choice.finish_reason
                if entry:
                    yield {"choices": [entry]}
                if usage is not None:
                    yield {"usage": normalize_usage(usage)}
        finally:
            # An abandoned stream holds its HTTP response open until the GC gets to it.
            await aclose_client(stream)


class OfficialAnthropicBackend(BaseBackend):
    supports_tools = True

    async def _client(self, ch: dict):
        key = ch["config"]["api_key"]

        def factory():
            import anthropic
            return anthropic.AsyncAnthropic(api_key=key)

        return await self._clients.get(key, factory)

    def _kwargs(self, req: CanonicalRequest) -> Dict[str, Any]:
        messages = canonical_messages_to_anthropic(req)
        system = req.system
        if not messages:
            # The API requires at least one message, but a system-only prompt is legitimate
            # in the OpenAI dialect, so carry the system text as the opening turn.
            if not system:
                raise ValueError("request has no prompt content")
            messages = [{"role": "user", "content": system}]
            system = None

        kwargs: Dict[str, Any] = {
            "model": req.model or settings.DEFAULT_MODELS["official_claude"],
            "max_tokens": req.max_tokens or settings.ANTHROPIC_DEFAULT_MAX_TOKENS,
            "messages": messages,
        }
        if req.temperature is not None:
            # Anthropic's range is 0..1; the canonical value carries OpenAI's 0..2.
            kwargs["temperature"] = min(1.0, req.temperature)
        # The SDK treats an omitted argument and an explicit None differently: None is
        # serialized as a literal JSON null, which the API rejects.
        if system:
            kwargs["system"] = system
        if (req.tool_choice or {}).get("mode") != "none":
            tools = canonical_tools_to_anthropic(req.tools)
            if tools:
                kwargs["tools"] = tools
                choice = canonical_tool_choice_to_anthropic(req.tool_choice)
                if choice is not None:
                    kwargs["tool_choice"] = choice
        return kwargs

    async def generate(self, req: CanonicalRequest, ch: dict) -> Tuple[CanonicalResponse, dict]:
        client = await self._client(ch)
        kwargs = self._kwargs(req)
        raw_api = getattr(client.messages, "with_raw_response", None)
        if raw_api is not None:
            raw = await raw_api.create(**kwargs)
            msg, response_headers = raw.parse(), dict(raw.headers)
        else:
            msg, response_headers = await client.messages.create(**kwargs), {}

        text_parts: list = []
        tool_calls: list = []
        for block in (getattr(msg, "content", None) or []):
            kind = getattr(block, "type", None)
            if kind == "tool_use":
                tool_calls.append({
                    "id": getattr(block, "id", None) or new_tool_call_id(),
                    "type": "function",
                    "function": {
                        "name": getattr(block, "name", "") or "",
                        "arguments": json.dumps(getattr(block, "input", None) or {}, ensure_ascii=False),
                    },
                })
            else:
                text_parts.append(getattr(block, "text", "") or "")

        canon = make_canonical_response(
            "".join(text_parts),
            model=getattr(msg, "model", None) or req.model,
            tool_calls=tool_calls or None,
            usage=getattr(msg, "usage", None),
            finish_reason=OPENAI_FINISH_REASONS.get(getattr(msg, "stop_reason", None)),
        )
        return canon, response_headers

    async def stream(self, req: CanonicalRequest, ch: dict):
        client = await self._client(ch)
        stream = await client.messages.create(**self._kwargs(req), stream=True)
        # Anthropic numbers every content block; we only number the tool_use ones.
        tool_slots: Dict[int, int] = {}
        try:
            async for event in stream:
                kind = getattr(event, "type", None)
                if kind == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    slot = len(tool_slots)
                    tool_slots[getattr(event, "index", slot)] = slot
                    yield {"choices": [{"delta": {"tool_calls": [{
                        "index": slot,
                        "id": getattr(block, "id", None) or new_tool_call_id(),
                        "type": "function",
                        "function": {"name": getattr(block, "name", "") or "", "arguments": ""},
                    }]}}]}
                elif kind == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_kind = getattr(delta, "type", None)
                    if delta_kind == "text_delta" and getattr(delta, "text", None):
                        yield {"choices": [{"delta": {"content": delta.text}}]}
                    elif delta_kind == "input_json_delta":
                        partial = getattr(delta, "partial_json", "") or ""
                        if partial:
                            yield {"choices": [{"delta": {"tool_calls": [{
                                "index": tool_slots.get(getattr(event, "index", 0), 0),
                                "type": "function",
                                "function": {"arguments": partial},
                            }]}}]}
                elif kind == "message_delta":
                    # stop_reason and usage arrive together in this one event.
                    out: Dict[str, Any] = {}
                    stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None)
                    if stop_reason:
                        out["choices"] = [{"finish_reason": OPENAI_FINISH_REASONS.get(stop_reason, "stop")}]
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        out["usage"] = normalize_usage(usage)
                    if out:
                        yield out
        finally:
            await aclose_client(stream)


# The native ARM64 package ships without google-genai (its google-auth -> cryptography
# chain has no win_arm64 wheels), so the import failure needs a message that says what
# to do about it rather than a bare ModuleNotFoundError.
GENAI_MISSING_HINT = (
    "google-genai is not installed in this build (the native ARM64 package excludes it "
    "because its cryptography dependency ships no ARM64 wheels). Use a web_gemini channel, "
    "or the x64 build for official Gemini."
)


class OfficialGeminiBackend(BaseBackend):
    # Gemini's schema dialect is a strict subset of JSON Schema, so forwarding arbitrary agent
    # tool definitions would fail the whole request. The router prefers tool-capable channels
    # instead; see SmartRouter.compute_score.
    supports_tools = False

    async def _client(self, ch: dict):
        key = ch["config"]["api_key"]

        def factory():
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(GENAI_MISSING_HINT) from exc
            return genai.Client(api_key=key)

        return await self._clients.get(key, factory)

    def _config(self, req: CanonicalRequest):
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(GENAI_MISSING_HINT) from exc
        kwargs: Dict[str, Any] = {"temperature": req.temperature}
        if req.system:
            kwargs["system_instruction"] = req.system
        if req.max_tokens:
            kwargs["max_output_tokens"] = req.max_tokens
        return types.GenerateContentConfig(**kwargs)

    async def generate(self, req: CanonicalRequest, ch: dict) -> Tuple[CanonicalResponse, dict]:
        client = await self._client(ch)
        model = req.model or settings.DEFAULT_MODELS["official_gemini"]
        resp = await client.aio.models.generate_content(
            model=model,
            contents=canonical_to_text_prompt(req),
            config=self._config(req),
        )
        canon = make_canonical_response(
            getattr(resp, "text", None) or "",
            model=model,
            usage=getattr(resp, "usage_metadata", None),
        )
        return canon, {}

    async def stream(self, req: CanonicalRequest, ch: dict):
        client = await self._client(ch)
        model = req.model or settings.DEFAULT_MODELS["official_gemini"]
        stream = await client.aio.models.generate_content_stream(
            model=model,
            contents=canonical_to_text_prompt(req),
            config=self._config(req),
        )
        usage: Any = None
        async for chunk in stream:
            if getattr(chunk, "text", None):
                yield {"choices": [{"delta": {"content": chunk.text}}]}
            if getattr(chunk, "usage_metadata", None) is not None:
                usage = chunk.usage_metadata
        if usage is not None:
            yield {"usage": normalize_usage(usage)}


class WebBrowserBackend(BaseBackend):
    """Shared by every web_* channel: drives an already-authenticated browser session."""
    supports_tools = False

    async def generate(self, req: CanonicalRequest, ch: dict) -> Tuple[CanonicalResponse, dict]:
        prompt = latest_user_prompt(req)
        text = await webdrive.drive_web_chat(ch, prompt)
        canon = make_canonical_response(
            text,
            model=ch.get("name", "web"),
            usage={"prompt_tokens": estimate_tokens(prompt), "completion_tokens": estimate_tokens(text)},
        )
        return canon, {}  # a browser session exposes no rate-limit headers

    async def stream(self, req: CanonicalRequest, ch: dict):
        prompt = latest_user_prompt(req)
        produced = 0
        async for chunk in webdrive.drive_web_chat_stream(ch, prompt):
            entry = (chunk.get("choices") or [{}])[0]
            produced += len((entry.get("delta") or {}).get("content") or "")
            yield chunk
        yield {"usage": {"prompt_tokens": estimate_tokens(prompt), "completion_tokens": produced // 4}}


BACKENDS: Dict[str, BaseBackend] = {
    "official_openai": OfficialOpenAIBackend(),
    "official_claude": OfficialAnthropicBackend(),
    "official_gemini": OfficialGeminiBackend(),
    "web_gemini": WebBrowserBackend(),
    "web_claude": WebBrowserBackend(),
    "web_chatgpt": WebBrowserBackend(),
    "web_codex": WebBrowserBackend(),  # separate quota bucket from plain web_chatgpt
}


async def aclose_all() -> None:
    """Close every backend's client cache. Used by the app's lifespan."""
    for backend in set(BACKENDS.values()):
        try:
            await backend.aclose()
        except Exception:
            pass
