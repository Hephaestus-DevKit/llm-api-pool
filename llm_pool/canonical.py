"""Unified canonical request/response format and the OpenAI/Anthropic dialect adapters.

Incoming OpenAI or Anthropic requests are normalized to this shape, backends consume and
produce it, and responses are converted back to whichever dialect the caller used.

Canonical message content is either a plain string or a list of blocks. Blocks use the
Anthropic-style vocabulary because it is the more expressive of the two:
  {"type": "text",        "text": str}
  {"type": "tool_use",    "id": str, "name": str, "input": dict}
  {"type": "tool_result", "tool_use_id": str, "content": str, "is_error": bool}
  {"type": "image",       "media_type": str, "data": str}   (base64)
  {"type": "image_url",   "url": str}
Assistant *responses* keep the OpenAI shape ({"content", "tool_calls"}) inside
CanonicalChoice.message, since that maps cleanly onto both dialects on the way out.

Both dialects are lossy in different places, so every conversion is explicit. Malformed
fragments degrade to text instead of raising: a single bad tool declaration must never
fail an otherwise valid agent request.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class CanonicalMessage(BaseModel):
    role: str
    content: Optional[str | list] = None
    name: Optional[str] = None


class CanonicalTool(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class CanonicalRequest(BaseModel):
    model: str
    messages: List[CanonicalMessage]
    system: Optional[str] = None
    tools: Optional[List[CanonicalTool]] = None
    tool_choice: Optional[Dict[str, Any]] = None  # {"mode": auto|none|required|tool, "name": str}
    temperature: Optional[float] = None           # None: leave the provider default alone
    max_tokens: Optional[int] = None
    stream: bool = False
    metadata: dict = Field(default_factory=dict)  # e.g. {"original_format": "anthropic"}


class CanonicalChoice(BaseModel):
    index: int = 0
    message: dict   # {"role": "assistant", "content": str|None, "tool_calls": [...]}
    finish_reason: Optional[str] = "stop"


class CanonicalResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[CanonicalChoice]
    usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


EMPTY_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {}}
# Stands in for a turn that carried no content, so dropping it does not merge its neighbours.
EMPTY_TURN_PLACEHOLDER = "(no content)"

ANTHROPIC_STOP_REASONS = {
    None: "end_turn",
    "": "end_turn",
    "stop": "end_turn",
    "end_turn": "end_turn",
    "length": "max_tokens",
    "max_tokens": "max_tokens",
    "tool_calls": "tool_use",
    "tool_use": "tool_use",
    "stop_sequence": "stop_sequence",
    "content_filter": "refusal",
    "refusal": "refusal",
}

# Anthropic stop reasons expressed in OpenAI's vocabulary (anthropic_stop_reason is the inverse).
OPENAI_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def anthropic_stop_reason(finish_reason: Optional[str]) -> str:
    return ANTHROPIC_STOP_REASONS.get(finish_reason, "end_turn")


def model_to_dict(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def canonical_choices_to_dicts(resp: CanonicalResponse) -> List[dict]:
    return [model_to_dict(choice) for choice in resp.choices]


def stringify_content(content: Any) -> str:
    """Flatten any dialect's content payload to text. Never emits a Python repr."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" or "text" in block:
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if content.get("type") == "text" or "text" in content:
            return str(content.get("text", ""))
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def parse_json_object(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}


def new_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def coerce_temperature(value: Any) -> Optional[float]:
    """None means "the caller did not ask", which is not the same as asking for 0.7.

    Substituting a value would override every provider's own default, and reasoning models
    reject any explicit temperature at all.
    """
    if value is None:
        return None
    try:
        return min(2.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return None


def coerce_positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def estimate_tokens(text: Any) -> int:
    return max(1, len(str(text)) // 4) if text else 0


def _first_int(source: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[int]:
    for key in keys:
        value = source.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def normalize_usage(usage: Any, fallback_completion: Any = "") -> Dict[str, int]:
    """Prefer the provider's real counts; estimate only when the provider reports nothing."""
    prompt = completion = None
    if usage is not None:
        raw = usage if isinstance(usage, dict) else model_to_dict(usage)
        if isinstance(raw, dict):
            prompt = _first_int(raw, ("prompt_tokens", "input_tokens", "prompt_token_count"))
            completion = _first_int(raw, ("completion_tokens", "output_tokens", "candidates_token_count"))
    if prompt is None:
        prompt = 0
    if completion is None:
        completion = estimate_tokens(fallback_completion)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def tools_to_canonical(tools: Any) -> Optional[List[CanonicalTool]]:
    """Accept OpenAI ({"type": "function", "function": {...}}) and Anthropic
    ({"name", "input_schema"}) declarations. Entries without a name are skipped."""
    if not isinstance(tools, list):
        return None
    out: List[CanonicalTool] = []
    for tool in tools:
        if isinstance(tool, CanonicalTool):
            out.append(tool)
            continue
        if not isinstance(tool, dict):
            continue
        spec = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = spec.get("name")
        if not name:
            continue
        schema = spec.get("parameters")
        if not isinstance(schema, dict):
            schema = spec.get("input_schema")
        out.append(CanonicalTool(
            name=str(name),
            description=spec.get("description") or None,
            parameters=schema if isinstance(schema, dict) else {},
        ))
    return out or None


def canonical_tools_to_openai(tools: Optional[List[CanonicalTool]]) -> List[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters or dict(EMPTY_SCHEMA),
            },
        }
        for tool in (tools or [])
    ]


def canonical_tools_to_anthropic(tools: Optional[List[CanonicalTool]]) -> List[dict]:
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.parameters or dict(EMPTY_SCHEMA),
        }
        for tool in (tools or [])
    ]


def tool_choice_to_canonical(choice: Any) -> Optional[Dict[str, Any]]:
    if isinstance(choice, str):
        return {"mode": choice} if choice in {"auto", "none", "required"} else None
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "function":
        name = (choice.get("function") or {}).get("name")
        return {"mode": "tool", "name": name} if name else {"mode": "required"}
    if kind == "tool" and choice.get("name"):
        return {"mode": "tool", "name": choice["name"]}
    if kind == "any":
        return {"mode": "required"}
    if kind in {"auto", "none"}:
        return {"mode": kind}
    return None


def canonical_tool_choice_to_openai(choice: Optional[Dict[str, Any]]) -> Any:
    if not choice:
        return None
    mode = choice.get("mode")
    if mode == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return mode if mode in {"auto", "none", "required"} else None


def canonical_tool_choice_to_anthropic(choice: Optional[Dict[str, Any]]) -> Any:
    """Anthropic has no "none" mode; callers drop the tools array instead."""
    if not choice:
        return None
    mode = choice.get("mode")
    if mode == "tool" and choice.get("name"):
        return {"type": "tool", "name": choice["name"]}
    if mode == "required":
        return {"type": "any"}
    if mode == "auto":
        return {"type": "auto"}
    return None


def image_block_to_canonical(block: dict) -> Optional[dict]:
    kind = block.get("type")
    if kind == "image":
        source = block.get("source") if isinstance(block.get("source"), dict) else {}
        if source.get("data"):
            return {"type": "image", "media_type": source.get("media_type") or "image/png", "data": source["data"]}
        url = source.get("url") or block.get("url")
        return {"type": "image_url", "url": url} if url else None
    if kind == "image_url":
        raw = block.get("image_url")
        url = raw.get("url") if isinstance(raw, dict) else (raw or block.get("url"))
        if not url:
            return None
        if str(url).startswith("data:"):
            head, _, data = str(url).partition(",")
            if data:
                return {"type": "image", "media_type": head[5:].split(";")[0] or "image/png", "data": data}
        return {"type": "image_url", "url": url}
    return None


def tool_result_content_to_canonical(content: Any) -> Any:
    """Keep tool output structured when the caller sent it that way.

    Anthropic tool_result content may be a list of text and image blocks. Flattening it to a
    string loses the image and injects its base64 payload into the prompt as text.
    """
    if not isinstance(content, list):
        return stringify_content(content)
    blocks: List[dict] = []
    for block in content:
        if not isinstance(block, dict):
            blocks.append({"type": "text", "text": str(block)})
            continue
        if block.get("type") in {"image", "image_url"}:
            image = image_block_to_canonical(block)
            if image:
                blocks.append(image)
            continue
        blocks.append({"type": "text", "text": stringify_content(block)})
    if not blocks:
        return ""
    if all(block["type"] == "text" for block in blocks):
        return "\n".join(block["text"] for block in blocks if block["text"])
    return blocks


def canonical_tool_result_to_anthropic(content: Any) -> Any:
    if not isinstance(content, list):
        return content or ""
    out: List[dict] = []
    for block in content:
        if block.get("type") == "image":
            out.append({"type": "image", "source": {
                "type": "base64",
                "media_type": block.get("media_type") or "image/png",
                "data": block.get("data") or "",
            }})
        elif block.get("type") == "image_url":
            out.append({"type": "image", "source": {"type": "url", "url": block.get("url") or ""}})
        else:
            out.append({"type": "text", "text": block.get("text") or ""})
    return out


def _collapse_blocks(role: str, blocks: List[dict], name: Optional[str] = None) -> CanonicalMessage:
    if not blocks:
        return CanonicalMessage(role=role, content="", name=name)
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return CanonicalMessage(role=role, content=blocks[0].get("text", ""), name=name)
    return CanonicalMessage(role=role, content=blocks, name=name)


def openai_message_to_canonical(msg: dict) -> CanonicalMessage:
    role = msg.get("role") or "user"
    content = msg.get("content")
    if role == "tool":
        # OpenAI carries tool output in a dedicated message; canonical keeps it as a user block.
        return CanonicalMessage(role="user", content=[{
            "type": "tool_result",
            "tool_use_id": msg.get("tool_call_id") or "",
            "content": stringify_content(content),
            "is_error": False,
        }])

    blocks: List[dict] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                blocks.append({"type": "text", "text": str(block)})
                continue
            if block.get("type") in {"image", "image_url"}:
                image = image_block_to_canonical(block)
                if image:
                    blocks.append(image)
                continue
            blocks.append({"type": "text", "text": stringify_content(block)})
    elif isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})

    for call in msg.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        blocks.append({
            "type": "tool_use",
            "id": call.get("id") or new_tool_call_id(),
            "name": function.get("name") or "",
            "input": parse_json_object(function.get("arguments")),
        })

    return _collapse_blocks(role, blocks, msg.get("name"))


def anthropic_message_to_canonical(msg: dict) -> CanonicalMessage:
    role = msg.get("role") or "user"
    content = msg.get("content")
    if not isinstance(content, list):
        return CanonicalMessage(role=role, content=stringify_content(content))

    blocks: List[dict] = []
    for block in content:
        if not isinstance(block, dict):
            blocks.append({"type": "text", "text": str(block)})
            continue
        kind = block.get("type")
        if kind == "text":
            blocks.append({"type": "text", "text": str(block.get("text", ""))})
        elif kind == "tool_use":
            blocks.append({
                "type": "tool_use",
                "id": block.get("id") or new_tool_call_id(),
                "name": block.get("name") or "",
                "input": parse_json_object(block.get("input")),
            })
        elif kind == "tool_result":
            blocks.append({
                "type": "tool_result",
                "tool_use_id": block.get("tool_use_id") or "",
                # Kept structured when it is: a computer-use or screenshot tool returns image
                # blocks here, and flattening them would inline a base64 blob as prompt text.
                "content": tool_result_content_to_canonical(block.get("content")),
                "is_error": bool(block.get("is_error")),
            })
        elif kind in {"image", "image_url"}:
            image = image_block_to_canonical(block)
            if image:
                blocks.append(image)
        elif kind in {"thinking", "redacted_thinking"}:
            # Replayed reasoning from a previous Anthropic turn. No other provider accepts
            # it, and stringifying it would inject the raw JSON into the prompt as prose.
            continue
        else:
            blocks.append({"type": "text", "text": stringify_content(block)})

    return _collapse_blocks(role, blocks)


def openai_to_canonical(body: dict) -> CanonicalRequest:
    messages: List[CanonicalMessage] = []
    system_parts: List[str] = []
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            text = stringify_content(msg.get("content"))
            if text:
                system_parts.append(text)
            continue
        messages.append(openai_message_to_canonical(msg))
    return CanonicalRequest(
        model=body.get("model") or "auto",
        messages=messages,
        system="\n\n".join(system_parts) or None,
        tools=tools_to_canonical(body.get("tools")),
        tool_choice=tool_choice_to_canonical(body.get("tool_choice")),
        temperature=coerce_temperature(body.get("temperature")),
        # Newer OpenAI clients send max_completion_tokens instead of max_tokens.
        max_tokens=coerce_positive_int(body.get("max_tokens")) or coerce_positive_int(body.get("max_completion_tokens")),
        stream=bool(body.get("stream")),
        metadata={"original_format": "openai"},
    )


def anthropic_to_canonical(body: dict) -> CanonicalRequest:
    system = body.get("system")
    if isinstance(system, list):
        system = "\n\n".join(part for part in (stringify_content(b) for b in system) if part)
    elif system is not None and not isinstance(system, str):
        system = stringify_content(system)
    messages = [anthropic_message_to_canonical(m) for m in (body.get("messages") or []) if isinstance(m, dict)]
    return CanonicalRequest(
        model=body.get("model") or "auto",
        messages=messages,
        system=system or None,
        tools=tools_to_canonical(body.get("tools")),
        tool_choice=tool_choice_to_canonical(body.get("tool_choice")),
        temperature=coerce_temperature(body.get("temperature")),
        max_tokens=coerce_positive_int(body.get("max_tokens")),
        stream=bool(body.get("stream")),
        metadata={"original_format": "anthropic"},
    )


def canonical_messages_to_openai(req: CanonicalRequest) -> List[dict]:
    out: List[dict] = []
    if req.system:
        out.append({"role": "system", "content": req.system})
    for msg in req.messages:
        if not isinstance(msg.content, list):
            entry: Dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
            if msg.name:
                entry["name"] = msg.name
            out.append(entry)
            continue

        parts: List[dict] = []
        tool_calls: List[dict] = []
        for block in msg.content:
            kind = block.get("type")
            if kind == "tool_result":
                # Tool output must precede the turn that reacts to it. OpenAI's tool message
                # is text-only, so structured output is flattened here and only here.
                out.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") or "",
                    "content": stringify_content(block.get("content")),
                })
            elif kind == "tool_use":
                tool_calls.append({
                    "id": block.get("id") or new_tool_call_id(),
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                })
            elif kind == "image":
                data_url = f"data:{block.get('media_type') or 'image/png'};base64,{block.get('data') or ''}"
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
            elif kind == "image_url":
                parts.append({"type": "image_url", "image_url": {"url": block.get("url") or ""}})
            else:
                parts.append({"type": "text", "text": block.get("text") or ""})

        if not parts and not tool_calls:
            continue
        if not parts:
            content: Any = None
        elif all(part["type"] == "text" for part in parts):
            content = "\n".join(part["text"] for part in parts if part["text"])
        else:
            content = parts
        entry = {"role": msg.role, "content": content}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        if msg.name:
            entry["name"] = msg.name
        out.append(entry)
    return out


def _as_anthropic_blocks(content: Any) -> List[dict]:
    if isinstance(content, list):
        return list(content)
    return [{"type": "text", "text": str(content)}] if content else []


def canonical_messages_to_anthropic(req: CanonicalRequest) -> List[dict]:
    converted: List[dict] = []
    for msg in req.messages:
        role = "assistant" if msg.role == "assistant" else "user"
        if not isinstance(msg.content, list):
            converted.append({"role": role, "content": msg.content or ""})
            continue
        blocks: List[dict] = []
        for block in msg.content:
            kind = block.get("type")
            if kind == "tool_use":
                blocks.append({
                    "type": "tool_use",
                    "id": block.get("id") or new_tool_call_id(),
                    "name": block.get("name") or "",
                    "input": block.get("input") or {},
                })
            elif kind == "tool_result":
                entry: Dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.get("tool_use_id") or "",
                    "content": canonical_tool_result_to_anthropic(block.get("content")),
                }
                if block.get("is_error"):
                    entry["is_error"] = True
                blocks.append(entry)
            elif kind == "image":
                blocks.append({"type": "image", "source": {
                    "type": "base64",
                    "media_type": block.get("media_type") or "image/png",
                    "data": block.get("data") or "",
                }})
            elif kind == "image_url":
                blocks.append({"type": "image", "source": {"type": "url", "url": block.get("url") or ""}})
            elif block.get("text"):
                blocks.append({"type": "text", "text": block["text"]})
        converted.append({"role": role, "content": blocks})

    # Anthropic rejects empty content and wants alternating roles, but an OpenAI history with
    # several tool results in a row legitimately produces consecutive user turns.
    #
    # An empty turn is dropped, and dropping it must not fuse its neighbours: an empty
    # assistant reply between two user turns is a real boundary in the conversation, and
    # merging across it would present two separate questions as one.
    merged: List[dict] = []
    previous_kept = False
    for entry in converted:
        if not entry["content"]:
            previous_kept = False
            continue
        if previous_kept and merged and merged[-1]["role"] == entry["role"]:
            merged[-1]["content"] = _as_anthropic_blocks(merged[-1]["content"]) + _as_anthropic_blocks(entry["content"])
        else:
            merged.append(entry)
        previous_kept = True

    # A gap can still leave two same-role turns adjacent, which the API refuses. Separate
    # them with an explicit marker rather than silently gluing the turns together.
    spaced: List[dict] = []
    for entry in merged:
        if spaced and spaced[-1]["role"] == entry["role"]:
            spaced.append({
                "role": "assistant" if entry["role"] == "user" else "user",
                "content": [{"type": "text", "text": EMPTY_TURN_PLACEHOLDER}],
            })
        spaced.append(entry)
    return spaced


def canonical_to_openai(resp: CanonicalResponse, stream: bool = False) -> dict:
    choices = canonical_choices_to_dicts(resp)
    if stream:
        return {"model": resp.model, "choices": choices}
    return {
        "id": resp.id,
        "object": "chat.completion",
        "created": resp.created,
        "model": resp.model,
        "choices": choices,
        "usage": resp.usage,
    }


def canonical_to_anthropic(resp: CanonicalResponse, stream: bool = False) -> dict:
    content: List[dict] = []
    choices = canonical_choices_to_dicts(resp)
    for choice in choices:
        message = choice.get("message") or {}
        text = message.get("content")
        if text:
            content.append({"type": "text", "text": text})
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            content.append({
                "type": "tool_use",
                "id": call.get("id") or new_tool_call_id(),
                "name": function.get("name") or "",
                "input": parse_json_object(function.get("arguments")),
            })
    if not content:
        content.append({"type": "text", "text": ""})
    if stream:
        return {"type": "message", "content": content}
    return {
        "id": resp.id,
        "type": "message",
        "role": "assistant",
        "model": resp.model,
        "content": content,
        "stop_reason": anthropic_stop_reason(choices[0].get("finish_reason") if choices else "stop"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": resp.usage.get("prompt_tokens", 0),
            "output_tokens": resp.usage.get("completion_tokens", 0),
        },
    }


def make_canonical_response(
    text: str,
    model: str = "pooled",
    tool_calls: Optional[List[dict]] = None,
    usage: Any = None,
    finish_reason: Optional[str] = None,
) -> CanonicalResponse:
    message: Dict[str, Any] = {"role": "assistant", "content": text or (None if tool_calls else "")}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if finish_reason is None:
        finish_reason = "tool_calls" if tool_calls else "stop"
    return CanonicalResponse(
        id=f"pool-{uuid.uuid4()}",
        created=int(time.time()),
        model=model,
        choices=[CanonicalChoice(index=0, message=message, finish_reason=finish_reason)],
        usage=normalize_usage(usage, fallback_completion=text),
    )


def canonical_to_text_prompt(req: CanonicalRequest) -> str:
    """Flatten a conversation for text-only channels (web sessions, Gemini)."""
    lines = []
    if req.system:
        lines.append(f"System: {req.system}")
    for msg in req.messages:
        text = stringify_content(msg.content)
        if not text:
            continue
        label = {"assistant": "Assistant", "system": "System"}.get(msg.role, "User")
        lines.append(f"{label}: {text}")
    return "\n\n".join(lines)


def latest_user_prompt(req: CanonicalRequest) -> str:
    """Web channels drive a chat box, so they only get the newest user turn."""
    for msg in reversed(req.messages):
        if msg.role == "user":
            text = stringify_content(msg.content)
            if text:
                return text
    if req.messages:
        return stringify_content(req.messages[-1].content)
    return req.system or ""
