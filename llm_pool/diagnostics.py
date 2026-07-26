"""Sanitized in-memory diagnostics: redaction helpers and the recent-event ring buffer."""
from __future__ import annotations

import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from .envtools import int_env

DIAGNOSTIC_EVENT_LIMIT = int_env("DIAGNOSTIC_EVENT_LIMIT", 200, 20)
DIAGNOSTIC_EVENTS: deque = deque(maxlen=DIAGNOSTIC_EVENT_LIMIT)
SENSITIVE_DETAIL_MARKERS = (
    "authorization",
    "api_key",
    "apikey",
    "cookie",
    "password",
    "secret",
    "token",
    "x-api-key",
    "x-admin-token",
)

# Compiled once: redaction runs on every diagnostic event and upstream error message.
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[^,\s;}\]]+")
_KEYVALUE_RE = re.compile(
    r"(?i)\b(authorization|x-api-key|x-admin-token|api[_-]?key|token|password|cookie)s?(\s*[:=]\s*)([^,\s;}\]]+)"
)
_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
_GOOGLE_KEY_RE = re.compile(r"AIza[A-Za-z0-9_\-]{10,}")


def redact_sensitive_text(value: Any) -> str:
    text = str(value)
    text = _BEARER_RE.sub("Bearer <redacted>", text)
    text = _KEYVALUE_RE.sub(r"\1\2<redacted>", text)
    text = _OPENAI_KEY_RE.sub("<redacted>", text)
    text = _GOOGLE_KEY_RE.sub("<redacted>", text)
    return text


def sanitize_for_diagnostics(value: Any, key_name: str = "", depth: int = 0) -> Any:
    key_l = key_name.lower()
    if any(marker in key_l for marker in SENSITIVE_DETAIL_MARKERS):
        if key_l == "cookies" and isinstance(value, dict):
            return f"<redacted:{len(value)} cookies>"
        return "<redacted>"
    if depth >= 4:
        return "<truncated>"
    if isinstance(value, dict):
        return {str(k): sanitize_for_diagnostics(v, str(k), depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        items = [sanitize_for_diagnostics(v, key_name, depth + 1) for v in value[:25]]
        if len(value) > 25:
            items.append(f"<truncated:{len(value) - 25}>")
        return items
    if isinstance(value, tuple):
        return [sanitize_for_diagnostics(v, key_name, depth + 1) for v in value[:25]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            text = redact_sensitive_text(value)
            return text if len(text) <= 500 else text[:500] + "...<truncated>"
        return value
    return redact_sensitive_text(value)


def safe_path_for_diagnostics(value: Any) -> str:
    text = str(value)
    try:
        home = str(Path.home())
        if home and text.lower().startswith(home.lower()):
            return "~" + text[len(home):]
    except Exception:
        pass
    return text


def record_diagnostic_event(level: str, message: str, **details: Any) -> None:
    try:
        DIAGNOSTIC_EVENTS.append({
            "ts": round(time.time(), 3),
            "level": level,
            "message": redact_sensitive_text(message)[:160],
            "details": sanitize_for_diagnostics(details),
        })
    except Exception:
        pass
