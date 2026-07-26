"""Environment-derived configuration and the non-local-bind startup guard.

Values that tests or the CLI rebind at runtime (HOST, tokens, routing flags) must be
read attribute-style (``settings.HOST``) by other modules, never ``from``-imported.
"""
from __future__ import annotations

import os
import secrets

from .envtools import bool_env, int_env, parse_csv_env

PORT = int_env("PORT", 8080)
HOST = os.getenv("HOST", "127.0.0.1")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()
GENERATED_ADMIN_TOKEN = False
if not ADMIN_TOKEN:
    ADMIN_TOKEN = secrets.token_urlsafe(24)
    GENERATED_ADMIN_TOKEN = True
API_TOKEN = os.getenv("API_TOKEN", "")  # Optional protection for /v1 endpoints on remote instances
RATE_LIMIT_PER_MINUTE = int_env("RATE_LIMIT_PER_MINUTE", 120, 0)
DEBUG_ERRORS = bool_env("DEBUG_ERRORS")

# Only meaningful behind a reverse proxy that overwrites the header itself.
TRUST_PROXY_HEADERS = bool_env("TRUST_PROXY_HEADERS", False)
# Allow answering e.g. a claude-* request from an OpenAI channel when no claude channel exists.
CROSS_PROVIDER_FALLBACK = bool_env("CROSS_PROVIDER_FALLBACK", False)
# How many channels a single request may try before giving up.
MAX_ROUTE_ATTEMPTS = int_env("MAX_ROUTE_ATTEMPTS", 3, 1)
# Anthropic requires max_tokens; OpenAI-dialect callers routinely omit it. 4096 is accepted
# by every current model, so it is the safe floor. Raise it if your models allow more.
ANTHROPIC_DEFAULT_MAX_TOKENS = int_env("ANTHROPIC_DEFAULT_MAX_TOKENS", 4096, 1)

VALID_CHANNEL_TYPES = {
    "official_gemini",
    "official_claude",
    "official_openai",
    "web_gemini",
    "web_claude",
    "web_chatgpt",
    "web_codex",
}

# Starting points only; every channel can override with config["default_model"].
DEFAULT_MODELS = {
    "official_openai": "gpt-4o-mini",
    "official_claude": "claude-sonnet-5",
    "official_gemini": "gemini-2.5-flash",
}

VERSION = "0.1.0"

# How often the router's usage/health snapshot is written to disk. 0 disables the
# periodic save; the shutdown save still runs.
ROUTER_STATE_SAVE_SECONDS = int_env("ROUTER_STATE_SAVE_SECONDS", 60, 0)

QUOTA_EXHAUSTED_THRESHOLD = int_env("QUOTA_EXHAUSTED_THRESHOLD", 100, 0)   # tokens
QUOTA_EXHAUSTED_REQUESTS = int_env("QUOTA_EXHAUSTED_REQUESTS", 1, 0)       # requests
QUOTA_COOLDOWN_SECONDS = float(int_env("QUOTA_COOLDOWN_SECONDS", 60, 1))
# A reading older than this is ignored. Streaming responses expose no headers, so without an
# expiry one low reading would park a streaming-only channel (Claude Code, Cursor) forever.
QUOTA_STALE_SECONDS = float(int_env("QUOTA_STALE_SECONDS", 300, 30))

WEB_RESPONSE_TIMEOUT_MS = int_env("WEB_RESPONSE_TIMEOUT_MS", 120000, 5000)
# Consecutive unchanged polls that mean the streamed answer has finished rendering.
WEB_STREAM_SETTLE_POLLS = int_env("WEB_STREAM_SETTLE_POLLS", 8, 2)

CORS_ORIGINS = parse_csv_env("CORS_ORIGINS")
CORS_ORIGIN_REGEX = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
)

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def is_loopback_bind(host: str) -> bool:
    return (host or "").lower() in LOOPBACK_HOSTS


def require_admin_token() -> bool:
    return bool(ADMIN_TOKEN)


def validate_startup_security():
    """Fail closed on a non-local bind that has no tokens.

    Called at webapp import as well as from the CLI, so `uvicorn main:app` with HOST=0.0.0.0
    cannot quietly expose an unauthenticated admin API by skipping the CLI entry point.
    """
    if not is_loopback_bind(HOST):
        if GENERATED_ADMIN_TOKEN:
            raise SystemExit("Refusing non-local bind without explicit ADMIN_TOKEN.")
        if not API_TOKEN:
            raise SystemExit("Refusing non-local bind without API_TOKEN.")
