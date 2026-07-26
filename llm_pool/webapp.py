"""FastAPI application: dashboard, admin API, and the unified /v1 endpoints."""
from __future__ import annotations

import html
import json
import os
import platform
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import backends, dispatch, routing, security, settings, store, webdrive
from .canonical import anthropic_to_canonical, canonical_to_anthropic, canonical_to_openai, openai_to_canonical
from .diagnostics import (
    DIAGNOSTIC_EVENTS,
    record_diagnostic_event,
    safe_path_for_diagnostics,
)
from .paths import _is_windows, get_app_dir, get_resource_path
from .secretbox import SECRET_CONFIG_KEYS, redact_config


class AddChannelRequest(BaseModel):
    type: Optional[str] = None
    name: Optional[str] = None
    api_key: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    # Values are either a raw cookie string or a full Playwright cookie dict
    # ({"value", "domain", "path", ...}); get_or_create_web_context handles both.
    cookies: Optional[Dict[str, Any]] = None
    quota: Optional[int] = None  # estimated quota for monitoring
    quota_category: Optional[str] = None  # "chat", "codex", "general"
    aliases: Optional[Dict[str, str]] = None  # e.g. {"sonnet": "claude-sonnet-5"}
    # None means "not provided". A concrete default here would make every PUT that omits
    # the field silently reset the stored value (the update loop treats non-None as "set").
    priority: Optional[int] = None
    max_concurrent: Optional[int] = None
    default_model: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup messages are in cli.main() for a clearer local desktop experience.
    yield
    print("Shutting down...")
    await backends.aclose_all()
    await webdrive.shutdown()
    print("Shutdown complete.")


def is_document_navigation(request: Request) -> bool:
    """True when the browser is loading this URL as a top-level page.

    Fetch/XHR sends Sec-Fetch-Dest: empty; a real navigation sends "document". Non-browser
    clients (curl, the CI smoke test, the test suite) send neither and are treated as
    navigations. Every modern browser sends these headers and they cannot be set from script.
    """
    dest = request.headers.get("sec-fetch-dest")
    mode = request.headers.get("sec-fetch-mode")
    if dest is None and mode is None:
        return True
    return dest == "document" and mode in (None, "navigate")


def may_receive_bootstrap_token(request: Request) -> bool:
    """Gate for injecting the generated admin token into the dashboard HTML.

    The default CORS policy allows any localhost origin, so without the navigation check any
    page served from another local port could fetch "/" cross-origin, scrape the token out of
    the HTML, and drive the whole admin API with it.
    """
    if not settings.GENERATED_ADMIN_TOKEN:
        return False
    if not (request.client and request.client.host in settings.LOOPBACK_HOSTS):
        return False
    return is_document_navigation(request)


settings.validate_startup_security()

app = FastAPI(title="LLM API Pool", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=settings.CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Admin-Token", "X-Api-Key"],
)

require_admin = security.require_admin
require_api_access = security.require_api_access


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    """Serve the external dashboard.html.
    - Fast Python startup (small entry module, external html)
    - Works when run as python or as PyInstaller exe (onefile extracts assets to _MEIPASS)
    - Same dashboard.html can be dropped as index.html on GitHub Pages for the static "web version" UI (just enter your backend URL once).
    """
    try:
        p = get_resource_path("dashboard.html")
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        if may_receive_bootstrap_token(request):
            bootstrap = (
                "<script>"
                f"window.__LLM_POOL_BOOTSTRAP__={{adminToken:{json.dumps(settings.ADMIN_TOKEN)},generatedAdminToken:true}};"
                "</script>"
            )
            text = text.replace("</head>", f"{bootstrap}\n</head>", 1)
        # The token must never be cached or reused by a differently-scoped request.
        return HTMLResponse(text, headers={"Cache-Control": "no-store", "Vary": "Sec-Fetch-Dest, Sec-Fetch-Mode"})
    except Exception as e:
        details = ""
        if settings.DEBUG_ERRORS:
            app_dir = html.escape(str(get_app_dir()))
            meip = html.escape(str(getattr(sys, "_MEIPASS", None)))
            err = html.escape(str(e))
            details = f"<p>Error: {err}</p><p>cwd/app_dir: {app_dir}</p><p>_MEIPASS: {meip}</p>"
        return HTMLResponse(
            "<h1>dashboard.html not found</h1>"
            "<p>Make sure dashboard.html is next to the exe or source file, then rebuild if needed.</p>"
            f"{details}",
            status_code=404,
        )


@app.get("/health")
async def health():
    return {"status": "ok", "channels": len(store.CHANNELS), "arch": runtime_info()["arch"]}


def runtime_info() -> dict:
    arch = os.getenv("PROCESSOR_ARCHITEW6432") or os.getenv("PROCESSOR_ARCHITECTURE") or platform.machine() or "unknown"
    return {
        "host": settings.HOST,
        "port": settings.PORT,
        "arch": arch,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "admin_token_generated": settings.GENERATED_ADMIN_TOKEN,
        "api_token_required": bool(settings.API_TOKEN),
        "rate_limit_per_minute": settings.RATE_LIMIT_PER_MINUTE,
        "secrets_encryption": "dpapi" if _is_windows() else "plaintext-fallback",
    }


def channel_diagnostics() -> List[Dict[str, Any]]:
    rows = []
    for ch in store.CHANNELS:
        cfg = ch.get("config") or {}
        rows.append({
            "id": ch.get("id"),
            "type": ch.get("type"),
            "name": ch.get("name"),
            "config": redact_config(cfg),
            "secret_fields_present": sorted([key for key in SECRET_CONFIG_KEYS if cfg.get(key)]),
            "has_web_context": bool(ch.get("id") in webdrive._web_contexts),
        })
    return rows


def diagnostics_payload() -> Dict[str, Any]:
    return {
        "timestamp": time.time(),
        "runtime": runtime_info(),
        "paths": {
            "app_dir": safe_path_for_diagnostics(get_app_dir()),
            "channels_file": safe_path_for_diagnostics(store.CHANNELS_FILE),
            "channels_file_exists": os.path.exists(store.CHANNELS_FILE),
        },
        "security": {
            "admin_token_required": settings.require_admin_token(),
            "admin_token_generated": settings.GENERATED_ADMIN_TOKEN,
            "api_token_required": bool(settings.API_TOKEN),
            "remote_bind": not settings.is_loopback_bind(settings.HOST),
            "cors_origins": settings.CORS_ORIGINS,
            "cors_origin_regex": settings.CORS_ORIGIN_REGEX,
            "rate_limit_per_minute": settings.RATE_LIMIT_PER_MINUTE,
            "rate_limit_tracked_keys": security.api_rate_limiter.tracked_keys(),
            "trust_proxy_headers": settings.TRUST_PROXY_HEADERS,
            "secrets_encryption": "dpapi" if _is_windows() else "plaintext-fallback",
        },
        "routing": {
            "max_route_attempts": settings.MAX_ROUTE_ATTEMPTS,
            "cross_provider_fallback": settings.CROSS_PROVIDER_FALLBACK,
            "quota_exhausted_threshold": settings.QUOTA_EXHAUSTED_THRESHOLD,
            "quota_cooldown_seconds": settings.QUOTA_COOLDOWN_SECONDS,
            "default_models": dict(settings.DEFAULT_MODELS),
        },
        "browser": {
            "playwright_checked": webdrive._browser_checked,
            "open_contexts": len(webdrive._web_contexts),
        },
        "channels": channel_diagnostics(),
        "router_status": routing.router.get_status(),
        "events": list(DIAGNOSTIC_EVENTS),
    }


@app.get("/admin/diagnostics")
async def diagnostics(_admin: None = Depends(require_admin)):
    return diagnostics_payload()


@app.get("/admin/channels")
async def list_channels(_admin: None = Depends(require_admin)):
    return [
        {
            "id": c["id"],
            "type": c["type"],
            "name": c["name"],
            "config": redact_config(c.get("config", {}))
        }
        for c in store.CHANNELS
    ]


@app.post("/admin/channels")
async def add_channel(body: AddChannelRequest, _admin: None = Depends(require_admin)):
    if not body.type or body.type not in settings.VALID_CHANNEL_TYPES:
        raise HTTPException(400, f"type must be one of: {', '.join(sorted(settings.VALID_CHANNEL_TYPES))}")
    cid = store.new_channel_id()
    ch = {
        "id": cid,
        "type": body.type,
        "name": body.name or f"{body.type}-{cid}",
        "config": {}
    }

    if body.type.startswith("official_"):
        if not body.api_key:
            raise HTTPException(400, "api_key required for official")
        ch["config"]["api_key"] = body.api_key
    else:
        # web (including codex as separate GPT account category for quotas)
        if body.cookies:
            ch["config"]["cookies"] = body.cookies
            if body.email:
                ch["config"]["email"] = body.email
        elif body.email and body.password:
            print(f"[login] Attempting headless login for {body.type} {body.email}")
            try:
                cookies = await webdrive.extract_cookies_with_playwright(body.email, body.password, body.type)
                ch["config"]["cookies"] = cookies
                ch["config"]["email"] = body.email
                print(f"[login] Success for {body.type}")
                record_diagnostic_event("info", "web_login_succeeded", type=body.type)
            except Exception as e:
                print(f"[login] Failed: {e}")
                record_diagnostic_event("warn", "web_login_failed", type=body.type, error=str(e))
                # Note: For accounts with 2FA/SMS/captcha, password login often fails or requires interaction.
                # Strongly recommend: login manually in real browser, then paste the cookies JSON in the form.
                # Password mode works best for simple no-2FA accounts.
                raise HTTPException(400, f"Web login failed. Paste cookies instead. Detail: {str(e)[:180]}") from e
        else:
            raise HTTPException(400, "For web channel provide cookies or email+password")

        # Codex-style usage is metered in requests, plain chat in tokens, so they start from
        # very different budgets. Both remain overridable per channel.
        if body.type == "web_codex":
            ch["config"]["quota_category"] = body.quota_category or "codex"
            ch["config"]["quota"] = body.quota or 300
        elif "chatgpt" in body.type:
            ch["config"]["quota_category"] = body.quota_category or "chat"
            ch["config"]["quota"] = body.quota or 100000

    if body.quota is not None:
        ch["config"]["quota"] = body.quota
    if body.quota_category:
        ch["config"]["quota_category"] = body.quota_category
    if body.aliases:
        ch["config"]["aliases"] = body.aliases
    if body.priority is not None:
        ch["config"]["priority"] = body.priority
    if body.max_concurrent is not None:
        ch["config"]["max_concurrent"] = body.max_concurrent
    if body.default_model:
        ch["config"]["default_model"] = body.default_model

    async with store.channels_lock:
        store.CHANNELS.append(ch)
        await store.save_channels_async()
    record_diagnostic_event("info", "channel_added", channel_id=cid, type=ch["type"], name=ch["name"])
    return {"id": cid, "status": "added", "note": "For web with password, cookies were extracted if successful."}


@app.put("/admin/channels/{cid}")
async def update_channel(cid: str, body: AddChannelRequest, _admin: None = Depends(require_admin)):
    async with store.channels_lock:
        ch = next((c for c in store.CHANNELS if c["id"] == cid), None)
        if ch is None:
            raise HTTPException(404, "Channel not found")

        config = ch.get("config") or {}
        ch_type = body.type or ch["type"]
        was_official = ch["type"].startswith("official_")
        is_official = ch_type.startswith("official_")

        # Validate before touching anything: a rejected update must leave the channel exactly
        # as it was, not half-applied and waiting to be persisted by the next admin call.
        if ch_type not in settings.VALID_CHANNEL_TYPES:
            raise HTTPException(400, f"type must be one of: {', '.join(sorted(settings.VALID_CHANNEL_TYPES))}")
        # An existing credential only counts when the channel is staying in the same family.
        if is_official:
            if not (body.api_key or (was_official and config.get("api_key"))):
                raise HTTPException(400, "api_key required for official channels")
        else:
            if not (body.cookies or (not was_official and config.get("cookies"))):
                raise HTTPException(400, "cookies required for web channels; re-add the channel to log in by password")

        ch["type"] = ch_type
        if body.name:
            ch["name"] = body.name
        if is_official != was_official:
            # Switching families leaves the previous credential stranded on disk.
            config.pop("cookies" if is_official else "api_key", None)
        if is_official and body.api_key:
            config["api_key"] = body.api_key
        if not is_official:
            if body.cookies:
                config["cookies"] = body.cookies
            if body.email:
                config["email"] = body.email
        # Passwords are never stored; re-add the channel to redo a password login.
        for field, value in (("quota", body.quota), ("quota_category", body.quota_category),
                             ("aliases", body.aliases), ("priority", body.priority),
                             ("default_model", body.default_model)):
            if value is not None:
                config[field] = value
        if body.max_concurrent is not None:
            config["max_concurrent"] = body.max_concurrent
            # Safe to drop: in-flight requests release the semaphore they captured.
            routing.router.sems.pop(cid, None)
        ch["config"] = config
        await store.save_channels_async()

    record_diagnostic_event("info", "channel_updated", channel_id=cid, type=ch["type"], name=ch["name"])
    return {"id": cid, "status": "updated"}


@app.delete("/admin/channels/{cid}")
async def delete_channel(cid: str, _admin: None = Depends(require_admin)):
    async with store.channels_lock:
        deleted = next((c for c in store.CHANNELS if c["id"] == cid), None)
        if not deleted:
            raise HTTPException(404, "Channel not found")
        store.CHANNELS = [c for c in store.CHANNELS if c["id"] != cid]
        await store.save_channels_async()
    await webdrive.close_web_context(cid)
    routing.router.forget(cid)
    record_diagnostic_event("info", "channel_deleted", channel_id=cid, type=deleted.get("type"), name=deleted.get("name"))
    return {"deleted": cid}


@app.post("/v1/chat/completions")
async def chat_completions(body: dict, _api: None = Depends(require_api_access)):
    """OpenAI-compatible entry point (Cursor, Continue, Aider, the OpenAI SDK)."""
    canon = openai_to_canonical(body)
    if canon.stream:
        lease = await dispatch.open_stream_with_failover(canon, "/v1/chat/completions")
        return dispatch.stream_with_lease(lease, dispatch.openai_stream_response, "/v1/chat/completions")
    resp, _ch = await dispatch.generate_with_failover(canon, "/v1/chat/completions")
    return JSONResponse(canonical_to_openai(resp, stream=False))


@app.post("/v1/messages")
async def anthropic_messages(body: dict, _api: None = Depends(require_api_access)):
    """Anthropic-compatible entry point (Claude Code, Cline, the Anthropic SDK)."""
    canon = anthropic_to_canonical(body)
    if canon.stream:
        lease = await dispatch.open_stream_with_failover(canon, "/v1/messages")
        return dispatch.stream_with_lease(lease, dispatch.anthropic_stream_response, "/v1/messages")
    resp, _ch = await dispatch.generate_with_failover(canon, "/v1/messages")
    return JSONResponse(canonical_to_anthropic(resp, stream=False))


@app.get("/v1/models")
async def list_models(_api: None = Depends(require_api_access)):
    """Model ids this pool answers to, in OpenAI's list shape."""
    seen: Dict[str, dict] = {}
    created = int(time.time())
    for ch in store.CHANNELS:
        config = ch.get("config") or {}
        default_model = config.get("default_model") or settings.DEFAULT_MODELS.get(ch["type"])
        entries = [(default_model, None)] if default_model else []
        entries += [(alias, target) for alias, target in (config.get("aliases") or {}).items()]
        for model_id, root in entries:
            if not model_id or model_id in seen:
                continue
            entry = {"id": model_id, "object": "model", "created": created, "owned_by": ch["name"]}
            if root:
                entry["root"] = root
            seen[model_id] = entry
    if "auto" not in seen and store.CHANNELS:
        seen["auto"] = {"id": "auto", "object": "model", "created": created, "owned_by": "llm-api-pool"}
    return {"object": "list", "data": list(seen.values())}


# Status & monitoring (real-time quotas, health, for agents and dashboard)
@app.get("/admin/status")
async def pool_status(_admin: None = Depends(require_admin)):
    return {
        "channels": routing.router.get_status(),
        "total_channels": len(store.CHANNELS),
        "timestamp": time.time(),
        "runtime": runtime_info(),
    }
