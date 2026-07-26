"""Channel list state and its persistence to channels.json.

CHANNELS is the single live copy shared by the router, the API layer and the tests;
mutate it in place or rebind it via this module (``store.CHANNELS = ...``) only.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import uuid
from typing import Any, Dict, List

from .diagnostics import record_diagnostic_event, safe_path_for_diagnostics
from .paths import get_app_dir
from .secretbox import (
    SECRET_CONFIG_KEYS,
    has_plaintext_secret,
    is_secret_envelope,
    protect_secret_value,
    unprotect_secret_value,
)

CHANNELS_FILE = os.getenv("CHANNELS_FILE", str(get_app_dir() / "channels.json"))
CHANNELS: List[Dict[str, Any]] = []
channels_lock = asyncio.Lock()  # serializes admin mutations of CHANNELS + the file


def encrypt_channel_for_disk(ch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(ch)
    # Live counters belong to the router, not the config file: persisting them bloated every
    # save and the values were never read back on startup anyway.
    out.pop("stats", None)
    cfg = out.get("config") or {}
    for key in SECRET_CONFIG_KEYS:
        if key in cfg:
            cfg[key] = protect_secret_value(cfg[key])
    out["config"] = cfg
    return out


def decrypt_channel_from_disk(ch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(ch)
    out.pop("stats", None)  # drop counters written by older versions
    cfg = out.get("config") or {}
    for key in SECRET_CONFIG_KEYS:
        if key in cfg and is_secret_envelope(cfg[key]):
            try:
                cfg[key] = unprotect_secret_value(cfg[key])
            except Exception as e:
                print(f"[secrets] Could not decrypt {key} for channel {out.get('id')}: {e}")
    out["config"] = cfg
    return out


def restrict_channels_file_permissions():
    # Meaningful on POSIX. On Windows chmod only toggles the read-only bit, which is why every
    # secret field is DPAPI-encrypted for the current user rather than relying on file mode.
    try:
        os.chmod(CHANNELS_FILE, 0o600)
    except Exception:
        pass


def load_channels():
    global CHANNELS
    if os.path.exists(CHANNELS_FILE):
        try:
            with open(CHANNELS_FILE, "r", encoding="utf-8-sig") as f:  # utf-8-sig tolerates BOM from Notepad/PS writes
                raw_channels = json.load(f)
            needs_resave = any(has_plaintext_secret(c) for c in raw_channels)
            CHANNELS = [decrypt_channel_from_disk(c) for c in raw_channels]
            print(f"Loaded {len(CHANNELS)} channels from {CHANNELS_FILE}")
            record_diagnostic_event("info", "channels_loaded", count=len(CHANNELS), channels_file=safe_path_for_diagnostics(CHANNELS_FILE))
            if needs_resave:
                print("[secrets] Migrating plaintext channel secrets to encrypted storage.")
                record_diagnostic_event("info", "plaintext_secrets_migrated", count=len(CHANNELS), channels_file=safe_path_for_diagnostics(CHANNELS_FILE))
                save_channels()
        except Exception as e:
            print(f"Failed to load channels: {e}")
            record_diagnostic_event("error", "channels_load_failed", error=str(e), channels_file=safe_path_for_diagnostics(CHANNELS_FILE))


def save_channels():
    """Atomic save to prevent corruption on crash/power loss (critical for portable exe)."""
    tmp = CHANNELS_FILE + ".tmp"
    try:
        disk_channels = [encrypt_channel_for_disk(c) for c in CHANNELS]
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(disk_channels, f, indent=2, ensure_ascii=False)
        os.replace(tmp, CHANNELS_FILE)  # atomic on POSIX/Windows
        restrict_channels_file_permissions()
    except Exception as e:
        print(f"Failed to save channels: {e}")
        record_diagnostic_event("error", "channels_save_failed", error=str(e), channels_file=safe_path_for_diagnostics(CHANNELS_FILE))
        # best effort cleanup
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


async def save_channels_async() -> None:
    """Admin writes go through a worker thread so a slow disk cannot stall the event loop."""
    await asyncio.to_thread(save_channels)


def router_state_file() -> str:
    """Router stats live next to channels.json so both follow CHANNELS_FILE overrides."""
    parent = os.path.dirname(os.path.abspath(CHANNELS_FILE))
    return os.path.join(parent, "router_state.json")


def new_channel_id() -> str:
    while True:
        cid = uuid.uuid4().hex[:12]
        if not any(c.get("id") == cid for c in CHANNELS):
            return cid


# Load on start
load_channels()
