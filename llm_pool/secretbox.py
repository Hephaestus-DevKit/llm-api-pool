"""At-rest protection for channel secrets: Windows DPAPI envelopes with a plaintext fallback."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from collections import OrderedDict
from typing import Any, Dict

from .paths import _is_windows

SECRET_CONFIG_KEYS = {"api_key", "password", "cookies"}
SECRET_ENVELOPE_KEY = "__llm_pool_secret__"


def _dpapi_protect(data: bytes) -> bytes:
    if not _is_windows():
        raise RuntimeError("Windows DPAPI is unavailable on this platform")
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.c_void_p)]

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(DATA_BLOB),
        wintypes.LPCWSTR,
        ctypes.POINTER(DATA_BLOB),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DATA_BLOB),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    in_buf = ctypes.create_string_buffer(data)
    in_blob = DATA_BLOB(len(data), ctypes.cast(in_buf, ctypes.c_void_p))
    out_blob = DATA_BLOB()
    if not crypt32.CryptProtectData(ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(out_blob.pbData)


def _dpapi_unprotect(data: bytes) -> bytes:
    if not _is_windows():
        raise RuntimeError("Windows DPAPI is unavailable on this platform")
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.c_void_p)]

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(DATA_BLOB),
        ctypes.c_void_p,
        ctypes.POINTER(DATA_BLOB),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DATA_BLOB),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    in_buf = ctypes.create_string_buffer(data)
    in_blob = DATA_BLOB(len(data), ctypes.cast(in_buf, ctypes.c_void_p))
    out_blob = DATA_BLOB()
    if not crypt32.CryptUnprotectData(ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(out_blob.pbData)


def is_secret_envelope(value: Any) -> bool:
    return isinstance(value, dict) and value.get(SECRET_ENVELOPE_KEY) == "dpapi"


# DPAPI is a blocking ctypes call, so re-encrypting unchanged keys on every save would stall
# the event loop for no benefit. Envelopes are deterministic per plaintext, so memoize them.
_SECRET_ENVELOPE_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_SECRET_ENVELOPE_CACHE_MAX = 256


def protect_secret_value(value: Any) -> Any:
    if value is None or is_secret_envelope(value):
        return value
    if not _is_windows():
        if os.getenv("LLM_POOL_ALLOW_PLAINTEXT_SECRETS") != "1":
            print("[secrets] DPAPI unavailable; saving plaintext secrets. Set LLM_POOL_ALLOW_PLAINTEXT_SECRETS=1 to silence this warning.")
        return value
    payload = json.dumps({"value": value}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    cached = _SECRET_ENVELOPE_CACHE.get(digest)
    if cached is not None:
        _SECRET_ENVELOPE_CACHE.move_to_end(digest)
        return dict(cached)
    envelope = {
        SECRET_ENVELOPE_KEY: "dpapi",
        "scope": "current_user",
        "value": base64.b64encode(_dpapi_protect(payload)).decode("ascii"),
    }
    _SECRET_ENVELOPE_CACHE[digest] = envelope
    while len(_SECRET_ENVELOPE_CACHE) > _SECRET_ENVELOPE_CACHE_MAX:
        _SECRET_ENVELOPE_CACHE.popitem(last=False)
    return dict(envelope)


def unprotect_secret_value(value: Any) -> Any:
    if not is_secret_envelope(value):
        return value
    encrypted = base64.b64decode(value.get("value", ""))
    payload = _dpapi_unprotect(encrypted)
    return json.loads(payload.decode("utf-8")).get("value")


def has_plaintext_secret(ch: Dict[str, Any]) -> bool:
    cfg = ch.get("config") or {}
    return any(key in cfg and cfg[key] is not None and not is_secret_envelope(cfg[key]) for key in SECRET_CONFIG_KEYS)


def redact_config(config: Dict[str, Any]) -> Dict[str, Any]:
    redacted = {}
    for key, value in (config or {}).items():
        if key in SECRET_CONFIG_KEYS:
            if key == "cookies" and isinstance(value, dict):
                redacted[key] = f"<redacted:{len(value)} cookies>"
            else:
                redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted
