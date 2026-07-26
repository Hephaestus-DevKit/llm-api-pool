"""At-rest protection for channel secrets.

Windows uses DPAPI (built in, per-user). macOS/Linux use the OS keychain via the optional
`keyring` package when it is installed and has a working backend; otherwise secrets fall
back to plaintext with a warning. Envelopes on disk carry only a pointer or ciphertext,
never the secret itself.
"""
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
SECRET_ENVELOPE_KINDS = {"dpapi", "keyring"}
KEYRING_SERVICE = "llm-api-pool"


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
    return isinstance(value, dict) and value.get(SECRET_ENVELOPE_KEY) in SECRET_ENVELOPE_KINDS


def _keyring_module():
    """The optional keyring backend, or None when it is missing or has no usable store
    (headless Linux without a Secret Service, for example)."""
    try:
        import keyring
        return keyring
    except Exception:
        return None


# DPAPI is a blocking ctypes call, so re-encrypting unchanged keys on every save would stall
# the event loop for no benefit. Envelopes are deterministic per plaintext, so memoize them.
_SECRET_ENVELOPE_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_SECRET_ENVELOPE_CACHE_MAX = 256


def _cache_envelope(digest: str, envelope: dict) -> dict:
    _SECRET_ENVELOPE_CACHE[digest] = envelope
    while len(_SECRET_ENVELOPE_CACHE) > _SECRET_ENVELOPE_CACHE_MAX:
        _SECRET_ENVELOPE_CACHE.popitem(last=False)
    return dict(envelope)


def protect_secret_value(value: Any) -> Any:
    if value is None or is_secret_envelope(value):
        return value
    payload = json.dumps({"value": value}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    cached = _SECRET_ENVELOPE_CACHE.get(digest)
    if cached is not None:
        _SECRET_ENVELOPE_CACHE.move_to_end(digest)
        return dict(cached)

    if _is_windows():
        envelope = {
            SECRET_ENVELOPE_KEY: "dpapi",
            "scope": "current_user",
            "value": base64.b64encode(_dpapi_protect(payload)).decode("ascii"),
        }
        return _cache_envelope(digest, envelope)

    keyring = _keyring_module()
    if keyring is not None:
        # Entry keys are the payload digest, so re-saving an unchanged secret is idempotent
        # and identical values across channels share one entry. Changing a secret leaves the
        # old entry behind in the keychain; harmless, and noted in the README.
        try:
            keyring.set_password(KEYRING_SERVICE, digest, payload.decode("utf-8"))
            envelope = {SECRET_ENVELOPE_KEY: "keyring", "service": KEYRING_SERVICE, "entry": digest}
            return _cache_envelope(digest, envelope)
        except Exception as e:
            print(f"[secrets] OS keyring rejected the write ({e}); falling back to plaintext.")

    if os.getenv("LLM_POOL_ALLOW_PLAINTEXT_SECRETS") != "1":
        print("[secrets] No DPAPI and no usable OS keyring; saving plaintext secrets. "
              "Install `keyring` for keychain-backed storage, or set "
              "LLM_POOL_ALLOW_PLAINTEXT_SECRETS=1 to silence this warning.")
    return value


def unprotect_secret_value(value: Any) -> Any:
    if not is_secret_envelope(value):
        return value
    kind = value.get(SECRET_ENVELOPE_KEY)
    if kind == "keyring":
        keyring = _keyring_module()
        if keyring is None:
            raise RuntimeError("secret is stored in the OS keyring but the `keyring` package is not installed")
        payload = keyring.get_password(value.get("service") or KEYRING_SERVICE, value.get("entry") or "")
        if payload is None:
            raise RuntimeError("keyring entry is missing (deleted, or a different OS user account)")
        return json.loads(payload).get("value")
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
