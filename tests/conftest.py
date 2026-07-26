"""Shared fixtures.

CHANNELS_FILE must point somewhere disposable *before* main is imported, because the
store module loads channels at import time and would otherwise touch the developer's
real pool.

Reads may go through the `main` facade; monkeypatches must target the owning module
(store, routing, security, settings, ...) because rebinding an attribute on the facade
does not reach the implementation.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TMP = tempfile.TemporaryDirectory()
os.environ["CHANNELS_FILE"] = str(Path(_TMP.name) / "channels.json")
os.environ.setdefault("LLM_POOL_ALLOW_PLAINTEXT_SECRETS", "1")

import main  # noqa: E402
from llm_pool import routing, security, store  # noqa: E402


@pytest.fixture
def pool(monkeypatch, tmp_path):
    """A clean pool: no channels, no router state, saves redirected to a temp file."""
    monkeypatch.setattr(store, "CHANNELS_FILE", str(tmp_path / "channels.json"))
    monkeypatch.setattr(store, "CHANNELS", [])
    monkeypatch.setattr(routing, "router", routing.SmartRouter())
    monkeypatch.setattr(security, "api_rate_limiter", security.SlidingWindowRateLimiter(0))
    main.DIAGNOSTIC_EVENTS.clear()
    return main


@pytest.fixture
def client(pool):
    from fastapi.testclient import TestClient

    with TestClient(pool.app, raise_server_exceptions=False) as test_client:
        yield test_client


class FakeBackend(main.BaseBackend):
    """Stands in for a provider SDK. Records what the adapter layer handed it."""

    supports_tools = True

    def __init__(self, text="ok", tool_calls=None, usage=None, error=None,
                 stream_chunks=None, stream_error=None, headers=None):
        super().__init__()
        self.text = text
        self.tool_calls = tool_calls
        self.usage = usage
        self.error = error
        self.stream_chunks = stream_chunks
        self.stream_error = stream_error
        self.headers = headers or {}
        self.calls: list = []

    async def generate(self, req, ch):
        self.calls.append(req)
        if self.error:
            raise self.error
        return main.make_canonical_response(
            self.text, model=req.model, tool_calls=self.tool_calls, usage=self.usage,
        ), self.headers

    async def stream(self, req, ch):
        self.calls.append(req)
        if self.stream_error:
            raise self.stream_error
        for chunk in (self.stream_chunks if self.stream_chunks is not None else
                      [{"choices": [{"delta": {"content": self.text}}]}]):
            yield chunk


@pytest.fixture
def make_channel(pool):
    """Register a channel backed by a FakeBackend and return (channel, backend)."""
    created = []

    def _make(ch_type="official_claude", name=None, backend=None, **config):
        backend = backend or FakeBackend()
        channel = {
            "id": f"ch{len(pool.CHANNELS)}",
            "type": ch_type,
            "name": name or f"{ch_type}-{len(pool.CHANNELS)}",
            "config": {"api_key": "test-key", **config},
        }
        pool.CHANNELS.append(channel)
        pool.BACKENDS[ch_type] = backend
        created.append((ch_type, backend))
        return channel, backend

    original = dict(pool.BACKENDS)
    yield _make
    pool.BACKENDS.clear()
    pool.BACKENDS.update(original)
