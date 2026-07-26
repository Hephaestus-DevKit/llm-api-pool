"""channels.json round-tripping, at-rest secret handling, and crash safety."""
from __future__ import annotations

import json
import os

import main
import pytest

from llm_pool import secretbox


def read_saved(tmp_path):
    return json.loads((tmp_path / "channels.json").read_text(encoding="utf-8"))


def test_save_then_load_round_trips(pool, tmp_path):
    pool.CHANNELS.append({"id": "c1", "type": "official_openai", "name": "mine",
                          "config": {"api_key": "sk-live-1", "quota": 42}})
    pool.save_channels()
    pool.CHANNELS.clear()
    pool.load_channels()
    assert pool.CHANNELS[0]["config"] == {"api_key": "sk-live-1", "quota": 42}


def test_save_is_atomic_and_leaves_no_temp_file(pool, tmp_path):
    pool.CHANNELS.append({"id": "c1", "type": "official_openai", "name": "n", "config": {}})
    pool.save_channels()
    assert (tmp_path / "channels.json").exists()
    assert not (tmp_path / "channels.json.tmp").exists()


def test_a_corrupt_file_does_not_crash_startup(pool, tmp_path):
    (tmp_path / "channels.json").write_text("{ not json", encoding="utf-8")
    pool.load_channels()  # must not raise
    assert any(event["message"] == "channels_load_failed" for event in pool.DIAGNOSTIC_EVENTS)


def test_a_byte_order_mark_is_tolerated(pool, tmp_path):
    """Notepad and PowerShell both like to write a BOM."""
    payload = [{"id": "c1", "type": "official_openai", "name": "n", "config": {"api_key": "k"}}]
    (tmp_path / "channels.json").write_text(json.dumps(payload), encoding="utf-8-sig")
    pool.load_channels()
    assert pool.CHANNELS[0]["id"] == "c1"


def test_runtime_counters_are_stripped_on_load(pool, tmp_path):
    payload = [{"id": "c1", "type": "official_openai", "name": "n",
                "config": {"api_key": "k"}, "stats": {"calls": 999}}]
    (tmp_path / "channels.json").write_text(json.dumps(payload), encoding="utf-8")
    pool.load_channels()
    assert "stats" not in pool.CHANNELS[0]


def test_missing_file_is_not_an_error(pool, tmp_path):
    assert not (tmp_path / "channels.json").exists()
    pool.load_channels()
    assert pool.CHANNELS == []


# ---------------------------------------------------------------- secret envelopes

def test_envelope_detection():
    assert main.is_secret_envelope({main.SECRET_ENVELOPE_KEY: "dpapi", "value": "x"}) is True
    assert main.is_secret_envelope({"value": "x"}) is False
    assert main.is_secret_envelope("plain") is False


def test_plaintext_channels_are_flagged_for_migration():
    assert main.has_plaintext_secret({"config": {"api_key": "sk-x"}}) is True
    assert main.has_plaintext_secret(
        {"config": {"api_key": {main.SECRET_ENVELOPE_KEY: "dpapi", "value": "x"}}}) is False
    assert main.has_plaintext_secret({"config": {"quota": 5}}) is False


@pytest.mark.skipif(not main._is_windows(), reason="DPAPI is Windows only")
def test_dpapi_round_trip_and_ciphertext_at_rest(pool, tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_POOL_ALLOW_PLAINTEXT_SECRETS", raising=False)
    pool.CHANNELS.append({"id": "c1", "type": "web_claude", "name": "n",
                          "config": {"cookies": {"sessionKey": "sk-live-secret"}}})
    pool.save_channels()
    raw = (tmp_path / "channels.json").read_text(encoding="utf-8")
    assert "sk-live-secret" not in raw
    assert main.SECRET_ENVELOPE_KEY in raw

    pool.CHANNELS.clear()
    pool.load_channels()
    assert pool.CHANNELS[0]["config"]["cookies"] == {"sessionKey": "sk-live-secret"}


@pytest.mark.skipif(not main._is_windows(), reason="DPAPI is Windows only")
def test_envelope_encryption_is_memoized(monkeypatch):
    """Re-encrypting unchanged secrets on every save meant a blocking ctypes call per key."""
    calls = []
    real = secretbox._dpapi_protect
    monkeypatch.setattr(secretbox, "_dpapi_protect", lambda data: (calls.append(1), real(data))[1])
    main._SECRET_ENVELOPE_CACHE.clear()
    for _ in range(20):
        main.protect_secret_value("sk-repeated-value")
    assert len(calls) == 1


def test_an_already_protected_value_is_not_double_wrapped():
    envelope = {main.SECRET_ENVELOPE_KEY: "dpapi", "scope": "current_user", "value": "abc"}
    assert main.protect_secret_value(envelope) is envelope


def test_none_secrets_pass_through():
    assert main.protect_secret_value(None) is None
    assert main.unprotect_secret_value(None) is None


def test_undecryptable_secret_does_not_break_the_load(pool, tmp_path, capsys):
    """A profile copied to another machine cannot be decrypted; the app must still start."""
    payload = [{"id": "c1", "type": "official_openai", "name": "n", "config": {
        "api_key": {main.SECRET_ENVELOPE_KEY: "dpapi", "scope": "current_user", "value": "bm90LXJlYWw="}}}]
    (tmp_path / "channels.json").write_text(json.dumps(payload), encoding="utf-8")
    pool.load_channels()
    assert len(pool.CHANNELS) == 1


def test_channel_ids_stay_unique(pool):
    pool.CHANNELS.extend({"id": f"c{i}", "type": "official_openai", "name": "n", "config": {}}
                         for i in range(50))
    assert main.new_channel_id() not in {c["id"] for c in pool.CHANNELS}


def test_saved_file_permissions_are_restricted_on_posix(pool, tmp_path):
    pool.CHANNELS.append({"id": "c1", "type": "official_openai", "name": "n", "config": {}})
    pool.save_channels()
    if os.name != "nt":
        assert (os.stat(tmp_path / "channels.json").st_mode & 0o777) == 0o600
