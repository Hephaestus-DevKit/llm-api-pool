"""Auth, rate limiting, redaction, and the bootstrap-token exposure rules."""
from __future__ import annotations

import main
import pytest
from conftest import FakeBackend
from llm_pool import security, settings


@pytest.fixture
def admin_client(pool, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(settings, "ADMIN_TOKEN", "admin-secret")
    monkeypatch.setattr(settings, "GENERATED_ADMIN_TOKEN", False)
    with TestClient(pool.app, raise_server_exceptions=False) as test_client:
        yield test_client


# ---------------------------------------------------------------- admin auth

def test_admin_endpoints_reject_a_missing_token(admin_client):
    for path in ("/admin/channels", "/admin/status", "/admin/diagnostics"):
        assert admin_client.get(path).status_code == 401, path


def test_admin_endpoints_reject_a_wrong_token(admin_client):
    assert admin_client.get("/admin/status", headers={"X-Admin-Token": "nope"}).status_code == 401


def test_admin_endpoints_accept_either_header_form(admin_client):
    assert admin_client.get("/admin/status", headers={"X-Admin-Token": "admin-secret"}).status_code == 200
    assert admin_client.get("/admin/status", headers={"Authorization": "Bearer admin-secret"}).status_code == 200


def test_an_api_token_does_not_unlock_the_admin_api(admin_client, monkeypatch):
    monkeypatch.setattr(settings, "API_TOKEN", "api-secret")
    assert admin_client.get("/admin/status", headers={"X-Api-Key": "api-secret"}).status_code == 401


def test_delete_requires_admin(admin_client):
    assert admin_client.delete("/admin/channels/anything").status_code == 401


# ---------------------------------------------------------------- api auth

def test_v1_requires_the_api_token_when_configured(client, monkeypatch, make_channel):
    monkeypatch.setattr(settings, "API_TOKEN", "api-secret")
    make_channel("official_claude", backend=FakeBackend())
    body = {"model": "claude-sonnet-5", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/messages", json=body).status_code == 401
    assert client.post("/v1/messages", json=body, headers={"X-Api-Key": "api-secret"}).status_code == 200
    assert client.post("/v1/messages", json=body,
                       headers={"Authorization": "Bearer api-secret"}).status_code == 200


def test_v1_is_open_when_no_api_token_is_configured(client, make_channel):
    make_channel("official_claude", backend=FakeBackend())
    assert client.post("/v1/messages", json={
        "model": "claude-sonnet-5", "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}]}).status_code == 200


def test_token_comparison_is_constant_time():
    assert main.safe_token_match("a", "a") is True
    assert main.safe_token_match("a", "b") is False
    assert main.safe_token_match(None, "a") is False
    assert main.safe_token_match("a", "") is False


# ---------------------------------------------------------------- bootstrap token exposure

SECRET_TOKEN = "bootstrap-token-Zq7Yx"
# Only the injected script assigns to the global; dashboard.html merely reads it.
INJECTED_MARKER = "__LLM_POOL_BOOTSTRAP__={"


def loopback_client(pool):
    from fastapi.testclient import TestClient

    return TestClient(pool.app, client=("127.0.0.1", 54321))


def test_a_top_level_navigation_receives_the_generated_token(pool, monkeypatch):
    monkeypatch.setattr(settings, "GENERATED_ADMIN_TOKEN", True)
    monkeypatch.setattr(settings, "ADMIN_TOKEN", SECRET_TOKEN)
    with loopback_client(pool) as test_client:
        page = test_client.get("/", headers={"Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate"})
    assert SECRET_TOKEN in page.text
    assert page.headers["cache-control"] == "no-store"


def test_a_cross_origin_fetch_cannot_scrape_the_token(pool, monkeypatch):
    """Default CORS allows any localhost origin, so a page served from another local port
    could fetch "/" and read the admin token straight out of the HTML."""
    monkeypatch.setattr(settings, "GENERATED_ADMIN_TOKEN", True)
    monkeypatch.setattr(settings, "ADMIN_TOKEN", SECRET_TOKEN)
    with loopback_client(pool) as test_client:
        page = test_client.get("/", headers={
            "Origin": "http://localhost:3000",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
        })
    assert page.status_code == 200
    assert SECRET_TOKEN not in page.text
    assert INJECTED_MARKER not in page.text


def test_a_remote_client_never_receives_the_token(pool, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(settings, "GENERATED_ADMIN_TOKEN", True)
    monkeypatch.setattr(settings, "ADMIN_TOKEN", SECRET_TOKEN)
    with TestClient(pool.app, client=("10.0.0.7", 4000)) as test_client:
        page = test_client.get("/", headers={"Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate"})
    assert SECRET_TOKEN not in page.text


def test_no_token_is_injected_when_one_was_configured_explicitly(pool, monkeypatch):
    monkeypatch.setattr(settings, "GENERATED_ADMIN_TOKEN", False)
    monkeypatch.setattr(settings, "ADMIN_TOKEN", SECRET_TOKEN)
    with loopback_client(pool) as test_client:
        page = test_client.get("/")
    assert SECRET_TOKEN not in page.text
    assert INJECTED_MARKER not in page.text


def test_non_browser_clients_are_treated_as_navigations():
    """curl, the CI smoke test, and the packaged app's own launch have no Sec-Fetch headers."""
    class Req:
        headers: dict = {}

    assert main.is_document_navigation(Req()) is True


def test_a_subresource_load_is_not_a_navigation():
    class Req:
        headers = {"sec-fetch-dest": "iframe", "sec-fetch-mode": "navigate"}

    assert main.is_document_navigation(Req()) is False


# ---------------------------------------------------------------- rate limiting

def test_rate_limiter_enforces_the_window():
    limiter = main.SlidingWindowRateLimiter(3)
    assert [limiter.check("k") for _ in range(4)] == [True, True, True, False]


def test_rate_limiter_zero_means_unlimited():
    limiter = main.SlidingWindowRateLimiter(0)
    assert all(limiter.check("k") for _ in range(1000))


def test_rate_limiter_keys_are_independent():
    limiter = main.SlidingWindowRateLimiter(1)
    assert limiter.check("a") and limiter.check("b")
    assert not limiter.check("a")


def test_rate_limiter_key_table_stays_bounded():
    """The key includes the caller's token, so rotating tokens used to grow it without end."""
    limiter = main.SlidingWindowRateLimiter(120, max_keys=64)
    for i in range(5000):
        limiter.check(f"1.2.3.4:token{i}:/v1/chat/completions")
    assert len(limiter._hits) <= 64


def test_rate_limiter_sweeps_idle_keys(monkeypatch):
    limiter = main.SlidingWindowRateLimiter(120, sweep_interval=0.0)
    clock = [1000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    limiter.check("stale")
    clock[0] += 3600
    limiter.check("fresh")
    assert "stale" not in limiter._hits


def test_rate_limited_requests_return_429(pool, monkeypatch, make_channel):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(security, "api_rate_limiter", main.SlidingWindowRateLimiter(1))
    make_channel("official_claude", backend=FakeBackend())
    body = {"model": "claude-sonnet-5", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    with TestClient(pool.app, raise_server_exceptions=False) as test_client:
        assert test_client.post("/v1/messages", json=body).status_code == 200
        assert test_client.post("/v1/messages", json=body).status_code == 429


def test_forwarded_headers_are_ignored_unless_trusted(monkeypatch):
    class Req:
        headers = {"x-forwarded-for": "9.9.9.9"}
        client = type("C", (), {"host": "127.0.0.1"})()

    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)
    assert main.client_ip(Req()) == "127.0.0.1"
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", True)
    assert main.client_ip(Req()) == "9.9.9.9"


# ---------------------------------------------------------------- redaction

@pytest.mark.parametrize("text, leaked", [
    ("Authorization: Bearer sk-ant-api03-AAAAAAAAAAAAAAAAAA", "sk-ant-api03-AAAAAAAAAAAAAAAAAA"),
    ("api_key=sk-proj-BBBBBBBBBBBBBBBBBB", "sk-proj-BBBBBBBBBBBBBBBBBB"),
    ("key AIzaSyCCCCCCCCCCCCCCCCCC", "AIzaSyCCCCCCCCCCCCCCCCCC"),
    ("x-admin-token: hunter2hunter2", "hunter2hunter2"),
])
def test_secrets_are_redacted_from_text(text, leaked):
    assert leaked not in main.redact_sensitive_text(text)


def test_redact_config_hides_every_secret_field():
    redacted = main.redact_config({"api_key": "sk-x", "password": "pw",
                                   "cookies": {"a": "1", "b": "2"}, "quota": 10})
    assert redacted["api_key"] == "<redacted>"
    assert redacted["password"] == "<redacted>"
    assert redacted["cookies"] == "<redacted:2 cookies>"
    assert redacted["quota"] == 10


def test_diagnostics_never_expose_credentials(admin_client, pool):
    pool.CHANNELS.append({"id": "c", "type": "web_claude", "name": "c",
                          "config": {"cookies": {"sessionKey": "sk-live-secret"}, "api_key": "sk-key"}})
    payload = admin_client.get("/admin/diagnostics", headers={"X-Admin-Token": "admin-secret"}).json()
    body = main.json.dumps(payload)
    assert "sk-live-secret" not in body and "sk-key" not in body
    assert payload["channels"][0]["secret_fields_present"] == ["api_key", "cookies"]


def test_upstream_error_text_is_redacted_before_diagnostics(client, make_channel):
    make_channel("official_claude", backend=FakeBackend(
        error=RuntimeError("401 from provider, api_key=sk-proj-LEAKEDLEAKEDLEAKED")))
    response = client.post("/v1/messages", json={
        "model": "claude-sonnet-5", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
    assert "sk-proj-LEAKEDLEAKEDLEAKED" not in response.text
    assert "sk-proj-LEAKEDLEAKEDLEAKED" not in main.json.dumps(list(main.DIAGNOSTIC_EVENTS))


def test_diagnostic_events_are_capped():
    for i in range(main.DIAGNOSTIC_EVENT_LIMIT + 50):
        main.record_diagnostic_event("info", f"event {i}")
    assert len(main.DIAGNOSTIC_EVENTS) == main.DIAGNOSTIC_EVENT_LIMIT


# ---------------------------------------------------------------- startup posture

def test_non_local_bind_without_tokens_is_refused(monkeypatch):
    monkeypatch.setattr(settings, "HOST", "0.0.0.0")
    monkeypatch.setattr(settings, "GENERATED_ADMIN_TOKEN", True)
    with pytest.raises(SystemExit):
        main.validate_startup_security()


def test_non_local_bind_needs_an_api_token_too(monkeypatch):
    monkeypatch.setattr(settings, "HOST", "0.0.0.0")
    monkeypatch.setattr(settings, "GENERATED_ADMIN_TOKEN", False)
    monkeypatch.setattr(settings, "API_TOKEN", "")
    with pytest.raises(SystemExit):
        main.validate_startup_security()


def test_non_local_bind_with_both_tokens_is_allowed(monkeypatch):
    monkeypatch.setattr(settings, "HOST", "0.0.0.0")
    monkeypatch.setattr(settings, "GENERATED_ADMIN_TOKEN", False)
    monkeypatch.setattr(settings, "API_TOKEN", "api")
    main.validate_startup_security()


def test_loopback_bind_needs_nothing(monkeypatch):
    monkeypatch.setattr(settings, "HOST", "127.0.0.1")
    monkeypatch.setattr(settings, "GENERATED_ADMIN_TOKEN", True)
    main.validate_startup_security()


# ---------------------------------------------------------------- channel admin validation

def test_unknown_channel_type_is_rejected(admin_client):
    response = admin_client.post("/admin/channels", json={"type": "web_myspace", "api_key": "k"},
                                 headers={"X-Admin-Token": "admin-secret"})
    assert response.status_code == 400


def test_official_channel_requires_an_api_key(admin_client):
    response = admin_client.post("/admin/channels", json={"type": "official_openai"},
                                 headers={"X-Admin-Token": "admin-secret"})
    assert response.status_code == 400


def test_web_channel_requires_cookies_or_credentials(admin_client):
    response = admin_client.post("/admin/channels", json={"type": "web_claude"},
                                 headers={"X-Admin-Token": "admin-secret"})
    assert response.status_code == 400


def test_added_channel_is_listed_with_its_secret_redacted(admin_client):
    headers = {"X-Admin-Token": "admin-secret"}
    created = admin_client.post("/admin/channels",
                                json={"type": "official_openai", "api_key": "sk-live-1234567890"},
                                headers=headers)
    assert created.status_code == 200
    listing = admin_client.get("/admin/channels", headers=headers).json()
    assert listing[0]["config"]["api_key"] == "<redacted>"

    deleted = admin_client.delete(f"/admin/channels/{created.json()['id']}", headers=headers)
    assert deleted.status_code == 200
    assert admin_client.get("/admin/channels", headers=headers).json() == []
