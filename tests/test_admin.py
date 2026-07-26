"""Channel CRUD through the admin API."""
from __future__ import annotations

import main
import pytest

from llm_pool import settings


@pytest.fixture
def admin(pool, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(settings, "ADMIN_TOKEN", "admin-secret")
    monkeypatch.setattr(settings, "GENERATED_ADMIN_TOKEN", False)
    with TestClient(pool.app, raise_server_exceptions=False) as client:
        client.headers.update({"X-Admin-Token": "admin-secret"})
        yield client


def add(admin, **body):
    return admin.post("/admin/channels", json=body)


def test_add_official_channel(admin):
    response = add(admin, type="official_openai", name="mine", api_key="sk-1", quota=500)
    assert response.status_code == 200
    channel = main.CHANNELS[0]
    assert channel["name"] == "mine"
    assert channel["config"]["api_key"] == "sk-1"
    assert channel["config"]["quota"] == 500


def test_add_web_channel_with_cookies(admin):
    response = add(admin, type="web_claude", cookies={"sessionKey": "abc"})
    assert response.status_code == 200
    assert main.CHANNELS[0]["config"]["cookies"] == {"sessionKey": "abc"}


def test_web_codex_gets_its_own_quota_bucket(admin):
    add(admin, type="web_codex", cookies={"a": "b"})
    assert main.CHANNELS[0]["config"]["quota_category"] == "codex"


def test_cookie_values_may_be_full_playwright_dicts(admin):
    """get_or_create_web_context handles both raw strings and full cookie dicts."""
    response = add(admin, type="web_claude",
                   cookies={"sessionKey": {"value": "v", "domain": ".claude.ai", "path": "/"}})
    assert response.status_code == 200
    assert main.CHANNELS[0]["config"]["cookies"]["sessionKey"]["domain"] == ".claude.ai"


def test_update_changes_name_and_key(admin):
    cid = add(admin, type="official_openai", api_key="sk-old").json()["id"]
    response = admin.put(f"/admin/channels/{cid}", json={"name": "renamed", "api_key": "sk-new"})
    assert response.status_code == 200
    assert main.CHANNELS[0]["name"] == "renamed"
    assert main.CHANNELS[0]["config"]["api_key"] == "sk-new"


def test_a_rejected_update_leaves_the_channel_untouched(admin):
    """Validation used to run after the type and name had already been overwritten."""
    cid = add(admin, type="official_openai", name="original", api_key="sk-1").json()["id"]
    response = admin.put(f"/admin/channels/{cid}", json={"type": "web_claude", "name": "clobbered"})
    assert response.status_code == 400
    assert main.CHANNELS[0]["type"] == "official_openai"
    assert main.CHANNELS[0]["name"] == "original"


def test_unknown_type_on_update_is_rejected(admin):
    cid = add(admin, type="official_openai", api_key="sk-1").json()["id"]
    assert admin.put(f"/admin/channels/{cid}", json={"type": "web_myspace"}).status_code == 400
    assert main.CHANNELS[0]["type"] == "official_openai"


def test_switching_family_drops_the_stale_credential(admin):
    cid = add(admin, type="official_openai", api_key="sk-1").json()["id"]
    response = admin.put(f"/admin/channels/{cid}",
                         json={"type": "web_chatgpt", "cookies": {"s": "1"}})
    assert response.status_code == 200
    assert "api_key" not in main.CHANNELS[0]["config"]
    assert main.CHANNELS[0]["config"]["cookies"] == {"s": "1"}


def test_switching_to_official_requires_a_fresh_key(admin):
    cid = add(admin, type="web_claude", cookies={"s": "1"}).json()["id"]
    assert admin.put(f"/admin/channels/{cid}", json={"type": "official_claude"}).status_code == 400


def test_update_can_clear_aliases(admin):
    cid = add(admin, type="official_openai", api_key="sk-1", aliases={"fast": "gpt-4o-mini"}).json()["id"]
    admin.put(f"/admin/channels/{cid}", json={"aliases": {}})
    assert main.CHANNELS[0]["config"]["aliases"] == {}


def test_changing_concurrency_resets_the_semaphore(admin):
    cid = add(admin, type="official_openai", api_key="sk-1", max_concurrent=2).json()["id"]
    main.router.get_sem(main.CHANNELS[0])
    admin.put(f"/admin/channels/{cid}", json={"max_concurrent": 5})
    assert cid not in main.router.sems
    assert main.router.get_sem(main.CHANNELS[0])._value == 5


def test_update_without_priority_preserves_it(admin):
    """priority defaulted to 1 in the request model, so any unrelated PUT reset it."""
    cid = add(admin, type="official_openai", api_key="sk-1", priority=3).json()["id"]
    assert main.CHANNELS[0]["config"]["priority"] == 3
    assert admin.put(f"/admin/channels/{cid}", json={"name": "renamed"}).status_code == 200
    assert main.CHANNELS[0]["config"]["priority"] == 3


def test_update_of_a_missing_channel_is_404(admin):
    assert admin.put("/admin/channels/nope", json={"name": "x"}).status_code == 404


def test_delete_clears_router_state(admin):
    cid = add(admin, type="official_openai", api_key="sk-1").json()["id"]
    main.router.record_result(main.CHANNELS[0], False, 0.1)
    main.router.get_sem(main.CHANNELS[0])
    assert admin.delete(f"/admin/channels/{cid}").status_code == 200
    assert cid not in main.router.stats
    assert cid not in main.router.sems
    assert cid not in main.router.cooldowns


def test_delete_of_a_missing_channel_is_404(admin):
    assert admin.delete("/admin/channels/nope").status_code == 404


def test_channel_ids_are_unique_across_adds(admin):
    ids = {add(admin, type="official_openai", api_key=f"sk-{i}").json()["id"] for i in range(10)}
    assert len(ids) == 10
