from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_tmp = tempfile.TemporaryDirectory()
os.environ["CHANNELS_FILE"] = str(Path(_tmp.name) / "channels.json")

import main  # noqa: E402


def fail(message: str) -> None:
    raise SystemExit(f"contract validation failed: {message}")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


expected_types = {
    "official_gemini",
    "official_claude",
    "official_openai",
    "web_gemini",
    "web_claude",
    "web_chatgpt",
    "web_codex",
}
web_types = {item for item in expected_types if item.startswith("web_")}
official_types = expected_types - web_types

assert_true(main.VALID_CHANNEL_TYPES == expected_types, "VALID_CHANNEL_TYPES drifted")
assert_true(set(main.BACKENDS.keys()) == expected_types, "BACKENDS does not cover every channel type")

for ch_type in web_types:
    domains = main.cookie_domains_for_channel(ch_type)
    assert_true(bool(domains), f"{ch_type} has no cookie domains")
    assert_true(all(isinstance(domain, str) and domain for domain in domains), f"{ch_type} has invalid cookie domains")

for ch_type in official_types:
    assert_true(ch_type in main.DEFAULT_MODELS, f"{ch_type} has no default model")

routes = {getattr(route, "path", "") for route in main.app.routes}
for path in {
    "/",
    "/health",
    "/admin/status",
    "/admin/diagnostics",
    "/admin/channels",
    "/v1/chat/completions",
    "/v1/messages",
    "/v1/models",
}:
    assert_true(path in routes, f"{path} route missing")

dashboard = (ROOT / "dashboard.html").read_text(encoding="utf-8")
for marker in {
    "__LLM_POOL_BOOTSTRAP__",
    "diagnosticsButton",
    "downloadDiagnostics",
    "/admin/diagnostics",
    "llmPoolApiToken",
    'h["X-Api-Key"] = API_TOKEN',
    "BOOTSTRAP.generatedAdminToken",
    "localStorage.removeItem(\"llmPoolAdminToken\")",
}:
    assert_true(marker in dashboard, f"dashboard marker missing: {marker}")

assert_true(
    'let ADMIN_TOKEN = BOOTSTRAP.generatedAdminToken' in dashboard,
    "generated local admin token must take precedence over stale localStorage",
)

assert_true("https://cdn" not in dashboard.lower(), "dashboard should remain self-contained")
assert_true("<script src=" not in dashboard.lower(), "dashboard should not load external scripts")
assert_true("<link rel=\"stylesheet\"" not in dashboard.lower(), "dashboard should not load external CSS")

print("Contract validation passed.")
