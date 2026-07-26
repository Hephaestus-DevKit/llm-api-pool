from __future__ import annotations

import json
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

for ch_type, backend in main.BACKENDS.items():
    assert_true(isinstance(backend, main.BaseBackend), f"{ch_type} backend is not a BaseBackend")
    assert_true(isinstance(backend.supports_tools, bool), f"{ch_type} does not declare tool support")

# Both dialects must survive a tool-carrying request. This is the exact shape Claude Code
# sends on every call, and getting it wrong returns HTTP 500 rather than a routing error.
anthropic_tool_request = {
    "model": "claude-sonnet-5",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    "tools": [{"name": "probe", "description": "d", "input_schema": {"type": "object"}}],
}
canonical = main.anthropic_to_canonical(anthropic_tool_request)
assert_true([tool.name for tool in canonical.tools or []] == ["probe"], "anthropic tools did not normalize")
assert_true(bool(main.canonical_tools_to_openai(canonical.tools)), "canonical tools do not render as OpenAI")
assert_true(bool(main.canonical_messages_to_anthropic(canonical)), "canonical messages do not render as Anthropic")

openai_tool_request = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hi"}],
    "tools": [{"type": "function", "function": {"name": "probe", "parameters": {"type": "object"}}}],
}
assert_true(
    [tool.name for tool in main.openai_to_canonical(openai_tool_request).tools or []] == ["probe"],
    "openai tools did not normalize",
)

# Tool output must never reach a provider as a Python repr.
tool_result = main.anthropic_to_canonical({"model": "m", "messages": [
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "42"}]}]})
assert_true("'type':" not in json.dumps(tool_result.messages[0].content), "tool_result leaked a Python repr")

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
    "supports_tools",
    "tool_calls",
    "partial_json",
}:
    assert_true(marker in dashboard, f"dashboard marker missing: {marker}")

# Both SSE extractors must split on the record separator, otherwise a frame straddling a
# chunk boundary is parsed as garbage and its text is silently dropped.
for extractor in ("extractOpenAIStream", "extractAnthropicStream"):
    start = dashboard.index(f"function {extractor}(")
    body = dashboard[start:start + 900]
    assert_true("split(/\\n\\n/)" in body, f"{extractor} must split on the SSE record separator")
    assert_true("events.pop()" in body, f"{extractor} must return the trailing partial record")

assert_true(
    'let ADMIN_TOKEN = BOOTSTRAP.generatedAdminToken' in dashboard,
    "generated local admin token must take precedence over stale localStorage",
)

assert_true("https://cdn" not in dashboard.lower(), "dashboard should remain self-contained")
assert_true("<script src=" not in dashboard.lower(), "dashboard should not load external scripts")
assert_true("<link rel=\"stylesheet\"" not in dashboard.lower(), "dashboard should not load external CSS")

print("Contract validation passed.")
