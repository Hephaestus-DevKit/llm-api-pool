# LLM API Pool

Local Windows software gateway for multiple LLM API keys and account sessions.

The app runs a FastAPI server, opens a desktop-style dashboard, and exposes OpenAI-compatible and Anthropic-compatible endpoints for tools such as Cursor, Continue, Aider, Claude Code, Cline, and custom scripts.

Repository: https://github.com/Hephaestus-DevKit/llm-api-pool

## What It Provides

- Local API server at `http://localhost:8080`.
- OpenAI-compatible endpoint: `POST /v1/chat/completions`.
- Anthropic-compatible endpoint: `POST /v1/messages`.
- Model list endpoint: `GET /v1/models`.
- Official API key channels for OpenAI, Anthropic, and Gemini.
- Advanced web-session channels for ChatGPT, Codex-style GPT usage, Claude, and Gemini through Playwright browser automation.
- Tool calling in both dialects, streaming included, with tool definitions, tool calls, and tool results translated between the OpenAI and Anthropic shapes.
- Automatic failover: a request that fails upstream is retried on the next-best channel.
- Smart routing by provider compatibility, health, real quota headers, latency, in-flight load, tool support, priority, and cooldown state.
- Self-contained dashboard with account health, quota estimate, latency, in-flight load, and a playground.
- Admin diagnostics export for sanitized runtime, channel, router, browser, and recent event state.
- Portable Windows `--onedir` build for faster startup than PyInstaller onefile extraction.

Official API channels are the recommended production path. Web-session channels are useful for personal quota pooling, but they are inherently more fragile because provider pages, cookies, 2FA, captcha, and browser automation can change.

### Format Translation

The pool normalizes every request into one internal format and renders the answer back in whichever dialect the caller used, so an OpenAI client can talk to a Claude channel and vice versa. Translated in both directions:

- Tool declarations (`{"type": "function", ...}` and `{"name", "input_schema"}`).
- Tool choice, including "use this specific tool" and "must call something".
- Assistant tool calls and tool results, whether carried as OpenAI `tool_calls` plus `role: "tool"` messages or as Anthropic `tool_use` / `tool_result` content blocks.
- Base64 and URL images, including images returned inside an Anthropic `tool_result`.
- Streaming, in both SSE protocols. Text streams incrementally. Tool-call arguments stream incrementally in the OpenAI protocol; in the Anthropic protocol they are emitted as a complete block, because that protocol allows only one open content block at a time and never reopens a closed one.
- Token usage and stop reasons, using the provider's real counts rather than an estimate.

Gemini and web-session channels are text-only. A request carrying tools is routed only to a tool-capable channel, unless the pool has none.

## Release Packages

Use the package that matches your Windows device:

- `llm-pool-windows-x64.zip`: primary Windows x64 package.
- `llm-pool-windows-arm64.zip`: native ARM64 package, built best-effort (see below).
- Each zip is published with a `.sha256` file.

The native ARM64 package is built from `requirements-arm64.txt` and goes through the same test, lint, contract, and frozen-exe smoke gates as x64, with two deliberate differences forced by win_arm64 wheel availability: `uvicorn` runs without the `[standard]` extras (httptools has no ARM64 wheel; this app serves no websockets), and the official Gemini SDK is excluded (google-auth hard-requires `cryptography`, which publishes no ARM64 wheels). Official Gemini channels on ARM64 return a clear error pointing at `web_gemini` or the x64 build; everything else is at full parity. The ARM64 leg is marked best-effort: if the ARM64 ecosystem regresses, the x64 release still ships and only the ARM64 asset is skipped. When a release has no ARM64 zip, use the x64 package under Windows' built-in x64 emulation.

Do not use a 32-bit x86 package; this project depends on Playwright/Chromium and modern Python packages, so 32-bit Windows is not a sensible support target.

## Quick Start: Windows App

1. Download the release zip for your architecture.
2. Extract it.
3. Run `llm-pool.exe` inside the extracted `llm-pool` folder.
4. The dashboard opens automatically.
5. Add an official API key first; add web-session channels only if you need them.
6. Use `http://localhost:8080/v1` as the base URL in your tool.

When no `ADMIN_TOKEN` is configured, the app generates a random local admin token at startup and injects it only into the loopback dashboard. This keeps double-click local usage smooth while still requiring a token for admin API calls.

Runtime data such as `channels.json` and Playwright profiles is stored next to the executable. API keys and cookies in `channels.json` are encrypted with Windows DPAPI for the current Windows user. Do not share that app folder if it contains personal accounts or browser profiles.

The first web-session channel triggers a one-time Chromium download (~150 MB). To avoid it on machines without internet, copy an `ms-playwright` folder next to the executable; the app uses a bundled browser directory in preference to the user profile.

## Quick Start: Source

```powershell
pip install -r requirements.txt
python main.py
```

Useful options:

```powershell
python main.py --host 127.0.0.1 --port 8080
python main.py --no-open
python main.py --install-browser
```

## API Usage

OpenAI-compatible:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello from the pool"}],
    "stream": false
  }'
```

Anthropic-compatible:

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello from the pool"}],
    "stream": false
  }'
```

If `API_TOKEN` is configured, send it as either:

```text
Authorization: Bearer <token>
```

or:

```text
X-Api-Key: <token>
```

## Accounts

Official channels:

- `official_openai`
- `official_claude`
- `official_gemini`

Advanced web-session channels:

- `web_chatgpt`
- `web_codex`
- `web_claude`
- `web_gemini`

For web-session channels, pasted cookies are more reliable than password login when an account uses 2FA, captcha, SSO, or device verification. Password login is best-effort browser automation and may fail when the provider changes its login UI.

## Remote Hosting

Remote hosting must be explicit. The app refuses non-local binds unless both `ADMIN_TOKEN` and `API_TOKEN` are set:

```powershell
$env:HOST="0.0.0.0"
$env:PORT="8080"
$env:ADMIN_TOKEN="replace-with-strong-admin-token"
$env:API_TOKEN="replace-with-strong-api-token"
$env:CORS_ORIGINS="https://your-dashboard.example"
python main.py
```

Security defaults:

- `/admin/*` always requires `ADMIN_TOKEN`.
- `/v1/*` requires `API_TOKEN` when configured, and remote mode requires it.
- Tokens are accepted in headers, not URL query strings.
- The non-local bind check runs at import, so `uvicorn main:app` cannot bypass it.
- API requests are rate-limited by client/token/path. Default: `RATE_LIMIT_PER_MINUTE=120`. The key table is bounded and swept, so rotating tokens cannot grow it without limit.
- Client identity comes from the peer address. `X-Forwarded-For` is honoured only when `TRUST_PROXY_HEADERS=1`, because a spoofable header would hand every caller a fresh rate-limit bucket.
- CORS defaults to localhost only. Add exact external origins through `CORS_ORIGINS`.
- The generated local admin token is injected into `/` only for top-level browser navigations from loopback. A cross-origin `fetch` from another localhost page gets the dashboard without the token.
- `/admin/diagnostics` returns a sanitized JSON snapshot for debugging. It redacts API keys, passwords, cookies, and tokens.
- Set `DEBUG_ERRORS=1` only while debugging packaging issues locally.

## Routing

Each request is scored across every eligible channel and one is chosen at random *weighted by that score*, so `priority`, health, latency, reported quota and in-flight load genuinely steer traffic while a burst still spreads across accounts instead of hammering the single best one. A saturated channel is scored down rather than queued behind, so load routes around congestion.

A failed call is retried on the next-best channel, up to `MAX_ROUTE_ATTEMPTS` (default 3). Streaming requests are retried only while nothing has reached the client yet; once the first chunk is out, switching channels would splice two answers together.

If no channel matches the requested provider, the pool returns `503` naming the model rather than answering from a different provider. Set `CROSS_PROVIDER_FALLBACK=1` if you would rather get any answer than an error. When a request carries tools, only tool-capable channels are considered unless the pool has none — a text-only channel would drop the tools and answer in prose, which reads as the model ignoring its instructions.

Channels are cooled down after three consecutive failures (backoff grows with the streak, capped at 10 minutes) or when a provider reports its remaining quota below the exhaustion threshold. Token and request budgets are thresholded separately, and a quota reading older than `QUOTA_STALE_SECONDS` is ignored — streaming responses carry no rate-limit headers, so a stale reading would otherwise park a streaming-only channel permanently.

Request shaping is per-provider: an unspecified temperature is left unspecified rather than defaulted, OpenAI reasoning models (`o1`/`o3`/`o4`) get `max_completion_tokens` and no temperature, and Anthropic's 0–1 temperature range is respected.

## Diagnostics

Use the dashboard `Diagnostics` button or call:

```powershell
Invoke-RestMethod http://localhost:8080/admin/diagnostics -Headers @{"X-Admin-Token"="<token>"}
```

The export includes runtime information, security posture, data-file location, channel health, router state, browser context count, and recent sanitized events. It is intended for local debugging and issue reports; it does not include raw prompts, API keys, cookies, passwords, or tokens.

## Dashboard and GitHub Pages

`dashboard.html` is self-contained and has no external CSS or JavaScript dependency. It works in the packaged app and can also be hosted as a static page.

For a static dashboard:

1. Publish `dashboard.html` as `index.html`.
2. Open the page.
3. Set the backend URL to your running local or remote server.
4. Enter the admin token.
5. Configure `CORS_ORIGINS` on the backend if the page is not served from localhost.

The static page does not store accounts by itself. API keys, cookies, and channel data stay on the backend.

## Build Windows Portable App

Run:

```powershell
.\build_exe.bat
```

The script discovers Conda dynamically. If a `happy` environment exists, it uses it; otherwise it falls back to `base`, then to `python` on PATH. It runs the test suite and the contract check before packaging and refuses to build if either fails.

Output:

```text
dist\llm-pool\llm-pool.exe
dist\llm-pool-windows-x64.zip
dist\llm-pool-windows-x64.zip.sha256
llm-pool-launch.bat
```

The GitHub Actions workflow runs the test suite, lint, and contract check, builds the package, launches the frozen exe, checks `/health`, verifies the dashboard HTML, local admin-token bootstrap and `/admin/diagnostics`, confirms a tool-carrying Anthropic request reaches the router instead of crashing, zips the build, and uploads SHA256 files on releases. The same job matrix runs on `windows-2025` (x64) and `windows-11-arm` (native ARM64, best-effort); a failed ARM64 leg never blocks the x64 release.

`Dependency Probe` runs weekly and on demand with the current unlocked `requirements.txt` set. It validates imports, web-channel contracts, `pip check`, and Playwright Chromium installability so upstream dependency drift is visible before a release rebuild.

## Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest
python -m ruff check llm_pool main.py tests scripts
```

The suite runs without network access or provider SDKs: fake backends stand in for the providers, so it exercises the dialect adapters, router scoring, failover, SSE encoders, rate limiting, redaction, and the at-rest secret handling directly. Both GitHub workflows run it, plus the `ruff` correctness lint configured in `ruff.toml`.

## Files

- `main.py`: entry point (PyInstaller target, `uvicorn main:app`) and compatibility facade over the package.
- `llm_pool/`: the implementation, split by responsibility with dependencies flowing strictly downward:
  - `paths.py`, `envtools.py`: frozen/source path resolution and typed env accessors.
  - `diagnostics.py`: redaction helpers and the sanitized recent-event ring buffer.
  - `secretbox.py`: DPAPI envelopes for at-rest channel secrets.
  - `settings.py`: environment-derived configuration and the non-local-bind startup guard.
  - `store.py`: the live channel list and its atomic persistence to `channels.json`.
  - `canonical.py`: the dialect-neutral request/response model and OpenAI/Anthropic adapters.
  - `security.py`: admin/API auth and the sliding-window rate limiter.
  - `webdrive.py`: the Playwright web-session driver and cookie login helper.
  - `backends.py`: provider backends (official SDKs plus web sessions).
  - `routing.py`: SmartRouter scoring, quota tracking, cooldowns, concurrency permits.
  - `dispatch.py`: failover, stream leases, and both SSE encoders.
  - `webapp.py`: the FastAPI application and routes.
  - `cli.py`: the command-line entry point.
- `dashboard.html`: self-contained dashboard and playground.
- `requirements.txt`: runtime dependencies.
- `requirements-arm64.txt`: runtime dependencies for the native ARM64 build (no `uvicorn[standard]`, no `google-genai`; the file explains why).
- `requirements-dev.txt`: runtime dependencies plus the test tooling.
- `requirements-lock.txt`: resolved dependency lock used for reproducible CI builds when present.
- `tests/`: pytest suite, no network or provider credentials needed.
- `build_exe.bat`: local Windows build script.
- `.github/workflows/build-exe.yml`: release/manual Windows build matrix producing the x64 package and a best-effort native ARM64 package through identical gates.
- `.github/workflows/dependency-probe.yml`: scheduled unlocked dependency drift check.
- `scripts/validate_web_contracts.py`: local/CI contract check for channel mappings, routes, dialect conversion, and dashboard hooks.
- `examples/usage.md`: short integration examples.

## Known Limits

- Web UI selectors may need updates when providers change their pages.
- Web-session channels are text-only: they drive a chat box, so they cannot carry tool calls or images, and streaming is best-effort because it polls rendered output.
- Gemini channels are text-only. Its schema dialect is a strict subset of JSON Schema, so forwarding arbitrary agent tool definitions would fail the whole request rather than degrade.
- DPAPI-encrypted `channels.json` secrets are tied to the current Windows user; copying the folder to another machine or account loses them.
- Default model ids are only starting points. Set `default_model` per channel for whatever your key actually has access to.
