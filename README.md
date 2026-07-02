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
- Smart routing by provider compatibility, health, quota estimate, latency, in-flight load, priority, and cooldown state.
- Self-contained dashboard with account health, quota estimate, latency, in-flight load, and a playground.
- Admin diagnostics export for sanitized runtime, channel, router, browser, and recent event state.
- Portable Windows `--onedir` build for faster startup than PyInstaller onefile extraction.

Official API channels are the recommended production path. Web-session channels are useful for personal quota pooling, but they are inherently more fragile because provider pages, cookies, 2FA, captcha, and browser automation can change.

## Release Packages

Use the package that matches your Windows device:

- `llm-pool-windows-x64.zip`: primary Windows x64 package.
- Each zip is published with a `.sha256` file.

Windows on Arm users should use the x64 package under Windows' built-in x64 emulation. A native ARM64 package is not published yet because the full native dependency stack is not currently installable in the `windows-11-arm` runner with this dependency set; the current probe fails before packaging while building native dependencies such as `cryptography`, and Playwright is therefore not available either.

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
- API requests are rate-limited by client/token/path. Default: `RATE_LIMIT_PER_MINUTE=120`.
- CORS defaults to localhost only. Add exact external origins through `CORS_ORIGINS`.
- `/admin/diagnostics` returns a sanitized JSON snapshot for debugging. It redacts API keys, passwords, cookies, and tokens.
- Set `DEBUG_ERRORS=1` only while debugging packaging issues locally.

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

The script discovers Conda dynamically. If a `happy` environment exists, it uses it; otherwise it falls back to `base`, then to `python` on PATH.

Output:

```text
dist\llm-pool\llm-pool.exe
dist\llm-pool-windows-x64.zip
dist\llm-pool-windows-x64.zip.sha256
llm-pool-launch.bat
```

The GitHub Actions workflow builds the x64 package, launches the frozen exe, checks `/health`, verifies the dashboard HTML, local admin-token bootstrap, and `/admin/diagnostics`, zips the build, and uploads SHA256 files on releases. A separate ARM64 compatibility probe tracks whether native ARM64 packaging has become feasible.

`Dependency Probe` runs weekly and on demand with the current unlocked `requirements.txt` set. It validates imports, web-channel contracts, `pip check`, and Playwright Chromium installability so upstream dependency drift is visible before a release rebuild.

## Files

- `main.py`: FastAPI server, routing, provider backends, Playwright web sessions, security, CLI.
- `dashboard.html`: self-contained dashboard and playground.
- `requirements.txt`: runtime dependencies.
- `requirements-lock.txt`: resolved dependency lock used for reproducible CI builds when present.
- `build_exe.bat`: local Windows build script.
- `.github/workflows/build-exe.yml`: release/manual Windows x64 build workflow plus ARM64 compatibility probe.
- `.github/workflows/dependency-probe.yml`: scheduled unlocked dependency drift check.
- `scripts/validate_web_contracts.py`: local/CI contract check for channel mappings, routes, and dashboard hooks.
- `examples/usage.md`: short integration examples.

## Known Limits

- Web UI selectors may need updates when providers change their pages.
- Web streaming is best-effort because it polls rendered browser output.
- DPAPI-encrypted `channels.json` secrets are tied to the current Windows user.
- Official Gemini support uses Google's current `google-genai` SDK.
- Tool calls, vision, and file inputs are normalized but not fully implemented for every backend.
