"""Playwright web-session driver for web_* channels, plus the cookie login helper.

Keeps browser contexts alive per channel so the web login session (from email+pass or
cookies) can be reused to actually drive the real chat UI and turn it into an API.
This is "heavy" but the most reliable way to use the web quotas without fragile HTTP
reverse engineering. Selectors may need occasional updates if the sites change their DOM.
"""
from __future__ import annotations

import asyncio
import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from . import settings
from .diagnostics import record_diagnostic_event
from .paths import get_app_dir, get_resource_path

_playwright = None
_web_contexts: Dict[str, Any] = {}  # channel_id -> BrowserContext
_web_context_locks: Dict[str, asyncio.Lock] = {}  # channel_id -> creation lock
_browser_checked = False


async def get_playwright():
    global _playwright
    if _playwright is None:
        from playwright.async_api import async_playwright
        _playwright = await async_playwright().start()
    return _playwright


def cookie_domains_for_channel(ch_type: str) -> List[str]:
    if "gemini" in ch_type:
        return [".google.com", "gemini.google.com"]
    if "claude" in ch_type:
        return [".claude.ai", "claude.ai", ".anthropic.com"]
    if "chatgpt" in ch_type or "codex" in ch_type or "openai" in ch_type:
        return [".chatgpt.com", "chatgpt.com", ".openai.com", "auth.openai.com"]
    return []


async def get_or_create_web_context(ch: Dict[str, Any]):
    """Get or create a persistent browser context injected with the channel's cookies.

    Creation is serialized per channel: two concurrent first requests would otherwise both
    launch a persistent context on the same user-data dir, and Chromium's profile lock makes
    the second launch fail (or worse, corrupts the profile).
    """
    cid = ch["id"]
    existing = _web_contexts.get(cid)
    if existing is not None:
        return existing

    lock = _web_context_locks.setdefault(cid, asyncio.Lock())
    async with lock:
        existing = _web_contexts.get(cid)
        if existing is not None:
            return existing
        return await _create_web_context(ch)


async def _create_web_context(ch: Dict[str, Any]):
    cid = ch["id"]
    cookies = ch.get("config", {}).get("cookies", {})
    if not cookies:
        record_diagnostic_event("warn", "web_channel_missing_cookies", channel_id=cid, type=ch.get("type"), name=ch.get("name"))
        raise HTTPException(400, f"web channel '{ch.get('name')}' has no cookies. Use email+password in /admin to auto-login, or paste cookies.")

    await ensure_browsers_installed()
    p = await get_playwright()
    # Use a per-channel user data dir so logins/cookies can persist across restarts.
    # Use get_app_dir() so it works correctly in PyInstaller onedir (next to exe) and dev.
    user_data = str(get_app_dir() / f".pw_data_{cid}")
    os.makedirs(user_data, exist_ok=True)

    context = await p.chromium.launch_persistent_context(
        user_data_dir=user_data,
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
        ],
        viewport={"width": 1280, "height": 800},
    )

    try:
        # Inject/override cookies (works even if some are already in the profile)
        pw_cookies = []
        domains = cookie_domains_for_channel(ch["type"])
        for name, value in cookies.items():
            if isinstance(value, dict):
                cookie = {**value}
                cookie.setdefault("name", name)
                cookie["value"] = str(cookie.get("value", ""))
                cookie.setdefault("path", "/")
                pw_cookies.append(cookie)
                continue
            for domain in domains:
                pw_cookies.append({
                    "name": name,
                    "value": str(value),
                    "domain": domain,
                    "path": "/",
                })

        if pw_cookies:
            await context.add_cookies(pw_cookies)
    except Exception:
        # Never leave an orphaned browser process behind on a failed setup.
        try:
            await context.close()
        except Exception:
            pass
        raise

    # A crashed or closed browser leaves a context object that fails every later request.
    # BrowserContext.browser is None for a persistent context, so there is nothing to poll;
    # the close event is the reliable signal that the entry must be rebuilt on next use.
    try:
        context.on("close", lambda *_: _web_contexts.pop(cid, None))
    except Exception:
        pass

    _web_contexts[cid] = context
    print(f"[web] Browser context ready for {ch['name']} ({ch['type']})")
    record_diagnostic_event("info", "web_context_ready", channel_id=cid, type=ch.get("type"), name=ch.get("name"))
    return context


async def close_web_context(cid: str) -> None:
    """Free the browser profile; a long-running instance can churn through many channels."""
    context = _web_contexts.pop(cid, None)
    _web_context_locks.pop(cid, None)
    if context is not None:
        try:
            await context.close()
        except Exception:
            pass


async def shutdown() -> None:
    """Close every browser context and stop Playwright. Used by the app's lifespan."""
    global _playwright
    for ctx in list(_web_contexts.values()):
        try:
            await ctx.close()
        except Exception:
            pass
    _web_contexts.clear()
    _web_context_locks.clear()
    if _playwright:
        try:
            await _playwright.stop()
        except Exception:
            pass
        _playwright = None


# Shared last-resort input path. Receives the prompt as a bound argument, so no escaping.
JS_SET_PROMPT = """
(p) => {
    const el = document.querySelector('textarea') ||
               document.querySelector('[contenteditable="true"]') ||
               document.querySelector('[role="textbox"]');
    if (!el) return false;
    el.focus();
    if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
        el.value = p;
    } else {
        el.innerText = p;
    }
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
}
"""

# Everything the browser driver needs to know about a provider, in one place. The streaming
# and non-streaming paths used to keep private copies of this, and they had already drifted:
# the streaming path had weaker input selectors and no error detection at all.
WEB_PROFILES: Dict[str, Dict[str, Any]] = {
    "gemini": {
        "url": "https://gemini.google.com/app",
        "input_locators": [
            'textarea[aria-label*="prompt" i]',
            'div[contenteditable="true"][role="textbox"]',
            'textarea[placeholder*="Message" i]',
            "textarea",
        ],
        "response_selector": '.model-response, [class*="response"], [class*="markdown"], [data-test-id*="model-response"]',
        "error_indicators": ("rate limit", "try again later", "something went wrong"),
    },
    "claude": {
        "url": "https://claude.ai/chat",
        "input_locators": [
            'div[contenteditable="true"]',
            'textarea[placeholder*="Message" i]',
            '[role="textbox"]',
            "textarea",
        ],
        "response_selector": '[data-test-id*="message"], [class*="message-content"], .prose',
        "error_indicators": ("rate limit", "overloaded", "try again"),
    },
    "chatgpt": {
        "url": "https://chatgpt.com/",
        "input_locators": [
            "textarea#prompt-textarea",
            'textarea[placeholder*="Message" i]',
            '[contenteditable="true"]',
            "textarea",
        ],
        "response_selector": '[data-message-author-role="assistant"], [class*="message"], .markdown',
        "error_indicators": ("rate limit", "too many requests", "chatgpt is at capacity"),
    },
}


def web_profile_for(ch_type: str) -> Dict[str, Any]:
    if "gemini" in ch_type:
        return WEB_PROFILES["gemini"]
    if "claude" in ch_type:
        return WEB_PROFILES["claude"]
    return WEB_PROFILES["chatgpt"]  # web_chatgpt and web_codex share the same session


def build_extract_js(selector: str, min_length: int) -> str:
    """Read the newest rendered answer. Walks backwards so a trailing empty container does
    not mask the reply, and skips fragments shorter than min_length."""
    return f"""() => {{
        const nodes = document.querySelectorAll({json.dumps(selector)});
        for (let i = nodes.length - 1; i >= 0; i--) {{
            const text = (nodes[i].innerText || '').trim();
            if (text.length > {min_length}) return text;
        }}
        return '';
    }}"""


async def submit_web_prompt(page, prompt: str, locators: List[str]) -> bool:
    """Type the prompt and send it. Returns False when it fell back to direct DOM assignment."""
    for selector in locators:
        try:
            locator = page.locator(selector).first
            await locator.wait_for(timeout=6000, state="visible")
            await locator.click()
            await locator.fill(prompt)  # more reliable than type() for long prompts
            await page.keyboard.press("Enter")
            return True
        except Exception:
            continue
    # Last resort. The prompt is a bound argument and is never spliced into the JS source,
    # so it must not be escaped for a template literal: that would type the escapes verbatim.
    await page.evaluate(JS_SET_PROMPT, prompt)
    await page.keyboard.press("Enter")
    return False


def raise_if_web_error(ch: dict, text: str, indicators) -> None:
    """A rendered error page is not an answer. Raising lets the router cool this channel down
    and fail over instead of handing the caller an error message as if the model said it."""
    lowered = text.lower()
    if any(indicator in lowered for indicator in indicators):
        raise RuntimeError(f"WEB_RATE_LIMIT:{ch.get('name')}: {text.strip()[:120]}")


def classify_web_failure(ch: dict, error: BaseException) -> RuntimeError:
    message = str(error)
    if message.startswith("WEB_RATE_LIMIT"):
        return error if isinstance(error, RuntimeError) else RuntimeError(message)
    lowered = message.lower()
    if any(marker in lowered for marker in ("rate limit", "overloaded", "capacity", "too many requests")):
        return RuntimeError(f"WEB_RATE_LIMIT:{ch.get('name')}")
    return RuntimeError(f"Web drive failed for {ch.get('name')}: {message[:200]}")


async def drive_web_chat(ch: Dict[str, Any], prompt: str, timeout_ms: Optional[int] = None) -> str:
    """Drive an already-authenticated web chat UI and return the finished reply."""
    timeout_ms = timeout_ms or settings.WEB_RESPONSE_TIMEOUT_MS
    profile = web_profile_for(ch["type"])
    extract_js = build_extract_js(profile["response_selector"], min_length=10)
    context = await get_or_create_web_context(ch)
    page = await context.new_page()

    try:
        await page.goto(profile["url"], wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(800)  # let the app hydrate
        await submit_web_prompt(page, prompt, profile["input_locators"])
        await page.wait_for_timeout(1800)

        deadline = time.monotonic() + timeout_ms / 1000.0
        last_text = ""
        while time.monotonic() < deadline:
            try:
                text = await page.evaluate(extract_js)
            except Exception:
                text = ""  # transient DOM churn mid-render; try again next tick
            # Deliberately outside the try: an error page must propagate, not be swallowed.
            if text and len(text) > 8:
                raise_if_web_error(ch, text, profile["error_indicators"])
                if text != last_text:
                    last_text = text
                    # Give the answer a moment to finish growing before accepting it.
                    await page.wait_for_timeout(900)
                    try:
                        settled = await page.evaluate(extract_js)
                    except Exception:
                        settled = ""
                    if settled and len(settled) > len(last_text):
                        last_text = settled
                    if len(last_text) > 15:
                        break
            await page.wait_for_timeout(600)

        final = last_text.strip()
        if not final:
            raise RuntimeError(
                "no response could be read from the page; the site layout may have changed "
                "or the session may have expired (try pasting fresh cookies)"
            )
        return final

    except HTTPException:
        raise  # e.g. "browser could not be installed" already carries the right status
    except Exception as e:
        raise classify_web_failure(ch, e) from e
    finally:
        try:
            await page.close()
        except Exception:
            pass  # best effort; the context is reused


async def drive_web_chat_stream(ch: dict, prompt: str, interval: float = 0.35,
                                timeout_ms: Optional[int] = None):
    """Best-effort streaming for web sessions: poll the rendered answer and yield the growth.

    Shares WEB_PROFILES with drive_web_chat, so selectors and error detection cannot drift
    between the streaming and non-streaming paths.
    """
    timeout_ms = timeout_ms or settings.WEB_RESPONSE_TIMEOUT_MS
    profile = web_profile_for(ch["type"])
    # min_length 0: partial text should stream as soon as it renders.
    extract_js = build_extract_js(profile["response_selector"], min_length=0)
    context = await get_or_create_web_context(ch)
    page = await context.new_page()
    try:
        await page.goto(profile["url"], wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(800)
        await submit_web_prompt(page, prompt, profile["input_locators"])
        await page.wait_for_timeout(1500)

        seen = ""
        checked_for_errors = False
        idle_polls = 0
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            try:
                current = await page.evaluate(extract_js)
            except Exception:
                current = ""  # transient DOM churn mid-render
            if current and not checked_for_errors and len(current) > 8:
                checked_for_errors = True
                # Outside the try above so a rendered error page propagates to the router.
                raise_if_web_error(ch, current, profile["error_indicators"])
            if current and current != seen:
                # Emit the growth while the answer extends what was already sent. When the
                # page rewrites the answer there is no stable diff, so the rewritten text is
                # emitted in full even though the client already saw a prefix of it.
                delta = current[len(seen):] if current.startswith(seen) else current
                seen = current
                idle_polls = 0
                if delta.strip():
                    yield {"choices": [{"delta": {"content": delta}}]}
            elif seen:
                # The answer stopped growing. There is no completion signal in the DOM, so
                # settling is the only end-of-turn we get; without it every streamed web
                # request would hold its permit, page and connection for the full timeout.
                idle_polls += 1
                if idle_polls >= settings.WEB_STREAM_SETTLE_POLLS:
                    break
            await asyncio.sleep(interval)
    except HTTPException:
        raise
    except Exception as e:
        raise classify_web_failure(ch, e) from e
    finally:
        try:
            await page.close()
        except Exception:
            pass


# ==================== Login helper (Playwright for email+pass -> cookies) ====================

MANUAL_BROWSER_INSTALL_HINT = (
    "Install Chromium once with:  python -m playwright install chromium\n"
    "  (run it from a normal Python install, then restart this app)"
)


def install_chromium_blocking() -> None:
    """Run Playwright's installer in-process.

    A frozen build must not shell out to sys.executable: that is this very app, so
    `<app>.exe -m playwright install` would relaunch the server with junk arguments instead
    of installing anything. Playwright's CLI entry point works in-process in both cases.
    """
    saved_argv = sys.argv
    sys.argv = ["playwright", "install", "chromium"]
    try:
        runpy.run_module("playwright", run_name="__main__", alter_sys=True)
    except SystemExit as exit_signal:  # the CLI exits when it is done
        if exit_signal.code not in (0, None):
            raise RuntimeError(f"playwright install exited with code {exit_signal.code}") from exit_signal
    finally:
        sys.argv = saved_argv


def bundled_browsers_dir() -> Optional[Path]:
    """A build may ship browsers next to the app; prefer them over the user profile."""
    for candidate in (get_resource_path("ms-playwright"), get_app_dir() / "ms-playwright"):
        try:
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate
        except OSError:
            continue
    return None


def use_bundled_browsers_if_present() -> None:
    if os.getenv("PLAYWRIGHT_BROWSERS_PATH"):
        return
    bundled = bundled_browsers_dir()
    if bundled is not None:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(bundled)


async def ensure_browsers_installed():
    """Download Chromium on first use of a web channel, so local setup stays one step."""
    global _browser_checked
    if _browser_checked:
        return True
    use_bundled_browsers_if_present()
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            await browser.close()
        _browser_checked = True
        return True
    except Exception as e:
        msg = str(e).lower()
        if not any(marker in msg for marker in ("executable", "browser", "not found", "playwright")):
            record_diagnostic_event("error", "browser_check_failed", error=str(e))
            raise
        print("\n[LLM Pool] First run for web sessions: downloading Chromium (~150MB, once).")
        print("This can take a minute or two depending on your connection.")
        try:
            await asyncio.to_thread(install_chromium_blocking)
        except Exception as install_err:
            print(f"[LLM Pool] Automatic install failed: {install_err}")
            print(MANUAL_BROWSER_INSTALL_HINT)
            record_diagnostic_event("error", "browser_auto_install_failed", error=str(install_err))
            raise HTTPException(500, "Could not install the browser automatically. " + MANUAL_BROWSER_INSTALL_HINT) from install_err
        print("[LLM Pool] Browser installed. Continuing...")
        _browser_checked = True
        return True


async def extract_cookies_with_playwright(email: str, password: str, provider: str) -> Dict[str, str]:
    """
    Headless login to get session cookies.
    User just inputs email+password in the UI/API.
    This is the "direct account password" flow.
    Risk: May trigger account review / captcha / 2FA. User accepts.
    """
    await ensure_browsers_installed()

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise HTTPException(500, "playwright not installed in env. pip install playwright && playwright install chromium") from exc

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        cookies = {}

        if provider == "web_gemini":
            # Go to gemini, trigger Google login flow
            await page.goto("https://gemini.google.com/")
            # Click sign in if needed
            try:
                await page.get_by_role("button", name="Sign in").click(timeout=5000)
            except Exception:
                pass

            # The flow goes to accounts.google.com
            # Fill email
            await page.fill('input[type="email"]', email, timeout=10000)
            await page.click('button:has-text("Next")')
            await page.wait_for_timeout(1500)

            # Password
            await page.fill('input[type="password"]', password)
            await page.click('button:has-text("Next")')

            # Wait for redirect back to gemini
            await page.wait_for_url("**/gemini.google.com/**", timeout=30000)

            # Extract the important cookies
            all_cookies = await context.cookies()
            for c in all_cookies:
                if c["name"] in ["__Secure-1PSID", "__Secure-1PSIDTS", "__Secure-1PSIDCC"]:
                    cookies[c["name"]] = c["value"]

            if not cookies.get("__Secure-1PSID"):
                # Sometimes needs more wait or "continue" button
                await page.wait_for_timeout(3000)
                all_cookies = await context.cookies()
                for c in all_cookies:
                    if c["name"].startswith("__Secure-1PSID"):
                        cookies[c["name"]] = c["value"]
            if not cookies:
                raise RuntimeError("Login may have failed (2FA, captcha, or UI change). Please login manually in browser and paste cookies instead of using password.")

        elif provider == "web_claude":
            await page.goto("https://claude.ai/login")
            # Modern Claude accounts usually use a magic link or SSO; this only covers the
            # password form. Failures are reported rather than swallowed, otherwise the
            # channel would be created with cookies that cannot authenticate anything.
            login_error = None
            try:
                await page.fill('input[type="email"]', email)
                await page.click('button:has-text("Continue")')
                await page.wait_for_timeout(1500)
                await page.fill('input[type="password"]', password)
                await page.click('button:has-text("Continue")')
                await page.wait_for_url("**/claude.ai/**", timeout=30000)
            except Exception as e:
                login_error = str(e)[:160]
            all_cookies = await context.cookies()
            for c in all_cookies:
                if "claude" in c.get("domain", "") or c["name"] in ["sessionKey", "intercom-session"]:
                    cookies[c["name"]] = c["value"]
            if "sessionKey" not in cookies:
                raise RuntimeError(
                    "Claude login did not produce a session cookie"
                    + (f" ({login_error})" if login_error else "")
                    + ". Log in with a real browser and paste the cookies instead."
                )

        elif provider in ("web_chatgpt", "web_codex"):
            # web_chatgpt and web_codex share the chatgpt.com session but are tracked as
            # separate channels so their quotas can be budgeted independently.
            await page.goto("https://chatgpt.com/")
            try:
                # Common OpenAI login
                await page.get_by_role("button", name="Log in").click(timeout=5000)
            except Exception:
                pass
            await page.fill('input[type="email"]', email)
            await page.click('button:has-text("Continue")')
            await page.wait_for_timeout(1500)
            await page.fill('input[type="password"]', password)
            await page.click('button:has-text("Continue")')
            await page.wait_for_url("**/chatgpt.com/**", timeout=30000)
            # For codex, after login, optionally navigate to account or copilot settings to "activate" the quota
            if provider == "web_codex":
                await page.goto("https://chatgpt.com/#settings", timeout=10000)
                await page.wait_for_timeout(2000)
            all_cookies = await context.cookies()
            for c in all_cookies:
                domain = c.get("domain", "")
                if "openai" in domain or "chatgpt" in domain or c["name"].startswith("_"):
                    cookies[c["name"]] = c["value"]

        await browser.close()
        if not cookies:
            raise RuntimeError(
                "Login produced no session cookies (2FA, captcha, or a changed login page). "
                "Log in with a real browser and paste the cookies instead."
            )
        return cookies
