"""Web-session driver logic that can be exercised without a real browser."""
from __future__ import annotations

import asyncio

import main
import pytest

from llm_pool import webdrive


class FakePage:
    """Enough of the Playwright page surface for the driver's control flow."""

    def __init__(self, responses, locators_visible=True, evaluate_error=None):
        self.responses = list(responses)
        self.locators_visible = locators_visible
        self.evaluate_error = evaluate_error
        self.evaluated: list = []
        self.keys: list = []
        self.filled: list = []
        self.closed = False
        self.goto_url = None

    async def goto(self, url, **_kwargs):
        self.goto_url = url

    async def wait_for_timeout(self, _ms):
        await asyncio.sleep(0)  # yield like the real one, without the wall-clock cost

    def locator(self, selector):
        page = self

        class Locator:
            first = None

            async def wait_for(self, **_kwargs):
                if not page.locators_visible:
                    raise RuntimeError(f"{selector} not visible")

            async def click(self):
                return None

            async def fill(self, value):
                page.filled.append((selector, value))

        loc = Locator()
        loc.first = loc
        return loc

    @property
    def keyboard(self):
        page = self

        class Keyboard:
            async def press(self, key):
                page.keys.append(key)

        return Keyboard()

    async def evaluate(self, script, *args):
        self.evaluated.append((script, args))
        if script is main.JS_SET_PROMPT:
            return True  # input path, not a response read: must not consume the queue
        if self.evaluate_error:
            raise self.evaluate_error
        return self.responses.pop(0) if self.responses else ""

    async def close(self):
        self.closed = True


class FakeContext:
    def __init__(self, page):
        self._page = page

    async def new_page(self):
        return self._page


@pytest.fixture
def drive(monkeypatch):
    def _install(page):
        async def fake_context(_ch):
            return FakeContext(page)
        monkeypatch.setattr(webdrive, "get_or_create_web_context", fake_context)
    return _install


CHANNEL = {"id": "c1", "type": "web_claude", "name": "my-claude", "config": {"cookies": {"a": "b"}}}


# ---------------------------------------------------------------- profiles

def test_every_web_type_resolves_to_a_profile():
    for ch_type in (t for t in main.VALID_CHANNEL_TYPES if t.startswith("web_")):
        profile = main.web_profile_for(ch_type)
        assert profile["url"].startswith("https://")
        assert profile["input_locators"] and profile["response_selector"]
        assert profile["error_indicators"]


def test_chatgpt_and_codex_share_one_profile():
    assert main.web_profile_for("web_codex") is main.web_profile_for("web_chatgpt")


def test_extract_js_is_valid_javascript_with_an_escaped_selector():
    script = main.build_extract_js('a[x="1"], .b', min_length=10)
    assert '"a[x=\\"1\\"], .b"' in script
    assert "length > 10" in script


# ---------------------------------------------------------------- prompt submission

def test_prompt_is_typed_verbatim_without_escaping(drive):
    """Escaping for a template literal used to type the escape characters into the chat box."""
    prompt = r'path C:\temp and `code` and ${x}'
    page = FakePage(responses=["a full answer that is long enough"])
    drive(page)
    asyncio.run(main.drive_web_chat(CHANNEL, prompt))
    assert page.filled[0][1] == prompt


def test_js_fallback_passes_the_prompt_as_a_bound_argument(drive):
    prompt = r"back\slash and `tick`"
    page = FakePage(responses=["a full answer that is long enough"], locators_visible=False)
    drive(page)
    asyncio.run(main.drive_web_chat(CHANNEL, prompt))
    script, args = page.evaluated[0]
    assert script is main.JS_SET_PROMPT
    assert args == (prompt,)          # never interpolated into the source
    assert prompt not in script


def test_prompt_submission_presses_enter(drive):
    page = FakePage(responses=["a full answer that is long enough"])
    drive(page)
    asyncio.run(main.drive_web_chat(CHANNEL, "hi"))
    assert "Enter" in page.keys


# ---------------------------------------------------------------- error handling

def test_a_rendered_rate_limit_page_is_raised_not_returned(drive):
    """It used to be returned as the model's answer, so the channel was never cooled down."""
    page = FakePage(responses=["You are rate limited, try again later"] * 5)
    drive(page)
    with pytest.raises(RuntimeError, match="WEB_RATE_LIMIT"):
        asyncio.run(main.drive_web_chat(CHANNEL, "hi"))


def test_an_empty_page_raises_rather_than_returning_prose(drive):
    page = FakePage(responses=[""] * 20)
    drive(page)
    with pytest.raises(RuntimeError, match="no response could be read"):
        asyncio.run(main.drive_web_chat(CHANNEL, "hi", timeout_ms=50))


def test_the_page_is_always_closed(drive):
    page = FakePage(responses=["You are rate limited"] * 5)
    drive(page)
    with pytest.raises(RuntimeError):
        asyncio.run(main.drive_web_chat(CHANNEL, "hi"))
    assert page.closed


def test_failure_classification():
    rate_limited = main.classify_web_failure(CHANNEL, RuntimeError("Model is overloaded"))
    assert str(rate_limited).startswith("WEB_RATE_LIMIT")
    passthrough = main.classify_web_failure(CHANNEL, RuntimeError("WEB_RATE_LIMIT:x"))
    assert str(passthrough) == "WEB_RATE_LIMIT:x"
    generic = main.classify_web_failure(CHANNEL, RuntimeError("selector timeout"))
    assert "my-claude" in str(generic) and "WEB_RATE_LIMIT" not in str(generic)


def test_raise_if_web_error_is_case_insensitive():
    with pytest.raises(RuntimeError):
        main.raise_if_web_error(CHANNEL, "TOO MANY REQUESTS", ("too many requests",))
    main.raise_if_web_error(CHANNEL, "a normal answer", ("rate limit",))


def test_a_transient_evaluate_failure_does_not_end_the_run(drive):
    class Flaky(FakePage):
        def __init__(self):
            super().__init__(responses=[])
            self.tick = 0

        async def evaluate(self, script, *args):
            self.tick += 1
            if self.tick < 3:
                raise RuntimeError("Execution context was destroyed")
            return "a settled answer that is long enough"

    page = Flaky()
    drive(page)
    assert asyncio.run(main.drive_web_chat(CHANNEL, "hi")) == "a settled answer that is long enough"


# ---------------------------------------------------------------- streaming

def test_stream_yields_only_forward_progress(drive):
    page = FakePage(responses=["Hel", "Hello", "Hello there", "Hello there"])
    drive(page)

    async def collect():
        out = []
        agen = main.drive_web_chat_stream(CHANNEL, "hi", interval=0)
        async for chunk in agen:
            out.append(chunk["choices"][0]["delta"]["content"])
            if len(out) == 3:
                await agen.aclose()
                break
        return out

    assert "".join(collect_result := asyncio.run(collect())) == "Hello there"
    assert collect_result[0] == "Hel"


def test_stream_raises_on_a_rendered_error_page(drive):
    page = FakePage(responses=["We are at capacity, try again"] * 10)
    drive(page)

    async def collect():
        async for _chunk in main.drive_web_chat_stream(CHANNEL, "hi", interval=0):
            pass

    with pytest.raises(RuntimeError, match="WEB_RATE_LIMIT"):
        asyncio.run(collect())


def test_stream_uses_the_same_profile_as_the_blocking_driver(drive):
    page = FakePage(responses=["done enough to stream"] * 3)
    drive(page)

    async def collect():
        agen = main.drive_web_chat_stream(CHANNEL, "hi", interval=0)
        async for _chunk in agen:
            await agen.aclose()
            break

    asyncio.run(collect())
    assert page.goto_url == main.WEB_PROFILES["claude"]["url"]


# ---------------------------------------------------------------- cookie domains

def test_cookie_domains_cover_each_web_provider():
    for ch_type in (t for t in main.VALID_CHANNEL_TYPES if t.startswith("web_")):
        domains = main.cookie_domains_for_channel(ch_type)
        assert domains and all(isinstance(d, str) and d for d in domains), ch_type
    assert ".claude.ai" in main.cookie_domains_for_channel("web_claude")
    assert ".google.com" in main.cookie_domains_for_channel("web_gemini")
    assert ".chatgpt.com" in main.cookie_domains_for_channel("web_codex")
    assert main.cookie_domains_for_channel("official_claude") == [".claude.ai", "claude.ai", ".anthropic.com"]


# ---------------------------------------------------------------- context creation race

def test_concurrent_first_requests_share_one_context(monkeypatch):
    """Two first requests used to both launch a persistent context on the same profile
    dir; Chromium's profile lock makes the second launch fail."""
    launches = []

    class FakeCtx:
        def __init__(self):
            self.closed = False

        async def add_cookies(self, cookies):
            pass

        def on(self, *_args):
            pass

        async def close(self):
            self.closed = True

    class FakeChromium:
        async def launch_persistent_context(self, **_kwargs):
            launches.append(1)
            await asyncio.sleep(0)  # force interleaving of both callers
            return FakeCtx()

    class FakePlaywright:
        chromium = FakeChromium()

    async def fake_ensure():
        return True

    async def fake_get_playwright():
        return FakePlaywright()

    monkeypatch.setattr(webdrive, "ensure_browsers_installed", fake_ensure)
    monkeypatch.setattr(webdrive, "get_playwright", fake_get_playwright)
    monkeypatch.setattr(webdrive, "_web_contexts", {})
    monkeypatch.setattr(webdrive, "_web_context_locks", {})

    async def run_both():
        ch = {"id": "race1", "type": "web_claude", "name": "n", "config": {"cookies": {"a": "b"}}}
        return await asyncio.gather(
            webdrive.get_or_create_web_context(ch),
            webdrive.get_or_create_web_context(ch),
        )

    first, second = asyncio.run(run_both())
    assert first is second
    assert len(launches) == 1


def test_deleting_a_channel_forgets_its_creation_lock(monkeypatch):
    monkeypatch.setattr(webdrive, "_web_contexts", {})
    monkeypatch.setattr(webdrive, "_web_context_locks", {"c9": asyncio.Lock()})
    asyncio.run(webdrive.close_web_context("c9"))
    assert "c9" not in webdrive._web_context_locks
