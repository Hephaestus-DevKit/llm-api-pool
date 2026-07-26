"""Command-line entry point for the desktop/server app."""
from __future__ import annotations

import argparse
import threading
import time
import webbrowser

from . import settings, webdrive
from .diagnostics import record_diagnostic_event
from .webapp import app, runtime_info


def main():
    parser = argparse.ArgumentParser(description="LLM API Pool - Unified OpenAI/Anthropic proxy with web sessions")
    parser.add_argument("--host", default=settings.HOST, help="Host to bind (default 127.0.0.1 for local)")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Port (default 8080)")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    parser.add_argument("--open-browser-delay", type=int, default=2, help="Seconds to wait before opening browser")
    parser.add_argument("--install-browser", action="store_true", help="Install Playwright Chromium for web login features (run once)")
    args = parser.parse_args()

    if args.install_browser:
        print("Installing Playwright Chromium for web sessions...")
        webdrive.use_bundled_browsers_if_present()
        try:
            webdrive.install_chromium_blocking()
        except Exception as install_err:
            print(f"Install failed: {install_err}")
            print(webdrive.MANUAL_BROWSER_INSTALL_HINT)
            raise SystemExit(1) from install_err
        print("Browser installed. Web-session channels can now be used.")
        return

    settings.HOST = args.host
    settings.PORT = args.port
    settings.validate_startup_security()

    print("=" * 60)
    print("LLM API Pool started")
    print(f"  Dashboard:   http://{settings.HOST}:{settings.PORT}/")
    print(f"  OpenAI API:  http://{settings.HOST}:{settings.PORT}/v1/chat/completions")
    print(f"  Anthropic API: http://{settings.HOST}:{settings.PORT}/v1/messages")
    print(f"  Status:      http://{settings.HOST}:{settings.PORT}/admin/status")
    print(f"  Architecture: {runtime_info()['arch']}")
    if settings.GENERATED_ADMIN_TOKEN:
        print(f"  Local admin token: {settings.ADMIN_TOKEN}")
        print("  The local dashboard receives this token automatically on loopback.")
    print("=" * 60)
    print("Open the dashboard URL in your browser to manage accounts and monitor the pool.")
    record_diagnostic_event("info", "server_starting", runtime=runtime_info())

    if not args.no_open:
        def open_browser():
            time.sleep(args.open_browser_delay)
            try:
                webbrowser.open(f"http://{settings.HOST}:{settings.PORT}/")
            except Exception as e:
                print(f"Could not auto-open browser: {e}")
        threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    # Pass the app object directly (not a "module:app" string). Critical for PyInstaller
    # bundles, where re-importing the entry module by name fails.
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, reload=False, log_level="info")
