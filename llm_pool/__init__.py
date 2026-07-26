"""LLM API Pool - unified OpenAI + Anthropic gateway over pooled channels.

Package layout (dependencies flow strictly downward):

    paths, envtools          leaf helpers (no package imports)
    diagnostics              redaction + sanitized event ring buffer
    secretbox                DPAPI envelopes for at-rest channel secrets
    settings                 environment-derived configuration + startup guard
    store                    channel list persistence (channels.json)
    canonical                dialect-neutral request/response model + adapters
    security                 admin/API auth, rate limiting
    webdrive                 Playwright web-session driver + login helper
    backends                 provider backends (official SDKs + web sessions)
    routing                  SmartRouter channel selection + quota tracking
    dispatch                 failover, stream leases, SSE encoders
    webapp                   FastAPI application and routes
    cli                      command-line entry point

Mutable runtime state (channel list, router, rate limiter, browser contexts) is
always accessed through its owning module (``store.CHANNELS``, ``routing.router``,
...) so that tests and admin endpoints observe one shared instance.
"""

# Every submodule import runs this first, so .env is loaded before any module
# reads configuration from the environment - same ordering as the old monolith.
from dotenv import load_dotenv

load_dotenv()
