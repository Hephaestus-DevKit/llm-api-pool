#!/usr/bin/env python3
"""
LLM API Pool - entry point.

The implementation lives in the ``llm_pool`` package (see its module docstrings for the
layout). This module stays the PyInstaller build target and keeps `uvicorn main:app`
working, and it re-exports the package's public names so existing imports keep resolving.

Compatibility note: the re-export below is *read-only*. Rebinding an attribute on this
module (e.g. ``main.CHANNELS = []``) does not reach the implementation - patch the owning
module instead (``llm_pool.store.CHANNELS``, ``llm_pool.settings.HOST``, ...).
"""

from llm_pool import (
    backends,
    canonical,
    cli,
    dispatch,
    envtools,
    paths,
    routing,
    secretbox,
    security,
    settings,
    store,
    webapp,
    webdrive,
)
# Aliased: binding the plain name here would shadow the facade for `main.diagnostics`,
# which in the monolith was the /admin/diagnostics route handler (webapp.diagnostics).
from llm_pool import diagnostics as _diagnostics
from llm_pool.cli import main
from llm_pool.webapp import app  # noqa: F401  (re-export: `uvicorn main:app` / PyInstaller entry)

# Owning modules for the dynamic re-export, most-referenced first. All names are unique
# across modules apart from stdlib imports (time, json, ...), which are identical objects
# everywhere, so first-match lookup is deterministic.
_FACADE_MODULES = (
    settings,
    store,
    canonical,
    security,
    webdrive,
    backends,
    routing,
    dispatch,
    webapp,
    secretbox,
    _diagnostics,
    paths,
    envtools,
    cli,
)


def __getattr__(name: str):
    for module in _FACADE_MODULES:
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
