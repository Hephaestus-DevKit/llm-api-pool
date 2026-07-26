"""Filesystem locations that differ between source checkouts and frozen builds."""
from __future__ import annotations

import sys
from pathlib import Path


def get_app_dir() -> Path:
    """Directory for runtime user data (channels.json, .pw_data_*). For exe: next to the .exe (cwd at launch)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path.cwd()


def get_resource_path(name: str) -> Path:
    """Location of bundled static assets (dashboard.html) inside PyInstaller onefile/onedir or source tree.
    Critical for foolproof exe (onedir preferred for fast startup).
    """
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            # onefile: assets extracted to temp
            return Path(sys._MEIPASS) / name
        else:
            # onedir: assets are loose next to the executable
            return Path(sys.executable).parent / name
    # dev/source: repo root, one level above this package
    return Path(__file__).resolve().parent.parent / name


def _is_windows() -> bool:
    return sys.platform.startswith("win")
