from __future__ import annotations

from pathlib import Path

_THEME_PATH = Path(__file__).with_name("theme.qss")
_ICONS_DIR = Path(__file__).with_name("icons")


def load_stylesheet() -> str:
    """The QSS with ``url(icons/...)`` resolved to absolute paths (Qt has no relative url base)."""
    return _THEME_PATH.read_text(encoding="utf-8").replace("url(icons/", f"url({_ICONS_DIR.as_posix()}/")
