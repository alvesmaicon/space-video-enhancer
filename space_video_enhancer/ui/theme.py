from __future__ import annotations

from pathlib import Path

_THEME_PATH = Path(__file__).with_name("theme.qss")


def load_stylesheet() -> str:
    return _THEME_PATH.read_text(encoding="utf-8")
