"""The stylesheet is a template over the palette: every token resolves and every icon it points at exists."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from space_video_enhancer.ui import palette
from space_video_enhancer.ui.theme import load_stylesheet


def test_every_token_resolves() -> None:
    qss = load_stylesheet()
    assert "$" not in qss
    assert palette.WINDOW in qss and palette.ACCENT in qss


def test_every_icon_url_exists() -> None:
    urls = re.findall(r"url\(([^)]+)\)", load_stylesheet())
    assert urls
    for url in urls:
        assert Path(url).is_file(), url


def test_tokens_are_hex_colours() -> None:
    assert palette.TOKENS
    for name, value in palette.TOKENS.items():
        assert re.fullmatch(r"#[0-9a-f]{6}", value), name
        assert palette.qcolor(value).isValid()
    assert palette.qcolor(palette.ACCENT, 60).alpha() == 60
