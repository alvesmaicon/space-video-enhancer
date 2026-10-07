"""The stylesheet and its icons are templates over the palette: every token resolves, every icon exists."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from space_video_enhancer.ui import palette
from space_video_enhancer.ui.theme import load_stylesheet, themed_icons_dir


def test_every_token_resolves() -> None:
    qss = load_stylesheet()
    assert "$" not in qss
    assert palette.WINDOW in qss and palette.ACCENT in qss


def test_every_icon_url_exists() -> None:
    urls = re.findall(r"url\(([^)]+)\)", load_stylesheet())
    assert urls
    for url in urls:
        assert Path(url).is_file(), url


def test_icons_get_the_palette_colours() -> None:
    icons = themed_icons_dir()
    assert palette.ON_ACCENT in (icons / "check.svg").read_text(encoding="utf-8")
    assert palette.TEXT_MUTED in (icons / "chevron-down.svg").read_text(encoding="utf-8")
    assert not any("$" in path.read_text(encoding="utf-8") for path in icons.glob("*.svg"))


def test_tokens_are_hex_colours() -> None:
    assert palette.TOKENS
    for name, value in palette.TOKENS.items():
        assert re.fullmatch(r"#[0-9a-f]{6}", value), name
        assert palette.qcolor(value).isValid()
    assert palette.qcolor(palette.ACCENT, 60).alpha() == 60
