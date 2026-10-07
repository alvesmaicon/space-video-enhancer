from __future__ import annotations

from pathlib import Path
from string import Template

from space_video_enhancer.ui.palette import TOKENS

_THEME_PATH = Path(__file__).with_name("theme.qss")
_ICONS_DIR = Path(__file__).with_name("icons")


def load_stylesheet() -> str:
    """The QSS with its ``$TOKEN`` colours filled from :mod:`palette` and ``url(icons/...)`` made absolute.

    Qt has no relative url base. ``substitute`` (not ``safe_``) so a misspelt token fails loudly.
    """
    qss = Template(_THEME_PATH.read_text(encoding="utf-8")).substitute(TOKENS)
    return qss.replace("url(icons/", f"url({_ICONS_DIR.as_posix()}/")
