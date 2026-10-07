from __future__ import annotations

import hashlib
import tempfile
from functools import lru_cache
from pathlib import Path
from string import Template

from space_video_enhancer.ui.palette import TOKENS

_THEME_PATH = Path(__file__).with_name("theme.qss")
_ICONS_DIR = Path(__file__).with_name("icons")


def _fill(template: str) -> str:
    """``$TOKEN`` colours filled from :mod:`palette`; strict ``substitute`` so a misspelt token fails loudly."""
    return Template(template).substitute(TOKENS)


@lru_cache(maxsize=1)
def themed_icons_dir() -> Path:
    """The SVG icons with their ``$TOKEN`` colours filled, written once to a temp dir named by their content.

    QSS ``url()`` only takes files, so the coloured icons have to exist on disk. Same palette, same
    directory: later runs reuse it, and a palette change lands in a new one.
    """
    rendered = {path.name: _fill(path.read_text(encoding="utf-8")) for path in sorted(_ICONS_DIR.glob("*.svg"))}
    digest = hashlib.sha1("".join(f"{name}\0{svg}" for name, svg in rendered.items()).encode()).hexdigest()[:12]
    target = Path(tempfile.gettempdir()) / f"sve-theme-icons-{digest}"
    target.mkdir(exist_ok=True)
    for name, svg in rendered.items():
        path = target / name
        if not path.is_file() or path.read_text(encoding="utf-8") != svg:
            path.write_text(svg, encoding="utf-8")
    return target


def load_stylesheet() -> str:
    """The QSS with its colours filled from :mod:`palette` and ``url(icons/...)`` pointing at the themed icons."""
    return _fill(_THEME_PATH.read_text(encoding="utf-8")).replace("url(icons/", f"url({themed_icons_dir().as_posix()}/")
