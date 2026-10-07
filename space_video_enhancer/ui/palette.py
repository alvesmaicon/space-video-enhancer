"""Colour tokens of the app's chrome: one source for ``theme.qss`` and for the widgets that paint themselves.

A dark, neutral-grey look in the spirit of DaVinci Resolve: graphite panels, a near-black viewer
so the image is judged against dark, low-contrast borders, and one accent (teal, the app icon's)
for what is active or selected. ``theme.qss`` refers to these as ``$NAME``; painted widgets use
:func:`qcolor`. Colours drawn *over the video* (crop/sample overlays, motion vectors, captions)
belong to the image, not the chrome, and stay where they are drawn.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

# Surfaces, darkest to lightest.
VIEWER = "#111214"
"""Behind the image: the preview, the separate monitor."""
WINDOW = "#1b1c1f"
PANEL = "#232428"
"""Tab pane, group boxes, menu and status bars."""
SURFACE = "#2c2d32"
"""Inputs, secondary buttons, unselected tabs, timeline tracks."""
SURFACE_HOVER = "#36383e"
SURFACE_PRESSED = "#3f4148"

BORDER = "#34363c"
BORDER_STRONG = "#4a4d55"

TEXT = "#e4e4e7"
TEXT_MUTED = "#a1a1aa"
TEXT_DISABLED = "#62636b"

ACCENT = "#14b8a6"
"""Lines, handles, indicators: what is active, drawn thin on dark."""
ACCENT_FILL = "#0f766e"
"""Filled controls (primary buttons, the checked tab): dark enough to carry white text."""
ACCENT_FILL_HOVER = "#0d9488"
ON_ACCENT = "#ffffff"

WARNING = "#f59e0b"
DANGER = "#ef4444"
DANGER_FILL = "#b91c1c"
MARKER = "#a78bfa"
"""Keyframes."""

TOKENS: dict[str, str] = {name: value for name, value in globals().items() if name.isupper() and isinstance(value, str)}
"""Every token by name, for ``string.Template`` substitution in the QSS."""


def qcolor(token: str, alpha: int = 255) -> QColor:
    """``token`` (a hex from this module) as a :class:`QColor`, optionally translucent."""
    color = QColor(token)
    color.setAlpha(alpha)
    return color
