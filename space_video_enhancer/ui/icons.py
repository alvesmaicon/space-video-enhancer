"""Vector glyphs for the transport buttons, drawn with QPainter (no font or platform dependence).

Icons are rendered at 1x and 2x so they stay crisp on HiDPI displays.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QPolygonF

DEVICE_PIXEL_RATIOS = (1.0, 2.0)
BAR_WIDTH = 0.14
"""Width of the "stop" bars as a fraction of the icon size."""


def glyph_icon(kind: str, color: QColor, size: int = 18) -> QIcon:
    """``kind``: first | previous | play | pause | next | last."""
    icon = QIcon()
    for ratio in DEVICE_PIXEL_RATIOS:
        pixmap = QPixmap(int(size * ratio), int(size * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        _draw_glyph(painter, kind, float(size))
        painter.end()
        icon.addPixmap(pixmap)
    return icon


def _draw_glyph(painter: QPainter, kind: str, size: float) -> None:
    """Shapes live in a box with 20% padding; triangles point left/right, bars are vertical."""
    pad = size * 0.2
    left, right = pad, size - pad
    top, bottom = pad, size - pad
    middle = size / 2.0
    bar = size * BAR_WIDTH

    def triangle(x_from: float, x_to: float) -> QPolygonF:
        return QPolygonF([QPointF(x_from, top), QPointF(x_from, bottom), QPointF(x_to, middle)])

    if kind == "play":
        painter.drawPolygon(triangle(left + size * 0.05, right + size * 0.05))
    elif kind == "pause":
        gap = size * 0.1
        painter.drawRoundedRect(QRectF(middle - gap - bar, top, bar, bottom - top), 1.5, 1.5)
        painter.drawRoundedRect(QRectF(middle + gap, top, bar, bottom - top), 1.5, 1.5)
    elif kind == "previous":
        painter.drawPolygon(triangle(right, left))
    elif kind == "next":
        painter.drawPolygon(triangle(left, right))
    elif kind == "first":
        painter.drawRoundedRect(QRectF(left, top, bar, bottom - top), 1.5, 1.5)
        painter.drawPolygon(triangle(right, left + bar + size * 0.06))
    elif kind == "last":
        painter.drawRoundedRect(QRectF(right - bar, top, bar, bottom - top), 1.5, 1.5)
        painter.drawPolygon(triangle(left, right - bar - size * 0.06))
    else:
        raise ValueError(f"unknown glyph: {kind}")
