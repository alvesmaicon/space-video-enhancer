"""The application icon, drawn with QPainter so it is crisp at every size and needs no binary asset.

A dark rounded tile with the app's signature: a frame split down the middle — the muted "before"
half and the teal "after" half — with the round drag handle on the divider. :func:`app_icon` gives
a :class:`QIcon` with every common size (window, Alt-Tab, dock/taskbar); :func:`write_png` exports one
size to disk for the Linux desktop entry.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
TILE_COLOR = QColor("#0b0f19")
BEFORE_COLOR = QColor("#8a6a4f")
AFTER_COLOR = QColor("#14b8a6")
DIVIDER_COLOR = QColor("#f9fafb")


def paint_app_icon(painter: QPainter, size: float) -> None:
    """Draw the icon into the ``size`` x ``size`` square at the painter's origin."""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    radius = size * 0.22
    tile = QRectF(0, 0, size, size)
    painter.setBrush(TILE_COLOR)
    painter.drawRoundedRect(tile, radius, radius)

    margin = size * 0.17
    frame = QRectF(margin, margin * 1.25, size - 2 * margin, size - 2.5 * margin)
    clip = QPainterPath()
    clip.addRoundedRect(frame, size * 0.07, size * 0.07)
    painter.save()
    painter.setClipPath(clip)
    split_x = frame.left() + frame.width() * 0.5
    painter.setBrush(BEFORE_COLOR)
    painter.drawRect(QRectF(frame.left(), frame.top(), split_x - frame.left(), frame.height()))
    painter.setBrush(AFTER_COLOR)
    painter.drawRect(QRectF(split_x, frame.top(), frame.right() - split_x, frame.height()))
    painter.restore()

    line_width = max(1.0, size * 0.045)
    painter.setPen(QPen(DIVIDER_COLOR, line_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
    painter.drawLine(QPointF(split_x, frame.top()), QPointF(split_x, frame.bottom()))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(DIVIDER_COLOR)
    handle = size * 0.075
    painter.drawEllipse(QPointF(split_x, frame.center().y()), handle, handle)


def app_icon_pixmap(size: int) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    paint_app_icon(painter, float(size))
    painter.end()
    return pixmap


def app_icon() -> QIcon:
    """The icon at every size in :data:`ICON_SIZES` (Qt picks the closest and scales)."""
    icon = QIcon()
    for size in ICON_SIZES:
        icon.addPixmap(app_icon_pixmap(size))
    return icon


def write_png(path: Path, size: int = 256) -> Path:
    """Save the icon as a PNG (what a ``.desktop`` entry or a store listing needs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not app_icon_pixmap(size).save(str(path), "PNG"):
        raise OSError(f"could not write {path}")
    return path
