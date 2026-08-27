"""The application icon, drawn with QPainter so it is crisp at every size and needs no binary asset.

A dark rounded tile with two overlapping frames: the muted "before" frame behind, the teal "after"
frame in front carrying a play triangle — the app in one glance: compare, then play. :func:`app_icon`
gives a :class:`QIcon` with every common size (window, Alt-Tab, dock/taskbar); :func:`write_png`
exports one size to disk for the Linux desktop entry.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
TILE_COLOR = QColor("#0b0f19")
BACK_COLOR = QColor("#64748b")
FRONT_COLOR = QColor("#14b8a6")
PLAY_COLOR = QColor("#f9fafb")


def paint_app_icon(painter: QPainter, size: float) -> None:
    """Draw the icon into the ``size`` x ``size`` square at the painter's origin."""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    radius = size * 0.22
    painter.setBrush(TILE_COLOR)
    painter.drawRoundedRect(QRectF(0, 0, size, size), radius, radius)

    width, height = size * 0.52, size * 0.36
    corner = size * 0.05
    back = QRectF(size * 0.16, size * 0.20, width, height)
    front = QRectF(size * 0.32, size * 0.42, width, height)
    painter.setBrush(BACK_COLOR)
    painter.drawRoundedRect(back, corner, corner)
    painter.setPen(QPen(TILE_COLOR, max(1.0, size * 0.035)))  # a gap in the tile colour separates the frames
    painter.setBrush(FRONT_COLOR)
    painter.drawRoundedRect(front, corner, corner)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(PLAY_COLOR)
    centre = front.center()
    play = QPolygonF(
        [
            QPointF(centre.x() - size * 0.05, centre.y() - size * 0.08),
            QPointF(centre.x() - size * 0.05, centre.y() + size * 0.08),
            QPointF(centre.x() + size * 0.09, centre.y()),
        ]
    )
    painter.drawPolygon(play)


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
