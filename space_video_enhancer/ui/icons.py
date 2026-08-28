"""Vector glyphs for the transport and timeline-tool buttons, drawn with QPainter (no font or platform dependence).

Icons are rendered at 1x and 2x so they stay crisp on HiDPI displays. The transport glyphs are
filled shapes; the timeline tools (markers, scissors, trash, arrows) are stroked with round caps.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

DEVICE_PIXEL_RATIOS = (1.0, 2.0)
BAR_WIDTH = 0.14
"""Width of the "stop" bars as a fraction of the icon size."""
STROKE_WIDTH = 0.11
"""Line width of the stroked glyphs as a fraction of the icon size."""
FILLED_GLYPHS = frozenset({"first", "previous", "play", "pause", "next", "last"})
STROKED_GLYPHS = frozenset(
    {"mark_in", "mark_out", "remove_range", "split", "trash", "move_left", "move_right", "detach", "hand"}
)


def glyph_icon(kind: str, color: QColor, size: int = 18) -> QIcon:
    """``kind``: one of :data:`FILLED_GLYPHS` (transport) or :data:`STROKED_GLYPHS` (timeline tools)."""
    if kind not in FILLED_GLYPHS and kind not in STROKED_GLYPHS:
        raise ValueError(f"unknown glyph: {kind}")
    icon = QIcon()
    for ratio in DEVICE_PIXEL_RATIOS:
        pixmap = QPixmap(int(size * ratio), int(size * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if kind in FILLED_GLYPHS:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            _draw_filled(painter, kind, float(size))
        else:
            pen = QPen(color, size * STROKE_WIDTH)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            _draw_stroked(painter, kind, float(size), color)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


def _draw_filled(painter: QPainter, kind: str, size: float) -> None:
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


def _draw_stroked(painter: QPainter, kind: str, size: float, color: QColor) -> None:
    """Timeline tools, in the same 20%-padded box: brackets with a pointer, a struck strip, scissors, a bin, arrows."""
    pad = size * 0.16
    left, right = pad, size - pad
    top, bottom = pad, size - pad
    middle = size / 2.0

    if kind in ("mark_in", "mark_out"):
        arm = size * 0.2
        edge = left if kind == "mark_in" else right
        direction = 1.0 if kind == "mark_in" else -1.0
        bracket = QPainterPath(QPointF(edge + direction * arm, top))
        bracket.lineTo(edge, top)
        bracket.lineTo(edge, bottom)
        bracket.lineTo(edge + direction * arm, bottom)
        painter.drawPath(bracket)
        base = edge + direction * (arm + size * 0.14)
        tip = edge + direction * (right - left - size * 0.02)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(
            QPolygonF([QPointF(base, middle - size * 0.17), QPointF(base, middle + size * 0.17), QPointF(tip, middle)])
        )
    elif kind == "remove_range":
        # Two range ends with an X between them: what lies between the marks goes away.
        for edge in (left, right):
            painter.drawLine(QPointF(edge, top + size * 0.06), QPointF(edge, bottom - size * 0.06))
        cross = size * 0.15
        painter.drawLine(QPointF(middle - cross, middle - cross), QPointF(middle + cross, middle + cross))
        painter.drawLine(QPointF(middle - cross, middle + cross), QPointF(middle + cross, middle - cross))
    elif kind == "split":
        pivot = QPointF(middle, middle + size * 0.02)
        painter.drawLine(pivot, QPointF(left + size * 0.04, top))
        painter.drawLine(pivot, QPointF(right - size * 0.04, top))
        radius = size * 0.11
        for side in (-1.0, 1.0):
            centre = QPointF(middle + side * size * 0.17, bottom - radius)
            painter.drawLine(pivot, QPointF(middle + side * size * 0.09, bottom - 2 * radius))
            painter.drawEllipse(centre, radius, radius)
    elif kind == "trash":
        lid = top + size * 0.1
        painter.drawLine(QPointF(left, lid), QPointF(right, lid))
        painter.drawLine(QPointF(middle - size * 0.11, top), QPointF(middle + size * 0.11, top))
        inset = size * 0.08
        body = QRectF(left + inset, lid + size * 0.08, right - left - 2 * inset, bottom - lid - size * 0.08)
        painter.drawRoundedRect(body, 2.0, 2.0)
        for offset in (-size * 0.1, size * 0.1):
            painter.drawLine(
                QPointF(middle + offset, body.top() + size * 0.1), QPointF(middle + offset, body.bottom() - size * 0.1)
            )
    elif kind == "detach":
        # A window with an arrow leaving its top-right corner: open this somewhere else.
        gap = size * 0.3
        frame = QPainterPath(QPointF(right - gap, top))
        frame.lineTo(left, top)
        frame.lineTo(left, bottom)
        frame.lineTo(right, bottom)
        frame.lineTo(right, top + gap)
        painter.drawPath(frame)
        painter.drawLine(QPointF(middle, middle), QPointF(right, top))
        painter.drawLine(QPointF(right - size * 0.28, top), QPointF(right, top))
        painter.drawLine(QPointF(right, top), QPointF(right, top + size * 0.28))
    elif kind == "hand":
        # An open hand: palm, four fingers with round tips, thumb out to the left.
        palm = QRectF(left + size * 0.2, middle - size * 0.02, size * 0.42, size * 0.36)
        painter.drawRoundedRect(palm, size * 0.1, size * 0.1)
        for index, (offset, tip) in enumerate(((0.06, 0.22), (0.17, 0.12), (0.28, 0.14), (0.38, 0.26))):
            del index
            x = palm.left() + size * offset + size * 0.01
            painter.drawLine(QPointF(x, palm.top()), QPointF(x, top + size * tip))
        painter.drawLine(QPointF(palm.left(), palm.center().y()), QPointF(left, middle - size * 0.16))
    elif kind in ("move_left", "move_right"):
        head = right if kind == "move_right" else left
        tail = left if kind == "move_right" else right
        direction = 1.0 if kind == "move_right" else -1.0
        painter.drawLine(QPointF(tail, middle), QPointF(head, middle))
        wing = size * 0.22
        painter.drawLine(QPointF(head - direction * wing, middle - wing), QPointF(head, middle))
        painter.drawLine(QPointF(head - direction * wing, middle + wing), QPointF(head, middle))
