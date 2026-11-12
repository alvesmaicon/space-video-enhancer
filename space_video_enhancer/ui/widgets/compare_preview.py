"""Before/after preview with a draggable split (or two panels), selection overlays and an empty-state CTA.

Four ways to compare (:class:`CompareMode`): a vertical divider dragged left/right, a horizontal one
dragged up/down, the two frames side by side, or one above the other. The overlays, the HUD and
the hit-testing always refer to the *after* rectangle; the motion vectors are drawn on the *before*
side, wherever it is.

Two sizing modes: *fit* (default) fills whatever the scroll area gives it and scales the image to
fit; *zoom* sizes itself so the output frame shows at ``zoom_factor`` logical pixels per output
pixel (1.0 = 1:1) — per panel, so side by side is twice as wide — and lets the scroll area pan.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, QSizeF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
    QResizeEvent,
)
from PySide6.QtWidgets import QPushButton, QSizePolicy, QWidget

from space_video_enhancer.pipeline.perspective import order_quad, quad_grid_lines
from space_video_enhancer.pipeline.preview import MotionOverlay
from space_video_enhancer.ui import palette
from space_video_enhancer.ui.palette import qcolor


class CompareMode(str, Enum):
    SPLIT_VERTICAL = "split_vertical"
    """One image, a vertical divider: before on the left, after on the right."""
    SPLIT_HORIZONTAL = "split_horizontal"
    """One image, a horizontal divider: before above, after below."""
    SIDE_BY_SIDE = "side_by_side"
    STACKED = "stacked"

    @property
    def is_split(self) -> bool:
        return self in (CompareMode.SPLIT_VERTICAL, CompareMode.SPLIT_HORIZONTAL)


PANEL_GAP = 10


@dataclass(frozen=True)
class PanelLayout:
    """Where the two frames go: the panel each one is clipped to, and the rectangle its whole image is drawn in.

    In the split modes all four are the same rectangle. With two panels at a fixed zoom the image
    rectangles are larger than the panels and sit at the same offset in both, so the panels show the
    same region and pan together.
    """

    before_panel: QRectF
    after_panel: QRectF
    before_image: QRectF
    after_image: QRectF


def _axis_offset(panel_start: float, panel_length: float, image_length: float, center: float) -> float:
    """Start of an image of ``image_length`` inside a panel: centred when it fits, else placed so the normalized
    ``center`` of the image sits mid-panel, clamped so the panel is always covered."""
    if image_length <= panel_length:
        return panel_start + (panel_length - image_length) / 2.0
    start = panel_start + panel_length / 2.0 - center * image_length
    return min(panel_start, max(panel_start + panel_length - image_length, start))


"""Pixels between the two panels in side-by-side / stacked modes."""
HANDLE_SIZE = 10.0
MIN_SELECTION_SIZE = 0.08
IMAGE_MARGIN = 18
MIN_WIDGET_SIZE = QSize(320, 200)
MAX_WIDGET_SIDE = 16_000
DEFAULT_REFERENCE_SIZE = QSize(1280, 720)
MIN_ZOOM = 0.1
MAX_ZOOM = 8.0
PLACEHOLDER_TEXT = "Abra um vídeo para comparar antes e depois."
OPEN_BUTTON_TEXT = "Abrir vídeo…"
BACKGROUND_COLOR = qcolor(palette.VIEWER)
PLACEHOLDER_TEXT_COLOR = qcolor(palette.TEXT_MUTED)
SPLIT_LINE_COLOR = QColor("#ffffff")
CAPTION_BACKGROUND = QColor(17, 24, 39, 165)
CAPTION_TEXT_COLOR = QColor("#f9fafb")
PILL_PADDING_X = 7
PILL_PADDING_Y = 2
DIM_COLOR = QColor(17, 24, 39, 120)
HUD_BACKGROUND = QColor(17, 24, 39, 178)
MOTION_INLIER_COLOR = QColor(45, 212, 191)
MOTION_OUTLIER_COLOR = QColor(251, 146, 60)
MOTION_LOCK_COLOR = QColor(74, 222, 128)
MOTION_VECTOR_GAIN = 3.0
"""Inter-frame vectors are a few pixels long; drawn this many times longer so the direction reads."""
MOTION_POINT_RADIUS = 2.5
HUD_TEXT_COLOR = QColor("#e5e7eb")
HUD_PADDING = 8
HUD_LINE_HEIGHT = 15
HUD_FONT_SIZE = 10

_ANCHOR_CORNER = {
    "top_left": QRectF.bottomRight,
    "top_right": QRectF.bottomLeft,
    "bottom_left": QRectF.topRight,
    "bottom_right": QRectF.topLeft,
}


def fit_rect(image_size: QSize, bounds: QRectF) -> QRectF:
    """``image_size`` scaled to fit ``bounds`` keeping its aspect ratio, centred; empty when either is empty."""
    if image_size.width() <= 0 or image_size.height() <= 0 or bounds.width() <= 0 or bounds.height() <= 0:
        return QRectF()
    scale = min(bounds.width() / image_size.width(), bounds.height() / image_size.height())
    width = image_size.width() * scale
    height = image_size.height() * scale
    left = bounds.left() + (bounds.width() - width) / 2.0
    top = bounds.top() + (bounds.height() - height) / 2.0
    return QRectF(left, top, width, height)


def clamp_rect_within_unit(rect: QRectF) -> QRectF:
    x = min(max(0.0, rect.x()), 1.0 - rect.width())
    y = min(max(0.0, rect.y()), 1.0 - rect.height())
    return QRectF(x, y, rect.width(), rect.height())


def resize_rect_from_handle(pointer_norm: QPointF, anchor: QPointF, aspect_ratio: float | None) -> QRectF:
    """Rectangle spanned by ``anchor`` and the pointer, optionally constrained to ``aspect_ratio``."""
    dx = pointer_norm.x() - anchor.x()
    dy = pointer_norm.y() - anchor.y()
    sign_x = 1.0 if dx >= 0 else -1.0
    sign_y = 1.0 if dy >= 0 else -1.0
    width = max(MIN_SELECTION_SIZE, abs(dx))
    height = max(MIN_SELECTION_SIZE, abs(dy))

    if aspect_ratio is not None:
        if width / max(height, 1e-6) > aspect_ratio:
            width = height * aspect_ratio
        else:
            height = width / aspect_ratio

        max_width = anchor.x() if sign_x < 0 else (1.0 - anchor.x())
        max_height = anchor.y() if sign_y < 0 else (1.0 - anchor.y())
        width = min(width, max_width)
        height = min(height, max_height)
        if width / max(height, 1e-6) > aspect_ratio:
            width = height * aspect_ratio
        else:
            height = width / aspect_ratio

    x2 = anchor.x() + (sign_x * width)
    y2 = anchor.y() + (sign_y * height)
    rect = QRectF(min(anchor.x(), x2), min(anchor.y(), y2), abs(x2 - anchor.x()), abs(y2 - anchor.y()))
    return clamp_rect_within_unit(rect)


def _handles(rect: QRectF) -> dict[str, QRectF]:
    half = HANDLE_SIZE / 2.0
    return {
        "top_left": QRectF(rect.left() - half, rect.top() - half, HANDLE_SIZE, HANDLE_SIZE),
        "top_right": QRectF(rect.right() - half, rect.top() - half, HANDLE_SIZE, HANDLE_SIZE),
        "bottom_left": QRectF(rect.left() - half, rect.bottom() - half, HANDLE_SIZE, HANDLE_SIZE),
        "bottom_right": QRectF(rect.right() - half, rect.bottom() - half, HANDLE_SIZE, HANDLE_SIZE),
    }


def _cursor_for_handle(handle: str) -> Qt.CursorShape:
    if handle in {"top_left", "bottom_right"}:
        return Qt.CursorShape.SizeFDiagCursor
    return Qt.CursorShape.SizeBDiagCursor


@dataclass(frozen=True)
class OverlayStyle:
    outline_active: QColor
    outline_idle: QColor
    handle: QColor
    fill: QColor | None = None
    label: str | None = None
    label_color: QColor | None = None
    dim_outside: bool = False
    dashed: bool = False


CROP_STYLE = OverlayStyle(
    outline_active=QColor("#f59e0b"),
    outline_idle=QColor(245, 158, 11, 140),
    handle=QColor("#f59e0b"),
    dim_outside=True,
)
SAMPLE_STYLE = OverlayStyle(
    outline_active=QColor("#2563eb"),
    outline_idle=QColor("#60a5fa"),
    handle=QColor("#2563eb"),
    fill=QColor(37, 99, 235, 35),
    label="Amostra de ruído",
    label_color=QColor("#1d4ed8"),
    dashed=True,
)
FOCUS_STYLE = OverlayStyle(
    outline_active=QColor("#16a34a"),
    outline_idle=QColor("#4ade80"),
    handle=QColor("#16a34a"),
    fill=QColor(22, 163, 74, 28),
    label="Ponto focal",
    label_color=QColor("#166534"),
    dashed=True,
)
WHITE_BALANCE_STYLE = OverlayStyle(
    outline_active=QColor("#a21caf"),
    outline_idle=QColor("#e879f9"),
    handle=QColor("#a21caf"),
    fill=QColor(162, 28, 175, 30),
    label="Cinza de referência",
    label_color=QColor("#86198f"),
    dashed=True,
)


class SelectionOverlay:
    """A normalized rectangle the user can move/resize over the preview image."""

    def __init__(self, style: OverlayStyle, rect_norm: QRectF, on_changed: Callable[[QRectF], None]) -> None:
        self.style = style
        self.rect_norm = QRectF(rect_norm)
        self.enabled = False
        self.editing = False
        self.aspect_ratio: float | None = None
        self._on_changed = on_changed
        self._drag_mode: str | None = None
        self._drag_offset = QPointF()
        self._anchor = QPointF()

    @property
    def interactive(self) -> bool:
        return self.enabled and self.editing

    @property
    def dragging(self) -> bool:
        return self._drag_mode is not None

    def set_state(self, enabled: bool, rect_norm: QRectF, editing: bool, aspect_ratio: float | None = None) -> None:
        self.enabled = enabled
        self.rect_norm = QRectF(rect_norm)
        self.editing = editing and enabled
        self.aspect_ratio = aspect_ratio
        if not self.interactive:
            self._drag_mode = None

    def display_rect(self, image_rect: QRectF) -> QRectF:
        if image_rect.width() <= 0 or image_rect.height() <= 0:
            return QRectF()
        return QRectF(
            image_rect.left() + (self.rect_norm.x() * image_rect.width()),
            image_rect.top() + (self.rect_norm.y() * image_rect.height()),
            self.rect_norm.width() * image_rect.width(),
            self.rect_norm.height() * image_rect.height(),
        )

    def hover_cursor(self, position: QPointF, image_rect: QRectF) -> Qt.CursorShape | None:
        rect = self.display_rect(image_rect)
        handle = self._handle_at(position, rect)
        if handle is not None:
            return _cursor_for_handle(handle)
        if rect.contains(position):
            return Qt.CursorShape.SizeAllCursor
        return None

    def begin_drag(self, position: QPointF, image_rect: QRectF) -> Qt.CursorShape | None:
        """Start a move/resize if ``position`` hits this overlay; returns the cursor to show."""
        rect = self.display_rect(image_rect)
        handle = self._handle_at(position, rect)
        if handle is not None:
            self._drag_mode = handle
            self._anchor = QPointF(_ANCHOR_CORNER[handle](self.rect_norm))
            return _cursor_for_handle(handle)
        if rect.contains(position):
            self._drag_mode = "move"
            self._drag_offset = _point_to_normalized(position, image_rect) - self.rect_norm.topLeft()
            return Qt.CursorShape.SizeAllCursor
        return None

    def drag_to(self, position: QPointF, image_rect: QRectF) -> None:
        pointer_norm = _point_to_normalized(position, image_rect)
        if self._drag_mode == "move":
            moved = QRectF(
                pointer_norm.x() - self._drag_offset.x(),
                pointer_norm.y() - self._drag_offset.y(),
                self.rect_norm.width(),
                self.rect_norm.height(),
            )
            self.rect_norm = clamp_rect_within_unit(moved)
        else:
            self.rect_norm = resize_rect_from_handle(pointer_norm, self._anchor, self.aspect_ratio)
        self._on_changed(QRectF(self.rect_norm))

    def end_drag(self) -> None:
        self._drag_mode = None

    def paint(self, painter: QPainter, image_rect: QRectF) -> None:
        rect = self.display_rect(image_rect)
        if rect.isNull() or rect.width() <= 0 or rect.height() <= 0:
            return
        style = self.style

        if style.dim_outside:
            painter.fillRect(
                QRectF(image_rect.left(), image_rect.top(), image_rect.width(), rect.top() - image_rect.top()),
                DIM_COLOR,
            )
            painter.fillRect(
                QRectF(image_rect.left(), rect.bottom(), image_rect.width(), image_rect.bottom() - rect.bottom()),
                DIM_COLOR,
            )
            painter.fillRect(
                QRectF(image_rect.left(), rect.top(), rect.left() - image_rect.left(), rect.height()), DIM_COLOR
            )
            painter.fillRect(
                QRectF(rect.right(), rect.top(), image_rect.right() - rect.right(), rect.height()), DIM_COLOR
            )
        if style.fill is not None:
            painter.fillRect(rect, style.fill)

        outline = style.outline_active if self.interactive else style.outline_idle
        pen_style = Qt.PenStyle.DashLine if style.dashed else Qt.PenStyle.SolidLine
        painter.setPen(QPen(outline, 2, pen_style))
        painter.drawRect(rect)

        if style.label and style.label_color is not None:
            draw_pill(painter, QPointF(rect.left() + 6, rect.top() + 6), style.label, style.label_color)

        if self.interactive:
            for handle_rect in _handles(rect).values():
                painter.fillRect(handle_rect, style.handle)

    @staticmethod
    def _handle_at(position: QPointF, rect: QRectF) -> str | None:
        for handle, handle_rect in _handles(rect).items():
            if handle_rect.contains(position):
                return handle
        return None


PERSPECTIVE_COLOR = QColor("#22d3ee")
PERSPECTIVE_INVALID_COLOR = QColor("#f87171")
PERSPECTIVE_GRID_COLOR = QColor(34, 211, 238, 120)
PERSPECTIVE_LABEL_COLOR = QColor("#0e7490")
PERSPECTIVE_INVALID_LABEL_COLOR = QColor("#b91c1c")
PERSPECTIVE_POINT_RADIUS = 7.0
PERSPECTIVE_HIT_RADIUS = 14.0
"""Logical pixels around a point that grab it (a ring of radius 7 is a small target on a trackpad)."""
PERSPECTIVE_GRID_DIVISIONS = 4


class QuadOverlay:
    """Four points the user drags onto the corners of a plane (perspective correction).

    Same hit-testing protocol as :class:`SelectionOverlay` (``interactive``, ``begin_drag``…), so the
    preview treats it as one more overlay. The points are normalized in the displayed frame and keep
    their identity while dragged; the outline and the grid follow the canonical order the pipeline
    uses (:func:`order_quad`), so two crossed handles never draw a bow tie. Dragging inside the
    quadrilateral moves all four. The points are hollow rings so the pixel under each stays visible.
    """

    def __init__(self, on_changed: Callable[[list[QPointF]], None]) -> None:
        self.points: list[QPointF] = []
        self.enabled = False
        self.editing = False
        self.valid = True
        self._on_changed = on_changed
        self._drag_index: int | None = None
        """The point being dragged, ``-1`` for the whole quadrilateral, ``None`` when idle."""
        self._drag_origin = QPointF()
        self._drag_start: list[QPointF] = []

    @property
    def interactive(self) -> bool:
        return self.enabled and self.editing

    @property
    def dragging(self) -> bool:
        return self._drag_index is not None

    def set_state(self, enabled: bool, points: list[QPointF], editing: bool, valid: bool = True) -> None:
        self.enabled = enabled and len(points) == 4
        if not self.dragging:  # mid-drag the overlay owns the points; the echo from the window is the same
            self.points = [QPointF(point) for point in points]
        self.editing = editing and self.enabled
        self.valid = valid
        if not self.interactive:
            self._drag_index = None

    def display_points(self, image_rect: QRectF) -> list[QPointF]:
        return [_normalized_to_display(point.x(), point.y(), image_rect) for point in self.points]

    def _ordered_display(self, image_rect: QRectF) -> list[QPointF]:
        ordered = order_quad(np.array([[point.x(), point.y()] for point in self.points]))
        return [_normalized_to_display(x, y, image_rect) for x, y in ordered]

    def point_at(self, position: QPointF, image_rect: QRectF) -> int | None:
        """Index of the point under ``position`` (the nearest one when several are close)."""
        best, best_distance = None, PERSPECTIVE_HIT_RADIUS
        for index, point in enumerate(self.display_points(image_rect)):
            distance = math.hypot(point.x() - position.x(), point.y() - position.y())
            if distance <= best_distance:
                best, best_distance = index, distance
        return best

    def _contains(self, position: QPointF, image_rect: QRectF) -> bool:
        return QPolygonF(self._ordered_display(image_rect)).containsPoint(position, Qt.FillRule.OddEvenFill)

    def hover_cursor(self, position: QPointF, image_rect: QRectF) -> Qt.CursorShape | None:
        if len(self.points) != 4:
            return None
        if self.point_at(position, image_rect) is not None:
            return Qt.CursorShape.CrossCursor
        if self._contains(position, image_rect):
            return Qt.CursorShape.SizeAllCursor
        return None

    def begin_drag(self, position: QPointF, image_rect: QRectF) -> Qt.CursorShape | None:
        if len(self.points) != 4:
            return None
        index = self.point_at(position, image_rect)
        if index is None and self._contains(position, image_rect):
            index = -1
        if index is None:
            return None
        self._drag_index = index
        self._drag_origin = _point_to_normalized(position, image_rect)
        self._drag_start = [QPointF(point) for point in self.points]
        return Qt.CursorShape.CrossCursor if index >= 0 else Qt.CursorShape.SizeAllCursor

    def drag_to(self, position: QPointF, image_rect: QRectF) -> None:
        if self._drag_index is None:
            return
        pointer = _point_to_normalized(position, image_rect)  # clamped to the frame
        if self._drag_index >= 0:
            self.points[self._drag_index] = pointer
        else:
            xs = [point.x() for point in self._drag_start]
            ys = [point.y() for point in self._drag_start]
            dx = min(max(pointer.x() - self._drag_origin.x(), -min(xs)), 1.0 - max(xs))
            dy = min(max(pointer.y() - self._drag_origin.y(), -min(ys)), 1.0 - max(ys))
            self.points = [QPointF(point.x() + dx, point.y() + dy) for point in self._drag_start]
        self._on_changed([QPointF(point) for point in self.points])

    def end_drag(self) -> None:
        self._drag_index = None

    def paint(self, painter: QPainter, image_rect: QRectF) -> None:
        if len(self.points) != 4 or image_rect.width() <= 0 or image_rect.height() <= 0:
            return
        color = PERSPECTIVE_COLOR if self.valid else PERSPECTIVE_INVALID_COLOR
        painter.save()
        if self.valid:
            painter.setPen(QPen(PERSPECTIVE_GRID_COLOR, 1))
            normalized = np.array([[point.x(), point.y()] for point in self.points])
            for (x1, y1), (x2, y2) in quad_grid_lines(normalized, PERSPECTIVE_GRID_DIVISIONS):
                painter.drawLine(_normalized_to_display(x1, y1, image_rect), _normalized_to_display(x2, y2, image_rect))
        ordered = self._ordered_display(image_rect)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, 2))
        painter.drawPolygon(QPolygonF(ordered))
        for point in self.display_points(image_rect):
            painter.setPen(QPen(QColor(0, 0, 0, 160), 4))  # dark halo: the ring reads on bright footage too
            painter.drawEllipse(point, PERSPECTIVE_POINT_RADIUS, PERSPECTIVE_POINT_RADIUS)
            painter.setPen(QPen(color, 2))
            painter.drawEllipse(point, PERSPECTIVE_POINT_RADIUS, PERSPECTIVE_POINT_RADIUS)
            painter.drawPoint(point)
        label = "Plano da perspectiva" if self.valid else "Pontos inválidos: forme um quadrilátero"
        background = PERSPECTIVE_LABEL_COLOR if self.valid else PERSPECTIVE_INVALID_LABEL_COLOR
        top = min(ordered, key=lambda point: point.y())
        size = pill_size(painter, label)
        at = QPointF(top.x() - size.width() / 2.0, top.y() - size.height() - PERSPECTIVE_POINT_RADIUS - 6)
        at.setX(min(max(at.x(), image_rect.left() + 4), image_rect.right() - size.width() - 4))
        at.setY(max(at.y(), image_rect.top() + 4))
        draw_pill(painter, at, label, background)
        painter.restore()


def pill_size(painter: QPainter, text: str) -> QSizeF:
    metrics = painter.fontMetrics()
    return QSizeF(metrics.horizontalAdvance(text) + (2 * PILL_PADDING_X), metrics.height() + (2 * PILL_PADDING_Y))


def draw_pill(painter: QPainter, top_left: QPointF, text: str, background: QColor) -> None:
    """Small rounded label that stays readable over any footage."""
    size = pill_size(painter, text)
    box = QRectF(top_left, size)
    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(background)
    painter.drawRoundedRect(box, size.height() / 2.0, size.height() / 2.0)
    painter.setPen(CAPTION_TEXT_COLOR)
    painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
    painter.restore()


def _normalized_to_display(x: float, y: float, image_rect: QRectF) -> QPointF:
    return QPointF(image_rect.left() + x * image_rect.width(), image_rect.top() + y * image_rect.height())


def _point_to_normalized(point: QPointF, image_rect: QRectF) -> QPointF:
    if image_rect.width() <= 0 or image_rect.height() <= 0:
        return QPointF(0.0, 0.0)
    norm_x = (point.x() - image_rect.left()) / image_rect.width()
    norm_y = (point.y() - image_rect.top()) / image_rect.height()
    return QPointF(min(1.0, max(0.0, norm_x)), min(1.0, max(0.0, norm_y)))


class ComparePreview(QWidget):
    cropChanged = Signal(QRectF)
    sampleChanged = Signal(QRectF)
    stabilizationChanged = Signal(QRectF)
    focusRegionChanged = Signal(int, QRectF)
    """One of the object lock's numbered regions was moved/resized: ``(index, rect)``."""
    whiteBalanceChanged = Signal(QRectF)
    perspectiveChanged = Signal(object)
    """The perspective points were dragged: a list of four normalized ``QPointF`` (displayed frame)."""
    openRequested = Signal()
    """The empty-state "Abrir vídeo" button was clicked."""
    panRequested = Signal(int, int)
    """Drag with the hand tool (or the middle button): scroll the view by this many pixels (dx, dy)."""

    def __init__(self) -> None:
        super().__init__()
        self._before: QImage | None = None
        self._after: QImage | None = None
        self._onion: QImage | None = None  # manual stabilization: the reference frame drawn as a ghost
        self._onion_opacity = 0.5
        self._split_ratio = 0.5
        self._compare_mode = CompareMode.SPLIT_VERTICAL
        self._image_rect = QRectF()
        """Where the *after* frame is drawn (the whole image in split modes): overlays and hit-tests live here."""
        self._before_rect = QRectF()
        self._reference_size = QSize(DEFAULT_REFERENCE_SIZE)
        """Output frame size in pixels; zoom 1.0 shows it 1:1 in logical pixels."""
        self._zoom_factor = 1.0
        self._fit_mode = True
        self._smooth_transform_enabled = True
        self._hud_visible = False
        self._hud_lines: list[str] = []
        self._motion: MotionOverlay | None = None
        self._pan_mode = False
        self._pan_anchor: QPointF | None = None
        """Global position of the last pan event (global: the widget moves under the mouse while it scrolls)."""
        self._view_center = QPointF(0.5, 0.5)
        """Two panels at a fixed zoom: the point of the frame (normalized) shown at the centre of each panel."""
        self._panels: PanelLayout | None = None
        """What the last paint laid out."""

        self.crop = SelectionOverlay(CROP_STYLE, QRectF(0.1, 0.1, 0.8, 0.8), self.cropChanged.emit)
        self.sample = SelectionOverlay(SAMPLE_STYLE, QRectF(0.33, 0.33, 0.34, 0.34), self.sampleChanged.emit)
        self.focus = SelectionOverlay(FOCUS_STYLE, QRectF(0.4, 0.3, 0.2, 0.2), self.stabilizationChanged.emit)
        self.focus_regions: list[SelectionOverlay] = []
        """The object lock's numbered regions (see :meth:`set_focus_regions_state`)."""
        self.white_balance = SelectionOverlay(
            WHITE_BALANCE_STYLE, QRectF(0.4, 0.4, 0.2, 0.2), self.whiteBalanceChanged.emit
        )
        self.perspective = QuadOverlay(self.perspectiveChanged.emit)

        self.open_button = QPushButton(OPEN_BUTTON_TEXT, self)
        self.open_button.setObjectName("previewOpenButton")
        self.open_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_button.clicked.connect(self.openRequested.emit)

        self.setMinimumSize(MIN_WIDGET_SIZE)
        self._apply_size_policy()
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # click the preview, then drive it with the keyboard
        self._position_open_button()

    # --- sizing -------------------------------------------------------------------------
    def sizeHint(self) -> QSize:
        if self.expands_to_viewport():
            return QSize(MIN_WIDGET_SIZE)
        width = int(self._reference_size.width() * self._zoom_factor) + (2 * IMAGE_MARGIN)
        height = int(self._reference_size.height() * self._zoom_factor) + (2 * IMAGE_MARGIN)
        return QSize(
            max(MIN_WIDGET_SIZE.width(), min(MAX_WIDGET_SIDE, width)),
            max(MIN_WIDGET_SIZE.height(), min(MAX_WIDGET_SIDE, height)),
        )

    def minimumSizeHint(self) -> QSize:
        return QSize(MIN_WIDGET_SIZE)

    def expands_to_viewport(self) -> bool:
        """True when the widget should fill the scroll area: fit mode, or a two-panel mode at any zoom.

        With two panels the zoom scales the frames *inside* the panels — both stay on screen and pan
        together — instead of growing the widget; a split mode at a fixed zoom does grow, and scrolls.
        """
        return self._fit_mode or not self._compare_mode.is_split

    @property
    def fit_mode(self) -> bool:
        return self._fit_mode

    def set_fit_mode(self, fit: bool) -> None:
        if fit == self._fit_mode:
            return
        self._fit_mode = fit
        self._apply_size_policy()
        self._relayout()

    def set_display_reference_size(self, width: int, height: int) -> None:
        """Size of the output frame (after scale and rotation); defines what 100% means."""
        size = QSize(width, height) if width > 0 and height > 0 else QSize(DEFAULT_REFERENCE_SIZE)
        if size == self._reference_size:
            return
        self._reference_size = size
        self._relayout()

    def set_zoom_factor(self, zoom_factor: float) -> None:
        self._zoom_factor = max(MIN_ZOOM, min(MAX_ZOOM, zoom_factor))
        self._relayout()

    def effective_zoom(self) -> float:
        """Displayed pixels per output pixel; in fit mode this follows the widget size."""
        if not self._fit_mode:
            return self._zoom_factor
        reference_width = self._reference_size.width()
        if reference_width <= 0:
            return 1.0
        if self._image_rect.width() > 0:
            return self._image_rect.width() / reference_width
        return self._layout(self._reference_size).after_image.width() / reference_width

    def set_render_quality(self, smooth_transform_enabled: bool) -> None:
        self._smooth_transform_enabled = smooth_transform_enabled
        self.update()

    def _apply_size_policy(self) -> None:
        policy = QSizePolicy.Policy.Expanding if self.expands_to_viewport() else QSizePolicy.Policy.Fixed
        self.setSizePolicy(policy, policy)

    def _relayout(self) -> None:
        self.updateGeometry()
        if not self.expands_to_viewport():
            self.resize(self.sizeHint())
        self.update()

    # --- state ---------------------------------------------------------------------------
    @property
    def compare_mode(self) -> CompareMode:
        return self._compare_mode

    def set_compare_mode(self, mode: CompareMode) -> None:
        if mode is self._compare_mode:
            return
        self._compare_mode = mode
        self.setCursor(self._idle_cursor())
        self._apply_size_policy()
        self._relayout()

    def before_rect(self) -> QRectF:
        """Where the last paint drew the whole *before* frame (equal to :meth:`after_rect` in the split modes)."""
        return QRectF(self._before_rect)

    def after_rect(self) -> QRectF:
        return QRectF(self._image_rect)

    def before_panel_rect(self) -> QRectF:
        """The area the *before* frame is clipped to (the frame itself when it fits; see :class:`PanelLayout`)."""
        return QRectF(self._panels.before_panel) if self._panels is not None else self.before_rect()

    def after_panel_rect(self) -> QRectF:
        return QRectF(self._panels.after_panel) if self._panels is not None else self.after_rect()

    @property
    def pan_mode(self) -> bool:
        return self._pan_mode

    def set_pan_mode(self, enabled: bool) -> None:
        """Hand tool: left-drag scrolls the zoomed image instead of moving the split or an overlay."""
        self._pan_mode = enabled
        self._pan_anchor = None
        self.setCursor(self._idle_cursor())

    def _idle_cursor(self) -> Qt.CursorShape:
        if self._pan_mode:
            return Qt.CursorShape.OpenHandCursor
        if self._compare_mode is CompareMode.SPLIT_VERTICAL:
            return Qt.CursorShape.SizeHorCursor
        if self._compare_mode is CompareMode.SPLIT_HORIZONTAL:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def set_hud_visible(self, visible: bool) -> None:
        self._hud_visible = visible
        self.update()

    def set_motion_overlay(self, motion: MotionOverlay | None) -> None:
        """Tracked motion vectors (or lock points) drawn over the "Antes" side; ``None`` hides them."""
        self._motion = motion
        self.update()

    def motion_overlay(self) -> MotionOverlay | None:
        return self._motion

    def set_onion_skin(self, image: QImage | None, opacity: float = 0.5) -> None:
        """A reference frame drawn semi-transparent over the corrected frame, to align manual keyframes by eye."""
        self._onion = image
        self._onion_opacity = max(0.0, min(1.0, opacity))
        self.update()

    def has_onion_skin(self) -> bool:
        return self._onion is not None

    def set_hud_lines(self, lines: list[str]) -> None:
        self._hud_lines = list(lines)
        if self._hud_visible:
            self.update()

    def set_images(self, before: QImage | None, after: QImage | None) -> None:
        self._before = before
        self._after = after
        self.open_button.setVisible(before is None or after is None)
        self.update()

    def set_crop_state(
        self, enabled: bool, crop_rect_norm: QRectF, aspect_ratio: float | None, *, editing: bool
    ) -> None:
        """``editing`` is true only while the geometry tab is active; elsewhere the crop is just shown."""
        self.crop.set_state(enabled, crop_rect_norm, editing=editing, aspect_ratio=aspect_ratio)
        if not enabled:
            self.setCursor(self._idle_cursor())
        self.update()

    def set_sample_state(self, enabled: bool, sample_rect_norm: QRectF, editing: bool) -> None:
        self.sample.set_state(enabled, sample_rect_norm, editing)
        self.update()

    def set_stabilization_state(self, enabled: bool, stabilization_rect_norm: QRectF, editing: bool) -> None:
        self.focus.set_state(enabled, stabilization_rect_norm, editing)
        self.update()

    def set_focus_regions_state(self, enabled: bool, rects: list[QRectF], editing: bool) -> None:
        """Show the object lock's numbered regions (one draggable overlay each); an empty list clears them."""
        while len(self.focus_regions) > len(rects):
            self.focus_regions.pop()
        while len(self.focus_regions) < len(rects):
            index = len(self.focus_regions)
            style = replace(FOCUS_STYLE, label=f"Região {index + 1}")
            self.focus_regions.append(SelectionOverlay(style, rects[index], self._region_callback(index)))
        for overlay, rect in zip(self.focus_regions, rects):
            overlay.set_state(enabled, rect, editing)
        self.update()

    def _region_callback(self, index: int) -> Callable[[QRectF], None]:
        def emit(rect: QRectF) -> None:
            self.focusRegionChanged.emit(index, rect)

        return emit

    def set_perspective_state(self, enabled: bool, points: list[QPointF], *, valid: bool) -> None:
        """The four perspective points, shown (and draggable) only while they are being marked."""
        self.perspective.set_state(enabled, points, editing=enabled, valid=valid)
        if not enabled:
            self.setCursor(self._idle_cursor())
        self.update()

    def _active_overlays(self) -> tuple[SelectionOverlay | QuadOverlay, ...]:
        """Hit-testing priority: perspective points, numbered regions, focus region, gray reference, sample, crop."""
        return (self.perspective, *self.focus_regions, self.focus, self.white_balance, self.sample, self.crop)

    def set_white_balance_state(self, enabled: bool, rect_norm: QRectF, editing: bool) -> None:
        self.white_balance.set_state(enabled, rect_norm, editing)
        self.update()

    # --- painting ------------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._smooth_transform_enabled)
        painter.fillRect(self.rect(), BACKGROUND_COLOR)

        if self._before is None or self._after is None:
            painter.setPen(PLACEHOLDER_TEXT_COLOR)
            text_rect = QRectF(0.0, self.height() / 2.0 - 64.0, float(self.width()), 24.0)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, PLACEHOLDER_TEXT)
            return

        layout = self._layout(self._after.size())
        self._panels = layout
        self._before_rect, self._image_rect = layout.before_image, layout.after_image
        if self._onion is not None:
            self._paint_onion(painter)  # manual alignment: the corrected frame under the reference ghost, no split
        elif self._compare_mode.is_split:
            self._paint_split(painter)
        else:
            self._paint_panels(painter, layout)

        painter.save()
        if self._onion is None and not self._compare_mode.is_split:
            painter.setClipRect(layout.after_panel)  # the zoomed frame may extend past its panel
        overlays: tuple[SelectionOverlay | QuadOverlay, ...] = (
            self.crop,
            self.sample,
            self.focus,
            *self.focus_regions,
            self.white_balance,
            self.perspective,
        )
        for overlay in overlays:
            if overlay.enabled:
                overlay.paint(painter, self._image_rect)
        painter.restore()

        if self._hud_visible and self._hud_lines:
            self._draw_hud(painter, layout.after_panel)

    def _paint_onion(self, painter: QPainter) -> None:
        """Manual alignment: the corrected frame full, with the reference ghost laid over it at ``_onion_opacity``."""
        rect = self._image_rect
        assert self._after is not None and self._onion is not None
        painter.drawImage(rect, self._after)
        painter.setOpacity(self._onion_opacity)
        painter.drawImage(rect, self._onion)
        painter.setOpacity(1.0)
        draw_pill(painter, QPointF(rect.left() + 10, rect.top() + 10), "Fantasma (referência)", CAPTION_BACKGROUND)

    def _paint_split(self, painter: QPainter) -> None:
        """One image: the before frame shows on one side of a draggable divider, the after frame on the other."""
        rect = self._image_rect
        assert self._before is not None and self._after is not None
        painter.drawImage(rect, self._after)
        vertical = self._compare_mode is CompareMode.SPLIT_VERTICAL
        if vertical:
            split = rect.left() + rect.width() * self._split_ratio
            before_clip = QRectF(rect.left(), rect.top(), split - rect.left(), rect.height())
        else:
            split = rect.top() + rect.height() * self._split_ratio
            before_clip = QRectF(rect.left(), rect.top(), rect.width(), split - rect.top())
        painter.save()
        painter.setClipRect(before_clip)
        painter.drawImage(rect, self._before)
        if self._motion is not None:
            self._draw_motion(painter, rect)
        painter.restore()

        painter.setPen(QPen(SPLIT_LINE_COLOR, 3))
        if vertical:
            painter.drawLine(int(split), int(rect.top()), int(split), int(rect.bottom()))
        else:
            painter.drawLine(int(rect.left()), int(split), int(rect.right()), int(split))

        draw_pill(painter, QPointF(rect.left() + 10, rect.top() + 10), "Antes", CAPTION_BACKGROUND)
        after_size = pill_size(painter, "Depois")
        if vertical:
            after_at = QPointF(rect.right() - 10 - after_size.width(), rect.top() + 10)
        else:
            after_at = QPointF(rect.left() + 10, rect.bottom() - 10 - after_size.height())
        draw_pill(painter, after_at, "Depois", CAPTION_BACKGROUND)

    def _paint_panels(self, painter: QPainter, layout: PanelLayout) -> None:
        """Two panels (side by side or stacked), each clipped to its panel: the same region of before and after."""
        assert self._before is not None and self._after is not None
        panels = (
            (layout.before_panel, layout.before_image, self._before, "Antes"),
            (layout.after_panel, layout.after_image, self._after, "Depois"),
        )
        for panel, image_rect, image, label in panels:
            painter.save()
            painter.setClipRect(panel)
            painter.drawImage(image_rect, image)
            if label == "Antes" and self._motion is not None:
                self._draw_motion(painter, image_rect)
            painter.restore()
            draw_pill(painter, panel.topLeft() + QPointF(10, 10), label, CAPTION_BACKGROUND)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_open_button()

    # --- mouse ---------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._pan_mode
        ):
            self._pan_anchor = event.globalPosition()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._image_rect.contains(event.position()) and self.after_panel_rect().contains(event.position()):
            for overlay in self._active_overlays():
                if not overlay.interactive:
                    continue
                cursor = overlay.begin_drag(event.position(), self._image_rect)
                if cursor is not None:
                    self.setCursor(cursor)
                    return
        self._update_split_ratio(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._pan_anchor is not None:
            if event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton):
                delta = self._pan_anchor - event.globalPosition()
                self._pan_anchor = event.globalPosition()
                if self._pans_in_place():
                    self._pan_view(delta)
                else:
                    self.panRequested.emit(round(delta.x()), round(delta.y()))
            return
        if self._pan_mode:
            return  # nothing else to hover or drag with the hand tool
        left_pressed = bool(event.buttons() & Qt.MouseButton.LeftButton)
        for overlay in self._active_overlays():
            if overlay.interactive and overlay.dragging and left_pressed:
                overlay.drag_to(event.position(), self._image_rect)
                self.update()
                return

        for overlay in self._active_overlays():
            if overlay.interactive:
                cursor = overlay.hover_cursor(event.position(), self._image_rect)
                self.setCursor(cursor if cursor is not None else self._idle_cursor())
                break

        if left_pressed:
            self._update_split_ratio(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        del event
        if self._pan_anchor is not None:
            self._pan_anchor = None
            self.setCursor(self._idle_cursor())
            return
        for overlay in self._active_overlays():
            overlay.end_drag()
        if any(overlay.enabled for overlay in self._active_overlays()):
            self.setCursor(self._idle_cursor())

    # --- helpers -------------------------------------------------------------------------
    def _layout(self, size: QSize) -> PanelLayout:
        """Where a frame of ``size`` goes in the current compare mode (see :class:`PanelLayout`).

        Two panels sit next to each other (``PANEL_GAP`` apart) and the pair is centred. In fit mode each
        frame is fitted into half of the area; at a fixed zoom the frame is drawn at that zoom and each
        panel is at most half of the area, showing the region around ``_view_center`` — the same in both.
        """
        available = QRectF(self.rect().adjusted(IMAGE_MARGIN, IMAGE_MARGIN, -IMAGE_MARGIN, -IMAGE_MARGIN))
        if size.width() <= 0 or size.height() <= 0:
            return PanelLayout(available, available, available, available)
        if self._compare_mode.is_split:
            rect = fit_rect(size, available)
            return PanelLayout(rect, rect, rect, rect)
        side_by_side = self._compare_mode is CompareMode.SIDE_BY_SIDE
        if side_by_side:
            half = QRectF(0.0, 0.0, max(1.0, (available.width() - PANEL_GAP) / 2.0), max(1.0, available.height()))
        else:
            half = QRectF(0.0, 0.0, max(1.0, available.width()), max(1.0, (available.height() - PANEL_GAP) / 2.0))
        if self._fit_mode:
            image_size = fit_rect(size, half).size()
        else:
            zoomed = QRectF(
                0.0,
                0.0,
                self._reference_size.width() * self._zoom_factor,
                self._reference_size.height() * self._zoom_factor,
            )
            image_size = fit_rect(size, zoomed).size()
        panel_size = QSizeF(min(image_size.width(), half.width()), min(image_size.height(), half.height()))
        if side_by_side:
            pair = QSizeF(2.0 * panel_size.width() + PANEL_GAP, panel_size.height())
            step = QPointF(panel_size.width() + PANEL_GAP, 0.0)
        else:
            pair = QSizeF(panel_size.width(), 2.0 * panel_size.height() + PANEL_GAP)
            step = QPointF(0.0, panel_size.height() + PANEL_GAP)
        origin = available.center() - QPointF(pair.width() / 2.0, pair.height() / 2.0)
        before_panel = QRectF(origin, panel_size)
        before_image = QRectF(
            _axis_offset(before_panel.left(), panel_size.width(), image_size.width(), self._view_center.x()),
            _axis_offset(before_panel.top(), panel_size.height(), image_size.height(), self._view_center.y()),
            image_size.width(),
            image_size.height(),
        )
        return PanelLayout(before_panel, before_panel.translated(step), before_image, before_image.translated(step))

    def _pans_in_place(self) -> bool:
        """Two panels at a fixed zoom move their content; everything else scrolls the widget (``panRequested``)."""
        return not self._fit_mode and not self._compare_mode.is_split

    def _pan_view(self, delta: QPointF) -> None:
        """Move the region both panels show by ``delta`` display pixels (the scroll convention: drag left = +x)."""
        layout = self._panels
        if layout is None or layout.after_image.width() <= 0 or layout.after_image.height() <= 0:
            return
        image, panel = layout.after_image, layout.after_panel
        # From the centre actually shown (after clamping), so a drag never has to undo an out-of-range value first.
        center_x = (panel.center().x() - image.left() + delta.x()) / image.width()
        center_y = (panel.center().y() - image.top() + delta.y()) / image.height()
        self._view_center = QPointF(min(1.0, max(0.0, center_x)), min(1.0, max(0.0, center_y)))
        self.update()

    def _position_open_button(self) -> None:
        hint = self.open_button.sizeHint()
        self.open_button.resize(hint)
        self.open_button.move((self.width() - hint.width()) // 2, self.height() // 2 - 24)

    def _update_split_ratio(self, position: QPointF) -> None:
        """Move the divider to the pointer along the axis of the current split mode (no-op with two panels)."""
        rect = self._image_rect
        if self._compare_mode is CompareMode.SPLIT_VERTICAL and rect.width() > 0:
            ratio = (position.x() - rect.left()) / rect.width()
        elif self._compare_mode is CompareMode.SPLIT_HORIZONTAL and rect.height() > 0:
            ratio = (position.y() - rect.top()) / rect.height()
        else:
            return
        self._split_ratio = min(1.0, max(0.0, ratio))
        self.update()

    def _draw_motion(self, painter: QPainter, image_rect: QRectF) -> None:
        """Vectors from where a point was to where it went (camera inliers teal, others orange); lock points green."""
        motion = self._motion
        if motion is None or motion.count == 0:
            return
        left, top = image_rect.left(), image_rect.top()
        width, height = image_rect.width(), image_rect.height()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if motion.kind == "lock":
            painter.setPen(QPen(MOTION_LOCK_COLOR, 1.5))
            painter.setBrush(MOTION_LOCK_COLOR)
            for x, y in motion.current:
                painter.drawEllipse(
                    QPointF(left + x * width, top + y * height), MOTION_POINT_RADIUS, MOTION_POINT_RADIUS
                )
        else:
            for start, end, inlier in zip(motion.previous, motion.current, motion.inliers):
                color = MOTION_INLIER_COLOR if inlier else MOTION_OUTLIER_COLOR
                tail = QPointF(left + start[0] * width, top + start[1] * height)
                head = QPointF(
                    left + (start[0] + (end[0] - start[0]) * MOTION_VECTOR_GAIN) * width,
                    top + (start[1] + (end[1] - start[1]) * MOTION_VECTOR_GAIN) * height,
                )
                painter.setPen(QPen(color, 1.5))
                painter.drawLine(tail, head)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(head, MOTION_POINT_RADIUS, MOTION_POINT_RADIUS)
        painter.restore()
        label = (
            f"lock: {motion.count} pontos"
            if motion.kind == "lock"
            else f"vetores: {motion.inlier_count} câmera · {motion.count - motion.inlier_count} outros"
        )
        below_caption = image_rect.top() + 10 + pill_size(painter, "Antes").height() + 6  # the HUD owns the bottom
        draw_pill(painter, QPointF(left + 10, below_caption), label, HUD_BACKGROUND)

    def _draw_hud(self, painter: QPainter, anchor: QRectF) -> None:
        """Metrics box (render/decode timings, queue, backend) in the bottom-left corner of ``anchor``."""
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(HUD_FONT_SIZE)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = max(metrics.horizontalAdvance(line) for line in self._hud_lines) + (HUD_PADDING * 2)
        height = (HUD_LINE_HEIGHT * len(self._hud_lines)) + (HUD_PADDING * 2)
        left = anchor.left() + 12
        top = anchor.bottom() - height - 12
        box = QRectF(left, top, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(HUD_BACKGROUND)
        painter.drawRoundedRect(box, 6, 6)
        painter.setPen(HUD_TEXT_COLOR)
        baseline = top + HUD_PADDING + metrics.ascent()
        for line in self._hud_lines:
            painter.drawText(int(left + HUD_PADDING), int(baseline), line)
            baseline += HUD_LINE_HEIGHT
        painter.setFont(QFont())
