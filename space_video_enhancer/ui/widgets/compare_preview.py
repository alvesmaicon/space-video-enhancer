"""Before/after preview with a draggable split, up to three selection overlays and an empty-state CTA.

Two sizing modes: *fit* (default) fills whatever the scroll area gives it and scales the
image to fit; *zoom* sizes itself so the output frame shows at ``zoom_factor`` logical
pixels per output pixel (1.0 = 1:1) and lets the scroll area pan.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
    QResizeEvent,
)
from PySide6.QtWidgets import QPushButton, QSizePolicy, QWidget

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
BACKGROUND_COLOR = QColor("#f3f4f6")
PLACEHOLDER_TEXT_COLOR = QColor("#4b5563")
SPLIT_LINE_COLOR = QColor("#ffffff")
CAPTION_BACKGROUND = QColor(17, 24, 39, 165)
CAPTION_TEXT_COLOR = QColor("#f9fafb")
PILL_PADDING_X = 7
PILL_PADDING_Y = 2
DIM_COLOR = QColor(17, 24, 39, 120)
HUD_BACKGROUND = QColor(17, 24, 39, 178)
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
    openRequested = Signal()
    """The empty-state "Abrir vídeo" button was clicked."""

    def __init__(self) -> None:
        super().__init__()
        self._before: QImage | None = None
        self._after: QImage | None = None
        self._split_ratio = 0.5
        self._image_rect = QRectF()
        self._reference_size = QSize(DEFAULT_REFERENCE_SIZE)
        """Output frame size in pixels; zoom 1.0 shows it 1:1 in logical pixels."""
        self._zoom_factor = 1.0
        self._fit_mode = True
        self._smooth_transform_enabled = True
        self._hud_visible = False
        self._hud_lines: list[str] = []

        self.crop = SelectionOverlay(CROP_STYLE, QRectF(0.1, 0.1, 0.8, 0.8), self.cropChanged.emit)
        self.sample = SelectionOverlay(SAMPLE_STYLE, QRectF(0.33, 0.33, 0.34, 0.34), self.sampleChanged.emit)
        self.focus = SelectionOverlay(FOCUS_STYLE, QRectF(0.4, 0.3, 0.2, 0.2), self.stabilizationChanged.emit)
        # Hit-testing priority: focus region, then noise sample, then crop.
        self._overlays = (self.focus, self.sample, self.crop)

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
        if self._fit_mode:
            return QSize(MIN_WIDGET_SIZE)
        width = int(self._reference_size.width() * self._zoom_factor) + (2 * IMAGE_MARGIN)
        height = int(self._reference_size.height() * self._zoom_factor) + (2 * IMAGE_MARGIN)
        return QSize(
            max(MIN_WIDGET_SIZE.width(), min(MAX_WIDGET_SIDE, width)),
            max(MIN_WIDGET_SIZE.height(), min(MAX_WIDGET_SIDE, height)),
        )

    def minimumSizeHint(self) -> QSize:
        return QSize(MIN_WIDGET_SIZE)

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
        return self._fit_rect_for_size(self._reference_size).width() / reference_width

    def set_render_quality(self, smooth_transform_enabled: bool) -> None:
        self._smooth_transform_enabled = smooth_transform_enabled
        self.update()

    def _apply_size_policy(self) -> None:
        policy = QSizePolicy.Policy.Expanding if self._fit_mode else QSizePolicy.Policy.Fixed
        self.setSizePolicy(policy, policy)

    def _relayout(self) -> None:
        self.updateGeometry()
        if not self._fit_mode:
            self.resize(self.sizeHint())
        self.update()

    # --- state ---------------------------------------------------------------------------
    def set_hud_visible(self, visible: bool) -> None:
        self._hud_visible = visible
        self.update()

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
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.update()

    def set_sample_state(self, enabled: bool, sample_rect_norm: QRectF, editing: bool) -> None:
        self.sample.set_state(enabled, sample_rect_norm, editing)
        self.update()

    def set_stabilization_state(self, enabled: bool, stabilization_rect_norm: QRectF, editing: bool) -> None:
        self.focus.set_state(enabled, stabilization_rect_norm, editing)
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

        self._image_rect = self._fit_rect_for_size(self._after.size())
        painter.drawImage(self._image_rect, self._after)

        split_x = self._image_rect.left() + (self._image_rect.width() * self._split_ratio)
        painter.save()
        painter.setClipRect(
            QRectF(
                self._image_rect.left(),
                self._image_rect.top(),
                split_x - self._image_rect.left(),
                self._image_rect.height(),
            )
        )
        painter.drawImage(self._image_rect, self._before)
        painter.restore()

        painter.setPen(QPen(SPLIT_LINE_COLOR, 3))
        painter.drawLine(int(split_x), int(self._image_rect.top()), int(split_x), int(self._image_rect.bottom()))

        draw_pill(
            painter, QPointF(self._image_rect.left() + 10, self._image_rect.top() + 10), "Antes", CAPTION_BACKGROUND
        )
        after_width = pill_size(painter, "Depois").width()
        draw_pill(
            painter,
            QPointF(self._image_rect.right() - 10 - after_width, self._image_rect.top() + 10),
            "Depois",
            CAPTION_BACKGROUND,
        )

        for overlay in (self.crop, self.sample, self.focus):
            if overlay.enabled:
                overlay.paint(painter, self._image_rect)

        if self._hud_visible and self._hud_lines:
            self._draw_hud(painter)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_open_button()

    # --- mouse ---------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._image_rect.contains(event.position()):
            for overlay in self._overlays:
                if not overlay.interactive:
                    continue
                cursor = overlay.begin_drag(event.position(), self._image_rect)
                if cursor is not None:
                    self.setCursor(cursor)
                    return
        self._update_split_ratio(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        left_pressed = bool(event.buttons() & Qt.MouseButton.LeftButton)
        for overlay in self._overlays:
            if overlay.interactive and overlay.dragging and left_pressed:
                overlay.drag_to(event.position(), self._image_rect)
                self.update()
                return

        for overlay in self._overlays:
            if overlay.interactive:
                cursor = overlay.hover_cursor(event.position(), self._image_rect)
                self.setCursor(cursor if cursor is not None else Qt.CursorShape.SizeHorCursor)
                break

        if left_pressed:
            self._update_split_ratio(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        del event
        for overlay in self._overlays:
            overlay.end_drag()
        if any(overlay.enabled for overlay in self._overlays):
            self.setCursor(Qt.CursorShape.SizeHorCursor)

    # --- helpers -------------------------------------------------------------------------
    def _fit_rect_for_size(self, size: QSize) -> QRectF:
        available = self.rect().adjusted(IMAGE_MARGIN, IMAGE_MARGIN, -IMAGE_MARGIN, -IMAGE_MARGIN)
        if size.width() <= 0 or size.height() <= 0:
            return QRectF(available)

        image_ratio = size.width() / size.height()
        available_ratio = available.width() / max(1, available.height())
        if image_ratio > available_ratio:
            target_width = float(available.width())
            target_height = target_width / image_ratio
        else:
            target_height = float(available.height())
            target_width = target_height * image_ratio

        offset_x = available.x() + ((available.width() - target_width) / 2)
        offset_y = available.y() + ((available.height() - target_height) / 2)
        return QRectF(offset_x, offset_y, target_width, target_height)

    def _position_open_button(self) -> None:
        hint = self.open_button.sizeHint()
        self.open_button.resize(hint)
        self.open_button.move((self.width() - hint.width()) // 2, self.height() // 2 - 24)

    def _update_split_ratio(self, mouse_x: float) -> None:
        if self._image_rect.width() <= 0:
            return
        ratio = (mouse_x - self._image_rect.left()) / self._image_rect.width()
        self._split_ratio = min(1.0, max(0.0, ratio))
        self.update()

    def _draw_hud(self, painter: QPainter) -> None:
        """Metrics box (render/decode timings, queue, backend) in the bottom-left corner of the image."""
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(HUD_FONT_SIZE)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = max(metrics.horizontalAdvance(line) for line in self._hud_lines) + (HUD_PADDING * 2)
        height = (HUD_LINE_HEIGHT * len(self._hud_lines)) + (HUD_PADDING * 2)
        left = self._image_rect.left() + 12
        top = self._image_rect.bottom() - height - 12
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
