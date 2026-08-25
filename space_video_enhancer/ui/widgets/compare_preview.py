"""Before/after preview with a draggable split and up to three selection overlays."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

HANDLE_SIZE = 10.0
MIN_SELECTION_SIZE = 0.08
IMAGE_MARGIN = 18
MIN_WIDGET_SIZE = QSize(320, 200)
BACKGROUND_COLOR = QColor("#f3f4f6")
PLACEHOLDER_TEXT_COLOR = QColor("#4b5563")
SPLIT_LINE_COLOR = QColor("#ffffff")
CAPTION_COLOR = QColor("#111827")
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
    always_editable: bool = False
    """Crop is editable whenever visible; sample/focus need an explicit edit mode."""


CROP_STYLE = OverlayStyle(
    outline_active=QColor("#f59e0b"),
    outline_idle=QColor("#f59e0b"),
    handle=QColor("#f59e0b"),
    dim_outside=True,
    always_editable=True,
)
SAMPLE_STYLE = OverlayStyle(
    outline_active=QColor("#2563eb"),
    outline_idle=QColor("#60a5fa"),
    handle=QColor("#2563eb"),
    fill=QColor(37, 99, 235, 35),
    label="Amostra ruido",
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
        return self.enabled and (self.editing or self.style.always_editable)

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
            painter.setPen(style.label_color)
            painter.drawText(int(rect.left() + 8), int(rect.top() + 20), style.label)

        if self.interactive:
            for handle_rect in _handles(rect).values():
                painter.fillRect(handle_rect, style.handle)

    @staticmethod
    def _handle_at(position: QPointF, rect: QRectF) -> str | None:
        for handle, handle_rect in _handles(rect).items():
            if handle_rect.contains(position):
                return handle
        return None


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
    DEFAULT_BASE_SIZE = QSize(960, 540)

    def __init__(self) -> None:
        super().__init__()
        self._before: QImage | None = None
        self._after: QImage | None = None
        self._split_ratio = 0.5
        self._image_rect = QRectF()
        self._base_size = QSize(self.DEFAULT_BASE_SIZE)
        self._zoom_factor = 1.0
        self._smooth_transform_enabled = True
        self._hud_visible = False
        self._hud_lines: list[str] = []

        self.crop = SelectionOverlay(CROP_STYLE, QRectF(0.1, 0.1, 0.8, 0.8), self.cropChanged.emit)
        self.sample = SelectionOverlay(SAMPLE_STYLE, QRectF(0.33, 0.33, 0.34, 0.34), self.sampleChanged.emit)
        self.focus = SelectionOverlay(FOCUS_STYLE, QRectF(0.4, 0.3, 0.2, 0.2), self.stabilizationChanged.emit)
        # Hit-testing priority: focus region, then noise sample, then crop.
        self._overlays = (self.focus, self.sample, self.crop)

        self.setMinimumSize(MIN_WIDGET_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.resize(self.sizeHint())
        self.setCursor(Qt.CursorShape.SizeHorCursor)

    # --- sizing -------------------------------------------------------------------------
    def sizeHint(self) -> QSize:
        width = max(MIN_WIDGET_SIZE.width(), int(self._base_size.width() * self._zoom_factor))
        height = max(MIN_WIDGET_SIZE.height(), int(self._base_size.height() * self._zoom_factor))
        return QSize(width, height)

    def minimumSizeHint(self) -> QSize:
        return QSize(MIN_WIDGET_SIZE)

    def set_display_reference_size(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            self._base_size = QSize(self.DEFAULT_BASE_SIZE)
        else:
            bound_width = self.DEFAULT_BASE_SIZE.width()
            bound_height = self.DEFAULT_BASE_SIZE.height()
            aspect_ratio = width / height
            if aspect_ratio >= bound_width / bound_height:
                target_width = bound_width
                target_height = max(MIN_WIDGET_SIZE.height(), int(target_width / aspect_ratio))
            else:
                target_height = bound_height
                target_width = max(MIN_WIDGET_SIZE.width(), int(target_height * aspect_ratio))
            self._base_size = QSize(target_width, target_height)
        self._relayout()

    def set_zoom_factor(self, zoom_factor: float) -> None:
        self._zoom_factor = max(0.25, min(8.0, zoom_factor))
        self._relayout()

    def set_render_quality(self, smooth_transform_enabled: bool) -> None:
        self._smooth_transform_enabled = smooth_transform_enabled
        self._relayout()

    def _relayout(self) -> None:
        self.updateGeometry()
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
        self.update()

    def set_crop_state(self, enabled: bool, crop_rect_norm: QRectF, aspect_ratio: float | None) -> None:
        self.crop.set_state(enabled, crop_rect_norm, editing=True, aspect_ratio=aspect_ratio)
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
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Selecione um video para comparar antes e depois."
            )
            return

        self._image_rect = self._fit_rect(self._after)
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

        painter.setPen(CAPTION_COLOR)
        painter.drawText(int(self._image_rect.left() + 12), int(self._image_rect.top() + 28), "Antes")
        painter.drawText(
            QRectF(self._image_rect.right() - 92, self._image_rect.top() + 12, 80, 24),
            Qt.AlignmentFlag.AlignRight,
            "Depois",
        )

        for overlay in (self.crop, self.sample, self.focus):
            if overlay.enabled:
                overlay.paint(painter, self._image_rect)

        if self._hud_visible and self._hud_lines:
            self._draw_hud(painter)

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
    def _fit_rect(self, image: QImage) -> QRectF:
        available = self.rect().adjusted(IMAGE_MARGIN, IMAGE_MARGIN, -IMAGE_MARGIN, -IMAGE_MARGIN)
        if image.width() <= 0 or image.height() <= 0:
            return QRectF(available)

        image_ratio = image.width() / image.height()
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
