from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QStyle,
    QStyleOptionSlider,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def frame_to_qimage(frame: np.ndarray) -> QImage:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_frame.shape
    bytes_per_line = width * channels
    return QImage(
        rgb_frame.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()


@dataclass
class FilterSettings:
    denoise_mode: str
    denoise_strength: int
    denoise_color_strength: int
    denoise_template_window: int
    denoise_search_window: int
    sharpen: int
    contrast: int
    brightness: int
    gamma: int
    saturation: int
    temperature: int
    scale_factor: float


@dataclass(frozen=True)
class ProcessingContext:
    denoise_strength: Optional[float]
    denoise_color_strength: Optional[float]
    denoise_template_window: int
    denoise_search_window: int
    contrast_alpha: float
    brightness_beta: float
    gamma_lookup_table: Optional[np.ndarray]
    saturation: float
    temperature_shift: float
    sharpen_amount: float
    scale_factor: float


@dataclass(frozen=True)
class ImageFilterPreset:
    filter_id: str
    name: str
    description: str
    shadow_lift: float
    highlight_rolloff: float
    local_contrast: float
    balance_strength: float
    warmth_shift: float
    saturation_boost: float
    warm_suppression: float
    cool_suppression: float


class LabeledSlider(QWidget):
    valueChanged = Signal(int)
    resetRequested = Signal()

    def __init__(self, label: str, minimum: int, maximum: int, value: int) -> None:
        super().__init__()
        self._title = QLabel(label)
        self._value_label = QLabel(str(value))
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._reset_button = QToolButton()
        self._reset_button.setAutoRaise(True)
        self._reset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        self._reset_button.setToolTip("Voltar este controle ao preset ativo")
        self._reset_button.clicked.connect(self.resetRequested.emit)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self._handle_value_changed)

        header = QHBoxLayout()
        header.addWidget(self._title)
        header.addStretch(1)
        header.addWidget(self._value_label)
        header.addWidget(self._reset_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addWidget(self.slider)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:
        self.slider.setValue(value)

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self._reset_button.setEnabled(enabled)

    def _handle_value_changed(self, value: int) -> None:
        self._value_label.setText(str(value))
        self.valueChanged.emit(value)


class FrameRangeSlider(QSlider):
    def __init__(self, orientation: Qt.Orientation) -> None:
        super().__init__(orientation)
        self._marker_start = 0
        self._marker_end = 0
        self._show_markers = False

    def set_markers(self, start: int, end: int, show_markers: bool = True) -> None:
        self._marker_start = start
        self._marker_end = end
        self._show_markers = show_markers
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if not self._show_markers or self.maximum() <= self.minimum():
            return

        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        if groove_rect.width() <= 0:
            return

        start_x = self._position_for_value(min(self._marker_start, self._marker_end), groove_rect, option)
        end_x = self._position_for_value(max(self._marker_start, self._marker_end), groove_rect, option)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        highlight_rect = groove_rect.adjusted(0, -2, 0, 2)
        highlight_rect.setLeft(min(start_x, end_x))
        highlight_rect.setRight(max(start_x, end_x))
        painter.fillRect(highlight_rect, QColor(15, 118, 110, 70))

        painter.setPen(QPen(QColor("#0f766e"), 2))
        painter.drawLine(start_x, groove_rect.top() - 6, start_x, groove_rect.bottom() + 6)
        painter.setPen(QPen(QColor("#b45309"), 2))
        painter.drawLine(end_x, groove_rect.top() - 6, end_x, groove_rect.bottom() + 6)

    def _position_for_value(
        self,
        value: int,
        groove_rect,
        option: QStyleOptionSlider,
    ) -> int:
        relative = QStyle.sliderPositionFromValue(
            self.minimum(),
            self.maximum(),
            value,
            max(1, groove_rect.width() - 1),
            option.upsideDown,
        )
        return groove_rect.x() + relative


class ComparePreview(QWidget):
    cropChanged = Signal(object)
    sampleChanged = Signal(object)
    DEFAULT_BASE_SIZE = QSize(960, 540)

    def __init__(self) -> None:
        super().__init__()
        self._before: Optional[QImage] = None
        self._after: Optional[QImage] = None
        self._split_ratio = 0.5
        self._image_rect = QRectF()
        self._crop_enabled = False
        self._crop_rect_norm = QRectF(0.1, 0.1, 0.8, 0.8)
        self._crop_aspect_ratio: Optional[float] = None
        self._crop_drag_mode: Optional[str] = None
        self._crop_drag_offset = QPointF()
        self._crop_anchor_norm = QPointF()
        self._sample_enabled = False
        self._sample_editing = False
        self._sample_rect_norm = QRectF(0.33, 0.33, 0.34, 0.34)
        self._sample_drag_mode: Optional[str] = None
        self._sample_drag_offset = QPointF()
        self._sample_anchor_norm = QPointF()
        self._base_size = QSize(self.DEFAULT_BASE_SIZE)
        self._zoom_factor = 1.0
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.resize(self.sizeHint())
        self.setCursor(Qt.CursorShape.SizeHorCursor)

    def sizeHint(self) -> QSize:  # noqa: N802
        width = max(320, int(self._base_size.width() * self._zoom_factor))
        height = max(200, int(self._base_size.height() * self._zoom_factor))
        return QSize(width, height)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(320, 200)

    def set_display_reference_size(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            self._base_size = QSize(self.DEFAULT_BASE_SIZE)
        else:
            bound_width = self.DEFAULT_BASE_SIZE.width()
            bound_height = self.DEFAULT_BASE_SIZE.height()
            aspect_ratio = width / height
            bound_ratio = bound_width / bound_height
            if aspect_ratio >= bound_ratio:
                target_width = bound_width
                target_height = max(200, int(target_width / aspect_ratio))
            else:
                target_height = bound_height
                target_width = max(320, int(target_height * aspect_ratio))
            self._base_size = QSize(target_width, target_height)

        self.updateGeometry()
        self.resize(self.sizeHint())
        self.update()

    def set_zoom_factor(self, zoom_factor: float) -> None:
        self._zoom_factor = max(0.25, min(4.0, zoom_factor))
        self.updateGeometry()
        self.resize(self.sizeHint())
        self.update()

    def set_images(self, before: Optional[QImage], after: Optional[QImage]) -> None:
        self._before = before
        self._after = after
        self.update()

    def set_crop_state(
        self,
        enabled: bool,
        crop_rect_norm: QRectF,
        aspect_ratio: Optional[float],
    ) -> None:
        self._crop_enabled = enabled
        self._crop_rect_norm = QRectF(crop_rect_norm)
        self._crop_aspect_ratio = aspect_ratio
        if not enabled:
            self._crop_drag_mode = None
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.update()

    def set_sample_state(
        self,
        enabled: bool,
        sample_rect_norm: QRectF,
        editing: bool,
    ) -> None:
        self._sample_enabled = enabled
        self._sample_rect_norm = QRectF(sample_rect_norm)
        self._sample_editing = editing and enabled
        if not self._sample_editing:
            self._sample_drag_mode = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#f3f4f6"))

        if self._before is None or self._after is None:
            painter.setPen(QColor("#4b5563"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Selecione um video para comparar antes e depois.",
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

        painter.setPen(QPen(QColor("#ffffff"), 3))
        painter.drawLine(
            int(split_x),
            int(self._image_rect.top()),
            int(split_x),
            int(self._image_rect.bottom()),
        )

        painter.setPen(QColor("#111827"))
        painter.drawText(
            int(self._image_rect.left() + 12),
            int(self._image_rect.top() + 28),
            "Antes",
        )
        painter.drawText(
            QRectF(
                self._image_rect.right() - 92,
                self._image_rect.top() + 12,
                80,
                24,
            ),
            Qt.AlignmentFlag.AlignRight,
            "Depois",
        )

        if self._crop_enabled:
            self._draw_crop_overlay(painter)
        if self._sample_enabled:
            self._draw_sample_overlay(painter)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._image_rect.contains(event.position()):
            if self._sample_editing:
                sample_rect = self._display_rect(self._sample_rect_norm)
                handle = self._selection_handle_at(event.position(), sample_rect)
                if handle is not None:
                    self._sample_drag_mode = handle
                    self._sample_anchor_norm = self._anchor_for_handle(handle, self._sample_rect_norm)
                    self.setCursor(self._cursor_for_handle(handle))
                    return
                if sample_rect.contains(event.position()):
                    self._sample_drag_mode = "move"
                    pointer_norm = self._point_to_normalized(event.position())
                    self._sample_drag_offset = pointer_norm - self._sample_rect_norm.topLeft()
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    return

            if self._crop_enabled:
                crop_rect = self._display_crop_rect()
                handle = self._selection_handle_at(event.position(), crop_rect)
                if handle is not None:
                    self._crop_drag_mode = handle
                    self._crop_anchor_norm = self._anchor_for_handle(handle, self._crop_rect_norm)
                    self.setCursor(self._cursor_for_handle(handle))
                    return
                if crop_rect.contains(event.position()):
                    self._crop_drag_mode = "move"
                    pointer_norm = self._point_to_normalized(event.position())
                    self._crop_drag_offset = pointer_norm - self._crop_rect_norm.topLeft()
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    return
        if event.button() == Qt.MouseButton.LeftButton:
            self._update_split_ratio(event.position().x())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._sample_editing and self._sample_drag_mode is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self._update_sample_drag(event.position())
            return

        if self._crop_enabled and self._crop_drag_mode is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self._update_crop_drag(event.position())
            return

        if self._sample_editing:
            sample_rect = self._display_rect(self._sample_rect_norm)
            handle = self._selection_handle_at(event.position(), sample_rect)
            if handle is not None:
                self.setCursor(self._cursor_for_handle(handle))
            elif sample_rect.contains(event.position()):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif self._crop_enabled:
            crop_rect = self._display_crop_rect()
            handle = self._selection_handle_at(event.position(), crop_rect)
            if handle is not None:
                self.setCursor(self._cursor_for_handle(handle))
            elif crop_rect.contains(event.position()):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeHorCursor)

        if event.buttons() & Qt.MouseButton.LeftButton:
            self._update_split_ratio(event.position().x())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        del event
        self._crop_drag_mode = None
        self._sample_drag_mode = None
        if self._sample_enabled or self._crop_enabled:
            self.setCursor(Qt.CursorShape.SizeHorCursor)

    def _fit_rect(self, image: QImage) -> QRectF:
        margin = 18
        available = self.rect().adjusted(margin, margin, -margin, -margin)
        if image.width() <= 0 or image.height() <= 0:
            return QRectF(available)

        image_ratio = image.width() / image.height()
        available_ratio = available.width() / max(1, available.height())

        if image_ratio > available_ratio:
            target_width = available.width()
            target_height = target_width / image_ratio
        else:
            target_height = available.height()
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

    def _draw_crop_overlay(self, painter: QPainter) -> None:
        crop_rect = self._display_crop_rect()
        if crop_rect.isNull() or crop_rect.width() <= 0 or crop_rect.height() <= 0:
            return

        dim_color = QColor(17, 24, 39, 120)
        painter.fillRect(
            QRectF(self._image_rect.left(), self._image_rect.top(), self._image_rect.width(), crop_rect.top() - self._image_rect.top()),
            dim_color,
        )
        painter.fillRect(
            QRectF(self._image_rect.left(), crop_rect.bottom(), self._image_rect.width(), self._image_rect.bottom() - crop_rect.bottom()),
            dim_color,
        )
        painter.fillRect(
            QRectF(self._image_rect.left(), crop_rect.top(), crop_rect.left() - self._image_rect.left(), crop_rect.height()),
            dim_color,
        )
        painter.fillRect(
            QRectF(crop_rect.right(), crop_rect.top(), self._image_rect.right() - crop_rect.right(), crop_rect.height()),
            dim_color,
        )

        painter.setPen(QPen(QColor("#f59e0b"), 2))
        painter.drawRect(crop_rect)

        for handle_rect in self._selection_handles(crop_rect).values():
            painter.fillRect(handle_rect, QColor("#f59e0b"))

    def _draw_sample_overlay(self, painter: QPainter) -> None:
        sample_rect = self._display_rect(self._sample_rect_norm)
        if sample_rect.isNull() or sample_rect.width() <= 0 or sample_rect.height() <= 0:
            return

        outline_color = QColor("#2563eb") if self._sample_editing else QColor("#60a5fa")
        fill_color = QColor(37, 99, 235, 35)
        painter.fillRect(sample_rect, fill_color)
        painter.setPen(QPen(outline_color, 2, Qt.PenStyle.DashLine))
        painter.drawRect(sample_rect)
        painter.setPen(QColor("#1d4ed8"))
        painter.drawText(
            int(sample_rect.left() + 8),
            int(sample_rect.top() + 20),
            "Amostra ruido",
        )

        if self._sample_editing:
            for handle_rect in self._selection_handles(sample_rect).values():
                painter.fillRect(handle_rect, QColor("#2563eb"))

    def _display_crop_rect(self) -> QRectF:
        return self._display_rect(self._crop_rect_norm)

    def _display_rect(self, rect_norm: QRectF) -> QRectF:
        if self._image_rect.width() <= 0 or self._image_rect.height() <= 0:
            return QRectF()
        return QRectF(
            self._image_rect.left() + (rect_norm.x() * self._image_rect.width()),
            self._image_rect.top() + (rect_norm.y() * self._image_rect.height()),
            rect_norm.width() * self._image_rect.width(),
            rect_norm.height() * self._image_rect.height(),
        )

    def _selection_handles(self, selection_rect: QRectF) -> dict[str, QRectF]:
        size = 10.0
        half = size / 2.0
        return {
            "top_left": QRectF(selection_rect.left() - half, selection_rect.top() - half, size, size),
            "top_right": QRectF(selection_rect.right() - half, selection_rect.top() - half, size, size),
            "bottom_left": QRectF(selection_rect.left() - half, selection_rect.bottom() - half, size, size),
            "bottom_right": QRectF(selection_rect.right() - half, selection_rect.bottom() - half, size, size),
        }

    def _selection_handle_at(self, position: QPointF, selection_rect: QRectF) -> Optional[str]:
        for handle, handle_rect in self._selection_handles(selection_rect).items():
            if handle_rect.contains(position):
                return handle
        return None

    def _cursor_for_handle(self, handle: str) -> Qt.CursorShape:
        if handle in {"top_left", "bottom_right"}:
            return Qt.CursorShape.SizeFDiagCursor
        return Qt.CursorShape.SizeBDiagCursor

    def _point_to_normalized(self, point: QPointF) -> QPointF:
        if self._image_rect.width() <= 0 or self._image_rect.height() <= 0:
            return QPointF(0.0, 0.0)
        norm_x = (point.x() - self._image_rect.left()) / self._image_rect.width()
        norm_y = (point.y() - self._image_rect.top()) / self._image_rect.height()
        return QPointF(min(1.0, max(0.0, norm_x)), min(1.0, max(0.0, norm_y)))

    def _anchor_for_handle(self, handle: str, rect: QRectF) -> QPointF:
        anchor_map = {
            "top_left": rect.bottomRight(),
            "top_right": rect.bottomLeft(),
            "bottom_left": rect.topRight(),
            "bottom_right": rect.topLeft(),
        }
        return QPointF(anchor_map[handle])

    def _update_crop_drag(self, position: QPointF) -> None:
        pointer_norm = self._point_to_normalized(position)
        if self._crop_drag_mode == "move":
            new_rect = QRectF(
                pointer_norm.x() - self._crop_drag_offset.x(),
                pointer_norm.y() - self._crop_drag_offset.y(),
                self._crop_rect_norm.width(),
                self._crop_rect_norm.height(),
            )
            new_rect = self._move_rect_within_bounds(new_rect)
        else:
            new_rect = self._resize_rect(
                self._crop_drag_mode,
                pointer_norm,
                self._crop_anchor_norm,
                self._crop_aspect_ratio,
            )

        self._crop_rect_norm = new_rect
        self.cropChanged.emit(QRectF(self._crop_rect_norm))
        self.update()

    def _update_sample_drag(self, position: QPointF) -> None:
        pointer_norm = self._point_to_normalized(position)
        if self._sample_drag_mode == "move":
            new_rect = QRectF(
                pointer_norm.x() - self._sample_drag_offset.x(),
                pointer_norm.y() - self._sample_drag_offset.y(),
                self._sample_rect_norm.width(),
                self._sample_rect_norm.height(),
            )
            new_rect = self._move_rect_within_bounds(new_rect)
        else:
            new_rect = self._resize_rect(
                self._sample_drag_mode,
                pointer_norm,
                self._sample_anchor_norm,
                None,
            )

        self._sample_rect_norm = new_rect
        self.sampleChanged.emit(QRectF(self._sample_rect_norm))
        self.update()

    def _move_rect_within_bounds(self, rect: QRectF) -> QRectF:
        x = min(max(0.0, rect.x()), 1.0 - rect.width())
        y = min(max(0.0, rect.y()), 1.0 - rect.height())
        return QRectF(x, y, rect.width(), rect.height())

    def _resize_rect(
        self,
        handle: str,
        pointer_norm: QPointF,
        anchor: QPointF,
        aspect_ratio: Optional[float],
    ) -> QRectF:
        min_size = 0.08
        dx = pointer_norm.x() - anchor.x()
        dy = pointer_norm.y() - anchor.y()
        sign_x = 1.0 if dx >= 0 else -1.0
        sign_y = 1.0 if dy >= 0 else -1.0
        width = max(min_size, abs(dx))
        height = max(min_size, abs(dy))

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
        left = min(anchor.x(), x2)
        top = min(anchor.y(), y2)
        rect = QRectF(left, top, abs(x2 - anchor.x()), abs(y2 - anchor.y()))
        return self._move_rect_within_bounds(rect)


class MainWindow(QMainWindow):
    EXPORT_CANCELLED_MESSAGE = "Exportacao cancelada pelo usuario."
    PROCESSING_MODE_OPTIONS = [
        ("CPU", "cpu"),
        ("GPU automatica", "gpu"),
    ]
    DENOISE_MODE_OPTIONS = [
        ("Manual", "manual"),
        ("Auto", "auto"),
        ("Amostra do crop", "sample"),
    ]
    DENOISE_TEMPLATE_OPTIONS = [3, 5, 7, 9]
    DENOISE_SEARCH_OPTIONS = [11, 15, 21, 27, 35]
    FILTER_THUMBNAIL_SIZE = QSize(152, 86)
    SCALE_OPTIONS = [
        ("1.0x (original)", 1.0),
        ("1.5x (bom para 720p -> 1080p)", 1.5),
        ("2.0x", 2.0),
    ]
    CROP_OPTIONS = [
        ("Sem crop", None),
        ("Livre", "free"),
        ("16:9", 16 / 9),
        ("4:3", 4 / 3),
        ("1:1", 1.0),
        ("9:16", 9 / 16),
    ]
    PREVIEW_ZOOM_OPTIONS = [50, 75, 100, 125, 150, 200]
    IMAGE_FILTER_PRESETS = [
        ImageFilterPreset(
            "original",
            "Original",
            "Mantem apenas os ajustes manuais atuais, sem correcao tonal extra.",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        ImageFilterPreset(
            "shadow_lift",
            "Sombras abertas",
            "Clareia areas escuras e segura os brancos para cenas com contraste alto.",
            0.55,
            0.20,
            0.10,
            0.08,
            0.0,
            0.03,
            0.0,
            0.0,
        ),
        ImageFilterPreset(
            "highlight_guard",
            "Realces suaves",
            "Comprime altas luzes para reduzir estouro em areas claras e janelas.",
            0.18,
            0.65,
            0.08,
            0.10,
            -0.02,
            -0.03,
            0.0,
            0.0,
        ),
        ImageFilterPreset(
            "dynamic_balance",
            "Faixa dinamica",
            "Equilibra sombras e realces ao mesmo tempo, bom para cenas muito duras.",
            0.48,
            0.48,
            0.16,
            0.14,
            0.0,
            0.04,
            0.0,
            0.0,
        ),
        ImageFilterPreset(
            "neutral_color",
            "Cor neutra",
            "Corrige dominantes de cor e entrega um balanco mais natural de branco e pele.",
            0.22,
            0.28,
            0.10,
            0.38,
            0.0,
            0.02,
            0.0,
            0.0,
        ),
        ImageFilterPreset(
            "smart_contrast",
            "Contraste inteligente",
            "Aumenta contraste local sem endurecer tanto as altas luzes.",
            0.20,
            0.30,
            0.38,
            0.12,
            0.02,
            0.05,
            0.0,
            0.0,
        ),
        ImageFilterPreset(
            "micro_contrast",
            "Microcontraste",
            "Realca textura fina e separacao entre planos sem abrir tanto as sombras.",
            0.14,
            0.24,
            0.52,
            0.10,
            0.01,
            0.04,
            0.0,
            0.0,
        ),
        ImageFilterPreset(
            "contrast_punch",
            "Contraste punch",
            "Entrega mais presenca e recorte nos detalhes, com um look mais energico.",
            0.10,
            0.34,
            0.62,
            0.08,
            0.0,
            0.06,
            0.0,
            0.0,
        ),
        ImageFilterPreset(
            "warm_tamer",
            "Atenuar quentes",
            "Reduz vermelhos, laranjas e amarelos mantendo verdes, azuis e neutros mais intactos.",
            0.18,
            0.24,
            0.16,
            0.10,
            -0.04,
            -0.02,
            0.48,
            0.0,
        ),
        ImageFilterPreset(
            "cool_tamer",
            "Atenuar frias",
            "Suaviza azuis e cianos mantendo quentes e tons de pele com mais consistencia.",
            0.18,
            0.22,
            0.14,
            0.10,
            0.04,
            -0.01,
            0.0,
            0.48,
        ),
    ]
    DEFAULT_FILTER_SETTINGS = FilterSettings(
        denoise_mode="manual",
        denoise_strength=20,
        denoise_color_strength=20,
        denoise_template_window=7,
        denoise_search_window=21,
        sharpen=25,
        contrast=50,
        brightness=50,
        gamma=50,
        saturation=50,
        temperature=50,
        scale_factor=1.0,
    )
    FILTER_PRESETS = [
        ("Personalizado", None),
        (
            "Limpeza leve",
            FilterSettings(
                denoise_mode="manual",
                denoise_strength=12,
                denoise_color_strength=10,
                denoise_template_window=7,
                denoise_search_window=21,
                sharpen=12,
                contrast=52,
                brightness=50,
                gamma=50,
                saturation=50,
                temperature=50,
                scale_factor=1.0,
            ),
        ),
        (
            "Limpeza clara",
            FilterSettings(
                denoise_mode="manual",
                denoise_strength=25,
                denoise_color_strength=20,
                denoise_template_window=7,
                denoise_search_window=21,
                sharpen=15,
                contrast=50,
                brightness=50,
                gamma=35,
                saturation=40,
                temperature=50,
                scale_factor=1.0,
            ),
        ),
        (
            "Balanceado 1080p",
            FilterSettings(
                denoise_mode="manual",
                denoise_strength=22,
                denoise_color_strength=18,
                denoise_template_window=7,
                denoise_search_window=21,
                sharpen=24,
                contrast=54,
                brightness=51,
                gamma=50,
                saturation=52,
                temperature=50,
                scale_factor=1.5,
            ),
        ),
        (
            "Recuperacao forte",
            FilterSettings(
                denoise_mode="manual",
                denoise_strength=38,
                denoise_color_strength=34,
                denoise_template_window=7,
                denoise_search_window=27,
                sharpen=32,
                contrast=56,
                brightness=52,
                gamma=52,
                saturation=54,
                temperature=50,
                scale_factor=1.5,
            ),
        ),
        (
            "Mais detalhe 2x",
            FilterSettings(
                denoise_mode="manual",
                denoise_strength=24,
                denoise_color_strength=18,
                denoise_template_window=7,
                denoise_search_window=21,
                sharpen=38,
                contrast=55,
                brightness=50,
                gamma=50,
                saturation=55,
                temperature=50,
                scale_factor=2.0,
            ),
        ),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Space Video Enhancer")
        self.resize(1220, 760)

        self.video_path: Optional[Path] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.fps = 30.0
        self.source_size = (0, 0)
        self.current_frame_index = 0
        self.clip_start_frame = 0
        self.clip_end_frame = 0
        self.current_frame: Optional[np.ndarray] = None
        self.active_preset_settings = self.DEFAULT_FILTER_SETTINGS
        self.active_image_filter_id = self.IMAGE_FILTER_PRESETS[0].filter_id
        self.active_image_filter_intensity = 100
        self.playback_end_frame = 0
        self.crop_rect_norm = QRectF(0.1, 0.1, 0.8, 0.8)
        self.denoise_sample_rect_norm = QRectF(0.33, 0.33, 0.34, 0.34)
        self.export_cancel_requested = False
        self._last_export_ui_update = 0.0
        self._opencl_checked = False
        self._opencl_available = False
        self._opencl_failed = False
        self._ffmpeg_encoder_cache: Optional[set[str]] = None
        self._ffmpeg_encoder_cache_path: Optional[str] = None
        self._ffmpeg_encoder_probe_cache: dict[tuple[str, str], bool] = {}
        self._current_export_backend_label = "CPU + libx264"
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_playback_frame)

        self.preview = ComparePreview()
        self.preview.cropChanged.connect(self._handle_preview_crop_changed)
        self.preview.sampleChanged.connect(self._handle_preview_sample_changed)
        self.preview_zoom_out_button = QPushButton("−")
        self.preview_zoom_in_button = QPushButton("+")
        self.preview_zoom_combo = QComboBox()
        for zoom_percent in self.PREVIEW_ZOOM_OPTIONS:
            self.preview_zoom_combo.addItem(f"{zoom_percent}%", zoom_percent)
        default_zoom_index = self.preview_zoom_combo.findData(100)
        if default_zoom_index >= 0:
            self.preview_zoom_combo.setCurrentIndex(default_zoom_index)
        self.preview_zoom_combo.setToolTip("Define o zoom exibido no preview")
        self.select_button = QPushButton("Selecionar video")
        self.export_button = QPushButton("Aplicar e exportar")
        self.thread_label = QLabel("Threads")
        self.thread_combo = QComboBox()
        for thread_count in (1, 2, 4, 6, 8):
            self.thread_combo.addItem(str(thread_count), thread_count)
        self.thread_combo.setCurrentIndex(0)
        self.thread_combo.setToolTip("Escolha quantas threads usar na exportacao")
        self.acceleration_combo = QComboBox()
        for label, mode in self.PROCESSING_MODE_OPTIONS:
            self.acceleration_combo.addItem(label, mode)
        self.acceleration_combo.setCurrentIndex(0)
        self.acceleration_combo.setToolTip(
            "Usa OpenCL no processamento e encoder por hardware no FFmpeg quando houver suporte"
        )
        self.cancel_export_button = QPushButton("Cancelar exportacao")
        self.cancel_export_button.setObjectName("cancelExportButton")
        self.cancel_export_button.setEnabled(False)
        self.file_label = QLabel("Nenhum video selecionado")
        self.file_label.setWordWrap(True)
        self.frame_label = QLabel("Frame 0 / 0")
        self.status_label = QLabel("Ajuste os controles e revise um frame antes de exportar.")
        self.status_label.setWordWrap(True)

        self.frame_slider = FrameRangeSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.go_first_button = QPushButton("⏮")
        self.prev_frame_button = QPushButton("◀")
        self.play_button = QPushButton("⏵")
        self.next_frame_button = QPushButton("▶")
        self.go_last_button = QPushButton("⏭")
        for button, tooltip in (
            (self.go_first_button, "Ir para o inicio"),
            (self.prev_frame_button, "Voltar um frame"),
            (self.play_button, "Reproduzir ou pausar a visualizacao"),
            (self.next_frame_button, "Avancar um frame"),
            (self.go_last_button, "Ir para o final"),
        ):
            button.setObjectName("transportButton")
            button.setFixedSize(42, 38)
            button.setToolTip(tooltip)

        self.denoise_mode_combo = QComboBox()
        for label, mode in self.DENOISE_MODE_OPTIONS:
            self.denoise_mode_combo.addItem(label, mode)
        self.denoise_strength_slider = LabeledSlider(
            "Forca de luminancia",
            0,
            100,
            self.DEFAULT_FILTER_SETTINGS.denoise_strength,
        )
        self.denoise_color_slider = LabeledSlider(
            "Forca de cor",
            0,
            100,
            self.DEFAULT_FILTER_SETTINGS.denoise_color_strength,
        )
        self.denoise_template_combo = QComboBox()
        for value in self.DENOISE_TEMPLATE_OPTIONS:
            self.denoise_template_combo.addItem(f"{value} px", value)
        self.denoise_search_combo = QComboBox()
        for value in self.DENOISE_SEARCH_OPTIONS:
            self.denoise_search_combo.addItem(f"{value} px", value)
        self.denoise_sample_edit_button = QPushButton("Editar amostra no preview")
        self.denoise_sample_edit_button.setCheckable(True)
        self.denoise_sample_reset_button = QPushButton("Resetar amostra")
        self.denoise_info_label = QLabel()
        self.denoise_info_label.setWordWrap(True)
        self.denoise_sample_info_label = QLabel()
        self.denoise_sample_info_label.setWordWrap(True)
        self.filter_preset_buttons: dict[str, QToolButton] = {}
        self.filter_button_group = QButtonGroup(self)
        self.filter_button_group.setExclusive(True)
        self.filter_description_label = QLabel()
        self.filter_description_label.setWordWrap(True)
        self.filter_intensity_slider = LabeledSlider("Intensidade do filtro", 0, 100, self.active_image_filter_intensity)
        self.sharpen_slider = LabeledSlider("Nitidez", 0, 100, self.DEFAULT_FILTER_SETTINGS.sharpen)
        self.contrast_slider = LabeledSlider("Contraste", 0, 100, self.DEFAULT_FILTER_SETTINGS.contrast)
        self.brightness_slider = LabeledSlider("Brilho", 0, 100, self.DEFAULT_FILTER_SETTINGS.brightness)
        self.gamma_slider = LabeledSlider("Gama", 0, 100, self.DEFAULT_FILTER_SETTINGS.gamma)
        self.saturation_slider = LabeledSlider("Saturacao", 0, 100, self.DEFAULT_FILTER_SETTINGS.saturation)
        self.temperature_slider = LabeledSlider("Temperatura", 0, 100, self.DEFAULT_FILTER_SETTINGS.temperature)
        self.preset_combo = QComboBox()
        for label, preset in self.FILTER_PRESETS:
            self.preset_combo.addItem(label, preset)
        self.scale_combo = QComboBox()
        for label, factor in self.SCALE_OPTIONS:
            self.scale_combo.addItem(label, factor)
        self.crop_mode_combo = QComboBox()
        for label, mode in self.CROP_OPTIONS:
            self.crop_mode_combo.addItem(label, mode)
        self.crop_reset_button = QPushButton("Reset crop")
        self.crop_info_label = QLabel("Crop: desativado")
        self.crop_info_label.setWordWrap(True)
        self.mark_start_button = QPushButton("Marcar inicio")
        self.mark_end_button = QPushButton("Marcar fim")
        self.reset_range_button = QPushButton("Video inteiro")
        self.range_label = QLabel("Corte: video inteiro")
        self.range_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")

        self._build_ui()
        self._build_menu_bar()
        self._bind_events()
        self._apply_settings_to_controls(self.DEFAULT_FILTER_SETTINGS)
        self._handle_preview_zoom_changed(self.preview_zoom_combo.currentIndex())
        self._update_denoise_controls_state()
        self._set_controls_enabled(False)
        self.select_button.setEnabled(True)
        self.file_label.setEnabled(True)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        left_column = QVBoxLayout()
        left_column.setSpacing(16)

        preview_toolbar = QHBoxLayout()
        preview_toolbar.setSpacing(10)
        preview_toolbar.addWidget(QLabel("Zoom do preview"))
        self.preview_zoom_out_button.setObjectName("previewZoomButton")
        self.preview_zoom_out_button.setFixedWidth(38)
        self.preview_zoom_out_button.setToolTip("Diminuir zoom do preview")
        preview_toolbar.addWidget(self.preview_zoom_out_button)
        preview_toolbar.addWidget(self.preview_zoom_combo)
        self.preview_zoom_in_button.setObjectName("previewZoomButton")
        self.preview_zoom_in_button.setFixedWidth(38)
        self.preview_zoom_in_button.setToolTip("Aumentar zoom do preview")
        preview_toolbar.addWidget(self.preview_zoom_in_button)
        preview_toolbar.addStretch(1)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setObjectName("previewScrollArea")
        self.preview_scroll.setWidget(self.preview)
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        frame_row = QHBoxLayout()
        frame_row.addWidget(self.frame_label)
        frame_row.addWidget(self.frame_slider, 1)

        transport_row = QHBoxLayout()
        transport_row.setSpacing(8)
        transport_row.addStretch(1)
        transport_row.addWidget(self.go_first_button)
        transport_row.addWidget(self.prev_frame_button)
        transport_row.addWidget(self.play_button)
        transport_row.addWidget(self.next_frame_button)
        transport_row.addWidget(self.go_last_button)
        transport_row.addStretch(1)

        left_column.addLayout(preview_toolbar)
        left_column.addWidget(self.preview_scroll, 1)
        left_column.addLayout(frame_row)
        left_column.addLayout(transport_row)

        self.settings_tabs = QTabWidget()
        self.settings_tabs.setObjectName("settingsTabs")
        self.settings_tabs.setMinimumWidth(340)
        self.settings_tabs.addTab(self._build_color_tab(), "Cor e detalhe")
        self.settings_tabs.addTab(self._build_filters_tab(), "Filtros")
        self.settings_tabs.addTab(self._build_geometry_tab(), "Geometria")
        self.settings_tabs.addTab(self._build_noise_tab(), "Ruido")
        self.settings_tabs.addTab(self._build_export_tab(), "Midia")

        layout = QHBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)
        layout.addLayout(left_column, 3)
        layout.addWidget(self.settings_tabs, 1)

        self.setStyleSheet(
            """
            QWidget {
                background: #fcfaf7;
                color: #1f2937;
                font-size: 13px;
            }
            QPushButton {
                background: #0f766e;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 16px;
                font-weight: 600;
            }
            QPushButton:disabled {
                background: #9ca3af;
            }
            QPushButton#cancelExportButton {
                background: #b91c1c;
            }
            QPushButton#cancelExportButton:disabled {
                background: #9ca3af;
            }
            QPushButton#transportButton {
                padding: 4px;
                font-size: 16px;
                font-weight: 500;
            }
            QPushButton#previewZoomButton {
                padding: 4px;
                font-size: 18px;
                min-width: 38px;
            }
            QToolButton#filterPresetButton {
                background: white;
                border: 1px solid #d6d3d1;
                border-radius: 12px;
                padding: 8px;
                color: #1f2937;
                font-weight: 600;
            }
            QToolButton#filterPresetButton:checked {
                border: 2px solid #0f766e;
                background: #ecfeff;
            }
            QGroupBox {
                border: 1px solid #d6d3d1;
                border-radius: 14px;
                margin-top: 14px;
                padding: 18px 14px 14px 14px;
                background: #f8f5ef;
                font-weight: 600;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QMenuBar {
                background: #f8f5ef;
                border-bottom: 1px solid #d6d3d1;
                padding: 4px 8px;
            }
            QMenuBar::item {
                padding: 6px 10px;
                border-radius: 8px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background: #efe9de;
            }
            QMenu {
                background: #fffdf8;
                border: 1px solid #d6d3d1;
                padding: 6px;
            }
            QMenu::item {
                padding: 8px 18px;
                border-radius: 8px;
            }
            QMenu::item:selected {
                background: #0f766e;
                color: white;
            }
            QTabWidget::pane {
                border: 1px solid #d6d3d1;
                border-radius: 14px;
                background: #f8f5ef;
                top: -1px;
            }
            QTabBar::tab {
                background: #efe9de;
                border: 1px solid #d6d3d1;
                border-bottom: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 10px 14px;
                margin-right: 4px;
                color: #4b5563;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #0f766e;
                color: white;
            }
            QTabBar::tab:!selected {
                margin-top: 3px;
            }
            QSlider::groove:horizontal {
                border-radius: 999px;
                height: 8px;
                background: #d6d3d1;
            }
            QSlider::handle:horizontal {
                background: #0f766e;
                border: 2px solid #ecfeff;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QComboBox {
                background: white;
                border: 1px solid #d6d3d1;
                border-radius: 8px;
                padding: 6px 10px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollArea#previewScrollArea {
                background: #f3f4f6;
                border: 1px solid #d6d3d1;
                border-radius: 14px;
            }
            QProgressBar {
                border: 1px solid #d6d3d1;
                border-radius: 9px;
                background: #efe9de;
                text-align: center;
                min-height: 20px;
                color: #1f2937;
            }
            QProgressBar::chunk {
                border-radius: 8px;
                background: #0f766e;
            }
            """
        )

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("Arquivo")
        open_action = QAction("Abrir video...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._select_video)
        file_menu.addAction(open_action)

        export_action = QAction("Exportar video...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export_video)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        exit_action = QAction("Sair", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menu_bar.addMenu("Editar")
        restore_preset_action = QAction("Restaurar preset ativo", self)
        restore_preset_action.setShortcut("Ctrl+R")
        restore_preset_action.triggered.connect(self._restore_active_preset_settings)
        edit_menu.addAction(restore_preset_action)

        reset_crop_action = QAction("Resetar crop", self)
        reset_crop_action.triggered.connect(self._reset_crop_rect)
        edit_menu.addAction(reset_crop_action)

        reset_range_action = QAction("Resetar corte temporal", self)
        reset_range_action.triggered.connect(self._reset_clip_range)
        edit_menu.addAction(reset_range_action)

        playback_menu = menu_bar.addMenu("Reproducao")
        play_pause_action = QAction("Reproduzir ou pausar", self)
        play_pause_action.setShortcut("Space")
        play_pause_action.triggered.connect(self._toggle_playback)
        playback_menu.addAction(play_pause_action)

        playback_menu.addSeparator()
        first_frame_action = QAction("Primeiro frame", self)
        first_frame_action.setShortcut("Home")
        first_frame_action.triggered.connect(self._go_to_first_frame)
        playback_menu.addAction(first_frame_action)

        previous_frame_action = QAction("Frame anterior", self)
        previous_frame_action.setShortcut("Left")
        previous_frame_action.triggered.connect(self._go_to_previous_frame)
        playback_menu.addAction(previous_frame_action)

        next_frame_action = QAction("Proximo frame", self)
        next_frame_action.setShortcut("Right")
        next_frame_action.triggered.connect(self._go_to_next_frame)
        playback_menu.addAction(next_frame_action)

        last_frame_action = QAction("Ultimo frame", self)
        last_frame_action.setShortcut("End")
        last_frame_action.triggered.connect(self._go_to_last_frame)
        playback_menu.addAction(last_frame_action)

        playback_menu.addSeparator()
        mark_start_action = QAction("Marcar inicio do corte", self)
        mark_start_action.setShortcut("I")
        mark_start_action.triggered.connect(self._mark_clip_start)
        playback_menu.addAction(mark_start_action)

        mark_end_action = QAction("Marcar fim do corte", self)
        mark_end_action.setShortcut("O")
        mark_end_action.triggered.connect(self._mark_clip_end)
        playback_menu.addAction(mark_end_action)

        help_menu = menu_bar.addMenu("Ajuda")
        about_action = QAction("Sobre", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

    def _build_color_tab(self) -> QScrollArea:
        color_group = QGroupBox("Cor e look")
        color_layout = QVBoxLayout(color_group)
        color_layout.setSpacing(14)
        color_form = QFormLayout()
        color_form.addRow("Preset", self.preset_combo)
        color_layout.addLayout(color_form)
        color_layout.addWidget(self.contrast_slider)
        color_layout.addWidget(self.brightness_slider)
        color_layout.addWidget(self.gamma_slider)
        color_layout.addWidget(self.saturation_slider)
        color_layout.addWidget(self.temperature_slider)

        detail_group = QGroupBox("Detalhe")
        detail_layout = QVBoxLayout(detail_group)
        detail_layout.setSpacing(14)
        detail_layout.addWidget(self.sharpen_slider)

        return self._wrap_tab_content(color_group, detail_group)

    def _build_filters_tab(self) -> QScrollArea:
        filters_group = QGroupBox("Presets de imagem")
        filters_layout = QVBoxLayout(filters_group)
        filters_layout.setSpacing(14)

        filters_grid = QGridLayout()
        filters_grid.setHorizontalSpacing(12)
        filters_grid.setVerticalSpacing(12)
        for index, preset in enumerate(self.IMAGE_FILTER_PRESETS):
            button = QToolButton()
            button.setObjectName("filterPresetButton")
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setIconSize(self.FILTER_THUMBNAIL_SIZE)
            button.setMinimumSize(170, 140)
            button.setText(preset.name)
            button.setToolTip(preset.description)
            button.clicked.connect(lambda _checked=False, filter_id=preset.filter_id: self._select_image_filter(filter_id))
            self.filter_button_group.addButton(button)
            self.filter_preset_buttons[preset.filter_id] = button
            filters_grid.addWidget(button, index // 2, index % 2)

        filters_layout.addLayout(filters_grid)
        filters_layout.addWidget(self.filter_intensity_slider)
        filters_layout.addWidget(self.filter_description_label)
        self._update_filter_selection_ui()
        self._update_filter_description()

        return self._wrap_tab_content(filters_group)

    def _build_geometry_tab(self) -> QScrollArea:
        crop_group = QGroupBox("Enquadramento")
        crop_layout = QVBoxLayout(crop_group)
        crop_layout.setSpacing(14)
        crop_form = QFormLayout()
        crop_form.addRow("Crop", self.crop_mode_combo)
        crop_layout.addLayout(crop_form)
        crop_layout.addWidget(self.crop_reset_button)
        crop_layout.addWidget(self.crop_info_label)

        return self._wrap_tab_content(crop_group)

    def _build_noise_tab(self) -> QScrollArea:
        noise_group = QGroupBox("Limpeza e reducao de ruido")
        noise_layout = QVBoxLayout(noise_group)
        noise_layout.setSpacing(14)
        noise_form = QFormLayout()
        noise_form.addRow("Modo", self.denoise_mode_combo)
        noise_form.addRow("Janela de template", self.denoise_template_combo)
        noise_form.addRow("Janela de busca", self.denoise_search_combo)
        noise_layout.addLayout(noise_form)
        noise_layout.addWidget(self.denoise_strength_slider)
        noise_layout.addWidget(self.denoise_color_slider)
        noise_layout.addWidget(self.denoise_sample_edit_button)
        noise_layout.addWidget(self.denoise_sample_reset_button)
        noise_layout.addWidget(self.denoise_info_label)
        noise_layout.addWidget(self.denoise_sample_info_label)

        return self._wrap_tab_content(noise_group)

    def _build_export_tab(self) -> QScrollArea:
        source_group = QGroupBox("Midia de entrada")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(12)
        source_layout.addWidget(self.select_button)
        source_layout.addWidget(self.file_label)

        action_group = QGroupBox("Acoes")
        action_layout = QVBoxLayout(action_group)
        action_layout.setSpacing(12)
        action_layout.addWidget(self.export_button)
        action_layout.addWidget(self.cancel_export_button)

        processing_group = QGroupBox("Processamento")
        processing_layout = QVBoxLayout(processing_group)
        processing_layout.setSpacing(14)
        processing_form = QFormLayout()
        processing_form.addRow("Escala de saida", self.scale_combo)
        processing_form.addRow("Aceleracao", self.acceleration_combo)
        processing_form.addRow(self.thread_label, self.thread_combo)
        processing_layout.addLayout(processing_form)

        range_group = QGroupBox("Corte temporal")
        range_layout = QVBoxLayout(range_group)
        range_layout.setSpacing(12)
        range_buttons = QHBoxLayout()
        range_buttons.setSpacing(10)
        range_buttons.addWidget(self.mark_start_button)
        range_buttons.addWidget(self.mark_end_button)
        range_buttons.addWidget(self.reset_range_button)
        range_layout.addLayout(range_buttons)
        range_layout.addWidget(self.range_label)

        export_group = QGroupBox("Exportacao")
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(14)
        export_layout.addWidget(self.progress_bar)
        export_layout.addWidget(self.status_label)

        return self._wrap_tab_content(source_group, action_group, processing_group, range_group, export_group)

    def _wrap_tab_content(self, *sections: QWidget) -> QScrollArea:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)
        for section in sections:
            layout.addWidget(section)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def _bind_events(self) -> None:
        self.select_button.clicked.connect(self._select_video)
        self.export_button.clicked.connect(self._export_video)
        self.cancel_export_button.clicked.connect(self._request_export_cancel)
        self.preview_zoom_out_button.clicked.connect(lambda: self._step_preview_zoom(-1))
        self.preview_zoom_in_button.clicked.connect(lambda: self._step_preview_zoom(1))
        self.preview_zoom_combo.currentIndexChanged.connect(self._handle_preview_zoom_changed)
        self.frame_slider.valueChanged.connect(self._load_frame)
        self.go_first_button.clicked.connect(self._go_to_first_frame)
        self.prev_frame_button.clicked.connect(self._go_to_previous_frame)
        self.play_button.clicked.connect(self._toggle_playback)
        self.next_frame_button.clicked.connect(self._go_to_next_frame)
        self.go_last_button.clicked.connect(self._go_to_last_frame)
        self.preset_combo.currentIndexChanged.connect(self._handle_preset_changed)
        self.crop_mode_combo.currentIndexChanged.connect(self._handle_crop_mode_changed)
        self.crop_reset_button.clicked.connect(self._reset_crop_rect)
        self.denoise_mode_combo.currentIndexChanged.connect(self._handle_denoise_mode_changed)
        self.denoise_template_combo.currentIndexChanged.connect(self._handle_filter_control_change)
        self.denoise_search_combo.currentIndexChanged.connect(self._handle_filter_control_change)
        self.denoise_sample_edit_button.toggled.connect(self._handle_denoise_sample_edit_toggled)
        self.denoise_sample_reset_button.clicked.connect(self._reset_denoise_sample_rect)
        self.filter_intensity_slider.valueChanged.connect(self._handle_filter_intensity_changed)
        self.scale_combo.currentIndexChanged.connect(self._handle_filter_control_change)
        self.acceleration_combo.currentIndexChanged.connect(self._handle_acceleration_mode_changed)
        self.mark_start_button.clicked.connect(self._mark_clip_start)
        self.mark_end_button.clicked.connect(self._mark_clip_end)
        self.reset_range_button.clicked.connect(self._reset_clip_range)

        for control in (
            self.denoise_strength_slider,
            self.denoise_color_slider,
            self.sharpen_slider,
            self.contrast_slider,
            self.brightness_slider,
            self.gamma_slider,
            self.saturation_slider,
            self.temperature_slider,
        ):
            control.valueChanged.connect(self._handle_filter_control_change)

        self.denoise_strength_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("denoise_strength"))
        self.denoise_color_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("denoise_color_strength"))
        self.sharpen_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("sharpen"))
        self.contrast_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("contrast"))
        self.brightness_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("brightness"))
        self.gamma_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("gamma"))
        self.saturation_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("saturation"))
        self.temperature_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("temperature"))

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.export_button,
            self.thread_label,
            self.thread_combo,
            self.frame_slider,
            self.go_first_button,
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.go_last_button,
            self.denoise_mode_combo,
            self.denoise_strength_slider,
            self.denoise_color_slider,
            self.denoise_template_combo,
            self.denoise_search_combo,
            self.denoise_sample_edit_button,
            self.denoise_sample_reset_button,
            self.filter_intensity_slider,
            self.sharpen_slider,
            self.contrast_slider,
            self.brightness_slider,
            self.gamma_slider,
            self.saturation_slider,
            self.temperature_slider,
            self.preset_combo,
            self.crop_mode_combo,
            self.crop_reset_button,
            self.scale_combo,
            self.acceleration_combo,
            self.mark_start_button,
            self.mark_end_button,
            self.reset_range_button,
        ):
            widget.setEnabled(enabled)
        for button in self.filter_preset_buttons.values():
            button.setEnabled(enabled)
        self._update_denoise_controls_state()
        self._update_filter_description()

    def _select_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar video",
            str(Path.home()),
            "Videos (*.mp4 *.mov *.mkv *.avi *.m4v)",
        )
        if not file_path:
            return

        self._open_video(Path(file_path))

    def _open_video(self, path: Path) -> None:
        self._stop_playback()
        if self.capture is not None:
            self.capture.release()

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            QMessageBox.critical(self, "Erro", "Nao foi possivel abrir o video selecionado.")
            return

        self.video_path = path
        self.capture = capture
        self.frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_size = (width, height)
        self.preview.set_display_reference_size(width, height)
        self.current_frame_index = 0
        self.clip_start_frame = 0
        self.clip_end_frame = self.frame_count - 1
        self.crop_rect_norm = self._default_crop_rect(self._normalized_crop_aspect_ratio())
        self.denoise_sample_rect_norm = self._default_denoise_sample_rect()
        self.file_label.setText(f"{path.name} | {width}x{height} @ {self.fps:.2f} fps")
        self.frame_slider.setRange(0, self.frame_count - 1)
        midpoint = min(self.frame_count - 1, self.frame_count // 2)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self._set_controls_enabled(True)
        self._update_frame_slider_markers()
        self._update_range_label()
        self._sync_crop_preview()
        self._sync_denoise_sample_preview()
        self._update_crop_info_label()
        self.status_label.setText("Arraste o divisor na imagem e os sliders ao lado para comparar o resultado.")

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(midpoint)
        self.frame_slider.blockSignals(False)
        self._load_frame(midpoint)

    def _handle_preview_zoom_changed(self, _index: int) -> None:
        zoom_percent = int(self.preview_zoom_combo.currentData() or 100)
        self.preview.set_zoom_factor(zoom_percent / 100.0)
        self._update_preview_zoom_controls()

    def _step_preview_zoom(self, step: int) -> None:
        current_index = self.preview_zoom_combo.currentIndex()
        if current_index < 0:
            return
        target_index = max(0, min(self.preview_zoom_combo.count() - 1, current_index + step))
        if target_index != current_index:
            self.preview_zoom_combo.setCurrentIndex(target_index)

    def _update_preview_zoom_controls(self) -> None:
        current_index = self.preview_zoom_combo.currentIndex()
        max_index = self.preview_zoom_combo.count() - 1
        self.preview_zoom_out_button.setEnabled(current_index > 0)
        self.preview_zoom_in_button.setEnabled(current_index < max_index)

    def _image_filter_preset(self, filter_id: str) -> ImageFilterPreset:
        for preset in self.IMAGE_FILTER_PRESETS:
            if preset.filter_id == filter_id:
                return preset
        return self.IMAGE_FILTER_PRESETS[0]

    def _select_image_filter(self, filter_id: str) -> None:
        if filter_id == self.active_image_filter_id:
            self._update_filter_selection_ui()
            return

        self.active_image_filter_id = filter_id
        self._update_filter_selection_ui()
        self._update_filter_description()
        self._refresh_preview_impl()

    def _handle_filter_intensity_changed(self, value: int) -> None:
        self.active_image_filter_intensity = value
        self._update_filter_description()
        self._refresh_preview_impl()

    def _update_filter_selection_ui(self) -> None:
        for filter_id, button in self.filter_preset_buttons.items():
            button.blockSignals(True)
            button.setChecked(filter_id == self.active_image_filter_id)
            button.blockSignals(False)

    def _update_filter_description(self) -> None:
        preset = self._image_filter_preset(self.active_image_filter_id)
        intensity = self._image_filter_intensity()
        self.filter_description_label.setText(
            f"{preset.name}: {preset.description} Intensidade atual: {int(intensity * 100)}%."
        )
        self.filter_intensity_slider.setEnabled(self.frame_count > 0 and preset.filter_id != "original")

    def _refresh_filter_thumbnails(self) -> None:
        if self.playback_timer.isActive():
            return

        if self.current_frame is None:
            for button in self.filter_preset_buttons.values():
                button.setIcon(QIcon())
            return

        settings = replace(self._current_settings(), scale_factor=1.0)
        thumbnail_source = self._thumbnail_source_frame(self.current_frame)
        for preset in self.IMAGE_FILTER_PRESETS:
            thumbnail_frame = self._render_filtered_frame(
                thumbnail_source,
                settings,
                preset.filter_id,
                1.0,
            )
            thumbnail_image = self._thumbnail_qimage(thumbnail_frame)
            self.filter_preset_buttons[preset.filter_id].setIcon(
                QIcon(QPixmap.fromImage(thumbnail_image))
            )

    def _thumbnail_source_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        target_width = self.FILTER_THUMBNAIL_SIZE.width()
        target_height = self.FILTER_THUMBNAIL_SIZE.height()
        scale = min(target_width / max(1, width), target_height / max(1, height))
        if scale >= 1.0:
            return frame.copy()
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def _thumbnail_qimage(self, frame: np.ndarray) -> QImage:
        preview_image = frame_to_qimage(frame)
        return preview_image.scaled(
            self.FILTER_THUMBNAIL_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _load_frame(self, frame_index: int) -> None:
        if self.capture is None:
            return

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            self.status_label.setText("Nao foi possivel carregar esse frame.")
            return

        self.current_frame_index = frame_index
        self.current_frame = frame
        self.frame_label.setText(f"Frame {frame_index + 1} / {self.frame_count}")
        self._update_denoise_controls_state()
        self._refresh_preview()

    def _go_to_first_frame(self) -> None:
        if self.frame_count <= 0:
            return
        self._stop_playback()
        self.frame_slider.setValue(0)

    def _go_to_previous_frame(self) -> None:
        if self.frame_count <= 0:
            return
        self._stop_playback()
        self.frame_slider.setValue(max(self.frame_slider.minimum(), self.frame_slider.value() - 1))

    def _go_to_next_frame(self) -> None:
        if self.frame_count <= 0:
            return
        self._stop_playback()
        self.frame_slider.setValue(min(self.frame_slider.maximum(), self.frame_slider.value() + 1))

    def _go_to_last_frame(self) -> None:
        if self.frame_count <= 0:
            return
        self._stop_playback()
        self.frame_slider.setValue(self.frame_slider.maximum())

    def _toggle_playback(self) -> None:
        if self.frame_count <= 1:
            return
        if self.playback_timer.isActive():
            self._stop_playback()
            return

        current_frame = self.frame_slider.value()
        start_frame, end_frame = self._selected_frame_range()
        full_video_selected = start_frame == 0 and end_frame == self.frame_slider.maximum()
        if not full_video_selected and not (start_frame <= current_frame <= end_frame):
            current_frame = start_frame
            self.frame_slider.setValue(current_frame)

        self.playback_end_frame = end_frame if not full_video_selected else self.frame_slider.maximum()
        if current_frame >= self.playback_end_frame:
            self._stop_playback()
            return

        self.play_button.setText("⏸")
        self.playback_timer.start(self._playback_interval_ms())

    def _stop_playback(self) -> None:
        was_active = self.playback_timer.isActive()
        if was_active:
            self.playback_timer.stop()
        self.play_button.setText("⏵")
        if was_active:
            self._refresh_filter_thumbnails()

    def _advance_playback_frame(self) -> None:
        if self.frame_slider.value() >= self.playback_end_frame:
            self._stop_playback()
            return
        self.frame_slider.setValue(self.frame_slider.value() + 1)

    def _playback_interval_ms(self) -> int:
        preview_fps = min(max(self.fps, 1.0), 12.0)
        return max(40, int(1000 / preview_fps))

    def _current_settings(self) -> FilterSettings:
        return FilterSettings(
            denoise_mode=str(self.denoise_mode_combo.currentData()),
            denoise_strength=self.denoise_strength_slider.value(),
            denoise_color_strength=self.denoise_color_slider.value(),
            denoise_template_window=int(self.denoise_template_combo.currentData()),
            denoise_search_window=int(self.denoise_search_combo.currentData()),
            sharpen=self.sharpen_slider.value(),
            contrast=self.contrast_slider.value(),
            brightness=self.brightness_slider.value(),
            gamma=self.gamma_slider.value(),
            saturation=self.saturation_slider.value(),
            temperature=self.temperature_slider.value(),
            scale_factor=float(self.scale_combo.currentData()),
        )

    def _crop_mode_data(self):
        return self.crop_mode_combo.currentData()

    def _crop_enabled(self) -> bool:
        return self._crop_mode_data() is not None

    def _crop_aspect_ratio(self) -> Optional[float]:
        crop_mode = self._crop_mode_data()
        if crop_mode in {None, "free"}:
            return None
        return float(crop_mode)

    def _source_aspect_ratio(self) -> float:
        if self.source_size[0] > 0 and self.source_size[1] > 0:
            return self.source_size[0] / self.source_size[1]
        return 1.0

    def _normalized_crop_aspect_ratio(self) -> Optional[float]:
        aspect_ratio = self._crop_aspect_ratio()
        if aspect_ratio is None:
            return None
        return aspect_ratio / self._source_aspect_ratio()

    def _mark_clip_start(self) -> None:
        self.clip_start_frame = self.current_frame_index
        if self.clip_start_frame > self.clip_end_frame:
            self.clip_end_frame = self.clip_start_frame
        self._update_frame_slider_markers()
        self._update_range_label()

    def _mark_clip_end(self) -> None:
        self.clip_end_frame = self.current_frame_index
        if self.clip_end_frame < self.clip_start_frame:
            self.clip_start_frame = self.clip_end_frame
        self._update_frame_slider_markers()
        self._update_range_label()

    def _reset_clip_range(self) -> None:
        if self.frame_count <= 0:
            return
        self.clip_start_frame = 0
        self.clip_end_frame = self.frame_count - 1
        self._update_frame_slider_markers()
        self._update_range_label()

    def _selected_frame_range(self) -> tuple[int, int]:
        if self.frame_count <= 0:
            return (0, 0)

        start_frame = max(0, min(self.clip_start_frame, self.frame_count - 1))
        end_frame = max(0, min(self.clip_end_frame, self.frame_count - 1))
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame
        return (start_frame, end_frame)

    def _selected_frame_count(self) -> int:
        start_frame, end_frame = self._selected_frame_range()
        return max(1, (end_frame - start_frame) + 1)

    def _selected_start_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        start_frame, _ = self._selected_frame_range()
        return start_frame / self.fps

    def _selected_duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self._selected_frame_count() / self.fps

    def _update_range_label(self) -> None:
        if self.frame_count <= 0:
            self.range_label.setText("Corte: video inteiro")
            return

        start_frame, end_frame = self._selected_frame_range()
        frame_count = self._selected_frame_count()
        duration_seconds = self._selected_duration_seconds()
        self.range_label.setText(
            f"Corte: frames {start_frame + 1}-{end_frame + 1} | {frame_count} frames | {duration_seconds:.2f}s"
        )

    def _handle_crop_mode_changed(self, _index: int) -> None:
        if self._crop_enabled():
            current_rect = QRectF(self.crop_rect_norm)
            aspect_ratio = self._normalized_crop_aspect_ratio()
            if self.frame_count <= 0:
                self.crop_rect_norm = self._default_crop_rect(aspect_ratio)
            elif aspect_ratio is None:
                self.crop_rect_norm = current_rect if current_rect.width() > 0 else self._default_crop_rect(None)
            else:
                self.crop_rect_norm = self._crop_rect_with_aspect(current_rect, aspect_ratio)
        self._sync_crop_preview()
        self._update_crop_info_label()
        self._update_denoise_controls_state()
        self._refresh_preview_impl()

    def _handle_preview_crop_changed(self, rect: object) -> None:
        if isinstance(rect, QRectF):
            aspect_ratio = self._normalized_crop_aspect_ratio()
            if aspect_ratio is None:
                self.crop_rect_norm = QRectF(rect)
            else:
                self.crop_rect_norm = self._crop_rect_with_aspect(QRectF(rect), aspect_ratio)
            self._sync_crop_preview()
            self._update_crop_info_label()
            self._update_denoise_controls_state()

    def _handle_preview_sample_changed(self, rect: object) -> None:
        if isinstance(rect, QRectF):
            self.denoise_sample_rect_norm = QRectF(rect)
            self._sync_denoise_sample_preview()
            self._update_denoise_sample_info_label()
            self._refresh_preview_impl()

    def _reset_crop_rect(self) -> None:
        self.crop_rect_norm = self._default_crop_rect(self._normalized_crop_aspect_ratio())
        self._sync_crop_preview()
        self._update_crop_info_label()
        self._update_denoise_controls_state()
        self._refresh_preview_impl()

    def _default_denoise_sample_rect(self) -> QRectF:
        return QRectF(0.33, 0.33, 0.34, 0.34)

    def _handle_denoise_sample_edit_toggled(self, checked: bool) -> None:
        self._sync_denoise_sample_preview(editing_override=checked)

    def _reset_denoise_sample_rect(self) -> None:
        self.denoise_sample_rect_norm = self._default_denoise_sample_rect()
        self._sync_denoise_sample_preview()
        self._update_denoise_sample_info_label()
        self._refresh_preview_impl()

    def _sync_crop_preview(self) -> None:
        self.preview.set_crop_state(self._crop_enabled(), self.crop_rect_norm, self._normalized_crop_aspect_ratio())

    def _sync_denoise_sample_preview(self, editing_override: Optional[bool] = None) -> None:
        denoise_mode = str(self.denoise_mode_combo.currentData())
        sample_enabled = self.frame_count > 0 and denoise_mode == "sample"
        editing = self.denoise_sample_edit_button.isChecked() if editing_override is None else editing_override
        self.preview.set_sample_state(sample_enabled, self.denoise_sample_rect_norm, editing)

    def _update_denoise_sample_info_label(self) -> None:
        if self.frame_count <= 0:
            self.denoise_sample_info_label.setText("Amostra: carregue um video para escolher a regiao de analise.")
            return

        width_ratio = self.denoise_sample_rect_norm.width()
        height_ratio = self.denoise_sample_rect_norm.height()
        sample_pixels = self._denoise_sample_rect_pixels(self.source_size[0], self.source_size[1])
        if sample_pixels is not None:
            _, _, sample_width, sample_height = sample_pixels
            self.denoise_sample_info_label.setText(
                f"Amostra: {sample_width}x{sample_height} | {width_ratio * 100:.0f}% x {height_ratio * 100:.0f}%"
            )
            return

        self.denoise_sample_info_label.setText(
            f"Amostra: {width_ratio * 100:.0f}% x {height_ratio * 100:.0f}%"
        )

    def _update_crop_info_label(self) -> None:
        if not self._crop_enabled():
            self.crop_info_label.setText("Crop: desativado")
            return

        crop_mode = self._crop_mode_data()
        if crop_mode == "free":
            prefix = "Crop: Livre"
            ratio_text = self._rect_ratio_text(self.crop_rect_norm)
        else:
            prefix = f"Crop: {self.crop_mode_combo.currentText()}"
            ratio_text = self.crop_mode_combo.currentText()

        width_ratio = self.crop_rect_norm.width()
        height_ratio = self.crop_rect_norm.height()
        if self.source_size[0] > 0 and self.source_size[1] > 0:
            crop_pixels = self._crop_rect_pixels(self.source_size[0], self.source_size[1])
            if crop_pixels is not None:
                _, _, crop_width, crop_height = crop_pixels
                self.crop_info_label.setText(
                    f"{prefix} | {crop_width}x{crop_height} | {ratio_text}"
                )
                return

        self.crop_info_label.setText(
            f"{prefix} | {width_ratio * 100:.0f}% x {height_ratio * 100:.0f}%"
        )

    def _rect_ratio_text(self, rect: QRectF) -> str:
        if rect.height() <= 0 or rect.width() <= 0:
            return "Livre"

        ratio = (rect.width() / rect.height()) * self._source_aspect_ratio()
        ratio_candidates = [
            (16 / 9, "16:9"),
            (4 / 3, "4:3"),
            (1.0, "1:1"),
            (9 / 16, "9:16"),
        ]
        best_label = None
        best_distance = None
        for candidate_ratio, label in ratio_candidates:
            distance = abs(candidate_ratio - ratio)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_label = label
        if best_label is not None and best_distance is not None and best_distance < 0.04:
            return best_label
        return f"{ratio:.2f}:1"

    def _default_crop_rect(self, aspect_ratio: Optional[float]) -> QRectF:
        max_size = 0.82
        if aspect_ratio is None:
            return QRectF(0.09, 0.09, max_size, max_size)

        width = max_size
        height = width / aspect_ratio
        if height > max_size:
            height = max_size
            width = height * aspect_ratio
        return QRectF((1.0 - width) / 2.0, (1.0 - height) / 2.0, width, height)

    def _crop_rect_with_aspect(self, rect: QRectF, aspect_ratio: float) -> QRectF:
        if rect.width() <= 0 or rect.height() <= 0:
            return self._default_crop_rect(aspect_ratio)

        center = rect.center()
        width = rect.width()
        height = width / aspect_ratio
        if height > rect.height():
            height = rect.height()
            width = height * aspect_ratio

        max_width = min(center.x(), 1.0 - center.x()) * 2.0
        max_height = min(center.y(), 1.0 - center.y()) * 2.0
        if height > max_height:
            height = max_height
            width = height * aspect_ratio
        if width > max_width:
            width = max_width
            height = width / aspect_ratio

        if width < 0.08 or height < 0.08:
            return self._default_crop_rect(aspect_ratio)

        return QRectF(center.x() - (width / 2.0), center.y() - (height / 2.0), width, height)

    def _update_frame_slider_markers(self) -> None:
        if self.frame_count <= 0:
            self.frame_slider.set_markers(0, 0, False)
            return

        start_frame, end_frame = self._selected_frame_range()
        self.frame_slider.set_markers(start_frame, end_frame, True)

    def _handle_preset_changed(self, index: int) -> None:
        preset = self.preset_combo.itemData(index)
        if preset is None:
            self._refresh_preview_impl()
            return

        self.active_preset_settings = preset
        self._apply_settings_to_controls(preset)
        self._refresh_preview_impl()
        if self.current_frame is not None:
            self.status_label.setText("Preset aplicado. Ajuste os sliders se quiser um refinamento manual.")

    def _reset_slider_to_preset(self, field_name: str) -> None:
        preset_settings = self.active_preset_settings or self.DEFAULT_FILTER_SETTINGS
        target_value = int(getattr(preset_settings, field_name))
        slider_map = {
            "denoise_strength": self.denoise_strength_slider,
            "denoise_color_strength": self.denoise_color_slider,
            "sharpen": self.sharpen_slider,
            "contrast": self.contrast_slider,
            "brightness": self.brightness_slider,
            "gamma": self.gamma_slider,
            "saturation": self.saturation_slider,
            "temperature": self.temperature_slider,
        }
        slider = slider_map[field_name]
        slider.setValue(target_value)

    def _restore_active_preset_settings(self) -> None:
        target_settings = self.active_preset_settings or self.DEFAULT_FILTER_SETTINGS
        self._apply_settings_to_controls(target_settings)
        self._sync_preset_selection()
        self._refresh_preview_impl()
        if self.current_frame is not None:
            self.status_label.setText("Ajustes restaurados para o preset ativo.")

    def _show_about_dialog(self) -> None:
        QMessageBox.information(
            self,
            "Sobre",
            "Space Video Enhancer\n\n"
            "Editor offline com preview antes/depois, crop interativo e exportacao local.",
        )

    def _apply_settings_to_controls(self, settings: FilterSettings) -> None:
        widgets = [
            self.denoise_mode_combo,
            self.denoise_strength_slider,
            self.denoise_color_slider,
            self.denoise_template_combo,
            self.denoise_search_combo,
            self.sharpen_slider,
            self.contrast_slider,
            self.brightness_slider,
            self.gamma_slider,
            self.saturation_slider,
            self.temperature_slider,
            self.scale_combo,
        ]
        for widget in widgets:
            widget.blockSignals(True)

        try:
            denoise_mode_index = self.denoise_mode_combo.findData(settings.denoise_mode)
            if denoise_mode_index >= 0:
                self.denoise_mode_combo.setCurrentIndex(denoise_mode_index)
            self.denoise_strength_slider.slider.setValue(settings.denoise_strength)
            self.denoise_color_slider.slider.setValue(settings.denoise_color_strength)
            template_index = self.denoise_template_combo.findData(settings.denoise_template_window)
            if template_index >= 0:
                self.denoise_template_combo.setCurrentIndex(template_index)
            search_index = self.denoise_search_combo.findData(settings.denoise_search_window)
            if search_index >= 0:
                self.denoise_search_combo.setCurrentIndex(search_index)
            self.sharpen_slider.slider.setValue(settings.sharpen)
            self.contrast_slider.slider.setValue(settings.contrast)
            self.brightness_slider.slider.setValue(settings.brightness)
            self.gamma_slider.slider.setValue(settings.gamma)
            self.saturation_slider.slider.setValue(settings.saturation)
            self.temperature_slider.slider.setValue(settings.temperature)
            scale_index = self.scale_combo.findData(settings.scale_factor)
            if scale_index >= 0:
                self.scale_combo.setCurrentIndex(scale_index)
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._update_denoise_controls_state()

    def _handle_filter_control_change(self, *_args) -> None:
        self._sync_preset_selection()
        self._update_denoise_controls_state()
        self._refresh_preview_impl()

    def _handle_denoise_mode_changed(self, *_args) -> None:
        self._sync_preset_selection()
        self._update_denoise_controls_state()
        self._refresh_preview_impl()

    def _update_denoise_controls_state(self) -> None:
        denoise_mode = str(self.denoise_mode_combo.currentData())
        manual_mode = denoise_mode == "manual"
        sample_mode = denoise_mode == "sample"
        controls_enabled = self.frame_count > 0 and self.denoise_mode_combo.isEnabled()
        self.denoise_strength_slider.setEnabled(controls_enabled and manual_mode)
        self.denoise_color_slider.setEnabled(controls_enabled and manual_mode)
        self.denoise_sample_edit_button.setEnabled(controls_enabled and sample_mode)
        self.denoise_sample_reset_button.setEnabled(controls_enabled and sample_mode)
        if not (controls_enabled and sample_mode) and self.denoise_sample_edit_button.isChecked():
            self.denoise_sample_edit_button.blockSignals(True)
            self.denoise_sample_edit_button.setChecked(False)
            self.denoise_sample_edit_button.blockSignals(False)
        self._sync_denoise_sample_preview()
        self._update_denoise_sample_info_label()
        sample_source = self._denoise_source_description(denoise_mode)
        if manual_mode:
            self.denoise_info_label.setText(
                "Manual: ajuste forca de luminancia, forca de cor e o tamanho das janelas do filtro."
            )
            return

        if self.current_frame is None:
            self.denoise_info_label.setText(sample_source)
            return

        estimated_luma, estimated_color = self._estimate_denoise_strengths(self.current_frame, denoise_mode)
        self.denoise_info_label.setText(
            f"{sample_source} Estimativa atual: luminancia {estimated_luma:.1f} | cor {estimated_color:.1f}."
        )

    def _denoise_source_description(self, denoise_mode: str) -> str:
        if denoise_mode == "auto":
            return "Auto: calcula a intensidade com base no frame atual inteiro."
        if denoise_mode == "sample":
            return "Amostra: usa a regiao azul selecionada no preview como referencia de ruido."
        return ""

    def _sync_preset_selection(self) -> None:
        current_settings = self._current_settings()
        matching_index = 0
        for index in range(1, self.preset_combo.count()):
            preset = self.preset_combo.itemData(index)
            if preset == current_settings:
                matching_index = index
                break

        if self.preset_combo.currentIndex() != matching_index:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(matching_index)
            self.preset_combo.blockSignals(False)

    def _refresh_preview_impl(self) -> None:
        if self.current_frame is None:
            self._refresh_filter_thumbnails()
            return

        settings = self._current_settings()
        self._sync_crop_preview()
        self._sync_denoise_sample_preview()
        processed = self._render_preview_frame(self.current_frame, settings)
        self.preview.set_images(frame_to_qimage(self.current_frame), frame_to_qimage(processed))
        self._refresh_filter_thumbnails()

    def _refresh_preview(self, *_args) -> None:
        self._refresh_preview_impl()

    def _handle_acceleration_mode_changed(self, *_args) -> None:
        if self._gpu_acceleration_enabled():
            self._opencl_failed = False
        self._refresh_preview_impl()

    def _render_preview_frame(self, frame: np.ndarray, settings: FilterSettings) -> np.ndarray:
        return self._render_filtered_frame(frame, settings, self.active_image_filter_id)

    def _render_filtered_frame(
        self,
        frame: np.ndarray,
        settings: FilterSettings,
        image_filter_id: str,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        processing_context = self._build_processing_context(settings, frame)
        processed = self._apply_opencv_filters(frame, processing_context)
        target_intensity = self._image_filter_intensity() if intensity is None else intensity
        return self._apply_image_filter_preset(processed, image_filter_id, target_intensity)

    def _image_filter_intensity(self) -> float:
        return max(0.0, min(1.0, self.active_image_filter_intensity / 100.0))

    def _build_processing_context(
        self,
        settings: FilterSettings,
        reference_frame: Optional[np.ndarray] = None,
    ) -> ProcessingContext:
        denoise_strength: Optional[float] = None
        denoise_color_strength: Optional[float] = None
        if settings.denoise_mode == "manual":
            if settings.denoise_strength > 0:
                denoise_strength = max(1.0, settings.denoise_strength / 7.5)
            if settings.denoise_color_strength > 0:
                denoise_color_strength = max(1.0, settings.denoise_color_strength / 7.5)
        elif reference_frame is not None:
            denoise_strength, denoise_color_strength = self._estimate_denoise_strengths(
                reference_frame,
                settings.denoise_mode,
            )

        if denoise_strength is not None and denoise_color_strength is None:
            denoise_color_strength = denoise_strength
        if denoise_color_strength is not None and denoise_strength is None:
            denoise_strength = denoise_color_strength

        contrast = 1.0 + ((settings.contrast - 50) / 60.0)
        brightness = ((settings.brightness - 50) / 50.0) * 24.0
        gamma = self._gamma_value(settings.gamma)
        gamma_lookup_table: Optional[np.ndarray] = None
        if abs(gamma - 1.0) > 0.01:
            gamma_lookup_table = np.array(
                [np.clip(((index / 255.0) ** gamma) * 255.0, 0, 255) for index in range(256)],
                dtype=np.uint8,
            )

        saturation = self._saturation_value(settings.saturation)
        temperature_shift = self._temperature_shift(settings.temperature)
        sharpen_amount = 0.0
        if settings.sharpen > 0:
            sharpen_amount = (settings.sharpen / 100.0) * 1.8

        return ProcessingContext(
            denoise_strength=denoise_strength,
            denoise_color_strength=denoise_color_strength,
            denoise_template_window=settings.denoise_template_window,
            denoise_search_window=settings.denoise_search_window,
            contrast_alpha=max(0.2, contrast),
            brightness_beta=brightness,
            gamma_lookup_table=gamma_lookup_table,
            saturation=saturation,
            temperature_shift=temperature_shift,
            sharpen_amount=sharpen_amount,
            scale_factor=settings.scale_factor,
        )

    def _estimate_denoise_strengths(
        self,
        frame: np.ndarray,
        denoise_mode: str,
    ) -> tuple[float, float]:
        sample_frame = self._denoise_reference_frame(frame, denoise_mode)
        if sample_frame.size == 0:
            return (2.0, 2.0)

        gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
        luma_std = float(cv2.meanStdDev(gray)[1][0][0])
        lab_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2LAB)
        color_std = float(np.mean(cv2.meanStdDev(lab_frame[:, :, 1:])[1]))
        estimated_luma = float(np.clip(1.0 + (luma_std / 10.0), 1.0, 12.0))
        estimated_color = float(np.clip(0.8 + (color_std / 12.0), 1.0, 12.0))
        return (estimated_luma, estimated_color)

    def _denoise_reference_frame(self, frame: np.ndarray, denoise_mode: str) -> np.ndarray:
        if denoise_mode == "sample":
            sample_region = self._denoise_sample_region(frame)
            if sample_region is not None:
                return sample_region
        return frame

    def _denoise_sample_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        frame_height, frame_width = frame.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return None

        sample_pixels = self._denoise_sample_rect_pixels(frame_width, frame_height)
        if sample_pixels is not None:
            sample_x, sample_y, sample_width, sample_height = sample_pixels
            sample = frame[sample_y : sample_y + sample_height, sample_x : sample_x + sample_width]
            if sample.size > 0:
                return sample

        sample_width = max(32, int(frame_width * 0.35))
        sample_height = max(32, int(frame_height * 0.35))
        start_x = max(0, (frame_width - sample_width) // 2)
        start_y = max(0, (frame_height - sample_height) // 2)
        return frame[start_y : start_y + sample_height, start_x : start_x + sample_width]

    def _gpu_acceleration_enabled(self) -> bool:
        return self.acceleration_combo.currentData() == "gpu"

    def _ensure_opencl_available(self) -> bool:
        if self._opencl_checked:
            return self._opencl_available and not self._opencl_failed

        self._opencl_checked = True
        try:
            if not hasattr(cv2, "ocl"):
                self._opencl_available = False
                return False

            self._opencl_available = bool(cv2.ocl.haveOpenCL())
            if self._opencl_available:
                cv2.ocl.setUseOpenCL(True)
                self._opencl_available = bool(cv2.ocl.useOpenCL())
        except cv2.error:
            self._opencl_available = False

        return self._opencl_available and not self._opencl_failed

    def _current_processing_backend_label(self) -> str:
        if not self._gpu_acceleration_enabled():
            return "CPU"
        if self._opencl_failed:
            return "CPU (fallback do OpenCL)"
        if self._ensure_opencl_available():
            return "OpenCL"
        return "CPU (OpenCL indisponivel)"

    def _apply_opencv_filters(self, frame: np.ndarray, processing_context: ProcessingContext) -> np.ndarray:
        if self._gpu_acceleration_enabled() and self._ensure_opencl_available():
            try:
                return self._apply_opencv_filters_opencl(frame, processing_context)
            except cv2.error:
                self._opencl_failed = True
                if hasattr(cv2, "ocl"):
                    cv2.ocl.setUseOpenCL(False)

        return self._apply_opencv_filters_cpu(frame, processing_context)

    def _apply_image_filter_preset(self, frame: np.ndarray, filter_id: str, intensity: float) -> np.ndarray:
        preset = self._image_filter_preset(filter_id)
        if preset.filter_id == "original" or intensity <= 0.0:
            return frame

        working = frame.astype(np.float32) / 255.0

        if preset.balance_strength > 0:
            channel_means = working.reshape(-1, 3).mean(axis=0)
            target_mean = float(np.mean(channel_means))
            gains = np.clip(target_mean / np.maximum(channel_means, 1e-4), 0.85, 1.18)
            balanced = np.clip(working * gains.reshape(1, 1, 3), 0.0, 1.0)
            working = cv2.addWeighted(working, 1.0 - preset.balance_strength, balanced, preset.balance_strength, 0.0)

        if abs(preset.warmth_shift) > 0.001:
            red_gain = 1.0 + (0.10 * preset.warmth_shift)
            blue_gain = 1.0 - (0.10 * preset.warmth_shift)
            working[:, :, 2] *= red_gain
            working[:, :, 0] *= blue_gain
            working = np.clip(working, 0.0, 1.0)

        if preset.warm_suppression > 0 or preset.cool_suppression > 0:
            hsv_image = cv2.cvtColor((working * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hue = hsv_image[:, :, 0]
            saturation = hsv_image[:, :, 1]

            if preset.warm_suppression > 0:
                warm_mask = np.where((hue <= 28) | (hue >= 166), 1.0, 0.0)
                warm_mask = np.maximum(warm_mask, np.where((hue >= 29) & (hue <= 42), 0.6, 0.0))
                saturation *= 1.0 - (warm_mask * preset.warm_suppression * 0.55)

            if preset.cool_suppression > 0:
                cool_mask = np.where((hue >= 82) & (hue <= 128), 1.0, 0.0)
                cool_mask = np.maximum(cool_mask, np.where((hue >= 68) & (hue <= 81), 0.55, 0.0))
                saturation *= 1.0 - (cool_mask * preset.cool_suppression * 0.55)

            hsv_image[:, :, 1] = np.clip(saturation, 0.0, 255.0)
            working = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

        lab_image = cv2.cvtColor((working * 255.0).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel = lab_image[:, :, 0] / 255.0

        if preset.shadow_lift > 0:
            shadow_mask = np.clip(1.0 - (l_channel * 1.45), 0.0, 1.0)
            l_channel = np.clip(l_channel + (preset.shadow_lift * shadow_mask * 0.26), 0.0, 1.0)

        if preset.highlight_rolloff > 0:
            highlight_mask = np.clip((l_channel - 0.58) / 0.42, 0.0, 1.0)
            l_channel = np.clip(l_channel - (preset.highlight_rolloff * highlight_mask * 0.22), 0.0, 1.0)

        if preset.local_contrast > 0:
            clahe = cv2.createCLAHE(clipLimit=1.4 + (preset.local_contrast * 2.2), tileGridSize=(8, 8))
            contrast_l = clahe.apply((l_channel * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
            blend = min(0.65, preset.local_contrast)
            l_channel = cv2.addWeighted(l_channel, 1.0 - blend, contrast_l, blend, 0.0)

        lab_image[:, :, 0] = np.clip(l_channel * 255.0, 0.0, 255.0)
        filtered = cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2BGR)

        if abs(preset.saturation_boost) > 0.001:
            hsv_image = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1.0 + (preset.saturation_boost * 0.35)), 0.0, 255.0)
            filtered = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if intensity >= 0.999:
            return filtered

        return cv2.addWeighted(frame, 1.0 - intensity, filtered, intensity, 0.0)

    def _apply_opencv_filters_cpu(self, frame: np.ndarray, processing_context: ProcessingContext) -> np.ndarray:
        result = frame.copy()

        if processing_context.denoise_strength is not None:
            result = cv2.fastNlMeansDenoisingColored(
                result,
                None,
                processing_context.denoise_strength,
                processing_context.denoise_color_strength or processing_context.denoise_strength,
                processing_context.denoise_template_window,
                processing_context.denoise_search_window,
            )

        result = cv2.convertScaleAbs(
            result,
            alpha=processing_context.contrast_alpha,
            beta=processing_context.brightness_beta,
        )

        if processing_context.gamma_lookup_table is not None:
            result = cv2.LUT(result, processing_context.gamma_lookup_table)

        if abs(processing_context.saturation - 1.0) > 0.01:
            hsv_image = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * processing_context.saturation, 0, 255)
            result = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

        result = self._apply_temperature_adjustment(result, processing_context.temperature_shift)

        if processing_context.sharpen_amount > 0:
            blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.2)
            result = cv2.addWeighted(
                result,
                1.0 + processing_context.sharpen_amount,
                blurred,
                -processing_context.sharpen_amount,
                0,
            )

        if processing_context.scale_factor > 1.0:
            result = cv2.resize(
                result,
                None,
                fx=processing_context.scale_factor,
                fy=processing_context.scale_factor,
                interpolation=cv2.INTER_CUBIC,
            )

        return result

    def _apply_opencv_filters_opencl(
        self,
        frame: np.ndarray,
        processing_context: ProcessingContext,
    ) -> np.ndarray:
        result = cv2.UMat(frame)

        if processing_context.denoise_strength is not None:
            result = cv2.fastNlMeansDenoisingColored(
                result,
                None,
                processing_context.denoise_strength,
                processing_context.denoise_color_strength or processing_context.denoise_strength,
                processing_context.denoise_template_window,
                processing_context.denoise_search_window,
            )

        result = cv2.convertScaleAbs(
            result,
            alpha=processing_context.contrast_alpha,
            beta=processing_context.brightness_beta,
        )

        if processing_context.gamma_lookup_table is not None:
            result = cv2.LUT(result, processing_context.gamma_lookup_table)

        result_cpu = result.get()

        if abs(processing_context.saturation - 1.0) > 0.01:
            hsv_image = cv2.cvtColor(result_cpu, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * processing_context.saturation, 0, 255)
            result_cpu = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

        result_cpu = self._apply_temperature_adjustment(result_cpu, processing_context.temperature_shift)

        if processing_context.sharpen_amount > 0 or processing_context.scale_factor > 1.0:
            result = cv2.UMat(result_cpu)

            if processing_context.sharpen_amount > 0:
                blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.2)
                result = cv2.addWeighted(
                    result,
                    1.0 + processing_context.sharpen_amount,
                    blurred,
                    -processing_context.sharpen_amount,
                    0,
                )

            if processing_context.scale_factor > 1.0:
                result = cv2.resize(
                    result,
                    None,
                    fx=processing_context.scale_factor,
                    fy=processing_context.scale_factor,
                    interpolation=cv2.INTER_CUBIC,
                )

            return result.get()

        return result_cpu

    def _gamma_value(self, slider_value: int) -> float:
        if slider_value >= 50:
            return 1.0 + ((slider_value - 50) / 50.0)
        return 0.2 + ((slider_value / 50.0) * 0.8)

    def _saturation_value(self, slider_value: int) -> float:
        return max(0.0, slider_value / 50.0)

    def _temperature_shift(self, slider_value: int) -> float:
        return (slider_value - 50) / 50.0

    def _apply_temperature_adjustment(self, frame: np.ndarray, temperature_shift: float) -> np.ndarray:
        if abs(temperature_shift) < 0.01:
            return frame

        adjusted = frame.astype(np.float32)
        red_gain = 1.0 + (0.22 * temperature_shift)
        blue_gain = 1.0 - (0.22 * temperature_shift)
        green_gain = 1.0 - (0.05 * abs(temperature_shift))
        adjusted[:, :, 2] *= red_gain
        adjusted[:, :, 1] *= green_gain
        adjusted[:, :, 0] *= blue_gain
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _export_video(self) -> None:
        if self.video_path is None:
            return

        default_name = self.video_path.with_stem(f"{self.video_path.stem}_enhanced")
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar video processado",
            str(default_name.with_suffix(".mp4")),
            "MP4 (*.mp4)",
        )
        if not output_path:
            return

        settings = self._current_settings()
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            QMessageBox.warning(
                self,
                "FFmpeg nao encontrado",
                "Instale o FFmpeg e deixe o executavel disponivel no PATH para exportar e muxar o audio do video.",
            )
            return

        worker_count = self._export_worker_count()
        encoder_args, encoder_label = self._resolve_export_encoder(ffmpeg_path)
        self._current_export_backend_label = (
            f"{self._current_processing_backend_label()} + {encoder_label}"
        )
        self._begin_export_ui(
            "Exportando video frame a frame com OpenCV... "
            f"usando {worker_count} thread(s) | {self._current_export_backend_label}."
        )
        success, detail = self._export_video_with_opencv(
            Path(output_path),
            settings,
            worker_count,
            encoder_args,
        )
        self._end_export_ui()
        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText("Video exportado com sucesso.")
            QMessageBox.information(self, "Concluido", "Video exportado com sucesso.")
        else:
            self.progress_bar.setValue(0)
            if detail == self.EXPORT_CANCELLED_MESSAGE:
                self.status_label.setText(detail)
            else:
                self.status_label.setText("A exportacao falhou. Revise os parametros e tente novamente.")
                QMessageBox.warning(self, "Falha na exportacao", detail)

    def _begin_export_ui(self, message: str) -> None:
        self._stop_playback()
        self.export_cancel_requested = False
        self._last_export_ui_update = 0.0
        self._set_controls_enabled(False)
        self.select_button.setEnabled(False)
        self.cancel_export_button.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(message)

    def _update_export_progress(
        self,
        processed_frames: int,
        total_frames: int,
        worker_count: int,
        *,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and (now - self._last_export_ui_update) < 0.1:
            return

        progress_value = int((processed_frames / max(1, total_frames)) * 100)
        self.progress_bar.setValue(progress_value)
        self.status_label.setText(
            "Exportando video com OpenCV... "
            f"{progress_value}% | frame {processed_frames}/{total_frames} | "
            f"{worker_count} thread(s) | {self._current_export_backend_label}"
        )
        QApplication.processEvents()
        self._last_export_ui_update = now

    def _end_export_ui(self) -> None:
        self.export_cancel_requested = False
        self._set_controls_enabled(True)
        self.select_button.setEnabled(True)
        self.cancel_export_button.setEnabled(False)

    def _request_export_cancel(self) -> None:
        if self.export_cancel_requested or not self.cancel_export_button.isEnabled():
            return

        self.export_cancel_requested = True
        self.cancel_export_button.setEnabled(False)
        self.status_label.setText("Cancelando exportacao...")

    def _cancel_export_if_requested(self, process: subprocess.Popen[bytes], output_path: Path) -> bool:
        if not self.export_cancel_requested:
            return False

        if process.stdin is not None:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return True

    def _export_worker_count(self) -> int:
        current_value = self.thread_combo.currentData()
        if isinstance(current_value, int) and current_value in {1, 2, 4, 6, 8}:
            return current_value

        try:
            parsed_value = int(self.thread_combo.currentText())
        except ValueError:
            return 1

        return parsed_value if parsed_value in {1, 2, 4, 6, 8} else 1

    def _process_export_frame(self, frame: np.ndarray, processing_context: ProcessingContext) -> np.ndarray:
        processed = self._apply_opencv_filters(frame, processing_context)
        processed = self._apply_image_filter_preset(
            processed,
            self.active_image_filter_id,
            self._image_filter_intensity(),
        )
        return self._apply_crop_to_frame(processed)

    def _available_ffmpeg_encoders(self, ffmpeg_path: str) -> set[str]:
        if self._ffmpeg_encoder_cache_path == ffmpeg_path and self._ffmpeg_encoder_cache is not None:
            return self._ffmpeg_encoder_cache

        encoders: set[str] = set()
        try:
            result = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0].startswith("V"):
                    encoders.add(parts[1])
        except (OSError, subprocess.SubprocessError):
            encoders = set()

        self._ffmpeg_encoder_cache = encoders
        self._ffmpeg_encoder_cache_path = ffmpeg_path
        return encoders

    def _resolve_export_encoder(self, ffmpeg_path: str) -> tuple[list[str], str]:
        if not self._gpu_acceleration_enabled():
            return (["-c:v", "libx264", "-preset", "medium", "-crf", "18"], "libx264")

        available_encoders = self._available_ffmpeg_encoders(ffmpeg_path)
        if "h264_nvenc" in available_encoders and self._probe_ffmpeg_encoder(ffmpeg_path, "h264_nvenc"):
            return (
                ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19", "-b:v", "0"],
                "NVIDIA NVENC",
            )

        vaapi_device = Path("/dev/dri/renderD128")
        if (
            "h264_vaapi" in available_encoders
            and vaapi_device.exists()
            and self._probe_ffmpeg_encoder(ffmpeg_path, "h264_vaapi")
        ):
            return (
                [
                    "-vaapi_device",
                    str(vaapi_device),
                    "-vf",
                    "format=nv12,hwupload",
                    "-c:v",
                    "h264_vaapi",
                    "-qp",
                    "18",
                ],
                "VAAPI",
            )

        return (
            ["-c:v", "libx264", "-preset", "medium", "-crf", "18"],
            "libx264 (fallback)",
        )

    def _probe_ffmpeg_encoder(self, ffmpeg_path: str, encoder_name: str) -> bool:
        cache_key = (ffmpeg_path, encoder_name)
        if cache_key in self._ffmpeg_encoder_probe_cache:
            return self._ffmpeg_encoder_probe_cache[cache_key]

        command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=0.1",
            "-frames:v",
            "1",
            "-an",
        ]

        if encoder_name == "h264_nvenc":
            command.extend(["-c:v", "h264_nvenc", "-f", "null", "-"])
        elif encoder_name == "h264_vaapi":
            command.extend(
                [
                    "-vaapi_device",
                    "/dev/dri/renderD128",
                    "-vf",
                    "format=nv12,hwupload",
                    "-c:v",
                    "h264_vaapi",
                    "-f",
                    "null",
                    "-",
                ]
            )
        else:
            self._ffmpeg_encoder_probe_cache[cache_key] = False
            return False

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
            ok = result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False

        self._ffmpeg_encoder_probe_cache[cache_key] = ok
        return ok

    def _export_video_with_opencv(
        self,
        output_path: Path,
        settings: FilterSettings,
        worker_count: int,
        encoder_args: list[str],
    ) -> tuple[bool, str]:
        if self.video_path is None:
            return (False, "Nenhum video foi selecionado.")

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            return (
                False,
                "Instale o FFmpeg e deixe o executavel disponivel no PATH para exportar o video.",
            )

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            return (False, "Nao foi possivel reabrir o video para exportacao.")

        start_frame, _end_frame = self._selected_frame_range()
        total_frames = self._selected_frame_count()
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            return (False, "Nao foi possivel ler o frame inicial do corte selecionado.")

        processing_context = self._build_processing_context(settings, frame)
        first_processed = self._process_export_frame(frame, processing_context)
        height, width = first_processed.shape[:2]
        duration_seconds = self._selected_duration_seconds()
        start_seconds = self._selected_start_seconds()
        command = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "pipe:0",
            "-ss",
            f"{start_seconds:.6f}",
            "-t",
            f"{duration_seconds:.6f}",
            "-i",
            str(self.video_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            *encoder_args,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        worker_count = max(1, worker_count)
        processed_frames = 0
        submitted_frames = 1
        read_failed = False
        last_error = ""
        previous_cv_threads: Optional[int] = None
        try:
            if worker_count > 1 and hasattr(cv2, "setNumThreads") and hasattr(cv2, "getNumThreads"):
                previous_cv_threads = cv2.getNumThreads()
                cv2.setNumThreads(1)

            if process.stdin is None:
                raise BrokenPipeError("FFmpeg stdin indisponivel.")

            process.stdin.write(first_processed.tobytes())
            processed_frames = 1
            self._update_export_progress(processed_frames, total_frames, worker_count, force=True)
            if self._cancel_export_if_requested(process, output_path):
                return (False, self.EXPORT_CANCELLED_MESSAGE)

            if worker_count == 1:
                while processed_frames < total_frames:
                    ok, next_frame = capture.read()
                    if not ok:
                        read_failed = True
                        break

                    processed_frame = self._process_export_frame(next_frame, processing_context)
                    if process.stdin is None:
                        raise BrokenPipeError("FFmpeg stdin indisponivel.")
                    process.stdin.write(processed_frame.tobytes())
                    processed_frames += 1
                    submitted_frames += 1
                    self._update_export_progress(processed_frames, total_frames, worker_count)
                    if self._cancel_export_if_requested(process, output_path):
                        return (False, self.EXPORT_CANCELLED_MESSAGE)
            else:
                max_pending = max(1, worker_count * 2)
                pending_futures: deque[Future[np.ndarray]] = deque()
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    while processed_frames < total_frames:
                        while (
                            not read_failed
                            and submitted_frames < total_frames
                            and len(pending_futures) < max_pending
                        ):
                            ok, next_frame = capture.read()
                            if not ok:
                                read_failed = True
                                break
                            pending_futures.append(
                                executor.submit(self._process_export_frame, next_frame, processing_context)
                            )
                            submitted_frames += 1

                        if not pending_futures:
                            break

                        processed_frame = pending_futures.popleft().result()
                        if process.stdin is None:
                            raise BrokenPipeError("FFmpeg stdin indisponivel.")
                        process.stdin.write(processed_frame.tobytes())
                        processed_frames += 1
                        self._update_export_progress(processed_frames, total_frames, worker_count)
                        if self._cancel_export_if_requested(process, output_path):
                            return (False, self.EXPORT_CANCELLED_MESSAGE)

            self._update_export_progress(processed_frames, total_frames, worker_count, force=True)
            if self._cancel_export_if_requested(process, output_path):
                return (False, self.EXPORT_CANCELLED_MESSAGE)
            if process.stdin is not None:
                process.stdin.close()
            stderr_output = b""
            if process.stderr is not None:
                stderr_output = process.stderr.read()
            return_code = process.wait()
            last_error = stderr_output.decode("utf-8", errors="ignore").strip()
        except BrokenPipeError:
            stderr_output = b""
            if process.stderr is not None:
                stderr_output = process.stderr.read()
            return_code = process.wait()
            last_error = stderr_output.decode("utf-8", errors="ignore").strip()
        finally:
            if previous_cv_threads is not None and hasattr(cv2, "setNumThreads"):
                cv2.setNumThreads(previous_cv_threads)
            capture.release()

        if return_code != 0:
            return (False, last_error or "O FFmpeg falhou ao receber os frames processados pelo OpenCV.")
        if read_failed or processed_frames != total_frames:
            return (False, "A leitura do corte terminou antes do numero esperado de frames.")
        return (True, "")

    def _denoise_sample_rect_pixels(self, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
        if width <= 0 or height <= 0:
            return None

        x = int(round(self.denoise_sample_rect_norm.x() * width))
        y = int(round(self.denoise_sample_rect_norm.y() * height))
        sample_width = int(round(self.denoise_sample_rect_norm.width() * width))
        sample_height = int(round(self.denoise_sample_rect_norm.height() * height))

        x = max(0, min(x, width - 2))
        y = max(0, min(y, height - 2))
        sample_width = max(2, min(sample_width, width - x))
        sample_height = max(2, min(sample_height, height - y))
        return (x, y, sample_width, sample_height)

    def _crop_rect_pixels(self, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
        if not self._crop_enabled() or width <= 0 or height <= 0:
            return None

        x = int(round(self.crop_rect_norm.x() * width))
        y = int(round(self.crop_rect_norm.y() * height))
        crop_width = int(round(self.crop_rect_norm.width() * width))
        crop_height = int(round(self.crop_rect_norm.height() * height))

        x = max(0, min(x, width - 2))
        y = max(0, min(y, height - 2))
        crop_width = max(2, min(crop_width, width - x))
        crop_height = max(2, min(crop_height, height - y))

        if crop_width % 2 == 1 and crop_width > 2:
            crop_width -= 1
        if crop_height % 2 == 1 and crop_height > 2:
            crop_height -= 1
        if x % 2 == 1 and x > 0:
            x -= 1
        if y % 2 == 1 and y > 0:
            y -= 1

        crop_width = min(crop_width, width - x)
        crop_height = min(crop_height, height - y)
        if crop_width >= width and crop_height >= height and x == 0 and y == 0:
            return None
        return (x, y, crop_width, crop_height)

    def _apply_crop_to_frame(self, frame: np.ndarray) -> np.ndarray:
        crop_pixels = self._crop_rect_pixels(frame.shape[1], frame.shape[0])
        if crop_pixels is None:
            return frame

        x, y, crop_width, crop_height = crop_pixels
        return frame[y : y + crop_height, x : x + crop_width]


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())