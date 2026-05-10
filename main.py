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
    QCheckBox,
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
    stabilizationChanged = Signal(object)
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
        self._stabilization_enabled = False
        self._stabilization_editing = False
        self._stabilization_rect_norm = QRectF(0.4, 0.3, 0.2, 0.2)
        self._stabilization_drag_mode: Optional[str] = None
        self._stabilization_drag_offset = QPointF()
        self._stabilization_anchor_norm = QPointF()
        self._base_size = QSize(self.DEFAULT_BASE_SIZE)
        self._zoom_factor = 1.0
        self._smooth_transform_enabled = True
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
        self._zoom_factor = max(0.25, min(8.0, zoom_factor))
        self.updateGeometry()
        self.resize(self.sizeHint())
        self.update()

    def set_render_quality(self, smooth_transform_enabled: bool) -> None:
        self._smooth_transform_enabled = smooth_transform_enabled
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

    def set_stabilization_state(
        self,
        enabled: bool,
        stabilization_rect_norm: QRectF,
        editing: bool,
    ) -> None:
        self._stabilization_enabled = enabled
        self._stabilization_rect_norm = QRectF(stabilization_rect_norm)
        self._stabilization_editing = editing and enabled
        if not self._stabilization_editing:
            self._stabilization_drag_mode = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._smooth_transform_enabled)
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
        if self._stabilization_enabled:
            self._draw_stabilization_overlay(painter)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._image_rect.contains(event.position()):
            if self._stabilization_editing:
                stabilization_rect = self._display_rect(self._stabilization_rect_norm)
                handle = self._selection_handle_at(event.position(), stabilization_rect)
                if handle is not None:
                    self._stabilization_drag_mode = handle
                    self._stabilization_anchor_norm = self._anchor_for_handle(handle, self._stabilization_rect_norm)
                    self.setCursor(self._cursor_for_handle(handle))
                    return
                if stabilization_rect.contains(event.position()):
                    self._stabilization_drag_mode = "move"
                    pointer_norm = self._point_to_normalized(event.position())
                    self._stabilization_drag_offset = pointer_norm - self._stabilization_rect_norm.topLeft()
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    return

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
        if self._stabilization_editing and self._stabilization_drag_mode is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self._update_stabilization_drag(event.position())
            return

        if self._sample_editing and self._sample_drag_mode is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self._update_sample_drag(event.position())
            return

        if self._crop_enabled and self._crop_drag_mode is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self._update_crop_drag(event.position())
            return

        if self._stabilization_editing:
            stabilization_rect = self._display_rect(self._stabilization_rect_norm)
            handle = self._selection_handle_at(event.position(), stabilization_rect)
            if handle is not None:
                self.setCursor(self._cursor_for_handle(handle))
            elif stabilization_rect.contains(event.position()):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif self._sample_editing:
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
        self._stabilization_drag_mode = None
        if self._stabilization_enabled or self._sample_enabled or self._crop_enabled:
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

    def _draw_stabilization_overlay(self, painter: QPainter) -> None:
        stabilization_rect = self._display_rect(self._stabilization_rect_norm)
        if stabilization_rect.isNull() or stabilization_rect.width() <= 0 or stabilization_rect.height() <= 0:
            return

        outline_color = QColor("#16a34a") if self._stabilization_editing else QColor("#4ade80")
        fill_color = QColor(22, 163, 74, 28)
        painter.fillRect(stabilization_rect, fill_color)
        painter.setPen(QPen(outline_color, 2, Qt.PenStyle.DashLine))
        painter.drawRect(stabilization_rect)
        painter.setPen(QColor("#166534"))
        painter.drawText(
            int(stabilization_rect.left() + 8),
            int(stabilization_rect.top() + 20),
            "Ponto focal",
        )

        if self._stabilization_editing:
            for handle_rect in self._selection_handles(stabilization_rect).values():
                painter.fillRect(handle_rect, QColor("#16a34a"))

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

    def _update_stabilization_drag(self, position: QPointF) -> None:
        pointer_norm = self._point_to_normalized(position)
        if self._stabilization_drag_mode == "move":
            new_rect = QRectF(
                pointer_norm.x() - self._stabilization_drag_offset.x(),
                pointer_norm.y() - self._stabilization_drag_offset.y(),
                self._stabilization_rect_norm.width(),
                self._stabilization_rect_norm.height(),
            )
            new_rect = self._move_rect_within_bounds(new_rect)
        else:
            new_rect = self._resize_rect(
                self._stabilization_drag_mode,
                pointer_norm,
                self._stabilization_anchor_norm,
                None,
            )

        self._stabilization_rect_norm = new_rect
        self.stabilizationChanged.emit(QRectF(self._stabilization_rect_norm))
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
    ROTATION_OPTIONS = [
        ("Sem rotacao", "none"),
        ("90°", "90"),
        ("180°", "180"),
        ("270°", "270"),
    ]
    STABILIZATION_MODE_OPTIONS = [
        ("Desativado", "off"),
        ("Automatico (translacao)", "auto"),
        ("Automatico por similaridade", "similarity"),
        ("Automatico por perspectiva", "perspective"),
        ("Camera lock (regiao)", "camera_lock"),
        ("Hibrido: camera lock + fallback", "hybrid"),
    ]
    DENOISE_MODE_OPTIONS = [
        ("Desativado", "off"),
        ("Manual", "manual"),
        ("Auto", "auto"),
        ("Amostra do preview", "sample"),
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
    PREVIEW_ZOOM_OPTIONS = [50, 75, 100, 125, 150, 200, 300, 400, 600]
    PREVIEW_RENDER_MODE_OPTIONS = [
        ("Alta qualidade", "high"),
        ("Balanceado", "balanced"),
        ("Rapido", "fast"),
    ]
    PREVIEW_FRAME_SKIP_OPTIONS = [
        ("Sem pulo", 0),
        ("Pular 1", 1),
        ("Pular 2", 2),
        ("Pular 3", 3),
    ]
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
        denoise_mode="off",
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
        self.stabilization_rect_norm = QRectF(0.4, 0.3, 0.2, 0.2)
        self.stabilization_reference_frame: Optional[np.ndarray] = None
        self.stabilization_reference_frame_index = 0
        self._preview_auto_stabilization_state = self._new_auto_stabilization_state()
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
        self._preview_refresh_timer = QTimer(self)
        self._preview_refresh_timer.setSingleShot(True)
        self._preview_refresh_timer.timeout.connect(self._refresh_preview_impl)
        self._thumbnail_refresh_timer = QTimer(self)
        self._thumbnail_refresh_timer.setSingleShot(True)
        self._thumbnail_refresh_timer.timeout.connect(self._refresh_filter_thumbnails)

        self.preview = ComparePreview()
        self.preview.cropChanged.connect(self._handle_preview_crop_changed)
        self.preview.sampleChanged.connect(self._handle_preview_sample_changed)
        self.preview.stabilizationChanged.connect(self._handle_preview_stabilization_changed)
        self.preview_zoom_out_button = QPushButton("−")
        self.preview_zoom_in_button = QPushButton("+")
        self.preview_zoom_combo = QComboBox()
        for zoom_percent in self.PREVIEW_ZOOM_OPTIONS:
            self.preview_zoom_combo.addItem(f"{zoom_percent}%", zoom_percent)
        default_zoom_index = self.preview_zoom_combo.findData(100)
        if default_zoom_index >= 0:
            self.preview_zoom_combo.setCurrentIndex(default_zoom_index)
        self.preview_zoom_combo.setToolTip("Define o zoom exibido no preview")
        self.preview_render_mode_combo = QComboBox()
        for label, mode in self.PREVIEW_RENDER_MODE_OPTIONS:
            self.preview_render_mode_combo.addItem(label, mode)
        self.preview_render_mode_combo.setToolTip("Troca qualidade por velocidade apenas no preview")
        self.preview_frame_skip_combo = QComboBox()
        for label, skip_count in self.PREVIEW_FRAME_SKIP_OPTIONS:
            self.preview_frame_skip_combo.addItem(label, skip_count)
        self.preview_frame_skip_combo.setToolTip("Pula frames apenas na reproducao do preview para ficar mais leve")
        self.select_button = QPushButton("Selecionar video")
        self.export_button = QPushButton("Aplicar e exportar")
        self.thread_label = QLabel("Threads")
        self.thread_combo = QComboBox()
        for thread_count in (1, 2, 4, 6, 8):
            self.thread_combo.addItem(str(thread_count), thread_count)
        self.thread_combo.setCurrentIndex(0)
        self.thread_combo.setToolTip("Escolha quantas threads usar na exportacao")
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
        self.stabilization_mode_combo = QComboBox()
        for label, mode in self.STABILIZATION_MODE_OPTIONS:
            self.stabilization_mode_combo.addItem(label, mode)
        self.stabilization_intensity_slider = LabeledSlider("Intensidade da estabilizacao", 0, 100, 65)
        self.stabilization_smoothing_slider = LabeledSlider("Suavizacao temporal", 0, 100, 35)
        self.stabilization_edit_button = QPushButton("Editar ponto/regiao no preview")
        self.stabilization_edit_button.setCheckable(True)
        self.stabilization_reset_button = QPushButton("Resetar ponto focal")
        self.stabilization_auto_crop_checkbox = QCheckBox("Crop automatico para esconder bordas")
        self.stabilization_auto_crop_checkbox.setChecked(True)
        self.stabilization_zoom_out_checkbox = QCheckBox("Zoom out automatico para preservar bordas")
        self.stabilization_zoom_out_checkbox.setChecked(False)
        self.stabilization_info_label = QLabel()
        self.stabilization_info_label.setWordWrap(True)
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
        self.rotation_combo = QComboBox()
        for label, rotation_mode in self.ROTATION_OPTIONS:
            self.rotation_combo.addItem(label, rotation_mode)
        self.flip_horizontal_checkbox = QCheckBox("Espelhar horizontalmente")
        self.flip_vertical_checkbox = QCheckBox("Espelhar verticalmente")
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
        preview_toolbar.addWidget(QLabel("Qualidade"))
        preview_toolbar.addWidget(self.preview_render_mode_combo)
        preview_toolbar.addWidget(QLabel("Playback"))
        preview_toolbar.addWidget(self.preview_frame_skip_combo)
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
        self.settings_tabs.addTab(self._build_stabilization_tab(), "Estabilizacao")
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

    def _build_stabilization_tab(self) -> QScrollArea:
        stabilization_group = QGroupBox("Estabilizacao")
        stabilization_layout = QVBoxLayout(stabilization_group)
        stabilization_layout.setSpacing(14)

        stabilization_form = QFormLayout()
        stabilization_form.addRow("Modo", self.stabilization_mode_combo)
        stabilization_layout.addLayout(stabilization_form)
        stabilization_layout.addWidget(self.stabilization_intensity_slider)
        stabilization_layout.addWidget(self.stabilization_smoothing_slider)
        stabilization_layout.addWidget(self.stabilization_edit_button)
        stabilization_layout.addWidget(self.stabilization_reset_button)
        stabilization_layout.addWidget(self.stabilization_auto_crop_checkbox)
        stabilization_layout.addWidget(self.stabilization_zoom_out_checkbox)
        stabilization_layout.addWidget(self.stabilization_info_label)

        return self._wrap_tab_content(stabilization_group)

    def _build_geometry_tab(self) -> QScrollArea:
        orientation_group = QGroupBox("Orientacao")
        orientation_layout = QVBoxLayout(orientation_group)
        orientation_layout.setSpacing(12)
        orientation_form = QFormLayout()
        orientation_form.addRow("Rotacao", self.rotation_combo)
        orientation_layout.addLayout(orientation_form)
        orientation_layout.addWidget(self.flip_horizontal_checkbox)
        orientation_layout.addWidget(self.flip_vertical_checkbox)

        crop_group = QGroupBox("Enquadramento")
        crop_layout = QVBoxLayout(crop_group)
        crop_layout.setSpacing(14)
        crop_form = QFormLayout()
        crop_form.addRow("Crop", self.crop_mode_combo)
        crop_layout.addLayout(crop_form)
        crop_layout.addWidget(self.crop_reset_button)
        crop_layout.addWidget(self.crop_info_label)

        return self._wrap_tab_content(orientation_group, crop_group)

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
        self.preview_render_mode_combo.currentIndexChanged.connect(self._handle_preview_render_mode_changed)
        self.preview_frame_skip_combo.currentIndexChanged.connect(self._handle_preview_frame_skip_changed)
        self.frame_slider.valueChanged.connect(self._load_frame)
        self.go_first_button.clicked.connect(self._go_to_first_frame)
        self.prev_frame_button.clicked.connect(self._go_to_previous_frame)
        self.play_button.clicked.connect(self._toggle_playback)
        self.next_frame_button.clicked.connect(self._go_to_next_frame)
        self.go_last_button.clicked.connect(self._go_to_last_frame)
        self.preset_combo.currentIndexChanged.connect(self._handle_preset_changed)
        self.rotation_combo.currentIndexChanged.connect(self._handle_filter_control_change)
        self.flip_horizontal_checkbox.toggled.connect(self._handle_filter_control_change)
        self.flip_vertical_checkbox.toggled.connect(self._handle_filter_control_change)
        self.crop_mode_combo.currentIndexChanged.connect(self._handle_crop_mode_changed)
        self.crop_reset_button.clicked.connect(self._reset_crop_rect)
        self.stabilization_mode_combo.currentIndexChanged.connect(self._handle_stabilization_mode_changed)
        self.stabilization_intensity_slider.valueChanged.connect(self._handle_stabilization_control_change)
        self.stabilization_smoothing_slider.valueChanged.connect(self._handle_stabilization_control_change)
        self.stabilization_edit_button.toggled.connect(self._handle_stabilization_edit_toggled)
        self.stabilization_reset_button.clicked.connect(self._reset_stabilization_rect)
        self.stabilization_auto_crop_checkbox.toggled.connect(self._handle_stabilization_control_change)
        self.stabilization_zoom_out_checkbox.toggled.connect(self._handle_stabilization_zoom_out_toggled)
        self.denoise_mode_combo.currentIndexChanged.connect(self._handle_denoise_mode_changed)
        self.denoise_template_combo.currentIndexChanged.connect(self._handle_filter_control_change)
        self.denoise_search_combo.currentIndexChanged.connect(self._handle_filter_control_change)
        self.denoise_sample_edit_button.toggled.connect(self._handle_denoise_sample_edit_toggled)
        self.denoise_sample_reset_button.clicked.connect(self._reset_denoise_sample_rect)
        self.filter_intensity_slider.valueChanged.connect(self._handle_filter_intensity_changed)
        self.scale_combo.currentIndexChanged.connect(self._handle_filter_control_change)
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
            self.stabilization_mode_combo,
            self.stabilization_intensity_slider,
            self.stabilization_smoothing_slider,
            self.stabilization_edit_button,
            self.stabilization_reset_button,
            self.stabilization_auto_crop_checkbox,
            self.stabilization_zoom_out_checkbox,
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
            self.rotation_combo,
            self.flip_horizontal_checkbox,
            self.flip_vertical_checkbox,
            self.crop_mode_combo,
            self.crop_reset_button,
            self.scale_combo,
            self.mark_start_button,
            self.mark_end_button,
            self.reset_range_button,
        ):
            widget.setEnabled(enabled)
        for button in self.filter_preset_buttons.values():
            button.setEnabled(enabled)
        self._update_denoise_controls_state()
        self._update_stabilization_controls_state()
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
        self._opencl_checked = False
        self._opencl_failed = False
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
        self.stabilization_rect_norm = self._default_stabilization_rect()
        self.stabilization_reference_frame = None
        self.stabilization_reference_frame_index = 0
        self._reset_preview_auto_stabilization_state()
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
        self._sync_stabilization_preview()
        self._update_crop_info_label()
        self._update_stabilization_info_label()
        self.status_label.setText("Arraste o divisor na imagem e os sliders ao lado para comparar o resultado.")

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(midpoint)
        self.frame_slider.blockSignals(False)
        self._load_frame(midpoint)

    def _handle_preview_zoom_changed(self, _index: int) -> None:
        zoom_percent = int(self.preview_zoom_combo.currentData() or 100)
        self.preview.set_zoom_factor(zoom_percent / 100.0)
        self._update_preview_zoom_controls()

    def _handle_preview_render_mode_changed(self, _index: int) -> None:
        self.preview.set_render_quality(self._preview_render_mode() != "fast")
        self._reset_preview_auto_stabilization_state()
        self._schedule_preview_refresh(immediate=True)

    def _handle_preview_frame_skip_changed(self, _index: int) -> None:
        if self.playback_timer.isActive():
            self.playback_timer.start(self._playback_interval_ms())

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

    def _preview_render_mode(self) -> str:
        return str(self.preview_render_mode_combo.currentData() or "high")

    def _preview_frame_skip(self) -> int:
        return int(self.preview_frame_skip_combo.currentData() or 0)

    def _preview_processing_scale_limit(self) -> Optional[int]:
        mode = self._preview_render_mode()
        if mode == "fast":
            return 960
        if mode == "balanced":
            return 1280
        return None

    def _preview_output_scale_factor(self, current_scale_factor: float) -> float:
        mode = self._preview_render_mode()
        if mode == "high":
            return current_scale_factor
        if mode == "balanced":
            return min(1.0, current_scale_factor)
        return 1.0

    def _preview_refresh_delay_ms(self) -> int:
        mode = self._preview_render_mode()
        if mode == "fast":
            return 20
        if mode == "balanced":
            return 35
        return 50

    def _schedule_preview_refresh(self, *, immediate: bool = False) -> None:
        if immediate:
            if self._preview_refresh_timer.isActive():
                self._preview_refresh_timer.stop()
            self._refresh_preview_impl()
            return

        self._preview_refresh_timer.start(self._preview_refresh_delay_ms())

    def _schedule_filter_thumbnails_refresh(self) -> None:
        if self.playback_timer.isActive():
            return
        self._thumbnail_refresh_timer.start(140)

    def _prepare_preview_frame(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray], float]:
        size_limit = self._preview_processing_scale_limit()
        if size_limit is None:
            return frame, self.stabilization_reference_frame, 1.0

        height, width = frame.shape[:2]
        largest_dimension = max(width, height)
        if largest_dimension <= 0:
            return frame, self.stabilization_reference_frame, 1.0

        input_scale = min(1.0, size_limit / largest_dimension)
        if input_scale >= 0.999:
            return frame, self.stabilization_reference_frame, 1.0

        preview_frame = cv2.resize(frame, None, fx=input_scale, fy=input_scale, interpolation=cv2.INTER_AREA)
        reference_override = None
        if self.stabilization_reference_frame is not None:
            reference_override = cv2.resize(
                self.stabilization_reference_frame,
                None,
                fx=input_scale,
                fy=input_scale,
                interpolation=cv2.INTER_AREA,
            )
        return preview_frame, reference_override, input_scale

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
        self._schedule_preview_refresh()

    def _handle_filter_intensity_changed(self, value: int) -> None:
        self.active_image_filter_intensity = value
        self._update_filter_description()
        self._schedule_preview_refresh()

    def _current_stabilization_mode(self) -> str:
        return str(self.stabilization_mode_combo.currentData())

    def _stabilization_enabled(self) -> bool:
        return self._current_stabilization_mode() != "off"

    def _stabilization_zoom_out_enabled(self) -> bool:
        return self.stabilization_zoom_out_checkbox.isChecked()

    def _stabilization_uses_focal_point(self) -> bool:
        return self._current_stabilization_mode() in {"camera_lock", "hybrid"}

    def _stabilization_uses_auto_motion(self) -> bool:
        return self._current_stabilization_mode() in {"auto", "similarity", "perspective"}

    def _stabilization_uses_hybrid_motion(self) -> bool:
        return self._current_stabilization_mode() == "hybrid"

    def _stabilization_requires_sequential_processing(self) -> bool:
        return (
            self._stabilization_uses_auto_motion()
            or self._stabilization_uses_hybrid_motion()
            or self._stabilization_temporal_smoothing() > 0.0
        )

    def _new_auto_stabilization_state(self) -> dict[str, object]:
        return {
            "prev_gray": None,
            "prev_frame_index": None,
            "last_shift": None,
            "last_shift_index": None,
            "last_transform": None,
            "last_transform_index": None,
            "smoothed_transform": None,
            "smoothed_frame_index": None,
            "hybrid_fallback_bias": 0.0,
        }

    def _reset_preview_auto_stabilization_state(self) -> None:
        self._preview_auto_stabilization_state = self._new_auto_stabilization_state()

    def _default_stabilization_rect(self) -> QRectF:
        return QRectF(0.4, 0.3, 0.2, 0.2)

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

        if self._thumbnail_refresh_timer.isActive():
            self._thumbnail_refresh_timer.stop()

        if self.current_frame is None:
            for button in self.filter_preset_buttons.values():
                button.setIcon(QIcon())
            return

        settings = replace(self._current_settings(), scale_factor=1.0)
        thumbnail_source = self._thumbnail_source_frame(self.current_frame)
        thumbnail_reference_override = None
        if self.stabilization_reference_frame is not None:
            thumbnail_reference_override = self._thumbnail_source_frame(self.stabilization_reference_frame)
        for preset in self.IMAGE_FILTER_PRESETS:
            thumbnail_frame = self._render_filtered_frame(
                thumbnail_source,
                settings,
                preset.filter_id,
                1.0,
                stabilization_reference_override=thumbnail_reference_override,
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
        self._update_stabilization_controls_state()
        self._update_stabilization_info_label()
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
        frame_step = 1 + self._preview_frame_skip()
        self.frame_slider.setValue(min(self.playback_end_frame, self.frame_slider.value() + frame_step))

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
        self._schedule_preview_refresh()

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
            self._schedule_preview_refresh()

    def _handle_preview_stabilization_changed(self, rect: object) -> None:
        if isinstance(rect, QRectF):
            self.stabilization_rect_norm = QRectF(rect)
            self._capture_stabilization_reference_from_current_frame()
            self._reset_preview_auto_stabilization_state()
            self._sync_stabilization_preview()
            self._update_stabilization_info_label()
            self._schedule_preview_refresh()

    def _reset_crop_rect(self) -> None:
        self.crop_rect_norm = self._default_crop_rect(self._normalized_crop_aspect_ratio())
        self._sync_crop_preview()
        self._update_crop_info_label()
        self._update_denoise_controls_state()
        self._schedule_preview_refresh()

    def _default_denoise_sample_rect(self) -> QRectF:
        return QRectF(0.33, 0.33, 0.34, 0.34)

    def _handle_denoise_sample_edit_toggled(self, checked: bool) -> None:
        if checked and self.stabilization_edit_button.isChecked():
            self.stabilization_edit_button.blockSignals(True)
            self.stabilization_edit_button.setChecked(False)
            self.stabilization_edit_button.blockSignals(False)
            self._sync_stabilization_preview(editing_override=False)
        self._sync_denoise_sample_preview(editing_override=checked)

    def _handle_stabilization_mode_changed(self, *_args) -> None:
        self._reset_preview_auto_stabilization_state()
        if self._stabilization_uses_focal_point():
            self._capture_stabilization_reference_from_current_frame()
        if not self._stabilization_uses_focal_point() and self.stabilization_edit_button.isChecked():
            self.stabilization_edit_button.blockSignals(True)
            self.stabilization_edit_button.setChecked(False)
            self.stabilization_edit_button.blockSignals(False)
        self._update_stabilization_controls_state()
        self._sync_stabilization_preview()
        self._update_stabilization_info_label()
        self._schedule_preview_refresh()

    def _handle_stabilization_edit_toggled(self, checked: bool) -> None:
        if checked and self.denoise_sample_edit_button.isChecked():
            self.denoise_sample_edit_button.blockSignals(True)
            self.denoise_sample_edit_button.setChecked(False)
            self.denoise_sample_edit_button.blockSignals(False)
            self._sync_denoise_sample_preview(editing_override=False)
        if checked:
            self._capture_stabilization_reference_from_current_frame()
        self._sync_stabilization_preview(editing_override=checked)

    def _handle_stabilization_zoom_out_toggled(self, checked: bool) -> None:
        if checked and self.stabilization_auto_crop_checkbox.isChecked():
            self.stabilization_auto_crop_checkbox.blockSignals(True)
            self.stabilization_auto_crop_checkbox.setChecked(False)
            self.stabilization_auto_crop_checkbox.blockSignals(False)
        self._handle_stabilization_control_change()

    def _handle_stabilization_control_change(self, *_args) -> None:
        self._reset_preview_auto_stabilization_state()
        self._update_stabilization_controls_state()
        self._update_stabilization_info_label()
        self._schedule_preview_refresh()

    def _reset_denoise_sample_rect(self) -> None:
        self.denoise_sample_rect_norm = self._default_denoise_sample_rect()
        self._sync_denoise_sample_preview()
        self._update_denoise_sample_info_label()
        self._schedule_preview_refresh()

    def _reset_stabilization_rect(self) -> None:
        self.stabilization_rect_norm = self._default_stabilization_rect()
        self._capture_stabilization_reference_from_current_frame()
        self._reset_preview_auto_stabilization_state()
        self._sync_stabilization_preview()
        self._update_stabilization_info_label()
        self._schedule_preview_refresh()

    def _sync_crop_preview(self) -> None:
        self.preview.set_crop_state(self._crop_enabled(), self.crop_rect_norm, self._normalized_crop_aspect_ratio())

    def _sync_denoise_sample_preview(self, editing_override: Optional[bool] = None) -> None:
        denoise_mode = str(self.denoise_mode_combo.currentData())
        sample_enabled = self.frame_count > 0 and denoise_mode == "sample"
        editing = self.denoise_sample_edit_button.isChecked() if editing_override is None else editing_override
        self.preview.set_sample_state(sample_enabled, self.denoise_sample_rect_norm, editing)

    def _sync_stabilization_preview(self, editing_override: Optional[bool] = None) -> None:
        enabled = self.frame_count > 0 and self._stabilization_uses_focal_point()
        editing = self.stabilization_edit_button.isChecked() if editing_override is None else editing_override
        self.preview.set_stabilization_state(enabled, self.stabilization_rect_norm, editing)

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

    def _capture_stabilization_reference_from_current_frame(self) -> None:
        if self.current_frame is None:
            return
        self.stabilization_reference_frame = self.current_frame.copy()
        self.stabilization_reference_frame_index = self.current_frame_index

    def _update_stabilization_info_label(self) -> None:
        if self.frame_count <= 0:
            self.stabilization_info_label.setText("Estabilizacao: carregue um video para definir o ponto focal.")
            return

        if not self._stabilization_enabled():
            self.stabilization_info_label.setText("Estabilizacao desativada.")
            return

        crop_mode = "ligado" if self.stabilization_auto_crop_checkbox.isChecked() else "desligado"
        zoom_out_mode = "ligado" if self._stabilization_zoom_out_enabled() else "desligado"
        smoothing = int(self._stabilization_temporal_smoothing() * 100)
        mode = self._current_stabilization_mode()
        if mode == "auto":
            self.stabilization_info_label.setText(
                "Automatico por translacao: acompanha o deslocamento global entre frames consecutivos. "
                f"Crop automatico: {crop_mode}. Zoom out: {zoom_out_mode}. Suavizacao temporal: {smoothing}%. Exportacao usa processamento sequencial nesse modo."
            )
            return

        if mode == "similarity":
            self.stabilization_info_label.setText(
                "Automatico por similaridade: compensa translacao, rotacao leve e pequenas variacoes de escala "
                "entre frames consecutivos. "
                f"Crop automatico: {crop_mode}. Zoom out: {zoom_out_mode}. Suavizacao temporal: {smoothing}%. Exportacao usa processamento sequencial nesse modo."
            )
            return

        if mode == "perspective":
            self.stabilization_info_label.setText(
                "Automatico por perspectiva: estima uma homografia global para lidar melhor com parallax leve "
                "e mudancas de perspectiva pequenas. "
                f"Crop automatico: {crop_mode}. Zoom out: {zoom_out_mode}. Suavizacao temporal: {smoothing}%. Exportacao usa processamento sequencial nesse modo."
            )
            return

        region_pixels = self._stabilization_rect_pixels(self.source_size[0], self.source_size[1])
        if region_pixels is None:
            self.stabilization_info_label.setText("Estabilizacao: selecione um ponto ou regiao focal no preview.")
            return

        _, _, region_width, region_height = region_pixels
        reference_frame = self.stabilization_reference_frame_index + 1
        if mode == "hybrid":
            self.stabilization_info_label.setText(
                "Hibrido ativo. Tenta prender a regiao verde com camera lock e, se perder o alvo, faz fallback "
                "progressivo para estabilizacao global. "
                f"Referencia: frame {reference_frame}. Regiao: {region_width}x{region_height}. "
                f"Crop automatico: {crop_mode}. Zoom out: {zoom_out_mode}. Suavizacao temporal: {smoothing}%."
            )
            return

        self.stabilization_info_label.setText(
            "Camera lock ativo. Usa features da regiao verde para estimar uma transformacao robusta e tenta "
            "manter esse enquadramento preso ao frame de referencia. "
            f"Referencia: frame {reference_frame}. Regiao: {region_width}x{region_height}. "
            f"Crop automatico: {crop_mode}. Zoom out: {zoom_out_mode}. Suavizacao temporal: {smoothing}%."
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
            self._schedule_preview_refresh()
            return

        self.active_preset_settings = preset
        self._apply_settings_to_controls(preset)
        self._schedule_preview_refresh()
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
        self._schedule_preview_refresh()
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
        self._schedule_preview_refresh()

    def _handle_denoise_mode_changed(self, *_args) -> None:
        self._sync_preset_selection()
        self._update_denoise_controls_state()
        self._schedule_preview_refresh()

    def _update_denoise_controls_state(self) -> None:
        denoise_mode = str(self.denoise_mode_combo.currentData())
        denoise_off = denoise_mode == "off"
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
        if denoise_off:
            self.denoise_info_label.setText(
                "Denoise desativado: o preview e a exportacao pulam esta etapa para poupar processamento."
            )
            return

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

    def _update_stabilization_controls_state(self) -> None:
        controls_enabled = self.frame_count > 0 and self.stabilization_mode_combo.isEnabled()
        stabilization_enabled = controls_enabled and self._stabilization_enabled()
        self.stabilization_intensity_slider.setEnabled(stabilization_enabled)
        self.stabilization_smoothing_slider.setEnabled(stabilization_enabled)
        zoom_out_enabled = stabilization_enabled and self.stabilization_zoom_out_checkbox.isChecked()
        self.stabilization_auto_crop_checkbox.setEnabled(stabilization_enabled and not zoom_out_enabled)
        self.stabilization_zoom_out_checkbox.setEnabled(stabilization_enabled)
        focal_mode = stabilization_enabled and self._stabilization_uses_focal_point()
        self.stabilization_edit_button.setEnabled(focal_mode)
        self.stabilization_reset_button.setEnabled(focal_mode)
        if not focal_mode and self.stabilization_edit_button.isChecked():
            self.stabilization_edit_button.blockSignals(True)
            self.stabilization_edit_button.setChecked(False)
            self.stabilization_edit_button.blockSignals(False)
        self._sync_stabilization_preview()
        self._update_stabilization_info_label()

    def _denoise_source_description(self, denoise_mode: str) -> str:
        if denoise_mode == "off":
            return "Desativado: nao aplica reducao de ruido."
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
            self._schedule_filter_thumbnails_refresh()
            return

        settings = self._current_settings()
        self._sync_crop_preview()
        self._sync_denoise_sample_preview()
        self._sync_stabilization_preview()
        preview_source, _reference_override, _input_scale = self._prepare_preview_frame(self.current_frame)
        preview_before = self._apply_geometry_transforms(preview_source)
        processed = self._render_preview_frame(self.current_frame, settings)
        self.preview.set_images(frame_to_qimage(preview_before), frame_to_qimage(processed))
        self._schedule_filter_thumbnails_refresh()

    def _refresh_preview(self, *_args) -> None:
        self._schedule_preview_refresh(immediate=True)

    def _render_preview_frame(self, frame: np.ndarray, settings: FilterSettings) -> np.ndarray:
        preview_frame, reference_override, _input_scale = self._prepare_preview_frame(frame)
        preview_settings = replace(
            settings,
            scale_factor=self._preview_output_scale_factor(settings.scale_factor),
        )
        return self._render_filtered_frame(
            preview_frame,
            preview_settings,
            self.active_image_filter_id,
            stabilization_frame_index=self.current_frame_index,
            stabilization_state=self._preview_auto_stabilization_state,
            stabilization_reference_override=reference_override,
        )

    def _render_filtered_frame(
        self,
        frame: np.ndarray,
        settings: FilterSettings,
        image_filter_id: str,
        intensity: Optional[float] = None,
        stabilization_frame_index: Optional[int] = None,
        stabilization_state: Optional[dict[str, object]] = None,
        stabilization_reference_override: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        stabilized_frame = self._apply_stabilization(
            frame,
            stabilization_frame_index,
            stabilization_state,
            stabilization_reference_override,
        )
        processing_context = self._build_processing_context(settings, stabilized_frame)
        processed = self._apply_opencv_filters(stabilized_frame, processing_context)
        target_intensity = self._image_filter_intensity() if intensity is None else intensity
        processed = self._apply_image_filter_preset(processed, image_filter_id, target_intensity)
        return self._apply_geometry_transforms(processed)

    def _image_filter_intensity(self) -> float:
        return max(0.0, min(1.0, self.active_image_filter_intensity / 100.0))

    def _current_rotation_mode(self) -> str:
        return str(self.rotation_combo.currentData() or "none")

    def _apply_geometry_transforms(self, frame: np.ndarray) -> np.ndarray:
        transformed = frame
        rotation_mode = self._current_rotation_mode()
        if rotation_mode == "90":
            transformed = cv2.rotate(transformed, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_mode == "180":
            transformed = cv2.rotate(transformed, cv2.ROTATE_180)
        elif rotation_mode == "270":
            transformed = cv2.rotate(transformed, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.flip_horizontal_checkbox.isChecked():
            transformed = cv2.flip(transformed, 1)
        if self.flip_vertical_checkbox.isChecked():
            transformed = cv2.flip(transformed, 0)
        return transformed

    def _stabilization_intensity(self) -> float:
        return max(0.0, min(1.0, self.stabilization_intensity_slider.value() / 100.0))

    def _stabilization_temporal_smoothing(self) -> float:
        return max(0.0, min(1.0, self.stabilization_smoothing_slider.value() / 100.0))

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
        elif settings.denoise_mode in {"auto", "sample"} and reference_frame is not None:
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

    def _apply_stabilization(
        self,
        frame: np.ndarray,
        frame_index: Optional[int] = None,
        runtime_state: Optional[dict[str, object]] = None,
        reference_frame_override: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self._stabilization_enabled():
            return frame

        intensity = self._stabilization_intensity()
        if intensity <= 0.0:
            return frame

        transform = self._estimate_stabilization_transform(
            frame,
            frame_index,
            runtime_state,
            reference_frame_override,
        )
        if transform is None:
            return frame

        transform = self._smooth_stabilization_transform(transform, frame_index, runtime_state)

        blended_transform = self._blend_stabilization_transform(transform, intensity)
        if self._stabilization_zoom_out_enabled():
            blended_transform = self._fit_stabilization_transform_with_zoom_out(blended_transform, frame.shape)
        if self._stabilization_transform_is_negligible(blended_transform):
            return frame

        stabilized = self._warp_stabilized_frame(frame, blended_transform)

        if self.stabilization_auto_crop_checkbox.isChecked():
            stabilized = self._auto_crop_stabilized_frame(stabilized, blended_transform)
        return stabilized

    def _estimate_stabilization_transform(
        self,
        frame: np.ndarray,
        frame_index: Optional[int],
        runtime_state: Optional[dict[str, object]],
        reference_frame_override: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        mode = self._current_stabilization_mode()
        if mode == "auto":
            if runtime_state is None:
                return None
            shift = self._estimate_automatic_stabilization_shift(frame, frame_index, runtime_state)
            if shift is None:
                return None
            return self._translation_to_affine(shift[0], shift[1])

        if mode in {"similarity", "perspective"}:
            if runtime_state is None:
                return None
            return self._estimate_automatic_geometric_transform(frame, frame_index, runtime_state, mode)

        if mode == "camera_lock":
            reference_frame = self.stabilization_reference_frame if reference_frame_override is None else reference_frame_override
            if reference_frame is None:
                return None
            return self._estimate_camera_lock_transform(frame, reference_frame)

        if mode == "hybrid":
            reference_frame = self.stabilization_reference_frame if reference_frame_override is None else reference_frame_override
            return self._estimate_hybrid_stabilization_transform(frame, frame_index, runtime_state, reference_frame)

        return None

    def _estimate_hybrid_stabilization_transform(
        self,
        frame: np.ndarray,
        frame_index: Optional[int],
        runtime_state: Optional[dict[str, object]],
        reference_frame: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if runtime_state is None:
            return None

        fallback_bias = float(runtime_state.get("hybrid_fallback_bias", 0.0) or 0.0)
        lock_transform = None
        if reference_frame is not None:
            lock_transform = self._estimate_camera_lock_transform(frame, reference_frame)

        similarity_transform = self._estimate_automatic_geometric_transform(frame, frame_index, runtime_state, "similarity")
        translation_shift = self._estimate_automatic_stabilization_shift(frame, frame_index, runtime_state)
        translation_transform = None
        if translation_shift is not None:
            translation_transform = self._translation_to_affine(translation_shift[0], translation_shift[1])

        if lock_transform is not None:
            fallback_bias = max(0.0, fallback_bias - 0.22)
            runtime_state["hybrid_fallback_bias"] = fallback_bias
            if similarity_transform is not None and fallback_bias > 0.01:
                return self._mix_transforms(lock_transform, similarity_transform, min(0.35, fallback_bias * 0.5))
            return lock_transform

        fallback_bias = min(1.0, fallback_bias + 0.28)
        runtime_state["hybrid_fallback_bias"] = fallback_bias

        if similarity_transform is not None and translation_transform is not None and fallback_bias > 0.55:
            translation_weight = min(0.55, (fallback_bias - 0.55) / 0.45)
            return self._mix_transforms(similarity_transform, translation_transform, translation_weight)
        if similarity_transform is not None:
            return similarity_transform
        return translation_transform

    def _smooth_stabilization_transform(
        self,
        transform: np.ndarray,
        frame_index: Optional[int],
        runtime_state: Optional[dict[str, object]],
    ) -> np.ndarray:
        smoothing = self._stabilization_temporal_smoothing()
        if runtime_state is None or smoothing <= 0.0:
            return transform

        previous_transform = runtime_state.get("smoothed_transform")
        previous_frame_index = runtime_state.get("smoothed_frame_index")
        if (
            previous_transform is None
            or not isinstance(previous_transform, np.ndarray)
            or frame_index is None
            or previous_frame_index is None
            or int(previous_frame_index) != (frame_index - 1)
        ):
            runtime_state["smoothed_transform"] = transform
            runtime_state["smoothed_frame_index"] = frame_index
            return transform

        current_weight = max(0.08, 1.0 - (smoothing * 0.92))
        smoothed_transform = self._mix_transforms(previous_transform, transform, current_weight)
        runtime_state["smoothed_transform"] = smoothed_transform
        runtime_state["smoothed_frame_index"] = frame_index
        return smoothed_transform

    def _automatic_stabilization_gray(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            return np.empty((0, 0), dtype=np.uint8), 1.0

        scale = min(1.0, 480.0 / max(width, height))
        if scale < 0.999:
            resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            resized = frame
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray, scale

    def _estimate_automatic_stabilization_shift(
        self,
        frame: np.ndarray,
        frame_index: Optional[int],
        runtime_state: dict[str, object],
    ) -> Optional[tuple[float, float]]:
        current_gray, scale = self._automatic_stabilization_gray(frame)
        if current_gray.size == 0:
            return None

        previous_gray = runtime_state.get("prev_gray")
        previous_frame_index = runtime_state.get("prev_frame_index")
        last_shift_index = runtime_state.get("last_shift_index")
        last_shift = runtime_state.get("last_shift")

        if frame_index is not None and last_shift_index == frame_index and isinstance(last_shift, tuple):
            return last_shift

        if previous_gray is None or not isinstance(previous_gray, np.ndarray):
            runtime_state["prev_gray"] = current_gray
            runtime_state["prev_frame_index"] = frame_index
            runtime_state["last_shift"] = None
            runtime_state["last_shift_index"] = frame_index
            return None

        if frame_index is not None and previous_frame_index is not None and frame_index != (int(previous_frame_index) + 1):
            runtime_state["prev_gray"] = current_gray
            runtime_state["prev_frame_index"] = frame_index
            runtime_state["last_shift"] = None
            runtime_state["last_shift_index"] = frame_index
            return None

        features = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=160,
            qualityLevel=0.01,
            minDistance=18,
            blockSize=5,
        )
        shift: Optional[tuple[float, float]] = None
        if features is not None and len(features) >= 8:
            next_points, status, _errors = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, features, None)
            if next_points is not None and status is not None:
                valid_mask = status.reshape(-1) == 1
                if np.count_nonzero(valid_mask) >= 6:
                    valid_previous = features.reshape(-1, 2)[valid_mask]
                    valid_next = next_points.reshape(-1, 2)[valid_mask]
                    deltas = valid_next - valid_previous
                    delta_x = float(np.median(deltas[:, 0]))
                    delta_y = float(np.median(deltas[:, 1]))
                    shift = (-delta_x / scale, -delta_y / scale)

        if shift is None:
            phase_shift, response = cv2.phaseCorrelate(previous_gray.astype(np.float32), current_gray.astype(np.float32))
            if response >= 0.05:
                shift = (-float(phase_shift[0]) / scale, -float(phase_shift[1]) / scale)

        runtime_state["prev_gray"] = current_gray
        runtime_state["prev_frame_index"] = frame_index
        runtime_state["last_shift"] = shift
        runtime_state["last_shift_index"] = frame_index
        return shift

    def _estimate_automatic_geometric_transform(
        self,
        frame: np.ndarray,
        frame_index: Optional[int],
        runtime_state: dict[str, object],
        mode: str,
    ) -> Optional[np.ndarray]:
        current_gray, scale = self._automatic_stabilization_gray(frame)
        if current_gray.size == 0:
            return None

        previous_gray = runtime_state.get("prev_gray")
        previous_frame_index = runtime_state.get("prev_frame_index")
        last_transform = runtime_state.get("last_transform")
        last_transform_index = runtime_state.get("last_transform_index")

        if frame_index is not None and last_transform_index == frame_index and isinstance(last_transform, np.ndarray):
            return last_transform

        if previous_gray is None or not isinstance(previous_gray, np.ndarray):
            runtime_state["prev_gray"] = current_gray
            runtime_state["prev_frame_index"] = frame_index
            runtime_state["last_transform"] = None
            runtime_state["last_transform_index"] = frame_index
            return None

        if frame_index is not None and previous_frame_index is not None and frame_index != (int(previous_frame_index) + 1):
            runtime_state["prev_gray"] = current_gray
            runtime_state["prev_frame_index"] = frame_index
            runtime_state["last_transform"] = None
            runtime_state["last_transform_index"] = frame_index
            return None

        transform = self._estimate_geometric_transform_from_gray(
            previous_gray,
            current_gray,
            scale,
            frame.shape,
            mode,
        )

        runtime_state["prev_gray"] = current_gray
        runtime_state["prev_frame_index"] = frame_index
        runtime_state["last_transform"] = transform
        runtime_state["last_transform_index"] = frame_index
        return transform

    def _estimate_geometric_transform_from_gray(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        scale: float,
        frame_shape: tuple[int, ...],
        mode: str,
    ) -> Optional[np.ndarray]:
        features = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=220,
            qualityLevel=0.01,
            minDistance=14,
            blockSize=5,
        )
        if features is None or len(features) < 10:
            return None

        next_points, status, _errors = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            features,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None:
            return None

        valid_mask = status.reshape(-1) == 1
        if np.count_nonzero(valid_mask) < 8:
            return None

        previous_points = features.reshape(-1, 2)[valid_mask]
        current_points = next_points.reshape(-1, 2)[valid_mask]

        if mode == "perspective":
            perspective_transform = self._estimate_perspective_transform(
                current_points,
                previous_points,
                scale,
                frame_shape,
            )
            if perspective_transform is not None:
                return perspective_transform

        return self._estimate_similarity_transform(
            current_points,
            previous_points,
            scale,
            frame_shape,
        )

    def _estimate_similarity_transform(
        self,
        source_points: np.ndarray,
        destination_points: np.ndarray,
        scale: float,
        frame_shape: tuple[int, ...],
    ) -> Optional[np.ndarray]:
        if len(source_points) < 6 or len(destination_points) < 6:
            return None

        transform, inliers = cv2.estimateAffinePartial2D(
            source_points,
            destination_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=3000,
            confidence=0.99,
            refineIters=10,
        )
        if transform is None or inliers is None or int(np.count_nonzero(inliers)) < 6:
            return None

        scaled_transform = self._rescale_affine_transform(transform, scale)
        if not self._stabilization_transform_is_reasonable(scaled_transform, frame_shape):
            return None
        return scaled_transform

    def _estimate_perspective_transform(
        self,
        source_points: np.ndarray,
        destination_points: np.ndarray,
        scale: float,
        frame_shape: tuple[int, ...],
    ) -> Optional[np.ndarray]:
        if len(source_points) < 8 or len(destination_points) < 8:
            return None

        transform, inliers = cv2.findHomography(
            source_points,
            destination_points,
            cv2.RANSAC,
            3.0,
        )
        if transform is None or inliers is None or int(np.count_nonzero(inliers)) < 8:
            return None

        scaled_transform = self._rescale_homography(transform, scale)
        if not self._stabilization_transform_is_reasonable(scaled_transform, frame_shape, allow_perspective=True):
            return None
        return scaled_transform

    def _estimate_camera_lock_transform(
        self,
        frame: np.ndarray,
        reference_frame: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        reference_source = self.stabilization_reference_frame if reference_frame is None else reference_frame
        if reference_source is None:
            return None

        reference_height, reference_width = reference_source.shape[:2]
        current_height, current_width = frame.shape[:2]
        reference_pixels = self._stabilization_rect_pixels(reference_width, reference_height)
        current_pixels = self._stabilization_rect_pixels(current_width, current_height)
        if reference_pixels is None or current_pixels is None:
            return None

        scale = min(1.0, 720.0 / max(reference_width, reference_height, current_width, current_height))
        reference_gray = self._scaled_gray_frame(reference_source, scale)
        current_gray = self._scaled_gray_frame(frame, scale)
        if reference_gray.size == 0 or current_gray.size == 0:
            return None

        reference_mask = np.zeros_like(reference_gray, dtype=np.uint8)
        current_mask = np.zeros_like(current_gray, dtype=np.uint8)
        ref_x, ref_y, ref_width, ref_height = self._scaled_rect(reference_pixels, scale, reference_gray.shape[1], reference_gray.shape[0])
        cur_search_rect = self._expanded_search_rect(current_pixels, current_width, current_height, 2.6)
        cur_x, cur_y, cur_width, cur_height = self._scaled_rect(cur_search_rect, scale, current_gray.shape[1], current_gray.shape[0])
        reference_mask[ref_y : ref_y + ref_height, ref_x : ref_x + ref_width] = 255
        current_mask[cur_y : cur_y + cur_height, cur_x : cur_x + cur_width] = 255

        transform = self._estimate_camera_lock_feature_transform(
            reference_gray,
            current_gray,
            reference_mask,
            current_mask,
            scale,
            frame.shape,
        )
        if transform is not None:
            return transform

        transform = self._estimate_camera_lock_feature_transform(
            reference_gray,
            current_gray,
            reference_mask,
            None,
            scale,
            frame.shape,
        )
        if transform is not None:
            return transform

        shift = self._estimate_stabilization_shift(frame, reference_source)
        if shift is None:
            return None
        return self._translation_to_affine(shift[0], shift[1])

    def _estimate_camera_lock_feature_transform(
        self,
        reference_gray: np.ndarray,
        current_gray: np.ndarray,
        reference_mask: np.ndarray,
        current_mask: Optional[np.ndarray],
        scale: float,
        frame_shape: tuple[int, ...],
    ) -> Optional[np.ndarray]:
        orb = cv2.ORB_create(
            nfeatures=900,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=10,
        )
        reference_keypoints, reference_descriptors = orb.detectAndCompute(reference_gray, reference_mask)
        current_keypoints, current_descriptors = orb.detectAndCompute(current_gray, current_mask)
        if (
            reference_descriptors is None
            or current_descriptors is None
            or reference_keypoints is None
            or current_keypoints is None
            or len(reference_keypoints) < 8
            or len(current_keypoints) < 8
        ):
            return None

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = matcher.knnMatch(reference_descriptors, current_descriptors, k=2)
        good_matches = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < (0.78 * second.distance):
                good_matches.append(first)

        if len(good_matches) < 8:
            return None

        good_matches.sort(key=lambda match: match.distance)
        good_matches = good_matches[:120]
        reference_points = np.float32([reference_keypoints[match.queryIdx].pt for match in good_matches])
        current_points = np.float32([current_keypoints[match.trainIdx].pt for match in good_matches])
        return self._estimate_similarity_transform(current_points, reference_points, scale, frame_shape)

    def _estimate_stabilization_shift(
        self,
        frame: np.ndarray,
        reference_frame: Optional[np.ndarray] = None,
    ) -> Optional[tuple[float, float]]:
        reference_source = self.stabilization_reference_frame if reference_frame is None else reference_frame
        if reference_source is None:
            return None

        reference_height, reference_width = reference_source.shape[:2]
        current_height, current_width = frame.shape[:2]
        reference_pixels = self._stabilization_rect_pixels(reference_width, reference_height)
        current_pixels = self._stabilization_rect_pixels(current_width, current_height)
        if reference_pixels is None or current_pixels is None:
            return None

        ref_x, ref_y, ref_width, ref_height = reference_pixels
        cur_x, cur_y, cur_width, cur_height = current_pixels
        template = reference_source[ref_y : ref_y + ref_height, ref_x : ref_x + ref_width]
        if template.size == 0:
            return None

        search_margin_x = max(32, int(cur_width * 1.6))
        search_margin_y = max(32, int(cur_height * 1.6))
        search_left = max(0, cur_x - search_margin_x)
        search_top = max(0, cur_y - search_margin_y)
        search_right = min(current_width, cur_x + cur_width + search_margin_x)
        search_bottom = min(current_height, cur_y + cur_height + search_margin_y)
        search_region = frame[search_top:search_bottom, search_left:search_right]
        if (
            search_region.shape[0] < template.shape[0]
            or search_region.shape[1] < template.shape[1]
        ):
            return None

        search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_loc = cv2.minMaxLoc(result)
        if max_value < 0.28:
            return None

        match_x = search_left + max_loc[0]
        match_y = search_top + max_loc[1]
        reference_center_x = ref_x + (ref_width / 2.0)
        reference_center_y = ref_y + (ref_height / 2.0)
        match_center_x = match_x + (ref_width / 2.0)
        match_center_y = match_y + (ref_height / 2.0)
        return (reference_center_x - match_center_x, reference_center_y - match_center_y)

    def _translation_to_affine(self, shift_x: float, shift_y: float) -> np.ndarray:
        return np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)

    def _scaled_gray_frame(self, frame: np.ndarray, scale: float) -> np.ndarray:
        if scale < 0.999:
            resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            resized = frame
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def _scaled_rect(
        self,
        rect: tuple[int, int, int, int],
        scale: float,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x, y, rect_width, rect_height = rect
        scaled_x = max(0, min(int(round(x * scale)), max(0, width - 2)))
        scaled_y = max(0, min(int(round(y * scale)), max(0, height - 2)))
        scaled_width = max(2, min(int(round(rect_width * scale)), width - scaled_x))
        scaled_height = max(2, min(int(round(rect_height * scale)), height - scaled_y))
        return (scaled_x, scaled_y, scaled_width, scaled_height)

    def _expanded_search_rect(
        self,
        rect: tuple[int, int, int, int],
        width: int,
        height: int,
        multiplier: float,
    ) -> tuple[int, int, int, int]:
        x, y, rect_width, rect_height = rect
        margin_x = max(40, int(rect_width * multiplier))
        margin_y = max(40, int(rect_height * multiplier))
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(width, x + rect_width + margin_x)
        bottom = min(height, y + rect_height + margin_y)
        return (left, top, max(2, right - left), max(2, bottom - top))

    def _affine_to_homography(self, transform: np.ndarray) -> np.ndarray:
        if transform.shape == (3, 3):
            return transform.astype(np.float32)
        return np.vstack([transform, np.array([0.0, 0.0, 1.0], dtype=np.float32)]).astype(np.float32)

    def _rescale_affine_transform(self, transform: np.ndarray, scale: float) -> np.ndarray:
        if abs(scale - 1.0) < 1e-3:
            return transform.astype(np.float32)
        homography = self._affine_to_homography(transform)
        scaled = self._rescale_homography(homography, scale)
        return scaled[:2, :].astype(np.float32)

    def _rescale_homography(self, transform: np.ndarray, scale: float) -> np.ndarray:
        homography = self._affine_to_homography(transform)
        if abs(scale - 1.0) < 1e-3:
            normalized = homography.astype(np.float32)
        else:
            scaling_matrix = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            inverse_scaling = np.array(
                [[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            normalized = inverse_scaling @ homography @ scaling_matrix
        if abs(float(normalized[2, 2])) > 1e-6:
            normalized = normalized / float(normalized[2, 2])
        return normalized.astype(np.float32)

    def _blend_stabilization_transform(self, transform: np.ndarray, intensity: float) -> np.ndarray:
        if transform.shape == (2, 3):
            identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            return ((identity * (1.0 - intensity)) + (transform.astype(np.float32) * intensity)).astype(np.float32)

        identity = np.eye(3, dtype=np.float32)
        blended = (identity * (1.0 - intensity)) + (transform.astype(np.float32) * intensity)
        if abs(float(blended[2, 2])) > 1e-6:
            blended = blended / float(blended[2, 2])
        return blended.astype(np.float32)

    def _mix_transforms(
        self,
        first_transform: np.ndarray,
        second_transform: np.ndarray,
        second_weight: float,
    ) -> np.ndarray:
        weight = max(0.0, min(1.0, second_weight))
        if first_transform.shape == (2, 3) and second_transform.shape == (2, 3):
            return (
                (first_transform.astype(np.float32) * (1.0 - weight))
                + (second_transform.astype(np.float32) * weight)
            ).astype(np.float32)

        first_h = self._affine_to_homography(first_transform)
        second_h = self._affine_to_homography(second_transform)
        mixed = (first_h * (1.0 - weight)) + (second_h * weight)
        if abs(float(mixed[2, 2])) > 1e-6:
            mixed = mixed / float(mixed[2, 2])
        return mixed.astype(np.float32)

    def _stabilization_transform_is_negligible(self, transform: np.ndarray) -> bool:
        if transform.shape == (2, 3):
            translation = float(max(abs(transform[0, 2]), abs(transform[1, 2])))
            rotation_scale_delta = float(np.max(np.abs(transform[:, :2] - np.eye(2, dtype=np.float32))))
            return translation < 0.5 and rotation_scale_delta < 0.004

        translation = float(max(abs(transform[0, 2]), abs(transform[1, 2])))
        matrix_delta = float(np.max(np.abs(transform - np.eye(3, dtype=np.float32))))
        return translation < 0.5 and matrix_delta < 0.006

    def _stabilization_transform_is_reasonable(
        self,
        transform: np.ndarray,
        frame_shape: tuple[int, ...],
        allow_perspective: bool = False,
    ) -> bool:
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return False

        corners = np.array(
            [[[0.0, 0.0]], [[width - 1.0, 0.0]], [[width - 1.0, height - 1.0]], [[0.0, height - 1.0]]],
            dtype=np.float32,
        )
        if transform.shape == (2, 3):
            projected = cv2.transform(corners, transform)
        else:
            projected = cv2.perspectiveTransform(corners, transform)
        if not np.isfinite(projected).all():
            return False

        projected_points = projected.reshape(-1, 2)
        original_points = corners.reshape(-1, 2)
        displacement = np.linalg.norm(projected_points - original_points, axis=1)
        if float(np.median(displacement)) > (max(width, height) * 0.35):
            return False

        bbox_width = float(projected_points[:, 0].max() - projected_points[:, 0].min())
        bbox_height = float(projected_points[:, 1].max() - projected_points[:, 1].min())
        if bbox_width < (width * 0.55) or bbox_width > (width * 1.45):
            return False
        if bbox_height < (height * 0.55) or bbox_height > (height * 1.45):
            return False

        if allow_perspective:
            projected_area = abs(float(cv2.contourArea(projected_points.astype(np.float32))))
            original_area = float(max(1, width * height))
            if projected_area < (original_area * 0.45) or projected_area > (original_area * 1.8):
                return False

        return True

    def _fit_stabilization_transform_with_zoom_out(
        self,
        transform: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> np.ndarray:
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return transform

        corners = np.array(
            [[[0.0, 0.0]], [[width - 1.0, 0.0]], [[width - 1.0, height - 1.0]], [[0.0, height - 1.0]]],
            dtype=np.float32,
        )
        if transform.shape == (2, 3):
            projected = cv2.transform(corners, transform)
        else:
            projected = cv2.perspectiveTransform(corners, transform)

        if not np.isfinite(projected).all():
            return transform

        projected_points = projected.reshape(-1, 2)
        min_x = float(projected_points[:, 0].min())
        max_x = float(projected_points[:, 0].max())
        min_y = float(projected_points[:, 1].min())
        max_y = float(projected_points[:, 1].max())
        bbox_width = max(1.0, max_x - min_x)
        bbox_height = max(1.0, max_y - min_y)

        safe_width = max(1.0, width * 0.98)
        safe_height = max(1.0, height * 0.98)
        if min_x >= 0.0 and min_y >= 0.0 and max_x <= (width - 1.0) and max_y <= (height - 1.0):
            if bbox_width <= safe_width and bbox_height <= safe_height:
                return transform

        fit_scale = min(1.0, safe_width / bbox_width, safe_height / bbox_height)
        bbox_center_x = (min_x + max_x) / 2.0
        bbox_center_y = (min_y + max_y) / 2.0
        output_center_x = (width - 1.0) / 2.0
        output_center_y = (height - 1.0) / 2.0
        adjust = np.array(
            [
                [fit_scale, 0.0, output_center_x - (bbox_center_x * fit_scale)],
                [0.0, fit_scale, output_center_y - (bbox_center_y * fit_scale)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        composed = adjust @ self._affine_to_homography(transform)
        if abs(float(composed[2, 2])) > 1e-6:
            composed = composed / float(composed[2, 2])
        if transform.shape == (2, 3):
            return composed[:2, :].astype(np.float32)
        return composed.astype(np.float32)

    def _warp_stabilized_frame(self, frame: np.ndarray, transform: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if transform.shape == (2, 3):
            return cv2.warpAffine(
                frame,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        return cv2.warpPerspective(
            frame,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _auto_crop_stabilized_frame(self, frame: np.ndarray, transform: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        mask = np.full((height, width), 255, dtype=np.uint8)
        if transform.shape == (2, 3):
            valid_mask = cv2.warpAffine(
                mask,
                transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            valid_mask = cv2.warpPerspective(
                mask,
                transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        erosion_size = max(1, int(round(min(width, height) * 0.015)))
        if erosion_size > 1:
            kernel = np.ones((erosion_size, erosion_size), dtype=np.uint8)
            valid_mask = cv2.erode(valid_mask, kernel, iterations=1)

        non_zero = cv2.findNonZero(valid_mask)
        if non_zero is None:
            return frame

        left, top, cropped_width, cropped_height = cv2.boundingRect(non_zero)
        if cropped_width >= (width - 2) and cropped_height >= (height - 2):
            return frame
        if cropped_width < max(16, int(width * 0.55)) or cropped_height < max(16, int(height * 0.55)):
            return frame

        cropped = frame[top : top + cropped_height, left : left + cropped_width]
        if cropped.size == 0:
            return frame
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)

    def _gpu_acceleration_enabled(self) -> bool:
        return True

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
        if self._stabilization_requires_sequential_processing():
            worker_count = 1
        self._opencl_checked = False
        self._opencl_failed = False
        encoder_args, encoder_label = self._resolve_export_encoder(ffmpeg_path)
        self._current_export_backend_label = (
            f"{self._current_processing_backend_label()} + {encoder_label}"
        )
        export_note = ""
        if self._stabilization_requires_sequential_processing():
            export_note = " | estabilizacao temporal usa processamento sequencial"
        self._begin_export_ui(
            "Exportando video frame a frame com OpenCV... "
            f"usando {worker_count} thread(s) | {self._current_export_backend_label}{export_note}."
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

    def _process_export_frame(
        self,
        frame: np.ndarray,
        processing_context: ProcessingContext,
        frame_index: Optional[int] = None,
        stabilization_state: Optional[dict[str, object]] = None,
    ) -> np.ndarray:
        processed = self._apply_stabilization(frame, frame_index, stabilization_state)
        processed = self._apply_opencv_filters(processed, processing_context)
        processed = self._apply_image_filter_preset(
            processed,
            self.active_image_filter_id,
            self._image_filter_intensity(),
        )
        processed = self._apply_crop_to_frame(processed)
        return self._apply_geometry_transforms(processed)

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

        export_stabilization_state = self._new_auto_stabilization_state()
        processing_context = self._build_processing_context(settings, frame)
        first_processed = self._process_export_frame(frame, processing_context, start_frame, export_stabilization_state)
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

                    next_frame_index = start_frame + processed_frames
                    processed_frame = self._process_export_frame(
                        next_frame,
                        processing_context,
                        next_frame_index,
                        export_stabilization_state,
                    )
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
                                executor.submit(
                                    self._process_export_frame,
                                    next_frame,
                                    processing_context,
                                    start_frame + submitted_frames,
                                    None,
                                )
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

    def _stabilization_rect_pixels(self, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
        if width <= 0 or height <= 0:
            return None

        x = int(round(self.stabilization_rect_norm.x() * width))
        y = int(round(self.stabilization_rect_norm.y() * height))
        region_width = int(round(self.stabilization_rect_norm.width() * width))
        region_height = int(round(self.stabilization_rect_norm.height() * height))

        x = max(0, min(x, width - 2))
        y = max(0, min(y, height - 2))
        region_width = max(2, min(region_width, width - x))
        region_height = max(2, min(region_height, height - y))
        return (x, y, region_width, region_height)

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