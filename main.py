from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QStyle,
    QStyleOptionSlider,
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
    denoise: int
    sharpen: int
    contrast: int
    brightness: int
    gamma: int
    saturation: int
    temperature: int
    scale_factor: float


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
        self.setMinimumSize(760, 440)
        self.setCursor(Qt.CursorShape.SizeHorCursor)

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

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._crop_enabled and event.button() == Qt.MouseButton.LeftButton and self._image_rect.contains(event.position()):
            crop_rect = self._display_crop_rect()
            handle = self._crop_handle_at(event.position(), crop_rect)
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
        if self._crop_enabled and self._crop_drag_mode is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self._update_crop_drag(event.position())
            return

        if self._crop_enabled:
            crop_rect = self._display_crop_rect()
            handle = self._crop_handle_at(event.position(), crop_rect)
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
        if self._crop_enabled:
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

        for handle_rect in self._crop_handles(crop_rect).values():
            painter.fillRect(handle_rect, QColor("#f59e0b"))

    def _display_crop_rect(self) -> QRectF:
        if self._image_rect.width() <= 0 or self._image_rect.height() <= 0:
            return QRectF()
        return QRectF(
            self._image_rect.left() + (self._crop_rect_norm.x() * self._image_rect.width()),
            self._image_rect.top() + (self._crop_rect_norm.y() * self._image_rect.height()),
            self._crop_rect_norm.width() * self._image_rect.width(),
            self._crop_rect_norm.height() * self._image_rect.height(),
        )

    def _crop_handles(self, crop_rect: QRectF) -> dict[str, QRectF]:
        size = 10.0
        half = size / 2.0
        return {
            "top_left": QRectF(crop_rect.left() - half, crop_rect.top() - half, size, size),
            "top_right": QRectF(crop_rect.right() - half, crop_rect.top() - half, size, size),
            "bottom_left": QRectF(crop_rect.left() - half, crop_rect.bottom() - half, size, size),
            "bottom_right": QRectF(crop_rect.right() - half, crop_rect.bottom() - half, size, size),
        }

    def _crop_handle_at(self, position: QPointF, crop_rect: QRectF) -> Optional[str]:
        for handle, handle_rect in self._crop_handles(crop_rect).items():
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
            new_rect = self._move_crop_rect_within_bounds(new_rect)
        else:
            new_rect = self._resize_crop_rect(self._crop_drag_mode, pointer_norm)

        self._crop_rect_norm = new_rect
        self.cropChanged.emit(QRectF(self._crop_rect_norm))
        self.update()

    def _move_crop_rect_within_bounds(self, rect: QRectF) -> QRectF:
        x = min(max(0.0, rect.x()), 1.0 - rect.width())
        y = min(max(0.0, rect.y()), 1.0 - rect.height())
        return QRectF(x, y, rect.width(), rect.height())

    def _resize_crop_rect(self, handle: str, pointer_norm: QPointF) -> QRectF:
        anchor = self._crop_anchor_norm
        min_size = 0.08
        dx = pointer_norm.x() - anchor.x()
        dy = pointer_norm.y() - anchor.y()
        sign_x = 1.0 if dx >= 0 else -1.0
        sign_y = 1.0 if dy >= 0 else -1.0
        width = max(min_size, abs(dx))
        height = max(min_size, abs(dy))

        if self._crop_aspect_ratio is not None:
            if width / max(height, 1e-6) > self._crop_aspect_ratio:
                width = height * self._crop_aspect_ratio
            else:
                height = width / self._crop_aspect_ratio

            max_width = anchor.x() if sign_x < 0 else (1.0 - anchor.x())
            max_height = anchor.y() if sign_y < 0 else (1.0 - anchor.y())
            width = min(width, max_width)
            height = min(height, max_height)
            if width / max(height, 1e-6) > self._crop_aspect_ratio:
                width = height * self._crop_aspect_ratio
            else:
                height = width / self._crop_aspect_ratio

        x2 = anchor.x() + (sign_x * width)
        y2 = anchor.y() + (sign_y * height)
        left = min(anchor.x(), x2)
        top = min(anchor.y(), y2)
        rect = QRectF(left, top, abs(x2 - anchor.x()), abs(y2 - anchor.y()))
        return self._move_crop_rect_within_bounds(rect)


class MainWindow(QMainWindow):
    EXPORT_CANCELLED_MESSAGE = "Exportacao cancelada pelo usuario."
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
    DEFAULT_FILTER_SETTINGS = FilterSettings(20, 25, 50, 50, 50, 50, 50, 1.0)
    FILTER_PRESETS = [
        ("Personalizado", None),
        ("Limpeza leve", FilterSettings(12, 12, 52, 50, 50, 50, 50, 1.0)),
        ("Limpeza clara", FilterSettings(25, 15, 50, 50, 35, 40, 50, 1.0)),
        ("Balanceado 1080p", FilterSettings(22, 24, 54, 51, 50, 52, 50, 1.5)),
        ("Recuperacao forte", FilterSettings(38, 32, 56, 52, 52, 54, 50, 1.5)),
        ("Mais detalhe 2x", FilterSettings(24, 38, 55, 50, 50, 55, 50, 2.0)),
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
        self.playback_end_frame = 0
        self.crop_rect_norm = QRectF(0.1, 0.1, 0.8, 0.8)
        self.export_cancel_requested = False
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_playback_frame)

        self.preview = ComparePreview()
        self.preview.cropChanged.connect(self._handle_preview_crop_changed)
        self.select_button = QPushButton("Selecionar video")
        self.export_button = QPushButton("Aplicar e exportar")
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

        self.denoise_slider = LabeledSlider("Denoise", 0, 100, self.DEFAULT_FILTER_SETTINGS.denoise)
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
        self._bind_events()
        self._set_controls_enabled(False)
        self.select_button.setEnabled(True)
        self.file_label.setEnabled(True)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        left_column = QVBoxLayout()
        left_column.setSpacing(16)

        header = QHBoxLayout()
        header.setSpacing(12)
        header.addWidget(self.select_button)
        header.addWidget(self.export_button)
        header.addWidget(self.cancel_export_button)
        header.addWidget(self.file_label, 1)

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

        range_row = QHBoxLayout()
        range_row.setSpacing(10)
        range_row.addWidget(self.mark_start_button)
        range_row.addWidget(self.mark_end_button)
        range_row.addWidget(self.reset_range_button)
        range_row.addWidget(self.range_label, 1)

        left_column.addLayout(header)
        left_column.addWidget(self.preview, 1)
        left_column.addLayout(frame_row)
        left_column.addLayout(transport_row)
        left_column.addLayout(range_row)

        controls_box = QGroupBox("Ajustes")
        controls_layout = QVBoxLayout(controls_box)
        controls_layout.setSpacing(14)
        controls_layout.addWidget(self.denoise_slider)
        controls_layout.addWidget(self.sharpen_slider)
        controls_layout.addWidget(self.contrast_slider)
        controls_layout.addWidget(self.brightness_slider)
        controls_layout.addWidget(self.gamma_slider)
        controls_layout.addWidget(self.saturation_slider)
        controls_layout.addWidget(self.temperature_slider)

        scale_form = QFormLayout()
        scale_form.addRow("Preset", self.preset_combo)
        scale_form.addRow("Crop", self.crop_mode_combo)
        scale_form.addRow("Escala de saida", self.scale_combo)
        controls_layout.addLayout(scale_form)
        controls_layout.addWidget(self.crop_reset_button)
        controls_layout.addWidget(self.crop_info_label)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.status_label)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setWidget(controls_box)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)
        layout.addLayout(left_column, 3)
        layout.addWidget(controls_scroll, 1)

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

    def _bind_events(self) -> None:
        self.select_button.clicked.connect(self._select_video)
        self.export_button.clicked.connect(self._export_video)
        self.cancel_export_button.clicked.connect(self._request_export_cancel)
        self.frame_slider.valueChanged.connect(self._load_frame)
        self.go_first_button.clicked.connect(self._go_to_first_frame)
        self.prev_frame_button.clicked.connect(self._go_to_previous_frame)
        self.play_button.clicked.connect(self._toggle_playback)
        self.next_frame_button.clicked.connect(self._go_to_next_frame)
        self.go_last_button.clicked.connect(self._go_to_last_frame)
        self.preset_combo.currentIndexChanged.connect(self._handle_preset_changed)
        self.crop_mode_combo.currentIndexChanged.connect(self._handle_crop_mode_changed)
        self.crop_reset_button.clicked.connect(self._reset_crop_rect)
        self.scale_combo.currentIndexChanged.connect(self._handle_filter_control_change)
        self.mark_start_button.clicked.connect(self._mark_clip_start)
        self.mark_end_button.clicked.connect(self._mark_clip_end)
        self.reset_range_button.clicked.connect(self._reset_clip_range)

        for control in (
            self.denoise_slider,
            self.sharpen_slider,
            self.contrast_slider,
            self.brightness_slider,
            self.gamma_slider,
            self.saturation_slider,
            self.temperature_slider,
        ):
            control.valueChanged.connect(self._handle_filter_control_change)

        self.denoise_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("denoise"))
        self.sharpen_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("sharpen"))
        self.contrast_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("contrast"))
        self.brightness_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("brightness"))
        self.gamma_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("gamma"))
        self.saturation_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("saturation"))
        self.temperature_slider.resetRequested.connect(lambda: self._reset_slider_to_preset("temperature"))

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.export_button,
            self.frame_slider,
            self.go_first_button,
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.go_last_button,
            self.denoise_slider,
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
            self.mark_start_button,
            self.mark_end_button,
            self.reset_range_button,
        ):
            widget.setEnabled(enabled)

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
        self.current_frame_index = 0
        self.clip_start_frame = 0
        self.clip_end_frame = self.frame_count - 1
        self.crop_rect_norm = self._default_crop_rect(self._normalized_crop_aspect_ratio())
        self.file_label.setText(f"{path.name} | {width}x{height} @ {self.fps:.2f} fps")
        self.frame_slider.setRange(0, self.frame_count - 1)
        midpoint = min(self.frame_count - 1, self.frame_count // 2)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self._set_controls_enabled(True)
        self._update_frame_slider_markers()
        self._update_range_label()
        self._sync_crop_preview()
        self._update_crop_info_label()
        self.status_label.setText("Arraste o divisor na imagem e os sliders ao lado para comparar o resultado.")

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(midpoint)
        self.frame_slider.blockSignals(False)
        self._load_frame(midpoint)

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
        if self.playback_timer.isActive():
            self.playback_timer.stop()
        self.play_button.setText("⏵")

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
            denoise=self.denoise_slider.value(),
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

    def _reset_crop_rect(self) -> None:
        self.crop_rect_norm = self._default_crop_rect(self._normalized_crop_aspect_ratio())
        self._sync_crop_preview()
        self._update_crop_info_label()
        self._refresh_preview_impl()

    def _sync_crop_preview(self) -> None:
        self.preview.set_crop_state(self._crop_enabled(), self.crop_rect_norm, self._normalized_crop_aspect_ratio())

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
            "denoise": self.denoise_slider,
            "sharpen": self.sharpen_slider,
            "contrast": self.contrast_slider,
            "brightness": self.brightness_slider,
            "gamma": self.gamma_slider,
            "saturation": self.saturation_slider,
            "temperature": self.temperature_slider,
        }
        slider = slider_map[field_name]
        slider.setValue(target_value)

    def _apply_settings_to_controls(self, settings: FilterSettings) -> None:
        widgets = [
            self.denoise_slider,
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
            self.denoise_slider.slider.setValue(settings.denoise)
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

    def _handle_filter_control_change(self, *_args) -> None:
        self._sync_preset_selection()
        self._refresh_preview_impl()

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
            return

        settings = self._current_settings()
        self._sync_crop_preview()
        processed = self._render_preview_frame(self.current_frame, settings)
        self.preview.set_images(frame_to_qimage(self.current_frame), frame_to_qimage(processed))

    def _refresh_preview(self, *_args) -> None:
        self._refresh_preview_impl()

    def _render_preview_frame(self, frame: np.ndarray, settings: FilterSettings) -> np.ndarray:
        return self._apply_opencv_filters(frame, settings)

    def _apply_opencv_filters(self, frame: np.ndarray, settings: FilterSettings) -> np.ndarray:
        result = frame.copy()

        if settings.denoise > 0:
            strength = max(1.0, settings.denoise / 7.5)
            result = cv2.fastNlMeansDenoisingColored(result, None, strength, strength, 7, 21)

        contrast = 1.0 + ((settings.contrast - 50) / 60.0)
        brightness = ((settings.brightness - 50) / 50.0) * 24.0
        result = cv2.convertScaleAbs(result, alpha=max(0.2, contrast), beta=brightness)

        gamma = self._gamma_value(settings.gamma)
        if abs(gamma - 1.0) > 0.01:
            lookup_table = np.array(
                [np.clip(((index / 255.0) ** gamma) * 255.0, 0, 255) for index in range(256)],
                dtype=np.uint8,
            )
            result = cv2.LUT(result, lookup_table)

        saturation = self._saturation_value(settings.saturation)
        if abs(saturation - 1.0) > 0.01:
            hsv_image = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation, 0, 255)
            result = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

        result = self._apply_temperature_adjustment(result, settings.temperature)

        if settings.sharpen > 0:
            amount = (settings.sharpen / 100.0) * 1.8
            blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.2)
            result = cv2.addWeighted(result, 1.0 + amount, blurred, -amount, 0)

        if settings.scale_factor > 1.0:
            result = cv2.resize(
                result,
                None,
                fx=settings.scale_factor,
                fy=settings.scale_factor,
                interpolation=cv2.INTER_CUBIC,
            )

        return result

    def _gamma_value(self, slider_value: int) -> float:
        if slider_value >= 50:
            return 1.0 + ((slider_value - 50) / 50.0)
        return 0.2 + ((slider_value / 50.0) * 0.8)

    def _saturation_value(self, slider_value: int) -> float:
        return max(0.0, slider_value / 50.0)

    def _temperature_shift(self, slider_value: int) -> float:
        return (slider_value - 50) / 50.0

    def _apply_temperature_adjustment(self, frame: np.ndarray, slider_value: int) -> np.ndarray:
        temperature_shift = self._temperature_shift(slider_value)
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
        self._begin_export_ui(
            f"Exportando video frame a frame com OpenCV... usando {worker_count} thread(s)."
        )
        success, detail = self._export_video_with_opencv(Path(output_path), settings, worker_count)
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
        self._set_controls_enabled(False)
        self.select_button.setEnabled(False)
        self.cancel_export_button.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(message)

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
        return 1

    def _process_export_frame(self, frame: np.ndarray, settings: FilterSettings) -> np.ndarray:
        processed = self._apply_opencv_filters(frame, settings)
        return self._apply_crop_to_frame(processed)

    def _export_video_with_opencv(
        self,
        output_path: Path,
        settings: FilterSettings,
        worker_count: int,
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

        first_processed = self._process_export_frame(frame, settings)
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
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
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
        try:
            if process.stdin is None:
                raise BrokenPipeError("FFmpeg stdin indisponivel.")

            process.stdin.write(first_processed.tobytes())
            processed_frames = 1
            self.progress_bar.setValue(int((processed_frames / total_frames) * 100))
            self.status_label.setText(
                f"Exportando video com OpenCV... {self.progress_bar.value()}% | frame {processed_frames}/{total_frames} | {worker_count} thread(s)"
            )
            QApplication.processEvents()
            if self._cancel_export_if_requested(process, output_path):
                return (False, self.EXPORT_CANCELLED_MESSAGE)

            if worker_count == 1:
                while processed_frames < total_frames:
                    ok, next_frame = capture.read()
                    if not ok:
                        read_failed = True
                        break

                    processed_frame = self._process_export_frame(next_frame, settings)
                    if process.stdin is None:
                        raise BrokenPipeError("FFmpeg stdin indisponivel.")
                    process.stdin.write(processed_frame.tobytes())
                    processed_frames += 1
                    submitted_frames += 1
                    progress_value = int((processed_frames / total_frames) * 100)
                    self.progress_bar.setValue(progress_value)
                    self.status_label.setText(
                        f"Exportando video com OpenCV... {progress_value}% | frame {processed_frames}/{total_frames} | {worker_count} thread(s)"
                    )
                    QApplication.processEvents()
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
                                executor.submit(self._process_export_frame, next_frame.copy(), settings)
                            )
                            submitted_frames += 1

                        if not pending_futures:
                            break

                        processed_frame = pending_futures.popleft().result()
                        if process.stdin is None:
                            raise BrokenPipeError("FFmpeg stdin indisponivel.")
                        process.stdin.write(processed_frame.tobytes())
                        processed_frames += 1
                        progress_value = int((processed_frames / total_frames) * 100)
                        self.progress_bar.setValue(progress_value)
                        self.status_label.setText(
                            f"Exportando video com OpenCV... {progress_value}% | frame {processed_frames}/{total_frames} | {worker_count} thread(s)"
                        )
                        QApplication.processEvents()
                        if self._cancel_export_if_requested(process, output_path):
                            return (False, self.EXPORT_CANCELLED_MESSAGE)

            QApplication.processEvents()
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
            capture.release()

        if return_code != 0:
            return (False, last_error or "O FFmpeg falhou ao receber os frames processados pelo OpenCV.")
        if read_failed or processed_frames != total_frames:
            return (False, "A leitura do corte terminou antes do numero esperado de frames.")
        return (True, "")

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