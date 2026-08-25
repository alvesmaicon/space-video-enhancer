"""Main window: builds the widgets, snapshots them into ``RenderSettings`` and drives preview/export.

No image processing lives here. Everything the pipeline needs is read from widgets
exactly once per render into an immutable snapshot (see :mod:`space_video_enhancer.settings`).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Slot
from PySide6.QtGui import QAction, QCloseEvent, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
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
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from space_video_enhancer.export.ffmpeg import FfmpegProbe, find_ffmpeg, resolve_encoder
from space_video_enhancer.export.job import ExportRequest, ExportResult
from space_video_enhancer.export.worker import ExportController
from space_video_enhancer.pipeline.denoise import estimate_denoise_strengths
from space_video_enhancer.pipeline.filters import OpenCLBackend
from space_video_enhancer.pipeline.presets import (
    IMAGE_FILTER_PRESETS,
    ORIGINAL_PRESET_ID,
    ImageFilterPreset,
    get_image_filter_preset,
)
from space_video_enhancer.pipeline.preview import PreviewRequest, ThumbnailRequest
from space_video_enhancer.settings import (
    DEFAULT_CROP_RECT,
    DEFAULT_DENOISE_SAMPLE_RECT,
    DEFAULT_FILTER_SETTINGS,
    DEFAULT_STABILIZATION_RECT,
    FILTER_PRESETS,
    CropSettings,
    DenoiseMode,
    FilterSettings,
    GeometrySettings,
    ImageFilterSelection,
    NormalizedRect,
    PreviewRenderMode,
    RenderSettings,
    RotationMode,
    StabilizationMode,
    StabilizationSettings,
    clamp_unit,
)
from space_video_enhancer.ui.preview_worker import PreviewController, PreviewImages, ThumbnailImages
from space_video_enhancer.ui.theme import load_stylesheet
from space_video_enhancer.ui.widgets.compare_preview import ComparePreview
from space_video_enhancer.ui.widgets.frame_range_slider import FrameRangeSlider
from space_video_enhancer.ui.widgets.labeled_slider import LabeledSlider
from space_video_enhancer.video import VideoOpenError, VideoSource

E = TypeVar("E", DenoiseMode, StabilizationMode, RotationMode, PreviewRenderMode)

THREAD_OPTIONS = (1, 2, 4, 6, 8)
PREVIEW_MAX_FPS = 12.0
PREVIEW_MIN_INTERVAL_MS = 40
THUMBNAIL_REFRESH_DELAY_MS = 140

PREVIEW_SIZE_LIMIT = {PreviewRenderMode.FAST: 960, PreviewRenderMode.BALANCED: 1280}
PREVIEW_REFRESH_DELAY_MS = {PreviewRenderMode.FAST: 20, PreviewRenderMode.BALANCED: 35, PreviewRenderMode.HIGH: 50}


def qrect_to_norm(rect: QRectF) -> NormalizedRect:
    return NormalizedRect(rect.x(), rect.y(), rect.width(), rect.height())


def norm_to_qrect(rect: NormalizedRect) -> QRectF:
    return QRectF(rect.x, rect.y, rect.width, rect.height)


def _default_thread_count() -> int:
    half = max(1, (os.cpu_count() or 2) // 2)
    return max(option for option in THREAD_OPTIONS if option <= half)


class MainWindow(QMainWindow):
    ROTATION_OPTIONS = [
        ("Sem rotacao", RotationMode.NONE),
        ("90°", RotationMode.CW_90),
        ("180°", RotationMode.ROTATE_180),
        ("270°", RotationMode.CW_270),
    ]
    STABILIZATION_MODE_OPTIONS = [
        ("Desativado", StabilizationMode.OFF),
        ("Automatico (translacao)", StabilizationMode.AUTO),
        ("Automatico por similaridade", StabilizationMode.SIMILARITY),
        ("Automatico por perspectiva", StabilizationMode.PERSPECTIVE),
        ("Camera lock (regiao)", StabilizationMode.CAMERA_LOCK),
        ("Hibrido: camera lock + fallback", StabilizationMode.HYBRID),
    ]
    DENOISE_MODE_OPTIONS = [
        ("Desativado", DenoiseMode.OFF),
        ("Manual", DenoiseMode.MANUAL),
        ("Auto", DenoiseMode.AUTO),
        ("Amostra do preview", DenoiseMode.SAMPLE),
    ]
    DENOISE_TEMPLATE_OPTIONS = [3, 5, 7, 9]
    DENOISE_SEARCH_OPTIONS = [11, 15, 21, 27, 35]
    FILTER_THUMBNAIL_SIZE = QSize(152, 86)
    SCALE_OPTIONS = [
        ("1.0x (original)", 1.0),
        ("1.5x (bom para 720p -> 1080p)", 1.5),
        ("2.0x", 2.0),
    ]
    CROP_OPTIONS: list[tuple[str, object | None]] = [
        ("Sem crop", None),
        ("Livre", "free"),
        ("16:9", 16 / 9),
        ("4:3", 4 / 3),
        ("1:1", 1.0),
        ("9:16", 9 / 16),
    ]
    PREVIEW_ZOOM_OPTIONS = [50, 75, 100, 125, 150, 200, 300, 400, 600]
    PREVIEW_RENDER_MODE_OPTIONS = [
        ("Alta qualidade", PreviewRenderMode.HIGH),
        ("Balanceado", PreviewRenderMode.BALANCED),
        ("Rapido", PreviewRenderMode.FAST),
    ]
    PREVIEW_FRAME_SKIP_OPTIONS = [("Sem pulo", 0), ("Pular 1", 1), ("Pular 2", 2), ("Pular 3", 3)]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Space Video Enhancer")
        self.resize(1220, 760)

        # --- media state -------------------------------------------------------------------
        self.video_path: Path | None = None
        self.video: VideoSource | None = None
        self.frame_count = 0
        self.fps = 30.0
        self.source_size = (0, 0)
        self.current_frame_index = 0
        self.clip_start_frame = 0
        self.clip_end_frame = 0
        self.playback_end_frame = 0
        self.current_frame: np.ndarray | None = None

        # --- editing state -----------------------------------------------------------------
        self.active_preset_settings: FilterSettings = DEFAULT_FILTER_SETTINGS
        self.active_image_filter_id = ORIGINAL_PRESET_ID
        self.active_image_filter_intensity = 100
        self.crop_rect_norm = norm_to_qrect(DEFAULT_CROP_RECT)
        self.denoise_sample_rect_norm = norm_to_qrect(DEFAULT_DENOISE_SAMPLE_RECT)
        self.stabilization_rect_norm = norm_to_qrect(DEFAULT_STABILIZATION_RECT)
        self.stabilization_reference_frame: np.ndarray | None = None
        self.stabilization_reference_frame_index = 0
        self._stabilization_epoch = 0  # bumped whenever inter-frame stabilization memory must restart

        # --- processing / export infra -----------------------------------------------------
        self._backend = OpenCLBackend()
        self._preview_controller = PreviewController(self._backend, self.FILTER_THUMBNAIL_SIZE, self)
        self._preview_controller.previewReady.connect(self._on_preview_ready)
        self._preview_controller.thumbnailsReady.connect(self._on_thumbnails_ready)
        self._preview_controller.renderFailed.connect(self._on_render_failed)
        self._preview_generation = 0
        self._applied_preview_generation = -1
        self._thumbnail_generation = 0
        self._applied_thumbnail_generation = -1
        self.last_preview_render_ms = 0.0
        self._ffmpeg_probe: FfmpegProbe | None = None
        self._export_controller: ExportController | None = None
        self._export_backend_label = ""

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_playback_frame)
        self._preview_refresh_timer = QTimer(self)
        self._preview_refresh_timer.setSingleShot(True)
        self._preview_refresh_timer.timeout.connect(self._refresh_preview_impl)
        self._thumbnail_refresh_timer = QTimer(self)
        self._thumbnail_refresh_timer.setSingleShot(True)
        self._thumbnail_refresh_timer.timeout.connect(self._refresh_filter_thumbnails)

        self._create_widgets()
        self._build_ui()
        self._build_menu_bar()
        self._bind_events()
        self._apply_settings_to_controls(DEFAULT_FILTER_SETTINGS)
        self._handle_preview_zoom_changed(self.preview_zoom_combo.currentIndex())
        self._update_denoise_controls_state()
        self._set_controls_enabled(False)
        self.select_button.setEnabled(True)
        self.file_label.setEnabled(True)

    # =====================================================================================
    # Widget construction
    # =====================================================================================
    def _create_widgets(self) -> None:
        self.preview = ComparePreview()
        self.preview_zoom_out_button = QPushButton("−")  # noqa: RUF001 - typographic minus on purpose
        self.preview_zoom_in_button = QPushButton("+")
        self.preview_zoom_combo = QComboBox()
        for zoom_percent in self.PREVIEW_ZOOM_OPTIONS:
            self.preview_zoom_combo.addItem(f"{zoom_percent}%", zoom_percent)
        self.preview_zoom_combo.setCurrentIndex(max(0, self.preview_zoom_combo.findData(100)))
        self.preview_zoom_combo.setToolTip("Define o zoom exibido no preview")
        self.preview_render_mode_combo = self._enum_combo(self.PREVIEW_RENDER_MODE_OPTIONS)
        self.preview_render_mode_combo.setToolTip("Troca qualidade por velocidade apenas no preview")
        self.preview_frame_skip_combo = QComboBox()
        for label, skip_count in self.PREVIEW_FRAME_SKIP_OPTIONS:
            self.preview_frame_skip_combo.addItem(label, skip_count)
        self.preview_frame_skip_combo.setToolTip("Pula frames apenas na reproducao do preview para ficar mais leve")

        self.select_button = QPushButton("Selecionar video")
        self.export_button = QPushButton("Aplicar e exportar")
        self.thread_label = QLabel("Threads")
        self.thread_combo = QComboBox()
        for thread_count in THREAD_OPTIONS:
            self.thread_combo.addItem(str(thread_count), thread_count)
        self.thread_combo.setCurrentIndex(max(0, self.thread_combo.findData(_default_thread_count())))
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

        defaults = DEFAULT_FILTER_SETTINGS
        self.denoise_mode_combo = self._enum_combo(self.DENOISE_MODE_OPTIONS)
        self.denoise_strength_slider = LabeledSlider("Forca de luminancia", 0, 100, defaults.denoise_strength)
        self.denoise_color_slider = LabeledSlider("Forca de cor", 0, 100, defaults.denoise_color_strength)
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

        self.stabilization_mode_combo = self._enum_combo(self.STABILIZATION_MODE_OPTIONS)
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
        self.filter_intensity_slider = LabeledSlider(
            "Intensidade do filtro", 0, 100, self.active_image_filter_intensity
        )

        self.sharpen_slider = LabeledSlider("Nitidez", 0, 100, defaults.sharpen)
        self.contrast_slider = LabeledSlider("Contraste", 0, 100, defaults.contrast)
        self.brightness_slider = LabeledSlider("Brilho", 0, 100, defaults.brightness)
        self.gamma_slider = LabeledSlider("Gama", 0, 100, defaults.gamma)
        self.saturation_slider = LabeledSlider("Saturacao", 0, 100, defaults.saturation)
        self.temperature_slider = LabeledSlider("Temperatura", 0, 100, defaults.temperature)
        self.preset_combo = QComboBox()
        for label, preset in FILTER_PRESETS:
            self.preset_combo.addItem(label, preset)
        self.scale_combo = QComboBox()
        for label, factor in self.SCALE_OPTIONS:
            self.scale_combo.addItem(label, factor)

        self.rotation_combo = self._enum_combo(self.ROTATION_OPTIONS)
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

    @staticmethod
    def _enum_combo(options: list[tuple[str, E]]) -> QComboBox:
        combo = QComboBox()
        for label, member in options:
            combo.addItem(label, member.value)
        return combo

    @staticmethod
    def _combo_enum(combo: QComboBox, enum_type: type[E], default: E) -> E:
        value = combo.currentData()
        try:
            return enum_type(value)
        except ValueError:
            return default

    @staticmethod
    def _select_combo_data(combo: QComboBox, data: object) -> None:
        index = combo.findData(data)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        left_column = QVBoxLayout()
        left_column.setSpacing(16)

        preview_toolbar = QHBoxLayout()
        preview_toolbar.setSpacing(10)
        preview_toolbar.addWidget(QLabel("Zoom do preview"))
        for button, tooltip in (
            (self.preview_zoom_out_button, "Diminuir zoom do preview"),
            (self.preview_zoom_in_button, "Aumentar zoom do preview"),
        ):
            button.setObjectName("previewZoomButton")
            button.setFixedWidth(38)
            button.setToolTip(tooltip)
        preview_toolbar.addWidget(self.preview_zoom_out_button)
        preview_toolbar.addWidget(self.preview_zoom_combo)
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

        frame_row = QHBoxLayout()
        frame_row.addWidget(self.frame_label)
        frame_row.addWidget(self.frame_slider, 1)

        transport_row = QHBoxLayout()
        transport_row.setSpacing(8)
        transport_row.addStretch(1)
        for button in (
            self.go_first_button,
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.go_last_button,
        ):
            transport_row.addWidget(button)
        transport_row.addStretch(1)

        left_column.addLayout(preview_toolbar)
        left_column.addWidget(self.preview_scroll, 1)
        left_column.addLayout(frame_row)
        left_column.addLayout(transport_row)

        self.settings_tabs = QTabWidget()
        self.settings_tabs.setObjectName("settingsTabs")
        self.settings_tabs.setMinimumWidth(340)
        self.settings_tabs.addTab(self._build_color_tab(), "Cor e detalhe")
        self._filters_tab_index = self.settings_tabs.addTab(self._build_filters_tab(), "Filtros")
        self.settings_tabs.addTab(self._build_stabilization_tab(), "Estabilizacao")
        self.settings_tabs.addTab(self._build_geometry_tab(), "Geometria")
        self.settings_tabs.addTab(self._build_noise_tab(), "Ruido")
        self.settings_tabs.addTab(self._build_export_tab(), "Midia")

        layout = QHBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)
        layout.addLayout(left_column, 3)
        layout.addWidget(self.settings_tabs, 1)

        self.setStyleSheet(load_stylesheet())

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("Arquivo")
        self._add_action(file_menu, "Abrir video...", self._select_video, QKeySequence.StandardKey.Open)
        self._add_action(file_menu, "Exportar video...", self._export_video, "Ctrl+E")
        file_menu.addSeparator()
        self._add_action(file_menu, "Sair", self.close, QKeySequence.StandardKey.Quit)

        edit_menu = menu_bar.addMenu("Editar")
        self._add_action(edit_menu, "Restaurar preset ativo", self._restore_active_preset_settings, "Ctrl+R")
        self._add_action(edit_menu, "Resetar crop", self._reset_crop_rect)
        self._add_action(edit_menu, "Resetar corte temporal", self._reset_clip_range)

        playback_menu = menu_bar.addMenu("Reproducao")
        self._add_action(playback_menu, "Reproduzir ou pausar", self._toggle_playback, "Space")
        playback_menu.addSeparator()
        self._add_action(playback_menu, "Primeiro frame", self._go_to_first_frame, "Home")
        self._add_action(playback_menu, "Frame anterior", self._go_to_previous_frame, "Left")
        self._add_action(playback_menu, "Proximo frame", self._go_to_next_frame, "Right")
        self._add_action(playback_menu, "Ultimo frame", self._go_to_last_frame, "End")
        playback_menu.addSeparator()
        self._add_action(playback_menu, "Marcar inicio do corte", self._mark_clip_start, "I")
        self._add_action(playback_menu, "Marcar fim do corte", self._mark_clip_end, "O")

        help_menu = menu_bar.addMenu("Ajuda")
        self._add_action(help_menu, "Sobre", self._show_about_dialog)

    def _add_action(
        self,
        menu: QMenu,
        text: str,
        slot: Callable[[], object],
        shortcut: QKeySequence.StandardKey | str | None = None,
    ) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_color_tab(self) -> QScrollArea:
        color_group = QGroupBox("Cor e look")
        color_layout = QVBoxLayout(color_group)
        color_layout.setSpacing(14)
        color_form = QFormLayout()
        color_form.addRow("Preset", self.preset_combo)
        color_layout.addLayout(color_form)
        for slider in (
            self.contrast_slider,
            self.brightness_slider,
            self.gamma_slider,
            self.saturation_slider,
            self.temperature_slider,
        ):
            color_layout.addWidget(slider)

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
        for index, preset in enumerate(IMAGE_FILTER_PRESETS):
            button = QToolButton()
            button.setObjectName("filterPresetButton")
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setIconSize(self.FILTER_THUMBNAIL_SIZE)
            button.setMinimumSize(170, 140)
            button.setText(preset.name)
            button.setToolTip(preset.description)
            button.clicked.connect(
                lambda _checked=False, filter_id=preset.filter_id: self._select_image_filter(filter_id)
            )
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
        group = QGroupBox("Estabilizacao")
        layout = QVBoxLayout(group)
        layout.setSpacing(14)
        form = QFormLayout()
        form.addRow("Modo", self.stabilization_mode_combo)
        layout.addLayout(form)
        for widget in (
            self.stabilization_intensity_slider,
            self.stabilization_smoothing_slider,
            self.stabilization_edit_button,
            self.stabilization_reset_button,
            self.stabilization_auto_crop_checkbox,
            self.stabilization_zoom_out_checkbox,
            self.stabilization_info_label,
        ):
            layout.addWidget(widget)
        return self._wrap_tab_content(group)

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
        group = QGroupBox("Limpeza e reducao de ruido")
        layout = QVBoxLayout(group)
        layout.setSpacing(14)
        form = QFormLayout()
        form.addRow("Modo", self.denoise_mode_combo)
        form.addRow("Janela de template", self.denoise_template_combo)
        form.addRow("Janela de busca", self.denoise_search_combo)
        layout.addLayout(form)
        for widget in (
            self.denoise_strength_slider,
            self.denoise_color_slider,
            self.denoise_sample_edit_button,
            self.denoise_sample_reset_button,
            self.denoise_info_label,
            self.denoise_sample_info_label,
        ):
            layout.addWidget(widget)
        return self._wrap_tab_content(group)

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

    @staticmethod
    def _wrap_tab_content(*sections: QWidget) -> QScrollArea:
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
        self.preview.cropChanged.connect(self._handle_preview_crop_changed)
        self.preview.sampleChanged.connect(self._handle_preview_sample_changed)
        self.preview.stabilizationChanged.connect(self._handle_preview_stabilization_changed)

        self.settings_tabs.currentChanged.connect(self._handle_tab_changed)
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
        self.rotation_combo.currentIndexChanged.connect(self._handle_geometry_change)
        self.flip_horizontal_checkbox.toggled.connect(self._handle_geometry_change)
        self.flip_vertical_checkbox.toggled.connect(self._handle_geometry_change)
        self.crop_mode_combo.currentIndexChanged.connect(self._handle_crop_mode_changed)
        self.crop_reset_button.clicked.connect(self._reset_crop_rect)
        self.stabilization_mode_combo.currentIndexChanged.connect(self._handle_stabilization_mode_changed)
        self.stabilization_intensity_slider.valueChanged.connect(self._handle_stabilization_control_change)
        self.stabilization_smoothing_slider.valueChanged.connect(self._handle_stabilization_control_change)
        self.stabilization_edit_button.toggled.connect(self._handle_stabilization_edit_toggled)
        self.stabilization_reset_button.clicked.connect(self._reset_stabilization_rect)
        self.stabilization_auto_crop_checkbox.toggled.connect(self._handle_stabilization_control_change)
        self.stabilization_zoom_out_checkbox.toggled.connect(self._handle_stabilization_zoom_out_toggled)
        self.denoise_mode_combo.currentIndexChanged.connect(self._handle_filter_control_change)
        self.denoise_template_combo.currentIndexChanged.connect(self._handle_filter_control_change)
        self.denoise_search_combo.currentIndexChanged.connect(self._handle_filter_control_change)
        self.denoise_sample_edit_button.toggled.connect(self._handle_denoise_sample_edit_toggled)
        self.denoise_sample_reset_button.clicked.connect(self._reset_denoise_sample_rect)
        self.filter_intensity_slider.valueChanged.connect(self._handle_filter_intensity_changed)
        self.scale_combo.currentIndexChanged.connect(self._handle_filter_control_change)
        self.mark_start_button.clicked.connect(self._mark_clip_start)
        self.mark_end_button.clicked.connect(self._mark_clip_end)
        self.reset_range_button.clicked.connect(self._reset_clip_range)

        for field_name, slider in self._filter_sliders().items():
            slider.valueChanged.connect(self._handle_filter_control_change)
            slider.resetRequested.connect(lambda name=field_name: self._reset_slider_to_preset(name))

    def _filter_sliders(self) -> dict[str, LabeledSlider]:
        return {
            "denoise_strength": self.denoise_strength_slider,
            "denoise_color_strength": self.denoise_color_slider,
            "sharpen": self.sharpen_slider,
            "contrast": self.contrast_slider,
            "brightness": self.brightness_slider,
            "gamma": self.gamma_slider,
            "saturation": self.saturation_slider,
            "temperature": self.temperature_slider,
        }

    def _set_controls_enabled(self, enabled: bool) -> None:
        widgets: list[QWidget] = [
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
            self.denoise_template_combo,
            self.denoise_search_combo,
            self.denoise_sample_edit_button,
            self.denoise_sample_reset_button,
            self.filter_intensity_slider,
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
            *self._filter_sliders().values(),
            *self.filter_preset_buttons.values(),
        ]
        for widget in widgets:
            widget.setEnabled(enabled)
        self._update_denoise_controls_state()
        self._update_stabilization_controls_state()
        self._update_filter_description()

    # =====================================================================================
    # Settings snapshot (the only place widgets are read for processing)
    # =====================================================================================
    def _current_filter_settings(self) -> FilterSettings:
        return FilterSettings(
            denoise_mode=self._combo_enum(self.denoise_mode_combo, DenoiseMode, DenoiseMode.OFF),
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

    def _current_geometry(self) -> GeometrySettings:
        return GeometrySettings(
            rotation=self._combo_enum(self.rotation_combo, RotationMode, RotationMode.NONE),
            flip_horizontal=self.flip_horizontal_checkbox.isChecked(),
            flip_vertical=self.flip_vertical_checkbox.isChecked(),
        )

    def _current_crop(self) -> CropSettings:
        return CropSettings(enabled=self._crop_enabled(), rect=qrect_to_norm(self.crop_rect_norm))

    def _current_stabilization(self) -> StabilizationSettings:
        return StabilizationSettings(
            mode=self._current_stabilization_mode(),
            intensity=clamp_unit(self.stabilization_intensity_slider.value() / 100.0),
            temporal_smoothing=clamp_unit(self.stabilization_smoothing_slider.value() / 100.0),
            focus_rect=qrect_to_norm(self.stabilization_rect_norm),
            auto_crop=self.stabilization_auto_crop_checkbox.isChecked(),
            zoom_out=self.stabilization_zoom_out_checkbox.isChecked(),
        )

    def _current_image_filter(self) -> ImageFilterSelection:
        return ImageFilterSelection(self.active_image_filter_id, clamp_unit(self.active_image_filter_intensity / 100.0))

    def _current_render_settings(self) -> RenderSettings:
        return RenderSettings(
            filters=self._current_filter_settings(),
            image_filter=self._current_image_filter(),
            geometry=self._current_geometry(),
            crop=self._current_crop(),
            stabilization=self._current_stabilization(),
            denoise_sample_rect=qrect_to_norm(self.denoise_sample_rect_norm),
        )

    def _current_stabilization_mode(self) -> StabilizationMode:
        return self._combo_enum(self.stabilization_mode_combo, StabilizationMode, StabilizationMode.OFF)

    def _current_denoise_mode(self) -> DenoiseMode:
        return self._combo_enum(self.denoise_mode_combo, DenoiseMode, DenoiseMode.OFF)

    def _preview_render_mode(self) -> PreviewRenderMode:
        return self._combo_enum(self.preview_render_mode_combo, PreviewRenderMode, PreviewRenderMode.HIGH)

    # =====================================================================================
    # Media
    # =====================================================================================
    def _select_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar video", str(Path.home()), "Videos (*.mp4 *.mov *.mkv *.avi *.m4v)"
        )
        if file_path:
            self._open_video(Path(file_path))

    def _open_video(self, path: Path) -> None:
        self._stop_playback()
        try:
            video = VideoSource(path)
        except VideoOpenError:
            QMessageBox.critical(self, "Erro", "Nao foi possivel abrir o video selecionado.")
            return
        if self.video is not None:
            self.video.release()

        self.video = video
        self.video_path = path
        self._backend.reset()
        self._discard_pending_previews()
        self.frame_count = video.frame_count
        self.fps = video.fps
        width, height = video.size
        self.source_size = (width, height)
        self.preview.set_display_reference_size(width, height)
        self.current_frame_index = 0
        self.clip_start_frame = 0
        self.clip_end_frame = self.frame_count - 1
        self.crop_rect_norm = self._default_crop_rect(self._normalized_crop_aspect_ratio())
        self.denoise_sample_rect_norm = norm_to_qrect(DEFAULT_DENOISE_SAMPLE_RECT)
        self.stabilization_rect_norm = norm_to_qrect(DEFAULT_STABILIZATION_RECT)
        self.stabilization_reference_frame = None
        self.stabilization_reference_frame_index = 0
        self._stabilization_epoch += 1
        self.file_label.setText(f"{path.name} | {width}x{height} @ {self.fps:.2f} fps")
        self.frame_slider.setRange(0, self.frame_count - 1)
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

        midpoint = min(self.frame_count - 1, self.frame_count // 2)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(midpoint)
        self.frame_slider.blockSignals(False)
        self._load_frame(midpoint)

    def _load_frame(self, frame_index: int) -> None:
        if self.video is None:
            return
        frame = self.video.read_frame(frame_index)
        if frame is None:
            self.status_label.setText("Nao foi possivel carregar esse frame.")
            return

        self.current_frame_index = frame_index
        self.current_frame = frame
        self.frame_label.setText(f"Frame {frame_index + 1} / {self.frame_count}")
        self._update_denoise_controls_state()
        self._update_stabilization_controls_state()
        self._update_stabilization_info_label()
        self._schedule_preview_refresh(immediate=True)

    # --- transport -------------------------------------------------------------------------
    def _go_to_first_frame(self) -> None:
        if self.frame_count > 0:
            self._stop_playback()
            self.frame_slider.setValue(0)

    def _go_to_previous_frame(self) -> None:
        if self.frame_count > 0:
            self._stop_playback()
            self.frame_slider.setValue(max(self.frame_slider.minimum(), self.frame_slider.value() - 1))

    def _go_to_next_frame(self) -> None:
        if self.frame_count > 0:
            self._stop_playback()
            self.frame_slider.setValue(min(self.frame_slider.maximum(), self.frame_slider.value() + 1))

    def _go_to_last_frame(self) -> None:
        if self.frame_count > 0:
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
            self._schedule_filter_thumbnails_refresh()

    def _advance_playback_frame(self) -> None:
        if self.frame_slider.value() >= self.playback_end_frame:
            self._stop_playback()
            return
        frame_step = 1 + int(self.preview_frame_skip_combo.currentData() or 0)
        self.frame_slider.setValue(min(self.playback_end_frame, self.frame_slider.value() + frame_step))

    def _playback_interval_ms(self) -> int:
        preview_fps = min(max(self.fps, 1.0), PREVIEW_MAX_FPS)
        return max(PREVIEW_MIN_INTERVAL_MS, int(1000 / preview_fps))

    # --- clip range ------------------------------------------------------------------------
    def _mark_clip_start(self) -> None:
        self.clip_start_frame = self.current_frame_index
        self.clip_end_frame = max(self.clip_end_frame, self.clip_start_frame)
        self._update_frame_slider_markers()
        self._update_range_label()

    def _mark_clip_end(self) -> None:
        self.clip_end_frame = self.current_frame_index
        self.clip_start_frame = min(self.clip_start_frame, self.clip_end_frame)
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

    def _update_range_label(self) -> None:
        if self.frame_count <= 0:
            self.range_label.setText("Corte: video inteiro")
            return
        start_frame, end_frame = self._selected_frame_range()
        frame_count = end_frame - start_frame + 1
        duration_seconds = frame_count / self.fps if self.fps > 0 else 0.0
        self.range_label.setText(
            f"Corte: frames {start_frame + 1}-{end_frame + 1} | {frame_count} frames | {duration_seconds:.2f}s"
        )

    def _update_frame_slider_markers(self) -> None:
        if self.frame_count <= 0:
            self.frame_slider.set_markers(0, 0, False)
            return
        start_frame, end_frame = self._selected_frame_range()
        self.frame_slider.set_markers(start_frame, end_frame, True)

    # =====================================================================================
    # Preview
    # =====================================================================================
    def _handle_preview_zoom_changed(self, _index: int) -> None:
        zoom_percent = int(self.preview_zoom_combo.currentData() or 100)
        self.preview.set_zoom_factor(zoom_percent / 100.0)
        current_index = self.preview_zoom_combo.currentIndex()
        self.preview_zoom_out_button.setEnabled(current_index > 0)
        self.preview_zoom_in_button.setEnabled(current_index < self.preview_zoom_combo.count() - 1)

    def _handle_preview_render_mode_changed(self, _index: int) -> None:
        self.preview.set_render_quality(self._preview_render_mode() is not PreviewRenderMode.FAST)
        self._stabilization_epoch += 1
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

    def _preview_output_scale_factor(self, scale_factor: float) -> float:
        mode = self._preview_render_mode()
        if mode is PreviewRenderMode.HIGH:
            return scale_factor
        if mode is PreviewRenderMode.BALANCED:
            return min(1.0, scale_factor)
        return 1.0

    def _schedule_preview_refresh(self, *, immediate: bool = False) -> None:
        if immediate:
            self._preview_refresh_timer.stop()
            self._refresh_preview_impl()
            return
        self._preview_refresh_timer.start(PREVIEW_REFRESH_DELAY_MS[self._preview_render_mode()])

    def _schedule_filter_thumbnails_refresh(self) -> None:
        if self.playback_timer.isActive() or not self._filters_tab_visible():
            return
        self._thumbnail_refresh_timer.start(THUMBNAIL_REFRESH_DELAY_MS)

    def _filters_tab_visible(self) -> bool:
        return self.settings_tabs.currentIndex() == self._filters_tab_index

    def _handle_tab_changed(self, _index: int) -> None:
        if self._filters_tab_visible():
            self._schedule_filter_thumbnails_refresh()

    def _refresh_preview_impl(self) -> None:
        """Snapshot the widgets and hand the frame to the render thread (latest request wins)."""
        if self.current_frame is None:
            return

        settings = self._current_render_settings()
        self._sync_crop_preview()
        self._sync_denoise_sample_preview()
        self._sync_stabilization_preview()

        preview_settings = replace(
            settings,
            filters=replace(
                settings.filters, scale_factor=self._preview_output_scale_factor(settings.filters.scale_factor)
            ),
        )
        self._preview_generation += 1
        self._preview_controller.submit_preview(
            PreviewRequest(
                generation=self._preview_generation,
                frame=self.current_frame,
                frame_index=self.current_frame_index,
                settings=preview_settings,
                size_limit=PREVIEW_SIZE_LIMIT.get(self._preview_render_mode()),
                stabilization_reference=self.stabilization_reference_frame,
                stabilization_epoch=self._stabilization_epoch,
            )
        )

    @Slot(object)
    def _on_preview_ready(self, images: PreviewImages) -> None:
        if images.generation <= self._applied_preview_generation:
            return  # stale: a newer request already landed (or the video changed)
        self._applied_preview_generation = images.generation
        self.last_preview_render_ms = images.elapsed_ms
        self.preview.set_images(images.before, images.after)
        self._schedule_filter_thumbnails_refresh()

    def _discard_pending_previews(self) -> None:
        """Results still in flight for the previous video must not land on the new one."""
        self._applied_preview_generation = self._preview_generation
        self._applied_thumbnail_generation = self._thumbnail_generation

    def _refresh_filter_thumbnails(self) -> None:
        self._thumbnail_refresh_timer.stop()
        if self.playback_timer.isActive() or not self._filters_tab_visible():
            return
        if self.current_frame is None:
            for button in self.filter_preset_buttons.values():
                button.setIcon(QIcon())
            return

        settings = self._current_render_settings()
        settings = replace(settings, filters=replace(settings.filters, scale_factor=1.0))
        reference = None
        if self.stabilization_reference_frame is not None:
            reference = self._thumbnail_source_frame(self.stabilization_reference_frame)
        self._thumbnail_generation += 1
        self._preview_controller.submit_thumbnails(
            ThumbnailRequest(
                generation=self._thumbnail_generation,
                frame=self._thumbnail_source_frame(self.current_frame),
                settings=settings,
                stabilization_reference=reference,
                preset_ids=tuple(preset.filter_id for preset in IMAGE_FILTER_PRESETS),
            )
        )

    @Slot(object)
    def _on_thumbnails_ready(self, result: ThumbnailImages) -> None:
        if result.generation <= self._applied_thumbnail_generation:
            return
        self._applied_thumbnail_generation = result.generation
        for preset_id, image in result.images.items():
            button = self.filter_preset_buttons.get(preset_id)
            if button is not None:
                button.setIcon(QIcon(QPixmap.fromImage(image)))

    @Slot(str)
    def _on_render_failed(self, message: str) -> None:
        self.status_label.setText(f"Falha ao renderizar o preview: {message}")

    def _thumbnail_source_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = min(
            self.FILTER_THUMBNAIL_SIZE.width() / max(1, width),
            self.FILTER_THUMBNAIL_SIZE.height() / max(1, height),
        )
        if scale >= 1.0:
            return frame.copy()
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # =====================================================================================
    # Image filter presets
    # =====================================================================================
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

    def _update_filter_selection_ui(self) -> None:
        for filter_id, button in self.filter_preset_buttons.items():
            button.blockSignals(True)
            button.setChecked(filter_id == self.active_image_filter_id)
            button.blockSignals(False)

    def _update_filter_description(self) -> None:
        preset: ImageFilterPreset = get_image_filter_preset(self.active_image_filter_id)
        intensity = clamp_unit(self.active_image_filter_intensity / 100.0)
        self.filter_description_label.setText(
            f"{preset.name}: {preset.description} Intensidade atual: {int(intensity * 100)}%."
        )
        self.filter_intensity_slider.setEnabled(self.frame_count > 0 and preset.filter_id != ORIGINAL_PRESET_ID)

    # =====================================================================================
    # Filter presets (sliders)
    # =====================================================================================
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
        self._filter_sliders()[field_name].setValue(int(getattr(self.active_preset_settings, field_name)))

    def _restore_active_preset_settings(self) -> None:
        self._apply_settings_to_controls(self.active_preset_settings)
        self._sync_preset_selection()
        self._schedule_preview_refresh()
        if self.current_frame is not None:
            self.status_label.setText("Ajustes restaurados para o preset ativo.")

    def _apply_settings_to_controls(self, settings: FilterSettings) -> None:
        widgets: list[QWidget] = [
            self.denoise_mode_combo,
            self.denoise_template_combo,
            self.denoise_search_combo,
            self.scale_combo,
            *self._filter_sliders().values(),
        ]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self._select_combo_data(self.denoise_mode_combo, settings.denoise_mode.value)
            self._select_combo_data(self.denoise_template_combo, settings.denoise_template_window)
            self._select_combo_data(self.denoise_search_combo, settings.denoise_search_window)
            self._select_combo_data(self.scale_combo, settings.scale_factor)
            for field_name, slider in self._filter_sliders().items():
                slider.slider.setValue(int(getattr(settings, field_name)))
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._update_denoise_controls_state()

    def _handle_filter_control_change(self, *_args: object) -> None:
        self._sync_preset_selection()
        self._update_denoise_controls_state()
        self._schedule_preview_refresh()

    def _handle_geometry_change(self, *_args: object) -> None:
        self._update_crop_info_label()
        self._schedule_preview_refresh()

    def _sync_preset_selection(self) -> None:
        current_settings = self._current_filter_settings()
        matching_index = 0
        for index in range(1, self.preset_combo.count()):
            if self.preset_combo.itemData(index) == current_settings:
                matching_index = index
                break
        if self.preset_combo.currentIndex() != matching_index:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(matching_index)
            self.preset_combo.blockSignals(False)

    # =====================================================================================
    # Denoise
    # =====================================================================================
    def _update_denoise_controls_state(self) -> None:
        denoise_mode = self._current_denoise_mode()
        manual_mode = denoise_mode is DenoiseMode.MANUAL
        sample_mode = denoise_mode is DenoiseMode.SAMPLE
        controls_enabled = self.frame_count > 0 and self.denoise_mode_combo.isEnabled()
        self.denoise_strength_slider.setEnabled(controls_enabled and manual_mode)
        self.denoise_color_slider.setEnabled(controls_enabled and manual_mode)
        self.denoise_sample_edit_button.setEnabled(controls_enabled and sample_mode)
        self.denoise_sample_reset_button.setEnabled(controls_enabled and sample_mode)
        if not (controls_enabled and sample_mode):
            self._set_checked_silently(self.denoise_sample_edit_button, False)
        self._sync_denoise_sample_preview()
        self._update_denoise_sample_info_label()

        if denoise_mode is DenoiseMode.OFF:
            self.denoise_info_label.setText(
                "Denoise desativado: o preview e a exportacao pulam esta etapa para poupar processamento."
            )
            return
        if manual_mode:
            self.denoise_info_label.setText(
                "Manual: ajuste forca de luminancia, forca de cor e o tamanho das janelas do filtro."
            )
            return

        source = (
            "Auto: calcula a intensidade com base no frame atual inteiro."
            if denoise_mode is DenoiseMode.AUTO
            else "Amostra: usa a regiao azul selecionada no preview como referencia de ruido."
        )
        if self.current_frame is None:
            self.denoise_info_label.setText(source)
            return
        luma, color = estimate_denoise_strengths(
            self.current_frame, denoise_mode, qrect_to_norm(self.denoise_sample_rect_norm)
        )
        self.denoise_info_label.setText(f"{source} Estimativa atual: luminancia {luma:.1f} | cor {color:.1f}.")

    def _handle_denoise_sample_edit_toggled(self, checked: bool) -> None:
        if checked and self.stabilization_edit_button.isChecked():
            self._set_checked_silently(self.stabilization_edit_button, False)
            self._sync_stabilization_preview(editing_override=False)
        self._sync_denoise_sample_preview(editing_override=checked)

    def _handle_preview_sample_changed(self, rect: QRectF) -> None:
        self.denoise_sample_rect_norm = QRectF(rect)
        self._sync_denoise_sample_preview()
        self._update_denoise_sample_info_label()
        self._schedule_preview_refresh()

    def _reset_denoise_sample_rect(self) -> None:
        self.denoise_sample_rect_norm = norm_to_qrect(DEFAULT_DENOISE_SAMPLE_RECT)
        self._sync_denoise_sample_preview()
        self._update_denoise_sample_info_label()
        self._schedule_preview_refresh()

    def _sync_denoise_sample_preview(self, editing_override: bool | None = None) -> None:
        sample_enabled = self.frame_count > 0 and self._current_denoise_mode() is DenoiseMode.SAMPLE
        editing = self.denoise_sample_edit_button.isChecked() if editing_override is None else editing_override
        self.preview.set_sample_state(sample_enabled, self.denoise_sample_rect_norm, editing)

    def _update_denoise_sample_info_label(self) -> None:
        if self.frame_count <= 0:
            self.denoise_sample_info_label.setText("Amostra: carregue um video para escolher a regiao de analise.")
            return
        rect = qrect_to_norm(self.denoise_sample_rect_norm)
        percent = f"{rect.width * 100:.0f}% x {rect.height * 100:.0f}%"
        pixels = rect.to_pixels(*self.source_size)
        if pixels is not None:
            self.denoise_sample_info_label.setText(f"Amostra: {pixels[2]}x{pixels[3]} | {percent}")
        else:
            self.denoise_sample_info_label.setText(f"Amostra: {percent}")

    # =====================================================================================
    # Stabilization
    # =====================================================================================
    def _handle_stabilization_mode_changed(self, *_args: object) -> None:
        self._stabilization_epoch += 1
        mode = self._current_stabilization_mode()
        if mode.uses_focal_point:
            self._capture_stabilization_reference_from_current_frame()
        else:
            self._set_checked_silently(self.stabilization_edit_button, False)
        self._update_stabilization_controls_state()
        self._sync_stabilization_preview()
        self._update_stabilization_info_label()
        self._schedule_preview_refresh()

    def _handle_stabilization_edit_toggled(self, checked: bool) -> None:
        if checked and self.denoise_sample_edit_button.isChecked():
            self._set_checked_silently(self.denoise_sample_edit_button, False)
            self._sync_denoise_sample_preview(editing_override=False)
        if checked:
            self._capture_stabilization_reference_from_current_frame()
        self._sync_stabilization_preview(editing_override=checked)

    def _handle_stabilization_zoom_out_toggled(self, checked: bool) -> None:
        if checked and self.stabilization_auto_crop_checkbox.isChecked():
            self._set_checked_silently(self.stabilization_auto_crop_checkbox, False)
        self._handle_stabilization_control_change()

    def _handle_stabilization_control_change(self, *_args: object) -> None:
        self._stabilization_epoch += 1
        self._update_stabilization_controls_state()
        self._update_stabilization_info_label()
        self._schedule_preview_refresh()

    def _handle_preview_stabilization_changed(self, rect: QRectF) -> None:
        self.stabilization_rect_norm = QRectF(rect)
        self._capture_stabilization_reference_from_current_frame()
        self._stabilization_epoch += 1
        self._sync_stabilization_preview()
        self._update_stabilization_info_label()
        self._schedule_preview_refresh()

    def _reset_stabilization_rect(self) -> None:
        self.stabilization_rect_norm = norm_to_qrect(DEFAULT_STABILIZATION_RECT)
        self._capture_stabilization_reference_from_current_frame()
        self._stabilization_epoch += 1
        self._sync_stabilization_preview()
        self._update_stabilization_info_label()
        self._schedule_preview_refresh()

    def _capture_stabilization_reference_from_current_frame(self) -> None:
        if self.current_frame is None:
            return
        self.stabilization_reference_frame = self.current_frame.copy()
        self.stabilization_reference_frame_index = self.current_frame_index

    def _update_stabilization_controls_state(self) -> None:
        mode = self._current_stabilization_mode()
        controls_enabled = self.frame_count > 0 and self.stabilization_mode_combo.isEnabled()
        stabilization_enabled = controls_enabled and mode is not StabilizationMode.OFF
        self.stabilization_intensity_slider.setEnabled(stabilization_enabled)
        self.stabilization_smoothing_slider.setEnabled(stabilization_enabled)
        zoom_out_enabled = stabilization_enabled and self.stabilization_zoom_out_checkbox.isChecked()
        self.stabilization_auto_crop_checkbox.setEnabled(stabilization_enabled and not zoom_out_enabled)
        self.stabilization_zoom_out_checkbox.setEnabled(stabilization_enabled)
        focal_mode = stabilization_enabled and mode.uses_focal_point
        self.stabilization_edit_button.setEnabled(focal_mode)
        self.stabilization_reset_button.setEnabled(focal_mode)
        if not focal_mode:
            self._set_checked_silently(self.stabilization_edit_button, False)
        self._sync_stabilization_preview()
        self._update_stabilization_info_label()

    def _sync_stabilization_preview(self, editing_override: bool | None = None) -> None:
        enabled = self.frame_count > 0 and self._current_stabilization_mode().uses_focal_point
        editing = self.stabilization_edit_button.isChecked() if editing_override is None else editing_override
        self.preview.set_stabilization_state(enabled, self.stabilization_rect_norm, editing)

    def _update_stabilization_info_label(self) -> None:
        if self.frame_count <= 0:
            self.stabilization_info_label.setText("Estabilizacao: carregue um video para definir o ponto focal.")
            return
        settings = self._current_stabilization()
        if not settings.enabled:
            self.stabilization_info_label.setText("Estabilizacao desativada.")
            return

        flags = (
            f"Crop automatico: {'ligado' if settings.auto_crop else 'desligado'}. "
            f"Zoom out: {'ligado' if settings.zoom_out else 'desligado'}. "
            f"Suavizacao temporal: {int(settings.temporal_smoothing * 100)}%."
        )
        sequential_note = " Exportacao usa processamento sequencial nesse modo."
        descriptions = {
            StabilizationMode.AUTO: (
                "Automatico por translacao: acompanha o deslocamento global entre frames consecutivos. "
            ),
            StabilizationMode.SIMILARITY: (
                "Automatico por similaridade: compensa translacao, rotacao leve e pequenas variacoes de escala "
                "entre frames consecutivos. "
            ),
            StabilizationMode.PERSPECTIVE: (
                "Automatico por perspectiva: estima uma homografia global para lidar melhor com parallax leve "
                "e mudancas de perspectiva pequenas. "
            ),
        }
        if settings.mode in descriptions:
            self.stabilization_info_label.setText(descriptions[settings.mode] + flags + sequential_note)
            return

        region = settings.focus_rect.to_pixels(*self.source_size)
        if region is None:
            self.stabilization_info_label.setText("Estabilizacao: selecione um ponto ou regiao focal no preview.")
            return
        reference = (
            f"Referencia: frame {self.stabilization_reference_frame_index + 1}. Regiao: {region[2]}x{region[3]}. "
        )
        if settings.mode is StabilizationMode.HYBRID:
            self.stabilization_info_label.setText(
                "Hibrido ativo. Tenta prender a regiao verde com camera lock e, se perder o alvo, faz fallback "
                "progressivo para estabilizacao global. " + reference + flags
            )
            return
        self.stabilization_info_label.setText(
            "Camera lock ativo. Usa features da regiao verde para estimar uma transformacao robusta e tenta "
            "manter esse enquadramento preso ao frame de referencia. " + reference + flags
        )

    # =====================================================================================
    # Crop
    # =====================================================================================
    def _crop_mode_data(self) -> object:
        return self.crop_mode_combo.currentData()

    def _crop_enabled(self) -> bool:
        return self._crop_mode_data() is not None

    def _crop_aspect_ratio(self) -> float | None:
        crop_mode = self._crop_mode_data()
        if crop_mode is None or crop_mode == "free":
            return None
        return float(crop_mode)  # type: ignore[arg-type]

    def _displayed_size(self) -> tuple[int, int]:
        """Frame size the crop rectangle refers to: source, scaled, then rotated."""
        scale = float(self.scale_combo.currentData() or 1.0)
        width = round(self.source_size[0] * scale)
        height = round(self.source_size[1] * scale)
        return self._current_geometry().output_size(width, height)

    def _displayed_aspect_ratio(self) -> float:
        width, height = self._displayed_size()
        return width / height if width > 0 and height > 0 else 1.0

    def _normalized_crop_aspect_ratio(self) -> float | None:
        aspect_ratio = self._crop_aspect_ratio()
        if aspect_ratio is None:
            return None
        return aspect_ratio / self._displayed_aspect_ratio()

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
        self._schedule_preview_refresh()

    def _handle_preview_crop_changed(self, rect: QRectF) -> None:
        aspect_ratio = self._normalized_crop_aspect_ratio()
        self.crop_rect_norm = (
            QRectF(rect) if aspect_ratio is None else self._crop_rect_with_aspect(QRectF(rect), aspect_ratio)
        )
        self._sync_crop_preview()
        self._update_crop_info_label()

    def _reset_crop_rect(self) -> None:
        self.crop_rect_norm = self._default_crop_rect(self._normalized_crop_aspect_ratio())
        self._sync_crop_preview()
        self._update_crop_info_label()
        self._schedule_preview_refresh()

    def _sync_crop_preview(self) -> None:
        self.preview.set_crop_state(self._crop_enabled(), self.crop_rect_norm, self._normalized_crop_aspect_ratio())

    def _update_crop_info_label(self) -> None:
        if not self._crop_enabled():
            self.crop_info_label.setText("Crop: desativado")
            return

        if self._crop_mode_data() == "free":
            prefix = "Crop: Livre"
            ratio_text = self._rect_ratio_text(self.crop_rect_norm)
        else:
            prefix = f"Crop: {self.crop_mode_combo.currentText()}"
            ratio_text = self.crop_mode_combo.currentText()

        width, height = self._displayed_size()
        pixels = self._current_crop().pixel_rect(width, height)
        if pixels is not None:
            self.crop_info_label.setText(f"{prefix} | {pixels[2]}x{pixels[3]} | {ratio_text}")
            return
        self.crop_info_label.setText(
            f"{prefix} | {self.crop_rect_norm.width() * 100:.0f}% x {self.crop_rect_norm.height() * 100:.0f}%"
        )

    def _rect_ratio_text(self, rect: QRectF) -> str:
        if rect.height() <= 0 or rect.width() <= 0:
            return "Livre"
        ratio = (rect.width() / rect.height()) * self._displayed_aspect_ratio()
        candidates = [(16 / 9, "16:9"), (4 / 3, "4:3"), (1.0, "1:1"), (9 / 16, "9:16")]
        best_ratio, best_label = min(candidates, key=lambda candidate: abs(candidate[0] - ratio))
        if abs(best_ratio - ratio) < 0.04:
            return best_label
        return f"{ratio:.2f}:1"

    @staticmethod
    def _default_crop_rect(aspect_ratio: float | None) -> QRectF:
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

    # =====================================================================================
    # Export
    # =====================================================================================
    def _export_video(self) -> None:
        if self.video_path is None or self._export_controller is not None:
            return

        default_name = self.video_path.with_stem(f"{self.video_path.stem}_enhanced").with_suffix(".mp4")
        output_path, _ = QFileDialog.getSaveFileName(self, "Salvar video processado", str(default_name), "MP4 (*.mp4)")
        if not output_path:
            return

        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path is None:
            QMessageBox.warning(
                self,
                "FFmpeg nao encontrado",
                "Instale o FFmpeg e deixe o executavel disponivel no PATH para exportar e muxar o audio do video.",
            )
            return

        if self._ffmpeg_probe is None or self._ffmpeg_probe.ffmpeg_path != ffmpeg_path:
            self._ffmpeg_probe = FfmpegProbe(ffmpeg_path)
        encoder = resolve_encoder(self._ffmpeg_probe)
        self._backend.reset()

        start_frame, end_frame = self._selected_frame_range()
        request = ExportRequest(
            source_path=self.video_path,
            output_path=Path(output_path),
            settings=self._current_render_settings(),
            start_frame=start_frame,
            end_frame=end_frame,
            fps=self.fps,
            worker_count=int(self.thread_combo.currentData() or 1),
            ffmpeg_path=ffmpeg_path,
            encoder=encoder,
            stabilization_reference=self.stabilization_reference_frame,
        )
        self._export_backend_label = f"{self._backend.label} + {encoder.label}"
        note = ""
        if request.effective_worker_count < request.worker_count:
            note = " | estabilizacao temporal usa processamento sequencial"

        controller = ExportController(request, self._backend, self)
        controller.progress.connect(self._on_export_progress)
        controller.finished.connect(self._on_export_finished)
        self._export_controller = controller
        self._begin_export_ui(
            "Exportando video frame a frame com OpenCV... "
            f"usando {request.effective_worker_count} thread(s) | {self._export_backend_label}{note}."
        )
        controller.start()

    def _begin_export_ui(self, message: str) -> None:
        self._stop_playback()
        self._set_controls_enabled(False)
        self.select_button.setEnabled(False)
        self.cancel_export_button.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(message)

    def _end_export_ui(self) -> None:
        self._set_controls_enabled(True)
        self.select_button.setEnabled(True)
        self.cancel_export_button.setEnabled(False)

    @Slot(int, int)
    def _on_export_progress(self, processed_frames: int, total_frames: int) -> None:
        progress_value = int((processed_frames / max(1, total_frames)) * 100)
        self.progress_bar.setValue(progress_value)
        self.status_label.setText(
            "Exportando video com OpenCV... "
            f"{progress_value}% | frame {processed_frames}/{total_frames} | {self._export_backend_label}"
        )

    @Slot(object)
    def _on_export_finished(self, result: ExportResult) -> None:
        if self._export_controller is not None:
            self._export_controller.deleteLater()
            self._export_controller = None
        self._end_export_ui()

        if result.success:
            self.progress_bar.setValue(100)
            message = "Video exportado com sucesso."
            if result.frame_count_mismatch:
                message += f" {result.message}"
            self.status_label.setText(message)
            QMessageBox.information(self, "Concluido", message)
            return

        self.progress_bar.setValue(0)
        if result.cancelled:
            self.status_label.setText(result.message)
            return
        self.status_label.setText("A exportacao falhou. Revise os parametros e tente novamente.")
        QMessageBox.warning(self, "Falha na exportacao", result.message)

    def _request_export_cancel(self) -> None:
        if self._export_controller is None or not self.cancel_export_button.isEnabled():
            return
        self._export_controller.request_cancel()
        self.cancel_export_button.setEnabled(False)
        self.status_label.setText("Cancelando exportacao...")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_playback()
        if self._export_controller is not None:
            self._export_controller.stop()
            self._export_controller = None
        self._preview_controller.stop()
        if self.video is not None:
            self.video.release()
            self.video = None
        super().closeEvent(event)

    # =====================================================================================
    # Misc
    # =====================================================================================
    @staticmethod
    def _set_checked_silently(button: QPushButton | QCheckBox, checked: bool) -> None:
        if button.isChecked() != checked:
            button.blockSignals(True)
            button.setChecked(checked)
            button.blockSignals(False)

    def _show_about_dialog(self) -> None:
        QMessageBox.information(
            self,
            "Sobre",
            "Space Video Enhancer\n\nEditor offline com preview antes/depois, crop interativo e exportacao local.",
        )
