"""Main window: wires preview, transport, tabs, toolbar and status bar; snapshots widgets into ``RenderSettings``.

No image processing lives here. Everything the pipeline needs is read from widgets
exactly once per render into an immutable snapshot (see :mod:`space_video_enhancer.settings`).
The tabs (:mod:`space_video_enhancer.ui.tabs`) own their widgets; this class orchestrates.

The media is a :class:`~space_video_enhancer.sequence.Sequence` of clips. The playhead, the
export range and the cuts live in *timeline frames*; the tabs always show (and edit) the
settings of the clip under the playhead — the *active clip* — and every render stores the
widgets back into that clip, so switching clips swaps the whole panel.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QIcon, QImage, QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from space_video_enhancer.audio import AudioPeaks
from space_video_enhancer.export.ffmpeg import FfmpegProbe, find_ffmpeg, resolve_encoder
from space_video_enhancer.export.job import ExportRequest, ExportResult
from space_video_enhancer.export.worker import ExportController
from space_video_enhancer.pipeline.exposure import gains_in_ev, region_medians
from space_video_enhancer.pipeline.filters import OpenCLBackend
from space_video_enhancer.pipeline.grade import neutral_gains
from space_video_enhancer.pipeline.presets import IMAGE_FILTER_PRESETS
from space_video_enhancer.pipeline.preview import PreviewRequest, ThumbnailRequest
from space_video_enhancer.pipeline.stabilization.stabilize import CORRECTION_BUDGET_FRACTION
from space_video_enhancer.sequence import Clip, ClipSource
from space_video_enhancer.sequence import Sequence as ClipSequence
from space_video_enhancer.settings import (
    DEFAULT_CROP_RECT,
    DEFAULT_DENOISE_SAMPLE_RECT,
    DEFAULT_FILTER_SETTINGS,
    DEFAULT_STABILIZATION_RECT,
    DEFAULT_WHITE_BALANCE_RECT,
    NEUTRAL_GAINS,
    CropSettings,
    DenoiseMode,
    FilterSettings,
    GeometrySettings,
    NormalizedRect,
    PreviewRenderMode,
    RenderSettings,
    StabilizationSettings,
)
from space_video_enhancer.telemetry.diff import diff as diff_settings
from space_video_enhancer.telemetry.log import current_session, get_logger, log_event
from space_video_enhancer.telemetry.stats import RollingStats
from space_video_enhancer.timeline import FrameRange, frame_count, kept_segments, next_kept_frame
from space_video_enhancer.ui.diagnostics_dialog import DiagnosticsDialog
from space_video_enhancer.ui.format import format_duration, format_timecode, parse_timecode
from space_video_enhancer.ui.preview_worker import PreviewController, PreviewImages, ThumbnailImages
from space_video_enhancer.ui.tabs.base import TabPage, combo_enum, enum_combo, select_combo_data
from space_video_enhancer.ui.tabs.color import ColorTab
from space_video_enhancer.ui.tabs.export import ExportTab
from space_video_enhancer.ui.tabs.filters import THUMBNAIL_SIZE, FiltersTab
from space_video_enhancer.ui.tabs.geometry import GeometryTab
from space_video_enhancer.ui.tabs.grade import ColorGradeTab
from space_video_enhancer.ui.tabs.noise import NoiseTab
from space_video_enhancer.ui.tabs.stabilization import StabilizationTab
from space_video_enhancer.ui.theme import load_stylesheet
from space_video_enhancer.ui.timeline_media import FilmstripCell, FilmstripController, WaveformController
from space_video_enhancer.ui.widgets.compare_preview import ComparePreview
from space_video_enhancer.ui.widgets.status_panel import StatusPanel
from space_video_enhancer.ui.widgets.timeline import ClipBlock
from space_video_enhancer.ui.widgets.transport_bar import TransportBar
from space_video_enhancer.video import VideoOpenError, VideoSource

LOG = get_logger("ui")

DEFAULT_WINDOW_SIZE = QSize(1280, 800)
PREVIEW_MAX_FPS = 12.0
PREVIEW_MIN_INTERVAL_MS = 40
THUMBNAIL_REFRESH_DELAY_MS = 140
PREVIEW_SIZE_LIMIT = {PreviewRenderMode.FAST: 960, PreviewRenderMode.BALANCED: 1280}
INTERACTIVE_DENOISE_SIZE_LIMIT = 640
"""While scrubbing or playing with denoise on, NLMeans runs on a frame no larger than this (cost ~ pixels)."""
IDLE_REFRESH_DELAY_MS = 250
"""After the last interactive render, the resting frame is re-rendered at full quality."""
INTERACTIVE_REASONS = ("scrub", "playback")
WAVEFORM_BUCKETS_PER_FRAME = 4
"""Audio peaks resolution for the timeline: four buckets per video frame keeps a 1 h clip under 4 MB."""
PREVIEW_REFRESH_DELAY_MS = {PreviewRenderMode.FAST: 20, PreviewRenderMode.BALANCED: 35, PreviewRenderMode.HIGH: 50}
PREVIEW_ZOOM_STEPS = (25, 50, 100, 200, 400)
"""Fixed zoom levels; 100% shows one output pixel per logical screen pixel. "Ajustar" (fit) is the default."""
PREVIEW_RENDER_MODE_OPTIONS: list[tuple[str, PreviewRenderMode]] = [
    ("Alta qualidade", PreviewRenderMode.HIGH),
    ("Balanceado", PreviewRenderMode.BALANCED),
    ("Rápido", PreviewRenderMode.FAST),
]
PREVIEW_FRAME_SKIP_OPTIONS = [("Sem pulo", 0), ("Pular 1", 1), ("Pular 2", 2), ("Pular 3", 3)]
VIDEO_FILE_FILTER = "Vídeos (*.mp4 *.mov *.mkv *.avi *.m4v)"


def qrect_to_norm(rect: QRectF) -> NormalizedRect:
    return NormalizedRect(rect.x(), rect.y(), rect.width(), rect.height())


def norm_to_qrect(rect: NormalizedRect) -> QRectF:
    return QRectF(rect.x, rect.y, rect.width, rect.height)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Space Video Enhancer")
        self.resize(DEFAULT_WINDOW_SIZE)

        # --- media state -------------------------------------------------------------------
        self.sequence = ClipSequence()
        """The clips on the timeline; each carries its own settings, cuts and references (see sequence.py)."""
        self._sources: dict[Path, VideoSource] = {}
        """Decoders handed to the preview worker, one per file; the GUI thread only reads their metadata."""
        self._reference_frames: dict[tuple[Path, int], np.ndarray] = {}
        """Camera-lock reference frames by (file, source frame); clips split from one another share them."""
        self._waveforms_requested: set[Path] = set()
        self.active_clip_index = -1
        """The clip under the playhead: the tabs show and edit its settings."""
        self.current_frame_index = 0
        """Playhead, in timeline frames."""
        self.clip_start_frame = 0
        self.clip_end_frame = 0
        """Export range, in timeline frames (the I/O markers on the ruler)."""
        self.playback_end_frame = 0
        self._pending_cut_start: int | None = None
        self.current_frame: np.ndarray | None = None

        # --- editing state -----------------------------------------------------------------
        self.active_preset_settings: FilterSettings = DEFAULT_FILTER_SETTINGS
        self.crop_rect_norm = norm_to_qrect(DEFAULT_CROP_RECT)
        self.denoise_sample_rect_norm = norm_to_qrect(DEFAULT_DENOISE_SAMPLE_RECT)
        self.white_balance_rect_norm = norm_to_qrect(DEFAULT_WHITE_BALANCE_RECT)
        self.white_balance_gains: tuple[float, float, float] = NEUTRAL_GAINS
        self.stabilization_rect_norm = norm_to_qrect(DEFAULT_STABILIZATION_RECT)
        self.stabilization_reference_frame: np.ndarray | None = None
        self.stabilization_reference_frame_index = 0
        self._stabilization_epoch = 0  # bumped whenever inter-frame stabilization memory must restart

        # --- processing / export infra -----------------------------------------------------
        self._backend = OpenCLBackend()
        self._preview_controller = PreviewController(self._backend, self)
        self._preview_controller.previewReady.connect(self._on_preview_ready)
        self._preview_controller.thumbnailsReady.connect(self._on_thumbnails_ready)
        self._filmstrip = FilmstripController(self)  # its own decoder thread: timeline seeks never stall the preview
        self._filmstrip.thumbnailReady.connect(self._on_filmstrip_thumbnail)
        self._waveform = WaveformController(self)
        self._waveform.peaksReady.connect(self._on_waveform_ready)
        self._preview_controller.renderFailed.connect(self._on_render_failed)
        self._preview_generation = 0
        self._applied_preview_generation = -1
        self._thumbnail_generation = 0
        self._applied_thumbnail_generation = -1
        self.last_preview_render_ms = 0.0
        self._last_preview: PreviewImages | None = None
        self._last_logged_settings: RenderSettings | None = None
        self._pending_preview_reason = "settings"
        self._session_started = time.monotonic()
        self._counters: Counter[str] = Counter()
        self._playback_started_at: float | None = None
        self._playback_frames_shown = 0
        self._playback_intervals = RollingStats(window=60)
        self._playback_stabilization: Counter[str] = Counter()
        self._playback_last_frame_at: float | None = None
        self._playback_video_counters = (0, 0)
        # Decode metrics arrive with each render result; the GUI never touches the decoder itself.
        self._decoded_frame_index = 0
        """Source frame of ``current_frame`` (in ``_decoded_source``), for reference captures."""
        self._decoded_source: Path | None = None
        self._decode_stats = RollingStats(window=120)
        self._last_decode_ms = 0.0
        self._last_decode_sequential: bool | None = None
        self._video_read_counters = (0, 0)  # (seeks, sequential reads) after the latest decode
        self._hud_visible = False
        self._ffmpeg_probe: FfmpegProbe | None = None
        self._export_controller: ExportController | None = None
        self._export_backend_label = ""
        self._export_started_at = 0.0
        self._export_output_path: Path | None = None

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_playback_frame)
        self._preview_refresh_timer = QTimer(self)
        self._preview_refresh_timer.setSingleShot(True)
        self._preview_refresh_timer.timeout.connect(self._refresh_preview_impl)
        self._thumbnail_refresh_timer = QTimer(self)
        self._thumbnail_refresh_timer.setSingleShot(True)
        self._thumbnail_refresh_timer.timeout.connect(self._refresh_filter_thumbnails)
        self._idle_refresh_timer = QTimer(self)
        self._idle_refresh_timer.setSingleShot(True)
        self._idle_refresh_timer.timeout.connect(self._refresh_preview_at_full_quality)

        self._create_widgets()
        self._create_actions()
        self._build_ui()
        self._build_toolbar()
        self._build_menu_bar()
        self._bind_events()
        self._apply_settings_to_controls(DEFAULT_FILTER_SETTINGS)
        self._apply_preview_zoom()
        self._set_controls_enabled(False)
        self.status.show_message("Abra um vídeo para começar: ajuste os controles e revise um frame antes de exportar.")

    # =====================================================================================
    # Widget construction
    # =====================================================================================
    def _create_widgets(self) -> None:
        self.preview = ComparePreview()
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setObjectName("previewScrollArea")
        self.preview_scroll.setWidget(self.preview)
        self.preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.transport = TransportBar()
        self.status = StatusPanel()

        self.color_tab = ColorTab()
        self.grade_tab = ColorGradeTab()
        self.filters_tab = FiltersTab()
        self.stabilization_tab = StabilizationTab()
        self.geometry_tab = GeometryTab()
        self.noise_tab = NoiseTab()
        self.export_tab = ExportTab()
        self._tabs: tuple[tuple[str, TabPage], ...] = (
            ("Cor", self.color_tab),
            ("HSL", self.grade_tab),
            ("Filtros", self.filters_tab),
            ("Estabilizar", self.stabilization_tab),
            ("Geometria", self.geometry_tab),
            ("Ruído", self.noise_tab),
            ("Exportar", self.export_tab),
        )

        self.preview_zoom_combo = QComboBox()
        self.preview_zoom_combo.addItem("Ajustar", None)
        for percent in PREVIEW_ZOOM_STEPS:
            self.preview_zoom_combo.addItem(f"{percent}%", percent)
        self.preview_zoom_combo.setToolTip("Ajustar à janela ou zoom fixo (100% = 1 pixel do vídeo por pixel da tela)")
        self.preview_zoom_out_button = QToolButton()
        self.preview_zoom_out_button.setText("−")
        self.preview_zoom_out_button.setToolTip("Diminuir zoom do preview")
        self.preview_zoom_in_button = QToolButton()
        self.preview_zoom_in_button.setText("+")
        self.preview_zoom_in_button.setToolTip("Aumentar zoom do preview")
        for button in (self.preview_zoom_out_button, self.preview_zoom_in_button):
            button.setObjectName("previewZoomButton")
        self.preview_render_mode_combo = enum_combo(PREVIEW_RENDER_MODE_OPTIONS)
        self.preview_render_mode_combo.setToolTip("Troca qualidade por velocidade apenas no preview")
        self.preview_frame_skip_combo = QComboBox()
        for label, skip_count in PREVIEW_FRAME_SKIP_OPTIONS:
            self.preview_frame_skip_combo.addItem(label, skip_count)
        self.preview_frame_skip_combo.setToolTip("Pula frames apenas na reprodução do preview para ficar mais leve")

    def _create_actions(self) -> None:
        style = self.style()
        self.open_action = self._make_action(
            "Abrir vídeo…",
            self._select_video,
            QKeySequence.StandardKey.Open,
            icon=style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
        )
        self.add_video_action = self._make_action(
            "Adicionar vídeo à sequência…",
            self._select_video_to_add,
            "Ctrl+Shift+O",
            icon=style.standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder),
        )
        self.add_video_action.setToolTip("Junta outro arquivo ao fim da linha do tempo (Ctrl+Shift+O)")
        self.export_action = self._make_action(
            "Exportar…",
            self._export_video,
            "Ctrl+E",
            icon=style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
        )
        self.split_clip_action = self._make_action("Dividir clipe no playhead", self._split_clip_at_playhead, "Ctrl+K")
        self.remove_clip_action = self._make_action("Remover clipe ativo", self._remove_active_clip)
        self.quit_action = self._make_action("Sair", self.close, QKeySequence.StandardKey.Quit)
        self.restore_preset_action = self._make_action(
            "Restaurar preset ativo", self._restore_active_preset_settings, "Ctrl+R"
        )
        self.reset_crop_action = self._make_action("Redefinir crop", self._reset_crop_rect)
        self.reset_range_action = self._make_action("Redefinir corte temporal", self._reset_clip_range)
        self.play_action = self._make_action("Reproduzir ou pausar", self._toggle_playback, "Space")
        self.first_frame_action = self._make_action("Primeiro frame", self._go_to_first_frame, "Home")
        self.previous_frame_action = self._make_action("Frame anterior", self._go_to_previous_frame, "Left")
        self.next_frame_action = self._make_action("Próximo frame", self._go_to_next_frame, "Right")
        self.last_frame_action = self._make_action("Último frame", self._go_to_last_frame, "End")
        self.mark_start_action = self._make_action("Marcar início do corte", self._mark_clip_start, "I")
        self.mark_end_action = self._make_action("Marcar fim do corte", self._mark_clip_end, "O")
        self.cut_toggle_action = self._make_action("Iniciar/terminar corte aqui", self._toggle_cut_mark, "X")
        self.remove_cut_action = self._make_action("Remover corte selecionado", self._remove_selected_cut, "Del")
        self.remove_cut_action.setShortcuts([QKeySequence("Del"), QKeySequence("Backspace")])
        # Bare keys must not steal arrows from sliders or Space from buttons: as QAction shortcuts they
        # only fire while the preview has focus; everywhere else keyPressEvent handles what widgets left.
        for action in self._transport_actions():
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            self.preview.addAction(action)
        self.zoom_fit_action = self._make_action("Ajustar à janela", self._zoom_to_fit, "Ctrl+0")
        self.zoom_in_action = self._make_action(
            "Aumentar zoom", lambda: self._step_preview_zoom(1), QKeySequence.StandardKey.ZoomIn
        )
        self.zoom_out_action = self._make_action(
            "Diminuir zoom", lambda: self._step_preview_zoom(-1), QKeySequence.StandardKey.ZoomOut
        )
        self.hud_action = QAction("Métricas", self)
        self.hud_action.setCheckable(True)
        self.hud_action.setShortcut("Ctrl+M")
        self.hud_action.setToolTip("Mostrar métricas de render, fila e decode sobre o preview (Ctrl+M)")
        self.hud_action.toggled.connect(self._toggle_hud)
        self.diagnostics_action = self._make_action("Diagnóstico do sistema…", self._show_diagnostics)
        self.open_logs_action = self._make_action("Abrir pasta de logs", self._open_log_folder)
        self.about_action = self._make_action("Sobre", self._show_about_dialog)

    def _make_action(
        self,
        text: str,
        slot: Callable[[], object],
        shortcut: QKeySequence.StandardKey | str | None = None,
        *,
        icon: QIcon | None = None,
    ) -> QAction:
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(icon)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        return action

    def _build_ui(self) -> None:
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        left_layout.addWidget(self.preview_scroll, 1)
        left_layout.addWidget(self.transport)

        self.settings_tabs = QTabWidget()
        self.settings_tabs.setObjectName("settingsTabs")
        for title, tab in self._tabs:
            self.settings_tabs.addTab(tab, title)
        self._filters_tab_index = self.settings_tabs.indexOf(self.filters_tab)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self.settings_tabs)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setHandleWidth(10)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 12, 18, 6)
        layout.addWidget(self.splitter)
        self.setCentralWidget(central)

        status_bar = self.statusBar()
        status_bar.setSizeGripEnabled(False)
        status_bar.addWidget(self.status, 1)

        self.setStyleSheet(load_stylesheet())
        self._fit_settings_panel()

    def _fit_settings_panel(self) -> None:
        """Never clip the tabs: the panel's minimum width is what its tab bar and widest content need."""
        for child in self.settings_tabs.findChildren(QWidget):
            child.ensurePolished()
        tab_bar = self.settings_tabs.tabBar()
        tab_bar.ensurePolished()
        needed = tab_bar.sizeHint().width() + 8
        for _title, tab in self._tabs:
            needed = max(needed, tab.content_min_width())
        self.settings_tabs.setMinimumWidth(needed)
        self.splitter.setSizes([10_000, needed])  # relative weights: the panel gets its minimum, preview the rest

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Principal", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toolbar.setIconSize(QSize(16, 16))
        toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.add_video_action)
        toolbar.addAction(self.export_action)
        for action in (self.open_action, self.export_action):
            widget = toolbar.widgetForAction(action)
            if widget is not None:
                widget.setObjectName("toolbarPrimary")
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Zoom"))
        toolbar.addWidget(self.preview_zoom_out_button)
        toolbar.addWidget(self.preview_zoom_combo)
        toolbar.addWidget(self.preview_zoom_in_button)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Qualidade"))
        toolbar.addWidget(self.preview_render_mode_combo)
        toolbar.addWidget(QLabel("Reprodução"))
        toolbar.addWidget(self.preview_frame_skip_combo)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addAction(self.hud_action)
        self.toolbar = toolbar
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Arquivo")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.add_video_action)
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        edit_menu = menu_bar.addMenu("Editar")
        edit_menu.addAction(self.restore_preset_action)
        edit_menu.addAction(self.reset_crop_action)
        edit_menu.addAction(self.reset_range_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.split_clip_action)
        edit_menu.addAction(self.remove_clip_action)

        playback_menu = menu_bar.addMenu("Reprodução")
        playback_menu.addAction(self.play_action)
        playback_menu.addSeparator()
        playback_menu.addAction(self.first_frame_action)
        playback_menu.addAction(self.previous_frame_action)
        playback_menu.addAction(self.next_frame_action)
        playback_menu.addAction(self.last_frame_action)
        playback_menu.addSeparator()
        playback_menu.addAction(self.mark_start_action)
        playback_menu.addAction(self.mark_end_action)
        playback_menu.addAction(self.cut_toggle_action)
        playback_menu.addAction(self.remove_cut_action)

        view_menu = menu_bar.addMenu("Exibir")
        view_menu.addAction(self.zoom_fit_action)
        view_menu.addAction(self.zoom_in_action)
        view_menu.addAction(self.zoom_out_action)
        view_menu.addSeparator()
        view_menu.addAction(self.hud_action)

        help_menu = menu_bar.addMenu("Ajuda")
        help_menu.addAction(self.diagnostics_action)
        help_menu.addAction(self.open_logs_action)
        help_menu.addAction(self.about_action)

    def _bind_events(self) -> None:
        self.preview.cropChanged.connect(self._handle_preview_crop_changed)
        self.preview.sampleChanged.connect(self._handle_preview_sample_changed)
        self.preview.stabilizationChanged.connect(self._handle_preview_stabilization_changed)
        self.preview.whiteBalanceChanged.connect(self._handle_preview_white_balance_changed)
        self.preview.openRequested.connect(self._select_video)

        self.settings_tabs.currentChanged.connect(self._handle_tab_changed)
        self.preview_zoom_out_button.clicked.connect(lambda: self._step_preview_zoom(-1))
        self.preview_zoom_in_button.clicked.connect(lambda: self._step_preview_zoom(1))
        self.preview_zoom_combo.currentIndexChanged.connect(self._handle_preview_zoom_changed)
        self.preview_render_mode_combo.currentIndexChanged.connect(self._handle_preview_render_mode_changed)
        self.preview_frame_skip_combo.currentIndexChanged.connect(self._handle_preview_frame_skip_changed)
        self.status.cancelExportRequested.connect(self._request_export_cancel)
        self.status.revealExportRequested.connect(self._reveal_export)

        self.transport.frameChanged.connect(self._load_frame)
        self.transport.firstRequested.connect(self._go_to_first_frame)
        self.transport.previousRequested.connect(self._go_to_previous_frame)
        self.transport.playToggled.connect(self._toggle_playback)
        self.transport.nextRequested.connect(self._go_to_next_frame)
        self.transport.lastRequested.connect(self._go_to_last_frame)
        self.transport.cutsEdited.connect(self._set_cuts_from_timeline)
        self.transport.clipRangeEdited.connect(self._set_clip_range_from_timeline)
        self.transport.cutSelected.connect(self.export_tab.select_cut)
        self.transport.cutRemoveRequested.connect(self._remove_cut)
        self.transport.cutToggleRequested.connect(self._toggle_cut_mark)
        self.export_tab.cutSelectionChanged.connect(self.transport.select_cut)
        self.transport.thumbnailsRequested.connect(self._request_filmstrip)
        self.transport.clipSplitRequested.connect(self._split_clip_at_playhead)
        self.transport.clipRemoveRequested.connect(self._remove_clip)
        self.export_tab.clipAddRequested.connect(self._select_video_to_add)
        self.export_tab.clipRemoveRequested.connect(self._remove_clip)
        self.export_tab.clipMoveRequested.connect(self._move_clip)
        self.export_tab.clipSplitRequested.connect(self._split_clip_at_playhead)
        self.export_tab.clipActivated.connect(self._go_to_clip)

        self.color_tab.presetChanged.connect(self._handle_preset_changed)
        self.color_tab.valueChanged.connect(self._handle_filter_control_change)
        self.color_tab.resetRequested.connect(self._reset_slider_to_preset)
        self.grade_tab.valueChanged.connect(self._handle_grade_change)
        self.grade_tab.neutralizeRequested.connect(self._neutralize_white_balance)
        self.grade_tab.gainsResetRequested.connect(self._reset_white_balance_gains)
        self.grade_tab.regionResetRequested.connect(self._reset_white_balance_rect)
        self.grade_tab.exposureChanged.connect(self._handle_exposure_change)
        self.grade_tab.exposureCaptureRequested.connect(self._capture_exposure_reference)
        self.grade_tab.exposureClearRequested.connect(self._clear_exposure_reference)
        self.grade_tab.exposureScopeChanged.connect(self._exposure_region_moved)
        self.filters_tab.filterSelected.connect(self._handle_image_filter_changed)
        self.filters_tab.intensityChanged.connect(self._handle_image_filter_changed)
        self.stabilization_tab.modeChanged.connect(self._handle_stabilization_mode_changed)
        self.stabilization_tab.controlChanged.connect(self._handle_stabilization_control_change)
        self.stabilization_tab.resetRequested.connect(self._reset_stabilization_rect)
        self.geometry_tab.geometryChanged.connect(self._handle_geometry_change)
        self.geometry_tab.cropModeChanged.connect(self._handle_crop_mode_changed)
        self.geometry_tab.cropResetRequested.connect(self._reset_crop_rect)
        self.noise_tab.controlChanged.connect(self._handle_filter_control_change)
        self.noise_tab.resetRequested.connect(self._reset_slider_to_preset)
        self.noise_tab.sampleResetRequested.connect(self._reset_denoise_sample_rect)
        self.export_tab.scaleChanged.connect(self._handle_filter_control_change)
        self.export_tab.markStartRequested.connect(self._mark_clip_start)
        self.export_tab.markEndRequested.connect(self._mark_clip_end)
        self.export_tab.resetRangeRequested.connect(self._reset_clip_range)
        self.export_tab.cutToggleRequested.connect(self._toggle_cut_mark)
        self.export_tab.cutAddRequested.connect(self._add_cut_from_text)
        self.export_tab.cutRemoveRequested.connect(self._remove_cut)
        self.export_tab.cutsClearRequested.connect(self._clear_cuts)
        self.export_tab.cutActivated.connect(self._go_to_cut)

    def _filter_sliders(self) -> dict[str, Any]:
        return {**self.color_tab.sliders, **self.noise_tab.sliders}

    def _set_controls_enabled(self, enabled: bool) -> None:
        for _title, tab in self._tabs:
            tab.set_controls_enabled(enabled)
        self.transport.setEnabled(enabled)
        self.export_action.setEnabled(enabled and not self.sequence.is_empty)
        self.split_clip_action.setEnabled(enabled and not self.sequence.is_empty)
        self.remove_clip_action.setEnabled(enabled and len(self.sequence) > 1)
        self.filters_tab.set_video_loaded(self.frame_count > 0)
        self._update_denoise_controls_state()
        self._update_stabilization_controls_state()

    # =====================================================================================
    # Settings snapshot (the only place widgets are read for processing)
    # =====================================================================================
    def _current_filter_settings(self) -> FilterSettings:
        color = self.color_tab
        noise = self.noise_tab
        return FilterSettings(
            denoise_mode=noise.mode(),
            denoise_strength=noise.strength_slider.value(),
            denoise_color_strength=noise.color_slider.value(),
            denoise_template_window=noise.template_window(),
            denoise_search_window=noise.search_window(),
            sharpen=color.sharpen_slider.value(),
            contrast=color.contrast_slider.value(),
            brightness=color.brightness_slider.value(),
            gamma=color.gamma_slider.value(),
            saturation=color.saturation_slider.value(),
            temperature=color.temperature_slider.value(),
            scale_factor=self.export_tab.scale_factor(),
        )

    def _current_geometry(self) -> GeometrySettings:
        return self.geometry_tab.geometry_settings()

    def _current_crop(self) -> CropSettings:
        return CropSettings(
            enabled=self.geometry_tab.crop_enabled(),
            rect=qrect_to_norm(self.crop_rect_norm),
            aspect_ratio=self.geometry_tab.crop_aspect_ratio(),
        )

    def _current_stabilization(self) -> StabilizationSettings:
        return self.stabilization_tab.settings(qrect_to_norm(self.stabilization_rect_norm))

    def _current_render_settings(self) -> RenderSettings:
        return RenderSettings(
            filters=self._current_filter_settings(),
            image_filter=self.filters_tab.selection(),
            geometry=self._current_geometry(),
            crop=self._current_crop(),
            stabilization=self._current_stabilization(),
            denoise_sample_rect=qrect_to_norm(self.denoise_sample_rect_norm),
            grade=replace(
                self.grade_tab.settings(self.white_balance_gains), region=qrect_to_norm(self.white_balance_rect_norm)
            ),
            exposure=self.grade_tab.exposure_settings(qrect_to_norm(self.white_balance_rect_norm)),
        )

    def _preview_render_mode(self) -> PreviewRenderMode:
        return combo_enum(self.preview_render_mode_combo, PreviewRenderMode, PreviewRenderMode.HIGH)

    # =====================================================================================
    # Media: the sequence and its sources
    # =====================================================================================
    @property
    def frame_count(self) -> int:
        """Timeline frames of the whole sequence (cuts included)."""
        return self.sequence.timeline_count

    @property
    def fps(self) -> float:
        return self.sequence.fps

    @property
    def active_clip(self) -> Clip | None:
        if 0 <= self.active_clip_index < len(self.sequence.clips):
            return self.sequence.clips[self.active_clip_index]
        return None

    @property
    def video_path(self) -> Path | None:
        """File of the active clip (the first clip's before any frame was shown)."""
        clip = self.active_clip
        if clip is None and self.sequence.clips:
            clip = self.sequence.clips[0]
        return clip.source.path if clip is not None else None

    @property
    def video(self) -> VideoSource | None:
        """Decoder of the active clip's file (adopted by the preview worker; read metadata only)."""
        path = self.video_path
        return self._sources.get(path) if path is not None else None

    @property
    def source_size(self) -> tuple[int, int]:
        """Frame size of the active clip's source (the regions and labels refer to it)."""
        clip = self.active_clip
        if clip is None and self.sequence.clips:
            clip = self.sequence.clips[0]
        return (clip.source.width, clip.source.height) if clip is not None else (0, 0)

    @property
    def cuts(self) -> list[tuple[int, int]]:
        """Removed ranges in timeline frames (inclusive), merged and sorted."""
        return [(cut.start, cut.end) for cut in self.sequence.timeline_cuts()]

    def _select_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar vídeo", str(Path.home()), VIDEO_FILE_FILTER)
        if file_path:
            self._open_video(Path(file_path))

    def _select_video_to_add(self) -> None:
        start_dir = str(self.video_path.parent) if self.video_path is not None else str(Path.home())
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Adicionar vídeos à sequência", start_dir, VIDEO_FILE_FILTER)
        for file_path in file_paths:
            self._add_video(Path(file_path))

    def _open_source(self, path: Path) -> ClipSource | None:
        """Open (or reuse) the decoder for ``path`` and describe the file; ``None`` with a message on failure."""
        video = self._sources.get(path)
        if video is None:
            opened_at = time.perf_counter()
            try:
                video = VideoSource(path)
            except VideoOpenError as error:
                log_event(LOG, "video.open_failed", level=logging.WARNING, name=path.name, error=str(error))
                QMessageBox.critical(self, "Erro", f"Não foi possível abrir o vídeo {path.name}.")
                return None
            self._sources[path] = video
            log_event(
                LOG,
                "video.open",
                name=path.name,
                width=video.width,
                height=video.height,
                fps=round(video.fps, 3),
                frame_count=video.frame_count,
                codec=video.codec,
                open_ms=round((time.perf_counter() - opened_at) * 1000.0, 1),
            )
        return ClipSource(path, video.frame_count, video.fps, video.width, video.height)

    def _fresh_clip_settings(self, source: ClipSource) -> RenderSettings:
        """Settings for a clip that is new to the sequence: the current look, with regions and references reset."""
        base = self._current_render_settings()
        crop_aspect = self.geometry_tab.crop_aspect_ratio()
        displayed_width, displayed_height = base.geometry.output_size(source.width, source.height)
        displayed_aspect = displayed_width / displayed_height if displayed_width > 0 and displayed_height > 0 else 1.0
        crop_rect = self._default_crop_rect(None if crop_aspect is None else crop_aspect / displayed_aspect)
        return replace(
            base,
            crop=replace(base.crop, rect=qrect_to_norm(crop_rect)),
            denoise_sample_rect=DEFAULT_DENOISE_SAMPLE_RECT,
            stabilization=replace(base.stabilization, focus_rect=DEFAULT_STABILIZATION_RECT),
            grade=replace(base.grade, white_balance_gains=NEUTRAL_GAINS, region=DEFAULT_WHITE_BALANCE_RECT),
            exposure=replace(
                base.exposure,
                reference=None,
                region=DEFAULT_WHITE_BALANCE_RECT if base.exposure.region is not None else None,
            ),
        )

    def _open_video(self, path: Path) -> None:
        """Start over with a sequence of one clip (the sliders carry over; regions and references reset)."""
        self._stop_playback()
        previous_sources = self._sources
        self._sources = {}  # a fresh decoder even for the same file; the worker releases the old ones
        source = self._open_source(path)
        if source is None:
            self._sources = previous_sources
            return
        self.active_clip_index = -1
        self.sequence = ClipSequence.single(source, settings=self._fresh_clip_settings(source))
        self._reference_frames.clear()
        self._waveforms_requested.clear()
        self.current_frame = None
        self._decoded_frame_index = 0
        self._decoded_source = None
        self._decode_stats = RollingStats(window=120)
        self._last_decode_ms = 0.0
        self._last_decode_sequential = None
        self._video_read_counters = (0, 0)
        self._backend.reset()
        self._discard_pending_previews()
        self._preview_controller.set_videos(tuple(self._sources.values()))  # the worker releases the previous ones
        self._last_logged_settings = None  # next preview logs a full settings.snapshot for this sequence
        self.current_frame_index = 0
        self.clip_start_frame = 0
        self.clip_end_frame = self.frame_count - 1
        self._pending_cut_start = None
        self._stabilization_epoch += 1
        self.transport.set_frame_count(self.frame_count, self.fps, source.aspect_ratio)
        self._sequence_changed()
        self._set_controls_enabled(True)
        self.status.show_message(
            "Arraste o divisor na imagem e ajuste os controles à direita para comparar o resultado."
        )

        midpoint = min(self.frame_count - 1, self.frame_count // 2)
        self.transport.set_value_silently(midpoint)
        self._load_frame(midpoint)
        self.preview.setFocus()

    def _add_video(self, path: Path) -> None:
        """Append ``path`` as a clip (same fps as the sequence); the new clip starts with the current look."""
        if self.sequence.is_empty:
            self._open_video(path)
            return
        self._stop_playback()
        already_open = path in self._sources
        source = self._open_source(path)
        if source is None:
            return
        self._store_active_clip_settings()
        candidate = self.sequence.add(Clip.whole(source, settings=self._fresh_clip_settings(source)))
        mismatch = candidate.fps_mismatch()
        if mismatch is not None:
            if not already_open:
                self._sources.pop(path).release()  # never handed to the worker, so releasing here is safe
            log_event(LOG, "sequence.add_refused", name=path.name, reason=mismatch)
            QMessageBox.warning(self, "fps diferente", mismatch)
            return
        covers_end = self.clip_end_frame >= self.frame_count - 1
        self.active_clip_index = -1
        self.sequence = candidate
        if covers_end:
            self.clip_end_frame = self.frame_count - 1  # an export range that reached the end keeps reaching it
        self._preview_controller.set_videos(tuple(self._sources.values()))
        log_event(LOG, "sequence.add", name=path.name, clips=len(self.sequence), frames=self.frame_count)
        self._sequence_changed()
        self.status.show_message(
            f"{path.name} adicionado à sequência: {len(self.sequence)} clipes, "
            f"{format_duration(self.frame_count / self.fps)}."
        )

    def _begin_sequence_edit(self) -> None:
        """Before changing the clip list: save the widgets into the active clip, then forget which one it was."""
        self._stop_playback()
        self._store_active_clip_settings()
        self.active_clip_index = -1

    def _split_clip_at_playhead(self) -> None:
        if self.sequence.is_empty:
            return
        self._store_active_clip_settings()
        split = self.sequence.split(self.current_frame_index)
        if split is None:
            self.status.show_message("O playhead já está no início de um clipe: nada a dividir.")
            return
        self._begin_sequence_edit()
        self.sequence = split
        log_event(LOG, "sequence.split", frame=self.current_frame_index, clips=len(self.sequence))
        self._sequence_changed()
        self.status.show_message(
            f"Clipe dividido em {format_timecode(self.current_frame_index, self.fps)}: "
            "cada parte agora tem ajustes próprios."
        )

    def _remove_clip(self, index: int) -> None:
        if not (0 <= index < len(self.sequence.clips)) or len(self.sequence) <= 1:
            return
        self._begin_sequence_edit()
        removed = self.sequence.clips[index]
        span = self.sequence.clip_span(index)
        self.sequence = self.sequence.remove(index)
        if self.current_frame_index > span.end:
            self.current_frame_index -= span.count
        elif self.current_frame_index >= span.start:
            self.current_frame_index = min(span.start, self.frame_count - 1)
        self.clip_start_frame, self.clip_end_frame = 0, self.frame_count - 1  # the range referred to the old layout
        self._prune_reference_frames()
        self._sources = {path: video for path, video in self._sources.items() if path in self.sequence.source_paths}
        self._preview_controller.set_videos(tuple(self._sources.values()))  # the worker releases the dropped ones
        log_event(LOG, "sequence.remove", name=removed.name, clips=len(self.sequence))
        self._sequence_changed()
        self.transport.set_value_silently(self.current_frame_index)
        self._load_frame(self.current_frame_index)

    def _remove_active_clip(self) -> None:
        self._remove_clip(self.active_clip_index)

    def _move_clip(self, index: int, new_index: int) -> None:
        if index == new_index or not (0 <= index < len(self.sequence.clips)):
            return
        self._begin_sequence_edit()
        local = self.current_frame_index - self.sequence.clip_span(index).start
        self.sequence = self.sequence.move(index, new_index)
        moved = max(0, min(new_index, len(self.sequence.clips) - 1))
        if 0 <= local < self.sequence.clips[moved].timeline_count:
            self.current_frame_index = self.sequence.clip_span(moved).start + local  # the playhead follows its clip
        self.clip_start_frame, self.clip_end_frame = 0, self.frame_count - 1
        log_event(LOG, "sequence.move", index=index, new_index=moved)
        self._sequence_changed()
        self.transport.set_value_silently(self.current_frame_index)
        self._load_frame(self.current_frame_index)

    def _go_to_clip(self, index: int) -> None:
        """The clip picked in the export tab's list: put the playhead on it (that makes it the active clip)."""
        if 0 <= index < len(self.sequence.clips) and self.sequence.clip_at(self.current_frame_index) != index:
            self._stop_playback()
            self.transport.set_value(self.sequence.clip_span(index).start)

    def _sequence_changed(self) -> None:
        """The clips changed (added, removed, moved, split): push the layout everywhere it is shown."""
        last = self.frame_count - 1
        self.clip_start_frame = max(0, min(self.clip_start_frame, last))
        self.clip_end_frame = max(self.clip_start_frame, min(self.clip_end_frame, last))
        self.current_frame_index = max(0, min(self.current_frame_index, last))
        if self._pending_cut_start is not None and self._pending_cut_start > last:
            self._pending_cut_start = None
        self.transport.set_clips(self._clip_blocks())
        self._filmstrip.set_videos(self.sequence.source_paths)
        self._load_waveforms()
        self._sync_active_clip(force=True)
        self.export_tab.set_clips(self._clip_items(), self.active_clip_index)
        self._update_timeline_markers()
        self._update_range_label()
        self._update_sequence_info()
        self.export_action.setEnabled(not self.sequence.is_empty and self._export_controller is None)
        self.remove_clip_action.setEnabled(len(self.sequence) > 1 and self._export_controller is None)
        self.split_clip_action.setEnabled(not self.sequence.is_empty and self._export_controller is None)

    def _clip_blocks(self) -> list[ClipBlock]:
        offsets = self.sequence.clip_offsets()
        return [
            ClipBlock(offset, clip.timeline_count, clip.name, clip.source.path, clip.range.start)
            for offset, clip in zip(offsets, self.sequence.clips)
        ]

    def _clip_items(self) -> list[str]:
        items = []
        for number, (offset, clip) in enumerate(zip(self.sequence.clip_offsets(), self.sequence.clips), start=1):
            text = (
                f"{number}. {clip.name} · {format_timecode(offset, self.fps)} → "
                f"{format_timecode(offset + clip.timeline_count, self.fps)} ({clip.timeline_count} frames)"
            )
            if clip.range != clip.source.full_range:
                text += f" · frames {clip.range.start + 1}–{clip.range.end + 1} do arquivo"
            items.append(text)
        return items

    def _update_sequence_info(self) -> None:
        clip = self.active_clip
        if clip is None:
            self.status.set_video_info("")
            self.setWindowTitle("Space Video Enhancer")
            return
        first = self.sequence.clips[0]
        width, height = clip.source.width, clip.source.height
        if len(self.sequence) == 1:
            info = f"{clip.name} · {width}x{height} · {self.fps:.2f} fps · {self.frame_count} frames"
            title = f"{clip.name} — Space Video Enhancer"
        else:
            info = (
                f"{len(self.sequence)} clipes · {self.frame_count} frames · {self.fps:.2f} fps · "
                f"clipe ativo {self.active_clip_index + 1}: {clip.name} ({width}x{height})"
            )
            title = f"{first.name} +{len(self.sequence) - 1} — Space Video Enhancer"
        self.status.set_video_info(info)
        self.setWindowTitle(title)

    def _load_waveforms(self) -> None:
        """One ffmpeg decode per file of the sequence (files already requested are not decoded again)."""
        ffmpeg_path = find_ffmpeg()
        for path in self.sequence.source_paths:
            if path in self._waveforms_requested:
                continue
            self._waveforms_requested.add(path)
            if ffmpeg_path is None:
                self.transport.set_audio_state(path, "unavailable")
            else:
                self.transport.set_audio_state(path, "loading")
                self._waveform.load(path, ffmpeg_path, self.fps * WAVEFORM_BUCKETS_PER_FRAME)

    def _canvas_aspect(self) -> float | None:
        """Width/height of the export canvas (the first clip's output); ``None`` when there is nothing to fit."""
        if len(self.sequence) < 2:
            return None
        first = self.sequence.clips[0]
        width, height = first.settings.output_size(first.source.width, first.source.height)
        return width / height if width > 0 and height > 0 else None

    # --- the active clip and its settings ------------------------------------------------------
    def _sync_active_clip(self, *, force: bool = False) -> None:
        """Make the clip under the playhead the active one: save the old clip's widgets, load the new one's."""
        if self.sequence.is_empty:
            self.active_clip_index = -1
            return
        index = self.sequence.clip_at(self.current_frame_index)
        if not force and index == self.active_clip_index:
            return
        self._store_active_clip_settings()
        self.active_clip_index = index
        self._load_clip_into_controls(self.sequence.clips[index])
        self._stabilization_epoch += 1  # inter-frame memory must not cross into another clip
        self.transport.set_active_clip(index)
        self.export_tab.select_clip(index)
        self._update_sequence_info()
        log_event(LOG, "ui.clip_active", clip=index, name=self.sequence.clips[index].name)

    def _store_active_clip_settings(self) -> None:
        """Snapshot the widgets into the active clip (called on every render and before switching clips)."""
        clip = self.active_clip
        if clip is None:
            return
        settings = self._current_render_settings()
        if settings != clip.settings:
            self.sequence = self.sequence.with_clip_settings(self.active_clip_index, settings)

    def _update_active_clip(self, update: Callable[[Clip], Clip]) -> None:
        clip = self.active_clip
        if clip is not None:
            self.sequence = self.sequence.replace_clip(self.active_clip_index, update(clip))

    def _load_clip_into_controls(self, clip: Clip) -> None:
        """Show ``clip.settings`` in every tab and overlay without triggering the change handlers."""
        settings = clip.settings
        self.color_tab.apply(settings.filters)
        self.noise_tab.apply(settings.filters)
        self.export_tab.apply(settings.filters)
        self.filters_tab.apply(settings.image_filter)
        self.geometry_tab.apply(settings.geometry, settings.crop)
        self.crop_rect_norm = norm_to_qrect(settings.crop.rect)
        self.denoise_sample_rect_norm = norm_to_qrect(settings.denoise_sample_rect)
        self.stabilization_tab.apply(settings.stabilization)
        self.stabilization_rect_norm = norm_to_qrect(settings.stabilization.focus_rect)
        self.white_balance_gains = settings.grade.white_balance_gains
        self.white_balance_rect_norm = norm_to_qrect(settings.grade.region)
        self.grade_tab.apply(settings.grade, settings.exposure, exposure_frame=clip.exposure_reference_frame)
        reference_frame = clip.stabilization_reference_frame
        self.stabilization_reference_frame = (
            self._reference_frames.get((clip.source.path, reference_frame)) if reference_frame is not None else None
        )
        self.stabilization_reference_frame_index = reference_frame or 0
        self._sync_preset_selection()
        self._update_denoise_controls_state()
        self._update_stabilization_controls_state()
        self._sync_overlays()
        self._update_crop_info_label()
        self._update_white_balance_info()
        self._update_exposure_info()
        self.preview.set_display_reference_size(*self._displayed_size())

    def _prune_reference_frames(self) -> None:
        wanted = {
            (clip.source.path, clip.stabilization_reference_frame)
            for clip in self.sequence.clips
            if clip.stabilization_reference_frame is not None
        }
        for key in [key for key in self._reference_frames if key not in wanted]:
            del self._reference_frames[key]

    def _load_frame(self, frame_index: int) -> None:
        """Ask the render thread for timeline frame ``frame_index``; decoding happens there, never on the GUI."""
        if self.sequence.is_empty:
            return
        playing = self.playback_timer.isActive()
        self.current_frame_index = frame_index
        self.transport.set_position(frame_index)
        self._sync_active_clip()
        self._schedule_preview_refresh(immediate=True, reason="playback" if playing else "scrub")

    def _go_to_first_frame(self) -> None:
        if self.frame_count > 0:
            self._stop_playback()
            self.transport.set_value(0)

    def _go_to_previous_frame(self) -> None:
        if self.frame_count > 0:
            self._stop_playback()
            self.transport.set_value(max(0, self.transport.value() - 1))

    def _go_to_next_frame(self) -> None:
        if self.frame_count > 0:
            self._stop_playback()
            self.transport.set_value(min(self.transport.maximum(), self.transport.value() + 1))

    def _go_to_last_frame(self) -> None:
        if self.frame_count > 0:
            self._stop_playback()
            self.transport.set_value(self.transport.maximum())

    def _toggle_playback(self) -> None:
        if self.frame_count <= 1:
            return
        if self.playback_timer.isActive():
            self._stop_playback()
            return

        current_frame = self.transport.value()
        start_frame, end_frame = self._selected_frame_range()
        full_video_selected = start_frame == 0 and end_frame == self.transport.maximum()
        if not full_video_selected and not (start_frame <= current_frame <= end_frame):
            current_frame = start_frame
            self.transport.set_value(current_frame)

        self.playback_end_frame = end_frame if not full_video_selected else self.transport.maximum()
        kept = next_kept_frame(current_frame, self.cuts, self.playback_end_frame)
        if kept is None:
            self.status.show_message("Os cortes removem tudo daqui até o fim do trecho selecionado.")
            return
        if kept != current_frame:
            current_frame = kept
            self.transport.set_value(current_frame)
        if current_frame >= self.playback_end_frame:
            self._stop_playback()
            return

        self.transport.set_playing(True)
        self._playback_started_at = time.monotonic()
        self._playback_frames_shown = 0
        self._playback_intervals = RollingStats(window=60)
        self._playback_stabilization = Counter()
        self._playback_last_frame_at = None
        self._playback_video_counters = self._video_read_counters
        log_event(
            LOG,
            "playback.start",
            frame=current_frame,
            end_frame=self.playback_end_frame,
            interval_ms=self._playback_interval_ms(),
            frame_skip=int(self.preview_frame_skip_combo.currentData() or 0),
        )
        self.playback_timer.start(self._playback_interval_ms())

    def _stop_playback(self) -> None:
        was_active = self.playback_timer.isActive()
        if was_active:
            self.playback_timer.stop()
        self.transport.set_playing(False)
        if was_active:
            self._log_playback_stop()
            self._schedule_filter_thumbnails_refresh()
            self._update_hud()

    def _log_playback_stop(self) -> None:
        seconds = time.monotonic() - self._playback_started_at if self._playback_started_at else 0.0
        seeks_before, sequential_before = self._playback_video_counters
        seeks_now, sequential_now = self._video_read_counters
        fields: dict[str, Any] = {
            "seconds": round(seconds, 2),
            "frames_shown": self._playback_frames_shown,
            "effective_fps": round(self._playback_frames_shown / seconds, 2) if seconds > 0 else None,
            "dropped_total": self._preview_controller.mailbox.dropped,
            "seeks": seeks_now - seeks_before,
            "sequential_reads": sequential_now - sequential_before,
            "decode_ms": self._decode_stats.summary(),
        }
        if self._playback_stabilization:
            fields["stabilization"] = dict(self._playback_stabilization)
        self._counters["playback_frames"] += self._playback_frames_shown
        log_event(LOG, "playback.stop", **fields)
        self._playback_started_at = None

    def _advance_playback_frame(self) -> None:
        if self.transport.value() >= self.playback_end_frame:
            self._stop_playback()
            return
        frame_step = 1 + int(self.preview_frame_skip_combo.currentData() or 0)
        next_frame = min(self.playback_end_frame, self.transport.value() + frame_step)
        kept = next_kept_frame(next_frame, self.cuts, self.playback_end_frame)  # jump over removed ranges
        if kept is None:
            self._stop_playback()
            return
        self.transport.set_value(kept)

    def _playback_interval_ms(self) -> int:
        preview_fps = min(max(self.fps, 1.0), PREVIEW_MAX_FPS)
        return max(PREVIEW_MIN_INTERVAL_MS, int(1000 / preview_fps))

    # --- clip range ------------------------------------------------------------------------
    def _mark_clip_start(self) -> None:
        if self.frame_count <= 0:
            return
        self.clip_start_frame = self.current_frame_index
        self.clip_end_frame = max(self.clip_end_frame, self.clip_start_frame)
        log_event(LOG, "ui.clip_range", start=self.clip_start_frame, end=self.clip_end_frame, edge="start")
        self._update_timeline_markers()
        self._update_range_label()

    def _mark_clip_end(self) -> None:
        if self.frame_count <= 0:
            return
        self.clip_end_frame = self.current_frame_index
        self.clip_start_frame = min(self.clip_start_frame, self.clip_end_frame)
        log_event(LOG, "ui.clip_range", start=self.clip_start_frame, end=self.clip_end_frame, edge="end")
        self._update_timeline_markers()
        self._update_range_label()

    def _reset_clip_range(self) -> None:
        if self.frame_count <= 0:
            return
        self.clip_start_frame = 0
        self.clip_end_frame = self.frame_count - 1
        self._set_timeline_cuts([])
        self._pending_cut_start = None
        self._update_timeline_markers()
        self._update_range_label()

    # --- cuts inside the clip -----------------------------------------------------------------
    def _set_timeline_cuts(self, cuts: Sequence[tuple[int, int]]) -> None:
        """Replace every clip's cuts from ranges in timeline frames (normalized by the model)."""
        self.sequence = self.sequence.with_timeline_cuts(cuts)

    def _toggle_cut_mark(self) -> None:
        """First press marks where a removed range starts, second press where it ends."""
        if self.frame_count <= 0:
            return
        if self._pending_cut_start is None:
            self._pending_cut_start = self.current_frame_index
            self.status.show_message(
                f"Início do corte em {format_timecode(self.current_frame_index, self.fps)} — "
                "vá até o fim do trecho e pressione X de novo."
            )
        else:
            start, self._pending_cut_start = self._pending_cut_start, None
            self._add_cut(start, self.current_frame_index)
        self._update_timeline_markers()
        self._update_range_label()

    def _add_cut(self, start: int, end: int) -> None:
        self._set_timeline_cuts([*self.cuts, (start, end)])
        log_event(LOG, "ui.cuts", action="add", start=min(start, end), end=max(start, end), cuts=self.cuts)
        self._update_timeline_markers()
        self._update_range_label()
        self.status.show_message(
            f"Trecho {format_timecode(min(start, end), self.fps)} → {format_timecode(max(start, end) + 1, self.fps)} "
            "removido da exportação."
        )

    def _add_cut_from_text(self, start_text: str, end_text: str) -> None:
        if self.frame_count <= 0:
            return
        start = parse_timecode(start_text, self.fps)
        end = parse_timecode(end_text, self.fps)
        if start is None or end is None:
            self.status.show_message("Tempos inválidos: use mm:ss, h:mm:ss ou mm:ss.ff (ff = frames).")
            return
        if start == end:
            self.status.show_message("O fim do corte precisa ser diferente do início.")
            return
        self._add_cut(start, max(start, end) - 1 if end > start else end)  # "até 2:15" excludes 2:15 itself
        self.export_tab.clear_typed_cut()

    def _remove_cut(self, index: int) -> None:
        cuts = self.cuts
        if 0 <= index < len(cuts):
            removed = cuts.pop(index)
            self._set_timeline_cuts(cuts)
            log_event(LOG, "ui.cuts", action="remove", start=removed[0], end=removed[1], cuts=self.cuts)
            self._update_timeline_markers()
            self._update_range_label()

    def _remove_selected_cut(self) -> None:
        """Delete/Backspace: the cut selected on the timeline (or in the export list)."""
        index = self.transport.timeline.selected_cut()
        if index < 0:
            index = self.export_tab.selected_cut()
        if index >= 0:
            self._remove_cut(index)

    def _request_filmstrip(self, frames: Sequence[int], height: int, device_pixel_ratio: float) -> None:
        """Timeline frames missing from the filmstrip -> the (file, source frame) each of them shows."""
        if self.sequence.is_empty:
            return
        cells = []
        for frame in frames:
            location = self.sequence.locate(frame)
            cells.append(
                FilmstripCell(frame, self.sequence.clips[location.clip_index].source.path, location.source_frame)
            )
        width = round(self.transport.timeline.thumbnail_size()[0] * device_pixel_ratio)
        self._filmstrip.request(cells, width, height, device_pixel_ratio)

    def _on_filmstrip_thumbnail(self, _generation: int, frame: int, image: QImage) -> None:
        self.transport.set_thumbnail(frame, image)

    def _on_waveform_ready(self, path: Path, peaks: AudioPeaks | None) -> None:
        if path in self.sequence.source_paths:  # a result for a file that left the sequence meanwhile is dropped
            self.transport.set_waveform(path, peaks)

    def _set_cuts_from_timeline(self, cuts: Sequence[tuple[int, int]], finished: bool) -> None:
        """A cut edge or body dragged on the timeline; logged once, when the drag ends."""
        if self.frame_count <= 0:
            return
        self._set_timeline_cuts(cuts)
        if finished:
            log_event(LOG, "ui.cuts", action="drag", cuts=self.cuts)
        self._update_timeline_markers()
        self._update_range_label()

    def _set_clip_range_from_timeline(self, start: int, end: int, finished: bool) -> None:
        if self.frame_count <= 0:
            return
        self.clip_start_frame, self.clip_end_frame = start, end
        if finished:
            log_event(LOG, "ui.clip_range", start=start, end=end, edge="drag")
        self._update_timeline_markers()
        self._update_range_label()

    def _clear_cuts(self) -> None:
        if not self.cuts and self._pending_cut_start is None:
            return
        self._set_timeline_cuts([])
        self._pending_cut_start = None
        log_event(LOG, "ui.cuts", action="clear", cuts=[])
        self._update_timeline_markers()
        self._update_range_label()

    def _go_to_cut(self, index: int) -> None:
        if 0 <= index < len(self.cuts):
            self._stop_playback()
            self.transport.set_value(self.cuts[index][0])

    def _cut_items(self) -> list[str]:
        return [
            f"{number}. {format_timecode(start, self.fps)} → {format_timecode(end + 1, self.fps)} "
            f"({end - start + 1} frames)"
            for number, (start, end) in enumerate(self.cuts, start=1)
        ]

    def _selected_frame_range(self) -> tuple[int, int]:
        if self.frame_count <= 0:
            return (0, 0)
        start_frame = max(0, min(self.clip_start_frame, self.frame_count - 1))
        end_frame = max(0, min(self.clip_end_frame, self.frame_count - 1))
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame
        return (start_frame, end_frame)

    def _update_range_label(self) -> None:
        self.export_tab.set_cuts(self._cut_items())
        self.export_tab.set_pending_cut(self._pending_cut_start is not None)
        if self.frame_count <= 0:
            self.export_tab.set_range_text("Corte: vídeo inteiro")
            return
        start_frame, end_frame = self._selected_frame_range()
        clip = FrameRange(start_frame, end_frame)
        segments = kept_segments(clip, self.cuts)
        exported = frame_count(segments)
        text = (
            f"Corte: {format_timecode(start_frame, self.fps)} → {format_timecode(end_frame + 1, self.fps)} "
            f"(frames {start_frame + 1}–{end_frame + 1}, {clip.count / self.fps:.2f} s)"
        )
        if self.cuts:
            removed = clip.count - exported
            text += (
                f" · {len(self.cuts)} trecho(s) removido(s), {format_duration(removed / self.fps)} · "
                f"exporta {exported} frames ({format_duration(exported / self.fps)})"
            )
        self.export_tab.set_range_text(text)

    def _update_timeline_markers(self) -> None:
        self.transport.set_cuts(self.cuts)
        self.transport.set_pending_cut(self._pending_cut_start)
        if self.frame_count <= 0:
            self.transport.set_markers(0, 0, False)
            return
        start_frame, end_frame = self._selected_frame_range()
        self.transport.set_markers(start_frame, end_frame, True)

    # =====================================================================================
    # Preview
    # =====================================================================================
    def _handle_preview_zoom_changed(self, _index: int) -> None:
        self._apply_preview_zoom()
        percent = self.preview_zoom_combo.currentData()
        log_event(
            LOG,
            "ui.preview_zoom",
            mode="fit" if percent is None else "fixed",
            percent=round(self.preview.effective_zoom() * 100) if percent is None else int(percent),
        )

    def _apply_preview_zoom(self) -> None:
        percent = self.preview_zoom_combo.currentData()
        if percent is None:
            self.preview.set_fit_mode(True)
            self.preview_scroll.setWidgetResizable(True)
        else:
            self.preview.set_fit_mode(False)
            self.preview_scroll.setWidgetResizable(False)
            self.preview.set_zoom_factor(int(percent) / 100.0)
        self._update_zoom_buttons()

    def _update_zoom_buttons(self) -> None:
        current = self.preview.effective_zoom() * 100.0
        self.preview_zoom_in_button.setEnabled(current < PREVIEW_ZOOM_STEPS[-1] - 0.5)
        self.preview_zoom_out_button.setEnabled(current > PREVIEW_ZOOM_STEPS[0] + 0.5)
        self.zoom_in_action.setEnabled(self.preview_zoom_in_button.isEnabled())
        self.zoom_out_action.setEnabled(self.preview_zoom_out_button.isEnabled())

    def _zoom_to_fit(self) -> None:
        self.preview_zoom_combo.setCurrentIndex(0)

    def _step_preview_zoom(self, step: int) -> None:
        """Next fixed level above/below what is on screen now (also meaningful from fit mode)."""
        current = self.preview.effective_zoom() * 100.0
        if step > 0:
            candidates = [percent for percent in PREVIEW_ZOOM_STEPS if percent > current + 0.5]
            target = candidates[0] if candidates else None
        else:
            candidates = [percent for percent in PREVIEW_ZOOM_STEPS if percent < current - 0.5]
            target = candidates[-1] if candidates else None
        if target is not None:
            select_combo_data(self.preview_zoom_combo, target)

    def _handle_preview_render_mode_changed(self, _index: int) -> None:
        self.preview.set_render_quality(self._preview_render_mode() is not PreviewRenderMode.FAST)
        self._stabilization_epoch += 1
        log_event(LOG, "ui.render_mode", mode=self._preview_render_mode().value)
        self._schedule_preview_refresh(immediate=True, reason="mode")

    def _handle_preview_frame_skip_changed(self, _index: int) -> None:
        if self.playback_timer.isActive():
            self.playback_timer.start(self._playback_interval_ms())

    def _preview_output_scale_factor(self, scale_factor: float) -> float:
        mode = self._preview_render_mode()
        if mode is PreviewRenderMode.HIGH:
            return scale_factor
        if mode is PreviewRenderMode.BALANCED:
            return min(1.0, scale_factor)
        return 1.0

    def _schedule_preview_refresh(self, *, immediate: bool = False, reason: str = "settings") -> None:
        self._pending_preview_reason = reason
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

    def _handle_tab_changed(self, index: int) -> None:
        log_event(LOG, "ui.tab", tab=self.settings_tabs.tabText(index))
        self._sync_overlays()
        if self._filters_tab_visible():
            self._schedule_filter_thumbnails_refresh()

    def _tab_is_active(self, tab: TabPage) -> bool:
        return self.settings_tabs.currentWidget() is tab

    def _sync_overlays(self) -> None:
        """Each selection rectangle is editable only on its own tab; the crop stays visible elsewhere."""
        self._sync_crop_preview()
        self._sync_denoise_sample_preview()
        self._sync_stabilization_preview()
        self._sync_white_balance_preview()

    def _refresh_preview_impl(self) -> None:
        """Snapshot the widgets into the active clip and ask the render thread for the current frame."""
        if self.sequence.is_empty:
            return

        self._store_active_clip_settings()
        settings = self._current_render_settings()
        self._log_settings(settings)
        self._sync_crop_preview()
        self._sync_denoise_sample_preview()
        self._sync_stabilization_preview()
        self.preview.set_display_reference_size(*self._displayed_size())

        preview_settings = replace(
            settings,
            filters=replace(
                settings.filters, scale_factor=self._preview_output_scale_factor(settings.filters.scale_factor)
            ),
        )
        size_limit = PREVIEW_SIZE_LIMIT.get(self._preview_render_mode())
        interactive = (
            self._pending_preview_reason in INTERACTIVE_REASONS and settings.filters.denoise_mode is not DenoiseMode.OFF
        )
        if interactive:
            size_limit = min(size_limit or INTERACTIVE_DENOISE_SIZE_LIMIT, INTERACTIVE_DENOISE_SIZE_LIMIT)
        location = self.sequence.locate(self.current_frame_index)
        self._preview_generation += 1
        self.status.set_render_busy(True)
        self._preview_controller.submit_preview(
            PreviewRequest(
                generation=self._preview_generation,
                frame=None,  # decoded on the render thread
                frame_index=location.source_frame,
                settings=preview_settings,
                size_limit=size_limit,
                stabilization_reference=self.stabilization_reference_frame,
                stabilization_epoch=self._stabilization_epoch,
                reason=self._pending_preview_reason,
                source_path=self.sequence.clips[location.clip_index].source.path,
                canvas_aspect=self._canvas_aspect(),
            )
        )
        self._pending_preview_reason = "settings"
        if interactive:
            self._idle_refresh_timer.start(IDLE_REFRESH_DELAY_MS)
        else:
            self._idle_refresh_timer.stop()

    def _refresh_preview_at_full_quality(self) -> None:
        """The frame the user stopped on gets a render without the interactive size limit."""
        if self.sequence.is_empty or self.playback_timer.isActive():
            return
        self._schedule_preview_refresh(immediate=True, reason="idle")

    def _log_settings(self, settings: RenderSettings) -> None:
        """Full snapshot after opening a video; afterwards only the fields that changed."""
        if self._last_logged_settings is None:
            log_event(LOG, "settings.snapshot", settings=diff_settings(None, settings))
        else:
            changes = diff_settings(self._last_logged_settings, settings)
            if changes:
                log_event(LOG, "settings.change", changes=changes, reason=self._pending_preview_reason)
        self._last_logged_settings = settings

    @Slot(object)
    def _on_preview_ready(self, images: PreviewImages) -> None:
        if images.generation <= self._applied_preview_generation:
            return  # stale: a newer request already landed (or the video changed)
        self._applied_preview_generation = images.generation
        self.last_preview_render_ms = images.elapsed_ms
        self._last_preview = images
        self.current_frame = images.source
        self._decoded_frame_index = images.frame_index
        self._decoded_source = images.source_path
        self._counters["renders"] += 1
        if images.decode_sequential is not None:
            self._last_decode_ms = images.decode_ms
            self._last_decode_sequential = images.decode_sequential
            self._decode_stats.add(images.decode_ms)
            self._video_read_counters = (images.video_seeks, images.video_sequential_reads)
        if images.reason == "playback":
            now = time.perf_counter()
            if self._playback_last_frame_at is not None:
                self._playback_intervals.add((now - self._playback_last_frame_at) * 1000.0)
            self._playback_last_frame_at = now
            self._playback_frames_shown += 1
            if "stabilization.result" in images.trace:
                self._playback_stabilization[str(images.trace["stabilization.result"])] += 1
        self.preview.set_images(images.before, images.after)
        if images.generation >= self._preview_generation:
            self.status.set_render_busy(False)
        self._update_denoise_info()
        self._update_exposure_info()
        self._update_hud()
        self._schedule_filter_thumbnails_refresh()

    def _discard_pending_previews(self) -> None:
        """Results still in flight for the previous video must not land on the new one."""
        self._applied_preview_generation = self._preview_generation
        self._applied_thumbnail_generation = self._thumbnail_generation
        self._idle_refresh_timer.stop()
        self.status.set_render_busy(False)

    def _refresh_filter_thumbnails(self) -> None:
        self._thumbnail_refresh_timer.stop()
        if self.playback_timer.isActive() or not self._filters_tab_visible():
            return
        if self.current_frame is None:
            self.filters_tab.clear_thumbnails()
            return

        settings = self._current_render_settings()
        settings = replace(settings, filters=replace(settings.filters, scale_factor=1.0))
        ratio = self.devicePixelRatioF()  # render thumbnails in device pixels so Retina shows them crisp
        icon_size = (round(THUMBNAIL_SIZE.width() * ratio), round(THUMBNAIL_SIZE.height() * ratio))
        reference = None
        if self.stabilization_reference_frame is not None:
            reference = self._thumbnail_source_frame(self.stabilization_reference_frame, icon_size)
        self._thumbnail_generation += 1
        self._preview_controller.submit_thumbnails(
            ThumbnailRequest(
                generation=self._thumbnail_generation,
                frame=self._thumbnail_source_frame(self.current_frame, icon_size),
                settings=settings,
                stabilization_reference=reference,
                preset_ids=tuple(preset.filter_id for preset in IMAGE_FILTER_PRESETS),
                icon_size=icon_size,
                device_pixel_ratio=ratio,
            )
        )

    @Slot(object)
    def _on_thumbnails_ready(self, result: ThumbnailImages) -> None:
        if result.generation <= self._applied_thumbnail_generation:
            return
        self._applied_thumbnail_generation = result.generation
        self.filters_tab.set_thumbnails(result.images, result.device_pixel_ratio)

    @Slot(str)
    def _on_render_failed(self, message: str) -> None:
        self._counters["render_errors"] += 1
        self.status.set_render_busy(False)
        self.status.show_message(f"Falha ao renderizar o preview: {message}")

    @staticmethod
    def _thumbnail_source_frame(frame: np.ndarray, icon_size: tuple[int, int]) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = min(icon_size[0] / max(1, width), icon_size[1] / max(1, height))
        if scale >= 1.0:
            return frame.copy()
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # =====================================================================================
    # Image filter presets / slider presets
    # =====================================================================================
    def _handle_image_filter_changed(self, *_args: object) -> None:
        self._schedule_preview_refresh()

    def _handle_preset_changed(self, index: int) -> None:
        preset = self.color_tab.preset_at(index)
        if preset is None:
            self._schedule_preview_refresh()
            return
        self.active_preset_settings = preset
        log_event(LOG, "ui.preset", preset=self.color_tab.preset_name(index))
        self._apply_settings_to_controls(preset)
        self._schedule_preview_refresh()
        if self.current_frame is not None:
            self.status.show_message("Preset aplicado. Ajuste os sliders se quiser um refinamento manual.")

    def _reset_slider_to_preset(self, field_name: str) -> None:
        self._filter_sliders()[field_name].setValue(int(getattr(self.active_preset_settings, field_name)))

    def _restore_active_preset_settings(self) -> None:
        self._apply_settings_to_controls(self.active_preset_settings)
        self._sync_preset_selection()
        self._schedule_preview_refresh()
        if self.current_frame is not None:
            self.status.show_message("Ajustes restaurados para o preset ativo.")

    def _apply_settings_to_controls(self, settings: FilterSettings) -> None:
        self.color_tab.apply(settings)
        self.noise_tab.apply(settings)
        self.export_tab.apply(settings)
        self._update_denoise_controls_state()

    def _handle_filter_control_change(self, *_args: object) -> None:
        self._sync_preset_selection()
        self._update_denoise_controls_state()
        self._update_crop_info_label()
        self._schedule_preview_refresh()

    def _handle_geometry_change(self, *_args: object) -> None:
        self._update_crop_info_label()
        self._schedule_preview_refresh()

    def _sync_preset_selection(self) -> None:
        self.color_tab.select_matching_preset(self._current_filter_settings())

    # =====================================================================================
    # Denoise
    # =====================================================================================
    def _update_denoise_controls_state(self) -> None:
        self.noise_tab.update_controls_state(self.frame_count > 0)
        self._sync_denoise_sample_preview()
        self._update_denoise_sample_info_label()
        self._update_denoise_info()

    def _update_denoise_info(self) -> None:
        """Mode explanation plus the strengths the last render actually used (nothing is estimated here)."""
        mode = self.noise_tab.mode()
        if mode is DenoiseMode.OFF:
            self.noise_tab.set_info(
                "Denoise desativado: o preview e a exportação pulam esta etapa para poupar processamento."
            )
            return
        if mode is DenoiseMode.MANUAL:
            self.noise_tab.set_info(
                "Manual: ajuste força de luminância, força de cor e o tamanho das janelas do filtro."
            )
            return

        source = (
            "Auto: calcula a intensidade com base no frame atual inteiro."
            if mode is DenoiseMode.AUTO
            else "Amostra: usa a região azul selecionada no preview como referência de ruído."
        )
        if self.sequence.is_empty:
            self.noise_tab.set_info(source)
            return
        trace = self._last_preview.trace if self._last_preview is not None else {}
        estimate = trace.get("denoise.h")
        if trace.get("denoise.mode") == mode.value and isinstance(estimate, (list, tuple)) and len(estimate) == 2:
            sigma = trace.get("denoise.sigma")
            noise = f"ruído σ {sigma[0]:.1f} → " if isinstance(sigma, (list, tuple)) and len(sigma) == 2 else ""
            self.noise_tab.set_info(
                f"{source} Último render: {noise}luminância {estimate[0]:.1f} | cor {estimate[1]:.1f}."
            )
        else:
            self.noise_tab.set_info(f"{source} Aguardando o render para mostrar a estimativa…")

    def _handle_preview_sample_changed(self, rect: QRectF) -> None:
        self.denoise_sample_rect_norm = QRectF(rect)
        self._sync_denoise_sample_preview()
        self._update_denoise_sample_info_label()
        self._schedule_preview_refresh()

    # --- white balance / HSL ---------------------------------------------------------------
    def _handle_grade_change(self) -> None:
        self._schedule_preview_refresh()

    def _handle_preview_white_balance_changed(self, rect: QRectF) -> None:
        self.white_balance_rect_norm = QRectF(rect)
        self._sync_white_balance_preview()
        self._update_white_balance_info()
        self._exposure_region_moved()

    def _reset_white_balance_rect(self) -> None:
        self.white_balance_rect_norm = norm_to_qrect(DEFAULT_WHITE_BALANCE_RECT)
        self._sync_white_balance_preview()
        self._update_white_balance_info()
        self._exposure_region_moved()

    def _sync_white_balance_preview(self) -> None:
        enabled = self.frame_count > 0 and self._tab_is_active(self.grade_tab)
        self.preview.set_white_balance_state(enabled, self.white_balance_rect_norm, editing=True)

    def _update_white_balance_info(self) -> None:
        if self.frame_count <= 0:
            self.grade_tab.set_region_info("Abra um vídeo para escolher a região de cinza.")
            return
        pixels = qrect_to_norm(self.white_balance_rect_norm).to_pixels(*self.source_size)
        size = f"{pixels[2]}x{pixels[3]} px" if pixels is not None else ""
        self.grade_tab.set_region_info(
            f"Região de cinza: {size} — arraste o retângulo roxo no preview sobre algo que deveria ser cinza "
            "ou branco e clique em Neutralizar."
        )

    def _neutralize_white_balance(self, whole_frame: bool) -> None:
        """One-off: read the decoded frame's region and turn it into channel gains (stored in the settings)."""
        if self.current_frame is None:
            self.status.show_message("Abra um vídeo e aguarde o primeiro render para neutralizar as cores.")
            return
        rect = None if whole_frame else qrect_to_norm(self.white_balance_rect_norm)
        gains = neutral_gains(self.current_frame, rect)
        self._set_white_balance_gains(gains)
        log_event(LOG, "ui.white_balance", source="frame" if whole_frame else "region", gains=list(gains))
        self.status.show_message(f"Balanço de branco: ganhos B {gains[0]:.2f} · G {gains[1]:.2f} · R {gains[2]:.2f}.")

    def _reset_white_balance_gains(self) -> None:
        self._set_white_balance_gains(NEUTRAL_GAINS)

    # --- exposure lock ---------------------------------------------------------------------
    def _handle_exposure_change(self) -> None:
        self._update_exposure_info()
        self._schedule_preview_refresh()

    def _exposure_region(self) -> NormalizedRect | None:
        return qrect_to_norm(self.white_balance_rect_norm) if self.grade_tab.exposure_uses_region() else None

    def _capture_exposure_reference(self) -> None:
        """Meter the decoded frame (same frame and coordinates as the "Antes" preview) and lock to it."""
        if self.current_frame is None:
            self.status.show_message("Abra um vídeo e aguarde o primeiro render para capturar a referência.")
            return
        reference = region_medians(self.current_frame, self._exposure_region())
        frame_index = self._decoded_frame_index
        self._update_active_clip(lambda clip: clip.with_exposure_reference(frame_index))
        self.grade_tab.set_exposure_reference(reference, frame_index=frame_index)
        log_event(
            LOG,
            "ui.exposure",
            action="capture",
            frame=self._decoded_frame_index,
            reference=[round(value, 1) for value in reference],
            region=self.grade_tab.exposure_uses_region(),
        )
        self.status.show_message(f"Referência de exposição capturada no frame {self._decoded_frame_index + 1}.")
        self._handle_exposure_change()

    def _clear_exposure_reference(self, reason: str | None = None) -> None:
        if self.grade_tab.exposure_reference() is None:
            return
        self.grade_tab.set_exposure_reference(None)
        self._update_active_clip(lambda clip: clip.with_exposure_reference(None))
        log_event(LOG, "ui.exposure", action="clear", reason=reason or "user")
        if reason:
            self.status.show_message(reason)
        self._handle_exposure_change()

    def _exposure_region_moved(self) -> None:
        """The metering area changed: re-meter if we are still on the reference frame, else forget the reference."""
        if self.grade_tab.exposure_reference() is None:
            return
        clip = self.active_clip
        on_reference_frame = (
            clip is not None
            and self.current_frame is not None
            and self._decoded_source == clip.source.path
            and self._decoded_frame_index == clip.exposure_reference_frame
        )
        if on_reference_frame:
            self._capture_exposure_reference()
        else:
            self._clear_exposure_reference(
                "Região de referência alterada: volte ao frame de referência e capture a exposição de novo."
            )

    def _update_exposure_info(self) -> None:
        if self.frame_count <= 0:
            self.grade_tab.set_exposure_status("")
            return
        settings = self.grade_tab.exposure_settings(self._exposure_region())
        if not settings.enabled:
            text = "Desligado."
        elif settings.reference is None:
            text = "Ligado, mas sem referência: capture um frame bem exposto."
        else:
            trace = self._last_preview.trace if self._last_preview is not None else {}
            gains = trace.get("exposure.gains")
            if isinstance(gains, (list, tuple)) and len(gains) == 3:
                stops = gains_in_ev((float(gains[0]), float(gains[1]), float(gains[2])))
                text = (
                    f"Último render: ganhos B {gains[0]:.2f} · G {gains[1]:.2f} · R {gains[2]:.2f} ({stops:+.2f} EV)."
                )
            else:
                text = "Aguardando o render para mostrar os ganhos…"
        self.grade_tab.set_exposure_status(text)

    def _set_white_balance_gains(self, gains: tuple[float, float, float]) -> None:
        self.white_balance_gains = gains
        self.grade_tab.set_gains(gains)
        self._schedule_preview_refresh()

    def _reset_denoise_sample_rect(self) -> None:
        self.denoise_sample_rect_norm = norm_to_qrect(DEFAULT_DENOISE_SAMPLE_RECT)
        self._sync_denoise_sample_preview()
        self._update_denoise_sample_info_label()
        self._schedule_preview_refresh()

    def _sync_denoise_sample_preview(self) -> None:
        enabled = (
            self.frame_count > 0 and self.noise_tab.mode() is DenoiseMode.SAMPLE and self._tab_is_active(self.noise_tab)
        )
        self.preview.set_sample_state(enabled, self.denoise_sample_rect_norm, editing=True)

    def _update_denoise_sample_info_label(self) -> None:
        if self.frame_count <= 0:
            self.noise_tab.set_sample_info("Amostra: abra um vídeo para escolher a região de análise.")
            return
        rect = qrect_to_norm(self.denoise_sample_rect_norm)
        percent = f"{rect.width * 100:.0f}% x {rect.height * 100:.0f}%"
        pixels = rect.to_pixels(*self.source_size)
        if pixels is not None:
            self.noise_tab.set_sample_info(f"Amostra: {pixels[2]}x{pixels[3]} | {percent}")
        else:
            self.noise_tab.set_sample_info(f"Amostra: {percent}")

    # =====================================================================================
    # Stabilization
    # =====================================================================================
    def _handle_stabilization_mode_changed(self) -> None:
        self._stabilization_epoch += 1
        if self.stabilization_tab.mode().uses_focal_point:
            self._capture_stabilization_reference_from_current_frame()
        self._update_stabilization_controls_state()
        self._schedule_preview_refresh()

    def _handle_stabilization_control_change(self) -> None:
        self._stabilization_epoch += 1
        self._update_stabilization_controls_state()
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
        if self.current_frame is None or self._decoded_source is None:
            return
        self.stabilization_reference_frame = self.current_frame.copy()
        self.stabilization_reference_frame_index = self._decoded_frame_index
        frame_index = self._decoded_frame_index
        self._reference_frames[(self._decoded_source, frame_index)] = self.stabilization_reference_frame
        self._update_active_clip(lambda clip: clip.with_reference(frame_index))
        self._prune_reference_frames()

    def _update_stabilization_controls_state(self) -> None:
        self.stabilization_tab.update_controls_state(self.frame_count > 0)
        self._sync_stabilization_preview()
        self._update_stabilization_info_label()

    def _sync_stabilization_preview(self) -> None:
        enabled = (
            self.frame_count > 0
            and self.stabilization_tab.mode().uses_focal_point
            and self._tab_is_active(self.stabilization_tab)
        )
        self.preview.set_stabilization_state(enabled, self.stabilization_rect_norm, editing=True)

    def _update_stabilization_info_label(self) -> None:
        """Facts only (the mode description lives in the tab's hint and tooltips)."""
        if self.frame_count <= 0:
            self.stabilization_tab.set_info("Abra um vídeo para ajustar a estabilização.")
            return
        settings = self._current_stabilization()
        if not settings.enabled:
            self.stabilization_tab.set_info("")
            return

        facts = [
            (
                f"Crop automático ligado (margem fixa de {CORRECTION_BUDGET_FRACTION:.0%})"
                if settings.auto_crop
                else "Crop automático desligado"
            ),
            f"zoom out {'ligado' if settings.zoom_out else 'desligado'}",
            f"suavização {int(settings.temporal_smoothing * 100)}%",
        ]
        if settings.mode.uses_focal_point:
            region = settings.focus_rect.to_pixels(*self.source_size)
            if region is None:
                facts.insert(0, "Selecione a região focal no preview")
            else:
                facts.insert(0, f"Referência: frame {self.stabilization_reference_frame_index + 1}")
                facts.insert(1, f"região {region[2]}x{region[3]} px (arraste o retângulo verde no preview)")
        else:
            facts.append("exportação sequencial")
        self.stabilization_tab.set_info(" · ".join(facts) + ".")

    # =====================================================================================
    # Crop
    # =====================================================================================
    def _displayed_size(self) -> tuple[int, int]:
        """Frame size the crop rectangle refers to: source, scaled, then rotated."""
        scale = self.export_tab.scale_factor()
        width = round(self.source_size[0] * scale)
        height = round(self.source_size[1] * scale)
        return self._current_geometry().output_size(width, height)

    def _displayed_aspect_ratio(self) -> float:
        width, height = self._displayed_size()
        return width / height if width > 0 and height > 0 else 1.0

    def _normalized_crop_aspect_ratio(self) -> float | None:
        aspect_ratio = self.geometry_tab.crop_aspect_ratio()
        if aspect_ratio is None:
            return None
        return aspect_ratio / self._displayed_aspect_ratio()

    def _handle_crop_mode_changed(self) -> None:
        if self.geometry_tab.crop_enabled():
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
        self.preview.set_crop_state(
            self.geometry_tab.crop_enabled(),
            self.crop_rect_norm,
            self._normalized_crop_aspect_ratio(),
            editing=self._tab_is_active(self.geometry_tab),
        )

    def _update_crop_info_label(self) -> None:
        if not self.geometry_tab.crop_enabled():
            self.geometry_tab.set_crop_info("Crop: desativado")
            return

        if self.geometry_tab.crop_is_free():
            prefix = "Crop: Livre"
            ratio_text = self._rect_ratio_text(self.crop_rect_norm)
        else:
            prefix = f"Crop: {self.geometry_tab.crop_mode_text()}"
            ratio_text = self.geometry_tab.crop_mode_text()

        width, height = self._displayed_size()
        pixels = self._current_crop().pixel_rect(width, height)
        if pixels is not None:
            self.geometry_tab.set_crop_info(f"{prefix} | {pixels[2]}x{pixels[3]} | {ratio_text}")
            return
        self.geometry_tab.set_crop_info(
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
        if self.sequence.is_empty or self._export_controller is not None:
            return
        self._store_active_clip_settings()
        start_frame, end_frame = self._selected_frame_range()
        export_sequence = self.sequence.for_export(start_frame, end_frame)
        if export_sequence.is_empty:
            self.status.show_message("Os cortes removem o trecho inteiro: nada a exportar.")
            return

        first_path = self.sequence.clips[0].source.path
        default_name = first_path.with_stem(f"{first_path.stem}_enhanced").with_suffix(".mp4")
        output_path, _ = QFileDialog.getSaveFileName(self, "Salvar vídeo processado", str(default_name), "MP4 (*.mp4)")
        if not output_path:
            return

        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path is None:
            QMessageBox.warning(
                self,
                "FFmpeg não encontrado",
                "Instale o FFmpeg e deixe o executável disponível no PATH para exportar e muxar o áudio do vídeo.",
            )
            return

        if self._ffmpeg_probe is None or self._ffmpeg_probe.ffmpeg_path != ffmpeg_path:
            self._ffmpeg_probe = FfmpegProbe(ffmpeg_path)
        encoder = resolve_encoder(self._ffmpeg_probe)
        self._backend.reset()

        request = ExportRequest(
            export_sequence, Path(output_path), self.export_tab.worker_count(), ffmpeg_path, encoder
        )
        self._export_backend_label = f"{self._backend.label} + {encoder.label}"
        self._export_output_path = Path(output_path)
        self._export_started_at = time.monotonic()
        note = ""
        if request.effective_worker_count < request.worker_count:
            note = " · estabilização temporal usa processamento sequencial"

        controller = ExportController(request, self._backend, self)
        controller.progress.connect(self._on_export_progress)
        controller.finished.connect(self._on_export_finished)
        self._export_controller = controller
        self._begin_export_ui(
            f"Exportando {request.expected_frames} frames com {request.effective_worker_count} thread(s) · "
            f"{self._export_backend_label}{note}"
        )
        controller.start()

    def _begin_export_ui(self, message: str) -> None:
        self._stop_playback()
        self._set_controls_enabled(False)
        self.open_action.setEnabled(False)
        self.add_video_action.setEnabled(False)
        self.status.begin_export(message)

    def _end_export_ui(self) -> None:
        self._set_controls_enabled(True)
        self.open_action.setEnabled(True)
        self.add_video_action.setEnabled(True)

    @Slot(int, int)
    def _on_export_progress(self, processed_frames: int, total_frames: int) -> None:
        percent = int((processed_frames / max(1, total_frames)) * 100)
        elapsed = time.monotonic() - self._export_started_at
        fps = processed_frames / elapsed if elapsed > 0 else 0.0
        remaining = f" · restam {format_duration((total_frames - processed_frames) / fps)}" if fps > 0 else ""
        self.status.set_export_progress(
            percent,
            f"Exportando {percent}% · frame {processed_frames}/{total_frames} · {fps:.0f} fps{remaining} · "
            f"{self._export_backend_label}",
        )

    @Slot(object)
    def _on_export_finished(self, result: ExportResult) -> None:
        if self._export_controller is not None:
            self._export_controller.deleteLater()
            self._export_controller = None
        self._end_export_ui()
        self._counters["exports"] += 1
        if not result.success and not result.cancelled:
            self._counters["export_failures"] += 1

        if result.success:
            elapsed = format_duration(time.monotonic() - self._export_started_at)
            name = self._export_output_path.name if self._export_output_path else "vídeo"
            message = f"Vídeo exportado com sucesso: {name} ({elapsed})."
            if result.frame_count_mismatch:
                message += f" {result.message}"
            self.status.end_export(message, success=True)
            return

        if result.cancelled:
            self.status.end_export(result.message, success=False)
            return
        self.status.end_export("A exportação falhou. Revise os parâmetros e tente novamente.", success=False)
        QMessageBox.warning(self, "Falha na exportação", result.message)

    def _request_export_cancel(self) -> None:
        if self._export_controller is None:
            return
        self._export_controller.request_cancel()
        self.status.set_cancel_enabled(False)
        self.status.show_message("Cancelando exportação…")

    def _reveal_export(self) -> None:
        if self._export_output_path is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._export_output_path.parent)))

    # =====================================================================================
    # Keyboard
    # =====================================================================================
    def _transport_actions(self) -> tuple[QAction, ...]:
        return (
            self.play_action,
            self.first_frame_action,
            self.previous_frame_action,
            self.next_frame_action,
            self.last_frame_action,
            self.mark_start_action,
            self.mark_end_action,
            self.cut_toggle_action,
            self.remove_cut_action,
        )

    def _bare_key_handlers(self) -> dict[int, Callable[[], None]]:
        return {
            Qt.Key.Key_Space: self._toggle_playback,
            Qt.Key.Key_Left: self._go_to_previous_frame,
            Qt.Key.Key_Right: self._go_to_next_frame,
            Qt.Key.Key_Home: self._go_to_first_frame,
            Qt.Key.Key_End: self._go_to_last_frame,
            Qt.Key.Key_I: self._mark_clip_start,
            Qt.Key.Key_O: self._mark_clip_end,
            Qt.Key.Key_X: self._toggle_cut_mark,
            Qt.Key.Key_Delete: self._remove_selected_cut,
            Qt.Key.Key_Backspace: self._remove_selected_cut,
        }

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Transport keys reach here only when the focused widget did not use them.

        A focused slider keeps its arrow keys and a focused button keeps Space; from anywhere
        else the same keys drive playback and the clip markers.
        """
        if event.modifiers() in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.KeypadModifier):
            handler = self._bare_key_handlers().get(event.key())
            if handler is not None:
                handler()
                event.accept()
                return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_playback()
        self._log_session_summary()
        if self._export_controller is not None:
            self._export_controller.stop()
            self._export_controller = None
        self._filmstrip.stop()
        self._waveform.stop()
        self._preview_controller.stop()  # also releases the decoders it owns
        self._sources.clear()
        super().closeEvent(event)

    # =====================================================================================
    # Misc
    # =====================================================================================
    def _toggle_hud(self, checked: bool) -> None:
        self._hud_visible = checked
        self.preview.set_hud_visible(checked)
        log_event(LOG, "ui.hud", visible=checked)
        self._update_hud()

    def _playback_fps(self) -> float | None:
        if len(self._playback_intervals) < 2 or self._playback_intervals.mean <= 0:
            return None
        return 1000.0 / self._playback_intervals.mean

    def _update_hud(self) -> None:
        """Compact metrics box drawn over the preview (toggle with Ctrl+M)."""
        if not self._hud_visible:
            return
        lines: list[str] = []
        images = self._last_preview
        trace: dict[str, object] = images.trace if images is not None else {}
        stages: dict[str, float] = trace.get("stages", {})  # type: ignore[assignment]
        if images is not None:
            breakdown = " ".join(f"{name}={value:.0f}" for name, value in stages.items() if "." not in name)
            lines.append(f"render {images.elapsed_ms:.0f} ms  {breakdown}")
            queue = f"fila {images.queue_wait_ms:.0f} ms  descartados {self._preview_controller.mailbox.dropped}"
            if "filters.denoise" in stages:
                queue = f"denoise {stages['filters.denoise']:.0f} ms  {queue}"
            lines.append(queue)
            if "stabilization.result" in trace:
                detail = f"  inliers {trace['stabilization.inliers']}" if "stabilization.inliers" in trace else ""
                lines.append(
                    f"estab {trace.get('stabilization.mode')} -> {trace.get('stabilization.estimator')} "
                    f"-> {trace['stabilization.result']}{detail}"
                )
        if self.video is not None:
            fps = self._playback_fps()
            playback = f"  playback {fps:.1f} fps" if fps else ""
            seeks, sequential = self._video_read_counters
            mean_decode = self._decode_stats.mean if len(self._decode_stats) else 0.0
            lines.append(
                f"decode {self._last_decode_ms:.1f} ms (med {mean_decode:.1f}, worker)  "
                f"seeks {seeks}  seq {sequential}{playback}"
            )
        backend = str(trace.get("filters.backend", self._backend.label))
        shape_in = trace.get("input_shape")
        shape_out = trace.get("output_shape")
        shapes = f"  {shape_in[1]}x{shape_in[0]} -> {shape_out[1]}x{shape_out[0]}" if shape_in and shape_out else ""  # type: ignore[index]
        zoom = f"  zoom {self.preview.effective_zoom() * 100:.0f}%"
        lines.append(f"{backend}  {self.preview_render_mode_combo.currentText()}{shapes}{zoom}")
        self.preview.set_hud_lines(lines)

    def _show_diagnostics(self) -> None:
        """OpenCL device, ICDs and ffmpeg encoders with hints — what to paste when asking for help."""
        self._diagnostics_dialog = DiagnosticsDialog(find_ffmpeg(), self)
        self._diagnostics_dialog.show()

    def _open_log_folder(self) -> None:
        session = current_session()
        if session is None:
            QMessageBox.information(self, "Logs", "O logging não foi inicializado nesta sessão.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(session.log_path.parent)))

    def _log_session_summary(self) -> None:
        log_event(
            LOG,
            "session.summary",
            seconds=round(time.monotonic() - self._session_started, 1),
            renders=self._counters["renders"],
            dropped=self._preview_controller.mailbox.dropped,
            playback_frames=self._counters["playback_frames"],
            exports=self._counters["exports"],
            export_failures=self._counters["export_failures"],
            render_errors=self._counters["render_errors"],
            video=self.video_path.name if self.video_path else None,
            clips=len(self.sequence),
        )

    def _show_about_dialog(self) -> None:
        QMessageBox.information(
            self,
            "Sobre",
            "Space Video Enhancer\n\nEditor offline com preview antes/depois, crop interativo e exportação local.",
        )
