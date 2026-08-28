"""Main window: wires preview, transport, tabs, menus and status bar; snapshots widgets into ``RenderSettings``.

No image processing lives here. Everything the pipeline needs is read from widgets
exactly once per render into an immutable snapshot (see :mod:`space_video_enhancer.settings`).
The tabs (:mod:`space_video_enhancer.ui.tabs`) own their widgets; this class orchestrates.

This module builds the window — widgets, actions, menus, signal wiring — and reads the controls
into settings. The behaviour is split by concern into mixins under :mod:`space_video_enhancer.ui.sections`
(media, projects, editing, playback, preview, adjustments, exporting, monitor), all over the shared
state declared in :mod:`space_video_enhancer.ui.sections.state`.

The media is a :class:`~space_video_enhancer.sequence.Sequence` of clips. The playhead, the
export range and the cuts live in *timeline frames*; the tabs always show (and edit) the
settings of the clip under the playhead — the *active clip* — and every render stores the
widgets back into that clip, so switching clips swaps the whole panel.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QIcon,
    QKeyEvent,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from space_video_enhancer.export.ffmpeg import FfmpegProbe, find_ffmpeg
from space_video_enhancer.export.worker import ExportController
from space_video_enhancer.pipeline.filters import OpenCLBackend
from space_video_enhancer.project import (
    RecentProjects,
)
from space_video_enhancer.sequence import Sequence as ClipSequence
from space_video_enhancer.settings import (
    DEFAULT_CROP_RECT,
    DEFAULT_DENOISE_SAMPLE_RECT,
    DEFAULT_FILTER_SETTINGS,
    DEFAULT_STABILIZATION_RECT,
    DEFAULT_WHITE_BALANCE_RECT,
    NEUTRAL_GAINS,
    CropSettings,
    FilterSettings,
    GeometrySettings,
    PreviewRenderMode,
    RenderSettings,
    StabilizationSettings,
)
from space_video_enhancer.telemetry.log import current_session, get_logger, log_event
from space_video_enhancer.telemetry.stats import RollingStats
from space_video_enhancer.ui.audio_preview import (
    PcmController,
    PreviewAudioPlayer,
)
from space_video_enhancer.ui.diagnostics_dialog import DiagnosticsDialog
from space_video_enhancer.ui.history import EditHistory
from space_video_enhancer.ui.icons import glyph_icon
from space_video_enhancer.ui.monitor_window import MonitorWindow
from space_video_enhancer.ui.preview_worker import PreviewController, PreviewImages
from space_video_enhancer.ui.reference_frames import ReferenceFrameController
from space_video_enhancer.ui.sections.adjustments import AdjustmentsSection, norm_to_qrect, qrect_to_norm
from space_video_enhancer.ui.sections.editing import EditingSection, EditState
from space_video_enhancer.ui.sections.exporting import ExportSection
from space_video_enhancer.ui.sections.media import MediaSection
from space_video_enhancer.ui.sections.monitor import MonitorSection
from space_video_enhancer.ui.sections.playback import PlaybackSection
from space_video_enhancer.ui.sections.preview import PREVIEW_RENDER_MODE_OPTIONS, PREVIEW_ZOOM_STEPS, PreviewSection
from space_video_enhancer.ui.sections.projects import ProjectSection
from space_video_enhancer.ui.sections.state import WindowState
from space_video_enhancer.ui.tabs.base import TabPage, combo_enum, enum_combo
from space_video_enhancer.ui.tabs.color import ColorTab
from space_video_enhancer.ui.tabs.export import ExportTab
from space_video_enhancer.ui.tabs.filters import FiltersTab
from space_video_enhancer.ui.tabs.geometry import GeometryTab
from space_video_enhancer.ui.tabs.grade import ColorGradeTab
from space_video_enhancer.ui.tabs.noise import NoiseTab
from space_video_enhancer.ui.tabs.stabilization import StabilizationTab
from space_video_enhancer.ui.theme import load_stylesheet
from space_video_enhancer.ui.timeline_media import FilmstripController, WaveformController
from space_video_enhancer.ui.widgets.compare_preview import ComparePreview
from space_video_enhancer.ui.widgets.status_panel import StatusPanel
from space_video_enhancer.ui.widgets.transport_bar import TransportBar
from space_video_enhancer.video import VideoSource

LOG = get_logger("ui")

DEFAULT_WINDOW_SIZE = QSize(1280, 800)


class MainWindow(
    MediaSection,
    ProjectSection,
    EditingSection,
    PlaybackSection,
    PreviewSection,
    AdjustmentsSection,
    ExportSection,
    MonitorSection,
    WindowState,
):
    """The application window: construction and wiring here, behaviour in the section mixins."""

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
        self._history: EditHistory[EditState] = EditHistory(self._edit_state())
        """Undo/redo over (sequence, export range) snapshots (see ui/history.py); reset when a video is opened."""
        self._missing_sources: set[Path] = set()
        """Files the open project expected and were not found: their clips stay, marked, nothing of them decodes."""
        self.project_path: Path | None = None
        """The ``.svep`` file this document was opened from or saved to (``None`` for a bare video)."""
        self._saved_snapshot: tuple[EditState, int] | None = None
        """(editing state, export threads) as last opened or saved; the title shows ``*`` while they differ."""
        self.confirm_discard = True
        """Ask before discarding unsaved changes (closing, opening another video or project)."""
        self._recent_projects = RecentProjects()

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
        self._reference_loader = ReferenceFrameController(self)  # re-decodes camera-lock references of a project
        self._reference_loader.frameReady.connect(self._on_reference_frame_ready)
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
        # Playback follows the wall clock: the playhead is the kept frame the elapsed time calls for.
        self._playback_kept: np.ndarray = np.zeros(0, dtype=np.int64)
        """Timeline frames still to play, cuts removed, from the anchor on."""
        self._playback_anchor_time = 0.0
        self._playback_anchor_index = 0
        self._playback_seeking = False
        self._playback_audio_samples = 0
        self._audio_player = PreviewAudioPlayer(parent=self)
        self._pcm_loader = PcmController(self)
        self._pcm_loader.pcmReady.connect(self._on_pcm_ready)
        self._pcm_requested: set[tuple[Path, float]] = set()
        self._audio_warned = False
        # Decode metrics arrive with each render result; the GUI never touches the decoder itself.
        self._decoded_frame_index = 0
        """Source frame of ``current_frame`` (in ``_decoded_source``), for reference captures."""
        self._decoded_source: Path | None = None
        self._decode_stats = RollingStats(window=120)
        self._last_decode_ms = 0.0
        self._last_decode_sequential: bool | None = None
        self._video_read_counters = (0, 0)  # (seeks, sequential reads) after the latest decode
        self._hud_visible = False
        self._vectors_visible = False
        self._ffmpeg_probe: FfmpegProbe | None = None
        self._export_controller: ExportController | None = None
        self._export_backend_label = ""
        self._export_started_at = 0.0
        self._export_output_path: Path | None = None

        self._monitor: MonitorWindow | None = None  # the detached preview window, created on first use
        self._main_preview_paused = False  # while the monitor shows the frame, this window stops painting it

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
            ("Ajustes", self.color_tab),
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
        self.preview_zoom_combo.setToolTip(
            "Zoom do preview: ajustar à janela ou um nível fixo (100% = 1 pixel do vídeo por pixel da tela)"
        )
        self.preview_zoom_out_button = QToolButton()
        self.preview_zoom_out_button.setText("−")
        self.preview_zoom_out_button.setToolTip("Diminuir zoom do preview")
        self.preview_zoom_in_button = QToolButton()
        self.preview_zoom_in_button.setText("+")
        self.preview_zoom_in_button.setToolTip("Aumentar zoom do preview")
        for button in (self.preview_zoom_out_button, self.preview_zoom_in_button):
            button.setObjectName("previewZoomButton")
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.preview_render_mode_combo = enum_combo(PREVIEW_RENDER_MODE_OPTIONS)
        self.preview_render_mode_combo.setToolTip(
            "Qualidade do preview: troca fidelidade por velocidade só na tela; a exportação é sempre completa"
        )
        self.preview_speed_combo = self.transport.speed_combo  # lives next to the play buttons

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
        self.open_project_action = self._make_action("Abrir projeto…", self._open_project_dialog, "Ctrl+P")
        self.save_project_action = self._make_action(
            "Salvar projeto", self._save_project, QKeySequence.StandardKey.Save
        )
        self.save_project_as_action = self._make_action(
            "Salvar projeto como…", self._save_project_as, QKeySequence.StandardKey.SaveAs
        )
        self.clear_recent_action = self._make_action("Limpar lista", self._clear_recent_projects)
        self.save_project_action.setEnabled(False)
        self.save_project_as_action.setEnabled(False)
        self.export_action = self._make_action(
            "Exportar…",
            self._export_video,
            "Ctrl+E",
            icon=style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
        )
        self.split_clip_action = self._make_action("Dividir clipe no playhead", self._split_clip_at_playhead, "Ctrl+K")
        self.remove_clip_action = self._make_action("Remover clipe ativo", self._remove_active_clip)
        self.quit_action = self._make_action("Sair", self.close, QKeySequence.StandardKey.Quit)
        self.undo_action = self._make_action("Desfazer", self._undo_edit, QKeySequence.StandardKey.Undo)
        self.redo_action = self._make_action("Refazer", self._redo_edit, QKeySequence.StandardKey.Redo)
        redo_keys = list(QKeySequence.keyBindings(QKeySequence.StandardKey.Redo))
        for extra in ("Ctrl+Shift+Z", "Ctrl+Y"):  # both conventions, whatever the platform binds by default
            if QKeySequence(extra) not in redo_keys:
                redo_keys.append(QKeySequence(extra))
        self.redo_action.setShortcuts(redo_keys)
        self.undo_action.setEnabled(False)
        self.redo_action.setEnabled(False)
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
        self.clear_cuts_action = self._make_action("Limpar trechos removidos", self._clear_cuts)
        # Bare keys must not steal arrows from sliders or Space from buttons: as QAction shortcuts they
        # only fire while the preview has focus; everywhere else keyPressEvent handles what widgets left.
        for action in self._transport_actions():
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            self.preview.addAction(action)
        self.pan_action = QAction("Mão: arrastar o preview", self)
        self.pan_action.setCheckable(True)
        self.pan_action.setShortcut("H")
        self.pan_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.pan_action.setIcon(glyph_icon("hand", QColor("#1f2937")))
        self.pan_action.setToolTip(
            "Com o preview maior que a área, arraste a imagem com o botão esquerdo em vez de mover o divisor (H). "
            "O botão do meio do mouse arrasta sempre."
        )
        self.pan_action.toggled.connect(self._toggle_pan)
        self.preview.addAction(self.pan_action)
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
        self.vectors_action = QAction("Vetores de movimento", self)
        self.vectors_action.setCheckable(True)
        self.vectors_action.setShortcut("Ctrl+Shift+M")
        self.vectors_action.setToolTip(
            "Mostrar sobre o lado “Antes” os pontos que a estabilização rastreou: em verde-azulado os que "
            "definiram o movimento da câmera, em laranja os descartados (objeto em movimento) (Ctrl+Shift+M)"
        )
        self.vectors_action.toggled.connect(self._toggle_vectors)
        self.monitor_action = QAction("Preview em janela separada", self)
        self.monitor_action.setCheckable(True)
        self.monitor_action.setShortcut("Ctrl+Shift+D")
        self.monitor_action.setIcon(glyph_icon("detach", QColor("#1f2937")))
        self.monitor_action.setToolTip(
            "Abrir o preview em uma janela separada com os controles de reprodução — leve para outra tela e use "
            "F11 para tela cheia (Ctrl+Shift+D)"
        )
        self.monitor_action.toggled.connect(self._toggle_monitor)
        self.pause_preview_action = QAction("Pausar o preview desta janela", self)
        self.pause_preview_action.setCheckable(True)
        self.pause_preview_action.setChecked(True)
        self.pause_preview_action.setEnabled(False)  # only means anything while the monitor is open
        self.pause_preview_action.setToolTip(
            "Com o preview em janela separada, esta janela para de desenhar o frame (o render e a janela separada "
            "seguem iguais). As abas que ajustam regiões no preview o mostram aqui mesmo assim."
        )
        self.pause_preview_action.toggled.connect(self._handle_pause_preview_toggled)
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
        left_layout.addWidget(self._build_preview_bar())
        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(self.preview_scroll)
        self.preview_stack.addWidget(self._build_preview_paused_panel())
        left_layout.addWidget(self.preview_stack, 1)
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

    def _build_preview_bar(self) -> QWidget:
        """Zoom, render quality and the diagnostic overlays: what changes only how the preview is shown."""
        bar = QWidget()
        bar.setObjectName("previewBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self.preview_zoom_out_button)
        row.addWidget(self.preview_zoom_combo)
        row.addWidget(self.preview_zoom_in_button)
        row.addWidget(self.preview_render_mode_combo)
        self.pan_button = QToolButton()
        self.pan_button.setDefaultAction(self.pan_action)
        self.pan_button.setObjectName("previewToggle")
        self.pan_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.pan_button.setIconSize(QSize(16, 16))
        self.pan_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row.addWidget(self.pan_button)
        row.addStretch(1)
        self.hud_button = QToolButton()
        self.hud_button.setDefaultAction(self.hud_action)
        self.vectors_button = QToolButton()
        self.vectors_button.setDefaultAction(self.vectors_action)
        self.vectors_button.setText("Vetores")
        for button in (self.hud_button, self.vectors_button):
            button.setObjectName("previewToggle")
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            row.addWidget(button)
        self.monitor_button = QToolButton()
        self.monitor_button.setDefaultAction(self.monitor_action)
        self.monitor_button.setObjectName("previewToggle")
        self.monitor_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.monitor_button.setIconSize(QSize(16, 16))
        self.monitor_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row.addWidget(self.monitor_button)
        self.preview_bar = bar
        return bar

    def _build_preview_paused_panel(self) -> QWidget:
        """What stands in for the preview while the monitor is showing the frame instead."""
        panel = QWidget()
        panel.setObjectName("previewPaused")
        panel.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # the transport keys must keep working from this window
        for action in self._transport_actions():
            panel.addAction(action)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)
        layout.addStretch(1)
        title = QLabel("Preview rodando em pop-up")
        title.setObjectName("previewPausedTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        explanation = QLabel("Esta janela parou de desenhar o frame.")
        explanation.setObjectName("previewPausedHint")
        explanation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        explanation.setToolTip(
            "As abas Geometria, Ruído, Estabilizar e HSL mostram o preview aqui mesmo assim: nelas a região é "
            "arrastada sobre a imagem."
        )
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)
        self.preview_resume_button = QPushButton("Mostrar aqui também")
        self.preview_resume_button.setObjectName("secondaryButton")
        self.preview_resume_button.setToolTip("Voltar a desenhar o frame nesta janela, com a janela separada aberta")
        self.preview_resume_button.clicked.connect(lambda: self.pause_preview_action.setChecked(False))
        self.close_monitor_button = QPushButton("Fechar a janela separada")
        self.close_monitor_button.setObjectName("secondaryButton")
        self.close_monitor_button.clicked.connect(lambda: self.monitor_action.setChecked(False))
        buttons.addWidget(self.preview_resume_button)
        buttons.addWidget(self.close_monitor_button)
        buttons.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(explanation)
        layout.addLayout(buttons)
        layout.addStretch(1)
        self.preview_paused_panel = panel
        return panel

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

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Arquivo")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.add_video_action)
        file_menu.addSeparator()
        file_menu.addAction(self.open_project_action)
        self.recent_projects_menu = QMenu("Projetos recentes", self)
        self.recent_projects_menu.aboutToShow.connect(self._rebuild_recent_projects_menu)
        file_menu.addMenu(self.recent_projects_menu)
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.save_project_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        self._rebuild_recent_projects_menu()

        edit_menu = menu_bar.addMenu("Editar")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
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
        playback_menu.addAction(self.clear_cuts_action)

        view_menu = menu_bar.addMenu("Exibir")
        view_menu.addAction(self.zoom_fit_action)
        view_menu.addAction(self.zoom_in_action)
        view_menu.addAction(self.zoom_out_action)
        view_menu.addAction(self.pan_action)
        view_menu.addSeparator()
        view_menu.addAction(self.hud_action)
        view_menu.addAction(self.vectors_action)
        view_menu.addSeparator()
        view_menu.addAction(self.monitor_action)
        view_menu.addAction(self.pause_preview_action)

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
        self.preview.panRequested.connect(self._handle_preview_pan)

        self.settings_tabs.currentChanged.connect(self._handle_tab_changed)
        self.preview_zoom_out_button.clicked.connect(lambda: self._step_preview_zoom(-1))
        self.preview_zoom_in_button.clicked.connect(lambda: self._step_preview_zoom(1))
        self.preview_zoom_combo.currentIndexChanged.connect(self._handle_preview_zoom_changed)
        self.preview_render_mode_combo.currentIndexChanged.connect(self._handle_preview_render_mode_changed)
        self.preview_speed_combo.currentIndexChanged.connect(self._handle_playback_speed_changed)
        self.transport.soundToggled.connect(self._set_sound_enabled)
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
        self.transport.cutRemoveRequested.connect(self._remove_cut)
        self.transport.cutToggleRequested.connect(self._toggle_cut_mark)
        self.transport.markInRequested.connect(self._mark_clip_start)
        self.transport.markOutRequested.connect(self._mark_clip_end)
        self.transport.thumbnailsRequested.connect(self._request_filmstrip)
        self.transport.clipSplitRequested.connect(self._split_clip_at_playhead)
        self.transport.clipRemoveRequested.connect(self._remove_clip)
        self.transport.clipDeleteRequested.connect(self._remove_active_clip)
        self.transport.clipMoveRequested.connect(self._move_clip)
        self.transport.clipMoveStepRequested.connect(self._move_active_clip)
        self.export_tab.exportRequested.connect(self._export_video)
        self.export_action.changed.connect(self._sync_export_button)

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
        self.stabilization_tab.captureChanged.connect(self._handle_stabilization_control_change)
        self.stabilization_tab.resetRequested.connect(self._reset_stabilization_rect)
        self.stabilization_tab.referenceRequested.connect(self._set_stabilization_reference_here)
        self.geometry_tab.geometryChanged.connect(self._handle_geometry_change)
        self.geometry_tab.cropModeChanged.connect(self._handle_crop_mode_changed)
        self.geometry_tab.cropResetRequested.connect(self._reset_crop_rect)
        self.noise_tab.controlChanged.connect(self._handle_filter_control_change)
        self.noise_tab.resetRequested.connect(self._reset_slider_to_preset)
        self.noise_tab.sampleResetRequested.connect(self._reset_denoise_sample_rect)
        self.export_tab.scaleChanged.connect(self._handle_filter_control_change)
        self.export_tab.thread_combo.currentIndexChanged.connect(lambda _index: self._update_window_title())

    def _filter_sliders(self) -> dict[str, Any]:
        return {**self.color_tab.sliders, **self.noise_tab.sliders}

    def _set_controls_enabled(self, enabled: bool) -> None:
        for _title, tab in self._tabs:
            tab.set_controls_enabled(enabled)
        self.transport.setEnabled(enabled)
        if self._monitor is not None:
            self._monitor.set_controls_enabled(enabled)
        self.export_action.setEnabled(enabled and not self.sequence.is_empty)
        self.save_project_action.setEnabled(enabled and not self.sequence.is_empty)
        self.save_project_as_action.setEnabled(enabled and not self.sequence.is_empty)
        self.split_clip_action.setEnabled(enabled and not self.sequence.is_empty)
        self.remove_clip_action.setEnabled(enabled and len(self.sequence) > 1)
        self._update_history_actions()
        self._sync_export_button()
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
            capture=self.stabilization_tab.capture_settings(),
        )

    def _preview_render_mode(self) -> PreviewRenderMode:
        return combo_enum(self.preview_render_mode_combo, PreviewRenderMode, PreviewRenderMode.HIGH)

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
            Qt.Key.Key_H: self.pan_action.toggle,
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
        if not self._confirm_discard_changes():
            event.ignore()
            return
        self._stop_playback()
        if self._monitor is not None:
            self._monitor.close()
        self._log_session_summary()
        if self._export_controller is not None:
            self._export_controller.stop()
            self._export_controller = None
        self._filmstrip.stop()
        self._waveform.stop()
        self._reference_loader.stop()
        self._audio_player.stop()
        self._pcm_loader.stop()
        self._preview_controller.stop()  # also releases the decoders it owns
        self._sources.clear()
        super().closeEvent(event)

    # =====================================================================================
    # Misc
    # =====================================================================================
    def report_previous_session_failure(self, crash_log: Path, report: str | None) -> None:
        """Tell the user the last session died and where the evidence is (a crash dump, or just a missing goodbye)."""
        if report is not None:
            self.status.show_message(
                f"A sessão anterior fechou com um erro fatal; a pilha está em {crash_log.name} "
                "(Ajuda › Abrir pasta de logs). Mande esse arquivo junto com o relato do que estava fazendo."
            )
        else:
            self.status.show_message(
                "A sessão anterior não fechou normalmente (sem relatório de erro: fechada à força ou queda de energia)."
            )

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
