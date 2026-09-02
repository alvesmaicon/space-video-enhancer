"""The state every section of the main window shares, declared once.

``MainWindow`` is assembled from mixins (``ui/sections/*.py``), each a slice of the window's behaviour.
They all derive from :class:`WindowState`, which declares the attributes ``MainWindow.__init__`` assigns
and stubs the methods one section calls on another, so each section type-checks on its own. Nothing here
is instantiated directly: the MRO of ``MainWindow`` puts every section before this class.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QRectF, QTimer
from PySide6.QtGui import QAction, QImage
from PySide6.QtWidgets import (
    QComboBox,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QWidget,
)

from space_video_enhancer.export.ffmpeg import FfmpegProbe
from space_video_enhancer.export.worker import ExportController
from space_video_enhancer.pipeline.filters import OpenCLBackend
from space_video_enhancer.pipeline.stabilization.mesh import MeshCorrections, MeshPath
from space_video_enhancer.pipeline.stabilization.wobble import WobbleTrack
from space_video_enhancer.project import RecentProjects
from space_video_enhancer.sequence import Clip, ClipSource
from space_video_enhancer.sequence import Sequence as ClipSequence
from space_video_enhancer.settings import (
    CropSettings,
    FilterSettings,
    GeometrySettings,
    PreviewRenderMode,
    RenderSettings,
    StabilizationKeyframe,
    StabilizationSettings,
)
from space_video_enhancer.telemetry.stats import RollingStats
from space_video_enhancer.ui.audio_preview import (
    PcmController,
    PreviewAudioPlayer,
)
from space_video_enhancer.ui.diagnostics_dialog import DiagnosticsDialog
from space_video_enhancer.ui.history import EditHistory
from space_video_enhancer.ui.mesh_analysis import MeshAnalysisController, MeshAnalysisRequest
from space_video_enhancer.ui.monitor_window import MonitorWindow
from space_video_enhancer.ui.motion_path_analysis import MotionPathController, MotionPathRequest
from space_video_enhancer.ui.preview_worker import PreviewController, PreviewImages
from space_video_enhancer.ui.reference_frames import ReferenceFrameController
from space_video_enhancer.ui.tabs.base import TabPage
from space_video_enhancer.ui.tabs.color import ColorTab
from space_video_enhancer.ui.tabs.color_page import ColorPage
from space_video_enhancer.ui.tabs.export import ExportTab
from space_video_enhancer.ui.tabs.filters import FiltersTab
from space_video_enhancer.ui.tabs.geometry import GeometryTab
from space_video_enhancer.ui.tabs.grade import ColorGradeTab
from space_video_enhancer.ui.tabs.noise import NoiseTab
from space_video_enhancer.ui.tabs.stabilization import StabilizationTab
from space_video_enhancer.ui.tabs.tempo import TempoTab
from space_video_enhancer.ui.timeline_media import FilmstripController, WaveformController
from space_video_enhancer.ui.widgets.compare_preview import ComparePreview
from space_video_enhancer.ui.widgets.status_panel import StatusPanel
from space_video_enhancer.ui.widgets.transport_bar import TransportBar
from space_video_enhancer.ui.wobble_analysis import WobbleAnalysisController, WobbleAnalysisRequest
from space_video_enhancer.video import VideoSource

if TYPE_CHECKING:
    from space_video_enhancer.ui.sections.editing import EditState  # a section: imported only for typing


class WindowState(QMainWindow):
    """Attributes assigned by ``MainWindow.__init__`` and the widgets it builds; see the module docstring."""

    # --- attributes assigned by MainWindow.__init__ and its builders ---
    sequence: ClipSequence
    _sources: dict[Path, VideoSource]
    _reference_frames: dict[tuple[Path, int], np.ndarray]
    _waveforms_requested: set[Path]
    active_clip_index: int
    _clip_clipboard: Clip | None
    current_frame_index: int
    clip_start_frame: int
    clip_end_frame: int
    playback_end_frame: int
    _pending_cut_start: int | None
    current_frame: np.ndarray | None
    _history: EditHistory[EditState]
    _missing_sources: set[Path]
    project_path: Path | None
    _saved_snapshot: tuple[EditState, int] | None
    confirm_discard: bool
    _recent_projects: RecentProjects
    active_preset_settings: FilterSettings
    crop_rect_norm: QRectF
    denoise_sample_rect_norm: QRectF
    white_balance_rect_norm: QRectF
    white_balance_gains: tuple[float, float, float]
    stabilization_rect_norm: QRectF
    focus_regions_norm: list[QRectF]
    """The object lock's tracked regions (1..MAX_FOCUS_REGIONS), per-clip like the focal rect."""
    stabilization_reference_frame: np.ndarray | None
    stabilization_reference_frame_index: int
    _manual_keyframes: tuple[StabilizationKeyframe, ...]
    _onion_reference: QImage | None
    _stabilization_epoch: int
    _backend: OpenCLBackend
    _preview_controller: PreviewController
    _filmstrip: FilmstripController
    _waveform: WaveformController
    _reference_loader: ReferenceFrameController
    _wobble_analysis: WobbleAnalysisController
    _wobble_tracks: list[WobbleTrack]
    """Rolling-shutter wobble analyses done so far (one per source range); see ``_wobble_track_for``."""
    _wobble_pending: list[WobbleAnalysisRequest]
    """Analyses queued or running (one per source range); a clip covered by one waits instead of re-queuing."""
    _wobble_progress: dict[int, tuple[int, int]]
    """``(frames done, total)`` per pending analysis generation, for the progress text."""
    _mesh_analysis: MeshAnalysisController
    _mesh_tracks: list[MeshPath]
    """Mesh stabilization analyses done so far (one per source range); see ``_mesh_track_for``."""
    _mesh_pending: list[MeshAnalysisRequest]
    _mesh_progress: dict[int, tuple[int, int]]
    _mesh_corrections: dict[tuple[int, int, int, str, tuple[tuple[int, int, int, int], ...]], MeshCorrections]
    """Correction fields per ``(id(track), smoothing/margin per mil, mode, regions)``: recomputed on change."""
    _readout_measure_pending: set[tuple[Path, int, int]]
    """Clip ranges whose 'Medir do clipe' waits for the wobble analysis; filled when it lands."""
    _motion_path_analysis: MotionPathController
    """Tracks a clip once to fill manual keyframes with the measured motion; see ``_fill_manual_keyframes``."""
    _motion_fill_request: MotionPathRequest | None
    """The fill analysis in flight, or ``None``; a late result that is not this one is dropped."""
    _preview_generation: int
    _applied_preview_generation: int
    _thumbnail_generation: int
    _applied_thumbnail_generation: int
    last_preview_render_ms: float
    _last_preview: PreviewImages | None
    _last_logged_settings: RenderSettings | None
    _pending_preview_reason: str
    _session_started: float
    _counters: Counter[str]
    _playback_started_at: float | None
    _playback_frames_shown: int
    _playback_intervals: RollingStats
    _playback_stabilization: Counter[str]
    _playback_last_frame_at: float | None
    _playback_video_counters: tuple[int, int]
    _playback_kept: np.ndarray
    _playback_anchor_time: float
    _playback_anchor_index: int
    _playback_seeking: bool
    _playback_audio_samples: int
    _audio_player: PreviewAudioPlayer
    _pcm_loader: PcmController
    _pcm_requested: set[tuple[Path, float]]
    _audio_warned: bool
    _decoded_frame_index: int
    _decoded_source: Path | None
    _decode_stats: RollingStats
    _last_decode_ms: float
    _last_decode_sequential: bool | None
    _video_read_counters: tuple[int, int]
    _hud_visible: bool
    _vectors_visible: bool
    _ffmpeg_probe: FfmpegProbe | None
    _export_controller: ExportController | None
    _export_backend_label: str
    _export_started_at: float
    _export_output_path: Path | None
    _monitor: MonitorWindow | None
    _main_preview_paused: bool
    playback_timer: QTimer
    _preview_refresh_timer: QTimer
    _thumbnail_refresh_timer: QTimer
    _idle_refresh_timer: QTimer
    preview: ComparePreview
    preview_scroll: QScrollArea
    transport: TransportBar
    status: StatusPanel
    color_tab: ColorTab
    grade_tab: ColorGradeTab
    color_page: ColorPage
    filters_tab: FiltersTab
    stabilization_tab: StabilizationTab
    geometry_tab: GeometryTab
    noise_tab: NoiseTab
    tempo_tab: TempoTab
    export_tab: ExportTab
    _tabs: tuple[tuple[str, TabPage], ...]
    preview_zoom_combo: QComboBox
    preview_zoom_out_button: QToolButton
    preview_zoom_in_button: QToolButton
    preview_render_mode_combo: QComboBox
    preview_compare_combo: QComboBox
    preview_speed_combo: QComboBox
    open_action: QAction
    add_video_action: QAction
    open_project_action: QAction
    save_project_action: QAction
    save_project_as_action: QAction
    clear_recent_action: QAction
    export_action: QAction
    split_clip_action: QAction
    copy_clip_action: QAction
    paste_clip_action: QAction
    paste_attributes_action: QAction
    remove_clip_action: QAction
    quit_action: QAction
    undo_action: QAction
    redo_action: QAction
    restore_preset_action: QAction
    reset_crop_action: QAction
    reset_range_action: QAction
    play_action: QAction
    first_frame_action: QAction
    previous_frame_action: QAction
    next_frame_action: QAction
    last_frame_action: QAction
    mark_start_action: QAction
    mark_end_action: QAction
    cut_toggle_action: QAction
    remove_cut_action: QAction
    clear_cuts_action: QAction
    zoom_fit_action: QAction
    zoom_in_action: QAction
    zoom_out_action: QAction
    hud_action: QAction
    vectors_action: QAction
    monitor_action: QAction
    pause_preview_action: QAction
    diagnostics_action: QAction
    open_logs_action: QAction
    about_action: QAction
    preview_stack: QStackedWidget
    settings_tabs: QTabWidget
    _filters_tab_index: int
    splitter: QSplitter
    hud_button: QToolButton
    vectors_button: QToolButton
    monitor_button: QToolButton
    preview_bar: QWidget
    preview_resume_button: QPushButton
    close_monitor_button: QPushButton
    preview_paused_panel: QWidget
    recent_projects_menu: QMenu
    _diagnostics_dialog: DiagnosticsDialog

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

    # --- methods one section calls on another; each is implemented by the section named in its body ---

    # implemented by adjustments.py
    def _wobble_track_for(
        self, clip: Clip, stabilization: StabilizationSettings, *, want_residual: bool | None = None
    ) -> WobbleTrack | None:
        raise NotImplementedError  # adjustments

    def _on_wobble_progress(self, request: object, done: int, total: int) -> None:
        raise NotImplementedError  # adjustments

    def _on_wobble_finished(self, request: object, track: object, elapsed_ms: float) -> None:
        raise NotImplementedError  # adjustments

    def _mesh_corrections_for(self, clip: Clip, stabilization: StabilizationSettings) -> MeshCorrections | None:
        raise NotImplementedError  # adjustments

    def _on_mesh_progress(self, request: object, done: int, total: int) -> None:
        raise NotImplementedError  # adjustments

    def _on_mesh_finished(self, request: object, track: object, elapsed_ms: float) -> None:
        raise NotImplementedError  # adjustments

    def _update_capture_info(self) -> None:
        raise NotImplementedError  # adjustments

    @staticmethod
    def _default_crop_rect(aspect_ratio: float | None) -> QRectF:
        raise NotImplementedError  # adjustments

    def _displayed_size(self) -> tuple[int, int]:
        raise NotImplementedError  # adjustments

    def _sync_crop_preview(self) -> None:
        raise NotImplementedError  # adjustments

    def _sync_denoise_sample_preview(self) -> None:
        raise NotImplementedError  # adjustments

    def _sync_preset_selection(self) -> None:
        raise NotImplementedError  # adjustments

    def _sync_stabilization_preview(self) -> None:
        raise NotImplementedError  # adjustments

    def _refresh_manual_controls(self) -> None:
        raise NotImplementedError  # adjustments

    def _cancel_motion_fill(self) -> None:
        raise NotImplementedError  # adjustments

    def _sync_keyframe_markers(self) -> None:
        raise NotImplementedError  # adjustments

    def _sync_white_balance_preview(self) -> None:
        raise NotImplementedError  # adjustments

    def _update_crop_info_label(self) -> None:
        raise NotImplementedError  # adjustments

    def _update_denoise_controls_state(self) -> None:
        raise NotImplementedError  # adjustments

    def _update_denoise_info(self) -> None:
        raise NotImplementedError  # adjustments

    def _update_exposure_info(self) -> None:
        raise NotImplementedError  # adjustments

    def _update_stabilization_controls_state(self) -> None:
        raise NotImplementedError  # adjustments

    def _update_stabilization_info_label(self) -> None:
        raise NotImplementedError  # adjustments

    def _update_white_balance_info(self) -> None:
        raise NotImplementedError  # adjustments

    # implemented by editing.py
    def _commit_edit(self, label: str, key: Hashable | None = None) -> None:
        raise NotImplementedError  # editing

    def _edit_state(self) -> EditState:
        raise NotImplementedError  # editing

    def _selected_frame_range(self) -> tuple[int, int]:
        raise NotImplementedError  # editing

    def _update_history_actions(self) -> None:
        raise NotImplementedError  # editing

    def _update_timeline_markers(self) -> None:
        raise NotImplementedError  # editing

    # implemented by exporting.py
    def _update_export_summary(self) -> None:
        raise NotImplementedError  # exporting

    # implemented by main_window.py
    def _current_crop(self) -> CropSettings:
        raise NotImplementedError  # main_window

    def _current_filter_settings(self) -> FilterSettings:
        raise NotImplementedError  # main_window

    def _current_geometry(self) -> GeometrySettings:
        raise NotImplementedError  # main_window

    def _current_render_settings(self) -> RenderSettings:
        raise NotImplementedError  # main_window

    def _current_stabilization(self) -> StabilizationSettings:
        raise NotImplementedError  # main_window

    def _filter_sliders(self) -> dict[str, Any]:
        raise NotImplementedError  # main_window

    def _preview_render_mode(self) -> PreviewRenderMode:
        raise NotImplementedError  # main_window

    def _set_controls_enabled(self, enabled: bool) -> None:
        raise NotImplementedError  # main_window

    def _transport_actions(self) -> tuple[QAction, ...]:
        raise NotImplementedError  # main_window

    # implemented by media.py
    def _canvas_aspect(self) -> float | None:
        raise NotImplementedError  # media

    def _open_source(self, path: Path, *, quiet: bool = False) -> ClipSource | None:
        raise NotImplementedError  # media

    def _prune_reference_frames(self) -> None:
        raise NotImplementedError  # media

    def _sequence_changed(self) -> None:
        raise NotImplementedError  # media

    def _start_document(
        self,
        sequence: ClipSequence,
        *,
        playhead: int,
        clip_range: tuple[int, int] | None = None,
        missing: Iterable[Path] = (),
        project_path: Path | None = None,
    ) -> None:
        raise NotImplementedError  # media

    def _store_active_clip_settings(self) -> None:
        raise NotImplementedError  # media

    def _sync_active_clip(self, *, force: bool = False) -> None:
        raise NotImplementedError  # media

    def _update_active_clip(self, update: Callable[[Clip], Clip], label: str, key: Hashable | None = None) -> None:
        raise NotImplementedError  # media

    # implemented by monitor.py
    def _focus_preview_surface(self) -> None:
        raise NotImplementedError  # monitor

    def _sync_preview_pause(self) -> None:
        raise NotImplementedError  # monitor

    # implemented by playback.py
    def _go_to_first_frame(self) -> None:
        raise NotImplementedError  # playback

    def _go_to_last_frame(self) -> None:
        raise NotImplementedError  # playback

    def _go_to_next_frame(self) -> None:
        raise NotImplementedError  # playback

    def _go_to_previous_frame(self) -> None:
        raise NotImplementedError  # playback

    def _load_frame(self, frame_index: int) -> None:
        raise NotImplementedError  # playback

    def _load_preview_audio(self) -> None:
        raise NotImplementedError  # playback

    def _playback_fps(self) -> float | None:
        raise NotImplementedError  # playback

    def _stop_playback(self) -> None:
        raise NotImplementedError  # playback

    def _toggle_playback(self) -> None:
        raise NotImplementedError  # playback

    # implemented by preview.py
    def _discard_pending_previews(self) -> None:
        raise NotImplementedError  # preview

    def _schedule_filter_thumbnails_refresh(self) -> None:
        raise NotImplementedError  # preview

    def _schedule_preview_refresh(self, *, immediate: bool = False, reason: str = "settings") -> None:
        raise NotImplementedError  # preview

    def _sync_overlays(self) -> None:
        raise NotImplementedError  # preview

    def _tab_is_active(self, tab: TabPage) -> bool:
        raise NotImplementedError  # preview

    def _update_hud(self) -> None:
        raise NotImplementedError  # preview

    # implemented by projects.py
    def _confirm_discard_changes(self) -> bool:
        raise NotImplementedError  # projects

    def _document_name(self) -> str:
        raise NotImplementedError  # projects

    def _project_snapshot(self) -> tuple[EditState, int]:
        raise NotImplementedError  # projects

    def _update_window_title(self) -> None:
        raise NotImplementedError  # projects
