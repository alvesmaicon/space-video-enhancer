"""Opening files, the sequence of clips and their decoders, and the active clip whose settings the tabs show."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from space_video_enhancer.audio import AudioPeaks
from space_video_enhancer.export.ffmpeg import find_ffmpeg
from space_video_enhancer.sequence import Clip, ClipSource
from space_video_enhancer.sequence import Sequence as ClipSequence
from space_video_enhancer.settings import (
    DEFAULT_DENOISE_SAMPLE_RECT,
    DEFAULT_STABILIZATION_RECT,
    DEFAULT_WHITE_BALANCE_RECT,
    NEUTRAL_GAINS,
    RenderSettings,
)
from space_video_enhancer.telemetry.diff import diff as diff_settings
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.telemetry.stats import RollingStats
from space_video_enhancer.ui.format import format_duration, format_timecode
from space_video_enhancer.ui.sections.adjustments import norm_to_qrect, qrect_to_norm
from space_video_enhancer.ui.sections.editing import settings_edit_label
from space_video_enhancer.ui.sections.state import WindowState
from space_video_enhancer.ui.timeline_media import FilmstripCell
from space_video_enhancer.ui.widgets.timeline import ClipBlock
from space_video_enhancer.video import VideoOpenError, VideoSource

LOG = get_logger("ui")

VIDEO_FILE_FILTER = "Vídeos (*.mp4 *.mov *.mkv *.avi *.m4v)"


WAVEFORM_BUCKETS_PER_FRAME = 4
"""Audio peaks resolution for the timeline: four buckets per video frame keeps a 1 h clip under 4 MB."""


class MediaSection(WindowState):
    """Opening files, the sequence of clips and their decoders, and the active clip whose settings the tabs show."""

    def _select_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar vídeo", str(Path.home()), VIDEO_FILE_FILTER)
        if file_path and self._confirm_discard_changes():
            self._open_video(Path(file_path))

    def _select_video_to_add(self) -> None:
        start_dir = str(self.video_path.parent) if self.video_path is not None else str(Path.home())
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Adicionar vídeos à sequência", start_dir, VIDEO_FILE_FILTER)
        for file_path in file_paths:
            self._add_video(Path(file_path))

    def _open_source(self, path: Path, *, quiet: bool = False) -> ClipSource | None:
        """Open (or reuse) the decoder for ``path`` and describe the file; ``None`` (with a message) on failure."""
        video = self._sources.get(path)
        if video is None:
            opened_at = time.perf_counter()
            try:
                video = VideoSource(path)
            except VideoOpenError as error:
                log_event(LOG, "video.open_failed", level=logging.WARNING, name=path.name, error=str(error))
                if not quiet:
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
        sequence = ClipSequence.single(source, settings=self._fresh_clip_settings(source))
        self._start_document(sequence, playhead=min(sequence.timeline_count - 1, sequence.timeline_count // 2))
        self.status.show_message(
            "Arraste o divisor na imagem e ajuste os controles à direita para comparar o resultado."
        )

    def _start_document(
        self,
        sequence: ClipSequence,
        *,
        playhead: int,
        clip_range: tuple[int, int] | None = None,
        missing: Iterable[Path] = (),
        project_path: Path | None = None,
    ) -> None:
        """Replace the document with ``sequence`` (decoders already in ``_sources``); history and caches restart."""
        self.active_clip_index = -1
        self.sequence = sequence
        self._missing_sources = set(missing)
        self.project_path = project_path
        self._reference_frames.clear()
        self._wobble_analysis.cancel()
        self._wobble_pending.clear()
        self._wobble_progress.clear()
        self._wobble_tracks.clear()
        self._waveforms_requested.clear()
        self._pcm_requested.clear()
        self._audio_player.keep_only(())
        self.transport.set_sound_available(self._audio_player.available)
        self.current_frame = None
        self._decoded_frame_index = 0
        self._decoded_source = None
        self._decode_stats = RollingStats(window=120)
        self._last_decode_ms = 0.0
        self._last_decode_sequential = None
        self._video_read_counters = (0, 0)
        self._backend.reset()
        self._discard_pending_previews()
        self._last_preview = None  # the frames on screen belong to the document being replaced
        if self._monitor is not None:
            self._monitor.set_images(None, None)
        self._preview_controller.set_videos(tuple(self._sources.values()))  # the worker releases the previous ones
        self._last_logged_settings = None  # next preview logs a full settings.snapshot for this sequence
        last = self.frame_count - 1
        self.current_frame_index = max(0, min(playhead, last))
        start, end = clip_range if clip_range is not None else (0, last)
        self.clip_start_frame = max(0, min(start, last))
        self.clip_end_frame = max(self.clip_start_frame, min(end, last))
        self._pending_cut_start = None
        self._history.reset(self._edit_state())  # a new document: nothing to undo
        self._saved_snapshot = self._project_snapshot()  # what is on screen is what the file (or nothing) holds
        self._update_history_actions()
        self._stabilization_epoch += 1
        self.transport.set_frame_count(self.frame_count, self.fps, self.sequence.clips[0].source.aspect_ratio)
        self._sequence_changed()
        self._set_controls_enabled(True)
        self.transport.set_value_silently(self.current_frame_index)
        self._load_frame(self.current_frame_index)
        self._focus_preview_surface()

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
        self._commit_edit("adicionar vídeo")
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
        self._commit_edit("dividir clipe")
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
        self._commit_edit("remover clipe")
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
        self._commit_edit("mover clipe")
        self.transport.set_value_silently(self.current_frame_index)
        self._load_frame(self.current_frame_index)

    def _move_active_clip(self, step: int) -> None:
        """The arrow tools under the timeline: the clip under the playhead moves one position back or forward."""
        index = self.active_clip_index
        if 0 <= index < len(self.sequence.clips) and 0 <= index + step < len(self.sequence.clips):
            self._move_clip(index, index + step)

    def _sequence_changed(self) -> None:
        """The clips changed (added, removed, moved, split): push the layout everywhere it is shown."""
        last = self.frame_count - 1
        self.clip_start_frame = max(0, min(self.clip_start_frame, last))
        self.clip_end_frame = max(self.clip_start_frame, min(self.clip_end_frame, last))
        self.current_frame_index = max(0, min(self.current_frame_index, last))
        if self._pending_cut_start is not None and self._pending_cut_start > last:
            self._pending_cut_start = None
        self.transport.set_clips(self._clip_blocks())
        if self._monitor is not None:
            self._monitor.set_frame_count(self.frame_count, self.fps)
        self._filmstrip.set_videos(self.sequence.source_paths)
        self._load_waveforms()
        self._audio_player.keep_only(self.sequence.source_paths)
        self._pcm_requested = {key for key in self._pcm_requested if key[0] in self.sequence.source_paths}
        self._load_preview_audio()
        self._sync_active_clip(force=True)
        self._update_timeline_markers()
        self._update_export_summary()
        self._update_sequence_info()
        self.export_action.setEnabled(not self.sequence.is_empty and self._export_controller is None)
        self.remove_clip_action.setEnabled(len(self.sequence) > 1 and self._export_controller is None)
        self.split_clip_action.setEnabled(not self.sequence.is_empty and self._export_controller is None)
        self.save_project_action.setEnabled(not self.sequence.is_empty and self._export_controller is None)
        self.save_project_as_action.setEnabled(not self.sequence.is_empty and self._export_controller is None)

    def _clip_blocks(self) -> list[ClipBlock]:
        offsets = self.sequence.clip_offsets()
        return [
            ClipBlock(
                offset,
                clip.timeline_count,
                clip.name,
                clip.source.path,
                clip.range.start,
                missing=clip.source.path in self._missing_sources,
            )
            for offset, clip in zip(offsets, self.sequence.clips)
        ]

    def _update_sequence_info(self) -> None:
        clip = self.active_clip
        if clip is None:
            self.status.set_video_info("")
            self._update_window_title()
            return
        width, height = clip.source.width, clip.source.height
        if len(self.sequence) == 1:
            info = f"{clip.name} · {width}x{height} · {self.fps:.2f} fps · {self.frame_count} frames"
        else:
            info = (
                f"{len(self.sequence)} clipes · {self.frame_count} frames · {self.fps:.2f} fps · "
                f"clipe ativo {self.active_clip_index + 1}: {clip.name} ({width}x{height})"
            )
        self.status.set_video_info(info)
        self._update_window_title()

    def _load_waveforms(self) -> None:
        """One ffmpeg decode per file of the sequence (files already requested are not decoded again)."""
        ffmpeg_path = find_ffmpeg()
        for path in self.sequence.source_paths:
            if path in self._waveforms_requested:
                continue
            self._waveforms_requested.add(path)
            if path in self._missing_sources:
                self.transport.set_audio_state(path, "missing")
            elif ffmpeg_path is None:
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
        self._update_sequence_info()
        log_event(LOG, "ui.clip_active", clip=index, name=self.sequence.clips[index].name)

    def _store_active_clip_settings(self) -> None:
        """Snapshot the widgets into the active clip (called on every render and before switching clips).

        Each store is an undo step labelled by the section that changed; the stores of a slider drag
        (one per render, same fields) coalesce into one.
        """
        clip = self.active_clip
        if clip is None:
            return
        settings = self._current_render_settings()
        if settings != clip.settings:
            changes = diff_settings(clip.settings, settings)
            self.sequence = self.sequence.with_clip_settings(self.active_clip_index, settings)
            key = ("settings", self.active_clip_index, tuple(sorted(changes)))
            self._commit_edit(settings_edit_label(changes), key)

    def _update_active_clip(self, update: Callable[[Clip], Clip], label: str, key: Hashable | None = None) -> None:
        """Change the active clip (a reference capture): the widgets are stored first so it is one undo step."""
        clip = self.active_clip
        if clip is None:
            return
        updated = update(clip.with_settings(self._current_render_settings()))
        self.sequence = self.sequence.replace_clip(self.active_clip_index, updated)
        self._commit_edit(label, key)

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
        self.stabilization_tab.apply_capture(settings.capture)
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
        """Drop reference frames no clip uses — in the sequence shown or in any state undo/redo can return to."""
        sequences = [self.sequence, *(state.sequence for state in self._history.states())]
        wanted = {
            (clip.source.path, clip.stabilization_reference_frame)
            for sequence in sequences
            for clip in sequence.clips
            if clip.stabilization_reference_frame is not None
        }
        for key in [key for key in self._reference_frames if key not in wanted]:
            del self._reference_frames[key]

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
