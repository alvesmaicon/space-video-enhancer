"""Frame navigation and playback: the wall clock is the master, the sound follows the same runs."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Slot

from space_video_enhancer.audio import PREVIEW_SAMPLE_RATE
from space_video_enhancer.export.ffmpeg import find_ffmpeg
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.telemetry.stats import RollingStats
from space_video_enhancer.timeline import next_kept_frame
from space_video_enhancer.ui.audio_preview import (
    MAX_PREVIEW_AUDIO_SECONDS,
    AudioPiece,
)
from space_video_enhancer.ui.sections.state import WindowState

LOG = get_logger("ui")

PLAYBACK_TICK_MAX_HZ = 30.0
"""Playback consults the clock at most this often; the frame shown is whatever the elapsed time calls for."""


class PlaybackSection(WindowState):
    """Frame navigation and playback: the wall clock is the master, the sound follows the same runs."""

    def _load_frame(self, frame_index: int) -> None:
        """Ask the render thread for timeline frame ``frame_index``; decoding happens there, never on the GUI."""
        if self.sequence.is_empty:
            return
        playing = self.playback_timer.isActive()
        self.current_frame_index = frame_index
        self.transport.set_position(frame_index)
        if self._monitor is not None:
            self._monitor.set_position(frame_index)
        self._sync_active_clip()
        if playing and not self._playback_seeking:
            self._start_playback_clock(frame_index)  # the user seeked while playing: video and sound restart here
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

        self._set_playing_indicators(True)
        self._playback_started_at = time.monotonic()
        self._playback_frames_shown = 0
        self._playback_intervals = RollingStats(window=60)
        self._playback_stabilization = Counter()
        self._playback_last_frame_at = None
        self._playback_video_counters = self._video_read_counters
        self._start_playback_clock(current_frame)
        log_event(
            LOG,
            "playback.start",
            frame=current_frame,
            end_frame=self.playback_end_frame,
            interval_ms=self._playback_interval_ms(),
            speed=self._playback_speed(),
            audio=self._audio_player.playing,
        )
        self.playback_timer.start(self._playback_interval_ms())

    def _playback_speed(self) -> float:
        return float(self.preview_speed_combo.currentData() or 1.0)

    def _start_playback_clock(self, frame: int) -> None:
        """Anchor the wall clock at ``frame``: from now on the playhead is a function of the elapsed time.

        The kept frames up to the end of the range are laid out once, so cuts and clip boundaries
        are crossed without a lookup per tick, and the sound is started over the same runs.
        """
        runs = self.sequence.playback_runs(frame, self.playback_end_frame)
        spans = [np.arange(run.span.start, run.span.end + 1, dtype=np.int64) for run in runs]
        self._playback_kept = np.concatenate(spans) if spans else np.zeros(0, dtype=np.int64)
        self._playback_anchor_time = time.monotonic()
        self._playback_anchor_index = 0
        speed = self._playback_speed()
        # A retimed or muted clip plays silent in the preview (its span in timeline seconds); the export
        # stretches the audio with atempo, but doing that live per clip is not worth the desync risk here.
        pieces = [
            AudioPiece(None, 0.0, run.span.count / self.sequence.fps)
            if run.clip.settings.capture.is_retimed or run.clip.settings.capture.mute
            else AudioPiece(run.clip.source.path, *run.seconds)
            for run in runs
        ]
        started = self._audio_player.start(pieces, speed)
        if not started and not self._audio_player.available and not self._audio_warned:
            self._audio_warned = True
            self.status.show_message("Sem dispositivo de áudio: a reprodução segue sem som.")

    def _set_playback_frame(self, frame: int) -> None:
        self._playback_seeking = True
        try:
            self.transport.set_value(frame)
        finally:
            self._playback_seeking = False

    def _stop_playback(self) -> None:
        was_active = self.playback_timer.isActive()
        if was_active:
            self.playback_timer.stop()
        self._playback_audio_samples = self._audio_player.samples_written
        self._audio_player.stop()
        self._set_playing_indicators(False)
        if was_active:
            self._log_playback_stop()
            self._schedule_filter_thumbnails_refresh()
            self._update_hud()

    def _set_playing_indicators(self, playing: bool) -> None:
        self.transport.set_playing(playing)
        if self._monitor is not None:
            self._monitor.set_playing(playing)

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
            "speed": self._playback_speed(),
            "audio_seconds": round(self._playback_audio_samples / PREVIEW_SAMPLE_RATE, 2),
        }
        if self._playback_stabilization:
            fields["stabilization"] = dict(self._playback_stabilization)
        self._counters["playback_frames"] += self._playback_frames_shown
        log_event(LOG, "playback.stop", **fields)
        self._playback_started_at = None

    def _advance_playback_frame(self) -> None:
        """Show the kept frame the elapsed time calls for (frames the render cannot keep up with are skipped)."""
        elapsed = time.monotonic() - self._playback_anchor_time
        index = self._playback_anchor_index + int(elapsed * self.fps * self._playback_speed())
        if index >= len(self._playback_kept):
            if len(self._playback_kept) and self.transport.value() != int(self._playback_kept[-1]):
                self._set_playback_frame(int(self._playback_kept[-1]))
            self._stop_playback()
            return
        frame = int(self._playback_kept[index])
        if frame != self.transport.value():
            self._set_playback_frame(frame)

    def _playback_interval_ms(self) -> int:
        rate = max(1.0, self.fps * self._playback_speed())
        return max(int(1000 / PLAYBACK_TICK_MAX_HZ), int(1000 / rate))

    def _load_preview_audio(self) -> None:
        """Decode (once per file and speed) the PCM the player needs for the current speed."""
        if self.sequence.is_empty or not self._audio_player.available:
            return
        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path is None:
            return
        speed = self._playback_speed()
        for clip in self.sequence.clips:
            key = (clip.source.path, speed)
            if key in self._pcm_requested:
                continue
            self._pcm_requested.add(key)
            if clip.source.path in self._missing_sources:
                self._audio_player.set_pcm(clip.source.path, speed, None)  # plays as silence
                continue
            seconds = clip.source.frame_count / self.fps
            if seconds > MAX_PREVIEW_AUDIO_SECONDS:
                self._audio_player.set_pcm(clip.source.path, speed, None)
                log_event(LOG, "audio.preview_skipped", name=clip.name, seconds=round(seconds, 1))
                self.status.show_message(f"{clip.name} é longo demais para o som do preview; a reprodução segue muda.")
                continue
            self._pcm_loader.load(clip.source.path, ffmpeg_path, speed)

    @Slot(object, float, object)
    def _on_pcm_ready(self, path: Path, tempo: float, samples: np.ndarray | None) -> None:
        if path in self.sequence.source_paths:  # the feeder reads the shared PCM table: a running playback picks it up
            self._audio_player.set_pcm(path, tempo, samples)

    def _handle_playback_speed_changed(self, _index: int) -> None:
        log_event(LOG, "ui.playback_speed", speed=self._playback_speed())
        self._load_preview_audio()
        if self.playback_timer.isActive():
            self._start_playback_clock(self.current_frame_index)
            self.playback_timer.start(self._playback_interval_ms())

    def _set_sound_enabled(self, enabled: bool) -> None:
        self._audio_player.set_muted(not enabled)
        log_event(LOG, "ui.sound", enabled=enabled)
        if enabled and self.playback_timer.isActive():
            self._start_playback_clock(self.current_frame_index)

    def _playback_fps(self) -> float | None:
        if len(self._playback_intervals) < 2 or self._playback_intervals.mean <= 0:
            return None
        return 1000.0 / self._playback_intervals.mean
