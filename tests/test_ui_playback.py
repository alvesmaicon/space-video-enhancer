"""Playback in the main window: wall clock, speed, cuts, seeking while playing, and the sound that follows it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from space_video_enhancer.audio import PREVIEW_SAMPLE_RATE
from space_video_enhancer.ui.audio_preview import PreviewAudioPlayer
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.conftest import UI_VIDEO_B_FRAMES, UI_VIDEO_FPS, UI_VIDEO_FRAMES
from tests.qt_helpers import wait_for_preview, wait_until
from tests.test_audio_preview import _FakeSink


def _elapse(window: MainWindow, seconds: float) -> None:
    """Pretend ``seconds`` went by since the playback clock was anchored."""
    window._playback_anchor_time -= seconds


def test_playback_follows_the_wall_clock_at_the_chosen_speed(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window.transport.set_value(0)
    wait_for_preview(app, window)
    select_combo_data(window.preview_speed_combo, 0.5)
    window._toggle_playback()
    assert window.playback_timer.isActive() and window.playback_timer.interval() == int(1000 / (UI_VIDEO_FPS * 0.5))
    window._advance_playback_frame()
    assert window.transport.value() == 0, "no time went by: the playhead stays"
    _elapse(window, 0.5)
    window._advance_playback_frame()
    assert window.transport.value() == 6, "0.5 s at half speed of 24 fps"
    select_combo_data(window.preview_speed_combo, 1.0)  # re-anchors at the current frame
    assert window.playback_timer.interval() == int(1000 / UI_VIDEO_FPS)
    _elapse(window, 0.125)
    window._advance_playback_frame()
    assert window.transport.value() == 9 and window.playback_timer.isActive()
    _elapse(window, 10.0)
    window._advance_playback_frame()
    assert window.transport.value() == UI_VIDEO_FRAMES - 1 and not window.playback_timer.isActive(), (
        "ends on the last frame"
    )


def test_seeking_while_playing_reanchors_the_clock(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window._add_cut(4, 8)
    window.transport.set_value(0)
    wait_for_preview(app, window)
    window._toggle_playback()
    assert window._playback_kept.tolist() == [0, 1, 2, 3, 9, 10, 11]
    _elapse(window, 4 / UI_VIDEO_FPS + 0.001)
    window._advance_playback_frame()
    assert window.transport.value() == 9, "the cut is jumped over by the frame layout, not per tick"
    window.transport.set_value(1)  # the user clicks the timeline while it plays
    assert window.playback_timer.isActive() and window._playback_kept.tolist() == [1, 2, 3, 9, 10, 11]
    _elapse(window, 1 / UI_VIDEO_FPS + 0.001)
    window._advance_playback_frame()
    assert window.transport.value() == 2
    window._stop_playback()


def test_sound_plays_the_kept_runs_of_every_clip(
    app: QApplication, window: MainWindow, ui_video: Path, ui_video_b: Path
) -> None:
    sink = _FakeSink()
    window._audio_player = PreviewAudioPlayer(sink_factory=lambda: sink, parent=window)
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window._add_video(ui_video_b)
    wait_until(app, lambda: len(window._preview_controller.videos) == 2, what="decoders")
    assert window.transport.sound_button.isEnabled() and window.transport.sound_enabled()
    player = window._audio_player
    # Deterministic PCM instead of whatever ffmpeg decoded (the fixtures have no audio anyway).
    player.set_pcm(ui_video, 1.0, np.full(PREVIEW_SAMPLE_RATE, 1000, dtype=np.int16))
    player.set_pcm(ui_video_b, 1.0, np.full(PREVIEW_SAMPLE_RATE, -1000, dtype=np.int16))
    window._add_cut(10, 13)
    window.transport.set_value(8)
    wait_for_preview(app, window)
    window._toggle_playback()
    assert player.playing and sink.started == 1
    feeder = player._feeder
    assert feeder is not None
    expected_seconds = (2 + 6) / UI_VIDEO_FPS  # frames 8-9 of the first clip, 2-7 of the second (0-1 are cut)
    assert feeder.total_samples == pytest.approx(expected_seconds * PREVIEW_SAMPLE_RATE, abs=2)
    sink.drain(len(sink.buffer))
    for _ in range(50):
        player._feed()
        sink.drain(len(sink.buffer))
    assert player.samples_written == feeder.total_samples
    window._stop_playback()
    assert sink.stopped >= 1 and not player.playing

    window.transport.sound_button.setChecked(False)
    assert player.muted
    window._toggle_playback()
    assert window.playback_timer.isActive() and not player.playing, "muted: video plays, no sound"
    window.transport.sound_button.setChecked(True)
    assert player.playing, "unmuting while playing restarts the sound at the playhead"
    window._stop_playback()
    assert window.frame_count == UI_VIDEO_FRAMES + UI_VIDEO_B_FRAMES


def test_without_an_output_device_the_speaker_button_is_off(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    window._open_video(ui_video)  # SVE_PREVIEW_AUDIO=0 in the test-suite: no sink
    wait_for_preview(app, window)
    assert not window._audio_player.available
    assert not window.transport.sound_button.isEnabled() and not window.transport.sound_enabled()
    window._toggle_playback()
    assert window.playback_timer.isActive() and "sem som" in window.status.message()
    window._stop_playback()
