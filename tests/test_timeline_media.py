"""Filmstrip thumbnails for the timeline: decoded on their own thread, requested only for missing cells."""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from space_video_enhancer.audio import AudioPeaks
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.timeline_media import FilmstripController, WaveformController
from space_video_enhancer.ui.widgets.timeline import THUMBNAIL_CACHE_SIZE, VIDEO_TRACK_HEIGHT, TimelineWidget
from space_video_enhancer.video import VideoSource
from tests.conftest import UI_VIDEO_FRAMES
from tests.qt_helpers import pump, wait_for_preview, wait_until
from tests.test_audio import tone_file


def test_filmstrip_decodes_on_its_own_thread(
    app: QApplication, ui_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decode_threads: list[str] = []
    original = VideoSource.read_frame

    def spy(self: VideoSource, index: int):  # type: ignore[no-untyped-def]
        decode_threads.append(threading.current_thread().name)
        return original(self, index)

    monkeypatch.setattr(VideoSource, "read_frame", spy)
    controller = FilmstripController()
    received: dict[int, QImage] = {}
    finished: list[tuple[int, int, float]] = []
    controller.thumbnailReady.connect(lambda _generation, frame, image: received.__setitem__(frame, image))
    controller.requestFinished.connect(lambda *args: finished.append(args))
    try:
        controller.set_video(ui_video)
        generation = controller.request((0, 3, 6, 9), 40, 2.0)
        wait_until(app, lambda: len(finished) >= 1, what="filmstrip request")
        assert finished[-1][0] == generation and finished[-1][1] == 4
        assert sorted(received) == [0, 3, 6, 9]
        image = received[3]
        assert image.height() == 40 and image.width() == round(40 * 320 / 180)
        assert image.devicePixelRatio() == pytest.approx(2.0)
        assert decode_threads and threading.main_thread().name not in decode_threads
        assert controller.video is not None and controller.video.path == ui_video
    finally:
        controller.stop()
    assert controller.video is None or not controller.video._capture.isOpened()


def test_newer_request_supersedes_the_pending_one(app: QApplication, ui_video: Path) -> None:
    controller = FilmstripController()
    finished: list[tuple[int, int, float]] = []
    controller.requestFinished.connect(lambda *args: finished.append(args))
    try:
        controller.set_video(ui_video)
        first = controller.request(tuple(range(UI_VIDEO_FRAMES)) * 20, 24, 1.0)  # 240 decodes
        second = controller.request((1, 2), 24, 1.0)
        wait_until(
            app, lambda: any(generation == second for generation, _count, _ms in finished), what="second request"
        )
        by_generation = {generation: count for generation, count, _ms in finished}
        assert by_generation[second] == 2
        assert by_generation.get(first, 0) < UI_VIDEO_FRAMES * 20, "the first request stopped early"
    finally:
        controller.stop()


def test_timeline_requests_only_the_missing_cells(app: QApplication) -> None:
    timeline = TimelineWidget()
    timeline.resize(800, timeline.sizeHint().height())
    timeline.show()
    pump(app, 0.02)
    requests: list[tuple[tuple[int, ...], int, float]] = []
    timeline.thumbnailsRequested.connect(lambda frames, height, ratio: requests.append((tuple(frames), height, ratio)))
    timeline.set_media(1200, 30.0, aspect_ratio=16 / 9)
    wait_until(app, lambda: len(requests) == 1, what="initial filmstrip request")
    frames, height, ratio = requests[0]
    width, track_height = timeline.thumbnail_size()
    assert width == round(VIDEO_TRACK_HEIGHT * 16 / 9) and height == round(track_height * ratio)
    assert frames == tuple(frame for frame, _x in timeline.thumbnail_slots())
    assert len(frames) == 800 // width + 1 and frames[0] == 0 and frames == tuple(sorted(frames))
    for frame in frames[:-1]:
        timeline.set_thumbnail(frame, QImage(width, track_height, QImage.Format.Format_RGB888))
    timeline.request_missing_thumbnails()
    assert requests[-1][0] == (frames[-1],), "only the cell without a thumbnail is asked for again"
    timeline.set_thumbnail(frames[-1], QImage(width, track_height, QImage.Format.Format_RGB888))
    timeline.request_missing_thumbnails()
    assert len(requests) == 2, "nothing missing: nothing requested"
    # Scrolling keeps the cell grid anchored to frame 0, so cached frames are reused.
    timeline.zoom_by(4.0, 0.0)
    wait_until(app, lambda: len(requests) == 3, what="request after zoom")
    assert all(frame not in timeline._thumbnails for frame in requests[-1][0])
    timeline.grab()  # paints the filmstrip with nearest-neighbour fallbacks
    timeline.close()


def test_thumbnail_cache_is_bounded_and_finds_the_nearest() -> None:
    timeline = TimelineWidget()
    timeline.set_media(10_000, 30.0)
    blank = QImage(16, 9, QImage.Format.Format_RGB888)
    for frame in range(0, THUMBNAIL_CACHE_SIZE * 2 + 2, 2):  # one more than the cache holds
        timeline.set_thumbnail(frame, blank)
    assert timeline.thumbnail_count() == THUMBNAIL_CACHE_SIZE
    assert 0 not in timeline._thumbnails and timeline._thumbnail_frames == sorted(timeline._thumbnails)
    assert timeline.nearest_thumbnail(1_000, tolerance=0) is not None
    assert timeline.nearest_thumbnail(1_001, tolerance=0.5) is None
    assert timeline.nearest_thumbnail(1_001, tolerance=1) is not None
    assert timeline.nearest_thumbnail(0, tolerance=1) is None  # evicted
    timeline.clear_thumbnails()
    assert timeline.thumbnail_count() == 0 and timeline.nearest_thumbnail(2, 5) is None


def test_main_window_fills_the_filmstrip_after_opening(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    timeline = window.transport.timeline
    expected = {frame for frame, _x in timeline.thumbnail_slots()}
    wait_until(app, lambda: expected <= set(timeline._thumbnails), what="filmstrip thumbnails")
    assert window._filmstrip.video is not window.video, "the filmstrip decodes with its own VideoSource"
    assert timeline.thumbnail_size()[0] == round(VIDEO_TRACK_HEIGHT * 320 / 180)
    window._open_video(ui_video)  # reopening clears and refills
    wait_for_preview(app, window)
    wait_until(app, lambda: expected <= set(timeline._thumbnails), what="filmstrip refill")


def _loud_between(seconds_from: float, seconds_to: float, total_seconds: float, bps: float) -> AudioPeaks:
    count = round(total_seconds * bps)
    minimums = np.zeros(count, dtype=np.float32)
    maximums = np.zeros(count, dtype=np.float32)
    loud = slice(round(seconds_from * bps), round(seconds_to * bps))
    minimums[loud] = -0.8
    maximums[loud] = 0.8
    return AudioPeaks(minimums, maximums, bps)


def test_waveform_columns_follow_the_zoom(app: QApplication) -> None:
    timeline = TimelineWidget()
    timeline.resize(800, timeline.sizeHint().height())
    timeline.show()
    pump(app, 0.02)
    timeline.set_media(1200, 30.0)  # 40 s
    assert timeline.audio_state() == "none"
    timeline.set_audio_state("loading")
    timeline.grab()  # paints the state text
    timeline.set_waveform(_loud_between(10.0, 20.0, 40.0, 120.0))
    assert timeline.audio_state() == "ready"
    columns = timeline.waveform_columns(0, 800)
    assert columns is not None
    minimums, maximums = columns
    assert minimums.shape == (800,) and maximums.shape == (800,)
    scale = timeline.scale
    assert maximums[int(scale.x_of(450))] == pytest.approx(0.8) and minimums[int(scale.x_of(450))] == pytest.approx(
        -0.8
    )
    assert maximums[int(scale.x_of(150))] == 0.0 and maximums[int(scale.x_of(900))] == 0.0
    loud_columns = int(np.count_nonzero(maximums > 0.5))
    assert loud_columns == pytest.approx(200, abs=3), "10 s of 40 s in 800 px"
    timeline.zoom_by(8.0, scale.x_of(450))  # ~5 s in view around 15 s: everything is loud
    columns = timeline.waveform_columns(0, 800)
    assert columns is not None and np.all(columns[1] > 0.5)
    timeline.grab()
    timeline.set_waveform(None)
    assert timeline.audio_state() == "none" and timeline.waveform_columns(0, 800) is None
    timeline.close()


def test_waveform_columns_past_the_audio_end_are_silent(app: QApplication) -> None:
    timeline = TimelineWidget()
    timeline.resize(800, timeline.sizeHint().height())
    timeline.set_media(1200, 30.0)  # 40 s of video ...
    timeline.set_waveform(_loud_between(0.0, 20.0, 20.0, 120.0))  # ... but only 20 s of audio
    columns = timeline.waveform_columns(0, 800)
    assert columns is not None
    assert np.all(columns[1][:395] > 0.5) and np.all(columns[1][405:] == 0.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_waveform_controller_loads_on_its_own_thread(app: QApplication, tmp_path: Path) -> None:
    controller = WaveformController()
    results: list[tuple[Path, AudioPeaks | None]] = []
    controller.peaksReady.connect(lambda path, peaks: results.append((path, peaks)))
    try:
        tone = tone_file(tmp_path, seconds=0.5)
        controller.load(tone, shutil.which("ffmpeg") or "ffmpeg", 100.0)
        wait_until(app, lambda: len(results) == 1, what="waveform peaks")
        path, peaks = results[0]
        assert path == tone and peaks is not None and peaks.bucket_count == 50
    finally:
        controller.stop()


def test_main_window_reports_the_audio_state_of_the_open_video(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    timeline = window.transport.timeline
    wait_until(app, lambda: timeline.audio_state() != "loading", what="waveform state")
    assert timeline.audio_state() == ("none" if shutil.which("ffmpeg") else "unavailable")
    assert timeline.audio_track_rect().top() > timeline.video_track_rect().bottom()
