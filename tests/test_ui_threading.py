"""The GUI thread must not decode frames or estimate noise (UI review, iteration C)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSize
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from space_video_enhancer.pipeline import denoise as denoise_module
from space_video_enhancer.pipeline import filters as filters_module
from space_video_enhancer.pipeline.preview import PreviewRequest
from space_video_enhancer.settings import DenoiseMode, RenderSettings
from space_video_enhancer.ui.images import device_pixel_ratio
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.preview_worker import PreviewController, PreviewImages
from space_video_enhancer.ui.tabs.base import select_combo_data
from space_video_enhancer.ui.tabs.filters import FiltersTab
from space_video_enhancer.video import VideoSource
from tests.qt_helpers import wait_for_preview, wait_until


def test_frames_are_decoded_on_the_render_thread(
    app: QApplication, window: MainWindow, ui_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decode_threads: list[str] = []
    original = VideoSource.read_frame

    def spy(self: VideoSource, index: int):  # type: ignore[no-untyped-def]
        decode_threads.append(threading.current_thread().name)
        return original(self, index)

    monkeypatch.setattr(VideoSource, "read_frame", spy)
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window.transport.set_value(2)
    wait_for_preview(app, window)

    assert len(decode_threads) >= 2
    assert threading.main_thread().name not in decode_threads
    assert window._preview_controller.video is window.video, "the worker adopted the decoder"
    assert window.current_frame is not None and window._decoded_frame_index == 2
    assert window.current_frame.shape[:2] == (180, 320)
    assert window._decode_stats.count >= 2 and window._video_read_counters[1] >= 1


def test_switching_videos_releases_the_previous_one_on_the_worker(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    first = window.video
    assert first is not None
    window._open_video(ui_video)
    wait_for_preview(app, window)
    assert window.video is not first and window._preview_controller.video is window.video
    assert not first._capture.isOpened(), "the old decoder was released by the worker"


def test_denoise_estimate_is_read_from_the_render_trace(
    app: QApplication, window: MainWindow, ui_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    estimate_threads: list[str] = []
    original = denoise_module.estimate_denoise_strengths

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        estimate_threads.append(threading.current_thread().name)
        return original(*args, **kwargs)

    for module in (denoise_module, filters_module):
        if hasattr(module, "estimate_denoise_strengths"):
            monkeypatch.setattr(module, "estimate_denoise_strengths", spy)

    window._open_video(ui_video)
    wait_for_preview(app, window)
    select_combo_data(window.noise_tab.mode_combo, DenoiseMode.AUTO.value)
    wait_for_preview(app, window)
    wait_until(app, lambda: "luminância" in window.noise_tab.info_label.text(), what="denoise estimate label")
    assert estimate_threads, "the pipeline still estimates (on the worker)"
    assert threading.main_thread().name not in estimate_threads


def test_opening_the_filters_tab_renders_thumbnails_without_crashing(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    """Regression: reading the device pixel ratio for the thumbnails must not abort the process.

    ``QWidget.devicePixelRatioF()`` on the main window aborts under PySide6 6.10 (a QThread-still-running
    fatal); the section reads it from the widget's screen instead. Before the fix this switch crashed the
    interpreter the moment the Filtros tab asked for its preset thumbnails.
    """
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window.settings_tabs.setCurrentIndex(window._filters_tab_index)
    wait_until(
        app,
        lambda: (
            window._applied_thumbnail_generation == window._thumbnail_generation and window._thumbnail_generation > 0
        ),
        what="preset thumbnails",
    )
    assert any(not button.icon().isNull() for button in window.filters_tab.buttons.values()), "thumbnails were applied"


def test_reading_the_device_pixel_ratio_off_the_main_window_is_safe(app: QApplication, window: MainWindow) -> None:
    """The helper the thumbnails use must never call the aborting ``QMainWindow.devicePixelRatioF()``."""
    ratio = device_pixel_ratio(window)
    assert ratio > 0.0


def test_thumbnails_are_requested_in_device_pixels(app: QApplication) -> None:
    tab = FiltersTab()
    image = QImage(304, 172, QImage.Format.Format_RGB888)
    image.fill(0x336699)
    tab.set_thumbnails({"original": image}, device_pixel_ratio=2.0)
    pixmap = tab.buttons["original"].icon().pixmap(QSize(152, 86), 2.0)
    assert pixmap.devicePixelRatio() == pytest.approx(2.0)
    assert pixmap.size() == QSize(304, 172)


def test_preview_worker_decodes_from_the_requested_source(app: QApplication, ui_video: Path, ui_video_b: Path) -> None:
    controller = PreviewController(None)
    results: list[PreviewImages] = []
    controller.previewReady.connect(results.append)
    first, second = VideoSource(ui_video), VideoSource(ui_video_b)
    try:
        controller.set_videos((first, second))
        for generation, path in ((1, ui_video_b), (2, ui_video)):
            controller.submit_preview(
                PreviewRequest(generation, None, 2, RenderSettings(), None, None, 0, source_path=path)
            )
            wait_until(app, lambda expected=generation: len(results) >= expected, what="preview")
        assert results[0].source.shape[:2] == (180, 240) and results[0].source_path == ui_video_b
        assert results[1].source.shape[:2] == (180, 320) and results[1].source_path == ui_video
        controller.set_videos((second,))
        wait_until(app, lambda: not first._capture.isOpened(), what="dropped decoder released on the worker")
        assert second._capture.isOpened() and controller.videos == (second,)
    finally:
        controller.stop()
    assert not second._capture.isOpened()
