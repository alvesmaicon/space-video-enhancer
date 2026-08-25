"""Offscreen layout checks for the main window.

They guard the P0 findings of the UI review: tabs must never clip or hide behind scroll
arrows, the preview must fit its viewport by default, and status/progress live in a
persistent status bar instead of inside one tab.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.conftest import make_textured_frame

VIDEO_SIZE = (320, 180)
VIDEO_FRAMES = 12


@pytest.fixture(scope="module")
def app() -> QApplication:
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


def pump(app: QApplication, seconds: float = 0.05) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)


def wait_until(app: QApplication, condition: Callable[[], bool], timeout: float = 20.0, what: str = "") -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timeout waiting for {what or 'condition'}")
        pump(app, 0.01)


def wait_for_preview(app: QApplication, window: MainWindow) -> None:
    wait_until(
        app,
        lambda: window._preview_generation > 0 and window._applied_preview_generation == window._preview_generation,
        what="preview render",
    )
    pump(app, 0.05)


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("ui") / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, VIDEO_SIZE)
    base = cv2.resize(make_textured_frame(), VIDEO_SIZE, interpolation=cv2.INTER_AREA)
    for index in range(VIDEO_FRAMES):
        writer.write(np.roll(base, index * 3, axis=1))
    writer.release()
    return path


@pytest.fixture
def window(app: QApplication) -> Iterator[MainWindow]:
    window = MainWindow()
    window.show()
    pump(app, 0.1)
    yield window
    window.close()
    pump(app, 0.05)


def assert_tabs_unclipped(app: QApplication, window: MainWindow) -> None:
    tabs = window.settings_tabs
    assert tabs.tabBar().sizeHint().width() <= tabs.width(), "tab bar needs scroll arrows"
    for index in range(tabs.count()):
        tabs.setCurrentIndex(index)
        pump(app, 0.02)
        page = tabs.widget(index)
        content_width = page.content.minimumSizeHint().width()
        assert content_width <= page.viewport().width(), f"{tabs.tabText(index)} content is clipped"


def test_every_tab_fits_at_the_default_window_size(app: QApplication, window: MainWindow) -> None:
    assert_tabs_unclipped(app, window)


def test_tabs_still_fit_in_a_small_window(app: QApplication, window: MainWindow) -> None:
    window.resize(1000, 640)
    pump(app, 0.1)
    assert window.width() == 1000, "the window refuses to shrink to 1000 px"
    assert_tabs_unclipped(app, window)


def test_empty_state_offers_to_open_a_video(window: MainWindow) -> None:
    assert window.preview.open_button.isVisible()
    assert window.open_action.isEnabled()
    assert not window.export_action.isEnabled()
    assert not window.transport.isEnabled()
    assert not window.status.progress_bar.isVisible()


def test_preview_fits_by_default_and_zooms_one_to_one(
    app: QApplication, window: MainWindow, sample_video: Path
) -> None:
    window._open_video(sample_video)
    wait_for_preview(app, window)
    window.preview.grab()  # forces a paint so the image rect is known

    viewport = window.preview_scroll.viewport()
    assert window.preview.fit_mode
    assert not window.preview.open_button.isVisible()
    assert window.preview.width() <= viewport.width() and window.preview.height() <= viewport.height()
    assert not window.preview_scroll.horizontalScrollBar().isVisible()
    assert window.preview._image_rect.right() <= window.preview.width()

    select_combo_data(window.preview_zoom_combo, 100)
    pump(app, 0.1)
    window.preview.grab()
    assert not window.preview.fit_mode
    assert window.preview.effective_zoom() == pytest.approx(1.0)
    assert round(window.preview._image_rect.width()) == VIDEO_SIZE[0]

    window._step_preview_zoom(1)
    pump(app, 0.05)
    assert window.preview_zoom_combo.currentData() == 200
    window._zoom_to_fit()
    pump(app, 0.05)
    assert window.preview.fit_mode


def test_status_bar_reports_the_open_video(app: QApplication, window: MainWindow, sample_video: Path) -> None:
    window._open_video(sample_video)
    wait_for_preview(app, window)
    assert sample_video.name in window.status.video_label.text()
    assert sample_video.name in window.windowTitle()
    assert window.status.message()
    assert window.export_action.isEnabled()
    assert window.transport.isEnabled()
    assert not window.status.render_label.isVisible()
