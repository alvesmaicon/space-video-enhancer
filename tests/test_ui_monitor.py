"""The detached preview window: it mirrors the preview and drives the transport, fullscreen hides its controls."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.monitor_window import MonitorView, MonitorWindow, fit_rect, next_screen
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.conftest import UI_VIDEO_FRAMES
from tests.qt_helpers import pump, wait_for_preview, wait_until


def _solid(color: str, size: tuple[int, int] = (320, 180)) -> QImage:
    image = QImage(size[0], size[1], QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    return image


def test_fit_rect_keeps_the_aspect_ratio_and_centres() -> None:
    assert fit_rect(QSize(320, 180), QRectF(0, 0, 640, 640)) == QRectF(0, 140, 640, 360)
    assert fit_rect(QSize(320, 180), QRectF(0, 0, 320, 90)) == QRectF(80, 0, 160, 90)
    assert fit_rect(QSize(0, 180), QRectF(0, 0, 320, 90)).isEmpty()


def test_next_screen_cycles_and_needs_two_screens(app: QApplication) -> None:
    screens = QGuiApplication.screens()
    assert len(screens) == 1 and next_screen(screens, screens[0]) is None
    a, b = object(), object()
    assert next_screen([a, b], a) is b and next_screen([a, b], b) is a  # type: ignore[list-item, arg-type]
    assert next_screen([a, b], None) is a  # type: ignore[list-item]


def test_monitor_view_shows_the_result_the_source_or_a_split(app: QApplication) -> None:
    view = MonitorView()
    view.resize(640, 640)
    view.show()
    pump(app, 0.05)
    image = view.grab().toImage()
    assert image.pixelColor(20, 20) == QColor("#0b0f19"), "dark background and a placeholder until images arrive"
    assert view.image_rect().isEmpty()

    view.set_images(_solid("#ff0000"), _solid("#0000ff"))
    image = view.grab().toImage()
    assert view.image_rect() == QRectF(0, 140, 640, 360)
    assert image.pixelColor(100, 320) == QColor("#0000ff") and image.pixelColor(540, 320) == QColor("#0000ff")
    assert image.pixelColor(320, 60) == QColor("#0b0f19"), "letterboxed"

    view.set_mode("before")
    assert view.grab().toImage().pixelColor(320, 320) == QColor("#ff0000")
    view.set_mode("split")
    image = view.grab().toImage()
    assert image.pixelColor(100, 400) == QColor("#ff0000") and image.pixelColor(540, 400) == QColor("#0000ff")
    with pytest.raises(ValueError):
        view.set_mode("sideways")
    view.close()


def test_monitor_window_mirrors_the_preview_and_drives_the_transport(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    assert window.monitor_button.defaultAction() is window.monitor_action and window._monitor is None
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window.transport.set_value(3)
    wait_for_preview(app, window)

    window.monitor_action.trigger()
    monitor = window._monitor
    assert monitor is not None and monitor.isVisible() and window.monitor_action.isChecked()
    assert monitor.view.has_images(), "the last rendered frame is shown right away"
    assert monitor.scrubber.maximum() == UI_VIDEO_FRAMES - 1 and monitor.scrubber.value() == 3
    assert monitor.timecode_label.text().startswith(window.transport.timecode_label.text())
    assert f"frame 4 / {UI_VIDEO_FRAMES}" in monitor.timecode_label.text()
    assert monitor.windowTitle() == f"Monitor — {ui_video.name}"
    assert monitor.play_button.isEnabled() and not monitor.screen_button.isVisible(), "one screen: nowhere to move"

    monitor.next_frame_button.click()
    wait_for_preview(app, window)
    assert window.current_frame_index == 4 and monitor.scrubber.value() == 4
    monitor.scrubber.setValue(7)  # what dragging the scrubber does
    wait_for_preview(app, window)
    assert window.current_frame_index == 7 and window.transport.value() == 7
    monitor.go_first_button.click()
    wait_for_preview(app, window)
    assert window.current_frame_index == 0
    monitor.go_last_button.click()
    wait_for_preview(app, window)
    assert window.current_frame_index == UI_VIDEO_FRAMES - 1
    monitor.prev_frame_button.click()
    wait_for_preview(app, window)
    assert window.current_frame_index == UI_VIDEO_FRAMES - 2

    window.transport.set_value(0)
    wait_for_preview(app, window)
    generation = window._applied_preview_generation
    monitor.play_button.click()
    assert window.playback_timer.isActive() and monitor.play_button.toolTip().startswith("Pausar")
    wait_until(app, lambda: window._applied_preview_generation > generation + 1, what="frames during playback")
    monitor.play_button.click()
    assert not window.playback_timer.isActive() and monitor.play_button.toolTip().startswith("Reproduzir")

    assert window.first_frame_action in monitor.actions() and window.play_action in monitor.actions()
    monitor.activateWindow()
    monitor.view.setFocus()
    pump(app, 0.05)
    window.transport.set_value(5)
    wait_for_preview(app, window)
    QTest.keyClick(monitor.view, Qt.Key.Key_Home)
    wait_for_preview(app, window)
    assert window.current_frame_index == 0, "the transport keys work with the monitor focused"

    select_combo_data(monitor.view_combo, "split")
    assert monitor.view.mode == "split"

    monitor.close()
    pump(app, 0.05)
    assert not monitor.isVisible() and not window.monitor_action.isChecked()
    window.monitor_action.trigger()
    assert window._monitor is monitor and monitor.isVisible(), "reopened, same window"
    window.monitor_action.trigger()
    assert not monitor.isVisible() and not window.monitor_action.isChecked()


def test_monitor_follows_a_new_document_and_closes_with_the_window(
    app: QApplication, ui_video: Path, ui_video_b: Path
) -> None:
    window = MainWindow()
    window.confirm_discard = False
    window.show()
    pump(app, 0.1)
    window.monitor_action.trigger()
    monitor = window._monitor
    assert monitor is not None and not monitor.play_button.isEnabled() and not monitor.view.has_images()
    assert monitor.windowTitle() == "Monitor — Space Video Enhancer"

    window._open_video(ui_video)
    wait_for_preview(app, window)
    assert monitor.play_button.isEnabled() and monitor.view.has_images()
    assert monitor.scrubber.maximum() == UI_VIDEO_FRAMES - 1 and monitor.windowTitle() == f"Monitor — {ui_video.name}"
    window._add_video(ui_video_b)
    wait_until(app, lambda: len(window._preview_controller.videos) == 2, what="worker adopted both decoders")
    assert monitor.scrubber.maximum() == window.frame_count - 1, "the scrubber spans the whole sequence"

    window.close()
    pump(app, 0.1)
    assert not monitor.isVisible()


def test_fullscreen_hides_the_controls_when_the_mouse_rests(app: QApplication) -> None:
    monitor = MonitorWindow()
    monitor.show()
    pump(app, 0.05)
    assert monitor.controls.isVisible() and not monitor.fullscreen_action.isChecked()

    monitor.fullscreen_action.trigger()
    pump(app, 0.05)
    assert monitor.isFullScreen() and monitor.fullscreen_action.text() == "Sair da tela cheia"
    assert monitor._idle_timer.isActive()
    monitor._hide_controls()
    assert monitor.controls_hidden and not monitor.controls.isVisible()
    assert monitor.view.cursor().shape() == Qt.CursorShape.BlankCursor

    monitor._on_activity()  # the mouse moved
    assert not monitor.controls_hidden and monitor.controls.isVisible() and monitor._idle_timer.isActive()

    QTest.keyClick(monitor, Qt.Key.Key_Escape)
    pump(app, 0.05)
    assert not monitor.isFullScreen() and not monitor.fullscreen_action.isChecked()
    assert monitor.controls.isVisible() and not monitor._idle_timer.isActive()
    monitor._hide_controls()
    assert monitor.controls.isVisible(), "windowed: the controls never hide"

    monitor.toggle_fullscreen()
    pump(app, 0.05)
    assert monitor.isFullScreen()
    monitor.close()
    pump(app, 0.05)
    assert not monitor.isVisible() and not monitor.fullscreen_action.isChecked()
