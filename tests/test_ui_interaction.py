"""Keyboard, overlay, transport and formatting behaviour of the main window (UI review, iteration B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from space_video_enhancer.ui.format import format_duration, format_percent, format_signed, format_timecode

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from space_video_enhancer.settings import DenoiseMode, StabilizationMode
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.conftest import UI_VIDEO_FRAMES
from tests.qt_helpers import pump, wait_for_preview

# --- pure formatting ---------------------------------------------------------------------------


def test_timecode_uses_frames_not_milliseconds() -> None:
    assert format_timecode(0, 24.0) == "00:00.00"
    assert format_timecode(30, 24.0) == "00:01.06"
    assert format_timecode(24 * 61 + 5, 24.0) == "01:01.05"
    assert format_timecode(30 * 3600 + 2, 30.0) == "1:00:00.02"
    assert format_timecode(10, 0.0) == "00:00.00"


def test_signed_percent_and_duration_labels() -> None:
    assert format_signed(50) == "0" and format_signed(62) == "+12" and format_signed(38) == "-12"
    assert format_percent(35) == "35%"
    assert format_duration(83.4) == "1:23" and format_duration(-1) == "0:00"


# --- keyboard ----------------------------------------------------------------------------------


def open_video(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)


def test_arrow_keys_stay_with_a_focused_slider(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    open_video(app, window, ui_video)
    frame_before = window.transport.value()
    slider = window.color_tab.contrast_slider
    slider.slider.setFocus()
    pump(app)

    QTest.keyClick(slider.slider, Qt.Key.Key_Right)
    pump(app, 0.1)
    assert slider.value() == 51 and slider.value_text() == "+1"
    assert window.transport.value() == frame_before, "the arrow key must not step the frame"


def test_arrow_keys_step_frames_from_the_preview_or_the_window(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    open_video(app, window, ui_video)
    start = window.transport.value()
    assert window.preview.hasFocus(), "opening a video hands the keyboard to the preview"

    QTest.keyClick(window.preview, Qt.Key.Key_Right)
    wait_for_preview(app, window)
    assert window.transport.value() == start + 1

    window.settings_tabs.setFocus()  # somewhere neutral: the key event bubbles up to the window
    QTest.keyClick(window, Qt.Key.Key_Left)
    wait_for_preview(app, window)
    assert window.transport.value() == start

    QTest.keyClick(window, Qt.Key.Key_End)
    wait_for_preview(app, window)
    assert window.transport.value() == UI_VIDEO_FRAMES - 1
    QTest.keyClick(window, Qt.Key.Key_Home)
    wait_for_preview(app, window)
    assert window.transport.value() == 0


def test_space_clicks_a_focused_button_but_plays_from_the_preview(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    open_video(app, window, ui_video)
    clicks: list[bool] = []
    button = window.export_tab.mark_end_button
    button.clicked.connect(lambda: clicks.append(True))
    button.setFocus()
    pump(app)

    QTest.keyClick(button, Qt.Key.Key_Space)
    pump(app, 0.1)
    assert clicks == [True]
    assert not window.playback_timer.isActive(), "Space on a button must click it, not start playback"

    window.transport.set_value(0)
    wait_for_preview(app, window)
    window.preview.setFocus()
    QTest.keyClick(window.preview, Qt.Key.Key_Space)
    pump(app, 0.1)
    assert window.playback_timer.isActive()
    window._stop_playback()


def test_clip_markers_from_the_keyboard(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    open_video(app, window, ui_video)
    window.transport.set_value(3)
    wait_for_preview(app, window)
    QTest.keyClick(window.preview, Qt.Key.Key_I)
    window.transport.set_value(8)
    wait_for_preview(app, window)
    QTest.keyClick(window.preview, Qt.Key.Key_O)
    assert window._selected_frame_range() == (3, 8)
    assert "00:00.03 → 00:00.09" in window.export_tab.range_label.text()


# --- overlays follow the active tab ------------------------------------------------------------


def test_overlays_are_editable_only_on_their_tab(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    open_video(app, window, ui_video)
    tabs = window.settings_tabs
    preview = window.preview

    select_combo_data(window.geometry_tab.crop_mode_combo, 16 / 9)
    tabs.setCurrentWidget(window.geometry_tab)
    pump(app)
    assert preview.crop.enabled and preview.crop.interactive
    tabs.setCurrentWidget(window.color_tab)
    pump(app)
    assert preview.crop.enabled and not preview.crop.interactive, "crop stays visible but frozen elsewhere"

    select_combo_data(window.noise_tab.mode_combo, DenoiseMode.SAMPLE.value)
    tabs.setCurrentWidget(window.noise_tab)
    pump(app)
    assert preview.sample.enabled and preview.sample.interactive
    assert not preview.focus.enabled
    tabs.setCurrentWidget(window.color_tab)
    pump(app)
    assert not preview.sample.enabled

    select_combo_data(window.stabilization_tab.mode_combo, StabilizationMode.CAMERA_LOCK.value)
    tabs.setCurrentWidget(window.stabilization_tab)
    pump(app)
    assert preview.focus.enabled and preview.focus.interactive
    assert not preview.sample.enabled, "only one analysis region at a time"
    tabs.setCurrentWidget(window.noise_tab)
    pump(app)
    assert preview.sample.enabled and not preview.focus.enabled


# --- transport ---------------------------------------------------------------------------------


def test_transport_shows_timecode_and_jumps_to_clicks(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    open_video(app, window, ui_video)
    assert window.transport.timecode_label.text() == "00:00.06 / 00:00.12"
    assert window.transport.frame_label.text() == f"frame 7 / {UI_VIDEO_FRAMES}"

    slider = window.transport.frame_slider
    QTest.mouseClick(slider, Qt.MouseButton.LeftButton, pos=QPoint(slider.width() - 4, slider.height() // 2))
    wait_for_preview(app, window)
    assert slider.value() >= UI_VIDEO_FRAMES - 2, "clicking the groove jumps there instead of paging"
    assert window.transport.timecode_label.text().startswith(f"00:00.{slider.value():02d} /")
