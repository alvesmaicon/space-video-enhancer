"""Cuts inside the clip: marking, typing, playback skipping and the export request."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from space_video_enhancer.ui.format import parse_timecode
from space_video_enhancer.ui.main_window import MainWindow
from tests.conftest import UI_VIDEO_FPS
from tests.qt_helpers import wait_for_preview


def test_parse_timecode_accepts_the_displayed_formats() -> None:
    fps = 24.0
    assert parse_timecode("2:10", fps) == 130 * 24
    assert parse_timecode("02:10.12", fps) == 130 * 24 + 12
    assert parse_timecode("1:02:03", fps) == (3600 + 123) * 24
    assert parse_timecode("45", fps) == 45 * 24
    assert parse_timecode(" 0:05 ", fps) == 120
    assert parse_timecode("2:10", 0.0) is None
    for bad in ("", "abc", "2:", "1:2:3:4", "2:10.123"):
        assert parse_timecode(bad, fps) is None, bad


def _open(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window._clear_cuts()


def test_x_marks_the_start_and_end_of_a_cut(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    _open(app, window, ui_video)
    window.transport.set_value(3)
    wait_for_preview(app, window)
    QTest.keyClick(window.preview, Qt.Key.Key_X)
    assert window._pending_cut_start == 3 and "Terminar" in window.export_tab.cut_toggle_button.text()
    window.transport.set_value(6)
    wait_for_preview(app, window)
    QTest.keyClick(window.preview, Qt.Key.Key_X)
    assert window.cuts == [(3, 6)] and window._pending_cut_start is None
    assert window.export_tab.cut_list.count() == 1 and "4 frames" in window.export_tab.cut_list.item(0).text()
    assert "1 trecho(s) removido(s)" in window.export_tab.range_label.text()
    assert window.transport.frame_slider._cuts == ((3, 6),)
    # Marking backwards and overlapping merges into one range.
    window.transport.set_value(8)
    wait_for_preview(app, window)
    window._toggle_cut_mark()
    window.transport.set_value(5)
    wait_for_preview(app, window)
    window._toggle_cut_mark()
    assert window.cuts == [(3, 8)]


def test_typed_cuts_use_timecodes_and_exclude_the_end_time(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _open(app, window, ui_video)
    assert UI_VIDEO_FPS == 24.0
    window._add_cut_from_text("0:00.02", "0:00.05")  # frames 2..4: "até 0:00.05" stops right before frame 5
    assert window.cuts == [(2, 4)]
    window._add_cut_from_text("bogus", "0:00.05")
    assert window.cuts == [(2, 4)] and "inválidos" in window.status.message()
    window._remove_cut(0)
    assert window.cuts == []


def test_playback_jumps_over_the_cuts(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    _open(app, window, ui_video)
    window._add_cut(2, 6)
    window.transport.set_value(1)
    wait_for_preview(app, window)
    window._toggle_playback()
    assert window.playback_timer.isActive()
    window._advance_playback_frame()
    assert window.transport.value() == 7  # 2..6 skipped
    window._stop_playback()
    # Starting inside a cut moves to the first kept frame.
    window.transport.set_value(4)
    wait_for_preview(app, window)
    window._toggle_playback()
    assert window.transport.value() == 7
    window._stop_playback()


def test_reset_range_clears_the_cuts_and_export_uses_them(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _open(app, window, ui_video)
    window._add_cut(2, 3)
    window._add_cut(8, 9)
    assert "exporta 8 frames" in window.export_tab.range_label.text()
    window._reset_clip_range()
    assert window.cuts == [] and window.export_tab.cut_list.count() == 0
