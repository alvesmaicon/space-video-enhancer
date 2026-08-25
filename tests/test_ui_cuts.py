"""Cuts inside the clip: marking, typing, playback skipping and the export request."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from space_video_enhancer.ui.format import parse_timecode
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.widgets.timeline import RULER_HEIGHT
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
    assert window.transport.timeline.cuts() == ((3, 6),)
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


def _timeline_drag(window: MainWindow, x0: float, x1: float) -> None:
    timeline = window.transport.timeline
    y = float(timeline.video_track_rect().center().y())
    for kind, x, buttons in (
        (QEvent.Type.MouseButtonPress, x0, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseMove, (x0 + x1) / 2, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseMove, x1, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseButtonRelease, x1, Qt.MouseButton.NoButton),
    ):
        event = QMouseEvent(
            kind, QPointF(x, y), QPointF(x, y), Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier
        )
        QApplication.sendEvent(timeline, event)


def test_dragging_on_the_timeline_edits_the_cuts_and_the_export_list(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _open(app, window, ui_video)
    window._add_cut(3, 6)
    timeline = window.transport.timeline
    scale = timeline.scale
    _timeline_drag(window, scale.x_of(7), scale.x_of(9))  # right edge -> cut ends before frame 9
    assert window.cuts == [(3, 8)]
    assert "6 frames" in window.export_tab.cut_list.item(0).text()
    assert timeline.cuts() == ((3, 8),), "the normalized model came back to the timeline"
    _timeline_drag(window, scale.x_of(5), scale.x_of(6))  # body -> shifted one frame later
    assert window.cuts == [(4, 9)]
    assert timeline.selected_cut() == 0 and window.export_tab.selected_cut() == 0
    # Dragging one cut over another merges them in the model.
    window._add_cut(0, 1)
    _timeline_drag(window, scale.x_of(2), scale.x_of(4))  # right edge of (0,1) reaches the other cut
    assert window.cuts == [(0, 9)]


def test_selection_is_shared_between_list_and_timeline_and_delete_removes(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _open(app, window, ui_video)
    window._add_cut(2, 3)
    window._add_cut(8, 9)
    window.export_tab.cut_list.setCurrentRow(1)
    assert window.transport.timeline.selected_cut() == 1
    window.transport.timeline.select_cut(0)
    assert window.export_tab.selected_cut() == 0
    QTest.keyClick(window.preview, Qt.Key.Key_Delete)
    assert window.cuts == [(8, 9)] and window.export_tab.cut_list.count() == 1
    window.transport.timeline.select_cut(-1)
    QTest.keyClick(window.preview, Qt.Key.Key_Backspace)  # nothing selected anywhere: no-op
    assert window.cuts == [(8, 9)]
    window.export_tab.cut_list.setCurrentRow(0)
    window._remove_selected_cut()
    assert window.cuts == []


def test_dragging_the_clip_markers_updates_the_clip_range(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _open(app, window, ui_video)
    timeline = window.transport.timeline
    scale = timeline.scale
    y = RULER_HEIGHT / 2
    for kind, x, buttons in (
        (QEvent.Type.MouseButtonPress, scale.x_of(12), Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseMove, scale.x_of(9), Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseButtonRelease, scale.x_of(9), Qt.MouseButton.NoButton),
    ):
        QApplication.sendEvent(
            timeline,
            QMouseEvent(
                kind, QPointF(x, y), QPointF(x, y), Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier
            ),
        )
    assert (window.clip_start_frame, window.clip_end_frame) == (0, 8)
    assert "frames 1\u20139" in window.export_tab.range_label.text()
