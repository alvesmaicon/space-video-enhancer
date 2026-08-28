"""Capture artefacts in the UI (deinterlace, rolling shutter) and the hand tool that pans a zoomed preview."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from space_video_enhancer.project import load_project
from space_video_enhancer.sequence import Clip, ClipSource
from space_video_enhancer.settings import (
    DeinterlaceMode,
    ReadoutAxis,
    RenderSettings,
    RetimingMode,
    StabilizationSettings,
)
from space_video_enhancer.timeline import FrameRange
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from space_video_enhancer.ui.wobble_analysis import WobbleAnalysisRequest, WobbleAnalysisWorker
from tests.qt_helpers import pump, wait_for_preview, wait_until


def _open(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)


def test_capture_controls_flow_into_the_clip_settings_and_the_project(
    app: QApplication, window: MainWindow, ui_video: Path, tmp_path: Path
) -> None:
    tab = window.stabilization_tab
    assert not tab.deinterlace_combo.isEnabled(), "no video yet"
    _open(app, window, ui_video)
    assert tab.deinterlace_combo.isEnabled() and tab.rolling_shutter_slider.isEnabled()
    assert tab.mode_combo.currentData() == "off", "capture fixes do not need a stabilization mode"
    assert "Entrelaçamento improvável" in tab.capture_info_label.text(), "the synthetic video is progressive"

    select_combo_data(tab.deinterlace_combo, DeinterlaceMode.FIELD.value)
    wait_for_preview(app, window)
    clip = window.sequence.clips[0]
    assert clip.settings.capture.deinterlace is DeinterlaceMode.FIELD
    select_combo_data(tab.retiming_combo, RetimingMode.BLEND.value)
    wait_for_preview(app, window)
    assert window.sequence.clips[0].settings.capture.retiming is RetimingMode.BLEND
    trace = window._last_preview.trace
    assert trace["deinterlace.mode"] == "field" and "capture" in trace["stages"]
    assert window.undo_action.text().endswith("captura"), window.undo_action.text()

    tab.rolling_shutter_slider.setValue(70)
    tab.wobble_checkbox.setChecked(True)
    select_combo_data(tab.readout_axis_combo, ReadoutAxis.COLUMNS.value)
    wait_for_preview(app, window)
    settings = window.sequence.clips[0].settings.stabilization
    assert settings.rolling_shutter == pytest.approx(0.7) and settings.requires_sequential_processing
    assert settings.rolling_shutter_auto and settings.readout_axis is ReadoutAxis.COLUMNS
    assert "exportação sequencial" not in tab.info_label.text(), "the stabilization mode itself is still off"
    assert "rolling shutter" in tab.capture_info_label.text()
    # The wobble checkbox started an analysis of the clip on its own thread; the preview follows once it lands.
    wait_until(app, lambda: not window._wobble_pending and bool(window._wobble_tracks), what="wobble analysis")
    wait_for_preview(app, window)
    track = window._wobble_tracks[0]
    assert track.source == ui_video and track.covers(0, 11) and track.count == 12
    assert "oscilação compensada" in tab.capture_info_label.text(), tab.capture_info_label.text()
    window.transport.set_value(1)  # a second consecutive frame gives the unskew its motion estimate
    wait_for_preview(app, window)
    trace = window._last_preview.trace
    assert trace.get("rolling_shutter.result") in ("applied", "negligible", "no_motion")
    assert "rolling_shutter.wobble_px" in trace, "the analysed wobble reaches the render"
    assert trace["rolling_shutter.axis"] == "columns" and trace["rolling_shutter.axis_source"] == "forced"
    assert "leitura do sensor: colunas" in tab.capture_info_label.text(), tab.capture_info_label.text()
    assert window._wobble_track_for(window.sequence.clips[0]) is track, "the same range is not analysed twice"

    project_path = tmp_path / "capture.svep"
    assert window._write_project(project_path)
    loaded = load_project(project_path).sequence.clips[0].settings
    assert loaded.capture.deinterlace is DeinterlaceMode.FIELD and loaded.stabilization.rolling_shutter == 0.7
    assert loaded.stabilization.rolling_shutter_auto and loaded.stabilization.readout_axis is ReadoutAxis.COLUMNS
    assert loaded.capture.retiming is RetimingMode.BLEND

    select_combo_data(tab.deinterlace_combo, DeinterlaceMode.OFF.value)
    tab.rolling_shutter_slider.setValue(0)
    tab.wobble_checkbox.setChecked(False)
    select_combo_data(tab.readout_axis_combo, ReadoutAxis.AUTO.value)
    select_combo_data(tab.retiming_combo, RetimingMode.NEAREST.value)
    wait_for_preview(app, window)
    _open(app, window, ui_video)
    assert window._open_project(project_path)
    wait_for_preview(app, window)
    assert tab.deinterlace_combo.currentData() == "field" and tab.rolling_shutter_slider.value() == 70
    assert tab.wobble_checkbox.isChecked() and tab.readout_axis_combo.currentData() == "columns"
    assert tab.retiming_combo.currentData() == "blend"


def _drag(widget: object, start: QPointF, end: QPointF, button: Qt.MouseButton) -> None:
    def event(kind: QMouseEvent.Type, position: QPointF, buttons: Qt.MouseButton) -> QMouseEvent:
        global_position = widget.mapToGlobal(position.toPoint()).toPointF()  # type: ignore[attr-defined]
        return QMouseEvent(kind, position, global_position, button, buttons, Qt.KeyboardModifier.NoModifier)

    widget.mousePressEvent(event(QMouseEvent.Type.MouseButtonPress, start, button))  # type: ignore[attr-defined]
    widget.mouseMoveEvent(event(QMouseEvent.Type.MouseMove, end, button))  # type: ignore[attr-defined]
    widget.mouseReleaseEvent(event(QMouseEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton))  # type: ignore[attr-defined]


def test_hand_tool_pans_the_zoomed_preview_instead_of_moving_the_split(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    assert window.pan_button.defaultAction() is window.pan_action
    bar_children = [child for child in window.preview_bar.children() if hasattr(child, "objectName")]
    assert window.pan_button in bar_children and window.pan_action in window.preview.actions()
    _open(app, window, ui_video)
    select_combo_data(window.preview_zoom_combo, 400)
    pump(app, 0.1)
    window.preview.grab()
    horizontal = window.preview_scroll.horizontalScrollBar()
    vertical = window.preview_scroll.verticalScrollBar()
    assert horizontal.maximum() > 100 and vertical.maximum() > 50, "400% zoom: there is room to pan"
    horizontal.setValue(80)
    vertical.setValue(60)

    split_before = window.preview._split_ratio
    _drag(window.preview, QPointF(200, 150), QPointF(150, 120), Qt.MouseButton.LeftButton)
    assert window.preview._split_ratio != split_before, "without the hand tool a left drag moves the divider"
    assert (horizontal.value(), vertical.value()) == (80, 60)

    window.pan_action.trigger()
    assert window.preview.pan_mode and window.preview.cursor().shape() == Qt.CursorShape.OpenHandCursor
    split_before = window.preview._split_ratio
    _drag(window.preview, QPointF(200, 150), QPointF(150, 120), Qt.MouseButton.LeftButton)
    assert (horizontal.value(), vertical.value()) == (130, 90), "dragging left/up scrolls right/down"
    assert window.preview._split_ratio == split_before
    assert window.preview.cursor().shape() == Qt.CursorShape.OpenHandCursor

    window.pan_action.trigger()
    assert not window.preview.pan_mode
    _drag(window.preview, QPointF(200, 150), QPointF(230, 170), Qt.MouseButton.MiddleButton)
    assert (horizontal.value(), vertical.value()) == (100, 70), "the middle button pans in any mode"
    assert window.preview.cursor().shape() == Qt.CursorShape.SizeHorCursor

    window.keyPressEvent(_key_event(Qt.Key.Key_H))
    assert window.pan_action.isChecked(), "H toggles the hand from anywhere the key is not used"
    window.pan_action.setChecked(False)


def test_switching_clips_does_not_restart_a_running_wobble_analysis(
    app: QApplication, window: MainWindow, ui_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The complaint: leaving a clip mid-analysis and coming back used to cancel and restart it from zero."""
    source = ClipSource(ui_video, frame_count=400, fps=30.0, width=320, height=180)
    settings = RenderSettings(stabilization=StabilizationSettings(rolling_shutter_auto=True))
    clip_a = Clip(source, FrameRange(50, 90), settings=settings)
    clip_b = Clip(source, FrameRange(250, 290), settings=settings)

    calls: list[tuple[int, int]] = []
    generation = [0]

    def record(source_path: Path, first: int, last: int, analysis_window: int) -> WobbleAnalysisRequest:
        # Record the request instead of starting the real thread (which would need the frames decoded).
        generation[0] += 1
        calls.append((first, last))
        return WobbleAnalysisRequest(generation[0], source_path, first, last, analysis_window)

    monkeypatch.setattr(window._wobble_analysis, "analyze", record)  # the real controller stays alive for teardown
    window._wobble_pending.clear()
    window._wobble_tracks.clear()

    assert window._wobble_track_for(clip_a) is None, "clip A has no track yet: it starts one"
    assert window._wobble_track_for(clip_b) is None, "switching to B queues B without touching A"
    assert window._wobble_track_for(clip_a) is None, "back on A: still pending, not restarted"
    assert window._wobble_track_for(clip_b) is None
    assert len(calls) == 2, f"each clip is analysed once, not restarted on every visit: {calls}"
    assert {call[0] for call in calls} == {0, 250 - 60}, "one analysis per clip range (2 s margin at 30 fps)"


def test_the_analysis_queue_keeps_running_work_and_takes_the_newest_next(app: QApplication) -> None:
    """The worker never drops the running analysis for a newer one; it just queues, newest first."""
    worker = WobbleAnalysisWorker()
    a = WobbleAnalysisRequest(1, Path("v.mp4"), 0, 10, 3)
    b = WobbleAnalysisRequest(2, Path("v.mp4"), 20, 30, 3)
    c = WobbleAnalysisRequest(3, Path("v.mp4"), 40, 50, 3)
    worker.post(a)
    worker.post(b)
    worker.post(c)
    assert worker._next() is c, "the clip the user just moved to is analysed next"
    assert worker._next() is b
    assert worker._next() is a
    assert worker._next() is None

    worker.post(a)
    worker.post(b)
    worker.cancel(2)  # a new document: drop everything up to generation 2
    assert worker._next() is None, "the queue is cleared on a document change"
    later = WobbleAnalysisRequest(3, Path("w.mp4"), 0, 10, 3)
    worker.post(later)
    assert worker._next() is later, "an analysis queued after the change still runs"


def _key_event(key: Qt.Key) -> QKeyEvent:
    return QKeyEvent(QKeyEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
