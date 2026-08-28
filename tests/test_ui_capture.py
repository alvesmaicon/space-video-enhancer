"""Capture artefacts in the UI (deinterlace, rolling shutter) and the hand tool that pans a zoomed preview."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from space_video_enhancer.project import load_project
from space_video_enhancer.settings import DeinterlaceMode
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.qt_helpers import pump, wait_for_preview


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
    trace = window._last_preview.trace
    assert trace["deinterlace.mode"] == "field" and "capture" in trace["stages"]
    assert window.undo_action.text().endswith("captura"), window.undo_action.text()

    tab.rolling_shutter_slider.setValue(70)
    wait_for_preview(app, window)
    settings = window.sequence.clips[0].settings.stabilization
    assert settings.rolling_shutter == pytest.approx(0.7) and settings.requires_sequential_processing
    assert "exportação sequencial" not in tab.info_label.text(), "the stabilization mode itself is still off"
    assert "rolling shutter" in tab.capture_info_label.text()
    window.transport.set_value(1)  # a second consecutive frame gives the unskew its motion estimate
    wait_for_preview(app, window)
    assert window._last_preview.trace.get("rolling_shutter.result") in ("applied", "negligible", "no_motion")

    project_path = tmp_path / "capture.svep"
    assert window._write_project(project_path)
    loaded = load_project(project_path).sequence.clips[0].settings
    assert loaded.capture.deinterlace is DeinterlaceMode.FIELD and loaded.stabilization.rolling_shutter == 0.7

    select_combo_data(tab.deinterlace_combo, DeinterlaceMode.OFF.value)
    tab.rolling_shutter_slider.setValue(0)
    wait_for_preview(app, window)
    _open(app, window, ui_video)
    assert window._open_project(project_path)
    wait_for_preview(app, window)
    assert tab.deinterlace_combo.currentData() == "field" and tab.rolling_shutter_slider.value() == 70


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


def _key_event(key: Qt.Key) -> QKeyEvent:
    return QKeyEvent(QKeyEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
