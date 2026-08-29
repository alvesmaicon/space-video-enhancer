"""Manual keyframe stabilization in the main window: keyframes, the onion-skin ghost, navigation."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from space_video_enhancer.settings import StabilizationMode
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.qt_helpers import wait_for_preview


def _enter_manual(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window.settings_tabs.setCurrentWidget(window.stabilization_tab)
    select_combo_data(window.stabilization_tab.mode_combo, StabilizationMode.MANUAL.value)
    app.processEvents()


def test_manual_mode_enables_the_keyframe_controls_and_shows_the_ghost(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _enter_manual(app, window, ui_video)
    tab = window.stabilization_tab
    assert tab.manual_x_slider.isEnabled() and tab.manual_rotation_slider.isEnabled()
    assert not tab.intensity_slider.isEnabled(), "intensity/smoothing belong to the estimated modes, not manual"
    assert window.preview.has_onion_skin(), "entering manual mode captures the current frame as the ghost"


def test_adjusting_a_slider_creates_a_keyframe_that_interpolates_and_undoes(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _enter_manual(app, window, ui_video)
    tab = window.stabilization_tab
    before = window.frame_count
    window.transport.set_value(6)
    wait_for_preview(app, window)
    tab.manual_x_slider.slider.setValue(80)  # +8% of the width
    wait_for_preview(app, window)

    keyframes = window.sequence.clips[0].settings.stabilization.keyframes
    assert [(k.frame, round(k.tx, 3)) for k in keyframes] == [(6, 0.08)]
    assert window.sequence.clips[0].settings.stabilization.mode is StabilizationMode.MANUAL
    assert window.frame_count == before, "a keyframe never changes the clip's length"

    window.transport.set_value(0)  # before the only keyframe: the value is held flat
    wait_for_preview(app, window)
    assert tab.manual_adjust()[0] == pytest.approx(0.08)

    assert window.undo_action.text().endswith("estabilização")
    window._undo_edit()
    assert window.sequence.clips[0].settings.stabilization.keyframes == ()


def test_remove_and_navigate_between_keyframes(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    _enter_manual(app, window, ui_video)
    tab = window.stabilization_tab
    for frame, value in ((2, 40), (9, -60)):
        window.transport.set_value(frame)
        wait_for_preview(app, window)
        tab.manual_y_slider.slider.setValue(value)
        wait_for_preview(app, window)
    assert [k.frame for k in window.sequence.clips[0].settings.stabilization.keyframes] == [2, 9]

    window.transport.set_value(0)
    wait_for_preview(app, window)
    window._go_to_keyframe(1)  # next keyframe from frame 0 -> 2
    wait_for_preview(app, window)
    assert window._decoded_frame_index == 2 and tab.manual_remove_button.isEnabled()

    window._remove_manual_keyframe()
    wait_for_preview(app, window)
    assert [k.frame for k in window.sequence.clips[0].settings.stabilization.keyframes] == [9]


def test_the_ghost_only_shows_on_the_stabilization_tab_in_manual_mode(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _enter_manual(app, window, ui_video)
    assert window.preview.has_onion_skin()
    window.settings_tabs.setCurrentWidget(window.filters_tab)
    app.processEvents()
    assert not window.preview.has_onion_skin(), "the ghost is only for aligning on the stabilization tab"
