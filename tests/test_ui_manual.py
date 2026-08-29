"""Manual keyframe stabilization in the main window: keyframes, the onion-skin ghost, navigation."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from space_video_enhancer.settings import StabilizationMode
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.qt_helpers import wait_for_preview, wait_until


def _keyframes(window: MainWindow) -> list[tuple[int, float]]:
    return [(k.frame, round(k.tx, 3)) for k in window.sequence.clips[0].settings.stabilization.keyframes]


def _enter_manual_at(app: QApplication, window: MainWindow, ui_video: Path, frame: int) -> None:
    """Open the video, park on ``frame`` and switch to manual mode — which anchors that frame."""
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window.transport.set_value(frame)
    wait_for_preview(app, window)
    window.settings_tabs.setCurrentWidget(window.stabilization_tab)
    select_combo_data(window.stabilization_tab.mode_combo, StabilizationMode.MANUAL.value)
    app.processEvents()
    wait_for_preview(app, window)


def test_manual_mode_enables_the_keyframe_controls_and_anchors_the_current_frame(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _enter_manual_at(app, window, ui_video, 0)
    tab = window.stabilization_tab
    assert tab.manual_x_slider.isEnabled() and tab.manual_rotation_slider.isEnabled()
    assert not tab.intensity_slider.isEnabled(), "intensity/smoothing belong to the estimated modes, not manual"
    assert window.preview.has_onion_skin(), "entering manual mode captures the current frame as the ghost"
    assert _keyframes(window) == [(0, 0.0)], "the reference frame is pinned with an identity keyframe"


def test_adjusting_a_slider_creates_a_keyframe_that_interpolates_and_undoes(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _enter_manual_at(app, window, ui_video, 0)  # anchor at frame 0
    tab = window.stabilization_tab
    before = window.frame_count
    window.transport.set_value(6)
    wait_for_preview(app, window)
    tab.manual_x_slider.slider.setValue(80)  # +8% of the width at frame 6
    wait_for_preview(app, window)

    assert _keyframes(window) == [(0, 0.0), (6, 0.08)]
    assert window.sequence.clips[0].settings.stabilization.mode is StabilizationMode.MANUAL
    assert window.frame_count == before, "a keyframe never changes the clip's length"

    window.transport.set_value(3)  # halfway between the anchor (0) and the keyframe (0.08)
    wait_for_preview(app, window)
    assert tab.manual_adjust()[0] == pytest.approx(0.04)

    assert window.undo_action.text().endswith("estabilização")
    window._undo_edit()
    assert _keyframes(window) == [(0, 0.0)], "undo drops the frame-6 keyframe, the anchor stays"


def test_remove_and_navigate_between_keyframes(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    _enter_manual_at(app, window, ui_video, 0)  # anchor at frame 0
    tab = window.stabilization_tab
    for frame, value in ((2, 40), (9, -60)):
        window.transport.set_value(frame)
        wait_for_preview(app, window)
        tab.manual_y_slider.slider.setValue(value)
        wait_for_preview(app, window)
    assert [k.frame for k in window.sequence.clips[0].settings.stabilization.keyframes] == [0, 2, 9]

    window.transport.set_value(5)
    wait_for_preview(app, window)
    window._go_to_keyframe(1)  # next keyframe after 5 -> 9
    wait_for_preview(app, window)
    assert window._decoded_frame_index == 9
    window._go_to_keyframe(-1)  # previous before 9 -> 2
    wait_for_preview(app, window)
    assert window._decoded_frame_index == 2 and tab.manual_remove_button.isEnabled()

    window._remove_manual_keyframe()
    wait_for_preview(app, window)
    assert [k.frame for k in window.sequence.clips[0].settings.stabilization.keyframes] == [0, 9]


def test_the_ghost_only_shows_on_the_stabilization_tab_in_manual_mode(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _enter_manual_at(app, window, ui_video, 0)
    assert window.preview.has_onion_skin()
    window.settings_tabs.setCurrentWidget(window.filters_tab)
    app.processEvents()
    assert not window.preview.has_onion_skin(), "the ghost is only for aligning on the stabilization tab"


def test_keyframes_show_as_markers_on_the_timeline_only_in_manual_mode(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _enter_manual_at(app, window, ui_video, 0)  # anchor keyframe at frame 0
    for frame, value in ((5, 40), (9, -30)):
        window.transport.set_value(frame)
        wait_for_preview(app, window)
        window.stabilization_tab.manual_x_slider.slider.setValue(value)
        wait_for_preview(app, window)
    # one clip at the timeline fps: source frames map to the same timeline frames
    assert window.transport.timeline._keyframes == (0, 5, 9), "a diamond marks each keyframe on the ruler"

    select_combo_data(window.stabilization_tab.mode_combo, StabilizationMode.OFF.value)
    app.processEvents()
    assert window.transport.timeline._keyframes == (), "the markers only show while aligning in manual mode"


def test_fill_button_tracks_the_clip_and_densifies_the_keyframes_as_one_undo_step(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    tab = window.stabilization_tab
    _enter_manual_at(app, window, ui_video, 0)  # anchor keyframe at frame 0
    assert not tab.manual_fill_button.isEnabled(), "one keyframe is not enough to fill a gap"
    last = window.frame_count - 1
    window.transport.set_value(last)
    wait_for_preview(app, window)
    tab.manual_x_slider.slider.setValue(60)  # a second keyframe, far from the anchor
    wait_for_preview(app, window)
    before = len(window.sequence.clips[0].settings.stabilization.keyframes)
    assert tab.manual_fill_button.isEnabled(), "two keyframes bridge a gap the fill can cover"

    tab.manual_fill_button.click()
    wait_until(app, lambda: window._motion_fill_request is None, timeout=30.0, what="motion-path fill analysis")
    app.processEvents()

    after = len(window.sequence.clips[0].settings.stabilization.keyframes)
    assert after > before, "the fill inserts keyframes across the gap"
    assert window.undo_action.text().endswith("estabilização")
    window._undo_edit()
    assert len(window.sequence.clips[0].settings.stabilization.keyframes) == before, "one undo drops the whole fill"
