"""Undo/redo in the main window, and the button that re-takes the camera-lock reference on the current frame."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication

from space_video_enhancer.settings import StabilizationMode
from space_video_enhancer.ui.main_window import MainWindow, settings_edit_label
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.conftest import UI_VIDEO_B_FRAMES, UI_VIDEO_FRAMES
from tests.qt_helpers import wait_for_preview, wait_until


def _open(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)


def _go_to(app: QApplication, window: MainWindow, frame: int) -> None:
    window.transport.set_value(frame)
    wait_for_preview(app, window)


def _undo(app: QApplication, window: MainWindow) -> None:
    window._undo_edit()
    wait_for_preview(app, window)


def _redo(app: QApplication, window: MainWindow) -> None:
    window._redo_edit()
    wait_for_preview(app, window)


def test_settings_edit_labels_name_the_section_that_changed() -> None:
    assert settings_edit_label({"stabilization.mode": ["off", "auto"]}) == "ajuste de estabilização"
    assert settings_edit_label({"filters.brightness": [50, 80], "filters.contrast": [50, 60]}) == "ajustes de imagem"
    assert settings_edit_label({"crop.rect.x": [0.0, 0.1], "geometry.rotation": ["none", "cw_90"]}) == "ajustes"
    assert settings_edit_label({"unknown.thing": [0, 1]}) == "ajustes"


def test_opening_a_video_starts_with_nothing_to_undo(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    assert not window.undo_action.isEnabled() and not window.redo_action.isEnabled()
    _open(app, window, ui_video)
    _go_to(app, window, 4)
    assert not window.undo_action.isEnabled(), "loading the clip into the tabs and rendering is not an edit"
    assert window.undo_action.text() == "Desfazer" and window.redo_action.text() == "Refazer"
    assert window.undo_action.shortcut().toString() in ("Ctrl+Z", "Meta+Z")


def test_slider_changes_undo_and_redo_with_the_drag_coalesced(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _open(app, window, ui_video)
    window.color_tab.brightness_slider.setValue(60)
    wait_for_preview(app, window)
    window.color_tab.brightness_slider.setValue(80)
    wait_for_preview(app, window)
    assert window.sequence.clips[0].settings.filters.brightness == 80
    assert len(window._history) == 1, "two renders of the same slider within the window are one step"
    assert window.undo_action.isEnabled() and window.undo_action.text() == "Desfazer ajustes de imagem"

    window._history.seal()
    window.color_tab.contrast_slider.setValue(70)
    wait_for_preview(app, window)
    assert len(window._history) == 2

    _undo(app, window)
    assert window.color_tab.contrast_slider.value() == 50 and window.color_tab.brightness_slider.value() == 80
    assert window.sequence.clips[0].settings.filters.contrast == 50
    assert window.redo_action.isEnabled() and window.redo_action.text() == "Refazer ajustes de imagem"
    _undo(app, window)
    assert window.color_tab.brightness_slider.value() == 50
    assert window.sequence.clips[0].settings.filters.brightness == 50
    assert not window.undo_action.isEnabled()

    _redo(app, window)
    assert window.color_tab.brightness_slider.value() == 80
    assert window.sequence.clips[0].settings.filters.brightness == 80
    _redo(app, window)
    assert window.color_tab.contrast_slider.value() == 70
    assert not window.redo_action.isEnabled()

    window.color_tab.brightness_slider.setValue(20)
    wait_for_preview(app, window)
    _undo(app, window)
    assert window.color_tab.brightness_slider.value() == 80, "a new edit after redo undoes back to the redone state"


def test_clip_edits_undo_and_redo_including_a_dropped_file(
    app: QApplication, window: MainWindow, ui_video: Path, ui_video_b: Path
) -> None:
    _open(app, window, ui_video)
    window._add_video(ui_video_b)
    wait_until(app, lambda: len(window._preview_controller.videos) == 2, what="worker adopted both decoders")
    assert window.undo_action.text() == "Desfazer adicionar vídeo"

    _go_to(app, window, 6)
    window._split_clip_at_playhead()
    wait_for_preview(app, window)
    assert len(window.sequence) == 3 and window.undo_action.text() == "Desfazer dividir clipe"
    _undo(app, window)
    assert len(window.sequence) == 2 and window.current_frame_index == 6, "the playhead stays put"
    assert [block.count for block in window.transport.timeline.clips()] == [UI_VIDEO_FRAMES, UI_VIDEO_B_FRAMES]
    assert window.redo_action.text() == "Refazer dividir clipe"
    _redo(app, window)
    assert len(window.sequence) == 3 and len(window.transport.timeline.clips()) == 3

    window._remove_clip(2)  # the only clip of the second file: its decoder goes away with it
    wait_for_preview(app, window)
    wait_until(app, lambda: len(window._preview_controller.videos) == 1, what="worker released the dropped decoder")
    assert ui_video_b not in window._sources and window.frame_count == UI_VIDEO_FRAMES
    _undo(app, window)
    assert len(window.sequence) == 3 and window.frame_count == UI_VIDEO_FRAMES + UI_VIDEO_B_FRAMES
    assert ui_video_b in window._sources, "the file was reopened for the restored clip"
    wait_until(app, lambda: len(window._preview_controller.videos) == 2, what="worker adopted the reopened decoder")
    _go_to(app, window, UI_VIDEO_FRAMES + 2)
    assert window.active_clip_index == 2 and window._decoded_source == ui_video_b

    while window.undo_action.isEnabled():
        window._undo_edit()
    wait_for_preview(app, window)
    assert len(window.sequence) == 1 and window.frame_count == UI_VIDEO_FRAMES
    assert window.redo_action.text() == "Refazer adicionar vídeo"

    _open(app, window, ui_video)
    assert not window.undo_action.isEnabled() and not window.redo_action.isEnabled(), "a new document forgets"


def test_cuts_and_export_markers_are_undone_without_moving_the_playhead(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _open(app, window, ui_video)
    window._clear_cuts()
    _go_to(app, window, 3)
    window._toggle_cut_mark()
    _go_to(app, window, 6)
    window._toggle_cut_mark()
    assert window.cuts == [(3, 6)] and window.undo_action.text() == "Desfazer adicionar corte"
    window._mark_clip_start()
    assert window.clip_start_frame == 6 and window.undo_action.text() == "Desfazer marcar início"

    _undo(app, window)
    assert window.clip_start_frame == 0 and window.cuts == [(3, 6)] and window.current_frame_index == 6
    _undo(app, window)
    assert window.cuts == [] and window.transport.timeline.cuts() == ()
    _redo(app, window)
    assert window.cuts == [(3, 6)] and window.transport.timeline.cuts() == ((3, 6),)

    window._set_clip_range_from_timeline(1, 9, False)
    window._set_clip_range_from_timeline(2, 9, False)
    window._set_clip_range_from_timeline(2, 8, True)
    assert window.undo_action.text() == "Desfazer mover marcador"
    _undo(app, window)
    assert (window.clip_start_frame, window.clip_end_frame) == (0, UI_VIDEO_FRAMES - 1), "one step for the drag"


def test_reference_button_retakes_the_lock_reference_where_the_region_is(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _open(app, window, ui_video)
    _go_to(app, window, 2)
    assert not window.stabilization_tab.reference_button.isEnabled()
    select_combo_data(window.stabilization_tab.mode_combo, StabilizationMode.CAMERA_LOCK.value)
    wait_for_preview(app, window)
    assert window.stabilization_tab.reference_button.isEnabled()
    assert window.sequence.clips[0].stabilization_reference_frame == 2
    assert len(window._history) == 1 and window.undo_action.text() == "Desfazer ajuste de estabilização"

    region = QRectF(0.1, 0.1, 0.3, 0.3)
    window._handle_preview_stabilization_changed(region)
    wait_for_preview(app, window)
    assert window.undo_action.text() == "Desfazer mover ponto focal"
    _go_to(app, window, 7)
    assert window.sequence.clips[0].stabilization_reference_frame == 2, "scrubbing does not move the reference"

    window.stabilization_tab.reference_button.click()
    wait_for_preview(app, window)
    clip = window.sequence.clips[0]
    assert clip.stabilization_reference_frame == 7 and window.stabilization_reference_frame_index == 7
    assert window.stabilization_rect_norm == region, "the focal region stayed where it was"
    assert clip.settings.stabilization.focus_rect.x == pytest.approx(0.1)
    assert "Referência: frame 8" in window.stabilization_tab.info_label.text()
    assert window.undo_action.text() == "Desfazer referência da estabilização"
    assert (ui_video, 7) in window._reference_frames and (ui_video, 2) in window._reference_frames

    _undo(app, window)
    assert window.sequence.clips[0].stabilization_reference_frame == 2
    assert window.stabilization_reference_frame is not None and window.stabilization_reference_frame_index == 2
    assert "Referência: frame 3" in window.stabilization_tab.info_label.text()
    assert window.stabilization_rect_norm == region
    _redo(app, window)
    assert window.sequence.clips[0].stabilization_reference_frame == 7


def test_exposure_reference_capture_is_one_undo_step(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    _open(app, window, ui_video)
    _go_to(app, window, 5)
    window.grade_tab.exposure_checkbox.setChecked(True)
    wait_for_preview(app, window)
    steps = len(window._history)
    window._capture_exposure_reference()
    wait_for_preview(app, window)
    clip = window.sequence.clips[0]
    assert clip.exposure_reference_frame == 5 and clip.settings.exposure.reference is not None
    assert len(window._history) == steps + 1 and window.undo_action.text() == "Desfazer referência de exposição"

    _undo(app, window)
    clip = window.sequence.clips[0]
    assert clip.exposure_reference_frame is None and clip.settings.exposure.reference is None
    assert window.grade_tab.exposure_reference() is None
