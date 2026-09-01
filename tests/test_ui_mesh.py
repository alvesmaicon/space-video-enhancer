"""The MESH stabilization mode in the window: background analysis, preview, controls, persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication

from space_video_enhancer.project import load_project
from space_video_enhancer.settings import StabilizationMode
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.qt_helpers import wait_for_preview, wait_until


def test_mesh_mode_analyses_the_clip_and_stabilizes_the_preview(
    app: QApplication, window: MainWindow, ui_video: Path, tmp_path: Path
) -> None:
    tab = window.stabilization_tab
    window._open_video(ui_video)
    wait_for_preview(app, window)

    select_combo_data(tab.mode_combo, StabilizationMode.MESH.value)
    wait_for_preview(app, window)
    assert not tab.wobble_checkbox.isEnabled(), "the mesh absorbs the wobble: the checkbox is moot"
    assert not tab.zoom_out_checkbox.isEnabled(), "the mesh field is clamped to the crop margin"
    assert tab.intensity_slider.isEnabled() and tab.smoothing_slider.isEnabled()
    assert window.sequence.clips[0].settings.stabilization.mode is StabilizationMode.MESH

    wait_until(app, lambda: not window._mesh_pending and bool(window._mesh_tracks), what="mesh analysis")
    wait_for_preview(app, window)
    track = window._mesh_tracks[0]
    assert track.source == ui_video and track.covers(0, 11)
    trace = window._last_preview.trace
    assert trace.get("stabilization.mode") == "mesh"
    assert trace.get("stabilization.result") == "applied"
    assert "Malha pronta" in tab.info_label.text(), tab.info_label.text()

    project_path = tmp_path / "mesh.svep"
    assert window._write_project(project_path)
    loaded = load_project(project_path).sequence.clips[0].settings
    assert loaded.stabilization.mode is StabilizationMode.MESH

    select_combo_data(tab.mode_combo, StabilizationMode.OFF.value)
    wait_for_preview(app, window)
    assert window._mesh_track_for(window.sequence.clips[0]) is track, "the track survives leaving the mode"


def test_object_mode_uses_the_mesh_analysis_and_the_anchored_fields(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    tab = window.stabilization_tab
    window._open_video(ui_video)
    wait_for_preview(app, window)

    select_combo_data(tab.mode_combo, StabilizationMode.OBJECT.value)
    wait_for_preview(app, window)
    assert not tab.wobble_checkbox.isEnabled(), "the mesh under the object mode absorbs the wobble"
    assert window.focus_regions_norm, "entering the mode seeds the first region"

    wait_until(app, lambda: not window._mesh_pending and bool(window._mesh_tracks), what="mesh analysis")
    wait_for_preview(app, window)
    trace = window._last_preview.trace
    assert trace.get("stabilization.mode") == "object"
    assert trace.get("stabilization.mesh") in ("field", "handoff"), trace.get("stabilization.mesh")
    assert "Malha" in tab.info_label.text(), tab.info_label.text()
    assert "região rastreada" in tab.info_label.text(), tab.info_label.text()

    corrections = window._mesh_corrections_for(window.sequence.clips[0], window._current_stabilization())
    window._handle_preview_focus_region_changed(0, QRectF(0.55, 0.55, 0.3, 0.3))
    wait_for_preview(app, window)
    moved = window._mesh_corrections_for(window.sequence.clips[0], window._current_stabilization())
    assert moved is not corrections, "dragging a region rebuilds the fields from the cached track"


def test_mesh_mode_stabilizes_with_the_live_fallback_while_unanalysed(
    app: QApplication, window: MainWindow, ui_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MainWindow, "_mesh_track_for", lambda self, clip: None)  # the analysis never lands
    tab = window.stabilization_tab
    window._open_video(ui_video)
    wait_for_preview(app, window)

    select_combo_data(tab.mode_combo, StabilizationMode.MESH.value)
    assert tab.mode_combo.currentText() == "Automática"
    wait_for_preview(app, window)
    window.transport.set_value(window.transport.value() + 1)  # a second consecutive frame: motion is estimated
    wait_for_preview(app, window)

    trace = window._last_preview.trace
    assert trace.get("stabilization.mode") == "mesh"
    assert trace.get("stabilization.mesh") == "fallback", "the live path covers the wait for the analysis"
    assert trace.get("stabilization.result") in ("applied", "negligible"), trace.get("stabilization.result")
    assert "aguardando a análise" in tab.info_label.text(), tab.info_label.text()
