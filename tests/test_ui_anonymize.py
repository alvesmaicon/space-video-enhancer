"""The anonymise controls (in the Filtros tab): they reach the clip and the render, and survive a project round trip."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from space_video_enhancer.project import load_project
from space_video_enhancer.settings import AnonymizeMode
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.qt_helpers import wait_for_preview


def test_anonymize_settings_reach_the_clip_the_render_and_the_project(
    app: QApplication, window: MainWindow, ui_video: Path, tmp_path: Path
) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    tab = window.filters_tab
    assert tab.anonymize_strength_slider.isEnabled()  # controls are enabled once a video is open

    tab.anonymize_checkbox.setChecked(True)
    tab.anonymize_strength_slider.setValue(80)
    select_combo_data(tab.anonymize_mode_combo, AnonymizeMode.BLUR.value)
    wait_for_preview(app, window)

    settings = window.sequence.clips[0].settings.anonymize
    assert settings.enabled and settings.strength == 80 and settings.detect_faces
    assert settings.mode is AnonymizeMode.BLUR
    trace = window._last_preview.trace
    assert "anonymize" in trace["stages"] and trace["anonymize.mode"] == "blur"

    path = tmp_path / "anon.svep"
    assert window._write_project(path)
    loaded = load_project(path).sequence.clips[0].settings.anonymize
    assert loaded.enabled and loaded.strength == 80 and loaded.mode is AnonymizeMode.BLUR

    tab.anonymize_checkbox.setChecked(False)
    wait_for_preview(app, window)
    assert "anonymize" not in window._last_preview.trace["stages"], "off -> the stage does not run"
