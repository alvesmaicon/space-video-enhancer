"""The HSL tab feeds ColorGradeSettings and owns the gray-reference overlay."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from space_video_enhancer.settings import HUE_BAND_NAMES, NEUTRAL_GAINS
from space_video_enhancer.ui.main_window import MainWindow
from tests.qt_helpers import wait_for_preview


def test_hsl_tab_feeds_the_render_settings(app: QApplication, window: MainWindow) -> None:
    tab = window.grade_tab
    green = HUE_BAND_NAMES.index("green")
    tab.select_band(green)
    tab.hue_slider.setValue(12)
    tab.saturation_slider.setValue(-40)
    tab.tint_slider.setValue(70)
    grade = window._current_render_settings().grade
    assert grade.bands[green].hue_shift == 12 and grade.bands[green].saturation == -40
    assert grade.bands[HUE_BAND_NAMES.index("red")].is_neutral
    assert grade.tint == 70 and grade.white_balance_gains == NEUTRAL_GAINS
    tab.select_band(0)
    assert tab.hue_slider.value() == 0  # sliders show the selected band
    tab.select_band(green)
    assert tab.hue_slider.value() == 12
    tab.reset_bands()
    tab.tint_slider.setValue(50)
    assert window._current_render_settings().grade.is_neutral


def test_neutralize_turns_the_current_frame_into_gains(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window._neutralize_white_balance(True)
    gains = window.white_balance_gains
    assert len(gains) == 3 and all(0.5 <= gain <= 2.0 for gain in gains)
    assert window._current_render_settings().grade.white_balance_gains == gains
    assert "Ganhos" in window.grade_tab.gains_label.text()
    window._reset_white_balance_gains()
    assert window.white_balance_gains == NEUTRAL_GAINS and "neutro" in window.grade_tab.gains_label.text()


def test_gray_reference_overlay_is_editable_only_on_the_hsl_tab(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window.settings_tabs.setCurrentWidget(window.grade_tab)
    app.processEvents()
    assert window.preview.white_balance.interactive
    window.settings_tabs.setCurrentWidget(window.color_tab)
    app.processEvents()
    assert not window.preview.white_balance.enabled
