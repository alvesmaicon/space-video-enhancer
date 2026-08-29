"""The HSL tab feeds ColorGradeSettings and owns the gray-reference overlay."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication

from space_video_enhancer.settings import HUE_BAND_NAMES, NEUTRAL_GAINS
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.qt_helpers import wait_for_preview, wait_until


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


def test_white_patch_button_turns_the_highlights_into_gains(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window.grade_tab.white_patch_button.click()
    gains = window.white_balance_gains
    assert len(gains) == 3 and all(0.5 <= gain <= 2.0 for gain in gains)
    assert window._current_render_settings().grade.white_balance_gains == gains
    assert "realce" in window.status.message().lower()


def test_hsl_sliders_show_the_neighbour_colours_of_the_selected_band(app: QApplication, window: MainWindow) -> None:
    tab = window.grade_tab
    tab.select_band(3)  # green
    assert "Amarelo" in tab.hue_low_label.text() and "Ciano" in tab.hue_high_label.text()
    assert "qlineargradient" in tab.hue_slider.slider.styleSheet()
    tab.select_band(5)  # blue: sits between cyan and purple
    assert "Ciano" in tab.hue_low_label.text() and "Roxo" in tab.hue_high_label.text()


def test_gray_reference_overlay_is_editable_only_on_the_color_tab(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window.settings_tabs.setCurrentWidget(window.color_page)
    app.processEvents()
    assert window.preview.white_balance.interactive
    window.settings_tabs.setCurrentWidget(window.filters_tab)
    app.processEvents()
    assert not window.preview.white_balance.enabled


def test_exposure_lock_captures_the_current_frame_and_feeds_the_settings(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    tab = window.grade_tab
    assert not window._current_render_settings().exposure.is_active and "Desligado" in tab.exposure_status_label.text()
    window._capture_exposure_reference()
    exposure = window._current_render_settings().exposure
    assert exposure.enabled and exposure.is_active and exposure.region is not None and exposure.color
    assert exposure.reference is not None and all(0 <= value <= 255 for value in exposure.reference)
    assert tab.exposure_checkbox.isChecked() and f"frame {window._decoded_frame_index + 1}" in tab.exposure_label.text()
    wait_for_preview(app, window)
    wait_until(app, lambda: "ganhos" in tab.exposure_status_label.text(), what="exposure gains label")
    assert "+0.00 EV" in tab.exposure_status_label.text(), "the reference frame itself needs no correction"
    region_reference = tab.exposure_reference()
    select_combo_data(tab.exposure_scope_combo, False)  # whole frame: re-metered on the same frame
    exposure = window._current_render_settings().exposure
    assert exposure.region is None and exposure.reference is not None and exposure.reference != region_reference
    tab.exposure_limit_slider.setValue(2)
    select_combo_data(tab.exposure_mode_combo, False)
    tab.exposure_strength_slider.setValue(40)
    exposure = window._current_render_settings().exposure
    assert exposure.max_ev == pytest.approx(0.5) and not exposure.color and exposure.strength == 40
    tab.exposure_strength_slider.setValue(0)
    assert not window._current_render_settings().exposure.is_active
    window._clear_exposure_reference()
    assert tab.exposure_reference() is None and not window._current_render_settings().exposure.is_active
    assert "Sem referência" in tab.exposure_label.text() and not tab.exposure_clear_button.isEnabled()


def test_moving_the_region_on_another_frame_forgets_the_exposure_reference(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window._capture_exposure_reference()
    reference_frame = window._decoded_frame_index
    window.transport.set_value(2)
    wait_for_preview(app, window)
    assert window._decoded_frame_index == 2 != reference_frame
    window._handle_preview_white_balance_changed(QRectF(0.1, 0.1, 0.25, 0.25))
    assert window.grade_tab.exposure_reference() is None and "capture" in window.status.message()
    window._capture_exposure_reference()
    first = window.grade_tab.exposure_reference()
    window._handle_preview_white_balance_changed(QRectF(0.5, 0.5, 0.3, 0.3))  # same frame: re-metered, not forgotten
    assert window.grade_tab.exposure_reference() is not None and window.grade_tab.exposure_reference() != first
    window._open_video(ui_video)  # a new video starts without a reference
    wait_for_preview(app, window)
    assert window.grade_tab.exposure_reference() is None
