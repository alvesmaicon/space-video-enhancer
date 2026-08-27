"""The tools under the timeline, the preview bar, look presets and the export tab after the reorganization."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QMenu, QToolBar

from space_video_enhancer.settings import DEFAULT_FILTER_SETTINGS, DenoiseMode
from space_video_enhancer.ui import main_window as main_window_module
from space_video_enhancer.ui.icons import FILLED_GLYPHS, STROKED_GLYPHS, glyph_icon
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.conftest import UI_VIDEO_B_FRAMES, UI_VIDEO_FRAMES, UI_VIDEO_SIZE
from tests.qt_helpers import pump, wait_for_preview, wait_until


def _open(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)


def _go_to(app: QApplication, window: MainWindow, frame: int) -> None:
    window.transport.set_value(frame)
    wait_for_preview(app, window)


def _painted_pixels(image: QImage) -> int:
    return sum(1 for x in range(image.width()) for y in range(image.height()) if image.pixelColor(x, y).alpha() > 0)


def test_every_glyph_renders_something(app: QApplication) -> None:
    for kind in sorted(FILLED_GLYPHS | STROKED_GLYPHS):
        icon = glyph_icon(kind, QColor("#000000"))
        pixmap = icon.pixmap(18, 18)
        assert not pixmap.isNull() and _painted_pixels(pixmap.toImage()) > 12, kind
    with pytest.raises(ValueError):
        glyph_icon("nope", QColor("#000000"))


def test_clip_tools_need_somewhere_to_go(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    transport = window.transport
    assert not transport.isEnabled(), "no video: the whole transport is off"
    _open(app, window, ui_video)
    assert transport.mark_in_button.isEnabled() and transport.mark_out_button.isEnabled()
    assert transport.remove_range_button.isEnabled() and transport.split_button.isEnabled()
    assert not transport.delete_clip_button.isEnabled(), "the only clip cannot be removed"
    assert not transport.move_left_button.isEnabled() and not transport.move_right_button.isEnabled()


def test_timeline_tools_act_on_the_playhead_and_the_clip_under_it(
    app: QApplication, window: MainWindow, ui_video: Path, ui_video_b: Path
) -> None:
    _open(app, window, ui_video)
    window._add_video(ui_video_b)
    wait_until(app, lambda: len(window._preview_controller.videos) == 2, what="worker adopted both decoders")
    transport = window.transport

    _go_to(app, window, 3)
    transport.mark_in_button.click()
    _go_to(app, window, 15)
    transport.mark_out_button.click()
    assert (window.clip_start_frame, window.clip_end_frame) == (3, 15)

    _go_to(app, window, 5)
    transport.remove_range_button.click()
    assert window._pending_cut_start == 5 and transport.pending_cut(), "pressed while the end is not marked"
    _go_to(app, window, 7)
    transport.remove_range_button.click()
    assert window.cuts == [(5, 7)] and not transport.pending_cut()

    _go_to(app, window, 9)
    transport.split_button.click()
    wait_for_preview(app, window)
    assert len(window.sequence) == 3 and window.active_clip_index == 1
    assert transport.delete_clip_button.isEnabled()
    assert transport.move_left_button.isEnabled() and transport.move_right_button.isEnabled()

    transport.move_right_button.click()
    wait_for_preview(app, window)
    assert [clip.source.path for clip in window.sequence.clips] == [ui_video, ui_video_b, ui_video]
    assert window.active_clip_index == 2, "the playhead follows the moved clip"
    assert not transport.move_right_button.isEnabled() and transport.move_left_button.isEnabled()
    transport.move_left_button.click()
    wait_for_preview(app, window)
    assert [clip.source.path for clip in window.sequence.clips] == [ui_video, ui_video, ui_video_b]
    assert window.active_clip_index == 1

    transport.delete_clip_button.click()
    wait_for_preview(app, window)
    assert len(window.sequence) == 2 and window.frame_count == UI_VIDEO_FRAMES - 3 + UI_VIDEO_B_FRAMES
    assert window.undo_action.text() == "Desfazer remover clipe"


def test_preview_controls_sit_above_the_preview_and_there_is_no_toolbar(window: MainWindow) -> None:
    bar = window.preview_bar
    assert window.preview_zoom_combo.parentWidget() is bar and window.preview_render_mode_combo.parentWidget() is bar
    assert window.hud_button.parentWidget() is bar and window.hud_button.defaultAction() is window.hud_action
    assert window.vectors_button.defaultAction() is window.vectors_action
    assert window.preview_speed_combo is window.transport.speed_combo, "speed lives next to the play buttons"
    assert window.transport.sound_button.parentWidget() is window.transport.playback_controls

    assert window.findChildren(QToolBar) == [], "files and history live in the menus and shortcuts only"
    menus = {menu.title(): menu for menu in window.menuBar().findChildren(QMenu)}
    assert window.open_action in menus["Arquivo"].actions() and window.add_video_action in menus["Arquivo"].actions()
    assert window.undo_action in menus["Editar"].actions() and window.redo_action in menus["Editar"].actions()

    window.hud_button.click()
    assert window.hud_action.isChecked() and window._hud_visible
    window.hud_button.click()
    assert not window._hud_visible


def test_export_tab_keeps_processing_and_the_export_button(
    app: QApplication, window: MainWindow, ui_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = window.export_tab
    assert [tab.scale_combo.itemData(index) for index in range(tab.scale_combo.count())] == [0.5, 0.75, 1.0, 1.5, 2.0]
    assert tab.scale_factor() == 1.0
    assert not tab.export_button.isEnabled() and not hasattr(tab, "cut_list") and not hasattr(tab, "clip_list")

    _open(app, window, ui_video)
    assert tab.export_button.isEnabled()
    summary = tab.summary_label.text()
    assert "Saída: 12 frames" in summary and f"{UI_VIDEO_SIZE[0]}x{UI_VIDEO_SIZE[1]}" in summary
    select_combo_data(tab.scale_combo, 0.5)
    wait_for_preview(app, window)
    assert f"{UI_VIDEO_SIZE[0] // 2}x{UI_VIDEO_SIZE[1] // 2}" in tab.summary_label.text()
    assert window.sequence.clips[0].settings.filters.scale_factor == 0.5

    asked: list[bool] = []
    monkeypatch.setattr(
        main_window_module.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (asked.append(True), ("", ""))[1]
    )
    tab.export_button.click()
    pump(app, 0.05)
    assert asked == [True], "the button goes through the export action"


def test_presets_change_the_look_but_not_denoise_or_scale(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _open(app, window, ui_video)
    select_combo_data(window.noise_tab.mode_combo, DenoiseMode.MANUAL.value)
    window.noise_tab.strength_slider.setValue(40)
    select_combo_data(window.export_tab.scale_combo, 1.5)
    wait_for_preview(app, window)

    combo = window.color_tab.preset_combo
    combo.setCurrentIndex(combo.findText("Balanceado"))
    wait_for_preview(app, window)
    filters = window.sequence.clips[0].settings.filters
    assert (filters.contrast, filters.sharpen) == (54, 24)
    assert filters.denoise_mode is DenoiseMode.MANUAL and filters.denoise_strength == 40 and filters.scale_factor == 1.5
    assert combo.currentText() == "Balanceado", "still recognised as the preset although denoise differs"

    window.color_tab.contrast_slider.setValue(60)
    wait_for_preview(app, window)
    assert combo.currentText() == "Personalizado"
    window._restore_active_preset_settings()
    wait_for_preview(app, window)
    assert window.color_tab.contrast_slider.value() == 54 and window.noise_tab.strength_slider.value() == 40

    window.noise_tab.strength_slider.setValue(70)
    window._reset_slider_to_preset("denoise_strength")
    assert window.noise_tab.strength_slider.value() == DEFAULT_FILTER_SETTINGS.denoise_strength, "not the preset's"
    window.color_tab.contrast_slider.setValue(10)
    window._reset_slider_to_preset("contrast")
    assert window.color_tab.contrast_slider.value() == 54
