"""Geometria > Perspectiva: mark four points on the uncorrected frame, confirm, reset; the crop follows."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from space_video_enhancer.settings import PerspectiveFit
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from space_video_enhancer.ui.widgets.compare_preview import ComparePreview
from tests.conftest import UI_VIDEO_SIZE
from tests.qt_helpers import pump, wait_for_preview


def _drag(widget: ComparePreview, start: QPointF, end: QPointF) -> None:
    def event(kind: QMouseEvent.Type, position: QPointF, buttons: Qt.MouseButton) -> QMouseEvent:
        return QMouseEvent(kind, position, position, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)

    widget.mousePressEvent(event(QMouseEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton))
    widget.mouseMoveEvent(event(QMouseEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton))
    widget.mouseReleaseEvent(event(QMouseEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton))


def _on_image(preview: ComparePreview, x: float, y: float) -> QPointF:
    rect = preview._image_rect
    return QPointF(rect.left() + x * rect.width(), rect.top() + y * rect.height())


def _start_marking(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window.settings_tabs.setCurrentWidget(window.geometry_tab)
    pump(app)
    window.geometry_tab.perspective_mark_button.click()
    wait_for_preview(app, window)


def test_marking_shows_the_uncorrected_frame_with_four_draggable_points(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    tab = window.geometry_tab
    window.settings_tabs.setCurrentWidget(tab)
    select_combo_data(tab.crop_mode_combo, 16 / 9)
    pump(app)
    assert not tab.perspective_confirm_button.isEnabled() and not tab.perspective_reset_button.isEnabled()
    assert window.preview.crop.enabled

    tab.perspective_mark_button.click()
    wait_for_preview(app, window)
    preview = window.preview
    assert tab.perspective_mark_button.isChecked() and tab.perspective_confirm_button.isEnabled()
    assert preview.perspective.interactive and len(preview.perspective.points) == 4
    assert not preview.crop.enabled, "the crop refers to the corrected frame: hidden while marking"
    assert "perspective.result" not in window._last_preview.trace, "points go on the frame as filmed"

    handle = preview.perspective.display_points(preview._image_rect)[0]
    _drag(preview, handle, _on_image(preview, 0.05, 0.1))
    draft = window._perspective_draft
    assert draft is not None
    assert (draft[0].x, draft[0].y) == pytest.approx((0.05, 0.1), abs=0.01)
    assert not window.active_clip.settings.perspective.enabled, "points not confirmed stay out of the clip"


def test_confirm_applies_the_correction_and_reset_removes_it(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _start_marking(app, window, ui_video)
    tab, preview = window.geometry_tab, window.preview
    _drag(preview, preview.perspective.display_points(preview._image_rect)[1], _on_image(preview, 0.9, 0.05))

    tab.perspective_confirm_button.click()
    wait_for_preview(app, window)
    perspective = window.active_clip.settings.perspective
    assert perspective.enabled
    assert (perspective.points[1].x, perspective.points[1].y) == pytest.approx((0.9, 0.05), abs=0.01)
    assert window._history.undo_label == "correção de perspectiva"
    assert window._last_preview.trace["perspective.result"] == "applied"
    assert not preview.perspective.enabled and not tab.perspective_mark_button.isChecked()
    assert tab.perspective_reset_button.isEnabled()
    assert tab.perspective_info_label.text().startswith("Perspectiva corrigida")

    tab.perspective_mark_button.click()  # marking again starts from the confirmed points
    pump(app)
    assert window._perspective_draft == perspective.points
    tab.perspective_mark_button.click()  # ... and pressing it again leaves without touching them
    wait_for_preview(app, window)
    assert window._perspective_draft is None and window.active_clip.settings.perspective == perspective

    tab.perspective_reset_button.click()
    wait_for_preview(app, window)
    assert not window.active_clip.settings.perspective.enabled
    assert "perspective.result" not in window._last_preview.trace
    assert not tab.perspective_reset_button.isEnabled()
    assert window._history.undo_label == "redefinir perspectiva", "its own step, not merged into the confirm"

    window._undo_edit()
    wait_for_preview(app, window)
    assert window.active_clip.settings.perspective == perspective, "undo brings the correction back"


def test_leaving_the_tab_cancels_the_marking(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    _start_marking(app, window, ui_video)
    window.settings_tabs.setCurrentWidget(window.color_page)
    wait_for_preview(app, window)
    assert window._perspective_draft is None and not window.preview.perspective.enabled
    assert not window.geometry_tab.perspective_mark_button.isChecked()


def test_invalid_points_cannot_be_confirmed(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    _start_marking(app, window, ui_video)
    preview, tab = window.preview, window.geometry_tab
    points = preview.perspective.display_points(preview._image_rect)
    _drag(preview, points[0], _on_image(preview, 0.75, 0.75))  # pushed inside: a dart, not a plane
    assert not preview.perspective.valid and not tab.perspective_confirm_button.isEnabled()
    assert tab.perspective_info_label.text().startswith("Pontos inválidos")


def test_stretch_turns_the_plane_into_a_9_16_output_and_the_crop_follows(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    _start_marking(app, window, ui_video)
    tab = window.geometry_tab
    tab.perspective_confirm_button.click()
    select_combo_data(tab.crop_mode_combo, 9 / 16)
    select_combo_data(tab.perspective_fit_combo, PerspectiveFit.STRETCH_9_16.value)
    wait_for_preview(app, window)

    settings = window.active_clip.settings
    width, height = UI_VIDEO_SIZE
    assert settings.perspective.fit is PerspectiveFit.STRETCH_9_16
    assert settings.perspective.output_size(width, height) == (180, 320), "the same pixel count, 9:16"
    after = window._last_preview.after
    assert after.width() * 16 == pytest.approx(after.height() * 9, rel=0.02), "the preview shows the stretched frame"
    crop_width, crop_height = settings.output_size(width, height)
    assert crop_width / crop_height == pytest.approx(9 / 16, rel=0.05), "a 9:16 crop of the 9:16 frame"
    assert "esticado para 9:16" in tab.perspective_info_label.text()
