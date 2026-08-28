"""The four ways to compare before and after: two draggable dividers and two two-panel layouts."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, QSize, QSizeF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent
from PySide6.QtWidgets import QApplication

from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from space_video_enhancer.ui.widgets.compare_preview import PANEL_GAP, CompareMode, ComparePreview
from tests.qt_helpers import pump, wait_for_preview

RED, BLUE = QColor("#ff0000"), QColor("#0000ff")


def _solid(color: QColor, size: tuple[int, int] = (320, 180)) -> QImage:
    image = QImage(size[0], size[1], QImage.Format.Format_RGB32)
    image.fill(color)
    return image


def _press_move(widget: ComparePreview, start: QPointF, end: QPointF) -> None:
    def event(kind: QMouseEvent.Type, position: QPointF) -> QMouseEvent:
        return QMouseEvent(
            kind,
            position,
            position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

    widget.mousePressEvent(event(QMouseEvent.Type.MouseButtonPress, start))
    widget.mouseMoveEvent(event(QMouseEvent.Type.MouseMove, end))
    widget.mouseReleaseEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            end,
            end,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )


def test_split_modes_show_before_on_one_side_of_a_draggable_line(app: QApplication) -> None:
    preview = ComparePreview()
    preview.resize(800, 500)
    preview.show()
    preview.set_images(_solid(RED), _solid(BLUE))
    image = preview.grab().toImage()
    rect = preview.after_rect()
    assert preview.before_rect() == rect, "split modes draw both frames in the same rectangle"
    assert image.pixelColor(int(rect.left() + 40), int(rect.center().y())) == RED
    assert image.pixelColor(int(rect.right() - 40), int(rect.center().y())) == BLUE
    assert preview.cursor().shape() == Qt.CursorShape.SizeHorCursor

    _press_move(preview, rect.center(), QPointF(rect.left() + rect.width() * 0.25, rect.center().y()))
    image = preview.grab().toImage()
    assert image.pixelColor(int(rect.left() + rect.width() * 0.4), int(rect.center().y())) == BLUE, "divider moved left"

    preview.set_compare_mode(CompareMode.SPLIT_HORIZONTAL)
    assert preview.cursor().shape() == Qt.CursorShape.SizeVerCursor
    _press_move(preview, rect.center(), rect.center())  # divider back to the middle, now horizontal
    image = preview.grab().toImage()
    x = int(rect.center().x())
    assert image.pixelColor(x, int(rect.top() + 30)) == RED and image.pixelColor(x, int(rect.bottom() - 30)) == BLUE
    assert image.pixelColor(int(rect.left() + 30), int(rect.top() + 30)) == RED, "a vertical position no longer matters"
    preview.close()


@pytest.mark.parametrize("mode", [CompareMode.SIDE_BY_SIDE, CompareMode.STACKED])
def test_two_panel_modes_show_both_whole_frames(app: QApplication, mode: CompareMode) -> None:
    preview = ComparePreview()
    preview.resize(900, 700)
    preview.show()
    preview.set_compare_mode(mode)
    assert preview.cursor().shape() == Qt.CursorShape.ArrowCursor
    preview.set_images(_solid(RED), _solid(BLUE))
    image = preview.grab().toImage()
    before, after = preview.before_rect(), preview.after_rect()
    assert before.size() == after.size() and not before.intersects(after)
    assert before == preview.before_panel_rect() and after == preview.after_panel_rect(), "fit mode: the whole frame"
    if mode is CompareMode.SIDE_BY_SIDE:
        assert abs(after.left() - before.right() - PANEL_GAP) < 1 and abs(before.top() - after.top()) < 1, "adjacent"
        assert abs((before.left() + after.right()) / 2 - preview.width() / 2) < 1.5, "the pair is centred"
    else:
        assert abs(after.top() - before.bottom() - PANEL_GAP) < 1 and abs(before.left() - after.left()) < 1, "adjacent"
        assert abs((before.top() + after.bottom()) / 2 - preview.height() / 2) < 1.5, "the pair is centred"
    assert abs(after.width() / after.height() - 320 / 180) < 0.02, "each panel keeps the frame's aspect ratio"
    assert image.pixelColor(before.center().toPoint()) == RED and image.pixelColor(after.center().toPoint()) == BLUE

    _press_move(preview, after.center(), after.topLeft() + QPointF(5, 5))
    assert image.pixelColor(after.center().toPoint()) == BLUE, "there is no divider to drag"

    # At a fixed zoom the frames grow inside the panels, both stay on screen and the hand pans them together.
    pans: list[tuple[int, int]] = []
    preview.panRequested.connect(lambda dx, dy: pans.append((dx, dy)))
    preview.set_fit_mode(False)
    preview.set_display_reference_size(320, 180)
    preview.set_zoom_factor(4.0)  # 1280x720 per frame: more than a panel can show
    assert preview.expands_to_viewport() and preview.sizeHint() == preview.minimumSizeHint()
    image = preview.grab().toImage()
    before_panel, after_panel = preview.before_panel_rect(), preview.after_panel_rect()
    before, after = preview.before_rect(), preview.after_rect()
    assert not before_panel.intersects(after_panel) and before_panel.size() == after_panel.size()
    assert before.size() == QSizeF(1280, 720) == after.size(), "drawn at 400%"
    assert before_panel.width() < 1280 or before_panel.height() < 720, "clipped to the panel"
    assert before.topLeft() - before_panel.topLeft() == after.topLeft() - after_panel.topLeft(), "same region in both"
    assert image.pixelColor(before_panel.center().toPoint()) == RED
    assert image.pixelColor(after_panel.center().toPoint()) == BLUE
    preview.set_pan_mode(True)
    _press_move(preview, after_panel.center(), after_panel.center() + QPointF(-60, 0))
    preview.grab()
    assert preview.before_rect().left() == pytest.approx(before.left() - 60), "the content followed the hand"
    assert preview.after_rect().left() == pytest.approx(after.left() - 60), "both panels moved together"
    assert not pans, "the panels pan in place: nothing for the scroll area to do"
    preview.set_pan_mode(False)

    preview.set_compare_mode(CompareMode.SPLIT_VERTICAL)
    assert not preview.expands_to_viewport(), "a split at a fixed zoom grows the widget and scrolls"
    assert preview.sizeHint() == QSize(1280 + 2 * 18, 720 + 2 * 18)
    preview.close()


def test_compare_selector_sits_next_to_the_quality_selector(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    combo = window.preview_compare_combo
    assert combo.parentWidget() is window.preview_bar and combo.currentData() == CompareMode.SPLIT_VERTICAL.value
    bar_widgets = [
        item.widget()
        for item in (window.preview_bar.layout().itemAt(i) for i in range(window.preview_bar.layout().count()))
        if item.widget() is not None
    ]
    quality = bar_widgets.index(window.preview_render_mode_combo)
    assert bar_widgets[quality + 1] is combo
    window._open_video(ui_video)
    wait_for_preview(app, window)
    select_combo_data(combo, CompareMode.SIDE_BY_SIDE.value)
    pump(app, 0.05)
    assert window.preview.compare_mode is CompareMode.SIDE_BY_SIDE
    window.preview.grab()
    assert not window.preview.before_rect().intersects(window.preview.after_rect())
    select_combo_data(window.preview_zoom_combo, 200)
    pump(app, 0.05)
    assert window.preview_scroll.widgetResizable(), "two panels at a fixed zoom stay inside the viewport"
    select_combo_data(combo, CompareMode.SPLIT_VERTICAL.value)
    assert window.preview.compare_mode is CompareMode.SPLIT_VERTICAL
    assert not window.preview_scroll.widgetResizable(), "a split at a fixed zoom scrolls"
    window.preview_zoom_combo.setCurrentIndex(0)
    assert window.preview_scroll.widgetResizable()
