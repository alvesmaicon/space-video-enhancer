"""The zoomable timeline: frame/pixel scale, ruler steps, hit testing and mouse edits."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from space_video_enhancer.ui.widgets.timeline import (
    MAX_PIXELS_PER_FRAME,
    MIN_LABEL_SPACING_PX,
    RULER_HEIGHT,
    ClipBlock,
    Hit,
    TimelineScale,
    TimelineWidget,
    ruler_label,
    ruler_steps,
)
from tests.qt_helpers import pump

WIDTH = 800
FRAMES = 1200
FPS = 30.0


# --- pure geometry ------------------------------------------------------------------------------


def test_scale_fits_the_clip_and_clamps_zoom_and_scroll() -> None:
    scale = TimelineScale(frame_count=FRAMES, width=WIDTH)
    scale.fit()
    assert scale.is_fit() and scale.pixels_per_frame == pytest.approx(WIDTH / FRAMES)
    assert scale.frame_index_at(0) == 0 and scale.frame_index_at(WIDTH) == FRAMES - 1
    assert scale.frame_index_at(WIDTH - 1) == 1198  # pixel 799 covers frames 1198.5-1200
    assert scale.frame_index_at(-50) == 0 and scale.frame_index_at(5000) == FRAMES - 1
    scale.zoom_by(1e9)
    assert scale.is_max_zoom() and scale.pixels_per_frame == MAX_PIXELS_PER_FRAME
    scale.scroll_to(1e9)
    assert scale.scroll == pytest.approx(FRAMES - WIDTH / MAX_PIXELS_PER_FRAME)
    scale.zoom_by(1e-9)
    assert scale.is_fit() and scale.scroll == 0.0


def test_zoom_keeps_the_frame_under_the_anchor() -> None:
    scale = TimelineScale(frame_count=FRAMES, width=WIDTH)
    scale.fit()
    anchor_x = 600.0
    anchor_frame = scale.frame_at(anchor_x)
    scale.zoom_by(4.0, anchor_x)
    assert scale.pixels_per_frame == pytest.approx(4 * WIDTH / FRAMES)
    assert scale.frame_at(anchor_x) == pytest.approx(anchor_frame)
    assert scale.x_of(anchor_frame) == pytest.approx(anchor_x)
    scale.zoom_by(1.5, 0.0)  # anchored at the left edge: the first visible frame does not move
    assert scale.frame_at(0.0) == pytest.approx(scale.scroll)


def test_resize_keeps_the_fit_or_the_zoom() -> None:
    scale = TimelineScale(frame_count=FRAMES, width=WIDTH)
    scale.fit()
    scale.set_width(1000)
    assert scale.is_fit() and scale.pixels_per_frame == pytest.approx(1000 / FRAMES)
    scale.zoom_by(3.0)
    zoomed = scale.pixels_per_frame
    scale.set_width(400)
    assert scale.pixels_per_frame == pytest.approx(zoomed) and scale.scroll <= scale.max_scroll()


def test_follow_scrolls_only_when_the_frame_is_off_screen() -> None:
    scale = TimelineScale(frame_count=FRAMES, width=WIDTH)
    scale.fit()
    assert scale.follow(1100) is False  # everything is visible when fitted
    scale.zoom_by(10.0, 0.0)  # frames 0..~150 visible
    assert scale.follow(100) is False
    assert scale.follow(300) is True
    assert scale.is_visible(300)
    assert scale.x_of(300) == pytest.approx(scale.visible_frames * 0.1 * scale.pixels_per_frame)
    assert scale.follow(FRAMES - 1) is True and scale.scroll == pytest.approx(scale.max_scroll())


def test_ruler_steps_keep_labels_apart_and_switch_to_frames_when_zoomed() -> None:
    fit = WIDTH / FRAMES  # 0.67 px per frame: 40 s clip in 800 px
    major, minor = ruler_steps(FPS, fit)
    assert major * fit >= MIN_LABEL_SPACING_PX and major % minor == 0 and major % 30 == 0
    assert ruler_label(major, FPS, with_frames=False).count(":") == 1
    assert "." not in ruler_label(90, FPS, with_frames=False)
    major, minor = ruler_steps(FPS, MAX_PIXELS_PER_FRAME)
    assert major < FPS and major * MAX_PIXELS_PER_FRAME >= MIN_LABEL_SPACING_PX
    assert ruler_label(7, FPS, with_frames=True) == "00:00.07"
    for pixels_per_frame in (0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 48.0):
        major, minor = ruler_steps(FPS, pixels_per_frame)
        assert major * pixels_per_frame >= MIN_LABEL_SPACING_PX
        assert minor * pixels_per_frame >= 4 and major % minor == 0


# --- widget -------------------------------------------------------------------------------------


@pytest.fixture
def timeline(app: QApplication):  # type: ignore[no-untyped-def]
    widget = TimelineWidget()
    widget.resize(WIDTH, widget.sizeHint().height())
    widget.show()
    pump(app, 0.05)
    widget.set_media(FRAMES, FPS)
    widget.set_clip_range(0, FRAMES - 1, True)
    yield widget
    widget.close()


def track_y(widget: TimelineWidget) -> float:
    return widget.video_track_rect().center().y()


def mouse(widget: TimelineWidget, kind: QEvent.Type, x: float, y: float, modifiers=Qt.KeyboardModifier.NoModifier):  # type: ignore[no-untyped-def]
    buttons = Qt.MouseButton.LeftButton if kind != QEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton
    event = QMouseEvent(kind, QPointF(x, y), QPointF(x, y), Qt.MouseButton.LeftButton, buttons, modifiers)
    QApplication.sendEvent(widget, event)


def drag(widget: TimelineWidget, x0: float, x1: float, y: float) -> None:
    mouse(widget, QEvent.Type.MouseButtonPress, x0, y)
    for x in (x0 + (x1 - x0) / 2, x1):
        mouse(widget, QEvent.Type.MouseMove, x, y)
    mouse(widget, QEvent.Type.MouseButtonRelease, x1, y)


def test_widget_reports_its_size_and_scrollbar_only_when_zoomed(app: QApplication, timeline: TimelineWidget) -> None:
    assert timeline.width() == WIDTH and timeline.scale.width == WIDTH
    assert timeline.is_fit() and not timeline.scrollbar.isVisible()
    timeline.zoom_in()
    pump(app, 0.02)
    assert not timeline.is_fit() and timeline.scrollbar.isVisible()
    assert timeline.scrollbar.maximum() == pytest.approx(timeline.scale.max_scroll(), abs=1)
    timeline.scrollbar.setValue(200)
    assert timeline.scale.scroll == 200.0
    timeline.zoom_to_fit()
    pump(app, 0.02)
    assert timeline.is_fit() and not timeline.scrollbar.isVisible()


def test_hit_testing_prefers_edges_then_bodies_then_seek(timeline: TimelineWidget) -> None:
    timeline.set_cuts([(300, 599)])
    scale = timeline.scale
    y = track_y(timeline)
    assert timeline.hit_test(QPointF(scale.x_of(300), y)) == Hit("cut_start", 0)
    assert timeline.hit_test(QPointF(scale.x_of(600) - 1, y)) == Hit("cut_end", 0)
    assert timeline.hit_test(QPointF(scale.x_of(450), y)) == Hit("cut_body", 0)
    assert timeline.hit_test(QPointF(scale.x_of(100), y)) == Hit("seek")
    assert timeline.hit_test(QPointF(scale.x_of(450), RULER_HEIGHT / 2)) == Hit("seek")  # the ruler always seeks
    assert timeline.hit_test(QPointF(scale.x_of(FRAMES), RULER_HEIGHT / 2)) == Hit("clip_end")
    assert timeline.hit_test(QPointF(scale.x_of(0) + 2, RULER_HEIGHT / 2)) == Hit("clip_start")
    assert timeline.hit_test(QPointF(10, timeline.content_height() + 5)) == Hit("none")
    assert Hit("cut_start", 0).cursor == Qt.CursorShape.SizeHorCursor
    assert Hit("cut_body", 0).cursor == Qt.CursorShape.OpenHandCursor


def test_clicking_and_dragging_empty_timeline_scrubs(timeline: TimelineWidget) -> None:
    seen: list[int] = []
    timeline.valueChanged.connect(seen.append)
    y = track_y(timeline)
    mouse(timeline, QEvent.Type.MouseButtonPress, timeline.scale.x_of(450) + 0.5, y)
    assert timeline.value() == 450
    mouse(timeline, QEvent.Type.MouseMove, timeline.scale.x_of(900) + 0.5, y)
    mouse(timeline, QEvent.Type.MouseButtonRelease, timeline.scale.x_of(900) + 0.5, y)
    assert timeline.value() == 900 and seen == [450, 900]
    assert timeline.cuts() == ()  # nothing edited


def test_dragging_a_cut_edge_resizes_it_and_reports_the_final_state(timeline: TimelineWidget) -> None:
    timeline.set_cuts([(300, 599)])
    edits: list[tuple[tuple[tuple[int, int], ...], bool]] = []
    timeline.cutsEdited.connect(lambda cuts, finished: edits.append((tuple(cuts), finished)))
    scale = timeline.scale
    y = track_y(timeline)
    drag(timeline, scale.x_of(600), scale.x_of(750), y)  # right edge: the cut now ends before frame 750
    assert timeline.cuts() == ((300, 749),)
    assert edits[-1] == (((300, 749),), True) and all(not finished for _cuts, finished in edits[:-1])
    drag(timeline, scale.x_of(300), scale.x_of(200), y)  # left edge moves earlier
    assert timeline.cuts() == ((200, 749),)
    drag(timeline, scale.x_of(200), scale.x_of(1100), y)  # cannot cross the other edge
    assert timeline.cuts() == ((749, 749),)
    timeline.set_cuts([(300, 599)])
    drag(timeline, scale.x_of(600), scale.x_of(100), y)
    assert timeline.cuts() == ((300, 300),)


def test_dragging_a_cut_body_moves_it_and_selects_it(timeline: TimelineWidget) -> None:
    timeline.set_cuts([(300, 599), (900, 999)])
    selected: list[int] = []
    timeline.cutSelected.connect(selected.append)
    scale = timeline.scale
    y = track_y(timeline)
    drag(timeline, scale.x_of(450), scale.x_of(550), y)
    assert timeline.cuts() == ((400, 699), (900, 999)) and selected == [0]
    drag(timeline, scale.x_of(950), scale.x_of(1190), y)  # clamped to the end of the clip
    assert timeline.cuts() == ((400, 699), (1100, FRAMES - 1)) and selected == [0, 1]
    mouse(timeline, QEvent.Type.MouseButtonPress, scale.x_of(50), y)  # empty area clears the selection
    mouse(timeline, QEvent.Type.MouseButtonRelease, scale.x_of(50), y)
    assert timeline.selected_cut() == -1 and selected[-1] == -1
    # A press on a body without moving only selects.
    edits: list[object] = []
    timeline.cutsEdited.connect(lambda *args: edits.append(args))
    mouse(timeline, QEvent.Type.MouseButtonPress, scale.x_of(500), y)
    mouse(timeline, QEvent.Type.MouseButtonRelease, scale.x_of(500), y)
    assert edits == [] and timeline.selected_cut() == 0


def test_model_updates_are_ignored_while_a_cut_is_being_dragged(timeline: TimelineWidget) -> None:
    timeline.set_cuts([(300, 599)])
    scale = timeline.scale
    y = track_y(timeline)
    mouse(timeline, QEvent.Type.MouseButtonPress, scale.x_of(600), y)
    mouse(timeline, QEvent.Type.MouseMove, scale.x_of(700), y)
    timeline.set_cuts([(0, 10)])  # the model echoes a normalized copy mid-drag: the drag wins
    assert timeline.cuts() == ((300, 699),)
    mouse(timeline, QEvent.Type.MouseButtonRelease, scale.x_of(700), y)
    timeline.set_cuts([(0, 10)])
    assert timeline.cuts() == ((0, 10),)


def test_dragging_the_clip_markers_on_the_ruler(timeline: TimelineWidget) -> None:
    edits: list[tuple[int, int, bool]] = []
    timeline.clipRangeEdited.connect(lambda start, end, finished: edits.append((start, end, finished)))
    scale = timeline.scale
    y = RULER_HEIGHT / 2
    drag(timeline, scale.x_of(FRAMES), scale.x_of(900), y)
    assert edits[-1] == (0, 899, True)
    drag(timeline, scale.x_of(0), scale.x_of(150), y)
    assert edits[-1] == (150, 899, True)
    drag(timeline, scale.x_of(150), scale.x_of(1100), y)  # the start cannot pass the end
    assert edits[-1] == (899, 899, True)


def test_ctrl_wheel_zooms_and_plain_wheel_scrolls(app: QApplication, timeline: TimelineWidget) -> None:
    scale = timeline.scale
    anchor_x = 200.0
    anchor_frame = scale.frame_at(anchor_x)

    def wheel(x: float, angle_y: int, modifiers: Qt.KeyboardModifier) -> None:
        event = QWheelEvent(
            QPointF(x, track_y(timeline)),
            QPointF(x, track_y(timeline)),
            QPoint(),
            QPoint(0, angle_y),
            Qt.MouseButton.NoButton,
            modifiers,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(timeline, event)

    for _ in range(6):
        wheel(anchor_x, 120, Qt.KeyboardModifier.ControlModifier)
    assert not scale.is_fit() and scale.frame_at(anchor_x) == pytest.approx(anchor_frame, abs=0.01)
    before = scale.scroll
    wheel(anchor_x, -120, Qt.KeyboardModifier.NoModifier)  # wheel down: later frames
    assert scale.scroll == pytest.approx(before + scale.visible_frames * 0.1)
    wheel(anchor_x, 120, Qt.KeyboardModifier.NoModifier)
    assert scale.scroll == pytest.approx(before)


def test_playhead_off_screen_scrolls_the_view_along(app: QApplication, timeline: TimelineWidget) -> None:
    timeline.zoom_by(8.0, 0.0)
    assert timeline.visible_range()[1] < 200
    timeline.setValue(600)
    first, last = timeline.visible_range()
    assert first <= 600 < last and abs(timeline.scale.x_of(600) - WIDTH * 0.1) < timeline.scale.pixels_per_frame
    timeline.setValue(610)  # still visible: no scroll
    assert timeline.visible_range() == (first, last)


def test_context_menu_offers_cut_actions(timeline: TimelineWidget) -> None:
    timeline.set_cuts([(300, 599)])
    removed: list[int] = []
    toggled: list[bool] = []
    timeline.cutRemoveRequested.connect(removed.append)
    timeline.cutToggleRequested.connect(lambda: toggled.append(True))
    y = track_y(timeline)
    menu = timeline.context_menu(QPointF(timeline.scale.x_of(450), y))
    texts = [action.text() for action in menu.actions() if not action.isSeparator()]
    assert texts[:3] == ["Remover corte", "Ir para o início do corte", "Ir para o fim do corte"]
    menu.actions()[2].trigger()
    assert timeline.value() == 599
    menu.actions()[0].trigger()
    assert removed == [0] and timeline.selected_cut() == 0
    menu = timeline.context_menu(QPointF(timeline.scale.x_of(100) + 0.5, y))
    assert menu.actions()[0].text() == "Iniciar corte aqui"
    menu.actions()[0].trigger()
    assert toggled == [True] and timeline.value() == 100
    timeline.set_pending_cut(100)
    assert timeline.context_menu(QPointF(timeline.scale.x_of(120), y)).actions()[0].text() == "Terminar corte aqui"


def test_empty_timeline_ignores_the_mouse(app: QApplication) -> None:
    widget = TimelineWidget()
    widget.resize(WIDTH, widget.sizeHint().height())
    widget.show()
    pump(app, 0.02)
    seen: list[int] = []
    widget.valueChanged.connect(seen.append)
    mouse(widget, QEvent.Type.MouseButtonPress, 100, track_y(widget))
    mouse(widget, QEvent.Type.MouseButtonRelease, 100, track_y(widget))
    assert seen == [] and widget.hit_test(QPointF(100, track_y(widget))) == Hit("none")
    widget.grab()  # paints the hint without a video
    widget.close()


def test_clip_strip_names_the_clips_and_offers_split_and_remove(app: QApplication) -> None:
    timeline = TimelineWidget()
    timeline.resize(800, timeline.sizeHint().height())
    timeline.show()
    timeline.set_media(100, 24.0)
    timeline.set_clips([ClipBlock(0, 60, "a.mp4", Path("a.mp4")), ClipBlock(60, 40, "b.mp4", Path("b.mp4"), 5)])
    scale = timeline.scale
    assert timeline.frame_count == 100 and timeline.clip_at(59) == 0 and timeline.clip_at(60) == 1
    assert timeline.clip_track_rect().bottom() < timeline.video_track_rect().top()
    assert timeline.hit_test(QPointF(scale.x_of(70.5), timeline.clip_track_rect().center().y())) == Hit("clip", 1)
    assert timeline.hit_test(QPointF(scale.x_of(70.5), timeline.video_track_rect().center().y())) == Hit("seek")

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(scale.x_of(70.5), timeline.clip_track_rect().center().y()),
        QPointF(scale.x_of(70.5), timeline.clip_track_rect().center().y()),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(timeline, press)
    assert timeline.value() == 70, "clicking the strip seeks like the empty timeline"

    splits: list[int] = []
    removed: list[int] = []
    timeline.clipSplitRequested.connect(lambda: splits.append(timeline.value()))
    timeline.clipRemoveRequested.connect(removed.append)
    menu = timeline.context_menu(QPointF(scale.x_of(80.5), timeline.video_track_rect().center().y()))
    actions = {action.text(): action for action in menu.actions() if action.text()}
    assert actions["Dividir clipe aqui"].isEnabled() and "Remover clipe 2 (b.mp4)" in actions
    actions["Dividir clipe aqui"].trigger()
    actions["Remover clipe 2 (b.mp4)"].trigger()
    assert splits == [80] and removed == [1]
    at_boundary = timeline.context_menu(QPointF(scale.x_of(60.5), timeline.video_track_rect().center().y()))
    boundary_actions = {action.text(): action for action in at_boundary.actions() if action.text()}
    assert not boundary_actions["Dividir clipe aqui"].isEnabled(), "a clip already starts here"

    timeline.set_active_clip(1)
    assert timeline.active_clip() == 1
    image = QImage(16, 9, QImage.Format.Format_RGB888)
    timeline.set_thumbnail(3, image)
    timeline.zoom_in()
    timeline.zoom_in()  # well above the fit of the shorter layout below (800 px / 60 frames)
    zoom = scale.pixels_per_frame
    timeline.set_clips([ClipBlock(0, 60, "a.mp4", Path("a.mp4"))])
    assert timeline.frame_count == 60 and timeline.thumbnail_count() == 0, "the layout changed: cells are stale"
    assert timeline.value() == 59 and timeline.active_clip() == -1
    assert scale.pixels_per_frame == pytest.approx(zoom), "the zoom level survives a layout change"
    timeline.grab()
    timeline.close()
