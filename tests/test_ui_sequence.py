"""Sequences in the main window: several sources on one timeline, settings per clip, split/remove/move, export."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from space_video_enhancer.export.ffmpeg import LIBX264
from space_video_enhancer.settings import StabilizationMode
from space_video_enhancer.timeline import FrameRange
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.sections import exporting as exporting_module
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.conftest import UI_VIDEO_B_FRAMES, UI_VIDEO_FPS, UI_VIDEO_FRAMES
from tests.qt_helpers import wait_for_preview, wait_until

TOTAL = UI_VIDEO_FRAMES + UI_VIDEO_B_FRAMES


def _open_two(app: QApplication, window: MainWindow, ui_video: Path, ui_video_b: Path) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window._add_video(ui_video_b)
    wait_until(app, lambda: len(window._preview_controller.videos) == 2, what="worker adopted both decoders")


def _go_to(app: QApplication, window: MainWindow, frame: int) -> None:
    window.transport.set_value(frame)
    wait_for_preview(app, window)


def test_adding_a_video_extends_the_timeline_and_the_export_range(
    app: QApplication, window: MainWindow, ui_video: Path, ui_video_b: Path
) -> None:
    _open_two(app, window, ui_video, ui_video_b)
    assert len(window.sequence) == 2 and window.frame_count == TOTAL
    assert (window.clip_start_frame, window.clip_end_frame) == (0, TOTAL - 1), "a range reaching the end still does"
    blocks = window.transport.timeline.clips()
    assert [(block.start, block.count, block.name) for block in blocks] == [
        (0, UI_VIDEO_FRAMES, ui_video.name),
        (UI_VIDEO_FRAMES, UI_VIDEO_B_FRAMES, ui_video_b.name),
    ]
    assert window.transport.timeline.active_clip() == 0
    assert "2 clipes" in window.status.video_info_label.text() if hasattr(window.status, "video_info_label") else True
    assert window.windowTitle().startswith(f"{ui_video.name} +1")
    assert window.transport.timecode_label.text().endswith("/ 00:00.20")
    assert window.sequence.clips[1].settings.filters == window.sequence.clips[0].settings.filters
    assert window.sequence.fps_mismatch() is None


def test_tabs_show_and_store_the_settings_of_the_clip_under_the_playhead(
    app: QApplication, window: MainWindow, ui_video: Path, ui_video_b: Path
) -> None:
    _open_two(app, window, ui_video, ui_video_b)
    _go_to(app, window, 6)
    assert window.active_clip_index == 0
    window.color_tab.brightness_slider.setValue(80)
    wait_for_preview(app, window)
    assert window.sequence.clips[0].settings.filters.brightness == 80

    _go_to(app, window, 15)
    assert window.active_clip_index == 1 and window.transport.timeline.active_clip() == 1
    assert window.transport.delete_clip_button.isEnabled() and window.transport.move_left_button.isEnabled()
    assert window.color_tab.brightness_slider.value() == 50, "the second clip kept the look it was added with"
    window.color_tab.brightness_slider.setValue(30)
    select_combo_data(window.stabilization_tab.mode_combo, StabilizationMode.CAMERA_LOCK.value)
    wait_for_preview(app, window)
    assert window.sequence.clips[1].settings.filters.brightness == 30
    assert window.sequence.clips[1].stabilization_reference_frame == 3, "captured on this clip's source frame"
    assert window.sequence.clips[0].settings.filters.brightness == 80, "the first clip did not change"

    _go_to(app, window, 2)
    assert window.active_clip_index == 0
    assert window.color_tab.brightness_slider.value() == 80
    assert window.stabilization_tab.mode() is StabilizationMode.OFF and window.stabilization_reference_frame is None
    _go_to(app, window, 14)
    assert window.stabilization_tab.mode() is StabilizationMode.CAMERA_LOCK
    assert window.stabilization_reference_frame is not None, "the reference frame came back from the cache"
    assert "1.10" not in window.grade_tab.gains_label.text()


def test_preview_letterboxes_other_clips_to_the_first_clips_canvas(
    app: QApplication, window: MainWindow, ui_video: Path, ui_video_b: Path
) -> None:
    _open_two(app, window, ui_video, ui_video_b)
    _go_to(app, window, 15)
    images = window._last_preview
    assert images is not None and images.source.shape[:2] == (180, 240), "decoded from the 4:3 file"
    assert (images.after.width(), images.after.height()) == (320, 180), "padded to the 16:9 canvas"
    _go_to(app, window, 4)
    images = window._last_preview
    assert images is not None and (images.after.width(), images.after.height()) == (320, 180)


def test_split_at_the_playhead_gives_each_half_its_own_settings(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    window.color_tab.brightness_slider.setValue(70)
    _go_to(app, window, 5)
    window._split_clip_at_playhead()
    assert [clip.range for clip in window.sequence.clips] == [FrameRange(0, 4), FrameRange(5, UI_VIDEO_FRAMES - 1)]
    assert all(clip.settings.filters.brightness == 70 for clip in window.sequence.clips)
    assert window.frame_count == UI_VIDEO_FRAMES and window.active_clip_index == 1
    assert len(window.transport.timeline.clips()) == 2
    window.color_tab.brightness_slider.setValue(20)
    wait_for_preview(app, window)
    assert [clip.settings.filters.brightness for clip in window.sequence.clips] == [70, 20]
    _go_to(app, window, 5)
    window._split_clip_at_playhead()
    assert len(window.sequence) == 2 and "nada a dividir" in window.status.message()


def test_moving_and_removing_clips_keeps_the_playhead_on_its_frame(
    app: QApplication, window: MainWindow, ui_video: Path, ui_video_b: Path
) -> None:
    _open_two(app, window, ui_video, ui_video_b)
    _go_to(app, window, 6)
    window._move_clip(0, 1)
    assert [clip.name for clip in window.sequence.clips] == [ui_video_b.name, ui_video.name]
    assert window.current_frame_index == UI_VIDEO_B_FRAMES + 6 and window.active_clip_index == 1
    wait_for_preview(app, window)
    assert window._last_preview is not None and window._last_preview.source_path == ui_video
    window._remove_clip(1)
    assert [clip.name for clip in window.sequence.clips] == [ui_video_b.name]
    assert window.frame_count == UI_VIDEO_B_FRAMES and window.current_frame_index == UI_VIDEO_B_FRAMES - 1
    assert (window.clip_start_frame, window.clip_end_frame) == (0, UI_VIDEO_B_FRAMES - 1)
    wait_until(app, lambda: len(window._preview_controller.videos) == 1, what="dropped decoder released")
    window._remove_clip(0)
    assert len(window.sequence) == 1, "the last clip cannot be removed"


def test_a_video_with_another_fps_is_retimed_onto_the_timeline(
    app: QApplication, window: MainWindow, ui_video: Path, ui_video_other_fps: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(str(args[2])))
    window._open_video(ui_video)  # 24 fps: this pins the timeline rate
    wait_for_preview(app, window)
    window._add_video(ui_video_other_fps)  # 30 fps, 6 frames -> retimed to ~5 frames at 24 fps
    assert len(window.sequence) == 2 and ui_video_other_fps in window._sources
    assert not warnings, "a different fps is retimed now, not refused"
    assert window.sequence.timeline_fps == UI_VIDEO_FPS
    assert UI_VIDEO_FRAMES < window.sequence.timeline_count < UI_VIDEO_FRAMES + 6  # the 30 fps clip is compressed


class _FakeExportController(QObject):
    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)
    created: list[_FakeExportController] = []

    def __init__(self, request: object, backend: object, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.request = request
        _FakeExportController.created.append(self)

    def start(self) -> None:
        pass

    def request_cancel(self) -> None:
        pass

    def stop(self) -> None:
        pass


def test_cuts_span_clips_and_the_export_joins_the_sequence(
    app: QApplication,
    window: MainWindow,
    ui_video: Path,
    ui_video_b: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _open_two(app, window, ui_video, ui_video_b)
    window._add_cut(10, 13)  # straddles the boundary at frame 12
    assert window.cuts == [(10, 13)] and window.transport.timeline.cuts() == ((10, 13),)
    assert window.sequence.clips[0].cuts == (FrameRange(10, 11),) and window.sequence.clips[1].cuts == (
        FrameRange(0, 1),
    )
    _go_to(app, window, 2)
    window._mark_clip_start()
    _go_to(app, window, 17)
    window._mark_clip_end()
    assert "Saída: 12 frames" in window.export_tab.summary_label.text()

    monkeypatch.setattr(exporting_module, "ExportController", _FakeExportController)
    monkeypatch.setattr(exporting_module, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(exporting_module, "resolve_encoder", lambda probe, **kwargs: LIBX264)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(tmp_path / "out.mp4"), ""))
    window._export_video()
    assert len(_FakeExportController.created) == 1
    request = _FakeExportController.created[-1].request
    assert request.expected_frames == 12 and request.sequence.source_paths == (ui_video, ui_video_b)
    assert [clip.segments for clip in request.sequence.clips] == [(FrameRange(2, 9),), (FrameRange(2, 5),)]
    assert request.output_path == tmp_path / "out.mp4" and window.status.exporting
    assert not window.add_video_action.isEnabled() and not window.open_action.isEnabled()


def test_filmstrip_and_waveform_read_every_source(
    app: QApplication, window: MainWindow, ui_video: Path, ui_video_b: Path
) -> None:
    _open_two(app, window, ui_video, ui_video_b)
    timeline = window.transport.timeline
    wait_until(
        app, lambda: {video.path for video in window._filmstrip.videos} == {ui_video, ui_video_b}, what="filmstrip"
    )
    wait_until(app, lambda: timeline.thumbnail_count() >= len(timeline.thumbnail_slots()), what="filmstrip cells")
    cell_width, _cell_height = timeline.thumbnail_size()
    last_cell = timeline.nearest_thumbnail(TOTAL - 1, cell_width)
    assert last_cell is not None and last_cell.width() / last_cell.devicePixelRatio() == pytest.approx(
        cell_width, abs=1
    )
    for path in (ui_video, ui_video_b):
        wait_until(app, lambda path=path: timeline.audio_state(path) != "loading", what="waveform state")
        assert timeline.audio_state(path) in ("none", "unavailable")


def test_dragging_a_clip_on_the_strip_reorders_the_sequence(
    app: QApplication, window: MainWindow, ui_video: Path, ui_video_b: Path
) -> None:
    _open_two(app, window, ui_video, ui_video_b)
    _go_to(app, window, 3)
    timeline = window.transport.timeline
    scale = timeline.scale
    y = timeline.clip_track_rect().center().y()
    for kind, x, buttons in (
        (QEvent.Type.MouseButtonPress, scale.x_of(4), Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseMove, scale.x_of(10), Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseMove, scale.x_of(TOTAL - 0.5), Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseButtonRelease, scale.x_of(TOTAL - 0.5), Qt.MouseButton.NoButton),
    ):
        point = QPointF(x, y)
        QApplication.sendEvent(
            timeline,
            QMouseEvent(kind, point, point, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier),
        )
    assert [clip.name for clip in window.sequence.clips] == [ui_video_b.name, ui_video.name]
    assert window.current_frame_index == UI_VIDEO_B_FRAMES + 3, "the playhead stays on its frame of the moved clip"
    assert [block.name for block in timeline.clips()] == [ui_video_b.name, ui_video.name]


def test_changing_the_timeline_fps_retimes_the_sequence_and_undoes(
    app: QApplication, window: MainWindow, ui_video: Path, ui_video_other_fps: Path
) -> None:
    window._open_video(ui_video)  # 24 fps: pins the timeline
    wait_for_preview(app, window)
    window._add_video(ui_video_other_fps)  # 30 fps, retimed onto the 24 fps timeline
    assert window.sequence.timeline_fps == UI_VIDEO_FPS
    faster = window.sequence.clips[1].fps  # 30 fps, offered in the combo as a clip rate
    window.transport.set_value(window.frame_count - 1)  # playhead at the end
    before_seconds = window.current_frame_index / window.sequence.fps

    select_combo_data(window.export_tab.timeline_fps_combo, faster)  # fires timelineFpsChanged

    assert window.sequence.timeline_fps == faster and window.sequence.fps == faster
    assert window.export_tab.timeline_fps() == faster
    assert window.current_frame_index / window.sequence.fps == pytest.approx(before_seconds, abs=1 / faster)
    assert window.undo_action.text().endswith("linha do tempo"), window.undo_action.text()

    window._undo_edit()
    assert window.sequence.timeline_fps == UI_VIDEO_FPS
    assert window.export_tab.timeline_fps() == UI_VIDEO_FPS


def test_copy_and_paste_duplicates_the_clip_under_the_playhead(
    app: QApplication, window: MainWindow, ui_video: Path, ui_video_b: Path
) -> None:
    _open_two(app, window, ui_video, ui_video_b)  # clip 0: ui_video, clip 1: ui_video_b
    _go_to(app, window, 3)  # inside clip 0
    window.color_tab.brightness_slider.setValue(65)
    wait_for_preview(app, window)

    window._copy_active_clip()
    assert window.paste_clip_action.isEnabled()
    window._paste_clip()

    assert len(window.sequence) == 3
    assert window.sequence.clips[1].source.path == ui_video, "the copy sits right after the original"
    assert window.sequence.clips[1].settings.filters.brightness == 65, "and carries its look"
    assert window.sequence.clips[2].source.path == ui_video_b, "the old clip 1 shifted right"
    assert window.active_clip_index == 1, "the playhead landed on the pasted copy"

    window._undo_edit()
    assert len(window.sequence) == 2, "undo removes the pasted copy"
    window._redo_edit()
    assert len(window.sequence) == 3, "redo brings it back"

    window.color_tab.brightness_slider.setValue(20)  # editing the copy is independent of the original
    wait_for_preview(app, window)
    assert window.sequence.clips[1].settings.filters.brightness == 20
    assert window.sequence.clips[0].settings.filters.brightness == 65


def test_paste_attributes_copies_the_look_not_the_video(
    app: QApplication, window: MainWindow, ui_video: Path, ui_video_b: Path
) -> None:
    _open_two(app, window, ui_video, ui_video_b)
    _go_to(app, window, 3)  # clip 0
    window.color_tab.brightness_slider.setValue(70)
    wait_for_preview(app, window)
    window._copy_active_clip()

    _go_to(app, window, 15)  # clip 1 (the other file), still at its default look
    assert window.active_clip_index == 1 and window.color_tab.brightness_slider.value() == 50
    window._paste_clip_attributes()

    assert len(window.sequence) == 2, "no new clip: only the look is pasted"
    assert window.sequence.clips[1].source.path == ui_video_b, "the video is untouched"
    assert window.sequence.clips[1].settings.filters.brightness == 70, "the look is copied"
    assert window.color_tab.brightness_slider.value() == 70, "the tabs reloaded to show it"
    assert window.sequence.clips[0].settings.filters.brightness == 70, "the source clip is unchanged"

    window._undo_edit()
    assert window.sequence.clips[1].settings.filters.brightness == 50
