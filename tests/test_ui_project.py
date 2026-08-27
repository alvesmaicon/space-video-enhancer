"""Projects in the window: save, reopen, the dirty title, recents, missing media and the unsaved-changes prompt."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from space_video_enhancer.project import load_project
from space_video_enhancer.settings import StabilizationMode
from space_video_enhancer.ui import main_window as main_window_module
from space_video_enhancer.ui.main_window import MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.conftest import UI_VIDEO_B_FRAMES, UI_VIDEO_FRAMES
from tests.qt_helpers import pump, wait_for_preview, wait_until


def _open(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)


def _add(app: QApplication, window: MainWindow, ui_video_b: Path) -> None:
    window._add_video(ui_video_b)
    wait_until(app, lambda: len(window._preview_controller.videos) == 2, what="worker adopted both decoders")


def _go_to(app: QApplication, window: MainWindow, frame: int) -> None:
    window.transport.set_value(frame)
    wait_for_preview(app, window)


def test_save_and_reopen_a_project_restores_the_document(
    app: QApplication,
    window: MainWindow,
    ui_video: Path,
    ui_video_b: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _open(app, window, ui_video)
    assert window.windowTitle() == f"{ui_video.name} — Space Video Enhancer" and not window.has_unsaved_changes
    _add(app, window, ui_video_b)
    assert window.windowTitle() == f"{ui_video.name} +1 * — Space Video Enhancer", "adding a clip is an edit"

    _go_to(app, window, 3)
    window.color_tab.brightness_slider.setValue(70)
    wait_for_preview(app, window)
    window._add_cut(5, 6)
    select_combo_data(window.stabilization_tab.mode_combo, StabilizationMode.CAMERA_LOCK.value)
    wait_for_preview(app, window)
    assert window.sequence.clips[0].stabilization_reference_frame == 3
    _go_to(app, window, 15)
    window.color_tab.brightness_slider.setValue(30)
    wait_for_preview(app, window)
    select_combo_data(window.export_tab.thread_combo, 2)

    project_path = tmp_path / "meu-projeto.svep"
    monkeypatch.setattr(
        main_window_module.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(project_path), "")
    )
    assert window._save_project(), "no file yet: Salvar goes through Salvar como…"
    assert project_path.is_file() and window.project_path == project_path
    assert window.windowTitle() == "meu-projeto — Space Video Enhancer" and not window.has_unsaved_changes
    assert project_path in window._recent_projects.paths

    window.color_tab.brightness_slider.setValue(40)
    wait_for_preview(app, window)
    assert window.windowTitle() == "meu-projeto * — Space Video Enhancer"
    monkeypatch.setattr(
        main_window_module.QFileDialog, "getSaveFileName", lambda *args, **kwargs: pytest.fail("no dialog")
    )
    assert window._save_project(), "with a file, Ctrl+S saves in place"
    assert not window.has_unsaved_changes

    _open(app, window, ui_video)  # something else entirely, then back to the project
    assert window.project_path is None and len(window.sequence) == 1
    assert window._open_project(project_path)
    wait_for_preview(app, window)
    assert len(window.sequence) == 2 and window.project_path == project_path and not window.has_unsaved_changes
    assert window.windowTitle() == "meu-projeto — Space Video Enhancer"
    assert window.current_frame_index == 15 and window.active_clip_index == 1
    assert (window.clip_start_frame, window.clip_end_frame) == (0, UI_VIDEO_FRAMES + UI_VIDEO_B_FRAMES - 1)
    assert window.export_tab.worker_count() == 2
    assert [clip.settings.filters.brightness for clip in window.sequence.clips] == [70, 40]
    assert window.cuts == [(5, 6)] and window.transport.timeline.cuts() == ((5, 6),)
    first = window.sequence.clips[0]
    assert (
        first.settings.stabilization.mode is StabilizationMode.CAMERA_LOCK and first.stabilization_reference_frame == 3
    )
    wait_until(app, lambda: (ui_video, 3) in window._reference_frames, what="reference frame decoded again")
    _go_to(app, window, 3)
    assert window.stabilization_reference_frame is not None and window.stabilization_reference_frame_index == 3
    assert "Referência: frame 4" in window.stabilization_tab.info_label.text()
    assert not window.undo_action.isEnabled(), "opening a project starts a fresh history"

    window._rebuild_recent_projects_menu()
    texts = [action.text() for action in window.recent_projects_menu.actions()]
    assert texts[0] == "meu-projeto.svep" and texts[-1] == "Limpar lista"


def test_a_project_whose_media_moved_opens_with_the_clip_marked(
    app: QApplication, window: MainWindow, ui_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    moved = media / "gone.mp4"
    shutil.copy(ui_video, moved)
    _open(app, window, moved)
    window._add_video(ui_video)
    wait_until(app, lambda: len(window._preview_controller.videos) == 2, what="worker adopted both decoders")
    project_path = tmp_path / "p.svep"
    assert window._write_project(project_path)

    _open(app, window, ui_video)  # let go of the decoders, then make the first file disappear
    wait_until(app, lambda: len(window._preview_controller.videos) == 1, what="worker released the old decoders")
    moved.unlink()
    assert window._open_project(project_path)
    pump(app, 0.1)
    assert window._missing_sources == {moved}
    assert [block.missing for block in window.transport.timeline.clips()] == [True, False]
    assert "Mídia não encontrada" in window.status.message() and "gone.mp4" in window.status.message()
    assert window.transport.timeline.audio_state(moved) == "missing"
    assert len(window._preview_controller.videos) == 1, "only the file that exists gets a decoder"

    generation = window._preview_generation
    window.transport.set_value(2)  # inside the missing clip: nothing to render, the status says why
    pump(app, 0.1)
    assert window._preview_generation == generation and "Mídia não encontrada" in window.status.message()
    _go_to(app, window, UI_VIDEO_FRAMES + 2)
    assert window._decoded_source == ui_video, "the clip that is present still renders"

    warnings: list[str] = []
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window._export_video()
    assert warnings and "gone.mp4" in warnings[0]

    assert window._save_project(), "saving keeps the missing file referenced for when it comes back"
    assert load_project(project_path).sequence.clips[0].source.path == moved


def test_unsaved_changes_are_confirmed_before_closing_or_replacing(
    app: QApplication, window: MainWindow, ui_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _open(app, window, ui_video)
    window.confirm_discard = True
    answers: list[str] = []
    monkeypatch.setattr(window, "_ask_unsaved_changes", lambda: answers.pop(0))
    assert window._confirm_discard_changes(), "nothing to lose: no question asked"

    window.color_tab.brightness_slider.setValue(70)
    wait_for_preview(app, window)
    assert window.has_unsaved_changes and window.windowTitle().endswith("* — Space Video Enhancer")
    answers.append("cancel")
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted() and window.isVisible() and answers == []

    answers.append("discard")
    assert window._confirm_discard_changes()

    project_path = tmp_path / "s.svep"
    monkeypatch.setattr(
        main_window_module.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(project_path), "")
    )
    answers.append("save")
    assert window._confirm_discard_changes() and project_path.is_file() and not window.has_unsaved_changes

    window.color_tab.brightness_slider.setValue(20)
    wait_for_preview(app, window)
    monkeypatch.setattr(main_window_module.QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(ui_video), ""))
    answers.append("cancel")
    window._select_video()
    assert window.color_tab.brightness_slider.value() == 20 and window.project_path == project_path, "cancelled"
    window.confirm_discard = False


def test_recent_projects_menu_drops_a_project_that_no_longer_exists(
    app: QApplication, window: MainWindow, ui_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _open(app, window, ui_video)
    project_path = tmp_path / "r.svep"
    assert window._write_project(project_path)
    window._rebuild_recent_projects_menu()
    assert window.recent_projects_menu.actions()[0].text() == "r.svep"

    project_path.unlink()
    errors: list[str] = []
    monkeypatch.setattr(main_window_module.QMessageBox, "critical", lambda *args: errors.append(args[2]))
    assert not window._open_project(project_path)
    assert errors and "r.svep" in errors[0] and project_path not in window._recent_projects.paths
    assert window.project_path == project_path, "the failed open leaves the document alone"
    window._clear_recent_projects()
    assert [action.text() for action in window.recent_projects_menu.actions()] == ["Nenhum projeto recente"]
