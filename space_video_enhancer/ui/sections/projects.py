"""Project files (.svep): save, open, recents, the dirty title and the unsaved-changes prompt."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from space_video_enhancer.project import (
    PROJECT_FILE_FILTER,
    PROJECT_SUFFIX,
    Project,
    ProjectError,
    load_project,
    save_project,
    with_sources,
)
from space_video_enhancer.sequence import ClipSource
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.ui.format import format_duration
from space_video_enhancer.ui.sections.editing import EditState
from space_video_enhancer.ui.sections.state import WindowState
from space_video_enhancer.ui.tabs.base import select_combo_data

LOG = get_logger("ui")


class ProjectSection(WindowState):
    """Project files (.svep): save, open, recents, the dirty title and the unsaved-changes prompt."""

    def _project_snapshot(self) -> tuple[EditState, int]:
        return (self._edit_state(), self.export_tab.worker_count())

    @property
    def has_unsaved_changes(self) -> bool:
        """Edits since the document was opened or saved (a freshly opened video counts as saved until edited)."""
        return not self.sequence.is_empty and self._saved_snapshot != self._project_snapshot()

    def _document_name(self) -> str:
        if self.project_path is not None:
            return self.project_path.stem
        first = self.sequence.clips[0].name
        return first if len(self.sequence) == 1 else f"{first} +{len(self.sequence) - 1}"

    def _update_window_title(self) -> None:
        if self.sequence.is_empty:
            self.setWindowTitle("Space Video Enhancer")
            if self._monitor is not None:
                self._monitor.set_document_name(None)
            return
        marker = " *" if self.has_unsaved_changes else ""
        self.setWindowTitle(f"{self._document_name()}{marker} — Space Video Enhancer")
        if self._monitor is not None:
            self._monitor.set_document_name(self._document_name())

    def _current_project(self) -> Project:
        return Project(
            self.sequence,
            (self.clip_start_frame, self.clip_end_frame),
            self.current_frame_index,
            self.export_tab.worker_count(),
            tuple(sorted(self._missing_sources)),
        )

    def _save_project(self) -> bool:
        if self.sequence.is_empty:
            return False
        if self.project_path is None:
            return self._save_project_as()
        return self._write_project(self.project_path)

    def _save_project_as(self) -> bool:
        if self.sequence.is_empty:
            return False
        suggested = self.project_path or self.sequence.clips[0].source.path.with_suffix(PROJECT_SUFFIX)
        file_path, _ = QFileDialog.getSaveFileName(self, "Salvar projeto", str(suggested), PROJECT_FILE_FILTER)
        if not file_path:
            return False
        path = Path(file_path)
        if path.suffix.lower() != PROJECT_SUFFIX:
            path = path.with_suffix(PROJECT_SUFFIX)
        return self._write_project(path)

    def _write_project(self, path: Path) -> bool:
        self._store_active_clip_settings()
        try:
            save_project(self._current_project(), path)
        except OSError as error:
            log_event(LOG, "ui.project", level=logging.WARNING, action="save_failed", name=path.name, error=str(error))
            QMessageBox.critical(self, "Salvar projeto", f"Não foi possível salvar {path.name}: {error}")
            return False
        self.project_path = path
        self._saved_snapshot = self._project_snapshot()
        self._recent_projects.add(path)
        self._update_window_title()
        log_event(
            LOG,
            "ui.project",
            action="save",
            name=path.name,
            clips=len(self.sequence),
            missing=len(self._missing_sources),
        )
        self.status.show_message(f"Projeto salvo em {path.name}.")
        return True

    def _open_project_dialog(self) -> None:
        start_dir = str(self.project_path.parent) if self.project_path is not None else str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir projeto", start_dir, PROJECT_FILE_FILTER)
        if file_path:
            self._open_project(Path(file_path))

    def _open_project(self, path: Path) -> bool:
        """Replace the document with the project in ``path``; media that moved stays in, marked as missing."""
        if not self._confirm_discard_changes():
            return False
        try:
            project = load_project(path)
        except (OSError, ProjectError) as error:
            log_event(LOG, "ui.project", level=logging.WARNING, action="open_failed", name=path.name, error=str(error))
            QMessageBox.critical(self, "Abrir projeto", f"Não foi possível abrir {path.name}: {error}")
            if not path.is_file():
                self._recent_projects.remove(path)
            return False
        self._stop_playback()
        self._sources = {}  # fresh decoders; the worker releases the previous ones when it adopts these
        probed: dict[Path, ClipSource] = {}
        missing = set(project.missing)
        for source_path in project.sequence.source_paths:
            if source_path in missing:
                continue
            source = self._open_source(source_path, quiet=True)
            if source is None:
                missing.add(source_path)
            else:
                probed[source_path] = source
        select_combo_data(self.export_tab.thread_combo, project.threads)  # part of the saved snapshot
        self._start_document(
            with_sources(project.sequence, probed),
            playhead=project.playhead,
            clip_range=project.clip_range,
            missing=missing,
            project_path=path,
        )
        self._recent_projects.add(path)
        self._request_reference_frames()
        log_event(
            LOG,
            "ui.project",
            action="open",
            name=path.name,
            clips=len(self.sequence),
            missing=sorted(item.name for item in missing),
        )
        if missing:
            names = ", ".join(sorted(item.name for item in missing))
            self.status.show_message(
                f"Projeto {path.name} aberto. Mídia não encontrada: {names} — "
                "coloque os arquivos de volta no lugar e reabra o projeto."
            )
        else:
            self.status.show_message(
                f"Projeto {path.name} aberto: {len(self.sequence)} clipe(s), "
                f"{format_duration(self.frame_count / self.fps)}."
            )
        return True

    def _request_reference_frames(self) -> None:
        """A project stores camera-lock references as frame indices: decode the pixels again, off the GUI."""
        wanted = {
            (clip.source.path, clip.stabilization_reference_frame)
            for clip in self.sequence.clips
            if clip.stabilization_reference_frame is not None and clip.source.path not in self._missing_sources
        }
        for path, frame in sorted(wanted, key=lambda item: (str(item[0]), item[1])):
            if (path, frame) not in self._reference_frames:
                self._reference_loader.load(path, frame)

    @Slot(object, int, object)
    def _on_reference_frame_ready(self, path: Path, frame_index: int, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        sequences = [self.sequence, *(state.sequence for state in self._history.states())]
        if not any(
            clip.source.path == path and clip.stabilization_reference_frame == frame_index
            for sequence in sequences
            for clip in sequence.clips
        ):
            return  # the document moved on while the frame was being decoded
        self._reference_frames[(path, frame_index)] = frame
        clip = self.active_clip
        if (
            clip is not None
            and clip.source.path == path
            and clip.stabilization_reference_frame == frame_index
            and self.stabilization_reference_frame is None
        ):
            self.stabilization_reference_frame = frame
            self.stabilization_reference_frame_index = frame_index
            self._stabilization_epoch += 1
            self._update_stabilization_info_label()
            self._schedule_preview_refresh()

    def _confirm_discard_changes(self) -> bool:
        """Before replacing or closing the document: offer to save. ``False`` when the user cancels."""
        self._store_active_clip_settings()
        if not self.confirm_discard or not self.has_unsaved_changes:
            return True
        answer = self._ask_unsaved_changes()
        if answer == "save":
            return self._save_project()
        return answer == "discard"

    def _ask_unsaved_changes(self) -> str:
        """Modal question: ``"save"``, ``"discard"`` or ``"cancel"`` (its own method so tests can answer it)."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Alterações não salvas")
        box.setText(f"{self._document_name()} tem alterações que ainda não foram salvas.")
        box.setInformativeText("Salvar o projeto antes de continuar?")
        save = box.addButton("Salvar", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton("Não salvar", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save:
            return "save"
        if clicked is discard:
            return "discard"
        return "cancel"

    def _rebuild_recent_projects_menu(self) -> None:
        menu = self.recent_projects_menu
        menu.clear()
        paths = self._recent_projects.paths
        if not paths:
            menu.addAction("Nenhum projeto recente").setEnabled(False)
            return
        for path in paths:
            action = menu.addAction(path.name)
            action.setToolTip(str(path))
            action.setStatusTip(str(path))
            action.triggered.connect(lambda _checked=False, target=path: self._open_project(target))
        menu.addSeparator()
        menu.addAction(self.clear_recent_action)

    def _clear_recent_projects(self) -> None:
        self._recent_projects.clear()
        self._rebuild_recent_projects_menu()
