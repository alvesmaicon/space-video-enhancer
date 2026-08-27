"""Undo/redo over the editing state, the export range (I/O markers) and the cuts inside the clips."""

from __future__ import annotations

import logging
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

from space_video_enhancer.sequence import Sequence as ClipSequence
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.ui.format import format_timecode
from space_video_enhancer.ui.sections.state import WindowState

LOG = get_logger("ui")


@dataclass(frozen=True)
class EditState:
    """Everything undo/redo restores: the clips (settings, cuts, references included) and the export range."""

    sequence: ClipSequence
    clip_start_frame: int
    clip_end_frame: int


SETTINGS_SECTION_NAMES = {
    "filters": "ajustes de imagem",
    "image_filter": "filtro",
    "geometry": "geometria",
    "crop": "crop",
    "stabilization": "estabilização",
    "denoise_sample_rect": "amostra de ruído",
    "grade": "HSL",
    "exposure": "exposição",
}
"""Undo labels for a change confined to one section of ``RenderSettings`` (keys of ``diff_settings``)."""


def settings_edit_label(changes: dict[str, Any]) -> str:
    """Undo label: "ajuste de estabilização" for a change inside one section, "ajustes" when several changed."""
    sections = {key.split(".", 1)[0] for key in changes}
    if len(sections) == 1:
        name = SETTINGS_SECTION_NAMES.get(next(iter(sections)))
        if name is not None:
            return name if name.startswith("ajuste") else f"ajuste de {name}"
    return "ajustes"


class EditingSection(WindowState):
    """Undo/redo over the editing state, the export range (I/O markers) and the cuts inside the clips."""

    def _edit_state(self) -> EditState:
        return EditState(self.sequence, self.clip_start_frame, self.clip_end_frame)

    def _commit_edit(self, label: str, key: Hashable | None = None) -> None:
        """Record the state after the edit ``label``; consecutive edits with the same ``key`` form one step."""
        if self._history.commit(self._edit_state(), label, key):
            log_event(LOG, "ui.edit", level=logging.DEBUG, label=label, steps=len(self._history))
        self._update_history_actions()

    def _update_history_actions(self) -> None:
        editable = self._export_controller is None
        undo_label, redo_label = self._history.undo_label, self._history.redo_label
        self.undo_action.setEnabled(editable and undo_label is not None)
        self.undo_action.setText("Desfazer" if undo_label is None else f"Desfazer {undo_label}")
        self.redo_action.setEnabled(editable and redo_label is not None)
        self.redo_action.setText("Refazer" if redo_label is None else f"Refazer {redo_label}")
        self._update_window_title()

    def _undo_edit(self) -> None:
        if self._export_controller is not None:
            return
        self._store_active_clip_settings()  # widgets not yet rendered still count as the last edit
        step = self._history.undo()
        if step is None:
            return
        state, label = step
        if not self._restore_edit_state(state):
            self._history.redo()  # a file of that state is gone: leave things as they were
            self._update_history_actions()
            return
        log_event(LOG, "ui.undo", label=label, steps=len(self._history))
        self.status.show_message(f"Desfeito: {label}.")
        self._update_history_actions()

    def _redo_edit(self) -> None:
        if self._export_controller is not None:
            return
        step = self._history.redo()
        if step is None:
            return
        state, label = step
        if not self._restore_edit_state(state):
            self._history.undo()
            self._update_history_actions()
            return
        log_event(LOG, "ui.redo", label=label, steps=len(self._history))
        self.status.show_message(f"Refeito: {label}.")
        self._update_history_actions()

    def _restore_edit_state(self, state: EditState) -> bool:
        """Show ``state`` on the timeline and in the tabs; ``False`` when one of its files cannot be opened.

        The playhead stays where it is (clamped to the restored length). Decoders of files no
        reachable state uses are released; a file that left the sequence and comes back is reopened.
        """
        self._stop_playback()
        for path in state.sequence.source_paths:
            if path in self._missing_sources or path in self._sources:
                continue
            if self._open_source(path) is None:
                return False
        self.active_clip_index = -1  # the tabs reload from the restored clip under the playhead
        self.sequence = state.sequence
        self.clip_start_frame, self.clip_end_frame = state.clip_start_frame, state.clip_end_frame
        self._pending_cut_start = None
        referenced = {path for snapshot in self._history.states() for path in snapshot.sequence.source_paths}
        self._sources = {path: video for path, video in self._sources.items() if path in referenced}
        self._preview_controller.set_videos(tuple(self._sources.values()))  # the worker releases the dropped ones
        self._prune_reference_frames()
        self._stabilization_epoch += 1
        self._sequence_changed()
        self.transport.set_value_silently(self.current_frame_index)
        self._load_frame(self.current_frame_index)
        return True

    def _mark_clip_start(self) -> None:
        if self.frame_count <= 0:
            return
        self.clip_start_frame = self.current_frame_index
        self.clip_end_frame = max(self.clip_end_frame, self.clip_start_frame)
        log_event(LOG, "ui.clip_range", start=self.clip_start_frame, end=self.clip_end_frame, edge="start")
        self._commit_edit("marcar início")
        self._update_timeline_markers()
        self._update_export_summary()

    def _mark_clip_end(self) -> None:
        if self.frame_count <= 0:
            return
        self.clip_end_frame = self.current_frame_index
        self.clip_start_frame = min(self.clip_start_frame, self.clip_end_frame)
        log_event(LOG, "ui.clip_range", start=self.clip_start_frame, end=self.clip_end_frame, edge="end")
        self._commit_edit("marcar fim")
        self._update_timeline_markers()
        self._update_export_summary()

    def _reset_clip_range(self) -> None:
        if self.frame_count <= 0:
            return
        self.clip_start_frame = 0
        self.clip_end_frame = self.frame_count - 1
        self._set_timeline_cuts([], label="redefinir corte temporal")
        self._pending_cut_start = None
        self._update_timeline_markers()
        self._update_export_summary()

    def _set_timeline_cuts(self, cuts: Sequence[tuple[int, int]], *, label: str, key: Hashable | None = None) -> None:
        """Replace every clip's cuts from ranges in timeline frames (normalized by the model); one undo step."""
        self.sequence = self.sequence.with_timeline_cuts(cuts)
        self._commit_edit(label, key)

    def _toggle_cut_mark(self) -> None:
        """First press marks where a removed range starts, second press where it ends."""
        if self.frame_count <= 0:
            return
        if self._pending_cut_start is None:
            self._pending_cut_start = self.current_frame_index
            self.status.show_message(
                f"Início do corte em {format_timecode(self.current_frame_index, self.fps)} — "
                "vá até o fim do trecho e pressione X (ou o mesmo botão) de novo."
            )
        else:
            start, self._pending_cut_start = self._pending_cut_start, None
            self._add_cut(start, self.current_frame_index)
        self._update_timeline_markers()
        self._update_export_summary()

    def _add_cut(self, start: int, end: int) -> None:
        self._set_timeline_cuts([*self.cuts, (start, end)], label="adicionar corte")
        log_event(LOG, "ui.cuts", action="add", start=min(start, end), end=max(start, end), cuts=self.cuts)
        self._update_timeline_markers()
        self._update_export_summary()
        self.status.show_message(
            f"Trecho {format_timecode(min(start, end), self.fps)} → {format_timecode(max(start, end) + 1, self.fps)} "
            "removido da exportação."
        )

    def _remove_cut(self, index: int) -> None:
        cuts = self.cuts
        if 0 <= index < len(cuts):
            removed = cuts.pop(index)
            self._set_timeline_cuts(cuts, label="remover corte")
            log_event(LOG, "ui.cuts", action="remove", start=removed[0], end=removed[1], cuts=self.cuts)
            self._update_timeline_markers()
            self._update_export_summary()

    def _remove_selected_cut(self) -> None:
        """Delete/Backspace: the cut selected on the timeline."""
        index = self.transport.timeline.selected_cut()
        if index >= 0:
            self._remove_cut(index)

    def _set_cuts_from_timeline(self, cuts: Sequence[tuple[int, int]], finished: bool) -> None:
        """A cut edge or body dragged on the timeline; logged once, when the drag ends."""
        if self.frame_count <= 0:
            return
        self._set_timeline_cuts(cuts, label="ajustar corte", key="cut_drag")  # the whole drag is one undo step
        if finished:
            self._history.seal()
            log_event(LOG, "ui.cuts", action="drag", cuts=self.cuts)
        self._update_timeline_markers()
        self._update_export_summary()

    def _set_clip_range_from_timeline(self, start: int, end: int, finished: bool) -> None:
        if self.frame_count <= 0:
            return
        self.clip_start_frame, self.clip_end_frame = start, end
        self._commit_edit("mover marcador", "range_drag")
        if finished:
            self._history.seal()
            log_event(LOG, "ui.clip_range", start=start, end=end, edge="drag")
        self._update_timeline_markers()
        self._update_export_summary()

    def _clear_cuts(self) -> None:
        if not self.cuts and self._pending_cut_start is None:
            return
        self._set_timeline_cuts([], label="limpar cortes")
        self._pending_cut_start = None
        log_event(LOG, "ui.cuts", action="clear", cuts=[])
        self._update_timeline_markers()
        self._update_export_summary()

    def _selected_frame_range(self) -> tuple[int, int]:
        if self.frame_count <= 0:
            return (0, 0)
        start_frame = max(0, min(self.clip_start_frame, self.frame_count - 1))
        end_frame = max(0, min(self.clip_end_frame, self.frame_count - 1))
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame
        return (start_frame, end_frame)

    def _update_timeline_markers(self) -> None:
        self.transport.set_cuts(self.cuts)
        self.transport.set_pending_cut(self._pending_cut_start)
        if self.frame_count <= 0:
            self.transport.set_markers(0, 0, False)
            return
        start_frame, end_frame = self._selected_frame_range()
        self.transport.set_markers(start_frame, end_frame, True)
