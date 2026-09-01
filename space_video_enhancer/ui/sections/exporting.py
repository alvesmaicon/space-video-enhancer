"""Exporting: the summary in the Exportar tab, the ffmpeg job on its thread, progress and completion."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QUrl, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from space_video_enhancer.export.ffmpeg import FfmpegProbe, find_ffmpeg, resolve_encoder
from space_video_enhancer.export.job import ExportRequest, ExportResult
from space_video_enhancer.export.worker import ExportController
from space_video_enhancer.telemetry.log import get_logger
from space_video_enhancer.timeline import FrameRange, frame_count, kept_segments
from space_video_enhancer.ui.format import format_duration, format_timecode
from space_video_enhancer.ui.sections.state import WindowState

LOG = get_logger("ui")


def seconds_text(seconds: float) -> str:
    """Short durations with two decimals ("0.50 s"), longer ones as m:ss."""
    return f"{seconds:.2f} s" if seconds < 60 else format_duration(seconds)


class ExportSection(WindowState):
    """Exporting: the summary in the Exportar tab, the ffmpeg job on its thread, progress and completion."""

    def _update_export_summary(self) -> None:
        """What the export will contain — range, removed ranges, output — shown in the export tab."""
        if self.frame_count <= 0:
            self.export_tab.set_summary("Abra um vídeo para exportar.")
            return
        start_frame, end_frame = self._selected_frame_range()
        clip = FrameRange(start_frame, end_frame)
        exported = frame_count(kept_segments(clip, self.cuts))
        first = self.sequence.clips[0]
        width, height = first.settings.output_size(first.source.width, first.source.height)
        lines = [
            f"Trecho: {format_timecode(start_frame, self.fps)} → {format_timecode(end_frame + 1, self.fps)} "
            f"(frames {start_frame + 1}–{end_frame + 1}, {seconds_text(clip.count / self.fps)})"
        ]
        if self.cuts:
            removed = clip.count - exported
            lines.append(f"{len(self.cuts)} trecho(s) removido(s): {seconds_text(removed / self.fps)}")
        output = [
            f"{exported} frames ({seconds_text(exported / self.fps)})",
            f"{width}x{height}",
            f"{self.fps:.2f} fps",
        ]
        if len(self.sequence) > 1:
            output.append(f"{len(self.sequence)} clipes")
        lines.append("Saída: " + " · ".join(output))
        self.export_tab.set_summary("\n".join(lines))

    def _export_video(self) -> None:
        if self.sequence.is_empty or self._export_controller is not None:
            return
        self._store_active_clip_settings()
        start_frame, end_frame = self._selected_frame_range()
        export_sequence = self.sequence.for_export(start_frame, end_frame)
        if export_sequence.is_empty:
            self.status.show_message("Os cortes removem o trecho inteiro: nada a exportar.")
            return
        offline = sorted({clip.name for clip in export_sequence.clips if clip.source.path in self._missing_sources})
        if offline:
            QMessageBox.warning(
                self,
                "Mídia não encontrada",
                f"Não dá para exportar: {', '.join(offline)} não está onde o projeto esperava. "
                "Coloque o arquivo de volta no lugar e reabra o projeto.",
            )
            return

        first_path = self.sequence.clips[0].source.path
        default_name = first_path.with_stem(f"{first_path.stem}_enhanced").with_suffix(".mp4")
        output_path, _ = QFileDialog.getSaveFileName(self, "Salvar vídeo processado", str(default_name), "MP4 (*.mp4)")
        if not output_path:
            return

        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path is None:
            QMessageBox.warning(
                self,
                "FFmpeg não encontrado",
                "Instale o FFmpeg e deixe o executável disponível no PATH para exportar e muxar o áudio do vídeo.",
            )
            return

        if self._ffmpeg_probe is None or self._ffmpeg_probe.ffmpeg_path != ffmpeg_path:
            self._ffmpeg_probe = FfmpegProbe(ffmpeg_path)
        encoder = resolve_encoder(self._ffmpeg_probe)
        self._backend.reset()

        request = ExportRequest(
            export_sequence,
            Path(output_path),
            self.export_tab.worker_count(),
            ffmpeg_path,
            encoder,
            wobble_tracks=tuple(self._wobble_tracks),  # clips these do not cover are analysed by the job itself
            mesh_tracks=tuple(self._mesh_tracks),
        )
        self._export_backend_label = f"{self._backend.label} + {encoder.label}"
        self._export_output_path = Path(output_path)
        self._export_started_at = time.monotonic()
        note = ""
        if request.effective_worker_count < request.worker_count:
            note = " · estabilização temporal usa processamento sequencial"

        controller = ExportController(request, self._backend, self)
        controller.progress.connect(self._on_export_progress)
        controller.status.connect(self._on_export_status)
        controller.finished.connect(self._on_export_finished)
        self._export_controller = controller
        self._begin_export_ui(
            f"Exportando {request.expected_frames} frames com {request.effective_worker_count} thread(s) · "
            f"{self._export_backend_label}{note}"
        )
        controller.start()

    def _begin_export_ui(self, message: str) -> None:
        self._stop_playback()
        self._set_controls_enabled(False)
        self.open_action.setEnabled(False)
        self.add_video_action.setEnabled(False)
        self.open_project_action.setEnabled(False)
        self.status.begin_export(message)

    def _end_export_ui(self) -> None:
        self._set_controls_enabled(True)
        self.open_action.setEnabled(True)
        self.add_video_action.setEnabled(True)
        self.open_project_action.setEnabled(True)

    @Slot(int, int)
    def _on_export_progress(self, processed_frames: int, total_frames: int) -> None:
        percent = int((processed_frames / max(1, total_frames)) * 100)
        elapsed = time.monotonic() - self._export_started_at
        fps = processed_frames / elapsed if elapsed > 0 else 0.0
        remaining = f" · restam {format_duration((total_frames - processed_frames) / fps)}" if fps > 0 else ""
        self.status.set_export_progress(
            percent,
            f"Exportando {percent}% · frame {processed_frames}/{total_frames} · {fps:.0f} fps{remaining} · "
            f"{self._export_backend_label}",
        )

    @Slot(str)
    def _on_export_status(self, message: str) -> None:
        """A phase before the first frame is written (the wobble analysis of a clip)."""
        self.status.set_export_progress(0, f"{message} · {self._export_backend_label}")

    @Slot(object)
    def _on_export_finished(self, result: ExportResult) -> None:
        if self._export_controller is not None:
            self._export_controller.deleteLater()
            self._export_controller = None
        self._end_export_ui()
        self._counters["exports"] += 1
        if not result.success and not result.cancelled:
            self._counters["export_failures"] += 1

        if result.success:
            elapsed = format_duration(time.monotonic() - self._export_started_at)
            name = self._export_output_path.name if self._export_output_path else "vídeo"
            message = f"Vídeo exportado com sucesso: {name} ({elapsed})."
            if result.frame_count_mismatch:
                message += f" {result.message}"
            self.status.end_export(message, success=True)
            return

        if result.cancelled:
            self.status.end_export(result.message, success=False)
            return
        self.status.end_export("A exportação falhou. Revise os parâmetros e tente novamente.", success=False)
        QMessageBox.warning(self, "Falha na exportação", result.message)

    def _request_export_cancel(self) -> None:
        if self._export_controller is None:
            return
        self._export_controller.request_cancel()
        self.status.set_cancel_enabled(False)
        self.status.show_message("Cancelando exportação…")

    def _reveal_export(self) -> None:
        if self._export_output_path is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._export_output_path.parent)))

    def _sync_export_button(self) -> None:
        """The button in the export tab mirrors the export action: on with a sequence, off while exporting."""
        self.export_tab.export_button.setEnabled(self.export_action.isEnabled())
