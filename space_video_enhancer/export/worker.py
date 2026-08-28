"""Qt adapter that runs :func:`run_export` on a ``QThread`` and forwards progress as signals."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QThread, Signal, Slot

from space_video_enhancer.export.job import ExportRequest, ExportResult, run_export
from space_video_enhancer.pipeline.filters import OpenCLBackend


class ExportWorker(QObject):
    progress = Signal(int, int)
    status = Signal(str)  # phases before the first frame (wobble analysis)
    finished = Signal(object)  # ExportResult

    def __init__(self, request: ExportRequest, backend: OpenCLBackend | None) -> None:
        super().__init__()
        self._request = request
        self._backend = backend
        self._cancel = threading.Event()

    @Slot()
    def run(self) -> None:
        result = run_export(
            self._request,
            progress=self.progress.emit,
            cancel=self._cancel,
            backend=self._backend,
            status=self.status.emit,
        )
        self.finished.emit(result)

    def request_cancel(self) -> None:
        self._cancel.set()


class ExportController(QObject):
    """Owns the thread/worker pair for one export and tears it down when done."""

    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)  # ExportResult

    def __init__(self, request: ExportRequest, backend: OpenCLBackend | None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._worker = ExportWorker(request, backend)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress)
        self._worker.status.connect(self.status)
        self._worker.finished.connect(self._on_finished)

    def start(self) -> None:
        self._thread.start()

    def request_cancel(self) -> None:
        self._worker.request_cancel()

    def is_running(self) -> bool:
        return self._thread.isRunning()

    def stop(self, timeout_ms: int = 10_000) -> None:
        """Cancel and block until the worker thread exits (used when the window closes mid-export)."""
        self._worker.request_cancel()
        self._thread.quit()
        self._thread.wait(timeout_ms)

    @Slot(object)
    def _on_finished(self, result: ExportResult) -> None:
        self._thread.quit()
        self._thread.wait()
        self._worker.deleteLater()
        self.finished.emit(result)
