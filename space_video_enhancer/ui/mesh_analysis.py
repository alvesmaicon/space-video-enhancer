"""Mesh stabilization analysis on its own thread (see :mod:`pipeline.stabilization.mesh`).

Same shape as the wobble analysis: the GUI asks for a range of one source file; the worker opens its
own decoder, tracks the frames into a :class:`MeshPath` and reports progress a few times a second.
A queue holds requests newest-first (the clip in view is analysed next); cancelling happens only when
the document changes. Results carry the request they answer, so a late one is simply dropped.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

from space_video_enhancer.pipeline.stabilization.mesh import analyze_mesh_path
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.video import VideoOpenError, VideoSource

LOG = get_logger("mesh")
PROGRESS_INTERVAL_SECONDS = 0.15


@dataclass(frozen=True)
class MeshAnalysisRequest:
    generation: int
    source: Path
    first: int
    last: int
    """Inclusive source frame indices."""

    @property
    def count(self) -> int:
        return self.last - self.first + 1

    def covers(self, source: Path, first: int, last: int) -> bool:
        return self.source == source and self.first <= first and last <= self.last


class MeshAnalysisWorker(QObject):
    progress = Signal(object, int, int)  # request, frames done, total
    finished = Signal(object, object, float)  # request, MeshPath | None, elapsed ms

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._queue: deque[MeshAnalysisRequest] = deque()
        self._cancel_floor = 0
        """Requests at or below this generation are dropped: set only when the document changes."""

    def post(self, request: MeshAnalysisRequest) -> None:
        with self._lock:
            self._queue.append(request)

    def cancel(self, floor: int) -> None:
        with self._lock:
            self._queue.clear()
            self._cancel_floor = floor

    def _next(self) -> MeshAnalysisRequest | None:
        with self._lock:
            while self._queue:
                request = self._queue.pop()  # newest first: the clip the user just moved to
                if request.generation > self._cancel_floor:
                    return request
            return None

    @Slot()
    def process(self) -> None:
        while True:
            request = self._next()
            if request is None:
                return
            self._analyze(request)

    def _analyze(self, request: MeshAnalysisRequest) -> None:
        try:
            video = VideoSource(request.source)
        except VideoOpenError as error:
            log_event(LOG, "mesh.open_failed", level=logging.WARNING, name=request.source.name, error=str(error))
            self.finished.emit(request, None, 0.0)
            return
        last_report = 0.0

        def report(done: int, total: int) -> None:
            nonlocal last_report
            now = time.monotonic()
            if done == total or now - last_report >= PROGRESS_INTERVAL_SECONDS:
                last_report = now
                self.progress.emit(request, done, total)

        try:
            analysis = analyze_mesh_path(
                video.read_frame,
                request.source,
                request.first,
                request.last,
                progress=report,
                cancelled=lambda: request.generation <= self._cancel_floor,
            )
        finally:
            video.release()
        if analysis.cancelled:
            log_event(LOG, "mesh.cancelled", level=logging.DEBUG, name=request.source.name, frames=analysis.frames_read)
            return
        track = analysis.track
        log_event(
            LOG,
            "mesh.analysis",
            name=request.source.name,
            first=request.first,
            last=request.last,
            frames=analysis.frames_read,
            measured=round(track.measured_fraction, 3) if track is not None else None,
            elapsed_ms=round(analysis.elapsed_ms, 1),
        )
        self.finished.emit(request, track, analysis.elapsed_ms)


class MeshAnalysisController(QObject):
    """Owns the analysis thread; the window asks for ranges and receives progress and tracks on the GUI thread."""

    progress = Signal(object, int, int)  # MeshAnalysisRequest, frames done, total
    finished = Signal(object, object, float)  # MeshAnalysisRequest, MeshPath | None, elapsed ms
    _wake = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._generation = 0
        self._thread = QThread(self)
        self._thread.setObjectName("mesh-analysis")
        self._worker = MeshAnalysisWorker()
        self._worker.moveToThread(self._thread)
        self._wake.connect(self._worker.process)  # cross-thread -> queued on the worker thread
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self.finished)
        self._thread.start()

    def analyze(self, source: Path, first: int, last: int) -> MeshAnalysisRequest:
        """Queue one range without disturbing a running analysis; returns the request the results will carry."""
        self._generation += 1
        request = MeshAnalysisRequest(self._generation, source, first, last)
        self._worker.post(request)
        self._wake.emit()
        return request

    def cancel(self) -> None:
        self._generation += 1
        self._worker.cancel(self._generation)

    def stop(self, timeout_ms: int = 10_000) -> None:
        self.cancel()
        self._thread.quit()
        self._thread.wait(timeout_ms)
        self._worker.deleteLater()
