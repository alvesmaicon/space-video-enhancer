"""Camera-path analysis on its own thread, to fill manual keyframes (see :mod:`pipeline.stabilization.motion_path`).

The GUI asks for one range of a source file when the user clicks "Preencher"; the worker opens its own
decoder (the preview's must not be seeked from another thread), tracks the frames and reports progress a
few times a second. A newer request or a document change cancels the running one between two frames, and
results carry the request they answer, so a late one for a clip that is gone is dropped by the receiver.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

from space_video_enhancer.pipeline.stabilization.motion_path import analyze_motion_path
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.video import VideoOpenError, VideoSource

LOG = get_logger("motion_path")
PROGRESS_INTERVAL_SECONDS = 0.15


@dataclass(frozen=True)
class MotionPathRequest:
    generation: int
    source: Path
    first: int
    last: int
    """Inclusive source frame indices."""

    @property
    def count(self) -> int:
        return self.last - self.first + 1


class MotionPathWorker(QObject):
    progress = Signal(object, int, int)  # request, frames done, total
    finished = Signal(object, object, float)  # request, MotionPath | None, elapsed ms

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._queue: deque[MotionPathRequest] = deque()
        self._cancel_floor = 0
        """Requests at or below this generation are dropped (a newer request or a document change)."""

    def post(self, request: MotionPathRequest) -> None:
        with self._lock:
            self._queue.append(request)

    def cancel(self, floor: int) -> None:
        with self._lock:
            self._queue.clear()
            self._cancel_floor = floor

    def _next(self) -> MotionPathRequest | None:
        with self._lock:
            while self._queue:
                request = self._queue.pop()  # newest first
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

    def _analyze(self, request: MotionPathRequest) -> None:
        try:
            video = VideoSource(request.source)
        except VideoOpenError as error:
            log_event(LOG, "motion_path.open_failed", level=logging.WARNING, name=request.source.name, error=str(error))
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
            analysis = analyze_motion_path(
                video.read_frame,
                request.first,
                request.last,
                progress=report,
                cancelled=lambda: request.generation <= self._cancel_floor,
            )
        finally:
            video.release()
        if analysis.cancelled:
            log_event(LOG, "motion_path.cancelled", level=logging.DEBUG, name=request.source.name)
            return
        log_event(
            LOG,
            "motion_path.analysis",
            name=request.source.name,
            first=request.first,
            last=request.last,
            frames=analysis.frames_read,
            measured=round(analysis.track.measured_fraction, 3) if analysis.track is not None else None,
            elapsed_ms=round(analysis.elapsed_ms, 1),
        )
        self.finished.emit(request, analysis.track, analysis.elapsed_ms)


class MotionPathController(QObject):
    """Owns the analysis thread; the window asks for one range and receives progress and a path on the GUI thread."""

    progress = Signal(object, int, int)  # MotionPathRequest, frames done, total
    finished = Signal(object, object, float)  # MotionPathRequest, MotionPath | None, elapsed ms
    _wake = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._generation = 0
        self._thread = QThread(self)
        self._thread.setObjectName("motion-path-analysis")
        self._worker = MotionPathWorker()
        self._worker.moveToThread(self._thread)
        self._wake.connect(self._worker.process)  # cross-thread -> queued on the worker thread
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self.finished)
        self._thread.start()

    def analyze(self, source: Path, first: int, last: int) -> MotionPathRequest:
        """Queue one range and cancel any earlier one; returns the request the results will carry."""
        self._generation += 1
        self._worker.cancel(self._generation - 1)  # a new fill replaces a pending one
        request = MotionPathRequest(self._generation, source, first, last)
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
