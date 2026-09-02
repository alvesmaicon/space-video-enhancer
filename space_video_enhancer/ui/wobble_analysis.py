"""Rolling-shutter wobble analysis on its own thread (see :mod:`pipeline.stabilization.wobble`).

The GUI asks for a range of one source file; the worker opens its own decoder (the preview's must
not be seeked from another thread), tracks the frames and reports progress a few times a second. A
newer request cancels the running one between two frames. Results carry the request they answer,
so a late one for a clip that is gone is simply dropped by the receiver.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

from space_video_enhancer.pipeline.stabilization.mesh import MeshCorrections
from space_video_enhancer.pipeline.stabilization.stabilize import mesh_stabilized_for_analysis
from space_video_enhancer.pipeline.stabilization.wobble import analyze_wobble
from space_video_enhancer.settings import StabilizationSettings
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.video import VideoOpenError, VideoSource

LOG = get_logger("wobble")
PROGRESS_INTERVAL_SECONDS = 0.15


@dataclass(frozen=True)
class WobbleAnalysisRequest:
    generation: int
    source: Path
    first: int
    last: int
    """Inclusive source frame indices."""
    window: int
    corrections: MeshCorrections | None = None
    """When given, the analysis is *residual*: each frame is warped by these mesh fields (with
    ``stabilization``'s margin and intensity) before tracking, so the track holds only what the mesh
    did not absorb."""
    stabilization: StabilizationSettings | None = None

    @property
    def count(self) -> int:
        return self.last - self.first + 1

    @property
    def residual(self) -> bool:
        return self.corrections is not None

    def covers(self, source: Path, first: int, last: int) -> bool:
        return self.source == source and self.first <= first and last <= self.last


class WobbleAnalysisWorker(QObject):
    progress = Signal(object, int, int)  # request, frames done, total
    finished = Signal(object, object, float)  # request, WobbleTrack | None, elapsed ms

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._queue: deque[WobbleAnalysisRequest] = deque()
        """Waiting analyses, oldest at the left; the newest is taken first so the clip in view is next."""
        self._cancel_floor = 0
        """Requests at or below this generation are dropped: set only when the document changes."""

    def post(self, request: WobbleAnalysisRequest) -> None:
        """GUI thread: queue one analysis. Switching clips never cancels the running one — it just adds work."""
        with self._lock:
            self._queue.append(request)

    def cancel(self, floor: int) -> None:
        """GUI thread: drop every queued analysis and stop the running one (``floor`` = clear up to this generation)."""
        with self._lock:
            self._queue.clear()
            self._cancel_floor = floor

    def _next(self) -> WobbleAnalysisRequest | None:
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

    def _analyze(self, request: WobbleAnalysisRequest) -> None:
        try:
            video = VideoSource(request.source)
        except VideoOpenError as error:
            log_event(LOG, "wobble.open_failed", level=logging.WARNING, name=request.source.name, error=str(error))
            self.finished.emit(request, None, 0.0)
            return
        last_report = 0.0

        def report(done: int, total: int) -> None:
            nonlocal last_report
            now = time.monotonic()
            if done == total or now - last_report >= PROGRESS_INTERVAL_SECONDS:
                last_report = now
                self.progress.emit(request, done, total)

        def read(frame_index: int) -> object:
            frame = video.read_frame(frame_index)
            if frame is None or request.corrections is None or request.stabilization is None:
                return frame
            return mesh_stabilized_for_analysis(frame, request.stabilization, frame_index, request.corrections)

        try:
            analysis = analyze_wobble(
                read,  # type: ignore[arg-type]
                request.source,
                request.first,
                request.last,
                request.window,
                rotation=video.rotation,
                residual=request.residual,
                progress=report,
                cancelled=lambda: request.generation <= self._cancel_floor,
            )
        finally:
            video.release()
        if analysis.cancelled:
            log_event(
                LOG,
                "wobble.cancelled",
                level=logging.DEBUG,
                name=request.source.name,
                frames=analysis.frames_read,
                elapsed_ms=round(analysis.elapsed_ms, 1),
            )
            return
        track = analysis.track
        log_event(
            LOG,
            "wobble.analysis",
            name=request.source.name,
            residual=request.residual,
            first=request.first,
            last=request.last,
            frames=analysis.frames_read,
            measured=round(track.measured_fraction, 3) if track is not None else None,
            axis=track.detected_axis.value if track is not None else None,
            rotation=video.rotation,
            elapsed_ms=round(analysis.elapsed_ms, 1),
        )
        self.finished.emit(request, track, analysis.elapsed_ms)


class WobbleAnalysisController(QObject):
    """Owns the analysis thread; the window asks for ranges and receives progress and tracks on the GUI thread."""

    progress = Signal(object, int, int)  # WobbleAnalysisRequest, frames done, total
    finished = Signal(object, object, float)  # WobbleAnalysisRequest, WobbleTrack | None, elapsed ms
    _wake = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._generation = 0
        self._thread = QThread(self)
        self._thread.setObjectName("wobble-analysis")
        self._worker = WobbleAnalysisWorker()
        self._worker.moveToThread(self._thread)
        self._wake.connect(self._worker.process)  # cross-thread -> queued on the worker thread
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self.finished)
        self._thread.start()

    def analyze(
        self,
        source: Path,
        first: int,
        last: int,
        window: int,
        corrections: MeshCorrections | None = None,
        stabilization: StabilizationSettings | None = None,
    ) -> WobbleAnalysisRequest:
        """Queue one range without disturbing a running analysis; returns the request the results will carry.

        With ``corrections`` (and the ``stabilization`` they were built for) the analysis is residual:
        see :class:`WobbleAnalysisRequest`.
        """
        self._generation += 1
        request = WobbleAnalysisRequest(self._generation, source, first, last, window, corrections, stabilization)
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
