"""Background decoding for the timeline tracks: filmstrip thumbnails.

The filmstrip has its own thread and its own :class:`VideoSource`, so a thumbnail seek never
delays the preview decoder (and a scrub never delays the thumbnails). The GUI thread only posts
requests; a newer request cancels whatever is left of the previous one between two frames.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, QThread, Signal, Slot

from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.ui.images import frame_to_qimage
from space_video_enhancer.video import VideoOpenError, VideoSource

LOG = get_logger("timeline")


@dataclass(frozen=True)
class FilmstripRequest:
    generation: int
    frames: tuple[int, ...]
    height: int
    """Thumbnail height in device pixels."""
    device_pixel_ratio: float


class FilmstripWorker(QObject):
    thumbnailReady = Signal(int, int, object)  # generation, frame index, QImage
    requestFinished = Signal(int, int, float)  # generation, thumbnails decoded, elapsed ms

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._pending: FilmstripRequest | None = None
        self._pending_path: Path | None = None
        self._path_changed = False
        self._latest_generation = 0
        self._video: VideoSource | None = None

    def post(self, request: FilmstripRequest) -> None:
        """GUI thread: replace whatever request is waiting; the worker checks the generation between frames."""
        with self._lock:
            self._pending = request
            self._latest_generation = request.generation

    def set_path(self, path: Path | None) -> None:
        with self._lock:
            self._pending_path = path
            self._path_changed = True
            self._pending = None

    @property
    def video(self) -> VideoSource | None:
        return self._video

    def release_video(self) -> None:
        if self._video is not None:
            self._video.release()
            self._video = None

    @Slot()
    def process(self) -> None:
        while True:
            with self._lock:
                if self._path_changed:
                    self._path_changed = False
                    path = self._pending_path
                    self.release_video()
                    self._open(path)
                request = self._pending
                self._pending = None
            if request is None:
                return
            self._decode(request)

    def _open(self, path: Path | None) -> None:
        if path is None:
            return
        try:
            self._video = VideoSource(path)
        except VideoOpenError as error:
            log_event(LOG, "filmstrip.open_failed", level=logging.WARNING, name=path.name, error=str(error))

    def _decode(self, request: FilmstripRequest) -> None:
        video = self._video
        if video is None or request.height <= 0:
            self.requestFinished.emit(request.generation, 0, 0.0)
            return
        started = time.perf_counter()
        decoded = 0
        for frame_index in request.frames:
            if self._latest_generation != request.generation:
                break  # a newer request is waiting: its frames matter more than the rest of these
            frame = video.read_frame(frame_index)
            if frame is None:
                continue
            height = request.height
            width = max(1, round(height * frame.shape[1] / frame.shape[0]))
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            image = frame_to_qimage(small)
            image.setDevicePixelRatio(request.device_pixel_ratio)
            decoded += 1
            self.thumbnailReady.emit(request.generation, frame_index, image)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        log_event(
            LOG,
            "filmstrip.render",
            level=logging.DEBUG,
            generation=request.generation,
            requested=len(request.frames),
            decoded=decoded,
            elapsed_ms=round(elapsed_ms, 1),
            decode_ms=video.decode_stats.summary(),
        )
        self.requestFinished.emit(request.generation, decoded, elapsed_ms)


class FilmstripController(QObject):
    """Owns the filmstrip thread; the timeline asks for frames and receives thumbnails one by one."""

    thumbnailReady = Signal(int, int, object)
    requestFinished = Signal(int, int, float)
    _wake = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._generation = 0
        self._thread = QThread(self)
        self._thread.setObjectName("timeline-filmstrip")
        self._worker = FilmstripWorker()
        self._worker.moveToThread(self._thread)
        self._wake.connect(self._worker.process)  # cross-thread -> queued on the worker thread
        self._worker.thumbnailReady.connect(self.thumbnailReady)
        self._worker.requestFinished.connect(self.requestFinished)
        self._thread.start()

    @property
    def video(self) -> VideoSource | None:
        return self._worker.video

    def set_video(self, path: Path | None) -> None:
        self._worker.set_path(path)
        self._wake.emit()

    def request(self, frames: tuple[int, ...], height: int, device_pixel_ratio: float) -> int:
        self._generation += 1
        self._worker.post(FilmstripRequest(self._generation, tuple(frames), height, device_pixel_ratio))
        self._wake.emit()
        return self._generation

    def stop(self, timeout_ms: int = 5_000) -> None:
        self._worker.set_path(None)
        self._thread.quit()
        self._thread.wait(timeout_ms)
        self._worker.release_video()
        self._worker.deleteLater()
