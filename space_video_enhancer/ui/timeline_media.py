"""Background decoding for the timeline tracks: filmstrip thumbnails and the audio waveform.

The filmstrip has its own thread and its own :class:`VideoSource` per file of the sequence, so a
thumbnail seek never delays the preview decoder (and a scrub never delays the thumbnails). The
GUI thread only posts requests — cells addressed by timeline frame plus the ``(file, source
frame)`` to decode — and a newer request cancels whatever is left of the previous one between
two frames. The waveform runs one ffmpeg decode per file on another thread and reports back
tagged with the path, so a result for a file that is no longer in the sequence is simply dropped.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

from space_video_enhancer.audio import AudioPeaks, load_audio_peaks
from space_video_enhancer.pipeline.geometry import fit_to_canvas
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.ui.images import frame_to_qimage
from space_video_enhancer.video import VideoOpenError, VideoSource

LOG = get_logger("timeline")


@dataclass(frozen=True)
class FilmstripCell:
    """One filmstrip cell: the timeline frame it stands for and the source frame that shows there."""

    timeline_frame: int
    source: Path
    source_frame: int


@dataclass(frozen=True)
class FilmstripRequest:
    generation: int
    cells: tuple[FilmstripCell, ...]
    width: int
    height: int
    """Cell size in device pixels; every source is fitted into it (letterboxed when the aspect differs)."""
    device_pixel_ratio: float


class FilmstripWorker(QObject):
    thumbnailReady = Signal(int, int, object)  # generation, timeline frame, QImage
    requestFinished = Signal(int, int, float)  # generation, thumbnails decoded, elapsed ms

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._pending: FilmstripRequest | None = None
        self._pending_paths: tuple[Path, ...] = ()
        self._paths_changed = False
        self._latest_generation = 0
        self._videos: dict[Path, VideoSource] = {}

    def post(self, request: FilmstripRequest) -> None:
        """GUI thread: replace whatever request is waiting; the worker checks the generation between frames."""
        with self._lock:
            self._pending = request
            self._latest_generation = request.generation

    def set_paths(self, paths: Sequence[Path]) -> None:
        """GUI thread: the files the sequence uses; the worker closes the others (and opens these lazily)."""
        with self._lock:
            self._pending_paths = tuple(dict.fromkeys(paths))
            self._paths_changed = True
            self._pending = None

    @property
    def videos(self) -> tuple[VideoSource, ...]:
        return tuple(self._videos.values())

    @property
    def video(self) -> VideoSource | None:
        """The first open decoder (the one-file case)."""
        return next(iter(self._videos.values()), None)

    def release_videos(self) -> None:
        for video in self._videos.values():
            video.release()
        self._videos.clear()

    @Slot()
    def process(self) -> None:
        while True:
            with self._lock:
                if self._paths_changed:
                    self._paths_changed = False
                    self._keep_only(self._pending_paths)
                request = self._pending
                self._pending = None
            if request is None:
                return
            self._decode(request)

    def _keep_only(self, paths: tuple[Path, ...]) -> None:
        for path in list(self._videos):
            if path not in paths:
                self._videos.pop(path).release()

    def _video_for(self, path: Path) -> VideoSource | None:
        video = self._videos.get(path)
        if video is None:
            try:
                video = VideoSource(path)
            except VideoOpenError as error:
                log_event(LOG, "filmstrip.open_failed", level=logging.WARNING, name=path.name, error=str(error))
                return None
            self._videos[path] = video
        return video

    def _decode(self, request: FilmstripRequest) -> None:
        if request.height <= 0 or request.width <= 0:
            self.requestFinished.emit(request.generation, 0, 0.0)
            return
        started = time.perf_counter()
        decoded = 0
        decode_ms = 0.0
        for cell in request.cells:
            if self._latest_generation != request.generation:
                break  # a newer request is waiting: its frames matter more than the rest of these
            video = self._video_for(cell.source)
            if video is None:
                continue
            frame = video.read_frame(cell.source_frame)
            if frame is None:
                continue
            decode_ms += video.last_decode_ms
            image = frame_to_qimage(fit_to_canvas(frame, request.width, request.height))
            image.setDevicePixelRatio(request.device_pixel_ratio)
            decoded += 1
            self.thumbnailReady.emit(request.generation, cell.timeline_frame, image)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        log_event(
            LOG,
            "filmstrip.render",
            level=logging.DEBUG,
            generation=request.generation,
            requested=len(request.cells),
            decoded=decoded,
            sources=len(self._videos),
            elapsed_ms=round(elapsed_ms, 1),
            decode_ms=round(decode_ms / decoded, 2) if decoded else None,
        )
        self.requestFinished.emit(request.generation, decoded, elapsed_ms)


class FilmstripController(QObject):
    """Owns the filmstrip thread; the timeline asks for cells and receives thumbnails one by one."""

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

    @property
    def videos(self) -> tuple[VideoSource, ...]:
        return self._worker.videos

    def set_videos(self, paths: Sequence[Path]) -> None:
        self._worker.set_paths(paths)
        self._wake.emit()

    def set_video(self, path: Path | None) -> None:
        """The one-file case of :meth:`set_videos`."""
        self.set_videos(() if path is None else (path,))

    def request(self, cells: Sequence[FilmstripCell], width: int, height: int, device_pixel_ratio: float) -> int:
        self._generation += 1
        self._worker.post(FilmstripRequest(self._generation, tuple(cells), width, height, device_pixel_ratio))
        self._wake.emit()
        return self._generation

    def stop(self, timeout_ms: int = 5_000) -> None:
        self._worker.set_paths(())
        self._thread.quit()
        self._thread.wait(timeout_ms)
        self._worker.release_videos()
        self._worker.deleteLater()


class WaveformWorker(QObject):
    peaksReady = Signal(object, object)  # Path, AudioPeaks | None

    @Slot(object, str, float)
    def load(self, path: Path, ffmpeg_path: str, buckets_per_second: float) -> None:
        started = time.perf_counter()
        peaks: AudioPeaks | None = load_audio_peaks(ffmpeg_path, path, buckets_per_second)
        log_event(
            LOG,
            "waveform.load",
            name=path.name,
            has_audio=peaks is not None,
            buckets=peaks.bucket_count if peaks is not None else 0,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
        )
        self.peaksReady.emit(path, peaks)


class WaveformController(QObject):
    """Owns the waveform thread; results carry the path they belong to."""

    peaksReady = Signal(object, object)
    _load = Signal(object, str, float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._thread.setObjectName("timeline-waveform")
        self._worker = WaveformWorker()
        self._worker.moveToThread(self._thread)
        self._load.connect(self._worker.load)  # queued on the worker thread
        self._worker.peaksReady.connect(self.peaksReady)
        self._thread.start()

    def load(self, path: Path, ffmpeg_path: str, buckets_per_second: float) -> None:
        self._load.emit(path, ffmpeg_path, buckets_per_second)

    def stop(self, timeout_ms: int = 10_000) -> None:
        self._thread.quit()
        self._thread.wait(timeout_ms)
        self._worker.deleteLater()
