"""Qt adapter that decodes and renders preview/thumbnail requests on a ``QThread``.

The worker owns the open :class:`VideoSource` objects (one per file of the sequence): the UI
hands them over with :meth:`PreviewController.set_videos` and from then on only asks for
``(file, frame index)``, so no decoding ever happens on the GUI thread. Every preview render is logged as a
``preview.render`` event with decode time, stage timings and pipeline notes; playback
frames go to DEBUG so an INFO log stays readable.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QSize, QThread, Signal, Slot
from PySide6.QtGui import QImage

from space_video_enhancer.pipeline.filters import OpenCLBackend
from space_video_enhancer.pipeline.preview import (
    FrameUnavailableError,
    PreviewRequest,
    RenderMailbox,
    ThumbnailRequest,
    render_preview,
    render_thumbnails,
)
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.ui.images import frame_to_qimage, thumbnail_qimage
from space_video_enhancer.video import VideoSource

LOG = get_logger("preview")


@dataclass(frozen=True)
class PreviewImages:
    generation: int
    frame_index: int
    source: np.ndarray
    """The decoded source frame (full resolution) the UI keeps for reference captures and thumbnails."""
    before: QImage
    after: QImage
    elapsed_ms: float
    queue_wait_ms: float
    reason: str
    trace: dict[str, Any]
    """``RenderTrace.as_dict()``: total_ms, stages and pipeline notes (for the HUD)."""
    decode_ms: float
    decode_sequential: bool | None
    video_seeks: int
    video_sequential_reads: int
    source_path: Path | None = None
    """File ``frame_index`` belongs to (a sequence shows frames of several files)."""


@dataclass(frozen=True)
class ThumbnailImages:
    generation: int
    images: dict[str, QImage]
    elapsed_ms: float
    device_pixel_ratio: float


class PreviewWorker(QObject):
    previewReady = Signal(object)  # PreviewImages
    thumbnailsReady = Signal(object)  # ThumbnailImages
    renderFailed = Signal(str)

    def __init__(self, mailbox: RenderMailbox, backend: OpenCLBackend | None) -> None:
        super().__init__()
        self._mailbox = mailbox
        self._backend = backend
        self._state = StabilizationState()
        self._epoch: int | None = None
        self._videos: tuple[VideoSource, ...] = ()
        self._pending_videos: tuple[VideoSource, ...] = ()
        self._videos_pending = False
        self._video_lock = threading.Lock()

    @property
    def videos(self) -> tuple[VideoSource, ...]:
        return self._videos

    @property
    def video(self) -> VideoSource | None:
        """The first adopted decoder (the one-video case)."""
        return self._videos[0] if self._videos else None

    def set_videos(self, videos: Sequence[VideoSource]) -> None:
        """Called from the GUI thread; the swap (and the release of dropped videos) happens on the worker."""
        with self._video_lock:
            self._pending_videos = tuple(videos)
            self._videos_pending = True

    def release_videos(self) -> None:
        """Release whatever is open. Only safe once the worker thread has stopped."""
        with self._video_lock:
            pending, self._pending_videos, self._videos_pending = self._pending_videos, (), False
        released: list[int] = []
        for video in (*self._videos, *pending):
            if id(video) not in released:
                released.append(id(video))
                video.release()
        self._videos = ()

    def _video_for(self, path: Path | None) -> VideoSource | None:
        if path is None:
            return self.video
        for video in self._videos:
            if video.path == path:
                return video
        return None

    @Slot()
    def process(self) -> None:
        """Drain the mailbox. Extra wake-ups find it empty and return immediately."""
        while True:
            self._adopt_pending_video()
            job = self._mailbox.take()
            if job is None:
                return
            try:
                if isinstance(job, PreviewRequest):
                    self._render_preview(job)
                else:
                    self._render_thumbnails(job)
            except FrameUnavailableError as error:
                log_event(
                    LOG,
                    "preview.frame_unavailable",
                    level=logging.WARNING,
                    generation=job.generation,
                    frame_index=getattr(job, "frame_index", None),
                    error=str(error),
                )
                self.renderFailed.emit(str(error))
            except Exception as error:  # keep the worker alive, report to the UI and the log
                log_event(
                    LOG,
                    "preview.error",
                    level=logging.ERROR,
                    exc_info=True,
                    kind=type(job).__name__,
                    generation=job.generation,
                    settings=asdict(job.settings),
                )
                self.renderFailed.emit(f"{type(error).__name__}: {error}")

    def _adopt_pending_video(self) -> None:
        with self._video_lock:
            if not self._videos_pending:
                return
            videos, self._pending_videos, self._videos_pending = self._pending_videos, (), False
        previous, self._videos = self._videos, videos
        kept = {id(video) for video in videos}
        for video in previous:
            if id(video) not in kept:
                video.release()

    def _render_preview(self, job: PreviewRequest) -> None:
        if job.stabilization_epoch != self._epoch:
            self._state.reset()
            self._epoch = job.stabilization_epoch
        video = self._video_for(job.source_path)
        if video is None and job.frame is None:
            name = job.source_path.name if job.source_path is not None else "?"
            raise FrameUnavailableError(f"o vídeo {name} não está aberto no worker de preview")
        result = render_preview(job, self._state, self._backend, video)
        trace = result.trace.as_dict()
        log_event(
            LOG,
            "preview.render",
            level=logging.DEBUG if job.reason == "playback" else logging.INFO,
            generation=job.generation,
            frame_index=job.frame_index,
            source=job.source_path.name if job.source_path is not None else None,
            reason=job.reason,
            queue_wait_ms=round(result.queue_wait_ms, 1),
            dropped_total=self._mailbox.dropped,
            decode_ms=round(result.decode_ms, 2),
            decode_sequential=result.decode_sequential,
            **trace,
        )
        self.previewReady.emit(
            PreviewImages(
                result.generation,
                result.frame_index,
                result.source,
                frame_to_qimage(result.before),
                frame_to_qimage(result.after),
                result.elapsed_ms,
                result.queue_wait_ms,
                job.reason,
                trace,
                result.decode_ms,
                result.decode_sequential,
                result.video_seeks,
                result.video_sequential_reads,
                job.source_path if job.source_path is not None else (video.path if video is not None else None),
            )
        )

    def _render_thumbnails(self, job: ThumbnailRequest) -> None:
        result = render_thumbnails(job, self._backend)
        icon_size = QSize(*job.icon_size)
        images = {preset_id: thumbnail_qimage(frame, icon_size) for preset_id, frame in result.frames.items()}
        log_event(
            LOG,
            "thumbnails.render",
            generation=job.generation,
            count=len(images),
            elapsed_ms=round(result.elapsed_ms, 1),
            device_pixel_ratio=job.device_pixel_ratio,
        )
        self.thumbnailsReady.emit(ThumbnailImages(result.generation, images, result.elapsed_ms, job.device_pixel_ratio))


class PreviewController(QObject):
    """Owns the render thread; the UI submits requests and listens to the ready signals."""

    previewReady = Signal(object)  # PreviewImages
    thumbnailsReady = Signal(object)  # ThumbnailImages
    renderFailed = Signal(str)
    _wake = Signal()

    def __init__(self, backend: OpenCLBackend | None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.mailbox = RenderMailbox()
        self._thread = QThread(self)
        self._thread.setObjectName("preview-render")
        self._worker = PreviewWorker(self.mailbox, backend)
        self._worker.moveToThread(self._thread)
        self._wake.connect(self._worker.process)  # cross-thread -> queued
        self._worker.previewReady.connect(self.previewReady)
        self._worker.thumbnailsReady.connect(self.thumbnailsReady)
        self._worker.renderFailed.connect(self.renderFailed)
        self._thread.start()

    @property
    def video(self) -> VideoSource | None:
        """The first decoder adopted by the worker (for diagnostics and tests)."""
        return self._worker.video

    @property
    def videos(self) -> tuple[VideoSource, ...]:
        return self._worker.videos

    def set_videos(self, videos: Sequence[VideoSource]) -> None:
        """Hand the sequence's open videos to the render thread; it releases the ones no longer listed."""
        self._worker.set_videos(videos)
        self._wake.emit()

    def set_video(self, video: VideoSource | None) -> None:
        """The one-video case of :meth:`set_videos`."""
        self.set_videos(() if video is None else (video,))

    def submit_preview(self, request: PreviewRequest) -> None:
        self.mailbox.put_preview(request)
        self._wake.emit()

    def submit_thumbnails(self, request: ThumbnailRequest) -> None:
        self.mailbox.put_thumbnails(request)
        self._wake.emit()

    def stop(self, timeout_ms: int = 5_000) -> None:
        self.mailbox.clear()
        self._thread.quit()
        self._thread.wait(timeout_ms)
        self._worker.release_videos()
        self._worker.deleteLater()
