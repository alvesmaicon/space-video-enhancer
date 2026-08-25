"""Qt adapter that renders preview/thumbnail requests on a ``QThread`` and emits ``QImage`` results."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, QSize, QThread, Signal, Slot
from PySide6.QtGui import QImage

from space_video_enhancer.pipeline.filters import OpenCLBackend
from space_video_enhancer.pipeline.preview import (
    PreviewRequest,
    RenderMailbox,
    ThumbnailRequest,
    render_preview,
    render_thumbnails,
)
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.ui.images import frame_to_qimage, thumbnail_qimage


@dataclass(frozen=True)
class PreviewImages:
    generation: int
    frame_index: int
    before: QImage
    after: QImage
    elapsed_ms: float


@dataclass(frozen=True)
class ThumbnailImages:
    generation: int
    images: dict[str, QImage]
    elapsed_ms: float


class PreviewWorker(QObject):
    previewReady = Signal(object)  # PreviewImages
    thumbnailsReady = Signal(object)  # ThumbnailImages
    renderFailed = Signal(str)

    def __init__(self, mailbox: RenderMailbox, backend: OpenCLBackend | None, thumbnail_size: QSize) -> None:
        super().__init__()
        self._mailbox = mailbox
        self._backend = backend
        self._thumbnail_size = thumbnail_size
        self._state = StabilizationState()
        self._epoch: int | None = None

    @Slot()
    def process(self) -> None:
        """Drain the mailbox. Extra wake-ups find it empty and return immediately."""
        while True:
            job = self._mailbox.take()
            if job is None:
                return
            try:
                if isinstance(job, PreviewRequest):
                    self._render_preview(job)
                else:
                    self._render_thumbnails(job)
            except Exception as error:
                self.renderFailed.emit(f"{type(error).__name__}: {error}")

    def _render_preview(self, job: PreviewRequest) -> None:
        if job.stabilization_epoch != self._epoch:
            self._state.reset()
            self._epoch = job.stabilization_epoch
        result = render_preview(job, self._state, self._backend)
        self.previewReady.emit(
            PreviewImages(
                result.generation,
                result.frame_index,
                frame_to_qimage(result.before),
                frame_to_qimage(result.after),
                result.elapsed_ms,
            )
        )

    def _render_thumbnails(self, job: ThumbnailRequest) -> None:
        result = render_thumbnails(job, self._backend)
        images = {
            preset_id: thumbnail_qimage(frame, self._thumbnail_size) for preset_id, frame in result.frames.items()
        }
        self.thumbnailsReady.emit(ThumbnailImages(result.generation, images, result.elapsed_ms))


class PreviewController(QObject):
    """Owns the render thread; the UI submits requests and listens to the ready signals."""

    previewReady = Signal(object)  # PreviewImages
    thumbnailsReady = Signal(object)  # ThumbnailImages
    renderFailed = Signal(str)
    _wake = Signal()

    def __init__(self, backend: OpenCLBackend | None, thumbnail_size: QSize, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.mailbox = RenderMailbox()
        self._thread = QThread(self)
        self._thread.setObjectName("preview-render")
        self._worker = PreviewWorker(self.mailbox, backend, thumbnail_size)
        self._worker.moveToThread(self._thread)
        self._wake.connect(self._worker.process)  # cross-thread -> queued
        self._worker.previewReady.connect(self.previewReady)
        self._worker.thumbnailsReady.connect(self.thumbnailsReady)
        self._worker.renderFailed.connect(self.renderFailed)
        self._thread.start()

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
        self._worker.deleteLater()
