"""Decoding of camera-lock reference frames on their own thread (needed when a project is opened).

A project stores only the *index* of each clip's reference frame; the pixels must be decoded again
before the lock can work. That never happens on the GUI thread: the worker opens the file, reads
the frame, releases the decoder and reports back tagged with ``(path, frame)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.video import VideoOpenError, VideoSource

LOG = get_logger("reference")


class ReferenceFrameWorker(QObject):
    frameReady = Signal(object, int, object)  # Path, source frame, np.ndarray | None

    @Slot(object, int)
    def load(self, path: Path, frame_index: int) -> None:
        frame = None
        try:
            video = VideoSource(path)
        except VideoOpenError as error:
            log_event(LOG, "reference.open_failed", level=logging.WARNING, name=path.name, error=str(error))
        else:
            try:
                frame = video.read_frame(frame_index)
            finally:
                video.release()
            log_event(LOG, "reference.decoded", name=path.name, frame=frame_index, ok=frame is not None)
        self.frameReady.emit(path, frame_index, frame)


class ReferenceFrameController(QObject):
    """Owns the decoding thread; results arrive on the GUI thread through ``frameReady``."""

    frameReady = Signal(object, int, object)
    _load = Signal(object, int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._thread.setObjectName("reference-frames")
        self._worker = ReferenceFrameWorker()
        self._worker.moveToThread(self._thread)
        self._load.connect(self._worker.load)  # queued on the worker thread
        self._worker.frameReady.connect(self.frameReady)
        self._thread.start()

    def load(self, path: Path, frame_index: int) -> None:
        self._load.emit(path, frame_index)

    def stop(self, timeout_ms: int = 5_000) -> None:
        self._thread.quit()
        self._thread.wait(timeout_ms)
        self._worker.deleteLater()
