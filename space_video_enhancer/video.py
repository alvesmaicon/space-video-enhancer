"""Frame access for the preview.

``cv2.VideoCapture.set(CAP_PROP_POS_FRAMES)`` seeks to the previous keyframe and decodes
forward, which for H.264 can mean dozens of decodes per requested frame. Playback and
forward scrubbing request consecutive (or nearly consecutive) frames, so we read
sequentially whenever the target is a short distance ahead of the decoder position.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

GRAB_FORWARD_LIMIT = 30
"""Targets up to this many frames ahead are reached with ``grab()`` instead of a seek."""


class VideoOpenError(RuntimeError):
    pass


class VideoSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise VideoOpenError(f"Nao foi possivel abrir o video: {path}")
        self.frame_count = max(1, int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next_index: int | None = 0  # decoder position right after opening
        self.sequential_reads = 0
        self.seeks = 0

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def read_frame(self, index: int) -> np.ndarray | None:
        """Decode frame ``index`` (clamped to the known range); ``None`` when the stream ends early."""
        index = max(0, min(index, self.frame_count - 1))
        distance = None if self._next_index is None else index - self._next_index

        if distance is not None and 0 <= distance < GRAB_FORWARD_LIMIT:
            for _ in range(distance):
                if not self._capture.grab():
                    self._next_index = None
                    return None
            self.sequential_reads += 1
        else:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.seeks += 1

        ok, frame = self._capture.read()
        if not ok:
            self._next_index = None
            return None
        self._next_index = index + 1
        return frame

    def release(self) -> None:
        self._capture.release()
