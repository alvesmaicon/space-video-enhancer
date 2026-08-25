"""Mutable per-sequence state for inter-frame stabilization.

One instance per independent frame sequence (preview scrubbing has its own, each
export job has its own). Never share an instance across threads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StabilizationState:
    prev_gray: np.ndarray | None = None
    prev_frame_index: int | None = None
    last_shift: tuple[float, float] | None = None
    last_shift_index: int | None = None
    last_transform: np.ndarray | None = None
    last_transform_index: int | None = None
    correction: np.ndarray | None = None
    """Full-strength correction (before the intensity blend) of the last processed frame."""
    correction_frame_index: int | None = None
    lock_lost_frames: int = 0
    """Hybrid: recent frames without a camera-lock match (capped); eases the re-acquisition."""

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def breaks_sequence(self, frame_index: int | None) -> bool:
        """True when ``frame_index`` is known and is not the frame right after the remembered one."""
        return (
            frame_index is not None and self.prev_frame_index is not None and frame_index != self.prev_frame_index + 1
        )

    def remember_gray(self, gray: np.ndarray, frame_index: int | None) -> None:
        self.prev_gray = gray
        self.prev_frame_index = frame_index

    def remember_shift(self, shift: tuple[float, float] | None, frame_index: int | None) -> None:
        self.last_shift = shift
        self.last_shift_index = frame_index

    def remember_transform(self, transform: np.ndarray | None, frame_index: int | None) -> None:
        self.last_transform = transform
        self.last_transform_index = frame_index

    def remember_correction(self, correction: np.ndarray | None, frame_index: int | None) -> None:
        self.correction = correction
        self.correction_frame_index = frame_index

    def cached_correction(self, frame_index: int | None) -> np.ndarray | None:
        """The correction already computed for ``frame_index`` (a re-render of the same frame)."""
        if frame_index is None or self.correction_frame_index != frame_index:
            return None
        return self.correction

    def previous_correction(self, frame_index: int | None) -> np.ndarray | None:
        """The correction of the frame right before ``frame_index``, if that is what was processed last."""
        if frame_index is None or self.correction_frame_index != frame_index - 1:
            return None
        return self.correction
