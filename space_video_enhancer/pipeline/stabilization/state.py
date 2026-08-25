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
    smoothed_transform: np.ndarray | None = None
    smoothed_frame_index: int | None = None
    hybrid_fallback_bias: float = 0.0

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
