"""Mutable per-sequence state for inter-frame stabilization.

One instance per independent frame sequence (preview scrubbing has its own, each
export job has its own). Never share an instance across threads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from space_video_enhancer.pipeline.stabilization import transforms

MAX_FRAME_GAP = 12
"""Frames a sequence may skip and still continue. The preview drops frames whenever rendering is
slower than playback (40%+ with heavy filters at 1080p); restarting the path on every drop made
the stabilization flicker on and off. Larger jumps (scrubbing far away) still restart."""


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
    correction_width: int | None = None
    """Width of the frame ``correction`` is expressed in. The preview renders the same sequence at
    different sizes (interactive 640 px, full-size idle render), so corrections are rescaled on reuse."""
    lock_lost_frames: int = 0
    """Hybrid: recent frames without a camera-lock match (capped); eases the re-acquisition."""

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def breaks_sequence(self, frame_index: int | None) -> bool:
        """True when ``frame_index`` is known and does not continue the remembered frame (see :func:`frame_gap`)."""
        return (
            frame_index is not None
            and self.prev_frame_index is not None
            and frame_gap(self.prev_frame_index, frame_index) is None
        )

    def correction_gap(self, frame_index: int | None) -> int | None:
        """Frames between the last correction and ``frame_index`` when the sequence continues, else ``None``."""
        if frame_index is None or self.correction_frame_index is None:
            return None
        return frame_gap(self.correction_frame_index, frame_index)

    def remember_gray(self, gray: np.ndarray, frame_index: int | None) -> None:
        self.prev_gray = gray
        self.prev_frame_index = frame_index

    def remember_shift(self, shift: tuple[float, float] | None, frame_index: int | None) -> None:
        self.last_shift = shift
        self.last_shift_index = frame_index

    def remember_transform(self, transform: np.ndarray | None, frame_index: int | None) -> None:
        self.last_transform = transform
        self.last_transform_index = frame_index

    def remember_correction(self, correction: np.ndarray | None, frame_index: int | None, width: int) -> None:
        self.correction = correction
        self.correction_frame_index = frame_index
        self.correction_width = width

    def adopt_frame_width(self, width: int) -> None:
        """Estimator caches hold pixel values of the previous size; drop them when the size changes."""
        if self.correction_width is not None and self.correction_width != width:
            self.last_shift_index = None
            self.last_transform_index = None

    def cached_correction(self, frame_index: int | None, width: int) -> np.ndarray | None:
        """The correction already computed for ``frame_index`` (a re-render of the same frame)."""
        if frame_index is None or self.correction_frame_index != frame_index:
            return None
        return self._correction_in_units_of(width)

    def previous_correction(self, frame_index: int | None, width: int) -> np.ndarray | None:
        """The correction of the last processed frame when ``frame_index`` continues that sequence."""
        if self.correction_gap(frame_index) is None:
            return None
        return self._correction_in_units_of(width)

    def _correction_in_units_of(self, width: int) -> np.ndarray | None:
        if self.correction is None or self.correction_width in (None, width):
            return self.correction
        assert self.correction_width is not None
        scale = self.correction_width / width  # the stored correction lives on a frame ``scale`` times this one
        if transforms.is_affine(self.correction):
            return transforms.rescale_affine(self.correction, scale)
        return transforms.rescale_homography(self.correction, scale)


def frame_gap(previous_index: int, frame_index: int) -> int | None:
    """Signed distance from ``previous_index`` to ``frame_index`` when it continues the sequence.

    Consecutive frames give 1; dropped frames give a small positive gap; stepping backwards gives a
    negative one (motion estimation works either way). Re-rendering the same frame or jumping more
    than :data:`MAX_FRAME_GAP` frames does not continue the sequence.
    """
    gap = frame_index - previous_index
    if gap == 0 or abs(gap) > MAX_FRAME_GAP:
        return None
    return gap
