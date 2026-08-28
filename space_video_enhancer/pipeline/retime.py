"""Compose the frame an output time falls on when a clip is retimed to the timeline fps.

:meth:`Sequence.export_samples` / :meth:`Sequence.preview_sample` decide *which* source frames a
retimed output frame comes from (a :class:`~space_video_enhancer.sequence.RetimeSample`); this module
turns that plan into pixels. Three qualities, matching the clip's :class:`RetimingMode`:

* ``NEAREST`` — the single frame the sample already resolved to (nothing to compose).
* ``BLEND`` — a cross-dissolve of the two straddling frames, weighted by the sample.
* ``OPTICAL_FLOW`` — the in-between frame warped along the motion between the two frames, so a
  panning subject moves to where it would actually be rather than ghosting (dense flow: DIS, or
  Farneback where DIS is unavailable). Costly, so the preview falls back to a blend.

``compose`` reads through a ``read(index) -> frame | None`` callable, so the caller keeps owning the
decoder; a frame that cannot be read falls back to whichever neighbour did decode.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from space_video_enhancer.sequence import RetimeSample
from space_video_enhancer.settings import RetimingMode


def blend(frame_a: np.ndarray, frame_b: np.ndarray, weight: float) -> np.ndarray:
    """Cross-dissolve: ``(1 - weight)·frame_a + weight·frame_b``."""
    if weight <= 0.0:
        return frame_a
    if weight >= 1.0:
        return frame_b
    return cv2.addWeighted(frame_a, 1.0 - weight, frame_b, weight, 0.0)


def _dense_flow(previous_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    """Per-pixel motion from ``previous`` to ``next`` (``next(x + flow) ≈ previous(x)``)."""
    try:
        estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return estimator.calc(previous_gray, next_gray, None)
    except cv2.error:  # a build without DIS: the always-present pyramidal estimator
        return cv2.calcOpticalFlowFarneback(previous_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _warp(frame: np.ndarray, flow: np.ndarray, fraction: float) -> np.ndarray:
    """``frame`` moved a ``fraction`` of the way along ``flow`` (backward remap; holes take the nearest edge)."""
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x - flow[..., 0] * fraction
    map_y = grid_y - flow[..., 1] * fraction
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def optical_flow_interpolate(frame_a: np.ndarray, frame_b: np.ndarray, weight: float) -> np.ndarray:
    """The frame ``weight`` of the way from ``frame_a`` to ``frame_b``, each warped along the motion, then blended."""
    if weight <= 0.0:
        return frame_a
    if weight >= 1.0:
        return frame_b
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    forward = _dense_flow(gray_a, gray_b)  # a -> b, so a's pixels move to where they are in b
    backward = _dense_flow(gray_b, gray_a)  # b -> a
    warped_a = _warp(frame_a, forward, weight)
    warped_b = _warp(frame_b, backward, 1.0 - weight)
    return cv2.addWeighted(warped_a, 1.0 - weight, warped_b, weight, 0.0)


def compose(
    sample: RetimeSample, read: Callable[[int], np.ndarray | None], *, allow_optical_flow: bool = True
) -> np.ndarray | None:
    """Build the output frame described by ``sample``; ``None`` only when neither source frame decodes.

    ``allow_optical_flow=False`` downgrades ``OPTICAL_FLOW`` to a blend — the preview uses it so scrubbing
    stays responsive, the export leaves it on for the final quality.
    """
    frame_a = read(sample.a)
    if sample.is_single:
        return frame_a
    frame_b = read(sample.b)
    if frame_a is None:
        return frame_b
    if frame_b is None:
        return frame_a
    if sample.mode is RetimingMode.OPTICAL_FLOW and allow_optical_flow:
        return optical_flow_interpolate(frame_a, frame_b, sample.weight)
    return blend(frame_a, frame_b, sample.weight)
