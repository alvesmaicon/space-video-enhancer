"""Rolling-shutter correction driven by the inter-frame motion the stabilizer already measures.

A CMOS sensor reads its rows one after another, so the bottom of a frame is exposed later than
the top. Two things follow, and each has its own fix here:

* **Skew** under steady motion (a pan): vertical lines lean, the image stretches or squashes. If
  the scene moved ``d`` pixels between frames and the readout takes a fraction ``r`` of the frame
  period, row ``y`` is displaced by ``d·r·(y/h - ½)``. ``r`` cannot be measured from the footage
  (steady motion skews every frame the same way), so it is the "Correção de rolling shutter" slider.

* **Wobble** ("jello") under shake: the velocity changes *during* the readout, so each frame is
  stretched or leaned differently from the last. That one is measurable without ``r``: between two
  frames, tracked points at the top and at the bottom move by different amounts, and the gradient
  of that displacement down the frame is exactly how much the intra-frame distortion changed from
  one frame to the next (Grundmann et al., "Calibration-free rolling shutter removal", 2012). The
  gradient is fitted robustly over all tracked points and accumulated with a slow decay into the
  current wobble, which is then undone.

Both corrections are one per-row shift, linear in the row: one ``cv2.remap`` samples row ``y`` at
``q + G·(y/h - ½)`` with ``G`` the total gradient (skew plus wobble), in pixels across the height.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from space_video_enhancer.pipeline.stabilization import transforms

NEGLIGIBLE_SKEW_PX = 0.5
"""Below this much displacement at the top or bottom edge the remap would move nothing visible."""
WOBBLE_DECAY = 0.97
"""Per-frame factor on the accumulated wobble. The wobble itself is a sum of measured changes; the decay only
keeps measurement noise from drifting (it fades in ~30 frames) — a shake oscillates around zero anyway."""
MAX_WOBBLE_FRACTION = 0.08
"""Cap on the accumulated wobble, as a fraction of the frame height (a phone's jello stays well below)."""
GRADIENT_MIN_POINTS = 12
GRADIENT_MIN_SPREAD = 0.4
"""Tracked points must span this fraction of the height for a gradient down the frame to mean anything."""
GRADIENT_INLIER_PX = 2.0
GRADIENT_RANSAC_ITERATIONS = 48
GRADIENT_MIN_INLIER_FRACTION = 0.5


def center_displacement(delta: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[float, float]:
    """How far the scene moved at the frame centre between the previous frame and this one, in pixels.

    ``delta`` aligns this frame onto the previous one (the stabilizer's inter-frame transform), so
    the scene displacement is the opposite of where ``delta`` sends the centre.
    """
    height, width = frame_shape[:2]
    centre = np.array([width / 2.0, height / 2.0, 1.0], dtype=np.float64)
    moved = transforms.affine_to_homography(delta).astype(np.float64) @ centre
    if abs(moved[2]) > 1e-9:
        moved = moved / moved[2]
    return float(centre[0] - moved[0]), float(centre[1] - moved[1])


def row_motion_gradient(vectors: Any, frame_height: int) -> tuple[float, float] | None:
    """How the scene displacement between two frames grows from the top to the bottom of the frame.

    ``vectors`` is the estimator's :class:`~global_motion.MotionVectors` (all tracked points, analysis
    coordinates). Fits ``s(y) = a + g·(y/h - ½)`` to the displacements ``current - previous`` with a
    small RANSAC (moving subjects are outliers) and returns ``g`` in full-resolution pixels across the
    whole height, or ``None`` when too few points, too little vertical spread or no consistent fit.
    """
    previous = np.asarray(vectors.previous, dtype=np.float64).reshape(-1, 2)
    current = np.asarray(vectors.current, dtype=np.float64).reshape(-1, 2)
    count = len(previous)
    if count < GRADIENT_MIN_POINTS:
        return None
    _width, analysis_height = vectors.size
    if analysis_height <= 0:
        return None
    rows = (current[:, 1] + previous[:, 1]) / (2.0 * analysis_height) - 0.5  # the point's row midway between frames
    if rows.max() - rows.min() < GRADIENT_MIN_SPREAD:
        return None
    displacement = current - previous

    design = np.column_stack([np.ones(count), rows])
    rng = np.random.default_rng(count)  # deterministic: the same frame always gives the same answer
    best_inliers: np.ndarray | None = None
    for _ in range(GRADIENT_RANSAC_ITERATIONS):
        sample = rng.choice(count, 2, replace=False)
        if abs(rows[sample[0]] - rows[sample[1]]) < 0.05:
            continue
        params, *_rest = np.linalg.lstsq(design[sample], displacement[sample], rcond=None)
        residual = np.linalg.norm(design @ params - displacement, axis=1)
        inliers = residual <= GRADIENT_INLIER_PX
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < max(GRADIENT_MIN_POINTS, GRADIENT_MIN_INLIER_FRACTION * count):
        return None
    if rows[best_inliers].max() - rows[best_inliers].min() < GRADIENT_MIN_SPREAD:
        return None
    params, *_rest = np.linalg.lstsq(design[best_inliers], displacement[best_inliers], rcond=None)
    to_full = frame_height / analysis_height
    return float(params[1, 0] * to_full), float(params[1, 1] * to_full)


def accumulate_wobble(
    previous: tuple[float, float], gradient: tuple[float, float] | None, frame_height: int
) -> tuple[float, float]:
    """The wobble of this frame from the last one's: decays, then adds how much the distortion changed."""
    limit = frame_height * MAX_WOBBLE_FRACTION
    gx, gy = gradient if gradient is not None else (0.0, 0.0)
    wobble_x = float(np.clip(previous[0] * WOBBLE_DECAY + gx, -limit, limit))
    wobble_y = float(np.clip(previous[1] * WOBBLE_DECAY + gy, -limit, limit))
    return wobble_x, wobble_y


def skew_maps(frame_shape: tuple[int, ...], gradient: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """``(map_x, map_y)`` for :func:`cv2.remap`: row ``y`` samples ``gradient·(y/h - ½)`` away from itself."""
    height, width = frame_shape[:2]
    gx, gy = gradient
    ramp = np.arange(height, dtype=np.float32) / max(1, height - 1) - 0.5
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    map_x = xs[None, :] + (np.float32(gx) * ramp)[:, None]
    map_y = np.broadcast_to((ys + np.float32(gy) * ramp)[:, None], (height, width))
    return np.ascontiguousarray(map_x), np.ascontiguousarray(map_y)


def correct(frame: np.ndarray, gradient: tuple[float, float]) -> np.ndarray:
    """``frame`` with a row-linear distortion of ``gradient`` pixels (top to bottom) undone.

    Returns ``frame`` itself when the correction would move the edges by less than half a pixel.
    """
    gx, gy = gradient
    if abs(gx) * 0.5 < NEGLIGIBLE_SKEW_PX and abs(gy) * 0.5 < NEGLIGIBLE_SKEW_PX:
        return frame
    map_x, map_y = skew_maps(frame.shape, gradient)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def skew(frame: np.ndarray, displacement: tuple[float, float], readout_fraction: float) -> np.ndarray:
    """Simulate a rolling shutter: the scene moved ``displacement`` px during a readout of ``readout_fraction``."""
    dx, dy = displacement
    map_x, map_y = skew_maps(frame.shape, (-dx * readout_fraction, -dy * readout_fraction))
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def wobble(frame: np.ndarray, gradient: tuple[float, float]) -> np.ndarray:
    """Simulate a frame whose rows are displaced by ``gradient·(y/h - ½)`` (tests and benchmarks)."""
    map_x, map_y = skew_maps(frame.shape, (-gradient[0], -gradient[1]))
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
