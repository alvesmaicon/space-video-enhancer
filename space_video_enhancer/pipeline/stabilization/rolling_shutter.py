"""Rolling-shutter correction driven by the inter-frame motion the stabilizer already measures.

A CMOS sensor reads its rows one after another, so the bottom of a frame is exposed later than
the top — or, for a phone held upright, the right side later than the left: the readout axis
(:class:`~settings.ReadoutAxis`) is a parameter of everything below, "rows" being the text's case.
Two things follow, and each has its own fix here:

* **Skew** under steady motion (a pan): vertical lines lean, the image stretches or squashes. If
  the scene moved ``d`` pixels between frames and the readout takes a fraction ``r`` of the frame
  period, row ``y`` is displaced by ``d·r·(y/h - ½)``. ``r`` cannot be measured from the footage
  (steady motion skews every frame the same way), so it is the "Correção de rolling shutter" slider.

* **Wobble** ("jello") under shake: the velocity changes *during* the readout, so each frame is
  stretched or leaned differently from the last. That one is measurable without ``r``: between two
  frames, tracked points at the top and at the bottom move by different amounts, and the gradient
  of that displacement down the frame is exactly how much the intra-frame distortion changed from
  one frame to the next (Grundmann et al., "Calibration-free rolling shutter removal", 2012). The
  gradient is fitted robustly over all tracked points here, together with the frame's roll and zoom
  so those are not taken for it; summing it over a whole range of frames and pinning the sum down
  is :mod:`wobble`'s job.

Both corrections are one per-row shift, linear in the row: one ``cv2.remap`` samples row ``y`` at
``q + G·(y/h - ½)`` with ``G`` the total gradient (skew plus wobble), in pixels across the height.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from space_video_enhancer.pipeline.stabilization import transforms
from space_video_enhancer.settings import ReadoutAxis

NEGLIGIBLE_SKEW_PX = 0.5
"""Below this much displacement at the top or bottom edge the remap would move nothing visible."""
GRADIENT_MIN_POINTS = 12
GRADIENT_MIN_SPREAD = 0.4
"""Tracked points must span this fraction of the height for a gradient down the frame to mean anything."""
GRADIENT_INLIER_PX = 2.0
GRADIENT_RANSAC_ITERATIONS = 48
GRADIENT_MIN_INLIER_FRACTION = 0.5
GRADIENT_SAMPLE_SIZE = 3
"""Points per RANSAC sample: six unknowns (shift, roll, zoom, gradient), two equations each."""


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


def motion_gradient(vectors: Any, axis_length: int, axis: ReadoutAxis = ReadoutAxis.ROWS) -> tuple[float, float] | None:
    """How the scene displacement between two frames grows along the readout axis.

    ``vectors`` is the estimator's :class:`~global_motion.MotionVectors` (all tracked points, analysis
    coordinates). With ``X, Y`` the point's position from the centre (midway between the two frames)
    and ``u`` its position along the axis as a fraction (-½ .. ½), each point contributes::

        dx = a_x - θ·Y + s·X + gx·u        dy = a_y + θ·X + s·Y + gy·u

    so a camera roll (``θ``) or zoom (``s``) between the frames — which also make the displacement vary
    across the frame — is not mistaken for rolling shutter. A small RANSAC keeps moving subjects out.
    Returns ``(gx, gy)`` in pixels across the whole axis of a frame whose axis measures
    ``axis_length``, or ``None`` when too few points, too little spread along the axis or no consistent fit.
    """
    previous = np.asarray(vectors.previous, dtype=np.float64).reshape(-1, 2)
    current = np.asarray(vectors.current, dtype=np.float64).reshape(-1, 2)
    count = len(previous)
    if count < GRADIENT_MIN_POINTS:
        return None
    width, height = vectors.size
    if width <= 0 or height <= 0:
        return None
    midpoint = (current + previous) / 2.0
    centred_x = midpoint[:, 0] - width / 2.0
    centred_y = midpoint[:, 1] - height / 2.0
    along = centred_x / width if axis is ReadoutAxis.COLUMNS else centred_y / height
    if along.max() - along.min() < GRADIENT_MIN_SPREAD:
        return None
    displacement = current - previous

    zeros, ones = np.zeros(count), np.ones(count)
    design = np.concatenate(
        [
            np.column_stack([ones, zeros, -centred_y, centred_x, along, zeros]),
            np.column_stack([zeros, ones, centred_x, centred_y, zeros, along]),
        ]
    )
    target = np.concatenate([displacement[:, 0], displacement[:, 1]])

    def fit(mask: np.ndarray) -> np.ndarray:
        rows = np.concatenate([np.flatnonzero(mask), np.flatnonzero(mask) + count])
        params, *_rest = np.linalg.lstsq(design[rows], target[rows], rcond=None)
        return params

    def residuals(params: np.ndarray) -> np.ndarray:
        residual = design @ params - target
        return np.hypot(residual[:count], residual[count:])

    rng = np.random.default_rng(count)  # deterministic: the same frame always gives the same answer
    best_inliers: np.ndarray | None = None
    for _ in range(GRADIENT_RANSAC_ITERATIONS):
        sample = rng.choice(count, GRADIENT_SAMPLE_SIZE, replace=False)
        if along[sample].max() - along[sample].min() < 0.05:
            continue
        mask = np.zeros(count, dtype=bool)
        mask[sample] = True
        inliers = residuals(fit(mask)) <= GRADIENT_INLIER_PX
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < max(GRADIENT_MIN_POINTS, GRADIENT_MIN_INLIER_FRACTION * count):
        return None
    if along[best_inliers].max() - along[best_inliers].min() < GRADIENT_MIN_SPREAD:
        return None
    params = fit(best_inliers)
    to_full = axis_length / (width if axis is ReadoutAxis.COLUMNS else height)
    return float(params[4] * to_full), float(params[5] * to_full)


def row_motion_gradient(vectors: Any, frame_height: int) -> tuple[float, float] | None:
    """:func:`motion_gradient` along the rows (the sensor read top to bottom)."""
    return motion_gradient(vectors, frame_height, ReadoutAxis.ROWS)


def skew_maps(
    frame_shape: tuple[int, ...], gradient: tuple[float, float], axis: ReadoutAxis = ReadoutAxis.ROWS
) -> tuple[np.ndarray, np.ndarray]:
    """``(map_x, map_y)`` for :func:`cv2.remap`: the line at ``u`` along ``axis`` samples ``gradient·(u - ½)`` away."""
    height, width = frame_shape[:2]
    gx, gy = gradient
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    if axis is ReadoutAxis.COLUMNS:
        ramp = xs / max(1, width - 1) - 0.5
        map_x = np.broadcast_to((xs + np.float32(gx) * ramp)[None, :], (height, width))
        map_y = ys[:, None] + (np.float32(gy) * ramp)[None, :]
    else:
        ramp = ys / max(1, height - 1) - 0.5
        map_x = xs[None, :] + (np.float32(gx) * ramp)[:, None]
        map_y = np.broadcast_to((ys + np.float32(gy) * ramp)[:, None], (height, width))
    return np.ascontiguousarray(map_x, dtype=np.float32), np.ascontiguousarray(map_y, dtype=np.float32)


def correct(frame: np.ndarray, gradient: tuple[float, float], axis: ReadoutAxis = ReadoutAxis.ROWS) -> np.ndarray:
    """``frame`` with a distortion of ``gradient`` pixels (across the readout axis, linear along it) undone.

    Returns ``frame`` itself when the correction would move the edges by less than half a pixel.
    """
    gx, gy = gradient
    if abs(gx) * 0.5 < NEGLIGIBLE_SKEW_PX and abs(gy) * 0.5 < NEGLIGIBLE_SKEW_PX:
        return frame
    map_x, map_y = skew_maps(frame.shape, gradient, axis)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def skew(
    frame: np.ndarray,
    displacement: tuple[float, float],
    readout_fraction: float,
    axis: ReadoutAxis = ReadoutAxis.ROWS,
) -> np.ndarray:
    """Simulate a rolling shutter: the scene moved ``displacement`` px during a readout of ``readout_fraction``."""
    dx, dy = displacement
    map_x, map_y = skew_maps(frame.shape, (-dx * readout_fraction, -dy * readout_fraction), axis)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def wobble(frame: np.ndarray, gradient: tuple[float, float], axis: ReadoutAxis = ReadoutAxis.ROWS) -> np.ndarray:
    """Simulate a frame whose lines along ``axis`` are displaced by ``gradient·(u - ½)`` (tests and benchmarks)."""
    map_x, map_y = skew_maps(frame.shape, (-gradient[0], -gradient[1]), axis)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
