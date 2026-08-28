"""Rolling-shutter skew correction driven by the inter-frame motion the stabilizer already measures.

A CMOS sensor reads its rows one after another, so the bottom of a frame is exposed later than
the top. Pan the camera and vertical lines lean; move it up or down and the image stretches or
squashes. If the scene moved by ``d`` pixels between the previous frame and this one, the row at
relative time offset ``τ(y)`` (from -r/2 at the top to +r/2 at the bottom, ``r`` being the readout
time as a fraction of the frame period) shows the scene displaced by ``d·τ(y)``. Undoing it is one
``cv2.remap`` that samples row ``y`` at ``q + d·τ(y)``.

``r`` cannot be measured from the footage alone; it is the "Correção de rolling shutter" slider.
"""

from __future__ import annotations

import cv2
import numpy as np

from space_video_enhancer.pipeline.stabilization import transforms

NEGLIGIBLE_SKEW_PX = 0.5
"""Below this much displacement at the top or bottom edge the remap would move nothing visible."""


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


def skew_maps(
    frame_shape: tuple[int, ...], displacement: tuple[float, float], readout_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(map_x, map_y)`` for :func:`cv2.remap`: where each output pixel samples the skewed source."""
    height, width = frame_shape[:2]
    dx, dy = displacement
    tau = (np.arange(height, dtype=np.float32) / max(1, height - 1) - 0.5) * np.float32(readout_fraction)
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    map_x = xs[None, :] + (np.float32(dx) * tau)[:, None]
    map_y = np.broadcast_to((ys + np.float32(dy) * tau)[:, None], (height, width))
    return np.ascontiguousarray(map_x), np.ascontiguousarray(map_y)


def correct(frame: np.ndarray, displacement: tuple[float, float], readout_fraction: float) -> np.ndarray:
    """``frame`` unskewed for a scene that moved ``displacement`` px during the last frame period.

    Returns ``frame`` itself when the correction would be smaller than half a pixel at the edges.
    """
    dx, dy = displacement
    edge = 0.5 * max(0.0, min(1.0, readout_fraction))
    if abs(dx) * edge < NEGLIGIBLE_SKEW_PX and abs(dy) * edge < NEGLIGIBLE_SKEW_PX:
        return frame
    map_x, map_y = skew_maps(frame.shape, displacement, readout_fraction)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def skew(frame: np.ndarray, displacement: tuple[float, float], readout_fraction: float) -> np.ndarray:
    """The inverse of :func:`correct`: simulate a rolling shutter (used by tests and benchmarks)."""
    dx, dy = displacement
    map_x, map_y = skew_maps(frame.shape, (-dx, -dy), readout_fraction)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
