"""Manual stabilization: turn the hand-set keyframes into a per-frame corrective transform.

The user sets a :class:`StabilizationKeyframe` every so often (translation as a fraction of the
frame, rotation in degrees, scale as a ratio). For any frame the four fields are interpolated
between the two bracketing keyframes — linearly, held flat outside the first/last — and composed
into a similarity affine (rotation + scale about the centre, then translation). It is a pure
function of the keyframe list and the frame index, so preview and export agree and nothing is
stateful. Only position/rotation/zoom, never perspective, so it cannot warp the image.
"""

from __future__ import annotations

import cv2
import numpy as np

from space_video_enhancer.settings import StabilizationKeyframe


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interpolate_keyframes(
    keyframes: tuple[StabilizationKeyframe, ...], frame_index: int | None
) -> tuple[float, float, float, float] | None:
    """The ``(tx, ty, rotation, scale)`` for ``frame_index``, or ``None`` when there are no keyframes.

    ``keyframes`` must be sorted by frame (``StabilizationSettings`` keeps them so). Before the first
    keyframe and after the last one the value is held flat, so the correction never extrapolates.
    """
    if not keyframes:
        return None
    if frame_index is None:
        frame_index = keyframes[0].frame
    if frame_index <= keyframes[0].frame:
        first = keyframes[0]
        return (first.tx, first.ty, first.rotation, first.scale)
    if frame_index >= keyframes[-1].frame:
        last = keyframes[-1]
        return (last.tx, last.ty, last.rotation, last.scale)
    for before, after in zip(keyframes, keyframes[1:]):
        if before.frame <= frame_index <= after.frame:
            span = after.frame - before.frame
            weight = 0.0 if span <= 0 else (frame_index - before.frame) / span
            return (
                _lerp(before.tx, after.tx, weight),
                _lerp(before.ty, after.ty, weight),
                _lerp(before.rotation, after.rotation, weight),
                _lerp(before.scale, after.scale, weight),
            )
    return None


def manual_correction(
    keyframes: tuple[StabilizationKeyframe, ...], frame_index: int | None, shape: tuple[int, ...]
) -> np.ndarray | None:
    """The 2x3 similarity affine for ``frame_index`` (rotation/scale about the centre + translation)."""
    params = interpolate_keyframes(keyframes, frame_index)
    if params is None:
        return None
    tx, ty, rotation, scale = params
    height, width = shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), rotation, scale)
    matrix[0, 2] += tx * width
    matrix[1, 2] += ty * height
    return matrix.astype(np.float32)
