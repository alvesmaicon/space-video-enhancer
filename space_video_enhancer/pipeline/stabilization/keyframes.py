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


ManualParams = tuple[float, float, float, float, float, float]
"""``(tx, ty, rotation, scale, shear, stretch_y)`` — one interpolated correction."""


def _params(keyframe: StabilizationKeyframe) -> ManualParams:
    return (keyframe.tx, keyframe.ty, keyframe.rotation, keyframe.scale, keyframe.shear, keyframe.stretch_y)


def interpolate_keyframes(keyframes: tuple[StabilizationKeyframe, ...], frame_index: int | None) -> ManualParams | None:
    """The ``(tx, ty, rotation, scale, shear, stretch_y)`` for ``frame_index``, or ``None`` when there are none.

    ``keyframes`` must be sorted by frame (``StabilizationSettings`` keeps them so). Before the first
    keyframe and after the last one the value is held flat, so the correction never extrapolates.
    """
    if not keyframes:
        return None
    if frame_index is None:
        frame_index = keyframes[0].frame
    if frame_index <= keyframes[0].frame:
        return _params(keyframes[0])
    if frame_index >= keyframes[-1].frame:
        return _params(keyframes[-1])
    for before, after in zip(keyframes, keyframes[1:]):
        if before.frame <= frame_index <= after.frame:
            span = after.frame - before.frame
            weight = 0.0 if span <= 0 else (frame_index - before.frame) / span
            return (
                _lerp(before.tx, after.tx, weight),
                _lerp(before.ty, after.ty, weight),
                _lerp(before.rotation, after.rotation, weight),
                _lerp(before.scale, after.scale, weight),
                _lerp(before.shear, after.shear, weight),
                _lerp(before.stretch_y, after.stretch_y, weight),
            )
    return None


def manual_correction(
    keyframes: tuple[StabilizationKeyframe, ...], frame_index: int | None, shape: tuple[int, ...]
) -> np.ndarray | None:
    """The 2x3 affine for ``frame_index``: rotation/scale/shear/vertical-stretch about the centre + translation."""
    params = interpolate_keyframes(keyframes, frame_index)
    if params is None:
        return None
    tx, ty, rotation, scale, shear, stretch_y = params
    height, width = shape[:2]
    center_x, center_y = width / 2.0, height / 2.0

    # Shear (x leans with height) and vertical stretch, both about the centre — the manual jelly correction.
    jelly = np.array(
        [[1.0, shear, -shear * center_y], [0.0, stretch_y, (1.0 - stretch_y) * center_y]],
        dtype=np.float64,
    )
    # Rotation + uniform scale about the centre, lifted to 3x3 so it composes on top of the jelly.
    rotate_scale = np.vstack([cv2.getRotationMatrix2D((center_x, center_y), rotation, scale), (0.0, 0.0, 1.0)])
    matrix = rotate_scale @ np.vstack([jelly, (0.0, 0.0, 1.0)])
    matrix[0, 2] += tx * width
    matrix[1, 2] += ty * height
    return matrix[:2, :].astype(np.float32)
