"""Pure helpers for 2x3 affine and 3x3 homography transforms."""

from __future__ import annotations

import cv2
import numpy as np

AFFINE_SHAPE = (2, 3)
HOMOGRAPHY_SHAPE = (3, 3)

NEGLIGIBLE_TRANSLATION_PX = 0.5
NEGLIGIBLE_AFFINE_DELTA = 0.004
NEGLIGIBLE_HOMOGRAPHY_DELTA = 0.006

MAX_MEDIAN_DISPLACEMENT_FRACTION = 0.35
BBOX_MIN_FRACTION = 0.55
BBOX_MAX_FRACTION = 1.45
PERSPECTIVE_AREA_MIN_FRACTION = 0.45
PERSPECTIVE_AREA_MAX_FRACTION = 1.8

ZOOM_OUT_SAFE_FRACTION = 0.98
AUTO_CROP_EROSION_FRACTION = 0.015
AUTO_CROP_MIN_FRACTION = 0.55


def is_affine(transform: np.ndarray) -> bool:
    return transform.shape == AFFINE_SHAPE


def translation_to_affine(shift_x: float, shift_y: float) -> np.ndarray:
    return np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)


def affine_to_homography(transform: np.ndarray) -> np.ndarray:
    if transform.shape == HOMOGRAPHY_SHAPE:
        return transform.astype(np.float32)
    return np.vstack([transform, np.array([0.0, 0.0, 1.0], dtype=np.float32)]).astype(np.float32)


def _normalize_homography(homography: np.ndarray) -> np.ndarray:
    if abs(float(homography[2, 2])) > 1e-6:
        homography = homography / float(homography[2, 2])
    return homography.astype(np.float32)


def rescale_homography(transform: np.ndarray, scale: float) -> np.ndarray:
    """Express a transform estimated on a ``scale``-downsampled image in full-resolution pixels."""
    homography = affine_to_homography(transform)
    if abs(scale - 1.0) < 1e-3:
        return _normalize_homography(homography.astype(np.float32))
    scaling = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    inverse_scaling = np.array([[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return _normalize_homography(inverse_scaling @ homography @ scaling)


def rescale_affine(transform: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-3:
        return transform.astype(np.float32)
    return rescale_homography(affine_to_homography(transform), scale)[:2, :].astype(np.float32)


def blend_with_identity(transform: np.ndarray, intensity: float) -> np.ndarray:
    if is_affine(transform):
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return ((identity * (1.0 - intensity)) + (transform.astype(np.float32) * intensity)).astype(np.float32)
    identity = np.eye(3, dtype=np.float32)
    return _normalize_homography((identity * (1.0 - intensity)) + (transform.astype(np.float32) * intensity))


def mix_transforms(first: np.ndarray, second: np.ndarray, second_weight: float) -> np.ndarray:
    weight = max(0.0, min(1.0, second_weight))
    if is_affine(first) and is_affine(second):
        return ((first.astype(np.float32) * (1.0 - weight)) + (second.astype(np.float32) * weight)).astype(np.float32)
    mixed = (affine_to_homography(first) * (1.0 - weight)) + (affine_to_homography(second) * weight)
    return _normalize_homography(mixed)


def is_negligible(transform: np.ndarray) -> bool:
    translation = float(max(abs(transform[0, 2]), abs(transform[1, 2])))
    if is_affine(transform):
        delta = float(np.max(np.abs(transform[:, :2] - np.eye(2, dtype=np.float32))))
        return translation < NEGLIGIBLE_TRANSLATION_PX and delta < NEGLIGIBLE_AFFINE_DELTA
    delta = float(np.max(np.abs(transform - np.eye(3, dtype=np.float32))))
    return translation < NEGLIGIBLE_TRANSLATION_PX and delta < NEGLIGIBLE_HOMOGRAPHY_DELTA


def _frame_corners(width: int, height: int) -> np.ndarray:
    return np.array(
        [[[0.0, 0.0]], [[width - 1.0, 0.0]], [[width - 1.0, height - 1.0]], [[0.0, height - 1.0]]],
        dtype=np.float32,
    )


def project_corners(transform: np.ndarray, width: int, height: int) -> np.ndarray:
    """Frame corners after ``transform`` as an ``(4, 2)`` array (may contain non-finite values)."""
    corners = _frame_corners(width, height)
    if is_affine(transform):
        projected = cv2.transform(corners, transform)
    else:
        projected = cv2.perspectiveTransform(corners, transform)
    return np.asarray(projected).reshape(-1, 2)


def is_reasonable(transform: np.ndarray, frame_shape: tuple[int, ...], *, allow_perspective: bool = False) -> bool:
    """Reject estimates that would move or distort the frame implausibly."""
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return False

    projected = project_corners(transform, width, height)
    if not np.isfinite(projected).all():
        return False

    original = _frame_corners(width, height).reshape(-1, 2)
    displacement = np.linalg.norm(projected - original, axis=1)
    if float(np.median(displacement)) > (max(width, height) * MAX_MEDIAN_DISPLACEMENT_FRACTION):
        return False

    bbox_width = float(projected[:, 0].max() - projected[:, 0].min())
    bbox_height = float(projected[:, 1].max() - projected[:, 1].min())
    if bbox_width < (width * BBOX_MIN_FRACTION) or bbox_width > (width * BBOX_MAX_FRACTION):
        return False
    if bbox_height < (height * BBOX_MIN_FRACTION) or bbox_height > (height * BBOX_MAX_FRACTION):
        return False

    if allow_perspective:
        projected_area = abs(float(cv2.contourArea(projected.astype(np.float32))))
        original_area = float(max(1, width * height))
        if projected_area < (original_area * PERSPECTIVE_AREA_MIN_FRACTION):
            return False
        if projected_area > (original_area * PERSPECTIVE_AREA_MAX_FRACTION):
            return False

    return True


def fit_with_zoom_out(transform: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    """Compose a uniform scale/recentre so the warped frame stays inside the output."""
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return transform

    projected = project_corners(transform, width, height)
    if not np.isfinite(projected).all():
        return transform

    min_x, max_x = float(projected[:, 0].min()), float(projected[:, 0].max())
    min_y, max_y = float(projected[:, 1].min()), float(projected[:, 1].max())
    bbox_width = max(1.0, max_x - min_x)
    bbox_height = max(1.0, max_y - min_y)

    safe_width = max(1.0, width * ZOOM_OUT_SAFE_FRACTION)
    safe_height = max(1.0, height * ZOOM_OUT_SAFE_FRACTION)
    inside = min_x >= 0.0 and min_y >= 0.0 and max_x <= (width - 1.0) and max_y <= (height - 1.0)
    if inside and bbox_width <= safe_width and bbox_height <= safe_height:
        return transform

    fit_scale = min(1.0, safe_width / bbox_width, safe_height / bbox_height)
    bbox_center_x = (min_x + max_x) / 2.0
    bbox_center_y = (min_y + max_y) / 2.0
    output_center_x = (width - 1.0) / 2.0
    output_center_y = (height - 1.0) / 2.0
    adjust = np.array(
        [
            [fit_scale, 0.0, output_center_x - (bbox_center_x * fit_scale)],
            [0.0, fit_scale, output_center_y - (bbox_center_y * fit_scale)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    composed = _normalize_homography(adjust @ affine_to_homography(transform))
    if is_affine(transform):
        return composed[:2, :].astype(np.float32)
    return composed


def warp(frame: np.ndarray, transform: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if is_affine(transform):
        return cv2.warpAffine(frame, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.warpPerspective(frame, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def auto_crop(frame: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Crop away reflected borders introduced by ``warp`` and scale back to the original size."""
    height, width = frame.shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    warp_fn = cv2.warpAffine if is_affine(transform) else cv2.warpPerspective
    valid_mask = warp_fn(
        mask,
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    erosion_size = max(1, round(min(width, height) * AUTO_CROP_EROSION_FRACTION))
    if erosion_size > 1:
        kernel = np.ones((erosion_size, erosion_size), dtype=np.uint8)
        valid_mask = cv2.erode(valid_mask, kernel, iterations=1)

    non_zero = cv2.findNonZero(valid_mask)
    if non_zero is None:
        return frame

    left, top, cropped_width, cropped_height = cv2.boundingRect(non_zero)
    if cropped_width >= (width - 2) and cropped_height >= (height - 2):
        return frame
    min_width = max(16, int(width * AUTO_CROP_MIN_FRACTION))
    min_height = max(16, int(height * AUTO_CROP_MIN_FRACTION))
    if cropped_width < min_width or cropped_height < min_height:
        return frame

    cropped = frame[top : top + cropped_height, left : left + cropped_width]
    if cropped.size == 0:
        return frame
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)
