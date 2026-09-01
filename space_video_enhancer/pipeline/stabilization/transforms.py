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
CROP_EROSION_FRACTION = 0.015
CROP_MIN_FRACTION = 0.55
CROP_SAFETY_PX = 2.0


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


MARGIN_FIT_ITERATIONS = 20


def fit_within_margin(transform: np.ndarray, frame_shape: tuple[int, ...], margin_fraction: float) -> np.ndarray:
    """The largest identity-blend of ``transform`` whose warp still fills the fixed crop margin.

    The automatic crop promises a constant zoom: the same inset on every frame. The translation cap
    keeps pure shifts inside it, but rotation, zoom and keystone move the corners further — measured
    on real walking footage the correction escaped the margin on half the frames, the crop fell back
    to that frame's valid area and the output size pulsed by up to 4% between consecutive frames.
    Instead, the correction saturates: blend toward the identity (which always fits) just enough to
    fit again. Returns ``transform`` itself when it already fits.
    """
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return transform
    margin_x = round(width * margin_fraction)
    margin_y = round(height * margin_fraction)
    rect = (margin_x, margin_y, max(1, width - 2 * margin_x), max(1, height - 2 * margin_y))
    if rect_inside_projection(rect, transform, width, height):
        return transform
    identity = identity_like(transform)
    fits, escapes = 0.0, 1.0  # the identity always fits; the full correction just failed
    for _ in range(MARGIN_FIT_ITERATIONS):
        middle = (fits + escapes) / 2.0
        if rect_inside_projection(rect, mix_transforms(identity, transform, middle), width, height):
            fits = middle
        else:
            escapes = middle
    return mix_transforms(identity, transform, fits)


def warp(frame: np.ndarray, transform: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if is_affine(transform):
        return cv2.warpAffine(frame, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.warpPerspective(frame, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def valid_area_rect(transform: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    """Bounding rect of the pixels ``warp`` fills with real content (reflected borders excluded)."""
    height, width = frame_shape[:2]
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
    erosion_size = max(1, round(min(width, height) * CROP_EROSION_FRACTION))
    if erosion_size > 1:
        kernel = np.ones((erosion_size, erosion_size), dtype=np.uint8)
        valid_mask = cv2.erode(valid_mask, kernel, iterations=1)
    non_zero = cv2.findNonZero(valid_mask)
    if non_zero is None:
        return None
    left, top, rect_width, rect_height = cv2.boundingRect(non_zero)
    return (int(left), int(top), int(rect_width), int(rect_height))


def rect_inside_projection(rect: tuple[int, int, int, int], transform: np.ndarray, width: int, height: int) -> bool:
    """True when every corner of ``rect`` lands on real content after ``warp`` (with a small safety margin)."""
    projected = project_corners(transform, width, height)
    if not np.isfinite(projected).all():
        return False
    contour = projected.astype(np.float32).reshape(-1, 1, 2)
    x, y, rect_width, rect_height = rect
    corners = ((x, y), (x + rect_width - 1, y), (x + rect_width - 1, y + rect_height - 1), (x, y + rect_height - 1))
    return all(
        cv2.pointPolygonTest(contour, (float(corner_x), float(corner_y)), True) >= CROP_SAFETY_PX
        for corner_x, corner_y in corners
    )


def crop_rect_for(
    transform: np.ndarray | None, frame_shape: tuple[int, ...], margin_fraction: float
) -> tuple[int, int, int, int]:
    """The crop that hides stabilization borders: a fixed inset of ``margin_fraction`` per side.

    The inset is the same for every frame, so the zoom does not breathe with the correction. It
    only grows (toward the actually valid area) when a correction exceeds the margin, which the
    inter-frame modes never do because their translation is capped at the same fraction.
    """
    height, width = frame_shape[:2]
    margin_x = round(width * margin_fraction)
    margin_y = round(height * margin_fraction)
    rect = (margin_x, margin_y, max(1, width - 2 * margin_x), max(1, height - 2 * margin_y))
    if transform is None or rect_inside_projection(rect, transform, width, height):
        return rect

    valid = valid_area_rect(transform, frame_shape)
    if valid is None:
        return rect
    left = max(rect[0], valid[0])
    top = max(rect[1], valid[1])
    right = min(rect[0] + rect[2], valid[0] + valid[2])
    bottom = min(rect[1] + rect[3], valid[1] + valid[3])
    min_width = max(16, int(width * CROP_MIN_FRACTION))
    min_height = max(16, int(height * CROP_MIN_FRACTION))
    if right - left < min_width or bottom - top < min_height:
        return rect
    return (left, top, right - left, bottom - top)


def stabilization_crop(frame: np.ndarray, transform: np.ndarray | None, margin_fraction: float) -> np.ndarray:
    """Apply :func:`crop_rect_for` and scale back to the original size."""
    height, width = frame.shape[:2]
    x, y, rect_width, rect_height = crop_rect_for(transform, frame.shape, margin_fraction)
    if x == 0 and y == 0 and rect_width >= width and rect_height >= height:
        return frame
    cropped = frame[y : y + rect_height, x : x + rect_width]
    if cropped.size == 0:
        return frame
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)


def identity_like(transform: np.ndarray) -> np.ndarray:
    """Identity in the same shape (affine or homography) as ``transform``."""
    if is_affine(transform):
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return np.eye(3, dtype=np.float32)


def compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """``outer ∘ inner``: apply ``inner`` first, then ``outer``. Stays affine when both are."""
    composed = affine_to_homography(outer).astype(np.float64) @ affine_to_homography(inner).astype(np.float64)
    if is_affine(outer) and is_affine(inner):
        return composed[:2, :].astype(np.float32)
    return _normalize_homography(composed)


def clamp_translation(transform: np.ndarray, frame_shape: tuple[int, ...], fraction: float) -> np.ndarray:
    """Limit the translation to ``fraction`` of the frame size in each axis."""
    height, width = frame_shape[:2]
    clamped = transform.astype(np.float32).copy()
    clamped[0, 2] = float(np.clip(clamped[0, 2], -width * fraction, width * fraction))
    clamped[1, 2] = float(np.clip(clamped[1, 2], -height * fraction, height * fraction))
    return clamped
