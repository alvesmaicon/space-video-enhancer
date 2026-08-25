"""Motion estimators: inter-frame (auto modes) and reference-locked (camera lock)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from space_video_enhancer.pipeline.stabilization import transforms
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import NormalizedRect, PixelRect, StabilizationMode

# --- inter-frame analysis -------------------------------------------------------------------
ANALYSIS_MAX_DIMENSION = 480.0
ANALYSIS_BLUR_KERNEL = (5, 5)

TRANSLATION_GFTT = {"maxCorners": 160, "qualityLevel": 0.01, "minDistance": 18, "blockSize": 5}
TRANSLATION_MIN_FEATURES = 8
TRANSLATION_MIN_TRACKED = 6
PHASE_CORRELATE_MIN_RESPONSE = 0.05

GEOMETRIC_GFTT = {"maxCorners": 220, "qualityLevel": 0.01, "minDistance": 14, "blockSize": 5}
GEOMETRIC_MIN_FEATURES = 10
GEOMETRIC_MIN_TRACKED = 8
LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
}

SIMILARITY_MIN_POINTS = 6
SIMILARITY_RANSAC = {
    "method": cv2.RANSAC,
    "ransacReprojThreshold": 3.0,
    "maxIters": 3000,
    "confidence": 0.99,
    "refineIters": 10,
}
PERSPECTIVE_MIN_POINTS = 8
PERSPECTIVE_RANSAC_THRESHOLD = 3.0

# --- camera lock ----------------------------------------------------------------------------
CAMERA_LOCK_MAX_DIMENSION = 720.0
CAMERA_LOCK_SEARCH_EXPANSION = 2.6
ORB_PARAMS = {
    "nfeatures": 900,
    "scaleFactor": 1.2,
    "nlevels": 8,
    "edgeThreshold": 15,
    "patchSize": 31,
    "fastThreshold": 10,
}
ORB_MIN_KEYPOINTS = 8
ORB_RATIO_TEST = 0.78
ORB_MIN_GOOD_MATCHES = 8
ORB_MAX_MATCHES = 120

TEMPLATE_SEARCH_MARGIN_FACTOR = 1.6
TEMPLATE_SEARCH_MIN_MARGIN = 32
TEMPLATE_MIN_SCORE = 0.28
SEARCH_RECT_MIN_MARGIN = 40


def analysis_gray(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscaled, blurred grayscale used by the inter-frame estimators; returns (gray, scale)."""
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return np.empty((0, 0), dtype=np.uint8), 1.0

    scale = min(1.0, ANALYSIS_MAX_DIMENSION / max(width, height))
    resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 0.999 else frame
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, ANALYSIS_BLUR_KERNEL, 0), scale


def scaled_gray(frame: np.ndarray, scale: float) -> np.ndarray:
    resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 0.999 else frame
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def scaled_rect(rect: PixelRect, scale: float, width: int, height: int) -> PixelRect:
    x, y, rect_width, rect_height = rect
    scaled_x = max(0, min(round(x * scale), max(0, width - 2)))
    scaled_y = max(0, min(round(y * scale), max(0, height - 2)))
    scaled_width = max(2, min(round(rect_width * scale), width - scaled_x))
    scaled_height = max(2, min(round(rect_height * scale), height - scaled_y))
    return (scaled_x, scaled_y, scaled_width, scaled_height)


def expanded_search_rect(rect: PixelRect, width: int, height: int, multiplier: float) -> PixelRect:
    x, y, rect_width, rect_height = rect
    margin_x = max(SEARCH_RECT_MIN_MARGIN, int(rect_width * multiplier))
    margin_y = max(SEARCH_RECT_MIN_MARGIN, int(rect_height * multiplier))
    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(width, x + rect_width + margin_x)
    bottom = min(height, y + rect_height + margin_y)
    return (left, top, max(2, right - left), max(2, bottom - top))


def _track_features(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    gftt: Mapping[str, Any],
    min_features: int,
    min_tracked: int,
    lk_params: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """GFTT + Lucas-Kanade; returns (previous_points, current_points) or ``None``."""
    features = cv2.goodFeaturesToTrack(previous_gray, **gftt)
    if features is None or len(features) < min_features:
        return None
    next_points, status, _errors = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, features, None, **(lk_params or {})
    )
    if next_points is None or status is None:
        return None
    valid_mask = status.reshape(-1) == 1
    if np.count_nonzero(valid_mask) < min_tracked:
        return None
    return features.reshape(-1, 2)[valid_mask], next_points.reshape(-1, 2)[valid_mask]


def estimate_translation_shift(
    frame: np.ndarray,
    frame_index: int | None,
    state: StabilizationState,
) -> tuple[float, float] | None:
    """Global translation between the remembered previous frame and ``frame`` (full-res pixels)."""
    current_gray, scale = analysis_gray(frame)
    if current_gray.size == 0:
        return None

    if frame_index is not None and state.last_shift_index == frame_index and isinstance(state.last_shift, tuple):
        return state.last_shift

    if state.prev_gray is None or state.breaks_sequence(frame_index):
        state.remember_gray(current_gray, frame_index)
        state.remember_shift(None, frame_index)
        return None

    shift: tuple[float, float] | None = None
    tracked = _track_features(
        state.prev_gray, current_gray, TRANSLATION_GFTT, TRANSLATION_MIN_FEATURES, TRANSLATION_MIN_TRACKED
    )
    if tracked is not None:
        deltas = tracked[1] - tracked[0]
        shift = (-float(np.median(deltas[:, 0])) / scale, -float(np.median(deltas[:, 1])) / scale)

    if shift is None:
        phase_shift, response = cv2.phaseCorrelate(state.prev_gray.astype(np.float32), current_gray.astype(np.float32))
        if response >= PHASE_CORRELATE_MIN_RESPONSE:
            shift = (-float(phase_shift[0]) / scale, -float(phase_shift[1]) / scale)

    state.remember_gray(current_gray, frame_index)
    state.remember_shift(shift, frame_index)
    return shift


def estimate_geometric_transform(
    frame: np.ndarray,
    frame_index: int | None,
    state: StabilizationState,
    mode: StabilizationMode,
) -> np.ndarray | None:
    """Similarity or perspective transform between the remembered previous frame and ``frame``."""
    current_gray, scale = analysis_gray(frame)
    if current_gray.size == 0:
        return None

    if (
        frame_index is not None
        and state.last_transform_index == frame_index
        and isinstance(state.last_transform, np.ndarray)
    ):
        return state.last_transform

    if state.prev_gray is None or state.breaks_sequence(frame_index):
        state.remember_gray(current_gray, frame_index)
        state.remember_transform(None, frame_index)
        return None

    transform = geometric_transform_from_gray(state.prev_gray, current_gray, scale, frame.shape, mode)
    state.remember_gray(current_gray, frame_index)
    state.remember_transform(transform, frame_index)
    return transform


def geometric_transform_from_gray(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    scale: float,
    frame_shape: tuple[int, ...],
    mode: StabilizationMode,
) -> np.ndarray | None:
    tracked = _track_features(
        previous_gray, current_gray, GEOMETRIC_GFTT, GEOMETRIC_MIN_FEATURES, GEOMETRIC_MIN_TRACKED, LK_PARAMS
    )
    if tracked is None:
        return None
    previous_points, current_points = tracked

    if mode is StabilizationMode.PERSPECTIVE:
        perspective = estimate_perspective_transform(current_points, previous_points, scale, frame_shape)
        if perspective is not None:
            return perspective
    return estimate_similarity_transform(current_points, previous_points, scale, frame_shape)


def estimate_similarity_transform(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    scale: float,
    frame_shape: tuple[int, ...],
) -> np.ndarray | None:
    if len(source_points) < SIMILARITY_MIN_POINTS or len(destination_points) < SIMILARITY_MIN_POINTS:
        return None
    transform, inliers = cv2.estimateAffinePartial2D(source_points, destination_points, **SIMILARITY_RANSAC)
    if transform is None or inliers is None or int(np.count_nonzero(inliers)) < SIMILARITY_MIN_POINTS:
        return None
    scaled = transforms.rescale_affine(transform, scale)
    return scaled if transforms.is_reasonable(scaled, frame_shape) else None


def estimate_perspective_transform(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    scale: float,
    frame_shape: tuple[int, ...],
) -> np.ndarray | None:
    if len(source_points) < PERSPECTIVE_MIN_POINTS or len(destination_points) < PERSPECTIVE_MIN_POINTS:
        return None
    transform, inliers = cv2.findHomography(source_points, destination_points, cv2.RANSAC, PERSPECTIVE_RANSAC_THRESHOLD)
    if transform is None or inliers is None or int(np.count_nonzero(inliers)) < PERSPECTIVE_MIN_POINTS:
        return None
    scaled = transforms.rescale_homography(transform, scale)
    return scaled if transforms.is_reasonable(scaled, frame_shape, allow_perspective=True) else None


def estimate_camera_lock_transform(
    frame: np.ndarray,
    reference_frame: np.ndarray,
    focus_rect: NormalizedRect,
) -> np.ndarray | None:
    """Transform that maps ``frame`` back onto ``reference_frame`` around the focus region."""
    reference_height, reference_width = reference_frame.shape[:2]
    current_height, current_width = frame.shape[:2]
    reference_pixels = focus_rect.to_pixels(reference_width, reference_height)
    current_pixels = focus_rect.to_pixels(current_width, current_height)
    if reference_pixels is None or current_pixels is None:
        return None

    scale = min(1.0, CAMERA_LOCK_MAX_DIMENSION / max(reference_width, reference_height, current_width, current_height))
    reference_gray = scaled_gray(reference_frame, scale)
    current_gray = scaled_gray(frame, scale)
    if reference_gray.size == 0 or current_gray.size == 0:
        return None

    reference_mask = np.zeros_like(reference_gray, dtype=np.uint8)
    current_mask = np.zeros_like(current_gray, dtype=np.uint8)
    ref_x, ref_y, ref_w, ref_h = scaled_rect(reference_pixels, scale, reference_gray.shape[1], reference_gray.shape[0])
    search_rect = expanded_search_rect(current_pixels, current_width, current_height, CAMERA_LOCK_SEARCH_EXPANSION)
    cur_x, cur_y, cur_w, cur_h = scaled_rect(search_rect, scale, current_gray.shape[1], current_gray.shape[0])
    reference_mask[ref_y : ref_y + ref_h, ref_x : ref_x + ref_w] = 255
    current_mask[cur_y : cur_y + cur_h, cur_x : cur_x + cur_w] = 255

    for mask in (current_mask, None):
        transform = camera_lock_feature_transform(
            reference_gray, current_gray, reference_mask, mask, scale, frame.shape
        )
        if transform is not None:
            return transform

    shift = estimate_template_shift(frame, reference_frame, focus_rect)
    if shift is None:
        return None
    return transforms.translation_to_affine(shift[0], shift[1])


def camera_lock_feature_transform(
    reference_gray: np.ndarray,
    current_gray: np.ndarray,
    reference_mask: np.ndarray,
    current_mask: np.ndarray | None,
    scale: float,
    frame_shape: tuple[int, ...],
) -> np.ndarray | None:
    orb = cv2.ORB_create(**ORB_PARAMS)
    reference_keypoints, reference_descriptors = orb.detectAndCompute(reference_gray, reference_mask)
    current_keypoints, current_descriptors = orb.detectAndCompute(current_gray, current_mask)
    if (
        reference_descriptors is None
        or current_descriptors is None
        or len(reference_keypoints) < ORB_MIN_KEYPOINTS
        or len(current_keypoints) < ORB_MIN_KEYPOINTS
    ):
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good_matches = [
        pair[0]
        for pair in matcher.knnMatch(reference_descriptors, current_descriptors, k=2)
        if len(pair) >= 2 and pair[0].distance < (ORB_RATIO_TEST * pair[1].distance)
    ]
    if len(good_matches) < ORB_MIN_GOOD_MATCHES:
        return None

    good_matches.sort(key=lambda match: match.distance)
    good_matches = good_matches[:ORB_MAX_MATCHES]
    reference_points = np.array([reference_keypoints[match.queryIdx].pt for match in good_matches], dtype=np.float32)
    current_points = np.array([current_keypoints[match.trainIdx].pt for match in good_matches], dtype=np.float32)
    return estimate_similarity_transform(current_points, reference_points, scale, frame_shape)


def estimate_template_shift(
    frame: np.ndarray,
    reference_frame: np.ndarray,
    focus_rect: NormalizedRect,
) -> tuple[float, float] | None:
    """Translation that re-centres the focus region on its position in ``reference_frame``."""
    reference_height, reference_width = reference_frame.shape[:2]
    current_height, current_width = frame.shape[:2]
    reference_pixels = focus_rect.to_pixels(reference_width, reference_height)
    current_pixels = focus_rect.to_pixels(current_width, current_height)
    if reference_pixels is None or current_pixels is None:
        return None

    ref_x, ref_y, ref_w, ref_h = reference_pixels
    cur_x, cur_y, cur_w, cur_h = current_pixels
    template = reference_frame[ref_y : ref_y + ref_h, ref_x : ref_x + ref_w]
    if template.size == 0:
        return None

    margin_x = max(TEMPLATE_SEARCH_MIN_MARGIN, int(cur_w * TEMPLATE_SEARCH_MARGIN_FACTOR))
    margin_y = max(TEMPLATE_SEARCH_MIN_MARGIN, int(cur_h * TEMPLATE_SEARCH_MARGIN_FACTOR))
    search_left = max(0, cur_x - margin_x)
    search_top = max(0, cur_y - margin_y)
    search_right = min(current_width, cur_x + cur_w + margin_x)
    search_bottom = min(current_height, cur_y + cur_h + margin_y)
    search_region = frame[search_top:search_bottom, search_left:search_right]
    if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
        return None

    result = cv2.matchTemplate(
        cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(template, cv2.COLOR_BGR2GRAY),
        cv2.TM_CCOEFF_NORMED,
    )
    _, max_value, _, max_loc = cv2.minMaxLoc(result)
    if max_value < TEMPLATE_MIN_SCORE:
        return None

    match_center_x = search_left + max_loc[0] + (ref_w / 2.0)
    match_center_y = search_top + max_loc[1] + (ref_h / 2.0)
    reference_center_x = ref_x + (ref_w / 2.0)
    reference_center_y = ref_y + (ref_h / 2.0)
    return (reference_center_x - match_center_x, reference_center_y - match_center_y)
