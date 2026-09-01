"""Motion estimators: inter-frame (auto modes) and reference-locked (camera lock).

The inter-frame estimate comes from :mod:`global_motion` (grid features, rigid-motion hypotheses
chosen by spread, cell memory), the lock from :mod:`lock` (tracks carried by Lucas-Kanade and
re-anchored with ORB). This module keeps the analysis helpers, the reference-feature cache and
the public entry points the stabilizer and the tests use.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from space_video_enhancer.pipeline.stabilization import transforms
from space_video_enhancer.pipeline.stabilization.global_motion import estimate_global_motion, new_cell_memory
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import NormalizedRect, PixelRect, StabilizationMode
from space_video_enhancer.telemetry.trace import note

# --- inter-frame analysis -------------------------------------------------------------------
ANALYSIS_MAX_DIMENSION = 480.0
ANALYSIS_BLUR_KERNEL = (5, 5)

PHASE_CORRELATE_MIN_RESPONSE = 0.05

SIMILARITY_MIN_POINTS = 6
SIMILARITY_RANSAC = {
    "method": cv2.RANSAC,
    "ransacReprojThreshold": 3.0,
    "maxIters": 3000,
    "confidence": 0.99,
    "refineIters": 10,
}

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
REFERENCE_CACHE_SIZE = 4


@dataclass(frozen=True)
class ReferenceFeatures:
    """ORB keypoints of a camera-lock reference inside its focus region, computed once and reused.

    The reference does not change between frames, so detecting its features every frame was
    ~40% of the camera-lock cost. Entries are keyed by the reference array identity, the focus
    region in pixels and the analysis scale; the array is kept alive by the entry.
    """

    frame: np.ndarray
    gray: np.ndarray
    points: np.ndarray
    """``(N, 2)`` float32 keypoint positions in ``gray`` coordinates."""
    descriptors: np.ndarray | None


_reference_cache: OrderedDict[tuple[Any, ...], ReferenceFeatures] = OrderedDict()
_reference_cache_lock = threading.Lock()
reference_cache_stats = {"hits": 0, "misses": 0}


def reference_features(reference_frame: np.ndarray, reference_pixels: PixelRect, scale: float) -> ReferenceFeatures:
    key = (id(reference_frame), reference_frame.shape, reference_pixels, round(scale, 6))
    with _reference_cache_lock:
        cached = _reference_cache.get(key)
        if cached is not None and cached.frame is reference_frame:
            _reference_cache.move_to_end(key)
            reference_cache_stats["hits"] += 1
            return cached

    gray = scaled_gray(reference_frame, scale)
    mask = np.zeros_like(gray, dtype=np.uint8)
    ref_x, ref_y, ref_w, ref_h = scaled_rect(reference_pixels, scale, gray.shape[1], gray.shape[0])
    mask[ref_y : ref_y + ref_h, ref_x : ref_x + ref_w] = 255
    keypoints, descriptors = cv2.ORB_create(**ORB_PARAMS).detectAndCompute(gray, mask)
    points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32).reshape(-1, 2)
    features = ReferenceFeatures(reference_frame, gray, points, descriptors)

    with _reference_cache_lock:
        reference_cache_stats["misses"] += 1
        _reference_cache[key] = features
        while len(_reference_cache) > REFERENCE_CACHE_SIZE:
            _reference_cache.popitem(last=False)
    return features


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


def _memory_of(state: StabilizationState | None) -> np.ndarray | None:
    """The state's cell memory (created on first use), or ``None`` for a stateless estimate."""
    if state is None:
        return None
    if state.cell_memory is None:
        state.cell_memory = new_cell_memory()
    return state.cell_memory


def translation_from_gray(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    scale: float,
    state: StabilizationState | None = None,
) -> tuple[float, float] | None:
    """Global translation (full-res pixels) that moves ``current_gray`` back onto ``previous_gray``."""
    motion = estimate_global_motion(previous_gray, current_gray, StabilizationMode.AUTO, _memory_of(state))
    if state is not None:
        state.last_vectors = None if motion is None else motion.vectors
    if motion is not None:
        note("stabilization.estimator", "lk_translation")
        return (float(motion.transform[0, 2]) / scale, float(motion.transform[1, 2]) / scale)

    phase_shift, response = cv2.phaseCorrelate(previous_gray.astype(np.float32), current_gray.astype(np.float32))
    note("stabilization.phase_response", round(float(response), 3))
    if response >= PHASE_CORRELATE_MIN_RESPONSE:
        note("stabilization.estimator", "phase_correlate")
        return (-float(phase_shift[0]) / scale, -float(phase_shift[1]) / scale)
    note("stabilization.estimator", "none")
    return None


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
        note("stabilization.estimator", "cached")
        return state.last_shift

    if state.prev_gray is None or state.breaks_sequence(frame_index):
        note("stabilization.estimator", "primed" if state.prev_gray is None else "sequence_break")
        state.cell_memory = None  # a new take: what was background may not be any more
        state.remember_gray(current_gray, frame_index)
        state.remember_shift(None, frame_index)
        return None

    shift = translation_from_gray(state.prev_gray, current_gray, scale, state)
    state.remember_gray(current_gray, frame_index)
    state.remember_shift(shift, frame_index)
    return shift


def estimate_inter_frame_transform(
    frame: np.ndarray,
    frame_index: int | None,
    state: StabilizationState,
    mode: StabilizationMode,
    *,
    translation_fallback: bool = False,
) -> np.ndarray | None:
    """Transform that aligns ``frame`` onto the remembered previous frame, in full-res pixels.

    Translation only for :attr:`StabilizationMode.AUTO`, similarity or homography otherwise.
    ``translation_fallback`` tries a plain translation when the geometric fit fails. Returns
    ``None`` on the first frame of a sequence, after a jump, or when nothing could be tracked.
    """
    current_gray, scale = analysis_gray(frame)
    if current_gray.size == 0:
        return None

    if (
        frame_index is not None
        and state.last_transform_index == frame_index
        and isinstance(state.last_transform, np.ndarray)
    ):
        note("stabilization.estimator", "cached")
        return state.last_transform

    if state.prev_gray is None or state.breaks_sequence(frame_index):
        note("stabilization.estimator", "primed" if state.prev_gray is None else "sequence_break")
        state.cell_memory = None  # a new take: what was background may not be any more
        state.remember_gray(current_gray, frame_index)
        state.remember_transform(None, frame_index)
        return None

    transform: np.ndarray | None
    if mode is StabilizationMode.AUTO:
        shift = translation_from_gray(state.prev_gray, current_gray, scale, state)
        transform = None if shift is None else transforms.translation_to_affine(shift[0], shift[1])
    else:
        transform = geometric_transform_from_gray(state.prev_gray, current_gray, scale, frame.shape, mode, state)
        if transform is None and translation_fallback:
            shift = translation_from_gray(state.prev_gray, current_gray, scale, state)
            transform = None if shift is None else transforms.translation_to_affine(shift[0], shift[1])
    state.remember_gray(current_gray, frame_index)
    state.remember_transform(transform, frame_index)
    return transform


def geometric_transform_from_gray(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    scale: float,
    frame_shape: tuple[int, ...],
    mode: StabilizationMode,
    state: StabilizationState | None = None,
) -> np.ndarray | None:
    """Similarity (or homography in perspective mode) aligning ``current_gray`` onto ``previous_gray``, full-res."""
    motion = estimate_global_motion(previous_gray, current_gray, mode, _memory_of(state))
    if state is not None:
        state.last_vectors = None if motion is None else motion.vectors
    if motion is None:
        note("stabilization.estimator", "none")
        return None
    note("stabilization.estimator", "lk_perspective" if mode is StabilizationMode.PERSPECTIVE else "lk_similarity")
    if transforms.is_affine(motion.transform):
        scaled = transforms.rescale_affine(motion.transform, scale)
        if not transforms.is_reasonable(scaled, frame_shape):
            note("stabilization.rejected", "unreasonable_similarity")
            return None
        return scaled
    homography = transforms.rescale_homography(motion.transform, scale)
    if not transforms.is_reasonable(homography, frame_shape, allow_perspective=True):
        note("stabilization.rejected", "unreasonable_perspective")
        return None
    return homography


def estimate_similarity_transform(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    scale: float,
    frame_shape: tuple[int, ...],
) -> np.ndarray | None:
    if len(source_points) < SIMILARITY_MIN_POINTS or len(destination_points) < SIMILARITY_MIN_POINTS:
        return None
    transform, inliers = cv2.estimateAffinePartial2D(source_points, destination_points, **SIMILARITY_RANSAC)
    inlier_count = 0 if inliers is None else int(np.count_nonzero(inliers))
    note("stabilization.inliers", inlier_count)
    if transform is None or inlier_count < SIMILARITY_MIN_POINTS:
        note("stabilization.rejected", "too_few_inliers")
        return None
    scaled = transforms.rescale_affine(transform, scale)
    if not transforms.is_reasonable(scaled, frame_shape):
        note("stabilization.rejected", "unreasonable_similarity")
        return None
    return scaled


def estimate_camera_lock_transform(
    frame: np.ndarray,
    reference_frame: np.ndarray,
    focus_rect: NormalizedRect,
    state: StabilizationState | None = None,
    frame_index: int | None = None,
) -> np.ndarray | None:
    """Transform that maps ``frame`` back onto ``reference_frame`` around the focus region (see :mod:`lock`)."""
    from space_video_enhancer.pipeline.stabilization import lock  # noqa: PLC0415  (lock builds on this module)

    return lock.lock_transform(frame, reference_frame, focus_rect, state, frame_index)


def estimate_object_lock_transform(
    frame: np.ndarray,
    reference_frame: np.ndarray,
    regions: tuple[NormalizedRect, ...],
    state: StabilizationState | None = None,
    frame_index: int | None = None,
) -> np.ndarray | None:
    """Transform that holds the union of the tracked regions to the reference (see :mod:`lock`)."""
    from space_video_enhancer.pipeline.stabilization import lock  # noqa: PLC0415  (lock builds on this module)

    return lock.object_lock_transform(frame, reference_frame, regions, state, frame_index)


def orb_matches(
    reference: ReferenceFeatures, current_gray: np.ndarray, current_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(reference points, current points)`` of the ratio-tested ORB matches, best first; ``None`` when too few."""
    if reference.descriptors is None or len(reference.points) < ORB_MIN_KEYPOINTS:
        return None
    current_keypoints, current_descriptors = cv2.ORB_create(**ORB_PARAMS).detectAndCompute(current_gray, current_mask)
    if current_descriptors is None or len(current_keypoints) < ORB_MIN_KEYPOINTS:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good_matches = [
        pair[0]
        for pair in matcher.knnMatch(reference.descriptors, current_descriptors, k=2)
        if len(pair) >= 2 and pair[0].distance < (ORB_RATIO_TEST * pair[1].distance)
    ]
    note("stabilization.good_matches", len(good_matches))
    if len(good_matches) < ORB_MIN_GOOD_MATCHES:
        return None

    good_matches.sort(key=lambda match: match.distance)
    good_matches = good_matches[:ORB_MAX_MATCHES]
    reference_points = reference.points[[match.queryIdx for match in good_matches]]
    current_points = np.array([current_keypoints[match.trainIdx].pt for match in good_matches], dtype=np.float32)
    return reference_points, current_points


def camera_lock_feature_transform(
    reference: ReferenceFeatures,
    current_gray: np.ndarray,
    current_mask: np.ndarray | None,
    scale: float,
    frame_shape: tuple[int, ...],
) -> np.ndarray | None:
    matches = orb_matches(reference, current_gray, current_mask)
    if matches is None:
        return None
    reference_points, current_points = matches
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
    note("stabilization.template_score", round(float(max_value), 3))
    if max_value < TEMPLATE_MIN_SCORE:
        return None

    match_center_x = search_left + max_loc[0] + (ref_w / 2.0)
    match_center_y = search_top + max_loc[1] + (ref_h / 2.0)
    reference_center_x = ref_x + (ref_w / 2.0)
    reference_center_y = ref_y + (ref_h / 2.0)
    return (reference_center_x - match_center_x, reference_center_y - match_center_y)
