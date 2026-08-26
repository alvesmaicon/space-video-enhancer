"""Camera lock that survives blur and gradual distortion: track the region's points, re-anchor when sharp.

The original lock matched ORB descriptors of the reference region on every frame. ORB needs
crisp corners: a defocused or motion-blurred frame has none, the match fails and the lock is
lost exactly when the camera is moving most. Descriptors also only fit a similarity, so a tilt
or a rolling-shutter skew biased the estimate.

Here the reference correspondences are carried frame to frame with Lucas-Kanade, which tracks
blobs rather than corners and follows a slowly distorting region; the fit is always reference ->
current, so drift only enters through the tracks themselves. To bound that drift the set is
re-anchored with fresh ORB matches whenever they are available and agree with the tracks (every
few frames, or as soon as the tracked set thins out). When ORB finds nothing (the whole take is
blurred) the tracks are seeded by Lucas-Kanade straight from the reference, starting from the
shift template matching suggests: a lock is by definition near the reference pose. Seeding and
anchoring compare like with like: the reference is blurred until its region is as sharp as the
current frame's, otherwise a crisp corner and its smeared twin never line up. With enough
well-spread points the fit is a homography, so a tilted region lands where it should. Template
matching alone remains the last resort.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from space_video_enhancer.pipeline.stabilization import estimators, transforms
from space_video_enhancer.pipeline.stabilization.global_motion import grid_features, track_points
from space_video_enhancer.pipeline.stabilization.state import StabilizationState, frame_gap
from space_video_enhancer.settings import NormalizedRect
from space_video_enhancer.telemetry.trace import note

LOCK_LK_PARAMS = {
    "winSize": (31, 31),
    "maxLevel": 4,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
}
"""Wider windows and one more pyramid level than the global estimator: blurred regions have soft gradients."""
LOCK_FORWARD_BACKWARD_PX = 2.0
"""Blur widens the round-trip error of a good track; the global estimator's 1 px would drop them all."""
LOCK_MIN_POINTS = 4
"""A similarity needs two correspondences; four RANSAC inliers with a small residual are a sound fit."""
SEED_FEATURES_PER_CELL = 6
LOCK_REANCHOR_INTERVAL = 4
"""Frames between re-anchoring attempts (ORB, else a Lucas-Kanade seed from the reference) while tracks are healthy."""
RESIDUAL_TOLERANCE = 1.5
"""A drift-free anchor replaces the carried tracks unless its fit residual is this much worse than theirs."""
BLUR_MATCH_THRESHOLD = 0.7
"""The reference is blurred to match only when the current region is clearly softer (sharpness ratio below this)."""
LOCK_RANSAC_PX = 3.0
HOMOGRAPHY_MIN_POINTS = 12
HOMOGRAPHY_MIN_EXTENT = 0.28
"""A homography is fitted only when the points span this fraction of the frame in both axes."""
BLUR_MATCH_SIGMAS = (0.0, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5)
"""Blur levels tried on the reference to match the current frame's sharpness (variance of the Laplacian)."""


@dataclass
class LockTrack:
    """Reference <-> last-frame correspondences carried across frames (analysis coordinates)."""

    reference_id: int
    scale: float
    reference_points: np.ndarray
    current_points: np.ndarray
    prev_gray: np.ndarray
    prev_index: int | None
    frames_since_anchor: int = 0

    def continues(self, reference_frame: np.ndarray, scale: float, frame_index: int | None) -> bool:
        if self.reference_id != id(reference_frame) or abs(self.scale - scale) > 1e-9:
            return False
        if frame_index is None or self.prev_index is None:
            return True
        return frame_gap(self.prev_index, frame_index) is not None


def lock_transform(
    frame: np.ndarray,
    reference_frame: np.ndarray,
    focus_rect: NormalizedRect,
    state: StabilizationState | None = None,
    frame_index: int | None = None,
) -> np.ndarray | None:
    """Transform (full-res pixels) mapping ``frame`` back onto ``reference_frame`` around the focus region."""
    reference_height, reference_width = reference_frame.shape[:2]
    current_height, current_width = frame.shape[:2]
    reference_pixels = focus_rect.to_pixels(reference_width, reference_height)
    current_pixels = focus_rect.to_pixels(current_width, current_height)
    if reference_pixels is None or current_pixels is None:
        return None
    scale = min(
        1.0,
        estimators.CAMERA_LOCK_MAX_DIMENSION / max(reference_width, reference_height, current_width, current_height),
    )
    reference = estimators.reference_features(reference_frame, reference_pixels, scale)
    current_gray = estimators.scaled_gray(frame, scale)
    if reference.gray.size == 0 or current_gray.size == 0:
        return None

    track = state.lock if state is not None else None
    if track is not None and not track.continues(reference_frame, scale, frame_index):
        track = None
    tracked_reference = tracked_current = np.empty((0, 2), dtype=np.float32)
    if track is not None:
        tracked_reference, tracked_current = _advance_tracks(track, current_gray)
    note("stabilization.lock_tracked", len(tracked_current))
    tracks_fit = None
    if len(tracked_current) >= LOCK_MIN_POINTS:
        tracks_fit = _fit(tracked_current, tracked_reference, scale, frame.shape)

    # Drift-free candidates: fresh ORB matches, else corners tracked straight from the (blur-matched) reference.
    anchor_fit = None
    anchor_kind = ""
    anchor_points: tuple[np.ndarray, np.ndarray] | None = None
    shift: tuple[float, float] | None = None
    if tracks_fit is None or track is None or track.frames_since_anchor >= LOCK_REANCHOR_INTERVAL:
        reference_rect = estimators.scaled_rect(
            reference_pixels, scale, reference.gray.shape[1], reference.gray.shape[0]
        )
        current_rect = estimators.scaled_rect(current_pixels, scale, current_gray.shape[1], current_gray.shape[0])
        matched_gray, sigma = matched_reference(reference, current_gray, reference_rect, current_rect)
        note("stabilization.lock_blur_match", sigma)
        blurred = matched_gray if matched_gray is not reference.gray else None
        anchor_points = _orb_correspondences(
            reference, current_gray, current_pixels, frame.shape, scale, blurred, reference_rect
        )
        anchor_kind = "orb_anchor"
        if anchor_points is None:
            shift = estimators.estimate_template_shift(frame, reference_frame, focus_rect)
            anchor_points = _seed_from_reference(matched_gray, reference_rect, current_gray, scale, shift, sigma)
            anchor_kind = "lk_seed"
        if anchor_points is not None:
            anchor_fit = _fit(anchor_points[1], anchor_points[0], scale, frame.shape)

    chosen_fit = tracks_fit
    chosen_points = (tracked_reference, tracked_current)
    estimator = "lk_lock"
    if (
        anchor_fit is not None
        and anchor_points is not None
        and (tracks_fit is None or anchor_fit[2] <= tracks_fit[2] * RESIDUAL_TOLERANCE)
    ):
        chosen_fit, chosen_points, estimator = anchor_fit, anchor_points, anchor_kind
    if chosen_fit is None:
        if state is not None:
            state.lock = None
        if shift is None and anchor_points is None:
            shift = estimators.estimate_template_shift(frame, reference_frame, focus_rect)
        if shift is None:
            note("stabilization.estimator", "none")
            return None
        note("stabilization.estimator", "template_match")
        return transforms.translation_to_affine(shift[0], shift[1])

    transform, inliers, residual = chosen_fit
    reference_points, current_points = chosen_points
    note("stabilization.estimator", estimator)
    note("stabilization.inliers", int(np.count_nonzero(inliers)))
    note("stabilization.lock_residual_px", round(residual, 2))
    if state is not None:
        since_anchor = 0
        if track is not None and estimator == "lk_lock":
            since_anchor = track.frames_since_anchor + 1
        state.lock = LockTrack(
            id(reference_frame),
            scale,
            reference_points[inliers],
            current_points[inliers],
            current_gray,
            frame_index,
            since_anchor,
        )
    return transform


def _advance_tracks(track: LockTrack, current_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Carry the correspondences to ``current_gray`` with LK; the ones that fail the round trip are dropped."""
    points = track.current_points.reshape(-1, 1, 2).astype(np.float32)
    start, moved = track_points(track.prev_gray, current_gray, points, LOCK_LK_PARAMS, LOCK_FORWARD_BACKWARD_PX)
    if len(start) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty
    # track_points keeps order among survivors; recover which reference points they belong to.
    survivors = _match_rows(track.current_points, start)
    return track.reference_points[survivors], moved


def _match_rows(original: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Indices in ``original`` of the rows that appear (in order) in ``kept``."""
    indices = []
    cursor = 0
    for row in kept:
        while cursor < len(original) and not np.array_equal(original[cursor], row):
            cursor += 1
        if cursor >= len(original):
            break
        indices.append(cursor)
        cursor += 1
    return np.array(indices, dtype=np.int64)


SHARPNESS_PRESMOOTH_SIGMA = 1.0
"""Sensor noise and resampling artefacts dominate a raw Laplacian; smoothing first leaves the actual blur."""


def _sharpness(gray: np.ndarray, rect: tuple[int, int, int, int]) -> float:
    x, y, width, height = rect
    region = gray[y : y + height, x : x + width]
    if region.size == 0:
        return 0.0
    smoothed = cv2.GaussianBlur(region, (0, 0), SHARPNESS_PRESMOOTH_SIGMA)
    return float(cv2.Laplacian(smoothed, cv2.CV_64F).var())


def matched_reference(
    reference: estimators.ReferenceFeatures,
    current_gray: np.ndarray,
    reference_rect: tuple[int, int, int, int],
    current_rect: tuple[int, int, int, int],
) -> tuple[np.ndarray, float]:
    """The reference gray blurred until its region is about as sharp as the current frame's; ``(gray, sigma)``."""
    target = _sharpness(current_gray, current_rect)
    crisp = _sharpness(reference.gray, reference_rect)
    best_gray, best_sigma, best_gap = reference.gray, 0.0, abs(crisp - target)
    if crisp <= 0.0 or target >= crisp * BLUR_MATCH_THRESHOLD:
        return reference.gray, 0.0
    for sigma in BLUR_MATCH_SIGMAS[1:]:
        blurred = cv2.GaussianBlur(reference.gray, (0, 0), sigma)
        gap = abs(_sharpness(blurred, reference_rect) - target)
        if gap < best_gap:
            best_gray, best_sigma, best_gap = blurred, sigma, gap
        else:
            break  # sharpness falls monotonically with sigma: past the target it only gets worse
    return best_gray, best_sigma


def _seed_from_reference(
    reference_gray: np.ndarray,
    reference_rect: tuple[int, int, int, int],
    current_gray: np.ndarray,
    scale: float,
    shift: tuple[float, float] | None,
    sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Track corners of the reference region straight into the current frame (the lock is near its reference pose).

    ``reference_gray`` should already be blur-matched (by ``sigma``) to the current frame; the
    round-trip tolerance grows with the blur, since a soft blob cannot be pinned to the pixel.
    ``shift`` (full-res pixels, current -> reference, from template matching) seeds the search so
    larger displacements converge; without it LK starts from the reference positions.
    """
    ref_x, ref_y, ref_w, ref_h = reference_rect
    region = reference_gray[ref_y : ref_y + ref_h, ref_x : ref_x + ref_w]
    corners = grid_features(region, per_cell=SEED_FEATURES_PER_CELL)
    if len(corners) < LOCK_MIN_POINTS:
        return None
    corners[:, 0, 0] += ref_x
    corners[:, 0, 1] += ref_y
    guess = corners.copy()
    if shift is not None:
        guess[:, 0, 0] -= shift[0] * scale  # the shift moves current -> reference; the guess goes the other way
        guess[:, 0, 1] -= shift[1] * scale
    params = dict(LOCK_LK_PARAMS)
    forward, status, _errors = cv2.calcOpticalFlowPyrLK(
        reference_gray, current_gray, corners, guess, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **params
    )
    if forward is None or status is None:
        return None
    backward, back_status, _errors = cv2.calcOpticalFlowPyrLK(current_gray, reference_gray, forward, None, **params)
    keep = status.reshape(-1) == 1
    if backward is not None and back_status is not None:
        drift = np.linalg.norm(backward.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
        tolerance = max(LOCK_FORWARD_BACKWARD_PX, 0.75 * sigma)
        keep &= (back_status.reshape(-1) == 1) & (drift <= tolerance)
    if np.count_nonzero(keep) < LOCK_MIN_POINTS:
        return None
    return corners.reshape(-1, 2)[keep].astype(np.float32), forward.reshape(-1, 2)[keep].astype(np.float32)


def _orb_correspondences(
    reference: estimators.ReferenceFeatures,
    current_gray: np.ndarray,
    current_pixels: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    scale: float,
    blurred_reference: np.ndarray | None = None,
    reference_rect: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Reference <-> current ORB matches around the focus region (RANSAC inliers only), or ``None``.

    With ``blurred_reference`` the reference descriptors are recomputed on it (a blurred frame only
    matches a blurred reference); otherwise the cached crisp ones are used.
    """
    if blurred_reference is not None and reference_rect is not None:
        mask = np.zeros_like(blurred_reference, dtype=np.uint8)
        ref_x, ref_y, ref_w, ref_h = reference_rect
        mask[ref_y : ref_y + ref_h, ref_x : ref_x + ref_w] = 255
        keypoints, descriptors = cv2.ORB_create(**estimators.ORB_PARAMS).detectAndCompute(blurred_reference, mask)
        points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32).reshape(-1, 2)
        reference = estimators.ReferenceFeatures(reference.frame, blurred_reference, points, descriptors)
    if reference.descriptors is None or len(reference.points) < estimators.ORB_MIN_KEYPOINTS:
        return None
    height, width = frame_shape[:2]
    search = estimators.expanded_search_rect(current_pixels, width, height, estimators.CAMERA_LOCK_SEARCH_EXPANSION)
    cur_x, cur_y, cur_w, cur_h = estimators.scaled_rect(search, scale, current_gray.shape[1], current_gray.shape[0])
    mask = np.zeros_like(current_gray, dtype=np.uint8)
    mask[cur_y : cur_y + cur_h, cur_x : cur_x + cur_w] = 255
    for candidate_mask in (mask, None):
        matches = estimators.orb_matches(reference, current_gray, candidate_mask)
        if matches is None:
            continue
        reference_points, current_points = matches
        _transform, inliers = cv2.estimateAffinePartial2D(
            current_points, reference_points, method=cv2.RANSAC, ransacReprojThreshold=LOCK_RANSAC_PX
        )
        if inliers is None:
            continue
        keep = inliers.reshape(-1).astype(bool)
        if np.count_nonzero(keep) >= LOCK_MIN_POINTS:
            return reference_points[keep], current_points[keep]
    return None


def _residual(
    transform: np.ndarray, current_points: np.ndarray, reference_points: np.ndarray, keep: np.ndarray
) -> float:
    """Median reprojection error (analysis px) of the inliers under ``transform`` (current -> reference)."""
    points = current_points[keep].reshape(-1, 1, 2).astype(np.float32)
    if transform.shape == (2, 3):
        projected = cv2.transform(points, transform)
    else:
        projected = cv2.perspectiveTransform(points, transform)
    return float(np.median(np.linalg.norm(projected.reshape(-1, 2) - reference_points[keep], axis=1)))


def _fit(
    current_points: np.ndarray, reference_points: np.ndarray, scale: float, frame_shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Similarity (or homography when the points allow it) mapping current -> reference, with inliers and residual."""
    transform, inliers = cv2.estimateAffinePartial2D(
        current_points, reference_points, method=cv2.RANSAC, ransacReprojThreshold=LOCK_RANSAC_PX, refineIters=10
    )
    if transform is None or inliers is None:
        return None
    keep = inliers.reshape(-1).astype(bool)
    if np.count_nonzero(keep) < LOCK_MIN_POINTS:
        return None
    result = transforms.rescale_affine(transform, scale)
    if not transforms.is_reasonable(result, frame_shape):
        note("stabilization.rejected", "unreasonable_similarity")
        return None
    note("stabilization.lock_model", "similarity")
    residual = _residual(transform, current_points, reference_points, keep)

    height, width = frame_shape[:2]
    spread = np.ptp(current_points[keep], axis=0) / scale
    if (
        np.count_nonzero(keep) >= HOMOGRAPHY_MIN_POINTS
        and spread[0] >= width * HOMOGRAPHY_MIN_EXTENT
        and (spread[1] >= height * HOMOGRAPHY_MIN_EXTENT)
    ):
        homography, h_inliers = cv2.findHomography(current_points, reference_points, cv2.RANSAC, LOCK_RANSAC_PX)
        if homography is not None and h_inliers is not None:
            h_keep = h_inliers.reshape(-1).astype(bool)
            candidate = transforms.rescale_homography(homography, scale)
            if np.count_nonzero(h_keep) >= HOMOGRAPHY_MIN_POINTS and transforms.is_reasonable(
                candidate, frame_shape, allow_perspective=True
            ):
                note("stabilization.lock_model", "homography")
                return candidate, h_keep, _residual(homography, current_points, reference_points, h_keep)
    return result, keep, residual
