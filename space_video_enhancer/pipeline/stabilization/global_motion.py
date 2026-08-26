"""Camera motion between two frames when something else moves in the picture too.

The old estimators fed every tracked corner to a median (translation) or to RANSAC (similarity,
perspective): whichever motion had the *most* corners won. A textured person in front of a plain
wall has most of the corners, so the stabilizer followed the person. Three things separate the
camera from a moving object without counting heads:

* **Grid features** — corners are picked per cell of a grid, so a smooth background still gets
  representatives (a global threshold would drop them all in favour of the textured subject).
* **Spread, not majority** — the tracked vectors are split into rigid-motion hypotheses (the
  dominant one and the dominant one among its outliers). Camera motion is the hypothesis whose
  inliers are spread over the frame; an object's inliers are compact. The score is the extent of
  the inliers (their robust bounding box, in grid cells — a sparse background still spans the
  whole frame), weighted by the cell memory below.
* **Cell memory** — cells that agreed with the chosen camera motion gain weight, cells that kept
  disagreeing lose it, so a subject that walks through the frame stops voting after a few frames
  and recovers when it stops moving.

Everything here works in the downscaled analysis image; callers rescale the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from space_video_enhancer.settings import StabilizationMode
from space_video_enhancer.telemetry.trace import note

GRID_COLUMNS = 8
GRID_ROWS = 5
FEATURES_PER_CELL = 6
CELL_GFTT = {"qualityLevel": 0.02, "minDistance": 6, "blockSize": 5}
"""Per-cell corner detection: the quality is relative to the cell's own strongest corner."""
MIN_CORNER_RESPONSE_FRACTION = 0.002
"""Corners weaker than this fraction of the frame's strongest are noise on a flat wall: skipped."""
LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
}
FORWARD_BACKWARD_MAX_PX = 1.0
"""A track is kept only if tracking it back lands within this distance of where it started."""
MIN_TRACKED = 6
TRANSLATION_INLIER_PX = 1.5
"""Vectors within this distance of a translation hypothesis belong to it (analysis pixels)."""
GEOMETRIC_INLIER_PX = 2.0
MIN_TRANSLATION_POINTS = 4
"""Four vectors agreeing within 1.5 px do not happen by chance; a thin strip of background is enough."""
MIN_GEOMETRIC_POINTS = 4
MAX_HYPOTHESES = 3
HOMOGRAPHY_MIN_POINTS = 12
HOMOGRAPHY_MIN_EXTENT = 0.28
"""Perspective mode fits a homography to the winner only when its inliers span this fraction of the frame."""
MEMORY_GAIN = 0.25
MEMORY_DECAY = 0.6
MEMORY_FLOOR = 0.05
"""Cell weight bounds: a cell never loses its vote completely (the subject may stop moving)."""
SPREAD_PERCENTILES = (5.0, 95.0)
"""The inliers' extent ignores the outermost few points, so one stray match cannot inflate it."""
SIMILARITY_RANSAC = {
    "method": cv2.RANSAC,
    "ransacReprojThreshold": GEOMETRIC_INLIER_PX,
    "maxIters": 3000,
    "confidence": 0.99,
    "refineIters": 10,
}


@dataclass(frozen=True)
class MotionVectors:
    """What the estimator saw on one frame (analysis coordinates), for the diagnostic overlay."""

    previous: np.ndarray
    """``(N, 2)`` positions in the previous frame."""
    current: np.ndarray
    """``(N, 2)`` positions in the current frame."""
    inliers: np.ndarray
    """``(N,)`` bool: part of the chosen (camera) hypothesis."""
    size: tuple[int, int]
    """``(width, height)`` of the analysis image the coordinates refer to."""
    chosen: int
    """Index of the chosen hypothesis (``0`` = the majority one)."""
    hypotheses: int


@dataclass(frozen=True)
class MotionHypothesis:
    transform: np.ndarray
    """Affine ``(2, 3)`` or homography ``(3, 3)`` in analysis pixels, mapping current -> previous."""
    inliers: np.ndarray
    """``(N,)`` bool over the tracked points."""
    score: float
    cells: int
    kind: str

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inliers))


def new_cell_memory() -> np.ndarray:
    return np.ones((GRID_ROWS, GRID_COLUMNS), dtype=np.float32)


def cell_indices(points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``(N,)`` flat grid cell index of each analysis-pixel point."""
    width, height = size
    columns = np.clip((points[:, 0] * GRID_COLUMNS / max(1, width)).astype(np.int64), 0, GRID_COLUMNS - 1)
    rows = np.clip((points[:, 1] * GRID_ROWS / max(1, height)).astype(np.int64), 0, GRID_ROWS - 1)
    return rows * GRID_COLUMNS + columns


def grid_features(gray: np.ndarray, per_cell: int = FEATURES_PER_CELL) -> np.ndarray:
    """Up to ``per_cell`` corners in every grid cell, skipping corners too weak to be trackable.

    Returns ``(N, 1, 2)`` float32 like ``goodFeaturesToTrack`` (empty when nothing qualifies).
    """
    height, width = gray.shape[:2]
    if width < GRID_COLUMNS or height < GRID_ROWS:
        return np.empty((0, 1, 2), dtype=np.float32)
    response = cv2.cornerMinEigenVal(gray, CELL_GFTT["blockSize"])
    floor = float(response.max()) * MIN_CORNER_RESPONSE_FRACTION
    found = []
    for row in range(GRID_ROWS):
        top = row * height // GRID_ROWS
        bottom = (row + 1) * height // GRID_ROWS
        for column in range(GRID_COLUMNS):
            left = column * width // GRID_COLUMNS
            right = (column + 1) * width // GRID_COLUMNS
            cell = gray[top:bottom, left:right]
            corners = cv2.goodFeaturesToTrack(cell, per_cell, **CELL_GFTT)
            if corners is None:
                continue
            corners = corners.reshape(-1, 2)
            corners[:, 0] += left
            corners[:, 1] += top
            strong = response[corners[:, 1].astype(int), corners[:, 0].astype(int)] >= floor
            if np.any(strong):
                found.append(corners[strong])
    if not found:
        return np.empty((0, 1, 2), dtype=np.float32)
    return np.concatenate(found).reshape(-1, 1, 2).astype(np.float32)


def track_points(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    points: np.ndarray,
    lk_params: dict[str, object] | None = None,
    forward_backward_px: float = FORWARD_BACKWARD_MAX_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Lucas-Kanade with a forward-backward check: ``(previous, current)`` ``(N, 2)`` arrays of the survivors."""
    params = LK_PARAMS if lk_params is None else lk_params
    if len(points) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty
    forward, status, _errors = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None, **params)
    if forward is None or status is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty
    backward, back_status, _errors = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, forward, None, **params)
    start = points.reshape(-1, 2)
    forward = forward.reshape(-1, 2)
    if backward is None or back_status is None:
        keep = status.reshape(-1) == 1
    else:
        drift = np.linalg.norm(backward.reshape(-1, 2) - start, axis=1)
        keep = (status.reshape(-1) == 1) & (back_status.reshape(-1) == 1) & (drift <= forward_backward_px)
    height, width = current_gray.shape[:2]
    inside = (forward[:, 0] >= 0) & (forward[:, 0] < width) & (forward[:, 1] >= 0) & (forward[:, 1] < height)
    keep &= inside
    return start[keep].astype(np.float32), forward[keep].astype(np.float32)


def _translation_mode(vectors: np.ndarray, available: np.ndarray) -> np.ndarray | None:
    """Mean-shift from the median to the densest cluster of the ``available`` vectors."""
    candidates = vectors[available]
    if len(candidates) < MIN_TRANSLATION_POINTS:
        return None
    centre = np.median(candidates, axis=0)
    for _ in range(10):
        near = np.linalg.norm(candidates - centre, axis=1) <= TRANSLATION_INLIER_PX
        if not np.any(near):
            break
        updated = candidates[near].mean(axis=0)
        if np.allclose(updated, centre, atol=1e-3):
            break
        centre = updated
    return centre


def translation_hypotheses(previous: np.ndarray, current: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rigid translations explaining the vectors, densest first; each is ``(shift, inlier mask)``."""
    vectors = current - previous
    available = np.ones(len(vectors), dtype=bool)
    found = []
    for _ in range(MAX_HYPOTHESES):
        centre = _translation_mode(vectors, available)
        if centre is None:
            break
        inliers = available & (np.linalg.norm(vectors - centre, axis=1) <= TRANSLATION_INLIER_PX)
        if np.count_nonzero(inliers) < MIN_TRANSLATION_POINTS:
            break
        found.append((vectors[inliers].mean(axis=0), inliers))
        available &= ~inliers
    return found


def geometric_hypotheses(previous: np.ndarray, current: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Similarity fits, dominant first, each on the points the earlier ones left over.

    The split always uses a similarity: a homography has too many degrees of freedom to tell a
    sparse background from a moving object. Perspective mode upgrades the winner afterwards
    (:func:`upgrade_to_homography`).
    """
    available = np.ones(len(previous), dtype=bool)
    found = []
    for _ in range(MAX_HYPOTHESES):
        indices = np.flatnonzero(available)
        if len(indices) < MIN_GEOMETRIC_POINTS:
            break
        transform, mask = cv2.estimateAffinePartial2D(current[indices], previous[indices], **SIMILARITY_RANSAC)
        if transform is None or mask is None:
            break
        inlier_indices = indices[mask.reshape(-1).astype(bool)]
        if len(inlier_indices) < MIN_GEOMETRIC_POINTS:
            break
        inliers = np.zeros(len(previous), dtype=bool)
        inliers[inlier_indices] = True
        found.append((transform.astype(np.float32), inliers))
        available &= ~inliers
    return found


def upgrade_to_homography(
    previous: np.ndarray, current: np.ndarray, inliers: np.ndarray, size: tuple[int, int]
) -> np.ndarray | None:
    """A homography over the winner's inliers when they are numerous and spread enough to support one."""
    chosen = np.flatnonzero(inliers)
    if len(chosen) < HOMOGRAPHY_MIN_POINTS:
        return None
    spread = np.ptp(previous[chosen], axis=0)
    if spread[0] < size[0] * HOMOGRAPHY_MIN_EXTENT or spread[1] < size[1] * HOMOGRAPHY_MIN_EXTENT:
        return None
    homography, mask = cv2.findHomography(current[chosen], previous[chosen], cv2.RANSAC, GEOMETRIC_INLIER_PX)
    if homography is None or mask is None or np.count_nonzero(mask) < HOMOGRAPHY_MIN_POINTS:
        return None
    return homography.astype(np.float32)


def spread_score(
    inliers: np.ndarray, points: np.ndarray, size: tuple[int, int], memory: np.ndarray
) -> tuple[float, int]:
    """``(weighted extent, extent)``: grid cells spanned by the inliers' robust bounding box.

    Extent (not the count of occupied cells) is what tells a sparse background from a dense
    object: sixteen points along the edges of the frame span all 40 cells, seventy points on a
    subject span twelve. The weight is the mean cell memory over the cells the inliers occupy.
    """
    chosen = points[inliers]
    if len(chosen) == 0:
        return 0.0, 0
    width, height = size
    low = np.percentile(chosen, SPREAD_PERCENTILES[0], axis=0)
    high = np.percentile(chosen, SPREAD_PERCENTILES[1], axis=0)
    columns = int(high[0] * GRID_COLUMNS / max(1, width)) - int(low[0] * GRID_COLUMNS / max(1, width)) + 1
    rows = int(high[1] * GRID_ROWS / max(1, height)) - int(low[1] * GRID_ROWS / max(1, height)) + 1
    extent = max(1, min(GRID_COLUMNS, columns)) * max(1, min(GRID_ROWS, rows))
    covered = np.unique(cell_indices(chosen, size))
    weight = float(memory.reshape(-1)[covered].mean())
    return extent * weight, extent


def choose_hypothesis(
    candidates: list[tuple[np.ndarray, np.ndarray]],
    points: np.ndarray,
    size: tuple[int, int],
    memory: np.ndarray,
) -> tuple[int, list[MotionHypothesis]]:
    """Score every candidate by spread and return the index of the winner (ties go to more inliers)."""
    scored = []
    for transform, inliers in candidates:
        score, extent = spread_score(inliers, points, size, memory)
        scored.append(MotionHypothesis(transform, inliers, score, extent, "camera"))
    best = max(range(len(scored)), key=lambda index: (scored[index].score, scored[index].inlier_count))
    return best, scored


def update_cell_memory(memory: np.ndarray, cells: np.ndarray, inliers: np.ndarray) -> None:
    """Cells mostly explained by the chosen motion gain weight; cells that disagreed lose it (in place)."""
    flat = memory.reshape(-1)
    total = np.bincount(cells, minlength=flat.size)
    agreeing = np.bincount(cells[inliers], minlength=flat.size)
    seen = total > 0
    agrees = seen & (agreeing * 2 >= total)
    disagrees = seen & ~agrees
    flat[agrees] = np.minimum(1.0, flat[agrees] + MEMORY_GAIN)
    flat[disagrees] = np.maximum(MEMORY_FLOOR, flat[disagrees] * MEMORY_DECAY)


@dataclass(frozen=True)
class GlobalMotion:
    transform: np.ndarray
    """Analysis-pixel transform mapping the current frame onto the previous one."""
    vectors: MotionVectors


def estimate_global_motion(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    mode: StabilizationMode,
    memory: np.ndarray | None = None,
) -> GlobalMotion | None:
    """Camera motion from ``previous_gray`` to ``current_gray`` in analysis pixels, or ``None`` when untrackable.

    ``memory`` (see :func:`new_cell_memory`) is updated in place with what this frame taught about
    which cells belong to the background.
    """
    points = grid_features(previous_gray)
    previous, current = track_points(previous_gray, current_gray, points)
    note("stabilization.tracked_points", len(previous))
    if len(previous) < MIN_TRACKED:
        return None
    height, width = previous_gray.shape[:2]
    size = (int(width), int(height))
    cells = cell_indices(previous, size)
    weights = memory if memory is not None else new_cell_memory()

    if mode is StabilizationMode.AUTO:
        candidates = [
            (np.array([[1.0, 0.0, -shift[0]], [0.0, 1.0, -shift[1]]], dtype=np.float32), inliers)
            for shift, inliers in translation_hypotheses(previous, current)
        ]
    else:
        candidates = geometric_hypotheses(previous, current)
    if not candidates:
        note("stabilization.rejected", "no_consistent_motion")
        return None

    chosen, scored = choose_hypothesis(candidates, previous, size, weights)
    winner = scored[chosen]
    transform = winner.transform
    if mode is StabilizationMode.PERSPECTIVE:
        homography = upgrade_to_homography(previous, current, winner.inliers, size)
        if homography is not None:
            transform = homography
        note("stabilization.model", "homography" if homography is not None else "similarity")
    note("stabilization.hypotheses", [[hypothesis.inlier_count, hypothesis.cells] for hypothesis in scored])
    note("stabilization.chosen", chosen)
    note("stabilization.inliers", winner.inlier_count)
    if memory is not None:
        update_cell_memory(memory, cells, winner.inliers)
    vectors = MotionVectors(previous, current, winner.inliers, size, chosen, len(scored))
    return GlobalMotion(transform, vectors)
