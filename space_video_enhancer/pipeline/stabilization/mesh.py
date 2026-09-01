"""MeshFlow: one camera path per region of the frame, for the distortion no global warp explains.

Walking footage carries parallax — objects at different depths move at different speeds in the
image — and rolling-shutter jello that is not linear across the frame. A single transform per frame
(even a homography) leaves both behind. Following MeshFlow (Liu et al., 2016), a coarse grid of
vertices is laid over the frame and each vertex accumulates its own motion path from the sparse
tracked features around it: the **median** of the feature motions near the vertex (the paper's f1),
then a median filter across the vertex grid (f2). Medians keep moving subjects from bending the
mesh while preserving the region-dependent motion of the static world — which is exactly why the
global estimator's inlier mask is *not* used here: parallax is an outlier to a similarity fit, and
it is the signal.

:func:`analyze_mesh_path` is one tracking pass over a range of frames (the analysis image, like the
wobble analysis) producing an immutable, resolution-free :class:`MeshPath`. :func:`mesh_correction`
turns it into per-frame vertex displacements that cancel the fast part of every vertex path (an
offline centred smoothing — frames before and after), spatially coherent and clamped to the crop
margin. :func:`apply_mesh` warps one frame by one correction field with a single ``cv2.remap``.
Everything is pure and Qt-free; a correction field per frame makes renders stateless, so the export
stays parallel.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from space_video_enhancer.pipeline.stabilization.estimators import analysis_gray
from space_video_enhancer.pipeline.stabilization.global_motion import grid_features, track_points
from space_video_enhancer.settings import NormalizedRect

VERTEX_ROWS = 6
VERTEX_COLUMNS = 9
"""One vertex per corner of the feature grid's cells (8x5 cells): every vertex has features nearby."""
GATHER_REACH = 1.0
"""Features within this many cell sizes of a vertex vote on its motion (median). Wider reach smooths
the field but biases border vertices inward (their votes all sit on one side), flattening exactly the
gradients the mesh exists to capture — one cell of reach keeps that bias at ~20% of the edge value."""
MIN_VERTEX_VOTES = 3
"""Fewer votes than this and the vertex is filled from its neighbours instead (one stray track
must not bend the mesh)."""
CORRECTION_LIMIT_FRACTION = 0.05
"""Per-vertex cap on the correction, the same margin the automatic crop removes."""
SPATIAL_PASSES = 0
"""3x3 average passes over each frame's correction field. Measured on real walking footage: the f2
median at the transfer already keeps the field coherent, and every blur pass flattens exactly the
regional part the mesh exists for (trapezoid residual 3.9 px with no blur, 5.9 with one pass, 7.2
with two — the jello barely moved). Kept as a knob for footage where local jelly shows up."""


def vertex_positions(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """``(xs, ys)`` pixel coordinates of the vertex grid columns and rows on a ``width`` x ``height`` frame."""
    return (
        np.linspace(0.0, float(width - 1), VERTEX_COLUMNS),
        np.linspace(0.0, float(height - 1), VERTEX_ROWS),
    )


@dataclass(frozen=True)
class MeshPath:
    """Cumulative motion of every mesh vertex per source frame, relative to ``first_frame``.

    ``dx``/``dy`` are ``(count, VERTEX_ROWS, VERTEX_COLUMNS)`` float32 arrays holding **fractions of
    the frame width/height** — the same track serves the preview at any size and the full-size
    export. ``measured[i]`` is ``False`` for a frame whose pair could not be tracked (the path holds).
    """

    source: Path
    first_frame: int
    dx: np.ndarray
    dy: np.ndarray
    measured: np.ndarray

    @property
    def count(self) -> int:
        return len(self.dx)

    @property
    def last_frame(self) -> int:
        return self.first_frame + self.count - 1

    @property
    def measured_fraction(self) -> float:
        return float(self.measured.mean()) if self.count else 0.0

    def covers(self, first: int, last: int) -> bool:
        return self.first_frame <= first and last <= self.last_frame

    def spread_px(self, width: int, height: int) -> float:
        """How much the vertices disagree with their own frame mean, in pixels (RMS over all frames).

        Zero for a purely global motion; what a global warp cannot cancel lives here.
        """
        if not self.count:
            return 0.0
        deviation_x = (self.dx - self.dx.mean(axis=(1, 2), keepdims=True)) * width
        deviation_y = (self.dy - self.dy.mean(axis=(1, 2), keepdims=True)) * height
        return float(np.sqrt(np.mean(deviation_x**2 + deviation_y**2)))


@dataclass(frozen=True)
class MeshPathAnalysis:
    track: MeshPath | None
    cancelled: bool
    frames_read: int
    elapsed_ms: float


def _vertex_motion(previous: np.ndarray, current: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    """``(VERTEX_ROWS, VERTEX_COLUMNS, 2)`` median feature motion around every vertex, analysis pixels.

    The paper's transfer: each vertex takes the median of the motions of the features within reach
    (f1), then a 3x3 median across the grid (f2). Vertices with too few votes borrow from their
    neighbours; ``None`` when no vertex could be measured at all.
    """
    vectors = current - previous
    width, height = size
    xs, ys = vertex_positions(width, height)
    reach_x = GATHER_REACH * width / (VERTEX_COLUMNS - 1)
    reach_y = GATHER_REACH * height / (VERTEX_ROWS - 1)
    field = np.full((VERTEX_ROWS, VERTEX_COLUMNS, 2), np.nan, dtype=np.float64)
    for row in range(VERTEX_ROWS):
        near_row = np.abs(previous[:, 1] - ys[row]) <= reach_y
        for column in range(VERTEX_COLUMNS):
            votes = near_row & (np.abs(previous[:, 0] - xs[column]) <= reach_x)
            if int(votes.sum()) >= MIN_VERTEX_VOTES:
                field[row, column] = np.median(vectors[votes], axis=0)
    if np.isnan(field[..., 0]).all():
        return None

    # Empty vertices borrow the mean of their measured neighbours until the field is full.
    for _ in range(VERTEX_ROWS + VERTEX_COLUMNS):
        missing = np.isnan(field[..., 0])
        if not missing.any():
            break
        filled = np.where(np.isnan(field), 0.0, field)
        counts = cv2.blur((~missing).astype(np.float32), (3, 3), borderType=cv2.BORDER_REPLICATE)
        sums = cv2.blur(filled.astype(np.float32), (3, 3), borderType=cv2.BORDER_REPLICATE)
        with np.errstate(invalid="ignore"):
            neighbours = sums / counts[..., None]
        take = missing & (counts > 0)
        field[take] = neighbours[take]

    for axis in range(2):  # f2: median across the grid keeps one bad vertex from bending the mesh
        field[..., axis] = cv2.medianBlur(field[..., axis].astype(np.float32), 3)
    return field


def analyze_mesh_path(
    read: Callable[[int], np.ndarray | None],
    source: Path,
    first: int,
    last: int,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> MeshPathAnalysis:
    """Track frames ``first`` .. ``last`` (inclusive) and accumulate one path per mesh vertex.

    ``read(frame_index)`` returns the decoded frame or ``None`` (the stream ended early: the path
    stops there). Only tracking runs — nothing is rendered. ``progress(done, total)`` is called per
    frame; ``cancelled()`` is polled between frames.
    """
    started = time.perf_counter()
    count = max(0, last - first + 1)
    dx = np.zeros((count, VERTEX_ROWS, VERTEX_COLUMNS), dtype=np.float32)
    dy = np.zeros((count, VERTEX_ROWS, VERTEX_COLUMNS), dtype=np.float32)
    measured = np.zeros(count, dtype=bool)
    previous_gray: np.ndarray | None = None
    frames_read = 0

    for offset in range(count):
        if cancelled is not None and cancelled():
            return MeshPathAnalysis(None, cancelled=True, frames_read=frames_read, elapsed_ms=_ms(started))
        frame = read(first + offset)
        if frame is None:
            break
        frames_read += 1
        gray, _scale = analysis_gray(frame)
        height, width = gray.shape[:2]
        if width < VERTEX_COLUMNS or height < VERTEX_ROWS:
            break
        if offset > 0:
            dx[offset] = dx[offset - 1]
            dy[offset] = dy[offset - 1]
        if previous_gray is not None and gray.shape == previous_gray.shape:
            points = grid_features(previous_gray)
            tracked_previous, tracked_current = track_points(previous_gray, gray, points)
            if len(tracked_previous) >= MIN_VERTEX_VOTES:
                field = _vertex_motion(tracked_previous, tracked_current, (width, height))
                if field is not None:
                    dx[offset] += (field[..., 0] / width).astype(np.float32)
                    dy[offset] += (field[..., 1] / height).astype(np.float32)
                    measured[offset] = True
        previous_gray = gray
        if progress is not None:
            progress(offset + 1, count)

    if frames_read == 0:
        return MeshPathAnalysis(None, cancelled=False, frames_read=0, elapsed_ms=_ms(started))
    track = MeshPath(source, first, dx[:frames_read], dy[:frames_read], measured[:frames_read])
    return MeshPathAnalysis(track, cancelled=False, frames_read=frames_read, elapsed_ms=_ms(started))


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _centred_smooth(values: np.ndarray, half: int) -> np.ndarray:
    """Hann-weighted centred moving average along axis 0, one-sided at the ends (see wobble.filter_wobble)."""
    count = len(values)
    kernel = np.hanning(2 * half + 3)[1:-1]
    kernel /= kernel.sum()
    flat = values.reshape(count, -1)
    weight = np.convolve(np.pad(np.ones(count), half), kernel, mode="valid")
    smoothed = np.stack(
        [np.convolve(np.pad(flat[:, column], half), kernel, mode="valid") for column in range(flat.shape[1])],
        axis=1,
    )
    return (smoothed / weight[:, None]).reshape(values.shape)


def mesh_correction(
    path: MeshPath,
    smoothing_frames: float,
    *,
    spatial_passes: int = SPATIAL_PASSES,
    limit: float = CORRECTION_LIMIT_FRACTION,
) -> np.ndarray:
    """Per-frame vertex displacements (fractions) that put every vertex on its smoothed path.

    ``smoothing_frames`` is the virtual camera's time constant (the smoothing slider, see
    ``stabilize.smoothing_time_constant``): each vertex path is replaced by its centred moving
    average over that many frames on each side, and the correction is the difference. The field is
    then blurred across the grid (``spatial_passes``) so neighbouring vertices agree, and
    clamped to ``limit`` (the stabilization margin) so the fixed crop always hides it.

    Returns ``(count, VERTEX_ROWS, VERTEX_COLUMNS, 2)`` float32, ``[..., 0]`` = x fraction.
    """
    half = max(1, round(smoothing_frames))
    stacked = np.stack([path.dx, path.dy], axis=-1).astype(np.float64)
    correction = _centred_smooth(stacked, half) - stacked
    for index in range(len(correction)):
        for _ in range(spatial_passes):
            correction[index] = cv2.blur(correction[index].astype(np.float32), (3, 3), borderType=cv2.BORDER_REPLICATE)
    return np.clip(correction, -limit, limit).astype(np.float32)


@dataclass(frozen=True)
class MeshCorrections:
    """The per-frame correction fields of one clip, ready for :func:`apply_mesh` (fractions, size-free).

    Computed once from a :class:`MeshPath` and the smoothing time constant (:func:`corrections_for`);
    every render — preview at any size, export in any worker — just indexes its source frame.
    """

    first_frame: int
    fields: np.ndarray
    """``(count, VERTEX_ROWS, VERTEX_COLUMNS, 2)`` float32 (see :func:`mesh_correction`)."""

    def at(self, frame_index: int) -> np.ndarray | None:
        offset = frame_index - self.first_frame
        if offset < 0 or offset >= len(self.fields):
            return None
        return self.fields[offset]


def corrections_for(
    path: MeshPath, smoothing_frames: float, limit: float = CORRECTION_LIMIT_FRACTION
) -> MeshCorrections:
    """The clip's correction fields for one smoothing and margin (see :func:`mesh_correction`)."""
    return MeshCorrections(path.first_frame, mesh_correction(path, smoothing_frames, limit=limit))


ANCHOR_SAMPLE_OFFSETS = ((0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
"""Where each anchor region samples the vertex paths: its centre and four corners — five spread
points per region, so even one region determines a similarity and regions in opposite corners pin
the rotation and scale exactly (the Wave-2 idea, fed from the mesh instead of a live tracker)."""


def _sample_path_at(dx: np.ndarray, dy: np.ndarray, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame displacement of the content at fraction position ``(x, y)``: bilinear over the vertex grid.

    ``dx``/``dy`` are ``(count, VERTEX_ROWS, VERTEX_COLUMNS)``; returns two ``(count,)`` series.
    """
    column = min(max(x, 0.0), 1.0) * (VERTEX_COLUMNS - 1)
    row = min(max(y, 0.0), 1.0) * (VERTEX_ROWS - 1)
    c0, r0 = int(column), int(row)
    c1, r1 = min(c0 + 1, VERTEX_COLUMNS - 1), min(r0 + 1, VERTEX_ROWS - 1)
    wc, wr = column - c0, row - r0
    weights = ((1 - wr) * (1 - wc), (1 - wr) * wc, wr * (1 - wc), wr * wc)
    cells = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    sample_x = sum(weight * dx[:, r, c] for weight, (r, c) in zip(weights, cells))
    sample_y = sum(weight * dy[:, r, c] for weight, (r, c) in zip(weights, cells))
    return np.asarray(sample_x, dtype=np.float64), np.asarray(sample_y, dtype=np.float64)


def _anchor_similarities(path: MeshPath, regions: tuple[NormalizedRect, ...], aspect: float) -> np.ndarray:
    """``(count, 4)`` per-frame similarity ``(a, b, tx, ty)`` of the anchor content's cumulative motion.

    Sampled from the vertex paths at each region's centre and corners and fitted per frame in
    *height units* (x scaled by ``aspect``) so rotation and scale are metrically true on any frame
    shape. The similarity maps a point's reference position onto its position in the frame:
    ``(X, Y) -> (a X - b Y + tx, b X + a Y + ty)``. Frame 0 is the identity (paths start at zero).
    """
    positions = []
    for region in regions:
        for offset_x, offset_y in ANCHOR_SAMPLE_OFFSETS:
            positions.append((region.x + offset_x * region.width, region.y + offset_y * region.height))
    count = path.count
    base = np.array([(x * aspect, y) for x, y in positions])  # height units
    displaced_x = np.stack([_sample_path_at(path.dx, path.dy, x, y)[0] for x, y in positions], axis=1) * aspect
    displaced_y = np.stack([_sample_path_at(path.dx, path.dy, x, y)[1] for x, y in positions], axis=1)

    design = np.zeros((2 * len(positions), 4))
    design[0::2, 0] = base[:, 0]
    design[0::2, 1] = -base[:, 1]
    design[0::2, 2] = 1.0
    design[1::2, 0] = base[:, 1]
    design[1::2, 1] = base[:, 0]
    design[1::2, 3] = 1.0
    pseudo_inverse = np.linalg.pinv(design)

    similarities = np.zeros((count, 4))
    similarities[:, 0] = 1.0
    targets = np.empty(2 * len(positions))
    for index in range(count):
        targets[0::2] = base[:, 0] + displaced_x[index]
        targets[1::2] = base[:, 1] + displaced_y[index]
        similarities[index] = pseudo_inverse @ targets
    return similarities


def _similarity_matrices(params: np.ndarray) -> np.ndarray:
    """``(count, 2, 3)`` affine matrices for ``(count, 4)`` similarity params ``(a, b, tx, ty)``."""
    count = len(params)
    matrices = np.zeros((count, 2, 3))
    matrices[:, 0, 0] = params[:, 0]
    matrices[:, 0, 1] = -params[:, 1]
    matrices[:, 0, 2] = params[:, 2]
    matrices[:, 1, 0] = params[:, 1]
    matrices[:, 1, 1] = params[:, 0]
    matrices[:, 1, 2] = params[:, 3]
    return matrices


def anchor_correction_field(
    path: MeshPath, regions: tuple[NormalizedRect, ...], smoothing_frames: float, aspect: float
) -> np.ndarray:
    """Per-frame vertex displacements (fractions) that put the anchor content on its *smoothed* pose.

    The anchors' pooled similarity path is smoothed entry-wise over time (similarities are closed
    under averaging) and each frame's correction is ``smoothed ∘ current⁻¹`` — the exact transform
    that maps where the anchors are onto where their smoothed trajectory says they should be —
    evaluated at every vertex. Rotation and scale correct about the anchors themselves, not the
    frame corner, because the pose fit carries the anchors' geometry.
    """
    half = max(1, round(smoothing_frames))
    poses = _anchor_similarities(path, regions, aspect)
    smoothed = _centred_smooth(poses, half)
    current = _similarity_matrices(poses)
    target = _similarity_matrices(smoothed)

    xs = np.linspace(0.0, 1.0, VERTEX_COLUMNS) * aspect
    ys = np.linspace(0.0, 1.0, VERTEX_ROWS)
    grid_x, grid_y = np.meshgrid(xs, ys)  # (ROWS, COLUMNS), height units
    points = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)  # (R, C, 3)

    field = np.zeros((path.count, VERTEX_ROWS, VERTEX_COLUMNS, 2))
    identity = np.eye(3)
    for index in range(path.count):
        forward = identity.copy()
        forward[:2] = current[index]
        inverse = np.linalg.inv(forward)[:2]
        correction = target[index] @ np.vstack([inverse, [0.0, 0.0, 1.0]])  # smoothed ∘ current⁻¹, 2x3
        moved = points @ correction.T  # (R, C, 2) height units
        field[index, ..., 0] = (moved[..., 0] - grid_x) / aspect  # back to width fractions
        field[index, ..., 1] = moved[..., 1] - grid_y
    return field


def corrections_for_object(
    path: MeshPath,
    regions: tuple[NormalizedRect, ...],
    smoothing_frames: float,
    *,
    limit: float = CORRECTION_LIMIT_FRACTION,
    aspect: float = 1.0,
) -> MeshCorrections:
    """The object mode's fields: the anchors follow their smoothed pose, the mesh corrects around them.

    Assembly: the anchor correction (:func:`anchor_correction_field`) is applied to every vertex,
    the vertex paths are re-expressed as they would move *after* that warp, and the standard mesh
    smoothing runs on those adjusted paths — so the anchors land exactly on their smoothed
    trajectory (smoothing an already-smooth path is a no-op there) while the rest of the frame
    keeps its slow parallax relative to the anchors and loses only the fast part. With no regions
    this degenerates to :func:`corrections_for`. ``aspect`` is the analysed frames' width/height —
    it makes the pose's rotation and scale metrically true. The total is clamped to ``limit`` (the
    margin) so the fixed crop always hides it.
    """
    if not regions:
        return corrections_for(path, smoothing_frames, limit)
    half = max(1, round(smoothing_frames))
    anchor = anchor_correction_field(path, regions, smoothing_frames, aspect)
    adjusted = np.stack([path.dx, path.dy], axis=-1).astype(np.float64) + anchor
    residual = _centred_smooth(adjusted, half) - adjusted
    total = np.clip(anchor + residual, -limit, limit).astype(np.float32)
    return MeshCorrections(path.first_frame, total)


def apply_mesh(frame: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """``frame`` warped so its content moves by ``correction`` (one ``(VERTEX_ROWS, VERTEX_COLUMNS, 2)``
    field of fractions), interpolated bilinearly between vertices — a single ``cv2.remap``."""
    height, width = frame.shape[:2]
    shift_x = cv2.resize(correction[..., 0] * width, (width, height), interpolation=cv2.INTER_LINEAR)
    shift_y = cv2.resize(correction[..., 1] * height, (width, height), interpolation=cv2.INTER_LINEAR)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return cv2.remap(
        frame,
        grid_x - shift_x.astype(np.float32),
        grid_y - shift_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
