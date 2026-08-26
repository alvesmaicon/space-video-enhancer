"""Camera motion vs a moving object (global_motion) and a camera lock that survives blur and distortion (lock)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from space_video_enhancer.pipeline.stabilization import estimators, global_motion, lock
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import NormalizedRect, StabilizationMode
from space_video_enhancer.telemetry import trace as tracing
from tests.conftest import make_textured_frame

WIDTH, HEIGHT = 640, 360
OBJECT_SPEED_PX = 6
FOCUS = NormalizedRect(0.3, 0.3, 0.4, 0.4)


def object_scene(
    count: int, object_fraction: float, background_dots: int, seed: int = 3
) -> tuple[list[np.ndarray], np.ndarray]:
    """A jittering, slowly panning camera over a plain gradient with dots, plus a big textured block walking right.

    Returns the frames and, per frame after the first, the shift that aligns it onto the previous one.
    """
    rng = np.random.default_rng(seed)
    canvas = np.zeros((HEIGHT + 200, WIDTH + 200, 3), np.uint8)
    canvas[..., 0] = np.linspace(60, 140, WIDTH + 200, dtype=np.uint8)[None, :]
    canvas[..., 1] = np.linspace(80, 160, HEIGHT + 200, dtype=np.uint8)[:, None]
    canvas[..., 2] = 120
    for _ in range(background_dots):
        x, y = int(rng.integers(0, WIDTH + 190)), int(rng.integers(0, HEIGHT + 190))
        cv2.circle(canvas, (x, y), int(rng.integers(3, 9)), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    object_width, object_height = int(WIDTH * object_fraction), int(HEIGHT * object_fraction)
    texture = rng.integers(0, 255, (object_height // 6, object_width // 6, 3), dtype=np.uint8)
    block = cv2.resize(texture, (object_width, object_height), interpolation=cv2.INTER_NEAREST)
    camera = 100.0 + rng.normal(0.0, 3.0, (count, 2))
    camera[0] = 100.0
    camera[:, 0] += np.linspace(0, 20, count)
    frames = []
    for index, (x, y) in enumerate(camera):
        matrix = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y]], np.float32)
        frame = cv2.warpAffine(canvas, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
        left = int(WIDTH * 0.15 + OBJECT_SPEED_PX * index) % (WIDTH - object_width)
        top = int(HEIGHT * 0.25)
        frame[top : top + object_height, left : left + object_width] = block
        frames.append(frame)
    return frames, np.diff(camera, axis=0)


def estimated_shifts(frames: list[np.ndarray], mode: StabilizationMode) -> list[np.ndarray | None]:
    state = StabilizationState()
    shifts: list[np.ndarray | None] = []
    for index, frame in enumerate(frames):
        transform = estimators.estimate_inter_frame_transform(frame, index, state, mode)
        if index > 0:
            shifts.append(None if transform is None else np.array([transform[0, 2], transform[1, 2]]))
    return shifts


@pytest.mark.parametrize("mode", [StabilizationMode.AUTO, StabilizationMode.SIMILARITY, StabilizationMode.PERSPECTIVE])
@pytest.mark.parametrize(("object_fraction", "dots"), [(0.4, 300), (0.4, 40), (0.55, 40)])
def test_auto_modes_follow_the_camera_not_a_large_moving_object(
    mode: StabilizationMode, object_fraction: float, dots: int
) -> None:
    frames, truth = object_scene(30, object_fraction, dots)
    shifts = estimated_shifts(frames, mode)
    errors = np.array([np.linalg.norm(shift - expected) for shift, expected in zip(shifts, truth) if shift is not None])
    assert len(errors) >= 27, "almost every frame gets an estimate"
    assert np.count_nonzero(errors < 1.5) >= 0.9 * len(errors), (
        f"camera motion within 1.5 px on 90%+ of frames: {errors}"
    )
    followed_object = sum(
        abs((shift - expected)[0] + OBJECT_SPEED_PX) < 2.0
        for shift, expected in zip(shifts, truth)
        if shift is not None
    )
    assert followed_object <= 1, "the object (almost) never becomes the reference"


def test_grid_features_reach_smooth_cells_and_skip_flat_noise() -> None:
    frames, _truth = object_scene(1, 0.4, 40)
    gray, _scale = estimators.analysis_gray(frames[0])
    points = global_motion.grid_features(gray).reshape(-1, 2)
    cells = np.unique(global_motion.cell_indices(points, (gray.shape[1], gray.shape[0])))
    assert len(cells) >= 20, "dots on the plain background give the outer cells representatives"
    flat = np.full((90, 160), 128, np.uint8)
    flat[:, 80:] = 129  # a single faint step: not trackable
    assert len(global_motion.grid_features(flat)) == 0


def test_cell_memory_learns_the_moving_region() -> None:
    frames, _truth = object_scene(12, 0.4, 300)
    state = StabilizationState()
    for index, frame in enumerate(frames):
        estimators.estimate_inter_frame_transform(frame, index, state, StabilizationMode.AUTO)
    memory = state.cell_memory
    assert memory is not None and memory.shape == (global_motion.GRID_ROWS, global_motion.GRID_COLUMNS)
    assert memory.min() < 0.5 < memory.max(), "cells under the walking block lost weight, background cells kept it"
    assert state.last_vectors is not None and state.last_vectors.inliers.any()
    estimators.estimate_inter_frame_transform(frames[0], 40, state, StabilizationMode.AUTO)  # far jump: new take
    assert state.cell_memory is None


# --- camera lock ------------------------------------------------------------------------------

_canvas = cv2.resize(make_textured_frame(960, 540, seed=5), (WIDTH + 200, HEIGHT + 200))


def lock_view(x: float, y: float, *, blur: float = 0.0, zoom: float = 1.0, angle: float = 0.0, tilt: float = 0.0):
    source = np.float32([[x, y], [x + WIDTH, y], [x + WIDTH, y + HEIGHT], [x, y + HEIGHT]])
    destination = np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    destination[0, 0] += tilt * WIDTH
    destination[1, 0] -= tilt * WIDTH
    homography = cv2.getPerspectiveTransform(source, destination)
    if zoom != 1.0 or angle:
        rotation = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), angle, zoom)
        homography = np.vstack([rotation, [0, 0, 1]]) @ homography
    frame = cv2.warpPerspective(_canvas, homography, (WIDTH, HEIGHT))
    if blur > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    return frame, homography


def centre_error(transform: np.ndarray, reference_homography: np.ndarray, current_homography: np.ndarray) -> float:
    centre = np.array([[[WIDTH * 0.5, HEIGHT * 0.5]]], np.float32)
    truth = cv2.perspectiveTransform(
        cv2.perspectiveTransform(centre, np.linalg.inv(current_homography)), reference_homography
    ).reshape(2)
    if transform.shape == (2, 3):
        placed = cv2.transform(centre, transform).reshape(2)
    else:
        placed = cv2.perspectiveTransform(centre, transform).reshape(2)
    return float(np.linalg.norm(placed - truth))


def lock_errors(cases: list[dict[str, float]], seed: int = 1) -> tuple[list[float], list[str]]:
    rng = np.random.default_rng(seed)
    reference, reference_homography = lock_view(100, 100)
    state = StabilizationState()
    errors, estimators_used = [], []
    for index, case in enumerate(cases):
        dx, dy = rng.normal(0, 4, 2)
        frame, homography = lock_view(100 + dx, 100 + dy, **case)
        trace = tracing.RenderTrace()
        with tracing.active(trace):
            transform = lock.lock_transform(frame, reference, FOCUS, state, index)
        estimators_used.append(str(trace.notes.get("stabilization.estimator")))
        errors.append(centre_error(transform, reference_homography, homography) if transform is not None else np.nan)
    return errors, estimators_used


def test_lock_keeps_tracking_through_a_blur_ramp_with_zoom() -> None:
    errors, used = lock_errors([{"zoom": 1.04, "blur": blur} for blur in np.linspace(0, 4, 16)])
    assert not np.isnan(errors).any()
    assert np.mean(errors) < 2.6, f"mean centre error {np.mean(errors):.2f} px: {errors}"
    assert "lk_lock" in used, "the tracks carried the lock through the blurred frames"
    assert "orb_anchor" in used, "sharp frames re-anchor the tracks"


def test_lock_seeds_tracks_from_the_reference_when_orb_finds_nothing() -> None:
    errors, used = lock_errors([{"blur": 3.5, "zoom": 1.03}] * 8)
    assert not np.isnan(errors).any() and np.mean(errors) < 2.0, errors
    assert used[0] in ("lk_seed", "orb_anchor") and "template_match" not in used


def test_lock_fits_a_homography_to_a_tilted_region() -> None:
    errors, _used = lock_errors([{"tilt": tilt} for tilt in np.linspace(0, 0.06, 8)])
    assert not np.isnan(errors).any() and float(np.max(errors)) < 2.0, errors


def test_lock_forgets_its_tracks_when_the_reference_changes() -> None:
    reference, _homography = lock_view(100, 100)
    other = reference.copy()
    frame, _homography = lock_view(103, 98)
    state = StabilizationState()
    assert lock.lock_transform(frame, reference, FOCUS, state, 1) is not None
    first = state.lock
    assert first is not None and first.reference_id == id(reference)
    assert lock.lock_transform(frame, other, FOCUS, state, 2) is not None
    assert state.lock is not None and state.lock.reference_id == id(other)
