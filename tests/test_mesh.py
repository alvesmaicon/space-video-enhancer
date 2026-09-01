from __future__ import annotations

import cv2
import numpy as np
import pytest

from space_video_enhancer.pipeline.stabilization.mesh import (
    CORRECTION_LIMIT_FRACTION,
    VERTEX_COLUMNS,
    VERTEX_ROWS,
    MeshPath,
    analyze_mesh_path,
    apply_mesh,
    mesh_correction,
    vertex_positions,
)

WIDTH, HEIGHT = 480, 270


def _canvas(margin: int = 80) -> np.ndarray:
    rng = np.random.default_rng(21)
    canvas = rng.integers(0, 255, (HEIGHT + 2 * margin, WIDTH + 2 * margin), dtype=np.uint8)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.2)
    for _ in range(260):
        x = int(rng.integers(0, WIDTH + 2 * margin))
        y = int(rng.integers(0, HEIGHT + 2 * margin))
        cv2.circle(canvas, (x, y), int(rng.integers(3, 10)), int(rng.integers(0, 255)), -1)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _frames_from_displacement(displacement: object, count: int, margin: int = 80) -> list[np.ndarray]:
    """Frames sampled from one big canvas; ``displacement(k) -> (H, W, 2)`` pixel shift of the content."""
    canvas = _canvas(margin)
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    frames = []
    for index in range(count):
        field = displacement(index)  # type: ignore[operator]
        map_x = xs + margin - field[..., 0].astype(np.float32)
        map_y = ys + margin - field[..., 1].astype(np.float32)
        frames.append(cv2.remap(canvas, map_x, map_y, cv2.INTER_LINEAR))
    return frames


def _analyze(frames: list[np.ndarray]) -> MeshPath:
    analysis = analyze_mesh_path(lambda index: frames[index] if 0 <= index < len(frames) else None, 0, len(frames) - 1)
    assert analysis.track is not None and not analysis.cancelled
    return analysis.track


def test_uniform_motion_reaches_every_vertex() -> None:
    def displacement(index: int) -> np.ndarray:
        field = np.zeros((HEIGHT, WIDTH, 2))
        field[..., 0] = 1.5 * index
        field[..., 1] = -0.8 * index
        return field

    track = _analyze(_frames_from_displacement(displacement, 10))
    assert track.measured_fraction > 0.8
    assert track.dx[-1] * WIDTH == pytest.approx(1.5 * 9, abs=0.8)
    assert track.dy[-1] * HEIGHT == pytest.approx(-0.8 * 9, abs=0.8)
    assert track.spread_px(WIDTH, HEIGHT) < 0.8, "a global motion leaves nothing region-dependent"


def test_gradient_motion_bends_the_mesh_like_jello() -> None:
    gradient = 6.0  # px of extra x-shift from the top row to the bottom row, per frame pair

    def displacement(index: int) -> np.ndarray:
        ys = np.linspace(-0.5, 0.5, HEIGHT)
        field = np.zeros((HEIGHT, WIDTH, 2))
        field[..., 0] = index * gradient * ys[:, None]
        return field

    track = _analyze(_frames_from_displacement(displacement, 8))
    _xs, vertex_ys = vertex_positions(WIDTH, HEIGHT)
    unit = (vertex_ys / (HEIGHT - 1)) - 0.5
    measured = track.dx[-1].mean(axis=1) * WIDTH  # column-mean x path per vertex row
    expected = 7 * gradient * unit
    assert np.abs(measured[1:-1] - expected[1:-1]).max() < 1.0, f"{measured} vs {expected}"
    # The outermost vertices only get votes from one side (the frame ends there): the local median
    # centres ~half a cell inward, so the edge rows capture most — not all — of the gradient.
    assert measured[0] / expected[0] > 0.7 and measured[-1] / expected[-1] > 0.7, f"{measured} vs {expected}"


def test_two_depths_move_their_own_vertices() -> None:
    """Parallax: the left half of the scene slides, the right half stands still — the mesh keeps both."""

    def displacement(index: int) -> np.ndarray:
        weight = np.clip(np.linspace(1.5, -0.5, WIDTH), 0.0, 1.0)  # 1 on the left, 0 on the right, soft middle
        field = np.zeros((HEIGHT, WIDTH, 2))
        field[..., 0] = 2.0 * index * weight[None, :]
        return field

    track = _analyze(_frames_from_displacement(displacement, 8))
    left = track.dx[-1][:, 0].mean() * WIDTH
    right = track.dx[-1][:, -1].mean() * WIDTH
    assert left == pytest.approx(14.0, abs=1.5), "the moving side follows its own motion"
    assert abs(right) < 1.5, "the still side stays still"
    assert track.spread_px(WIDTH, HEIGHT) > 2.0, "the disagreement is exactly what a global warp cannot cancel"


def test_a_small_moving_object_does_not_bend_the_mesh() -> None:
    def displacement(index: int) -> np.ndarray:
        return np.zeros((HEIGHT, WIDTH, 2))

    frames = _frames_from_displacement(displacement, 8)
    rng = np.random.default_rng(3)
    for index, frame in enumerate(frames):  # a textured block walking across the static scene
        x = 40 + 12 * index
        block = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
        frame[100:150, x : x + 50] = block
    track = _analyze(frames)
    assert float(np.abs(track.dx[-1]).max()) * WIDTH < 2.0, "the medians keep the walker out of the mesh"
    assert float(np.abs(track.dy[-1]).max()) * HEIGHT < 2.0


def test_mesh_correction_cancels_the_fast_path_and_keeps_the_pan() -> None:
    count = 120
    frames_axis = np.arange(count)
    pan = 0.0008 * frames_axis  # a slow, deliberate drift (fraction per frame)
    shake = 0.01 * np.sin(2.0 * np.pi * frames_axis / 6.0)  # 5 Hz at 30 fps
    dx = np.tile((pan + shake)[:, None, None], (1, VERTEX_ROWS, VERTEX_COLUMNS)).astype(np.float32)
    path = MeshPath(0, dx, np.zeros_like(dx), np.ones(count, dtype=bool))

    correction = mesh_correction(path, smoothing_frames=15.0)
    stabilized = dx + correction[..., 0]
    interior = slice(20, count - 20)
    residual = stabilized[interior, 0, 0] - pan[interior]
    assert float(np.sqrt(np.mean(residual**2))) < 0.35 * float(np.sqrt(np.mean(shake[interior] ** 2)))
    drift = stabilized[interior, 0, 0]
    assert drift[-1] - drift[0] == pytest.approx(pan[interior][-1] - pan[interior][0], abs=0.004)
    assert float(np.abs(correction).max()) <= CORRECTION_LIMIT_FRACTION + 1e-6


def test_apply_mesh_moves_the_content_by_the_field() -> None:
    frame = _canvas(0)[:HEIGHT, :WIDTH]
    correction = np.zeros((VERTEX_ROWS, VERTEX_COLUMNS, 2), dtype=np.float32)
    correction[..., 0] = 10.0 / WIDTH
    correction[..., 1] = -6.0 / HEIGHT
    moved = apply_mesh(frame, correction)
    reference = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result = cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (shift_x, shift_y), _response = cv2.phaseCorrelate(reference, result)
    assert shift_x == pytest.approx(10.0, abs=0.3)
    assert shift_y == pytest.approx(-6.0, abs=0.3)
