from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from space_video_enhancer.pipeline.stabilization import transforms
from space_video_enhancer.pipeline.stabilization.mesh import (
    CORRECTION_LIMIT_FRACTION,
    VERTEX_COLUMNS,
    VERTEX_ROWS,
    MeshCorrections,
    MeshPath,
    analyze_mesh_path,
    apply_mesh,
    mesh_correction,
    vertex_positions,
)
from space_video_enhancer.pipeline.stabilization.stabilize import MESH_HANDOFF_FRAMES, apply_stabilization
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import StabilizationMode, StabilizationSettings
from space_video_enhancer.telemetry import trace as tracing
from space_video_enhancer.telemetry.trace import RenderTrace

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
    analysis = analyze_mesh_path(
        lambda index: frames[index] if 0 <= index < len(frames) else None, Path("clip.mp4"), 0, len(frames) - 1
    )
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
    path = MeshPath(Path("clip.mp4"), 0, dx, np.zeros_like(dx), np.ones(count, dtype=bool))

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


def test_apply_stabilization_mesh_mode_warps_by_the_clip_field() -> None:

    frame = _canvas(0)[:HEIGHT, :WIDTH]
    settings = StabilizationSettings(mode=StabilizationMode.MESH, temporal_smoothing=1.0)
    fields = np.zeros((3, VERTEX_ROWS, VERTEX_COLUMNS, 2), dtype=np.float32)
    fields[..., 0] = 12.0 / WIDTH
    corrections = MeshCorrections(first_frame=0, fields=fields)

    baseline = apply_stabilization(frame, settings, frame_index=1, mesh_corrections=None)  # crop only
    trace = RenderTrace()
    with tracing.active(trace):
        moved = apply_stabilization(frame, settings, frame_index=1, mesh_corrections=corrections)
    assert trace.notes.get("stabilization.result") == "applied"
    assert moved.shape == frame.shape == baseline.shape
    reference = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result = cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (shift_x, _shift_y), _response = cv2.phaseCorrelate(reference, result)
    assert shift_x == pytest.approx(12.0 / 0.9, abs=0.5), "the field moves the content (scaled by the fixed crop)"

    trace = RenderTrace()
    with tracing.active(trace):
        outside = apply_stabilization(frame, settings, frame_index=99, mesh_corrections=corrections)
    assert trace.notes.get("stabilization.result") == "unanalysed"
    assert np.array_equal(outside, baseline), "a frame outside the track passes through (with the constant crop)"


def _measured_shift(reference: np.ndarray, frame: np.ndarray) -> tuple[float, float]:
    gray_a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (shift_x, shift_y), _response = cv2.phaseCorrelate(gray_a, gray_b)
    return float(shift_x), float(shift_y)


def test_mesh_mode_falls_back_to_the_live_path_until_the_field_arrives() -> None:
    """Unanalysed clip + sequential frames = the live translation fallback stabilizes at once."""
    shake = [0.0, 9.0, -7.0, 8.0, -9.0, 7.0]

    def displacement(index: int) -> np.ndarray:
        field = np.zeros((HEIGHT, WIDTH, 2))
        field[..., 0] = shake[index]
        return field

    frames = _frames_from_displacement(displacement, len(shake))
    settings = StabilizationSettings(mode=StabilizationMode.MESH, temporal_smoothing=1.0)
    state = StabilizationState()
    stabilized_first = apply_stabilization(frames[0], settings, frame_index=0, state=state, mesh_corrections=None)
    baseline_first = transforms.stabilization_crop(frames[0], None, settings.margin)
    raw_shifts, corrected_shifts = [], []
    for index, frame in enumerate(frames[1:], start=1):
        trace = RenderTrace()
        with tracing.active(trace):
            stabilized = apply_stabilization(frame, settings, frame_index=index, state=state, mesh_corrections=None)
        assert trace.notes.get("stabilization.mesh") == "fallback"
        assert trace.notes.get("stabilization.result") == "applied", trace.notes
        raw_shifts.append(abs(shake[index]))
        corrected_shifts.append(abs(_measured_shift(stabilized_first, stabilized)[0]))
        baseline = transforms.stabilization_crop(frame, None, settings.margin)
        assert abs(_measured_shift(baseline_first, baseline)[0]) > 4.0, "sanity: the raw crop carries the shake"
    assert max(corrected_shifts) < 0.5 * max(raw_shifts), f"{corrected_shifts} vs {raw_shifts}"


def test_mesh_field_fades_in_from_the_live_path_when_it_lands() -> None:
    """The field arriving mid-sequence crossfades in over the handoff instead of jumping."""
    frame = _canvas(0)[:HEIGHT, :WIDTH]
    settings = StabilizationSettings(mode=StabilizationMode.MESH, temporal_smoothing=1.0)
    fields = np.zeros((60, VERTEX_ROWS, VERTEX_COLUMNS, 2), dtype=np.float32)
    fields[..., 0] = 12.0 / WIDTH
    corrections = MeshCorrections(first_frame=0, fields=fields)
    baseline = apply_stabilization(frame, settings, frame_index=0, mesh_corrections=None)  # crop only

    state = StabilizationState()
    for index in range(3):  # the fallback renders (static clip: a near-identity live path) and arms the handoff
        apply_stabilization(frame, settings, frame_index=index, state=state, mesh_corrections=None)
    assert state.mesh_handoff == 0.0

    trace = RenderTrace()
    with tracing.active(trace):
        first = apply_stabilization(frame, settings, frame_index=3, state=state, mesh_corrections=corrections)
    assert trace.notes.get("stabilization.mesh") == "handoff"
    first_shift = _measured_shift(baseline, first)[0]
    assert abs(first_shift) < 3.0, "the field fades in: the first frame moves a fraction of the 12 px"

    last = first
    for index in range(4, 4 + MESH_HANDOFF_FRAMES):
        trace = RenderTrace()
        with tracing.active(trace):
            last = apply_stabilization(frame, settings, frame_index=index, state=state, mesh_corrections=corrections)
    assert trace.notes.get("stabilization.mesh") == "field", "the handoff ends on the full field"
    assert state.mesh_handoff is None
    assert _measured_shift(baseline, last)[0] == pytest.approx(12.0 / 0.9, abs=0.5)


def test_mesh_handoff_settles_on_the_field_when_paused_or_fresh() -> None:
    frame = _canvas(0)[:HEIGHT, :WIDTH]
    settings = StabilizationSettings(mode=StabilizationMode.MESH, temporal_smoothing=1.0)
    fields = np.zeros((10, VERTEX_ROWS, VERTEX_COLUMNS, 2), dtype=np.float32)
    fields[..., 0] = 12.0 / WIDTH
    corrections = MeshCorrections(first_frame=0, fields=fields)
    baseline = apply_stabilization(frame, settings, frame_index=0, mesh_corrections=None)

    # A fresh state (an export worker) never blends: the field applies at full strength at once.
    fresh = apply_stabilization(
        frame, settings, frame_index=5, state=StabilizationState(), mesh_corrections=corrections
    )
    assert _measured_shift(baseline, fresh)[0] == pytest.approx(12.0 / 0.9, abs=0.5)

    # Paused on the frame that armed the handoff: the re-render settles instead of fading (no motion to hide in).
    state = StabilizationState()
    apply_stabilization(frame, settings, frame_index=2, state=state, mesh_corrections=None)
    paused = apply_stabilization(frame, settings, frame_index=2, state=state, mesh_corrections=corrections)
    assert state.mesh_handoff is None
    assert _measured_shift(baseline, paused)[0] == pytest.approx(12.0 / 0.9, abs=0.5)


def test_mesh_settings_flags() -> None:

    settings = StabilizationSettings(mode=StabilizationMode.MESH, temporal_smoothing=1.0, rolling_shutter_auto=True)
    assert settings.enabled and settings.needs_mesh_analysis
    assert not settings.requires_sequential_processing, "per-frame fields keep the export parallel"
    assert not settings.applies_wobble, "the mesh already absorbs the wobble (measured: correcting twice deforms)"
    assert not settings.mode.uses_auto_motion and not settings.mode.uses_focal_point


def test_margin_setting_widens_the_mesh_budget_and_the_crop() -> None:
    count = 40
    dx = np.tile((0.09 * np.sin(np.arange(count)))[:, None, None], (1, VERTEX_ROWS, VERTEX_COLUMNS))
    path = MeshPath(
        Path("clip.mp4"), 0, dx.astype(np.float32), np.zeros_like(dx, dtype=np.float32), np.ones(count, dtype=bool)
    )
    tight = mesh_correction(path, smoothing_frames=4.0, limit=0.05)
    wide = mesh_correction(path, smoothing_frames=4.0, limit=0.12)
    assert float(np.abs(tight).max()) <= 0.05 + 1e-6
    assert 0.05 < float(np.abs(wide).max()) <= 0.12 + 1e-6, "a wider margin lets more correction through"

    frame = _canvas(0)[:HEIGHT, :WIDTH]

    for margin in (0.05, 0.12):
        settings = StabilizationSettings(mode=StabilizationMode.MESH, margin=margin)
        output = apply_stabilization(frame, settings, frame_index=0, mesh_corrections=None)
        expected = transforms.stabilization_crop(frame, None, margin)
        assert np.array_equal(output, expected), f"the crop follows the margin setting ({margin})"
