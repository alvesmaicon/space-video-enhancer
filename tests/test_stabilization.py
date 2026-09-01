from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from space_video_enhancer.pipeline.stabilization import rolling_shutter, transforms
from space_video_enhancer.pipeline.stabilization.estimators import (
    estimate_camera_lock_transform,
    estimate_inter_frame_transform,
    estimate_translation_shift,
    reference_cache_stats,
)
from space_video_enhancer.pipeline.stabilization.global_motion import MotionVectors
from space_video_enhancer.pipeline.stabilization.stabilize import (
    CORRECTION_BUDGET_FRACTION,
    apply_stabilization,
    estimate_correction,
    lock_weight,
    path_decay,
    smoothing_time_constant,
)
from space_video_enhancer.pipeline.stabilization.state import MAX_FRAME_GAP, StabilizationState, frame_gap
from space_video_enhancer.pipeline.stabilization.wobble import (
    WobbleTrack,
    analyze_wobble,
    detect_readout_axis,
    filter_wobble,
    wobble_window_frames,
)
from space_video_enhancer.settings import NormalizedRect, ReadoutAxis, StabilizationMode, StabilizationSettings
from space_video_enhancer.telemetry import trace as tracing
from space_video_enhancer.telemetry.trace import RenderTrace

SHIFT_DOWN, SHIFT_LEFT = 5, 7
FOCUS = NormalizedRect(0.10, 0.12, 0.28, 0.42)  # around the green rectangle of the textured frame
CENTER = NormalizedRect(0.3, 0.3, 0.4, 0.4)


@pytest.fixture
def previous_frame(textured_frame: np.ndarray) -> np.ndarray:
    return np.roll(textured_frame, (SHIFT_DOWN, -SHIFT_LEFT), axis=(0, 1))


# --- synthetic hand-held sequence -------------------------------------------------------------


def shaky_sequence(count: int = 40, jitter: float = 5.0, pan: float = 16.0) -> tuple[list[np.ndarray], np.ndarray]:
    """Frames of a textured scene seen through a jittering window that also pans ``pan`` px to the right."""
    rng = np.random.default_rng(7)
    width, height = 480, 270  # 5% (the correction budget) is 24 x 13 px here, well above the jitter
    canvas = cv2.resize(
        rng.integers(0, 255, (45, 75, 3), dtype=np.uint8), (width + 120, height + 120), interpolation=cv2.INTER_CUBIC
    )
    for _ in range(40):
        x, y = int(rng.integers(0, width + 90)), int(rng.integers(0, height + 100))
        cv2.rectangle(canvas, (x, y), (x + 24, y + 16), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    drift = np.stack([np.linspace(0.0, pan, count), np.zeros(count)], axis=1)
    shake = rng.normal(0.0, jitter, (count, 2))
    shake[0] = 0.0  # the first frame sits on the intended path: the virtual camera starts there
    positions = 60.0 + shake + drift
    frames = []
    for x, y in positions:
        matrix = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y]], dtype=np.float32)
        frames.append(cv2.warpAffine(canvas, matrix, (width, height), flags=cv2.INTER_LINEAR))
    return frames, positions


def measured_positions(frames: list[np.ndarray], reference: np.ndarray) -> np.ndarray:
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shifts = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), _response = cv2.phaseCorrelate(reference_gray, gray)
        shifts.append((dx, dy))
    return np.array(shifts)


def jitter_std(positions: np.ndarray) -> float:
    """Spread around the straight line joining the first and last positions (the intended pan)."""
    count = len(positions)
    trend = np.stack(
        [np.linspace(positions[0, axis], positions[-1, axis], count) for axis in range(positions.shape[1])], axis=1
    )
    return float(np.std(positions - trend))


def stabilize_sequence(
    frames: list[np.ndarray],
    mode: StabilizationMode,
    smoothing: float,
    *,
    intensity: float = 1.0,
    reference: np.ndarray | None = None,
    reference_missing_at: frozenset[int] = frozenset(),
) -> tuple[list[np.ndarray], StabilizationState]:
    settings = StabilizationSettings(
        mode=mode, intensity=intensity, temporal_smoothing=smoothing, focus_rect=CENTER, auto_crop=False
    )
    state = StabilizationState()
    outputs = []
    for index, frame in enumerate(frames):
        current_reference = None if index in reference_missing_at else reference
        outputs.append(
            apply_stabilization(frame, settings, frame_index=index, state=state, reference_frame=current_reference)
        )
    return outputs, state


@pytest.mark.parametrize("mode", [StabilizationMode.AUTO, StabilizationMode.SIMILARITY])
def test_auto_modes_remove_jitter_and_keep_the_pan(mode: StabilizationMode) -> None:
    frames, _true_positions = shaky_sequence()
    outputs, _state = stabilize_sequence(frames, mode, smoothing=0.5)
    before = measured_positions(frames, frames[0])
    after = measured_positions(outputs, frames[0])
    assert jitter_std(after) < jitter_std(before) * 0.4
    # The slow pan survives (this is not a camera lock): the clip still ends well to the right of where it started.
    assert abs(after[-1, 0] - after[0, 0]) > 8.0


def test_more_smoothing_is_steadier() -> None:
    frames, _positions = shaky_sequence()
    gentle, _ = stabilize_sequence(frames, StabilizationMode.AUTO, smoothing=0.2)
    strong, _ = stabilize_sequence(frames, StabilizationMode.AUTO, smoothing=0.9)
    assert jitter_std(measured_positions(strong, frames[0])) < jitter_std(measured_positions(gentle, frames[0]))


def test_smoothing_maps_to_a_time_constant() -> None:
    assert smoothing_time_constant(0.0) == pytest.approx(2.0)
    assert smoothing_time_constant(1.0) == pytest.approx(60.0)
    assert smoothing_time_constant(0.35) < smoothing_time_constant(0.8)
    assert path_decay(0.0) == pytest.approx(0.5)
    assert 0.98 < path_decay(1.0) < 1.0
    assert lock_weight(0.0) == pytest.approx(1.0)
    assert lock_weight(1.0) == pytest.approx(0.3)


def test_same_frame_reuses_the_correction(textured_frame: np.ndarray, previous_frame: np.ndarray) -> None:
    settings = StabilizationSettings(mode=StabilizationMode.AUTO, temporal_smoothing=0.5, auto_crop=False)
    state = StabilizationState()
    apply_stabilization(previous_frame, settings, frame_index=9, state=state)
    apply_stabilization(textured_frame, settings, frame_index=10, state=state)
    first = state.correction
    assert first is not None
    apply_stabilization(textured_frame, settings, frame_index=10, state=state)
    assert state.correction is first  # not accumulated twice
    assert state.correction_frame_index == 10


def test_dropped_frames_do_not_restart_the_path() -> None:
    """Rendering only every 2nd/3rd frame (what the preview does under load) must still stabilize."""
    frames, _positions = shaky_sequence(count=40)
    rendered = [index for index in range(40) if index % 5 not in (1, 3)]  # drops two of every five frames
    settings = StabilizationSettings(mode=StabilizationMode.AUTO, temporal_smoothing=0.5, auto_crop=False)
    state = StabilizationState()
    outputs = []
    for index in rendered:
        outputs.append(apply_stabilization(frames[index], settings, frame_index=index, state=state))
        if index > 0:
            assert state.correction is not None, f"frame {index} restarted the path"
    before = measured_positions([frames[index] for index in rendered], frames[0])
    after = measured_positions(outputs, frames[0])
    assert jitter_std(after) < jitter_std(before) * 0.5


def test_frame_gap_rules() -> None:
    assert frame_gap(10, 11) == 1 and frame_gap(10, 13) == 3 and frame_gap(10, 8) == -2
    assert frame_gap(10, 10) is None and frame_gap(10, 10 + MAX_FRAME_GAP + 1) is None
    state = StabilizationState(prev_frame_index=10)
    assert not state.breaks_sequence(11) and not state.breaks_sequence(10 + MAX_FRAME_GAP)
    assert state.breaks_sequence(10) and state.breaks_sequence(40) and not state.breaks_sequence(None)


def test_sequence_break_restarts_the_path(textured_frame: np.ndarray, previous_frame: np.ndarray) -> None:
    settings = StabilizationSettings(mode=StabilizationMode.AUTO, temporal_smoothing=0.5, auto_crop=False)
    state = StabilizationState()
    apply_stabilization(previous_frame, settings, frame_index=9, state=state)
    apply_stabilization(textured_frame, settings, frame_index=10, state=state)
    assert state.correction is not None
    assert apply_stabilization(previous_frame, settings, frame_index=30, state=state) is previous_frame
    assert state.correction is None and state.correction_frame_index == 30


def test_corrections_are_rescaled_when_the_same_sequence_is_rendered_at_another_size() -> None:
    """The preview renders scrub/playback frames at 640 px and the resting frame in full size."""
    frames, _positions = shaky_sequence(count=8)
    small = [cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for frame in frames]
    settings = StabilizationSettings(mode=StabilizationMode.AUTO, temporal_smoothing=0.5, auto_crop=False)

    reference = StabilizationState()
    for index, frame in enumerate(frames):
        expected = estimate_correction(frame, settings, index, reference, None)
    assert expected is not None

    state = StabilizationState()
    for index, frame in enumerate(small):
        estimate_correction(frame, settings, index, state, None)
    idle = estimate_correction(frames[-1], settings, len(frames) - 1, state, None)  # same index, full size
    assert idle is not None
    assert idle[:, 2] == pytest.approx(expected[:, 2], abs=1.0)
    assert state.correction_width == frames[-1].shape[1]

    # And the other way round: continuing a full-size sequence with a small frame keeps the units.
    resumed = estimate_correction(small[-1], settings, len(frames) - 1, state, None)
    assert resumed is not None and resumed[:, 2] == pytest.approx(expected[:, 2] / 2.0, abs=0.5)


def test_camera_lock_holds_the_last_correction_when_the_target_is_lost() -> None:
    frames, _positions = shaky_sequence(count=6)
    settings = StabilizationSettings(mode=StabilizationMode.CAMERA_LOCK, focus_rect=CENTER, auto_crop=False)
    state = StabilizationState()
    locked = estimate_correction(frames[1], settings, 1, state, frames[0])
    assert locked is not None
    held = estimate_correction(frames[2], settings, 2, state, None)
    assert held is not None and np.array_equal(held, locked)
    assert estimate_correction(frames[3], settings, 3, StabilizationState(), None) is None  # nothing to hold


def test_hybrid_dead_reckons_while_lost_and_eases_back() -> None:
    frames, _positions = shaky_sequence(count=12)
    outputs, state = stabilize_sequence(
        frames, StabilizationMode.HYBRID, smoothing=0.0, reference=frames[0], reference_missing_at=frozenset({4, 5, 6})
    )
    positions = measured_positions(outputs, frames[0])
    # Locked frames sit on the reference; dead-reckoned frames (4-6) stay within a couple of pixels of it.
    assert np.abs(positions[1:4]).max() < 1.0
    assert np.abs(positions[4:7]).max() < 3.0
    assert np.abs(positions[9:]).max() < 1.0
    assert state.lock_lost_frames == 0


def test_translation_estimator_recovers_global_shift(textured_frame: np.ndarray, previous_frame: np.ndarray) -> None:
    state = StabilizationState()
    assert estimate_translation_shift(previous_frame, 9, state) is None  # first frame only primes the state
    shift = estimate_translation_shift(textured_frame, 10, state)
    assert shift is not None
    assert shift[0] == pytest.approx(-SHIFT_LEFT, abs=0.5)
    assert shift[1] == pytest.approx(SHIFT_DOWN, abs=0.5)
    # Same frame index again -> cached
    assert estimate_translation_shift(textured_frame, 10, state) == shift


def test_inter_frame_similarity_recovers_global_shift(textured_frame: np.ndarray, previous_frame: np.ndarray) -> None:
    state = StabilizationState()
    assert estimate_inter_frame_transform(previous_frame, 9, state, StabilizationMode.SIMILARITY) is None
    transform = estimate_inter_frame_transform(textured_frame, 10, state, StabilizationMode.SIMILARITY)
    assert transform is not None and transform.shape == (2, 3)
    assert transform[0, 2] == pytest.approx(-SHIFT_LEFT, abs=1.0)
    assert transform[1, 2] == pytest.approx(SHIFT_DOWN, abs=1.0)
    assert transform[0, 0] == pytest.approx(1.0, abs=0.02)


def test_translation_estimator_resets_on_sequence_break(textured_frame: np.ndarray, previous_frame: np.ndarray) -> None:
    state = StabilizationState()
    estimate_translation_shift(previous_frame, 9, state)
    assert estimate_translation_shift(textured_frame, 42, state) is None
    assert state.prev_frame_index == 42


def test_camera_lock_maps_frame_onto_reference(textured_frame: np.ndarray, previous_frame: np.ndarray) -> None:
    transform = estimate_camera_lock_transform(textured_frame, previous_frame, FOCUS)
    assert transform is not None and transform.shape == (2, 3)
    assert transform[0, 2] == pytest.approx(-SHIFT_LEFT, abs=1.5)
    assert transform[1, 2] == pytest.approx(SHIFT_DOWN, abs=1.5)
    assert transform[0, 0] == pytest.approx(1.0, abs=0.02)


def test_camera_lock_reuses_the_reference_features(textured_frame: np.ndarray, previous_frame: np.ndarray) -> None:
    reference = previous_frame.copy()
    before = dict(reference_cache_stats)
    first = estimate_camera_lock_transform(textured_frame, reference, FOCUS)
    second = estimate_camera_lock_transform(textured_frame, reference, FOCUS)
    assert first is not None and second is not None and np.array_equal(first, second)
    assert reference_cache_stats["misses"] == before["misses"] + 1
    assert reference_cache_stats["hits"] == before["hits"] + 1
    estimate_camera_lock_transform(textured_frame, previous_frame.copy(), FOCUS)  # another array: computed again
    assert reference_cache_stats["misses"] == before["misses"] + 2


def test_apply_stabilization_keeps_frame_size_and_moves_content(
    textured_frame: np.ndarray, previous_frame: np.ndarray
) -> None:
    # Smoothing 1.0 = a 60-frame time constant: the second frame is pulled almost fully onto the first.
    settings = StabilizationSettings(
        mode=StabilizationMode.AUTO, intensity=1.0, temporal_smoothing=1.0, auto_crop=False
    )
    state = StabilizationState()
    apply_stabilization(previous_frame, settings, frame_index=9, state=state)
    stabilized = apply_stabilization(textured_frame, settings, frame_index=10, state=state)
    assert stabilized.shape == textured_frame.shape
    interior = (slice(40, 230), slice(40, 440))
    error_before = np.abs(textured_frame[interior].astype(int) - previous_frame[interior].astype(int)).mean()
    error_after = np.abs(stabilized[interior].astype(int) - previous_frame[interior].astype(int)).mean()
    assert error_after < error_before * 0.5


def test_disabled_or_zero_intensity_returns_input(textured_frame: np.ndarray) -> None:
    assert apply_stabilization(textured_frame, StabilizationSettings()) is textured_frame
    settings = StabilizationSettings(mode=StabilizationMode.CAMERA_LOCK, intensity=0.0)
    assert apply_stabilization(textured_frame, settings, reference_frame=textured_frame) is textured_frame


def test_camera_lock_without_reference_only_applies_the_constant_crop(textured_frame: np.ndarray) -> None:
    plain = StabilizationSettings(mode=StabilizationMode.CAMERA_LOCK, auto_crop=False)
    assert apply_stabilization(textured_frame, plain, reference_frame=None) is textured_frame
    cropping = StabilizationSettings(mode=StabilizationMode.CAMERA_LOCK, auto_crop=True)
    cropped = apply_stabilization(textured_frame, cropping, reference_frame=None)
    assert cropped.shape == textured_frame.shape and not np.array_equal(cropped, textured_frame)
    # Same crop as a frame that did get a correction: the zoom does not pop.
    expected = transforms.stabilization_crop(textured_frame, None, CORRECTION_BUDGET_FRACTION)
    assert np.array_equal(cropped, expected)


def test_crop_rect_is_constant_for_corrections_within_the_margin() -> None:
    shape = (270, 480, 3)
    margin = CORRECTION_BUDGET_FRACTION
    baseline = transforms.crop_rect_for(None, shape, margin)
    assert baseline == (24, 14, 432, 242)
    for shift in ((0.0, 0.0), (3.0, 2.0), (-8.0, 5.0), (480 * margin - 3, -(270 * margin - 3))):
        assert transforms.crop_rect_for(transforms.translation_to_affine(*shift), shape, margin) == baseline
    beyond = transforms.crop_rect_for(transforms.translation_to_affine(60.0, 0.0), shape, margin)
    assert beyond != baseline and beyond[0] >= 60 - 1 and beyond[1] == baseline[1]


def test_fit_within_margin_keeps_the_crop_rect_constant_for_geometric_corrections() -> None:
    """Rotation/zoom/keystone escape the margin even with the translation capped: the fit must saturate them.

    Measured on real walking footage before the fix: the correction escaped on 51% of the frames and the
    crop fell back to that frame's valid area — the output size pulsed by up to 4% between frames.
    """
    shape = (270, 480, 3)
    margin = CORRECTION_BUDGET_FRACTION
    baseline = transforms.crop_rect_for(None, shape, margin)

    fitting = [
        transforms.translation_to_affine(3.0, 2.0),
        cv2.getRotationMatrix2D((240.0, 135.0), 0.0, 1.12).astype(np.float32),  # zoom-in covers past the margin
    ]
    for transform in fitting:
        assert transforms.fit_within_margin(transform, shape, margin) is transform, "a fitting correction passes"

    keystone = np.array([[1.0, 0.05, 0.0], [0.0, 1.06, 0.0], [8e-4, 0.0, 1.0]], dtype=np.float32)
    adversarial = [
        cv2.getRotationMatrix2D((240.0, 135.0), 8.0, 1.0).astype(np.float32),  # rotation swings the corners
        cv2.getRotationMatrix2D((240.0, 135.0), 0.0, 0.86).astype(np.float32),  # zoom-out reveals the border
        keystone,
        transforms.compose(keystone, transforms.translation_to_affine(20.0, -12.0)),
    ]
    for transform in adversarial:
        fitted = transforms.fit_within_margin(transform, shape, margin)
        assert fitted is not transform, "an escaping correction must be reduced"
        assert transforms.crop_rect_for(fitted, shape, margin) == baseline, "the crop never grows"
        assert not np.allclose(fitted, transforms.identity_like(fitted), atol=1e-4), (
            "the fit saturates the correction, it does not discard it"
        )


def test_auto_crop_correction_saturates_instead_of_growing_the_crop(textured_frame: np.ndarray) -> None:
    """End to end: a big rotated correction with auto crop on still comes out at the frame size, fitted."""
    settings = StabilizationSettings(mode=StabilizationMode.SIMILARITY, temporal_smoothing=0.0, auto_crop=True)
    state = StabilizationState()
    rotated = cv2.warpAffine(
        textured_frame,
        cv2.getRotationMatrix2D((textured_frame.shape[1] / 2.0, textured_frame.shape[0] / 2.0), 9.0, 1.0),
        (textured_frame.shape[1], textured_frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    trace = RenderTrace()
    with tracing.active(trace):
        apply_stabilization(textured_frame, settings, frame_index=0, state=state)
        output = apply_stabilization(rotated, settings, frame_index=1, state=state)
    assert output.shape == textured_frame.shape
    assert trace.notes.get("stabilization.margin_fit") == 1, "the 9-degree correction had to saturate"


def test_stabilization_crop_returns_same_size(textured_frame: np.ndarray) -> None:
    transform = transforms.translation_to_affine(12.0, 8.0)
    warped = transforms.warp(textured_frame, transform)
    assert transforms.stabilization_crop(warped, transform, 0.05).shape == textured_frame.shape
    assert transforms.stabilization_crop(warped, transforms.translation_to_affine(200.0, 0.0), 0.05).shape == (
        textured_frame.shape
    )
    assert transforms.stabilization_crop(textured_frame, None, 0.0) is textured_frame


def test_blend_and_negligible() -> None:
    translation = transforms.translation_to_affine(10.0, -4.0)
    half = transforms.blend_with_identity(translation, 0.5)
    assert half[0, 2] == pytest.approx(5.0) and half[1, 2] == pytest.approx(-2.0)
    assert transforms.is_negligible(transforms.translation_to_affine(0.2, 0.1))
    assert not transforms.is_negligible(translation)


def test_compose_clamp_and_identity() -> None:
    first = transforms.translation_to_affine(3.0, 1.0)
    second = transforms.translation_to_affine(-1.0, 2.0)
    composed = transforms.compose(first, second)
    assert composed.shape == (2, 3)
    assert composed[0, 2] == pytest.approx(2.0) and composed[1, 2] == pytest.approx(3.0)
    homography = transforms.compose(transforms.affine_to_homography(first), second)
    assert homography.shape == (3, 3) and homography[0, 2] == pytest.approx(2.0)
    clamped = transforms.clamp_translation(transforms.translation_to_affine(100.0, -100.0), (100, 200, 3), 0.1)
    assert clamped[0, 2] == pytest.approx(20.0) and clamped[1, 2] == pytest.approx(-10.0)
    assert transforms.identity_like(first).shape == (2, 3)
    assert np.array_equal(transforms.identity_like(homography), np.eye(3, dtype=np.float32))


def test_zoom_out_fit_keeps_corners_inside_frame() -> None:
    shape = (270, 480, 3)
    transform = transforms.translation_to_affine(30.0, 20.0)
    fitted = transforms.fit_with_zoom_out(transform, shape)
    corners = transforms.project_corners(fitted, shape[1], shape[0])
    assert corners[:, 0].min() >= -0.5 and corners[:, 0].max() <= shape[1] - 0.5
    assert corners[:, 1].min() >= -0.5 and corners[:, 1].max() <= shape[0] - 0.5


def test_rescale_homography_roundtrip() -> None:
    affine = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]], dtype=np.float32)
    rescaled = transforms.rescale_affine(affine, 0.5)  # estimated at half resolution -> double the translation
    assert rescaled[0, 2] == pytest.approx(6.0) and rescaled[1, 2] == pytest.approx(-4.0)


def test_state_reset_clears_everything() -> None:
    state = StabilizationState(prev_frame_index=3, lock_lost_frames=2)
    state.reset()
    assert state == StabilizationState()


# --- rolling shutter ------------------------------------------------------------------------------


def _bars_scene() -> np.ndarray:
    frame = np.full((180, 320, 3), 40, dtype=np.uint8)
    for x in (60, 140, 220):
        frame[:, x : x + 6] = 230
    return frame


def _edge_columns(frame: np.ndarray, row: int) -> list[int]:
    luma = frame[row, :, 1].astype(int)
    return [int(x) for x in np.flatnonzero(np.diff(luma) > 60) + 1]


def test_rolling_shutter_correction_straightens_a_skewed_frame() -> None:
    scene = _bars_scene()
    skewed = rolling_shutter.skew(scene, (16.0, 0.0), 1.0)  # the scene moved 16 px/frame during a full readout
    top, bottom = _edge_columns(skewed, 2), _edge_columns(skewed, 177)
    assert bottom[0] - top[0] >= 14, "the simulated shutter leans the bars"
    fixed = rolling_shutter.correct(skewed, (16.0, 0.0))  # gradient = displacement x readout fraction
    top, bottom = _edge_columns(fixed, 2), _edge_columns(fixed, 177)
    assert abs(bottom[0] - top[0]) <= 1 and abs(top[0] - 60) <= 1, "after correction the bars stand upright"
    assert rolling_shutter.correct(scene, (0.4, 0.0)) is scene, "sub-pixel skew is left alone"


def _edge_rows(frame: np.ndarray, column: int) -> list[int]:
    luma = frame[:, column, 1].astype(int)
    return [int(y) for y in np.flatnonzero(np.diff(luma) > 60) + 1]


def test_column_readout_skews_and_unskews_along_the_width() -> None:
    """A phone held upright reads along the frame's columns: horizontal bars lean when the scene moves vertically."""
    scene = np.ascontiguousarray(np.transpose(_bars_scene(), (1, 0, 2)))  # 320x180: the bars are now horizontal
    skewed = rolling_shutter.skew(scene, (0.0, 16.0), 1.0, ReadoutAxis.COLUMNS)
    left, right = _edge_rows(skewed, 2), _edge_rows(skewed, 177)
    assert right[0] - left[0] >= 14, "the simulated readout leans the bars"
    fixed = rolling_shutter.correct(skewed, (0.0, 16.0), ReadoutAxis.COLUMNS)
    left, right = _edge_rows(fixed, 2), _edge_rows(fixed, 177)
    assert abs(right[0] - left[0]) <= 1 and abs(left[0] - 60) <= 1, "after the correction the bars lie flat"
    wrong_axis = rolling_shutter.correct(skewed, (0.0, 16.0), ReadoutAxis.ROWS)
    left, right = _edge_rows(wrong_axis, 2), _edge_rows(wrong_axis, 177)
    assert right[0] - left[0] >= 12, "shifting rows cannot undo a distortion that varies along the columns"


def test_center_displacement_is_the_opposite_of_where_delta_sends_the_centre() -> None:
    delta = transforms.translation_to_affine(-12.0, 5.0)  # aligns this frame onto the previous one
    assert rolling_shutter.center_displacement(delta, (180, 320)) == pytest.approx((12.0, -5.0))
    homography = transforms.affine_to_homography(delta)
    assert rolling_shutter.center_displacement(homography, (180, 320)) == pytest.approx((12.0, -5.0))


def test_apply_stabilization_unskews_with_the_mode_off_and_needs_the_previous_frame() -> None:
    scene = _bars_scene()
    settings = StabilizationSettings(rolling_shutter=1.0)
    assert settings.corrects_rolling_shutter and settings.requires_sequential_processing and not settings.enabled
    assert apply_stabilization(scene, settings, frame_index=0, state=None) is scene, "no state: nothing to measure"

    state = StabilizationState()
    previous = rolling_shutter.skew(scene, (16.0, 0.0), 1.0)
    current = rolling_shutter.skew(np.roll(scene, 16, axis=1), (16.0, 0.0), 1.0)
    assert apply_stabilization(previous, settings, frame_index=0, state=state) is previous, "first frame: primes"
    fixed = apply_stabilization(current, settings, frame_index=1, state=state)
    assert fixed is not current
    top, bottom = _edge_columns(fixed, 2), _edge_columns(fixed, 177)
    assert abs(bottom[0] - top[0]) <= 2, f"bars should be upright after the unskew, got {top} / {bottom}"
    assert apply_stabilization(scene, StabilizationSettings(), frame_index=2, state=state) is scene


def _wobbly_sequence(scene: np.ndarray, count: int, amplitude: float) -> tuple[list[np.ndarray], list[float]]:
    """A static scene seen through a rolling shutter whose vertical stretch oscillates frame to frame (jello)."""
    gradients = [amplitude * np.sin(2.0 * np.pi * k / 6.0) for k in range(count)]
    return [rolling_shutter.wobble(scene, (0.0, g)) for g in gradients], gradients


def test_row_motion_gradient_reads_the_change_of_wobble_between_frames(textured_frame: np.ndarray) -> None:
    frames, gradients = _wobbly_sequence(textured_frame, 4, 14.0)
    state = StabilizationState()
    assert estimate_inter_frame_transform(frames[0], 0, state, StabilizationMode.AUTO) is None, "primes"
    estimate_inter_frame_transform(frames[1], 1, state, StabilizationMode.AUTO)
    measured = rolling_shutter.row_motion_gradient(state.last_vectors, textured_frame.shape[0])
    assert measured is not None
    expected = gradients[1] - gradients[0]  # how much the intra-frame distortion changed
    assert abs(measured[1] - expected) < 2.5 and abs(measured[0]) < 2.5, (measured, expected)
    few = MotionVectors(
        state.last_vectors.previous[:6], state.last_vectors.current[:6], np.ones(6, bool), state.last_vectors.size, 0, 1
    )
    assert rolling_shutter.row_motion_gradient(few, textured_frame.shape[0]) is None, "too few points: no gradient"


def _pair_vectors(first: np.ndarray, second: np.ndarray) -> MotionVectors:
    state = StabilizationState()
    estimate_inter_frame_transform(first, 0, state, StabilizationMode.AUTO)
    estimate_inter_frame_transform(second, 1, state, StabilizationMode.AUTO)
    assert state.last_vectors is not None
    return state.last_vectors  # type: ignore[no-any-return]


def test_motion_gradient_reads_the_readout_axis_and_ignores_roll_and_zoom(textured_frame: np.ndarray) -> None:
    height, width = textured_frame.shape[:2]
    stretched = rolling_shutter.wobble(textured_frame, (10.0, 0.0), ReadoutAxis.COLUMNS)  # wider on the right
    vectors = _pair_vectors(textured_frame, stretched)
    along_columns = rolling_shutter.motion_gradient(vectors, width, ReadoutAxis.COLUMNS)
    along_rows = rolling_shutter.motion_gradient(vectors, height, ReadoutAxis.ROWS)
    assert along_columns is not None and abs(along_columns[0] - 10.0) < 2.5 and abs(along_columns[1]) < 2.5
    # Read along the rows, the same field is a zoom plus the opposite vertical stretch: one pair of frames cannot
    # tell the two axes apart (the analysis uses the correlation with the camera's acceleration over time).
    assert along_rows is not None and abs(along_rows[0]) < 2.5 and abs(along_rows[1] + 10.0 * height / width) < 2.5

    centre = (width / 2.0, height / 2.0)
    for label, matrix in (
        ("roll", cv2.getRotationMatrix2D(centre, 1.0, 1.0)),
        ("zoom", cv2.getRotationMatrix2D(centre, 0.0, 1.03)),
    ):
        moved = cv2.warpAffine(textured_frame, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
        vectors = _pair_vectors(textured_frame, moved)
        for axis, length in ((ReadoutAxis.ROWS, height), (ReadoutAxis.COLUMNS, width)):
            gradient = rolling_shutter.motion_gradient(vectors, length, axis)
            assert gradient is not None, (label, axis)
            assert abs(gradient[0]) < 2.5 and abs(gradient[1]) < 2.5, (
                label,
                axis,
                gradient,
            )  # a 1° roll would read ~4.7 px


def _band_shift(image: np.ndarray, reference: np.ndarray, rows: slice) -> float:
    """Vertical offset of ``image`` against ``reference`` inside a horizontal band (phase correlation)."""
    to_gray = lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)[rows, 20:-20]  # noqa: E731
    (_dx, dy), _response = cv2.phaseCorrelate(to_gray(reference), to_gray(image))
    return float(dy)


def _residual_gradient(image: np.ndarray, reference: np.ndarray) -> float:
    """How much more the bottom band is displaced than the top one: the wobble left in ``image``."""
    return _band_shift(image, reference, slice(15, 95)) - _band_shift(image, reference, slice(175, 255))


def _band_shift_x(image: np.ndarray, reference: np.ndarray, columns: slice) -> float:
    """Horizontal offset of ``image`` against ``reference`` inside a vertical band (phase correlation)."""
    to_gray = lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)[20:-20, columns]  # noqa: E731
    (dx, _dy), _response = cv2.phaseCorrelate(to_gray(reference), to_gray(image))
    return float(dx)


def _residual_gradient_x(image: np.ndarray, reference: np.ndarray) -> float:
    """How much more the right band is displaced than the left one: the column-wise wobble left in ``image``."""
    return _band_shift_x(image, reference, slice(15, 95)) - _band_shift_x(image, reference, slice(385, 465))


def _shaky_rolling_shutter_sequence(
    scene: np.ndarray, count: int, axis: ReadoutAxis, readout: float = 0.7
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """A camera shaking over a static scene, read by a rolling shutter along ``axis``: ``(distorted, clean)`` frames.

    Each frame's distortion is the readout time times the displacement since the previous frame — the
    physics the axis detection relies on.
    """
    distorted: list[np.ndarray] = []
    clean: list[np.ndarray] = []
    position = np.zeros(2, dtype=int)
    for index in range(count):
        velocity = np.array(
            [round(16.0 * np.sin(2.0 * np.pi * index / 7.0)), round(9.0 * np.sin(2.0 * np.pi * index / 5.0))]
        )
        position = position + velocity
        moved = np.roll(scene, (int(position[1]), int(position[0])), axis=(0, 1))
        clean.append(moved)
        distorted.append(rolling_shutter.skew(moved, (float(velocity[0]), float(velocity[1])), readout, axis))
    return distorted, clean


def test_wobble_analysis_detects_the_readout_axis_and_the_setting_can_force_it(textured_frame: np.ndarray) -> None:
    window = wobble_window_frames(30.0)
    along_rows, _clean_rows = _shaky_rolling_shutter_sequence(textured_frame, 40, ReadoutAxis.ROWS)
    along_columns, clean = _shaky_rolling_shutter_sequence(textured_frame, 40, ReadoutAxis.COLUMNS)
    still = [textured_frame] * 30

    rows_track = analyze_wobble(_read_from(along_rows), Path("rows.mp4"), 0, 39, window).track
    assert rows_track is not None and rows_track.detected_axis is ReadoutAxis.ROWS
    assert rows_track.correlation(ReadoutAxis.ROWS) > 0.8 > rows_track.correlation(ReadoutAxis.COLUMNS) + 0.3
    columns_track = analyze_wobble(_read_from(along_columns), Path("columns.mp4"), 0, 39, window).track
    assert columns_track is not None and columns_track.detected_axis is ReadoutAxis.COLUMNS
    assert columns_track.correlation(ReadoutAxis.COLUMNS) > 0.8 > columns_track.correlation(ReadoutAxis.ROWS) + 0.3
    upright = analyze_wobble(_read_from(still), Path("still.mp4"), 0, 29, window, rotation=90).track
    landscape = analyze_wobble(_read_from(still), Path("still.mp4"), 0, 29, window, rotation=0).track
    assert upright is not None and upright.detected_axis is ReadoutAxis.COLUMNS, "no motion: the rotation decides"
    assert landscape is not None and landscape.detected_axis is ReadoutAxis.ROWS
    assert detect_readout_axis(0.2, 0.25, 270) is ReadoutAxis.COLUMNS, "too weak a signature: the rotation decides"
    assert detect_readout_axis(0.6, 0.1, 270) is ReadoutAxis.ROWS, "a clear signature beats the rotation"
    assert detect_readout_axis(0.5, 0.45, 0) is ReadoutAxis.ROWS, "too close to call: the rotation decides"

    automatic = StabilizationSettings(rolling_shutter_auto=True)
    assert automatic.needs_wobble_analysis and StabilizationSettings(rolling_shutter=0.5).needs_wobble_analysis
    assert not StabilizationSettings(rolling_shutter=0.5, readout_axis=ReadoutAxis.ROWS).needs_wobble_analysis
    assert not StabilizationSettings().needs_wobble_analysis
    before: list[float] = []
    after: list[float] = []
    for index in range(window, 40 - window):
        trace = RenderTrace()
        with tracing.active(trace):
            corrected = apply_stabilization(
                along_columns[index], automatic, frame_index=index, state=None, wobble_track=columns_track
            )
        assert trace.notes["rolling_shutter.axis"] == "columns"
        assert trace.notes["rolling_shutter.axis_source"] == "detected"
        before.append(abs(_residual_gradient_x(along_columns[index], clean[index])))
        after.append(abs(_residual_gradient_x(corrected, clean[index])))
    assert np.mean(before) > 3.0 and np.mean(after) < 0.3 * np.mean(before), (np.mean(before), np.mean(after))

    forced = replace(automatic, readout_axis=ReadoutAxis.ROWS)
    trace = RenderTrace()
    with tracing.active(trace):
        apply_stabilization(along_columns[15], forced, frame_index=15, state=None, wobble_track=columns_track)
    assert trace.notes["rolling_shutter.axis"] == "rows" and trace.notes["rolling_shutter.axis_source"] == "forced"
    trace = RenderTrace()
    with tracing.active(trace):
        apply_stabilization(along_columns[15], automatic, frame_index=15, state=None)
    assert trace.notes["rolling_shutter.axis_source"] == "default", "no analysis yet: rows"


def _read_from(frames: list[np.ndarray]) -> Callable[[int], np.ndarray | None]:
    return lambda index: frames[index] if 0 <= index < len(frames) else None


def test_wobble_analysis_measures_each_frame_and_the_correction_flattens_the_jello(textured_frame: np.ndarray) -> None:
    frames, gradients = _wobbly_sequence(textured_frame, 40, 14.0)
    window = wobble_window_frames(30.0)
    progress: list[tuple[int, int]] = []
    analysis = analyze_wobble(
        _read_from(frames),
        Path("jello.mp4"),
        0,
        39,
        window,
        progress=lambda done, total: progress.append((done, total)),
    )
    track = analysis.track
    assert track is not None and analysis.frames_read == 40 and not analysis.cancelled
    assert track.count == 40 and track.covers(0, 39) and not track.covers(0, 40) and progress[-1] == (40, 40)
    assert track.measured_fraction > 0.9, "nearly every consecutive pair yields a gradient on this texture"

    settings = StabilizationSettings(rolling_shutter_auto=True)
    assert settings.corrects_rolling_shutter and not settings.requires_sequential_processing, (
        "the wobble alone needs no previous frame: each frame knows its own distortion"
    )
    height = textured_frame.shape[0]
    before: list[float] = []
    after: list[float] = []
    for index in range(window, 40 - window):  # the interior, where the zero-mean window is two-sided
        wobble = track.wobble_at(index)
        assert wobble is not None
        assert abs(wobble[1] * height - gradients[index]) < 3.0, (index, wobble[1] * height, gradients[index])
        corrected = apply_stabilization(frames[index], settings, frame_index=index, state=None, wobble_track=track)
        before.append(abs(_residual_gradient(frames[index], textured_frame)))
        after.append(abs(_residual_gradient(corrected, textured_frame)))
    assert np.mean(before) > 3.0, "the synthetic jello is visible to the metric"
    assert np.mean(after) < 0.25 * np.mean(before), (
        f"residual wobble {np.mean(after):.2f} px vs {np.mean(before):.2f} px"
    )
    assert track.wobble_at(40) is None and track.wobble_at(-1) is None
    assert apply_stabilization(frames[5], settings, frame_index=5, state=None) is frames[5], "no track yet: untouched"

    cancelled = analyze_wobble(_read_from(frames), Path("jello.mp4"), 0, 39, window, cancelled=lambda: True)
    assert cancelled.cancelled and cancelled.track is None
    assert analyze_wobble(_read_from([]), Path("empty.mp4"), 0, 9, window).track is None


def _leaky_sum(changes: np.ndarray, decay: float) -> np.ndarray:
    """The first version of the wobble: ``W_k = decay * W_{k-1} + g_k`` (kept here as the yardstick)."""
    out = np.zeros_like(changes)
    for index in range(1, len(changes)):
        out[index] = decay * out[index - 1] + changes[index]
    return out


def test_filter_wobble_keeps_the_fast_jello_and_drops_drift_and_slow_skew() -> None:
    rng = np.random.default_rng(3)
    count = 600
    frames = np.arange(count)
    fast = 0.02 * np.sin(2.0 * np.pi * frames / 6.0)  # jello at 5 Hz (30 fps), 2% of the height
    slow = 0.05 * np.sin(2.0 * np.pi * frames / 240.0)  # a pan speeding up and slowing down over 8 s
    truth = fast + slow
    noise = rng.normal(0.0, 0.002, count)  # what the tracker gets wrong per pair
    changes = np.column_stack([np.zeros(count), np.diff(truth, prepend=0.0) + noise])
    displacements = np.zeros((count, 2))

    wobble = filter_wobble(changes, displacements, 0.0, wobble_window_frames(30.0))[:, 1]
    interior = slice(40, count - 40)
    rms = lambda values: float(np.sqrt(np.mean(np.square(values))))  # noqa: E731
    residual = rms(wobble[interior] - fast[interior])
    assert residual < 0.4 * rms(fast[interior]), f"residual {residual:.4f} vs jello {rms(fast[interior]):.4f}"
    assert abs(np.corrcoef(wobble[interior], slow[interior])[0, 1]) < 0.2, "the slow skew is not part of the wobble"

    raw = np.cumsum(changes[:, 1])
    assert np.abs(raw - truth).max() > 0.02, "the plain sum wanders by more than the jello itself"
    leaky = _leaky_sum(changes[:, 1], 0.97)
    assert rms(leaky[interior] - fast[interior]) > 2.0 * residual, "the leaky sum kept the drift (the slow jello)"

    with_skew = filter_wobble(changes, np.column_stack([np.zeros(count), slow / 0.7]), 0.7, wobble_window_frames(30.0))
    assert rms(with_skew[interior, 1] - fast[interior]) < 0.4 * rms(fast[interior]), "the slider part is taken out"
    assert filter_wobble(np.zeros((0, 2)), np.zeros((0, 2)), 0.0, 5).shape == (0, 2)


# --- signed readout fraction --------------------------------------------------------------------


def _synthetic_track(readout: float, axis: ReadoutAxis = ReadoutAxis.ROWS) -> WobbleTrack:
    rng = np.random.default_rng(5)
    count = 120
    velocities = np.cumsum(rng.normal(0.0, 1.5, (count, 2)), axis=0)  # a wandering camera speed
    changes = np.zeros((count, 2))
    changes[1:] = readout * (velocities[1:] - velocities[:-1])
    zeros = np.zeros((count, 2))
    return WobbleTrack(
        source=Path("clip.mp4"),
        first_frame=0,
        row_changes=changes if axis is ReadoutAxis.ROWS else zeros,
        column_changes=changes if axis is ReadoutAxis.COLUMNS else zeros,
        displacements=velocities,
        measured=np.ones(count, dtype=bool),
        analysis_size=(480, 270),
        rotation=0,
        window=15,
        detected_axis=axis,
    )


def test_readout_fraction_estimate_recovers_the_signed_fraction() -> None:
    assert _synthetic_track(-0.3).readout_fraction_estimate(ReadoutAxis.ROWS) == pytest.approx(-0.3, abs=0.02)
    assert _synthetic_track(0.6).readout_fraction_estimate(ReadoutAxis.ROWS) == pytest.approx(0.6, abs=0.02)
    assert _synthetic_track(0.5).readout_fraction_estimate(ReadoutAxis.COLUMNS) == pytest.approx(0.0), (
        "the axis without gradients measures nothing"
    )


def test_filter_wobble_takes_out_a_negative_readout_skew() -> None:
    count = 600
    frames = np.arange(count)
    slow = 0.05 * np.sin(2.0 * np.pi * frames / 240.0)
    changes = np.column_stack([np.zeros(count), np.diff(slow, prepend=0.0)])
    displacements = np.column_stack([np.zeros(count), slow / -0.7])  # the sensor reads against the axis
    wobble = filter_wobble(changes, displacements, -0.7, wobble_window_frames(30.0))
    assert float(np.abs(wobble[40:-40, 1]).max()) < 0.002, "a negative fraction takes out the inverted skew"


def test_wobble_is_suppressed_under_perspective_but_the_slider_still_works() -> None:
    both = StabilizationSettings(mode=StabilizationMode.PERSPECTIVE, rolling_shutter=-0.3, rolling_shutter_auto=True)
    assert not both.applies_wobble and both.corrects_rolling_shutter_skew and both.corrects_rolling_shutter
    assert both.needs_wobble_analysis, "the AUTO axis of the slider still needs the clip's analysis"
    slider_off = replace(both, rolling_shutter=0.0)
    assert not slider_off.needs_wobble_analysis and not slider_off.corrects_rolling_shutter
    elsewhere = replace(both, mode=StabilizationMode.SIMILARITY)
    assert elsewhere.applies_wobble and elsewhere.needs_wobble_analysis
