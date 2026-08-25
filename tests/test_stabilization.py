from __future__ import annotations

import cv2
import numpy as np
import pytest

from space_video_enhancer.pipeline.stabilization import transforms
from space_video_enhancer.pipeline.stabilization.estimators import (
    estimate_camera_lock_transform,
    estimate_inter_frame_transform,
    estimate_translation_shift,
    reference_cache_stats,
)
from space_video_enhancer.pipeline.stabilization.stabilize import (
    apply_stabilization,
    estimate_correction,
    lock_weight,
    path_decay,
    smoothing_time_constant,
)
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import NormalizedRect, StabilizationMode, StabilizationSettings

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
    width, height = 320, 180
    canvas = cv2.resize(
        rng.integers(0, 255, (30, 50, 3), dtype=np.uint8), (width + 120, height + 120), interpolation=cv2.INTER_CUBIC
    )
    for _ in range(30):
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


def test_sequence_break_restarts_the_path(textured_frame: np.ndarray, previous_frame: np.ndarray) -> None:
    settings = StabilizationSettings(mode=StabilizationMode.AUTO, temporal_smoothing=0.5, auto_crop=False)
    state = StabilizationState()
    apply_stabilization(previous_frame, settings, frame_index=9, state=state)
    apply_stabilization(textured_frame, settings, frame_index=10, state=state)
    assert state.correction is not None
    assert apply_stabilization(previous_frame, settings, frame_index=30, state=state) is previous_frame
    assert state.correction is None and state.correction_frame_index == 30


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


def test_camera_lock_without_reference_is_noop(textured_frame: np.ndarray) -> None:
    settings = StabilizationSettings(mode=StabilizationMode.CAMERA_LOCK)
    assert apply_stabilization(textured_frame, settings, reference_frame=None) is textured_frame


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


def test_auto_crop_returns_same_size(textured_frame: np.ndarray) -> None:
    transform = transforms.translation_to_affine(12.0, 8.0)
    warped = transforms.warp(textured_frame, transform)
    cropped = transforms.auto_crop(warped, transform)
    assert cropped.shape == textured_frame.shape


def test_state_reset_clears_everything() -> None:
    state = StabilizationState(prev_frame_index=3, lock_lost_frames=2)
    state.reset()
    assert state == StabilizationState()
