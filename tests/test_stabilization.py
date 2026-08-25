from __future__ import annotations

import numpy as np
import pytest

from space_video_enhancer.pipeline.stabilization import transforms
from space_video_enhancer.pipeline.stabilization.estimators import (
    estimate_camera_lock_transform,
    estimate_translation_shift,
)
from space_video_enhancer.pipeline.stabilization.stabilize import apply_stabilization, smooth_transform
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import NormalizedRect, StabilizationMode, StabilizationSettings

SHIFT_DOWN, SHIFT_LEFT = 5, 7
FOCUS = NormalizedRect(0.10, 0.12, 0.28, 0.42)  # around the green rectangle of the textured frame


@pytest.fixture
def previous_frame(textured_frame: np.ndarray) -> np.ndarray:
    return np.roll(textured_frame, (SHIFT_DOWN, -SHIFT_LEFT), axis=(0, 1))


def test_translation_estimator_recovers_global_shift(textured_frame: np.ndarray, previous_frame: np.ndarray) -> None:
    state = StabilizationState()
    assert estimate_translation_shift(previous_frame, 9, state) is None  # first frame only primes the state
    shift = estimate_translation_shift(textured_frame, 10, state)
    assert shift is not None
    assert shift[0] == pytest.approx(-SHIFT_LEFT, abs=0.5)
    assert shift[1] == pytest.approx(SHIFT_DOWN, abs=0.5)
    # Same frame index again -> cached
    assert estimate_translation_shift(textured_frame, 10, state) == shift


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


def test_apply_stabilization_keeps_frame_size_and_moves_content(
    textured_frame: np.ndarray, previous_frame: np.ndarray
) -> None:
    settings = StabilizationSettings(
        mode=StabilizationMode.AUTO, intensity=1.0, temporal_smoothing=0.0, auto_crop=False
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


def test_smoothing_resets_on_non_consecutive_frames() -> None:
    state = StabilizationState()
    first = transforms.translation_to_affine(10.0, 0.0)
    assert np.array_equal(smooth_transform(first, 0.5, 10, state), first)
    second = transforms.translation_to_affine(0.0, 0.0)
    smoothed = smooth_transform(second, 0.5, 11, state)
    assert 0.0 < smoothed[0, 2] < 10.0
    jumped = smooth_transform(first, 0.5, 30, state)
    assert np.array_equal(jumped, first)


def test_state_reset_clears_everything() -> None:
    state = StabilizationState(prev_frame_index=3, hybrid_fallback_bias=0.7)
    state.reset()
    assert state == StabilizationState()
