"""Stabilization orchestration: pick an estimator, smooth, blend and warp."""

from __future__ import annotations

import numpy as np

from space_video_enhancer.pipeline.stabilization import estimators, transforms
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import StabilizationMode, StabilizationSettings
from space_video_enhancer.telemetry.trace import note

HYBRID_LOCK_RECOVERY_STEP = 0.22
HYBRID_LOCK_LOSS_STEP = 0.28
HYBRID_LOCK_MIX_MAX = 0.35
HYBRID_TRANSLATION_THRESHOLD = 0.55
HYBRID_TRANSLATION_MAX_WEIGHT = 0.55
SMOOTHING_MIN_CURRENT_WEIGHT = 0.08
SMOOTHING_STRENGTH = 0.92


def apply_stabilization(
    frame: np.ndarray,
    settings: StabilizationSettings,
    *,
    frame_index: int | None = None,
    state: StabilizationState | None = None,
    reference_frame: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``frame`` stabilized according to ``settings`` (or unchanged when nothing applies).

    ``state`` carries inter-frame memory for the auto/hybrid modes and temporal smoothing;
    ``reference_frame`` is the locked reference for camera lock / hybrid.
    """
    if not settings.enabled or settings.intensity <= 0.0:
        return frame
    note("stabilization.mode", settings.mode.value)

    transform = estimate_transform(frame, settings, frame_index, state, reference_frame)
    if transform is None:
        note("stabilization.result", "no_transform")
        return frame

    transform = smooth_transform(transform, settings.temporal_smoothing, frame_index, state)
    blended = transforms.blend_with_identity(transform, settings.intensity)
    if settings.zoom_out:
        blended = transforms.fit_with_zoom_out(blended, frame.shape)
    note("stabilization.translation_px", [round(float(blended[0, 2]), 2), round(float(blended[1, 2]), 2)])
    if transforms.is_negligible(blended):
        note("stabilization.result", "negligible")
        return frame

    stabilized = transforms.warp(frame, blended)
    if settings.auto_crop:
        stabilized = transforms.auto_crop(stabilized, blended)
    note("stabilization.result", "applied")
    return stabilized


def estimate_transform(
    frame: np.ndarray,
    settings: StabilizationSettings,
    frame_index: int | None,
    state: StabilizationState | None,
    reference_frame: np.ndarray | None,
) -> np.ndarray | None:
    mode = settings.mode
    if mode is StabilizationMode.AUTO:
        if state is None:
            return None
        shift = estimators.estimate_translation_shift(frame, frame_index, state)
        return None if shift is None else transforms.translation_to_affine(shift[0], shift[1])

    if mode in (StabilizationMode.SIMILARITY, StabilizationMode.PERSPECTIVE):
        if state is None:
            return None
        return estimators.estimate_geometric_transform(frame, frame_index, state, mode)

    if mode is StabilizationMode.CAMERA_LOCK:
        if reference_frame is None:
            return None
        return estimators.estimate_camera_lock_transform(frame, reference_frame, settings.focus_rect)

    if mode is StabilizationMode.HYBRID:
        return estimate_hybrid_transform(frame, settings, frame_index, state, reference_frame)

    return None


def estimate_hybrid_transform(
    frame: np.ndarray,
    settings: StabilizationSettings,
    frame_index: int | None,
    state: StabilizationState | None,
    reference_frame: np.ndarray | None,
) -> np.ndarray | None:
    """Camera lock while the target is found; progressive fallback to global motion when lost."""
    if state is None:
        return None

    lock_transform = None
    if reference_frame is not None:
        lock_transform = estimators.estimate_camera_lock_transform(frame, reference_frame, settings.focus_rect)

    similarity_transform = estimators.estimate_geometric_transform(
        frame, frame_index, state, StabilizationMode.SIMILARITY
    )
    translation_shift = estimators.estimate_translation_shift(frame, frame_index, state)
    translation_transform = None
    if translation_shift is not None:
        translation_transform = transforms.translation_to_affine(translation_shift[0], translation_shift[1])

    if lock_transform is not None:
        state.hybrid_fallback_bias = max(0.0, state.hybrid_fallback_bias - HYBRID_LOCK_RECOVERY_STEP)
        note("stabilization.hybrid_bias", round(state.hybrid_fallback_bias, 3))
        if similarity_transform is not None and state.hybrid_fallback_bias > 0.01:
            weight = min(HYBRID_LOCK_MIX_MAX, state.hybrid_fallback_bias * 0.5)
            note("stabilization.hybrid_path", "lock_mixed_with_similarity")
            return transforms.mix_transforms(lock_transform, similarity_transform, weight)
        note("stabilization.hybrid_path", "lock")
        return lock_transform

    state.hybrid_fallback_bias = min(1.0, state.hybrid_fallback_bias + HYBRID_LOCK_LOSS_STEP)
    bias = state.hybrid_fallback_bias
    note("stabilization.hybrid_bias", round(bias, 3))
    if similarity_transform is not None and translation_transform is not None and bias > HYBRID_TRANSLATION_THRESHOLD:
        translation_weight = min(
            HYBRID_TRANSLATION_MAX_WEIGHT,
            (bias - HYBRID_TRANSLATION_THRESHOLD) / (1.0 - HYBRID_TRANSLATION_THRESHOLD),
        )
        note("stabilization.hybrid_path", "similarity_mixed_with_translation")
        return transforms.mix_transforms(similarity_transform, translation_transform, translation_weight)
    if similarity_transform is not None:
        note("stabilization.hybrid_path", "similarity")
        return similarity_transform
    note("stabilization.hybrid_path", "translation" if translation_transform is not None else "none")
    return translation_transform


def smooth_transform(
    transform: np.ndarray,
    smoothing: float,
    frame_index: int | None,
    state: StabilizationState | None,
) -> np.ndarray:
    """Exponential smoothing across consecutive frames; resets whenever the sequence breaks."""
    if state is None or smoothing <= 0.0:
        return transform

    previous = state.smoothed_transform
    previous_index = state.smoothed_frame_index
    consecutive = (
        isinstance(previous, np.ndarray)
        and frame_index is not None
        and previous_index is not None
        and previous_index == frame_index - 1
    )
    if not consecutive:
        state.smoothed_transform = transform
        state.smoothed_frame_index = frame_index
        note("stabilization.smoothed", False)
        return transform
    note("stabilization.smoothed", True)

    assert previous is not None
    current_weight = max(SMOOTHING_MIN_CURRENT_WEIGHT, 1.0 - (smoothing * SMOOTHING_STRENGTH))
    smoothed = transforms.mix_transforms(previous, transform, current_weight)
    state.smoothed_transform = smoothed
    state.smoothed_frame_index = frame_index
    return smoothed
