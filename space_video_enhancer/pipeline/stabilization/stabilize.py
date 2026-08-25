"""Stabilization orchestration: estimate motion, follow a smoothed camera path, warp.

Inter-frame modes (translation, similarity, perspective) never just align a frame onto the
previous *raw* frame — that only delays the shake by one frame. They accumulate the motion
into a correction that follows a low-passed version of the camera path::

    correction_k = decay * (correction_{k-1} o delta_k)

``delta_k`` aligns frame k onto frame k-1; ``decay`` comes from the smoothing slider and is
how slowly the virtual camera follows the real one (a time constant in frames), so fast
jitter is removed while pans survive. The translation part is capped so the corrected
frame never wanders further than :data:`CORRECTION_BUDGET_FRACTION` of the frame — which
is also the fixed margin the automatic crop removes, on every frame, so the zoom is constant.

Camera lock estimates an absolute transform onto the reference frame every frame and holds
the last one while the target is lost. Hybrid does the same but, while lost, keeps the lock
alive by dead reckoning with the inter-frame motion, releasing it slowly; when the target is
found again it eases back onto the lock instead of snapping.
"""

from __future__ import annotations

import numpy as np

from space_video_enhancer.pipeline.stabilization import estimators, transforms
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import StabilizationMode, StabilizationSettings, clamp_unit
from space_video_enhancer.telemetry.trace import note

SMOOTHING_MIN_FRAMES = 2.0
SMOOTHING_MAX_FRAMES = 60.0
"""Time constant of the virtual camera at smoothing 0 and 1 (log scale in between)."""
LOCK_SMOOTHING_MAX = 0.7
"""At smoothing 1 a fresh camera-lock estimate replaces only 30% of the previous correction."""
CORRECTION_BUDGET_FRACTION = 0.05
"""Cap on the accumulated translation per axis, and the fixed margin of the automatic crop."""
HYBRID_RELEASE_FRAMES = 30.0
HYBRID_REACQUIRE_WEIGHT = 0.5
HYBRID_REACQUIRE_FRAMES = 6


def smoothing_time_constant(smoothing: float) -> float:
    """Frames the virtual camera takes to catch up with the real one."""
    ratio = SMOOTHING_MAX_FRAMES / SMOOTHING_MIN_FRAMES
    return SMOOTHING_MIN_FRAMES * ratio ** clamp_unit(smoothing)


def path_decay(smoothing: float) -> float:
    """Per-frame factor that pulls the accumulated correction back toward the identity."""
    return 1.0 - 1.0 / smoothing_time_constant(smoothing)


def lock_weight(smoothing: float) -> float:
    """Share of a fresh camera-lock estimate that replaces the previous correction."""
    return 1.0 - LOCK_SMOOTHING_MAX * clamp_unit(smoothing)


def apply_stabilization(
    frame: np.ndarray,
    settings: StabilizationSettings,
    *,
    frame_index: int | None = None,
    state: StabilizationState | None = None,
    reference_frame: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``frame`` stabilized according to ``settings`` (or unchanged when nothing applies).

    ``state`` carries the inter-frame memory (previous frame, accumulated correction);
    ``reference_frame`` is the locked reference for camera lock / hybrid.
    """
    if not settings.enabled or settings.intensity <= 0.0:
        return frame
    note("stabilization.mode", settings.mode.value)

    correction = estimate_correction(frame, settings, frame_index, state, reference_frame)
    applied: np.ndarray | None = None
    if correction is None:
        note("stabilization.result", "no_transform")
    else:
        blended = transforms.blend_with_identity(correction, settings.intensity)
        if settings.zoom_out:
            blended = transforms.fit_with_zoom_out(blended, frame.shape)
        note("stabilization.translation_px", [round(float(blended[0, 2]), 2), round(float(blended[1, 2]), 2)])
        if transforms.is_negligible(blended):
            note("stabilization.result", "negligible")
        else:
            note("stabilization.result", "applied")
            applied = blended

    stabilized = frame if applied is None else transforms.warp(frame, applied)
    if settings.auto_crop:
        # Always, even without a correction: the zoom must not pop in and out between frames.
        stabilized = transforms.stabilization_crop(stabilized, applied, CORRECTION_BUDGET_FRACTION)
    return stabilized


def estimate_correction(
    frame: np.ndarray,
    settings: StabilizationSettings,
    frame_index: int | None,
    state: StabilizationState | None,
    reference_frame: np.ndarray | None,
) -> np.ndarray | None:
    """Full-strength correction for ``frame`` (before the intensity blend), or ``None``."""
    width = int(frame.shape[1])
    if state is not None:
        state.adopt_frame_width(width)
        cached = state.cached_correction(frame_index, width)
        if cached is not None:
            note("stabilization.estimator", "cached")
            state.remember_correction(cached, frame_index, width)  # keep the stored units current
            return cached

    mode = settings.mode
    if mode.uses_auto_motion:
        correction = follow_smoothed_path(frame, settings, frame_index, state)
    elif mode is StabilizationMode.CAMERA_LOCK:
        correction = camera_lock(frame, settings, frame_index, state, reference_frame)
    elif mode is StabilizationMode.HYBRID:
        correction = hybrid(frame, settings, frame_index, state, reference_frame)
    else:
        correction = None

    if state is not None:
        state.remember_correction(correction, frame_index, width)
    return correction


def follow_smoothed_path(
    frame: np.ndarray,
    settings: StabilizationSettings,
    frame_index: int | None,
    state: StabilizationState | None,
) -> np.ndarray | None:
    if state is None:
        return None
    delta = estimators.estimate_inter_frame_transform(frame, frame_index, state, settings.mode)
    previous = state.previous_correction(frame_index, frame.shape[1])
    if delta is None:
        if previous is None:
            note("stabilization.path", "start")
            return None
        # Motion unknown for this frame: assume the camera held still and keep decaying.
        note("stabilization.path", "held")
        delta = transforms.identity_like(previous)
    else:
        note("stabilization.path", "follow")

    base = previous if previous is not None else transforms.identity_like(delta)
    accumulated = transforms.compose(base, delta)
    correction = transforms.mix_transforms(
        transforms.identity_like(accumulated), accumulated, path_decay(settings.temporal_smoothing)
    )
    correction = transforms.clamp_translation(correction, frame.shape, CORRECTION_BUDGET_FRACTION)
    if not transforms.is_reasonable(correction, frame.shape, allow_perspective=not transforms.is_affine(correction)):
        note("stabilization.rejected", "accumulated")
        return None
    return correction


def camera_lock(
    frame: np.ndarray,
    settings: StabilizationSettings,
    frame_index: int | None,
    state: StabilizationState | None,
    reference_frame: np.ndarray | None,
) -> np.ndarray | None:
    lock = None
    if reference_frame is not None:
        lock = estimators.estimate_camera_lock_transform(frame, reference_frame, settings.focus_rect)
    previous = state.previous_correction(frame_index, frame.shape[1]) if state is not None else None
    if lock is None:
        if previous is None:
            return None
        note("stabilization.path", "held")
        return previous
    note("stabilization.path", "lock")
    if previous is None:
        return lock
    return transforms.mix_transforms(previous, lock, lock_weight(settings.temporal_smoothing))


def hybrid(
    frame: np.ndarray,
    settings: StabilizationSettings,
    frame_index: int | None,
    state: StabilizationState | None,
    reference_frame: np.ndarray | None,
) -> np.ndarray | None:
    """Camera lock while the target is found; dead reckoning on global motion while it is lost."""
    if state is None:
        return None

    lock = None
    if reference_frame is not None:
        lock = estimators.estimate_camera_lock_transform(frame, reference_frame, settings.focus_rect)
    delta = estimators.estimate_inter_frame_transform(
        frame, frame_index, state, StabilizationMode.SIMILARITY, translation_fallback=True
    )
    previous = state.previous_correction(frame_index, frame.shape[1])

    if lock is not None:
        if previous is None:
            state.lock_lost_frames = 0
            note("stabilization.path", "lock")
            return lock
        propagated = previous if delta is None else transforms.compose(previous, delta)
        if state.lock_lost_frames > 0:
            state.lock_lost_frames -= 1
            note("stabilization.path", "reacquire")
            weight = min(HYBRID_REACQUIRE_WEIGHT, lock_weight(settings.temporal_smoothing))
        else:
            note("stabilization.path", "lock")
            weight = lock_weight(settings.temporal_smoothing)
        return transforms.mix_transforms(propagated, lock, weight)

    state.lock_lost_frames = min(HYBRID_REACQUIRE_FRAMES, state.lock_lost_frames + 1)
    if previous is None:
        if delta is None:
            note("stabilization.path", "start")
            return None
        base = transforms.identity_like(delta)
    else:
        base = previous
        if delta is None:
            delta = transforms.identity_like(previous)
    accumulated = transforms.compose(base, delta)
    release = 1.0 - 1.0 / HYBRID_RELEASE_FRAMES
    correction = transforms.mix_transforms(transforms.identity_like(accumulated), accumulated, release)
    if not transforms.is_reasonable(correction, frame.shape, allow_perspective=not transforms.is_affine(correction)):
        note("stabilization.rejected", "accumulated")
        return None
    note("stabilization.path", "dead_reckoning")
    return correction
