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

Sequences tolerate skipped frames (the preview drops frames when rendering is slower than
playback): a gap of up to ``MAX_FRAME_GAP`` frames continues the path, decaying it once per
skipped frame; only larger jumps restart it.

Camera lock estimates an absolute transform onto the reference frame every frame and holds
the last one while the target is lost. Hybrid does the same but, while lost, keeps the lock
alive by dead reckoning with the inter-frame motion, releasing it slowly; when the target is
found again it eases back onto the lock instead of snapping.
"""

from __future__ import annotations

import numpy as np

from space_video_enhancer.pipeline.stabilization import estimators, rolling_shutter, transforms
from space_video_enhancer.pipeline.stabilization.keyframes import manual_correction
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.pipeline.stabilization.wobble import WobbleTrack
from space_video_enhancer.settings import ReadoutAxis, StabilizationMode, StabilizationSettings, clamp_unit
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
    wobble_track: WobbleTrack | None = None,
) -> np.ndarray:
    """Return ``frame`` stabilized according to ``settings`` (or unchanged when nothing applies).

    ``state`` carries the inter-frame memory (previous frame, accumulated correction);
    ``reference_frame`` is the locked reference for camera lock / hybrid; ``wobble_track`` is the
    clip's rolling-shutter wobble analysis (see :mod:`wobble`).
    """
    stabilizing = settings.enabled and settings.intensity > 0.0
    if not stabilizing and not settings.corrects_rolling_shutter:
        return frame

    correction: np.ndarray | None = None
    if stabilizing:
        note("stabilization.mode", settings.mode.value)
        correction = estimate_correction(frame, settings, frame_index, state, reference_frame)
    if settings.corrects_rolling_shutter:
        # Uses the inter-frame motion estimated just above (or estimates a translation when no mode did).
        frame = unskew_rolling_shutter(frame, settings, frame_index, state, wobble_track)
    if not stabilizing:
        return frame

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


def unskew_rolling_shutter(
    frame: np.ndarray,
    settings: StabilizationSettings,
    frame_index: int | None,
    state: StabilizationState | None,
    wobble_track: WobbleTrack | None = None,
) -> np.ndarray:
    """``frame`` with its rolling-shutter distortion undone (see :mod:`rolling_shutter`): one remap for both parts.

    The skew (slider) needs the scene displacement since the previous frame: the inter-frame transform
    already estimated for this frame when an automatic mode (or hybrid) ran, else a plain translation
    estimated here — without a previous frame in ``state`` there is no motion to speak of. The wobble
    comes from ``wobble_track`` (the clip's analysis, see :mod:`wobble`) and needs no memory at all.
    """
    axis, axis_source = resolve_readout_axis(settings, wobble_track)
    note("rolling_shutter.axis", axis.value)
    note("rolling_shutter.axis_source", axis_source)
    gradient_x = gradient_y = 0.0
    contributions = 0
    if settings.corrects_rolling_shutter_skew:
        displacement = _scene_displacement(frame, frame_index, state)
        if displacement is not None:
            note("rolling_shutter.shift_px", [round(displacement[0], 2), round(displacement[1], 2)])
            gradient_x += displacement[0] * settings.rolling_shutter
            gradient_y += displacement[1] * settings.rolling_shutter
            contributions += 1
    if settings.rolling_shutter_auto:
        wobble = None
        if wobble_track is not None and frame_index is not None:
            wobble = wobble_track.wobble_at(frame_index, settings.rolling_shutter, axis)
        if wobble is None:
            note("rolling_shutter.wobble", "unanalysed")
        else:
            length = float(frame.shape[1] if axis is ReadoutAxis.COLUMNS else frame.shape[0])
            note("rolling_shutter.wobble_px", [round(wobble[0] * length, 2), round(wobble[1] * length, 2)])
            gradient_x += wobble[0] * length
            gradient_y += wobble[1] * length
            contributions += 1
    if contributions == 0:
        note("rolling_shutter.result", "no_motion" if settings.corrects_rolling_shutter_skew else "unanalysed")
        return frame
    corrected = rolling_shutter.correct(frame, (gradient_x, gradient_y), axis)
    note("rolling_shutter.result", "applied" if corrected is not frame else "negligible")
    return corrected


def resolve_readout_axis(settings: StabilizationSettings, wobble_track: WobbleTrack | None) -> tuple[ReadoutAxis, str]:
    """The axis the corrections work along and where it came from: ``forced`` (the setting), ``detected``
    (the clip's wobble analysis) or ``default`` (rows, until an analysis exists)."""
    if settings.readout_axis is not ReadoutAxis.AUTO:
        return settings.readout_axis, "forced"
    if wobble_track is not None:
        return wobble_track.detected_axis, "detected"
    return ReadoutAxis.ROWS, "default"


def _scene_displacement(
    frame: np.ndarray, frame_index: int | None, state: StabilizationState | None
) -> tuple[float, float] | None:
    """How far the scene moved at the centre since the previous frame (px), or ``None`` when unknown."""
    if state is None:
        return None
    if frame_index is not None and state.last_transform_index == frame_index:
        delta = state.last_transform  # None when that estimate failed: nothing to gain by trying again
    else:
        delta = estimators.estimate_inter_frame_transform(frame, frame_index, state, StabilizationMode.AUTO)
    if delta is None:
        return None
    return rolling_shutter.center_displacement(delta, frame.shape)


def needs_previous_frame(state: StabilizationState | None, frame_index: int | None) -> bool:
    """True when the rolling-shutter unskew of ``frame_index`` has no previous frame to measure motion against.

    That is the case on a fresh state (the memory restarts whenever a stabilization setting changes),
    after a jump on the timeline, or when this very frame was already measured and found no motion.
    """
    if state is None or frame_index is None or frame_index <= 0:
        return False
    if state.last_transform_index == frame_index:
        return state.last_transform is None
    return state.prev_gray is None or state.breaks_sequence(frame_index)


def prime_with_previous(previous_frame: np.ndarray, previous_index: int, state: StabilizationState) -> None:
    """Remember ``previous_frame`` as the frame before the one about to be rendered, so its motion can be measured.

    Callers decode it on purpose (the preview when paused on a frame, the export at the start of a
    segment); any "no motion" cached for the current frame is forgotten so the estimate runs again.
    """
    gray, _scale = estimators.analysis_gray(previous_frame)
    state.remember_gray(gray, previous_index)
    state.remember_transform(None, None)
    state.cell_memory = None


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
    elif mode is StabilizationMode.MANUAL:
        correction = manual_correction(settings.keyframes, frame_index, frame.shape)
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
    gap = abs(state.correction_gap(frame_index) or 1)  # skipped frames decay the path as if rendered
    correction = transforms.mix_transforms(
        transforms.identity_like(accumulated), accumulated, path_decay(settings.temporal_smoothing) ** gap
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
        lock = estimators.estimate_camera_lock_transform(
            frame, reference_frame, settings.focus_rect, state, frame_index
        )
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
        lock = estimators.estimate_camera_lock_transform(
            frame, reference_frame, settings.focus_rect, state, frame_index
        )
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
    release = (1.0 - 1.0 / HYBRID_RELEASE_FRAMES) ** abs(state.correction_gap(frame_index) or 1)
    correction = transforms.mix_transforms(transforms.identity_like(accumulated), accumulated, release)
    if not transforms.is_reasonable(correction, frame.shape, allow_perspective=not transforms.is_affine(correction)):
        note("stabilization.rejected", "accumulated")
        return None
    note("stabilization.path", "dead_reckoning")
    return correction
