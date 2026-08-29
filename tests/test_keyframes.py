"""Manual keyframe stabilization: pure interpolation and the affine it drives."""

from __future__ import annotations

import numpy as np
import pytest

from space_video_enhancer.pipeline.stabilization.keyframes import interpolate_keyframes, manual_correction
from space_video_enhancer.pipeline.stabilization.stabilize import apply_stabilization
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import StabilizationKeyframe, StabilizationMode, StabilizationSettings

KF = StabilizationKeyframe


def test_no_keyframes_gives_no_correction() -> None:
    assert interpolate_keyframes((), 5) is None
    assert manual_correction((), 5, (100, 100, 3)) is None


def test_value_is_held_flat_outside_the_first_and_last_keyframe() -> None:
    keyframes = (KF(10, tx=0.1), KF(20, tx=0.3))
    assert interpolate_keyframes(keyframes, 0) == (0.1, 0.0, 0.0, 1.0, 0.0, 1.0), "before the first: hold"
    assert interpolate_keyframes(keyframes, 99) == (0.3, 0.0, 0.0, 1.0, 0.0, 1.0), "after the last: hold"


def test_every_field_interpolates_linearly_between_two_keyframes() -> None:
    keyframes = (
        KF(10, tx=0.0, ty=0.2, rotation=0.0, scale=1.0, shear=0.0, stretch_y=1.0),
        KF(20, tx=0.4, ty=0.0, rotation=10.0, scale=1.5, shear=0.2, stretch_y=1.4),
    )
    tx, ty, rotation, scale, shear, stretch_y = interpolate_keyframes(keyframes, 15)  # halfway
    assert tx == pytest.approx(0.2) and ty == pytest.approx(0.1)
    assert rotation == pytest.approx(5.0) and scale == pytest.approx(1.25)
    assert shear == pytest.approx(0.1) and stretch_y == pytest.approx(1.2)


def test_keyframes_are_kept_sorted_and_deduped_by_frame() -> None:
    settings = StabilizationSettings(keyframes=(KF(20, tx=0.3), KF(10, tx=0.1), KF(20, tx=0.9)))
    assert [k.frame for k in settings.keyframes] == [10, 20]
    assert settings.keyframes[1].tx == 0.9, "the later keyframe wins its frame"


def test_manual_mode_is_stateless_so_the_export_stays_parallel() -> None:
    manual = StabilizationSettings(mode=StabilizationMode.MANUAL, keyframes=(KF(0, tx=0.1),), temporal_smoothing=0.5)
    assert not manual.requires_sequential_processing, "each frame interpolates from the keyframes, not frame N-1"


def test_a_translation_keyframe_shifts_the_frame_content() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[38:42, 48:52] = 255  # a bright marker in the middle (its BORDER_REFLECT mirror lands off in the black)
    settings = StabilizationSettings(
        mode=StabilizationMode.MANUAL,
        keyframes=(KF(0, tx=0.2),),  # move content a fifth of the width to the right
        auto_crop=False,  # inspect the raw warp, no crop-and-rescale
    )
    result = apply_stabilization(frame, settings, frame_index=0)
    moved = np.argwhere(result.max(axis=2) > 128)
    assert moved.size, "the marker survived the warp"
    center_x = moved[:, 1].mean()
    assert center_x == pytest.approx(50 + 0.2 * 120, abs=2.0), "the marker moved right by tx*width"


def test_a_vertical_stretch_keyframe_elongates_the_frame_about_its_centre() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[20:24, 58:62] = 255  # a marker above the centre row (40)
    settings = StabilizationSettings(mode=StabilizationMode.MANUAL, keyframes=(KF(0, stretch_y=1.5),), auto_crop=False)
    result = apply_stabilization(frame, settings, frame_index=0)
    center_y = np.argwhere(result.max(axis=2) > 128)[:, 0].mean()
    # 40 + 1.5 * (22 - 40): the marker moves away from the centre as the frame stretches vertically
    assert center_y == pytest.approx(40 + 1.5 * (22 - 40), abs=2.0)


def test_a_shear_keyframe_leans_the_frame_horizontally_with_height() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[20:24, 58:62] = 255  # a marker above the centre row (40)
    settings = StabilizationSettings(mode=StabilizationMode.MANUAL, keyframes=(KF(0, shear=0.2),), auto_crop=False)
    result = apply_stabilization(frame, settings, frame_index=0)
    center_x = np.argwhere(result.max(axis=2) > 128)[:, 1].mean()
    # 60 + 0.2 * (22 - 40): above the centre the content leans left by shear * distance-from-centre
    assert center_x == pytest.approx(60 + 0.2 * (22 - 40), abs=2.0)


def test_editing_a_manual_keyframe_is_not_masked_by_the_correction_cache() -> None:
    """A frame's correction is cached per (frame, width); editing its keyframe must still take effect.

    Regression: the first render of frame 6 cached an identity correction; moving the slider changed
    the keyframe but the cache replayed the stale identity, so the preview never moved (result went
    ``negligible``). Manual mode must recompute from the current keyframes every time.
    """
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[38:42, 58:62] = 255  # a bright marker just right of centre
    state = StabilizationState()  # one state reused across renders, exactly like the preview worker

    at_rest = StabilizationSettings(mode=StabilizationMode.MANUAL, keyframes=(KF(6, tx=0.0),), auto_crop=False)
    apply_stabilization(frame, at_rest, frame_index=6, state=state)  # would cache an identity for frame 6

    moved = StabilizationSettings(mode=StabilizationMode.MANUAL, keyframes=(KF(6, tx=0.2),), auto_crop=False)
    result = apply_stabilization(frame, moved, frame_index=6, state=state)  # same frame, edited keyframe
    center_x = np.argwhere(result.max(axis=2) > 128)[:, 1].mean()
    assert center_x == pytest.approx(60 + 0.2 * 120, abs=2.0), "the edited keyframe shifts the frame despite the cache"
