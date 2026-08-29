"""Phase 2 — learn the camera path and fill keyframes with it (:mod:`pipeline.stabilization.motion_path`)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from space_video_enhancer.pipeline.stabilization.motion_path import (
    MotionPath,
    analyze_motion_path,
    fill_keyframes,
)
from space_video_enhancer.settings import StabilizationKeyframe
from tests.conftest import make_textured_frame

KF = StabilizationKeyframe


def _flat_path(dx: np.ndarray) -> MotionPath:
    zero = np.zeros(len(dx))
    return MotionPath(0, dx.astype(float), zero.copy(), zero.copy(), np.ones(len(dx), dtype=bool))


def test_fill_without_a_path_or_below_two_keyframes_leaves_the_keyframes_alone() -> None:
    keyframes = (KF(0), KF(10, tx=0.2))
    assert fill_keyframes(keyframes, None) == keyframes
    assert fill_keyframes((KF(3, tx=0.1),), _flat_path(np.linspace(0, 0.4, 11))) == (KF(3, tx=0.1),)


def test_fill_leaves_a_steady_pan_as_plain_linear_interpolation() -> None:
    path = _flat_path(np.linspace(0.0, 0.4, 11))  # constant velocity: no deviation to cancel
    filled = fill_keyframes((KF(0, tx=0.0), KF(10, tx=0.2)), path)
    assert [k.frame for k in filled] == list(range(11))
    for keyframe in filled:
        assert keyframe.tx == pytest.approx(0.02 * keyframe.frame), "reduces to lerp when the motion is linear"


def test_fill_cancels_a_wiggle_so_the_stabilized_residual_runs_straight() -> None:
    dx = np.array([0.0, 0.05, 0.30, 0.05, 0.0])  # a bump in the middle, back to zero at the ends
    path = _flat_path(dx)
    filled = fill_keyframes((KF(0), KF(4)), path)
    residual = [path.sample(k.frame)[0] - k.tx for k in filled]  # motion - correction
    assert max(residual) - min(residual) < 1e-9, "the correction absorbs the bump; the residual is flat"


def test_fill_preserves_hand_set_keyframes_and_never_overwrites_them() -> None:
    path = _flat_path(np.array([0.0, 0.2, 0.0, 0.2, 0.0]))
    filled = {k.frame: k for k in fill_keyframes((KF(0), KF(2, tx=0.9), KF(4)), path)}
    assert filled[0].is_identity and filled[4].is_identity, "the end keyframes stay identity"
    assert filled[2].tx == 0.9, "the middle keyframe the user set is not replaced by the fill"


def test_fill_skips_a_gap_the_path_does_not_cover() -> None:
    path = MotionPath(0, np.linspace(0, 0.4, 5), np.zeros(5), np.zeros(5), np.ones(5, dtype=bool))  # frames 0..4
    filled = fill_keyframes((KF(0), KF(20, tx=0.2)), path)  # gap runs past the analysed range
    assert [k.frame for k in filled] == [0, 20], "an uncovered gap is left as it was"


def _wiggly_reader(width: int, height: int, offsets: list[int]):
    base = cv2.resize(make_textured_frame(seed=11), (width, height), interpolation=cv2.INTER_AREA)

    def read(index: int) -> np.ndarray | None:
        if index < 0 or index >= len(offsets):
            return None
        return np.roll(base, offsets[index], axis=1)  # a horizontal pan by a known amount

    return read


def test_analysis_recovers_a_monotonic_pan() -> None:
    offsets = [i * 4 for i in range(12)]  # a steady rightward pan, 4 px per frame
    path = analyze_motion_path(_wiggly_reader(320, 240, offsets), 0, 11).track
    assert path is not None and path.measured_fraction > 0.8, "most frames tracked"
    steps = np.diff(path.dx)
    assert np.all(steps < 0) or np.all(steps > 0), "a steady pan gives a monotonic path"
    assert abs(path.dx[-1]) > abs(path.dx[0]), "the path grows away from the reference frame"


def test_filling_from_a_tracked_clip_straightens_the_stabilized_path() -> None:
    linear = np.linspace(0.0, 44.0, 15)
    wiggle = 6.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, 15))  # shake on top of a pan
    offsets = [round(value) for value in linear + wiggle]
    path = analyze_motion_path(_wiggly_reader(320, 240, offsets), 0, 14).track
    assert path is not None

    keyframes = (KF(0), KF(14))  # the user only pins the two ends
    without_fill = [path.sample(frame)[0] for frame in range(15)]  # residual with zero correction = the raw path
    filled = {k.frame: k for k in fill_keyframes(keyframes, path)}
    with_fill = [path.sample(frame)[0] - filled[frame].tx for frame in range(15)]

    def roughness(values: list[float]) -> float:
        return float(np.max(np.abs(np.diff(values, n=2))))  # second difference: 0 for a straight line

    assert roughness(with_fill) < 0.3 * roughness(without_fill), "the fill removes most of the wiggle"
