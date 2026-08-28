"""Frame compositing for fps retiming: blend, optical-flow interpolation, and the compose dispatcher."""

from __future__ import annotations

import numpy as np

from space_video_enhancer.pipeline import retime
from space_video_enhancer.sequence import RetimeSample
from space_video_enhancer.settings import RetimingMode


def test_blend_is_a_weighted_average_and_returns_an_endpoint_at_the_extremes() -> None:
    dark = np.zeros((4, 4, 3), np.uint8)
    bright = np.full((4, 4, 3), 100, np.uint8)
    assert retime.blend(dark, bright, 0.0) is dark and retime.blend(dark, bright, 1.0) is bright
    mid = retime.blend(dark, bright, 0.25)
    assert mid.dtype == np.uint8 and int(mid[0, 0, 0]) == 25


def test_optical_flow_places_a_translating_bar_between_the_two_frames() -> None:
    height, width = 32, 64
    before = np.zeros((height, width, 3), np.uint8)
    before[:, 10:14] = 255
    after = np.zeros((height, width, 3), np.uint8)
    after[:, 30:34] = 255  # the same bar, moved 20 px to the right

    middle = retime.optical_flow_interpolate(before, after, 0.5)

    assert middle.shape == (height, width, 3) and middle.dtype == np.uint8
    column = middle[:, :, 0].mean(axis=0)
    centre = float((np.arange(width) * column).sum() / column.sum())
    assert 16 <= centre <= 28, f"the bar should sit near the middle (~22), got {centre:.1f}"


def test_compose_dispatches_on_mode_and_falls_back_when_a_frame_is_missing() -> None:
    frames = {0: np.zeros((4, 4, 3), np.uint8), 1: np.full((4, 4, 3), 200, np.uint8)}

    def read(index: int) -> np.ndarray | None:
        return frames.get(index)

    assert retime.compose(RetimeSample(0, 0, 0.0, RetimingMode.NEAREST), read) is frames[0]
    blended = retime.compose(RetimeSample(0, 1, 0.5, RetimingMode.BLEND), read)
    assert blended is not None and int(blended[0, 0, 0]) == 100
    downgraded = retime.compose(RetimeSample(0, 1, 0.5, RetimingMode.OPTICAL_FLOW), read, allow_optical_flow=False)
    assert downgraded is not None and int(downgraded[0, 0, 0]) == 100  # preview downgrade: a blend, not flow
    assert retime.compose(RetimeSample(0, 9, 0.5, RetimingMode.BLEND), read) is frames[0]  # second frame missing
    assert retime.compose(RetimeSample(9, 9, 0.0, RetimingMode.NEAREST), read) is None  # nothing decodes
