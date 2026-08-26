"""Exposure lock: region medians, gain math and the effect through the render pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from space_video_enhancer.pipeline.exposure import exposure_gains, exposure_lock_table, gains_in_ev, region_medians
from space_video_enhancer.pipeline.render import render_frame
from space_video_enhancer.settings import ExposureLockSettings, FilterSettings, NormalizedRect, RenderSettings
from space_video_enhancer.telemetry import trace as tracing
from space_video_enhancer.telemetry.trace import RenderTrace
from tests.conftest import make_textured_frame

WALL = NormalizedRect(0.1, 0.1, 0.3, 0.3)


def _flat_frame(color: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    frame[:] = color
    frame[60:, :] = (250, 250, 250)  # a bright floor outside the wall region
    return frame


def test_region_medians_read_the_region_or_the_whole_frame() -> None:
    frame = _flat_frame((40, 120, 200))
    assert region_medians(frame, WALL) == (40.0, 120.0, 200.0)
    frame[:, :80] = (0, 0, 0)  # half of the frame black: the whole-frame median lands on the dominant side
    whole = region_medians(frame, None)
    assert whole[0] in (0.0, 40.0, 250.0)
    assert region_medians(frame, NormalizedRect(0.5, 0.5, 0.0, 0.0)) == (40.0, 120.0, 200.0), "degenerate rect: 1 px"
    assert region_medians(frame, None) == whole
    big = np.full((1200, 1600, 3), 77, dtype=np.uint8)
    assert region_medians(big, None) == (77.0, 77.0, 77.0)  # subsampled path


def test_exposure_gains_are_clamped_eased_and_optionally_luma_only() -> None:
    reference = (100.0, 100.0, 100.0)
    assert exposure_gains(reference, (50.0, 100.0, 200.0)) == (2.0, 1.0, 0.5)
    assert exposure_gains(reference, (20.0, 100.0, 400.0), max_ev=1.0) == (2.0, 1.0, 0.5), "±1 EV clamp"
    b, _g, r = exposure_gains(reference, (50.0, 100.0, 200.0), max_ev=0.5)
    assert b == pytest.approx(2**0.5, abs=1e-3) and r == pytest.approx(2**-0.5, abs=1e-3)
    assert exposure_gains(reference, (50.0, 100.0, 200.0), strength=0.5) == (1.5, 1.0, 0.75)
    assert exposure_gains(reference, (50.0, 100.0, 200.0), strength=0.0) == (1.0, 1.0, 1.0)
    luma_only = exposure_gains(reference, (50.0, 50.0, 50.0), color=False)
    assert luma_only == (2.0, 2.0, 2.0)
    mixed = exposure_gains(reference, (50.0, 100.0, 200.0), color=False)
    assert mixed[0] == mixed[1] == mixed[2] and 0.5 < mixed[0] < 1.0
    assert exposure_gains(reference, (0.0, 0.0, 0.0)) == (2.0, 2.0, 2.0), "black region never divides by zero"
    assert gains_in_ev((2.0, 2.0, 2.0)) == pytest.approx(1.0) and gains_in_ev((1.0, 1.0, 1.0)) == 0.0


def test_lock_table_is_none_until_a_reference_exists_and_notes_the_trace() -> None:
    frame = _flat_frame((40, 120, 200))
    assert exposure_lock_table(frame, ExposureLockSettings(enabled=True, region=WALL)) is None
    assert exposure_lock_table(frame, ExposureLockSettings(enabled=False, reference=(1.0, 1.0, 1.0))) is None
    matched = ExposureLockSettings(enabled=True, region=WALL, reference=(40.0, 120.0, 200.0))
    assert exposure_lock_table(frame, matched) is None, "already at the reference: no LUT to apply"
    trace = RenderTrace()
    with tracing.active(trace):
        table = exposure_lock_table(
            frame, ExposureLockSettings(enabled=True, region=WALL, reference=(80.0, 120.0, 100.0))
        )
    assert table is not None and table.shape == (1, 256, 3)
    assert trace.notes["exposure.current"] == [40.0, 120.0, 200.0]
    assert trace.notes["exposure.gains"] == [2.0, 1.0, 0.5]


def test_render_frame_brings_a_darker_frame_back_to_the_reference(textured_frame: np.ndarray) -> None:
    reference = region_medians(textured_frame, WALL)
    darker = cv2.convertScaleAbs(textured_frame, alpha=0.7, beta=0)
    settings = RenderSettings(
        filters=FilterSettings(sharpen=0),
        exposure=ExposureLockSettings(enabled=True, region=WALL, reference=reference),
    )
    trace = RenderTrace()
    restored = render_frame(darker, settings, trace=trace)
    assert trace.notes["exposure.gains"][1] == pytest.approx(1 / 0.7, abs=0.05)
    assert np.allclose(region_medians(restored, WALL), reference, atol=2.0)
    assert np.abs(restored.astype(np.int16) - textured_frame.astype(np.int16)).mean() < 2.0
    untouched = render_frame(darker, RenderSettings(filters=FilterSettings(sharpen=0)), trace=RenderTrace())
    assert np.array_equal(untouched, darker)


def test_hunting_sequence_is_flattened_by_a_static_region() -> None:
    base = make_textured_frame(480, 270)
    rng = np.random.default_rng(3)
    ev = np.cumsum(rng.normal(0.0, 0.08, 40))
    ev -= ev[0]  # the first frame is the reference exposure
    wall = NormalizedRect(0.55, 0.05, 0.4, 0.25)  # gradient area, no shapes
    settings = RenderSettings(
        filters=FilterSettings(sharpen=0),
        exposure=ExposureLockSettings(enabled=True, region=wall, reference=region_medians(base, wall)),
    )
    errors_before, errors_after = [], []
    for stops in ev[1:]:
        gain = 2.0**stops
        shot = np.clip(base.astype(np.float32) * gain * np.array([1.05, 1.0, 0.95]), 0, 255).astype(np.uint8)
        fixed = render_frame(shot, settings)
        errors_before.append(np.abs(shot.astype(np.int16) - base.astype(np.int16)).mean())
        errors_after.append(np.abs(fixed.astype(np.int16) - base.astype(np.int16)).mean())
    assert np.mean(errors_before) > 8.0
    assert np.mean(errors_after) < np.mean(errors_before) * 0.3
