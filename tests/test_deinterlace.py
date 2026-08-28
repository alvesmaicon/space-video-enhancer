"""Deinterlacing: the comb detector and the two filters on synthetic interlaced frames."""

from __future__ import annotations

import numpy as np

from space_video_enhancer.pipeline.deinterlace import (
    LIKELY_INTERLACED_SCORE,
    comb_score,
    deinterlace,
    likely_interlaced,
    prepare_source,
)
from space_video_enhancer.pipeline.render import render_frame
from space_video_enhancer.settings import CaptureSettings, DeinterlaceMode, RenderSettings
from space_video_enhancer.telemetry.trace import RenderTrace

WIDTH, HEIGHT = 160, 96


def _scene(edge_x: int) -> np.ndarray:
    """A dark frame with a bright vertical bar whose left edge is at ``edge_x`` and a static bright box."""
    frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
    frame[:, edge_x : edge_x + 24] = 220
    frame[8:24, 8:40] = 200  # static detail: must survive untouched
    return frame


def _interlaced(shift: int = 12) -> np.ndarray:
    """Even rows from the scene at one instant, odd rows a half frame later (the bar moved ``shift`` px)."""
    first, second = _scene(70), _scene(70 + shift)
    frame = first.copy()
    frame[1::2] = second[1::2]
    return frame


def test_comb_score_separates_interlaced_motion_from_progressive_frames() -> None:
    progressive = comb_score(_scene(70))
    combed = comb_score(_interlaced())
    assert progressive < LIKELY_INTERLACED_SCORE / 4 and not likely_interlaced(progressive)
    assert combed > 3 * LIKELY_INTERLACED_SCORE and likely_interlaced(combed)


def test_adaptive_deinterlace_rebuilds_only_the_combed_area() -> None:
    frame = _interlaced()
    result = deinterlace(frame, DeinterlaceMode.ADAPTIVE)
    assert result is not frame and result.shape == frame.shape
    assert np.array_equal(result[0::2], frame[0::2]), "the even field is the one kept"
    assert np.array_equal(result[:, :60], frame[:, :60]), "static columns (including the box) are untouched"
    # Inside the comb both bar edges disagreed between fields; afterwards the odd rows follow the even ones.
    combed_columns = slice(70, 70 + 24 + 12)
    assert np.abs(result[1:-1:2, combed_columns].astype(int) - result[0:-2:2, combed_columns].astype(int)).max() <= 1
    assert comb_score(result) < LIKELY_INTERLACED_SCORE / 4


def test_field_mode_interpolates_every_odd_row_and_off_is_a_no_op() -> None:
    frame = _interlaced()
    assert deinterlace(frame, DeinterlaceMode.OFF) is frame
    result = deinterlace(frame, DeinterlaceMode.FIELD)
    expected = ((frame[0:-2:2].astype(np.uint16) + frame[2::2].astype(np.uint16) + 1) >> 1).astype(np.uint8)
    assert np.array_equal(result[1:-1:2], expected)
    assert np.array_equal(result[-1], frame[-2]), "the last odd row repeats the even row above it"
    assert comb_score(result) == 0.0


def test_render_frame_deinterlaces_first_and_reports_the_score() -> None:
    frame = _interlaced()
    trace = RenderTrace()
    settings = RenderSettings(capture=CaptureSettings(deinterlace=DeinterlaceMode.FIELD))
    rendered = render_frame(frame, settings, trace=trace)
    assert trace.notes["deinterlace.mode"] == "field" and trace.notes["deinterlace.comb"] > LIKELY_INTERLACED_SCORE
    assert comb_score(rendered) == 0.0
    prepared, score = prepare_source(frame, DeinterlaceMode.OFF)
    assert prepared is frame and score == trace.notes["deinterlace.comb"]
    skipped = RenderTrace()
    render_frame(frame, settings, trace=skipped, source_prepared=True)
    assert "deinterlace.comb" not in skipped.notes, "the preview prepares the full-size frame itself"
