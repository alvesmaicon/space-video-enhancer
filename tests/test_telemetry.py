from __future__ import annotations

import io
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np

from space_video_enhancer.pipeline.preview import PreviewRequest, render_preview
from space_video_enhancer.pipeline.render import render_frame
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import (
    DenoiseMode,
    FilterSettings,
    NormalizedRect,
    RenderSettings,
    StabilizationMode,
    StabilizationSettings,
)
from space_video_enhancer.telemetry import trace as tracing
from space_video_enhancer.telemetry.diff import diff, flatten
from space_video_enhancer.telemetry.environment import collect_environment
from space_video_enhancer.telemetry.log import (
    LOGGER_NAME,
    JsonLinesFormatter,
    current_session,
    get_logger,
    log_event,
    setup_logging,
)
from space_video_enhancer.telemetry.stats import Histogram, RollingStats, StageStats
from space_video_enhancer.telemetry.trace import RenderTrace

# --- trace -----------------------------------------------------------------------------------


def test_trace_records_stages_and_notes() -> None:
    trace = RenderTrace()
    with tracing.active(trace):
        assert tracing.current() is trace
        with tracing.stage("work"):
            pass
        tracing.note("answer", 42)
    assert tracing.current() is None
    payload = trace.finish().as_dict()
    assert payload["stages"]["work"] >= 0.0
    assert payload["answer"] == 42
    assert payload["total_ms"] >= 0.0


def test_stage_and_note_are_noops_without_an_active_trace() -> None:
    with tracing.stage("ignored"):
        tracing.note("ignored", True)
    assert tracing.current() is None


def test_render_frame_trace_has_pipeline_stages(textured_frame: np.ndarray) -> None:
    trace = RenderTrace()
    render_frame(textured_frame, RenderSettings(), trace=trace)
    assert {"stabilize", "context", "filters", "preset", "geometry"} <= set(trace.stages)
    assert trace.notes["filters.backend"] in {"cpu", "opencl"}
    assert trace.notes["input_shape"] == [270, 480]
    assert trace.notes["preset"] == "original"


def test_trace_records_translation_stabilization_decisions(textured_frame: np.ndarray) -> None:
    previous = np.roll(textured_frame, (5, -7), axis=(0, 1))
    settings = RenderSettings(stabilization=StabilizationSettings(mode=StabilizationMode.AUTO, temporal_smoothing=0.0))
    state = StabilizationState()

    primed = RenderTrace()
    render_frame(previous, settings, frame_index=9, stabilization_state=state, trace=primed)
    assert primed.notes["stabilization.estimator"] == "primed"
    assert primed.notes["stabilization.result"] == "no_transform"

    applied = RenderTrace()
    render_frame(textured_frame, settings, frame_index=10, stabilization_state=state, trace=applied)
    assert applied.notes["stabilization.estimator"] == "lk_translation"
    assert applied.notes["stabilization.result"] == "applied"
    assert applied.notes["stabilization.tracked_points"] >= 6
    assert abs(applied.notes["stabilization.translation_px"][0]) > 1.0


def test_trace_records_camera_lock_path(textured_frame: np.ndarray) -> None:
    previous = np.roll(textured_frame, (5, -7), axis=(0, 1))
    settings = RenderSettings(
        stabilization=StabilizationSettings(
            mode=StabilizationMode.CAMERA_LOCK,
            temporal_smoothing=0.0,
            focus_rect=NormalizedRect(0.10, 0.12, 0.28, 0.42),
        )
    )
    trace = RenderTrace()
    render_frame(textured_frame, settings, stabilization_reference=previous, trace=trace)
    assert trace.notes["stabilization.estimator"].startswith("orb")
    assert trace.notes["stabilization.good_matches"] >= 8
    assert trace.notes["stabilization.inliers"] >= 6


def test_trace_records_denoise_estimate(textured_frame: np.ndarray) -> None:
    trace = RenderTrace()
    render_frame(textured_frame, RenderSettings(filters=FilterSettings(denoise_mode=DenoiseMode.AUTO)), trace=trace)
    assert trace.notes["denoise.mode"] == "auto"
    assert len(trace.notes["denoise.h"]) == 2 and trace.notes["denoise.h"][0] > 1.0
    assert "filters.denoise" in trace.stages


def test_render_preview_reports_queue_wait_and_trace(textured_frame: np.ndarray) -> None:
    request = PreviewRequest(1, textured_frame, 0, RenderSettings(), None, None, 0, reason="scrub")
    result = render_preview(request, StabilizationState(), None)
    assert result.queue_wait_ms >= 0.0
    assert result.trace.notes["preview.reason"] == "scrub"
    assert "before" in result.trace.stages


# --- stats -----------------------------------------------------------------------------------


def test_rolling_stats_summary() -> None:
    stats = RollingStats(window=10)
    assert stats.summary() == {"count": 0}
    for value in range(1, 21):
        stats.add(value)
    assert stats.count == 20 and len(stats) == 10  # window keeps the last 10
    assert stats.last == 20.0
    assert stats.mean == 15.5
    assert stats.percentile(50) == 15.5
    assert stats.max == 20.0
    assert set(stats.summary()) == {"count", "mean", "p50", "p95", "max"}


def test_stage_stats_and_histogram() -> None:
    stage_stats = StageStats()
    for _ in range(3):
        trace = RenderTrace()
        trace.stages = {"filters": 5.0, "preset": 1.0}
        trace.total_ms = 6.0
        stage_stats.add(trace)
    summary = stage_stats.summary()
    assert summary["total"]["count"] == 3 and summary["stages"]["filters"]["mean"] == 5.0

    histogram = Histogram()
    for key in ("lk", "lk", None, "orb"):
        histogram.add(key)
    assert histogram.as_dict() == {"lk": 2, "orb": 1} and len(histogram) == 3


# --- diff ------------------------------------------------------------------------------------


def test_flatten_and_diff_settings() -> None:
    base = RenderSettings()
    flat = flatten(base)
    assert flat["filters.contrast"] == 50 and flat["stabilization.mode"] == "off"
    changed = replace(base, filters=replace(base.filters, contrast=62, denoise_mode=DenoiseMode.MANUAL))
    assert diff(base, changed) == {"filters.contrast": [50, 62], "filters.denoise_mode": ["off", "manual"]}
    assert diff(None, base)["crop.rect.x"] == [None, 0.1]


# --- logging ---------------------------------------------------------------------------------


def test_json_formatter_serializes_awkward_values() -> None:
    logger = logging.getLogger("sve-test-json")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonLinesFormatter("abc123"))
    logger.addHandler(handler)

    log_event(
        logger, "unit.event", value=np.float32(1.5), mode=DenoiseMode.AUTO, array=np.zeros((2, 3)), path=Path("x")
    )
    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "unit.event" and payload["session"] == "abc123" and payload["level"] == "INFO"
    assert payload["value"] == 1.5 and payload["mode"] == "auto" and payload["path"] == "x"
    assert payload["array"] == {"ndarray": [2, 3], "dtype": "float64"}
    assert payload["ts"].endswith("Z")


def test_setup_logging_writes_json_lines(tmp_path: Path) -> None:
    session = setup_logging(log_dir=tmp_path, level="DEBUG", stderr=False)
    try:
        assert current_session() is session and session.log_path.parent == tmp_path
        log_event(get_logger("unit"), "unit.hello", answer=42)
        log_event(get_logger("unit"), "unit.debug", level=logging.DEBUG)
        lines = session.log_path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        assert [event["event"] for event in events] == ["unit.hello", "unit.debug"]
        assert (
            events[0]["answer"] == 42
            and events[0]["logger"] == "sve.unit"
            and events[0]["session"] == session.session_id
        )
    finally:
        root = logging.getLogger(LOGGER_NAME)
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()


# --- environment -----------------------------------------------------------------------------


def test_collect_environment_without_ffmpeg() -> None:
    env = collect_environment(None)
    assert {"app_version", "python", "platform", "cpu_count", "opencv", "numpy", "opencl", "ffmpeg"} <= set(env)
    assert env["ffmpeg"] is None and "available" in env["opencl"]
