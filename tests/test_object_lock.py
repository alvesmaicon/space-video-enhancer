"""The OBJECT mode: multi-region lock that survives losing regions (see docs/PLANO-ESTABILIZACAO-UNIFICADA.md)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from space_video_enhancer.pipeline.stabilization.lock import object_lock_transform
from space_video_enhancer.pipeline.stabilization.stabilize import apply_stabilization
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.project import settings_from_dict, settings_to_dict
from space_video_enhancer.settings import NormalizedRect, StabilizationMode, StabilizationSettings
from space_video_enhancer.telemetry import trace as tracing
from space_video_enhancer.telemetry.trace import RenderTrace

WIDTH, HEIGHT = 480, 270
MARGIN = 60
REGION_A = NormalizedRect(0.06, 0.08, 0.22, 0.30)
REGION_B = NormalizedRect(0.72, 0.62, 0.22, 0.30)
"""Two regions in opposite corners of the frame — the case one anchor cannot serve."""


def _canvas() -> np.ndarray:
    rng = np.random.default_rng(7)
    canvas = rng.integers(0, 255, (HEIGHT + 2 * MARGIN, WIDTH + 2 * MARGIN), dtype=np.uint8)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.2)
    for _ in range(300):
        x = int(rng.integers(0, WIDTH + 2 * MARGIN))
        y = int(rng.integers(0, HEIGHT + 2 * MARGIN))
        cv2.circle(canvas, (x, y), int(rng.integers(3, 10)), int(rng.integers(0, 255)), -1)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _shifted_frame(canvas: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """The camera moved by ``(dx, dy)``: the window into the canvas slides the other way."""
    transform = np.array([[1.0, 0.0, MARGIN - dx], [0.0, 1.0, MARGIN - dy]], dtype=np.float32)
    return cv2.warpAffine(canvas, transform, (WIDTH, HEIGHT), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


def _occlude(frame: np.ndarray, region: NormalizedRect, pad: int = 48) -> np.ndarray:
    """Cover ``region`` (and the search margin around it) with a flat block: nothing to track or match."""
    pixels = region.to_pixels(WIDTH, HEIGHT)
    assert pixels is not None
    x, y, w, h = pixels
    covered = frame.copy()
    covered[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad] = 128
    return covered


def _measured_shift(reference: np.ndarray, frame: np.ndarray) -> tuple[float, float]:
    gray_a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (shift_x, shift_y), _response = cv2.phaseCorrelate(gray_a, gray_b)
    return float(shift_x), float(shift_y)


def test_object_lock_survives_alternating_occlusions() -> None:
    """The plan's acceptance: one region always alive keeps the target still while the other is covered."""
    canvas = _canvas()
    reference = _shifted_frame(canvas, 0.0, 0.0)
    shifts = [(3.0, -2.0), (6.0, 1.0), (4.0, 4.0), (7.0, -3.0), (2.0, 2.0), (5.0, 0.0), (6.0, -4.0), (3.0, 3.0)]
    settings = StabilizationSettings(
        mode=StabilizationMode.OBJECT,
        temporal_smoothing=0.0,
        auto_crop=False,
        focus_regions=(REGION_A, REGION_B),
    )
    state = StabilizationState()
    for index, (dx, dy) in enumerate(shifts):
        frame = _shifted_frame(canvas, dx, dy)
        if 2 <= index < 5:
            frame = _occlude(frame, REGION_A)
        elif index >= 5:
            frame = _occlude(frame, REGION_B)
        trace = RenderTrace()
        with tracing.active(trace):
            corrected = apply_stabilization(
                frame, settings, frame_index=index + 1, state=state, reference_frame=reference
            )
        residual = _measured_shift(reference, corrected)
        live, total = trace.notes.get("stabilization.lock_regions", (0, 0))
        assert total == 2 and live >= 1, f"frame {index}: {trace.notes.get('stabilization.lock_regions')}"
        assert abs(residual[0]) < 1.5 and abs(residual[1]) < 1.5, (
            f"frame {index} (occluded={'A' if 2 <= index < 5 else 'B' if index >= 5 else '-'}): "
            f"residual {residual} against shift {(dx, dy)}"
        )


def test_two_opposite_regions_recover_the_rotation() -> None:
    canvas = _canvas()
    reference = _shifted_frame(canvas, 0.0, 0.0)
    angle_degrees = 2.0
    rotation = cv2.getRotationMatrix2D((WIDTH / 2.0, HEIGHT / 2.0), angle_degrees, 1.0)
    rotated = cv2.warpAffine(reference, rotation, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)

    transform = object_lock_transform(rotated, reference, (REGION_A, REGION_B))
    assert transform is not None
    recovered = math.degrees(math.atan2(float(transform[1, 0]), float(transform[0, 0])))
    assert recovered == pytest.approx(angle_degrees, abs=0.3), "current -> reference undoes the rotation"


def test_all_regions_lost_dead_reckons_on_the_global_motion() -> None:
    canvas = _canvas()
    reference = _shifted_frame(canvas, 0.0, 0.0)
    settings = StabilizationSettings(
        mode=StabilizationMode.OBJECT,
        temporal_smoothing=0.0,
        auto_crop=False,
        focus_regions=(REGION_A, REGION_B),
    )
    state = StabilizationState()
    paths = []
    for index in range(1, 7):
        frame = _shifted_frame(canvas, 3.0 * index, 0.0)
        if index >= 3:
            frame = _occlude(_occlude(frame, REGION_A), REGION_B)
        trace = RenderTrace()
        with tracing.active(trace):
            apply_stabilization(frame, settings, frame_index=index, state=state, reference_frame=reference)
        paths.append(trace.notes.get("stabilization.path"))
    assert paths[0] == "lock" and paths[1] == "lock"
    assert "dead_reckoning" in paths[2:], f"all regions lost must fall back to the global motion: {paths}"


def test_object_settings_flags_and_project_round_trip() -> None:
    settings = StabilizationSettings(mode=StabilizationMode.OBJECT, focus_regions=(REGION_A, REGION_B))
    assert settings.enabled and settings.mode.uses_focus_regions and settings.mode.needs_reference_frame
    assert not settings.mode.uses_focal_point, "the legacy single-region flag stays with CAMERA_LOCK/HYBRID"
    assert settings.needs_mesh_analysis, "the object mode builds its fields from the same mesh track"
    assert not settings.requires_sequential_processing, "per-frame fields keep the export parallel"
    wobbling = StabilizationSettings(mode=StabilizationMode.OBJECT, rolling_shutter_auto=True)
    assert not wobbling.applies_wobble, "the mesh under the object mode already absorbs the wobble"

    data = settings_to_dict(settings)
    loaded = settings_from_dict(StabilizationSettings, data)
    assert loaded == settings, "focus_regions survive the project file round trip"
