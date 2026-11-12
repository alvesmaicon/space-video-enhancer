"""Perspective correction: four points on a real rectangle -> a rectangle on screen, before the crop."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from space_video_enhancer.pipeline.geometry import (
    apply_crop,
    apply_geometry,
    normalized_from_geometry,
    normalized_through_geometry,
)
from space_video_enhancer.pipeline.perspective import (
    apply_perspective,
    estimate_plane_aspect,
    order_quad,
    plan_perspective,
    points_tuple,
    quad_grid_lines,
)
from space_video_enhancer.pipeline.preview import PreviewRequest, render_preview
from space_video_enhancer.pipeline.render import render_frame
from space_video_enhancer.settings import (
    CropSettings,
    FilterSettings,
    GeometrySettings,
    NormalizedRect,
    PerspectiveFit,
    PerspectiveSettings,
    RenderSettings,
    RotationMode,
)
from space_video_enhancer.telemetry import trace as tracing
from space_video_enhancer.telemetry.trace import RenderTrace

NEUTRAL_FILTERS = FilterSettings(sharpen=0)
WIDTH, HEIGHT = 640, 360
GEOMETRIES = [
    GeometrySettings(),
    GeometrySettings(RotationMode.CW_90),
    GeometrySettings(RotationMode.ROTATE_180),
    GeometrySettings(RotationMode.CW_270),
    GeometrySettings(flip_horizontal=True),
    GeometrySettings(RotationMode.CW_90, flip_vertical=True),
]


def _rotation(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _filmed_rectangle(
    aspect: float, angles: tuple[float, float, float], *, focal: float, size: tuple[int, int] = (WIDTH, HEIGHT)
) -> np.ndarray:
    """Pixel corners (TL, TR, BR, BL) of an ``aspect`` rectangle filmed by a pinhole camera at ``angles``."""
    width, height = size
    corners = np.array([[-aspect / 2, -0.5, 0], [aspect / 2, -0.5, 0], [aspect / 2, 0.5, 0], [-aspect / 2, 0.5, 0]])
    rotation = _rotation("z", angles[2]) @ _rotation("y", angles[1]) @ _rotation("x", angles[0])
    camera = corners @ rotation.T + np.array([0.05, -0.03, 2.4])
    u = focal * camera[:, 0] / camera[:, 2] + width / 2
    v = focal * camera[:, 1] / camera[:, 2] + height / 2
    return np.stack([u, v], axis=1)


def _settings_for(quad_pixels: np.ndarray, fit: PerspectiveFit = PerspectiveFit.REAL) -> PerspectiveSettings:
    normalized = (quad_pixels + 0.5) / np.array([WIDTH, HEIGHT])
    return PerspectiveSettings(enabled=True, points=points_tuple(normalized), fit=fit)


def _plane_frame(quad_pixels: np.ndarray) -> np.ndarray:
    """Black frame with the filmed rectangle painted white."""
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    cv2.fillConvexPoly(frame, np.round(quad_pixels).astype(np.int32), (255, 255, 255), lineType=cv2.LINE_AA)
    return frame


def _white_box(frame: np.ndarray) -> tuple[int, int, int, int, float]:
    """Bounding box (x, y, w, h) of the white pixels and how much of the box they fill."""
    mask = frame[..., 0] > 127
    ys, xs = np.nonzero(mask)
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
    return x, y, w, h, float(mask[y : y + h, x : x + w].mean())


# --- ordering and validity ---------------------------------------------------------------------------


def test_points_are_ordered_clockwise_from_top_left_whatever_the_drag_order() -> None:
    expected = np.array([[10.0, 20.0], [300.0, 5.0], [320.0, 200.0], [0.0, 180.0]])
    for permutation in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
        assert np.array_equal(order_quad(expected[permutation]), expected)


@pytest.mark.parametrize(
    "points",
    [
        [(0.2, 0.2), (0.8, 0.2), (0.3, 0.3), (0.2, 0.8)],  # a dart: one corner pushed inside
        [(0.2, 0.2), (0.5, 0.2), (0.8, 0.2), (0.5, 0.21)],  # nearly collinear
        [(0.5, 0.5), (0.51, 0.5), (0.51, 0.51), (0.5, 0.51)],  # a misclick, not a plane
    ],
)
def test_points_that_do_not_make_a_plane_leave_the_frame_alone(points: list[tuple[float, float]]) -> None:
    settings = PerspectiveSettings(enabled=True, points=points_tuple(np.array(points)))
    frame = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
    assert plan_perspective(settings, GeometrySettings(), WIDTH, HEIGHT) is None
    trace = RenderTrace()
    with tracing.active(trace):
        assert apply_perspective(frame, settings, GeometrySettings()) is frame
    assert trace.notes["perspective.result"] == "invalid"


def test_disabled_correction_is_free() -> None:
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    assert apply_perspective(frame, PerspectiveSettings(), GeometrySettings()) is frame
    assert render_frame(frame, RenderSettings(filters=NEUTRAL_FILTERS)).shape == frame.shape


# --- the plane's real proportion -----------------------------------------------------------------------


@pytest.mark.parametrize("aspect", [4 / 3, 16 / 9, 1.0, 0.7])
@pytest.mark.parametrize("angles", [(0.45, 0.5, 0.08), (0.6, -0.35, 0.0), (-0.3, 0.4, -0.05)])
def test_real_proportion_is_measured_from_the_vanishing_points(
    aspect: float, angles: tuple[float, float, float]
) -> None:
    focal = 1.1 * WIDTH  # not the assumed lens: the estimate must come from the points
    quad = order_quad(_filmed_rectangle(aspect, angles, focal=focal))
    estimated, source = estimate_plane_aspect(quad, WIDTH, HEIGHT)
    assert source == "measured"
    assert estimated == pytest.approx(aspect, rel=1e-3)


def test_symmetric_keystone_falls_back_to_a_typical_lens() -> None:
    """Top and bottom parallel on screen: one vanishing point is at infinity and the focal cannot be measured."""
    quad = order_quad(_filmed_rectangle(4 / 3, (0.5, 0.0, 0.0), focal=0.72 * WIDTH))
    estimated, source = estimate_plane_aspect(quad, WIDTH, HEIGHT)
    assert source == "assumed"
    assert estimated == pytest.approx(4 / 3, rel=1e-3), "a phone-like lens assumed: exact for one"


# --- the warp -----------------------------------------------------------------------------------------


def test_real_fit_turns_the_plane_into_a_rectangle_of_its_proportion_in_the_same_frame() -> None:
    quad = _filmed_rectangle(4 / 3, (0.45, 0.5, 0.08), focal=1.1 * WIDTH)
    frame = _plane_frame(quad)
    corrected = apply_perspective(frame, _settings_for(quad), GeometrySettings())
    assert corrected.shape == frame.shape
    _x, _y, w, h, fill = _white_box(corrected)
    assert fill > 0.97, "the white plane fills its bounding box: it became a rectangle"
    assert w / h == pytest.approx(4 / 3, rel=0.03)
    assert w * h == pytest.approx(np.count_nonzero(frame[..., 0] > 127), rel=0.06), "about the same area"


@pytest.mark.parametrize(
    ("fit", "size"), [(PerspectiveFit.STRETCH_16_9, (640, 360)), (PerspectiveFit.STRETCH_9_16, (360, 640))]
)
def test_stretch_makes_the_plane_fill_a_16_9_or_9_16_output(fit: PerspectiveFit, size: tuple[int, int]) -> None:
    quad = _filmed_rectangle(1.0, (0.45, 0.5, 0.08), focal=1.1 * WIDTH)
    corrected = apply_perspective(_plane_frame(quad), _settings_for(quad, fit), GeometrySettings())
    assert corrected.shape[1::-1] == size, "the same pixel count in the chosen shape"
    inner = corrected[3:-3, 3:-3, 0]
    assert inner.min() > 200, "the plane covers the whole output: the correction no longer shrinks the picture"


def test_output_size_follows_the_stretch_and_the_crop_comes_after() -> None:
    stretched = PerspectiveSettings(enabled=True, fit=PerspectiveFit.STRETCH_9_16)
    assert stretched.output_size(1920, 1080) == (1080, 1920)
    assert PerspectiveSettings(enabled=True, fit=PerspectiveFit.STRETCH_16_9).output_size(1080, 1920) == (1920, 1080)
    assert replace_fit(stretched, PerspectiveFit.REAL).output_size(1920, 1080) == (1920, 1080)
    assert PerspectiveSettings(fit=PerspectiveFit.STRETCH_9_16).output_size(1920, 1080) == (1920, 1080), "off"
    settings = RenderSettings(perspective=stretched, crop=CropSettings(True, NormalizedRect(0.0, 0.0, 1.0, 0.5)))
    assert settings.output_size(1920, 1080) == (1080, 960)


def replace_fit(settings: PerspectiveSettings, fit: PerspectiveFit) -> PerspectiveSettings:
    return PerspectiveSettings(settings.enabled, settings.points, fit)


def test_nothing_beyond_the_plane_horizon_is_wrapped_into_the_picture() -> None:
    """A floor seen at a grazing angle: its vanishing line crosses the frame (y ~ 0.59).

    Without the mask about 7% of the output samples the region past that line, mirrored.
    """
    frame = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
    frame[: int(HEIGHT * 0.55)] = (0, 0, 255)  # beyond the floor's horizon: a red wall
    points = points_tuple(np.array([[0.45, 0.6], [0.55, 0.6], [0.9, 0.7], [0.1, 0.7]]))
    corrected = apply_perspective(frame, PerspectiveSettings(enabled=True, points=points), GeometrySettings())
    red = (corrected[..., 2] > 200) & (corrected[..., 1] < 60)
    assert not red.any(), "the wall beyond the horizon showed up mirrored"
    assert (corrected == 128).all(axis=2).mean() > 0.2, "the floor itself is still there"


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_points_stay_on_the_content_when_the_orientation_changes(
    quadrant_frame: np.ndarray, geometry: GeometrySettings
) -> None:
    source_points = np.array([[0.1, 0.15], [0.85, 0.05], [0.95, 0.9], [0.05, 0.8]])
    settings = PerspectiveSettings(enabled=True, points=points_tuple(source_points))
    rendered = render_frame(
        quadrant_frame, RenderSettings(filters=NEUTRAL_FILTERS, geometry=geometry, perspective=settings)
    )
    displayed = apply_geometry(quadrant_frame, geometry)
    on_screen = PerspectiveSettings(
        enabled=True, points=points_tuple(normalized_through_geometry(source_points, geometry))
    )
    assert np.array_equal(rendered, apply_perspective(displayed, on_screen, GeometrySettings()))


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_displayed_points_map_back_to_the_source(geometry: GeometrySettings) -> None:
    points = np.array([[0.1, 0.2], [0.7, 0.05], [0.9, 0.6], [0.3, 0.95]])
    assert np.allclose(normalized_from_geometry(normalized_through_geometry(points, geometry), geometry), points)


def test_crop_is_applied_to_the_corrected_frame(quadrant_frame: np.ndarray) -> None:
    perspective = PerspectiveSettings(
        enabled=True, points=points_tuple(np.array([[0.2, 0.1], [0.75, 0.2], [0.85, 0.85], [0.1, 0.95]]))
    )
    crop = CropSettings(True, NormalizedRect(0.25, 0.25, 0.5, 0.5))
    exported = render_frame(
        quadrant_frame, RenderSettings(filters=NEUTRAL_FILTERS, perspective=perspective, crop=crop), crop=True
    )
    corrected = render_frame(quadrant_frame, RenderSettings(filters=NEUTRAL_FILTERS, perspective=perspective))
    assert np.array_equal(exported, apply_crop(corrected, crop)), "the rectangle drawn on the corrected preview"
    assert exported.shape[:2] == RenderSettings(perspective=perspective, crop=crop).output_size(WIDTH, HEIGHT)[::-1]


def test_render_trace_reports_the_correction(quadrant_frame: np.ndarray) -> None:
    quad = _filmed_rectangle(4 / 3, (0.45, 0.5, 0.08), focal=1.1 * WIDTH)
    trace = RenderTrace()
    render_frame(quadrant_frame, RenderSettings(filters=NEUTRAL_FILTERS, perspective=_settings_for(quad)), trace=trace)
    assert "perspective" in trace.stages
    assert trace.notes["perspective.result"] == "applied" and trace.notes["perspective.focal"] == "measured"
    assert trace.notes["perspective.plane_aspect"] == pytest.approx(4 / 3, rel=1e-2)


def test_stretched_preview_pads_the_before_frame_instead_of_squashing_it(quadrant_frame: np.ndarray) -> None:
    perspective = PerspectiveSettings(enabled=True, fit=PerspectiveFit.STRETCH_9_16)
    request = PreviewRequest(
        generation=1,
        frame=quadrant_frame,
        frame_index=0,
        settings=RenderSettings(filters=NEUTRAL_FILTERS, perspective=perspective),
        size_limit=None,
        stabilization_reference=None,
        stabilization_epoch=0,
    )
    result = render_preview(request, None, None)
    assert result.after.shape[:2] == (640, 360)
    assert result.before.shape[:2] == (1138, 640), "the whole 16:9 frame, padded to 9:16"
    assert np.array_equal(result.before[389:749], quadrant_frame)


def test_grid_lines_follow_the_plane() -> None:
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    lines = quad_grid_lines(square, 4)
    assert len(lines) == 6
    assert np.allclose(lines[0], [[0.25, 0.0], [0.25, 1.0]])
    assert quad_grid_lines(np.zeros((4, 2))) == []
