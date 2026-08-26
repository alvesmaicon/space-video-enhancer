"""Regression: the crop drawn on the (rotated/flipped) preview must be the crop applied on export."""

from __future__ import annotations

import numpy as np
import pytest

from space_video_enhancer.pipeline.geometry import (
    apply_crop,
    apply_geometry,
    ensure_even_size,
    fit_to_aspect,
    fit_to_canvas,
)
from space_video_enhancer.pipeline.render import render_frame
from space_video_enhancer.settings import (
    CropSettings,
    FilterSettings,
    GeometrySettings,
    NormalizedRect,
    RenderSettings,
    RotationMode,
)

NEUTRAL_FILTERS = FilterSettings(sharpen=0)


def _dominant_color(frame: np.ndarray) -> tuple[int, int, int]:
    mean = frame.reshape(-1, 3).mean(axis=0)
    return tuple(round(channel) for channel in mean)  # type: ignore[return-value]


@pytest.mark.parametrize(
    "geometry",
    [
        GeometrySettings(RotationMode.CW_90),
        GeometrySettings(RotationMode.CW_270),
        GeometrySettings(RotationMode.ROTATE_180),
        GeometrySettings(flip_horizontal=True),
        GeometrySettings(flip_vertical=True),
        GeometrySettings(RotationMode.CW_90, flip_horizontal=True),
    ],
)
def test_export_crop_matches_preview_overlay(quadrant_frame: np.ndarray, geometry: GeometrySettings) -> None:
    crop_rect = NormalizedRect(0.0, 0.0, 0.5, 0.5)  # top-left quadrant of what the user sees
    settings = RenderSettings(filters=NEUTRAL_FILTERS, geometry=geometry, crop=CropSettings(True, crop_rect))

    displayed = apply_geometry(quadrant_frame, geometry)
    height, width = displayed.shape[:2]
    under_overlay = displayed[: height // 2, : width // 2]

    exported = render_frame(quadrant_frame, settings, crop=True)

    assert _dominant_color(exported) == _dominant_color(under_overlay)
    assert exported.shape[:2] == under_overlay.shape[:2]


def test_export_crop_without_rotation_is_unchanged_behaviour(quadrant_frame: np.ndarray) -> None:
    settings = RenderSettings(filters=NEUTRAL_FILTERS, crop=CropSettings(True, NormalizedRect(0.5, 0.5, 0.5, 0.5)))
    exported = render_frame(quadrant_frame, settings, crop=True)
    assert _dominant_color(exported) == (0, 255, 255)  # bottom-right quadrant is yellow


def test_preview_render_never_crops(quadrant_frame: np.ndarray) -> None:
    settings = RenderSettings(filters=NEUTRAL_FILTERS, crop=CropSettings(True, NormalizedRect(0.0, 0.0, 0.5, 0.5)))
    preview = render_frame(quadrant_frame, settings, crop=False)
    assert preview.shape == quadrant_frame.shape


def test_apply_crop_is_noop_when_disabled(quadrant_frame: np.ndarray) -> None:
    assert apply_crop(quadrant_frame, CropSettings(enabled=False)) is quadrant_frame


def test_ensure_even_size_trims_at_most_one_pixel() -> None:
    odd = np.zeros((65, 33, 3), np.uint8)
    assert ensure_even_size(odd).shape[:2] == (64, 32)
    even = np.zeros((64, 32, 3), np.uint8)
    assert ensure_even_size(even) is even


def test_fit_to_canvas_letterboxes_and_is_identity_when_sizes_match() -> None:
    frame = np.full((48, 96, 3), 200, dtype=np.uint8)
    assert fit_to_canvas(frame, 96, 48) is frame
    fitted = fit_to_canvas(frame, 64, 64)
    assert fitted.shape == (64, 64, 3)
    assert fitted[:16].max() == 0 and fitted[48:].max() == 0 and fitted[16:48].min() == 200
    tall = fit_to_canvas(np.full((100, 20, 3), 90, dtype=np.uint8), 64, 64)
    assert tall.shape == (64, 64, 3) and tall[:, :25].max() == 0 and tall[:, 26:38].min() == 90
    gray = fit_to_canvas(np.full((10, 10), 7, dtype=np.uint8), 20, 12)
    assert gray.shape == (12, 20) and gray[:, 4:16].min() == 7


def test_fit_to_aspect_pads_without_scaling() -> None:
    frame = np.full((180, 240, 3), 200, np.uint8)  # 4:3
    wide = fit_to_aspect(frame, 16 / 9)
    assert wide.shape == (180, 320, 3)
    assert wide[:, :40].max() == 0 and wide[:, -40:].max() == 0 and wide[:, 40:280].min() == 200
    tall = fit_to_aspect(frame, 1.0)
    assert tall.shape == (240, 240, 3) and tall[:30].max() == 0 and tall[30:210].min() == 200
    assert fit_to_aspect(frame, 4 / 3) is frame and fit_to_aspect(frame, None) is frame
    assert fit_to_aspect(frame, 4 / 3 * 1.005) is frame, "within the tolerance nothing is padded"
