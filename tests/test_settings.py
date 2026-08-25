from __future__ import annotations

import dataclasses

import pytest

from space_video_enhancer.settings import (
    FILTER_PRESETS,
    CropSettings,
    DenoiseMode,
    FilterSettings,
    GeometrySettings,
    NormalizedRect,
    RenderSettings,
    RotationMode,
    StabilizationMode,
    StabilizationSettings,
)


def test_normalized_rect_to_pixels_clamps_and_keeps_minimum_size() -> None:
    rect = NormalizedRect(0.95, 0.95, 0.5, 0.5)
    assert rect.to_pixels(100, 50) == (95, 48, 5, 2)
    assert NormalizedRect(0, 0, 1, 1).to_pixels(0, 10) is None


def test_normalized_rect_even_alignment() -> None:
    rect = NormalizedRect(0.1, 0.1, 0.5, 0.5)
    x, y, width, height = rect.to_pixels(101, 51, even=True) or (1, 1, 1, 1)
    assert x % 2 == 0 and y % 2 == 0 and width % 2 == 0 and height % 2 == 0


def test_crop_pixel_rect_is_none_when_disabled_or_full_frame() -> None:
    assert CropSettings(enabled=False).pixel_rect(640, 360) is None
    assert CropSettings(enabled=True, rect=NormalizedRect(0, 0, 1, 1)).pixel_rect(640, 360) is None
    assert CropSettings(enabled=True, rect=NormalizedRect(0.25, 0.25, 0.5, 0.5)).pixel_rect(640, 360) == (
        160,
        90,
        320,
        180,
    )


def test_geometry_output_size_swaps_axes_for_quarter_turns() -> None:
    assert GeometrySettings(RotationMode.CW_90).output_size(1920, 1080) == (1080, 1920)
    assert GeometrySettings(RotationMode.ROTATE_180).output_size(1920, 1080) == (1920, 1080)
    assert RotationMode.CW_270.swaps_axes


@pytest.mark.parametrize(
    ("mode", "smoothing", "expected"),
    [
        (StabilizationMode.OFF, 0.5, False),
        (StabilizationMode.CAMERA_LOCK, 0.0, False),
        (StabilizationMode.CAMERA_LOCK, 0.2, True),
        (StabilizationMode.AUTO, 0.0, True),
        (StabilizationMode.SIMILARITY, 0.0, True),
        (StabilizationMode.HYBRID, 0.0, True),
    ],
)
def test_stabilization_requires_sequential(mode: StabilizationMode, smoothing: float, expected: bool) -> None:
    settings = StabilizationSettings(mode=mode, temporal_smoothing=smoothing)
    assert settings.requires_sequential_processing is expected


def test_render_settings_is_frozen_and_hashable() -> None:
    settings = RenderSettings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.filters = FilterSettings()  # type: ignore[misc]
    assert hash(settings) == hash(RenderSettings())


def test_filter_presets_match_by_equality() -> None:
    labels = [label for label, preset in FILTER_PRESETS if preset is not None]
    assert "Balanceado 1080p" in labels
    balanced = dict(FILTER_PRESETS)["Balanceado 1080p"]
    assert balanced is not None
    assert balanced == dataclasses.replace(balanced)
    assert balanced.denoise_mode is DenoiseMode.MANUAL
    assert balanced != FilterSettings()
