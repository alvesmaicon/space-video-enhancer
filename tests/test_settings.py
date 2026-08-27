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
    same_look,
    with_look,
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


def test_filter_presets_are_looks_that_leave_denoise_and_scale_alone() -> None:
    labels = [label for label, preset in FILTER_PRESETS if preset is not None]
    assert "Balanceado" in labels
    balanced = dict(FILTER_PRESETS)["Balanceado"]
    assert balanced is not None and balanced != FilterSettings()
    assert balanced.denoise_mode is DenoiseMode.OFF and balanced.scale_factor == 1.0, "a look never sets those"
    current = FilterSettings(denoise_mode=DenoiseMode.MANUAL, denoise_strength=40, scale_factor=2.0, contrast=70)
    applied = with_look(current, balanced)
    assert (applied.denoise_mode, applied.denoise_strength, applied.scale_factor) == (DenoiseMode.MANUAL, 40, 2.0)
    assert (applied.contrast, applied.sharpen) == (balanced.contrast, balanced.sharpen)
    assert same_look(applied, balanced) and not same_look(current, balanced)
    assert same_look(FilterSettings(), FilterSettings(denoise_strength=99, scale_factor=0.5))


def test_render_settings_output_size_follows_scale_rotation_and_crop() -> None:
    assert RenderSettings().output_size(1920, 1080) == (1920, 1080)
    scaled = RenderSettings(filters=FilterSettings(scale_factor=1.5))
    assert scaled.output_size(1280, 720) == (1920, 1080)
    rotated = RenderSettings(geometry=GeometrySettings(rotation=RotationMode.CW_90))
    assert rotated.output_size(1920, 1080) == (1080, 1920)
    cropped = RenderSettings(crop=CropSettings(enabled=True, rect=NormalizedRect(0.25, 0.0, 0.5, 1.0)))
    assert cropped.output_size(1920, 1080) == (960, 1080)
    assert RenderSettings(crop=CropSettings(enabled=False)).output_size(640, 480) == (640, 480)
