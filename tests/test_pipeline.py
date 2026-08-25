from __future__ import annotations

import cv2
import numpy as np
import pytest

from space_video_enhancer.pipeline.filters import OpenCLBackend, apply_basic_filters_cpu, build_processing_context
from space_video_enhancer.pipeline.presets import IMAGE_FILTER_PRESETS, apply_image_filter_preset
from space_video_enhancer.pipeline.render import render_frame
from space_video_enhancer.settings import (
    DenoiseMode,
    FilterSettings,
    ImageFilterSelection,
    NormalizedRect,
    RenderSettings,
)


def test_neutral_settings_are_identity(textured_frame: np.ndarray) -> None:
    settings = RenderSettings(filters=FilterSettings(sharpen=0))
    assert np.array_equal(render_frame(textured_frame, settings), textured_frame)


def test_processing_context_maps_slider_values() -> None:
    context = build_processing_context(
        FilterSettings(
            contrast=70, brightness=40, gamma=30, saturation=80, temperature=70, sharpen=60, scale_factor=1.5
        )
    )
    assert context.contrast_alpha == pytest.approx(1.3333333)
    assert context.brightness_beta == pytest.approx(-4.8)
    assert context.saturation == pytest.approx(1.6)
    assert context.temperature_shift == pytest.approx(0.4)
    assert context.sharpen_amount == pytest.approx(1.08)
    assert context.gamma_lookup_table is not None and context.gamma_lookup_table.shape == (256,)
    assert context.denoise_strength is None


def test_manual_denoise_strength_scaling() -> None:
    context = build_processing_context(
        FilterSettings(denoise_mode=DenoiseMode.MANUAL, denoise_strength=30, denoise_color_strength=0)
    )
    assert context.denoise_strength == pytest.approx(4.0)
    assert context.denoise_color_strength == pytest.approx(4.0)  # falls back to luma strength


def test_auto_denoise_needs_reference_frame(textured_frame: np.ndarray) -> None:
    without_frame = build_processing_context(FilterSettings(denoise_mode=DenoiseMode.AUTO))
    assert without_frame.denoise_strength is None
    with_frame = build_processing_context(
        FilterSettings(denoise_mode=DenoiseMode.AUTO), textured_frame, NormalizedRect(0, 0, 1, 1)
    )
    assert with_frame.denoise_strength is not None and with_frame.denoise_strength > 1.0


def test_denoise_reduces_noise(textured_frame: np.ndarray) -> None:
    context = build_processing_context(
        FilterSettings(denoise_mode=DenoiseMode.MANUAL, denoise_strength=40, denoise_color_strength=40, sharpen=0)
    )
    result = apply_basic_filters_cpu(textured_frame, context)
    patch = (slice(180, 260), slice(20, 100))  # flat gradient area
    assert cv2.meanStdDev(result[patch])[1].mean() < cv2.meanStdDev(textured_frame[patch])[1].mean()


def test_scale_factor_resizes_output(textured_frame: np.ndarray) -> None:
    result = render_frame(textured_frame, RenderSettings(filters=FilterSettings(scale_factor=1.5)))
    assert result.shape[:2] == (405, 720)


def test_image_preset_original_and_zero_intensity_are_identity(textured_frame: np.ndarray) -> None:
    assert apply_image_filter_preset(textured_frame, "original", 1.0) is textured_frame
    assert apply_image_filter_preset(textured_frame, "dynamic_balance", 0.0) is textured_frame
    assert apply_image_filter_preset(textured_frame, "does-not-exist", 1.0) is textured_frame


@pytest.mark.parametrize("preset", [preset for preset in IMAGE_FILTER_PRESETS if preset.filter_id != "original"])
def test_every_image_preset_changes_the_frame(textured_frame: np.ndarray, preset) -> None:  # type: ignore[no-untyped-def]
    result = apply_image_filter_preset(textured_frame, preset.filter_id, 1.0)
    assert result.shape == textured_frame.shape and result.dtype == np.uint8
    assert not np.array_equal(result, textured_frame)


def test_partial_intensity_blends_towards_original(textured_frame: np.ndarray) -> None:
    full = apply_image_filter_preset(textured_frame, "contrast_punch", 1.0).astype(np.int16)
    half = apply_image_filter_preset(textured_frame, "contrast_punch", 0.5).astype(np.int16)
    original = textured_frame.astype(np.int16)
    assert np.abs(half - original).mean() < np.abs(full - original).mean()


def test_render_frame_image_filter_override(textured_frame: np.ndarray) -> None:
    settings = RenderSettings(filters=FilterSettings(sharpen=0), image_filter=ImageFilterSelection("original", 1.0))
    overridden = render_frame(textured_frame, settings, image_filter=ImageFilterSelection("shadow_lift", 1.0))
    assert not np.array_equal(overridden, textured_frame)


def test_opencl_backend_disables_after_failure() -> None:
    backend = OpenCLBackend(enabled=True)
    backend.mark_failed()
    assert backend.is_available() is False
    assert "fallback" in backend.label
    disabled = OpenCLBackend(enabled=False)
    assert disabled.is_available() is False and disabled.label == "CPU"
