from __future__ import annotations

import cv2
import numpy as np
import pytest

from space_video_enhancer.pipeline.denoise import estimate_denoise_strengths, estimate_noise_sigma
from space_video_enhancer.pipeline.filters import (
    OpenCLBackend,
    apply_basic_filters,
    apply_basic_filters_cpu,
    apply_basic_filters_opencl,
    apply_saturation,
    apply_temperature,
    build_processing_context,
    saturation_lookup_table,
    temperature_lookup_table,
)
from space_video_enhancer.pipeline.presets import IMAGE_FILTER_PRESETS, apply_image_filter_preset
from space_video_enhancer.pipeline.render import render_frame
from space_video_enhancer.settings import (
    DenoiseMode,
    FilterSettings,
    ImageFilterSelection,
    NormalizedRect,
    RenderSettings,
)
from space_video_enhancer.telemetry import trace as tracing
from space_video_enhancer.telemetry.trace import RenderTrace


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
    assert with_frame.denoise_strength is not None and with_frame.denoise_strength >= 1.0  # clean frame: floor


def _with_noise(frame: np.ndarray, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(int(sigma * 10))
    noise = rng.normal(0.0, sigma, frame.shape[:2])[..., None] + rng.normal(0.0, sigma * 0.5, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _smooth_frame() -> np.ndarray:
    """Gradients plus a few hard edges and no noise (the textured fixture carries synthetic noise on purpose)."""
    rows = np.linspace(30, 220, 240, dtype=np.float32)[:, None]
    cols = np.linspace(0, 60, 320, dtype=np.float32)[None, :]
    frame = np.stack([rows + cols, rows * 0.8 + 20 + 0 * cols, 250 - rows + 0 * cols], axis=2)
    frame[60:120, 40:140] = (200, 40, 90)
    frame[150:200, 180:300] = (30, 180, 60)
    return np.clip(frame, 0, 255).astype(np.uint8)


def test_noise_estimate_tracks_the_actual_noise() -> None:
    clean = _smooth_frame()
    assert estimate_noise_sigma(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)) < 1.0
    for sigma in (3.0, 6.0, 12.0):
        noisy_gray = cv2.cvtColor(_with_noise(clean, sigma), cv2.COLOR_BGR2GRAY)
        assert estimate_noise_sigma(noisy_gray) == pytest.approx(sigma, abs=1.0)
    everything = NormalizedRect(0, 0, 1, 1)
    strengths = [
        estimate_denoise_strengths(_with_noise(clean, sigma), DenoiseMode.AUTO, everything)
        for sigma in (3.0, 6.0, 12.0)
    ]
    assert strengths[0][0] < strengths[1][0] < strengths[2][0]
    assert strengths[1][0] == pytest.approx(0.8 * 6.0, abs=1.0)
    assert estimate_noise_sigma(np.zeros((4, 4), dtype=np.uint8)) == 0.0  # too small to say anything


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


def test_scale_factor_below_one_downscales_before_denoise(textured_frame: np.ndarray) -> None:
    """0.5x must actually halve the output (it used to be a no-op), and the denoise runs on the smaller frame."""
    settings = RenderSettings(
        filters=FilterSettings(denoise_mode=DenoiseMode.MANUAL, denoise_strength=20, scale_factor=0.5)
    )
    trace = RenderTrace()
    result = render_frame(textured_frame, settings, trace=trace)
    height, width = textured_frame.shape[:2]
    assert result.shape[:2] == (height // 2, width // 2), "the output is half size"
    order = list(trace.stages)
    assert order.index("filters.scale") < order.index("filters.denoise"), "downscale happens before the denoise"


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


def _reference_saturation(frame: np.ndarray, saturation: float) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 1] = np.clip(np.rint(hsv[:, :, 1] * saturation), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _reference_temperature(frame: np.ndarray, shift: float) -> np.ndarray:
    adjusted = frame.astype(np.float64)
    adjusted[:, :, 2] *= 1.0 + (0.22 * shift)
    adjusted[:, :, 1] *= 1.0 - (0.05 * abs(shift))
    adjusted[:, :, 0] *= 1.0 - (0.22 * shift)
    return np.clip(np.rint(adjusted), 0, 255).astype(np.uint8)


def test_saturation_and_temperature_lookup_tables_match_the_float_math(textured_frame: np.ndarray) -> None:
    saturated = apply_saturation(textured_frame, saturation_lookup_table(1.3))
    assert isinstance(saturated, np.ndarray)
    assert np.array_equal(saturated, _reference_saturation(textured_frame, 1.3))
    warmed = apply_temperature(textured_frame, temperature_lookup_table(0.4))
    assert isinstance(warmed, np.ndarray)
    assert np.array_equal(warmed, _reference_temperature(textured_frame, 0.4))
    cooled = apply_temperature(textured_frame, temperature_lookup_table(-0.6))
    assert isinstance(cooled, np.ndarray)
    assert np.array_equal(cooled, _reference_temperature(textured_frame, -0.6))
    assert saturation_lookup_table(1.0) is None and temperature_lookup_table(0.0) is None
    assert apply_saturation(textured_frame, None) is textured_frame


def test_umat_chain_matches_the_cpu_chain(textured_frame: np.ndarray) -> None:
    context = build_processing_context(
        FilterSettings(contrast=58, brightness=47, gamma=42, saturation=64, temperature=38, sharpen=30)
    )
    cpu = apply_basic_filters_cpu(textured_frame, context)
    device = apply_basic_filters_opencl(textured_frame, context)  # UMat runs on the CPU when OpenCL is absent
    assert device.shape == cpu.shape
    # OpenCL kernels (HSV conversion, blur) round differently by a level here and there; the sharpen
    # then amplifies that. Identical when OpenCL is absent.
    difference = np.abs(device.astype(np.int16) - cpu.astype(np.int16))
    assert difference.mean() < 1.0 and np.percentile(difference, 99) <= 2


class _AlwaysAvailableBackend(OpenCLBackend):
    def is_available(self) -> bool:
        return True


def _backend_note(frame: np.ndarray, settings: FilterSettings) -> str:
    trace = RenderTrace()
    with tracing.active(trace):
        apply_basic_filters(frame, build_processing_context(settings), _AlwaysAvailableBackend())
    return str(trace.notes["filters.backend"])


def test_opencl_is_only_used_when_denoise_is_on(textured_frame: np.ndarray) -> None:
    assert _backend_note(textured_frame, FilterSettings(saturation=70, sharpen=40)) == "cpu_no_denoise"
    denoised = FilterSettings(denoise_mode=DenoiseMode.MANUAL, denoise_strength=20, denoise_color_strength=20)
    assert _backend_note(textured_frame, denoised) == "opencl"
