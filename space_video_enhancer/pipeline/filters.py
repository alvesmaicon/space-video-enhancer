"""Basic colour/detail chain: denoise, contrast, brightness, gamma, saturation, temperature, sharpen, scale.

Everything after the denoise is lookup tables and OpenCV primitives on ``uint8``, so the same
:func:`apply_chain` runs on ``np.ndarray`` (CPU) or ``cv2.UMat`` (OpenCL) with a single
upload/download. OpenCL is only used when NLMeans is on: for the light filters the transfer
costs more than the work.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from space_video_enhancer.pipeline.denoise import estimate_denoise_strengths
from space_video_enhancer.settings import DenoiseMode, FilterSettings, NormalizedRect
from space_video_enhancer.telemetry import trace as tracing

MANUAL_DENOISE_DIVISOR = 7.5
CONTRAST_DIVISOR = 60.0
CONTRAST_MIN_ALPHA = 0.2
BRIGHTNESS_RANGE = 24.0
GAMMA_EPSILON = 0.01
SATURATION_EPSILON = 0.01
TEMPERATURE_EPSILON = 0.01
SHARPEN_MAX_AMOUNT = 1.8
SHARPEN_SIGMA = 1.2
TEMPERATURE_RB_GAIN = 0.22
TEMPERATURE_G_LOSS = 0.05


@dataclass(frozen=True)
class ProcessingContext:
    """Filter parameters resolved from :class:`FilterSettings` (and, for auto/sample denoise, a frame)."""

    denoise_strength: float | None
    denoise_color_strength: float | None
    denoise_template_window: int
    denoise_search_window: int
    contrast_alpha: float
    brightness_beta: float
    gamma_lookup_table: np.ndarray | None
    saturation: float
    saturation_lookup_table: np.ndarray | None
    """S-channel table for ``saturation``; ``None`` when neutral."""
    temperature_shift: float
    temperature_lookup_table: np.ndarray | None
    """Per-channel (B, G, R) table for ``temperature_shift``; ``None`` when neutral."""
    sharpen_amount: float
    scale_factor: float


def gamma_value(slider_value: int) -> float:
    if slider_value >= 50:
        return 1.0 + ((slider_value - 50) / 50.0)
    return 0.2 + ((slider_value / 50.0) * 0.8)


def saturation_value(slider_value: int) -> float:
    return max(0.0, slider_value / 50.0)


def temperature_shift(slider_value: int) -> float:
    return (slider_value - 50) / 50.0


def gamma_lookup_table(gamma: float) -> np.ndarray | None:
    if abs(gamma - 1.0) <= GAMMA_EPSILON:
        return None
    levels = np.arange(256, dtype=np.float64) / 255.0
    return np.clip((levels**gamma) * 255.0, 0, 255).astype(np.uint8)


def saturation_lookup_table(saturation: float) -> np.ndarray | None:
    """Table applied to the HSV saturation channel (rounded, not truncated)."""
    if abs(saturation - 1.0) <= SATURATION_EPSILON:
        return None
    return np.clip(np.rint(np.arange(256, dtype=np.float64) * saturation), 0, 255).astype(np.uint8)


def temperature_lookup_table(shift: float) -> np.ndarray | None:
    """``(1, 256, 3)`` table so one ``cv2.LUT`` call scales B, G and R by their own gains."""
    if abs(shift) < TEMPERATURE_EPSILON:
        return None
    gains = (
        1.0 - (TEMPERATURE_RB_GAIN * shift),
        1.0 - (TEMPERATURE_G_LOSS * abs(shift)),
        1.0 + (TEMPERATURE_RB_GAIN * shift),
    )
    levels = np.arange(256, dtype=np.float64)
    table = np.stack([np.clip(np.rint(levels * gain), 0, 255) for gain in gains], axis=1)
    return table.astype(np.uint8).reshape(1, 256, 3)


def build_processing_context(
    settings: FilterSettings,
    reference_frame: np.ndarray | None = None,
    sample_rect: NormalizedRect | None = None,
) -> ProcessingContext:
    denoise_strength: float | None = None
    denoise_color_strength: float | None = None
    if settings.denoise_mode is DenoiseMode.MANUAL:
        if settings.denoise_strength > 0:
            denoise_strength = max(1.0, settings.denoise_strength / MANUAL_DENOISE_DIVISOR)
        if settings.denoise_color_strength > 0:
            denoise_color_strength = max(1.0, settings.denoise_color_strength / MANUAL_DENOISE_DIVISOR)
    elif (
        settings.denoise_mode in (DenoiseMode.AUTO, DenoiseMode.SAMPLE)
        and reference_frame is not None
        and sample_rect is not None
    ):
        denoise_strength, denoise_color_strength = estimate_denoise_strengths(
            reference_frame,
            settings.denoise_mode,
            sample_rect,
        )

    if denoise_strength is not None and denoise_color_strength is None:
        denoise_color_strength = denoise_strength
    if denoise_color_strength is not None and denoise_strength is None:
        denoise_strength = denoise_color_strength
    tracing.note("denoise.mode", settings.denoise_mode.value)
    if denoise_strength is not None:
        tracing.note("denoise.h", [round(denoise_strength, 2), round(denoise_color_strength or 0.0, 2)])

    contrast = 1.0 + ((settings.contrast - 50) / CONTRAST_DIVISOR)
    saturation = saturation_value(settings.saturation)
    shift = temperature_shift(settings.temperature)
    brightness = ((settings.brightness - 50) / 50.0) * BRIGHTNESS_RANGE
    sharpen_amount = (settings.sharpen / 100.0) * SHARPEN_MAX_AMOUNT if settings.sharpen > 0 else 0.0

    return ProcessingContext(
        denoise_strength=denoise_strength,
        denoise_color_strength=denoise_color_strength,
        denoise_template_window=settings.denoise_template_window,
        denoise_search_window=settings.denoise_search_window,
        contrast_alpha=max(CONTRAST_MIN_ALPHA, contrast),
        brightness_beta=brightness,
        gamma_lookup_table=gamma_lookup_table(gamma_value(settings.gamma)),
        saturation=saturation,
        saturation_lookup_table=saturation_lookup_table(saturation),
        temperature_shift=shift,
        temperature_lookup_table=temperature_lookup_table(shift),
        sharpen_amount=sharpen_amount,
        scale_factor=settings.scale_factor,
    )


class OpenCLBackend:
    """Thread-safe switch for the OpenCV T-API (UMat) path.

    Availability is probed once; a failure anywhere disables the path for the rest of
    the session without touching OpenCV's global flag from worker threads.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._lock = threading.Lock()
        self._checked = False
        self._available = False
        self._failed = False

    def reset(self) -> None:
        with self._lock:
            self._checked = False
            self._available = False
            self._failed = False

    def is_available(self) -> bool:
        if not self._enabled:
            return False
        with self._lock:
            if not self._checked:
                self._checked = True
                self._available = self._probe()
            return self._available and not self._failed

    def mark_failed(self) -> None:
        with self._lock:
            self._failed = True

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def label(self) -> str:
        if not self._enabled:
            return "CPU"
        if self._failed:
            return "CPU (fallback do OpenCL)"
        if self.is_available():
            return "OpenCL"
        return "CPU (OpenCL indisponivel)"

    @staticmethod
    def _probe() -> bool:
        try:
            if not hasattr(cv2, "ocl") or not cv2.ocl.haveOpenCL():
                return False
            cv2.ocl.setUseOpenCL(True)
            return bool(cv2.ocl.useOpenCL())
        except cv2.error:
            return False


def apply_saturation(image: object, table: np.ndarray | None) -> object:
    """Scale the HSV saturation channel through ``table``; works on ``np.ndarray`` or ``cv2.UMat``."""
    if table is None:
        return image
    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    return cv2.cvtColor(cv2.merge([hue, cv2.LUT(saturation, table), value]), cv2.COLOR_HSV2BGR)


def apply_temperature(image: object, table: np.ndarray | None) -> object:
    if table is None:
        return image
    return cv2.LUT(image, table)


def apply_chain(image: object, context: ProcessingContext) -> object:
    """The whole basic chain on one image type (``np.ndarray`` or ``cv2.UMat``)."""
    result = image
    if context.denoise_strength is not None:
        with tracing.stage("filters.denoise"):
            result = cv2.fastNlMeansDenoisingColored(
                result,
                None,
                context.denoise_strength,
                context.denoise_color_strength or context.denoise_strength,
                context.denoise_template_window,
                context.denoise_search_window,
            )
    result = cv2.convertScaleAbs(result, alpha=context.contrast_alpha, beta=context.brightness_beta)
    if context.gamma_lookup_table is not None:
        result = cv2.LUT(result, context.gamma_lookup_table)
    result = apply_saturation(result, context.saturation_lookup_table)
    result = apply_temperature(result, context.temperature_lookup_table)
    if context.sharpen_amount > 0:
        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=SHARPEN_SIGMA)
        result = cv2.addWeighted(result, 1.0 + context.sharpen_amount, blurred, -context.sharpen_amount, 0)
    if context.scale_factor > 1.0:
        result = cv2.resize(
            result,
            None,
            fx=context.scale_factor,
            fy=context.scale_factor,
            interpolation=cv2.INTER_CUBIC,
        )
    return result


def apply_basic_filters_cpu(frame: np.ndarray, context: ProcessingContext) -> np.ndarray:
    result = apply_chain(frame, context)
    assert isinstance(result, np.ndarray)
    return result


def apply_basic_filters_opencl(frame: np.ndarray, context: ProcessingContext) -> np.ndarray:
    """One upload, the whole chain on the device, one download."""
    result = apply_chain(cv2.UMat(frame), context)
    assert isinstance(result, cv2.UMat)
    return result.get()


def apply_basic_filters(
    frame: np.ndarray,
    context: ProcessingContext,
    backend: OpenCLBackend | None = None,
) -> np.ndarray:
    if backend is not None and backend.is_available():
        if context.denoise_strength is None:
            # Without NLMeans the upload/download costs more than the whole chain on the CPU.
            tracing.note("filters.backend", "cpu_no_denoise")
            return apply_basic_filters_cpu(frame, context)
        try:
            result = apply_basic_filters_opencl(frame, context)
        except cv2.error as error:
            backend.mark_failed()
            tracing.note("filters.opencl_error", str(error).splitlines()[0][:200])
        else:
            tracing.note("filters.backend", "opencl")
            return result
    tracing.note("filters.backend", "cpu")
    return apply_basic_filters_cpu(frame, context)
