"""Noise estimation used by the AUTO and SAMPLE denoise modes.

The noise level is the robust spread of a Laplacian response: white noise of standard
deviation ``sigma`` produces a response of standard deviation ``6 * sigma`` (the kernel's L2
norm), while edges and texture are sparse outliers that the median absolute deviation ignores.
The previous estimate used the standard deviation of the image itself, which mostly measured
contrast: it returned ``h ~ 6.9`` for the same frame whether ``sigma`` was 0 or 12.
"""

from __future__ import annotations

import cv2
import numpy as np

from space_video_enhancer.pipeline.geometry import region
from space_video_enhancer.settings import DenoiseMode, NormalizedRect
from space_video_enhancer.telemetry.trace import note

FALLBACK_SAMPLE_FRACTION = 0.35
FALLBACK_SAMPLE_MIN_PIXELS = 32
DEFAULT_ESTIMATE = (2.0, 2.0)
LAPLACIAN_KERNEL = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32)
LAPLACIAN_NOISE_GAIN = 6.0
"""L2 norm of the kernel: white noise of std ``sigma`` gives a response of std ``6 * sigma``."""
MAD_TO_SIGMA = 1.4826
ESTIMATE_STRIDE = 4
"""One response sample in 16 (57k at 720p) gives the same median as all of them; 15 -> 7 ms at 720p."""
MIN_SAMPLE_SIDE = 8
NOISE_TO_LUMA_H = 0.8
NOISE_TO_COLOR_H = 0.8
"""NLMeans ``h`` per unit of noise std (benchmarks/denoise_variants.py: within 0.3 dB of the PSNR optimum)."""
ESTIMATE_MIN = 1.0
ESTIMATE_MAX = 12.0


def sample_region(frame: np.ndarray, sample_rect: NormalizedRect) -> np.ndarray:
    """The user-selected sample region, falling back to a centered patch."""
    frame_height, frame_width = frame.shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        return frame

    selected = region(frame, sample_rect)
    if selected is not None:
        return selected

    sample_width = max(FALLBACK_SAMPLE_MIN_PIXELS, int(frame_width * FALLBACK_SAMPLE_FRACTION))
    sample_height = max(FALLBACK_SAMPLE_MIN_PIXELS, int(frame_height * FALLBACK_SAMPLE_FRACTION))
    start_x = max(0, (frame_width - sample_width) // 2)
    start_y = max(0, (frame_height - sample_height) // 2)
    return frame[start_y : start_y + sample_height, start_x : start_x + sample_width]


def reference_region(frame: np.ndarray, mode: DenoiseMode, sample_rect: NormalizedRect) -> np.ndarray:
    if mode is DenoiseMode.SAMPLE:
        return sample_region(frame, sample_rect)
    return frame


def estimate_noise_sigma(channel: np.ndarray) -> float:
    """Robust noise std of one 8-bit channel (median absolute deviation of its Laplacian response)."""
    if channel.ndim != 2 or channel.shape[0] < MIN_SAMPLE_SIDE or channel.shape[1] < MIN_SAMPLE_SIDE:
        return 0.0
    response = cv2.filter2D(channel, cv2.CV_32F, LAPLACIAN_KERNEL, borderType=cv2.BORDER_REFLECT)
    interior = response[1:-1:ESTIMATE_STRIDE, 1:-1:ESTIMATE_STRIDE]
    deviation = np.abs(interior - np.median(interior))
    return MAD_TO_SIGMA * float(np.median(deviation)) / LAPLACIAN_NOISE_GAIN


def estimate_denoise_strengths(
    frame: np.ndarray,
    mode: DenoiseMode,
    sample_rect: NormalizedRect,
) -> tuple[float, float]:
    """Estimate (luma, color) NLMeans ``h`` values from the noise in the reference region."""
    sample = reference_region(frame, mode, sample_rect)
    if sample.size == 0 or sample.ndim != 3:
        return DEFAULT_ESTIMATE

    luma, a, b = cv2.split(cv2.cvtColor(sample, cv2.COLOR_BGR2Lab))
    sigma_luma = estimate_noise_sigma(luma)
    sigma_color = (estimate_noise_sigma(a) + estimate_noise_sigma(b)) / 2.0
    note("denoise.sigma", [round(sigma_luma, 2), round(sigma_color, 2)])
    estimated_luma = float(np.clip(NOISE_TO_LUMA_H * sigma_luma, ESTIMATE_MIN, ESTIMATE_MAX))
    estimated_color = float(np.clip(NOISE_TO_COLOR_H * sigma_color, ESTIMATE_MIN, ESTIMATE_MAX))
    return (estimated_luma, estimated_color)
