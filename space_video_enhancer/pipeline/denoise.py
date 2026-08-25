"""Noise estimation used by the AUTO and SAMPLE denoise modes."""

from __future__ import annotations

import cv2
import numpy as np

from space_video_enhancer.pipeline.geometry import region
from space_video_enhancer.settings import DenoiseMode, NormalizedRect

FALLBACK_SAMPLE_FRACTION = 0.35
FALLBACK_SAMPLE_MIN_PIXELS = 32
DEFAULT_ESTIMATE = (2.0, 2.0)
LUMA_STD_DIVISOR = 10.0
COLOR_STD_DIVISOR = 12.0
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


def estimate_denoise_strengths(
    frame: np.ndarray,
    mode: DenoiseMode,
    sample_rect: NormalizedRect,
) -> tuple[float, float]:
    """Estimate (luma, color) NLMeans ``h`` values from the noise in the reference region."""
    sample = reference_region(frame, mode, sample_rect)
    if sample.size == 0:
        return DEFAULT_ESTIMATE

    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    luma_std = float(cv2.meanStdDev(gray)[1][0][0])
    lab_frame = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
    color_std = float(np.mean(cv2.meanStdDev(lab_frame[:, :, 1:])[1]))
    estimated_luma = float(np.clip(1.0 + (luma_std / LUMA_STD_DIVISOR), ESTIMATE_MIN, ESTIMATE_MAX))
    estimated_color = float(np.clip(0.8 + (color_std / COLOR_STD_DIVISOR), ESTIMATE_MIN, ESTIMATE_MAX))
    return (estimated_luma, estimated_color)
