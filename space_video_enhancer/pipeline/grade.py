"""Color grading as lookup tables: white balance (channel gains + tint) and per-hue-band HSL.

Ambient light leaves footage yellowish or greenish; :func:`neutral_gains` turns a region that
should be gray into per-channel gains, and the tint slider covers the green/magenta axis the
temperature slider cannot. The hue bands are the "hue vs hue / hue vs sat / hue vs lum" curves
of grading tools reduced to eight anchors (red, orange, yellow, green, cyan, blue, purple,
magenta) with linear interpolation around the color circle, so a change to "green" fades out
toward yellow and cyan instead of leaving a hard edge.

Everything is precomputed into :class:`GradeTables` once per settings and applied with
``cv2.LUT`` / ``cv2.multiply`` on ``uint8``, on ``np.ndarray`` or ``cv2.UMat`` alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from space_video_enhancer.pipeline.geometry import region
from space_video_enhancer.settings import (
    HUE_BAND_CENTERS_DEG,
    WHITE_BALANCE_GAIN_MAX,
    WHITE_BALANCE_GAIN_MIN,
    ColorGradeSettings,
    NormalizedRect,
)

HUE_LEVELS = 180
"""OpenCV stores 8-bit hue as degrees / 2."""
TINT_GREEN_GAIN = 0.15
TINT_RB_GAIN = 0.075
GAIN_FIXED_POINT = 64.0
"""Saturation/luminance gains travel as uint8 with 1.0 == 64 (max 3.98x)."""
GAIN_MAX = 255.0 / GAIN_FIXED_POINT


@dataclass(frozen=True)
class GradeTables:
    channel_gains: np.ndarray | None
    """``(1, 256, 3)`` uint8: white-balance gains times tint, per B/G/R level. ``None`` when neutral."""
    hue: np.ndarray | None
    """``(256,)`` uint8: new hue for every OpenCV hue (entries >= 180 unused). ``None`` when no shift."""
    saturation_gain: np.ndarray | None
    """``(256,)`` uint8 fixed-point gain by original hue. ``None`` when all bands are at 0."""
    luminance_gain: np.ndarray | None

    @property
    def has_hue_bands(self) -> bool:
        return self.hue is not None or self.saturation_gain is not None or self.luminance_gain is not None


def band_weights(hue_degrees: np.ndarray) -> np.ndarray:
    """``(len(hue_degrees), bands)`` linear weights around the color circle; each row sums to 1."""
    centers = np.array(HUE_BAND_CENTERS_DEG, dtype=np.float64)
    count = len(centers)
    hue = np.mod(np.asarray(hue_degrees, dtype=np.float64), 360.0)
    weights = np.zeros((len(hue), count), dtype=np.float64)
    for index in range(count):
        start = centers[index]
        end = centers[(index + 1) % count] + (360.0 if index + 1 == count else 0.0)
        span = end - start
        # Hues in [start, end) are shared between band ``index`` and the next one.
        position = np.mod(hue - start, 360.0)
        inside = position < span
        fraction = np.where(inside, position / span, 0.0)
        weights[inside, index] += 1.0 - fraction[inside]
        weights[inside, (index + 1) % count] += fraction[inside]
    return weights


def tint_gains(tint: int) -> tuple[float, float, float]:
    """B, G, R gains for the tint slider: above 50 pulls toward magenta (less green), below toward green."""
    amount = (tint - 50) / 50.0
    return (1.0 + TINT_RB_GAIN * amount, 1.0 - TINT_GREEN_GAIN * amount, 1.0 + TINT_RB_GAIN * amount)


def channel_gain_table(gains: tuple[float, float, float]) -> np.ndarray | None:
    if all(abs(gain - 1.0) <= 1e-3 for gain in gains):
        return None
    levels = np.arange(256, dtype=np.float64)
    table = np.stack([np.clip(np.rint(levels * gain), 0, 255) for gain in gains], axis=1)
    return table.astype(np.uint8).reshape(1, 256, 3)


def build_grade_tables(settings: ColorGradeSettings) -> GradeTables | None:
    """Resolve ``settings`` into lookup tables, or ``None`` when everything is neutral."""
    if settings.is_neutral:
        return None
    tint = tint_gains(settings.tint)
    gains = tuple(wb * t for wb, t in zip(settings.white_balance_gains, tint))
    channel_gains = channel_gain_table((gains[0], gains[1], gains[2]))

    hue = saturation_gain = luminance_gain = None
    if settings.has_hue_bands:
        weights = band_weights(np.arange(HUE_LEVELS) * 2.0)
        shifts = weights @ np.array([band.hue_shift for band in settings.bands], dtype=np.float64)
        saturation = weights @ np.array([1.0 + band.saturation / 100.0 for band in settings.bands])
        luminance = weights @ np.array([1.0 + band.luminance / 100.0 for band in settings.bands])
        if np.any(shifts != 0.0):
            new_hue = np.mod(np.rint(np.arange(HUE_LEVELS) + shifts / 2.0), HUE_LEVELS)
            hue = np.arange(256, dtype=np.uint8)
            hue[:HUE_LEVELS] = new_hue.astype(np.uint8)
        if np.any(saturation != 1.0):
            saturation_gain = _gain_table(saturation)
        if np.any(luminance != 1.0):
            luminance_gain = _gain_table(luminance)
    return GradeTables(channel_gains, hue, saturation_gain, luminance_gain)


def _gain_table(gain_by_hue: np.ndarray) -> np.ndarray:
    table = np.full(256, GAIN_FIXED_POINT, dtype=np.float64)
    table[:HUE_LEVELS] = np.clip(gain_by_hue, 0.0, GAIN_MAX) * GAIN_FIXED_POINT
    return np.rint(table).astype(np.uint8)


def apply_white_balance(image: object, tables: GradeTables | None) -> object:
    if tables is None or tables.channel_gains is None:
        return image
    return cv2.LUT(image, tables.channel_gains)


def apply_hue_bands(image: object, tables: GradeTables | None) -> object:
    """Per-hue hue shift and saturation/luminance gains; works on ``np.ndarray`` or ``cv2.UMat``."""
    if tables is None or not tables.has_hue_bands:
        return image
    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    original_hue = hue
    scale = 1.0 / GAIN_FIXED_POINT
    if tables.saturation_gain is not None:
        saturation = cv2.multiply(saturation, cv2.LUT(original_hue, tables.saturation_gain), scale=scale)
    if tables.luminance_gain is not None:
        value = cv2.multiply(value, cv2.LUT(original_hue, tables.luminance_gain), scale=scale)
    if tables.hue is not None:
        hue = cv2.LUT(original_hue, tables.hue)
    return cv2.cvtColor(cv2.merge([hue, saturation, value]), cv2.COLOR_HSV2BGR)


def neutral_gains(frame: np.ndarray, rect: NormalizedRect | None) -> tuple[float, float, float]:
    """B, G, R gains that make ``rect`` (or the whole frame: gray world) neutral gray."""
    sample = frame if rect is None else region(frame, rect)
    if sample is None or sample.size == 0:
        sample = frame
    means = np.asarray(cv2.mean(sample)[:3], dtype=np.float64)
    if np.any(means <= 0.0):
        return (1.0, 1.0, 1.0)
    target = float(means.mean())
    gains = np.clip(target / means, WHITE_BALANCE_GAIN_MIN, WHITE_BALANCE_GAIN_MAX)
    return (round(float(gains[0]), 3), round(float(gains[1]), 3), round(float(gains[2]), 3))
