"""Exposure lock: hold a reference region's brightness (and colour) constant across the clip.

Cameras on auto exposure / auto white balance re-meter when the focus hunts or something
passes the lens, so the same wall gets brighter or bluer from one second to the next. The
fix is stateless per frame: measure a static region (per-channel medians), compare with the
same region in a reference frame, apply the ratio as per-channel gains to the whole frame. No
temporal state means preview, sequential export and parallel export all agree.

Measured on the *source* frame (before stabilization), the region means the same thing as
the rectangle drawn on the "Antes" side of the preview and as a click on "use this frame".
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from space_video_enhancer.pipeline.grade import channel_gain_table, region
from space_video_enhancer.settings import ExposureLockSettings, NormalizedRect
from space_video_enhancer.telemetry import trace as tracing

LUMA_WEIGHTS_BGR = (0.114, 0.587, 0.299)
MEDIAN_SUBSAMPLE_PIXELS = 400_000
"""Larger samples are read with a stride of 2: the median of a wall does not need every pixel."""


def region_medians(frame: np.ndarray, rect: NormalizedRect | None) -> tuple[float, float, float]:
    """Per-channel (B, G, R) medians of ``rect`` — the whole frame when ``None`` or degenerate."""
    sample = frame if rect is None else region(frame, rect)
    if sample is None or sample.size == 0:
        sample = frame
    if sample.shape[0] * sample.shape[1] > MEDIAN_SUBSAMPLE_PIXELS:
        sample = sample[::2, ::2]
    sample = np.ascontiguousarray(sample)
    medians = []
    for channel in range(3):
        histogram = cv2.calcHist([sample], [channel], None, [256], [0, 256]).ravel()
        cumulative = np.cumsum(histogram)
        medians.append(float(np.searchsorted(cumulative, cumulative[-1] / 2.0)))
    return (medians[0], medians[1], medians[2])


def exposure_gains(
    reference: tuple[float, float, float],
    current: tuple[float, float, float],
    *,
    strength: float = 1.0,
    max_ev: float = 1.0,
    color: bool = True,
) -> tuple[float, float, float]:
    """B, G, R gains that bring ``current`` to ``reference``, clamped to ±``max_ev`` and eased by ``strength``.

    ``color=False`` uses one gain for all channels (brightness only, from the luma of the medians).
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.maximum(np.asarray(current, dtype=np.float64), 1.0)
    if color:
        raw = ref / cur
    else:
        weights = np.asarray(LUMA_WEIGHTS_BGR)
        raw = np.full(3, float(weights @ ref) / max(float(weights @ cur), 1.0))
    limit = 2.0 ** max(0.0, max_ev)
    raw = np.clip(raw, 1.0 / limit, limit)
    gains = 1.0 + (raw - 1.0) * min(max(strength, 0.0), 1.0)
    return (round(float(gains[0]), 4), round(float(gains[1]), 4), round(float(gains[2]), 4))


def gains_in_ev(gains: tuple[float, float, float]) -> float:
    """Luma gain expressed in stops (for labels and logs)."""
    luma = sum(weight * gain for weight, gain in zip(LUMA_WEIGHTS_BGR, gains))
    return math.log2(max(luma, 1e-6))


def exposure_lock_table(frame: np.ndarray, settings: ExposureLockSettings) -> np.ndarray | None:
    """The per-channel LUT that locks ``frame`` to the reference; ``None`` when the lock is off or already matched.

    Notes ``exposure.current`` (this frame's medians) and ``exposure.gains`` in the active trace.
    """
    if not settings.is_active or settings.reference is None:
        return None
    current = region_medians(frame, settings.region)
    gains = exposure_gains(
        settings.reference,
        current,
        strength=settings.strength / 100.0,
        max_ev=settings.max_ev,
        color=settings.color,
    )
    tracing.note("exposure.current", [round(value, 1) for value in current])
    tracing.note("exposure.gains", list(gains))
    return channel_gain_table(gains)
