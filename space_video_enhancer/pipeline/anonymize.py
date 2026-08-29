"""Blur or pixelate faces and a chosen region, to anonymise people or hide a detail (a tattoo).

Two sources of boxes are covered on every frame: the faces a Haar cascade finds automatically
(``haarcascade_frontalface_default.xml``, which ships with OpenCV — no model to download) and one
manual rectangle the user drew. Each box is either mosaicked (``PIXELATE``) or Gaussian-blurred
(``BLUR``); ``strength`` sets how coarse. Detection runs on a downscaled grey copy for speed and the
boxes are scaled back. The cascade is thread-local: the export renders frames on several threads and
``cv2.CascadeClassifier`` must not be shared across them.

This hides — it does not *remove*. A tattoo under a blur is covered, not replaced with skin; convincing
removal needs generative inpainting, which is out of scope here.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from space_video_enhancer.settings import AnonymizeMode, AnonymizeSettings, NormalizedRect

FACE_CASCADE_FILE = "haarcascade_frontalface_default.xml"
DETECT_MAX_SIDE = 640
"""Detect on a copy no larger than this on its long side: faces this small are still found, much faster."""
FACE_PADDING = 0.18
"""Grow each detected face box by this fraction so hair and chin are covered, not just the tight face."""

_local = threading.local()


def _cascade() -> cv2.CascadeClassifier:
    cascade = getattr(_local, "cascade", None)
    if cascade is None:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_CASCADE_FILE)
        _local.cascade = cascade
    return cascade


def _clamp_box(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(x, width))
    y0 = max(0, min(y, height))
    x1 = max(x0, min(x + w, width))
    y1 = max(y0, min(y + h, height))
    return x0, y0, x1 - x0, y1 - y0


def detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Frontal-face boxes ``(x, y, w, h)`` in ``frame`` pixels, padded a little; empty when none are found."""
    cascade = _cascade()
    if cascade.empty():
        return []
    height, width = frame.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(1, width, height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    minimum = max(24, int(min(gray.shape[:2]) * 0.08))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(minimum, minimum))
    boxes = []
    for x, y, w, h in faces:
        pad_x, pad_y = w * FACE_PADDING, h * FACE_PADDING
        boxes.append(
            _clamp_box(
                round((x - pad_x) / scale),
                round((y - pad_y) / scale),
                round((w + 2 * pad_x) / scale),
                round((h + 2 * pad_y) / scale),
                width,
                height,
            )
        )
    return boxes


def _region_box(region: NormalizedRect, width: int, height: int) -> tuple[int, int, int, int]:
    return _clamp_box(
        round(region.x * width),
        round(region.y * height),
        round(region.width * width),
        round(region.height * height),
        width,
        height,
    )


def _pixelate(patch: np.ndarray, strength: int) -> np.ndarray:
    height, width = patch.shape[:2]
    # strength 0..100 -> blocks from ~1/6 of the side (fine) down to a 3x3 mosaic (coarse).
    divisor = 3 + (100 - strength) / 100.0 * 15.0
    blocks_x = max(1, int(width / divisor))
    blocks_y = max(1, int(height / divisor))
    small = cv2.resize(patch, (blocks_x, blocks_y), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def _blur(patch: np.ndarray, strength: int) -> np.ndarray:
    side = min(patch.shape[:2])
    radius = max(1, int(side * (0.06 + strength / 100.0 * 0.24)))
    kernel = radius * 2 + 1  # odd
    return cv2.GaussianBlur(patch, (kernel, kernel), 0)


def apply_anonymize(frame: np.ndarray, settings: AnonymizeSettings) -> np.ndarray:
    """``frame`` with every face (when asked) and the manual region blurred or pixelated; unchanged when inactive."""
    if not settings.is_active:
        return frame
    height, width = frame.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    if settings.region is not None and settings.region.is_valid():
        boxes.append(_region_box(settings.region, width, height))
    if settings.detect_faces:
        boxes.extend(detect_faces(frame))
    result = frame.copy()
    cover = _pixelate if settings.mode is AnonymizeMode.PIXELATE else _blur
    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        result[y : y + h, x : x + w] = cover(result[y : y + h, x : x + w], settings.strength)
    return result
