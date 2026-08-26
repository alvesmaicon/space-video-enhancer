"""Rotation, flipping and cropping.

Canonical order in the pipeline is *geometry first, crop last*: the crop rectangle
is expressed in the displayed (rotated/flipped) frame, exactly as the user drew it
on the preview, so preview and export agree.
"""

from __future__ import annotations

import cv2
import numpy as np

from space_video_enhancer.settings import CropSettings, GeometrySettings, NormalizedRect, RotationMode

_ROTATION_CODES = {
    RotationMode.CW_90: cv2.ROTATE_90_CLOCKWISE,
    RotationMode.ROTATE_180: cv2.ROTATE_180,
    RotationMode.CW_270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def apply_geometry(frame: np.ndarray, geometry: GeometrySettings) -> np.ndarray:
    transformed = frame
    rotation_code = _ROTATION_CODES.get(geometry.rotation)
    if rotation_code is not None:
        transformed = cv2.rotate(transformed, rotation_code)
    if geometry.flip_horizontal:
        transformed = cv2.flip(transformed, 1)
    if geometry.flip_vertical:
        transformed = cv2.flip(transformed, 0)
    return transformed


def apply_crop(frame: np.ndarray, crop: CropSettings) -> np.ndarray:
    height, width = frame.shape[:2]
    pixels = crop.pixel_rect(width, height)
    if pixels is None:
        return frame
    x, y, crop_width, crop_height = pixels
    return frame[y : y + crop_height, x : x + crop_width]


def region(frame: np.ndarray, rect: NormalizedRect) -> np.ndarray | None:
    """Slice ``rect`` out of ``frame`` (a view), or ``None`` when the frame is empty."""
    height, width = frame.shape[:2]
    pixels = rect.to_pixels(width, height)
    if pixels is None:
        return None
    x, y, rect_width, rect_height = pixels
    sliced = frame[y : y + rect_height, x : x + rect_width]
    return sliced if sliced.size > 0 else None


def fit_to_canvas(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale ``frame`` to fit ``width x height`` keeping its aspect ratio and pad the rest with black.

    Identity when the size already matches — the common case, so it costs nothing there.
    """
    frame_height, frame_width = frame.shape[:2]
    if (frame_width, frame_height) == (width, height):
        return frame
    scale = min(width / frame_width, height / frame_height)
    new_width = max(1, min(width, round(frame_width * scale)))
    new_height = max(1, min(height, round(frame_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
    canvas = np.zeros((height, width, frame.shape[2]) if frame.ndim == 3 else (height, width), dtype=frame.dtype)
    top = (height - new_height) // 2
    left = (width - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = scaled
    return canvas


def fit_to_aspect(frame: np.ndarray, aspect_ratio: float | None, tolerance: float = 0.01) -> np.ndarray:
    """Pad ``frame`` with black (never scale it) until its width/height is ``aspect_ratio``.

    The preview uses it to show the letterbox the export adds when a clip's output does not
    match the sequence canvas. Identity for ``None`` or when the aspect already matches.
    """
    if aspect_ratio is None or aspect_ratio <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame
    current = width / height
    if abs(current - aspect_ratio) <= tolerance * aspect_ratio:
        return frame
    if current < aspect_ratio:
        return fit_to_canvas(frame, max(width, round(height * aspect_ratio)), height)
    return fit_to_canvas(frame, width, max(height, round(width / aspect_ratio)))


def ensure_even_size(frame: np.ndarray) -> np.ndarray:
    """Trim at most one row/column so both dimensions are even (required by yuv420p encoders)."""
    height, width = frame.shape[:2]
    even_height = height - (height % 2)
    even_width = width - (width % 2)
    if even_height == height and even_width == width:
        return frame
    return frame[:even_height, :even_width]
