"""numpy <-> Qt image conversion."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage


def frame_to_qimage(frame: np.ndarray) -> QImage:
    """BGR uint8 frame -> detached ``QImage`` (safe to keep after ``frame`` is freed)."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_frame.shape
    return QImage(rgb_frame.data, width, height, width * channels, QImage.Format.Format_RGB888).copy()


def thumbnail_qimage(frame: np.ndarray, size: QSize) -> QImage:
    return frame_to_qimage(frame).scaled(
        size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )
