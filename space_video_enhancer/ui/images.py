"""numpy <-> Qt image conversion."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtWidgets import QWidget


def device_pixel_ratio(widget: QWidget) -> float:
    """The display scale for ``widget``, read from its screen.

    Not ``widget.devicePixelRatioF()``: that getter, inherited from ``QPaintDevice``, aborts the process
    when called on the ``QMainWindow`` itself under PySide6 6.10 (``QThread``-still-running fatal during a
    spurious wrapper teardown). The widget's ``QScreen`` gives the same number and is safe.
    """
    screen = widget.screen() or QGuiApplication.primaryScreen()
    return float(screen.devicePixelRatio()) if screen is not None else 1.0


def frame_to_qimage(frame: np.ndarray) -> QImage:
    """BGR uint8 frame -> detached ``QImage`` (safe to keep after ``frame`` is freed)."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_frame.shape
    return QImage(rgb_frame.data, width, height, width * channels, QImage.Format.Format_RGB888).copy()


def thumbnail_qimage(frame: np.ndarray, size: QSize) -> QImage:
    return frame_to_qimage(frame).scaled(
        size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )
