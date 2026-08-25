from __future__ import annotations

import cv2
import numpy as np
import pytest


def make_textured_frame(width: int = 480, height: int = 270, seed: int = 42) -> np.ndarray:
    """Deterministic frame with gradient, noise and a few high-contrast shapes (good for feature tracking)."""
    rng = np.random.default_rng(seed)
    gradient = np.linspace(30, 220, width, dtype=np.float32)[None, :, None].repeat(height, 0).repeat(3, 2)
    gradient[..., 0] *= 0.8
    gradient[..., 2] *= 1.1
    frame = np.clip(gradient + rng.normal(0, 10, (height, width, 3)), 0, 255).astype(np.uint8)
    cv2.rectangle(frame, (60, 40), (180, 140), (30, 200, 60), -1)
    cv2.circle(frame, (340, 170), 55, (200, 40, 40), -1)
    cv2.putText(frame, "GOLD", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
    return frame


def make_quadrant_frame(width: int = 640, height: int = 360) -> np.ndarray:
    """Four solid quadrants (BGR): TL red, TR green, BL blue, BR yellow."""
    frame = np.zeros((height, width, 3), np.uint8)
    frame[: height // 2, : width // 2] = (0, 0, 255)
    frame[: height // 2, width // 2 :] = (0, 255, 0)
    frame[height // 2 :, : width // 2] = (255, 0, 0)
    frame[height // 2 :, width // 2 :] = (0, 255, 255)
    return frame


@pytest.fixture
def textured_frame() -> np.ndarray:
    return make_textured_frame()


@pytest.fixture
def quadrant_frame() -> np.ndarray:
    return make_quadrant_frame()
