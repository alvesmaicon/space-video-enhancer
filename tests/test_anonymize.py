"""Blur/pixelate anonymising: the region is covered, the rest is untouched, and it is a no-op when inactive."""

from __future__ import annotations

import numpy as np

from space_video_enhancer.pipeline.anonymize import apply_anonymize, detect_faces
from space_video_enhancer.settings import AnonymizeMode, AnonymizeSettings, NormalizedRect


def _noisy_frame(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (120, 160, 3), dtype=np.uint8)


def test_region_pixelate_covers_the_region_and_leaves_the_rest_untouched() -> None:
    frame = _noisy_frame()
    region = NormalizedRect(0.25, 0.25, 0.5, 0.5)
    settings = AnonymizeSettings(
        enabled=True, mode=AnonymizeMode.PIXELATE, strength=80, detect_faces=False, region=region
    )

    out = apply_anonymize(frame, settings)

    assert out.shape == frame.shape and out is not frame
    assert np.array_equal(out[:20, :20], frame[:20, :20]), "a corner outside the region is untouched"
    x0, y0, x1, y1 = int(0.25 * 160), int(0.25 * 120), int(0.75 * 160), int(0.75 * 120)
    assert out[y0:y1, x0:x1].var() < frame[y0:y1, x0:x1].var(), "the region lost detail"


def test_inactive_settings_return_the_same_frame() -> None:
    frame = _noisy_frame()
    assert apply_anonymize(frame, AnonymizeSettings(enabled=False)) is frame
    assert apply_anonymize(frame, AnonymizeSettings(enabled=True, detect_faces=False, region=None)) is frame
    assert apply_anonymize(frame, AnonymizeSettings(enabled=True, strength=0, detect_faces=True)) is frame


def test_auto_faces_on_a_frame_without_faces_changes_nothing() -> None:
    frame = _noisy_frame()
    assert detect_faces(frame) == []
    out = apply_anonymize(frame, AnonymizeSettings(enabled=True, detect_faces=True, strength=50))
    assert np.array_equal(out, frame), "no faces found -> nothing covered"


def test_pixelate_and_blur_produce_different_pixels() -> None:
    frame = _noisy_frame(2)
    common = {"enabled": True, "strength": 60, "detect_faces": False, "region": NormalizedRect(0.2, 0.2, 0.6, 0.6)}
    pixelated = apply_anonymize(frame, AnonymizeSettings(mode=AnonymizeMode.PIXELATE, **common))
    blurred = apply_anonymize(frame, AnonymizeSettings(mode=AnonymizeMode.BLUR, **common))
    assert not np.array_equal(pixelated, blurred)
