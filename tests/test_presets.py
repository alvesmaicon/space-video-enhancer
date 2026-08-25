"""The LUT-based presets must reproduce the reference float32 implementation.

The only accepted deviation: colour-balance gains now come from the exact channel mean
(``cv2.mean``) instead of numpy's float32 mean, which can flip a handful of LUT entries
by one level on frames where the two means round differently.
"""

from __future__ import annotations

import numpy as np
import pytest

from space_video_enhancer.pipeline.presets import IMAGE_FILTER_PRESETS, apply_image_filter_preset
from tests.conftest import make_textured_frame
from tests.reference_presets import apply_image_filter_preset_reference

PRESET_IDS = [preset.filter_id for preset in IMAGE_FILTER_PRESETS if preset.filter_id != "original"]
MAX_ABS_DIFF = 6
MAX_CHANGED_FRACTION = 0.01


def _colorful_frame() -> np.ndarray:
    """Textured frame plus saturated warm/cool patches so hue taming has something to bite."""
    frame = make_textured_frame().copy()
    frame[10:60, 300:380] = (20, 40, 230)  # warm (red-orange)
    frame[200:260, 380:470] = (230, 120, 20)  # cool (blue)
    frame[150:200, 20:80] = (30, 220, 220)  # yellow
    return frame


@pytest.mark.parametrize("preset_id", PRESET_IDS)
@pytest.mark.parametrize("intensity", [1.0, 0.6])
def test_lut_presets_match_reference(preset_id: str, intensity: float) -> None:
    frame = _colorful_frame()
    fast = apply_image_filter_preset(frame, preset_id, intensity)
    reference = apply_image_filter_preset_reference(frame, preset_id, intensity)
    diff = np.abs(fast.astype(np.int16) - reference.astype(np.int16))
    changed = float((diff > 0).mean())
    assert int(diff.max()) <= MAX_ABS_DIFF, f"max diff {diff.max()} on {changed * 100:.3f}% of values"
    assert changed <= MAX_CHANGED_FRACTION, f"{changed * 100:.3f}% of values differ"


def test_lut_presets_are_deterministic() -> None:
    frame = _colorful_frame()
    first = apply_image_filter_preset(frame, "warm_tamer", 1.0)
    second = apply_image_filter_preset(frame, "warm_tamer", 1.0)
    assert np.array_equal(first, second)
