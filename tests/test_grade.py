from __future__ import annotations

import cv2
import numpy as np
import pytest

from space_video_enhancer.pipeline.grade import (
    apply_hue_bands,
    apply_white_balance,
    band_weights,
    build_grade_tables,
    neutral_gains,
    tint_gains,
    white_patch_gains,
)
from space_video_enhancer.pipeline.render import render_frame
from space_video_enhancer.settings import (
    HUE_BAND_NAMES,
    ColorGradeSettings,
    HueBandAdjust,
    NormalizedRect,
    RenderSettings,
)
from space_video_enhancer.telemetry.diff import flatten

YELLOW, GREEN, BLUE, RED = (0, 255, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255)


def with_band(name: str, **adjust: int) -> ColorGradeSettings:
    bands = list(ColorGradeSettings().bands)
    bands[HUE_BAND_NAMES.index(name)] = HueBandAdjust(**adjust)
    return ColorGradeSettings(bands=tuple(bands))


def patches(*colors: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((40, 40 * len(colors), 3), dtype=np.uint8)
    for index, color in enumerate(colors):
        frame[:, index * 40 : (index + 1) * 40] = color
    return frame


def patch_hsv(frame: np.ndarray, index: int) -> tuple[float, float, float]:
    hsv = cv2.cvtColor(frame[:, index * 40 : (index + 1) * 40], cv2.COLOR_BGR2HSV)
    return tuple(float(channel) for channel in cv2.mean(hsv)[:3])  # type: ignore[return-value]


def test_band_weights_are_a_partition_of_unity_anchored_at_the_centers() -> None:
    weights = band_weights(np.arange(0, 360, 1.0))
    assert weights.shape == (360, len(HUE_BAND_NAMES))
    assert np.allclose(weights.sum(axis=1), 1.0)
    green = HUE_BAND_NAMES.index("green")
    assert weights[120, green] == pytest.approx(1.0)
    assert weights[90, HUE_BAND_NAMES.index("yellow")] == pytest.approx(0.5)
    assert weights[90, green] == pytest.approx(0.5)
    assert weights[340, HUE_BAND_NAMES.index("magenta")] == pytest.approx(0.5)  # wraps around to red
    assert weights[340, HUE_BAND_NAMES.index("red")] == pytest.approx(0.5)


def test_neutral_settings_build_no_tables() -> None:
    assert ColorGradeSettings().is_neutral
    assert build_grade_tables(ColorGradeSettings()) is None
    frame = patches(YELLOW, BLUE)
    assert apply_hue_bands(frame, None) is frame and apply_white_balance(frame, None) is frame


def test_hue_shift_moves_only_the_targeted_band() -> None:
    frame = patches(YELLOW, BLUE)
    tables = build_grade_tables(with_band("yellow", hue_shift=30))
    shifted = apply_hue_bands(frame, tables)
    assert isinstance(shifted, np.ndarray)
    assert patch_hsv(frame, 0)[0] == pytest.approx(30.0)  # OpenCV hue = degrees / 2
    assert patch_hsv(shifted, 0)[0] == pytest.approx(45.0, abs=1.0)  # +30 degrees toward green
    assert patch_hsv(shifted, 1) == pytest.approx(patch_hsv(frame, 1), abs=1.0)


def test_saturation_and_luminance_gains_apply_per_band() -> None:
    frame = patches(GREEN, RED)
    desaturated = apply_hue_bands(frame, build_grade_tables(with_band("green", saturation=-100)))
    assert isinstance(desaturated, np.ndarray)
    assert patch_hsv(desaturated, 0)[1] < 2.0  # green is now gray
    assert patch_hsv(desaturated, 1)[1] == pytest.approx(255.0)  # red untouched
    darker = apply_hue_bands(frame, build_grade_tables(with_band("red", luminance=-50)))
    assert isinstance(darker, np.ndarray)
    assert patch_hsv(darker, 1)[2] == pytest.approx(127.5, abs=2.0)
    assert patch_hsv(darker, 0)[2] == pytest.approx(255.0)
    brighter = apply_hue_bands(frame, build_grade_tables(with_band("green", luminance=50)))
    assert isinstance(brighter, np.ndarray)
    assert patch_hsv(brighter, 0)[2] == pytest.approx(255.0)  # saturates instead of wrapping


def test_neutral_gains_balance_a_tinted_gray() -> None:
    tinted = np.zeros((30, 30, 3), dtype=np.uint8)
    tinted[:] = (110, 140, 128)  # greenish-yellow cast on a gray card
    gains = neutral_gains(tinted, None)
    assert gains[1] < 1.0 < gains[0]
    balanced = apply_white_balance(tinted, build_grade_tables(ColorGradeSettings(white_balance_gains=gains)))
    assert isinstance(balanced, np.ndarray)
    means = cv2.mean(balanced)[:3]
    assert max(means) - min(means) <= 1.5
    assert neutral_gains(np.zeros((8, 8, 3), dtype=np.uint8), None) == (1.0, 1.0, 1.0)  # black: nothing to infer


def test_white_patch_balances_by_the_highlights_when_there_is_no_gray() -> None:
    # A scene dominated by a saturated color (green) with a small warm-tinted highlight.
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:] = (20, 160, 20)  # green field: gray world would chase this
    frame[:6, :6] = (150, 190, 230)  # bright warm patch (the scene "white", reddish)
    rect = NormalizedRect(0.0, 0.0, 0.2, 0.2)  # aim the region at the highlight
    gains = white_patch_gains(frame, rect)
    assert gains[0] > 1.0 > gains[2]  # lift blue, pull red down: neutralizes the warm cast
    patch = np.array([[150.0, 190.0, 230.0]])
    balanced = patch * np.array(gains)
    assert balanced.max() - balanced.min() < patch.max() - patch.min()  # closer to neutral


def test_white_patch_is_neutral_on_black() -> None:
    assert white_patch_gains(np.zeros((8, 8, 3), dtype=np.uint8), None) == (1.0, 1.0, 1.0)


def test_tint_moves_green_against_magenta() -> None:
    gray = np.full((20, 20, 3), 128, dtype=np.uint8)
    magenta = apply_white_balance(gray, build_grade_tables(ColorGradeSettings(tint=80)))
    green = apply_white_balance(gray, build_grade_tables(ColorGradeSettings(tint=20)))
    assert isinstance(magenta, np.ndarray) and isinstance(green, np.ndarray)
    assert magenta[0, 0, 1] < 128 < magenta[0, 0, 2] and magenta[0, 0, 0] > 128
    assert green[0, 0, 1] > 128 > green[0, 0, 2]
    assert tint_gains(50) == (1.0, 1.0, 1.0)


def test_umat_path_matches_ndarray(textured_frame: np.ndarray) -> None:
    settings = ColorGradeSettings(
        tint=62, white_balance_gains=(1.1, 0.95, 1.02), bands=with_band("blue", hue_shift=-12, saturation=30).bands
    )
    tables = build_grade_tables(settings)
    cpu = apply_hue_bands(apply_white_balance(textured_frame, tables), tables)
    device = apply_hue_bands(apply_white_balance(cv2.UMat(textured_frame), tables), tables)
    assert isinstance(cpu, np.ndarray) and isinstance(device, cv2.UMat)
    difference = np.abs(device.get().astype(np.int16) - cpu.astype(np.int16))
    assert difference.mean() < 1.0 and np.percentile(difference, 99) <= 2


def test_render_frame_applies_the_grade(textured_frame: np.ndarray) -> None:
    neutral = render_frame(textured_frame, RenderSettings())
    graded = render_frame(textured_frame, RenderSettings(grade=with_band("green", saturation=-80)))
    assert graded.shape == neutral.shape and not np.array_equal(graded, neutral)


def test_settings_diff_flattens_the_bands() -> None:
    flat = flatten(RenderSettings(grade=with_band("cyan", hue_shift=5)))
    assert flat["grade.bands.4.hue_shift"] == 5 and flat["grade.tint"] == 50
    assert hash(RenderSettings(grade=with_band("cyan", hue_shift=5)))  # still hashable
