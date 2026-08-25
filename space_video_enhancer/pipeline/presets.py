"""Tonal "image filter" presets (shadows, highlights, local contrast, colour balance, hue taming).

Every per-pixel step of a preset is a function of a single channel value, so the whole
preset compiles into a few 256-entry lookup tables applied on uint8 data (``cv2.LUT``),
plus one LAB and one or two HSV round trips. The tables replay the reference float32
arithmetic, so results match the previous implementation almost bit for bit while running
several times faster (see ``tests/test_presets.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ORIGINAL_PRESET_ID = "original"


@dataclass(frozen=True)
class ImageFilterPreset:
    filter_id: str
    name: str
    description: str
    shadow_lift: float
    highlight_rolloff: float
    local_contrast: float
    balance_strength: float
    warmth_shift: float
    saturation_boost: float
    warm_suppression: float
    cool_suppression: float


# fmt: off
IMAGE_FILTER_PRESETS: tuple[ImageFilterPreset, ...] = (
    ImageFilterPreset(
        ORIGINAL_PRESET_ID, "Original",
        "Mantém apenas os ajustes manuais atuais, sem correção tonal extra.",
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ),
    ImageFilterPreset(
        "shadow_lift", "Sombras abertas",
        "Clareia áreas escuras e segura os brancos para cenas com contraste alto.",
        0.55, 0.20, 0.10, 0.08, 0.0, 0.03, 0.0, 0.0,
    ),
    ImageFilterPreset(
        "highlight_guard", "Realces suaves",
        "Comprime altas luzes para reduzir estouro em áreas claras e janelas.",
        0.18, 0.65, 0.08, 0.10, -0.02, -0.03, 0.0, 0.0,
    ),
    ImageFilterPreset(
        "dynamic_balance", "Faixa dinâmica",
        "Equilibra sombras e realces ao mesmo tempo, bom para cenas muito duras.",
        0.48, 0.48, 0.16, 0.14, 0.0, 0.04, 0.0, 0.0,
    ),
    ImageFilterPreset(
        "neutral_color", "Cor neutra",
        "Corrige dominantes de cor e entrega um balanço mais natural de branco e pele.",
        0.22, 0.28, 0.10, 0.38, 0.0, 0.02, 0.0, 0.0,
    ),
    ImageFilterPreset(
        "smart_contrast", "Contraste inteligente",
        "Aumenta contraste local sem endurecer tanto as altas luzes.",
        0.20, 0.30, 0.38, 0.12, 0.02, 0.05, 0.0, 0.0,
    ),
    ImageFilterPreset(
        "micro_contrast", "Microcontraste",
        "Realça textura fina e separação entre planos sem abrir tanto as sombras.",
        0.14, 0.24, 0.52, 0.10, 0.01, 0.04, 0.0, 0.0,
    ),
    ImageFilterPreset(
        "contrast_punch", "Contraste punch",
        "Entrega mais presença e recorte nos detalhes, com um look mais enérgico.",
        0.10, 0.34, 0.62, 0.08, 0.0, 0.06, 0.0, 0.0,
    ),
    ImageFilterPreset(
        "warm_tamer", "Atenuar quentes",
        "Reduz vermelhos, laranjas e amarelos mantendo verdes, azuis e neutros mais intactos.",
        0.18, 0.24, 0.16, 0.10, -0.04, -0.02, 0.48, 0.0,
    ),
    ImageFilterPreset(
        "cool_tamer", "Atenuar frias",
        "Suaviza azuis e cianos mantendo quentes e tons de pele com mais consistência.",
        0.18, 0.22, 0.14, 0.10, 0.04, -0.01, 0.0, 0.48,
    ),
)
# fmt: on

_PRESETS_BY_ID = {preset.filter_id: preset for preset in IMAGE_FILTER_PRESETS}

BALANCE_GAIN_MIN = 0.85
BALANCE_GAIN_MAX = 1.18
WARMTH_GAIN = 0.10
HUE_SUPPRESSION_STRENGTH = 0.55
SHADOW_MASK_SLOPE = 1.45
SHADOW_LIFT_AMOUNT = 0.26
HIGHLIGHT_KNEE = 0.58
HIGHLIGHT_RANGE = 0.42
HIGHLIGHT_ROLLOFF_AMOUNT = 0.22
CLAHE_BASE_CLIP = 1.4
CLAHE_CLIP_GAIN = 2.2
CLAHE_MAX_BLEND = 0.65
SATURATION_BOOST_GAIN = 0.35
EPSILON = 0.001

_LEVELS32 = np.arange(256, dtype=np.float32) / 255.0  # same dtype path as frame.astype(float32) / 255.0
_HUES = np.arange(256, dtype=np.float64)  # OpenCV 8-bit hue spans 0..179; padded to 256 for cv2.LUT

# The tables below deliberately run the reference float32 arithmetic on a 256-level vector:
# every step is elementwise per channel, so the LUT reproduces the per-pixel result (including
# float32 rounding) while the image itself stays uint8.


def get_image_filter_preset(filter_id: str) -> ImageFilterPreset:
    return _PRESETS_BY_ID.get(filter_id, IMAGE_FILTER_PRESETS[0])


def _to_uint8(unit_values: np.ndarray) -> np.ndarray:
    """``(x * 255).astype(uint8)`` exactly as the float pipeline quantized (truncation)."""
    return (unit_values * 255.0).astype(np.uint8)


def channel_gain_lut(preset: ImageFilterPreset, channel_means: np.ndarray | None) -> np.ndarray | None:
    """``(256, 1, 3)`` uint8 LUT applying colour balance and warmth per BGR channel; ``None`` when neutral."""
    balance = preset.balance_strength > 0 and channel_means is not None
    warmth = abs(preset.warmth_shift) > EPSILON
    if not balance and not warmth:
        return None

    gains = None
    if balance:
        assert channel_means is not None
        means32 = channel_means.astype(np.float32)
        target_mean = float(np.mean(means32))
        gains = np.clip(target_mean / np.maximum(means32, 1e-4), BALANCE_GAIN_MIN, BALANCE_GAIN_MAX)
    warmth_gain = {0: 1.0 - (WARMTH_GAIN * preset.warmth_shift), 2: 1.0 + (WARMTH_GAIN * preset.warmth_shift)}

    table = np.empty((256, 1, 3), dtype=np.uint8)
    for channel in range(3):
        values = _LEVELS32
        if gains is not None:
            balanced = np.clip(values * gains[channel], 0.0, 1.0)
            blended = cv2.addWeighted(values, 1.0 - preset.balance_strength, balanced, preset.balance_strength, 0.0)
            values = np.asarray(blended, dtype=np.float32).reshape(-1)
        if warmth:
            values = np.clip(values * warmth_gain.get(channel, 1.0), 0.0, 1.0)
        table[:, 0, channel] = _to_uint8(values)
    return table


def hue_saturation_multiplier_lut(preset: ImageFilterPreset) -> np.ndarray | None:
    """``(256,)`` float64 saturation multiplier indexed by hue; ``None`` when the preset tames no hues."""
    if preset.warm_suppression <= 0 and preset.cool_suppression <= 0:
        return None
    hue = _HUES
    multiplier = np.ones(256, dtype=np.float64)
    if preset.warm_suppression > 0:
        warm_mask = np.where((hue <= 28) | (hue >= 166), 1.0, 0.0)
        warm_mask = np.maximum(warm_mask, np.where((hue >= 29) & (hue <= 42), 0.6, 0.0))
        multiplier *= 1.0 - (warm_mask * preset.warm_suppression * HUE_SUPPRESSION_STRENGTH)
    if preset.cool_suppression > 0:
        cool_mask = np.where((hue >= 82) & (hue <= 128), 1.0, 0.0)
        cool_mask = np.maximum(cool_mask, np.where((hue >= 68) & (hue <= 81), 0.55, 0.0))
        multiplier *= 1.0 - (cool_mask * preset.cool_suppression * HUE_SUPPRESSION_STRENGTH)
    return multiplier


def luma_unit_lut(preset: ImageFilterPreset) -> np.ndarray | None:
    """``(256,)`` float32 LUT: LAB L (0..255) -> shaped L in [0, 1] (shadow lift + highlight roll-off)."""
    if preset.shadow_lift <= 0 and preset.highlight_rolloff <= 0:
        return None
    luma = _LEVELS32
    if preset.shadow_lift > 0:
        shadow_mask = np.clip(1.0 - (luma * SHADOW_MASK_SLOPE), 0.0, 1.0)
        luma = np.clip(luma + (preset.shadow_lift * shadow_mask * SHADOW_LIFT_AMOUNT), 0.0, 1.0)
    if preset.highlight_rolloff > 0:
        highlight_mask = np.clip((luma - HIGHLIGHT_KNEE) / HIGHLIGHT_RANGE, 0.0, 1.0)
        luma = np.clip(luma - (preset.highlight_rolloff * highlight_mask * HIGHLIGHT_ROLLOFF_AMOUNT), 0.0, 1.0)
    return luma


def _saturation_gain_lut(gain: float) -> np.ndarray:
    return np.clip(np.arange(256, dtype=np.float32) * gain, 0.0, 255.0).astype(np.uint8)


def _tame_hues(bgr: np.ndarray, hue_lut: np.ndarray) -> np.ndarray:
    hue, saturation, value = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    multiplier = cv2.LUT(hue, hue_lut)  # float64 map, like the reference mask arithmetic
    scaled = (saturation.astype(np.float64) * multiplier).astype(np.float32)
    saturation = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([hue, saturation, value]), cv2.COLOR_HSV2BGR)


def _shape_luma(bgr: np.ndarray, preset: ImageFilterPreset, unit_lut: np.ndarray | None) -> np.ndarray:
    luma, a_channel, b_channel = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    luma_unit = cv2.LUT(luma, unit_lut) if unit_lut is not None else luma.astype(np.float32) / 255.0
    if preset.local_contrast > 0:
        clip_limit = CLAHE_BASE_CLIP + (preset.local_contrast * CLAHE_CLIP_GAIN)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        contrast = clahe.apply(_to_uint8(luma_unit)).astype(np.float32) / 255.0
        blend = min(CLAHE_MAX_BLEND, preset.local_contrast)
        luma_unit = cv2.addWeighted(luma_unit, 1.0 - blend, contrast, blend, 0.0)
    luma = np.clip(luma_unit * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([luma, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def _boost_saturation(bgr: np.ndarray, gain: float) -> np.ndarray:
    hue, saturation, value = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    saturation = cv2.LUT(saturation, _saturation_gain_lut(gain))
    return cv2.cvtColor(cv2.merge([hue, saturation, value]), cv2.COLOR_HSV2BGR)


def apply_image_filter_preset(frame: np.ndarray, filter_id: str, intensity: float) -> np.ndarray:
    preset = get_image_filter_preset(filter_id)
    if preset.filter_id == ORIGINAL_PRESET_ID or intensity <= 0.0:
        return frame

    filtered = frame
    channel_means = None
    if preset.balance_strength > 0:
        channel_means = np.asarray(cv2.mean(frame)[:3], dtype=np.float64) / 255.0
    gain_lut = channel_gain_lut(preset, channel_means)
    if gain_lut is not None:
        filtered = cv2.LUT(filtered, gain_lut)

    hue_lut = hue_saturation_multiplier_lut(preset)
    if hue_lut is not None:
        filtered = _tame_hues(filtered, hue_lut)

    unit_lut = luma_unit_lut(preset)
    if unit_lut is not None or preset.local_contrast > 0:
        filtered = _shape_luma(filtered, preset, unit_lut)

    if abs(preset.saturation_boost) > EPSILON:
        filtered = _boost_saturation(filtered, 1.0 + (preset.saturation_boost * SATURATION_BOOST_GAIN))

    if intensity >= 0.999:
        return filtered
    return cv2.addWeighted(frame, 1.0 - intensity, filtered, intensity, 0.0)
