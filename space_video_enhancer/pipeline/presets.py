"""Tonal "image filter" presets (shadows, highlights, local contrast, colour balance, hue taming)."""

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


IMAGE_FILTER_PRESETS: tuple[ImageFilterPreset, ...] = (
    ImageFilterPreset(
        ORIGINAL_PRESET_ID,
        "Original",
        "Mantem apenas os ajustes manuais atuais, sem correcao tonal extra.",
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    ImageFilterPreset(
        "shadow_lift",
        "Sombras abertas",
        "Clareia areas escuras e segura os brancos para cenas com contraste alto.",
        0.55,
        0.20,
        0.10,
        0.08,
        0.0,
        0.03,
        0.0,
        0.0,
    ),
    ImageFilterPreset(
        "highlight_guard",
        "Realces suaves",
        "Comprime altas luzes para reduzir estouro em areas claras e janelas.",
        0.18,
        0.65,
        0.08,
        0.10,
        -0.02,
        -0.03,
        0.0,
        0.0,
    ),
    ImageFilterPreset(
        "dynamic_balance",
        "Faixa dinamica",
        "Equilibra sombras e realces ao mesmo tempo, bom para cenas muito duras.",
        0.48,
        0.48,
        0.16,
        0.14,
        0.0,
        0.04,
        0.0,
        0.0,
    ),
    ImageFilterPreset(
        "neutral_color",
        "Cor neutra",
        "Corrige dominantes de cor e entrega um balanco mais natural de branco e pele.",
        0.22,
        0.28,
        0.10,
        0.38,
        0.0,
        0.02,
        0.0,
        0.0,
    ),
    ImageFilterPreset(
        "smart_contrast",
        "Contraste inteligente",
        "Aumenta contraste local sem endurecer tanto as altas luzes.",
        0.20,
        0.30,
        0.38,
        0.12,
        0.02,
        0.05,
        0.0,
        0.0,
    ),
    ImageFilterPreset(
        "micro_contrast",
        "Microcontraste",
        "Realca textura fina e separacao entre planos sem abrir tanto as sombras.",
        0.14,
        0.24,
        0.52,
        0.10,
        0.01,
        0.04,
        0.0,
        0.0,
    ),
    ImageFilterPreset(
        "contrast_punch",
        "Contraste punch",
        "Entrega mais presenca e recorte nos detalhes, com um look mais energico.",
        0.10,
        0.34,
        0.62,
        0.08,
        0.0,
        0.06,
        0.0,
        0.0,
    ),
    ImageFilterPreset(
        "warm_tamer",
        "Atenuar quentes",
        "Reduz vermelhos, laranjas e amarelos mantendo verdes, azuis e neutros mais intactos.",
        0.18,
        0.24,
        0.16,
        0.10,
        -0.04,
        -0.02,
        0.48,
        0.0,
    ),
    ImageFilterPreset(
        "cool_tamer",
        "Atenuar frias",
        "Suaviza azuis e cianos mantendo quentes e tons de pele com mais consistencia.",
        0.18,
        0.22,
        0.14,
        0.10,
        0.04,
        -0.01,
        0.0,
        0.48,
    ),
)

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


def get_image_filter_preset(filter_id: str) -> ImageFilterPreset:
    return _PRESETS_BY_ID.get(filter_id, IMAGE_FILTER_PRESETS[0])


def _suppress_hues(working: np.ndarray, preset: ImageFilterPreset) -> np.ndarray:
    hsv_image = cv2.cvtColor((working * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv_image[:, :, 0]
    saturation = hsv_image[:, :, 1]

    if preset.warm_suppression > 0:
        warm_mask = np.where((hue <= 28) | (hue >= 166), 1.0, 0.0)
        warm_mask = np.maximum(warm_mask, np.where((hue >= 29) & (hue <= 42), 0.6, 0.0))
        saturation *= 1.0 - (warm_mask * preset.warm_suppression * HUE_SUPPRESSION_STRENGTH)

    if preset.cool_suppression > 0:
        cool_mask = np.where((hue >= 82) & (hue <= 128), 1.0, 0.0)
        cool_mask = np.maximum(cool_mask, np.where((hue >= 68) & (hue <= 81), 0.55, 0.0))
        saturation *= 1.0 - (cool_mask * preset.cool_suppression * HUE_SUPPRESSION_STRENGTH)

    hsv_image[:, :, 1] = np.clip(saturation, 0.0, 255.0)
    return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0


def apply_image_filter_preset(frame: np.ndarray, filter_id: str, intensity: float) -> np.ndarray:
    preset = get_image_filter_preset(filter_id)
    if preset.filter_id == ORIGINAL_PRESET_ID or intensity <= 0.0:
        return frame

    working = frame.astype(np.float32) / 255.0

    if preset.balance_strength > 0:
        channel_means = working.reshape(-1, 3).mean(axis=0)
        target_mean = float(np.mean(channel_means))
        gains = np.clip(target_mean / np.maximum(channel_means, 1e-4), BALANCE_GAIN_MIN, BALANCE_GAIN_MAX)
        balanced = np.clip(working * gains.reshape(1, 1, 3), 0.0, 1.0)
        working = cv2.addWeighted(working, 1.0 - preset.balance_strength, balanced, preset.balance_strength, 0.0)

    if abs(preset.warmth_shift) > 0.001:
        working[:, :, 2] *= 1.0 + (WARMTH_GAIN * preset.warmth_shift)
        working[:, :, 0] *= 1.0 - (WARMTH_GAIN * preset.warmth_shift)
        working = np.clip(working, 0.0, 1.0)

    if preset.warm_suppression > 0 or preset.cool_suppression > 0:
        working = _suppress_hues(working, preset)

    lab_image = cv2.cvtColor((working * 255.0).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab_image[:, :, 0] / 255.0

    if preset.shadow_lift > 0:
        shadow_mask = np.clip(1.0 - (l_channel * SHADOW_MASK_SLOPE), 0.0, 1.0)
        l_channel = np.clip(l_channel + (preset.shadow_lift * shadow_mask * SHADOW_LIFT_AMOUNT), 0.0, 1.0)

    if preset.highlight_rolloff > 0:
        highlight_mask = np.clip((l_channel - HIGHLIGHT_KNEE) / HIGHLIGHT_RANGE, 0.0, 1.0)
        l_channel = np.clip(
            l_channel - (preset.highlight_rolloff * highlight_mask * HIGHLIGHT_ROLLOFF_AMOUNT), 0.0, 1.0
        )

    if preset.local_contrast > 0:
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_BASE_CLIP + (preset.local_contrast * CLAHE_CLIP_GAIN), tileGridSize=(8, 8)
        )
        contrast_l = clahe.apply((l_channel * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
        blend = min(CLAHE_MAX_BLEND, preset.local_contrast)
        l_channel = cv2.addWeighted(l_channel, 1.0 - blend, contrast_l, blend, 0.0)

    lab_image[:, :, 0] = np.clip(l_channel * 255.0, 0.0, 255.0)
    filtered = cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2BGR)

    if abs(preset.saturation_boost) > 0.001:
        hsv_image = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_image[:, :, 1] = np.clip(
            hsv_image[:, :, 1] * (1.0 + (preset.saturation_boost * SATURATION_BOOST_GAIN)), 0.0, 255.0
        )
        filtered = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if intensity >= 0.999:
        return filtered
    return cv2.addWeighted(frame, 1.0 - intensity, filtered, intensity, 0.0)
