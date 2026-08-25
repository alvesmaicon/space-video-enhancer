"""Reference (pre-optimization) float32 implementation of the image presets.

Kept verbatim so ``test_presets.py`` can prove the LUT implementation stays within tolerance.
"""

from __future__ import annotations

import cv2
import numpy as np

from space_video_enhancer.pipeline.presets import ORIGINAL_PRESET_ID, get_image_filter_preset


def apply_image_filter_preset_reference(frame: np.ndarray, filter_id: str, intensity: float) -> np.ndarray:
    preset = get_image_filter_preset(filter_id)
    if preset.filter_id == ORIGINAL_PRESET_ID or intensity <= 0.0:
        return frame

    working = frame.astype(np.float32) / 255.0

    if preset.balance_strength > 0:
        channel_means = working.reshape(-1, 3).mean(axis=0)
        target_mean = float(np.mean(channel_means))
        gains = np.clip(target_mean / np.maximum(channel_means, 1e-4), 0.85, 1.18)
        balanced = np.clip(working * gains.reshape(1, 1, 3), 0.0, 1.0)
        working = cv2.addWeighted(working, 1.0 - preset.balance_strength, balanced, preset.balance_strength, 0.0)

    if abs(preset.warmth_shift) > 0.001:
        working[:, :, 2] *= 1.0 + (0.10 * preset.warmth_shift)
        working[:, :, 0] *= 1.0 - (0.10 * preset.warmth_shift)
        working = np.clip(working, 0.0, 1.0)

    if preset.warm_suppression > 0 or preset.cool_suppression > 0:
        hsv_image = cv2.cvtColor((working * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hue = hsv_image[:, :, 0]
        saturation = hsv_image[:, :, 1]
        if preset.warm_suppression > 0:
            warm_mask = np.where((hue <= 28) | (hue >= 166), 1.0, 0.0)
            warm_mask = np.maximum(warm_mask, np.where((hue >= 29) & (hue <= 42), 0.6, 0.0))
            saturation *= 1.0 - (warm_mask * preset.warm_suppression * 0.55)
        if preset.cool_suppression > 0:
            cool_mask = np.where((hue >= 82) & (hue <= 128), 1.0, 0.0)
            cool_mask = np.maximum(cool_mask, np.where((hue >= 68) & (hue <= 81), 0.55, 0.0))
            saturation *= 1.0 - (cool_mask * preset.cool_suppression * 0.55)
        hsv_image[:, :, 1] = np.clip(saturation, 0.0, 255.0)
        working = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    lab_image = cv2.cvtColor((working * 255.0).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab_image[:, :, 0] / 255.0

    if preset.shadow_lift > 0:
        shadow_mask = np.clip(1.0 - (l_channel * 1.45), 0.0, 1.0)
        l_channel = np.clip(l_channel + (preset.shadow_lift * shadow_mask * 0.26), 0.0, 1.0)

    if preset.highlight_rolloff > 0:
        highlight_mask = np.clip((l_channel - 0.58) / 0.42, 0.0, 1.0)
        l_channel = np.clip(l_channel - (preset.highlight_rolloff * highlight_mask * 0.22), 0.0, 1.0)

    if preset.local_contrast > 0:
        clahe = cv2.createCLAHE(clipLimit=1.4 + (preset.local_contrast * 2.2), tileGridSize=(8, 8))
        contrast_l = clahe.apply((l_channel * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
        blend = min(0.65, preset.local_contrast)
        l_channel = cv2.addWeighted(l_channel, 1.0 - blend, contrast_l, blend, 0.0)

    lab_image[:, :, 0] = np.clip(l_channel * 255.0, 0.0, 255.0)
    filtered = cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2BGR)

    if abs(preset.saturation_boost) > 0.001:
        hsv_image = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1.0 + (preset.saturation_boost * 0.35)), 0.0, 255.0)
        filtered = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if intensity >= 0.999:
        return filtered
    return cv2.addWeighted(frame, 1.0 - intensity, filtered, intensity, 0.0)
