"""Audio decoding through ffmpeg: peaks for the timeline waveform and PCM for the preview's sound.

Both go through :func:`decode_pcm` (mono ``s16le`` at a chosen rate). The waveform only needs
8 kHz; the preview plays 22.05 kHz and can ask ffmpeg to stretch the tempo (``atempo``, which
keeps the pitch) so the sound follows a slowed-down playback.
"""

from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from space_video_enhancer.telemetry.log import get_logger, log_event

LOG = get_logger("audio")
SAMPLE_RATE = 8000
"""Enough for peaks: the waveform never shows anything finer than a few samples per pixel."""
PREVIEW_SAMPLE_RATE = 22050
"""Preview playback: mono, fine for speech and most music, ~160 MB per hour of video."""
DECODE_TIMEOUT_SECONDS = 180
ATEMPO_RANGE = (0.5, 2.0)
"""One ``atempo`` filter covers this range; slower or faster tempos chain several."""


@dataclass(frozen=True)
class AudioPeaks:
    minimums: np.ndarray
    """Per-bucket minimum, ``float32`` in ``[-1, 1]``."""
    maximums: np.ndarray
    buckets_per_second: float

    @property
    def bucket_count(self) -> int:
        return int(self.minimums.size)

    @property
    def duration_seconds(self) -> float:
        return self.bucket_count / self.buckets_per_second if self.buckets_per_second > 0 else 0.0


def peaks_from_samples(samples: np.ndarray, samples_per_bucket: int) -> tuple[np.ndarray, np.ndarray]:
    """``(minimums, maximums)`` of ``int16`` samples per bucket, normalized; the tail bucket repeats its last sample."""
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    samples_per_bucket = max(1, samples_per_bucket)
    count = math.ceil(samples.size / samples_per_bucket)
    padded = np.empty(count * samples_per_bucket, dtype=np.int16)
    padded[: samples.size] = samples
    padded[samples.size :] = samples[-1]
    buckets = padded.reshape(count, samples_per_bucket)
    scale = np.float32(1.0 / 32768.0)
    return buckets.min(axis=1).astype(np.float32) * scale, buckets.max(axis=1).astype(np.float32) * scale


def tempo_filter(tempo: float) -> str:
    """ffmpeg ``atempo`` chain that plays the audio ``tempo`` times faster (0.5 = half speed, pitch kept)."""
    if tempo <= 0:
        raise ValueError("tempo must be positive")
    if abs(tempo - 1.0) < 1e-9:
        return ""
    low, high = ATEMPO_RANGE
    filters = []
    remaining = tempo
    while remaining < low - 1e-9:
        filters.append(f"atempo={low}")
        remaining /= low
    while remaining > high + 1e-9:
        filters.append(f"atempo={high}")
        remaining /= high
    filters.append(f"atempo={remaining:.6g}")
    return ",".join(filters)


def decode_pcm(ffmpeg_path: str, source_path: Path, sample_rate: int, *, tempo: float = 1.0) -> np.ndarray | None:
    """Mono ``int16`` samples of the first audio stream; ``None`` when there is no audio or ffmpeg fails."""
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(source_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
    ]
    filters = tempo_filter(tempo)
    if filters:
        command += ["-af", filters]
    command += ["-f", "s16le", "-"]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=DECODE_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as error:
        log_event(LOG, "audio.decode_failed", level=logging.WARNING, name=source_path.name, error=str(error))
        return None
    if result.returncode != 0 or len(result.stdout) < 2:
        message = result.stderr.decode("utf-8", "replace").strip().splitlines()
        log_event(
            LOG,
            "audio.no_stream",
            level=logging.DEBUG,
            name=source_path.name,
            returncode=result.returncode,
            stderr=message[0][:200] if message else "",
        )
        return None
    return np.frombuffer(result.stdout[: len(result.stdout) // 2 * 2], dtype="<i2")


def load_preview_audio(ffmpeg_path: str, source_path: Path, tempo: float = 1.0) -> np.ndarray | None:
    """PCM for the preview player, already stretched to ``tempo`` (source second ``t`` sits at ``t / tempo``)."""
    return decode_pcm(ffmpeg_path, source_path, PREVIEW_SAMPLE_RATE, tempo=tempo)


def load_audio_peaks(ffmpeg_path: str, source_path: Path, buckets_per_second: float) -> AudioPeaks | None:
    """Peaks of the first audio stream of ``source_path``; ``None`` when it has no audio or ffmpeg fails."""
    samples = decode_pcm(ffmpeg_path, source_path, SAMPLE_RATE)
    if samples is None:
        return None
    samples_per_bucket = max(1, round(SAMPLE_RATE / max(buckets_per_second, 1e-6)))
    minimums, maximums = peaks_from_samples(samples, samples_per_bucket)
    return AudioPeaks(minimums, maximums, SAMPLE_RATE / samples_per_bucket)
