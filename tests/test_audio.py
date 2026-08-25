"""Audio peaks for the waveform track."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from space_video_enhancer.audio import SAMPLE_RATE, AudioPeaks, load_audio_peaks, peaks_from_samples

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not on PATH")


def test_peaks_from_samples_buckets_min_and_max_and_pads_the_tail() -> None:
    samples = np.array([0, 100, -200, 300, 400], dtype=np.int16)
    minimums, maximums = peaks_from_samples(samples, 2)
    assert minimums.dtype == np.float32 and maximums.dtype == np.float32
    assert minimums.tolist() == pytest.approx([0.0, -200 / 32768, 400 / 32768])
    assert maximums.tolist() == pytest.approx([100 / 32768, 300 / 32768, 400 / 32768])
    empty_min, empty_max = peaks_from_samples(np.zeros(0, dtype=np.int16), 10)
    assert empty_min.size == 0 and empty_max.size == 0
    peaks = AudioPeaks(minimums, maximums, 2.0)
    assert peaks.bucket_count == 3 and peaks.duration_seconds == pytest.approx(1.5)


def tone_file(directory: Path, seconds: float = 1.0) -> Path:
    path = directory / "tone.wav"
    subprocess.run(
        [
            FFMPEG or "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate={SAMPLE_RATE}:duration={seconds}",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
    )
    return path


@requires_ffmpeg
def test_load_audio_peaks_from_a_tone_and_none_without_audio(tmp_path: Path, ui_video: Path) -> None:
    assert FFMPEG is not None
    peaks = load_audio_peaks(FFMPEG, tone_file(tmp_path), 100.0)
    assert peaks is not None
    assert peaks.bucket_count == 100 and peaks.buckets_per_second == pytest.approx(100.0)
    assert peaks.duration_seconds == pytest.approx(1.0)
    assert 0.1 < peaks.maximums.max() < 0.15 and -0.15 < peaks.minimums.min() < -0.1  # ffmpeg's sine is 1/8 scale
    assert np.all(peaks.maximums >= peaks.minimums)
    assert load_audio_peaks(FFMPEG, ui_video, 100.0) is None, "VideoWriter output has no audio stream"
    assert load_audio_peaks("definitely-not-ffmpeg", tmp_path / "tone.wav", 100.0) is None
