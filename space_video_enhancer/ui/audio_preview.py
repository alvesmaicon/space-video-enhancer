"""Sound for the preview: PCM decoded per file (and per speed) on a thread, pushed into a ``QAudioSink``.

Playback keeps the wall clock as its master: the window moves the playhead to the frame the
elapsed time calls for (dropping video frames when the render is slow) and the sound plays
continuously from PCM that ffmpeg already stretched to the chosen speed, so the two stay in
sync by construction. The feeder walks the sequence's kept runs (:class:`AudioPiece` per run),
so cuts and clip boundaries are joined ahead of time instead of flushing when the playhead
jumps; only a seek by the user restarts it.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

from space_video_enhancer.audio import PREVIEW_SAMPLE_RATE, load_preview_audio
from space_video_enhancer.telemetry.log import get_logger, log_event

LOG = get_logger("audio")
SAMPLE_RATE = PREVIEW_SAMPLE_RATE
BYTES_PER_SAMPLE = 2
BUFFER_SECONDS = 0.3
"""How far ahead of the speaker the feeder keeps the sink filled."""
FEED_INTERVAL_MS = 40
MIN_WRITE_SAMPLES = 512
"""Below this many free samples the feeder waits for the next tick instead of writing crumbs."""
MAX_PREVIEW_AUDIO_SECONDS = 1800
"""Longer files are played without sound: their PCM would not fit comfortably in memory."""
DISABLE_ENV = "SVE_PREVIEW_AUDIO"
"""Set to ``0`` to run without sound (tests, machines without an output device)."""

PcmKey = tuple[Path, float]


@dataclass(frozen=True)
class AudioPiece:
    """What plays under one run of frames: a slice of one file (source seconds) or silence (``source=None``)."""

    source: Path | None
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


class AudioFeeder:
    """Turns pieces into consecutive ``int16`` chunks read from the stretched PCM of each file.

    Source time ``t`` of a file stretched to ``tempo`` sits at ``t / tempo`` seconds; a piece whose
    PCM is missing (file without audio, or not decoded yet) yields silence of the same length.
    """

    def __init__(
        self,
        pieces: Sequence[AudioPiece],
        pcm: Mapping[PcmKey, np.ndarray | None],
        tempo: float,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._pieces = list(pieces)
        self._pcm = pcm
        self._tempo = tempo if tempo > 0 else 1.0
        self._rate = sample_rate
        self._index = 0
        self._cursor = 0
        self.samples_written = 0

    @property
    def finished(self) -> bool:
        return self._index >= len(self._pieces)

    @property
    def total_samples(self) -> int:
        return sum(end - start for start, end in (self._bounds(piece) for piece in self._pieces))

    def _bounds(self, piece: AudioPiece) -> tuple[int, int]:
        start = round(piece.start_seconds / self._tempo * self._rate)
        end = round(piece.end_seconds / self._tempo * self._rate)
        return start, max(start, end)

    def next_chunk(self, max_samples: int) -> np.ndarray | None:
        """Up to ``max_samples`` samples (never crossing a piece boundary); ``None`` once everything was handed out."""
        while not self.finished:
            piece = self._pieces[self._index]
            start, end = self._bounds(piece)
            remaining = end - start - self._cursor
            if remaining <= 0:
                self._index += 1
                self._cursor = 0
                continue
            count = max(0, min(max_samples, remaining))
            if count == 0:
                return np.zeros(0, dtype=np.int16)
            chunk = np.zeros(count, dtype=np.int16)
            samples = self._pcm.get((piece.source, self._tempo)) if piece.source is not None else None
            if samples is not None:
                low = start + self._cursor
                high = min(low + count, int(samples.size))
                if high > low:
                    chunk[: high - low] = samples[low:high]
            self._cursor += count
            self.samples_written += count
            return chunk
        return None


class AudioSinkLike(Protocol):
    """The slice of ``QAudioSink`` the player uses (a fake stands in for it in tests)."""

    def start(self) -> Any: ...

    def stop(self) -> None: ...

    def bytesFree(self) -> int: ...

    def setBufferSize(self, size: int) -> None: ...

    def setVolume(self, volume: float) -> None: ...


def preview_audio_format(sample_rate: int = SAMPLE_RATE) -> QAudioFormat:
    audio_format = QAudioFormat()
    audio_format.setSampleRate(sample_rate)
    audio_format.setChannelCount(1)
    audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    return audio_format


def default_sink_factory() -> AudioSinkLike | None:
    """A ``QAudioSink`` on the default output, or ``None`` when disabled, absent or incompatible."""
    if os.environ.get(DISABLE_ENV, "1") == "0":
        log_event(LOG, "audio.preview_disabled", reason="env")
        return None
    device = QMediaDevices.defaultAudioOutput()
    audio_format = preview_audio_format()
    if device.isNull() or not device.isFormatSupported(audio_format):
        log_event(
            LOG,
            "audio.preview_disabled",
            reason="no_device" if device.isNull() else "format",
            device=device.description(),
        )
        return None
    log_event(LOG, "audio.preview_device", device=device.description(), sample_rate=SAMPLE_RATE)
    return QAudioSink(device, audio_format)


class PreviewAudioPlayer(QObject):
    """Owns the sink and the decoded PCM; :meth:`start` plays a list of pieces from the beginning."""

    def __init__(
        self, sink_factory: Callable[[], AudioSinkLike | None] | None = None, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._factory = sink_factory or default_sink_factory
        self._sink: AudioSinkLike | None = None
        self._device: Any = None
        self._probed = False
        self._pcm: dict[PcmKey, np.ndarray | None] = {}
        self._feeder: AudioFeeder | None = None
        self._muted = False
        self._timer = QTimer(self)
        self._timer.setInterval(FEED_INTERVAL_MS)
        self._timer.timeout.connect(self._feed)

    # --- device ----------------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """Whether there is a sink to play into (probed once, on first use)."""
        if not self._probed:
            self._probed = True
            self._sink = self._factory()
            if self._sink is not None:
                self._sink.setBufferSize(int(SAMPLE_RATE * BYTES_PER_SAMPLE * BUFFER_SECONDS))
        return self._sink is not None

    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        """Muting stops the sound at once; the window restarts it when unmuted during playback."""
        self._muted = muted
        if muted:
            self.stop()

    # --- pcm -------------------------------------------------------------------------------
    def set_pcm(self, path: Path, tempo: float, samples: np.ndarray | None) -> None:
        """Decoded (and stretched) samples of ``path``; ``None`` records that the file has no audio."""
        self._pcm[(path, tempo)] = samples

    def has_pcm(self, path: Path, tempo: float) -> bool:
        return (path, tempo) in self._pcm

    def keep_only(self, paths: Sequence[Path]) -> None:
        wanted = set(paths)
        for key in [key for key in self._pcm if key[0] not in wanted]:
            del self._pcm[key]

    def pcm_bytes(self) -> int:
        return sum(int(samples.nbytes) for samples in self._pcm.values() if samples is not None)

    # --- playback --------------------------------------------------------------------------
    @property
    def playing(self) -> bool:
        return self._feeder is not None

    @property
    def samples_written(self) -> int:
        return self._feeder.samples_written if self._feeder is not None else 0

    def start(self, pieces: Sequence[AudioPiece], tempo: float) -> bool:
        """Play ``pieces`` back to back from their start; ``False`` when muted or without a device."""
        self.stop()
        if self._muted or not self.available or self._sink is None:
            return False
        self._feeder = AudioFeeder(pieces, self._pcm, tempo)
        self._device = self._sink.start()
        if self._device is None:
            self._feeder = None
            return False
        self._feed()
        self._timer.start()
        return True

    def stop(self) -> None:
        self._timer.stop()
        if self._feeder is not None and self._sink is not None:
            self._sink.stop()
        self._feeder = None
        self._device = None

    @Slot()
    def _feed(self) -> None:
        """Keep the sink's buffer full; each write is a run of samples that never crosses a piece boundary."""
        feeder, sink, device = self._feeder, self._sink, self._device
        if feeder is None or sink is None or device is None:
            return
        while True:
            free_samples = sink.bytesFree() // BYTES_PER_SAMPLE
            if free_samples < MIN_WRITE_SAMPLES:
                return
            chunk = feeder.next_chunk(free_samples)
            if chunk is None:
                self._timer.stop()  # everything was handed over; the sink drains on its own
                return
            if chunk.size == 0:
                continue
            written = device.write(chunk.tobytes())
            if written <= 0:
                return


class PcmWorker(QObject):
    pcmReady = Signal(object, float, object)  # path, tempo, samples | None

    @Slot(object, str, float)
    def load(self, path: Path, ffmpeg_path: str, tempo: float) -> None:
        started = time.perf_counter()
        samples = load_preview_audio(ffmpeg_path, path, tempo)
        log_event(
            LOG,
            "audio.preview_decode",
            level=logging.INFO,
            name=path.name,
            tempo=tempo,
            has_audio=samples is not None,
            seconds=round(samples.size / SAMPLE_RATE, 2) if samples is not None else 0.0,
            megabytes=round(samples.nbytes / 1e6, 1) if samples is not None else 0.0,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
        )
        self.pcmReady.emit(path, tempo, samples)


class PcmController(QObject):
    """Owns the decode thread; results carry the file and the tempo they were decoded for."""

    pcmReady = Signal(object, float, object)
    _load = Signal(object, str, float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._thread.setObjectName("preview-audio-decode")
        self._worker = PcmWorker()
        self._worker.moveToThread(self._thread)
        self._load.connect(self._worker.load)  # queued on the worker thread
        self._worker.pcmReady.connect(self.pcmReady)
        self._thread.start()

    def load(self, path: Path, ffmpeg_path: str, tempo: float) -> None:
        self._load.emit(path, ffmpeg_path, tempo)

    def stop(self, timeout_ms: int = 10_000) -> None:
        self._thread.quit()
        self._thread.wait(timeout_ms)
        self._worker.deleteLater()
