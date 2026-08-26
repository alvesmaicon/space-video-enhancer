"""Sound for the preview: PCM decoding with tempo, the piece feeder, the player over a fake sink, the decode thread."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from space_video_enhancer.audio import PREVIEW_SAMPLE_RATE, load_preview_audio, tempo_filter
from space_video_enhancer.sequence import Clip, ClipSource, Sequence
from space_video_enhancer.timeline import FrameRange
from space_video_enhancer.ui.audio_preview import (
    BUFFER_SECONDS,
    AudioFeeder,
    AudioPiece,
    PcmController,
    PreviewAudioPlayer,
)
from tests.qt_helpers import wait_until
from tests.test_audio import tone_file

A = Path("a.mp4")
B = Path("b.mp4")


def test_tempo_filter_chains_atempo_within_its_range() -> None:
    assert tempo_filter(1.0) == ""
    assert tempo_filter(0.5) == "atempo=0.5"
    assert tempo_filter(0.25) == "atempo=0.5,atempo=0.5"
    assert tempo_filter(3.0) == "atempo=2.0,atempo=1.5"
    with pytest.raises(ValueError):
        tempo_filter(0.0)


def test_feeder_reads_each_piece_from_its_file_at_the_stretched_position() -> None:
    rate = 1000
    pcm = {(A, 1.0): np.arange(1, 1001, dtype=np.int16), (A, 0.5): np.arange(1, 2001, dtype=np.int16)}
    pieces = [AudioPiece(A, 0.5, 1.0), AudioPiece(None, 0.0, 0.25), AudioPiece(B, 0.0, 0.1)]
    feeder = AudioFeeder(pieces, pcm, 1.0, sample_rate=rate)
    assert feeder.total_samples == 500 + 250 + 100
    first = feeder.next_chunk(300)
    assert first is not None and first.tolist() == list(range(501, 801)), "0.5 s into a.mp4"
    rest = feeder.next_chunk(10_000)
    assert rest is not None and rest.size == 200 and rest[-1] == 1000, "a chunk never crosses a piece boundary"
    silence = feeder.next_chunk(10_000)
    assert silence is not None and silence.size == 250 and not silence.any()
    missing = feeder.next_chunk(10_000)
    assert missing is not None and missing.size == 100 and not missing.any(), "no PCM for b.mp4 yet: silence"
    assert feeder.next_chunk(10) is None and feeder.finished and feeder.samples_written == 850

    slow = AudioFeeder([AudioPiece(A, 0.5, 1.0)], pcm, 0.5, sample_rate=rate)
    chunk = slow.next_chunk(10_000)
    assert chunk is not None and chunk.size == 1000 and chunk[0] == 1001, "source second 0.5 sits at 1.0 s when slowed"
    beyond = AudioFeeder([AudioPiece(A, 0.9, 1.5)], pcm, 1.0, sample_rate=rate).next_chunk(10_000)
    assert beyond is not None and beyond.size == 600 and beyond[99] == 1000 and not beyond[100:].any()


class _FakeDevice:
    def __init__(self, sink: _FakeSink) -> None:
        self._sink = sink

    def write(self, data: bytes) -> int:
        self._sink.buffer += data
        return len(data)


class _FakeSink:
    """Enough of ``QAudioSink`` for the player: a bounded buffer that only drains when the test says so."""

    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = capacity
        self.buffer = bytearray()
        self.consumed = 0
        self.started = 0
        self.stopped = 0

    def start(self) -> _FakeDevice:
        self.started += 1
        return _FakeDevice(self)

    def stop(self) -> None:
        self.stopped += 1

    def bytesFree(self) -> int:  # noqa: N802
        return self.capacity - len(self.buffer)

    def setBufferSize(self, size: int) -> None:  # noqa: N802
        self.capacity = size

    def setVolume(self, volume: float) -> None:  # noqa: N802
        pass

    def drain(self, count: int) -> None:
        taken = self.buffer[:count]
        del self.buffer[:count]
        self.consumed += len(taken)


def test_player_fills_the_sink_ahead_and_stops_when_the_pieces_run_out(app: QApplication) -> None:
    sink = _FakeSink()
    player = PreviewAudioPlayer(sink_factory=lambda: sink)
    assert player.available and sink.capacity == int(PREVIEW_SAMPLE_RATE * 2 * BUFFER_SECONDS)
    samples = np.arange(1, PREVIEW_SAMPLE_RATE + 1, dtype=np.int16)  # 1 s
    player.set_pcm(A, 1.0, samples)
    assert player.start([AudioPiece(A, 0.0, 1.0)], 1.0) and player.playing
    assert len(sink.buffer) == sink.capacity, "the buffer is filled at once, up to the sink's capacity"
    assert np.frombuffer(bytes(sink.buffer[:20]), dtype="<i2").tolist() == list(range(1, 11))
    sink.drain(2000)
    player._feed()
    assert len(sink.buffer) == sink.capacity and player.samples_written == (sink.capacity + 2000) // 2
    sink.drain(len(sink.buffer))
    for _ in range(20):
        player._feed()
        sink.drain(len(sink.buffer))
    assert player.samples_written == PREVIEW_SAMPLE_RATE and not player._timer.isActive(), "handed over everything"
    player.stop()
    assert sink.stopped == 1 and not player.playing
    player.set_muted(True)
    assert not player.start([AudioPiece(A, 0.0, 1.0)], 1.0)
    player.set_muted(False)
    assert player.start([AudioPiece(A, 0.0, 1.0)], 1.0) and sink.started == 2
    player.keep_only([B])
    assert not player.has_pcm(A, 1.0) and player.pcm_bytes() == 0
    player.stop()


def test_player_without_a_device_is_simply_unavailable(app: QApplication) -> None:
    player = PreviewAudioPlayer(sink_factory=lambda: None)
    assert not player.available
    assert not player.start([AudioPiece(A, 0.0, 1.0)], 1.0) and not player.playing


def test_playback_runs_follow_cuts_and_clips() -> None:
    a = ClipSource(A, 10, 24.0)
    b = ClipSource(B, 10, 24.0)
    sequence = Sequence((Clip.whole(a, cuts=(FrameRange(4, 5),)), Clip.whole(b, cuts=(FrameRange(0, 1),))))
    runs = sequence.playback_runs(2, 17)
    assert [(run.clip_index, run.span, run.frames) for run in runs] == [
        (0, FrameRange(2, 3), FrameRange(2, 3)),
        (0, FrameRange(6, 9), FrameRange(6, 9)),
        (1, FrameRange(12, 17), FrameRange(2, 7)),
    ]
    assert runs[2].seconds == pytest.approx((2 / 24, 8 / 24))
    assert sequence.playback_runs(4, 5) == () and sequence.playback_runs(19, 19)[0].frames == FrameRange(9, 9)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_preview_audio_is_stretched_without_changing_the_pitch(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    tone = tone_file(tmp_path, seconds=1.0)  # 440 Hz
    normal = load_preview_audio(ffmpeg, tone, 1.0)
    slow = load_preview_audio(ffmpeg, tone, 0.5)
    assert normal is not None and slow is not None
    assert normal.size == pytest.approx(PREVIEW_SAMPLE_RATE, rel=0.02)
    assert slow.size == pytest.approx(2 * PREVIEW_SAMPLE_RATE, rel=0.03), "half speed lasts twice as long"

    def crossings_per_second(samples: np.ndarray) -> float:
        signs = np.signbit(samples.astype(np.int32))
        return float(np.count_nonzero(signs[1:] != signs[:-1])) / (samples.size / PREVIEW_SAMPLE_RATE)

    assert crossings_per_second(normal) == pytest.approx(880, rel=0.05)
    assert crossings_per_second(slow) == pytest.approx(880, rel=0.08), "atempo keeps the pitch"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_pcm_controller_decodes_on_its_own_thread(app: QApplication, tmp_path: Path) -> None:
    controller = PcmController()
    results: list[tuple[Path, float, np.ndarray | None]] = []
    controller.pcmReady.connect(lambda path, tempo, samples: results.append((path, tempo, samples)))
    try:
        tone = tone_file(tmp_path, seconds=0.5)
        controller.load(tone, shutil.which("ffmpeg") or "ffmpeg", 0.5)
        wait_until(app, lambda: len(results) == 1, what="preview pcm")
        path, tempo, samples = results[0]
        assert path == tone and tempo == 0.5 and samples is not None
        assert samples.size == pytest.approx(PREVIEW_SAMPLE_RATE, rel=0.05)
    finally:
        controller.stop()
