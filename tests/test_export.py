"""Export command construction plus an end-to-end encode through the real FFmpeg (when installed)."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from space_video_enhancer.export.ffmpeg import (
    LIBX264,
    NVENC,
    VAAPI,
    AudioSegment,
    FfmpegProbe,
    atempo_chain,
    audio_concat_filter,
    build_export_command,
    resolve_encoder,
    source_has_audio,
)
from space_video_enhancer.export.job import ExportRequest, _validation_problem, run_export
from space_video_enhancer.pipeline.stabilization.wobble import analyze_wobble, wobble_window_frames
from space_video_enhancer.sequence import Clip, ClipSource, Sequence
from space_video_enhancer.settings import (
    CaptureSettings,
    CropSettings,
    FilterSettings,
    GeometrySettings,
    NormalizedRect,
    RenderSettings,
    RetimingMode,
    RotationMode,
    StabilizationSettings,
)
from space_video_enhancer.timeline import FrameRange
from space_video_enhancer.video import VideoSource
from tests.conftest import make_textured_frame

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
requires_ffmpeg = pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe not on PATH")


def _command(encoder, width: int = 64, height: int = 64) -> list[str]:  # type: ignore[no-untyped-def]
    return build_export_command(
        "ffmpeg",
        width=width,
        height=height,
        fps=30.0,
        sources=[Path("in.mp4")],
        audio=[AudioSegment(0, 0.0, 1.0)],
        encoder=encoder,
        output_path=Path("out.mp4"),
    )


def test_software_and_nvenc_commands_force_yuv420p() -> None:
    for encoder in (LIBX264, NVENC):
        command = _command(encoder)
        assert command[command.index("-pix_fmt", 10) + 1] == "yuv420p"  # the second -pix_fmt is the output one
        assert "+faststart" in command


def test_vaapi_command_keeps_its_own_pixel_format_filter() -> None:
    command = _command(VAAPI)
    assert command.count("-pix_fmt") == 1  # only the rawvideo input format
    assert "format=nv12,hwupload" in command


def test_odd_dimensions_are_rejected_early() -> None:
    with pytest.raises(ValueError, match="even"):
        _command(LIBX264, width=65, height=64)


def test_resolve_encoder_software_only() -> None:
    assert resolve_encoder(FfmpegProbe("ffmpeg"), prefer_hardware=False) is LIBX264


def _write_source_video(path: Path, frames: int = 12, size: tuple[int, int] = (65, 65)) -> None:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (width, height))
    assert writer.isOpened(), "OpenCV could not open a VideoWriter (mp4v)"
    base = cv2.resize(make_textured_frame(), (width, height))
    for index in range(frames):
        writer.write(np.roll(base, index, axis=1))
    writer.release()


def _probe_video(path: Path) -> dict:  # type: ignore[type-arg]
    assert FFPROBE is not None
    output = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=pix_fmt,width,height,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(output)["streams"][0]


def _request(
    source: Path,
    output: Path,
    settings: RenderSettings,
    workers: int,
    cuts: tuple[tuple[int, int], ...] = (),
) -> ExportRequest:
    assert FFMPEG is not None
    return ExportRequest.single(source, output, settings, 0, 11, 24.0, workers, FFMPEG, LIBX264, cuts=cuts)


def _multi_command(audio: list[AudioSegment], sources: list[Path] | None = None) -> list[str]:
    return build_export_command(
        "ffmpeg",
        width=64,
        height=64,
        fps=24.0,
        sources=sources or [Path("in.mp4")],
        audio=audio,
        encoder=LIBX264,
        output_path=Path("out.mp4"),
    )


def test_multi_segment_command_trims_and_concatenates_the_audio() -> None:
    segments = [AudioSegment(0, 0.0, 1.0), AudioSegment(0, 2.0, 3.5)]
    command = _multi_command(segments)
    assert "-ss" not in command and "-filter_complex" in command
    graph = command[command.index("-filter_complex") + 1]
    assert graph == audio_concat_filter(segments)
    assert graph.count("atrim=") == 2 and "concat=n=2:v=0:a=1,apad[aout]" in graph and "asetpts=PTS-STARTPTS" in graph
    assert graph.startswith("[1:a]atrim=start=0.000000:end=1.000000") and "aresample=48000" in graph
    assert command[command.index("-map") + 1] == "0:v:0" and "[aout]" in command
    silent = _multi_command([AudioSegment(0, 0.0, 1.0, False), AudioSegment(0, 2.0, 3.5, False)])
    assert "-an" in silent and "-filter_complex" not in silent and "aac" not in silent
    single = _command(LIBX264)
    assert "-ss" in single and "1:a?" in single  # one segment keeps the original, proven command
    with pytest.raises(ValueError, match="audio segment"):
        _multi_command([])


def test_sequence_command_uses_one_input_per_source_and_silence_where_there_is_no_audio() -> None:
    sources = [Path("a.mp4"), Path("b.mp4")]
    audio = [AudioSegment(0, 0.0, 0.5), AudioSegment(1, 1.0, 1.25, has_audio=False), AudioSegment(0, 2.0, 2.5)]
    command = _multi_command(audio, sources)
    inputs = [command[index + 1] for index, token in enumerate(command) if token == "-i"]
    assert inputs == ["pipe:0", "a.mp4", "b.mp4"]
    graph = command[command.index("-filter_complex") + 1]
    parts = graph.split(";")
    assert parts[0].startswith("[1:a]atrim=start=0.000000:end=0.500000")
    assert parts[1].startswith("anullsrc=r=48000:cl=stereo,atrim=0:0.250000") and "[a1]" in parts[1]
    assert parts[2].startswith("[1:a]atrim=start=2.000000")
    assert parts[-1] == "[a0][a1][a2]concat=n=3:v=0:a=1,apad[aout]"


def test_request_segments_follow_the_cuts(tmp_path: Path) -> None:
    request = ExportRequest.single(
        tmp_path / "in.mp4",
        tmp_path / "out.mp4",
        RenderSettings(),
        0,
        11,
        24.0,
        1,
        "ffmpeg",
        LIBX264,
        cuts=((3, 4), (8, 9)),
    )
    assert [(s.frames.start, s.frames.end) for s in request.segments] == [(0, 2), (5, 7), (10, 11)]
    assert request.expected_frames == 8 and request.duration_seconds == 8 / 24.0
    assert request.segments[1].seconds == (5 / 24.0, 8 / 24.0)
    assert request.source_path == tmp_path / "in.mp4" and request.fps == 24.0


@requires_ffmpeg
@pytest.mark.parametrize("workers", [1, 3])
def test_end_to_end_export_produces_playable_yuv420p(tmp_path: Path, workers: int) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    _write_source_video(source)  # 65x65: odd on purpose
    progress: list[tuple[int, int]] = []

    result = run_export(
        _request(source, output, RenderSettings(filters=FilterSettings(sharpen=10)), workers),
        progress=lambda done, total: progress.append((done, total)),
        cancel=threading.Event(),
    )

    assert result.success, result.message
    assert result.processed_frames == 12
    assert result.metrics["frames"] == 12 and result.metrics["workers"] == workers
    assert result.metrics["render"]["total"]["count"] == 12
    assert set(result.metrics["render"]["stages"]) >= {"filters", "preset", "geometry", "crop"}
    assert result.metrics["fps"] is not None and result.metrics["output_bytes"] > 0
    assert progress and progress[-1] == (12, 12)
    stream = _probe_video(output)
    assert stream["pix_fmt"] == "yuv420p"
    assert (stream["width"], stream["height"]) == (64, 64)
    assert int(stream["nb_read_frames"]) == 12


@requires_ffmpeg
@pytest.mark.parametrize("workers", [1, 3])
def test_cuts_inside_the_clip_are_removed_from_the_export(tmp_path: Path, workers: int) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    _write_source_video(source)
    assert source_has_audio(FFMPEG or "ffmpeg", source) is False  # VideoWriter output carries no audio
    settings = RenderSettings(filters=FilterSettings(sharpen=0))
    request = _request(source, output, settings, workers, cuts=((3, 4), (8, 9)))
    result = run_export(request, progress=lambda *_: None, cancel=threading.Event())
    assert result.success, result.message
    assert result.processed_frames == 8 and result.expected_frames == 8
    assert int(_probe_video(output)["nb_read_frames"]) == 8


@requires_ffmpeg
def test_cuts_that_remove_everything_fail_before_ffmpeg_starts(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    _write_source_video(source)
    request = _request(source, output, RenderSettings(), 1, cuts=((0, 11),))
    result = run_export(request, progress=lambda *_: None, cancel=threading.Event())
    assert not result.success and "nada a exportar" in result.message and not output.exists()


@requires_ffmpeg
def test_export_applies_rotation_and_crop_consistently(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    _write_source_video(source, size=(96, 64))
    settings = RenderSettings(
        filters=FilterSettings(sharpen=0),
        geometry=GeometrySettings(RotationMode.CW_90),
        crop=CropSettings(True, NormalizedRect(0.0, 0.0, 0.5, 0.5)),
    )
    result = run_export(_request(source, output, settings, 1), progress=lambda *_: None, cancel=threading.Event())
    assert result.success, result.message
    stream = _probe_video(output)
    # 96x64 rotated -> 64x96; top-left half -> 32x48
    assert (stream["width"], stream["height"]) == (32, 48)


@requires_ffmpeg
def test_cancelled_export_leaves_no_file(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    _write_source_video(source)
    cancel = threading.Event()
    cancel.set()
    result = run_export(_request(source, output, RenderSettings(), 1), progress=lambda *_: None, cancel=cancel)
    assert result.cancelled and not result.success
    assert not output.exists()


def test_missing_source_fails_gracefully(tmp_path: Path) -> None:
    request = ExportRequest.single(
        tmp_path / "missing.mp4", tmp_path / "out.mp4", RenderSettings(), 0, 0, 24.0, 1, "ffmpeg", LIBX264
    )
    result = run_export(request, progress=lambda *_: None, cancel=threading.Event())
    assert not result.success and "reabrir" in result.message


def _write_source_with_audio(path: Path, frames: int = 12, size: tuple[int, int] = (64, 64)) -> None:
    """A tiny clip with a 440 Hz tone (VideoWriter output never has audio)."""
    subprocess.run(
        [
            FFMPEG or "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={size[0]}x{size[1]}:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


def _probe_audio_duration(path: Path) -> float | None:
    assert FFPROBE is not None
    output = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    streams = json.loads(output)["streams"]
    return float(streams[0]["duration"]) if streams else None


def _source(path: Path, frames: int, size: tuple[int, int]) -> ClipSource:
    return ClipSource(path, frames, 24.0, size[0], size[1])


@requires_ffmpeg
@pytest.mark.parametrize("workers", [1, 3])
def test_sequence_export_joins_two_sources_with_their_own_cuts_and_settings(tmp_path: Path, workers: int) -> None:
    first_path, second_path, output = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "out.mp4"
    _write_source_video(first_path, size=(64, 64))
    _write_source_video(second_path, size=(96, 48))  # different shape: letterboxed into 64x64
    bright = RenderSettings(filters=FilterSettings(brightness=100, sharpen=0))
    sequence = Sequence(
        (
            Clip(
                _source(first_path, 12, (64, 64)),
                FrameRange(0, 11),
                settings=RenderSettings(filters=FilterSettings(sharpen=0)),
            ),
            Clip(_source(second_path, 12, (96, 48)), FrameRange(0, 11), cuts=(FrameRange(2, 3),), settings=bright),
        )
    )
    assert sequence.export_count == 22
    request = ExportRequest(sequence, output, workers, FFMPEG or "ffmpeg", LIBX264)
    result = run_export(request, progress=lambda *_: None, cancel=threading.Event())
    assert result.success, result.message
    assert result.processed_frames == 22 and result.expected_frames == 22
    stream = _probe_video(output)
    assert (stream["width"], stream["height"]) == (64, 64) and int(stream["nb_read_frames"]) == 22
    capture = cv2.VideoCapture(str(output))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    assert len(decoded) == 22
    first_half = np.mean([frame.mean() for frame in decoded[:12]])
    second_half = np.mean([frame[16:48].mean() for frame in decoded[12:]])  # the picture area of the letterbox
    assert second_half > first_half + 15, "the second clip carries its own (brighter: beta +24) settings"
    assert np.mean([frame[:12].mean() for frame in decoded[14:]]) < 16, "letterbox bars above the 96x48 picture"


@requires_ffmpeg
def test_sequence_export_concatenates_audio_and_fills_silence_for_sources_without_it(tmp_path: Path) -> None:
    with_audio, silent, output = tmp_path / "tone.mp4", tmp_path / "silent.mp4", tmp_path / "out.mp4"
    _write_source_with_audio(with_audio)
    _write_source_video(silent, size=(64, 64))
    assert source_has_audio(FFMPEG or "ffmpeg", with_audio) and not source_has_audio(FFMPEG or "ffmpeg", silent)
    sequence = Sequence(
        (
            Clip(_source(with_audio, 12, (64, 64)), FrameRange(0, 11), cuts=(FrameRange(4, 7),)),
            Clip(_source(silent, 12, (64, 64)), FrameRange(0, 11)),
            Clip(_source(with_audio, 12, (64, 64)), FrameRange(6, 11)),
        )
    )
    assert sequence.export_count == 8 + 12 + 6 and sequence.source_paths == (with_audio, silent)
    result = run_export(
        ExportRequest(sequence, output, 2, FFMPEG or "ffmpeg", LIBX264),
        progress=lambda *_: None,
        cancel=threading.Event(),
    )
    assert result.success, result.message
    assert int(_probe_video(output)["nb_read_frames"]) == 26
    duration = _probe_audio_duration(output)
    assert duration is not None and duration == pytest.approx(26 / 24.0, abs=0.15)


def test_mixed_fps_sequences_pass_validation_and_count_retimed_frames(tmp_path: Path) -> None:
    sequence = Sequence(
        (
            Clip(ClipSource(tmp_path / "a.mp4", 10, 24.0), FrameRange(0, 9)),  # 24 fps -> 10 timeline frames (1:1)
            Clip(ClipSource(tmp_path / "b.mp4", 20, 48.0), FrameRange(0, 19)),  # 48 fps, 20 source -> 10 frames
        )
    )
    request = ExportRequest(sequence, tmp_path / "out.mp4", 1, "ffmpeg", LIBX264)
    assert _validation_problem(request) is None  # a different fps no longer refuses the export
    assert sequence.fps == 24.0 and request.expected_frames == 20


@requires_ffmpeg
def test_rolling_shutter_export_measures_motion_at_every_segment_start(tmp_path: Path) -> None:
    """Cuts split the run; the unskew still gets a previous frame at each restart (read on purpose)."""
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    _write_source_video(source)
    settings = RenderSettings(stabilization=StabilizationSettings(rolling_shutter=0.7))
    request = _request(source, output, settings, 3, cuts=((4, 6),))
    assert request.effective_worker_count == 1, "the unskew is sequential"
    result = run_export(request, progress=lambda *_: None, cancel=threading.Event())
    assert result.success, result.message
    assert result.processed_frames == 9 and int(_probe_video(output)["nb_read_frames"]) == 9
    assert "stabilize" in result.metrics["render"]["stages"]


def test_wobble_export_analyses_the_clip_first_and_renders_in_parallel(tmp_path: Path) -> None:
    """Every frame knows its own wobble from the analysis pass, so the frames need no order; the pass reports status."""
    source = tmp_path / "source.mp4"
    output = tmp_path / "out.mp4"
    _write_source_video(source)
    settings = RenderSettings(stabilization=StabilizationSettings(rolling_shutter_auto=True))
    request = _request(source, output, settings, 2)
    assert request.effective_worker_count == 2, "the wobble alone is not sequential"
    statuses: list[str] = []
    result = run_export(request, progress=lambda *_: None, cancel=threading.Event(), status=statuses.append)
    assert result.success, result.message
    assert result.processed_frames == 12 and int(_probe_video(output)["nb_read_frames"]) == 12
    [analysis] = result.metrics["wobble_analysis"]
    assert analysis["clip"] == 0 and analysis["frames"] == 12 and (analysis["first"], analysis["last"]) == (0, 11)
    assert analysis["axis"] == "rows", "no wobble and no rotation metadata: a landscape readout"
    assert statuses and "oscilação" in statuses[-1] and "100%" in statuses[-1], statuses

    video = VideoSource(source)
    try:
        track = analyze_wobble(video.read_frame, source, 0, 11, wobble_window_frames(24.0)).track
    finally:
        video.release()
    assert track is not None
    provided = replace(request, wobble_tracks=(track,))
    statuses.clear()
    result = run_export(provided, progress=lambda *_: None, cancel=threading.Event(), status=statuses.append)
    assert result.success and "wobble_analysis" not in result.metrics and not statuses, (
        "the preview's track spares the pass"
    )

    cancel = threading.Event()
    cancel.set()
    cancelled_output = tmp_path / "cancelled.mp4"
    cancelled = run_export(_request(source, cancelled_output, settings, 2), progress=lambda *_: None, cancel=cancel)
    assert cancelled.cancelled and not cancelled.success and not cancelled_output.exists(), "cancelled during the pass"


def _write_source_video_at(path: Path, fps: float, frames: int = 12, size: tuple[int, int] = (64, 64)) -> None:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "OpenCV could not open a VideoWriter (mp4v)"
    base = cv2.resize(make_textured_frame(), (width, height))
    for index in range(frames):
        writer.write(np.roll(base, index, axis=1))
    writer.release()


@requires_ffmpeg
@pytest.mark.parametrize("workers", [1, 3])
def test_mixed_frame_rate_clips_are_retimed_to_one_timeline(tmp_path: Path, workers: int) -> None:
    fast, slow, output = tmp_path / "fast.mp4", tmp_path / "slow.mp4", tmp_path / "out.mp4"
    _write_source_video_at(fast, 24.0, 12)
    _write_source_video_at(slow, 12.0, 12)
    a = ClipSource(fast, 12, 24.0, 64, 64)
    b = ClipSource(slow, 12, 12.0, 64, 64)  # half the timeline rate -> every source frame is doubled
    sequence = Sequence((Clip.whole(a), Clip.whole(b)))

    assert sequence.fps == 24.0  # the first clip sets the timeline rate
    assert sequence.export_count == 36  # 12 frames 1:1 + 12 frames doubled to 24

    request = ExportRequest(sequence, output, workers, FFMPEG or "ffmpeg", LIBX264)
    result = run_export(request, progress=lambda *_: None, cancel=threading.Event())

    assert result.success, result.message
    assert result.processed_frames == 36 and result.expected_frames == 36
    assert int(_probe_video(output)["nb_read_frames"]) == 36


@requires_ffmpeg
@pytest.mark.parametrize("mode", [RetimingMode.NEAREST, RetimingMode.BLEND, RetimingMode.OPTICAL_FLOW])
def test_retiming_modes_all_export_the_same_retimed_frame_count(tmp_path: Path, mode: RetimingMode) -> None:
    fast, slow, output = tmp_path / "fast.mp4", tmp_path / "slow.mp4", tmp_path / "out.mp4"
    _write_source_video_at(fast, 24.0, 12)
    _write_source_video_at(slow, 12.0, 12)
    a = ClipSource(fast, 12, 24.0, 64, 64)
    b = ClipSource(slow, 12, 12.0, 64, 64)
    retimed = RenderSettings(capture=CaptureSettings(retiming=mode))
    sequence = Sequence((Clip.whole(a), Clip.whole(b, settings=retimed)))

    result = run_export(
        ExportRequest(sequence, output, 1, FFMPEG or "ffmpeg", LIBX264),
        progress=lambda *_: None,
        cancel=threading.Event(),
    )

    assert result.success, result.message
    assert result.processed_frames == 36  # 12 (1:1) + 12 doubled to 24; the mode only changes the pixels
    assert int(_probe_video(output)["nb_read_frames"]) == 36


def test_retimed_audio_is_stretched_with_atempo_and_silence_uses_the_output_length() -> None:
    assert atempo_chain(0.5) == "atempo=0.500000"
    assert atempo_chain(0.25) == "atempo=0.500000,atempo=0.500000"  # below 0.5 needs a chain
    assert atempo_chain(4.0) == "atempo=2.000000,atempo=2.000000"

    graph = audio_concat_filter([AudioSegment(0, 0.0, 1.0, tempo=0.5), AudioSegment(0, 2.0, 3.0, tempo=0.5)])
    assert graph.count("atempo=0.500000") == 2, "each retimed segment is stretched"
    silent = audio_concat_filter([AudioSegment(0, 0.0, 1.0, has_audio=False, tempo=0.5), AudioSegment(0, 2.0, 3.0)])
    assert "atrim=0:2.000000" in silent, "1 s of a 0.5x clip is 2 s of silence on the timeline"


def test_a_single_retimed_or_muted_clip_takes_the_right_audio_path() -> None:
    def command(segment: AudioSegment) -> list[str]:
        return build_export_command(
            "ffmpeg",
            width=64,
            height=64,
            fps=30.0,
            sources=[Path("in.mp4")],
            audio=[segment],
            encoder=LIBX264,
            output_path=Path("out.mp4"),
        )

    retimed = command(AudioSegment(0, 0.0, 1.0, tempo=0.5))
    assert "-filter_complex" in retimed and "-ss" not in retimed, "atempo needs the filter graph"
    assert "atempo=0.500000" in retimed[retimed.index("-filter_complex") + 1]
    muted = command(AudioSegment(0, 0.0, 1.0, has_audio=False))
    assert "-an" in muted and "-filter_complex" not in muted and "aac" not in muted


@requires_ffmpeg
def test_slow_motion_doubles_the_output_frames(tmp_path: Path) -> None:
    source, output = tmp_path / "src.mp4", tmp_path / "out.mp4"
    _write_source_video_at(source, 24.0, 12)
    clip_source = ClipSource(source, 12, 24.0, 64, 64)
    slow = RenderSettings(capture=CaptureSettings(speed=0.5, retiming=RetimingMode.BLEND))
    sequence = Sequence((Clip.whole(clip_source, settings=slow),))
    assert sequence.timeline_count == 24  # 12 frames of 24 fps footage played at half speed

    result = run_export(
        ExportRequest(sequence, output, 1, FFMPEG or "ffmpeg", LIBX264),
        progress=lambda *_: None,
        cancel=threading.Event(),
    )

    assert result.success, result.message
    assert result.processed_frames == 24 and int(_probe_video(output)["nb_read_frames"]) == 24
