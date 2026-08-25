"""Export command construction plus an end-to-end encode through the real FFmpeg (when installed)."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from space_video_enhancer.export.ffmpeg import (
    LIBX264,
    NVENC,
    VAAPI,
    FfmpegProbe,
    audio_concat_filter,
    build_export_command,
    resolve_encoder,
    source_has_audio,
)
from space_video_enhancer.export.job import ExportRequest, run_export
from space_video_enhancer.settings import (
    CropSettings,
    FilterSettings,
    GeometrySettings,
    NormalizedRect,
    RenderSettings,
    RotationMode,
)
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
        source_path=Path("in.mp4"),
        start_seconds=0.0,
        duration_seconds=1.0,
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
    return ExportRequest(
        source_path=source,
        output_path=output,
        settings=settings,
        start_frame=0,
        end_frame=11,
        fps=24.0,
        worker_count=workers,
        ffmpeg_path=FFMPEG,
        encoder=LIBX264,
        cuts=cuts,
    )


def test_multi_segment_command_trims_and_concatenates_the_audio() -> None:
    segments = ((0.0, 1.0), (2.0, 3.5))
    command = build_export_command(
        "ffmpeg",
        width=64,
        height=64,
        fps=24.0,
        source_path=Path("in.mp4"),
        start_seconds=0.0,
        duration_seconds=2.5,
        encoder=LIBX264,
        output_path=Path("out.mp4"),
        segments_seconds=segments,
        has_audio=True,
    )
    assert "-ss" not in command and "-filter_complex" in command
    graph = command[command.index("-filter_complex") + 1]
    assert graph == audio_concat_filter(segments)
    assert graph.count("atrim=") == 2 and "concat=n=2:v=0:a=1[aout]" in graph and "asetpts=PTS-STARTPTS" in graph
    assert command[command.index("-map") + 1] == "0:v:0" and "[aout]" in command
    silent = build_export_command(
        "ffmpeg",
        width=64,
        height=64,
        fps=24.0,
        source_path=Path("in.mp4"),
        start_seconds=0.0,
        duration_seconds=2.5,
        encoder=LIBX264,
        output_path=Path("out.mp4"),
        segments_seconds=segments,
        has_audio=False,
    )
    assert "-an" in silent and "-filter_complex" not in silent and "aac" not in silent
    single = _command(LIBX264)
    assert "-ss" in single and "1:a?" in single  # one segment keeps the original, proven command


def test_request_segments_follow_the_cuts(tmp_path: Path) -> None:
    request = ExportRequest(
        source_path=tmp_path / "in.mp4",
        output_path=tmp_path / "out.mp4",
        settings=RenderSettings(),
        start_frame=0,
        end_frame=11,
        fps=24.0,
        worker_count=1,
        ffmpeg_path="ffmpeg",
        encoder=LIBX264,
        cuts=((3, 4), (8, 9)),
    )
    assert [(segment.start, segment.end) for segment in request.segments] == [(0, 2), (5, 7), (10, 11)]
    assert request.expected_frames == 8 and request.duration_seconds == 8 / 24.0
    assert request.segments_seconds[1] == (5 / 24.0, 8 / 24.0)


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
    request = ExportRequest(
        source_path=tmp_path / "missing.mp4",
        output_path=tmp_path / "out.mp4",
        settings=RenderSettings(),
        start_frame=0,
        end_frame=0,
        fps=24.0,
        worker_count=1,
        ffmpeg_path="ffmpeg",
        encoder=LIBX264,
    )
    result = run_export(request, progress=lambda *_: None, cancel=threading.Event())
    assert not result.success and "reabrir" in result.message
