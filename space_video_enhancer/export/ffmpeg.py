"""FFmpeg discovery, encoder selection and command construction."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

VAAPI_DEVICE = Path("/dev/dri/renderD128")
PROBE_TIMEOUT_SECONDS = 8
ENCODER_LIST_TIMEOUT_SECONDS = 5
AUDIO_BITRATE = "192k"
AUDIO_PROBE_TIMEOUT_SECONDS = 8

# Encoders pick yuv444p for a bgr24 source unless told otherwise, and yuv444p H.264 does not
# play on QuickTime/iOS/most hardware decoders. Always force yuv420p (which needs even dims).
OUTPUT_PIXEL_FORMAT = "yuv420p"


@dataclass(frozen=True)
class EncoderChoice:
    args: tuple[str, ...]
    label: str
    sets_own_pixel_format: bool = False
    """True when ``args`` already contain a filter that fixes the pixel format (VAAPI)."""


LIBX264 = EncoderChoice(("-c:v", "libx264", "-preset", "medium", "-crf", "18"), "libx264")
LIBX264_FALLBACK = EncoderChoice(LIBX264.args, "libx264 (fallback)")
NVENC = EncoderChoice(("-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19", "-b:v", "0"), "NVIDIA NVENC")
VAAPI = EncoderChoice(
    ("-vaapi_device", str(VAAPI_DEVICE), "-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-qp", "18"),
    "VAAPI",
    sets_own_pixel_format=True,
)


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


class FfmpegProbe:
    """Caches ``ffmpeg -encoders`` and per-encoder smoke tests for one ffmpeg binary."""

    def __init__(self, ffmpeg_path: str) -> None:
        self.ffmpeg_path = ffmpeg_path
        self._encoders: set[str] | None = None
        self._encoder_works: dict[str, bool] = {}

    def available_encoders(self) -> set[str]:
        if self._encoders is not None:
            return self._encoders
        encoders: set[str] = set()
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=False,
                timeout=ENCODER_LIST_TIMEOUT_SECONDS,
            )
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0].startswith("V"):
                    encoders.add(parts[1])
        except (OSError, subprocess.SubprocessError):
            encoders = set()
        self._encoders = encoders
        return encoders

    def encoder_works(self, encoder_name: str) -> bool:
        """Encode one black frame with ``encoder_name`` to confirm the driver actually works."""
        if encoder_name in self._encoder_works:
            return self._encoder_works[encoder_name]

        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=0.1",
            "-frames:v",
            "1",
            "-an",
        ]
        if encoder_name == "h264_nvenc":
            command += ["-c:v", "h264_nvenc", "-f", "null", "-"]
        elif encoder_name == "h264_vaapi":
            command += [*VAAPI.args[:4], "-c:v", "h264_vaapi", "-f", "null", "-"]
        else:
            self._encoder_works[encoder_name] = False
            return False

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=PROBE_TIMEOUT_SECONDS)
            ok = result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        self._encoder_works[encoder_name] = ok
        return ok


def resolve_encoder(probe: FfmpegProbe, *, prefer_hardware: bool = True) -> EncoderChoice:
    if not prefer_hardware:
        return LIBX264
    available = probe.available_encoders()
    if "h264_nvenc" in available and probe.encoder_works("h264_nvenc"):
        return NVENC
    if "h264_vaapi" in available and VAAPI_DEVICE.exists() and probe.encoder_works("h264_vaapi"):
        return VAAPI
    return LIBX264_FALLBACK


def find_ffprobe(ffmpeg_path: str) -> str | None:
    sibling = Path(ffmpeg_path).with_name("ffprobe")
    if sibling.exists():
        return str(sibling)
    return shutil.which("ffprobe")


def source_has_audio(ffmpeg_path: str, source_path: Path) -> bool:
    """Whether ``source_path`` has an audio stream (needed before building an audio filter graph)."""
    ffprobe_path = find_ffprobe(ffmpeg_path)
    try:
        if ffprobe_path is not None:
            result = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "error",
                    "-select_streams",
                    "a",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    str(source_path),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=AUDIO_PROBE_TIMEOUT_SECONDS,
            )
            return result.returncode == 0 and "audio" in result.stdout
        result = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-v",
                "error",
                "-i",
                str(source_path),
                "-map",
                "0:a:0",
                "-t",
                "0.01",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=AUDIO_PROBE_TIMEOUT_SECONDS,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def audio_concat_filter(segments_seconds: Sequence[tuple[float, float]]) -> str:
    """``atrim`` each kept segment of the source audio and ``concat`` them in order (sample accurate)."""
    parts = [
        f"[1:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{index}]"
        for index, (start, end) in enumerate(segments_seconds)
    ]
    inputs = "".join(f"[a{index}]" for index in range(len(segments_seconds)))
    parts.append(f"{inputs}concat=n={len(segments_seconds)}:v=0:a=1[aout]")
    return ";".join(parts)


def build_export_command(
    ffmpeg_path: str,
    *,
    width: int,
    height: int,
    fps: float,
    source_path: Path,
    start_seconds: float,
    duration_seconds: float,
    encoder: EncoderChoice,
    output_path: Path,
    segments_seconds: Sequence[tuple[float, float]] | None = None,
    has_audio: bool = True,
) -> list[str]:
    """Raw bgr24 frames on stdin + original audio -> H.264 MP4 that plays everywhere.

    With a single segment the audio is trimmed with ``-ss``/``-t``. With several (cuts inside
    the clip) the kept pieces are trimmed and concatenated so the audio follows the video;
    that needs to know whether the source has audio at all (``has_audio``).
    """
    if width % 2 or height % 2:
        raise ValueError(f"Frame size must be even for {OUTPUT_PIXEL_FORMAT}, got {width}x{height}")

    multi_segment = segments_seconds is not None and len(segments_seconds) > 1
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
    ]
    if multi_segment and has_audio:
        assert segments_seconds is not None
        command += [
            "-i",
            str(source_path),
            "-filter_complex",
            audio_concat_filter(segments_seconds),
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
        ]
    elif multi_segment:
        command += ["-map", "0:v:0", "-an"]
    else:
        command += [
            "-ss",
            f"{start_seconds:.6f}",
            "-t",
            f"{duration_seconds:.6f}",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
        ]
    command += list(encoder.args)
    if not encoder.sets_own_pixel_format:
        command += ["-pix_fmt", OUTPUT_PIXEL_FORMAT]
    if not (multi_segment and not has_audio):
        command += ["-c:a", "aac", "-b:a", AUDIO_BITRATE]
    command += ["-shortest", "-movflags", "+faststart", str(output_path)]
    return command
