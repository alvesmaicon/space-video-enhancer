"""FFmpeg discovery, encoder selection and command construction."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

VAAPI_DEVICE = Path("/dev/dri/renderD128")
PROBE_TIMEOUT_SECONDS = 8
ENCODER_LIST_TIMEOUT_SECONDS = 5
AUDIO_BITRATE = "192k"

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
) -> list[str]:
    """Raw bgr24 frames on stdin + original audio (trimmed) -> H.264 MP4 that plays everywhere."""
    if width % 2 or height % 2:
        raise ValueError(f"Frame size must be even for {OUTPUT_PIXEL_FORMAT}, got {width}x{height}")

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
        *encoder.args,
    ]
    if not encoder.sets_own_pixel_format:
        command += ["-pix_fmt", OUTPUT_PIXEL_FORMAT]
    command += [
        "-c:a",
        "aac",
        "-b:a",
        AUDIO_BITRATE,
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return command
