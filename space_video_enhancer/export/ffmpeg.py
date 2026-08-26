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
AUDIO_SAMPLE_RATE = 48000
AUDIO_NORMALIZE = f"aresample={AUDIO_SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo"
"""Every concatenated piece gets the same rate and layout, so clips from different files still join."""

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


@dataclass(frozen=True)
class AudioSegment:
    """The audio to play under one run of exported frames: a slice of one source, or silence."""

    source_index: int
    """Index into the ``sources`` given to :func:`build_export_command`."""
    start_seconds: float
    end_seconds: float
    """Exclusive."""
    has_audio: bool = True

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


def audio_concat_filter(segments: Sequence[AudioSegment]) -> str:
    """``atrim`` each segment from its source (or generate silence) and ``concat`` them in order.

    Every piece is resampled to the same rate and layout so sources that disagree still concat.
    The result is padded with silence (``apad``): a source whose audio ends a few ms before its
    video would otherwise make ``-shortest`` drop the last video frame.
    """
    parts = []
    for index, segment in enumerate(segments):
        if segment.has_audio:
            parts.append(
                f"[{segment.source_index + 1}:a]atrim=start={segment.start_seconds:.6f}:end={segment.end_seconds:.6f},"
                f"asetpts=PTS-STARTPTS,{AUDIO_NORMALIZE}[a{index}]"
            )
        else:
            parts.append(
                f"anullsrc=r={AUDIO_SAMPLE_RATE}:cl=stereo,atrim=0:{segment.duration_seconds:.6f},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
    inputs = "".join(f"[a{index}]" for index in range(len(segments)))
    parts.append(f"{inputs}concat=n={len(segments)}:v=0:a=1,apad[aout]")
    return ";".join(parts)


def build_export_command(
    ffmpeg_path: str,
    *,
    width: int,
    height: int,
    fps: float,
    sources: Sequence[Path],
    audio: Sequence[AudioSegment],
    encoder: EncoderChoice,
    output_path: Path,
) -> list[str]:
    """Raw bgr24 frames on stdin + the sources' audio -> H.264 MP4 that plays everywhere.

    With a single audio segment the source is trimmed with ``-ss``/``-t`` (the original, proven
    command). With several (cuts, or more than one clip) every source becomes an input and the
    pieces are trimmed, normalized and concatenated so the audio follows the video; segments of
    sources without audio become silence of the same length. No audio anywhere -> ``-an``.
    """
    if width % 2 or height % 2:
        raise ValueError(f"Frame size must be even for {OUTPUT_PIXEL_FORMAT}, got {width}x{height}")
    if not audio:
        raise ValueError("at least one audio segment is needed (it defines the timing)")

    multi_segment = len(audio) > 1
    has_audio = any(segment.has_audio for segment in audio)
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
        for source in sources:
            command += ["-i", str(source)]
        command += ["-filter_complex", audio_concat_filter(audio), "-map", "0:v:0", "-map", "[aout]"]
    elif multi_segment:
        command += ["-map", "0:v:0", "-an"]
    else:
        segment = audio[0]
        command += [
            "-ss",
            f"{segment.start_seconds:.6f}",
            "-t",
            f"{segment.duration_seconds:.6f}",
            "-i",
            str(sources[segment.source_index]),
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
