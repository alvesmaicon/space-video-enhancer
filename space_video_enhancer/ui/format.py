"""Human-readable numbers for the UI: timecodes, durations, signed slider values."""

from __future__ import annotations

import re

TIMECODE_PATTERN = re.compile(r"^\s*(?:(\d+):)?(?:(\d+):)?(\d+)(?:\.(\d{1,2}))?\s*$")


def format_timecode(frame_index: int, fps: float, *, force_hours: bool = False) -> str:
    """``mm:ss.ff`` (frames, not milliseconds) for ``frame_index`` at ``fps``.

    Hours appear when needed, or always with ``force_hours`` so a position and the clip
    duration it is shown next to keep the same width.
    """
    if fps <= 0:
        return "0:00:00.00" if force_hours else "00:00.00"
    frames_per_second = max(1, round(fps))
    total_seconds, frame = divmod(max(0, int(frame_index)), frames_per_second)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours or force_hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}.{frame:02d}"
    return f"{minutes:02d}:{seconds:02d}.{frame:02d}"


def shows_hours(frame_count: int, fps: float) -> bool:
    """A clip of an hour or more shows hours in every timecode, so position and duration keep one width."""
    return fps > 0 and frame_count / fps >= 3600


def position_labels(frame_index: int, frame_count: int, fps: float) -> tuple[str, str]:
    """``("mm:ss.ff / mm:ss.ff", "frame n / N")`` for the playhead: what the transport bar and the monitor show."""
    hours = shows_hours(frame_count, fps)
    position = format_timecode(frame_index, fps, force_hours=hours)
    duration = format_timecode(frame_count, fps, force_hours=hours)
    shown = frame_index + 1 if frame_count else 0
    return f"{position} / {duration}", f"frame {shown} / {frame_count}"


def format_duration(seconds: float) -> str:
    """``m:ss`` for ETAs and elapsed times."""
    total = max(0, round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def format_signed(value: int, center: int = 50) -> str:
    """Slider value relative to its neutral point: ``-12``, ``0``, ``+7``."""
    offset = value - center
    return f"{offset:+d}" if offset else "0"


def format_percent(value: int) -> str:
    return f"{value}%"


def parse_timecode(text: str, fps: float) -> int | None:
    """Frame index for ``[h:][m:]s[.ff]`` (``ff`` = frames, as displayed) or plain seconds; ``None`` if invalid."""
    match = TIMECODE_PATTERN.match(text)
    if match is None or fps <= 0:
        return None
    first, second, seconds_text, frames_text = match.groups()
    hours = minutes = 0
    if first is not None and second is not None:
        hours, minutes = int(first), int(second)
    elif first is not None:
        minutes = int(first)
    total_seconds = hours * 3600 + minutes * 60 + int(seconds_text)
    frames = int(frames_text) if frames_text else 0
    return round(total_seconds * fps) + frames
