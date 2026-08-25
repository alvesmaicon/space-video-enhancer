"""Human-readable numbers for the UI: timecodes, durations, signed slider values."""

from __future__ import annotations


def format_timecode(frame_index: int, fps: float) -> str:
    """``mm:ss.ff`` (frames, not milliseconds) for ``frame_index`` at ``fps``; hours appear only when needed."""
    if fps <= 0:
        return "00:00.00"
    frames_per_second = max(1, round(fps))
    total_seconds, frame = divmod(max(0, int(frame_index)), frames_per_second)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}.{frame:02d}"
    return f"{minutes:02d}:{seconds:02d}.{frame:02d}"


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
