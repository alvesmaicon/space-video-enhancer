"""Frame ranges on the clip timeline: the clip, the cuts removed inside it, and what remains.

Frames are absolute indices into the source video; ranges are inclusive on both ends.
Everything here is pure so the UI, the export job and the tests share one definition of
"what gets exported".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Union

RangeLike = Union["FrameRange", tuple[int, int]]


@dataclass(frozen=True, order=True)
class FrameRange:
    start: int
    end: int
    """Inclusive."""

    @property
    def count(self) -> int:
        return max(0, self.end - self.start + 1)

    def contains(self, frame: int) -> bool:
        return self.start <= frame <= self.end

    def clamped(self, bounds: FrameRange) -> FrameRange | None:
        start = max(self.start, bounds.start)
        end = min(self.end, bounds.end)
        return FrameRange(start, end) if start <= end else None


def as_range(value: RangeLike) -> FrameRange:
    if isinstance(value, FrameRange):
        return FrameRange(min(value.start, value.end), max(value.start, value.end))
    start, end = value
    return FrameRange(min(start, end), max(start, end))


def normalize_cuts(cuts: Iterable[RangeLike], bounds: FrameRange | None = None) -> tuple[FrameRange, ...]:
    """Sorted cuts, clamped to ``bounds``, with overlapping or adjacent ones merged."""
    ranges = []
    for value in cuts:
        cut = as_range(value)
        if bounds is not None:
            clamped = cut.clamped(bounds)
            if clamped is None:
                continue
            cut = clamped
        ranges.append(cut)
    ranges.sort()
    merged: list[FrameRange] = []
    for cut in ranges:
        if merged and cut.start <= merged[-1].end + 1:
            merged[-1] = FrameRange(merged[-1].start, max(merged[-1].end, cut.end))
        else:
            merged.append(cut)
    return tuple(merged)


def kept_segments(clip: FrameRange, cuts: Iterable[RangeLike]) -> tuple[FrameRange, ...]:
    """``clip`` minus ``cuts``, in playback order."""
    segments: list[FrameRange] = []
    cursor = clip.start
    for cut in normalize_cuts(cuts, clip):
        if cut.start > cursor:
            segments.append(FrameRange(cursor, cut.start - 1))
        cursor = cut.end + 1
    if cursor <= clip.end:
        segments.append(FrameRange(cursor, clip.end))
    return tuple(segments)


def frame_count(segments: Iterable[FrameRange]) -> int:
    return sum(segment.count for segment in segments)


def cut_at(frame: int, cuts: Iterable[RangeLike]) -> FrameRange | None:
    for cut in normalize_cuts(cuts):
        if cut.contains(frame):
            return cut
    return None


def next_kept_frame(frame: int, cuts: Iterable[RangeLike], last_frame: int) -> int | None:
    """``frame`` itself when it is kept, else the first kept frame after the cut; ``None`` past the end."""
    cut = cut_at(frame, cuts)
    if cut is None:
        return frame if frame <= last_frame else None
    following = cut.end + 1
    if following > last_frame:
        return None
    return next_kept_frame(following, cuts, last_frame)


def segments_seconds(segments: Sequence[FrameRange], fps: float) -> tuple[tuple[float, float], ...]:
    """Source-time (start, end) in seconds of each segment; ``end`` is exclusive (the frame after the last)."""
    if fps <= 0:
        return tuple((0.0, 0.0) for _segment in segments)
    return tuple((segment.start / fps, (segment.end + 1) / fps) for segment in segments)
