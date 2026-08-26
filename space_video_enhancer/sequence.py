"""A sequence of clips on the timeline: trimmed pieces of source files, each with its own cuts and settings.

Two coordinate systems. *Timeline frames* run over every clip's trimmed range back to back,
cuts included — the timeline shows removed ranges as red bands that stay navigable. *Export
frames* are what leaves the app: timeline frames minus the cuts. Every clip carries its own
:class:`RenderSettings`, so "different adjustments per part" is just splitting a clip in two.
Everything here is pure and immutable; editing operations return a new :class:`Sequence`.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from space_video_enhancer.settings import RenderSettings
from space_video_enhancer.timeline import (
    FrameRange,
    RangeLike,
    as_range,
    cut_at,
    frame_count,
    kept_segments,
    next_kept_frame,
    normalize_cuts,
)

FPS_TOLERANCE = 0.01
DEFAULT_ASPECT_RATIO = 16 / 9


@dataclass(frozen=True)
class ClipSource:
    """Facts about one video file."""

    path: Path
    frame_count: int
    fps: float
    width: int = 0
    height: int = 0

    @property
    def last_frame(self) -> int:
        return max(0, self.frame_count - 1)

    @property
    def full_range(self) -> FrameRange:
        return FrameRange(0, self.last_frame)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.width > 0 and self.height > 0 else DEFAULT_ASPECT_RATIO


@dataclass(frozen=True)
class Clip:
    """A trimmed piece of a source (``range`` in source frames, inclusive) with cuts and settings of its own."""

    source: ClipSource
    range: FrameRange
    cuts: tuple[FrameRange, ...] = ()
    settings: RenderSettings = field(default_factory=RenderSettings)
    stabilization_reference_frame: int | None = None
    """Source frame the camera-lock reference was taken from; the export decodes it."""

    def __post_init__(self) -> None:
        trimmed = as_range(self.range).clamped(self.source.full_range)
        if trimmed is None:
            raise ValueError(f"clip range {self.range} lies outside the source ({self.source.full_range})")
        object.__setattr__(self, "range", trimmed)
        object.__setattr__(self, "cuts", normalize_cuts(self.cuts, trimmed))

    @classmethod
    def whole(cls, source: ClipSource, **fields: object) -> Clip:
        return cls(source, source.full_range, **fields)  # type: ignore[arg-type]

    @property
    def fps(self) -> float:
        return self.source.fps

    @property
    def timeline_count(self) -> int:
        """Frames shown on the timeline for this clip (cuts included)."""
        return self.range.count

    @property
    def segments(self) -> tuple[FrameRange, ...]:
        return kept_segments(self.range, self.cuts)

    @property
    def export_count(self) -> int:
        return frame_count(self.segments)

    @property
    def duration_seconds(self) -> float:
        return self.export_count / self.fps if self.fps > 0 else 0.0

    def with_cuts(self, cuts: Iterable[RangeLike]) -> Clip:
        return replace(self, cuts=tuple(as_range(cut) for cut in cuts))

    def with_settings(self, settings: RenderSettings) -> Clip:
        return replace(self, settings=settings)

    def with_range(self, start: int, end: int) -> Clip:
        """Re-trim; cuts are clamped to the new range."""
        return replace(self, range=FrameRange(min(start, end), max(start, end)))

    def with_reference(self, source_frame: int | None) -> Clip:
        return replace(self, stabilization_reference_frame=source_frame)

    def split_at(self, source_frame: int) -> tuple[Clip, Clip] | None:
        """Two clips — up to ``source_frame - 1`` and from ``source_frame`` on — or ``None`` at the edges."""
        if not self.range.start < source_frame <= self.range.end:
            return None
        left = replace(self, range=FrameRange(self.range.start, source_frame - 1))
        right = replace(self, range=FrameRange(source_frame, self.range.end))
        return left, right


@dataclass(frozen=True)
class Location:
    clip_index: int
    source_frame: int


@dataclass(frozen=True)
class ExportSegment:
    """A run of kept source frames of one clip, in export order."""

    clip_index: int
    clip: Clip
    frames: FrameRange

    @property
    def seconds(self) -> tuple[float, float]:
        """Source time ``(start, end)``; ``end`` is exclusive."""
        fps = self.clip.fps
        if fps <= 0:
            return (0.0, 0.0)
        return (self.frames.start / fps, (self.frames.end + 1) / fps)


@dataclass(frozen=True)
class Sequence:
    clips: tuple[Clip, ...] = ()

    @classmethod
    def single(
        cls,
        source: ClipSource,
        clip_range: FrameRange | None = None,
        *,
        cuts: Iterable[RangeLike] = (),
        settings: RenderSettings | None = None,
        stabilization_reference_frame: int | None = None,
    ) -> Sequence:
        """The one-video case: a sequence of one clip."""
        clip = Clip(
            source,
            clip_range if clip_range is not None else source.full_range,
            tuple(as_range(cut) for cut in cuts),
            settings if settings is not None else RenderSettings(),
            stabilization_reference_frame,
        )
        return cls((clip,))

    def __len__(self) -> int:
        return len(self.clips)

    @property
    def is_empty(self) -> bool:
        return not self.clips

    @property
    def fps(self) -> float:
        return self.clips[0].fps if self.clips else 30.0

    @property
    def timeline_count(self) -> int:
        return sum(clip.timeline_count for clip in self.clips)

    @property
    def last_timeline_frame(self) -> int:
        return max(0, self.timeline_count - 1)

    @property
    def export_count(self) -> int:
        return sum(clip.export_count for clip in self.clips)

    @property
    def duration_seconds(self) -> float:
        return sum(clip.duration_seconds for clip in self.clips)

    @property
    def source_paths(self) -> tuple[Path, ...]:
        """Distinct source files in order of first use (the ffmpeg inputs)."""
        return tuple(dict.fromkeys(clip.source.path for clip in self.clips))

    def fps_mismatch(self) -> str | None:
        """A message when clips disagree on fps (v1 keeps one fps for the whole sequence), else ``None``."""
        for clip in self.clips[1:]:
            if abs(clip.fps - self.fps) > FPS_TOLERANCE:
                return f"Os clipes precisam ter o mesmo fps: {self.fps:.2f} e {clip.fps:.2f} ({clip.source.path.name})."
        return None

    # --- timeline coordinates -------------------------------------------------------------
    def clip_offsets(self) -> tuple[int, ...]:
        offsets = []
        total = 0
        for clip in self.clips:
            offsets.append(total)
            total += clip.timeline_count
        return tuple(offsets)

    def clip_span(self, index: int) -> FrameRange:
        """Timeline frames covered by clip ``index``."""
        offset = self.clip_offsets()[index]
        return FrameRange(offset, offset + self.clips[index].timeline_count - 1)

    def clip_at(self, timeline_frame: int) -> int:
        if not self.clips:
            raise IndexError("empty sequence")
        offsets = self.clip_offsets()
        index = bisect.bisect_right(offsets, max(0, timeline_frame)) - 1
        return max(0, min(index, len(self.clips) - 1))

    def locate(self, timeline_frame: int) -> Location:
        index = self.clip_at(timeline_frame)
        clip = self.clips[index]
        local = max(0, min(timeline_frame - self.clip_offsets()[index], clip.timeline_count - 1))
        return Location(index, clip.range.start + local)

    def timeline_frame(self, clip_index: int, source_frame: int) -> int:
        clip = self.clips[clip_index]
        local = max(0, min(source_frame - clip.range.start, clip.timeline_count - 1))
        return self.clip_offsets()[clip_index] + local

    def timeline_cuts(self) -> tuple[FrameRange, ...]:
        """Every clip's cuts expressed in timeline frames (adjacent ones across a boundary merge)."""
        shifted: list[FrameRange] = []
        for index, clip in enumerate(self.clips):
            offset = self.clip_offsets()[index] - clip.range.start
            shifted.extend(FrameRange(cut.start + offset, cut.end + offset) for cut in clip.cuts)
        return normalize_cuts(shifted)

    def is_kept(self, timeline_frame: int) -> bool:
        return cut_at(timeline_frame, self.timeline_cuts()) is None

    def next_kept_frame(self, timeline_frame: int, last_frame: int | None = None) -> int | None:
        last = self.last_timeline_frame if last_frame is None else min(last_frame, self.last_timeline_frame)
        return next_kept_frame(timeline_frame, self.timeline_cuts(), last)

    def export_segments(self) -> tuple[ExportSegment, ...]:
        return tuple(
            ExportSegment(index, clip, segment) for index, clip in enumerate(self.clips) for segment in clip.segments
        )

    # --- editing ----------------------------------------------------------------------------
    def with_timeline_cuts(self, cuts: Iterable[RangeLike]) -> Sequence:
        """Replace every clip's cuts from ranges in timeline frames (a range may span several clips)."""
        if not self.clips:
            return self
        normalized = normalize_cuts(cuts, FrameRange(0, self.last_timeline_frame))
        offsets = self.clip_offsets()
        clips = []
        for index, clip in enumerate(self.clips):
            span = self.clip_span(index)
            shift = clip.range.start - offsets[index]
            local = []
            for cut in normalized:
                part = cut.clamped(span)
                if part is not None:
                    local.append(FrameRange(part.start + shift, part.end + shift))
            clips.append(clip.with_cuts(local))
        return Sequence(tuple(clips))

    def add(self, clip: Clip, index: int | None = None) -> Sequence:
        clips = list(self.clips)
        clips.insert(len(clips) if index is None else index, clip)
        return Sequence(tuple(clips))

    def remove(self, index: int) -> Sequence:
        clips = list(self.clips)
        del clips[index]
        return Sequence(tuple(clips))

    def move(self, index: int, new_index: int) -> Sequence:
        clips = list(self.clips)
        clip = clips.pop(index)
        clips.insert(max(0, min(new_index, len(clips))), clip)
        return Sequence(tuple(clips))

    def replace_clip(self, index: int, clip: Clip) -> Sequence:
        clips = list(self.clips)
        clips[index] = clip
        return Sequence(tuple(clips))

    def with_clip_settings(self, index: int, settings: RenderSettings) -> Sequence:
        return self.replace_clip(index, self.clips[index].with_settings(settings))

    def split(self, timeline_frame: int) -> Sequence | None:
        """Split the clip under ``timeline_frame`` so that frame starts the right half; ``None`` at clip edges."""
        if not self.clips:
            return None
        location = self.locate(timeline_frame)
        halves = self.clips[location.clip_index].split_at(location.source_frame)
        if halves is None:
            return None
        clips = list(self.clips)
        clips[location.clip_index : location.clip_index + 1] = list(halves)
        return Sequence(tuple(clips))
