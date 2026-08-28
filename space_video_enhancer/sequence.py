"""A sequence of clips on the timeline: trimmed pieces of source files, each with its own cuts and settings.

Two coordinate systems. *Timeline frames* run over every clip's trimmed range back to back,
cuts included — the timeline shows removed ranges as red bands that stay navigable. *Export
frames* are what leaves the app: timeline frames minus the cuts. Every clip carries its own
:class:`RenderSettings`, so "different adjustments per part" is just splitting a clip in two.
Everything here is pure and immutable; editing operations return a new :class:`Sequence`.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from space_video_enhancer.settings import RenderSettings, RetimingMode
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


def _rates_match(source_fps: float, timeline_fps: float) -> bool:
    """True when no resampling is needed: same rate (or a rate we cannot use), so timeline == source frames 1:1."""
    return source_fps <= 0 or timeline_fps <= 0 or abs(source_fps - timeline_fps) <= FPS_TOLERANCE


def timeline_length(source_count: int, source_fps: float, timeline_fps: float) -> int:
    """How many timeline frames a run of ``source_count`` source frames occupies at ``timeline_fps``.

    Nearest-frame resampling preserves the run's *duration*: a 24 fps second (24 frames) becomes 30 on a
    30 fps timeline. A non-empty run never vanishes (at least one frame), so a kept frame is always shown.
    """
    if source_count <= 0:
        return 0
    if _rates_match(source_fps, timeline_fps):
        return source_count
    return max(1, round(source_count * timeline_fps / source_fps))


def source_offset(local_timeline: int, source_fps: float, timeline_fps: float) -> int:
    """The clip-local source frame nearest ``local_timeline`` timeline frames in (nearest-frame resample)."""
    if _rates_match(source_fps, timeline_fps):
        return local_timeline
    return round(local_timeline * source_fps / timeline_fps)


def timeline_offset(local_source: int, source_fps: float, timeline_fps: float) -> int:
    """Inverse of :func:`source_offset`: the timeline frame nearest ``local_source`` source frames in."""
    if _rates_match(source_fps, timeline_fps):
        return local_source
    return round(local_source * timeline_fps / source_fps)


@dataclass(frozen=True)
class RetimeSample:
    """Which source frame(s) an output frame is built from when a clip is retimed to the timeline.

    ``a`` and ``b`` are absolute source frame indices; the output is ``(1 - weight)·a + weight·b`` (blended
    or motion-interpolated per ``mode``). ``a == b`` (or a zero weight, or ``NEAREST``) means a single frame.
    """

    a: int
    b: int
    weight: float
    mode: RetimingMode = RetimingMode.NEAREST

    @property
    def is_single(self) -> bool:
        return self.mode is RetimingMode.NEAREST or self.b == self.a or self.weight <= 0.0


def _retime_sample(first: int, last: int, position: float, mode: RetimingMode) -> RetimeSample:
    """The sample for an output frame ``position`` source frames past ``first`` (clamped to ``[first, last]``)."""
    exact = first + position
    if mode is RetimingMode.NEAREST:
        frame = min(last, max(first, round(exact)))
        return RetimeSample(frame, frame, 0.0, mode)
    low = math.floor(exact)
    a = min(last, max(first, low))
    b = min(last, a + 1)
    weight = 0.0 if b == a else min(1.0, max(0.0, exact - low))
    if weight <= 1e-6:  # an exact hit (or the clamped edge): a single frame, no in-between
        return RetimeSample(a, a, 0.0, mode)
    if weight >= 1.0 - 1e-6:
        return RetimeSample(b, b, 0.0, mode)
    return RetimeSample(a, b, weight, mode)


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
    exposure_reference_frame: int | None = None
    """Source frame the exposure-lock reference (``settings.exposure.reference``) was metered on."""

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

    def with_exposure_reference(self, source_frame: int | None) -> Clip:
        return replace(self, exposure_reference_frame=source_frame)

    @property
    def name(self) -> str:
        return self.source.path.name

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
class PlaybackRun:
    """Kept frames between two timeline positions: where the run sits on the timeline and what it shows."""

    clip_index: int
    clip: Clip
    span: FrameRange
    """Timeline frames."""
    frames: FrameRange
    """Source frames of ``clip`` shown over ``span``."""

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
    timeline_fps: float | None = None
    """The master timeline rate; clips of another fps are retimed to it. ``None`` = the first clip's fps."""

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
        if self.timeline_fps and self.timeline_fps > 0:
            return self.timeline_fps
        return self.clips[0].fps if self.clips else 30.0

    def _clip_length(self, clip: Clip) -> int:
        """Timeline frames clip ``clip`` occupies (its whole trimmed range, cuts included), retimed to the timeline."""
        return timeline_length(clip.timeline_count, clip.fps, self.fps)

    def export_segment_length(self, segment: ExportSegment) -> int:
        """Output frames a kept run yields after retiming — what the export writes and the encoder receives."""
        return timeline_length(segment.frames.count, segment.clip.fps, self.fps)

    def resampled_export_frames(self, segment: ExportSegment) -> tuple[int, ...]:
        """The source frame index behind each output frame of ``segment`` (nearest-frame; may repeat or skip)."""
        clip = segment.clip
        start, end = segment.frames.start, segment.frames.end
        length = self.export_segment_length(segment)
        if length == segment.frames.count:
            return tuple(range(start, end + 1))
        return tuple(min(end, start + source_offset(j, clip.fps, self.fps)) for j in range(length))

    def export_samples(self, segment: ExportSegment) -> tuple[RetimeSample, ...]:
        """One :class:`RetimeSample` per output frame of ``segment``, honouring the clip's retiming mode."""
        clip = segment.clip
        first, last = segment.frames.start, segment.frames.end
        mode = clip.settings.capture.retiming
        if _rates_match(clip.fps, self.fps):
            return tuple(RetimeSample(frame, frame, 0.0, mode) for frame in range(first, last + 1))
        ratio = clip.fps / self.fps
        length = self.export_segment_length(segment)
        return tuple(_retime_sample(first, last, index * ratio, mode) for index in range(length))

    def preview_sample(self, timeline_frame: int) -> tuple[int, RetimeSample]:
        """Clip index and the sample the preview should show at ``timeline_frame`` (its ``a`` is :meth:`locate`)."""
        index = self.clip_at(timeline_frame)
        clip = self.clips[index]
        local = max(0, min(timeline_frame - self.clip_offsets()[index], self._clip_length(clip) - 1))
        mode = clip.settings.capture.retiming
        if _rates_match(clip.fps, self.fps):
            source = min(clip.range.start + local, clip.range.end)
            return index, RetimeSample(source, source, 0.0, mode)
        return index, _retime_sample(clip.range.start, clip.range.end, local * clip.fps / self.fps, mode)

    @property
    def timeline_count(self) -> int:
        return sum(self._clip_length(clip) for clip in self.clips)

    @property
    def last_timeline_frame(self) -> int:
        return max(0, self.timeline_count - 1)

    @property
    def export_count(self) -> int:
        return sum(self.export_segment_length(segment) for segment in self.export_segments())

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
            total += self._clip_length(clip)
        return tuple(offsets)

    def clip_span(self, index: int) -> FrameRange:
        """Timeline frames covered by clip ``index``."""
        offset = self.clip_offsets()[index]
        return FrameRange(offset, offset + self._clip_length(self.clips[index]) - 1)

    def clip_at(self, timeline_frame: int) -> int:
        if not self.clips:
            raise IndexError("empty sequence")
        offsets = self.clip_offsets()
        index = bisect.bisect_right(offsets, max(0, timeline_frame)) - 1
        return max(0, min(index, len(self.clips) - 1))

    def locate(self, timeline_frame: int) -> Location:
        index = self.clip_at(timeline_frame)
        clip = self.clips[index]
        local = max(0, min(timeline_frame - self.clip_offsets()[index], self._clip_length(clip) - 1))
        source_local = min(source_offset(local, clip.fps, self.fps), clip.timeline_count - 1)
        return Location(index, clip.range.start + source_local)

    def timeline_frame(self, clip_index: int, source_frame: int) -> int:
        clip = self.clips[clip_index]
        source_local = max(0, min(source_frame - clip.range.start, clip.timeline_count - 1))
        local = min(timeline_offset(source_local, clip.fps, self.fps), self._clip_length(clip) - 1)
        return self.clip_offsets()[clip_index] + local

    def timeline_cuts(self) -> tuple[FrameRange, ...]:
        """Every clip's cuts expressed in timeline frames (adjacent ones across a boundary merge)."""
        offsets = self.clip_offsets()
        shifted: list[FrameRange] = []
        for index, clip in enumerate(self.clips):
            base = offsets[index]
            for cut in clip.cuts:
                start = base + timeline_offset(cut.start - clip.range.start, clip.fps, self.fps)
                end = base + timeline_offset(cut.end - clip.range.start, clip.fps, self.fps)
                shifted.append(FrameRange(start, end))
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

    def playback_runs(self, start: int, end: int) -> tuple[PlaybackRun, ...]:
        """The kept runs within timeline frames ``[start, end]``, in playback order (video and sound follow them)."""
        window = FrameRange(min(start, end), max(start, end))
        offsets = self.clip_offsets()
        runs = []
        for segment in self.export_segments():
            clip = segment.clip
            base = offsets[segment.clip_index]
            tl_start = base + timeline_offset(segment.frames.start - clip.range.start, clip.fps, self.fps)
            tl_end = base + timeline_offset(segment.frames.end - clip.range.start, clip.fps, self.fps)
            span = FrameRange(tl_start, tl_end).clamped(window)
            if span is not None:
                src_start = clip.range.start + source_offset(span.start - base, clip.fps, self.fps)
                src_end = clip.range.start + source_offset(span.end - base, clip.fps, self.fps)
                frames = FrameRange(min(src_start, src_end), max(src_start, src_end))
                runs.append(PlaybackRun(segment.clip_index, clip, span, frames))
        return tuple(runs)

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
            base = offsets[index]
            local = []
            for cut in normalized:
                part = cut.clamped(span)
                if part is not None:
                    src_start = clip.range.start + source_offset(part.start - base, clip.fps, self.fps)
                    src_end = clip.range.start + source_offset(part.end - base, clip.fps, self.fps)
                    local.append(FrameRange(min(src_start, src_end), max(src_start, src_end)))
            clips.append(clip.with_cuts(local))
        return replace(self, clips=tuple(clips))

    def add(self, clip: Clip, index: int | None = None) -> Sequence:
        clips = list(self.clips)
        clips.insert(len(clips) if index is None else index, clip)
        return replace(self, clips=tuple(clips))

    def remove(self, index: int) -> Sequence:
        clips = list(self.clips)
        del clips[index]
        return replace(self, clips=tuple(clips))

    def move(self, index: int, new_index: int) -> Sequence:
        clips = list(self.clips)
        clip = clips.pop(index)
        clips.insert(max(0, min(new_index, len(clips))), clip)
        return replace(self, clips=tuple(clips))

    def replace_clip(self, index: int, clip: Clip) -> Sequence:
        clips = list(self.clips)
        clips[index] = clip
        return replace(self, clips=tuple(clips))

    def with_clip_settings(self, index: int, settings: RenderSettings) -> Sequence:
        return self.replace_clip(index, self.clips[index].with_settings(settings))

    def without_empty_clips(self) -> Sequence:
        """Drop clips whose cuts remove every frame (they contribute nothing to the export)."""
        return replace(self, clips=tuple(clip for clip in self.clips if clip.export_count > 0))

    def for_export(self, start: int | None = None, end: int | None = None) -> Sequence:
        """What leaves the app: the timeline outside ``[start, end]`` becomes cuts, empty clips are dropped."""
        if not self.clips:
            return self
        last = self.last_timeline_frame
        first = 0 if start is None else max(0, min(start, last))
        final = last if end is None else max(first, min(end, last))
        outside: list[FrameRange] = []
        if first > 0:
            outside.append(FrameRange(0, first - 1))
        if final < last:
            outside.append(FrameRange(final + 1, last))
        return self.with_timeline_cuts([*self.timeline_cuts(), *outside]).without_empty_clips()

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
        return replace(self, clips=tuple(clips))
