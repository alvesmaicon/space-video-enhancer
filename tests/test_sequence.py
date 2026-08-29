"""The sequence model: clips with cuts and settings, timeline <-> source mapping, editing operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from space_video_enhancer.sequence import Clip, ClipSource, Location, RetimeSample, Sequence
from space_video_enhancer.settings import CaptureSettings, FilterSettings, RenderSettings, RetimingMode
from space_video_enhancer.timeline import FrameRange

A = ClipSource(Path("a.mp4"), 100, 30.0, 1920, 1080)
B = ClipSource(Path("b.mp4"), 50, 30.0, 1280, 720)


def test_clip_normalizes_its_range_and_cuts() -> None:
    clip = Clip(A, FrameRange(80, 10), cuts=(FrameRange(5, 12), FrameRange(70, 90), FrameRange(30, 40)))
    assert clip.range == FrameRange(10, 80)
    assert clip.cuts == (FrameRange(10, 12), FrameRange(30, 40), FrameRange(70, 80))
    assert clip.timeline_count == 71 and clip.export_count == 71 - 3 - 11 - 11
    assert clip.segments == (FrameRange(13, 29), FrameRange(41, 69))
    assert clip.duration_seconds == pytest.approx(clip.export_count / 30.0)
    assert Clip.whole(A).range == FrameRange(0, 99) and A.aspect_ratio == pytest.approx(16 / 9)
    assert ClipSource(Path("x"), 10, 30.0).aspect_ratio == pytest.approx(16 / 9)  # unknown size
    with pytest.raises(ValueError):
        Clip(A, FrameRange(100, 120))
    retrimmed = clip.with_range(35, 75)
    assert retrimmed.range == FrameRange(35, 75) and retrimmed.cuts == (FrameRange(35, 40), FrameRange(70, 75))


def test_split_keeps_settings_and_clamps_cuts() -> None:
    bright = RenderSettings(filters=FilterSettings(brightness=80))
    clip = Clip(A, FrameRange(0, 99), cuts=(FrameRange(45, 55),), settings=bright, stabilization_reference_frame=3)
    halves = clip.split_at(50)
    assert halves is not None
    left, right = halves
    assert left.range == FrameRange(0, 49) and right.range == FrameRange(50, 99)
    assert left.cuts == (FrameRange(45, 49),) and right.cuts == (FrameRange(50, 55),)
    assert left.settings is bright and right.settings is bright
    assert left.stabilization_reference_frame == right.stabilization_reference_frame == 3
    assert clip.split_at(0) is None and clip.split_at(100) is None and clip.split_at(99) is not None


def _sequence() -> Sequence:
    return Sequence(
        (
            Clip(A, FrameRange(10, 39), cuts=(FrameRange(20, 24),)),  # 30 timeline frames, 25 exported
            Clip(B, FrameRange(0, 19)),  # 20 frames
            Clip(A, FrameRange(60, 69), cuts=(FrameRange(60, 62),)),  # 10 frames, 7 exported
        )
    )


def test_sequence_counts_offsets_and_mapping() -> None:
    sequence = _sequence()
    assert sequence.timeline_count == 60 and sequence.export_count == 52 and len(sequence) == 3
    assert sequence.clip_offsets() == (0, 30, 50) and sequence.clip_span(1) == FrameRange(30, 49)
    assert sequence.fps == 30.0 and sequence.duration_seconds == pytest.approx(52 / 30.0)
    assert sequence.source_paths == (Path("a.mp4"), Path("b.mp4"))
    assert sequence.locate(0) == Location(0, 10) and sequence.locate(29) == Location(0, 39)
    assert sequence.locate(30) == Location(1, 0) and sequence.locate(59) == Location(2, 69)
    assert sequence.locate(-5) == Location(0, 10) and sequence.locate(1000) == Location(2, 69)
    for frame in (0, 17, 29, 30, 45, 50, 59):
        location = sequence.locate(frame)
        assert sequence.timeline_frame(location.clip_index, location.source_frame) == frame
    assert sequence.clip_at(49) == 1 and sequence.clip_at(50) == 2
    assert sequence.timeline_cuts() == (FrameRange(10, 14), FrameRange(50, 52))
    assert not sequence.is_kept(12) and sequence.is_kept(15)
    assert (
        sequence.next_kept_frame(10) == 15 and sequence.next_kept_frame(49) == 49 and sequence.next_kept_frame(50) == 53
    )
    assert sequence.next_kept_frame(50, last_frame=52) is None
    segments = sequence.export_segments()
    assert [(s.clip_index, s.frames.start, s.frames.end) for s in segments] == [
        (0, 10, 19),
        (0, 25, 39),
        (1, 0, 19),
        (2, 63, 69),
    ]
    assert segments[0].seconds == (10 / 30.0, 20 / 30.0)
    assert sequence.fps_mismatch() is None
    slow = sequence.add(Clip(ClipSource(Path("c.mp4"), 10, 25.0), FrameRange(0, 9)))
    assert slow.fps_mismatch() is not None and "25.00" in slow.fps_mismatch()


def test_timeline_cuts_are_distributed_over_the_clips_they_span() -> None:
    sequence = _sequence().with_timeline_cuts([(25, 34), (58, 70)])
    assert sequence.clips[0].cuts == (FrameRange(35, 39),)  # timeline 25..29 -> source 35..39 (old cut replaced)
    assert sequence.clips[1].cuts == (FrameRange(0, 4),)  # timeline 30..34 -> source 0..4
    assert sequence.clips[2].cuts == (FrameRange(68, 69),)  # clamped to the end of the sequence
    assert sequence.timeline_cuts() == (FrameRange(25, 34), FrameRange(58, 59))
    assert sequence.with_timeline_cuts([]).export_count == sequence.timeline_count
    assert Sequence().with_timeline_cuts([(0, 1)]).is_empty


def test_editing_operations_return_new_sequences() -> None:
    sequence = _sequence()
    moved = sequence.move(2, 0)
    assert [clip.range.start for clip in moved.clips] == [60, 10, 0] and sequence.clips[0].range.start == 10
    removed = sequence.remove(1)
    assert len(removed) == 2 and removed.timeline_count == 40
    inserted = sequence.add(Clip.whole(B), 1)
    assert len(inserted) == 4 and inserted.clips[1].range == FrameRange(0, 49)
    bright = RenderSettings(filters=FilterSettings(brightness=90))
    tuned = sequence.with_clip_settings(1, bright)
    assert tuned.clips[1].settings is bright and sequence.clips[1].settings == RenderSettings()
    split = sequence.split(35)  # inside clip 1 at source frame 5
    assert split is not None and len(split) == 4
    assert split.clips[1].range == FrameRange(0, 4) and split.clips[2].range == FrameRange(5, 19)
    assert split.timeline_count == sequence.timeline_count and split.export_count == sequence.export_count
    assert sequence.split(30) is None, "a clip boundary cannot be split again"
    assert Sequence().split(0) is None
    single = Sequence.single(A, FrameRange(5, 9), cuts=[(6, 6)], settings=bright, stabilization_reference_frame=7)
    assert single.export_count == 4 and single.clips[0].settings is bright
    assert single.clips[0].stabilization_reference_frame == 7 and Sequence.single(A).timeline_count == 100


def test_for_export_turns_the_range_into_cuts_and_drops_empty_clips() -> None:
    a = ClipSource(Path("a.mp4"), 10, 24.0)
    b = ClipSource(Path("b.mp4"), 10, 24.0)
    sequence = Sequence((Clip.whole(a, cuts=(FrameRange(2, 3),)), Clip.whole(b)))
    exported = sequence.for_export(5, 14)  # timeline frames 5..14: a[5..9] + b[0..4]
    assert [clip.segments for clip in exported.clips] == [(FrameRange(5, 9),), (FrameRange(0, 4),)]
    assert exported.export_count == 10 and exported.duration_seconds == pytest.approx(10 / 24)
    only_b = sequence.for_export(12, 19)
    assert [clip.source.path.name for clip in only_b.clips] == ["b.mp4"]
    assert only_b.clips[0].segments == (FrameRange(2, 9),)
    assert sequence.for_export().export_count == 18, "no range: everything minus the cuts"
    assert sequence.with_timeline_cuts([(0, 19)]).for_export().is_empty


def test_exposure_reference_and_name_follow_the_clip_through_a_split() -> None:
    source = ClipSource(Path("clip.mp4"), 20, 24.0)
    clip = Clip.whole(source).with_exposure_reference(7).with_reference(3)
    assert clip.name == "clip.mp4"
    left, right = Sequence((clip,)).split(10).clips
    assert (left.exposure_reference_frame, right.exposure_reference_frame) == (7, 7)
    assert (left.stabilization_reference_frame, right.stabilization_reference_frame) == (3, 3)
    assert left.with_exposure_reference(None).exposure_reference_frame is None


C60 = ClipSource(Path("c.mp4"), 60, 60.0, 1920, 1080)
D24 = ClipSource(Path("d.mp4"), 24, 24.0, 1920, 1080)


def _mixed_sequence() -> Sequence:
    """A 30 fps timeline: a 30 fps clip, then a 60 fps one (halved) and a 24 fps one (stretched)."""
    return Sequence(
        (
            Clip(A, FrameRange(0, 29)),  # 30 fps -> 30 timeline frames (1:1)
            Clip(C60, FrameRange(0, 59)),  # 60 fps, 60 source -> 30 timeline frames
            Clip(D24, FrameRange(0, 23)),  # 24 fps, 24 source -> 30 timeline frames
        )
    )


def test_timeline_fps_defaults_to_the_first_clip() -> None:
    sequence = _mixed_sequence()
    assert sequence.fps == 30.0  # from the first clip, no resampling of it
    assert sequence.clip_offsets() == (0, 30, 60)
    assert sequence.timeline_count == 90  # 30 + 30 + 30, all retimed to 30 fps


def test_locate_retimes_each_clip_to_the_timeline_rate() -> None:
    sequence = _mixed_sequence()
    # The 60 fps clip: timeline frame 30 -> source 0, 59 -> source ~58 (every other source frame).
    assert sequence.locate(30) == Location(1, 0)
    assert sequence.locate(45) == Location(1, 30)
    assert sequence.locate(59) == Location(1, 58)
    # The 24 fps clip: timeline frame 60 -> source 0, 89 -> source 23 (some source frames repeat).
    assert sequence.locate(60) == Location(2, 0)
    assert sequence.locate(89) == Location(2, 23)
    # Round-trips land back on the timeline frame the source frame first appears at.
    for frame in (0, 15, 30, 45, 60, 75, 89):
        location = sequence.locate(frame)
        assert sequence.locate(sequence.timeline_frame(location.clip_index, location.source_frame)) == location


def test_export_resamples_each_clip_and_preserves_duration() -> None:
    sequence = _mixed_sequence()
    segments = sequence.export_segments()
    # 30 fps clip: 1:1. 60 fps clip: halved. 24 fps clip: stretched. Each ends at 30 output frames.
    assert [sequence.export_segment_length(s) for s in segments] == [30, 30, 30]
    assert sequence.export_count == 90
    halved = sequence.resampled_export_frames(segments[1])
    assert halved == tuple(range(0, 60, 2))  # every other source frame of the 60 fps clip
    stretched = sequence.resampled_export_frames(segments[2])
    assert len(stretched) == 30 and stretched[0] == 0 and stretched[-1] == 23
    assert stretched == tuple(sorted(stretched))  # monotonic; some source frames repeat
    assert max(stretched.count(f) for f in set(stretched)) == 2  # a frame is shown at most twice
    # Duration is preserved: each clip keeps its own length in seconds.
    assert segments[1].seconds == (0.0, 60 / 60.0)  # 1 s of 60 fps footage
    assert segments[2].seconds == (0.0, 24 / 24.0)  # 1 s of 24 fps footage


def test_single_fps_export_frames_are_untouched() -> None:
    sequence = _sequence()  # all 30 fps
    for segment in sequence.export_segments():
        frames = sequence.resampled_export_frames(segment)
        assert frames == tuple(range(segment.frames.start, segment.frames.end + 1))


def _retimed(source: ClipSource, mode: RetimingMode) -> Clip:
    return Clip.whole(source, settings=RenderSettings(capture=CaptureSettings(retiming=mode)))


def test_export_samples_blend_straddles_the_two_neighbouring_frames() -> None:
    sequence = Sequence((Clip.whole(A), _retimed(D24, RetimingMode.BLEND)))  # 24 fps clip on a 30 fps timeline
    samples = sequence.export_samples(sequence.export_segments()[1])
    assert len(samples) == 30 and sequence.export_segment_length(sequence.export_segments()[1]) == 30
    assert samples[0] == RetimeSample(0, 0, 0.0, RetimingMode.BLEND) and samples[0].is_single  # exact hit on frame 0
    assert samples[1].a == 0 and samples[1].b == 1 and samples[1].weight == pytest.approx(0.8)
    assert samples[2].a == 1 and samples[2].b == 2 and samples[2].weight == pytest.approx(0.6)
    assert samples[5].weight == pytest.approx(0.0) and samples[5].is_single  # 5 * 0.8 = 4.0, an exact hit
    assert samples[-1].b <= D24.last_frame  # never straddles past the end of the clip


def test_export_samples_on_an_integer_downsample_never_blend() -> None:
    sequence = Sequence((Clip.whole(A), _retimed(C60, RetimingMode.OPTICAL_FLOW)))  # 60 -> 30 fps, exactly 2:1
    samples = sequence.export_samples(sequence.export_segments()[1])
    assert [s.a for s in samples] == list(range(0, 60, 2))
    assert all(s.is_single for s in samples), "landing on real frames needs no in-between"


def test_preview_sample_matches_locate_on_the_nearest_frame() -> None:
    first = Clip(A, FrameRange(0, 29))  # 30 frames -> the 24 fps clip starts at timeline frame 30
    blend = RenderSettings(capture=CaptureSettings(retiming=RetimingMode.BLEND))
    sequence = Sequence((first, Clip(D24, FrameRange(0, 23), settings=blend)))
    for frame in (0, 15, 29, 30, 59):
        index, sample = sequence.preview_sample(frame)
        location = sequence.locate(frame)
        assert index == location.clip_index and sample.a == location.source_frame  # `a` is the nearest source frame
    index, sample = sequence.preview_sample(31)  # one frame into the 24 fps clip
    assert index == 1 and sample.mode is RetimingMode.BLEND and sample.a == 0 and sample.b == 1
    assert sample.weight == pytest.approx(0.8)


def test_a_single_fps_clip_never_retimes_even_in_blend_mode() -> None:
    sequence = Sequence((_retimed(A, RetimingMode.OPTICAL_FLOW),))  # one 30 fps clip, timeline 30 fps
    samples = sequence.export_samples(sequence.export_segments()[0])
    assert all(s.is_single and s.a == s.b for s in samples)  # rates match -> nothing to interpolate


def test_clip_speed_stretches_the_timeline_and_the_duration() -> None:
    slow = RenderSettings(capture=CaptureSettings(speed=0.5))
    sequence = Sequence((Clip(A, FrameRange(0, 29), settings=slow),))  # 30 frames of 30 fps footage at half speed
    assert sequence.timeline_count == 60 and sequence.duration_seconds == pytest.approx(2.0)
    segment = sequence.export_segments()[0]
    assert sequence.export_segment_length(segment) == 60
    samples = sequence.export_samples(segment)
    assert len(samples) == 60 and samples[0].a == 0 and samples[-1].a == 29
    assert samples[30].a in (14, 15), "the halfway output frame sits mid-clip"
    # The playhead maps to the right source frame both ways.
    assert sequence.locate(30).source_frame in (14, 15)
    assert sequence.locate(sequence.timeline_frame(0, 15)).source_frame in (14, 15, 16)


def test_double_speed_halves_the_clip() -> None:
    fast = RenderSettings(capture=CaptureSettings(speed=2.0))
    sequence = Sequence((Clip(A, FrameRange(0, 29), settings=fast),))
    assert sequence.timeline_count == 15 and sequence.duration_seconds == pytest.approx(0.5)
    assert [s.a for s in sequence.export_samples(sequence.export_segments()[0])] == list(range(0, 30, 2))
