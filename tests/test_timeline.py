from __future__ import annotations

from space_video_enhancer.timeline import (
    FrameRange,
    cut_at,
    frame_count,
    kept_segments,
    next_kept_frame,
    normalize_cuts,
    segments_seconds,
)


def test_normalize_sorts_clamps_and_merges() -> None:
    bounds = FrameRange(0, 99)
    cuts = normalize_cuts([(40, 50), (10, 20), (21, 25), (48, 60), (-5, 3), (120, 130), (70, 70)], bounds)
    assert cuts == (FrameRange(0, 3), FrameRange(10, 25), FrameRange(40, 60), FrameRange(70, 70))
    assert normalize_cuts([(30, 10)]) == (FrameRange(10, 30),)  # reversed pairs are fine
    assert normalize_cuts([]) == ()


def test_kept_segments_is_the_clip_minus_the_cuts() -> None:
    clip = FrameRange(0, 99)
    assert kept_segments(clip, []) == (clip,)
    segments = kept_segments(clip, [(10, 19), (50, 59)])
    assert segments == (FrameRange(0, 9), FrameRange(20, 49), FrameRange(60, 99))
    assert frame_count(segments) == 80
    assert kept_segments(clip, [(0, 4), (95, 99)]) == (FrameRange(5, 94),)
    assert kept_segments(clip, [(0, 99)]) == ()
    assert kept_segments(FrameRange(20, 30), [(0, 100)]) == ()
    assert kept_segments(FrameRange(20, 30), [(0, 10)]) == (FrameRange(20, 30),)  # cut outside the clip


def test_next_kept_frame_skips_cuts_for_playback() -> None:
    cuts = [(10, 19), (50, 59)]
    assert next_kept_frame(5, cuts, 99) == 5
    assert next_kept_frame(10, cuts, 99) == 20
    assert next_kept_frame(15, cuts, 99) == 20
    assert next_kept_frame(55, cuts, 99) == 60
    assert next_kept_frame(55, cuts, 58) is None  # the cut runs past the end of playback
    assert next_kept_frame(100, cuts, 99) is None
    assert cut_at(12, cuts) == FrameRange(10, 19) and cut_at(20, cuts) is None


def test_segments_seconds_uses_exclusive_ends() -> None:
    assert segments_seconds((FrameRange(0, 23), FrameRange(48, 71)), 24.0) == ((0.0, 1.0), (2.0, 3.0))
    assert segments_seconds((FrameRange(0, 5),), 0.0) == ((0.0, 0.0),)
