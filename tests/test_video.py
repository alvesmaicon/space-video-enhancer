from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from space_video_enhancer.video import GRAB_FORWARD_LIMIT, VideoOpenError, VideoSource

FRAMES = 60
WIDTH, HEIGHT = 128, 32
BAR_WIDTH = 2


def _write_marker_video(path: Path) -> None:
    """Frame ``i`` carries a white bar at x = 2*i; lossy codecs keep edge positions, not gray levels."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for index in range(FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, index * BAR_WIDTH : (index + 1) * BAR_WIDTH] = 255
        writer.write(frame)
    writer.release()


def _index_of(frame: np.ndarray) -> int:
    column_brightness = frame.mean(axis=(0, 2))
    return int(np.argmax(column_brightness)) // BAR_WIDTH


@pytest.fixture
def marker_video(tmp_path: Path) -> Iterator[VideoSource]:
    path = tmp_path / "markers.mp4"
    _write_marker_video(path)
    source = VideoSource(path)
    yield source
    source.release()


def test_open_reports_metadata(marker_video: VideoSource) -> None:
    assert marker_video.frame_count == FRAMES
    assert marker_video.size == (WIDTH, HEIGHT)
    assert marker_video.fps == pytest.approx(24.0)


def test_sequential_reads_avoid_seeking(marker_video: VideoSource) -> None:
    for index in range(10):
        frame = marker_video.read_frame(index)
        assert frame is not None and _index_of(frame) == index
    assert marker_video.seeks == 0
    assert marker_video.sequential_reads == 10


def test_short_forward_jump_grabs_instead_of_seeking(marker_video: VideoSource) -> None:
    marker_video.read_frame(0)
    frame = marker_video.read_frame(7)
    assert frame is not None and _index_of(frame) == 7
    assert marker_video.seeks == 0


def test_backward_and_far_jumps_seek(marker_video: VideoSource) -> None:
    marker_video.read_frame(10)  # forward from the start: grabbed, not seeked
    back = marker_video.read_frame(3)
    assert back is not None and _index_of(back) == 3
    far_index = 3 + GRAB_FORWARD_LIMIT + 5
    far = marker_video.read_frame(far_index)
    assert far is not None and _index_of(far) == far_index
    assert marker_video.seeks == 2


def test_index_is_clamped(marker_video: VideoSource) -> None:
    last = marker_video.read_frame(10_000)
    assert last is not None and _index_of(last) == FRAMES - 1
    first = marker_video.read_frame(-5)
    assert first is not None and _index_of(first) == 0


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(VideoOpenError):
        VideoSource(tmp_path / "nope.mp4")
