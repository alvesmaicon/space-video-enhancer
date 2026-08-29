"""Learn the camera path so manual keyframes can be filled with the measured motion (Phase 2).

The manual mode lets the user set a correction at a few frames; :func:`interpolate_keyframes`
draws a *straight line* between them, which leaves the real shake in the middle untouched. Here we
track the clip once (the same robust background-vs-subject estimator the automatic modes use, see
:mod:`global_motion`), accumulate the camera's path relative to the segment start, and use it to
**fill** intermediate keyframes: between two hand-set keyframes the stabilized result now follows a
straight line while the correction cancels the measured wobble in between. Where the motion is
already linear the fill reduces to the old linear interpolation, so it never fights a steady pan.

Everything here is a pure function of the decoded frames (the analysis) or of the keyframes plus a
:class:`MotionPath` (the fill), so the fill produces ordinary keyframes the stateless pipeline and
the parallel export already handle — nothing new travels with a render.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from space_video_enhancer.pipeline.stabilization.estimators import analysis_gray
from space_video_enhancer.pipeline.stabilization.global_motion import estimate_global_motion, new_cell_memory
from space_video_enhancer.settings import StabilizationKeyframe, StabilizationMode


@dataclass(frozen=True)
class MotionPath:
    """The cumulative camera motion per source frame, relative to ``first_frame`` (resolution-free).

    ``dx``/``dy`` are fractions of the frame and ``rotation`` is in degrees — the same units and centre
    convention as a :class:`StabilizationKeyframe`, so a sample can be compared with a hand-set correction
    directly. ``measured[i]`` is ``False`` for a frame the tracker could not follow (the value is held).
    """

    first_frame: int
    dx: np.ndarray
    dy: np.ndarray
    rotation: np.ndarray
    measured: np.ndarray

    @property
    def last_frame(self) -> int:
        return self.first_frame + len(self.dx) - 1

    @property
    def count(self) -> int:
        return len(self.dx)

    @property
    def measured_fraction(self) -> float:
        return float(self.measured.mean()) if self.count else 0.0

    def covers(self, first: int, last: int) -> bool:
        return self.first_frame <= first and last <= self.last_frame

    def sample(self, frame: int) -> tuple[float, float, float]:
        """The cumulative ``(dx, dy, rotation)`` at ``frame`` (clamped into the analysed range)."""
        index = min(max(frame - self.first_frame, 0), self.count - 1)
        return (float(self.dx[index]), float(self.dy[index]), float(self.rotation[index]))


@dataclass(frozen=True)
class MotionPathAnalysis:
    track: MotionPath | None
    cancelled: bool
    frames_read: int
    elapsed_ms: float


def _to_3x3(transform: np.ndarray) -> np.ndarray:
    if transform.shape == (3, 3):
        return transform.astype(np.float64)
    return np.vstack([transform.astype(np.float64), (0.0, 0.0, 1.0)])


def analyze_motion_path(
    read: Callable[[int], np.ndarray | None],
    first: int,
    last: int,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> MotionPathAnalysis:
    """Track frames ``first`` .. ``last`` (inclusive) and accumulate the camera path relative to ``first``.

    ``read(frame_index)`` returns the decoded frame or ``None`` (the stream ended early: the path stops
    there). Frames are only tracked, never rendered, so deinterlacing never enters. ``progress(done, total)``
    is called after each frame; ``cancelled()`` is polled between frames.
    """
    started = time.perf_counter()
    count = max(0, last - first + 1)
    dx = np.zeros(count, dtype=np.float64)
    dy = np.zeros(count, dtype=np.float64)
    rotation = np.zeros(count, dtype=np.float64)
    measured = np.zeros(count, dtype=bool)
    memory = new_cell_memory()
    accumulated = np.eye(3, dtype=np.float64)  # maps the current frame onto ``first`` (analysis pixels)
    previous_gray: np.ndarray | None = None
    frames_read = 0

    for offset in range(count):
        if cancelled is not None and cancelled():
            return MotionPathAnalysis(None, cancelled=True, frames_read=frames_read, elapsed_ms=_ms(started))
        frame = read(first + offset)
        if frame is None:
            break
        frames_read += 1
        gray, _scale = analysis_gray(frame)
        height, width = gray.shape[:2]
        if width < 2 or height < 2:
            break
        if previous_gray is not None:
            motion = estimate_global_motion(previous_gray, gray, StabilizationMode.SIMILARITY, memory)
            if motion is not None:
                accumulated = accumulated @ _to_3x3(motion.transform)  # frame k -> k-1 -> ... -> first
                measured[offset] = True
        # Decompose the accumulated alignment into the keyframe convention: rotation about the centre plus
        # the displacement the centre picks up (as a fraction of the frame). Scale is dropped (camera shake
        # barely zooms) so this stays a position/rotation path.
        centre = accumulated @ np.array([width / 2.0, height / 2.0, 1.0])
        dx[offset] = (centre[0] - width / 2.0) / width
        dy[offset] = (centre[1] - height / 2.0) / height
        rotation[offset] = math.degrees(math.atan2(accumulated[1, 0], accumulated[0, 0]))
        previous_gray = gray
        if progress is not None:
            progress(offset + 1, count)

    if frames_read == 0:
        return MotionPathAnalysis(None, cancelled=False, frames_read=0, elapsed_ms=_ms(started))
    track = MotionPath(first, dx[:frames_read], dy[:frames_read], rotation[:frames_read], measured[:frames_read])
    return MotionPathAnalysis(track, cancelled=False, frames_read=frames_read, elapsed_ms=_ms(started))


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def fill_keyframes(
    keyframes: tuple[StabilizationKeyframe, ...], path: MotionPath | None, *, step: int = 1
) -> tuple[StabilizationKeyframe, ...]:
    """Insert keyframes between the hand-set ones so the stabilized result follows the measured motion.

    The user's keyframes are preserved exactly; between each consecutive pair a keyframe is added every
    ``step`` frames whose ``tx``/``ty``/``rotation`` cancel the camera's deviation from a straight line
    (``scale``/``shear``/``stretch_y`` stay linearly interpolated). A gap the ``path`` does not cover, or a
    gap of one frame, is left as it is. Returns the keyframes sorted by frame.
    """
    if path is None or len(keyframes) < 2 or step < 1:
        return keyframes
    ordered = sorted(keyframes, key=lambda item: item.frame)
    filled: dict[int, StabilizationKeyframe] = {item.frame: item for item in ordered}
    for start, end in zip(ordered, ordered[1:]):
        span = end.frame - start.frame
        if span <= 1 or not path.covers(start.frame, end.frame):
            continue
        start_motion = path.sample(start.frame)
        end_motion = path.sample(end.frame)
        for frame in range(start.frame + step, end.frame, step):
            if frame in filled:
                continue  # never overwrite a hand-set keyframe caught by the step
            weight = (frame - start.frame) / span
            motion = path.sample(frame)
            filled[frame] = StabilizationKeyframe(
                frame,
                tx=_deshake(start.tx, end.tx, weight, motion[0], start_motion[0], end_motion[0]),
                ty=_deshake(start.ty, end.ty, weight, motion[1], start_motion[1], end_motion[1]),
                rotation=_deshake(start.rotation, end.rotation, weight, motion[2], start_motion[2], end_motion[2]),
                scale=_lerp(start.scale, end.scale, weight),
                shear=_lerp(start.shear, end.shear, weight),
                stretch_y=_lerp(start.stretch_y, end.stretch_y, weight),
            )
    return tuple(filled[frame] for frame in sorted(filled))


def _deshake(
    start_value: float, end_value: float, weight: float, motion: float, start_motion: float, end_motion: float
) -> float:
    """Linear correction between the keyframes plus the camera's deviation from its own straight line.

    ``motion`` is the measured correction-to-reference at this frame; ``start_motion``/``end_motion`` bracket
    it. The stabilized residual is ``motion - correction``; forcing it to run straight between the keyframes
    gives ``correction = linear + (motion - start) - weight * (end - start)``. That deviation is exactly the
    shake the straight line missed and is zero at both ends, so the hand-set keyframes are preserved.
    """
    relative = motion - start_motion
    deviation = relative - weight * (end_motion - start_motion)
    return _lerp(start_value, end_value, weight) + deviation
