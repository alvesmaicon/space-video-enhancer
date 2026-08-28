"""Rolling-shutter wobble measured over a whole range of frames, then pinned down with a zero-mean prior.

Between two frames only the *change* of the intra-frame distortion is observable (see
:mod:`rolling_shutter`): the distortion itself is that change summed up, and a sum of noisy
measurements wanders. Accumulating it live with a slow decay (the first version) turned a fast
jello into a slow one — the estimate drifted for tens of frames and the correction painted that
drift onto frames that were fine.

What pins the sum down is that a shake oscillates around the smooth camera path: over a second or
so the wobble averages to zero. So a range of frames is analysed once (:func:`analyze_wobble`, a
pass that only tracks, at the estimator's analysis size), the changes are summed, and a centred
moving average — frames before *and after* each one — is subtracted (:func:`filter_wobble`). What
is left is the fast part, which is the jello; the slow part (a pan speeding up, measurement drift)
is left alone — the steady skew of a pan is the slider's job.

A :class:`WobbleTrack` is immutable and Qt-free: the preview keeps one per clip range and hands it
to every render, the export analyses the clips it was not given a track for. Values are fractions of
the frame height, so a track made from the source serves renders at any size.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from space_video_enhancer.pipeline.stabilization.estimators import analysis_gray
from space_video_enhancer.pipeline.stabilization.global_motion import estimate_global_motion, new_cell_memory
from space_video_enhancer.pipeline.stabilization.rolling_shutter import center_displacement, row_motion_gradient
from space_video_enhancer.settings import StabilizationMode

WOBBLE_WINDOW_SECONDS = 1.0
"""Width of the zero-mean window. Wobble slower than this counts as skew (or drift) and is left alone;
hand shake sits at 3-12 Hz, well inside it."""
MAX_WOBBLE_FRACTION = 0.08
"""Cap on the wobble, as a fraction of the frame height (a phone's jello stays well below)."""
MIN_WINDOW_FRAMES = 2


def wobble_window_frames(fps: float) -> int:
    """Frames on each side of a frame that its wobble is averaged against (half of the window)."""
    return max(MIN_WINDOW_FRAMES, round(WOBBLE_WINDOW_SECONDS * max(1.0, fps) / 2.0))


def filter_wobble(changes: np.ndarray, displacements: np.ndarray, readout_fraction: float, window: int) -> np.ndarray:
    """Per-frame wobble ``(N, 2)`` from the measured changes ``(N, 2)`` and centre displacements ``(N, 2)``.

    The summed changes are the distortion of each frame relative to the first one. The part the slider
    already undoes (``readout_fraction`` times the frame's own displacement) is taken out, then the
    local mean over ``window`` frames on each side (Hann-weighted, one-sided at the ends of the range),
    so the constant of integration and any slow drift vanish. Clamped to :data:`MAX_WOBBLE_FRACTION`.
    """
    total = np.cumsum(np.asarray(changes, dtype=np.float64), axis=0)
    if readout_fraction > 0.0:
        total = total - readout_fraction * np.asarray(displacements, dtype=np.float64)
    count = len(total)
    if count == 0:
        return total
    half = max(1, window)
    kernel = np.hanning(2 * half + 3)[1:-1]  # 2*half+1 taps; the plain window has zero ends, keep every tap useful
    kernel /= kernel.sum()

    def local_mean(values: np.ndarray) -> np.ndarray:
        # Zero-padded so the result always has ``count`` entries (np.convolve's "same" follows the longer input).
        return np.convolve(np.pad(values, half), kernel, mode="valid")

    weight = local_mean(np.ones(count))  # how much of the window falls inside the range (one-sided at the ends)
    smooth = np.column_stack([local_mean(total[:, axis]) for axis in range(2)]) / weight[:, None]
    return np.clip(total - smooth, -MAX_WOBBLE_FRACTION, MAX_WOBBLE_FRACTION)


@dataclass(frozen=True, eq=False)
class WobbleTrack:
    """What one pass over ``source`` frames ``first_frame`` .. ``last_frame`` measured (see the module docstring)."""

    source: Path
    first_frame: int
    changes: np.ndarray
    """``(N, 2)``: change of the row-motion gradient between frame k-1 and k, as a fraction of the height
    (x and y); zero where the pair could not be measured."""
    displacements: np.ndarray
    """``(N, 2)``: scene displacement at the frame centre between k-1 and k, fraction of the height."""
    measured: np.ndarray
    """``(N,)`` bool: the pair ending at this frame gave a gradient."""
    window: int
    """Frames on each side used by the zero-mean filter (see :func:`wobble_window_frames`)."""

    @property
    def count(self) -> int:
        return len(self.changes)

    @property
    def last_frame(self) -> int:
        return self.first_frame + self.count - 1

    @property
    def measured_fraction(self) -> float:
        return float(np.count_nonzero(self.measured)) / self.count if self.count else 0.0

    def covers(self, first: int, last: int) -> bool:
        return self.first_frame <= first and last <= self.last_frame

    def wobble(self, readout_fraction: float = 0.0) -> np.ndarray:
        """``(N, 2)`` wobble per frame, fraction of the height (see :func:`filter_wobble`)."""
        return filter_wobble(self.changes, self.displacements, readout_fraction, self.window)

    def wobble_at(self, frame_index: int, readout_fraction: float = 0.0) -> tuple[float, float] | None:
        """The wobble of ``frame_index`` (fraction of the height, x and y), or ``None`` outside the range."""
        offset = frame_index - self.first_frame
        if offset < 0 or offset >= self.count:
            return None
        row = self.wobble(readout_fraction)[offset]
        return float(row[0]), float(row[1])


@dataclass(frozen=True)
class WobbleAnalysis:
    track: WobbleTrack | None
    """``None`` when cancelled or when not a single frame could be read."""
    frames_read: int
    elapsed_ms: float
    cancelled: bool = False


def analyze_wobble(
    read: Callable[[int], np.ndarray | None],
    source: Path,
    first: int,
    last: int,
    window: int,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> WobbleAnalysis:
    """Track the motion through frames ``first`` .. ``last`` (inclusive) of ``source`` and record each change of wobble.

    ``read(frame_index)`` returns the decoded frame or ``None`` (the stream ended early: the track stops
    there). The frames are only tracked, never rendered, so deinterlacing does not enter — the comb of
    an interlaced frame vanishes at the analysis size. ``progress(done, total)`` is called after every
    frame; ``cancelled()`` is polled between frames.
    """
    started = time.perf_counter()
    count = max(0, last - first + 1)
    changes = np.zeros((count, 2), dtype=np.float64)
    displacements = np.zeros((count, 2), dtype=np.float64)
    measured = np.zeros(count, dtype=bool)
    memory = new_cell_memory()
    previous_gray: np.ndarray | None = None
    frames_read = 0
    for offset in range(count):
        if cancelled is not None and cancelled():
            return WobbleAnalysis(None, frames_read, (time.perf_counter() - started) * 1000.0, cancelled=True)
        frame = read(first + offset)
        if frame is None:
            break
        frames_read += 1
        gray, _scale = analysis_gray(frame)
        if previous_gray is not None and gray.shape == previous_gray.shape and gray.size:
            motion = estimate_global_motion(previous_gray, gray, StabilizationMode.AUTO, memory)
            if motion is not None:
                height = int(gray.shape[0])
                gradient = row_motion_gradient(motion.vectors, height)
                if gradient is not None:
                    changes[offset] = (gradient[0] / height, gradient[1] / height)
                    measured[offset] = True
                shift_x, shift_y = center_displacement(motion.transform, gray.shape)
                displacements[offset] = (shift_x / height, shift_y / height)
        previous_gray = gray
        if progress is not None:
            progress(offset + 1, count)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if frames_read == 0:
        return WobbleAnalysis(None, 0, elapsed_ms)
    read_count = slice(0, frames_read)
    track = WobbleTrack(source, first, changes[read_count], displacements[read_count], measured[read_count], window)
    return WobbleAnalysis(track, frames_read, elapsed_ms)
