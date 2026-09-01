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
to every render, the export analyses the clips it was not given a track for. It measures both readout
axes — a phone held upright reads along what ends up as the frame's columns — and decides which one
the footage was read along (:func:`detect_readout_axis`: a single pair of frames cannot tell, the
gradients' correlation with the camera's acceleration over the range can). Values come out as fractions of the axis
length, so a track made from the source serves renders at any size.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from space_video_enhancer.pipeline.stabilization.estimators import analysis_gray
from space_video_enhancer.pipeline.stabilization.global_motion import estimate_global_motion, new_cell_memory
from space_video_enhancer.pipeline.stabilization.rolling_shutter import center_displacement, motion_gradient
from space_video_enhancer.settings import ReadoutAxis, StabilizationMode

WOBBLE_WINDOW_SECONDS = 1.0
"""Width of the zero-mean window. Wobble slower than this counts as skew (or drift) and is left alone;
hand shake sits at 3-12 Hz, well inside it."""
MAX_WOBBLE_FRACTION = 0.08
"""Cap on the wobble, as a fraction of the frame height (a phone's jello stays well below)."""
MIN_WINDOW_FRAMES = 2
AXIS_MIN_CORRELATION = 0.3
"""How well the gradient changes must follow the camera's acceleration for the footage to decide the readout axis."""
AXIS_MIN_MARGIN = 0.1
"""...and by how much one axis must beat the other; otherwise the file's rotation decides."""
AXIS_MIN_SAMPLES = 8


def wobble_window_frames(fps: float) -> int:
    """Frames on each side of a frame that its wobble is averaged against (half of the window)."""
    return max(MIN_WINDOW_FRAMES, round(WOBBLE_WINDOW_SECONDS * max(1.0, fps) / 2.0))


def filter_wobble(changes: np.ndarray, displacements: np.ndarray, readout_fraction: float, window: int) -> np.ndarray:
    """Per-frame wobble ``(N, 2)`` from the measured changes ``(N, 2)`` and centre displacements ``(N, 2)``.

    The summed changes are the distortion of each frame relative to the first one. The part the slider
    already undoes (``readout_fraction`` times the frame's own displacement) is taken out, then the
    local mean over ``window`` frames on each side (Hann-weighted, one-sided at the ends of the range),
    so the constant of integration and any slow drift vanish. Inputs are fractions of the readout axis
    length; the result is clamped to :data:`MAX_WOBBLE_FRACTION`.
    """
    total = np.cumsum(np.asarray(changes, dtype=np.float64), axis=0)
    if readout_fraction != 0.0:
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


def readout_correlation(changes: np.ndarray, displacements: np.ndarray, measured: np.ndarray) -> float:
    """How well the gradient changes along an axis follow the camera's acceleration — the rolling-shutter signature.

    A frame's distortion is the readout time times the velocity during it, so the change between two
    frames is ``r · (v_k - v_{k-1})``: positively correlated with the acceleration of the measured
    displacement along the true axis. Seen along the other axis the same footage reads as a zoom or roll
    plus a gradient in the *other* component (the two decompositions fit any single pair equally well),
    which does not follow the acceleration. Pearson correlation over both components; ``0.0`` when the
    camera barely moved.
    """
    both = measured[1:] & measured[:-1]
    if int(both.sum()) < AXIS_MIN_SAMPLES:
        return 0.0
    acceleration = (displacements[1:] - displacements[:-1])[both].ravel()
    gradient = changes[1:][both].ravel()
    if np.std(acceleration) < 1e-6 or np.std(gradient) < 1e-6:
        return 0.0
    return float(np.corrcoef(acceleration, gradient)[0, 1])


def detect_readout_axis(row_correlation: float, column_correlation: float, rotation: int) -> ReadoutAxis:
    """The axis whose gradients clearly follow the camera's acceleration; else what the file's rotation implies.

    A file recorded upright carries a 90°/270° rotation and the decoder already turns it: the sensor's
    rows are then the frame's columns.
    """
    best = max(row_correlation, column_correlation)
    if best >= AXIS_MIN_CORRELATION and abs(row_correlation - column_correlation) >= AXIS_MIN_MARGIN:
        return ReadoutAxis.ROWS if row_correlation > column_correlation else ReadoutAxis.COLUMNS
    return ReadoutAxis.COLUMNS if rotation % 180 == 90 else ReadoutAxis.ROWS


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


@dataclass(frozen=True, eq=False)
class WobbleTrack:
    """What one pass over ``source`` frames ``first_frame`` .. ``last_frame`` measured (see the module docstring).

    Measurements are kept in analysis pixels along with the analysis size; :meth:`wobble` converts to
    fractions of the readout axis, which is what a render at any size needs.
    """

    source: Path
    first_frame: int
    row_changes: np.ndarray
    """``(N, 2)``: change of the motion gradient along the rows between frame k-1 and k (x and y, analysis
    pixels across the height); zero where the pair could not be measured."""
    column_changes: np.ndarray
    """``(N, 2)``: the same along the columns (analysis pixels across the width)."""
    displacements: np.ndarray
    """``(N, 2)``: scene displacement at the frame centre between k-1 and k, analysis pixels."""
    measured: np.ndarray
    """``(N,)`` bool: the pair ending at this frame was tracked."""
    analysis_size: tuple[int, int]
    """``(width, height)`` of the analysis image the measurements refer to."""
    rotation: int
    """Rotation the file asks for (degrees); 90/270 means the sensor's rows are the frame's columns."""
    window: int
    """Frames on each side used by the zero-mean filter (see :func:`wobble_window_frames`)."""
    detected_axis: ReadoutAxis
    """What :func:`detect_readout_axis` concluded for this range (never ``AUTO``)."""

    @property
    def count(self) -> int:
        return len(self.row_changes)

    def changes(self, axis: ReadoutAxis) -> np.ndarray:
        return self.column_changes if axis is ReadoutAxis.COLUMNS else self.row_changes

    def axis_length(self, axis: ReadoutAxis) -> int:
        width, height = self.analysis_size
        return width if axis is ReadoutAxis.COLUMNS else height

    @property
    def last_frame(self) -> int:
        return self.first_frame + self.count - 1

    @property
    def measured_fraction(self) -> float:
        return float(np.count_nonzero(self.measured)) / self.count if self.count else 0.0

    def covers(self, first: int, last: int) -> bool:
        return self.first_frame <= first and last <= self.last_frame

    def wobble(self, readout_fraction: float = 0.0, axis: ReadoutAxis = ReadoutAxis.ROWS) -> np.ndarray:
        """``(N, 2)`` wobble per frame along ``axis``, as a fraction of that axis (see :func:`filter_wobble`)."""
        length = float(max(1, self.axis_length(axis)))
        return filter_wobble(self.changes(axis) / length, self.displacements / length, readout_fraction, self.window)

    def wobble_at(
        self, frame_index: int, readout_fraction: float = 0.0, axis: ReadoutAxis = ReadoutAxis.ROWS
    ) -> tuple[float, float] | None:
        """The wobble of ``frame_index`` (fraction of the axis length, x and y), or ``None`` outside the range."""
        offset = frame_index - self.first_frame
        if offset < 0 or offset >= self.count:
            return None
        row = self.wobble(readout_fraction, axis)[offset]
        return float(row[0]), float(row[1])

    def energy(self, axis: ReadoutAxis) -> float:
        """Root-mean-square wobble along ``axis`` (fraction of the axis): how much jello that axis carries."""
        return _rms(self.wobble(0.0, axis))

    def correlation(self, axis: ReadoutAxis) -> float:
        """See :func:`readout_correlation`: the rolling-shutter signature of ``axis`` in this range."""
        return readout_correlation(self.changes(axis), self.displacements, self.measured)

    def readout_fraction_estimate(self, axis: ReadoutAxis) -> float:
        """The signed readout fraction the footage implies: the slider value, measured instead of guessed.

        Least-squares slope (through the origin) of the gradient changes against the camera's acceleration
        (``change = r · (v_k - v_{k-1})``, see :func:`readout_correlation`). Negative means the sensor read
        against the frame's axis — some phones recorded upright do. ``0.0`` when the range holds too little
        motion to tell.
        """
        both = self.measured[1:] & self.measured[:-1]
        if int(both.sum()) < AXIS_MIN_SAMPLES:
            return 0.0
        acceleration = (self.displacements[1:] - self.displacements[:-1])[both].ravel()
        gradient = self.changes(axis)[1:][both].ravel()
        denominator = float(np.dot(acceleration, acceleration))
        if denominator < 1e-9:
            return 0.0
        return float(np.clip(np.dot(gradient, acceleration) / denominator, -1.0, 1.0))


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
    rotation: int = 0,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> WobbleAnalysis:
    """Track the motion through frames ``first`` .. ``last`` (inclusive) of ``source`` and record each change of wobble.

    ``read(frame_index)`` returns the decoded frame or ``None`` (the stream ended early: the track stops
    there). The frames are only tracked, never rendered, so deinterlacing does not enter — the comb of
    an interlaced frame vanishes at the analysis size. Both readout axes are measured; ``rotation`` (the
    file's, in degrees) breaks the tie when neither carries wobble. ``progress(done, total)`` is called
    after every frame; ``cancelled()`` is polled between frames.
    """
    started = time.perf_counter()
    count = max(0, last - first + 1)
    row_changes = np.zeros((count, 2), dtype=np.float64)
    column_changes = np.zeros((count, 2), dtype=np.float64)
    displacements = np.zeros((count, 2), dtype=np.float64)
    measured = np.zeros(count, dtype=bool)
    analysis_size = (0, 0)
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
        if gray.size:
            analysis_size = (int(gray.shape[1]), int(gray.shape[0]))
        if previous_gray is not None and gray.shape == previous_gray.shape and gray.size:
            motion = estimate_global_motion(previous_gray, gray, StabilizationMode.AUTO, memory)
            if motion is not None:
                width, height = analysis_size
                measured[offset] = True
                along_rows = motion_gradient(motion.vectors, height, ReadoutAxis.ROWS)
                if along_rows is not None:
                    row_changes[offset] = along_rows
                along_columns = motion_gradient(motion.vectors, width, ReadoutAxis.COLUMNS)
                if along_columns is not None:
                    column_changes[offset] = along_columns
                displacements[offset] = center_displacement(motion.transform, gray.shape)
        previous_gray = gray
        if progress is not None:
            progress(offset + 1, count)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if frames_read == 0:
        return WobbleAnalysis(None, 0, elapsed_ms)
    row_changes, column_changes = row_changes[:frames_read], column_changes[:frames_read]
    displacements, measured = displacements[:frames_read], measured[:frames_read]
    row_correlation = readout_correlation(row_changes, displacements, measured)
    column_correlation = readout_correlation(column_changes, displacements, measured)
    track = WobbleTrack(
        source,
        first,
        row_changes,
        column_changes,
        displacements,
        measured,
        analysis_size,
        rotation,
        window,
        detect_readout_axis(row_correlation, column_correlation, rotation),
    )
    return WobbleAnalysis(track, frames_read, elapsed_ms)
