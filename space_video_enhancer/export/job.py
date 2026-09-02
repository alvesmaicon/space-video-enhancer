"""Qt-free export core: decode -> render_frame -> raw frames into FFmpeg.

Exports a :class:`Sequence`: clips of one or more source files, each with its own cuts and
settings, played back to back. Frames are decoded per source (one capture each, seeking only
where a cut or a clip boundary breaks the run), rendered with the clip's settings and fitted
to the output size of the first frame (letterbox), and the audio of every kept run is trimmed
and concatenated by ffmpeg. Runs on a worker thread, reports progress through a callback,
cooperates with a cancel event, never leaves FFmpeg running or a half-written file behind,
and logs ``export.start`` / ``export.progress`` / ``export.finish`` (or ``.cancelled`` /
``.failed``) with per-stage timing statistics and stabilization estimator histograms.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from space_video_enhancer.export.ffmpeg import AudioSegment, EncoderChoice, build_export_command, source_has_audio
from space_video_enhancer.pipeline import retime
from space_video_enhancer.pipeline.filters import OpenCLBackend, ProcessingContext
from space_video_enhancer.pipeline.geometry import ensure_even_size, fit_to_canvas
from space_video_enhancer.pipeline.render import build_context, render_frame
from space_video_enhancer.pipeline.stabilization.mesh import (
    MeshCorrections,
    MeshPath,
    analyze_mesh_path,
    corrections_for,
    corrections_for_object,
)
from space_video_enhancer.pipeline.stabilization.stabilize import (
    mesh_stabilized_for_analysis,
    needs_previous_frame,
    prime_with_previous,
    smoothing_time_constant,
)
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.pipeline.stabilization.wobble import WobbleTrack, analyze_wobble, wobble_window_frames
from space_video_enhancer.sequence import Clip, ClipSource, ExportSegment, Sequence
from space_video_enhancer.settings import RenderSettings, StabilizationMode
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.telemetry.stats import Histogram, StageStats
from space_video_enhancer.telemetry.trace import RenderTrace
from space_video_enhancer.timeline import FrameRange
from space_video_enhancer.video import capture_rotation

LOG = get_logger("export")
ProgressCallback = Callable[[int, int], None]
StatusCallback = Callable[[str], None]

PROGRESS_INTERVAL_SECONDS = 0.1
PROGRESS_LOG_STEP_PERCENT = 10
STDERR_TAIL_BYTES = 64 * 1024
CANCELLED_MESSAGE = "Exportacao cancelada pelo usuario."
FFMPEG_TERMINATE_TIMEOUT = 1.0


@dataclass(frozen=True)
class ExportRequest:
    sequence: Sequence
    output_path: Path
    worker_count: int
    ffmpeg_path: str
    encoder: EncoderChoice
    wobble_tracks: tuple[WobbleTrack, ...] = ()
    """Rolling-shutter wobble analyses already done (the preview's); a clip none of them covers is analysed here."""
    mesh_tracks: tuple[MeshPath, ...] = ()
    """Mesh stabilization analyses already done (the preview's); a MESH clip none covers is analysed here."""

    @classmethod
    def single(
        cls,
        source_path: Path,
        output_path: Path,
        settings: RenderSettings,
        start_frame: int,
        end_frame: int,
        fps: float,
        worker_count: int,
        ffmpeg_path: str,
        encoder: EncoderChoice,
        *,
        cuts: tuple[tuple[int, int], ...] = (),
        stabilization_reference_frame: int | None = None,
        frame_count: int | None = None,
        width: int = 0,
        height: int = 0,
    ) -> ExportRequest:
        """The one-video case: a clip of ``source_path`` from ``start_frame`` to ``end_frame`` minus ``cuts``."""
        source = ClipSource(source_path, frame_count if frame_count is not None else end_frame + 1, fps, width, height)
        sequence = Sequence.single(
            source,
            FrameRange(start_frame, end_frame),
            cuts=cuts,
            settings=settings,
            stabilization_reference_frame=stabilization_reference_frame,
        )
        return cls(sequence, output_path, worker_count, ffmpeg_path, encoder)

    @property
    def fps(self) -> float:
        return self.sequence.fps

    @property
    def segments(self) -> tuple[ExportSegment, ...]:
        """What actually gets exported: every clip minus its cuts, in order."""
        return self.sequence.export_segments()

    @property
    def expected_frames(self) -> int:
        return max(1, self.sequence.export_count)

    @property
    def duration_seconds(self) -> float:
        return self.sequence.duration_seconds

    @property
    def source_path(self) -> Path:
        """First source (for messages); a sequence may use several."""
        return self.sequence.clips[0].source.path if self.sequence.clips else Path()

    @property
    def effective_worker_count(self) -> int:
        """Inter-frame stabilization / smoothing in any clip needs frames in order: one worker."""
        if any(clip.settings.stabilization.requires_sequential_processing for clip in self.sequence.clips):
            return 1
        return max(1, self.worker_count)


@dataclass(frozen=True)
class ExportResult:
    success: bool
    message: str
    processed_frames: int
    expected_frames: int
    cancelled: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count_mismatch(self) -> bool:
        return self.success and self.processed_frames != self.expected_frames


class _ExportCancelledError(Exception):
    pass


class _StderrDrain(threading.Thread):
    """Reads FFmpeg's stderr continuously so a chatty encoder can never block the pipe."""

    def __init__(self, stream) -> None:  # type: ignore[no-untyped-def]
        super().__init__(daemon=True, name="ffmpeg-stderr")
        self._stream = stream
        self._chunks: deque[bytes] = deque()
        self._size = 0

    def run(self) -> None:
        for chunk in iter(lambda: self._stream.read(4096), b""):
            self._chunks.append(chunk)
            self._size += len(chunk)
            while self._size > STDERR_TAIL_BYTES and len(self._chunks) > 1:
                self._size -= len(self._chunks.popleft())

    def text(self) -> str:
        self.join(timeout=1.0)
        return b"".join(self._chunks).decode("utf-8", errors="ignore").strip()


class _ExportMetrics:
    """Thread-safe aggregation of per-frame traces for the end-of-job summary."""

    def __init__(self, request: ExportRequest, backend: OpenCLBackend | None) -> None:
        self.started = time.perf_counter()
        self.stages = StageStats(window=4000)
        self.estimators = Histogram()
        self.outcomes = Histogram()
        self.backends = Histogram()
        self.workers = request.effective_worker_count
        self.encoder = request.encoder.label
        self.backend = backend.label if backend is not None else "CPU"
        self.wobble_analysis: list[dict[str, Any]] = []
        """One entry per clip whose wobble was analysed by this job (none when the preview's tracks served)."""
        self.mesh_analysis: list[dict[str, Any]] = []
        """One entry per MESH clip this job had to analyse itself."""
        self._lock = threading.Lock()

    def record(self, trace: RenderTrace) -> None:
        with self._lock:
            self.stages.add(trace)
            self.estimators.add(trace.notes.get("stabilization.estimator"))
            self.outcomes.add(trace.notes.get("stabilization.result"))
            self.backends.add(trace.notes.get("filters.backend"))

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started

    def summary(self, processed_frames: int, output_path: Path) -> dict[str, Any]:
        elapsed = self.elapsed_seconds()
        with self._lock:
            summary: dict[str, Any] = {
                "elapsed_s": round(elapsed, 2),
                "frames": processed_frames,
                "fps": round(processed_frames / elapsed, 2) if elapsed > 0 else None,
                "workers": self.workers,
                "encoder": self.encoder,
                "backend": self.backend,
                "render": self.stages.summary(),
                "filters_backend": self.backends.as_dict(),
            }
            if len(self.estimators) or len(self.outcomes):
                summary["stabilization"] = {
                    "estimators": self.estimators.as_dict(),
                    "outcomes": self.outcomes.as_dict(),
                }
            if self.wobble_analysis:
                summary["wobble_analysis"] = list(self.wobble_analysis)
            if self.mesh_analysis:
                summary["mesh_analysis"] = list(self.mesh_analysis)
        if output_path.exists():
            summary["output_bytes"] = output_path.stat().st_size
        return summary


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is not None:
        with contextlib.suppress(OSError):
            process.stdin.close()
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=FFMPEG_TERMINATE_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _discard_output(output_path: Path) -> None:
    with contextlib.suppress(OSError):
        output_path.unlink(missing_ok=True)


class _Throttle:
    def __init__(self, callback: ProgressCallback, total: int) -> None:
        self._callback = callback
        self._total = total
        self._last = 0.0

    def __call__(self, processed: int, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or (now - self._last) >= PROGRESS_INTERVAL_SECONDS:
            self._last = now
            self._callback(processed, self._total)


class _StatusThrottle:
    """Free-text progress for the phases before frames flow (the wobble analysis), at most a few times a second."""

    def __init__(self, callback: StatusCallback | None) -> None:
        self._callback = callback
        self._last = 0.0

    def __call__(self, message: str, *, force: bool = False) -> None:
        if self._callback is None:
            return
        now = time.monotonic()
        if force or (now - self._last) >= PROGRESS_INTERVAL_SECONDS:
            self._last = now
            self._callback(message)


def _clip_summary(clip: Clip) -> dict[str, Any]:
    return {
        "source": clip.source.path.name,
        "range": [clip.range.start, clip.range.end],
        "cuts": [[cut.start, cut.end] for cut in clip.cuts],
        "segments": [[segment.start, segment.end] for segment in clip.segments],
        "frames": clip.export_count,
        "reference_frame": clip.stabilization_reference_frame,
        "settings": asdict(clip.settings),
    }


def _request_summary(request: ExportRequest) -> dict[str, Any]:
    return {
        "sources": [path.name for path in request.sequence.source_paths],
        "output": request.output_path.name,
        "frames": request.expected_frames,
        "clips": [_clip_summary(clip) for clip in request.sequence.clips],
        "fps": round(request.fps, 3),
        "workers": request.effective_worker_count,
        "requested_workers": request.worker_count,
        "encoder": request.encoder.label,
    }


class _SourceReaders:
    """One ``VideoCapture`` per source file, opened on first use, with the decoder position tracked."""

    def __init__(self) -> None:
        self._captures: dict[Path, cv2.VideoCapture] = {}
        self._positions: dict[Path, int | None] = {}

    def open(self, path: Path) -> cv2.VideoCapture | None:
        capture = self._captures.get(path)
        if capture is None:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                return None
            self._captures[path] = capture
            self._positions[path] = 0
        return capture

    def read(self, path: Path, frame_index: int) -> np.ndarray | None:
        capture = self.open(path)
        if capture is None:
            return None
        if self._positions.get(path) != frame_index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            self._positions[path] = None
            return None
        self._positions[path] = frame_index + 1
        return frame

    def rotation(self, path: Path) -> int:
        capture = self.open(path)
        return capture_rotation(capture) if capture is not None else 0

    def release_all(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()
        self._positions.clear()


class _ClipRenderer:
    """Per-clip processing state: settings, the denoise context (resolved once), the stabilization path, the wobble."""

    def __init__(
        self,
        clip: Clip,
        first_frame: np.ndarray,
        reference: np.ndarray | None,
        wobble_track: WobbleTrack | None,
        mesh_track: MeshPath | None = None,
    ) -> None:
        self.settings = clip.settings
        # Denoise strength is resolved once, from the clip's first frame, so it stays constant across the clip.
        self.context: ProcessingContext = build_context(clip.settings, first_frame)
        self.state = StabilizationState()
        self.reference = reference
        self.wobble_track = wobble_track
        self.mesh_corrections = _clip_mesh_corrections(clip, mesh_track, first_frame.shape)


def _clip_mesh_corrections(
    clip: Clip, mesh_track: MeshPath | None, first_frame_shape: tuple[int, ...]
) -> MeshCorrections | None:
    """The clip's mesh fields for its own settings; ``None`` when the mode needs none (or no track yet)."""
    if mesh_track is None or not clip.settings.stabilization.needs_mesh_analysis:
        return None
    stabilization = clip.settings.stabilization
    smoothing = smoothing_time_constant(stabilization.temporal_smoothing)
    if stabilization.mode is StabilizationMode.OBJECT:
        height, width = first_frame_shape[:2]  # the analysed frames' shape: the pose fit needs it
        return corrections_for_object(
            mesh_track,
            stabilization.focus_regions,
            smoothing,
            limit=stabilization.margin,
            aspect=width / max(1, height),
        )
    return corrections_for(mesh_track, smoothing, stabilization.margin)


def _analyze_clip_wobble(
    readers: _SourceReaders,
    index: int,
    clip: Clip,
    cancel: threading.Event,
    status: _StatusThrottle,
    corrections: MeshCorrections | None = None,
) -> tuple[WobbleTrack | None, dict[str, Any] | None]:
    """Track the clip's range (plus the filter window on each side) on the export thread; ``None`` when cancelled.

    With ``corrections`` the analysis is *residual*: every frame is mesh-stabilized before tracking
    (see :func:`mesh_stabilized_for_analysis`), so the track holds only what the mesh did not absorb.
    The range is then clamped to the fields' coverage — a stabilized/unstabilized boundary inside the
    range would read as a fake jello step.
    """
    path = clip.source.path
    window = wobble_window_frames(clip.fps)
    first = max(0, clip.range.start - window)
    last = min(clip.source.last_frame, clip.range.end + window)
    if corrections is not None:
        first = max(first, corrections.first_frame)
        last = min(last, corrections.first_frame + len(corrections.fields) - 1)

    def read(frame_index: int) -> np.ndarray | None:
        frame = readers.read(path, frame_index)
        if frame is None or corrections is None:
            return frame
        return mesh_stabilized_for_analysis(frame, clip.settings.stabilization, frame_index, corrections)

    def report(done: int, total: int) -> None:
        label = "a oscilação residual" if corrections is not None else "a oscilação do rolling shutter"
        status(
            f"Analisando {label} de {clip.name}: {done * 100 // max(1, total)}% ({done}/{total} frames)",
            force=done == total,
        )

    analysis = analyze_wobble(
        read,
        path,
        first,
        last,
        window,
        rotation=readers.rotation(path),
        residual=corrections is not None,
        progress=report,
        cancelled=cancel.is_set,
    )
    if analysis.cancelled:
        return None, None
    track = analysis.track
    summary: dict[str, Any] = {
        "clip": index,
        "name": clip.name,
        "first": first,
        "last": last,
        "frames": analysis.frames_read,
        "measured": round(track.measured_fraction, 3) if track is not None else None,
        "axis": track.detected_axis.value if track is not None else None,
        "residual": corrections is not None,
        "elapsed_ms": round(analysis.elapsed_ms, 1),
    }
    log_event(LOG, "export.wobble_analysis", **summary)
    return analysis.track, summary


def _wobble_tracks(
    readers: _SourceReaders,
    request: ExportRequest,
    cancel: threading.Event,
    status: _StatusThrottle,
    mesh_tracks: dict[int, MeshPath],
) -> tuple[dict[int, WobbleTrack], list[dict[str, Any]]] | None:
    """A wobble track per clip that needs one (wobble, or the readout axis): the one handed in when it covers the
    clip, else analysed now.

    A clip in a mesh mode gets a *residual* track — always analysed here, with the export's own
    fields, never reused from the preview: the preview's residual may have been measured against
    older slider values (harmless there, but the export must match exactly what it renders).

    ``None`` when the export was cancelled during an analysis.
    """
    tracks: dict[int, WobbleTrack] = {}
    summaries: list[dict[str, Any]] = []
    for index, clip in enumerate(request.sequence.clips):
        if not clip.settings.stabilization.needs_wobble_analysis:
            continue
        corrections: MeshCorrections | None = None
        if clip.settings.stabilization.applies_residual_wobble:
            mesh_track = mesh_tracks.get(index)
            if mesh_track is None:
                continue  # no mesh path (unreadable clip): there is no residual to measure
            probe = readers.read(clip.source.path, clip.range.start)
            if probe is None:
                continue
            corrections = _clip_mesh_corrections(clip, mesh_track, probe.shape)
        else:
            provided = next(
                (
                    track
                    for track in request.wobble_tracks
                    if not track.residual
                    and track.source == clip.source.path
                    and track.covers(clip.range.start, clip.range.end)
                ),
                None,
            )
            if provided is not None:
                tracks[index] = provided
                continue
        track, summary = _analyze_clip_wobble(readers, index, clip, cancel, status, corrections)
        if summary is None:
            return None
        summaries.append(summary)
        if track is not None:
            tracks[index] = track
    return tracks, summaries


def _analyze_clip_mesh(
    readers: _SourceReaders, index: int, clip: Clip, cancel: threading.Event, status: _StatusThrottle
) -> tuple[MeshPath | None, dict[str, Any] | None]:
    """Track the clip's range (plus the smoothing window on each side) into a mesh path; ``None`` when cancelled."""
    path = clip.source.path
    margin = round(smoothing_time_constant(clip.settings.stabilization.temporal_smoothing))
    first = max(0, clip.range.start - margin)
    last = min(clip.source.last_frame, clip.range.end + margin)

    def read(frame_index: int) -> np.ndarray | None:
        return readers.read(path, frame_index)

    def report(done: int, total: int) -> None:
        status(
            f"Analisando a malha de {clip.name}: {done * 100 // max(1, total)}% ({done}/{total} frames)",
            force=done == total,
        )

    analysis = analyze_mesh_path(read, path, first, last, progress=report, cancelled=cancel.is_set)
    if analysis.cancelled:
        return None, None
    track = analysis.track
    summary: dict[str, Any] = {
        "clip": index,
        "name": clip.name,
        "first": first,
        "last": last,
        "frames": analysis.frames_read,
        "measured": round(track.measured_fraction, 3) if track is not None else None,
        "elapsed_ms": round(analysis.elapsed_ms, 1),
    }
    log_event(LOG, "export.mesh_analysis", **summary)
    return analysis.track, summary


def _mesh_tracks(
    readers: _SourceReaders, request: ExportRequest, cancel: threading.Event, status: _StatusThrottle
) -> tuple[dict[int, MeshPath], list[dict[str, Any]]] | None:
    """A mesh path per MESH clip: the one handed in when it covers the clip, else analysed now.

    ``None`` when the export was cancelled during an analysis.
    """
    tracks: dict[int, MeshPath] = {}
    summaries: list[dict[str, Any]] = []
    for index, clip in enumerate(request.sequence.clips):
        if not clip.settings.stabilization.needs_mesh_analysis:
            continue
        provided = next(
            (
                track
                for track in request.mesh_tracks
                if track.source == clip.source.path and track.covers(clip.range.start, clip.range.end)
            ),
            None,
        )
        if provided is not None:
            tracks[index] = provided
            continue
        track, summary = _analyze_clip_mesh(readers, index, clip, cancel, status)
        if summary is None:
            return None
        summaries.append(summary)
        if track is not None:
            tracks[index] = track
    return tracks, summaries


def _validation_problem(request: ExportRequest) -> str | None:
    """Why the request cannot be exported at all (checked before any file is opened)."""
    sequence = request.sequence
    if sequence.is_empty:
        return "Nenhum clipe na sequencia."
    if not request.segments:
        return "Os cortes removem o video inteiro: nada a exportar."
    return None


def run_export(
    request: ExportRequest,
    *,
    progress: ProgressCallback,
    cancel: threading.Event,
    backend: OpenCLBackend | None = None,
    status: StatusCallback | None = None,
) -> ExportResult:
    """Blocking export. Safe to call from any thread; touches no Qt objects.

    ``progress(processed, total)`` counts written frames; ``status(message)`` narrates the phases before
    the first frame (the rolling-shutter wobble analysis of each clip).
    """
    log_event(LOG, "export.start", **_request_summary(request))
    problem = _validation_problem(request)
    if problem is not None:
        log_event(LOG, "export.failed", level=logging.ERROR, message=problem, output=request.output_path.name)
        return ExportResult(False, problem, 0, request.expected_frames)
    readers = _SourceReaders()
    if readers.open(request.segments[0].clip.source.path) is None:
        message = "Nao foi possivel reabrir o video para exportacao."
        log_event(LOG, "export.failed", level=logging.ERROR, message=message, output=request.output_path.name)
        readers.release_all()
        return ExportResult(False, message, 0, request.expected_frames)

    try:
        return _run_export(readers, request, progress, cancel, backend, _StatusThrottle(status))
    finally:
        readers.release_all()


def _run_export(
    readers: _SourceReaders,
    request: ExportRequest,
    progress: ProgressCallback,
    cancel: threading.Event,
    backend: OpenCLBackend | None,
    status: _StatusThrottle,
) -> ExportResult:
    total = request.expected_frames
    report = _Throttle(progress, total)
    sequence = request.sequence
    metrics = _ExportMetrics(request, backend)

    def finish(result: ExportResult, event: str, level: int = logging.INFO, **extra: Any) -> ExportResult:
        summary = metrics.summary(result.processed_frames, request.output_path)
        log_event(LOG, event, level=level, output=request.output_path.name, message=result.message, **summary, **extra)
        return ExportResult(
            result.success, result.message, result.processed_frames, result.expected_frames, result.cancelled, summary
        )

    # Camera-lock references are decoded up front: the captures are not shared with render workers.
    references = {
        index: readers.read(clip.source.path, clip.stabilization_reference_frame)
        for index, clip in enumerate(sequence.clips)
        if clip.stabilization_reference_frame is not None
    }
    # Analyses run before any frame is rendered, so every clip can render in parallel. The mesh goes
    # first: a mesh clip's wobble is *residual* — measured over its frames already warped by the fields.
    mesh_analysed = _mesh_tracks(readers, request, cancel, status)
    if mesh_analysed is None:
        return finish(ExportResult(False, CANCELLED_MESSAGE, 0, total, cancelled=True), "export.cancelled")
    mesh_tracks, metrics.mesh_analysis = mesh_analysed
    analysed = _wobble_tracks(readers, request, cancel, status, mesh_tracks)
    if analysed is None:
        return finish(ExportResult(False, CANCELLED_MESSAGE, 0, total, cancelled=True), "export.cancelled")
    wobble_tracks, metrics.wobble_analysis = analysed

    frames = _iter_frames(readers, request)
    first = next(frames, None)
    if first is None:
        message = "Nao foi possivel ler o frame inicial do corte selecionado."
        return finish(ExportResult(False, message, 0, total), "export.failed", logging.ERROR)
    first_clip, first_index, first_frame = first

    renderers: dict[int, _ClipRenderer] = {}

    def renderer_for(clip_index: int, frame: np.ndarray) -> _ClipRenderer:
        """Created on the clip's first frame, on the export thread, before any worker sees it."""
        renderer = renderers.get(clip_index)
        if renderer is None:
            clip = sequence.clips[clip_index]
            renderer = _ClipRenderer(
                clip, frame, references.get(clip_index), wobble_tracks.get(clip_index), mesh_tracks.get(clip_index)
            )
            renderers[clip_index] = renderer
        return renderer

    def prepare(clip_index: int, frame_index: int, frame: np.ndarray) -> None:
        """On the export thread, before dispatch: the clip's renderer, and the frame before a segment's first one.

        The rolling-shutter unskew measures motion against the previous frame; at the start of a segment
        (a cut, a clip boundary, the export range) that frame is outside the run, so it is read here on
        purpose — the readers are not thread-safe, which is why this cannot happen inside ``process``.
        """
        renderer = renderer_for(clip_index, frame)
        settings = renderer.settings
        if settings.stabilization.corrects_rolling_shutter_skew and needs_previous_frame(renderer.state, frame_index):
            previous = readers.read(sequence.clips[clip_index].source.path, frame_index - 1)
            if previous is not None:
                prime_with_previous(previous, frame_index - 1, renderer.state)

    def process(frame: np.ndarray, clip_index: int, frame_index: int, sequential: bool) -> np.ndarray:
        renderer = renderers[clip_index]
        trace = RenderTrace()
        rendered = render_frame(
            frame,
            renderer.settings,
            frame_index=frame_index,
            stabilization_state=renderer.state if sequential else None,
            stabilization_reference=renderer.reference,
            processing_context=renderer.context,
            crop=True,
            backend=backend,
            trace=trace,
            wobble_track=renderer.wobble_track,
            mesh_corrections=renderer.mesh_corrections,
        )
        metrics.record(trace.finish())
        return np.ascontiguousarray(ensure_even_size(rendered))

    prepare(first_clip, first_index, first_frame)
    first_processed = process(first_frame, first_clip, first_index, True)
    height, width = first_processed.shape[:2]
    sources = sequence.source_paths
    # A single unretimed clip keeps the proven -ss path (no probe); anything that needs the filter graph
    # (several segments, or a retimed clip whose audio must be stretched) probes each source for real audio.
    retimed = any(segment.clip.settings.capture.is_retimed for segment in request.segments)
    probe = len(request.segments) > 1 or retimed
    has_audio = {path: (source_has_audio(request.ffmpeg_path, path) if probe else True) for path in sources}
    audio = [
        AudioSegment(
            sources.index(segment.clip.source.path),
            *segment.seconds,
            has_audio=has_audio[segment.clip.source.path] and not segment.clip.settings.capture.mute,
            tempo=segment.clip.settings.capture.speed,
        )
        for segment in request.segments
    ]
    command = build_export_command(
        request.ffmpeg_path,
        width=width,
        height=height,
        fps=request.fps,
        sources=sources,
        audio=audio,
        encoder=request.encoder,
        output_path=request.output_path,
    )

    worker_count = request.effective_worker_count
    process_handle = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process_handle.stdin is not None and process_handle.stderr is not None
    stdin = process_handle.stdin
    stderr = _StderrDrain(process_handle.stderr)
    stderr.start()

    processed_frames = 0
    read_failed = False
    previous_cv_threads: int | None = None
    next_log_percent = PROGRESS_LOG_STEP_PERCENT

    def write(frame: np.ndarray) -> None:
        nonlocal processed_frames, next_log_percent
        if cancel.is_set():
            raise _ExportCancelledError()
        stdin.write(np.ascontiguousarray(fit_to_canvas(frame, width, height)).tobytes())
        processed_frames += 1
        report(processed_frames)
        percent = processed_frames * 100 // total
        if percent >= next_log_percent and processed_frames < total:
            elapsed = metrics.elapsed_seconds()
            log_event(
                LOG,
                "export.progress",
                percent=percent,
                processed=processed_frames,
                total=total,
                elapsed_s=round(elapsed, 1),
                fps=round(processed_frames / elapsed, 1) if elapsed > 0 else None,
            )
            next_log_percent = (percent // PROGRESS_LOG_STEP_PERCENT + 1) * PROGRESS_LOG_STEP_PERCENT

    try:
        if worker_count > 1:
            previous_cv_threads = cv2.getNumThreads()
            cv2.setNumThreads(1)

        write(first_processed)

        if worker_count == 1:
            for clip_index, frame_index, frame in frames:
                renderer_for(clip_index, frame)
                write(process(frame, clip_index, frame_index, True))
        else:
            _run_parallel(frames, prepare, process, write, worker_count)
        read_failed = processed_frames < total

        report(processed_frames, force=True)
        stdin.close()
        return_code = process_handle.wait()
    except _ExportCancelledError:
        _terminate(process_handle)
        _discard_output(request.output_path)
        return finish(
            ExportResult(False, CANCELLED_MESSAGE, processed_frames, total, cancelled=True), "export.cancelled"
        )
    except BrokenPipeError:
        return_code = process_handle.wait()
    except Exception as error:  # anything else must still clean up and surface a message
        _terminate(process_handle)
        _discard_output(request.output_path)
        message = f"Falha ao processar frames: {type(error).__name__}: {error}"
        log_event(LOG, "export.error", level=logging.ERROR, exc_info=True, processed=processed_frames)
        return finish(ExportResult(False, message, processed_frames, total), "export.failed", logging.ERROR)
    finally:
        if previous_cv_threads is not None:
            cv2.setNumThreads(previous_cv_threads)

    if return_code != 0:
        _discard_output(request.output_path)
        stderr_text = stderr.text()
        message = stderr_text or "O FFmpeg falhou ao receber os frames processados pelo OpenCV."
        return finish(
            ExportResult(False, message, processed_frames, total),
            "export.failed",
            logging.ERROR,
            return_code=return_code,
            stderr_tail=stderr_text[-2000:],
        )
    if processed_frames == 0:
        _discard_output(request.output_path)
        return finish(ExportResult(False, "Nenhum frame foi exportado.", 0, total), "export.failed", logging.ERROR)
    if read_failed:
        # CAP_PROP_FRAME_COUNT is an estimate; a shorter stream is a valid export, not an error.
        message = f"Exportados {processed_frames} frames (estimativa era {total})."
        return finish(ExportResult(True, message, processed_frames, total), "export.finish", frame_count_mismatch=True)
    return finish(ExportResult(True, "", processed_frames, total), "export.finish")


class _SegmentReader:
    """Decode source frames for one segment, keeping the last few so a blend reuses a neighbour, not a re-seek."""

    def __init__(self, readers: _SourceReaders, path: Path, window: int = 3) -> None:
        self._readers = readers
        self._path = path
        self._window = window
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __call__(self, index: int) -> np.ndarray | None:
        frame = self._cache.get(index)
        if frame is None:
            frame = self._readers.read(self._path, index)
            if frame is not None:
                self._cache[index] = frame
                while len(self._cache) > self._window:
                    self._cache.popitem(last=False)
        return frame


def _iter_frames(readers: _SourceReaders, request: ExportRequest) -> Iterator[tuple[int, int, np.ndarray]]:
    """``(clip index, source frame index, frame)`` for every output frame in order; a short stream skips ahead.

    A clip whose fps differs from the timeline is retimed per its :class:`RetimingMode`: nearest-frame (whole
    frames repeat or drop), a blend, or a motion-interpolated in-between. The reported source index is the
    nearer of a blend's two frames, so the wobble track and the stabilizer's continuity still line up.
    """
    sequence = request.sequence
    for segment in request.segments:
        read = _SegmentReader(readers, segment.clip.source.path)
        previous_single: int | None = None
        for sample in sequence.export_samples(segment):
            composed = retime.compose(sample, read)
            if composed is None:
                break
            if sample.is_single:
                if sample.a == previous_single:
                    composed = composed.copy()  # a repeated nearest frame: a distinct array per output frame
                previous_single = sample.a
            else:
                previous_single = None
            yield segment.clip_index, (sample.a if sample.weight < 0.5 else sample.b), composed


def _run_parallel(
    frames: Iterator[tuple[int, int, np.ndarray]],
    prepare: Callable[[int, int, np.ndarray], object],
    process: Callable[[np.ndarray, int, int, bool], np.ndarray],
    write: Callable[[np.ndarray], None],
    worker_count: int,
) -> None:
    """Frames are rendered concurrently but written strictly in order; ``prepare`` runs here, before dispatch."""
    max_pending = max(1, worker_count * 2)
    pending: deque[Future[np.ndarray]] = deque()
    exhausted = False
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="export-render") as executor:
        while True:
            while not exhausted and len(pending) < max_pending:
                item = next(frames, None)
                if item is None:
                    exhausted = True
                    break
                clip_index, frame_index, frame = item
                prepare(clip_index, frame_index, frame)
                pending.append(executor.submit(process, frame, clip_index, frame_index, False))
            if not pending:
                break
            write(pending.popleft().result())
