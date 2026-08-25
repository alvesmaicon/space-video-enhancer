"""Qt-free export core: decode -> render_frame -> raw frames into FFmpeg.

Runs on a worker thread. Reports progress through a callback, cooperates with a
cancel event, never leaves FFmpeg running or a half-written file behind, and logs
``export.start`` / ``export.progress`` / ``export.finish`` (or ``.cancelled`` /
``.failed``) with per-stage timing statistics and stabilization estimator histograms.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from space_video_enhancer.export.ffmpeg import EncoderChoice, build_export_command, source_has_audio
from space_video_enhancer.pipeline.filters import OpenCLBackend
from space_video_enhancer.pipeline.geometry import ensure_even_size
from space_video_enhancer.pipeline.render import build_context, render_frame
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import RenderSettings
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.telemetry.stats import Histogram, StageStats
from space_video_enhancer.telemetry.trace import RenderTrace
from space_video_enhancer.timeline import FrameRange, frame_count, kept_segments, segments_seconds

LOG = get_logger("export")
ProgressCallback = Callable[[int, int], None]

PROGRESS_INTERVAL_SECONDS = 0.1
PROGRESS_LOG_STEP_PERCENT = 10
STDERR_TAIL_BYTES = 64 * 1024
CANCELLED_MESSAGE = "Exportacao cancelada pelo usuario."
FFMPEG_TERMINATE_TIMEOUT = 1.0


@dataclass(frozen=True)
class ExportRequest:
    source_path: Path
    output_path: Path
    settings: RenderSettings
    start_frame: int
    end_frame: int
    fps: float
    worker_count: int
    ffmpeg_path: str
    encoder: EncoderChoice
    stabilization_reference: np.ndarray | None = None
    cuts: tuple[tuple[int, int], ...] = ()
    """Frame ranges (inclusive, absolute) removed from inside the clip; see :mod:`timeline`."""

    @property
    def clip(self) -> FrameRange:
        return FrameRange(self.start_frame, self.end_frame)

    @property
    def segments(self) -> tuple[FrameRange, ...]:
        """What actually gets exported: the clip minus the cuts, in order."""
        return kept_segments(self.clip, self.cuts)

    @property
    def segments_seconds(self) -> tuple[tuple[float, float], ...]:
        return segments_seconds(self.segments, self.fps)

    @property
    def expected_frames(self) -> int:
        return max(1, frame_count(self.segments))

    @property
    def start_seconds(self) -> float:
        return self.start_frame / self.fps if self.fps > 0 else 0.0

    @property
    def duration_seconds(self) -> float:
        return self.expected_frames / self.fps if self.fps > 0 else 0.0

    @property
    def effective_worker_count(self) -> int:
        """Inter-frame stabilization / smoothing needs frames in order: one worker."""
        if self.settings.stabilization.requires_sequential_processing:
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


def _request_summary(request: ExportRequest) -> dict[str, Any]:
    return {
        "source": request.source_path.name,
        "output": request.output_path.name,
        "frames": request.expected_frames,
        "start_frame": request.start_frame,
        "end_frame": request.end_frame,
        "cuts": [[cut[0], cut[1]] for cut in request.cuts],
        "segments": [[segment.start, segment.end] for segment in request.segments],
        "fps": round(request.fps, 3),
        "workers": request.effective_worker_count,
        "requested_workers": request.worker_count,
        "encoder": request.encoder.label,
        "settings": asdict(request.settings),
    }


def run_export(
    request: ExportRequest,
    *,
    progress: ProgressCallback,
    cancel: threading.Event,
    backend: OpenCLBackend | None = None,
) -> ExportResult:
    """Blocking export. Safe to call from any thread; touches no Qt objects."""
    log_event(LOG, "export.start", **_request_summary(request))
    capture = cv2.VideoCapture(str(request.source_path))
    if not capture.isOpened():
        message = "Nao foi possivel reabrir o video para exportacao."
        log_event(LOG, "export.failed", level=logging.ERROR, message=message, output=request.output_path.name)
        return ExportResult(False, message, 0, request.expected_frames)

    try:
        return _run_export(capture, request, progress, cancel, backend)
    finally:
        capture.release()


def _run_export(
    capture: cv2.VideoCapture,
    request: ExportRequest,
    progress: ProgressCallback,
    cancel: threading.Event,
    backend: OpenCLBackend | None,
) -> ExportResult:
    total = request.expected_frames
    report = _Throttle(progress, total)
    settings = request.settings
    metrics = _ExportMetrics(request, backend)

    def finish(result: ExportResult, event: str, level: int = logging.INFO, **extra: Any) -> ExportResult:
        summary = metrics.summary(result.processed_frames, request.output_path)
        log_event(LOG, event, level=level, output=request.output_path.name, message=result.message, **summary, **extra)
        return ExportResult(
            result.success, result.message, result.processed_frames, result.expected_frames, result.cancelled, summary
        )

    if not request.segments:
        message = "Os cortes removem o video inteiro: nada a exportar."
        return finish(ExportResult(False, message, 0, total), "export.failed", logging.ERROR)
    frames = _iter_frames(capture, request)
    first = next(frames, None)
    if first is None:
        message = "Nao foi possivel ler o frame inicial do corte selecionado."
        return finish(ExportResult(False, message, 0, total), "export.failed", logging.ERROR)
    first_index, first_frame = first

    # Denoise strength is resolved once, from the first frame, so it stays constant across the clip.
    context = build_context(settings, first_frame)
    state = StabilizationState()

    def process(frame: np.ndarray, frame_index: int, frame_state: StabilizationState | None) -> np.ndarray:
        trace = RenderTrace()
        rendered = render_frame(
            frame,
            settings,
            frame_index=frame_index,
            stabilization_state=frame_state,
            stabilization_reference=request.stabilization_reference,
            processing_context=context,
            crop=True,
            backend=backend,
            trace=trace,
        )
        metrics.record(trace.finish())
        return np.ascontiguousarray(ensure_even_size(rendered))

    first_processed = process(first_frame, first_index, state)
    height, width = first_processed.shape[:2]
    segments = request.segments_seconds
    has_audio = source_has_audio(request.ffmpeg_path, request.source_path) if len(segments) > 1 else True
    command = build_export_command(
        request.ffmpeg_path,
        width=width,
        height=height,
        fps=request.fps,
        source_path=request.source_path,
        start_seconds=request.start_seconds,
        duration_seconds=request.duration_seconds,
        encoder=request.encoder,
        output_path=request.output_path,
        segments_seconds=segments,
        has_audio=has_audio,
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
        stdin.write(frame.tobytes())
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
            for frame_index, frame in frames:
                write(process(frame, frame_index, state))
        else:
            _run_parallel(frames, process, write, worker_count)
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


def _iter_frames(capture: cv2.VideoCapture, request: ExportRequest) -> Iterator[tuple[int, np.ndarray]]:
    """Decoded frames of every kept segment in order, seeking only where a cut was removed."""
    position: int | None = None
    for segment in request.segments:
        if position != segment.start:
            capture.set(cv2.CAP_PROP_POS_FRAMES, segment.start)
        for frame_index in range(segment.start, segment.end + 1):
            ok, frame = capture.read()
            if not ok:
                return
            position = frame_index + 1
            yield frame_index, frame


def _run_parallel(
    frames: Iterator[tuple[int, np.ndarray]],
    process: Callable[[np.ndarray, int, StabilizationState | None], np.ndarray],
    write: Callable[[np.ndarray], None],
    worker_count: int,
) -> None:
    """Frames are rendered concurrently but written strictly in order."""
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
                frame_index, frame = item
                pending.append(executor.submit(process, frame, frame_index, None))
            if not pending:
                break
            write(pending.popleft().result())
