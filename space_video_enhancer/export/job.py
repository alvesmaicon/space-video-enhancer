"""Qt-free export core: decode -> render_frame -> raw frames into FFmpeg.

Runs on a worker thread. Reports progress through a callback, cooperates with a
cancel event, and never leaves FFmpeg running or a half-written file behind.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from space_video_enhancer.export.ffmpeg import EncoderChoice, build_export_command
from space_video_enhancer.pipeline.filters import OpenCLBackend
from space_video_enhancer.pipeline.geometry import ensure_even_size
from space_video_enhancer.pipeline.render import build_context, render_frame
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import RenderSettings

ProgressCallback = Callable[[int, int], None]

PROGRESS_INTERVAL_SECONDS = 0.1
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

    @property
    def expected_frames(self) -> int:
        return max(1, self.end_frame - self.start_frame + 1)

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


def run_export(
    request: ExportRequest,
    *,
    progress: ProgressCallback,
    cancel: threading.Event,
    backend: OpenCLBackend | None = None,
) -> ExportResult:
    """Blocking export. Safe to call from any thread; touches no Qt objects."""
    capture = cv2.VideoCapture(str(request.source_path))
    if not capture.isOpened():
        return ExportResult(False, "Nao foi possivel reabrir o video para exportacao.", 0, request.expected_frames)

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

    capture.set(cv2.CAP_PROP_POS_FRAMES, request.start_frame)
    ok, first_frame = capture.read()
    if not ok:
        return ExportResult(False, "Nao foi possivel ler o frame inicial do corte selecionado.", 0, total)

    # Denoise strength is resolved once, from the first frame, so it stays constant across the clip.
    context = build_context(settings, first_frame)
    state = StabilizationState()

    def process(frame: np.ndarray, frame_index: int, frame_state: StabilizationState | None) -> np.ndarray:
        rendered = render_frame(
            frame,
            settings,
            frame_index=frame_index,
            stabilization_state=frame_state,
            stabilization_reference=request.stabilization_reference,
            processing_context=context,
            crop=True,
            backend=backend,
        )
        return np.ascontiguousarray(ensure_even_size(rendered))

    first_processed = process(first_frame, request.start_frame, state)
    height, width = first_processed.shape[:2]
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

    def write(frame: np.ndarray) -> None:
        nonlocal processed_frames
        if cancel.is_set():
            raise _ExportCancelledError()
        stdin.write(frame.tobytes())
        processed_frames += 1
        report(processed_frames)

    try:
        if worker_count > 1:
            previous_cv_threads = cv2.getNumThreads()
            cv2.setNumThreads(1)

        write(first_processed)

        if worker_count == 1:
            while processed_frames < total:
                ok, frame = capture.read()
                if not ok:
                    read_failed = True
                    break
                write(process(frame, request.start_frame + processed_frames, state))
        else:
            read_failed = _run_parallel(capture, request, process, write, worker_count, total)

        report(processed_frames, force=True)
        stdin.close()
        return_code = process_handle.wait()
    except _ExportCancelledError:
        _terminate(process_handle)
        _discard_output(request.output_path)
        return ExportResult(False, CANCELLED_MESSAGE, processed_frames, total, cancelled=True)
    except BrokenPipeError:
        return_code = process_handle.wait()
    except Exception as error:
        _terminate(process_handle)
        _discard_output(request.output_path)
        return ExportResult(
            False, f"Falha ao processar frames: {type(error).__name__}: {error}", processed_frames, total
        )
    finally:
        if previous_cv_threads is not None:
            cv2.setNumThreads(previous_cv_threads)

    if return_code != 0:
        _discard_output(request.output_path)
        return ExportResult(
            False,
            stderr.text() or "O FFmpeg falhou ao receber os frames processados pelo OpenCV.",
            processed_frames,
            total,
        )
    if processed_frames == 0:
        _discard_output(request.output_path)
        return ExportResult(False, "Nenhum frame foi exportado.", 0, total)
    if read_failed:
        # CAP_PROP_FRAME_COUNT is an estimate; a shorter stream is a valid export, not an error.
        return ExportResult(
            True, f"Exportados {processed_frames} frames (estimativa era {total}).", processed_frames, total
        )
    return ExportResult(True, "", processed_frames, total)


def _run_parallel(
    capture: cv2.VideoCapture,
    request: ExportRequest,
    process: Callable[[np.ndarray, int, StabilizationState | None], np.ndarray],
    write: Callable[[np.ndarray], None],
    worker_count: int,
    total: int,
) -> bool:
    """Frames are rendered concurrently but written strictly in order. Returns ``read_failed``."""
    max_pending = max(1, worker_count * 2)
    pending: deque[Future[np.ndarray]] = deque()
    submitted = 1
    read_failed = False
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="export-render") as executor:
        while True:
            while not read_failed and submitted < total and len(pending) < max_pending:
                ok, frame = capture.read()
                if not ok:
                    read_failed = True
                    break
                pending.append(executor.submit(process, frame, request.start_frame + submitted, None))
                submitted += 1
            if not pending:
                break
            write(pending.popleft().result())
    return read_failed
