"""Qt-free building blocks for asynchronous preview rendering.

The UI puts requests into a :class:`RenderMailbox`; a worker thread drains it. Only the
newest request of each kind survives, so a burst of slider moves costs one render. The
worker also owns the decoder: a preview request normally carries just ``frame_index`` and
the frame is read on the worker thread (see :func:`render_preview`), never on the GUI.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from space_video_enhancer.pipeline.deinterlace import prepare_source
from space_video_enhancer.pipeline.filters import OpenCLBackend
from space_video_enhancer.pipeline.geometry import apply_geometry, fit_to_aspect, normalized_through_geometry
from space_video_enhancer.pipeline.render import render_frame
from space_video_enhancer.pipeline.stabilization.stabilize import needs_previous_frame, prime_with_previous
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import GeometrySettings, ImageFilterSelection, RenderSettings, StabilizationSettings
from space_video_enhancer.telemetry.trace import RenderTrace
from space_video_enhancer.video import VideoSource

DEFAULT_THUMBNAIL_SIZE = (152, 86)


class FrameUnavailableError(RuntimeError):
    """The decoder could not deliver the requested frame (no video open, or the stream ended early)."""


@dataclass(frozen=True)
class PreviewRequest:
    generation: int
    frame: np.ndarray | None
    """An already decoded frame, or ``None`` to have the worker decode ``frame_index`` itself."""
    frame_index: int
    settings: RenderSettings
    """Already adjusted for the preview (output scale capped by the render mode)."""
    size_limit: int | None
    """Downscale the source so its largest side is at most this many pixels (fast/balanced modes)."""
    stabilization_reference: np.ndarray | None
    stabilization_epoch: int
    """Changes whenever the inter-frame stabilization memory must start over."""
    reason: str = "settings"
    """Why the UI asked: open | scrub | playback | settings | mode | idle. Drives log verbosity."""
    requested_at: float = field(default_factory=time.perf_counter)
    source_path: Path | None = None
    """Which open video to decode from (a sequence has several); ``None`` means the only one."""
    canvas_aspect: float | None = None
    """Width/height of the sequence's export canvas: the result is letterboxed to it like the export does."""


@dataclass(frozen=True)
class MotionOverlay:
    """What the stabilizer tracked on this frame, in normalized coordinates of the displayed ("Antes") frame.

    ``kind`` is ``"flow"`` for inter-frame vectors (previous -> current; ``inliers`` marks the ones the
    camera hypothesis kept) or ``"lock"`` for the camera lock's tracked points (``previous == current``).
    """

    previous: np.ndarray
    current: np.ndarray
    inliers: np.ndarray
    kind: str

    @property
    def count(self) -> int:
        return len(self.current)

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inliers))


def motion_overlay(
    state: StabilizationState | None, settings: StabilizationSettings, geometry: GeometrySettings
) -> MotionOverlay | None:
    """Overlay of the last estimate held by ``state``; ``None`` when nothing was tracked or stabilization is off."""
    if state is None or not settings.enabled:
        return None
    vectors = state.last_vectors
    if vectors is not None and settings.mode.uses_auto_motion:
        width, height = vectors.size
        scale = np.array([1.0 / max(1, width), 1.0 / max(1, height)])
        previous = normalized_through_geometry(vectors.previous * scale, geometry)
        current = normalized_through_geometry(vectors.current * scale, geometry)
        return MotionOverlay(previous, current, vectors.inliers.astype(bool), "flow")
    track = state.lock
    if track is not None and settings.mode.uses_focal_point:
        height, width = track.prev_gray.shape[:2]
        scale = np.array([1.0 / max(1, width), 1.0 / max(1, height)])
        current = normalized_through_geometry(track.current_points * scale, geometry)
        return MotionOverlay(current, current, np.ones(len(current), dtype=bool), "lock")
    return None


@dataclass(frozen=True)
class PreviewResult:
    generation: int
    frame_index: int
    source: np.ndarray
    """The decoded frame the render started from (full resolution): reference captures, thumbnails."""
    before: np.ndarray
    after: np.ndarray
    elapsed_ms: float
    queue_wait_ms: float
    """Time the request sat in the mailbox (or behind another render) before starting."""
    trace: RenderTrace
    decode_ms: float = 0.0
    decode_sequential: bool | None = None
    """``None`` when the request already carried a frame (nothing was decoded here)."""
    video_seeks: int = 0
    video_sequential_reads: int = 0
    """Decoder counters after this read, so the UI can report them without touching the decoder."""
    motion: MotionOverlay | None = None
    """What the stabilizer tracked on this frame (for the diagnostic overlay)."""


@dataclass(frozen=True)
class ThumbnailRequest:
    generation: int
    frame: np.ndarray
    """Already thumbnail-sized (in device pixels)."""
    settings: RenderSettings
    stabilization_reference: np.ndarray | None
    preset_ids: tuple[str, ...]
    icon_size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE
    """Target size in device pixels (logical icon size times the display's pixel ratio)."""
    device_pixel_ratio: float = 1.0


@dataclass(frozen=True)
class ThumbnailResult:
    generation: int
    frames: dict[str, np.ndarray]
    elapsed_ms: float


RenderJob = Union[PreviewRequest, ThumbnailRequest]


def downscale_to_limit(frame: np.ndarray, size_limit: int | None) -> np.ndarray:
    if size_limit is None:
        return frame
    largest = max(frame.shape[1], frame.shape[0])
    if largest <= 0:
        return frame
    scale = min(1.0, size_limit / largest)
    if scale >= 0.999:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def render_preview(
    request: PreviewRequest,
    state: StabilizationState | None,
    backend: OpenCLBackend | None,
    video: VideoSource | None = None,
) -> PreviewResult:
    """Decode (if needed) and render one preview frame; raises :class:`FrameUnavailableError` when it cannot."""
    started = time.perf_counter()
    queue_wait_ms = (started - request.requested_at) * 1000.0
    trace = RenderTrace()
    trace.note("preview.reason", request.reason)
    trace.note("preview.size_limit", request.size_limit)

    frame = request.frame
    decode_ms = 0.0
    decode_sequential: bool | None = None
    seeks = sequential_reads = 0
    if frame is None:
        if video is None:
            raise FrameUnavailableError("nenhum vídeo aberto para decodificar o frame")
        with trace.stage("decode"):
            frame = video.read_frame(request.frame_index)
        if frame is None:
            raise FrameUnavailableError(f"o frame {request.frame_index + 1} não pôde ser decodificado")
        decode_ms = video.last_decode_ms
        decode_sequential = video.last_read_was_sequential
        seeks, sequential_reads = video.seeks, video.sequential_reads
        trace.note("decode.sequential", decode_sequential)

    with trace.stage("capture"):  # on the native rows: a resize would blur the comb away and back
        prepared, comb = prepare_source(frame, request.settings.capture.deinterlace)
    trace.note("deinterlace.mode", request.settings.capture.deinterlace.value)
    trace.note("deinterlace.comb", round(comb, 4))
    source = downscale_to_limit(prepared, request.size_limit)
    if (
        request.settings.stabilization.corrects_rolling_shutter
        and video is not None
        and needs_previous_frame(state, request.frame_index)
    ):
        # Paused on a frame (or after a jump) there is nothing to measure motion against: fetch the frame before.
        assert state is not None
        with trace.stage("previous"):
            previous = video.read_frame(request.frame_index - 1)
        if previous is not None:
            previous_prepared, _comb = prepare_source(previous, request.settings.capture.deinterlace)
            prime_with_previous(
                downscale_to_limit(previous_prepared, request.size_limit), request.frame_index - 1, state
            )
    reference = request.stabilization_reference
    if reference is not None:
        reference = downscale_to_limit(reference, request.size_limit)

    with trace.stage("before"):
        before = apply_geometry(downscale_to_limit(frame, request.size_limit), request.settings.geometry)
    after = render_frame(
        source,
        request.settings,
        frame_index=request.frame_index,
        stabilization_state=state,
        stabilization_reference=reference,
        backend=backend,
        trace=trace,
        source_prepared=True,
    )
    if request.canvas_aspect is not None:
        with trace.stage("letterbox"):
            after = fit_to_aspect(after, request.canvas_aspect)
    trace.finish()
    return PreviewResult(
        request.generation,
        request.frame_index,
        frame,
        before,
        after,
        trace.total_ms,
        queue_wait_ms,
        trace,
        decode_ms,
        decode_sequential,
        seeks,
        sequential_reads,
        motion_overlay(state, request.settings.stabilization, request.settings.geometry),
    )


def render_thumbnails(request: ThumbnailRequest, backend: OpenCLBackend | None) -> ThumbnailResult:
    started = time.perf_counter()
    frames = {
        preset_id: render_frame(
            request.frame,
            request.settings,
            image_filter=ImageFilterSelection(preset_id, 1.0),
            stabilization_reference=request.stabilization_reference,
            backend=backend,
        )
        for preset_id in request.preset_ids
    }
    return ThumbnailResult(request.generation, frames, (time.perf_counter() - started) * 1000.0)


class RenderMailbox:
    """Thread-safe, latest-wins slots for one preview and one thumbnail request.

    A new preview request also drops any pending thumbnails: the UI re-requests them
    once the preview lands, so rendering the old ones would be wasted work.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._preview: PreviewRequest | None = None
        self._thumbnails: ThumbnailRequest | None = None
        self.dropped = 0

    def put_preview(self, request: PreviewRequest) -> None:
        with self._lock:
            if self._preview is not None:
                self.dropped += 1
            self._preview = request
            self._thumbnails = None

    def put_thumbnails(self, request: ThumbnailRequest) -> None:
        with self._lock:
            if self._thumbnails is not None:
                self.dropped += 1
            self._thumbnails = request

    def take(self) -> RenderJob | None:
        """Newest preview request first, then thumbnails; ``None`` when idle."""
        with self._lock:
            if self._preview is not None:
                job: RenderJob | None = self._preview
                self._preview = None
                return job
            job = self._thumbnails
            self._thumbnails = None
            return job

    def clear(self) -> None:
        with self._lock:
            self._preview = None
            self._thumbnails = None
