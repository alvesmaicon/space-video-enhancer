"""Qt-free building blocks for asynchronous preview rendering.

The UI puts requests into a :class:`RenderMailbox`; a worker thread drains it. Only the
newest request of each kind survives, so a burst of slider moves costs one render.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np

from space_video_enhancer.pipeline.filters import OpenCLBackend
from space_video_enhancer.pipeline.geometry import apply_geometry
from space_video_enhancer.pipeline.render import render_frame
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import ImageFilterSelection, RenderSettings


@dataclass(frozen=True)
class PreviewRequest:
    generation: int
    frame: np.ndarray
    frame_index: int
    settings: RenderSettings
    """Already adjusted for the preview (output scale capped by the render mode)."""
    size_limit: int | None
    """Downscale the source so its largest side is at most this many pixels (fast/balanced modes)."""
    stabilization_reference: np.ndarray | None
    stabilization_epoch: int
    """Changes whenever the inter-frame stabilization memory must start over."""


@dataclass(frozen=True)
class PreviewResult:
    generation: int
    frame_index: int
    before: np.ndarray
    after: np.ndarray
    elapsed_ms: float


@dataclass(frozen=True)
class ThumbnailRequest:
    generation: int
    frame: np.ndarray
    """Already thumbnail-sized."""
    settings: RenderSettings
    stabilization_reference: np.ndarray | None
    preset_ids: tuple[str, ...]


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
) -> PreviewResult:
    started = time.perf_counter()
    source = downscale_to_limit(request.frame, request.size_limit)
    reference = request.stabilization_reference
    if reference is not None:
        reference = downscale_to_limit(reference, request.size_limit)

    before = apply_geometry(source, request.settings.geometry)
    after = render_frame(
        source,
        request.settings,
        frame_index=request.frame_index,
        stabilization_state=state,
        stabilization_reference=reference,
        backend=backend,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return PreviewResult(request.generation, request.frame_index, before, after, elapsed_ms)


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
