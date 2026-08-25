"""The single frame pipeline shared by preview, thumbnails and export.

Order: stabilize -> basic filters (incl. output scale) -> image preset -> geometry -> crop.
Crop is last so the rectangle drawn on the (rotated/flipped) preview means the same
thing on export.
"""

from __future__ import annotations

import numpy as np

from space_video_enhancer.pipeline.filters import (
    OpenCLBackend,
    ProcessingContext,
    apply_basic_filters,
    build_processing_context,
)
from space_video_enhancer.pipeline.geometry import apply_crop, apply_geometry
from space_video_enhancer.pipeline.presets import apply_image_filter_preset
from space_video_enhancer.pipeline.stabilization.stabilize import apply_stabilization
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import ImageFilterSelection, RenderSettings, clamp_unit


def build_context(settings: RenderSettings, reference_frame: np.ndarray | None) -> ProcessingContext:
    """Resolve filter parameters; ``reference_frame`` feeds the auto/sample denoise estimate."""
    return build_processing_context(settings.filters, reference_frame, settings.denoise_sample_rect)


def render_frame(
    frame: np.ndarray,
    settings: RenderSettings,
    *,
    frame_index: int | None = None,
    stabilization_state: StabilizationState | None = None,
    stabilization_reference: np.ndarray | None = None,
    processing_context: ProcessingContext | None = None,
    image_filter: ImageFilterSelection | None = None,
    crop: bool = False,
    backend: OpenCLBackend | None = None,
) -> np.ndarray:
    """Run the full pipeline on one frame.

    - ``processing_context``: pass a precomputed context to keep denoise strength constant
      across an export; leave ``None`` to resolve it from this frame (preview behaviour).
    - ``image_filter``: override the preset selection (used by the thumbnail strip).
    - ``crop``: apply the crop (export); the preview draws it as an overlay instead.
    """
    stabilized = apply_stabilization(
        frame,
        settings.stabilization,
        frame_index=frame_index,
        state=stabilization_state,
        reference_frame=stabilization_reference,
    )
    context = processing_context if processing_context is not None else build_context(settings, stabilized)
    processed = apply_basic_filters(stabilized, context, backend)

    selection = image_filter if image_filter is not None else settings.image_filter
    processed = apply_image_filter_preset(processed, selection.preset_id, clamp_unit(selection.intensity))

    processed = apply_geometry(processed, settings.geometry)
    if crop:
        processed = apply_crop(processed, settings.crop)
    return processed
