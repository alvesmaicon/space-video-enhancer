"""The single frame pipeline shared by preview, thumbnails and export.

Order: capture fixes (deinterlace, at native size) -> exposure lock metering (on the source frame) ->
stabilize (with rolling-shutter unskew) -> basic filters (exposure gains, white balance, tonal, hue
bands, scale) -> image preset -> geometry -> crop.
Crop is last so the rectangle drawn on the (rotated/flipped) preview means the same
thing on export.

Pass a :class:`RenderTrace` to get per-stage timings and the decisions taken along the
way (which stabilization estimator ran, how many inliers, denoise strength, backend...).
"""

from __future__ import annotations

import numpy as np

from space_video_enhancer.pipeline.anonymize import apply_anonymize
from space_video_enhancer.pipeline.deinterlace import prepare_source
from space_video_enhancer.pipeline.exposure import exposure_lock_table
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
from space_video_enhancer.pipeline.stabilization.wobble import WobbleTrack
from space_video_enhancer.settings import ImageFilterSelection, RenderSettings, clamp_unit
from space_video_enhancer.telemetry import trace as tracing
from space_video_enhancer.telemetry.trace import RenderTrace


def build_context(settings: RenderSettings, reference_frame: np.ndarray | None) -> ProcessingContext:
    """Resolve filter parameters; ``reference_frame`` feeds the auto/sample denoise estimate."""
    return build_processing_context(settings.filters, reference_frame, settings.denoise_sample_rect, settings.grade)


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
    trace: RenderTrace | None = None,
    source_prepared: bool = False,
    wobble_track: WobbleTrack | None = None,
) -> np.ndarray:
    """Run the full pipeline on one frame.

    - ``processing_context``: pass a precomputed context to keep denoise strength constant
      across an export; leave ``None`` to resolve it from this frame (preview behaviour).
    - ``image_filter``: override the preset selection (used by the thumbnail strip).
    - ``crop``: apply the crop (export); the preview draws it as an overlay instead.
    - ``trace``: collect stage timings and pipeline decisions for this render.
    - ``source_prepared``: the caller already ran the capture stage on the full-size frame (the
      preview does, before downscaling — deinterlacing must see the native rows).
    - ``wobble_track``: the clip's rolling-shutter wobble analysis, when the settings ask for that correction.
    """
    with tracing.active(trace):
        tracing.note("input_shape", list(frame.shape[:2]))

        if not source_prepared:
            with tracing.stage("capture"):
                frame, comb = prepare_source(frame, settings.capture.deinterlace)
            tracing.note("deinterlace.mode", settings.capture.deinterlace.value)
            tracing.note("deinterlace.comb", round(comb, 4))

        with tracing.stage("exposure"):
            # Metered on the source frame so the region matches the rectangle on the "before" preview.
            exposure_table = exposure_lock_table(frame, settings.exposure)

        with tracing.stage("stabilize"):
            stabilized = apply_stabilization(
                frame,
                settings.stabilization,
                frame_index=frame_index,
                state=stabilization_state,
                reference_frame=stabilization_reference,
                wobble_track=wobble_track,
            )

        with tracing.stage("context"):
            context = processing_context if processing_context is not None else build_context(settings, stabilized)

        with tracing.stage("filters"):
            processed = apply_basic_filters(stabilized, context, backend, exposure_table)

        selection = image_filter if image_filter is not None else settings.image_filter
        with tracing.stage("preset"):
            processed = apply_image_filter_preset(processed, selection.preset_id, clamp_unit(selection.intensity))
        tracing.note("preset", selection.preset_id)

        if settings.anonymize.is_active:
            with tracing.stage("anonymize"):
                processed = apply_anonymize(processed, settings.anonymize)
            tracing.note("anonymize.mode", settings.anonymize.mode.value)

        with tracing.stage("geometry"):
            processed = apply_geometry(processed, settings.geometry)

        if crop:
            with tracing.stage("crop"):
                processed = apply_crop(processed, settings.crop)

        tracing.note("output_shape", list(processed.shape[:2]))
        return processed
