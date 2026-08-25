from __future__ import annotations

import numpy as np

from space_video_enhancer.pipeline.geometry import apply_geometry
from space_video_enhancer.pipeline.preview import (
    PreviewRequest,
    RenderMailbox,
    ThumbnailRequest,
    downscale_to_limit,
    render_preview,
    render_thumbnails,
)
from space_video_enhancer.pipeline.render import render_frame
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.settings import (
    FilterSettings,
    GeometrySettings,
    ImageFilterSelection,
    RenderSettings,
    RotationMode,
)


def _preview_request(frame: np.ndarray, generation: int, **overrides: object) -> PreviewRequest:
    fields: dict[str, object] = {
        "generation": generation,
        "frame": frame,
        "frame_index": generation,
        "settings": RenderSettings(),
        "size_limit": None,
        "stabilization_reference": None,
        "stabilization_epoch": 0,
    }
    fields.update(overrides)
    return PreviewRequest(**fields)  # type: ignore[arg-type]


def _thumbnail_request(frame: np.ndarray, generation: int) -> ThumbnailRequest:
    return ThumbnailRequest(generation, frame, RenderSettings(), None, ("original", "shadow_lift"))


def test_mailbox_keeps_only_the_newest_preview(textured_frame: np.ndarray) -> None:
    mailbox = RenderMailbox()
    for generation in range(5):
        mailbox.put_preview(_preview_request(textured_frame, generation))
    job = mailbox.take()
    assert isinstance(job, PreviewRequest) and job.generation == 4
    assert mailbox.dropped == 4
    assert mailbox.take() is None


def test_mailbox_prefers_preview_and_drops_stale_thumbnails(textured_frame: np.ndarray) -> None:
    mailbox = RenderMailbox()
    mailbox.put_thumbnails(_thumbnail_request(textured_frame, 1))
    mailbox.put_preview(_preview_request(textured_frame, 2))
    job = mailbox.take()
    assert isinstance(job, PreviewRequest)
    assert mailbox.take() is None  # thumbnails for the older frame were discarded

    mailbox.put_thumbnails(_thumbnail_request(textured_frame, 3))
    assert isinstance(mailbox.take(), ThumbnailRequest)
    mailbox.put_preview(_preview_request(textured_frame, 4))
    mailbox.clear()
    assert mailbox.take() is None


def test_render_preview_matches_render_frame(textured_frame: np.ndarray) -> None:
    settings = RenderSettings(
        filters=FilterSettings(contrast=60),
        geometry=GeometrySettings(RotationMode.CW_90),
        image_filter=ImageFilterSelection("dynamic_balance", 0.8),
    )
    result = render_preview(_preview_request(textured_frame, 7, settings=settings), StabilizationState(), None)
    assert result.generation == 7 and result.frame_index == 7
    assert np.array_equal(result.after, render_frame(textured_frame, settings))
    assert np.array_equal(result.before, apply_geometry(textured_frame, settings.geometry))
    assert result.elapsed_ms > 0


def test_render_preview_honours_size_limit(textured_frame: np.ndarray) -> None:
    result = render_preview(_preview_request(textured_frame, 1, size_limit=240), None, None)
    assert result.after.shape[1] == 240 and result.before.shape[1] == 240
    assert downscale_to_limit(textured_frame, 10_000) is textured_frame


def test_render_thumbnails_returns_one_frame_per_preset(textured_frame: np.ndarray) -> None:
    result = render_thumbnails(_thumbnail_request(textured_frame, 3), None)
    assert set(result.frames) == {"original", "shadow_lift"}
    assert not np.array_equal(result.frames["original"], result.frames["shadow_lift"])
