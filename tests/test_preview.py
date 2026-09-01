from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from space_video_enhancer.pipeline.geometry import apply_geometry
from space_video_enhancer.pipeline.preview import (
    FrameUnavailableError,
    PreviewRequest,
    RenderMailbox,
    ThumbnailRequest,
    downscale_to_limit,
    motion_overlay,
    render_preview,
    render_thumbnails,
)
from space_video_enhancer.pipeline.render import render_frame
from space_video_enhancer.pipeline.stabilization.mesh import VERTEX_COLUMNS, VERTEX_ROWS, MeshCorrections
from space_video_enhancer.pipeline.stabilization.state import StabilizationState
from space_video_enhancer.pipeline.stabilization.wobble import analyze_wobble, wobble_window_frames
from space_video_enhancer.settings import (
    FilterSettings,
    GeometrySettings,
    ImageFilterSelection,
    RenderSettings,
    RotationMode,
    StabilizationMode,
    StabilizationSettings,
)
from space_video_enhancer.video import VideoSource


def _preview_request(frame: np.ndarray | None, generation: int, **overrides: object) -> PreviewRequest:
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


def test_render_preview_decodes_the_requested_frame_on_demand(ui_video: Path) -> None:
    video = VideoSource(ui_video)
    try:
        result = render_preview(_preview_request(None, 3, frame_index=5), StabilizationState(), None, video=video)
        assert result.frame_index == 5 and result.source.shape[:2] == (180, 320)
        assert result.decode_sequential is True  # 5 frames ahead of the decoder: grab(), no seek
        assert result.decode_ms >= 0.0 and "decode" in result.trace.stages
        assert (result.video_seeks, result.video_sequential_reads) == (0, 1)
        assert np.array_equal(result.after, render_frame(result.source, RenderSettings()))
    finally:
        video.release()


def test_render_preview_without_a_frame_needs_a_video() -> None:
    with pytest.raises(FrameUnavailableError):
        render_preview(_preview_request(None, 1, frame_index=0), StabilizationState(), None, video=None)


def test_render_preview_letterboxes_to_the_canvas_aspect(textured_frame: np.ndarray) -> None:
    frame = textured_frame[:, :360]  # 360x270, 4:3
    result = render_preview(_preview_request(frame, 1, canvas_aspect=16 / 9), None, None)
    assert result.after.shape[:2] == (270, 480)
    assert result.after[:, :50].max() == 0 and result.after[:, 60:420].mean() > 20
    assert result.before.shape[:2] == (270, 360), "the 'before' image stays the source"
    assert "letterbox" in result.trace.as_dict()["stages"]
    same = render_preview(_preview_request(frame, 2, canvas_aspect=4 / 3), None, None)
    assert same.after.shape[:2] == (270, 360)


def test_render_preview_carries_the_stabilizer_motion_overlay(textured_frame: np.ndarray) -> None:
    settings = RenderSettings(
        stabilization=StabilizationSettings(mode=StabilizationMode.AUTO, temporal_smoothing=0.0, auto_crop=False)
    )
    state = StabilizationState()
    previous = np.roll(textured_frame, (4, -6), axis=(0, 1))
    first = render_preview(_preview_request(previous, 1, frame_index=1, settings=settings), state, None)
    assert first.motion is None, "the first frame only primes the estimator"
    second = render_preview(_preview_request(textured_frame, 2, frame_index=2, settings=settings), state, None)
    assert second.motion is not None and second.motion.kind == "flow"
    assert second.motion.inlier_count >= 6 and second.motion.inlier_count <= second.motion.count
    assert np.all((second.motion.current >= -0.05) & (second.motion.current <= 1.05)), "normalized coordinates"
    off = render_preview(_preview_request(textured_frame, 3, frame_index=3), state, None)
    assert off.motion is None, "no overlay without stabilization"


def test_motion_overlay_shows_the_analysed_track_once_the_field_runs(textured_frame: np.ndarray) -> None:
    """With the field applied nothing is tracked live any more: the overlay comes from the analysis payload."""
    fields = np.zeros((4, VERTEX_ROWS, VERTEX_COLUMNS, 2), dtype=np.float32)
    motion = np.zeros_like(fields)
    motion[..., 0] = 0.01
    anchors = np.full((4, 5, 2), 0.25, dtype=np.float32)
    corrections = MeshCorrections(0, fields, motion=motion, anchor_points=anchors)
    geometry = GeometrySettings()

    mesh = StabilizationSettings(mode=StabilizationMode.MESH)
    flow = motion_overlay(StabilizationState(), mesh, geometry, corrections, 1)
    assert flow is not None and flow.kind == "flow"
    assert flow.count == VERTEX_ROWS * VERTEX_COLUMNS, "one vector per mesh vertex"
    assert np.allclose(flow.current - flow.previous, (0.01, 0.0), atol=1e-6), "the measured per-frame motion"

    lock = motion_overlay(None, StabilizationSettings(mode=StabilizationMode.OBJECT), geometry, corrections, 2)
    assert lock is not None and lock.kind == "lock" and lock.count == 5, "the object mode shows its anchor samples"
    assert np.allclose(lock.current, 0.25)

    assert motion_overlay(None, mesh, geometry, corrections, 99) is None, "outside the analysed range"
    auto = StabilizationSettings(mode=StabilizationMode.AUTO)
    assert motion_overlay(StabilizationState(), auto, geometry, corrections, 1) is None, "not a mesh mode"

    request = _preview_request(
        textured_frame,
        5,
        frame_index=1,
        settings=RenderSettings(stabilization=mesh),
        mesh_corrections=corrections,
    )
    result = render_preview(request, StabilizationState(), None)
    assert result.motion is not None and result.motion.kind == "flow", "render_preview hands the payload through"


def test_render_preview_fetches_the_previous_frame_for_the_rolling_shutter_when_paused(ui_video: Path) -> None:
    """Paused on frame 5 with a fresh memory there is nothing to measure motion against: the worker reads frame 4."""
    settings = RenderSettings(stabilization=StabilizationSettings(rolling_shutter=0.8))
    video = VideoSource(ui_video)
    try:
        state = StabilizationState()
        result = render_preview(_preview_request(None, 1, frame_index=5, settings=settings), state, None, video=video)
        assert "previous" in result.trace.stages, "the frame before was decoded on purpose"
        assert result.trace.notes["rolling_shutter.result"] in ("applied", "negligible")
        assert "rolling_shutter.shift_px" in result.trace.notes
        again = render_preview(_preview_request(None, 2, frame_index=5, settings=settings), state, None, video=video)
        assert "previous" not in again.trace.stages, "the same frame again: the motion is remembered"
        assert again.trace.notes["rolling_shutter.result"] == result.trace.notes["rolling_shutter.result"]

        first = render_preview(
            _preview_request(None, 3, frame_index=0, settings=settings), StabilizationState(), None, video=video
        )
        assert first.trace.notes["rolling_shutter.result"] == "no_motion", "frame 0 has no frame before it"
        plain = render_preview(_preview_request(None, 4, frame_index=5), StabilizationState(), None, video=video)
        assert "previous" not in plain.trace.stages, "without the unskew nothing extra is decoded"
    finally:
        video.release()


def test_render_preview_undoes_the_wobble_from_the_clip_track_without_the_previous_frame(ui_video: Path) -> None:
    settings = RenderSettings(stabilization=StabilizationSettings(rolling_shutter_auto=True))
    video = VideoSource(ui_video)
    try:
        track = analyze_wobble(video.read_frame, ui_video, 0, 11, wobble_window_frames(24.0)).track
        assert track is not None and track.count == 12
        request = _preview_request(None, 1, frame_index=5, settings=settings, wobble_track=track)
        result = render_preview(request, StabilizationState(), None, video=video)
        assert "previous" not in result.trace.stages, "the wobble needs no motion estimate of its own"
        assert "rolling_shutter.wobble_px" in result.trace.notes
        assert result.trace.notes["rolling_shutter.result"] in ("applied", "negligible")

        bare = render_preview(
            _preview_request(None, 2, frame_index=5, settings=settings), StabilizationState(), None, video=video
        )
        assert bare.trace.notes["rolling_shutter.result"] == "unanalysed", "no track yet: the frame passes as is"
        assert bare.trace.notes["rolling_shutter.wobble"] == "unanalysed" and "previous" not in bare.trace.stages
    finally:
        video.release()
