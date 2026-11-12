"""The preview: zoom and render quality, refresh scheduling, results from the render thread, thumbnails, HUD."""

from __future__ import annotations

import time
from dataclasses import replace

import cv2
import numpy as np
from PySide6.QtCore import Slot

from space_video_enhancer.pipeline.presets import IMAGE_FILTER_PRESETS
from space_video_enhancer.pipeline.preview import PreviewRequest, ThumbnailRequest
from space_video_enhancer.settings import (
    DenoiseMode,
    PreviewRenderMode,
    RenderSettings,
)
from space_video_enhancer.telemetry.diff import diff as diff_settings
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.ui.images import device_pixel_ratio
from space_video_enhancer.ui.preview_worker import PreviewImages, ThumbnailImages
from space_video_enhancer.ui.sections.state import WindowState
from space_video_enhancer.ui.tabs.base import TabPage, combo_enum, select_combo_data
from space_video_enhancer.ui.tabs.filters import THUMBNAIL_SIZE
from space_video_enhancer.ui.widgets.compare_preview import CompareMode

LOG = get_logger("ui")

THUMBNAIL_REFRESH_DELAY_MS = 140
PREVIEW_SIZE_LIMIT = {PreviewRenderMode.FAST: 960, PreviewRenderMode.BALANCED: 1280}
INTERACTIVE_DENOISE_SIZE_LIMIT = 640
"""While scrubbing or playing with denoise on, NLMeans runs on a frame no larger than this (cost ~ pixels)."""
IDLE_REFRESH_DELAY_MS = 250
"""After the last interactive render, the resting frame is re-rendered at full quality."""
INTERACTIVE_REASONS = ("scrub", "playback")
PREVIEW_REFRESH_DELAY_MS = {PreviewRenderMode.FAST: 20, PreviewRenderMode.BALANCED: 35, PreviewRenderMode.HIGH: 50}
PREVIEW_ZOOM_STEPS = (25, 50, 100, 200, 400)
COMPARE_MODE_OPTIONS: list[tuple[str, CompareMode]] = [
    ("Divisor vertical", CompareMode.SPLIT_VERTICAL),
    ("Divisor horizontal", CompareMode.SPLIT_HORIZONTAL),
    ("Lado a lado", CompareMode.SIDE_BY_SIDE),
    ("Um sobre o outro", CompareMode.STACKED),
]
COMPARE_MODE_DESCRIPTIONS: dict[CompareMode, str] = {
    CompareMode.SPLIT_VERTICAL: "Uma imagem só: antes à esquerda, depois à direita; arraste a linha para os lados.",
    CompareMode.SPLIT_HORIZONTAL: "Uma imagem só: antes em cima, depois embaixo; arraste a linha na vertical.",
    CompareMode.SIDE_BY_SIDE: "Os dois frames inteiros, um ao lado do outro (bom para vídeo em pé).",
    CompareMode.STACKED: "Os dois frames inteiros, um sobre o outro (bom para vídeo deitado).",
}
"""Fixed zoom levels; 100% shows one output pixel per logical screen pixel. "Ajustar" (fit) is the default."""
PREVIEW_RENDER_MODE_OPTIONS: list[tuple[str, PreviewRenderMode]] = [
    ("Alta", PreviewRenderMode.HIGH),
    ("Média", PreviewRenderMode.BALANCED),
    ("Rápida", PreviewRenderMode.FAST),
]


class PreviewSection(WindowState):
    """The preview: zoom and render quality, refresh scheduling, results from the render thread, thumbnails, HUD."""

    def _handle_preview_zoom_changed(self, _index: int) -> None:
        self._apply_preview_zoom()
        percent = self.preview_zoom_combo.currentData()
        log_event(
            LOG,
            "ui.preview_zoom",
            mode="fit" if percent is None else "fixed",
            percent=round(self.preview.effective_zoom() * 100) if percent is None else int(percent),
        )

    def _apply_preview_zoom(self) -> None:
        percent = self.preview_zoom_combo.currentData()
        if percent is None:
            self.preview.set_fit_mode(True)
        else:
            self.preview.set_fit_mode(False)
            self.preview.set_zoom_factor(int(percent) / 100.0)
        self._sync_preview_scroll()
        self._update_zoom_buttons()

    def _sync_preview_scroll(self) -> None:
        """The scroll area sizes the preview to its viewport, unless a split mode at a fixed zoom has to scroll.

        Sized here, after the flag: while the area is still resizable it snaps the widget back to the viewport.
        """
        expands = self.preview.expands_to_viewport()
        self.preview_scroll.setWidgetResizable(expands)
        if not expands:
            self.preview.resize(self.preview.sizeHint())

    def _update_zoom_buttons(self) -> None:
        current = self.preview.effective_zoom() * 100.0
        self.preview_zoom_in_button.setEnabled(current < PREVIEW_ZOOM_STEPS[-1] - 0.5)
        self.preview_zoom_out_button.setEnabled(current > PREVIEW_ZOOM_STEPS[0] + 0.5)
        self.zoom_in_action.setEnabled(self.preview_zoom_in_button.isEnabled())
        self.zoom_out_action.setEnabled(self.preview_zoom_out_button.isEnabled())

    def _zoom_to_fit(self) -> None:
        self.preview_zoom_combo.setCurrentIndex(0)

    def _step_preview_zoom(self, step: int) -> None:
        """Next fixed level above/below what is on screen now (also meaningful from fit mode)."""
        current = self.preview.effective_zoom() * 100.0
        if step > 0:
            candidates = [percent for percent in PREVIEW_ZOOM_STEPS if percent > current + 0.5]
            target = candidates[0] if candidates else None
        else:
            candidates = [percent for percent in PREVIEW_ZOOM_STEPS if percent < current - 0.5]
            target = candidates[-1] if candidates else None
        if target is not None:
            select_combo_data(self.preview_zoom_combo, target)

    def _handle_compare_mode_changed(self, _index: int) -> None:
        mode = combo_enum(self.preview_compare_combo, CompareMode, CompareMode.SPLIT_VERTICAL)
        self.preview.set_compare_mode(mode)
        self._sync_preview_scroll()
        self._update_zoom_buttons()
        log_event(LOG, "ui.compare_mode", mode=mode.value)

    def _toggle_pan(self, checked: bool) -> None:
        self.preview.set_pan_mode(checked)
        log_event(LOG, "ui.pan_tool", enabled=checked)

    def _handle_preview_pan(self, dx: int, dy: int) -> None:
        """Scroll the zoomed preview by the drag delta (nothing happens in fit mode: there is nothing to scroll)."""
        horizontal = self.preview_scroll.horizontalScrollBar()
        vertical = self.preview_scroll.verticalScrollBar()
        horizontal.setValue(horizontal.value() + dx)
        vertical.setValue(vertical.value() + dy)

    def _handle_preview_render_mode_changed(self, _index: int) -> None:
        self.preview.set_render_quality(self._preview_render_mode() is not PreviewRenderMode.FAST)
        if self._monitor is not None:
            self._monitor.set_render_quality(self._preview_render_mode() is not PreviewRenderMode.FAST)
        self._stabilization_epoch += 1
        log_event(LOG, "ui.render_mode", mode=self._preview_render_mode().value)
        self._schedule_preview_refresh(immediate=True, reason="mode")

    def _preview_output_scale_factor(self, scale_factor: float) -> float:
        mode = self._preview_render_mode()
        if mode is PreviewRenderMode.HIGH:
            return scale_factor
        if mode is PreviewRenderMode.BALANCED:
            return min(1.0, scale_factor)
        return 1.0

    def _schedule_preview_refresh(self, *, immediate: bool = False, reason: str = "settings") -> None:
        self._pending_preview_reason = reason
        if immediate:
            self._preview_refresh_timer.stop()
            self._refresh_preview_impl()
            return
        self._preview_refresh_timer.start(PREVIEW_REFRESH_DELAY_MS[self._preview_render_mode()])

    def _schedule_filter_thumbnails_refresh(self) -> None:
        if self.playback_timer.isActive() or not self._filters_tab_visible():
            return
        self._thumbnail_refresh_timer.start(THUMBNAIL_REFRESH_DELAY_MS)

    def _filters_tab_visible(self) -> bool:
        return self.settings_tabs.currentIndex() == self._filters_tab_index

    def _handle_tab_changed(self, index: int) -> None:
        log_event(LOG, "ui.tab", tab=self.settings_tabs.tabText(index))
        if self._perspective_draft is not None and not self._tab_is_active(self.geometry_tab):
            self._end_perspective_marking("cancel_tab")  # the points are marked on the Geometria tab only
        self._sync_preview_pause()  # the region tabs need the image in this window
        self._sync_overlays()
        if self._filters_tab_visible():
            self._schedule_filter_thumbnails_refresh()

    def _tab_is_active(self, tab: TabPage) -> bool:
        return self.settings_tabs.currentWidget() is tab

    def _sync_overlays(self) -> None:
        """Each selection rectangle is editable only on its own tab; the crop stays visible elsewhere."""
        self._sync_crop_preview()
        self._sync_perspective_preview()
        self._sync_denoise_sample_preview()
        self._sync_stabilization_preview()
        self._sync_white_balance_preview()

    def _refresh_preview_impl(self) -> None:
        """Snapshot the widgets into the active clip and ask the render thread for the current frame."""
        if self.sequence.is_empty:
            return

        self._store_active_clip_settings()
        settings = self._current_render_settings()
        self._log_settings(settings)
        self._update_export_summary()  # output size follows scale, rotation and crop
        self._sync_crop_preview()
        self._sync_denoise_sample_preview()
        self._sync_stabilization_preview()
        self.preview.set_display_reference_size(*self._preview_reference_size())

        preview_settings = replace(
            settings,
            filters=replace(
                settings.filters, scale_factor=self._preview_output_scale_factor(settings.filters.scale_factor)
            ),
        )
        if self._perspective_draft is not None:
            # Marking the points: they go on the frame as filmed, so the correction is left out of this render.
            preview_settings = replace(preview_settings, perspective=replace(settings.perspective, enabled=False))
        size_limit = PREVIEW_SIZE_LIMIT.get(self._preview_render_mode())
        interactive = (
            self._pending_preview_reason in INTERACTIVE_REASONS and settings.filters.denoise_mode is not DenoiseMode.OFF
        )
        if interactive:
            size_limit = min(size_limit or INTERACTIVE_DENOISE_SIZE_LIMIT, INTERACTIVE_DENOISE_SIZE_LIMIT)
        location = self.sequence.locate(self.current_frame_index)
        _clip_index, retime_sample = self.sequence.preview_sample(self.current_frame_index)
        clip = self.sequence.clips[location.clip_index]
        source_path = clip.source.path
        if source_path in self._missing_sources:
            self._pending_preview_reason = "settings"
            self.status.set_render_busy(False)
            self.status.show_message(
                f"Mídia não encontrada: {source_path.name} — coloque o arquivo de volta no lugar e reabra o projeto."
            )
            return
        wobble_track = (
            self._wobble_track_for(clip, preview_settings.stabilization)
            if preview_settings.stabilization.needs_wobble_analysis
            else None
        )
        mesh_corrections = (
            self._mesh_corrections_for(clip, preview_settings.stabilization)
            if preview_settings.stabilization.needs_mesh_analysis
            else None
        )
        self._preview_generation += 1
        self.status.set_render_busy(True)
        self._preview_controller.submit_preview(
            PreviewRequest(
                generation=self._preview_generation,
                frame=None,  # decoded on the render thread
                frame_index=location.source_frame,
                settings=preview_settings,
                size_limit=size_limit,
                stabilization_reference=self.stabilization_reference_frame,
                stabilization_epoch=self._stabilization_epoch,
                reason=self._pending_preview_reason,
                source_path=source_path,
                canvas_aspect=self._canvas_aspect(),
                wobble_track=wobble_track,
                mesh_corrections=mesh_corrections,
                retime=retime_sample,
            )
        )
        self._pending_preview_reason = "settings"
        if interactive:
            self._idle_refresh_timer.start(IDLE_REFRESH_DELAY_MS)
        else:
            self._idle_refresh_timer.stop()

    def _refresh_preview_at_full_quality(self) -> None:
        """The frame the user stopped on gets a render without the interactive size limit."""
        if self.sequence.is_empty or self.playback_timer.isActive():
            return
        self._schedule_preview_refresh(immediate=True, reason="idle")

    def _log_settings(self, settings: RenderSettings) -> None:
        """Full snapshot after opening a video; afterwards only the fields that changed."""
        if self._last_logged_settings is None:
            log_event(LOG, "settings.snapshot", settings=diff_settings(None, settings))
        else:
            changes = diff_settings(self._last_logged_settings, settings)
            if changes:
                log_event(LOG, "settings.change", changes=changes, reason=self._pending_preview_reason)
        self._last_logged_settings = settings

    @Slot(object)
    def _on_preview_ready(self, images: PreviewImages) -> None:
        if images.generation <= self._applied_preview_generation:
            return  # stale: a newer request already landed (or the video changed)
        self._applied_preview_generation = images.generation
        self.last_preview_render_ms = images.elapsed_ms
        self._last_preview = images
        self.current_frame = images.source
        self._decoded_frame_index = images.frame_index
        self._decoded_source = images.source_path
        self._counters["renders"] += 1
        if images.decode_sequential is not None:
            self._last_decode_ms = images.decode_ms
            self._last_decode_sequential = images.decode_sequential
            self._decode_stats.add(images.decode_ms)
            self._video_read_counters = (images.video_seeks, images.video_sequential_reads)
        if images.reason == "playback":
            now = time.perf_counter()
            if self._playback_last_frame_at is not None:
                self._playback_intervals.add((now - self._playback_last_frame_at) * 1000.0)
            self._playback_last_frame_at = now
            self._playback_frames_shown += 1
            if "stabilization.result" in images.trace:
                self._playback_stabilization[str(images.trace["stabilization.result"])] += 1
        if self._monitor is not None and self._monitor.isVisible():
            self._monitor.set_images(images.before, images.after)
        if not self._main_preview_paused:  # paused: the frame is on the other window, this one draws nothing
            self.preview.set_images(images.before, images.after)
            self.preview.set_motion_overlay(images.motion if self._vectors_visible else None)
        if images.generation >= self._preview_generation:
            self.status.set_render_busy(False)
        self._update_denoise_info()
        self._update_exposure_info()
        self._update_capture_info()
        self._refresh_manual_controls()  # show the interpolated transform for the frame that just landed
        if not self._main_preview_paused:
            self._update_hud()
        self._schedule_filter_thumbnails_refresh()

    def _discard_pending_previews(self) -> None:
        """Results still in flight for the previous video must not land on the new one."""
        self._applied_preview_generation = self._preview_generation
        self._applied_thumbnail_generation = self._thumbnail_generation
        self._idle_refresh_timer.stop()
        self.status.set_render_busy(False)

    def _refresh_filter_thumbnails(self) -> None:
        self._thumbnail_refresh_timer.stop()
        if self.playback_timer.isActive() or not self._filters_tab_visible():
            return
        if self.current_frame is None:
            self.filters_tab.clear_thumbnails()
            return

        settings = self._current_render_settings()
        settings = replace(settings, filters=replace(settings.filters, scale_factor=1.0))
        ratio = device_pixel_ratio(self)  # device pixels so Retina shows them crisp (never self.devicePixelRatioF())
        icon_size = (round(THUMBNAIL_SIZE.width() * ratio), round(THUMBNAIL_SIZE.height() * ratio))
        reference = None
        if self.stabilization_reference_frame is not None:
            reference = self._thumbnail_source_frame(self.stabilization_reference_frame, icon_size)
        self._thumbnail_generation += 1
        self._preview_controller.submit_thumbnails(
            ThumbnailRequest(
                generation=self._thumbnail_generation,
                frame=self._thumbnail_source_frame(self.current_frame, icon_size),
                settings=settings,
                stabilization_reference=reference,
                preset_ids=tuple(preset.filter_id for preset in IMAGE_FILTER_PRESETS),
                icon_size=icon_size,
                device_pixel_ratio=ratio,
            )
        )

    @Slot(object)
    def _on_thumbnails_ready(self, result: ThumbnailImages) -> None:
        if result.generation <= self._applied_thumbnail_generation:
            return
        self._applied_thumbnail_generation = result.generation
        self.filters_tab.set_thumbnails(result.images, result.device_pixel_ratio)

    @Slot(str)
    def _on_render_failed(self, message: str) -> None:
        self._counters["render_errors"] += 1
        self.status.set_render_busy(False)
        self.status.show_message(f"Falha ao renderizar o preview: {message}")

    @staticmethod
    def _thumbnail_source_frame(frame: np.ndarray, icon_size: tuple[int, int]) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = min(icon_size[0] / max(1, width), icon_size[1] / max(1, height))
        if scale >= 1.0:
            return frame.copy()
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def _toggle_hud(self, checked: bool) -> None:
        self._hud_visible = checked
        self.preview.set_hud_visible(checked)
        log_event(LOG, "ui.hud", visible=checked)
        self._update_hud()

    def _toggle_vectors(self, checked: bool) -> None:
        self._vectors_visible = checked
        log_event(LOG, "ui.motion_vectors", visible=checked)
        motion = self._last_preview.motion if self._last_preview is not None and checked else None
        self.preview.set_motion_overlay(motion)

    def _update_hud(self) -> None:
        """Compact metrics box drawn over the preview (toggle with Ctrl+M)."""
        if not self._hud_visible:
            return
        lines: list[str] = []
        images = self._last_preview
        trace: dict[str, object] = images.trace if images is not None else {}
        stages: dict[str, float] = trace.get("stages", {})  # type: ignore[assignment]
        if images is not None:
            breakdown = " ".join(f"{name}={value:.0f}" for name, value in stages.items() if "." not in name)
            lines.append(f"render {images.elapsed_ms:.0f} ms  {breakdown}")
            queue = f"fila {images.queue_wait_ms:.0f} ms  descartados {self._preview_controller.mailbox.dropped}"
            if "filters.denoise" in stages:
                queue = f"denoise {stages['filters.denoise']:.0f} ms  {queue}"
            lines.append(queue)
            if "stabilization.result" in trace:
                detail = f"  inliers {trace['stabilization.inliers']}" if "stabilization.inliers" in trace else ""
                hypotheses = trace.get("stabilization.hypotheses")
                if isinstance(hypotheses, list) and hypotheses:
                    chosen = trace.get("stabilization.chosen", 0)
                    described = " ".join(
                        f"{'[' if index == chosen else ''}{count}pts/{extent}cel{']' if index == chosen else ''}"
                        for index, (count, extent) in enumerate(hypotheses)
                    )
                    detail += f"  hipóteses {described}"
                lines.append(
                    f"estab {trace.get('stabilization.mode')} -> {trace.get('stabilization.estimator')} "
                    f"-> {trace['stabilization.result']}{detail}"
                )
        if self.video is not None:
            fps = self._playback_fps()
            playback = f"  playback {fps:.1f} fps" if fps else ""
            seeks, sequential = self._video_read_counters
            mean_decode = self._decode_stats.mean if len(self._decode_stats) else 0.0
            lines.append(
                f"decode {self._last_decode_ms:.1f} ms (med {mean_decode:.1f}, worker)  "
                f"seeks {seeks}  seq {sequential}{playback}"
            )
        backend = str(trace.get("filters.backend", self._backend.label))
        shape_in = trace.get("input_shape")
        shape_out = trace.get("output_shape")
        shapes = f"  {shape_in[1]}x{shape_in[0]} -> {shape_out[1]}x{shape_out[0]}" if shape_in and shape_out else ""  # type: ignore[index]
        zoom = f"  zoom {self.preview.effective_zoom() * 100:.0f}%"
        lines.append(f"{backend}  {self.preview_render_mode_combo.currentText()}{shapes}{zoom}")
        self.preview.set_hud_lines(lines)
