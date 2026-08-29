"""The tabs' controls and the preview overlays: presets, denoise, white balance, exposure, stabilization, crop."""

from __future__ import annotations

from collections.abc import Hashable

from PySide6.QtCore import QRectF

from space_video_enhancer.pipeline.deinterlace import likely_interlaced
from space_video_enhancer.pipeline.exposure import gains_in_ev, region_medians
from space_video_enhancer.pipeline.grade import neutral_gains, white_patch_gains
from space_video_enhancer.pipeline.stabilization.keyframes import interpolate_keyframes
from space_video_enhancer.pipeline.stabilization.stabilize import CORRECTION_BUDGET_FRACTION
from space_video_enhancer.pipeline.stabilization.wobble import WobbleTrack, wobble_window_frames
from space_video_enhancer.sequence import Clip
from space_video_enhancer.settings import (
    DEFAULT_DENOISE_SAMPLE_RECT,
    DEFAULT_FILTER_SETTINGS,
    DEFAULT_STABILIZATION_RECT,
    DEFAULT_WHITE_BALANCE_RECT,
    NEUTRAL_GAINS,
    DeinterlaceMode,
    DenoiseMode,
    FilterSettings,
    NormalizedRect,
    StabilizationKeyframe,
    StabilizationMode,
)
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.ui.sections.state import WindowState
from space_video_enhancer.ui.wobble_analysis import WobbleAnalysisRequest

LOG = get_logger("ui")
WOBBLE_ANALYSIS_MARGIN_SECONDS = 2.0
"""The wobble analysis covers this much beyond the clip's range, so trimming the in/out points a little reuses it."""


def qrect_to_norm(rect: QRectF) -> NormalizedRect:
    return NormalizedRect(rect.x(), rect.y(), rect.width(), rect.height())


def norm_to_qrect(rect: NormalizedRect) -> QRectF:
    return QRectF(rect.x, rect.y, rect.width, rect.height)


class AdjustmentsSection(WindowState):
    """The tabs' controls and the preview overlays: presets, denoise, white balance, exposure, stabilization, crop."""

    def _handle_image_filter_changed(self, *_args: object) -> None:
        self._schedule_preview_refresh()

    def _handle_preset_changed(self, index: int) -> None:
        preset = self.color_tab.preset_at(index)
        if preset is None:
            self._schedule_preview_refresh()
            return
        self.active_preset_settings = preset
        log_event(LOG, "ui.preset", preset=self.color_tab.preset_name(index))
        self._apply_look(preset)
        self._schedule_preview_refresh()
        if self.current_frame is not None:
            self.status.show_message("Preset aplicado a tom, cor e nitidez; ruído e escala de saída não mudam.")

    def _apply_look(self, look: FilterSettings) -> None:
        """A preset moves only the sliders of the "Ajustes" tab: denoise and output scale are not part of a look."""
        self.color_tab.apply(look)

    def _reset_slider_to_preset(self, field_name: str) -> None:
        """A slider's reset button: the active preset's value for look sliders, the default for denoise ones."""
        source = self.active_preset_settings if field_name in self.color_tab.sliders else DEFAULT_FILTER_SETTINGS
        self._filter_sliders()[field_name].setValue(int(getattr(source, field_name)))

    def _restore_active_preset_settings(self) -> None:
        self._apply_look(self.active_preset_settings)
        self._sync_preset_selection()
        self._schedule_preview_refresh()
        if self.current_frame is not None:
            self.status.show_message("Tom, cor e nitidez restaurados para o preset ativo.")

    def _apply_settings_to_controls(self, settings: FilterSettings) -> None:
        self.color_tab.apply(settings)
        self.noise_tab.apply(settings)
        self.export_tab.apply(settings)
        self._update_denoise_controls_state()

    def _handle_filter_control_change(self, *_args: object) -> None:
        self._sync_preset_selection()
        self._update_denoise_controls_state()
        self._update_crop_info_label()
        self._schedule_preview_refresh()

    def _handle_geometry_change(self, *_args: object) -> None:
        self._update_crop_info_label()
        self._schedule_preview_refresh()

    def _sync_preset_selection(self) -> None:
        self.color_tab.select_matching_preset(self._current_filter_settings())

    def _update_denoise_controls_state(self) -> None:
        self.noise_tab.update_controls_state(self.frame_count > 0)
        self._sync_denoise_sample_preview()
        self._update_denoise_sample_info_label()
        self._update_denoise_info()

    def _update_denoise_info(self) -> None:
        """Mode explanation plus the strengths the last render actually used (nothing is estimated here)."""
        mode = self.noise_tab.mode()
        if mode is DenoiseMode.OFF:
            self.noise_tab.set_info(
                "Denoise desativado: o preview e a exportação pulam esta etapa para poupar processamento."
            )
            return
        if mode is DenoiseMode.MANUAL:
            self.noise_tab.set_info(
                "Manual: ajuste força de luminância, força de cor e o tamanho das janelas do filtro."
            )
            return

        source = (
            "Auto: calcula a intensidade com base no frame atual inteiro."
            if mode is DenoiseMode.AUTO
            else "Amostra: usa a região azul selecionada no preview como referência de ruído."
        )
        if self.sequence.is_empty:
            self.noise_tab.set_info(source)
            return
        trace = self._last_preview.trace if self._last_preview is not None else {}
        estimate = trace.get("denoise.h")
        if trace.get("denoise.mode") == mode.value and isinstance(estimate, (list, tuple)) and len(estimate) == 2:
            sigma = trace.get("denoise.sigma")
            noise = f"ruído σ {sigma[0]:.1f} → " if isinstance(sigma, (list, tuple)) and len(sigma) == 2 else ""
            self.noise_tab.set_info(
                f"{source} Último render: {noise}luminância {estimate[0]:.1f} | cor {estimate[1]:.1f}."
            )
        else:
            self.noise_tab.set_info(f"{source} Aguardando o render para mostrar a estimativa…")

    def _handle_preview_sample_changed(self, rect: QRectF) -> None:
        self.denoise_sample_rect_norm = QRectF(rect)
        self._sync_denoise_sample_preview()
        self._update_denoise_sample_info_label()
        self._schedule_preview_refresh()

    def _handle_grade_change(self) -> None:
        self._schedule_preview_refresh()

    def _handle_preview_white_balance_changed(self, rect: QRectF) -> None:
        self.white_balance_rect_norm = QRectF(rect)
        self._sync_white_balance_preview()
        self._update_white_balance_info()
        self._exposure_region_moved()

    def _reset_white_balance_rect(self) -> None:
        self.white_balance_rect_norm = norm_to_qrect(DEFAULT_WHITE_BALANCE_RECT)
        self._sync_white_balance_preview()
        self._update_white_balance_info()
        self._exposure_region_moved()

    def _sync_white_balance_preview(self) -> None:
        enabled = self.frame_count > 0 and self._tab_is_active(self.color_page)
        self.preview.set_white_balance_state(enabled, self.white_balance_rect_norm, editing=True)

    def _update_white_balance_info(self) -> None:
        if self.frame_count <= 0:
            self.grade_tab.set_region_info("Abra um vídeo para escolher a região de cinza.")
            return
        pixels = qrect_to_norm(self.white_balance_rect_norm).to_pixels(*self.source_size)
        size = f"{pixels[2]}x{pixels[3]} px" if pixels is not None else ""
        self.grade_tab.set_region_info(
            f"Região: {size} — arraste o retângulo roxo sobre algo cinza/branco e clique em Neutralizar; "
            "sem cinza na cena, mire num brilho e use “pelo realce”."
        )

    def _neutralize_white_balance(self, whole_frame: bool) -> None:
        """One-off: read the decoded frame's region and turn it into channel gains (stored in the settings)."""
        if self.current_frame is None:
            self.status.show_message("Abra um vídeo e aguarde o primeiro render para neutralizar as cores.")
            return
        rect = None if whole_frame else qrect_to_norm(self.white_balance_rect_norm)
        gains = neutral_gains(self.current_frame, rect)
        self._set_white_balance_gains(gains)
        log_event(LOG, "ui.white_balance", source="frame" if whole_frame else "region", gains=list(gains))
        self.status.show_message(f"Balanço de branco: ganhos B {gains[0]:.2f} · G {gains[1]:.2f} · R {gains[2]:.2f}.")

    def _white_patch_white_balance(self) -> None:
        """Balance by the brightest pixels of the region — for scenes with no gray to point at."""
        if self.current_frame is None:
            self.status.show_message("Abra um vídeo e aguarde o primeiro render para neutralizar as cores.")
            return
        rect = qrect_to_norm(self.white_balance_rect_norm)
        gains = white_patch_gains(self.current_frame, rect)
        self._set_white_balance_gains(gains)
        log_event(LOG, "ui.white_balance", source="white_patch", gains=list(gains))
        self.status.show_message(f"Balanço pelo realce: ganhos B {gains[0]:.2f} · G {gains[1]:.2f} · R {gains[2]:.2f}.")

    def _reset_white_balance_gains(self) -> None:
        self._set_white_balance_gains(NEUTRAL_GAINS)

    def _handle_exposure_change(self) -> None:
        self._update_exposure_info()
        self._schedule_preview_refresh()

    def _exposure_region(self) -> NormalizedRect | None:
        return qrect_to_norm(self.white_balance_rect_norm) if self.grade_tab.exposure_uses_region() else None

    def _capture_exposure_reference(self) -> None:
        """Meter the decoded frame (same frame and coordinates as the "Antes" preview) and lock to it."""
        if self.current_frame is None:
            self.status.show_message("Abra um vídeo e aguarde o primeiro render para capturar a referência.")
            return
        reference = region_medians(self.current_frame, self._exposure_region())
        frame_index = self._decoded_frame_index
        self.grade_tab.set_exposure_reference(reference, frame_index=frame_index)
        self._update_active_clip(lambda clip: clip.with_exposure_reference(frame_index), "referência de exposição")
        log_event(
            LOG,
            "ui.exposure",
            action="capture",
            frame=self._decoded_frame_index,
            reference=[round(value, 1) for value in reference],
            region=self.grade_tab.exposure_uses_region(),
        )
        self.status.show_message(f"Referência de exposição capturada no frame {self._decoded_frame_index + 1}.")
        self._handle_exposure_change()

    def _clear_exposure_reference(self, reason: str | None = None) -> None:
        if self.grade_tab.exposure_reference() is None:
            return
        self.grade_tab.set_exposure_reference(None)
        self._update_active_clip(lambda clip: clip.with_exposure_reference(None), "limpar referência de exposição")
        log_event(LOG, "ui.exposure", action="clear", reason=reason or "user")
        if reason:
            self.status.show_message(reason)
        self._handle_exposure_change()

    def _exposure_region_moved(self) -> None:
        """The metering area changed: re-meter if we are still on the reference frame, else forget the reference."""
        if self.grade_tab.exposure_reference() is None:
            return
        clip = self.active_clip
        on_reference_frame = (
            clip is not None
            and self.current_frame is not None
            and self._decoded_source == clip.source.path
            and self._decoded_frame_index == clip.exposure_reference_frame
        )
        if on_reference_frame:
            self._capture_exposure_reference()
        else:
            self._clear_exposure_reference(
                "Região de referência alterada: volte ao frame de referência e capture a exposição de novo."
            )

    def _update_exposure_info(self) -> None:
        if self.frame_count <= 0:
            self.grade_tab.set_exposure_status("")
            return
        settings = self.grade_tab.exposure_settings(self._exposure_region())
        if not settings.enabled:
            text = "Desligado."
        elif settings.reference is None:
            text = "Ligado, mas sem referência: capture um frame bem exposto."
        else:
            trace = self._last_preview.trace if self._last_preview is not None else {}
            gains = trace.get("exposure.gains")
            if isinstance(gains, (list, tuple)) and len(gains) == 3:
                stops = gains_in_ev((float(gains[0]), float(gains[1]), float(gains[2])))
                text = (
                    f"Último render: ganhos B {gains[0]:.2f} · G {gains[1]:.2f} · R {gains[2]:.2f} ({stops:+.2f} EV)."
                )
            else:
                text = "Aguardando o render para mostrar os ganhos…"
        self.grade_tab.set_exposure_status(text)

    def _set_white_balance_gains(self, gains: tuple[float, float, float]) -> None:
        self.white_balance_gains = gains
        self.grade_tab.set_gains(gains)
        self._schedule_preview_refresh()

    def _reset_denoise_sample_rect(self) -> None:
        self.denoise_sample_rect_norm = norm_to_qrect(DEFAULT_DENOISE_SAMPLE_RECT)
        self._sync_denoise_sample_preview()
        self._update_denoise_sample_info_label()
        self._schedule_preview_refresh()

    def _sync_denoise_sample_preview(self) -> None:
        enabled = (
            self.frame_count > 0 and self.noise_tab.mode() is DenoiseMode.SAMPLE and self._tab_is_active(self.noise_tab)
        )
        self.preview.set_sample_state(enabled, self.denoise_sample_rect_norm, editing=True)

    def _update_denoise_sample_info_label(self) -> None:
        if self.frame_count <= 0:
            self.noise_tab.set_sample_info("Amostra: abra um vídeo para escolher a região de análise.")
            return
        rect = qrect_to_norm(self.denoise_sample_rect_norm)
        percent = f"{rect.width * 100:.0f}% x {rect.height * 100:.0f}%"
        pixels = rect.to_pixels(*self.source_size)
        if pixels is not None:
            self.noise_tab.set_sample_info(f"Amostra: {pixels[2]}x{pixels[3]} | {percent}")
        else:
            self.noise_tab.set_sample_info(f"Amostra: {percent}")

    def _handle_stabilization_mode_changed(self) -> None:
        self._stabilization_epoch += 1
        mode = self.stabilization_tab.mode()
        if mode.uses_focal_point:
            self._capture_stabilization_reference_from_current_frame(label="ajuste de estabilização")
        if mode is StabilizationMode.MANUAL and self._onion_reference is None and self._last_preview is not None:
            self._onion_reference = self._last_preview.before  # start the ghost on the current frame
            self._ensure_anchor_keyframe(self._decoded_frame_index)  # pin this frame so others move relative to it
        self._update_stabilization_controls_state()
        self._refresh_manual_controls()
        self._schedule_preview_refresh()

    # --- manual keyframes -----------------------------------------------------------------
    def _ensure_anchor_keyframe(self, frame: int) -> bool:
        """Guarantee a keyframe exists at ``frame`` (an identity one if absent), so it is pinned. Returns if added."""
        if any(item.frame == frame for item in self._manual_keyframes):
            return False
        anchor = StabilizationKeyframe(frame)  # identity: this frame stays exactly as it is
        self._manual_keyframes = tuple(sorted((*self._manual_keyframes, anchor), key=lambda item: item.frame))
        return True

    def _handle_manual_adjust(self) -> None:
        """A slider moved: create or move the keyframe at the current frame to the sliders' values."""
        if self.active_clip is None or self.current_frame is None:
            return
        frame = self._decoded_frame_index
        tx, ty, rotation, scale = self.stabilization_tab.manual_adjust()
        keyframe = StabilizationKeyframe(frame, tx, ty, rotation, scale)
        others = tuple(existing for existing in self._manual_keyframes if existing.frame != frame)
        self._manual_keyframes = tuple(sorted((*others, keyframe), key=lambda item: item.frame))
        self._store_active_clip_settings()  # folds the keyframes into the clip and commits one undo step
        self._update_manual_info()
        self._schedule_preview_refresh()

    def _capture_onion_reference(self) -> None:
        if self._last_preview is None:
            self.status.show_message("Aguarde o primeiro render para definir a referência.")
            return
        self._onion_reference = self._last_preview.before  # the raw current frame, canvas-sized, as the ghost
        if self._ensure_anchor_keyframe(self._decoded_frame_index):
            self._store_active_clip_settings()
        self._refresh_manual_controls()
        self._sync_stabilization_preview()
        log_event(LOG, "ui.manual_reference", frame=self._decoded_frame_index)
        self.status.show_message(f"Referência (fantasma): frame {self._decoded_frame_index + 1} fixado como âncora.")

    def _remove_manual_keyframe(self) -> None:
        if self.active_clip is None:
            return
        frame = self._decoded_frame_index
        remaining = tuple(item for item in self._manual_keyframes if item.frame != frame)
        if remaining == self._manual_keyframes:
            return
        self._manual_keyframes = remaining
        self._store_active_clip_settings()
        self._refresh_manual_controls()
        self._schedule_preview_refresh()

    def _go_to_keyframe(self, direction: int) -> None:
        if not self._manual_keyframes or self.active_clip is None:
            return
        frame = self._decoded_frame_index
        frames = sorted(item.frame for item in self._manual_keyframes)
        if direction < 0:
            below = [value for value in frames if value < frame]
            target_source = max(below) if below else frames[0]
        else:
            above = [value for value in frames if value > frame]
            target_source = min(above) if above else frames[-1]
        self._stop_playback()
        self.transport.set_value(self.sequence.timeline_frame(self.active_clip_index, target_source))

    def _refresh_manual_controls(self) -> None:
        """Show the interpolated transform for the current frame on the sliders (no keyframe is created)."""
        if self.stabilization_tab.mode() is not StabilizationMode.MANUAL:
            return
        if self.playback_timer.isActive():
            return  # do not chase the sliders frame by frame during playback
        params = interpolate_keyframes(self._manual_keyframes, self._decoded_frame_index)
        self.stabilization_tab.set_manual_adjust(*(params or (0.0, 0.0, 0.0, 1.0)))
        self._update_manual_info()

    def _update_manual_info(self) -> None:
        frame = self._decoded_frame_index
        count = len(self._manual_keyframes)
        on_keyframe = any(item.frame == frame for item in self._manual_keyframes)
        self.stabilization_tab.manual_remove_button.setEnabled(on_keyframe and self.frame_count > 0)
        if count == 0:
            text = "Ande até um frame que saiu do lugar e mova posição/rotação/zoom até casar com o fantasma."
        else:
            where = "keyframe deste frame (mova os sliders para ajustar)" if on_keyframe else "interpolado"
            text = (
                f"{count} keyframe(s) · frame {frame + 1}: {where}. "
                "Ande alguns frames, alinhe ao fantasma e repita; o meio é preenchido sozinho."
            )
        self.stabilization_tab.set_manual_info(text)

    def _handle_stabilization_control_change(self) -> None:
        self._stabilization_epoch += 1
        self._update_stabilization_controls_state()
        before = self.sequence.timeline_count
        self._store_active_clip_settings()  # fold the capture change in now, so a speed change resizes the clip
        if self.sequence.timeline_count != before:
            self._sequence_changed()  # speed changed the clip's length: refresh the timeline, transport and markers
            self.transport.set_value_silently(self.current_frame_index)
        self._schedule_preview_refresh()

    def _handle_preview_stabilization_changed(self, rect: QRectF) -> None:
        self.stabilization_rect_norm = QRectF(rect)
        self._capture_stabilization_reference_from_current_frame(
            label="mover ponto focal", key=("focal_rect", self.active_clip_index)
        )
        self._stabilization_epoch += 1
        self._sync_stabilization_preview()
        self._update_stabilization_info_label()
        self._schedule_preview_refresh()

    def _reset_stabilization_rect(self) -> None:
        self.stabilization_rect_norm = norm_to_qrect(DEFAULT_STABILIZATION_RECT)
        self._capture_stabilization_reference_from_current_frame(label="redefinir ponto focal")
        self._stabilization_epoch += 1
        self._sync_stabilization_preview()
        self._update_stabilization_info_label()
        self._schedule_preview_refresh()

    def _set_stabilization_reference_here(self) -> None:
        """The frame on screen becomes the camera-lock reference; the focal region stays where it is."""
        if self.current_frame is None or self._decoded_source is None:
            self.status.show_message("Abra um vídeo e aguarde o primeiro render para definir a referência.")
            return
        self._capture_stabilization_reference_from_current_frame(label="referência da estabilização")
        self._stabilization_epoch += 1
        self._update_stabilization_info_label()
        self._schedule_preview_refresh()
        log_event(LOG, "ui.stabilization_reference", frame=self._decoded_frame_index, source=self._decoded_source.name)
        self.status.show_message(
            f"Referência da estabilização: frame {self._decoded_frame_index + 1} (a região focal não mudou)."
        )

    def _capture_stabilization_reference_from_current_frame(self, *, label: str, key: Hashable | None = None) -> None:
        """The decoded frame (and its focal region as the widgets show it) becomes the lock reference: one undo step."""
        if self.current_frame is None or self._decoded_source is None:
            return
        self.stabilization_reference_frame = self.current_frame.copy()
        self.stabilization_reference_frame_index = self._decoded_frame_index
        frame_index = self._decoded_frame_index
        self._reference_frames[(self._decoded_source, frame_index)] = self.stabilization_reference_frame
        self._update_active_clip(lambda clip: clip.with_reference(frame_index), label, key)
        self._prune_reference_frames()

    def _update_stabilization_controls_state(self) -> None:
        self.stabilization_tab.update_controls_state(self.frame_count > 0)
        self._sync_stabilization_preview()
        self._update_stabilization_info_label()
        self._update_capture_info()

    def _update_capture_info(self) -> None:
        """What the last render measured: how much comb the frame carries, and the motion the unskew used."""
        if self.frame_count <= 0:
            self.stabilization_tab.set_capture_info("")
            return
        trace = self._last_preview.trace if self._last_preview is not None else {}
        facts: list[str] = []
        comb = trace.get("deinterlace.comb")
        if isinstance(comb, (int, float)):
            if likely_interlaced(comb):
                facts.append(f"Entrelaçamento provável: {comb:.1%} dos pixels com pente")
                if self.stabilization_tab.deinterlace_mode() is DeinterlaceMode.OFF:
                    facts.append("ative o desentrelaçamento")
            else:
                facts.append(f"Entrelaçamento improvável ({comb:.2%} com pente)")
        stabilization = self._current_stabilization()
        if stabilization.corrects_rolling_shutter_skew:
            shift = trace.get("rolling_shutter.shift_px")
            if isinstance(shift, (list, tuple)) and len(shift) == 2:
                facts.append(f"rolling shutter: movimento {shift[0]:+.1f}, {shift[1]:+.1f} px/frame")
            elif trace.get("rolling_shutter.result") == "no_motion":
                facts.append("rolling shutter: sem movimento mensurável neste frame (primeiro frame ou sem textura)")
            else:
                facts.append("rolling shutter: aguardando o render")
        if stabilization.rolling_shutter_auto:
            facts.append(self._wobble_status_text(trace))
        if stabilization.corrects_rolling_shutter:
            facts.append(self._readout_axis_text(trace))
        self.stabilization_tab.set_capture_info(" · ".join(facts) + ("." if facts else ""))

    @staticmethod
    def _readout_axis_text(trace: dict[str, object]) -> str:
        names = {"rows": "linhas (topo → base)", "columns": "colunas (esquerda → direita)"}
        name = names.get(str(trace.get("rolling_shutter.axis")))
        if name is None:
            return "leitura do sensor: aguardando o render"
        sources = {"forced": "escolhida", "detected": "detectada", "default": "padrão até a análise"}
        source = sources.get(str(trace.get("rolling_shutter.axis_source")))
        return f"leitura do sensor: {name}" + (f" ({source})" if source else "")

    def _wobble_status_text(self, trace: dict[str, object]) -> str:
        pending = self._pending_analysis_for_active_clip()
        if pending is not None:
            done, total = self._wobble_progress.get(pending.generation, (0, pending.count))
            return f"analisando a oscilação: {done * 100 // max(1, total)}% ({done}/{total} frames)"
        wobble = trace.get("rolling_shutter.wobble_px")
        if isinstance(wobble, (list, tuple)) and len(wobble) == 2:
            return f"oscilação compensada {wobble[0]:+.1f}, {wobble[1]:+.1f} px (topo→base)"
        return "oscilação: aguardando o render"

    def _wobble_track_for(self, clip: Clip) -> WobbleTrack | None:
        """The clip's wobble analysis when it is done; otherwise starts it (once) and returns ``None`` for now.

        The analysis runs on its own thread over the clip's range plus a margin; the preview renders
        without the wobble until ``_on_wobble_finished`` stores the track and asks for a refresh.
        """
        first, last = clip.range.start, clip.range.end
        for track in self._wobble_tracks:
            if track.source == clip.source.path and track.covers(first, last):
                return track
        if any(request.covers(clip.source.path, first, last) for request in self._wobble_pending):
            return None  # already queued or running: switching away and back does not restart it
        if clip.source.path in self._missing_sources:
            return None
        window = wobble_window_frames(clip.fps)
        margin = max(window, round(WOBBLE_ANALYSIS_MARGIN_SECONDS * clip.fps))
        start = max(0, first - margin)
        end = min(clip.source.last_frame, last + margin)
        request = self._wobble_analysis.analyze(clip.source.path, start, end, window)
        self._wobble_pending.append(request)
        self._wobble_progress[request.generation] = (0, end - start + 1)
        log_event(LOG, "ui.wobble_analysis_start", clip=clip.name, first=start, last=end, window=window)
        self.status.show_message(
            f"Analisando a oscilação do rolling shutter de {clip.name} ({end - start + 1} frames) em segundo plano…"
        )
        self._update_capture_info()
        return None

    def _pending_analysis_for_active_clip(self) -> WobbleAnalysisRequest | None:
        clip = self.active_clip
        if clip is None:
            return None
        first, last = clip.range.start, clip.range.end
        return next(
            (request for request in self._wobble_pending if request.covers(clip.source.path, first, last)), None
        )

    def _answers_pending_analysis(self, request: object) -> bool:
        return isinstance(request, WobbleAnalysisRequest) and any(
            request.generation == pending.generation for pending in self._wobble_pending
        )

    def _on_wobble_progress(self, request: object, done: int, total: int) -> None:
        if isinstance(request, WobbleAnalysisRequest) and self._answers_pending_analysis(request):
            self._wobble_progress[request.generation] = (done, total)
            self._update_capture_info()

    def _on_wobble_finished(self, request: object, track: object, elapsed_ms: float) -> None:
        if not isinstance(request, WobbleAnalysisRequest) or not self._answers_pending_analysis(request):
            return  # answers a request from a document that is gone
        self._wobble_pending = [pending for pending in self._wobble_pending if pending.generation != request.generation]
        self._wobble_progress.pop(request.generation, None)
        if not isinstance(track, WobbleTrack):
            log_event(LOG, "ui.wobble_analysis_failed", name=request.source.name)
            self.status.show_message(f"Não foi possível analisar a oscilação de {request.source.name}.")
            self._update_capture_info()
            return
        sources = set(self.sequence.source_paths)
        self._wobble_tracks = [
            kept
            for kept in self._wobble_tracks
            if kept.source in sources
            and not (kept.source == track.source and track.covers(kept.first_frame, kept.last_frame))
        ]
        self._wobble_tracks.append(track)
        log_event(
            LOG,
            "ui.wobble_analysis_finish",
            name=track.source.name,
            frames=track.count,
            measured=round(track.measured_fraction, 3),
            elapsed_ms=round(elapsed_ms, 1),
        )
        self.status.show_message(
            f"Oscilação do rolling shutter analisada: {track.count} frames em {elapsed_ms / 1000.0:.1f} s."
        )
        self._update_capture_info()
        self._schedule_preview_refresh(immediate=True)

    def _sync_stabilization_preview(self) -> None:
        active = self.frame_count > 0 and self._tab_is_active(self.stabilization_tab)
        mode = self.stabilization_tab.mode()
        self.preview.set_stabilization_state(
            active and mode.uses_focal_point, self.stabilization_rect_norm, editing=True
        )
        show_ghost = active and mode is StabilizationMode.MANUAL and self._onion_reference is not None
        ghost = self._onion_reference if show_ghost else None
        self.preview.set_onion_skin(ghost, self.stabilization_tab.onion_opacity())

    def _update_stabilization_info_label(self) -> None:
        """Facts only (the mode description lives in the tab's hint and tooltips)."""
        if self.frame_count <= 0:
            self.stabilization_tab.set_info("Abra um vídeo para ajustar a estabilização.")
            return
        settings = self._current_stabilization()
        if not settings.enabled:
            self.stabilization_tab.set_info("")
            return

        facts = [
            (
                f"Crop automático ligado (margem fixa de {CORRECTION_BUDGET_FRACTION:.0%})"
                if settings.auto_crop
                else "Crop automático desligado"
            ),
            f"zoom out {'ligado' if settings.zoom_out else 'desligado'}",
            f"suavização {int(settings.temporal_smoothing * 100)}%",
        ]
        if settings.mode.uses_focal_point:
            region = settings.focus_rect.to_pixels(*self.source_size)
            if region is None:
                facts.insert(0, "Selecione a região focal no preview")
            else:
                facts.insert(0, f"Referência: frame {self.stabilization_reference_frame_index + 1}")
                facts.insert(1, f"região {region[2]}x{region[3]} px (arraste o retângulo verde no preview)")
        else:
            facts.append("exportação sequencial")
        self.stabilization_tab.set_info(" · ".join(facts) + ".")

    def _displayed_size(self) -> tuple[int, int]:
        """Frame size the crop rectangle refers to: source, scaled, then rotated."""
        scale = self.export_tab.scale_factor()
        width = round(self.source_size[0] * scale)
        height = round(self.source_size[1] * scale)
        return self._current_geometry().output_size(width, height)

    def _displayed_aspect_ratio(self) -> float:
        width, height = self._displayed_size()
        return width / height if width > 0 and height > 0 else 1.0

    def _normalized_crop_aspect_ratio(self) -> float | None:
        aspect_ratio = self.geometry_tab.crop_aspect_ratio()
        if aspect_ratio is None:
            return None
        return aspect_ratio / self._displayed_aspect_ratio()

    def _handle_crop_mode_changed(self) -> None:
        if self.geometry_tab.crop_enabled():
            current_rect = QRectF(self.crop_rect_norm)
            aspect_ratio = self._normalized_crop_aspect_ratio()
            if self.frame_count <= 0:
                self.crop_rect_norm = self._default_crop_rect(aspect_ratio)
            elif aspect_ratio is None:
                self.crop_rect_norm = current_rect if current_rect.width() > 0 else self._default_crop_rect(None)
            else:
                self.crop_rect_norm = self._crop_rect_with_aspect(current_rect, aspect_ratio)
        self._sync_crop_preview()
        self._update_crop_info_label()
        self._schedule_preview_refresh()

    def _handle_preview_crop_changed(self, rect: QRectF) -> None:
        aspect_ratio = self._normalized_crop_aspect_ratio()
        self.crop_rect_norm = (
            QRectF(rect) if aspect_ratio is None else self._crop_rect_with_aspect(QRectF(rect), aspect_ratio)
        )
        self._sync_crop_preview()
        self._update_crop_info_label()

    def _reset_crop_rect(self) -> None:
        self.crop_rect_norm = self._default_crop_rect(self._normalized_crop_aspect_ratio())
        self._sync_crop_preview()
        self._update_crop_info_label()
        self._schedule_preview_refresh()

    def _sync_crop_preview(self) -> None:
        self.preview.set_crop_state(
            self.geometry_tab.crop_enabled(),
            self.crop_rect_norm,
            self._normalized_crop_aspect_ratio(),
            editing=self._tab_is_active(self.geometry_tab),
        )

    def _update_crop_info_label(self) -> None:
        if not self.geometry_tab.crop_enabled():
            self.geometry_tab.set_crop_info("Crop: desativado")
            return

        if self.geometry_tab.crop_is_free():
            prefix = "Crop: Livre"
            ratio_text = self._rect_ratio_text(self.crop_rect_norm)
        else:
            prefix = f"Crop: {self.geometry_tab.crop_mode_text()}"
            ratio_text = self.geometry_tab.crop_mode_text()

        width, height = self._displayed_size()
        pixels = self._current_crop().pixel_rect(width, height)
        if pixels is not None:
            self.geometry_tab.set_crop_info(f"{prefix} | {pixels[2]}x{pixels[3]} | {ratio_text}")
            return
        self.geometry_tab.set_crop_info(
            f"{prefix} | {self.crop_rect_norm.width() * 100:.0f}% x {self.crop_rect_norm.height() * 100:.0f}%"
        )

    def _rect_ratio_text(self, rect: QRectF) -> str:
        if rect.height() <= 0 or rect.width() <= 0:
            return "Livre"
        ratio = (rect.width() / rect.height()) * self._displayed_aspect_ratio()
        candidates = [(16 / 9, "16:9"), (4 / 3, "4:3"), (1.0, "1:1"), (9 / 16, "9:16")]
        best_ratio, best_label = min(candidates, key=lambda candidate: abs(candidate[0] - ratio))
        if abs(best_ratio - ratio) < 0.04:
            return best_label
        return f"{ratio:.2f}:1"

    @staticmethod
    def _default_crop_rect(aspect_ratio: float | None) -> QRectF:
        max_size = 0.82
        if aspect_ratio is None:
            return QRectF(0.09, 0.09, max_size, max_size)
        width = max_size
        height = width / aspect_ratio
        if height > max_size:
            height = max_size
            width = height * aspect_ratio
        return QRectF((1.0 - width) / 2.0, (1.0 - height) / 2.0, width, height)

    def _crop_rect_with_aspect(self, rect: QRectF, aspect_ratio: float) -> QRectF:
        if rect.width() <= 0 or rect.height() <= 0:
            return self._default_crop_rect(aspect_ratio)

        center = rect.center()
        width = rect.width()
        height = width / aspect_ratio
        if height > rect.height():
            height = rect.height()
            width = height * aspect_ratio

        max_width = min(center.x(), 1.0 - center.x()) * 2.0
        max_height = min(center.y(), 1.0 - center.y()) * 2.0
        if height > max_height:
            height = max_height
            width = height * aspect_ratio
        if width > max_width:
            width = max_width
            height = width / aspect_ratio

        if width < 0.08 or height < 0.08:
            return self._default_crop_rect(aspect_ratio)
        return QRectF(center.x() - (width / 2.0), center.y() - (height / 2.0), width, height)
