"""The detached preview window and pausing this window's preview while it shows the frame."""

from __future__ import annotations

from PySide6.QtGui import QGuiApplication

from space_video_enhancer.settings import PreviewRenderMode
from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.ui.monitor_window import MonitorWindow
from space_video_enhancer.ui.sections.state import WindowState

LOG = get_logger("ui")


class MonitorSection(WindowState):
    """The detached preview window and pausing this window's preview while it shows the frame."""

    def _toggle_monitor(self, checked: bool) -> None:
        if not checked:
            if self._monitor is not None and self._monitor.isVisible():
                self._monitor.hide()
                log_event(LOG, "ui.monitor", state="closed")
            self._sync_preview_pause()
            return
        monitor = self._monitor_window()
        monitor.set_frame_count(self.frame_count, self.fps)
        monitor.set_position(self.current_frame_index)
        monitor.set_playing(self.playback_timer.isActive())
        monitor.set_controls_enabled(self.transport.isEnabled())
        monitor.set_render_quality(self._preview_render_mode() is not PreviewRenderMode.FAST)
        monitor.set_document_name(None if self.sequence.is_empty else self._document_name())
        images = self._last_preview
        monitor.set_images(images.before if images is not None else None, images.after if images is not None else None)
        monitor.show_near(self)
        self._sync_preview_pause()
        log_event(LOG, "ui.monitor", state="open", screens=len(QGuiApplication.screens()))

    def _handle_pause_preview_toggled(self, checked: bool) -> None:
        log_event(LOG, "ui.monitor", state="pause_preview" if checked else "resume_preview")
        self._sync_preview_pause()

    def _tab_edits_a_preview_region(self) -> bool:
        """Tabs whose rectangle is dragged over the image: they need the preview in this window."""
        return self.settings_tabs.currentWidget() in (
            self.geometry_tab,
            self.noise_tab,
            self.stabilization_tab,
            self.grade_tab,
        )

    def _should_pause_main_preview(self) -> bool:
        showing_elsewhere = self._monitor is not None and self._monitor.isVisible()
        return showing_elsewhere and self.pause_preview_action.isChecked() and not self._tab_edits_a_preview_region()

    @property
    def main_preview_paused(self) -> bool:
        """True while this window leaves the frame to the monitor and paints a panel instead."""
        return self._main_preview_paused

    def _sync_preview_pause(self) -> None:
        """Show either the preview or the paused panel; the render itself never stops (the monitor needs it)."""
        self.pause_preview_action.setEnabled(self._monitor is not None and self._monitor.isVisible())
        paused = self._should_pause_main_preview()
        if paused == self._main_preview_paused:
            return
        self._main_preview_paused = paused
        had_focus = self.preview.hasFocus() or self.preview_paused_panel.hasFocus()
        if paused:
            self.preview_stack.setCurrentWidget(self.preview_paused_panel)
            self.preview.set_images(None, None)  # nothing to hold: the frames live on the monitor now
            self.preview.set_motion_overlay(None)
        else:
            self.preview_stack.setCurrentWidget(self.preview_scroll)
            images = self._last_preview
            if images is not None:
                self.preview.set_images(images.before, images.after)
                self.preview.set_motion_overlay(images.motion if self._vectors_visible else None)
            self._update_hud()
        if had_focus:
            self._focus_preview_surface()

    def _focus_preview_surface(self) -> None:
        """Focus whichever of the two is on screen, so the bare transport keys keep working."""
        (self.preview_paused_panel if self._main_preview_paused else self.preview).setFocus()

    def _monitor_window(self) -> MonitorWindow:
        """The detached window, created the first time it is asked for and kept (hidden) afterwards."""
        if self._monitor is None:
            monitor = MonitorWindow()
            monitor.firstRequested.connect(self._go_to_first_frame)
            monitor.previousRequested.connect(self._go_to_previous_frame)
            monitor.playToggled.connect(self._toggle_playback)
            monitor.nextRequested.connect(self._go_to_next_frame)
            monitor.lastRequested.connect(self._go_to_last_frame)
            monitor.frameRequested.connect(self._seek_from_monitor)
            monitor.closed.connect(self._on_monitor_closed)
            # The same key bindings as the preview: Space, Home/End, arrows, I/O/X work with the monitor focused too.
            for action in self._transport_actions():
                monitor.addAction(action)
            monitor.addAction(self.monitor_action)  # Ctrl+Shift+D closes it from there as well
            self._monitor = monitor
        return self._monitor

    def _seek_from_monitor(self, frame: int) -> None:
        if self.frame_count > 0:
            self._stop_playback()
            self.transport.set_value(max(0, min(frame, self.transport.maximum())))

    def _on_monitor_closed(self) -> None:
        if self.monitor_action.isChecked():
            self.monitor_action.blockSignals(True)
            self.monitor_action.setChecked(False)
            self.monitor_action.blockSignals(False)
            log_event(LOG, "ui.monitor", state="closed")
        self._sync_preview_pause()
