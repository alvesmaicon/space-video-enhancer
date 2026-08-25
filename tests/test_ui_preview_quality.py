"""Interactive renders with denoise on use a smaller frame; the resting frame is re-rendered in full."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from space_video_enhancer.settings import DenoiseMode
from space_video_enhancer.ui.main_window import INTERACTIVE_DENOISE_SIZE_LIMIT, MainWindow
from space_video_enhancer.ui.tabs.base import select_combo_data
from tests.qt_helpers import wait_for_preview, wait_until


def test_scrubbing_with_denoise_renders_small_then_idle_renders_in_full(
    app: QApplication, window: MainWindow, ui_video: Path
) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    select_combo_data(window.noise_tab.mode_combo, DenoiseMode.AUTO.value)
    wait_for_preview(app, window)
    assert window._last_preview is not None and window._last_preview.trace["preview.size_limit"] is None

    window.transport.set_value(3)  # scrub
    wait_for_preview(app, window)
    scrubbed = window._last_preview
    assert scrubbed is not None and scrubbed.reason == "scrub"
    assert scrubbed.trace["preview.size_limit"] == INTERACTIVE_DENOISE_SIZE_LIMIT
    assert window._idle_refresh_timer.isActive()

    wait_until(
        app,
        lambda: window._last_preview is not None and window._last_preview.reason == "idle",
        what="idle full-quality render",
    )
    assert window._last_preview is not None
    assert window._last_preview.trace["preview.size_limit"] is None
    assert window._last_preview.frame_index == 3


def test_scrubbing_without_denoise_keeps_the_full_size(app: QApplication, window: MainWindow, ui_video: Path) -> None:
    window._open_video(ui_video)
    wait_for_preview(app, window)
    select_combo_data(window.noise_tab.mode_combo, DenoiseMode.OFF.value)
    wait_for_preview(app, window)
    window.transport.set_value(4)
    wait_for_preview(app, window)
    assert window._last_preview is not None and window._last_preview.trace["preview.size_limit"] is None
    assert not window._idle_refresh_timer.isActive()
