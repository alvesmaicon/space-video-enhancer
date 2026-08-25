"""Helpers for offscreen Qt tests: one application, event pumping and waiting for the preview worker."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

if TYPE_CHECKING:
    from space_video_enhancer.ui.main_window import MainWindow


def qt_app() -> QApplication:
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


def pump(app: QApplication, seconds: float = 0.05) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)


def wait_until(app: QApplication, condition: Callable[[], bool], timeout: float = 20.0, what: str = "") -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timeout waiting for {what or 'condition'}")
        pump(app, 0.01)


def wait_for_preview(app: QApplication, window: MainWindow) -> None:
    """Wait for the debounced refresh to be requested and for its render to land."""
    wait_until(app, lambda: not window._preview_refresh_timer.isActive(), what="preview refresh debounce")
    wait_until(
        app,
        lambda: window._preview_generation > 0 and window._applied_preview_generation == window._preview_generation,
        what="preview render",
    )
    pump(app, 0.05)
