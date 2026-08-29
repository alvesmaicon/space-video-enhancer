#!/usr/bin/env python3
"""Regenerate the README screenshots from the current UI, offscreen and reproducibly.

Builds a synthetic demo clip (a warm test pattern with grain, so the before/after and the denoise
have something to show), opens it in the real ``MainWindow``, sets a representative state — a colour
grade, two clips with a removed segment — and grabs the window. Then pops the separate monitor into
its "Antes | Depois" split and grabs that too. Run with the project's virtualenv:

    QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/screenshots.py

The two PNGs land in ``docs/screenshots/`` at the sizes the README expects (1360x820 and 960x600).
It never plays audio and touches nothing outside a temp dir and ``docs/screenshots/``.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SVE_PREVIEW_AUDIO", "0")

import cv2
import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication

from space_video_enhancer.ui.main_window import MainWindow

REPO_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = REPO_ROOT / "docs" / "screenshots"
WIDTH, HEIGHT, FRAMES, FPS = 640, 360, 78, 24.0
RNG = np.random.default_rng(7)


def _demo_frame() -> np.ndarray:
    """A warm gradient with a green block, a purple disc and a title — a stand-in for real footage."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    top, bottom = np.array([150, 190, 235], np.float64), np.array([40, 70, 120], np.float64)  # BGR sky -> ground
    for y in range(HEIGHT):
        frame[y, :] = top + (bottom - top) * (y / HEIGHT)
    cv2.rectangle(frame, (40, 30), (250, 150), (40, 160, 90), -1)  # a green block, top-left
    cv2.circle(frame, (470, 210), 78, (150, 70, 150), -1)  # a purple disc, right
    cv2.putText(frame, "DEMO", (150, 120), cv2.FONT_HERSHEY_DUPLEX, 3.0, (210, 225, 235), 6, cv2.LINE_AA)
    return frame


def _write_demo_clip(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    base = _demo_frame()
    for index in range(FRAMES):
        drifted = np.roll(base, index * 2, axis=1)  # a slow pan, so the timeline thumbnails differ
        grain = RNG.normal(0.0, 7.0, drifted.shape)  # light film grain for denoise to bite on
        writer.write(np.clip(drifted.astype(np.float64) + grain, 0, 255).astype(np.uint8))
    writer.release()


def _wait(app: QApplication, window, seconds: float = 1.2) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def main() -> None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])

    clip_dir = Path(tempfile.mkdtemp(prefix="sve-shots-"))
    clip = clip_dir / "clip_a.mp4"
    _write_demo_clip(clip)

    window = MainWindow()
    window.confirm_discard = False
    window.resize(1360, 820)
    window.show()
    _wait(app, window, 0.3)

    window._open_video(clip)
    _wait(app, window)
    window._add_video(clip)  # a second clip on the timeline
    _wait(app, window)
    window.transport.set_value(21)  # park on a frame with the pattern well inside
    window.color_tab.contrast_slider.slider.setValue(62)  # +12
    window.color_tab.saturation_slider.slider.setValue(58)  # +8
    window.color_tab.sharpen_slider.slider.setValue(25)
    window._add_cut(30, 42)  # a removed segment, to show the export trimming
    window.settings_tabs.setCurrentWidget(window.color_page)
    window.white_balance_rect_norm = QRectF(0.05, 0.62, 0.15, 0.18)  # tuck the grey-reference box off the title
    window._sync_white_balance_preview()
    _wait(app, window)

    main_path = SCREENSHOTS_DIR / "janela-principal.png"
    window.grab().save(str(main_path))
    print(f"wrote {main_path.relative_to(REPO_ROOT)}")

    window.monitor_action.trigger()  # pop out the separate monitor
    monitor = window._monitor
    assert monitor is not None
    monitor.resize(960, 600)
    for index in range(monitor.view_combo.count()):
        if monitor.view_combo.itemData(index) == "split":  # "Antes | Depois"
            monitor.view_combo.setCurrentIndex(index)
            break
    monitor.show()
    _wait(app, window)
    monitor_path = SCREENSHOTS_DIR / "monitor.png"
    monitor.grab().save(str(monitor_path))
    print(f"wrote {monitor_path.relative_to(REPO_ROOT)}")

    window.monitor_action.trigger()  # close the monitor before teardown
    window.close()
    _wait(app, window, 0.2)


if __name__ == "__main__":
    main()
