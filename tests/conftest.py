from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from space_video_enhancer.ui.main_window import MainWindow
from tests.qt_helpers import pump, qt_app


def make_textured_frame(width: int = 480, height: int = 270, seed: int = 42) -> np.ndarray:
    """Deterministic frame with gradient, noise and a few high-contrast shapes (good for feature tracking)."""
    rng = np.random.default_rng(seed)
    gradient = np.linspace(30, 220, width, dtype=np.float32)[None, :, None].repeat(height, 0).repeat(3, 2)
    gradient[..., 0] *= 0.8
    gradient[..., 2] *= 1.1
    frame = np.clip(gradient + rng.normal(0, 10, (height, width, 3)), 0, 255).astype(np.uint8)
    cv2.rectangle(frame, (60, 40), (180, 140), (30, 200, 60), -1)
    cv2.circle(frame, (340, 170), 55, (200, 40, 40), -1)
    cv2.putText(frame, "GOLD", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
    return frame


def make_quadrant_frame(width: int = 640, height: int = 360) -> np.ndarray:
    """Four solid quadrants (BGR): TL red, TR green, BL blue, BR yellow."""
    frame = np.zeros((height, width, 3), np.uint8)
    frame[: height // 2, : width // 2] = (0, 0, 255)
    frame[: height // 2, width // 2 :] = (0, 255, 0)
    frame[height // 2 :, : width // 2] = (255, 0, 0)
    frame[height // 2 :, width // 2 :] = (0, 255, 255)
    return frame


@pytest.fixture
def textured_frame() -> np.ndarray:
    return make_textured_frame()


@pytest.fixture
def quadrant_frame() -> np.ndarray:
    return make_quadrant_frame()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Every test in a module that touches Qt (PySide6 or the ``ui`` package) is marked ``ui``.

    The Qt tests are most of the suite's wall time; ``pytest -m "not ui"`` is the fast loop for the pipeline
    and the model, the full run stays the gate. Marking by module (not by fixture) also lets the autouse
    fixture below guarantee a ``QApplication`` for tests that build widgets without asking for ``app``.
    """
    qt_modules: dict[Path, bool] = {}
    for item in items:
        path = Path(str(item.fspath))
        if path not in qt_modules:
            text = path.read_text(encoding="utf-8")
            qt_modules[path] = "PySide6" in text or "space_video_enhancer.ui" in text
        if qt_modules[path]:
            item.add_marker(pytest.mark.ui)


@pytest.fixture(autouse=True)
def _qt_application_for_ui_tests(request: pytest.FixtureRequest) -> None:
    """A widget created before the first ``app`` fixture ran would abort the process; never depend on test order."""
    if request.node.get_closest_marker("ui") is not None:
        request.getfixturevalue("app")


# --- offscreen Qt fixtures (skipped when PySide6 is not installed) ---------------------------

UI_VIDEO_SIZE = (320, 180)
UI_VIDEO_FRAMES = 12
UI_VIDEO_FPS = 24.0


@pytest.fixture(scope="session")
def app():  # type: ignore[no-untyped-def]
    return qt_app()


@pytest.fixture(scope="session")
def ui_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("ui") / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), UI_VIDEO_FPS, UI_VIDEO_SIZE)
    base = cv2.resize(make_textured_frame(), UI_VIDEO_SIZE, interpolation=cv2.INTER_AREA)
    for index in range(UI_VIDEO_FRAMES):
        writer.write(np.roll(base, index * 3, axis=1))
    writer.release()
    return path


UI_VIDEO_B_SIZE = (240, 180)
UI_VIDEO_B_FRAMES = 8


def _write_video(path: Path, size: tuple[int, int], frames: int, fps: float, *, tint: tuple[int, int, int]) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    base = cv2.resize(make_textured_frame(seed=7), size, interpolation=cv2.INTER_AREA)
    base = cv2.addWeighted(base, 0.6, np.full_like(base, tint), 0.4, 0)
    for index in range(frames):
        writer.write(np.roll(base, -index * 4, axis=0))
    writer.release()
    return path


@pytest.fixture(scope="session")
def ui_video_b(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A second clip for sequences: another size (4:3), same fps, a warm tint to tell it apart."""
    return _write_video(
        tmp_path_factory.mktemp("ui") / "second.mp4",
        UI_VIDEO_B_SIZE,
        UI_VIDEO_B_FRAMES,
        UI_VIDEO_FPS,
        tint=(40, 90, 200),
    )


@pytest.fixture(scope="session")
def ui_video_other_fps(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_video(tmp_path_factory.mktemp("ui") / "thirty.mp4", UI_VIDEO_SIZE, 6, 30.0, tint=(200, 90, 40))


@pytest.fixture
def window(app):  # type: ignore[no-untyped-def]
    window = MainWindow()
    window.confirm_discard = False  # tests open and close freely; the prompt has its own test
    window.show()
    pump(app, 0.1)
    yield window
    window.close()
    pump(app, 0.05)
