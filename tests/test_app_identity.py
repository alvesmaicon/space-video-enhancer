"""The process presents itself as the app, not as Python: names, desktop id and a drawn icon."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from space_video_enhancer import __version__
from space_video_enhancer.app import APPLICATION_NAME, DESKTOP_FILE_NAME, configure_application
from space_video_enhancer.ui.app_icon import BACK_COLOR, FRONT_COLOR, ICON_SIZES, TILE_COLOR, app_icon, write_png


def test_configure_application_sets_the_identity(app: QApplication) -> None:
    configure_application(app)
    assert app.applicationName() == APPLICATION_NAME == "Space Video Enhancer"
    assert app.applicationDisplayName() == APPLICATION_NAME
    assert app.desktopFileName() == DESKTOP_FILE_NAME == "space-video-enhancer"
    assert app.applicationVersion() == __version__
    assert not app.windowIcon().isNull()


def test_the_icon_is_drawn_at_every_size_with_its_two_frames(app: QApplication) -> None:
    icon = app_icon()
    sizes = {(size.width(), size.height()) for size in icon.availableSizes()}
    assert sizes == {(s, s) for s in ICON_SIZES}
    image = icon.pixmap(256, 256).toImage()
    assert image.pixelColor(2, 2).alpha() == 0, "rounded corners are transparent"
    assert image.pixelColor(128, 8) == TILE_COLOR
    assert image.pixelColor(64, 77) == BACK_COLOR, "the muted frame behind"
    assert image.pixelColor(200, 190) == FRONT_COLOR, "the teal frame in front"
    assert image.pixelColor(150, 153).lightness() > 200, "the play triangle on the front frame"


def test_write_png_exports_the_icon(app: QApplication, tmp_path: Path) -> None:
    path = write_png(tmp_path / "icons" / "space-video-enhancer.png", 64)
    image = QImage(str(path))
    assert path.is_file() and image.width() == 64 and image.height() == 64
    assert image.pixelColor(37, 38).lightness() > 200, "the play triangle"
