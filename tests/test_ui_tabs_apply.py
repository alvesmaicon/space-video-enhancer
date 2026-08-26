"""Every settings tab can show another clip's settings (``apply``) silently and read them back unchanged."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from space_video_enhancer.settings import (
    ColorGradeSettings,
    CropSettings,
    ExposureLockSettings,
    GeometrySettings,
    HueBandAdjust,
    ImageFilterSelection,
    NormalizedRect,
    RotationMode,
    StabilizationMode,
    StabilizationSettings,
)
from space_video_enhancer.ui.tabs.filters import FiltersTab
from space_video_enhancer.ui.tabs.geometry import GeometryTab
from space_video_enhancer.ui.tabs.grade import ColorGradeTab
from space_video_enhancer.ui.tabs.stabilization import StabilizationTab


def test_filters_tab_apply_round_trips_without_signals(app: QApplication) -> None:
    tab = FiltersTab()
    emitted: list[object] = []
    tab.filterSelected.connect(emitted.append)
    tab.intensityChanged.connect(emitted.append)
    selection = ImageFilterSelection("shadow_lift", 0.4)
    tab.apply(selection)
    assert tab.selection() == selection and tab.buttons["shadow_lift"].isChecked()
    tab.apply(ImageFilterSelection("does-not-exist", 1.0))
    assert tab.selection().preset_id == "original"
    assert emitted == []


def test_geometry_tab_apply_round_trips_without_signals(app: QApplication) -> None:
    tab = GeometryTab()
    emitted: list[str] = []
    tab.geometryChanged.connect(lambda: emitted.append("geometry"))
    tab.cropModeChanged.connect(lambda: emitted.append("crop"))
    geometry = GeometrySettings(rotation=RotationMode.CW_90, flip_horizontal=True)
    tab.apply(geometry, CropSettings(enabled=True, aspect_ratio=4 / 3))
    assert tab.geometry_settings() == geometry
    assert tab.crop_enabled() and tab.crop_aspect_ratio() == pytest.approx(4 / 3) and not tab.crop_is_free()
    tab.apply(GeometrySettings(), CropSettings(enabled=True, aspect_ratio=None))
    assert tab.crop_is_free() and tab.geometry_settings() == GeometrySettings()
    tab.apply(GeometrySettings(), CropSettings(enabled=False))
    assert not tab.crop_enabled()
    assert emitted == []


def test_stabilization_tab_apply_round_trips_without_signals(app: QApplication) -> None:
    tab = StabilizationTab()
    emitted: list[str] = []
    tab.modeChanged.connect(lambda: emitted.append("mode"))
    tab.controlChanged.connect(lambda: emitted.append("control"))
    rect = NormalizedRect(0.2, 0.2, 0.3, 0.3)
    settings = StabilizationSettings(
        mode=StabilizationMode.SIMILARITY, intensity=0.6, temporal_smoothing=0.2, focus_rect=rect, auto_crop=False
    )
    tab.apply(settings)
    assert tab.settings(rect) == settings
    assert "rotação leve" in tab.hint_label.text()
    tab.apply(StabilizationSettings(mode=StabilizationMode.OFF, zoom_out=True, auto_crop=True))
    assert tab.settings(rect).zoom_out and not tab.settings(rect).auto_crop, "zoom out excludes the automatic crop"
    assert emitted == []


def test_hsl_tab_apply_round_trips_without_signals(app: QApplication) -> None:
    tab = ColorGradeTab()
    emitted: list[str] = []
    tab.valueChanged.connect(lambda: emitted.append("value"))
    tab.exposureChanged.connect(lambda: emitted.append("exposure"))
    tab.exposureScopeChanged.connect(lambda: emitted.append("scope"))
    bands = list(tab.settings((1.0, 1.0, 1.0)).bands)
    bands[3] = HueBandAdjust(hue_shift=-12, saturation=30, luminance=-5)
    grade = ColorGradeSettings(tint=62, white_balance_gains=(1.1, 1.0, 0.9), bands=tuple(bands))
    exposure = ExposureLockSettings(
        enabled=True,
        region=NormalizedRect(0.1, 0.1, 0.2, 0.2),
        reference=(10.0, 20.0, 30.0),
        strength=40,
        max_ev=1.5,
        color=False,
    )
    tab.apply(grade, exposure, exposure_frame=4)
    assert tab.settings(grade.white_balance_gains) == grade
    assert tab.exposure_settings(exposure.region) == exposure
    assert "frame 5" in tab.exposure_label.text() and "1.10" in tab.gains_label.text()
    tab.select_band(3)
    assert tab.hue_slider.value() == -12 and tab.saturation_slider.value() == 30
    whole_frame = ExposureLockSettings(enabled=False, region=None, reference=None)
    tab.apply(ColorGradeSettings(), whole_frame)
    assert not tab.exposure_uses_region() and tab.exposure_settings(NormalizedRect(0, 0, 1, 1)) == whole_frame
    assert tab.settings((1.0, 1.0, 1.0)) == ColorGradeSettings()
    assert emitted == []
