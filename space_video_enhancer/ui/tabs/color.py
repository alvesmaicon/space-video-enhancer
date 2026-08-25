"""Tab "Cor": tonal sliders, sharpening and the slider presets."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox

from space_video_enhancer.settings import DEFAULT_FILTER_SETTINGS, FILTER_PRESETS, FilterSettings
from space_video_enhancer.ui.tabs.base import TabPage, form_rows, group_box
from space_video_enhancer.ui.widgets.labeled_slider import LabeledSlider


class ColorTab(TabPage):
    presetChanged = Signal(int)
    """Index chosen in the preset combo (0 = "Personalizado")."""
    valueChanged = Signal()
    resetRequested = Signal(str)
    """A slider's "back to preset" button; carries the ``FilterSettings`` field name."""

    def __init__(self) -> None:
        super().__init__()
        defaults = DEFAULT_FILTER_SETTINGS
        self.preset_combo = QComboBox()
        for label, preset in FILTER_PRESETS:
            self.preset_combo.addItem(label, preset)
        self.contrast_slider = LabeledSlider("Contraste", 0, 100, defaults.contrast)
        self.brightness_slider = LabeledSlider("Brilho", 0, 100, defaults.brightness)
        self.gamma_slider = LabeledSlider("Gama", 0, 100, defaults.gamma)
        self.saturation_slider = LabeledSlider("Saturação", 0, 100, defaults.saturation)
        self.temperature_slider = LabeledSlider("Temperatura", 0, 100, defaults.temperature)
        self.sharpen_slider = LabeledSlider("Nitidez", 0, 100, defaults.sharpen)
        self.sliders: dict[str, LabeledSlider] = {
            "contrast": self.contrast_slider,
            "brightness": self.brightness_slider,
            "gamma": self.gamma_slider,
            "saturation": self.saturation_slider,
            "temperature": self.temperature_slider,
            "sharpen": self.sharpen_slider,
        }
        for field_name, slider in self.sliders.items():
            slider.valueChanged.connect(lambda _value: self.valueChanged.emit())
            slider.resetRequested.connect(lambda name=field_name: self.resetRequested.emit(name))
        self.preset_combo.currentIndexChanged.connect(self.presetChanged.emit)

        self.add_sections(
            group_box(
                "Cor e look",
                form_rows(("Preset", self.preset_combo)),
                self.contrast_slider,
                self.brightness_slider,
                self.gamma_slider,
                self.saturation_slider,
                self.temperature_slider,
            ),
            group_box("Detalhe", self.sharpen_slider),
        )

    def apply(self, settings: FilterSettings) -> None:
        """Move the sliders to ``settings`` without emitting ``valueChanged``."""
        for field_name, slider in self.sliders.items():
            slider.blockSignals(True)
            slider.slider.setValue(int(getattr(settings, field_name)))
            slider.blockSignals(False)

    def select_matching_preset(self, settings: FilterSettings) -> None:
        """Show the preset equal to ``settings``, or "Personalizado" when none matches (no signal)."""
        matching_index = 0
        for index in range(1, self.preset_combo.count()):
            if self.preset_combo.itemData(index) == settings:
                matching_index = index
                break
        if self.preset_combo.currentIndex() != matching_index:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(matching_index)
            self.preset_combo.blockSignals(False)

    def preset_at(self, index: int) -> FilterSettings | None:
        data = self.preset_combo.itemData(index)
        return data if isinstance(data, FilterSettings) else None

    def preset_name(self, index: int) -> str:
        return self.preset_combo.itemText(index)

    def set_controls_enabled(self, enabled: bool) -> None:
        self.preset_combo.setEnabled(enabled)
        for slider in self.sliders.values():
            slider.setEnabled(enabled)
