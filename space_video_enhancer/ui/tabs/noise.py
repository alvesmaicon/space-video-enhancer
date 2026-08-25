"""Tab "Ruído": denoise mode, NLMeans windows, strengths and the noise-sample region controls."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton

from space_video_enhancer.settings import DEFAULT_FILTER_SETTINGS, DenoiseMode, FilterSettings
from space_video_enhancer.ui.tabs.base import (
    TabPage,
    combo_enum,
    enum_combo,
    form_rows,
    group_box,
    select_combo_data,
    set_checked_silently,
)
from space_video_enhancer.ui.widgets.labeled_slider import LabeledSlider

MODE_OPTIONS: list[tuple[str, DenoiseMode]] = [
    ("Desativado", DenoiseMode.OFF),
    ("Manual", DenoiseMode.MANUAL),
    ("Auto", DenoiseMode.AUTO),
    ("Amostra do preview", DenoiseMode.SAMPLE),
]
TEMPLATE_OPTIONS = (3, 5, 7, 9)
SEARCH_OPTIONS = (11, 15, 21, 27, 35)


class NoiseTab(TabPage):
    controlChanged = Signal()
    resetRequested = Signal(str)
    """A slider's "back to preset" button; carries the ``FilterSettings`` field name."""
    sampleEditToggled = Signal(bool)
    sampleResetRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        defaults = DEFAULT_FILTER_SETTINGS
        self.mode_combo = enum_combo(MODE_OPTIONS)
        self.template_combo = QComboBox()
        for value in TEMPLATE_OPTIONS:
            self.template_combo.addItem(f"{value} px", value)
        self.search_combo = QComboBox()
        for value in SEARCH_OPTIONS:
            self.search_combo.addItem(f"{value} px", value)
        self.strength_slider = LabeledSlider("Força de luminância", 0, 100, defaults.denoise_strength)
        self.color_slider = LabeledSlider("Força de cor", 0, 100, defaults.denoise_color_strength)
        self.sliders: dict[str, LabeledSlider] = {
            "denoise_strength": self.strength_slider,
            "denoise_color_strength": self.color_slider,
        }
        self.sample_edit_button = QPushButton("Editar amostra no preview")
        self.sample_edit_button.setCheckable(True)
        self.sample_reset_button = QPushButton("Resetar amostra")
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.sample_info_label = QLabel()
        self.sample_info_label.setWordWrap(True)

        for combo in (self.mode_combo, self.template_combo, self.search_combo):
            combo.currentIndexChanged.connect(lambda _index: self.controlChanged.emit())
        for field_name, slider in self.sliders.items():
            slider.valueChanged.connect(lambda _value: self.controlChanged.emit())
            slider.resetRequested.connect(lambda name=field_name: self.resetRequested.emit(name))
        self.sample_edit_button.toggled.connect(self.sampleEditToggled.emit)
        self.sample_reset_button.clicked.connect(self.sampleResetRequested.emit)

        self.add_sections(
            group_box(
                "Limpeza e redução de ruído",
                form_rows(
                    ("Modo", self.mode_combo),
                    ("Janela de template", self.template_combo),
                    ("Janela de busca", self.search_combo),
                ),
                self.strength_slider,
                self.color_slider,
                self.sample_edit_button,
                self.sample_reset_button,
                self.info_label,
                self.sample_info_label,
            )
        )

    def mode(self) -> DenoiseMode:
        return combo_enum(self.mode_combo, DenoiseMode, DenoiseMode.OFF)

    def template_window(self) -> int:
        return int(self.template_combo.currentData())

    def search_window(self) -> int:
        return int(self.search_combo.currentData())

    def apply(self, settings: FilterSettings) -> None:
        """Move the denoise controls to ``settings`` without emitting ``controlChanged``."""
        widgets = [self.mode_combo, self.template_combo, self.search_combo, *self.sliders.values()]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            select_combo_data(self.mode_combo, settings.denoise_mode.value)
            select_combo_data(self.template_combo, settings.denoise_template_window)
            select_combo_data(self.search_combo, settings.denoise_search_window)
            for field_name, slider in self.sliders.items():
                slider.slider.setValue(int(getattr(settings, field_name)))
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def update_controls_state(self, has_video: bool) -> None:
        """Strengths only in manual mode, sample buttons only in sample mode."""
        mode = self.mode()
        controls_enabled = has_video and self.mode_combo.isEnabled()
        manual = controls_enabled and mode is DenoiseMode.MANUAL
        sample = controls_enabled and mode is DenoiseMode.SAMPLE
        self.strength_slider.setEnabled(manual)
        self.color_slider.setEnabled(manual)
        self.sample_edit_button.setEnabled(sample)
        self.sample_reset_button.setEnabled(sample)
        if not sample:
            set_checked_silently(self.sample_edit_button, False)

    def set_info(self, text: str) -> None:
        self.info_label.setText(text)

    def set_sample_info(self, text: str) -> None:
        self.sample_info_label.setText(text)

    def set_sample_edit_checked_silently(self, checked: bool) -> None:
        set_checked_silently(self.sample_edit_button, checked)

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.mode_combo,
            self.template_combo,
            self.search_combo,
            self.strength_slider,
            self.color_slider,
            self.sample_edit_button,
            self.sample_reset_button,
        ):
            widget.setEnabled(enabled)
