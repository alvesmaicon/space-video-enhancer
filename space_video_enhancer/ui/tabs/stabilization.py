"""Tab "Estabilizar": mode, strength, smoothing, focal region controls and border handling."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton

from space_video_enhancer.settings import NormalizedRect, StabilizationMode, StabilizationSettings, clamp_unit
from space_video_enhancer.ui.tabs.base import (
    TabPage,
    combo_enum,
    enum_combo,
    form_rows,
    group_box,
    set_checked_silently,
)
from space_video_enhancer.ui.widgets.labeled_slider import LabeledSlider

MODE_OPTIONS: list[tuple[str, StabilizationMode]] = [
    ("Desativado", StabilizationMode.OFF),
    ("Automático (translação)", StabilizationMode.AUTO),
    ("Automático por similaridade", StabilizationMode.SIMILARITY),
    ("Automático por perspectiva", StabilizationMode.PERSPECTIVE),
    ("Camera lock (região)", StabilizationMode.CAMERA_LOCK),
    ("Híbrido: camera lock + fallback", StabilizationMode.HYBRID),
]
DEFAULT_INTENSITY = 65
DEFAULT_SMOOTHING = 35


class StabilizationTab(TabPage):
    modeChanged = Signal()
    controlChanged = Signal()
    """Intensity, smoothing, auto crop or zoom out changed (inter-frame memory must restart)."""
    editToggled = Signal(bool)
    resetRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.mode_combo = enum_combo(MODE_OPTIONS)
        self.intensity_slider = LabeledSlider("Intensidade da estabilização", 0, 100, DEFAULT_INTENSITY)
        self.smoothing_slider = LabeledSlider("Suavização temporal", 0, 100, DEFAULT_SMOOTHING)
        self.edit_button = QPushButton("Editar ponto/região no preview")
        self.edit_button.setCheckable(True)
        self.reset_button = QPushButton("Resetar ponto focal")
        self.auto_crop_checkbox = QCheckBox("Crop automático para esconder bordas")
        self.auto_crop_checkbox.setChecked(True)
        self.zoom_out_checkbox = QCheckBox("Zoom out automático para preservar bordas")
        self.zoom_out_checkbox.setChecked(False)
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)

        self.mode_combo.currentIndexChanged.connect(lambda _index: self.modeChanged.emit())
        self.intensity_slider.valueChanged.connect(lambda _value: self.controlChanged.emit())
        self.smoothing_slider.valueChanged.connect(lambda _value: self.controlChanged.emit())
        self.auto_crop_checkbox.toggled.connect(lambda _checked: self.controlChanged.emit())
        self.zoom_out_checkbox.toggled.connect(self._handle_zoom_out_toggled)
        self.edit_button.toggled.connect(self.editToggled.emit)
        self.reset_button.clicked.connect(self.resetRequested.emit)

        self.add_sections(
            group_box(
                "Estabilização",
                form_rows(("Modo", self.mode_combo)),
                self.intensity_slider,
                self.smoothing_slider,
                self.edit_button,
                self.reset_button,
                self.auto_crop_checkbox,
                self.zoom_out_checkbox,
                self.info_label,
            )
        )

    def mode(self) -> StabilizationMode:
        return combo_enum(self.mode_combo, StabilizationMode, StabilizationMode.OFF)

    def settings(self, focus_rect: NormalizedRect) -> StabilizationSettings:
        return StabilizationSettings(
            mode=self.mode(),
            intensity=clamp_unit(self.intensity_slider.value() / 100.0),
            temporal_smoothing=clamp_unit(self.smoothing_slider.value() / 100.0),
            focus_rect=focus_rect,
            auto_crop=self.auto_crop_checkbox.isChecked(),
            zoom_out=self.zoom_out_checkbox.isChecked(),
        )

    def update_controls_state(self, has_video: bool) -> None:
        """Enable only what the current mode uses; leaves the edit button unchecked outside focal modes."""
        mode = self.mode()
        controls_enabled = has_video and self.mode_combo.isEnabled()
        stabilization_enabled = controls_enabled and mode is not StabilizationMode.OFF
        self.intensity_slider.setEnabled(stabilization_enabled)
        self.smoothing_slider.setEnabled(stabilization_enabled)
        zoom_out_enabled = stabilization_enabled and self.zoom_out_checkbox.isChecked()
        self.auto_crop_checkbox.setEnabled(stabilization_enabled and not zoom_out_enabled)
        self.zoom_out_checkbox.setEnabled(stabilization_enabled)
        focal_mode = stabilization_enabled and mode.uses_focal_point
        self.edit_button.setEnabled(focal_mode)
        self.reset_button.setEnabled(focal_mode)
        if not focal_mode:
            set_checked_silently(self.edit_button, False)

    def set_info(self, text: str) -> None:
        self.info_label.setText(text)

    def set_edit_checked_silently(self, checked: bool) -> None:
        set_checked_silently(self.edit_button, checked)

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.mode_combo,
            self.intensity_slider,
            self.smoothing_slider,
            self.edit_button,
            self.reset_button,
            self.auto_crop_checkbox,
            self.zoom_out_checkbox,
        ):
            widget.setEnabled(enabled)

    def _handle_zoom_out_toggled(self, checked: bool) -> None:
        if checked and self.auto_crop_checkbox.isChecked():
            set_checked_silently(self.auto_crop_checkbox, False)
        self.controlChanged.emit()
