"""Tab "Estabilizar": mode, strength, smoothing, focal region and border handling.

The focal region itself is edited directly on the preview whenever this tab is active
and the mode uses a focal point; there is no separate "edit" toggle.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton

from space_video_enhancer.settings import NormalizedRect, StabilizationMode, StabilizationSettings, clamp_unit
from space_video_enhancer.ui.format import format_percent
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
MODE_DESCRIPTIONS: dict[StabilizationMode, str] = {
    StabilizationMode.OFF: "Nenhuma compensação de movimento.",
    StabilizationMode.AUTO: (
        "Segue uma versão suavizada da trajetória da câmera (só translação): remove o tremor, preserva panorâmicas."
    ),
    StabilizationMode.SIMILARITY: (
        "Como o automático, compensando também rotação leve e pequenas variações de escala."
    ),
    StabilizationMode.PERSPECTIVE: (
        "Como o automático, com uma homografia global para parallax leve e pequenas mudanças de perspectiva."
    ),
    StabilizationMode.CAMERA_LOCK: (
        "Usa features da região verde para manter esse enquadramento preso ao frame de referência; "
        "segura a última posição se perder o alvo."
    ),
    StabilizationMode.HYBRID: (
        "Prende a região verde com camera lock; se perder o alvo, segue o movimento global até reencontrá-lo."
    ),
}
DEFAULT_INTENSITY = 100
DEFAULT_SMOOTHING = 35


class StabilizationTab(TabPage):
    modeChanged = Signal()
    controlChanged = Signal()
    """Intensity, smoothing, auto crop or zoom out changed (inter-frame memory must restart)."""
    resetRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.mode_combo = enum_combo(MODE_OPTIONS)
        for index, (_label, mode) in enumerate(MODE_OPTIONS):
            self.mode_combo.setItemData(index, MODE_DESCRIPTIONS[mode], Qt.ItemDataRole.ToolTipRole)
        self.hint_label = QLabel()
        self.hint_label.setObjectName("hintLabel")
        self.hint_label.setWordWrap(True)
        self.intensity_slider = LabeledSlider(
            "Intensidade da estabilização", 0, 100, DEFAULT_INTENSITY, display=format_percent
        )
        self.smoothing_slider = LabeledSlider("Suavização temporal", 0, 100, DEFAULT_SMOOTHING, display=format_percent)
        self.smoothing_slider.setToolTip(
            "Quão devagar a câmera virtual segue a real: "
            "baixo remove só o tremor rápido, alto quase trava o enquadramento."
        )
        self.intensity_slider.setToolTip("Fração da correção aplicada: 100% segue a trajetória suavizada por completo.")
        self.reset_button = QPushButton("Redefinir ponto focal")
        self.reset_button.setObjectName("secondaryButton")
        self.auto_crop_checkbox = QCheckBox("Crop automático para esconder bordas")
        self.auto_crop_checkbox.setChecked(True)
        self.zoom_out_checkbox = QCheckBox("Zoom out automático para preservar bordas")
        self.zoom_out_checkbox.setChecked(False)
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)

        self.mode_combo.currentIndexChanged.connect(self._handle_mode_changed)
        self.intensity_slider.valueChanged.connect(lambda _value: self.controlChanged.emit())
        self.smoothing_slider.valueChanged.connect(lambda _value: self.controlChanged.emit())
        self.auto_crop_checkbox.toggled.connect(lambda _checked: self.controlChanged.emit())
        self.zoom_out_checkbox.toggled.connect(self._handle_zoom_out_toggled)
        self.reset_button.clicked.connect(self.resetRequested.emit)

        self.add_sections(
            group_box(
                "Estabilização",
                form_rows(("Modo", self.mode_combo)),
                self.hint_label,
                self.intensity_slider,
                self.smoothing_slider,
                self.auto_crop_checkbox,
                self.zoom_out_checkbox,
                self.reset_button,
                self.info_label,
            )
        )
        self._update_hint()

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
        """Enable only what the current mode uses."""
        mode = self.mode()
        controls_enabled = has_video and self.mode_combo.isEnabled()
        stabilization_enabled = controls_enabled and mode is not StabilizationMode.OFF
        self.intensity_slider.setEnabled(stabilization_enabled)
        self.smoothing_slider.setEnabled(stabilization_enabled)
        zoom_out_enabled = stabilization_enabled and self.zoom_out_checkbox.isChecked()
        self.auto_crop_checkbox.setEnabled(stabilization_enabled and not zoom_out_enabled)
        self.zoom_out_checkbox.setEnabled(stabilization_enabled)
        self.reset_button.setEnabled(stabilization_enabled and mode.uses_focal_point)

    def set_info(self, text: str) -> None:
        self.info_label.setText(text)

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.mode_combo,
            self.intensity_slider,
            self.smoothing_slider,
            self.reset_button,
            self.auto_crop_checkbox,
            self.zoom_out_checkbox,
        ):
            widget.setEnabled(enabled)

    def _handle_mode_changed(self, _index: int) -> None:
        self._update_hint()
        self.modeChanged.emit()

    def _update_hint(self) -> None:
        description = MODE_DESCRIPTIONS[self.mode()]
        self.hint_label.setText(description)
        self.mode_combo.setToolTip(description)

    def _handle_zoom_out_toggled(self, checked: bool) -> None:
        if checked and self.auto_crop_checkbox.isChecked():
            set_checked_silently(self.auto_crop_checkbox, False)
        self.controlChanged.emit()
