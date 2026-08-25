"""Tab "HSL": white balance (tint, neutral gray region) and per-hue-band hue/saturation/luminance.

The gray reference region is edited directly on the preview while this tab is active; the
window turns it into channel gains with :func:`pipeline.grade.neutral_gains` when the user
clicks "Neutralizar". The eight color swatches select which band the three sliders edit.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QToolButton

from space_video_enhancer.settings import (
    BAND_LUMINANCE_LIMIT,
    BAND_SATURATION_LIMIT,
    DEFAULT_HUE_BANDS,
    HUE_BAND_NAMES,
    HUE_SHIFT_LIMIT_DEG,
    NEUTRAL_GAINS,
    ColorGradeSettings,
    HueBandAdjust,
)
from space_video_enhancer.ui.format import format_signed
from space_video_enhancer.ui.tabs.base import TabPage, group_box
from space_video_enhancer.ui.widgets.labeled_slider import LabeledSlider

DEFAULT_TINT = 50
BAND_LABELS = {
    "red": "Vermelho",
    "orange": "Laranja",
    "yellow": "Amarelo",
    "green": "Verde",
    "cyan": "Ciano",
    "blue": "Azul",
    "purple": "Roxo",
    "magenta": "Magenta",
}
BAND_SWATCHES = {
    "red": "#ef4444",
    "orange": "#f97316",
    "yellow": "#eab308",
    "green": "#22c55e",
    "cyan": "#06b6d4",
    "blue": "#3b82f6",
    "purple": "#8b5cf6",
    "magenta": "#ec4899",
}
SWATCH_SIZE = QSize(34, 26)


def format_degrees(value: int) -> str:
    return f"{value:+d}°" if value else "0°"


def format_signed_percent(value: int) -> str:
    return f"{value:+d}%" if value else "0%"


class ColorGradeTab(TabPage):
    valueChanged = Signal()
    neutralizeRequested = Signal(bool)
    """``True`` = whole frame (gray world), ``False`` = the gray reference region on the preview."""
    gainsResetRequested = Signal()
    regionResetRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._bands: list[HueBandAdjust] = list(DEFAULT_HUE_BANDS)
        self._current_band = 0

        self.tint_slider = LabeledSlider("Tint (verde ↔ magenta)", 0, 100, DEFAULT_TINT, display=format_signed)
        self.tint_slider.setToolTip(
            "Complementa a temperatura: para a esquerda compensa magenta, para a direita compensa verde."
        )
        self.neutralize_button = QPushButton("Neutralizar pela região")
        self.neutralize_button.setToolTip("Calcula os ganhos B/G/R para a região roxa do preview virar cinza neutro.")
        self.auto_button = QPushButton("Automático (frame inteiro)")
        self.auto_button.setToolTip("Gray world: assume que a média do frame inteiro deveria ser neutra.")
        self.gains_reset_button = QPushButton("Redefinir ganhos")
        self.gains_reset_button.setObjectName("secondaryButton")
        self.region_reset_button = QPushButton("Redefinir região")
        self.region_reset_button.setObjectName("secondaryButton")
        self.gains_label = QLabel()
        self.gains_label.setWordWrap(True)
        self.region_label = QLabel()
        self.region_label.setObjectName("hintLabel")
        self.region_label.setWordWrap(True)

        self.band_group = QButtonGroup(self)
        self.band_group.setExclusive(True)
        self.band_buttons: dict[str, QToolButton] = {}
        swatches = QHBoxLayout()
        swatches.setSpacing(6)
        for index, name in enumerate(HUE_BAND_NAMES):
            button = QToolButton()
            button.setObjectName("hueBandButton")
            button.setCheckable(True)
            button.setFixedSize(SWATCH_SIZE)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(BAND_LABELS[name])
            button.setStyleSheet(
                f"QToolButton {{ background-color: {BAND_SWATCHES[name]}; "
                "border: 1px solid #d6d3d1; border-radius: 6px; }"
                " QToolButton:checked { border: 3px solid #111827; }"
            )
            self.band_group.addButton(button, index)
            self.band_buttons[name] = button
            swatches.addWidget(button)
        swatches.addStretch(1)
        self.band_label = QLabel()
        self.hue_slider = LabeledSlider("Matiz", -HUE_SHIFT_LIMIT_DEG, HUE_SHIFT_LIMIT_DEG, 0, display=format_degrees)
        self.saturation_slider = LabeledSlider(
            "Saturação", -BAND_SATURATION_LIMIT, BAND_SATURATION_LIMIT, 0, display=format_signed_percent
        )
        self.luminance_slider = LabeledSlider(
            "Luminância", -BAND_LUMINANCE_LIMIT, BAND_LUMINANCE_LIMIT, 0, display=format_signed_percent
        )
        self.band_sliders = {
            "hue_shift": self.hue_slider,
            "saturation": self.saturation_slider,
            "luminance": self.luminance_slider,
        }
        self.band_reset_button = QPushButton("Redefinir esta cor")
        self.band_reset_button.setObjectName("secondaryButton")
        self.all_bands_reset_button = QPushButton("Redefinir todas as cores")
        self.all_bands_reset_button.setObjectName("secondaryButton")

        self.tint_slider.valueChanged.connect(lambda _value: self.valueChanged.emit())
        self.tint_slider.resetRequested.connect(lambda: self.tint_slider.setValue(DEFAULT_TINT))
        self.neutralize_button.clicked.connect(lambda: self.neutralizeRequested.emit(False))
        self.auto_button.clicked.connect(lambda: self.neutralizeRequested.emit(True))
        self.gains_reset_button.clicked.connect(self.gainsResetRequested.emit)
        self.region_reset_button.clicked.connect(self.regionResetRequested.emit)
        self.band_group.idClicked.connect(self.select_band)
        for field_name, slider in self.band_sliders.items():
            slider.valueChanged.connect(lambda value, name=field_name: self._store_band_value(name, value))
            slider.resetRequested.connect(lambda slider=slider: slider.setValue(0))
        self.band_reset_button.clicked.connect(lambda: self.set_band(self._current_band, HueBandAdjust(), emit=True))
        self.all_bands_reset_button.clicked.connect(self.reset_bands)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.neutralize_button)
        buttons_row.addWidget(self.auto_button)
        reset_row = QHBoxLayout()
        reset_row.addWidget(self.gains_reset_button)
        reset_row.addWidget(self.region_reset_button)
        reset_row.addStretch(1)
        band_reset_row = QHBoxLayout()
        band_reset_row.addWidget(self.band_reset_button)
        band_reset_row.addWidget(self.all_bands_reset_button)
        band_reset_row.addStretch(1)
        self.add_sections(
            group_box(
                "Balanço de branco",
                self.tint_slider,
                buttons_row,
                reset_row,
                self.gains_label,
                self.region_label,
            ),
            group_box(
                "Por cor (HSL)",
                swatches,
                self.band_label,
                self.hue_slider,
                self.saturation_slider,
                self.luminance_slider,
                band_reset_row,
            ),
        )
        self.set_gains(NEUTRAL_GAINS)
        self.set_region_info("")
        self.select_band(0)

    # --- settings -------------------------------------------------------------------------
    def settings(self, gains: tuple[float, float, float]) -> ColorGradeSettings:
        return ColorGradeSettings(tint=self.tint_slider.value(), white_balance_gains=gains, bands=tuple(self._bands))

    def band(self, index: int) -> HueBandAdjust:
        return self._bands[index]

    # --- bands ----------------------------------------------------------------------------
    def select_band(self, index: int) -> None:
        self._current_band = index
        button = self.band_group.button(index)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        self._show_band(self._bands[index])
        self.band_label.setText(f"Cor selecionada: {BAND_LABELS[HUE_BAND_NAMES[index]]}")

    def set_band(self, index: int, adjust: HueBandAdjust, *, emit: bool = False) -> None:
        self._bands[index] = adjust
        if index == self._current_band:
            self._show_band(adjust)
        if emit:
            self.valueChanged.emit()

    def reset_bands(self) -> None:
        self._bands = list(DEFAULT_HUE_BANDS)
        self._show_band(self._bands[self._current_band])
        self.valueChanged.emit()

    def _show_band(self, adjust: HueBandAdjust) -> None:
        for field_name, slider in self.band_sliders.items():
            slider.blockSignals(True)
            slider.slider.setValue(int(getattr(adjust, field_name)))
            slider.blockSignals(False)

    def _store_band_value(self, field_name: str, value: int) -> None:
        self._bands[self._current_band] = replace(self._bands[self._current_band], **{field_name: value})
        self.valueChanged.emit()

    # --- white balance --------------------------------------------------------------------
    def set_gains(self, gains: tuple[float, float, float]) -> None:
        neutral = all(abs(gain - 1.0) <= 1e-3 for gain in gains)
        text = f"Ganhos B {gains[0]:.2f} · G {gains[1]:.2f} · R {gains[2]:.2f}"
        self.gains_label.setText(f"{text} (neutro)." if neutral else f"{text}.")
        self.gains_reset_button.setEnabled(not neutral and self.tint_slider.isEnabled())

    def set_region_info(self, text: str) -> None:
        self.region_label.setText(
            text
            or "Arraste o retângulo roxo no preview sobre algo que deveria ser cinza ou branco e clique em Neutralizar."
        )

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.tint_slider,
            self.neutralize_button,
            self.auto_button,
            self.region_reset_button,
            self.hue_slider,
            self.saturation_slider,
            self.luminance_slider,
            self.band_reset_button,
            self.all_bands_reset_button,
        ):
            widget.setEnabled(enabled)
        for button in self.band_buttons.values():
            button.setEnabled(enabled)
        self.gains_reset_button.setEnabled(enabled and "neutro" not in self.gains_label.text())
