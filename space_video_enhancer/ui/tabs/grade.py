"""Tab "HSL": white balance, the exposure lock and per-hue-band hue/saturation/luminance.

The purple reference region is edited directly on the preview while this tab is active; the
window turns it into channel gains with :func:`pipeline.grade.neutral_gains` when the user
clicks "Neutralizar", and into the exposure-lock reference (:mod:`pipeline.exposure`) when the
user clicks "Usar este frame". The eight color swatches select which band the three sliders edit.
"""

from __future__ import annotations

import colorsys
from dataclasses import replace

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QToolButton

from space_video_enhancer.settings import (
    BAND_LUMINANCE_LIMIT,
    BAND_SATURATION_LIMIT,
    DEFAULT_HUE_BANDS,
    EXPOSURE_DEFAULT_MAX_EV,
    EXPOSURE_MAX_EV_LIMIT,
    HUE_BAND_CENTERS_DEG,
    HUE_BAND_NAMES,
    HUE_SHIFT_LIMIT_DEG,
    NEUTRAL_GAINS,
    ColorGradeSettings,
    ExposureLockSettings,
    HueBandAdjust,
    NormalizedRect,
)
from space_video_enhancer.ui import palette
from space_video_enhancer.ui.format import format_percent, format_signed
from space_video_enhancer.ui.tabs.base import TabPage, form_rows, group_box, select_combo_data, set_checked_silently
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
EXPOSURE_LIMIT_STEP_EV = 0.25
"""The limit slider moves in quarter stops."""
EXPOSURE_SCOPES: list[tuple[str, bool]] = [("Região roxa do preview", True), ("Frame inteiro", False)]
EXPOSURE_MODES: list[tuple[str, bool]] = [("Brilho e cor (B/G/R)", True), ("Só brilho", False)]


def hsv_hex(hue_deg: float, saturation: float, value: float) -> str:
    """A ``#rrggbb`` string for an HSV colour; used to paint the HSL slider grooves."""
    red, green, blue = colorsys.hsv_to_rgb((hue_deg % 360.0) / 360.0, saturation, value)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def format_degrees(value: int) -> str:
    return f"{value:+d}°" if value else "0°"


def format_signed_percent(value: int) -> str:
    return f"{value:+d}%" if value else "0%"


def format_ev_limit(value: int) -> str:
    return f"±{value * EXPOSURE_LIMIT_STEP_EV:.2f} EV"


class ColorGradeTab(TabPage):
    valueChanged = Signal()
    neutralizeRequested = Signal(bool)
    """``True`` = whole frame (gray world), ``False`` = the gray reference region on the preview."""
    whitePatchRequested = Signal()
    """Balance by the brightest pixels of the region — no gray card needed, just a highlight."""
    gainsResetRequested = Signal()
    regionResetRequested = Signal()
    exposureChanged = Signal()
    """Any exposure-lock control moved (the window re-renders the preview)."""
    exposureCaptureRequested = Signal()
    """Meter the current frame's region and make it the exposure reference."""
    exposureClearRequested = Signal()
    exposureScopeChanged = Signal()
    """Region <-> whole frame: the reference was metered the other way and must be re-metered."""

    def __init__(self) -> None:
        super().__init__()
        self._bands: list[HueBandAdjust] = list(DEFAULT_HUE_BANDS)
        self._current_band = 0
        self._exposure_reference: tuple[float, float, float] | None = None

        self.tint_slider = LabeledSlider("Tint (verde ↔ magenta)", 0, 100, DEFAULT_TINT, display=format_signed)
        self.tint_slider.setToolTip(
            "Complementa a temperatura: para a esquerda compensa magenta, para a direita compensa verde."
        )
        self.neutralize_button = QPushButton("Neutralizar pela região")
        self.neutralize_button.setToolTip("Calcula os ganhos B/G/R para a região roxa do preview virar cinza neutro.")
        self.white_patch_button = QPushButton("Neutralizar pelo realce")
        self.white_patch_button.setToolTip(
            "Sem cinza na cena? Mire a região roxa num brilho/reflexo (a parte mais clara) e use isto: "
            "trata os pixels mais claros como branco e corrige o dominante por ali."
        )
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
                f"border: 1px solid {palette.BORDER_STRONG}; border-radius: 4px; }}"
                f" QToolButton:checked {{ border: 3px solid {palette.TEXT}; }}"
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
        self.hue_low_label = QLabel()
        self.hue_low_label.setObjectName("hintLabel")
        self.hue_high_label = QLabel()
        self.hue_high_label.setObjectName("hintLabel")
        self.hue_high_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.band_reset_button = QPushButton("Redefinir esta cor")
        self.band_reset_button.setObjectName("secondaryButton")
        self.all_bands_reset_button = QPushButton("Redefinir todas as cores")
        self.all_bands_reset_button.setObjectName("secondaryButton")

        self.exposure_checkbox = QCheckBox("Travar exposição no frame de referência")
        self.exposure_checkbox.setToolTip(
            "Mede a região em cada frame e aplica ganhos B/G/R para ela ficar igual ao frame de referência: "
            "corrige o brilho e o balanço de branco que a câmera muda sozinha (auto-exposição / AWB)."
        )
        self.exposure_capture_button = QPushButton("Usar este frame como referência")
        self.exposure_capture_button.setToolTip(
            "Escolha um frame com a exposição certa e sem nada tapando a região; a região deve ser uma parte fixa "
            "da cena (parede, chão), em meio-tom e sem estourar."
        )
        self.exposure_clear_button = QPushButton("Limpar referência")
        self.exposure_clear_button.setObjectName("secondaryButton")
        self.exposure_scope_combo = QComboBox()
        for label, use_region in EXPOSURE_SCOPES:
            self.exposure_scope_combo.addItem(label, use_region)
        self.exposure_scope_combo.setToolTip(
            "Frame inteiro só funciona em cena parada: alguém entrando no quadro muda a média e a correção pulsa."
        )
        self.exposure_mode_combo = QComboBox()
        for label, color in EXPOSURE_MODES:
            self.exposure_mode_combo.addItem(label, color)
        self.exposure_strength_slider = LabeledSlider("Intensidade", 0, 100, 100, display=format_percent)
        self.exposure_limit_slider = LabeledSlider(
            "Limite",
            1,
            round(EXPOSURE_MAX_EV_LIMIT / EXPOSURE_LIMIT_STEP_EV),
            round(EXPOSURE_DEFAULT_MAX_EV / EXPOSURE_LIMIT_STEP_EV),
            display=format_ev_limit,
        )
        self.exposure_limit_slider.setToolTip("Maior correção permitida, em stops (EV), para cima ou para baixo.")
        self.exposure_label = QLabel()
        self.exposure_label.setWordWrap(True)
        self.exposure_status_label = QLabel()
        self.exposure_status_label.setObjectName("hintLabel")
        self.exposure_status_label.setWordWrap(True)

        self.tint_slider.valueChanged.connect(lambda _value: self.valueChanged.emit())
        self.tint_slider.resetRequested.connect(lambda: self.tint_slider.setValue(DEFAULT_TINT))
        self.neutralize_button.clicked.connect(lambda: self.neutralizeRequested.emit(False))
        self.white_patch_button.clicked.connect(self.whitePatchRequested.emit)
        self.auto_button.clicked.connect(lambda: self.neutralizeRequested.emit(True))
        self.gains_reset_button.clicked.connect(self.gainsResetRequested.emit)
        self.region_reset_button.clicked.connect(self.regionResetRequested.emit)
        self.band_group.idClicked.connect(self.select_band)
        for field_name, slider in self.band_sliders.items():
            slider.valueChanged.connect(lambda value, name=field_name: self._store_band_value(name, value))
            slider.resetRequested.connect(lambda slider=slider: slider.setValue(0))
        self.band_reset_button.clicked.connect(lambda: self.set_band(self._current_band, HueBandAdjust(), emit=True))
        self.all_bands_reset_button.clicked.connect(self.reset_bands)
        self.exposure_checkbox.toggled.connect(lambda _checked: self._exposure_controls_changed())
        self.exposure_capture_button.clicked.connect(self.exposureCaptureRequested.emit)
        self.exposure_clear_button.clicked.connect(self.exposureClearRequested.emit)
        self.exposure_scope_combo.currentIndexChanged.connect(lambda _index: self.exposureScopeChanged.emit())
        self.exposure_scope_combo.currentIndexChanged.connect(lambda _index: self._exposure_controls_changed())
        self.exposure_mode_combo.currentIndexChanged.connect(lambda _index: self._exposure_controls_changed())
        self.exposure_strength_slider.valueChanged.connect(lambda _value: self.exposureChanged.emit())
        self.exposure_strength_slider.resetRequested.connect(lambda: self.exposure_strength_slider.setValue(100))
        self.exposure_limit_slider.valueChanged.connect(lambda _value: self.exposureChanged.emit())
        self.exposure_limit_slider.resetRequested.connect(
            lambda: self.exposure_limit_slider.setValue(round(EXPOSURE_DEFAULT_MAX_EV / EXPOSURE_LIMIT_STEP_EV))
        )

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.neutralize_button)
        buttons_row.addWidget(self.white_patch_button)
        auto_row = QHBoxLayout()
        auto_row.addWidget(self.auto_button)
        auto_row.addStretch(1)
        reset_row = QHBoxLayout()
        reset_row.addWidget(self.gains_reset_button)
        reset_row.addWidget(self.region_reset_button)
        reset_row.addStretch(1)
        hue_ends_row = QHBoxLayout()
        hue_ends_row.addWidget(self.hue_low_label)
        hue_ends_row.addStretch(1)
        hue_ends_row.addWidget(self.hue_high_label)
        band_reset_row = QHBoxLayout()
        band_reset_row.addWidget(self.band_reset_button)
        band_reset_row.addWidget(self.all_bands_reset_button)
        band_reset_row.addStretch(1)
        exposure_buttons = QHBoxLayout()
        exposure_buttons.addWidget(self.exposure_capture_button)
        exposure_buttons.addWidget(self.exposure_clear_button)
        exposure_buttons.addStretch(1)
        self.add_sections(
            group_box(
                "Balanço de branco",
                self.tint_slider,
                buttons_row,
                auto_row,
                reset_row,
                self.gains_label,
                self.region_label,
            ),
            group_box(
                "Iluminação constante",
                self.exposure_checkbox,
                exposure_buttons,
                form_rows(("Medir em", self.exposure_scope_combo), ("Corrigir", self.exposure_mode_combo)),
                self.exposure_strength_slider,
                self.exposure_limit_slider,
                self.exposure_label,
                self.exposure_status_label,
                spacing=10,
            ),
            group_box(
                "Por cor (HSL)",
                swatches,
                self.band_label,
                self.hue_slider,
                hue_ends_row,
                self.saturation_slider,
                self.luminance_slider,
                band_reset_row,
            ),
        )
        self.set_gains(NEUTRAL_GAINS)
        self.set_region_info("")
        self.set_exposure_reference(None)
        self.set_exposure_status("")
        self.select_band(0)

    # --- settings -------------------------------------------------------------------------
    def settings(self, gains: tuple[float, float, float]) -> ColorGradeSettings:
        return ColorGradeSettings(tint=self.tint_slider.value(), white_balance_gains=gains, bands=tuple(self._bands))

    def band(self, index: int) -> HueBandAdjust:
        return self._bands[index]

    def apply(
        self,
        grade: ColorGradeSettings,
        exposure: ExposureLockSettings,
        *,
        exposure_frame: int | None = None,
    ) -> None:
        """Show another clip's white balance, hue bands and exposure lock without emitting any signal."""
        widgets = (
            self.tint_slider,
            self.exposure_checkbox,
            self.exposure_scope_combo,
            self.exposure_mode_combo,
            self.exposure_strength_slider,
            self.exposure_limit_slider,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.tint_slider.slider.setValue(int(grade.tint))
            self._bands = list(grade.bands) if len(grade.bands) == len(HUE_BAND_NAMES) else list(DEFAULT_HUE_BANDS)
            self._show_band(self._bands[self._current_band])
            self.set_gains(grade.white_balance_gains)
            set_checked_silently(self.exposure_checkbox, exposure.enabled)
            select_combo_data(self.exposure_scope_combo, exposure.region is not None)
            select_combo_data(self.exposure_mode_combo, exposure.color)
            self.exposure_strength_slider.slider.setValue(int(exposure.strength))
            self.exposure_limit_slider.slider.setValue(max(1, round(exposure.max_ev / EXPOSURE_LIMIT_STEP_EV)))
            self._exposure_reference = exposure.reference
            self._show_exposure_reference(exposure.reference, exposure_frame)
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._update_exposure_buttons()

    # --- bands ----------------------------------------------------------------------------
    def select_band(self, index: int) -> None:
        self._current_band = index
        button = self.band_group.button(index)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        self._show_band(self._bands[index])
        self.band_label.setText(f"Cor selecionada: {BAND_LABELS[HUE_BAND_NAMES[index]]}")
        self._paint_band_sliders(index)

    def _paint_band_sliders(self, index: int) -> None:
        """Colour the three sliders for the selected band so each direction reads as a colour, not a number."""
        center = float(HUE_BAND_CENTERS_DEG[index])
        count = len(HUE_BAND_NAMES)
        previous = BAND_LABELS[HUE_BAND_NAMES[(index - 1) % count]]
        following = BAND_LABELS[HUE_BAND_NAMES[(index + 1) % count]]
        span = float(HUE_SHIFT_LIMIT_DEG)
        self.hue_slider.set_groove_gradient(
            [hsv_hex(center - span + fraction * 2 * span, 0.85, 0.90) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)]
        )
        self.saturation_slider.set_groove_gradient(
            [hsv_hex(center, saturation, 0.88) for saturation in (0.10, 0.40, 0.70, 1.0)]
        )
        self.luminance_slider.set_groove_gradient([hsv_hex(center, 0.55, value) for value in (0.22, 0.45, 0.70, 1.0)])
        self.hue_low_label.setText(f"◀ {previous}")
        self.hue_high_label.setText(f"{following} ▶")

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

    # --- exposure lock --------------------------------------------------------------------
    def exposure_uses_region(self) -> bool:
        return bool(self.exposure_scope_combo.currentData())

    def exposure_settings(self, region: NormalizedRect | None) -> ExposureLockSettings:
        """``region`` is the purple rectangle; ignored (whole frame) when the scope says so."""
        return ExposureLockSettings(
            enabled=self.exposure_checkbox.isChecked(),
            region=region if self.exposure_uses_region() else None,
            reference=self._exposure_reference,
            strength=self.exposure_strength_slider.value(),
            max_ev=self.exposure_limit_slider.value() * EXPOSURE_LIMIT_STEP_EV,
            color=bool(self.exposure_mode_combo.currentData()),
        )

    def exposure_reference(self) -> tuple[float, float, float] | None:
        return self._exposure_reference

    def set_exposure_reference(
        self, reference: tuple[float, float, float] | None, *, frame_index: int | None = None
    ) -> None:
        """The reference medians (or ``None``); also turns the lock on when a reference is captured."""
        self._exposure_reference = reference
        self._show_exposure_reference(reference, frame_index)
        if reference is not None and not self.exposure_checkbox.isChecked():
            self.exposure_checkbox.setChecked(True)  # emits exposureChanged through the toggle
        self._update_exposure_buttons()

    def _show_exposure_reference(self, reference: tuple[float, float, float] | None, frame_index: int | None) -> None:
        if reference is None:
            self.exposure_label.setText(
                "Sem referência. Vá até um frame bem exposto e clique em “Usar este frame como referência”."
            )
        else:
            where = f" (frame {frame_index + 1})" if frame_index is not None else ""
            self.exposure_label.setText(
                f"Referência{where}: medianas B {reference[0]:.0f} · G {reference[1]:.0f} · R {reference[2]:.0f}."
            )

    def set_exposure_status(self, text: str) -> None:
        self.exposure_status_label.setText(text)

    def _exposure_controls_changed(self) -> None:
        self._update_exposure_buttons()
        self.exposureChanged.emit()

    def _update_exposure_buttons(self) -> None:
        enabled = self.exposure_capture_button.isEnabled() or self.exposure_clear_button.isEnabled()
        self.exposure_clear_button.setEnabled(enabled and self._exposure_reference is not None)

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
            self.white_patch_button,
            self.auto_button,
            self.region_reset_button,
            self.hue_slider,
            self.saturation_slider,
            self.luminance_slider,
            self.band_reset_button,
            self.all_bands_reset_button,
            self.exposure_checkbox,
            self.exposure_capture_button,
            self.exposure_scope_combo,
            self.exposure_mode_combo,
            self.exposure_strength_slider,
            self.exposure_limit_slider,
        ):
            widget.setEnabled(enabled)
        for button in self.band_buttons.values():
            button.setEnabled(enabled)
        self.gains_reset_button.setEnabled(enabled and "neutro" not in self.gains_label.text())
        self.exposure_clear_button.setEnabled(enabled and self._exposure_reference is not None)
