"""Tab "Estabilizar": mode, strength, smoothing, focal region, border handling — and the capture artefacts.

The second group, "Artefatos de captura", holds what the sensor and the scan did to the frame:
deinterlacing (first stage of the pipeline) and the rolling-shutter unskew (inside the
stabilization stage, since it needs the same inter-frame motion). Both work with the mode off.

The focal region itself is edited directly on the preview whenever this tab is active
and the mode uses a focal point; there is no separate "edit" toggle. The reference frame
the lock holds on to is the one on screen when the mode is chosen or the region moves;
"Usar este frame como referência" re-takes it on the current frame without moving the region.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton

from space_video_enhancer.settings import (
    CaptureSettings,
    DeinterlaceMode,
    NormalizedRect,
    StabilizationMode,
    StabilizationSettings,
    clamp_unit,
)
from space_video_enhancer.ui.format import format_percent
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

MODE_OPTIONS: list[tuple[str, StabilizationMode]] = [
    ("Desligada", StabilizationMode.OFF),
    ("Suavizar tremor (só deslocamento)", StabilizationMode.AUTO),
    ("Suavizar tremor + rotação e zoom", StabilizationMode.SIMILARITY),
    ("Suavizar tremor + perspectiva", StabilizationMode.PERSPECTIVE),
    ("Travar uma região (camera lock)", StabilizationMode.CAMERA_LOCK),
    ("Travar uma região, suavizar se perder", StabilizationMode.HYBRID),
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
        "Mantém a região verde (arraste-a no preview) presa a onde ela estava no frame de referência; "
        "segura a última posição se perder o alvo."
    ),
    StabilizationMode.HYBRID: (
        "Prende a região verde (arraste-a no preview) com camera lock; se perder o alvo, segue o movimento "
        "global até reencontrá-lo."
    ),
}
DEFAULT_INTENSITY = 100
DEFAULT_SMOOTHING = 35
DEINTERLACE_OPTIONS: list[tuple[str, DeinterlaceMode]] = [
    ("Desligado", DeinterlaceMode.OFF),
    ("Automático (só onde há pente)", DeinterlaceMode.ADAPTIVE),
    ("Campo único (meia resolução)", DeinterlaceMode.FIELD),
]
DEINTERLACE_DESCRIPTIONS: dict[DeinterlaceMode, str] = {
    DeinterlaceMode.OFF: "Mantém os dois campos como estão.",
    DeinterlaceMode.ADAPTIVE: (
        "Onde as linhas pares e ímpares discordam (o pente que o movimento deixa), as ímpares são refeitas a "
        "partir das pares; o que está parado fica intacto, com a resolução inteira."
    ),
    DeinterlaceMode.FIELD: (
        "Descarta o campo ímpar em todo o frame e o reconstrói a partir do par: elimina qualquer pente, mas "
        "reduz o detalhe vertical à metade."
    ),
}
WOBBLE_TOOLTIP = (
    "Com tremor forte, a velocidade da câmera muda enquanto o sensor lê o frame e cada frame sai esticado ou "
    "inclinado de um jeito diferente (jello). Isso é medido comparando o movimento do topo e da base do quadro "
    "entre frames e desfeito sem precisar do percentual de leitura. Não corrige a inclinação constante de uma "
    "panorâmica — para essa, use o slider."
)
ROLLING_SHUTTER_TOOLTIP = (
    "Desfaz a inclinação que o sensor deixa quando a câmera se move (postes que tombam, imagem que estica): "
    "cada linha é reposicionada pelo movimento medido entre frames. O valor é quanto do intervalo entre "
    "frames o sensor gasta lendo a imagem — 60 a 80% é típico; ajuste até as verticais ficarem retas."
)


class StabilizationTab(TabPage):
    modeChanged = Signal()
    controlChanged = Signal()
    """Intensity, smoothing, auto crop or zoom out changed (inter-frame memory must restart)."""
    resetRequested = Signal()
    """Put the focal region back at the centre (and take the reference on the current frame)."""
    referenceRequested = Signal()
    """Make the frame on screen the camera-lock reference; the focal region stays where it is."""
    captureChanged = Signal()
    """Deinterlacing or the rolling-shutter correction changed."""

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
        self.reference_button = QPushButton("Usar este frame como referência")
        self.reference_button.setToolTip(
            "Prende o camera lock ao frame que está no preview, mantendo a região verde onde está. "
            "Escolha um frame nítido, com a região bem enquadrada."
        )
        self.reset_button = QPushButton("Redefinir ponto focal")
        self.reset_button.setObjectName("secondaryButton")
        self.reset_button.setToolTip(
            "Volta a região verde para o centro do quadro e usa o frame atual como referência."
        )
        self.auto_crop_checkbox = QCheckBox("Crop automático para esconder bordas")
        self.auto_crop_checkbox.setChecked(True)
        self.auto_crop_checkbox.setToolTip(
            "Recorta uma margem fixa em todos os frames (o mesmo teto da correção), então o zoom não oscila."
        )
        self.zoom_out_checkbox = QCheckBox("Zoom out automático para preservar bordas")
        self.zoom_out_checkbox.setChecked(False)
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)

        self.deinterlace_combo = enum_combo(DEINTERLACE_OPTIONS)
        for index, (_label, deinterlace_mode) in enumerate(DEINTERLACE_OPTIONS):
            self.deinterlace_combo.setItemData(
                index, DEINTERLACE_DESCRIPTIONS[deinterlace_mode], Qt.ItemDataRole.ToolTipRole
            )
        self.rolling_shutter_slider = LabeledSlider("Correção de rolling shutter", 0, 100, 0, display=format_percent)
        self.rolling_shutter_slider.setToolTip(ROLLING_SHUTTER_TOOLTIP)
        self.wobble_checkbox = QCheckBox("Compensar a oscilação do rolling shutter (jello) automaticamente")
        self.wobble_checkbox.setToolTip(WOBBLE_TOOLTIP)
        self.capture_info_label = QLabel()
        self.capture_info_label.setObjectName("hintLabel")
        self.capture_info_label.setWordWrap(True)
        self.deinterlace_combo.currentIndexChanged.connect(self._handle_deinterlace_changed)
        self.rolling_shutter_slider.valueChanged.connect(lambda _value: self.captureChanged.emit())
        self.wobble_checkbox.toggled.connect(lambda _checked: self.captureChanged.emit())

        self.mode_combo.currentIndexChanged.connect(self._handle_mode_changed)
        self.intensity_slider.valueChanged.connect(lambda _value: self.controlChanged.emit())
        self.smoothing_slider.valueChanged.connect(lambda _value: self.controlChanged.emit())
        self.auto_crop_checkbox.toggled.connect(lambda _checked: self.controlChanged.emit())
        self.zoom_out_checkbox.toggled.connect(self._handle_zoom_out_toggled)
        self.reset_button.clicked.connect(self.resetRequested.emit)
        self.reference_button.clicked.connect(self.referenceRequested.emit)
        reference_buttons = QHBoxLayout()
        reference_buttons.addWidget(self.reference_button)
        reference_buttons.addWidget(self.reset_button)

        self.add_sections(
            group_box(
                "Estabilização",
                form_rows(("Modo", self.mode_combo)),
                self.hint_label,
                self.intensity_slider,
                self.smoothing_slider,
                self.auto_crop_checkbox,
                self.zoom_out_checkbox,
                reference_buttons,
                self.info_label,
            ),
            group_box(
                "Artefatos de captura",
                form_rows(("Desentrelaçar", self.deinterlace_combo)),
                self.rolling_shutter_slider,
                self.wobble_checkbox,
                self.capture_info_label,
            ),
        )
        self._update_hint()
        self._update_deinterlace_tooltip()

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
            rolling_shutter=clamp_unit(self.rolling_shutter_slider.value() / 100.0),
            rolling_shutter_auto=self.wobble_checkbox.isChecked(),
        )

    def deinterlace_mode(self) -> DeinterlaceMode:
        return combo_enum(self.deinterlace_combo, DeinterlaceMode, DeinterlaceMode.OFF)

    def capture_settings(self) -> CaptureSettings:
        return CaptureSettings(deinterlace=self.deinterlace_mode())

    def apply_capture(self, capture: CaptureSettings) -> None:
        """Show another clip's deinterlacing without emitting ``captureChanged``."""
        self.deinterlace_combo.blockSignals(True)
        try:
            select_combo_data(self.deinterlace_combo, capture.deinterlace.value)
        finally:
            self.deinterlace_combo.blockSignals(False)
        self._update_deinterlace_tooltip()

    def apply(self, settings: StabilizationSettings) -> None:
        """Show another clip's stabilization without emitting ``modeChanged``/``controlChanged``."""
        widgets = (
            self.mode_combo,
            self.intensity_slider,
            self.smoothing_slider,
            self.auto_crop_checkbox,
            self.zoom_out_checkbox,
            self.rolling_shutter_slider,
            self.wobble_checkbox,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            select_combo_data(self.mode_combo, settings.mode.value)
            self.intensity_slider.slider.setValue(round(settings.intensity * 100))
            self.smoothing_slider.slider.setValue(round(settings.temporal_smoothing * 100))
            self.rolling_shutter_slider.slider.setValue(round(settings.rolling_shutter * 100))
            set_checked_silently(self.wobble_checkbox, settings.rolling_shutter_auto)
            set_checked_silently(self.zoom_out_checkbox, settings.zoom_out)
            set_checked_silently(self.auto_crop_checkbox, settings.auto_crop and not settings.zoom_out)
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._update_hint()

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
        self.reference_button.setEnabled(stabilization_enabled and mode.uses_focal_point)
        self.deinterlace_combo.setEnabled(controls_enabled)  # capture fixes do not need a stabilization mode
        self.rolling_shutter_slider.setEnabled(controls_enabled)
        self.wobble_checkbox.setEnabled(controls_enabled)

    def set_info(self, text: str) -> None:
        self.info_label.setText(text)

    def set_capture_info(self, text: str) -> None:
        self.capture_info_label.setText(text)

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.mode_combo,
            self.intensity_slider,
            self.smoothing_slider,
            self.reset_button,
            self.reference_button,
            self.auto_crop_checkbox,
            self.zoom_out_checkbox,
            self.deinterlace_combo,
            self.rolling_shutter_slider,
            self.wobble_checkbox,
        ):
            widget.setEnabled(enabled)

    def _handle_deinterlace_changed(self, _index: int) -> None:
        self._update_deinterlace_tooltip()
        self.captureChanged.emit()

    def _update_deinterlace_tooltip(self) -> None:
        self.deinterlace_combo.setToolTip(DEINTERLACE_DESCRIPTIONS[self.deinterlace_mode()])

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
