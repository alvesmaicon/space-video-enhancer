"""Tab "Tempo": everything about time — the timeline's frame rate and each clip's speed/retiming.

The timeline fps is a property of the whole sequence (the rate of the first clip, changeable here);
speed, the fps-retiming mode and the per-clip mute are per clip, so they follow the selected clip.
Both feed :class:`CaptureSettings` (assembled by the window) except the fps, which retimes the sequence.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel

from space_video_enhancer.settings import CaptureSettings, RetimingMode
from space_video_enhancer.ui.tabs.base import (
    TabPage,
    combo_enum,
    enum_combo,
    form_rows,
    group_box,
    select_combo_data,
    set_checked_silently,
)

COMMON_TIMELINE_FPS = (23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0)
SPEED_OPTIONS: list[tuple[str, float]] = [
    ("0.25x (bem lento)", 0.25),
    ("0.5x (lento)", 0.5),
    ("1x (normal)", 1.0),
    ("2x (rápido)", 2.0),
    ("4x (bem rápido)", 4.0),
]
RETIMING_OPTIONS: list[tuple[str, RetimingMode]] = [
    ("Frame mais próximo", RetimingMode.NEAREST),
    ("Mistura de quadros", RetimingMode.BLEND),
    ("Fluxo óptico", RetimingMode.OPTICAL_FLOW),
]
RETIMING_DESCRIPTIONS: dict[RetimingMode, str] = {
    RetimingMode.NEAREST: (
        "Mostra o quadro mais próximo no tempo; repete ou descarta quadros inteiros. Rápido, pode trepidar."
    ),
    RetimingMode.BLEND: (
        "Faz uma dissolvência entre os dois quadros vizinhos. Mais suave, deixa fantasma em movimento rápido."
    ),
    RetimingMode.OPTICAL_FLOW: (
        "Sintetiza o quadro intermediário seguindo o movimento. Melhor qualidade, mais lento — aplicado só na "
        "exportação; o preview mostra a mistura de quadros."
    ),
}
RETIMING_TOOLTIP = (
    "Como este clipe é reajustado quando o fps dele difere do fps da linha do tempo, seja porque o clipe foi "
    "gravado em outra taxa ou porque você mudou a velocidade dele. Fluxo óptico dá o slow-mo mais suave."
)


def _fps_label(value: float) -> str:
    return str(round(value)) if abs(value - round(value)) < 0.01 else f"{value:g}"


class TempoTab(TabPage):
    timelineFpsChanged = Signal()
    """The user picked another timeline fps; the window retimes the whole sequence to it."""
    captureChanged = Signal()
    """Speed, the retiming mode or the mute changed (all per-clip ``CaptureSettings`` fields)."""

    def __init__(self) -> None:
        super().__init__()
        self.timeline_fps_combo = QComboBox()
        self.timeline_fps_combo.setToolTip(
            "fps da linha do tempo e do arquivo exportado (o do primeiro clipe, por padrão). Clipes com outro fps "
            "são reajustados para este, pelo modo de reajuste escolhido abaixo."
        )
        self.speed_combo = QComboBox()
        for label, factor in SPEED_OPTIONS:
            self.speed_combo.addItem(label, factor)
        select_combo_data(self.speed_combo, 1.0)
        self.speed_combo.setToolTip(
            "Velocidade deste clipe: abaixo de 1x é câmera lenta (o modo de reajuste faz a interpolação; "
            "fluxo óptico dá o slow-mo mais suave), acima é câmera rápida. O áudio é esticado para acompanhar."
        )
        self.retiming_combo = enum_combo(RETIMING_OPTIONS)
        self.retiming_combo.setToolTip(RETIMING_TOOLTIP)
        for index, (_label, retiming_mode) in enumerate(RETIMING_OPTIONS):
            self.retiming_combo.setItemData(index, RETIMING_DESCRIPTIONS[retiming_mode], Qt.ItemDataRole.ToolTipRole)
        self.mute_checkbox = QCheckBox("Sem áudio neste clipe")
        self.timeline_hint = QLabel(
            "Vale para a sequência inteira. Ao abrir um vídeo ela assume o fps do primeiro clipe."
        )
        self.timeline_hint.setObjectName("hintLabel")
        self.timeline_hint.setWordWrap(True)
        self.speed_hint = QLabel(
            "Valem só para o clipe sob o playhead. A câmera lenta usa o modo de reajuste para inventar os "
            "quadros intermediários."
        )
        self.speed_hint.setObjectName("hintLabel")
        self.speed_hint.setWordWrap(True)

        self.timeline_fps_combo.currentIndexChanged.connect(lambda _index: self.timelineFpsChanged.emit())
        self.speed_combo.currentIndexChanged.connect(lambda _index: self.captureChanged.emit())
        self.retiming_combo.currentIndexChanged.connect(lambda _index: self.captureChanged.emit())
        self.mute_checkbox.toggled.connect(lambda _checked: self.captureChanged.emit())

        self.add_sections(
            group_box(
                "Linha do tempo",
                form_rows(("fps da linha do tempo", self.timeline_fps_combo)),
                self.timeline_hint,
            ),
            group_box(
                "Velocidade do clipe",
                form_rows(("Velocidade", self.speed_combo), ("Reajuste de fps", self.retiming_combo)),
                self.mute_checkbox,
                self.speed_hint,
            ),
        )

    # --- timeline fps ---------------------------------------------------------------------
    def timeline_fps(self) -> float:
        return float(self.timeline_fps_combo.currentData() or 0.0)

    def set_timeline_fps_options(self, clip_fps: list[float], current: float) -> None:
        """Rebuild the fps list (the common rates plus the clips' own) and select ``current`` without emitting."""
        clip_keys = {round(value, 3) for value in clip_fps if value > 0}
        by_key = {round(value, 3): value for value in (*COMMON_TIMELINE_FPS, *clip_fps, current) if value > 0}
        values = [by_key[key] for key in sorted(by_key)]
        self.timeline_fps_combo.blockSignals(True)
        try:
            self.timeline_fps_combo.clear()
            for value in values:
                label = _fps_label(value) + (" (clipe)" if round(value, 3) in clip_keys else "")
                self.timeline_fps_combo.addItem(label, value)
            if values:
                select_combo_data(self.timeline_fps_combo, min(values, key=lambda value: abs(value - current)))
        finally:
            self.timeline_fps_combo.blockSignals(False)

    # --- per-clip speed / retiming --------------------------------------------------------
    def retiming_mode(self) -> RetimingMode:
        return combo_enum(self.retiming_combo, RetimingMode, RetimingMode.NEAREST)

    def speed(self) -> float:
        return float(self.speed_combo.currentData() or 1.0)

    def mute(self) -> bool:
        return self.mute_checkbox.isChecked()

    def apply_capture(self, capture: CaptureSettings) -> None:
        """Show another clip's speed, retiming and mute without emitting ``captureChanged``."""
        for combo, value in ((self.retiming_combo, capture.retiming.value), (self.speed_combo, capture.speed)):
            combo.blockSignals(True)
            try:
                select_combo_data(combo, value)
            finally:
                combo.blockSignals(False)
        set_checked_silently(self.mute_checkbox, capture.mute)

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.timeline_fps_combo, self.speed_combo, self.retiming_combo, self.mute_checkbox):
            widget.setEnabled(enabled)
