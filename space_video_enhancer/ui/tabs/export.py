"""Tab "Exportar": output scale, worker threads, a summary of what the export will contain, and the export button.

*What* is exported — the clips, the I/O markers, the removed ranges — is edited on the timeline;
this tab only says how the frames are processed and sums the result up.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton

from space_video_enhancer.settings import FilterSettings
from space_video_enhancer.ui.tabs.base import TabPage, form_rows, group_box, select_combo_data

THREAD_OPTIONS = (1, 2, 4, 6, 8)
SCALE_OPTIONS: list[tuple[str, float]] = [
    ("0.5x (metade do tamanho)", 0.5),
    ("0.75x", 0.75),
    ("1.0x (tamanho original)", 1.0),
    ("1.5x (720p → 1080p)", 1.5),
    ("2.0x (dobro do tamanho)", 2.0),
]
DEFAULT_SCALE = 1.0


def default_thread_count() -> int:
    half = max(1, (os.cpu_count() or 2) // 2)
    return max(option for option in THREAD_OPTIONS if option <= half)


class ExportTab(TabPage):
    scaleChanged = Signal()
    exportRequested = Signal()
    """The export button (the window's export action does the work)."""

    def __init__(self) -> None:
        super().__init__()
        self.scale_combo = QComboBox()
        for label, factor in SCALE_OPTIONS:
            self.scale_combo.addItem(label, factor)
        select_combo_data(self.scale_combo, DEFAULT_SCALE)
        self.scale_combo.setToolTip(
            "Tamanho do vídeo exportado em relação ao original; o preview mostra o resultado na mesma escala."
        )
        self.thread_combo = QComboBox()
        for thread_count in THREAD_OPTIONS:
            self.thread_combo.addItem(str(thread_count), thread_count)
        self.thread_combo.setCurrentIndex(max(0, self.thread_combo.findData(default_thread_count())))
        self.thread_combo.setToolTip(
            "Quantos frames processar em paralelo. A estabilização entre frames é sequencial por natureza "
            "e usa uma thread só."
        )
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.export_button = QPushButton("Exportar vídeo…")
        self.export_button.setToolTip(
            "Processa o trecho entre os marcadores, sem os trechos removidos, e salva um MP4 com o áudio (Ctrl+E)"
        )
        self.hint_label = QLabel(
            "O que sai no arquivo é definido na linha do tempo: os marcadores [ e ] delimitam o trecho, "
            "os cortes marcam trechos removidos e os clipes saem na ordem da faixa superior."
        )
        self.hint_label.setObjectName("hintLabel")
        self.hint_label.setWordWrap(True)

        self.scale_combo.currentIndexChanged.connect(lambda _index: self.scaleChanged.emit())
        self.export_button.clicked.connect(self.exportRequested.emit)

        self.add_sections(
            group_box(
                "Processamento",
                form_rows(("Escala de saída", self.scale_combo), ("Threads", self.thread_combo)),
            ),
            group_box("Exportação", self.summary_label, self.export_button, self.hint_label, spacing=12),
        )
        self.set_summary("Abra um vídeo para exportar.")

    def scale_factor(self) -> float:
        return float(self.scale_combo.currentData() or 1.0)

    def worker_count(self) -> int:
        return int(self.thread_combo.currentData() or 1)

    def apply(self, settings: FilterSettings) -> None:
        """Select ``settings``' output scale without emitting ``scaleChanged``."""
        self.scale_combo.blockSignals(True)
        try:
            select_combo_data(self.scale_combo, settings.scale_factor)
        finally:
            self.scale_combo.blockSignals(False)

    def set_summary(self, text: str) -> None:
        """Range, removed ranges and output size of the export, as the window computes them."""
        self.summary_label.setText(text)

    def set_controls_enabled(self, enabled: bool) -> None:
        """Processing options only; the export button follows the window's export action."""
        self.scale_combo.setEnabled(enabled)
        self.thread_combo.setEnabled(enabled)
