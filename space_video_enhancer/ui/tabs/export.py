"""Tab "Exportar": output scale, worker threads and the temporal cut."""

from __future__ import annotations

import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton

from space_video_enhancer.settings import FilterSettings
from space_video_enhancer.ui.tabs.base import TabPage, form_rows, group_box, select_combo_data

THREAD_OPTIONS = (1, 2, 4, 6, 8)
SCALE_OPTIONS: list[tuple[str, float]] = [
    ("1.0x (original)", 1.0),
    ("1.5x (bom para 720p -> 1080p)", 1.5),
    ("2.0x", 2.0),
]


def default_thread_count() -> int:
    half = max(1, (os.cpu_count() or 2) // 2)
    return max(option for option in THREAD_OPTIONS if option <= half)


class ExportTab(TabPage):
    scaleChanged = Signal()
    markStartRequested = Signal()
    markEndRequested = Signal()
    resetRangeRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.scale_combo = QComboBox()
        for label, factor in SCALE_OPTIONS:
            self.scale_combo.addItem(label, factor)
        self.thread_combo = QComboBox()
        for thread_count in THREAD_OPTIONS:
            self.thread_combo.addItem(str(thread_count), thread_count)
        self.thread_combo.setCurrentIndex(max(0, self.thread_combo.findData(default_thread_count())))
        self.thread_combo.setToolTip("Quantas threads usar na exportação")
        self.mark_start_button = QPushButton("Marcar início")
        self.mark_end_button = QPushButton("Marcar fim")
        self.reset_range_button = QPushButton("Vídeo inteiro")
        self.range_label = QLabel("Corte: vídeo inteiro")
        self.range_label.setWordWrap(True)

        self.scale_combo.currentIndexChanged.connect(lambda _index: self.scaleChanged.emit())
        self.mark_start_button.clicked.connect(self.markStartRequested.emit)
        self.mark_end_button.clicked.connect(self.markEndRequested.emit)
        self.reset_range_button.clicked.connect(self.resetRangeRequested.emit)

        range_buttons = QHBoxLayout()
        range_buttons.setSpacing(10)
        range_buttons.addWidget(self.mark_start_button)
        range_buttons.addWidget(self.mark_end_button)
        range_buttons.addWidget(self.reset_range_button)

        self.add_sections(
            group_box(
                "Processamento",
                form_rows(("Escala de saída", self.scale_combo), ("Threads", self.thread_combo)),
            ),
            group_box("Corte temporal", range_buttons, self.range_label, spacing=12),
        )

    def scale_factor(self) -> float:
        return float(self.scale_combo.currentData() or 1.0)

    def worker_count(self) -> int:
        return int(self.thread_combo.currentData() or 1)

    def apply(self, settings: FilterSettings) -> None:
        """Select the preset's output scale without emitting ``scaleChanged``."""
        self.scale_combo.blockSignals(True)
        try:
            select_combo_data(self.scale_combo, settings.scale_factor)
        finally:
            self.scale_combo.blockSignals(False)

    def set_range_text(self, text: str) -> None:
        self.range_label.setText(text)

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.scale_combo,
            self.thread_combo,
            self.mark_start_button,
            self.mark_end_button,
            self.reset_range_button,
        ):
            widget.setEnabled(enabled)
