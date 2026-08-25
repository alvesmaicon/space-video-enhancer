"""Tab "Exportar": output scale, worker threads, the clip range and the cuts removed inside it."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
)

from space_video_enhancer.settings import FilterSettings
from space_video_enhancer.ui.tabs.base import TabPage, form_rows, group_box, select_combo_data

THREAD_OPTIONS = (1, 2, 4, 6, 8)
SCALE_OPTIONS: list[tuple[str, float]] = [
    ("1.0x (original)", 1.0),
    ("1.5x (bom para 720p -> 1080p)", 1.5),
    ("2.0x", 2.0),
]
CUT_LIST_MAX_HEIGHT = 110


def default_thread_count() -> int:
    half = max(1, (os.cpu_count() or 2) // 2)
    return max(option for option in THREAD_OPTIONS if option <= half)


class ExportTab(TabPage):
    scaleChanged = Signal()
    markStartRequested = Signal()
    markEndRequested = Signal()
    resetRangeRequested = Signal()
    cutToggleRequested = Signal()
    """Start a cut at the current frame, or finish the pending one there."""
    cutAddRequested = Signal(str, str)
    """Typed timecodes: (from, to)."""
    cutRemoveRequested = Signal(int)
    cutsClearRequested = Signal()
    cutActivated = Signal(int)
    """Double-click on a cut: jump to it."""
    cutSelectionChanged = Signal(int)
    """Selected row (``-1`` for none), so the timeline highlights the same cut."""

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
        self.reset_range_button = QPushButton("Redefinir corte")
        for button in (self.mark_start_button, self.mark_end_button, self.reset_range_button):
            button.setObjectName("secondaryButton")
        self.mark_start_button.setToolTip("Início do corte no frame atual (I)")
        self.mark_end_button.setToolTip("Fim do corte no frame atual (O)")
        self.reset_range_button.setToolTip("Volta ao vídeo inteiro e apaga os cortes internos")
        self.range_label = QLabel("Corte: vídeo inteiro")
        self.range_label.setWordWrap(True)

        self.cut_toggle_button = QPushButton("Iniciar corte aqui")
        self.cut_toggle_button.setToolTip(
            "Marca o início do trecho a remover no frame atual; clique de novo no fim (X)"
        )
        self.cut_list = QListWidget()
        self.cut_list.setObjectName("cutList")
        self.cut_list.setMaximumHeight(CUT_LIST_MAX_HEIGHT)
        self.cut_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.cut_list.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.remove_cut_button = QPushButton("Remover corte")
        self.clear_cuts_button = QPushButton("Limpar cortes")
        for button in (self.remove_cut_button, self.clear_cuts_button):
            button.setObjectName("secondaryButton")
        self.cut_from_edit = QLineEdit()
        self.cut_from_edit.setPlaceholderText("de (mm:ss.ff)")
        self.cut_to_edit = QLineEdit()
        self.cut_to_edit.setPlaceholderText("até")
        self.add_cut_button = QPushButton("Adicionar corte")
        self.add_cut_button.setObjectName("secondaryButton")
        self.cuts_label = QLabel()
        self.cuts_label.setObjectName("hintLabel")
        self.cuts_label.setWordWrap(True)

        self.scale_combo.currentIndexChanged.connect(lambda _index: self.scaleChanged.emit())
        self.mark_start_button.clicked.connect(self.markStartRequested.emit)
        self.mark_end_button.clicked.connect(self.markEndRequested.emit)
        self.reset_range_button.clicked.connect(self.resetRangeRequested.emit)
        self.cut_toggle_button.clicked.connect(self.cutToggleRequested.emit)
        self.remove_cut_button.clicked.connect(self._emit_remove_selected)
        self.clear_cuts_button.clicked.connect(self.cutsClearRequested.emit)
        self.add_cut_button.clicked.connect(self._emit_add_from_text)
        self.cut_to_edit.returnPressed.connect(self._emit_add_from_text)
        self.cut_list.itemDoubleClicked.connect(lambda item: self.cutActivated.emit(self.cut_list.row(item)))
        self.cut_list.itemSelectionChanged.connect(self._update_cut_buttons)
        self.cut_list.itemSelectionChanged.connect(lambda: self.cutSelectionChanged.emit(self.selected_cut()))

        range_buttons = QHBoxLayout()
        range_buttons.setSpacing(10)
        range_buttons.addWidget(self.mark_start_button)
        range_buttons.addWidget(self.mark_end_button)
        range_buttons.addWidget(self.reset_range_button)

        cut_buttons = QHBoxLayout()
        cut_buttons.setSpacing(10)
        cut_buttons.addWidget(self.cut_toggle_button)
        cut_buttons.addWidget(self.remove_cut_button)
        cut_buttons.addWidget(self.clear_cuts_button)
        typed_cut = QHBoxLayout()
        typed_cut.setSpacing(8)
        typed_cut.addWidget(self.cut_from_edit)
        typed_cut.addWidget(self.cut_to_edit)
        typed_cut.addWidget(self.add_cut_button)

        self.add_sections(
            group_box(
                "Processamento",
                form_rows(("Escala de saída", self.scale_combo), ("Threads", self.thread_combo)),
            ),
            group_box("Corte temporal", range_buttons, self.range_label, spacing=12),
            group_box(
                "Cortes internos (trechos removidos)",
                cut_buttons,
                self.cut_list,
                typed_cut,
                self.cuts_label,
                spacing=10,
            ),
        )
        self.set_cuts([])
        self.set_pending_cut(False)

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

    def set_cuts(self, items: list[str]) -> None:
        """Replace the list, keeping the selection where possible."""
        selected = self.selected_cut()
        self.cut_list.blockSignals(True)
        self.cut_list.clear()
        for text in items:
            self.cut_list.addItem(QListWidgetItem(text))
        if 0 <= selected < len(items):
            self.cut_list.setCurrentRow(selected)
        self.cut_list.blockSignals(False)
        self.cut_list.setVisible(bool(items))
        self.cuts_label.setText(
            'Nenhum trecho removido. Use "Iniciar corte aqui" (X) nas duas pontas do trecho, '
            "ou digite os tempos e adicione."
            if not items
            else "Duplo clique em um corte para ir até ele; Delete remove o selecionado. "
            "Na linha do tempo, arraste as bordas ou o corpo de um corte para ajustá-lo."
        )
        self._update_cut_buttons()

    def selected_cut(self) -> int:
        return self.cut_list.currentRow() if self.cut_list.selectedItems() else -1

    def select_cut(self, index: int) -> None:
        """Mirror the timeline's selection without echoing it back."""
        if index == self.selected_cut():
            return
        self.cut_list.blockSignals(True)
        try:
            if 0 <= index < self.cut_list.count():
                self.cut_list.setCurrentRow(index)
            else:
                self.cut_list.clearSelection()
                self.cut_list.setCurrentRow(-1)
        finally:
            self.cut_list.blockSignals(False)
        self._update_cut_buttons()

    def set_pending_cut(self, pending: bool) -> None:
        self.cut_toggle_button.setText("Terminar corte aqui" if pending else "Iniciar corte aqui")

    def take_typed_cut(self) -> tuple[str, str]:
        return (self.cut_from_edit.text(), self.cut_to_edit.text())

    def clear_typed_cut(self) -> None:
        self.cut_from_edit.clear()
        self.cut_to_edit.clear()

    def _emit_remove_selected(self) -> None:
        index = self.selected_cut()
        if index >= 0:
            self.cutRemoveRequested.emit(index)

    def _emit_add_from_text(self) -> None:
        self.cutAddRequested.emit(*self.take_typed_cut())

    def _update_cut_buttons(self) -> None:
        enabled = self.cut_toggle_button.isEnabled()
        self.remove_cut_button.setEnabled(enabled and self.selected_cut() >= 0)
        self.clear_cuts_button.setEnabled(enabled and self.cut_list.count() > 0)

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.scale_combo,
            self.thread_combo,
            self.mark_start_button,
            self.mark_end_button,
            self.reset_range_button,
            self.cut_toggle_button,
            self.cut_from_edit,
            self.cut_to_edit,
            self.add_cut_button,
        ):
            widget.setEnabled(enabled)
        self._update_cut_buttons()
