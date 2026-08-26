"""Tab "Exportar": the sequence of clips, output scale, worker threads, the export range and the cuts."""

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
CLIP_LIST_MAX_HEIGHT = 96


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
    clipAddRequested = Signal()
    clipRemoveRequested = Signal(int)
    clipMoveRequested = Signal(int, int)
    """``(index, new_index)``."""
    clipSplitRequested = Signal()
    """Split the clip under the playhead there."""
    clipActivated = Signal(int)
    """The user picked a clip in the list: the playhead goes to it (which makes it the active clip)."""

    def __init__(self) -> None:
        super().__init__()
        self.clip_list = QListWidget()
        self.clip_list.setObjectName("clipList")
        self.clip_list.setMaximumHeight(CLIP_LIST_MAX_HEIGHT)
        self.clip_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.clip_list.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.add_clip_button = QPushButton("Adicionar vídeo…")
        self.add_clip_button.setToolTip("Acrescenta outro arquivo ao fim da sequência (mesmo fps)")
        self.split_clip_button = QPushButton("Dividir no playhead")
        self.split_clip_button.setToolTip("Divide o clipe no frame atual: cada metade passa a ter ajustes próprios")
        self.remove_clip_button = QPushButton("Remover clipe")
        self.move_clip_up_button = QPushButton("↑")
        self.move_clip_down_button = QPushButton("↓")
        for button in (
            self.split_clip_button,
            self.remove_clip_button,
            self.move_clip_up_button,
            self.move_clip_down_button,
        ):
            button.setObjectName("secondaryButton")
        self.sequence_label = QLabel()
        self.sequence_label.setObjectName("hintLabel")
        self.sequence_label.setWordWrap(True)
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

        self.add_clip_button.clicked.connect(self.clipAddRequested.emit)
        self.split_clip_button.clicked.connect(self.clipSplitRequested.emit)
        self.remove_clip_button.clicked.connect(self._emit_remove_clip)
        self.move_clip_up_button.clicked.connect(lambda: self._emit_move_clip(-1))
        self.move_clip_down_button.clicked.connect(lambda: self._emit_move_clip(1))
        self.clip_list.itemClicked.connect(lambda item: self.clipActivated.emit(self.clip_list.row(item)))
        self.clip_list.itemSelectionChanged.connect(self._update_clip_buttons)
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

        clip_buttons = QHBoxLayout()
        clip_buttons.setSpacing(10)
        clip_buttons.addWidget(self.add_clip_button)
        clip_buttons.addWidget(self.split_clip_button)
        clip_order = QHBoxLayout()
        clip_order.setSpacing(10)
        clip_order.addWidget(self.remove_clip_button)
        clip_order.addWidget(self.move_clip_up_button)
        clip_order.addWidget(self.move_clip_down_button)
        clip_order.addStretch(1)

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
            group_box("Sequência de clipes", clip_buttons, self.clip_list, clip_order, self.sequence_label, spacing=10),
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
        self.set_clips([], -1)

    # --- sequence ---------------------------------------------------------------------------
    def set_clips(self, items: list[str], active: int) -> None:
        """Replace the clip list; ``active`` (the clip under the playhead) is the selected row."""
        self.clip_list.blockSignals(True)
        try:
            self.clip_list.clear()
            for text in items:
                self.clip_list.addItem(QListWidgetItem(text))
            if 0 <= active < len(items):
                self.clip_list.setCurrentRow(active)
        finally:
            self.clip_list.blockSignals(False)
        self.sequence_label.setText(
            "Abra um vídeo e use “Adicionar vídeo…” para juntar outros arquivos na mesma linha do tempo. "
            "Os ajustes das outras abas valem para o clipe sob o playhead."
            if len(items) <= 1
            else "Os ajustes das abas valem para o clipe sob o playhead (destacado na linha do tempo). "
            "Clique em um clipe para ir até ele; a exportação junta todos na ordem da lista."
        )
        self._update_clip_buttons()

    def select_clip(self, index: int) -> None:
        """Mirror the active clip without emitting ``clipActivated``."""
        if index == self.selected_clip():
            return
        self.clip_list.blockSignals(True)
        try:
            if 0 <= index < self.clip_list.count():
                self.clip_list.setCurrentRow(index)
            else:
                self.clip_list.clearSelection()
                self.clip_list.setCurrentRow(-1)
        finally:
            self.clip_list.blockSignals(False)
        self._update_clip_buttons()

    def selected_clip(self) -> int:
        return self.clip_list.currentRow() if self.clip_list.selectedItems() else -1

    def _emit_remove_clip(self) -> None:
        index = self.selected_clip()
        if index >= 0:
            self.clipRemoveRequested.emit(index)

    def _emit_move_clip(self, step: int) -> None:
        index = self.selected_clip()
        if index >= 0 and 0 <= index + step < self.clip_list.count():
            self.clipMoveRequested.emit(index, index + step)

    def _update_clip_buttons(self) -> None:
        enabled = self.add_clip_button.isEnabled()
        count = self.clip_list.count()
        index = self.selected_clip()
        self.split_clip_button.setEnabled(enabled and count > 0)
        self.remove_clip_button.setEnabled(enabled and count > 1 and index >= 0)
        self.move_clip_up_button.setEnabled(enabled and index > 0)
        self.move_clip_down_button.setEnabled(enabled and 0 <= index < count - 1)

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
            self.add_clip_button,
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
        self._update_clip_buttons()
