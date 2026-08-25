"""Persistent status strip: message, render indicator, export progress/cancel and video info."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QWidget

RENDER_INDICATOR_DELAY_MS = 150
"""Only show "renderizando" when a preview takes longer than this, so fast renders do not flicker."""
PROGRESS_BAR_WIDTH = 200


class StatusPanel(QWidget):
    cancelExportRequested = Signal()
    revealExportRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("statusPanel")
        self.message_label = QLabel()
        self.message_label.setObjectName("statusMessage")
        self.message_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.render_label = QLabel("● renderizando…")
        self.render_label.setObjectName("renderIndicator")
        self.render_label.hide()
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("statusProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(PROGRESS_BAR_WIDTH)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.hide()
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.setObjectName("statusCancel")
        self.cancel_button.hide()
        self.reveal_button = QPushButton("Abrir pasta")
        self.reveal_button.setObjectName("statusReveal")
        self.reveal_button.setToolTip("Abrir a pasta do vídeo exportado")
        self.reveal_button.hide()
        self.video_label = QLabel("Nenhum vídeo aberto")
        self.video_label.setObjectName("videoInfo")

        self.cancel_button.clicked.connect(self.cancelExportRequested.emit)
        self.reveal_button.clicked.connect(self.revealExportRequested.emit)

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(RENDER_INDICATOR_DELAY_MS)
        self._render_timer.timeout.connect(self.render_label.show)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(12)
        layout.addWidget(self.message_label, 1)
        layout.addWidget(self.render_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.reveal_button)
        layout.addWidget(self.video_label)

    def show_message(self, text: str) -> None:
        self.message_label.setText(text)
        self.message_label.setToolTip(text)

    def message(self) -> str:
        return self.message_label.text()

    def set_video_info(self, text: str) -> None:
        self.video_label.setText(text)

    def set_render_busy(self, busy: bool) -> None:
        if busy:
            if not self.render_label.isVisible() and not self._render_timer.isActive():
                self._render_timer.start()
            return
        self._render_timer.stop()
        self.render_label.hide()

    @property
    def exporting(self) -> bool:
        return self.cancel_button.isVisible()

    def begin_export(self, message: str) -> None:
        self.reveal_button.hide()
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.show_message(message)

    def set_export_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(max(0, min(100, percent)))
        self.show_message(message)

    def set_cancel_enabled(self, enabled: bool) -> None:
        self.cancel_button.setEnabled(enabled)

    def end_export(self, message: str, *, success: bool) -> None:
        self.cancel_button.hide()
        if success:
            self.progress_bar.setValue(100)
        else:
            self.progress_bar.hide()
        self.reveal_button.setVisible(success)
        self.show_message(message)
