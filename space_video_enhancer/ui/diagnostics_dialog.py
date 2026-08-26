"""Ajuda -> Diagnóstico do sistema: the OpenCL/ffmpeg report as copyable text."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from space_video_enhancer.diagnostics import collect_diagnostics, diagnostics_text
from space_video_enhancer.telemetry.log import get_logger, log_event

LOG = get_logger("diagnostics")
DIALOG_SIZE = (760, 520)


class DiagnosticsDialog(QDialog):
    def __init__(self, ffmpeg_path: str | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ffmpeg_path = ffmpeg_path
        self.setWindowTitle("Diagnóstico do sistema")
        self.resize(*DIALOG_SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.intro_label = QLabel(
            "O que a máquina oferece para acelerar o app e o que falta quando o export diz "
            "“CPU (OpenCL indisponivel)”. Copie e cole este texto ao pedir ajuda."
        )
        self.intro_label.setWordWrap(True)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.text_edit.setFont(font)
        self.copy_button = QPushButton("Copiar")
        self.refresh_button = QPushButton("Atualizar")
        self.refresh_button.setObjectName("secondaryButton")
        self.close_button = QPushButton("Fechar")
        self.close_button.setObjectName("secondaryButton")
        self.feedback_label = QLabel()
        self.feedback_label.setObjectName("hintLabel")

        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.refresh_button.clicked.connect(self.refresh)
        self.close_button.clicked.connect(self.close)

        buttons = QHBoxLayout()
        buttons.addWidget(self.feedback_label, 1)
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.copy_button)
        buttons.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(self.intro_label)
        layout.addWidget(self.text_edit, 1)
        layout.addLayout(buttons)
        self.refresh()

    def text(self) -> str:
        return self.text_edit.toPlainText()

    def refresh(self) -> None:
        diagnostics = collect_diagnostics(self._ffmpeg_path)
        self.text_edit.setPlainText(diagnostics_text(diagnostics))
        self.feedback_label.setText("")
        log_event(
            LOG, "diagnostics.shown", opencl=diagnostics.opencl.status, encoders=diagnostics.ffmpeg.hardware_encoders
        )

    def copy_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self.text())
        self.feedback_label.setText("Copiado para a área de transferência.")
