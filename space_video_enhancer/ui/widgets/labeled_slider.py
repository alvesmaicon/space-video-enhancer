from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QStyle, QToolButton, QVBoxLayout, QWidget


class LabeledSlider(QWidget):
    """Horizontal slider with a title, live value label and a "back to preset" button."""

    valueChanged = Signal(int)
    resetRequested = Signal()

    def __init__(self, label: str, minimum: int, maximum: int, value: int) -> None:
        super().__init__()
        self._title = QLabel(label)
        self._value_label = QLabel(str(value))
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._reset_button = QToolButton()
        self._reset_button.setAutoRaise(True)
        self._reset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self._reset_button.setToolTip("Voltar este controle ao preset ativo")
        self._reset_button.clicked.connect(self.resetRequested.emit)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self._handle_value_changed)

        header = QHBoxLayout()
        header.addWidget(self._title)
        header.addStretch(1)
        header.addWidget(self._value_label)
        header.addWidget(self._reset_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addWidget(self.slider)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:
        self.slider.setValue(value)

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self._reset_button.setEnabled(enabled)

    def _handle_value_changed(self, value: int) -> None:
        self._value_label.setText(str(value))
        self.valueChanged.emit(value)
