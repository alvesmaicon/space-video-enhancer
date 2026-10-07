from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QToolButton, QVBoxLayout, QWidget

from space_video_enhancer.ui import palette
from space_video_enhancer.ui.icons import glyph_icon
from space_video_enhancer.ui.palette import qcolor

VALUE_LABEL_MIN_WIDTH = 40
RESET_ICON_SIZE = 14


class LabeledSlider(QWidget):
    """Horizontal slider with a title, live value label and a "back to preset" button.

    ``display`` turns the raw value into the label text, e.g. ``+12`` for controls whose
    neutral point is the middle of the range or ``35%`` for intensities.
    """

    valueChanged = Signal(int)
    resetRequested = Signal()

    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        value: int,
        *,
        display: Callable[[int], str] | None = None,
    ) -> None:
        super().__init__()
        self._display: Callable[[int], str] = display or str
        self._title = QLabel(label)
        self._value_label = QLabel(self._display(value))
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._value_label.setMinimumWidth(VALUE_LABEL_MIN_WIDTH)
        self._reset_button = QToolButton()
        self._reset_button.setAutoRaise(True)
        self._reset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_button.setObjectName("resetButton")
        self._reset_button.setIcon(glyph_icon("reset", qcolor(palette.TEXT_MUTED), RESET_ICON_SIZE))
        self._reset_button.setIconSize(QSize(RESET_ICON_SIZE, RESET_ICON_SIZE))
        self._reset_button.setToolTip("Voltar este controle ao valor do preset ativo")
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

    def set_groove_gradient(self, colors: list[str]) -> None:
        """Paint the groove with a left-to-right gradient of ``colors`` (hex), or clear it when empty.

        Used by the HSL sliders so the groove shows where each direction takes the color; the
        handle is restyled too, otherwise Qt drops back to an unstyled square once the groove is set.
        """
        if not colors:
            self.slider.setStyleSheet("")
            return
        if len(colors) == 1:
            colors = [colors[0], colors[0]]
        last = len(colors) - 1
        stops = ", ".join(f"stop:{index / last:.3f} {color}" for index, color in enumerate(colors))
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 8px; border-radius: 4px;"
            f" background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops}); }}"
            f" QSlider::handle:horizontal {{ background: {palette.ON_ACCENT}; border: 2px solid {palette.VIEWER};"
            " width: 12px; margin: -5px 0; border-radius: 7px; }"
        )

    def value(self) -> int:
        return self.slider.value()

    def value_text(self) -> str:
        return self._value_label.text()

    def setValue(self, value: int) -> None:
        self.slider.setValue(value)

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self._reset_button.setEnabled(enabled)

    def _handle_value_changed(self, value: int) -> None:
        self._value_label.setText(self._display(value))
        self.valueChanged.emit(value)
