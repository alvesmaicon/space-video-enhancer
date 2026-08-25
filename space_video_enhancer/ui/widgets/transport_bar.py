"""Frame slider with clip markers plus the transport buttons, below the preview."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from space_video_enhancer.ui.widgets.frame_range_slider import FrameRangeSlider

BUTTON_SIZE = (42, 38)


class TransportBar(QWidget):
    frameChanged = Signal(int)
    firstRequested = Signal()
    previousRequested = Signal()
    playToggled = Signal()
    nextRequested = Signal()
    lastRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.frame_label = QLabel("Frame 0 / 0")
        self.frame_slider = FrameRangeSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.go_first_button = QPushButton("⏮")
        self.prev_frame_button = QPushButton("◀")
        self.play_button = QPushButton("⏵")
        self.next_frame_button = QPushButton("▶")
        self.go_last_button = QPushButton("⏭")
        for button, tooltip in (
            (self.go_first_button, "Ir para o início (Home)"),
            (self.prev_frame_button, "Voltar um frame (←)"),
            (self.play_button, "Reproduzir ou pausar (Espaço)"),
            (self.next_frame_button, "Avançar um frame (→)"),
            (self.go_last_button, "Ir para o final (End)"),
        ):
            button.setObjectName("transportButton")
            button.setFixedSize(*BUTTON_SIZE)
            button.setToolTip(tooltip)

        self.frame_slider.valueChanged.connect(self.frameChanged.emit)
        self.go_first_button.clicked.connect(self.firstRequested.emit)
        self.prev_frame_button.clicked.connect(self.previousRequested.emit)
        self.play_button.clicked.connect(self.playToggled.emit)
        self.next_frame_button.clicked.connect(self.nextRequested.emit)
        self.go_last_button.clicked.connect(self.lastRequested.emit)

        frame_row = QHBoxLayout()
        frame_row.addWidget(self.frame_label)
        frame_row.addWidget(self.frame_slider, 1)
        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        button_row.addStretch(1)
        for button in (
            self.go_first_button,
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.go_last_button,
        ):
            button_row.addWidget(button)
        button_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(frame_row)
        layout.addLayout(button_row)

    def value(self) -> int:
        return self.frame_slider.value()

    def maximum(self) -> int:
        return self.frame_slider.maximum()

    def set_frame_count(self, frame_count: int) -> None:
        self.frame_slider.setRange(0, max(0, frame_count - 1))

    def set_value(self, index: int) -> None:
        self.frame_slider.setValue(index)

    def set_value_silently(self, index: int) -> None:
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)

    def set_frame_text(self, text: str) -> None:
        self.frame_label.setText(text)

    def set_playing(self, playing: bool) -> None:
        self.play_button.setText("⏸" if playing else "⏵")

    def set_markers(self, start: int, end: int, show: bool) -> None:
        self.frame_slider.set_markers(start, end, show)
