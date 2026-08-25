"""Timecode, frame slider with clip markers and the transport buttons, below the preview."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from space_video_enhancer.ui.format import format_timecode
from space_video_enhancer.ui.icons import glyph_icon
from space_video_enhancer.ui.widgets.frame_range_slider import FrameRangeSlider

PRIMARY_COLOR = QColor("#0f766e")
ON_PRIMARY_COLOR = QColor("#ffffff")
BUTTON_SIZE = QSize(42, 38)
PLAY_BUTTON_SIZE = QSize(54, 40)
ICON_SIZE = QSize(18, 18)
PLAY_ICON_SIZE = QSize(22, 22)


class TransportBar(QWidget):
    frameChanged = Signal(int)
    firstRequested = Signal()
    previousRequested = Signal()
    playToggled = Signal()
    nextRequested = Signal()
    lastRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._fps = 30.0
        self._frame_count = 0
        self.timecode_label = QLabel()
        self.timecode_label.setObjectName("timecodeLabel")
        self.frame_label = QLabel()
        self.frame_label.setObjectName("frameLabel")
        self.frame_slider = FrameRangeSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)

        self._play_icon = glyph_icon("play", ON_PRIMARY_COLOR)
        self._pause_icon = glyph_icon("pause", ON_PRIMARY_COLOR)
        self.go_first_button = QPushButton()
        self.prev_frame_button = QPushButton()
        self.play_button = QPushButton()
        self.next_frame_button = QPushButton()
        self.go_last_button = QPushButton()
        for button, kind, tooltip in (
            (self.go_first_button, "first", "Ir para o início (Home)"),
            (self.prev_frame_button, "previous", "Voltar um frame (←)"),
            (self.next_frame_button, "next", "Avançar um frame (→)"),
            (self.go_last_button, "last", "Ir para o final (End)"),
        ):
            button.setObjectName("transportButton")
            button.setFixedSize(BUTTON_SIZE)
            button.setIconSize(ICON_SIZE)
            button.setIcon(glyph_icon(kind, PRIMARY_COLOR))
            button.setToolTip(tooltip)
        self.play_button.setObjectName("transportPlayButton")
        self.play_button.setFixedSize(PLAY_BUTTON_SIZE)
        self.play_button.setIconSize(PLAY_ICON_SIZE)
        self.play_button.setIcon(self._play_icon)
        self.play_button.setToolTip("Reproduzir ou pausar (Espaço)")

        self.frame_slider.valueChanged.connect(self.frameChanged.emit)
        self.go_first_button.clicked.connect(self.firstRequested.emit)
        self.prev_frame_button.clicked.connect(self.previousRequested.emit)
        self.play_button.clicked.connect(self.playToggled.emit)
        self.next_frame_button.clicked.connect(self.nextRequested.emit)
        self.go_last_button.clicked.connect(self.lastRequested.emit)

        frame_row = QHBoxLayout()
        frame_row.setSpacing(12)
        frame_row.addWidget(self.timecode_label)
        frame_row.addWidget(self.frame_slider, 1)
        frame_row.addWidget(self.frame_label)
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
        self.set_position(0)

    def value(self) -> int:
        return self.frame_slider.value()

    def maximum(self) -> int:
        return self.frame_slider.maximum()

    def set_frame_count(self, frame_count: int, fps: float) -> None:
        self._frame_count = max(0, frame_count)
        self._fps = fps if fps > 0 else 30.0
        self.frame_slider.setRange(0, max(0, frame_count - 1))
        self.set_position(self.frame_slider.value())

    def set_value(self, index: int) -> None:
        self.frame_slider.setValue(index)

    def set_value_silently(self, index: int) -> None:
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)

    def set_position(self, frame_index: int) -> None:
        """Timecode of ``frame_index`` next to the clip duration, plus the frame counter."""
        duration = format_timecode(self._frame_count, self._fps)
        self.timecode_label.setText(f"{format_timecode(frame_index, self._fps)} / {duration}")
        shown = frame_index + 1 if self._frame_count else 0
        self.frame_label.setText(f"frame {shown} / {self._frame_count}")

    def set_playing(self, playing: bool) -> None:
        self.play_button.setIcon(self._pause_icon if playing else self._play_icon)
        self.play_button.setToolTip("Pausar (Espaço)" if playing else "Reproduzir (Espaço)")

    def set_markers(self, start: int, end: int, show: bool) -> None:
        self.frame_slider.set_markers(start, end, show)
