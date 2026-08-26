"""Timecode, the zoomable timeline with clip markers and cuts, and the transport buttons, below the preview.

The two labels above the timeline have fixed widths sized for the widest text they can show
for the open video, so changing digits never moves anything.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget

from space_video_enhancer.audio import AudioPeaks
from space_video_enhancer.ui.format import format_timecode
from space_video_enhancer.ui.icons import glyph_icon
from space_video_enhancer.ui.widgets.timeline import ClipBlock, TimelineWidget

PRIMARY_COLOR = QColor("#0f766e")
ON_PRIMARY_COLOR = QColor("#ffffff")
BUTTON_SIZE = QSize(42, 38)
PLAY_BUTTON_SIZE = QSize(54, 40)
ICON_SIZE = QSize(18, 18)
PLAY_ICON_SIZE = QSize(22, 22)
LABEL_SLACK_PX = 4
SECONDS_PER_HOUR = 3600


def widest_digits_width(metrics: QFontMetrics, template: str) -> int:
    """Width of ``template`` with every digit replaced by the widest digit of the font."""
    return max(metrics.horizontalAdvance(re.sub(r"\d", digit, template)) for digit in "0123456789")


def with_tabular_digits(font: QFont) -> QFont:
    """Ask for equal-width digits (OpenType ``tnum``) when the font supports it."""
    tabular = QFont(font)
    with contextlib.suppress(AttributeError, TypeError):
        tabular.setFeature(QFont.Tag("tnum"), 1)
    return tabular


class TransportBar(QWidget):
    frameChanged = Signal(int)
    firstRequested = Signal()
    previousRequested = Signal()
    playToggled = Signal()
    nextRequested = Signal()
    lastRequested = Signal()
    cutsEdited = Signal(object, bool)
    """``(cuts, finished)`` from dragging a cut's edge or body on the timeline."""
    clipRangeEdited = Signal(int, int, bool)
    """``(start, end, finished)`` from dragging a clip marker on the timeline."""
    cutSelected = Signal(int)
    cutRemoveRequested = Signal(int)
    cutToggleRequested = Signal()
    clipSplitRequested = Signal()
    clipRemoveRequested = Signal(int)
    soundToggled = Signal(bool)
    """The speaker button: ``True`` = play sound with the preview."""
    thumbnailsRequested = Signal(object, int, float)
    """Filmstrip cells missing from the timeline view: ``(frames, height_device_px, device_pixel_ratio)``."""

    def __init__(self) -> None:
        super().__init__()
        self._fps = 30.0
        self._frame_count = 0
        self.timecode_label = QLabel()
        self.timecode_label.setObjectName("timecodeLabel")
        self.timecode_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.frame_label = QLabel()
        self.frame_label.setObjectName("frameLabel")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.timeline = TimelineWidget()

        self.zoom_out_button = QToolButton()
        self.zoom_out_button.setText("−")
        self.zoom_out_button.setToolTip("Diminuir zoom da linha do tempo (Ctrl+roda do mouse ou pinça)")
        self.zoom_fit_button = QToolButton()
        self.zoom_fit_button.setText("Ajustar")
        self.zoom_fit_button.setToolTip("Mostrar o vídeo inteiro na linha do tempo")
        self.zoom_in_button = QToolButton()
        self.zoom_in_button.setText("+")
        self.zoom_in_button.setToolTip("Aumentar zoom da linha do tempo (Ctrl+roda do mouse ou pinça)")
        self.sound_button = QToolButton()
        self.sound_button.setText("Som")
        self.sound_button.setCheckable(True)
        self.sound_button.setChecked(True)
        self.sound_button.setToolTip("Tocar o áudio dos clipes durante a reprodução do preview")
        for zoom_button in (self.zoom_out_button, self.zoom_fit_button, self.zoom_in_button, self.sound_button):
            zoom_button.setObjectName("previewZoomButton")
            zoom_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sound_button.toggled.connect(self.soundToggled.emit)

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

        self.timeline.valueChanged.connect(self.frameChanged.emit)
        self.timeline.cutsEdited.connect(self.cutsEdited.emit)
        self.timeline.clipRangeEdited.connect(self.clipRangeEdited.emit)
        self.timeline.cutSelected.connect(self.cutSelected.emit)
        self.timeline.cutRemoveRequested.connect(self.cutRemoveRequested.emit)
        self.timeline.cutToggleRequested.connect(self.cutToggleRequested.emit)
        self.timeline.clipSplitRequested.connect(self.clipSplitRequested.emit)
        self.timeline.clipRemoveRequested.connect(self.clipRemoveRequested.emit)
        self.timeline.zoomChanged.connect(self._update_zoom_buttons)
        self.timeline.thumbnailsRequested.connect(self.thumbnailsRequested.emit)
        self.zoom_out_button.clicked.connect(self.timeline.zoom_out)
        self.zoom_fit_button.clicked.connect(self.timeline.zoom_to_fit)
        self.zoom_in_button.clicked.connect(self.timeline.zoom_in)
        self.go_first_button.clicked.connect(self.firstRequested.emit)
        self.prev_frame_button.clicked.connect(self.previousRequested.emit)
        self.play_button.clicked.connect(self.playToggled.emit)
        self.next_frame_button.clicked.connect(self.nextRequested.emit)
        self.go_last_button.clicked.connect(self.lastRequested.emit)

        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        header_row.addWidget(self.timecode_label)
        header_row.addStretch(1)
        header_row.addWidget(self.zoom_out_button)
        header_row.addWidget(self.zoom_fit_button)
        header_row.addWidget(self.zoom_in_button)
        header_row.addSpacing(12)
        header_row.addWidget(self.sound_button)
        header_row.addStretch(1)
        header_row.addWidget(self.frame_label)
        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        button_row.addStretch(1)
        for transport_button in (
            self.go_first_button,
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.go_last_button,
        ):
            button_row.addWidget(transport_button)
        button_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(header_row)
        layout.addWidget(self.timeline)
        layout.addLayout(button_row)
        self._fit_labels()
        self._update_zoom_buttons()
        self.set_position(0)

    def value(self) -> int:
        return self.timeline.value()

    def maximum(self) -> int:
        return self.timeline.maximum()

    def set_frame_count(self, frame_count: int, fps: float, aspect_ratio: float = 16 / 9) -> None:
        self._frame_count = max(0, frame_count)
        self._fps = fps if fps > 0 else 30.0
        self.timeline.set_media(self._frame_count, self._fps, aspect_ratio)
        self._fit_labels()
        self.set_position(self.timeline.value())

    def set_value(self, index: int) -> None:
        self.timeline.setValue(index)

    def set_value_silently(self, index: int) -> None:
        self.timeline.blockSignals(True)
        self.timeline.setValue(index)
        self.timeline.blockSignals(False)

    def set_position(self, frame_index: int) -> None:
        """Timecode of ``frame_index`` next to the clip duration, plus the frame counter."""
        hours = self._shows_hours()
        position = format_timecode(frame_index, self._fps, force_hours=hours)
        duration = format_timecode(self._frame_count, self._fps, force_hours=hours)
        self.timecode_label.setText(f"{position} / {duration}")
        shown = frame_index + 1 if self._frame_count else 0
        self.frame_label.setText(f"frame {shown} / {self._frame_count}")

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._fit_labels()

    def _shows_hours(self) -> bool:
        return self._frame_count / self._fps >= SECONDS_PER_HOUR

    def _fit_labels(self) -> None:
        """Fixed label widths for the widest text this video can produce, so nothing shifts as digits change."""
        for label in (self.timecode_label, self.frame_label):
            label.setFont(with_tabular_digits(label.font()))
        hours = self._shows_hours()
        duration = format_timecode(self._frame_count, self._fps, force_hours=hours)
        timecode_template = f"{duration} / {duration}"
        frame_template = f"frame {self._frame_count} / {self._frame_count}"
        metrics = self.timecode_label.fontMetrics()
        self.timecode_label.setFixedWidth(widest_digits_width(metrics, timecode_template) + LABEL_SLACK_PX)
        self.frame_label.setFixedWidth(
            widest_digits_width(self.frame_label.fontMetrics(), frame_template) + LABEL_SLACK_PX
        )

    def _update_zoom_buttons(self) -> None:
        loaded = self.timeline.frame_count > 1
        self.zoom_in_button.setEnabled(loaded and not self.timeline.is_max_zoom())
        self.zoom_out_button.setEnabled(loaded and not self.timeline.is_fit())
        self.zoom_fit_button.setEnabled(loaded and not self.timeline.is_fit())

    def set_sound_available(self, available: bool) -> None:
        """Without an output device the speaker button is off and disabled (the tooltip says why)."""
        self.sound_button.setEnabled(available)
        if not available:
            self.sound_button.setChecked(False)
            self.sound_button.setToolTip("Sem dispositivo de áudio: a reprodução segue sem som")
        else:
            self.sound_button.setToolTip("Tocar o áudio dos clipes durante a reprodução do preview")

    def sound_enabled(self) -> bool:
        return self.sound_button.isChecked()

    def set_playing(self, playing: bool) -> None:
        self.play_button.setIcon(self._pause_icon if playing else self._play_icon)
        self.play_button.setToolTip("Pausar (Espaço)" if playing else "Reproduzir (Espaço)")

    def set_markers(self, start: int, end: int, show: bool) -> None:
        self.timeline.set_clip_range(start, end, show)

    def set_cuts(self, cuts: Sequence[tuple[int, int]]) -> None:
        self.timeline.set_cuts(cuts)

    def set_pending_cut(self, frame: int | None) -> None:
        self.timeline.set_pending_cut(frame)

    def select_cut(self, index: int) -> None:
        self.timeline.select_cut(index)

    def set_clips(self, clips: Sequence[ClipBlock]) -> None:
        """The sequence layout; the labels follow the new total length."""
        self.timeline.set_clips(clips)
        self._frame_count = self.timeline.frame_count
        self._fit_labels()
        self.set_position(self.timeline.value())
        self._update_zoom_buttons()

    def set_active_clip(self, index: int) -> None:
        self.timeline.set_active_clip(index)

    def set_thumbnail(self, frame: int, image: QImage) -> None:
        self.timeline.set_thumbnail(frame, image)

    def set_waveform(self, path: Path, peaks: AudioPeaks | None) -> None:
        self.timeline.set_waveform(path, peaks)

    def set_audio_state(self, path: Path, state: str) -> None:
        self.timeline.set_audio_state(path, state)
