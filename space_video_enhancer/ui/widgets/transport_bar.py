"""Timecode, the zoomable timeline with clip markers and cuts, the timeline tools and the transport buttons.

Below the timeline, from left to right: the tools that act on the playhead and the clip under it
(mark in/out, remove a range, split, delete, move back/forward), then the transport buttons, and
at the right edge the playback speed and sound controls. One left-aligned strip keeps the row's
minimum width honest (no forced symmetry), so the window still fits a 1280 px screen. The two
labels above the timeline have fixed widths sized for the widest text they can show for the
open video, so changing digits never moves anything.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget

from space_video_enhancer.audio import AudioPeaks
from space_video_enhancer.ui.format import format_timecode
from space_video_enhancer.ui.icons import glyph_icon
from space_video_enhancer.ui.widgets.timeline import ClipBlock, TimelineWidget

PRIMARY_COLOR = QColor("#0f766e")
ON_PRIMARY_COLOR = QColor("#ffffff")
TOOL_COLOR = QColor("#374151")
BUTTON_SIZE = QSize(38, 34)
PLAY_BUTTON_SIZE = QSize(46, 36)
TOOL_BUTTON_SIZE = QSize(28, 26)
ICON_SIZE = QSize(18, 18)
PLAY_ICON_SIZE = QSize(22, 22)
TOOL_ICON_SIZE = QSize(16, 16)
LABEL_SLACK_PX = 4
SECONDS_PER_HOUR = 3600
PLAYBACK_SPEED_OPTIONS: list[tuple[str, float]] = [("1×", 1.0), ("½×", 0.5), ("¼×", 0.25)]
"""Playhead speed relative to real time; the sound is stretched to match (pitch kept)."""
REMOVE_RANGE_TOOLTIP = "Marcar o início de um trecho a remover no frame atual (X)"
REMOVE_RANGE_PENDING_TOOLTIP = "Terminar aqui o trecho a remover (X); o início já está marcado"
SOUND_TOOLTIP = "Tocar o áudio dos clipes durante a reprodução do preview"


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
    """The remove-range tool or the timeline menu: start a removed range here, or finish the pending one."""
    markInRequested = Signal()
    markOutRequested = Signal()
    """The bracket tools: the export range starts / ends at the current frame."""
    clipSplitRequested = Signal()
    clipRemoveRequested = Signal(int)
    clipDeleteRequested = Signal()
    """The trash tool: remove the clip under the playhead."""
    clipMoveRequested = Signal(int, int)
    clipMoveStepRequested = Signal(int)
    """The arrow tools: move the clip under the playhead one position back (``-1``) or forward (``+1``)."""
    soundToggled = Signal(bool)
    """The speaker button: ``True`` = play sound with the preview."""
    thumbnailsRequested = Signal(object, int, float)
    """Filmstrip cells missing from the timeline view: ``(frames, height_device_px, device_pixel_ratio)``."""

    def __init__(self) -> None:
        super().__init__()
        self._fps = 30.0
        self._frame_count = 0
        self._clip_count = 0
        self._active_clip = -1
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
        for zoom_button in (self.zoom_out_button, self.zoom_fit_button, self.zoom_in_button):
            zoom_button.setObjectName("previewZoomButton")
            zoom_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # --- timeline tools: act on the playhead and on the clip under it -------------------
        self.mark_in_button = self._tool_button("mark_in", "Início do trecho exportado no frame atual (I)")
        self.mark_out_button = self._tool_button("mark_out", "Fim do trecho exportado no frame atual (O)")
        self.remove_range_button = self._tool_button("remove_range", REMOVE_RANGE_TOOLTIP)
        self.remove_range_button.setCheckable(True)
        self.split_button = self._tool_button(
            "split", "Dividir o clipe no frame atual: cada parte passa a ter ajustes próprios (Ctrl+K)"
        )
        self.delete_clip_button = self._tool_button("trash", "Remover o clipe sob o playhead da sequência")
        self.move_left_button = self._tool_button("move_left", "Mover o clipe sob o playhead uma posição para trás")
        self.move_right_button = self._tool_button(
            "move_right", "Mover o clipe sob o playhead uma posição para a frente"
        )

        # --- playback: speed and sound ---------------------------------------------------------
        self.speed_combo = QComboBox()
        for label, speed in PLAYBACK_SPEED_OPTIONS:
            self.speed_combo.addItem(label, speed)
        self.speed_combo.setToolTip(
            "Velocidade da reprodução do preview; o som acompanha (a câmera lenta mantém o tom)"
        )
        self.speed_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sound_button = QToolButton()
        self.sound_button.setText("Som")
        self.sound_button.setCheckable(True)
        self.sound_button.setChecked(True)
        self.sound_button.setToolTip(SOUND_TOOLTIP)
        self.sound_button.setObjectName("previewZoomButton")
        self.sound_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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
        self.timeline.clipMoveRequested.connect(self.clipMoveRequested.emit)
        self.timeline.zoomChanged.connect(self._update_zoom_buttons)
        self.timeline.thumbnailsRequested.connect(self.thumbnailsRequested.emit)
        self.zoom_out_button.clicked.connect(self.timeline.zoom_out)
        self.zoom_fit_button.clicked.connect(self.timeline.zoom_to_fit)
        self.zoom_in_button.clicked.connect(self.timeline.zoom_in)
        self.mark_in_button.clicked.connect(self.markInRequested.emit)
        self.mark_out_button.clicked.connect(self.markOutRequested.emit)
        self.remove_range_button.clicked.connect(self.cutToggleRequested.emit)
        self.split_button.clicked.connect(self.clipSplitRequested.emit)
        self.delete_clip_button.clicked.connect(self.clipDeleteRequested.emit)
        self.move_left_button.clicked.connect(lambda: self.clipMoveStepRequested.emit(-1))
        self.move_right_button.clicked.connect(lambda: self.clipMoveStepRequested.emit(1))
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
        header_row.addStretch(1)
        header_row.addWidget(self.frame_label)

        self.tools = QWidget()
        self.tools.setObjectName("timelineTools")
        tools_row = QHBoxLayout(self.tools)
        tools_row.setContentsMargins(0, 0, 0, 0)
        tools_row.setSpacing(3)
        for tool in (self.mark_in_button, self.mark_out_button, self.remove_range_button):
            tools_row.addWidget(tool)
        tools_row.addSpacing(6)
        for tool in (self.split_button, self.delete_clip_button, self.move_left_button, self.move_right_button):
            tools_row.addWidget(tool)
        self.playback_controls = QWidget()
        playback_row = QHBoxLayout(self.playback_controls)
        playback_row.setContentsMargins(0, 0, 0, 0)
        playback_row.setSpacing(6)
        playback_row.addWidget(self.speed_combo)
        playback_row.addWidget(self.sound_button)

        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        button_row.addWidget(self.tools)
        button_row.addSpacing(10)
        for transport_button in (
            self.go_first_button,
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.go_last_button,
        ):
            button_row.addWidget(transport_button)
        button_row.addStretch(1)
        button_row.addWidget(self.playback_controls)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(header_row)
        layout.addWidget(self.timeline)
        layout.addLayout(button_row)
        self._fit_labels()
        self._update_zoom_buttons()
        self._update_tool_buttons()
        self.set_position(0)

    @staticmethod
    def _tool_button(kind: str, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setObjectName("timelineToolButton")
        button.setFixedSize(TOOL_BUTTON_SIZE)
        button.setIconSize(TOOL_ICON_SIZE)
        button.setIcon(glyph_icon(kind, TOOL_COLOR))
        button.setToolTip(tooltip)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return button

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
        self._update_tool_buttons()

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

    def _update_tool_buttons(self) -> None:
        """Markers and split need a video; delete and move need a clip under the playhead (and somewhere to go)."""
        loaded = self._frame_count > 0
        for tool in (self.mark_in_button, self.mark_out_button, self.remove_range_button, self.split_button):
            tool.setEnabled(loaded)
        active, count = self._active_clip, self._clip_count
        self.delete_clip_button.setEnabled(loaded and count > 1 and 0 <= active < count)
        self.move_left_button.setEnabled(loaded and 0 < active < count)
        self.move_right_button.setEnabled(loaded and 0 <= active < count - 1)

    def set_sound_available(self, available: bool) -> None:
        """Without an output device the speaker button is off and disabled (the tooltip says why)."""
        self.sound_button.setEnabled(available)
        if not available:
            self.sound_button.setChecked(False)
            self.sound_button.setToolTip("Sem dispositivo de áudio: a reprodução segue sem som")
        else:
            self.sound_button.setToolTip(SOUND_TOOLTIP)

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
        """A removed range has its start marked: the timeline shows it and the tool stays pressed until the end."""
        self.timeline.set_pending_cut(frame)
        pending = frame is not None
        self.remove_range_button.setChecked(pending)
        self.remove_range_button.setToolTip(REMOVE_RANGE_PENDING_TOOLTIP if pending else REMOVE_RANGE_TOOLTIP)

    def pending_cut(self) -> bool:
        return self.remove_range_button.isChecked()

    def select_cut(self, index: int) -> None:
        self.timeline.select_cut(index)

    def set_clips(self, clips: Sequence[ClipBlock]) -> None:
        """The sequence layout; the labels follow the new total length."""
        self.timeline.set_clips(clips)
        self._frame_count = self.timeline.frame_count
        self._clip_count = len(clips)
        self._fit_labels()
        self.set_position(self.timeline.value())
        self._update_zoom_buttons()
        self._update_tool_buttons()

    def set_active_clip(self, index: int) -> None:
        self.timeline.set_active_clip(index)
        self._active_clip = index
        self._update_tool_buttons()

    def set_thumbnail(self, frame: int, image: QImage) -> None:
        self.timeline.set_thumbnail(frame, image)

    def set_waveform(self, path: Path, peaks: AudioPeaks | None) -> None:
        self.timeline.set_waveform(path, peaks)

    def set_audio_state(self, path: Path, state: str) -> None:
        self.timeline.set_audio_state(path, state)
