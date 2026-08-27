"""A second window for the preview — the rendered frame plus transport controls — to put on another screen.

The monitor never renders anything itself: the main window hands it the same images its preview
receives (only while the monitor is visible) and the monitor's controls emit the same requests as
the transport bar. It shows the result by default ("Depois"), or the source, or both split down the
middle, scaled to fit a dark background — no overlays, no HUD. F11 (or a double click) makes it fill
its screen; in fullscreen the controls and the cursor hide after a moment without mouse movement.
"Outra tela" moves it to the next screen when there is more than one.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QGuiApplication,
    QImage,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QScreen,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from space_video_enhancer.telemetry.log import get_logger, log_event
from space_video_enhancer.ui.format import position_labels
from space_video_enhancer.ui.icons import glyph_icon
from space_video_enhancer.ui.theme import load_stylesheet
from space_video_enhancer.ui.widgets.compare_preview import CAPTION_BACKGROUND, draw_pill, fit_rect
from space_video_enhancer.ui.widgets.transport_bar import BUTTON_SIZE, ICON_SIZE, PLAY_BUTTON_SIZE, PLAY_ICON_SIZE

LOG = get_logger("ui")

BACKGROUND_COLOR = QColor("#0b0f19")
PLACEHOLDER_COLOR = QColor("#6b7280")
SPLIT_LINE_COLOR = QColor("#ffffff")
ICON_COLOR = QColor("#e5e7eb")
PLAY_ICON_COLOR = QColor("#ffffff")
PLACEHOLDER_TEXT = "Abra um vídeo na janela principal."
DEFAULT_SIZE = QSize(960, 600)
MIN_VIEW_SIZE = QSize(320, 180)
IDLE_HIDE_MS = 2500
"""In fullscreen, the controls and the cursor disappear this long after the mouse last moved."""
VIEW_MODES: tuple[tuple[str, str], ...] = (("Depois", "after"), ("Antes", "before"), ("Antes | Depois", "split"))
"""Combo label -> what the view draws: the result, the source, or both split down the middle."""


def next_screen(screens: Sequence[QScreen], current: QScreen | None) -> QScreen | None:
    """The screen after ``current`` in ``screens`` (wrapping), or ``None`` when there is nowhere else to go."""
    if len(screens) < 2:
        return None
    if current is None or current not in screens:
        return screens[0]
    return screens[(list(screens).index(current) + 1) % len(screens)]


class MonitorView(QWidget):
    """The frame, scaled to fit over a dark background: the result, the source, or a split of both."""

    doubleClicked = Signal()
    activity = Signal()
    """The mouse moved over the view (fullscreen uses it to bring the controls back)."""

    def __init__(self) -> None:
        super().__init__()
        self._before: QImage | None = None
        self._after: QImage | None = None
        self._mode = "after"
        self._smooth = True
        self._image_rect = QRectF()
        self.setMinimumSize(MIN_VIEW_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # so the transport shortcuts have somewhere to land
        self.setMouseTracking(True)

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in {value for _label, value in VIEW_MODES}:
            raise ValueError(f"unknown monitor mode: {mode}")
        self._mode = mode
        self.update()

    def has_images(self) -> bool:
        return self._before is not None and self._after is not None

    def set_images(self, before: QImage | None, after: QImage | None) -> None:
        self._before = before
        self._after = after
        self.update()

    def set_render_quality(self, smooth_transform_enabled: bool) -> None:
        self._smooth = smooth_transform_enabled
        self.update()

    def image_rect(self) -> QRectF:
        """Where the last paint put the frame (empty before the first paint or without images)."""
        return QRectF(self._image_rect)

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._smooth)
        painter.fillRect(self.rect(), BACKGROUND_COLOR)
        if self._before is None or self._after is None:
            self._image_rect = QRectF()
            painter.setPen(PLACEHOLDER_COLOR)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, PLACEHOLDER_TEXT)
            return
        shown = self._before if self._mode == "before" else self._after
        self._image_rect = fit_rect(shown.size(), QRectF(self.rect()))
        painter.drawImage(self._image_rect, shown)
        if self._mode == "split":
            self._draw_split(painter)

    def _draw_split(self, painter: QPainter) -> None:
        assert self._before is not None
        rect = self._image_rect
        split_x = rect.left() + rect.width() / 2.0
        painter.save()
        painter.setClipRect(QRectF(rect.left(), rect.top(), split_x - rect.left(), rect.height()))
        painter.drawImage(rect, self._before)
        painter.restore()
        painter.setPen(QPen(SPLIT_LINE_COLOR, 2))
        painter.drawLine(int(split_x), int(rect.top()), int(split_x), int(rect.bottom()))
        draw_pill(painter, QPointF(rect.left() + 10, rect.top() + 10), "Antes", CAPTION_BACKGROUND)
        after_width = painter.fontMetrics().horizontalAdvance("Depois") + 14
        draw_pill(painter, QPointF(rect.right() - 10 - after_width, rect.top() + 10), "Depois", CAPTION_BACKGROUND)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.activity.emit()
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class MonitorWindow(QWidget):
    """Top-level window (no parent, so it is never forced above the main window): view + controls."""

    firstRequested = Signal()
    previousRequested = Signal()
    playToggled = Signal()
    nextRequested = Signal()
    lastRequested = Signal()
    frameRequested = Signal(int)
    """The scrubber was dragged to this timeline frame."""
    closed = Signal()

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.setObjectName("monitorWindow")
        self.setWindowTitle("Monitor — Space Video Enhancer")
        self._fps = 30.0
        self._frame_count = 0
        self._placed = False
        self._controls_hidden = False

        self.view = MonitorView()
        self.view.doubleClicked.connect(self.toggle_fullscreen)
        self.view.activity.connect(self._on_activity)

        self._play_icon = glyph_icon("play", PLAY_ICON_COLOR)
        self._pause_icon = glyph_icon("pause", PLAY_ICON_COLOR)
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
            button.setObjectName("monitorButton")
            button.setFixedSize(BUTTON_SIZE)
            button.setIconSize(ICON_SIZE)
            button.setIcon(glyph_icon(kind, ICON_COLOR))
            button.setToolTip(tooltip)
        self.play_button.setObjectName("monitorPlayButton")
        self.play_button.setFixedSize(PLAY_BUTTON_SIZE)
        self.play_button.setIconSize(PLAY_ICON_SIZE)
        self.set_playing(False)

        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setObjectName("monitorScrubber")
        self.scrubber.setRange(0, 0)
        self.scrubber.setToolTip("Posição na linha do tempo")
        self.scrubber.valueChanged.connect(self.frameRequested.emit)  # silenced when the main window moves it

        self.timecode_label = QLabel()
        self.timecode_label.setObjectName("monitorTimecode")
        self.view_combo = QComboBox()
        self.view_combo.setObjectName("monitorCombo")
        for label, mode in VIEW_MODES:
            self.view_combo.addItem(label, mode)
        self.view_combo.setToolTip("O que o monitor mostra: o resultado, o original ou os dois lado a lado")
        self.view_combo.currentIndexChanged.connect(lambda _index: self.view.set_mode(self.view_combo.currentData()))

        self.fullscreen_action = QAction("Tela cheia", self)
        self.fullscreen_action.setCheckable(True)
        self.fullscreen_action.setShortcuts([QKeySequence("F11"), QKeySequence("Ctrl+Shift+F")])
        self.fullscreen_action.setToolTip("Ocupar a tela inteira (F11 ou clique duplo; Esc volta)")
        self.fullscreen_action.toggled.connect(self.set_fullscreen)
        self.addAction(self.fullscreen_action)
        self.fullscreen_button = QToolButton()
        self.fullscreen_button.setObjectName("monitorToggle")
        self.fullscreen_button.setDefaultAction(self.fullscreen_action)
        self.fullscreen_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.screen_button = QToolButton()
        self.screen_button.setObjectName("monitorToggle")
        self.screen_button.setText("Outra tela")
        self.screen_button.setToolTip("Levar o monitor para a próxima tela")
        self.screen_button.clicked.connect(self.move_to_next_screen)
        self.close_action = QAction("Fechar monitor", self)
        self.close_action.setShortcut(QKeySequence.StandardKey.Close)
        self.close_action.triggered.connect(self.close)
        self.addAction(self.close_action)

        for widget in (
            self.go_first_button,
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.go_last_button,
            self.scrubber,
            self.view_combo,
            self.fullscreen_button,
            self.screen_button,
        ):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Space and the arrows always drive the transport

        self.go_first_button.clicked.connect(self.firstRequested.emit)
        self.prev_frame_button.clicked.connect(self.previousRequested.emit)
        self.play_button.clicked.connect(self.playToggled.emit)
        self.next_frame_button.clicked.connect(self.nextRequested.emit)
        self.go_last_button.clicked.connect(self.lastRequested.emit)

        self.controls = QWidget()
        self.controls.setObjectName("monitorControls")
        self.controls.setMouseTracking(True)
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(14, 8, 14, 10)
        controls_layout.setSpacing(6)
        controls_layout.addWidget(self.scrubber)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.timecode_label)
        row.addStretch(1)
        for button in (
            self.go_first_button,
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.go_last_button,
        ):
            row.addWidget(button)
        row.addStretch(1)
        row.addWidget(self.view_combo)
        row.addWidget(self.screen_button)
        row.addWidget(self.fullscreen_button)
        controls_layout.addLayout(row)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.controls)

        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.setInterval(IDLE_HIDE_MS)
        self._idle_timer.timeout.connect(self._hide_controls)
        self.setMouseTracking(True)
        self.setStyleSheet(load_stylesheet())
        self.resize(DEFAULT_SIZE)
        self.set_frame_count(0, 30.0)
        self.set_controls_enabled(False)
        application = QGuiApplication.instance()
        if isinstance(application, QGuiApplication):
            application.screenAdded.connect(self._update_screen_button)
            application.screenRemoved.connect(self._update_screen_button)
        self._update_screen_button()

    # --- state pushed by the main window ---------------------------------------------------------
    def set_document_name(self, name: str | None) -> None:
        self.setWindowTitle(f"Monitor — {name}" if name else "Monitor — Space Video Enhancer")

    def set_frame_count(self, frame_count: int, fps: float) -> None:
        self._frame_count = max(0, frame_count)
        self._fps = fps if fps > 0 else 30.0
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, max(0, self._frame_count - 1))
        self.scrubber.blockSignals(False)
        self.set_position(min(self.scrubber.value(), max(0, self._frame_count - 1)))

    def set_position(self, frame_index: int) -> None:
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(frame_index)
        self.scrubber.blockSignals(False)
        timecode, frame = position_labels(frame_index, self._frame_count, self._fps)
        self.timecode_label.setText(f"{timecode}   ·   {frame}")

    def set_playing(self, playing: bool) -> None:
        self.play_button.setIcon(self._pause_icon if playing else self._play_icon)
        self.play_button.setToolTip("Pausar (Espaço)" if playing else "Reproduzir (Espaço)")

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.go_first_button,
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.go_last_button,
            self.scrubber,
        ):
            widget.setEnabled(enabled)

    def set_images(self, before: QImage | None, after: QImage | None) -> None:
        self.view.set_images(before, after)

    def set_render_quality(self, smooth_transform_enabled: bool) -> None:
        self.view.set_render_quality(smooth_transform_enabled)

    # --- screens and fullscreen ----------------------------------------------------------------
    def show_near(self, reference: QWidget) -> None:
        """First show: on another screen than ``reference``'s if there is one, else over it; later shows stay put."""
        if not self._placed:
            self._placed = True
            screens = QGuiApplication.screens()
            others = [screen for screen in screens if screen is not reference.screen()]
            if others:
                self.setGeometry(others[0].availableGeometry())
            else:
                self.move(reference.frameGeometry().topLeft() + reference.rect().center() - self.rect().center())
        self.show()
        self.raise_()
        self.activateWindow()

    def move_to_next_screen(self) -> None:
        target = next_screen(QGuiApplication.screens(), self.screen())
        if target is None:
            return
        fullscreen = self.isFullScreen()
        if fullscreen:
            self.showNormal()
        self.setGeometry(target.availableGeometry() if fullscreen else self._centered_on(target))
        if fullscreen:
            self.showFullScreen()
        log_event(LOG, "ui.monitor", state="screen", screen=target.name())

    def _centered_on(self, screen: QScreen) -> QRect:
        available = screen.availableGeometry()
        size = self.size().boundedTo(available.size())
        top_left = available.center() - QPoint(size.width() // 2, size.height() // 2)
        return QRect(top_left, size)

    def toggle_fullscreen(self) -> None:
        self.fullscreen_action.setChecked(not self.fullscreen_action.isChecked())

    def set_fullscreen(self, fullscreen: bool) -> None:
        if fullscreen == self.isFullScreen():
            self._sync_fullscreen_action()
            return
        if fullscreen:
            self.showFullScreen()
            self._idle_timer.start()
        else:
            self.showNormal()
            self._idle_timer.stop()
            self._show_controls()
        self._sync_fullscreen_action()
        log_event(LOG, "ui.monitor", state="fullscreen" if fullscreen else "windowed")

    def _sync_fullscreen_action(self) -> None:
        fullscreen = self.isFullScreen()
        if self.fullscreen_action.isChecked() != fullscreen:
            self.fullscreen_action.blockSignals(True)
            self.fullscreen_action.setChecked(fullscreen)
            self.fullscreen_action.blockSignals(False)
        self.fullscreen_action.setText("Sair da tela cheia" if fullscreen else "Tela cheia")

    def _update_screen_button(self, *_args: object) -> None:
        self.screen_button.setVisible(len(QGuiApplication.screens()) > 1)

    # --- idle controls in fullscreen -----------------------------------------------------------
    def _on_activity(self) -> None:
        if self._controls_hidden:
            self._show_controls()
        if self.isFullScreen():
            self._idle_timer.start()

    def _hide_controls(self) -> None:
        if not self.isFullScreen():
            return
        if self.controls.underMouse():
            self._idle_timer.start()  # the user is on the controls: leave them
            return
        self._controls_hidden = True
        self.controls.hide()
        self.view.setCursor(Qt.CursorShape.BlankCursor)

    def _show_controls(self) -> None:
        self._controls_hidden = False
        self.controls.show()
        self.view.unsetCursor()

    @property
    def controls_hidden(self) -> bool:
        return self._controls_hidden

    # --- events ----------------------------------------------------------------------------------
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._on_activity()
        super().mouseMoveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.fullscreen_action.setChecked(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._sync_fullscreen_action()  # the window manager may have changed it
            if not self.isFullScreen():
                self._idle_timer.stop()
                self._show_controls()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.view.setFocus()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.isFullScreen():
            self.fullscreen_action.setChecked(False)
        super().closeEvent(event)
        self.closed.emit()
