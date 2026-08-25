from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider

START_MARKER_COLOR = QColor("#0f766e")
END_MARKER_COLOR = QColor("#b45309")
RANGE_HIGHLIGHT_COLOR = QColor(15, 118, 110, 70)
CUT_COLOR = QColor(220, 38, 38, 120)
CUT_EDGE_COLOR = QColor("#b91c1c")
PENDING_CUT_COLOR = QColor("#dc2626")


class FrameRangeSlider(QSlider):
    """Frame slider that highlights the selected clip range and jumps to where you click."""

    def __init__(self, orientation: Qt.Orientation) -> None:
        super().__init__(orientation)
        self._marker_start = 0
        self._marker_end = 0
        self._show_markers = False
        self._cuts: tuple[tuple[int, int], ...] = ()
        self._pending_cut: int | None = None

    def set_markers(self, start: int, end: int, show_markers: bool = True) -> None:
        self._marker_start = start
        self._marker_end = end
        self._show_markers = show_markers
        self.update()

    def set_cuts(self, cuts: Sequence[tuple[int, int]]) -> None:
        """Frame ranges removed from the export, drawn as red bands over the groove."""
        self._cuts = tuple((int(start), int(end)) for start, end in cuts)
        self.update()

    def set_pending_cut(self, frame: int | None) -> None:
        """Start of a cut whose end has not been marked yet (dashed red line)."""
        self._pending_cut = frame
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Clicking the groove moves the handle there (instead of paging), then the drag starts."""
        if event.button() == Qt.MouseButton.LeftButton and self.maximum() > self.minimum():
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            style = self.style()
            handle = style.subControlRect(
                QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, self
            )
            if not handle.contains(event.position().toPoint()):
                groove = style.subControlRect(
                    QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderGroove, self
                )
                span = max(1, groove.width() - handle.width())
                position = int(event.position().x()) - groove.x() - (handle.width() // 2)
                value = QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(), position, span, option.upsideDown
                )
                self.setValue(value)
        super().mousePressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self.maximum() <= self.minimum():
            return

        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        if groove_rect.width() <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_cuts(painter, groove_rect, option)
        if not self._show_markers:
            return

        start_x = self._position_for_value(min(self._marker_start, self._marker_end), groove_rect, option)
        end_x = self._position_for_value(max(self._marker_start, self._marker_end), groove_rect, option)

        highlight_rect = groove_rect.adjusted(0, -2, 0, 2)
        highlight_rect.setLeft(min(start_x, end_x))
        highlight_rect.setRight(max(start_x, end_x))
        painter.fillRect(highlight_rect, RANGE_HIGHLIGHT_COLOR)

        painter.setPen(QPen(START_MARKER_COLOR, 2))
        painter.drawLine(start_x, groove_rect.top() - 6, start_x, groove_rect.bottom() + 6)
        painter.setPen(QPen(END_MARKER_COLOR, 2))
        painter.drawLine(end_x, groove_rect.top() - 6, end_x, groove_rect.bottom() + 6)

    def _paint_cuts(self, painter: QPainter, groove_rect: QRect, option: QStyleOptionSlider) -> None:
        band = groove_rect.adjusted(0, -3, 0, 3)
        for start, end in self._cuts:
            left = self._position_for_value(start, groove_rect, option)
            if end < self.maximum():
                right = self._position_for_value(end + 1, groove_rect, option)
            else:
                right = groove_rect.right()
            rect = QRect(band)
            rect.setLeft(left)
            rect.setRight(max(left + 1, right))
            painter.fillRect(rect, CUT_COLOR)
            painter.setPen(QPen(CUT_EDGE_COLOR, 1))
            painter.drawLine(rect.left(), rect.top(), rect.left(), rect.bottom())
            painter.drawLine(rect.right(), rect.top(), rect.right(), rect.bottom())
        if self._pending_cut is not None:
            x = self._position_for_value(self._pending_cut, groove_rect, option)
            pen = QPen(PENDING_CUT_COLOR, 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(x, groove_rect.top() - 7, x, groove_rect.bottom() + 7)

    def _position_for_value(self, value: int, groove_rect: QRect, option: QStyleOptionSlider) -> int:
        relative = QStyle.sliderPositionFromValue(
            self.minimum(),
            self.maximum(),
            value,
            max(1, groove_rect.width() - 1),
            option.upsideDown,
        )
        return groove_rect.x() + relative
