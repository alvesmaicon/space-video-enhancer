from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider

START_MARKER_COLOR = QColor("#0f766e")
END_MARKER_COLOR = QColor("#b45309")
RANGE_HIGHLIGHT_COLOR = QColor(15, 118, 110, 70)


class FrameRangeSlider(QSlider):
    """Frame slider that highlights the selected clip range and jumps to where you click."""

    def __init__(self, orientation: Qt.Orientation) -> None:
        super().__init__(orientation)
        self._marker_start = 0
        self._marker_end = 0
        self._show_markers = False

    def set_markers(self, start: int, end: int, show_markers: bool = True) -> None:
        self._marker_start = start
        self._marker_end = end
        self._show_markers = show_markers
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
        if not self._show_markers or self.maximum() <= self.minimum():
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

        start_x = self._position_for_value(min(self._marker_start, self._marker_end), groove_rect, option)
        end_x = self._position_for_value(max(self._marker_start, self._marker_end), groove_rect, option)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        highlight_rect = groove_rect.adjusted(0, -2, 0, 2)
        highlight_rect.setLeft(min(start_x, end_x))
        highlight_rect.setRight(max(start_x, end_x))
        painter.fillRect(highlight_rect, RANGE_HIGHLIGHT_COLOR)

        painter.setPen(QPen(START_MARKER_COLOR, 2))
        painter.drawLine(start_x, groove_rect.top() - 6, start_x, groove_rect.bottom() + 6)
        painter.setPen(QPen(END_MARKER_COLOR, 2))
        painter.drawLine(end_x, groove_rect.top() - 6, end_x, groove_rect.bottom() + 6)

    def _position_for_value(self, value: int, groove_rect: QRect, option: QStyleOptionSlider) -> int:
        relative = QStyle.sliderPositionFromValue(
            self.minimum(),
            self.maximum(),
            value,
            max(1, groove_rect.width() - 1),
            option.upsideDown,
        )
        return groove_rect.x() + relative
