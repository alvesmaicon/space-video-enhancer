"""Shared building blocks for the settings tabs (right-hand panel)."""

from __future__ import annotations

from typing import TypeVar, Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from space_video_enhancer.settings import (
    AnonymizeMode,
    DeinterlaceMode,
    DenoiseMode,
    PreviewRenderMode,
    ReadoutAxis,
    RetimingMode,
    RotationMode,
    StabilizationMode,
)
from space_video_enhancer.ui.widgets.compare_preview import CompareMode

E = TypeVar(
    "E",
    DenoiseMode,
    StabilizationMode,
    RotationMode,
    PreviewRenderMode,
    DeinterlaceMode,
    CompareMode,
    ReadoutAxis,
    RetimingMode,
    AnonymizeMode,
)
Section = Union[QWidget, QLayout]

CONTENT_MARGIN = 14
SECTION_SPACING = 14


class TabPage(QScrollArea):
    """One settings tab: group boxes stacked vertically, scrolling only vertically.

    Subclasses create their widgets, then call :meth:`add_sections`. They also implement
    :meth:`set_controls_enabled` so the window can freeze everything during an export.
    """

    def __init__(self) -> None:
        super().__init__()
        self.content = QWidget()
        self.content.setObjectName("tabContent")
        self._layout = QVBoxLayout(self.content)
        self._layout.setContentsMargins(CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN)
        self._layout.setSpacing(SECTION_SPACING)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # arrow keys belong to the transport, not to scrolling
        self.setWidget(self.content)

    def add_sections(self, *sections: QWidget) -> None:
        for section in sections:
            self._layout.addWidget(section)
        self._layout.addStretch(1)

    def content_min_width(self) -> int:
        """Narrowest width that shows the content without clipping (horizontal scrolling is off)."""
        return self.content.minimumSizeHint().width() + self.verticalScrollBar().sizeHint().width()

    def set_controls_enabled(self, enabled: bool) -> None:
        raise NotImplementedError


def group_box(title: str, *items: Section, spacing: int = SECTION_SPACING) -> QGroupBox:
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    layout.setSpacing(spacing)
    for item in items:
        if isinstance(item, QLayout):
            layout.addLayout(item)
        else:
            layout.addWidget(item)
    return box


def form_rows(*rows: tuple[str, QWidget]) -> QFormLayout:
    form = QFormLayout()
    for label, widget in rows:
        form.addRow(label, widget)
    return form


def enum_combo(options: list[tuple[str, E]]) -> QComboBox:
    combo = QComboBox()
    for label, member in options:
        combo.addItem(label, member.value)
    return combo


def combo_enum(combo: QComboBox, enum_type: type[E], default: E) -> E:
    try:
        return enum_type(combo.currentData())
    except ValueError:
        return default


def select_combo_data(combo: QComboBox, data: object) -> None:
    index = combo.findData(data)
    if index >= 0:
        combo.setCurrentIndex(index)


def set_checked_silently(button: QAbstractButton, checked: bool) -> None:
    if button.isChecked() != checked:
        button.blockSignals(True)
        button.setChecked(checked)
        button.blockSignals(False)
