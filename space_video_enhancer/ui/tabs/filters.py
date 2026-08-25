"""Tab "Filtros": image-preset thumbnails and their intensity."""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QButtonGroup, QGridLayout, QLabel, QToolButton

from space_video_enhancer.pipeline.presets import IMAGE_FILTER_PRESETS, ORIGINAL_PRESET_ID, get_image_filter_preset
from space_video_enhancer.settings import ImageFilterSelection, clamp_unit
from space_video_enhancer.ui.format import format_percent
from space_video_enhancer.ui.tabs.base import TabPage, group_box
from space_video_enhancer.ui.widgets.labeled_slider import LabeledSlider

THUMBNAIL_SIZE = QSize(152, 86)
PRESET_BUTTON_MIN_SIZE = QSize(170, 140)
DEFAULT_INTENSITY = 100


class FiltersTab(TabPage):
    filterSelected = Signal(str)
    intensityChanged = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.active_filter_id = ORIGINAL_PRESET_ID
        self.intensity = DEFAULT_INTENSITY
        self._controls_enabled = False
        self._video_loaded = False

        self.buttons: dict[str, QToolButton] = {}
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        for index, preset in enumerate(IMAGE_FILTER_PRESETS):
            button = QToolButton()
            button.setObjectName("filterPresetButton")
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setIconSize(THUMBNAIL_SIZE)
            button.setMinimumSize(PRESET_BUTTON_MIN_SIZE)
            button.setText(preset.name)
            button.setToolTip(preset.description)
            button.clicked.connect(lambda _checked=False, filter_id=preset.filter_id: self._select(filter_id))
            self.button_group.addButton(button)
            self.buttons[preset.filter_id] = button
            grid.addWidget(button, index // 2, index % 2)

        self.intensity_slider = LabeledSlider(
            "Intensidade do filtro", 0, 100, DEFAULT_INTENSITY, display=format_percent
        )
        self.intensity_slider.valueChanged.connect(self._handle_intensity_changed)
        self.intensity_slider.resetRequested.connect(lambda: self.intensity_slider.setValue(DEFAULT_INTENSITY))
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)

        self.add_sections(group_box("Presets de imagem", grid, self.intensity_slider, self.description_label))
        self._sync_buttons()
        self._update_description()

    def selection(self) -> ImageFilterSelection:
        return ImageFilterSelection(self.active_filter_id, clamp_unit(self.intensity / 100.0))

    def set_thumbnails(self, images: Mapping[str, QImage]) -> None:
        for preset_id, image in images.items():
            button = self.buttons.get(preset_id)
            if button is not None:
                button.setIcon(QIcon(QPixmap.fromImage(image)))

    def clear_thumbnails(self) -> None:
        for button in self.buttons.values():
            button.setIcon(QIcon())

    def set_video_loaded(self, loaded: bool) -> None:
        self._video_loaded = loaded
        self._update_description()

    def set_controls_enabled(self, enabled: bool) -> None:
        self._controls_enabled = enabled
        for button in self.buttons.values():
            button.setEnabled(enabled)
        self._update_description()

    def _select(self, filter_id: str) -> None:
        if filter_id == self.active_filter_id:
            self._sync_buttons()
            return
        self.active_filter_id = filter_id
        self._sync_buttons()
        self._update_description()
        self.filterSelected.emit(filter_id)

    def _handle_intensity_changed(self, value: int) -> None:
        self.intensity = value
        self._update_description()
        self.intensityChanged.emit(value)

    def _sync_buttons(self) -> None:
        for filter_id, button in self.buttons.items():
            button.blockSignals(True)
            button.setChecked(filter_id == self.active_filter_id)
            button.blockSignals(False)

    def _update_description(self) -> None:
        preset = get_image_filter_preset(self.active_filter_id)
        self.description_label.setText(f"{preset.name}: {preset.description} Intensidade atual: {self.intensity}%.")
        adjustable = self._controls_enabled and self._video_loaded and preset.filter_id != ORIGINAL_PRESET_ID
        self.intensity_slider.setEnabled(adjustable)
