"""Tab "Geometria": rotation, mirroring and the crop mode."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton

from space_video_enhancer.settings import CropSettings, GeometrySettings, RotationMode
from space_video_enhancer.ui.tabs.base import (
    TabPage,
    combo_enum,
    enum_combo,
    form_rows,
    group_box,
    select_combo_data,
    set_checked_silently,
)

ROTATION_OPTIONS: list[tuple[str, RotationMode]] = [
    ("Sem rotação", RotationMode.NONE),
    ("90° no sentido horário", RotationMode.CW_90),
    ("180°", RotationMode.ROTATE_180),
    ("90° no sentido anti-horário", RotationMode.CW_270),
]
FREE_CROP = "free"
CROP_OPTIONS: list[tuple[str, object]] = [
    ("Desligado", None),
    ("Livre (arraste no preview)", FREE_CROP),
    ("16:9", 16 / 9),
    ("4:3", 4 / 3),
    ("1:1", 1.0),
    ("9:16", 9 / 16),
]


class GeometryTab(TabPage):
    geometryChanged = Signal()
    cropModeChanged = Signal()
    cropResetRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.rotation_combo = enum_combo(ROTATION_OPTIONS)
        self.flip_horizontal_checkbox = QCheckBox("Espelhar horizontalmente")
        self.flip_vertical_checkbox = QCheckBox("Espelhar verticalmente")
        self.crop_mode_combo = QComboBox()
        for label, mode in CROP_OPTIONS:
            self.crop_mode_combo.addItem(label, mode)
        self.crop_reset_button = QPushButton("Redefinir crop")
        self.crop_reset_button.setObjectName("secondaryButton")
        self.crop_info_label = QLabel("Crop: desativado")
        self.crop_info_label.setWordWrap(True)

        self.rotation_combo.currentIndexChanged.connect(lambda _index: self.geometryChanged.emit())
        self.flip_horizontal_checkbox.toggled.connect(lambda _checked: self.geometryChanged.emit())
        self.flip_vertical_checkbox.toggled.connect(lambda _checked: self.geometryChanged.emit())
        self.crop_mode_combo.currentIndexChanged.connect(lambda _index: self.cropModeChanged.emit())
        self.crop_reset_button.clicked.connect(self.cropResetRequested.emit)

        self.add_sections(
            group_box(
                "Orientação",
                form_rows(("Rotação", self.rotation_combo)),
                self.flip_horizontal_checkbox,
                self.flip_vertical_checkbox,
                spacing=12,
            ),
            group_box(
                "Enquadramento",
                form_rows(("Crop", self.crop_mode_combo)),
                self.crop_reset_button,
                self.crop_info_label,
            ),
        )

    def geometry_settings(self) -> GeometrySettings:
        return GeometrySettings(
            rotation=combo_enum(self.rotation_combo, RotationMode, RotationMode.NONE),
            flip_horizontal=self.flip_horizontal_checkbox.isChecked(),
            flip_vertical=self.flip_vertical_checkbox.isChecked(),
        )

    def apply(self, geometry: GeometrySettings, crop: CropSettings) -> None:
        """Show another clip's orientation and crop mode without emitting any signal."""
        widgets = (
            self.rotation_combo,
            self.flip_horizontal_checkbox,
            self.flip_vertical_checkbox,
            self.crop_mode_combo,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            select_combo_data(self.rotation_combo, geometry.rotation.value)
            set_checked_silently(self.flip_horizontal_checkbox, geometry.flip_horizontal)
            set_checked_silently(self.flip_vertical_checkbox, geometry.flip_vertical)
            self.crop_mode_combo.setCurrentIndex(self._crop_mode_index(crop))
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def _crop_mode_index(self, crop: CropSettings) -> int:
        if not crop.enabled:
            return 0
        if crop.aspect_ratio is None:
            return self.crop_mode_combo.findData(FREE_CROP)
        for index in range(self.crop_mode_combo.count()):
            data = self.crop_mode_combo.itemData(index)
            if isinstance(data, float) and abs(data - crop.aspect_ratio) < 1e-6:
                return index
        return self.crop_mode_combo.findData(FREE_CROP)

    def crop_mode_data(self) -> object:
        return self.crop_mode_combo.currentData()

    def crop_enabled(self) -> bool:
        return self.crop_mode_data() is not None

    def crop_is_free(self) -> bool:
        return self.crop_mode_data() == FREE_CROP

    def crop_aspect_ratio(self) -> float | None:
        """Fixed aspect ratio of the chosen crop preset; ``None`` for free crop or no crop."""
        data = self.crop_mode_data()
        if data is None or data == FREE_CROP:
            return None
        return float(data)  # type: ignore[arg-type]

    def crop_mode_text(self) -> str:
        return self.crop_mode_combo.currentText()

    def set_crop_info(self, text: str) -> None:
        self.crop_info_label.setText(text)

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.rotation_combo,
            self.flip_horizontal_checkbox,
            self.flip_vertical_checkbox,
            self.crop_mode_combo,
            self.crop_reset_button,
        ):
            widget.setEnabled(enabled)
