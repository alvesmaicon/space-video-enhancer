"""Tab "Geometria": rotation, mirroring, perspective correction and the crop mode."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton

from space_video_enhancer.settings import (
    CropSettings,
    GeometrySettings,
    PerspectiveFit,
    PerspectiveSettings,
    RotationMode,
)
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
PERSPECTIVE_FIT_OPTIONS: list[tuple[str, PerspectiveFit]] = [
    ("Proporção real", PerspectiveFit.REAL),
    ("Esticar para 16:9", PerspectiveFit.STRETCH_16_9),
    ("Esticar para 9:16", PerspectiveFit.STRETCH_9_16),
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
    perspectiveMarkToggled = Signal(bool)
    """"Marcar pontos" was pressed (``True``: start marking) or pressed again (``False``: leave without applying)."""
    perspectiveConfirmRequested = Signal()
    perspectiveResetRequested = Signal()
    perspectiveFitChanged = Signal()

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

        self.perspective_fit_combo = enum_combo(PERSPECTIVE_FIT_OPTIONS)
        self.perspective_fit_combo.setToolTip(
            "Proporção real: o plano fica com a proporção estimada e o resto do quadro acompanha (o crop "
            "recorta as bordas).\nEsticar: os 4 pontos viram os cantos do vídeo nesse formato, sem bordas."
        )
        self.perspective_mark_button = QPushButton("Marcar pontos")
        self.perspective_mark_button.setCheckable(True)
        self.perspective_mark_button.setObjectName("secondaryButton")  # a toggle; "Confirmar" is the action
        self.perspective_mark_button.setToolTip(
            "Mostra o frame sem correção com 4 pontos para arrastar até os cantos do plano.\n"
            "Clique de novo para sair sem aplicar."
        )
        self.perspective_confirm_button = QPushButton("Confirmar")
        self.perspective_reset_button = QPushButton("Redefinir perspectiva")
        self.perspective_reset_button.setObjectName("secondaryButton")
        self.perspective_info_label = QLabel("Perspectiva: sem correção")
        self.perspective_info_label.setWordWrap(True)
        self._controls_enabled = True
        self._perspective_marking = False
        self._perspective_confirmed = False
        self._perspective_valid = True

        self.rotation_combo.currentIndexChanged.connect(lambda _index: self.geometryChanged.emit())
        self.flip_horizontal_checkbox.toggled.connect(lambda _checked: self.geometryChanged.emit())
        self.flip_vertical_checkbox.toggled.connect(lambda _checked: self.geometryChanged.emit())
        self.crop_mode_combo.currentIndexChanged.connect(lambda _index: self.cropModeChanged.emit())
        self.crop_reset_button.clicked.connect(self.cropResetRequested.emit)
        self.perspective_fit_combo.currentIndexChanged.connect(lambda _index: self.perspectiveFitChanged.emit())
        self.perspective_mark_button.toggled.connect(self.perspectiveMarkToggled.emit)
        self.perspective_confirm_button.clicked.connect(self.perspectiveConfirmRequested.emit)
        self.perspective_reset_button.clicked.connect(self.perspectiveResetRequested.emit)

        perspective_buttons = QHBoxLayout()
        perspective_buttons.addWidget(self.perspective_mark_button)
        perspective_buttons.addWidget(self.perspective_confirm_button)
        perspective_buttons.addStretch(1)
        perspective_reset_row = QHBoxLayout()
        perspective_reset_row.addWidget(self.perspective_reset_button)
        perspective_reset_row.addStretch(1)

        self.add_sections(
            group_box(
                "Orientação",
                form_rows(("Rotação", self.rotation_combo)),
                self.flip_horizontal_checkbox,
                self.flip_vertical_checkbox,
                spacing=12,
            ),
            group_box(
                "Perspectiva",
                form_rows(("Formato", self.perspective_fit_combo)),
                perspective_buttons,
                perspective_reset_row,
                self.perspective_info_label,
            ),
            group_box(
                "Enquadramento",
                form_rows(("Crop", self.crop_mode_combo)),
                self.crop_reset_button,
                self.crop_info_label,
            ),
        )
        self._sync_perspective_buttons()

    def geometry_settings(self) -> GeometrySettings:
        return GeometrySettings(
            rotation=combo_enum(self.rotation_combo, RotationMode, RotationMode.NONE),
            flip_horizontal=self.flip_horizontal_checkbox.isChecked(),
            flip_vertical=self.flip_vertical_checkbox.isChecked(),
        )

    def perspective_fit(self) -> PerspectiveFit:
        return combo_enum(self.perspective_fit_combo, PerspectiveFit, PerspectiveFit.REAL)

    def apply(
        self, geometry: GeometrySettings, crop: CropSettings, perspective: PerspectiveSettings | None = None
    ) -> None:
        """Show another clip's orientation, perspective format and crop mode without emitting any signal."""
        widgets = (
            self.rotation_combo,
            self.flip_horizontal_checkbox,
            self.flip_vertical_checkbox,
            self.crop_mode_combo,
            self.perspective_fit_combo,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            select_combo_data(self.rotation_combo, geometry.rotation.value)
            set_checked_silently(self.flip_horizontal_checkbox, geometry.flip_horizontal)
            set_checked_silently(self.flip_vertical_checkbox, geometry.flip_vertical)
            self.crop_mode_combo.setCurrentIndex(self._crop_mode_index(crop))
            if perspective is not None:
                select_combo_data(self.perspective_fit_combo, perspective.fit.value)
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def set_perspective_state(self, *, marking: bool, confirmed: bool, valid: bool) -> None:
        """Buttons follow the window: marking (points on screen), a confirmed correction, a usable quadrilateral."""
        self._perspective_marking = marking
        self._perspective_confirmed = confirmed
        self._perspective_valid = valid
        set_checked_silently(self.perspective_mark_button, marking)
        self._sync_perspective_buttons()

    def set_perspective_info(self, text: str) -> None:
        self.perspective_info_label.setText(text)

    def _sync_perspective_buttons(self) -> None:
        enabled = self._controls_enabled
        self.perspective_mark_button.setEnabled(enabled)
        self.perspective_confirm_button.setEnabled(enabled and self._perspective_marking and self._perspective_valid)
        self.perspective_reset_button.setEnabled(enabled and (self._perspective_marking or self._perspective_confirmed))
        self.perspective_fit_combo.setEnabled(enabled)

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
        self._controls_enabled = enabled
        for widget in (
            self.rotation_combo,
            self.flip_horizontal_checkbox,
            self.flip_vertical_checkbox,
            self.crop_mode_combo,
            self.crop_reset_button,
        ):
            widget.setEnabled(enabled)
        self._sync_perspective_buttons()
