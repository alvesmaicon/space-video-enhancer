"""Tab "Cor": the tone/look controls and the white-balance/exposure/HSL controls on one page.

Two long-standing tabs — "Ajustes" (:class:`ColorTab`) and "HSL" (:class:`ColorGradeTab`) — were
merged so all colour work lives together (as a colour page does in editing suites). The two widgets
keep every method and signal; this page only adopts their group boxes and drives both at once.
"""

from __future__ import annotations

from space_video_enhancer.ui.tabs.base import TabPage
from space_video_enhancer.ui.tabs.color import ColorTab
from space_video_enhancer.ui.tabs.grade import ColorGradeTab


class ColorPage(TabPage):
    """Stacks the tone sliders above the white balance, exposure lock and per-hue-band controls."""

    def __init__(self, tone: ColorTab, grade: ColorGradeTab) -> None:
        super().__init__()
        self.tone = tone
        self.grade = grade
        self.add_sections(*tone.section_widgets(), *grade.section_widgets())

    def set_controls_enabled(self, enabled: bool) -> None:
        self.tone.set_controls_enabled(enabled)
        self.grade.set_controls_enabled(enabled)
