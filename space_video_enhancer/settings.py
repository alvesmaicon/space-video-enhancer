"""Domain settings shared by UI, preview and export.

Everything here is Qt-free and immutable. The UI builds a :class:`RenderSettings`
snapshot from its widgets and hands it to the pipeline; the pipeline never reads
widgets. That single seam is what makes the processing code testable, safe to run
in worker threads and easy to instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DenoiseMode(str, Enum):
    OFF = "off"
    MANUAL = "manual"
    AUTO = "auto"
    SAMPLE = "sample"


class StabilizationMode(str, Enum):
    OFF = "off"
    AUTO = "auto"
    SIMILARITY = "similarity"
    PERSPECTIVE = "perspective"
    CAMERA_LOCK = "camera_lock"
    HYBRID = "hybrid"

    @property
    def uses_focal_point(self) -> bool:
        return self in (StabilizationMode.CAMERA_LOCK, StabilizationMode.HYBRID)

    @property
    def uses_auto_motion(self) -> bool:
        return self in (StabilizationMode.AUTO, StabilizationMode.SIMILARITY, StabilizationMode.PERSPECTIVE)


class RotationMode(str, Enum):
    NONE = "none"
    CW_90 = "90"
    ROTATE_180 = "180"
    CW_270 = "270"

    @property
    def swaps_axes(self) -> bool:
        return self in (RotationMode.CW_90, RotationMode.CW_270)


class PreviewRenderMode(str, Enum):
    HIGH = "high"
    BALANCED = "balanced"
    FAST = "fast"


PixelRect = tuple[int, int, int, int]
"""(x, y, width, height) in integer pixels."""


@dataclass(frozen=True)
class NormalizedRect:
    """Rectangle in [0, 1] coordinates relative to a frame.

    Replaces ``QRectF`` inside the domain so the pipeline does not depend on Qt.
    """

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def is_valid(self) -> bool:
        return self.width > 0.0 and self.height > 0.0

    def to_pixels(self, frame_width: int, frame_height: int, *, even: bool = False) -> PixelRect | None:
        """Convert to a clamped pixel rectangle (at least 2x2), or ``None`` for an empty frame.

        With ``even=True`` the origin and size are aligned to even values, which
        video encoders (yuv420p) require.
        """
        if frame_width <= 0 or frame_height <= 0:
            return None

        x = round(self.x * frame_width)
        y = round(self.y * frame_height)
        width = round(self.width * frame_width)
        height = round(self.height * frame_height)

        x = max(0, min(x, frame_width - 2))
        y = max(0, min(y, frame_height - 2))
        width = max(2, min(width, frame_width - x))
        height = max(2, min(height, frame_height - y))

        if even:
            if width % 2 == 1 and width > 2:
                width -= 1
            if height % 2 == 1 and height > 2:
                height -= 1
            if x % 2 == 1 and x > 0:
                x -= 1
            if y % 2 == 1 and y > 0:
                y -= 1
            width = min(width, frame_width - x)
            height = min(height, frame_height - y)

        return (x, y, width, height)

    def covers_whole_frame(self, frame_width: int, frame_height: int, *, even: bool = False) -> bool:
        pixels = self.to_pixels(frame_width, frame_height, even=even)
        if pixels is None:
            return True
        x, y, width, height = pixels
        return x == 0 and y == 0 and width >= frame_width and height >= frame_height


DEFAULT_CROP_RECT = NormalizedRect(0.1, 0.1, 0.8, 0.8)
DEFAULT_DENOISE_SAMPLE_RECT = NormalizedRect(0.33, 0.33, 0.34, 0.34)
DEFAULT_STABILIZATION_RECT = NormalizedRect(0.4, 0.3, 0.2, 0.2)


@dataclass(frozen=True)
class FilterSettings:
    """Slider-level values (0-100 style) for the basic colour/detail chain."""

    denoise_mode: DenoiseMode = DenoiseMode.OFF
    denoise_strength: int = 20
    denoise_color_strength: int = 20
    denoise_template_window: int = 7
    denoise_search_window: int = 21
    sharpen: int = 25
    contrast: int = 50
    brightness: int = 50
    gamma: int = 50
    saturation: int = 50
    temperature: int = 50
    scale_factor: float = 1.0


DEFAULT_FILTER_SETTINGS = FilterSettings()

FILTER_PRESETS: tuple[tuple[str, FilterSettings | None], ...] = (
    ("Personalizado", None),
    (
        "Limpeza leve",
        FilterSettings(
            denoise_mode=DenoiseMode.MANUAL,
            denoise_strength=12,
            denoise_color_strength=10,
            sharpen=12,
            contrast=52,
        ),
    ),
    (
        "Limpeza clara",
        FilterSettings(
            denoise_mode=DenoiseMode.MANUAL,
            denoise_strength=25,
            denoise_color_strength=20,
            sharpen=15,
            gamma=35,
            saturation=40,
        ),
    ),
    (
        "Balanceado 1080p",
        FilterSettings(
            denoise_mode=DenoiseMode.MANUAL,
            denoise_strength=22,
            denoise_color_strength=18,
            sharpen=24,
            contrast=54,
            brightness=51,
            saturation=52,
            scale_factor=1.5,
        ),
    ),
    (
        "Recuperacao forte",
        FilterSettings(
            denoise_mode=DenoiseMode.MANUAL,
            denoise_strength=38,
            denoise_color_strength=34,
            denoise_search_window=27,
            sharpen=32,
            contrast=56,
            brightness=52,
            gamma=52,
            saturation=54,
            scale_factor=1.5,
        ),
    ),
    (
        "Mais detalhe 2x",
        FilterSettings(
            denoise_mode=DenoiseMode.MANUAL,
            denoise_strength=24,
            denoise_color_strength=18,
            sharpen=38,
            contrast=55,
            saturation=55,
            scale_factor=2.0,
        ),
    ),
)


@dataclass(frozen=True)
class ImageFilterSelection:
    preset_id: str = "original"
    intensity: float = 1.0
    """0.0-1.0 blend between the untouched frame and the preset result."""


@dataclass(frozen=True)
class GeometrySettings:
    rotation: RotationMode = RotationMode.NONE
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def output_size(self, width: int, height: int) -> tuple[int, int]:
        """Size of a ``width x height`` frame after rotation."""
        if self.rotation.swaps_axes:
            return (height, width)
        return (width, height)


@dataclass(frozen=True)
class CropSettings:
    enabled: bool = False
    rect: NormalizedRect = DEFAULT_CROP_RECT
    """Expressed in the *displayed* frame (after rotation/flip), exactly as drawn on the preview."""

    def pixel_rect(self, frame_width: int, frame_height: int) -> PixelRect | None:
        """Even-aligned crop rectangle, or ``None`` when disabled or covering the whole frame."""
        if not self.enabled:
            return None
        if self.rect.covers_whole_frame(frame_width, frame_height, even=True):
            return None
        return self.rect.to_pixels(frame_width, frame_height, even=True)


@dataclass(frozen=True)
class StabilizationSettings:
    mode: StabilizationMode = StabilizationMode.OFF
    intensity: float = 0.65
    temporal_smoothing: float = 0.35
    focus_rect: NormalizedRect = DEFAULT_STABILIZATION_RECT
    auto_crop: bool = True
    zoom_out: bool = False

    @property
    def enabled(self) -> bool:
        return self.mode is not StabilizationMode.OFF

    @property
    def requires_sequential_processing(self) -> bool:
        """True when frame N depends on frame N-1 (inter-frame motion or temporal smoothing)."""
        return self.enabled and (
            self.mode.uses_auto_motion or self.mode is StabilizationMode.HYBRID or self.temporal_smoothing > 0.0
        )


@dataclass(frozen=True)
class RenderSettings:
    """Complete, immutable snapshot of what the user configured."""

    filters: FilterSettings = DEFAULT_FILTER_SETTINGS
    image_filter: ImageFilterSelection = field(default_factory=ImageFilterSelection)
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    crop: CropSettings = field(default_factory=CropSettings)
    stabilization: StabilizationSettings = field(default_factory=StabilizationSettings)
    denoise_sample_rect: NormalizedRect = DEFAULT_DENOISE_SAMPLE_RECT


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))
