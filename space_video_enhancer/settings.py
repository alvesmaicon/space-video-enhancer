"""Domain settings shared by UI, preview and export.

Everything here is Qt-free and immutable. The UI builds a :class:`RenderSettings`
snapshot from its widgets and hands it to the pipeline; the pipeline never reads
widgets. That single seam is what makes the processing code testable, safe to run
in worker threads and easy to instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    MANUAL = "manual"
    """Hand-set keyframes (position/rotation/zoom) interpolated between; no estimation, no warp surprises."""
    MESH = "mesh"
    """One camera path per region of the frame (MeshFlow): the clip is analysed once in the background
    and every render warps by that clip-wide field — parallax and region-dependent distortion included."""
    OBJECT = "object"
    """Hold a target still by 1-4 tracked regions (even in opposite corners): a similarity fitted over
    the live regions' correspondences, so a lost region does not freeze anything and two spread regions
    give the target's rotation and scale. All lost = dead reckoning on the global motion (see
    :mod:`pipeline.stabilization.lock`). Supersedes CAMERA_LOCK and HYBRID."""

    @property
    def uses_focal_point(self) -> bool:
        """The legacy single-region lock modes (one green rectangle, ``focus_rect``)."""
        return self in (StabilizationMode.CAMERA_LOCK, StabilizationMode.HYBRID)

    @property
    def uses_focus_regions(self) -> bool:
        """The multi-region object lock (numbered rectangles, ``focus_regions``)."""
        return self is StabilizationMode.OBJECT

    @property
    def needs_reference_frame(self) -> bool:
        """The mode holds onto a reference frame the target is locked to."""
        return self.uses_focal_point or self.uses_focus_regions

    @property
    def uses_auto_motion(self) -> bool:
        return self in (StabilizationMode.AUTO, StabilizationMode.SIMILARITY, StabilizationMode.PERSPECTIVE)


class RetimingMode(str, Enum):
    """How a clip whose fps differs from the timeline is resampled onto it (see ``pipeline.retime``)."""

    NEAREST = "nearest"
    """Show the source frame closest in time; whole frames repeat or drop. Cheapest, can judder."""
    BLEND = "blend"
    """Cross-dissolve the two frames straddling each output time. Smoother, ghosts on fast motion."""
    OPTICAL_FLOW = "optical_flow"
    """Synthesise the in-between frame along the motion. Best, slowest — export only; the preview blends."""


class ReadoutAxis(str, Enum):
    """Which way the sensor's readout runs across the *stored* frame — the axis of the rolling-shutter distortion."""

    AUTO = "auto"
    """Detected from the footage by the wobble analysis (the file's rotation breaks ties); rows until it is done."""
    ROWS = "rows"
    """Top to bottom: a landscape recording."""
    COLUMNS = "columns"
    """Left to right: a phone held upright — its sensor rows are the video's columns."""


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
MAX_FOCUS_REGIONS = 4
"""How many tracked regions the object lock accepts (one similarity fit does not gain from more)."""


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

LOOK_FIELDS = ("contrast", "brightness", "gamma", "saturation", "temperature", "sharpen")
"""What a slider preset (a *look*) sets. Denoise and the output scale are not part of a look and stay as they are."""


def same_look(settings: FilterSettings, other: FilterSettings) -> bool:
    """True when the two settings agree on every look field (denoise and scale may differ)."""
    return all(getattr(settings, name) == getattr(other, name) for name in LOOK_FIELDS)


def with_look(settings: FilterSettings, look: FilterSettings) -> FilterSettings:
    """``settings`` with the look fields taken from ``look``."""
    return replace(settings, **{name: getattr(look, name) for name in LOOK_FIELDS})


FILTER_PRESETS: tuple[tuple[str, FilterSettings | None], ...] = (
    ("Personalizado", None),
    ("Suave", FilterSettings(sharpen=12, contrast=52)),
    ("Claro e suave", FilterSettings(sharpen=15, gamma=35, saturation=40)),
    ("Balanceado", FilterSettings(sharpen=24, contrast=54, brightness=51, saturation=52)),
    ("Recuperação forte", FilterSettings(sharpen=32, contrast=56, brightness=52, gamma=52, saturation=54)),
    ("Mais detalhe", FilterSettings(sharpen=38, contrast=55, saturation=55)),
)
"""Slider presets: only look fields (see :data:`LOOK_FIELDS`); applying one leaves denoise and scale alone."""


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
    aspect_ratio: float | None = None
    """Fixed width/height the rectangle is constrained to (the crop preset); ``None`` is a free crop."""

    def pixel_rect(self, frame_width: int, frame_height: int) -> PixelRect | None:
        """Even-aligned crop rectangle, or ``None`` when disabled or covering the whole frame."""
        if not self.enabled:
            return None
        if self.rect.covers_whole_frame(frame_width, frame_height, even=True):
            return None
        return self.rect.to_pixels(frame_width, frame_height, even=True)


@dataclass(frozen=True, order=True)
class StabilizationKeyframe:
    """One hand-set correction at a source frame; translation is a fraction of the frame so it is resolution-free.

    Positive ``tx``/``ty`` move the frame's content right/down; ``rotation`` is degrees, ``scale`` a ratio (1 = none).
    ``shear`` leans the frame horizontally with height and ``stretch_y`` scales it vertically (both 0/1 = none) —
    the manual "jelly" correction for rolling shutter. Between keyframes every field is interpolated
    (see :mod:`pipeline.stabilization.keyframes`).
    """

    frame: int
    tx: float = 0.0
    ty: float = 0.0
    rotation: float = 0.0
    scale: float = 1.0
    shear: float = 0.0
    stretch_y: float = 1.0

    @property
    def is_identity(self) -> bool:
        return (
            self.tx == 0.0
            and self.ty == 0.0
            and self.rotation == 0.0
            and abs(self.scale - 1.0) <= 1e-6
            and self.shear == 0.0
            and abs(self.stretch_y - 1.0) <= 1e-6
        )


@dataclass(frozen=True)
class StabilizationSettings:
    mode: StabilizationMode = StabilizationMode.OFF
    intensity: float = 1.0
    temporal_smoothing: float = 0.35
    focus_rect: NormalizedRect = DEFAULT_STABILIZATION_RECT
    """The legacy single-region lock modes' rectangle (CAMERA_LOCK/HYBRID)."""
    focus_regions: tuple[NormalizedRect, ...] = ()
    """OBJECT mode: the tracked regions (1..``MAX_FOCUS_REGIONS``), each held to its place in the
    reference frame; empty means the UI has not seeded them yet (it starts with one at the centre)."""
    auto_crop: bool = True
    zoom_out: bool = False
    keyframes: tuple[StabilizationKeyframe, ...] = ()
    """Manual mode only: the hand-set corrections, kept sorted by frame (see :meth:`__post_init__`)."""
    rolling_shutter: float = 0.0
    """Rolling-shutter skew to undo, as the sensor readout time over the frame period (0 = off, ±1 = a full
    frame). Negative when the sensor reads against the frame's axis — some phones recorded upright do."""
    rolling_shutter_auto: bool = False
    """Also undo the *wobble* (jello): the clip is analysed once (see :mod:`pipeline.stabilization.wobble`) and
    each frame gets the distortion measured for it; needs no readout time. Ignored under PERSPECTIVE
    (see :attr:`applies_wobble`)."""
    readout_axis: ReadoutAxis = ReadoutAxis.AUTO
    """Along which axis of the frame the distortion varies (both corrections use it)."""
    margin: float = 0.05
    """Correction budget per side, and the fixed inset the automatic crop removes (0.05 .. 0.15).
    Wider = smoother (fewer corrections saturate; measured on walking footage: 5% -> 0.92% RMS,
    8% -> 0.73%, 12% -> 0.50%) at the cost of a bigger constant zoom (1/(1-2*margin))."""

    def __post_init__(self) -> None:
        if self.keyframes:
            by_frame = {keyframe.frame: keyframe for keyframe in self.keyframes}  # a later one wins its frame
            ordered = tuple(by_frame[frame] for frame in sorted(by_frame))
            if ordered != self.keyframes:
                object.__setattr__(self, "keyframes", ordered)

    @property
    def uses_manual_keyframes(self) -> bool:
        return self.mode is StabilizationMode.MANUAL

    @property
    def enabled(self) -> bool:
        return self.mode is not StabilizationMode.OFF

    @property
    def applies_wobble(self) -> bool:
        """The wobble correction actually runs. PERSPECTIVE and MESH suppress it: both already absorb the
        same shear/keystone in their own warp, and correcting it twice deforms (measured on real footage:
        jello 8,5 -> 13 px under the homography, 16,7 -> 23,3 px under the mesh)."""
        return self.rolling_shutter_auto and self.mode not in (
            StabilizationMode.PERSPECTIVE,
            StabilizationMode.MESH,
        )

    @property
    def needs_mesh_analysis(self) -> bool:
        """The clip must be tracked once into a mesh path (see :mod:`pipeline.stabilization.mesh`)."""
        return self.mode is StabilizationMode.MESH

    @property
    def corrects_rolling_shutter(self) -> bool:
        return self.rolling_shutter != 0.0 or self.applies_wobble

    @property
    def corrects_rolling_shutter_skew(self) -> bool:
        """The slider part: needs the motion since the previous frame (the wobble comes from the clip's analysis)."""
        return self.rolling_shutter != 0.0

    @property
    def needs_wobble_analysis(self) -> bool:
        """The clip must be tracked once: for the wobble itself, or to detect the readout axis the slider should use."""
        return self.applies_wobble or (self.corrects_rolling_shutter_skew and self.readout_axis is ReadoutAxis.AUTO)

    @property
    def requires_sequential_processing(self) -> bool:
        """True when frame N depends on frame N-1 (inter-frame motion, temporal smoothing, rolling-shutter skew)."""
        if self.corrects_rolling_shutter_skew:
            return True
        if self.mode in (StabilizationMode.MANUAL, StabilizationMode.MESH):
            return False  # per-frame values come from the keyframes / the clip's mesh analysis, not from frame N-1
        return self.enabled and (
            self.mode.uses_auto_motion
            or self.mode in (StabilizationMode.HYBRID, StabilizationMode.OBJECT)  # dead reckoning carries state
            or self.temporal_smoothing > 0.0
        )


SPEED_MIN = 0.25
SPEED_MAX = 4.0


@dataclass(frozen=True)
class CaptureSettings:
    """How the clip's frames map onto the timeline (fps resampling, speed, audio)."""

    retiming: RetimingMode = RetimingMode.NEAREST
    """How this clip is resampled when its fps differs from the timeline's (no effect when they match)."""
    speed: float = 1.0
    """Playback speed: below 1 is slow motion (frames interpolated per ``retiming``), above 1 is fast."""
    mute: bool = False
    """Drop this clip's audio in the export (its span becomes silence)."""

    @property
    def is_retimed(self) -> bool:
        return abs(self.speed - 1.0) > 1e-6


HUE_BAND_NAMES = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")
HUE_BAND_CENTERS_DEG = (0, 30, 60, 120, 180, 240, 280, 320)
HUE_SHIFT_LIMIT_DEG = 30
BAND_SATURATION_LIMIT = 100
BAND_LUMINANCE_LIMIT = 50
WHITE_BALANCE_GAIN_MIN = 0.5
WHITE_BALANCE_GAIN_MAX = 2.0
NEUTRAL_GAINS = (1.0, 1.0, 1.0)
DEFAULT_WHITE_BALANCE_RECT = NormalizedRect(0.4, 0.4, 0.2, 0.2)


@dataclass(frozen=True)
class HueBandAdjust:
    """Per-hue-band HSL adjustment: hue shift in degrees, saturation and luminance in percent."""

    hue_shift: int = 0
    saturation: int = 0
    luminance: int = 0

    @property
    def is_neutral(self) -> bool:
        return self.hue_shift == 0 and self.saturation == 0 and self.luminance == 0


DEFAULT_HUE_BANDS: tuple[HueBandAdjust, ...] = tuple(HueBandAdjust() for _name in HUE_BAND_NAMES)


@dataclass(frozen=True)
class ColorGradeSettings:
    """White balance (per-channel gains + green/magenta tint) and the per-hue-band adjustments."""

    tint: int = 50
    """0-100, 50 neutral: below pulls toward green, above toward magenta."""
    white_balance_gains: tuple[float, float, float] = NEUTRAL_GAINS
    """B, G, R multipliers, e.g. from :func:`pipeline.grade.neutral_gains` on a gray region."""
    region: NormalizedRect = DEFAULT_WHITE_BALANCE_RECT
    """The gray reference rectangle drawn on the preview (also the exposure lock's metering area)."""
    bands: tuple[HueBandAdjust, ...] = DEFAULT_HUE_BANDS

    @property
    def has_white_balance(self) -> bool:
        return self.tint != 50 or any(abs(gain - 1.0) > 1e-3 for gain in self.white_balance_gains)

    @property
    def has_hue_bands(self) -> bool:
        return any(not band.is_neutral for band in self.bands)

    @property
    def is_neutral(self) -> bool:
        return not self.has_white_balance and not self.has_hue_bands


EXPOSURE_MAX_EV_LIMIT = 2.0
EXPOSURE_DEFAULT_MAX_EV = 1.0


@dataclass(frozen=True)
class ExposureLockSettings:
    """Per-frame gains that hold a reference region's brightness (and colour) constant across the clip.

    See :mod:`pipeline.exposure`. ``reference`` holds the region's B, G, R medians in the frame
    the user picked; ``None`` means nothing was captured yet, so the lock does nothing.
    """

    enabled: bool = False
    region: NormalizedRect | None = DEFAULT_WHITE_BALANCE_RECT
    """Static part of the scene to meter; ``None`` meters the whole frame (content-sensitive)."""
    reference: tuple[float, float, float] | None = None
    strength: int = 100
    """0-100: how much of the measured correction to apply."""
    max_ev: float = EXPOSURE_DEFAULT_MAX_EV
    """Gains are clamped to ``2 ** ±max_ev``."""
    color: bool = True
    """Per-channel gains (brightness and white balance) instead of one gain for all channels."""

    @property
    def is_active(self) -> bool:
        return self.enabled and self.reference is not None and self.strength > 0


class AnonymizeMode(str, Enum):
    BLUR = "blur"
    """Gaussian blur: softens the region, still shows shapes."""
    PIXELATE = "pixelate"
    """Mosaic: replaces the region with coarse blocks (the classic anonymising look)."""


@dataclass(frozen=True)
class AnonymizeSettings:
    """Hide people or details: blur/pixelate detected faces and a chosen region (see :mod:`pipeline.anonymize`)."""

    enabled: bool = False
    mode: AnonymizeMode = AnonymizeMode.PIXELATE
    strength: int = 50
    """0-100: how coarse the mosaic / how strong the blur."""
    detect_faces: bool = True
    """Find faces automatically and cover each one (frontal faces; a profile may be missed)."""
    region: NormalizedRect | None = None
    """A rectangle covered on every frame — for a tattoo, or a face the detector keeps missing."""

    @property
    def is_active(self) -> bool:
        has_region = self.region is not None and self.region.is_valid()
        return self.enabled and self.strength > 0 and (self.detect_faces or has_region)


@dataclass(frozen=True)
class RenderSettings:
    """Complete, immutable snapshot of what the user configured."""

    filters: FilterSettings = DEFAULT_FILTER_SETTINGS
    image_filter: ImageFilterSelection = field(default_factory=ImageFilterSelection)
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    crop: CropSettings = field(default_factory=CropSettings)
    stabilization: StabilizationSettings = field(default_factory=StabilizationSettings)
    denoise_sample_rect: NormalizedRect = DEFAULT_DENOISE_SAMPLE_RECT
    grade: ColorGradeSettings = field(default_factory=ColorGradeSettings)
    exposure: ExposureLockSettings = field(default_factory=ExposureLockSettings)
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    anonymize: AnonymizeSettings = field(default_factory=AnonymizeSettings)

    def output_size(self, width: int, height: int) -> tuple[int, int]:
        """Size of the exported frame for a ``width x height`` source: scale, rotation, then crop.

        Stabilization's automatic crop trims the same fraction on every side, so it keeps this
        aspect ratio; the export fits every clip into the first clip's output canvas.
        """
        scaled_width = max(1, round(width * self.filters.scale_factor))
        scaled_height = max(1, round(height * self.filters.scale_factor))
        displayed_width, displayed_height = self.geometry.output_size(scaled_width, scaled_height)
        pixels = self.crop.pixel_rect(displayed_width, displayed_height)
        if pixels is None:
            return (displayed_width, displayed_height)
        return (pixels[2], pixels[3])


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))
