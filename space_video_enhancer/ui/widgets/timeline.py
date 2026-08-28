"""Zoomable timeline under the preview: ruler, clip strip, video/audio tracks, cuts, markers and the playhead.

Frames map to pixels through :class:`TimelineScale` (zoom = pixels per frame, scroll = first
visible frame), so hit testing and drawing share one conversion at every zoom level. A cut
``(start, end)`` is inclusive and occupies the pixels from ``x_of(start)`` to ``x_of(end + 1)``;
its edges, the clip markers and the playhead are grabbed in pixel distance from those boundaries.
Frames are *timeline frames* of the sequence; :class:`ClipBlock` tells which source frames each
clip shows there, which the filmstrip and the waveform need to fetch the right media.
"""

from __future__ import annotations

import bisect
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QContextMenuEvent,
    QImage,
    QMouseEvent,
    QNativeGestureEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QMenu, QScrollBar, QSizePolicy, QWidget

from space_video_enhancer.audio import AudioPeaks
from space_video_enhancer.ui.format import format_timecode
from space_video_enhancer.ui.images import device_pixel_ratio

RULER_HEIGHT = 22
TRACK_GAP = 4
CLIP_TRACK_HEIGHT = 18
"""The strip naming the clips of the sequence; boundaries between clips continue down the tracks."""
VIDEO_TRACK_HEIGHT = 56
AUDIO_TRACK_HEIGHT = 36
SCROLLBAR_HEIGHT = 12
MIN_WIDTH = 240
MAX_PIXELS_PER_FRAME = 48.0
"""At full zoom one frame is this many pixels wide: enough to grab a single frame comfortably."""
ZOOM_STEP = 1.6
WHEEL_ZOOM_PER_UNIT = 1.0025
"""Ctrl + wheel zoom factor per wheel unit (an eighth of a degree; a notch is 120 units -> x1.35)."""
WHEEL_SCROLL_FRACTION = 0.1
"""A wheel notch without modifiers scrolls this fraction of the visible frames."""
MIN_LABEL_SPACING_PX = 84
EDGE_GRAB_PX = 6
MARKER_GRAB_PX = 7
FOLLOW_LEAD_FRACTION = 0.1
"""When the playhead leaves the view it is brought back this far from the left edge."""
TICK_FRAMES = (1, 2, 5, 10)
THUMBNAIL_REQUEST_DELAY_MS = 60
"""Zoom/scroll settle for this long before the missing filmstrip thumbnails are requested."""
THUMBNAIL_CACHE_SIZE = 600
DEFAULT_ASPECT_RATIO = 16 / 9
TICK_SECONDS = (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)

BACKGROUND_COLOR = QColor("#fcfaf7")
RULER_COLOR = QColor("#efe9de")
RULER_BORDER_COLOR = QColor("#d6d3d1")
TICK_COLOR = QColor("#78716c")
LABEL_COLOR = QColor("#57534e")
TRACK_COLOR = QColor("#e7e5e4")
TRACK_OUTSIDE_CLIP_COLOR = QColor(120, 113, 108, 60)
RANGE_COLOR = QColor(15, 118, 110, 70)
START_MARKER_COLOR = QColor("#0f766e")
END_MARKER_COLOR = QColor("#b45309")
CUT_COLOR = QColor(220, 38, 38, 110)
CUT_SELECTED_COLOR = QColor(220, 38, 38, 170)
CUT_EDGE_COLOR = QColor("#b91c1c")
PENDING_CUT_COLOR = QColor("#dc2626")
CLIP_COLORS = (QColor(15, 118, 110, 55), QColor(180, 83, 9, 45))
CLIP_ACTIVE_COLOR = QColor(15, 118, 110, 120)
CLIP_BORDER_COLOR = QColor("#0f766e")
CLIP_BOUNDARY_COLOR = QColor(31, 41, 55, 110)
CLIP_TEXT_COLOR = QColor("#1f2937")
CLIP_MIN_TEXT_WIDTH = 28
CLIP_DRAG_THRESHOLD_PX = 6
"""A press on the clip strip becomes a reorder drag after this much horizontal travel; less is a click (seek)."""
CLIP_GHOST_COLOR = QColor(15, 118, 110, 90)
CLIP_INSERT_COLOR = QColor("#0f766e")
PLAYHEAD_COLOR = QColor("#1f2937")
PLAYHEAD_CELL_COLOR = QColor(31, 41, 55, 40)
TRACK_LABEL_COLOR = QColor("#57534e")
TRACK_LABEL_BACKGROUND = QColor(252, 250, 247, 200)
HINT_COLOR = QColor("#9ca3af")
WAVEFORM_COLOR = QColor(15, 118, 110, 150)
WAVEFORM_CENTER_COLOR = QColor(15, 118, 110, 90)
WAVEFORM_MAX_GAIN = 20.0
"""Quiet audio is scaled up so its loudest peak fills the track, but never more than this."""
AUDIO_STATE_TEXT = {
    "loading": "Carregando áudio…",
    "none": "Sem áudio",
    "unavailable": "FFmpeg não encontrado: sem forma de onda",
    "missing": "Arquivo não encontrado",
}
CLIP_MISSING_COLOR = QColor(220, 38, 38, 120)
CLIP_MISSING_TEXT_COLOR = QColor("#7f1d1d")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class TimelineScale:
    """Frames <-> pixels for a viewport ``width`` pixels wide."""

    frame_count: int = 0
    width: int = 1
    pixels_per_frame: float = 1.0
    scroll: float = 0.0
    """First visible frame (fractional while zoomed in)."""

    def fit_pixels_per_frame(self) -> float:
        return max(1e-6, self.width / max(1, self.frame_count))

    def max_pixels_per_frame(self) -> float:
        return max(MAX_PIXELS_PER_FRAME, self.fit_pixels_per_frame())

    @property
    def visible_frames(self) -> float:
        return self.width / self.pixels_per_frame

    def max_scroll(self) -> float:
        return max(0.0, self.frame_count - self.visible_frames)

    def is_fit(self) -> bool:
        return self.pixels_per_frame <= self.fit_pixels_per_frame() * (1.0 + 1e-6)

    def is_max_zoom(self) -> bool:
        return self.pixels_per_frame >= self.max_pixels_per_frame() * (1.0 - 1e-6)

    def fit(self) -> None:
        self.pixels_per_frame = self.fit_pixels_per_frame()
        self.scroll = 0.0

    def set_width(self, width: int) -> None:
        """A resize keeps the fit if it was fitted, otherwise the zoom level (and clamps the scroll)."""
        was_fit = self.is_fit()
        self.width = max(1, width)
        if was_fit:
            self.fit()
        else:
            self.clamp()

    def clamp(self) -> None:
        self.pixels_per_frame = _clamp(self.pixels_per_frame, self.fit_pixels_per_frame(), self.max_pixels_per_frame())
        self.scroll = _clamp(self.scroll, 0.0, self.max_scroll())

    def zoom_to(self, pixels_per_frame: float, anchor_x: float | None = None) -> None:
        """Change the zoom keeping the frame under ``anchor_x`` (default: the middle) where it is."""
        anchor_x = self.width / 2.0 if anchor_x is None else _clamp(anchor_x, 0.0, self.width)
        anchor_frame = self.frame_at(anchor_x)
        self.pixels_per_frame = _clamp(pixels_per_frame, self.fit_pixels_per_frame(), self.max_pixels_per_frame())
        self.scroll = anchor_frame - anchor_x / self.pixels_per_frame
        self.clamp()

    def zoom_by(self, factor: float, anchor_x: float | None = None) -> None:
        self.zoom_to(self.pixels_per_frame * factor, anchor_x)

    def scroll_to(self, frame: float) -> None:
        self.scroll = frame
        self.clamp()

    def x_of(self, frame: float) -> float:
        return (frame - self.scroll) * self.pixels_per_frame

    def frame_at(self, x: float) -> float:
        return self.scroll + x / self.pixels_per_frame

    def frame_index_at(self, x: float) -> int:
        """The frame whose cell contains pixel ``x`` (clamped to the clip)."""
        if self.frame_count <= 0:
            return 0
        return int(_clamp(math.floor(self.frame_at(x)), 0, self.frame_count - 1))

    def boundary_at(self, x: float) -> int:
        """The nearest frame boundary to pixel ``x`` (``0`` .. ``frame_count``)."""
        return int(_clamp(round(self.frame_at(x)), 0, max(0, self.frame_count)))

    def is_visible(self, frame: int) -> bool:
        x = self.x_of(frame)
        return 0.0 <= x < self.width

    def follow(self, frame: int) -> bool:
        """Scroll so an off-screen ``frame`` lands near the left edge; ``False`` when nothing moved."""
        if self.frame_count <= 0 or self.is_fit() or self.is_visible(frame):
            return False
        previous = self.scroll
        self.scroll_to(frame - self.visible_frames * FOLLOW_LEAD_FRACTION)
        return self.scroll != previous


def ruler_steps(fps: float, pixels_per_frame: float, min_label_px: int = MIN_LABEL_SPACING_PX) -> tuple[int, int]:
    """``(major, minor)`` tick spacing in frames: the smallest round step whose labels stay apart."""
    fps = fps if fps > 0 else 30.0
    candidates = sorted({*TICK_FRAMES, *(max(1, round(fps * seconds)) for seconds in TICK_SECONDS)})
    major = next((step for step in candidates if step * pixels_per_frame >= min_label_px), candidates[-1])
    while major * pixels_per_frame < min_label_px:
        major *= 2
    for divisor in (5, 4, 2):
        if major % divisor == 0 and major // divisor * pixels_per_frame >= 4:
            return major, major // divisor
    return major, major


def ruler_label(frame: int, fps: float, *, with_frames: bool) -> str:
    text = format_timecode(frame, fps)
    return text if with_frames else text.rsplit(".", 1)[0]


@dataclass(frozen=True)
class ClipBlock:
    """One clip of the sequence as the timeline sees it: where it sits and which source frames it shows."""

    start: int
    """First timeline frame."""
    count: int
    name: str
    source: Path
    source_start: int = 0
    """Source frame shown at ``start``; the clip is contiguous, so frame ``start + k`` is ``source_start + k``."""
    missing: bool = False
    """The file is not where the project says: the block is hatched and marked, nothing of it can be decoded."""

    @property
    def end(self) -> int:
        return self.start + self.count - 1

    def contains(self, frame: int) -> bool:
        return self.start <= frame <= self.end


@dataclass(frozen=True)
class Hit:
    """What is under the mouse: ``seek`` (empty timeline), a clip block, a cut edge/body or a clip marker."""

    kind: str
    index: int = -1

    @property
    def is_cut(self) -> bool:
        return self.kind.startswith("cut_")

    @property
    def cursor(self) -> Qt.CursorShape:
        if self.kind in ("cut_start", "cut_end", "clip_start", "clip_end"):
            return Qt.CursorShape.SizeHorCursor
        if self.kind in ("cut_body", "clip"):
            return Qt.CursorShape.OpenHandCursor
        return Qt.CursorShape.ArrowCursor


@dataclass
class _Drag:
    hit: Hit
    press_frame: float
    cuts: list[tuple[int, int]]
    clip: tuple[int, int]
    moved: bool = False
    last_cuts: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class _ClipDrag:
    """A press on the clip strip: a click until the mouse travels, then a reorder drag with an insertion point."""

    index: int
    press_x: float
    current_x: float
    target: int
    """Insertion gap (``0`` .. number of clips) nearest to the mouse."""
    active: bool = False


class TimelineWidget(QWidget):
    """Ruler + video track with a playhead; the model (cuts, clip range) is pushed in and edits come out as signals."""

    valueChanged = Signal(int)
    """Playhead moved (by the user or by :meth:`setValue`), like ``QSlider``."""
    cutsEdited = Signal(object, bool)
    """``(cuts, finished)`` while a cut edge or body is dragged; ``finished`` on release."""
    clipRangeEdited = Signal(int, int, bool)
    """``(start, end, finished)`` while a clip marker is dragged."""
    cutSelected = Signal(int)
    """Index of the selected cut, ``-1`` for none."""
    cutRemoveRequested = Signal(int)
    cutToggleRequested = Signal()
    """Context menu: start or finish a cut at the playhead (moved there first)."""
    clipSplitRequested = Signal()
    """Context menu: split the clip at the playhead (moved there first)."""
    clipRemoveRequested = Signal(int)
    clipMoveRequested = Signal(int, int)
    """``(index, new_index)``: a clip block dragged to another place on the strip."""
    zoomChanged = Signal()
    thumbnailsRequested = Signal(object, int, float)
    """``(frames, height_device_px, device_pixel_ratio)``: filmstrip thumbnails missing from the view."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = TimelineScale()
        self._fps = 30.0
        self._aspect_ratio = DEFAULT_ASPECT_RATIO
        self._thumbnails: OrderedDict[int, QImage] = OrderedDict()
        self._thumbnail_frames: list[int] = []
        """Sorted keys of ``_thumbnails`` for the nearest-neighbour fallback while new ones decode."""
        self._thumbnail_key: tuple[int, float] | None = None
        self._clips: list[ClipBlock] = []
        self._active_clip = -1
        self._peaks: dict[Path, AudioPeaks] = {}
        self._waveform_gain = 1.0
        self._audio_states: dict[Path, str] = {}
        self._thumbnail_timer = QTimer(self)
        self._thumbnail_timer.setSingleShot(True)
        self._thumbnail_timer.timeout.connect(self.request_missing_thumbnails)
        self._value = 0
        self._clip = (0, 0)
        self._show_clip = False
        self._cuts: list[tuple[int, int]] = []
        self._pending_cut: int | None = None
        self._selected_cut = -1
        self._drag: _Drag | None = None
        self._clip_drag: _ClipDrag | None = None
        self._scrubbing = False
        self._syncing_scrollbar = False
        self._hover = Hit("seek")

        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal, self)
        self.scrollbar.setObjectName("timelineScrollBar")
        self.scrollbar.valueChanged.connect(self._scroll_from_bar)
        self.scrollbar.hide()

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    # --- geometry -----------------------------------------------------------------------
    def track_heights(self) -> tuple[int, ...]:
        return (CLIP_TRACK_HEIGHT, VIDEO_TRACK_HEIGHT, AUDIO_TRACK_HEIGHT)

    def content_height(self) -> int:
        return RULER_HEIGHT + sum(TRACK_GAP + height for height in self.track_heights())

    def sizeHint(self) -> QSize:
        return QSize(MIN_WIDTH * 2, self.content_height() + SCROLLBAR_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(MIN_WIDTH, self.content_height() + SCROLLBAR_HEIGHT)

    def ruler_rect(self) -> QRect:
        return QRect(0, 0, self.width(), RULER_HEIGHT)

    def track_rect(self, index: int) -> QRect:
        top = RULER_HEIGHT
        for track, height in enumerate(self.track_heights()):
            top += TRACK_GAP
            if track == index:
                return QRect(0, top, self.width(), height)
            top += height
        raise IndexError(index)

    def clip_track_rect(self) -> QRect:
        return self.track_rect(0)

    def video_track_rect(self) -> QRect:
        return self.track_rect(1)

    def audio_track_rect(self) -> QRect:
        return self.track_rect(2)

    def tracks_rect(self) -> QRect:
        return QRect(0, RULER_HEIGHT, self.width(), self.content_height() - RULER_HEIGHT)

    def media_tracks_rect(self) -> QRect:
        """Video and audio tracks together (where cuts and the playhead cell are drawn)."""
        top = self.video_track_rect().top()
        return QRect(0, top, self.width(), self.content_height() - top)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.scale.set_width(self.width())
        self.scrollbar.setGeometry(0, self.content_height(), self.width(), SCROLLBAR_HEIGHT)
        self._view_changed()

    # --- model in ------------------------------------------------------------------------
    def set_media(self, frame_count: int, fps: float, aspect_ratio: float = DEFAULT_ASPECT_RATIO) -> None:
        """A new sequence: fit it whole in the view, forget the selection, the filmstrip and the waveforms."""
        self.scale.frame_count = max(0, frame_count)
        self._fps = fps if fps > 0 else 30.0
        self._aspect_ratio = aspect_ratio if aspect_ratio > 0 else DEFAULT_ASPECT_RATIO
        self.clear_thumbnails()
        self._clips = []
        self._active_clip = -1
        self._peaks = {}
        self._waveform_gain = 1.0
        self._audio_states = {}
        self.scale.set_width(max(1, self.width()))  # resizeEvent has not run yet when the widget is hidden
        self.scale.fit()
        self._value = 0
        self._selected_cut = -1
        self._drag = None
        self._scrubbing = False
        self._view_changed()

    def set_clips(self, clips: Sequence[ClipBlock]) -> None:
        """The sequence's layout changed (clip added, removed, moved, trimmed or split).

        The timeline length follows the clips; thumbnails are dropped because a timeline frame
        may now show a different source frame, waveforms of files still in use are kept.
        """
        self._clips = list(clips)
        self._clip_drag = None
        total = sum(clip.count for clip in self._clips)
        if total != self.scale.frame_count:
            self.scale.frame_count = total
            self.scale.set_width(max(1, self.width()))
        self.clear_thumbnails()
        used = {clip.source for clip in self._clips}
        self._peaks = {path: peaks for path, peaks in self._peaks.items() if path in used}
        self._audio_states = {path: state for path, state in self._audio_states.items() if path in used}
        self._update_waveform_gain()
        self._value = int(_clamp(self._value, 0, self.maximum()))
        if self._active_clip >= len(self._clips):
            self._active_clip = -1
        if self._selected_cut >= len(self._cuts):
            self._set_selected(-1)
        self._view_changed()

    def clips(self) -> tuple[ClipBlock, ...]:
        return tuple(self._clips)

    def clip_at(self, frame: int) -> int:
        """Index of the clip showing timeline ``frame`` (``-1`` without clips)."""
        for index, clip in enumerate(self._clips):
            if clip.contains(frame):
                return index
        return len(self._clips) - 1 if self._clips and frame > self._clips[-1].end else (0 if self._clips else -1)

    def set_active_clip(self, index: int) -> None:
        """Highlight the clip whose settings the tabs show."""
        index = index if 0 <= index < len(self._clips) else -1
        if index != self._active_clip:
            self._active_clip = index
            self.update()

    def active_clip(self) -> int:
        return self._active_clip

    @property
    def frame_count(self) -> int:
        return self.scale.frame_count

    @property
    def fps(self) -> float:
        return self._fps

    def value(self) -> int:
        return self._value

    def maximum(self) -> int:
        return max(0, self.frame_count - 1)

    def setValue(self, value: int) -> None:
        """Move the playhead; scrolls the view along when the frame is off-screen (playback)."""
        value = int(_clamp(value, 0, self.maximum()))
        if value == self._value:
            return
        self._value = value
        if not self._scrubbing and self._drag is None and self.scale.follow(value):
            self._view_changed()
        else:
            self.update()
        self.valueChanged.emit(value)

    def set_clip_range(self, start: int, end: int, show: bool = True) -> None:
        self._clip = (min(start, end), max(start, end))
        self._show_clip = show
        self.update()

    def set_cuts(self, cuts: Sequence[tuple[int, int]]) -> None:
        """Cuts from the model; ignored while the user is dragging one (the drag is the source of truth)."""
        if self._drag is not None and self._drag.hit.is_cut:
            return
        self._cuts = [(min(start, end), max(start, end)) for start, end in cuts]
        if self._selected_cut >= len(self._cuts):
            self._set_selected(-1)
        self.update()

    def cuts(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._cuts)

    def set_pending_cut(self, frame: int | None) -> None:
        self._pending_cut = frame
        self.update()

    def selected_cut(self) -> int:
        return self._selected_cut

    def select_cut(self, index: int) -> None:
        self._set_selected(index if 0 <= index < len(self._cuts) else -1)

    def _set_selected(self, index: int) -> None:
        if index == self._selected_cut:
            return
        self._selected_cut = index
        self.update()
        self.cutSelected.emit(index)

    # --- zoom and scroll -----------------------------------------------------------------
    def is_fit(self) -> bool:
        return self.scale.is_fit()

    def is_max_zoom(self) -> bool:
        return self.scale.is_max_zoom()

    def zoom_to_fit(self) -> None:
        self.scale.fit()
        self._view_changed()

    def zoom_in(self) -> None:
        self.zoom_by(ZOOM_STEP)

    def zoom_out(self) -> None:
        self.zoom_by(1.0 / ZOOM_STEP)

    def zoom_by(self, factor: float, anchor_x: float | None = None) -> None:
        """Zoom around ``anchor_x`` (default: the playhead when visible, else the middle of the view)."""
        if self.frame_count <= 0:
            return
        if anchor_x is None and self.scale.is_visible(self._value):
            anchor_x = self.scale.x_of(self._value + 0.5)
        self.scale.zoom_by(factor, anchor_x)
        self._view_changed()

    def scroll_to(self, frame: float) -> None:
        self.scale.scroll_to(frame)
        self._view_changed()

    def visible_range(self) -> tuple[int, int]:
        """First and last (exclusive) frame index in view."""
        first = int(_clamp(math.floor(self.scale.scroll), 0, self.frame_count))
        last = int(_clamp(math.ceil(self.scale.frame_at(self.width())), 0, self.frame_count))
        return first, last

    def _view_changed(self) -> None:
        self._sync_scrollbar()
        self.update()
        self.zoomChanged.emit()
        self._thumbnail_timer.start(THUMBNAIL_REQUEST_DELAY_MS)

    def _sync_scrollbar(self) -> None:
        self._syncing_scrollbar = True
        try:
            max_scroll = self.scale.max_scroll()
            self.scrollbar.setRange(0, math.ceil(max_scroll))
            self.scrollbar.setPageStep(max(1, int(self.scale.visible_frames)))
            self.scrollbar.setSingleStep(max(1, int(self.scale.visible_frames * WHEEL_SCROLL_FRACTION)))
            self.scrollbar.setValue(round(self.scale.scroll))
            self.scrollbar.setVisible(max_scroll >= 0.5)
        finally:
            self._syncing_scrollbar = False

    def _scroll_from_bar(self, value: int) -> None:
        if self._syncing_scrollbar:
            return
        self.scale.scroll_to(float(value))
        self.update()
        self.zoomChanged.emit()
        self._thumbnail_timer.start(THUMBNAIL_REQUEST_DELAY_MS)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.frame_count <= 0:
            event.ignore()
            return
        angle = event.angleDelta()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            units = angle.y() or angle.x()
            if units:
                self.zoom_by(WHEEL_ZOOM_PER_UNIT**units, event.position().x())
            event.accept()
            return
        pixels = event.pixelDelta()
        if not pixels.isNull():
            delta_px = pixels.x() if pixels.x() else pixels.y()
            frames = -delta_px / self.scale.pixels_per_frame
        else:
            notches = (angle.x() if angle.x() else angle.y()) / 120.0
            frames = -notches * self.scale.visible_frames * WHEEL_SCROLL_FRACTION
        if frames:
            self.scroll_to(self.scale.scroll + frames)
        event.accept()

    def event(self, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.NativeGesture
            and isinstance(event, QNativeGestureEvent)
            and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture
            and self.frame_count > 0
        ):
            self.zoom_by(1.0 + event.value(), event.position().x())  # trackpad pinch
            return True
        return super().event(event)

    # --- filmstrip ------------------------------------------------------------------------
    def thumbnail_size(self) -> tuple[int, int]:
        """Logical (width, height) of one filmstrip cell: the track height at the video's aspect ratio."""
        height = VIDEO_TRACK_HEIGHT
        return max(1, round(height * self._aspect_ratio)), height

    def thumbnail_slots(self) -> tuple[tuple[int, float], ...]:
        """``(frame, x)`` of every filmstrip cell in view; cells are anchored to frame 0, so scrolling reuses them."""
        if self.frame_count <= 0 or self.width() <= 0:
            return ()
        width, _height = self.thumbnail_size()
        ppf = self.scale.pixels_per_frame
        content_left = self.scale.scroll * ppf
        first = math.floor(content_left / width)
        last = math.floor((content_left + self.width()) / width)
        slots = []
        for slot in range(first, last + 1):
            frame = int(_clamp(math.floor(slot * width / ppf), 0, self.frame_count - 1))
            slots.append((frame, slot * width - content_left))
        return tuple(slots)

    def request_missing_thumbnails(self) -> None:
        """Ask for the visible cells that have no exact thumbnail yet (emits nothing when complete)."""
        self._thumbnail_timer.stop()
        if self.frame_count <= 0:
            return
        ratio = device_pixel_ratio(self)
        _width, height = self.thumbnail_size()
        key = (round(height * ratio), ratio)
        if self._thumbnail_key != key:
            self.clear_thumbnails()
            self._thumbnail_key = key
        missing = tuple(dict.fromkeys(frame for frame, _x in self.thumbnail_slots() if frame not in self._thumbnails))
        if missing:
            self.thumbnailsRequested.emit(missing, key[0], ratio)

    def set_thumbnail(self, frame: int, image: QImage) -> None:
        if frame in self._thumbnails:
            self._thumbnails.move_to_end(frame)
        else:
            bisect.insort(self._thumbnail_frames, frame)
            while len(self._thumbnails) >= THUMBNAIL_CACHE_SIZE:
                evicted, _image = self._thumbnails.popitem(last=False)
                del self._thumbnail_frames[bisect.bisect_left(self._thumbnail_frames, evicted)]
        self._thumbnails[frame] = image
        self.update()

    def clear_thumbnails(self) -> None:
        self._thumbnails.clear()
        self._thumbnail_frames.clear()
        self.update()

    def thumbnail_count(self) -> int:
        return len(self._thumbnails)

    def nearest_thumbnail(self, frame: int, tolerance: float) -> QImage | None:
        """The exact thumbnail, else the closest cached one within ``tolerance`` frames."""
        exact = self._thumbnails.get(frame)
        if exact is not None or not self._thumbnail_frames:
            return exact
        position = bisect.bisect_left(self._thumbnail_frames, frame)
        candidates = self._thumbnail_frames[max(0, position - 1) : position + 1]
        best = min(candidates, key=lambda cached: abs(cached - frame))
        return self._thumbnails[best] if abs(best - frame) <= tolerance else None

    # --- waveform -------------------------------------------------------------------------
    def set_waveform(self, path: Path, peaks: AudioPeaks | None) -> None:
        """Peaks of one source file; every clip of that file draws its slice of them."""
        if peaks is not None and peaks.bucket_count:
            self._peaks[path] = peaks
            self._audio_states[path] = "ready"
        else:
            self._peaks.pop(path, None)
            self._audio_states[path] = "none"
        self._update_waveform_gain()
        self.update()

    def _update_waveform_gain(self) -> None:
        """One display gain for the whole track: the loudest source's peak fills it (up to a cap)."""
        gains = []
        for peaks in self._peaks.values():
            loudest = float(max(peaks.maximums.max(), -peaks.minimums.min(), 0.0))
            gains.append(float(_clamp(0.95 / loudest, 1.0, WAVEFORM_MAX_GAIN)) if loudest > 0.0 else 1.0)
        self._waveform_gain = min(gains) if gains else 1.0

    def waveform_gain(self) -> float:
        return self._waveform_gain

    def set_audio_state(self, path: Path, state: str) -> None:
        """``loading`` while ffmpeg decodes, ``unavailable`` without ffmpeg; :meth:`set_waveform` sets the rest."""
        self._audio_states[path] = state
        self.update()

    def audio_state(self, path: Path | None = None) -> str:
        """State of one file's waveform; without ``path``, of the first clip's file (``none`` without clips)."""
        if path is None:
            if not self._clips:
                return "none"
            path = self._clips[0].source
        return self._audio_states.get(path, "none")

    def waveform_columns(self, left: int, right: int) -> tuple[np.ndarray, np.ndarray] | None:
        """``(minimums, maximums)`` per pixel column in ``[left, right)``: the extremes of the buckets it covers.

        Each clip contributes the columns it occupies, read from its own file at its own offset;
        columns of clips without audio (or past the end of a file's audio) are silent. ``None``
        when no clip in view has a waveform.
        """
        if right <= left:
            return None
        minimums = np.zeros(right - left, dtype=np.float32)
        maximums = np.zeros(right - left, dtype=np.float32)
        found = False
        for clip in self._clips:
            peaks = self._peaks.get(clip.source)
            if peaks is None or peaks.bucket_count == 0:
                continue
            first = max(left, math.ceil(self.scale.x_of(clip.start) - 0.5))
            last = min(right, math.ceil(self.scale.x_of(clip.end + 1) - 0.5))
            if last <= first:
                continue
            found = True
            xs = np.arange(first, last + 1, dtype=np.float64)
            frames = self.scale.scroll + xs / self.scale.pixels_per_frame - clip.start + clip.source_start
            raw_edges = np.floor(frames / self._fps * peaks.buckets_per_second).astype(np.int64)
            edges = np.maximum.accumulate(np.clip(raw_edges, 0, peaks.bucket_count - 1))  # reduceat: non-decreasing
            column_minimums = np.minimum.reduceat(peaks.minimums, edges)[:-1]
            column_maximums = np.maximum.reduceat(peaks.maximums, edges)[:-1]
            silent = (raw_edges[:-1] >= peaks.bucket_count) | (raw_edges[1:] <= 0)
            minimums[first - left : last - left] = np.where(silent, 0.0, column_minimums)
            maximums[first - left : last - left] = np.where(silent, 0.0, column_maximums)
        return (minimums, maximums) if found else None

    # --- hit testing and mouse -----------------------------------------------------------
    def hit_test(self, pos: QPointF) -> Hit:
        x, y = pos.x(), pos.y()
        if self.frame_count <= 0 or not (0 <= y < self.content_height()):
            return Hit("none")
        if y < RULER_HEIGHT:
            if self._show_clip:
                for kind, frame in (("clip_end", self._clip[1] + 1), ("clip_start", self._clip[0])):
                    if abs(self.scale.x_of(frame) - x) <= MARKER_GRAB_PX:
                        return Hit(kind)
            return Hit("seek")
        if self._clips and self.clip_track_rect().contains(QPoint(int(x), int(y))):
            return Hit("clip", self.clip_at(self.scale.frame_index_at(x)))
        if self.tracks_rect().contains(QPoint(int(x), int(y))):
            nearest: tuple[float, Hit] | None = None
            for index, (start, end) in enumerate(self._cuts):
                for kind, frame in (("cut_start", start), ("cut_end", end + 1)):
                    distance = abs(self.scale.x_of(frame) - x)
                    if distance <= EDGE_GRAB_PX and (nearest is None or distance < nearest[0]):
                        nearest = (distance, Hit(kind, index))
            if nearest is not None:
                return nearest[1]
            for index, (start, end) in enumerate(self._cuts):
                if self.scale.x_of(start) <= x < self.scale.x_of(end + 1):
                    return Hit("cut_body", index)
        return Hit("seek")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self.frame_count <= 0:
            super().mousePressEvent(event)
            return
        pos = event.position()
        hit = self.hit_test(pos)
        if hit.kind == "none":
            return
        if hit.kind == "clip":
            self._set_selected(-1)
            self._clip_drag = _ClipDrag(hit.index, pos.x(), pos.x(), hit.index)  # click or reorder: decided on move
            return
        if hit.kind == "seek":
            self._set_selected(-1)
            self._scrubbing = True
            self.setValue(self.scale.frame_index_at(pos.x()))
            return
        if hit.kind == "cut_body":
            self._set_selected(hit.index)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self._drag = _Drag(hit, self.scale.frame_at(pos.x()), list(self._cuts), self._clip)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._clip_drag is not None:
            self._move_clip_drag(pos.x())
            return
        if self._scrubbing:
            self.setValue(self.scale.frame_index_at(pos.x()))
            return
        if self._drag is not None:
            self._apply_drag(pos.x(), finished=False)
            return
        hit = self.hit_test(pos)
        if hit != self._hover:
            self._hover = hit
            self.setCursor(hit.cursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self._clip_drag is not None:
            self._finish_clip_drag()
            self.setCursor(self.hit_test(event.position()).cursor)
            return
        if self._scrubbing:
            self._scrubbing = False
            return
        if self._drag is not None:
            self._apply_drag(event.position().x(), finished=True)
            self.setCursor(self.hit_test(event.position()).cursor)

    def leaveEvent(self, event: QEvent) -> None:
        super().leaveEvent(event)
        self._hover = Hit("seek")
        self.unsetCursor()

    # --- reordering clips on the strip ------------------------------------------------------
    def clip_insertion_index(self, x: float) -> int:
        """The gap between clips (``0`` .. ``len(clips)``) whose boundary is nearest to pixel ``x``."""
        if not self._clips:
            return 0
        boundaries = [self.scale.x_of(clip.start) for clip in self._clips]
        boundaries.append(self.scale.x_of(self._clips[-1].end + 1))
        return min(range(len(boundaries)), key=lambda gap: abs(boundaries[gap] - x))

    def _move_clip_drag(self, x: float) -> None:
        drag = self._clip_drag
        assert drag is not None
        drag.current_x = x
        if not drag.active and abs(x - drag.press_x) >= CLIP_DRAG_THRESHOLD_PX:
            drag.active = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        if drag.active:
            drag.target = self.clip_insertion_index(x)
            self.update()

    def _finish_clip_drag(self) -> None:
        drag, self._clip_drag = self._clip_drag, None
        assert drag is not None
        if not drag.active:
            self.setValue(self.scale.frame_index_at(drag.press_x))  # a plain click on the strip seeks there
            return
        self.update()
        new_index = drag.target if drag.target <= drag.index else drag.target - 1
        if new_index != drag.index:
            self.clipMoveRequested.emit(drag.index, new_index)

    def clip_drag_target(self) -> int | None:
        """Insertion gap of the reorder drag in progress, ``None`` when nothing is being dragged."""
        drag = self._clip_drag
        return drag.target if drag is not None and drag.active else None

    def _apply_drag(self, x: float, *, finished: bool) -> None:
        drag = self._drag
        assert drag is not None
        boundary = self.scale.boundary_at(x)
        last_frame = self.frame_count - 1
        if drag.hit.kind in ("clip_start", "clip_end"):
            start, end = drag.clip
            if drag.hit.kind == "clip_start":
                start = int(_clamp(boundary, 0, end))
            else:
                end = int(_clamp(boundary - 1, start, last_frame))
            changed = (start, end) != self._clip
            drag.moved = drag.moved or changed
            if changed or (finished and drag.moved):
                self._clip = (start, end)
                self.update()
                self.clipRangeEdited.emit(start, end, finished)
        else:
            start, end = drag.cuts[drag.hit.index]
            if drag.hit.kind == "cut_start":
                start = int(_clamp(boundary, 0, end))
            elif drag.hit.kind == "cut_end":
                end = int(_clamp(boundary - 1, start, last_frame))
            else:
                length = end - start
                start = int(_clamp(start + round(self.scale.frame_at(x) - drag.press_frame), 0, last_frame - length))
                end = start + length
            cuts = list(drag.cuts)
            cuts[drag.hit.index] = (start, end)
            changed = cuts != self._cuts
            drag.moved = drag.moved or changed
            if changed or (finished and drag.moved):
                self._cuts = cuts
                self.update()
                if finished:
                    self._drag = None  # let the model's normalized cuts come back through set_cuts
                self.cutsEdited.emit(tuple(cuts), finished)
        if finished:
            self._drag = None

    # --- context menu ---------------------------------------------------------------------
    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        if self.frame_count <= 0:
            return
        menu = self.context_menu(QPointF(event.pos()))
        menu.exec(event.globalPos())

    def context_menu(self, pos: QPointF) -> QMenu:
        menu = QMenu(self)
        hit = self.hit_test(pos)
        if hit.is_cut and 0 <= hit.index < len(self._cuts):
            start, end = self._cuts[hit.index]
            self._set_selected(hit.index)
            menu.addAction("Remover corte", lambda: self.cutRemoveRequested.emit(hit.index))
            menu.addAction("Ir para o início do corte", lambda: self.setValue(start))
            menu.addAction("Ir para o fim do corte", lambda: self.setValue(end))
        else:
            frame = self.scale.frame_index_at(pos.x())
            label = "Terminar corte aqui" if self._pending_cut is not None else "Iniciar corte aqui"
            menu.addAction(label, lambda: self._toggle_cut_at(frame))
            if self._clips:
                clip_index = self.clip_at(frame)
                split = menu.addAction("Dividir clipe aqui", lambda: self._split_at(frame))
                split.setEnabled(clip_index >= 0 and frame != self._clips[clip_index].start)
                if len(self._clips) > 1 and clip_index >= 0:
                    name = self._clips[clip_index].name
                    menu.addAction(
                        f"Remover clipe {clip_index + 1} ({name})", lambda: self.clipRemoveRequested.emit(clip_index)
                    )
        menu.addSeparator()
        zoom = menu.addAction("Ajustar zoom ao vídeo", self.zoom_to_fit)
        zoom.setEnabled(not self.is_fit())
        return menu

    def _toggle_cut_at(self, frame: int) -> None:
        self.setValue(frame)
        self.cutToggleRequested.emit()

    def _split_at(self, frame: int) -> None:
        self.setValue(frame)
        self.clipSplitRequested.emit()

    # --- painting -------------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(QRect(0, 0, self.width(), self.content_height()), BACKGROUND_COLOR)
        painter.fillRect(self.ruler_rect(), RULER_COLOR)
        for index in range(len(self.track_heights())):
            painter.fillRect(self.track_rect(index), TRACK_COLOR)
        if self.frame_count <= 0:
            painter.setPen(HINT_COLOR)
            painter.drawText(
                self.tracks_rect(), Qt.AlignmentFlag.AlignCenter, "Abra um vídeo para ver a linha do tempo"
            )
            return
        self._paint_ruler(painter)
        self._paint_filmstrip(painter)
        self._paint_waveform(painter)
        self._paint_tracks(painter)
        self._paint_clips(painter)
        self._paint_cuts(painter)
        self._paint_clip_markers(painter)
        self._paint_playhead(painter)
        self._paint_clip_drag(painter)
        self._paint_track_labels(painter)

    def _paint_ruler(self, painter: QPainter) -> None:
        ruler = self.ruler_rect()
        painter.setPen(QPen(RULER_BORDER_COLOR, 1))
        painter.drawLine(ruler.left(), ruler.bottom(), ruler.right(), ruler.bottom())
        major, minor = ruler_steps(self._fps, self.scale.pixels_per_frame)
        with_frames = major < round(self._fps)
        font = painter.font()
        font.setPointSizeF(max(8.0, font.pointSizeF() - 2.0))
        painter.setFont(font)
        first, last = self.visible_range()
        frame = (first // minor) * minor
        while frame <= min(last + minor, self.frame_count):
            x = self.scale.x_of(frame)
            is_major = frame % major == 0
            painter.setPen(QPen(TICK_COLOR, 1))
            tick_height = 8 if is_major else 4
            painter.drawLine(QPointF(x, ruler.bottom() - tick_height), QPointF(x, ruler.bottom()))
            if is_major:
                painter.setPen(LABEL_COLOR)
                painter.drawText(
                    QPointF(x + 3, ruler.top() + 11), ruler_label(frame, self._fps, with_frames=with_frames)
                )
            frame += minor

    def _paint_filmstrip(self, painter: QPainter) -> None:
        if not self._thumbnails:
            return
        track = self.video_track_rect()
        width, height = self.thumbnail_size()
        tolerance = width / self.scale.pixels_per_frame  # one cell of frames
        painter.setClipRect(track)
        for frame, x in self.thumbnail_slots():
            image = self.nearest_thumbnail(frame, tolerance)
            if image is not None:
                painter.drawImage(QRectF(x, track.top(), width, height), image)
        painter.setClipping(False)

    def _paint_waveform(self, painter: QPainter) -> None:
        track = self.audio_track_rect()
        painter.setPen(HINT_COLOR)
        for clip in self._clips:
            state = self._audio_states.get(clip.source, "none")
            if state == "ready":
                continue
            span = self._span_rect(track, self.scale.x_of(clip.start), self.scale.x_of(clip.end + 1))
            if span.width() >= 40:
                painter.drawText(span, Qt.AlignmentFlag.AlignCenter, AUDIO_STATE_TEXT.get(state, ""))
        columns = self.waveform_columns(0, self.width())
        if columns is None:
            return
        minimums, maximums = columns
        center = track.top() + track.height() / 2.0
        half = track.height() / 2.0 - 3.0
        scale = half * self._waveform_gain
        xs = np.arange(self.width(), dtype=np.float64) + 0.5
        upper = [QPointF(x, center - _clamp(float(value) * scale, 0.5, half)) for x, value in zip(xs, maximums)]
        lower = [
            QPointF(x, center - _clamp(float(value) * scale, -half, -0.5)) for x, value in zip(xs[::-1], minimums[::-1])
        ]
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(WAVEFORM_COLOR)
        painter.drawPolygon(QPolygonF(upper + lower))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(WAVEFORM_CENTER_COLOR, 1))
        painter.drawLine(QPointF(track.left(), center), QPointF(track.right() + 1, center))

    def _paint_tracks(self, painter: QPainter) -> None:
        if not self._show_clip:
            return
        start_x = self.scale.x_of(self._clip[0])
        end_x = self.scale.x_of(self._clip[1] + 1)
        first_x = self.scale.x_of(0)
        last_x = self.scale.x_of(self.frame_count)
        for index in range(len(self.track_heights())):
            track = self.track_rect(index)
            if index == 0 and not self._thumbnails:  # with a filmstrip the kept range is the untinted part
                painter.fillRect(self._span_rect(track, start_x, end_x), RANGE_COLOR)
            painter.fillRect(self._span_rect(track, first_x, start_x), TRACK_OUTSIDE_CLIP_COLOR)
            painter.fillRect(self._span_rect(track, end_x, last_x), TRACK_OUTSIDE_CLIP_COLOR)

    def _paint_clips(self, painter: QPainter) -> None:
        if not self._clips:
            return
        track = self.clip_track_rect()
        media = self.media_tracks_rect()
        font = painter.font()
        font.setPointSizeF(max(7.5, font.pointSizeF() - 3.0))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        for index, clip in enumerate(self._clips):
            rect = self._span_rect(track, self.scale.x_of(clip.start), self.scale.x_of(clip.end + 1))
            if rect.isEmpty():
                continue
            active = index == self._active_clip
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(CLIP_ACTIVE_COLOR if active else CLIP_COLORS[index % len(CLIP_COLORS)])
            painter.drawRect(rect.adjusted(0, 1, 0, -1))
            if clip.missing:  # hatched in red: the file is gone from where the project expects it
                painter.setBrush(QBrush(CLIP_MISSING_COLOR, Qt.BrushStyle.BDiagPattern))
                painter.drawRect(rect.adjusted(0, 1, 0, -1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(CLIP_BORDER_COLOR if active else CLIP_BOUNDARY_COLOR, 1))
            painter.drawRect(rect.adjusted(0.5, 1.5, -0.5, -1.5))
            if rect.width() >= CLIP_MIN_TEXT_WIDTH:
                prefix = "⚠ " if clip.missing else ""
                label = metrics.elidedText(
                    f"{prefix}{index + 1}. {clip.name}", Qt.TextElideMode.ElideMiddle, int(rect.width()) - 8
                )
                painter.setPen(CLIP_MISSING_TEXT_COLOR if clip.missing else CLIP_TEXT_COLOR)
                painter.drawText(
                    rect.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label
                )
            if index > 0:  # the boundary continues down the media tracks
                x = self.scale.x_of(clip.start)
                if 0 <= x <= self.width():
                    pen = QPen(CLIP_BOUNDARY_COLOR, 1)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(QPointF(x, media.top()), QPointF(x, media.bottom() + 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_clip_drag(self, painter: QPainter) -> None:
        """The dragged block follows the mouse as a ghost; a bar marks where it would be inserted."""
        drag = self._clip_drag
        if drag is None or not drag.active or not (0 <= drag.index < len(self._clips)):
            return
        clip = self._clips[drag.index]
        track = self.clip_track_rect()
        width = self.scale.x_of(clip.end + 1) - self.scale.x_of(clip.start)
        left = self.scale.x_of(clip.start) + (drag.current_x - drag.press_x)
        ghost = self._span_rect(track, left, left + width, min_width=4.0)
        if not ghost.isEmpty():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(CLIP_GHOST_COLOR)
            painter.drawRect(ghost.adjusted(0, 1, 0, -1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        boundaries = [self.scale.x_of(block.start) for block in self._clips]
        boundaries.append(self.scale.x_of(self._clips[-1].end + 1))
        x = _clamp(boundaries[drag.target], 1.0, self.width() - 1.0)
        painter.setPen(QPen(CLIP_INSERT_COLOR, 3))
        painter.drawLine(QPointF(x, track.top()), QPointF(x, self.content_height()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(CLIP_INSERT_COLOR)
        painter.drawPolygon(
            QPolygonF([QPointF(x - 5, track.top() - 1), QPointF(x + 5, track.top() - 1), QPointF(x, track.top() + 6)])
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_cuts(self, painter: QPainter) -> None:
        tracks = self.media_tracks_rect()
        for index, (start, end) in enumerate(self._cuts):
            rect = self._span_rect(tracks, self.scale.x_of(start), self.scale.x_of(end + 1), min_width=2.0)
            if rect.isEmpty():
                continue
            selected = index == self._selected_cut
            painter.fillRect(rect, CUT_SELECTED_COLOR if selected else CUT_COLOR)
            painter.setPen(QPen(CUT_EDGE_COLOR, 2 if selected else 1))
            painter.drawLine(QPointF(rect.left(), rect.top()), QPointF(rect.left(), rect.bottom()))
            painter.drawLine(QPointF(rect.right(), rect.top()), QPointF(rect.right(), rect.bottom()))
            if selected:
                painter.drawRect(rect.adjusted(1, 1, -1, -1))
        if self._pending_cut is not None:
            x = self.scale.x_of(self._pending_cut)
            pen = QPen(PENDING_CUT_COLOR, 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(x, tracks.top()), QPointF(x, tracks.bottom()))

    def _paint_clip_markers(self, painter: QPainter) -> None:
        if not self._show_clip:
            return
        ruler = self.ruler_rect()
        bottom = self.content_height()
        for color, frame, direction in (
            (START_MARKER_COLOR, self._clip[0], 1.0),
            (END_MARKER_COLOR, self._clip[1] + 1, -1.0),
        ):
            x = self.scale.x_of(frame)
            painter.setPen(QPen(color, 2))
            painter.drawLine(QPointF(x, ruler.bottom() - 8), QPointF(x, bottom))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            flag = QPolygonF(
                [
                    QPointF(x, ruler.bottom() - 9),
                    QPointF(x + 8 * direction, ruler.bottom() - 9),
                    QPointF(x, ruler.bottom() + 1),
                ]
            )
            painter.drawPolygon(flag)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_playhead(self, painter: QPainter) -> None:
        ruler = self.ruler_rect()
        x = self.scale.x_of(self._value)
        if self.scale.pixels_per_frame >= 6:
            cell = self._span_rect(self.media_tracks_rect(), x, self.scale.x_of(self._value + 1))
            painter.fillRect(cell, PLAYHEAD_CELL_COLOR)
        painter.setPen(QPen(PLAYHEAD_COLOR, 2))
        painter.drawLine(QPointF(x, ruler.top()), QPointF(x, self.content_height()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PLAYHEAD_COLOR)
        painter.drawPolygon(
            QPolygonF([QPointF(x - 6, ruler.top()), QPointF(x + 6, ruler.top()), QPointF(x, ruler.top() + 8)])
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_track_labels(self, painter: QPainter) -> None:
        font = painter.font()
        font.setPointSizeF(max(7.5, font.pointSizeF() - 3.0))
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        for index, label in enumerate(self.track_labels()):
            if not label:
                continue
            track = self.track_rect(index)
            pill = QRectF(
                track.left() + 4, track.top() + 3, metrics.horizontalAdvance(label) + 10, metrics.height() + 2
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(TRACK_LABEL_BACKGROUND)
            painter.drawRoundedRect(pill, 4, 4)
            painter.setPen(TRACK_LABEL_COLOR)
            painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, label)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def track_labels(self) -> tuple[str, ...]:
        return ("", "VÍDEO", "ÁUDIO")

    @staticmethod
    def _span_rect(track: QRect, left: float, right: float, min_width: float = 0.0) -> QRectF:
        """Pixel span ``[left, right)`` clipped to ``track``; empty when it falls outside."""
        left = max(float(track.left()), left)
        right = min(float(track.right() + 1), right)
        if right - left < min_width:
            right = left + min_width
        if right <= left or left >= track.right() + 1 or right <= track.left():
            return QRectF()
        return QRectF(left, track.top(), right - left, track.height())
