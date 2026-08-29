"""Project files (``.svep``): the sequence, the export range, the playhead and the export threads, as JSON.

A project is everything that defines the work — the clips (file, trimmed range, cuts, per-clip
``RenderSettings``, reference frame indices), the I/O markers, the playhead and the export
thread count — written as readable JSON. Settings are stored field by field with enums as their
values, so a project written by another version still opens: missing fields take the dataclass
defaults and unknown ones are ignored (:func:`settings_from_dict` infers each field's type from
its default). Media is stored as the absolute path plus the path relative to the project file; a
file that is not where it was is restored from the saved facts (frames, fps, size) and listed in
:attr:`Project.missing` — the window marks such clips, it does not relink them. Reference frames
are not stored as pixels: only their indices are, and the window decodes them again on open.

The list of recently opened projects lives next to the logs (``~/.space-video-enhancer``, or
``SVE_STATE_DIR``).
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from space_video_enhancer.sequence import Clip, ClipSource, Sequence
from space_video_enhancer.settings import RenderSettings
from space_video_enhancer.timeline import FrameRange

PROJECT_VERSION = 1
PROJECT_SUFFIX = ".svep"
PROJECT_KIND = "space-video-enhancer-project"
PROJECT_FILE_FILTER = "Projeto Space Video Enhancer (*.svep)"
DEFAULT_THREADS = 4
MAX_RECENT_PROJECTS = 8
ENV_STATE_DIR = "SVE_STATE_DIR"
DEFAULT_STATE_DIR = Path.home() / ".space-video-enhancer"


class ProjectError(ValueError):
    """The file is not a project this version can read."""


@dataclass(frozen=True)
class Project:
    sequence: Sequence
    clip_range: tuple[int, int]
    """Export range (I/O markers) in timeline frames, inclusive."""
    playhead: int = 0
    threads: int = DEFAULT_THREADS
    missing: tuple[Path, ...] = ()
    """Sources whose file was not found when loading; their facts came from the project itself."""


# --- settings <-> plain data ------------------------------------------------------------------
def settings_to_dict(value: Any) -> Any:
    """Dataclasses become dicts, enums their values, tuples lists, paths strings (recursively)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: settings_to_dict(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [settings_to_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def settings_from_dict(cls: type, data: Any) -> Any:
    """Rebuild the dataclass ``cls`` from ``data``: missing fields keep their defaults, unknown keys are ignored.

    Field types are inferred from the defaults (enum, nested dataclass, tuple, number), which is
    what lets a project written by an older or newer version load.
    """
    if not isinstance(data, Mapping):
        raise ProjectError(f"{cls.__name__}: esperava um objeto, veio {type(data).__name__}")
    defaults = _field_defaults(cls)
    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name in data:
            kwargs[field.name] = _coerce(data[field.name], defaults.get(field.name), f"{cls.__name__}.{field.name}")
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as error:
        raise ProjectError(f"{cls.__name__}: {error}") from error


def _field_defaults(cls: type) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            defaults[field.name] = field.default_factory()
    return defaults


def _coerce(value: Any, default: Any, where: str) -> Any:
    """``value`` (plain JSON data) shaped like ``default``."""
    if value is None:
        return None
    try:
        if isinstance(default, Enum):
            return type(default)(value)
        if dataclasses.is_dataclass(default) and not isinstance(default, type):
            return settings_from_dict(type(default), value)
        if isinstance(default, tuple) or (default is None and isinstance(value, list)):
            element = default[0] if isinstance(default, tuple) and default else None
            if dataclasses.is_dataclass(element) and not isinstance(element, type):
                return tuple(settings_from_dict(type(element), item) for item in value)
            return tuple(value)
        if isinstance(default, bool):
            return bool(value)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return str(value)
    except (TypeError, ValueError) as error:
        raise ProjectError(f"{where}: valor inválido {value!r}") from error
    return value


# --- project <-> plain data -------------------------------------------------------------------
def project_to_dict(project: Project, base_dir: Path | None = None) -> dict[str, Any]:
    """The JSON-ready form of ``project``; media paths are also stored relative to ``base_dir``."""
    sources = list(dict.fromkeys(clip.source for clip in project.sequence.clips))
    index = {source: position for position, source in enumerate(sources)}
    return {
        "kind": PROJECT_KIND,
        "version": PROJECT_VERSION,
        "sources": [
            {
                "path": str(source.path),
                "relative": _relative_path(source.path, base_dir),
                "frame_count": source.frame_count,
                "fps": source.fps,
                "width": source.width,
                "height": source.height,
            }
            for source in sources
        ],
        "clips": [
            {
                "source": index[clip.source],
                "range": [clip.range.start, clip.range.end],
                "cuts": [[cut.start, cut.end] for cut in clip.cuts],
                "settings": settings_to_dict(clip.settings),
                "stabilization_reference_frame": clip.stabilization_reference_frame,
                "exposure_reference_frame": clip.exposure_reference_frame,
            }
            for clip in project.sequence.clips
        ],
        "range": [project.clip_range[0], project.clip_range[1]],
        "playhead": project.playhead,
        "threads": project.threads,
        "timeline_fps": project.sequence.timeline_fps,
    }


def _relative_path(path: Path, base_dir: Path | None) -> str | None:
    if base_dir is None:
        return None
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:  # another drive on Windows
        return None


def project_from_dict(
    data: Any, base_dir: Path | None = None, *, exists: Callable[[Path], bool] = Path.is_file
) -> Project:
    """Rebuild a :class:`Project`; sources are looked up by absolute path, then relative to ``base_dir``."""
    if not isinstance(data, Mapping) or data.get("kind") != PROJECT_KIND:
        raise ProjectError("não é um arquivo de projeto do Space Video Enhancer")
    version = data.get("version")
    if not isinstance(version, int) or version < 1:
        raise ProjectError("versão do projeto ausente ou inválida")
    if version > PROJECT_VERSION:
        raise ProjectError(f"projeto na versão {version}; esta versão do programa lê até a {PROJECT_VERSION}")
    try:
        return _read_project(data, base_dir, exists)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        if isinstance(error, ProjectError):
            raise
        raise ProjectError(f"projeto malformado: {error}") from error


def _read_project(data: Mapping[str, Any], base_dir: Path | None, exists: Callable[[Path], bool]) -> Project:
    sources: list[ClipSource] = []
    missing: list[Path] = []
    for entry in data.get("sources", []):
        path, found = _resolve_source(entry, base_dir, exists)
        sources.append(
            ClipSource(
                path,
                int(entry["frame_count"]),
                float(entry["fps"]),
                int(entry.get("width", 0)),
                int(entry.get("height", 0)),
            )
        )
        if not found:
            missing.append(path)
    clips = []
    for entry in data.get("clips", []):
        source = sources[int(entry["source"])]
        start, end = entry.get("range", [0, source.last_frame])
        cuts = tuple(FrameRange(int(cut[0]), int(cut[1])) for cut in entry.get("cuts", []))
        settings = settings_from_dict(RenderSettings, entry.get("settings", {}))
        clips.append(
            Clip(
                source,
                FrameRange(int(start), int(end)),
                cuts,
                settings,
                _optional_int(entry.get("stabilization_reference_frame")),
                _optional_int(entry.get("exposure_reference_frame")),
            )
        )
    if not clips:
        raise ProjectError("o projeto não tem clipes")
    timeline_fps_raw = data.get("timeline_fps")
    timeline_fps = float(timeline_fps_raw) if timeline_fps_raw is not None else None
    sequence = Sequence(tuple(clips), timeline_fps=timeline_fps)
    last = sequence.timeline_count - 1
    range_start, range_end = data.get("range", [0, last])
    clip_range = (max(0, min(int(range_start), last)), max(0, min(int(range_end), last)))
    playhead = max(0, min(int(data.get("playhead", 0)), last))
    threads = max(1, int(data.get("threads", DEFAULT_THREADS)))
    return Project(sequence, (min(clip_range), max(clip_range)), playhead, threads, tuple(missing))


def _resolve_source(
    entry: Mapping[str, Any], base_dir: Path | None, exists: Callable[[Path], bool]
) -> tuple[Path, bool]:
    absolute = Path(str(entry["path"]))
    if exists(absolute):
        return absolute, True
    relative = entry.get("relative")
    if relative and base_dir is not None:
        candidate = Path(os.path.normpath(base_dir / str(relative)))
        if exists(candidate):
            return candidate, True
    return absolute, False


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def with_sources(sequence: Sequence, sources: Mapping[Path, ClipSource]) -> Sequence:
    """``sequence`` with every clip of a file in ``sources`` pointing at that (freshly probed) source.

    Ranges and cuts are clamped by the clip itself, so a file that got shorter still loads.
    """
    clips = tuple(
        replace(clip, source=sources[clip.source.path]) if clip.source.path in sources else clip
        for clip in sequence.clips
    )
    return Sequence(clips)


# --- files -----------------------------------------------------------------------------------
def save_project(project: Project, path: Path) -> None:
    text = json.dumps(project_to_dict(project, path.parent), indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def load_project(path: Path) -> Project:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProjectError(f"JSON inválido: {error}") from error
    return project_from_dict(data, path.parent)


# --- recent projects ------------------------------------------------------------------------
def state_dir() -> Path:
    """Where per-user state lives (``SVE_STATE_DIR`` overrides the default next to the logs)."""
    return Path(os.environ.get(ENV_STATE_DIR, str(DEFAULT_STATE_DIR)))


UI_STATE_FILE = "ui_state.json"


def _read_ui_state() -> dict[str, Any]:
    try:
        data = json.loads((state_dir() / UI_STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_ui_state(**updates: Any) -> None:
    state = {**_read_ui_state(), **updates}
    try:
        path = state_dir() / UI_STATE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
    except OSError:  # a read-only state dir must never break the file dialogs
        pass


def last_media_dir() -> str | None:
    """The folder the last video was opened from, if it still exists; else ``None`` (fall back to home)."""
    value = _read_ui_state().get("media_dir")
    return value if isinstance(value, str) and Path(value).is_dir() else None


def remember_media_dir(folder: Path) -> None:
    """Remember where the user last browsed for a video, so the next import opens there."""
    _write_ui_state(media_dir=str(folder))


class RecentProjects:
    """The most recently opened/saved project files, newest first, kept in a small JSON file."""

    def __init__(self, path: Path | None = None, limit: int = MAX_RECENT_PROJECTS) -> None:
        self._path = path
        self._limit = limit

    @property
    def path(self) -> Path:
        return self._path if self._path is not None else state_dir() / "recent_projects.json"

    @property
    def paths(self) -> list[Path]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(data, list):
            return []
        return [Path(item) for item in data if isinstance(item, str)][: self._limit]

    def add(self, path: Path) -> None:
        self._write([path, *(item for item in self.paths if item != path)][: self._limit])

    def remove(self, path: Path) -> None:
        self._write([item for item in self.paths if item != path])

    def clear(self) -> None:
        self._write([])

    def _write(self, paths: list[Path]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([str(item) for item in paths], indent=2), encoding="utf-8")
        except OSError:
            pass  # a recents list that cannot be written is not worth an error dialog
