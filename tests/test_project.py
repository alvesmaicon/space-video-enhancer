"""Project files: round trip of the sequence, tolerance to other versions, missing media, recents."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from space_video_enhancer.project import (
    PROJECT_KIND,
    PROJECT_VERSION,
    Project,
    ProjectError,
    RecentProjects,
    last_media_dir,
    load_project,
    project_from_dict,
    project_to_dict,
    remember_media_dir,
    save_project,
    settings_from_dict,
    settings_to_dict,
    with_sources,
)
from space_video_enhancer.sequence import Clip, ClipSource, Sequence
from space_video_enhancer.settings import (
    ColorGradeSettings,
    CropSettings,
    DenoiseMode,
    ExposureLockSettings,
    FilterSettings,
    GeometrySettings,
    HueBandAdjust,
    NormalizedRect,
    RenderSettings,
    RotationMode,
    StabilizationKeyframe,
    StabilizationMode,
    StabilizationSettings,
)
from space_video_enhancer.timeline import FrameRange


def _settings() -> RenderSettings:
    bands = list(RenderSettings().grade.bands)
    bands[2] = HueBandAdjust(hue_shift=-12, saturation=20, luminance=-5)
    return RenderSettings(
        filters=FilterSettings(denoise_mode=DenoiseMode.SAMPLE, denoise_strength=33, contrast=61, scale_factor=1.5),
        geometry=GeometrySettings(rotation=RotationMode.CW_90, flip_horizontal=True),
        crop=CropSettings(enabled=True, rect=NormalizedRect(0.1, 0.2, 0.5, 0.6), aspect_ratio=16 / 9),
        stabilization=StabilizationSettings(
            mode=StabilizationMode.HYBRID,
            focus_rect=NormalizedRect(0.3, 0.3, 0.2, 0.2),
            keyframes=(
                StabilizationKeyframe(0, tx=0.02, ty=-0.01),
                StabilizationKeyframe(30, tx=-0.03, rotation=1.5, scale=1.02),
            ),
        ),
        grade=ColorGradeSettings(tint=42, white_balance_gains=(1.1, 1.0, 0.9), bands=tuple(bands)),
        exposure=ExposureLockSettings(enabled=True, region=None, reference=(100.5, 110.25, 120.0), max_ev=1.5),
    )


def _project(tmp_path: Path) -> Project:
    (tmp_path / "a.mp4").write_bytes(b"a")
    (tmp_path / "b.mp4").write_bytes(b"b")
    source_a = ClipSource(tmp_path / "a.mp4", 240, 24.0, 1280, 720)
    source_b = ClipSource(tmp_path / "b.mp4", 96, 24.0, 640, 480)
    clips = (
        Clip(source_a, FrameRange(0, 99), (FrameRange(10, 20),), _settings(), 5, 7),
        Clip(source_a, FrameRange(100, 239), (), RenderSettings()),
        Clip(source_b, FrameRange(0, 95), (FrameRange(90, 95),), RenderSettings(), None, 3),
    )
    return Project(Sequence(clips), (12, 300), playhead=150, threads=6)


def test_settings_round_trip_through_plain_data() -> None:
    settings = _settings()
    data = settings_to_dict(settings)
    assert data["filters"]["denoise_mode"] == "sample" and data["geometry"]["rotation"] == RotationMode.CW_90.value
    assert data["exposure"]["reference"] == [100.5, 110.25, 120.0] and data["exposure"]["region"] is None
    assert json.loads(json.dumps(data)) == data, "plain JSON, nothing exotic"
    keyframe = data["stabilization"]["keyframes"][1]
    assert keyframe == {
        "frame": 30,
        "tx": -0.03,
        "ty": 0.0,
        "rotation": 1.5,
        "scale": 1.02,
        "shear": 0.0,
        "stretch_y": 1.0,
    }
    loaded = settings_from_dict(RenderSettings, json.loads(json.dumps(data)))
    assert loaded == settings
    assert isinstance(loaded.stabilization.keyframes[0], StabilizationKeyframe), "not left as a raw dict"


def test_settings_from_other_versions_take_defaults_and_ignore_unknown_fields() -> None:
    data = settings_to_dict(_settings())
    del data["stabilization"]["temporal_smoothing"]  # an older file without the field
    del data["grade"]
    data["filters"]["future_knob"] = 3  # a newer file with a field this version does not know
    data["brand_new_section"] = {"x": 1}
    loaded = settings_from_dict(RenderSettings, data)
    assert loaded.stabilization.mode is StabilizationMode.HYBRID
    assert loaded.stabilization.temporal_smoothing == StabilizationSettings().temporal_smoothing
    assert loaded.grade == ColorGradeSettings()
    assert loaded.filters.denoise_strength == 33

    with pytest.raises(ProjectError):
        settings_from_dict(RenderSettings, {"filters": {"denoise_mode": "no-such-mode"}})
    with pytest.raises(ProjectError):
        settings_from_dict(RenderSettings, ["not", "a", "mapping"])


def test_project_round_trips_through_a_file(tmp_path: Path) -> None:
    project = _project(tmp_path)
    path = tmp_path / "edit.svep"
    save_project(project, path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    assert data["kind"] == PROJECT_KIND and data["version"] == PROJECT_VERSION
    assert [source["relative"] for source in data["sources"]] == ["a.mp4", "b.mp4"], "relative to the project file"
    assert len(data["sources"]) == 2 and [clip["source"] for clip in data["clips"]] == [0, 0, 1]

    loaded = load_project(path)
    assert loaded.sequence == project.sequence
    assert (loaded.clip_range, loaded.playhead, loaded.threads, loaded.missing) == ((12, 300), 150, 6, ())


def test_media_is_found_next_to_a_moved_project_or_reported_missing(tmp_path: Path) -> None:
    project = _project(tmp_path)
    data = project_to_dict(project, tmp_path)
    elsewhere = tmp_path / "moved"
    elsewhere.mkdir()
    (elsewhere / "a.mp4").write_bytes(b"a")
    # The absolute paths point at files that no longer exist; a.mp4 sits next to the project, b.mp4 is gone.
    (tmp_path / "a.mp4").unlink()
    (tmp_path / "b.mp4").unlink()
    loaded = project_from_dict(data, elsewhere)
    paths = [clip.source.path for clip in loaded.sequence.clips]
    assert paths[0] == elsewhere / "a.mp4" and paths[1] == elsewhere / "a.mp4"
    assert paths[2] == tmp_path / "b.mp4" and loaded.missing == (tmp_path / "b.mp4",)
    assert loaded.sequence.clips[2].source.frame_count == 96, "facts of a missing file come from the project"
    assert loaded.sequence.clips[0].settings == _settings()


def test_with_sources_swaps_in_probed_facts_and_clamps(tmp_path: Path) -> None:
    project = _project(tmp_path)
    shorter = ClipSource(tmp_path / "a.mp4", 150, 24.0, 1280, 720)  # the file lost frames since the project was saved
    updated = with_sources(project.sequence, {shorter.path: shorter})
    assert updated.clips[0].source is shorter and updated.clips[1].range == FrameRange(100, 149)
    assert updated.clips[2].source == project.sequence.clips[2].source, "other files untouched"


def test_malformed_and_foreign_files_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.svep"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProjectError, match="JSON"):
        load_project(path)
    with pytest.raises(ProjectError, match="não é um arquivo de projeto"):
        project_from_dict({"kind": "something-else", "version": 1})
    with pytest.raises(ProjectError, match="versão"):
        project_from_dict({"kind": PROJECT_KIND, "version": PROJECT_VERSION + 1, "sources": [], "clips": []})
    with pytest.raises(ProjectError, match="clipes"):
        project_from_dict({"kind": PROJECT_KIND, "version": 1, "sources": [], "clips": []})
    with pytest.raises(ProjectError, match="malformado"):
        project_from_dict({"kind": PROJECT_KIND, "version": 1, "sources": [], "clips": [{"source": 3}]})


def test_range_and_playhead_are_clamped_to_the_sequence(tmp_path: Path) -> None:
    project = replace(_project(tmp_path), clip_range=(0, 10_000), playhead=-4)
    data = project_to_dict(project, tmp_path)
    loaded = project_from_dict(data, tmp_path)
    assert loaded.clip_range == (0, loaded.sequence.timeline_count - 1) and loaded.playhead == 0


def test_recent_projects_keep_newest_first_without_duplicates(tmp_path: Path) -> None:
    recent = RecentProjects(tmp_path / "state" / "recent.json", limit=3)
    assert recent.paths == []
    for name in ("a", "b", "c", "a", "d"):
        recent.add(tmp_path / f"{name}.svep")
    assert [path.name for path in recent.paths] == ["d.svep", "a.svep", "c.svep"]
    recent.remove(tmp_path / "a.svep")
    assert [path.name for path in recent.paths] == ["d.svep", "c.svep"]
    recent.clear()
    assert recent.paths == []


def test_timeline_fps_is_saved_and_restored(tmp_path: Path) -> None:
    (tmp_path / "a.mp4").write_bytes(b"a")
    (tmp_path / "b.mp4").write_bytes(b"b")
    a = ClipSource(tmp_path / "a.mp4", 240, 24.0, 1280, 720)
    b = ClipSource(tmp_path / "b.mp4", 96, 48.0, 640, 480)  # a different fps clip on the same timeline
    sequence = Sequence((Clip.whole(a), Clip.whole(b)), timeline_fps=24.0)
    project = Project(sequence, (0, sequence.last_timeline_frame), playhead=0, threads=1)

    data = project_from_dict(project_to_dict(project, tmp_path), tmp_path)
    assert data.sequence.timeline_fps == 24.0 and data.sequence == sequence

    legacy = project_to_dict(project, tmp_path)
    del legacy["timeline_fps"]  # a project saved before the field existed
    assert project_from_dict(legacy, tmp_path).sequence.timeline_fps is None


def test_last_media_dir_round_trips_and_ignores_a_folder_that_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SVE_STATE_DIR", str(tmp_path / "state"))
    assert last_media_dir() is None  # nothing remembered yet
    folder = tmp_path / "videos"
    folder.mkdir()
    remember_media_dir(folder)
    assert last_media_dir() == str(folder)
    remember_media_dir(tmp_path / "gone")  # a folder that does not exist is not offered back
    assert last_media_dir() is None
