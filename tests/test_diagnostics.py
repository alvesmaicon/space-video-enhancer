"""System diagnostics: OpenCL/ffmpeg reports, hints per situation and the dialog."""

from __future__ import annotations

import json
import shutil

import pytest

from space_video_enhancer.diagnostics import (
    Diagnostics,
    FfmpegReport,
    OpenCLReport,
    SystemReport,
    collect_diagnostics,
    diagnostics_fields,
    diagnostics_text,
    expected_status_label,
    ffmpeg_hint,
    ffmpeg_report,
    opencl_hint,
    opencl_report,
    opencv_build_line,
)


def _system(os_name: str = "Linux", *, wsl: bool = False, env: dict[str, str] | None = None) -> SystemReport:
    return SystemReport("0.0", "3.9", os_name, os_name, "x86_64", 12, "4.10", wsl, env or {})


def _opencl(**overrides) -> OpenCLReport:  # type: ignore[no-untyped-def]
    values = {
        "build_line": "YES (no extra features)",
        "build_enabled": True,
        "runtime_available": False,
        "in_use": False,
    }
    values.update(overrides)
    return OpenCLReport(**values)


def test_build_line_is_read_from_the_build_information() -> None:
    assert opencv_build_line(
        "  Other:  x\n  OpenCL:                        YES (no extra features)\n    Include path: NO\n"
    ) == ("YES (no extra features)")
    assert opencv_build_line("nothing here") == ""
    assert opencv_build_line() != "", "the installed OpenCV reports its OpenCL build flag"


def test_live_opencl_report_is_consistent() -> None:
    report = opencl_report()
    assert report.status in ("ok", "no_runtime", "not_built", "disabled", "error")
    if report.status == "ok":
        assert report.runtime_available and report.in_use and report.device_name
    else:
        assert not (report.runtime_available and report.in_use and report.build_enabled and not report.error)
    assert opencl_hint(report, _system("Darwin")), "every status has a hint"


def test_opencl_hints_name_what_is_missing() -> None:
    linux = _system("Linux")
    assert "ocl-icd-libopencl1" in opencl_hint(_opencl(), linux) and "clinfo" in opencl_hint(_opencl(), linux)
    with_icd = _opencl(icd_files=("/etc/OpenCL/vendors/nvidia.icd",))
    assert "nvidia.icd" in opencl_hint(with_icd, linux) and "libOpenCL.so.1" in opencl_hint(with_icd, linux)
    assert "WSL" in opencl_hint(_opencl(), _system("Linux", wsl=True))
    assert "opencv-python" in opencl_hint(_opencl(build_line="NO", build_enabled=False), linux)
    disabled = _opencl(runtime_available=True, in_use=False)
    assert "OPENCV_OPENCL_RUNTIME" in opencl_hint(disabled, _system(env={"OPENCV_OPENCL_RUNTIME": "disabled"}))
    ok = _opencl(
        runtime_available=True,
        in_use=True,
        device_name="NVIDIA GeForce RTX 4060",
        device_vendor="NVIDIA",
        device_version="OpenCL 3.0",
        device_memory_mb=8188,
    )
    assert ok.status == "ok" and "RTX 4060" in opencl_hint(ok, linux) and "8188 MB" in opencl_hint(ok, linux)
    assert "driver" in opencl_hint(_opencl(), _system("Windows"))
    assert "macOS" in opencl_hint(_opencl(), _system("Darwin"))
    errored = _opencl(error="clGetPlatformIDs failed")
    assert errored.status == "error" and "clGetPlatformIDs" in opencl_hint(errored, linux)


def test_ffmpeg_hints_and_expected_status_label() -> None:
    linux = _system("Linux")
    missing = FfmpegReport(None, None, {}, ())
    assert "apt install ffmpeg" in ffmpeg_hint(missing, linux)
    cpu_only = FfmpegReport("/usr/bin/ffmpeg", "6.1", {"libx264": True, "h264_nvenc": False}, ())
    assert "libx264" in ffmpeg_hint(cpu_only, linux) and cpu_only.hardware_encoders == ()
    nvenc = FfmpegReport("/usr/bin/ffmpeg", "6.1", {"libx264": True, "h264_nvenc": True, "h264_vaapi": True}, ())
    hint = ffmpeg_hint(nvenc, linux)
    assert "NVENC" in hint and "renderD" in hint, "vaapi listed without a render device gets its own warning"
    no_runtime = Diagnostics(linux, _opencl(), nvenc)
    assert expected_status_label(no_runtime) == "CPU (OpenCL indisponivel) + NVIDIA NVENC"
    ready = Diagnostics(linux, _opencl(runtime_available=True, in_use=True, device_name="RTX"), nvenc)
    assert expected_status_label(ready) == "OpenCL + NVIDIA NVENC"
    vaapi = Diagnostics(linux, _opencl(), FfmpegReport("f", "6", {"h264_vaapi": True}, ("/dev/dri/renderD128",)))
    assert expected_status_label(vaapi).endswith("+ VAAPI")
    assert expected_status_label(Diagnostics(linux, _opencl(), cpu_only)).endswith("+ libx264 (fallback)")
    mac = FfmpegReport("/opt/homebrew/bin/ffmpeg", "8.1", {"libx264": True, "h264_videotoolbox": True}, ())
    assert expected_status_label(Diagnostics(_system("Darwin"), _opencl(), mac)).endswith("+ libx264 (fallback)")
    assert "só usa NVENC e VAAPI" in ffmpeg_hint(mac, _system("Darwin"))


def test_live_ffmpeg_report() -> None:
    absent = ffmpeg_report(None)
    assert absent.path is None and absent.encoders == {} and absent.hardware_encoders == ()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    report = ffmpeg_report(ffmpeg)
    assert report.path == ffmpeg and report.version and "libx264" in report.encoders


def test_full_report_is_loggable_and_readable() -> None:
    diagnostics = collect_diagnostics(shutil.which("ffmpeg"))
    fields = diagnostics_fields(diagnostics)
    json.dumps(fields)  # the log event must serialize
    assert fields["opencl"]["status"] == diagnostics.opencl.status and "expected_status" in fields
    text = diagnostics_text(diagnostics)
    assert "[OpenCL]" in text and "[FFmpeg]" in text and "barra de status deve mostrar" in text
    assert fields["expected_status"] in text
