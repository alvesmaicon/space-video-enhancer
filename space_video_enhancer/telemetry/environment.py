"""Snapshot of the runtime environment, logged once at startup."""

from __future__ import annotations

import os
import platform
import subprocess
from importlib import metadata
from typing import Any

import cv2
import numpy as np

from space_video_enhancer import __version__

FFMPEG_VERSION_TIMEOUT = 3


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def opencl_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        if hasattr(cv2, "ocl") and cv2.ocl.haveOpenCL():
            info["available"] = True
            device = cv2.ocl.Device.getDefault()
            info["device"] = device.name()
            info["vendor"] = device.vendorName()
    except cv2.error as error:
        info["error"] = str(error)
    return info


def ffmpeg_version(ffmpeg_path: str | None) -> str | None:
    if not ffmpeg_path:
        return None
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"], capture_output=True, text=True, check=False, timeout=FFMPEG_VERSION_TIMEOUT
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    return first_line.replace("ffmpeg version ", "").split(" Copyright")[0] or None


def collect_environment(ffmpeg_path: str | None = None) -> dict[str, Any]:
    return {
        "app_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pyside6": _package_version("PySide6"),
        "opencl": opencl_info(),
        "ffmpeg": ffmpeg_version(ffmpeg_path),
    }
