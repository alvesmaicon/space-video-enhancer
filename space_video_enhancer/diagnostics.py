"""System diagnostics for GPU acceleration: OpenCL (OpenCV T-API) and the ffmpeg encoders.

Answers "why does the export say *CPU (OpenCL indisponivel)*?" without asking the user to run
commands: which OpenCL ICDs the machine exposes, whether OpenCV was built with OpenCL and sees
a runtime, which device it would use, and which encoders ffmpeg offers — each with a hint in
Portuguese saying what is missing. Logged once at startup (event ``diagnostics``, background
thread) and shown in *Ajuda -> Diagnóstico do sistema*.
"""

from __future__ import annotations

import glob
import os
import platform
import re
from dataclasses import asdict, dataclass
from typing import Any

import cv2

from space_video_enhancer import __version__
from space_video_enhancer.export.ffmpeg import FfmpegProbe
from space_video_enhancer.telemetry.environment import ffmpeg_version
from space_video_enhancer.telemetry.log import get_logger, log_event

LOG = get_logger("diagnostics")
ICD_DIRECTORIES = ("/etc/OpenCL/vendors",)
RENDER_DEVICE_GLOB = "/dev/dri/renderD*"
INTERESTING_ENCODERS = (
    "libx264",
    "h264_nvenc",
    "hevc_nvenc",
    "h264_vaapi",
    "h264_qsv",
    "h264_amf",
    "h264_videotoolbox",
)
HARDWARE_ENCODERS = tuple(name for name in INTERESTING_ENCODERS if name != "libx264")
USED_HARDWARE_ENCODERS = ("h264_nvenc", "h264_vaapi")
"""The ones :func:`export.ffmpeg.resolve_encoder` actually picks; the others are listed for information."""
OPENCL_ENV_VARS = ("OPENCV_OPENCL_RUNTIME", "OPENCV_OPENCL_DEVICE", "OCL_ICD_VENDORS", "OPENCV_OPENCL_DISABLE")


@dataclass(frozen=True)
class SystemReport:
    app_version: str
    python: str
    os_name: str
    """``platform.system()``: Linux, Darwin, Windows."""
    platform: str
    machine: str
    cpu_count: int | None
    opencv: str
    in_wsl: bool
    env: dict[str, str]
    """The OpenCL-related environment variables that are set."""


@dataclass(frozen=True)
class OpenCLReport:
    build_line: str
    """The ``OpenCL:`` line of ``cv2.getBuildInformation()``."""
    build_enabled: bool | None
    runtime_available: bool
    """``cv2.ocl.haveOpenCL()``: a runtime (ICD loader + at least one platform) answered."""
    in_use: bool
    """``cv2.ocl.useOpenCL()`` after asking for it."""
    device_name: str | None = None
    device_vendor: str | None = None
    device_version: str | None = None
    device_type: str | None = None
    device_memory_mb: int | None = None
    icd_files: tuple[str, ...] = ()
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.build_enabled is False:
            return "not_built"
        if not self.runtime_available:
            return "no_runtime"
        if not self.in_use:
            return "disabled"
        return "ok"


@dataclass(frozen=True)
class FfmpegReport:
    path: str | None
    version: str | None
    encoders: dict[str, bool]
    """Presence of each :data:`INTERESTING_ENCODERS` entry."""
    render_devices: tuple[str, ...]
    """``/dev/dri/renderD*`` (VAAPI) on Linux."""

    @property
    def hardware_encoders(self) -> tuple[str, ...]:
        return tuple(name for name in HARDWARE_ENCODERS if self.encoders.get(name))


@dataclass(frozen=True)
class Diagnostics:
    system: SystemReport
    opencl: OpenCLReport
    ffmpeg: FfmpegReport


# --- collection ---------------------------------------------------------------------------------


def system_report() -> SystemReport:
    release = platform.release().lower()
    return SystemReport(
        app_version=__version__,
        python=platform.python_version(),
        os_name=platform.system(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
        opencv=cv2.__version__,
        in_wsl="microsoft" in release or "WSL_DISTRO_NAME" in os.environ,
        env={name: os.environ[name] for name in OPENCL_ENV_VARS if name in os.environ},
    )


def opencv_build_line(build_information: str | None = None) -> str:
    text = build_information if build_information is not None else cv2.getBuildInformation()
    match = re.search(r"^\s*OpenCL:\s*(.+?)\s*$", text, re.MULTILINE)
    return match.group(1) if match else ""


def icd_files() -> tuple[str, ...]:
    found: list[str] = []
    for directory in ICD_DIRECTORIES:
        found.extend(sorted(glob.glob(os.path.join(directory, "*.icd"))))
    return tuple(found)


def _device_type_name(device: Any) -> str | None:
    try:
        kind = int(device.type())
    except (cv2.error, AttributeError, TypeError):
        return None
    for label, flag in (("GPU", cv2.ocl.Device_TYPE_GPU), ("CPU", cv2.ocl.Device_TYPE_CPU)):
        if kind & flag:
            return label
    return "acelerador" if kind & cv2.ocl.Device_TYPE_ACCELERATOR else None


def opencl_report() -> OpenCLReport:
    build_line = opencv_build_line()
    build_enabled = build_line.upper().startswith("YES") if build_line else None
    icds = icd_files()
    if not hasattr(cv2, "ocl"):
        return OpenCLReport(build_line, False, False, False, icd_files=icds)
    try:
        available = bool(cv2.ocl.haveOpenCL())
        if not available:
            return OpenCLReport(build_line, build_enabled, False, False, icd_files=icds)
        cv2.ocl.setUseOpenCL(True)
        in_use = bool(cv2.ocl.useOpenCL())
        device = cv2.ocl.Device.getDefault()
        return OpenCLReport(
            build_line,
            build_enabled,
            True,
            in_use,
            device_name=device.name() or None,
            device_vendor=device.vendorName() or None,
            device_version=device.version().strip() or None,
            device_type=_device_type_name(device),
            device_memory_mb=int(device.globalMemSize()) // (1024 * 1024) if device.globalMemSize() else None,
            icd_files=icds,
        )
    except cv2.error as error:
        return OpenCLReport(build_line, build_enabled, False, False, icd_files=icds, error=str(error).strip()[:300])


def ffmpeg_report(ffmpeg_path: str | None) -> FfmpegReport:
    devices = tuple(sorted(glob.glob(RENDER_DEVICE_GLOB)))
    if not ffmpeg_path:
        return FfmpegReport(None, None, {}, devices)
    available = FfmpegProbe(ffmpeg_path).available_encoders()
    encoders = {name: name in available for name in INTERESTING_ENCODERS}
    return FfmpegReport(ffmpeg_path, ffmpeg_version(ffmpeg_path), encoders, devices)


def collect_diagnostics(ffmpeg_path: str | None) -> Diagnostics:
    return Diagnostics(system_report(), opencl_report(), ffmpeg_report(ffmpeg_path))


# --- hints ---------------------------------------------------------------------------------------


def opencl_hint(report: OpenCLReport, system: SystemReport) -> str:
    """What to do about the OpenCL status, in the words the status bar uses."""
    if report.status == "ok":
        memory = f", {report.device_memory_mb} MB" if report.device_memory_mb else ""
        return (
            f"OpenCL pronto: {report.device_name} ({report.device_vendor}, {report.device_version}{memory}). "
            "O denoise NLMeans roda nesse dispositivo; os filtros leves e a estabilização ficam na CPU por escolha."
        )
    if report.status == "error":
        return f"O OpenCL respondeu com erro ao inicializar: {report.error}. O app segue na CPU."
    if report.status == "not_built":
        return (
            "Este OpenCV foi compilado sem OpenCL. Instale o pacote oficial `opencv-python` do PyPI "
            "(`python -m pip install --force-reinstall opencv-python`), que vem com OpenCL."
        )
    if report.status == "disabled":
        variables = ", ".join(f"{name}={value}" for name, value in system.env.items()) or "nenhuma definida"
        return (
            "OpenCL detectado, mas desativado neste processo. Verifique as variáveis de ambiente "
            f"OPENCV_OPENCL_RUNTIME / OPENCV_OPENCL_DEVICE ({variables})."
        )
    # no runtime
    if system.os_name == "Linux":
        if system.in_wsl:
            return (
                "Este Linux roda no WSL: a NVIDIA expõe CUDA, mas não OpenCL, dentro do WSL. "
                "Use o app no Windows ou num Linux nativo para ter o denoise na GPU."
            )
        if not report.icd_files:
            return (
                "Nenhum ICD OpenCL em /etc/OpenCL/vendors. Instale o loader e o ICD do driver: "
                "`sudo apt install ocl-icd-libopencl1 clinfo` e, para NVIDIA, `libnvidia-compute-<versão>` "
                "(vem com o metapacote `nvidia-driver-<versão>`); Intel: `intel-opencl-icd`; "
                "AMD: `mesa-opencl-icd` ou `rocm-opencl-runtime`. Confira com `clinfo`."
            )
        names = ", ".join(os.path.basename(path) for path in report.icd_files)
        return (
            f"Há ICDs em /etc/OpenCL/vendors ({names}), mas o OpenCV não conseguiu carregar `libOpenCL.so.1`. "
            "Instale o loader `ocl-icd-libopencl1` e confira `ldconfig -p | grep libOpenCL` e `clinfo`."
        )
    if system.os_name == "Darwin":
        return "O OpenCL do macOS não respondeu. Reinicie o app; se persistir, o export segue na CPU."
    return "Nenhum runtime OpenCL: instale ou atualize o driver da GPU (NVIDIA, AMD ou Intel), que traz o ICD."


def ffmpeg_hint(report: FfmpegReport, system: SystemReport) -> str:
    if not report.path:
        install = "`sudo apt install ffmpeg`" if system.os_name == "Linux" else "veja Requisitos no README"
        return f"FFmpeg não encontrado no PATH: o export fica indisponível ({install})."
    hardware = report.hardware_encoders
    if not hardware:
        return "Nenhum encoder de hardware (nvenc, vaapi, qsv, videotoolbox): o export codifica com libx264 na CPU."
    parts = [f"Encoders de hardware disponíveis: {', '.join(hardware)}."]
    unused = [name for name in hardware if name not in USED_HARDWARE_ENCODERS]
    if unused and not any(name in USED_HARDWARE_ENCODERS for name in hardware):
        parts.append(
            f"O app só usa NVENC e VAAPI; {', '.join(unused)} fica(m) de fora e o export codifica com libx264 na CPU."
        )
    if "h264_nvenc" in hardware:
        parts.append(
            "O export usa NVIDIA NVENC quando o teste de fumaça passa (motivo de recusa em `export.start` no log)."
        )
    if "h264_vaapi" in hardware and system.os_name == "Linux" and not report.render_devices:
        parts.append(
            "h264_vaapi existe, mas não há /dev/dri/renderD*: falta o driver VAAPI "
            "(`intel-media-va-driver-non-free` ou `mesa-va-drivers`) ou o usuário não está no grupo `render`/`video`."
        )
    return " ".join(parts)


def expected_status_label(diagnostics: Diagnostics) -> str:
    """What the export status bar will show as *backend + encoder* on this machine.

    Mirrors :func:`export.ffmpeg.resolve_encoder` (minus its smoke test): NVENC, then VAAPI with a
    render device, else the libx264 fallback.
    """
    backend = "OpenCL" if diagnostics.opencl.status == "ok" else "CPU (OpenCL indisponivel)"
    hardware = diagnostics.ffmpeg.hardware_encoders
    if "h264_nvenc" in hardware:
        encoder = "NVIDIA NVENC"
    elif "h264_vaapi" in hardware and diagnostics.ffmpeg.render_devices:
        encoder = "VAAPI"
    else:
        encoder = "libx264 (fallback)"
    return f"{backend} + {encoder}"


# --- presentation --------------------------------------------------------------------------------


def diagnostics_fields(diagnostics: Diagnostics) -> dict[str, Any]:
    """JSON-friendly summary for the ``diagnostics`` log event."""
    opencl = asdict(diagnostics.opencl)
    opencl["status"] = diagnostics.opencl.status
    opencl["icd_files"] = list(diagnostics.opencl.icd_files)
    ffmpeg = asdict(diagnostics.ffmpeg)
    ffmpeg["render_devices"] = list(diagnostics.ffmpeg.render_devices)
    return {
        "system": asdict(diagnostics.system),
        "opencl": opencl,
        "ffmpeg": ffmpeg,
        "opencl_hint": opencl_hint(diagnostics.opencl, diagnostics.system),
        "ffmpeg_hint": ffmpeg_hint(diagnostics.ffmpeg, diagnostics.system),
        "expected_status": expected_status_label(diagnostics),
    }


def diagnostics_text(diagnostics: Diagnostics) -> str:
    """The report shown in the dialog (and copied to the clipboard)."""
    system, opencl, ffmpeg = diagnostics.system, diagnostics.opencl, diagnostics.ffmpeg
    yes_no = {True: "sim", False: "não", None: "?"}
    lines = [
        f"Space Video Enhancer {system.app_version} — diagnóstico do sistema",
        "",
        "[Sistema]",
        f"  SO: {system.platform} ({system.machine}){' — WSL' if system.in_wsl else ''}",
        f"  Python {system.python} · OpenCV {system.opencv} · CPUs: {system.cpu_count}",
    ]
    if system.env:
        lines.append("  Variáveis: " + ", ".join(f"{name}={value}" for name, value in system.env.items()))
    lines += [
        "",
        "[OpenCL]",
        f"  OpenCV compilado com OpenCL: {yes_no[opencl.build_enabled]} ({opencl.build_line or 'sem linha OpenCL'})",
        f"  Runtime encontrado (haveOpenCL): {yes_no[opencl.runtime_available]} · em uso: {yes_no[opencl.in_use]}",
    ]
    if opencl.device_name:
        memory = f" · {opencl.device_memory_mb} MB" if opencl.device_memory_mb else ""
        lines.append(
            f"  Dispositivo: {opencl.device_name} ({opencl.device_vendor}, {opencl.device_type or '?'}, "
            f"{opencl.device_version}{memory})"
        )
    if system.os_name == "Linux":
        lines.append("  ICDs em /etc/OpenCL/vendors: " + (", ".join(opencl.icd_files) or "nenhum"))
    if opencl.error:
        lines.append(f"  Erro: {opencl.error}")
    lines += [f"  → {opencl_hint(opencl, system)}", "", "[FFmpeg]"]
    if ffmpeg.path:
        lines.append(f"  {ffmpeg.path} — versão {ffmpeg.version or '?'}")
        present = [name for name, ok in ffmpeg.encoders.items() if ok]
        missing = [name for name, ok in ffmpeg.encoders.items() if not ok]
        lines.append(f"  Encoders presentes: {', '.join(present) or 'nenhum dos conhecidos'}")
        lines.append(f"  Encoders ausentes: {', '.join(missing) or '—'}")
        if system.os_name == "Linux":
            lines.append("  Dispositivos VAAPI: " + (", ".join(ffmpeg.render_devices) or "nenhum"))
    else:
        lines.append("  não encontrado no PATH")
    lines += [
        f"  → {ffmpeg_hint(ffmpeg, system)}",
        "",
        f"[Export] a barra de status deve mostrar: {expected_status_label(diagnostics)}",
    ]
    return "\n".join(lines)


def log_diagnostics(ffmpeg_path: str | None) -> Diagnostics:
    """Collect and log the ``diagnostics`` event (meant for a background thread at startup)."""
    diagnostics = collect_diagnostics(ffmpeg_path)
    log_event(LOG, "diagnostics", **diagnostics_fields(diagnostics))
    return diagnostics
