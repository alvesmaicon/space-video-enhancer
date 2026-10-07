"""Application entry point: logging, crash hooks, environment snapshot, then the window."""

from __future__ import annotations

import logging
import sys
import threading
import time
from types import TracebackType

from PySide6.QtCore import QMessageLogContext, Qt, QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

from space_video_enhancer import __version__
from space_video_enhancer.diagnostics import log_diagnostics
from space_video_enhancer.export.ffmpeg import find_ffmpeg
from space_video_enhancer.telemetry import crash
from space_video_enhancer.telemetry.environment import collect_environment
from space_video_enhancer.telemetry.log import get_logger, log_event, setup_logging
from space_video_enhancer.ui.app_icon import app_icon
from space_video_enhancer.ui.main_window import MainWindow

LOG = get_logger("app")

APPLICATION_NAME = "Space Video Enhancer"
DESKTOP_FILE_NAME = "space-video-enhancer"
"""Basename of the Linux ``.desktop`` entry; Qt also uses it as the X11 WM_CLASS and the Wayland app id."""
WINDOWS_APP_USER_MODEL_ID = "SpaceVideoEnhancer.App"

_QT_LEVELS = {
    QtMsgType.QtDebugMsg: logging.DEBUG,
    QtMsgType.QtInfoMsg: logging.INFO,
    QtMsgType.QtWarningMsg: logging.WARNING,
    QtMsgType.QtCriticalMsg: logging.ERROR,
    QtMsgType.QtFatalMsg: logging.CRITICAL,
}


def _qt_message_handler(mode: QtMsgType, context: QMessageLogContext, message: str) -> None:
    log_event(
        LOG, "qt.message", level=_QT_LEVELS.get(mode, logging.WARNING), message=message, category=context.category
    )


def install_crash_hooks() -> None:
    """Route uncaught exceptions (main and worker threads) and Qt messages into the log."""
    previous_hook = sys.excepthook

    def excepthook(exc_type: type[BaseException], value: BaseException, traceback: TracebackType | None) -> None:
        log_event(LOG, "app.unhandled", level=logging.CRITICAL, exc_info=(exc_type, value, traceback))
        previous_hook(exc_type, value, traceback)

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        log_event(
            LOG,
            "app.thread_unhandled",
            level=logging.CRITICAL,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            thread=args.thread.name if args.thread else None,
        )

    sys.excepthook = excepthook
    threading.excepthook = thread_hook
    qInstallMessageHandler(_qt_message_handler)


def configure_application(app: QApplication) -> None:
    """Give the process the app's identity instead of the interpreter's.

    Without this the window manager shows "python" and the Python icon: the icon here goes to the
    window, Alt-Tab and the dock/taskbar; the desktop-file name is what GNOME/KDE match against an
    installed ``space-video-enhancer.desktop`` (see ``scripts/install-desktop-entry.sh``) to show the
    right name and icon in the dock. On macOS the menu-bar name still comes from the executable's
    bundle — only packaging into a ``.app`` changes it — but the Dock icon does follow the window icon.

    The scheme is forced dark to match the theme: the QSS paints the widgets, but the title bar and the
    native dialogs (open/save, message boxes) follow the system's light/dark setting unless told otherwise.
    """
    app.setApplicationName(APPLICATION_NAME)
    app.setApplicationDisplayName(APPLICATION_NAME)
    app.setOrganizationName(APPLICATION_NAME)
    app.setApplicationVersion(__version__)
    app.setDesktopFileName(DESKTOP_FILE_NAME)
    app.setWindowIcon(app_icon())
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark)


def claim_windows_taskbar_identity() -> None:
    """Windows: group the window under our own taskbar id, not python.exe's, so it gets our icon.

    Must run before any window exists — the taskbar reads the id when the first window is created —
    so ``main`` calls it before creating the ``QApplication``.
    """
    if sys.platform != "win32":
        return
    import ctypes  # noqa: PLC0415  (Windows only)

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_USER_MODEL_ID)


def main() -> int:
    claim_windows_taskbar_identity()
    session = setup_logging()
    # Fatal signals (segfault/abort in Qt, OpenCV, numpy) bypass every Python hook: faulthandler writes the
    # stacks to crash.log next to app.jsonl, and the next start reports what the dead session left there.
    capture = crash.install(session.log_path.parent, session.session_id)
    install_crash_hooks()
    started = time.monotonic()
    ffmpeg_path = find_ffmpeg()
    log_event(LOG, "app.start", log_path=str(session.log_path), **collect_environment(ffmpeg_path))
    if capture.previous_report is not None:
        log_event(LOG, "app.previous_crash", level=logging.ERROR, report=capture.previous_report)
    elif capture.previous_unclean:
        log_event(LOG, "app.previous_session_unclean", level=logging.WARNING)
    # OpenCL device + ffmpeg encoders with hints; off the GUI thread so startup does not wait for ffmpeg.
    threading.Thread(target=log_diagnostics, args=(ffmpeg_path,), name="diagnostics", daemon=True).start()

    app = QApplication(sys.argv)
    configure_application(app)
    window = MainWindow()
    window.show()
    if capture.previous_session_failed:
        window.report_previous_session_failure(capture.path, capture.previous_report)
    exit_code = app.exec()

    log_event(LOG, "app.stop", exit_code=exit_code, seconds=round(time.monotonic() - started, 1))
    crash.finish(capture)  # a clean exit: the running marker goes away
    logging.shutdown()
    return exit_code
