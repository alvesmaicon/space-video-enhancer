"""Capture of fatal crashes — the ones no ``except`` ever sees — and detection of a session that never said goodbye.

A segfault or abort inside Qt, OpenCV or numpy kills the process before ``sys.excepthook`` runs; the
window just vanishes. :mod:`faulthandler` is the tool for that: on SIGSEGV/SIGABRT/SIGBUS/SIGFPE/SIGILL
it writes the Python stack of every thread straight to a file descriptor, signal-safe. Each session
appends a header line to ``crash.log`` in the log folder and points ``faulthandler`` at it, so a dump
lands right under the header of the session that died.

The next start reads what follows the last header: a traceback means the previous session crashed
and the report is handed back to the app (logged as ``app.previous_crash`` and shown in the status
bar). A ``session.running`` marker, written at start and removed at a clean exit, tells a kill or a
power cut apart from a crash with a report.
"""

from __future__ import annotations

import contextlib
import faulthandler
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

CRASH_LOG_NAME = "crash.log"
RUNNING_MARKER_NAME = "session.running"
HEADER_PREFIX = "=== session "
MAX_CRASH_LOG_BYTES = 1_000_000
KEEP_TAIL_BYTES = 200_000
"""When crash.log grows past the maximum, only this much of its tail is kept."""
REPORT_LIMIT = 6_000
"""Characters of the previous report handed to the app (the file keeps everything)."""


@dataclass
class CrashCapture:
    path: Path
    """``crash.log``: one header per session, faulthandler dumps under the session that died."""
    marker: Path
    previous_report: str | None
    """Traceback the previous session left behind, or ``None`` when it ended cleanly (or left no dump)."""
    previous_unclean: bool
    """The previous session did not remove its running marker: crash, kill or power loss."""
    _file: TextIO

    @property
    def previous_session_failed(self) -> bool:
        return self.previous_report is not None or self.previous_unclean


def install(log_dir: Path, session_id: str) -> CrashCapture:
    """Start capturing fatal crashes into ``log_dir/crash.log``; report what the previous session left."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / CRASH_LOG_NAME
    marker = log_dir / RUNNING_MARKER_NAME
    _trim(path)
    previous_report = last_session_report(path)
    previous_unclean = marker.exists()
    marker.write_text(f"{session_id} {time.strftime('%Y-%m-%dT%H:%M:%S')} pid {os.getpid()}\n", encoding="utf-8")
    file = path.open("a", encoding="utf-8", buffering=1)
    file.write(f"{HEADER_PREFIX}{session_id} started {time.strftime('%Y-%m-%dT%H:%M:%S')} pid {os.getpid()} ===\n")
    file.flush()
    faulthandler.enable(file=file, all_threads=True)
    return CrashCapture(path, marker, previous_report, previous_unclean, file)


def finish(capture: CrashCapture) -> None:
    """A clean exit: stop capturing and drop the running marker (the header stays, with nothing under it)."""
    faulthandler.disable()
    capture._file.close()
    with contextlib.suppress(FileNotFoundError):
        capture.marker.unlink()


def last_session_report(path: Path) -> str | None:
    """Whatever follows the last session header in ``path`` — a faulthandler dump — or ``None``."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    start = text.rfind(HEADER_PREFIX)
    if start < 0:
        return None
    body_start = text.find("\n", start)
    body = text[body_start + 1 :].strip() if body_start >= 0 else ""
    return body[-REPORT_LIMIT:] if body else None


def _trim(path: Path) -> None:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return
    if size <= MAX_CRASH_LOG_BYTES:
        return
    data = path.read_bytes()[-KEEP_TAIL_BYTES:]
    cut = data.find(b"\n")  # start on a whole line
    path.write_bytes(data[cut + 1 :] if cut >= 0 else data)
