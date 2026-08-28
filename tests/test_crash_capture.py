"""Fatal-crash capture: faulthandler dumps land under the session header and are reported at the next start."""

from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
from pathlib import Path

from space_video_enhancer.telemetry import crash


def test_clean_sessions_leave_no_report(tmp_path: Path) -> None:
    first = crash.install(tmp_path, "aaaa0001")
    assert first.previous_report is None and not first.previous_unclean and first.marker.is_file()
    assert faulthandler.is_enabled()
    crash.finish(first)
    assert not first.marker.exists() and not faulthandler.is_enabled()
    text = first.path.read_text(encoding="utf-8")
    assert text.startswith("=== session aaaa0001 started") and text.count("=== session") == 1

    second = crash.install(tmp_path, "bbbb0002")
    assert second.previous_report is None and not second.previous_unclean and not second.previous_session_failed
    crash.finish(second)


def test_a_dump_without_a_clean_exit_is_reported_once(tmp_path: Path) -> None:
    dying = crash.install(tmp_path, "cccc0003")
    faulthandler.dump_traceback(file=dying._file, all_threads=False)  # what a fatal signal would write
    dying._file.close()  # the process died: no finish(), the marker stays
    faulthandler.disable()

    survivor = crash.install(tmp_path, "dddd0004")
    assert survivor.previous_unclean and survivor.previous_session_failed
    assert survivor.previous_report is not None
    assert "test_a_dump_without_a_clean_exit_is_reported_once" in survivor.previous_report
    assert "=== session" not in survivor.previous_report, "only the body under the last header"
    crash.finish(survivor)

    third = crash.install(tmp_path, "eeee0005")
    assert third.previous_report is None and not third.previous_unclean, "reported once, then the slate is clean"
    crash.finish(third)


def test_a_kill_without_a_dump_is_flagged_but_has_no_report(tmp_path: Path) -> None:
    killed = crash.install(tmp_path, "ffff0006")
    killed._file.close()
    faulthandler.disable()
    after = crash.install(tmp_path, "0000aaaa")
    assert after.previous_unclean and after.previous_report is None and after.previous_session_failed
    crash.finish(after)


def test_the_log_is_trimmed_to_its_tail_when_it_grows(tmp_path: Path) -> None:
    path = tmp_path / crash.CRASH_LOG_NAME
    path.write_text("x" * (crash.MAX_CRASH_LOG_BYTES + 10) + "\n=== session old started ===\nkept\n", encoding="utf-8")
    capture = crash.install(tmp_path, "abcd1234")
    text = path.read_text(encoding="utf-8")
    assert len(text) < crash.KEEP_TAIL_BYTES + 200 and text.endswith(
        "=== session abcd1234 started" + text.split("=== session abcd1234 started")[1]
    )
    assert capture.previous_report == "kept", "the report survives the trim"
    crash.finish(capture)


def test_a_real_abort_in_a_child_process_is_captured(tmp_path: Path) -> None:
    script = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from space_video_enhancer.telemetry import crash\n"
        "crash.install(Path(sys.argv[1]), 'child001')\n"
        "def deep(): os.abort()\n"
        "deep()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, cwd=os.getcwd(), check=False
    )
    assert result.returncode != 0
    text = (tmp_path / crash.CRASH_LOG_NAME).read_text(encoding="utf-8")
    assert "Fatal Python error: Aborted" in text and "deep" in text
    assert (tmp_path / crash.RUNNING_MARKER_NAME).exists(), "the marker outlives the crashed process"
    survivor = crash.install(tmp_path, "parent01")
    assert survivor.previous_report is not None and "Aborted" in survivor.previous_report
    crash.finish(survivor)
