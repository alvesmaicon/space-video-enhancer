"""Structured JSON-lines logging.

One event per line in ``~/.space-video-enhancer/logs/app.jsonl`` (rotated at 10 MB x 5).
Every line carries ``ts``, ``level``, ``logger``, ``event``, ``session`` and ``thread``
plus the event's own fields, so the file can be grepped, loaded with pandas or shipped
to Elasticsearch/Kibana as-is.

Environment overrides: ``SVE_LOG_DIR``, ``SVE_LOG_LEVEL`` (default INFO; DEBUG adds
per-frame events during playback) and ``SVE_LOG_STDERR=1`` to mirror everything to stderr.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "sve"
LOG_FILE_NAME = "app.jsonl"
DEFAULT_LOG_DIR = Path.home() / ".space-video-enhancer" / "logs"
ENV_LOG_DIR = "SVE_LOG_DIR"
ENV_LOG_LEVEL = "SVE_LOG_LEVEL"
ENV_LOG_STDERR = "SVE_LOG_STDERR"
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5
_CORE_KEYS = ("ts", "level", "logger", "event", "session", "thread")


@dataclass(frozen=True)
class LoggingSession:
    session_id: str
    log_path: Path
    level: int


_session: LoggingSession | None = None


def json_default(value: Any) -> Any:
    """Make numpy scalars, enums, paths, dataclasses and containers JSON-friendly."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"ndarray": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


class JsonLinesFormatter(logging.Formatter):
    def __init__(self, session_id: str) -> None:
        super().__init__()
        self._session_id = session_id

    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", None) or {}
        payload: dict[str, Any] = {key: value for key, value in fields.items() if key not in _CORE_KEYS}
        payload.update(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                "level": record.levelname,
                "logger": record.name,
                "event": getattr(record, "event", None) or record.getMessage(),
                "session": self._session_id,
                "thread": record.threadName,
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=json_default, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event", None) or record.getMessage()
        fields = getattr(record, "fields", None) or {}
        text = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name} {event}"
        if fields:
            text += " " + json.dumps(fields, default=json_default, ensure_ascii=False)
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return text


def setup_logging(
    *,
    log_dir: Path | None = None,
    level: str | int | None = None,
    stderr: bool | None = None,
) -> LoggingSession:
    """Configure the ``sve`` logger tree. Idempotent: calling again replaces the handlers."""
    global _session  # noqa: PLW0603 - single process-wide logging session by design
    directory = Path(log_dir or os.environ.get(ENV_LOG_DIR) or DEFAULT_LOG_DIR).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    resolved_level = _resolve_level(level if level is not None else os.environ.get(ENV_LOG_LEVEL))
    mirror_stderr = stderr if stderr is not None else os.environ.get(ENV_LOG_STDERR, "") in ("1", "true", "yes")
    session_id = uuid.uuid4().hex[:8]

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolved_level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    log_path = directory / LOG_FILE_NAME
    file_handler = RotatingFileHandler(log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
    file_handler.setFormatter(JsonLinesFormatter(session_id))
    logger.addHandler(file_handler)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(resolved_level if mirror_stderr else logging.WARNING)
    stderr_handler.setFormatter(HumanFormatter())
    logger.addHandler(stderr_handler)

    _session = LoggingSession(session_id, log_path, resolved_level)
    return _session


def current_session() -> LoggingSession | None:
    return _session


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def log_event(
    logger: logging.Logger,
    event: str,
    /,
    level: int = logging.INFO,
    exc_info: Any = None,
    **fields: Any,
) -> None:
    """Emit one structured event. ``fields`` become top-level JSON keys."""
    if not logger.isEnabledFor(level):
        return
    logger.log(level, event, extra={"event": event, "fields": fields}, exc_info=exc_info)


def _resolve_level(level: str | int | None) -> int:
    if level is None or level == "":
        return logging.INFO
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).upper())
    return resolved if isinstance(resolved, int) else logging.INFO
