"""Per-render trace: stage timings plus free-form notes recorded by the pipeline.

The pipeline never receives a trace object explicitly below ``render_frame``: deep code
(estimators, filters) calls :func:`stage` / :func:`note`, which write into the trace that
the caller activated for the current thread via :func:`active`. When nothing is active
the calls are no-ops, so instrumentation costs nothing in untraced paths.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RenderTrace:
    stages: dict[str, float] = field(default_factory=dict)
    """Milliseconds per stage, in execution order.

    Nested stages ("filters.denoise") are also counted inside their parent ("filters").
    """
    notes: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)
    total_ms: float = 0.0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - started) * 1000.0

    def note(self, key: str, value: Any) -> None:
        self.notes[key] = value

    def finish(self) -> RenderTrace:
        self.total_ms = (time.perf_counter() - self.started_at) * 1000.0
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_ms": round(self.total_ms, 2),
            "stages": {name: round(value, 2) for name, value in self.stages.items()},
            **self.notes,
        }


_current: ContextVar[RenderTrace | None] = ContextVar("sve_render_trace", default=None)


def current() -> RenderTrace | None:
    return _current.get()


@contextmanager
def active(trace: RenderTrace | None) -> Iterator[RenderTrace | None]:
    """Make ``trace`` the target of :func:`stage`/:func:`note` on this thread. ``None`` is a no-op."""
    if trace is None:
        yield None
        return
    token = _current.set(trace)
    try:
        yield trace
    finally:
        _current.reset(token)


@contextmanager
def stage(name: str) -> Iterator[None]:
    trace = _current.get()
    if trace is None:
        yield
        return
    with trace.stage(name):
        yield


def note(key: str, value: Any) -> None:
    trace = _current.get()
    if trace is not None:
        trace.note(key, value)
