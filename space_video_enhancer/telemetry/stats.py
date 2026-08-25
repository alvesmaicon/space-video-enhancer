"""Small rolling aggregates for HUD display and end-of-job summaries."""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

import numpy as np

from space_video_enhancer.telemetry.trace import RenderTrace


class RollingStats:
    """Keeps the last ``window`` samples and answers mean / percentiles / max."""

    def __init__(self, window: int = 200) -> None:
        self._values: deque[float] = deque(maxlen=window)
        self.count = 0

    def add(self, value: float) -> None:
        self._values.append(float(value))
        self.count += 1

    def __len__(self) -> int:
        return len(self._values)

    @property
    def last(self) -> float | None:
        return self._values[-1] if self._values else None

    @property
    def mean(self) -> float:
        return float(np.mean(self._values)) if self._values else 0.0

    @property
    def max(self) -> float:
        return float(max(self._values)) if self._values else 0.0

    def percentile(self, percent: float) -> float:
        return float(np.percentile(self._values, percent)) if self._values else 0.0

    def summary(self, digits: int = 1) -> dict[str, float | int]:
        if not self._values:
            return {"count": self.count}
        return {
            "count": self.count,
            "mean": round(self.mean, digits),
            "p50": round(self.percentile(50), digits),
            "p95": round(self.percentile(95), digits),
            "max": round(self.max, digits),
        }


class StageStats:
    """Aggregates :class:`RenderTrace` timings across many renders."""

    def __init__(self, window: int = 2000) -> None:
        self._window = window
        self.total = RollingStats(window)
        self.stages: dict[str, RollingStats] = {}

    def add(self, trace: RenderTrace) -> None:
        self.total.add(trace.total_ms)
        for name, value in trace.stages.items():
            self.stages.setdefault(name, RollingStats(self._window)).add(value)

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total.summary(),
            "stages": {name: stats.summary() for name, stats in self.stages.items()},
        }


class Histogram:
    """Counts categorical outcomes (e.g. which stabilization estimator was used)."""

    def __init__(self) -> None:
        self._counter: Counter[str] = Counter()

    def add(self, key: object) -> None:
        if key is not None:
            self._counter[str(key)] += 1

    def as_dict(self) -> dict[str, int]:
        return dict(self._counter.most_common())

    def __len__(self) -> int:
        return sum(self._counter.values())
