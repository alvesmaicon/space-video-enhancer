"""Undo/redo as a history of editing states.

Everything the user edits lives in immutable values — the ``Sequence`` (clips, their settings,
cuts, references) and the export range — so a snapshot is a handful of references and taking one
on every edit costs nothing. The history keeps the state *before* each edit (for undo) and the
states that were undone (for redo).

Edits of one kind in quick succession coalesce into a single step: the renders during a slider
drag each store the widgets into the clip, a marker drag reports every mouse move, and none of
those should be a step of its own. A commit carries a ``key`` describing what changed; while the
key stays the same and the commits keep coming within :data:`COALESCE_SECONDS` of each other the
step is extended instead of a new one being opened. :meth:`EditHistory.seal` ends the current
step early (a drag finished), and undo/redo never merge later edits into a step they moved past.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

S = TypeVar("S")

DEFAULT_LIMIT = 50
"""Undo steps kept. Every step pins a sequence (and the reference frames its clips use), so this bounds memory."""
COALESCE_SECONDS = 1.0
"""Commits with the same key closer than this extend the current step instead of opening a new one."""


@dataclass
class HistoryEntry(Generic[S]):
    state: S
    """For the undo stack: the state before the edit. For the redo stack: the state the undo left."""
    label: str
    """What the edit was, for the menu ("Desfazer dividir clipe")."""
    key: Hashable | None
    """Coalescing key of the edit that created the entry; ``None`` once the step is sealed."""
    touched_at: float


class EditHistory(Generic[S]):
    """Undo/redo stacks over snapshots of an editing state (any value with ``==``)."""

    def __init__(
        self,
        initial: S,
        *,
        limit: int = DEFAULT_LIMIT,
        coalesce_seconds: float = COALESCE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._current = initial
        self._undo: list[HistoryEntry[S]] = []
        self._redo: list[HistoryEntry[S]] = []
        self._limit = max(1, limit)
        self._coalesce_seconds = coalesce_seconds
        self._clock = clock

    @property
    def current(self) -> S:
        """The last committed state (what the editor is showing, once it caught up)."""
        return self._current

    def __len__(self) -> int:
        """Undo steps available."""
        return len(self._undo)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> str | None:
        return self._undo[-1].label if self._undo else None

    @property
    def redo_label(self) -> str | None:
        return self._redo[-1].label if self._redo else None

    def reset(self, state: S) -> None:
        """Start over from ``state`` (a new document): nothing to undo or redo."""
        self._current = state
        self._undo.clear()
        self._redo.clear()

    def commit(self, state: S, label: str, key: Hashable | None = None) -> bool:
        """Record ``state`` as the result of the edit ``label``; ``True`` when a new undo step was opened.

        An unchanged state is ignored. With a ``key`` equal to the current step's, committed within
        the coalescing window, the step is extended (its "before" state stays, ``state`` becomes the
        new "after"). Any commit discards the redo stack.
        """
        if state is self._current or state == self._current:
            return False
        now = self._clock()
        self._redo.clear()
        top = self._undo[-1] if self._undo else None
        if key is not None and top is not None and top.key == key and now - top.touched_at <= self._coalesce_seconds:
            top.touched_at = now
            self._current = state
            return False
        self._undo.append(HistoryEntry(self._current, label, key, now))
        del self._undo[: -self._limit]
        self._current = state
        return True

    def seal(self) -> None:
        """End the current step: the next commit opens a new one even with the same key (a drag finished)."""
        if self._undo:
            self._undo[-1].key = None

    def undo(self) -> tuple[S, str] | None:
        """Step back: ``(state to show, label of the undone edit)``, or ``None`` when there is nothing to undo."""
        if not self._undo:
            return None
        entry = self._undo.pop()
        self._redo.append(HistoryEntry(self._current, entry.label, None, self._clock()))
        self._current = entry.state
        self.seal()  # a later edit must not merge into a step the user already stepped past
        return entry.state, entry.label

    def redo(self) -> tuple[S, str] | None:
        """Step forward again: ``(state to show, label of the redone edit)``, or ``None``."""
        if not self._redo:
            return None
        entry = self._redo.pop()
        self._undo.append(HistoryEntry(self._current, entry.label, None, self._clock()))
        self._current = entry.state
        return entry.state, entry.label

    def states(self) -> Iterator[S]:
        """Every state the history can still return to (current first); resources they use must stay alive."""
        yield self._current
        for entry in reversed(self._undo):
            yield entry.state
        for entry in reversed(self._redo):
            yield entry.state
