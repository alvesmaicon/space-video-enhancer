"""Undo/redo history: steps, coalescing of quick same-kind edits, sealing, limits and reachable states."""

from __future__ import annotations

from space_video_enhancer.ui.history import EditHistory


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make(initial: str = "a", **kwargs: object) -> tuple[EditHistory[str], FakeClock]:
    clock = FakeClock()
    return EditHistory(initial, clock=clock, **kwargs), clock  # type: ignore[arg-type]


def test_commit_undo_redo_walk_the_states_with_their_labels() -> None:
    history, _clock = make()
    assert not history.can_undo and not history.can_redo and history.undo_label is None
    assert history.commit("b", "dividir clipe") and history.commit("c", "remover clipe")
    assert len(history) == 2 and history.current == "c" and history.undo_label == "remover clipe"

    assert history.undo() == ("b", "remover clipe")
    assert history.current == "b" and history.redo_label == "remover clipe" and history.undo_label == "dividir clipe"
    assert history.undo() == ("a", "dividir clipe")
    assert history.undo() is None and not history.can_undo
    assert history.redo() == ("b", "dividir clipe")
    assert history.redo() == ("c", "remover clipe")
    assert history.redo() is None and history.current == "c"


def test_unchanged_state_is_not_a_step_and_a_new_edit_drops_the_redo_stack() -> None:
    history, _clock = make()
    assert not history.commit("a", "nada")
    history.commit("b", "um")
    history.commit("c", "dois")
    history.undo()
    assert history.can_redo
    assert history.commit("d", "três")
    assert not history.can_redo and history.undo_label == "três" and history.undo() == ("b", "três")


def test_same_key_within_the_window_extends_the_step_instead_of_opening_one() -> None:
    history, clock = make(coalesce_seconds=1.0)
    assert history.commit("b", "ajuste", ("brightness",))
    clock.now += 0.3
    assert not history.commit("c", "ajuste", ("brightness",)), "a slider drag: one render after another"
    clock.now += 0.3
    assert not history.commit("d", "ajuste", ("brightness",))
    assert len(history) == 1 and history.current == "d"
    assert history.undo() == ("a", "ajuste"), "undo goes back to before the whole drag"
    assert history.redo() == ("d", "ajuste"), "redo lands on the last value of the drag"

    clock.now += 0.3
    assert history.commit("e", "ajuste", ("contrast",)), "another field is another step"
    clock.now += 1.5
    assert history.commit("f", "ajuste", ("contrast",)), "the same field after a pause is another step too"
    assert len(history) == 3


def test_seal_ends_the_current_step_and_undo_never_merges_into_an_older_one() -> None:
    history, clock = make()
    history.commit("b", "corte", "drag")
    history.seal()
    clock.now += 0.1
    assert history.commit("c", "corte", "drag"), "a second drag right after the first is its own step"
    assert len(history) == 2

    history.undo()
    clock.now += 0.1
    assert history.commit("d", "corte", "drag"), "after an undo the surviving step is closed"
    assert history.undo() == ("b", "corte") and history.undo() == ("a", "corte")


def test_key_none_never_coalesces() -> None:
    history, _clock = make()
    assert history.commit("b", "marcar") and history.commit("c", "marcar")
    assert len(history) == 2


def test_limit_drops_the_oldest_steps() -> None:
    history, clock = make(limit=3)
    for index, state in enumerate("bcdef"):
        clock.now += 10
        history.commit(state, f"passo {index}")
    assert len(history) == 3
    assert [history.undo() for _ in range(3)] == [("e", "passo 4"), ("d", "passo 3"), ("c", "passo 2")]
    assert history.undo() is None and history.current == "c"


def test_states_lists_everything_reachable_and_reset_forgets_it() -> None:
    history, clock = make()
    for state in "bcd":
        clock.now += 10
        history.commit(state, state)
    history.undo()
    assert list(history.states()) == ["c", "b", "a", "d"]
    history.reset("z")
    assert list(history.states()) == ["z"] and not history.can_undo and not history.can_redo
