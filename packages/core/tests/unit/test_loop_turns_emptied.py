"""``loop_turns.emptied`` frees sets and dicts as it frees lists: in place, on its thread, taking turns with the loop.

A set or dict dropped by its last reference is one deallocation in C that holds the interpreter lock throughout (a
state's keys, a set of 100k tuples, in the reports pod), so the loop waits on it whichever thread drops it. ``emptied``
took lists only; a consumer emptying a set or dict wrote its own copy (the reports pod's ``emptied_unordered``). These
count the turns, never the clock: a set or dict of ``n`` items looks at its turn once every
:data:`~threetears.core.loop_turns.STEP_ITEMS` items, as a list does once a slice.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

import threetears.core.loop_turns as loop_turns
from threetears.core.loop_turns import STEP_ITEMS, LoopTurns, emptied, emptied_off_loop

#: items enough for several slices
_ITEMS = STEP_ITEMS * 5 + 3


@pytest.fixture
def looks(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """the number of looks at the turn, counted by each call of :meth:`LoopTurns.take`."""
    counted: list[int] = []
    taken = LoopTurns.take

    def counting(self: LoopTurns) -> None:
        counted.append(1)
        taken(self)

    monkeypatch.setattr(loop_turns.LoopTurns, "take", counting)
    return counted


def _expected_looks(items: int) -> int:
    return -(-items // STEP_ITEMS)


def test_a_set_is_emptied_in_place_a_turn_every_step(looks: list[int]) -> None:
    """Flips if ``emptied`` refuses a set, leaves any of it, or frees it without looking at its turn every
    ``STEP_ITEMS`` items."""
    keys: set[Any] = {(index, str(index)) for index in range(_ITEMS)}
    emptied([keys])
    assert keys == set()
    assert len(looks) == _expected_looks(_ITEMS)


def test_a_dict_is_emptied_in_place_a_turn_every_step(looks: list[int]) -> None:
    """Flips if ``emptied`` refuses a dict, leaves any of it, or frees it without looking at its turn every
    ``STEP_ITEMS`` items."""
    rows: dict[Any, Any] = {index: {"value": index} for index in range(_ITEMS)}
    emptied([rows])
    assert rows == {}
    assert len(looks) == _expected_looks(_ITEMS)


def test_lists_sets_and_dicts_share_one_call_and_its_turns(looks: list[int]) -> None:
    """Flips if a list's slices change, or a mixed call skips one of its collections."""
    rows: list[Any] = list(range(_ITEMS))
    keys: set[Any] = set(range(_ITEMS))
    held: dict[Any, Any] = dict.fromkeys(range(_ITEMS))
    emptied([rows, keys, held])
    assert (rows, keys, held) == ([], set(), {})
    assert len(looks) == 3 * _expected_looks(_ITEMS)


def test_emptied_off_loop_takes_sets_and_dicts_on_a_thread(looks: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if ``emptied_off_loop`` refuses a set or dict, or empties it on the loop's own thread."""
    keys: set[Any] = set(range(_ITEMS))
    held: dict[Any, Any] = dict.fromkeys(range(_ITEMS))
    threads: list[threading.Thread] = []
    real = loop_turns.emptied

    def watched(collections: list[Any], loop: asyncio.AbstractEventLoop | None = None) -> None:
        threads.append(threading.current_thread())
        real(collections, loop)

    monkeypatch.setattr(loop_turns, "emptied", watched)
    asyncio.run(emptied_off_loop([keys, held]))
    assert (keys, held) == (set(), {})
    assert len(threads) == 1
    assert threads[0] is not threading.main_thread()
    assert len(looks) == 2 * _expected_looks(_ITEMS)


def test_an_empty_collection_takes_no_turn(looks: list[int]) -> None:
    """Flips if an empty set or dict costs a turn (a hand of the loop for nothing)."""
    emptied([set(), {}, []])
    assert looks == []
