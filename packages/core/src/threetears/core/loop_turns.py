"""Work on a worker thread that takes turns with the caller's event loop.

CPU work moved off the event loop onto a thread still holds the interpreter lock while it runs Python: the loop's
thread asks for the lock and waits CPython's switch interval (5 ms by default) for it, and on a loaded host a waiting
thread is often not the one handed it, so the loop fell behind by several intervals at a time while the thread ran. A
thread that hands the loop a turn whenever it has held the lock :data:`HOLD_SECONDS` keeps every stretch of the loop's
wait under a millisecond, whatever the switch interval (:class:`LoopTurns`). Its per-item loops take a turn every
:data:`STEP_ITEMS` items (:func:`stepped`).

Born in the Arrow copy's build (:mod:`threetears.core.backends.arrow_copy`); shared since a bulk write's per-key work
(its keys checked and enrolled, its L1 fence keys made) ran on threads too, and at 33,786 keys held the loop 15 ms at a
time under the default switch interval (the WD2 bench, 2026-11-16).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterable, Iterator
from typing import Any, Final, TypeVar

__all__ = ["HOLD_SECONDS", "STEP_ITEMS", "LoopTurns", "emptied", "stepped"]

_Item = TypeVar("_Item")

#: the longest the build holds the interpreter lock before it lets the caller's event loop run: measured, a 12 MB copy's
#: build held the caller's loop 31 ms behind on a thread that never blocked (a waiting thread is not handed the lock
#: while the holder keeps releasing and taking it back in pyarrow's calls). With one slice past it (:data:`STEP_ITEMS`)
#: a stretch stays under 1 ms: on a loaded host the OS can stop the build while it holds the lock, and the caller's
#: loop is late by that stop only when its tick falls inside a stretch, so the shorter the stretches the rarer that is
#: (1 ms stretches: 47.5 ms worst, p99 4.7 ms of added lateness on a Linux container with every CPU busy)
HOLD_SECONDS: Final = 0.0005

#: how long the build blocks when it lets the loop run and has no loop to hand the turn to
_TURN_SECONDS: Final = 0.0002

#: the longest the build waits for the caller's loop to take a turn it was handed: a loop that has not run by then is
#: busy with work of its own, not waiting for the interpreter lock, and the build goes on
_TURN_WAIT_SECONDS: Final = 0.05

#: the items a per-item loop on the thread runs between two looks at its turn (:func:`stepped`): as the Arrow copy's
#: build reads values (``_SLICE_VALUES``), few enough that no stretch nears :data:`HOLD_SECONDS`
STEP_ITEMS: Final = 128


class LoopTurns:
    """the build's turns with the caller's event loop: past :data:`HOLD_SECONDS` of holding the interpreter lock, the
    build hands the loop a turn and blocks until the loop has run.

    A turn is handed, not offered: the build schedules a callback on the loop and waits for it to run. A sleep offered
    the lock for :data:`_TURN_SECONDS` only, and on a loaded host the loop's thread was often not scheduled within it:
    the build took the lock back, and the loop waited for CPython's 5 ms switch interval, turn after turn (pool
    diagnostics: the loop 8-20 ms late in ``select`` while the build ran ``_sliced_array`` / ``_column_of``). With no
    loop (a build called off any loop), the turn is the sleep.
    """

    __slots__ = ("_handed", "_loop", "_since")

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._since = time.monotonic()
        self._loop = loop
        self._handed = threading.Event()

    def take(self) -> None:
        """let the caller's loop run, when the build has held the lock its share.

        :return: nothing
        :rtype: None
        """
        if time.monotonic() - self._since >= HOLD_SECONDS:
            self.hand()

    def hand(self) -> None:
        """let the caller's loop run now.

        :return: nothing
        :rtype: None
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            time.sleep(_TURN_SECONDS)
        else:
            self._handed.clear()
            loop.call_soon_threadsafe(self._handed.set)
            self._handed.wait(_TURN_WAIT_SECONDS)
        self._since = time.monotonic()


def stepped(items: Iterable[_Item], turns: LoopTurns, every: int = STEP_ITEMS) -> Iterator[_Item]:
    """``items``, the thread taking its turn (:meth:`LoopTurns.take`) every ``every`` of them.

    :param items: the items
    :ptype items: Iterable[_Item]
    :param turns: the thread's turns with the caller's loop
    :ptype turns: LoopTurns
    :param every: the items between two looks at the turn
    :ptype every: int
    :return: the items, in order
    :rtype: Iterator[_Item]
    """
    for index, item in enumerate(items):
        if index % every == 0:
            turns.take()
        yield item


def emptied(lists: Iterable[list[Any]], loop: asyncio.AbstractEventLoop | None = None) -> None:
    """empty each of ``lists`` in place, a slice of :data:`STEP_ITEMS` at a time with the thread's turns between: what
    they alone hold is freed here, on the thread, and never in one stretch. Blocking: run it on a thread.

    A write's last reference to its rows' values dropped where it happened to: a bulk upsert's 33,786 rows' parameters
    (179 a row), freed as the call returned, held the loop 10-15 ms (the WD2 bench, 2026-11-16).

    :param lists: the lists
    :ptype lists: Iterable[list[Any]]
    :param loop: the caller's event loop, handed the turns; None off any loop
    :ptype loop: asyncio.AbstractEventLoop | None
    :return: nothing
    :rtype: None
    """
    turns = LoopTurns(loop)
    for each in lists:
        while each:
            del each[-STEP_ITEMS:]
            turns.take()
