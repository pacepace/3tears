"""How a process's collection key reads were answered, by table and tier.

A key read (:meth:`~threetears.core.collections.base.BaseCollection.ensure`, and every read built on it) is
answered by L1 (this process), L2 (NATS KV), or L3 (the store). The aim of the tiers is that reads stay at
L1, and a table whose reads keep reaching L2 or L3 is one whose cache is being lost: dropped, evicted, or
never filled. These counters make that visible per table, cheaply: one dictionary increment per read,
under a lock, since reads also run on worker threads.

:func:`_log_tier_reads` writes the counts since it was last called as one INFO line, when any read happened,
and starts them again; :func:`_log_tier_reads_every` does that on a timer.

The counters are the process's, and one timer logs them, owned by the event loop the process serves on:
:func:`hold_tier_reads_timer` starts it on the running loop for that loop's first holder, and
:func:`release_tier_reads_timer` stops it, and logs the last counts, when the loop's last holder lets go. Two tool
servers in one process run on its one loop, so they share one line a minute rather than each draining the other's reads
into a line of its own. A timer belongs to its loop because its task does: the first holder on another loop starts a
timer of its own, so a loop that has gone takes its timer and its holders with it, and never leaves a later loop
believing a timer runs.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections import Counter
from enum import StrEnum
from typing import Final

from threetears.observe import get_logger
from threetears.observe.background import cancel_and_wait

__all__ = ["ReadTier", "hold_tier_reads_timer", "record_tier_read", "release_tier_reads_timer"]

log = get_logger(__name__)

#: how often a long-lived process logs its counts
_TIER_READS_LOG_SECONDS: Final = 60.0


class ReadTier(StrEnum):
    """the tier that answered a key read.

    :cvar L1: this process's own cache
    :cvar L2: the shared cache (NATS KV)
    :cvar L3: the store, past both caches; a read that found no row counts here too
    """

    L1 = "l1"
    L2 = "l2"
    L3 = "l3"


_counts: Counter[tuple[str, str]] = Counter()
_lock = threading.Lock()


def record_tier_read(table: str, tier: ReadTier) -> None:
    """count one key read of ``table`` answered by ``tier``.

    :param table: the table read
    :ptype table: str
    :param tier: the tier that answered
    :ptype tier: ReadTier
    :return: nothing
    :rtype: None
    """
    with _lock:
        _counts[(table, tier.value)] += 1


def _take_tier_reads() -> dict[str, dict[str, int]]:
    """the counts since the last take, by table then tier, and start them again.

    :return: table -> tier -> reads; empty when no read happened
    :rtype: dict[str, dict[str, int]]
    """
    with _lock:
        taken = dict(_counts)
        _counts.clear()
    by_table: dict[str, dict[str, int]] = {}
    for (table, tier), reads in sorted(taken.items()):
        by_table.setdefault(table, {})[tier] = reads
    return by_table


def _log_tier_reads() -> dict[str, dict[str, int]]:
    """log the counts since the last take as one INFO line, when any read happened, and start them again.

    :return: what was logged: table -> tier -> reads; empty when nothing was
    :rtype: dict[str, dict[str, int]]
    """
    taken = _take_tier_reads()
    if taken:
        log.info("collection reads by tier", extra={"extra_data": {"reads": taken}})
    return taken


async def _log_tier_reads_every(every: float = _TIER_READS_LOG_SECONDS) -> None:
    """log the counts every ``every`` seconds (:func:`_log_tier_reads`), until cancelled.

    :param every: seconds between lines
    :ptype every: float
    :return: nothing; runs until cancelled
    :rtype: None
    """
    while True:
        await asyncio.sleep(every)
        _log_tier_reads()


class _LoopTimer:
    """the :func:`_log_tier_reads_every` task, the loop it runs on, and how many holders on that loop keep it running."""

    def __init__(self) -> None:
        self.loop: weakref.ref[asyncio.AbstractEventLoop] | None = None
        self.task: asyncio.Task[None] | None = None
        self.holders = 0

    def on(self, loop: asyncio.AbstractEventLoop) -> bool:
        """whether this is ``loop``'s timer.

        :param loop: the running loop
        :ptype loop: asyncio.AbstractEventLoop
        :return: True when the timer was started on ``loop``
        :rtype: bool
        """
        return self.loop is not None and self.loop() is loop


_timer = _LoopTimer()


def hold_tier_reads_timer(every: float | None = None) -> None:
    """hold the running loop's timer, starting it when this is the loop's first holder.

    :param every: seconds between lines, :data:`_TIER_READS_LOG_SECONDS` when None; the first holder's value is the
        one the timer runs with
    :ptype every: float | None
    :return: nothing
    :rtype: None
    """
    loop = asyncio.get_running_loop()
    if not _timer.on(loop):
        # the first holder on this loop: a timer on another loop is that loop's, and goes with it
        _timer.loop, _timer.task, _timer.holders = weakref.ref(loop), None, 0
    _timer.holders += 1
    if _timer.task is None:
        _timer.task = asyncio.create_task(_log_tier_reads_every(_TIER_READS_LOG_SECONDS if every is None else every))


async def release_tier_reads_timer() -> dict[str, dict[str, int]]:
    """let go of the running loop's timer; its last holder stops it and logs the counts since its last line.

    :return: what the last holder logged: table -> tier -> reads; empty when another holder keeps the timer running,
        or no read happened since the last line
    :rtype: dict[str, dict[str, int]]
    :raises RuntimeError: nothing on the running loop holds the timer
    """
    if not _timer.on(asyncio.get_running_loop()) or _timer.holders == 0:
        raise RuntimeError("the tier-read timer was released more times than it was held")
    _timer.holders -= 1
    if _timer.holders > 0 or _timer.task is None:
        return {}
    task, _timer.task = _timer.task, None
    await cancel_and_wait(task)
    return _log_tier_reads()
