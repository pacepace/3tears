"""How a process's collection key reads were answered, by table and tier.

A key read (:meth:`~threetears.core.collections.base.BaseCollection.ensure`, and every read built on it) is
answered by L1 (this process), L2 (NATS KV), or L3 (the store). The aim of the tiers is that reads stay at
L1, and a table whose reads keep reaching L2 or L3 is one whose cache is being lost: dropped, evicted, or
never filled. These counters make that visible per table, cheaply: one dictionary increment per read,
under a lock, since reads also run on worker threads.

:func:`log_tier_reads` writes the counts since it was last called as one INFO line, when any read happened,
and starts them again; :func:`log_tier_reads_every` does that on a timer.

The counters are the process's, so the process has one timer, and one owner of it: :func:`hold_tier_reads_timer`
starts it for the first holder and :func:`release_tier_reads_timer` stops it, and logs the last counts, when the
last holder lets go. Two tool servers in one process therefore share one line a minute, rather than each draining
the other's reads into a line of its own.
"""

from __future__ import annotations

import asyncio
import threading
from collections import Counter
from enum import StrEnum
from typing import Final

from threetears.observe import get_logger
from threetears.observe.background import cancel_and_wait

__all__ = [
    "TIER_READS_LOG_SECONDS",
    "ReadTier",
    "hold_tier_reads_timer",
    "log_tier_reads",
    "log_tier_reads_every",
    "record_tier_read",
    "release_tier_reads_timer",
]

log = get_logger(__name__)

#: how often a long-lived process logs its counts
TIER_READS_LOG_SECONDS: Final = 60.0


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


def log_tier_reads() -> dict[str, dict[str, int]]:
    """log the counts since the last take as one INFO line, when any read happened, and start them again.

    :return: what was logged: table -> tier -> reads; empty when nothing was
    :rtype: dict[str, dict[str, int]]
    """
    taken = _take_tier_reads()
    if taken:
        log.info("collection reads by tier", extra={"extra_data": {"reads": taken}})
    return taken


async def log_tier_reads_every(every: float = TIER_READS_LOG_SECONDS) -> None:
    """log the counts every ``every`` seconds (:func:`log_tier_reads`), until cancelled.

    :param every: seconds between lines
    :ptype every: float
    :return: nothing; runs until cancelled
    :rtype: None
    """
    while True:
        await asyncio.sleep(every)
        log_tier_reads()


class _ProcessTimer:
    """the process's one :func:`log_tier_reads_every` task, and how many holders keep it running."""

    def __init__(self) -> None:
        self.task: asyncio.Task[None] | None = None
        self.holders = 0


_timer = _ProcessTimer()


def hold_tier_reads_timer(every: float | None = None) -> None:
    """hold the process's timer, starting it on the running loop when this is the first holder.

    :param every: seconds between lines, :data:`TIER_READS_LOG_SECONDS` when None; the first holder's value is the
        one the timer runs with
    :ptype every: float | None
    :return: nothing
    :rtype: None
    """
    _timer.holders += 1
    if _timer.task is None:
        _timer.task = asyncio.create_task(log_tier_reads_every(TIER_READS_LOG_SECONDS if every is None else every))


async def release_tier_reads_timer() -> None:
    """let go of the process's timer; the last holder stops it and logs the counts since its last line.

    :return: nothing, once the timer has stopped when this was the last holder
    :rtype: None
    :raises RuntimeError: nothing holds the timer
    """
    if _timer.holders == 0:
        raise RuntimeError("the tier-read timer was released more times than it was held")
    _timer.holders -= 1
    if _timer.holders > 0 or _timer.task is None:
        return
    task, _timer.task = _timer.task, None
    await cancel_and_wait(task)
    log_tier_reads()
