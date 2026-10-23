"""How a process's collection key reads were answered, by table and tier.

A key read (:meth:`~threetears.core.collections.base.BaseCollection.ensure`, and every read built on it) is
answered by L1 (this process), L2 (NATS KV), or L3 (the store). The aim of the tiers is that reads stay at
L1, and a table whose reads keep reaching L2 or L3 is one whose cache is being lost: dropped, evicted, or
never filled. These counters make that visible per table, cheaply: one dictionary increment per read,
under a lock, since reads also run on worker threads.

:func:`take_tier_reads` returns the counts since it was last called and starts them again;
:func:`log_tier_reads` writes them as one INFO line when any read happened. A long-lived process calls it on
a timer (the tool server does, once a minute).
"""

from __future__ import annotations

import threading
from collections import Counter
from enum import StrEnum
from typing import Final

from threetears.observe import get_logger

__all__ = ["TIER_READS_LOG_SECONDS", "ReadTier", "log_tier_reads", "record_tier_read", "take_tier_reads"]

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


def take_tier_reads() -> dict[str, dict[str, int]]:
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
    taken = take_tier_reads()
    if taken:
        log.info("collection reads by tier", extra={"extra_data": {"reads": taken}})
    return taken
