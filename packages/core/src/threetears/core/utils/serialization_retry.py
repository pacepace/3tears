"""run a unit of statements again when YugabyteDB answers it with a transaction conflict it asks the client to retry:
a serialization conflict (SQLSTATE ``40001``) or a deadlock (``40P01``).

YugabyteDB answers a write that meets a concurrently committed one with ``SerializationError`` ("conflict with
concurrently committed data", kConflict) and asks the client to retry. Under READ COMMITTED it retries a statement
itself, but not always: the message says when it could not ("query layer retry isn't possible, ... some data was
already sent to the user"), and the conflict reaches the caller. PostgreSQL under READ COMMITTED raises no such
error for a plain write, so on PostgreSQL the unit runs once.

A deadlock is the same race met another way. One YugabyteDB statement writes its row and each of its index entries to
their own tablets at once, so two racing inserts of the same row can each take some of its keys first and wait on the
other for the rest; YugabyteDB's deadlock detector aborts one ("aborted to break a deadlock cycle", ``40P01``, its
query layer unable to retry it, as for the conflict above). ``NamespaceCollection.ensure_namespace``'s racing inserts
met it 113 times in 23,280 rounds of eight racers, its row's key and six index tablets in the cycle (2026-11-26).
PostgreSQL inserts a row's keys one at a time in one order, and its losers wait on the first, so it never deadlocks
there. PostgreSQL documents a deadlock failure as one to retry like a serialization failure (13.4, "Serialization
Failure Handling"); the attempt was aborted, so retrying it is safe on the same terms.

The unit is re-run whole, so it must be safe to run again: idempotent, or reading again what it decides on (a
lookup before an insert). A conflict or a deadlock aborts the statement's transaction, so a failed attempt left
nothing written.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import asyncpg

__all__ = ["retry_serialization_conflicts"]

#: what YugabyteDB answers a racing write with and asks its client to retry: SQLSTATE 40001 and 40P01 (the module)
_RETRYABLE: tuple[type[asyncpg.PostgresError], ...] = (asyncpg.SerializationError, asyncpg.DeadlockDetectedError)

_T = TypeVar("_T")


async def retry_serialization_conflicts(
    attempt: Callable[[], Awaitable[_T]], *, attempts: int, first_wait: float
) -> _T:
    """``attempt()``'s answer, run again while it raises ``asyncpg.SerializationError`` or
    ``asyncpg.DeadlockDetectedError``; any other error, or the conflict on the last attempt, is raised.

    :param attempt: the unit, re-runnable (see the module)
    :ptype attempt: Callable[[], Awaitable[_T]]
    :param attempts: the runs tried, at least one
    :ptype attempts: int
    :param first_wait: seconds before the first retry, doubled for each after
    :ptype first_wait: float
    :return: what the first attempt that did not conflict answered
    :rtype: _T
    :raises asyncpg.SerializationError: when every attempt conflicts (the last one's error)
    :raises asyncpg.DeadlockDetectedError: when every attempt conflicts and the last was a deadlock
    :raises ValueError: when ``attempts`` is under one
    """
    if attempts < 1:
        raise ValueError(f"a unit is tried at least once, not {attempts!r} times")
    wait = first_wait
    for tried in range(1, attempts + 1):
        try:
            return await attempt()
        except _RETRYABLE:
            if tried == attempts:
                raise
            await asyncio.sleep(wait)
            wait *= 2
    raise AssertionError("unreachable: the last attempt returns or raises")
