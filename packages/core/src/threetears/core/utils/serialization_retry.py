"""run a unit of statements again when YugabyteDB answers it with a serialization conflict (SQLSTATE ``40001``).

YugabyteDB answers a write that meets a concurrently committed one with ``SerializationError`` ("conflict with
concurrently committed data", kConflict) and asks the client to retry. Under READ COMMITTED it retries a statement
itself, but not always: the message says when it could not ("query layer retry isn't possible, ... some data was
already sent to the user"), and the conflict reaches the caller. PostgreSQL under READ COMMITTED raises no such
error for a plain write, so on PostgreSQL the unit runs once.

The unit is re-run whole, so it must be safe to run again: idempotent, or reading again what it decides on (a
lookup before an insert). A conflict aborts the statement's transaction, so a failed attempt left nothing written.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import asyncpg

__all__ = ["retry_serialization_conflicts"]

_T = TypeVar("_T")


async def retry_serialization_conflicts(
    attempt: Callable[[], Awaitable[_T]], *, attempts: int, first_wait: float
) -> _T:
    """``attempt()``'s answer, run again while it raises ``asyncpg.SerializationError``; any other error, or the
    conflict on the last attempt, is raised.

    :param attempt: the unit, re-runnable (see the module)
    :ptype attempt: Callable[[], Awaitable[_T]]
    :param attempts: the runs tried, at least one
    :ptype attempts: int
    :param first_wait: seconds before the first retry, doubled for each after
    :ptype first_wait: float
    :return: what the first attempt that did not conflict answered
    :rtype: _T
    :raises asyncpg.SerializationError: when every attempt conflicts
    :raises ValueError: when ``attempts`` is under one
    """
    if attempts < 1:
        raise ValueError(f"a unit is tried at least once, not {attempts!r} times")
    wait = first_wait
    for tried in range(1, attempts + 1):
        try:
            return await attempt()
        except asyncpg.SerializationError:
            if tried == attempts:
                raise
            await asyncio.sleep(wait)
            wait *= 2
    raise AssertionError("unreachable: the last attempt returns or raises")
