"""the one owner of the asyncio private name ``threetears.core``'s loop timing depends on: ``Handle._run``.

:func:`~threetears.core.testing.loop_lateness.turn_holds` times each turn of a test's event loop on the loop thread's
CPU clock, to bound how long one turn of a serving process's loop may hold it. asyncio has no public hook around a
callback's run: ``loop.slow_callback_duration`` (debug mode) times the wall clock, which a host's load stretches, and
only logs. Asyncio's own loop runs every callback -- a task's step, a socket's read -- through ``Handle._run``, so this
module wraps it for a block and puts it back.

**Verified against CPython 3.14** (``Lib/asyncio/events.py``: ``Handle._run``, called by ``BaseEventLoop._run_once``
for every ready handle, ``TimerHandle`` included through inheritance). Moving to a new release: confirm
``_run_once`` still calls ``handle._run()``, and run ``packages/core/tests/unit/testing/test_turn_holds.py`` on it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

__all__ = ["timed_handles"]


@contextmanager
def timed_handles(record: Callable[[float, str], None]) -> Iterator[None]:
    """every callback asyncio's loop runs inside the block, handed to ``record`` with its thread CPU seconds.

    :param record: takes each callback's CPU seconds and its repr, as it ends
    :ptype record: Callable[[float, str], None]
    :return: nothing; the block
    :rtype: Iterator[None]
    """
    original = asyncio.events.Handle._run

    def timed(handle: asyncio.events.Handle) -> None:
        began = time.thread_time()
        try:
            original(handle)
        finally:
            record(time.thread_time() - began, repr(handle)[:120])

    asyncio.events.Handle._run = timed  # type: ignore[method-assign,assignment]
    try:
        yield
    finally:
        asyncio.events.Handle._run = original  # type: ignore[method-assign]
