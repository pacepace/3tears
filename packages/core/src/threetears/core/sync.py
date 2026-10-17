"""calling 3tears from synchronous code: a CLI, a batch script, a worker thread with no loop.

3tears is async. A synchronous caller runs one call to the end with :func:`run`::

    from threetears.core import sync

    page = sync.run(lambda: client.get("/status"), timeout=30)

Every call runs on ONE shared event loop, in a daemon thread started on first use, so what one
call opens stays usable by the next: a NATS connection, an HTTP client, a :class:`KVLease`. Open
such a resource through :func:`run` and use it only through :func:`run` afterwards; a resource
opened on another loop cannot be used on this one.

A lease held across synchronous work is :meth:`KVLease.hold_sync
<threetears.core.coordination.KVLease.hold_sync>`: the same loop renews it in the background
while the caller's thread does the work::

    nats = sync.run(connect_nats)                      # opened on the shared loop
    lease = KVLease(nats, bucket_name="leases")
    with lease.hold_sync("set.june", ttl=60, renew_every=20, max_wait_seconds=600) as held:
        for step in steps:
            held.check()                               # raises LeaseLost once it is lost
            step()

:func:`run` refuses to run inside a running event loop: blocking a loop's own thread on work
that needs a loop is a deadlock, or (on another loop) a stall of everything that loop serves.
Async code awaits the coroutine instead.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

from threetears.core._bridge import submit

__all__ = ["run"]

T = TypeVar("T")


async def _awaited(work: Coroutine[Any, Any, T] | Callable[[], Awaitable[T]]) -> T:
    """await ``work`` on the loop that runs this, building it there first when it is a factory.

    :param work: a coroutine, or a callable that returns an awaitable
    :ptype work: Coroutine[Any, Any, T] | Callable[[], Awaitable[T]]
    :return: what it returned
    :rtype: T
    """
    if callable(work):
        return await work()
    return await work


def _in_running_loop() -> bool:
    """whether this thread is running an event loop right now.

    :return: ``True`` inside a running loop
    :rtype: bool
    """
    try:
        asyncio.get_running_loop()
    # NOSILENT: no running loop in this thread is the answer asked for, not a failure
    except RuntimeError:
        return False
    return True


def run(work: Coroutine[Any, Any, T] | Callable[[], Awaitable[T]], *, timeout: float | None = None) -> T:
    """run ``work`` to the end on the shared loop and return its result, from synchronous code.

    pass a factory (``lambda: client.get(url)``, or an ``async def`` with no arguments) rather than
    a coroutine where you can: it is then called ON the shared loop, so anything it builds there
    (a lock, a future, a client) belongs to that loop. A coroutine object works too.

    the work's own exception is raised here unchanged. Ctrl-C (or any other exception) while this
    thread waits cancels the work before it propagates.

    :param work: a coroutine, or a callable returning an awaitable
    :ptype work: Coroutine[Any, Any, T] | Callable[[], Awaitable[T]]
    :param timeout: longest to wait, in seconds; ``None`` waits until the work ends. On timeout
        the work is cancelled (its cancellation may still be unwinding on the loop when this
        raises)
    :ptype timeout: float | None
    :return: what the work returned
    :rtype: T
    :raises RuntimeError: called from inside a running event loop (await the work there instead)
    :raises ValueError: ``timeout`` is not positive
    :raises TimeoutError: the work did not finish within ``timeout``
    """
    if _in_running_loop():
        if isinstance(work, Coroutine):
            work.close()  # never started: closed here, so nothing warns it was never awaited
        raise RuntimeError(
            "threetears.core.sync.run was called from inside a running event loop; it blocks the thread "
            "until the work ends, which would stall or deadlock that loop -- await the work instead"
        )
    if timeout is not None and timeout <= 0:
        if isinstance(work, Coroutine):
            work.close()
        raise ValueError(f"timeout must be positive or None, got {timeout}")
    future = submit(_awaited(work))
    try:
        done, _ = concurrent.futures.wait([future], timeout=timeout)
    except BaseException:
        future.cancel()
        raise
    if not done:
        future.cancel()
        raise TimeoutError(f"the work did not finish within {timeout}s and was cancelled")
    return future.result()
