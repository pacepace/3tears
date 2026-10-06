"""Waiting on a blocking call that was handed to an executor, through a cancellation of the waiter.

Handing the call over is :func:`~threetears.evals.contracts.offload.run_blocking`, which lives in the
contracts because the shared contract's own coroutines need it; what to do when the coroutine awaiting
such a call is cancelled is the runner's concern alone, and lives here.
"""

from __future__ import annotations

import asyncio
from typing import Any


async def wait_through_cancellation(work: asyncio.Future[Any]) -> bool:
    """Wait for ``work`` to finish even if the waiter is cancelled meanwhile.

    A blocking call handed to a worker cannot be interrupted, so cancelling the coroutine awaiting
    it does not stop it — it only stops anyone reading its outcome. For a write whose outcome the
    caller must record (was the row persisted?), that turns a cancel into a false record. So the
    wait is shielded and resumed until the work ends, and the caller is TOLD it was cancelled, to
    deliver that cancellation itself once it has recorded what the work did.

    Args:
        work: The future of the blocking call, typically from :func:`~threetears.evals.contracts.offload.run_blocking`.

    Not the same contract as ``threetears.evals.run.jobs._finish_even_if_cancelled``, on purpose: that one
    ABSORBS the caller's cancel, because its caller (a terminal status write) is the last thing a
    job does and re-raises its own cancel afterwards. This one REPORTS it, because its callers have
    more to do after the write and must not carry on as if uncancelled. Pick by whether the caller
    re-raises on its own.

    Returns:
        Whether a cancellation aimed at the waiter arrived while it waited. The caller owes a
        ``raise asyncio.CancelledError`` once it has read ``work``'s outcome.

    Raises:
        asyncio.CancelledError: ``work`` itself was cancelled — a queued work item the executor
            dropped — so there is no outcome to record.
    """
    cancelled = False
    # ``asyncio.wait``, not ``shield``: it neither cancels what it waits on when the waiter is
    # cancelled nor raises the work's own exception, which is the caller's to read from ``work``.
    while not work.done():
        try:
            await asyncio.wait({work})
        except asyncio.CancelledError:
            cancelled = True
    if work.cancelled():
        raise asyncio.CancelledError
    return cancelled


__all__ = ["wait_through_cancellation"]
