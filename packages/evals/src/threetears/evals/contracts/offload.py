"""Blocking calls off the event loop, onto an executor the host chooses.

The eval engine's storage and its host's internal APIs are synchronous: a database round-trip or an
IPC request blocks the thread that makes it. Made on the event loop, each one stalls every other
coroutine in the process for its duration — in a web process that is every eval run and every
API surface at once, and enough of them together can stall it past its heartbeat.

The engine names no executor of its own. A host passes one (a dedicated, bounded pool, say) so
this work cannot starve the loop's
DEFAULT executor, which a host's liveness probes may depend on. ``None`` means that default
executor, which is what a host without a pool of its own gets.

It lives in the contracts because the shared contract's own coroutines make store calls too — an
out-of-run budget writing its ledger row, case generation reading and writing test cases, an analysis
generation recording its attempt — and none may import the run package. The run package's public root
exports :func:`run_blocking` for hosts.

:func:`wait_through_cancellation` is the other half: what a coroutine does when it is cancelled while a
write it must record is still running on a worker, which the runner, the job manager, a launch and an
analysis generation all face.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
from collections.abc import Callable
from concurrent.futures import Executor
from typing import Any


async def run_blocking[**P, T](
    executor: Executor | None, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    """Run ``fn(*args, **kwargs)`` on ``executor`` and await its result.

    The caller's context is copied onto the worker, as ``asyncio.to_thread`` does and a bare
    ``run_in_executor`` does not: several ContextVars carry meaning across this hop — the eval-mode
    flag that decides whether a tool actuates for real, the request id that correlates traces, the
    active span — and dropping them would not fail, it would run the call under their defaults.

    Args:
        executor: Where to run it; ``None`` for the loop's default executor.
        fn: The blocking callable.
        *args: Positional arguments for ``fn``.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        What ``fn`` returned. Typed through ``fn``'s own signature, so a call missing an argument
        ``fn`` requires is a type error at the call site rather than a ``TypeError`` on a worker.
    """
    context = contextvars.copy_context()
    call = functools.partial(context.run, fn, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(executor, call)


async def wait_through_cancellation(work: asyncio.Future[Any]) -> bool:
    """Wait for ``work`` to finish even if the waiter is cancelled meanwhile.

    A blocking call handed to a worker cannot be interrupted, so cancelling the coroutine awaiting
    it does not stop it — it only stops anyone reading its outcome. For a write whose outcome the
    caller must record (was the row persisted?), that turns a cancel into a false record. So the
    wait is shielded and resumed until the work ends, and the caller is TOLD it was cancelled, to
    deliver that cancellation itself once it has recorded what the work did.

    Args:
        work: The future of the blocking call, typically from :func:`run_blocking`.

    Not the same contract as the job manager's terminal-status write (:mod:`threetears.evals.run.jobs`), on purpose: that one
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


__all__ = ["run_blocking", "wait_through_cancellation"]
