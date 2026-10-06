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
out-of-run budget writing its ledger row, case generation reading and writing test cases — and
neither may import the run package. The run package's public root exports it for hosts.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
from collections.abc import Callable
from concurrent.futures import Executor


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


__all__ = ["run_blocking"]
