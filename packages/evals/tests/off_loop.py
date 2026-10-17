"""A test's own synchronous store call, made off the event loop's thread as the engine makes its own.

``EvalStorage`` refuses a call made on a thread running an event loop
(:class:`~threetears.evals.schema.StoreCalledOnEventLoop`). An async test that seeds a store, or reads one
back to assert on it, makes that call from a coroutine; :func:`off_loop` makes it on a worker thread and
waits for it there, so the test keeps reading as plain synchronous calls while the refusal stays on for
every path the engine runs.

It blocks the loop while it waits, which a test's setup and assertions may and the engine may not: the
engine offloads with :func:`~threetears.evals.kernel.offload.run_blocking` and awaits. It must not wrap a
store bridged onto the test's own loop (:func:`~threetears.evals.storage.bridge_async_store`), whose call
would wait on the loop this blocks; await ``run_blocking`` for that.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

__all__ = ["off_loop"]


def off_loop[**P, T](fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    """Call ``fn(*args, **kwargs)`` on a worker thread, in the caller's context, and return what it returned.

    :param fn: The synchronous call: a store write or read.
    :ptype fn: Callable[P, T]
    :param *args: Its positional arguments.
    :ptype args: P.args
    :param **kwargs: Its keyword arguments.
    :ptype kwargs: P.kwargs
    :return: What ``fn`` returned; what it raised is raised here.
    :rtype: T
    """
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as worker:
        return worker.submit(context.run, fn, *args, **kwargs).result()
