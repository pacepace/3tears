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

:func:`drain_then_cut` is the one interrupt rule every holder of runs keeps (a quick comparison, a launch's job, a
foreground caller awaiting jobs): a first interrupt asks the work to stop starting and lets what is running finish and
record, and a further one, a cancel or a second Ctrl+C, cuts it.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Executor
from typing import Any

from threetears.evals.schema.logs import contained, get_logger

log = get_logger(__name__)


async def run_blocking[**P, T](
    executor: Executor | None, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    """Run ``fn(*args, **kwargs)`` on ``executor`` and await its result.

    The caller's context is copied onto the worker, as ``asyncio.to_thread`` does and a bare
    ``run_in_executor`` does not: several ContextVars carry meaning across this hop — the eval-mode
    flag that decides whether a tool actuates for real, the request id that correlates traces, the
    active span — and dropping them would not fail, it would run the call under their defaults.

    :param executor: Where to run it; ``None`` for the loop's default executor.
    :ptype executor: Executor | None
    :param fn: The blocking callable.
    :ptype fn: Callable[P, T]
    :param *args: Positional arguments for ``fn``.
    :ptype args: P.args
    :param **kwargs: Keyword arguments for ``fn``.
    :ptype kwargs: P.kwargs
    :return: What ``fn`` returned. Typed through ``fn``'s own signature, so a call missing an argument
        ``fn`` requires is a type error at the call site rather than a ``TypeError`` on a worker.
    :rtype: T
    """
    context = contextvars.copy_context()
    call = functools.partial(context.run, fn, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(executor, call)


async def wait_through_cancellation(
    work: asyncio.Future[Any], *, on_cancel: Callable[[], object] | None = None
) -> bool:
    """Wait for ``work`` to finish even if the waiter is cancelled meanwhile.

    A blocking call handed to a worker cannot be interrupted, so cancelling the coroutine awaiting
    it does not stop it — it only stops anyone reading its outcome. For a write whose outcome the
    caller must record (was the row persisted?), that turns a cancel into a false record. So the
    wait is shielded and resumed until the work ends, and the caller is TOLD it was cancelled, to
    deliver that cancellation itself once it has recorded what the work did.

    :param work: The future of the blocking call, typically from :func:`run_blocking`.
    :ptype work: asyncio.Future[Any]
    :param on_cancel: Called each time a cancellation aimed at the waiter arrives, before the wait goes on: what the
        waiter does about it meanwhile (cut the work short, say), so a cancel is acted on without abandoning the work.
    :ptype on_cancel: Callable[[], object] | None


    Not the same contract as the job manager's terminal-status write (:mod:`threetears.evals.run.jobs`), on purpose:
    that one ABSORBS the caller's cancel, because its caller (a terminal status write) is the last thing a
    job does and re-raises its own cancel afterwards. This one REPORTS it, because its callers have
    more to do after the write and must not carry on as if uncancelled. Pick by whether the caller
    re-raises on its own.

    :return: Whether a cancellation aimed at the waiter arrived while it waited. The caller owes a
        ``raise asyncio.CancelledError`` once it has read ``work``'s outcome.
    :rtype: bool
    :raises asyncio.CancelledError: ``work`` itself was cancelled — a queued work item the executor
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
            if on_cancel is not None:
                on_cancel()
    if work.cancelled():
        raise asyncio.CancelledError
    return cancelled


def _to_its_end(work: Callable[[], Awaitable[object]]) -> asyncio.Future[None]:
    """A future that ends once ``work`` has run to its end, begun again each time its task is cancelled.

    For a wait whose end a caller must reach however the loop is stopping. ``asyncio.run``, stopped by a
    ``KeyboardInterrupt``, cancels every task once and runs the loop until they end: the caller's own wait is told and
    goes on (:func:`wait_through_cancellation`), but a task it awaits is cancelled too, and what that task waited for
    has not ended. This future is no task, so nothing cancels it, and the work is begun again in a task created after
    that one cancel, as it is where a ``KeyboardInterrupt`` landing in the work's own step ended it. ``work`` must be
    safe to begin again: a wait for jobs to settle, or a shutdown, which cancels and waits for what is still running.

    :param work: Makes the awaitable to run, once per beginning.
    :ptype work: Callable[[], Awaitable[object]]
    :return: Ends with ``None`` once ``work`` returns, or with what it raised.
    :rtype: asyncio.Future[None]
    """
    loop = asyncio.get_running_loop()
    ended: asyncio.Future[None] = loop.create_future()
    # The task running the work now, held so it is not collected mid-run: the loop keeps only a weak reference.
    held: list[asyncio.Future[object]] = []

    def begin() -> None:
        task = asyncio.ensure_future(work())
        held[:] = [task]
        task.add_done_callback(finished)

    def finished(task: asyncio.Future[object]) -> None:
        held.clear()
        # Cancelled, or ended by a ``KeyboardInterrupt`` landing in its step (which has left the loop already, and is
        # reported where it lands): what the work waited for has not ended, so it is begun again.
        if task.cancelled() or isinstance(task.exception(), KeyboardInterrupt):
            begin()
        elif (raised := task.exception()) is not None:
            ended.set_exception(raised)
        else:
            ended.set_result(None)

    begin()
    return ended


async def drain_then_cut(
    ask: Callable[[], object],
    cut: Callable[[], object],
    settle: Sequence[tuple[Callable[[], Awaitable[object]], str]],
) -> KeyboardInterrupt | None:
    """Keep the interrupt rule: ``ask`` the work to start nothing more, and wait for it to settle; a further interrupt
    meanwhile cuts it (``cut``).

    The first interrupt is the caller's own cancel, already landed: ``ask`` lets what is running finish and record
    (each piece bounded by its own deadline), so a slow piece is not drawn afresh in place of the answer it was giving.
    A further cancel of the waiting task, or a second Ctrl+C delivered into it as a ``KeyboardInterrupt``, calls
    ``cut``, which stops what is still running at once; the wait goes on until every step of ``settle`` has run to its
    end, each begun again if a cancel ended it (:func:`_to_its_end`), so whatever was cut is settled and recorded all
    the same. A step that fails is recorded with its words and the next is waited for.

    In a terminal a second Ctrl+C is not a second cancel: ``asyncio.run`` raises ``KeyboardInterrupt`` wherever the main
    thread is. Where that is the waiting task's step it arrives here, cuts, and is returned for the caller to raise once
    it has recorded what it holds; anywhere else it leaves the loop, and ``asyncio.run`` cancels every task once before
    reporting it, which cuts here as a further cancel does.

    :param ask: Asks the work to start nothing more: what a first interrupt does.
    :ptype ask: Callable[[], object]
    :param cut: Stops what is still running: what each further interrupt does.
    :ptype cut: Callable[[], object]
    :param settle: Each wait, with what to call its failure, run in order to its end: safe to begin again.
    :ptype settle: Sequence[tuple[Callable[[], Awaitable[object]], str]]
    :return: The ``KeyboardInterrupt`` that cut, where one arrived here; ``None`` otherwise. The caller raises it after
        the cancel that landed first.
    :rtype: KeyboardInterrupt | None
    """
    interrupted: list[KeyboardInterrupt] = []
    try:
        ask()
    except KeyboardInterrupt as second:
        # NOSILENT: the second Ctrl+C, landing as the first is asked; it cuts, and the caller raises it once settled.
        interrupted.append(second)
        cut()
    for wait, what in settle:
        ended = _to_its_end(wait)
        while True:
            try:
                await wait_through_cancellation(ended, on_cancel=cut)
                break
            except KeyboardInterrupt as second:
                # NOSILENT: the second Ctrl+C, delivered into this task; it cuts as a further cancel does, and the
                # caller raises it once what it holds is settled.
                interrupted.append(second)
                cut()
        # Everything is settled whatever this wait met; a failure of it is recorded.
        with contained(what, log=log):
            ended.result()
    return interrupted[0] if interrupted else None


__all__ = ["drain_then_cut", "run_blocking", "wait_through_cancellation"]
