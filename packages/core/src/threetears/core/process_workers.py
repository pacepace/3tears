"""CPU work in worker processes, plain data in and out: the one helper every serving process's CPU half goes through.

**Why.** CPU work in a serving process belongs in worker processes, never on threads: a thread running Python holds
the interpreter lock and the event loop waits behind it. The loop keeps the I/O; a job's CPU runs in a worker
process forked by multiprocessing's fork server (:func:`~threetears.core.processes.start_worker_server`), never by the
serving process: a serving process runs threads, whose locks a fork copies mid-hold, and a spawn is a ``fork_exec`` of
the whole serving process under the interpreter lock.

**Plain data across, checked.** A job's arguments and its answer are plain data -- bytes, text, numbers, booleans,
``None``, and tuples, lists and frozensets of them (an Arrow IPC stream is bytes) -- or a class, which pickles as
its name. Never an object graph: a pickled object is decoded on the receiving side by whatever its class does, and
a large one pickles on the executor's feeder thread holding the interpreter lock. Both directions are checked, so a
job that would hand over an object fails loudly, in development, rather than quietly costing the loop. Tables too
large for bytes cross through :mod:`threetears.core.shared_arrow` instead.

**One code path, two places.** :meth:`ProcessWorkers.run` takes ``inline=True`` for work small enough that the
round trip to a worker costs the loop more than the work: the same job runs on the caller's thread. The caller
decides by a measured bound, never a guess.

**One pool per kind of work, made at first use.** A :class:`ProcessWorkers` makes its pool at the first job that is
not inline -- on a thread, since a fresh pool starts its processes on submission, and the first one in a process
starts the fork server -- and keeps it. A worker that dies
fails the jobs it held (``BrokenProcessPool``) and the pool is replaced at the next job. :meth:`ProcessWorkers.close`
shuts it down; at interpreter exit ``concurrent.futures`` joins it anyway. Each worker starts below the serving
process's priority, and exits when the serving process dies.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from threetears.core.processes import exit_with_parent, name_process, start_worker_server
from threetears.observe import get_logger

__all__ = ["BLAS_THREADS", "ProcessWorkers"]

log = get_logger(__name__)

#: the variable that bounds the threads numpy's OpenBLAS starts as it loads: a worker sets it to one (:func:`_below_the_server`)
BLAS_THREADS = "OPENBLAS_NUM_THREADS"

#: the scalar types that cross as they are
_PLAIN_SCALARS = (bytes, str, int, float, bool, type(None))

#: the containers that cross when everything in them does
_PLAIN_CONTAINERS = (tuple, list, frozenset)


def require_plain(value: Any, *, what: str) -> None:
    """refuse a value that is not plain data (see the module docstring).

    :param value: the value
    :ptype value: Any
    :param what: what it is, for the message
    :ptype what: str
    :return: nothing
    :rtype: None
    :raises TypeError: when the value, or anything in it, is not plain data
    """
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, _PLAIN_SCALARS) or isinstance(item, type):
            continue
        if isinstance(item, _PLAIN_CONTAINERS):
            pending.extend(item)
            continue
        raise TypeError(
            f"{what} holds a {type(item).__name__}: a worker process is handed plain data (bytes, text, numbers, "
            "and tuples, lists and frozensets of them), never an object"
        )


def _plain_call(job: Callable[..., Any], args: tuple[Any, ...]) -> Any:
    """run a job in a worker, refusing an answer that is not plain data.

    :param job: the job, a module function
    :ptype job: Callable[..., Any]
    :param args: its arguments
    :ptype args: tuple[Any, ...]
    :return: its answer
    :rtype: Any
    :raises TypeError: when the answer is not plain data
    """
    answer = job(*args)
    require_plain(answer, what=f"the answer of {job.__qualname__}")
    return answer


def _below_the_server(serving: int, process_name: str | None = None) -> None:
    """a worker's start: it exits when the serving process dies (:func:`~threetears.core.processes.exit_with_parent`,
    never an orphan waiting on its queue: its parent is the fork server, which outlives a killed serving process while
    any worker lives), and is scheduled below it, so the server stays responsive.

    A process already at the lowest priority cannot go lower: macOS refuses ``nice`` past it with ``EPERM`` (Linux
    clamps). The worker then keeps the priority it has and says so once, at its start; raising there would fail the
    initializer and with it every worker of the pool (``BrokenProcessPool``).

    :param serving: the serving process's pid
    :ptype serving: int
    :param process_name: the name the worker gives itself (:func:`~threetears.core.processes.name_process`); None
        to keep the fork server's
    :ptype process_name: str | None
    :return: nothing
    :rtype: None
    """
    exit_with_parent(serving=serving)
    # before any job imports numpy: its Linux OpenBLAS starts a thread a core as it loads (10 on a 10-CPU container, 0
    # with this, 2026-11-25), each counted against the pod's thread limit, and no job here runs BLAS. A copy's worker
    # on a many-core pool host so took the threads its test processes needed ("can't start new thread", monad-1,
    # 2026-11-25). An operator's own setting is kept
    os.environ.setdefault(BLAS_THREADS, "1")
    if process_name is not None:
        name_process(process_name)
    try:
        os.nice(5)
    except OSError as error:
        log.warning(
            "process worker keeps its priority: it cannot be lowered further",
            extra={"extra_data": {"niceness": os.getpriority(os.PRIO_PROCESS, 0), "error": str(error)}},
        )


class ProcessWorkers:
    """a pool of worker processes for one kind of CPU work, made at its first job.

    :param workers: the most processes, side by side
    :ptype workers: int
    :param name: the work, for the log
    :ptype name: str
    :param process_name: the name each worker gives itself as it starts, as ``ps`` and ``/proc/<pid>/comm`` say it
        on Linux (:func:`~threetears.core.processes.name_process`, cut to 15 bytes): every worker the fork server
        forks has its command line, so this tells this pool's workers from another's; None keeps the server's
    :ptype process_name: str | None
    """

    def __init__(self, *, workers: int, name: str, process_name: str | None = None) -> None:
        if workers < 1:
            raise ValueError("a pool needs a worker")
        self._workers = workers
        self._name = name
        self._process_name = process_name
        #: held while a thread makes, replaces or shuts the pool: never by the event loop
        self._lock = threading.Lock()
        self._pool: ProcessPoolExecutor | None = None
        self._broken = False

    @property
    def workers(self) -> int:
        """the most processes the pool runs side by side.

        :return: the count
        :rtype: int
        """
        return self._workers

    async def run(self, job: Callable[..., Any], *args: Any, inline: bool = False) -> Any:
        """``job(*args)`` in a worker, or on the caller's thread when ``inline``; the loop only waits.

        :param job: the job: a module function, importable by name in a fresh process
        :ptype job: Callable[..., Any]
        :param args: its arguments: plain data
        :ptype args: Any
        :param inline: run it here, for work below the caller's measured bound
        :ptype inline: bool
        :return: its answer: plain data
        :rtype: Any
        :raises TypeError: when an argument or the answer is not plain data
        :raises BrokenProcessPool: when its worker died (the pool is replaced at the next job)
        :raises Exception: what the job raised
        """
        require_plain(args, what=f"the arguments of {job.__qualname__}")
        if inline:
            return _plain_call(job, args)
        [answer] = await self.run_each(job, [args])
        return answer

    async def run_each(self, job: Callable[..., Any], argument_sets: Sequence[tuple[Any, ...]]) -> list[Any]:
        """``job`` once per argument set, side by side across the workers; the answers in the sets' order.

        :param job: the job: a module function, importable by name in a fresh process
        :ptype job: Callable[..., Any]
        :param argument_sets: each run's arguments: plain data
        :ptype argument_sets: Sequence[tuple[Any, ...]]
        :return: each run's answer
        :rtype: list[Any]
        :raises TypeError: when an argument or an answer is not plain data
        :raises BrokenProcessPool: when a worker died (the pool is replaced at the next job)
        :raises Exception: what a run raised
        """
        for args in argument_sets:
            require_plain(args, what=f"the arguments of {job.__qualname__}")
        if not argument_sets:
            return []
        futures = await asyncio.to_thread(self._submitted, job, list(argument_sets))
        try:
            return list(await asyncio.gather(*(asyncio.wrap_future(future) for future in futures)))
        except BrokenProcessPool:
            if not self._broken:
                log.warning(
                    "a worker process died; its pool is replaced at the next job",
                    extra={"extra_data": {"pool": self._name}},
                )
            self._broken = True
            raise

    def _submitted(self, job: Callable[..., Any], argument_sets: list[tuple[Any, ...]]) -> list[Future[Any]]:
        """submit each run to the pool, made (or replaced, after a worker died) first: on a thread.

        :param job: the job
        :ptype job: Callable[..., Any]
        :param argument_sets: each run's arguments
        :ptype argument_sets: list[tuple[Any, ...]]
        :return: each run's future
        :rtype: list[Future[Any]]
        """
        with self._lock:
            if self._pool is not None and self._broken:
                self._pool.shutdown(wait=False)
                self._pool = None
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    max_workers=self._workers,
                    mp_context=start_worker_server(),
                    initializer=_below_the_server,
                    initargs=(os.getpid(), self._process_name),
                )
                self._broken = False
            pool = self._pool
            return [pool.submit(_plain_call, job, args) for args in argument_sets]

    async def close(self, *, wait: bool = True) -> None:
        """shut the workers down; the next job starts a fresh pool.

        ``wait=False`` is the stop bound's: a serving process stopping within a deadline cannot wait for a job that
        may run for minutes, so every job not started is cancelled and every worker terminated at once (``SIGTERM``,
        :meth:`~concurrent.futures.ProcessPoolExecutor.terminate_workers`); a job running fails its caller with
        ``BrokenProcessPool``. A job's work is left as a killed process leaves it, so a job that writes must leave
        nothing a reader takes until its last rename (every job of this kind does).

        :param wait: wait for the jobs running to end; False cancels and terminates them
        :ptype wait: bool
        :return: nothing
        :rtype: None
        """
        if wait:
            await asyncio.to_thread(self._close)
        else:
            self._terminate()

    def _close(self) -> None:
        """:meth:`close`, on a thread.

        :return: nothing
        :rtype: None
        """
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def _terminate(self) -> None:
        """:meth:`close` without waiting: the pool taken, its queued jobs cancelled, its workers terminated. Never
        blocks: ``terminate_workers`` signals each process and shuts the pool down without joining it.

        The pool is taken without the lock (a thread making the pool holds it for as long as the fork server takes to
        start): one swap, on the event loop; a job submitted to the old pool meanwhile fails as broken, which is what a
        stopping process's job does.

        :return: nothing
        :rtype: None
        """
        pool, self._pool = self._pool, None
        if pool is None:
            return
        # shuts the pool down without waiting and cancels its queued jobs, then signals each live worker
        pool.terminate_workers()
        log.info(
            "process workers terminated without waiting (a bounded stop)", extra={"extra_data": {"pool": self._name}}
        )
