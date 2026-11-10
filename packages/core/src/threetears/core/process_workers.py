"""CPU work in worker processes, plain data in and out: the one helper every serving process's CPU half goes through.

**Why.** CPU work in a serving process belongs in worker processes, never on threads: a thread running Python holds
the interpreter lock and the event loop waits behind it. The loop keeps the I/O; a job's CPU runs in a spawned
process (spawned, never forked: a serving process runs threads, whose locks a fork copies mid-hold).

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
not inline -- on a thread, since a fresh pool spawns its processes on submission -- and keeps it. A worker that dies
fails the jobs it held (``BrokenProcessPool``) and the pool is replaced at the next job. :meth:`ProcessWorkers.close`
shuts it down; at interpreter exit ``concurrent.futures`` joins it anyway. Each worker starts below the serving
process's priority.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from threetears.observe import get_logger

__all__ = ["ProcessWorkers"]

log = get_logger(__name__)

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


def _below_the_server() -> None:
    """a worker's start: scheduled below the serving process, so it stays responsive.

    :return: nothing
    :rtype: None
    """
    os.nice(5)


class ProcessWorkers:
    """a pool of spawned worker processes for one kind of CPU work, made at its first job.

    :param workers: the most processes, side by side
    :ptype workers: int
    :param name: the work, for the log
    :ptype name: str
    """

    def __init__(self, *, workers: int, name: str) -> None:
        if workers < 1:
            raise ValueError("a pool needs a worker")
        self._workers = workers
        self._name = name
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
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_below_the_server,
                )
                self._broken = False
            pool = self._pool
            return [pool.submit(_plain_call, job, args) for args in argument_sets]

    async def close(self) -> None:
        """shut the workers down, waiting for them; the next job starts a fresh pool.

        :return: nothing
        :rtype: None
        """
        await asyncio.to_thread(self._close)

    def _close(self) -> None:
        """:meth:`close`, on a thread.

        :return: nothing
        :rtype: None
        """
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)
