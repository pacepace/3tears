"""How a run's cells are executed: one at a time, or several at once, through one seam.

A run executes its cells serially when its launch declares latency under test
(``measure_latency=True``): a cell's wall-clock is then read with nothing else of its run beside it. When
latency is not under test, serial execution is pure cost, so the cells run several at a time, up to the
run's width (:attr:`~threetears.evals.run.launch.LaunchSettings.max_concurrent_cells`). The latency such a
run still records is stamped measured under concurrency (the ``execution_mode`` covariate), and the
analysis keeps it out of every comparison.

**The seam is :class:`CellExecutor`.** :func:`~threetears.evals.run.runner.execute_run` hands it every
cell of the run as a :data:`CellWork` — a closure the runner built, which records, saves and reports its
own cell — and the width. The executor decides only when each closure runs: a host with a process-wide
pool or a queue of its own supplies one on
:attr:`~threetears.evals.run.launch.LaunchHost.cell_executor`; :class:`InProcessCellExecutor` is the
default and covers both modes.

**What every executor owes the runner**, because the runner's guarantees rest on it:

- every cell runs exactly once, and never more than ``width`` at once;
- cells START in the order given (the run's shuffled order, :func:`~threetears.evals.run.runner.cell_execution_order`),
  so which cell runs when is still derived from the run's id;
- with ``width == 1`` each cell finishes before the next starts;
- when a cell raises, no further cell starts, the cells in flight are cancelled and waited for, and the
  first exception is raised;
- when the executor itself is cancelled (the run was), the cells in flight are cancelled and waited for —
  each records itself as cancelled with its run, and saves what it spent — before the cancellation
  propagates.

A cell the run's stop (its cost cap, its account) reached before it began returns at once without
running; that is the runner's decision, taken inside the closure, so an executor never needs to know
about it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

#: The width a run executes its cells at when latency is not under test and the host's settings name
#: none: enough to turn hours of a slow candidate into a fraction of that, few enough that a provider's
#: rate limit is not the first thing a default launch meets.
DEFAULT_MAX_CONCURRENT_CELLS = 4

#: One cell of a run, as the runner hands it to an executor: it runs the cell, records it and reports it.
CellWork = Callable[[], Awaitable[None]]


class CellExecutor(Protocol):
    """Runs a run's cells, at most ``width`` at once, starting them in the order given.

    See the module docstring for what every implementation owes the runner.
    """

    async def execute(self, cells: Sequence[CellWork], *, width: int) -> None:
        """Run every cell exactly once, never more than ``width`` at once.

        Args:
            cells: The run's cells, in the order they must start.
            width: The most that may run at once; at least 1. ``1`` is serial.

        Raises:
            BaseException: The first exception a cell raised, once every cell in flight has ended.
        """
        ...  # pragma: no cover — protocol


class InProcessCellExecutor:
    """The default executor: ``width`` workers on this event loop, each taking the next cell in order.

    At ``width == 1`` it awaits each cell in the caller's own task, which is exactly the serial loop a
    run executed before cells could run concurrently — same task, same cancellation, same order.
    """

    async def execute(self, cells: Sequence[CellWork], *, width: int) -> None:
        """Run every cell exactly once, never more than ``width`` at once, starting them in order.

        Args:
            cells: The run's cells, in the order they must start.
            width: The most that may run at once; at least 1.

        Raises:
            ValueError: ``width`` is below 1.
            BaseException: The first exception a cell raised, once every cell in flight has ended.
        """
        if width < 1:
            raise ValueError(f"a run executes at least one cell at a time; got width={width}")
        if width == 1:
            for cell in cells:
                await cell()
            return
        pending = iter(cells)

        async def worker() -> None:
            # One shared iterator: each worker takes the next cell when it is free, so cells start in the
            # order given and at most `width` are ever running. `next` never yields to the loop, so two
            # workers cannot take the same cell.
            for cell in pending:
                await cell()

        workers = [asyncio.create_task(worker()) for _ in range(min(width, len(cells)))]
        try:
            await asyncio.wait(workers, return_when=asyncio.FIRST_EXCEPTION)
        except asyncio.CancelledError:
            await _cancel_and_wait(workers)
            raise
        failed = next((task for task in workers if task.done() and task.exception() is not None), None)
        if failed is None:
            return
        await _cancel_and_wait(workers)
        error = failed.exception()
        assert error is not None
        raise error


async def _cancel_and_wait(tasks: Sequence[asyncio.Task[None]]) -> None:
    """Cancel every task not done, and wait for each to end however it ends."""
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


__all__ = [
    "DEFAULT_MAX_CONCURRENT_CELLS",
    "CellExecutor",
    "CellWork",
    "InProcessCellExecutor",
]
