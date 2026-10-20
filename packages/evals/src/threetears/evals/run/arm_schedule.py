"""How the arms of one launch share time: interleaved case by case, one after another, or side by side.

The arms of a comparison run on shared provider capacity, and how they share it is not neutral. Side by side, each at
its own width (``concurrent``), a launch loads its providers with every arm at once, and a provider rationing one
model's calls rations whichever arm runs on it: one arm's sample shrinks and the other's does not, and the comparison
reads host capacity as a difference between arms. One after another (``sequential``), no arm shares another's load,
but each arm meets the provider at another time, so a bad minute lands on whichever arm was running.

**Interleaved** (``interleaved``, the default) runs every arm at once and takes their cells in turn, case by case
(case 1 on every arm, then case 2), under ONE width shared by the whole launch: the widest any of its runs executes
at alone. A provider's bad minute then lands on every arm alike, the launch loads its providers no more than one of
its runs would alone, and the arms' same cases run close together in time (exactly side by side at a width of 1,
within a few cells of each other wider). The width is one launch's: separate launches still load their providers
side by side. Every arm plays its cases in one order,
derived from the launch's id (:func:`~threetears.evals.run.runner.cell_execution_order`).

A launch declaring latency under test runs its arms one after another, the only schedule that reads latency with
nothing beside it (:func:`resolve_arm_schedule`). Every run records the schedule it ran under
(``EvalRun.arm_schedule``), a disclosure that is never part of its identity.

**The scheduler works over any number of runs** (:class:`CellInterleaver`): runs of one template or of several, with
case lists alike or not. Each run's loop hands its cells to its own lane (:meth:`CellInterleaver.lane`, a
:class:`~threetears.evals.run.executor.CellExecutor`), and no cell starts until every run of the launch has handed its
cells over or ended without doing so. A cell then starts whenever the launch is under its width, from the run that has
started the smallest share of its own cells (ties to the earlier arm), so runs of unequal size are spread over one
window too. Each run keeps every promise its executor owes it: its cells start in its own order, never more than its
own width run at once, and a cell that raises, or a cancel, stops that run alone.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections import deque
from collections.abc import Sequence
from fractions import Fraction
from functools import partial
from typing import Final

from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.run.executor import CellExecutor, CellWork, InProcessCellExecutor
from threetears.evals.schema.models import ArmSchedule

#: The schedule a launch runs its arms under when it names none and does not declare latency under test.
DEFAULT_ARM_SCHEDULE: Final[ArmSchedule] = "interleaved"


def resolve_arm_schedule(asked: ArmSchedule | None, *, measure_latency: bool) -> ArmSchedule:
    """The schedule a launch's arms run under: the one asked, or its default.

    :param asked: The schedule the launch named, or ``None`` for its default.
    :ptype asked: ArmSchedule | None
    :param measure_latency: Whether the launch declared latency under test.
    :ptype measure_latency: bool
    :return: ``asked`` when given; otherwise ``sequential`` under latency and :data:`DEFAULT_ARM_SCHEDULE` without.
    :rtype: ArmSchedule
    :raises ValidationFailedError: Latency is under test and the launch asked for its arms at once
        (``interleaved`` or ``concurrent``): latency is read clean only with nothing beside the cell.
    """
    if not measure_latency:
        return asked or DEFAULT_ARM_SCHEDULE
    if asked not in (None, "sequential"):
        raise ValidationFailedError(
            f"arm_schedule={asked!r} runs a launch's arms at once, and measure_latency reads each cell's latency with "
            "nothing beside it, so a launch declaring latency under test runs its arms one after another",
            remedy="drop arm_schedule (or pass 'sequential') to measure latency, or drop measure_latency",
        )
    return "sequential"


class _Lane:
    """One run's share of an interleaved launch: its cells, its width, and how far it has got."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.registered = False
        self.withdrawn = False
        self.closed = False
        self.pending: deque[CellWork] = deque()
        self.total = 0
        self.started = 0
        self.width = 1
        self.executor: CellExecutor | None = None
        self.context: contextvars.Context | None = None
        self.running: set[asyncio.Task[None]] = set()
        self.error: BaseException | None = None
        self.done: asyncio.Future[None] | None = None

    @property
    def startable(self) -> bool:
        return (
            self.registered
            and not self.closed
            and self.error is None
            and bool(self.pending)
            and len(self.running) < self.width
        )


class _LaneExecutor:
    """The :class:`~threetears.evals.run.executor.CellExecutor` one run of an interleaved launch hands its cells to."""

    def __init__(self, interleaver: CellInterleaver, run_id: str, executor: CellExecutor | None) -> None:
        self._interleaver = interleaver
        self._run_id = run_id
        self._executor = executor

    async def execute(self, cells: Sequence[CellWork], *, width: int) -> None:
        """Run the run's cells in turn with its siblings', never more than ``width`` of its own at once.

        :param cells: The run's cells, in the order they must start.
        :ptype cells: Sequence[CellWork]
        :param width: The most of the run's own cells that may run at once; at least 1.
        :ptype width: int
        :raises ValueError: ``width`` is below 1.
        :raises BaseException: The first exception one of the run's cells raised, once its cells in flight ended.
        """
        await self._interleaver.run_lane(self._run_id, cells, width=width, executor=self._executor)


class CellInterleaver:
    """Takes the cells of several runs in turn, case by case, under one width the runs share.

    Built empty with its launch; each run is expected in arm order (:meth:`expect`) as it is prepared, and the launch
    seals it (:meth:`seal`) as it starts them. A run that ends without handing over its cells — cancelled while
    queued, or refused before its loop began — is withdrawn (:meth:`withdraw`), so its siblings do not wait for it.
    """

    def __init__(self) -> None:
        """An interleaver expecting no run yet."""
        self._lanes: dict[str, _Lane] = {}
        self._sealed = False
        self._width: int | None = None

    def expect(self, run_id: str) -> None:
        """Expect a run's cells; runs are expected in arm order, which breaks ties between them.

        :param run_id: The run.
        :ptype run_id: str
        :raises RuntimeError: The interleaver is sealed, or already expects the run.
        """
        if self._sealed or run_id in self._lanes:
            raise RuntimeError(f"run {run_id} cannot join this launch's turns: it is sealed, or the run is already in")
        self._lanes[run_id] = _Lane(len(self._lanes))

    def seal(self) -> None:
        """Expect no further run: once every expected run has handed over its cells or ended, cells start."""
        self._sealed = True
        self._dispatch()

    def withdraw(self, run_id: str) -> None:
        """A run ended; if it never handed over its cells, its siblings stop waiting for it. Idempotent.

        :param run_id: The run.
        :ptype run_id: str
        """
        lane = self._lanes.get(run_id)
        if lane is not None and not lane.registered:
            lane.withdrawn = True
            self._dispatch()

    def lane(self, run_id: str, executor: CellExecutor | None) -> CellExecutor:
        """The executor a run hands its cells to.

        :param run_id: The run, one this interleaver expects.
        :ptype run_id: str
        :param executor: What runs each of the run's cells once its turn comes (a host's pool), or ``None`` to run
            it on this event loop.
        :ptype executor: CellExecutor | None
        :return: The run's lane.
        :rtype: CellExecutor
        """
        return _LaneExecutor(self, run_id, executor)

    @property
    def in_flight(self) -> int:
        """How many cells of the launch are running now, across every run."""
        return sum(len(lane.running) for lane in self._lanes.values())

    async def run_lane(
        self, run_id: str, cells: Sequence[CellWork], *, width: int, executor: CellExecutor | None
    ) -> None:
        """Run one run's cells in turn with its siblings' (what its lane's ``execute`` does).

        A run this interleaver does not expect, or one already withdrawn, runs its cells alone, as its own executor
        would.

        :param run_id: The run.
        :ptype run_id: str
        :param cells: Its cells, in the order they must start.
        :ptype cells: Sequence[CellWork]
        :param width: The most of its own cells that may run at once; at least 1.
        :ptype width: int
        :param executor: What runs each cell, or ``None`` for this event loop.
        :ptype executor: CellExecutor | None
        :raises ValueError: ``width`` is below 1.
        :raises BaseException: The first exception one of its cells raised, once its cells in flight ended.
        """
        if width < 1:
            raise ValueError(f"a run executes at least one cell at a time; got width={width}")
        lane = self._lanes.get(run_id)
        if lane is None or lane.registered or lane.withdrawn:
            await (executor or InProcessCellExecutor()).execute(cells, width=width)
            return
        lane.registered = True
        lane.pending = deque(cells)
        lane.total = len(cells)
        lane.width = width
        lane.executor = executor
        # Each cell runs in a copy of the run's own context (its job, its identity), never a sibling's.
        lane.context = contextvars.copy_context()
        done = lane.done = asyncio.get_running_loop().create_future()
        self._settle(lane)
        self._dispatch()
        try:
            # Only a signal that the lane settled, never its outcome: an exception parked on a future the cancel
            # below abandons would never be retrieved, and a cancel a cell raised would read as this caller's own.
            await done
        except asyncio.CancelledError:
            # The run was cancelled: none of its cells starts after this, and the ones running are cancelled and
            # waited for, each recording itself cancelled with its run, before the cancel goes on.
            lane.closed = True
            lane.pending.clear()
            running = list(lane.running)
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            self._dispatch()
            raise
        finally:
            lane.closed = True
        if lane.error is not None:
            raise lane.error

    def _ready(self) -> bool:
        return self._sealed and all(lane.registered or lane.withdrawn for lane in self._lanes.values())

    def _dispatch(self) -> None:
        """Start cells while the launch is under its width, each from the run furthest behind its own cells."""
        if not self._ready():
            return
        if self._width is None:
            # The widest any run executes at alone: the launch loads its providers no more than that run would.
            self._width = max((lane.width for lane in self._lanes.values() if lane.registered), default=1)
        while self.in_flight < self._width:
            startable = [lane for lane in self._lanes.values() if lane.startable]
            if not startable:
                return
            self._start(min(startable, key=lambda lane: (Fraction(lane.started, lane.total), lane.index)))

    def _start(self, lane: _Lane) -> None:
        cell = lane.pending.popleft()
        lane.started += 1
        assert lane.context is not None  # set as the lane registered
        task = asyncio.get_running_loop().create_task(_run_cell(cell, lane.executor), context=lane.context.copy())
        lane.running.add(task)
        task.add_done_callback(partial(self._cell_ended, lane))

    def _cell_ended(self, lane: _Lane, task: asyncio.Task[None]) -> None:
        lane.running.discard(task)
        error: BaseException | None = None
        if task.cancelled():
            # This interleaver cancels a lane's cells only once the lane has failed or closed. A cell that ended
            # cancelled otherwise raised the cancel itself (the runner re-raises one it recorded, a kind's scope or a
            # host's pool can raise one), and its run must stop on it as a run executing alone would.
            if not lane.closed and lane.error is None:
                error = asyncio.CancelledError(f"a cell of run lane {lane.index} ended cancelled")
        else:
            error = task.exception()
        if error is not None and lane.error is None:
            # The run's first failure: none of its cells starts after it, and its cells in flight are cancelled.
            lane.error = error
            lane.pending.clear()
            for running in lane.running:
                running.cancel()
        self._settle(lane)
        self._dispatch()

    def _settle(self, lane: _Lane) -> None:
        """End the run's wait once none of its cells is left to start or still running."""
        if lane.closed or lane.pending or lane.running or lane.done is None or lane.done.done():
            return
        lane.done.set_result(None)


async def _run_cell(cell: CellWork, executor: CellExecutor | None) -> None:
    """Run one cell, through the run's own executor when it has one."""
    if executor is None:
        await cell()
    else:
        await executor.execute([cell], width=1)


__all__ = ["DEFAULT_ARM_SCHEDULE", "CellInterleaver", "resolve_arm_schedule"]
