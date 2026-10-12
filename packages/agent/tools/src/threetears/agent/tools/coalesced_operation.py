"""A pod's long operation, coordinated across its replicas: one unit of work at a time, none lost, reported by all.

**What it puts together.** A tool pod runs as several replicas, and a tool call is answered by
whichever takes it. A load, a refresh or a rebuild that a pod runs on request therefore needs all
of the following, and :class:`CoalescedOperation` is the one object that holds them:

- **One unit of work at a time, everywhere, and no request lost.** The unit runs under a lease in
  the replicas' shared bucket (:class:`~threetears.core.coordination.coalesced_run.CoalescedRun` on
  a :class:`~threetears.core.coordination.lease.KVLease`). A request made while a unit runs, here or
  on another replica, is one key beside the lease; the holder takes it when its unit ends and runs
  once more, however many requests there were.
- **In the background, started and reported by tools.** The drain of those requests is a
  :class:`~threetears.agent.tools.background_operation.BackgroundOperation`
  (:attr:`CoalescedOperation.operation`), which a pod registers ``RequestOperationTool`` and
  ``OperationStatusTool`` on.
- **The newest finished run, reported by every replica.** Each drain that ran a unit is written
  beside the lease (:class:`~threetears.agent.tools.background_operation.SharedOutcome`).
- **A run in progress, reported by every replica.** A replica that is not running asks the lease,
  and shows the holder's run with the progress it last reported
  (:class:`~threetears.agent.tools.background_operation.SharedProgress`,
  :meth:`CoalescedOperation.report_progress`).
- **A request left behind a holder that died is run.** A drain that found the lease held leaves its
  request to the holder, and looks again once the lease has had its lifetime.
- **A stop that ends all of it,** and says whether it did.

**A drain that ran nothing is not a run.** A drain started on a replica that cannot hold the lease,
or for a request the drain before it already took, runs no unit. It ends at once, and it is not the
operation's last run: it is not written to the shared record, it does not replace this replica's
last drain that did run a unit, and the status goes on reporting the last real run (and the run in
progress on the holder). A pod writes no branch for it.

Usage::

    load = CoalescedOperation("enr-load", lease, "enr/refresh", refresher.refresh,
                              ttl=timedelta(seconds=60), renew_every=timedelta(seconds=20),
                              render_result=rendered)
    server.register(RequestOperationTool(name="enr.reload", description="...",
                                         operation=lambda: load.operation, status_tool="enr.load_status"))
    server.register(OperationStatusTool(name="enr.load_status", description="...",
                                        operation=lambda: load.operation, render_result=rendered))
    load.start_when_needed(refresher.needs_load, retry_on=(DataLayerUnavailableError,), retry_seconds=30)
    ...
    load.report_progress({"summary": "Loading · 37 of 51 states"})   # from the unit, as it moves
    ...
    finished = await load.stop(within=20)   # on shutdown, before the NATS connection drains
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from threetears.core.coordination.coalesced_run import CoalescedRun
from threetears.observe import get_logger

from threetears.agent.tools.background_operation import (
    BackgroundOperation,
    RunningElsewhere,
    SharedOutcome,
    SharedProgress,
)

if TYPE_CHECKING:
    from threetears.core.coordination.lease import KVLease

__all__ = ["CoalescedOperation", "Drained", "rendered_drain"]

log = get_logger(__name__)

UnitT = TypeVar("UnitT")


@dataclass(frozen=True)
class Drained(Generic[UnitT]):
    """what one run of the operation did on this replica: the units it ran, each with its own result.

    :ivar results: what each unit returned, in the order they ran: one for the request the run was
        started for, and one more for a request recorded while it ran; empty when it ran none
        (another replica holds the lease and runs what was asked, or nothing was left to run)
    """

    results: tuple[UnitT, ...] = ()

    @property
    def runs(self) -> int:
        """how many units this run ran here.

        :return: the count; 0 for a run that ran nothing
        :rtype: int
        """
        return len(self.results)


def rendered_drain(drained: Drained[Any]) -> dict[str, Any]:
    """a run as JSON values, when its units' results already are: how the operation renders one by default.

    :param drained: the run
    :ptype drained: Drained[Any]
    :return: ``runs``, and ``results`` (each unit's own, in order)
    :rtype: dict[str, Any]
    """
    return {"runs": drained.runs, "results": list(drained.results)}


def _ran(drained: Drained[Any]) -> bool:
    """whether a run ran a unit here, and so is one to record as the operation's last run.

    :param drained: the run
    :ptype drained: Drained[Any]
    :return: True when it ran at least one unit
    :rtype: bool
    """
    return drained.runs > 0


class CoalescedOperation(Generic[UnitT]):
    """one unit of work, run in the background by one replica at a time, requests coalesced, reported by all.

    **Keys.** Everything is kept beside the lease, in the lease's bucket under its owner scope:
    ``<key>`` (the lease), ``<key>.requested`` (a run is wanted), ``<key>.outcome`` (the newest
    finished run) and ``<key>.progress`` (the run in progress). Replicas of one pod are built with
    the same ``key`` over lease factories of the same scope, each with its own ``pod_id``.

    **Stopping.** :meth:`stop` ends when the unit in progress has ended, not before: the coalesced
    run holds a cancellation until its unit has stopped, so that the lease is never let go under
    work that is still writing. A pod bounds it with ``stop(within=seconds)``, which answers whether
    everything ended in time and leaves the stop running if not. ``asyncio.wait_for(load.stop(), t)``
    does NOT bound it: on its timeout ``wait_for`` cancels the stop and then waits for it to end,
    which it does only when the unit has.

    :param name: the operation's name, for task names and logs
    :ptype name: str
    :param lease: the pod's lease factory; every replica's is over the same bucket and owner scope
    :ptype lease: KVLease
    :param key: the operation's lease key; its request, outcome and progress are the keys beside it
    :ptype key: str
    :param unit: one unit of the work (a load, a refresh), run under the lease; what it returns is
        that unit's own result
    :ptype unit: Callable[[], Awaitable[UnitT]]
    :param ttl: how long the lease outlives a missed renewal: how long a holder that died keeps the
        other replicas out, and is still shown as running
    :ptype ttl: timedelta
    :param renew_every: how often the lease is renewed while a unit runs; shorter than ``ttl``
    :ptype renew_every: timedelta
    :param lease_retry_seconds: how long a run that found the lease held waits before it looks
        again; the lease's lifetime (``ttl``) when None, so a lease its dead holder left has lapsed
    :ptype lease_retry_seconds: float | None
    :param render_result: turns a run (:class:`Drained`) into JSON values, for the shared record;
        give the status tool the same one. :func:`rendered_drain` when None
    :ptype render_result: Callable[[Drained[UnitT]], Any] | None
    :param ran_nothing: called when this replica learns the work was not its to run: after a run
        here that ran no unit, and after the later look that finds the request taken. For a pod
        whose own state follows a unit another replica ran. One that raises is logged
    :ptype ran_nothing: Callable[[], None] | None
    :raises ValueError: when ``renew_every`` is not shorter than ``ttl``
    """

    def __init__(
        self,
        name: str,
        lease: KVLease,
        key: str,
        unit: Callable[[], Awaitable[UnitT]],
        *,
        ttl: timedelta,
        renew_every: timedelta,
        lease_retry_seconds: float | None = None,
        render_result: Callable[[Drained[UnitT]], Any] | None = None,
        ran_nothing: Callable[[], None] | None = None,
    ) -> None:
        self._name = name
        self._lease = lease
        self._key = key
        self._unit = unit
        self._lease_retry_seconds = ttl.total_seconds() if lease_retry_seconds is None else lease_retry_seconds
        self._render_result: Callable[[Drained[UnitT]], Any] = render_result or rendered_drain
        self._ran_nothing = ran_nothing
        # set by stop: nothing takes the lease, waits to try it again or writes progress from then on
        self._stopping = False
        self._stop: asyncio.Future[None] | None = None
        # the wait of a run that found the lease held, for stop to cancel; one at a time
        self._retry: asyncio.Task[None] | None = None
        # whether a unit runs here now: only then is progress this replica's to write
        self._in_unit = False
        self._latest: Mapping[str, Any] | None = None
        self._writing: asyncio.Task[None] | None = None
        self._run = CoalescedRun(lease, key, self._run_unit, ttl=ttl, renew_every=renew_every)
        self._progress = SharedProgress(lease, key)
        # the operation owns its request step, so a reload tool's request and the run it starts are
        # one object's
        self._operation: BackgroundOperation[Drained[UnitT]] = BackgroundOperation(
            name,
            self._drain,
            request=self._run.request,
            shared=SharedOutcome(lease, key, render_result=self._render_result, ran=_ran),
            shared_progress=self._progress,
        )

    @property
    def name(self) -> str:
        """the operation's name.

        :return: the name
        :rtype: str
        """
        return self._name

    @property
    def operation(self) -> BackgroundOperation[Drained[UnitT]]:
        """the background operation the request and status tools answer for.

        :return: the operation
        :rtype: BackgroundOperation[Drained[UnitT]]
        """
        return self._operation

    @property
    def render_result(self) -> Callable[[Drained[UnitT]], Any]:
        """how a run is rendered for the shared record; the status tool is given the same.

        :return: the renderer
        :rtype: Callable[[Drained[UnitT]], Any]
        """
        return self._render_result

    @property
    def stopping(self) -> bool:
        """whether :meth:`stop` has been called: no unit starts here from then on.

        :return: True once stopping
        :rtype: bool
        """
        return self._stopping

    async def request(self) -> bool:
        """record that a unit is wanted, then start a run here, or have the run in progress here run once more.

        :return: True when a run was started here; False when the run in progress here will run once more
        :rtype: bool
        """
        return await self._operation.request()

    async def requested(self) -> bool:
        """whether a unit is wanted and no replica has taken it yet.

        :return: True while a request is recorded
        :rtype: bool
        """
        return await self._run.requested()

    async def running_elsewhere(self) -> RunningElsewhere | None:
        """the run another replica makes now, with its progress; asked of the lease, never remembered.

        :return: the run; None when this replica runs or no other replica holds the lease
        :rtype: RunningElsewhere | None
        """
        return await self._operation.running_elsewhere()

    def start_when_needed(
        self,
        needed: Callable[[], Awaitable[bool]],
        *,
        retry_on: tuple[type[BaseException], ...],
        retry_seconds: float,
    ) -> bool:
        """in the background: once ``needed`` answers, start a run if a unit is wanted.

        For a pod's first run, when it cannot know yet whether one is needed (its tables are out of
        reach until the hub grants it storage). A unit is wanted when ``needed`` says so, or when a
        request is still recorded: one no replica has run (the unit that took it was stopped by a
        shutdown, which records it again). A needed first run is a request like any other, so the
        drain runs it, and replicas that start together ask for one unit, not one each. No request
        is recorded while a run is in progress here: that run is the first run, and a second
        request would have it followed by another.

        Only the errors in ``retry_on`` are retried, as
        :meth:`~threetears.agent.tools.background_operation.BackgroundOperation.start_when_needed` does.

        :param needed: whether the work has never been done (a table is empty)
        :ptype needed: Callable[[], Awaitable[bool]]
        :param retry_on: the errors that mean "not reachable yet", retried after ``retry_seconds``
        :ptype retry_on: tuple[type[BaseException], ...]
        :param retry_seconds: the wait between attempts
        :ptype retry_seconds: float
        :return: True when the wait started; False when one was already deciding
        :rtype: bool
        """

        async def wanted() -> bool:
            answer = await needed()
            if not answer:
                answer = await self._run.requested()
            elif not self._operation.running:
                await self._run.request()
            return answer

        return self._operation.start_when_needed(wanted, retry_on=retry_on, retry_seconds=retry_seconds)

    def report_progress(self, progress: Mapping[str, Any]) -> None:
        """the unit running here moved: have every replica show this as its progress.

        Called by the unit (or what it reports to) as often as it likes: the latest is written, one
        write at a time, and a write that fails is logged and the unit goes on. Outside a unit
        running here, and once stopping, it does nothing: the progress is the lease holder's to write.

        :param progress: what the unit is doing, as JSON values; a short ``summary`` line is what
            the status tool's sentence shows. A ``holder`` field is the record's own and is replaced
        :ptype progress: Mapping[str, Any]
        :return: nothing
        :rtype: None
        """
        if self._in_unit and not self._stopping:
            self._latest = dict(progress)
            if self._writing is None or self._writing.done():
                self._writing = asyncio.get_running_loop().create_task(
                    self._write_latest(), name=f"{self._name}-progress"
                )

    async def stop(self, *, within: float | None = None) -> bool:
        """end everything: the wait to start, the unit in progress, the wait to try a held lease again.

        A unit cancelled here is stopped before its lease is let go, and its request is recorded
        again, so the next replica to start (or the next request) runs it. A run started after this
        (a request answered while the pod shuts down) takes neither the lease nor the request: the
        connection it would hold them on is about to close. Call it while that connection still
        answers, ahead of the pod's NATS drain; stopped after, it can neither let the lease go nor
        record the request, and the other replicas wait out the lease for a unit nobody runs.

        Asked twice, it is one stop: the second call waits on the first. It never raises for what
        stopping raised, which is logged.

        :param within: how long to wait, in seconds, for everything to end; None waits until it has.
            The stop goes on past it: the unit is still being stopped, and its lease is let go when
            it ends, or lapses
        :ptype within: float | None
        :return: True when everything has ended; False when ``within`` passed first, or stopping failed
        :rtype: bool
        """
        self._stopping = True
        if self._stop is None:
            self._stop = asyncio.ensure_future(self._stop_all())
        ended, _running = await asyncio.wait({self._stop}, timeout=within)
        finished = bool(ended)
        if not finished:
            log.warning(
                "the operation did not stop within its bound; its lease is let go when its unit ends, or lapses",
                extra={"extra_data": {"operation": self._name, "seconds": within}},
            )
        elif not self._stop.cancelled() and self._stop.exception() is not None:
            failure = self._stop.exception()
            finished = False
            log.error(
                "the operation failed while stopping",
                extra={"extra_data": {"operation": self._name, "error": f"{type(failure).__name__}: {failure}"}},
            )
        return finished

    async def _stop_all(self) -> None:
        """the body of :meth:`stop`: the run first, then the waits and writes it could have left.

        :return: nothing
        :rtype: None
        """
        # the run first: once it has ended nothing adds a wait or a write, so each is cancelled below
        await self._operation.stop()
        for task in (self._retry, self._writing):
            if task is not None and not task.done():
                task.cancel()
                # NOSILENT: the cancellation requested on the line above, awaited so the task ends first
                with suppress(asyncio.CancelledError):
                    await task

    async def _drain(self) -> Drained[UnitT]:
        """the operation's body: run the unit once per recorded request while this replica holds the lease.

        :return: what this run did here
        :rtype: Drained[UnitT]
        :raises Exception: what a unit raised; its request is taken and not retried
        """
        results: list[UnitT] = []
        if self._stopping:
            log.info(
                "the pod is stopping; the request is left for the next replica to start",
                extra={"extra_data": {"operation": self._name, "lease": self._key}},
            )
        elif not (results := await self._run.drain_results()) and not self._stopping:
            if await self._left_to_a_holder():
                # another replica holds the lease and runs the request; if it never does, this
                # replica does, later
                if self._retry is None or self._retry.done():
                    self._retry = asyncio.get_running_loop().create_task(self._look_again(), name=f"{self._name}-retry")
                log.info(
                    "no unit ran on this replica; the request is left for the replica that holds the lease",
                    extra={"extra_data": {"operation": self._name, "lease": self._key}},
                )
            else:
                log.info(
                    "no unit ran on this replica; the request it was started for was already taken",
                    extra={"extra_data": {"operation": self._name, "lease": self._key}},
                )
            self._tell_ran_nothing()
        return Drained(results=tuple(results))

    async def _left_to_a_holder(self) -> bool:
        """whether a run that ran nothing left its request behind: another replica holds the lease.

        :return: True while a request is recorded; also when the bucket cannot be asked, since a
            request behind a dead holder must not go unlooked for
        :rtype: bool
        """
        left = True
        try:
            left = await self._run.requested()
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- the run has ended; whatever the bucket's client raises, the request may still be recorded, so it is taken for recorded and looked for again; logged
            log.warning(
                "cannot tell whether the request is still recorded; it is taken for recorded",
                extra={"extra_data": {"operation": self._name, "error": f"{type(exc).__name__}: {exc}"}},
            )
        return left

    async def _run_unit(self) -> UnitT:
        """one unit, under the lease, its progress shared while it runs and removed when it ends.

        :return: the unit's own result
        :rtype: UnitT
        """
        self._in_unit = True
        self._latest = None
        # the entry is this holder's from the start, so a run before it is never shown as this one's
        await self._write({})
        try:
            result = await self._unit()
        finally:
            self._in_unit = False
            await self._end_progress()
        return result

    async def _write(self, progress: Mapping[str, Any]) -> bool:
        """write this replica's progress; a failure is logged, and the unit goes on.

        :param progress: the progress
        :ptype progress: Mapping[str, Any]
        :return: True when it was written
        :rtype: bool
        """
        written = False
        try:
            await self._progress.publish(progress)
            written = True
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- sharing progress is a courtesy to the other replicas; whatever the bucket's client or the encoding raises, this replica's unit goes on; logged
            log.warning(
                "the unit's progress could not be shared",
                extra={"extra_data": {"operation": self._name, "error": f"{type(exc).__name__}: {exc}"}},
            )
        return written

    async def _write_latest(self) -> None:
        """write the latest progress reported, again while a newer one was reported during the write.

        :return: nothing
        :rtype: None
        """
        written: Mapping[str, Any] | None = None
        failed = False
        while not failed and self._latest is not None and self._latest is not written:
            current = self._latest
            failed = not await self._write(current)
            written = current

    async def _end_progress(self) -> None:
        """the unit has ended: let the write in flight land, then remove this replica's progress.

        :return: nothing
        :rtype: None
        """
        writing = self._writing
        if writing is not None and not writing.done():
            await asyncio.wait({writing})
        try:
            await self._progress.clear()
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- a progress entry that cannot be removed is ignored by readers once the lease is let go; the unit's own ending must not be changed by it; logged
            log.warning(
                "the unit's progress could not be removed; readers ignore it once the lease is let go",
                extra={"extra_data": {"operation": self._name, "error": f"{type(exc).__name__}: {exc}"}},
            )

    async def _look_again(self) -> None:
        """after the lease's lifetime, run the request this replica left, if no holder is there to.

        A run that found another replica holding the lease leaves its request for that replica. A
        holder that is alive takes it, and this finds none. While a holder is alive and has not
        taken it yet, this waits another lifetime rather than start a run that could not hold the
        lease. If the holder died, nothing takes the request until its lease lapses; then this
        replica runs it. While the bucket cannot be asked this waits again: nothing else would run
        a request left behind a dead holder.

        :return: nothing
        :rtype: None
        """
        asked = False
        settled = False
        while not settled:
            await asyncio.sleep(self._lease_retry_seconds)
            try:
                asked = await self._run.requested()
                held = asked and await self._lease.holder(self._key) is not None
                settled = not held
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- a wait with no caller: whatever the bucket's client raises, the request may still be behind a dead holder, so it is logged and asked again
                log.warning(
                    "cannot tell whether the request is still recorded; asking again",
                    extra={
                        "extra_data": {
                            "operation": self._name,
                            "lease": self._key,
                            "error": f"{type(exc).__name__}: {exc}",
                            "retry_seconds": self._lease_retry_seconds,
                        }
                    },
                )
        if asked:
            # a run started while the pod is stopping takes neither the lease nor the request
            self._operation.start()
        else:
            self._tell_ran_nothing()

    def _tell_ran_nothing(self) -> None:
        """tell the pod the work was not this replica's to run; a callback that raises is logged.

        :return: nothing
        :rtype: None
        """
        if self._ran_nothing is not None:
            try:
                self._ran_nothing()
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- the pod's own callback, called from a run and a wait with no caller; whatever it raises must not fail either; logged
                log.warning(
                    "the callback for a run that ran nothing failed",
                    extra={"extra_data": {"operation": self._name, "error": f"{type(exc).__name__}: {exc}"}},
                )
