"""A pod's long operation, coordinated across its replicas: one unit of work at a time, none lost, reported by all.

**What it puts together.** A tool pod runs as several replicas, and a tool call is answered by
whichever takes it. A load, a refresh or a rebuild that a pod runs on request therefore needs all
of the following, and :class:`CoalescedOperation` is the one object that holds them:

- **One unit of work at a time, everywhere, and no request lost while its holder lives.** The unit
  runs under a lease in the replicas' shared bucket
  (:class:`~threetears.core.coordination.coalesced_run.CoalescedRun` on a
  :class:`~threetears.core.coordination.lease.KVLease`). A request made while a unit runs, here or
  on another replica, is one key beside the lease; the holder takes it when its unit ends and runs
  once more, however many requests there were. A unit that is stopped, or whose lease is lost, has
  its request recorded again and run later. The one request that is lost is the one a holder had
  taken when its process was killed before its unit ended: the request is taken before the unit
  runs, and a process that dies records nothing. The next request, or a first run that is still
  needed when a pod starts, runs it.
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
- **A request left behind is run.** A drain that found the lease held leaves its request to the
  holder, and one whose lease was lost has its request recorded again with nobody draining; either
  looks again once the lease has had its lifetime, and runs the request if no live holder is there to.
- **A stop that ends all of it,** and says whether it did.

**A drain that ran nothing is not a run.** A drain started on a replica that cannot hold the lease,
or for a request the drain before it already took, runs no unit. It ends at once, and it is not the
operation's last run: it is not written to the shared record and it is never this replica's last
run, so the status goes on reporting the last real run (and the run in progress on the holder). A
replica that has run no unit and sees none recorded reports ``idle``, whatever runs of that kind it
has made. A pod writes no branch for it.

**The same line whichever replica answers.** What the unit reports (:meth:`CoalescedOperation.report_progress`)
is what the replica running it shows of its own run and what every other replica reads of it, so a
pod gives the status tool no ``progress=`` of its own for the run's progress. A unit that holds its
lease past its work says so (:meth:`CoalescedOperation.work_over`), and from then on is no run in
progress to the others.

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
    load.report_progress(progress.render)   # from the unit, as it moves: rendered only when written or shown
    ...
    finished = await load.stop(within=20)   # on shutdown, before the NATS connection drains
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from threetears.core.coordination.coalesced_run import CoalescedRun
from threetears.core.coordination.lease import LeaseLost
from threetears.observe import cancel_and_wait, contain, get_logger

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
_AnswerT = TypeVar("_AnswerT")


class _BesideTheLeaseUnreachable(Exception):
    """the bucket beside the lease could not be asked or written while the first run was being decided."""


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
        # how many times the unit running here has reported, so the writer knows a newer report from
        # the one it wrote without comparing what they say
        self._reports = 0
        # whether a unit of the drain running here said its work is over: an entry saying so is left
        # until the lease is let go, since removing it under the lease would read as a run in
        # progress again
        self._over = False
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
        drain runs it. No request is recorded while a run is in progress here: that run is the
        first run, and a second request would have it followed by another.

        **Replicas that start together run one unit, not one each.** A replica whose ``needed``
        says yes while the lease is held (and no run is in progress here) records nothing: the run
        in progress under it is the first run. It waits, a lease's lifetime at a time, and asks ``needed`` again once no
        live holder remains; so a first run that failed, or did not do what was needed, is still
        run. (Two replicas that both find no holder in the same instant each record the request,
        which is one request unless the second lands after the first was taken.)

        Only the errors in ``retry_on`` end nothing and are retried, as
        :meth:`~threetears.agent.tools.background_operation.BackgroundOperation.start_when_needed`
        does; and so is the bucket beside the lease when it cannot be asked or written, whatever
        its client raises: a pod names the errors of its own ``needed``, not the bucket's.

        :param needed: whether the work has never been done (a table is empty)
        :ptype needed: Callable[[], Awaitable[bool]]
        :param retry_on: the errors of ``needed`` that mean "not reachable yet", retried after ``retry_seconds``
        :ptype retry_on: tuple[type[BaseException], ...]
        :param retry_seconds: the wait between attempts
        :ptype retry_seconds: float
        :return: True when the wait started; False when one was already deciding
        :rtype: bool
        """

        async def wanted() -> bool:
            answer = await needed()
            while answer and not self._operation.running and await self._beside_the_lease(self._held()):
                # the run in progress there is the first run; whether one is still needed is asked
                # again once it is over, or its holder is gone
                await asyncio.sleep(self._lease_retry_seconds)
                answer = await needed()
            if not answer:
                answer = await self._beside_the_lease(self._run.requested())
            elif not self._operation.running:
                await self._beside_the_lease(self._run.request())
            return answer

        return self._operation.start_when_needed(
            wanted, retry_on=(*retry_on, _BesideTheLeaseUnreachable), retry_seconds=retry_seconds
        )

    async def _held(self) -> bool:
        """whether the operation's lease is held now, by a holder that is not known to be gone.

        :return: True while a live holder is named
        :rtype: bool
        """
        return await self._lease.holder(self._key) is not None

    async def _beside_the_lease(self, asked: Awaitable[_AnswerT]) -> _AnswerT:
        """what a read or write of the bucket beside the lease answers, its failure made one the first run's wait retries.

        :param asked: the read or write
        :ptype asked: Awaitable[_AnswerT]
        :return: its answer
        :rtype: _AnswerT
        :raises _BesideTheLeaseUnreachable: whatever it raised
        """
        try:
            answer = await asked
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- whatever the bucket's client raises, the first run's wait must go on rather than end for good: raised as the one error that wait retries, which logs it
            raise _BesideTheLeaseUnreachable(f"{type(exc).__name__}: {exc}") from exc
        return answer

    def report_progress(self, progress: Mapping[str, Any] | Callable[[], Mapping[str, Any]]) -> None:
        """the unit running here moved: have every replica show this as its progress.

        Called by the unit (or what it reports to) as often as it likes: the latest is written, one
        write at a time, and a write that fails is logged and the unit goes on. This replica's own
        status shows it too (:meth:`BackgroundOperation.status_across_replicas`), so a reader gets
        the same line whichever replica answers. Outside a unit running here, after
        :meth:`work_over`, and once stopping, it does nothing: the progress is the lease holder's
        to write, while its work goes on.

        Given a callable, the progress is asked of it only when it is about to be written or
        shown, never when the call is dropped, and each time anew: a pod passes what renders its
        progress rather than rendering on every step. One that raises is logged, and nothing is
        written or shown for it.

        :param progress: what the unit is doing, as JSON values, or what answers that when asked; a
            short ``summary`` line is what the status tool's sentence shows. The ``holder`` and
            ``work_over`` fields are the record's own and are replaced
        :ptype progress: Mapping[str, Any] | Callable[[], Mapping[str, Any]]
        :return: nothing
        :rtype: None
        """
        if self._in_unit and not self._stopping:
            self._progress.report(progress)
            self._reports += 1
            if self._writing is None or self._writing.done():
                self._writing = asyncio.get_running_loop().create_task(
                    self._write_latest(), name=f"{self._name}-progress"
                )

    async def work_over(self) -> None:
        """the unit running here says its work is over, though it has not returned and still holds the lease.

        For a unit that does something after its work whose length is not the work's (it waits for
        what it wrote to be answered from, say). From here until the unit returns, the other
        replicas no longer see a run in progress: the progress entry says the work is over, which a
        reader takes over the held lease. This replica still reports its run as running (it has not
        ended, and a request now still follows it), with no progress. Reports made after this are
        dropped. Outside a unit running here it does nothing; a write that fails is logged, and the
        other replicas go on seeing the run until the lease is let go.

        :return: nothing
        :rtype: None
        """
        if self._in_unit:
            self._in_unit = False
            self._over = True
            self._progress.report(None)
            writing = self._writing
            if writing is not None and not writing.done():
                await asyncio.wait({writing})
            try:
                await self._progress.over()
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- a courtesy to the other replicas, which see the run until the lease is let go without it; whatever the bucket's client raises, the unit goes on; logged
                contain(
                    exc,
                    log=log,
                    what="could not tell the other replicas the unit's work is over; they see it until the lease is let go",
                    extra={"operation": self._name, "error": f"{type(exc).__name__}: {exc}"},
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
                await cancel_and_wait(task)

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
        else:
            try:
                results = await self._run.drain_results()
            except LeaseLost:
                # the unit was cancelled and its request recorded again, with nobody draining: this
                # replica looks for it again, as it does for one it left to a holder
                self._look_again_later()
                raise
            finally:
                if self._over:
                    # the lease is let go: the entry saying the work was over has nothing left to say
                    self._over = False
                    await self._clear_progress()
            if not results and not self._stopping:
                if await self._left_to_a_holder():
                    # another replica holds the lease and runs the request; if it never does, this
                    # replica does, later
                    self._look_again_later()
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

    def _look_again_later(self) -> None:
        """have this replica look for the recorded request again, unless it is already waiting to.

        :return: nothing
        :rtype: None
        """
        if self._retry is None or self._retry.done():
            self._retry = asyncio.get_running_loop().create_task(self._look_again(), name=f"{self._name}-retry")

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
            contain(
                exc,
                log=log,
                what="cannot tell whether the request is still recorded; it is taken for recorded",
                extra={"operation": self._name, "error": f"{type(exc).__name__}: {exc}"},
            )
        return left

    async def _run_unit(self) -> UnitT:
        """one unit, under the lease, its progress shared while it runs and removed when it ends.

        :return: the unit's own result
        :rtype: UnitT
        """
        self._in_unit = True
        # a unit that starts is a run in progress, whatever the unit before it said of its own work
        self._over = False
        # the entries of this unit name the holding of the lease it runs under, so one left by an
        # earlier holding (this replica's own included) is never taken for this run's
        await self._begin_progress()
        # the entry is this holder's from the start, so a held lease reads as a run in progress
        # with nothing reported yet
        self._progress.report({})
        await self._write_reported()
        try:
            result = await self._unit()
        finally:
            self._in_unit = False
            self._progress.report(None)
            await self._end_progress()
        return result

    async def _begin_progress(self) -> None:
        """learn the holding of the lease this unit's entries are written under; a failure is logged.

        :return: nothing
        :rtype: None
        """
        try:
            await self._progress.begin()
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- sharing progress is a courtesy to the other replicas; whatever the bucket's client raises, this replica's unit goes on, its entries naming no holding; logged
            contain(
                exc,
                log=log,
                what="cannot read the lease the unit runs under; its progress names no holding of it",
                extra={"operation": self._name, "error": f"{type(exc).__name__}: {exc}"},
            )

    async def _write_reported(self) -> bool:
        """write what the unit running here last reported; a failure is logged, and the unit goes on.

        :return: True when it was written; False when it could not be, or no report stands
        :rtype: bool
        """
        written = False
        try:
            reported = self._progress.reported()
            if reported is not None:
                await self._progress.publish(reported)
                written = True
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- sharing progress is a courtesy to the other replicas; whatever the bucket's client, the encoding or the pod's own rendering raises, this replica's unit goes on; logged
            contain(
                exc,
                log=log,
                what="the unit's progress could not be shared",
                extra={"operation": self._name, "error": f"{type(exc).__name__}: {exc}"},
            )
        return written

    async def _write_latest(self) -> None:
        """write the latest progress reported, again while a newer one was reported during the write.

        :return: nothing
        :rtype: None
        """
        written = 0
        failed = False
        while not failed and written != self._reports:
            # taken before the write: a report made during it is a newer one, written next
            written = self._reports
            failed = not await self._write_reported()

    async def _end_progress(self) -> None:
        """the unit has ended: let the write in flight land, then remove this replica's progress.

        An entry saying the work is over is left: the lease is still held, and without the entry a
        reader would take the held lease for a run in progress. The drain removes it once the lease
        is let go.

        :return: nothing
        :rtype: None
        """
        writing = self._writing
        if writing is not None and not writing.done():
            await asyncio.wait({writing})
        if not self._over:
            await self._clear_progress()

    async def _clear_progress(self) -> None:
        """remove this replica's progress entry; a failure is logged.

        :return: nothing
        :rtype: None
        """
        try:
            await self._progress.clear()
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- an entry that cannot be removed is no run's once the lease is let go, and names the holding it was written under so no later one takes it for its own; the unit's own ending must not be changed by it; logged
            contain(
                exc,
                log=log,
                what="the unit's progress could not be removed; it names the holding it was written under, "
                "so it is no later run's",
                extra={"operation": self._name, "error": f"{type(exc).__name__}: {exc}"},
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
                contain(
                    exc,
                    log=log,
                    what="cannot tell whether the request is still recorded; asking again",
                    extra={
                        "operation": self._name,
                        "lease": self._key,
                        "error": f"{type(exc).__name__}: {exc}",
                        "retry_seconds": self._lease_retry_seconds,
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
                contain(
                    exc,
                    log=log,
                    what="the callback for a run that ran nothing failed",
                    extra={"operation": self._name, "error": f"{type(exc).__name__}: {exc}"},
                )
