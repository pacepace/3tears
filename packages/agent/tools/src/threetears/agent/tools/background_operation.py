"""A long operation a pod runs in the background: started (or requested) by one tool, reported by another.

**Why two tools, not one.** Some operations a tool pod runs on request take minutes (a
load from a warehouse, a rebuild of every map layer), longer than the hub waits on a tool
call. A tool that awaited one would answer every successful run with a timeout. So the
start tool begins the run and answers at once, and the status tool says whether it is
running and how the last one ended.

**One run at a time, and one way in.** :meth:`BackgroundOperation.start` is the only way
to begin a run, whether a pod starts one itself (a first load when its tables are empty,
:meth:`BackgroundOperation.start_when_needed`) or a caller asks through a tool, so
:attr:`~BackgroundOperation.running` and :attr:`~BackgroundOperation.last` describe every
run. Two tools start one, for two kinds of operation:

- :class:`StartOperationTool`, for an operation a second run of would repeat: a start while one
  runs is refused, answered :data:`~threetears.agent.tools.base_tool.CONFLICT`.
- :class:`RequestOperationTool`, for an operation that drains requests (its body a
  ``CoalescedRun.drain``, built with ``request=``): :meth:`BackgroundOperation.request` records the
  request and starts a run, and a request while one runs is never refused nor lost: the run in
  progress is followed by one more, which takes it (the drain may already have made its last look
  for requests).

:class:`OperationStatusTool` reports either kind.

**A run that raises is kept as failed, with its reason.** The run is a background task
with no caller to raise to, so its exception is recorded on the outcome and logged; the
status tool reports it until the next run ends.

**The outcome lives in this process, and, when asked, beside the operation's lease.** A tool pod
runs as several replicas and a status call is answered by whichever takes it, so an outcome kept
only in the process that ran makes the answer depend on the replica. An operation built with
``shared=`` (:class:`SharedOutcome`) also writes each finished run to one key in its lease's bucket,
and :class:`OperationStatusTool` reports the newer of the replica's own last run and that record,
saying so when the run was another replica's. The newest run is the one that finished last; nothing
expires it. Without ``shared=`` a replica knows only its own runs, and a restarted pod reports
``idle`` until its next run ends. An operation whose result must be queried, or kept run by run,
records it in its own tables; this keeps only what the status tool reports.

**A run that did nothing is not the newest run.** An operation run over a ``CoalescedRun`` returns
at once, with a result saying so, on a replica that could not hold the lease: it ended, and it ran
nothing. Recorded, each such return would stand as the newest outcome and hide the load the other
replica is still running, or the one that last failed. ``SharedOutcome(ran=)`` says which results
are runs that did the work; one that is not is never written, and the replica it ended on reports
the record's run over it.

**A record that could not be written is written when the status is next asked.** A replica whose
own last run is newer than the record (the bucket was out of reach when the run ended) writes it
on its next status call, so the other replicas report it from then on.

**The record is an aid to the status, never a condition of the run.** A record that cannot be
written (the bucket is out of reach, the result does not serialise) or read (absent, written in a
form this code does not know) is logged, and the replica reports its own outcome as it would
without one. So replicas on code that writes no record and replicas that do run side by side.

Usage::

    load = BackgroundOperation("enr-load", loader.load, shared=SharedOutcome(lease, "enr-load"))
    server.register(StartOperationTool(name="enr.reload", description="...",
                                       operation=lambda: load, status_tool="enr.load_status"))
    server.register(OperationStatusTool(name="enr.load_status", description="...",
                                        operation=lambda: load))
    load.start_when_needed(loader.needs_load, retry_on=(DataLayerUnavailableError,), retry_seconds=30)
    ...
    await load.stop()  # on shutdown

The tools take the operation through a callable, so a pod can register them before the
operation exists (its storage is reached only after it connects); until then both refuse
the call, saying so.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Generic, Literal, TypeVar

from threetears.core.serialization import serialize_to_json
from threetears.observe import get_logger

from threetears.agent.tools.base_tool import CONFLICT, TOOL_NOT_READY, MCPToolDefinition, TearsTool, ToolResult

if TYPE_CHECKING:
    from threetears.core.coordination.lease import KVLease

__all__ = [
    "BackgroundOperation",
    "OperationOutcome",
    "OperationState",
    "OperationStatusTool",
    "RecordedOutcome",
    "RequestOperationTool",
    "SharedOutcome",
    "StartOperationTool",
]

log = get_logger(__name__)

ResultT = TypeVar("ResultT")

#: where an operation stands: never run since the process started, running, or how the last run ended
OperationState = Literal["idle", "running", "succeeded", "failed"]

#: the version both tools carry unless the pod names another
_DEFAULT_TOOL_VERSION: Final = "1.0"

#: what the shared outcome's key adds to the operation's lease key
_OUTCOME_SUFFIX: Final = ".outcome"

#: how many times a write of the shared outcome is tried again after another replica's write beat it
_DEFAULT_PUBLISH_ATTEMPTS: Final = 8

#: what both tools answer before the pod has built the operation
_NOT_READY: Final = "the pod is not ready to run this yet (its storage is still being reached); try again shortly"


@dataclass(frozen=True)
class OperationOutcome(Generic[ResultT]):
    """how one run ended.

    :ivar started_at: when the run began
    :ivar finished_at: when it ended
    :ivar result: what the run returned; None when it failed
    :ivar error: why it failed, as ``<ExceptionType>: <message>``; None when it succeeded
    """

    started_at: datetime
    finished_at: datetime
    result: ResultT | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """whether the run returned rather than raised.

        :return: True for a run that returned
        :rtype: bool
        """
        return self.error is None


@dataclass(frozen=True)
class RecordedOutcome:
    """a finished run as the shared record holds it.

    :ivar outcome: how the run ended; its ``result`` is the JSON form that was stored
        (:class:`SharedOutcome`), not what the run returned
    :ivar replica: the replica that ran it (its lease factory's ``pod_id``)
    :ivar own: whether that replica is the one reading
    """

    outcome: OperationOutcome[Any]
    replica: str
    own: bool


class SharedOutcome:
    """the newest finished run of one operation, kept beside its lease for every replica to report.

    One KV entry in the lease's bucket, under the lease's owner scope (``lease.stored_key``), so a pod
    granted its own keys of a shared bucket needs nothing more. :meth:`publish` writes a run unless
    the entry already holds one that finished later; the write is a compare-and-swap on the entry's
    revision, so of two replicas finishing together the later finish stands whichever writes first.
    Nothing expires the entry: it is replaced only by a run that finished later.

    **How a result is carried.** ``OperationOutcome.result`` is whatever the run returned, which a KV
    entry cannot hold. What is stored is ``render_result(result)`` (the result itself when no
    ``render_result`` is given), encoded as JSON by :func:`~threetears.core.serialization.serialize_to_json`,
    which also writes a datetime, UUID, Decimal, bytes or Enum as its stored string. So a result
    read back is that JSON form, already rendered: give this the ``render_result`` the status tool
    is given, and the tool reports a run the same way whichever replica ran it. A result that does
    not encode makes :meth:`publish` raise, and the operation, which logs it, keeps its own outcome.

    **Which runs are recorded.** Every failed run, and every succeeded run whose result ``ran`` says
    did the work (every one, without ``ran``). A succeeded run that did nothing (another replica
    held the lease, so this one returned at once) is not the operation's last run: :meth:`publish`
    writes nothing for it.

    Finish times are each replica's own clock, so replicas whose clocks disagree by more than the
    time between two runs can order those runs wrongly.

    :param lease: the operation's lease factory, whose bucket, owner scope and ``pod_id`` this shares
    :ptype lease: KVLease
    :param key: the operation's lease key (the outcome's key is this plus ``.outcome``)
    :ptype key: str
    :param render_result: turns a result into JSON values to store; the result as it is when None
    :ptype render_result: Callable[[Any], Any] | None
    :param ran: whether a succeeded run's result (as the run returned it) is of a run that did the
        work, and so one to record; every result is when None. Not asked of a failed run
    :ptype ran: Callable[[Any], bool] | None
    :param attempts: how many compare-and-swap rounds a write is allowed
    :ptype attempts: int
    """

    def __init__(
        self,
        lease: KVLease,
        key: str,
        *,
        render_result: Callable[[Any], Any] | None = None,
        ran: Callable[[Any], bool] | None = None,
        attempts: int = _DEFAULT_PUBLISH_ATTEMPTS,
    ) -> None:
        self._lease = lease
        self._key = lease.stored_key(f"{key}{_OUTCOME_SUFFIX}")
        self._render_result = render_result
        self._ran = ran
        self._attempts = attempts

    @property
    def replica(self) -> str:
        """the name this replica's runs are recorded under: its lease factory's ``pod_id``.

        :return: the replica's name
        :rtype: str
        """
        return self._lease.pod_id

    def records(self, outcome: OperationOutcome[Any]) -> bool:
        """whether a finished run is one the record holds: it failed, or it did the work.

        :param outcome: how the run ended
        :ptype outcome: OperationOutcome[Any]
        :return: False for a succeeded run whose result ``ran`` says did nothing; True otherwise
        :rtype: bool
        """
        return not outcome.succeeded or self._ran is None or bool(self._ran(outcome.result))

    async def publish(self, outcome: OperationOutcome[Any]) -> bool:
        """record a finished run, unless it did nothing or the record holds one that finished later.

        :param outcome: how the run ended
        :ptype outcome: OperationOutcome[Any]
        :return: True when the record now holds this run; False when the run is not one to record
            (:meth:`records`), a later run stands, or every round lost to another replica's write
        :rtype: bool
        :raises TypeError: when the result does not encode as JSON
        :raises ValueError: when the result does not encode as JSON, or a time is naive
        """
        written = False
        if self.records(outcome):
            written = await self._write(outcome)
        return written

    async def _write(self, outcome: OperationOutcome[Any]) -> bool:
        """write a run over the entry by compare-and-swap, unless the entry's run finished later.

        :param outcome: how the run ended
        :ptype outcome: OperationOutcome[Any]
        :return: True when the entry now holds this run
        :rtype: bool
        :raises TypeError: when the result does not encode as JSON
        :raises ValueError: when the result does not encode as JSON, or a time is naive
        """
        value = self._encoded(outcome)
        bucket = await self._lease.bucket()
        written = False
        newer = False
        attempts = 0
        while not written and not newer and attempts < self._attempts:
            attempts += 1
            entry = await bucket.get_entry(key=self._key)
            stored = None if entry is None else self._decoded(entry[0])
            if stored is not None and stored.outcome.finished_at > outcome.finished_at:
                newer = True
            elif entry is None:
                written = await bucket.create(key=self._key, value=value) is not None
            else:
                # only over the revision read: a run recorded since is compared before it is replaced
                written = await bucket.update(key=self._key, value=value, revision=entry[1]) is not None
        if not written and not newer:
            log.warning(
                "the shared outcome was not written: every attempt lost to another replica's write",
                extra={"extra_data": {"key": self._key, "attempts": attempts}},
            )
        return written

    async def read(self) -> RecordedOutcome | None:
        """the newest finished run recorded, by any replica.

        :return: the run; None when none is recorded, or the entry is not in a form this reads
        :rtype: RecordedOutcome | None
        """
        bucket = await self._lease.bucket()
        value = await bucket.get(key=self._key)
        return None if value is None else self._decoded(value)

    def _encoded(self, outcome: OperationOutcome[Any]) -> bytes:
        """the entry's value for a run.

        :param outcome: how the run ended
        :ptype outcome: OperationOutcome[Any]
        :return: the JSON bytes
        :rtype: bytes
        :raises TypeError: when the result does not encode as JSON
        :raises ValueError: when the result does not encode as JSON, or a time is naive
        """
        result: Any = outcome.result
        if result is not None and self._render_result is not None:
            result = self._render_result(result)
        return serialize_to_json(
            {
                "replica": self.replica,
                "started_at": outcome.started_at,
                "finished_at": outcome.finished_at,
                "result": result,
                "error": outcome.error,
            }
        )

    def _decoded(self, value: bytes) -> RecordedOutcome | None:
        """the run an entry's value holds.

        :param value: the entry's value
        :ptype value: bytes
        :return: the run; None when the value is not in a form this reads (logged)
        :rtype: RecordedOutcome | None
        """
        recorded: RecordedOutcome | None = None
        try:
            body = json.loads(value)
            replica = body["replica"]
            error = body["error"]
            started_at = datetime.fromisoformat(body["started_at"])
            finished_at = datetime.fromisoformat(body["finished_at"])
            if not isinstance(replica, str) or not (error is None or isinstance(error, str)):
                raise TypeError("replica must be a string, and error a string or null")
            if started_at.tzinfo is None or finished_at.tzinfo is None:
                raise ValueError("a time without a zone names no instant to compare")
            recorded = RecordedOutcome(
                outcome=OperationOutcome(
                    started_at=started_at, finished_at=finished_at, result=body.get("result"), error=error
                ),
                replica=replica,
                own=replica == self.replica,
            )
        except (ValueError, TypeError, KeyError) as exc:
            # another version of this code may write another form; the reader carries on without it
            log.warning(
                "the shared outcome is not in a form this reads; it is ignored",
                extra={"extra_data": {"key": self._key, "error": f"{type(exc).__name__}: {exc}"}},
            )
        return recorded


class BackgroundOperation(Generic[ResultT]):
    """one operation, run in the background at most once at a time, its last outcome kept.

    :param name: the operation's name, for task names and logs
    :ptype name: str
    :param run: the operation's body; each start awaits a fresh call of it
    :ptype run: Callable[[], Awaitable[ResultT]]
    :param request: records that a run is wanted (``CoalescedRun.request``), for an operation whose
        body drains requests; :meth:`request` needs it
    :ptype request: Callable[[], Awaitable[None]] | None
    :param shared: where each finished run is also recorded for the other replicas, and read from by
        :meth:`status_across_replicas`; None keeps the outcome in this process only
    :ptype shared: SharedOutcome | None
    """

    def __init__(
        self,
        name: str,
        run: Callable[[], Awaitable[ResultT]],
        *,
        request: Callable[[], Awaitable[None]] | None = None,
        shared: SharedOutcome | None = None,
    ) -> None:
        self._name = name
        self._run = run
        self._request = request
        self._shared = shared
        self._again = False
        self._task: asyncio.Task[None] | None = None
        self._watcher: asyncio.Task[None] | None = None
        self._started_at: datetime | None = None
        self._last: OperationOutcome[ResultT] | None = None
        # whether the last outcome is a run's and one the shared record holds; not a wait that failed
        # (no run), nor a run that did nothing
        self._last_recordable = False

    @property
    def name(self) -> str:
        """the operation's name.

        :return: the name
        :rtype: str
        """
        return self._name

    @property
    def running(self) -> bool:
        """whether a run is in progress.

        :return: True while one runs
        :rtype: bool
        """
        return self._task is not None and not self._task.done()

    @property
    def started_at(self) -> datetime | None:
        """when the run in progress began; None when none runs.

        :return: the start time
        :rtype: datetime | None
        """
        return self._started_at if self.running else None

    @property
    def last(self) -> OperationOutcome[ResultT] | None:
        """how the last finished run ended; None before one has finished in this process.

        :return: the outcome
        :rtype: OperationOutcome[ResultT] | None
        """
        return self._last

    @property
    def state(self) -> OperationState:
        """running, or how the last run ended, or idle before any has.

        :return: the state
        :rtype: OperationState
        """
        state: OperationState
        if self.running:
            state = "running"
        elif self._last is None:
            state = "idle"
        elif self._last.succeeded:
            state = "succeeded"
        else:
            state = "failed"
        return state

    def start(self) -> bool:
        """begin a run in the background, unless one is running.

        :return: True when a run was started; False when one was already running
        :rtype: bool
        """
        started = not self.running
        if started:
            # a run that starts now sees every request already recorded
            self._again = False
            self._started_at = datetime.now(UTC)
            self._task = asyncio.create_task(self._run_and_keep_outcome(), name=self._name)
            log.info("background operation started", extra={"extra_data": {"operation": self._name}})
        return started

    async def request(self) -> bool:
        """record that a run is wanted, then start one, or have the run in progress run once more.

        The request is recorded first, so a run on another replica may take it too; a run in progress
        here is followed by one more, since it may already have made its last look for requests. Its
        end checks for that in the same step it ends in, so no request falls between.

        :return: True when a run was started; False when the run in progress will run once more
        :rtype: bool
        :raises TypeError: when the operation was built without ``request=``
        """
        if self._request is None:
            raise TypeError(f"{self._name}: built without request=, so it cannot take a request")
        await self._request()
        started = self.start()
        if not started:
            self._again = True
        return started

    async def wait(self) -> None:
        """wait for the run in progress, if any, to end.

        :return: nothing
        :rtype: None
        """
        if self._task is not None:
            await self._task

    def start_when_needed(
        self,
        needed: Callable[[], Awaitable[bool]],
        *,
        retry_on: tuple[type[BaseException], ...],
        retry_seconds: float,
    ) -> bool:
        """in the background: ask ``needed`` until it answers, and start a run if it says yes.

        For a pod's first run, when it cannot know yet whether one is needed (its tables are
        out of reach until the hub grants it storage). Only the errors in ``retry_on`` are
        retried; any other ends the wait, is kept as the last outcome (the status reports it failed, with
        its reason) and is raised from :meth:`wait_until_settled`. If a
        run was started some other way meanwhile, that run is the one; none is added. While
        one such wait is still deciding, another is refused, so :meth:`stop` and
        :meth:`wait_until_settled` always reach the wait that is running.

        :param needed: whether a run is needed now
        :ptype needed: Callable[[], Awaitable[bool]]
        :param retry_on: the errors that mean "not reachable yet", retried after ``retry_seconds``
        :ptype retry_on: tuple[type[BaseException], ...]
        :param retry_seconds: the wait between attempts
        :ptype retry_seconds: float
        :return: True when the wait started; False when one was already deciding
        :rtype: bool
        """
        started = self._watcher is None or self._watcher.done()
        if started:
            self._watcher = asyncio.create_task(
                self._start_when_needed(needed, retry_on=retry_on, retry_seconds=retry_seconds),
                name=f"{self._name}-when-needed",
            )
        else:
            log.warning(
                "a wait to start the operation is already deciding; this one is refused",
                extra={"extra_data": {"operation": self._name}},
            )
        return started

    async def wait_until_settled(self) -> None:
        """wait for :meth:`start_when_needed` to decide, and for any run it started to end.

        :return: nothing
        :rtype: None
        :raises Exception: what ``needed`` raised that ``retry_on`` did not name
        """
        if self._watcher is not None:
            await self._watcher
        await self.wait()

    async def stop(self) -> None:
        """cancel the wait of :meth:`start_when_needed` and the run in progress, if any.

        :return: nothing
        :rtype: None
        """
        for task in (self._watcher, self._task):
            if task is not None and not task.done():
                task.cancel()
                # NOSILENT: the cancellation requested on the line above, awaited so the task ends first
                with suppress(asyncio.CancelledError):
                    await task

    def status(self, render_result: Callable[[ResultT], Any] | None = None) -> dict[str, Any]:
        """where the operation stands on this replica, as plain JSON values.

        :param render_result: turns a result into JSON values; the result as it is when None
        :ptype render_result: Callable[[ResultT], Any] | None
        :return: ``state``, ``running``, ``started_at`` (of the run in progress) and ``last``
            (``started_at``, ``finished_at``, ``result``, ``error``), times in ISO 8601
        :rtype: dict[str, Any]
        """
        return self._status_of(self._rendered_last(render_result))

    async def status_across_replicas(self, render_result: Callable[[ResultT], Any] | None = None) -> dict[str, Any]:
        """where the operation stands, its last run the newest any replica finished.

        Without ``shared=`` this is :meth:`status`. With it, ``last`` is the newer of this replica's
        own last run and the shared record's, and names its replica; a record that is absent, cannot
        be read, or is older than this replica's own run leaves this replica's own, and that run is
        then written to the record (it could not be when it ended), so the other replicas report it
        too. A run of this replica's that did nothing (``SharedOutcome(ran=)``) is reported only
        while there is no record. ``running`` and
        the top-level ``started_at`` are this replica's: a run in progress on another replica is not
        known here.

        :param render_result: turns this replica's own result into JSON values; a result from the
            record is reported as stored, already rendered by the replica that wrote it
        :ptype render_result: Callable[[ResultT], Any] | None
        :return: as :meth:`status`; with ``shared=``, ``last`` also carries ``replica`` (who ran it)
            and ``other_replica`` (whether that is not this replica)
        :rtype: dict[str, Any]
        """
        rendered = self._rendered_last(render_result)
        if self._shared is not None:
            if rendered is not None:
                rendered = {**rendered, "replica": self._shared.replica, "other_replica": False}
            recorded = await self._recorded(self._shared)
            own = self._last
            if (
                own is not None
                and self._last_recordable
                and (recorded is None or recorded.outcome.finished_at < own.finished_at)
            ):
                # the record lacks this replica's newer run: its write when the run ended did not land
                await self._publish(self._shared, own)
            if recorded is not None and self._reported_over_own(recorded):
                rendered = {
                    "started_at": recorded.outcome.started_at.isoformat(),
                    "finished_at": recorded.outcome.finished_at.isoformat(),
                    "result": recorded.outcome.result,
                    "error": recorded.outcome.error,
                    "replica": recorded.replica,
                    "other_replica": not recorded.own,
                }
        return self._status_of(rendered)

    def _rendered_last(self, render_result: Callable[[ResultT], Any] | None) -> dict[str, Any] | None:
        """this replica's own last run, as plain JSON values.

        :param render_result: turns a result into JSON values; the result as it is when None
        :ptype render_result: Callable[[ResultT], Any] | None
        :return: ``started_at``, ``finished_at``, ``result`` and ``error``; None before a run has ended here
        :rtype: dict[str, Any] | None
        """
        last = self._last
        rendered: dict[str, Any] | None = None
        if last is not None:
            result: Any = last.result
            if result is not None and render_result is not None:
                result = render_result(result)
            rendered = {
                "started_at": last.started_at.isoformat(),
                "finished_at": last.finished_at.isoformat(),
                "result": result,
                "error": last.error,
            }
        return rendered

    def _status_of(self, last: dict[str, Any] | None) -> dict[str, Any]:
        """the status that reports ``last`` as the last run.

        :param last: the last run as plain JSON values, or None when none is known
        :ptype last: dict[str, Any] | None
        :return: ``state``, ``running``, ``started_at`` (of the run in progress) and ``last``
        :rtype: dict[str, Any]
        """
        started = self.started_at
        state: OperationState
        if self.running:
            state = "running"
        elif last is None:
            state = "idle"
        elif last["error"] is None:
            state = "succeeded"
        else:
            state = "failed"
        return {
            "state": state,
            "running": self.running,
            "started_at": None if started is None else started.isoformat(),
            "last": last,
        }

    def _reported_over_own(self, recorded: RecordedOutcome) -> bool:
        """whether the recorded run, not this replica's own last, is the newest known here.

        :param recorded: the shared record's run
        :ptype recorded: RecordedOutcome
        :return: True when no run has ended here, the last to end here did nothing, the recorded one
            finished later, or another replica's finished at the same instant (its write is the one
            the record kept)
        :rtype: bool
        """
        own = self._last
        return (
            own is None
            # a run that did nothing ended, but is not the operation's last run while one is recorded
            or (own.succeeded and not self._last_recordable)
            or recorded.outcome.finished_at > own.finished_at
            or (recorded.outcome.finished_at == own.finished_at and not recorded.own)
        )

    async def _recorded(self, shared: SharedOutcome) -> RecordedOutcome | None:
        """the shared record's run, or None when there is none or it cannot be reached.

        :param shared: the shared record
        :ptype shared: SharedOutcome
        :return: the run, or None
        :rtype: RecordedOutcome | None
        """
        recorded: RecordedOutcome | None = None
        try:
            recorded = await shared.read()
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- the status must answer from this replica's own outcome whatever the bucket's client raises; logged
            log.warning(
                "the shared outcome could not be read; this replica reports its own last run",
                extra={"extra_data": {"operation": self._name, "error": f"{type(exc).__name__}: {exc}"}},
            )
        return recorded

    async def _share(self, outcome: OperationOutcome[ResultT]) -> None:
        """record a run that just ended for the other replicas, when the operation shares its outcome.

        :param outcome: how the run ended
        :ptype outcome: OperationOutcome[ResultT]
        :return: nothing
        :rtype: None
        """
        # a run counts as one to record until its record says otherwise, so a ``ran`` that raises
        # leaves the status comparing this run by its finish time, as it does without a ``ran``
        self._last_recordable = True
        if self._shared is not None:
            try:
                self._last_recordable = self._shared.records(outcome)
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- the run has ended and its outcome is kept here; the caller's ran= raising must not make it a failed run; logged
                log.warning(
                    "cannot tell whether the run is one to record; it is taken for one",
                    extra={"extra_data": {"operation": self._name, "error": f"{type(exc).__name__}: {exc}"}},
                )
            await self._publish(self._shared, outcome)

    async def _publish(self, shared: SharedOutcome, outcome: OperationOutcome[ResultT]) -> None:
        """write a run to the shared record; a write that fails is logged and the run stays this replica's.

        :param shared: the shared record
        :ptype shared: SharedOutcome
        :param outcome: how the run ended
        :ptype outcome: OperationOutcome[ResultT]
        :return: nothing
        :rtype: None
        """
        try:
            await shared.publish(outcome)
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- the run has ended and its outcome is kept here; a record that cannot be written (the bucket's client, the caller's render_result or ran, a result that does not encode) must not make it a failed run nor fail the status; logged
            log.warning(
                "the outcome could not be shared; only this replica reports this run",
                extra={"extra_data": {"operation": self._name, "error": f"{type(exc).__name__}: {exc}"}},
            )

    async def _start_when_needed(
        self,
        needed: Callable[[], Awaitable[bool]],
        *,
        retry_on: tuple[type[BaseException], ...],
        retry_seconds: float,
    ) -> None:
        """the body of :meth:`start_when_needed`.

        :param needed: whether a run is needed now
        :ptype needed: Callable[[], Awaitable[bool]]
        :param retry_on: the errors retried
        :ptype retry_on: tuple[type[BaseException], ...]
        :param retry_seconds: the wait between attempts
        :ptype retry_seconds: float
        :return: nothing
        :rtype: None
        """
        answer: bool | None = None
        waiting_since = datetime.now(UTC)
        while answer is None:
            try:
                answer = await needed()
            except retry_on as exc:
                log.warning(
                    "cannot tell yet whether the operation is needed; retrying",
                    extra={
                        "extra_data": {
                            "operation": self._name,
                            "error": f"{type(exc).__name__}: {exc}",
                            "retry_seconds": retry_seconds,
                        }
                    },
                )
                await asyncio.sleep(retry_seconds)
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- kept, logged and re-raised: the wait may never be awaited, so the status and the log are where this error is sure to be seen
                error = f"{type(exc).__name__}: {exc}"
                # kept as the last outcome, so the status says failed with the reason rather than idle;
                # unless a run ended after the wait began, which is newer news than this failure
                if self._last is None or self._last.finished_at < waiting_since:
                    self._last = OperationOutcome(started_at=waiting_since, finished_at=datetime.now(UTC), error=error)
                    # this replica's failure to decide, not a run: reported here, never recorded
                    self._last_recordable = False
                log.error(
                    "cannot tell whether the operation is needed; no run will start",
                    extra={"extra_data": {"operation": self._name, "error": error}},
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                raise
        if answer:
            self.start()
        else:
            log.info("background operation not needed", extra={"extra_data": {"operation": self._name}})

    async def _run_and_keep_outcome(self) -> None:
        """run, once more for each request made meanwhile, and keep how the last run ended.

        :return: nothing
        :rtype: None
        """
        again = True
        while again:
            await self._run_once()
            # read and cleared in the step the task ends in: a request after this finds the task done
            # and starts a run of its own
            again, self._again = self._again, False
            if again:
                self._started_at = datetime.now(UTC)
                log.info(
                    "background operation runs once more for a request", extra={"extra_data": {"operation": self._name}}
                )

    async def _run_once(self) -> None:
        """run once and keep how it ended, a run that raised included.

        :return: nothing
        :rtype: None
        """
        started = self._started_at or datetime.now(UTC)
        outcome: OperationOutcome[ResultT]
        try:
            result = await self._run()
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- a background run has no caller to raise to; the failure is logged and kept for the status tool
            error = f"{type(exc).__name__}: {exc}"
            outcome = OperationOutcome(started_at=started, finished_at=datetime.now(UTC), error=error)
            log.error(
                "background operation failed",
                extra={"extra_data": {"operation": self._name, "error": error}},
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        else:
            outcome = OperationOutcome(started_at=started, finished_at=datetime.now(UTC), result=result)
            log.info("background operation finished", extra={"extra_data": {"operation": self._name}})
        # kept here before it is shared, so this replica's status has it whatever becomes of the record
        self._last = outcome
        await self._share(outcome)


class _OperationTool(TearsTool):
    """what both tools share: a name, a version, a description, and the operation once built.

    :param name: the tool's namespaced name (``<provider>.<verb>``)
    :ptype name: str
    :param description: the sentence a caller routes on
    :ptype description: str
    :param operation: the operation, or None until the pod has built it
    :ptype operation: Callable[[], BackgroundOperation[Any] | None]
    :param version: the tool's version
    :ptype version: str
    """

    # an operator's action as much as an agent's: reachable over the tools API too. the face only
    # makes the call addressable; the hub still requires the caller's grant before it runs
    face_api = True

    def __init__(
        self,
        *,
        name: str,
        description: str,
        operation: Callable[[], BackgroundOperation[Any] | None],
        version: str = _DEFAULT_TOOL_VERSION,
    ) -> None:
        super().__init__()
        self._name = name
        self._description = description
        self._operation = operation
        self._version = version

    async def execute(self, **kwargs: Any) -> ToolResult:
        """refuse until the operation exists, then answer for it.

        :param kwargs: none are taken
        :ptype kwargs: Any
        :return: the tool's answer
        :rtype: ToolResult
        """
        operation = self._operation()
        result: ToolResult
        if operation is None:
            result = ToolResult(success=False, content="", error=_NOT_READY, error_code=TOOL_NOT_READY)
        else:
            result = self.answer(operation)
        return result

    def answer(self, operation: BackgroundOperation[Any]) -> ToolResult:
        """the tool's answer for a built operation.

        :param operation: the operation
        :ptype operation: BackgroundOperation[Any]
        :return: the answer
        :rtype: ToolResult
        :raises NotImplementedError: in this base
        """
        raise NotImplementedError

    def mcp_schema(self) -> MCPToolDefinition:
        """the tool's definition; it takes no arguments.

        :return: the definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self._name,
            version=self._version,
            description=self._description,
            input_schema={"type": "object", "properties": {}},
        )

    def mcp_name(self) -> str:
        """the tool's namespaced name.

        :return: the name given
        :rtype: str
        """
        return self._name

    def mcp_version(self) -> str:
        """the tool's version.

        :return: the version given
        :rtype: str
        """
        return self._version


class StartOperationTool(_OperationTool):
    """starts the operation and answers at once; a start while one runs is a ``CONFLICT``.

    :param name: the tool's namespaced name (``<provider>.<verb>``)
    :ptype name: str
    :param description: the sentence a caller routes on
    :ptype description: str
    :param operation: the operation, or None until the pod has built it
    :ptype operation: Callable[[], BackgroundOperation[Any] | None]
    :param status_tool: the status tool's name, which the answer points the caller to
    :ptype status_tool: str
    :param version: the tool's version
    :ptype version: str
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        operation: Callable[[], BackgroundOperation[Any] | None],
        status_tool: str,
        version: str = _DEFAULT_TOOL_VERSION,
    ) -> None:
        super().__init__(name=name, description=description, operation=operation, version=version)
        self._status_tool = status_tool

    def answer(self, operation: BackgroundOperation[Any]) -> ToolResult:
        """start a run, or refuse because one is running.

        :param operation: the operation
        :ptype operation: BackgroundOperation[Any]
        :return: that it started, or a ``CONFLICT`` naming the run in progress
        :rtype: ToolResult
        """
        result: ToolResult
        if operation.start():
            result = ToolResult(
                success=True,
                content=f"started; {self._status_tool} says when it ends and how",
                metadata={"started": True},
            )
        else:
            started = operation.started_at
            since = "" if started is None else f" since {started.isoformat()}"
            result = ToolResult(
                success=False,
                content="",
                error=f"a run is already in progress{since}; {self._status_tool} says when it ends",
                error_code=CONFLICT,
            )
        return result


class RequestOperationTool(_OperationTool):
    """asks the operation to run (:meth:`BackgroundOperation.request`); a request while one runs is not refused.

    For an operation that drains requests, built with ``request=``, so the request and the run are
    one object's and cannot name different runs. Where :class:`StartOperationTool` answers a second
    start ``CONFLICT``, this answers that the run in progress is followed by one that takes it.

    :param name: the tool's namespaced name (``<provider>.<verb>``)
    :ptype name: str
    :param description: the sentence a caller routes on
    :ptype description: str
    :param operation: the operation, or None until the pod has built it
    :ptype operation: Callable[[], BackgroundOperation[Any] | None]
    :param status_tool: the status tool's name, which the answer points the caller to
    :ptype status_tool: str
    :param version: the tool's version
    :ptype version: str
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        operation: Callable[[], BackgroundOperation[Any] | None],
        status_tool: str,
        version: str = _DEFAULT_TOOL_VERSION,
    ) -> None:
        super().__init__(name=name, description=description, operation=operation, version=version)
        self._status_tool = status_tool

    async def execute(self, **kwargs: Any) -> ToolResult:
        """refuse until the operation exists; then ask it to run.

        :param kwargs: none are taken
        :ptype kwargs: Any
        :return: that it started, or that the run in progress is followed by one that takes the request
        :rtype: ToolResult
        """
        operation = self._operation()
        result: ToolResult
        if operation is None:
            result = ToolResult(success=False, content="", error=_NOT_READY, error_code=TOOL_NOT_READY)
        else:
            started = await operation.request()
            content = (
                f"started; {self._status_tool} says when it ends and how"
                if started
                else f"requested; it runs after the run in progress, and {self._status_tool} says when it ends"
            )
            result = ToolResult(success=True, content=content, metadata={"started": started})
        return result


class OperationStatusTool(_OperationTool):
    """says whether the operation is running and how the last run ended.

    For an operation that shares its outcome (``BackgroundOperation(shared=)``) the last run is the
    newest any replica finished, and the answer says when it was another replica's.

    :param name: the tool's namespaced name (``<provider>.<noun>``)
    :ptype name: str
    :param description: the sentence a caller routes on
    :ptype description: str
    :param operation: the operation, or None until the pod has built it
    :ptype operation: Callable[[], BackgroundOperation[Any] | None]
    :param render_result: turns a result into JSON values for the answer; the result as it is when
        None. Give an operation's :class:`SharedOutcome` the same one, so another replica's run reads alike
    :ptype render_result: Callable[[Any], Any] | None
    :param progress: what the operation is doing now, as JSON values with a short ``summary`` line
        (phase, counts, timings): answered as ``progress`` whether or not a run is in progress, so a
        caller sees a long run move and what the operation's other work is doing between runs
    :ptype progress: Callable[[], dict[str, Any]] | None
    :param version: the tool's version
    :ptype version: str
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        operation: Callable[[], BackgroundOperation[Any] | None],
        render_result: Callable[[Any], Any] | None = None,
        progress: Callable[[], dict[str, Any]] | None = None,
        version: str = _DEFAULT_TOOL_VERSION,
    ) -> None:
        super().__init__(name=name, description=description, operation=operation, version=version)
        self._render_result = render_result
        self._progress = progress

    async def execute(self, **kwargs: Any) -> ToolResult:
        """refuse until the operation exists; then report it, its last run the newest of any replica's.

        :param kwargs: none are taken
        :ptype kwargs: Any
        :return: the status (:meth:`BackgroundOperation.status_across_replicas`)
        :rtype: ToolResult
        """
        operation = self._operation()
        result: ToolResult
        if operation is None:
            result = ToolResult(success=False, content="", error=_NOT_READY, error_code=TOOL_NOT_READY)
        else:
            result = self._answer_with(operation, await operation.status_across_replicas(self._render_result))
        return result

    def answer(self, operation: BackgroundOperation[Any]) -> ToolResult:
        """the operation's status on this replica, as a sentence and as metadata.

        :param operation: the operation
        :ptype operation: BackgroundOperation[Any]
        :return: the status (:meth:`BackgroundOperation.status`)
        :rtype: ToolResult
        """
        return self._answer_with(operation, operation.status(self._render_result))

    def _answer_with(self, operation: BackgroundOperation[Any], status: dict[str, Any]) -> ToolResult:
        """a status as a sentence and as metadata, with the operation's progress.

        :param operation: the operation
        :ptype operation: BackgroundOperation[Any]
        :param status: the operation's status
        :ptype status: dict[str, Any]
        :return: the answer
        :rtype: ToolResult
        """
        last = status["last"]
        sentence = status["state"]
        if status["running"]:
            sentence = f"running since {status['started_at']}"
        if self._progress is not None:
            try:
                progress = self._progress()
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- the status (the last run's error above all) must answer even when what reports progress fails; logged and said
                log.exception("operation %s: its progress could not be read", operation.name)
                progress = {"summary": f"progress unavailable ({type(exc).__name__})", "error": str(exc)}
            status["progress"] = progress
            if progress.get("summary"):
                sentence += f"; now: {progress['summary']}"
        if last is not None:
            ended = "failed" if last["error"] is not None else "succeeded"
            detail = last["error"] if last["error"] is not None else last["result"]
            where = f", on another replica ({last['replica']})," if last.get("other_replica") else ""
            sentence += f"; the last run{where} {ended} at {last['finished_at']}: {detail}"
        return ToolResult(success=True, content=sentence, metadata=status)
