"""Job leases: which process runs a job is written on the job's record, so every process answers it alike.

A job's work — a run, a sweep, an analysis generation — runs in one process, while its record lives in a store
several processes share. Liveness used to be answered from the job manager's memory, so a poll in a process that
did not start a job read a live one as lost, and a cancel there was refused. Here it is answered from the record:

- **The holder keeps a lease.** A job's record is written with a lease (:class:`~threetears.evals.schema.JobLease`:
  the holding process's id and an expiry) before its work starts, and :class:`JobLeases` — one per job manager,
  so one per process — renews it with a conditional write every third of its duration while the work runs.
- **A job is lost only when its record has not ended and its lease has lapsed** (or it carries none: a record
  written before leases). Every process answers that the same, and the first reader to see it WRITES it
  (:func:`settle_job_record`): the record takes its own ending and :class:`~threetears.evals.schema.JobLoss`, with
  ``if_match``, so a race between two readers, or a reader and a renewal, is lost by one of them, which re-reads.
  Once written, the answer is the record's, never recomputed. There is no rule about the order a reader asks
  things in any more: there is one thing to read.
- **A holder that has lost its lease writes nothing more.** Its renewal finding the record ended by someone else,
  or its own lease lapsed (which it writes lost itself, as any reader would), stops the work at once; a write the
  work makes of its own record goes through :func:`write_held_record`, which refuses a record already ended and
  raises :class:`~threetears.evals.kernel.JobFenced`. A run's own writes are refused the same way: its status
  write takes nothing over a record carrying ``lost``, its own ending included, and the process that lost it
  neither broadcasts that ending nor tells the run-end hook.
- **A cancel is a field on the record** (``cancel_requested_at``), written by whichever process was asked
  (:func:`request_job_cancel`); the holder reads it at its next renewal and ends the job ``cancelled``, so a cancel
  lands within one renewal interval from anywhere.
- **A graceful shutdown ends every job the process holds as lost** (:meth:`JobLeases.shut_down`), its cause the
  shutdown, releasing the lease, before the work is cancelled.

**Time** is the store's when it keeps a clock (:class:`~threetears.evals.schema.StoreClock`), else each process's
own (:meth:`~threetears.evals.kernel.EvalStorage.store_time`); processes sharing a store without a clock must keep
their clocks in step to well inside a lease.

**One piece for every job record.** A record type takes the four lease fields
(:mod:`~threetears.evals.schema.job_lease`) and a :class:`LeasedJobKind` saying when it has ended and how it reads
lost; everything here is generic over it. :data:`RUN_JOB`, :data:`SWEEP_JOB` and :data:`ANALYSIS_JOB` are the three
the engine runs.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, cast

from threetears.evals.kernel.campaign import EvalAnalysisAttempt, EvalSweep
from threetears.evals.kernel.errors import ConflictError, JobFenced
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.scoring import reconstruct_completeness
from threetears.evals.kernel.storage import JobRecordStore, LeaseStore
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.job_lease import JobLease, JobLoss, LeasedRecord
from threetears.evals.schema.models import TERMINAL_RUN_STATUSES, EvalRun
from threetears.observe import get_logger

log = get_logger(__name__)

#: How long a lease lasts unless renewed, in seconds: the engine's default. A host sets its own on the launching
#: host (``LaunchHost.job_lease_s``); a job is read lost this long after its process last renewed it.
DEFAULT_JOB_LEASE_S = 60.0

#: How many times a lease write (a renewal, a lost write, a holder's own write) re-reads and re-applies after losing
#: its conditional write to another writer, before it gives up for now. Small: each attempt is a fresh read, and a
#: record that refuses three in a row is contended past what a retry fixes — the next renewal tries again.
LEASE_WRITE_ATTEMPTS = 3

#: The lost cause a graceful shutdown writes on every job its process holds.
SHUTDOWN_CAUSE = "process shut down"


@dataclass(frozen=True, kw_only=True)
class LeasedJobKind[R: EvalDocumentModel]:
    """One type of job record, as the lease machinery reads and ends it.

    Attributes:
        name: The job kind's name, for a message.
        model: The record's model; it declares the four lease fields (:class:`~threetears.evals.schema.LeasedRecord`).
        ended: Whether the record says, by its own status, that the job has ended.
        end_lost: The record as it is written lost: its own ending set and ``loss`` recorded on it. Given the store
            because a run's ending reconstructs its completeness from its stored results.
    """

    name: str
    model: type[R]
    ended: Callable[[R], bool]
    end_lost: Callable[[R, JobLoss, LeaseStore], R]


def _run_lost(run: EvalRun, loss: JobLoss, store: LeaseStore) -> EvalRun:
    """A run written lost: ``cancelled`` — the status a run that stopped without finishing has always taken — its
    reason the loss, and its completeness reconstructed from the results it stored unless its loop recorded its own."""
    completeness = run.completeness or reconstruct_completeness(
        run, store.query_eval_results_by_run(run.id, run.scope_id)
    )
    return run.model_copy(
        update={
            "status": "cancelled",
            "cancellation_reason": f"lost: {loss.cause}",
            "completed_at": loss.at,
            "completeness": completeness,
            "lost": loss,
        }
    )


def _sweep_lost(sweep: EvalSweep, loss: JobLoss, _store: LeaseStore) -> EvalSweep:
    """A sweep written lost: ``cancelled``, no further arm launched, its detail the loss."""
    return sweep.model_copy(
        update={"outcome": "cancelled", "detail": f"lost: {loss.cause}", "ended_at": loss.at, "lost": loss}
    )


def _attempt_lost(attempt: EvalAnalysisAttempt, loss: JobLoss, _store: LeaseStore) -> EvalAnalysisAttempt:
    """An analysis attempt written lost: ``cancelled`` mid-flight, its error the loss."""
    return attempt.model_copy(
        update={"outcome": "cancelled", "error": f"lost: {loss.cause}", "created_at": loss.at, "lost": loss}
    )


#: A launched run's record.
RUN_JOB: LeasedJobKind[EvalRun] = LeasedJobKind(
    name="run", model=EvalRun, ended=lambda run: run.status in TERMINAL_RUN_STATUSES, end_lost=_run_lost
)

#: A sweep's record.
SWEEP_JOB: LeasedJobKind[EvalSweep] = LeasedJobKind(
    name="sweep", model=EvalSweep, ended=lambda sweep: sweep.outcome != "running", end_lost=_sweep_lost
)

#: An analysis generation's record: its attempt.
ANALYSIS_JOB: LeasedJobKind[EvalAnalysisAttempt] = LeasedJobKind(
    name="analysis",
    model=EvalAnalysisAttempt,
    ended=lambda attempt: attempt.outcome != "running",
    end_lost=_attempt_lost,
)


def _leased(record: EvalDocumentModel) -> LeasedRecord:
    """``record`` read through its lease fields — every :class:`LeasedJobKind`'s model declares them."""
    return cast(LeasedRecord, record)


def _document(record: EvalDocumentModel) -> dict[str, Any]:
    """A record as it is written: whole, never a run a listing left payload out of, which would delete it.

    Raises:
        ValueError: A run read by a listing that left payload paths out.
    """
    if isinstance(record, EvalRun) and record.elided_payload_paths:
        raise ValueError(
            f"eval run {record.id} was read by a listing that left {sorted(record.elided_payload_paths)} out; "
            "a job record is written only as read whole"
        )
    return record.to_dict()


def lease_lapsed(record: LeasedRecord, now: datetime) -> bool:
    """Whether a record's lease has lapsed at ``now`` — or it has none, which no process can be renewing.

    **No grace for a record with no lease.** One was written by a build before leases, which never renews one; a
    grace measured from when it started, or was last written, would still read a live job lost once the grace ran
    out, and would only hide that the two builds cannot share a store — the older one's own reclaim reads every
    job it is not running as abandoned, the newer one's included. So the rule is stated rather than guessed at:
    every process sharing a store runs a build that keeps leases before any job starts (stop every worker, then
    upgrade — ``MIGRATIONS.toml``), and a lease-less record that has not ended is lost.

    Args:
        record: A job record.
        now: The time to read it at, in the store's time when the store keeps one.

    Returns:
        ``True`` when the record carries no lease or its expiry is at or before ``now``.
    """
    return record.lease is None or record.lease.expires() <= now


def _lapse_cause(record: LeasedRecord) -> str:
    """Why a record whose lease lapsed is lost, in a poll's words."""
    if record.lease is None:
        return (
            "it carries no lease, so no process can be running it — it was started by a build that kept a job's "
            "liveness in its own process, which has since gone. Every process sharing a store must run a build "
            "that keeps leases before any job starts: stop every worker, then upgrade"
        )
    return (
        f"its lease, held by process {record.lease.holder}, lapsed at {record.lease.expires_at} without renewal — "
        "the process running it ended or stalled"
    )


def settle_job_record[R: EvalDocumentModel](
    store: LeaseStore, kind: LeasedJobKind[R], record_id: str, scope_id: str
) -> tuple[R | None, bool]:
    """Read a job record as every process reads it: written lost first, when it has not ended and its lease has lapsed.

    The first reader to find the lapse writes the record's lost ending with ``if_match``; a reader that loses that
    race — to another reader, or to the holder's renewal — re-reads and answers from what won. Idempotent: a record
    already ended, lost or otherwise, is returned as it is.

    Args:
        store: Where the record lives.
        kind: Its type.
        record_id: Its id.
        scope_id: The scope it is read in; a record in another scope reads absent.

    Returns:
        ``(record, wrote_lost)``: the record as it now stands, or None when there is none; and whether THIS call
        wrote it lost.

    Raises:
        ConflictError: Every attempt to write the lapse lost its race; nothing was decided, read it again.
        StorageError: A read or write failed.
    """
    for _ in range(LEASE_WRITE_ATTEMPTS):
        record, etag = store.load_job_record(kind.model, record_id, scope_id)
        if record is None:
            return None, False
        if kind.ended(record):
            return record, False
        now = store.store_time()
        if not lease_lapsed(_leased(record), now):
            return record, False
        lost = kind.end_lost(record, JobLoss(at=now.isoformat(), cause=_lapse_cause(_leased(record))), store)
        try:
            store.save_job_record(_document(lost), if_match=etag)
        except ConflictError:
            # NOSILENT: a lost race re-reads; whoever won (a renewal, another reader) decides what this answers
            continue
        log.warning(
            "eval.job_lease %s=%s scope=%s written lost: %s",
            kind.name,
            record_id,
            scope_id,
            _lapse_cause(_leased(record)),
        )
        return lost, True
    raise ConflictError(
        f"{kind.name} job '{record_id}' kept changing while its lapsed lease was being written lost",
        remedy="read it again with job_poll",
    )


def require_held[R: EvalDocumentModel](
    store: LeaseStore, kind: LeasedJobKind[R], record_id: str, scope_id: str, *, leases: JobLeases | None = None
) -> None:
    """Refuse to go on with a job whose record has ended under it: the check a job makes before it spends again.

    A job's own writes are fenced (:func:`write_held_record`), but a job that spends between two writes — a sweep
    launching its next arm — would pay for work no write of its ever records once its record has ended elsewhere.
    So it asks first: the record is read as every process reads it (:func:`settle_job_record`, which writes it lost
    when its lease has lapsed), and one that has ended is refused. A record never stored is not refused.

    Args:
        store: Where the record lives.
        kind: Its type.
        record_id: Its id.
        scope_id: Its scope.
        leases: The leases of the process asking; a record it has let go of is refused (see :func:`write_held_record`).

    Raises:
        JobFenced: The record has ended — written lost by a reader, by a shutdown, or here because its lease lapsed —
            or this process let go of it.
        ConflictError: The record kept changing while its lapse was being written; nothing was decided.
        StorageError: A read or write failed.
    """
    _refuse_let_go(leases, kind, record_id)
    record, _ = settle_job_record(store, kind, record_id, scope_id)
    if record is not None and kind.ended(record):
        how = "was lost" if _leased(record).lost is not None else "has ended"
        raise JobFenced(f"{kind.name} job '{record_id}' {how}; it starts nothing more")


def _refuse_let_go(leases: JobLeases | None, kind: LeasedJobKind[Any], record_id: str) -> None:
    """Refuse a job's own write or spend once its process has let go of its lease — the one rule every kind keeps.

    Raises:
        JobFenced: ``leases`` has let go of ``record_id``.
    """
    if leases is not None and leases.fenced(record_id):
        raise JobFenced(
            f"{kind.name} job '{record_id}': this process let go of its lease; it writes and starts nothing more"
        )


def request_job_cancel(store: LeaseStore, record_id: str, scope_id: str, *, reason: str | None) -> bool:
    """Ask a job to stop by writing the request on its record, from any process.

    The holder reads it at its next renewal and ends the job ``cancelled``; the write is unconditional and mints a
    new etag, so a holder mid-write re-reads and sees it.

    Args:
        store: Where the record lives.
        record_id: The job's record.
        scope_id: Its scope.
        reason: Why, when the asker said; recorded on the record and on a run's ``cancellation_reason``.

    Returns:
        Whether the record was written (False: there is no such document).

    Raises:
        StorageError: The write failed.
    """
    fields: dict[str, Any] = {"cancel_requested_at": store.store_time().isoformat(), "cancel_request_reason": reason}
    return store.merge_job_record_fields(record_id, scope_id, fields)


def write_held_record[R: EvalDocumentModel](
    store: JobRecordStore, kind: LeasedJobKind[R], record: R, *, leases: JobLeases | None = None
) -> R:
    """The job's own write of its record: lands only while the stored record has not ended, keeping its lease fields.

    Reads the stored record and writes ``record`` over it with ``if_match``, carrying the stored lease, cancel
    request and loss rather than this copy's (the holder's copy is older than a renewal or a request); a lost race
    re-reads and re-applies. A stored record that has ended — written lost by a reader that found the lease lapsed,
    or by a shutdown — refuses the write: the job writes nothing more. A record never stored (a generation run
    inline, which records its attempt only as it ends) is written as it is.

    Args:
        store: Where the record lives.
        kind: Its type.
        record: The record as the job would write it.
        leases: The leases of the process writing it. A record this process has let go of
            (:meth:`JobLeases.fenced`) is refused here before anything is read: the store may not say so yet — no
            reader has written the loss, or the store's clock has not reached the expiry — and the process that
            let go is the one that knows. ``None`` for a write no process holds a lease for.

    Returns:
        The record as written.

    Raises:
        JobFenced: The stored record has already ended, or this process let go of it; nothing was written.
        ConflictError: Every attempt lost its race.
        StorageError: A read or write failed.
    """
    _refuse_let_go(leases, kind, _leased(record).id)
    for _ in range(LEASE_WRITE_ATTEMPTS):
        stored, etag = store.load_job_record(kind.model, _leased(record).id, _leased(record).scope_id)
        if stored is not None and kind.ended(stored):
            how = "was lost" if _leased(stored).lost is not None else "has ended"
            raise JobFenced(f"{kind.name} job '{_leased(record).id}' {how} by another writer; it writes nothing more")
        written = record
        if stored is not None:
            held = _leased(stored)
            written = record.model_copy(
                update={
                    "lease": held.lease,
                    "cancel_requested_at": held.cancel_requested_at,
                    "cancel_request_reason": held.cancel_request_reason,
                    "lost": held.lost,
                }
            )
        try:
            store.save_job_record(_document(written), if_match=etag)
        except ConflictError:
            # NOSILENT: a lost race re-reads and re-applies; exhausting the attempts raises below
            continue
        return written
    raise ConflictError(
        f"{kind.name} job '{_leased(record).id}' kept changing while its own write was being made",
        remedy="read it again with job_poll",
    )


#: What one renewal of one held record found: renewed (and nothing asked of it), a cancel asked of it, the job
#: ended by its own writes, the holder fenced off (its record ended by another, or its lease lapsed), or the
#: write lost every race (tried again next interval).
RenewalVerdict = Literal["renewed", "cancel", "ended", "fenced", "contended"]


@dataclass(kw_only=True)
class _Held:
    """One record this process holds: where it is, and what to do when it is cancelled or fenced off."""

    kind: LeasedJobKind[Any]
    record_id: str
    scope_id: str
    cancel: Callable[[str | None], object]
    stop: Callable[[], object]
    cancel_delivered: bool = field(default=False)
    #: When the lease was last known extended, on this process's monotonic clock: as it was held (just after the
    #: write carrying its grant), then as each successful renewal's write was issued — before it, so the reckoning
    #: never credits the lease with more life than the store gave it.
    renewed_at: float = field(default_factory=time.monotonic)


class JobLeases:
    """The leases one process holds on its jobs' records, and the task that renews them.

    Built once per job manager, which a launching host builds once per process
    (:class:`~threetears.evals.run.LaunchHost`), so :attr:`holder_id` — a uuid7 minted here — names the process on
    every record it holds. A job is held from its record's first write (:meth:`grant` stamps the lease that write
    carries, :meth:`hold` starts renewing it) to its own terminal write (:meth:`release`).

    Every :attr:`renew_every_s` (a third of :attr:`lease_s`) each held record is re-read and its lease extended with
    a conditional write. What the read finds is acted on there: a cancel asked of the job is handed to the job's
    ``cancel`` (once), and a record ended by someone else, or a lease that lapsed before it was renewed, stops the
    job through its ``stop`` and releases it — the holder writes nothing more. **A holder whose renewals keep
    failing fences itself**: once its last successful renewal is a whole :attr:`lease_s` old on its own clock, its
    lease has lapsed whatever the store can say, so the job is stopped the same way (and written lost, when the
    store takes the write) rather than left spending past an expiry no reader honours.
    """

    def __init__(
        self,
        store: LeaseStore,
        *,
        lease_s: float = DEFAULT_JOB_LEASE_S,
        blocking_executor: Executor | None = None,
    ) -> None:
        """Bind the leases to the store the records live in.

        Args:
            store: Where every held record lives.
            lease_s: How long a lease lasts unless renewed, in seconds; renewed every third of it.
            blocking_executor: Where the store round-trips run; ``None`` for the loop's default executor.

        Raises:
            ValueError: ``lease_s`` is not positive.
        """
        if not lease_s > 0:
            raise ValueError(f"a job lease lasts some time; lease_s must be positive, got {lease_s!r}")
        self._store = store
        self._lease_s = float(lease_s)
        self._executor = blocking_executor
        self._holder_id = str(uuid.uuid7())
        self._held: dict[str, _Held] = {}
        # Records this process let go of — a renewal found them ended by another writer, or the lease lapsed by
        # this process's own count — whose jobs have not yet ended. See fenced.
        self._let_go: set[str] = set()
        self._renewer: asyncio.Task[None] | None = None

    @property
    def holder_id(self) -> str:
        """The id this process's leases name it by."""
        return self._holder_id

    @property
    def lease_s(self) -> float:
        """How long a lease lasts unless renewed, in seconds."""
        return self._lease_s

    @property
    def renew_every_s(self) -> float:
        """How often every held lease is renewed: a third of :attr:`lease_s`, so two renewals may fail before it lapses."""
        return self._lease_s / 3

    def grant(self) -> JobLease:
        """A fresh lease for a record about to be written: this process, until :attr:`lease_s` from the store's now.

        Blocking: it reads the store's time.

        Returns:
            The lease.
        """
        return JobLease(holder=self._holder_id, expires_at=self._expiry(self._store.store_time()))

    def _expiry(self, now: datetime) -> str:
        return (now + timedelta(seconds=self._lease_s)).isoformat()

    def holds(self, record_id: str) -> bool:
        """Whether this process holds ``record_id``'s lease."""
        return record_id in self._held

    def hold(
        self,
        kind: LeasedJobKind[Any],
        record_id: str,
        scope_id: str,
        *,
        cancel: Callable[[str | None], object],
        stop: Callable[[], object],
    ) -> None:
        """Start renewing a record's lease — its first write carried one from :meth:`grant` — until it is released.

        Called on the event loop, which the renewal task runs on.

        Args:
            kind: The record's type.
            record_id: The record — also the job's id in the manager.
            scope_id: Its scope.
            cancel: Ends the job ``cancelled``, given the reason asked; called once, when a renewal finds a cancel
                asked of it.
            stop: Stops the job at once; called when a renewal finds its record ended by another or its lease
                lapsed, after which the job's writes are refused.
        """
        self._held[record_id] = _Held(kind=kind, record_id=record_id, scope_id=scope_id, cancel=cancel, stop=stop)
        if self._renewer is None or self._renewer.done():
            self._renewer = asyncio.create_task(self._renew_while_held(), name="eval-job-leases")

    def release(self, record_id: str) -> None:
        """Stop renewing a record's lease: its job has written its own ending. Idempotent."""
        self._held.pop(record_id, None)

    def let_go(self, record_id: str) -> None:
        """Give a record's lease up as lost: stop renewing it, and refuse every later write of it from this process.

        Done here when a renewal finds the record ended by another writer or its lease lapsed by this process's own
        count, and by a job that finds its record lost at its own write. From then until the job ends
        (:meth:`forget`), :meth:`fenced` answers ``True``, and every job kind's own writes — a run's status, a
        held record's write (:func:`write_held_record`), a check before spending (:func:`require_held`) — are
        refused by it. Idempotent.

        Args:
            record_id: The record.
        """
        self._held.pop(record_id, None)
        self._let_go.add(record_id)

    def fenced(self, record_id: str) -> bool:
        """Whether this process let go of ``record_id``'s lease (:meth:`let_go`) and its job has not yet ended."""
        return record_id in self._let_go

    def forget(self, record_id: str) -> None:
        """Forget a record this process let go of: its job has ended, and writes nothing more. Idempotent."""
        self._let_go.discard(record_id)

    async def _renew_while_held(self) -> None:
        """Renew every held lease each interval, until none is held."""
        while self._held:
            await asyncio.sleep(self.renew_every_s)
            for record_id, held in list(self._held.items()):
                issued = time.monotonic()
                verdict: RenewalVerdict | None = None
                reason: str | None = None
                try:
                    verdict, reason = await run_blocking(self._executor, self._renew, held)
                # prawduct:ok-broad-except — a failed renewal is retried next interval; the lease outlives two failures
                except Exception:
                    log.exception(
                        "eval.job_lease renewal failed %s=%s scope=%s — retried in %.0fs",
                        held.kind.name,
                        record_id,
                        held.scope_id,
                        self.renew_every_s,
                    )
                if self._held.get(record_id) is not held:
                    continue  # released while the renewal ran: its job wrote its own ending
                if verdict in ("renewed", "cancel"):
                    held.renewed_at = issued
                elif verdict in (None, "contended") and time.monotonic() - held.renewed_at >= self._lease_s:
                    # Not renewed — failed, or every write lost its race — for a whole lease.
                    await self._expire(held)
                    continue
                if verdict is not None:
                    self._act_on(held, verdict, reason)

    async def _expire(self, held: _Held) -> None:
        """Stop a job whose renewals have failed for a whole lease: its lease has lapsed by this process's own count.

        Every reader may already read it lost, and the next one to look writes so — so the holder stops at once
        rather than spend past an expiry it cannot extend. It writes the loss itself when it can (the store may be
        what failed); either way the job's own writes are refused from here on (:meth:`let_go`), even once the
        store recovers.
        """
        self.let_go(held.record_id)
        log.error(
            "eval.job_lease %s=%s scope=%s could not be renewed for %.0fs, its whole lease — stopping it as lost",
            held.kind.name,
            held.record_id,
            held.scope_id,
            self._lease_s,
        )
        try:
            await run_blocking(self._executor, settle_job_record, self._store, held.kind, held.record_id, held.scope_id)
        # prawduct:ok-broad-except — the store may be what failed; the record reads lost to the next reader regardless
        except Exception:
            log.warning(
                "eval.job_lease %s=%s scope=%s: the loss could not be written here; the next reader writes it",
                held.kind.name,
                held.record_id,
                held.scope_id,
                exc_info=True,
            )
        held.stop()

    def _act_on(self, held: _Held, verdict: RenewalVerdict, reason: str | None) -> None:
        """Do what a renewal found: deliver a cancel once, or release and stop a job fenced off its record."""
        if verdict == "cancel" and not held.cancel_delivered:
            held.cancel_delivered = True
            log.info(
                "eval.job_lease cancel asked of %s=%s scope=%s reason=%s",
                held.kind.name,
                held.record_id,
                held.scope_id,
                reason,
            )
            held.cancel(reason)
        elif verdict == "ended":
            self.release(held.record_id)
        elif verdict == "fenced":
            self.let_go(held.record_id)
            log.warning(
                "eval.job_lease %s=%s scope=%s lost its lease — stopping it; its record answers how it ended",
                held.kind.name,
                held.record_id,
                held.scope_id,
            )
            held.stop()
        elif verdict == "contended":
            log.warning(
                "eval.job_lease %s=%s scope=%s renewal lost %d races — tried again in %.0fs",
                held.kind.name,
                held.record_id,
                held.scope_id,
                LEASE_WRITE_ATTEMPTS,
                self.renew_every_s,
            )

    def _renew(self, held: _Held) -> tuple[RenewalVerdict, str | None]:
        """Renew one held lease: the blocking half of a renewal, one worker hop.

        Returns:
            The verdict, and the cancel's reason when one was asked.
        """
        for _ in range(LEASE_WRITE_ATTEMPTS):
            record, etag = self._store.load_job_record(held.kind.model, held.record_id, held.scope_id)
            if record is None:
                return "fenced", None
            leased = _leased(record)
            if leased.lost is not None:
                return "fenced", None
            if held.kind.ended(record):
                return "ended", None
            if leased.lease is None or leased.lease.holder != self._holder_id:
                return "fenced", None
            now = self._store.store_time()
            if lease_lapsed(leased, now):
                # Past its own expiry the lease is gone, whoever has noticed: written lost here as any reader
                # would, so this holder's later writes are refused by the record rather than by its own memory.
                settle_job_record(self._store, held.kind, held.record_id, held.scope_id)
                return "fenced", None
            renewed = record.model_copy(
                update={"lease": JobLease(holder=self._holder_id, expires_at=self._expiry(now))}
            )
            asked = leased.cancel_requested_at is not None
            try:
                self._store.save_job_record(_document(renewed), if_match=etag)
            except ConflictError:
                # NOSILENT: another writer (the job's own progress, a cancel request) moved the record; re-read
                continue
            return ("cancel", leased.cancel_request_reason) if asked else ("renewed", None)
        return "contended", None

    async def shut_down(self, cause: str = SHUTDOWN_CAUSE) -> list[str]:
        """End every job this process holds as lost, releasing each lease, and stop renewing.

        Called by a graceful shutdown BEFORE the jobs are cancelled: each record then reads lost with ``cause`` at
        once, in every process, rather than ``running`` until its lease lapses, and the jobs' own endings, written
        as they unwind, are refused.

        Args:
            cause: The loss's cause.

        Returns:
            The records written lost; one already ended, or whose write failed (logged), is not among them.
        """
        if self._renewer is not None:
            self._renewer.cancel()
            self._renewer = None
        held, self._held = list(self._held.values()), {}
        ended: list[str] = []
        for one in held:
            try:
                if await run_blocking(self._executor, self._end_lost, one, cause):
                    ended.append(one.record_id)
            # prawduct:ok-broad-except — one record's failed write must not keep the rest of a shutdown from ending theirs
            except Exception:
                log.exception(
                    "eval.job_lease shutdown could not write %s=%s scope=%s lost — it reads lost once its lease lapses",
                    one.kind.name,
                    one.record_id,
                    one.scope_id,
                )
        return ended

    def _end_lost(self, held: _Held, cause: str) -> bool:
        """Write one held record lost with ``cause`` and no lease, unless it has ended; whether it was written."""
        for _ in range(LEASE_WRITE_ATTEMPTS):
            record, etag = self._store.load_job_record(held.kind.model, held.record_id, held.scope_id)
            if record is None or held.kind.ended(record):
                return False
            now = self._store.store_time()
            lost = held.kind.end_lost(record, JobLoss(at=now.isoformat(), cause=cause), self._store)
            try:
                self._store.save_job_record(_document(lost.model_copy(update={"lease": None})), if_match=etag)
            except ConflictError:
                # NOSILENT: the job's own write moved the record; re-read and end it from there
                continue
            return True
        raise ConflictError(
            f"{held.kind.name} job '{held.record_id}' kept changing while being ended at shutdown",
            remedy="read it again with job_poll once the process has stopped",
        )


__all__ = [
    "ANALYSIS_JOB",
    "DEFAULT_JOB_LEASE_S",
    "LEASE_WRITE_ATTEMPTS",
    "RUN_JOB",
    "SHUTDOWN_CAUSE",
    "SWEEP_JOB",
    "JobLeases",
    "LeasedJobKind",
    "RenewalVerdict",
    "lease_lapsed",
    "request_job_cancel",
    "require_held",
    "settle_job_record",
    "write_held_record",
]
