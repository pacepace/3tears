"""Job leases: which process runs a job is written on the job's record, so every process answers it alike.

A job's work — a run, a sweep, an analysis generation — runs in one process, while its record lives in a store
several processes share. Liveness used to be answered from the job manager's memory, so a poll in a process that
did not start a job read a live one as lost, and a cancel there was refused. Here it is answered from the record:

- **The holder keeps a lease.** A job's record is written with a lease (:class:`~threetears.evals.schema.JobLease`:
  the holding process's id and an expiry) before its work starts, and :class:`JobLeases` — one per job manager,
  so one per process — renews it with a conditional write every
  :data:`~threetears.evals.schema.JOB_LEASE_RENEWAL_FRACTION` of its duration while the work runs.
- **A job is lost only when its record has not ended and its lease has lapsed** (or it carries none: a run a
  launch prepared and never started). Every process answers that the same, and the first reader to see it WRITES it
  (:func:`settle_job_record`): the record takes its own ending and :class:`~threetears.evals.schema.JobLoss`, with
  ``if_match``, so a race between two readers, or a reader and a renewal, is lost by one of them, which re-reads.
  Once written, the answer is the record's, never recomputed. There is no rule about the order a reader asks
  things in any more: there is one thing to read.
- **A holder that has lost its lease writes nothing more.** Its renewal finding the record ended by someone else,
  or its own lease lapsed (which it writes lost itself, as any reader would), stops the work at once; a write the
  work makes of its own record goes through :func:`write_held_record`, which refuses a record already ended and
  raises :class:`~threetears.evals.kernel.JobFenced`. So does a holder's write finding the stored lease passed to
  another holding (another process, or a later ``epoch`` of the job, :meth:`JobLeases.write_own`), and a run's cell
  write into a run written lost or held at a later epoch (:meth:`~threetears.evals.kernel.EvalStorage.save_eval_result`).
  A run's own writes are refused the same way: its status write takes nothing over a record carrying ``lost``, its
  own ending included, and the process that lost it neither broadcasts that ending nor tells the run-end hook. The
  one thing a holder whose renewals failed for a whole lease still writes is why, as a fault on its record
  (``faults``, every kind alike), never its status or its ending.
- **A holder's own writes keep its lease too.** Every conditional write a job makes of its own record — a run's
  progress and status, a launch's, a sweep's or an attempt's record — goes through :meth:`JobLeases.write_own`, which
  extends the lease as a renewal does. So a job whose writes keep beating its renewals to the record keeps its lease,
  and hears a cancel asked of it, as a renewal would have; a lease the store's clock reads lapsed is never revived.
- **The one way back is a resume, under a new epoch.** No renewal and no write of the holding that lost a job brings
  it back. A run that was lost (or cancelled, or stopped by its budget or its account) is continued only when someone
  asks (:func:`~threetears.evals.run.resume_run`, the ``job_resume`` action): its record is reopened, conditionally on
  the record as read, under a lease at the next ``epoch`` held by the process resuming it, and every holding at an
  earlier epoch is fenced as above. Nothing resumes a job on its own, at boot or otherwise.
- **A cancel is a field on the record** (``cancel_requested_at``), written by whichever process was asked
  (:func:`request_job_cancel`); the holder reads it at its next renewal and ends the job ``cancelled``, so a cancel
  lands within one renewal interval from anywhere.
- **A graceful shutdown ends every job the process holds as lost** (:meth:`JobLeases.shut_down`), its cause the
  shutdown, releasing the lease, before the work is cancelled.

**Time** is the store's when it keeps a clock (:class:`~threetears.evals.schema.StoreClock`), else each process's
own (:meth:`~threetears.evals.kernel.EvalStorage.store_time`); processes sharing a store without a clock must keep
their clocks in step to well inside a lease.

**One piece for every job record.** A record type takes the five lease fields
(:mod:`~threetears.evals.schema.job_lease`) and a :class:`LeasedJobKind` saying when it has ended and how it reads
lost; everything here is generic over it. :data:`RUN_JOB`, :data:`SWEEP_JOB` and :data:`ANALYSIS_JOB` are the three
the engine runs.
"""

from __future__ import annotations

import asyncio
import contextvars
import random
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast

from pydantic import ValidationError

from threetears.evals.kernel.campaign import EvalAnalysisAttempt, EvalSweep
from threetears.evals.kernel.errors import ConflictError, JobFenced, StorageError
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.run_pages import iter_stored_results
from threetears.evals.kernel.scoring import reconstruct_completeness
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.job_lease import (
    DEFAULT_JOB_LEASE_S,
    JOB_LEASE_RENEWAL_FRACTION,
    LEASE_POOL_WORKERS,
    JobLease,
    JobLoss,
    LeasedRecord,
    with_fault,
)
from threetears.evals.schema.models import TERMINAL_RUN_STATUSES, EvalRun
from threetears.evals.schema.logs import contain, get_logger

if TYPE_CHECKING:
    # The store settles what it reads through this module, so its ports are named for the checker alone.
    from threetears.evals.kernel.storage import JobRecordStore, LeaseStore

log = get_logger(__name__)

#: How many times a lease write (a renewal, a lost write, a holder's own write) re-reads and re-applies after losing
#: its conditional write to another writer, before it gives up for now. Small: each attempt is a fresh read, and a
#: record that refuses three in a row is contended past what a retry fixes — the next renewal tries again.
LEASE_WRITE_ATTEMPTS = 3

#: How many more times a renewal that lost :data:`LEASE_WRITE_ATTEMPTS` races to the holder's own writes is tried again,
#: within the same interval, before it waits the whole interval.
CONTENDED_RENEWAL_RETRIES = 4

#: The jittered wait, in seconds, before a contended renewal is tried again (a short lease caps it at a quarter of the
#: renewal interval, so the retries always fit inside one).
CONTENDED_BACKOFF_S = (0.5, 2.0)

#: The lost cause a graceful shutdown writes on every job its process holds.
SHUTDOWN_CAUSE = "process shut down"

#: The status a run written lost takes (:data:`RUN_JOB`): what a run that stopped without finishing has always read,
#: its cause on ``cancellation_reason`` and ``lost``.
RUN_LOST_STATUS: Final = "cancelled"


@dataclass(frozen=True, kw_only=True)
class LeasedJobKind[R: EvalDocumentModel]:
    """One type of job record, as the lease machinery reads and ends it.

    :ivar name: The job kind's name, for a message.
    :ivar model: The record's model; it declares the five lease fields (:class:`~threetears.evals.schema.LeasedRecord`).
    :ivar ended: Whether the record says, by its own status, that the job has ended.
    :ivar end_lost: The record as it is written lost: its own ending set and ``loss`` recorded on it. Given the store
        because a run's ending reconstructs its completeness from its stored results.
    :ivar named_runs: The runs the record names as its own (a launch's, a sweep's arms'), which are settled with it
        when it is written lost (:func:`settle_job_record`); none by default.


    Every kind constructed is registered by its model, so every store read of a record of that model settles it
    (:func:`leased_kind_of`, :meth:`~threetears.evals.kernel.EvalStorage.load_eval_run` and every other read) — a
    host's own kinds, built with ``leased_job_kind``, included.
    """

    name: str
    model: type[R]
    ended: Callable[[R], bool]
    end_lost: Callable[[R, JobLoss, LeaseStore], R]
    named_runs: Callable[[R], Sequence[str]] = lambda _record: ()

    def __post_init__(self) -> None:
        """Register the kind by its model, so a read of the model's records can settle them."""
        _KIND_OF_MODEL[self.model] = self


#: Every leased kind constructed, by its record model: what a store read settles a record of that model by.
_KIND_OF_MODEL: dict[type[EvalDocumentModel], LeasedJobKind[Any]] = {}


def leased_kind_of(model: type[EvalDocumentModel]) -> LeasedJobKind[Any] | None:
    """The leased kind whose records are ``model``'s, or ``None`` for a model no job records itself in.

    :param model: A stored model.
    :ptype model: type[EvalDocumentModel]
    :return: Its kind, once a :class:`LeasedJobKind` of it has been constructed — at the import of the module that
        defines the model's kind, which a reader of the model has already imported.
    :rtype: LeasedJobKind[Any] | None
    """
    return _KIND_OF_MODEL.get(model)


def _run_lost(run: EvalRun, loss: JobLoss, store: LeaseStore) -> EvalRun:
    """A run written lost: ``cancelled`` — the status a run that stopped without finishing has always taken — its
    reason the loss, and its completeness reconstructed from the results it stored unless its loop recorded its own.

    **Settled from the run record, whatever its results hold.** The completeness is the one part read from anything
    else, and a result this build cannot reconstruct (a corpus that outlived the model that wrote it) fails the
    whole listing of them. That must not keep the run from settling — it would read ``running`` on every surface,
    and break every list that settles it — so the run is written lost with its completeness unknown (``None``, which
    every reader of the field reads as unknown), and why it is unknown is written on the run itself, as an
    ``error_details`` entry naming the validation error: a result the run stored cannot be read, which is a fault.
    """
    completeness = run.completeness
    unreadable: list[str] = []
    if completeness is None:
        try:
            # Read a page at a time: the run is lost at its largest, and the count keeps nothing of a result.
            completeness = reconstruct_completeness(run, iter_stored_results(store, run.id, run.scope_id))
        # A result this build cannot reconstruct refuses as a ValidationError; one whose key (id, case, k, model, when
        # stored) it cannot read refuses the paged read's pass over the run as a StorageError. Either leaves the run's
        # completeness unknown, said on the run, and the run still settles.
        except (ValidationError, StorageError) as refused:
            # NOSILENT: the completeness is written None, which every reader reads as unknown, and why is written
            # on the run itself as an error_details entry below.
            log.exception(
                "eval run %s (scope %s) is written lost with its completeness unknown: a result it stored cannot "
                "be read under the current model, so what it delivered cannot be counted",
                run.id,
                run.scope_id,
            )
            unreadable.append(
                "written lost with its completeness unknown: a result it stored cannot be read under the current "
                f"model, so what it delivered cannot be counted — {refused}"
            )
    lost = run.model_copy(
        update={
            "cancellation_reason": f"lost: {loss.cause}",
            "completed_at": loss.at,
            "completeness": completeness,
            "error_details": [*run.error_details, *unreadable],
            "lost": loss,
        }
    )
    # Set, not copied in: the run's model derives its current holding's ending from its own as it is set, so a run
    # taken up again loses that holding with it.
    lost.status = RUN_LOST_STATUS
    return lost


def _sweep_lost(sweep: EvalSweep, loss: JobLoss, _store: LeaseStore) -> EvalSweep:
    """A sweep written lost: ``cancelled``, no further arm launched, its detail the loss, and a sweep taken up again
    losing its current holding with it."""
    lost = sweep.model_copy(update={"detail": f"lost: {loss.cause}", "ended_at": loss.at, "lost": loss})
    # Set, not copied in: the sweep's model derives its current holding's ending from its own as it is set.
    lost.outcome = "cancelled"
    return lost


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
    name="sweep",
    model=EvalSweep,
    ended=lambda sweep: sweep.outcome != "running",
    end_lost=_sweep_lost,
    named_runs=lambda sweep: [arm.run_id for arm in sweep.arms if arm.run_id is not None],
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

    :raises ValueError: A run read by a listing that left payload paths out.
    """
    if isinstance(record, EvalRun) and record.elided_payload_paths:
        raise ValueError(
            f"eval run {record.id} was read by a listing that left {sorted(record.elided_payload_paths)} out; "
            "a job record is written only as read whole"
        )
    return record.to_dict()


def lease_lapsed(record: LeasedRecord, now: datetime) -> bool:
    """Whether a record's lease has lapsed at ``now`` — or it has none, which no process can be renewing.

    **No grace for a record with no lease.** A job's record takes its lease before its work starts, so one that
    has not ended and carries none was left by a process that ended first — a run its launch prepared and never
    started — and nothing will ever renew it. A grace measured from when it was written would only delay the same
    answer, so the rule is stated rather than guessed at: a lease-less record that has not ended is lost.

    :param record: A job record.
    :ptype record: LeasedRecord
    :param now: The time to read it at, in the store's time when the store keeps one.
    :ptype now: datetime
    :return: ``True`` when the record carries no lease or its expiry is at or before ``now``.
    :rtype: bool
    """
    return record.lease is None or record.lease.expires() <= now


def _passed(lease: JobLease | None, holder: str, epoch: int) -> bool:
    """Whether a stored lease has passed from the holding ``(holder, epoch)``: it names another process, or the same
    process at another epoch — the job was taken up again. A record carrying no lease has passed from nobody: it reads
    lost by its own rule (:func:`lease_lapsed`)."""
    return lease is not None and (lease.holder, lease.epoch) != (holder, epoch)


def _lapse_cause(record: LeasedRecord) -> str:
    """Why a record whose lease lapsed is lost, in a poll's words."""
    if record.lease is None:
        return (
            "it carries no lease, so no process can be running it — a job's record takes its lease before its "
            "work starts, and the process that wrote this one ended before that"
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

    :param store: Where the record lives.
    :ptype store: LeaseStore
    :param kind: Its type.
    :ptype kind: LeasedJobKind[R]
    :param record_id: Its id.
    :ptype record_id: str
    :param scope_id: The scope it is read in; a record in another scope reads absent.
    :ptype scope_id: str
    :return: ``(record, wrote_lost)``: the record as it now stands, or None when there is none; and whether THIS call
        wrote it lost.
    :rtype: tuple[R | None, bool]
    :raises ConflictError: Every attempt to write the lapse lost its race; nothing was decided, read it again.
    :raises StorageError: A read or write failed.
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
        lease = _leased(record).lease
        loss = JobLoss(
            at=now.isoformat(), cause=_lapse_cause(_leased(record)), epoch=lease.epoch if lease is not None else 0
        )
        lost = kind.end_lost(record, loss, store)
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
        _settle_named_runs(store, kind, lost, scope_id)
        return lost, True
    raise ConflictError(
        f"{kind.name} job '{record_id}' kept changing while its lapsed lease was being written lost",
        remedy="read it again with job_poll",
    )


def _settle_named_runs[R: EvalDocumentModel](store: LeaseStore, kind: LeasedJobKind[R], lost: R, scope_id: str) -> None:
    """Settle every run a record just written lost names: each is read as any reader would read it.

    A launch or a sweep holds its runs in the process that held it, so when its lease lapsed theirs did too, and a run
    it prepared but never started carries no lease at all; either reads lost. A run whose lease another process still
    renews is left running. One run another writer keeps moving (:func:`settle_tolerantly`) is left to its own next
    read, and never keeps the record that named it, or the next run it names, from settling. A run this build cannot
    read raises, once the record that named it is written lost, as any read of that run does.
    """
    for run_id in kind.named_runs(lost):
        settle_tolerantly(store, RUN_JOB, run_id, scope_id)


def settle_tolerantly[R: EvalDocumentModel](
    store: LeaseStore, kind: LeasedJobKind[R], record_id: str, scope_id: str
) -> R | None:
    """:func:`settle_job_record` for a read of many records: one another writer keeps moving is left as stored.

    For every read that settles records as it lists them: a record another writer keeps moving (``ConflictError``)
    is live enough to be read again, and its own next read settles it, so it never breaks a list — of runs, of a
    sweep's arms, of a launch's runs — and is logged. Nothing else is tolerated. A storage failure is not one
    record's, and a record this build cannot read, or cannot write lost (``ValidationError``), raises here as every
    other read of it does: left as stored, it read as a job still running, with nothing saying it was unreadable.

    :param store: Where the record lives.
    :ptype store: LeaseStore
    :param kind: Its type.
    :ptype kind: LeasedJobKind[R]
    :param record_id: Its id.
    :ptype record_id: str
    :param scope_id: Its scope.
    :ptype scope_id: str
    :return: The record as it now stands, or ``None`` where another writer kept moving it (or there is none): the caller
        keeps what it read.
    :rtype: R | None
    :raises StorageError: A read or write failed.
    :raises ValidationError: The record cannot be read under the current model, or a model refused a value while it was
        written lost.
    """
    try:
        current, _ = settle_job_record(store, kind, record_id, scope_id)
    except ConflictError:
        # NOSILENT: a race whose loser re-reads -- a record another writer keeps moving is live, and its own next
        # read settles it; the caller keeps the record as it read it, which is the record as last written.
        log.warning("eval.job_lease %s=%s scope=%s kept changing while it was settled", kind.name, record_id, scope_id)
        return None
    return current


def require_held[R: EvalDocumentModel](
    store: LeaseStore, kind: LeasedJobKind[R], record_id: str, scope_id: str, *, leases: JobLeases | None = None
) -> None:
    """Refuse to go on with a job whose record has ended under it: the check a job makes before it spends again.

    A job's own writes are fenced (:func:`write_held_record`), but a job that spends between two writes — a sweep
    launching its next arm — would pay for work no write of its ever records once its record has ended elsewhere.
    So it asks first: the record is read as every process reads it (:func:`settle_job_record`, which writes it lost
    when its lease has lapsed), and one that has ended is refused. A record never stored is not refused.

    :param store: Where the record lives.
    :ptype store: LeaseStore
    :param kind: Its type.
    :ptype kind: LeasedJobKind[R]
    :param record_id: Its id.
    :ptype record_id: str
    :param scope_id: Its scope.
    :ptype scope_id: str
    :param leases: The leases of the process asking; a record it has let go of is refused (see
        :func:`write_held_record`).
    :ptype leases: JobLeases | None
    :raises JobFenced: The record has ended — written lost by a reader, by a shutdown, or here because its lease lapsed
        — or this process let go of it.
    :raises ConflictError: The record kept changing while its lapse was being written; nothing was decided.
    :raises StorageError: A read or write failed.
    """
    _refuse_let_go(leases, kind, record_id)
    record, _ = settle_job_record(store, kind, record_id, scope_id)
    if record is not None and kind.ended(record):
        how = "was lost" if _leased(record).lost is not None else "has ended"
        raise JobFenced(f"{kind.name} job '{record_id}' {how}; it starts nothing more")


def _refuse_let_go(leases: JobLeases | None, kind: LeasedJobKind[Any], record_id: str) -> None:
    """Refuse a job's own write or spend once its process has let go of its lease — the one rule every kind keeps.

    :raises JobFenced: ``leases`` has let go of ``record_id``.
    """
    if leases is not None and leases.fenced(record_id):
        raise JobFenced(
            f"{kind.name} job '{record_id}': this process let go of its lease; it writes and starts nothing more"
        )


def request_job_cancel(store: LeaseStore, record_id: str, scope_id: str, *, reason: str | None) -> bool:
    """Ask a job to stop by writing the request on its record, from any process.

    The holder reads it at its next renewal and ends the job ``cancelled``; the write is unconditional and mints a
    new etag, so a holder mid-write re-reads and sees it.

    :param store: Where the record lives.
    :ptype store: LeaseStore
    :param record_id: The job's record.
    :ptype record_id: str
    :param scope_id: Its scope.
    :ptype scope_id: str
    :param reason: Why, when the asker said; recorded on the record and on a run's ``cancellation_reason``.
    :ptype reason: str | None
    :return: Whether the record was written (False: there is no such document).
    :rtype: bool
    :raises StorageError: The write failed.
    """
    fields: dict[str, Any] = {"cancel_requested_at": store.store_time().isoformat(), "cancel_request_reason": reason}
    return store.merge_job_record_fields(record_id, scope_id, fields)


def write_held_record[R: EvalDocumentModel](
    store: JobRecordStore, kind: LeasedJobKind[R], record: R, *, leases: JobLeases | None = None
) -> R:
    """The job's own write of its record: lands only while the stored record has not ended, keeping its lease fields.

    Reads the stored record and writes ``record`` over it with ``if_match``, carrying the stored lease, cancel
    request, loss and faults rather than this copy's (the holder's copy is older than a renewal, a request or a
    fault written on the record); a lost race re-reads and re-applies. A stored record that has ended — written lost
    by a reader that found the lease lapsed, or by a shutdown — refuses the write: the job writes nothing more. A
    record never stored (a generation run inline, which records its attempt only as it ends) is written as it is.

    With ``leases``, the write is the holder's own (:meth:`JobLeases.write_own`), which extends the lease it holds.

    :param store: Where the record lives.
    :ptype store: JobRecordStore
    :param kind: Its type.
    :ptype kind: LeasedJobKind[R]
    :param record: The record as the job would write it.
    :ptype record: R
    :param leases: The leases of the process writing it. A record this process has let go of
        (:meth:`JobLeases.fenced`) is refused here before anything is read: the store may not say so yet — no
        reader has written the loss, or the store's clock has not reached the expiry — and the process that
        let go is the one that knows. ``None`` for a write no process holds a lease for.
    :ptype leases: JobLeases | None
    :return: The record as written.
    :rtype: R
    :raises JobFenced: The stored record has already ended, its lease has lapsed by the store's time, or this process
        let go of it; nothing was written.
    :raises ConflictError: Every attempt lost its race.
    :raises StorageError: A read or write failed.
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
                    "faults": held.faults,
                }
            )

        def save(document: R, etag: str | None = etag) -> None:
            store.save_job_record(_document(document), if_match=etag)

        try:
            if leases is not None and stored is not None:
                return leases.write_own(stored, written, save)
            save(written)
        except ConflictError:
            # NOSILENT: a lost race re-reads and re-applies; exhausting the attempts raises below
            continue
        return written
    raise ConflictError(
        f"{kind.name} job '{_leased(record).id}' kept changing while its own write was being made",
        remedy="read it again with job_poll",
    )


class _Reopened(Protocol):
    """A job record a take-up reopens: its lease is the one field the take-up's write sets."""

    lease: JobLease | None


def read_to_take_up[R: EvalDocumentModel](
    store: LeaseStore, kind: LeasedJobKind[R], record_id: str, scope_id: str
) -> tuple[R | None, str | None]:
    """A job record as a take-up reads it: settled as every reader settles it, then read with the token its reopen is
    written conditionally on (:func:`write_reopened`).

    :param store: Where the record lives.
    :ptype store: LeaseStore
    :param kind: Its type.
    :ptype kind: LeasedJobKind[R]
    :param record_id: The record.
    :ptype record_id: str
    :param scope_id: Its scope.
    :ptype scope_id: str
    :return: The record, or ``None`` when there is none, and its token.
    :rtype: tuple[R | None, str | None]
    """
    settle_job_record(store, kind, record_id, scope_id)
    return store.load_job_record(kind.model, record_id, scope_id)


def taken_up_meanwhile(kind: LeasedJobKind[Any], record_id: str) -> ConflictError:
    """The refusal of a take-up whose record moved after it was read: another take-up of it won, or another writer
    moved it.

    :param kind: The record's type.
    :ptype kind: LeasedJobKind[Any]
    :param record_id: The record.
    :ptype record_id: str
    :return: The refusal, for the caller to raise from the store's conflict.
    :rtype: ConflictError
    """
    return ConflictError(
        f"{kind.name} '{record_id}' changed while it was being taken up — another take-up of it won, or another writer "
        "moved it — so this one started nothing",
        remedy="read it with job_poll; call job_resume again if it is still not running",
    )


def write_reopened[R: EvalDocumentModel](
    store: JobRecordStore,
    kind: LeasedJobKind[R],
    record: R,
    *,
    leases: JobLeases | None,
    etag: str | None,
    epoch: int,
) -> None:
    """Write a reopened job record under a lease at its next epoch, conditionally on the record the take-up read.

    :param store: Where the record lives.
    :ptype store: JobRecordStore
    :param kind: Its type.
    :ptype kind: LeasedJobKind[R]
    :param record: The record as the take-up reopened it; its lease is set here, so the job that runs it holds it.
    :ptype record: R
    :param leases: The leases of the process taking it up; ``None`` writes it with no lease.
    :ptype leases: JobLeases | None
    :param etag: The token the take-up read it with (:func:`read_to_take_up`).
    :ptype etag: str | None
    :param epoch: The epoch it is taken up under.
    :ptype epoch: int
    :raises ConflictError: The record moved since it was read: another take-up won, or another writer wrote it.
    """
    cast(_Reopened, record).lease = leases.grant(epoch) if leases is not None else None
    try:
        store.save_job_record(_document(record), if_match=etag)
    except ConflictError as moved:
        raise taken_up_meanwhile(kind, _leased(record).id) from moved


def reopened_job_fields() -> dict[str, Any]:
    """What a take-up clears on a sweep's or a launch's record to reopen it: ``running`` again, its ending, its loss,
    its refusal and its cancel request cleared.

    :return: The fields, for the record's ``model_copy(update=...)``; a new mapping on each call.
    :rtype: dict[str, Any]
    """
    return {
        "outcome": "running",
        "detail": None,
        "ended_at": None,
        "lost": None,
        "refusal": None,
        "cancel_requested_at": None,
        "cancel_request_reason": None,
    }


def _note_fault(store: LeaseStore, kind: LeasedJobKind[Any], record_id: str, scope_id: str, entry: str) -> bool:
    """Append ``entry`` to a job record's ``faults``, conditionally and touching nothing else; whether it was.

    Every kind alike: the faults sit beside the record's ending, so a loss written after this keeps them. The lease
    is not moved, so a store refusing only renewals takes it. ``False`` for a record that is gone. A lost race
    re-reads and re-applies.

    :raises ConflictError: Every attempt lost its race.
    :raises StorageError: A read or write failed.
    """
    for _ in range(LEASE_WRITE_ATTEMPTS):
        record, etag = store.load_job_record(kind.model, record_id, scope_id)
        if record is None:
            return False
        noted = record.model_copy(update={"faults": with_fault(_leased(record).faults, entry)})
        try:
            store.save_job_record(_document(noted), if_match=etag)
        except ConflictError:
            # NOSILENT: a lost race re-reads and re-applies; exhausting the attempts raises below
            continue
        return True
    raise ConflictError(
        f"{kind.name} job '{record_id}' kept changing while a fault was being written on it",
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
    #: The epoch of the lease this process was granted on the record: a stored lease at another epoch, or naming
    #: another holder, is no longer this holding's (:func:`_passed`).
    epoch: int = field(default=0)
    cancel_delivered: bool = field(default=False)
    #: Why the last renewal failed, as ``<error type>: <message>``, until one succeeds: the fault the record is given
    #: when the failures cost the lease (:meth:`JobLeases._expire`).
    last_failure: str | None = field(default=None)
    #: When the lease was last known extended, on this process's monotonic clock: as it was held (just after the
    #: write carrying its grant), then as each successful renewal's write, or the holder's own write extending it
    #: (:meth:`JobLeases.write_own`), was issued — before it, so the reckoning never credits the lease with more life
    #: than the store gave it. Moved on the event loop only.
    renewed_at: float = field(default_factory=time.monotonic)
    #: The renewal store call that outlived its deadline and has not yet returned, and when it was issued: while it is
    #: stuck no second renewal of the record is sent (:meth:`JobLeases._renew_one`).
    stuck_call: Future[tuple[float, RenewalVerdict, str | None]] | None = field(default=None)
    stuck_since: float = field(default=0.0)


class JobLeases:
    """The leases one process holds on its jobs' records, and the task that renews them.

    Built once per job manager, which a launching host builds once per process
    (:class:`~threetears.evals.run.LaunchHost`), so :attr:`holder_id` — a uuid7 minted here — names the process on
    every record it holds. A job is held from its record's first write (:meth:`grant` stamps the lease that write
    carries, :meth:`hold` starts renewing it) to its own terminal write (:meth:`release`).

    Every :attr:`renew_every_s` (:data:`~threetears.evals.schema.JOB_LEASE_RENEWAL_FRACTION` of :attr:`lease_s`) each
    held record is re-read and its lease extended with
    a conditional write. What the read finds is acted on there: a cancel asked of the job is handed to the job's
    ``cancel`` (once), and a record ended by someone else, or a lease that lapsed before it was renewed, stops the
    job through its ``stop`` and releases it — the holder writes nothing more. **A holder whose renewals keep
    failing fences itself**: once its last successful renewal is a whole :attr:`lease_s` old on its own clock, its
    lease has lapsed whatever the store can say, so the job is stopped the same way (and written lost, when the
    store takes the write) rather than left spending past an expiry no reader honours.

    **Renewal never waits on job work.** The store round-trips here (a renewal, an expiry's writes, a record written
    lost, the shutdown's) run on a small pool of this object's own, created when a lease is first held and let go when
    none is, never on the host's blocking executor, which a busy launch's cells fill for longer than a lease. Every
    held lease renews at once, one hop each, so one slow record delays no other. The store is called from that pool's
    threads as it is from the host's: a store must be safe to call from several threads. A renewal that loses its
    conditional writes to the holder's own progress writes is tried again within the interval after a short jittered
    wait (:data:`CONTENDED_BACKOFF_S`), and the holder's own writes (:meth:`write_own`) extend the lease themselves,
    so a renewal that keeps losing to them costs the job nothing.

    **A renewal that lands late says so**: one landing more than one and a half renewal intervals after the lease was
    last extended logs a warning naming how long since, the interval and the lease, how long the renewal waited to be
    scheduled and how long its store call took — so a starved process reads apart from a slow store in the logs.
    """

    def __init__(
        self,
        store: LeaseStore,
        *,
        lease_s: float = DEFAULT_JOB_LEASE_S,
    ) -> None:
        """Bind the leases to the store the records live in.

        :param store: Where every held record lives.
        :ptype store: LeaseStore
        :param lease_s: How long a lease lasts unless renewed, in seconds; renewed every
            :data:`~threetears.evals.schema.JOB_LEASE_RENEWAL_FRACTION` of it.
        :ptype lease_s: float
        :raises ValueError: ``lease_s`` is not positive.
        """
        if not lease_s > 0:
            raise ValueError(f"a job lease lasts some time; lease_s must be positive, got {lease_s!r}")
        self._store = store
        self._lease_s = float(lease_s)
        self._executor: ThreadPoolExecutor | None = None
        self._holder_id = str(uuid.uuid7())
        self._held: dict[str, _Held] = {}
        # Records this process let go of — a renewal found them ended by another writer, or the lease lapsed by
        # this process's own count — whose jobs have not yet ended. See fenced.
        self._let_go: set[str] = set()
        self._renewer: asyncio.Task[None] | None = None
        # The loop the leases are held on (:meth:`hold`), where an own write's bookkeeping is done (:meth:`write_own`).
        self._loop: asyncio.AbstractEventLoop | None = None

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
        """How often every held lease is renewed: :data:`~threetears.evals.schema.JOB_LEASE_RENEWAL_FRACTION` of
        :attr:`lease_s`."""
        return self._lease_s * JOB_LEASE_RENEWAL_FRACTION

    def grant(self, epoch: int = 0) -> JobLease:
        """A fresh lease for a record about to be written: this process, until :attr:`lease_s` from the store's now.

        Blocking: it reads the store's time.

        :param epoch: Which holding of the job the lease is (:attr:`~threetears.evals.schema.JobLease.epoch`): 0 for
            a job starting fresh.
        :ptype epoch: int
        :return: The lease.
        :rtype: JobLease
        """
        return JobLease(holder=self._holder_id, expires_at=self._expiry(self._store.store_time()), epoch=epoch)

    def _pool(self) -> ThreadPoolExecutor:
        """The leases' own pool, made when first needed (:meth:`_close_pool` lets it go; the next need makes another)."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=LEASE_POOL_WORKERS, thread_name_prefix="eval-lease")
        return self._executor

    def _close_pool(self) -> None:
        """Let the pool go: work already handed to it finishes, and its idle threads exit. Done when no lease is held,
        and when a renewal's call is found stuck in it, which retires it with the stuck thread."""
        pool, self._executor = self._executor, None
        if pool is not None:
            pool.shutdown(wait=False)

    def _expiry(self, now: datetime) -> str:
        return (now + timedelta(seconds=self._lease_s)).isoformat()

    def _extended(self, leased: LeasedRecord, now: datetime) -> JobLease | None:
        """The lease ``leased`` is extended to at ``now``, by a renewal or a holder's own write; ``None`` when it is not
        this process's to extend — another holder's, none, or lapsed by ``now``, which nothing revives."""
        lease = leased.lease
        if lease is None or lease.holder != self._holder_id or lease_lapsed(leased, now):
            return None
        return JobLease(holder=self._holder_id, expires_at=self._expiry(now), epoch=lease.epoch)

    def write_own[R: EvalDocumentModel](self, stored: R, written: R, save: Callable[[R], None]) -> R:
        """The holder's own conditional write of a record it holds: ``written`` lands through ``save``, keeping the lease.

        ``stored`` is the record as just read, and ``save`` makes the conditional write under the etag read with it,
        raising :class:`~threetears.evals.kernel.ConflictError` on a lost race. When this process holds the record
        and the stored lease names it, the record goes out with the lease **extended as a renewal extends it** (to
        :attr:`lease_s` from the store's now), and once the write lands it counts as a renewal: the lease is reckoned
        extended from when the write was issued, and a cancel asked of the job is handed to it, on the event loop the
        leases are held on. So a job whose own writes keep beating its renewals to the record keeps its lease, and
        hears a cancel. A record this process has let go of refuses every write. A record this process holds whose
        stored lease names another holding — another process, or this one at another epoch, because the job was taken
        up again — refuses the write: this process lets it go and stops its job (unless the write was its ending), and
        nothing is written. Any other record this process does not hold goes out with its lease as stored, and so does a write that ends the job (its lease no longer matters,
        and a cancel would reach a job already over) and one made while the lease was extended under half a renewal
        interval ago (no renewal is due, so the store's clock is not read for it: a job ticking fast pays that
        round-trip at most twice an interval).

        A lease the store's clock reads lapsed is never revived: the record is written lost, as a renewal writes it,
        this process lets it go and stops its job, and nothing is written. Only a resume continues the job, under a
        new lease at the next epoch (:func:`~threetears.evals.run.resume_run`). A store clock that cannot be read
        leaves the lease as stored, logged: the write it was only to extend still lands.

        Blocking: call it off the event loop, as every store call is.

        :param stored: The record as read for this write.
        :ptype stored: R
        :param written: The record as the job writes it; its lease is replaced.
        :ptype written: R
        :param save: Makes the conditional write of the record handed to it.
        :ptype save: Callable[[R], None]
        :return: The record as written.
        :rtype: R
        :raises JobFenced: This process let go of the record (:meth:`let_go`), or the lease has lapsed by the store's
            time, or has passed to another holding of the job (another process, or another epoch) while this process
            holds the record; nothing was written.
        :raises ConflictError: ``save`` lost its race.
        :raises StorageError: ``save`` failed.
        """
        leased = _leased(stored)
        if leased.id in self._let_go:
            raise JobFenced(f"job '{leased.id}': this process let go of its lease; it writes nothing more")
        held = self._held.get(leased.id)
        lease = leased.lease
        if held is not None and lease is not None and _passed(lease, self._holder_id, held.epoch):
            self._fence_passed(held, lease, ending=held.kind.ended(written))
            raise JobFenced(
                f"{held.kind.name} job '{leased.id}': its lease passed to process {lease.holder} at epoch "
                f"{lease.epoch}; this holding (epoch {held.epoch}) writes nothing more"
            )
        issued = time.monotonic()  # before the store's time is read for the extension, as a renewal's is
        extended: JobLease | None = None
        if (
            held is not None
            and lease is not None
            and lease.holder == self._holder_id
            and not held.kind.ended(written)
            and issued - held.renewed_at >= self.renew_every_s / 2
        ):
            try:
                now = self._store.store_time()
            # prawduct:ok-broad-except — the clock read only extends the lease; whatever failed it, the write still lands
            except Exception as unread:
                # NOSILENT: the write carries the lease as stored, as an own write did before it extended one; the
                # renewals keep the lease as they did then, and this line says the extension was skipped and why.
                contain(
                    unread,
                    log=log,
                    what=(
                        f"eval.job_lease {held.kind.name}={leased.id} scope={leased.scope_id}: the store's time could "
                        "not be read to extend the lease on the job's own write; the write carries it as stored"
                    ),
                    level="warning",
                )
            else:
                extended = self._extended(leased, now)
                if extended is None:
                    self._fence_lapsed(held)
                    raise JobFenced(
                        f"{held.kind.name} job '{leased.id}': its lease lapsed at {lease.expires_at} by the store's "
                        "time; it writes nothing more"
                    )
        out = written.model_copy(update={"lease": extended if extended is not None else lease})
        save(out)
        if held is not None and extended is not None:
            asked = (leased.cancel_request_reason,) if leased.cancel_requested_at is not None else None
            self._on_loop(self._renewed_by_own_write, held, issued, asked)
        return out

    def _fence_passed(self, held: _Held, lease: JobLease, *, ending: bool) -> None:
        """Let go of a record whose stored lease names another holding, and stop its job on the loop unless the refused
        write was its ending. Nothing is written: the record is the other holding's. Called off the loop, from
        :meth:`write_own`."""
        self.let_go(held.record_id)
        log.warning(
            "eval.job_lease %s=%s scope=%s: its lease passed to process %s at epoch %d (this holding: epoch %d) — "
            "stopping it",
            held.kind.name,
            held.record_id,
            held.scope_id,
            lease.holder,
            lease.epoch,
            held.epoch,
        )
        if not ending:
            self._on_loop(held.stop)

    def _fence_lapsed(self, held: _Held) -> None:
        """Let go of a record whose lease its own write found lapsed by the store's time, write it lost as a renewal
        does, and stop its job on the loop — asked first, so a loss write that raises never keeps the job running.
        Called off the loop, from :meth:`write_own`."""
        self.let_go(held.record_id)
        log.warning(
            "eval.job_lease %s=%s scope=%s: its own write found its lease lapsed by the store's time — stopping it",
            held.kind.name,
            held.record_id,
            held.scope_id,
        )
        self._on_loop(held.stop)  # first: the loss write below may raise, and the job must stop regardless
        try:
            settle_job_record(self._store, held.kind, held.record_id, held.scope_id)
        except (ConflictError, StorageError) as unsettled:
            # The job is let go of and stopped regardless, and its lapsed lease reads lost to every reader, the first
            # of which writes the loss this write could not.
            contain(
                unsettled,
                log=log,
                what=(
                    f"eval.job_lease {held.kind.name}={held.record_id} scope={held.scope_id}: the loss could not be "
                    "written here; the next reader writes it"
                ),
                level="warning",
            )

    def _on_loop(self, callback: Callable[..., object], *args: object) -> None:
        """Run ``callback(*args)`` on the loop the leases are held on: now when called there, else as soon as it can."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return  # nothing is held on a loop that is gone: there is nothing to keep or stop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            callback(*args)
            return
        try:
            loop.call_soon_threadsafe(callback, *args)
        except RuntimeError as closed:
            # The loop closed between the check above and the call: a process shutting down. Dropping the callback
            # is safe — it only counts a lease renewed (left counted older than it is) or stops a job the shutdown is
            # stopping — and the write that posted it has landed, so it must not read as failed.
            contain(
                closed, log=log, what="eval.job_lease bookkeeping dropped: the event loop had closed", level="warning"
            )

    def _renewed_by_own_write(self, held: _Held, issued: float, asked: tuple[str | None] | None) -> None:
        """Count a landed extending own write as a renewal, on the loop: from when it was issued, and with the cancel
        asked of the job, if any, handed to it once."""
        if self._held.get(held.record_id) is not held:
            return  # released or let go since: its job has ended, or no longer holds it
        self._renewed(held, issued)
        if asked is not None:
            self._act_on(held, "cancel", asked[0], self.renew_every_s)

    @staticmethod
    def _renewed(held: _Held, issued: float) -> None:
        """Reckon ``held``'s lease extended by a write issued at ``issued``, clearing the last renewal failure."""
        held.renewed_at = max(held.renewed_at, issued)
        held.last_failure = None

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
        epoch: int = 0,
    ) -> None:
        """Start renewing a record's lease — its first write carried one from :meth:`grant` — until it is released.

        Called on the event loop, which the renewal task runs on.

        :param kind: The record's type.
        :ptype kind: LeasedJobKind[Any]
        :param record_id: The record — also the job's id in the manager.
        :ptype record_id: str
        :param scope_id: Its scope.
        :ptype scope_id: str
        :param cancel: Ends the job ``cancelled``, given the reason asked; called once, when a renewal finds a cancel
            asked of it.
        :ptype cancel: Callable[[str | None], object]
        :param stop: Stops the job at once; called when a renewal finds its record ended by another or its lease
            lapsed or passed to another holding, after which the job's writes are refused.
        :ptype stop: Callable[[], object]
        :param epoch: The epoch of the lease :meth:`grant` gave the record's write; a stored lease at another epoch has
            passed from this holding.
        :ptype epoch: int
        """
        self._held[record_id] = _Held(
            kind=kind, record_id=record_id, scope_id=scope_id, cancel=cancel, stop=stop, epoch=epoch
        )
        self._loop = asyncio.get_running_loop()
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

        :param record_id: The record.
        :ptype record_id: str
        """
        self._held.pop(record_id, None)
        self._let_go.add(record_id)

    def fenced(self, record_id: str) -> bool:
        """Whether this process let go of ``record_id``'s lease (:meth:`let_go`) and its job has not yet ended.

        A held lease whose last successful renewal is a whole :attr:`lease_s` old is fenced too, before its own
        renewal task gets to :meth:`_expire` it: each record renews in a task of its own, so a job stopped by
        another record's expiry (a launch cancelling its runs) would otherwise write its own ending past a lease
        that has lapsed by this process's count.
        """
        if record_id in self._let_go:
            return True
        held = self._held.get(record_id)
        return held is not None and time.monotonic() - held.renewed_at >= self._lease_s

    def forget(self, record_id: str) -> None:
        """Forget a record this process let go of: its job has ended, and writes nothing more. Idempotent."""
        self._let_go.discard(record_id)

    async def _renew_while_held(self) -> None:
        """Renew every held lease each interval, each in a task of its own, until none is held.

        A record whose renewal is still running (a slow round-trip) is not started again, and delays no other. A
        renewal is itself given only what is left of its lease (:meth:`_renew_one`), so one that never returns still
        ends, and its record is expired.
        """
        renewing: dict[str, asyncio.Task[None]] = {}
        try:
            while self._held:
                await asyncio.sleep(self.renew_every_s)
                for record_id, task in list(renewing.items()):
                    if task.done():
                        del renewing[record_id]
                        self._log_renewal_fault(record_id, task)
                due = time.monotonic()
                for record_id, held in list(self._held.items()):
                    if record_id not in renewing:
                        renewing[record_id] = asyncio.create_task(
                            self._renew_one(record_id, held, due), name=f"eval-job-lease-{record_id}"
                        )
        finally:
            for task in renewing.values():
                task.cancel()
            self._close_pool()

    @staticmethod
    def _log_renewal_fault(record_id: str, task: asyncio.Task[None]) -> None:
        """Log a renewal task that ended by raising, which nothing else awaits."""
        if not task.cancelled() and task.exception() is not None:
            # NOSILENT: logged here; the next interval starts the record's renewal again
            log.error("eval.job_lease renewal of %s raised", record_id, exc_info=task.exception())

    async def _renew_one(self, record_id: str, held: _Held, due: float) -> None:
        """Renew one held lease, due at ``due`` (monotonic), and act on what the renewal found.

        A renewal that lost every conditional write to the holder's own writes is tried again after a short jittered
        wait, a few times, before it is left to the next interval. **The store call is bounded by what is left of the
        lease** by this holder's own count: one that has not returned by then counts as a failed renewal for a whole
        lease and expires the record (:meth:`_expire`) whether or not the call ever returns, its fault naming the
        hung renewal — unless the job's own writes extended the lease while it hung (:meth:`write_own`). The call's
        worker thread is left to finish on its own, and **no second renewal of the record is sent while it is stuck**.
        The pool it is stuck in is retired (:meth:`_close_pool`): it takes no new work, and the next renewal of any
        record makes a fresh one, so a stuck call costs one abandoned thread, never a slot of the live pool, however
        many hangs a long-lived process meets. While the call stays stuck, the record is kept by its job's own writes,
        each interval logging that it is stuck; with none, it is expired once the lease is a whole lease unextended.
        """
        if held.stuck_call is not None:
            if not held.stuck_call.done():
                await self._while_stuck(held)
                return
            held.stuck_call = None
        for round_ in range(CONTENDED_RENEWAL_RETRIES + 1):
            issued = time.monotonic()
            started = issued
            verdict: RenewalVerdict | None = None
            reason: str | None = None
            # The store call gets only what is left of the lease by this holder's own count: a call that never
            # returns must not keep the record from being expired (_expire), as the check below only runs once it has.
            hung = False
            pool: ThreadPoolExecutor | None = None
            call: Future[tuple[float, RenewalVerdict, str | None]] | None = None
            left = held.renewed_at + self._lease_s - time.monotonic()
            try:
                if (
                    left > 0
                ):  # none left: the lease is already unrenewed for a whole lease, and the last failure says why
                    pool = self._pool()
                    call = pool.submit(contextvars.copy_context().run, self._renew, held)
                    async with asyncio.timeout(left) as deadline:
                        renewal = await asyncio.wrap_future(call)
                    started, verdict, reason = renewal
            # prawduct:ok-broad-except — a failed renewal is retried next interval; the lease outlives two failures
            except Exception as error:
                hung = deadline.expired()
                if hung and call is not None and not call.done():
                    held.stuck_call, held.stuck_since = call, issued
                    if self._executor is pool:
                        self._close_pool()  # retired: the stuck thread is abandoned with it, the next need makes another
                # The store's own error when it raised one; a hung call is named as such, not as the timeout's
                # cancellation. Only a renewal that lost every write (a "contended" verdict, never an exception) is
                # a lost race.
                failed: Exception = (
                    TimeoutError(
                        f"the renewal's store call had not returned in the {self._lease_s:g}s of the lease it was renewing"
                    )
                    if hung
                    else error
                )
                # NOSILENT: retried next interval; the verdict stays None, so failures lasting a whole lease stop
                # the job as lost (_expire), with this failure written on its record as a fault.
                # Not written on the record here: the renewal IS this process's write of the record, and the
                # store just failed it. Kept on the hold instead, and written on the record if the failures go
                # on to cost the lease (_expire); one a later renewal recovers from cost the job nothing.
                held.last_failure = f"{type(failed).__name__}: {failed}"
                contain(
                    failed,
                    log=log,
                    what=(
                        f"eval.job_lease renewal failed {held.kind.name}={record_id} scope={held.scope_id} "
                        f"— retried in {self.renew_every_s:.0f}s"
                    ),
                    level="error",
                )
            if self._held.get(record_id) is not held:
                return  # released while the renewal ran: its job wrote its own ending
            if verdict in ("renewed", "cancel"):
                self._log_if_late(held, due, issued, started, time.monotonic())
                self._renewed(held, issued)
            elif verdict in (None, "contended") and time.monotonic() - held.renewed_at >= self._lease_s:
                # Not renewed — failed, hung, or every write lost its race — for a whole lease. A hung call's deadline
                # was the lease as it stood when the call began; the job's own writes may have extended it since
                # (write_own), and then the lease is not lost, and the next interval renews it again.
                await self._expire(held)
                return
            wait = 0.0
            if verdict == "contended" and round_ < CONTENDED_RENEWAL_RETRIES:
                low, high = CONTENDED_BACKOFF_S
                cap = self.renew_every_s / 4
                wait = random.uniform(min(low, cap / 2), min(high, cap))
            if verdict is not None:
                self._act_on(held, verdict, reason, wait or self.renew_every_s)
            if not wait:
                return
            await asyncio.sleep(wait)
            due = time.monotonic()
            if self._held.get(record_id) is not held:
                return

    async def _while_stuck(self, held: _Held) -> None:
        """A renewal of ``held`` is due while its last one's store call is still stuck: send none. Expire the record once
        its lease is a whole lease unextended, as any record no renewal reaches; otherwise its job's own writes are
        keeping it, and the stuck call is logged."""
        now = time.monotonic()
        stuck = TimeoutError(
            f"the renewal's store call has not returned in {now - held.stuck_since:.1f}s; no second renewal is sent "
            "while it is stuck, and its lease-pool thread was abandoned with the pool it ran on"
        )
        if now - held.renewed_at >= self._lease_s:
            held.last_failure = f"{type(stuck).__name__}: {stuck}"
            await self._expire(held)
            return
        # NOSILENT: the job's own writes are keeping the lease (write_own); this line names the call still stuck.
        contain(
            stuck,
            log=log,
            what=(
                f"eval.job_lease {held.kind.name}={held.record_id} scope={held.scope_id}: renewal stuck — kept by "
                "the job's own writes"
            ),
            level="warning",
        )

    def _log_if_late(self, held: _Held, due: float, issued: float, started: float, landed: float) -> None:
        """Log a renewal landing more than one and a half intervals after the lease was last extended.

        The line separates the ways a renewal runs late: the loop's wait (from when it was due to when the loop sent
        it) and the pool's wait (from then to when a lease-pool thread started it) are a starved or crowded process;
        the store call (from then until it returned) is a slow store.
        """
        since = landed - held.renewed_at
        if since <= 1.5 * self.renew_every_s:
            return
        log.warning(
            "eval.job_lease %s=%s scope=%s renewal landed late: %.1fs since the lease was last extended, against a "
            "%.1fs renewal interval and a %.1fs lease; the loop's wait %.2fs, the lease pool's wait %.2fs, the store "
            "call %.2fs",
            held.kind.name,
            held.record_id,
            held.scope_id,
            since,
            self.renew_every_s,
            self._lease_s,
            issued - due,
            started - issued,
            landed - started,
        )

    async def _expire(self, held: _Held) -> None:
        """Stop a job whose renewals have failed for a whole lease: its lease has lapsed by this process's own count.

        Every reader may already read it lost, and the next one to look writes so — so the holder stops at once
        rather than spend past an expiry it cannot extend. It writes the loss itself when it can (the store may be
        what failed); either way the job's own writes are refused from here on (:meth:`let_go`), even once the
        store recovers.

        **Why it stopped is written on the record**, whatever its kind: a ``faults`` entry naming the last renewal's
        error (or that every renewal lost its race), so the loss a reader writes later reads with its cause beside
        it, rather than as a process that simply went away. Written before the loss, conditionally and without
        moving the lease, so a store refusing only renewals takes it — and in ``faults`` rather than the record's
        ending (a run's ``error_details`` included), because the loss writes a sweep's ``detail`` or an attempt's
        ``error`` over, and a run's own faults are its work's, not its lease's. A write that fails is logged with
        both errors.
        """
        self.let_go(held.record_id)
        why = (
            f"the last renewal failed: {held.last_failure}"
            if held.last_failure is not None
            else f"every renewal lost its race to another writer, {LEASE_WRITE_ATTEMPTS} times each"
        )
        log.error(
            "eval.job_lease %s=%s scope=%s could not be renewed for %.0fs, its whole lease — stopping it as lost (%s)",
            held.kind.name,
            held.record_id,
            held.scope_id,
            self._lease_s,
            why,
        )
        entry = (
            f"job lease: its holder, process {self._holder_id}, could not renew it for {self._lease_s:g}s, its "
            f"whole lease, and stopped the job — {why}"
        )
        try:
            async with asyncio.timeout(self._lease_s):  # a hung store must not keep the job from stopping
                await run_blocking(
                    self._pool(), _note_fault, self._store, held.kind, held.record_id, held.scope_id, entry
                )
        # prawduct:ok-broad-except — the store may be what failed; logged with the renewal failure it was to record
        except Exception as unwritten:
            # NOSILENT: the job is still fenced and stopped below and its record still reads lost; only the cause it
            # was to carry is missing, because the store that would hold it is what failed, so it is logged whole.
            contain(
                unwritten,
                log=log,
                what=(
                    f"eval.job_lease {held.kind.name}={held.record_id} scope={held.scope_id}: why it was stopped "
                    f"could not be written on its record either; the renewal failure: {why}"
                ),
                level="error",
            )
        try:
            async with asyncio.timeout(self._lease_s):  # likewise: the stop below never waits on the store
                await run_blocking(
                    self._pool(), settle_job_record, self._store, held.kind, held.record_id, held.scope_id
                )
        # prawduct:ok-broad-except — the store may be what failed; the record reads lost to the next reader regardless
        except Exception as unsettled:
            # NOSILENT: the job is fenced and stopped below regardless, and its lapsed lease reads lost to every
            # reader, the first of which writes the loss this write could not.
            contain(
                unsettled,
                log=log,
                what=(
                    f"eval.job_lease {held.kind.name}={held.record_id} scope={held.scope_id}: the loss could not be "
                    f"written here; the next reader writes it. The renewal failure: {why}"
                ),
                level="error",
            )
        held.stop()

    async def end_lost(self, record_id: str, cause: str) -> bool:
        """Stop renewing a held record and write it lost with ``cause``, unless it has already ended.

        For the job manager's task boundary: a task whose work raised past its own ending leaves its record
        unended, and released, it would read lost only once its lease lapsed — with a cause saying its process went
        away, when the process knows exactly why. Written as a shutdown writes it (the record's own lost ending,
        no lease, conditionally), so a record the work did end keeps that ending.

        :param record_id: The record — the job's id in the manager.
        :ptype record_id: str
        :param cause: The loss's cause, in a poll's words.
        :ptype cause: str
        :return: Whether it was written; ``False`` for a record this process does not hold, or one already ended.
        :rtype: bool
        :raises ConflictError: The record kept changing while it was being written.
        :raises StorageError: A read or write failed.
        """
        held = self._held.pop(record_id, None)
        if held is None:
            return False
        return await run_blocking(self._pool(), self._end_lost, held, cause)

    def _act_on(self, held: _Held, verdict: RenewalVerdict, reason: str | None, retry_in_s: float) -> None:
        """Do what a renewal found: deliver a cancel once, or release and stop a job fenced off its record.

        ``retry_in_s`` is the true wait before a contended renewal is tried again.
        """
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
                "eval.job_lease %s=%s scope=%s renewal lost %d races — tried again in %.1fs",
                held.kind.name,
                held.record_id,
                held.scope_id,
                LEASE_WRITE_ATTEMPTS,
                retry_in_s,
            )

    def _renew(self, held: _Held) -> tuple[float, RenewalVerdict, str | None]:
        """Renew one held lease: the blocking half of a renewal, one worker hop.

        :return: When the worker started it (:func:`time.monotonic`), the verdict, and the cancel's reason when one was
            asked.
        :rtype: tuple[float, RenewalVerdict, str | None]
        """
        started = time.monotonic()
        return (started, *self._renewal_round_trips(held))

    def _renewal_round_trips(self, held: _Held) -> tuple[RenewalVerdict, str | None]:
        """A renewal's reads and conditional writes: the verdict, and the cancel's reason when one was asked."""
        for _ in range(LEASE_WRITE_ATTEMPTS):
            record, etag = self._store.load_job_record(held.kind.model, held.record_id, held.scope_id)
            if record is None:
                return "fenced", None
            leased = _leased(record)
            if leased.lost is not None:
                return "fenced", None
            if held.kind.ended(record):
                return "ended", None
            if leased.lease is None or _passed(leased.lease, self._holder_id, held.epoch):
                return "fenced", None
            extended = self._extended(leased, self._store.store_time())
            if extended is None:
                # This process's lease, so lapsed: past its own expiry the lease is gone, whoever has noticed. Written
                # lost here as any reader would, so this holder's later writes are refused by the record rather than
                # by its own memory.
                settle_job_record(self._store, held.kind, held.record_id, held.scope_id)
                return "fenced", None
            renewed = record.model_copy(update={"lease": extended})
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

        :param cause: The loss's cause.
        :ptype cause: str
        :return: The records written lost; one already ended, or whose write failed (logged), is not among them.
        :rtype: list[str]
        """
        if self._renewer is not None:
            self._renewer.cancel()
            self._renewer = None
        held, self._held = list(self._held.values()), {}
        ended: list[str] = []
        for one in held:
            try:
                if await run_blocking(self._pool(), self._end_lost, one, cause):
                    ended.append(one.record_id)
            # prawduct:ok-broad-except — one record's failed write must not keep the rest of a shutdown from ending theirs
            except Exception as unwritten_loss:
                # NOSILENT: the record is left out of the records this returns as written lost, which is true; the
                # failed write IS the write of this record, so nothing is left to carry the error, and its job, not
                # fenced, writes its own ending as it unwinds — or it reads lost once its lease lapses.
                contain(
                    unwritten_loss,
                    log=log,
                    what=(
                        f"eval.job_lease shutdown could not write {one.kind.name}={one.record_id} "
                        f"scope={one.scope_id} lost — it reads lost once its lease lapses"
                    ),
                    level="error",
                )
        self._close_pool()
        return ended

    def _end_lost(self, held: _Held, cause: str) -> bool:
        """Write one held record lost with ``cause`` and no lease, unless it has ended or its lease has passed to another
        holding; whether it was written.

        A record whose lease has passed is another holding's to end, so nothing is written, and this process lets it
        go (:meth:`let_go`): its job, unwinding after this, must not write its own ending, its completeness or anything
        else over the newer holding's record, and no loss on the record will refuse those writes for it.
        """
        for _ in range(LEASE_WRITE_ATTEMPTS):
            record, etag = self._store.load_job_record(held.kind.model, held.record_id, held.scope_id)
            if record is None or held.kind.ended(record):
                return False
            if _passed(_leased(record).lease, self._holder_id, held.epoch):
                self.let_go(held.record_id)
                return False
            now = self._store.store_time()
            # The lease is released with the loss, which keeps the epoch it was held at: a take-up holds a higher one.
            loss = JobLoss(at=now.isoformat(), cause=cause, epoch=held.epoch)
            lost = held.kind.end_lost(record, loss, self._store)
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
    "LEASE_WRITE_ATTEMPTS",
    "RUN_JOB",
    "SHUTDOWN_CAUSE",
    "SWEEP_JOB",
    "JobLeases",
    "LeasedJobKind",
    "RenewalVerdict",
    "lease_lapsed",
    "read_to_take_up",
    "reopened_job_fields",
    "request_job_cancel",
    "require_held",
    "settle_job_record",
    "taken_up_meanwhile",
    "write_held_record",
    "write_reopened",
]
