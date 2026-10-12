"""One operation's work as a job: its record, the awaitable that runs it to its end, and the two ways to run it.

No transport call waits on a model. An operation that calls one — a launch generating its cases, a second judge's
pass, a judge temperature comparison, a host's own drafting action — is split where its spending starts: the CALL
prices it, checks its cap and admits it, and refuses there with nothing spent; the WORK makes the calls. The work
is a :class:`JobWork`: the record it writes (running as it starts, and its ending however it ends), and one
awaitable that runs to that ending. It is run either way, and writes the same record either way:

- **as a job** (:func:`start_job_work`): the record is written carrying a lease from the job manager's
  :class:`~threetears.evals.kernel.job_leases.JobLeases`, and the work runs as a background task the manager
  tracks — the form a transport uses, which returns at once with a job to poll
  (:func:`~threetears.evals.ops.start_job`);
- **in the foreground** (:func:`run_job_work`): the record is written and held the same way, and the caller awaits
  the work to its end in its own frame — the form a short-lived consumer uses (a script, the command line), which
  loses nothing to the job contract: it holds the result when the await returns.

**The record is the job's answer.** Every record here declares the four lease fields
(:mod:`~threetears.evals.schema.job_lease`) and the same three of its own — ``outcome`` (``running``, then
``completed``, ``failed``, ``cancelled`` or ``timed_out``), ``detail`` (why it ended other than completed) and
``ended_at`` — so one
:class:`~threetears.evals.kernel.job_leases.LeasedJobKind` shape serves them all (:func:`leased_job_kind`): a record
whose lease lapses is written lost, as a run, a sweep and an analysis attempt are. The work's own writes go through
:class:`JobRecorder`, which refuses once the record has ended elsewhere
(:func:`~threetears.evals.kernel.job_leases.write_held_record`), so a holder that lost its lease writes nothing more.

**A job with no domain record** — a host action that produces a draft, not a run — runs as an
:class:`EvalActionJob` (:func:`action_job_work`): the generic record carrying the lease, the outcome and the
action's typed result as JSON, which polling the job returns.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from threetears.evals.kernel.errors import ConflictError, EvalServiceError, JobFenced, StorageError
from threetears.evals.kernel.refusals import job_refusal
from threetears.evals.kernel.job_leases import JobLeases, LeasedJobKind, write_held_record
from threetears.evals.kernel.offload import run_blocking, wait_through_cancellation
from threetears.evals.kernel.storage import JobRecordStore, LeaseStore
from threetears.evals.kernel.job_timeout import EvalJobTimeout, JobTimeoutFactory, timed_out_detail
from threetears.evals.run.jobs import EvalJobManager
from threetears.evals.schema.base import CoreDocumentModel, EvalDocumentModel
from threetears.evals.schema.job_lease import (
    CancelRequestedAtField,
    CancelRequestReasonField,
    FaultsField,
    JobLoss,
    LeaseField,
    LostField,
)
from threetears.evals.schema.refusal import RefusalField
from threetears.evals.schema.models import CoreSchemaVersion, utc_now_iso
from threetears.evals.schema.versioning import CORE_SCHEMA_VERSION
from threetears.evals.schema.logs import get_logger

log = get_logger(__name__)

#: Where a job's record stands: ``running`` (started, not ended), or how it ended — ``completed`` (its work ran to
#: its end), ``failed`` (refused or broken part-way; ``detail`` says which), ``cancelled`` (asked to stop, shut
#: down, or lost — which the record's ``lost`` says) or ``timed_out`` (its own wall-clock budget ran out; ``detail``
#: names it). A budget is not a person deciding to stop, so the two are never one outcome.
JobOutcome = Literal["running", "completed", "failed", "cancelled", "timed_out"]

#: An action job's wall-clock ceiling unless its host names another: an hour, the job manager's own default.
DEFAULT_ACTION_JOB_BUDGET_S = 3600.0

#: The detail a job cancelled before it ended records, beside the cancel's reason on the record.
CANCELLED_DETAIL = "cancelled before it ended"

#: How long a job waits before each further attempt at its ending write when the store fails it, in seconds. The
#: ending carries the job's result, and a job whose ending is never written reads lost once its lease lapses, so a
#: store that fails for a moment is waited out rather than costing the result; the lease is held meanwhile.
ENDING_WRITE_RETRY_DELAYS_S: tuple[float, ...] = (0.5, 2.0, 5.0)


def end_record[R: EvalDocumentModel](record: R, outcome: JobOutcome, detail: str | None = None, **fields: Any) -> R:
    """``record`` as it ends: its outcome, why (unless it completed), when — and any other field the ending sets.

    Args:
        record: A job record of this module's shape.
        outcome: How it ended.
        detail: Why it ended other than completed; ``None`` for a completion.
        **fields: The ending's own fields — a report, a result, the runs a launch started.

    Returns:
        A copy carrying the ending.
    """
    return record.model_copy(update={"outcome": outcome, "detail": detail, "ended_at": utc_now_iso(), **fields})


def _written_lost[R: EvalDocumentModel](record: R, loss: JobLoss, _store: LeaseStore) -> R:
    """A record written lost: ``cancelled`` — it stopped without finishing — its detail the loss, and the loss recorded."""
    return record.model_copy(
        update={"outcome": "cancelled", "detail": f"lost: {loss.cause}", "ended_at": loss.at, "lost": loss}
    )


def leased_job_kind[R: EvalDocumentModel](
    name: str, model: type[R], *, named_runs: Callable[[R], Sequence[str]] | None = None
) -> LeasedJobKind[R]:
    """The lease machinery's reading of a record of this module's shape: ended once its outcome is not ``running``.

    Args:
        name: The job kind's name, for a message.
        model: The record's model, declaring ``outcome``, ``detail``, ``ended_at`` and the four lease fields.
        named_runs: The runs the record names as its own, settled with it when it is written lost; ``None`` for none.

    Returns:
        The kind, registered by its model so every store read of a record of it settles it.
    """
    return LeasedJobKind(
        name=name,
        model=model,
        ended=lambda record: getattr(record, "outcome") != "running",
        end_lost=_written_lost,
        **({"named_runs": named_runs} if named_runs is not None else {}),
    )


class JobRecorder[R: EvalDocumentModel]:
    """A job's own writes of its record while its work runs: each one guarded, and the latest one kept.

    The work writes progress through :meth:`save` (a launch its runs as they start); the job runner writes the
    ending through it too, starting from :attr:`record` — the newest copy written — so an ending never drops what
    the work recorded on the way.
    """

    def __init__(
        self,
        store: JobRecordStore,
        executor: Executor | None,
        kind: LeasedJobKind[R],
        record: R,
        *,
        leases: JobLeases | None,
    ) -> None:
        """Bind the writes to the record's store.

        Args:
            store: Where the record lives.
            executor: Where the store round-trips run; ``None`` for the loop's default executor.
            kind: The record's type.
            record: The record as it was started.
            leases: The leases the record is held under: a record they have let go of takes no write from here.
        """
        self._store = store
        self._executor = executor
        self._kind = kind
        self._record = record
        self._leases = leases

    @property
    def kind(self) -> LeasedJobKind[R]:
        """The record's type."""
        return self._kind

    @property
    def record(self) -> R:
        """The record as last written."""
        return self._record

    async def save(self, record: R) -> R:
        """Write ``record`` as the job's own write, carrying the stored lease and cancel fields.

        Args:
            record: The record as the job would write it.

        Returns:
            The record as written, which :attr:`record` now is.

        Raises:
            JobFenced: The stored record has ended — written lost, by a reader that found the lease lapsed or by a
                shutdown — or this process let go of its lease, so the job writes nothing more.
            ConflictError: Every attempt lost its race.
            StorageError: A read or write failed.
        """

        def write() -> R:
            return write_held_record(self._store, self._kind, record, leases=self._leases)

        written = await run_blocking(self._executor, write)
        self._record = written
        return written


@dataclass(frozen=True, kw_only=True)
class JobWork[R: EvalDocumentModel]:
    """One operation admitted and not yet run: its record as it starts, and the awaitable that runs it to its end.

    Built by the operation's admission — the half that prices, checks the cap and refuses — so whatever holds one
    holds work that will be paid for only when it runs. **It runs once**: what admission priced and admitted is spent
    by one run, so a second start of the same work is refused before anything is written (as
    :class:`~threetears.evals.run.AdmittedLaunch` refuses a second start), and its record's first write is
    create-only, so no run writes over another's.

    Attributes:
        kind: The record's type, as the lease machinery reads it.
        record: The record as the job starts: ``outcome`` ``running``, no lease yet (the runner grants it).
        run: Runs the work to its end and returns the record as it ends (:func:`end_record`), writing any progress
            through the recorder on the way. What it raises ends the job ``failed`` with the error as its detail —
            and, for an engine refusal, its kind and status as the record's ``refusal`` — a cancel ends it
            ``cancelled``, and a fenced write ends it as the record already says.
        budget_s: The work's wall-clock ceiling when it runs as a job: when it runs out, the work is stopped and the
            record ends ``timed_out``. ``None`` for work that bounds what it does itself — a launch bounds its
            preparation and leaves its runs to their own budgets, so time its runs spend queued for a slot is never
            counted against it.
        discard: Releases what admission built (reserved room, admitted clients) when the work never runs: its
            start refused, or its task cancelled before its first step. ``None`` when admission holds nothing.
    """

    kind: LeasedJobKind[R]
    record: R
    run: Callable[[JobRecorder[R]], Awaitable[R]]
    budget_s: float | None
    discard: Callable[[], Awaitable[None]] | None = None
    _used: list[bool] = field(default_factory=list, init=False, repr=False, compare=False)

    def claim(self) -> None:
        """Take the work's one run: what :func:`start_job_work` and :func:`run_job_work` do before they write anything.

        Raises:
            RuntimeError: It was already started — the caller holds work its admission priced for one run.
        """
        if self._used:
            raise RuntimeError(f"{self.kind.name} job {self.record_id!r} was already started; admitted work runs once")
        self._used.append(True)

    @property
    def record_id(self) -> str:
        """The record's id — the job's id in the job manager."""
        return str(getattr(self.record, "id"))

    @property
    def scope_id(self) -> str:
        """The scope the record lives in."""
        return str(getattr(self.record, "scope_id"))


def _failure_detail(broke: BaseException) -> str:
    """Why a job's work failed, in a poll's words: an engine refusal as it was said, anything else by its class too."""
    if isinstance(broke, EvalServiceError):
        return str(broke)
    return f"{type(broke).__name__}: {broke}"


async def _run_to_end[R: EvalDocumentModel](
    work: JobWork[R], recorder: JobRecorder[R], *, deadline: JobTimeoutFactory | None
) -> R:
    """Run ``work`` and write its ending — completed, failed, cancelled or timed out — whatever the work did.

    Args:
        work: The admitted work.
        recorder: Its record's writes.
        deadline: What bounds the work by its ``budget_s`` — entered here, around the work alone, so a budget that
            fires is caught here and the record says ``timed_out``; a deadline entered around this whole frame
            would reach it as a cancel. ``None`` runs the work unbounded (the foreground form).

    Returns:
        The record as it ended; as last written when the record was fenced off (written lost elsewhere).
    """
    record_id, scope_id, name = work.record_id, work.scope_id, work.kind.name
    try:
        if deadline is None or work.budget_s is None:
            ended = await work.run(recorder)
        else:
            async with deadline(work.budget_s):
                ended = await work.run(recorder)
    except EvalJobTimeout as timed_out:
        log.warning("eval.job %s=%s scope=%s timed out: %s", name, record_id, scope_id, timed_out)
        ended = end_record(recorder.record, "timed_out", timed_out_detail(timed_out))
    except asyncio.CancelledError:
        # The ending is written to its end even through a second cancel, then the cancel is re-raised.
        writing = asyncio.ensure_future(recorder.save(end_record(recorder.record, "cancelled", CANCELLED_DETAIL)))
        await wait_through_cancellation(writing)
        if (fenced := writing.exception()) is not None and not isinstance(fenced, JobFenced):
            log.error(
                "eval.job %s=%s scope=%s: its cancelled ending was not written: %s", name, record_id, scope_id, fenced
            )
        raise
    except JobFenced as fenced:
        log.warning("eval.job %s=%s scope=%s stopped: %s", name, record_id, scope_id, fenced)
        return recorder.record
    # prawduct:ok-broad-except — the job's record is where its ending is stated; the boundary writes it failed
    except Exception as broke:
        log.exception("eval.job %s=%s scope=%s failed", name, record_id, scope_id)
        # An engine refusal is kept as data beside its words, so a poll serves its kind as a refused call's.
        refusal = job_refusal(broke) if isinstance(broke, EvalServiceError) else None
        ended = end_record(recorder.record, "failed", _failure_detail(broke), refusal=refusal)
    try:
        return await _write_ending(recorder, ended, name=name, record_id=record_id, scope_id=scope_id)
    except JobFenced as fenced:
        log.warning(
            "eval.job %s=%s scope=%s ended, but its record was lost first: %s", name, record_id, scope_id, fenced
        )
        return recorder.record


async def _write_ending[R: EvalDocumentModel](
    recorder: JobRecorder[R], ended: R, *, name: str, record_id: str, scope_id: str
) -> R:
    """Write a job's ending, waiting out a store that fails it; one never written is logged whole, never dropped.

    The ending is where the job's result is kept, so a failed write is tried again after each of
    :data:`ENDING_WRITE_RETRY_DELAYS_S`, the lease held meanwhile. A write still failing after the last — or a
    cancel arriving while it waits — leaves the record to read lost once its lease lapses, and the ending it would
    have written is logged at ERROR in full, so what the job paid for can be recovered from the log.

    Returns:
        The record as written; ``ended`` itself when no attempt was, so a foreground caller still holds the result.

    Raises:
        JobFenced: The record ended elsewhere first; nothing was written.
        asyncio.CancelledError: Cancelled while waiting to try again; the ending was logged.
    """
    delays = iter(ENDING_WRITE_RETRY_DELAYS_S)
    attempt = 0
    while True:
        attempt += 1
        try:
            return await recorder.save(ended)
        except JobFenced:
            raise
        except (ConflictError, StorageError) as failed:
            delay = next(delays, None)
            if delay is None:
                log.error(
                    "eval.job %s=%s scope=%s: its ending was NOT written after %d attempts (%s); the record reads lost "
                    "once its lease lapses. The ending it would have written: %s",
                    name,
                    record_id,
                    scope_id,
                    attempt,
                    failed,
                    ended.model_dump_json(),
                )
                return ended
            log.warning(
                "eval.job %s=%s scope=%s: its ending write failed (%s) — tried again in %.1fs",
                name,
                record_id,
                scope_id,
                failed,
                delay,
            )
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            log.error(
                "eval.job %s=%s scope=%s: cancelled while its failed ending write waited to be tried again; the "
                "ending it would have written: %s",
                name,
                record_id,
                scope_id,
                ended.model_dump_json(),
            )
            raise


async def _save_started[R: EvalDocumentModel](
    work: JobWork[R], store: JobRecordStore, executor: Executor | None, leases: JobLeases
) -> R:
    """Write the record as it starts, carrying a fresh lease, so its first poll in any process finds it running.

    Create-only: a record already stored under the id is a job already started, and is never written over.
    """

    def save() -> R:
        started = work.record.model_copy(update={"lease": leases.grant()})
        store.save_job_record(started.to_dict(), if_match=None, create_only=True)
        return started

    return await run_blocking(executor, save)


async def start_job_work[R: EvalDocumentModel](
    manager: EvalJobManager,
    work: JobWork[R],
    *,
    store: JobRecordStore,
    executor: Executor | None,
    key: str | None = None,
) -> R:
    """Write the work's record, leased, and run the work as a background job the manager tracks.

    Args:
        manager: The process's job manager; it must hold leases.
        work: The admitted work.
        store: Where the record lives — the store the manager's leases renew it in.
        executor: Where the store round-trips run.
        key: An exclusivity key: while a job holding it is live, this start is refused. ``None`` for none.

    Returns:
        The record as started.

    Raises:
        ValueError: The manager holds no leases — its job could be read lost by every other process at once.
        RuntimeError: The work was already started; nothing was written.
        ConflictError: A live job holds ``key``; the record is written ``failed`` and admission's holdings released.
            Or a record is already stored under the work's id; it is left as it is.
        StorageError: The record could not be written; nothing started.
    """
    leases = manager.leases
    if leases is None:
        raise ValueError(f"{work.kind.name} job {work.record_id!r} runs as a job only under a manager holding leases")
    work.claim()
    try:
        started = await _save_started(work, store, executor, leases)
    except BaseException:
        if work.discard is not None:
            await work.discard()
        raise
    recorder = JobRecorder(store, executor, work.kind, started, leases=leases)

    async def run() -> None:
        await _run_to_end(work, recorder, deadline=manager.job_timeout_factory)

    try:
        manager.start_task(
            work.record_id,
            run,
            # The work enters its own budget (_run_to_end), so its record can say it timed out.
            budget_s=None,
            key=key,
            record=(work.kind, work.scope_id),
            discard=work.discard,
        )
    except ConflictError as refused:
        if work.discard is not None:
            await work.discard()
        await recorder.save(
            end_record(started, "failed", f"refused before it started: {refused}", refusal=job_refusal(refused))
        )
        raise
    return started


async def run_job_work[R: EvalDocumentModel](
    work: JobWork[R], *, store: JobRecordStore, executor: Executor | None, leases: JobLeases
) -> R:
    """Write the work's record, leased, and run the work to its end in the caller's own frame.

    The foreground form: the same record, written and held the same way, so a poll from any process reads the work
    while it runs, and a cancel asked on the record cancels the caller's await. Not bounded by the work's
    ``budget_s``, which bounds it as a job: the caller waits as long as the work takes, and stops it by cancelling.
    Not marked as a job's work
    (:func:`~threetears.evals.kernel.running_job` stays the caller's), because it is not one: it is for a caller
    that is not a transport — a script, the command line — and waits as long as the work takes.

    Args:
        work: The admitted work.
        store: Where the record lives.
        executor: Where the store round-trips run.
        leases: The leases the record is held under while the work runs.

    Returns:
        The record as it ended.

    Raises:
        asyncio.CancelledError: The caller was cancelled, or a cancel was asked on the record; it is written
            ``cancelled`` first.
        RuntimeError: The work was already started; nothing was written.
        ConflictError: A record is already stored under the work's id; it is left as it is, and nothing ran.
        StorageError: The record could not be written as it started; nothing ran.
    """
    work.claim()
    try:
        started = await _save_started(work, store, executor, leases)
    except BaseException:
        if work.discard is not None:
            await work.discard()
        raise
    caller = asyncio.current_task()
    assert caller is not None  # an awaited coroutine always runs in a task

    def stop(_reason: str | None = None) -> None:
        caller.cancel()

    leases.hold(work.kind, work.record_id, work.scope_id, cancel=stop, stop=stop)
    try:
        return await _run_to_end(work, JobRecorder(store, executor, work.kind, started, leases=leases), deadline=None)
    finally:
        leases.release(work.record_id)
        leases.forget(work.record_id)


# --- the generic record ---------------------------------------------------------------------------


class EvalActionJob(CoreDocumentModel):
    """A job whose work has no domain record of its own: what it was, how it ended, and the result it produced.

    A host action that drafts something — a rubric proposal, cases for a classifier — produces a value, not a run,
    and its work calls a model, so it runs as a job (:func:`action_job_work`) and this is its record: written as it
    starts (``running``, carrying the job's lease) and again as it ends. Polling the job returns :attr:`result` — the
    action's typed result as its JSON — once it has completed.

    **Kept** (:mod:`threetears.evals.schema.versioning`): it records what nothing stored can make again, so every
    build reads it, upgraded across core versions and never dropped.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_action_job"] = "eval_action_job"
    schema_version: CoreSchemaVersion = CORE_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    action: str = Field(min_length=1, description="The catalogue action that started the job, by name.")
    label: str = Field(default="", description="A short line naming the work, as the start returned it.")
    outcome: JobOutcome = "running"
    result: dict[str, Any] | None = Field(
        default=None,
        description="The action's typed result, as its model dumps to JSON; set exactly when the job completed.",
    )
    result_model: str | None = Field(
        default=None, description="The name of the result's model, so a reader knows which shape `result` is."
    )
    detail: str | None = Field(default=None, description="Why it ended other than completed.")
    created_at: str = Field(default_factory=utc_now_iso)
    ended_at: str | None = None
    lease: LeaseField = None
    cancel_requested_at: CancelRequestedAtField = None
    cancel_request_reason: CancelRequestReasonField = None
    lost: LostField = None
    faults: FaultsField = Field(default_factory=list)
    refusal: RefusalField = None

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_action_job":
            raise ValueError(f"doc_type must be 'eval_action_job', got '{v}'")
        return v


#: An action job's record.
ACTION_JOB: LeasedJobKind[EvalActionJob] = leased_job_kind("action", EvalActionJob)


def action_job_work(
    scope_id: str,
    *,
    action: str,
    run: Callable[[], Awaitable[BaseModel]],
    label: str = "",
    budget_s: float = DEFAULT_ACTION_JOB_BUDGET_S,
    discard: Callable[[], Awaitable[None]] | None = None,
) -> JobWork[EvalActionJob]:
    """A host action's work as a job: ``run`` awaited to its end, its result written on an :class:`EvalActionJob`.

    The action's handler makes its refusals and prices what it will spend first, in the call, and hands the
    spending half here as ``run``.

    Args:
        scope_id: The caller's scope, where the record lives.
        action: The action's name.
        run: The work: makes the calls and returns the action's typed result.
        label: A short line naming the work.
        budget_s: The work's wall-clock ceiling as a job.
        discard: Releases what the handler built for ``run`` when the work never runs.

    Returns:
        The work, its record running.

    Raises:
        ValueError: ``budget_s`` is not positive.
    """
    if not budget_s > 0:
        raise ValueError(f"an action job's budget_s must be positive; got {budget_s!r}")

    async def produce(recorder: JobRecorder[EvalActionJob]) -> EvalActionJob:
        result = await run()
        return end_record(
            recorder.record, "completed", result=result.model_dump(mode="json"), result_model=type(result).__name__
        )

    return JobWork(
        kind=ACTION_JOB,
        record=EvalActionJob(scope_id=scope_id, action=action, label=label),
        run=produce,
        budget_s=budget_s,
        discard=discard,
    )


__all__ = [
    "ACTION_JOB",
    "CANCELLED_DETAIL",
    "DEFAULT_ACTION_JOB_BUDGET_S",
    "ENDING_WRITE_RETRY_DELAYS_S",
    "EvalActionJob",
    "JobOutcome",
    "JobRecorder",
    "JobWork",
    "action_job_work",
    "end_record",
    "leased_job_kind",
    "run_job_work",
    "start_job_work",
]
