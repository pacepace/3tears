"""One job contract for every long-running operation: start, a job id, poll, cancel.

A launch, a sweep and an analysis generation all take minutes, and each already records its own ending
durably — a run its ``status``, a sweep its :class:`~threetears.evals.kernel.campaign.EvalSweep`, a generation its
:class:`~threetears.evals.kernel.campaign.EvalAnalysisAttempt`. So a job here is not a new record: it is a **name
for the record that says how the work stands**, and polling reads that record. Nothing about a job lives only in
one process's memory, so a job id handed to an agent is answered the same by every process sharing the store, and
after a restart.

**The job id names its record.** ``run:<run id>`` for a launched run (one per arm),
``analysis:<campaign id>:<attempt id>`` for a generation — the attempt is filed under its campaign, so the id
carries both — and ``sweep:<sweep id>`` for a sweep, whose record names its arms and their runs. :func:`parse_job_id`
is the one reader of that shape.

**Whether a job is still running is its record's answer, through its lease** (:mod:`threetears.evals.kernel.job_leases`).
The process running a job holds a lease on its record and renews it; a record that has not ended and whose lease
has lapsed — or that carries none, written by a build before leases — is ``lost``, and the first poll to see that
writes it on the record (:func:`~threetears.evals.kernel.job_leases.settle_job_record`), so every later poll, in any
process, reads the same. A **cancel** is written on the record too (``cancel_requested_at``), from whichever process
was asked; the process running the job reads it at its next renewal — at once when it is this one — and the job
ends ``cancelled``. A job is therefore cancelled from any process, and a cancel lands within one renewal interval.

**A loss bounds what the lost holder may still write, it does not erase what it measured.** Between another
process writing a run lost and the holder's next write of its own run (a progress tick, at most one cell later) or
its next renewal (at most a third of a lease later), the holder may still store results — at most the cells
in flight then, typically one. Each is a real measurement of a cell the run did execute, and a lost run keeps
what it measured; the run's status, completeness and ending are the loss's and are never overwritten.

**A job is answered only inside the caller's scope.** Its record is read scoped, so another scope's run, sweep or
generation reads absent — not found, on poll and cancel alike — and a generation's attempt is answered only under
the campaign it was filed under, so a caller typing another campaign into the id learns nothing.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from threetears.evals.schema.base import EvalBaseModel, EvalDocumentModel
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.campaign import EvalAnalysisAttempt, EvalSweep
from threetears.evals.schema.models import EvalRun
from threetears.evals.ops.host import OpsHost
from threetears.evals.kernel.job_leases import (
    ANALYSIS_JOB,
    RUN_JOB,
    SWEEP_JOB,
    LeasedJobKind,
    request_job_cancel,
    settle_job_record,
)
from threetears.evals.run.lifecycle import require_cancellable
from threetears.evals.kernel.offload import run_blocking

#: What a job's work is: a launched run, an analysis generation, or a sweep launching its arms in order.
JobKind = Literal["run", "analysis", "sweep"]

#: Where a job stands. ``running`` — still going (a run ``pending`` or ``running``, a generation live);
#: ``completed`` — ended with what it was for; ``stopped`` — a run ended short by its budget or its
#: account, keeping what it measured; ``failed``; ``cancelled``; ``lost`` — the process running it went
#: away before it ended: its lease lapsed, or its process shut down, and its record says so.
JobState = Literal["running", "completed", "stopped", "failed", "cancelled", "lost"]

#: The states a job does not leave.
TERMINAL_JOB_STATES: frozenset[str] = frozenset({"completed", "stopped", "failed", "cancelled", "lost"})

#: The prefix of a launched run's job id.
RUN_JOB_PREFIX = "run:"
#: The prefix of an analysis generation's job id.
ANALYSIS_JOB_PREFIX = "analysis:"

#: The prefix of a sweep's job id.
SWEEP_JOB_PREFIX = "sweep:"

#: A run's stored status, as the job state it is.
_RUN_STATES: dict[str, JobState] = {
    "pending": "running",
    "running": "running",
    "completed": "completed",
    "budget_stopped": "stopped",
    "exhausted": "stopped",
    "failed": "failed",
    "cancelled": "cancelled",
}

#: A generation attempt's recorded outcome, as the job state it is.
_ATTEMPT_STATES: dict[str, JobState] = {
    "stored": "completed",
    "refused": "failed",
    "failed": "failed",
    "cancelled": "cancelled",
}


class JobHandle(EvalBaseModel):
    """A started job: the id to poll, and what it is working on."""

    job_id: str = Field(description="The id to poll and cancel the job by.")
    kind: JobKind = Field(description="What the job's work is: a launched run, an analysis generation, or a sweep.")
    target_id: str = Field(
        description="The run the job runs, the campaign the generation analyses, or the campaign a sweep's runs join."
    )
    label: str = Field(default="", description="A short line naming the work (the run's model, the campaign).")


class JobsStarted(EvalBaseModel):
    """What starting long work returns: one handle per job, in the order the work was asked for."""

    jobs: list[JobHandle] = Field(min_length=1)


class JobStatus(EvalBaseModel):
    """Where one job stands, read from the record its work writes."""

    job_id: str
    kind: JobKind
    state: JobState
    status: str = Field(description="The record's own word: the run's status, or the attempt's outcome.")
    done: bool = Field(description="Whether the job has left the running state for good.")
    progress: dict[str, Any] = Field(
        default_factory=dict,
        description="The run's last progress write, or a sweep's arms launched and finished; empty else.",
    )
    run_id: str | None = None
    campaign_id: str | None = None
    analysis_id: str | None = Field(default=None, description="The analysis a completed generation stored.")
    sweep_id: str | None = Field(default=None, description="The sweep a sweep job runs.")
    detail: str | None = Field(default=None, description="Why it ended other than completed, when the record says.")


def generation_key(campaign_id: str, scope_id: str) -> str:
    """The exclusivity key one campaign's generations share in this process: one starts at a time per scope.

    Within a process the key makes the check and the start one step; across processes a start also refuses while
    the campaign has an attempt whose lease is live (:func:`~threetears.evals.ops.analysis_generate`).
    """
    return f"analysis-generation:{scope_id}:{campaign_id}"


def sweep_job_id(sweep_id: str) -> str:
    """The job id of a sweep."""
    return f"{SWEEP_JOB_PREFIX}{sweep_id}"


def sweep_key(sweep_id: str, scope_id: str) -> str:
    """The task key a sweep runs under in the process that started it, scoped like every key."""
    return f"eval-sweep:{scope_id}:{sweep_id}"


def sweep_progress(sweep: EvalSweep, runs: dict[str, EvalRun]) -> dict[str, Any]:
    """A sweep's progress: how many arms it has launched and finished, and each launched arm's run status.

    Args:
        sweep: Its record.
        runs: Its launched arms' runs, by id.

    Returns:
        ``{"arms_total", "arms_launched", "arms_finished", "arms": {label: status}}``, a never-launched arm's
        status ``not_launched``.
    """
    finished = {"completed", "budget_stopped", "exhausted", "failed", "cancelled"}
    statuses = {
        arm.label: runs[arm.run_id].status if arm.run_id is not None and arm.run_id in runs else "not_launched"
        for arm in sweep.arms
    }
    return {
        "arms_total": len(sweep.arms),
        "arms_launched": sum(1 for arm in sweep.arms if arm.run_id is not None),
        "arms_finished": sum(1 for status in statuses.values() if status in finished),
        "arms": statuses,
    }


def _sweep_status(sweep: EvalSweep, runs: dict[str, EvalRun]) -> JobStatus:
    """A sweep's job status from its record and its launched arms' runs.

    Args:
        sweep: Its record, as settled.
        runs: Its launched arms' runs, by id.

    Returns:
        The status, its progress counting the arms launched and finished.
    """
    state: JobState = "lost" if sweep.lost is not None else ("running" if sweep.outcome == "running" else sweep.outcome)
    detail = sweep.detail
    if sweep.lost is not None:
        detail = f"{sweep.lost.cause}; its launched arms are campaign members, and no further arm will be launched"
    return JobStatus(
        job_id=sweep_job_id(sweep.id),
        kind="sweep",
        state=state,
        status=sweep.outcome,
        done=state in TERMINAL_JOB_STATES,
        progress=sweep_progress(sweep, runs),
        campaign_id=sweep.campaign_id,
        sweep_id=sweep.id,
        detail=detail if state != "completed" else None,
    )


def run_job_id(run_id: str) -> str:
    """The job id of a launched run."""
    return f"{RUN_JOB_PREFIX}{run_id}"


def analysis_job_id(campaign_id: str, attempt_id: str) -> str:
    """The job id of an analysis generation: its campaign, then its attempt."""
    return f"{ANALYSIS_JOB_PREFIX}{campaign_id}:{attempt_id}"


def parse_job_id(job_id: str) -> tuple[JobKind, str, str | None]:
    """Read a job id back into what it names.

    Args:
        job_id: An id :func:`run_job_id` or :func:`analysis_job_id` made.

    Returns:
        ``("run", run_id, None)``, ``("analysis", campaign_id, attempt_id)`` or ``("sweep", sweep_id, None)``.

    Raises:
        ValidationFailedError: The id is neither shape — a typed or truncated id, which names no job.
    """
    if job_id.startswith(RUN_JOB_PREFIX) and job_id[len(RUN_JOB_PREFIX) :]:
        return "run", job_id[len(RUN_JOB_PREFIX) :], None
    if job_id.startswith(SWEEP_JOB_PREFIX) and job_id[len(SWEEP_JOB_PREFIX) :]:
        return "sweep", job_id[len(SWEEP_JOB_PREFIX) :], None
    if job_id.startswith(ANALYSIS_JOB_PREFIX):
        # The attempt id is a uuid and holds no colon, so the LAST colon splits it from a campaign id
        # that might hold one.
        campaign_id, colon, attempt_id = job_id[len(ANALYSIS_JOB_PREFIX) :].rpartition(":")
        if colon and campaign_id and attempt_id:
            return "analysis", campaign_id, attempt_id
    raise ValidationFailedError(
        f"job id {job_id!r} names no job: a job id is 'run:<run id>', 'analysis:<campaign id>:<attempt id>' or "
        "'sweep:<sweep id>', exactly as the start returned it"
    )


def _run_status(run: EvalRun) -> JobStatus:
    """A run's job status from its stored document, as settled.

    Args:
        run: The run as stored.

    Returns:
        The status: ``lost`` when its record says so, else its own status's job state.
    """
    state: JobState = "lost" if run.lost is not None else _RUN_STATES[run.status]
    detail = run.cancellation_reason or run.budget_stop_reason or ("; ".join(run.error_details) or None)
    if run.lost is not None:
        detail = run.lost.cause
    return JobStatus(
        job_id=run_job_id(run.id),
        kind="run",
        state=state,
        status=run.status,
        done=state in TERMINAL_JOB_STATES,
        progress=dict(run.progress),
        run_id=run.id,
        detail=detail if state != "completed" else None,
    )


def _attempt_status(job_id: str, attempt: EvalAnalysisAttempt) -> JobStatus:
    """A generation's job status from its attempt record, as settled.

    Args:
        job_id: The job's id.
        attempt: Its attempt, as stored.

    Returns:
        The status: ``running`` while the attempt is, ``lost`` when its record says so, else its outcome's state.
    """
    if attempt.lost is not None:
        state: JobState = "lost"
    elif attempt.outcome == "running":
        state = "running"
    else:
        state = _ATTEMPT_STATES[attempt.outcome]
    return JobStatus(
        job_id=job_id,
        kind="analysis",
        state=state,
        status=attempt.outcome,
        done=state in TERMINAL_JOB_STATES,
        campaign_id=attempt.campaign_id,
        analysis_id=attempt.analysis_id,
        detail=attempt.lost.cause if attempt.lost is not None else (attempt.error if state != "running" else None),
    )


async def _settled[R: EvalDocumentModel](
    host: OpsHost, kind: LeasedJobKind[R], record_id: str, scope_id: str
) -> R | None:
    """A job's record as every process reads it — written lost first when its lease has lapsed — on the blocking executor."""
    eval_host = host.eval_host
    record, _ = await run_blocking(
        eval_host.blocking_executor, settle_job_record, eval_host.storage, kind, record_id, scope_id
    )
    return record


async def _settled_attempt(
    host: OpsHost, job_id: str, campaign_id: str, attempt_id: str, scope_id: str
) -> EvalAnalysisAttempt:
    """A generation's attempt, settled, answered only under the campaign the job id names.

    Raises:
        NotFoundError: No attempt with that id in the scope, or it is filed under another campaign.
    """
    attempt = await _settled(host, ANALYSIS_JOB, attempt_id, scope_id)
    if attempt is None or attempt.campaign_id != campaign_id:
        raise NotFoundError("analysis job", job_id)
    return attempt


async def job_poll(host: OpsHost, job_id: str, scope_id: str) -> JobStatus:
    """Where a job stands, read from the record its work writes — the same answer in every process.

    The record is read on the host's blocking executor and settled: one that has not ended and whose lease has
    lapsed is written lost first (:func:`~threetears.evals.kernel.job_leases.settle_job_record`).

    Args:
        host: The host whose store holds the job's record.
        job_id: The id the start returned.
        scope_id: The scope the work's record lives in.

    Returns:
        The job's status.

    Raises:
        ValidationFailedError: The id names no job.
        NotFoundError: No record of the job in the scope (another scope's job reads so), or a generation's
            attempt filed under another campaign than the id names.
        ConflictError: The record kept changing while its lapsed lease was being written lost; poll again.
    """
    kind, target_id, attempt_id = parse_job_id(job_id)
    eval_host = host.eval_host
    if kind == "run":
        run = await _settled(host, RUN_JOB, target_id, scope_id)
        if run is None:
            raise NotFoundError("run", target_id)
        return _run_status(run)
    if kind == "sweep":
        sweep = await _settled(host, SWEEP_JOB, target_id, scope_id)
        if sweep is None:
            raise NotFoundError("sweep", target_id)
        run_ids = [arm.run_id for arm in sweep.arms if arm.run_id is not None]
        runs = await run_blocking(eval_host.blocking_executor, eval_host.storage.load_eval_runs, run_ids, scope_id)
        return _sweep_status(sweep, {run.id: run for run in runs})
    assert attempt_id is not None  # parse_job_id names an attempt for every analysis job
    return _attempt_status(job_id, await _settled_attempt(host, job_id, target_id, attempt_id, scope_id))


async def job_cancel(host: OpsHost, job_id: str, scope_id: str, *, reason: str | None = None) -> JobStatus:
    """Ask a running job to stop — from any process — then report where it stands.

    The cancel is written on the job's record (:func:`~threetears.evals.kernel.job_leases.request_job_cancel`), which the
    process running the job reads at its next lease renewal; when that process is this one, the job is cancelled
    here at once as well. Request-then-converge: the work records ``cancelled`` as it unwinds, so the status
    returned may still read ``running`` for up to one renewal interval — poll to see it land.

    Args:
        host: The host whose store holds the job's record and whose job manager may run it.
        job_id: The id the start returned.
        scope_id: The scope the work's record lives in.
        reason: Why, recorded on the record and on a cancelled run.

    Returns:
        The job's status after the request.

    Raises:
        ValidationFailedError: The id names no job, or the job has already ended — lost included, which this
            read may have just written.
        NotFoundError: No record of the job in the scope, or a generation's attempt filed under another campaign.
        ConflictError: The record kept changing while its lapsed lease was being written lost; nothing was asked.
        StorageError: The request could not be written.
    """
    kind, target_id, attempt_id = parse_job_id(job_id)
    manager, eval_host = host.launch.job_manager, host.eval_host
    record_id = target_id
    if kind == "run":
        run = await _settled(host, RUN_JOB, target_id, scope_id)
        if run is None:
            raise NotFoundError("run", target_id)
        require_cancellable(run)
    else:
        status = await job_poll(host, job_id, scope_id)
        if status.done:
            raise ValidationFailedError(
                f"{kind} job {job_id!r} is not running (it reads {status.state}); there is nothing to cancel"
            )
        if attempt_id is not None:
            record_id = attempt_id
    await run_blocking(
        eval_host.blocking_executor, request_job_cancel, eval_host.storage, record_id, scope_id, reason=reason
    )
    # This process may be the one running it: cancel it here now rather than at the next renewal. The record was
    # read in the caller's scope, so the id is one this caller may cancel.
    if kind == "run":
        manager.cancel_job(record_id, reason=reason)
    else:
        manager.cancel_task(record_id)
    return await job_poll(host, job_id, scope_id)


__all__ = [
    "ANALYSIS_JOB_PREFIX",
    "RUN_JOB_PREFIX",
    "SWEEP_JOB_PREFIX",
    "TERMINAL_JOB_STATES",
    "JobHandle",
    "JobKind",
    "JobState",
    "JobStatus",
    "JobsStarted",
    "analysis_job_id",
    "generation_key",
    "job_cancel",
    "parse_job_id",
    "job_poll",
    "run_job_id",
    "sweep_job_id",
    "sweep_key",
    "sweep_progress",
]
