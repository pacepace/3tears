"""One job contract for every long-running operation: start, a job id, poll, cancel.

A launch and an analysis generation both take minutes, and both already record their own ending
durably — a run its ``status``, a generation its :class:`~threetears.evals.contracts.campaign.EvalAnalysisAttempt`.
So a job here is not a new record: it is a **name for the record that will say how the work ended**,
and polling reads that record. Nothing about a job lives only in this process's memory, so a job id
handed to an agent stays answerable after a restart — the answer is then that nothing is running it.

**The job id names its record.** ``run:<run id>`` for a launched run (one per arm), and
``analysis:<campaign id>:<attempt id>`` for a generation — the attempt is filed under its campaign, so
the id carries both. :func:`parse_job_id` is the one reader of that shape.

**Liveness is read before the record.** A job ends by writing its record and only then leaving the job
manager, so asking "is it live?" first and reading the record second never sees a finished job as
lost; the other order can.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from threetears.evals.analysis.service import list_analysis_attempts
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.models import EvalRun
from threetears.evals.ops.host import OpsHost
from threetears.evals.run.lifecycle import cancel_run, get_run
from threetears.evals.run.offload import run_blocking

#: What a job's work is: a launched run, or an analysis generation.
JobKind = Literal["run", "analysis"]

#: Where a job stands. ``running`` — still going (a run ``pending`` or ``running``, a generation live);
#: ``completed`` — ended with what it was for; ``stopped`` — a run ended short by its budget or its
#: account, keeping what it measured; ``failed``; ``cancelled``; ``lost`` — its record says it never
#: ended and nothing in this process is running it (a restart took it).
JobState = Literal["running", "completed", "stopped", "failed", "cancelled", "lost"]

#: The states a job does not leave.
TERMINAL_JOB_STATES: frozenset[str] = frozenset({"completed", "stopped", "failed", "cancelled", "lost"})

RUN_JOB_PREFIX = "run:"
ANALYSIS_JOB_PREFIX = "analysis:"

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
    kind: JobKind = Field(description="What the job's work is: a launched run, or an analysis generation.")
    target_id: str = Field(description="The run the job runs, or the campaign the generation analyses.")
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
    progress: dict[str, Any] = Field(default_factory=dict, description="The run's last progress write; empty else.")
    run_id: str | None = None
    campaign_id: str | None = None
    analysis_id: str | None = Field(default=None, description="The analysis a completed generation stored.")
    detail: str | None = Field(default=None, description="Why it ended other than completed, when the record says.")


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
        ``("run", run_id, None)`` or ``("analysis", campaign_id, attempt_id)``.

    Raises:
        ValidationFailedError: The id is neither shape — a typed or truncated id, which names no job.
    """
    if job_id.startswith(RUN_JOB_PREFIX) and job_id[len(RUN_JOB_PREFIX) :]:
        return "run", job_id[len(RUN_JOB_PREFIX) :], None
    if job_id.startswith(ANALYSIS_JOB_PREFIX):
        # The attempt id is a uuid and holds no colon, so the LAST colon splits it from a campaign id
        # that might hold one.
        campaign_id, colon, attempt_id = job_id[len(ANALYSIS_JOB_PREFIX) :].rpartition(":")
        if colon and campaign_id and attempt_id:
            return "analysis", campaign_id, attempt_id
    raise ValidationFailedError(
        f"job id {job_id!r} names no job: a job id is 'run:<run id>' or 'analysis:<campaign id>:<attempt id>', "
        "exactly as the start returned it"
    )


def _run_status(run: EvalRun, *, live: bool) -> JobStatus:
    """A run's job status from its stored document and whether this process is running it.

    Args:
        run: The run as stored.
        live: Whether this process's job manager holds a live task for it — read BEFORE the run was.

    Returns:
        The status.
    """
    state = _RUN_STATES[run.status]
    detail = run.cancellation_reason or run.budget_stop_reason or ("; ".join(run.error_details) or None)
    if state == "running" and not live:
        state = "lost"
        detail = (
            f"the run reads {run.status} but no job in this process is running it — the process that started it "
            "ended; the host's boot sweep relabels it abandoned"
        )
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


async def job_poll(host: OpsHost, job_id: str, scope_id: str) -> JobStatus:
    """Where a job stands, read from the record its work writes.

    Liveness is asked of the job manager on the loop that owns it; the record is read on the host's
    blocking executor, after.

    Args:
        host: The host whose job manager runs the work and whose store holds its record.
        job_id: The id the start returned.
        scope_id: The scope the work's record lives in.

    Returns:
        The job's status.

    Raises:
        ValidationFailedError: The id names no job.
        NotFoundError: A run job whose run is not in the scope.
    """
    kind, target_id, attempt_id = parse_job_id(job_id)
    manager, eval_host = host.launch.job_manager, host.eval_host
    if kind == "run":
        live = manager.is_active(target_id)
        run = await run_blocking(eval_host.blocking_executor, get_run, eval_host.storage, target_id, scope_id)
        return _run_status(run, live=live)
    assert attempt_id is not None  # parse_job_id names an attempt for every analysis job
    if manager.is_task_active(attempt_id):
        return JobStatus(
            job_id=job_id, kind="analysis", state="running", status="running", done=False, campaign_id=target_id
        )
    attempts = await run_blocking(
        eval_host.blocking_executor, list_analysis_attempts, eval_host.storage, target_id, scope_id
    )
    attempt = next((recorded for recorded in attempts if recorded.id == attempt_id), None)
    if attempt is None:
        return JobStatus(
            job_id=job_id,
            kind="analysis",
            state="lost",
            status="unrecorded",
            done=True,
            campaign_id=target_id,
            detail=(
                "no generation in this process is running it and it recorded no attempt — the process that "
                "started it ended mid-generation, so what it spent is recorded nowhere"
            ),
        )
    state = _ATTEMPT_STATES[attempt.outcome]
    return JobStatus(
        job_id=job_id,
        kind="analysis",
        state=state,
        status=attempt.outcome,
        done=True,
        campaign_id=target_id,
        analysis_id=attempt.analysis_id,
        detail=attempt.error if state != "completed" else None,
    )


async def job_cancel(host: OpsHost, job_id: str, scope_id: str, *, reason: str | None = None) -> JobStatus:
    """Ask a running job to stop, then report where it stands.

    Cancellation is request-then-converge: the work records ``cancelled`` as it unwinds, so the status
    returned may still read ``running`` for a moment — poll to see it land.

    Args:
        host: The host whose job manager runs the work.
        job_id: The id the start returned.
        scope_id: The scope the work's record lives in.
        reason: Why, recorded on a cancelled run.

    Returns:
        The job's status after the request.

    Raises:
        ValidationFailedError: The id names no job, or the job has already ended.
        NotFoundError: A run job whose run is not in the scope.
    """
    kind, target_id, attempt_id = parse_job_id(job_id)
    if kind == "run":
        cancel_run(host.eval_host.storage, target_id, scope_id, job_manager=host.launch.job_manager, reason=reason)
        return await job_poll(host, job_id, scope_id)
    assert attempt_id is not None  # parse_job_id names an attempt for every analysis job
    if not host.launch.job_manager.cancel_task(attempt_id):
        status = await job_poll(host, job_id, scope_id)
        raise ValidationFailedError(
            f"analysis job {job_id!r} is not running here (it reads {status.state}); there is nothing to cancel"
        )
    return await job_poll(host, job_id, scope_id)


__all__ = [
    "ANALYSIS_JOB_PREFIX",
    "RUN_JOB_PREFIX",
    "TERMINAL_JOB_STATES",
    "JobHandle",
    "JobKind",
    "JobState",
    "JobStatus",
    "JobsStarted",
    "analysis_job_id",
    "job_cancel",
    "parse_job_id",
    "job_poll",
    "run_job_id",
]
