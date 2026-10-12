"""One job contract for every long-running or model-calling operation: start, a job id, poll, cancel.

**No transport call waits on a model.** Every operation whose work calls one — a launch (its case generation and
its runs), a sweep, an analysis generation, a second judge's pass, a judge drift check, a judge temperature
comparison, a host action that drafts something — prices, caps and admits its work in the call and starts the work
as a job, returning :class:`JobsStarted`. Each such work writes its own record durably — a launch its
:class:`~threetears.evals.run.EvalLaunch`, a run its ``status``, a sweep its
:class:`~threetears.evals.kernel.campaign.EvalSweep`, a generation its
:class:`~threetears.evals.kernel.campaign.EvalAnalysisAttempt`, a second judge's pass its
:class:`~threetears.evals.run.EvalSecondJudgePass`, a temperature comparison its
:class:`~threetears.evals.run.EvalJudgeTemperatureCheck`, a host action its
:class:`~threetears.evals.run.EvalActionJob`. So a job here is a **name for the record that says how the work
stands**, and polling reads that record — and, once the work has completed, returns what it came to
(:attr:`JobStatus.result`).

**Two ways to run one.** An admitted operation (:class:`AdmittedJob`) is started as a job (:func:`start_job`), the
form every action takes; a caller that is not a transport — a script, a test — may instead await what it started in
the foreground (:func:`await_job`), which returns the final status, result and all. Nothing about a job lives only in
one process's memory, so a job id handed to an agent is answered the same by every process sharing the store, and
after a restart.

**The job id names its record.** ``launch:<launch id>`` for a launch, whose record names its runs once they start;
``run:<run id>`` for one of its runs; ``analysis:<campaign id>:<attempt id>`` for a generation — the attempt is filed
under its campaign, so the id carries both; ``sweep:<sweep id>`` for a sweep, whose record names its arms and their
runs; ``second-judge:<pass id>`` for a second judge's pass (a drift check's included); ``judge-temperature:<id>`` for
a temperature comparison; and ``action:<id>`` for a host action's job. :func:`parse_job_id` is the one reader of
that shape.

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

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from threetears.evals.schema.base import EvalBaseModel, EvalDocumentModel
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.schema.refusal import JobRefusal
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
from threetears.evals.kernel.out_of_run import AdmittedCall, OutOfRunBudget, PlannedCall
from threetears.evals.kernel.host.eval_host import CompletionRole
from threetears.evals.schema.completion import BoundCompletionClient
from threetears.evals.schema.out_of_run_spend import OutOfRunPurpose
from threetears.evals.analysis.agreement import InterJudgeAgreement, inter_judge_agreement
from threetears.evals.analysis.judge_drift import JudgeDrift, judge_drift
from threetears.evals.analysis.judge_temperature import read_judge_temperatures
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from threetears.evals.run.job_work import (
    ACTION_JOB,
    EvalActionJob,
    JobOutcome,
    JobWork,
    action_job_work,
    start_job_work,
)
from threetears.evals.run.judge_second import SECOND_JUDGE_JOB, EvalSecondJudgePass, SecondJudgeReport
from threetears.evals.run.judge_temperature import JUDGE_TEMPERATURE_JOB, EvalJudgeTemperatureCheck
from threetears.evals.run.launch_job import LAUNCH_JOB, EvalLaunch
from threetears.evals.kernel.remedies import FIND_JOB, FIND_RUN

#: What a job's work is: a launch (its arms prepared and its runs run), one launched run, an analysis generation, a
#: sweep launching its arms in order, a second judge's pass, a judge temperature comparison, or a host action's work.
JobKind = Literal["launch", "run", "analysis", "sweep", "second_judge", "judge_temperature", "action"]

#: Where a job stands. ``running`` — still going (a run ``pending`` or ``running``, a generation live);
#: ``completed`` — ended with what it was for; ``stopped`` — ended short by a budget, keeping what it recorded: a
#: run by its cost or time budget or its account, a job's work by its time budget (its status ``timed_out``);
#: ``failed``; ``cancelled`` — asked to stop; ``lost`` — the process running it went away before it ended: its
#: lease lapsed, or its process shut down, and its record says so.
JobState = Literal["running", "completed", "stopped", "failed", "cancelled", "lost"]

#: The states a job does not leave.
TERMINAL_JOB_STATES: frozenset[str] = frozenset({"completed", "stopped", "failed", "cancelled", "lost"})

#: The prefix of a launched run's job id.
RUN_JOB_PREFIX = "run:"
#: The prefix of an analysis generation's job id.
ANALYSIS_JOB_PREFIX = "analysis:"

#: The prefix of a sweep's job id.
SWEEP_JOB_PREFIX = "sweep:"

#: The prefix of a launch's job id.
LAUNCH_JOB_PREFIX = "launch:"

#: The prefix of a second judge's pass's job id — a drift check's too.
SECOND_JUDGE_JOB_PREFIX = "second-judge:"

#: The prefix of a judge temperature comparison's job id.
JUDGE_TEMPERATURE_JOB_PREFIX = "judge-temperature:"

#: The prefix of a host action's job id.
ACTION_JOB_PREFIX = "action:"

#: Each single-id job kind by the prefix its id carries — every kind but the analysis generation, whose id
#: carries its campaign too.
_PREFIXED: dict[str, JobKind] = {
    RUN_JOB_PREFIX: "run",
    SWEEP_JOB_PREFIX: "sweep",
    LAUNCH_JOB_PREFIX: "launch",
    SECOND_JUDGE_JOB_PREFIX: "second_judge",
    JUDGE_TEMPERATURE_JOB_PREFIX: "judge_temperature",
    ACTION_JOB_PREFIX: "action",
}

#: The job id prefix of each single-id kind.
_PREFIX_OF: dict[JobKind, str] = {kind: prefix for prefix, kind in _PREFIXED.items()}

#: A job record's own outcome, as the job state it is.
_OUTCOME_STATES: dict[JobOutcome, JobState] = {
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "timed_out": "stopped",
}

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
    "timed_out": "stopped",
}

#: A sweep's recorded outcome, as the job state it is.
_SWEEP_STATES: dict[str, JobState] = {
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "timed_out": "stopped",
}


class JobHandle(EvalBaseModel):
    """A started job: the id to poll, and what it is working on."""

    job_id: str = Field(description="The id to poll and cancel the job by.")
    kind: JobKind = Field(
        description="What the job's work is: a launch, a run, an analysis generation, a sweep, a second judge's pass, "
        "a judge temperature comparison, or a host action."
    )
    target_id: str = Field(
        description="What the job works on: the template a launch runs, the run a run job or a judge job reads, the "
        "campaign a generation analyses or a sweep's runs join, or the action a host action's job runs."
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
    run_id: str | None = Field(default=None, description="The run a run job runs, or a judge job reads.")
    run_ids: list[str] = Field(
        default_factory=list, description="The runs a launch started, in arm order; empty until they start."
    )
    campaign_id: str | None = None
    analysis_id: str | None = Field(default=None, description="The analysis a completed generation stored.")
    sweep_id: str | None = Field(default=None, description="The sweep a sweep job runs.")
    detail: str | None = Field(default=None, description="Why it ended other than completed, when the record says.")
    faults: list[str] = Field(
        default_factory=list,
        description=(
            "Faults at the job's lease boundary its record keeps, oldest first — a holder whose renewals failed for "
            "a whole lease, which stopped the job, names the last renewal's error here. Kept beside the ending, so a "
            "job read lost still says why its holder stopped. Empty when there were none."
        ),
    )
    refusal: JobRefusal | None = Field(
        default=None,
        description=(
            "The engine refusal the job ended on, as data — its kind and status, the same pair a refused call's "
            "structured refusal carries — so a host branches on the kind, never on `detail`'s words. Set when the "
            "job's work was refused (a provider refusing its account is `provider_refused`); None otherwise, and "
            "for a run, whose `status` is its own ending's word (`exhausted` for an account the provider refused)."
        ),
    )
    result: dict[str, Any] | None = Field(
        default=None,
        description="What the work came to, once it completed: a second judge's read (its report, agreement and "
        "drift), a temperature comparison, or a host action's typed result, as JSON. None while it runs, when it "
        "ended otherwise, and for a job whose record is its result (a run, a launch, a sweep, a generation).",
    )


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
    state: JobState = "lost" if sweep.lost is not None else _SWEEP_STATES[sweep.outcome]
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
        faults=list(sweep.faults),
        refusal=sweep.refusal,
    )


def run_job_id(run_id: str) -> str:
    """The job id of a launched run."""
    return f"{RUN_JOB_PREFIX}{run_id}"


def analysis_job_id(campaign_id: str, attempt_id: str) -> str:
    """The job id of an analysis generation: its campaign, then its attempt."""
    return f"{ANALYSIS_JOB_PREFIX}{campaign_id}:{attempt_id}"


def launch_job_id(launch_id: str) -> str:
    """The job id of a launch."""
    return f"{LAUNCH_JOB_PREFIX}{launch_id}"


def second_judge_job_id(pass_id: str) -> str:
    """The job id of a second judge's pass."""
    return f"{SECOND_JUDGE_JOB_PREFIX}{pass_id}"


def judge_temperature_job_id(check_id: str) -> str:
    """The job id of a judge temperature comparison."""
    return f"{JUDGE_TEMPERATURE_JOB_PREFIX}{check_id}"


def action_job_id(record_id: str) -> str:
    """The job id of a host action's job."""
    return f"{ACTION_JOB_PREFIX}{record_id}"


def parse_job_id(job_id: str) -> tuple[JobKind, str, str | None]:
    """Read a job id back into what it names.

    Args:
        job_id: An id a start returned.

    Returns:
        ``(kind, record_id, None)`` for every kind but a generation, and ``("analysis", campaign_id, attempt_id)``
        for a generation.

    Raises:
        ValidationFailedError: The id is none of the shapes — a typed or truncated id, which names no job.
    """
    for prefix, kind in _PREFIXED.items():
        if job_id.startswith(prefix) and job_id[len(prefix) :]:
            return kind, job_id[len(prefix) :], None
    if job_id.startswith(ANALYSIS_JOB_PREFIX):
        # The attempt id is a uuid and holds no colon, so the LAST colon splits it from a campaign id
        # that might hold one.
        campaign_id, colon, attempt_id = job_id[len(ANALYSIS_JOB_PREFIX) :].rpartition(":")
        if colon and campaign_id and attempt_id:
            return "analysis", campaign_id, attempt_id
    raise ValidationFailedError(
        f"job id {job_id!r} names no job",
        remedy=(
            "pass a job id exactly as its start returned it: 'launch:<id>', 'run:<run id>', "
            "'analysis:<campaign id>:<attempt id>', 'sweep:<sweep id>', 'second-judge:<id>', 'judge-temperature:<id>' "
            "or 'action:<id>'"
        ),
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
        faults=list(run.faults),
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
        faults=list(attempt.faults),
        refusal=attempt.refusal,
    )


class SecondJudgeRead(EvalBaseModel):
    """A second judge's pass over a run, and what its pairs say: agreement between the judges, and drift.

    Both readings are over this pass's pairs only, read off the stored records by the code every other surface reads
    them with (:func:`~threetears.evals.analysis.inter_judge_agreement`, :func:`~threetears.evals.analysis.judge_drift`).
    """

    report: SecondJudgeReport = Field(description="What was asked, what was left out, and what it cost.")
    agreement: InterJudgeAgreement = Field(
        description="Per-dimension agreement between the run's judge and the second."
    )
    drift: JudgeDrift = Field(description="Per-dimension movement from the run's judge to the second, with intervals.")


def _record_status(
    job_id: str,
    kind: JobKind,
    record: EvalLaunch | EvalSecondJudgePass | EvalJudgeTemperatureCheck | EvalActionJob,
    *,
    run_id: str | None = None,
    run_ids: list[str] | None = None,
    progress: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> JobStatus:
    """A job's status from a record of the job-work shape (:mod:`threetears.evals.run.job_work`), as settled.

    Returns:
        The status: ``lost`` when its record says so, else its outcome's state; the result only once completed.
    """
    state: JobState = "lost" if record.lost is not None else _OUTCOME_STATES[record.outcome]
    detail = record.detail
    if record.lost is not None:
        detail = record.lost.cause
    elif state == "cancelled" and record.cancel_request_reason:
        detail = f"{detail}: {record.cancel_request_reason}" if detail else record.cancel_request_reason
    return JobStatus(
        job_id=job_id,
        kind=kind,
        state=state,
        status=record.outcome,
        done=state in TERMINAL_JOB_STATES,
        progress=progress or {},
        run_id=run_id,
        run_ids=run_ids or [],
        detail=detail if state != "completed" else None,
        faults=list(record.faults),
        result=result if state == "completed" else None,
        refusal=record.refusal,
    )


def launch_progress(launch: EvalLaunch, runs: dict[str, EvalRun]) -> dict[str, Any]:
    """A launch's progress: its runs started and finished, and each started run's status by its arm.

    Args:
        launch: Its record.
        runs: Its started runs, by id.

    Returns:
        ``{"runs_started", "runs_finished", "runs": {run id: status}}``.
    """
    finished = {"completed", "budget_stopped", "exhausted", "failed", "cancelled"}
    statuses = {run_id: runs[run_id].status if run_id in runs else "not_found" for run_id in launch.run_ids}
    return {
        "runs_started": len(launch.run_ids),
        "runs_finished": sum(1 for status in statuses.values() if status in finished),
        "runs": statuses,
    }


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
        raise NotFoundError("analysis job", job_id, remedy=FIND_JOB)
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
            raise NotFoundError("run", target_id, remedy=FIND_RUN)
        return _run_status(run)
    if kind == "sweep":
        sweep = await _settled(host, SWEEP_JOB, target_id, scope_id)
        if sweep is None:
            raise NotFoundError("sweep", target_id, remedy=FIND_JOB)
        run_ids = [arm.run_id for arm in sweep.arms if arm.run_id is not None]
        runs = await run_blocking(eval_host.blocking_executor, eval_host.storage.load_eval_runs, run_ids, scope_id)
        return _sweep_status(sweep, {run.id: run for run in runs})
    if kind == "launch":
        launch = await _settled(host, LAUNCH_JOB, target_id, scope_id)
        if launch is None:
            raise NotFoundError("launch", target_id, remedy=FIND_JOB)
        started = await run_blocking(
            eval_host.blocking_executor, eval_host.storage.load_eval_runs, launch.run_ids, scope_id
        )
        progress = launch_progress(launch, {run.id: run for run in started})
        return _record_status(job_id, kind, launch, run_ids=list(launch.run_ids), progress=progress)
    if kind == "second_judge":
        judged = await _settled(host, SECOND_JUDGE_JOB, target_id, scope_id)
        if judged is None:
            raise NotFoundError("second judge pass", target_id, remedy=FIND_JOB)
        read: dict[str, Any] | None = None
        if judged.outcome == "completed" and judged.report is not None and judged.lost is None:
            results = await run_blocking(
                eval_host.blocking_executor, eval_host.storage.query_eval_results_by_run, judged.run_id, scope_id
            )
            read = SecondJudgeRead(
                report=judged.report,
                agreement=inter_judge_agreement(results, pass_id=judged.id),
                drift=judge_drift(results, pass_id=judged.id),
            ).model_dump(mode="json")
        return _record_status(job_id, kind, judged, run_id=judged.run_id, result=read)
    if kind == "judge_temperature":
        check = await _settled(host, JUDGE_TEMPERATURE_JOB, target_id, scope_id)
        if check is None:
            raise NotFoundError("judge temperature comparison", target_id, remedy=FIND_JOB)
        comparison = None if check.answers is None else read_judge_temperatures(check.answers).model_dump(mode="json")
        return _record_status(job_id, kind, check, run_id=check.run_id, result=comparison)
    if kind == "action":
        acted = await _settled(host, ACTION_JOB, target_id, scope_id)
        if acted is None:
            raise NotFoundError("action job", target_id, remedy=FIND_JOB)
        return _record_status(job_id, kind, acted, result=acted.result)
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
            raise NotFoundError("run", target_id, remedy=FIND_RUN)
        require_cancellable(run)
    else:
        status = await job_poll(host, job_id, scope_id)
        if status.done:
            raise ValidationFailedError(
                f"{kind} job {job_id!r} is not running (it reads {status.state}); there is nothing to cancel",
                remedy="read how it ended with job_poll",
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


@dataclass(frozen=True, kw_only=True)
class AdmittedJob:
    """An operation admitted — priced, capped and refused where it would be — with the job it starts as.

    What every model-calling operation's call half returns: the work (:class:`~threetears.evals.run.JobWork`), whose
    record the job writes, and the handle a start returns for it.

    Attributes:
        work: The admitted work.
        kind: The job's kind.
        target_id: What the job works on, for its handle.
        label: A short line naming the work.
        key: An exclusivity key the job holds while live; ``None`` for none.
    """

    work: JobWork[Any]
    kind: JobKind
    target_id: str
    label: str = ""
    key: str | None = None

    @property
    def job_id(self) -> str:
        """The id the job is polled and cancelled by."""
        return f"{_PREFIX_OF[self.kind]}{self.work.record_id}"


async def start_job(host: OpsHost, admitted: AdmittedJob) -> JobsStarted:
    """Start admitted work as a job: its record written, leased, and the work run in the background.

    The form every action's call takes, so the call returns at once and no transport waits on the work.

    Args:
        host: The host whose job manager runs the job and whose store holds its record.
        admitted: The admitted operation.

    Returns:
        The job's one handle.

    Raises:
        ConflictError: A live job holds the admitted key.
        StorageError: The record could not be written; nothing started.
    """
    eval_host = host.eval_host
    await start_job_work(
        host.launch.job_manager,
        admitted.work,
        store=eval_host.storage,
        executor=eval_host.blocking_executor,
        key=admitted.key,
    )
    return JobsStarted(
        jobs=[JobHandle(job_id=admitted.job_id, kind=admitted.kind, target_id=admitted.target_id, label=admitted.label)]
    )


async def start_action_job(
    host: OpsHost,
    scope_id: str,
    *,
    action: str,
    run: Callable[[], Awaitable[BaseModel]],
    budget_s: float,
    label: str = "",
    discard: Callable[[], Awaitable[None]] | None = None,
) -> JobsStarted:
    """Start a host action's work as a job whose record (:class:`~threetears.evals.run.EvalActionJob`) holds its result.

    For a host action whose work calls a model and produces a value rather than a run — a rubric proposal, cases for
    a classifier. Its handler refuses and prices in the call, then hands the spending half here as ``run`` and
    returns what this returns; ``job_poll`` on the job returns ``run``'s typed result as JSON once it completes. The
    action is declared ``long_running`` with result :class:`JobsStarted`.

    Args:
        host: The host whose job manager runs the job.
        scope_id: The caller's scope, where the record lives.
        action: The action's name.
        run: The work: makes the calls and returns the action's typed result.
        label: A short line naming the work.
        budget_s: The work's wall-clock ceiling. Required: a forgotten budget is a ``TypeError`` here, never an hour
            nobody chose (:data:`~threetears.evals.run.DEFAULT_ACTION_JOB_BUDGET_S` is the value to pass for that).
        discard: Releases what the handler built for ``run`` when the work never runs.

    Returns:
        The job's one handle (``action:<id>``); its target is the action.

    Raises:
        ValueError: ``budget_s`` is not positive.
        StorageError: The record could not be written; nothing started.
    """
    work = action_job_work(scope_id, action=action, run=run, label=label, budget_s=budget_s, discard=discard)
    return await start_job(host, AdmittedJob(work=work, kind="action", target_id=action, label=label))


async def start_paid_call[R: BaseModel](
    host: OpsHost,
    scope_id: str,
    *,
    action: str,
    role: CompletionRole,
    model: str | None,
    purpose: OutOfRunPurpose,
    budget: OutOfRunBudget,
    call: PlannedCall,
    make: Callable[[BoundCompletionClient, AdmittedCall], Awaitable[R]],
    label: str,
    budget_s: float,
) -> JobsStarted:
    """Start a host action's one paid model call: admitted against its cap in the call, made in a job.

    The shape every spend action takes, for a host action that makes one model call and returns a value (a
    drafted rubric, drafted cases): the role's client is built, ``call`` is priced and admitted on it against
    ``budget`` **here**, so a call the cap cannot pay for is refused with no job started and nothing spent; then
    ``make`` runs **as a job** (:func:`start_action_job`), since no transport call waits on a model. The client
    is the work's from the moment it is built: released here if admission refuses, by the job's ``discard`` if
    the job never starts, and by ``make`` otherwise.

    Args:
        host: The host the action runs on; its client factory builds the client.
        scope_id: The caller's scope, where the job's record lives.
        action: The action's name, as the job records it.
        role: The completion role the client is built for.
        model: The model named, or ``None`` for the role's default as the host resolves it.
        purpose: What the call is for, as the out-of-run ledger records it.
        budget: The out-of-run budget that admits the call and ledgers it once made.
        call: The call, as planned.
        make: Makes the admitted call on the client (through ``budget``) and reads its result; owns the client's
            release from the moment it is called.
        label: What the job is, for a reader of ``job_poll``.
        budget_s: The job's wall-clock ceiling. Required, as :func:`start_action_job`'s is: a call takes as long as
            its model, and only the action knows what is too long.

    Returns:
        The started job; ``job_poll`` returns ``make``'s result once it completes.

    Raises:
        ValueError: The host supplies no completion clients, or ``budget_s`` is not positive.
        ValidationFailedError: The cap cannot pay for the call, or an enforced cap cannot price it; nothing was made.
        StorageError: The job's record could not be written; the client was released and nothing was made.
    """
    eval_host = host.eval_host
    client = eval_host.completion_clients(label)(role, model)
    try:
        # Pricing reads the store's ledger of what the cap has already spent, so it runs off the loop.
        (admitted,) = await run_blocking(eval_host.blocking_executor, budget.admit, client, purpose, [call])
    # prawduct:ok-broad-except — not handled: a client that admitted nothing is released, and what refused is re-raised
    except BaseException:
        await client.aclose()
        raise

    async def made() -> R:
        return await make(client, admitted)

    return await start_action_job(
        host, scope_id, action=action, run=made, label=label, budget_s=budget_s, discard=client.aclose
    )


def _record_id(job_id: str) -> str:
    """The id of the record a job id names — the job's id in the job manager."""
    kind, target_id, attempt_id = parse_job_id(job_id)
    return attempt_id if kind == "analysis" and attempt_id is not None else target_id


async def await_job(host: OpsHost, started: JobsStarted, scope_id: str) -> list[JobStatus]:
    """Await started jobs to their end in the foreground, and return how each ended — its result included.

    For a caller that is not a transport — a script, a test, a host's own batch — and that started the jobs in this
    process: it waits as long as the work takes, which no transport call may. **Waiting here is owning**: a caller
    cancelled while it waits cancels the jobs it was waiting on, as a script interrupted would want, then re-raises.

    Args:
        host: The host whose job manager runs the jobs.
        started: What the start returned.
        scope_id: The scope the jobs' records live in.

    Returns:
        Each job's final status, in the order started.

    Raises:
        asyncio.CancelledError: The caller was cancelled; the jobs were asked to stop.
    """
    manager = host.launch.job_manager
    record_ids = [_record_id(handle.job_id) for handle in started.jobs]
    try:
        await manager.wait_for(record_ids)
    except asyncio.CancelledError:
        for handle, record_id in zip(started.jobs, record_ids, strict=True):
            if handle.kind == "run":
                manager.cancel_job(record_id, reason="its foreground caller was cancelled")
            else:
                manager.cancel_task(record_id)
        await asyncio.shield(manager.wait_for(record_ids))
        raise
    return [await job_poll(host, handle.job_id, scope_id) for handle in started.jobs]


__all__ = [
    "ACTION_JOB_PREFIX",
    "JUDGE_TEMPERATURE_JOB_PREFIX",
    "LAUNCH_JOB_PREFIX",
    "SECOND_JUDGE_JOB_PREFIX",
    "AdmittedJob",
    "SecondJudgeRead",
    "action_job_id",
    "await_job",
    "judge_temperature_job_id",
    "launch_job_id",
    "launch_progress",
    "second_judge_job_id",
    "start_action_job",
    "start_job",
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
