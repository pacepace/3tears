"""One job contract for every long-running operation: start, a job id, poll, cancel.

A launch and an analysis generation both take minutes, and both already record their own ending
durably — a run its ``status``, a generation its :class:`~threetears.evals.kernel.campaign.EvalAnalysisAttempt`.
So a job here is not a new record: it is a **name for the record that will say how the work ended**,
and polling reads that record. Nothing about a job lives only in this process's memory, so a job id
handed to an agent stays answerable after a restart — the answer is then that nothing is running it.

**The job id names its record.** ``run:<run id>`` for a launched run (one per arm),
``analysis:<campaign id>:<attempt id>`` for a generation — the attempt is filed under its campaign, so
the id carries both — ``sweep:<sweep id>`` for a sweep, whose record
(:class:`~threetears.evals.kernel.campaign.EvalSweep`) names its arms and their runs, and ``judge:<job id>`` for a
judge job — a second judge's pass, a drift check or a temperature comparison — whose record
(:class:`~threetears.evals.kernel.EvalJudgeJob`) carries its progress and, once it ends, what it handed back, which
its poll reads into the report the action names (:class:`~threetears.evals.ops.SecondJudgeRead`,
:class:`~threetears.evals.analysis.JudgeTemperatureComparison`). :func:`parse_job_id` is the one reader of that
shape. A sweep and a judge job are live to a caller only under their own scope's task key (:func:`sweep_key`,
:func:`judge_key`), as a generation is.

**A job is answered only inside the caller's scope.** A run job's run is read scoped, and a generation is
live to a caller only when its task holds the exclusivity key of the campaign AND scope the caller names
(:func:`generation_key`) — the job manager is process-wide, so asking it "is this attempt live?" alone would
let any scope that learned an attempt id poll or cancel another scope's generation, and echo back whatever
campaign it typed as fact. A generation another scope runs therefore reads to this caller exactly as one
nothing runs: ``lost`` on poll, refused on cancel.

**Liveness is read before the record.** A job ends by writing its record and only then leaving the job
manager, so asking "is it live?" first and reading the record second never sees a finished job as
lost; the other order can.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from threetears.evals.analysis.judge_temperature import JudgeTemperatureComparison, read_judge_temperatures
from threetears.evals.analysis.service import list_analysis_attempts
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.campaign import EvalSweep
from threetears.evals.kernel.judge_jobs import EvalJudgeJob, JudgeJobAction
from threetears.evals.schema.models import EvalRun
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.judge_reads import SecondJudgeRead, read_second_judge
from threetears.evals.run.lifecycle import get_run, repair_abandoned_run, require_cancellable
from threetears.evals.kernel.offload import run_blocking

#: What a job's work is: a launched run, an analysis generation, a sweep launching its arms in order, or a judge job
#: re-judging a finished run's stored evidence (``judge_second``, ``judge_drift_check``, ``judge_temperature``).
JobKind = Literal["run", "analysis", "sweep", "judge"]

#: Where a job stands. ``running`` — still going (a run ``pending`` or ``running``, a generation live);
#: ``completed`` — ended with what it was for; ``stopped`` — a run ended short by its budget or its
#: account, keeping what it measured; ``failed``; ``cancelled``; ``lost`` — its record says it never
#: ended and nothing in this process is running it (a restart took it).
JobState = Literal["running", "completed", "stopped", "failed", "cancelled", "lost"]

#: The states a job does not leave.
TERMINAL_JOB_STATES: frozenset[str] = frozenset({"completed", "stopped", "failed", "cancelled", "lost"})

#: The prefix of a launched run's job id.
RUN_JOB_PREFIX = "run:"
#: The prefix of an analysis generation's job id.
ANALYSIS_JOB_PREFIX = "analysis:"

#: The prefix of a sweep's job id.
SWEEP_JOB_PREFIX = "sweep:"

#: The prefix of a judge job's id.
JUDGE_JOB_PREFIX = "judge:"

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
    kind: JobKind = Field(
        description="What the job's work is: a launched run, an analysis generation, a sweep, or a judge job."
    )
    target_id: str = Field(
        description="The run the job runs, the campaign the generation analyses, the campaign a sweep's runs join, "
        "or the run a judge job re-judges."
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
        description="The run's last progress write, a sweep's arms launched and finished, or a judge job's results "
        "asked and calls made; empty else.",
    )
    run_id: str | None = None
    campaign_id: str | None = None
    analysis_id: str | None = Field(default=None, description="The analysis a completed generation stored.")
    sweep_id: str | None = Field(default=None, description="The sweep a sweep job runs.")
    judge_action: JudgeJobAction | None = Field(default=None, description="The action a judge job was started by.")
    second_judge: SecondJudgeRead | None = Field(
        default=None,
        description="A judge_second or judge_drift_check job's report once it has ended, completed or cancelled (then "
        "partial): what the synchronous action returned.",
    )
    judge_temperature: JudgeTemperatureComparison | None = Field(
        default=None,
        description="A judge_temperature job's comparison once it has ended, completed or cancelled (then partial).",
    )
    detail: str | None = Field(default=None, description="Why it ended other than completed, when the record says.")


def generation_key(campaign_id: str, scope_id: str) -> str:
    """The exclusivity key one campaign's generations share: one runs at a time, and only its scope sees it."""
    return f"analysis-generation:{scope_id}:{campaign_id}"


def _live_generation(host: OpsHost, campaign_id: str, attempt_id: str, scope_id: str) -> bool:
    """Whether this process runs ``attempt_id`` as a generation of ``campaign_id`` in ``scope_id`` — the one scope check.

    Args:
        host: The host whose job manager runs generations.
        campaign_id: The campaign the job id names.
        attempt_id: The attempt the job id names.
        scope_id: The caller's scope.

    Returns:
        ``True`` only when the attempt is a live task under that campaign's and scope's key.
    """
    return attempt_id in host.launch.job_manager.active_task_ids(generation_key(campaign_id, scope_id))


def sweep_job_id(sweep_id: str) -> str:
    """The job id of a sweep."""
    return f"{SWEEP_JOB_PREFIX}{sweep_id}"


def sweep_key(sweep_id: str, scope_id: str) -> str:
    """The task key a sweep runs under: the scope is in it, so only its own scope sees it live."""
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


def _live_sweep(host: OpsHost, sweep_id: str, scope_id: str) -> bool:
    """Whether this process runs ``sweep_id`` in ``scope_id`` — another scope's sweep reads as none."""
    return sweep_id in host.launch.job_manager.active_task_ids(sweep_key(sweep_id, scope_id))


async def _sweep_status(host: OpsHost, sweep_id: str, scope_id: str) -> JobStatus:
    """A sweep's job status: its record, its arms' runs, and whether this process still runs it.

    Args:
        host: The host whose job manager runs the sweep.
        sweep_id: The sweep.
        scope_id: The caller's scope.

    Returns:
        The status, its progress counting the arms launched and finished.

    Raises:
        NotFoundError: No sweep with that id in the scope.
    """
    eval_host = host.eval_host
    live = _live_sweep(host, sweep_id, scope_id)
    sweep = await run_blocking(eval_host.blocking_executor, eval_host.storage.load_sweep, sweep_id, scope_id)
    if sweep is None:
        raise NotFoundError("sweep", sweep_id)
    run_ids = [arm.run_id for arm in sweep.arms if arm.run_id is not None]
    runs = await run_blocking(eval_host.blocking_executor, eval_host.storage.load_eval_runs, run_ids, scope_id)
    state: JobState = "running" if sweep.outcome == "running" else sweep.outcome
    detail = sweep.detail
    if state == "running" and not live:
        state = "lost"
        detail = (
            "the sweep reads running but nothing in this process is running it — the process that started it ended; "
            "its launched arms are campaign members, and no further arm will be launched"
        )
    return JobStatus(
        job_id=sweep_job_id(sweep_id),
        kind="sweep",
        state=state,
        status=sweep.outcome,
        done=state in TERMINAL_JOB_STATES,
        progress=sweep_progress(sweep, {run.id: run for run in runs}),
        campaign_id=sweep.campaign_id,
        sweep_id=sweep.id,
        detail=detail if state != "completed" else None,
    )


def judge_job_id(job_id: str) -> str:
    """The job id of a judge job: its record's id."""
    return f"{JUDGE_JOB_PREFIX}{job_id}"


def judge_key(job_id: str, scope_id: str) -> str:
    """The task key a judge job runs under: the scope is in it, so only its own scope sees it live."""
    return f"eval-judge:{scope_id}:{job_id}"


def _live_judge(host: OpsHost, job_id: str, scope_id: str) -> bool:
    """Whether this process runs judge job ``job_id`` in ``scope_id`` — another scope's job reads as none."""
    return job_id in host.launch.job_manager.active_task_ids(judge_key(job_id, scope_id))


def judge_progress(record: EvalJudgeJob) -> dict[str, Any]:
    """A judge job's progress: the results asked of those admitted, and the calls made of the most admitted.

    Args:
        record: Its record.

    Returns:
        ``{"results_total", "results_done", "max_calls", "calls_made"}``.
    """
    return {
        "results_total": record.results_total,
        "results_done": record.results_done,
        "max_calls": record.max_calls,
        "calls_made": record.calls_made,
    }


async def _judge_status(host: OpsHost, job_id: str, scope_id: str) -> JobStatus:
    """A judge job's status: its record, whether this process still runs it, and once it has ended, its report.

    Args:
        host: The host whose job manager runs the job and whose store holds its record and the run's results.
        job_id: The record's id.
        scope_id: The caller's scope.

    Returns:
        The status; an ended job carries the report its action names, read off the record by the work's own readers.

    Raises:
        NotFoundError: No judge job with that id in the scope.
    """
    eval_host = host.eval_host
    live = _live_judge(host, job_id, scope_id)
    record = await run_blocking(eval_host.blocking_executor, eval_host.storage.load_judge_job, job_id, scope_id)
    if record is None:
        raise NotFoundError("judge_job", job_id)
    state: JobState = "running" if record.outcome == "running" else record.outcome
    detail = record.detail
    if state == "running" and not live:
        state = "lost"
        detail = (
            "the judge job reads running but nothing in this process is running it — the process that started it "
            "ended; every call it made is on the out-of-run ledger, carrying the run's id"
        )
    done = state in TERMINAL_JOB_STATES
    second: SecondJudgeRead | None = None
    temperature: JudgeTemperatureComparison | None = None
    if done and state != "lost":
        if record.second_judge is not None:
            second = await read_second_judge(eval_host, record.second_judge, scope_id)
        if record.temperature_answers is not None:
            temperature = read_judge_temperatures(record.temperature_answers)
    return JobStatus(
        job_id=judge_job_id(record.id),
        kind="judge",
        state=state,
        status=record.outcome,
        done=done,
        progress=judge_progress(record),
        run_id=record.run_id,
        judge_action=record.action,
        second_judge=second,
        judge_temperature=temperature,
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
        ``("run", run_id, None)``, ``("analysis", campaign_id, attempt_id)``, ``("sweep", sweep_id, None)`` or
        ``("judge", job_id, None)``.

    Raises:
        ValidationFailedError: The id is neither shape — a typed or truncated id, which names no job.
    """
    if job_id.startswith(RUN_JOB_PREFIX) and job_id[len(RUN_JOB_PREFIX) :]:
        return "run", job_id[len(RUN_JOB_PREFIX) :], None
    if job_id.startswith(SWEEP_JOB_PREFIX) and job_id[len(SWEEP_JOB_PREFIX) :]:
        return "sweep", job_id[len(SWEEP_JOB_PREFIX) :], None
    if job_id.startswith(JUDGE_JOB_PREFIX) and job_id[len(JUDGE_JOB_PREFIX) :]:
        return "judge", job_id[len(JUDGE_JOB_PREFIX) :], None
    if job_id.startswith(ANALYSIS_JOB_PREFIX):
        # The attempt id is a uuid and holds no colon, so the LAST colon splits it from a campaign id
        # that might hold one.
        campaign_id, colon, attempt_id = job_id[len(ANALYSIS_JOB_PREFIX) :].rpartition(":")
        if colon and campaign_id and attempt_id:
            return "analysis", campaign_id, attempt_id
    raise ValidationFailedError(
        f"job id {job_id!r} names no job: a job id is 'run:<run id>', 'analysis:<campaign id>:<attempt id>', "
        "'sweep:<sweep id>' or 'judge:<job id>', exactly as the start returned it"
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
        NotFoundError: A run job whose run, a sweep job whose sweep, or a judge job whose record is not in the scope.
    """
    kind, target_id, attempt_id = parse_job_id(job_id)
    manager, eval_host = host.launch.job_manager, host.eval_host
    if kind == "run":
        live = manager.is_active(target_id)
        run = await run_blocking(eval_host.blocking_executor, get_run, eval_host.storage, target_id, scope_id)
        return _run_status(run, live=live)
    if kind == "sweep":
        return await _sweep_status(host, target_id, scope_id)
    if kind == "judge":
        return await _judge_status(host, target_id, scope_id)
    assert attempt_id is not None  # parse_job_id names an attempt for every analysis job
    if _live_generation(host, target_id, attempt_id, scope_id):
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
                "started it ended mid-generation, before writing its attempt record. Each model call it made is "
                "on the out-of-run spend ledger (purpose 'analysis', carrying this campaign's id), which "
                "scope_out_of_run_spend reads"
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
        ValidationFailedError: The id names no job, the job has already ended, or it is a generation, sweep or judge
            job no task runs under the scope the caller names (another scope's job reads as none).
        NotFoundError: A run job whose run, a sweep job whose sweep, or a judge job whose record is not in the scope.
        ConflictError: An abandoned run's repair lost its write race; nothing was repaired.
    """
    kind, target_id, attempt_id = parse_job_id(job_id)
    manager, eval_host = host.launch.job_manager, host.eval_host
    if kind == "run":
        # cancel_run's halves, split so its store calls run on the blocking executor and only the job
        # manager — which belongs to this loop — is asked on the loop.
        run = await run_blocking(eval_host.blocking_executor, get_run, eval_host.storage, target_id, scope_id)
        require_cancellable(run)
        if not manager.cancel_job(target_id, reason=reason):
            await run_blocking(
                eval_host.blocking_executor, repair_abandoned_run, eval_host.storage, target_id, scope_id, reason=reason
            )
        return await job_poll(host, job_id, scope_id)
    if kind == "sweep":
        if not _live_sweep(host, target_id, scope_id) or not manager.cancel_task(target_id):
            status = await job_poll(host, job_id, scope_id)
            raise ValidationFailedError(
                f"sweep job {job_id!r} is not running here (it reads {status.state}); there is nothing to cancel"
            )
        return await job_poll(host, job_id, scope_id)
    if kind == "judge":
        # The job stops before its next judge call and records what it did; every call already made stays on the
        # out-of-run ledger.
        if not _live_judge(host, target_id, scope_id) or not manager.cancel_task(target_id):
            status = await job_poll(host, job_id, scope_id)
            raise ValidationFailedError(
                f"judge job {job_id!r} is not running here (it reads {status.state}); there is nothing to cancel"
            )
        return await job_poll(host, job_id, scope_id)
    assert attempt_id is not None  # parse_job_id names an attempt for every analysis job
    if not _live_generation(host, target_id, attempt_id, scope_id) or not manager.cancel_task(attempt_id):
        status = await job_poll(host, job_id, scope_id)
        raise ValidationFailedError(
            f"analysis job {job_id!r} is not running here (it reads {status.state}); there is nothing to cancel"
        )
    return await job_poll(host, job_id, scope_id)


__all__ = [
    "ANALYSIS_JOB_PREFIX",
    "JUDGE_JOB_PREFIX",
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
    "judge_job_id",
    "judge_key",
    "judge_progress",
    "parse_job_id",
    "job_poll",
    "run_job_id",
    "sweep_job_id",
    "sweep_key",
    "sweep_progress",
]
