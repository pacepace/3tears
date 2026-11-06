"""Taking an interrupted job up again: a run's, a sweep's or a launch's (:func:`job_resume`).

Above both the job contract (:mod:`~threetears.evals.ops.jobs`) and the sweep's operations
(:mod:`~threetears.evals.ops.sweeps`), which build on that contract, since a resume dispatches to each.
"""

from __future__ import annotations

from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import JobHandle, JobsStarted, launch_job_id, parse_job_id, run_job_id
from threetears.evals.ops.sweeps import sweep_resume
from threetears.evals.run.resume import resume_launch, resume_run

__all__ = ["job_resume"]


async def job_resume(host: OpsHost, job_id: str, scope_id: str, *, max_cost_usd: float | None = None) -> JobsStarted:
    """Take an interrupted job up again as the same job, and start it: a run's, a sweep's or a launch's.

    A run that was lost, cancelled by an operator, or stopped by its budget or its provider account is taken up again
    (:func:`~threetears.evals.run.resume_run`); its cells not yet measured run as its job, polled by the run's own job
    id. A sweep (``sweep:<id>``) or a launch (``launch:<id>``) that was lost, cancelled, or stopped by its own time
    budget is taken up as a whole: each arm's unfinished run is taken up again, each arm of a sweep that never
    launched is launched, and the arm schedule is kept — an interleaved sweep's or launch's arms run in one group under
    its original launch group, so they take their cells in turn in the order they always had
    (:func:`~threetears.evals.ops.sweeps.sweep_resume`, :func:`~threetears.evals.run.resume.resume_launch`). Its job is
    polled by its own id, and reads ``completed`` once every arm has.

    Everything that could refuse it is decided in the call, before anything is spent: a job that completed, failed or
    is still live is refused, as is a run its budget stopped at its cap unless ``max_cost_usd`` is above what it
    spent, and one a launch of which would now measure otherwise than it did — the refusal names every difference,
    and for a sweep or a launch, every arm's. Any other kind of job is started again by the action that started it.

    :param host: The host whose launcher wires the runs again and whose job manager runs them.
    :ptype host: OpsHost
    :param job_id: The job's id, as its start returned it or ``resumable_jobs`` lists it.
    :ptype job_id: str
    :param scope_id: The scope the job lives in.
    :ptype scope_id: str
    :param max_cost_usd: The cost cap to resume under — every arm's, for a sweep or a launch — above what was spent;
        ``None`` keeps each run's recorded cap. Not part of the measurement: each holding records the cap it ran
        under.
    :ptype max_cost_usd: float | None
    :return: The job taken up (``run:<id>``, ``sweep:<id>`` or ``launch:<id>``).
    :rtype: JobsStarted
    :raises ValidationFailedError: The id names no job, or a job of a kind not resumed; the job cannot be taken up, or
        would measure otherwise now (every difference named); or a launch of a run would be refused now.
    :raises NotFoundError: No such job in the scope, or a template is gone.
    :raises ConflictError: Another resume of it won, or another writer moved it, while this one prepared; nothing
        was started.
    :raises AdmissionRefusedError: The host has no room for the runs.
    """
    kind, target_id, _ = parse_job_id(job_id)
    if kind == "sweep":
        return await sweep_resume(host, target_id, scope_id, max_cost_usd=max_cost_usd)
    if kind == "launch":
        launch = await resume_launch(host.launch, target_id, scope_id, max_cost_usd=max_cost_usd)
        label = ", ".join(launch.models) or "the kind's default model"
        return JobsStarted(
            jobs=[JobHandle(job_id=launch_job_id(launch.id), kind="launch", target_id=launch.template_id, label=label)]
        )
    if kind != "run":
        raise ValidationFailedError(
            f"{kind} job {job_id!r} is not resumed; only a run's, a sweep's or a launch's job is",
            remedy="start the work again with the action that started it",
        )
    run = await resume_run(host.launch, target_id, scope_id, max_cost_usd=max_cost_usd)
    return JobsStarted(
        jobs=[JobHandle(job_id=run_job_id(run.id), kind="run", target_id=run.id, label=run.candidate_model)]
    )
