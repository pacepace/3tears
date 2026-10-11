"""A launch as one job: its arms prepared, its runs started and waited for, under one record.

A launch spends before its runs exist: a launch that generates its cases (``n_variations``) asks the variation
writer once per ``llm`` axis, and every arm's launcher builds its clients, before the group starts. A transport
call that did that waited on a model. So a launch is split where :func:`~threetears.evals.run.admit_launch` splits
it — every refusal made, the admission ceiling's room reserved and every arm priced in the call — and the rest runs
as this job (:func:`launch_job_work`): the launchers, the start, and the runs to their end.

**The record is the launch.** :class:`EvalLaunch` is written as the job starts (``running``, carrying its lease),
again the moment its runs start (``run_ids``, in arm order), and as it ends: ``completed`` once every run has ended,
``failed`` when an arm was refused at its launcher or a run ended ``failed``, ``cancelled`` when it was asked to stop
— which stops the runs it started — and ``timed_out`` when its preparation outran :data:`LAUNCH_PREPARATION_BUDGET_S`.

**Its budget is its preparation's.** The launchers' work — a generating launch's paid calls, each arm's clients —
is bounded by :data:`LAUNCH_PREPARATION_BUDGET_S`; the wait for its runs is not bounded by the launch at all. Each
run is bounded by its own budget from the moment it holds a concurrency slot, and every run holding one is bounded
the same way, so the wait ends; a launch budget over the wait counted the time its runs spent queued behind other
launches, and cancelled healthy runs that had not yet started. Its id is the launch group's, which every run it starts carries as
``launch_group_id``, so the out-of-run ledger's rows for its case generation are found by it too.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import Field, field_validator

from threetears.evals.kernel.job_leases import LeasedJobKind
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.run.job_work import JobOutcome, JobRecorder, JobWork, end_record, leased_job_kind
from threetears.evals.run.jobs import JOB_TIMEOUT_CAP_S
from threetears.evals.run.launch import AdmittedLaunch, LaunchHost
from threetears.evals.schema.base import CoreDocumentModel
from threetears.evals.schema.job_lease import CancelRequestedAtField, CancelRequestReasonField, LeaseField, LostField
from threetears.evals.schema.models import CoreSchemaVersion, utc_now_iso
from threetears.evals.schema.versioning import CORE_SCHEMA_VERSION

#: The reason recorded on a run its launch's cancel stopped.
LAUNCH_CANCEL_REASON = "its launch was cancelled"

#: A launch's wall-clock ceiling on its preparation — its launchers, a generating launch's calls among them — in
#: seconds: the engine's cap on one run's job, a backstop against a launcher or a provider that hangs. The runs it
#: starts are bounded by their own budgets, never by this.
LAUNCH_PREPARATION_BUDGET_S = JOB_TIMEOUT_CAP_S


class EvalLaunch(CoreDocumentModel):
    """One launch of a template's arms, run as a job: what it named, the runs it started, and how it ended.

    Written by the launch's job — as it starts, as its runs start, and as it ends — conditionally, so a record another
    process has written lost or asked to cancel is never overwritten. Its id is the launch group's.

    **Kept** (:mod:`threetears.evals.schema.versioning`): it records what nothing stored can make again, so every
    build reads it, upgraded across core versions and never dropped.
    """

    id: str = Field(min_length=1, description="The launch group's id, which every run it starts carries.")
    doc_type: Literal["eval_launch"] = "eval_launch"
    schema_version: CoreSchemaVersion = CORE_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    template_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    models: list[str] = Field(
        description="The candidate models the launch named, one arm each; empty runs one arm on the kind's default."
    )
    k_runs: int = Field(ge=1, description="Repeats of every case.")
    n_variations: int = Field(ge=0, description="New cases the launch generates before its runs start; 0 for none.")
    run_ids: list[str] = Field(
        default_factory=list, description="The runs the launch started, in arm order; empty until they start."
    )
    outcome: JobOutcome = "running"
    detail: str | None = Field(default=None, description="Why it ended other than completed.")
    created_at: str = Field(default_factory=utc_now_iso)
    ended_at: str | None = None
    lease: LeaseField = None
    cancel_requested_at: CancelRequestedAtField = None
    cancel_request_reason: CancelRequestReasonField = None
    lost: LostField = None

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_launch":
            raise ValueError(f"doc_type must be 'eval_launch', got '{v}'")
        return v


#: A launch's record.
LAUNCH_JOB: LeasedJobKind[EvalLaunch] = leased_job_kind("launch", EvalLaunch, named_runs=lambda launch: launch.run_ids)


def launch_job_work(
    host: LaunchHost,
    admitted: AdmittedLaunch,
    *,
    template_id: str,
    subject_id: str,
    models: list[str],
    k_runs: int,
    n_variations: int,
    scope_id: str,
) -> JobWork[EvalLaunch]:
    """An admitted launch as work: its launchers run, its runs started together and waited for to their end.

    Args:
        host: The launching host, whose job manager runs the runs.
        admitted: The launch as :func:`~threetears.evals.run.admit_launch` admitted it.
        template_id: The template it launches.
        subject_id: The subject it measures.
        models: The models it named, one arm each.
        k_runs: Repeats of every case.
        n_variations: Cases it generates first.
        scope_id: The scope its runs live in.

    Returns:
        The work; its record is the launch, its id the launch group's.
    """
    manager = host.job_manager
    record = EvalLaunch(
        id=admitted.group.id,
        scope_id=scope_id,
        template_id=template_id,
        subject_id=subject_id,
        models=list(models),
        k_runs=k_runs,
        n_variations=n_variations,
    )

    async def launch(recorder: JobRecorder[EvalLaunch]) -> EvalLaunch:
        async with host.job_timeout_factory(LAUNCH_PREPARATION_BUDGET_S):
            runs = await admitted.start()
        run_ids = [run.id for run in runs]
        try:
            await recorder.save(recorder.record.model_copy(update={"run_ids": run_ids}))
            await manager.wait_for(run_ids)
        except BaseException:
            # Cancelled, fenced off, or the progress write failed: the runs it started stop with it, each
            # recording its own ending, rather than run on under a launch that has ended.
            for run_id in run_ids:
                manager.cancel_job(run_id, reason=LAUNCH_CANCEL_REASON)
            await asyncio.shield(manager.wait_for(run_ids))
            raise
        eval_host = host.eval_host
        ended = await run_blocking(eval_host.blocking_executor, eval_host.storage.load_eval_runs, run_ids, scope_id)
        failed = [run.id for run in ended if run.status == "failed"]
        if failed:
            return end_record(recorder.record, "failed", f"run(s) {', '.join(failed)} ended failed")
        return end_record(recorder.record, "completed")

    # No budget over the whole: the preparation bounds itself above, and each run its own wait (see the module).
    return JobWork(kind=LAUNCH_JOB, record=record, run=launch, budget_s=None, discard=admitted.discard)


__all__ = ["LAUNCH_CANCEL_REASON", "LAUNCH_JOB", "LAUNCH_PREPARATION_BUDGET_S", "EvalLaunch", "launch_job_work"]
