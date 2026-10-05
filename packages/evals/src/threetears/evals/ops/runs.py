"""The run side's operations: what a scope holds, one run's summary, a launch, and a run's curation.

Each takes the host and the scope and returns a typed value — never a ``dict`` a surface re-types — so
a CLI, an MCP action and a REST route that call one read one shape. A launch is long work and follows
the job contract (:mod:`threetears.evals.ops.jobs`): it returns one job per arm, and each job's record is
its run.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.host import EvalHost
from threetears.evals.contracts.models import EvalRun
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import JobHandle, JobsStarted, run_job_id
from threetears.evals.ops.summary import EvalSummary, summarize_run
from threetears.evals.run.authoring import list_templates
from threetears.evals.run.curation import delete_run, set_run_archived
from threetears.evals.run.launch import start_run
from threetears.evals.run.lifecycle import get_run
from threetears.evals.run.reads import list_runs


class TemplateLine(EvalBaseModel):
    """One template, as a listing shows it."""

    id: str
    name: str
    candidate_kind: str
    intent: str
    archived: bool


class TemplateListing(EvalBaseModel):
    """A scope's templates."""

    templates: list[TemplateLine]


class RunLine(EvalBaseModel):
    """One run, as a listing shows it."""

    id: str
    status: str
    candidate_model: str
    template_id: str | None
    created_at: str
    archived: bool


class RunListing(EvalBaseModel):
    """A scope's runs, newest first as the store lists them."""

    runs: list[RunLine]
    include_archived: bool = Field(description="Whether archived runs were listed; they are left out by default.")


class LaunchArguments(EvalBaseModel):
    """What a launch names: the template, the subject, one arm per model, and the run's own limits."""

    template_id: str
    subject_id: str
    models: list[str] = Field(default_factory=list)
    k_runs: int = 1
    overlays: dict[str, Any] | None = None
    max_cost_usd: float | None = None
    judge_model: str | None = None
    simulator_model: str | None = None


class RunDeleted(EvalBaseModel):
    """What deleting a run removed."""

    run_id: str
    results_deleted: int
    campaigns_detached: list[str]


def _line(run: EvalRun) -> RunLine:
    return RunLine(
        id=run.id,
        status=run.status,
        candidate_model=run.candidate_model,
        template_id=run.template_id,
        created_at=run.created_at,
        archived=run.archived,
    )


def templates_list(host: EvalHost, scope_id: str, *, archived: bool = False) -> TemplateListing:
    """The scope's templates.

    Args:
        host: The host whose store is read.
        scope_id: The scope.
        archived: List the archived templates instead of the active ones.

    Returns:
        The listing.
    """
    return TemplateListing(
        templates=[
            TemplateLine(
                id=template.id,
                name=template.name,
                candidate_kind=template.candidate_kind,
                intent=template.intent,
                archived=template.archived,
            )
            for template in list_templates(host.storage, scope_id, archived=archived)
        ]
    )


def runs_list(
    host: EvalHost, scope_id: str, *, status: str | None = None, include_archived: bool = False
) -> RunListing:
    """The scope's runs.

    Args:
        host: The host whose store is read.
        scope_id: The scope.
        status: Only runs with this stored status.
        include_archived: List archived runs too.

    Returns:
        The listing.

    Raises:
        ValidationFailedError: ``status`` is not a run status.
    """
    runs = list_runs(host, scope_id, status=status, include_archived=include_archived)
    return RunListing(runs=[_line(run) for run in runs], include_archived=include_archived)


def run_get(host: EvalHost, run_id: str, scope_id: str) -> EvalSummary:
    """One run, summarised from its stored results.

    Args:
        host: The host whose store holds the run.
        run_id: The run.
        scope_id: The scope it lives in.

    Returns:
        The summary.

    Raises:
        NotFoundError: No run with that id in the scope.
    """
    return summarize_run(host, run_id, scope_id)


async def run_launch(host: OpsHost, arguments: LaunchArguments, scope_id: str) -> JobsStarted:
    """Launch a template's runs, one per model, and return a job per run.

    Args:
        host: The launching host.
        arguments: What the launch names.
        scope_id: The scope the template is read in and the runs live in.

    Returns:
        One job per launched run, in the order the models were named.

    Raises:
        NotFoundError: The template is not in the scope.
        AdmissionRefusedError: The runs would pass the host's admission ceiling.
        ValidationFailedError: Any refusal :func:`~threetears.evals.run.start_run` makes.
    """
    runs = await start_run(
        host.launch,
        template_id=arguments.template_id,
        subject_id=arguments.subject_id,
        models=list(arguments.models),
        k_runs=arguments.k_runs,
        overlays=arguments.overlays,
        max_cost_usd=arguments.max_cost_usd,
        judge_model=arguments.judge_model,
        simulator_model=arguments.simulator_model,
        scope_id=scope_id,
    )
    return JobsStarted(
        jobs=[
            JobHandle(job_id=run_job_id(run.id), kind="run", target_id=run.id, label=run.candidate_model)
            for run in runs
        ]
    )


def run_archive(host: EvalHost, run_id: str, scope_id: str, *, archived: bool) -> RunLine:
    """Archive or restore a run — reversible exclusion from every cohort.

    Args:
        host: The host whose store holds the run.
        run_id: The run.
        scope_id: The scope it lives in.
        archived: ``True`` retires it, ``False`` restores it.

    Returns:
        The run as persisted.

    Raises:
        NotFoundError: No run with that id in the scope.
    """
    return _line(set_run_archived(host.storage, run_id, scope_id, archived=archived, profile=host.profile))


def run_delete(host: EvalHost, run_id: str, scope_id: str, *, confirm: str | None) -> RunDeleted:
    """Destroy a run, its results and its campaign memberships — unrecoverable; archive is the safe answer.

    Args:
        host: The host whose store holds the run.
        run_id: The run.
        scope_id: The scope it lives in.
        confirm: Must echo ``run_id``.

    Returns:
        What was removed.

    Raises:
        NotFoundError: No run with that id in the scope — refused before ``confirm`` is read.
        ValidationFailedError: ``confirm`` does not echo the id, or the run is still running.
        StorageError: The cascade stopped partway; the message says how far it got.
    """
    removed = delete_run(host.storage, get_run(host.storage, run_id, scope_id), scope_id, confirm=confirm)
    return RunDeleted(
        run_id=removed["run_id"],
        results_deleted=removed["results_deleted"],
        campaigns_detached=list(removed["campaigns_detached"]),
    )


__all__ = [
    "LaunchArguments",
    "RunDeleted",
    "RunLine",
    "RunListing",
    "TemplateLine",
    "TemplateListing",
    "run_archive",
    "run_delete",
    "run_get",
    "run_launch",
    "runs_list",
    "templates_list",
]
