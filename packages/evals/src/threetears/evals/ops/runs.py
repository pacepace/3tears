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
from threetears.evals.contracts.models import DEFAULT_LAUNCH_K_RUNS, EvalRun
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import JobHandle, JobsStarted, run_job_id
from threetears.evals.ops.summary import EvalSummary, summarize_run
from threetears.evals.run.authoring import list_templates
from threetears.evals.run.curation import delete_run, set_run_archived
from threetears.evals.run.launch import start_run
from threetears.evals.run.lifecycle import get_run
from threetears.evals.run.ratings import rate_result
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
    """What a launch names: the template, the subject, one arm per model, and the run's own limits.

    The one declaration of a launch's arguments — their types, bounds and descriptions. The ``run_launch``
    and ``launch_estimate`` actions' parameters derive from it, so the operation and what an agent is
    offered cannot drift apart.
    """

    template_id: str = Field(min_length=1, description="A template's id, as templates_list names it.")
    subject_id: str = Field(min_length=1, description="The subject the runs measure, as the host names it.")
    models: list[str] = Field(
        default_factory=list,
        description="Candidate models, one arm and one run each; empty runs the kind's own default.",
    )
    k_runs: int = Field(default=DEFAULT_LAUNCH_K_RUNS, ge=1, description="Repeats of every case, for pass^k.")
    n_variations: int = Field(
        default=0, ge=0, description="New cases to generate from the template's variation axes; 0 runs its stored cases."
    )
    variation_model: str | None = Field(
        default=None,
        description="The model that writes the template's llm variation axes' values; required when n_variations "
        "generates for such an axis, refused otherwise.",
    )
    overlays: dict[str, Any] | None = Field(
        default=None, description="The knobs this launch turns on the template's kind, by field."
    )
    apparatus_settings: dict[str, Any] | None = Field(
        default=None,
        description="Host-declared apparatus values to set the runs' rig up with, by apparatus dimension (e.g. who sits "
        "in an adjudicator's seat) — each a string, a bool or a number, and one the template's kind reads; refused "
        "otherwise. Recorded on every run and part of its measurement context, so one template can be compared at two.",
    )
    max_cost_usd: float | None = Field(
        default=None,
        gt=0,
        description="A per-run cost cap in dollars, at or below the host's ceiling; it can only lower that ceiling, "
        "and a value above it is refused.",
    )
    judge_model: str | None = Field(default=None, description="The judge model, where the kind is model-judged.")
    simulator_model: str | None = Field(
        default=None, description="The simulated user's model, where the kind has one."
    )


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
        n_variations=arguments.n_variations,
        variation_model=arguments.variation_model,
        overlays=arguments.overlays,
        apparatus_settings=arguments.apparatus_settings,
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


class ResultRated(EvalBaseModel):
    """A rating an agent wrote: what it rated, and that it is an agent's, never read as a person's."""

    rating_id: str
    result_id: str
    rubric_dim: str
    score: int
    rater: str
    rater_kind: str = Field(description="Always `agent` through an action: the agent rated, whatever account it acts for.")


def result_rate(
    host: EvalHost, result_id: str, scope_id: str, *, rubric_dim: str, score: int, reason: str, rater: str
) -> ResultRated:
    """Record an agent's rating of one judged dimension of one result — kept beside people's, never pooled with them.

    The operation an agent-facing surface rates through, so ``rater_kind`` is fixed here rather than taken from
    the caller: an agent writing through a tool is an ``agent`` whatever account it acts for, and only a person's
    rating is judge-versus-human agreement (:func:`~threetears.evals.run.rate_result`). A host recording a
    person's rating calls :func:`~threetears.evals.run.rate_result` with ``rater_kind="person"`` itself.

    Args:
        host: The host whose store holds the result.
        result_id: The result rated.
        scope_id: The scope it lives in.
        rubric_dim: The judged dimension, spelled as the result's score spells it.
        score: The score, on the dimension's scale.
        reason: The agent's own words for the score.
        rater: Who rated, as the calling surface names the agent.

    Returns:
        What was written.

    Raises:
        NotFoundError: No such result in the scope.
        ValidationFailedError: The judge scored no such dimension, or the score is off its scale.
        StorageError: The write failed.
    """
    rating = rate_result(
        host.storage,
        result_id=result_id,
        scope_id=scope_id,
        rubric_dim=rubric_dim,
        rater=rater,
        rater_kind="agent",
        score=score,
        reason=reason,
    )
    return ResultRated(
        rating_id=rating.id,
        result_id=rating.result_id,
        rubric_dim=rating.rubric_dim,
        score=rating.score,
        rater=rating.rater,
        rater_kind=rating.rater_kind,
    )


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
    "ResultRated",
    "RunDeleted",
    "RunLine",
    "RunListing",
    "TemplateLine",
    "TemplateListing",
    "run_archive",
    "run_delete",
    "run_get",
    "result_rate",
    "run_launch",
    "runs_list",
    "templates_list",
]
