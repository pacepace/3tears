"""The engine's own actions, each a thin binding of one operation to its parameters, class and rendering.

The actions are named as the operations they call (:mod:`threetears.evals.ops`), and each parameter is
declared once, as an annotated type below, so a name means one thing on every action that takes it —
which :meth:`~threetears.evals.actions.catalogue.ActionCatalogue.mount` holds.

The scope is never a parameter: it is the :class:`~threetears.evals.actions.catalogue.Caller`'s, which
the host resolves for every call. A synchronous operation runs on the host's blocking executor, so a
storage round-trip never stalls the loop a transport serves calls on.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from threetears.evals.actions import render
from threetears.evals.actions.catalogue import Action, ActionCatalogue, Caller
from threetears.evals.contracts import EvalRunStatus
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.ops import (
    AnalysisDeleted,
    AnalysisListing,
    CampaignDefinition,
    CampaignLine,
    CampaignListing,
    EvalSummary,
    JobsStarted,
    JobStatus,
    LaunchArguments,
    OpsHost,
    ReportDocument,
    RunDeleted,
    RunLine,
    RunListing,
    TemplateListing,
    analyses_list,
    analysis_delete,
    analysis_generate,
    campaign_archive,
    campaign_create,
    campaigns_list,
    job_cancel,
    job_poll,
    report_read,
    run_archive,
    run_delete,
    run_get,
    run_launch,
    runs_list,
    templates_list,
)
from threetears.evals.run import run_blocking

# --- the parameters, each declared once ---------------------------------------------------------------

RunId = Annotated[str, Field(min_length=1, description="A run's id, as runs_list or a launch's job names it.")]
TemplateId = Annotated[str, Field(min_length=1, description="A template's id, as templates_list names it.")]
SubjectId = Annotated[str, Field(min_length=1, description="The subject the runs measure, as the host names it.")]
CampaignId = Annotated[str, Field(min_length=1, description="A campaign's id, as campaigns_list names it.")]
AnalysisId = Annotated[str, Field(min_length=1, description="A stored analysis's id, as analyses_list names it.")]
JobId = Annotated[
    str, Field(min_length=1, description="A job's id, exactly as the action that started it returned it.")
]
Models = Annotated[
    list[str], Field(description="Candidate models, one arm and one run each; empty runs the kind's own default.")
]
KRuns = Annotated[int, Field(ge=1, description="Repeats of every case, for pass^k.")]
Overlays = Annotated[
    dict[str, Any] | None, Field(description="The knobs this launch turns on the template's kind, by field.")
]
MaxCostUsd = Annotated[
    float | None, Field(gt=0, description="A per-run cost cap in dollars, in place of the host default.")
]
JudgeModel = Annotated[str | None, Field(description="The judge model, where the kind is model-judged.")]
SimulatorModel = Annotated[str | None, Field(description="The simulated user's model, where the kind has one.")]
RunStatus = Annotated[EvalRunStatus | None, Field(description="List only runs with this stored status.")]
IncludeArchived = Annotated[bool, Field(description="List archived records too; they are left out by default.")]
Archived = Annotated[bool, Field(description="The state to set: true retires the record, false restores it.")]
Reason = Annotated[str | None, Field(description="Why, recorded on a cancelled run.")]
Confirm = Annotated[str, Field(description="Must echo the id of what is destroyed, exactly.")]
Name = Annotated[str, Field(min_length=1, description="The campaign's name, as an operator reads it.")]
Behavior = Annotated[str, Field(min_length=1, description="Which aspect of the subject is under test.")]
Description = Annotated[str, Field(description="A longer description of the campaign.")]
RunIds = Annotated[list[str], Field(description="The runs to put in the campaign, all in the caller's scope.")]
GeneratorModel = Annotated[str | None, Field(description="The analysis generator's model; omitted for the host's.")]
Format = Annotated[
    Literal["markdown", "json", "html"],
    Field(description="The report's form: markdown (the memo), json (the schema's form), html (script-free)."),
]


class NoParams(EvalBaseModel):
    """An action that takes nothing but the caller's scope."""


class RunsListParams(EvalBaseModel):
    """``runs_list``."""

    status: RunStatus = None
    include_archived: IncludeArchived = False


class RunParams(EvalBaseModel):
    """An action over one run."""

    run_id: RunId


class RunLaunchParams(EvalBaseModel):
    """``run_launch``."""

    template_id: TemplateId
    subject_id: SubjectId
    models: Models = Field(default_factory=list)
    k_runs: KRuns = 1
    overlays: Overlays = None
    max_cost_usd: MaxCostUsd = None
    judge_model: JudgeModel = None
    simulator_model: SimulatorModel = None


class JobParams(EvalBaseModel):
    """``job_poll``."""

    job_id: JobId


class JobCancelParams(EvalBaseModel):
    """``job_cancel``."""

    job_id: JobId
    reason: Reason = None


class RunArchiveParams(EvalBaseModel):
    """``run_archive``."""

    run_id: RunId
    archived: Archived = True


class CampaignsListParams(EvalBaseModel):
    """``campaigns_list``."""

    include_archived: IncludeArchived = False


class CampaignCreateParams(EvalBaseModel):
    """``campaign_create``."""

    name: Name
    subject_id: SubjectId
    behavior: Behavior
    description: Description = ""
    run_ids: RunIds = Field(default_factory=list)


class CampaignArchiveParams(EvalBaseModel):
    """``campaign_archive``."""

    campaign_id: CampaignId
    archived: Archived = True


class CampaignParams(EvalBaseModel):
    """An action over one campaign."""

    campaign_id: CampaignId


class AnalysisGenerateParams(EvalBaseModel):
    """``analysis_generate``."""

    campaign_id: CampaignId
    model: GeneratorModel = None


class ReportReadParams(EvalBaseModel):
    """``report_read``."""

    campaign_id: CampaignId
    format: Format = "markdown"


class RunDeleteParams(EvalBaseModel):
    """``run_delete``."""

    run_id: RunId
    confirm: Confirm


class AnalysisDeleteParams(EvalBaseModel):
    """``analysis_delete``."""

    analysis_id: AnalysisId
    confirm: Confirm


# --- the handlers ------------------------------------------------------------------------------------


async def _templates_list(host: OpsHost, caller: Caller, _: NoParams) -> TemplateListing:
    return await run_blocking(host.eval_host.blocking_executor, templates_list, host.eval_host, caller.scope_id)


async def _runs_list(host: OpsHost, caller: Caller, params: RunsListParams) -> RunListing:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        runs_list,
        eval_host,
        caller.scope_id,
        status=params.status,
        include_archived=params.include_archived,
    )


async def _run_get(host: OpsHost, caller: Caller, params: RunParams) -> EvalSummary:
    return await run_blocking(host.eval_host.blocking_executor, run_get, host.eval_host, params.run_id, caller.scope_id)


async def _run_launch(host: OpsHost, caller: Caller, params: RunLaunchParams) -> JobsStarted:
    return await run_launch(host, LaunchArguments.model_validate(params.model_dump()), caller.scope_id)


async def _job_poll(host: OpsHost, caller: Caller, params: JobParams) -> JobStatus:
    return await job_poll(host, params.job_id, caller.scope_id)


async def _job_cancel(host: OpsHost, caller: Caller, params: JobCancelParams) -> JobStatus:
    return await job_cancel(host, params.job_id, caller.scope_id, reason=params.reason)


async def _run_archive(host: OpsHost, caller: Caller, params: RunArchiveParams) -> RunLine:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor, run_archive, eval_host, params.run_id, caller.scope_id, archived=params.archived
    )


async def _campaigns_list(host: OpsHost, caller: Caller, params: CampaignsListParams) -> CampaignListing:
    eval_host = host.eval_host
    archived = None if params.include_archived else False
    return await run_blocking(
        eval_host.blocking_executor, campaigns_list, eval_host, caller.scope_id, archived=archived
    )


async def _campaign_create(host: OpsHost, caller: Caller, params: CampaignCreateParams) -> CampaignLine:
    eval_host = host.eval_host
    definition = CampaignDefinition.model_validate(params.model_dump())
    return await run_blocking(
        eval_host.blocking_executor,
        campaign_create,
        eval_host,
        definition,
        caller.scope_id,
        created_by=caller.identity,
    )


async def _campaign_archive(host: OpsHost, caller: Caller, params: CampaignArchiveParams) -> CampaignLine:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        campaign_archive,
        eval_host,
        params.campaign_id,
        caller.scope_id,
        archived=params.archived,
    )


async def _analyses_list(host: OpsHost, caller: Caller, params: CampaignParams) -> AnalysisListing:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor, analyses_list, eval_host, params.campaign_id, caller.scope_id
    )


async def _analysis_generate(host: OpsHost, caller: Caller, params: AnalysisGenerateParams) -> JobsStarted:
    return await analysis_generate(host, params.campaign_id, caller.scope_id, model=params.model)


async def _report_read(host: OpsHost, caller: Caller, params: ReportReadParams) -> ReportDocument:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        report_read,
        eval_host,
        params.campaign_id,
        caller.scope_id,
        format=params.format,
    )


async def _run_delete(host: OpsHost, caller: Caller, params: RunDeleteParams) -> RunDeleted:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor, run_delete, eval_host, params.run_id, caller.scope_id, confirm=params.confirm
    )


async def _analysis_delete(host: OpsHost, caller: Caller, params: AnalysisDeleteParams) -> AnalysisDeleted:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        analysis_delete,
        eval_host,
        params.analysis_id,
        caller.scope_id,
        confirm=params.confirm,
    )


# --- the catalogue -----------------------------------------------------------------------------------

#: The help index's groups, in the order a piece of work meets them.
DISCOVER = "Find what is there"
RUN = "Run and watch"
ANALYSE = "Analyse and report"
CURATE = "Curate"


def engine_actions() -> tuple[Action, ...]:
    """The engine's actions, in the order help lists them.

    Returns:
        The actions.
    """
    run_id, campaign_id, analysis_id = "0193a1b2-run", "0193a1b2-campaign", "0193a1b2-analysis"
    return (
        Action(
            name="templates_list",
            summary="List the scope's active templates — what a launch can run.",
            workflow=DISCOVER,
            permission="read",
            params=NoParams,
            result=TemplateListing,
            handler=_templates_list,
            render=render.render_templates,
            example={},
        ),
        Action(
            name="runs_list",
            summary="List the scope's runs, newest first.",
            workflow=DISCOVER,
            permission="read",
            params=RunsListParams,
            result=RunListing,
            handler=_runs_list,
            render=render.render_runs,
            example={"status": "completed"},
        ),
        Action(
            name="campaigns_list",
            summary="List the scope's campaigns — the sets of runs an analysis reads.",
            workflow=DISCOVER,
            permission="read",
            params=CampaignsListParams,
            result=CampaignListing,
            handler=_campaigns_list,
            render=render.render_campaigns,
            example={},
        ),
        Action(
            name="run_launch",
            summary="Launch a template's runs, one per model, each as a job to poll.",
            workflow=RUN,
            permission="spend",
            params=RunLaunchParams,
            result=JobsStarted,
            handler=_run_launch,
            render=render.render_jobs_started,
            example={"template_id": "tmpl-1", "subject_id": "subject-1", "models": ["model-a", "model-b"]},
            long_running=True,
            detail=(
                "Every run spends against its cost cap (the host's, or max_cost_usd when lower). A launch the "
                "process cannot admit, or the template's kind cannot honour, is refused before anything starts."
            ),
        ),
        Action(
            name="job_poll",
            summary="Read where a job stands — a launched run or an analysis generation.",
            workflow=RUN,
            permission="read",
            params=JobParams,
            result=JobStatus,
            handler=_job_poll,
            render=render.render_job_status,
            example={"job_id": f"run:{run_id}"},
        ),
        Action(
            name="job_cancel",
            summary="Ask a running job to stop; poll to see it land as cancelled.",
            workflow=RUN,
            permission="write",
            params=JobCancelParams,
            result=JobStatus,
            handler=_job_cancel,
            render=render.render_job_status,
            example={"job_id": f"run:{run_id}", "reason": "wrong template"},
        ),
        Action(
            name="run_get",
            summary="Summarise one run: how it ended, how its results came out, each measure's mean.",
            workflow=RUN,
            permission="read",
            params=RunParams,
            result=EvalSummary,
            handler=_run_get,
            render=render.render_summary,
            example={"run_id": run_id},
        ),
        Action(
            name="campaign_create",
            summary="Create a campaign over runs in the scope, for an analysis to read.",
            workflow=ANALYSE,
            permission="write",
            params=CampaignCreateParams,
            result=CampaignLine,
            handler=_campaign_create,
            render=render.render_campaign,
            example={"name": "model bake-off", "subject_id": "subject-1", "behavior": "accuracy", "run_ids": [run_id]},
        ),
        Action(
            name="analysis_generate",
            summary="Generate a campaign's analysis with a paid model call, as a job to poll.",
            workflow=ANALYSE,
            permission="spend",
            params=AnalysisGenerateParams,
            result=JobsStarted,
            handler=_analysis_generate,
            render=render.render_jobs_started,
            example={"campaign_id": campaign_id},
            long_running=True,
            detail=(
                "Refused before any spend when the campaign has no evidence or a generation of it is already "
                "running. The job's record is the generation attempt, written however it ends."
            ),
        ),
        Action(
            name="analyses_list",
            summary="List a campaign's stored analyses.",
            workflow=ANALYSE,
            permission="read",
            params=CampaignParams,
            result=AnalysisListing,
            handler=_analyses_list,
            render=render.render_analyses,
            example={"campaign_id": campaign_id},
        ),
        Action(
            name="report_read",
            summary="Read a campaign's report — its analysis, else its evidence alone — as Markdown, JSON or HTML.",
            workflow=ANALYSE,
            permission="read",
            params=ReportReadParams,
            result=ReportDocument,
            handler=_report_read,
            render=render.render_report,
            example={"campaign_id": campaign_id, "format": "markdown"},
        ),
        Action(
            name="run_archive",
            summary="Archive a run (or restore it): out of every cohort, nothing destroyed.",
            workflow=CURATE,
            permission="write",
            params=RunArchiveParams,
            result=RunLine,
            handler=_run_archive,
            render=render.render_run_line,
            example={"run_id": run_id, "archived": True},
        ),
        Action(
            name="campaign_archive",
            summary="Archive a campaign (or restore it): out of listings, nothing destroyed.",
            workflow=CURATE,
            permission="write",
            params=CampaignArchiveParams,
            result=CampaignLine,
            handler=_campaign_archive,
            render=render.render_campaign,
            example={"campaign_id": campaign_id, "archived": True},
        ),
        Action(
            name="run_delete",
            summary="Destroy a run, its results and its campaign memberships. Unrecoverable; archive instead.",
            workflow=CURATE,
            permission="destructive",
            params=RunDeleteParams,
            result=RunDeleted,
            handler=_run_delete,
            render=render.render_run_deleted,
            example={"run_id": run_id, "confirm": run_id},
        ),
        Action(
            name="analysis_delete",
            summary="Destroy a stored analysis; the insights it minted remain. Unrecoverable; archive instead.",
            workflow=CURATE,
            permission="destructive",
            params=AnalysisDeleteParams,
            result=AnalysisDeleted,
            handler=_analysis_delete,
            render=render.render_analysis_deleted,
            example={"analysis_id": analysis_id, "confirm": analysis_id},
        ),
    )


def eval_catalogue(host_actions: tuple[Action, ...] = ()) -> ActionCatalogue:
    """The engine's catalogue, with the host's own actions after the engine's.

    Args:
        host_actions: Actions the host contributes — its kinds' own, its curation — each named, classed and
            rendered like the engine's.

    Returns:
        The catalogue.

    Raises:
        ValueError: A host action takes a name the engine's catalogue already has.
    """
    return ActionCatalogue(engine_actions()).extended(host_actions)


__all__ = ["ANALYSE", "CURATE", "DISCOVER", "RUN", "engine_actions", "eval_catalogue"]
