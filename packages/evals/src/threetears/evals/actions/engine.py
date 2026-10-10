"""The engine's own actions, each a thin binding of one operation to its parameters, class and rendering.

The actions are named as the operations they call (:mod:`threetears.evals.ops`), and each parameter is
declared once, as an annotated type below, so a name means one thing on every action that takes it —
which :meth:`~threetears.evals.actions.catalogue.ActionCatalogue.mount` holds.

The scope is never a parameter: it is the :class:`~threetears.evals.actions.catalogue.Caller`'s, which
the host resolves for every call. A synchronous operation runs on the host's blocking executor, so a
storage round-trip never stalls the loop a transport serves calls on.
"""

from __future__ import annotations

from functools import partial
from typing import Annotated, Any, Literal

from pydantic import Field

from threetears.evals.actions import render
from threetears.evals.actions.catalogue import Action, ActionCatalogue, Caller
from threetears.evals.contracts import EvalRunStatus, ResultOutcome
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts import OutOfRunPurpose
from threetears.evals.ops import (
    AnalysisDeleted,
    AnalysisGenerationEstimate,
    AnalysisLine,
    AnalysisListing,
    CampaignDefinition,
    CampaignLine,
    CampaignListing,
    LaunchEstimate,
    EvalSummary,
    FrozenReporterCase,
    FrontierResult,
    HistoryResult,
    JobsStarted,
    JobStatus,
    LaunchArguments,
    OpsHost,
    OutOfRunSpendReport,
    PivotTable,
    ReportDocument,
    ReporterCaseFreeze,
    ReporterCaseListing,
    ResultDetail,
    ResultListing,
    ResultPart,
    ResultRated,
    RunDeleted,
    RunLine,
    RunListing,
    RunsCompared,
    ScoreExport,
    TemplateListing,
    analyses_list,
    analysis_archive,
    analysis_delete,
    analysis_estimate,
    analysis_generate,
    campaign_archive,
    campaign_create,
    campaigns_list,
    job_cancel,
    job_poll,
    launch_estimate,
    report_read,
    reporter_case_archive,
    reporter_case_freeze,
    reporter_cases_list,
    result_get,
    result_rate,
    results_list,
    run_archive,
    run_delete,
    run_get,
    run_launch,
    runs_compare,
    runs_list,
    scope_export,
    scope_frontier,
    scope_history,
    scope_out_of_run_spend,
    scope_pivot,
    templates_list,
)
from threetears.evals.run import run_blocking

# --- the parameters, each declared once ---------------------------------------------------------------

RunId = Annotated[str, Field(min_length=1, description="A run's id, as runs_list or a launch's job names it.")]
# The launch's own two, taken from its one declaration so an action naming a template or a subject means the
# same thing a launch does.
TemplateId = Annotated[str, LaunchArguments.model_fields["template_id"]]
SubjectId = Annotated[str, LaunchArguments.model_fields["subject_id"]]
ResultId = Annotated[str, Field(min_length=1, description="A result's id, as results_list names it.")]
CampaignId = Annotated[str, Field(min_length=1, description="A campaign's id, as campaigns_list names it.")]
AnalysisId = Annotated[str, Field(min_length=1, description="A stored analysis's id, as analyses_list names it.")]
JobId = Annotated[
    str, Field(min_length=1, description="A job's id, exactly as the action that started it returned it.")
]
RunStatus = Annotated[EvalRunStatus | None, Field(description="List only runs with this stored status.")]
IncludeArchived = Annotated[bool, Field(description="List archived records too; they are left out by default.")]
Archived = Annotated[bool, Field(description="The state to set: true retires the record, false restores it.")]
Reason = Annotated[str | None, Field(description="Why, recorded on a cancelled run.")]
Confirm = Annotated[str, Field(description="Must echo the id of what is destroyed, exactly.")]
ArchiveReason = Annotated[
    str | None,
    Field(
        description="Why the record is archived (an analysis shown false or superseded, a reporter case that can no "
        "longer measure anything); cleared on restore."
    ),
]
ReporterCaseId = Annotated[
    str,
    Field(min_length=1, description="A reporter case's id, as reporter_case_freeze or reporter_cases_list returns it."),
]
Name = Annotated[str, Field(min_length=1, description="The campaign's name, as an operator reads it.")]
Behavior = Annotated[str, Field(min_length=1, description="Which aspect of the subject is under test.")]
Description = Annotated[str, Field(description="A longer description of the campaign.")]
RunIds = Annotated[list[str], Field(description="The runs to put in the campaign, all in the caller's scope.")]
GeneratorModel = Annotated[str | None, Field(description="The analysis generator's model; omitted for the host's.")]
Format = Annotated[
    Literal["markdown", "json", "html"],
    Field(description="The report's form: markdown (the memo), json (the schema's form), html (script-free)."),
]
RowFactor = Annotated[
    str,
    Field(
        min_length=1,
        description="The coordinate the rows are: a declared one (model, template_id, ...) or a lever the host registers.",
    ),
]
ColumnFactor = Annotated[
    str,
    Field(
        min_length=1,
        description="The coordinate the columns are; a pooled ranking across the rows is checked for reversal on it.",
    ),
]
Metric = Annotated[str | None, Field(description="The measure to read; omitted reads the composite score.")]
Weighting = Annotated[
    str | None, Field(description="How a cell averages its observations; omitted takes equal per scenario.")
]
SubjectFilter = Annotated[str | None, Field(description="Read only this subject's runs; omitted reads every subject.")]
RunStatusFilter = Annotated[
    EvalRunStatus | Literal["all"],
    Field(
        description="Read only runs with this status, or 'all'. Completed unless named: a run still going is still "
        "adding results."
    ),
]
PredictedCost = Annotated[
    dict[str, Any] | None,
    Field(
        description="A cost pivot's plan: the structured result launch_estimate returned before these runs. The "
        "cell at each priced arm's model and template then shows its predicted cost beside the cost observed."
    ),
]
MinAbsoluteChange = Annotated[
    float,
    Field(description="The smallest move a regression flag counts, in the measure's unit; 0 lets the test decide."),
]
MinRelativeChange = Annotated[
    float,
    Field(description="The smallest move from the baseline a flag counts, as a fraction; 0 lets the test decide."),
]
ExportFormat = Annotated[
    Literal["csv", "json"],
    Field(
        description="The export's form: csv (flat rows, a column per lever) or json (the rows and what was left out)."
    ),
]
ExportRunIds = Annotated[
    list[str] | None,
    Field(description="Export only these runs, archived ones included since they are named; omitted exports all."),
]
# The condition's values spelled out rather than the enum itself, which would nest a definition in a flat
# parameter list; the handler reads it back as the enum, so a value the enum lacks fails there, loudly.
ConditionFilter = Annotated[
    Literal["ok", "candidate_fail", "infra_exclude"] | None,
    Field(
        description="List only results in this condition: ok (delivered, scored), candidate_fail (scored as a hard "
        "fail) or infra_exclude (a harness fault, in no aggregate)."
    ),
]
#: The rows one ``results_list`` page holds unless the caller asks for fewer, and the most it may ask for: a
#: page is read into an agent's context, and a run of a few hundred cases at k=3 is a thousand rows.
_RESULTS_PAGE_DEFAULT = 50
_RESULTS_PAGE_MAX = 200
Offset = Annotated[int, Field(ge=0, description="Skip this many rows; the previous page's next_offset.")]
Limit = Annotated[
    int, Field(ge=1, le=_RESULTS_PAGE_MAX, description=f"The most rows to return, up to {_RESULTS_PAGE_MAX}.")
]
CaseCount = Annotated[
    int | None,
    Field(ge=1, description="A case count to price each planned arm at in place of its plan's, for a what-if grid."),
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


class ResultsListParams(EvalBaseModel):
    """``results_list``."""

    run_id: RunId
    condition_filter: ConditionFilter = None
    offset: Offset = 0
    limit: Limit = _RESULTS_PAGE_DEFAULT


class ResultGetParams(EvalBaseModel):
    """``result_get``."""

    result_id: ResultId
    part: Annotated[
        ResultPart,
        Field(
            description="Which part of the stored trace to return: record (the output the kind stored, its call "
            "ledger and end state, beside the result), judge (what the judge was sent) or spans (the stored spans)."
        ),
    ] = "record"


class RunLaunchParams(LaunchArguments):
    """``run_launch`` — the launch's own arguments, declared once on :class:`~threetears.evals.ops.LaunchArguments`.

    Derived rather than restated, so a field the operation gains is a parameter the action offers, with the
    one description and the one bound, and cannot silently take its default on every launch an agent makes.
    """


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
    """``campaign_create`` — the operation's own definition, declared design and control run included."""

    name: Name
    subject_id: SubjectId
    behavior: Behavior
    description: Description = ""
    run_ids: RunIds = Field(default_factory=list)
    declared_design: Annotated[dict[str, Any] | None, CampaignDefinition.model_fields["declared_design"]] = None
    control_from_run_id: Annotated[str | None, CampaignDefinition.model_fields["control_from_run_id"]] = None


class CampaignArchiveParams(EvalBaseModel):
    """``campaign_archive``."""

    campaign_id: CampaignId
    archived: Archived = True


class CampaignParams(EvalBaseModel):
    """An action over one campaign."""

    campaign_id: CampaignId


class AnalysisArchiveParams(EvalBaseModel):
    """``analysis_archive``."""

    analysis_id: AnalysisId
    archived: Archived
    archive_reason: ArchiveReason = None


class AnalysisGenerateParams(EvalBaseModel):
    """``analysis_generate`` and ``analysis_estimate``."""

    campaign_id: CampaignId
    model: GeneratorModel = None


class ReportReadParams(EvalBaseModel):
    """``report_read``."""

    campaign_id: CampaignId
    format: Format = "markdown"


class ReporterCaseFreezeParams(ReporterCaseFreeze):
    """``reporter_case_freeze`` — the freeze's own arguments, declared once on :class:`~threetears.evals.ops.ReporterCaseFreeze`.

    Derived rather than restated, as :class:`RunLaunchParams` is, so a field the operation gains is a parameter
    the action offers.
    """


class ReporterCasesListParams(EvalBaseModel):
    """``reporter_cases_list``."""

    template_id: TemplateId
    include_archived: IncludeArchived = False


class ReporterCaseArchiveParams(EvalBaseModel):
    """``reporter_case_archive``."""

    reporter_case_id: ReporterCaseId
    archived: Archived = True
    archive_reason: ArchiveReason = None


class ScopePivotParams(EvalBaseModel):
    """``scope_pivot``."""

    row_factor: RowFactor
    column_factor: ColumnFactor
    metric: Metric = None
    weighting: Weighting = None
    subject_filter: SubjectFilter = None
    run_status: RunStatusFilter = "completed"
    predicted_cost: PredictedCost = None
    launched_run_ids: Annotated[
        list[str] | None,
        Field(
            description="The runs the estimated launch made (the run ids its jobs name), with predicted_cost: each "
            "predicted cell then says how many of its observations came from other runs."
        ),
    ] = None


class RunsCompareParams(EvalBaseModel):
    """``runs_compare``."""

    baseline_run_id: Annotated[
        str, Field(min_length=1, description="The run read as the baseline (A), as runs_list names it.")
    ]
    candidate_run_id: Annotated[
        str, Field(min_length=1, description="The run read against the baseline (B), as runs_list names it.")
    ]


class ScopeHistoryParams(EvalBaseModel):
    """``scope_history``."""

    metric: Metric = None
    min_absolute_change: MinAbsoluteChange = 0.0
    min_relative_change: MinRelativeChange = 0.0
    subject_filter: SubjectFilter = None
    run_status: RunStatusFilter = "completed"


class ScopeFrontierParams(EvalBaseModel):
    """``scope_frontier``."""

    bar: Annotated[
        float | None,
        Field(
            ge=0.0,
            le=1.0,
            description="The pass^k a variant must clear, from 0 to 1; omitted ranks the variants without a verdict.",
        ),
    ] = None
    subject_filter: SubjectFilter = None
    run_status: RunStatusFilter = "completed"


class ScopeExportParams(EvalBaseModel):
    """``scope_export``."""

    export_format: ExportFormat = "csv"
    run_status: RunStatusFilter = "completed"
    export_run_ids: ExportRunIds = None


class LaunchEstimateParams(RunLaunchParams):
    """``launch_estimate`` — what a ``run_launch`` with the same arguments would cost, priced by its own rule."""

    n_test_cases: CaseCount = None


class ScopeOutOfRunSpendParams(EvalBaseModel):
    """``scope_out_of_run_spend`` — what the engine spent outside any run, narrowed or not."""

    purpose_filter: Annotated[
        OutOfRunPurpose | None,
        Field(
            description="Only calls made for this purpose: variation (a launch's case generation), proposer (a "
            "rubric draft) or analysis (an analysis generation)."
        ),
    ] = None
    launch_group_filter: Annotated[
        str | None,
        Field(min_length=1, description="Only the calls one launch's case generation made; its runs carry this id."),
    ] = None
    template_filter: Annotated[
        str | None, Field(min_length=1, description="Only calls made for this template, by id.")
    ] = None


class ResultRateParams(EvalBaseModel):
    """``result_rate``."""

    result_id: ResultId
    rubric_dim: Annotated[
        str, Field(min_length=1, description="The judged dimension rated, spelled as the result's score spells it.")
    ]
    score: Annotated[int, Field(description="The score, on the dimension's scale: 1-5, or 1 (pass) / 0 (fail).")]
    rating_reason: Annotated[str, Field(min_length=1, description="Why that score, in the rater's own words.")]


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


async def _results_list(host: OpsHost, caller: Caller, params: ResultsListParams) -> ResultListing:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(
            results_list,
            condition=None if params.condition_filter is None else ResultOutcome(params.condition_filter),
            offset=params.offset,
            limit=params.limit,
        ),
        eval_host,
        params.run_id,
        caller.scope_id,
    )


async def _result_get(host: OpsHost, caller: Caller, params: ResultGetParams) -> ResultDetail:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(result_get, part=params.part),
        eval_host,
        params.result_id,
        caller.scope_id,
    )


async def _run_launch(host: OpsHost, caller: Caller, params: RunLaunchParams) -> JobsStarted:
    return await run_launch(host, params, caller.scope_id)


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


async def _analysis_estimate(
    host: OpsHost, caller: Caller, params: AnalysisGenerateParams
) -> AnalysisGenerationEstimate:
    return await analysis_estimate(host, params.campaign_id, caller.scope_id, model=params.model)


async def _analysis_archive(host: OpsHost, caller: Caller, params: AnalysisArchiveParams) -> AnalysisLine:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(analysis_archive, archived=params.archived, reason=params.archive_reason),
        eval_host,
        params.analysis_id,
        caller.scope_id,
    )


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


async def _reporter_case_freeze(host: OpsHost, caller: Caller, params: ReporterCaseFreezeParams) -> FrozenReporterCase:
    eval_host = host.eval_host
    freeze = ReporterCaseFreeze.model_validate(params.model_dump())
    return await run_blocking(eval_host.blocking_executor, reporter_case_freeze, eval_host, freeze, caller.scope_id)


async def _reporter_cases_list(host: OpsHost, caller: Caller, params: ReporterCasesListParams) -> ReporterCaseListing:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(reporter_cases_list, include_archived=params.include_archived),
        eval_host,
        params.template_id,
        caller.scope_id,
    )


async def _reporter_case_archive(
    host: OpsHost, caller: Caller, params: ReporterCaseArchiveParams
) -> FrozenReporterCase:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(reporter_case_archive, archived=params.archived, reason=params.archive_reason),
        eval_host,
        params.reporter_case_id,
        caller.scope_id,
    )


async def _scope_pivot(host: OpsHost, caller: Caller, params: ScopePivotParams) -> PivotTable:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        scope_pivot,
        eval_host,
        caller.scope_id,
        row_factor=params.row_factor,
        column_factor=params.column_factor,
        metric=params.metric,
        weighting=params.weighting,
        subject_id=params.subject_filter,
        status=params.run_status,
        predicted_cost=params.predicted_cost,
        launched_run_ids=params.launched_run_ids or (),
    )


async def _runs_compare(host: OpsHost, caller: Caller, params: RunsCompareParams) -> RunsCompared:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        runs_compare,
        eval_host,
        params.baseline_run_id,
        params.candidate_run_id,
        caller.scope_id,
    )


async def _scope_history(host: OpsHost, caller: Caller, params: ScopeHistoryParams) -> HistoryResult:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        scope_history,
        eval_host,
        caller.scope_id,
        metric=params.metric,
        min_absolute_change=params.min_absolute_change,
        min_relative_change=params.min_relative_change,
        subject_id=params.subject_filter,
        status=params.run_status,
    )


async def _scope_frontier(host: OpsHost, caller: Caller, params: ScopeFrontierParams) -> FrontierResult:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        scope_frontier,
        eval_host,
        caller.scope_id,
        bar=params.bar,
        subject_id=params.subject_filter,
        status=params.run_status,
    )


async def _scope_export(host: OpsHost, caller: Caller, params: ScopeExportParams) -> ScoreExport:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        scope_export,
        eval_host,
        caller.scope_id,
        format=params.export_format,
        status=params.run_status,
        run_ids=params.export_run_ids,
    )


async def _scope_out_of_run_spend(
    host: OpsHost, caller: Caller, params: ScopeOutOfRunSpendParams
) -> OutOfRunSpendReport:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(
            scope_out_of_run_spend,
            eval_host,
            caller.scope_id,
            purpose=params.purpose_filter,
            launch_group_id=params.launch_group_filter,
            template_id=params.template_filter,
        ),
    )


async def _launch_estimate(host: OpsHost, caller: Caller, params: LaunchEstimateParams) -> LaunchEstimate:
    return await launch_estimate(
        host,
        LaunchArguments.model_validate(params.model_dump(exclude={"n_test_cases"})),
        caller.scope_id,
        n_test_cases=params.n_test_cases,
    )


async def _result_rate(host: OpsHost, caller: Caller, params: ResultRateParams) -> ResultRated:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(
            result_rate,
            rubric_dim=params.rubric_dim,
            score=params.score,
            reason=params.rating_reason,
            rater=caller.identity,
        ),
        eval_host,
        params.result_id,
        caller.scope_id,
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
    reporter_template_id, reporter_case_id = "tmpl-reporter", "0193a1b2-reporter-case"
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
                "Every run spends against its cost cap: the host's, or max_cost_usd, which may only lower it — a "
                "max_cost_usd above the host's ceiling is refused. A launch the "
                "process cannot admit, or the template's kind cannot honour, is refused before anything starts. "
                "n_variations generates that many new cases first, shared by every arm; a template's llm axis is "
                "written by variation_model, whose calls run before the runs start and are outside every run's "
                "cost cap."
            ),
        ),
        Action(
            name="launch_estimate",
            summary="Estimate what a run_launch would cost, from what the scope's runs have spent.",
            workflow=RUN,
            permission="read",
            params=LaunchEstimateParams,
            result=LaunchEstimate,
            handler=_launch_estimate,
            render=render.render_estimate,
            example={"template_id": "tmpl-1", "subject_id": "subject-1", "models": ["model-a", "model-b"], "k_runs": 3},
            detail=(
                "Takes run_launch's own arguments and prices them by the launch's own rule: each arm planned by its "
                "kind (its cases, its model, its judges and simulator) and priced by the host's launch pricer, held to "
                "the cap the launch would hold it to. Every refusal the launch makes before pricing is returned as an "
                "error, as run_launch returns it; an arm the launch would refuse on its price is reported with the "
                "refusal, word for word. n_test_cases prices a hypothetical grid instead of each plan's cases; the "
                "generation calls a generating launch makes first are priced by the launch itself. Pass the structured "
                "result to scope_pivot as predicted_cost after the runs land, to set each prediction beside the cost "
                "observed. Spends nothing."
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
            name="results_list",
            summary="List one run's results, a light row each — case, repeat, condition, headline measures — paged.",
            workflow=RUN,
            permission="read",
            params=ResultsListParams,
            result=ResultListing,
            handler=_results_list,
            render=partial(render.render_results, default_limit=_RESULTS_PAGE_DEFAULT),
            example={"run_id": run_id, "condition_filter": "candidate_fail"},
            detail=(
                "Ordered by case, then repeat, then result id, so a finished run pages the same way every time; "
                "pass next_offset as offset for the next page. condition_filter narrows to the results that "
                "counted as delivered (ok), as a hard fail (candidate_fail) or not at all (infra_exclude), and "
                "total counts what matched across every page. A run not in the caller's scope is not found. "
                "result_get reads one row's result whole."
            ),
        ),
        Action(
            name="result_get",
            summary="Read one stored result: its record, usage rows and condition, and one part of its trace.",
            workflow=RUN,
            permission="read",
            params=ResultGetParams,
            result=ResultDetail,
            handler=_result_get,
            render=render.render_result,
            example={"result_id": "0193a1b2-result"},
            detail=(
                "part record (the default) returns the output documents exactly as the candidate's kind stored "
                "them, beside the call ledger (only the calls the kind recorded as succeeded) and the world's end "
                "state. What an output document says about each call is whatever the kind wrote: a failed call "
                "appears there only when the kind records failures. part judge returns what the judge was sent, "
                "and part spans the stored spans; the default leaves both out, text and data alike, since either "
                "can outweigh the rest. The condition is resolved as every surface resolves it, with the sentence "
                "a reader must not miss. A result not in the caller's scope is not found; results_list names a "
                "run's results."
            ),
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
            example={
                "name": "model bake-off",
                "subject_id": "subject-1",
                "behavior": "accuracy",
                "run_ids": [run_id],
                "declared_design": {
                    "axes": [
                        {
                            "axis_id": "model",
                            "values": [
                                {"content": "model-a", "display": "model-a"},
                                {"content": "model-b", "display": "model-b"},
                            ],
                        }
                    ],
                    "held_fixed": {"stimulus": "controlled", "apparatus": "commissioned"},
                },
                "control_from_run_id": run_id,
            },
            detail=(
                "declared_design states what the campaign sets out to learn — its axes and what it held fixed, and "
                "optionally its questions, bars and merit priority — and is refused, naming why, when this host "
                "cannot honour it: an axis it does not declare, a bar looser than the registered one. control_from_run_id "
                "names the run whose variant every other cell is read against. Omit both and the campaign is "
                "exploratory: its report and analysis say once that its readings confirm nothing, and its design is "
                "inferred from the runs."
            ),
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
                "running. Every call is priced before it is sent against the host's out-of-run cap: the first here, "
                "so one over the cap is refused with nothing spent; the one repair round-trip a refused output buys "
                "when its prompt exists, against what is left. analysis_estimate prices it first without spending. "
                "Each call is ledgered under purpose analysis, so scope_out_of_run_spend reads it; the job's record "
                "is the generation attempt, written however it ends."
            ),
        ),
        Action(
            name="analysis_estimate",
            summary="Price a campaign's analysis generation against the host's out-of-run cap, without spending.",
            workflow=ANALYSE,
            permission="read",
            params=AnalysisGenerateParams,
            result=AnalysisGenerationEstimate,
            handler=_analysis_estimate,
            render=render.render_analysis_estimate,
            example={"campaign_id": campaign_id},
            detail=(
                "The generation's own assembly and pricing rule, so would_start is analysis_generate's answer. "
                "Makes no model call."
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
            name="reporter_case_freeze",
            summary="Freeze a campaign's analysis bundle, and the memo it got, into a case of a reporter template.",
            workflow=ANALYSE,
            permission="write",
            params=ReporterCaseFreezeParams,
            result=FrozenReporterCase,
            handler=_reporter_case_freeze,
            render=render.render_reporter_case,
            example={
                "template_id": reporter_template_id,
                "campaign_id": campaign_id,
                "recorded_analysis_id": analysis_id,
                "labels": [{"dimension": "reporter.groundedness", "direction": "high", "quote": "every number checks"}],
            },
            detail=(
                "A reporter run measures the analysis writer, and its cases are never generated: each is one "
                "campaign's bundle, frozen with its fingerprint, and a reporter template launches nothing until one "
                "is frozen. With recorded_analysis_id the case also pins that analysis's memo, which the as-recorded "
                "candidate replays and labels are written about; without it only a generating candidate can run "
                "the case. One campaign and memo is one case: a freeze matching the pair's live case returns it, and "
                "one that differs (other labels, or evidence that moved) is refused unless supersedes names the live "
                "case. limits states what the frozen evidence cannot support. Then run_launch the template; "
                "reporter_cases_list reads its bank, reporter_case_archive retires a case. Calls no model."
            ),
        ),
        Action(
            name="reporter_cases_list",
            summary="List a reporter template's cases: which each campaign and memo launches, superseded or retired.",
            workflow=ANALYSE,
            permission="read",
            params=ReporterCasesListParams,
            result=ReporterCaseListing,
            handler=_reporter_cases_list,
            render=render.render_reporter_cases,
            example={"template_id": reporter_template_id},
            detail=(
                "Each case with whether a launch runs it and what superseded it. A case this build cannot read is "
                "listed rather than refused, since this is where it is found; while one is listed, which case is "
                "live cannot be decided. A pair holding more than one live case is named, since every launch of the "
                "template refuses until one freeze supersedes them all."
            ),
        ),
        Action(
            name="scope_pivot",
            summary="Aggregate one measure over the scope's observations by two coordinates, cell by cell.",
            workflow=ANALYSE,
            permission="read",
            params=ScopePivotParams,
            result=PivotTable,
            handler=_scope_pivot,
            render=render.render_pivot,
            example={"row_factor": "template_id", "column_factor": "model", "metric": "cost_usd"},
            detail=(
                "Every cell carries its denominators (observations and cases) and its spread, and the table names "
                "what it left out and which runs measured less than they promised. A ranking pooled over the rows "
                "that most rows contradict is flagged. A judged measure over several subjects is refused unless "
                "subject_filter names one."
            ),
        ),
        Action(
            name="runs_compare",
            summary="Compare one run's arm against another's: pass^k, mean composite, their deltas and the test.",
            workflow=ANALYSE,
            permission="read",
            params=RunsCompareParams,
            result=RunsCompared,
            handler=_runs_compare,
            render=render.render_runs_compared,
            example={"baseline_run_id": run_id, "candidate_run_id": "0193a1b2-run-b"},
            detail=(
                "pass^k is read on both arms at the shallower arm's depth. The composite is tested paired over the "
                "cases both runs scored, else unpaired, and the answer says which; a miss reads not significant, "
                "never no difference. Composites of two different subjects are not compared. Each run that "
                "delivered less than its matrix carries its sentence, and the answer says when the two were "
                "measured over spans that do not overlap or recorded different cassette modes. A run not in the "
                "caller's scope is not found."
            ),
        ),
        Action(
            name="scope_out_of_run_spend",
            summary=(
                "List what the engine spent outside any run — case generations, rubric proposals and analysis "
                "generations — with totals."
            ),
            workflow=ANALYSE,
            permission="read",
            params=ScopeOutOfRunSpendParams,
            result=OutOfRunSpendReport,
            handler=_scope_out_of_run_spend,
            render=render.render_out_of_run_spend,
            example={"purpose_filter": "variation"},
            detail=(
                "Read off the out-of-run ledger, one row per call, returned or raised: the spend no run's results "
                "carry, since a launch's case generation runs before its runs exist and an analysis after they end. Totals overall, per purpose and "
                "per launch; a call that reported no cost, or raised, is counted as unpriced and left out of the sum, "
                "which is then a floor. Narrow by purpose_filter, launch_group_filter (the launch_group_id a launch's runs "
                "carry) or template_filter. "
                "Spends nothing."
            ),
        ),
        Action(
            name="scope_history",
            summary="Series one measure over time for each contestant in the scope, flagging regressions.",
            workflow=ANALYSE,
            permission="read",
            params=ScopeHistoryParams,
            result=HistoryResult,
            handler=_scope_history,
            render=render.render_history,
            example={"metric": "cost_usd", "min_relative_change": 0.1},
            detail=(
                "A contestant is one resolved configuration within a subject, so a step is a re-run of the same "
                "thing. Each step against the previous point carries a verdict and the test it rests on; a step "
                "where the suite changed is marked so a new denominator does not read as a regression. A step that "
                "misses significance reads not_separated, never no change; only an equivalence test against the "
                "measure's declared materiality threshold reads equivalent."
            ),
        ),
        Action(
            name="scope_frontier",
            summary="Rank each subject's variants on quality, cost and latency, and name the cheapest that clears a bar.",
            workflow=ANALYSE,
            permission="read",
            params=ScopeFrontierParams,
            result=FrontierResult,
            handler=_scope_frontier,
            render=render.render_frontier,
            example={"bar": 0.8},
            detail=(
                "A point is one variant of a subject: pass^k with its interval, mean composite, production-replicating "
                "cost and latency, each over its own count. A point is dominated only when another is shown better on "
                "every axis it measured. With a bar, a variant clears it only when its whole pass^k interval is at or "
                "above it, and the verdict names the cheapest that cleared, beside every rival it was not shown cheaper "
                "than. The arms of one compare share a subject, so its models rank on one frontier. Spends nothing."
            ),
        ),
        Action(
            name="scope_export",
            summary="Export the scope's observations as flat rows, CSV or JSON, for analysis elsewhere.",
            workflow=ANALYSE,
            permission="read",
            params=ScopeExportParams,
            result=ScoreExport,
            handler=_scope_export,
            render=render.render_export,
            example={"export_format": "csv", "export_run_ids": [run_id]},
            detail=(
                "One row per observation, every lever and host measure its own column. CSV holds the rows alone, "
                "so the result says beside it how many observations were left out and why; JSON carries both."
            ),
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
            name="result_rate",
            summary="Rate one judged dimension of one result, as an agent — kept beside people's ratings, never pooled.",
            workflow=CURATE,
            permission="write",
            params=ResultRateParams,
            result=ResultRated,
            handler=_result_rate,
            render=render.render_result_rated,
            example={"result_id": "result-1", "rubric_dim": "chat.tone", "score": 4, "rating_reason": "warm, on point"},
            detail=(
                "Recorded as the caller's identity with rater_kind agent, fixed here: an agent rating through a tool "
                "is another model's opinion, so it is listed beside people's ratings and never enters the judge's "
                "agreement with people. A second rating of the same dimension of the same result replaces the first."
            ),
        ),
        Action(
            name="analysis_archive",
            summary="Archive a stored analysis (or restore it): marked, its insights retracted, nothing destroyed.",
            workflow=CURATE,
            permission="write",
            params=AnalysisArchiveParams,
            result=AnalysisLine,
            handler=_analysis_archive,
            render=render.render_analysis_line,
            example={"analysis_id": analysis_id, "archived": True, "archive_reason": "superseded"},
        ),
        Action(
            name="reporter_case_archive",
            summary="Retire a reporter case (or restore it): no launch runs it again, nothing destroyed.",
            workflow=CURATE,
            permission="write",
            params=ReporterCaseArchiveParams,
            result=FrozenReporterCase,
            handler=_reporter_case_archive,
            render=render.render_reporter_case,
            example={
                "reporter_case_id": reporter_case_id,
                "archived": True,
                "archive_reason": "its bundle no longer re-assembles",
            },
            detail=(
                "For a case that can no longer measure anything. It stays readable for every run measured against "
                "it. A restore is refused when it would give its campaign and memo a second live case."
            ),
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
