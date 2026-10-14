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
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError

from threetears.evals.kernel.remedies import FIND_JUDGE_CONFIG, SCHEMA_REMEDY
from threetears.evals.actions import render
from threetears.evals.actions.catalogue import Action, ActionCatalogue, Caller, ConformanceStage
from threetears.evals.schema import EvalRunStatus
from threetears.evals.kernel import ResultOutcome, ValidationFailedError
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema import KnownOutOfRunPurpose
from threetears.evals.ops import (
    SweepArguments,
    SweepSettings,
    sweep_launch,
    AnalysisDeleted,
    CaseSetLine,
    CaseSetListing,
    CaseSetMint,
    case_set_mint,
    case_sets_list,
    AnalysisGenerationEstimate,
    AnalysisLine,
    AnalysisListing,
    BarProposals,
    CampaignChanges,
    CampaignDefinition,
    CampaignLine,
    CampaignListing,
    LaunchEstimate,
    EvalSummary,
    FrozenReporterCase,
    FrontierResult,
    HistoryResult,
    InsightDeleted,
    InsightDetail,
    JudgeConfigListing,
    InsightListing,
    JobsStarted,
    JobStatus,
    LaunchArguments,
    OpsHost,
    OutOfRunSpendReport,
    PivotTable,
    ReportDocument,
    JudgeCaseAuthor,
    JudgeCasesFreeze,
    JudgeTemplateCreate,
    JudgeProfileListing,
    JudgeProfileRecording,
    JudgeProfilesRecord,
    ReporterCaseFreeze,
    ReporterCaseListing,
    ResultDetail,
    ResultListing,
    ResultPart,
    ResultRated,
    RunDeleted,
    RunLine,
    RunListing,
    UnreadableRunsListing,
    RunsBisected,
    RunsCompared,
    ScoreExport,
    TemplateLine,
    TemplateListing,
    UndescribableArmsListing,
    analyses_list,
    analyses_undescribable,
    analysis_archive,
    analysis_delete,
    analysis_estimate,
    analysis_generate,
    bars_propose,
    campaign_archive,
    campaign_create,
    campaign_runs_add,
    campaign_runs_remove,
    campaign_update,
    campaigns_list,
    insight_delete,
    insight_get,
    insights_list,
    job_cancel,
    job_poll,
    judge_drift_check,
    judge_second,
    judge_second_estimate,
    judge_temperature,
    judge_temperature_estimate,
    launch_estimate,
    report_read,
    reporter_case_archive,
    judge_case_author,
    judge_cases_freeze,
    judge_configs_list,
    judge_template_create,
    judge_profiles_list,
    judge_profiles_record,
    reporter_case_freeze,
    reporter_cases_list,
    result_get,
    result_rate,
    results_list,
    run_archive,
    run_label,
    run_delete,
    run_get,
    run_launch,
    runs_bisect,
    runs_compare,
    runs_list,
    runs_unreadable,
    scope_export,
    scope_frontier,
    scope_history,
    scope_out_of_run_spend,
    scope_pivot,
    templates_list,
    bar_proposals_text,
    estimate_text,
    export_text,
    frontier_text,
    history_text,
    out_of_run_spend_text,
    pivot_text,
    runs_bisected_text,
    runs_compared_text,
)
from threetears.evals.schema.models import RUN_LABELS_MAX, SecondJudge
from threetears.evals.kernel.judge_temperature import DEFAULT_TEMPERATURE_REPEATS, TemperatureSelection
from threetears.evals.run import JudgeCaseFreezeReport, JudgeTemperatureEstimate, SecondJudgeEstimate, run_blocking

# --- the parameters, each declared once ---------------------------------------------------------------

RunId = Annotated[str, Field(min_length=1, description="A run's id, as runs_list or a launch's job names it.")]
# The launch's own two, taken from its one declaration so an action naming a template or a subject means the
# same thing a launch does.
TemplateId = Annotated[str, LaunchArguments.model_fields["template_id"]]
SubjectId = Annotated[str, LaunchArguments.model_fields["subject_id"]]
ResultId = Annotated[str, Field(min_length=1, description="A result's id, as results_list names it.")]
CampaignId = Annotated[str, Field(min_length=1, description="A campaign's id, as campaigns_list names it.")]
AnalysisId = Annotated[str, Field(min_length=1, description="A stored analysis's id, as analyses_list names it.")]
InsightId = Annotated[str, Field(min_length=1, description="An insight's id, as insights_list names it.")]
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
RunIds = Annotated[
    list[str],
    Field(
        description="The runs, by id, all in the caller's scope: those a campaign is created with, or those added to "
        "or taken out of it."
    ),
]
GeneratorModel = Annotated[
    str | None,
    Field(
        description="The analysis generator's model; omitted for the host's. A model outside the host's allowed "
        "writers is refused before anything is spent."
    ),
]
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
SubjectFilter = Annotated[
    str | None, Field(description="Read only this subject's runs or insights; omitted reads every subject.")
]
CampaignFilter = Annotated[
    str | None,
    Field(description="List only the insights this campaign's analyses minted, by id; omitted lists every campaign's."),
]
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
    label: Annotated[
        str | None, Field(description="List only runs carrying this label (run_label sets them); every run if unset.")
    ] = None


class RunsUnreadableParams(EvalBaseModel):
    """``runs_unreadable`` — the caller's scope is the whole question, so it takes nothing."""


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


class RunLabelParams(EvalBaseModel):
    """``run_label``."""

    run_id: RunId
    run_labels: list[str] = Field(
        max_length=RUN_LABELS_MAX,
        description=(
            "The run's labels, replacing any it has; [] clears them. Each 1 to 64 characters, stored sorted and "
            "each once. A label names the run (a baseline, a production snapshot) and moves no key or fingerprint."
        ),
    )


class SweepLaunchParams(SweepSettings):
    """``sweep_launch`` — a sweep's shared settings, declared once on :class:`~threetears.evals.ops.SweepSettings`, and its arms."""

    arms: list[dict[str, Any]] = Field(
        min_length=1,
        description="The arms, launched in this order: each {model, label?, overlays?, apparatus_settings?, "
        "judge_model?, simulator_model?} — its own model and what it sets differently from its siblings.",
    )


class CaseSetMintParams(CaseSetMint):
    """``case_set_mint`` — the operation's own arguments, declared once on :class:`~threetears.evals.ops.CaseSetMint`."""


class CaseSetsListParams(EvalBaseModel):
    """``case_sets_list``."""

    case_set_filter: Annotated[str | None, Field(min_length=1, description="List only this case set's versions.")] = (
        None
    )


class AnalysesUndescribableParams(EvalBaseModel):
    """``analyses_undescribable`` — the caller's scope is the whole question, so it takes nothing."""


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
    template_id: Annotated[
        str | None, Field(min_length=1, description=LaunchArguments.model_fields["template_id"].description)
    ] = None
    declared_design: Annotated[dict[str, Any] | None, CampaignDefinition.model_fields["declared_design"]] = None
    control_from_run_id: Annotated[str | None, CampaignDefinition.model_fields["control_from_run_id"]] = None
    exploratory: Annotated[bool, CampaignDefinition.model_fields["exploratory"]] = False


class CampaignUpdateParams(EvalBaseModel):
    """``campaign_update`` — the operation's own changes (:class:`~threetears.evals.ops.CampaignChanges`), each only when given."""

    campaign_id: CampaignId
    name: Annotated[
        str | None, Field(min_length=1, description=CampaignCreateParams.model_fields["name"].description)
    ] = None
    description: Annotated[
        str | None, Field(description=CampaignCreateParams.model_fields["description"].description)
    ] = None
    declared_design: Annotated[dict[str, Any] | None, CampaignChanges.model_fields["declared_design"]] = None
    exploratory: Annotated[bool, CampaignChanges.model_fields["exploratory"]] = False
    control_from_run_id: Annotated[str | None, CampaignChanges.model_fields["control_from_run_id"]] = None


class CampaignRunsParams(EvalBaseModel):
    """``campaign_runs_add`` and ``campaign_runs_remove``."""

    campaign_id: CampaignId
    run_ids: RunIds


class JudgeConfigsListParams(EvalBaseModel):
    """``judge_configs_list``."""

    dim_filter: Annotated[
        str | None, Field(min_length=1, description="List only the configs asking for this dimension.")
    ] = None


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


class JudgeTemplateCreateParams(JudgeTemplateCreate):
    """``judge_template_create`` — the operation's own arguments, declared once on :class:`~threetears.evals.ops.JudgeTemplateCreate`."""


class JudgeCasesFreezeParams(JudgeCasesFreeze):
    """``judge_cases_freeze`` — the freeze's own arguments, declared once on :class:`~threetears.evals.ops.JudgeCasesFreeze`."""


class JudgeCaseAuthorParams(JudgeCaseAuthor):
    """``judge_case_author`` — the authoring's own arguments, declared once on :class:`~threetears.evals.ops.JudgeCaseAuthor`."""


class JudgeProfilesRecordParams(JudgeProfilesRecord):
    """``judge_profiles_record`` — the recording's own arguments, declared once on :class:`~threetears.evals.ops.JudgeProfilesRecord`."""


class JudgeProfilesListParams(EvalBaseModel):
    """``judge_profiles_list``."""

    criterion_dim: str | None = Field(
        default=None, min_length=1, description="Only the profiles of this criterion's dim; omitted lists every one."
    )


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
            description="The runs the estimated launch made (the run ids its launch job names, as run_ids), with predicted_cost: each "
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


class RunsBisectParams(RunsCompareParams):
    """``runs_bisect``: the same two runs ``runs_compare`` takes."""


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
        KnownOutOfRunPurpose | None,
        Field(
            description="Only calls made for this purpose: variation (a launch's case generation), proposer (a "
            "rubric draft), analysis (an analysis generation), judge (a judge's repeat or temperature comparison) "
            "or second_judge (a second judge's pass)."
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


SecondJudgeModel = Annotated[
    str, Field(min_length=1, description="The model the second judge sends every judged dimension to.")
]
SecondJudgeConfigIds = Annotated[
    dict[str, str] | None,
    Field(
        description="dim -> the versioned judge config whose prompt asks the second judge for that dim; omitted for "
        "the prompts the run recorded. A dim absent from a map given is asked with the built-in prompt."
    ),
]
SecondJudgeTemperature = Annotated[
    float | None,
    Field(
        ge=0.0,
        le=2.0,
        description="The temperature every second-judge call is requested at; omitted for what each prompt asks.",
    ),
]
ResultIds = Annotated[
    list[str] | None, Field(description="The run's results to draw from; omitted for every result of the run.")
]


class JudgeDriftParams(EvalBaseModel):
    """``judge_drift_check``."""

    run_id: RunId
    second_judge_model: SecondJudgeModel
    second_judge_config_ids: SecondJudgeConfigIds = None
    second_judge_temperature: SecondJudgeTemperature = None
    result_ids: ResultIds = None

    def judge(self) -> SecondJudge:
        """The second judge these parameters name.

        Raises:
            ValidationFailedError: The judge does not validate — a dimension in ``second_judge_config_ids`` not named
                as ``<context>.<dim>``, say. Raised as a refusal: a model's own error would leave the handler as a
                defect, which the call is not.
        """
        try:
            return SecondJudge(
                model=self.second_judge_model,
                config_ids=self.second_judge_config_ids,
                temperature=self.second_judge_temperature,
            )
        except ValidationError as e:
            raise ValidationFailedError(
                "invalid second judge: " + "; ".join(f"{error['msg']} ({error['loc']})" for error in e.errors()),
                remedy=f"name each dimension as the run's scores spell it ({FIND_JUDGE_CONFIG})",
            ) from e


class JudgeSecondParams(JudgeDriftParams):
    """``judge_second`` and ``judge_second_estimate``."""

    sample_fraction: Annotated[
        float,
        Field(gt=0.0, le=1.0, description="The share of the run's judged results the second judge is asked about."),
    ] = 1.0
    sample_seed: Annotated[
        int, Field(description="The seed the share is drawn with; the same seed draws the same results.")
    ] = 0


class JudgeTemperatureParams(EvalBaseModel):
    """``judge_temperature`` and ``judge_temperature_estimate``."""

    run_id: RunId
    selection: Annotated[
        TemperatureSelection,
        Field(
            description="Which scored dims are re-judged: borderline (stored score inside the scale, or a recorded "
            "repeat or second judge disagreed) or all."
        ),
    ] = "borderline"
    repeats: Annotated[
        int, Field(ge=2, le=50, description="How many times each dim is judged at each temperature.")
    ] = DEFAULT_TEMPERATURE_REPEATS
    result_ids: ResultIds = None


class RunDeleteParams(EvalBaseModel):
    """``run_delete``."""

    run_id: RunId
    confirm: Confirm


class AnalysisDeleteParams(EvalBaseModel):
    """``analysis_delete``."""

    analysis_id: AnalysisId
    confirm: Confirm


class InsightsListParams(EvalBaseModel):
    """``insights_list``."""

    subject_filter: SubjectFilter = None
    campaign_filter: CampaignFilter = None


class InsightParams(EvalBaseModel):
    """``insight_get``."""

    insight_id: InsightId


class InsightDeleteParams(EvalBaseModel):
    """``insight_delete``."""

    insight_id: InsightId
    confirm: Confirm


# --- the handlers ------------------------------------------------------------------------------------


async def _templates_list(host: OpsHost, caller: Caller, _: NoParams) -> TemplateListing:
    return await run_blocking(host.eval_host.blocking_executor, templates_list, host.eval_host, caller.scope_id)


async def _runs_unreadable(host: OpsHost, caller: Caller, params: RunsUnreadableParams) -> UnreadableRunsListing:
    eval_host = host.eval_host
    return await run_blocking(eval_host.blocking_executor, runs_unreadable, eval_host, caller.scope_id)


async def _runs_list(host: OpsHost, caller: Caller, params: RunsListParams) -> RunListing:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        runs_list,
        eval_host,
        caller.scope_id,
        status=params.status,
        include_archived=params.include_archived,
        label=params.label,
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


async def _run_label(host: OpsHost, caller: Caller, params: RunLabelParams) -> RunLine:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor, run_label, eval_host, params.run_id, caller.scope_id, labels=params.run_labels
    )


async def _sweep_launch(host: OpsHost, caller: Caller, params: SweepLaunchParams) -> JobsStarted:
    try:
        arguments = SweepArguments.model_validate(params.model_dump())
    except ValidationError as e:
        raise ValidationFailedError(
            f"invalid sweep: {e.errors()[0]['msg']} ({e.errors()[0]['loc']})", remedy=SCHEMA_REMEDY
        ) from e
    return await sweep_launch(host, arguments, caller.scope_id, created_by=caller.identity)


async def _case_set_mint(host: OpsHost, caller: Caller, params: CaseSetMintParams) -> CaseSetLine:
    eval_host = host.eval_host
    arguments = CaseSetMint.model_validate(params.model_dump())
    return await run_blocking(eval_host.blocking_executor, case_set_mint, eval_host, arguments, caller.scope_id)


async def _case_sets_list(host: OpsHost, caller: Caller, params: CaseSetsListParams) -> CaseSetListing:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(case_sets_list, name=params.case_set_filter),
        eval_host,
        caller.scope_id,
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


async def _campaign_update(host: OpsHost, caller: Caller, params: CampaignUpdateParams) -> CampaignLine:
    eval_host = host.eval_host
    # Only what the caller sent: an omitted declared_design leaves the declaration alone, a null one withdraws it.
    changes = CampaignChanges.model_validate(params.model_dump(include=params.model_fields_set - {"campaign_id"}))
    return await run_blocking(
        eval_host.blocking_executor,
        campaign_update,
        eval_host,
        params.campaign_id,
        changes,
        caller.scope_id,
        updated_by=caller.identity,
    )


async def _campaign_runs_add(host: OpsHost, caller: Caller, params: CampaignRunsParams) -> CampaignLine:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor, campaign_runs_add, eval_host, params.campaign_id, params.run_ids, caller.scope_id
    )


async def _campaign_runs_remove(host: OpsHost, caller: Caller, params: CampaignRunsParams) -> CampaignLine:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        campaign_runs_remove,
        eval_host,
        params.campaign_id,
        params.run_ids,
        caller.scope_id,
    )


async def _judge_configs_list(host: OpsHost, caller: Caller, params: JudgeConfigsListParams) -> JudgeConfigListing:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(judge_configs_list, rubric_dim_id=params.dim_filter),
        eval_host,
        caller.scope_id,
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


async def _analyses_undescribable(
    host: OpsHost, caller: Caller, params: AnalysesUndescribableParams
) -> UndescribableArmsListing:
    eval_host = host.eval_host
    return await run_blocking(eval_host.blocking_executor, analyses_undescribable, eval_host, caller.scope_id)


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


async def _bars_propose(host: OpsHost, caller: Caller, params: CampaignParams) -> BarProposals:
    eval_host = host.eval_host
    return await run_blocking(eval_host.blocking_executor, bars_propose, eval_host, params.campaign_id, caller.scope_id)


async def _judge_template_create(host: OpsHost, caller: Caller, params: JudgeTemplateCreateParams) -> TemplateLine:
    create = JudgeTemplateCreate.model_validate(params.model_dump())
    return await run_blocking(host.eval_host.blocking_executor, judge_template_create, host, create, caller.scope_id)


async def _judge_cases_freeze(host: OpsHost, caller: Caller, params: JudgeCasesFreezeParams) -> JudgeCaseFreezeReport:
    eval_host = host.eval_host
    freeze = JudgeCasesFreeze.model_validate(params.model_dump())
    return await run_blocking(eval_host.blocking_executor, judge_cases_freeze, eval_host, freeze, caller.scope_id)


async def _judge_case_author(host: OpsHost, caller: Caller, params: JudgeCaseAuthorParams) -> JudgeCaseFreezeReport:
    eval_host = host.eval_host
    case = JudgeCaseAuthor.model_validate(params.model_dump())
    # The caller authors the case: its verdict is their label, as a rating is its rater's.
    return await run_blocking(
        eval_host.blocking_executor,
        partial(judge_case_author, author=caller.identity),
        eval_host,
        case,
        caller.scope_id,
    )


async def _judge_profiles_record(
    host: OpsHost, caller: Caller, params: JudgeProfilesRecordParams
) -> JudgeProfileRecording:
    eval_host = host.eval_host
    record = JudgeProfilesRecord.model_validate(params.model_dump())
    return await run_blocking(eval_host.blocking_executor, judge_profiles_record, eval_host, record, caller.scope_id)


async def _judge_profiles_list(host: OpsHost, caller: Caller, params: JudgeProfilesListParams) -> JudgeProfileListing:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(judge_profiles_list, rubric_dim=params.criterion_dim),
        eval_host,
        caller.scope_id,
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


async def _runs_bisect(host: OpsHost, caller: Caller, params: RunsBisectParams) -> RunsBisected:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        runs_bisect,
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


async def _judge_second(host: OpsHost, caller: Caller, params: JudgeSecondParams) -> JobsStarted:
    return await judge_second(
        host,
        params.run_id,
        caller.scope_id,
        judge=params.judge(),
        sample_fraction=params.sample_fraction,
        seed=params.sample_seed,
        result_ids=params.result_ids,
    )


async def _judge_second_estimate(host: OpsHost, caller: Caller, params: JudgeSecondParams) -> SecondJudgeEstimate:
    return await judge_second_estimate(
        host,
        params.run_id,
        caller.scope_id,
        judge=params.judge(),
        sample_fraction=params.sample_fraction,
        seed=params.sample_seed,
        result_ids=params.result_ids,
    )


async def _judge_drift_check(host: OpsHost, caller: Caller, params: JudgeDriftParams) -> JobsStarted:
    return await judge_drift_check(
        host, params.run_id, caller.scope_id, judge=params.judge(), result_ids=params.result_ids
    )


async def _judge_temperature(host: OpsHost, caller: Caller, params: JudgeTemperatureParams) -> JobsStarted:
    return await judge_temperature(
        host,
        params.run_id,
        caller.scope_id,
        selection=params.selection,
        repeats=params.repeats,
        result_ids=params.result_ids,
    )


async def _judge_temperature_estimate(
    host: OpsHost, caller: Caller, params: JudgeTemperatureParams
) -> JudgeTemperatureEstimate:
    return await judge_temperature_estimate(
        host,
        params.run_id,
        caller.scope_id,
        selection=params.selection,
        repeats=params.repeats,
        result_ids=params.result_ids,
    )


async def _run_delete(host: OpsHost, caller: Caller, params: RunDeleteParams) -> RunDeleted:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor, run_delete, eval_host, params.run_id, caller.scope_id, confirm=params.confirm
    )


async def _insights_list(host: OpsHost, caller: Caller, params: InsightsListParams) -> InsightListing:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(insights_list, subject_id=params.subject_filter, source_campaign_id=params.campaign_filter),
        eval_host,
        caller.scope_id,
    )


async def _insight_get(host: OpsHost, caller: Caller, params: InsightParams) -> InsightDetail:
    eval_host = host.eval_host
    return await run_blocking(eval_host.blocking_executor, insight_get, eval_host, params.insight_id, caller.scope_id)


async def _insight_delete(host: OpsHost, caller: Caller, params: InsightDeleteParams) -> InsightDeleted:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(insight_delete, confirm=params.confirm),
        eval_host,
        params.insight_id,
        caller.scope_id,
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


# --- the happy paths: what reaches each spend action's work, for the jobs conformance kit ---------------


async def _run_launch_path(stage: ConformanceStage) -> Mapping[str, Any]:
    return {
        "template_id": stage.sample.template_id,
        "subject_id": stage.sample.subject_id,
        "models": list(stage.sample.models),
        "k_runs": 1,
    }


async def _sweep_launch_path(stage: ConformanceStage) -> Mapping[str, Any]:
    return {
        "template_id": stage.sample.template_id,
        "subject_id": stage.sample.subject_id,
        "campaign_name": "jobs conformance sweep",
        "campaign_behavior": "conformance",
        "k_runs": 1,
        "arms": [{"model": stage.sample.models[0]}],
    }


async def _analysis_generate_path(stage: ConformanceStage) -> Mapping[str, Any]:
    run_id = await stage.finished_run()
    eval_host = stage.host.eval_host
    campaign = await run_blocking(
        eval_host.blocking_executor,
        partial(
            campaign_create,
            eval_host,
            CampaignDefinition(
                name="jobs conformance campaign",
                subject_id=stage.sample.subject_id,
                behavior="conformance",
                run_ids=[run_id],
                exploratory=True,
            ),
            stage.caller.scope_id,
            created_by=stage.caller.identity,
        ),
    )
    return {"campaign_id": campaign.id}


async def _second_judge_path(stage: ConformanceStage) -> Mapping[str, Any]:
    return {"run_id": await stage.finished_run(), "second_judge_model": stage.sample.second_judge_model}


async def _judge_temperature_path(stage: ConformanceStage) -> Mapping[str, Any]:
    return {"run_id": await stage.finished_run(), "selection": "all", "repeats": 2}


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
    insight_id = "0193a1b2-insight"
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
            render=render.render_model,
            example={},
        ),
        Action(
            name="judge_configs_list",
            summary="List the scope's judge configs — the versioned prompts a second judge or a launch can name.",
            workflow=DISCOVER,
            permission="read",
            params=JudgeConfigsListParams,
            result=JudgeConfigListing,
            handler=_judge_configs_list,
            render=render.render_model,
            example={"dim_filter": "helpfulness"},
            detail=(
                "Each config asks a judge for one dimension, with its prompt, and may pin the judge model and "
                "temperature for it. A second judge's second_judge_config_ids maps a dimension to one of these ids. "
                "Configs are authored by the host, not through an action; only the active version of each is listed."
            ),
        ),
        Action(
            name="runs_list",
            summary="List the scope's runs, newest first.",
            workflow=DISCOVER,
            permission="read",
            params=RunsListParams,
            result=RunListing,
            handler=_runs_list,
            render=render.render_model,
            example={"status": "completed"},
            detail=(
                "A stored run this build cannot read is left out and counted in unreadable rather than failing the "
                "listing; runs_unreadable names each with why. Its archive flag and labels cannot be read, so the "
                "count is narrowed by status alone."
            ),
        ),
        Action(
            name="runs_unreadable",
            summary="List the scope's stored runs this build cannot read, each with why.",
            workflow=DISCOVER,
            permission="read",
            params=RunsUnreadableParams,
            result=UnreadableRunsListing,
            handler=_runs_unreadable,
            render=render.render_model,
            example={},
            detail=(
                "Read from every stored run of the scope on each call, any status, archived included, never stored: "
                "each run document this build's model refuses, with the fields it refused and why. A row that "
                "does not read never stops the rest from being read. Stranded ones were stored pending or running: "
                "the boot sweep counts them unreadable and can never settle them. runs_list leaves every row named "
                "here out and counts it; every other read that reaches one refuses, naming this action. run_delete "
                "removes one from its stored row. Most useful after an upgrade, when the corpus has outlived the "
                "model that wrote it."
            ),
        ),
        Action(
            name="sweep_launch",
            summary="Launch arms that differ beyond the model, one after another, into one campaign, as one job.",
            workflow=RUN,
            permission="spend",
            params=SweepLaunchParams,
            result=JobsStarted,
            handler=_sweep_launch,
            happy_path=_sweep_launch_path,
            example={
                "template_id": "tmpl-1",
                "subject_id": "subject-1",
                "campaign_name": "prompt bake-off",
                "campaign_behavior": "accuracy",
                "arms": [
                    {"model": "model-a", "label": "terse", "overlays": {"style": "terse"}},
                    {"model": "model-a", "label": "verbose", "overlays": {"style": "verbose"}},
                ],
            },
            long_running=True,
            detail=(
                "Every arm is refused up front if its launch would be. Arms run one at a time unless "
                "max_concurrent_arms is raised, each run joining the campaign as it is created. job_poll on the "
                "sweep's job counts the arms launched and finished; one job_cancel stops the arm in flight and "
                "launches none after. An arm refused at its launch, or whose run ends failed, ends the sweep "
                "failed. The judge and simulator are the sweep's unless an arm names its own: where the sweep "
                "names none, the first arm's resolved ones hold for the rest."
            ),
        ),
        Action(
            name="case_sets_list",
            summary="List the scope's named case sets, every version — what a launch can target by name.",
            workflow=DISCOVER,
            permission="read",
            params=CaseSetsListParams,
            result=CaseSetListing,
            handler=_case_sets_list,
            render=render.render_model,
            example={},
        ),
        Action(
            name="case_set_mint",
            summary="Store the next version of a named case set: a template's cases, frozen in order.",
            workflow=RUN,
            permission="write",
            params=CaseSetMintParams,
            result=CaseSetLine,
            handler=_case_set_mint,
            render=render.render_model,
            example={"case_set": "smoke", "template_id": "tmpl-1", "test_case_ids": ["case-1", "case-2"]},
            detail=(
                "Append-only: a new name starts at v1 and every change is the next version, so a launch naming "
                "smoke v1 (run_launch's case_set_name and case_set_version) runs the same cases however the template "
                "changes. A list the latest version already holds is refused."
            ),
        ),
        Action(
            name="campaigns_list",
            summary="List the scope's campaigns — the sets of runs an analysis reads.",
            workflow=DISCOVER,
            permission="read",
            params=CampaignsListParams,
            result=CampaignListing,
            handler=_campaigns_list,
            render=render.render_model,
            example={},
        ),
        Action(
            name="run_launch",
            summary="Launch a template's runs, one per model, as one job to poll: its arms prepared, its runs run.",
            workflow=RUN,
            permission="spend",
            params=RunLaunchParams,
            result=JobsStarted,
            handler=_run_launch,
            happy_path=_run_launch_path,
            example={"template_id": "tmpl-1", "subject_id": "subject-1", "models": ["model-a", "model-b"]},
            long_running=True,
            detail=(
                "Every run spends against its cost cap: the host's, or max_cost_usd, which may only lower it — a "
                "max_cost_usd above the host's ceiling is refused. A launch the "
                "process cannot admit, or the template's kind cannot honour, is refused before anything starts. "
                "Every arm is priced in the call; the job then prepares the arms and runs them. n_variations "
                "generates that many new cases first, in the job, shared by every arm; a template's llm axis is "
                "written by variation_model, whose calls run before the runs start, are priced against the host's "
                "out-of-run cap and are outside every run's cost cap. job_poll on the launch's job names its runs "
                "once they start (run_ids) and counts them finished; the job ends when every run has, and one "
                "job_cancel stops them all."
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
            render=estimate_text,
            example={"template_id": "tmpl-1", "subject_id": "subject-1", "models": ["model-a", "model-b"], "k_runs": 3},
            detail=(
                "Takes run_launch's own arguments and prices them by the launch's own rule: each arm planned by its "
                "kind (its cases, its model, its judges and simulator) and priced by the host's launch pricer, held to "
                "the cap the launch would hold it to. Every refusal the launch makes before pricing is returned as an "
                "error, as run_launch returns it; an arm the launch would refuse on its price is reported with the "
                "refusal, word for word. n_test_cases prices a hypothetical grid instead of each plan's cases; the "
                "generation calls a generating launch makes first are priced by the launch itself. Pass the structured "
                "result to scope_pivot as predicted_cost after the runs land, to set each prediction beside the cost "
                "observed. Beside the price, detectable_effect states per reading the smallest difference the paired "
                "comparison would find at 80% power, from earlier runs of the template's variance, or why it cannot. "
                "Spends nothing."
            ),
        ),
        Action(
            name="job_poll",
            summary="Read where a job stands — a launch, a run, a sweep, a generation, a judge pass — and its result.",
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
            render=EvalSummary.render,
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
            render=render.render_model,
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
                "names the run whose variant every other cell is read against. A campaign with no settled question "
                "says exploratory: true instead of declaring a design — exactly one of the two, never neither, never "
                "both — and its report and analysis say once that its readings confirm nothing, its design inferred "
                "from the runs. campaign_update declares one later. template_id names the template whose rubric "
                "dimensions and goal-state checks a bar may name; omitted, it is the one template every run named ran."
            ),
        ),
        Action(
            name="campaign_update",
            summary="Rename or redescribe a campaign, or declare, amend or withdraw its design before its first run.",
            workflow=ANALYSE,
            permission="write",
            params=CampaignUpdateParams,
            result=CampaignLine,
            handler=_campaign_update,
            render=render.render_model,
            example={
                "campaign_id": campaign_id,
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
            },
            detail=(
                "Changes only what it is sent: name, description, declared_design. A declared_design is validated and "
                "gated as campaign_create's is and replaces the stored one, its questions reconciled with the stored "
                "ones by id, so a question is retired rather than dropped; declaring one on an exploratory campaign "
                "makes it declared, since exploratory is read off whether a design is declared. declared_design: null "
                "withdraws the declaration and is refused unless exploratory: true is sent with it; exploratory: true "
                "beside a design is refused. control_from_run_id names one of the campaign's runs whose variant "
                "becomes the control, resolved for you, in the same write as any design sent beside it; null clears "
                "the control. It needs a declaration — sent here, or already stored — and a design sent here keeps no "
                "control it does not type or name. Once the campaign holds a run its design is frozen: "
                "declared_design is refused in any form, and so is re-pointing or clearing the control, which may "
                "be named once while none is declared; name and description stay editable, and an amended design "
                "is a new campaign (campaign_create). Runs join and leave through campaign_runs_add and "
                "campaign_runs_remove."
            ),
        ),
        Action(
            name="campaign_runs_add",
            summary="Put runs in a campaign — runs launched after it was created, or measured on their own first.",
            workflow=ANALYSE,
            permission="write",
            params=CampaignRunsParams,
            result=CampaignLine,
            handler=_campaign_runs_add,
            render=render.render_model,
            example={"campaign_id": campaign_id, "run_ids": [run_id]},
            detail=(
                "A run the campaign already holds is not added twice. A run outside the caller's scope is refused, "
                "and then none is added. Once a run is in, campaign_update can name it the control."
            ),
        ),
        Action(
            name="campaign_runs_remove",
            summary="Take runs out of a campaign; the runs and their results stay, in any other campaign too.",
            workflow=ANALYSE,
            permission="write",
            params=CampaignRunsParams,
            result=CampaignLine,
            handler=_campaign_runs_remove,
            render=render.render_model,
            example={"campaign_id": campaign_id, "run_ids": [run_id]},
            detail=(
                "Membership only: nothing is deleted. A run the campaign does not hold is refused, and then none is "
                "removed. The control is a variant, not a run, so it stays declared; where no remaining run carries "
                "it, the report says the control is unobserved."
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
            happy_path=_analysis_generate_path,
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
            render=render.render_model,
            example={"campaign_id": campaign_id},
            detail=(
                "The generation's own assembly and pricing rule, so would_start is analysis_generate's answer. "
                "Makes no model call."
            ),
        ),
        Action(
            name="judge_second",
            summary="Ask a second judge to score a seeded share of a finished run's judged results, as a job; read agreement.",
            workflow=ANALYSE,
            permission="spend",
            params=JudgeSecondParams,
            result=JobsStarted,
            handler=_judge_second,
            long_running=True,
            happy_path=_second_judge_path,
            example={"run_id": run_id, "second_judge_model": "judge/other", "sample_fraction": 0.5},
            detail=(
                "The evidence each result's judge read, sent to the second judge: the run's scores are never changed, "
                "and each second score is recorded beside the first it pairs with. Every call is priced and admitted "
                "against the host's out-of-run cap before the first is sent, and ledgered under purpose second_judge "
                "— measurement cost, never the candidate's. Returns per-dimension n, exact agreement and kappa "
                "(quadratic on 1-5, unweighted on pass/fail; undefined says why), and the drift between the judges — "
                "job_poll's result once the job completes. Refused in the call, with nothing spent, when the run "
                "cannot be asked about or the cap refuses. judge_second_estimate prices it first without spending."
            ),
        ),
        Action(
            name="judge_second_estimate",
            summary="Price a second judge's pass over a run against the host's out-of-run cap, without spending.",
            workflow=ANALYSE,
            permission="read",
            params=JudgeSecondParams,
            result=SecondJudgeEstimate,
            handler=_judge_second_estimate,
            render=render.render_model,
            example={"run_id": run_id, "second_judge_model": "judge/other", "sample_fraction": 0.5},
            detail="The pass's own collection, sample and admission, so would_start is judge_second's answer.",
        ),
        Action(
            name="judge_drift_check",
            summary="Re-score every judged result of a run under a changed judge, as a job; read how far each moved.",
            workflow=ANALYSE,
            permission="spend",
            params=JudgeDriftParams,
            result=JobsStarted,
            handler=_judge_drift_check,
            long_running=True,
            happy_path=_second_judge_path,
            example={"run_id": run_id, "second_judge_model": "judge/new"},
            detail=(
                "judge_second over the whole set. The run's judge, as it recorded it, is the first configuration; "
                "the second names the changed one. Each dimension's movement is read over cases with an interval, "
                "Holm-adjusted over the dimensions: separated, not separated or untested. It detects movement "
                "between the judges, never which is right. The stored scores are never changed. job_poll returns the "
                "read once the job completes."
            ),
        ),
        Action(
            name="judge_temperature",
            summary="Re-judge a run's borderline cases at temperature 0 and at the provider default, as a job; compare.",
            workflow=ANALYSE,
            permission="spend",
            params=JudgeTemperatureParams,
            result=JobsStarted,
            handler=_judge_temperature,
            long_running=True,
            happy_path=_judge_temperature_path,
            example={"run_id": run_id, "repeats": 5},
            detail=(
                "The measurement the judge temperature policy rests on (#633). Each borderline dim (stored score "
                "inside its scale, or a recorded repeat or second judge disagreed; selection=all for every scored "
                "dim) is judged `repeats` times at DEFAULT_JUDGE_TEMPERATURE and `repeats` times with no temperature "
                "sent, from the evidence its judge first read, with the prompt the run recorded. Returns, per "
                "dimension and side by side, the case count, the mean and largest score variance across repeats, the "
                "unstable cases, and exact agreement and kappa read by judge_self_agreement; comparable=false says "
                "when the judge's client did not send the temperature a side names. Every call is priced and "
                "admitted against the host's out-of-run cap before the first is sent, and ledgered under purpose "
                "judge, in the job. Nothing is written to the results: the job's record keeps every answer, and "
                "job_poll returns the comparison once it completes. judge_temperature_estimate prices it first."
            ),
        ),
        Action(
            name="judge_temperature_estimate",
            summary="Price a judge temperature comparison against the host's out-of-run cap, without spending.",
            workflow=ANALYSE,
            permission="read",
            params=JudgeTemperatureParams,
            result=JudgeTemperatureEstimate,
            handler=_judge_temperature_estimate,
            render=JudgeTemperatureEstimate.render,
            example={"run_id": run_id, "repeats": 5},
            detail="The comparison's own collection, selection and admission, so would_start is its answer.",
        ),
        Action(
            name="analyses_list",
            summary="List a campaign's stored analyses.",
            workflow=ANALYSE,
            permission="read",
            params=CampaignParams,
            result=AnalysisListing,
            handler=_analyses_list,
            render=render.render_model,
            example={"campaign_id": campaign_id},
        ),
        Action(
            name="insights_list",
            summary="List the scope's insights — the durable claims analyses minted — newest first.",
            workflow=ANALYSE,
            permission="read",
            params=InsightsListParams,
            result=InsightListing,
            handler=_insights_list,
            render=render.render_model,
            example={"campaign_filter": campaign_id},
            detail=(
                "Each insight with where it stands: live (fed to every later analysis of its subject as prior "
                "context), retracted (the analysis that minted it is archived) or orphaned (that analysis was "
                "deleted). Both filters match ids exactly; one that matches nothing lists nothing."
            ),
        ),
        Action(
            name="insight_get",
            summary="Read one insight in full: its statement, evidence runs, provenance and standing.",
            workflow=ANALYSE,
            permission="read",
            params=InsightParams,
            result=InsightDetail,
            handler=_insight_get,
            render=render.render_model,
            example={"insight_id": insight_id},
        ),
        Action(
            name="analyses_undescribable",
            summary="List the scope's analyses holding an arm whose levels this build cannot describe.",
            workflow=ANALYSE,
            permission="read",
            params=AnalysesUndescribableParams,
            result=UndescribableArmsListing,
            handler=_analyses_undescribable,
            render=render.render_model,
            example={},
            detail=(
                "Read from each stored analysis's variant index on every call, never stored: each such analysis "
                "with how many of its arms carry levels_unavailable and why. Most useful after an identity "
                "version bump, when arms stamped before it can no longer be described."
            ),
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
            name="bars_propose",
            summary="Propose bars from a single-cell baseline campaign's measured incumbent; registers nothing.",
            workflow=ANALYSE,
            permission="read",
            params=CampaignParams,
            result=BarProposals,
            handler=_bars_propose,
            render=bar_proposals_text,
            example={"campaign_id": campaign_id},
            detail=(
                "Measures the baseline campaign's one cell and proposes a bar on each host-declared measure with a "
                "better end, seeded from the incumbent's measured interval. A proposal nothing could fail is flagged "
                "vacuous with why; every reading nothing could be proposed on is named with why. Nothing is "
                "registered: a person adopts a bar by writing it into the host's registrations. A campaign of more "
                "than one cell is refused, since which arm is the incumbent is a person's choice."
            ),
        ),
        Action(
            name="reporter_case_freeze",
            summary="Freeze a campaign's analysis bundle, and the memo it got, into a case of a reporter template.",
            workflow=ANALYSE,
            permission="write",
            params=ReporterCaseFreezeParams,
            result=FrozenReporterCase,
            handler=_reporter_case_freeze,
            render=render.render_model,
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
            render=render.render_model,
            example={"template_id": reporter_template_id},
            detail=(
                "Each case with whether a launch runs it and what superseded it. A case this build cannot read is "
                "listed rather than refused, since this is where it is found; while one is listed, which case is "
                "live cannot be decided. A pair holding more than one live case is named, since every launch of the "
                "template refuses until one freeze supersedes them all."
            ),
        ),
        Action(
            name="judge_template_create",
            summary="Write a judge template: the judge kind and an intent, which judge cases are frozen into.",
            workflow=ANALYSE,
            permission="write",
            params=JudgeTemplateCreateParams,
            result=TemplateLine,
            handler=_judge_template_create,
            render=render.render_model,
            example={
                "template_name": "faithfulness judge",
                "intent": "Score how faithfully an extraction reproduces its invoice.",
            },
            detail=(
                "A judge campaign's template is the package's, not the host's: the judge kind, a name and an intent, "
                "and nothing else — its cases carry everything a trial reads. Freeze cases into it with "
                "judge_cases_freeze, then run_launch it. Refused on a host that does not launch the judge kind, and "
                "for a name the scope already holds. Calls no model."
            ),
        ),
        Action(
            name="judge_cases_freeze",
            summary="Freeze stored judged outputs, their criteria and their labels into cases of a judge template.",
            workflow=ANALYSE,
            permission="write",
            params=JudgeCasesFreezeParams,
            result=JudgeCaseFreezeReport,
            handler=_judge_cases_freeze,
            render=render.render_model,
            example={"template_id": "tmpl-judge", "judged_run_ids": [run_id], "into_case_set": "faithfulness"},
            detail=(
                "A judge campaign measures a judge configuration as its subject, and its cases are never generated: "
                "each is one stored output and the criterion it was judged on, with the evidence its judge read "
                "rebuilt from what its run recorded and the person ratings given on it as labels. One case per judged "
                "dim of each result; a result whose run did not record its judging, or whose template was edited "
                "since, is skipped with why. With into_case_set, the next version of that set lists exactly these cases. "
                "Then run_launch the judge template, one arm per judge (its model, and its overlays config_ids and "
                "temperature); the trials call only the judge. Calls no model."
            ),
        ),
        Action(
            name="judge_case_author",
            summary="Write a judge case by hand: an output, the rubric dimension it is judged on, and its verdict.",
            workflow=ANALYSE,
            permission="write",
            params=JudgeCaseAuthorParams,
            result=JudgeCaseFreezeReport,
            handler=_judge_case_author,
            render=render.render_model,
            example={
                "template_id": "tmpl-judge",
                "criterion_template_id": "tmpl-extraction",
                "criterion": "extraction.faithful",
                "output": '{"invoice_date": "2026-01-03"}',
                "case_material": "Invoice doc-42\ninvoice_date: 2026-03-01",
                "expected": "fail",
                "note": "the date is day-month swapped",
            },
            detail=(
                "Seeds the calibration cases no run stored: a hard negative that must fail a dimension, or an output "
                "that looks wrong and must pass it. The dimension is named by the subject template whose rubric words "
                "it, and the judge reads the case as it reads a stored result (that template's intent unless scenario "
                "is given, the case material, the output). The verdict is 'pass'/'fail' (or 1/0) on a pass/fail "
                "dimension and 1-5 on an ordinal one; a verdict the scale cannot hold is refused, naming both. The "
                "verdict is the case's one label and the caller its author, so a judge campaign's agreement and a "
                "judge profile read it as a labelled case. Authoring the same case again names the stored one. It "
                "mints no case set: case_set_mint lists the cases a launch runs. Calls no model."
            ),
        ),
        Action(
            name="judge_profiles_record",
            summary="Store what a judge campaign measured each judge to be, per criterion, for other campaigns to read.",
            workflow=ANALYSE,
            permission="write",
            params=JudgeProfilesRecordParams,
            result=JudgeProfileRecording,
            handler=_judge_profiles_record,
            render=render.render_model,
            example={"judge_run_ids": [run_id]},
            detail=(
                "Reads the judge campaign's finished runs (judge_run_ids, every arm) as judge_kind_readings does, and "
                "stores one profile per judge (model, prompt config, temperature) and criterion: agreement with the "
                "labels, self-agreement, parse validity, the cases and runs measured on and when. Replaces the stored "
                "profile of the same judge and criterion. A judge whose served model or temperature was not recorded "
                "is skipped with why. Any campaign that judge scores then reads its evidence tier from the profile "
                "where its own evidence decides none, and says so. Calls no model."
            ),
        ),
        Action(
            name="judge_profiles_list",
            summary="List the stored judge profiles: each judge and criterion measured, on what, when, and its tier.",
            workflow=ANALYSE,
            permission="read",
            params=JudgeProfilesListParams,
            result=JudgeProfileListing,
            handler=_judge_profiles_list,
            render=render.render_model,
            example={},
            detail=(
                "A profile is read only for the very judge it measured: a changed model, prompt config or temperature, "
                "or a reworded criterion, is another judge whose profile does not exist."
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
            render=pivot_text,
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
            render=runs_compared_text,
            example={"baseline_run_id": run_id, "candidate_run_id": "0193a1b2-run-b"},
            detail=(
                "pass^k is read on both arms at the shallower arm's depth, over the cases each averaged there, and "
                "names what its attempts were held to — goal checks, a judge's criteria at the threshold, or both. "
                "The composite is tested paired over the cases both runs scored, else unpaired, and the answer says "
                "which; where no case carries a composite in both, no test ran and it says why. A miss reads not "
                "significant, "
                "never no difference. Composites of two different subjects are not compared. Each run that "
                "delivered less than its matrix carries its sentence, and the answer says when the two were "
                "measured over spans that do not overlap or recorded different cassette modes. A run not in the "
                "caller's scope is not found."
            ),
        ),
        Action(
            name="runs_bisect",
            summary="Split the versioned inputs of two runs into those that differ, agree and cannot be decided.",
            workflow=ANALYSE,
            permission="read",
            params=RunsBisectParams,
            result=RunsBisected,
            handler=_runs_bisect,
            render=runs_bisected_text,
            example={"baseline_run_id": run_id, "candidate_run_id": "0193a1b2-run-b"},
            detail=(
                "The what-changed question behind a history step: every input the host declares as sweepable, each "
                "with both runs' values. An input one run or both recorded nothing for is undecided, never the same. "
                "Each run that delivered less than its matrix carries its sentence, and the answer says when the two "
                "were measured over spans that do not overlap. A run not in the caller's scope is not found."
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
            render=out_of_run_spend_text,
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
            render=history_text,
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
            render=frontier_text,
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
            render=export_text,
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
            render=render.render_model,
            example={"run_id": run_id, "archived": True},
        ),
        Action(
            name="run_label",
            summary="Name a run with labels (or clear them): runs_list lists runs by one, no key moves.",
            workflow=CURATE,
            permission="write",
            params=RunLabelParams,
            result=RunLine,
            handler=_run_label,
            render=render.render_model,
            example={"run_id": run_id, "run_labels": ["baseline"]},
        ),
        Action(
            name="campaign_archive",
            summary="Archive a campaign (or restore it): out of listings, nothing destroyed.",
            workflow=CURATE,
            permission="write",
            params=CampaignArchiveParams,
            result=CampaignLine,
            handler=_campaign_archive,
            render=render.render_model,
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
            render=render.render_model,
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
            render=render.render_model,
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
            render=render.render_model,
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
            render=render.render_model,
            example={"run_id": run_id, "confirm": run_id},
            detail=(
                "Removes the run's results (each with its trace) and its campaign memberships, then the run; "
                "insights citing it and ratings of its results stay. A run this build cannot read (runs_unreadable "
                "lists them) is deleted from its stored row, removing the same: its results are found by the run id "
                "they carry, never read. A pending or running run is refused; one this build cannot read only while "
                "its stored lease has not lapsed, or when its lease cannot be read either, since a stranded one can "
                "never be settled but one another build may still hold must not be destroyed."
            ),
        ),
        Action(
            name="analysis_delete",
            summary="Destroy a stored analysis; the insights it minted remain. Unrecoverable; archive instead.",
            workflow=CURATE,
            permission="destructive",
            params=AnalysisDeleteParams,
            result=AnalysisDeleted,
            handler=_analysis_delete,
            render=render.render_model,
            example={"analysis_id": analysis_id, "confirm": analysis_id},
        ),
        Action(
            name="insight_delete",
            summary="Destroy one insight, so no later analysis reads it as prior context. Unrecoverable.",
            workflow=CURATE,
            permission="destructive",
            params=InsightDeleteParams,
            result=InsightDeleted,
            handler=_insight_delete,
            render=render.render_model,
            example={"insight_id": insight_id, "confirm": insight_id},
            detail=(
                "An insight has no archive: a wrong one keeps steering every later analysis of its subject until it "
                "is gone, so deleting it is the intended answer. To withdraw everything one analysis minted, "
                "archive the analysis instead (analysis_archive), which retracts its insights reversibly."
            ),
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
