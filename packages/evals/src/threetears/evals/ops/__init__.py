"""Typed operations over a host: what every surface — a CLI, an MCP tool, a REST route — calls.

Each operation takes the host and the scope and returns a typed value, so the surfaces above read one
shape rather than each re-typing an untyped ``dict``. Long work follows **one job contract**: an
operation that starts it (:func:`run_launch`, :func:`analysis_generate`) returns a :class:`JobsStarted`
naming one job per piece of work, and :func:`job_poll` / :func:`job_cancel` take any job id either
returned. A job is the name of the durable record its work writes as it ends, so a job id stays
answerable across a restart.

The operations are named as the actions over them are (``noun_verb``), so a name in the action
catalogue (:mod:`threetears.evals.actions`) is the operation it calls.

The read lenses — :func:`scope_pivot`, :func:`scope_history`, :func:`scope_export`, :func:`scope_frontier`,
:func:`runs_compare` and :func:`launch_estimate` — return their result models (the first four the analysis package's own,
re-exported here because they are what these operations hand back; the comparison its own
:class:`RunsCompared`, the two-run lens's answer with the disclosures every comparison carries; the
estimate its own :class:`LaunchEstimate`, priced by the launch's rule), with the text a surface shows for each (:func:`pivot_text`
and its siblings) beside them, so a command line and an agent read one rendering.
:func:`scope_out_of_run_spend` reads the out-of-run ledger — the spend no run's results carry — into an
:class:`OutOfRunSpendReport` of its own, with :func:`out_of_run_spend_text` beside it.

A run's results are read by :func:`results_list` — one light row per result, paged, so an operator can find
the cell that came out wrong — and :func:`result_get`, which reads one result back as stored with one part of its
trace: by default the output its kind stored and the roles' spend, and the judge's evidence or the spans by name.

The reporter case bank's operations — :func:`reporter_case_freeze`, :func:`reporter_cases_list` and
:func:`reporter_case_archive` — are how a surface makes the cases a reporter run measures: each is one
campaign's analysis bundle, frozen, and a reporter template launches nothing until one is.

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.ops.analysis import (
    AnalysisDeleted,
    AnalysisGenerationEstimate,
    AnalysisLine,
    AnalysisListing,
    CampaignDefinition,
    CampaignLine,
    CampaignListing,
    ReportDocument,
    ReportFormat,
    analyses_list,
    analysis_archive,
    analysis_delete,
    analysis_estimate,
    analysis_generate,
    campaign_archive,
    campaign_create,
    campaigns_list,
    report_read,
    serialize_report,
)
from threetears.evals.ops.host import AnalysisGeneration, OpsHost
from threetears.evals.ops.jobs import (
    ANALYSIS_JOB_PREFIX,
    RUN_JOB_PREFIX,
    TERMINAL_JOB_STATES,
    JobHandle,
    JobKind,
    JobsStarted,
    JobState,
    JobStatus,
    analysis_job_id,
    generation_key,
    job_cancel,
    job_poll,
    parse_job_id,
    run_job_id,
)
from threetears.evals.analysis import CostEstimate, FrontierResult, HistoryResult, PivotTable, ScoreExport
from threetears.evals.ops.lenses import (
    ArmEstimate,
    LaunchEstimate,
    OutOfRunSpendReport,
    OutOfRunSpendTotals,
    estimate_text,
    export_text,
    frontier_text,
    history_text,
    history_launch_pricer,
    launch_estimate,
    out_of_run_spend_text,
    pivot_text,
    RunsCompared,
    runs_compare,
    runs_compared_text,
    scope_export,
    scope_frontier,
    scope_history,
    scope_out_of_run_spend,
    scope_pivot,
)
from threetears.evals.analysis.reporter_bank import FrozenReporterCase
from threetears.evals.ops.reporter import (
    AmbiguousReporterPair,
    ReporterCaseEntry,
    ReporterCaseFreeze,
    ReporterCaseListing,
    UnreadableReporterCase,
    reporter_case_archive,
    reporter_case_freeze,
    reporter_cases_list,
)
from threetears.evals.ops.results import (
    ResultDetail,
    ResultLine,
    ResultListing,
    ResultPart,
    TraceJudge,
    TraceRecord,
    TraceState,
    result_get,
    results_list,
)
from threetears.evals.ops.runs import (
    LaunchArguments,
    ResultRated,
    RunDeleted,
    RunLine,
    RunListing,
    TemplateLine,
    TemplateListing,
    judge_repeat,
    judge_repeat_estimate,
    result_rate,
    run_archive,
    run_delete,
    run_get,
    run_launch,
    runs_list,
    templates_list,
)
from threetears.evals.ops.summary import (
    CaseResult,
    DimensionSummary,
    EvalSummary,
    JudgeGrade,
    MeasureSummary,
    dollars_text,
    summarize_run,
)

__all__ = [
    "ANALYSIS_JOB_PREFIX",
    "RUN_JOB_PREFIX",
    "TERMINAL_JOB_STATES",
    "AmbiguousReporterPair",
    "AnalysisDeleted",
    "AnalysisGenerationEstimate",
    "ArmEstimate",
    "AnalysisGeneration",
    "AnalysisLine",
    "AnalysisListing",
    "CampaignDefinition",
    "CampaignLine",
    "CampaignListing",
    "CaseResult",
    "CostEstimate",
    "DimensionSummary",
    "EvalSummary",
    "FrozenReporterCase",
    "FrontierResult",
    "HistoryResult",
    "JobHandle",
    "JobKind",
    "JobState",
    "JobStatus",
    "JobsStarted",
    "JudgeGrade",
    "LaunchArguments",
    "LaunchEstimate",
    "MeasureSummary",
    "OpsHost",
    "OutOfRunSpendReport",
    "OutOfRunSpendTotals",
    "PivotTable",
    "ReportDocument",
    "ReportFormat",
    "ReporterCaseEntry",
    "ReporterCaseFreeze",
    "ReporterCaseListing",
    "ResultDetail",
    "ResultLine",
    "ResultListing",
    "ResultPart",
    "ResultRated",
    "RunDeleted",
    "RunLine",
    "RunListing",
    "RunsCompared",
    "ScoreExport",
    "TemplateLine",
    "TemplateListing",
    "TraceJudge",
    "TraceRecord",
    "TraceState",
    "UnreadableReporterCase",
    "analyses_list",
    "analysis_archive",
    "analysis_delete",
    "analysis_estimate",
    "analysis_generate",
    "analysis_job_id",
    "campaign_archive",
    "campaign_create",
    "campaigns_list",
    "dollars_text",
    "estimate_text",
    "export_text",
    "frontier_text",
    "generation_key",
    "history_text",
    "job_cancel",
    "job_poll",
    "history_launch_pricer",
    "launch_estimate",
    "out_of_run_spend_text",
    "parse_job_id",
    "pivot_text",
    "report_read",
    "reporter_case_archive",
    "reporter_case_freeze",
    "reporter_cases_list",
    "serialize_report",
    "result_get",
    "result_rate",
    "results_list",
    "run_archive",
    "run_delete",
    "run_get",
    "run_job_id",
    "run_launch",
    "runs_compare",
    "runs_compared_text",
    "runs_list",
    "scope_export",
    "scope_frontier",
    "scope_history",
    "scope_out_of_run_spend",
    "scope_pivot",
    "summarize_run",
    "templates_list",
    "judge_repeat",
    "judge_repeat_estimate",
]
