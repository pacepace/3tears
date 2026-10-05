"""Typed operations over a host: what every surface — a CLI, an MCP tool, a REST route — calls.

Each operation takes the host and the scope and returns a typed value, so the surfaces above read one
shape rather than each re-typing an untyped ``dict``. Long work follows **one job contract**: an
operation that starts it (:func:`run_launch`, :func:`analysis_generate`) returns a :class:`JobsStarted`
naming one job per piece of work, and :func:`job_poll` / :func:`job_cancel` take any job id either
returned. A job is the name of the durable record its work writes as it ends, so a job id stays
answerable across a restart.

The operations are named as the actions over them are (``noun_verb``), so a name in the action
catalogue (:mod:`threetears.evals.actions`) is the operation it calls.

The read lenses — :func:`scope_pivot`, :func:`scope_history`, :func:`scope_export` and
:func:`launch_estimate` — return the analysis package's own result models, re-exported here because
they are what these operations hand back, with the text a surface shows for each (:func:`pivot_text`
and its siblings) beside them, so a command line and an agent read one rendering.

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.ops.analysis import (
    AnalysisDeleted,
    AnalysisLine,
    AnalysisListing,
    CampaignDefinition,
    CampaignLine,
    CampaignListing,
    ReportDocument,
    ReportFormat,
    analyses_list,
    analysis_delete,
    analysis_generate,
    campaign_archive,
    campaign_create,
    campaigns_list,
    generation_key,
    report_read,
    serialize_report,
)
from threetears.evals.ops.host import AnalysisGeneration, OpsHost, TemplateCaseCounter
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
    job_cancel,
    job_poll,
    parse_job_id,
    run_job_id,
)
from threetears.evals.analysis import CostEstimate, HistoryResult, PivotTable, ScoreExport
from threetears.evals.ops.lenses import (
    estimate_text,
    export_text,
    history_text,
    launch_estimate,
    pivot_text,
    scope_export,
    scope_history,
    scope_pivot,
)
from threetears.evals.ops.runs import (
    LaunchArguments,
    RunDeleted,
    RunLine,
    RunListing,
    TemplateLine,
    TemplateListing,
    run_archive,
    run_delete,
    run_get,
    run_launch,
    runs_list,
    templates_list,
)
from threetears.evals.ops.summary import EvalSummary, MeasureSummary, summarize_run

__all__ = [
    "ANALYSIS_JOB_PREFIX",
    "RUN_JOB_PREFIX",
    "TERMINAL_JOB_STATES",
    "AnalysisDeleted",
    "AnalysisGeneration",
    "AnalysisLine",
    "AnalysisListing",
    "CampaignDefinition",
    "CampaignLine",
    "CampaignListing",
    "CostEstimate",
    "EvalSummary",
    "HistoryResult",
    "JobHandle",
    "JobKind",
    "JobState",
    "JobStatus",
    "JobsStarted",
    "LaunchArguments",
    "MeasureSummary",
    "OpsHost",
    "PivotTable",
    "ReportDocument",
    "ReportFormat",
    "RunDeleted",
    "RunLine",
    "RunListing",
    "ScoreExport",
    "TemplateCaseCounter",
    "TemplateLine",
    "TemplateListing",
    "analyses_list",
    "analysis_delete",
    "analysis_generate",
    "analysis_job_id",
    "campaign_archive",
    "campaign_create",
    "campaigns_list",
    "estimate_text",
    "export_text",
    "generation_key",
    "history_text",
    "job_cancel",
    "job_poll",
    "launch_estimate",
    "parse_job_id",
    "pivot_text",
    "report_read",
    "serialize_report",
    "run_archive",
    "run_delete",
    "run_get",
    "run_job_id",
    "run_launch",
    "runs_list",
    "scope_export",
    "scope_history",
    "scope_pivot",
    "summarize_run",
    "templates_list",
]
