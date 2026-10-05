"""What an agent reads back: help generated from the catalogue, refusals that teach, and each result as text.

Every string a mounted tool returns is made here, so the conventions hold in one place:

- **Help is generated, never written.** The index is the tool's actions grouped by workflow with each
  one's class and summary; a page is one action's parameters, what it returns and an example call —
  all read off the :class:`~threetears.evals.actions.catalogue.Action` and its models.
- **A refusal names what was wrong and what would be right**: the valid actions, the parameters the
  action accepts with what each means, and an example call that validates.
- **A result reads as the decision it supports.** A started job says how to poll it; a finished one
  says what to read next.

This module reads the catalogue's types only for annotations, so the catalogue imports it freely.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from threetears.evals.contracts.errors import EvalServiceError
from threetears.evals.ops import (
    AnalysisDeleted,
    AnalysisListing,
    CampaignLine,
    CampaignListing,
    CostEstimate,
    EvalSummary,
    HistoryResult,
    JobsStarted,
    JobStatus,
    OutOfRunSpendReport,
    PivotTable,
    ReportDocument,
    RunDeleted,
    RunLine,
    RunListing,
    ScoreExport,
    TemplateListing,
    estimate_text,
    export_text,
    history_text,
    out_of_run_spend_text,
    pivot_text,
)

if TYPE_CHECKING:
    from threetears.evals.actions.catalogue import Action, MountedTool


# --- help --------------------------------------------------------------------------------------------


def _type_word(schema: Mapping[str, Any]) -> str:
    """A parameter's type as a reader says it: ``string``, ``list of string``, ``one of a|b``, ``optional …``."""
    if "enum" in schema:
        return "one of " + "|".join(str(value) for value in schema["enum"])
    if "const" in schema:
        return str(schema["const"])
    if "anyOf" in schema:
        kinds = [option for option in schema["anyOf"] if option.get("type") != "null"]
        words = " or ".join(_type_word(option) for option in kinds)
        nullable = len(kinds) < len(schema["anyOf"])
        return f"optional {words}" if nullable else words
    kind = schema.get("type", "any")
    if kind == "array":
        return f"list of {_type_word(schema.get('items', {}))}"
    return str(kind)


def parameter_lines(action: Action) -> list[str]:
    """One line per parameter the action accepts: name, type, whether required, what it means."""
    schema = action.params.model_json_schema()
    required = set(schema.get("required", []))
    lines = []
    for name, prop in schema.get("properties", {}).items():
        need = "required" if name in required else f"default {json.dumps(prop['default'])}" if "default" in prop else ""
        lines.append(f"- {name} ({_type_word(prop)}{', ' + need if need else ''}): {prop.get('description', '')}")
    return lines


def example_call(action: Action) -> str:
    """The action's example, as the JSON arguments of a call."""
    return json.dumps({"action": action.name, **dict(action.example)}, sort_keys=False)


def render_help_index(tool: MountedTool) -> str:
    """The tool's actions grouped by workflow, in catalogue order, each with its class and summary."""
    groups: dict[str, list[Action]] = {}
    for action in tool.actions:
        groups.setdefault(action.workflow, []).append(action)
    lines = [f"{tool.name}: {tool.spec.description}", ""]
    for workflow, actions in groups.items():
        lines.append(f"## {workflow}")
        for action in actions:
            job = ", starts a job" if action.long_running else ""
            lines.append(f"- {action.name} ({action.permission}{job}): {action.summary}")
        lines.append("")
    lines.append("For one action's parameters and an example: action='help', topic='<action>'.")
    return "\n".join(lines)


def render_help_page(tool: MountedTool, action: Action) -> str:
    """One action's page: what it does, its class, its parameters, what it returns and an example call."""
    job = (
        " It starts long work and returns a job per piece of it; poll each with action='job_poll', job_id='<id>'."
        if action.long_running
        else ""
    )
    lines = [f"# {action.name}", "", action.summary + job]
    if action.detail:
        lines += ["", action.detail]
    lines += ["", f"Class: {action.permission} (tool {tool.name}).", "", "Parameters:"]
    lines += parameter_lines(action) or ["- none"]
    returned = ", ".join(action.result.model_fields)
    lines += ["", f"Returns {action.result.__name__}: {returned}.", "", f"Example: {example_call(action)}"]
    return "\n".join(lines)


# --- refusals ----------------------------------------------------------------------------------------


def _valid_actions(tool: MountedTool) -> str:
    return "valid actions: help, " + ", ".join(action.name for action in tool.actions)


def _accepted(action: Action) -> str:
    lines = parameter_lines(action) or ["- none"]
    return "\n".join([f"{action.name} accepts:", *lines, f"Example: {example_call(action)}"])


def refuse_no_action(tool: MountedTool) -> str:
    """A call that named no action."""
    return f"refused: name the action to run (`action`). {_valid_actions(tool)}. Start with action='help'."


def refuse_unknown_action(tool: MountedTool, name: str) -> str:
    """A call naming an action this tool does not carry — saying where it lives when the catalogue has it."""
    elsewhere = tool.catalogue.get(name)
    if elsewhere is not None:
        return (
            f"refused: {name} is a {elsewhere.permission} action, and tool {tool.name} carries "
            f"{', '.join(sorted(tool.spec.permissions))} actions only — call it on the tool that mounts "
            f"{elsewhere.permission} actions. {_valid_actions(tool)}."
        )
    return f"refused: there is no action {name!r}. {_valid_actions(tool)}."


def refuse_help_parameters(tool: MountedTool, undeclared: Sequence[str]) -> str:
    """A help call carrying parameters help does not take."""
    return (
        f"refused: help takes only `topic` (an action's name); it does not take {', '.join(undeclared)}. "
        f"{_valid_actions(tool)}."
    )


def refuse_undeclared(tool: MountedTool, action: Action, undeclared: Sequence[str]) -> str:
    """A call carrying parameters its action does not declare — refused rather than ignored."""
    del tool
    return f"refused: {action.name} does not take {', '.join(undeclared)}.\n{_accepted(action)}"


def refuse_invalid(tool: MountedTool, action: Action, invalid: ValidationError) -> str:
    """A call whose values do not validate against the action's parameters."""
    del tool
    problems = []
    for error in invalid.errors():
        where = ".".join(str(part) for part in error["loc"]) or "(the call)"
        problems.append(f"- {where}: {error['msg']}")
    return "\n".join([f"refused: {action.name} was called with values it cannot take:", *problems, _accepted(action)])


def refuse_by_engine(tool: MountedTool, action: Action, refused: EvalServiceError) -> str:
    """A call the engine refused, with the engine's own reason."""
    del tool
    return f"refused: {action.name}: {refused.message}"


# --- results -----------------------------------------------------------------------------------------


def render_templates(listing: TemplateListing) -> str:
    """A scope's templates."""
    lines = [f"templates ({len(listing.templates)})"]
    lines += [f"- {t.id}: {t.name} — kind {t.candidate_kind}; {t.intent}" for t in listing.templates]
    return "\n".join(lines)


def render_run_line(run: RunLine) -> str:
    """One run on one line."""
    archived = ", archived" if run.archived else ""
    return f"- {run.id}: {run.status}, model {run.candidate_model}, template {run.template_id}{archived}"


def render_runs(listing: RunListing) -> str:
    """A scope's runs."""
    scope = "including archived" if listing.include_archived else "archived runs left out"
    return "\n".join([f"runs ({len(listing.runs)}, {scope})", *(render_run_line(run) for run in listing.runs)])


def render_summary(summary: EvalSummary) -> str:
    """One run's summary."""
    return summary.render()


def render_jobs_started(started: JobsStarted) -> str:
    """What a start returned: each job, and how to poll it."""
    lines = [f"started {len(started.jobs)} job(s):"]
    lines += [
        f"- {job.job_id} ({job.kind} {job.target_id}{', ' + job.label if job.label else ''})" for job in started.jobs
    ]
    lines.append("Poll each with action='job_poll', job_id='<id>' until it reads done.")
    return "\n".join(lines)


def render_job_status(status: JobStatus) -> str:
    """Where a job stands, and what to read next when it has ended."""
    lines = [f"job {status.job_id}: {status.state} ({status.status}){'' if status.done else ', still going'}"]
    if status.progress:
        lines.append("progress: " + json.dumps(status.progress, sort_keys=True))
    if status.detail:
        lines.append(f"detail: {status.detail}")
    if status.state == "completed" and status.analysis_id and status.campaign_id:
        lines.append(f"Read it with action='report_read', campaign_id='{status.campaign_id}'.")
    elif status.done and status.run_id:
        lines.append(f"Read it with action='run_get', run_id='{status.run_id}'.")
    return "\n".join(lines)


def render_campaign(campaign: CampaignLine) -> str:
    """One campaign on one line."""
    archived = ", archived" if campaign.archived else ""
    return (
        f"- {campaign.id}: {campaign.name} — subject {campaign.subject_id}, {campaign.behavior}; "
        f"{campaign.run_count} run(s), {campaign.status}{archived}"
    )


def render_campaigns(listing: CampaignListing) -> str:
    """A scope's campaigns."""
    return "\n".join([f"campaigns ({len(listing.campaigns)})", *(render_campaign(c) for c in listing.campaigns)])


def render_analyses(listing: AnalysisListing) -> str:
    """A campaign's analyses."""
    lines = [f"analyses of campaign {listing.campaign_id} ({len(listing.analyses)})"]
    lines += [
        f"- {a.id}: {a.headline} ({a.generator_model}, {a.generated_at}{', archived' if a.archived else ''})"
        for a in listing.analyses
    ]
    return "\n".join(lines)


def render_report(document: ReportDocument) -> str:
    """The report itself, in the form asked for — which already says whether it is an analysis or code-only."""
    return document.body


def render_pivot(table: PivotTable) -> str:
    """A pivot: each cell with its denominators, and every caveat the table carries."""
    return pivot_text(table)


def render_history(result: HistoryResult) -> str:
    """Each contestant's series, with each step's regression verdict and the test behind it."""
    return history_text(result)


def render_out_of_run_spend(report: OutOfRunSpendReport) -> str:
    """The scope's out-of-run spend: totals overall, per purpose and per launch, then each call."""
    return out_of_run_spend_text(report)


def render_estimate(estimate: CostEstimate) -> str:
    """A launch's predicted cost, model by model, and the total with its band."""
    return estimate_text(estimate)


def render_export(export: ScoreExport) -> str:
    """The export, after a line saying how many rows it holds and what it left out."""
    return export_text(export)


def render_run_deleted(deleted: RunDeleted) -> str:
    """What deleting a run removed."""
    detached = ", ".join(deleted.campaigns_detached) or "none"
    return f"deleted run {deleted.run_id}, its {deleted.results_deleted} result(s); detached from campaigns: {detached}"


def render_analysis_deleted(deleted: AnalysisDeleted) -> str:
    """What deleting an analysis removed."""
    return f"deleted analysis {deleted.analysis_id} of campaign {deleted.campaign_id}; the insights it minted remain"


__all__ = [
    "example_call",
    "parameter_lines",
    "refuse_by_engine",
    "refuse_help_parameters",
    "refuse_invalid",
    "refuse_no_action",
    "refuse_undeclared",
    "refuse_unknown_action",
    "render_analyses",
    "render_analysis_deleted",
    "render_campaign",
    "render_campaigns",
    "render_estimate",
    "render_export",
    "render_help_index",
    "render_help_page",
    "render_history",
    "render_job_status",
    "render_jobs_started",
    "render_pivot",
    "render_report",
    "render_run_deleted",
    "render_run_line",
    "render_runs",
    "render_summary",
    "render_templates",
]
