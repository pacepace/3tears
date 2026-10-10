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

from threetears.evals.contracts import GoalStateOutcome, ResultOutcome, counted_goal_verdicts
from threetears.evals.contracts.errors import EvalServiceError
from threetears.evals.ops import (
    AnalysisDeleted,
    CaseSetLine,
    CaseSetListing,
    AnalysisGenerationEstimate,
    AnalysisLine,
    AnalysisListing,
    BarProposals,
    CampaignLine,
    CampaignListing,
    LaunchEstimate,
    EvalSummary,
    FrozenReporterCase,
    FrontierResult,
    HistoryResult,
    InsightDeleted,
    InsightDetail,
    InsightLine,
    InsightListing,
    JobsStarted,
    JobStatus,
    OutOfRunSpendReport,
    PivotTable,
    ReportDocument,
    ReporterCaseEntry,
    ReporterCaseListing,
    ResultDetail,
    ResultLine,
    ResultListing,
    ResultRated,
    RunDeleted,
    RunLine,
    RunListing,
    RunsCompared,
    ScoreExport,
    SecondJudgeRead,
    TemplateListing,
    bar_proposals_text,
    UndescribableArmsListing,
    dollars_text,
    estimate_text,
    format_number,
    export_text,
    frontier_text,
    history_text,
    out_of_run_spend_text,
    pivot_text,
    runs_compared_text,
)

if TYPE_CHECKING:
    from threetears.evals.actions.catalogue import Action, MountedTool
    from threetears.evals.run import SecondJudgeEstimate


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


def _usd(amount: float | None) -> str:
    """Spend as a run's summary prints it (:func:`~threetears.evals.ops.dollars_text`); unpriced says so, never $0."""
    return "unpriced" if amount is None else dollars_text(amount)


def _compact(value: Any) -> str:
    """A stored JSON value on one line, keys sorted, as written — never re-typed or summarised."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _result_line(line: ResultLine) -> str:
    """One result on one line: where it sits, its condition, and its headline measures."""
    parts = [
        f"{line.condition.value} ({line.termination})",
        f"cost {_usd(line.cost_usd)}",
        f"total_ms {_ms(line.total_ms)}",
    ]
    if line.goal_checks:
        passed = "not counted" if line.goal_checks_passed is None else f"{line.goal_checks_passed}/{line.goal_checks}"
        parts.append(f"goal checks {passed}")
    if line.judge_scores:
        parts.append("judged " + ", ".join(f"{dim}={score}" for dim, score in line.judge_scores.items()))
    if line.host_measures:
        parts.append("measures " + ", ".join(f"{name}={value}" for name, value in line.host_measures.items()))
    if not line.has_trace:
        parts.append("no trace")
    return f"- {line.id}: case {line.test_case_id} k={line.k_iteration}, {line.model}: " + "; ".join(parts)


def render_results(listing: ResultListing, *, default_limit: int | None = None) -> str:
    """One page of a run's results, and how to read the next page and one result.

    Args:
        listing: The page.
        default_limit: The page size the calling surface uses when none is named; the next-page hint names
            ``limit`` whenever this page's differs, so following the hint reads a page of the same size.

    Returns:
        The text.
    """
    narrowed = "" if listing.condition_filter is None else f" {listing.condition_filter.value}"
    end = listing.offset + len(listing.results)
    lines = [
        f"results of run {listing.run_id}: {listing.total}{narrowed}, rows {listing.offset + 1}-{end} shown"
        if listing.results
        else f"results of run {listing.run_id}: {listing.total}{narrowed}, none from row {listing.offset + 1}",
        *(_result_line(line) for line in listing.results),
    ]
    if listing.next_offset is not None:
        hint = f"More: action='results_list', run_id='{listing.run_id}', offset={listing.next_offset}"
        if listing.condition_filter is not None:
            hint += f", condition_filter='{listing.condition_filter.value}'"
        if listing.limit is not None and listing.limit != default_limit:
            hint += f", limit={listing.limit}"
        lines.append(hint + ".")
    if listing.results:
        lines.append("Read one with action='result_get', result_id='<id>'.")
    return "\n".join(lines)


def _goal_check_line(outcome: GoalStateOutcome, counted: bool | None, condition: ResultOutcome) -> str:
    """One goal check as evaluated, and as every rate counts it when the two differ.

    A candidate failure counts every check failed and a harness fault counts none
    (:func:`~threetears.evals.contracts.counted_goal_verdicts`), so a check that evaluated True on such a
    result says both, rather than reading as a pass the listing's count does not hold.
    """
    evaluated = "passed" if outcome.passed else "failed"
    if counted is None:
        verdict = f"{evaluated} as evaluated; not counted (harness fault)"
    elif counted != outcome.passed:
        cause = "candidate failure" if condition is ResultOutcome.CANDIDATE_FAIL else condition.value
        verdict = f"{evaluated} as evaluated; counts {'passed' if counted else 'failed'} ({cause})"
    else:
        verdict = evaluated
    return f"goal check {outcome.expression}: {verdict}" + (f" — {outcome.detail}" if outcome.detail else "")


def _ms(value: float | None) -> str:
    return "absent" if value is None else f"{format_number(value)}ms"


def _latency_lines(detail: ResultDetail) -> list[str]:
    """The stored latency block, an absent component shown as absent, then the remainder as the partition read it."""
    latency = detail.result.latency
    if latency is None:
        stored = "latency: none recorded"
    else:
        stored = "latency: " + ", ".join(
            f"{name} {_ms(getattr(latency, name))}"
            for name in ("total_ms", "llm_ms", "tool_ms", "async_wait_ms", "judge_ms")
        )
    partition = detail.latency_partition
    if partition.withheld is not None:
        return [stored, f"orchestration_ms withheld: {partition.withheld}"]
    return [stored, f"orchestration_ms {_ms(partition.orchestration_ms)} (total_ms less llm_ms and tool_ms)"]


def _record_lines(detail: ResultDetail) -> list[str]:
    """The record part: errors, spend, latency and usage rows, checks and scores, then what the kind stored."""
    result, condition = detail.result, detail.condition
    lines = [
        f"{label}: {error}"
        for label, error in (
            ("candidate error", result.candidate_error),
            ("infra error", result.infra_error),
            ("judge error", result.judge_error),
        )
        if error
    ]
    lines.append(f"cost {_usd(result.cost_usd)} over {', '.join(result.cost_roles) or 'no role'}")
    lines += _latency_lines(detail)
    lines.append(f"usage ({len(result.usage)} row(s)):")
    lines += [f"- {_compact(row.model_dump(mode='json', exclude_none=True))}" for row in result.usage]
    counted = counted_goal_verdicts(result)
    lines += [
        _goal_check_line(outcome, None if counted is None else counted[index][1], condition.scoring)
        for index, outcome in enumerate(result.goal_state_outcomes)
    ]
    lines += [f"judged {score.dim}: {score.score} ({score.scale})" for score in result.judge_scores()]
    lines += [f"judge could not tell on {dim}: {reason}" for dim, reason in result.judge_cannot_tell.items()]
    if result.host_measures:
        lines.append(f"host measures: {_compact(result.host_measures)}")
    record = detail.record
    if record is None:
        return lines
    lines.append(f"output ({len(record.output)} document(s), as the kind stored them):")
    lines += [f"- {_compact(document)}" for document in record.output]
    if record.call_ledger is None:
        lines.append("call ledger: none kept")
    else:
        lines.append(f"call ledger ({len(record.call_ledger.calls)} call(s) the kind recorded as succeeded):")
        lines += [f"- {call.tool}.{call.action} {_compact(call.params)}" for call in record.call_ledger.calls]
    if record.end_state is not None:
        lines.append(f"end state: {_compact(record.end_state)}")
    if record.judged_artifact is not None:
        lines.append(
            f"judge evidence ({record.judged_artifact}) left out: read it with action='result_get', "
            f"result_id='{result.id}', part='judge'."
        )
    if record.span_count:
        lines.append(
            f"{record.span_count} span(s) left out: read them with action='result_get', result_id='{result.id}', "
            "part='spans'."
        )
    return lines


def render_result(detail: ResultDetail) -> str:
    """One stored result and the part of its trace asked for.

    The record part prints the output documents one per line exactly as the kind stored them, so whatever the
    kind wrote about a call — that it failed, what the tool said — reads in the kind's own words; nothing here
    interprets them. The judge's evidence and the spans are printed only when their part is asked for, and the
    record part says how to ask.
    """
    result, condition = detail.result, detail.condition
    lines = [
        f"result {result.id} of run {result.eval_run_id}: case {result.test_case_id} k={result.k_iteration}, "
        f"model {result.model}, kind {result.candidate_kind}",
        f"condition: {condition.scoring.value} (termination {condition.termination}, judging {condition.judging})",
    ]
    if condition.disclosure:
        lines.append(f"  {condition.disclosure}")
    if detail.part == "record":
        lines += _record_lines(detail)
    if detail.trace_state == "none":
        lines.append("trace: none stored")
    elif detail.trace_state == "missing":
        lines.append("trace: recorded but its document is missing")
    elif detail.part == "judge":
        if detail.judge is None:
            lines.append("judge evidence: none stored — nothing was sent to a judge for this cell")
        else:
            evidence = detail.judge.evidence
            lines.append(f"judge evidence ({detail.judge.judged_artifact}):")
            if evidence.subject is not None:
                lines += ["subject:", evidence.subject]
            lines += ["case material:", evidence.case_material, "artifact:", evidence.artifact]
    elif detail.part == "spans":
        spans = detail.spans or []
        lines.append(f"spans ({len(spans)}):")
        lines += [f"- {_compact(span)}" for span in spans]
    return "\n".join(lines)


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
        f"{campaign.run_count} run(s){archived}"
    )


def render_case_set(case_set: CaseSetLine) -> str:
    """One version of a case set on one line."""
    tracked = "" if case_set.tracked else ", one-off"
    return (
        f"- {case_set.label}: template {case_set.template_id}, {len(case_set.test_case_ids)} case(s) "
        f"({', '.join(case_set.test_case_ids)}){tracked}"
    )


def render_case_sets(listing: CaseSetListing) -> str:
    """A scope's case sets, every version."""
    return "\n".join([f"case sets ({len(listing.case_sets)})", *(render_case_set(c) for c in listing.case_sets)])


def render_campaigns(listing: CampaignListing) -> str:
    """A scope's campaigns."""
    return "\n".join([f"campaigns ({len(listing.campaigns)})", *(render_campaign(c) for c in listing.campaigns)])


def render_analysis_line(analysis: AnalysisLine) -> str:
    """One stored analysis on one line."""
    archived = ", archived" if analysis.archived else ""
    return f"- {analysis.id}: {analysis.headline} ({analysis.generator_model}, {analysis.generated_at}{archived})"


def render_analyses(listing: AnalysisListing) -> str:
    """A campaign's analyses."""
    lines = [f"analyses of campaign {listing.campaign_id} ({len(listing.analyses)})"]
    lines += [render_analysis_line(a) for a in listing.analyses]
    return "\n".join(lines)


def render_undescribable_arms(listing: UndescribableArmsListing) -> str:
    """The scope's analyses holding an arm whose levels this build cannot describe."""
    lines = [
        f"analyses in scope {listing.scope_id} holding an arm this build cannot describe: "
        f"{len(listing.analyses)} of {listing.analyses_read}"
    ]
    for line in listing.analyses:
        why = "; ".join(line.reasons) or "no reason recorded"
        archived = ", archived" if line.archived else ""
        lines.append(
            f"- {line.analysis_id} (campaign {line.campaign_id}{archived}): "
            f"{line.undescribable_arms} of {line.arms} arm(s) — {why}"
        )
    return "\n".join(lines)


def render_analysis_estimate(estimate: AnalysisGenerationEstimate) -> str:
    """A generation's price before it starts: its first call's ceiling against the cap, and whether it would start."""
    ceiling = (
        f"${estimate.first_call_ceiling_usd:.4f}" if estimate.first_call_ceiling_usd is not None else "unpriceable"
    )
    cap = f"${estimate.cap_usd:.2f}" if estimate.cap_usd is not None else "none enforced"
    verdict = "would start" if estimate.would_start else f"would be refused: {estimate.refusal}"
    return (
        f"estimate: analysis of campaign {estimate.campaign_id} on {estimate.generator_model} — first call priced at "
        f"up to {ceiling}, out-of-run cap {cap}; {verdict}. A generation makes at most {estimate.max_calls} call(s): "
        "a refused output buys one repair round-trip, priced against what is left of the cap before it is sent."
    )


def render_bar_proposals(proposals: BarProposals) -> str:
    """Each proposed bar with its seed and any vacuity, then every reading nothing could be proposed on."""
    return bar_proposals_text(proposals)


def render_report(document: ReportDocument) -> str:
    """The report itself, in the form asked for — which already says whether it is an analysis or code-only."""
    return document.body


def render_reporter_case(case: FrozenReporterCase) -> str:
    """A reporter case's receipt: what it froze, its fingerprint, its labels and every limit it recorded."""
    memo = case.recorded_analysis_id or "none (only a generating candidate can run it)"
    lines = [
        f"reporter case {case.test_case_id} of template {case.template_id}: campaign {case.source_campaign_id}, "
        f"recorded memo {memo}",
        f"bundle fingerprint {case.bundle_fingerprint}, assembled {case.bundle_assembled_at}",
    ]
    if case.writer_message_check is not None:
        lines.append(f"writer message: {case.writer_message_check}")
    lines.append(f"labels ({len(case.labels)})")
    lines += [f'- {label.dimension}: {label.direction} — "{label.quote}"' for label in case.labels]
    lines.append(f"limits ({len(case.limits)}){'' if case.limits else ': none — the frozen evidence limits nothing'}")
    lines += [f"- {limit}" for limit in case.limits]
    if case.supersedes:
        lines.append(f"supersedes: {', '.join(case.supersedes)}")
    if case.archived:
        lines.append(f"retired{': ' + case.archived_reason if case.archived_reason else ''} — no launch runs it")
    return "\n".join(lines)


def _reporter_case_state(entry: ReporterCaseEntry) -> str:
    """Whether a launch runs the case, and if not, every reason it does not."""
    if entry.live:
        return "live"
    reasons = []
    if entry.superseded_by:
        reasons.append(f"superseded by {', '.join(entry.superseded_by)}")
    if entry.case.archived:
        reasons.append(f"retired{': ' + entry.case.archived_reason if entry.case.archived_reason else ''}")
    return "; ".join(reasons)


def render_reporter_cases(listing: ReporterCaseListing) -> str:
    """A reporter template's bank: each case and its state, then what cannot be read and what a launch refuses."""
    scope = "including retired" if listing.include_archived else "retired cases left out"
    lines = [f"reporter cases of template {listing.template_id} ({len(listing.cases)}, {scope})"]
    for entry in listing.cases:
        case = entry.case
        lines.append(
            f"- {case.test_case_id}: {_reporter_case_state(entry)} — campaign {case.source_campaign_id}, recorded memo "
            f"{case.recorded_analysis_id or 'none'}; fingerprint {case.bundle_fingerprint}; {len(case.labels)} "
            f"label(s), {len(case.limits)} limit(s)"
        )
    if listing.unreadable:
        lines.append(
            f"unreadable ({len(listing.unreadable)}) — which case is live cannot be decided until each is read with "
            "the build that wrote it:"
        )
        lines += [f"- {unread.test_case_id}: {unread.reason}" for unread in listing.unreadable]
    if listing.ambiguous:
        lines.append(
            f"pairs with more than one live case ({len(listing.ambiguous)}) — every launch of this template refuses "
            "until one reporter_case_freeze names all of a pair's live cases in supersedes:"
        )
        lines += [
            f"- campaign {pair.campaign_id}, recorded memo {pair.recorded_analysis_id or 'none'}: "
            f"{', '.join(pair.live_case_ids)}"
            for pair in listing.ambiguous
        ]
    return "\n".join(lines)


def render_pivot(table: PivotTable) -> str:
    """A pivot: each cell with its denominators, and every caveat the table carries."""
    return pivot_text(table)


def render_runs_compared(compared: RunsCompared) -> str:
    """Two runs compared: the arms, each reading with its delta and test, and every disclosure."""
    return runs_compared_text(compared)


def render_frontier(result: FrontierResult) -> str:
    """Each subject's variants on quality, cost and latency, and the cheapest that clears the bar."""
    return frontier_text(result)


def render_history(result: HistoryResult) -> str:
    """Each contestant's series, with each step's regression verdict and the test behind it."""
    return history_text(result)


def render_out_of_run_spend(report: OutOfRunSpendReport) -> str:
    """The scope's out-of-run spend: totals overall, per purpose and per launch, then each call."""
    return out_of_run_spend_text(report)


def render_estimate(estimate: LaunchEstimate) -> str:
    """A launch's price, arm by arm as the launch would judge it, and the total."""
    return estimate_text(estimate)


def render_export(export: ScoreExport) -> str:
    """The export, after a line saying how many rows it holds and what it left out."""
    return export_text(export)


def render_result_rated(rated: ResultRated) -> str:
    """What an agent's rating wrote, and that it is an agent's."""
    return (
        f"rated {rated.rubric_dim} of result {rated.result_id} at {rated.score} as {rated.rater} "
        f"({rated.rater_kind}; kept beside people's ratings, never read as one)"
    )


def _bounds(interval: tuple[float, float] | None) -> str:
    """An interval as ``[low, high]``, or a word saying there is none."""
    return "no interval" if interval is None else f"[{interval[0]:.3g}, {interval[1]:.3g}]"


def render_second_judge(read: SecondJudgeRead) -> str:
    """A second judge's pass: what it asked and spent apart from the candidate, then agreement and drift per dim."""
    report = read.report
    cost = _usd(report.cost_usd) if report.cost_usd is not None else "unpriced"
    lines = [
        f"second judge {report.judge.model} on run {report.run_id} (pass {report.pass_id}): asked about "
        f"{len(report.judged)} of {report.eligible} judgeable result(s) (fraction {report.sample_fraction:g}, seed "
        f"{report.sample_seed}); {report.scores_paired} score(s) paired, {report.scores_unanswered} unanswered; "
        f"{report.calls_made} call(s), {cost} — measurement cost, ledgered under second_judge, never the candidate's",
    ]
    if report.stopped:
        lines.append(f"stopped: {report.stopped}")
    lines += [f"skipped {skip.result_id}: {skip.reason}" for skip in report.skipped]
    lines += [f"unwritten {result_id}: paid for, record not stored" for result_id in report.unwritten]
    lines.append("agreement between the judges")
    for row in read.agreement.dimensions:
        kappa = (
            f"kappa ({row.kappa_weighting}) {row.kappa:.3g}, bounds {_bounds(row.agreement_interval)}"
            if row.kappa is not None
            else f"kappa {row.kappa_undefined}"
        )
        lines.append(f"- {row.rubric_dim}: n={row.n}, exact agreement {row.exact_agreement:.0%}, {kappa}")
    lines.append(read.drift.disclosure)
    for drift in read.drift.dimensions:
        delta = "n/a" if drift.delta is None else f"{drift.delta:+.3g}"
        lines.append(
            f"- {drift.rubric_dim}: {drift.verdict}, movement {delta} over {drift.n_cases} case(s), "
            f"{_bounds(drift.interval)}"
        )
    return "\n".join(lines)


def render_second_judge_estimate(estimate: SecondJudgeEstimate) -> str:
    """A second judge's price before it starts, against the cap, and whether it would start."""
    ceiling = _usd(estimate.ceiling_usd) if estimate.ceiling_usd is not None else "unpriceable"
    cap = _usd(estimate.cap_usd) if estimate.cap_usd is not None else "none enforced"
    verdict = "would start" if estimate.would_start else f"would be refused: {estimate.refusal}"
    return (
        f"estimate: second judge {estimate.judge.model} on run {estimate.run_id} — {len(estimate.sampled)} of "
        f"{estimate.eligible} judgeable result(s), {estimate.dims} dim(s), at most {estimate.max_calls} call(s) priced "
        f"at up to {ceiling}; out-of-run cap {cap}; {verdict}."
    )


def render_run_deleted(deleted: RunDeleted) -> str:
    """What deleting a run removed."""
    detached = ", ".join(deleted.campaigns_detached) or "none"
    return f"deleted run {deleted.run_id}, its {deleted.results_deleted} result(s); detached from campaigns: {detached}"


def _insight_line(insight: InsightLine) -> str:
    standing = "" if insight.standing == "live" else f", {insight.standing}"
    return (
        f"- {insight.id}: {insight.statement} ({insight.confidence}{standing}; subject {insight.subject_id}, "
        f"campaign {insight.source_campaign_id or '-'}, analysis {insight.source_analysis_id or '-'})"
    )


def render_insights(listing: InsightListing) -> str:
    """The scope's insights, saying what narrowed the read — an empty filtered read names what it searched."""
    narrowed = f" ({listing.filters})" if listing.filters else ""
    if not listing.insights:
        return f"no insights{narrowed}"
    lines = [f"insights{narrowed} ({len(listing.insights)})"]
    lines += [_insight_line(insight) for insight in listing.insights]
    return "\n".join(lines)


def render_insight(detail: InsightDetail) -> str:
    """One insight in full."""
    insight = detail.insight
    runs = ", ".join(insight.evidence_run_ids) or "none"
    results = ", ".join(insight.evidence_result_ids) or "none"
    models = ", ".join(f"{role}={model}" for role, model in sorted(insight.model_versions.items())) or "none"
    lines = [
        f"insight {insight.id} ({detail.standing}, {insight.confidence})",
        f"statement: {insight.statement}",
        f"subject: {insight.subject_id} ({insight.subject_kind or 'no kind'})",
        f"scope: {insight.scope or '-'}",
        f"evidence runs: {runs}",
        f"evidence results: {results}",
        f"model versions: {models}",
        f"observed at: {insight.observed_at}",
        f"minted by: analysis {insight.source_analysis_id or '-'} of campaign {insight.source_campaign_id or '-'}",
        f"retired when: {insight.invalidation_trigger or '-'}",
    ]
    return "\n".join(lines)


def render_insight_deleted(deleted: InsightDeleted) -> str:
    """What deleting an insight removed."""
    campaign = deleted.source_campaign_id or "no campaign"
    return f"deleted insight {deleted.insight_id} (minted from {campaign}); no later analysis reads it"


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
    "render_analysis_estimate",
    "render_analysis_line",
    "render_result",
    "render_result_rated",
    "render_results",
    "render_campaign",
    "render_campaigns",
    "render_bar_proposals",
    "render_estimate",
    "render_export",
    "render_help_index",
    "render_help_page",
    "render_frontier",
    "render_history",
    "render_insight",
    "render_insight_deleted",
    "render_insights",
    "render_job_status",
    "render_jobs_started",
    "render_pivot",
    "render_report",
    "render_reporter_case",
    "render_reporter_cases",
    "render_run_deleted",
    "render_run_line",
    "render_runs",
    "render_summary",
    "render_templates",
]
