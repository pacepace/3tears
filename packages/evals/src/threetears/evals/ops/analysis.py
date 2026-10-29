"""The analysis side's operations: campaigns, a generation started as a job, and the report read three ways.

A generation is the analysis side's long work, and it follows the same job contract a launch does
(:mod:`threetears.evals.ops.jobs`): :func:`analysis_generate` checks only what its caller can fix cheaply (the
campaign, the named writer, the report budget, the slot), records a running attempt and starts a background task,
returning its job at once. The task prepares — builds the host's ``analyst`` client, holds it to the allowed writers,
checks it is a turn model whose turns can be priced — and only then writes the report (the report writer, on the
host's model and budget); a refusal it meets while preparing ends the attempt ``failed`` with nothing frozen or spent,
and ``job_poll`` reads it. The job's record is the :class:`~threetears.evals.kernel.campaign.EvalAnalysisAttempt` the
generation writes however it ends.

One generation per campaign runs at a time: a second is refused while the first is live, BEFORE it
prepares, because preparing builds a client — work a refused request need not pay for.
"""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Final, Literal, NamedTuple, Protocol, cast

from pydantic import Field, ValidationError

from threetears.evals.analysis.bundle.insights import InsightStanding, insight_standing
from threetears.evals.analysis.bar_proposals import propose_bars
from threetears.evals.analysis.campaigns import (
    add_runs_to_campaign,
    create_campaign,
    list_campaigns,
    remove_runs_from_campaign,
    update_campaign,
)
from threetears.evals.kernel.numbers import format_number
from threetears.evals.analysis.planned_analysis import report_plan_of
from threetears.evals.analysis.report import ReportBasis, TrialDetail, analysis_verdict
from threetears.evals.analysis.report.build import analysis_campaign_bundle
from threetears.evals.analysis.report.serialize import ReportFormat, serialize_report
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.doc_analysis import doc_analysis
from threetears.evals.analysis.generator import GenerationTally
from threetears.evals.analysis.service import (
    AnalysisGenerationEstimate,
    WrittenAttempt,
    campaign_report,
    describe_insight_id_filters,
    get_analysis,
    get_report_plan,
    list_analyses,
    list_insights,
    record_campaign_call,
    record_written_generation,
    report_doc_report,
    running_analysis_attempt,
)
from threetears.evals.analysis.campaign_title import TITLE_ARMS, TITLE_ASPECTS, campaign_title, declared_title
from threetears.evals.analysis.report.campaign.memo_blocks import NO_LONGER_READS
from threetears.evals.kernel.report_doc import ReportDoc
from threetears.evals.schema.base import EvalBaseModel, VerbatimText
from threetears.evals.kernel.declaration import (
    ArmRole,
    CampaignArm,
    CampaignCall,
    Margin,
    OverriddenBar,
)
from threetears.evals.kernel.judge_cases import JUDGE_KIND
from threetears.evals.kernel.campaign import (
    AttemptOutcome,
    ConfidenceTier,
    EvalAnalysis,
    EvalCampaign,
    EvalInsight,
    EvalReportPlan,
)
from threetears.evals.kernel.errors import (
    ConflictError,
    EvalServiceError,
    JobFenced,
    NotFoundError,
    ProviderRefusedError,
    RefusedInput,
    StorageError,
    ValidationFailedError,
)
from threetears.evals.kernel.campaign import EvalGenerationSlot, generation_slot_id
from threetears.evals.kernel.job_leases import (
    ANALYSIS_JOB,
    LEASE_WRITE_ATTEMPTS,
    note_job_fault,
    settle_job_record,
    write_held_record,
)
from threetears.evals.kernel.inner_ceilings import writer_turns
from threetears.evals.kernel.job_timeout import EvalJobTimeout, timed_out_detail
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.provider import describe_failure, traceback_is_safe
from threetears.evals.kernel.refusals import job_refusal
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops.analyst import snapshot_input
from threetears.evals.ops.host import AnalysisGeneration, OpsHost
from threetears.evals.ops.jobs import JobHandle, JobsStarted, analysis_job_id, generation_key
from threetears.evals.ops.report_writer.loop import TurnModel, WriterLimits
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally
from threetears.evals.ops.report_writer.write import ReportNotStored, ReportWritten
from threetears.evals.run.curation import delete_analysis, delete_insight, set_analysis_archived, set_campaign_archived
from threetears.evals.schema.completion import BoundCompletionClient
from threetears.evals.schema.out_of_run_spend import OutOfRunSpend
from threetears.evals.schema.models import DECLARED_NAME_CHARS, utc_now_iso
from threetears.evals.schema.logs import contain, get_logger
from threetears.evals.kernel.remedies import FIND_CAMPAIGN, FIND_INSIGHT, FIND_RUN, WRITER_REMEDY


log = get_logger(__name__)


#: The most characters a campaign's name may hold when it is created or renamed: a qualifier beside the title code
#: builds (:func:`campaign_title`), as short as any declared short name. A name stored longer before the cap still
#: reads and lists; only a write is held to it.
CAMPAIGN_NAME_CHARS = DECLARED_NAME_CHARS

#: What a name over the cap is told to do instead.
CAMPAIGN_NAME_REMEDY = (
    f"name the campaign in at most {CAMPAIGN_NAME_CHARS} characters, and put the detail (the arms, the round, the case "
    "count, sweep or declared) in question or description"
)

#: The most characters a listing's short question holds, its ellipsis included.
QUESTION_SHORT_CHARS = 100

#: The longest an outcome with its warnings is spelled out; past it the warnings are counted (`2 warnings`).
OUTCOME_SHORT_CHARS = 48

#: Where a campaign's report stands, as a listing shows it: ``none``, no live analysis; ``written``, one this build reads
#: over the runs the campaign holds now; ``needs_regeneration``, one it cannot read, a report plan (rendered code-only),
#: or one written over other runs than the campaign holds now.
ReportState = Literal["none", "written", "needs_regeneration"]


class CampaignLineArm(EvalBaseModel):
    """One declared arm of a campaign, as a listing shows it."""

    label: str
    short_name: str | None = Field(description="The name the arm declares for where space is tight; None when none.")
    role: ArmRole


class CampaignLineCall(EvalBaseModel):
    """The call recorded over a campaign's readings, as a listing shows it."""

    arm: str = Field(description="The label of the arm the call names.")
    reason: str = Field(description="Why, in the words it was recorded with.")


class CampaignLineReport(EvalBaseModel):
    """A campaign's reported analysis (its newest live one), as a listing shows it."""

    analysis_id: str
    headline: str = Field(
        description="Code's verdict line over the evidence it was written from (AnalysisLine.verdict)."
    )
    why_regenerate: str | None = Field(
        description=(
            "Why the report needs regeneration (report_state needs_regeneration): this build cannot read it, it is a "
            "report plan, or the campaign holds other runs than it was written over. None when it is written."
        )
    )
    generated_at: str
    generator_model: str
    cost_usd: float | None = Field(
        description=(
            "What generating it cost in US dollars, summed over its attempt's calls; None when any call was unpriced "
            "(unpriced_calls), never 0, or when no attempt records it."
        )
    )
    unpriced_calls: int = Field(ge=0, description="Its attempt's calls whose cost was never observed; 0 with none.")


class CampaignLine(EvalBaseModel):
    """One campaign, as a listing shows it."""

    id: str
    title: str = Field(
        default="",
        description=(
            "What the campaign compares, built by code on every read and never stored: `<arms> · <measure> (<name>)`. "
            "The arms are its declared arms' short names (else labels), control first, joined ` vs `, or, past "
            f"{TITLE_ARMS}, `<control> vs N candidates`; a campaign declaring none names the distinct candidate models "
            "of its runs, and a judge campaign counts its judges. The measure is its aspects' printed names, or past "
            f"{TITLE_ASPECTS} `N aspects`. Nothing from a call, a recommendation or a verdict enters it."
        ),
    )
    name: str
    subject_id: str
    behavior: str
    run_count: int
    archived: bool
    exploratory: bool = Field(
        description="Whether no aspect of the campaign declares a design — derived on every read, never stored"
    )
    aspects: list[str] = Field(description="The labels of the campaign's aspects, in order; at least one.")
    created_at: str = Field(default="", description="When the campaign was created (ISO-8601).")
    question: str = Field(default="", description="The question it asks of its arms, as declared; empty asks none.")
    question_short: str = Field(
        default="",
        description=(
            f"The question cut to at most {QUESTION_SHORT_CHARS} characters at a word boundary, ending in an ellipsis "
            "where it was cut; empty asks none."
        ),
    )
    arms: list[CampaignLineArm] = Field(
        default_factory=list,
        description=(
            "Its declared arms, control first. A campaign that declares no candidate reads each other variant among "
            "its runs as one, which only its report lists."
        ),
    )
    call: CampaignLineCall | None = Field(default=None, description="The call recorded over it; None until one is.")
    verdict_short: str | None = Field(
        default=None,
        description=(
            "Where it stands in a few words, written by code and never by a model: from the recorded call, `Keep "
            "<arm>` for the control and `Adopt <arm>` for a candidate; else from its report's recommendation, the "
            "same way; `No call` for a report recommending no arm or more than one; None with neither a call nor a "
            "report. report.headline is the long form."
        ),
    )
    outcome_short: str | None = Field(
        default=None,
        description=(
            "What the evidence says in a few words, read by code off its report's frozen readings and never from the "
            "writer's prose or by cutting the headline; distinct from verdict_short, which is the decision. Arms are "
            "named by their short names (else labels). The answer leads and a warning only qualifies it. A change: "
            "`<arm> improved` or `<N> candidates improved` where a candidate is shown better, else `No candidate "
            "improved`, whether the candidates are equal, not separated or worse. A choice: `<arm> ahead` where one "
            "arm is shown ahead of another and behind none, else `Arms differ` where any pair separates, else `Not "
            "separated`. A measurement: `Measured`. `No readings yet` where the readings hold no row of their kind. "
            "Where the arms the answer names (the improving arms, the leader, the measured arm) broke a floor or "
            "missed a bar it adds `, broke a floor`, `, missed a bar` or `, broke a floor and missed a bar` (floor "
            "first; `Measured, <arm> missed a bar` names the arm; several arms count, `2 candidates improved, 1 broke "
            "a floor`), and past "
            f"{OUTCOME_SHORT_CHARS} characters counts the warnings (`, 2 warnings`). A warning on an arm the answer "
            "does not name changes nothing. None with no reported analysis, or one this build cannot read."
        ),
    )
    report_state: ReportState = Field(default="none", description="Where its report stands (`report`).")
    report: CampaignLineReport | None = Field(
        default=None, description="Its reported analysis; None exactly when report_state is `none`."
    )
    feeds: list[str] = Field(
        default_factory=list,
        description=(
            "For an exploratory campaign, the ids of the declared campaigns of its scope holding any of its runs (the "
            "ones that adopted its aspects), oldest first; empty for a declared campaign or one no other holds."
        ),
    )


class CampaignListing(EvalBaseModel):
    """A scope's campaigns, newest first."""

    campaigns: list[CampaignLine]


#: How long a start waits for an in-process generation whose attempt has already ended to finish unwinding (releasing
#: its client and its slot) before refusing it as still running. Unwinding is a few store writes, so this is generous.
ENDED_TASK_UNWIND_S = 30.0

#: What a campaign's question is, said once for the operations and the actions that take it.
QUESTION_DESCRIPTION = (
    "The one question the campaign asks of its arms, in the operator's words: what would make a candidate the one to "
    "adopt. A decision rule is never written here; its clauses are floors. Empty asks none."
)

#: What a campaign's arms are, said once.
ARMS_DESCRIPTION = (
    "The arms, each {label, role: control|candidate (default candidate), from_run_ids: [run id, ...], short_name?}: a "
    "contestant stack, read from the first of its runs, each a run one of the campaign's aspects holds, all carrying "
    "one variant key. short_name is what a page prints for the arm where space is tight (chart labels, table cells), "
    f"at most {DECLARED_NAME_CHARS} characters and no other arm's label or short name; omitted, the page computes one "
    "from what differs from the control. One control at most, read first. Omitted, an adopted campaign's arms are "
    "carried and every other variant among the runs is read as a candidate."
)

#: What a campaign's aspects are, said once.
ASPECTS_DESCRIPTION = (
    "Where the question is measured, each {label?, short_name?, template_id?, margin?, design?, run_ids?}. label "
    "omitted names the aspect after the campaign's behavior, each / written - (tools/search is tools-search, as "
    "aspect_label_for writes it); one aspect at most omits it. short_name is what a page prints for the aspect in "
    f"table cells and sentences, at most {DECLARED_NAME_CHARS} characters and no other aspect's label or short name; "
    "omitted, the label is printed whole. design is what the aspect sets out to learn: axes (at least one: {axis_id, values: [{content, display}], rationale?}, each axis_id a lever "
    "or open-family member this host declares), held_fixed ({stimulus: controlled|uncontrolled, stimulus_reason "
    "(required when uncontrolled), apparatus: commissioned|witnessed}), and optionally merit_axes, bars "
    "([{measure_id, threshold, direction}], no looser than the registered ones), guardrail_margins, merit_priority and "
    "intended_repetitions. margin, in points of a 0-100 pass rate, is how far a candidate may fall below the control "
    "there before it is shown worse, which a declared aspect states when the campaign asks a question and names a "
    "control, and a campaign naming no control never states (it holds its arms to bars). template_id "
    "omitted is the one template every "
    "run of the aspect ran. Labels are unique, and a run belongs to one aspect."
)

#: What a campaign's floors are, said once.
FLOORS_DESCRIPTION = (
    "Lines a candidate is warned for crossing, each {aspect, clause}: clause {kind: rate_drop, max_points, reading?} "
    "(no pass rate, or the named one, may drop by more than max_points) or {kind: cost_lower} (it must cost less per "
    "reply). A floor only warns, on an aspect that declares a design, of a campaign that names a control: without "
    "one a floor is read against nothing, and a bar is declared instead. Each aspect, kind and reading once."
)

#: What a margin given at adoption is, said once.
MARGINS_DESCRIPTION = (
    "An adopted aspect's label to its margin, in points of a 0-100 pass rate, more than 0 and less than 100: a margin "
    "declared at adoption, carrying the time it was added."
)

#: What adopting another campaign's aspects means, said once.
ASPECTS_FROM_DESCRIPTION = (
    "Campaigns of this scope whose aspects this one adopts whole — template, runs, design, margin, and when its design "
    "was declared and its question asked — with their floors and arms: each a campaign id, or {campaign_id, label} to "
    "adopt a campaign's one aspect under another label. An adopted aspect keeps its readings' standing; what this "
    "campaign adds to it (a question, a floor, a margin, an arm) carries the time it was added. An adopted aspect keeps "
    "its short_name, and an adopted arm its own. Two aspects under one label or printed alike, or one run in two of "
    "them, are refused."
)

#: What naming a campaign's aspects is, said once.
SHORT_NAMES_DESCRIPTION = (
    "An aspect's label to its short name, what a page prints for it in table cells and sentences (at most "
    f"{DECLARED_NAME_CHARS} characters, no other aspect's label or short name), or to null to print its label whole. An aspect "
    "not named keeps its own. Editable once runs have joined, since it changes nothing any run measures; refused "
    "beside aspects, whose own short_name says it."
)

#: What naming a campaign's arms is, said once.
ARM_SHORT_NAMES_DESCRIPTION = (
    "A named arm's label to its short name, what a page prints for it where space is tight (at most "
    f"{DECLARED_NAME_CHARS} characters, no other arm's label or short name), or to null to let the page compute one. "
    "An arm not named keeps its own. Editable once runs have joined, since it changes nothing any run measures."
)

#: What saying a campaign is exploratory means, said once for the operations and the actions that take it.
EXPLORATORY_DESCRIPTION = (
    "True says no aspect of the campaign declares a design: someone learning to run evals, or testing an intuition "
    "with no settled question. Its report and analysis say once that its readings confirm nothing, and each aspect's "
    "design is inferred from its runs; it names no question, arm, floor or margin. Required when no aspect declares a "
    "design, and refused beside one. An assertion checked and never stored: a campaign is exploratory exactly while "
    "no aspect declares a design."
)

#: What the control run is, said once for the operation and the action.
CONTROL_RUN_DESCRIPTION = (
    "One of the campaign's runs whose variant becomes the control arm, the stack every other is read against: its "
    "variant key is resolved from the run, so nobody types it. Needs a declared campaign. Null clears the control; "
    "omitted leaves it as it is."
)

#: What an aspect named beside a membership change is, said once.
MEMBERSHIP_ASPECT_DESCRIPTION = (
    "The label of the aspect the runs join or leave. Omitted, each run joins the one aspect gathered on the template "
    "it ran (a campaign of one aspect takes every run), and leaves the aspect holding it."
)


class CampaignDefinition(EvalBaseModel):
    """What creating a campaign names: what it is called, its subject and behavior, its question, arms, aspects and
    floors, and the campaigns whose aspects it adopts.

    It declares a design on some aspect or says ``exploratory``, exactly one: a campaign none of whose aspects declares
    one is exploratory, which its report and analysis say once at the top, so the caller chooses that rather than
    falling into it.
    """

    name: str
    subject_id: str
    behavior: str
    description: str = ""
    question: str = Field(default="", description=QUESTION_DESCRIPTION)
    arms: list[dict[str, Any]] = Field(default_factory=list, description=ARMS_DESCRIPTION)
    aspects: list[dict[str, Any]] = Field(default_factory=list, description=ASPECTS_DESCRIPTION)
    floors: list[dict[str, Any]] = Field(default_factory=list, description=FLOORS_DESCRIPTION)
    margins: dict[str, Margin] = Field(default_factory=dict, description=MARGINS_DESCRIPTION)
    aspects_from: list[str | dict[str, str]] = Field(default_factory=list, description=ASPECTS_FROM_DESCRIPTION)
    exploratory: bool = Field(default=False, description=EXPLORATORY_DESCRIPTION)


class CampaignChanges(EvalBaseModel):
    """What an update of a campaign may change: its name, its description, its declaration and its control — each
    only when given.

    Membership is not here: runs join and leave through :func:`campaign_runs_add` and :func:`campaign_runs_remove`,
    whose rules a generic update would skip. An omitted field is left as stored; ``aspects`` replaces the aspects
    whole, and ``control_from_run_id`` given as null clears the control — told apart from omitted by the fields the
    caller set.
    """

    name: str | None = Field(default=None, min_length=1, description="The campaign's new name.")
    description: str | None = Field(default=None, description="The campaign's new description.")
    question: str | None = Field(default=None, description=QUESTION_DESCRIPTION)
    aspects: list[dict[str, Any]] | None = Field(default=None, description=ASPECTS_DESCRIPTION)
    floors: list[dict[str, Any]] | None = Field(default=None, description=FLOORS_DESCRIPTION)
    short_names: dict[str, str | None] | None = Field(default=None, description=SHORT_NAMES_DESCRIPTION)
    arm_short_names: dict[str, str | None] | None = Field(default=None, description=ARM_SHORT_NAMES_DESCRIPTION)
    exploratory: bool = Field(default=False, description=EXPLORATORY_DESCRIPTION)
    control_from_run_id: str | None = Field(default=None, min_length=1, description=CONTROL_RUN_DESCRIPTION)


#: What wrote a stored analysis: ``memo``, the memo writer (every analysis stored before the report planner); ``plan``,
#: the report planner; or ``doc``, the report writer (the analyst and the report editor), which writes every analysis
#: ``analysis_generate`` generates now.
AnalysisBasis = Literal["memo", "plan", "doc"]


class AnalysisLine(EvalBaseModel):
    """One stored analysis, as a listing shows it."""

    id: str
    campaign_id: str
    title: str = Field(
        description=(
            "What the analysis compares, in a few words, as the report's first line heads it: `<arms> · <measure> "
            "(<campaign name>)`, built by code from the campaign bundle frozen with the analysis, never stored and "
            "never carrying a call, a recommendation or a verdict."
        )
    )
    basis: AnalysisBasis = Field(
        description=(
            "What wrote the analysis: `memo` (the memo writer; findings count the memo's findings), `plan` (the "
            "report planner, whose analyses report_read renders code-only; findings count the plan's evidence "
            "sections) or `doc` (the report writer; findings count the document's claims, report_doc_id names it, "
            "and report_read carries it)."
        )
    )
    report_plan_id: str | None = Field(
        description="The id of the report plan a `plan` analysis was written as; None for a memo or a document."
    )
    report_doc_id: str | None = Field(
        default=None, description="The id of the report document a `doc` analysis links; None for a memo or a plan."
    )
    verdict: str = Field(
        description=(
            "The report's verdict line over the evidence the analysis was written from, as code writes it — never a "
            "headline the memo wrote. It leads, as the report does, with every bar an arm misses and then every floor "
            "a candidate breaks, each in the report's own words; with neither, it is the verdict alone."
        )
    )
    findings: int = Field(
        ge=0,
        description=(
            "How many findings the memo holds, how many evidence sections the plan holds, or how many claims the "
            "report document holds."
        ),
    )
    recommendations: int = Field(
        ge=0,
        le=1,
        description=(
            "How many recommendations it makes: one, or none for a measurement, a choice with no clear winner, a "
            "plan's trade-off or no call, or a document recommending no arm or more than one."
        ),
    )
    generator_model: str
    generated_at: str
    archived: bool
    needs_regeneration: str | None = Field(
        default=None,
        description=(
            "Why this build cannot read the analysis, and what to do. Every report plan, and a report document that "
            "no longer reads (its shape moved, or it was deleted apart from it), is rendered code-only by report_read: "
            "its line is the one regenerate line, then why in parentheses where there is more to say, and a plan or "
            "document it cannot read counts no findings. A memo written over a bundle of another shape says why, and "
            "its report is refused until it is generated again (analysis_generate). None when it reads."
        ),
    )


class AnalysisListing(EvalBaseModel):
    """A campaign's stored analyses."""

    campaign_id: str
    analyses: list[AnalysisLine]


class ReportDocument(EvalBaseModel):
    """A campaign's report, serialized in one form."""

    campaign_id: str
    basis: ReportBasis = Field(
        description=(
            "`analysis` when the report renders a written report (the campaign's analysis, or a report document read "
            "by its id); `code_only` when it renders the campaign's evidence alone: it has no analysis, or "
            "needs_regeneration says why the one it has is not rendered."
        )
    )
    analysis_id: str | None = Field(
        description=(
            "The campaign's analysis the report renders, or stands in for where needs_regeneration is set; None when "
            "it has none, and for a report document read by its id."
        )
    )
    analysis_basis: AnalysisBasis | None = Field(
        default=None,
        description=(
            "What wrote that analysis: `memo`, `plan` (the report planner, whose analyses this version renders "
            "code-only) or `doc` (the report writer, whose stored document is `doc`); None with no analysis."
        ),
    )
    doc: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The stored report document the report renders, as `report-doc/1` JSON (its prose names figures by key); "
            "None for any other report."
        ),
    )
    needs_regeneration: str | None = Field(
        default=None,
        description=(
            "Set where the report stands in, code-only, for an analysis or report document this version no longer "
            "reads (every report-plan analysis; a document that no longer validates, or whose snapshot is gone): the "
            "one regenerate line, which the page's first screen prints too, then why in parentheses. The JSON body "
            "carries it too. None otherwise."
        ),
    )
    format: ReportFormat
    body: VerbatimText = Field(
        description="The report in that form, exactly as its serializer wrote it: Markdown, canonical JSON, or "
        "HTML (one page whose one script is the kit's, or, static, with no script and no link). A report "
        "document's Markdown and HTML are the page it draws; its JSON is the campaign's, with `report_doc` the "
        "stored document (and, asked for, `report_doc.sessions`, its transcripts)."
    )


class AnalysisDeleted(EvalBaseModel):
    """What deleting an analysis removed: the analysis, never the insights it minted."""

    analysis_id: str
    campaign_id: str


#: Where an insight stands, read from the analysis that minted it at every read and never stamped on the
#: insight (:func:`~threetears.evals.analysis.bundle.insights.insight_standing`): ``live`` — fed to later
#: generations as prior context; ``retracted`` — its analysis is archived, so no generation reads it;
#: ``orphaned`` — its analysis was deleted, so it is still read but its provenance cannot be followed.
InsightStandingName = Literal["live", "retracted", "orphaned"]


class InsightLine(EvalBaseModel):
    """One insight in the ledger, as a listing shows it."""

    id: str
    subject_id: str
    statement: str
    confidence: ConfidenceTier
    scope: str = Field(description="Where the insight applies, as the analysis wrote it (free text).")
    observed_at: str
    source_campaign_id: str
    source_analysis_id: str
    standing: InsightStandingName = Field(
        description="live (fed to later generations as prior context), retracted (its analysis is archived) or "
        "orphaned (its analysis was deleted; still read, provenance lost)."
    )


class InsightListing(EvalBaseModel):
    """The scope's insights, newest observation first, and the filters they were read under."""

    subject_id: str | None = Field(description="The subject filter, when one narrowed the read.")
    source_campaign_id: str | None = Field(description="The campaign filter, when one narrowed the read.")
    filters: str = Field(description="The id filters that narrowed the read, in words; empty when none did.")
    insights: list[InsightLine]


class InsightDetail(EvalBaseModel):
    """One insight in full — as stored — and where it stands."""

    insight: EvalInsight
    standing: InsightStandingName


class InsightDeleted(EvalBaseModel):
    """What deleting an insight removed: the one insight, never the analysis that minted it."""

    insight_id: str
    source_campaign_id: str


def _standing_of(insight: EvalInsight, standing: InsightStanding) -> InsightStandingName:
    if insight.id in standing.retracted:
        return "retracted"
    if insight.id in standing.orphaned:
        return "orphaned"
    return "live"


def _insight_standing(host: EvalHost, insights: list[EvalInsight], scope_id: str) -> InsightStanding:
    return insight_standing(insights, lambda analysis_id: host.storage.analysis_archived(analysis_id, scope_id))


def _campaign_line(campaign: EvalCampaign) -> CampaignLine:
    return CampaignLine(
        id=campaign.id,
        name=campaign.name,
        subject_id=campaign.subject_id,
        behavior=campaign.behavior,
        run_count=len(campaign.run_ids),
        archived=campaign.archived,
        exploratory=campaign.exploratory,
        aspects=[aspect.label for aspect in campaign.aspects],
        created_at=campaign.created_at,
        question=campaign.question,
        question_short=question_short(campaign.question),
        arms=[CampaignLineArm(label=arm.label, short_name=arm.short_name, role=arm.role) for arm in campaign.arms],
        call=(
            CampaignLineCall(arm=campaign.call.arm, reason=campaign.call.reason) if campaign.call is not None else None
        ),
        verdict_short=_decided(campaign.call.arm, campaign.arms) if campaign.call is not None else None,
    )


def _decided(label: str, arms: Sequence[CampaignArm]) -> str:
    """``Keep <arm>`` when the arm is the control, else ``Adopt <arm>``."""
    role = next((arm.role for arm in arms if arm.label == label), "candidate")
    return f"{'Keep' if role == 'control' else 'Adopt'} {label}"


def _outcome(readings: CampaignReadings, arms: Sequence[CampaignArm]) -> str:
    """What the readings show in a few words (:attr:`CampaignLine.outcome_short`), from their structure alone: the
    answer first, then the warnings of the arms the answer names as a qualifier, never in its place."""
    printed = {arm.variant_key: arm.short_name or arm.label for arm in arms}
    labels = {s.arm: s.arm_label for s in readings.floors}
    labels |= {s.arm: s.arm_label for s in readings.bars}
    labels |= {s.arm: s.arm_label for s in readings.readings}
    labels |= {s.arm: s.arm_label for s in readings.measures}
    for pair in readings.pairs:
        labels |= {pair.first: pair.first_label, pair.second: pair.second_label}

    def named(key: str) -> str:
        return printed.get(key, labels.get(key, key))

    def qualified(answer: str, subjects: set[str], *, name: bool = False) -> str:
        """``answer`` with the floors its subject arms broke and the bars they miss, floors first."""
        floors = [s for s in readings.broken_floors() if s.arm in subjects]
        bars = [s for s in readings.missed_bars() if s.arm in subjects]
        floor_arms, bar_arms = {s.arm for s in floors}, {s.arm for s in bars}
        if not floors and not bars:
            return answer
        if len(subjects) == 1 or (name and len({*floor_arms, *bar_arms}) == 1):
            who = f"{named(next(iter({*floor_arms, *bar_arms})))} " if name else ""
            phrases = [f"{who}broke a floor"] if floors else []
            phrases += [f"{'' if phrases else who}missed a bar"] if bars else []
        else:
            phrases = [f"{len(floor_arms)} broke a floor"] if floors else []
            phrases += [f"{len(bar_arms)} missed a bar"] if bars else []
        full = f"{answer}, {' and '.join(phrases)}"
        if len(full) <= OUTCOME_SHORT_CHARS:
            return full
        warnings = len(floors) + len(bars)
        return f"{answer}, {warnings} warning{'s' if warnings != 1 else ''}"

    if readings.kind == "measurement":
        if not readings.measures:
            return "No readings yet"
        return qualified(
            "Measured",
            {*(s.arm for s in readings.broken_floors()), *(s.arm for s in readings.missed_bars())},
            name=True,
        )
    if readings.kind == "choice":
        if not readings.pairs:
            return "No readings yet"
        ahead = {pair.ahead for pair in readings.pairs if pair.ahead is not None}
        behind = {
            pair.second if pair.ahead == pair.first else pair.first for pair in readings.pairs if pair.ahead is not None
        }
        if len(leaders := ahead - behind) == 1:
            return qualified(f"{named(next(iter(leaders)))} ahead", leaders)
        return "Arms differ" if ahead else "Not separated"
    if not readings.readings:
        return "No readings yet"
    better = list(dict.fromkeys(row.arm for row in readings.readings if row.reading == "better"))
    if not better:
        return "No candidate improved"
    answer = f"{named(better[0])} improved" if len(better) == 1 else f"{len(better)} candidates improved"
    return qualified(answer, set(better))


def _recommended(analysis: EvalAnalysis, campaign: EvalCampaign) -> str:
    """What a report recommends in a few words: the arm its recommended variant is, or ``No call``."""
    key = analysis.recommended_variant_key
    if key is None:
        return "No call"
    arms: list[CampaignArm] = list(campaign.arms)
    if not any(arm.variant_key == key for arm in arms):
        try:
            arms = list(analysis_campaign_bundle(analysis).arms)
        except RefusedInput:
            # NOSILENT: a bundle this build cannot read names no arm; the line's report says it needs regeneration.
            return "No call"
    arm = next((arm for arm in arms if arm.variant_key == key), None)
    return _decided(arm.label, arms) if arm is not None else "No call"


def _refuse_long_name(name: str) -> None:
    """Refuse a campaign name over :data:`CAMPAIGN_NAME_CHARS`, saying where its detail goes.

    :raises ValidationFailedError: The name is longer than the cap.
    """
    if len(name) > CAMPAIGN_NAME_CHARS:
        raise ValidationFailedError(
            f"a campaign's name is at most {CAMPAIGN_NAME_CHARS} characters; this one is {len(name)}",
            remedy=CAMPAIGN_NAME_REMEDY,
            parameters=("name",),
        )


def question_short(question: str) -> str:
    """A question cut to :data:`QUESTION_SHORT_CHARS` characters at a word boundary, an ellipsis marking the cut.

    :param question: The question, as declared.
    :ptype question: str
    :return: It whole when it fits; else its words up to the limit and ``…``, or its first characters and ``…`` when
        its first word alone is longer.
    :rtype: str
    """
    text = " ".join(question.split())
    if len(text) <= QUESTION_SHORT_CHARS:
        return text
    room = text[: QUESTION_SHORT_CHARS - 1]
    cut = room.rsplit(" ", 1)[0] if " " in room else room
    return cut.rstrip(" ,;:.-") + "…"


def _with_report(host: EvalHost, line: CampaignLine, campaign: EvalCampaign) -> CampaignLine:
    """The line with its reported analysis: what code reads of it, whether it needs regeneration, and its cost."""
    analysis = host.storage.load_reported_analysis(campaign.id, campaign.scope_id)
    if analysis is None:
        return line
    listed = _analysis_line(host, analysis, campaign)
    if not campaign.arms:
        line = line.model_copy(update={"title": _resolved_title(analysis, campaign)})
    why = listed.needs_regeneration
    if why is None and set(analysis.observation_refs) != set(campaign.run_ids):
        why = "the campaign holds other runs than the report was written over; generate it again (analysis_generate)"
    attempt = next(
        (
            attempt
            for attempt in host.storage.list_analysis_attempts_by_campaign(campaign.id, campaign.scope_id)
            if attempt.analysis_id == analysis.id
        ),
        None,
    )
    report = CampaignLineReport(
        analysis_id=analysis.id,
        headline=listed.verdict,
        why_regenerate=why,
        generated_at=listed.generated_at,
        generator_model=listed.generator_model,
        cost_usd=attempt.token_cost if attempt is not None and attempt.unpriced_calls == 0 else None,
        unpriced_calls=attempt.unpriced_calls if attempt is not None else 0,
    )
    try:
        bundle = analysis_campaign_bundle(analysis)
    except RefusedInput:
        # NOSILENT: a bundle this build cannot read has no readings to summarise; why_regenerate already says so.
        outcome = None
    else:
        outcome = _outcome(bundle.readings, bundle.arms)
    return line.model_copy(
        update={
            "report_state": "written" if why is None else "needs_regeneration",
            "report": report,
            "outcome_short": outcome,
            "verdict_short": line.verdict_short or _recommended(analysis, campaign),
        }
    )


def _stored_title(host: EvalHost, campaign: EvalCampaign) -> str:
    """A stored campaign's title (:func:`campaign_title`); its runs are read only for a campaign that declares no arm,
    whose title names their candidate models."""
    runs = [] if campaign.arms else host.storage.load_eval_runs(campaign.run_ids, campaign.scope_id)
    return campaign_title(
        campaign,
        candidate_models=sorted({run.candidate_model for run in runs}),
        variants=len({json.dumps(run.variant_levers, sort_keys=True, default=str) for run in runs}),
        judges=any(run.candidate_kind == JUDGE_KIND for run in runs),
    )


def _listed(host: EvalHost, campaign: EvalCampaign, every: list[EvalCampaign] | None = None) -> CampaignLine:
    """A campaign's whole line: its title, declared fields, report, and what it feeds among ``every`` campaign of its
    scope (read here when not given). Its runs are read only for a campaign that declares no arm, whose title names
    their candidate models."""
    every = every if every is not None else list_campaigns(host.storage, campaign.scope_id, archived=None)
    line = _with_report(host, _campaign_line(campaign), campaign)
    # A campaign declaring no arm is titled by the arms its report resolved, as its page and analysis line are; with
    # no readable report, by its runs' candidate models.
    title = line.title or _stored_title(host, campaign)
    return line.model_copy(update={"feeds": _feeds(campaign, every), "title": title})


def _feeds(campaign: EvalCampaign, campaigns: list[EvalCampaign]) -> list[str]:
    """The declared campaigns holding any of an exploratory campaign's runs, oldest first."""
    if not campaign.exploratory:
        return []
    runs = set(campaign.run_ids)
    held = [
        other for other in campaigns if other.id != campaign.id and not other.exploratory and runs & set(other.run_ids)
    ]
    return [other.id for other in sorted(held, key=lambda other: (other.created_at, other.id))]


def campaigns_list(host: EvalHost, scope_id: str, *, archived: bool | None = None) -> CampaignListing:
    """The scope's campaigns.

    :param host: The host whose store is read.
    :ptype host: EvalHost
    :param scope_id: The scope.
    :ptype scope_id: str
    :param archived: Only archived (``True``) or only active (``False``) campaigns; ``None`` for both.
    :ptype archived: bool | None
    :return: The listing: each campaign with its question, declared arms, recorded call, its report's state, headline
        and cost (one read of its reported analysis and its attempts), and the declared campaigns an exploratory one
        feeds, read among every campaign of the scope, archived or not.
    :rtype: CampaignListing
    """
    every = list_campaigns(host.storage, scope_id, archived=None)
    listed = [c for c in every if archived is None or c.archived == archived]
    return CampaignListing(campaigns=[_listed(host, campaign, every) for campaign in listed])


def campaign_create(host: EvalHost, definition: CampaignDefinition, scope_id: str, *, created_by: str) -> CampaignLine:
    """Create a campaign over runs already in the scope, declared as it is created when the definition says so.

    :param host: The host whose store the campaign is written to.
    :ptype host: EvalHost
    :param definition: What the campaign is.
    :ptype definition: CampaignDefinition
    :param scope_id: The scope it lives in, with its runs.
    :ptype scope_id: str
    :param created_by: Who is creating it, as the calling surface knows them.
    :ptype created_by: str
    :return: The campaign as created.
    :rtype: CampaignLine
    :raises NotFoundError: A campaign it adopts from, or a template an aspect names, is not in the scope.
    :raises ValidationFailedError: The definition declares neither a design nor ``exploratory``, or both; fails
        campaign validation; holds a run outside the scope or in two aspects; declares an aspect without a margin, or
        a design this host cannot honour; or names an arm from runs no aspect holds, or that carry two keys.
    :raises StorageError: The campaign failed to persist.
    """
    _refuse_long_name(definition.name)
    campaign = create_campaign(
        host.storage,
        definition.model_dump(exclude={"aspects_from", "exploratory"}),
        scope_id=scope_id,
        created_by=created_by,
        profile=host.profile,
        exploratory=definition.exploratory,
        aspects_from=definition.aspects_from,
    )
    return _listed(host, campaign)


def campaign_update(
    host: EvalHost, campaign_id: str, changes: CampaignChanges, scope_id: str, *, updated_by: str
) -> CampaignLine:
    """Amend a campaign's name, description, declaration or control — declaring a design on an exploratory one included.

    Only the fields the caller set are written (:attr:`CampaignChanges.model_fields_set`), so an omitted field is left
    as stored, and an omitted ``control_from_run_id`` leaves the control alone while a null one clears it. The
    declaration and its control are written together, so a refused designation leaves the declaration as it was.

    :param host: The host whose store holds the campaign.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param changes: What to change.
    :ptype changes: CampaignChanges
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param updated_by: Who is amending it, as the calling surface knows them; recorded on a declaration this writes.
    :ptype updated_by: str
    :return: The campaign as persisted.
    :rtype: CampaignLine
    :raises NotFoundError: No campaign with that id in the scope.
    :raises ConflictError: The campaign holds a run, and the change writes its declaration or re-points or clears its
        control.
    :raises ValidationFailedError: No field is set; a design this host cannot honour, an aspect with runs or with no
        margin; a withdrawal without ``exploratory``, or ``exploratory`` beside a design; or a control run on an
        exploratory campaign or that the campaign does not hold.
    :raises LeverCoordinateError: The control run recorded no lever map, and the host's variant map disagrees with
        its own registry — whether or not the run has results.
    :raises StorageError: The write failed.
    """
    if changes.name is not None:
        _refuse_long_name(changes.name)
    written = changes.model_fields_set - {"exploratory", "control_from_run_id"}
    campaign = update_campaign(
        host.storage,
        campaign_id,
        scope_id,
        {name: value for name, value in changes.model_dump(include=written).items() if value is not None},
        updated_by=updated_by,
        profile=host.profile,
        exploratory=changes.exploratory,
        control_from_run_id=changes.control_from_run_id,
        clear_control="control_from_run_id" in changes.model_fields_set and changes.control_from_run_id is None,
    )
    return _listed(host, campaign)


def _refuse_no_runs(run_ids: list[str], verb: str) -> None:
    """Refuse a membership change naming no run: it would write the campaign and change nothing."""
    if not run_ids:
        raise ValidationFailedError(f"no run was named to {verb}", remedy=f"name the runs by id ({FIND_RUN})")


def campaign_runs_add(
    host: EvalHost, campaign_id: str, run_ids: list[str], scope_id: str, *, aspect: str | None = None
) -> CampaignLine:
    """Put runs in a campaign's aspect — runs launched after it was created, or a run first measured on its own.

    Each run joins ``aspect``, or, when it is None, the one aspect gathered on the template it ran
    (:func:`~threetears.evals.analysis.add_runs_to_campaign`). A run already in that aspect is not added twice, and a
    run outside the scope, or held by another aspect, is refused, all-or-nothing, as at creation.

    :param host: The host whose store holds the campaign and the runs.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param run_ids: The runs to add.
    :ptype run_ids: list[str]
    :param scope_id: The scope they all live in.
    :ptype scope_id: str
    :param aspect: The aspect they join, by label; None routes each by its template.
    :ptype aspect: str | None
    :return: The campaign as persisted.
    :rtype: CampaignLine
    :raises NotFoundError: No campaign with that id in the scope.
    :raises ValidationFailedError: No run is named; the aspect is not the campaign's; a run does not exist in the
        scope, is held by another aspect, or ran a template no aspect or several aspects are gathered on.
    :raises StorageError: The write failed.
    """
    _refuse_no_runs(run_ids, "add")
    return _listed(host, add_runs_to_campaign(host.storage, campaign_id, scope_id, run_ids, aspect=aspect))


def campaign_runs_remove(
    host: EvalHost, campaign_id: str, run_ids: list[str], scope_id: str, *, aspect: str | None = None
) -> CampaignLine:
    """Take runs out of a campaign — membership only: the runs and their results stay, in any other campaign too.

    A run the campaign (or the named aspect) does not hold is refused, all-or-nothing
    (:func:`~threetears.evals.analysis.remove_runs_from_campaign`). An arm is a variant, not a run, so removing the
    run it was named from leaves it named; where no remaining run carries it, the report says the arm is unobserved.

    :param host: The host whose store holds the campaign.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param run_ids: The runs to take out; every one must be in the campaign.
    :ptype run_ids: list[str]
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param aspect: The aspect holding them, by label; None takes each from wherever it is held.
    :ptype aspect: str | None
    :return: The campaign as persisted.
    :rtype: CampaignLine
    :raises NotFoundError: No campaign with that id in the scope.
    :raises ValidationFailedError: No run is named, or one is not in the campaign (or the aspect); nothing was removed.
    :raises StorageError: The write failed.
    """
    _refuse_no_runs(run_ids, "remove")
    return _listed(host, remove_runs_from_campaign(host.storage, campaign_id, scope_id, run_ids, aspect=aspect))


class CampaignCallLine(EvalBaseModel):
    """The call a person recorded on a campaign, as stored."""

    campaign_id: str
    call: CampaignCall


def campaign_record(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
    *,
    arm: str,
    called_by: str,
    reason: str = "",
    floors_overridden: list[str] | None = None,
    bars_overridden: list[OverriddenBar] | None = None,
    analysis_id: str | None = None,
) -> CampaignCallLine:
    """Record a person's call on a campaign — the arm decided, why, and each broken floor and missed bar it overrides.

    Read against code's readings of the evidence as it stands, never a memo's words
    (:func:`~threetears.evals.analysis.record_campaign_call`): an arm with a broken floor or a missed bar is recorded
    only when the call names each one as overridden, and an override of one the arm did not break or miss is refused.
    A new call replaces the earlier one.

    :param host: The host whose store holds the campaign.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param arm: The decided arm, by the label its page shows or by its variant key.
    :ptype arm: str
    :param called_by: Who made the call, as the calling surface knows them.
    :ptype called_by: str
    :param reason: Why, in the decider's words.
    :ptype reason: str
    :param floors_overridden: The key of each broken floor the call overrides.
    :ptype floors_overridden: list[str] | None
    :param bars_overridden: Each missed bar the call overrides, by aspect and measure.
    :ptype bars_overridden: list[OverriddenBar] | None
    :param analysis_id: The analysis whose memo the call was made over; None for code's readings alone.
    :ptype analysis_id: str | None
    :return: The call as stored.
    :rtype: CampaignCallLine
    :raises NotFoundError: No campaign with that id in the scope, or no such analysis.
    :raises ValidationFailedError: See :func:`~threetears.evals.analysis.record_campaign_call`.
    :raises ConflictError: The campaign changed while its readings were read.
    """
    campaign = record_campaign_call(
        host,
        campaign_id,
        scope_id,
        arm=arm,
        called_by=called_by,
        reason=reason,
        floors_overridden=floors_overridden or (),
        bars_overridden=bars_overridden or (),
        analysis_id=analysis_id,
    )
    assert campaign.call is not None  # just recorded and re-read
    return CampaignCallLine(campaign_id=campaign.id, call=campaign.call)


def campaign_archive(host: EvalHost, campaign_id: str, scope_id: str, *, archived: bool) -> CampaignLine:
    """Archive or restore a campaign — retired from listings, nothing destroyed.

    :param host: The host whose store holds the campaign.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param archived: ``True`` retires it, ``False`` restores it.
    :ptype archived: bool
    :return: The campaign as persisted.
    :rtype: CampaignLine
    :raises NotFoundError: No campaign with that id in the scope.
    :raises StorageError: The write failed.
    """
    return _listed(host, set_campaign_archived(host.storage, campaign_id, scope_id, archived=archived))


def _resolved_title(analysis: EvalAnalysis, campaign: EvalCampaign) -> str:
    """A campaign's title from the arms its report resolved (its frozen bundle's), under the campaign's name and
    aspects as they stand; empty where that bundle is of a shape this build cannot read."""
    try:
        frozen = analysis_campaign_bundle(analysis)
    except RefusedInput:
        # NOSILENT: the line's report says it needs regeneration; the title is then read off the campaign's runs.
        return ""
    return declared_title(
        campaign.name,
        frozen.arms,
        [(aspect.label, aspect.short_name) for aspect in campaign.aspects],
        given={arm.label for arm in frozen.declaration.arms},
    )


def _analysis_title(host: EvalHost, analysis: EvalAnalysis, campaign: EvalCampaign | None) -> str:
    """The title the analysis is listed by: built from the campaign bundle frozen with it, so the line reads what the
    analysis was written over. Where that bundle is of a shape this build cannot read, the stored campaign's title,
    else the campaign's id."""
    try:
        frozen = analysis_campaign_bundle(analysis)
    except RefusedInput:
        # NOSILENT: the line says so in needs_regeneration (analysis_verdict refuses the same bundle); the title is
        # read from the campaign as it stands, or from its id where the campaign is gone, and is no claim of the bundle.
        stored = (
            campaign if campaign is not None else host.storage.load_campaign(analysis.campaign_id, analysis.scope_id)
        )
        return _stored_title(host, stored) if stored is not None else f"Campaign {analysis.campaign_id}"
    return declared_title(
        frozen.name,
        frozen.arms,
        [(bundle.aspect, bundle.short_name) for bundle in frozen.aspects],
        given={arm.label for arm in frozen.declaration.arms},
    )


def _analysis_line(host: EvalHost, analysis: EvalAnalysis, campaign: EvalCampaign | None = None) -> AnalysisLine:
    # One analysis frozen over a bundle this build cannot read, or whose plan or document no longer reads, is listed as
    # needing regeneration, never allowed to refuse the listing of every other one. Every plan, and a document that no
    # longer reads, is one the report read renders code-only: its line leads with the one regenerate line.
    needs_regeneration: str | None = None
    try:
        verdict = analysis_verdict(analysis)
    except RefusedInput as unreadable:
        needs_regeneration = f"{unreadable.fault}; {unreadable.remedy}" if unreadable.remedy else unreadable.fault
        verdict = f"Needs regeneration: {needs_regeneration}"
    plan: EvalReportPlan | None = None
    unread: str | None = None
    try:
        plan = get_report_plan(host.storage, analysis)
    except NotFoundError as missing:
        unread = missing.message
    except RefusedInput as unreadable:
        # NOSILENT: the stored plan no longer validates; the line names why and the regeneration, and the report of
        # this analysis is the code-only page saying the same, so nothing reads it as if it were whole.
        unread = unreadable.fault
    if analysis.document is not None:
        findings = len(analysis.document.findings)
    elif analysis.report_doc_id is not None:
        findings, unread = _doc_claims(host, analysis.report_doc_id, analysis.scope_id)
    else:
        findings = len(report_plan_of(plan).sections) if plan is not None else 0
    if analysis.report_plan_id is not None or unread is not None:
        why = "; ".join(reason for reason in (unread, needs_regeneration) if reason is not None)
        needs_regeneration = f"{NO_LONGER_READS} ({why})" if why else NO_LONGER_READS
    return AnalysisLine(
        id=analysis.id,
        campaign_id=analysis.campaign_id,
        title=_analysis_title(host, analysis, campaign),
        basis=analysis.basis,
        report_plan_id=analysis.report_plan_id,
        report_doc_id=analysis.report_doc_id,
        verdict=verdict,
        needs_regeneration=needs_regeneration,
        findings=findings,
        recommendations=0 if analysis.recommended_variant_key is None else 1,
        generator_model=analysis.generation.generator_model,
        generated_at=analysis.generation.generated_at,
        archived=analysis.archived,
    )


def _doc_claims(host: EvalHost, report_doc_id: str, scope_id: str) -> tuple[int, str | None]:
    """How many claims a ``doc`` analysis's document holds, and why it does not read when it does not. Blocking.

    A document that no longer reads (its shape moved, or it was deleted apart from its analysis) counts no claims and
    says why, never refusing the listing of every other analysis.
    """
    try:
        stored = host.storage.load_report_doc(report_doc_id, scope_id)
    except ValidationError as unread:
        # NOSILENT: the line leads with the one regenerate line and counts no claims, so nothing reads it as whole.
        return (
            0,
            f"its report document {report_doc_id!r} does not read under this build ({unread.error_count()} fault(s))",
        )
    if stored is None:
        return 0, f"its report document {report_doc_id!r} was deleted apart from it"
    return len(ReportDoc.model_validate(stored.doc).claims), None


def analysis_archive(
    host: EvalHost, analysis_id: str, scope_id: str, *, archived: bool, reason: str | None = None
) -> AnalysisLine:
    """Archive or restore a stored analysis — the reversible answer to deleting it.

    Archived, the analysis stays readable and marked, and the insights it minted are retracted from every
    later generation's context for as long as it stays archived; restored, they return
    (:func:`~threetears.evals.run.set_analysis_archived`).

    :param host: The host whose store holds the analysis.
    :ptype host: EvalHost
    :param analysis_id: The analysis.
    :ptype analysis_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param archived: ``True`` retires it, ``False`` restores it.
    :ptype archived: bool
    :param reason: Why it is archived; cleared on restore.
    :ptype reason: str | None
    :return: The analysis as persisted.
    :rtype: AnalysisLine
    :raises NotFoundError: No analysis with that id in the scope.
    :raises StorageError: The write failed.
    """
    return _analysis_line(
        host, set_analysis_archived(host.storage, analysis_id, scope_id, archived=archived, reason=reason)
    )


def analyses_list(host: EvalHost, campaign_id: str, scope_id: str) -> AnalysisListing:
    """A campaign's stored analyses.

    :param host: The host whose store is read.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :return: The listing.
    :rtype: AnalysisListing
    """
    return AnalysisListing(
        campaign_id=campaign_id,
        analyses=[_analysis_line(host, analysis) for analysis in list_analyses(host.storage, campaign_id, scope_id)],
    )


class UndescribableArmsLine(EvalBaseModel):
    """One stored analysis holding at least one arm whose levels this build cannot describe."""

    analysis_id: str
    campaign_id: str
    archived: bool = Field(description="Whether the analysis is archived.")
    undescribable_arms: int = Field(description="Arms in its variant index carrying levels_unavailable.")
    arms: int = Field(description="Arms in its variant index.")
    reasons: list[str] = Field(description="Each distinct reason its arms give, sorted; an arm may give none.")


class UndescribableArmsListing(EvalBaseModel):
    """The scope's stored analyses that hold an arm whose levels this build cannot describe."""

    scope_id: str
    analyses_read: int = Field(description="Every stored analysis in the scope that was read, archived included.")
    analyses: list[UndescribableArmsLine] = Field(
        description="Those holding at least one undescribable arm, most such arms first."
    )


def analyses_undescribable(host: EvalHost, scope_id: str) -> UndescribableArmsListing:
    """The scope's analyses holding an arm whose levels this build cannot describe, with each one's count and reasons.

    Derived from each stored analysis's frozen variant index on every read and never stored: a tally kept beside
    the rows would be a second derivation that drifts. The fleet view of the per-arm disclosure, which matters
    most after an ``IDENTITY_VERSION`` bump, when every arm stamped before it may become undescribable at once.
    Every campaign in the scope is read, archived ones and archived analyses included, since an archived analysis
    can still be read and restored.

    :param host: The host whose store is read.
    :ptype host: EvalHost
    :param scope_id: The scope.
    :ptype scope_id: str
    :return: The listing.
    :rtype: UndescribableArmsListing
    """
    lines: list[UndescribableArmsLine] = []
    read = 0
    for campaign in list_campaigns(host.storage, scope_id):
        for analysis in list_analyses(host.storage, campaign.id, scope_id):
            read += 1
            # `levels_unavailable` set, even to an empty reason, is the admission; None is a describable arm.
            index = list(
                {entry.variant_key: entry for aspect in analysis.aspects for entry in aspect.variant_index}.values()
            )
            reasons = [entry.levels_unavailable for entry in index if entry.levels_unavailable is not None]
            if reasons:
                lines.append(
                    UndescribableArmsLine(
                        analysis_id=analysis.id,
                        campaign_id=analysis.campaign_id,
                        archived=analysis.archived,
                        undescribable_arms=len(reasons),
                        arms=len(index),
                        reasons=sorted({reason for reason in reasons if reason}),
                    )
                )
    lines.sort(key=lambda line: (-line.undescribable_arms, line.campaign_id, line.analysis_id))
    return UndescribableArmsListing(scope_id=scope_id, analyses_read=read, analyses=lines)


def _generation_settings(host: OpsHost) -> AnalysisGeneration:
    """The host's generation settings, or a refusal saying it generates none here."""
    if host.generation is None:
        raise ValidationFailedError(
            "this host does not generate analyses here: it was mounted without generation settings "
            "(OpsHost.generation), so a generation has no report budget or time limit to run under",
            remedy=(
                "read the campaign's code-only report with report_read; a generation needs the host's operator to "
                "mount it with OpsHost.generation"
            ),
        )
    return host.generation


#: Where a host that generates analyses sets the report's budget, as the refusal of a host with none names it.
_REPORT_BUDGET_REMEDY = (
    "the host's operator sets AnalysisGeneration.report_budget (a callable returning the host's ReportBudget for one "
    "report, read from the host's configuration as each generation starts) on the OpsHost its actions are mounted with"
)

#: What a report writer's estimate does not check, as it says so: its answer reads no run.
_EVIDENCE_UNCHECKED: Final = (
    "the campaign's evidence: it is checked when the job freezes it, and a campaign whose runs resolve none, or "
    "cannot be read, ends the job refused before any turn"
)

#: What the host's ``analyst`` client forwards, beside ``model_name``, for the report writer to run on it.
_TURN_MODEL_MEMBERS: Final = ("turn_session", "turn_pricing_refusal")


class ReportWriter(Protocol):
    """What ``analysis_generate``'s job writes the report with: the action layer's writer, bound to its caller.

    :func:`~threetears.evals.actions.report_writing.campaign_report_writer` is the one a host's actions hand it:
    ``write_report`` on the campaign, which it freezes first, every tool call of both sessions run as the caller who
    started the generation, and the stored document naming them. The job calls it with the host and these keywords:
    ``model`` (the ``analyst`` client, as a turn model), ``budget`` (the report's budget), ``campaign_id``, ``limits``
    (the host's ``AnalysisGeneration.limits``, unchanged) and ``tally`` (which the job reads however the writing
    ends); it returns what ``write_report`` returns, and raises what it raises (``ReportNotStored`` when the store
    refused the document).

    Its call is declared opaque on purpose: a host never builds one or reads what it returns (the action hands the
    job its own), so none of the report writer's types is part of the surface ``analysis_generate`` names.
    """

    def __call__(self, host: OpsHost, **arguments: Any) -> Awaitable[Any]:  # Any: the writer's keywords and result
        """Write the campaign's report (the class docstring says with what)."""
        ...


class ReportWriterCall(Protocol):
    """:class:`ReportWriter`'s call, typed: as the job makes it and as the action's writer is checked against it, so a
    keyword renamed on either side is caught by the type checker."""

    async def __call__(
        self,
        host: OpsHost,
        *,
        model: TurnModel,
        budget: ReportBudget,
        campaign_id: str,
        limits: WriterLimits,
        tally: Tally,
    ) -> ReportWritten: ...


def _report_budget_of(generation: AnalysisGeneration) -> Callable[[], ReportBudget]:
    """The host's report budget, or a refusal saying where to set it."""
    if generation.report_budget is None:
        raise ValidationFailedError(
            "this host has set no budget for a report, so a generation has nothing to admit its turns against; "
            "nothing was frozen or sent",
            remedy=_REPORT_BUDGET_REMEDY,
        )
    return generation.report_budget


def _turn_model(client: BoundCompletionClient) -> TurnModel:
    """The host's ``analyst`` client as the turn model the report writer runs on, or a refusal naming what it lacks."""
    if missing := [name for name in _TURN_MODEL_MEMBERS if not callable(getattr(client, name, None))]:
        raise ValidationFailedError(
            f"the host's client for the analyst role ({type(client).__name__}, model {client.model_name!r}) is not a "
            f"turn model: it has no {' and no '.join(missing)}, so the report writer cannot run on it; nothing was "
            "frozen, sent or spent",
            remedy=(
                "return a turn model for the analyst role (providers.Model is one); a wrapper around one forwards its "
                "'turn_session' and 'turn_pricing_refusal' methods, and wraps the session client 'turn_session' returns"
            ),
        )
    return cast("TurnModel", client)


async def _analyst_client(host: EvalHost, model: str | None) -> BoundCompletionClient:
    """The host's ``analyst`` client, its default writer held to the host's allowed writers.

    :raises ValidationFailedError: The host supplies no clients, or its default writer is not one it allows (the client
        released).
    """
    try:
        clients = host.completion_clients("an analysis generation")
    except RefusedInput as refused:
        raise ValidationFailedError(refused.fault, remedy=refused.remedy) from refused
    client = clients("analyst", model)
    if model is None and (ineligible := host.profile.analysis_writer_refusal(client.model_name)) is not None:
        await _release(client)
        raise ValidationFailedError(f"the host's default writer: {ineligible}", remedy=WRITER_REMEDY)
    return client


async def _release(client: BoundCompletionClient) -> None:
    """Release a client the generation built and will not run on, as every role's client is released."""
    async with client:
        pass


async def _pricing_refusal(model: TurnModel) -> str | None:
    """Why the report's turns cannot be bounded on ``model``, as a sentence; ``None`` when every one can be."""
    try:
        unpriced = await model.turn_pricing_refusal()
    except ValueError as unsendable:
        return f"the model sends no turn ({unsendable})"
    return None if unpriced is None else f"its turns cannot be bounded before they are sent: {unpriced}"


def _skill_digest() -> str:
    """The analyst skill's digest: a report generation's provenance. Blocking (it reads the skill's files)."""
    return load_skill().digest


async def analysis_estimate(
    host: OpsHost, campaign_id: str, scope_id: str, *, model: str | None = None
) -> AnalysisGenerationEstimate:
    """What a campaign's generation would be held to before it starts: the report's budget as its bound, making no call.

    The start's own checks, in its call: the host generates analyses here, has set a report budget, the campaign
    exists, a writer the call names is one the host allows, and the host's ``analyst`` client is a turn model whose
    turns can be bounded before they are sent; ``would_start`` is their answer. It reads no run, so the campaign's
    evidence is not checked here but when the job freezes it, and the estimate says so (``unchecked``). The bound is the budget's cap, which
    every turn is admitted against, and ``max_calls`` the most turns the writer can send under the host's
    ``AnalysisGeneration.limits`` (:func:`~threetears.evals.kernel.inner_ceilings.writer_turns`), or the report
    budget's turn cap where that is fewer. No turn is priced here: a turn's price
    depends on the conversation before it.

    :param host: The host: its generation settings and its clients.
    :ptype host: OpsHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param model: The writer model, or ``None`` for the host's default.
    :ptype model: str | None
    :return: The estimate.
    :rtype: AnalysisGenerationEstimate
    :raises ValidationFailedError: The host generates no analyses here, has set no report budget, supplies no clients,
        or the writer model is not one the host allows.
    :raises NotFoundError: No campaign with that id in the scope.
    """
    generation = _generation_settings(host)
    budget = _report_budget_of(generation)()
    eval_host = host.eval_host
    await run_blocking(eval_host.blocking_executor, _refuse_unstartable, eval_host, campaign_id, scope_id, model)
    client = await _analyst_client(eval_host, model)
    try:
        refusal: str | None = None
        try:
            turn_model = _turn_model(client)
            # Priced only under a dollar budget, as the start prices it: a token and turn budget asks for no price.
            refusal = await _pricing_refusal(turn_model) if budget.usd is not None else None
        except ValidationFailedError as refused:
            refusal = str(refused)
        return AnalysisGenerationEstimate(
            campaign_id=campaign_id,
            generator_model=client.model_name,
            first_call_ceiling_usd=None,
            max_calls=min(writer_turns(generation.limits), budget.turns or writer_turns(generation.limits)),
            cap_usd=budget.usd,
            would_start=refusal is None,
            refusal=refusal,
            bound=f"{_budget_words(budget)}, which admits every turn before it is sent",
            unchecked=_EVIDENCE_UNCHECKED,
        )
    finally:
        await _release(client)


class _Prepared(NamedTuple):
    """What a generation's preparation built and checked: the ``analyst`` client, as a turn model, the budget and the
    loop's limits."""

    client: BoundCompletionClient
    model: TurnModel
    budget: ReportBudget
    limits: WriterLimits


async def analysis_generate(
    host: OpsHost, campaign_id: str, scope_id: str, *, writer: ReportWriter, model: str | None = None
) -> JobsStarted:
    """Start a campaign's generation as a job, after the checks that cost a read or two; the job writes its report.

    The call checks only what is cheap — the host generates analyses here and has set a report budget, the campaign
    exists, no generation of it is running, and a writer the call names is one the host allows — then records the
    attempt ``running``, carrying its lease, and returns its job. The job prepares: it builds the host's ``analyst``
    client, holds the host's default writer to its allowed list, refuses a client that is not a turn model (naming the
    two methods a wrapper forwards) and one whose turns cannot be bounded before they are sent. A refusal there ends the
    attempt ``failed`` with ``refused before any call: <why>`` and the refusal's kind (``job_poll`` reads it), releases
    the client, and freezes, sends and spends nothing.

    **The report writer writes the analysis.** The job runs ``writer`` (``write_report`` on the campaign, which freezes
    its evidence first) on the ``analyst`` client, both the analyst and the report editor, every turn admitted against
    the host's report budget (``AnalysisGeneration.report_budget``, the only admission: a caller cannot raise it), the
    whole held to ``AnalysisGeneration.limits`` (passed unchanged) and bounded by ``AnalysisGeneration.budget_s``. On an ending that stores a document (``stored``,
    ``stored_with_findings``, ``over_budget`` with a reviewed draft), the job mints an analysis linking it
    (``basis`` ``doc``, ``report_doc_id``), its recommended variant when the document recommends exactly one arm, and
    an insight per durable claim, and records the attempt ``stored``. An ending that stores nothing (``refused``,
    ``review_failed``, ``failed``, or a store that refused the document) ends the attempt ``refused`` or ``failed``
    with the ending and its reason. Either way the attempt keeps what the turns came to (turns as calls, the unpriced
    ones counted apart, the priced cost a floor beside them), and the report's spend is ledgered once, in one row under
    purpose ``analysis``, when the writing ends, so :func:`~threetears.evals.ops.scope_out_of_run_spend` reads it: its
    cost is ``None`` (unpriced) when any turn was unpriced, never $0. The host's out-of-run cap is not read: the
    report budget is the only admission. The attempt's and the analysis's provenance is the analyst skill's digest.

    :param host: The host: its generation settings, its clients and the job manager the task runs under.
    :ptype host: OpsHost
    :param campaign_id: The campaign to analyse.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param writer: What writes the report, bound to the caller who started it (the action's
        :func:`~threetears.evals.actions.report_writing.campaign_report_writer`).
    :ptype writer: ReportWriter
    :param model: The writer model, or ``None`` for the host's default.
    :ptype model: str | None
    :return: The generation's one job. Poll it: a refusal its preparation met ends the job refused rather than raising
        here.
    :rtype: JobsStarted
    :raises ValidationFailedError: The host generates no analyses here, has set no report budget, or the named writer
        model is not one the host allows (``HostProfile.analysis_writer_models``).
    :raises ConflictError: A generation of this campaign is already running — in this process, or in any process:
        the campaign's generation slot names an attempt whose lease is live. The refusal names that attempt's
        job.
    :raises NotFoundError: No campaign with that id in the scope.
    """
    generation = _generation_settings(host)
    report_budget = _report_budget_of(generation)
    manager = host.launch.job_manager
    key = generation_key(campaign_id, scope_id)
    eval_host = host.eval_host
    executor = eval_host.blocking_executor
    held = manager.active_task_ids(key)
    if held and not await run_blocking(executor, _live_attempt, eval_host, held[0], scope_id):
        # The task still holding the key in this process has already written its attempt's ending, so job_poll says
        # it is done: it is unwinding (releasing its client and its slot). Wait for that rather than refusing a start
        # its caller was just told it may make; a task that does not finish unwinding is refused as running.
        try:
            await asyncio.wait_for(manager.wait_for(held), timeout=ENDED_TASK_UNWIND_S)
        except TimeoutError:
            # NOSILENT: a task still unwinding after the wait keeps the key, and is refused as running below
            pass
        held = manager.active_task_ids(key)
    holder = held[0] if held else await run_blocking(executor, _live_slot_holder, eval_host, campaign_id, scope_id)
    if holder is not None:
        raise _already_running(campaign_id, holder)
    campaign = await run_blocking(executor, _refuse_unstartable, eval_host, campaign_id, scope_id, model)
    skill_digest = await run_blocking(executor, _skill_digest)

    started = running_analysis_attempt(
        attempt_id=str(uuid.uuid7()),
        campaign_id=campaign_id,
        scope_id=scope_id,
        model=model,
        prompt_id=skill_digest,
    )
    attempt_id = started.id
    # What the attempt reads as it ends without a call: the running record, and once the job has built the client, the
    # writer it resolved.
    current = started
    leases = host.launch.job_leases

    def save_started() -> None:
        # Written carrying its lease before the job starts, so its first poll — in any process — finds it running.
        started.lease = leases.grant()
        eval_host.storage.save_analysis_attempt(started)

    await run_blocking(executor, save_started)

    async def end_unrun(why: str, *, outcome: AttemptOutcome = "failed", refused_by: Exception | None = None) -> None:
        # The generation goes no further and made no call: end the attempt it recorded, so none is left reading
        # `running` that nothing runs. Every path that stops between the attempt's first write and its first turn comes
        # here — in the call (the slot lost, the job not started) and in the job (its preparation refused or
        # cancelled). A failed ending write is logged, not raised: the caller raises the reason it stopped.
        ended = current.model_copy(
            update={
                "outcome": outcome,
                "error": f"refused before any call: {why}" if outcome == "failed" else why,
                "created_at": utc_now_iso(),
                "refusal": job_refusal(refused_by) if isinstance(refused_by, EvalServiceError) else None,
            }
        )
        try:
            await run_blocking(executor, write_held_record, eval_host.storage, ANALYSIS_JOB, ended, leases=leases)
        except JobFenced:
            # NOSILENT: the attempt already reads ended (written lost elsewhere), which is the answer every poll reads
            log.warning(
                "eval.generate_analysis campaign=%s scope=%s attempt %s stopped before any call (%s); it had already "
                "ended",
                campaign_id,
                scope_id,
                attempt_id,
                why,
            )
        # prawduct:ok-broad-except — the reason the generation stopped is what its caller is told; this is logged beside it
        except Exception as unwritten:
            # NOSILENT: every caller of end_unrun raises the reason the generation stopped right after it, and the
            # attempt, which made no call, reads lost once its lease lapses rather than running.
            contain(
                unwritten,
                log=log,
                what=(
                    f"eval.generate_analysis campaign={campaign_id} scope={scope_id} attempt {attempt_id} stopped "
                    f"before any call ({why}), and its ending could not be written; it reads lost once its lease "
                    "lapses"
                ),
                level="error",
            )

    # The one step that decides between processes: a conditional write of the campaign's generation slot. The
    # attempt is recorded first, so a start that loses this race finds the winner's attempt to name.
    try:
        holder = await run_blocking(executor, _claim_generation_slot, eval_host, campaign_id, scope_id, attempt_id)
    except BaseException as broke:
        await end_unrun(
            "its start was cancelled"
            if isinstance(broke, asyncio.CancelledError)
            else f"the campaign's generation slot could not be claimed: {broke}"
        )
        raise
    if holder is not None:
        refusal = _already_running(campaign_id, holder)
        await end_unrun(str(refusal), refused_by=refusal)
        raise refusal

    async def resolved(writer_model: str) -> None:
        # The writer the client resolved goes on the running attempt before anything is frozen, so a refusal after it
        # leaves an attempt naming its writer. The ending writes it again, so a write that fails here loses nothing.
        nonlocal current
        current = started.model_copy(update={"generator_model": writer_model, "created_at": utc_now_iso()})
        try:
            await run_blocking(executor, write_held_record, eval_host.storage, ANALYSIS_JOB, current, leases=leases)
        except StorageError, ConflictError:
            # NOSILENT: the ending rewrites it; until then the running attempt reads as it was started
            log.warning(
                "eval.generate_analysis attempt %s campaign=%s: its resolved writer could not be recorded before its "
                "first turn; the ending records it",
                attempt_id,
                campaign_id,
            )

    async def built() -> _Prepared:
        # The client, the host's default writer, the turn model and its pricing. Whatever stops it, the client it
        # built is released before the stop goes on: nothing will run on it.
        client = await _analyst_client(eval_host, model)
        try:
            turn_model = _turn_model(client)
            await resolved(client.model_name)
            budget = report_budget()
            # A dollar budget admits each turn on its price, so a model whose turns cannot be priced is refused here; a
            # token and turn budget asks for no price.
            if budget.usd is not None and (unpriced := await _pricing_refusal(turn_model)) is not None:
                raise ValidationFailedError(
                    f"the report was not started, and nothing was frozen, sent or spent: {unpriced}",
                    remedy=(
                        "hand the analyst role a model whose turns the client can price (load its price catalogue, "
                        "give the model a pricer, or name a model it prices)"
                    ),
                )
            return _Prepared(client=client, model=turn_model, budget=budget, limits=generation.limits)
        except BaseException:
            await _release(client)
            raise

    async def prepare() -> _Prepared | None:
        # Every refusal ends the attempt with nothing frozen or sent. A refusal is a designed ending, not a task
        # failure: the attempt records it and this returns None, so the manager logs no error for it. What nobody
        # designed for (a store that raised) is recorded the same way and raised on as itself.
        try:
            async with manager.job_timeout_factory(generation.preparation_budget_s):
                return await built()
        except JobFenced:
            raise
        except asyncio.CancelledError:
            await end_unrun("cancelled before any call, while the generation was being prepared", outcome="cancelled")
            raise
        except EvalJobTimeout as timed_out:
            await end_unrun(f"preparation did not finish in {format_number(timed_out.budget_s)} s")
            return None
        except EvalServiceError as refused:
            await end_unrun(str(refused), refused_by=refused)
            return None
        # prawduct:ok-broad-except — recorded on the attempt, then raised on as itself
        except Exception as broke:
            await end_unrun(f"the generation could not be prepared ({type(broke).__name__}: {broke})")
            raise

    async def work() -> None:
        try:
            prepared = await prepare()
            if prepared is None:
                return
            await _write_doc_analysis(
                host,
                prepared,
                WrittenAttempt(
                    attempt_id=attempt_id,
                    campaign_id=campaign_id,
                    scope_id=scope_id,
                    generator_model=prepared.client.model_name,
                    prompt_id=skill_digest,
                    started_at=started.started_at,
                    leases=leases,
                ),
                campaign=campaign,
                writer=writer,
                deadline=manager.job_timeout_factory(generation.budget_s),
            )
        except JobFenced as fenced:
            # NOSILENT: the attempt's record, written lost elsewhere, is the answer every poll reads; nothing here
            # wrote after it.
            log.warning("eval.generate_analysis campaign=%s scope=%s stopped: %s", campaign_id, scope_id, fenced)

    try:
        # The writing enters its own budget (``deadline``), so its attempt can say it timed out: one entered around the
        # task would reach the writing as a cancel.
        manager.start_task(attempt_id, work, budget_s=None, key=key, record=(ANALYSIS_JOB, scope_id))
    except ConflictError as lost_race:
        # Another start in this process took the key while this one recorded its attempt.
        await end_unrun(str(lost_race), refused_by=lost_race)
        raise
    return JobsStarted(
        jobs=[
            JobHandle(
                job_id=analysis_job_id(campaign_id, attempt_id),
                kind="analysis",
                target_id=campaign_id,
                label=started.generator_model,
            )
        ]
    )


def _attempt_tally(tally: Tally, *, prompt_version: str, refusals: Sequence[str] = ()) -> GenerationTally:
    """What the attempt keeps of the turns sent: each turn sent a call, its reply back or not; each turn whose reply
    never came or reported no cost an unpriced call; the priced turns' cost (a floor whenever a turn is unpriced);
    and the model each reply named."""
    turns = tally.turns
    return GenerationTally(
        calls=tally.sent,
        returned=len(turns),
        unreported_cost=sum(1 for turn in turns if turn.cost_usd is None),
        token_cost=math.fsum(turn.cost_usd for turn in turns if turn.cost_usd is not None),
        reported_models=[turn.served_model or turn.model for turn in turns],
        refusals=list(refusals),
        prompt_version=prompt_version,
    )


def _summed(counts: Sequence[int | None]) -> int | None:
    """The sum, or ``None`` when any count is unknown."""
    return None if any(count is None for count in counts) else sum(count or 0 for count in counts)


def _budget_words(budget: ReportBudget) -> str:
    """The report's budget in words, as its ledger row and its estimate say what admitted the turns."""
    if budget.usd is not None:
        return f"the report budget of {budget.spelled}"
    # A token and turn budget names no dollars: it is never read as a $0 cap.
    return f"the {budget.spelled} report budget, with no dollar cap"


def _spend_row(
    tally: Tally, attempt: WrittenAttempt, campaign: EvalCampaign, budget: ReportBudget
) -> OutOfRunSpend | None:
    """The report's one out-of-run ledger row, ``None`` for a report that sent no turn.

    It states every turn sent (``calls``, a turn whose reply never came included), the priced share and the unpriced
    turns apart (``priced_usd``, ``unpriced_calls``) and what admitted them (``admitted_by``, the report budget). Its
    total, ``cost_usd``, is ``None`` whenever a turn is unpriced (``Tally.spent_usd``): never $0, and never the priced
    share read as the whole. Its tokens are ``None`` when a turn's reply never came, since nothing reported them.
    """
    if not tally.sent:
        return None
    turns = tally.turns
    complete = tally.sent == len(turns)
    priced, unpriced = tally.priced_usd(), tally.unpriced_turns()
    served = {turn.served_model for turn in turns}
    sources = sorted({turn.price_source for turn in turns if turn.price_source is not None})
    return OutOfRunSpend(
        scope_id=attempt.scope_id,
        purpose="analysis",
        model=attempt.generator_model,
        served_model=served.pop() if len(served) == 1 else None,
        outcome="completed",
        prompt_tokens=_summed([turn.input_tokens for turn in turns]) if complete else None,
        completion_tokens=_summed([turn.output_tokens for turn in turns]) if complete else None,
        reasoning_tokens=_summed([turn.reasoning_tokens for turn in turns]) if complete else None,
        # The total is the priced share itself when every turn is priced: one figure, never two sums of it.
        cost_usd=None if unpriced else priced,
        price_source=", ".join(sources) if sources else None,
        subject_id=campaign.subject_id,
        campaign_id=attempt.campaign_id,
        calls=tally.sent,
        unpriced_calls=unpriced,
        raised_calls=tally.unreturned_turns(),
        priced_usd=priced,
        admitted_by=_budget_words(budget),
    )


async def _ledgered(
    host: EvalHost, tally: Tally, attempt: WrittenAttempt, campaign: EvalCampaign, budget: ReportBudget
) -> None:
    """Ledger a report's spend under purpose ``analysis``, once its writing has ended; a failed write is logged."""
    row = _spend_row(tally, attempt, campaign, budget)
    if row is None:
        return

    def write() -> None:
        host.storage.save_out_of_run_spend(row)

    try:
        await run_blocking(host.blocking_executor, write)
    except StorageError:
        # NOSILENT: the attempt's ending, written next, keeps what the turns cost; the ledger misses this report.
        log.exception(
            "eval.generate_analysis attempt %s campaign=%s: its %d turn(s) could not be ledgered; the attempt keeps "
            "what they cost",
            attempt.attempt_id,
            attempt.campaign_id,
            tally.sent,
        )


async def _write_doc_analysis(
    host: OpsHost,
    prepared: _Prepared,
    attempt: WrittenAttempt,
    *,
    campaign: EvalCampaign,
    writer: ReportWriter,
    deadline: AbstractAsyncContextManager[Any],
) -> None:
    """Write the campaign's report on the prepared client within ``deadline``, then store its analysis and its ending.

    The account is :func:`analysis_generate`'s. Every ending is recorded on the attempt with the tally's counts; a
    stored document mints its analysis and insights (:func:`~threetears.evals.analysis.doc_analysis.doc_analysis`),
    committed by :func:`~threetears.evals.analysis.service.record_written_generation`.

    :raises EvalJobTimeout: ``deadline`` ran out; the attempt reads ``timed_out``.
    :raises JobFenced: The attempt was written lost before its ending.
    """
    eval_host = host.eval_host
    tally = Tally()
    written: ReportWritten | None = None
    not_stored: ReportNotStored | None = None

    ledgered = False

    async def ledger() -> None:
        # Once per report, whichever ending writes it.
        nonlocal ledgered
        if not ledgered:
            ledgered = True
            await _ledgered(eval_host, tally, attempt, campaign, prepared.budget)

    async def ended(
        outcome: AttemptOutcome, error: str, *, refused_by: EvalServiceError | None = None, refusals: Sequence[str] = ()
    ) -> None:
        # The ending is written whatever the ledger's write does: a report is never left with no ending.
        try:
            await ledger()
        finally:
            await record_written_generation(
                eval_host,
                attempt,
                outcome=outcome,
                tally=_attempt_tally(tally, prompt_version=attempt.prompt_id, refusals=refusals),
                bundle_fingerprint="",
                error=error,
                refused_by=refused_by,
            )

    try:
        async with prepared.client, deadline:
            try:
                # The public protocol is opaque (its docstring says why); the job calls it as :class:`ReportWriterCall`.
                written = await cast("ReportWriterCall", writer)(
                    host,
                    model=prepared.model,
                    budget=prepared.budget,
                    campaign_id=attempt.campaign_id,
                    limits=prepared.limits,
                    tally=tally,
                )
            except ReportNotStored as refused:
                not_stored, written = refused, refused.written
    except EvalJobTimeout as timed_out:
        # Its own budget ran out mid-flight: not someone deciding to stop it, so not `cancelled`.
        await ended("timed_out", timed_out_detail(timed_out))
        raise
    except asyncio.CancelledError:
        await ended(
            "cancelled",
            f"cancelled mid-flight by the caller's budget or disconnect, or a shutdown, after {tally.sent} turn(s)",
        )
        raise
    except (EvalServiceError, ValueError) as refused:
        if tally.sent:
            # Raised after turns were sent: no refusal of the start, and never recorded as one. Kept by its kind.
            await ended("failed", f"the report writer failed after {tally.sent} turn(s) ({type(refused).__name__})")
            raise
        # Refused before any turn: the freeze (the campaign resolves no runs), or the writer's own check of the model
        # (one it cannot bound, or one that sends no turn).
        unsent = (
            refused
            if isinstance(refused, EvalServiceError)
            else ValidationFailedError(str(refused), remedy="start the generation again")
        )
        await ended("failed", f"refused before any call: the report was not written: {refused}", refused_by=unsent)
        return
    # prawduct:ok-broad-except — recorded on the attempt by its kind alone (a provider's text may hold the account), then raised on as itself
    except Exception as broke:
        await ended("failed", f"the report writer failed after {tally.sent} turn(s) ({type(broke).__name__})")
        raise
    outcome = written.outcome
    if written.gaps_not_stored is not None:
        await _note_gaps_not_stored(eval_host, attempt, written)
    if not_stored is not None or written.stored is None:
        reason = (
            f"{not_stored}; any session it wrote before the store failed is under report document "
            f"{not_stored.report_doc_id}, which delete_report_doc removes"
            if not_stored is not None
            else outcome.reason or "it stored nothing"
        )
        why: EvalServiceError
        if outcome.failure is not None:
            # The client's failure: its text may hold a provider's account, so the record keeps what the host's
            # describer says is safe, as a planned generation's does, and a withheld payload is the provider's refusal.
            failure = describe_failure(
                eval_host.failure_describer, outcome.failure, logger=log, where="eval.generate_analysis"
            )
            error = f"the report writer ended {written.ending}: {failure.description}"
            why = (
                ValidationFailedError(error, remedy="start the generation again")
                if traceback_is_safe(failure)
                else ProviderRefusedError(f"analysis generation failed: {failure.description}")
            )
        else:
            error = f"the report writer ended {written.ending}: {reason}"
            why = ValidationFailedError(
                error, remedy="start the generation again; the attempt keeps what its turns cost"
            )
        await ended(
            "refused" if outcome.ending == "refused" else "failed",
            error,
            refused_by=why,
            refusals=[fault.line() for fault in outcome.faults],
        )
        return
    stored = written.stored
    # Everything between the stored document and its attempt's ending: a failure here is recorded as the ending, so a
    # stored document is never left with no attempt saying what became of it. From record_written_generation on, the
    # ending is its own to write.
    try:
        snapshot = await run_blocking(
            eval_host.blocking_executor, eval_host.storage.load_report_snapshot, written.snapshot_id, attempt.scope_id
        )
        if snapshot is None or snapshot.bundle is None or snapshot.bundle_assembled_at is None:
            raise StorageError(
                f"the report document {stored.id} is stored, and its snapshot {written.snapshot_id} holds no bundle"
            )
        minted = doc_analysis(
            CampaignBundle.from_dict(snapshot.bundle, what=f"report snapshot {snapshot.id}"),
            await snapshot_input(host, attempt.scope_id, written.snapshot_id),
            stored,
            input_digest=snapshot.report_input_digest,
            generator_model=attempt.generator_model,
            skill_digest=attempt.prompt_id,
            bundle_assembled_at=snapshot.bundle_assembled_at,
            token_cost=tally.spent_usd(),
            bounces=tally.to_json()["total"]["bounce_kinds"],
        )
        await ledger()
    except asyncio.CancelledError:
        await ended(
            "cancelled",
            f"cancelled after report document {stored.id} was stored, before its analysis was; {_unlinked(stored.id)}",
        )
        raise
    # prawduct:ok-broad-except — a stored document's analysis failing is a defect, never the writer's: recorded as the ending, then raised on as itself
    except Exception as unminted:
        await ended(
            "failed",
            f"the report document {stored.id} is stored, and its analysis could not be built or ledgered "
            f"({type(unminted).__name__}: {unminted}); {_unlinked(stored.id)}",
        )
        raise
    insights = await record_written_generation(
        eval_host,
        attempt,
        outcome="stored",
        tally=_attempt_tally(tally, prompt_version=attempt.prompt_id),
        bundle_fingerprint=minted.analysis.generation.bundle_fingerprint,
        stored=(minted.analysis, minted.insights),
    )
    log.info(
        "eval.generate_analysis stored campaign=%s analysis=%s report_doc=%s ending=%s insights=%d turns=%d model=%s",
        attempt.campaign_id,
        minted.analysis.id,
        stored.id,
        written.ending,
        len(insights),
        tally.sent,
        attempt.generator_model,
    )


async def _note_gaps_not_stored(host: EvalHost, attempt: WrittenAttempt, written: ReportWritten) -> None:
    """Note on the attempt that its report write's gaps were not stored, so ``job_poll`` shows it beside the ending.

    The write recorded the failure on its logger and carried it on its result (``ReportWritten.gaps_not_stored``); a
    note the store also refuses is recorded here at error level, and the job goes on to its ending either way.
    """
    entry = (
        f"report writer: {len(written.gaps)} gaps and notices were not stored ({written.gaps_not_stored}); the "
        "generation's ending is unaffected"
    )
    try:
        await run_blocking(
            host.blocking_executor,
            note_job_fault,
            host.storage,
            ANALYSIS_JOB,
            attempt.attempt_id,
            attempt.scope_id,
            entry,
        )
    except (ConflictError, StorageError) as unnoted:
        contain(unnoted, log=log, what=f"eval.generate_analysis attempt {attempt.attempt_id}: {entry}", level="error")


def _unlinked(report_doc_id: str) -> str:
    """What an attempt's ending says of a stored report document no analysis links: what removes it."""
    return f"no analysis links report document {report_doc_id}, which delete_report_doc removes"


def _refuse_unstartable(host: EvalHost, campaign_id: str, scope_id: str, model: str | None) -> EvalCampaign:
    """Refuse a start whose campaign does not resolve, or that names a writer the host does not allow — one read.

    The checks a start makes in its call. Everything that reads the campaign's runs is its job's.

    :return: The campaign.
    :rtype: EvalCampaign
    :raises NotFoundError: No campaign with that id in the scope.
    :raises ValidationFailedError: ``model`` is not one the host allows.
    """
    campaign = host.storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id, remedy=FIND_CAMPAIGN)
    if model is not None and (ineligible := host.profile.analysis_writer_refusal(model)) is not None:
        raise ValidationFailedError(ineligible, remedy=WRITER_REMEDY)
    return campaign


def _already_running(campaign_id: str, attempt_id: str) -> ConflictError:
    """The refusal of a second generation of a campaign, naming the live one's job."""
    return ConflictError(
        f"a generation of campaign '{campaign_id}' is already running (job {analysis_job_id(campaign_id, attempt_id)})",
        remedy="poll it with job_poll rather than starting another",
    )


def _live_attempt(host: EvalHost, attempt_id: str, scope_id: str) -> bool:
    """Whether an attempt still runs in some process: read as every process reads it, settled first."""
    attempt, _ = settle_job_record(host.storage, ANALYSIS_JOB, attempt_id, scope_id)
    return attempt is not None and not ANALYSIS_JOB.ended(attempt)


def _live_slot_holder(host: EvalHost, campaign_id: str, scope_id: str) -> str | None:
    """The attempt holding the campaign's generation slot while it still runs; ``None`` when the slot is free.

    A read before anything is prepared or spent, so a start that would lose is refused at once; the claim
    (:func:`_claim_generation_slot`) is what decides.
    """
    slot, _ = host.storage.load_generation_slot_with_etag(campaign_id, scope_id)
    if slot is None or not _live_attempt(host, slot.attempt_id, scope_id):
        return None
    return slot.attempt_id


def _claim_generation_slot(host: EvalHost, campaign_id: str, scope_id: str, attempt_id: str) -> str | None:
    """Claim the campaign's generation slot for ``attempt_id`` — atomically, against every process.

    The slot is created with a create-only write when the campaign has none, and taken over with ``if_match`` from
    an attempt that has ended (its lease lapsed counts: it is settled first). Of two starts racing, one write
    lands; the other's is refused, re-reads, and finds the winner's attempt — recorded before its claim — live.

    :param host: Where the slot and the attempts live.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: Its scope.
    :ptype scope_id: str
    :param attempt_id: The attempt claiming it, already recorded ``running`` with its lease.
    :ptype attempt_id: str
    :return: ``None`` when the slot is claimed; otherwise the live attempt that holds it.
    :rtype: str | None
    :raises ConflictError: The slot kept changing under every attempt to claim it.
    """
    for _ in range(LEASE_WRITE_ATTEMPTS):
        slot, etag = host.storage.load_generation_slot_with_etag(campaign_id, scope_id)
        if slot is not None and slot.attempt_id != attempt_id and _live_attempt(host, slot.attempt_id, scope_id):
            return slot.attempt_id
        claim = EvalGenerationSlot(
            id=generation_slot_id(campaign_id), scope_id=scope_id, campaign_id=campaign_id, attempt_id=attempt_id
        )
        try:
            host.storage.claim_generation_slot(claim, if_match=etag)
        except ConflictError:
            # NOSILENT: another start wrote the slot first; the re-read names its attempt
            continue
        return None
    raise ConflictError(
        f"campaign '{campaign_id}''s generation slot kept changing", remedy="start the generation again"
    )


def report_read(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
    *,
    format: ReportFormat,
    trials: TrialDetail = "table",
    static: bool = False,
    report_doc_id: str | None = None,
    include_sessions: bool = False,
) -> ReportDocument:
    """The campaign's report, serialized in one form — the same report the command line's ``report`` prints.

    What "the campaign's report" is has one answer, :func:`~threetears.evals.analysis.campaign_report`: its newest
    analysis that is not archived, else a code-only report of its evidence. ``basis`` on the result says which, and
    ``analysis_basis`` what wrote the analysis:

    - ``doc`` (the report writer, which ``analysis_generate`` runs): the page its stored report document draws on its
      own snapshot, its made line at its foot, read as a stored document is (a figure that no longer resolves prints
      ``—``, a block that no longer draws prints its values or its caption, with one note to regenerate); ``doc`` is
      the document, as ``report-doc/1`` JSON;
    - ``plan`` (the report planner): the code-only page, its first screen saying in one line that the analysis is
      written in a shape this version no longer reads, and to regenerate it with ``analysis_generate``;
    - ``memo`` (an analysis stored before the report planner): laid out as written.

    A report document that no longer reads (its version, its model, or a snapshot gone or holding no bundle) reads as a
    plan does. Neither is ever refused, and ``needs_regeneration`` carries that one line, as the page prints it, then
    why in parentheses; the JSON body carries it too, so a stand-in never reads as a campaign never analysed.

    **The trials.** By default each aspect's trials section is the census line, the table of its failed and lost
    trials, each row naming its stored result (read one with ``result_get``), and the table of its passed trials;
    ``trials="all"`` adds each failed or lost trial's full record (what it asked, what the judge saw and said, and
    why), which is most of a large campaign's page. A report document's page draws none of them by default; with
    ``trials="all"`` each aspect's fold follows the document, closed, as its evidence record, every such trial's record
    in it, the document above it as it was stored. The JSON form carries no records either way.

    **Any stored version, and its transcripts** (the audit read). ``report_doc_id`` reads one of the campaign's report
    documents, not just the newest (:func:`~threetears.evals.analysis.service.report_doc_report`): its page, as above.
    ``include_sessions`` adds, to the JSON form's ``report_doc``, ``sessions``: every session that wrote it (the
    analyst's and each review round's), by name, each its stored ``eval_report_session`` document whole. Only the JSON
    form carries a transcript: a page a reader sees never holds one, so ``include_sessions`` with ``markdown`` or
    ``html`` is refused, saying to ask for ``json``.

    :param host: The host whose store holds the campaign.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param format: ``markdown``, ``html`` (one page, its charts drawn by the kit's inlined script; see ``static``) or
        ``json`` (the page's machine form: code's readings across the campaign's aspects, the recommendation, the
        call, each aspect's bundle by its fingerprint, as ``campaign.json`` beside a written page holds it, and
        ``report_doc``, the stored report document the page draws, or None).
    :ptype format: ReportFormat
    :param trials: ``table`` (the default) or ``all``, which adds each failed or lost trial's full record.
    :ptype trials: TrialDetail
    :param static: For ``html``, the static page, for a host whose policy lets a page neither run nor load anything:
        no ``<script>``, no ``<link>``, the system faces, every chart its table of values. Markdown and JSON are
        the same either way.
    :ptype static: bool
    :param report_doc_id: One of the campaign's stored report documents, any version, to read in place of the
        campaign's report; None for the campaign's report.
    :ptype report_doc_id: str | None
    :param include_sessions: With ``json``, add the transcripts of the report document the page draws.
    :ptype include_sessions: bool
    :return: The report in that form.
    :rtype: ReportDocument
    :raises NotFoundError: No campaign with that id in the scope; or no report document with ``report_doc_id`` in the
        scope, or one written for another campaign.
    :raises ValidationFailedError: ``include_sessions`` with a form other than ``json``.
    :raises RefusedInput: The campaign's reported analysis is a memo written over a bundle of another shape: refused
        with the regeneration that mends it (``analysis_generate``).
    """
    if include_sessions and format != "json":
        raise ValidationFailedError(
            f"include_sessions adds a report document's transcripts, which only the json form carries; a {format} "
            "page is read by a person, and never holds a transcript",
            remedy="ask for format='json' with include_sessions, or leave include_sessions off for the page",
        )
    if report_doc_id is None:
        report = campaign_report(host, campaign_id, scope_id, trials=trials)
    else:
        report = report_doc_report(host, campaign_id, scope_id, report_doc_id, trials=trials)
    analysis_id = report.analysis_id or report.unread_analysis_id
    analysis = get_analysis(host.storage, analysis_id, scope_id) if analysis_id is not None else None
    body = serialize_report(report, format, static=static)
    if include_sessions and report.report_doc is not None:
        body = _with_sessions(host, report.json_document(), scope_id)
    return ReportDocument(
        campaign_id=campaign_id,
        basis=report.basis,
        analysis_id=analysis_id,
        analysis_basis=analysis.basis if analysis is not None else None,
        doc=dict(cast("dict[str, Any]", report.report_doc["doc"])) if report.report_doc is not None else None,
        needs_regeneration=report.needs_regeneration,
        format=format,
        body=body,
    )


def _with_sessions(host: EvalHost, document: dict[str, object], scope_id: str) -> str:
    """The JSON form with every session of the report document it carries, under ``report_doc.sessions``. Blocking."""
    served = dict(cast("dict[str, Any]", document["report_doc"]))  # Any: the stored document's JSON
    sessions = host.storage.load_report_sessions(str(served["id"]), scope_id)
    served["sessions"] = {name: session.to_dict() for name, session in sessions.items()}
    return json.dumps({**document, "report_doc": served}, indent=2)


class ProposedBar(EvalBaseModel):
    """One bar a baseline proposes, for a person to adopt, tighten or leave.

    :ivar measure: The measure the bar is on, a host-declared one.
    :ivar threshold: The proposed threshold, in the measure's unit.
    :ivar higher_is_better: Which side of ``threshold`` clears it.
    :ivar rationale: How the threshold was seeded from the incumbent's measurement.
    :ivar vacuous: Whether nothing could fail it — adopting it as written registers no standard.
    :ivar vacuous_reason: Why it is vacuous; ``None`` when it discriminates.
    """

    measure: str
    threshold: float
    higher_is_better: bool
    rationale: str
    vacuous: bool
    vacuous_reason: str | None


class BarProposals(EvalBaseModel):
    """What a baseline campaign proposes as its behavior's bars. Read-only: nothing here is registered.

    :ivar campaign_id: The baseline campaign measured.
    :ivar behavior: The behavior every proposal governs, the campaign's.
    :ivar variant_key: The incumbent: the baseline's one cell.
    :ivar proposals: One per declared measure with a better end that could be seeded, in measure-name order.
    :ivar not_proposed: ``{reading: why}`` for every reading the cell carries that no bar could be proposed on.
    """

    campaign_id: str
    behavior: str
    variant_key: str
    proposals: list[ProposedBar]
    not_proposed: dict[str, str]


def bars_propose(host: EvalHost, campaign_id: str, scope_id: str) -> BarProposals:
    """Propose a bar on every measure a single-cell baseline campaign measured its incumbent on.

    :func:`~threetears.evals.analysis.bar_proposals.propose_bars`, read through: each proposal seeded from the
    incumbent's measured interval and flagged vacuous where nothing could fail it, and every reading nothing
    could be proposed on named with why. **Nothing is registered**: a bar reaches a registry only when a person
    writes it into the host's registrations, since :class:`~threetears.evals.kernel.host.BarRegistry` has no
    mutation API.

    :param host: The host whose measures, bars and store this reads.
    :ptype host: EvalHost
    :param campaign_id: The baseline campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :return: The proposals.
    :rtype: BarProposals
    :raises NotFoundError: No campaign with that id in the scope.
    :raises ValidationFailedError: The campaign measured no cell or more than one: a baseline is one configuration
        under one rig.
    """
    proposed = propose_bars(host, campaign_id, scope_id=scope_id)
    return BarProposals(
        campaign_id=proposed.campaign_id,
        behavior=proposed.behavior,
        variant_key=proposed.variant_key,
        proposals=[
            ProposedBar(
                measure=proposal.bar.measure,
                threshold=proposal.bar.threshold,
                higher_is_better=proposal.bar.higher_is_better,
                rationale=proposal.bar.rationale,
                vacuous=proposal.vacuous,
                vacuous_reason=proposal.reason or None,
            )
            for proposal in proposed.proposals
        ],
        not_proposed=dict(proposed.not_proposed),
    )


def bar_proposals_text(proposals: BarProposals) -> str:
    """Proposals as a person reads them: each bar with its seed and any vacuity, then what was not proposed."""
    lines = [
        f"bar proposals for behavior {proposals.behavior!r} from baseline campaign {proposals.campaign_id} "
        f"(incumbent {proposals.variant_key}); nothing is registered — adopt a bar by writing it into the host's "
        "registrations"
    ]
    for bar in proposals.proposals:
        side = ">=" if bar.higher_is_better else "<="
        flag = f" — VACUOUS, do not adopt as written: {bar.vacuous_reason}" if bar.vacuous else ""
        lines.append(f"- {bar.measure} {side} {format_number(bar.threshold)}{flag}")
        lines.append(f"  seeded from {bar.rationale}")
    if not proposals.proposals:
        lines.append("- no bar could be proposed")
    for reading, why in sorted(proposals.not_proposed.items()):
        lines.append(f"- not proposed on {reading}: {why}")
    return "\n".join(lines)


def analysis_delete(host: EvalHost, analysis_id: str, scope_id: str, *, confirm: str | None) -> AnalysisDeleted:
    """Destroy a stored analysis — its insights stay; archive is the reversible answer.

    :param host: The host whose store holds the analysis.
    :ptype host: EvalHost
    :param analysis_id: The analysis.
    :ptype analysis_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param confirm: Must echo ``analysis_id``.
    :ptype confirm: str | None
    :return: What was removed.
    :rtype: AnalysisDeleted
    :raises NotFoundError: No analysis with that id in the scope — refused before ``confirm`` is read.
    :raises ValidationFailedError: ``confirm`` does not echo the id.
    :raises StorageError: The delete failed.
    """
    removed = delete_analysis(host.storage, get_analysis(host.storage, analysis_id, scope_id), confirm=confirm)
    return AnalysisDeleted(analysis_id=removed["analysis_id"], campaign_id=removed["campaign_id"])


def insights_list(
    host: EvalHost, scope_id: str, *, subject_id: str | None = None, source_campaign_id: str | None = None
) -> InsightListing:
    """The scope's insight ledger, newest observation first, each with where it stands.

    Over :func:`~threetears.evals.analysis.list_insights`: both filters match ids exactly, and an id that
    matches nothing is an answer (that campaign minted no insights), not a refusal.

    :param host: The host whose store holds the ledger.
    :ptype host: EvalHost
    :param scope_id: The scope whose ledger is read.
    :ptype scope_id: str
    :param subject_id: Only insights about this subject.
    :ptype subject_id: str | None
    :param source_campaign_id: Only insights an analysis of this campaign minted.
    :ptype source_campaign_id: str | None
    :return: The listing.
    :rtype: InsightListing
    """
    insights = list_insights(host.storage, scope_id, subject_id=subject_id, source_campaign_id=source_campaign_id)
    standing = _insight_standing(host, insights, scope_id)
    return InsightListing(
        subject_id=subject_id,
        source_campaign_id=source_campaign_id,
        filters=describe_insight_id_filters(subject_id, source_campaign_id),
        insights=[
            InsightLine(
                id=insight.id,
                subject_id=insight.subject_id,
                statement=insight.statement,
                confidence=insight.confidence,
                scope=insight.scope,
                observed_at=insight.observed_at,
                source_campaign_id=insight.source_campaign_id,
                source_analysis_id=insight.source_analysis_id,
                standing=_standing_of(insight, standing),
            )
            for insight in insights
        ],
    )


def insight_get(host: EvalHost, insight_id: str, scope_id: str) -> InsightDetail:
    """One insight in full: everything stored on it, and where it stands.

    :param host: The host whose store holds the ledger.
    :ptype host: EvalHost
    :param insight_id: The insight.
    :ptype insight_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :return: The insight and its standing.
    :rtype: InsightDetail
    :raises NotFoundError: No insight with that id in the scope.
    """
    insight = host.storage.load_insight(insight_id, scope_id)
    if insight is None:
        raise NotFoundError("insight", insight_id, remedy=FIND_INSIGHT)
    return InsightDetail(insight=insight, standing=_standing_of(insight, _insight_standing(host, [insight], scope_id)))


def insight_delete(host: EvalHost, insight_id: str, scope_id: str, *, confirm: str | None) -> InsightDeleted:
    """Destroy one insight — the intended answer to a wrong one, since an insight has no archive.

    A live insight is fed to every later generation over its subject as prior context, so a wrong one keeps
    steering analyses until it is gone (:func:`~threetears.evals.run.delete_insight`). Archiving its analysis
    retracts every insight that analysis minted; this removes one.

    :param host: The host whose store holds the ledger.
    :ptype host: EvalHost
    :param insight_id: The insight.
    :ptype insight_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param confirm: Must echo ``insight_id``.
    :ptype confirm: str | None
    :return: What was removed.
    :rtype: InsightDeleted
    :raises NotFoundError: No insight with that id in the scope — refused before ``confirm`` is read.
    :raises ValidationFailedError: ``confirm`` does not echo the id.
    :raises StorageError: The delete failed.
    """
    removed = delete_insight(host.storage, insight_id, scope_id, confirm=confirm)
    return InsightDeleted(insight_id=removed["insight_id"], source_campaign_id=removed["source_campaign_id"])


__all__ = [
    "ReportWriterCall",
    "AnalysisDeleted",
    "AnalysisGenerationEstimate",
    "AnalysisLine",
    "AnalysisListing",
    "BarProposals",
    "CampaignCallLine",
    "CampaignChanges",
    "CampaignDefinition",
    "CampaignLine",
    "CampaignListing",
    "InsightDeleted",
    "InsightDetail",
    "InsightLine",
    "InsightListing",
    "InsightStandingName",
    "ProposedBar",
    "ReportDocument",
    "UndescribableArmsLine",
    "UndescribableArmsListing",
    "analyses_list",
    "analyses_undescribable",
    "analysis_archive",
    "analysis_delete",
    "analysis_estimate",
    "analysis_generate",
    "bar_proposals_text",
    "bars_propose",
    "campaign_archive",
    "campaign_create",
    "campaign_record",
    "campaign_runs_add",
    "campaign_runs_remove",
    "campaign_update",
    "campaigns_list",
    "insight_delete",
    "insight_get",
    "insights_list",
    "report_read",
]
