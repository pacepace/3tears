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
import math
import uuid
from collections.abc import Callable, Sequence
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
from threetears.evals.kernel.numbers import format_number, group_thousands, spell_dollars
from threetears.evals.analysis.planned_analysis import report_plan_of
from threetears.evals.analysis.report import ReportBasis, TrialDetail, analysis_verdict
from threetears.evals.analysis.report.serialize import ReportFormat, serialize_report
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.campaign_bundle import CampaignBundle
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
    running_analysis_attempt,
)
from threetears.evals.kernel.report_doc import ReportDoc
from threetears.evals.schema.base import EvalBaseModel, VerbatimText
from threetears.evals.kernel.declaration import CampaignCall, Margin, OverriddenBar
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


class CampaignLine(EvalBaseModel):
    """One campaign, as a listing shows it."""

    id: str
    name: str
    subject_id: str
    behavior: str
    run_count: int
    archived: bool
    exploratory: bool = Field(
        description="Whether no aspect of the campaign declares a design — derived on every read, never stored"
    )
    aspects: list[str] = Field(description="The labels of the campaign's aspects, in order; at least one.")


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
    basis: AnalysisBasis = Field(
        description=(
            "What wrote the analysis: `memo` (the memo writer; findings count the memo's findings), `plan` (the "
            "report planner; findings count the plan's evidence sections, and report_read carries the plan) or `doc` "
            "(the report writer; findings count the document's claims, and report_doc_id names it)."
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
            "Why this build cannot read the evidence the analysis was frozen over, and what to do: set when it was "
            "written over a bundle of another shape, or as a report plan or document this build no longer reads (or "
            "deleted apart from it), so its report and page are refused until it is generated again "
            "(analysis_generate), and a plan or document it cannot read counts no findings. None when it reads."
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
        description="`analysis` when the report renders the campaign's analysis; `code_only` when it has none."
    )
    analysis_id: str | None = Field(description="The analysis the report renders; None on a code-only report.")
    analysis_basis: AnalysisBasis | None = Field(
        default=None,
        description=(
            "What wrote the analysis the report renders: `memo` or `plan` (whose plan is `plan`); None on a code-only "
            "report."
        ),
    )
    plan: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The report plan a `plan` analysis was written as, as `report-plan/1` JSON (its prose names figures by "
            "key); None for a memo or a code-only report."
        ),
    )
    format: ReportFormat
    body: VerbatimText = Field(
        description="The report in that form, exactly as its serializer wrote it: Markdown, canonical JSON, or "
        "HTML (one page whose one script is the kit's, or, static, with no script and no link). A `plan` analysis's "
        "Markdown and HTML are the page its plan draws, each aspect's fold a closed appendix after its Method."
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
    )


def campaigns_list(host: EvalHost, scope_id: str, *, archived: bool | None = None) -> CampaignListing:
    """The scope's campaigns.

    :param host: The host whose store is read.
    :ptype host: EvalHost
    :param scope_id: The scope.
    :ptype scope_id: str
    :param archived: Only archived (``True``) or only active (``False``) campaigns; ``None`` for both.
    :ptype archived: bool | None
    :return: The listing.
    :rtype: CampaignListing
    """
    return CampaignListing(
        campaigns=[_campaign_line(c) for c in list_campaigns(host.storage, scope_id, archived=archived)]
    )


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
    campaign = create_campaign(
        host.storage,
        definition.model_dump(exclude={"aspects_from", "exploratory"}),
        scope_id=scope_id,
        created_by=created_by,
        profile=host.profile,
        exploratory=definition.exploratory,
        aspects_from=definition.aspects_from,
    )
    return _campaign_line(campaign)


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
    return _campaign_line(campaign)


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
    return _campaign_line(add_runs_to_campaign(host.storage, campaign_id, scope_id, run_ids, aspect=aspect))


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
    return _campaign_line(remove_runs_from_campaign(host.storage, campaign_id, scope_id, run_ids, aspect=aspect))


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
    return _campaign_line(set_campaign_archived(host.storage, campaign_id, scope_id, archived=archived))


def _analysis_line(host: EvalHost, analysis: EvalAnalysis) -> AnalysisLine:
    # One analysis frozen over a bundle this build cannot read, or whose plan no longer resolves, is listed as needing
    # regeneration, never allowed to refuse the listing of every other one.
    needs_regeneration: str | None = None
    try:
        verdict = analysis_verdict(analysis)
    except RefusedInput as unreadable:
        needs_regeneration = f"{unreadable.fault}; {unreadable.remedy}" if unreadable.remedy else unreadable.fault
        verdict = f"Needs regeneration: {needs_regeneration}"
    plan: EvalReportPlan | None = None
    try:
        plan = get_report_plan(host.storage, analysis)
    except NotFoundError as missing:
        needs_regeneration = needs_regeneration or f"{missing.message}; {missing.remedy}"
    except RefusedInput as unreadable:
        # NOSILENT: the stored plan no longer validates; the line names why and the regeneration, and the report and
        # page of this analysis refuse the same way (get_report_plan), so nothing reads it as if it were whole.
        needs_regeneration = needs_regeneration or f"{unreadable.fault}; {unreadable.remedy}"
    if analysis.document is not None:
        findings = len(analysis.document.findings)
    elif analysis.report_doc_id is not None:
        claims, unread = _doc_claims(host, analysis.report_doc_id, analysis.scope_id)
        findings, needs_regeneration = claims, needs_regeneration or unread
    else:
        findings = len(report_plan_of(plan).sections) if plan is not None else 0
    return AnalysisLine(
        id=analysis.id,
        campaign_id=analysis.campaign_id,
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
    says so, never refusing the listing of every other analysis.
    """
    remedy = "generate the analysis again (analysis_generate)"
    try:
        stored = host.storage.load_report_doc(report_doc_id, scope_id)
        if stored is None:
            return 0, f"its report document {report_doc_id!r} was deleted apart from it; {remedy}"
        return len(ReportDoc.model_validate(stored.doc).claims), None
    except (RefusedInput, ValidationError) as unread:
        # NOSILENT: the line names why and the regeneration, and counts no claims, so nothing reads it as whole.
        return 0, f"its report document {report_doc_id!r} does not read under this build ({unread}); {remedy}"


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
    started the generation, and the stored document naming them.
    """

    async def __call__(
        self,
        host: OpsHost,
        *,
        model: TurnModel,
        budget: ReportBudget,
        campaign_id: str,
        limits: WriterLimits,
        tally: Tally,
    ) -> ReportWritten:
        """Freeze the campaign's evidence and write its report, storing what the loop's ending stores.

        :param host: The host the generation runs on.
        :ptype host: OpsHost
        :param model: The ``analyst`` client, as a turn model.
        :ptype model: TurnModel
        :param budget: The report's budget, which admits every turn.
        :ptype budget: ReportBudget
        :param campaign_id: The campaign.
        :ptype campaign_id: str
        :param limits: What the writing loop is held to: the host's ``AnalysisGeneration.limits``, unchanged.
        :ptype limits: WriterLimits
        :param tally: The tally every turn is counted in, which the job reads however the writing ends.
        :ptype tally: Tally
        :return: How it ended, and the stored document when its ending stores one.
        :rtype: ReportWritten
        :raises ReportNotStored: The loop ended with a document to store and the store refused it.
        """
        ...


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
                "return a turn model for the analyst role (threetears.evals.providers.Model is one); a wrapper around "
                "one forwards turn_session and turn_pricing_refusal, and wraps the ToolTurnClient turn_session returns"
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
    ``AnalysisGeneration.limits`` (:func:`~threetears.evals.kernel.inner_ceilings.writer_turns`). No turn is priced here: a turn's price
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
            refusal = await _pricing_refusal(_turn_model(client))
        except ValidationFailedError as refused:
            refusal = str(refused)
        return AnalysisGenerationEstimate(
            campaign_id=campaign_id,
            generator_model=client.model_name,
            first_call_ceiling_usd=None,
            max_calls=writer_turns(generation.limits),
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
            if (unpriced := await _pricing_refusal(turn_model)) is not None:
                raise ValidationFailedError(
                    f"the report was not started, and nothing was frozen, sent or spent: {unpriced}",
                    remedy=(
                        "hand the analyst role a model whose turns the client can price (load its price catalogue, "
                        "give the model a pricer, or name a model it prices)"
                    ),
                )
            return _Prepared(client=client, model=turn_model, budget=report_budget(), limits=generation.limits)
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
        return f"the report budget of {spell_dollars(budget.usd)}"
    # A token and turn budget names no dollars: it is never read as a $0 cap.
    caps = [
        f"{group_thousands(str(cap))} {name}"
        for name in ("tokens", "turns")
        if (cap := getattr(budget, name, None)) is not None
    ]
    return f"the report budget of {' and '.join(caps)}, with no dollar cap"


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
        cost_usd=tally.spent_usd(),
        price_source=", ".join(sources) if sources else None,
        subject_id=campaign.subject_id,
        campaign_id=attempt.campaign_id,
        calls=tally.sent,
        unpriced_calls=tally.unpriced_turns(),
        priced_usd=tally.priced_usd(),
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

    async def ended(
        outcome: AttemptOutcome, error: str, *, refused_by: EvalServiceError | None = None, refusals: Sequence[str] = ()
    ) -> None:
        await _ledgered(eval_host, tally, attempt, campaign, prepared.budget)
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
                written = await writer(
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
    snapshot = await run_blocking(
        eval_host.blocking_executor, eval_host.storage.load_report_snapshot, written.snapshot_id, attempt.scope_id
    )
    if snapshot is None or snapshot.bundle is None or snapshot.bundle_assembled_at is None:
        error = f"the report document {stored.id} is stored, and its snapshot {written.snapshot_id} holds no bundle"
        await ended("failed", error)
        raise StorageError(error)
    kinds = tally.to_json()["total"]["bounce_kinds"]
    try:
        minted = doc_analysis(
            CampaignBundle.from_dict(snapshot.bundle, what=f"report snapshot {snapshot.id}"),
            await snapshot_input(host, attempt.scope_id, written.snapshot_id),
            stored,
            input_digest=snapshot.report_input_digest,
            generator_model=attempt.generator_model,
            skill_digest=attempt.prompt_id,
            bundle_assembled_at=snapshot.bundle_assembled_at,
            token_cost=tally.spent_usd(),
            bounces=kinds,
        )
    except (EvalServiceError, ValueError) as unminted:
        # A stored document whose analysis cannot be built is a defect, never the writer's: recorded, raised as itself.
        await ended(
            "failed", f"the report document {stored.id} is stored, and its analysis could not be built ({unminted})"
        )
        raise
    await _ledgered(eval_host, tally, attempt, campaign, prepared.budget)
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
) -> ReportDocument:
    """The campaign's report, serialized in one form — the same report the command line's ``report`` prints.

    What "the campaign's report" is has one answer,
    :func:`~threetears.evals.analysis.campaign_report`: its newest analysis that is not archived, else a
    code-only report of its evidence. ``basis`` on the result says which, and ``analysis_basis`` what wrote the
    analysis: ``memo`` (one stored before the report planner, laid out as written) or ``plan``, whose report plan is
    ``plan``, as ``report-plan/1`` JSON. A planned analysis's page is the page its plan draws (the report kit's page:
    masthead, headline, lede, confidence and decision, options table and summary charts, the evidence sections, the
    explore part and Full results, Method), with each aspect's fold as a closed appendix after Method; a memo's and a
    code-only page are laid out as before.

    **The trials.** By default each aspect's trials section is the census line, the table of its failed and lost
    trials, each row naming its stored result (read one with ``result_get``), and the table of its passed trials;
    ``trials="all"`` adds each failed or lost trial's full record (what it asked, what the judge saw and said, and
    why), which is most of a large campaign's page. The JSON form carries no trial records either way.

    :param host: The host whose store holds the campaign.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param format: ``markdown``, ``html`` (one page, its charts drawn by the kit's inlined script; see ``static``) or ``json`` (the page's machine form: code's
        readings across the campaign's aspects, the recommendation, the call, and each aspect's bundle by its
        fingerprint, as ``campaign.json`` beside a written page holds it).
    :ptype format: ReportFormat
    :param trials: ``table`` (the default) or ``all``, which adds each failed or lost trial's full record.
    :ptype trials: TrialDetail
    :param static: For ``html``, the static page, for a host whose policy lets a page neither run nor load anything:
        no ``<script>``, no ``<link>``, the system faces, every chart its table of values. Markdown and JSON are
        the same either way.
    :ptype static: bool
    :return: The report in that form.
    :rtype: ReportDocument
    :raises NotFoundError: No campaign with that id in the scope.
    :raises RefusedInput: The campaign's reported analysis was written over a bundle of another shape, or as a report
        plan this build no longer reads: refused with the regeneration that mends it (``analysis_generate``).
    """
    report = campaign_report(host, campaign_id, scope_id, trials=trials)
    analysis = get_analysis(host.storage, report.analysis_id, scope_id) if report.analysis_id is not None else None
    plan = get_report_plan(host.storage, analysis) if analysis is not None else None
    return ReportDocument(
        campaign_id=campaign_id,
        basis=report.basis,
        analysis_id=report.analysis_id,
        analysis_basis=analysis.basis if analysis is not None else None,
        plan=dict(plan.plan) if plan is not None else None,
        format=format,
        body=serialize_report(report, format, static=static),
    )


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
