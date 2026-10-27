"""The analysis side's operations: campaigns, a generation started as a job, and the report read three ways.

A generation is the analysis side's long work, and it follows the same job contract a launch does
(:mod:`threetears.evals.ops.jobs`): :func:`analysis_generate` checks only what its caller can fix cheaply (the
campaign, the named writer, the slot), records a running attempt and starts a background task, returning its job at
once. The task prepares — assembles the bundle, resolves the prompt, builds the client, prices the first call — and
only then spends; a refusal it meets there ends the attempt ``failed`` with nothing spent, and ``job_poll`` reads it.
The job's record is the :class:`~threetears.evals.kernel.campaign.EvalAnalysisAttempt` the generation writes however
it ends.

One generation per campaign runs at a time: a second is refused while the first is live, BEFORE it
prepares, because preparing builds a client and resolves the prompt — work a refused request need not
pay for.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from threetears.evals.analysis.bundle.insights import InsightStanding, insight_standing
from threetears.evals.analysis.bar_proposals import propose_bars
from threetears.evals.analysis.campaigns import (
    add_runs_to_campaign,
    create_campaign,
    list_campaigns,
    remove_runs_from_campaign,
    update_campaign,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.report import ReportBasis, TrialDetail, analysis_verdict
from threetears.evals.analysis.report.serialize import ReportFormat, serialize_report
from threetears.evals.analysis.service import (
    AnalysisGenerationEstimate,
    campaign_report,
    describe_insight_id_filters,
    estimate_analysis_generation,
    get_analysis,
    list_analyses,
    list_insights,
    PreparedGeneration,
    record_campaign_call,
    prepare_analysis_generation,
    assembled_analysis_attempt,
    run_analysis_generation,
    running_analysis_attempt,
)
from threetears.evals.schema.base import EvalBaseModel, VerbatimText
from threetears.evals.kernel.declaration import CampaignCall, Margin, OverriddenBar
from threetears.evals.kernel.campaign import AttemptOutcome, ConfidenceTier, EvalAnalysis, EvalCampaign, EvalInsight
from threetears.evals.kernel.errors import (
    ConflictError,
    EvalServiceError,
    JobFenced,
    NotFoundError,
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
from threetears.evals.kernel.job_timeout import EvalJobTimeout
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.refusals import job_refusal
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops.host import AnalysisGeneration, OpsHost
from threetears.evals.ops.jobs import JobHandle, JobsStarted, analysis_job_id, generation_key
from threetears.evals.run.curation import delete_analysis, delete_insight, set_analysis_archived, set_campaign_archived
from threetears.evals.schema.models import DECLARED_NAME_CHARS, utc_now_iso
from threetears.evals.schema.logs import contain, get_logger
from threetears.evals.kernel.remedies import FIND_CAMPAIGN, FIND_INSIGHT, FIND_RUN, WRITER_REMEDY

if TYPE_CHECKING:
    from threetears.evals.analysis.campaign_bundle import CampaignBundle

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
    "The arms, each {label, role: control|candidate (default candidate), from_run_ids: [run id, ...]}: a contestant "
    "stack, read from the first of its runs, each a run one of the campaign's aspects holds, all carrying one variant "
    "key. One control at most, read first. Omitted, an adopted campaign's arms are carried and every other variant "
    "among the runs is read as a candidate."
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
    "its short_name. Two aspects under one label or printed alike, or one run in two of them, are refused."
)

#: What naming a campaign's aspects is, said once.
SHORT_NAMES_DESCRIPTION = (
    "An aspect's label to its short name, what a page prints for it in table cells and sentences (at most "
    f"{DECLARED_NAME_CHARS} characters, no other aspect's label or short name), or to null to print its label whole. An aspect "
    "not named keeps its own. Editable once runs have joined, since it changes nothing any run measures; refused "
    "beside aspects, whose own short_name says it."
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
    exploratory: bool = Field(default=False, description=EXPLORATORY_DESCRIPTION)
    control_from_run_id: str | None = Field(default=None, min_length=1, description=CONTROL_RUN_DESCRIPTION)


class AnalysisLine(EvalBaseModel):
    """One stored analysis, as a listing shows it."""

    id: str
    campaign_id: str
    verdict: str = Field(
        description=(
            "The report's verdict line over the evidence the analysis was written from, as code writes it — never a "
            "headline the memo wrote. It leads, as the report does, with every bar an arm misses and then every floor "
            "a candidate breaks, each in the report's own words; with neither, it is the verdict alone."
        )
    )
    findings: int = Field(ge=0, description="How many findings the memo holds.")
    recommendations: int = Field(
        ge=0,
        le=1,
        description="How many recommendations it makes: one, or none for a measurement or a choice with no clear winner.",
    )
    generator_model: str
    generated_at: str
    archived: bool
    needs_regeneration: str | None = Field(
        default=None,
        description=(
            "Why this build cannot read the evidence the analysis was frozen over, and what to do: set when it was "
            "written over a bundle of another shape, so its verdict, report and page are refused until it is generated "
            "again (analysis_generate). None when it reads."
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
    format: ReportFormat
    body: VerbatimText = Field(
        description="The report in that form, exactly as its serializer wrote it: Markdown, canonical JSON, or "
        "script-free HTML."
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


def _analysis_line(analysis: EvalAnalysis) -> AnalysisLine:
    # One analysis frozen over a bundle this build cannot read is listed as needing regeneration, never allowed to
    # refuse the listing of every other one.
    needs_regeneration: str | None = None
    try:
        verdict = analysis_verdict(analysis)
    except RefusedInput as unreadable:
        needs_regeneration = f"{unreadable.fault}; {unreadable.remedy}" if unreadable.remedy else unreadable.fault
        verdict = f"Needs regeneration: {needs_regeneration}"
    return AnalysisLine(
        id=analysis.id,
        campaign_id=analysis.campaign_id,
        verdict=verdict,
        needs_regeneration=needs_regeneration,
        findings=len(analysis.document.findings),
        recommendations=0 if analysis.document.recommended() is None else 1,
        generator_model=analysis.generation.generator_model,
        generated_at=analysis.generation.generated_at,
        archived=analysis.archived,
    )


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
    return _analysis_line(set_analysis_archived(host.storage, analysis_id, scope_id, archived=archived, reason=reason))


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
        analyses=[_analysis_line(analysis) for analysis in list_analyses(host.storage, campaign_id, scope_id)],
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
            "(OpsHost.generation), so a generation has no prompt, output cap or budget to run under",
            remedy=(
                "read the campaign's code-only report with report_read; a generation needs the host's operator to "
                "mount it with OpsHost.generation"
            ),
        )
    return host.generation


async def analysis_estimate(
    host: OpsHost, campaign_id: str, scope_id: str, *, model: str | None = None
) -> AnalysisGenerationEstimate:
    """What a campaign's generation would be priced at, against the cap it would be held to — making no call.

    The same assembly and pricing a generation's job makes before its first call (:func:`analysis_generate`'s job
    ends refused by the same rule), so ``would_start`` is its answer. Unlike the start, it answers in its call: it
    must return a number, and the bundle it assembles is read in a fixed number of store calls whatever the
    campaign's size.

    :param host: The host: its generation settings, its clients and its out-of-run cap.
    :ptype host: OpsHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param model: The generator model, or ``None`` for the host's default.
    :ptype model: str | None
    :return: The estimate.
    :rtype: AnalysisGenerationEstimate
    :raises ValidationFailedError: The host generates no analyses here, the bundle has no evidence, the
        prompt does not resolve, or the writer model is not one the host allows.
    :raises NotFoundError: No campaign with that id in the scope.
    """
    generation = _generation_settings(host)
    return await estimate_analysis_generation(
        host.eval_host,
        campaign_id,
        scope_id,
        model=model,
        resolve_prompt=generation.prompt,
        out_of_run_cap_usd=host.out_of_run_cap(),
    )


async def analysis_generate(host: OpsHost, campaign_id: str, scope_id: str, *, model: str | None = None) -> JobsStarted:
    """Start a campaign's generation as a job, after the checks that cost a read or two; the job prepares and prices it.

    The call checks only what is cheap — the host generates analyses here, the campaign exists, no generation of it
    is running, and a writer the call names is one the host allows — then records the attempt ``running``, carrying
    its lease, and returns its job. Everything else is the job's, because assembling the bundle reads every member
    run's results, which for a large campaign outlasts a host's call budget: it assembles the bundle, resolves the
    prompt, builds the client, holds the host's default writer to its allowed list, and prices the first call. A
    refusal at any of those steps ends the attempt ``failed`` with ``refused before any call: <why>`` and the refusal's
    kind (``job_poll`` reads it), releases the client it built, and makes no call.

    Every call the generation makes is priced before it is sent against the host's out-of-run cap
    (``LaunchSettings.max_out_of_run_cost_usd``, when enforcement is on) and ledgered under purpose
    ``analysis``, so :func:`~threetears.evals.ops.scope_out_of_run_spend` reads it beside case generations
    and rubric proposals. The first call is priced in the job before it is sent, so one over the cap ends the job
    refused with nothing spent (:func:`analysis_estimate` asks the same question in its call); the repair round-trip
    a refused output buys is priced when its prompt exists, against what is left of the same cap.

    :param host: The host: its generation settings, its clients and the job manager the task runs under.
    :ptype host: OpsHost
    :param campaign_id: The campaign to analyse.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param model: The generator model, or ``None`` for the host's default.
    :ptype model: str | None
    :return: The generation's one job. Poll it: a refusal its preparation met (the bundle has no evidence, the prompt
        does not resolve, the host's default writer is not one it allows, the first call cannot be priced under the
        enforced cap or is priced above it) ends the job refused rather than raising here.
    :rtype: JobsStarted
    :raises ValidationFailedError: The host generates no analyses here, or the named writer model is not one the
        host allows (``HostProfile.analysis_writer_models``).
    :raises ConflictError: A generation of this campaign is already running — in this process, or in any process:
        the campaign's generation slot names an attempt whose lease is live. The refusal names that attempt's
        job.
    :raises NotFoundError: No campaign with that id in the scope.
    """
    generation = _generation_settings(host)
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
    await run_blocking(executor, _refuse_unstartable, eval_host, campaign_id, scope_id, model)

    started = running_analysis_attempt(
        attempt_id=str(uuid.uuid7()),
        campaign_id=campaign_id,
        scope_id=scope_id,
        model=model,
        prompt_id=generation.prompt_id,
    )
    attempt_id = started.id
    # What the attempt reads as it ends without a call: the running record, and once the job has assembled the bundle
    # and built the client, the writer and fingerprint they resolved.
    current = started
    leases = host.launch.job_leases

    def save_started() -> None:
        # Written carrying its lease before the job starts, so its first poll — in any process — finds it running.
        started.lease = leases.grant()
        eval_host.storage.save_analysis_attempt(started)

    await run_blocking(executor, save_started)

    async def end_unrun(why: str, *, outcome: AttemptOutcome = "failed", refused_by: Exception | None = None) -> None:
        # The generation goes no further and made no call: end the attempt it recorded, so none is left reading
        # `running` that nothing runs. Every path that stops between the attempt's first write and its first call comes
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

    async def assembled(bundle: CampaignBundle, resolved_model: str) -> None:
        # What only the preparation can say goes on the running attempt before the first call is priced, so a price
        # that refuses it leaves an attempt naming its writer and bundle. The ending writes both again, so a write
        # that fails here loses nothing; one that finds the attempt ended stops the generation.
        nonlocal current
        current = assembled_analysis_attempt(started, bundle, resolved_model)
        try:
            await run_blocking(executor, write_held_record, eval_host.storage, ANALYSIS_JOB, current, leases=leases)
        except StorageError, ConflictError:
            # NOSILENT: the ending rewrites both values; until then the running attempt reads as it was started
            log.warning(
                "eval.generate_analysis attempt %s campaign=%s: its resolved writer could not be recorded before its "
                "first call; the ending records it",
                attempt_id,
                campaign_id,
            )

    async def prepare() -> PreparedGeneration | None:
        # Assembly, the prompt, the client, the host's default writer and the first call's price: every refusal ends
        # the attempt with no call made, and the preparation has released any client it built. A refusal is a designed
        # ending, not a task failure: the attempt records it and this returns None, so the manager logs no error for
        # it. What nobody designed for (a store that raised) is recorded the same way and raised on as itself.
        try:
            async with manager.job_timeout_factory(generation.preparation_budget_s):
                return await prepare_analysis_generation(
                    eval_host,
                    campaign_id,
                    scope_id,
                    model=model,
                    resolve_prompt=generation.prompt,
                    out_of_run_cap_usd=host.out_of_run_cap(),
                    attempt_id=attempt_id,
                    on_assembled=assembled,
                    started_at=started.started_at,
                )
        except JobFenced:
            raise
        except asyncio.CancelledError:
            await end_unrun("cancelled before any call, while the generation was being prepared", outcome="cancelled")
            raise
        except EvalJobTimeout as timed_out:
            await end_unrun(f"preparation did not finish in {timed_out.budget_s:g} s")
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
            await run_analysis_generation(
                eval_host,
                prepared,
                prompt_id=generation.prompt_id,
                max_output_tokens=generation.max_output_tokens,
                deadline=manager.job_timeout_factory(generation.budget_s),
                leases=leases,
            )
        except JobFenced as fenced:
            # NOSILENT: the attempt's record, written lost elsewhere, is the answer every poll reads; nothing here
            # wrote after it.
            # Another process wrote the attempt lost (its lease lapsed): its record is the answer, and the
            # generation has stored nothing more.
            log.warning("eval.generate_analysis campaign=%s scope=%s stopped: %s", campaign_id, scope_id, fenced)

    try:
        # The generation enters its own budget (``deadline``), so its attempt can say it timed out: one entered
        # around the task would reach the generation as a cancel.
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


def _refuse_unstartable(host: EvalHost, campaign_id: str, scope_id: str, model: str | None) -> None:
    """Refuse a start whose campaign does not resolve, or that names a writer the host does not allow — one read.

    The checks a start makes in its call. Everything that reads the campaign's runs is its job's.

    :raises NotFoundError: No campaign with that id in the scope.
    :raises ValidationFailedError: ``model`` is not one the host allows.
    """
    if host.storage.load_campaign(campaign_id, scope_id) is None:
        raise NotFoundError("campaign", campaign_id, remedy=FIND_CAMPAIGN)
    if model is not None and (ineligible := host.profile.analysis_writer_refusal(model)) is not None:
        raise ValidationFailedError(ineligible, remedy=WRITER_REMEDY)


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
    host: EvalHost, campaign_id: str, scope_id: str, *, format: ReportFormat, trials: TrialDetail = "table"
) -> ReportDocument:
    """The campaign's report, serialized in one form — the same report the command line's ``report`` prints.

    What "the campaign's report" is has one answer,
    :func:`~threetears.evals.analysis.campaign_report`: its newest analysis that is not archived, else a
    code-only report of its evidence. ``basis`` on the result says which.

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
    :param format: ``markdown``, ``html`` (one page that needs no script) or ``json`` (the page's machine form: code's
        readings across the campaign's aspects, the recommendation, the call, and each aspect's bundle by its
        fingerprint, as ``campaign.json`` beside a written page holds it).
    :ptype format: ReportFormat
    :param trials: ``table`` (the default) or ``all``, which adds each failed or lost trial's full record.
    :ptype trials: TrialDetail
    :return: The report in that form.
    :rtype: ReportDocument
    :raises NotFoundError: No campaign with that id in the scope.
    """
    report = campaign_report(host, campaign_id, scope_id, trials=trials)
    return ReportDocument(
        campaign_id=campaign_id,
        basis=report.basis,
        analysis_id=report.analysis_id,
        format=format,
        body=serialize_report(report, format),
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
