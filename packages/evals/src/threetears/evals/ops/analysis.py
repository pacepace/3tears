"""The analysis side's operations: campaigns, a generation started as a job, and the report read three ways.

A generation is the analysis side's long work, and it follows the same job contract a launch does
(:mod:`threetears.evals.ops.jobs`): :func:`analysis_generate` checks and builds everything before it
spends — so a refused generation raises to its caller and costs nothing — then starts the paid call as a
background task and returns its job. The job's record is the
:class:`~threetears.evals.kernel.campaign.EvalAnalysisAttempt` the generation writes however it ends.

One generation per campaign runs at a time: a second is refused while the first is live, BEFORE it
prepares, because preparing builds a client and resolves the prompt — work a refused request need not
pay for.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from threetears.evals.analysis.bundle.assemble import InsightStanding, insight_standing
from threetears.evals.analysis.bar_proposals import propose_bars
from threetears.evals.analysis.campaigns import create_campaign, list_campaigns
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.report import ReportBasis
from threetears.evals.analysis.report.serialize import ReportFormat, serialize_report
from threetears.evals.analysis.service import (
    AnalysisGenerationEstimate,
    campaign_report,
    describe_insight_id_filters,
    estimate_analysis_generation,
    get_analysis,
    list_analyses,
    list_insights,
    prepare_analysis_generation,
    run_analysis_generation,
)
from threetears.evals.schema.base import EvalBaseModel, VerbatimText
from threetears.evals.kernel.campaign import ConfidenceTier, EvalAnalysis, EvalCampaign, EvalInsight
from threetears.evals.kernel.errors import ConflictError, NotFoundError, ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops.host import AnalysisGeneration, OpsHost
from threetears.evals.ops.jobs import JobHandle, JobsStarted, analysis_job_id, generation_key
from threetears.evals.run.curation import delete_analysis, delete_insight, set_analysis_archived, set_campaign_archived


class CampaignLine(EvalBaseModel):
    """One campaign, as a listing shows it."""

    id: str
    name: str
    subject_id: str
    behavior: str
    run_count: int
    archived: bool


class CampaignListing(EvalBaseModel):
    """A scope's campaigns, newest first."""

    campaigns: list[CampaignLine]


#: What a declared design is, said once for the operation and the action that offer it.
DECLARED_DESIGN_DESCRIPTION = (
    "What the campaign sets out to learn, declared before it learns anything: axes (at least one: {axis_id, values: "
    "[{content, display}], rationale?}, each axis_id a lever or open-family member this host declares), held_fixed "
    "({stimulus: controlled|uncontrolled, stimulus_reason (required when uncontrolled), apparatus: "
    "commissioned|witnessed}), and optionally questions ([{id, text, merit_axes?}]), bars ([{measure_id, threshold, "
    "direction}], no looser than the registered ones), merit_priority and intended_repetitions. Validated and gated "
    "as every campaign declaration is. Optional: omitted, the campaign is exploratory — its report and analysis "
    "say once that its readings confirm nothing, and its design is inferred from the runs. Its control is named by control_from_run_id."
)

#: What the control run is, said once for the operation and the action.
CONTROL_RUN_DESCRIPTION = (
    "One of run_ids whose variant becomes the declared control, the cell every other is read against: its variant "
    "key is resolved from the run, as designating a control on an existing campaign resolves it. Requires "
    "declared_design."
)


class CampaignDefinition(EvalBaseModel):
    """What creating a campaign names: what it is called, its subject and behaviour, its runs, and what it set out to learn.

    The declared design is optional: a campaign created without one is exploratory, which its report and analysis
    say once at the top, and its analysis reads a design inferred from the runs, never one presented as declared.
    """

    name: str
    subject_id: str
    behavior: str
    description: str = ""
    run_ids: list[str] = Field(default_factory=list)
    declared_design: dict[str, Any] | None = Field(default=None, description=DECLARED_DESIGN_DESCRIPTION)
    control_from_run_id: str | None = Field(default=None, min_length=1, description=CONTROL_RUN_DESCRIPTION)


class AnalysisLine(EvalBaseModel):
    """One stored analysis, as a listing shows it."""

    id: str
    campaign_id: str
    headline: str
    generator_model: str
    generated_at: str
    archived: bool


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
#: insight (:func:`~threetears.evals.analysis.bundle.insight_standing`): ``live`` — fed to later
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
    )


def campaigns_list(host: EvalHost, scope_id: str, *, archived: bool | None = None) -> CampaignListing:
    """The scope's campaigns.

    Args:
        host: The host whose store is read.
        scope_id: The scope.
        archived: Only archived (``True``) or only active (``False``) campaigns; ``None`` for both.

    Returns:
        The listing.
    """
    return CampaignListing(
        campaigns=[_campaign_line(c) for c in list_campaigns(host.storage, scope_id, archived=archived)]
    )


def campaign_create(host: EvalHost, definition: CampaignDefinition, scope_id: str, *, created_by: str) -> CampaignLine:
    """Create a campaign over runs already in the scope, declared as it is created when the definition says so.

    Args:
        host: The host whose store the campaign is written to.
        definition: What the campaign is.
        scope_id: The scope it lives in, with its runs.
        created_by: Who is creating it, as the calling surface knows them.

    Returns:
        The campaign as created.

    Raises:
        ValidationFailedError: The definition fails campaign validation, names a run not in the scope, declares a
            design this host cannot honour, or names a control run with no design or outside its runs.
        StorageError: The campaign failed to persist.
    """
    campaign = create_campaign(
        host.storage,
        definition.model_dump(exclude={"control_from_run_id"}, exclude_none=True),
        scope_id=scope_id,
        created_by=created_by,
        profile=host.profile,
        control_from_run_id=definition.control_from_run_id,
    )
    return _campaign_line(campaign)


def campaign_archive(host: EvalHost, campaign_id: str, scope_id: str, *, archived: bool) -> CampaignLine:
    """Archive or restore a campaign — retired from listings, nothing destroyed.

    Args:
        host: The host whose store holds the campaign.
        campaign_id: The campaign.
        scope_id: The scope it lives in.
        archived: ``True`` retires it, ``False`` restores it.

    Returns:
        The campaign as persisted.

    Raises:
        NotFoundError: No campaign with that id in the scope.
        StorageError: The write failed.
    """
    return _campaign_line(set_campaign_archived(host.storage, campaign_id, scope_id, archived=archived))


def _analysis_line(analysis: EvalAnalysis) -> AnalysisLine:
    return AnalysisLine(
        id=analysis.id,
        campaign_id=analysis.campaign_id,
        headline=analysis.document.headline,
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

    Args:
        host: The host whose store holds the analysis.
        analysis_id: The analysis.
        scope_id: The scope it lives in.
        archived: ``True`` retires it, ``False`` restores it.
        reason: Why it is archived; cleared on restore.

    Returns:
        The analysis as persisted.

    Raises:
        NotFoundError: No analysis with that id in the scope.
        StorageError: The write failed.
    """
    return _analysis_line(set_analysis_archived(host.storage, analysis_id, scope_id, archived=archived, reason=reason))


def analyses_list(host: EvalHost, campaign_id: str, scope_id: str) -> AnalysisListing:
    """A campaign's stored analyses.

    Args:
        host: The host whose store is read.
        campaign_id: The campaign.
        scope_id: The scope it lives in.

    Returns:
        The listing.
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

    Args:
        host: The host whose store is read.
        scope_id: The scope.

    Returns:
        The listing.
    """
    lines: list[UndescribableArmsLine] = []
    read = 0
    for campaign in list_campaigns(host.storage, scope_id):
        for analysis in list_analyses(host.storage, campaign.id, scope_id):
            read += 1
            # `levels_unavailable` set, even to an empty reason, is the admission; None is a describable arm.
            reasons = [
                entry.levels_unavailable for entry in analysis.variant_index if entry.levels_unavailable is not None
            ]
            if reasons:
                lines.append(
                    UndescribableArmsLine(
                        analysis_id=analysis.id,
                        campaign_id=analysis.campaign_id,
                        archived=analysis.archived,
                        undescribable_arms=len(reasons),
                        arms=len(analysis.variant_index),
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
            "(OpsHost.generation), so a generation has no prompt, output cap or budget to run under"
        )
    return host.generation


async def analysis_estimate(
    host: OpsHost, campaign_id: str, scope_id: str, *, model: str | None = None
) -> AnalysisGenerationEstimate:
    """What a campaign's generation would be priced at, against the cap it would be held to — making no call.

    The start's own assembly and pricing (:func:`analysis_generate` refuses by the same rule), so
    ``would_start`` is its answer.

    Args:
        host: The host: its generation settings, its clients and its out-of-run cap.
        campaign_id: The campaign.
        scope_id: The scope it lives in.
        model: The generator model, or ``None`` for the host's default.

    Returns:
        The estimate.

    Raises:
        ValidationFailedError: The host generates no analyses here, the bundle has no evidence, the
            prompt does not resolve, or the writer model is not one the host allows.
        NotFoundError: No campaign with that id in the scope.
    """
    generation = _generation_settings(host)
    return await estimate_analysis_generation(
        host.eval_host,
        campaign_id,
        scope_id,
        model=model,
        resolve_prompt=generation.resolve_prompt,
        out_of_run_cap_usd=host.out_of_run_cap(),
    )


async def analysis_generate(host: OpsHost, campaign_id: str, scope_id: str, *, model: str | None = None) -> JobsStarted:
    """Check a campaign's generation, price its first call against the host's out-of-run cap, then start it as a job.

    Every call the generation makes is priced before it is sent against the host's out-of-run cap
    (``LaunchSettings.max_out_of_run_cost_usd``, when enforcement is on) and ledgered under purpose
    ``analysis``, so :func:`~threetears.evals.ops.scope_out_of_run_spend` reads it beside case generations
    and rubric proposals. The first call is priced here, so one over the cap is refused to the caller with
    nothing spent; the repair round-trip a refused output buys is priced when its prompt exists, against
    what is left of the same cap.

    Args:
        host: The host: its generation settings, its clients and the job manager the task runs under.
        campaign_id: The campaign to analyse.
        scope_id: The scope it lives in.
        model: The generator model, or ``None`` for the host's default.

    Returns:
        The generation's one job.

    Raises:
        ValidationFailedError: The host generates no analyses here, the bundle has no evidence, the
            prompt does not resolve, the writer model is not one the host allows
            (``HostProfile.analysis_writer_models``, checked before any provider request), or the first call cannot be priced under the enforced cap or is priced
            above it.
        ConflictError: A generation of this campaign is already running.
        NotFoundError: No campaign with that id in the scope.
    """
    generation = _generation_settings(host)
    manager = host.launch.job_manager
    key = generation_key(campaign_id, scope_id)
    running = manager.active_task_ids(key)
    if running:
        raise ConflictError(
            f"a generation of campaign '{campaign_id}' is already running "
            f"(job {analysis_job_id(campaign_id, running[0])}); poll it rather than starting another"
        )
    eval_host = host.eval_host
    prepared = await prepare_analysis_generation(
        eval_host,
        campaign_id,
        scope_id,
        model=model,
        resolve_prompt=generation.resolve_prompt,
        out_of_run_cap_usd=host.out_of_run_cap(),
    )

    async def work() -> None:
        await run_analysis_generation(
            eval_host, prepared, prompt_id=generation.prompt_id, max_output_tokens=generation.max_output_tokens
        )

    try:
        manager.start_task(prepared.attempt_id, work, budget_s=generation.budget_s, key=key)
    except ConflictError:
        # Another start won the race while this one prepared: release the client it built, which only
        # the generation itself would otherwise have entered and released.
        async with prepared.client:
            pass
        raise
    return JobsStarted(
        jobs=[
            JobHandle(
                job_id=analysis_job_id(campaign_id, prepared.attempt_id),
                kind="analysis",
                target_id=campaign_id,
                label=prepared.resolved_model,
            )
        ]
    )


def report_read(host: EvalHost, campaign_id: str, scope_id: str, *, format: ReportFormat) -> ReportDocument:
    """The campaign's report, serialized in one form — the same report the command line's ``report`` prints.

    What "the campaign's report" is has one answer,
    :func:`~threetears.evals.analysis.campaign_report`: its newest analysis that is not archived, else a
    code-only report of its evidence. ``basis`` on the result says which.

    Args:
        host: The host whose store holds the campaign.
        campaign_id: The campaign.
        scope_id: The scope it lives in.
        format: ``markdown``, ``json`` or ``html``.

    Returns:
        The report in that form.

    Raises:
        NotFoundError: No campaign with that id in the scope.
    """
    report = campaign_report(host, campaign_id, scope_id)
    return ReportDocument(
        campaign_id=campaign_id,
        basis=report.basis,
        analysis_id=report.source.analysis_id,
        format=format,
        body=serialize_report(report, format),
    )


class ProposedBar(EvalBaseModel):
    """One bar a baseline proposes, for a person to adopt, tighten or leave.

    Attributes:
        measure: The measure the bar is on, a host-declared one.
        threshold: The proposed threshold, in the measure's unit.
        higher_is_better: Which side of ``threshold`` clears it.
        rationale: How the threshold was seeded from the incumbent's measurement.
        vacuous: Whether nothing could fail it — adopting it as written registers no standard.
        vacuous_reason: Why it is vacuous; ``None`` when it discriminates.
    """

    measure: str
    threshold: float
    higher_is_better: bool
    rationale: str
    vacuous: bool
    vacuous_reason: str | None


class BarProposals(EvalBaseModel):
    """What a baseline campaign proposes as its behavior's bars. Read-only: nothing here is registered.

    Attributes:
        campaign_id: The baseline campaign measured.
        behavior: The behavior every proposal governs, the campaign's.
        variant_key: The incumbent: the baseline's one cell.
        proposals: One per declared measure with a better end that could be seeded, in measure-name order.
        not_proposed: ``{reading: why}`` for every reading the cell carries that no bar could be proposed on.
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

    Args:
        host: The host whose measures, bars and store this reads.
        campaign_id: The baseline campaign.
        scope_id: The scope it lives in.

    Returns:
        The proposals.

    Raises:
        NotFoundError: No campaign with that id in the scope.
        ValidationFailedError: The campaign measured no cell or more than one: a baseline is one configuration
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

    Args:
        host: The host whose store holds the analysis.
        analysis_id: The analysis.
        scope_id: The scope it lives in.
        confirm: Must echo ``analysis_id``.

    Returns:
        What was removed.

    Raises:
        NotFoundError: No analysis with that id in the scope — refused before ``confirm`` is read.
        ValidationFailedError: ``confirm`` does not echo the id.
        StorageError: The delete failed.
    """
    removed = delete_analysis(host.storage, get_analysis(host.storage, analysis_id, scope_id), confirm=confirm)
    return AnalysisDeleted(analysis_id=removed["analysis_id"], campaign_id=removed["campaign_id"])


def insights_list(
    host: EvalHost, scope_id: str, *, subject_id: str | None = None, source_campaign_id: str | None = None
) -> InsightListing:
    """The scope's insight ledger, newest observation first, each with where it stands.

    Over :func:`~threetears.evals.analysis.list_insights`: both filters match ids exactly, and an id that
    matches nothing is an answer (that campaign minted no insights), not a refusal.

    Args:
        host: The host whose store holds the ledger.
        scope_id: The scope whose ledger is read.
        subject_id: Only insights about this subject.
        source_campaign_id: Only insights an analysis of this campaign minted.

    Returns:
        The listing.
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

    Args:
        host: The host whose store holds the ledger.
        insight_id: The insight.
        scope_id: The scope it lives in.

    Returns:
        The insight and its standing.

    Raises:
        NotFoundError: No insight with that id in the scope.
    """
    insight = host.storage.load_insight(insight_id, scope_id)
    if insight is None:
        raise NotFoundError("insight", insight_id)
    return InsightDetail(insight=insight, standing=_standing_of(insight, _insight_standing(host, [insight], scope_id)))


def insight_delete(host: EvalHost, insight_id: str, scope_id: str, *, confirm: str | None) -> InsightDeleted:
    """Destroy one insight — the intended answer to a wrong one, since an insight has no archive.

    A live insight is fed to every later generation over its subject as prior context, so a wrong one keeps
    steering analyses until it is gone (:func:`~threetears.evals.run.delete_insight`). Archiving its analysis
    retracts every insight that analysis minted; this removes one.

    Args:
        host: The host whose store holds the ledger.
        insight_id: The insight.
        scope_id: The scope it lives in.
        confirm: Must echo ``insight_id``.

    Returns:
        What was removed.

    Raises:
        NotFoundError: No insight with that id in the scope — refused before ``confirm`` is read.
        ValidationFailedError: ``confirm`` does not echo the id.
        StorageError: The delete failed.
    """
    removed = delete_insight(host.storage, insight_id, scope_id, confirm=confirm)
    return InsightDeleted(insight_id=removed["insight_id"], source_campaign_id=removed["source_campaign_id"])


__all__ = [
    "AnalysisDeleted",
    "AnalysisGenerationEstimate",
    "AnalysisLine",
    "AnalysisListing",
    "BarProposals",
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
    "campaigns_list",
    "insight_delete",
    "insight_get",
    "insights_list",
    "report_read",
]
