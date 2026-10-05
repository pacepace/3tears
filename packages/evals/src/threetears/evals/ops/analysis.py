"""The analysis side's operations: campaigns, a generation started as a job, and the report read three ways.

A generation is the analysis side's long work, and it follows the same job contract a launch does
(:mod:`threetears.evals.ops.jobs`): :func:`analysis_generate` checks and builds everything before it
spends — so a refused generation raises to its caller and costs nothing — then starts the paid call as a
background task and returns its job. The job's record is the
:class:`~threetears.evals.contracts.campaign.EvalAnalysisAttempt` the generation writes however it ends.

One generation per campaign runs at a time: a second is refused while the first is live, BEFORE it
prepares, because preparing builds a client and resolves the prompt — work a refused request need not
pay for.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from threetears.evals.analysis.campaigns import create_campaign, list_campaigns
from threetears.evals.analysis.report import Report, ReportBasis, report_html, report_markdown
from threetears.evals.analysis.service import (
    campaign_report,
    get_analysis,
    list_analyses,
    prepare_analysis_generation,
    run_analysis_generation,
)
from threetears.evals.contracts.base import EvalBaseModel, VerbatimText
from threetears.evals.contracts.campaign import EvalCampaign
from threetears.evals.contracts.errors import ConflictError, ValidationFailedError
from threetears.evals.contracts.host import EvalHost
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import JobHandle, JobsStarted, analysis_job_id
from threetears.evals.run.curation import delete_analysis, set_campaign_archived

#: The forms a report is read in: Markdown (the memo, and what an agent reads), its canonical JSON (what
#: the published schema validates) and HTML that reads without any script.
ReportFormat = Literal["markdown", "json", "html"]


class CampaignLine(EvalBaseModel):
    """One campaign, as a listing shows it."""

    id: str
    name: str
    subject_id: str
    behavior: str
    run_count: int
    status: str
    archived: bool


class CampaignListing(EvalBaseModel):
    """A scope's campaigns, newest first."""

    campaigns: list[CampaignLine]


class CampaignDefinition(EvalBaseModel):
    """What creating a campaign names: what it is called, its subject and behaviour, and its runs."""

    name: str
    subject_id: str
    behavior: str
    description: str = ""
    run_ids: list[str] = Field(default_factory=list)


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


def _campaign_line(campaign: EvalCampaign) -> CampaignLine:
    return CampaignLine(
        id=campaign.id,
        name=campaign.name,
        subject_id=campaign.subject_id,
        behavior=campaign.behavior,
        run_count=len(campaign.run_ids),
        status=campaign.status,
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
    """Create a campaign over runs already in the scope.

    Args:
        host: The host whose store the campaign is written to.
        definition: What the campaign is.
        scope_id: The scope it lives in, with its runs.
        created_by: Who is creating it, as the calling surface knows them.

    Returns:
        The campaign as created.

    Raises:
        ValidationFailedError: The definition fails campaign validation, or names a run not in the scope.
        StorageError: The campaign failed to persist.
    """
    campaign = create_campaign(
        host.storage, definition.model_dump(), scope_id=scope_id, created_by=created_by, profile=host.profile
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
        analyses=[
            AnalysisLine(
                id=analysis.id,
                campaign_id=analysis.campaign_id,
                headline=analysis.document.headline,
                generator_model=analysis.generation.generator_model,
                generated_at=analysis.generation.generated_at,
                archived=analysis.archived,
            )
            for analysis in list_analyses(host.storage, campaign_id, scope_id)
        ],
    )


def generation_key(campaign_id: str, scope_id: str) -> str:
    """The exclusivity key one campaign's generations share: one runs at a time."""
    return f"analysis-generation:{scope_id}:{campaign_id}"


async def analysis_generate(host: OpsHost, campaign_id: str, scope_id: str, *, model: str | None = None) -> JobsStarted:
    """Check a campaign's generation, then start its paid call as a background job.

    Args:
        host: The host: its generation settings, its clients and the job manager the task runs under.
        campaign_id: The campaign to analyse.
        scope_id: The scope it lives in.
        model: The generator model, or ``None`` for the host's default.

    Returns:
        The generation's one job.

    Raises:
        ValidationFailedError: The host generates no analyses here, the bundle has no evidence, or the
            prompt does not resolve.
        ConflictError: A generation of this campaign is already running.
        NotFoundError: No campaign with that id in the scope.
    """
    generation = host.generation
    if generation is None:
        raise ValidationFailedError(
            "this host does not generate analyses here: it was mounted without generation settings "
            "(OpsHost.generation), so a generation has no prompt, output cap or budget to run under"
        )
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
        eval_host, campaign_id, scope_id, model=model, resolve_prompt=generation.resolve_prompt
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


def serialize_report(report: Report, format: ReportFormat) -> str:
    """A report in one of its three forms.

    Args:
        report: The report.
        format: ``markdown``, ``json`` (canonical, what the published schema validates) or ``html``.

    Returns:
        The serialized report.
    """
    match format:
        case "markdown":
            return report_markdown(report)
        case "html":
            return report_html(report)
        case "json":
            return report.to_canonical_json()


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


__all__ = [
    "AnalysisDeleted",
    "AnalysisLine",
    "AnalysisListing",
    "CampaignDefinition",
    "CampaignLine",
    "CampaignListing",
    "ReportDocument",
    "ReportFormat",
    "analyses_list",
    "analysis_delete",
    "analysis_generate",
    "campaign_archive",
    "campaign_create",
    "campaigns_list",
    "generation_key",
    "report_read",
    "serialize_report",
]
