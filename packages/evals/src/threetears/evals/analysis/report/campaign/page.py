"""Build a campaign's page: the writer's summary across its aspects, the recommendation, what code read on each
aspect, the person's call, then one fold per aspect.

Code reports, the writer recommends, a person decides, and the page keeps the three apart. What the first screen
holds depends on the campaign's kind (:func:`~threetears.evals.analysis.campaign_readings.campaign_kind`). For every
kind it opens, above the title, with **every bar an arm misses**, then **every broken floor**, then **every guardrail
an arm is not shown safe on** (:mod:`.warnings`). Then **the question** and each criterion set after the readings it
judges were visible (:mod:`.provenance`). Then **the writer's summary** and, for a change or a choice, **the writer's
recommendation** (:mod:`.memo_blocks`). Then code's readings by the campaign's kind (:mod:`.scorecard`), the figures
across aspects (:mod:`.aggregate`), how deep each arm ran (:mod:`.depth_blocks`), every kind's bars, and the person's
call (a measurement's only once one is recorded).

Then the memo's findings and next steps, and one closed fold per aspect: the writer's summary of the aspect (what
changed, how sure, what it means) when there is one, then the aspect's own page as
:func:`~threetears.evals.analysis.report.build.build_code_only_report` built it, which the serializers lay out in the
fold without its title, its question, its late criteria, its broken floors or its verdict line.

Every arm is named by its label, never by its role; every line code writes is held to
:mod:`~threetears.evals.analysis.style`, and the memo's words are printed as written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.depth_reading import PAIRING_METHOD
from threetears.evals.analysis.report.campaign.aggregate import aggregate_blocks
from threetears.evals.analysis.report.campaign.depth_blocks import depth_blocks
from threetears.evals.analysis.report.campaign.memo_blocks import (
    aspect_summary_blocks,
    behind_on_blocks,
    call_blocks,
    memo_blocks,
    memo_summary_blocks,
    recommendation_blocks,
)
from threetears.evals.analysis.report.campaign.provenance import question_blocks, timing_blocks
from threetears.evals.analysis.report.campaign.scorecard import (
    aspect_shows,
    bar_blocks,
    frontier_blocks,
    measure_blocks,
    pair_blocks,
    settle_blocks,
    table_blocks,
)
from threetears.evals.analysis.report.campaign.warnings import bar_warnings, floor_warnings, guardrail_warnings
from threetears.evals.analysis.report.model import (
    CampaignAspectFold,
    CampaignReport,
    DisclosureBlock,
    Report,
    ReportBlock,
)
from threetears.evals.kernel.authored import AuthoredCampaignAnalysis
from threetears.evals.kernel.campaign import EvalCampaign, FindingResolution
from threetears.evals.kernel.declaration import CampaignCall
from threetears.evals.kernel.errors import RefusedInput

__all__ = ["build_campaign_report"]


def build_campaign_report(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    aspect_reports: Mapping[str, Report],
    memo: AuthoredCampaignAnalysis | None,
    *,
    call: CampaignCall | None = None,
    recommended_variant_key: str | None = None,
    resolutions: Sequence[FindingResolution] = (),
    trial_ids: Mapping[str, tuple[str, str]] | None = None,
) -> CampaignReport:
    """Lay a campaign out as its page: code's warnings first, the writer's summary and recommendation, code's readings
    on each aspect, the person's call, the findings, then one fold per aspect.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it (:func:`~threetears.evals.analysis.campaign_readings.campaign_readings`).
    :ptype readings: CampaignReadings
    :param aspect_reports: Each aspect's own page, keyed by its label: one per aspect, each built over the bundle its
        readings were read from (:func:`~threetears.evals.analysis.report.build.build_code_only_report`).
    :ptype aspect_reports: Mapping[str, Report]
    :param memo: The writer's memo over the campaign; None where none was written.
    :ptype memo: AuthoredCampaignAnalysis | None
    :param call: The person's call; None while none is recorded.
    :ptype call: CampaignCall | None
    :param recommended_variant_key: The variant key of the arm the memo recommends, as the analysis recorded it
        (``EvalAnalysis.recommended_variant_key``); required where the page holds a recommendation, which a
        measurement's never does.
    :ptype recommended_variant_key: str | None
    :param resolutions: What code filled for each of the memo's findings, by position (``EvalAnalysis.resolutions``),
        which each finding's trace reads the readings it cites off; empty traces only the trials each names.
    :ptype resolutions: Sequence[FindingResolution]
    :param trial_ids: The ids the memo names trials by, each to its aspect and its id in that aspect's bundle
        (:meth:`~threetears.evals.analysis.campaign_bundle.CampaignBundle.trial_ids`); None traces no trial.
    :ptype trial_ids: Mapping[str, tuple[str, str]] | None
    :return: The page, ``basis="analysis"`` with a memo, ``code_only`` without one.
    :rtype: CampaignReport
    :raises RefusedInput: An aspect has no page, a page is keyed by a label the campaign does not hold, the readings
        are of another campaign, or the memo recommends an arm and no key is given for it.
    """
    reports = _checked_reports(campaign, readings, aspect_reports)
    recommendation = memo.recommended() if memo is not None and readings.kind != "measurement" else None
    if recommendation is not None and recommended_variant_key is None:
        raise RefusedInput(
            f"the memo recommends {recommendation.arm}, and the page is given no variant key for that arm",
            remedy="pass the key the analysis recorded for its recommended arm (EvalAnalysis.recommended_variant_key)",
        )
    kind = readings.kind
    bundles = [reports[aspect.label].bundle for aspect in campaign.aspects]
    blocks: list[ReportBlock] = [
        *bar_warnings(readings, memo),
        *floor_warnings(campaign, readings, memo),
        *guardrail_warnings(campaign, readings, reports),
        *question_blocks(campaign),
        *timing_blocks(bundles),
        # The writer's read across every aspect, then what it recommends, then the aspects code read them on.
        *memo_summary_blocks(memo),
    ]
    if kind != "measurement":
        blocks += recommendation_blocks(memo, kind)
    if kind == "change":
        blocks += table_blocks(campaign, readings)
        blocks += settle_blocks(readings)
    elif kind == "choice":
        blocks += pair_blocks(campaign, readings, reports)
        blocks += behind_on_blocks(memo)
        blocks += frontier_blocks(campaign, readings, reports)
    else:
        blocks += measure_blocks(campaign, readings, reports)
    blocks += aggregate_blocks(campaign, readings, reports)
    blocks += depth_blocks(campaign, readings, reports)
    blocks += bar_blocks(readings, reports)
    if kind != "measurement" or call is not None:
        blocks += call_blocks(campaign, readings, call)
    blocks += memo_blocks(memo, campaign, readings, reports, resolutions, trial_ids or {})
    summaries = {summary.aspect: summary for summary in (memo.aspect_summaries if memo is not None else [])}
    folds = [
        CampaignAspectFold(
            aspect=aspect.label,
            shows=aspect_shows(readings, aspect.label),
            summary=aspect_summary_blocks(summaries.get(aspect.label)),
            report=reports[aspect.label],
        )
        for aspect in campaign.aspects
    ]
    if readings.readings or readings.pairs:
        blocks.append(DisclosureBlock(section="methods", source="comparisons", text=PAIRING_METHOD))
    return CampaignReport(
        basis="analysis" if memo is not None else "code_only",
        title=campaign.name or f"Campaign {campaign.id}",
        campaign_id=campaign.id,
        campaign_name=campaign.name or None,
        scope_id=campaign.scope_id,
        finding_count=len(memo.findings) if memo is not None else 0,
        blocks=blocks,
        aspects=folds,
        readings=readings,
        call=call,
        recommendation=recommendation,
        recommended_variant_key=recommended_variant_key if recommendation is not None else None,
    )


def _checked_reports(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> dict[str, Report]:
    """Every aspect's page, refused unless there is exactly one per aspect and the readings are this campaign's."""
    if readings.campaign_id != campaign.id:
        raise RefusedInput(
            f"the readings are of campaign {readings.campaign_id}, not {campaign.id}",
            remedy="pass the readings read off this campaign, over its own aspects' bundles",
        )
    labels = [aspect.label for aspect in campaign.aspects]
    if missing := [label for label in labels if label not in reports]:
        raise RefusedInput(
            f"campaign {campaign.id} has no page for aspect(s) {', '.join(missing)}",
            remedy="pass one page per aspect, each built over that aspect's bundle and keyed by its label",
        )
    if stray := sorted(set(reports) - set(labels)):
        raise RefusedInput(
            f"campaign {campaign.id} holds no aspect labelled {', '.join(stray)}",
            remedy="key each page by the label of the aspect it reads",
        )
    return {label: reports[label] for label in labels}
