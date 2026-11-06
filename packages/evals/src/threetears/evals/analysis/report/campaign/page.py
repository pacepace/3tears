"""Build a campaign's page: the decision first, then what code read across its aspects, how the campaign was set up,
then one fold per aspect.

Code reports, the writer recommends, a person decides, and the page keeps the three apart. This page is code's: the
writer's report is a report document, which its own page draws (:mod:`threetears.evals.analysis.report.doc_page`).
The first screen is the decision. Under the title, for every kind: **the question**, or that none was asked
(:mod:`.provenance`); where a guardrail warns, **the lead** under it: one sentence per candidate on the first measure
the aspect's design lists in its graders' order, the one an eval's question asks and its printed line leads on; with no
order listed, on the question's measure, the aspect's pass rate, or with no question asked on the classifier's
accuracy; so the guardrail's warning sits beside the lead and never in its place (a page no guardrail warns on leads on
its scorecard); one line on where the campaign's
report is: the report document it is stored as, the analysis this version no longer reads or a retired writer wrote,
or that no recommendation was written (:mod:`.answer_blocks`); then **what warns**, one line per arm for the bars it
misses, the floors it breaks and the guardrails it is not shown safe on (:mod:`.warnings`), and each report-wide caveat
once (:mod:`.provenance`); then the person's call (a measurement's only once one is recorded).

Then, under **Scorecard**, code's readings by the campaign's kind (:mod:`.scorecard`), the figures across aspects
(:mod:`.aggregate`), how deep each arm ran (:mod:`.depth_blocks`) and every kind's bars; then **Design &
provenance**; then one closed fold per aspect: the aspect's own page as
:func:`~threetears.evals.analysis.report.build.build_code_only_report` built it, which the serializers lay out in the
fold without its title, its question, its broken floors or its verdict line, and without what the campaign states
once for every aspect (:func:`~.provenance.for_the_campaign`).

Every arm is named by its label, never by its role; every line code writes is held to
:mod:`~threetears.evals.analysis.style`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.campaign_title import declared_title, named_title
from threetears.evals.analysis.depth_reading import PAIRING_METHOD
from threetears.evals.analysis.report.campaign.aggregate import aggregate_blocks
from threetears.evals.analysis.report.campaign.depth_blocks import depth_blocks, depth_provenance_blocks
from threetears.evals.analysis.report.campaign.answer_blocks import (
    NO_LONGER_READS,
    NO_RECOMMENDATION,
    call_blocks,
    report_doc_blocks,
    unread_analysis_blocks,
)
from threetears.evals.analysis.report.campaign.names import AspectNames, check_legend_blocks
from threetears.evals.analysis.report.campaign.provenance import (
    caveat_blocks,
    for_the_campaign,
    provenance_blocks,
    question_blocks,
)
from threetears.evals.analysis.report.campaign.scorecard import (
    READING_WORDS,
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
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.analysis.quantities import is_money
from threetears.evals.kernel.declaration import Aspect, CampaignArm, CampaignCall
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.metrics import ACCURACY_MEASURE, METRIC_DESCRIPTORS

__all__ = ["build_campaign_report"]


def build_campaign_report(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    aspect_reports: Mapping[str, Report],
    *,
    call: CampaignCall | None = None,
    report_doc_id: str | None = None,
    unread_analysis_id: str | None = None,
    unread_report_doc_id: str | None = None,
    unread_line: str = NO_LONGER_READS,
    arms: Sequence[CampaignArm] | None = None,
) -> CampaignReport:
    """Lay a campaign out as code's page: under the title the question, where its report is, what warns and the
    person's call; then the Scorecard, Design & provenance, then one fold per aspect.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it (:func:`~threetears.evals.analysis.campaign_readings.campaign_readings`).
    :ptype readings: CampaignReadings
    :param aspect_reports: Each aspect's own page, keyed by its label: one per aspect, each built over the bundle its
        readings were read from (:func:`~threetears.evals.analysis.report.build.build_code_only_report`).
    :ptype aspect_reports: Mapping[str, Report]
    :param call: The person's call; None while none is recorded.
    :ptype call: CampaignCall | None
    :param report_doc_id: The report document of the analysis the page is laid out for, when it is one
        (``EvalAnalysis.report_doc_id``): the first screen says the report is written and stored there, in place of
        "no analysis has been generated"; None for any other page.
    :ptype report_doc_id: str | None
    :param unread_analysis_id: The campaign's analysis this version no longer reads, which this code-only page stands in
        for: the first screen says so in the one regenerate line
        (:data:`~threetears.evals.analysis.report.campaign.answer_blocks.NO_LONGER_READS`), in place of "no analysis has
        been generated", and the page names it (``CampaignReport.unread_analysis_id``); None for any other page.
    :ptype unread_analysis_id: str | None
    :param unread_report_doc_id: The report document, read by its id, this version no longer reads, which this
        code-only page stands in for: the first screen says so in the same line
        (:data:`~threetears.evals.analysis.report.campaign.answer_blocks.NO_LONGER_READS`); None for any other page.
    :ptype unread_report_doc_id: str | None
    :param unread_line: What ``unread_analysis_id`` was, as its line says:
        :data:`~threetears.evals.analysis.report.campaign.answer_blocks.NO_LONGER_READS`, or the line of the retired
        writer that wrote it (:func:`~threetears.evals.analysis.report.campaign.answer_blocks.retired_generation`).
    :ptype unread_line: str
    :param arms: The campaign's arms as a reader reads them (``CampaignBundle.arms``: the control first, then the derived
        candidates), which the title names; None names the arms its readings label.
    :ptype arms: Sequence[CampaignArm] | None
    :return: The page, ``basis="code_only"``.
    :rtype: CampaignReport
    :raises RefusedInput: An aspect has no page, a page is keyed by a label the campaign does not hold, or the readings
        are of another campaign.
    """
    reports = _checked_reports(campaign, readings, aspect_reports)
    kind = readings.kind
    bundles = [reports[aspect.label].bundle for aspect in campaign.aspects]
    names = AspectNames.of(bundles)
    # The first screen is the decision: the question, the lead where a guardrail warns, then where the campaign's report
    # is, then what warns, one line per arm, and each report-wide caveat once.
    guardrails = guardrail_warnings(campaign, readings, reports, names=names)
    blocks: list[ReportBlock] = [
        *question_blocks(campaign),
        *(_lead_blocks(campaign, readings, reports, names=names) if guardrails else []),
        *report_doc_blocks(report_doc_id),
        *unread_analysis_blocks(unread_analysis_id, unread_report_doc_id, line=unread_line),
    ]
    noted = any(named is not None for named in (report_doc_id, unread_analysis_id, unread_report_doc_id))
    if kind != "measurement" and not noted:
        blocks.append(DisclosureBlock(section="answer", source="memo", text=NO_RECOMMENDATION))
    blocks += [
        *bar_warnings(readings, names=names),
        *floor_warnings(readings, names=names),
        *guardrails,
        *caveat_blocks(bundles, section="warnings"),
    ]
    if kind == "change":
        blocks += table_blocks(campaign, readings, reports, names=names)
        blocks += settle_blocks(campaign, readings, reports, names=names)
    elif kind == "choice":
        blocks += pair_blocks(campaign, readings, reports, names=names)
        blocks += frontier_blocks(campaign, readings, reports, names=names)
    else:
        blocks += measure_blocks(campaign, readings, reports, names=names)
    blocks += aggregate_blocks(campaign, readings, reports)
    blocks += depth_blocks(campaign, readings, reports, names=names)
    blocks += bar_blocks(readings, reports, names=names)
    if kind != "measurement" or call is not None:
        blocks += call_blocks(campaign, readings, call, names=names, reports=reports)
    # Design & provenance: what each declared check name stands for, the design and when each criterion was set,
    # then what depth says once.
    blocks += check_legend_blocks(names)
    blocks += provenance_blocks(bundles, readings.arm_labels(), names=names)
    blocks += depth_provenance_blocks(readings)
    folds = [
        CampaignAspectFold(
            aspect=aspect.label,
            shows=aspect_shows(readings, aspect.label),
            # What the campaign states once for every aspect, its fold does not say again.
            report=for_the_campaign(reports[aspect.label]),
        )
        for aspect in campaign.aspects
    ]
    if readings.readings or readings.pairs:
        blocks.append(DisclosureBlock(section="methods", source="comparisons", text=PAIRING_METHOD))
    return CampaignReport(
        basis="code_only",
        title=_title(campaign, readings, arms),
        campaign_id=campaign.id,
        campaign_name=campaign.name or None,
        scope_id=campaign.scope_id,
        finding_count=0,
        blocks=blocks,
        aspects=folds,
        readings=readings,
        call=call,
        unread_analysis_id=unread_analysis_id,
    )


def _lead_blocks(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    *,
    names: AspectNames,
) -> list[ReportBlock]:
    """The page's lead, its verdict line, one sentence per candidate on each aspect in the campaign's order, naming
    the reading and both arms: never a cost reading, and never a guardrail's, whose warning follows it.

    Where the aspect's design lists its graders' measures (``AspectDesign.measure_order``, which an eval files in the
    order it lists its graders), the lead is the first of them a contrast tested that is no cost, in the aspect page's
    own words (``Accuracy: v2 and v1 not separated, a lead.``): the measure an eval's printed line leads on, and the
    one its question asks, where it asks one. Where it lists none, the lead is the classifier's accuracy where the
    aspect reads one and no question is asked of it; else the question's measure, the aspect's pass rate (``Pass rate:
    v2 against v1 shown better.``), or with no question the pass rate as every grader's reading together. An aspect
    with no candidate read against a control says what its fold heads it with. The tone is decided where every
    aspect is asked a question, else open."""
    several = len(campaign.aspects) > 1
    sentences: list[str] = []
    for aspect in campaign.aspects:
        on = f" on {names(aspect.label)}" if several else ""
        lead = _lead_measure(aspect, reports[aspect.label])
        led = [
            verdict
            for verdict in reports[aspect.label].verdicts
            if verdict.kind == "contrast" and verdict.reading != "trial" and verdict.name == lead
        ]
        if led:
            sentences += [
                f"{verdict.heading}{on}: {verdict.words}{'' if verdict.aspect is not None else ', a lead'}."
                for verdict in led
            ]
        elif rows := readings.on_aspect(aspect.label):
            sentences += [
                f"Pass rate{on}: {row.arm_label} against {readings.control_label} {READING_WORDS[row.reading]}"
                f"{'' if row.confirmatory else ', a lead'}."
                for row in rows
            ]
        else:
            sentences.append(f"On {names(aspect.label)}, {aspect_shows(readings, aspect.label)}.")
    bundles = [reports[aspect.label].bundle for aspect in campaign.aspects]
    rule = "declared" if bundles and all(bundle.question_declared for bundle in bundles) else "exploratory"
    return [
        DisclosureBlock(
            section="answer", source="verdict", text=" ".join(sentences), tone=DisclosureBlock.verdict_tone(rule)
        )
    ]


def _lead_measure(aspect: Aspect, report: Report) -> str | None:
    """The measure an aspect's lead reads: the first its design's ``measure_order`` lists that a contrast tested and
    that is no cost; with no order listed, the classifier's accuracy where no question is asked of the aspect; None
    where neither names one, and the lead reads the pass rate."""
    order = aspect.design.measure_order if aspect.design is not None else []
    if not order:
        return None if report.bundle.question_declared else ACCURACY_MEASURE
    tested = {verdict.name for verdict in report.verdicts if verdict.kind == "contrast" and verdict.reading != "trial"}
    return next((name for name in order if name in tested and not _a_cost(name, report)), None)


def _a_cost(measure: str, report: Report) -> bool:
    """Whether a measure is a cost: money, or on the cost merit axis, as the aspect's catalog describes it."""
    described = report.bundle.measure_catalog.get(measure) or METRIC_DESCRIPTORS.get(measure)
    return described is not None and (described.merit_axis == "cost" or is_money(described.unit or ""))


def _title(campaign: EvalCampaign, readings: CampaignReadings, arms: Sequence[CampaignArm] | None) -> str:
    """The campaign's title (:func:`~threetears.evals.analysis.campaign_title.named_title`): over ``arms`` where given,
    else over the arms the readings name (the control first), each by its label."""
    name = campaign.name or f"Campaign {campaign.id}"
    aspects = [(aspect.label, aspect.short_name) for aspect in campaign.aspects]
    if arms is not None:
        return declared_title(name, arms, aspects, given={arm.label for arm in campaign.arms})
    return named_title(
        name,
        control=[readings.control_label] if readings.control is not None else [],
        candidates=[label for key, label in readings.arm_labels().items() if key != readings.control],
        aspects=aspects,
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
