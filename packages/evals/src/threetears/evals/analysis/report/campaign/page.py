"""Build a campaign's page: the decision first, then what code read across its aspects, how the campaign was set up,
then one fold per aspect.

Code reports, the writer recommends, a person decides, and the page keeps the three apart. This page is code's: the
writer's report is a report document, which its own page draws (:mod:`threetears.evals.analysis.report.doc_page`).
The first screen is the decision, and says what the page is, never what it lacks. Under the title, for every kind:
**the question**, or with none asked of a page comparing arms that it is an exploratory comparison
(:mod:`.provenance`); then, on every page, **the lead**: one sentence per candidate on the first measure the aspect's
design lists in its graders' order, the one an eval's question asks and its printed line leads on; with no order
listed, on the question's measure, the aspect's pass rate, or with no question asked on the classifier's accuracy; for
one arm measured, its pass rate with the count it is computed over; so a guardrail's warning sits beside the lead and
never in its place; where the campaign's report is, where it has one: the report document it is stored as, or the
analysis this version no longer reads or a retired writer wrote (:mod:`.answer_blocks`); then **what warns**, one line
per arm for the bars it misses, the floors it breaks and the guardrails it is not shown safe on (:mod:`.warnings`), and
each report-wide caveat once (:mod:`.provenance`); then the person's call, once one is recorded.

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
from threetears.evals.analysis.campaign_title import arms_line, title_arm_names
from threetears.evals.analysis.depth_reading import PAIRING_METHOD
from threetears.evals.analysis.report.campaign.aggregate import aggregate_blocks
from threetears.evals.analysis.report.campaign.depth_blocks import depth_blocks, depth_provenance_blocks
from threetears.evals.analysis.report.campaign.answer_blocks import (
    NO_LONGER_READS,
    call_blocks,
    report_doc_blocks,
    stored_shape_blocks,
    unread_analysis_blocks,
)
from threetears.evals.analysis.report.campaign.names import AspectNames
from threetears.evals.analysis.report.campaign.provenance import (
    caveat_blocks,
    for_the_campaign,
    provenance_blocks,
    question_blocks,
)
from threetears.evals.analysis.report.campaign.scorecard import (
    aspect_shows,
    bar_blocks,
    frontier_blocks,
    measure_blocks,
    measured_figure,
    pair_blocks,
    pairs_words,
    row_figures,
    row_sentence,
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
from threetears.evals.kernel.declaration import Aspect, CampaignArm, CampaignCall
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.analysis.report.blocks import TableBlock, bare_columns, percent_columns
from threetears.evals.analysis.report.build import contrast_evidence
from threetears.evals.analysis.report.verdict import led_sentences
from threetears.evals.analysis.style import CANNOT_TELL_APART

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
    lead: Sequence[ReportBlock] | None = None,
    folds_alone: bool = False,
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
        (``EvalAnalysis.report_doc_id``): the first screen says the report is written and stored there; None for any
        other page.
    :ptype report_doc_id: str | None
    :param unread_analysis_id: The campaign's analysis this version no longer reads, which this code-only page stands in
        for: the first screen says so in the one regenerate line
        (:data:`~threetears.evals.analysis.report.campaign.answer_blocks.NO_LONGER_READS`), and the page names it
        (``CampaignReport.unread_analysis_id``); None for any other page.
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
        candidates), which the arms line under the heading names; None names the arms its readings label.
    :ptype arms: Sequence[CampaignArm] | None
    :param lead: The page's answer where its caller states its own, in place of the lead code writes from the
        readings: the way a harness whose eval reads a figure of its own (a judge's preference, a promotion's
        reading) states its answer on the page, as its own blocks with their own tone. They replace code's lead
        block in place, in their order: on every campaign page that is directly after the question line (or the
        line saying the page is an exploratory comparison) and the arms line under it, before where the report is and
        what warns. Each is in
        the ``answer`` section. None for code's lead; an empty sequence is refused. Nothing else on the page changes
        with it.
    :ptype lead: Sequence[ReportBlock] | None
    :param folds_alone: Whether the folds are laid out without the page's own blocks, as a report document's evidence
        record is (:func:`~.provenance.for_the_campaign`, ``alone``): each fold then keeps its judge's calibration
        line, which the page's own Design & provenance would otherwise state once.
    :ptype folds_alone: bool
    :return: The page, ``basis="code_only"``.
    :rtype: CampaignReport
    :raises RefusedInput: An aspect has no page, a page is keyed by a label the campaign does not hold, the readings
        are of another campaign, or a given lead is empty or holds a block outside the ``answer`` section.
    """
    reports = _checked_reports(campaign, readings, aspect_reports)
    if lead is not None:
        _check_lead(lead)
    kind = readings.kind
    bundles = [reports[aspect.label].bundle for aspect in campaign.aspects]
    names = AspectNames.of(bundles, campaign=campaign.name or None)
    # The first screen is the decision: the question, what the page compares, the lead, then where the campaign's
    # report is where it has one, then what warns, one line per arm, and each report-wide caveat once. Evidence of an
    # earlier shape this build reads as stored says so before all of it.
    blocks: list[ReportBlock] = [
        *stored_shape_blocks(bundle.schema_version for bundle in bundles),
        *question_blocks(campaign, compares=kind != "measurement"),
        *_arms_blocks(campaign, readings, arms),
        *(_lead_blocks(campaign, readings, reports, names=names) if lead is None else lead),
        *report_doc_blocks(report_doc_id),
        *unread_analysis_blocks(unread_analysis_id, unread_report_doc_id, line=unread_line),
        *bar_warnings(readings, names=names),
        *floor_warnings(readings, names=names),
        *guardrail_warnings(campaign, readings, reports, names=names),
        *caveat_blocks(bundles, section="warnings", names=names),
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
    blocks += call_blocks(campaign, readings, call, names=names, reports=reports)
    # Design & provenance: the design and when each criterion was set, then what depth says once. What each check's
    # name stands for is the glossary's (``CampaignReport.glossary``).
    blocks += provenance_blocks(bundles, readings.arm_labels(), names=names)
    blocks += depth_provenance_blocks(readings)
    folds = [
        CampaignAspectFold(
            aspect=aspect.label,
            name=_fold_heading(aspect, names, _title(campaign)),
            shows=aspect_shows(readings, aspect.label),
            # What the campaign states once for every aspect, its fold does not say again.
            report=for_the_campaign(reports[aspect.label], alone=folds_alone),
        )
        for aspect in campaign.aspects
    ]
    if readings.readings or readings.pairs:
        blocks.append(DisclosureBlock(section="methods", source="comparisons", text=PAIRING_METHOD))
    # Each table's percentages at one precision per column (percent_columns), as the aspects' pages read theirs.
    blocks = [
        block.model_copy(update={"rows": percent_columns(block.rows, bare=bare_columns(block.columns))})
        if isinstance(block, TableBlock)
        else block
        for block in blocks
    ]
    return CampaignReport(
        basis="code_only",
        title=_title(campaign),
        campaign_id=campaign.id,
        campaign_name=campaign.name or None,
        scope_id=campaign.scope_id,
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
    """The page's lead, its verdict line, on every page where its caller states no answer of its own: one sentence
    per candidate on each aspect in the campaign's order, naming the reading and both arms; for one arm measured, the arm's pass rate with its count. Never a cost reading, and never a
    guardrail's, whose warning follows it.

    Where the aspect's design lists its graders' measures (``AspectDesign.measure_order``, which an eval files in the
    order it lists its graders), the lead is the first of them a contrast tested that is no cost, in the aspect page's
    own words (``Accuracy: the evidence cannot tell v2 and v1 apart, a lead.``,
    :func:`~threetears.evals.analysis.report.verdict.led_sentences`): the measure an eval's printed line leads on, and the
    one its question asks, where it asks one. Where it lists none, the lead is the classifier's accuracy where the
    aspect reads one and no question is asked of it; else the question's measure, the aspect's pass rate (``Pass rate:
    v2 against v1 shown better.``), or with no question the pass rate as every grader's reading together. A reading
    the question does not ask is marked a lead, unless the campaign asks none, which its first line says once
    (``Exploratory comparison.``). A choice leads on each aspect's pairs. One arm measured leads on its pass rate, as
    its count (``reply passed 40 of 40 trials on 20 cases.``), its percent and interval in the scorecard under it. An aspect with none of these says what its fold heads it with. The tone is decided where every aspect
    is asked a question, else open."""
    bundles = [reports[aspect.label].bundle for aspect in campaign.aspects]
    rule = "declared" if bundles and all(bundle.question_declared for bundle in bundles) else "exploratory"
    tone = DisclosureBlock.verdict_tone(rule)
    several = len(campaign.aspects) > 1
    asked = bool(campaign.question.strip())
    sentences: list[str] = []
    for aspect in campaign.aspects:
        on = f" on {names(aspect.label)}" if several else ""
        report = reports[aspect.label]
        order = aspect.design.measure_order if aspect.design is not None else []
        # The aspect's own verdict line's sentence, as its page and the quick print say it (verdict.py).
        figures = contrast_evidence(report.bundle)
        if led := led_sentences(report.bundle, report.verdicts, order, asked=asked, on=on, evidence=figures):
            sentences += led
        elif rows := readings.on_aspect(aspect.label):
            sentences += [
                f"Pass rate{on}: {row_sentence(row, readings.control_label)}"
                f"{'' if row.confirmatory or not asked else ', a lead'}. {row_figures(row, readings.control_label)}"
                for row in rows
            ]
        elif paired := pairs_words(readings, aspect.label, marked=asked):
            # An arm's label is a name, printed as given, even where it opens the sentence; the undecided reading's
            # own words open with a capital, as a sentence does.
            opened = paired[:1].upper() + paired[1:] if paired.startswith("the evidence ") else paired
            sentences.append(f"On {names(aspect.label)}, {paired}." if several else f"{opened}.")
        elif measured := _measured_lead(aspect, readings):
            prefix = f"On {names(aspect.label)}, " if several else ""
            sentences += [f"{prefix}{said}." for said in measured]
        else:
            sentences.append(f"On {names(aspect.label)}, {aspect_shows(readings, aspect.label)}.")
    text = " ".join(sentences)
    # Where the lead says the evidence cannot tell arms apart, it links that phrase's one definition at its first use.
    term = CANNOT_TELL_APART if "cannot tell" in text else None
    return [DisclosureBlock(section="answer", source="verdict", text=text, tone=tone, term=term)]


def _check_lead(lead: Sequence[ReportBlock]) -> None:
    """Refuse a caller's lead that would not stand where code's does: empty, or with a block outside the answer.

    :raises RefusedInput: It is empty, or a block of it is in another section.
    """
    if not lead:
        raise RefusedInput(
            "the page's lead is given as no block at all",
            remedy="pass None for code's lead, or the blocks that state the caller's answer",
        )
    if stray := sorted({block.section for block in lead if block.section != "answer"}):
        raise RefusedInput(
            f"the page's lead holds a block in section(s) {', '.join(stray)}, not the answer",
            remedy='give each block of the lead section="answer", where code\'s lead stands',
        )


def _measured_lead(aspect: Aspect, readings: CampaignReadings) -> list[str]:
    """Each measured arm's pass rate on an aspect, as the count it is computed over
    (:func:`~.scorecard.measured_figure`): the figure every grader's reading makes together, as the aspect's printed
    line leads on its cases passed; none where no arm was measured there."""
    return [
        figure
        for reading in readings.measures
        # ``passed``: the trials' own pass rate, as the readings key it.
        if reading.aspect == aspect.label and reading.measure == "passed"
        if (figure := measured_figure(reading)) is not None
    ]


#: The heading of an aspect's fold where the aspect is called by the page's own name (one aspect, the quick path's).
_EVIDENCE_HEADING = "Evidence"


def _fold_heading(aspect: Aspect, names: AspectNames, title: str) -> str:
    """An aspect fold's heading: its full label, which a declared short name stands in for everywhere else; an
    engine-given label by the name the page calls the aspect; and a fold named as the page is, its evidence, so no
    heading prints twice."""
    heading = aspect.label if aspect.short_name is not None else names(aspect.label)
    return _EVIDENCE_HEADING if heading == title else heading


def _title(campaign: EvalCampaign) -> str:
    """The page's heading: the campaign's name, which says what the campaign is; what it compares and on what is the
    line under it (:func:`_arms_blocks`), so the heading never stacks names."""
    return campaign.name or f"Campaign {campaign.id}"


def _arms_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, arms: Sequence[CampaignArm] | None
) -> list[ReportBlock]:
    """The line under the heading (:func:`~threetears.evals.analysis.campaign_title.arms_line`): the arms, each by
    its name, and the aspects, each part the campaign's name does not already say; over ``arms`` where given (each as a
    title names it), else over the arms the readings name, each by its label. Nothing where the name says it all."""
    aspects = [(aspect.label, aspect.short_name) for aspect in campaign.aspects]
    if arms is not None:
        control, candidates = title_arm_names(arms, given={arm.label for arm in campaign.arms})
    else:
        control = [readings.control_label] if readings.control is not None else []
        candidates = [label for key, label in readings.arm_labels().items() if key != readings.control]
    line = arms_line(_title(campaign), control=control, candidates=candidates, aspects=aspects)
    return [DisclosureBlock(section="header", source="arms", text=line)] if line else []


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
