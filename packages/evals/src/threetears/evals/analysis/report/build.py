"""Build a :class:`~threetears.evals.analysis.report.model.Report` — code's page over an aspect's evidence.

**One builder, one page, every door.** :func:`build_code_only_report` lays out an aspect's evidence as code's page:
every count, verdict, chart and disclosure is computed here, from the bundle alone, and no door — a campaign's page,
which folds one per aspect, the quick path's ``compare``, the command line, an MCP action — adds, drops or reorders a
block. The writer's report is a report document, drawn by its own page.

The page, screen by screen:

1. **The first screen.** The title first, nothing above it: the campaign's name; the declared question quoted on its
   own line, and under it the arms line (each arm named in
   full once by the version that answered it, the control marked, the date); the verdict line
   (:func:`~threetears.evals.analysis.report.verdict.verdict_line`), result first; one line per candidate breaking a
   floor, and each report-wide caveat once; the scoreboard, a bare table of
   counts and the cost per reply, and its outcome bar; then the caveats, in normal weight.
2. **Decision.** The aspect's floors, each read on each candidate: broken, held or unclear (:mod:`.decision`), and
   for an unclear one the cases that would show it holds.
3. **What it means.** The contrasts that can resolve (one line when none can), the guardrails, and the readings by
   stratum and label.
4. **Cases** and **Trials.** One row per case, each arm's trials as marks, its trial ids last in their own order; then the trials' census in one line and only what went wrong: a row and a record (``T4 · case 1 · arm ·
   outcome``, open up to a cap) per failed or lost trial, and every passed trial as one row of a closed table.
5. **Not tested** and **Methods.** What the campaign could not speak to; then a table (arms, judge, settings sent,
   cost, the analysis's writer, the bundle) and the engine's disclosures folded.

The parts each screen is built from live in :mod:`threetears.evals.analysis.report.aspect`, one module per part; this
module holds the entry points, lays the screens out in order, and names what hosts import.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, refuse_a_bundle_of_another_shape
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.report.aspect.common import (
    NO_STRATUM,
    _arm_names,
    _case_namer,
    _cell_namer,
    _ChartLedger,
    _date,
    _namer,
    _note,
)
from threetears.evals.analysis.report.aspect.contrasts import (
    aspect_verdict,
    contrast_evidence,
    contrast_rows,
    contrast_verdicts,
    report_cell_labels,
)
from threetears.evals.analysis.report.aspect.guardrails import _bar_blocks, bar_verdicts, guardrail_verdicts
from threetears.evals.analysis.report.aspect.head import _floor_warning_blocks, _heading_blocks
from threetears.evals.analysis.report.aspect.method import (
    CHECK_PROOF_WORDS,
    IN_THE_WRITTEN_REPORT,
    NO_ANALYSIS,
    NOT_SHOWN_TO_MEASURE,
    _method_blocks,
    _not_tested_blocks,
)
from threetears.evals.analysis.report.aspect.question import _contrast_blocks, _decision_blocks, _question_blocks
from threetears.evals.analysis.report.aspect.readings import TOO_FEW_CASES
from threetears.evals.analysis.report.aspect.scoreboard import _scoreboard_blocks, _short_names
from threetears.evals.analysis.report.aspect.trials import _case_blocks, _trial_blocks
from threetears.evals.analysis.report.blocks import bare_columns
from threetears.evals.analysis.report.campaign.names import AspectNames, aspect_name
from threetears.evals.analysis.report.campaign.provenance import aspect_provenance_blocks, caveat_blocks, timing_caveats
from threetears.evals.analysis.report.campaign.scorecard import campaign_verdict
from threetears.evals.analysis.report.campaign.warnings import campaign_warnings
from threetears.evals.analysis.report.figures import report_rulers
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.model import (
    FOLDED_SECTIONS,
    SECTION_TITLES,
    DisclosureBlock,
    Report,
    ReportBlock,
    ReportSource,
    TableBlock,
    TrialDetail,
)
from threetears.evals.analysis.report.verdict import bundle_stopped_short, unfinished_line, verdict_line
from threetears.evals.analysis.report.words import input_arm_tags, reader_arm_labels
from threetears.evals.analysis.style import CANNOT_TELL_APART
from threetears.evals.kernel.campaign import EvalAnalysis


def analysis_writer(analysis: EvalAnalysis) -> str:
    """Who wrote a stored analysis, as a page's methods name its writer: the model, the prompt and its version, and
    the day.

    :param analysis: The analysis.
    :ptype analysis: EvalAnalysis
    :return: The words.
    :rtype: str
    """
    generation = analysis.generation
    return (
        f"{generation.generator_model}, prompt {generation.prompt_id} version {generation.prompt_version}, "
        f"{_date(generation.generated_at)}"
    )


def build_code_only_report(
    bundle: AnalysisContextBundle,
    *,
    assembled_at: str,
    campaign_name: str | None = None,
    arm_labels: Mapping[str, str] | None = None,
    arm_tags: Mapping[str, str] | None = None,
    arm_short_names: Mapping[str, str] | None = None,
    writer: str | None = None,
    bundle_fingerprint: str | None = None,
    trials: TrialDetail = "all",
    measure_order: Sequence[str] = (),
) -> Report:
    """Lay a campaign's evidence out as its page, when no analysis exists to report through.

    Everything here is code's, and the methods say no analysis was written. The report model refuses an author's
    words on a code-only report.

    :param bundle: The campaign's evidence, as
        :func:`~threetears.evals.analysis.bundle.assemble.assemble_context_bundle` assembled it.
    :ptype bundle: AnalysisContextBundle
    :param assembled_at: When the bundle was assembled (ISO-8601), stated as the report's date.
    :ptype assembled_at: str
    :param campaign_name: The campaign's name, which the title qualifies what it compares with; ``None`` names the
        campaign by its id.
    :ptype campaign_name: str | None
    :param arm_labels: The labels the campaign gives its arms, by variant key
        (:meth:`~threetears.evals.analysis.campaign_readings.CampaignReadings.arm_labels`), which name those arms
        everywhere on the page in place of their variants' names; None names every arm by its variant.
    :ptype arm_labels: Mapping[str, str] | None
    :param arm_tags: The letter tag each arm carries across the campaign's page, by variant key
        (:func:`~threetears.evals.analysis.report.words.campaign_arm_tags`), so an arm this aspect shares with others
        is tagged and coloured as on every other chart of that page; None tags this page's arms by themselves, as
        does a map that leaves out an arm of this page: in the order the aspect's variant index lists them, since an
        aspect's bundle does not carry its campaign's declared order, so a caller showing the page beside its
        campaign's passes the campaign's tags.
    :ptype arm_tags: Mapping[str, str] | None
    :param arm_short_names: The short names the campaign's arms declare, by variant key
        (:func:`~threetears.evals.analysis.report.words.declared_arm_names`), which a chart prints for those arms in
        its tight places ahead of any it would compute; None computes every arm's.
    :ptype arm_short_names: Mapping[str, str] | None
    :param writer: Who wrote the analysis of the campaign page this page is a fold of, as the methods name a writer
        (:func:`analysis_writer`); None says no analysis was written. This page lays out none of its words.
    :ptype writer: str | None
    :param bundle_fingerprint: The bundle's fingerprint in the current version
        (:meth:`~threetears.evals.analysis.bundle.schema.AnalysisContextBundle.fingerprint`), where the caller has
        just checked it against the bundle — a stored analysis's page, whose frame names it — so the bundle is not
        hashed a second time; None hashes it.
    :ptype bundle_fingerprint: str | None
    :param trials: How much of each failed or lost trial the page lays out (:data:`TrialDetail`): every record by
        default, or ``table`` for their table alone.
    :ptype trials: TrialDetail
    :param measure_order: The aspect's graders' measures, in the order its design lists them
        (``AspectDesign.measure_order``), which the verdict line leads on the first of, as the campaign's page and
        the quick print do (:func:`~threetears.evals.analysis.report.verdict.verdict_line`); empty where none is
        listed.
    :ptype measure_order: Sequence[str]
    :return: The aspect's page, code's alone.
    :rtype: Report
    """
    return build_code_only_report_in_forms(
        bundle,
        assembled_at=assembled_at,
        campaign_name=campaign_name,
        arm_labels=arm_labels,
        arm_tags=arm_tags,
        arm_short_names=arm_short_names,
        writer=writer,
        bundle_fingerprint=bundle_fingerprint,
        trials=trials,
        measure_order=measure_order,
        forms=PageForms.of_bundle(bundle),
    )


def build_code_only_report_in_forms(
    bundle: AnalysisContextBundle,
    *,
    assembled_at: str,
    campaign_name: str | None = None,
    arm_labels: Mapping[str, str] | None = None,
    arm_tags: Mapping[str, str] | None = None,
    arm_short_names: Mapping[str, str] | None = None,
    writer: str | None = None,
    bundle_fingerprint: str | None = None,
    trials: TrialDetail = "all",
    measure_order: Sequence[str] = (),
    forms: PageForms,
) -> Report:
    """An aspect's code-only page (:func:`build_code_only_report`), in forms its caller built over its bundle: the
    engine's route for a caller that also reads the page's verdict sentences apart (a campaign's page,
    :func:`contrast_evidence`) and builds them once for both. Not exported: the forms are the engine's own.

    :param bundle: The aspect's evidence, as :func:`build_code_only_report` reads it.
    :ptype bundle: AnalysisContextBundle
    :param assembled_at: When the bundle was assembled (ISO-8601), stated as the report's date.
    :ptype assembled_at: str
    :param campaign_name: As :func:`build_code_only_report` reads it.
    :ptype campaign_name: str | None
    :param arm_labels: As :func:`build_code_only_report` reads them.
    :ptype arm_labels: Mapping[str, str] | None
    :param arm_tags: As :func:`build_code_only_report` reads them.
    :ptype arm_tags: Mapping[str, str] | None
    :param arm_short_names: As :func:`build_code_only_report` reads them.
    :ptype arm_short_names: Mapping[str, str] | None
    :param writer: As :func:`build_code_only_report` reads it.
    :ptype writer: str | None
    :param bundle_fingerprint: As :func:`build_code_only_report` reads it.
    :ptype bundle_fingerprint: str | None
    :param trials: As :func:`build_code_only_report` reads it.
    :ptype trials: TrialDetail
    :param measure_order: As :func:`build_code_only_report` reads it.
    :ptype measure_order: Sequence[str]
    :param forms: The page's forms (:meth:`~threetears.evals.analysis.report.forms.PageForms.of_bundle`), built over
        ``bundle``.
    :ptype forms: PageForms
    :return: The aspect's page, code's alone.
    :rtype: Report
    :raises ValueError: ``forms`` were built from another bundle.
    """
    source = ReportSource(
        campaign_id=bundle.campaign_id,
        campaign_name=campaign_name or None,
        scope_id=bundle.scope_id,
        behavior=bundle.behavior,
        generated_at=assembled_at,
        bundle_fingerprint=bundle_fingerprint if bundle_fingerprint is not None else bundle.fingerprint(),
    )
    return _page(
        bundle,
        source=source,
        writer=writer,
        arm_labels=arm_labels,
        tags=arm_tags,
        arm_short_names=arm_short_names,
        trials=trials,
        measure_order=measure_order,
        forms=forms,
    )


def analysis_campaign_bundle(analysis: EvalAnalysis) -> CampaignBundle:
    """The campaign bundle a stored analysis was written over, put back together from its frame and frozen aspects.

    :param analysis: The analysis, of one aspect or several.
    :ptype analysis: EvalAnalysis
    :return: The bundle, as frozen at generation.
    :rtype: CampaignBundle
    :raises RefusedInput: The analysis was written over a bundle of another shape, refused before any aspect's bundle
        is read.
    """
    what = f"analysis {analysis.id}"
    remedy = "generate the analysis again (analysis_generate), which assembles the campaign's evidence as it stands"
    # Every aspect's stored shape is checked before any is read: reading one is the costly half, and an analysis
    # whose later aspect no longer reads is refused without paying for the earlier ones.
    for held in analysis.aspects:
        refuse_a_bundle_of_another_shape(held.bundle, what=what, remedy=remedy)
    bundles = [AnalysisContextBundle.from_dict(dict(held.bundle)) for held in analysis.aspects]
    return CampaignBundle.from_frame(analysis.campaign_bundle, bundles, what=what)


class AnalysisLead(NamedTuple):
    """What a stored analysis is listed by, written by code over what was frozen with it (:func:`analysis_lead`).

    :ivar caveats: What warns on the analysis's page, then its one caveat line on when its criteria were set, in the
        page's own words; empty with neither.
    :ivar verdict: The verdict alone; over evidence with a run that did not finish, what stopped it, with no verdict
        on the aspect it stopped.
    :ivar stopped: Whether a run of the evidence did not finish
        (:func:`~threetears.evals.analysis.report.verdict.bundle_stopped_short`), so :attr:`verdict` opens on what
        stopped it and the line leads with it.
    """

    caveats: list[str]
    verdict: str
    stopped: bool = False

    def line(self) -> str:
        """The caveats, then the verdict, as one line: what an analysis is listed by (``AnalysisLine.verdict``). Over
        a run that did not finish, what stopped it leads and the caveats follow it: a broken floor read over the
        results recorded before a run stopped is no opening.

        :return: The line.
        :rtype: str
        """
        return " ".join([self.verdict, *self.caveats] if self.stopped else [*self.caveats, self.verdict])


def analysis_lead(analysis: EvalAnalysis, *, bundle: CampaignBundle | None = None) -> AnalysisLead:
    """The verdict a stored analysis is listed by, and its caveats beside it, both written by code over what was frozen
    with it — never a writer's words.

    The caveats are what warns on the analysis's page: each arm missing a bar, then each candidate breaking a floor, in
    the page's own words (:func:`~threetears.evals.analysis.report.campaign_page.campaign_warnings`) over the readings
    frozen with it, without the writer's why beside them; then the page's one caveat line on when its criteria were
    set, where one holds (:func:`~threetears.evals.analysis.report.campaign_page.timing_caveats`), never a line per
    criterion. Every arm is named as a page names it
    (:func:`~threetears.evals.analysis.report.words.reader_arm_labels`): its label, else its computed name, a level too
    long to print by its lever (told apart by the tag the report document gives it, where two arms read alike), an
    empty text level as ``no <lever>``. The verdict of an analysis of
    one aspect is the verdict line that aspect's page opens with, over its frozen bundle. An analysis of several aspects is listed by its campaign's verdict, as the campaign's page
    reads it: what code read on each aspect, in the campaign's order
    (:func:`~threetears.evals.analysis.report.campaign_page.campaign_verdict`).

    :param analysis: The analysis.
    :ptype analysis: EvalAnalysis
    :param bundle: The campaign bundle frozen with it (:func:`analysis_campaign_bundle`), where the caller has put it
        back together already; None puts it back together here.
    :ptype bundle: CampaignBundle | None
    :return: The caveats and the verdict.
    :rtype: AnalysisLead
    :raises RefusedInput: The analysis was written over a bundle of another shape.
    """
    campaign = bundle if bundle is not None else analysis_campaign_bundle(analysis)
    # Every arm by the name a page prints it by, read at render time (its label, else its computed name, a cut level by
    # its lever), told apart by the tag the campaign's report document wears it by, since a listing prints that
    # document's headline beside these lines: the frozen readings are relabelled for print only.
    labels = reader_arm_labels(campaign, input_arm_tags(campaign))
    readings = campaign.readings.relabelled(labels)
    names = AspectNames.of(campaign.aspects, campaign=campaign.name or None)
    caveats = [
        *campaign_warnings(readings, names, forms=PageForms.off_page()),
        *timing_caveats(campaign.aspects, names),
    ]
    stopped = [aspect for aspect in campaign.aspects if bundle_stopped_short(aspect) is not None]
    if len(analysis.aspects) > 1:
        # An aspect with a run that did not finish is listed by what stopped it, first, and never by a reading.
        arm_labels = readings.arm_labels()
        opened = [
            f"{names(aspect.aspect)}: {line.text}"
            for aspect in stopped
            if (line := unfinished_line(aspect, lambda key: arm_labels.get(key, key))) is not None
        ]
        unfinished = {aspect.aspect for aspect in stopped}
        finished = [aspect.label for aspect in analysis.aspects if aspect.label not in unfinished]
        verdict = " ".join([*opened, *([campaign_verdict(readings, finished, names)] if finished else [])])
    else:
        aspect = campaign.aspects[0]
        declared = campaign.declaration.aspects[0]
        verdict = aspect_verdict(
            aspect,
            measure_order=declared.design.measure_order if declared.design is not None else [],
            arm_labels=readings.arm_labels(),
            # The aspect's page's forms, over its campaign's set of each measure, as the page itself reads them.
            forms=PageForms.of_aspects(campaign.aspects, readings)[aspect.aspect],
        ).text
    return AnalysisLead(caveats=caveats, verdict=verdict, stopped=bool(stopped))


def _page(
    bundle: AnalysisContextBundle,
    *,
    source: ReportSource,
    writer: str | None,
    arm_labels: Mapping[str, str] | None = None,
    tags: Mapping[str, str] | None = None,
    arm_short_names: Mapping[str, str] | None = None,
    trials: TrialDetail = "all",
    measure_order: Sequence[str] = (),
    forms: PageForms,
) -> Report:
    """Code's page over one bundle: the same screens, whatever door renders it.

    ``arm_labels`` are a campaign's own labels for its arms, by variant key, which name those arms everywhere on the
    page in place of their variants' names; ``tags`` the campaign's letter tags for them, which tag and colour them on
    its charts; ``arm_short_names`` the short names its arms declare, which its charts print in tight places.
    ``trials`` says whether each failed or lost trial's record is laid out. The title is built over the arms the page
    names and the one aspect its bundle reads.

    The contrasts are laid out before the scoreboard, though they print after it: each chart they leave out is counted
    on the way, for the scoreboard's one line about what was not charted.
    """
    surface = bundle_decision_surface(bundle)
    rulers = report_rulers(surface, bundle.measure_catalog)
    # Every figure the page states is decided once, for the whole page, by the bundle: by the forms its caller built
    # over it, which a campaign's page also reads the page's verdict sentences in.
    if forms.bundle is not bundle:
        raise ValueError("an aspect page's forms were handed in beside a bundle they were not built from")
    names = _arm_names(bundle, labels=arm_labels)
    arm = _namer(names)
    cell_name = _cell_namer(surface, names)
    charts = _ChartLedger(bundle, arm)
    guardrails_read = guardrail_verdicts(
        surface, bundle.guardrails, cell_name, rulers, check_names=bundle.named_checks(), forms=forms
    )
    bars_read = bar_verdicts(surface, cell_name, check_names=bundle.named_checks())
    contrasts_read = contrast_verdicts(bundle, surface, arm_labels=arm_labels, forms=forms)
    # The readings no declared question asked about are said with what was not tested, after what code says there.
    noted = _question_blocks(bundle)
    contrasts = _contrast_blocks(bundle, surface, contrasts_read, guardrails_read, charts, names, rulers, forms)
    contrasts += _bar_blocks(surface, bars_read, rulers, forms)
    notes = [block for block in contrasts if block.section == "methods"]
    contrasts = [block for block in contrasts if block.section != "methods"]
    if (not_charted := charts.sentence()) is not None:
        notes.append(_note(not_charted, "charts"))
    name = source.campaign_name or f"Campaign {source.campaign_id}"
    # The aspect as every line of the page names it: an engine-given label gives way to the campaign's name.
    printed = aspect_name(bundle, campaign=source.campaign_name or None)
    verdict = verdict_line(
        bundle,
        arm,
        case_name=_case_namer(bundle),
        verdicts=[*guardrails_read, *contrasts_read],
        order=measure_order,
        evidence=contrast_evidence(bundle, forms=forms),
    )
    blocks: list[ReportBlock] = [
        # Under the title: the question and the arms, the verdict, then what warns and each report-wide caveat once.
        *_heading_blocks(bundle, names, source, aspect=None if printed == name else printed),
        DisclosureBlock(
            section="header",
            source="verdict",
            text=verdict.text,
            tone=DisclosureBlock.verdict_tone(verdict.rule),
            # Code's own decisions: bold falls away where a host's name would cut a code span, never refusing the page.
            emphasis=DisclosureBlock.code_emphasis(verdict.text, verdict.decisions),
            # A verdict that offers leads links the word that says why they are not decisions: the exploratory rule's
            # line, and the too-small rule's where no question was declared, which is when it leads "Exploratory:".
            term=(
                "exploratory"
                if verdict.rule == "exploratory" or (verdict.rule == "too_small" and not bundle.question_declared)
                # Read off the verdicts the line states, never off its words.
                else CANNOT_TELL_APART
                if verdict.undecided
                else None
            ),
        ),
        *_floor_warning_blocks(bundle, arm, printed, forms),
        *caveat_blocks(
            [bundle], section="header", names=AspectNames.of([bundle], campaign=source.campaign_name or None)
        ),
        *_scoreboard_blocks(
            bundle, arm, _short_names(bundle, names, arm_labels or {}, arm_short_names or {}), forms, tags
        ),
        *_decision_blocks(bundle, arm, printed, forms),
        *contrasts,
        *_case_blocks(bundle, arm, forms),
        *_trial_blocks(bundle, arm, forms, records=trials == "all"),
        *_not_tested_blocks(bundle, surface, arm, also=noted),
        *_method_blocks(
            bundle,
            surface,
            arm,
            names,
            forms,
            source=source,
            writer=writer,
            left_disclosures=notes,
        ),
        # When each criterion set after its readings was set: the detail the first screen's one line points at.
        *aspect_provenance_blocks(bundle, forms=forms),
    ]
    # In reading order, each section's blocks in the order laid out (a code-only page's unasked readings are said
    # with what was not tested, though they are laid out with the questions).
    order = list(SECTION_TITLES)
    blocks.sort(key=lambda block: order.index(block.section))
    # Every cell's figure was decided by its reading's set as its builder made it, so no column rule respells it: each
    # table's rate and money cells are recorded as they print; then one cell keeps each reading's anchor.
    blocks = _anchored(
        [
            block.model_copy(update={"rows": forms.recorded(block.rows, bare=bare_columns(block.columns))})
            if isinstance(block, TableBlock)
            else block
            for block in blocks
        ]
    )
    return Report(
        # The heading is the campaign's name; what it compares is the arms line under it.
        title=name,
        source=source,
        blocks=blocks,
        verdicts=[*guardrails_read, *bars_read, *contrasts_read],
        bundle=bundle,
    )


def _anchored(blocks: Sequence[ReportBlock]) -> list[ReportBlock]:
    """The page with each reading's anchor kept by one cell, and a closed table holding one anchored by its fold.

    Every cell printing a reading's own figure at a cell carries that reading's id (:func:`_at`), and one cell keeps
    it: a cell shown open before one in a closed table, a contrast before any other table, else the first on the page.
    Where the cell keeping it sits in a table shown closed, the table's fold carries the anchor too, which every
    browser shows, since a browser that leaves the fold shut would not show a cell inside it.
    """
    placed = list(blocks)
    landing: dict[str, str] = {}
    claimed: set[str] = set()
    tables = {index: block for index, block in enumerate(placed) if isinstance(block, TableBlock)}
    rank = {"open": 0, "folded": 1, "hidden": 2}
    order = sorted(tables, key=lambda index: (rank[_shown(tables[index])], tables[index].section != "contrasts", index))
    for index in order:
        table = tables[index]
        shown = _shown(table)
        fold = f"t-{table.name}"
        rows = []
        for row in table.rows:
            kept = {}
            for key, cell in row.items():
                if cell.anchor is not None and cell.anchor in claimed:
                    cell = cell.model_copy(update={"anchor": None})
                elif cell.anchor is not None:
                    claimed.add(cell.anchor)
                    if shown == "open":
                        landing[cell.anchor] = cell.anchor
                    elif shown == "folded":
                        landing[cell.anchor] = fold
                kept[key] = cell
            rows.append(kept)
        update: dict[str, object] = {"rows": rows}
        if shown == "folded" and fold in {landing.get(cell.anchor or "") for row in rows for cell in row.values()}:
            update["anchor"] = fold
        placed[index] = table.model_copy(update=update)
    return placed


def _shown(table: TableBlock) -> Literal["open", "folded", "hidden"]:
    """How a table is laid out: ``open``; ``folded``, a closed table of its own in an open section, whose fold every
    browser shows; or ``hidden``, inside a closed section or a closed fold of several blocks."""
    if table.fold is not None or table.section in FOLDED_SECTIONS:
        return "hidden"
    return "folded" if table.details else "open"


__all__ = [
    "AnalysisLead",
    "aspect_verdict",
    "contrast_evidence",
    "IN_THE_WRITTEN_REPORT",
    "NO_ANALYSIS",
    "contrast_rows",
    "report_cell_labels",
    "analysis_campaign_bundle",
    "analysis_lead",
    "CHECK_PROOF_WORDS",
    "NOT_SHOWN_TO_MEASURE",
    "NO_STRATUM",
    "TOO_FEW_CASES",
    "bar_verdicts",
    "build_code_only_report",
    "analysis_writer",
    "contrast_verdicts",
    "guardrail_verdicts",
]
