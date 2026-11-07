"""Build a :class:`~threetears.evals.analysis.report.model.Report` — code's page over an aspect's evidence.

**One builder, one page, every door.** :func:`build_code_only_report` lays out an aspect's evidence as code's page:
every count, verdict, chart and disclosure is computed here, from the bundle alone, and no door — a campaign's page,
which folds one per aspect, the quick path's ``compare``, the command line, an MCP action — adds, drops or reorders a
block. The writer's report is a report document, drawn by its own page.

The page, screen by screen:

1. **The first screen.** The title first, nothing above it (what the campaign compares and on what, with its name as a
   qualifier: :func:`~threetears.evals.analysis.campaign_title.named_title`); the declared question quoted on its own line, and under it the arms line (each arm named in
   full once by the version that answered it, the control marked, the date); the verdict line
   (:func:`~threetears.evals.analysis.report.verdict.verdict_line`), result first; one line per candidate breaking a
   floor, and each report-wide caveat once; the scoreboard, a bare table of
   counts and the cost per reply, and its outcome bar; then the caveats, in normal weight.
2. **Decision.** The aspect's floors, each read on each candidate: broken, held or unclear (:mod:`.decision`), and
   for an unclear one the cases that would show it holds.
3. **What it means.** The contrasts that can resolve (one line when none can), the guardrails, and the readings by
   stratum and label.
4. **Cases** and **Trials.** One row per case, each arm's trials as marks, its trial ids last and the failed ones
   first; then the trials' census in one line and only what went wrong: a row and a record (``T4 · case 1 · arm ·
   outcome``, open up to a cap) per failed or lost trial, and every passed trial as one row of a closed table.
5. **Not tested** and **Methods.** What the campaign could not speak to; then a table (arms, judge, settings sent,
   cost, the analysis's writer, the bundle) and the engine's disclosures folded.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from itertools import chain
from typing import Literal, NamedTuple, Protocol

from threetears.evals.analysis.agreement import tier_sentence
from threetears.evals.analysis.arms import (
    arm_levels,
    arm_names,
    distinguishing_axes,
    elide_level,
    multi_rig_variants,
    short_digest,
    surface_order,
)
from threetears.evals.analysis.bundle.schema import (
    AnalysisContextBundle,
    ArmScoreboard,
    CellCoordinate,
    ComparedCell,
    FLOOR_COST_MEASURE,
    FamilyComparison,
    GoalCheckProofReading,
    TrialRow,
    UnexercisedFactor,
    refuse_a_bundle_of_another_shape,
)
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.lenses.frontier import PASS_HAT_K_CELL_READING, pass_hat_k_cell_lines
from threetears.evals.kernel.numbers import MoneyForm, format_number
from threetears.evals.analysis.references import resolve_reading
from threetears.evals.analysis.report.blocks import money_columns
from threetears.evals.analysis.report.cost_unit import PRICED_TRIALS_TEMPLATE, money_ruler
from threetears.evals.analysis.report.figures import (
    NO_SUCCESSFUL_RESULTS,
    SHARE,
    MeasureRuler,
    Rulers,
    reply_cost_ruler,
    report_rulers,
    surface_order_sentence,
)
from threetears.evals.analysis.report.model import (
    CellValue,
    ChartBlock,
    FOLDED_SECTIONS,
    SECTION_TITLES,
    DisclosureBlock,
    DisclosureSource,
    OutcomeBar,
    OutcomeSegment,
    Report,
    ReportBlock,
    ReportSource,
    TableBlock,
    TableCell,
    TableColumn,
    TrialBlock,
    TrialDetail,
    Verdict,
    VerdictReason,
    reading_anchor,
    table_row,
)
from threetears.evals.analysis.report.decision import (
    FloorVerdict,
    SampleProjection,
    case_cost_usd,
    floor_projection,
    floor_readings,
    floors_projection,
    projected_cell,
    projected_usd_value,
    projection_note,
    read_floor,
)
from threetears.evals.analysis.campaign_title import named_title
from threetears.evals.analysis.report.campaign.names import aspect_name
from threetears.evals.analysis.report.campaign.provenance import (
    aspect_provenance_blocks,
    calibration_blocks,
    caveat_blocks,
    setup_disclosure,
)
from threetears.evals.analysis.report.campaign.warnings import floor_line
from threetears.evals.analysis.report.campaign.names import AspectNames
from threetears.evals.analysis.report.campaign.provenance import timing_caveats
from threetears.evals.analysis.report.campaign.scorecard import campaign_verdict
from threetears.evals.analysis.report.campaign.warnings import campaign_warnings
from threetears.evals.analysis.report.verdict import (
    MIN_CASES,
    CauseWords,
    arm_exclusions,
    arm_run_statuses,
    cause_short,
    caveat_lines,
    fail_words,
    guardrail_lines,
    interval_words,
    loss_words,
    VerdictLine,
    verdict_line,
)
from threetears.evals.analysis.report.words import (
    BAR_DECISION_WORDS,
    arm_tags,
    comparison_words,
    guardrail_words,
    input_arm_tags,
    reader_arm_labels,
    PASS_RATE,
    judged_heading,
    one_figure_each,
    reading_verb,
    reading_heading,
)
from threetears.evals.analysis.reporting import self_judging_text
from threetears.evals.kernel.host.sweepables import CANDIDATE_MODEL_LEVER
from threetears.evals.analysis.style import (
    CANNOT_TELL_APART,
    ArmName,
    arm_display_name,
    arm_name,
    code,
    declared_check_name,
    humanize,
    lever_reader_name,
    measure_name,
)
from threetears.evals.analysis.stats import EQUIVALENCE_NEEDS_RANGE, INTERVAL_LEVEL
from threetears.evals.analysis.quantities import (
    MONEY,
    PERCENT,
    delta_unit,
    is_money,
    number_in,
    render_cell,
    with_unit,
)
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.kernel.campaign import (
    EvalAnalysis,
    ReadingKind,
    VariantIndexEntry,
)
from threetears.evals.kernel.declaration import (
    CostLower,
    RateDrop,
)
from threetears.evals.kernel.metrics import (
    ACCURACY_MEASURE,
    MATCH_MEASURE,
    METRIC_DESCRIPTORS,
    goal_check_of,
    RUN_MARGIN_MEASURES,
    ClassifierStatistic,
    classifier_label_of,
)
from threetears.evals.kernel.surface import (
    STRATUM_MIN_CASES,
    CellFacts,
    DecisionSurface,
    GuardrailCheck,
    GuardrailReadings,
    JudgedReading,
    PredictionPoweredReading,
    StratumFacts,
    TimeAxis,
    earlier_control_mark,
)
from threetears.evals.schema.models import CHECK_REFUSED_UNDER_CURRENT_GRAMMAR, ClientRequestSettings

#: What the methods say of a report no analysis was written for.
NO_ANALYSIS = "none written; the analysis_generate action writes one from this same evidence"

#: How a guardrail is decided, said once in the methods' disclosures.
_GUARDRAILS_DISCLOSURE = (
    "A guardrail is what an arm must not get worse on. Each is decided on its own 95% interval on the difference "
    "from the control: held when the whole interval lies on the good side of the margin, breached when it lies "
    "wholly beyond it, undecided otherwise. On a declared range the interval is the bounded test's, which holds "
    "its error rate at any number of cases; a reading with no declared range is never held, since no test holds "
    "its rate there, and can only be shown breached. A measure's margin is its declared materiality threshold, a "
    "judged dimension's the one its campaign declares; with none, a guardrail is held at zero change. Guardrails "
    "join no comparison and no composite, so no gain elsewhere offsets one; a breached guardrail warns, the writer may "
    "not recommend its arm, and an undecided guardrail is not known to be safe."
)


# =============================================================================
# The entry point
# =============================================================================


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


def _candidates(bundle: AnalysisContextBundle) -> list[str]:
    """Every arm on the scoreboard but the control, by variant key, in the scoreboard's order; empty with no control."""
    control = _control_key(bundle)
    if control is None:
        return []
    return [board.variant_key for board in bundle.scoreboard if board.variant_key != control]


def _floor_verdicts(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> dict[str, tuple[FloorVerdict, ...]]:
    """The aspect's floors read on each candidate against the control, each arm by ``arm``'s name, by the candidate's
    variant key; empty where the aspect declares no floor or no control ran."""
    declared = bundle.declared
    control = _control_key(bundle)
    if declared is None or not declared.floors or control is None:
        return {}
    read: dict[str, tuple[FloorVerdict, ...]] = {}
    for key in _candidates(bundle):
        readings = floor_readings(bundle, control, key, check_names=bundle.goal_check_names)
        read[key] = tuple(
            read_floor(floor, readings, control=arm(control), candidate=arm(key)) for floor in declared.floors
        )
    return read


def _question(bundle: AnalysisContextBundle) -> str | None:
    """The question the campaign asks the aspect, in its words, whenever it was asked; None where none is."""
    declared = bundle.declared
    if declared is None or not bundle.question_declared:
        return None
    return declared.question


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
        does a map that leaves out an arm of this page.
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
    :ivar verdict: The verdict alone.
    """

    caveats: list[str]
    verdict: str

    def line(self) -> str:
        """The caveats, then the verdict, as one line: what an analysis is listed by (``AnalysisLine.verdict``).

        :return: The line.
        :rtype: str
        """
        return " ".join([*self.caveats, self.verdict])


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
    names = AspectNames.of(campaign.aspects)
    caveats = [*campaign_warnings(readings, names), *timing_caveats(campaign.aspects)]
    if len(analysis.aspects) > 1:
        verdict = campaign_verdict(readings, [aspect.label for aspect in analysis.aspects], names)
    else:
        aspect = campaign.aspects[0]
        declared = campaign.declaration.aspects[0]
        verdict = aspect_verdict(
            aspect,
            measure_order=declared.design.measure_order if declared.design is not None else [],
            arm_labels=readings.arm_labels(),
        ).text
    return AnalysisLead(caveats=caveats, verdict=verdict)


def aspect_verdict(
    bundle: AnalysisContextBundle, *, measure_order: Sequence[str] = (), arm_labels: Mapping[str, str] | None = None
) -> VerdictLine:
    """The verdict line an aspect's own page opens with, each arm by its variant's name, read without laying the page
    out: what a stored analysis of one aspect is listed by, and what the quick print's first line and the exit line
    print (:func:`~threetears.evals.analysis.report.verdict.verdict_line`).

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param measure_order: The aspect's graders' measures, in the order its design lists them.
    :ptype measure_order: Sequence[str]
    :param arm_labels: The names a surface prints the arms by, by variant key, which replace their variants' names
        (:func:`~threetears.evals.analysis.report.words.reader_arm_labels`, as a listing reads them); None names each
        arm by its variant.
    :ptype arm_labels: Mapping[str, str] | None
    :return: The verdict line.
    :rtype: VerdictLine
    """
    return verdict_line(
        bundle,
        _namer(_arm_names(bundle, labels=arm_labels)),
        case_name=_case_namer(bundle),
        verdicts=_typed_verdicts(bundle, arm_labels=arm_labels),
        order=measure_order,
        evidence=contrast_evidence(bundle),
    )


def contrast_evidence(bundle: AnalysisContextBundle) -> Callable[[Verdict], str | None]:
    """The figures each contrast verdict over ``bundle`` rests on, as the sentences its verdict line says after the
    decision, each arm the subject: what each arm passed of its trials (or its mean, for a reading that is no rate),
    then what the interval allows the arm to be against the control, in points for a rate, over the cases the test
    read. Each figure is the contrasts table's, at its precision; the interval's level is the table's header's. A
    sentence carries at most three figures (:data:`~threetears.evals.analysis.style.MAX_NUMERALS`).

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :return: A verdict's figures, or None where it has no change or interval to state (it is untested, or no
        contrast of the bundle is its).
    :rtype: Callable[[Verdict], str | None]
    """
    surface = bundle_decision_surface(bundle)
    rulers = report_rulers(surface, bundle.measure_catalog)
    tested = {
        (comparison.contrast.variant_key, comparison.name, comparison.reading, family.aspect): comparison
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
    }
    forms = _change_forms(
        rulers, ((name, reading, held.delta, held.interval) for (_, name, reading, _), held in tested.items())
    )

    def figures(verdict: Verdict) -> str | None:
        comparison = tested.get((verdict.variant_key, verdict.name, verdict.reading, verdict.aspect))
        if comparison is None or verdict.control is None or comparison.delta is None or comparison.interval is None:
            return None
        if comparison.control.mean is None or comparison.contrast.mean is None:
            return None
        ruler = rulers.of(comparison.name, comparison.reading)
        form = forms.get((comparison.name, comparison.reading))
        sides = [
            (verdict.control, *_side_reading(bundle, surface, comparison, comparison.control, ruler)),
            (verdict.arm, *_side_reading(bundle, surface, comparison, comparison.contrast, ruler)),
        ]
        said = [_sides_sentence(sides, verdict.heading, comparison.reading)]
        low, high = (end * ruler.factor for end in comparison.interval)
        paired = (
            comparison.test == "paired" and not comparison.control.n_left_out and not comparison.contrast.n_left_out
        )
        allows = interval_words(
            verdict.arm,
            low,
            high,
            amount=lambda value: _amount(value, ruler, form),
            unit=" points" if ruler.unit == PERCENT else "",
            higher_is_better=comparison.higher_is_better,
        )
        if paired:
            said.append(f"Over {comparison.control.n_cases} paired cases, {allows[:1].lower()}{allows[1:]}")
        else:
            said += [f"The test read {_comparison_cases(comparison)} cases.", allows]
        return " ".join(said)

    return figures


def _sides_sentence(sides: Sequence[tuple[str, str, int | None, int | None]], heading: str, reading: str) -> str:
    """Both arms' figures in the reading's own words: the pass rate as what each arm passed (``Of 90 trials each, v1
    passed 73 and v2 88.``); a counted measure as what it was on each arm's trials, in its own verb
    (``Of 90 trials each, Accuracy was right on 90 for v1 and 89 for v2.``), never "passed", which is the pass rate's;
    one sentence per arm where they ran different trials; a reading that is no rate by its means."""
    (control, control_text, control_passed, control_trials), (arm, arm_text, arm_passed, arm_trials) = sides
    if control_passed is None or control_trials is None or arm_passed is None or arm_trials is None:
        return f"The mean is {control_text} on {control} and {arm_text} on {arm}."
    if reading == "trial":
        if control_trials == arm_trials:
            return f"Of {control_trials} trials each, {control} passed {control_passed} and {arm} {arm_passed}."
        return f"Of {control_trials} trials, {control} passed {control_passed}. Of {arm_trials}, {arm} passed {arm_passed}."
    verb = reading_verb(heading)
    if control_trials == arm_trials:
        return f"Of {control_trials} trials each, {heading} {verb} on {control_passed} for {control} and {arm_passed} for {arm}."
    return (
        f"{heading} {verb} on {control_passed} of {control_trials} trials for {control}. {heading} {verb} on "
        f"{arm_passed} of {arm_trials} for {arm}."
    )


def _amount(value: float, ruler: MeasureRuler, form: MoneyForm | None) -> str:
    """A change's size as a sentence states it, unsigned, on the measure's ruler as the contrasts table prints it:
    points bare (the sentence names them once), money in its form, any other unit beside the number."""
    if form is not None and is_money(ruler.unit):
        return form.spell(abs(value))
    unit = delta_unit(ruler.unit)
    number = number_in(abs(value), unit)
    return number if ruler.unit == PERCENT or not unit else f"{number} {unit}"


def _side_reading(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    comparison: FamilyComparison,
    side: ComparedCell,
    ruler: MeasureRuler,
) -> tuple[str, int | None, int | None]:
    """One arm's figure on a contrast's reading as the contrasts table prints it, without its own interval, and for a
    rate the trials it passed and the trials it is over: the pass rate's from the scoreboard, a pass/fail measure's
    from the arm's cell; None for each where no count is read."""
    _, figure = _side_text(surface, comparison, side, ruler)
    text = _mean_text(figure if figure is not None else side.mean or 0.0, ruler)
    if ruler.unit != PERCENT:
        return text, None, None
    if comparison.reading == "trial":
        board = next((row for row in bundle.scoreboard if row.variant_key == side.variant_key), None)
        return (text, None, None) if board is None else (text, board.passed, board.usable)
    described = bundle.measure_catalog.get(comparison.name) or METRIC_DESCRIPTORS.get(comparison.name)
    # A share of trials: a pass/fail check, a boolean measure, or a classifier's accuracy (its trials' label match).
    counted = described is not None and (
        described.pass_fail or described.data_type == "boolean" or comparison.name == ACCURACY_MEASURE
    )
    if comparison.reading != "measure" or not counted:
        return text, None, None
    try:
        resolved = resolve_reading(
            surface, cell_ref(side.variant_key, side.apparatus_class_id), comparison.name, comparison.reading
        )
    except UnresolvableReference:
        return text, None, None
    if resolved.n <= 0 or figure != resolved.mean:
        return text, None, None
    return text, round(resolved.mean * resolved.n), resolved.n


def _typed_verdicts(bundle: AnalysisContextBundle, *, arm_labels: Mapping[str, str] | None = None) -> list[Verdict]:
    """The guardrails' and the contrasts' typed verdicts over one bundle, each arm by its variant's name (or by
    ``arm_labels``, where a surface names it otherwise), as its page words them: what its verdict line prints, read
    without laying the page out."""
    surface = bundle_decision_surface(bundle)
    rulers = report_rulers(surface, bundle.measure_catalog)
    cell_name = _cell_namer(surface, _arm_names(bundle, labels=arm_labels))
    guardrails = guardrail_verdicts(surface, bundle.guardrails, cell_name, rulers, check_names=bundle.goal_check_names)
    return [*guardrails, *contrast_verdicts(bundle, surface, arm_labels=arm_labels)]


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
    names = _arm_names(bundle, labels=arm_labels)
    arm = _namer(names)
    cell_name = _cell_namer(surface, names)
    charts = _ChartLedger(bundle, arm)
    guardrails_read = guardrail_verdicts(
        surface, bundle.guardrails, cell_name, rulers, check_names=bundle.goal_check_names
    )
    bars_read = bar_verdicts(surface, cell_name, check_names=bundle.goal_check_names)
    contrasts_read = contrast_verdicts(bundle, surface, arm_labels=arm_labels)
    # The readings no declared question asked about are said with what was not tested, after what code says there.
    noted = _question_blocks(bundle)
    contrasts = _contrast_blocks(bundle, surface, contrasts_read, guardrails_read, charts, names, rulers)
    contrasts += _bar_blocks(surface, bars_read, rulers)
    notes = [block for block in contrasts if block.section == "methods"]
    contrasts = [block for block in contrasts if block.section != "methods"]
    if (not_charted := charts.sentence()) is not None:
        notes.append(_note(not_charted, "charts"))
    name = source.campaign_name or f"Campaign {source.campaign_id}"
    verdict = verdict_line(
        bundle,
        arm,
        case_name=_case_namer(bundle),
        verdicts=[*guardrails_read, *contrasts_read],
        order=measure_order,
        evidence=contrast_evidence(bundle),
    )
    blocks: list[ReportBlock] = [
        # Under the title: the question and the arms, the verdict, then what warns and each report-wide caveat once.
        *_heading_blocks(bundle, names, source),
        DisclosureBlock(
            section="header",
            source="verdict",
            text=verdict.text,
            tone=DisclosureBlock.verdict_tone(verdict.rule),
            # A verdict that offers leads links the word that says why they are not decisions: the exploratory rule's
            # line, and the too-small rule's where no question was declared, which is when it leads "Exploratory:".
            term=(
                "exploratory"
                if verdict.rule == "exploratory" or (verdict.rule == "too_small" and not bundle.question_declared)
                else CANNOT_TELL_APART
                if "cannot tell" in verdict.text
                else None
            ),
        ),
        *_floor_warning_blocks(bundle, arm),
        *caveat_blocks([bundle], section="header"),
        *_scoreboard_blocks(bundle, arm, _short_names(bundle, names, arm_labels or {}, arm_short_names or {}), tags),
        *_decision_blocks(bundle, arm),
        *contrasts,
        *_case_blocks(bundle, arm),
        *_trial_blocks(bundle, arm, records=trials == "all"),
        *_not_tested_blocks(bundle, surface, arm, also=noted),
        *_method_blocks(
            bundle,
            surface,
            arm,
            names,
            source=source,
            writer=writer,
            left_disclosures=notes,
        ),
        # When each criterion set after its readings was set: the detail the first screen's one line points at.
        *aspect_provenance_blocks(bundle),
    ]
    # In reading order, each section's blocks in the order laid out (a code-only page's unasked readings are said
    # with what was not tested, though they are laid out with the questions).
    order = list(SECTION_TITLES)
    blocks.sort(key=lambda block: order.index(block.section))
    blocks = [_in_one_form(block) for block in _anchored(blocks)]
    return Report(
        title=_title(bundle, name, arm, arm_short_names or {}),
        source=source,
        blocks=blocks,
        verdicts=[*guardrails_read, *bars_read, *contrasts_read],
        bundle=bundle,
    )


def _in_one_form(block: ReportBlock) -> ReportBlock:
    """A table with each column's money in one form chosen from the column's own values, by the one owner of money's
    scale (:func:`~threetears.evals.analysis.report.blocks.money_columns`, #993); any other block as it is."""
    if not isinstance(block, TableBlock):
        return block
    # The code-only report's money in a table is drawn per a count its heading names, so it never carries the mark.
    return block.model_copy(update={"rows": money_columns(block.rows, marked=False)})


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


# =============================================================================
# Screen 1: the title, the arms, the verdict and the scoreboard
# =============================================================================


def _control_key(bundle: AnalysisContextBundle) -> str | None:
    """The control's variant key, when it ran."""
    control = bundle.design.control_arm.variant_key if bundle.design.control_arm is not None else None
    return control if any(board.variant_key == control for board in bundle.scoreboard) else None


def _arm_names(
    bundle: AnalysisContextBundle,
    index: Sequence[VariantIndexEntry] | None = None,
    *,
    labels: Mapping[str, str] | None = None,
) -> dict[str, ArmName]:
    """Every arm's name, given label first; an arm whose label another shares is told apart by its digest.

    ``index`` names the arms of another variant index (a stored analysis's own) by the same rule; the bundle's when
    None. ``labels`` are a campaign's own labels for its arms, by variant key, which replace their variants' labels.
    """
    displays = arm_names(bundle.variant_index if index is None else index)
    control = _control_key(bundle)
    names = {key: arm_name(display, control=key == control) for key, display in displays.items()}
    # A name spelled from the arm's levers is printed by the one display-name rule, so every surface reads it alike.
    names = {key: replace(name, label=arm_display_name(name.label, name.levers)) for key, name in names.items()}
    if labels:
        # A campaign's own labels, where its page is being built: the label is the name, the model stays beside it.
        names = {
            key: replace(name, label=labels[key], given=True, levers=()) if key in labels else name
            for key, name in names.items()
        }
    taken = [name.label for name in names.values()]
    return {
        key: name
        if taken.count(name.label) == 1
        else ArmName(label=f"{name.label} ({short_digest(key)})", model=name.model, control=name.control)
        for key, name in names.items()
    }


def _namer(names: dict[str, ArmName]) -> Callable[[str], str]:
    """Name an arm by its variant key as every table of the report names it: by its label, never as a digest."""

    def name(variant_key: str) -> str:
        if variant_key in names:
            return names[variant_key].label
        return f"unplaced arm {short_digest(variant_key)}"

    return name


def _title(bundle: AnalysisContextBundle, name: str, arm: Callable[[str], str], short_names: Mapping[str, str]) -> str:
    """The title: what the campaign compares and on what, with its name as a qualifier
    (:func:`~threetears.evals.analysis.campaign_title.named_title`), over the arms the page names and the one aspect
    this bundle reads. A declared question is never in the title: it is quoted whole on its own line
    (:func:`_question_line`).
    """
    control = _control_key(bundle)

    def printed(key: str) -> str:
        return short_names.get(key) or arm(key)

    return named_title(
        name,
        control=[printed(control)] if control is not None else [],
        candidates=[printed(board.variant_key) for board in bundle.scoreboard if board.variant_key != control],
        aspects=[(bundle.aspect, bundle.short_name)],
    )


#: A first word that is a plain word, which a title may capitalize; an identifier (``queue_v5``, ``v4-flash``) is not.
_PLAIN_WORD = re.compile(r"[a-z]+(?:['-][a-z]+)*(?=[\s,;:?!.]|$)")


def _capitalized(title: str) -> str:
    """A title starting with a capital where its first word is a plain word; an identifier is left as written."""
    return title[:1].upper() + title[1:] if _PLAIN_WORD.match(title) else title


def _heading_blocks(
    bundle: AnalysisContextBundle, names: dict[str, ArmName], source: ReportSource
) -> list[ReportBlock]:
    """The first screen's lines between the title and the verdict.

    With a question declared, the question quoted on its own line; then the subhead carrying what the question need
    not: the arms line, each arm with its version. When its criteria were set is a caveat under the verdict
    (:func:`~threetears.evals.analysis.report.campaign.provenance.caveat_blocks`), said once.
    """
    arms = DisclosureBlock(section="header", source="arms", text=_arms_line(bundle, names, source))
    asked = _question_line(bundle)
    basis = [] if asked is None else [DisclosureBlock(section="header", source="basis", text=asked)]
    return [*basis, arms]


def _floor_warning_blocks(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[ReportBlock]:
    """One line per candidate breaking a floor, under the verdict, in the words a campaign's page warns in
    (:func:`~threetears.evals.analysis.report.campaign.warnings.floor_line`): what it means for the candidate, never the
    rate or the interval, which the Floors section reads clause by clause. A floor held or unclear says nothing here."""
    blocks: list[ReportBlock] = []
    for key, read in _floor_verdicts(bundle, arm).items():
        broken = [verdict.floor for verdict in read if verdict.outcome == "broken"]
        if broken:
            margins = [floor.clause.max_points if isinstance(floor.clause, RateDrop) else None for floor in broken]
            text = floor_line(arm(key), margins, aspect_name(bundle))
            blocks.append(DisclosureBlock(section="header", source="floor", text=text))
    return blocks


def _question_line(bundle: AnalysisContextBundle) -> str | None:
    """Under the title: the question declared, quoted whole and alone on its line — the contract the
    verdict reads, never paraphrased, its first plain word capitalized as a sentence starts; None where none was
    declared. The arms' versions are the subhead's (:func:`_heading_blocks`), so a question need not carry them.
    """
    question = _question(bundle)
    if question is None:
        return None
    return f'"{_capitalized(" ".join(question.split()))}"'


def _arms_line(bundle: AnalysisContextBundle, names: dict[str, ArmName], source: ReportSource) -> str:
    """The arms line: each arm named in full, once, the control first and marked, then the date, then the design.

    An arm named by its model is named by the version that answered it where the responses named one model other
    than the one asked for (``arm_served_models``, the served-models check's own reading): what ran, not the alias,
    with the label every table calls it by in brackets, ``gpt-4o-2024-08-06 (gpt-4o)``.
    """
    control = _control_key(bundle)
    keys = [board.variant_key for board in bundle.scoreboard]
    ordered = ([control] if control is not None else []) + [key for key in keys if key != control]
    served = {
        reading.variant_key: humanize("model", reading.served_models[0])
        for reading in bundle.arm_served_models
        if reading.state == "one" and len(reading.served_models) == 1
    }
    versioned = {
        key: replace(name, model=served[key], given=True)
        if name.model is not None and key in served and served[key] != humanize("model", name.model)
        else name
        for key, name in names.items()
    }
    arms = " vs ".join(versioned[key].first() if key in versioned else _namer(names)(key) for key in ordered)
    return " · ".join(
        part for part in (arms or "no arm ran", _date(source.generated_at), _design_words(bundle)) if part
    )


def _date(stamp: str) -> str:
    """An ISO-8601 instant as the page dates it (``14 Oct 2026``), or as given when it does not parse."""
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        # NOSILENT: an unparseable stamp is printed as stored rather than dropped or reformatted
        return stamp
    return f"{moment.day} {moment:%b %Y}"


def _dollars(amount: float | None, ruler: MeasureRuler) -> str | None:
    """A cost as the scoreboard prints it, on its ruler and in its one form (the one owner of money's scale, #993):
    ``$3.09`` per 1,000 replies beside every other arm's; None where none was measured."""
    if amount is None:
        return None
    if ruler.money is None:
        raise ValueError("a cost is spelled on a money ruler, and this one holds no form")
    return ruler.money.spell(amount * ruler.factor)


def _cost_ruler(bundle: AnalysisContextBundle) -> MeasureRuler:
    """The ruler every cost per reply on the page is stated on, chosen over the scoreboard's arms."""
    return reply_cost_ruler(board.reply_cost_usd for board in bundle.scoreboard)


def _per_reply(bundle: AnalysisContextBundle | None) -> str:
    """The words a sentence says a cost per reply is per: ``per reply``, or ``per 1,000 replies`` on that ruler."""
    per = _cost_ruler(bundle).per if bundle is not None else ""
    return per or "per reply"


def _short_names(
    bundle: AnalysisContextBundle,
    names: Mapping[str, ArmName],
    given: Mapping[str, str],
    declared: Mapping[str, str],
) -> dict[str, str]:
    """Each arm's short name, for a chart's tight places (GENERATOR.md §6.1): the short name the campaign declares for
    it, as declared; else the label the campaign gives it; the control's ``control``; else only the levels where it
    differs from the control, the model first and trimmed of the prefix it shares with the control's. With no control,
    or nothing that differs, an arm's short name is its name, and so is it where that name is no longer than the short
    one (a quick run's ``v4-flash``, never ``candidate v4-flash``). Declared names are unique, which the campaign
    refuses otherwise, so only a computed one can collide: one that agrees with any other arm's short name is named in
    full, so no two read alike."""
    control = _control_key(bundle)
    entries = {entry.variant_key: entry for entry in bundle.variant_index}
    distinguishing = distinguishing_axes(bundle.variant_index)
    base = {level.axis_id: level for level in arm_levels(entries.get(control) if control else None, distinguishing)}
    shorts: dict[str, str] = {}
    for key, name in names.items():
        if key in declared:
            shorts[key] = declared[key]
            continue
        if key in given:
            shorts[key] = given[key]
        elif key == control:
            shorts[key] = "control"
        elif control is None:
            shorts[key] = name.label
        else:
            differing = [
                level
                for level in arm_levels(entries.get(key), distinguishing)
                if level.axis_id not in base or base[level.axis_id].content_hash != level.content_hash
            ]
            ordered = sorted(differing, key=lambda level: level.axis_id != CANDIDATE_MODEL_LEVER)
            trimmed = [level.axis_id == CANDIDATE_MODEL_LEVER and level.axis_id in base for level in ordered]
            parts = [
                _trimmed(level.display, base[level.axis_id].display)
                if cut
                else f"{lever_reader_name(level.axis_id)} {elide_level(level.display)}"
                for level, cut in zip(ordered, trimmed, strict=True)
            ]
            # Computed from the levers it differs on, so printed by the one display-name rule; a model trimmed of the
            # control's prefix is a model's name, which no lever's reader name opens.
            levers = [level.axis_id for level, cut in zip(ordered, trimmed, strict=True) if not cut]
            shorts[key] = arm_display_name(" · ".join(parts), levers) or name.label
        if len(name.label) <= len(shorts[key]):
            shorts[key] = name.label
    taken = list(shorts.values())
    return {
        key: short if key in declared or taken.count(short) == 1 else names[key].label for key, short in shorts.items()
    }


#: What separates the parts of a model's name, where a shared prefix may be cut.
_MODEL_SEPARATORS = frozenset("-_ .")


def _trimmed(model: str, base: str) -> str:
    """``model`` without the prefix it shares with ``base``, cut at a separator, as GENERATOR.md §6.1 trims a
    candidate's model against the control's (``Meridian-4.1-flash`` against ``Meridian-4.1`` is ``flash``); the whole
    model where under four characters are shared or nothing is left."""
    shared = 0
    while shared < min(len(model), len(base)) and model[shared] == base[shared]:
        shared += 1
    if not (shared == len(base) and shared < len(model) and model[shared] in _MODEL_SEPARATORS):
        while shared > 0 and model[shared - 1] not in _MODEL_SEPARATORS:
            shared -= 1
    rest = model[shared:].lstrip("".join(_MODEL_SEPARATORS))
    return rest if shared >= 4 and rest else model


def _scoreboard_blocks(
    bundle: AnalysisContextBundle,
    arm: Callable[[str], str],
    shorts: Mapping[str, str],
    given_tags: Mapping[str, str] | None = None,
) -> list[ReportBlock]:
    """The scoreboard table, its outcome bar, then only what changes how the verdict is read, in normal weight.

    The outcome bar tags each arm by ``given_tags``, a campaign's, where they tag every arm of the scoreboard, and
    else by this page's own order.

    A fact that changes how the verdict should be read belongs on the first screen, and nothing else does: the case
    floor, then an arm whose every trial failed. Everything else that qualifies the table is a
    step down: what was not charted and where a cost is a floor are in the methods; a failure cause the scoreboard
    counts apart from checks and the judge heads its own column where it is the only one, and the outcome bar names
    each where there are several.
    """
    boards = bundle.scoreboard
    if not boards:
        return [
            DisclosureBlock(
                section="scoreboard", source="runs", text="No trial ran: the campaign resolved no result in this scope."
            )
        ]
    control = _control_key(bundle)
    keys = [board.variant_key for board in boards]
    tags = (
        dict(given_tags)
        if given_tags is not None and all(key in given_tags for key in keys)
        else arm_tags(keys, control)
    )
    earlier = bundle.design.earlier_control
    lost_any = any(board.lost for board in boards)
    other_causes = sorted({cause for board in boards for cause in board.failed if cause not in ("check", "judged")})
    cost = _cost_ruler(bundle)
    rows: list[dict[str, CellValue]] = []
    for board in boards:
        row: dict[str, CellValue] = {
            "arm": ("▸ " if board.variant_key == control else "")
            + arm(board.variant_key)
            + (f" ({earlier_control_mark(earlier)})" if earlier and board.variant_key == earlier.variant_key else ""),
            "ran": board.ran,
            "usable": board.usable,
            "passed": board.passed,
            "check": board.failed.get("check", 0),
            "judged": board.failed.get("judged", 0),
            # A number on the cost's ruler, not its text: the page right-aligns a column of numbers, and the cell
            # spells it with its currency sign (``$3.092``) the way every money cell does.
            "cost": None if board.reply_cost_usd is None else board.reply_cost_usd * cost.factor,
        }
        if lost_any:
            row["lost"] = sum(board.lost.values())
        if other_causes:
            row["other"] = sum(board.failed.get(cause, 0) for cause in other_causes)
        rows.append(row)
    columns = [
        # The ▸ marking the control is defined where the glossary defines the control.
        TableColumn(key="arm", header="Arm", term="control" if control is not None else None),
        TableColumn(key="ran", header="Ran", term="ran"),
        *([TableColumn(key="lost", header="Lost", term="lost")] if lost_any else []),
        TableColumn(key="usable", header="Usable", term="usable"),
        TableColumn(key="passed", header="Passed", term="passed"),
        TableColumn(key="check", header="Failed by check", term="failed by check"),
        TableColumn(key="judged", header="Failed by judge", term="failed by judge"),
        *([TableColumn(key="other", header=_other_failures_header(other_causes))] if other_causes else []),
        TableColumn(key="cost", header=f"Cost {_per_reply(bundle)}", term="cost per reply"),
    ]
    rows = [{column.key: row[column.key] for column in columns} for row in rows]
    surface = bundle_decision_surface(bundle)
    cells = [
        {
            **_scoreboard_cells(row),
            "cost": _arm_cost_at(_scoreboard_cells(row)["cost"], surface, board.variant_key, board.reply_cost_usd),
        }
        for row, board in zip(rows, boards, strict=True)
    ]
    blocks: list[ReportBlock] = [
        TableBlock(
            section="scoreboard",
            name="scoreboard",
            title="Scoreboard",
            columns=columns,
            rows=cells,
            order="the control first, then the other arms by name",
            total_rows=len(rows),
        ),
        ChartBlock(
            section="scoreboard",
            chart="outcome_bar",
            title="Outcomes",
            bars=[_outcome_bar(board, arm, tags[board.variant_key], shorts.get(board.variant_key)) for board in boards],
        ),
    ]
    blocks += [
        DisclosureBlock(section="scoreboard", source="caveat", text=line, term="case floor")
        for line in caveat_lines(bundle, arm)
    ]
    if bundle.all_failed_cells:
        every_failed = list(dict.fromkeys(arm(cell.variant_key) for cell in bundle.all_failed_cells))
        blocks.append(
            DisclosureBlock(
                section="scoreboard",
                source="runs",
                text=(
                    f"Every trial of {', '.join(every_failed)} failed: the model refused or errored on every call. "
                    "It took no turn, so it has no cost or latency to read, and every failure counts against it."
                ),
            )
        )
    return blocks


def _arm_cost_at(cell: TableCell, surface: DecisionSurface, variant_key: str, figure: float | None) -> TableCell:
    """An arm's cost per reply, carrying the id a finding citing that cost lands on where the arm ran under one test
    setup, so the arm is one cell, and the figure is that cell's own (:func:`_at`)."""
    held = [cell for cell in surface.cells if cell.variant_key == variant_key]
    if len(held) != 1:
        return cell
    return _at(cell, surface, held[0], FLOOR_COST_MEASURE, "measure", figure)


def _scoreboard_cells(row: Mapping[str, CellValue]) -> dict[str, TableCell]:
    """A scoreboard row's cells: counts and words as they are, the cost carrying money's unit so it prints as money."""
    return {
        key: TableCell(value=value, unit=MONEY if key == "cost" and isinstance(value, float) else "")
        for key, value in row.items()
    }


def _other_failures_header(causes: Sequence[str]) -> str:
    """The header over failures the scoreboard counts apart from checks and the judge: the one cause's own words
    (``Model failed``), or ``Failed otherwise`` over several, which the outcome bar then names one by one."""
    if len(causes) == 1:
        words = cause_short(causes[0])
        return words[:1].upper() + words[1:]
    return "Failed otherwise"


def _unpriced_cost_line(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str | None:
    """Where a trial's spend went unpriced, the arm's cost is a floor; None where every trial was priced."""
    unpriced = [board for board in bundle.scoreboard if board.n_cost_unpriced]
    if not unpriced:
        return None
    return (
        "The cost is a floor where a trial's spend went unpriced: "
        + "; ".join(f"{arm(board.variant_key)}, {board.n_cost_unpriced} of {board.ran}" for board in unpriced)
        + "."
    )


def _outcome_bar(board: ArmScoreboard, arm: Callable[[str], str], tag: str, short: str | None) -> OutcomeBar:
    """One arm's census as its stacked bar, tagged and short-named as every chart of the page tags and names it:
    passed, then each failure cause, then each loss cause."""
    segments = [OutcomeSegment(state="passed", label="passed", count=board.passed)] if board.passed else []
    segments += [
        OutcomeSegment(state="failed", label=f"failed: {cause_short(cause)}", count=count)
        for cause, count in sorted(board.failed.items(), key=lambda item: (-item[1], item[0]))
    ]
    segments += [
        OutcomeSegment(state="lost", label=f"lost: {cause_short(cause)}", count=count)
        for cause, count in sorted(board.lost.items(), key=lambda item: (-item[1], item[0]))
    ]
    return OutcomeBar(arm=arm(board.variant_key), segments=segments, tag=tag, short_name=short)


def _design_words(bundle: AnalysisContextBundle) -> str:
    """The design at the end of the arms line: ``30 cases × 3 repeats``; where the arms differ, each range its own segment
    (``4 to 6 cases per arm · 2 to 3 repeats``), so no segment carries more figures than a sentence may. Empty when
    no arm ran; the cases alone where no repeat count was recorded."""
    cases = sorted({board.cases for board in bundle.scoreboard})
    repeats = sorted(
        {
            value
            for cell in bundle.cell_measures
            for value in (cell.repeats_per_case_min, cell.repeats_per_case_max)
            if value is not None
        }
    )
    if not cases:
        return ""
    case_words = (
        f"{cases[0]} case{'' if cases[0] == 1 else 's'}"
        if len(cases) == 1
        else f"{cases[0]} to {cases[-1]} cases per arm"
    )
    if not repeats:
        return case_words
    repeat_words = (
        f"{repeats[0]} repeat{'' if repeats[0] == 1 else 's'}"
        if len(repeats) == 1
        else f"{repeats[0]} to {repeats[-1]} repeats"
    )
    if len(cases) == 1 and len(repeats) == 1:
        return f"{case_words} × {repeat_words}"
    return f"{case_words} · {repeat_words}"


def _decision_blocks(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[ReportBlock]:
    """The aspect's floors, each on each candidate against the control, as a table, never prose.

    Each floor's row opens with the floor as a sentence naming its arm. The control heads its column by name, as does
    a lone other arm; with several, an ``Arm`` column names each row's and its figures are under ``Value``. A rate
    floor has a row per rate it reads, each with that rate's own status and reason, so a status never sits beside a
    reading it is not about. A broken floor is also a line under the verdict (:func:`_floor_warning_blocks`); here it
    reads as plainly as a held one.

    An aspect with no floor declared has no floor block at all: the page says what it holds, never what it lacks.
    Where a floor is unclear, a ``To show the floors hold``
    table follows: for each unclear floor, what showing it holds needs if the arms do not differ and if they differ as
    observed, each count with its cost
    (:func:`~threetears.evals.analysis.report.decision.floor_projection`), then one line saying how each count was
    found and that it is about, what repeats do, and, where holding needs more cases than the run used on every side,
    that it does. A floor broken or held projects nothing.
    """
    control = _control_key(bundle)
    read = _floor_verdicts(bundle, arm)
    declared = bundle.declared
    if declared is not None and declared.unread_without_control:
        # A margin or floors stored on an aspect of a campaign naming no control are read against nothing: said once,
        # where the floors would have been.
        return [DisclosureBlock(section="decision", source="floor", text=declared.unread_without_control)]
    if control is None or not _candidates(bundle):
        return []
    if not read:
        return []
    cost = _cost_ruler(bundle)
    others = list(read)
    rows: list[dict[str, CellValue]] = []
    projected: list[tuple[FloorVerdict, str, SampleProjection]] = []
    joined: list[SampleProjection] = []
    pointed = False
    per_case = case_cost_usd(bundle)
    for key, verdicts in read.items():
        if (together := floors_projection(verdicts, per_case_usd=per_case)) is not None:
            # Only an unclear floor projects: holding is what more cases can show.
            joined.append(together)
            projected += [
                (verdict, arm(key), projection)
                for verdict in verdicts
                if (projection := floor_projection(verdict, per_case_usd=per_case)) is not None
            ]
        for verdict in verdicts:
            shown: list[tuple[str, str | None, str | None, str]] = []
            readings = verdict.readings
            if isinstance(verdict.floor.clause, CostLower):
                shown = [
                    (
                        f"cost {_per_reply(bundle)}",
                        _dollars(readings.control_cost if readings else None, cost),
                        _dollars(readings.candidate_cost if readings else None, cost),
                        f"{verdict.outcome}: {verdict.reason}",
                    )
                ]
            else:
                # A reading past the margin on its point estimate alone says so: the glossary defines the term.
                pointed = pointed or any("point estimate" in rate.reason for rate in verdict.rates)
                # Each rate its own status, so a row's status is about the reading on that row and no other.
                shown = [
                    (
                        PASS_RATE if rate.rate.measure is None else rate.rate.name,
                        f"{rate.rate.control_passed} of {rate.rate.control_n}",
                        f"{rate.rate.candidate_passed} of {rate.rate.candidate_n}",
                        f"{rate.outcome}: {rate.reason}",
                    )
                    for rate in verdict.rates
                ]
            for index, (reading, ours, theirs, outcome) in enumerate(
                shown or [("—", None, None, f"{verdict.outcome}: {verdict.reason}")]
            ):
                rows.append(
                    {
                        "floor": verdict.statement if index == 0 else None,
                        "arm": arm(key),
                        "reading": reading,
                        "control": ours,
                        "candidate": theirs,
                        "outcome": outcome,
                    }
                )
    # A lone other arm heads its own column by name; several share a "Value" column, each row naming its arm.
    single = len(others) == 1
    columns = [
        TableColumn(key="floor", header="Floor", term="floor"),
        *([] if single else [TableColumn(key="arm", header="Arm")]),
        TableColumn(key="reading", header="Reading", term="reading"),
        TableColumn(key="control", header=arm(control)),
        TableColumn(key="candidate", header=arm(others[0]) if single else "Value"),
        TableColumn(key="outcome", header="Status", term="floor status"),
    ]
    return [
        TableBlock(
            section="decision",
            name="decision",
            title=f"Floors on {bundle.aspect}",
            columns=columns,
            rows=[table_row({column.key: row[column.key] for column in columns}) for row in rows],
            order="floor by floor, each arm against the control",
            total_rows=len(rows),
        ),
        DisclosureBlock(
            section="decision",
            source="floor",
            text=FLOORS_WARN_ONLY,
            term="point estimate" if pointed else None,
        ),
        *_projection_blocks(bundle, projected, joined, single=single),
    ]


#: What the Decision section says once, under the floors: a floor warns and decides nothing, and which interval it
#: reads, so a floor unclear beside a contrast that is not separated reads as the two rulers it is.
FLOORS_WARN_ONLY = (
    "A floor warns and never decides: a broken one warns on the first screen. A floor reads its own interval, "
    "uncorrected, which can differ from the contrasts table's."
)


def _projection_blocks(
    bundle: AnalysisContextBundle,
    projected: Sequence[tuple[FloorVerdict, str, SampleProjection]],
    joined: Sequence[SampleProjection],
    *,
    single: bool,
) -> list[ReportBlock]:
    """The ``To show the floors hold`` table, one row per unclear floor and arm, and the line under it, which reads each arm's
    projection over its floors (``joined``); empty with none. A cost floor's no-difference cell is blank: it has no
    margin, and the line says so once."""
    if not projected:
        return []
    columns = [
        TableColumn(key="floor", header="Floor", term="floor"),
        *([] if single else [TableColumn(key="arm", header="Arm")]),
        TableColumn(key="none", header="If no difference"),
        TableColumn(key="observed", header="If as observed"),
    ]
    # Every projected cost the table states is one set of totals, in one form (#993).
    costs = money_ruler(
        (
            projected_usd_value(cost)
            for verdict, _, projection in projected
            for need in (
                (projection.if_observed,)
                if isinstance(verdict.floor.clause, CostLower)
                else (projection.if_none, projection.if_observed)
            )
            if (cost := projection.cost(need)) is not None
        ),
        per=0,
    ).form
    rows: list[dict[str, CellValue]] = [
        {
            "floor": verdict.statement,
            "arm": name,
            "none": None
            if isinstance(verdict.floor.clause, CostLower)
            else projected_cell(projection, projection.if_none, costs),
            "observed": projected_cell(projection, projection.if_observed, costs),
        }
        for verdict, name, projection in projected
    ]
    run_cases = max((board.cases for board in bundle.scoreboard), default=0)
    costed = any(isinstance(verdict.floor.clause, CostLower) for verdict, *_ in projected)
    return [
        TableBlock(
            section="decision",
            name="projection",
            title="To show the floors hold",
            columns=columns,
            rows=[table_row({column.key: row[column.key] for column in columns}) for row in rows],
            order="floor by floor, each arm against the control",
            total_rows=len(rows),
        ),
        DisclosureBlock(
            section="decision",
            source="floor",
            text=projection_note(joined, run_cases=run_cases, cost_clause=costed),
        ),
    ]


class _ChartLedger:
    """Which intervals were drawn in the tables, and why each left out was: the one sentence in the methods.

    An interval is drawn beside a figure only when its arm has :data:`MIN_CASES` usable cases and one was computed;
    the figure stands alone otherwise, no stub. Every interval left out is counted here, so the methods say once, in
    one sentence, what was not drawn and why.
    """

    def __init__(self, bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> None:
        self._arm = arm
        self._usable = {board.variant_key: board.usable_cases for board in bundle.scoreboard}
        self._small: dict[str, int] = {}
        self._no_interval: list[str] = []

    def drawable(self, variant_key: str, n_cases: int | None, interval: tuple[float, float] | None, what: str) -> bool:
        """Whether a row of ``variant_key``'s may be drawn; one that may not is counted for the sentence."""
        usable = min(self._usable.get(variant_key, 0), n_cases if n_cases is not None else 0)
        if usable < MIN_CASES:
            self._small[variant_key] = min(self._small.get(variant_key, usable), usable)
            return False
        if interval is None:
            self.left_out(what)
            return False
        return True

    def left_out(self, what: str) -> None:
        """Count a reading no row of which could be drawn, for want of an interval."""
        if what not in self._no_interval:
            self._no_interval.append(what)

    def sentence(self) -> str | None:
        """Every interval not drawn, in one sentence; None when nothing was left out, or no arm could have one drawn
        at all (the caveat under the scoreboard says so). Read after the contrasts are laid out."""
        if all(usable < MIN_CASES for usable in self._usable.values()):
            return None
        parts = []
        if self._small:
            named = "; ".join(
                f"{self._arm(key)}, {usable} usable case{'' if usable == 1 else 's'}"
                for key, usable in sorted(self._small.items(), key=lambda item: self._arm(item[0]))
            )
            parts.append(
                f"No interval drawn for {named}: an interval is estimated only from {MIN_CASES} independent cases."
            )
        if self._no_interval:
            parts.append(f"No interval drawn, none computed: {', '.join(self._no_interval)}.")
        return " ".join(parts) or None


# =============================================================================
# Screen 2: what it means — the unasked readings, then the contrasts against the control
# =============================================================================


def _question_blocks(bundle: AnalysisContextBundle) -> list[ReportBlock]:
    """On a code-only report: the readings the declared question did not ask about."""
    scope = bundle.reading_scope
    blocks: list[ReportBlock] = []
    catalog = bundle.measure_catalog
    exploratory = [
        *(
            measure_name(name, catalog[name].reader_name if name in catalog else None, declared=bundle.check_name(name))
            for name in one_figure_each(scope.exploratory_measures, read=catalog)
        ),
        *(judged_heading(name) for name in scope.exploratory_dimensions),
    ]
    if exploratory and scope.questions_declared:
        blocks.append(_readings_table("unasked", "Unasked readings", [{"reading": name} for name in exploratory]))
    return blocks


def _question_words(bundle: AnalysisContextBundle) -> str:
    """What the methods call the aspect's design: a question declared, or exploratory. When a criterion was set is
    said once, on the first screen."""
    return "question declared" if bundle.question_declared else "exploratory"


def _readings_table(name: str, title: str, rows: list[dict[str, CellValue]]) -> TableBlock:
    """Readings the page lists rather than strings into a sentence, one row each, each named once."""
    columns = [TableColumn(key="reading", header="Reading", term="reading")]
    if any("declare" in row for row in rows):
        columns.append(TableColumn(key="declare", header="Declaring one", term="margin"))
    return TableBlock(
        section="not_tested",
        name=name,
        title=title,
        columns=columns,
        rows=[table_row({column.key: row.get(column.key) for column in columns}) for row in rows],
        order="as the bundle lists them",
        total_rows=len(rows),
    )


def _contrast_blocks(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    contrasts_read: Sequence[Verdict],
    guardrails_read: Sequence[Verdict],
    charts: _ChartLedger,
    names: dict[str, ArmName],
    rulers: Rulers,
) -> list[ReportBlock]:
    """The contrasts against the control, their charts, the guardrails, and the readings by stratum and label.

    Engine text that explains how they were tested goes to the methods' closed ``Disclosures`` fold.
    """
    arm = _namer(names)
    blocks: list[ReportBlock] = []
    blocks.extend(_self_judging_blocks(bundle, surface, arm))
    blocks.extend(_guardrail_standing_blocks(bundle, arm))
    blocks.extend(_comparison_blocks(bundle, surface, contrasts_read, charts, arm, rulers))
    blocks.extend(_readings_blocks(bundle, surface, names, rulers))
    blocks.extend(_guardrail_blocks(surface, bundle.guardrails, guardrails_read, rulers))
    blocks.extend(
        _strata_blocks(surface, names, rulers, check_names=bundle.goal_check_names, stratum_names=bundle.stratum_names)
    )
    blocks.extend(_label_blocks(surface, names, rulers))
    return blocks


def _note(text: str, source: DisclosureSource = "comparisons") -> DisclosureBlock:
    """Engine text for the methods' closed ``Disclosures`` fold."""
    return DisclosureBlock(section="methods", source=source, text=text, fold="Disclosures")


def _self_judging_blocks(
    bundle: AnalysisContextBundle, surface: DecisionSurface, arm: Callable[[str], str]
) -> list[ReportBlock]:
    """A line for each arm whose answers its judge's own model produced: its judged contrasts may favour it."""
    variant_of_run = {run_id: cell.variant_key for cell in surface.cells for run_id in cell.run_ids}
    shared: dict[str, list[str]] = {}
    for summary in bundle.run_summaries:
        variant = variant_of_run.get(summary.run_id)
        if variant is None or not summary.judge_shares_candidate_model:
            continue
        models = shared.setdefault(variant, [])
        models.extend(model for model in summary.judge_shares_candidate_model if model not in models)
    return [
        DisclosureBlock(
            section="contrasts",
            source="apparatus",
            text=self_judging_text(models, f"arm {arm(variant)}")
            + "; a contrast on a judged dimension may favour that arm for that reason alone.",
        )
        for variant, models in sorted(shared.items(), key=lambda item: arm(item[0]))
    ]


def _guardrail_standing_blocks(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[ReportBlock]:
    """A line for each arm a guardrail leaves not shown safe, above the contrasts it might be picked from, in the
    verdict line's own words (:func:`~threetears.evals.analysis.report.verdict.guardrail_lines`)."""
    return [
        DisclosureBlock(section="contrasts", source="guardrails", text=line, term="guardrail")
        for line in guardrail_lines(bundle, arm)
    ]


def _small_n(comparison: FamilyComparison) -> int | None:
    """The fewer of a contrast's two case counts when it is below :data:`MIN_CASES`; None when both reach it."""
    n = min(comparison.control.n_cases, comparison.contrast.n_cases)
    return n if n < MIN_CASES else None


def _comparison_cases(comparison: FamilyComparison) -> str:
    """The cases a contrast's test read, and how many each side ran that it left out."""
    control, contrast = comparison.control, comparison.contrast
    if comparison.test == "paired":
        text = f"{control.n_cases} paired"
    else:
        text = f"{contrast.n_cases} vs {control.n_cases}" + (" unpaired" if comparison.test == "unpaired" else "")
    left_out = [
        f"{count} of the {side}'s"
        for side, count in (("arm", contrast.n_left_out), ("control", control.n_left_out))
        if count
    ]
    if left_out:
        text += f"; {' and '.join(left_out)} left out, not run by the other side"
    return text


def _figure(value: float | None, ruler: MeasureRuler) -> TableCell:
    """One figure of a measure as a table cell: restated onto the measure's ruler and carrying its unit, money in the
    form its ruler chose over every figure of the measure (so two arms' means in one row read alike, #993)."""
    if value is None:
        return TableCell(value=None)
    if ruler.money is not None and is_money(ruler.unit):
        form = ruler.money
        return TableCell(
            value=value * ruler.factor, unit=ruler.unit, decimals=form.decimals, per_thousand=form.per_thousand
        )
    return TableCell(value=value * ruler.factor, unit=ruler.unit)


class _OneCell(Protocol):
    """Anything naming one cell: an arm's variant under one rig."""

    @property
    def variant_key(self) -> str: ...

    @property
    def apparatus_class_id(self) -> str: ...


def _at(
    cell: TableCell, surface: DecisionSurface, at: _OneCell, name: str, reading: str, figure: float | None
) -> TableCell:
    """``cell``, printing ``figure`` for ``name`` read at ``at``'s cell, carrying the id a finding citing that reading
    links to (:func:`~threetears.evals.analysis.report.model.reading_anchor`) where ``figure`` is the cell's own: the
    mean a finding's evidence resolves to (:func:`~threetears.evals.analysis.references.resolve_reading`). A figure
    read over other cases (a paired test's mean over the cases both sides ran, a guardrail's mean of per-case means)
    is another number, and a link from the finding's figure never lands on another number; such a cell is handed back
    as it was.

    :func:`_traced` keeps the first such cell on the page.
    """
    kind: ReadingKind | None = "measure" if reading == "measure" else "judged" if reading == "judged" else None
    if figure is None or cell.value is None or kind is None:
        return cell
    ref = cell_ref(at.variant_key, at.apparatus_class_id)
    try:
        own = resolve_reading(surface, ref, name, kind).mean
    except UnresolvableReference:
        # NOSILENT: the cell holds no figure of its own for this reading, so a finding's figure has nothing to match
        return cell
    if not math.isclose(figure, own, rel_tol=1e-9, abs_tol=1e-12):
        return cell
    return cell.model_copy(update={"anchor": reading_anchor(ref, name, reading)})


def _change(
    delta: float | None,
    ruler: MeasureRuler,
    *,
    interval: tuple[float, float] | None = None,
    margin: float | None = None,
    form: MoneyForm | None = None,
) -> TableCell:
    """A change in a measure as a table cell: on the measure's ruler, in the unit a change in it is stated in.

    Handed the interval on the delta, the cell carries it, with the declared margin, on the same ruler, read against
    zero — what HTML draws beside the figure as the row's glyph; the row's interval column spells it. Handed the form
    the measure's changes read in (:func:`_change_forms`), a change in money is written in it, as its interval is.
    """
    if delta is None:
        return TableCell(value=None)
    unit = delta_unit(ruler.unit)
    if interval is None:
        cell = TableCell(value=delta * ruler.factor, unit=unit)
    else:
        cell = TableCell(
            value=delta * ruler.factor,
            unit=unit,
            low=interval[0] * ruler.factor,
            high=interval[1] * ruler.factor,
            margin=abs(margin) * ruler.factor if margin is not None else None,
            zero_line=True,
        )
    if form is None:
        return cell
    return cell.model_copy(update={"decimals": form.decimals, "per_thousand": form.per_thousand})


def _change_forms(
    rulers: Rulers, changes: Iterable[tuple[str, str, float | None, tuple[float, float] | None]]
) -> dict[tuple[str, str], MoneyForm]:
    """The one form each money measure's changes read in, a change and its interval's ends alike (#993): from every
    change a table states of the measure and the widths of their intervals, on the measure's ruler
    (:meth:`~threetears.evals.kernel.numbers.MoneyForm.for_changes`). A measure of another unit has none.

    :param rulers: The report's rulers.
    :ptype rulers: Rulers
    :param changes: Each change the table states: its measure's name and reading, its delta and its interval, as
        stored.
    :ptype changes: Iterable[tuple[str, str, float | None, tuple[float, float] | None]]
    :return: Each money measure's form, by name and reading.
    :rtype: dict[tuple[str, str], MoneyForm]
    """
    held: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for name, reading, delta, interval in changes:
        ruler = rulers.of(name, reading)
        if ruler.money is None or not is_money(ruler.unit):
            continue
        deltas, widths = held.setdefault((name, reading), ([], []))
        if delta is not None:
            deltas.append(delta * ruler.factor)
        if interval is not None:
            widths.append(abs(interval[1] - interval[0]) * ruler.factor)
    forms: dict[tuple[str, str], MoneyForm] = {}
    for (name, reading), (deltas, widths) in held.items():
        money = rulers.of(name, reading).money
        assert money is not None
        forms[name, reading] = money.for_changes(deltas, widths)
    return forms


def _side_cell(
    surface: DecisionSurface, comparison: FamilyComparison, side: ComparedCell, ruler: MeasureRuler
) -> TableCell:
    """One arm's figure on a contrast's reading (:func:`_side_text`), carrying the id a finding's figure lands on where
    it is the cell's own."""
    text, figure = _side_text(surface, comparison, side, ruler)
    return _at(TableCell(value=text), surface, side, comparison.name, comparison.reading, figure)


def _side_text(
    surface: DecisionSurface, comparison: FamilyComparison, side: ComparedCell, ruler: MeasureRuler
) -> tuple[str | None, float | None]:
    """One arm's figure on a contrast's reading, with its own interval where the cell has one: ``98.9% [88.1%,
    100%]``.

    The figure and its interval are the arm's cell's (:func:`~threetears.evals.analysis.references.resolve_reading`)
    where the test read every case the cell holds, so the pair is one estimate; where cases were left out, or the
    reading is the pass rate, which no cell holds as a measure, the figure is the mean the test read, alone. None
    where the side has no mean.

    :return: The text, and the figure it prints, unscaled.
    """
    if side.mean is None:
        return None, None
    if comparison.reading in ("measure", "judged") and side.n_left_out == 0:
        try:
            resolved = resolve_reading(
                surface, cell_ref(side.variant_key, side.apparatus_class_id), comparison.name, comparison.reading
            )
        except UnresolvableReference:
            resolved = None
        if resolved is not None and resolved.ci_low is not None and resolved.ci_high is not None:
            return _mean_text(resolved.mean, ruler) + _interval(resolved.ci_low, resolved.ci_high, ruler), resolved.mean
    return _mean_text(side.mean, ruler), side.mean


def _interval_cell(low: float, high: float, unit: str) -> str:
    """An interval as ``[low, high]`` with its unit: on each end where the unit sits on the number (``[88.1%,
    100%]``, ``[-$0.01, $0.06]``), once after the bracket otherwise (``[0.45, 0.55] s``)."""
    if unit == PERCENT or is_money(unit):
        return f"[{with_unit(low, unit)}, {with_unit(high, unit)}]"
    return f"[{number_in(low, unit)}, {number_in(high, unit)}]" + (f" {unit}" if unit else "")


def _interval_on_delta(
    interval: tuple[float, float] | None,
    ruler: MeasureRuler,
    level: float | None = None,
    *,
    form: MoneyForm | None = None,
) -> str | None:
    """An interval on a delta, on the measure's ruler in the unit a change is stated in: money's ends in the form its
    change reads in (``form``, :func:`_change_forms`).

    The level it holds at is said once, in the table's header, so a cell carries it only where a caller asks — a
    program's row (:func:`contrast_rows`), which has no header to carry it.
    """
    if interval is None:
        return None
    low, high = (bound * ruler.factor for bound in interval)
    if form is not None and is_money(ruler.unit):
        text = f"[{form.spell(low)}, {form.spell(high)}]"
    else:
        text = _interval_cell(low, high, delta_unit(ruler.unit))
    return text if level is None else f"{text} at {_level_text(level)}"


def _level_text(level: float) -> str:
    """An interval's level as a reader reads it: ``99%``."""
    return with_unit(100 * level, PERCENT)


def _interval_header(levels: set[float]) -> str:
    """The interval column's header, carrying the one level every row holds at; bare where the rows differ."""
    return f"Interval ({_level_text(next(iter(levels)))})" if len(levels) == 1 else "Interval"


def _contrast_margin(bundle: AnalysisContextBundle, comparison: FamilyComparison) -> float | None:
    """The margin a contrast's reading has — its equivalence margin, or the margin no equivalence test could read."""
    if comparison.equivalence_margin is not None:
        return comparison.equivalence_margin
    if comparison.reading != "measure":
        return None
    descriptor = bundle.measure_catalog.get(comparison.name)
    if descriptor is not None and descriptor.materiality_threshold is not None:
        return descriptor.materiality_threshold
    return bundle.run_margins.get(comparison.name)


def _margin_text(margin: float, ruler: MeasureRuler) -> str:
    """A margin as the row's reading is stated: on its measure's one ruler, in the unit a change in it is stated in."""
    return f"±{with_unit(margin * ruler.factor, delta_unit(ruler.unit))}"


def _immaterial_words(margin: float | None, heading: str, ruler: MeasureRuler) -> str:
    """The caveat an immaterial row carries, naming the margin it was read against and the reading it is on."""
    where = (
        "the margin declared on it" if margin is None else f"the margin of {_margin_text(margin, ruler)} on {heading}"
    )
    return f"; immaterial: the observed delta is inside {where}, which does not show the true difference is that small"


def _contrast_reason(comparison: FamilyComparison) -> tuple[VerdictReason, str | None]:
    """Why a contrast's verdict is what it is, as a code and, where there is more to say, in words."""
    if comparison.verdict == "untested":
        return "untestable", comparison.untested_reason
    if comparison.verdict in ("improved", "regressed"):
        return "separated", None
    if comparison.verdict == "equivalent":
        return "inside_margin", None
    if comparison.margin_source is None:
        return "no_margin", comparison.not_separated_reason
    if comparison.equivalence_p_adjusted is None:
        return "margin_untested", comparison.not_separated_reason or comparison.equivalence_untested_reason or (
            "the test was unpaired, and the equivalence test reads only cases both arms ran"
        )
    return "not_inside_margin", comparison.not_separated_reason


def contrast_verdicts(
    bundle: AnalysisContextBundle, surface: DecisionSurface, *, arm_labels: Mapping[str, str] | None = None
) -> list[Verdict]:
    """Each contrast the bundle tested against the control as a typed verdict, in the contrasts table's row order.

    A contrast with fewer than :data:`MIN_CASES` cases on either side is printed ``not tested (n=N)``: its typed
    outcome is the test's, and the words state no difference, as a rate on so few cases states none. ``arm_labels``
    name the arms as :func:`build_code_only_report` does.
    """
    arm = _cell_namer(surface, _arm_names(bundle, labels=arm_labels))
    rulers = report_rulers(surface, bundle.measure_catalog)
    verdicts = []
    for family in bundle.multiple_comparisons.families:
        for comparison in family.comparisons:
            heading = reading_heading(
                surface, comparison.name, comparison.reading, declared=bundle.check_name(comparison.name)
            )
            margin = _contrast_margin(bundle, comparison)
            ruler = rulers.of(comparison.name, comparison.reading)
            reason, detail = _contrast_reason(comparison)
            named = arm(comparison.contrast.variant_key, comparison.contrast.apparatus_class_id)
            control = arm(comparison.control.variant_key, comparison.control.apparatus_class_id)
            if (small := _small_n(comparison)) is not None:
                reason = "too_few_cases"
                detail = f"{small} case{'' if small == 1 else 's'}, under the {MIN_CASES} a rate needs"
                words = f"not tested (n={small})"
            else:
                words = _contrast_words(comparison, (named, control), margin, heading, ruler)
            verdicts.append(
                Verdict(
                    kind="contrast",
                    outcome=comparison.verdict,
                    reason=reason,
                    reason_detail=detail,
                    reading=comparison.reading,
                    name=comparison.name,
                    heading=heading,
                    arm=named,
                    variant_key=comparison.contrast.variant_key,
                    apparatus_class_id=comparison.contrast.apparatus_class_id,
                    control=control,
                    aspect=family.aspect,
                    delta=comparison.delta,
                    interval=comparison.interval,
                    interval_level=family.interval_level if comparison.interval is not None else None,
                    p_adjusted=comparison.p_adjusted,
                    margin=margin,
                    margin_source=comparison.margin_source,
                    materiality=comparison.materiality,
                    words=words,
                )
            )
    return verdicts


def _verdict_cell(verdict: Verdict) -> str:
    """A contrast's verdict as the contrasts table's cell says it: its words, the undecided reading in its short cell
    form (:data:`~threetears.evals.analysis.style.CANNOT_TELL_APART`), since the row's columns name both arms."""
    if verdict.outcome != "not_separated" or verdict.control is None:
        return verdict.words
    sentence = comparison_words("not_separated", verdict.arm, verdict.control)
    return CANNOT_TELL_APART + verdict.words.removeprefix(sentence)


def _contrast_words(
    comparison: FamilyComparison, arms: tuple[str, str], margin: float | None, heading: str, ruler: MeasureRuler
) -> str:
    """A tested contrast's verdict in words: the verdict naming the arm and the control (``arms``) by label, why where
    there is more to say, its margin, materiality."""
    on_runs = ", declared on the runs" if comparison.margin_source == "run" else ""
    return (
        comparison_words(comparison.verdict, *arms, higher_is_better=comparison.higher_is_better)
        + (f" ({comparison.untested_reason})" if comparison.untested_reason else "")
        + (f" ({comparison.not_separated_reason})" if comparison.not_separated_reason else "")
        + (
            f" (margin {_margin_text(margin, ruler)}{on_runs})"
            if comparison.verdict == "equivalent" and margin is not None
            else ""
        )
        + (
            _immaterial_words(margin, heading, ruler)
            if comparison.materiality == "immaterial" and comparison.verdict != "equivalent"
            else ""
        )
    )


def _comparison_blocks(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    verdicts: Sequence[Verdict],
    charts: _ChartLedger,
    arm: Callable[[str], str],
    rulers: Rulers,
) -> list[ReportBlock]:
    """The contrasts that can resolve, as one table of each arm's value beside the control's, with their statistics
    folded away.

    Each row is a reading: the control's figure with its interval, then the arm's, then the change with its interval
    drawn in the row (the ``change`` cell carries it, and HTML draws it as a glyph on one scale per unit), the
    interval as text, the verdict and the cases. Two arms, and each arm's figure heads its own column, so a row reads
    across; more, and each row is one reading on one arm, the control's figure repeated, since a column per arm and a
    change per arm would not stay legible at three or four.

    A contrast with a side under the case floor is never a row: when none can resolve, the section is one line, and
    otherwise one line counts the ones left out. A reading one side carries on no trial is no contrast at all, and the
    methods count it. The level the intervals hold at is said once, in the column's header, where every shown row
    holds at one level; where families differ, each cell says its own.
    """
    comparisons = bundle.multiple_comparisons
    if not comparisons.families:
        if comparisons.withheld:
            return [DisclosureBlock(section="contrasts", source="comparisons", text=comparisons.withheld)]
        return []
    typed = iter(verdicts)
    shown: list[dict[str, TableCell]] = []
    statistics: list[dict[str, TableCell]] = []
    under: list[int] = []
    # The level every shown interval holds at: one, and the header says it; more, and each cell says its own.
    levels = {
        family.interval_level
        for family in comparisons.families
        for comparison in family.comparisons
        if family.interval_level is not None and comparison.interval is not None and _small_n(comparison) is None
    }
    forms = _change_forms(
        rulers,
        (
            (comparison.name, comparison.reading, comparison.delta, comparison.interval)
            for family in comparisons.families
            for comparison in family.comparisons
            if min(comparison.control.n_cases, comparison.contrast.n_cases) > 0 and _small_n(comparison) is None
        ),
    )
    for family in comparisons.families:
        for comparison in family.comparisons:
            verdict = next(typed)
            if min(comparison.control.n_cases, comparison.contrast.n_cases) == 0:
                continue
            ruler = rulers.of(comparison.name, comparison.reading)
            if (small := _small_n(comparison)) is not None:
                under.append(small)
                continue
            heading = ruler.scaled(verdict.heading)
            form = forms.get((comparison.name, comparison.reading))
            interval = _interval_on_delta(
                comparison.interval, ruler, None if len(levels) == 1 else family.interval_level, form=form
            )
            # The interval is drawn in the row only where both arms have the cases and one was computed.
            arm_ok = charts.drawable(
                comparison.contrast.variant_key,
                min(comparison.contrast.n_cases, comparison.control.n_cases),
                comparison.interval if comparison.delta is not None else None,
                heading,
            )
            control_ok = charts.drawable(
                comparison.control.variant_key, comparison.control.n_cases, (0.0, 0.0), heading
            )
            shown.append(
                {
                    "reading": TableCell(value=heading),
                    "contrast": TableCell(value=verdict.arm),
                    "control_label": TableCell(value=verdict.control),
                    "control": _side_cell(surface, comparison, comparison.control, ruler),
                    "arm": _side_cell(surface, comparison, comparison.contrast, ruler),
                    "change": _change(
                        comparison.delta,
                        ruler,
                        interval=comparison.interval if arm_ok and control_ok else None,
                        margin=_contrast_margin(bundle, comparison),
                        form=form,
                    ),
                    "interval": TableCell(value=interval),
                    "verdict": TableCell(value=_verdict_cell(verdict)),
                    "cases": TableCell(value=_comparison_cases(comparison)),
                }
            )
            statistics.append(
                {
                    "question": TableCell(value="Q1" if family.aspect is not None else "campaign-wide"),
                    "reading": TableCell(value=heading),
                    "contrast": TableCell(value=verdict.arm),
                    "control_mean": _figure(comparison.control.mean, ruler),
                    "arm_mean": _figure(comparison.contrast.mean, ruler),
                    "hedges_g": TableCell(value=comparison.hedges_g),
                    "pairing_r": TableCell(
                        value=comparison.pairing_correlation.r if comparison.pairing_correlation is not None else None
                    ),
                    "p_adjusted": TableCell(value=comparison.p_adjusted),
                }
            )
    blocks: list[ReportBlock] = []
    if under and not shown:
        blocks.append(
            DisclosureBlock(
                section="contrasts",
                source="comparisons",
                text=(
                    f"No contrast can resolve: a side has {min(under)} usable case{'' if min(under) == 1 else 's'}; "
                    f"a difference is tested only from {MIN_CASES} independent cases."
                ),
                term="case floor",
            )
        )
    elif under:
        blocks.append(
            DisclosureBlock(
                section="contrasts",
                source="comparisons",
                text=f"{len(under)} more contrasts are not shown: a side has under {MIN_CASES} usable cases.",
                term="case floor",
            )
        )
    if shown:
        control = _control_key(bundle)
        candidates = list(dict.fromkeys(str(row["contrast"].value) for row in shown))
        controls = list(dict.fromkeys(str(row["control_label"].value) for row in shown))
        # Two arms: each arm's figure heads its own column, and an arm column would only repeat the header.
        two_arms = len(candidates) == 1 and len(controls) == 1
        columns = [
            TableColumn(key="reading", header="Reading"),
            *([] if two_arms else [TableColumn(key="contrast", header="Arm")]),
            TableColumn(key="control", header=f"▸ {controls[0]}" if len(controls) == 1 else "Control", term="control"),
            TableColumn(key="arm", header=candidates[0] if two_arms else "Value", term="arm mean"),
            TableColumn(key="change", header="Change", term="change"),
            TableColumn(key="interval", header=_interval_header(levels), term="interval"),
            TableColumn(key="verdict", header="Verdict", term=CANNOT_TELL_APART),
            TableColumn(key="cases", header="Cases", term="cases"),
        ]
        blocks += [
            TableBlock(
                section="contrasts",
                name="comparisons",
                title=f"Against {arm(control)}" if control is not None else "Against the control",
                columns=columns,
                rows=[{column.key: row[column.key] for column in columns} for row in shown],
                order="by declared question, then as the bundle tested them"
                + ("" if two_arms else "; one row per reading on each arm, the control's figure repeated"),
                total_rows=len(shown),
                key_columns=1 if two_arms else 2,
            ),
            TableBlock(
                section="contrasts",
                name="comparison_statistics",
                title="Statistics",
                columns=[
                    TableColumn(key="question", header="Question"),
                    TableColumn(key="reading", header="Reading"),
                    TableColumn(key="contrast", header="Arm"),
                    TableColumn(key="control_mean", header="Control mean", term="control mean"),
                    TableColumn(key="arm_mean", header="Arm mean", term="arm mean"),
                    TableColumn(key="hedges_g", header="Hedges' g", term="Hedges' g"),
                    TableColumn(key="pairing_r", header="Pairing r", term="pairing r"),
                    TableColumn(key="p_adjusted", header="Adjusted p", term="adjusted p"),
                ],
                rows=statistics,
                order="row for row as the contrasts above",
                total_rows=len(statistics),
                details=True,
            ),
        ]
    blocks.extend(_note(family.disclosure) for family in comparisons.families)
    unranged = sorted(
        {
            reading_heading(surface, comparison.name, comparison.reading, declared=bundle.check_name(comparison.name))
            for family in comparisons.families
            for comparison in family.comparisons
            if comparison.equivalence_untested_reason is not None
        }
    )
    if unranged:
        blocks.append(_note(f"Equivalence untested on {_listed(unranged)}: {EQUIVALENCE_NEEDS_RANGE}."))
    if bundle.run_margins_withheld is not None:
        blocks.append(_note(bundle.run_margins_withheld))
    blocks += [_note(text) for text in bundle.launch_declarations]
    return blocks


def contrast_rows(bundle: AnalysisContextBundle) -> list[dict[str, CellValue]]:
    """Every contrast the bundle tested against the control, one row each, in :func:`contrast_verdicts`' order.

    The page shows only the contrasts that can resolve; a program reads them all here. A contrast with a side under
    the case floor states no change, interval, effect size or p: on so few cases a rate states none. Every figure
    is the page's, on the reading's one ruler for the report, and ``unit`` names it: a share in percent (its change
    and interval in points), a cost on the scale its ``reading`` names, a time in the unit the page states it in. A
    program that wants the measure's own units reads :attr:`Report.verdicts` or the bundle, never these rows.

    :param bundle: The campaign's evidence.
    :ptype bundle: AnalysisContextBundle
    :return: One row per contrast, keyed ``question`` (in the words it was asked, or ``(campaign-wide)``), ``reading``,
        ``unit`` (the unit ``control_mean``, ``arm_mean`` and ``delta`` are in; a percent's delta is in points),
        ``contrast`` and ``control`` (each arm as the page names it), ``control_mean``, ``arm_mean``, ``delta`` and
        ``change`` (the delta with its interval, in words), ``interval`` (with the level it holds at), ``hedges_g``,
        ``pairing_r``, ``p_adjusted``, ``verdict`` (in words) and ``cases``.
    :rtype: list[dict[str, CellValue]]
    """
    surface = bundle_decision_surface(bundle)
    verdicts = iter(contrast_verdicts(bundle, surface))
    rulers = report_rulers(surface, bundle.measure_catalog)
    rows: list[dict[str, CellValue]] = []
    forms = _change_forms(
        rulers,
        (
            (comparison.name, comparison.reading, comparison.delta, comparison.interval)
            for family in bundle.multiple_comparisons.families
            for comparison in family.comparisons
            if _small_n(comparison) is None
        ),
    )
    for family in bundle.multiple_comparisons.families:
        for comparison in family.comparisons:
            verdict = next(verdicts)
            ruler = rulers.of(comparison.name, comparison.reading)
            small = _small_n(comparison)
            form = forms.get((comparison.name, comparison.reading))
            # A program's row has no header to carry the interval's level, so the cell says it.
            interval = _interval_on_delta(comparison.interval, ruler, family.interval_level, form=form)
            rows.append(
                {
                    "question": (
                        bundle.declared.question
                        if family.aspect is not None and bundle.declared is not None
                        else "(campaign-wide)"
                    ),
                    "reading": ruler.scaled(verdict.heading),
                    "unit": ruler.unit,
                    "contrast": verdict.arm,
                    "control": verdict.control,
                    "control_mean": _figure(comparison.control.mean, ruler).value,
                    "arm_mean": _figure(comparison.contrast.mean, ruler).value,
                    "delta": None if small is not None else _change(comparison.delta, ruler).value,
                    "change": None
                    if small is not None or comparison.delta is None
                    else _change(comparison.delta, ruler, form=form).text + (f" {interval}" if interval else ""),
                    "interval": None if small is not None else interval,
                    "hedges_g": None if small is not None else comparison.hedges_g,
                    "pairing_r": comparison.pairing_correlation.r
                    if comparison.pairing_correlation is not None and small is None
                    else None,
                    "p_adjusted": None if small is not None else comparison.p_adjusted,
                    "verdict": verdict.words,
                    "cases": _comparison_cases(comparison),
                }
            )
    return rows


def report_cell_labels(bundle: AnalysisContextBundle) -> dict[str, str]:
    """Each cell's arm as the page names it, by cell reference: its label, with its test setup where it ran under two.

    :param bundle: The campaign's evidence.
    :ptype bundle: AnalysisContextBundle
    :return: ``cell_ref -> name`` for every cell on the bundle's surface.
    :rtype: dict[str, str]
    """
    surface = bundle_decision_surface(bundle)
    name = _cell_namer(surface, _arm_names(bundle))
    return {
        cell_ref(cell.variant_key, cell.apparatus_class_id): name(cell.variant_key, cell.apparatus_class_id)
        for cell in surface.cells
    }


def _cell_namer(surface: DecisionSurface, names: dict[str, ArmName]) -> Callable[[str, str], str]:
    """Name a cell by its two coordinates as every table does: the arm's label, its test setup where it ran under two."""
    multi = multi_rig_variants(surface.cells)
    namer = _namer(names)

    def name(variant_key: str, apparatus_class_id: str) -> str:
        rig = f" (test setup {short_digest(apparatus_class_id)})" if variant_key in multi else ""
        return namer(variant_key) + rig

    return name


def _straddled(check: GuardrailCheck, ruler: MeasureRuler) -> str:
    """Why a guardrail whose interval straddles its line is undecided, on the page's ruler for its reading: the line
    in points for a rate, beside the interval and margin columns that state it so. The interval itself is its own
    column's, with the level in that column's header, so the reason does not say it twice."""
    line = -check.margin if check.higher_is_better else check.margin
    said = (
        f"{with_unit(line * ruler.factor, delta_unit(ruler.unit))}, the declared margin"
        if check.margin_declared
        else "zero, as no margin is declared"
    )
    return f"the interval reaches both sides of {said}, so the arm is shown neither within it nor beyond it"


def guardrail_verdicts(
    surface: DecisionSurface,
    guardrails: GuardrailReadings | None,
    arm: Callable[[str, str], str],
    rulers: Rulers | None = None,
    *,
    check_names: Mapping[str, str] | None = None,
) -> list[Verdict]:
    """Each guardrail check as a typed verdict, in the guardrails table's row order; the table prints their words.

    An undecided check whose interval straddles its line says why on ``rulers`` (the surface's own where None), the
    ruler the guardrails table states that reading on. ``check_names`` (``AnalysisContextBundle.goal_check_names``)
    heads a goal check by its declared display name.
    """
    if guardrails is None or guardrails.withheld is not None:
        return []
    rulers = rulers or report_rulers(surface)
    verdicts = []
    for check in guardrails.checks:
        reason: VerdictReason = (
            "within_margin"
            if check.decision == "held"
            else "beyond_margin"
            if check.decision == "breached"
            else "no_interval"
            if check.interval is None
            else "interval_straddles"
        )
        verdicts.append(
            Verdict(
                kind="guardrail",
                outcome=check.decision,
                reason=reason,
                reason_detail=check.undecided_reason,
                reading=check.reading,
                name=check.name,
                heading=reading_heading(
                    surface, check.name, check.reading, declared=declared_check_name(check_names or {}, check.name)
                ),
                arm=arm(check.contrast.variant_key, check.contrast.apparatus_class_id),
                variant_key=check.contrast.variant_key,
                apparatus_class_id=check.contrast.apparatus_class_id,
                control=arm(check.control.variant_key, check.control.apparatus_class_id),
                delta=check.delta,
                interval=check.interval,
                interval_level=INTERVAL_LEVEL if check.interval is not None else None,
                margin=check.margin,
                margin_source=(
                    None if not check.margin_declared else "campaign" if check.reading == "judged" else "measure"
                ),
                guardrail=True,
                words=guardrail_words(
                    check.decision,
                    arm(check.contrast.variant_key, check.contrast.apparatus_class_id),
                    arm(check.control.variant_key, check.control.apparatus_class_id),
                )
                + (
                    f" ({_straddled(check, rulers.of(check.name, check.reading))})"
                    if reason == "interval_straddles"
                    else f" ({check.undecided_reason})"
                    if check.undecided_reason
                    else ""
                ),
            )
        )
    return verdicts


def bar_verdicts(
    surface: DecisionSurface, arm: Callable[[str, str], str], *, check_names: Mapping[str, str] | None = None
) -> list[Verdict]:
    """Each adjudicated bar's reading on each arm as a typed verdict, bar by bar in the surface's order; a bar on a
    goal check headed by its declared display name in ``check_names`` (``AnalysisContextBundle.goal_check_names``)."""
    reasons: dict[str, VerdictReason] = {
        "cleared": "interval_clears",
        "missed": "interval_misses",
        "undecided": "interval_straddles",
        "no_interval": "too_few_observations",
        "no_data": "no_observations",
    }
    verdicts = []
    for bar in surface.bars:
        if bar.state != "adjudicated":
            continue
        reading: ReadingKind = "judged" if bar.measure_id in surface.dimensions else "measure"
        for read in bar.verdicts:
            interval = (read.ci_low, read.ci_high) if read.ci_low is not None and read.ci_high is not None else None
            verdicts.append(
                Verdict(
                    kind="bar",
                    outcome=read.decision,
                    reason=reasons[read.decision],
                    reading=reading,
                    name=bar.measure_id,
                    heading=reading_heading(
                        surface,
                        bar.measure_id,
                        reading,
                        declared=declared_check_name(check_names or {}, bar.measure_id),
                    ),
                    arm=arm(read.variant_key, read.apparatus_class_id),
                    variant_key=read.variant_key,
                    apparatus_class_id=read.apparatus_class_id,
                    interval=interval,
                    value=read.value,
                    threshold=bar.threshold,
                    margin=read.margin,
                    margin_source="measure" if read.margin is not None else None,
                    words=BAR_DECISION_WORDS[read.decision],
                )
            )
    return verdicts


def _guardrail_blocks(
    surface: DecisionSurface, guardrails: GuardrailReadings | None, verdicts: Sequence[Verdict], rulers: Rulers
) -> list[ReportBlock]:
    """Each guardrail, held, breached or undecided for each arm against the control; the rule said once beside it."""
    if guardrails is None:
        return []
    if guardrails.withheld is not None:
        return [DisclosureBlock(section="contrasts", source="guardrails", text=guardrails.withheld)]
    if not guardrails.checks:
        return []
    rows: list[dict[str, TableCell]] = []
    forms = _change_forms(
        rulers, ((check.name, check.reading, check.delta, check.interval) for check in guardrails.checks)
    )
    for check, verdict in zip(guardrails.checks, verdicts, strict=True):
        ruler = rulers.of(check.name, check.reading)
        form = forms.get((check.name, check.reading))
        interval = _interval_on_delta(check.interval, ruler, form=form)
        rows.append(
            {
                "guardrail": TableCell(value=ruler.scaled(verdict.heading)),
                "arm": TableCell(value=verdict.arm),
                "control_mean": _at(
                    _figure(check.control.mean, ruler),
                    surface,
                    check.control,
                    check.name,
                    check.reading,
                    check.control.mean,
                ),
                "arm_mean": _at(
                    _figure(check.contrast.mean, ruler),
                    surface,
                    check.contrast,
                    check.name,
                    check.reading,
                    check.contrast.mean,
                ),
                "cases": TableCell(
                    value=(
                        f"{check.control.n_cases} paired"
                        if check.test == "paired"
                        else f"{check.contrast.n_cases} vs {check.control.n_cases}"
                    )
                ),
                "delta": _change(check.delta, ruler, form=form),
                "interval": TableCell(
                    value=None
                    if interval is None
                    else interval + (" (t: no declared range)" if check.interval_basis == "t" else "")
                ),
                "margin": TableCell(
                    value=with_unit(check.margin * ruler.factor, delta_unit(ruler.unit))
                    if check.margin_declared
                    else "0 (none declared)"
                ),
                "decision": TableCell(value=verdict.words),
            }
        )
    return [
        TableBlock(
            section="contrasts",
            name="guardrails",
            title="Guardrails",
            columns=[
                TableColumn(key="guardrail", header="Guardrail", term="guardrail"),
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="control_mean", header="Control mean", term="control mean"),
                TableColumn(key="arm_mean", header="Arm mean", term="arm mean"),
                TableColumn(key="cases", header="Cases", term="cases"),
                TableColumn(key="delta", header="Change", term="change"),
                TableColumn(key="interval", header=_interval_header({INTERVAL_LEVEL}), term="interval"),
                TableColumn(key="margin", header="Margin", term="margin"),
                TableColumn(key="decision", header="Decision", term="decision"),
            ],
            rows=rows,
            order="by guardrail, then arm",
            total_rows=len(rows),
        ),
        _note(_GUARDRAILS_DISCLOSURE, "guardrails"),
    ]


def _bar_blocks(surface: DecisionSurface, verdicts: Sequence[Verdict], rulers: Rulers) -> list[ReportBlock]:
    """Each adjudicated bar, read on each arm, and every bar no arm could be read against."""
    blocks: list[ReportBlock] = []
    if verdicts:
        rows: list[dict[str, TableCell]] = []
        for verdict in verdicts:
            ruler = rulers.of(verdict.name, verdict.reading)
            rows.append(
                {
                    "bar": TableCell(value=ruler.scaled(verdict.heading)),
                    "arm": TableCell(value=verdict.arm),
                    "value": _at(
                        _figure(verdict.value, ruler), surface, verdict, verdict.name, verdict.reading, verdict.value
                    ),
                    "interval": TableCell(
                        value=None
                        if verdict.interval is None
                        else _interval_cell(
                            verdict.interval[0] * ruler.factor, verdict.interval[1] * ruler.factor, ruler.unit
                        )
                    ),
                    "threshold": _figure(verdict.threshold, ruler),
                    "decision": TableCell(value=verdict.words),
                }
            )
        blocks.append(
            TableBlock(
                section="contrasts",
                name="bars",
                title="Bars",
                columns=[
                    TableColumn(key="bar", header="Bar", term="bar"),
                    TableColumn(key="arm", header="Arm"),
                    TableColumn(key="value", header="Value"),
                    TableColumn(key="interval", header="Interval", term="interval"),
                    TableColumn(key="threshold", header="Threshold"),
                    TableColumn(key="decision", header="Decision", term="decision"),
                ],
                rows=rows,
                order="bar by bar, each arm in the surface's order",
                total_rows=len(rows),
                details=True,
            )
        )
    blocks.extend(
        _note(
            f"The {bar.source} bar on {humanize('dimension', bar.measure_id)} could be read against no arm: {bar.reason}."
        )
        for bar in surface.bars
        if bar.state != "adjudicated" and bar.reason
    )
    return blocks


# =============================================================================
# Readings by stratum, and a classifier's labels
# =============================================================================

#: The column a stratum's cases are listed under when they declare no stratum, in an arm where others do.
NO_STRATUM = "(no stratum)"

#: Said beside a stratum's case count when it holds fewer than :data:`STRATUM_MIN_CASES` cases.
TOO_FEW_CASES = "too few cases to read alone"


def _readings_blocks(
    bundle: AnalysisContextBundle, surface: DecisionSurface, names: dict[str, ArmName], rulers: Rulers
) -> list[ReportBlock]:
    """A one-arm page's readings over all its cases, one row each: the figure with its interval, or its count under
    the case floor, and the cases it rests on — nothing on a page with a contrast or more than one arm, whose
    readings the contrasts state.

    The readings are the ones a contrast would have tested, had there been a control (every measure with a better
    end on a merit axis, a guardrail among them, and every judged dimension), so the one-arm page reads the same
    figures a comparison would. The per-label statistics are the per-label table's, and the strata the stratum
    table's.
    """
    contrasted = any(family.comparisons for family in bundle.multiple_comparisons.families)
    if contrasted or len({cell.variant_key for cell in surface.cells}) != 1:
        return []
    cell_name = _cell_namer(surface, names)
    catalog = bundle.measure_catalog
    rows: list[dict[str, CellValue]] = []
    # Each row's figure is where a finding citing that reading at that cell lands.
    places: list[tuple[CellFacts, str, str, float | None]] = []
    for cell in surface.cells:
        label = cell_name(cell.variant_key, cell.apparatus_class_id)
        measures = {summary.name: summary for summary in cell.measures.measures}
        readings = [
            name
            for name, descriptor in catalog.items()
            if name in measures
            and descriptor.higher_is_better is not None
            and (descriptor.merit_axis is not None or descriptor.guardrail)
            and classifier_label_of(name) is None
            and not measures[name].texts
            and not measures[name].categories
        ]
        ordered = sorted(readings, key=lambda name: reading_heading(surface, name, declared=bundle.check_name(name)))
        for name in one_figure_each(ordered):
            summary = measures[name]
            places.append((cell, name, "measure", summary.rate if summary.rate is not None else summary.mean))
            rows.append(
                {
                    "arm": label,
                    "reading": rulers.of(name, "measure").scaled(
                        reading_heading(surface, name, declared=bundle.check_name(name))
                    ),
                    "value": _value_text(summary, rulers.of(name, "measure")),
                    "cases": summary.n_independent or None,
                }
            )
        for reading in sorted(cell.judged, key=lambda reading: judged_heading(reading.dimension)):
            ruler = rulers.of(reading.dimension, "judged")
            places.append((cell, reading.dimension, "judged", reading.mean))
            rows.append(
                {
                    "arm": label,
                    "reading": judged_heading(reading.dimension),
                    "value": None if reading.mean is None else _figure_text(reading.mean, reading.sem, ruler),
                    "cases": reading.n_independent or None,
                }
            )
    if not rows:
        return []
    columns = [
        *([TableColumn(key="arm", header="Arm")] if len(surface.cells) > 1 else []),
        TableColumn(key="reading", header="Reading"),
        TableColumn(key="value", header="Value", term="interval"),
        TableColumn(key="cases", header="Cases", term="cases"),
    ]
    return [
        TableBlock(
            section="contrasts",
            name="readings",
            title="All cases",
            columns=columns,
            rows=[
                {
                    **table_row({column.key: row[column.key] for column in columns}),
                    "value": _at(TableCell(value=row["value"]), surface, *place),
                }
                for row, place in zip(rows, places, strict=True)
            ],
            order="each measure by name, then each judged dimension",
            total_rows=len(rows),
        )
    ]


def _value_text(summary: MeasureSummary, ruler: MeasureRuler) -> str:
    """One measure's figure as a reading's value cell: a rate as a count under :data:`MIN_CASES` cases, else the
    figure with its interval (``98.9% [88.1%, 100%]``), or with its spread where no interval was computed."""
    if summary.rate is not None:
        if summary.n_independent < MIN_CASES:
            return f"{round(summary.rate * summary.n)} of {summary.n}"
        return f"{_mean_text(summary.rate, SHARE)}{_interval(summary.ci_low, summary.ci_high, SHARE)}"
    if summary.mean is None:
        return render_cell(None)
    if summary.ci_low is not None and summary.ci_high is not None:
        return f"{_mean_text(summary.mean, ruler)}{_interval(summary.ci_low, summary.ci_high, ruler)}"
    return _figure_text(summary.mean, summary.sem, ruler)


def _figure_text(mean: float, sem: float | None, ruler: MeasureRuler) -> str:
    """A mean on its ruler with its standard error beside it, where one was estimated."""
    spread = f" ± {number_in(sem * ruler.factor, ruler.unit)}" if sem is not None else ""
    return f"{_mean_text(mean, ruler)}{spread}"


def _strata_blocks(
    surface: DecisionSurface,
    arms: dict[str, ArmName],
    rulers: Rulers,
    *,
    check_names: Mapping[str, str],
    stratum_names: Mapping[str, str],
) -> list[ReportBlock]:
    """Each arm's figures per stratum beside its pooled figure — or nothing, when no case declares a stratum."""
    control = surface.control_variant_key
    cells = surface_order(
        (cell for cell in surface.cells if cell.strata),
        control=control,
        names={key: name.label for key, name in arms.items()},
    )
    if not cells:
        return []
    cell_name = _cell_namer(surface, arms)
    labels = {
        cell_ref(cell.variant_key, cell.apparatus_class_id): cell_name(cell.variant_key, cell.apparatus_class_id)
        for cell in surface.cells
    }
    names = sorted(
        {stratum.stratum for cell in cells for stratum in cell.strata}, key=lambda name: (name is None, name or "")
    )
    columns = {name: f"stratum_{index}" for index, name in enumerate(names)}
    # Where every stratum is too small to read alone, one line says so and no cell repeats it.
    sizes = [stratum.n_cases for cell in cells for stratum in cell.strata]
    all_thin = max(sizes) < STRATUM_MIN_CASES
    rows: list[dict[str, CellValue]] = []
    thin: list[str] = []
    for cell in cells:
        arm = labels[cell_ref(cell.variant_key, cell.apparatus_class_id)]
        arm = f"▸ {arm}" if cell.variant_key == control else arm
        by_name = {stratum.stratum: stratum for stratum in cell.strata}
        cases: dict[str, CellValue] = {
            "arm": arm,
            "reading": "cases",
            "all": _cases_text(cell.n_observations, cell.n_cases),
        }
        for name in names:
            cases[columns[name]] = None
        for name, stratum in by_name.items():
            cases[columns[name]] = _cases_text(stratum.n_observations, stratum.n_cases)
            if stratum.n_cases < STRATUM_MIN_CASES and not all_thin:
                cases[columns[name]] = f"{cases[columns[name]]} — {TOO_FEW_CASES}"
                plural = "" if stratum.n_cases == 1 else "s"
                thin.append(
                    f"{_stratum_word(name, stratum_names)} in {arm.removeprefix('▸ ')} ({stratum.n_cases} case{plural})"
                )
        rows.append(cases)
        rows.extend(_measure_rows(arm, cell, by_name, columns, surface, rulers, check_names=check_names))
        rows.extend(_judged_rows(arm, cell, by_name, columns, rulers))
    order = (
        surface_order_sentence(has_control_row=any(cell.variant_key == control for cell in cells))
        + " Within each arm: its cases, then each measure by name, then each judged dimension."
    )
    table = TableBlock(
        section="contrasts",
        name="strata",
        title="By stratum",
        columns=[
            TableColumn(key="arm", header="Arm", term="control" if control is not None else None),
            TableColumn(key="reading", header="Reading"),
            TableColumn(key="all", header="All cases", term="cases"),
            *(
                TableColumn(key=columns[name], header=_stratum_word(name, stratum_names), term="stratum")
                for name in names
            ),
        ],
        rows=[table_row(row) for row in rows],
        order=order,
        total_rows=len(rows),
        details=True,
        key_columns=2,
    )
    blocks: list[ReportBlock] = [_by_stratum_row(table) if len(names) > _STRATA_COLUMNS else table]
    if all_thin:
        low, high = min(sizes), max(sizes)
        span = f"{low} case{'' if low == 1 else 's'}" if low == high else f"{low} to {high} cases"
        blocks.insert(
            0,
            DisclosureBlock(
                section="contrasts",
                source="strata",
                text=f"Every stratum holds {span}; a stratum is read alone from {STRATUM_MIN_CASES} cases.",
            ),
        )
    elif thin:
        # A list of counts belongs beside the table it qualifies, in the methods' fold, not in a line of prose.
        blocks.append(
            _note(
                f"Too few cases to read alone (a stratum needs at least {STRATUM_MIN_CASES} cases), so read their "
                "intervals: " + "; ".join(thin) + ".",
                "strata",
            )
        )
    return blocks


#: The most strata the table by stratum lays out as columns, beside the arm, the reading and the pooled figure; more,
#: and a stratum is a row, each arm a column, so the table stays as wide as the arms, never as the strata.
_STRATA_COLUMNS = 6


def _by_stratum_row(table: TableBlock) -> TableBlock:
    """The table by stratum turned a stratum per row: all cases first, then each stratum, each reading under it in the
    arms' own order, every arm a column holding its cell as it was; a row no arm holds is left out."""
    arms = list(dict.fromkeys(row["arm"].text for row in table.rows))
    readings = list(dict.fromkeys(row["reading"].text for row in table.rows))
    held = {(row["arm"].text, row["reading"].text): row for row in table.rows}
    places = [column for column in table.columns if column.key not in ("arm", "reading")]
    rows: list[dict[str, TableCell]] = []
    for place in places:
        for reading in readings:
            cells = {
                f"arm_{at}": held[arm, reading][place.key] if (arm, reading) in held else TableCell(value=None)
                for at, arm in enumerate(arms)
            }
            if all(cell.value is None for cell in cells.values()):
                continue
            rows.append({"stratum": TableCell(value=place.header), "reading": TableCell(value=reading), **cells})
    control = next((column.term for column in table.columns if column.key == "arm"), None)
    return table.model_copy(
        update={
            "columns": [
                TableColumn(key="stratum", header="Stratum", term="stratum"),
                TableColumn(key="reading", header="Reading"),
                *(
                    TableColumn(key=f"arm_{at}", header=arm, term=control if arm.startswith("▸ ") else None)
                    for at, arm in enumerate(arms)
                ),
            ],
            "rows": rows,
            "order": "All cases, then each stratum by name; within each, its cases, then each measure by name, then "
            "each judged dimension. The arms are the columns, in the order the arms' rows would take: "
            + table.order.split(" Within each arm")[0],
            "total_rows": len(rows),
        }
    )


def _stratum_word(name: str | None, declared: Mapping[str, str]) -> str:
    """A stratum as a column header and a disclosure name it: its declared display name, else its id, else
    :data:`NO_STRATUM`."""
    return NO_STRATUM if name is None else declared.get(name, name)


def _cases_text(n_trials: int, n_cases: int | None) -> str:
    """How many cases and trials a figure rests on — cases first, since they are the independent draws; one case run
    once is its count alone (``1 case``), and a count of one is never plural."""
    trials = f"{n_trials} trial{'' if n_trials == 1 else 's'}"
    if n_cases is None:
        return f"{trials}, cases unrecorded"
    if n_cases == 1 and n_trials == 1:
        return "1 case"
    return f"{n_cases} case{'' if n_cases == 1 else 's'}, {trials}"


def _measure_rows(
    arm: str,
    cell: CellFacts,
    by_name: dict[str | None, StratumFacts],
    columns: dict[str | None, str],
    surface: DecisionSurface,
    rulers: Rulers,
    *,
    check_names: Mapping[str, str],
) -> list[dict[str, CellValue]]:
    """One row per measure the arm or any of its strata holds — every one but a text measure."""
    pooled = {summary.name: summary for summary in cell.measures.measures}
    per_stratum = {
        name: {summary.name: summary for summary in stratum.measures.measures} for name, stratum in by_name.items()
    }
    measures = sorted(
        {
            summary.name
            for summaries in (pooled, *per_stratum.values())
            for summary in summaries.values()
            if not summary.texts
        }
    )
    # A classifier's accuracy is its trials' label match: one row, named as the accuracy, in the match's own form,
    # which counts a stratum too small to estimate a rate on.
    doubled = ACCURACY_MEASURE in measures and MATCH_MEASURE in measures
    rows: list[dict[str, CellValue]] = []
    for measure in measures:
        if doubled and measure == ACCURACY_MEASURE:
            continue
        named = ACCURACY_MEASURE if doubled and measure == MATCH_MEASURE else measure
        ruler = rulers.of(measure, "measure")

        def figure(summaries: dict[str, MeasureSummary], facts_of: CellFacts | StratumFacts) -> CellValue:
            if measure in summaries:
                return _summary_text(summaries[measure], ruler)
            return NO_SUCCESSFUL_RESULTS if facts_of.all_failed else None

        row: dict[str, CellValue] = {
            "arm": arm,
            "reading": ruler.labelled(
                reading_heading(surface, named, declared=declared_check_name(check_names, named))
            ),
            "all": figure(pooled, cell),
        }
        for name in columns:
            row[columns[name]] = figure(per_stratum[name], by_name[name]) if name in per_stratum else None
        rows.append(row)
    return rows


def _judged_rows(
    arm: str,
    cell: CellFacts,
    by_name: dict[str | None, StratumFacts],
    columns: dict[str | None, str],
    rulers: Rulers,
) -> list[dict[str, CellValue]]:
    """One row per judged dimension the arm or any of its strata was scored on."""
    pooled = {reading.dimension: reading for reading in cell.judged}
    per_stratum = {
        name: {reading.dimension: reading for reading in stratum.judged} for name, stratum in by_name.items()
    }
    dimensions = sorted({dimension for readings in (pooled, *per_stratum.values()) for dimension in readings})
    rows: list[dict[str, CellValue]] = []
    for dimension in dimensions:
        ruler = rulers.of(dimension, "judged")
        row: dict[str, CellValue] = {
            "arm": arm,
            "reading": judged_heading(dimension),
            "all": _judged_text(pooled[dimension], ruler) if dimension in pooled else None,
        }
        for name in columns:
            readings = per_stratum.get(name, {})
            row[columns[name]] = _judged_text(readings[dimension], ruler) if dimension in readings else None
        rows.append(row)
    return rows


def _summary_text(summary: MeasureSummary, ruler: MeasureRuler) -> str:
    """One measure's figure with the n it rests on, on its ruler: a rate as a count below :data:`MIN_CASES` cases.

    A rate is a share by its shape, so it reads as a percent (:data:`SHARE`) whatever the catalog said of its
    measure; a mean is stated with its unit on the figure, and its spread as a bare number beside it on the same
    scale (``98.9% ± 6.5``, ``$0.0459 ± 0.007414``).
    """
    n = _n_text(summary.n, summary.n_independent)
    if summary.rate is not None:
        if summary.n_independent < MIN_CASES:
            return f"{round(summary.rate * summary.n)} of {summary.n} {n}"
        return f"{_mean_text(summary.rate, SHARE)}{_interval(summary.ci_low, summary.ci_high, SHARE)} {n}"
    if summary.mean is not None:
        spread = f" ± {number_in(summary.sem * ruler.factor, ruler.unit)}" if summary.sem is not None else ""
        return f"{_mean_text(summary.mean, ruler)}{spread} {n}"
    counts = sorted(summary.categories.items(), key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{category}: {count}" for category, count in counts) + f" {n}"


def _mean_text(value: float, ruler: MeasureRuler) -> str:
    """A mean on its ruler, the unit on the figure where the unit sits there (``98.9%``, ``$0.0459``), bare where the
    row's heading names it (a latency's ``950`` under ``Turn time (ms)``)."""
    scaled = value * ruler.factor
    if is_money(ruler.unit) and ruler.money is not None:
        # In the measure's one form, by the one owner of money's scale (#993).
        return ruler.money.spell(scaled)
    if ruler.unit == PERCENT or is_money(ruler.unit):
        return with_unit(scaled, ruler.unit)
    return number_in(scaled, ruler.unit)


def _interval(low: float | None, high: float | None, ruler: MeasureRuler) -> str:
    """An interval as ``[low, high]`` on its ruler, or nothing when either end is unestimated."""
    if low is None or high is None:
        return ""
    if is_money(ruler.unit) and ruler.money is not None:
        return f" [{ruler.money.spell(low * ruler.factor)}, {ruler.money.spell(high * ruler.factor)}]"
    return f" {_interval_cell(low * ruler.factor, high * ruler.factor, ruler.unit)}"


def _judged_text(reading: JudgedReading, ruler: MeasureRuler) -> str:
    """One judged dimension's mean, with its standard error and the scores it rests on, on the dimension's ruler."""
    if reading.mean is None:
        return f"no score (n={reading.n})"
    spread = f" ± {number_in(reading.sem * ruler.factor, ruler.unit)}" if reading.sem is not None else ""
    text = f"{_mean_text(reading.mean, ruler)}{spread} {_n_text(reading.n, reading.n_independent)}"
    if (powered := reading.prediction_powered) is not None:
        text += f"; {_prediction_powered_text(powered, ruler)}"
    return text


def _prediction_powered_text(powered: PredictionPoweredReading, ruler: MeasureRuler) -> str:
    """The prediction-powered estimate beside a judged mean: its value and interval, or why it is not available."""
    rated = f"{powered.n_labelled} rated by people"
    if powered.mean is None:
        return f"with people's ratings: not available ({rated}, under {powered.min_labelled})"
    rectifier = with_unit((powered.rectifier or 0.0) * ruler.factor, delta_unit(ruler.unit))
    return (
        f"with people's ratings: {_mean_text(powered.mean, ruler)}{_interval(powered.ci_low, powered.ci_high, ruler)} "
        f"(rectifier {rectifier}; {rated})"
    )


def _n_text(n: int, n_cases: int) -> str:
    """A figure's n, and where cases repeat, how many cases it is over."""
    if 0 < n_cases < n:
        return f"(n={n} over {n_cases} case{'' if n_cases == 1 else 's'})"
    return f"(n={n})"


#: The columns of the per-label table after the label and the arm, by the statistic each holds.
_LABEL_STATISTICS: tuple[tuple[ClassifierStatistic, str], ...] = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
)


def _label_blocks(surface: DecisionSurface, names: dict[str, ArmName], rulers: Rulers) -> list[ReportBlock]:
    """A classifier's per-label precision, recall and F1, every arm's, as one table — or nothing for no classifier."""
    control = surface.control_variant_key
    cell_name = _cell_namer(surface, names)
    labels = {
        cell_ref(cell.variant_key, cell.apparatus_class_id): cell_name(cell.variant_key, cell.apparatus_class_id)
        for cell in surface.cells
    }
    figures: dict[str, dict[str, dict[str, MeasureSummary]]] = {}
    for cell in surface.cells:
        ref = cell_ref(cell.variant_key, cell.apparatus_class_id)
        for summary in cell.measures.measures:
            if (classifier := classifier_label_of(summary.name)) is None:
                continue
            statistic, label = classifier
            figures.setdefault(label, {}).setdefault(ref, {})[statistic] = summary
    if not figures:
        return []
    order = sorted(
        ((cell_ref(cell.variant_key, cell.apparatus_class_id), cell) for cell in surface.cells),
        key=lambda item: (item[1].variant_key != control, labels[item[0]], item[1].apparatus_class_id),
    )
    classifying = {ref for by_cell in figures.values() for ref in by_cell}
    rows: list[dict[str, CellValue]] = []
    missing = False
    for label in sorted(figures):
        for ref, cell in order:
            if ref not in classifying:
                continue
            by_statistic = figures[label].get(ref, {})
            row: dict[str, CellValue] = {
                "label": label,
                "arm": f"▸ {labels[ref]}" if cell.variant_key == control else labels[ref],
            }
            for statistic, _ in _LABEL_STATISTICS:
                figure = by_statistic.get(statistic)
                row[statistic] = None if figure is None else _summary_text(figure, rulers.of(figure.name, "measure"))
                missing = missing or figure is None
            rows.append(row)
    said = [
        "Precision is counted over the trials an arm predicted as the label and recall over those expected as it, each "
        f"with its {with_unit(INTERVAL_LEVEL * 100, '%')} Wilson interval over the arm's cases; below "
        f"{MIN_CASES} cases a rate is stated as a count. F1 has no interval by construction."
    ]
    if missing:
        said.append(
            "A label an arm never predicted has no precision, one it never met has no recall, and either has no F1: "
            "each is shown as —."
        )
    return [
        TableBlock(
            section="contrasts",
            name="labels",
            title="Per label",
            columns=[
                TableColumn(key="label", header="Label", term="label"),
                TableColumn(key="arm", header="Arm", term="control" if control is not None else None),
                *(TableColumn(key=statistic, header=header) for statistic, header in _LABEL_STATISTICS),
            ],
            rows=[table_row(row) for row in rows],
            order="by label, then the control first and every other arm by name",
            total_rows=len(rows),
            details=True,
        ),
        _note(" ".join(said)),
    ]


# =============================================================================
# Screens 3 and 4: the cases, and each trial to drill into
# =============================================================================

#: A trial's state as a mark in the cases table, always beside words for a lost one.
_MARKS = {"passed": "✓", "failed": "✗", "lost": "–"}

#: The order the trials section lists what went wrong in: a failure (a record a reader reads) before a loss (the
#: test setup's), and a passed trial last, wherever trials are listed by state.
_STATE_ORDER = {"failed": 0, "lost": 1, "passed": 2}

#: How many failed or lost records the page shows open, in the trials table's order. Twelve reads every failure of
#: a small campaign without a click, and a campaign that failed hundreds still scrolls; the rest stay closed.
_OPEN_TRIALS = 12


def _case_numbers(bundle: AnalysisContextBundle) -> dict[str, int]:
    """Each case's number, as the cases table orders it: by label, from 1."""
    labels = {trial.case_id: trial.case_label for trial in bundle.trials}
    return {case: index for index, case in enumerate(sorted(labels, key=lambda case: (labels[case], case)), start=1)}


def _case_namer(bundle: AnalysisContextBundle) -> Callable[[str], str]:
    """Name a case as every table does: by the name its author gave it where its label reads as one
    (``double-charge``), else ``case N`` — never its first input."""
    labels = {trial.case_id: trial.case_label for trial in bundle.trials}
    numbers = _case_numbers(bundle)

    def name(case: str) -> str:
        if case not in numbers:
            return humanize("case", case) or "a case"
        return humanize("case", labels[case]) or f"case {numbers[case]}"

    return name


def _case_blocks(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[ReportBlock]:
    """One row per case: each arm's trials as marks with their causes, regressions first, each linked to its trials."""
    if not bundle.trials:
        return []
    control = _control_key(bundle)
    arms = [board.variant_key for board in bundle.scoreboard]
    numbers = _case_numbers(bundle)
    case_name = _case_namer(bundle)
    by_case: dict[str, list[TrialRow]] = {}
    for trial in bundle.trials:
        by_case.setdefault(trial.case_id, []).append(trial)
    strata = any(trial.stratum for trial in bundle.trials)

    def counts(case: str, key: str) -> tuple[int, int]:
        """How many of ``key``'s usable trials on ``case`` passed, of how many."""
        usable = [t for t in by_case[case] if t.arm == key and t.state != "lost"]
        return sum(t.state == "passed" for t in usable), len(usable)

    def group(case: str) -> int:
        """0 a regression, 1 an arm with no usable trial, 2 the rest."""
        # A case-level flag: a candidate's share of its usable trials passed below the control's. A tie or a gain
        # never flags.
        ours, of = counts(case, control) if control is not None else (0, 0)
        for key in arms:
            theirs, n = counts(case, key)
            if key != control and n and of and theirs * of < ours * n:
                return 0
        if any(counts(case, key)[1] == 0 for key in arms):
            return 1
        return 2

    rows: list[dict[str, CellValue]] = []
    for case in sorted(by_case, key=lambda case: (group(case), numbers[case])):
        trials = by_case[case]
        row: dict[str, CellValue] = {
            "case": case_name(case),
            "flag": {0: "regression", 1: "no usable trial", 2: None}[group(case)],
        }
        if strata:
            row["stratum"] = _stratum_word(trials[0].stratum, bundle.stratum_names) if trials[0].stratum else None
        for index, key in enumerate(arms):
            row[f"arm_{index}"] = _marks([trial for trial in trials if trial.arm == key])
        # The ids are navigation, not information: the failed and lost ones lead, since those are the records a
        # reader clicks through to; the passed ones follow, each a row of the closed table.
        row["trials"] = " ".join(trial.trial_id for trial in _wrong_first(trials))
        rows.append(row)
    blocks: list[ReportBlock] = [
        TableBlock(
            section="cases",
            name="cases",
            title="Cases",
            columns=[
                TableColumn(key="case", header="Case", term="case"),
                TableColumn(key="flag", header="Flag", term="flag"),
                *([TableColumn(key="stratum", header="Stratum", term="stratum")] if strata else []),
                *(
                    TableColumn(
                        key=f"arm_{index}", header=f"▸ {arm(key)}" if key == control else arm(key), term="marks"
                    )
                    for index, key in enumerate(arms)
                ),
                TableColumn(key="trials", header="Trials", term="trial"),
            ],
            rows=[table_row(row) for row in rows],
            order="regressions first, then cases an arm has no usable trial of, then the rest, each in case order",
            total_rows=len(rows),
        )
    ]
    if bundle.trials_omitted:
        blocks.append(
            DisclosureBlock(
                section="cases",
                source="runs",
                text=(
                    f"{bundle.trials_omitted} more trials count on the scoreboard and are not listed here, the passed "
                    "ones first."
                ),
            )
        )
    return blocks


def _marks(trials: Sequence[TrialRow]) -> str:
    """One arm's trials of one case as marks, repeat by repeat, its losses and failures named by cause."""
    ordered = sorted(trials, key=lambda trial: trial.repeat)
    marks = " ".join(_MARKS[trial.state] for trial in ordered) or "not run"
    named = Counter(
        f"{'lost' if trial.state == 'lost' else 'failed'}: {cause_short(trial.cause or 'unknown')}"
        for trial in ordered
        if trial.state != "passed"
    )
    if named:
        marks += " (" + ", ".join(f"{words} ×{count}" for words, count in sorted(named.items())) + ")"
    return marks


def _trial_why(trial: TrialRow, check_names: Mapping[str, str]) -> str | None:
    """Which check or judged dimension failed a trial, or why it was lost, on one line; None for a passed trial. A
    check by its declared display name in ``check_names``, else its expression."""
    if trial.state == "passed":
        return None
    if trial.state == "lost":
        return f"lost: {cause_short(trial.cause or 'unknown')}"
    if trial.cause == "check" and trial.failed_checks:
        # A check is named by its declared name, else as code, by its expression: never humanized, and never a
        # "pass rate" on one trial.
        return "check: " + ", ".join(
            check_names.get(check.expression) or code(check.expression) for check in trial.failed_checks
        )
    if trial.cause == "judged":
        under = [entry.dim for entry in trial.judge if entry.scale == "pass_fail" and entry.score == 0]
        return "judge: " + ", ".join(humanize("dimension", dim) for dim in under) if under else "failed by the judge"
    return cause_short(trial.cause or "unknown")


def _wrong_first(trials: Sequence[TrialRow]) -> list[TrialRow]:
    """``trials`` with the failed ones first, then the lost, then the passed, each group as the bundle numbers it."""
    return sorted(trials, key=lambda trial: _STATE_ORDER[trial.state])


def _census_line(bundle: AnalysisContextBundle) -> str:
    """The trials' census in one line, from the scoreboard, which counts every trial whether listed or not.

    ``180 trials ran. Of them 179 passed and 1 failed.`` The lost are counted only when there are any. Two
    sentences, the second opening on a word: the page's sentence rule would read a second sentence opening on a
    digit as the first one's tail, and four counts in one sentence is a table.
    """
    boards = bundle.scoreboard
    ran = sum(board.ran for board in boards)
    passed = sum(board.passed for board in boards)
    failed = sum(sum(board.failed.values()) for board in boards)
    lost = sum(sum(board.lost.values()) for board in boards)
    opened = f"{ran} trial{'' if ran == 1 else 's'} ran."
    if passed == ran:
        return f"{opened} {'It' if ran == 1 else 'Every one'} passed."
    counts = [f"{passed} passed", f"{failed} failed", *([f"{lost} were lost"] if lost else [])]
    return f"{opened} Of them {', '.join(counts[:-1])} and {counts[-1]}."


def _trial_blocks(bundle: AnalysisContextBundle, arm: Callable[[str], str], *, records: bool) -> list[ReportBlock]:
    """The census in one line, then only what went wrong, then the passed trials folded away in one place.

    A campaign that ran hundreds of trials and failed one is read for that one: the table lists the failed trials
    first and the lost ones after, each by trial id, and with ``records`` each has a record (what it asked, what the
    judge saw and said, and why) in the same order, the first :data:`_OPEN_TRIALS` shown open. Without them each
    row names its stored result instead, read with ``result_get``, and carries the trial's id, so a link to the trial
    lands on its row: the records are most of a large campaign's page. A passed trial has nothing to read, so it is
    one row of a closed table, there so its stored result stays reachable with ``result_get``. When every trial
    passed the line stands alone.
    """
    if not bundle.trials:
        return []
    blocks: list[ReportBlock] = [DisclosureBlock(section="trials", source="runs", text=_census_line(bundle))]
    arms = len(bundle.scoreboard)
    exclusions = arm_exclusions(bundle)
    statuses = arm_run_statuses(bundle)
    case_name = _case_namer(bundle)
    wrong = [trial for trial in _wrong_first(bundle.trials) if trial.state != "passed"]
    rows: list[dict[str, TableCell]] = []
    held: list[ReportBlock] = []
    for position, trial in enumerate(wrong):
        why = _trial_why(trial, bundle.goal_check_names)
        row = table_row(
            {
                "trial": trial.trial_id,
                "case": case_name(trial.case_id),
                "arm": arm(trial.arm),
                "outcome": trial.state,
                "why": why,
            }
        )
        if not records:
            # No record carries the trial's id, so its row does: a link to the trial lands here, beside its result.
            rows.append(
                {
                    **row,
                    "trial": TableCell(value=trial.trial_id, anchor=trial.trial_id),
                    "result": TableCell(value=code(trial.result_id)),
                }
            )
            continue
        rows.append(row)
        words: CauseWords
        if trial.state == "lost":
            words = loss_words(
                trial.cause or "unknown",
                exclusions=exclusions.get(trial.arm, ()),
                arms=arms,
                run_statuses=statuses.get(trial.arm, ()),
            )
        else:
            words = fail_words(trial.cause or "unknown")
        held.append(
            TrialBlock(
                section="trials",
                anchor=trial.trial_id,
                trial_id=trial.trial_id,
                case_label=case_name(trial.case_id),
                arm=arm(trial.arm),
                repeat=trial.repeat,
                state=trial.state,
                cause_words=f"{words.words}. {words.remedy[:1].upper()}{words.remedy[1:]}.",
                why=why,
                first_input=trial.first_input,
                excerpt=trial.excerpt,
                # The trial's own record keeps the rate name; the page prints a declared name beside the expression.
                failed_checks=[
                    check.model_copy(update={"name": bundle.goal_check_names[check.expression]})
                    if check.expression in bundle.goal_check_names
                    else check
                    for check in trial.failed_checks
                ],
                judge=list(trial.judge),
                error_words=trial.error_words,
                result_id=trial.result_id,
                open=position < _OPEN_TRIALS,
            )
        )
    if rows:
        columns = [
            TableColumn(key="trial", header="Trial", term="trial"),
            TableColumn(key="case", header="Case", term="case"),
            TableColumn(key="arm", header="Arm"),
            TableColumn(key="outcome", header="Outcome", term="outcome"),
            TableColumn(key="why", header="Why", term="why"),
        ]
        if not records:
            columns.append(TableColumn(key="result", header="Stored result"))
        blocks.append(
            TableBlock(
                section="trials",
                name="trials",
                title="Trials",
                columns=columns,
                rows=rows,
                order="the failed first, then the lost, each by trial id",
                total_rows=len(rows),
            )
        )
    blocks += held
    passed = [
        {
            "trial": trial.trial_id,
            "case": case_name(trial.case_id),
            "arm": arm(trial.arm),
            "result": code(trial.result_id),
        }
        for trial in bundle.trials
        if trial.state == "passed"
    ]
    if passed:
        blocks.append(
            TableBlock(
                section="trials",
                name="passed_trials",
                title="Passed trials",
                columns=[
                    TableColumn(key="trial", header="Trial", term="trial"),
                    TableColumn(key="case", header="Case", term="case"),
                    TableColumn(key="arm", header="Arm"),
                    TableColumn(key="result", header="Stored result"),
                ],
                rows=[table_row(row) for row in passed],
                order="as the bundle numbers them",
                total_rows=len(passed),
                details=True,
            )
        )
    return blocks


# =============================================================================
# Screen 5: what the campaign could not speak to
# =============================================================================


def _not_tested_blocks(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    arm: Callable[[str], str],
    *,
    also: Sequence[ReportBlock] = (),
) -> list[ReportBlock]:
    """Everything the campaign could not speak to, one line each — a list, never a score.

    ``also`` is the page's unasked readings (:func:`_question_blocks`), laid out after code's own lines and before
    the folded disclosures.
    """
    said: list[str] = []
    for row in bundle.coverage:
        if row.status in ("thin", "unswept"):
            why = "one level ran" if row.status == "unswept" else "too few repeats to read"
            said.append(f"Lever {row.name} is {row.status}: {why}.")
    said += [_unexercised_sentence(factor, bundle, arm) for factor in bundle.unexercised_factors]
    usable = {(trial.arm, trial.case_id) for trial in bundle.trials if trial.state != "lost"}
    case_name = _case_namer(bundle)
    for board in bundle.scoreboard:
        unusable = sorted(
            {
                trial.case_id
                for trial in bundle.trials
                if trial.arm == board.variant_key and (trial.arm, trial.case_id) not in usable
            },
            key=case_name,
        )
        if unusable:
            named = _listed([case_name(case) for case in unusable])
            said.append(f"{arm(board.variant_key)} has no usable trial of {named}.")
    for proof in bundle.goal_check_proofs:
        if proof.proof != "proven":
            said.append(_unproven_check_sentence(proof, bundle.goal_check_names.get(proof.check)))
    blocks: list[ReportBlock] = [DisclosureBlock(section="not_tested", source="not_tested", text=text) for text in said]
    blocks += _no_margin_blocks(bundle)
    blocks += calibration_blocks(bundle)
    blocks += also
    quoted = [bundle.held_fixed_reading.disclosure] if bundle.held_fixed_reading.disclosure else []
    if bundle.cost_unmeasured:
        quoted.append(bundle.cost_unmeasured + _some_arms(bundle, bundle.cost_unmeasured_cells, "Unmeasured", arm))
    blocks += [
        DisclosureBlock(section="not_tested", source="not_tested", text=text, fold="Disclosures") for text in quoted
    ]
    if not blocks:
        return [DisclosureBlock(section="not_tested", source="not_tested", text="Nothing declared went untested.")]
    return blocks


def _unexercised_sentence(factor: UnexercisedFactor, bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str:
    """A lever no comparison exercised, in one line: ``tool model: no tool call was made on either arm.``

    Calls in one arm only say so: ``observed in one arm only``.
    """
    name = humanize("dimension", factor.factor.rsplit(".", 1)[-1])
    call = f"{factor.role.replace('_', ' ')} call"
    arms = len(bundle.scoreboard)
    if factor.results == 0 or not factor.arms:
        where = "the arm" if arms <= 1 else "either arm" if arms == 2 else "any arm"
        return f"{name}: no {call} was made on {where}."
    others = "the other arm" if arms == 2 else "the other arms"
    if factor.results > 1:
        return (
            f"{name}: {factor.results} results made {call}s, all on {arm(factor.arms[0])}, and none on {others}, "
            "so it was observed in one arm only and is not a level."
        )
    if arms <= 1:
        return f"{name}: one {call} was made in all its trials, too few to read a level from."
    return f"{name}: one {call} was made, on {arm(factor.arms[0])}, and none on {others}, so it is not a level."


#: What a goal check not shown to beat doing nothing is, by its proof, as the page and the print both say it.
CHECK_PROOF_WORDS: dict[str, str] = {
    "refuted": "refuted: its control shows it cannot tell acting from doing nothing",
    "unproven": "unproven: no control shows it can tell acting from doing nothing",
}

#: What follows from a goal check not shown to beat doing nothing, as the page and the print both say it.
NOT_SHOWN_TO_MEASURE = "a candidate that did nothing might pass it, so it is not shown to measure the behaviour"


def _unproven_check_sentence(proof: GoalCheckProofReading, declared: str | None) -> str:
    """A goal check not shown to beat doing nothing: what it is, and what its pass rate is not. Named by its declared
    display name, else its expression."""
    check = declared if declared is not None else code(proof.check)
    if proof.refused is not None:
        return (
            f"Goal check {check} is {CHECK_REFUSED_UNDER_CURRENT_GRAMMAR} ({proof.refused}), so no arm was graded on "
            "it. Any pass rate it has predates the rule."
        )
    notes = []
    if proof.unrecorded:
        notes.append(f"{proof.unrecorded} of {proof.runs} runs recorded no proof")
    if proof.stale:
        notes.append(f"{proof.stale} of {proof.runs} runs proved it under an earlier rule; relaunch to re-prove")
    tail = f" ({'; '.join(notes)})" if notes else ""
    words = CHECK_PROOF_WORDS["refuted" if proof.proof == "refuted" else "unproven"]
    return f"Goal check {check} is {words}{tail}. {NOT_SHOWN_TO_MEASURE[:1].upper()}{NOT_SHOWN_TO_MEASURE[1:]}."


def _no_margin_blocks(bundle: AnalysisContextBundle) -> list[ReportBlock]:
    """The measures tested with no margin to be equivalent on: one sentence, then each once with how it takes one."""
    catalog = bundle.measure_catalog
    names = list(
        dict.fromkeys(
            tested.name
            for family in bundle.multiple_comparisons.families
            for tested in family.comparisons
            if tested.name in catalog and tested.reading == "measure" and tested.margin_source is None
        )
    )
    if not names:
        return []

    def declaring(name: str) -> str:
        if name not in METRIC_DESCRIPTORS:
            ranged = catalog[name].family == "mechanical" and catalog[name].value_range is None
            # A check is graded by its scenario, not a grader the caller names: no call to quote.
            call = "" if goal_check_of(name) is not None else f": `grader({name}, margin=...)`"
            return f"where it is graded{call}" + (", with its `range=`" if ranged else "")
        if name in RUN_MARGIN_MEASURES:
            return "on the classifier's field, on every arm's run: `grader('<field>', margin=...)`"
        return "cannot: the engine owns its description"

    rows: list[dict[str, CellValue]] = [
        {
            "reading": measure_name(name, catalog[name].reader_name, declared=bundle.check_name(name)),
            "declare": declaring(name),
        }
        for name in names
    ]
    return [
        DisclosureBlock(
            section="not_tested",
            source="not_tested",
            text=(
                f"No margin is declared on these, so none can read equivalent: {CANNOT_TELL_APART} never means the "
                "arms are alike."
            ),
        ),
        _readings_table("no_margin", "No margin", rows),
    ]


def _method_blocks(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    arm: Callable[[str], str],
    names: dict[str, ArmName],
    *,
    source: ReportSource,
    writer: str | None,
    left_disclosures: Sequence[ReportBlock] = (),
) -> list[ReportBlock]:
    """How everything on the page was measured and made: a table, then the engine's disclosures and those the
    contrasts and charts left for the methods (``left_disclosures``) in one closed fold."""
    blocks: list[ReportBlock] = [_methods_table(bundle, names, source=source, writer=writer)]

    def say(source_of: DisclosureSource, text: str | None) -> None:
        if text and text.strip():
            blocks.append(DisclosureBlock(section="methods", source=source_of, text=text.strip(), fold="Disclosures"))

    for tier in bundle.judge_evidence_tiers:
        # The dimension by the name every other surface gives it, never its stored id.
        say(
            "measurement",
            "Judge: " + tier_sentence(tier.model_copy(update={"rubric_dim": measure_name(tier.rubric_dim)})),
        )
    if (settings := setup_disclosure("settings", bundle, arm)) is not None:
        blocks.append(settings)
    for variant_key, cells in arm_exclusions(bundle).items():
        if truncated := sum(cell.by_cause.get("truncated", 0) for cell in cells):
            say(
                "measurement",
                f"{arm(variant_key)}: {truncated} of {sum(cell.n_trials for cell in cells)} trials lost because the "
                f"judge's reply was cut off at its output cap ({loss_words('truncated', exclusions=cells).words}).",
            )
    for flagged in bundle.differential_exclusions:
        others = ", ".join(
            f"{format_number(100 * share)}% of {arm(key)}'s" for key, share in sorted(flagged.other_arms.items())
        )
        say(
            "measurement",
            f"Differential loss: {format_number(100 * flagged.share)}% of {arm(flagged.variant_key)}'s trials were "
            f"lost to {cause_short(flagged.cause)}"
            + (f", against {others}" if others else "")
            + f"; above {format_number(100 * flagged.threshold)}%, the arms are not compared on the same trials.",
        )
    say("measurement", bundle.model_identity_disclosure)
    for reading in bundle.arm_served_models:
        if reading.state == "pooled":
            say(
                "measurement",
                f"Arm {arm(reading.variant_key)} asked for one model and its candidate was answered by "
                f"{_listed(reading.served_models)}, so its numbers mix those models.",
            )
    say("measurement", bundle.launch_disclosure)
    say("cost", _unpriced_cost_line(bundle, arm))
    if (routed := setup_disclosure("cost", bundle, arm)) is not None:
        blocks.append(routed)
    if (latency := setup_disclosure("latency", bundle, arm)) is not None:
        blocks.append(latency)
    for _, sentence in sorted(bundle.short_runs.items()):
        say("runs", sentence)
    _run_disclosures(bundle, say, arm)
    if (footing := _production_footing_sentence(bundle, arm)) is not None:
        say("measurement", footing)
    _design_disclosures(bundle, say, arm)
    _time_axis_disclosures(surface.time_axis, bundle.time_axis_withheld, say)
    blocks.extend(left_disclosures)
    return blocks


def _methods_table(
    bundle: AnalysisContextBundle,
    names: dict[str, ArmName],
    *,
    source: ReportSource,
    writer: str | None,
) -> TableBlock:
    """Models, judge, settings sent, cost, the analysis's writer and the bundle, one row each."""
    rows: list[tuple[str, str]] = []
    control = _control_key(bundle)
    keys = [board.variant_key for board in bundle.scoreboard]
    for key in ([control] if control is not None else []) + [key for key in keys if key != control]:
        rows.append(("Arm", names[key].first() if key in names else _namer(names)(key)))
    for level in bundle.judge_change.levels:
        rows.append(("Judge", level.judge_model))
        rows.append(
            (
                "Judge temperature",
                "not sent" if level.judge_temperature is None else format_number(level.judge_temperature),
            )
        )
        if level.judge_sent_settings:
            rows.append(("Judge settings sent", "; ".join(_settings_words(sent) for sent in level.judge_sent_settings)))
    # What each arm's trials cost: totals, which count nothing and are never scaled, in their one form (#993).
    totals = money_ruler((board.test_cost_usd for board in bundle.scoreboard), per=0)
    # A total that counts only part of its trials' spend is a floor, and says so beside itself (#993).
    costs = [
        f"{_namer(names)(board.variant_key)} {totals.spell(board.test_cost_usd)}"
        + (
            f" ({PRICED_TRIALS_TEMPLATE.format(priced=board.ran - board.n_cost_unpriced, trials=board.ran)})"
            if board.n_cost_unpriced
            else ""
        )
        for board in bundle.scoreboard
        if board.test_cost_usd is not None
    ]
    if costs:
        rows.append(("Test cost", "; ".join(costs)))
    rows.append(("Dataset", ", ".join(ref.label for ref in bundle.case_sets) or "each run's own cases"))
    rows.append(
        (
            "Design",
            _question_words(bundle)
            + (f", frozen {_date(bundle.design_frozen_at)}" if bundle.design_frozen_at else ", not frozen"),
        )
    )
    rows.append(("Analysis", writer or NO_ANALYSIS))
    rows.append(("Date", _date(source.generated_at)))
    if unshown := _uncarried(bundle):
        rows.append(("Readings not shown", f"{len(unshown)} carried by no trial: {_listed(unshown)}"))
    rows.append(
        (
            "Bundle",
            f"{source.bundle_fingerprint}, shape {bundle.schema_version}, 3tears-evals "
            f"{bundle.package_version or 'version not recorded'}, {len(bundle.run_ids)} runs; its JSON beside this "
            "page",
        )
    )
    return TableBlock(
        section="methods",
        name="methods",
        title="Methods",
        columns=[TableColumn(key="item", header="Item"), TableColumn(key="value", header="Value")],
        rows=[{"item": TableCell(value=item), "value": TableCell(value=value)} for item, value in rows],
        order="the arms, the judge, the cost, then how the page was made",
        total_rows=len(rows),
    )


def _settings_words(sent: ClientRequestSettings) -> str:
    """Request settings as words, never as ``key=value``."""
    parts = [f"max tokens {sent.max_tokens}"]
    if sent.reasoning_max_tokens is not None:
        parts.append(f"reasoning max tokens {sent.reasoning_max_tokens}")
    if sent.reasoning_effort is not None:
        parts.append(f"reasoning effort {sent.reasoning_effort}")
    if sent.strict_output:
        parts.append("strict output")
    return ", ".join(parts)


def _uncarried(bundle: AnalysisContextBundle) -> list[str]:
    """Each reading the campaign names that no arm carries on any trial, or that one side of a contrast never carried."""
    carried = {summary.name for cell in bundle.cell_measures for summary in cell.measures.measures if summary.n}
    names = [name for name in bundle.measure_catalog if name not in carried]
    names += [
        comparison.name
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
        if min(comparison.control.n_cases, comparison.contrast.n_cases) == 0
    ]
    catalog = bundle.measure_catalog
    return list(
        dict.fromkeys(
            measure_name(name, catalog[name].reader_name if name in catalog else None, declared=bundle.check_name(name))
            for name in names
        )
    )


def _run_disclosures(
    bundle: AnalysisContextBundle, say: Callable[[DisclosureSource, str | None], None], arm: Callable[[str], str]
) -> None:
    """The member runs left out, unfinished, short, or of unknown completeness, and every run an arm's pass^k kept out
    of its headline cell (:func:`~threetears.evals.analysis.lenses.frontier.pass_hat_k_cell_lines`),
    each line led by the arm, as the frontier's text leads it."""
    if not bundle.run_ids:
        say("runs", "The campaign resolved no runs in this scope, so no table or chart here holds anything.")
    if bundle.unresolved_run_ids:
        say(
            "runs",
            f"{len(bundle.unresolved_run_ids)} run(s) the campaign names were not found in this scope and are in no "
            f"table here: {_listed(bundle.unresolved_run_ids)}.",
        )
    if bundle.archived_run_ids:
        say(
            "runs",
            f"{len(bundle.archived_run_ids)} member run(s) were archived and are held out of everything here: "
            f"{_listed(bundle.archived_run_ids)}.",
        )
    if bundle.incomplete_runs:
        say(
            "runs",
            f"{len(bundle.incomplete_runs)} run(s) did not complete and are pooled as far as they got: "
            + "; ".join(f"{run_id} ({status})" for run_id, status in sorted(bundle.incomplete_runs.items()))
            + ".",
        )
    if bundle.completeness_unknown_run_ids:
        say(
            "runs",
            f"Whether {len(bundle.completeness_unknown_run_ids)} run(s) came up short is unknown — they carry no "
            f"completeness record: {_listed(bundle.completeness_unknown_run_ids)}.",
        )
    for cell in bundle.short_cells:
        say("runs", cell.sentence)
    cell_lines = [
        f"{arm(point.variant_key)}: {line}."
        for subject in bundle.frontier.subjects
        for point in subject.points
        for reading in point.pass_hat_k_cells
        for line in pass_hat_k_cell_lines(reading)
    ]
    if cell_lines:
        # What "kept out" covers, once, before the first: the arm's tables above still count those runs.
        say("runs", PASS_HAT_K_CELL_READING)
    for line in cell_lines:
        say("runs", line)


def _design_disclosures(
    bundle: AnalysisContextBundle, say: Callable[[DisclosureSource, str | None], None], arm: Callable[[str], str]
) -> None:
    """How the arms were set up and compared: the control, the confounds, the factors, the declared crossing."""
    say("comparisons", bundle.design.stale_control_disclosure)
    say("comparisons", bundle.design.earlier_control.disclosure if bundle.design.earlier_control else None)
    say("comparisons", _verdict_order_sentence(bundle))
    for instability in bundle.subject_key_instabilities:
        if instability.kind == "one_key_many_labels":
            text = f"Subject key {instability.key} is labelled {len(instability.counterparts)} ways: "
        else:
            text = f"Subject label {instability.label} names {len(instability.counterparts)} keys: "
        say("measurement", text + _listed(instability.counterparts) + ".")
    if bundle.apparatus_confounds:
        say(
            "apparatus",
            "The test setup did not hold still across the campaign: "
            + "; ".join(f"{confound.dimension} ({confound.status})" for confound in bundle.apparatus_confounds)
            + ".",
        )
    unverified = sorted(
        {
            confound.dimension
            for confounds in chain(
                (row.confounded_by for row in bundle.coverage),
                (divergence.confounded_by for divergence in bundle.scope_divergences),
            )
            for confound in confounds
            if confound.kind == "unverified_fold"
        }
    )
    for dimension in unverified:
        reason = bundle.confound_catalog[dimension]
        say("comparisons", reason[:1].upper() + reason[1:] + ".")
    for group in bundle.aliased_factors:
        say("comparisons", group.sentence)
    if bundle.factor_pairs is not None and bundle.factor_pairs.factors:
        say("comparisons", f"{bundle.factor_pairs.completeness} {bundle.factor_pairs.interaction_aliasing}")
        for pivot in bundle.factor_pairs.pivots:
            holes = [f"{cell.row_level} with {cell.column_level}" for cell in pivot.cells if cell.status == "not_run"]
            if holes:
                say(
                    "comparisons",
                    f"{pivot.row_factor} and {pivot.column_factor} co-vary; {len(holes)} of their {len(pivot.cells)} "
                    f"combinations never ran: {_listed(holes)}.",
                )
    if bundle.declared_crossing is not None:
        say("comparisons", bundle.declared_crossing.sentence)
        for state, lead in (("not_run", "Declared arms never run"), ("skipped_by_design", "Arms skipped by design")):
            cells = [" × ".join(cell.levels.values()) for cell in bundle.declared_crossing.cells if cell.state == state]
            if cells:
                say("comparisons", f"{lead}: {'; '.join(cells)}.")
    for row in bundle.coverage:
        not_run = [level.display for level in row.declared_levels if level.state == "not_run"]
        undetermined = [level.display for level in row.declared_levels if level.state == "undetermined"]
        if not_run:
            say("comparisons", f"Declared on axis {row.name} and never run: {_listed(not_run)}.")
        if undetermined:
            say(
                "comparisons",
                f"Declared on axis {row.name}, and whether any run sat at it cannot be established: "
                f"{_listed(undetermined)}.",
            )
        if row.cannot_be_an_arm:
            why = row.cannot_be_an_arm.rstrip(".")
            say("comparisons", f"Declared axis {row.name} reads unswept whatever the runs did: {why}.")
    for merge in bundle.refused_merges:
        dimensions = f" on {_listed(merge.dimensions)}" if merge.dimensions else ""
        say(
            "apparatus",
            f"Arm {arm(merge.variant_key)} was measured under {len(merge.apparatus_class_ids)} test setups that did "
            f"not pool ({merge.reason.replace('_', ' ')}{dimensions}).",
        )
    if bundle.refused_merges_omitted:
        say(
            "apparatus",
            f"{bundle.refused_merges_omitted} more pair(s) of one arm's test setups did not pool; they are left out of "
            "this list, the smallest first.",
        )


def _time_axis_disclosures(
    axis: TimeAxis | None, withheld: str | None, say: Callable[[DisclosureSource, str | None], None]
) -> None:
    """What the time axis's positions are, when the runs span two builds or two days."""
    say("measurement", withheld)
    if axis is None:
        return
    if axis.basis == "release":
        say("measurement", f"The runs span builds of {axis.release_label}, in the order each first ran.")
    else:
        say("measurement", f"The runs span UTC days, not builds: {axis.basis_reason}.")
    for position in axis.positions:
        say("measurement", position.control_disclosure)


def _verdict_order_sentence(bundle: AnalysisContextBundle) -> str | None:
    """The tie-break order the campaign declared over its merit axes, with the bars each tier holds — or nothing."""
    order = bundle.verdict_order
    if not order.merit_priority:
        return None
    tiers = "; ".join(
        f"{tier.axis} ({_listed(tier.bar_measure_ids) if tier.bar_measure_ids else 'no adjudicated bar'})"
        for tier in order.tiers
    )
    unranked = (
        f" Bars on no ranked axis, which the order does not place: {_listed(order.unranked_bar_measure_ids)}."
        if order.unranked_bar_measure_ids
        else ""
    )
    return f"The campaign ranks its merit axes, strongest first, to break a tie no bar decides: {tiers}.{unranked}"


def _production_footing_sentence(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str | None:
    """Say, per arm, what its runs set away from the subject's production configuration; None when nothing moved."""
    parts: list[str] = []
    qualified = False
    for variant_key, pooled in sorted(bundle.arm_production_footings.items()):
        footings = list(pooled.runs.values())
        moved = sorted({f"{name}={level}" for f in footings if f is not None for name, level in f.moved.items()})
        unchecked = sorted({name for f in footings if f is not None for name in f.unchecked})
        unread = sum(1 for f in footings if f is None)
        said = [
            *([f"set {', '.join(moved)}"] if moved else []),
            *([f"left unchecked whether {', '.join(unchecked)} is production's"] if unchecked else []),
            *([f"{unread} run(s) nobody checked"] if unread else []),
        ]
        qualified = qualified or bool(said)
        parts.append(f"arm {arm(variant_key)} " + ("; ".join(said) or "moved nothing off production"))
    if not qualified:
        return None
    return (
        "A per-reply cost is what production would spend only where its arm's runs moved nothing off the subject's "
        "production configuration: " + "; ".join(parts) + "."
    )


def _some_arms(
    bundle: AnalysisContextBundle, cells: Sequence[CellCoordinate], lead: str, arm: Callable[[str], str]
) -> str:
    """`` <lead>: <arms>.`` naming the arms a sentence holds for, or nothing when it holds for every arm."""
    keys = sorted({cell.variant_key for cell in cells})
    if set(keys) >= {board.variant_key for board in bundle.scoreboard}:
        return ""
    return f" {lead}: {'; '.join(arm(key) for key in keys)}."


def _listed(ids: Sequence[str]) -> str:
    """Ids, as a disclosure lists them."""
    return ", ".join(ids)


__all__ = [
    "AnalysisLead",
    "aspect_verdict",
    "contrast_evidence",
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
