"""What code read on each aspect, as one matrix per campaign kind, with every kind's bars, in the page's Scorecard.

The matrix's form follows the campaign's kind (:func:`~threetears.evals.analysis.campaign_readings.campaign_kind`):

- **A change**: one row per aspect, one column per candidate, each head naming the measure and the control
  (``v2 · pass rate vs v1``). Each cell is the candidate's own level, then its change from the control's, signed in
  points at one decimal (``23.0% · +2.5 pts``), its interval drawn against the aspect's margin, and what it shows
  (shown better, shown worse, no worse than the margin, or unclear), with a floor's status beside it where the floor
  reads otherwise. The control's column holds the level the changes are read from, the mean of the cases they read,
  so the levels close on each change: its trials passed where the two agree, else ``mean of N cases``; marked where it
  cannot move (a rate at or below the margin cannot show a drop; one at 100% cannot show a gain). The line under the
  matrix says the glyph's tick is the control and which way is better. Under it, folded, the same readings one row per aspect × candidate with their counts; then the
  lines that change how a cell reads; then one table of what would settle each unclear reading, on the pass rate and
  on every rate a floor reads.
- **A choice**: one row per aspect, one column per arm, each cell the arm's pass rate with its count and the arms it
  is shown ahead of; folded under it, every pair; then the frontier's reading of which arm clears a bar cheapest.
- **A measurement**: one row per aspect, one column per measure, where several arms were measured; for one arm, turned:
  a row per measure on each aspect, its value, interval and cases, with no glyph, a pass rate with its count; folded
  under it, every measure.

A level in a matrix carries the count it is computed over: the control's and a choice's arm's in its cell
(``100.0% (40 of 40 trials, 20 cases)``, or ``20.5% (mean of 380 cases)``), a candidate's in its row's Cases cell and,
counted in trials, in the fold under the matrix. How the intervals were read is the page's methods, not its
scorecard.

A change's or a choice's matrix carries one more row where
:func:`~threetears.evals.analysis.report.campaign.aggregate.equal_weight_cells` gives it each arm's figure across every
aspect (:data:`~threetears.evals.analysis.report.campaign.aggregate.EQUAL_WEIGHT_ROW`), each cell in its column's
unit, which the cell states (``TableCell.unit``) and the matrix checks: in a change, the control's mean pass rate in
percent and each candidate's mean change against it in points; in a choice, each arm's mean pass rate in percent. A
cell in another unit, for an arm with no column, or handed back where no matrix is laid out (a measurement, or a
change or choice with nothing read) is refused, never dropped. Every kind then shows its bars table. Beside each aspect's fold, a few words say what code read there (:func:`aspect_shows`).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import reduce

from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME, AnalysisContextBundle, FamilyComparison
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaign_readings import (
    AspectReading,
    BarStatus,
    CampaignReadings,
    MeasureReading,
    PairReading,
)
from threetears.evals.analysis.quantities import PERCENT, POINTS, is_money
from threetears.evals.analysis.report.campaign import aggregate
from threetears.evals.analysis.report.cost_unit import MoneyRuler
from threetears.evals.analysis.report.campaign.measure_kinds import (
    counted_rates,
    measure_summary,
    money_rulers,
    rate_measures,
    shown_measures,
)
from threetears.evals.analysis.report.figures import MeasureRuler, Rulers, report_rulers
from threetears.evals.analysis.report.forms import PageForms, Printed
from threetears.evals.kernel.surface import DecisionSurface
from threetears.evals.analysis.report.campaign.common import floor_of
from threetears.evals.analysis.report.campaign.names import AspectNames, measure_name_of
from threetears.evals.analysis.report.decision import (
    CasesNeeded,
    RateVerdict,
    SampleProjection,
    aspect_pass_rate,
    case_cost_usd,
    control_cannot_show,
    floor_readings,
    projected_cell,
    rate_projections,
    read_floor,
)
from threetears.evals.analysis.report.blocks import bare_columns
from threetears.evals.analysis.report.model import (
    CellValue,
    DisclosureBlock,
    Report,
    ReportBlock,
    TableBlock,
    TableCell,
    TableColumn,
    table_row,
)
from threetears.evals.analysis.report.verdict import interval_words, passed_clause
from threetears.evals.analysis.report.figure_keys import joint_guarantee
from threetears.evals.analysis.report.parts.words import FAMILY_LEVEL_TEMPLATE
from threetears.evals.analysis.report.words import PASS_RATE, reason_words
from threetears.evals.analysis.stats import INTERVAL_LEVEL, SMALL_N_BAND_FLOOR
from threetears.evals.analysis.style import CANNOT_TELL_APART, cannot_tell_apart, humanize
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import RateDrop

__all__ = [
    "BAR_WORDS",
    "CHOICE_NOTE",
    "READING_WORDS",
    "aspect_shows",
    "bar_blocks",
    "row_figures",
    "row_sentence",
    "campaign_verdict",
    "frontier_blocks",
    "measure_blocks",
    "measured_figure",
    "pair_blocks",
    "pairs_words",
    "settle_blocks",
    "table_blocks",
]

#: What each reading is called on the page. "Shown worse" is a drop shown past the margin, which is worse than the
#: control; the other direction says the margin, since within it the candidate may still be a little worse.
READING_WORDS: dict[str, str] = {
    "better": "shown better",
    "worse": "shown worse",
    "within_margin": "no worse than the margin",
    "unclear": "unclear",
}

#: The name of every kind's matrix: one row per aspect, laid out as the kind's shape asks.
_SCORECARD_TABLE = "scorecard"

#: The matrix's title, for every kind.
_MATRIX_TITLE = "By aspect"

#: The floor status a change's reading implies on the pass rate. A cell is flagged with its floors' status where they
#: read otherwise: a floor reads every rate its clause names at its own level and points, the row the pass rate alone.
_IMPLIED_FLOOR: dict[str, str] = {"worse": "broken", "within_margin": "held", "better": "held", "unclear": "unclear"}

#: What a contrast's verdict is called beside a row it disagrees with.
_CONTRAST_WORDS: dict[str, str] = {
    "improved": "improved",
    "regressed": "regressed",
    "equivalent": "equivalent",
    "not_separated": CANNOT_TELL_APART,
}

#: The contrast verdict that says what a row's reading says, for the two readings a contrast can also reach.
_CONTRAST_OF: dict[str, str] = {"better": "improved", "worse": "regressed"}


def _rate_level(forms: PageForms, level: float) -> Printed:
    """The pass rate's level as a change's or a choice's matrix draws it, in percent, in the page's one form for it,
    in a cell or in words alike."""
    return forms.drawn_level(TRIAL_PASS_NAME, level, kind="trial")


def _rate_held(forms: PageForms, level: float) -> float:
    """The number the pass rate's level, drawn in percent, is held at, for a decision that compares levels and prints
    none of them."""
    return forms.drawn_held(TRIAL_PASS_NAME, level, kind="trial")


def _rate_change(forms: PageForms, change: float) -> Printed:
    """The pass rate's change in points, in the page's one form for it."""
    return forms.change(TRIAL_PASS_NAME, change, kind="trial")


def _level_text(forms: PageForms, level: float) -> str:
    """An interval's level, held as a share, as the page states it: in percent to the precision that tells two levels
    apart."""
    return forms.confidence_share(level).text


def _told_apart(forms: PageForms, level: float) -> float:
    """An interval's level, held as a share, to the precision its text tells two levels apart at: two levels read
    alike, or one reads as the nominal level, where these agree. A comparison prints nothing, so nothing is issued."""
    return forms.told_apart(level)


def _shown_interval(row: AspectReading) -> tuple[tuple[float, float] | None, float | None]:
    """The interval a row's reading was read on, and its level: the gain's for shown better, the drop's for shown
    worse, the nominal level's (a floor's own) for no worse than the margin; for unclear, the wider of the gain's and
    the drop's, which reaches past zero and short of the margin as the reading says."""
    if row.reading == "better":
        return row.gain_interval, row.gain_confidence
    if row.reading == "worse":
        return row.drop_interval, row.drop_confidence
    if row.reading == "within_margin":
        return row.within_interval, INTERVAL_LEVEL
    sides = [
        (interval, level)
        for interval, level in ((row.gain_interval, row.gain_confidence), (row.drop_interval, row.drop_confidence))
        if interval is not None and level is not None
    ]
    if not sides:
        return None, None
    return max(sides, key=lambda side: side[1] or 0.0)


def _against_zero(row: AspectReading) -> bool:
    """Whether a row's drop is read against zero: a question asks its aspect and declares no margin (a superiority
    question)."""
    return row.margin is None and row.confirmatory


def _row_words(row: AspectReading) -> str:
    """What a row shows in a table cell's words (:data:`READING_WORDS`), unmarked as a lead; under a superiority
    question (asked with no margin) an unclear row whose interval was read on enough cases reads ``cannot tell apart``
    (:data:`~threetears.evals.analysis.style.CANNOT_TELL_APART`), as a contrast neither side of zero does. One with no
    interval, or under the case floor, stays ``unclear``, and its reason says why. A sentence says the row with
    :func:`row_sentence`.

    :param row: The row.
    :ptype row: AspectReading
    :return: The words.
    :rtype: str
    """
    if _undecided(row):
        return _CONTRAST_WORDS["not_separated"]
    return READING_WORDS[row.reading]


def row_figures(row: AspectReading, control: str, *, forms: PageForms) -> str:
    """The figures a row's reading rests on, as sentences the lead says after its decision, each arm the subject and
    each figure as the matrix's fold prints it: what each arm passed of its trials, then what the interval the row was
    read on allows the arm to be against the control, in points; and for a row left unclear against its margin, the
    count its To settle row states, one step down. A sentence carries at most three figures
    (:data:`~threetears.evals.analysis.style.MAX_NUMERALS`). The interval's level is the fold's.

    :param row: The row.
    :ptype row: AspectReading
    :param control: The control, as the page names it.
    :ptype control: str
    :param forms: The page's forms.
    :ptype forms: PageForms
    :return: The sentences.
    :rtype: str
    """
    unit = "trials" if row.control_trials != row.control_cases or row.arm_trials != row.arm_cases else "cases"
    if row.control_trials == row.arm_trials:
        said = [
            f"Of {row.control_trials} {unit} each, {control} passed {row.control_passed} and {row.arm_label} "
            f"{row.arm_passed}."
        ]
    else:
        said = [
            f"Of {row.control_trials} {unit}, {control} passed {row.control_passed}.",
            f"Of {row.arm_trials}, {row.arm_label} passed {row.arm_passed}.",
        ]
    interval, _ = _shown_interval(row)
    if interval is not None:
        # Both ends at one precision, as the interval's cell prints them: a point's one decimal, more where an end that
        # is not zero would read as one: a set reads at one precision. Each end is sized by its distance from zero.
        sized = forms.sizes(TRIAL_PASS_NAME, interval[0], interval[1], kind="trial")
        size = {abs(interval[0]): sized.low_text, abs(interval[1]): sized.high_text}
        said.append(
            interval_words(
                row.arm_label,
                interval[0],
                interval[1],
                amount=lambda value: size[abs(value)],
                unit=" points",
            )
        )
    need = row.projection.if_observed if row.projection is not None else None
    if row.reading == "unclear" and row.margin is not None and need is not None and need.kind == "more":
        settles = forms.count(need.cases).text
        said.append(f"If the arms differ as observed, settling it takes {settles} cases in all (the To settle table).")
    return " ".join(said)


def _undecided(row: AspectReading) -> bool:
    """Whether a row reads as a contrast neither side of zero does: unclear under a superiority question, its interval
    read on enough cases."""
    read = row.gain_interval is not None and min(row.control_cases, row.arm_cases) >= SMALL_N_BAND_FLOOR
    return row.reading == "unclear" and _against_zero(row) and read


def reads_undecided(readings: CampaignReadings, aspect: str) -> bool:
    """Whether what the page says of an aspect's rows or pairs leaves any arms undecided, so its words say the evidence
    cannot tell them apart (:func:`row_sentence`, :func:`pairs_words`): a row read against zero whose interval reaches
    either side of it, or a pair neither of whose arms is ahead. Decided from the readings, never from the words.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param aspect: The aspect's label.
    :ptype aspect: str
    :return: Whether some arms on it are undecided.
    :rtype: bool
    """
    return any(_undecided(row) for row in readings.on_aspect(aspect)) or any(
        pair.ahead is None for pair in readings.pairs if pair.aspect == aspect
    )


def row_sentence(row: AspectReading, control: str, *, against: bool = True) -> str:
    """What a row shows as a sentence says it, unmarked as a lead: ``v2 against v1 shown better``, or for a row the
    evidence leaves undecided ``the evidence cannot tell v2 and v1 apart``
    (:func:`~threetears.evals.analysis.style.cannot_tell_apart`). ``against`` False leaves the control unnamed where
    the row is read beside others on one aspect (``v2 shown better``).

    :param row: The row.
    :ptype row: AspectReading
    :param control: The control, as the page names it.
    :ptype control: str
    :param against: Whether the control is named.
    :ptype against: bool
    :return: The clause, with no closing stop.
    :rtype: str
    """
    if _undecided(row):
        return cannot_tell_apart([row.arm_label], control)
    interval, _ = _shown_interval(row)
    said = (
        f"{row.arm_label} against {control} {READING_WORDS[row.reading]}"
        if against
        else f"{row.arm_label} {READING_WORDS[row.reading]}"
    )
    if row.reading == "within_margin" and interval is not None and interval[1] < 0:
        # The reading in its own term first; that its whole interval lies below the control is the detail after it.
        return f"{said}, though below {control}"
    return (
        f"{row.arm_label} against {control} {READING_WORDS[row.reading]}"
        if against
        else (f"{row.arm_label} {READING_WORDS[row.reading]}")
    )


def _reading_words(row: AspectReading) -> str:
    """What a row shows, a lead marked as one."""
    words = _row_words(row)
    return words if row.confirmatory else f"{words}, a lead"


def _counted(passed: int, trials: int, cases: int) -> str:
    """An arm's passes as a table states them: of its trials, which are named as trials where they are not its cases,
    so a count over repeats never reads as one over cases."""
    return f"{passed} of {trials}" if trials == cases else f"{passed} of {trials} trials"


def _rate_count(passed: int, trials: int, cases: int) -> str:
    """The count a matrix's rate is computed over, beside it: ``40 of 40 trials, 20 cases``, or ``20 of 20 cases``
    where each case ran once."""
    return f"{passed} of {trials} cases" if trials == cases else f"{passed} of {trials} trials, {cases} cases"


def _marks(*marks: str | None) -> str | None:
    """A cell's marks, joined, or None for none."""
    return "; ".join(mark for mark in marks if mark) or None


def _cases(counts: Sequence[int]) -> CellValue:
    """The cases a matrix row was read over: one count, or the fewest to the most where its cells read different
    counts."""
    held = sorted(set(counts))
    if not held:
        return None
    return held[0] if len(held) == 1 else f"{held[0]} to {held[-1]}"


def _arm_column(key: str) -> str:
    """The column an arm's cells sit in, by its variant key: never a column's own key (``aspect``, ``cases``)."""
    return f"arm:{key}"


def _with_equal_weight_row(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    columns: Sequence[TableColumn],
    arms: Mapping[str, tuple[str, str]],
    levels: Mapping[str, tuple[float, str | None]] | None = None,
    *,
    forms: PageForms,
) -> list[dict[str, TableCell]]:
    """The row across every aspect, each aspect weighed alike, with each arm's cell
    (:func:`~threetears.evals.analysis.report.campaign.aggregate.equal_weight_cells`) in that arm's column; no row
    where no cell is given. ``arms`` is each arm's column and the unit it holds, by variant key. A figure states its
    unit, which must be its column's; it is placed without it where the column's header names it (``alpha (%)``),
    else with it at the unit's one decimal, a change signed, each held as the page's forms hold the matrix's own cells.
    A withheld figure (no value) carries none.

    A change's ``levels`` are each arm's level across the aspects on its change's own basis, by variant key, with any
    words its cell adds: the equal-weight mean of the levels printed above it (:func:`_across_levels`). The row's
    change is the equal-weight mean of the aspects' changes, so the control's figure is that mean of its cells, said
    so once on its cell, and each candidate's change is printed after its own (``23.4% · +1.9 pts``); the arithmetic
    then closes within rounding. The control's interval is the aggregate's, on its pooled rate, so the cell draws
    none beside a figure on another basis.

    :raises ValueError: A cell names an arm the matrix has no column for, holds words, or states a unit other than its
        column's: the matrix would drop it or print it under the wrong unit.
    """
    across = aggregate.equal_weight_cells(campaign, readings, reports)
    if not across:
        return []
    if stray := sorted(set(across) - set(arms)):
        raise ValueError(
            f"the {aggregate.EQUAL_WEIGHT_ROW!r} row names arm(s) {', '.join(stray)}, and the scorecard has a column "
            f"only for {', '.join(sorted(arms)) or 'none'}"
        )
    row = {column.key: TableCell(value=None) for column in columns}
    row["aspect"] = TableCell(value=aggregate.EQUAL_WEIGHT_ROW)
    bare = bare_columns(columns)
    held = levels or {}
    for key, cell in across.items():
        column, unit = arms[key]
        if cell.value is not None and (not cell.is_number or cell.unit != unit):
            stated = repr(cell.unit) if cell.is_number else f"the words {cell.value!r}"
            raise ValueError(
                f"the {aggregate.EQUAL_WEIGHT_ROW!r} cell for arm {key} states {stated}, and its column holds {unit!r}"
            )
        if column in bare:
            placed: dict[str, object] = {"unit": ""}
            if cell.value is not None:
                across_figure = _across(forms, float(cell.value), unit)
                placed |= {"value": across_figure.held, "decimals": across_figure.decimals}
            row[column] = cell.model_copy(update=placed)
        elif cell.value is not None:
            # A head that names no unit leaves it on each figure, at the unit's one decimal as the column's cells are:
            # a candidate's change signed, after its level across the aspects, as every change cell reads. A control
            # with its level across the aspects prints that level in place of its own figure.
            level, said = held.get(key, (None, None))
            figure = _across(forms, float(cell.value), unit) if level is None or unit == POINTS else None
            update: dict[str, object] = {"signed": unit == POINTS}
            if figure is not None:
                update |= {"value": figure.held, "decimals": figure.decimals}
            if level is not None:
                shown = _rate_level(forms, level)
                if unit == POINTS and figure is not None:
                    # Where the change lands, then the change, each in the pass rate's one form.
                    update |= {"level": shown.held, "value_text": f"{shown.text} · {figure.text}"}
                else:
                    update |= {"value": shown.held, "decimals": shown.decimals, "low": None, "high": None}
                if said is not None:
                    update["flag"] = said
            row[column] = cell.model_copy(update=update)
        else:
            row[column] = cell
        # A figure of the pass rate across the aspects, which the page's column rules record the cell under.
        row[column] = forms.of_measure(row[column], measure=TRIAL_PASS_NAME, kind="trial")
    return [row]


def _across(forms: PageForms, value: float, unit: str) -> Printed:
    """An equal-weight figure in its column's unit, as the matrix's own cells are held: a change in points, else a pass
    rate in percent."""
    return _rate_change(forms, value) if unit == POINTS else _rate_level(forms, value)


def _no_equal_weight_row(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report], why: str
) -> None:
    """Refuse a figure across every aspect where no matrix is laid out to hold it, rather than drop it.

    :raises ValueError: :func:`~threetears.evals.analysis.report.campaign.aggregate.equal_weight_cells` gives a cell.
    """
    if across := aggregate.equal_weight_cells(campaign, readings, reports):
        raise ValueError(
            f"the {aggregate.EQUAL_WEIGHT_ROW!r} row names arm(s) {', '.join(sorted(across))}, and the scorecard lays "
            f"out no matrix to hold it: {why}"
        )


# =============================================================================
# A change: the matrix, its rows folded, the lines under it, and what would settle each unclear reading
# =============================================================================


def table_blocks(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    *,
    names: AspectNames,
    forms: PageForms,
) -> list[ReportBlock]:
    """A change's scorecard: the matrix of aspects × candidates, the same readings row by row in a fold, and the lines
    under them that change how a cell reads.

    :param campaign: The campaign, a change.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :param forms: The page's forms.
    :ptype forms: PageForms
    :return: The matrix, its folded rows and its lines; one line where no control was named.
    :rtype: list[ReportBlock]
    :raises ValueError: An equal-weight cell cannot be placed: its arm has no column, its unit is not its column's, or
        no matrix is laid out.
    """
    if readings.control is None:
        _no_equal_weight_row(campaign, readings, reports, "no control was named")
        return [
            DisclosureBlock(
                section="scorecard",
                source="comparisons",
                text="Each arm is read on its own figures: a change is read against a named control.",
            )
        ]
    blocks: list[ReportBlock] = []
    if readings.readings:
        blocks += [
            _change_matrix(campaign, readings, reports, names, readings.control, forms),
            _change_rows(readings, names, forms),
        ]
    else:
        _no_equal_weight_row(campaign, readings, reports, "no candidate was compared on any aspect")
    levels = [
        level
        for row in readings.readings
        if row.reading != "within_margin" and (level := _shown_interval(row)[1]) is not None
    ]
    # How the intervals were read is one paragraph, a sentence each: the level, the error rate, the rows no worse. It is
    # a method, so it is the page's methods, not its scorecard.
    read = [
        *(_level_lines(levels, readings, forms) if readings.readings else []),
        *_within_lines(readings, names, forms),
    ]
    if read:
        blocks.append(DisclosureBlock(section="methods", source="comparisons", text=" ".join(read)))
    notes = [
        _rig_line(names(row.aspect), f"{row.arm_label}'s change against {readings.control_label}", row.rig)
        for row in readings.readings
        if row.rig is not None
    ]
    notes += _reconciled_lines(readings, reports, names, forms)
    notes += _unread_lines(campaign, readings, names)
    blocks += [DisclosureBlock(section="scorecard", source="comparisons", text=note) for note in notes if note]
    return blocks


def _change_matrix(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    names: AspectNames,
    control: str,
    forms: PageForms,
) -> TableBlock:
    """One row per aspect any candidate was compared on, one column per candidate, the control's pass rate beside the
    aspect; then the row across every aspect, where one is given. Each column's head names its arm and the measure, a
    candidate's what its change is against (``v2 · pass rate vs v1``, three words, a head's most), and each cell states
    its unit, so a change never reads as a bare number. The line under the matrix says what each glyph's mark is (:func:`_glyph_key`)."""
    labels = readings.arm_labels()
    named = readings.control_label
    candidates = list(dict.fromkeys(row.arm for row in readings.readings))
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="cases", header="Cases", term="cases"),
        TableColumn(key=_arm_column(control), header=f"▸ {named} · {_MEASURE}", term="control"),
        *(
            TableColumn(key=_arm_column(arm), header=f"{labels[arm]} · {_MEASURE} vs {named}", term="change")
            for arm in candidates
        ),
    ]
    rows: list[dict[str, TableCell]] = []
    # Each candidate's baseline on each aspect, and whether some aspect's candidates read different cases.
    stated: dict[str, list[float]] = {}
    split = False
    for aspect in campaign.aspects:
        on = {row.arm: row for row in readings.on_aspect(aspect.label)}
        if not on:
            continue
        bundle = reports[aspect.label].bundle
        baselines = {arm: _baseline(row, bundle, control) for arm, row in on.items()}
        shown = _control_cell(list(on.values()), bundle, control, baselines, forms)
        # Where the candidates' changes read different cases, the control's cell is its trials passed, and every
        # candidate names the baseline its own change is read from.
        uneven = len({(_rate_held(forms, level), cases) for level, cases in baselines.values()}) > 1
        split = split or uneven
        for arm, (level, _) in baselines.items():
            stated.setdefault(arm, []).append(level)
        cells = {
            "aspect": names.cell(aspect.label),
            "cases": TableCell(value=_cases([min(row.control_cases, row.arm_cases) for row in on.values()])),
            _arm_column(control): shown,
        }
        for arm in candidates:
            row = on.get(arm)
            cells[_arm_column(arm)] = (
                TableCell(value=None)
                if row is None
                else _change_cell(row, campaign, readings, baselines[arm][0], forms, stated=uneven)
            )
        rows.append(cells)
    arms = {control: (_arm_column(control), PERCENT), **{key: (_arm_column(key), POINTS) for key in candidates}}
    rows += _with_equal_weight_row(
        campaign,
        readings,
        reports,
        columns,
        arms,
        _across_levels(control, named, rows, stated, split=split, forms=forms),
        forms=forms,
    )
    return TableBlock(
        section="scorecard",
        name=_SCORECARD_TABLE,
        shared_scale=True,
        title=_MATRIX_TITLE,
        columns=columns,
        rows=rows,
        order="aspect by aspect, each candidate's change against the control",
        total_rows=len(rows),
        glyph_key=_glyph_key(readings, [labels[arm] for arm in candidates], forms),
    )


def _across_levels(
    control: str,
    named: str,
    rows: Sequence[Mapping[str, TableCell]],
    stated: Mapping[str, Sequence[float]],
    *,
    split: bool,
    forms: PageForms,
) -> dict[str, tuple[float, str | None]]:
    """Each arm's level across the aspects for the row under a change's matrix, on its change's own basis: the
    equal-weight mean of the levels its column prints above (the control's, each its changes' baseline; a
    candidate's, each its own level), with the words its cell adds. An arm lacking a level on some aspect has none
    here, as the aggregate withholds its change across them there. Where some aspect's candidates read different
    cases (``split``), the control's cell there is its trials passed, so each candidate names the equal-weight mean of
    its own baselines (``stated``), as its cells above name each.

    :param control: The control's variant key.
    :ptype control: str
    :param named: The control as the page names it.
    :ptype named: str
    :param rows: The matrix's rows, one per aspect.
    :ptype rows: Sequence[Mapping[str, TableCell]]
    :param stated: Each candidate's baseline on each aspect it was read on, by variant key.
    :ptype stated: Mapping[str, Sequence[float]]
    :param split: Whether some aspect's candidates read different cases.
    :ptype split: bool
    :param forms: The page's forms.
    :ptype forms: PageForms
    :return: Each arm's level and its cell's words (None for none), by variant key.
    :rtype: dict[str, tuple[float, str | None]]
    """

    def mean(key: str, read: Callable[[TableCell], float | None]) -> float | None:
        figures = [read(row[_arm_column(key)]) for row in rows]
        held = [figure for figure in figures if figure is not None]
        return math.fsum(held) / len(held) if held and len(held) == len(figures) else None

    levels: dict[str, tuple[float, str | None]] = {}
    if (base := mean(control, lambda cell: float(cell.value) if cell.is_number else None)) is not None:  # type: ignore[arg-type]
        levels[control] = (base, _ACROSS_BASIS)
    for key, baselines in stated.items():
        if key == control or (level := mean(key, lambda cell: cell.level)) is None:
            continue
        own = f"from {named}'s {_rate_level(forms, math.fsum(baselines) / len(baselines)).text}" if split else None
        levels[key] = (level, own)
    return levels


#: The basis the row under a change's matrix reads its levels on, said once on the control's cell.
_ACROSS_BASIS = "mean of the aspects above, each weighted equally"


#: The measure a change's matrix reads, which every one of its column heads names: the scorecard's own, whatever
#: measure the question asks, which the lead names.
_MEASURE = PASS_RATE.lower()


def _glyph_key(readings: CampaignReadings, candidates: Sequence[str], forms: PageForms) -> str:
    """What each glyph of a change's matrix draws, which the line under it opens with: whose change, from which
    control (the tick), what its interval guarantees (every interval drawn's, :func:`_shown_interval`), and which way
    is better (a pass rate's gain is to the right). An interval corrected across an arm's aspects states the joint
    guarantee its arm's readings hold at together (owner ruling, level in headings), never its own level, which the
    Methods paragraph states (:func:`_level_lines`)."""
    levels = {
        _told_apart(forms, level)
        for row in readings.readings
        for interval, level in [_shown_interval(row)]
        if interval is not None and level is not None
    }
    corrected = any(level > _told_apart(forms, INTERVAL_LEVEL) for level in levels)
    gains = Counter(row.arm for row in readings.readings if row.gain_interval is not None)
    drops = Counter(row.arm for row in readings.readings if row.drop_interval is not None)
    sizes = {max(gains[arm], drops[arm]) for arm in {row.arm for row in readings.readings}}
    guarantee = joint_guarantee(readings.alpha, max(sizes, default=0), corrected=corrected)
    if not levels:
        interval_said = "its interval"
    elif guarantee is None:
        interval_said = f"its {_level_text(forms, min(levels))} interval"
    elif len(sizes) == 1:
        joint = FAMILY_LEVEL_TEMPLATE.format(joint=f"{_level_text(forms, guarantee[0])} interval", size=guarantee[1])
        interval_said = f"its {joint}"
    else:
        interval_said = f"its {_level_text(forms, guarantee[0])} interval, adjusted across that candidate's aspects"
    whose = f"{candidates[0]}'s" if len(candidates) == 1 else "a candidate's"
    return (
        f"Each mark is {whose} change from {readings.control_label} (the tick) with {interval_said}; right is better."
    )


def _baseline(row: AspectReading, bundle: AnalysisContextBundle, control: str) -> tuple[float, int | None]:
    """The control's level a row's change is read from, in percent, and the cases it is a mean over: the mean of the
    control's per-case shares the change reads (the change is the candidate's mean less this one), so the candidate's
    level, this plus the change, closes with it on the page. Where no per-case shares were read the change is on the
    trial counts, and so is its baseline, over no cases named."""
    rate = aspect_pass_rate(bundle, control, row.arm)
    if rate is not None and rate.samples is not None and rate.samples[0]:
        shares = rate.samples[0]
        return 100 * math.fsum(shares) / len(shares), len(shares)
    return (100 * row.control_passed / row.control_trials if row.control_trials else 0.0), None


def _control_cell(
    rows: Sequence[AspectReading],
    bundle: AnalysisContextBundle,
    control: str,
    baselines: Mapping[str, tuple[float, int | None]],
    forms: PageForms,
) -> TableCell:
    """The control's pass rate on an aspect at one decimal, and what it is over: the level every candidate's change on
    it is read from (``baselines``), which is its trials passed (``18 of 20 cases``) where the two agree, else the mean
    of the cases the changes read (``mean of 380 cases``), its trial count left to the fold. Where the candidates'
    changes read different cases, it is its trials passed, and each candidate's cell names its own baseline. Marked
    where it cannot move (:func:`~threetears.evals.analysis.report.decision.control_cannot_show`, the rule the writer's
    view flags too)."""
    row = rows[0]
    if not row.control_trials:
        return TableCell(value=None)
    pooled = 100 * row.control_passed / row.control_trials
    bounds = control_cannot_show(
        bundle,
        control,
        [read.arm for read in rows],
        # Under a superiority question a drop is read against zero, so a control passing nothing cannot show one.
        margin=0.0 if _against_zero(row) else row.margin,
        passed=row.control_passed,
        trials=row.control_trials,
    )
    marks = [f"cannot show a {bound}" for bound in bounds]
    count = _rate_count(row.control_passed, row.control_trials, row.control_cases)
    # The baselines are compared at the precision they read at; only the level the cell prints is issued.
    level = pooled
    held = {(_rate_held(forms, figure), cases): figure for figure, cases in baselines.values()}
    if len(held) == 1:
        (((shown, cases), level),) = held.items()
        if cases is not None and shown != _rate_held(forms, pooled):
            count = f"mean of {cases} cases"
    printed = _rate_level(forms, level)
    cell = TableCell(value=printed.held, unit=printed.unit, decimals=printed.decimals, flag=_marks(count, *marks))
    # A figure of the pass rate, which the page's column rules record the cell under.
    return forms.of_measure(cell, measure=TRIAL_PASS_NAME, kind="trial")


def _change_cell(
    row: AspectReading,
    campaign: EvalCampaign,
    readings: CampaignReadings,
    baseline: float,
    forms: PageForms,
    *,
    stated: bool,
) -> TableCell:
    """A candidate's change on one aspect: the level it reads at, then its change from the control's, signed and in
    points at one decimal (``23.0% · +2.5 pts``); the interval its reading was read on, drawn against the margin where
    two or more cases bound it; the reading, and its floors' status where they read otherwise. Where the aspect's
    candidates read different baselines (``stated``), the control's cell is its trials passed, so the cell names its
    own (``from v1's 86.7%``), and its level less that is its change."""
    marks = [_reading_words(row)]
    if (status := _floor_state(row, campaign, readings)) is not None and status != _IMPLIED_FLOOR[row.reading]:
        marks.append(f"floor {status}")
    if stated:
        marks.append(f"from {readings.control_label}'s {_rate_level(forms, baseline).text}")
    figure = _rate_change(forms, row.change)
    level = _rate_level(forms, baseline + row.change)
    change = TableCell(
        value=figure.held,
        unit=figure.unit,
        decimals=figure.decimals,
        signed=True,
        level=level.held,
        # Where the change lands, then the change, each in the pass rate's one form.
        value_text=f"{level.text} · {figure.text}",
        flag="; ".join(marks),
    )
    interval, _ = _shown_interval(row)
    if interval is not None and min(row.control_cases, row.arm_cases) >= 2:
        change = change.model_copy(
            update={"low": interval[0], "high": interval[1], "margin": row.margin, "zero_line": True}
        )
    # A figure of the pass rate, which the page's column rules record the cell under.
    return forms.of_measure(change, measure=TRIAL_PASS_NAME, kind="trial")


def _floor_state(row: AspectReading, campaign: EvalCampaign, readings: CampaignReadings) -> str | None:
    """Where the rate floors on a row's aspect stand for its candidate, read together as each is read: broken where
    one is, held where every one is, unclear otherwise; None where no rate floor is declared there. A cost floor reads
    no pass rate, so it never marks a change."""
    statuses = [
        status.status
        for status in readings.floors
        if status.aspect == row.aspect
        and status.arm == row.arm
        and (floor := floor_of(campaign.floors, status.floor)) is not None
        and isinstance(floor.clause, RateDrop)
    ]
    if not statuses:
        return None
    if "broken" in statuses:
        return "broken"
    return "held" if all(status == "held" for status in statuses) else "unclear"


def _change_rows(readings: CampaignReadings, names: AspectNames, forms: PageForms) -> TableBlock:
    """The matrix's readings one row per aspect × candidate, folded: the cases, each arm's passes, the change and the
    interval it was read on, the margin and the reading."""
    control = readings.control_label
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="arm", header="Arm"),
        TableColumn(key="cases", header="Cases", term="cases"),
        TableColumn(key="control", header=f"▸ {control}", term="control"),
        TableColumn(key="passed", header="Arm passed", term="passed"),
        TableColumn(key="change", header="Change", term="change"),
        TableColumn(key="interval", header="Interval (pts)", term="interval"),
        TableColumn(key="margin", header="Margin (pts)", term="margin"),
        TableColumn(key="reading", header="Reading"),
    ]
    rows = []
    for row in readings.readings:
        interval, _ = _shown_interval(row)
        change = _rate_change(forms, row.change)
        rows.append(
            table_row(
                {
                    "aspect": names.cell(row.aspect),
                    "arm": row.arm_label,
                    "cases": min(row.control_cases, row.arm_cases),
                    "control": _counted(row.control_passed, row.control_trials, row.control_cases),
                    "passed": _counted(row.arm_passed, row.arm_trials, row.arm_cases),
                    # Signed and in points at one decimal, as the matrix cell reads it.
                    "change": forms.of_measure(
                        TableCell(value=change.held, unit=change.unit, decimals=change.decimals, signed=True),
                        measure=TRIAL_PASS_NAME,
                        kind="trial",
                    ),
                    "interval": None
                    if interval is None
                    else forms.of_measure(
                        TableCell(
                            # Bare, under the column's ``(pts)``.
                            value=forms.interval(
                                TRIAL_PASS_NAME, interval[0], interval[1], kind="trial", change=True
                            ).digits
                        ),
                        measure=TRIAL_PASS_NAME,
                        kind="trial",
                    ),
                    "margin": row.margin,
                    "reading": _reading_words(row),
                }
            )
        )
    return TableBlock(
        section="scorecard",
        name="aspects",
        title="By aspect, arm by arm",
        columns=columns,
        rows=rows,
        order="aspect by aspect, each arm against the control",
        total_rows=len(rows),
        details=True,
    )


def _rig_line(aspect: str, what: str, rig: str) -> str:
    """Which test setup a row's change and interval were read under, where the arms ran under several."""
    return (
        f"On {aspect}, {what} and its interval are read under test setup {short_digest(rig)}, the first both ran "
        "under; the pass counts are over every setup."
    )


def _level_lines(levels: Sequence[float], readings: CampaignReadings, forms: PageForms) -> list[str]:
    """The level the corrected intervals were read at (``levels``: every reading's but one no worse than the margin,
    whose sentence is :func:`_within_lines`; none where every reading is) and why it is wider than a plain interval,
    then how often one arm is shown better or worse wrongly, and, with several arms, how often some arm is, since the
    correction does not run across them. Each count is the page's own, and counts what is corrected together: an
    arm's gains are its rows with an interval, its drops those with an interval on an aspect with a margin or asked a
    question (``drop_interval``), as its two families are read. One sentence each, every one a whole template."""
    families = Counter(row.arm for row in readings.readings)
    gains = Counter(row.arm for row in readings.readings if row.gain_interval is not None)
    drops = Counter(row.arm for row in readings.readings if row.drop_interval is not None)
    sizes = {(gains[arm], drops[arm]) for arm in families}
    lines: list[str] = []
    if levels:
        low, high = min(levels), max(levels)
        if _told_apart(forms, high) <= _told_apart(forms, INTERVAL_LEVEL):
            # A family of one is read at the nominal level: no reading is corrected.
            lines.append(
                f"The interval is read at {_level_text(forms, low)}, as a floor's is."
                if len(levels) == 1
                else f"Each interval is read at {_level_text(forms, low)}, as a floor's is."
            )
        else:
            if len(sizes) > 1:
                lines.append("Each arm's readings are corrected together.")
            else:
                ((gained, dropped),) = sizes
                lines.append(
                    f"Each arm's {gained} readings are corrected together."
                    if gained == dropped
                    else f"Each arm's gains on {gained} aspects, and its drops on {dropped}, are each corrected "
                    "together."
                )
            # The Methods paragraph, the one place the per-interval level is stated (owner ruling, level in headings).
            read_at = (
                _level_text(forms, low)
                if _told_apart(forms, low) == _told_apart(forms, high)
                else f"{_level_text(forms, low)} to {_level_text(forms, high)}"
            )
            lines.append(f"So each interval is read at {read_at}, not {_level_text(forms, INTERVAL_LEVEL)}.")
    half = _level_text(forms, readings.alpha / 2)
    lines.append(
        f"For any one arm, the chance of being wrongly shown better on some aspect is at most {half}, and likewise "
        "for wrongly shown worse."
        if any(max(size) > 1 for size in sizes)
        else f"For any one arm, the chance of being wrongly shown better is at most {half}, and likewise for wrongly "
        "shown worse."
    )
    if len(families) > 1:
        lines.append(
            f"The correction does not run across the {len(families)} arms, so the chance that some arm is wrongly "
            f"shown better is up to {_level_text(forms, min(1.0, len(families) * readings.alpha / 2))}, and likewise "
            "for worse."
        )
    return lines


def _within_lines(readings: CampaignReadings, names: AspectNames, forms: PageForms) -> list[str]:
    """Where a row reads no worse than the margin, the level it was read at and why that needs no correction; where
    such a row's interval sits above zero, that it is still no gain shown, naming each such row."""
    within = [row for row in readings.readings if row.reading == "within_margin"]
    if not within:
        return []
    lines = [
        f"A row no worse than the margin reads {_level_text(forms, INTERVAL_LEVEL)}, uncorrected: an arm is no worse only where "
        "it holds on every asked aspect at once."
    ]
    clearing = [
        f"{row.arm_label} on {names(row.aspect)}"
        for row in within
        if row.within_interval is not None and row.within_interval[0] > 0
    ]
    if clearing:
        lines.append(
            f"Where such an interval sits above zero, as for {_listed(clearing)}, that is not a gain shown: a gain "
            "is read on the corrected interval, which reaches zero."
        )
    return lines


def _listed(items: Sequence[str]) -> str:
    """Items in a sentence: ``a``, ``a and b``, ``a, b and c``."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _reconciled_lines(
    readings: CampaignReadings, reports: Mapping[str, Report], names: AspectNames, forms: PageForms
) -> list[str]:
    """Where a contrast in an aspect's fold reads the very per-case values a row reads and reaches another verdict, one
    line naming both and the interval each was read on, so the disagreement is explained where it is seen.

    A row's gain is read on a bounded interval corrected across its arm's aspects; a contrast on the interval its
    comparison was drawn by (``FamilyComparison.interval_basis``: a rate's bounded one, another reading's t interval
    unless its values have no spread) corrected as its aspect's family says: a bundle's family is always
    Holm-corrected (``ComparisonFamily.correction``), so the page line says corrected. A contrast reading a drop past
    zero where the row reads none past the margin is said too, with the margin. An untested contrast reads nothing."""
    control = readings.control
    if control is None:
        return []
    lines: list[str] = []
    for row in readings.readings:
        _, level = _shown_interval(row)
        bundle = reports[row.aspect].bundle
        rate = aspect_pass_rate(bundle, control, row.arm)
        if level is None or rate is None or rate.samples is None:
            continue
        for comparison, _ in _contrasts(bundle, control, row.arm):
            if comparison.verdict not in _CONTRAST_WORDS or comparison.case_values is None:
                continue
            if not _same_values(comparison.case_values, rate.samples) or (comparison.test == "paired") != rate.paired:
                continue
            said = comparison.verdict if comparison.verdict in ("improved", "regressed") else None
            if said == _CONTRAST_OF.get(row.reading):
                continue
            lines.append(_reconciled(row, comparison, bundle, level, names(row.aspect), forms))
    return lines


def _contrasts(bundle: AnalysisContextBundle, control: str, arm: str) -> list[tuple[FamilyComparison, float]]:
    """Every contrast of ``arm`` against ``control`` the aspect's families tested on a reading where higher is better,
    each with its family's interval level."""
    return [
        (comparison, family.interval_level)
        for family in bundle.multiple_comparisons.families
        if family.interval_level is not None
        for comparison in family.comparisons
        if comparison.control.variant_key == control
        and comparison.contrast.variant_key == arm
        and comparison.higher_is_better
    ]


def _same_values(
    values: tuple[Sequence[float], Sequence[float]], samples: tuple[Sequence[float], Sequence[float]]
) -> bool:
    """Whether two pairs of per-case values are the same, case by case."""
    return all(
        len(ours) == len(theirs) and all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(ours, theirs, strict=True))
        for ours, theirs in zip(values, samples, strict=True)
    )


def _reconciled(
    row: AspectReading,
    comparison: FamilyComparison,
    bundle: AnalysisContextBundle,
    level: float,
    aspect: str,
    forms: PageForms,
) -> str:
    """One row and the contrast reading its values to another verdict: both verdicts, then how each was read."""
    if comparison.reading == "trial":
        reading = "the pass rate"
    elif comparison.reading == "judged":
        reading = humanize("dimension", comparison.name)
    else:
        reading = measure_name_of(bundle, comparison.name)
    # Each interval by what it guarantees, never its own adjusted level (owner ruling, level in headings): the row's at
    # the nominal level on its own, or corrected across the arm's aspects; the contrast's corrected in its family.
    own = (
        f"at {_level_text(forms, level)} on its own"
        if _told_apart(forms, level) <= _told_apart(forms, INTERVAL_LEVEL)
        else f"corrected across {row.arm_label}'s aspects"
    )
    # The interval the contrast is drawn on, which is not always the test its p came from: a rate's is its bounded one.
    basis = comparison.interval_basis or ("t" if comparison.basis == "t" else "bounded")
    said = (
        f"On {aspect}, {row.arm_label}'s row reads {_row_words(row)} and its contrast on {reading} reads "
        f"{_CONTRAST_WORDS[comparison.verdict]}, over the same per-case values. The row reads a bounded interval "
        f"{own}; the contrast a {basis} interval corrected within the aspect's family."
    )
    if comparison.verdict == "regressed" and row.reading != "worse" and row.margin is not None:
        said += f" The row counts a drop only past the {forms.declared(row.margin).text}-point margin."
    return said


def _unread_lines(campaign: EvalCampaign, readings: CampaignReadings, names: AspectNames) -> list[str]:
    """An aspect where only one side ran, read arm by arm; and an aspect with no margin asked no question, where a drop
    reads as the evidence leaving the arms undecided (asked one, its drop is read against zero): the one line where what the aspect lacks
    changes how its row reads."""
    lines: list[str] = []
    for aspect in campaign.aspects:
        rows = readings.on_aspect(aspect.label)
        if not rows:
            lines.append(f"{names(aspect.label)} holds the runs of one side only, so its figures are read arm by arm.")
        elif aspect.margin is None and not any(row.confirmatory for row in rows):
            lines.append(
                f"On {names(aspect.label)}, the evidence cannot tell a drop of any size from no change: a drop is "
                "read against a margin, and the aspect declares none."
            )
    return lines


@dataclass(frozen=True)
class _Sizing:
    """One unclear reading to settle: the aspect, the arm, what settling it shows, the rate that decides the count
    (None where none can be named), why it is unclear, the projection (None where nothing can be projected), and
    whether it is a floor's."""

    aspect: str
    arm: str
    shows: str
    decided_by: str | None
    why: str
    projection: SampleProjection | None
    floor: bool = False


def settle_blocks(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    *,
    names: AspectNames,
    forms: PageForms,
) -> list[ReportBlock]:
    """One table of what would settle each unclear reading of a change, aspect by aspect and arm by arm: a row left
    unclear against its margin, read on the pass rate alone; and each rate floor left unclear, read on every rate its
    clause names, joining each rate's count side by side as the aspect's own floor table does. Each count is the total
    independent cases, not cases to add, at which the interval would show it, if the arms do not differ and if they
    differ as observed, read at the nominal level over the same cases and values. So the two counts on one arm and
    aspect differ only where the floor reads other rates than the pass rate alone (every rate, or a check it names) or
    reads them against other points than the margin; where a floor's count is the row's own (the pass rate decides
    it, at the margin's points, to the same counts), the two are one row, saying both. The floor's row names the
    rate deciding its as-observed count; its no-difference count may be another rate's. A row on an aspect with no
    margin has nothing to show it within, and a settled reading nothing left to show, so neither is a row here.

    :param campaign: The campaign, a change.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label, whose evidence a floor's count is read from.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :param forms: The page's forms.
    :ptype forms: PageForms
    :return: The table and its note; nothing where no reading is to settle.
    :rtype: list[ReportBlock]
    """
    sizings = [sizing for row in readings.readings for sizing in _sizings(row, campaign, readings, reports, forms)]
    if not sizings:
        return []
    # The arm and both counts lead, so a frame as narrow as a host's (688 px) shows them without scrolling; the
    # aspect, what settling shows and why it is unclear follow, the last in words that wrap.
    columns = [
        TableColumn(key="arm", header="Arm"),
        TableColumn(key="none", header="If no difference"),
        TableColumn(key="observed", header="If as observed"),
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="shows", header="To show"),
        TableColumn(key="decided_by", header="Decided by", term="reading"),
        TableColumn(key="why", header="Why unclear"),
    ]
    rows = [
        table_row(
            {
                "aspect": names.cell(sizing.aspect),
                "arm": sizing.arm,
                "shows": sizing.shows,
                "decided_by": sizing.decided_by,
                "why": sizing.why or None,
                "none": None
                if sizing.projection is None
                else projected_cell(sizing.projection, sizing.projection.if_none, forms=forms),
                "observed": None
                if sizing.projection is None
                else projected_cell(sizing.projection, sizing.projection.if_observed, forms=forms),
            }
        )
        for sizing in sizings
    ]
    return [
        TableBlock(
            section="scorecard",
            name="to_settle",
            title="To settle",
            columns=columns,
            rows=rows,
            order="as the matrix, each arm's reading on the pass rate before its floors",
            total_rows=len(rows),
        ),
        DisclosureBlock(section="scorecard", source="comparisons", text=_settle_note(sizings, forms)),
    ]


def _sizings(
    row: AspectReading,
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    forms: PageForms,
) -> list[_Sizing]:
    """What would settle a row left unclear against its margin, then each rate floor on its aspect left unclear for
    its arm, each counted as the aspect's own page counts it."""
    found: list[_Sizing] = []
    own: _Sizing | None = None
    if row.reading == "unclear" and row.margin is not None:
        own = _Sizing(
            aspect=row.aspect,
            arm=row.arm_label,
            shows=f"no worse than the {forms.declared(row.margin).text}-point margin",
            decided_by=PASS_RATE,
            why=reason_words(row.reason) or "",
            projection=row.projection,
        )
        found.append(own)
    control = readings.control
    if control is None:
        return found
    bundle = reports[row.aspect].bundle
    figures = floor_readings(bundle, control, row.arm, check_names=bundle.named_checks())
    per_case = case_cost_usd(bundle)
    for status in readings.floors:
        floor = floor_of(campaign.floors, status.floor)
        if status.aspect != row.aspect or status.arm != row.arm or status.status != "unclear" or floor is None:
            continue
        clause = floor.clause
        if not isinstance(clause, RateDrop):
            continue
        verdict = read_floor(floor, figures, control=readings.control_label, candidate=row.arm_label)
        # One source for the count and the rate deciding it: where any unclear rate cannot be projected, neither is.
        rates = rate_projections(verdict, per_case_usd=per_case)
        deciding = None if rates is None else _deciding_rate(rates)
        projection = None if rates is None else reduce(SampleProjection.joined, (one for _, one in rates))
        if (
            own is not None
            and deciding is not None
            and deciding.rate.measure is None
            and clause.max_points == row.margin
            and projection == own.projection
        ):
            # The floor's count is the row's own: the pass rate, at the margin's points, over the same cases. One
            # quantity is one row, saying both what it shows and that the floor holds.
            found[found.index(own)] = replace(
                own,
                shows=(
                    f"no worse than the {forms.declared(row.margin).text}-point margin, and the "
                    f"{forms.declared(clause.max_points).text}-point floor holds"
                ),
            )
            own = None
            continue
        found.append(
            _Sizing(
                aspect=row.aspect,
                arm=row.arm_label,
                shows=f"the {forms.declared(clause.max_points).text}-point floor holds",
                decided_by=None if deciding is None else _rate_words(deciding),
                why=reason_words(verdict.reason if deciding is None else deciding.reason) or "",
                projection=projection,
                floor=True,
            )
        )
    return found


def _deciding_rate(rates: Sequence[tuple[RateVerdict, SampleProjection]]) -> RateVerdict:
    """Of a floor's unclear rates and their projections (:func:`~threetears.evals.analysis.report.decision.
    rate_projections`), the one deciding the floor's as-observed count, the column a decision reads first: the one
    needing the most there, then with no difference, the first of equals."""
    best, chosen = rates[0]
    for rate, projection in rates[1:]:
        if _more(projection.if_observed, chosen.if_observed) or (
            projection.if_observed == chosen.if_observed and _more(projection.if_none, chosen.if_none)
        ):
            best, chosen = rate, projection
    return best


def _more(need: CasesNeeded, than: CasesNeeded) -> bool:
    """Whether ``need`` asks strictly more than ``than``, as two floors' needs are weighed (:meth:`CasesNeeded.most`,
    which keeps the first of equals)."""
    return need != than and than.most(need) == need


def _rate_words(rate: RateVerdict) -> str:
    """A rate a floor reads, as the sizing table names it: ``Pass rate``, or the check's or dimension's own name."""
    return PASS_RATE if rate.rate.measure is None else rate.rate.name


def _settle_note(sizings: Sequence[_Sizing], forms: PageForms) -> str:
    """The sentences under the sizing table, each said only where a row needs it: how a count is read, what each side
    assumes, why a floor's count can exceed the pass rate's, what ``none would`` and ``re-run the lost trials`` mean,
    and how rough a count is."""
    needs = [
        need
        for sizing in sizings
        if sizing.projection is not None
        for need in (sizing.projection.if_none, sizing.projection.if_observed)
    ]
    floors = any(sizing.floor for sizing in sizings)
    said = [
        f"Each count is the cases a {_level_text(forms, INTERVAL_LEVEL)} interval needs in all, not cases to add, to rule out a "
        "drop past the margin or the floor's points.",
        "The no-difference count assumes the arms truly do not differ, and the as-observed count that they differ as "
        "this run saw.",
    ]
    if floors:
        said.append("A floor reads every rate its clause names, so its count can need more than the pass rate's.")
    if any(need.kind == "never" for need in needs):
        said.append(
            'A "none would" means the observed drop is already past it, so only a change to the arm could show it '
            "within."
        )
    if any(need.kind == "lost" for need in needs):
        said.append('A "re-run the lost trials" means only the lost trials keep the interval open.')
    said += [
        "A count is rough: it assumes new cases vary as these did, and it would not show an arm better.",
        "Repeats of a case narrow an interval far less than new cases.",
    ]
    return " ".join(said)


# =============================================================================
# Every kind: what each aspect's fold says, and the campaign's verdict line
# =============================================================================


def aspect_shows(readings: CampaignReadings, aspect: str) -> str:
    """What code read on one aspect, beside its fold's heading, by the campaign's kind: ``v2 shown worse``, ``alpha
    ahead of beta``, ``alpha misses a bar``, ``reply passed 40 of 40 trials on 20 cases``.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param aspect: The aspect's label.
    :ptype aspect: str
    :return: The words.
    :rtype: str
    """
    if readings.kind == "choice":
        return pairs_words(readings, aspect) or "not compared"
    if readings.kind == "measurement":
        missed = [status.arm_label for status in readings.missed_bars() if status.aspect == aspect]
        if missed:
            return f"{', '.join(dict.fromkeys(missed))} misses a bar"
        # The arm's pass rate as its count, as the lead states it, so no fold opens on a bare word.
        figures = [
            figure
            for reading in readings.measures
            # ``passed``: the trials' own pass rate, as the readings key it.
            if reading.aspect == aspect and reading.measure == "passed"
            if (figure := measured_figure(reading)) is not None
        ]
        return "; ".join(figures) or "measured"
    rows = readings.on_aspect(aspect)
    if not rows:
        return "not compared"
    return _rows_sentence(rows, readings.control_label)


def _rows_sentence(rows: Sequence[AspectReading], control: str) -> str:
    """An aspect's rows as one sentence says them: the arms the evidence leaves undecided in one clause
    (``the evidence cannot tell v2, v3 or v4 apart from v1``, :func:`~threetears.evals.analysis.style.cannot_tell_apart`),
    a lead's apart from an asked reading's; then each decided row (``v5 shown better``), the clauses apart by
    semicolons, never one undecided clause repeated per arm."""
    clauses: list[str] = []
    for confirmatory in (True, False):
        undecided = [row.arm_label for row in rows if _undecided(row) and row.confirmatory is confirmatory]
        if undecided:
            lead = "" if confirmatory else ", a lead" if len(undecided) == 1 else ", each a lead"
            clauses.append(cannot_tell_apart(undecided, control) + lead)
    decided = [
        row_sentence(row, control, against=False) + ("" if row.confirmatory else ", a lead")
        for row in rows
        if not _undecided(row)
    ]
    if decided:
        clauses.append(", ".join(decided))
    return "; ".join(clauses)


def campaign_verdict(readings: CampaignReadings, aspects: Sequence[str], names: AspectNames) -> str:
    """The verdict line of a campaign of several aspects, as code writes it: what code read on each aspect, as the
    campaign's page says it beside that aspect's fold, in the campaign's order — ``billing: v2 shown worse; tone: v2
    shown better``. Never the writer's words.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param aspects: The campaign's aspects' labels, in its order.
    :ptype aspects: Sequence[str]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The verdict line.
    :rtype: str
    """
    return "; ".join(f"{names(aspect)}: {aspect_shows(readings, aspect)}" for aspect in aspects)


def pairs_words(readings: CampaignReadings, aspect: str, *, marked: bool = True) -> str | None:
    """Every pair read on one aspect, as a fold's heading and the page's lead say them: ``alpha ahead of beta, the
    evidence cannot tell beta and gamma apart``, a pair the question does not ask marked a lead unless ``marked`` is False (a page
    whose campaign asks nothing, which says so once); None where no pair was read there.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param aspect: The aspect's label.
    :ptype aspect: str
    :param marked: Whether a pair the question does not ask is marked a lead.
    :ptype marked: bool
    :return: The words, or None.
    :rtype: str | None
    """
    pairs = [pair for pair in readings.pairs if pair.aspect == aspect]
    return ", ".join(_pair_words(pair, marked=marked) for pair in pairs) or None


def _pair_words(pair: PairReading, *, marked: bool = True) -> str:
    """A pair as a fold's heading says it: ``alpha ahead of beta``, ``the evidence cannot tell alpha and beta
    apart``."""
    if pair.ahead is None:
        said = cannot_tell_apart([pair.first_label], pair.second_label)
    else:
        ahead, behind = (
            (pair.first_label, pair.second_label) if pair.ahead == pair.first else (pair.second_label, pair.first_label)
        )
        said = f"{ahead} ahead of {behind}"
    return said if pair.confirmatory or not marked else f"{said}, a lead"


# =============================================================================
# A choice: the matrix of arms, every pair folded, and the frontier
# =============================================================================

#: What a choice's notes say once: the one family, and that ahead is a reading.
CHOICE_NOTE = (
    "A pair is read both ways at once, so every pair on every aspect is one family, corrected together. An arm is "
    "ahead only where the whole interval is one side of zero, and ahead is a reading, not a pick."
)


def pair_blocks(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    *,
    names: AspectNames,
    forms: PageForms,
) -> list[ReportBlock]:
    """A choice's scorecard: the matrix of aspects × arms, each cell the arm's pass rate and the arms it is shown ahead
    of; every pair folded under it, each arm's passes, the difference with its interval, and the arm shown ahead or
    no separation; then the lines that change how a row reads.

    :param campaign: The campaign, a choice.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :param forms: The page's forms.
    :ptype forms: PageForms
    :return: The matrix, the pairs and their lines.
    :rtype: list[ReportBlock]
    :raises ValueError: An equal-weight cell cannot be placed: its arm has no column, its unit is not its column's, or
        no matrix is laid out.
    """
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="first", header="First arm"),
        TableColumn(key="second", header="Second arm"),
        TableColumn(key="cases", header="Cases", term="cases"),
        TableColumn(key="first_passed", header="First passed", term="passed"),
        TableColumn(key="second_passed", header="Second passed", term="passed"),
        TableColumn(key="change", header="Change (pts)", term="change"),
        TableColumn(key="interval", header="Interval (pts)", term="interval"),
        TableColumn(key="ahead", header="Ahead"),
    ]
    rows: list[dict[str, TableCell]] = []
    levels: list[float] = []
    for pair in readings.pairs:
        if pair.confidence is not None:
            levels.append(pair.confidence)
        change = _rate_change(forms, pair.change)
        ahead = pair.ahead_label or "no separation"
        cells = table_row(
            {
                "aspect": names.cell(pair.aspect),
                "first": pair.first_label,
                "second": pair.second_label,
                "cases": min(pair.first_cases, pair.second_cases),
                "first_passed": _counted(pair.first_passed, pair.first_trials, pair.first_cases),
                "second_passed": _counted(pair.second_passed, pair.second_trials, pair.second_cases),
                "change": change.held,
                "interval": None
                if pair.interval is None
                else forms.interval(TRIAL_PASS_NAME, *pair.interval, kind="trial", change=True).digits,
                "ahead": ahead if pair.confirmatory else f"{ahead}, a lead",
            }
        )
        # Signed at a point's one decimal, as every other change on the page reads: a pair is direction-free, so an
        # unsigned 5.0 would read as a magnitude.
        cells["change"] = (
            TableCell(value=change.held, decimals=change.decimals, signed=True)
            if pair.interval is None
            else TableCell(
                value=change.held,
                low=pair.interval[0],
                high=pair.interval[1],
                zero_line=True,
                decimals=change.decimals,
                signed=True,
            )
        )
        rows.append(cells)
    blocks: list[ReportBlock] = []
    if readings.pairs:
        blocks.append(_choice_matrix(campaign, readings, reports, names, forms))
    else:
        _no_equal_weight_row(campaign, readings, reports, "no pair of arms was read on any aspect")
    blocks.append(
        TableBlock(
            section="scorecard",
            name="pairs",
            title="By aspect, pair by pair",
            columns=columns,
            rows=rows,
            order="aspect by aspect, each pair of arms in the campaign's order",
            total_rows=len(rows),
            details=True,
        )
    )
    notes = [CHOICE_NOTE]
    if (guarantee := joint_guarantee(readings.alpha, len(levels))) is not None:
        # The package's own words state what the pairs' intervals guarantee together (owner ruling, level in
        # headings): each pair's own step-down level is the Method's, never this line's.
        joint = FAMILY_LEVEL_TEMPLATE.format(joint=_level_text(forms, guarantee[0]), size=guarantee[1])
        notes.append(f"The intervals hold together at {joint}.")
    elif levels:
        notes.append(f"Each interval is read at {_level_text(forms, levels[0])}.")
    notes += [
        _rig_line(names(pair.aspect), f"the difference between {pair.first_label} and {pair.second_label}", pair.rig)
        for pair in readings.pairs
        if pair.rig is not None
    ]
    notes += _unread_without_control_lines(campaign, reports)
    blocks += [DisclosureBlock(section="scorecard", source="comparisons", text=note) for note in notes if note]
    return blocks


def _choice_matrix(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    names: AspectNames,
    forms: PageForms,
) -> TableBlock:
    """One row per aspect any pair was read on, one column per arm: its pass rate over its usable trials, and the
    arms it is shown ahead of there; then the row across every aspect, where one is given."""
    labels = readings.arm_labels()
    arms = list(dict.fromkeys(key for pair in readings.pairs for key in (pair.first, pair.second)))
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="cases", header="Cases", term="cases"),
        *(TableColumn(key=_arm_column(arm), header=f"{labels[arm]} (%)") for arm in arms),
    ]
    rows: list[dict[str, TableCell]] = []
    for aspect in campaign.aspects:
        pairs = [pair for pair in readings.pairs if pair.aspect == aspect.label]
        if not pairs:
            continue
        boards = {board.variant_key: board for board in reports[aspect.label].bundle.scoreboard}
        cells = {
            "aspect": names.cell(aspect.label),
            "cases": TableCell(value=_cases([boards[arm].usable_cases for arm in arms if arm in boards])),
        }
        for arm in arms:
            board = boards.get(arm)
            if board is None or not board.usable:
                cells[_arm_column(arm)] = TableCell(value=None, flag=_ahead_marks(arm, pairs))
                continue
            count = _rate_count(board.passed, board.usable, board.usable_cases)
            cells[_arm_column(arm)] = TableCell(
                value=_rate_level(forms, 100 * board.passed / board.usable).held,
                flag=_marks(count, _ahead_marks(arm, pairs)),
            )
        rows.append(cells)
    rows += _with_equal_weight_row(
        campaign, readings, reports, columns, {arm: (_arm_column(arm), PERCENT) for arm in arms}, forms=forms
    )
    return TableBlock(
        section="scorecard",
        name=_SCORECARD_TABLE,
        shared_scale=True,
        title=_MATRIX_TITLE,
        columns=columns,
        rows=rows,
        order="aspect by aspect, each arm's pass rate and the arms it is shown ahead of",
        total_rows=len(rows),
    )


def _ahead_marks(arm: str, pairs: Sequence[PairReading]) -> str | None:
    """The arms ``arm`` is shown ahead of among one aspect's pairs, a lead marked as one; None where it is ahead of
    none."""
    shown: list[str] = []
    leads: list[str] = []
    for pair in pairs:
        if pair.ahead != arm:
            continue
        behind = pair.second_label if pair.ahead == pair.first else pair.first_label
        (shown if pair.confirmatory else leads).append(behind)
    marks = ([f"ahead of {_listed(shown)}"] if shown else []) + (
        [f"ahead of {_listed(leads)}, a lead"] if leads else []
    )
    return "; ".join(marks) or None


def frontier_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report], *, names: AspectNames
) -> list[ReportBlock]:
    """Where an aspect's frontier read which arm clears its bar cheapest, one line saying what it read: the arm shown
    cheaper than every other that cleared, the only arm that cleared with a cost, or the arms that cleared with none
    shown cheaper than the rest. A reading of the bundle's frontier, never a pick.

    :param campaign: The campaign, a choice.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The lines, aspect by aspect.
    :rtype: list[ReportBlock]
    """
    labels = readings.arm_labels()
    lines: list[ReportBlock] = []
    for aspect in campaign.aspects:
        for subject in reports[aspect.label].bundle.frontier.subjects:
            verdict = subject.verdict
            if verdict is None:
                continue
            named = labels.get(verdict.variant_key, verdict.model)
            tied = [labels.get(tie.variant_key, tie.model) for tie in verdict.tied_with]
            lines.append(
                DisclosureBlock(
                    section="scorecard",
                    source="comparisons",
                    text=_frontier_words(names(aspect.label), named, tied, verdict.cost_decision),
                )
            )
    return lines


def _frontier_words(aspect: str, named: str, tied: Sequence[str], decision: str | None) -> str:
    """What one aspect's frontier verdict reads, by how its cost was decided."""
    if tied:
        arms = ", ".join([named, *tied[:-1]]) + f" and {tied[-1]}"
        why = (
            "too few priced cases to show any of them cheaper"
            if decision == "untested"
            else "none is shown cheaper than the rest"
        )
        return f"Arms shown to clear the bar on {aspect}: {arms}; {why}."
    if decision == "only_cleared":
        return f"Only arm shown to clear the bar on {aspect} at a known cost: {named}."
    if decision == "shown_cheapest":
        return f"Cheapest arm shown to clear the bar on {aspect}: {named}, shown cheaper than every other that cleared."
    return f"Arm shown to clear the bar on {aspect} at the lowest observed cost: {named}."


# =============================================================================
# A measurement: the matrix of measures, every measure folded
# =============================================================================


def measure_blocks(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    *,
    names: AspectNames,
    forms: PageForms,
) -> list[ReportBlock]:
    """A measurement's scorecard: the matrix of aspects × measures for its arm, each value with its interval; every
    measure folded under it, the pass rate as its count, with a Setup column where an arm was measured under more
    than one test setup, each cell's apart.

    :param campaign: The campaign, a measurement.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :param forms: The page's forms.
    :ptype forms: PageForms
    :return: The matrix, the measures and their lines.
    :rtype: list[ReportBlock]
    :raises ValueError: An equal-weight cell is given: a measurement's matrix has no arm columns to hold it.
    """
    rigs: dict[str, set[str]] = {}
    for reading in readings.measures:
        if reading.rig is not None:
            rigs.setdefault(reading.arm, set()).add(reading.rig)
    by_rig = any(len(held) > 1 for held in rigs.values())
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="arm", header="Arm"),
        *([TableColumn(key="setup", header="Setup")] if by_rig else []),
        TableColumn(key="measure", header="Measure", term="reading"),
        TableColumn(key="value", header="Value"),
        TableColumn(key="interval", header="Interval", term="interval"),
        TableColumn(key="cases", header="Cases", term="cases"),
    ]
    rows = []
    rulers = money_rulers(readings, reports)
    rates = rate_measures(readings, reports, {})
    for reading in shown_measures(readings):
        ruler = rulers.get(reading.measure)
        rate = reading.measure in rates
        values: dict[str, CellValue | TableCell] = {
            "aspect": names.cell(reading.aspect),
            "arm": reading.arm_label,
            "measure": _measure_words(reading, reports) + (_money_label(ruler) if ruler is not None else ""),
            "value": _measured(reading, forms, rate=rate)
            if ruler is None
            else _money_cell(forms, reading.measure, reading.value, None),
            "interval": _measured_interval(forms, reading.measure, reading.interval)
            if ruler is None
            else _money_interval(forms, reading.measure, reading.interval),
            "cases": reading.cases,
        }
        if by_rig:
            values["setup"] = "every setup" if reading.rig is None else short_digest(reading.rig)
        row = table_row(values)
        # The value and its interval are figures of the row's measure, which the page's column rules record them under.
        for key in ("value", "interval"):
            row[key] = forms.of_measure(row[key], measure=reading.measure, kind="measure")
        rows.append(row)
    _no_equal_weight_row(campaign, readings, reports, "a measurement's matrix holds its one arm's measures")
    blocks: list[ReportBlock] = []
    if readings.measures:
        blocks.append(_measure_matrix(campaign, readings, reports, names, forms))
    blocks += [
        TableBlock(
            section="scorecard",
            name="measures",
            title="By aspect, measure by measure",
            columns=columns,
            rows=rows,
            order="aspect by aspect, each measure as its cells summarise it",
            total_rows=len(rows),
            details=True,
        ),
    ]
    notes = _unread_without_control_lines(campaign, reports)
    notes += _unread_until_candidate_lines(campaign, names)
    blocks += [DisclosureBlock(section="scorecard", source="comparisons", text=note) for note in notes]
    return blocks


def _measure_words(reading: MeasureReading, reports: Mapping[str, Report]) -> str:
    """A measure as the page names it on its aspect, and as every other surface names it
    (:func:`~threetears.evals.analysis.report.campaign.names.measure_name_of`): a goal check by its name on the aspect
    (declared, else what it tests), any other measure by its reader name; the pass rate as a name, capitalised as the
    others are (``Pass rate``)."""
    if reading.measure == _PASS_RATE:
        return reading.name[:1].upper() + reading.name[1:]
    return measure_name_of(reports[reading.aspect].bundle, reading.measure)


#: The key the readings give an arm's own pass rate, every grader's reading together.
_PASS_RATE = "passed"


def _measure_header(key: str, readings: CampaignReadings, reports: Mapping[str, Report]) -> str:
    """A measure's name over a column spanning aspects: its name on every aspect reading it where they agree; where
    two aspects name one check apart, each name, in the campaign's order, which the glossary defines alike."""
    held = [reading for reading in readings.measures if reading.measure == key]
    return " / ".join(dict.fromkeys(_measure_words(reading, reports) for reading in held))


def _measure_matrix(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    names: AspectNames,
    forms: PageForms,
) -> TableBlock:
    """One row per aspect (per aspect and arm, where more than one arm was measured), one column per measure: its value
    with its interval, a pass rate in percent. A measure read under several test setups on one aspect has a cell for
    each in the fold, and none here: the matrix holds one figure per cell."""
    arms = list(dict.fromkeys(reading.arm for reading in readings.measures))
    labels = {reading.arm: reading.arm_label for reading in readings.measures}
    shown = shown_measures(readings)
    measures: dict[str, MeasureReading] = {}
    for reading in shown:
        measures.setdefault(reading.measure, reading)
    rulers = money_rulers(readings, reports)
    surfaces: dict[str, DecisionSurface] = {}
    rates = rate_measures(readings, reports, surfaces)
    counts = _rate_counts(readings, reports, counted_rates(readings, reports, surfaces), surfaces)
    named = {key: _measure_header(key, readings, reports) for key in measures}
    # Two checks on different aspects can share a name (each aspect names its own): each column then says the aspects
    # it reads, so no two headers read alike.
    twice = {name for name in named.values() if list(named.values()).count(name) > 1}
    for key in measures:
        if named[key] in twice:
            on = dict.fromkeys(names(reading.aspect) for reading in readings.measures if reading.measure == key)
            named[key] = f"{named[key]}, {', '.join(on)}"
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        *([TableColumn(key="arm", header="Arm")] if len(arms) > 1 else []),
        *(
            TableColumn(
                key=f"measure:{key}",
                header=f"{named[key]} ({PERCENT})"
                if key in rates
                else named[key] + _money_label(rulers[key])
                if key in rulers
                else named[key],
            )
            for key in measures
        ),
    ]
    rows: list[dict[str, TableCell]] = []
    for aspect in campaign.aspects:
        for arm in arms:
            held = [reading for reading in shown if reading.aspect == aspect.label and reading.arm == arm]
            if not held:
                continue
            cells = {"aspect": names.cell(aspect.label)}
            if len(arms) > 1:
                cells["arm"] = TableCell(value=labels[arm])
            for key in measures:
                found = [reading for reading in held if reading.measure == key]
                cells[f"measure:{key}"] = _measure_cell(found, forms, rulers.get(key), counts=counts)
            rows.append(cells)
    if len(arms) == 1:
        return _turned_matrix(
            campaign, rows, list(measures), reports, rulers, rates, shown, forms, title=f"{labels[arms[0]]} by aspect"
        )
    return TableBlock(
        section="scorecard",
        name=_SCORECARD_TABLE,
        shared_scale=True,
        title=_MATRIX_TITLE,
        columns=columns,
        rows=rows,
        order="aspect by aspect, each measure its value with its interval",
        total_rows=len(rows),
    )


def _turned_matrix(
    campaign: EvalCampaign,
    rows: Sequence[Mapping[str, TableCell]],
    measures: Sequence[str],
    reports: Mapping[str, Report],
    rulers: Mapping[str, MoneyRuler],
    rates: frozenset[str],
    shown: Sequence[MeasureReading],
    forms: PageForms,
    *,
    title: str,
) -> TableBlock:
    """One arm's matrix turned: a row per measure read on each aspect, its value, its interval and its cases in a
    column each, with no glyph. One arm has nothing to compare a glyph against, and a column per measure, each on its
    own scale, outgrows the frame. Each row names its measure as its own aspect does (:func:`_measure_words`), so no
    name has to span two aspects, and heads its row, so a check's name links its entry as a column's header does.
    ``rows`` are the matrix's rows by aspect, each measure's cell at its column's one precision; a rate states its
    percent on the figure, and an interval's ends read in the figure's form."""
    read = {(reading.aspect, reading.measure): reading for reading in shown}
    # A measure read under several test setups on an aspect has no one figure here, nor one count; the fold has each.
    setups = Counter((reading.aspect, reading.measure) for reading in shown)
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="measure", header="Measure", term="reading"),
        TableColumn(key="value", header="Value"),
        TableColumn(key="interval", header="Interval", term="interval"),
        TableColumn(key="cases", header="Cases", term="cases"),
    ]
    turned: list[dict[str, TableCell]] = []
    for aspect, row in zip([aspect for aspect in campaign.aspects if _read_on(aspect.label, shown)], rows, strict=True):
        for key in measures:
            reading = read.get((aspect.label, key))
            if reading is None:
                continue
            cell = row[f"measure:{key}"]
            if key in rates and cell.is_number:
                cell = cell.model_copy(update={"unit": PERCENT})
            interval = (
                None
                if cell.low is None or reading.interval is None
                else forms.bracketed(key, reading.value, *reading.interval)[1].text
            )
            named = _measure_words(reading, reports) + (_money_label(rulers[key]) if key in rulers else "")
            turned.append(
                {
                    "aspect": row["aspect"],
                    "measure": TableCell(value=named),
                    # Both are figures of the row's measure, which the page's column rules record them under.
                    "value": forms.of_measure(
                        cell.model_copy(update={"low": None, "high": None}), measure=key, kind="measure"
                    ),
                    "interval": forms.of_measure(TableCell(value=interval), measure=key, kind="measure"),
                    "cases": TableCell(value=reading.cases if setups[aspect.label, key] == 1 else None),
                }
            )
    return TableBlock(
        section="scorecard",
        name=_SCORECARD_TABLE,
        title=title,
        columns=columns,
        rows=turned,
        order="aspect by aspect, each measure its value, its interval and its cases",
        total_rows=len(turned),
    )


def _read_on(aspect: str, shown: Sequence[MeasureReading]) -> bool:
    """Whether any measure the scorecard shows was read on ``aspect``: the matrix has a row for it."""
    return any(reading.aspect == aspect for reading in shown)


def _measure_cell(
    found: Sequence[MeasureReading],
    forms: PageForms,
    ruler: MoneyRuler | None = None,
    *,
    counts: Mapping[tuple[str, str, str], str] | None = None,
) -> TableCell:
    """One measure on one aspect: its value with its interval, a rate (a pass rate, an accuracy) in percent, money in
    its measure's one form on its ruler (``ruler``, per 1,000 replies as its header says), any other measure at its
    column's one precision, the set's (``2.737`` beside ``0`` reads ``2.7``, ``99.2345`` beside ``100`` reads ``99``),
    so a figure reads here as the Full results and the line under the table read it; a dash where it was not read,
    and a marked dash where it was read under several test setups, whose figures the fold holds."""
    if not found:
        return TableCell(value=None)
    if len(found) > 1:
        return TableCell(value=None, flag="split by setup")
    (reading,) = found
    if ruler is not None:
        return _money_cell(forms, reading.measure, reading.value, reading.interval)
    if reading.value is None:
        return TableCell(value=None)
    figure = forms.level(reading.measure, reading.value)
    count = (
        _rate_count(reading.passed, reading.trials, reading.cases)
        if reading.passed is not None and reading.trials is not None
        else (counts or {}).get((reading.aspect, reading.arm, reading.measure))
    )
    if reading.interval is None:
        cell = TableCell(value=figure.held, decimals=figure.decimals, flag=count)
    else:
        factor = forms.measure(reading.measure).factor
        cell = TableCell(
            value=figure.held,
            decimals=figure.decimals,
            low=factor * reading.interval[0],
            high=factor * reading.interval[1],
            flag=count,
        )
    # A figure of its measure, which the page's column rules record the cell under.
    return forms.of_measure(cell, measure=reading.measure, kind="measure")


def measured_figure(reading: MeasureReading) -> str | None:
    """One arm's pass rate as a sentence states it: the count it is computed over, ``reply passed 40 of 40 trials on
    20 cases``, or ``reply passed 18 of 20 cases`` where each case ran once; None for a reading that is no pass rate,
    or not read. The percent and its interval are the scorecard's, beside the same count, one step down: a sentence
    carries at most three figures (:data:`~threetears.evals.analysis.style.MAX_NUMERALS`).

    :param reading: The arm's reading.
    :ptype reading: MeasureReading
    :return: The figure, as a clause with no closing stop, or None.
    :rtype: str | None
    """
    if reading.value is None or reading.passed is None or reading.trials is None:
        return None
    # The one-arm verdict line's own clause, as the aspect's page and the quick print say it.
    return passed_clause(reading.arm_label, reading.passed, reading.trials, reading.cases)


def _rate_counts(
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    rates: frozenset[str],
    surfaces: dict[str, DecisionSurface] | None = None,
) -> dict[tuple[str, str, str], str]:
    """The count each rate other than the pass rate is computed over (an accuracy's matches of its trials), by aspect,
    arm and measure, as the pass rate's beside it reads (``81 of 90 trials, 30 cases``), so a percent on the first
    screen carries its count (the review's A6); read off the arm's one cell on its aspect (:func:`~threetears.evals.analysis.report.campaign.measure_kinds.measure_summary`)."""
    held = {} if surfaces is None else surfaces
    counted: dict[tuple[str, str, str], str] = {}
    for reading in readings.measures:
        if reading.measure not in rates or reading.passed is not None:
            continue
        summary = measure_summary(reading, reports, held)
        share = None if summary is None else summary.rate if summary.rate is not None else summary.mean
        if summary is None or share is None or not summary.n:
            continue
        counted[reading.aspect, reading.arm, reading.measure] = _rate_count(
            round(share * summary.n), summary.n, summary.n_independent
        )
    return counted


def _money_label(ruler: MoneyRuler) -> str:
    """What a money measure's name adds for the count its figures are drawn per, as every page spells it after the
    name: ``, per 1,000 replies`` (:meth:`MoneyRuler.words`)."""
    return f", {ruler.words('replies')}"


def _money_cell(forms: PageForms, key: str, value: float | None, interval: tuple[float, float] | None) -> TableCell:
    """A money figure on its measure's ruler, in its one form, its interval where given: drawn beside it, the form then
    holding the interval's ends too."""
    if value is None:
        return TableCell(value=None)
    if interval is None:
        figure = forms.level(key, value)
        cell = TableCell(value=figure.held, unit=figure.unit, decimals=figure.decimals)
    else:
        figure, _ = forms.bracketed(key, value, *interval)
        factor = forms.measure(key).factor
        cell = TableCell(
            value=figure.held,
            unit=figure.unit,
            decimals=figure.decimals,
            low=factor * interval[0],
            high=factor * interval[1],
        )
    # A figure of its measure, which the page's column rules record the cell under.
    return forms.of_measure(cell, measure=key, kind="measure")


def _money_interval(forms: PageForms, key: str, interval: tuple[float, float] | None) -> str | None:
    """A money figure's interval as the fold's table states it, its ends in the figure's one form on its ruler."""
    if interval is None:
        return None
    return forms.interval(key, *interval).text


def _measured(reading: MeasureReading, forms: PageForms, *, rate: bool) -> CellValue | TableCell:
    """A measure's value as the fold's table states it: a pass rate as its count, any other rate in percent, any other
    value at its measure's one precision, as the matrix above states it (``0.10`` in both, never ``0.1`` here)."""
    if reading.passed is not None and reading.trials is not None:
        return _counted(reading.passed, reading.trials, reading.cases)
    if reading.value is None:
        return None
    figure = forms.level(reading.measure, reading.value)
    return figure.text if rate else TableCell(value=figure.held, decimals=figure.decimals)


def _measured_interval(forms: PageForms, key: str, interval: tuple[float, float] | None) -> str | None:
    """A measure's interval as the fold's table states it: a rate's in percent, any other in the measure's units, each
    end as the fold states a value."""
    if interval is None:
        return None
    return forms.stated_interval(key, *interval).text


def _unread_until_candidate_lines(campaign: EvalCampaign, names: AspectNames) -> list[str]:
    """A change whose control alone has run reads as a measurement, so its margins and floors wait for a candidate;
    one line saying which, so they are not silently dropped."""
    control = campaign.control
    margined = [names(aspect.label) for aspect in campaign.aspects if aspect.margin is not None]
    if control is None or not (margined or campaign.floors):
        return []
    waiting = []
    if margined:
        waiting.append(f"the margin{'s' if len(margined) > 1 else ''} on {', '.join(margined)}")
    if campaign.floors:
        waiting.append(f"{len(campaign.floors)} floor{'s' if len(campaign.floors) > 1 else ''}")
    plural = len(waiting) > 1 or len(margined) > 1 or len(campaign.floors) > 1
    return [
        f"Only the control, {control.label}, has run, so {' and '.join(waiting)} {'are' if plural else 'is'} read "
        "once a candidate runs."
    ]


# =============================================================================
# Every kind: the bars
# =============================================================================

#: What each bar status is called on the page.
BAR_WORDS: dict[str, str] = {"met": "met", "missed": "missed", "unclear": "unclear"}


def bar_threshold(
    forms: PageForms, ruler: MeasureRuler, threshold: float, *, measure: str, kind: str = "measure"
) -> TableCell:
    """A bar's threshold as a cell, on its measure's ruler: as it was declared, never at the measure's places (rule
    4: ``92%``, never ``92.0%``), money in its ruler's form; held as a figure of its measure.

    :param forms: The page's forms.
    :ptype forms: PageForms
    :param ruler: The measure's ruler on its aspect's page.
    :ptype ruler: MeasureRuler
    :param threshold: The threshold, as stored.
    :ptype threshold: float
    :param measure: The measure's key.
    :ptype measure: str
    :param kind: Its reading (``measure``, ``judged``).
    :ptype kind: str
    :return: The cell.
    :rtype: TableCell
    """
    if is_money(ruler.unit):
        return forms.ruled_level_cell(ruler, threshold, measure=measure, kind=kind)
    scaled = threshold * ruler.factor
    cell = TableCell(value=scaled, unit=ruler.unit, value_text=forms.declared(scaled, ruler.unit).text)
    return forms.of_measure(cell, measure=measure, kind=kind)


def bar_blocks(
    readings: CampaignReadings, reports: Mapping[str, Report], *, names: AspectNames, forms: PageForms
) -> list[ReportBlock]:
    """Every arm against every absolute bar, aspect by aspect, for every kind, each measure by its reader's name;
    nothing where no bar is declared.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :param forms: The page's forms.
    :ptype forms: PageForms
    :return: The bars table, or nothing.
    :rtype: list[ReportBlock]
    """
    if not readings.bars:
        return []

    def named(aspect: str, measure: str) -> str:
        return measure_name_of(reports[aspect].bundle, measure)

    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="arm", header="Arm"),
        TableColumn(key="measure", header="Measure", term="reading"),
        TableColumn(key="bar", header="Bar", term="bar"),
        TableColumn(key="value", header="Value"),
        TableColumn(key="interval", header="Interval", term="interval"),
        TableColumn(key="status", header="Status"),
    ]
    rulers: dict[str, Rulers] = {}

    def kind(status: BarStatus) -> str:
        # The reading its aspect's page reads the bar on: a judged dimension's bar is ``judged``, its scale its own.
        readings = {
            verdict.name: verdict.reading
            for verdict in reports[status.aspect].verdicts
            if verdict.kind == "bar" and verdict.variant_key == status.arm
        }
        return readings.get(status.measure, "measure")

    def ruler(status: BarStatus) -> MeasureRuler:
        # On the ruler its aspect's page states the measure on, under the measure's own reading, at the places its set
        # reads at there (rules 1 and 2): a rate in percent, money in its form, never a raw share.
        if status.aspect not in rulers:
            bundle = reports[status.aspect].bundle
            rulers[status.aspect] = report_rulers(bundle_decision_surface(bundle), bundle.measure_catalog)
        return rulers[status.aspect].of(status.measure, kind(status))

    rows = [
        table_row(
            {
                "aspect": names.cell(status.aspect),
                "arm": status.arm_label,
                "measure": named(status.aspect, status.measure),
                # Each figure of the bar's measure, which the page records the cell under.
                "bar": forms.of_measure(
                    TableCell(
                        value=f"{'at most' if status.direction == 'lower_is_better' else 'at least'} "
                        f"{bar_threshold(forms, ruler(status), status.bar, measure=status.measure, kind=kind(status)).figure}"
                    ),
                    measure=status.measure,
                    kind=kind(status),
                ),
                "value": forms.ruled_level_cell(ruler(status), status.value, measure=status.measure, kind=kind(status)),
                "interval": None
                if status.interval is None
                else forms.of_measure(
                    TableCell(
                        value=forms.ruled_interval(
                            ruler(status), *status.interval, measure=status.measure, kind=kind(status)
                        ).text
                    ),
                    measure=status.measure,
                    kind=kind(status),
                ),
                "status": BAR_WORDS[status.status],
            }
        )
        for status in readings.bars
    ]
    return [
        TableBlock(
            section="scorecard",
            name="bars",
            title="Bars",
            columns=columns,
            rows=rows,
            order="aspect by aspect, each bar, each arm",
            total_rows=len(rows),
        )
    ]


def _unread_without_control_lines(campaign: EvalCampaign, reports: Mapping[str, Report]) -> list[str]:
    """What an aspect stored that needs a control and was not read, as its declaration says it, once per aspect."""
    lines: list[str] = []
    for aspect in campaign.aspects:
        declared = reports[aspect.label].bundle.declared
        if declared is not None and declared.unread_without_control:
            lines.append(declared.unread_without_control)
    return lines
