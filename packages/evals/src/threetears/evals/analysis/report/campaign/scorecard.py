"""What code read on each aspect, as one matrix per campaign kind, with every kind's bars, in the page's Scorecard.

The matrix's form follows the campaign's kind (:func:`~threetears.evals.analysis.campaign_readings.campaign_kind`):

- **A change**: one row per aspect, one column per candidate. Each cell is the candidate's change against the control
  in points, its interval drawn against the aspect's margin, and what it shows (shown better, shown worse, no worse
  than the margin, or unclear), with a floor's status beside it where the floor reads otherwise. The control's column
  holds its pass rate, marked where it cannot move (a rate at or below the margin cannot show a drop; one at 100%
  cannot show a gain). Under it, folded, the same readings one row per aspect × candidate with their counts; then the
  lines that change how a cell reads; then one table of what would settle each unclear reading, on the pass rate and
  on every rate a floor reads.
- **A choice**: one row per aspect, one column per arm, each cell the arm's pass rate and the arms it is shown ahead
  of; folded under it, every pair; then the frontier's reading of which arm clears a bar cheapest.
- **A measurement**: one row per aspect, one column per measure for its one arm; folded under it, every measure.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce

from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, FamilyComparison
from threetears.evals.analysis.campaign_readings import AspectReading, CampaignReadings, MeasureReading, PairReading
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.quantities import PERCENT, POINTS
from threetears.evals.analysis.report.campaign import aggregate
from threetears.evals.analysis.report.campaign.common import floor_of, points
from threetears.evals.analysis.report.campaign.names import AspectNames, measure_name_of
from threetears.evals.analysis.report.decision import (
    CasesNeeded,
    RateVerdict,
    SampleProjection,
    aspect_pass_rate,
    case_cost_usd,
    floor_readings,
    projected_cell,
    rate_projections,
    read_floor,
)
from threetears.evals.analysis.report.model import (
    CellValue,
    DisclosureBlock,
    Report,
    ReportBlock,
    TableBlock,
    TableCell,
    TableColumn,
    interval_text,
    table_row,
)
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.analysis.style import humanize
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import RateDrop

__all__ = [
    "BAR_WORDS",
    "CHOICE_NOTE",
    "READING_WORDS",
    "aspect_shows",
    "bar_blocks",
    "campaign_verdict",
    "frontier_blocks",
    "measure_blocks",
    "pair_blocks",
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
    "not_separated": "not separated",
}

#: The contrast verdict that says what a row's reading says, for the two readings a contrast can also reach.
_CONTRAST_OF: dict[str, str] = {"better": "improved", "worse": "regressed"}


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


def _reading_words(row: AspectReading) -> str:
    """What a row shows, a lead marked as one."""
    words = READING_WORDS[row.reading]
    return words if row.confirmatory else f"{words}, a lead"


def _counted(passed: int, trials: int, cases: int) -> str:
    """An arm's passes as a table states them: of its trials, which are named as trials where they are not its cases,
    so a count over repeats never reads as one over cases."""
    return f"{passed} of {trials}" if trials == cases else f"{passed} of {trials} trials"


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
) -> list[dict[str, TableCell]]:
    """The row across every aspect, each aspect weighed alike, with each arm's cell
    (:func:`~threetears.evals.analysis.report.campaign.aggregate.equal_weight_cells`) in that arm's column; no row
    where no cell is given. ``arms`` is each arm's column and the unit it holds, by variant key. A figure states its
    unit, which must be its column's; it is placed without it, since the column's header names it. A withheld figure
    (no value) carries none.

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
    for key, cell in across.items():
        column, unit = arms[key]
        if cell.value is not None and (not cell.is_number or cell.unit != unit):
            stated = repr(cell.unit) if cell.is_number else f"the words {cell.value!r}"
            raise ValueError(
                f"the {aggregate.EQUAL_WEIGHT_ROW!r} cell for arm {key} states {stated}, and its column holds {unit!r}"
            )
        row[column] = cell.model_copy(update={"unit": ""})
    return [row]


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
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report], *, names: AspectNames
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
                text="No control was named, so no aspect is compared against one.",
            )
        ]
    blocks: list[ReportBlock] = []
    if readings.readings:
        blocks += [_change_matrix(campaign, readings, reports, names, readings.control), _change_rows(readings, names)]
    else:
        _no_equal_weight_row(campaign, readings, reports, "no candidate was compared on any aspect")
    levels = [
        level
        for row in readings.readings
        if row.reading != "within_margin" and (level := _shown_interval(row)[1]) is not None
    ]
    notes = _level_lines(levels, readings) if readings.readings else []
    notes += _within_lines(readings, names)
    notes += [
        _rig_line(names(row.aspect), f"{row.arm_label}'s change against {readings.control_label}", row.rig)
        for row in readings.readings
        if row.rig is not None
    ]
    notes += _reconciled_lines(readings, reports, names)
    notes += _unread_lines(campaign, readings, names)
    blocks += [DisclosureBlock(section="scorecard", source="comparisons", text=note) for note in notes if note]
    return blocks


def _change_matrix(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    names: AspectNames,
    control: str,
) -> TableBlock:
    """One row per aspect any candidate was compared on, one column per candidate, the control's pass rate beside the
    aspect; then the row across every aspect, where one is given."""
    labels = readings.arm_labels()
    candidates = list(dict.fromkeys(row.arm for row in readings.readings))
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="cases", header="Cases", term="cases"),
        TableColumn(key=_arm_column(control), header=f"▸ {readings.control_label} (%)", term="control"),
        *(TableColumn(key=_arm_column(arm), header=f"{labels[arm]} (pts)", term="change") for arm in candidates),
    ]
    rows: list[dict[str, TableCell]] = []
    for aspect in campaign.aspects:
        on = {row.arm: row for row in readings.on_aspect(aspect.label)}
        if not on:
            continue
        cells = {
            "aspect": TableCell(value=names(aspect.label)),
            "cases": TableCell(value=_cases([min(row.control_cases, row.arm_cases) for row in on.values()])),
            _arm_column(control): _control_cell(list(on.values()), reports[aspect.label].bundle, control),
        }
        for arm in candidates:
            row = on.get(arm)
            cells[_arm_column(arm)] = TableCell(value=None) if row is None else _change_cell(row, campaign, readings)
        rows.append(cells)
    arms = {control: (_arm_column(control), PERCENT), **{key: (_arm_column(key), POINTS) for key in candidates}}
    rows += _with_equal_weight_row(campaign, readings, reports, columns, arms)
    return TableBlock(
        section="scorecard",
        name=_SCORECARD_TABLE,
        title=_MATRIX_TITLE,
        columns=columns,
        rows=rows,
        order="aspect by aspect, each candidate's change against the control",
        total_rows=len(rows),
    )


def _control_cell(rows: Sequence[AspectReading], bundle: AnalysisContextBundle, control: str) -> TableCell:
    """The control's pass rate on an aspect (over its trials, as its count reads), marked where it cannot move.

    A drop is read on per-case shares, so whether one past the margin can show is decided on them too: the largest
    drop a row's values allow is the mean of the control's shares it reads, and where that is at or below the margin
    for every row read on shares, no candidate's drop can pass it. Shares and trials differ where cases ran an uneven
    number of usable trials, so the trial rate alone would mark a control that can still show a drop. At 100% of its
    trials every share is one, and no gain can show.
    """
    row = rows[0]
    if not row.control_trials:
        return TableCell(value=None)
    rate = 100 * row.control_passed / row.control_trials
    marks = []
    if row.margin is not None:
        rates = [aspect_pass_rate(bundle, control, read.arm) for read in rows]
        shares = [
            100 * sum(read.samples[0]) / len(read.samples[0]) for read in rates if read is not None and read.samples
        ]
        if shares and all(share <= row.margin for share in shares):
            marks.append("cannot show a drop")
    if row.control_passed == row.control_trials:
        marks.append("cannot show a gain")
    return TableCell(value=round(rate, 1), flag="; ".join(marks) or None)


def _change_cell(row: AspectReading, campaign: EvalCampaign, readings: CampaignReadings) -> TableCell:
    """A candidate's change on one aspect: the points, the interval its reading was read on drawn against the margin
    where two or more cases bound it, the reading, and its floors' status where they read otherwise."""
    change = round(row.change, 1)
    marks = [_reading_words(row)]
    if (status := _floor_state(row, campaign, readings)) is not None and status != _IMPLIED_FLOOR[row.reading]:
        marks.append(f"floor {status}")
    interval, _ = _shown_interval(row)
    if interval is None or min(row.control_cases, row.arm_cases) < 2:
        return TableCell(value=change, flag="; ".join(marks))
    return TableCell(
        value=change, low=interval[0], high=interval[1], margin=row.margin, zero_line=True, flag="; ".join(marks)
    )


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


def _change_rows(readings: CampaignReadings, names: AspectNames) -> TableBlock:
    """The matrix's readings one row per aspect × candidate, folded: the cases, each arm's passes, the change and the
    interval it was read on, the margin and the reading."""
    control = readings.control_label
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="arm", header="Arm"),
        TableColumn(key="cases", header="Cases", term="cases"),
        TableColumn(key="control", header=f"▸ {control}", term="control"),
        TableColumn(key="passed", header="Arm passed", term="passed"),
        TableColumn(key="change", header="Change (pts)", term="change"),
        TableColumn(key="interval", header="Interval (pts)", term="interval"),
        TableColumn(key="margin", header="Margin (pts)", term="margin"),
        TableColumn(key="reading", header="Reading"),
    ]
    rows = []
    for row in readings.readings:
        interval, _ = _shown_interval(row)
        rows.append(
            table_row(
                {
                    "aspect": names(row.aspect),
                    "arm": row.arm_label,
                    "cases": min(row.control_cases, row.arm_cases),
                    "control": _counted(row.control_passed, row.control_trials, row.control_cases),
                    "passed": _counted(row.arm_passed, row.arm_trials, row.arm_cases),
                    "change": round(row.change, 1),
                    "interval": None if interval is None else interval_text(interval[0], interval[1], POINTS),
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


def _percent(level: float) -> str:
    """A level as a percent, to the precision that tells two levels apart."""
    return format_number(round(100 * level, 2)) + "%"


def _level_lines(levels: Sequence[float], readings: CampaignReadings) -> list[str]:
    """The level the corrected intervals were read at (``levels``: every reading's but one no worse than the margin,
    whose line is :func:`_within_lines`; none where every reading is), how often an arm is shown better or worse
    wrongly, and, with several arms, that the correction does not run across them. Each count is the page's own, and
    counts what is corrected together: an arm's gains are its rows with an interval, its drops those with an interval
    on an aspect with a margin (``drop_interval``), as its two families are read."""
    families = Counter(row.arm for row in readings.readings)
    gains = Counter(row.arm for row in readings.readings if row.gain_interval is not None)
    drops = Counter(row.arm for row in readings.readings if row.drop_interval is not None)
    sizes = {(gains[arm], drops[arm]) for arm in families}
    lines: list[str] = []
    if levels:
        low, high = min(levels), max(levels)
        if round(high, 4) <= round(INTERVAL_LEVEL, 4):
            # A family of one is read at the nominal level: no reading is corrected.
            lines.append(
                f"{'The interval is' if len(levels) == 1 else 'Each interval is'} read at {_percent(low)}, as a "
                "floor's is."
            )
        else:
            if len(sizes) > 1:
                together = "each arm's readings"
            else:
                ((gained, dropped),) = sizes
                together = (
                    f"each arm's {gained} readings"
                    if gained == dropped
                    else f"each arm's gains on {gained} aspects, and its drops on {dropped}, are each"
                )
            lines.append(
                f"Intervals here are wider than a plain {_percent(INTERVAL_LEVEL)} interval, because {together}"
                f"{'' if together.endswith(' each') else ' are'} corrected together."
            )
            read_at = _percent(low) if round(low, 4) == round(high, 4) else f"{_percent(low)} to {_percent(high)}"
            lines.append(f"Each is read at {read_at}, and the reading beside it always agrees with it.")
    half = readings.alpha / 2
    where = " on some aspect" if any(max(size) > 1 for size in sizes) else ""
    lines.append(
        f"For any one arm, the chance of being wrongly shown better{where} is at most {_percent(half)}, and likewise "
        "for wrongly shown worse."
    )
    if len(families) > 1:
        lines.append(
            f"That correction runs within each arm, not across the {len(families)} arms, so the chance that some arm "
            f"is wrongly shown better is up to {_percent(min(1.0, len(families) * half))}, and likewise for worse."
        )
        lines.append("Correcting across arms would lower it, at the price of wider intervals.")
    return lines


def _within_lines(readings: CampaignReadings, names: AspectNames) -> list[str]:
    """Where a row reads no worse than the margin, the level it was read at and why that needs no correction; where
    such a row's interval sits above zero, that it is still no gain shown, naming each such row."""
    within = [row for row in readings.readings if row.reading == "within_margin"]
    if not within:
        return []
    lines = [
        f"A row no worse than the margin reads a plain {_percent(INTERVAL_LEVEL)} interval, as a floor does.",
        "An arm is no worse only where it holds on every asked aspect at once, so more aspects never make that "
        "easier, and it needs no correction.",
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


def _reconciled_lines(readings: CampaignReadings, reports: Mapping[str, Report], names: AspectNames) -> list[str]:
    """Where a contrast in an aspect's fold reads the very per-case values a row reads and reaches another verdict, one
    line naming both and the interval each was read on, so the disagreement is explained where it is seen.

    A row's gain is read on a bounded interval corrected across its arm's aspects; a contrast on a t interval (or a
    bounded one where its values have no spread) corrected within its aspect's family. A contrast reading a drop past
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
        for comparison, family_level in _contrasts(bundle, control, row.arm):
            if comparison.verdict not in _CONTRAST_WORDS or comparison.case_values is None:
                continue
            if not _same_values(comparison.case_values, rate.samples) or (comparison.test == "paired") != rate.paired:
                continue
            said = comparison.verdict if comparison.verdict in ("improved", "regressed") else None
            if said == _CONTRAST_OF.get(row.reading):
                continue
            lines.append(_reconciled(row, comparison, bundle, level, family_level, names(row.aspect)))
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
    family_level: float,
    aspect: str,
) -> str:
    """One row and the contrast reading its values to another verdict: both verdicts, then how each was read."""
    if comparison.reading == "trial":
        reading = "the pass rate"
    elif comparison.reading == "judged":
        reading = humanize("dimension", comparison.name)
    else:
        reading = measure_name_of(bundle, comparison.name)
    own = "on its own" if round(level, 4) <= round(INTERVAL_LEVEL, 4) else f"corrected across {row.arm_label}'s aspects"
    basis = "t" if comparison.basis == "t" else "bounded"
    said = (
        f"On {aspect}, {row.arm_label}'s row reads {READING_WORDS[row.reading]} and its contrast on {reading} reads "
        f"{_CONTRAST_WORDS[comparison.verdict]}, over the same per-case values. The row reads a bounded interval at "
        f"{_percent(level)}, {own}; the contrast a {basis} interval at {_percent(family_level)}, corrected within the "
        "aspect's family."
    )
    if comparison.verdict == "regressed" and row.reading != "worse" and row.margin is not None:
        said += f" The row counts a drop only past the {format_number(row.margin)}-point margin."
    return said


def _unread_lines(campaign: EvalCampaign, readings: CampaignReadings, names: AspectNames) -> list[str]:
    """An aspect no arm was compared on, and an aspect with no margin, which is never shown worse."""
    lines: list[str] = []
    for aspect in campaign.aspects:
        rows = readings.on_aspect(aspect.label)
        if not rows:
            named = names(aspect.label)
            lines.append(f"No arm was compared on {named}: the control or every other arm did not run there.")
        elif aspect.margin is None:
            lines.append(f"No margin was declared on {names(aspect.label)}, so no arm is ever shown worse there.")
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
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report], *, names: AspectNames
) -> list[ReportBlock]:
    """One table of what would settle each unclear reading of a change, aspect by aspect and arm by arm: a row left
    unclear against its margin, read on the pass rate alone; and each rate floor left unclear, read on every rate its
    clause names, joining each rate's count side by side as the aspect's own floor table does. Each count is the total
    independent cases, not cases to add, at which the interval would show it, if the arms do not differ and if they
    differ as observed, read at the nominal level over the same cases and values. So the two counts on one arm and
    aspect differ only where the floor reads other rates than the pass rate alone (every rate, or a check it names) or
    reads them against other points than the margin. The floor's row names the rate deciding its as-observed count;
    its no-difference count may be another rate's. A row on an aspect with no margin has nothing to show it within, and a settled reading
    nothing left to show, so neither is a row here.

    :param campaign: The campaign, a change.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label, whose evidence a floor's count is read from.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The table and its note; nothing where no reading is to settle.
    :rtype: list[ReportBlock]
    """
    sizings = [sizing for row in readings.readings for sizing in _sizings(row, campaign, readings, reports)]
    if not sizings:
        return []
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="arm", header="Arm"),
        TableColumn(key="shows", header="To show"),
        TableColumn(key="decided_by", header="Decided by", term="reading"),
        TableColumn(key="why", header="Why unclear"),
        TableColumn(key="none", header="If no difference"),
        TableColumn(key="observed", header="If as observed"),
    ]
    rows = [
        table_row(
            {
                "aspect": names(sizing.aspect),
                "arm": sizing.arm,
                "shows": sizing.shows,
                "decided_by": sizing.decided_by,
                "why": sizing.why or None,
                "none": None
                if sizing.projection is None
                else projected_cell(sizing.projection, sizing.projection.if_none),
                "observed": None
                if sizing.projection is None
                else projected_cell(sizing.projection, sizing.projection.if_observed),
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
        DisclosureBlock(section="scorecard", source="comparisons", text=_settle_note(sizings)),
    ]


def _sizings(
    row: AspectReading, campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> list[_Sizing]:
    """What would settle a row left unclear against its margin, then each rate floor on its aspect left unclear for
    its arm, each counted as the aspect's own page counts it."""
    found: list[_Sizing] = []
    if row.reading == "unclear" and row.margin is not None:
        found.append(
            _Sizing(
                aspect=row.aspect,
                arm=row.arm_label,
                shows=f"no worse than the {format_number(row.margin)}-point margin",
                decided_by="pass rate",
                why=row.reason,
                projection=row.projection,
            )
        )
    control = readings.control
    if control is None:
        return found
    bundle = reports[row.aspect].bundle
    figures = floor_readings(bundle, control, row.arm)
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
        found.append(
            _Sizing(
                aspect=row.aspect,
                arm=row.arm_label,
                shows=f"the {format_number(clause.max_points)}-point floor holds",
                decided_by=None if deciding is None else _rate_words(deciding),
                why=verdict.reason if deciding is None else deciding.reason,
                projection=None if rates is None else reduce(SampleProjection.joined, (one for _, one in rates)),
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
    """A rate a floor reads, as the sizing table names it: ``pass rate``, or the check's or dimension's own name."""
    return "pass rate" if rate.rate.measure is None else rate.rate.name


def _settle_note(sizings: Sequence[_Sizing]) -> str:
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
        f"Each count is the cases a {_percent(INTERVAL_LEVEL)} interval needs in all, not cases to add, to rule out a "
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
    ahead of beta``, ``alpha misses a bar``.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param aspect: The aspect's label.
    :ptype aspect: str
    :return: The words.
    :rtype: str
    """
    if readings.kind == "choice":
        pairs = [pair for pair in readings.pairs if pair.aspect == aspect]
        if not pairs:
            return "not compared"
        return ", ".join(_pair_words(pair) for pair in pairs)
    if readings.kind == "measurement":
        missed = [status.arm_label for status in readings.missed_bars() if status.aspect == aspect]
        return f"{', '.join(dict.fromkeys(missed))} misses a bar" if missed else "measured"
    rows = readings.on_aspect(aspect)
    if not rows:
        return "not compared"
    return ", ".join(f"{row.arm_label} {_reading_words(row)}" for row in rows)


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


def _pair_words(pair: PairReading) -> str:
    """A pair as a fold's heading says it: ``alpha ahead of beta``, ``alpha and beta not separated``."""
    if pair.ahead is None:
        said = f"{pair.first_label} and {pair.second_label} not separated"
    else:
        ahead, behind = (
            (pair.first_label, pair.second_label) if pair.ahead == pair.first else (pair.second_label, pair.first_label)
        )
        said = f"{ahead} ahead of {behind}"
    return said if pair.confirmatory else f"{said}, a lead"


# =============================================================================
# A choice: the matrix of arms, every pair folded, and the frontier
# =============================================================================

#: What a choice's notes say once: the one family, and that ahead is a reading.
CHOICE_NOTE = (
    "A pair is read both ways at once, so every pair on every aspect is one family, corrected together. An arm is "
    "ahead only where the whole interval is one side of zero, and ahead is a reading, not a pick."
)


def pair_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report], *, names: AspectNames
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
        change = round(pair.change, 1)
        ahead = pair.ahead_label or "no separation"
        cells = table_row(
            {
                "aspect": names(pair.aspect),
                "first": pair.first_label,
                "second": pair.second_label,
                "cases": min(pair.first_cases, pair.second_cases),
                "first_passed": _counted(pair.first_passed, pair.first_trials, pair.first_cases),
                "second_passed": _counted(pair.second_passed, pair.second_trials, pair.second_cases),
                "change": change,
                "interval": None if pair.interval is None else interval_text(*pair.interval, POINTS),
                "ahead": ahead if pair.confirmatory else f"{ahead}, a lead",
            }
        )
        if pair.interval is not None:
            cells["change"] = TableCell(value=change, low=pair.interval[0], high=pair.interval[1], zero_line=True)
        rows.append(cells)
    blocks: list[ReportBlock] = []
    if readings.pairs:
        blocks.append(_choice_matrix(campaign, readings, reports, names))
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
    if levels:
        low, high = min(levels), max(levels)
        notes.append(
            f"Each interval is read at {_percent(low)}."
            if round(low, 4) == round(high, 4)
            else f"Each interval is read at the level its pair was tested at, from {_percent(low)} to {_percent(high)}."
        )
    notes += [
        _rig_line(names(pair.aspect), f"the difference between {pair.first_label} and {pair.second_label}", pair.rig)
        for pair in readings.pairs
        if pair.rig is not None
    ]
    notes += _unread_without_control_lines(campaign, reports)
    blocks += [DisclosureBlock(section="scorecard", source="comparisons", text=note) for note in notes if note]
    return blocks


def _choice_matrix(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report], names: AspectNames
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
            "aspect": TableCell(value=names(aspect.label)),
            "cases": TableCell(value=_cases([boards[arm].usable_cases for arm in arms if arm in boards])),
        }
        for arm in arms:
            board = boards.get(arm)
            rate = None if board is None or not board.usable else round(100 * board.passed / board.usable, 1)
            cells[_arm_column(arm)] = TableCell(value=rate, flag=_ahead_marks(arm, pairs))
        rows.append(cells)
    rows += _with_equal_weight_row(
        campaign, readings, reports, columns, {arm: (_arm_column(arm), PERCENT) for arm in arms}
    )
    return TableBlock(
        section="scorecard",
        name=_SCORECARD_TABLE,
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
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report], *, names: AspectNames
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
    for reading in readings.measures:
        values: dict[str, CellValue] = {
            "aspect": names(reading.aspect),
            "arm": reading.arm_label,
            "measure": reading.name,
            "value": _measured(reading.value, reading.passed, reading.trials, reading.cases),
            "interval": _measured_interval(reading.interval, reading.passed is not None),
            "cases": reading.cases,
        }
        if by_rig:
            values["setup"] = "every setup" if reading.rig is None else short_digest(reading.rig)
        rows.append(table_row(values))
    _no_equal_weight_row(campaign, readings, reports, "a measurement's matrix holds its one arm's measures")
    blocks: list[ReportBlock] = []
    if readings.measures:
        blocks.append(_measure_matrix(campaign, readings, names))
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
        DisclosureBlock(
            section="scorecard",
            source="comparisons",
            text="One arm is measured, so nothing is compared and no arm is recommended; its bars are its standard.",
        ),
    ]
    notes = _unread_without_control_lines(campaign, reports)
    notes += _unread_until_candidate_lines(campaign, names)
    blocks += [DisclosureBlock(section="scorecard", source="comparisons", text=note) for note in notes]
    return blocks


def _measure_matrix(campaign: EvalCampaign, readings: CampaignReadings, names: AspectNames) -> TableBlock:
    """One row per aspect (per aspect and arm, where more than one arm was measured), one column per measure: its value
    with its interval, a pass rate in percent. A measure read under several test setups on one aspect has a cell for
    each in the fold, and none here: the matrix holds one figure per cell."""
    arms = list(dict.fromkeys(reading.arm for reading in readings.measures))
    labels = {reading.arm: reading.arm_label for reading in readings.measures}
    measures: dict[str, MeasureReading] = {}
    for reading in readings.measures:
        measures.setdefault(reading.measure, reading)
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        *([TableColumn(key="arm", header="Arm")] if len(arms) > 1 else []),
        *(
            TableColumn(
                key=f"measure:{key}", header=f"{first.name} ({PERCENT})" if first.passed is not None else first.name
            )
            for key, first in measures.items()
        ),
    ]
    rows: list[dict[str, TableCell]] = []
    for aspect in campaign.aspects:
        for arm in arms:
            held = [reading for reading in readings.measures if reading.aspect == aspect.label and reading.arm == arm]
            if not held:
                continue
            cells = {"aspect": TableCell(value=names(aspect.label))}
            if len(arms) > 1:
                cells["arm"] = TableCell(value=labels[arm])
            for key in measures:
                found = [reading for reading in held if reading.measure == key]
                cells[f"measure:{key}"] = _measure_cell(found)
            rows.append(cells)
    return TableBlock(
        section="scorecard",
        name=_SCORECARD_TABLE,
        title=_MATRIX_TITLE if len(arms) > 1 else f"{labels[arms[0]]} by aspect",
        columns=columns,
        rows=rows,
        order="aspect by aspect, each measure its value with its interval",
        total_rows=len(rows),
    )


def _measure_cell(found: Sequence[MeasureReading]) -> TableCell:
    """One measure on one aspect: its value with its interval, a pass rate in percent; a dash where it was not read,
    and a marked dash where it was read under several test setups, whose figures the fold holds."""
    if not found:
        return TableCell(value=None)
    if len(found) > 1:
        return TableCell(value=None, flag="split by setup")
    (reading,) = found
    scale = 100.0 if reading.passed is not None else 1.0
    places = 1 if reading.passed is not None else 4
    if reading.value is None:
        return TableCell(value=None)
    value = round(scale * reading.value, places)
    if reading.interval is None:
        return TableCell(value=value)
    return TableCell(value=value, low=scale * reading.interval[0], high=scale * reading.interval[1])


def _measured(value: float | None, passed: int | None, trials: int | None, cases: int) -> CellValue:
    """A measure's value as the table states it: a pass rate as its count, any other value as measured."""
    if passed is not None and trials is not None:
        return _counted(passed, trials, cases)
    return None if value is None else round(value, 4)


def _measured_interval(interval: tuple[float, float] | None, rate: bool) -> str | None:
    """A measure's interval as the table states it: a pass rate's in percent, any other in the measure's units."""
    if interval is None:
        return None
    if rate:
        return f"[{points(100 * interval[0])}{PERCENT}, {points(100 * interval[1])}{PERCENT}]"
    return f"[{format_number(round(interval[0], 4))}, {format_number(round(interval[1], 4))}]"


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


def bar_blocks(readings: CampaignReadings, reports: Mapping[str, Report], *, names: AspectNames) -> list[ReportBlock]:
    """Every arm against every absolute bar, aspect by aspect, for every kind, each measure by its reader's name;
    nothing where no bar is declared.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
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
    rows = [
        table_row(
            {
                "aspect": names(status.aspect),
                "arm": status.arm_label,
                "measure": named(status.aspect, status.measure),
                "bar": f"{'at most' if status.direction == 'lower_is_better' else 'at least'} {format_number(status.bar)}",
                "value": status.value,
                "interval": None
                if status.interval is None
                else f"[{format_number(status.interval[0])}, {format_number(status.interval[1])}]",
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
