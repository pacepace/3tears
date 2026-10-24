"""What code read on each aspect, laid out by the campaign's kind, and every kind's bars.

For **a change**:

1. **Code's table**, one row per aspect × candidate: the cases, each arm's pass count, the change in points with the
   interval it was read on, the aspect's margin, and what the row shows (shown better, shown worse, no worse than the
   margin, or unclear). Under it, the lines that change how a row reads: the level each interval was read at, an
   aspect with no margin (never shown worse), the test setup a row was read under where the arms ran under several.
2. **What would settle each unclear row** on an aspect with a margin: the cases that would show the arm within it,
   if the arms do not differ and if they differ as observed.

For **a choice**: the table of every pair of arms on every aspect and the arm shown ahead, then the frontier's reading
of which arm clears a bar cheapest. For **a measurement**: each aspect's measures for its one arm. Every kind then
shows its bars table. Beside each aspect's fold, a few words say what code read there (:func:`aspect_shows`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.campaign_readings import AspectReading, CampaignReadings, PairReading
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.quantities import PERCENT, POINTS
from threetears.evals.analysis.report.campaign.aggregate import EQUAL_WEIGHT_ROW, equal_weight_cells
from threetears.evals.analysis.report.campaign.common import floor_of, points, rate_name
from threetears.evals.analysis.report.campaign.names import AspectNames, measure_name_of
from threetears.evals.analysis.report.decision import projected_cell
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


def table_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report], *, names: AspectNames
) -> list[ReportBlock]:
    """Code's table, one row per aspect × candidate, then one row per arm across every aspect where
    :func:`~threetears.evals.analysis.report.campaign.aggregate.equal_weight_cells` gives it one, and the lines under
    it that change how a row reads.

    :param campaign: The campaign, a change.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The table and its lines; one line where no control was named.
    :rtype: list[ReportBlock]
    """
    if readings.control is None:
        return [
            DisclosureBlock(
                section="scoreboard",
                source="comparisons",
                text="No control was named, so no aspect is compared against one.",
            )
        ]
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
    rows: list[dict[str, TableCell]] = []
    levels: list[float] = []
    for row in readings.readings:
        interval, level = _shown_interval(row)
        if level is not None and row.reading != "within_margin":
            levels.append(level)
        change = round(row.change, 1)
        values: dict[str, CellValue] = {
            "aspect": names(row.aspect),
            "arm": row.arm_label,
            "cases": min(row.control_cases, row.arm_cases),
            "control": f"{row.control_passed} of {row.control_trials}",
            "passed": f"{row.arm_passed} of {row.arm_trials}",
            "change": change,
            "interval": None if interval is None else interval_text(interval[0], interval[1], POINTS),
            "margin": row.margin,
            "reading": _reading_words(row),
        }
        cells = table_row(values)
        if interval is not None and min(row.control_cases, row.arm_cases) >= 2:
            cells["change"] = TableCell(
                value=change,
                low=interval[0],
                high=interval[1],
                margin=row.margin,
                zero_line=True,
            )
        rows.append(cells)
    labels = readings.arm_labels()
    for arm, cell in equal_weight_cells(campaign, readings, reports).items():
        across = {column.key: TableCell(value=None) for column in columns}
        rows.append(
            across | {"aspect": TableCell(value=EQUAL_WEIGHT_ROW), "arm": TableCell(value=labels[arm]), "change": cell}
        )
    blocks: list[ReportBlock] = [
        TableBlock(
            section="scoreboard",
            name="aspects",
            title="By aspect",
            columns=columns,
            rows=rows,
            order="aspect by aspect, each arm against the control",
            total_rows=len(rows),
        )
    ]
    notes = _level_lines(levels, readings) if readings.readings else []
    notes += _within_lines(readings)
    notes += [
        _rig_line(names(row.aspect), f"{row.arm_label}'s change against {control}", row.rig)
        for row in readings.readings
        if row.rig is not None
    ]
    notes += _floor_beside_lines(campaign, readings, names)
    notes += _unread_lines(campaign, readings, names)
    notes += _no_question_lines(campaign)
    blocks += [DisclosureBlock(section="scoreboard", source="comparisons", text=note) for note in notes if note]
    return blocks


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
    """The level the table's corrected intervals were read at (``levels``: every row's but one no worse than the
    margin, whose line is :func:`_within_lines`; none where every row is), how often a reading can be wrong, and,
    with several arms, that nothing corrects across them."""
    lines: list[str] = []
    if levels:
        low, high = min(levels), max(levels)
        if len(levels) == 1 and round(low, 4) == round(INTERVAL_LEVEL, 4):
            lines.append(f"The interval is read at {_percent(low)}, as a floor's is.")
        elif len(levels) == 1:
            lines.append(f"The interval is read at {_percent(low)}, corrected across each arm's aspects.")
        elif round(low, 4) == round(high, 4):
            lines.append(f"Each interval is read at {_percent(low)}, corrected across each arm's aspects.")
        else:
            lines.append(
                f"Each interval is read at the level its row was tested at, from {_percent(low)} to {_percent(high)}, "
                "corrected across each arm's aspects; a reading never contradicts its interval."
            )
    half = _percent(readings.alpha / 2)
    lines.append(f"Each arm's gains and its drops are each shown wrongly at most {half} of the time.")
    if len({row.arm for row in readings.readings}) > 1:
        lines.append(
            "Nothing corrects across arms, so with several arms one is shown better or worse by chance more often."
        )
    return lines


def _within_lines(readings: CampaignReadings) -> list[str]:
    """Where a row reads no worse than the margin, the level it was read at and why that is not corrected; where
    such a row's interval clears zero, that its gain is still read corrected, so that is not a gain shown."""
    within = [row for row in readings.readings if row.reading == "within_margin"]
    if not within:
        return []
    lines = [
        f"A row no worse than the margin reads its interval at {_percent(INTERVAL_LEVEL)} on its own, as a floor "
        "does: a candidate is no worse only where it holds on every asked aspect, which needs no correction."
    ]
    if any(row.within_interval is not None and row.within_interval[0] > 0 for row in within):
        lines.append(
            "Its gain is still read corrected across the arm's aspects, so such an interval clearing zero is not a "
            "gain shown."
        )
    return lines


def _floor_beside_lines(campaign: EvalCampaign, readings: CampaignReadings, names: AspectNames) -> list[str]:
    """A broken floor beside a row that is not shown worse, and a held floor beside a row left unclear: the two read
    different intervals, or different points, and say which."""
    lines: list[str] = []
    for status in readings.broken_floors():
        row = next((row for row in readings.on_aspect(status.aspect) if row.arm == status.arm), None)
        if row is None or row.reading == "worse":
            continue
        lines.append(
            f"The floor broken on {names(status.aspect)} for {status.arm_label} reads its {rate_name(status)} at "
            f"{_percent(INTERVAL_LEVEL)} on its own, while its row reads only the pass rate, corrected across "
            f"{status.arm_label}'s aspects."
        )
    for status in readings.floors:
        row = next((row for row in readings.on_aspect(status.aspect) if row.arm == status.arm), None)
        floor = floor_of(campaign.floors, status.floor)
        if row is None or row.margin is None or floor is None or not isinstance(floor.clause, RateDrop):
            continue
        # A rate floor and its row read one interval at one level, so they part only on which rates they read (a
        # floor reads every rate its clause names, a row the pass rate alone) or on the points they read it against.
        if status.status == "held" and row.reading == "unclear":
            state = "held"
        elif status.status == "unclear" and row.reading == "within_margin":
            state = "left unclear"
        else:
            continue
        lead = f"The floor {state} on {names(status.aspect)} for {status.arm_label} reads its {rate_name(status)}"
        if floor.clause.max_points != row.margin:
            lines.append(
                f"{lead} against {format_number(floor.clause.max_points)} points, while its row reads the pass rate "
                f"against the {format_number(row.margin)}-point margin."
            )
        else:
            lines.append(f"{lead}, while its row reads only the pass rate, both at {_percent(INTERVAL_LEVEL)}.")
    return lines


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


def _no_question_lines(campaign: EvalCampaign) -> list[str]:
    """The line under code's table where the campaign asks no question: every reading is a lead."""
    return [] if campaign.question.strip() else ["No question was declared, so every reading is a lead."]


def settle_blocks(readings: CampaignReadings, *, names: AspectNames) -> list[ReportBlock]:
    """Each unclear row: why it is unclear, and the cases that would show the arm no worse than the control by more
    than the aspect's margin, if the arms do not differ and if they differ as observed. The count is read at the
    nominal level, as a floor's is, so where a floor reads the same rate against the same points the two counts are
    one. They would not show it better: the row stays unclear unless its gain clears too, and the note says so. A row
    on an aspect that declared no margin has nothing to show it within, and the table above already says no margin
    was declared there, so it is left out; a row already no worse than the margin has nothing left to show.

    :param readings: What code read off a change.
    :ptype readings: CampaignReadings
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The table and its note; nothing where no row is to settle.
    :rtype: list[ReportBlock]
    """
    unclear = [row for row in readings.readings if row.reading == "unclear" and row.margin is not None]
    if not unclear:
        return []
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="arm", header="Arm"),
        TableColumn(key="why", header="Why unclear"),
        TableColumn(key="none", header="If no difference"),
        TableColumn(key="observed", header="If as observed"),
    ]
    rows = [
        table_row(
            {
                "aspect": names(row.aspect),
                "arm": row.arm_label,
                "why": row.reason or None,
                "none": None if row.projection is None else projected_cell(row.projection, row.projection.if_none),
                "observed": None
                if row.projection is None
                else projected_cell(row.projection, row.projection.if_observed),
            }
        )
        for row in unclear
    ]
    return [
        TableBlock(
            section="scoreboard",
            name="to_settle",
            title="To show no worse than the margin",
            columns=columns,
            rows=rows,
            order="as the table above",
            total_rows=len(rows),
        ),
        DisclosureBlock(
            section="scoreboard",
            source="comparisons",
            text=(
                "A count is the independent cases that would show the arm no worse than the control by more than the "
                f"aspect's margin, read at {_percent(INTERVAL_LEVEL)}. A floor on the same rate and points gives the "
                "same count. It would not show the arm better, so its row stays unclear "
                "unless its gain clears too. Each count reads the interval again over this run's cases repeated to "
                "that many; repeats of a case narrow it far less than new cases."
            ),
        ),
    ]


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


#: What a choice's notes say once: the one family, and that ahead is a reading.
CHOICE_NOTE = (
    "A pair is read both ways at once, so every pair on every aspect is one family, corrected together. An arm is "
    "ahead only where the whole interval is one side of zero, and ahead is a reading, not a pick."
)


def pair_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report], *, names: AspectNames
) -> list[ReportBlock]:
    """A choice's table, one row per pair per aspect: each arm's pass count, the difference with its interval, and
    the arm shown ahead or no separation; then the lines that change how a row reads.

    :param campaign: The campaign, a choice.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The table and its lines.
    :rtype: list[ReportBlock]
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
                "first_passed": f"{pair.first_passed} of {pair.first_trials}",
                "second_passed": f"{pair.second_passed} of {pair.second_trials}",
                "change": change,
                "interval": None if pair.interval is None else interval_text(*pair.interval, POINTS),
                "ahead": ahead if pair.confirmatory else f"{ahead}, a lead",
            }
        )
        if pair.interval is not None:
            cells["change"] = TableCell(value=change, low=pair.interval[0], high=pair.interval[1], zero_line=True)
        rows.append(cells)
    blocks: list[ReportBlock] = [
        TableBlock(
            section="scoreboard",
            name="pairs",
            title="By aspect, pair by pair",
            columns=columns,
            rows=rows,
            order="aspect by aspect, each pair of arms in the campaign's order",
            total_rows=len(rows),
        )
    ]
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
    notes += _no_question_lines(campaign)
    notes += _unread_without_control_lines(campaign, reports)
    blocks += [DisclosureBlock(section="scoreboard", source="comparisons", text=note) for note in notes if note]
    return blocks


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
                    section="scoreboard",
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


def measure_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report], *, names: AspectNames
) -> list[ReportBlock]:
    """A measurement's table: each aspect's measures for its arm, each value with its interval, the pass rate as its
    count; a Setup column where an arm was measured under more than one test setup, each cell's apart.

    :param campaign: The campaign, a measurement.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The table and its lines.
    :rtype: list[ReportBlock]
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
            "value": _measured(reading.value, reading.passed, reading.trials),
            "interval": _measured_interval(reading.interval, reading.passed is not None),
            "cases": reading.cases,
        }
        if by_rig:
            values["setup"] = "every setup" if reading.rig is None else short_digest(reading.rig)
        rows.append(table_row(values))
    blocks: list[ReportBlock] = [
        TableBlock(
            section="scoreboard",
            name="measures",
            title="By aspect, measure by measure",
            columns=columns,
            rows=rows,
            order="aspect by aspect, each measure as its cells summarise it",
            total_rows=len(rows),
        ),
        DisclosureBlock(
            section="scoreboard",
            source="comparisons",
            text="One arm is measured, so nothing is compared and no arm is recommended; its bars are its standard.",
        ),
    ]
    notes = _no_question_lines(campaign) + _unread_without_control_lines(campaign, reports)
    notes += _unread_until_candidate_lines(campaign, names)
    blocks += [DisclosureBlock(section="scoreboard", source="comparisons", text=note) for note in notes]
    return blocks


def _measured(value: float | None, passed: int | None, trials: int | None) -> CellValue:
    """A measure's value as the table states it: a pass rate as its count, any other value as measured."""
    if passed is not None and trials is not None:
        return f"{passed} of {trials}"
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
            section="scoreboard",
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
