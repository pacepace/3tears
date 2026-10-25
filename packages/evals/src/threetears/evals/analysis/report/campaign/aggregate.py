"""What a campaign's page reads across its aspects, each aspect weighed alike: the scorecard's equal-weight row and
the cost/quality chart.

Both read the readings' own figures across aspects
(:func:`~threetears.evals.analysis.campaign_readings.campaign_aggregate`), so the page prints what the JSON states,
weighting included. They are descriptive: nothing here is tested, no figure decides a reading, a floor or a bar, and
no word says one arm is better.

Two seams, each with one caller:

- :func:`equal_weight_cells` is each arm's figure across every aspect, which the scorecard
  (:mod:`~threetears.evals.analysis.report.campaign.scorecard`) places in its :data:`EQUAL_WEIGHT_ROW`: in a change,
  the control's pass rate and each candidate's change against it; in a choice, each arm's pass rate. Given only where
  two or more aspects hold a matrix; one aspect's row would repeat it.
- :func:`aggregate_blocks` is the cost/quality chart, in the slot
  :mod:`~threetears.evals.analysis.report.campaign.page` gives it after the kind's own blocks: one point per arm, its
  cost per reply against its pass rate, each with its interval. On one aspect the points are that aspect's own
  figures. It is drawn where the campaign compares more than one pair of figures (two or more aspects, or three or
  more arms), since two arms on one aspect are one comparison, which the matrix's one cell states; and never for a
  measurement. Then a line for each figure withheld, naming the aspects lacking it, and once, in the methods, how a
  figure across two or more aspects weighs them.
"""

from __future__ import annotations

from collections.abc import Mapping

from threetears.evals.analysis.campaign_readings import (
    AggregateFigure,
    AggregateWeighting,
    ArmAggregate,
    CampaignAggregate,
    CampaignReadings,
)
from threetears.evals.analysis.quantities import MONEY, PERCENT, POINTS
from threetears.evals.analysis.report.campaign.names import AspectNames
from threetears.evals.analysis.report.words import arm_tags
from threetears.evals.analysis.report.model import (
    ChartBlock,
    CostQualityPoint,
    DisclosureBlock,
    Report,
    ReportBlock,
    TableCell,
)
from threetears.evals.kernel.campaign import EvalCampaign

__all__ = ["EQUAL_WEIGHT_ROW", "aggregate_blocks", "equal_weight_cells"]

#: What a figure across every aspect, each aspect weighed alike, is headed by.
_ACROSS = "all aspects (equal weight)"

#: The scorecard row holding each arm's figure across every aspect, each aspect weighed alike.
EQUAL_WEIGHT_ROW = "All aspects (equal weight)"

#: The methods line saying how a figure across aspects is read, once on a page that prints one, by the weighting the
#: readings state.
_WEIGHTING_LINES: dict[tuple[str, str], str] = {
    ("equal", "aspects_independent"): (
        "A figure across aspects weighs every aspect alike, and its interval takes the aspects as independent."
    ),
}

#: The marker a withheld cell carries, the line under the matrix naming the aspects that lack its figure.
_WITHHELD = "withheld"


def equal_weight_cells(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> dict[str, TableCell]:
    """Each arm's cell in the scorecard's :data:`EQUAL_WEIGHT_ROW`, by its variant key: the figure, its interval drawn
    beside it, in its column's unit; a withheld figure is a dash marked as withheld.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :return: One cell per arm the matrix holds a column for; none on one aspect, for a measurement, or where no matrix
        is laid out.
    :rtype: dict[str, TableCell]
    """
    aggregate = _row_aggregate(readings)
    if aggregate is None:
        return {}
    cells = {}
    for arm in aggregate.arms:
        figure, unit = _row_figure(arm)
        cells[arm.arm] = _cell(figure, unit)
    return cells


def aggregate_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> list[ReportBlock]:
    """The cost/quality chart where it is drawn, a line per figure withheld from the row or the chart, and the
    weighting once in the methods.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :return: The blocks; none for a measurement, or where neither the row nor the chart is laid out.
    :rtype: list[ReportBlock]
    """
    aggregate = readings.aggregate
    if aggregate is None:
        return []
    names = AspectNames.of(report.bundle for report in reports.values())
    row = _row_aggregate(readings) is not None
    charted = _charted(readings, aggregate)
    blocks: list[ReportBlock] = []
    lines: list[str] = []
    if row:
        lines += [
            _row_withheld_line(arm, names, figure)
            for arm in aggregate.arms
            if (figure := _row_figure(arm)[0]).value is None
        ]
    if charted:
        # Tagged in the order the aspects' scoreboards list the arms, so an arm is one tag, and one colour, on this
        # chart and on every aspect's outcome bar the page lays out.
        listed = list(
            dict.fromkeys(board.variant_key for report in reports.values() for board in report.bundle.scoreboard)
        )
        rank = {key: index for index, key in enumerate(listed)}
        order = sorted(aggregate.arms, key=lambda arm: (arm.arm != readings.control, rank.get(arm.arm, len(rank))))
        tags = arm_tags([arm.arm for arm in order], readings.control)
        points = [point for arm in aggregate.arms if (point := _point(arm, tags[arm.arm])) is not None]
        if len(points) >= 2:
            blocks.append(
                ChartBlock(
                    section="scorecard",
                    chart="cost_quality",
                    title=f"Cost and pass rate, {_ACROSS}" if len(readings.aspects) > 1 else "Cost and pass rate",
                    unit=MONEY,
                    points=points,
                )
            )
        lines += [line for arm in aggregate.arms for line in _chart_withheld_lines(arm, names)]
    blocks += [DisclosureBlock(section="scorecard", source="comparisons", text=line) for line in lines]
    if len(readings.aspects) > 1 and (row or blocks):
        line = _weighting_line(aggregate.weighting)
        blocks.append(DisclosureBlock(section="methods", source="comparisons", text=line))
    return blocks


def _weighting_line(weighting: AggregateWeighting) -> str:
    """The methods line for the weighting the figures were read with."""
    return _WEIGHTING_LINES[(weighting.basis, weighting.intervals)]


def _row_aggregate(readings: CampaignReadings) -> CampaignAggregate | None:
    """The figures the equal-weight row prints: only over two or more aspects, and only where a matrix holds arms (a
    change's readings, a choice's pairs)."""
    aggregate = readings.aggregate
    if aggregate is None or len(readings.aspects) < 2:
        return None
    if not (readings.readings if readings.kind == "change" else readings.pairs):
        return None
    return aggregate


def _row_figure(arm: ArmAggregate) -> tuple[AggregateFigure, str]:
    """The figure an arm's column prints in the equal-weight row, and its unit: a candidate's change in points, else
    the pass rate in percent."""
    return (arm.change, POINTS) if arm.change is not None else (arm.pass_rate, PERCENT)


def _cell(figure: AggregateFigure, unit: str) -> TableCell:
    """One figure as the row's cell: rounded as the matrix's own cells are, its interval drawn beside it, a change
    against zero; a withheld one a dash marked as withheld."""
    if figure.value is None or figure.interval is None:
        return TableCell(value=None, flag=_WITHHELD)
    low, high = figure.interval
    return TableCell(value=round(figure.value, 1), unit=unit, low=low, high=high, zero_line=unit == POINTS)


def _charted(readings: CampaignReadings, aggregate: CampaignAggregate) -> bool:
    """Whether the campaign compares more than one pair of figures: two or more aspects, or three or more arms."""
    return len(aggregate.arms) >= 2 and (len(readings.aspects) >= 2 or len(aggregate.arms) >= 3)


def _point(arm: ArmAggregate, tag: str) -> CostQualityPoint | None:
    """An arm's point on the chart, tagged as the page tags it and named by its label, which a campaign gives each
    arm; None where its cost or its pass rate is withheld."""
    rate, cost = arm.pass_rate, arm.cost
    if rate.value is None or rate.interval is None or cost.value is None or cost.interval is None:
        return None
    return CostQualityPoint(
        arm=arm.arm_label,
        tag=tag,
        short_name=arm.arm_label,
        cost=cost.value,
        cost_low=cost.interval[0],
        cost_high=cost.interval[1],
        pass_rate=rate.value,
        pass_rate_low=rate.interval[0],
        pass_rate_high=rate.interval[1],
    )


def _aspects(names: AspectNames, labels: list[str]) -> str:
    """The aspects a figure is missing on, by the names the page calls them."""
    return ", ".join(names(label) for label in labels)


def _row_withheld_line(arm: ArmAggregate, names: AspectNames, figure: AggregateFigure) -> str:
    """Why an arm's cell in the equal-weight row is withheld: the aspects lacking its figure."""
    if arm.change is not None:
        return f"{arm.arm_label}'s change across aspects is withheld: it has none on {_aspects(names, figure.missing)}."
    return f"{arm.arm_label}'s pass rate across aspects is withheld: it has none on {_aspects(names, figure.missing)}."


def _chart_withheld_lines(arm: ArmAggregate, names: AspectNames) -> list[str]:
    """Why an arm is left off the chart: the aspects lacking its cost per reply, then those lacking its pass rate."""
    lines = []
    if arm.cost.value is None:
        lines.append(
            f"{arm.arm_label} is left off the chart: it has no cost per reply on {_aspects(names, arm.cost.missing)}."
        )
    if arm.pass_rate.value is None:
        lines.append(
            f"{arm.arm_label} is left off the chart: it has no pass rate on {_aspects(names, arm.pass_rate.missing)}."
        )
    return lines
