"""What every chart resolver shares: the chart it is drawing, and the rules each kind fills its data by.

A resolver (one per kind, in a ``resolve_*`` module's ``RESOLVERS``) turns a stored chart's
:class:`~threetears.evals.kernel.report_plan.ChartSpec` into the kit's chart data, reading the rows its query
selects (:func:`~threetears.evals.analysis.report_rows.query_rows`, the one owner of that selection). It is handed a
:class:`ChartContext` and returns a :class:`Drawn`. The rules every kind fills by (the P4 design, §3) are here, once:

- **Units.** A figure's unit as the page reads it (:func:`figure_unit`): a measure's declared unit; a pass-fail
  rate's scale read from every figure the input holds for it (a share where each lies in 0 to 1, a percent
  otherwise), as the figure keys read it; an ordinal judged dimension a ``score``; a contrast of how each trial ended,
  the pass rate, a share; a scoreboard figure money (a reply's cost, or the trials' in all) or a count.
  :func:`~threetears.evals.analysis.report.plan_page.look.quantity` says how it is drawn. A contrast's change
  (:func:`contrast_unit`) is in points where it is a rate's, else in its figures' own unit: an ordinal dimension's is
  a score change, never points.
- **One unit per axis** (:func:`unit_refusal`, over the measures a chart draws, :func:`drawn_keys`, or the contrasts
  it draws): an axis draws measures of one unit, each at its own scale, two units being one where they are drawn
  alike (a share and a percent both as a percent; any rate's change in points), each measure still drawn at its own
  scale (:func:`unit_quantities`). A kind that draws each measure in its
  own panel (a heatmap's matrix, a dot strip's strip, bars in panels by slice or measure) gives each panel its
  measure's unit and format, panels of one unit sharing an axis's span; any other chart reading measures in several
  units is refused, and so is a chart of contrasts in several, whose groups share one axis. The plan checks refuse
  the same charts, by the same function and by no other unit rule, so a plan they accept draws and a chart the page
  draws is accepted.
- **A figure's value** (:func:`stat_value`): its mean; else the statistic a criterion on the measure reads; else its
  p95, else its p50. **Its interval** (:func:`stat_interval`): its ``ci``, else its mean plus or minus its half-width.
  **Intervals are drawn only where the chart states a level** (``ChartSpec.level``): the kit refuses intervals
  without one.
- **An interval computed here** (:func:`computed_interval`, :func:`figure_interval`): an arm's overall figure that
  states no interval, half-width or standard error, but is a pass/fail rate with a count (a rate graded pass/fail or
  by class label), has the interval its binomial spread gives at its level, the ``spread.<arm>.<key>.overall.interval``
  figure key (:func:`~threetears.evals.analysis.separation.spread_interval`: a normal approximation, held to none to
  all; none for a rate of none or all). A chart draws it only where its stored level says so (``95% computed``,
  :data:`COMPUTED`), which code completes a dot strip, bars or a table of arms' figures with where none of its
  figures states an interval: so a stored chart draws as it was stored, and the level names the interval as code's.
- **Reference lines** (:func:`reference_line`): the stored reference's figure key read off the input, in display
  units: a margin as minus the margin on a change axis, a floor or budget at the criterion's threshold. A kind that
  draws its criterion's line where the plan stores none (a dot strip's floor, range bars' budget, a heatmap's cell
  floor) reads it by :func:`criterion_line`: the one criterion on the measure, at the statistic drawn, read where the
  chart reads it.
- **A config's colour** (:func:`factor_token`, :func:`factor_legend`): a chart of a factorial grid's configs colours
  each by its first factor's level, the kit's series in that factor's order, and its legend names the levels.
- **A scatter's labels** (:func:`point_labels`): every point's name where a chart draws at most eight, else only the
  stressed and the recommended one's, so labels never bury the points; narrow, the arm's letter tag.
- **Row order** (:meth:`ChartContext.arm_order`): a summary chart's rows in the options table's order (GENERATOR
  §5.3); a section's in its query's order, else the page's.
- **The descriptive rule** (:func:`separated`): a change code computes from two point figures is bold where it lies
  beyond 1.96 times the two standard errors' hypotenuse (a half-width counts as 1.96 standard errors), muted
  otherwise, and uncorrected (GENERATOR §5.2).

Anything a resolver cannot draw is refused, never drawn empty: :meth:`ChartContext.refuse` raises an
:class:`~threetears.evals.analysis.errors.UnresolvableReference` naming the chart's kind and source, which the service
maps to "generate the analysis again".
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock
from threetears.evals.analysis.report.charts import ChartReference, KitChart, LegendItem
from threetears.evals.analysis.report.figure_keys import Figure, NamedKind, format_figure, resolve
from threetears.evals.analysis.report.plan_page.look import PageLook, Quantity, figure_text, quantity, series_token
from threetears.evals.analysis.report.plan_page.model import PLANNED_SECTION
from threetears.evals.analysis.report.plan_page.words import (
    ASPECT_MEASURE_TEMPLATE,
    COMPUTED_NOTE,
    COMPUTED_REPEATS_NOTE,
    MISSING_TEMPLATE,
    REFERENCE_TEMPLATE,
)
from threetears.evals.analysis.report_input import Cell, Contrast, Criterion, ReportInput, Stat, Stratum, Task
from threetears.evals.analysis.report_margins import criterion_aspect, criterion_place, lead_aspect
from threetears.evals.analysis.report_rows import SourceRow, query_rows
from threetears.evals.analysis.style import measure_name
from threetears.evals.kernel.metrics import goal_check_of
from threetears.evals.kernel.report_plan import ChartQuery, ChartSpec

__all__ = [
    "COMPUTED",
    "STORED_SPAN_REFUSAL",
    "ChartContext",
    "CriterionWhere",
    "Drawn",
    "KindResolver",
    "chart_block",
    "contrast_groups",
    "contrast_unit",
    "contrast_unit_of",
    "contrast_words",
    "count_text",
    "computed_interval",
    "computed_notes",
    "criterion_line",
    "drawn_keys",
    "factor_legend",
    "factor_token",
    "figure_interval",
    "figure_stat",
    "figure_unit",
    "interval_words",
    "measure_words",
    "misses_line",
    "missing_note",
    "point_labels",
    "reference_line",
    "separated",
    "slice_keys",
    "slice_name",
    "spread",
    "stat_interval",
    "stat_value",
    "unit_quantities",
    "unit_refusal",
]

#: The z a 95% interval spans either side, which the descriptive rule reads a change against (GENERATOR §5.2).
_Z95 = 1.96

#: The statistics a figure is drawn at where it has no mean, in the order they are tried after a criterion's own.
_FALLBACK_STATISTICS = ("p95", "p50")


@dataclass(frozen=True)
class ChartContext:
    """One chart being resolved: what the plan stores of it, the input it reads, and the page's look.

    :ivar spec: The stored chart.
    :ivar report_input: The report input the plan was written against.
    :ivar look: The page's look: arm colours, tags, names, order, survivors and the recommended arm.
    :ivar title: The chart's title as its block carries it: the plan's, else the section's headline.
    :ivar summary: Whether it is a summary chart, whose rows follow the options table (GENERATOR §5.3).
    """

    spec: ChartSpec
    report_input: ReportInput
    look: PageLook
    title: str
    summary: bool = False

    @property
    def query(self) -> ChartQuery:
        """The chart's query."""
        return self.spec.query

    def rows(self) -> list[SourceRow]:
        """Every row the chart's query selects, as the plan checks read them.

        :return: The rows, in the input's order.
        :rtype: list[SourceRow]
        """
        return query_rows(self.spec.query, self.report_input)

    def refuse(self, why: str) -> UnresolvableReference:
        """The refusal of this chart: its kind, its source and why it cannot be drawn.

        :param why: Why, as a clause.
        :ptype why: str
        :return: The error to raise.
        :rtype: UnresolvableReference
        """
        return UnresolvableReference(
            f"the {self.spec.kind!r} chart over {self.spec.query.source!r} ({self.title!r}) cannot be drawn: {why}"
        )

    def arm_order(self, arm_ids: Iterable[str]) -> list[str]:
        """``arm_ids`` in the order the chart shows arms: a summary chart's the options table's, a section's its
        query's arms where it lists them, else the page's.

        :param arm_ids: The arms the chart draws.
        :ptype arm_ids: Iterable[str]
        :return: The same arms, in the chart's order.
        :rtype: list[str]
        """
        given = list(dict.fromkeys(arm_ids))
        if self.summary or not self.spec.query.arms:
            return self.look.ordered(given)
        place = {arm: index for index, arm in enumerate(self.spec.query.arms)}
        return sorted(given, key=lambda arm: (place.get(arm, len(place)), given.index(arm)))

    @property
    def emphasis(self) -> str | None:
        """The one row or point the plan stresses, by id; None for none."""
        return self.spec.emphasis or None

    @property
    def level(self) -> str:
        """The interval level the chart states (``ChartSpec.level``); empty where it states none, and draws none."""
        return self.spec.level


@dataclass(frozen=True)
class Drawn:
    """What a resolver returns: the chart's block, and code's lines under it.

    :ivar block: The chart, carrying the kit's data, or a table.
    :ivar notes: Code's captions, in order.
    """

    block: ChartBlock | TableBlock
    notes: tuple[str, ...] = field(default=())


#: A chart kind's resolver: the context in, the drawn chart out.
KindResolver = Callable[[ChartContext], Drawn]


def chart_block(context: ChartContext, data: KitChart) -> ChartBlock:
    """The chart's block: its kind, its title and the kit's data, on the planned page's fixed section.

    :param context: The chart being resolved.
    :ptype context: ChartContext
    :param data: The kit's chart data.
    :ptype data: KitChart
    :return: The block.
    :rtype: ChartBlock
    """
    kind = context.spec.kind
    if kind == "table":
        raise context.refuse("a table is laid out as a table block, not drawn as a chart")
    return ChartBlock(section=PLANNED_SECTION, chart=kind, title=context.title, data=data)


def _held_means(report_input: ReportInput, key: str) -> list[float]:
    """Every mean the input holds for a measure or judged dimension: overall, per aspect, per task, stratum and cell."""
    stats: list[Stat] = []
    for table in (report_input.measures, report_input.judged):
        if key in table:
            stats.extend(table[key].per_arm.values())
            for figures in table[key].by_aspect.values():
                stats.extend(figures.values())
    slices: list[Task | Stratum | Cell] = [*report_input.tasks, *report_input.strata, *report_input.cells.values()]
    for held in slices:
        stats.extend(figures[key] for figures in held.per_arm.values() if key in figures)
    return [float(stat.mean) for stat in stats if stat.mean is not None]


def _rate_unit(values: Sequence[float]) -> str:
    return "share" if all(0 <= value <= 1 for value in values) else "percent"


#: The scoreboard's figures that are money, each to how it is stated: a reply's cost, and what the arm's trials cost
#: in all. Every other scoreboard figure is a count.
_SCOREBOARD_UNITS = {"reply_cost_usd": "usd", "test_cost_usd": "dollars"}


def figure_unit(report_input: ReportInput, key: str, *, kind: str | None = None, source: str | None = None) -> str:
    """A measure's or judged dimension's unit as the page reads it (this module's Units), wherever its figures sit; a
    scoreboard figure's, where the chart reads the scoreboard.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure or judged dimension, by key; a scoreboard figure's name over the scoreboard.
    :ptype key: str
    :param kind: A contrast's kind (``measure``, ``judged``, ``trial``), where the key is a contrast's; None otherwise.
    :ptype kind: str | None
    :param source: The chart's source, where ``scoreboard`` makes the key a scoreboard figure; None otherwise.
    :ptype source: str | None
    :return: ``share``, ``percent``, ``score``, a money unit, ``ms``, ``s``, any other declared unit, or empty (a
        bare number, as every scoreboard count is).
    :rtype: str
    """
    if source == "scoreboard":
        return _SCOREBOARD_UNITS.get(key, "")
    if kind == "trial":
        return "share"
    measure = report_input.measures.get(key)
    if measure is not None and kind != "judged":
        unit = measure.unit or ""
        if unit == "pct":
            return "percent"
        if not unit and measure.grade in ("pass_fail", "class_label"):
            return _rate_unit(_held_means(report_input, key))
        return unit
    judged = report_input.judged.get(key)
    if judged is not None:
        return _rate_unit(_held_means(report_input, key)) if judged.scale == "pass_fail" else "score"
    return ""


def measure_words(report_input: ReportInput, key: str, *, kind: str | None = None) -> str:
    """What a reader calls a measure or judged dimension: its label, else the package's name for it; how each trial
    ended (a contrast of kind ``trial``) is the pass rate. A goal check whose label is its bare expression (the input
    declares it no display name) is named as the package names an undeclared check, its expression set as code
    (:func:`~threetears.evals.analysis.style.measure_name`), so it reads as the check it is, never as a stray key.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The key.
    :ptype key: str
    :param kind: A contrast's kind, where the key is a contrast's.
    :ptype kind: str | None
    :return: The name.
    :rtype: str
    """
    if kind == "trial":
        return "pass rate"
    if key in report_input.measures and kind != "judged":
        label = report_input.measures[key].label
        return measure_name(key) if label is None or label == goal_check_of(key) else label
    if key in report_input.judged:
        return report_input.judged[key].label or measure_name(key)
    return measure_name(key)


@dataclass(frozen=True)
class ContrastGroup:
    """One measure's selected contrasts read on one aspect, by arm (:func:`contrast_groups`).

    :ivar key: The measure.
    :ivar aspect: The reader name of the aspect they were read on; None where the input names no aspect.
    :ivar split: Whether the measure's selected contrasts sit on more than one aspect, so a reader is told which.
    :ivar by_arm: The contrasts, by arm.
    """

    key: str
    aspect: str | None
    split: bool
    by_arm: dict[str, Contrast]


def contrast_groups(
    context: ChartContext, contrasts: Sequence[Contrast], keys: Iterable[str] = ()
) -> list[ContrastGroup]:
    """The selected contrasts by measure, then by the aspect each was read on.

    A query naming no aspect selects every aspect's contrasts, so two aspects may each hold a contrast of one arm on
    one measure key; keyed by arm alone, one would replace the other unseen. A contrast naming no aspect is read on
    the lead one (the first asked, else the first), as the input's by-arm maps are.

    :param context: The chart being resolved.
    :ptype context: ChartContext
    :param contrasts: The contrasts it selects.
    :ptype contrasts: Sequence[Contrast]
    :param keys: Measures to take first, in this order; the rest follow in the contrasts' order.
    :ptype keys: Iterable[str]
    :return: The groups, a measure's together, its aspects in the contrasts' order.
    :rtype: list[ContrastGroup]
    """
    aspects = context.report_input.aspects
    lead = next((aspect.id for aspect in aspects if aspect.asked), next((aspect.id for aspect in aspects), None))
    groups: list[ContrastGroup] = []
    for key in dict.fromkeys([*keys, *(contrast.measure for contrast in contrasts)]):
        on_key = [contrast for contrast in contrasts if contrast.measure == key]
        read_on = list(dict.fromkeys(contrast.aspect or lead for contrast in on_key))
        for aspect in read_on:
            groups.append(
                ContrastGroup(
                    key=key,
                    aspect=context.look.reader("aspect", aspect) if aspect is not None else None,
                    split=len(read_on) > 1,
                    by_arm={c.arm: c for c in on_key if (c.aspect or lead) == aspect},
                )
            )
    return groups


def contrast_words(report_input: ReportInput, contrast: Contrast, aspect: str | None) -> str:
    """What a reader calls a contrast's measure, naming the aspect it was read on where one is given
    (:class:`ContrastGroup`).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param contrast: The contrast.
    :ptype contrast: Contrast
    :param aspect: The aspect's reader name, where the reader is told it; None where not.
    :ptype aspect: str | None
    :return: The words.
    :rtype: str
    """
    words = measure_words(report_input, contrast.measure, kind=contrast.kind)
    return ASPECT_MEASURE_TEMPLATE.format(measure=words, aspect=aspect) if aspect is not None else words


def figure_stat(report_input: ReportInput, key: str, arm: str, aspect: str = "") -> Stat | None:
    """One arm's figure on a measure or judged dimension, on the query's aspect where the key holds it, else overall.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure or judged dimension.
    :ptype key: str
    :param arm: The arm.
    :ptype arm: str
    :param aspect: The query's aspect; empty for the lead aspect's figures.
    :ptype aspect: str
    :return: The figure; None where the input holds none.
    :rtype: Stat | None
    """
    for table in (report_input.measures, report_input.judged):
        if key in table:
            figures = table[key]
            per_arm = figures.by_aspect.get(aspect, figures.per_arm) if aspect else figures.per_arm
            return per_arm.get(arm)
    return None


def _criterion_statistic(report_input: ReportInput, key: str) -> str | None:
    return next(
        (c.statistic for c in report_input.criteria if c.measure == key and c.statistic != "mean"),
        None,
    )


def stat_value(report_input: ReportInput, key: str, stat: Stat) -> tuple[float, str] | None:
    """The value a figure is drawn at, and the statistic it is (this module's figure value).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure the figure is on.
    :ptype key: str
    :param stat: The figure.
    :ptype stat: Stat
    :return: ``(value, statistic)``, ``statistic`` being ``mean``, ``p50`` or ``p95``; None where it holds none.
    :rtype: tuple[float, str] | None
    """
    if stat.mean is not None:
        return float(stat.mean), "mean"
    statistic = _criterion_statistic(report_input, key)
    for name in (*([statistic] if statistic else []), *_FALLBACK_STATISTICS):
        value = getattr(stat, name)
        if value is not None:
            return float(value), name
    return None


#: The word a chart's level carries where its intervals are computed here, each from its rate's count.
COMPUTED = "computed"


def computed_interval(
    report_input: ReportInput, key: str, arm: str, stat: Stat
) -> tuple[tuple[float, float], float] | None:
    """The interval code computes for an arm's overall figure that states none (this module's interval computed
    here), in the figure's units, and its level as a share; None where the figure states a spread of its own, is not
    the arm's overall figure, or reads none (no count, no rate, a rate of none or all).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure or judged dimension.
    :ptype key: str
    :param arm: The arm.
    :ptype arm: str
    :param stat: The figure a chart draws.
    :ptype stat: Stat
    :return: ``((low, high), level)``; None for none.
    :rtype: tuple[tuple[float, float], float] | None
    """
    if stat.mean is None or stat.ci is not None or stat.half_width is not None or stat.sem is not None:
        return None
    table = report_input.measures.get(key) or report_input.judged.get(key)
    if table is None or table.per_arm.get(arm) is not stat:
        return None
    try:
        interval = resolve(f"spread.{arm}.{key}.overall.interval", report_input).value
        level = resolve(f"spread.{arm}.{key}.overall.level", report_input).value
    except UnresolvableReference:
        # NOSILENT: a figure with no count, no rate, or a rate of none or all reads no interval: the chart draws its
        # dot or bar with none, as it draws any figure that states none.
        return None
    assert isinstance(interval, tuple) and isinstance(level, float), (interval, level)
    return (float(interval[0]), float(interval[1])), level


def computed_notes(context: ChartContext) -> tuple[str, ...]:
    """What a chart whose intervals are computed here says of them (:data:`COMPUTED`), and, where a figure's count is
    trials (a case run again counts again), that each repeat was read as independent; nothing otherwise.

    :param context: The chart being resolved.
    :ptype context: ChartContext
    :return: The note, or none.
    :rtype: tuple[str, ...]
    """
    if not context.level.endswith(COMPUTED):
        return ()
    repeated = any(
        _counts_trials(context.report_input, row.arm, row.key)
        for row in context.rows()
        if isinstance(row.item, Stat) and computed_interval(context.report_input, row.key, row.arm, row.item)
    )
    return (COMPUTED_NOTE, *((COMPUTED_REPEATS_NOTE,) if repeated else ()))


def _counts_trials(report_input: ReportInput, arm: str, key: str) -> bool:
    """Whether an arm's overall spread on a figure rests on a count of trials (a case run again counts again)."""
    try:
        return resolve(f"spread.{arm}.{key}.overall.counts", report_input).value == "trials"
    except UnresolvableReference:
        # NOSILENT: a figure whose count is unstated rests on no trials; its interval reads none at all.
        return False


def figure_interval(context: ChartContext, key: str, arm: str, stat: Stat) -> tuple[float, float] | None:
    """The interval a chart draws on an arm's figure: its own (:func:`stat_interval`), else, where the chart's level
    is computed here (:data:`COMPUTED`), the one code computes (:func:`computed_interval`); None for neither.

    :param context: The chart being resolved.
    :ptype context: ChartContext
    :param key: The measure or judged dimension.
    :ptype key: str
    :param arm: The arm.
    :ptype arm: str
    :param stat: The figure.
    :ptype stat: Stat
    :return: ``(low, high)``, in the figure's units; None for none.
    :rtype: tuple[float, float] | None
    """
    own = stat_interval(stat)
    if own is not None or not context.level.endswith(COMPUTED):
        return own
    found = computed_interval(context.report_input, key, arm, stat)
    return found[0] if found is not None else None


def stat_interval(stat: Stat) -> tuple[float, float] | None:
    """A figure's interval: its ``ci``, else its mean plus or minus its half-width; None for neither.

    :param stat: The figure.
    :ptype stat: Stat
    :return: ``(low, high)``, in the figure's units; None where it carries none.
    :rtype: tuple[float, float] | None
    """
    if stat.ci is not None:
        return float(stat.ci[0]), float(stat.ci[1])
    if stat.mean is not None and stat.half_width is not None:
        return float(stat.mean - stat.half_width), float(stat.mean + stat.half_width)
    return None


def spread(stat: Stat) -> float | None:
    """A figure's 95% half-width for the descriptive rule: 1.96 standard errors, else its half-width.

    :param stat: The figure.
    :ptype stat: Stat
    :return: The half-width, in the figure's units; None where it carries neither.
    :rtype: float | None
    """
    if stat.sem is not None:
        return _Z95 * float(stat.sem)
    return float(stat.half_width) if stat.half_width is not None else None


def separated(change: float, spreads: Sequence[float | None]) -> bool | None:
    """Whether a change code computed from point figures lies beyond the two figures' 95% spreads combined (this
    module's descriptive rule).

    :param change: The change, in the figures' units.
    :ptype change: float
    :param spreads: Each figure's 95% half-width (:func:`spread`).
    :ptype spreads: Sequence[float | None]
    :return: True or False; None where a spread is missing, and the rule cannot be read.
    :rtype: bool | None
    """
    if any(one is None for one in spreads):
        return None
    return abs(change) > math.hypot(*(one for one in spreads if one is not None))


def interval_words(low: float, high: float, drawn: Quantity) -> str:
    """An interval in a tooltip: its two ends by the package's number rule, joined as prose joins them.

    :param low: Its low end, in display units.
    :ptype low: float
    :param high: Its high end, in display units.
    :ptype high: float
    :param drawn: The quantity drawn.
    :ptype drawn: Quantity
    :return: The words (``-10.8 to -3.4``).
    :rtype: str
    """
    return f"{figure_text(low, drawn)} to {figure_text(high, drawn)}"


def reference_line(context: ChartContext, drawn: Quantity) -> ChartReference | None:
    """The chart's stored reference line, read off the input in display units, labelled in code's words.

    :param context: The chart being resolved.
    :ptype context: ChartContext
    :param drawn: The quantity the chart draws.
    :ptype drawn: Quantity
    :return: The line; None where the plan stores none.
    :rtype: ChartReference | None
    :raises UnresolvableReference: The reference's key no longer resolves to a number.
    """
    if not context.spec.reference:
        return None
    stored = context.spec.reference[0]
    figure: Figure = resolve(stored.key, context.report_input)
    if figure.kind != "number" or not isinstance(figure.value, float):
        raise context.refuse(f"its {stored.kind} line's key {stored.key!r} names no single number")
    if stored.kind == "margin":
        # A margin is in points, on the change axis below zero, whatever the chart's own quantity scales.
        value = -figure.value
        label = REFERENCE_TEMPLATE.format(kind=stored.kind, value=figure_text(value, quantity("points")))
    else:
        value = figure.value * drawn.factor
        label = REFERENCE_TEMPLATE.format(kind=stored.kind, value=figure_text(value, drawn))
    return ChartReference(value=value, label=label, kind=stored.kind)


#: Where a chart reads a measure, for the criterion whose line it draws (:func:`criterion_line`): its overall figures,
#: every task's (a criterion read in each task), one task's (that, or one read within that task alone), or every
#: target-matrix cell's.
CriterionWhere = Literal["overall", "tasks", "cells"] | tuple[Literal["task"], str]


def _reads_there(criterion: Criterion, where: CriterionWhere) -> bool:
    """Whether a criterion reads its figure where the chart does, by where it reads
    (:func:`~threetears.evals.analysis.report_margins.criterion_place`, the one reading of its scope)."""
    place = criterion_place(criterion)
    if where == "overall":
        return place == "overall"
    if where == "cells":
        return place == "cell"
    if place != "task":
        return False
    if where == "tasks":
        return criterion.task is None
    _, task = where
    return criterion.task in (None, task)


def criterion_line(
    context: ChartContext, key: str, drawn: Quantity, *, where: CriterionWhere, statistic: str = "mean"
) -> ChartReference | None:
    """The line a declared criterion draws on a chart that stores none: the threshold of the one criterion that reads
    ``key`` at ``statistic`` where the chart reads it, on the chart's aspect (its query's, else the lead aspect), in
    display units, a floor where higher is better and a budget
    where lower is, labelled in code's words.

    :param context: The chart being resolved.
    :ptype context: ChartContext
    :param key: The measure or judged dimension drawn.
    :ptype key: str
    :param drawn: The quantity the chart draws it as.
    :ptype drawn: Quantity
    :param where: Where the chart reads the measure (:data:`CriterionWhere`).
    :ptype where: CriterionWhere
    :param statistic: The statistic drawn: ``mean``, ``p50`` or ``p95``.
    :ptype statistic: str
    :return: The line; None where no criterion, or more than one line, reads it there.
    :rtype: ChartReference | None
    """
    held = context.report_input
    aspect = context.query.aspect or lead_aspect(held)
    lines = {
        (float(criterion.threshold), criterion.direction)
        for criterion in held.criteria
        if criterion.measure == key
        and criterion.statistic == statistic
        and criterion_aspect(held, criterion) == aspect
        and _reads_there(criterion, where)
    }
    if len(lines) != 1:
        return None
    ((threshold, direction),) = lines
    kind: Literal["floor", "budget"] = "floor" if direction == "higher_is_better" else "budget"
    value = threshold * drawn.factor
    return ChartReference(
        value=value, label=REFERENCE_TEMPLATE.format(kind=kind, value=figure_text(value, drawn)), kind=kind
    )


def misses_line(value: float, line: ChartReference | None) -> bool:
    """Whether a drawn value misses a floor (under it) or a budget (over it); False for any other line or none.

    :param value: The value, in display units.
    :ptype value: float
    :param line: The chart's line.
    :ptype line: ChartReference | None
    :return: Whether it misses.
    :rtype: bool
    """
    if line is None:
        return False
    # Written as not meeting the line, so a value no comparison holds for (NaN) misses it rather than passing.
    if line.kind == "floor":
        return not value >= line.value
    return not value <= line.value if line.kind == "budget" else False


def slice_name(context: ChartContext, item: Stratum | Task) -> str:
    """A stratum's or task's reader name.

    :param context: The chart being resolved.
    :ptype context: ChartContext
    :param item: The stratum or task.
    :ptype item: Stratum | Task
    :return: Its name.
    :rtype: str
    """
    kind: NamedKind = "stratum" if isinstance(item, Stratum) else "task"
    return item.name or context.look.reader(kind, item.id)


def _held_keys(slices: Iterable[Stratum | Task | Cell]) -> list[str]:
    """Every measure some arm holds a figure on in the slices (or cells), each once, in the order first held."""
    return list(dict.fromkeys(key for item in slices for figures in item.per_arm.values() for key in figures))


#: Why a chart of strata or tasks that names no measure cannot be drawn when its slices hold other than one.
_ONE_HELD_TEMPLATE = "its query names no measure, and its slices hold {found}"
#: Why a chart cannot draw the measures it reads on one axis.
_UNITS_TEMPLATE = "its measures are in more than one unit ({units})"
#: Why a chart drawing its measures in several units, a panel each, cannot take the one axis span a plan stores.
STORED_SPAN_REFUSAL = "its stored axis spans one unit, and its measures are in more than one"


def slice_keys(context: ChartContext, slices: Sequence[Stratum | Task]) -> list[str]:
    """The measures a chart of strata or tasks reads: the ones its query names, else the one its slices hold.

    :param context: The chart being resolved.
    :ptype context: ChartContext
    :param slices: The slices it selects.
    :ptype slices: Sequence[Stratum | Task]
    :return: The measures, by key.
    :rtype: list[str]
    :raises UnresolvableReference: The query names none, and the slices hold other than one.
    """
    if context.query.measures:
        return list(context.query.measures)
    found = _held_keys(slices)
    if len(found) != 1:
        raise context.refuse(_ONE_HELD_TEMPLATE.format(found=found))
    return found


#: The kinds whose chart of slices or cells reads every measure they hold where its query names none: a matrix each.
_EVERY_HELD = frozenset({"heatmap", "diff_heatmap"})
#: The sources whose rows are figures by measure, one arm's each; and those whose rows hold figures by measure per arm.
_FIGURE_SOURCES = frozenset({"measures", "judged", "configs"})
_SLICE_SOURCES = frozenset({"strata", "tasks", "cells"})

#: The kinds the one-unit rule holds (:func:`unit_refusal`), each over the sources it draws figures by measure from.
_UNIT_RULED: dict[str, frozenset[str]] = {
    "bars": frozenset({"measures", "judged", "configs", "scoreboard", "strata", "tasks"}),
    "dot_interval": frozenset({"contrasts", "measures", "judged", "strata", "tasks"}),
    "heatmap": frozenset({"cells", "strata", "tasks"}),
    "diff_heatmap": frozenset({"cells", "strata", "tasks"}),
    "range_bars": frozenset({"measures", "tasks"}),
    "diverging_bars": frozenset({"contrasts", "measures", "judged", "strata", "tasks"}),
    "lines": frozenset({"strata", "tasks"}),
    "forest": frozenset({"contrasts"}),
}

#: The kinds that draw what they read as each arm less the control: a change, on a change axis.
_CHANGE_KINDS = frozenset({"forest", "diverging_bars"})


def _has_p95(stat: Stat | None) -> bool:
    return stat is not None and stat.p95 is not None


def drawn_keys(kind: str, query: ChartQuery, report_input: ReportInput) -> list[str]:
    """The measures a chart draws, by key, as its kind's resolver reads them (this module's Units), each once:

    - over measures, judged dimensions or configs: the keys of the figures its query selects; range bars only those
      holding a p95;
    - over the scoreboard: the figures its query names;
    - over strata, tasks or cells: the measures its query names, else every one the slices hold for a heatmap or a
      difference heatmap (a matrix each), and the one they hold for any other kind (:func:`slice_keys`); range bars
      keep only those an arm the query keeps holds a p95 on in some slice;
    - over any other source: none.

    :param kind: The chart's kind.
    :ptype kind: str
    :param query: The chart's query.
    :ptype query: ChartQuery
    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The measures, by key, in the order the chart reads them.
    :rtype: list[str]
    """
    source = query.source
    if source == "scoreboard":
        return list(dict.fromkeys(query.measures))
    rows = query_rows(query, report_input)
    if source in _FIGURE_SOURCES:
        return list(
            dict.fromkeys(
                row.key
                for row in rows
                if row.key and (kind != "range_bars" or (isinstance(row.item, Stat) and _has_p95(row.item)))
            )
        )
    if source not in _SLICE_SOURCES:
        return []
    slices = [row.item for row in rows if isinstance(row.item, Stratum | Task | Cell)]
    named = list(dict.fromkeys(query.measures))
    held = _held_keys(slices)
    if kind == "range_bars":
        arms = set(query.arms) or {arm.id for arm in report_input.arms}
        return [
            key
            for key in named or held
            if any(_has_p95(item.per_arm.get(arm, {}).get(key)) for item in slices for arm in arms)
        ]
    if named or kind in _EVERY_HELD:
        return named or held
    return held if len(held) == 1 else []


def unit_quantities(report_input: ReportInput, values: Mapping[str, Sequence[float]]) -> dict[str, Quantity]:
    """Each measure's quantity on a chart that draws it in its own unit (this module's one unit per axis): its unit's
    (:func:`figure_unit`), at its own factor, so each figure is drawn at its own scale; and the measures drawn alike (a
    share beside a percent) share one quantity but for the factor, its format following every figure the chart draws
    of them, so they read alike on one axis.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param values: Each measure the chart draws, by key, to the figures it draws of it, in the input's units.
    :ptype values: Mapping[str, Sequence[float]]
    :return: Each measure's quantity, by key.
    :rtype: dict[str, Quantity]
    """
    units = {key: figure_unit(report_input, key) for key in values}
    # A quantity's equality ignores its factor, so the quantity a unit draws as is the class of the units drawn alike.
    alike = {key: quantity(units[key]) for key in values}
    shown: dict[Quantity, list[float]] = {}
    lead: dict[Quantity, str] = {}
    for key, figures in values.items():
        lead.setdefault(alike[key], units[key])
        shown.setdefault(alike[key], []).extend(figure * alike[key].factor for figure in figures)
    drawn = {each: quantity(lead[each], shown[each]) for each in shown}
    return {key: dataclasses.replace(drawn[alike[key]], factor=alike[key].factor) for key in values}


def _a_panel_per_measure(kind: str, query: ChartQuery) -> bool:
    """Whether a chart draws each measure it reads in its own panel, on its own axis: a heatmap's matrix per measure,
    a dot strip's strip per measure (unless its panels are by arm), and bars in panels by slice or measure where the
    plan asks for them (not over configs, whose panels are a factor's levels)."""
    if kind == "heatmap":
        return True
    if kind == "dot_interval":
        return query.source in ("measures", "judged") or query.small_multiples_by != "arm"
    if kind == "bars":
        return query.source != "configs" and query.small_multiples_by in ("aspect", "task")
    return False


def unit_refusal(kind: str, query: ChartQuery, report_input: ReportInput) -> str:
    """Why a chart cannot draw the measures it reads (the one-unit rule the plan checks and the resolvers share): a
    chart of strata or tasks that names no measure where its slices hold other than one (a heatmap and a difference
    heatmap read every one), or measures in more than one unit (:func:`figure_unit`) on one axis. A kind that draws
    each measure in its own panel gives each panel its measure's unit and may read several.

    :param kind: The chart's kind.
    :ptype kind: str
    :param query: The chart's query.
    :ptype query: ChartQuery
    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Why, as a clause; empty where the chart can be drawn in its units, or its kind reads no measure by key
        over that source.
    :rtype: str
    """
    source = query.source
    if source not in _UNIT_RULED.get(kind, frozenset()):
        return ""
    if source in _SLICE_SOURCES and kind not in _EVERY_HELD and kind != "range_bars" and not query.measures:
        held = _held_keys(row.item for row in query_rows(query, report_input) if isinstance(row.item, Stratum | Task))
        if len(held) != 1:
            return _ONE_HELD_TEMPLATE.format(found=held)
    if source == "contrasts":
        units = _contrast_units(query, report_input)
        change = True
    else:
        units = {figure_unit(report_input, key, source=source) for key in drawn_keys(kind, query, report_input)}
        change = kind in _CHANGE_KINDS
    drawn_as = {quantity(unit, change=change) for unit in units}
    if len(drawn_as) > 1 and (source == "contrasts" or not _a_panel_per_measure(kind, query)):
        return _UNITS_TEMPLATE.format(units=sorted(units))
    return ""


def contrast_unit(report_input: ReportInput, contrast: Contrast) -> str:
    """The unit a contrast's change is in, as the page draws it (this module's Units): a rate's change (a pass-fail
    measure or judged dimension, a percent, how each trial ended) in ``points``; any other's in its figures' own unit,
    an ordinal judged dimension's in ``score``, a score change no point of a rate is. The one reading of a contrast's
    unit: the plan checks group a chart's contrasts and store its unit by it, and the one-unit rule compares it.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param contrast: One of its contrasts.
    :ptype contrast: Contrast
    :return: ``points``, ``score``, a money unit, ``ms``, ``s``, any other declared unit, or empty.
    :rtype: str
    """
    unit = figure_unit(report_input, contrast.measure, kind=contrast.kind)
    return "points" if quantity(unit, change=True) == quantity("points") else unit


def contrast_unit_of(report_input: ReportInput, key: str) -> str:
    """The unit a difference between two arms on a measure or judged dimension is in, as a contrast's is
    (:func:`contrast_unit`): ``points`` for a rate, else its figures' own unit.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure or judged dimension.
    :ptype key: str
    :return: ``points``, ``score``, a money unit, ``ms``, ``s``, any other declared unit, or empty.
    :rtype: str
    """
    unit = figure_unit(report_input, key)
    return "points" if quantity(unit, change=True) == quantity("points") else unit


def _contrast_units(query: ChartQuery, report_input: ReportInput) -> set[str]:
    """The units of the contrasts a chart of contrasts draws (:func:`contrast_unit`): each one with a change. The rows
    of every group share one axis, so they share one unit drawn."""
    return {
        contrast_unit(report_input, row.item)
        for row in query_rows(query, report_input)
        if isinstance(row.item, Contrast) and row.item.delta is not None
    }


def missing_note(missing: Sequence[str]) -> tuple[str, ...]:
    """The caption naming each comparison a chart asks for that the input does not hold; nothing for none.

    :param missing: Each missing comparison, in words.
    :ptype missing: Sequence[str]
    :return: The caption, or nothing.
    :rtype: tuple[str, ...]
    """
    return (MISSING_TEMPLATE.format(missing="; ".join(missing)),) if missing else ()


def _first_factor_place(report_input: ReportInput, arm: str) -> int | None:
    """A config's first factor's level by its place in that factor's levels; None where it ran none."""
    if not report_input.factors:
        return None
    first = report_input.factors[0]
    config = next((config for config in report_input.configs if config.arm == arm), None)
    level = config.levels.get(first.id) if config is not None else None
    ids = [one.id for one in first.levels]
    return ids.index(level) if level in ids else None


def factor_token(report_input: ReportInput, arm: str) -> str | None:
    """A config's colour (this module's config colour): the kit's series colour at its first factor's level.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm: The config's arm.
    :ptype arm: str
    :return: The kit colour token; None for an arm that ran no config, or an input with no factors.
    :rtype: str | None
    """
    place = _first_factor_place(report_input, arm)
    return series_token(place) if place is not None else None


def factor_legend(report_input: ReportInput, arms: Iterable[str]) -> list[LegendItem] | None:
    """A config chart's legend: the first factor's levels the drawn configs run, each in its colour, in order.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arms: The configs drawn, by arm.
    :ptype arms: Iterable[str]
    :return: The legend; None where no drawn arm ran a config.
    :rtype: list[LegendItem] | None
    """
    if not report_input.factors:
        return None
    shown = {_first_factor_place(report_input, arm) for arm in arms}
    first = report_input.factors[0]
    items = [
        LegendItem(label=level.name or level.id, color=series_token(place))
        for place, level in enumerate(first.levels)
        if place in shown
    ]
    return items or None


#: The most points a scatter labels every one of (this module's scatter labels).
_LABEL_ALL_UP_TO = 8


def point_labels(context: ChartContext, arm: str, name: str, count: int) -> tuple[str | None, str | None]:
    """A scatter point's label at 560 px and wider, and below (this module's scatter labels).

    :param context: The chart being resolved.
    :ptype context: ChartContext
    :param arm: The point's arm.
    :ptype arm: str
    :param name: What the point is called.
    :ptype name: str
    :param count: How many points the panel it sits in draws.
    :ptype count: int
    :return: ``(label, narrow_label)``: its name where the panel draws few points or it is stressed or recommended,
        else None; narrow, the arm's tag where it is labelled.
    :rtype: tuple[str | None, str | None]
    """
    stressed = arm in (context.emphasis, context.look.recommended)
    if count > _LABEL_ALL_UP_TO and not stressed:
        return None, None
    tag = context.look.arm(arm).tag if arm in context.look.arms else None
    return name, tag


def count_text(count: float) -> str:
    """A count as a chart's words print it: the figure keys' whole-number spelling, thousands separated
    (``1,234``), so a count reads alike on a chart and in the prose beside it.

    :param count: The count.
    :ptype count: float
    :return: The text.
    :rtype: str
    """
    return format_figure(Figure(path="count", kind="number", value=float(count), scale="count"), ("int",))
