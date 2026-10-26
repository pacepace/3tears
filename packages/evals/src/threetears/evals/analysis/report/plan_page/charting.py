"""What every chart resolver shares: the chart it is drawing, and the rules each kind fills its data by.

A resolver (one per kind, in a ``resolve_*`` module's ``RESOLVERS``) turns a stored chart's
:class:`~threetears.evals.kernel.report_plan.ChartSpec` into the kit's chart data, reading the rows its query
selects (:func:`~threetears.evals.analysis.report_rows.query_rows`, the one owner of that selection). It is handed a
:class:`ChartContext` and returns a :class:`Drawn`. The rules every kind fills by (the P4 design, §3) are here, once:

- **Units.** A figure's unit as the page reads it (:func:`figure_unit`): a measure's declared unit; a pass-fail
  rate's scale read from every figure the input holds for it (a share where each lies in 0 to 1, a percent
  otherwise), as the figure keys read it; an ordinal judged dimension a ``score``; a contrast of how each trial ended,
  the pass rate, a share. :func:`~threetears.evals.analysis.report.plan_page.look.quantity` says how it is drawn.
- **A figure's value** (:func:`stat_value`): its mean; else the statistic a criterion on the measure reads; else its
  p95, else its p50. **Its interval** (:func:`stat_interval`): its ``ci``, else its mean plus or minus its half-width.
  **Intervals are drawn only where the chart states a level** (``ChartSpec.level``): the kit refuses intervals
  without one.
- **Reference lines** (:func:`reference_line`): the stored reference's figure key read off the input, in display
  units: a margin as minus the margin on a change axis, a floor or budget at the criterion's threshold. A kind that
  draws its criterion's line where the plan stores none (a dot strip's floor, range bars' budget, a heatmap's cell
  floor) reads it by :func:`criterion_line`: the one criterion on the measure, at the statistic drawn, read where the
  chart reads it.
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

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock
from threetears.evals.analysis.report.charts import ChartReference, KitChart
from threetears.evals.analysis.report.figure_keys import Figure, NamedKind, resolve
from threetears.evals.analysis.report.plan_page.look import PageLook, Quantity, figure_text, quantity
from threetears.evals.analysis.report.plan_page.model import PLANNED_SECTION
from threetears.evals.analysis.report.plan_page.words import MISSING_TEMPLATE, REFERENCE_TEMPLATE
from threetears.evals.analysis.report_input import Cell, Criterion, ReportInput, Stat, Stratum, Task
from threetears.evals.analysis.report_rows import SourceRow, query_rows
from threetears.evals.analysis.style import measure_name
from threetears.evals.kernel.report_plan import ChartQuery, ChartSpec

__all__ = [
    "ChartContext",
    "CriterionWhere",
    "Drawn",
    "KindResolver",
    "chart_block",
    "criterion_line",
    "figure_stat",
    "figure_unit",
    "interval_words",
    "measure_words",
    "misses_line",
    "missing_note",
    "reference_line",
    "separated",
    "slice_keys",
    "slice_name",
    "spread",
    "stat_interval",
    "stat_value",
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


def figure_unit(report_input: ReportInput, key: str, *, kind: str | None = None) -> str:
    """A measure's or judged dimension's unit as the page reads it (this module's Units), wherever its figures sit.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure or judged dimension, by key.
    :ptype key: str
    :param kind: A contrast's kind (``measure``, ``judged``, ``trial``), where the key is a contrast's; None otherwise.
    :ptype kind: str | None
    :return: ``share``, ``percent``, ``score``, a money unit, ``ms``, ``s``, any other declared unit, or empty.
    :rtype: str
    """
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
    ended (a contrast of kind ``trial``) is the pass rate.

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
        return report_input.measures[key].label or measure_name(key)
    if key in report_input.judged:
        return report_input.judged[key].label or measure_name(key)
    return measure_name(key)


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
    if where == "overall":
        return criterion.scope == "overall" and criterion.task is None
    if where == "tasks":
        return criterion.scope == "each_task" and criterion.task is None
    if where == "cells":
        return criterion.scope == "each_cell"
    _, task = where
    return criterion.task == task or (criterion.scope == "each_task" and criterion.task is None)


def criterion_line(
    context: ChartContext, key: str, drawn: Quantity, *, where: CriterionWhere, statistic: str = "mean"
) -> ChartReference | None:
    """The line a declared criterion draws on a chart that stores none: the threshold of the one criterion that reads
    ``key`` at ``statistic`` where the chart reads it, in display units, a floor where higher is better and a budget
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
    lines = {
        (float(criterion.threshold), criterion.direction)
        for criterion in context.report_input.criteria
        if criterion.measure == key and criterion.statistic == statistic and _reads_there(criterion, where)
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
    if line.kind == "floor":
        return value < line.value
    return value > line.value if line.kind == "budget" else False


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
    found = list(dict.fromkeys(key for item in slices for figures in item.per_arm.values() for key in figures))
    if len(found) != 1:
        raise context.refuse(f"its query names no measure, and its slices hold {found}")
    return found


def missing_note(missing: Sequence[str]) -> tuple[str, ...]:
    """The caption naming each comparison a chart asks for that the input does not hold; nothing for none.

    :param missing: Each missing comparison, in words.
    :ptype missing: Sequence[str]
    :return: The caption, or nothing.
    :rtype: tuple[str, ...]
    """
    return (MISSING_TEMPLATE.format(missing="; ".join(missing)),) if missing else ()
