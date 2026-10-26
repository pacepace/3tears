"""One chart's rules, apart from any plan: what it can read, what code completes it with, and what refuses it.

A chart is a block of evidence: a kind (``forest``, ``bars``, a ``table``, …) over a query of the report input
(:class:`~threetears.evals.kernel.report_plan.ChartQuery`). A planned page draws the charts its plan stores, and an
analyst draws one to see what a claim's evidence would show (``block_render``). Both go through this module, so a
chart the plan checks accept is one an analyst sees drawn, and a chart refused one way is refused the other:

- :data:`BLOCK_SOURCES`: the sources each kind can draw (GENERATOR §5.1);
- :func:`complete_chart`: a chart as chosen to the chart as stored, with the reference key, interval level, unit and
  value axis code reads off the input (below);
- :func:`chart_faults` and :func:`small_multiple_faults`: what refuses one chart on its input (below);
- :func:`block_look`: the look a chart is drawn with when no plan orders the arms or recommends one.

The plan completion (:func:`~threetears.evals.analysis.report_plan_facts.complete_plan`) and the plan checks
(:func:`~threetears.evals.analysis.report_plan_checks.validate_plan`) call these for every chart a plan holds; what a
whole plan adds (a summary chart showing the options table's rows, the spine) stays theirs.

**A chart's reference line** sits at a line the input declares, and code fills its figure key:

- a **margin** at the chart's aspect's margin (``aspect.margin`` for the one asked aspect, else
  ``aspects.<id>.margin``);
- a **floor** or a **budget** at ``criteria.<id>.threshold``, the threshold of the one criterion that reads the chart's
  one measure (the measures its query names, else the ones its rows read): a floor's criterion is higher-is-better and
  a budget's lower-is-better, as the kit draws them (pass at or above a floor, at or below a budget); its statistic is
  one the chart draws (range bars draw p50 to p95, every other kind the mean); and it reads where the chart does, over
  the whole comparison or each task, or within the one task the chart keeps, or each cell on a chart of cells.

No such criterion, or several, is a fault at completion, naming why, and so is any other kind of line.

What the chart reads is :mod:`~threetears.evals.analysis.report_rows`' selection (:func:`query_aspect`,
:func:`query_rows`), the one the checks and the page read too.

**A chart's interval level** is the one its rows carry: a reading's confidence, a contrast's family level
(:func:`contrast_level`), a measure's or judged dimension's declared level (a stratum's with "uncorrected"). A
diverging chart over judged figures or strata whose figures carry a standard error and no interval is drawn at the
descriptive level, "95% uncorrected" (:data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL`): a bar is separated
from zero where its change exceeds 1.96 times the two arms' standard errors combined (GENERATOR §5.2).

**A chart's axis** is computed only where code holds a rule: a forest of readings spans every interval it draws, zero
and the margin line, one point beyond each end. Every other chart leaves its domain empty, which lets the renderer fit
it. A chart that would mix interval levels (§5.2: one interval system per chart) is refused at its query. Which
measures a chart draws, and whether it can draw them in their units, is the page's own rule and the only unit rule
here (:func:`~threetears.evals.analysis.report.plan_page.charting.unit_refusal`, §5.2: one unit per axis): a chart of
strata or tasks naming no measure where its slices hold several (a heatmap reads every one), or measures or contrasts
in more than one unit on one axis, is refused here by the same function the page refuses it by; a kind that draws each
measure in its own panel (a heatmap's matrix, a dot strip's strip, bars in panels by slice or measure) may read
several units, and its stored unit is then empty.

**What refuses a chart** (:func:`chart_faults`): its kind cannot draw its source; its query names an arm, aspect,
measure or stratum the input does not hold, or selects no row; a forest selects a row with no interval; its reference
key does not resolve; it stresses something that is no arm, stratum or task (or an option the plan's table lists);
and a chart drawn as each arm less the control (diverging bars over measures, judged figures, strata or tasks; a
difference heatmap naming fewer than two arms, which otherwise draws its second arm less its first) on an input that
names no control. :func:`small_multiple_faults` refuses panels on a kind the kit does not panel (its chart model
carries no ``panels``).

Every fault is ``(field path, why)``, the path under the one the caller names the chart by.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, get_args

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.report.charts import KIT_CHARTS
from threetears.evals.analysis.report.figure_keys import (
    DIRECTION_WORDS,
    format_figure,
    parse_key,
    reader_names,
    resolve,
)
from threetears.evals.analysis.report.plan_page.charting import contrast_unit, drawn_keys, unit_refusal
from threetears.evals.analysis.report.plan_page.look import PageLook, page_look
from threetears.evals.analysis.report_decision import control_arm, surviving_options
from threetears.evals.analysis.report_input import Cell, Contrast, Criterion, Reading, ReportInput, Stat, Stratum, Task
from threetears.evals.analysis.report_margins import criterion_place
from threetears.evals.analysis.report_rows import SourceRow, query_aspect, query_rows
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.kernel.report_plan import ChartQuery, ChartReference, ChartSpec, PlannedChart

if TYPE_CHECKING:  # an arm's facts are read by type only: the completion step that defines them imports this module.
    from threetears.evals.analysis.report_plan_facts import FactArm

__all__ = [
    "BLOCK_SOURCES",
    "INTERVAL_KINDS",
    "LINE_KINDS",
    "PANELLED_KINDS",
    "bare_key_faults",
    "block_look",
    "chart_faults",
    "complete_chart",
    "contrast_level",
    "line_statistics",
    "small_multiple_faults",
]

#: One fault: the field path it sits at and why it is refused.
_Fault = tuple[str, str]

#: Every source a chart query can name.
_SOURCES: tuple[str, ...] = get_args(ChartQuery.model_fields["source"].annotation)

#: The sources each chart kind can draw (GENERATOR §5.1), in the kit's order; a table draws any.
BLOCK_SOURCES: Mapping[str, frozenset[str]] = {
    "forest": frozenset({"readings", "contrasts"}),
    "bars": frozenset({"measures", "judged", "scoreboard", "strata", "tasks", "configs", "human_pref"}),
    "diverging_bars": frozenset(
        {"readings", "contrasts", "measures", "judged", "strata", "tasks", "configs", "human_pref"}
    ),
    "lines": frozenset({"measures", "judged", "strata", "tasks"}),
    "scatter_frontier": frozenset({"measures", "judged", "scoreboard", "configs", "tasks"}),
    "gate_scatter": frozenset({"measures", "judged", "configs", "tasks"}),
    "dot_interval": frozenset({"readings", "contrasts", "measures", "judged", "floors", "tasks", "strata"}),
    "range_bars": frozenset({"measures", "tasks"}),
    "gate_matrix": frozenset({"tasks", "readings", "floors", "measures", "configs"}),
    "lattice": frozenset({"configs"}),
    "heatmap": frozenset({"cells", "strata", "tasks"}),
    "diff_heatmap": frozenset({"cells", "strata", "tasks"}),
    "confusion": frozenset({"confusion"}),
    "examples": frozenset({"examples"}),
    "table": frozenset(_SOURCES),
}

#: The kinds that draw an interval on every row, so every row they select must carry one.
INTERVAL_KINDS = frozenset({"forest"})

#: The kinds the kit draws as small multiples: those whose chart model carries ``panels``
#: (:data:`~threetears.evals.analysis.report.charts.KIT_CHARTS`). A table is no kit chart, and panels nothing.
PANELLED_KINDS = frozenset(kind for kind, model in KIT_CHARTS.items() if "panels" in model.model_fields)

#: The direction of the criterion a floor or a budget line sits at: the kit passes a floor at or above, a budget at
#: or below.
_LINE_DIRECTIONS = {"floor": "higher_is_better", "budget": "lower_is_better"}

#: The reference lines code can key: a margin, and a line at a criterion (this module's reference lines).
LINE_KINDS: tuple[str, ...] = ("margin", *_LINE_DIRECTIONS)

#: The statistics a kind draws a measure at, where they are not the mean: range bars run from p50 to p95.
_KIND_STATISTICS: dict[str, frozenset[str]] = {"range_bars": frozenset({"p50", "p95"})}
_MEAN = frozenset({"mean"})

#: Sources a query names no measure of.
_NO_MEASURES = frozenset({"readings", "examples", "human_pref"})
#: A scoreboard's figures, which a scoreboard chart's measures name.
_SCOREBOARD_FIGURES = frozenset(
    {"ran", "usable", "passed", "cases", "lost", "failed", "reply_cost_usd", "test_cost_usd"}
)

#: The sources a diverging chart draws as each arm less the control, computed from point figures.
_LESS_THE_CONTROL = frozenset({"measures", "judged", "strata", "tasks"})
#: The arms a difference heatmap names to draw its second less its first, needing no control.
_COMPARED_PAIR = 2

#: How a fault opens where the page cannot draw a chart in the units its query selects.
_UNDRAWABLE = "cannot be drawn as it asks:"

#: The sources a diverging chart may draw at the descriptive level, and the word that labels it.
_DESCRIPTIVE_SOURCES = frozenset({"judged", "strata"})
_UNCORRECTED = "uncorrected"
#: The plain table's kind, which lays out every figure its rows hold.
_TABLE = "table"


def line_statistics(kind: str) -> frozenset[str]:
    """The statistics a chart of ``kind`` draws a measure at, which a floor or budget line's criterion must read.

    :param kind: The chart's kind.
    :ptype kind: str
    :return: ``p50`` and ``p95`` for range bars; the mean for every other kind.
    :rtype: frozenset[str]
    """
    return _KIND_STATISTICS.get(kind, _MEAN)


def contrast_level(report_input: ReportInput, contrast: Contrast) -> str:
    """The interval level a contrast is read at: its own, else its aspect's family's, else the one family's; empty
    for none. A chart of contrasts carries one (GENERATOR §5.2).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param contrast: One of its contrasts.
    :ptype contrast: Contrast
    :return: The level, as the input writes it.
    :rtype: str
    """
    families = report_input.contrast_families
    by_aspect = {family.aspect: family.level for family in families}
    only = families[0].level if len(families) == 1 else None
    return contrast.level or by_aspect.get(contrast.aspect) or only or ""


def block_look(report_input: ReportInput, arms: Sequence[FactArm]) -> PageLook:
    """The look one chart is drawn with when no plan stands behind it: each arm's tag, name, colour and mark as code's
    facts give them, the arms in the input's order, the survivors, the reader names, and no arm recommended.

    It is :func:`~threetears.evals.analysis.report.plan_page.look.page_look`, the planned page's own, with nothing a
    plan chooses: a chart whose query lists its arms draws them in that order on both.

    :param report_input: The report input the chart reads.
    :ptype report_input: ReportInput
    :param arms: Each arm as code's facts give it
        (:func:`~threetears.evals.analysis.report_plan_facts.fact_arms`).
    :ptype arms: Sequence[FactArm]
    :return: The look.
    :rtype: PageLook
    """
    return page_look(
        arms,
        {arm.id: arm.color_slot for arm in arms},
        order=(),
        survivors=surviving_options(report_input),
        recommended=None,
        names={str(kind): found for kind, found in reader_names(report_input).items()},
    )


# --- completing a chart ----------------------------------------------------------------------------------------


def complete_chart(chart: PlannedChart, report_input: ReportInput, path: str) -> tuple[ChartSpec, list[_Fault]]:
    """A chart as chosen, completed with its reference key, interval level, unit and value axis (this module's
    account), and every part code could not complete.

    :param chart: The chart as the planner or the analyst chose it.
    :ptype chart: PlannedChart
    :param report_input: The report input it reads.
    :ptype report_input: ReportInput
    :param path: The field path the chart sits at; each fault's path is under it.
    :ptype path: str
    :return: The chart as stored (a reference line code could not key is left out), and each fault as
        ``(field path, why)``: its reference lines', then its level's, then its unit's.
    :rtype: tuple[ChartSpec, list[tuple[str, str]]]
    """
    faults: list[_Fault] = []
    references = list(_references(chart, report_input, path, faults))
    level = _level(chart, report_input, path, faults)
    unit = _unit(chart, report_input, path, faults)
    spec = ChartSpec(
        kind=chart.kind,
        title=chart.title,
        query=chart.query,
        reference=references,
        emphasis=chart.emphasis,
        level=level,
        unit=unit,
        domain=_domain(chart, report_input, references),
    )
    return spec, faults


def _resolves(key: str, report_input: ReportInput) -> bool:
    try:
        resolve(key, report_input)
    except UnresolvableReference:
        return False
    return True


def _first_resolving(keys: Iterable[str], report_input: ReportInput) -> str | None:
    return next((key for key in keys if _resolves(key, report_input)), None)


def _references(
    chart: PlannedChart, report_input: ReportInput, path: str, faults: list[_Fault]
) -> Iterator[ChartReference]:
    """Each reference line with the figure key it sits at (this module's reference lines)."""
    for index, reference in enumerate(chart.reference):
        where = f"{path}.reference[{index}].kind"
        if reference.kind == "margin":
            key = _margin_key(chart, report_input, where, faults)
        elif reference.kind in _LINE_DIRECTIONS:
            key = _criterion_key(chart, reference.kind, report_input, where, faults)
        else:
            faults.append(
                (
                    where,
                    f"is {reference.kind!r}, and the report input holds no figure a {reference.kind} line sits at; "
                    "a margin, floor or budget line is what it can draw",
                )
            )
            continue
        if key is not None:
            yield ChartReference(kind=reference.kind, key=key)


def _margin_key(chart: PlannedChart, report_input: ReportInput, where: str, faults: list[_Fault]) -> str | None:
    """A margin line's key: its chart's aspect's margin."""
    aspect = query_aspect(chart.query.aspect, report_input)
    asked = query_aspect("", report_input)
    candidates = ["aspect.margin"] if aspect is not None and aspect is asked else []
    if aspect is not None:
        candidates.append(f"aspects.{aspect.id}.margin")
    key = _first_resolving(candidates, report_input)
    if key is None:
        named = f"aspect {aspect.id!r}" if aspect is not None else "the chart, which names no single aspect,"
        faults.append((where, f"is 'margin', and {named} declares no margin to draw"))
    return key


def _chart_measures(chart: PlannedChart, report_input: ReportInput) -> list[str]:
    """The measures a chart reads: those its query names, else those its rows read (a figure's, a contrast's, a
    floor's or an axis's key; a slice's or cell's figures for the arms it keeps), each once, in order."""
    if chart.query.measures:
        return list(dict.fromkeys(chart.query.measures))
    arms = set(chart.query.arms) or {arm.id for arm in report_input.arms}
    found: list[str] = []
    for row in query_rows(chart.query, report_input):
        if row.key:
            found.append(row.key)
        elif isinstance(row.item, Stratum | Task | Cell):
            found += [name for arm, figures in row.item.per_arm.items() if arm in arms for name in figures]
    return list(dict.fromkeys(found))


def _reads_where_the_chart_does(criterion: Criterion, chart: PlannedChart, report_input: ReportInput) -> bool:
    """Whether a criterion reads where the chart draws: over the whole comparison or each task, within the one task
    the chart keeps, or each cell on a chart of cells."""
    place = criterion_place(criterion)
    if place == "overall" or (place == "task" and criterion.task is None):
        return True
    if place == "cell":
        return chart.query.source == "cells"
    source = chart.query.source
    kept = [row.label for row in query_rows(chart.query, report_input)] if source == "tasks" else chart.query.strata
    return list(kept) == [criterion.task]


def _criterion_key(
    chart: PlannedChart, kind: str, report_input: ReportInput, where: str, faults: list[_Fault]
) -> str | None:
    """A floor or budget line's key: the threshold of the one criterion that reads the chart's one measure (this
    module's reference lines)."""
    measures = _chart_measures(chart, report_input)
    if len(measures) != 1:
        held = f"the measures {measures}" if measures else "no measure"
        faults.append(
            (
                where,
                f"is {kind!r}, and the chart reads {held}: a {kind} line sits at the criterion that reads the "
                "chart's one measure, so name that one measure in the query, or draw no line",
            )
        )
        return None
    (measure,) = measures
    drawn = line_statistics(chart.kind)
    direction = _LINE_DIRECTIONS[kind]
    found = [
        criterion
        for criterion in report_input.criteria
        if criterion.measure == measure
        and criterion.direction == direction
        and criterion.statistic in drawn
        and _reads_where_the_chart_does(criterion, chart, report_input)
    ]
    if len(found) == 1:
        return f"criteria.{found[0].id}.threshold"
    said = DIRECTION_WORDS[direction]
    if not found:
        faults.append(
            (
                where,
                f"is {kind!r}, and no criterion reads {measure!r} as a {kind} does ({said}, at a statistic a "
                f"{chart.kind!r} draws, {sorted(drawn)}, where the chart reads): draw no line, or the line a "
                "declared criterion gives",
            )
        )
        return None
    faults.append(
        (
            where,
            f"is {kind!r}, and the criteria {[criterion.id for criterion in found]} all read {measure!r} where the "
            "chart does: a line sits at one criterion's threshold, so draw no line, or narrow the chart to where only "
            "one reads (the one task a task's criterion reads)",
        )
    )
    return None


def _percent(level: float) -> str:
    """A confidence level as the chart prints it: a share (0.95) or a percent (95) as "95%"."""
    return format_number(level * 100 if level <= 1 else level) + "%"


def _one(path: str, what: str, values: Iterable[str], faults: list[_Fault]) -> str:
    """The one value a chart's rows agree on, empty for none; several are refused (§5.2: one per chart)."""
    found = list(dict.fromkeys(value for value in values if value))
    if len(found) > 1:
        faults.append((f"{path}.query", f"selects rows on {found}, and a chart carries one {what} (GENERATOR §5.2)"))
    return found[0] if found else ""


def _selected_contrasts(chart: PlannedChart, report_input: ReportInput) -> list[tuple[str, str]]:
    """The contrasts a chart of contrasts reads: each one's unit and interval level (:func:`contrast_unit`,
    :func:`contrast_level`)."""
    contrasts = [row.item for row in query_rows(chart.query, report_input) if isinstance(row.item, Contrast)]
    return [(contrast_unit(report_input, contrast), contrast_level(report_input, contrast)) for contrast in contrasts]


def _selected_readings(chart: PlannedChart, report_input: ReportInput) -> list[Reading]:
    """The readings a chart of readings reads."""
    return [row.item for row in query_rows(chart.query, report_input) if isinstance(row.item, Reading)]


def _figure_levels(chart: PlannedChart, report_input: ReportInput) -> list[str]:
    measures = set(chart.query.measures)
    levels = [m.level for key, m in report_input.measures.items() if not measures or key in measures]
    levels += [j.level for key, j in report_input.judged.items() if not measures or key in measures]
    return [level for level in levels if level]


def _level(chart: PlannedChart, report_input: ReportInput, path: str, faults: list[_Fault]) -> str:
    """The interval level printed on the chart, from the rows it reads; empty where they carry none."""
    source = chart.query.source
    if source == "readings":
        readings = _selected_readings(chart, report_input)
        levels = [_percent(r.gain_confidence) for r in readings if r.gain_confidence is not None]
        return _one(path, "interval level", levels, faults)
    if source == "contrasts":
        return _one(path, "interval level", (level for _, level in _selected_contrasts(chart, report_input)), faults)
    if source in ("measures", "judged"):
        table = report_input.measures if source == "measures" else report_input.judged
        measures = set(chart.query.measures)
        levels = [figures.level or "" for key, figures in table.items() if not measures or key in measures]
        level = _one(path, "interval level", levels, faults)
        return level or _descriptive_level(chart, report_input)
    if source == "strata":
        level = _one(path, "interval level", _figure_levels(chart, report_input), faults)
        return f"{level} {_UNCORRECTED}" if level else _descriptive_level(chart, report_input)
    if source in ("tasks", "cells", "configs"):
        # A task's or cell's figure carries its own interval, at its measure's declared level, uncorrected over the
        # slices as a stratum's is; a config's figure is an arm's, at the level a measure's is. Printed only where a
        # figure the chart reads carries one.
        held = _interval_keys(chart, report_input)
        declared = {key: m.level or "" for key, m in report_input.measures.items()}
        declared |= {key: j.level or "" for key, j in report_input.judged.items()}
        level = _one(path, "interval level", (declared.get(key, "") for key in held), faults)
        if not level or source == "configs":
            return level
        return f"{level} {_UNCORRECTED}"
    return ""


def _interval_keys(chart: PlannedChart, report_input: ReportInput) -> list[str]:
    """The measures and judged dimensions a chart of tasks, cells or configs shows whose figures there carry an
    interval (a ``ci`` or a half-width), each once, in order: a plain table shows every figure its rows hold (those
    its query names, where it names any), any other kind the keys it draws (:func:`drawn_keys`)."""
    shown: set[str] | None = None if chart.kind == _TABLE else set(drawn_keys(chart.kind, chart.query, report_input))
    if chart.kind == _TABLE and chart.query.measures:
        shown = set(chart.query.measures)
    arms = set(chart.query.arms) or {arm.id for arm in report_input.arms}
    held: list[str] = []
    for row in query_rows(chart.query, report_input):
        if isinstance(row.item, Task | Cell):
            figures = [(key, stat) for arm, on in row.item.per_arm.items() if arm in arms for key, stat in on.items()]
        elif isinstance(row.item, Stat):
            figures = [(row.key, row.item)]
        else:
            continue
        held += [
            key
            for key, stat in figures
            if (shown is None or key in shown) and (stat.ci is not None or stat.half_width is not None)
        ]
    return list(dict.fromkeys(held))


def _descriptive_level(chart: PlannedChart, report_input: ReportInput) -> str:
    """The descriptive level a diverging chart over judged figures or strata is drawn at where its figures carry a
    standard error and no interval: a bar is separated from zero when its change exceeds the interval level's normal
    multiple of the two arms' standard errors combined (GENERATOR §5.2: descriptive, uncorrected); empty otherwise."""
    if chart.kind != "diverging_bars" or chart.query.source not in _DESCRIPTIVE_SOURCES:
        return ""
    stats = _selected_stats(chart, report_input)
    if stats and all(stat.sem is not None and stat.ci is None and stat.half_width is None for stat in stats):
        return f"{_percent(INTERVAL_LEVEL)} {_UNCORRECTED}"
    return ""


def _selected_stats(chart: PlannedChart, report_input: ReportInput) -> list[Stat]:
    """Every figure a chart of judged figures or strata reads: a judged row's, or each kept arm's on each measure the
    query keeps in a stratum."""
    measures = set(chart.query.measures)
    arms = set(chart.query.arms) or {arm.id for arm in report_input.arms}
    found: list[Stat] = []
    for row in query_rows(chart.query, report_input):
        if isinstance(row.item, Stat):
            found.append(row.item)
        elif isinstance(row.item, Stratum):
            found += [
                stat
                for arm, figures in row.item.per_arm.items()
                if arm in arms
                for key, stat in figures.items()
                if not measures or key in measures
            ]
    return found


def _judged_unit(report_input: ReportInput, measures: Sequence[str]) -> list[str]:
    scales = [judged.scale for key, judged in report_input.judged.items() if not measures or key in measures]
    return ["percent" if scale == "pass_fail" else "score" for scale in scales]


def _unit(chart: PlannedChart, report_input: ReportInput, path: str, faults: list[_Fault]) -> str:
    """The unit the chart's rows are in: a change from the control in points, else the rows' own unit, empty where
    they are in several (each drawn in its own panel) or none. Whether the page can draw the chart in its units is the
    page's one rule, refused here by the same function and by no other (this module's account)."""
    query = chart.query
    source = query.source
    refusal = unit_refusal(chart.kind, query, report_input)
    if refusal:
        faults.append(
            (
                f"{path}.query",
                f"{_UNDRAWABLE} {refusal} (GENERATOR §5.2: one unit per axis, a panel each where the kind panels by measure)",
            )
        )
    if source in ("readings", "floors"):
        return "points"
    if source == "contrasts":
        return _only(unit for unit, _ in _selected_contrasts(chart, report_input))
    if source == "measures":
        return _only(
            m.unit or "" for key, m in report_input.measures.items() if not query.measures or key in query.measures
        )
    if source in ("judged", "strata"):
        if query.transform == "arm_minus_control":
            return "points"
        return _only(_judged_unit(report_input, query.measures))
    return ""


def _only(values: Iterable[str]) -> str:
    """The one value the rows agree on; empty for none, or for several."""
    found = list(dict.fromkeys(value for value in values if value))
    return found[0] if len(found) == 1 else ""


def _domain(chart: PlannedChart, report_input: ReportInput, references: Sequence[ChartReference]) -> list[float]:
    """A forest of readings spans its intervals, zero and its margin line, a point beyond each end; any other chart
    is left to the renderer (empty)."""
    if chart.query.source != "readings":
        return []
    aspect = query_aspect(chart.query.aspect, report_input)
    if aspect is None:
        return []
    ends = [end for r in _selected_readings(chart, report_input) if r.within_interval for end in r.within_interval]
    if not ends:
        return []
    ends.append(0.0)
    if any(reference.kind == "margin" for reference in references) and aspect.margin is not None:
        ends.append(-float(aspect.margin))
    return [float(math.floor(min(ends)) - 1), float(math.ceil(max(ends)) + 1)]


# --- refusing a chart ------------------------------------------------------------------------------------------


def chart_faults(
    path: str, chart: ChartSpec, report_input: ReportInput, *, options: Iterable[str] = ()
) -> Iterator[_Fault]:
    """What refuses one completed chart on its input (this module's account): its query's faults, then a chart drawn
    less a control the input does not name.

    :param path: The field path the chart sits at; each fault's path is under it.
    :ptype path: str
    :param chart: The chart, completed.
    :ptype chart: ChartSpec
    :param report_input: The report input it reads.
    :ptype report_input: ReportInput
    :param options: What else the chart may stress beside the input's arms, strata and tasks: the options a plan's
        table lists.
    :ptype options: Iterable[str]
    :return: Each fault as ``(field path, why)``.
    :rtype: Iterator[tuple[str, str]]
    """
    yield from _query_faults(path, chart, report_input, set(options))
    yield from _no_control_faults(path, chart, report_input)


def small_multiple_faults(path: str, chart: ChartSpec) -> Iterator[_Fault]:
    """A chart repeating its panel where the kit draws no small multiples of its kind.

    :param path: The field path the chart sits at.
    :ptype path: str
    :param chart: The chart.
    :ptype chart: ChartSpec
    :return: The fault, where there is one, at its ``query.small_multiples_by``.
    :rtype: Iterator[tuple[str, str]]
    """
    by = chart.query.small_multiples_by
    if by != "none" and chart.kind not in PANELLED_KINDS:
        yield (
            f"{path}.query.small_multiples_by",
            (
                f"is {by!r}, and the kit draws no small multiples of a {chart.kind!r}; it panels "
                f"{sorted(PANELLED_KINDS)}. Set it to 'none', or draw a kind that panels"
            ),
        )


def bare_key_faults(path: str, key: str, report_input: ReportInput) -> Iterator[_Fault]:
    """A figure key written without braces that does not parse, resolve on the input, or suit its filters.

    :param path: The field path the key sits at.
    :ptype path: str
    :param key: The key, its filters after ``|``.
    :ptype key: str
    :param report_input: The report input it reads.
    :ptype report_input: ReportInput
    :return: The fault, where there is one.
    :rtype: Iterator[tuple[str, str]]
    """
    try:
        parsed = parse_key(key)
        figure = resolve(parsed.path, report_input)
        if figure.kind != "arm":
            format_figure(figure, parsed.filters)
    except UnresolvableReference as refused:
        yield path, str(refused)


def _no_control_faults(path: str, chart: ChartSpec, report_input: ReportInput) -> Iterator[_Fault]:
    """A chart drawn as each arm less the control, on an input that names no control (this module's account)."""
    if control_arm(report_input) is not None:
        return
    if chart.kind == "diverging_bars" and chart.query.source in _LESS_THE_CONTROL:
        yield (
            f"{path}.kind",
            (
                f"is 'diverging_bars' over {chart.query.source!r}, which draws each arm less the control, and the "
                "report input names no control"
            ),
        )
    elif chart.kind == "diff_heatmap" and len(chart.query.arms) < _COMPARED_PAIR:
        yield (
            f"{path}.query.arms",
            (
                f"name {list(chart.query.arms)}, and a 'diff_heatmap' on an input with no control draws its second "
                "arm less its first: name the two arms it compares"
            ),
        )


def _query_faults(path: str, chart: ChartSpec, report_input: ReportInput, options: set[str]) -> Iterator[_Fault]:
    query = chart.query
    source = query.source
    if source not in BLOCK_SOURCES[chart.kind]:
        yield (
            f"{path}.kind",
            (f"is {chart.kind!r}, which cannot draw {source!r}; it draws {sorted(BLOCK_SOURCES[chart.kind])}"),
        )
    arms = [arm.id for arm in report_input.arms]
    if stray := [arm for arm in query.arms if arm not in arms]:
        yield f"{path}.query.arms", f"name {stray}, which are no arms; the arms are {arms}"
    aspects = [aspect.id for aspect in report_input.aspects]
    if query.aspect and query.aspect not in aspects:
        yield f"{path}.query.aspect", f"is {query.aspect!r}, which is no aspect; the aspects are {aspects}"
        return
    measures = _measure_vocabulary(source, report_input)
    if stray := [measure for measure in query.measures if measure not in measures]:
        held = f"it reads {sorted(measures)}" if measures else "it names none"
        yield f"{path}.query.measures", f"name {stray}, which the {source!r} source does not hold; {held}"
    strata = _strata_vocabulary(report_input)
    if stray := [stratum for stratum in query.strata if stratum not in strata]:
        yield f"{path}.query.strata", f"name {stray}, which are no strata or tasks of the report input"
    rows: list[SourceRow] = query_rows(query, report_input)
    if not rows:
        yield f"{path}.query", f"selects nothing from {source!r}: the chart would be empty"
    elif chart.kind in INTERVAL_KINDS and any(row.interval is None for row in rows):
        yield (
            f"{path}.kind",
            f"is {chart.kind!r}, and some row it selects carries no interval: a {chart.kind} draws intervals",
        )
    for index, reference in enumerate(chart.reference):
        yield from bare_key_faults(f"{path}.reference[{index}].key", reference.key, report_input)
    if chart.emphasis and chart.emphasis not in _emphasizable(options, report_input):
        yield f"{path}.emphasis", f"is {chart.emphasis!r}, which is no arm, option, stratum or task to stress"


def _measure_vocabulary(source: str, report_input: ReportInput) -> set[str]:
    if source in _NO_MEASURES:
        return set()
    if source == "contrasts":
        return {contrast.measure for contrast in report_input.contrasts}
    if source == "measures":
        return set(report_input.measures)
    if source == "judged":
        return set(report_input.judged)
    if source == "configs":
        return set(report_input.measures) | set(report_input.judged)
    if source == "scoreboard":
        return set(_SCOREBOARD_FIGURES)
    if source == "floors":
        return {floor.floor for floor in report_input.floors}
    if source == "confusion":
        return set(report_input.confusion)
    holders: Iterable[Mapping[str, Mapping[str, object]]]
    if source == "strata":
        holders = [stratum.per_arm for stratum in report_input.strata]
    elif source == "tasks":
        holders = [task.per_arm for task in report_input.tasks]
    else:
        holders = [cell.per_arm for cell in report_input.cells.values()]
    return {name for per_arm in holders for figures in per_arm.values() for name in figures}


def _strata_vocabulary(report_input: ReportInput) -> set[str]:
    names = {stratum.id for stratum in report_input.strata} | {task.id for task in report_input.tasks}
    names |= {example.stratum for example in report_input.examples if example.stratum}
    names |= set(report_input.cells)
    if report_input.human_pref is not None:
        names |= {row.id for row in report_input.human_pref.rows}
    return names


def _emphasizable(options: set[str], report_input: ReportInput) -> set[str]:
    return {arm.id for arm in report_input.arms} | options | _strata_vocabulary(report_input)
