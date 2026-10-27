"""What a report plan's chart query selects from the report input: the one owner of that selection.

A chart's :class:`~threetears.evals.kernel.report_plan.ChartQuery` names a source and filters on it. The plan checks
(:mod:`~threetears.evals.analysis.report_plan_checks`) refuse a query that selects nothing, or a forest whose rows
carry no interval, and a page draws the rows the query selects. Both read the selection here, so what the checks
pass is what the page draws.

**The rows of each source**, every one a :class:`SourceRow` holding the input's own record:

==============  ================================================================  =========================  ========
source          a row is                                                          label                      interval
==============  ================================================================  =========================  ========
readings        a :class:`Reading` of the query's aspect (:func:`query_aspect`)   the arm                    yes
contrasts       a :class:`Contrast`                                               ``arm/measure``            yes
measures        an arm's :class:`Stat` on a measure                               ``measure/arm``            yes
judged          an arm's :class:`Stat` on a judged dimension                      ``dimension/arm``          yes
configs         both of the above, for the arms that ran a config                 ``key/arm``                yes
scoreboard      an arm's :class:`ScoreboardRow`                                   the arm                    no
floors          a :class:`FloorRead`                                              ``floor/arm``              yes
strata, tasks   a :class:`Stratum` or :class:`Task` slice                         its id                     no
examples        an :class:`Example`                                               its trial, else its case   no
cells           a target-matrix :class:`Cell`                                     its id                     no
confusion       a :class:`ConfusionTable`, one per axis                           the axis's key             no
human_pref      a :class:`PreferenceRow` slice                                    its id                     yes
prices          a :class:`PriceStep` along the price frontier on a gain axis      ``from→to``                yes
                (each the query's ``measures`` keeps)
bar_reads       a :class:`BarRead`: an arm against an absolute bar                ``measure/arm``            yes
compare         an ``arms_compare`` row: the query's first arm less its second    its place                  yes
                on its one measure, overall or in one slice
settle          a ``settle_cases`` row: an arm against a criterion reading it     ``arm/criterion/place``    no
levers          a ``lever_effects`` effect or interaction on the one measure      the move                   yes
==============  ================================================================  =========================  ========

A figure row reads the query's aspect from the measure's ``by_aspect`` where it names one the measure holds, else its
``per_arm`` (the lead aspect's). A scoreboard reads the named aspect's own scoreboard where it has one. A row has an
interval where the input carries one: a reading's ``within_interval``, a contrast's or a floor's ``interval``, a
figure's ``ci``, a preference slice's half-width about its share, a price step's ends off its gain's interval (paired
where every step states a paired one, else unpaired). A price step whose gain's interval reaches zero has no high end:
the evidence allows any price above its low end, so the row carries no two-ended interval and its step holds the low
end it does state (:attr:`PriceStep.open_from`), which a chart draws as a whisker open above and a table writes as
having no upper bound, never as a bare price.

**The rows an analysis computes** (``compare``, ``settle``, ``levers``) are its function's own rows, never computed
again here: :func:`~threetears.evals.analysis.analyst.compare.arms_compare`,
:func:`~threetears.evals.analysis.analyst.settle.settle_cases` and
:func:`~threetears.evals.analysis.analyst.levers.lever_effects`, each figure in them a cited figure key. A ``compare``
query names two arms, in order (every row the first less the second), and one measure; ``strata`` keeps places
(``overall``, a task, a stratum or a cell by id). A ``settle`` query keeps the arms each settled against every
line ``settle_cases`` reads it against (``arms``, empty for all): each criterion reading it, else the margins and
floors its readings are read against; ``measures`` keeps the criteria on those measures (a margin's or floor's row,
on no one measure, only where it names none), and ``strata`` the places. A ``levers`` query names one measure and at most one place (empty for overall); ``arms`` keeps
the effects and interactions every arm of which it keeps. A call the function refuses selects no row, and the chart
checks name the function's reason. A ``bar_reads`` row is an arm against one absolute bar, as the pipeline read it.

**A configs query on an input holding no configs** reads the grid the arms' own settings make
(:func:`config_grid`), where they make one: the settings the arms vary that every arm states, two or more, crossed
fully (every combination of their values held by exactly one arm). Each is a factor (the input's declared factor of
that id, its levels matched by id or name, else one named for the setting), and each arm the config of its values.
So a lattice of a factorial campaign draws whether or not the snapshot recorded its configs.

**A price step** is one move along the frontier of the options best at some price per point
(:func:`~threetears.evals.analysis.report.figure_keys.price_frontier`), on each gain axis a ``prices`` query keeps (a
block draws one, named in ``measures``): its price per point a month at the declared volume, else per 1,000 of what the
cost counts, each figure read through the key that states it. ``arms`` keeps the steps both of whose options it keeps.

**The filters.** An empty filter keeps everything (:class:`~threetears.evals.kernel.report_plan.ChartQuery`).
``arms`` keeps the rows of those arms, where a row is of one arm (an example with no arm is kept; a preference slice
is of the arm rated against the control); ``measures`` the contrasts, figures, floors and confusion axes on those
keys; ``strata`` the slices, examples and cells by id; ``aspect`` the contrasts, floors and examples on it (an example
naming no aspect is kept), and the aspect readings, figures and a scoreboard read. Rows come in the input's order, whatever order the query lists its arms in: the
order a chart shows them in is the chart's.

**The limit.** ``limit`` is the most rows a strata or examples query keeps, 0 for no limit. Strata keep the largest
by ``n_cases`` (a stratum without a count as the smallest), largest first, ties in the input's order; examples keep
the first, in the input's order. A limit on any other source keeps every row.
"""

from __future__ import annotations

import math
import functools
import hashlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

from threetears.evals.analysis.analyst.compare import ArmsComparison, ComparisonRow, arms_compare
from threetears.evals.analysis.analyst.levers import EffectRow, InteractionRow, LeverEffects, lever_effects
from threetears.evals.analysis.analyst.settle import Settle, SettleRow, settle_cases
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import arm_settings, price_frontier, resolve
from threetears.evals.analysis.report_input import (
    Aspect,
    BarRead,
    Cell,
    Config,
    ConfusionTable,
    Contrast,
    Example,
    Factor,
    FloorRead,
    Judged,
    Level,
    Measure,
    PreferenceRow,
    Reading,
    ReportInput,
    ScoreboardRow,
    Stat,
    Stratum,
    Task,
)
from threetears.evals.analysis.style import humanize
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.report_plan import ChartQuery

__all__ = [
    "OVERALL",
    "PriceStep",
    "SourceRow",
    "config_grid",
    "lever_arms",
    "lever_place",
    "place_of",
    "query_aspect",
    "query_rows",
]

#: How a query's ``strata`` names an analysis row's overall place, beside the slices' ids.
OVERALL = "overall"

#: The most rows an analysis function is asked for at once: every row it has.
_EVERY_ROW = 1_000_000


@dataclass(frozen=True)
class PriceStep:
    """One step along a price frontier: the move from one frontier option to the next dearer, at its price per point.

    :ivar first: The cheaper option's id.
    :ivar second: The dearer option's id.
    :ivar gain_key: The gain axis, by key.
    :ivar key: The key that states the price drawn: ``monthly_per_point`` at the declared volume, else ``per_point``.
    :ivar value: The price per point the key reads.
    :ivar monthly: Whether it is a month's price at the declared volume (else per 1,000 of what the cost counts).
    :ivar low_key: The key of the interval's low end; empty where the step states no interval.
    :ivar high_key: The key of its high end (one refused as unbounded where the interval is open above); empty where
        the step states no interval.
    :ivar paired: Whether the ends are read off the paired interval (a contrast's), else off the unpaired one.
    :ivar open_from: The low end of an interval open above: the gain's interval reaches zero, so no price bounds the
        step above and the row carries no two-ended interval; None where the interval is bounded or unstated.
    """

    first: str
    second: str
    gain_key: str
    key: str
    value: float
    monthly: bool
    low_key: str = ""
    high_key: str = ""
    paired: bool = False
    open_from: float | None = None


@dataclass(frozen=True)
class SourceRow:
    """One row a chart query selects (this module's table of what each source's rows are).

    :ivar label: The row by a label (this module's table).
    :ivar item: The input's own record the row is.
    :ivar arm: The arm the row is of; empty for a row of no one arm (a slice, a cell, a confusion axis, an example
        no arm ran).
    :ivar key: The measure, judged dimension, floor or confusion axis the row reads; empty where it reads none.
    :ivar interval: The interval the row carries, low then high; None where it carries none.
    """

    label: str
    item: (
        Reading
        | Contrast
        | Stat
        | ScoreboardRow
        | FloorRead
        | Stratum
        | Task
        | Example
        | Cell
        | ConfusionTable
        | PreferenceRow
        | PriceStep
        | BarRead
        | ComparisonRow
        | SettleRow
        | EffectRow
        | InteractionRow
    )
    arm: str = ""
    key: str = ""
    interval: tuple[float, float] | None = None


def query_aspect(aspect: str, report_input: ReportInput) -> Aspect | None:
    """The aspect a chart query reads, given its ``aspect`` field: the one it names, or with none named the one
    asked aspect (the only one asked, else the only one).

    :param aspect: The query's ``aspect`` field, empty where it names none.
    :ptype aspect: str
    :param report_input: The report input the query reads.
    :ptype report_input: ReportInput
    :return: The aspect; None where the name matches no aspect, or none is named and no one aspect is asked.
    :rtype: Aspect | None
    """
    if aspect:
        return next((held for held in report_input.aspects if held.id == aspect), None)
    asked = [held for held in report_input.aspects if held.asked]
    if len(asked) == 1:
        return asked[0]
    return report_input.aspects[0] if not asked and len(report_input.aspects) == 1 else None


def query_rows(query: ChartQuery, report_input: ReportInput) -> list[SourceRow]:
    """Every row the query selects from its source, filtered and limited as this module says, in the input's order
    (strata largest first under a limit).

    :param query: The chart's query.
    :ptype query: ChartQuery
    :param report_input: The report input it reads.
    :ptype report_input: ReportInput
    :return: The rows; empty where the query selects nothing.
    :rtype: list[SourceRow]
    """
    return list(_SELECTORS[query.source](query, report_input))


def _arms(query: ChartQuery, report_input: ReportInput) -> set[str]:
    return set(query.arms) or {arm.id for arm in report_input.arms}


def _interval(pair: tuple[float, float] | None) -> tuple[float, float] | None:
    return (float(pair[0]), float(pair[1])) if pair is not None else None


def _readings(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    aspect = query_aspect(query.aspect, report_input)
    arms = _arms(query, report_input)
    for reading in aspect.readings if aspect is not None else []:
        if reading.arm in arms:
            yield SourceRow(reading.arm, reading, arm=reading.arm, interval=_interval(reading.within_interval))


def _contrasts(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    arms = _arms(query, report_input)
    measures = set(query.measures)
    for contrast in report_input.contrasts:
        if (
            contrast.arm in arms
            and (not measures or contrast.measure in measures)
            and (not query.aspect or contrast.aspect == query.aspect)
        ):
            yield SourceRow(
                f"{contrast.arm}/{contrast.measure}",
                contrast,
                arm=contrast.arm,
                key=contrast.measure,
                interval=_interval(contrast.interval),
            )


def _figures(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    """Each arm's figure on each measure or judged dimension the query keeps; a configs query keeps the arms that ran
    a config."""
    source = query.source
    tables: list[Mapping[str, Measure | Judged]] = []
    if source in ("measures", "configs"):
        tables.append(report_input.measures)
    if source in ("judged", "configs"):
        tables.append(report_input.judged)
    arms = _arms(query, report_input)
    if source == "configs":
        arms &= {config.arm for config in config_grid(report_input).configs}
    measures = set(query.measures)
    for table in tables:
        for key, figures in table.items():
            if measures and key not in measures:
                continue
            per_arm = figures.by_aspect.get(query.aspect, figures.per_arm) if query.aspect else figures.per_arm
            for arm, stat in per_arm.items():
                if arm in arms:
                    yield SourceRow(f"{key}/{arm}", stat, arm=arm, key=key, interval=_interval(stat.ci))


def _scoreboard(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    aspect = query_aspect(query.aspect, report_input) if query.aspect else None
    board = aspect.scoreboard if aspect is not None and aspect.scoreboard is not None else report_input.scoreboard
    arms = _arms(query, report_input)
    yield from (SourceRow(arm, row, arm=arm) for arm, row in board.items() if arm in arms)


def _floors(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    arms = _arms(query, report_input)
    measures = set(query.measures)
    for floor in report_input.floors:
        if (
            floor.arm in arms
            and (not query.aspect or floor.aspect == query.aspect)
            and (not measures or floor.floor in measures)
        ):
            yield SourceRow(
                f"{floor.floor}/{floor.arm}", floor, arm=floor.arm, key=floor.floor, interval=_interval(floor.interval)
            )


def _slices(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    """The strata or tasks the query keeps, each with a figure for some arm it keeps; strata to the limit."""
    arms = _arms(query, report_input)
    strata = set(query.strata)
    slices: list[Stratum] | list[Task] = report_input.strata if query.source == "strata" else report_input.tasks
    kept = [item for item in slices if (not strata or item.id in strata) and any(arm in arms for arm in item.per_arm)]
    if query.source == "strata" and query.limit:
        kept = sorted(kept, key=lambda item: -(item.n_cases or 0))[: query.limit]
    yield from (SourceRow(item.id, item) for item in kept)


def _examples(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    """The examples of the query's arms (or of no arm), strata and aspect (or of no aspect), the first ``limit``."""
    arms = _arms(query, report_input)
    strata = set(query.strata)
    kept = [
        example
        for example in report_input.examples
        if (example.arm is None or example.arm in arms)
        and (not strata or example.stratum in strata)
        and (not query.aspect or example.aspect in (None, query.aspect))
    ]
    for example in kept[: query.limit] if query.limit else kept:
        yield SourceRow(example.trial or example.case or "example", example, arm=example.arm or "")


def _cells(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    strata = set(query.strata)
    yield from (
        SourceRow(cell_id, cell) for cell_id, cell in report_input.cells.items() if not strata or cell_id in strata
    )


def _confusion(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    measures = set(query.measures)
    for axis, table in report_input.confusion.items():
        if not measures or axis in measures:
            yield SourceRow(axis, table, key=axis)


def _human_pref(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    """Each preference slice, a row of the arm rated against the control: kept where the query keeps that arm."""
    preference = report_input.human_pref
    if preference is None or preference.arm not in _arms(query, report_input):
        return
    for row in preference.rows:
        width = row.half_width
        interval = (row.prefers_control - width, row.prefers_control + width) if width is not None else None
        yield SourceRow(row.id, row, arm=preference.arm, interval=_interval(interval))


def _number_at(key: str, report_input: ReportInput) -> float | None:
    """The number a key reads, or None where it refuses: an end an interval leaves open, or one unstated."""
    try:
        value = resolve(key, report_input).value
    except UnresolvableReference:
        # NOSILENT: a price's end refused as unbounded or unstated is a row with no interval, which the step's keys
        # say; the price the row draws is read below and raises if it does not resolve.
        return None
    return float(value) if isinstance(value, int | float) else None


def _gain_keys(report_input: ReportInput) -> list[str]:
    """Every measure or judged dimension a price frontier can be read on: a judged dimension, or a measure with a
    direction."""
    directed = [key for key, measure in report_input.measures.items() if measure.higher_is_better is not None]
    return list(dict.fromkeys([*report_input.judged, *directed]))


def _prices(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    """Each step along the price frontier on each gain axis the query keeps (every one where it names none), kept
    where the query keeps both of its options; on each axis its interval paired where every kept step states a paired
    one (its low end, bounded above or open), else unpaired, else none."""
    keys = list(query.measures) or _gain_keys(report_input)
    for gain_key in keys:
        yield from _steps(query, report_input, gain_key)


def _steps(query: ChartQuery, report_input: ReportInput, gain_key: str) -> Iterator[SourceRow]:
    try:
        frontier = price_frontier(report_input, gain_key)
    except UnresolvableReference:
        # NOSILENT: a key naming no gain axis (no such measure, or one of no direction) selects no row; the chart
        # checks refuse a query naming a measure the source does not hold.
        return
    arms = _arms(query, report_input)
    monthly = report_input.meta.volume_per_month is not None
    field = "monthly_per_point" if monthly else "per_point"
    steps = [(first, second) for first, second in frontier.steps if first in arms and second in arms]

    def ends(first: str, second: str, suffix: str) -> tuple[str, str, float, float | None] | None:
        """The interval's keys and ends where its low end is stated: the high end None where the price is unbounded
        above (its low end bounds the gain's far end, so only its near end can reach zero); None where unstated."""
        low_key, high_key = (frontier.key(first, second, f"{field}_{end}{suffix}") for end in ("low", "high"))
        low = _number_at(low_key, report_input)
        return (low_key, high_key, low, _number_at(high_key, report_input)) if low is not None else None

    paired = bool(steps) and all(ends(first, second, "") is not None for first, second in steps)
    for first, second in steps:
        key = frontier.key(first, second, field)
        value = resolve(key, report_input).value
        assert isinstance(value, float)
        found = ends(first, second, "" if paired else "_unpaired")
        low_key, high_key, low, high = found if found is not None else ("", "", None, None)
        step = PriceStep(
            first=first,
            second=second,
            gain_key=gain_key,
            key=key,
            value=value,
            monthly=monthly,
            low_key=low_key,
            high_key=high_key,
            paired=paired and found is not None,
            open_from=low if high is None else None,
        )
        interval = (low, high) if low is not None and high is not None else None
        yield SourceRow(f"{first}→{second}", step, key=gain_key, interval=interval)


#: The fewest settings a grid of configs crosses.
_CROSSED = 2


def config_grid(report_input: ReportInput) -> ReportInput:
    """The input with the grid its configs make: its own where it holds configs, else the one its arms' settings make
    (this module's configs query on an input holding no configs); the input as it is where the settings every arm
    states and the arms vary are fewer than two, or do not cross fully (every combination of their values one arm).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The input, its ``factors`` and ``configs`` the grid's.
    :rtype: ReportInput
    """
    if report_input.configs:
        return report_input
    settings = arm_settings(report_input)
    names = list(dict.fromkeys(name for values in settings.values() for name in values))
    varying = [
        name
        for name in names
        if all(values.get(name) is not None for values in settings.values())
        and len({str(values[name]) for values in settings.values()}) > 1
    ]
    combinations = {tuple(str(values[name]) for name in varying) for values in settings.values()}
    sizes = [len({str(values[name]) for values in settings.values()}) for name in varying]
    if len(varying) < _CROSSED or len(combinations) != len(settings) or len(combinations) != math.prod(sizes):
        return report_input
    declared = {factor.id: factor for factor in report_input.factors}
    factors: list[Factor] = []
    ids: dict[str, dict[str, str]] = {}
    for name in varying:
        values = list(dict.fromkeys(str(values[name]) for values in settings.values()))
        own = declared.get(name)
        levels = own.levels if own is not None else []
        held = {spelled.casefold(): level.id for level in levels for spelled in (level.id, level.name or level.id)}
        if own is not None and all(value.casefold() in held for value in values):
            factors.append(own)
            ids[name] = {value: held[value.casefold()] for value in values}
        else:
            factors.append(
                Factor(
                    id=name, name=humanize("dimension", name), levels=[Level(id=value, name=value) for value in values]
                )
            )
            ids[name] = {value: value for value in values}
    configs = [
        Config(arm=arm, levels={name: ids[name][str(values[name])] for name in varying})
        for arm, values in settings.items()
    ]
    kept = [factor for factor in report_input.factors if factor.id not in ids]
    return report_input.model_copy(update={"factors": [*factors, *kept], "configs": configs})


def place_of(where: str, report_input: ReportInput) -> str:
    """An analysis row's place as a query's ``strata`` names it: :data:`OVERALL`, or the slice's id (a task's, a
    stratum's, or a cell's, which a separation key spells by its row and column).

    :param where: The place as a separation key spells it (``overall``, ``task.<task>``, ``strata.<stratum>``,
        ``cells.<row>.<column>``), or as a lever's place does (a task's or stratum's id, ``<row>.<column>``).
    :ptype where: str
    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The place.
    :rtype: str
    """
    kind, _dot, named = where.partition(".")
    spelled = {f"{cell.row}.{cell.column}": cell_id for cell_id, cell in report_input.cells.items()}
    if kind in ("task", "strata"):
        return named
    if kind == "cells":
        return spelled.get(named, named)
    return spelled.get(where, where)


def _kept_place(where: str, query: ChartQuery, report_input: ReportInput) -> bool:
    return not query.strata or place_of(where, report_input) in query.strata


def _bar_reads(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    arms = _arms(query, report_input)
    measures = set(query.measures)
    for read in report_input.bars:
        if (
            read.arm in arms
            and (not measures or read.measure in measures)
            and (not query.aspect or read.aspect == query.aspect)
        ):
            yield SourceRow(
                f"{read.measure}/{read.arm}", read, arm=read.arm, key=read.measure, interval=_interval(read.interval)
            )


def _cited_interval(figure: object) -> tuple[float, float] | None:
    value = getattr(figure, "value", None)
    return (float(value[0]), float(value[1])) if isinstance(value, tuple) else None


@dataclass(frozen=True, eq=False)
class _ByContent:
    """A report input read by its content: equal to, and hashed as, any input whose JSON is the same, so a memo of a
    pure analysis keyed by it answers an input from the result its content gives, whichever host holds it.

    :ivar report_input: The input.
    :ivar digest: The digest of its JSON.
    """

    report_input: ReportInput
    digest: str

    @classmethod
    def of(cls, report_input: ReportInput) -> _ByContent:
        return cls(report_input, hashlib.sha256(report_input.model_dump_json(by_alias=True).encode()).hexdigest())

    def __hash__(self) -> int:
        return hash(self.digest)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ByContent) and other.digest == self.digest


#: The analyses a chart's rows are read from, each read once per input content and call: a chart's checks, its
#: completion and its resolver select its rows alike, and the analysis behind them is pure in its input.
_MEMO = 256


@functools.lru_cache(maxsize=_MEMO)
def _compared(read: _ByContent, first: str, second: str, measure: str) -> ArmsComparison | RefusedInput:
    try:
        return arms_compare(read.report_input, (first, second), measure, limit=_EVERY_ROW)
    except RefusedInput as refused:
        return refused


@functools.lru_cache(maxsize=_MEMO)
def _settled(read: _ByContent, arm: str) -> Settle | RefusedInput:
    try:
        return settle_cases(read.report_input, [arm], limit=_EVERY_ROW)
    except RefusedInput as refused:
        return refused


@functools.lru_cache(maxsize=_MEMO)
def _levered(read: _ByContent, measure: str, place: str | None) -> LeverEffects | RefusedInput:
    try:
        return lever_effects(read.report_input, measure, place_filter=place, limit=_EVERY_ROW)
    except RefusedInput as refused:
        return refused


def _refused_or[T](read: T | RefusedInput) -> T:
    """An analysis's result, or its refusal raised."""
    if isinstance(read, RefusedInput):
        raise read.with_traceback(None)
    return read


def _compare(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    """The query's first arm less its second on its one measure, at each place it keeps (``arms_compare``'s rows)."""
    if len(query.arms) != 2 or len(query.measures) != 1:
        return
    (measure,) = query.measures
    try:
        first, second = query.arms
        read = _refused_or(_compared(_ByContent.of(report_input), first, second, measure))
    except RefusedInput:
        # NOSILENT: a pair the function refuses selects no row; the chart checks name its reason, read again there.
        return
    for row in read.rows:
        if _kept_place(row.where, query, report_input):
            yield SourceRow(place_of(row.where, report_input), row, key=measure, interval=_cited_interval(row.interval))


def _settle(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    """Each kept arm against every criterion reading it (``settle_cases``' rows), on the measures and places kept."""
    measures = set(query.measures)
    by_id = {criterion.id: criterion for criterion in report_input.criteria}
    for arm in [held.id for held in report_input.arms if held.id in _arms(query, report_input)]:
        try:
            read = _refused_or(_settled(_ByContent.of(report_input), arm))
        except RefusedInput:
            # NOSILENT: an arm no criterion reads, or none of whose readings settles, has no row; the checks say why.
            continue
        for row in read.rows:
            criterion = by_id.get(row.against.removeprefix("criterion "))
            # A margin's or a floor's row reads a change, on no one measure: kept where the query names none.
            key = criterion.measure if criterion is not None else ""
            if (measures and key not in measures) or not _kept_place(row.where, query, report_input):
                continue
            against = criterion.id if criterion is not None else row.against
            yield SourceRow(f"{arm}/{against}/{place_of(row.where, report_input)}", row, arm=arm, key=key)


def lever_place(query: ChartQuery, report_input: ReportInput) -> str | None:
    """The one place a levers query reads within, as ``lever_effects`` names it; None for overall.

    :param query: The query.
    :ptype query: ChartQuery
    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The place; None where the query names none, or names overall.
    :rtype: str | None
    """
    named = [place for place in query.strata if place != OVERALL]
    if not named:
        return None
    cell = report_input.cells.get(named[0])
    return f"{cell.row}.{cell.column}" if cell is not None else named[0]


def lever_arms(row: EffectRow | InteractionRow, report_input: ReportInput) -> list[str]:
    """The arms a lever row reads: an effect's two, or an interaction's four (each move's, found by its effect's key).

    :param row: An effect or an interaction, as ``lever_effects`` returns it.
    :ptype row: EffectRow | InteractionRow
    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The arms, by id: from, then to, for each move.
    :rtype: list[str]
    """
    if isinstance(row, EffectRow):
        return [str(row.from_arm.value), str(row.to_arm.value)]
    found: list[str] = []
    ids = [arm.id for arm in report_input.arms]
    for figure in (row.first, row.second):
        stem = f"lever.{row.setting}."
        move = next(
            (
                (first, second)
                for first in ids
                for second in ids
                if first != second and figure.key.startswith(f"{stem}{first}.{second}.")
            ),
            None,
        )
        found += list(move) if move is not None else []
    return found


def _levers(query: ChartQuery, report_input: ReportInput) -> Iterator[SourceRow]:
    """The effects, then the interactions, of every setting on the query's one measure at its place
    (``lever_effects``' rows), each every arm of which the query keeps."""
    named = [place for place in query.strata if place != OVERALL]
    if len(query.measures) != 1 or len(named) > 1:
        return
    (measure,) = query.measures
    try:
        read = _refused_or(_levered(_ByContent.of(report_input), measure, lever_place(query, report_input)))
    except RefusedInput:
        # NOSILENT: a measure or place the function refuses selects no row; the chart checks name its reason.
        return
    arms = _arms(query, report_input)
    rows: list[EffectRow | InteractionRow] = [*read.effects, *read.interactions]
    for row in rows:
        moved = lever_arms(row, report_input)
        if moved and set(moved) <= arms:
            interval = _cited_interval(row.figures[1]) if len(row.figures) > 1 else None
            yield SourceRow("→".join(moved), row, key=measure, interval=interval)


#: Each source's selector (this module's table).
_SELECTORS: dict[str, Callable[[ChartQuery, ReportInput], Iterator[SourceRow]]] = {
    "readings": _readings,
    "contrasts": _contrasts,
    "measures": _figures,
    "judged": _figures,
    "configs": _figures,
    "scoreboard": _scoreboard,
    "floors": _floors,
    "strata": _slices,
    "tasks": _slices,
    "examples": _examples,
    "cells": _cells,
    "confusion": _confusion,
    "human_pref": _human_pref,
    "prices": _prices,
    "bar_reads": _bar_reads,
    "compare": _compare,
    "settle": _settle,
    "levers": _levers,
}
