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
==============  ================================================================  =========================  ========

A figure row reads the query's aspect from the measure's ``by_aspect`` where it names one the measure holds, else its
``per_arm`` (the lead aspect's). A scoreboard reads the named aspect's own scoreboard where it has one. A row has an
interval where the input carries one: a reading's ``within_interval``, a contrast's or a floor's ``interval``, a
figure's ``ci``, a preference slice's half-width about its share.

**The filters.** An empty filter keeps everything (:class:`~threetears.evals.kernel.report_plan.ChartQuery`).
``arms`` keeps the rows of those arms, where a row is of one arm (an example with no arm is kept); ``measures`` the
contrasts, figures, floors and confusion axes on those keys; ``strata`` the slices, examples and cells by id;
``aspect`` the contrasts, floors and examples on it (an example naming no aspect is kept), and the aspect readings,
figures and a scoreboard read. Rows come in the input's order, whatever order the query lists its arms in: the
order a chart shows them in is the chart's.

**The limit.** ``limit`` is the most rows a strata or examples query keeps, 0 for no limit. Strata keep the largest
by ``n_cases`` (a stratum without a count as the smallest), largest first, ties in the input's order; examples keep
the first, in the input's order. A limit on any other source keeps every row.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

from threetears.evals.analysis.report_input import (
    Aspect,
    Cell,
    ConfusionTable,
    Contrast,
    Example,
    FloorRead,
    Judged,
    Measure,
    PreferenceRow,
    Reading,
    ReportInput,
    ScoreboardRow,
    Stat,
    Stratum,
    Task,
)
from threetears.evals.kernel.report_plan import ChartQuery

__all__ = ["SourceRow", "query_aspect", "query_rows"]


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
        arms &= {config.arm for config in report_input.configs}
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
    preference = report_input.human_pref
    if preference is None:
        return
    for row in preference.rows:
        width = row.half_width
        interval = (row.prefers_control - width, row.prefers_control + width) if width is not None else None
        yield SourceRow(row.id, row, arm=preference.arm, interval=_interval(interval))


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
}
