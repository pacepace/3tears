"""An interval for an overall figure that is the plain mean of its target-matrix cells, derived from the cells' own.

Design note.

**Which overalls qualify.** An arm's overall figure on a measure or judged dimension that states no spread of its own
(no interval, half-width or standard error) and is, as the input states it, the unweighted mean of one measure's
figures over the cells of the target matrix: a balanced accuracy beside its cells' accuracies is the usual case. It is
identified from the numbers alone, never from a measure's name, its definition's words or the eval's kind:

- the input holds a target matrix, and a cell for each of its row and column levels, once each. The target matrix's
  cells are the one slicing the input states as a partition: two label axes whose product is the cells a case is
  graded in, so each case falls in exactly one cell. That is what makes the cells disjoint. A task can cut across the
  others (``cross_cutting``) and strata need not cover the cases, so neither is read as a partition;
- one measure or judged dimension, of the same kind and unit as the overall's, is held in every cell for each arm
  checked, and its cells' plain mean equals the overall figure within the rounding both are stated to, for every arm
  that holds the overall figure and that measure in every cell (at least one). Two such measures are refused as
  ambiguous. A mean weighted by the cells' cases or traffic is not the plain mean, unless the cells weigh alike, and
  is refused; so is a figure the cells do not average to (a macro F1).

**The formula.** For ``k`` cells whose standard errors are ``se_i``, the overall's standard error is
``sqrt(sum(se_i ** 2)) / k``: the variance of a plain mean of independent figures. The cells are independent because
they are disjoint, no case read in two of them; that is the one assumption, and the input states the partition it
rests on. Each cell's standard error is read as the separation module reads one
(:func:`~threetears.evals.analysis.separation.stat_spread`): its interval or half-width over the two-sided normal
quantile at the stated level, else its stated standard error. A binomial spread read off a count is not the cell's
own spread, so it is not read here.

**The level.** The interval is the overall figure, as stated, plus and minus the two-sided normal quantile at the
cells' stated level times that standard error: a normal approximation, at the level the cells' intervals are stated
at (the cell measure's ``level``). It is centred on the overall figure the input states, so a gate, the page and the
interval read one number. Like a pooled figure's, its ends are not held to a rate's bounds.

**What it refuses** (:class:`NoCellMean`, its message saying why): an overall that states a spread of its own (there
is nothing to derive); an input with no target matrix, or cells that do not cover it once each; an overall that is no
plain mean of any measure's cells, or of two; any cell, for the arm, that states no spread of its own: then nothing is
derived for that arm, and the overall stays a point estimate with its ``point_estimate`` caveat; cells whose level is
unstated; and cells at a level other than the one the overall states for its own intervals, since the overall would
then read at two levels. The input states one level per measure, so the cells of one measure share theirs.

**Why no other rule applies.** A weighted mean's variance needs the weights, and a weight read off the cells' cases
is a guess at how the pipeline weighed them; strata or tasks that may overlap or leave cases out would count a case
twice or not at all, and their covariance is unstated; any other overall (a ratio, an F1 over pooled counts) has no
variance that follows from its slices' alone. Each is left a point estimate rather than given an interval its input
does not support.

Every reader of a criterion's figures reads the derived interval through
:func:`~threetears.evals.analysis.report_margins.criterion_figures`, so the gate check, the decision's caveats, the
figure keys (``overall_interval.<measure>.<arm>.*``, and a criterion figure's ``low`` and ``high``, each marked
derived) and the analyst's analyses read one interval.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from statistics import NormalDist

from threetears.evals.analysis.report_input import Judged, Measure, ReportInput, Stat
from threetears.evals.analysis.separation import NoSpread, level_share, stat_spread

__all__ = ["CellMeanInterval", "NoCellMean", "cell_mean_interval"]

#: A float's own noise in an arithmetic check, beside the rounding a stated figure carries.
_FLOAT_NOISE = 1e-9


class NoCellMean(ValueError):
    """An overall figure has no interval to derive from its cells; the message says why."""


@dataclass(frozen=True)
class CellMeanInterval:
    """An arm's overall figure as the plain mean of its cells, with the interval derived from theirs.

    :ivar key: The overall's measure or judged dimension, by key.
    :ivar arm: The arm's id.
    :ivar of: The measure or judged dimension whose cells the overall is the plain mean of, by key.
    :ivar mean: The overall figure, as the input states it.
    :ivar se: Its standard error, ``sqrt(sum(se_i ** 2)) / k`` over the ``k`` cells.
    :ivar level: The level the interval is at, as a share: the cells' stated level.
    :ivar cells: The cells averaged, by id, in the input's order.
    """

    key: str
    arm: str
    of: str
    mean: float
    se: float
    level: float
    cells: tuple[str, ...]

    @property
    def half_width(self) -> float:
        """The interval's half-width: the two-sided normal quantile at the level times the standard error."""
        return NormalDist().inv_cdf(0.5 + self.level / 2.0) * self.se

    @property
    def interval(self) -> tuple[float, float]:
        """The interval, low then high, about the overall figure as stated."""
        return self.mean - self.half_width, self.mean + self.half_width


def _table(report_input: ReportInput, key: str) -> Measure | Judged | None:
    return report_input.measures.get(key) or report_input.judged.get(key)


def _partition(report_input: ReportInput) -> tuple[str, ...]:
    """The target matrix's cells, by id, when the input holds one cell for each of its row and column levels.

    :raises NoCellMean: It holds no target matrix, or its cells do not cover it once each.
    """
    matrix = report_input.target_matrix
    if matrix is None or not report_input.cells:
        raise NoCellMean("the input states no target matrix whose cells partition its cases")
    grid = {(row.id, column.id) for row in matrix.rows.levels for column in matrix.columns.levels}
    placed = [(cell.row, cell.column) for cell in report_input.cells.values()]
    if len(placed) != len(set(placed)) or set(placed) != grid:
        raise NoCellMean("its cells do not cover the target matrix once each, so they are not read as a partition")
    return tuple(report_input.cells)


def _decimals(value: float) -> int:
    """The decimals a stated figure is written to, as the input holds it (``89.8`` is written to one)."""
    exponent = Decimal(repr(float(value))).as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def _rounding(values: list[float]) -> float:
    """Half a unit in the last decimal the least precise of ``values`` is written to: how far each may sit from the
    figure it rounds."""
    return 0.5 * 10.0 ** -min(_decimals(value) for value in values)


def _cell_means(report_input: ReportInput, cells: tuple[str, ...], arm: str, key: str) -> list[float] | None:
    """An arm's mean on ``key`` in every cell, in order; None where a cell states none."""
    stats = [report_input.cells[cell].per_arm.get(arm, {}).get(key) for cell in cells]
    means = [stat.mean for stat in stats if stat is not None and stat.mean is not None]
    return [float(mean) for mean in means] if len(means) == len(cells) else None


def _overall_mean(stat: Stat | None) -> float | None:
    return float(stat.mean) if stat is not None and stat.mean is not None and math.isfinite(stat.mean) else None


def _same_kind(report_input: ReportInput, key: str, other: str) -> bool:
    """Whether two keys are of one kind and unit: two measures in one unit, or two judged dimensions on one scale."""
    if key in report_input.measures:
        return other in report_input.measures and report_input.measures[other].unit == report_input.measures[key].unit
    return other in report_input.judged and report_input.judged[other].scale == report_input.judged[key].scale


def _averaged(report_input: ReportInput, key: str, per_arm: Mapping[str, Stat], cells: tuple[str, ...]) -> str:
    """The one measure whose cells' plain mean is the overall figure for every arm checked (this module's Which
    overalls qualify).

    :raises NoCellMean: None is, or two are.
    """
    held = dict.fromkeys(
        name for cell in cells for figures in report_input.cells[cell].per_arm.values() for name in figures
    )
    found: list[str] = []
    for other in held:
        if not _same_kind(report_input, key, other):
            continue
        checked = 0
        agrees = True
        for arm, stat in per_arm.items():
            overall = _overall_mean(stat)
            means = _cell_means(report_input, cells, arm, other)
            if overall is None or means is None:
                continue
            checked += 1
            mean = sum(means) / len(means)
            if abs(mean - overall) > _rounding([overall]) + _rounding(means) + _FLOAT_NOISE:
                agrees = False
                break
        if checked and agrees:
            found.append(other)
    if not found:
        raise NoCellMean(f"{key!r} is not the plain mean of any measure's cells, as the figures the input states read")
    if len(found) > 1:
        raise NoCellMean(
            f"{key!r} is the plain mean of more than one measure's cells ({', '.join(found)}), so which is unsaid"
        )
    return found[0]


def cell_mean_interval(report_input: ReportInput, key: str, arm: str) -> CellMeanInterval:
    """An arm's overall figure on ``key`` read as the plain mean of its target-matrix cells, with the interval derived
    from the cells' own (this module's design note).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The overall's measure or judged dimension, by key: its lead aspect's figure, the by-arm map's.
    :ptype key: str
    :param arm: The arm's id.
    :ptype arm: str
    :return: The overall figure, its standard error and the level its interval is at.
    :rtype: CellMeanInterval
    :raises NoCellMean: The figure does not qualify, or a cell gives it no interval; the message says why.
    """
    table = _table(report_input, key)
    stat = table.per_arm.get(arm) if table is not None else None
    overall = _overall_mean(stat)
    if table is None or stat is None or overall is None:
        raise NoCellMean(f"the input states no overall figure for {arm} on {key!r}")
    if stat.ci is not None or stat.half_width is not None or stat.sem is not None:
        raise NoCellMean(f"{arm}'s overall figure on {key!r} states a spread of its own, so none is derived")
    cells = _partition(report_input)
    of = _averaged(report_input, key, table.per_arm, cells)
    source = _table(report_input, of)
    level = level_share(source.level) if source is not None else None
    if level is None:
        raise NoCellMean(f"the cells' figures on {of!r} state no level to build the interval at")
    own = level_share(table.level)
    if own is not None and own != level:
        raise NoCellMean(
            f"the cells' intervals on {of!r} are at {source.level if source is not None else level} and {key!r} "
            f"states its own at {table.level}: cells at a level other than the overall's are not combined"
        )
    errors: list[float] = []
    for cell in cells:
        held = report_input.cells[cell].per_arm.get(arm, {}).get(of)
        if held is None:
            raise NoCellMean(f"cell {cell!r} states no figure for {arm} on {of!r}")
        try:
            spread = stat_spread(
                None if held.mean is None else float(held.mean),
                ci=None if held.ci is None else (float(held.ci[0]), float(held.ci[1])),
                half_width=None if held.half_width is None else float(held.half_width),
                sem=None if held.sem is None else float(held.sem),
                stated_level=level,
            )
        except NoSpread as none:
            raise NoCellMean(
                f"cell {cell!r} states no spread of its own for {arm} on {of!r} ({none}), so nothing is derived"
            ) from none
        errors.append(spread.se)
    se = math.sqrt(sum(error**2 for error in errors)) / len(errors)
    return CellMeanInterval(key=key, arm=arm, of=of, mean=overall, se=se, level=level, cells=cells)
