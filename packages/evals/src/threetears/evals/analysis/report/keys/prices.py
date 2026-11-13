"""Prices: one option priced in another on one gain axis, the price frontier, the calls that went unanswered beside
an arm's cost, two arms' costs apart, and a cost some trials did not report.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from threetears.evals.analysis.cell_mean import CellMeanInterval, NoCellMean, cell_mean_interval
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.cost_unit import per_noun
from threetears.evals.analysis.report.keys.counts import _spelled_cell
from threetears.evals.analysis.report.keys.figure import Figure, Scale, _PER_THOUSAND_FACTORS, _count
from threetears.evals.analysis.report.keys.names import _measure_reader_name
from threetears.evals.analysis.report.keys.reads import (
    _cell_spelling,
    _cost_measures,
    _costs_per_thousand,
    _one_stratum,
    _scale_of,
    cost_basis,
)
from threetears.evals.analysis.report.keys.vocabulary import (
    _control,
    _interval,
    _judged_scale,
    _level,
    _measure_scale,
    _number,
    _refuse,
    _task,
    _text,
    _unstated,
    on_lead_aspect,
)
from threetears.evals.analysis.report_input import Contrast, Judged, Measure, ReportInput, Stat, Unanswered
from threetears.evals.analysis.report_margins import aspect_board, surviving_arms, undecided_arms
from threetears.evals.kernel.numbers import group_thousands


# --- prices: one option priced in another, on one gain axis -------------------------------------------------------

#: What a price key reads for a move from one option to another, on one gain axis. ``gain`` is how much better the
#: second option reads, toward the axis's better side; ``gain_interval`` the interval on it, where a contrast states
#: one (paired); ``gain_interval_unpaired`` the gain plus or minus the two options' own half-widths on the axis
#: combined as independent (descriptive, uncorrected), where both state one; ``cost_gap`` what the second costs more
#: per 1,000 of what the cost counts; ``per_point`` the cost gap per point gained (per unit, on an axis not in
#: points), and its ``low`` and ``high`` ends read off the gain's interval, paired or (``_unpaired``) unpaired; the
#: ``monthly_`` fields the same at the declared monthly volume.
_PRICE_FIELDS = (
    "gain",
    "gain_interval",
    "gain_interval_unpaired",
    "cost_gap",
    "per_point",
    "per_point_low",
    "per_point_high",
    "per_point_low_unpaired",
    "per_point_high_unpaired",
    "monthly_cost_gap",
    "monthly_per_point",
    "monthly_per_point_low",
    "monthly_per_point_high",
    "monthly_per_point_low_unpaired",
    "monthly_per_point_high_unpaired",
)

#: What a price key within one slice (a task, a stratum, a cell) reads: the same on the slice's own figures, with no
#: paired interval and no monthly figure; ``cost_basis`` says whose cost it reads.
_SLICE_PRICE_FIELDS = (
    "gain",
    "gain_interval_unpaired",
    "cost_gap",
    "per_point",
    "per_point_low_unpaired",
    "per_point_high_unpaired",
    "cost_basis",
)

#: The price fields read at the declared monthly volume.
_MONTHLY_PREFIX = "monthly_"

#: The price fields read off the unpaired interval.
_UNPAIRED_SUFFIX = "_unpaired"

#: Where a gain axis places its arms: each candidate's contrast against the control, each arm's overall mean, each
#: arm's figures pooled over the slices that hold them, or each arm's mean within one slice.
GainBasis = Literal["contrast", "means", "pooled", "slice"]

#: How a pooled figure weighs its slices: by the traffic share the input states for each, or equally.
PoolWeights = Literal["traffic", "equal"]

#: How each weighting reads, as ``pooled.<measure>.<arm>.weights`` writes it.
POOL_WEIGHT_WORDS: Mapping[str, str] = {
    "traffic": "weighted by each slice's traffic share, as the input states it",
    "equal": "weighted equally: the input states no traffic share for every slice, so this is not a traffic mix",
}


@dataclass(frozen=True)
class GainAxis:
    """A measure or judged dimension as the axis a trade's gain is read on, and where it places each arm.

    ``positions`` puts each arm that holds a figure on the axis, signed so that higher is better: on the ``contrast``
    basis, each candidate's contrast against the control (the control at zero); on the ``means`` basis, each arm's
    overall mean; on the ``pooled`` basis, each arm's figures pooled over the slices holding them
    (:func:`pooled_figures`); on the ``slice`` basis, each arm's mean within one slice. ``spreads`` is each arm's own
    half-width on the axis, where its figure states one: what an unpaired interval on a gain is read from. A gain
    between two arms is the difference of their positions, so every price on one axis reads one basis. Code reads it
    to order options; a figure it prints is always a key's.
    """

    key: str
    higher_is_better: bool
    basis: GainBasis
    positions: Mapping[str, float]
    spreads: Mapping[str, float] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class PooledFigure:
    """One arm's figure on a measure or judged dimension pooled over the slices holding it, where no overall figure
    is stated (:func:`pooled_figures`).

    :ivar mean: The weighted mean of its slice means.
    :ivar half_width: The half-width of that mean, the slices' half-widths combined as independent; None unless every
        slice states one.
    :ivar weights: How the slices are weighted: by traffic share, or equally.
    :ivar slices: The slices pooled, by id (a cell as ``<row>.<column>``).
    """

    mean: float
    half_width: float | None
    weights: PoolWeights
    slices: tuple[str, ...]


def _half_width(stat: Stat) -> float | None:
    """A figure's half-width: as stated, else half its interval's width; None where it states neither."""
    if stat.half_width is not None:
        return float(stat.half_width)
    if stat.ci is not None:
        return (float(stat.ci[1]) - float(stat.ci[0])) / 2
    return None


def _holds_mean(per_arm: Mapping[str, Mapping[str, Stat]], key: str) -> bool:
    return any(key in figures and figures[key].mean is not None for figures in per_arm.values())


_Slice = tuple[str, float | None, Mapping[str, Mapping[str, Stat]]]


def _pool_slices(report_input: ReportInput, key: str) -> list[_Slice]:
    """The slices a measure's figures are pooled over: the tasks holding it; else the strata holding it, where they
    are of one stratification (strata of two would count a case twice); else the cells holding it."""
    tasks: list[_Slice] = [(t.id, t.share, t.per_arm) for t in report_input.tasks if _holds_mean(t.per_arm, key)]
    if tasks:
        return tasks
    strata = [stratum for stratum in report_input.strata if _holds_mean(stratum.per_arm, key)]
    if strata and len({(stratum.aspect, stratum.dimension) for stratum in strata}) == 1:
        return [(stratum.id, stratum.share, stratum.per_arm) for stratum in strata]
    return [
        (_cell_spelling(cell), None, cell.per_arm)
        for cell in report_input.cells.values()
        if _holds_mean(cell.per_arm, key)
    ]


def pooled_figures(report_input: ReportInput, key: str) -> dict[str, PooledFigure]:
    """Each arm's figure on a measure or judged dimension pooled over the slices holding it.

    The slices are the tasks holding the key, else the strata of one stratification, else the cells. Each is weighted
    by the traffic share the input states for it where it states one for every slice (a slice at a share of 0, a test
    category, then weighs nothing), else equally, which the weighting says. An arm is pooled only where it holds a
    figure in every slice that weighs something, so every arm pools the same slices. Its half-width is the slices'
    half-widths combined as independent, ``sqrt(sum((w / W)^2 * h^2))``, where every slice states one: descriptive and
    uncorrected.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure or judged dimension, by key.
    :ptype key: str
    :return: Each pooled arm's figure, by arm id; empty where no slice holds the key.
    :rtype: dict[str, PooledFigure]
    """
    slices = _pool_slices(report_input, key)
    shares = [share for _, share, _ in slices]
    traffic = bool(slices) and all(share is not None for share in shares) and sum(float(s or 0) for s in shares) > 0
    weighted = [(sid, float(share or 0) if traffic else 1.0, per_arm) for sid, share, per_arm in slices]
    weighted = [row for row in weighted if row[1] > 0]
    total = sum(weight for _, weight, _ in weighted)
    found: dict[str, PooledFigure] = {}
    for arm in dict.fromkeys(arm for _, _, per_arm in weighted for arm in per_arm):
        stats = [per_arm.get(arm, {}).get(key) for _, _, per_arm in weighted]
        means = [float(stat.mean) for stat in stats if stat is not None and stat.mean is not None]
        if len(means) != len(weighted):
            continue
        widths = [width for stat in stats if stat is not None and (width := _half_width(stat)) is not None]
        mean = sum(weight * value for (_, weight, _), value in zip(weighted, means, strict=True)) / total
        spread = None
        if len(widths) == len(weighted):
            pairs = zip(weighted, widths, strict=True)
            spread = math.sqrt(sum((weight / total) ** 2 * width**2 for (_, weight, _), width in pairs))
        found[arm] = PooledFigure(
            mean=mean,
            half_width=spread,
            weights="traffic" if traffic else "equal",
            slices=tuple(sid for sid, _, _ in weighted),
        )
    return found


_POOLED_FIELDS = ("mean", "half_width", "interval", "weights", "slices")


def _pooled_keys(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    keys = list(dict.fromkeys([*report_input.measures, *report_input.judged]))
    return [key for key in keys if _pool_slices(report_input, key)]


def _pooled_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(pooled_figures(report_input, prior[0]))


def _pooled(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    key, arm, statistic = values
    pooled = pooled_figures(report_input, key)[arm]
    scale, unit = _scale_of(report_input, key)
    if statistic == "mean":
        return _number(path, pooled.mean, "the pooled mean", scale, unit=unit)
    if statistic == "weights":
        return _text(path, POOL_WEIGHT_WORDS[pooled.weights], "how the slices are weighted")
    if statistic == "slices":
        return Figure(path=path, kind="number", value=float(len(pooled.slices)), scale="count")
    if pooled.half_width is None:
        raise _unstated(path, "names the pooled interval, which needs a half-width in every slice, and one is unstated")
    if statistic == "half_width":
        return _number(path, pooled.half_width, "the pooled half-width", scale, unit=unit)
    low, high = pooled.mean - pooled.half_width, pooled.mean + pooled.half_width
    return _interval(path, (low, high), "the pooled interval", scale, unit=unit)


#: What an overall interval derived from its cells reads: the overall figure as stated, its standard error and the
#: interval's half-width (in points for a rate), the interval and its ends, its level, the measure whose cells it is
#: the plain mean of, how many cells, which means of them the figure fits, and how it was built, in one sentence.
_OVERALL_INTERVAL_FIELDS = (
    "mean",
    "se",
    "half_width",
    "interval",
    "low",
    "high",
    "level",
    "of",
    "cells",
    "weights",
    "basis",
)

#: Which means of its cells an overall fits, as ``overall_interval.<measure>.<arm>.weights`` writes it.
_OVERALL_INTERVAL_WEIGHTS = {
    False: "the plain mean of the cells",
    True: "both the plain and the case-weighted mean of the cells; the interval is the wider of the two readings'",
}

#: How an overall interval derived from its cells was built, as ``overall_interval.<measure>.<arm>.basis`` writes it.
_OVERALL_INTERVAL_BASIS = (
    "derived here, not stated: the figure is the plain mean of {of} over the target matrix's cells, so its standard "
    "error is the root of the sum of the cells' squared standard errors over their count, the cells read as "
    "independent because no case falls in two, and its interval is built at the cells' stated level"
)

#: The same, for an overall the cells' case-weighted mean fits as well as their plain one.
_OVERALL_INTERVAL_BOTH_BASIS = (
    "derived here, not stated: the figure fits both a plain and a case-weighted mean of {of} over the target matrix's "
    "cells, so its interval uses the wider of the two standard errors, each combining the cells' as independent "
    "because no case falls in two, at the cells' stated level"
)


def _unspread_keys(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    """The measures and judged dimensions an overall interval may be derived for: an input with cells, and some arm's
    overall figure on the key stating a mean and no spread of its own."""
    if not report_input.cells:
        return []
    tables: list[tuple[str, Measure | Judged]] = [*report_input.measures.items(), *report_input.judged.items()]
    return list(
        dict.fromkeys(
            key
            for key, table in tables
            if any(
                stat.mean is not None and stat.ci is None and stat.half_width is None and stat.sem is None
                for stat in table.per_arm.values()
            )
        )
    )


def _unspread_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    table = report_input.measures.get(prior[0]) or report_input.judged[prior[0]]
    return [arm for arm, stat in table.per_arm.items() if stat.mean is not None]


def _overall_interval(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    key, arm, field = values
    try:
        derived: CellMeanInterval = cell_mean_interval(report_input, key, arm)
    except NoCellMean as none:
        raise _unstated(path, f"names an interval derived from the cells, and none is: {none}") from none
    scale, unit = _scale_of(report_input, key)
    figure: Figure
    if field == "mean":
        figure = _number(path, derived.mean, "the overall figure", scale, unit=unit)
    elif field in ("se", "half_width"):
        value = derived.se if field == "se" else derived.half_width
        figure = _apart(report_input, path, key, value, f"the derived interval's {field}", change=False)
    elif field == "interval":
        figure = _interval(path, derived.interval, "the derived interval", scale, unit=unit)
    elif field in ("low", "high"):
        end = derived.interval[0] if field == "low" else derived.interval[1]
        figure = _number(path, end, f"the derived interval's {field} end", scale, unit=unit)
    elif field == "level":
        figure = _level(path, derived.level, "the level the derived interval is at")
    elif field == "of":
        figure = _text(path, _measure_reader_name(derived.of, report_input), "the measure whose cells it averages")
    elif field == "cells":
        figure = Figure(path=path, kind="number", value=float(len(derived.cells)), scale="count")
    elif field == "weights":
        figure = _text(path, _OVERALL_INTERVAL_WEIGHTS[derived.weighted_too], "which means of the cells it fits")
    else:
        of = _measure_reader_name(derived.of, report_input)
        basis = _OVERALL_INTERVAL_BOTH_BASIS if derived.weighted_too else _OVERALL_INTERVAL_BASIS
        figure = _text(path, basis.format(of=of), "how the interval was built")
    return replace(figure, derived=True)


def _gain_axes(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    """Every measure or judged dimension a gain can be read on: a judged dimension, or a measure with a direction."""
    measures = [key for key, measure in report_input.measures.items() if measure.higher_is_better is not None]
    return list(dict.fromkeys([*report_input.judged, *measures]))


def _gain_table(report_input: ReportInput, key: str) -> Measure | Judged:
    """The axis's own table: a judged dimension's, where the key names one, else the measure's."""
    return report_input.judged[key] if key in report_input.judged else report_input.measures[key]


def _gain_contrasts(report_input: ReportInput, key: str, control: str) -> list[Contrast]:
    """The contrasts that state a candidate against the control on the axis, with their difference, on the lead aspect
    the axis's own figures belong to (:func:`on_lead_aspect`)."""
    return [
        contrast
        for contrast in report_input.contrasts
        if contrast.measure == key
        and on_lead_aspect(report_input, contrast)
        and contrast.delta is not None
        and contrast.arm != control
        and contrast.control in (None, control)
    ]


def _delta(contrast: Contrast) -> float:
    if contrast.delta is None:
        raise ValueError(f"the contrast of {contrast.arm} on {contrast.measure!r} states no difference")
    return float(contrast.delta)


def _direction(report_input: ReportInput, key: str, path: str) -> bool:
    """Whether higher is better on a gain axis; a key naming nothing, or a measure stating no direction, refused."""
    if key not in report_input.judged and key not in report_input.measures:
        raise _refuse(path, f"names {key!r}, which the input holds as no measure or judged dimension")
    table = _gain_table(report_input, key)
    higher = True if isinstance(table, Judged) else table.higher_is_better
    if higher is None:
        raise _refuse(path, f"reads a gain on {key!r}, which states no direction (higher or lower is better)")
    return higher


def gain_axis(report_input: ReportInput, key: str) -> GainAxis:
    """A measure or judged dimension as a trade's gain axis: which way is better, and where it places each arm.

    The axis reads its arms' **contrasts** against the control when the input has a control and every other arm
    holding a figure on the axis holds exactly one contrast on it that states its difference, on the lead aspect whose
    figures the arms' own means are: the paired estimate, which the contrast's interval belongs to. Otherwise it reads each arm's **overall mean**, which carries no paired
    interval; and where no arm states an overall mean, each arm's figures **pooled** over the slices holding them
    (:func:`pooled_figures`). A judged dimension is read higher is better.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure or judged dimension, by key.
    :ptype key: str
    :return: The axis.
    :rtype: GainAxis
    :raises UnresolvableReference: The key names no measure or judged dimension the input holds, or a measure that
        states no direction.
    """
    higher = _direction(report_input, key, f"price.<from>.<to>.{key}")
    table = _gain_table(report_input, key)
    sign = 1.0 if higher else -1.0
    means = {arm: float(stat.mean) for arm, stat in table.per_arm.items() if stat.mean is not None}
    spreads = {arm: width for arm, stat in table.per_arm.items() if (width := _half_width(stat)) is not None}
    control = _control(report_input)
    if control is not None:
        contrasts = _gain_contrasts(report_input, key, control.id)
        held = {*means, *(contrast.arm for contrast in contrasts)} - {control.id}
        arms = [contrast.arm for contrast in contrasts]
        if contrasts and all(arms.count(arm) == 1 for arm in held):
            positions = {control.id: 0.0}
            positions |= {contrast.arm: sign * _delta(contrast) for contrast in contrasts}
            return GainAxis(key=key, higher_is_better=higher, basis="contrast", positions=positions, spreads=spreads)
    if not means:
        pooled = pooled_figures(report_input, key)
        if pooled:
            return GainAxis(
                key=key,
                higher_is_better=higher,
                basis="pooled",
                positions={arm: sign * figure.mean for arm, figure in pooled.items()},
                spreads={arm: figure.half_width for arm, figure in pooled.items() if figure.half_width is not None},
            )
    return GainAxis(
        key=key,
        higher_is_better=higher,
        basis="means",
        positions={arm: sign * mean for arm, mean in means.items()},
        spreads=spreads,
    )


def _gain_scale(report_input: ReportInput, axis: GainAxis) -> tuple[Scale, str]:
    table = _gain_table(report_input, axis.key)
    if axis.basis in ("pooled", "slice") or not table.per_arm:
        return _scale_of(report_input, axis.key)
    if isinstance(table, Judged):
        return _judged_scale(table), ""
    return _measure_scale(table)


def _gain_interval(
    report_input: ReportInput, axis: GainAxis, first: str, second: str
) -> tuple[tuple[float, float] | None, str]:
    """The paired interval on the gain from ``first`` to ``second``, signed toward better, or None and why it is
    unstated."""
    control = _control(report_input)
    if axis.basis != "contrast":
        return None, (
            f"the gain on {axis.key!r} is read on the arms' own figures, not on contrasts against the control, so no "
            "paired interval is stated on it; an unpaired one may be (gain_interval_unpaired)"
        )
    if control is None or control.id not in (first, second):
        return None, "a paired interval is stated only between the control and a candidate, by their contrast"
    candidate, toward = (second, 1.0) if first == control.id else (first, -1.0)
    (contrast,) = (c for c in _gain_contrasts(report_input, axis.key, control.id) if c.arm == candidate)
    if contrast.interval is None:
        return None, f"the contrast of {candidate} against the control on {axis.key!r} states no interval"
    sign = toward * (1.0 if axis.higher_is_better else -1.0)
    low, high = sorted((sign * float(contrast.interval[0]), sign * float(contrast.interval[1])))
    return (low, high), ""


def _unpaired_interval(axis: GainAxis, first: str, second: str) -> tuple[tuple[float, float] | None, str]:
    """The unpaired interval on the gain from ``first`` to ``second``: the gain plus or minus the two options' own
    half-widths combined as independent; or None and why it is unstated."""
    missing = [arm for arm in (first, second) if arm not in axis.spreads]
    if missing:
        return None, f"{' and '.join(missing)} states no interval of its own on {axis.key!r}"
    gain = axis.positions[second] - axis.positions[first]
    width = math.hypot(axis.spreads[first], axis.spreads[second])
    return (gain - width, gain + width), ""


def _gain_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    """The arms a move can be priced to: every costed arm but the one it starts from."""
    return [arm for arm in _costs_per_thousand(report_input) if arm != prior[0]]


def _monthly_factor(report_input: ReportInput, path: str) -> float:
    """What turns a figure per 1,000 of what the cost counts into one per month at the declared volume."""
    volume = report_input.meta.volume_per_month
    if volume is None:
        raise _unstated(
            path,
            "names a monthly figure, which needs the monthly volume (meta.volume_per_month), which the input does "
            "not state",
        )
    return float(volume) / 1000.0


def _price_bounds(path: str, gap: float, gain: float, interval: tuple[float, float], field: str) -> float:
    """One end of the price per point the gain's interval allows: ``gap`` over each end of the interval's part on
    the gain's own side of zero. An end at or across zero bounds nothing: that end is unbounded, and refused so."""
    low, high = interval
    far, near = (high, low) if gain > 0 else (low, high)
    if far * gain <= 0:
        raise _refuse(path, "reads a gain whose interval does not hold it, so no price is read off the interval")
    if field == "per_point_low":
        return gap / far
    if near * gain <= 0:
        raise _refuse(
            path,
            "is unbounded: the gain's interval reaches zero, so the evidence allows a price per point as high as "
            "any; no number states it",
        )
    return gap / near


@dataclass(frozen=True)
class _PriceBasis:
    """What a price reads: the gain axis, each option's cost per 1,000 of what the cost counts, the slice it is read
    within (its key
    prefix; empty overall), whose cost it reads within a slice, and the measure the slice's own cost is read on (empty
    where it reads each option's overall cost)."""

    axis: GainAxis
    costs: Mapping[str, float]
    within: str = ""
    cost_basis: str = ""
    cost_key: str = ""


#: Whose cost a price within a slice reads, as ``cost_basis`` writes it.
SLICE_COST_WORDS: Mapping[str, str] = {
    "slice": "each option's own cost in this slice",
    "overall": "each option's overall cost: the slice states no cost for every option it places",
}


def _priced(basis: _PriceBasis, report_input: ReportInput, first: str, second: str, field: str, path: str) -> Figure:
    """One field of the move from ``first`` to ``second`` on a basis (this section's fields)."""
    axis = basis.axis
    for arm in (first, second):
        if arm not in axis.positions:
            raise _unstated(path, f"names {arm}'s figure on {axis.key!r}, which the input does not state")
    gain = axis.positions[second] - axis.positions[first]
    scale, unit = _gain_scale(report_input, axis)
    if field == "gain":
        return Figure(path=path, kind="number", value=gain, scale=scale, unit=unit, change=True)
    if field == "cost_basis":
        return _text(path, basis.cost_basis, "whose cost the price reads")
    unpaired = field.endswith(_UNPAIRED_SUFFIX)
    field = field.removesuffix(_UNPAIRED_SUFFIX)
    if unpaired:
        interval, unstated = _unpaired_interval(axis, first, second)
    else:
        interval, unstated = _gain_interval(report_input, axis, first, second)
    if field == "gain_interval":
        if interval is None:
            raise _unstated(path, f"names the gain's interval, which the input does not state: {unstated}")
        return Figure(path=path, kind="interval", value=interval, scale=scale, unit=unit, change=True)
    for arm in (first, second):
        if arm not in basis.costs:
            raise _unstated(path, f"names {arm}'s cost, which the input does not state")
    gap = basis.costs[second] - basis.costs[first]
    monthly = field.startswith(_MONTHLY_PREFIX)
    factor = _monthly_factor(report_input, path) if monthly else 1.0
    money: Scale = "usd" if monthly else "usd_per_1k"
    # Money read off a slice's own cost measure is per what that measure counts, not the page's cost basis.
    per = per_noun(report_input.measures[basis.cost_key]) if basis.cost_key and not monthly else ""
    field = field.removeprefix(_MONTHLY_PREFIX)
    if field == "cost_gap":
        return Figure(path=path, kind="number", value=gap * factor, scale=money, change=True, per=per, total=monthly)
    points = 100.0 if scale == "share" else 1.0
    if gain == 0 or gap * gain <= 0:
        raise _refuse(
            path,
            f"is no price: {second} reads no better than {first} on {axis.key!r} for no less cost, or better for no "
            "more, so one is at least as good on both and the move has no price per point",
        )
    if field == "per_point":
        return Figure(
            path=path, kind="number", value=gap / (gain * points) * factor, scale=money, per=per, total=monthly
        )
    if interval is None:
        raise _unstated(path, f"names an end of the price's interval, which the input does not state: {unstated}")
    bound = _price_bounds(path, gap, gain * points, (interval[0] * points, interval[1] * points), field)
    return Figure(path=path, kind="number", value=bound * factor, scale=money, per=per, total=monthly)


def _price(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    first, second, key, field = values
    basis = _PriceBasis(axis=gain_axis(report_input, key), costs=_costs_per_thousand(report_input))
    return _priced(basis, report_input, first, second, field, path)


#: The slice families a price is read within, by the word their keys begin with.
SliceKind = Literal["task", "strata", "cells"]


def _slice_figures(
    report_input: ReportInput, kind: SliceKind, slice_id: str, path: str
) -> Mapping[str, Mapping[str, Stat]]:
    """One slice's figures by arm: a task by id, a stratum by its id (one held on more than one aspect refused), a
    cell by its spelling."""
    if kind == "task":
        return _task(report_input, slice_id).per_arm
    if kind == "strata":
        return _one_stratum(report_input, None, slice_id, path).per_arm
    return report_input.cells[_spelled_cell(report_input, slice_id)].per_arm


def _slice_basis(report_input: ReportInput, kind: SliceKind, slice_id: str, key: str, path: str) -> _PriceBasis:
    """A price's basis within one slice: each arm's mean there on the axis, and each option's cost there where the
    slice states one for every option it places (on the measures cost is read from, in their order), else each
    option's overall cost."""
    higher = _direction(report_input, key, path)
    figures = _slice_figures(report_input, kind, slice_id, path)
    sign = 1.0 if higher else -1.0
    held = {arm: by_key[key] for arm, by_key in figures.items() if key in by_key and by_key[key].mean is not None}
    positions = {arm: sign * float(stat.mean) for arm, stat in held.items() if stat.mean is not None}
    spreads = {arm: width for arm, stat in held.items() if (width := _half_width(stat)) is not None}
    axis = GainAxis(key=key, higher_is_better=higher, basis="slice", positions=positions, spreads=spreads)
    within = f"{kind}.{slice_id}"
    first, lower = _cost_measures(report_input)
    for cost_key in dict.fromkeys([*first, *lower]):
        factor = _PER_THOUSAND_FACTORS[report_input.measures[cost_key].unit or ""]
        costs: dict[str, float] = {}
        for arm, by_key in figures.items():
            stat = by_key.get(cost_key)
            if stat is not None and stat.mean is not None:
                costs[arm] = float(stat.mean) * factor
        if positions and all(arm in costs for arm in positions):
            return _PriceBasis(axis, costs, within, SLICE_COST_WORDS["slice"], cost_key)
    return _PriceBasis(axis, _costs_per_thousand(report_input), within, SLICE_COST_WORDS["overall"])


def _slice_price(kind: SliceKind) -> Callable[[ReportInput, tuple[str, ...], str], Figure]:
    def resolve_price(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
        slice_id, first, second, key, field = values
        basis = _slice_basis(report_input, kind, slice_id, key, path)
        return _priced(basis, report_input, first, second, field, path)

    return resolve_price


def _slice_arms(kind: SliceKind) -> Callable[[ReportInput, tuple[str, ...]], list[str]]:
    """The arms a move within a slice can start from: every arm holding a figure there."""

    def arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
        figures = _slice_figures(report_input, kind, prior[0], f"{kind}.{prior[0]}.price")
        return [arm.id for arm in report_input.arms if arm.id in figures]

    return arms


def _slice_seconds(kind: SliceKind) -> Callable[[ReportInput, tuple[str, ...]], list[str]]:
    """The arms a move within a slice can go to: every other arm holding a figure there."""

    def arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
        return [arm for arm in _slice_arms(kind)(report_input, prior[:1]) if arm != prior[1]]

    return arms


def _means_of(report_input: ReportInput, key: str) -> list[float] | None:
    """The amounts a change on a measure compares: each arm's mean on it; None for a key that names no measure."""
    measure = report_input.measures.get(key)
    if measure is None:
        return None
    return [float(stat.mean) for stat in measure.per_arm.values() if stat.mean is not None]


def _gap_compares(report_input: ReportInput, costs: Mapping[str, float], field: str) -> list[float] | None:
    """The amounts a cost gap compares: each option's cost per 1,000 (per month for a ``monthly`` gap, where a volume is
    declared); None for a field that is no cost gap, which reads its family's own amounts."""
    if field in ("cost_gap", "per_1k"):
        return list(costs.values())
    if field in ("monthly_cost_gap", "monthly") and report_input.meta.volume_per_month is not None:
        factor = float(report_input.meta.volume_per_month) / 1000.0
        return [cost * factor for cost in costs.values()]
    return None


def _slice_compares(kind: SliceKind) -> Callable[[ReportInput, tuple[str, ...]], list[float] | None]:
    """The amounts a cost gap within a slice compares: the options' costs its price reads (:func:`_slice_basis`)."""

    def compares(report_input: ReportInput, values: tuple[str, ...]) -> list[float] | None:
        slice_id, _first, _second, key, field = values
        try:
            basis = _slice_basis(report_input, kind, slice_id, key, f"{kind}.{slice_id}.price")
        except UnresolvableReference:
            # NOSILENT: a slice that bases no price leaves its gap unread by resolve, which raises that refusal itself;
            # here it only means no compared amounts, so the family's own are read.
            return None
        return _gap_compares(report_input, basis.costs, field)

    return compares


def _cost_per_month(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    factor = _monthly_factor(report_input, path)
    costs = _costs_per_thousand(report_input)
    return Figure(path=path, kind="number", value=costs[values[0]] * factor, scale="usd", total=True)


# --- the price frontier: the options best at some price per point --------------------------------------------------

#: Why an option is not priced: a gate or guardrail removes it, a gate has no figure for it somewhere while none it has
#: fails (undecided: neither shown to pass nor removed), the cost rule states no cost for it, or the gain axis places it
#: nowhere.
Unpriced = Literal["gated", "undecided", "no_cost", "no_gain_figure"]


@dataclass(frozen=True)
class PriceFrontier:
    """The options a trade prices on one gain axis, overall or within one slice, and which are best at some price.

    :ivar gain_key: The gain axis, by key.
    :ivar within: The slice it is read within, as a key's prefix (``task.<task>``); empty overall.
    :ivar axis: The gain axis, as the basis places each arm.
    :ivar costs: Each option's cost per 1,000 of what the cost counts, as the basis reads it.
    :ivar cost_key: Within a slice, the measure its own cost is read on; empty where each option's overall cost is.
    :ivar priced: The options priced, in the input's order: each survives the gates (or is admitted), holds a cost and
        is placed on the axis.
    :ivar admitted: The options priced as a what-if although a gate or guardrail removes them.
    :ivar unpriced: Every other option, and why it is not priced.
    :ivar frontier: The options best at some price per point, cheapest first: the upper hull of the undominated.
    :ivar dominated: Each dominated option, and the first option (in the input's order) costing no more and reading no
        worse, ahead on one of the two.
    :ivar under: Each undominated option under the hull, and the two frontier options either side of it.
    """

    gain_key: str
    within: str
    axis: GainAxis
    costs: Mapping[str, float]
    cost_key: str
    priced: tuple[str, ...]
    admitted: tuple[str, ...]
    unpriced: Mapping[str, Unpriced]
    frontier: tuple[str, ...]
    dominated: Mapping[str, str]
    under: Mapping[str, tuple[str, str]]

    def key(self, first: str, second: str, field: str) -> str:
        """The key of one field of the move from ``first`` to ``second`` on this frontier's basis.

        :param first: The option moved from.
        :ptype first: str
        :param second: The option moved to.
        :ptype second: str
        :param field: The price field.
        :ptype field: str
        :return: ``price.<first>.<second>.<gain>.<field>``, within its slice where it is read within one.
        :rtype: str
        """
        prefix = f"{self.within}." if self.within else ""
        return f"{prefix}price.{first}.{second}.{self.gain_key}.{field}"

    @property
    def steps(self) -> list[tuple[str, str]]:
        """Each move along the frontier, cheapest first: where the best option moves as the price of a point rises."""
        return list(zip(self.frontier, self.frontier[1:], strict=False))


def _at_least(first: tuple[float, float], second: tuple[float, float]) -> bool:
    """Whether ``first`` (cost, position) costs no more than ``second`` and reads no worse."""
    return first[0] <= second[0] and first[1] >= second[1]


def _undominated(options: Mapping[str, tuple[float, float]]) -> tuple[list[str], dict[str, str]]:
    """The options no other is at least as good as on both axes while ahead on one; and, for each other, the first
    option (in the given order) that puts it off. Of two alike on both axes, the first in the given order stays."""
    names = list(options)
    kept: list[str] = []
    beaten: dict[str, str] = {}
    for index, option in enumerate(names):
        for rank, other in enumerate(names):
            if other == option or not _at_least(options[other], options[option]):
                continue
            if options[other] != options[option] or rank < index:
                beaten[option] = other
                break
        else:
            kept.append(option)
    return kept, beaten


def _passed_by(cheaper: tuple[float, float], middle: tuple[float, float], dearer: tuple[float, float]) -> bool:
    """Whether a point bought on the way to ``middle`` costs at least what one bought on from it does: then the best
    option moves from ``cheaper`` to ``dearer`` without stopping at ``middle``."""
    into = (middle[0] - cheaper[0]) * (dearer[1] - middle[1])
    onward = (dearer[0] - middle[0]) * (middle[1] - cheaper[1])
    return into >= onward


def _hull(
    options: Mapping[str, tuple[float, float]], kept: Sequence[str]
) -> tuple[list[str], dict[str, tuple[str, str]]]:
    """The undominated options best at some price per point (the upper hull, cheapest first), and, for each option
    under it, the two frontier options either side of it, whose step passes it by."""
    ordered = sorted(kept, key=lambda option: options[option])
    hull: list[str] = []
    for option in ordered:
        while len(hull) >= 2 and _passed_by(options[hull[-2]], options[hull[-1]], options[option]):
            hull.pop()
        hull.append(option)
    under: dict[str, tuple[str, str]] = {}
    for option in ordered:
        if option in hull:
            continue
        spend = options[option][0]
        below = max((held for held in hull if options[held][0] < spend), key=lambda held: options[held][0])
        above = min((held for held in hull if options[held][0] > spend), key=lambda held: options[held][0])
        under[option] = (below, above)
    return hull, under


def price_frontier(
    report_input: ReportInput,
    gain_key: str,
    *,
    within: tuple[SliceKind, str] | None = None,
    admit: Sequence[str] = (),
) -> PriceFrontier:
    """The options a trade prices on one gain axis and the frontier among them: the options best at some price per
    point.

    An option is priced when it survives every gate and guardrail (:func:`~threetears.evals.analysis.report_margins.
    surviving_arms`) or is admitted as a what-if; one that does not is unpriced as ``gated`` where a figure it has
    fails, or a guardrail read breached removes it, and as ``undecided`` where a gate has no figure for it somewhere
    and none it has fails (:func:`~threetears.evals.analysis.report_margins.undecided_arms`). It is priced when it also
    holds a cost per 1,000 (of what the cost counts), and is placed on the axis (overall, :func:`gain_axis`; within a
    slice, its mean there, at the slice's own cost where the slice states one for every option it places). An option is dominated when another costs no more and reads no worse, ahead on one of the two;
    of the undominated, the frontier is the upper hull, the options best at some price per point, where an option's
    net value at a price ``p`` is ``p`` times its position less its cost.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param gain_key: The gain axis: a measure with a direction, or a judged dimension, by key.
    :ptype gain_key: str
    :param within: The one slice to price within, as its key family's word and its id (a cell by ``<row>.<column>``);
        None overall.
    :ptype within: tuple[SliceKind, str] | None
    :param admit: Options priced although a gate or guardrail removes them: a what-if, never a recommendation.
    :ptype admit: Sequence[str]
    :return: The frontier.
    :rtype: PriceFrontier
    :raises UnresolvableReference: The gain key names no measure or judged dimension, or one stating no direction.
    """
    if within is None:
        basis = _PriceBasis(axis=gain_axis(report_input, gain_key), costs=_costs_per_thousand(report_input))
    else:
        kind, slice_id = within
        basis = _slice_basis(report_input, kind, slice_id, gain_key, f"{kind}.{slice_id}.price")
    survivors = set(surviving_arms(report_input))
    undecided = set(undecided_arms(report_input))
    admitted = tuple(arm.id for arm in report_input.arms if arm.id in admit and arm.id not in survivors)
    unpriced: dict[str, Unpriced] = {}
    options: dict[str, tuple[float, float]] = {}
    for arm in report_input.arms:
        if arm.id not in survivors and arm.id not in admitted:
            unpriced[arm.id] = "undecided" if arm.id in undecided else "gated"
        elif arm.id not in basis.costs:
            unpriced[arm.id] = "no_cost"
        elif arm.id not in basis.axis.positions:
            unpriced[arm.id] = "no_gain_figure"
        else:
            options[arm.id] = (basis.costs[arm.id], basis.axis.positions[arm.id])
    kept, beaten = _undominated(options)
    hull, under = _hull(options, kept)
    return PriceFrontier(
        gain_key=gain_key,
        within=basis.within,
        axis=basis.axis,
        costs=basis.costs,
        cost_key=basis.cost_key,
        priced=tuple(options),
        admitted=admitted,
        unpriced=unpriced,
        frontier=tuple(hull),
        dominated=beaten,
        under=under,
    )


def _frontier_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    frontier = price_frontier(report_input, values[0])
    return Figure(path=path, kind="number", value=float(len(frontier.frontier)), scale="count")


def _steps_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    frontier = price_frontier(report_input, values[0])
    return Figure(path=path, kind="number", value=float(len(frontier.steps)), scale="count")


# --- calls that went unanswered, beside an arm's cost ---------------------------------------------------------------

#: What an arm's unanswered calls read beside the bare count (retried and lost together): each kind's count, the
#: answered calls beside them, their share of every call sent, the reasons counted, and the trials that counted none.
_UNANSWERED_FIELDS = ("retried", "lost", "calls", "share", "reasons", "uncounted")

#: What every unanswered key says of an arm no role counted for, and of a count some trials left out.
_UNANSWERED_RULE = (
    "Only an arm a usage role counted unanswered calls for is keyed (not counted is never none); where some trials "
    "counted none (`uncounted`), the counts read 'at least'"
)


def _unanswered_rows(report_input: ReportInput, aspect_id: str | None, arm_id: str) -> list[Unanswered]:
    """The arm's unanswered calls by role, on one aspect (the lead's where none is named), in the input's role order."""
    board = aspect_board(report_input, aspect_id)
    row = board.get(arm_id)
    return list((row.unanswered or {}).values()) if row is not None else []


def _unanswered_arms(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return [arm for arm, row in report_input.scoreboard.items() if row.unanswered]


def _aspect_unanswered_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return [arm for arm, row in aspect_board(report_input, prior[0]).items() if row.unanswered]


def _unanswered_sum(rows: Sequence[Unanswered], field: str) -> int | None:
    """One count summed over the roles that counted it; None where no role did."""
    counted = [value for row in rows if (value := getattr(row, field)) is not None]
    return sum(counted) if counted else None


def _unanswered_figure(rows: Sequence[Unanswered], field: str | None, path: str) -> Figure:
    """One figure of an arm's unanswered calls, summed over every role that counted them: the bare count (retried and
    lost), a kind's count, the answered calls, the share of every call sent, the reasons, or the trials uncounted."""
    retried, lost = _unanswered_sum(rows, "retried"), _unanswered_sum(rows, "lost")
    if field in (None, "share"):
        # Both kinds together, so a kind some role did not count is not counted there, never a zero beside the other.
        uncounted = sorted({kind for row in rows for kind in ("retried", "lost") if getattr(row, kind) is None})
        # A role whose calls are counted, that sent some and states no share: one of its rows counted one kind only,
        # which its sums hide, so the two together are not counted there either (the bundle's own reading).
        if not uncounted and any(
            row.share is None and row.calls is not None and row.calls + (row.retried or 0) + (row.lost or 0) > 0
            for row in rows
        ):
            uncounted = ["retried", "lost"]
        if uncounted:
            raise _unstated(
                path,
                f"counts the retried and lost calls together, and a role of the arm did not count its "
                f"{' or '.join(uncounted)} ones on every row: not counted; read `.retried` and `.lost` apart",
            )
    if field is None:
        return _count(path, (retried or 0) + (lost or 0))
    if field in ("retried", "lost"):
        held = retried if field == "retried" else lost
        if held is None:
            raise _unstated(path, f"names the calls {field} unanswered, which no role of the arm counted")
        return _count(path, held)
    if field == "uncounted":
        return _count(path, sum(row.uncounted for row in rows))
    if field == "reasons":
        reasons: dict[str, int] = {}
        for row in rows:
            for word, count in (row.reasons or {}).items():
                reasons[word] = reasons.get(word, 0) + count
        if not reasons:
            raise _unstated(path, "names the reasons the arm's calls went unanswered, which no role counted")
        ordered = tuple(sorted(reasons.items(), key=lambda item: (-item[1], item[0])))
        # Each count is printed by the page's forms, as a count anywhere on the page is (Figure.counts); the value a
        # tool serves beside that text keeps the grouped form it has always had.
        value = ", ".join(f"{word} {group_thousands(str(count))}" for word, count in ordered)
        return Figure(path=path, kind="text", value=value, counts=ordered)
    if any(row.calls is None for row in rows):
        raise _unstated(
            path, f"names the arm's {field}, and a role that counted unanswered calls did not count its calls"
        )
    calls = sum(row.calls or 0 for row in rows)
    if field == "calls":
        return _count(path, calls)
    sent = calls + (retried or 0) + (lost or 0)
    if not sent:
        raise _unstated(path, "names the share of the arm's calls unanswered, and it sent none")
    # One role's share is the bundle's own; summed over several roles it is computed here, from their counts.
    if len(rows) == 1 and rows[0].share is not None:
        return _number(path, rows[0].share, "the share of calls unanswered", "share")
    return replace(
        _number(path, ((retried or 0) + (lost or 0)) / sent, "the share of calls unanswered", "share"), derived=True
    )


def _unanswered(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, *field = values
    return _unanswered_figure(_unanswered_rows(report_input, None, arm_id), field[0] if field else None, path)


def _aspect_unanswered(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, arm_id, *field = values
    return _unanswered_figure(_unanswered_rows(report_input, aspect_id, arm_id), field[0] if field else None, path)


# --- two arms' costs apart ------------------------------------------------------------------------------------------

#: What two arms' costs apart read: the first's cost per 1,000 of what the cost counts minus the second's, the same per
#: month at the declared volume, and the first's cost as a multiple of the second's.
_COST_GAP_FIELDS = ("per_1k", "monthly", "multiple")

#: The unit a cost multiple is written in.
MULTIPLE_UNIT = "times"

#: The units a figure keeps under every filter, rounding included: a multiple's ``times`` is what makes it one, and
#: ``21`` alone reads as a count. A multiple's own spelling rounds to the figures a ratio supports
#: (:func:`~threetears.evals.kernel.numbers.spell_multiple`: ``21 times``, ``2.4 times``); ``int`` writes it whole.
#: Every other unit is left off under ``int``, which prose writes after the key (``{…p95|int} ms``).
_KEPT_UNITS = frozenset({MULTIPLE_UNIT})


def _cost_gap_seconds(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return [arm for arm in _costs_per_thousand(report_input) if arm != prior[-1]]


def _cost_gap(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    first, second, field = values
    costs = _costs_per_thousand(report_input)
    gap = costs[first] - costs[second]
    if field == "per_1k":
        return Figure(path=path, kind="number", value=gap, scale="usd_per_1k", change=True)
    if field == "monthly":
        return Figure(
            path=path,
            kind="number",
            value=gap * _monthly_factor(report_input, path),
            scale="usd",
            change=True,
            total=True,
        )
    if not costs[second]:
        raise _unstated(path, f"names {first}'s cost as a multiple of {second}'s, and {second} costs nothing")
    return Figure(path=path, kind="number", value=costs[first] / costs[second], scale="plain", unit=MULTIPLE_UNIT)


# --- a cost some trials did not report -----------------------------------------------------------------------------

#: What a cost some trials did not report reads: how many did not, the cost per 1,000 with each counted free, and with
#: each counted at the arm's own 95th percentile.
_UNREPORTED_FIELDS = ("missing", "free", "at_p95")


def _unreported(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    """One arm's cost read with its unreported trials counted free, or at its 95th percentile (this module's account).

    :raises UnstatedFigure: The page's cost is not read off one measure's overall figures, the measure or the
        scoreboard leaves the observations or the trials run unstated, or the observations outnumber the trials.
    """
    arm, field = values
    source = cost_basis(report_input).source
    measure = report_input.measures.get(source)
    stat = measure.per_arm.get(arm) if measure is not None else None
    if measure is None or stat is None or stat.mean is None:
        raise _unstated(path, f"names the trials {arm}'s cost leaves out, and its cost is read off no measure's figure")
    row = report_input.scoreboard.get(arm)
    ran = row.ran if row is not None else None
    if stat.n is None or ran is None:
        raise _unstated(
            path,
            f"counts the trials {arm}'s cost leaves out, and the snapshot states no {'n' if stat.n is None else 'trials run'}",
        )
    if stat.n > ran:
        raise _unstated(
            path,
            f"counts the trials {arm}'s cost leaves out, and its cost has more observations than the arm ran trials, "
            "so they are not one per trial",
        )
    missing = ran - stat.n
    if field == "missing":
        return _count(path, missing)
    factor = _PER_THOUSAND_FACTORS[measure.unit or ""]
    if field == "free":
        return Figure(path=path, kind="number", value=stat.mean * stat.n / ran * factor, scale="usd_per_1k")
    if stat.p95 is None:
        raise _unstated(
            path, f"names {arm}'s cost with its missing trials at its 95th percentile, which it does not state"
        )
    filled = (stat.mean * stat.n + stat.p95 * missing) / ran
    return Figure(path=path, kind="number", value=filled * factor, scale="usd_per_1k")


def _apart(report_input: ReportInput, path: str, key: str, value: float, what: str, *, change: bool) -> Figure:
    """A difference, distance or spread on a figure: in points for a rate, else in the figure's units."""
    scale, unit = _scale_of(report_input, key)
    if scale in ("share", "percent"):
        return _number(path, value * 100 if scale == "share" else value, what, "points", change=change)
    return _number(path, value, what, scale, unit=unit, change=change)


__all__ = [
    "PriceFrontier",
    "POOL_WEIGHT_WORDS",
    "SLICE_COST_WORDS",
    "SliceKind",
    "gain_axis",
    "pooled_figures",
    "price_frontier",
]
