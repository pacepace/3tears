"""Figure keys: the report plan's prose names a figure by its place in the report input, and code writes the number.

The planner never types a figure into a sentence (GENERATOR.md §6.2). It writes a **figure key** —
``{<path>|<filter>|<filter>}`` — and :func:`render` replaces each with the value read from the
:class:`~threetears.evals.analysis.report_input.ReportInput`, spelled by the package's number rule
(:mod:`threetears.evals.analysis.numbers`, :mod:`threetears.evals.analysis.quantities`). ``{arm:<id>}`` prints an
arm's short name with its letter tag.

**Separate from the memo's figure references.** The memo writes ``{{<cell>|<measure>|<reading>|<stat>}}`` against a
decision surface (:mod:`threetears.evals.analysis.prose_refs`). A plan writes single braces against the report
input. The two never mix: a double brace in plan prose is refused, not read as a key.

**A path is matched, never split.** Ids carry dots (an arm ``qwen3.8-flash``, a dimension
``tone.kept``) and spaces, so a path is matched against a closed vocabulary built from the input: a family keyword,
then the ids that family takes (arms, aspects, measures, dimensions, strata, read from the input), then a statistic
from a fixed set. Every complete reading of the path is found. One is the figure. Two are an ambiguity, refused
naming both. None is refused naming the valid continuations at the point the path stopped matching. Each refusal is
an :class:`~threetears.evals.analysis.errors.UnresolvableReference`, so the planner's repair round can correct it.

**Derived values are descriptive.** ``cost.uplift.<arm>``, ``cost.uplift_range``, ``strata.<id>.loss_share`` and
``flip_margin.<arm>`` are computed here from the input's point figures. They carry no interval and no test, and the
grammar says so. Everything else is copied from the input.

**Digits only inside keys (D3).** :func:`literal_numbers` finds every digit that plan prose types outside a key,
outside a name the input holds, outside a percentile's name (``p95``) and outside one of the package's own unit
phrases (``per 1,000 replies``). A number a reader sees is code's.

**The grammar is taught by this module.** :func:`figure_key_grammar` writes the planner prompt's section from the
module's own families, statistics and filters, with examples rendered on a synthetic input, so the prompt can never
teach a form this parser refuses.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.quantities import MONEY, PERCENT, POINTS, number_in, signed_with_unit, with_unit
from threetears.evals.analysis.report.figures import PER_MILLION_REPLIES, PER_THOUSAND_LABEL, PER_THOUSAND_REPLIES
from threetears.evals.analysis.report.words import arm_tags
from threetears.evals.analysis.report_input import (
    Arm,
    Aspect,
    Criterion,
    Judged,
    Measure,
    Reading,
    ReportInput,
    Stat,
    Stratum,
    Task,
)
from threetears.evals.analysis.report_margins import Margin, flip_margin, weakest_margin
from threetears.evals.analysis.style import humanize, measure_name
from threetears.evals.analysis.summary import dollars_text

#: What a figure is: a number, an interval (low, high), a piece of text, or an arm (rendered by name and tag).
FigureKind = Literal["number", "interval", "text", "arm"]

#: The scale a figure's number is on, which decides the filters it takes. ``share``: 0 to 1. ``percent``: 0 to 100.
#: ``points``: a difference already in percentage points. ``usd``: dollars per reply or per unit. ``usd_per_1k``:
#: dollars per 1,000 replies. ``usd_per_1m``: dollars per 1,000,000 replies. ``count``: a count. ``plain``: any other
#: number, in ``Figure.unit`` when it has one.
Scale = Literal["share", "percent", "points", "usd", "usd_per_1k", "usd_per_1m", "count", "plain"]

#: The dollar scales, each with the factor that states its figure per 1,000 replies.
_PER_THOUSAND_FACTORS: dict[str, float] = {"usd": 1000.0, "usd_per_1k": 1.0, "usd_per_1m": 0.001}

#: The filters a key may carry. One spelling at most (``pct``, ``pts``, ``usd``, ``usd_per_1k``, ``int``,
#: ``pct10``), plus ``abs`` and ``interval``, in any order.
Filter = Literal["pct", "pts", "abs", "usd", "int", "pct10", "interval", "usd_per_1k"]
_FILTERS: tuple[str, ...] = get_args(Filter)
_SPELLINGS: frozenset[str] = frozenset({"pct", "pts", "usd", "usd_per_1k", "int", "pct10"})

#: How each filter reads, as the grammar teaches it.
_FILTER_MEANS: dict[str, str] = {
    "pct": "a rate as a percent",
    "pts": "a change in points, signed, one decimal at most",
    "abs": "the magnitude, without its sign",
    "usd": "dollars",
    "int": "the nearest whole number, its thousands separated",
    "pct10": "a rate rounded to the nearest 10%",
    "interval": "an interval's two ends, joined by 'to'",
    "usd_per_1k": "dollars per 1,000 replies, with the words",
}

#: The unit phrases a rendered figure is joined with, one template each, so a translation replaces them here.
INTERVAL_TEMPLATE = "{low} to {high}"
RANGE_TEMPLATE = "{low}–{high}"
PER_THOUSAND_TEMPLATE = "{amount} " + PER_THOUSAND_REPLIES
_PER_MILLION_TEMPLATE = "{amount} " + PER_MILLION_REPLIES
ARM_TEMPLATE = "{name} ({tag})"
#: What separates a whole number's thousands under the ``int`` filter (``12,000,000``). Prose only: the package's
#: number rule (:mod:`threetears.evals.analysis.numbers`) writes no separator inside its tables' intervals and lists.
THOUSANDS_SEPARATOR = ","
#: How a criterion's direction reads.
DIRECTION_WORDS = {"higher_is_better": "higher is better", "lower_is_better": "lower is better"}

#: The percentile names prose may write with their digits (``p95 latency``): the statistics a criterion or a measure
#: reads, and the other common ones. A fixed set, never read from the input's free text.
PERCENTILE_NAMES: tuple[str, ...] = ("p50", "p90", "p95", "p99")

#: The package's own unit phrases, which prose may write whole with their digits: each one a template a rendered
#: figure is joined with, so prose that says "per 1,000 replies" uses the package's words and no figure of its own.
_UNIT_PHRASES: tuple[str, ...] = (PER_THOUSAND_REPLIES, PER_THOUSAND_LABEL, PER_MILLION_REPLIES)

#: A key, as plan prose holds it: one pair of single braces around anything but braces.
_KEY = re.compile(r"\{([^{}]*)\}")

#: The arm form's prefix, ``{arm:<id>}``.
_ARM_PREFIX = "arm:"

#: A run of digits plan prose typed, with any decimal or thousands separators inside it. ``\d`` is every Unicode
#: decimal digit (category Nd), so an Arabic-Indic or Devanagari figure is caught as surely as an ASCII one.
_DIGITS = re.compile(r"\d(?:[.,]?\d)*")

#: How many valid continuations a refusal lists before it stops counting.
_CONTINUATIONS_SHOWN = 24


@dataclass(frozen=True)
class FigureKey:
    """One key as plan prose wrote it: its path, its filters, and where it sits in the text (None when parsed alone)."""

    path: str
    filters: tuple[str, ...] = ()
    span: tuple[int, int] | None = None


@dataclass(frozen=True)
class Figure:
    """What a key resolves to: the value, what kind of value it is, and the scale its number is on.

    ``change`` marks a difference (a reading's change, a contrast's delta): it reads signed and in points, never as a
    percent. ``whole`` marks a descriptive ratio (a cost uplift, a reasoning share, a stratum's loss share), read in
    whole percents because no interval stands behind a decimal.
    """

    path: str
    kind: FigureKind
    value: float | tuple[float, float] | str
    scale: Scale = "plain"
    unit: str = ""
    change: bool = False
    whole: bool = False


# --- the vocabulary a path is matched against -------------------------------------------------------------------


@dataclass(frozen=True)
class _Word:
    """A fixed word in a path: a family keyword or a sub-field."""

    text: str


@dataclass(frozen=True)
class _Ids:
    """A slot an id fills: its name in the grammar, and the ids the input offers given the slots matched before it."""

    name: str
    options: Callable[[ReportInput, tuple[str, ...]], Iterable[str]]


@dataclass(frozen=True)
class _Statistic:
    """A slot a statistic (or a field) from a fixed set fills, and its name in the grammar."""

    options: tuple[str, ...]
    name: str = "<statistic>"


_Slot = _Word | _Ids | _Statistic


@dataclass(frozen=True)
class _Family:
    """One path family: its slots in order, what it reads, and how its matched ids and statistic become a figure."""

    slots: tuple[_Slot, ...]
    what: str
    resolve: Callable[[ReportInput, tuple[str, ...], str], Figure]
    derived: bool = False

    def form(self) -> str:
        return ".".join(slot.text if isinstance(slot, _Word) else slot.name for slot in self.slots)

    def statistic(self) -> _Statistic | None:
        return next((slot for slot in self.slots if isinstance(slot, _Statistic)), None)


def _refuse(path: str, why: str) -> UnresolvableReference:
    return UnresolvableReference(f"figure key {{{path}}} {why}")


def _arm(report_input: ReportInput, arm_id: str) -> Arm:
    return next(arm for arm in report_input.arms if arm.id == arm_id)


def _control(report_input: ReportInput) -> Arm | None:
    return next((arm for arm in report_input.arms if arm.role == "control"), None)


def _candidates(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return [arm.id for arm in report_input.arms if arm.role == "challenger"]


def _asked(report_input: ReportInput) -> Aspect | None:
    """The one aspect the campaign asked about: the only aspect marked asked, else the only aspect; None otherwise."""
    asked = [aspect for aspect in report_input.aspects if aspect.asked]
    if len(asked) == 1:
        return asked[0]
    if not asked and len(report_input.aspects) == 1:
        return report_input.aspects[0]
    return None


def _asked_or_refuse(report_input: ReportInput, path: str) -> Aspect:
    aspect = _asked(report_input)
    if aspect is None:
        ids = ", ".join(f"`aspects.{a.id}.`" for a in report_input.aspects) or "none: the input holds no aspect"
        raise _refuse(path, f"reads the one asked aspect, and this input has no single one; name it: {ids}")
    return aspect


def _aspect_by_id(report_input: ReportInput, aspect_id: str) -> Aspect:
    return next(aspect for aspect in report_input.aspects if aspect.id == aspect_id)


def _reading(aspect: Aspect, arm_id: str) -> Reading:
    return next(reading for reading in aspect.readings if reading.arm == arm_id)


def _rate_scale(values: Iterable[float | None]) -> Scale:
    """A rate's scale read from its figures: shares when every one lies in 0 to 1, percents otherwise.

    The input copies each rate as the pipeline gave it, and a judged pass rate carries no unit, so its scale is read
    from the values themselves.
    """
    present = [value for value in values if value is not None]
    return "share" if all(0 <= value <= 1 for value in present) else "percent"


def _measure_scale(measure: Measure) -> tuple[Scale, str]:
    unit = measure.unit or ""
    if unit == "pct":
        return "percent", ""
    if unit == "usd":
        return "usd", ""
    if unit == "usd_per_1k":
        return "usd_per_1k", ""
    if unit == "usd_per_1m":
        return "usd_per_1m", ""
    if not unit and measure.grade in ("pass_fail", "class_label"):
        return "share", ""
    return "plain", unit


def _judged_scale(judged: Judged) -> Scale:
    if judged.scale == "pass_fail":
        return _rate_scale(stat.mean for stat in judged.per_arm.values())
    return "plain"


def _contrast_scale(report_input: ReportInput, measure: str, kind: str | None) -> tuple[Scale, str]:
    if measure in report_input.measures:
        return _measure_scale(report_input.measures[measure])
    if measure in report_input.judged:
        return _judged_scale(report_input.judged[measure]), ""
    if kind == "trial":
        return "share", ""
    return "plain", ""


def _text(path: str, value: str | None, what: str) -> Figure:
    if value is None or not value.strip():
        raise _refuse(path, f"names {what}, which the input does not state")
    return Figure(path=path, kind="text", value=value)


def _number(
    path: str,
    value: float | None,
    what: str,
    scale: Scale,
    *,
    unit: str = "",
    change: bool = False,
    whole: bool = False,
) -> Figure:
    if value is None:
        raise _refuse(path, f"names {what}, which the input does not state")
    return Figure(path=path, kind="number", value=float(value), scale=scale, unit=unit, change=change, whole=whole)


def _end(path: str, interval: tuple[float, float] | None, statistic: str, what: str) -> float:
    """One end of ``interval`` (``low`` or ``high``), refused by name where the input states no interval."""
    if interval is None:
        raise _refuse(path, f"names {what}, which the input does not state")
    return interval[0] if statistic == "low" else interval[1]


def _interval(
    path: str,
    value: tuple[float, float] | None,
    what: str,
    scale: Scale,
    *,
    unit: str = "",
    change: bool = False,
    whole: bool = False,
) -> Figure:
    if value is None:
        raise _refuse(path, f"names {what}, which the input does not state")
    low, high = value
    return Figure(
        path=path, kind="interval", value=(float(low), float(high)), scale=scale, unit=unit, change=change, whole=whole
    )


# --- what each family reads ------------------------------------------------------------------------------------

_META_FIELDS = (
    "campaign",
    "question",
    "behavior",
    "kind",
    "kind_means",
    "design_shape",
    "run_date",
    "provenance",
    "volume_per_month",
)


def _meta(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (field,) = values
    if field == "volume_per_month":
        return _number(path, report_input.meta.volume_per_month, "the monthly volume", "count")
    value = getattr(report_input.meta, field)
    return _text(path, value, f"the campaign's {field}")


_ARM_FIELDS = ("id", "name", "served_model", "description")


def _arm_field(arm: Arm, field: str, path: str) -> Figure:
    if field == "name":
        return _text(path, arm.name or arm.id, "the arm's name")
    return _text(path, getattr(arm, field), f"the arm's {field}")


def _control_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    control = _control(report_input)
    if control is None:
        raise _refuse(path, "names the control, and this campaign has none")
    return _arm_field(control, values[0], path)


def _n_candidates(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    return Figure(path=path, kind="number", value=float(len(_candidates(report_input))), scale="count")


_ASPECT_FIELDS = ("id", "name", "margin", "description")


def _aspect_field(aspect: Aspect, field: str, path: str) -> Figure:
    if field == "name":
        return _text(path, aspect.name or humanize("dimension", aspect.id), "the aspect's name")
    if field == "margin":
        return _number(path, aspect.margin, "the aspect's margin", "points")
    return _text(path, getattr(aspect, field), f"the aspect's {field}")


def _asked_aspect_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    return _aspect_field(_asked_or_refuse(report_input, path), values[0], path)


def _named_aspect_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, field = values
    return _aspect_field(_aspect_by_id(report_input, aspect_id), field, path)


_READING_STATISTICS = (
    "change",
    "interval",
    "low",
    "high",
    "margin",
    "level",
    "verdict",
    "control_passed",
    "arm_passed",
    "why",
)


def _reading_figure(reading: Reading, statistic: str, path: str) -> Figure:
    if statistic == "change":
        return _number(path, reading.change, "the change", "points", change=True)
    if statistic == "interval":
        return _interval(path, reading.within_interval, "the interval", "points", change=True)
    if statistic in ("low", "high"):
        end = _end(path, reading.within_interval, statistic, "the interval")
        return _number(path, end, f"the interval's {statistic} end", "points", change=True)
    if statistic == "margin":
        return _number(path, reading.margin, "the reading's margin", "points")
    if statistic == "level":
        return _number(path, reading.gain_confidence, "the interval's level", "share", whole=True)
    if statistic == "verdict":
        return _text(path, reading.reading, "the reading")
    return _text(path, getattr(reading, statistic), f"the reading's {statistic}")


def _reading_arms(aspect: Aspect | None) -> list[str]:
    return [reading.arm for reading in aspect.readings] if aspect else []


def _asked_reading(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, statistic = values
    return _reading_figure(_reading(_asked_or_refuse(report_input, path), arm_id), statistic, path)


def _named_reading(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, arm_id, statistic = values
    return _reading_figure(_reading(_aspect_by_id(report_input, aspect_id), arm_id), statistic, path)


_CONTRAST_STATISTICS = ("delta", "interval", "low", "high", "verdict", "p_adjusted")


def _contrasted(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(c.measure for c in report_input.contrasts if c.arm == prior[0]))


def _contrast(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, measure, statistic = values
    contrast = next(c for c in report_input.contrasts if c.arm == arm_id and c.measure == measure)
    scale, unit = _contrast_scale(report_input, measure, contrast.kind)
    if statistic == "delta":
        return _number(path, contrast.delta, "the delta", scale, unit=unit, change=True)
    if statistic == "interval":
        return _interval(path, contrast.interval, "the interval", scale, unit=unit, change=True)
    if statistic in ("low", "high"):
        end = _end(path, contrast.interval, statistic, "the interval")
        return _number(path, end, f"the interval's {statistic} end", scale, unit=unit, change=True)
    if statistic == "verdict":
        return _text(path, contrast.verdict, "the verdict")
    return _number(path, contrast.p_adjusted, "the adjusted p", "plain")


def _judged_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(report_input.judged[prior[0]].per_arm)


def _judged(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    dimension, arm_id = values
    judged = report_input.judged[dimension]
    return _number(path, judged.per_arm[arm_id].mean, "the dimension's mean", _judged_scale(judged))


_JUDGE_FIELDS = ("model", "tier", "calibration", "calibration_ratings", "agreement")
_JUDGED_ATTRIBUTES = {"model": "judge_model"}


def _judge_value(judged: Judged, field: str, path: str) -> Figure:
    value = getattr(judged, _JUDGED_ATTRIBUTES.get(field, field))
    if field == "calibration_ratings":
        return _number(path, value, "the judge's calibration ratings", "count")
    if field == "agreement":
        return _number(path, value, "the judge's agreement", "share", whole=True)
    return _text(path, value, f"the judge's {field}")


def _judged_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    dimension, field = values
    judged = report_input.judged[dimension]
    if field == "label":
        return _text(path, reader_name("judged", dimension, report_input), "the dimension's label")
    return _judge_value(judged, field, path)


def _judge(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (field,) = values
    attribute = _JUDGED_ATTRIBUTES.get(field, field)
    stated = {dimension: getattr(judged, attribute) for dimension, judged in report_input.judged.items()}
    distinct = {value for value in stated.values() if value is not None}
    if not distinct:
        raise _refuse(path, f"names the judge's {field}, which no judged dimension states")
    if len(distinct) > 1:
        per = ", ".join(f"`judged.{dimension}.{field}`" for dimension, value in stated.items() if value is not None)
        raise _refuse(path, f"names one judge {field}, and the judged dimensions differ; name one: {per}")
    dimension = next(dimension for dimension, value in stated.items() if value is not None)
    return _judge_value(report_input.judged[dimension], field, path)


def _passing(report_input: ReportInput) -> dict[str, float]:
    """Each arm's pass rate: passed over usable from the scoreboard, else the input's ``pass_rate`` measure."""
    board = {
        arm: row.passed / row.usable
        for arm, row in report_input.scoreboard.items()
        if row.passed is not None and row.usable
    }
    if board:
        return board
    measure = report_input.measures.get("pass_rate")
    if measure is None:
        return {}
    return {arm: float(stat.mean) for arm, stat in measure.per_arm.items() if stat.mean is not None}


def _pass_rate_scale(report_input: ReportInput) -> Scale:
    if any(row.passed is not None and row.usable for row in report_input.scoreboard.values()):
        return "share"
    return _measure_scale(report_input.measures["pass_rate"])[0]


def _pass_rate(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    return _number(path, _passing(report_input)[values[0]], "the pass rate", _pass_rate_scale(report_input))


def _cost_measures(report_input: ReportInput) -> tuple[list[str], list[str]]:
    """The measures an arm's cost may be read from, in the order they are tried: each one alone of its kind.

    First the measures read before the scoreboard: the one stated per 1,000 replies, the one stated per 1,000,000,
    the one in dollars on the cost merit axis. Then the one dollar measure, in any of those units, where lower is
    better, read after it.
    """
    measures = report_input.measures

    def alone(keys: list[str]) -> list[str]:
        return keys if len(keys) == 1 else []

    first = [
        *alone([key for key, m in measures.items() if m.unit == "usd_per_1k"]),
        *alone([key for key, m in measures.items() if m.unit == "usd_per_1m"]),
        *alone([key for key, m in measures.items() if m.unit == "usd" and m.merit_axis == "cost"]),
    ]
    lower = alone(
        [key for key, m in measures.items() if m.unit in _PER_THOUSAND_FACTORS and m.higher_is_better is False]
    )
    return first, lower


def _overall_cost(report_input: ReportInput, key: str) -> dict[str, float]:
    measure = report_input.measures[key]
    factor = _PER_THOUSAND_FACTORS[measure.unit or ""]
    return {arm: stat.mean * factor for arm, stat in measure.per_arm.items() if stat.mean is not None}


def _blended_cost(report_input: ReportInput, key: str) -> dict[str, float]:
    """Each arm's cost on one measure blended over the tasks: its task figures weighted by each task's traffic share.

    An arm is blended over the tasks that carry traffic and hold its figure, each weighted by its share of their
    traffic, and left out where any of those figures is unstated.
    """
    factor = _PER_THOUSAND_FACTORS[report_input.measures[key].unit or ""]
    blended: dict[str, float] = {}
    for arm in report_input.arms:
        parts = [
            (float(task.share), task.per_arm[arm.id][key].mean)
            for task in report_input.tasks
            if task.share and arm.id in task.per_arm and key in task.per_arm[arm.id]
        ]
        means = [mean for _, mean in parts if mean is not None]
        if parts and len(means) == len(parts):
            total = sum(share for share, _ in parts)
            blended[arm.id] = factor * sum(share * mean for (share, _), mean in zip(parts, means, strict=True)) / total
    return blended


def _costs_per_thousand(report_input: ReportInput) -> dict[str, float]:
    """Each arm's cost per 1,000 replies, from the first source that holds one: the one rule a page and the decision
    read cost by (``cost_per_1k.<arm>``).

    The arms' overall figures on a measure stated per 1,000 replies; else on one stated per 1,000,000, divided by
    1,000; else on the one dollar measure on the cost merit axis, times 1,000; else the scoreboard's cost per reply,
    times 1,000; else the one dollar measure where lower is better. Where none of those states a figure, each measure
    in the same order blended over the tasks by traffic share (:func:`_blended_cost`).
    """
    first, lower = _cost_measures(report_input)
    sources = [_overall_cost(report_input, key) for key in first]
    sources.append(
        {a: row.reply_cost_usd * 1000 for a, row in report_input.scoreboard.items() if row.reply_cost_usd is not None}
    )
    sources += [_overall_cost(report_input, key) for key in lower]
    sources += [_blended_cost(report_input, key) for key in dict.fromkeys([*first, *lower])]
    return next((source for source in sources if source), {})


def _cost_per_1k(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    return _number(path, _costs_per_thousand(report_input)[values[0]], "the cost", "usd_per_1k")


_MEASURE_STATISTICS = ("mean", "ci", "half_width", "sem", "n", "n_cases", "p50", "p95", "change")


def _measure_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(report_input.measures[prior[0]].per_arm)


def _measure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    key, arm_id, statistic = values
    measure = report_input.measures[key]
    stat = measure.per_arm[arm_id]
    scale, unit = _measure_scale(measure)
    if statistic in ("n", "n_cases"):
        return _number(path, getattr(stat, statistic), f"the measure's {statistic}", "count")
    if statistic == "ci":
        return _interval(path, stat.ci, "the measure's interval", scale, unit=unit)
    return _number(
        path, getattr(stat, statistic), f"the measure's {statistic}", scale, unit=unit, change=statistic == "change"
    )


_SCOREBOARD_STATISTICS = ("ran", "usable", "passed", "cases", "lost", "failed", "reply_cost_usd", "test_cost_usd")


def _scoreboard(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, statistic = values
    row = report_input.scoreboard[arm_id]
    if statistic in ("lost", "failed"):
        return _number(path, sum(getattr(row, statistic).values()), f"trials {statistic}", "count")
    if statistic.endswith("_usd"):
        return _number(path, getattr(row, statistic), f"the {statistic}", "usd")
    return _number(path, getattr(row, statistic), f"trials {statistic}", "count")


def _reasoning_arms(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [arm.id for arm in report_input.arms if arm.reasoning_ratio is not None]


def _reasoning_ratio(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    ratio = _arm(report_input, values[0]).reasoning_ratio
    return _number(path, ratio, "the share of output spent reasoning", "share", whole=True)


def _depth_figure(report_input: ReportInput, path: str, read: Callable[[ReportInput, Aspect], float | None]) -> Figure:
    value = read(report_input, _asked_or_refuse(report_input, path))
    if value is None:
        raise _refuse(path, "reads the asked aspect's depth, and its arms ran it to different depths or none is stated")
    return Figure(path=path, kind="number", value=float(value), scale="count")


def _depth_of(aspect: Aspect, report_input: ReportInput) -> list[tuple[int, float]]:
    """``(cases, mean repeats)`` for the control where it ran the aspect, else for every arm that did."""
    depth = aspect.depth or {}
    control = _control(report_input)
    if control is not None and control.id in depth:
        return [(depth[control.id].cases, float(depth[control.id].repeats_per_case[1]))]
    return [(d.cases, float(d.repeats_per_case[1])) for d in depth.values() if len(d.repeats_per_case) > 1]


def _agreed(values: Iterable[float]) -> float | None:
    distinct = set(values)
    return distinct.pop() if len(distinct) == 1 else None


def _cases(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    return _depth_figure(report_input, path, lambda ri, a: _agreed(c for c, _r in _depth_of(a, ri)))


def _repeats(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    return _depth_figure(report_input, path, lambda ri, a: _agreed(r for _c, r in _depth_of(a, ri)))


def _late_set_at(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    when = list(dict.fromkeys(late.set_at for late in report_input.caveats.late_criteria))
    if not when:
        raise _refuse(path, "names when the late criteria were set, and no criterion was set late")
    if len(when) > 1:
        raise _refuse(path, f"names one date, and the late criteria were set at {len(when)} different times")
    return Figure(path=path, kind="text", value=when[0])


_FAMILY_FIELDS = ("disclosure", "correction", "alpha", "family_size", "level")


def _family(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (field,) = values
    if len(report_input.contrast_families) != 1:
        held = len(report_input.contrast_families)
        raise _refuse(path, f"reads the one contrast family, and the input holds {held}")
    family = report_input.contrast_families[0]
    if field == "alpha":
        return _number(path, family.alpha, "the family's alpha", "plain")
    if field == "family_size":
        return _number(path, family.family_size, "the family's size", "count")
    return _text(path, getattr(family, field), f"the family's {field}")


def _latency(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    return _text(path, report_input.caveats.latency, "why latency is not compared")


def _held_values(report_input: ReportInput, key: str) -> list[float | None]:
    """Every mean the input holds for a measure or judged dimension: overall, in each task and in each cell."""
    stats: list[Stat] = []
    for table in (report_input.measures, report_input.judged):
        if key in table:
            stats.extend(table[key].per_arm.values())
    for per_arm in [task.per_arm for task in report_input.tasks] + [
        cell.per_arm for cell in report_input.cells.values()
    ]:
        stats.extend(figures[key] for figures in per_arm.values() if key in figures)
    return [stat.mean for stat in stats]


def _scale_of(report_input: ReportInput, key: str, extra: Iterable[float] = ()) -> tuple[Scale, str]:
    """The scale a measure or judged dimension's figures are on wherever they sit: a measure's unit, a pass-fail
    dimension's rate scale read from every figure it holds (and ``extra``, a threshold on it), else plain."""
    if key in report_input.measures:
        return _measure_scale(report_input.measures[key])
    judged = report_input.judged.get(key)
    if judged is not None and judged.scale == "pass_fail":
        return _rate_scale([*_held_values(report_input, key), *extra]), ""
    return "plain", ""


_TASK_FIELDS = ("name", "n_cases", "share")


def _task(report_input: ReportInput, task_id: str) -> Task:
    return next(task for task in report_input.tasks if task.id == task_id)


def _tasks(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [task.id for task in report_input.tasks]


def _task_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(_task(report_input, prior[0]).per_arm)


def _task_measures(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    task_id, arm_id = prior
    return list(_task(report_input, task_id).per_arm[arm_id])


def _task_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    task_id, field = values
    task = _task(report_input, task_id)
    if field == "name":
        return _text(path, task.name or humanize("dimension", task.id), "the task's name")
    if field == "n_cases":
        return _number(path, task.n_cases, "the task's cases", "count")
    return _number(path, task.share, "the task's share of traffic", "share", whole=True)


def _task_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    task_id, arm_id, key, statistic = values
    stat = _task(report_input, task_id).per_arm[arm_id][key]
    scale, unit = _scale_of(report_input, key)
    what = f"the task's {statistic} on {key}"
    if statistic in ("n", "n_cases"):
        return _number(path, getattr(stat, statistic), what, "count")
    if statistic == "ci":
        return _interval(path, stat.ci, what, scale, unit=unit)
    return _number(path, getattr(stat, statistic), what, scale, unit=unit, change=statistic == "change")


_CRITERION_FIELDS = ("threshold", "measure", "statistic", "direction")


def _criterion(report_input: ReportInput, criterion_id: str) -> Criterion:
    return next(criterion for criterion in report_input.criteria if criterion.id == criterion_id)


def _criterion_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion_id, field = values
    criterion = _criterion(report_input, criterion_id)
    if field == "threshold":
        threshold = float(criterion.threshold)
        scale, unit = _scale_of(report_input, criterion.measure, (threshold,))
        return _number(path, threshold, "the criterion's threshold", scale, unit=unit)
    if field == "measure":
        return _text(path, _measure_reader_name(criterion.measure, report_input), "the criterion's measure")
    if field == "statistic":
        return _text(path, criterion.statistic, "the criterion's statistic")
    return _text(path, DIRECTION_WORDS[criterion.direction], "the criterion's direction")


# --- derived values: descriptive, computed here from the input's point figures ----------------------------------


def _control_cost(report_input: ReportInput, path: str) -> tuple[dict[str, float], float]:
    costs = _costs_per_thousand(report_input)
    control = _control(report_input)
    if control is None or not costs.get(control.id):
        raise _refuse(path, "compares a cost with the control's, and the input holds no control cost")
    return costs, costs[control.id]


def _costed_candidates(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    costs = _costs_per_thousand(report_input)
    return [arm for arm in _candidates(report_input) if arm in costs]


def _uplift(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    costs, control = _control_cost(report_input, path)
    return Figure(path=path, kind="number", value=costs[values[0]] / control - 1, scale="share", whole=True)


def _uplift_range(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    costs, control = _control_cost(report_input, path)
    uplifts = [costs[arm] / control - 1 for arm in _candidates(report_input) if arm in costs]
    if not uplifts:
        raise _refuse(path, "spans the candidates' cost uplifts, and no candidate has a cost")
    return Figure(path=path, kind="interval", value=(min(uplifts), max(uplifts)), scale="share", whole=True)


_STRATUM_FIELDS = ("name", "n_cases", "share")


def _stratum(report_input: ReportInput, stratum_id: str) -> Stratum:
    return next(stratum for stratum in report_input.strata if stratum.id == stratum_id)


def _stratum_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    stratum_id, field = values
    stratum = _stratum(report_input, stratum_id)
    if field == "name":
        return _text(path, stratum.name or humanize("dimension", stratum.id), "the stratum's name")
    if field == "n_cases":
        return _number(path, stratum.n_cases, "the stratum's cases", "count")
    return _number(path, stratum.share, "the stratum's share of traffic", "share", whole=True)


def _loss_share_of(report_input: ReportInput, stratum_id: str, arm_id: str, path: str) -> float:
    """The share of ``arm_id``'s loss against the control, on the primary judged dimension, that one stratum carries.

    Loss in a stratum is the control's rate minus the arm's, weighted by the stratum's cases (by its traffic share
    where cases are not stated), over the strata of the same stratification.
    """
    dimension = report_input.primary_judged
    control = _control(report_input)
    if dimension is None or control is None:
        raise _refuse(path, "reads loss on the primary judged dimension against the control; the input lacks one")
    target = _stratum(report_input, stratum_id)
    group = [s for s in report_input.strata if s.dimension == target.dimension]
    weights: list[float | None] = [s.n_cases for s in group]
    if any(weight is None for weight in weights):
        weights = [s.share for s in group]
    if any(weight is None for weight in weights):
        raise _refuse(path, "weights each stratum by its cases or its share, and a stratum states neither")
    losses: dict[str, float] = {}
    for stratum, weight in zip(group, weights, strict=True):
        figures = stratum.per_arm
        held = figures.get(control.id, {}).get(dimension), figures.get(arm_id, {}).get(dimension)
        if held[0] is None or held[1] is None or held[0].mean is None or held[1].mean is None:
            raise _refuse(path, f"needs both arms' {dimension} in stratum {stratum.id}, which the input lacks")
        losses[stratum.id] = float(weight or 0) * (held[0].mean - held[1].mean)
    total = sum(losses.values())
    if total <= 0:
        raise _refuse(path, f"divides by {arm_id}'s loss on {dimension}, and it lost nothing overall")
    return losses[stratum_id] / total


def _loss_share(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    stratum_id = values[0]
    arms = values[1:] or tuple(_candidates(report_input))
    if not arms:
        raise _refuse(path, "reads each candidate's loss, and the campaign has no candidate")
    share = min(_loss_share_of(report_input, stratum_id, arm, path) for arm in arms)
    return Figure(path=path, kind="number", value=share, scale="share", whole=True)


def _flip_margin(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    reading = _reading(_asked_or_refuse(report_input, path), values[0])
    flip = flip_margin(reading)
    if flip is None:
        raise _refuse(path, "names the interval a flip is read on, which the input does not state")
    return Figure(path=path, kind="number", value=flip, scale="points")


# --- reader names: the one name a reader is shown for each id -----------------------------------------------------

#: The kinds of id a reader is given a name for (GENERATOR §6.3: reader names, not identifiers).
NamedKind = Literal["aspect", "measure", "judged", "stratum", "task", "criterion"]
NAMED_KINDS: tuple[NamedKind, ...] = get_args(NamedKind)


def _measure_reader_name(key: str, report_input: ReportInput) -> str:
    """A measure's or judged dimension's reader name, whichever the key names."""
    if key in report_input.measures:
        return reader_name("measure", key, report_input)
    if key in report_input.judged:
        return reader_name("judged", key, report_input)
    return measure_name(key)


def reader_name(kind: NamedKind, identifier: str, report_input: ReportInput) -> str:
    """The one name a reader is shown for an id: the name the input declares, else the package's reader name for it.

    An aspect, stratum or task by its declared name; a measure by its label (the catalog's reader name), a judged
    dimension by its label, a criterion by its label; each, where none is declared, as the package names it otherwise
    (:func:`~threetears.evals.analysis.style.measure_name` for a measure, the id with its underscores and dots as
    spaces for anything else, :func:`~threetears.evals.analysis.style.humanize`), and a criterion by its measure's
    reader name. The ``.name`` and ``.label`` figure keys print the same name.

    :param kind: What the id names.
    :ptype kind: NamedKind
    :param identifier: The id, as the input keys it.
    :ptype identifier: str
    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The reader name.
    :rtype: str
    :raises KeyError: The input holds no such id of that kind.
    """
    if kind == "aspect":
        aspect = _aspect_by_id(report_input, identifier)
        return aspect.name or humanize("dimension", aspect.id)
    if kind == "measure":
        return report_input.measures[identifier].label or measure_name(identifier)
    if kind == "judged":
        return report_input.judged[identifier].label or humanize("dimension", identifier)
    if kind == "stratum":
        stratum = _stratum(report_input, identifier)
        return stratum.name or humanize("dimension", stratum.id)
    if kind == "task":
        task = _task(report_input, identifier)
        return task.name or humanize("dimension", task.id)
    criterion = _criterion(report_input, identifier)
    return criterion.label or _measure_reader_name(criterion.measure, report_input)


def reader_names(report_input: ReportInput) -> dict[NamedKind, dict[str, str]]:
    """Every id's reader name (:func:`reader_name`), by kind, then id, in the input's order.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Aspects, measures, judged dimensions, strata, tasks and criteria, each kind present only where the input
        holds one.
    :rtype: dict[NamedKind, dict[str, str]]
    """
    ids: dict[NamedKind, list[str]] = {
        "aspect": [aspect.id for aspect in report_input.aspects],
        "measure": list(report_input.measures),
        "judged": list(report_input.judged),
        "stratum": [stratum.id for stratum in report_input.strata],
        "task": [task.id for task in report_input.tasks],
        "criterion": [criterion.id for criterion in report_input.criteria],
    }
    return {
        kind: {identifier: reader_name(kind, identifier, report_input) for identifier in dict.fromkeys(held)}
        for kind, held in ids.items()
        if held
    }


# --- the weakest margin --------------------------------------------------------------------------------------------

#: What every weakest-margin key begins with.
WEAKEST_PREFIX = "weakest."
_WEAKEST_FIELDS = ("margin", "arm", "criterion", "value", "threshold")


def _cell_name(report_input: ReportInput, cell_id: str) -> str:
    cell = report_input.cells[cell_id]
    matrix = report_input.target_matrix
    if matrix is None:
        return cell_id
    row = next((level.name or level.id for level in matrix.rows.levels if level.id == cell.row), cell.row)
    column = next((level.name or level.id for level in matrix.columns.levels if level.id == cell.column), cell.column)
    return f"{row} × {column}"


def margin_name(margin: Margin, report_input: ReportInput) -> str:
    """What a margin is read against, in reader words: a criterion by its reader name and where its figure sits, or an
    aspect's margin or floor by the aspect's reader name.

    :param margin: One margin (:mod:`~threetears.evals.analysis.report_margins`).
    :ptype margin: Margin
    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The words ``{weakest.criterion}`` prints.
    :rtype: str
    """
    if margin.kind == "criterion":
        name = reader_name("criterion", margin.source, report_input)
        if margin.task is not None:
            return f"{name} in {reader_name('task', margin.task, report_input)}"
        if margin.cell is not None:
            return f"{name} in {_cell_name(report_input, margin.cell)}"
        return name
    aspect = reader_name("aspect", margin.aspect or margin.source, report_input)
    return f"the margin on {aspect}" if margin.kind == "margin" else f"the floor on {aspect}"


def _weakest(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    found = weakest_margin(report_input)
    if found is None:
        raise _refuse(
            path, "names the weakest margin, and the input reads no figure against a criterion, margin or floor"
        )
    field = values[0]
    if field == "arm":
        return Figure(path=path, kind="arm", value=found.arm)
    if field == "criterion":
        return _text(path, margin_name(found, report_input), "what the weakest margin is read against")
    if found.kind != "criterion":
        if field == "margin":
            return _number(path, found.gap, "the weakest margin", "points")
        return _number(
            path, found.value if field == "value" else found.line, f"the weakest {field}", "points", change=True
        )
    scale, unit = _scale_of(report_input, found.measure or "", (found.line,))
    if field == "value":
        return _number(path, found.value, "the weakest margin's figure", scale, unit=unit)
    if field == "threshold":
        return _number(path, found.line, "the weakest margin's threshold", scale, unit=unit)
    if scale in ("share", "percent"):
        return _number(path, found.gap * 100 if scale == "share" else found.gap, "the weakest margin", "points")
    return _number(path, found.gap, "the weakest margin", scale, unit=unit)


# --- the families --------------------------------------------------------------------------------------------------


def _arms(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [arm.id for arm in report_input.arms]


def _aspects(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [aspect.id for aspect in report_input.aspects]


def _asked_reading_arms(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return _reading_arms(_asked(report_input))


def _named_reading_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return _reading_arms(_aspect_by_id(report_input, prior[0]))


def _strata(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [stratum.id for stratum in report_input.strata]


def _keyed(name: str, read: Callable[[ReportInput], Iterable[str]]) -> _Ids:
    return _Ids(name, lambda report_input, _prior: read(report_input))


_ARM = _Ids("<arm>", _arms)
_CANDIDATE = _Ids("<candidate>", _costed_candidates)


def _family_of(
    *slots: _Slot,
    what: str,
    resolve: Callable[[ReportInput, tuple[str, ...], str], Figure],
    derived: bool = False,
) -> _Family:
    return _Family(slots=slots, what=what, resolve=resolve, derived=derived)


#: Every path family, in the order the grammar teaches them.
_FAMILIES: tuple[_Family, ...] = (
    _family_of(_Word("meta"), _Statistic(_META_FIELDS, "<field>"), what="the campaign's own words", resolve=_meta),
    _family_of(_Word("control"), _Statistic(_ARM_FIELDS, "<field>"), what="the control arm", resolve=_control_field),
    _family_of(_Word("n_candidates"), what="how many candidates ran", resolve=_n_candidates),
    _family_of(
        _Word("aspect"), _Statistic(_ASPECT_FIELDS, "<field>"), what="the one asked aspect", resolve=_asked_aspect_field
    ),
    _family_of(
        _Word("aspects"),
        _Ids("<aspect>", _aspects),
        _Statistic(_ASPECT_FIELDS, "<field>"),
        what="any aspect, by id",
        resolve=_named_aspect_field,
    ),
    _family_of(
        _Word("reading"),
        _Ids("<arm>", _asked_reading_arms),
        _Statistic(_READING_STATISTICS),
        what="an arm's reading on the asked aspect: its change from the control, in points",
        resolve=_asked_reading,
    ),
    _family_of(
        _Word("aspects"),
        _Ids("<aspect>", _aspects),
        _Word("reading"),
        _Ids("<arm>", _named_reading_arms),
        _Statistic(_READING_STATISTICS),
        what="an arm's reading on any aspect",
        resolve=_named_reading,
    ),
    _family_of(
        _Word("contrast"),
        _keyed("<arm>", lambda ri: dict.fromkeys(c.arm for c in ri.contrasts)),
        _Ids("<measure>", _contrasted),
        _Statistic(_CONTRAST_STATISTICS),
        what="a corrected contrast of an arm against the control on a measure or judged dimension",
        resolve=_contrast,
    ),
    _family_of(
        _Word("judged"),
        _keyed("<dimension>", lambda ri: ri.judged),
        _Ids("<arm>", _judged_arms),
        what="an arm's mean on a judged dimension",
        resolve=_judged,
    ),
    _family_of(
        _Word("judged"),
        _keyed("<dimension>", lambda ri: ri.judged),
        _Statistic(("label", *_JUDGE_FIELDS), "<field>"),
        what="a judged dimension's judge",
        resolve=_judged_field,
    ),
    _family_of(
        _Word("pass_rate"),
        _keyed("<arm>", _passing),
        what="an arm's pass rate: trials passed over trials usable",
        resolve=_pass_rate,
    ),
    _family_of(
        _Word("cost_per_1k"),
        _keyed("<arm>", _costs_per_thousand),
        what="an arm's cost per 1,000 replies",
        resolve=_cost_per_1k,
    ),
    _family_of(
        _Word("measure"),
        _keyed("<measure>", lambda ri: ri.measures),
        _Ids("<arm>", _measure_arms),
        _Statistic(_MEASURE_STATISTICS),
        what="an arm's figure on a measure",
        resolve=_measure,
    ),
    _family_of(
        _Word("scoreboard"),
        _keyed("<arm>", lambda ri: ri.scoreboard),
        _Statistic(_SCOREBOARD_STATISTICS),
        what="how an arm's trials came out",
        resolve=_scoreboard,
    ),
    _family_of(
        _Word("reasoning_ratio"),
        _Ids("<arm>", _reasoning_arms),
        what="the share of an arm's output spent reasoning",
        resolve=_reasoning_ratio,
    ),
    _family_of(_Word("cases"), what="the cases the asked aspect ran", resolve=_cases),
    _family_of(_Word("repeats"), what="the repeats each case ran", resolve=_repeats),
    _family_of(_Word("late_criteria"), _Word("set_at"), what="when the late criteria were set", resolve=_late_set_at),
    _family_of(_Word("family"), _Statistic(_FAMILY_FIELDS, "<field>"), what="the one contrast family", resolve=_family),
    _family_of(
        _Word("judge"),
        _Statistic(_JUDGE_FIELDS, "<field>"),
        what="the judge, where every judged dimension shares it",
        resolve=_judge,
    ),
    _family_of(_Word("caveats"), _Word("latency"), what="why latency is not compared", resolve=_latency),
    _family_of(
        _Word("task"),
        _Ids("<task>", _tasks),
        _Statistic(_TASK_FIELDS, "<field>"),
        what="a task",
        resolve=_task_field,
    ),
    _family_of(
        _Word("task"),
        _Ids("<task>", _tasks),
        _Ids("<arm>", _task_arms),
        _Ids("<measure>", _task_measures),
        _Statistic(_MEASURE_STATISTICS),
        what="an arm's figure on a measure or judged dimension within one task",
        resolve=_task_figure,
    ),
    _family_of(
        _Word("criteria"),
        _keyed("<criterion>", lambda ri: [criterion.id for criterion in ri.criteria]),
        _Statistic(_CRITERION_FIELDS, "<field>"),
        what="a declared criterion: its threshold in its measure's units, the measure, statistic and direction",
        resolve=_criterion_field,
    ),
    _family_of(
        _Word("strata"),
        _Ids("<stratum>", _strata),
        _Statistic(_STRATUM_FIELDS, "<field>"),
        what="a stratum",
        resolve=_stratum_field,
    ),
    _family_of(
        _Word("cost.uplift"),
        _CANDIDATE,
        what="how much more a candidate costs per reply than the control",
        resolve=_uplift,
        derived=True,
    ),
    _family_of(
        _Word("cost.uplift_range"),
        what="the least and the most any candidate's cost uplift is",
        resolve=_uplift_range,
        derived=True,
    ),
    _family_of(
        _Word("strata"),
        _Ids("<stratum>", _strata),
        _Word("loss_share"),
        what="the least share of any candidate's loss on the primary judged dimension that the stratum carries",
        resolve=_loss_share,
        derived=True,
    ),
    _family_of(
        _Word("strata"),
        _Ids("<stratum>", _strata),
        _Word("loss_share"),
        _Ids("<candidate>", _candidates),
        what="the share of one candidate's loss the stratum carries",
        resolve=_loss_share,
        derived=True,
    ),
    _family_of(
        _Word("weakest"),
        _Statistic(_WEAKEST_FIELDS, "<field>"),
        what=(
            "the weakest margin: the figure nearest the line it is read against (a criterion, an aspect's margin or a "
            "floor), its arm, what it is read against, the figure, the line, and the gap between them"
        ),
        resolve=_weakest,
        derived=True,
    ),
    _family_of(
        _Word("flip_margin"),
        _Ids("<candidate>", _asked_reading_arms),
        what="the margin above which a candidate's asked reading holds: its interval's worst drop",
        resolve=_flip_margin,
        derived=True,
    ),
)


@dataclass(frozen=True)
class _Reading:
    """One complete reading of a path: the family and the ids and statistic it matched, in order."""

    family: _Family
    values: tuple[str, ...]


@dataclass
class _Stop:
    """Where the furthest partial reading stopped, and what could have come next there."""

    at: int = -1
    expected: tuple[str, ...] = ()


def _walk(
    path: str,
    at: int,
    family: _Family,
    slots: tuple[_Slot, ...],
    values: tuple[str, ...],
    report_input: ReportInput,
    found: list[_Reading],
    stop: _Stop,
) -> None:
    slot, rest = slots[0], slots[1:]
    if isinstance(slot, _Word):
        options: Sequence[str] = (slot.text,)
    elif isinstance(slot, _Ids):
        options = list(slot.options(report_input, values))
    else:
        options = slot.options
    advanced = False
    for option in options:
        end = at + len(option)
        if not option or not path.startswith(option, at):
            continue
        captured = values if isinstance(slot, _Word) else (*values, option)
        if not rest and end == len(path):
            found.append(_Reading(family, captured))
            advanced = True
        elif rest and end < len(path) and path[end] == ".":
            advanced = True
            _walk(path, end + 1, family, rest, captured, report_input, found, stop)
    if not advanced:
        expected = tuple(options) if options else (f"{_slot_name(slot)} (none in this input)",)
        if at > stop.at:
            stop.at, stop.expected = at, expected
        elif at == stop.at:
            stop.expected = tuple(dict.fromkeys((*stop.expected, *expected)))


def _slot_name(slot: _Slot) -> str:
    return slot.text if isinstance(slot, _Word) else slot.name


def _readings(path: str, report_input: ReportInput) -> tuple[list[_Reading], _Stop]:
    found: list[_Reading] = []
    stop = _Stop()
    for family in _FAMILIES:
        _walk(path, 0, family, family.slots, (), report_input, found, stop)
    return found, stop


def _named(reading: _Reading) -> str:
    """One reading of a path, for an ambiguity refusal: each slot's name and what filled it."""
    parts: list[str] = []
    values = iter(reading.values)
    for slot in reading.family.slots:
        parts.append(f"`{slot.text}`" if isinstance(slot, _Word) else f"{_slot_name(slot)} `{next(values)}`")
    return " · ".join(parts)


def _listed(options: Sequence[str]) -> str:
    shown = [f"`{option}`" for option in options[:_CONTINUATIONS_SHOWN]]
    more = len(options) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more > 0 else "")


# --- the public surface ---------------------------------------------------------------------------------------------


def parse_key(inner: str, span: tuple[int, int] | None = None) -> FigureKey:
    """Split one key's inside into its path and filters, refusing a filter the grammar does not define.

    :param inner: The text between the braces, or a bare path (a durable claim's key, a chart reference's key).
    :ptype inner: str
    :param span: Where the key sits in the text it came from, braces included; None for a key parsed alone.
    :ptype span: tuple[int, int] | None
    :return: The key.
    :rtype: FigureKey
    :raises UnresolvableReference: The path is empty, a filter is unknown or repeated, or two filters each choose
        the spelling.
    """
    path, *filters = (part.strip() for part in inner.split("|"))
    if not path:
        raise UnresolvableReference(f"figure key {{{inner}}} has no path")
    for name in filters:
        if name not in _FILTERS:
            raise _refuse(inner, f"names filter {name!r}; the filters are {_listed(_FILTERS)}")
    if len(set(filters)) != len(filters):
        raise _refuse(inner, "repeats a filter")
    spellings = [name for name in filters if name in _SPELLINGS]
    if len(spellings) > 1:
        raise _refuse(inner, f"names two spellings, {' and '.join(spellings)}; keep one")
    if path.startswith(_ARM_PREFIX) and filters:
        raise _refuse(inner, "is an arm, which takes no filter")
    return FigureKey(path=path, filters=tuple(filters), span=span)


def keys_in(text: str) -> list[FigureKey]:
    """Every figure key in a piece of plan prose, in order.

    :param text: Plan prose.
    :ptype text: str
    :return: Each key with its span.
    :rtype: list[FigureKey]
    :raises UnresolvableReference: The text holds a double brace (the memo's grammar, never a plan's), a brace that
        opens or closes no key, or a key :func:`parse_key` refuses.
    """
    if "{{" in text or "}}" in text:
        raise UnresolvableReference(
            "plan prose holds a double brace: a figure key is `{path|filter}` in single braces, and `{{…}}` is the "
            "memo's figure reference, which a plan never writes"
        )
    keys = [parse_key(match.group(1), match.span()) for match in _KEY.finditer(text)]
    left = _KEY.sub("", text)
    if "{" in left or "}" in left:
        raise UnresolvableReference(f"plan prose holds a brace that opens or closes no figure key: {text!r}")
    return keys


def resolve(key: str, report_input: ReportInput) -> Figure:
    """Read one key's path from the report input.

    :param key: The path, without braces or filters (``reading.qwen3.8-flash.change``, ``arm:1017``).
    :ptype key: str
    :param report_input: The report input the plan was written against.
    :ptype report_input: ReportInput
    :return: The figure the path names.
    :rtype: Figure
    :raises UnresolvableReference: The path names nothing the input holds (the refusal lists what could come next
        where it stopped matching), reads two ways (naming both), or names a figure the input leaves unstated.
    """
    if key.startswith(_ARM_PREFIX):
        arm_id = key.removeprefix(_ARM_PREFIX)
        if arm_id not in {arm.id for arm in report_input.arms}:
            raise _refuse(key, f"names no arm; the arms are {_listed(_arms(report_input, ()))}")
        return Figure(path=key, kind="arm", value=arm_id)
    found, stop = _readings(key, report_input)
    if len(found) > 1:
        readings = "; ".join(_named(reading) for reading in found)
        raise _refuse(key, f"reads {len(found)} ways: {readings}. Choose ids that leave one reading.")
    if not found:
        if stop.at <= 0:
            raise _refuse(key, f"names no figure; a key begins with {_listed(_beginnings())}")
        raise _refuse(key, f"names no figure: after `{key[: stop.at]}` comes {_listed(stop.expected)}")
    reading = found[0]
    return reading.family.resolve(report_input, reading.values, key)


@dataclass(frozen=True)
class KeySubjects:
    """What one key is about: the arms and aspects its path names, and whether it reads an arm against the control.

    ``arms`` are the ids the path's arm or candidate slots matched (or the arm of ``{arm:<id>}``); ``aspects`` the
    aspect ids it named, or the asked aspect for a family that reads it without naming it (``reading``, ``aspect``,
    ``cases``, ``flip_margin``). ``against_control`` is True for a figure that is a difference from the control (a
    reading, a contrast, a cost uplift, a flip margin, a stratum's loss share), whose evidence is the control's too.
    """

    arms: tuple[str, ...]
    aspects: tuple[str, ...]
    against_control: bool


#: The slots an arm id fills.
_ARM_SLOTS = frozenset({"<arm>", "<candidate>"})

#: The families that read the asked aspect without naming it.
_ASKED_FAMILIES = frozenset({"aspect", "reading", "cases", "flip_margin"})

#: The words that make a figure a difference from the control.
_AGAINST_CONTROL = frozenset({"reading", "contrast", "cost.uplift", "cost.uplift_range", "flip_margin", "loss_share"})


def key_subjects(key: str, report_input: ReportInput) -> KeySubjects:
    """The arms and aspects one key's path names, read off its one reading of the input.

    What a durable claim rests on is read from its keys: the runs behind each arm the key names on each aspect it
    names (:func:`~threetears.evals.analysis.planned_analysis.planned_insights`).

    :param key: The path, without braces or filters.
    :ptype key: str
    :param report_input: The report input the plan was written against.
    :ptype report_input: ReportInput
    :return: What the key is about.
    :rtype: KeySubjects
    :raises UnresolvableReference: The path does not resolve (:func:`resolve`).
    """
    resolve(key, report_input)
    if key.startswith(_ARM_PREFIX):
        return KeySubjects(arms=(key.removeprefix(_ARM_PREFIX),), aspects=(), against_control=False)
    (reading,), _ = _readings(key, report_input)
    arms: list[str] = []
    aspects: list[str] = []
    words: list[str] = []
    values = iter(reading.values)
    for slot in reading.family.slots:
        if isinstance(slot, _Word):
            words.append(slot.text)
            continue
        value = next(values)
        if isinstance(slot, _Ids) and slot.name in _ARM_SLOTS:
            arms.append(value)
        elif isinstance(slot, _Ids) and slot.name == "<aspect>":
            aspects.append(value)
    if not aspects and words[0] in _ASKED_FAMILIES and (asked := _asked(report_input)) is not None:
        aspects.append(asked.id)
    return KeySubjects(
        arms=tuple(arms), aspects=tuple(aspects), against_control=any(word in _AGAINST_CONTROL for word in words)
    )


def _beginnings() -> list[str]:
    return list(dict.fromkeys(_slot_name(family.slots[0]) for family in _FAMILIES))


@dataclass(frozen=True)
class _Spelled:
    """A number as it reads, and the suffix unit it carries (``%``), which a range writes once."""

    number: str
    suffix: str = ""


def _plus(text: str, value: float) -> str:
    """``text`` with the ``+`` a positive change carries, as :func:`~threetears.evals.analysis.numbers.format_signed`
    places it: never on a bound, never on a zero."""
    return f"+{text}" if value > 0 and not text.startswith(("<", "+")) and text != "0" else text


def _money(value: float, signed: bool) -> str:
    text = dollars_text(abs(value))
    if value < 0:
        return f"-{text}"
    return _plus(text, value) if signed else text


def _grouped(text: str) -> str:
    """A whole number as :func:`~threetears.evals.analysis.numbers.format_number` spells it, its thousands separated
    by :data:`THOUSANDS_SEPARATOR` (``12000000`` reads ``12,000,000``); any other text unchanged."""
    sign, digits = (text[:1], text[1:]) if text[:1] in "+-" else ("", text)
    if not digits.isdigit():
        return text
    groups: list[str] = []
    while len(digits) > 3:
        digits, tail = digits[:-3], digits[-3:]
        groups.insert(0, tail)
    return sign + THOUSANDS_SEPARATOR.join([digits, *groups])


#: The spelling a figure on the ``usd_per_1m`` scale reads in without a filter: dollars with the words, ``$180 per
#: 1,000,000 replies``. No filter names it: a key's own spelling, never chosen.
_PER_MILLION = "per_million"


def _spelling_for(figure: Figure, filters: tuple[str, ...]) -> str:
    chosen = next((name for name in filters if name in _SPELLINGS), None)
    if chosen is not None:
        return chosen
    defaults: dict[str, str] = {
        "share": "pts" if figure.change else "pct",
        "percent": "pts" if figure.change else "pct",
        "points": "pts" if figure.change else "points",
        "usd": "usd",
        "usd_per_1k": "usd",
        "usd_per_1m": _PER_MILLION,
    }
    return defaults.get(figure.scale, "plain")


def _spell(value: float, figure: Figure, spelling: str, absolute: bool) -> _Spelled:
    if absolute:
        value = abs(value)
    signed = figure.change and not absolute
    rate = figure.scale in ("share", "percent")
    hundred = value * 100 if figure.scale == "share" else value
    if spelling in ("pct", "pct10"):
        if not rate or figure.change:
            why = "a change in a rate reads in points (pts)" if rate else f"it is on the {figure.scale} scale"
            raise _refuse(figure.path, f"cannot read as a percent: {why}")
        if spelling == "pct10":
            return _Spelled(format_number(float(round(hundred / 10) * 10)), PERCENT)
        whole = float(round(hundred)) if figure.whole else hundred
        return _Spelled(number_in(whole, PERCENT), PERCENT)
    if spelling in ("pts", "points"):
        if figure.scale not in ("share", "percent", "points"):
            raise _refuse(figure.path, f"cannot read in points: it is on the {figure.scale} scale")
        text = number_in(hundred, POINTS)
        return _Spelled(_plus(text, hundred) if spelling == "pts" and not absolute else text)
    if spelling in ("usd", "usd_per_1k"):
        if figure.scale not in _PER_THOUSAND_FACTORS:
            raise _refuse(figure.path, f"cannot read as dollars: it is on the {figure.scale} scale")
        if spelling == "usd":
            return _Spelled(_money(value, signed))
        per_thousand = value * _PER_THOUSAND_FACTORS[figure.scale]
        return _Spelled(PER_THOUSAND_TEMPLATE.format(amount=_money(per_thousand, signed)))
    if spelling == _PER_MILLION:
        # The package's money spelling (the scoreboard's): a price per million replies is whole dollars or near it,
        # and two decimals on it are noise.
        amount = signed_with_unit(value, MONEY) if signed else with_unit(value, MONEY)
        return _Spelled(_PER_MILLION_TEMPLATE.format(amount=amount))
    if spelling == "int":
        # A price per 1,000,000 replies also reads as a whole number, as stored plans spell it.
        if figure.scale not in ("count", "plain", "points", "usd_per_1m"):
            raise _refuse(figure.path, f"cannot read as a whole number: it is on the {figure.scale} scale")
        text = _grouped(format_number(float(round(value))))
        return _Spelled(_plus(text, value) if signed else text)
    text = format_number(value)
    text = _plus(text, value) if signed else text
    if not figure.unit:
        return _Spelled(text)
    return _Spelled(f"{text} {figure.unit}")


def format_figure(figure: Figure, filters: Sequence[str] = ()) -> str:
    """Spell a figure for a sentence, by its filters or, without one, by its scale.

    A rate reads as a percent (one decimal; a descriptive ratio in whole percents), a change in points, money in
    dollars, a count whole. An interval reads ``low to high`` when an end carries a sign, and ``low–high`` with its
    unit once otherwise (``27–28%``).

    :param figure: What a key resolved to; not an arm, which :func:`render` names.
    :ptype figure: Figure
    :param filters: The key's filters, as :func:`parse_key` checked them.
    :ptype filters: Sequence[str]
    :return: The figure's text.
    :rtype: str
    :raises UnresolvableReference: A filter does not suit the figure: a percent of money, points of a count, an
        ``interval`` on a single number, any filter on text.
    :raises ValueError: The figure is an arm.
    """
    chosen = tuple(filters)
    if figure.kind == "arm":
        raise ValueError(f"{figure.path} is an arm, which render names with its tag")
    if figure.kind == "text":
        if chosen:
            raise _refuse(figure.path, "is text, which takes no filter")
        return str(figure.value)
    spelling = _spelling_for(figure, chosen)
    absolute = "abs" in chosen
    if figure.kind == "number":
        if "interval" in chosen:
            raise _refuse(figure.path, "is one number, and `interval` reads an interval")
        assert isinstance(figure.value, float)
        spelled = _spell(figure.value, figure, spelling, absolute)
        return spelled.number + spelled.suffix
    assert isinstance(figure.value, tuple)
    low, high = figure.value
    if absolute:
        low, high = sorted((abs(low), abs(high)))
    ends = [_spell(end, figure, spelling, absolute) for end in (low, high)]
    if ends[0] == ends[1]:
        return ends[0].number + ends[0].suffix
    signed = spelling == "pts" or figure.change or low < 0 or high < 0
    if "interval" in chosen or signed or ends[0].suffix != ends[1].suffix:
        return INTERVAL_TEMPLATE.format(low=ends[0].number + ends[0].suffix, high=ends[1].number + ends[1].suffix)
    return RANGE_TEMPLATE.format(low=ends[0].number, high=ends[1].number) + ends[1].suffix


def render(text: str, report_input: ReportInput, arm_names: Mapping[str, str]) -> str:
    """Replace every figure key in plan prose with its value, and every ``{arm:<id>}`` with the arm's name and tag.

    :param text: Plan prose.
    :ptype text: str
    :param report_input: The report input the plan was written against.
    :ptype report_input: ReportInput
    :param arm_names: Each arm's computed short name, by id (GENERATOR §6.1). An arm declaring a short name
        (``Arm.short_name``) prints that instead; one missing from both is named by its alias, else its id.
    :ptype arm_names: Mapping[str, str]
    :return: The prose with every key written out.
    :rtype: str
    :raises UnresolvableReference: A key does not parse, resolve or suit its filters.
    """
    control = _control(report_input)
    tags = arm_tags([arm.id for arm in report_input.arms], control.id if control else None)
    pieces: list[str] = []
    at = 0
    for key in keys_in(text):
        assert key.span is not None
        start, end = key.span
        figure = resolve(key.path, report_input)
        if figure.kind == "arm":
            arm = _arm(report_input, str(figure.value))
            written = ARM_TEMPLATE.format(
                name=arm.short_name or arm_names.get(arm.id) or arm.name or arm.id, tag=tags[arm.id]
            )
        else:
            written = format_figure(figure, key.filters)
        pieces.extend((text[at:start], written))
        at = end
    pieces.append(text[at:])
    return "".join(pieces)


def _allowed_names(report_input: ReportInput) -> set[str]:
    """Every name the input holds that prose may write with its digits: never one made only of digits."""
    names: set[str] = {report_input.meta.campaign}
    for arm in report_input.arms:
        names.update((arm.id, arm.name or "", arm.short_name or "", arm.served_model or ""))
        names.update(value for value in arm.settings.values() if isinstance(value, str))
    for aspect in report_input.aspects:
        names.update((aspect.id, aspect.name or ""))
    for key, measure in report_input.measures.items():
        names.update((key, measure.label or ""))
    for key, judged in report_input.judged.items():
        names.update((key, judged.label or ""))
    for stratum in report_input.strata:
        names.update((stratum.id, stratum.name or ""))
    for task in report_input.tasks:
        names.update((task.id, task.name or ""))
    for factor in report_input.factors:
        names.update((factor.id, factor.name or ""))
        names.update(name for level in factor.levels for name in (level.id, level.name or ""))
    for example in report_input.examples:
        names.update((example.trial or "", example.case or ""))
    return {name for name in names if name.strip() and not name.strip().isdecimal()}


def literal_numbers(text: str, report_input: ReportInput) -> list[tuple[int, int]]:
    """Every number plan prose typed itself (GENERATOR §6.2's D3 rule), as ``(start, end)`` offsets into ``text``.

    Keys are set aside first; then every name the input holds that carries digits (arm ids and names, served models,
    settings as displayed, aspect, measure, dimension, stratum, task and factor ids and names, the campaign's name,
    trial and case ids), every percentile name (``p50``, ``p90``, ``p95``, ``p99``) and every one of the package's
    unit phrases (``per 1,000 replies``, matched whole) is set aside where a non-word character, or the text's edge,
    bounds it on both sides. Any Unicode decimal digit left is an offence (``95 percent``, ``1,000 cases``). An id
    made only of digits (an arm ``1017``) is never set aside: prose writes ``{arm:1017}``.

    :param text: Plan prose.
    :ptype text: str
    :param report_input: The report input the plan was written against.
    :ptype report_input: ReportInput
    :return: Each run of typed digits, in order; empty when the prose types none.
    :rtype: list[tuple[int, int]]
    """
    masked = list(text)

    def mask(start: int, end: int) -> None:
        # A word character, so a masked name never makes its neighbour look bounded.
        masked[start:end] = "_" * (end - start)

    for match in _KEY.finditer(text):
        mask(*match.span())
    allowed = _allowed_names(report_input) | set(PERCENTILE_NAMES) | set(_UNIT_PHRASES)
    for name in sorted(allowed, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer("".join(masked)):
            mask(*match.span())
    return [match.span() for match in _DIGITS.finditer("".join(masked))]


# --- the grammar, as the planner prompt teaches it -----------------------------------------------------------------

#: The heading the planner prompt's figure-key section opens with.
GRAMMAR_HEADING = "Figure keys"


def _example_input() -> ReportInput:
    """A small made-up input the grammar's examples are rendered on, with a dotted arm id and a dotted dimension."""
    rate = {"base": 0.912, "fast.v2": 0.874}
    return ReportInput.model_validate(
        {
            "schema": "report-input/1",
            "meta": {"campaign": "Example", "kind": "change", "question": "Is fast.v2 as good as base?"},
            "arms": [
                {
                    "id": "base",
                    "name": "Baseline",
                    "role": "control",
                    "served_model": "vendor/base-7",
                    "settings": {"model": "vendor/base-7", "max_output": "64k"},
                    "reasoning_ratio": 0.62,
                },
                {
                    "id": "fast.v2",
                    "role": "challenger",
                    "served_model": "vendor/fast-2",
                    "settings": {"model": "vendor/fast-2"},
                    "reasoning_ratio": 0.41,
                },
            ],
            "aspects": [
                {
                    "id": "answers",
                    "asked": True,
                    "margin": 10,
                    "readings": [
                        {"arm": "fast.v2", "reading": "within_margin", "change": -4.26, "within_interval": [-7.1, -1.4]}
                    ],
                }
            ],
            "scoreboard": {
                "base": {"passed": 765, "usable": 897, "reply_cost_usd": 0.001892},
                "fast.v2": {"passed": 731, "usable": 895, "reply_cost_usd": 0.002427},
            },
            "measures": {},
            "judged": {
                "tone.kept": {
                    "scale": "pass_fail",
                    "per_arm": {arm: {"mean": mean} for arm, mean in rate.items()},
                }
            },
            "primary_judged": "tone.kept",
            "contrasts": [
                {
                    "arm": "fast.v2",
                    "measure": "tone.kept",
                    "kind": "judged",
                    "verdict": "regressed",
                    "delta": -0.038,
                    "interval": [-0.061, -0.015],
                }
            ],
            "strata": [
                {
                    "id": "off_topic",
                    "n_cases": 40,
                    "per_arm": {"base": {"tone.kept": {"mean": 0.70}}, "fast.v2": {"tone.kept": {"mean": 0.55}}},
                },
                {
                    "id": "greeting",
                    "n_cases": 60,
                    "per_arm": {"base": {"tone.kept": {"mean": 0.95}}, "fast.v2": {"tone.kept": {"mean": 0.85}}},
                },
            ],
            "caveats": {},
        }
    )


#: One example per filter, each a key on :func:`_example_input`.
_FILTER_EXAMPLES: dict[str, str] = {
    "pct": "pass_rate.base|pct",
    "pts": "contrast.fast.v2.tone.kept.delta|pts",
    "abs": "reading.fast.v2.change|abs|pts",
    "usd": "cost_per_1k.base|usd",
    "int": "aspect.margin|int",
    "pct10": "strata.off_topic.loss_share|pct10",
    "interval": "reading.fast.v2.interval|interval",
    "usd_per_1k": "scoreboard.base.reply_cost_usd|usd_per_1k",
}


def figure_key_grammar() -> str:
    """Write the planner prompt's section that teaches figure keys, from this module's own families and filters.

    Each family's form and statistics, each filter and every example come from the code that parses and renders
    them; the examples are rendered on a made-up input, so a change to a family, a filter or a spelling reaches the
    planner with the code.

    :return: The section, heading first.
    :rtype: str
    :raises UnresolvableReference: An example no longer resolves: the grammar changed under this function, and every
        plan would share the break, so it fails here.
    """
    example = _example_input()
    names = {"base": "Baseline", "fast.v2": "fast.v2"}
    lines = [
        f"## {GRAMMAR_HEADING}",
        "This section is written by the code that reads your keys, so it is current: where anything above describes "
        "figure keys differently, follow this section.",
        "Never type a number into prose. Where a sentence needs one, write a figure key: `{path|filter|filter}`, "
        "in single braces. (`{{…}}` is a different grammar and is refused here.) Write `{arm:<id>}` where a sentence "
        "names an arm: it prints the arm's short name and letter tag.",
        "A path is matched against the ids this input holds, never split at dots: write an id exactly as the input "
        "spells it, dots and spaces included (`reading.fast.v2.change` is arm `fast.v2`). A path that reads two ways, "
        "or names nothing the input holds, is refused and sent back naming what could come next.",
        "Paths:",
    ]
    for family in _FAMILIES:
        statistic = family.statistic()
        listed = f" ({statistic.name}: {', '.join(statistic.options)})" if statistic else ""
        tag = " Derived: descriptive, no interval." if family.derived else ""
        lines.append(f"- `{family.form()}`: {family.what}{listed}.{tag}")  # one line: module constants
    lines.append("Filters (one spelling at most, plus `abs` and `interval`):")
    for name in _FILTERS:
        key = parse_key(_FILTER_EXAMPLES[name])
        shown = format_figure(resolve(key.path, example), key.filters)
        means, example_key = _FILTER_MEANS[name], _FILTER_EXAMPLES[name]
        lines.append(f'- `{name}`: {means}. `{{{example_key}}}` reads "{shown}".')  # one line: module constants
    sentence = "{arm:fast.v2} is down {reading.fast.v2.change|abs|pts} points ({reading.fast.v2.interval|pts})."
    lines.append(f'For example, "{sentence}" reads "{render(sentence, example, names)}".')
    per_reply = "scoreboard.base.reply_cost_usd"
    lines.append(
        "A key prints its own unit (`%`, `$`, `ms`, `s`, `per 1,000 replies`); never write a unit after it. A figure "
        "in points alone prints its bare number, so the word `points` follows it, as above. Money per reply reads per "
        f"1,000 replies: write `{{{per_reply}|usd_per_1k}}`, which reads "
        f'"{format_figure(resolve(per_reply, example), ("usd_per_1k",))}", not `{{{per_reply}}}` '
        f'("{format_figure(resolve(per_reply, example))}", a fraction of a cent).'
    )
    lines.append(
        "Without a filter a rate reads as a percent, a change in points, money in dollars. A digit outside a key is "
        "refused unless it is part of a name the input holds: an arm's id or name, a served model, a setting as "
        "displayed (such as `64k`), an aspect, measure, dimension, stratum or task id or name, the campaign's name, "
        "a trial or case id. An id made only of digits is never such a name: write `{arm:1017}`, not `1017`. "
        f"Percentile names ({', '.join(f'`{name}`' for name in PERCENTILE_NAMES)}) and the package's unit phrases "
        f"({', '.join(f'`{phrase}`' for phrase in _UNIT_PHRASES)}, whole) are written as they are; any other number "
        "is a key."
    )
    return "\n".join(lines)


__all__ = [
    "NAMED_KINDS",
    "PERCENTILE_NAMES",
    "WEAKEST_PREFIX",
    "Figure",
    "FigureKey",
    "KeySubjects",
    "NamedKind",
    "figure_key_grammar",
    "flip_margin",
    "format_figure",
    "key_subjects",
    "keys_in",
    "literal_numbers",
    "margin_name",
    "parse_key",
    "reader_name",
    "reader_names",
    "render",
    "resolve",
]
