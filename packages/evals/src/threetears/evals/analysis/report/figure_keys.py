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

**Derived values are descriptive.** ``cost.uplift.<arm>``, ``cost.uplift_range``, every ``loss_share``, every
``flip_margin``, ``weakest.*``, a slice's ``vs_control``, a floor's ``held_above`` and ``broken_at`` and a criterion
figure's ``gap`` are computed here from the input's point figures. They carry no interval and no test, the grammar
says so, and their :class:`Figure` is marked ``derived``. Everything else is copied from the input.

**A trade is priced here too.** ``price.<from>.<to>.<gain>.<field>`` prices a move between two options on one gain
axis (:func:`gain_axis`): the gain, signed toward better, the cost gap per 1,000 replies, and the price per point
gained, whose low and high ends are read off the gain's interval where a contrast states one (an end the interval
leaves open is refused as unbounded, never written as a number). ``cost_per_month.<arm>`` and the ``monthly_`` price
fields read the same at the declared ``meta.volume_per_month``. All are derived and marked so.

**Two arms apart, and the cases that would settle a reading.** ``spread.<arm>.<measure>.<where>.*`` reads an arm's
spread on a figure, ``compare.<arm>.<arm>.<measure>.<where>.*`` the difference between two arms with its interval (a
contrast's own where one pairs them, else independent), and ``settle.*`` the cases a design needs to separate an arm
from a declared criterion's threshold, a hypothetical line (``settle.hypothetical.<line>.``, the line a number in the
key), or another arm (:mod:`~threetears.evals.analysis.separation`). Derived, read off stated spreads: a normal
approximation, never a test the pipeline ran.

**A figure held on more than one aspect is named by its aspect.** Every figure that sits on an aspect has an
``aspects.<aspect>.`` form. Where the input holds a contrast's arm and measure, a stratum id or a contrast family on
more than one aspect, the unqualified form is refused, naming the qualified key for each aspect, rather than reading
one of them.

**Digits only inside keys (D3).** :func:`literal_numbers` finds every digit that plan prose types outside a key,
outside a name the input holds, outside a percentile's name (``p95``) and outside one of the package's own unit
phrases (``per 1,000 replies``). A number a reader sees is code's.

**The grammar is taught by this module.** :func:`figure_key_grammar` writes the planner prompt's section from the
module's own families, statistics and filters, with examples rendered on a synthetic input, so the prompt can never
teach a form this parser refuses. A reader that takes the grammar a page at a time reads the same lines:
:func:`figure_key_index` (the rules and filters, and a line per group of families) and :func:`figure_key_pages` (each
group's forms).
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, get_args

from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.analysis.numbers import format_number, group_thousands
from threetears.evals.analysis.quantities import MONEY, PERCENT, POINTS, number_in, signed_with_unit, with_unit
from threetears.evals.analysis.report.figures import PER_MILLION_REPLIES, PER_THOUSAND_LABEL, PER_THOUSAND_REPLIES
from threetears.evals.analysis.report.words import arm_tags
from threetears.evals.analysis.report_input import (
    Arm,
    Aspect,
    Cell,
    Contrast,
    ContrastFamily,
    Criterion,
    Depth,
    Judged,
    LateCriterion,
    Measure,
    Reading,
    ReportInput,
    ScoreboardRow,
    SettingValue,
    Stat,
    Stratum,
    Task,
)
from threetears.evals.analysis.report_margins import (
    CriterionFigure,
    Margin,
    criterion_figures,
    criterion_misses,
    criterion_passes,
    criterion_places,
    flip_margin,
    surviving_arms,
    weakest_margin,
)
from threetears.evals.analysis.separation import (
    POWER,
    NoSpread,
    Spread,
    cases_to_separate,
    independent_difference,
    level_share,
    stat_spread,
    wrong_side_share,
)
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
Filter = Literal["pct", "pts", "abs", "usd", "int", "pct10", "interval", "usd_per_1k", "words"]
_FILTERS: tuple[str, ...] = get_args(Filter)
_SPELLINGS: frozenset[str] = frozenset({"pct", "pts", "usd", "usd_per_1k", "int", "pct10", "words"})
#: The filters that choose a key's spelling, at most one to a key, in the grammar's order.
FILTER_SPELLINGS: tuple[str, ...] = tuple(name for name in _FILTERS if name in _SPELLINGS)

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
    "words": "a count spelled as a word up to twenty, in grouped digits above",
}

#: The unit phrases a rendered figure is joined with, one template each, so a translation replaces them here.
INTERVAL_TEMPLATE = "{low} to {high}"
RANGE_TEMPLATE = "{low}–{high}"
PER_THOUSAND_TEMPLATE = "{amount} " + PER_THOUSAND_REPLIES
_PER_MILLION_TEMPLATE = "{amount} " + PER_MILLION_REPLIES
ARM_TEMPLATE = "{name} ({tag})"
#: How a count from zero to twenty reads as a word (the ``words`` filter), in order; a translation replaces them here.
COUNT_WORDS: tuple[str, ...] = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
)
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
    whole percents because no interval stands behind a decimal. ``derived`` marks a figure computed here from the
    input's point figures rather than copied (a derived family's, or a derived statistic's: ``vs_control``, a floor's
    ``held_above``, a criterion figure's ``gap``): descriptive, with no interval and no test behind it.
    """

    path: str
    kind: FigureKind
    value: float | tuple[float, float] | str
    scale: Scale = "plain"
    unit: str = ""
    change: bool = False
    whole: bool = False
    derived: bool = False


# --- the vocabulary a path is matched against -------------------------------------------------------------------


@dataclass(frozen=True)
class _Word:
    """A fixed word in a path: a family keyword or a sub-field, as the path spells it."""

    spelling: str


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


@dataclass(frozen=True)
class _Number:
    """A slot a number the caller chose fills (a hypothetical line), written as digits with an optional sign and
    decimals; its name in the grammar."""

    name: str


_Slot = _Word | _Ids | _Statistic | _Number

#: A number as a path spells one in a :class:`_Number` slot.
_NUMBER_AT = re.compile(r"-?\d+(?:\.\d+)?")


def _number_options(path: str, at: int) -> list[str]:
    """The numbers a path can be read to spell at ``at``: its whole part, and with its decimals where it has them
    (``97.5`` before a dot may be ``97`` then ``5``, or ``97.5``)."""
    match = _NUMBER_AT.match(path, at)
    if match is None:
        return []
    whole, _dot, decimals = match.group(0).partition(".")
    return [whole, match.group(0)] if decimals else [whole]


@dataclass(frozen=True)
class _Family:
    """One path family: its slots in order, what it reads, and how its matched ids and statistic become a figure.

    ``derived`` marks a family whose every figure is computed here; ``derived_statistics`` the statistics of an
    otherwise copied family that are (a slice's ``vs_control``, a floor's ``held_above``). ``example`` is a key on the
    grammar's made-up input, which the grammar renders. ``derived_note`` replaces the grammar's "descriptive, no
    interval" line for a derived family that is not descriptive: one whose figures are read off stated spreads.
    """

    slots: tuple[_Slot, ...]
    what: str
    resolve: Callable[[ReportInput, tuple[str, ...], str], Figure]
    derived: bool = False
    derived_statistics: tuple[str, ...] = ()
    example: str = ""
    derived_note: str = ""

    def form(self) -> str:
        return ".".join(slot.spelling if isinstance(slot, _Word) else slot.name for slot in self.slots)

    def statistic(self) -> _Statistic | None:
        return next((slot for slot in self.slots if isinstance(slot, _Statistic)), None)


def _refuse(path: str, why: str) -> UnresolvableReference:
    return UnresolvableReference(f"figure key {{{path}}} {why}")


def _unstated(path: str, why: str) -> UnstatedFigure:
    """The refusal for a well-formed key whose figure the input leaves unstated, typed so a listing reads it as absent
    (:class:`~threetears.evals.analysis.errors.UnstatedFigure`)."""
    return UnstatedFigure(f"figure key {{{path}}} {why}")


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


def _is_lead(report_input: ReportInput, aspect_id: str) -> bool:
    """Whether an aspect is the lead one, whose figures the input's by-arm maps hold: the first the campaign asks
    about, else the first (:mod:`~threetears.evals.analysis.report_input`)."""
    asked = [aspect.id for aspect in report_input.aspects if aspect.asked]
    lead = asked[0] if asked else next((aspect.id for aspect in report_input.aspects), None)
    return aspect_id == lead


def _held_apart(
    path: str, what: str, aspects: Sequence[str | None], qualified: Callable[[str], str]
) -> UnresolvableReference:
    """The refusal of an unqualified key whose figure the input holds more than once: it names the key per aspect.

    :param path: The key refused.
    :param what: What it names, in words.
    :param aspects: The aspect each held copy sits on, None where one names none.
    :param qualified: The qualified key for one aspect.
    """
    named = [f"`{qualified(aspect)}`" for aspect in dict.fromkeys(aspects) if aspect is not None]
    choose = f"name its aspect: {', '.join(named)}" if named else "and no aspect tells them apart"
    return _refuse(path, f"names {what}, which the input holds {len(aspects)} times; {choose}")


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
        raise _unstated(path, f"names {what}, which the input does not state")
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
        raise _unstated(path, f"names {what}, which the input does not state")
    return Figure(path=path, kind="number", value=float(value), scale=scale, unit=unit, change=change, whole=whole)


def _end(path: str, interval: tuple[float, float] | None, statistic: str, what: str) -> float:
    """One end of ``interval`` (``low`` or ``high``), refused by name where the input states no interval."""
    if interval is None:
        raise _unstated(path, f"names {what}, which the input does not state")
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
        raise _unstated(path, f"names {what}, which the input does not state")
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


_CONTRAST_STATISTICS = (
    "delta",
    "interval",
    "low",
    "high",
    "verdict",
    "p_adjusted",
    "level",
    "materiality",
    "reason",
)


def _contrasted(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(c.measure for c in report_input.contrasts if c.arm == prior[0]))


def _contrast_figure(report_input: ReportInput, contrast: Contrast, statistic: str, path: str) -> Figure:
    scale, unit = _contrast_scale(report_input, contrast.measure, contrast.kind)
    if statistic == "delta":
        return _number(path, contrast.delta, "the delta", scale, unit=unit, change=True)
    if statistic == "interval":
        return _interval(path, contrast.interval, "the interval", scale, unit=unit, change=True)
    if statistic in ("low", "high"):
        end = _end(path, contrast.interval, statistic, "the interval")
        return _number(path, end, f"the interval's {statistic} end", scale, unit=unit, change=True)
    if statistic == "verdict":
        return _text(path, contrast.verdict, "the verdict")
    if statistic == "p_adjusted":
        return _number(path, contrast.p_adjusted, "the adjusted p", "plain")
    if statistic == "level":
        return _text(path, contrast.level, "the interval's level")
    return _text(path, getattr(contrast, statistic), f"the contrast's {statistic}")


def _contrast(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, measure, statistic = values
    held = [c for c in report_input.contrasts if c.arm == arm_id and c.measure == measure]
    if len(held) > 1:
        raise _held_apart(
            path,
            f"{arm_id}'s contrast on {measure}",
            [c.aspect for c in held],
            lambda aspect: f"aspects.{aspect}.contrast.{arm_id}.{measure}.{statistic}",
        )
    return _contrast_figure(report_input, held[0], statistic, path)


def _aspect_contrast_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(c.arm for c in report_input.contrasts if c.aspect == prior[0]))


def _aspect_contrasted(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect_id, arm_id = prior
    return list(dict.fromkeys(c.measure for c in report_input.contrasts if c.aspect == aspect_id and c.arm == arm_id))


def _aspect_contrast(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, arm_id, measure, statistic = values
    held = [c for c in report_input.contrasts if (c.aspect, c.arm, c.measure) == (aspect_id, arm_id, measure)]
    if len(held) > 1:
        raise _refuse(path, f"names {arm_id}'s contrast on {measure}, which aspect {aspect_id} holds {len(held)} times")
    return _contrast_figure(report_input, held[0], statistic, path)


def _judged_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(report_input.judged[prior[0]].per_arm)


def _judged(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    dimension, arm_id = values
    judged = report_input.judged[dimension]
    return _number(path, judged.per_arm[arm_id].mean, "the dimension's mean", _judged_scale(judged))


def _judged_stat(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    dimension, arm_id, statistic = values
    stat = report_input.judged[dimension].per_arm[arm_id]
    scale, unit = _scale_of(report_input, dimension)
    return _stat_figure(path, stat, statistic, scale, unit, f"the dimension's {statistic}")


def _on_aspect(table: Measure | Judged, aspect_id: str, report_input: ReportInput) -> Mapping[str, Stat]:
    """A measure's or judged dimension's figures on one aspect, by arm: ``per_arm`` on the lead aspect, else
    ``by_aspect``'s entry for it."""
    return table.per_arm if _is_lead(report_input, aspect_id) else table.by_aspect.get(aspect_id, {})


def _aspect_judged_dimensions(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return [key for key, judged in report_input.judged.items() if _on_aspect(judged, prior[0], report_input)]


def _aspect_judged_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect_id, dimension = prior
    return list(_on_aspect(report_input.judged[dimension], aspect_id, report_input))


def _aspect_judged(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, dimension, arm_id, statistic = values
    stat = _on_aspect(report_input.judged[dimension], aspect_id, report_input)[arm_id]
    scale, unit = _scale_of(report_input, dimension)
    return _stat_figure(path, stat, statistic, scale, unit, f"the dimension's {statistic} on the aspect")


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


def _stat_figure(path: str, stat: Stat, statistic: str, scale: Scale, unit: str, what: str) -> Figure:
    """One statistic of one arm's figure (:data:`_MEASURE_STATISTICS`), on the scale its measure is on."""
    if statistic in ("n", "n_cases"):
        return _number(path, getattr(stat, statistic), what, "count")
    if statistic == "ci":
        return _interval(path, stat.ci, what, scale, unit=unit)
    return _number(path, getattr(stat, statistic), what, scale, unit=unit, change=statistic == "change")


def _aspect_measures(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return [key for key, measure in report_input.measures.items() if _on_aspect(measure, prior[0], report_input)]


def _aspect_measure_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect_id, key = prior
    return list(_on_aspect(report_input.measures[key], aspect_id, report_input))


def _aspect_measure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, key, arm_id, statistic = values
    measure = report_input.measures[key]
    scale, unit = _measure_scale(measure)
    stat = _on_aspect(measure, aspect_id, report_input)[arm_id]
    return _stat_figure(path, stat, statistic, scale, unit, f"the measure's {statistic} on the aspect")


#: A slice's statistics (a task's, a stratum's, a cell's): a figure's, and the arm's mean against the control's there.
_SLICE_STATISTICS = (*_MEASURE_STATISTICS, "vs_control")
#: The slice statistics computed here rather than copied.
_SLICE_DERIVED = ("vs_control",)


def _slice_figure(
    report_input: ReportInput,
    per_arm: Mapping[str, Mapping[str, Stat]],
    values: tuple[str, str, str],
    path: str,
    where: str,
) -> Figure:
    """One arm's statistic on a measure or judged dimension within a slice (a task, a stratum, a cell), or its mean
    minus the control's there (``vs_control``: derived, descriptive)."""
    arm_id, key, statistic = values
    scale, unit = _scale_of(report_input, key)
    what = f"the {statistic} on {key} in {where}"
    if statistic != "vs_control":
        return _stat_figure(path, per_arm[arm_id][key], statistic, scale, unit, what)
    control = _control(report_input)
    if control is None:
        raise _refuse(path, "compares an arm with the control, and this campaign has none")
    if arm_id == control.id:
        raise _refuse(path, "compares an arm with the control, and names the control itself")
    mine, theirs = per_arm[arm_id][key].mean, per_arm.get(control.id, {}).get(key)
    if mine is None or theirs is None or theirs.mean is None:
        raise _unstated(
            path, f"names the arm's and the control's mean on {key} in {where}, which the input does not state"
        )
    return _number(path, mine - theirs.mean, f"the change from the control in {where}", scale, unit=unit, change=True)


_SCOREBOARD_STATISTICS = ("ran", "usable", "passed", "cases", "lost", "failed", "reply_cost_usd", "test_cost_usd")


def _scoreboard(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, statistic = values
    return _board_figure(report_input.scoreboard[arm_id], statistic, path)


def _aspect_board(report_input: ReportInput, aspect_id: str) -> Mapping[str, ScoreboardRow]:
    """How each arm's trials came out on one aspect: its own scoreboard, else, on the lead aspect, the input's."""
    board = _aspect_by_id(report_input, aspect_id).scoreboard
    if board is not None:
        return board
    return report_input.scoreboard if _is_lead(report_input, aspect_id) else {}


def _aspect_board_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(_aspect_board(report_input, prior[0]))


def _aspect_scoreboard(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, arm_id, statistic = values
    return _board_figure(_aspect_board(report_input, aspect_id)[arm_id], statistic, path)


def _board_figure(row: ScoreboardRow, statistic: str, path: str) -> Figure:
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
    """The asked aspect's depth as one figure; refused as a figure the input does not state where it has no single
    asked aspect, or the aspect's arms ran it to different depths: the input's shape, never a key spelled wrong."""
    aspect = _asked(report_input)
    if aspect is None:
        ids = ", ".join(f"`aspects.{a.id}.`" for a in report_input.aspects) or "none: the input holds no aspect"
        raise _unstated(
            path,
            f"reads the one asked aspect's depth, which the input does not state: it has no single one; name it: {ids}",
        )
    value = read(report_input, aspect)
    if value is None:
        raise _unstated(
            path,
            "reads the asked aspect's depth as one figure, which the input does not state: its arms ran it to "
            "different depths or none is stated",
        )
    return Figure(path=path, kind="number", value=float(value), scale="count")


def _mean_repeats(depth: Depth) -> float | None:
    """The mean repeats a case got (``repeats_per_case`` is fewest, mean, most), where the input states it."""
    return float(depth.repeats_per_case[1]) if len(depth.repeats_per_case) > 1 else None


def _depth_of(aspect: Aspect, report_input: ReportInput) -> list[tuple[int, float]]:
    """``(cases, mean repeats)`` for the control where it ran the aspect, else for every arm that did."""
    depth = aspect.depth or {}
    control = _control(report_input)
    if control is not None and control.id in depth:
        return [(depth[control.id].cases, float(depth[control.id].repeats_per_case[1]))]
    return [(d.cases, mean) for d in depth.values() if (mean := _mean_repeats(d)) is not None]


def _agreed(values: Iterable[float]) -> float | None:
    distinct = set(values)
    return distinct.pop() if len(distinct) == 1 else None


def _cases(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    return _depth_figure(report_input, path, lambda ri, a: _agreed(c for c, _r in _depth_of(a, ri)))


def _repeats(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    return _depth_figure(report_input, path, lambda ri, a: _agreed(r for _c, r in _depth_of(a, ri)))


def _aspect_depth(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    """The cases (or mean repeats) one aspect ran as one figure: the control's where it ran the aspect, else the one
    every arm agrees on."""
    aspect_id, field = values
    depth = _depth_of(_aspect_by_id(report_input, aspect_id), report_input)
    value = _agreed(cases if field == "cases" else repeats for cases, repeats in depth)
    if value is None:
        raise _unstated(
            path,
            f"names the {field} aspect {aspect_id} ran as one figure, which the input does not state: its arms ran "
            f"it to different depths, or no depth is stated; read each arm's `aspects.{aspect_id}.depth.<arm>.{field}`",
        )
    return Figure(path=path, kind="number", value=float(value), scale="count")


_DEPTH_FIELDS = ("cases", "trials", "repeats")


def _depth_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(_aspect_by_id(report_input, prior[0]).depth or {})


def _arm_depth(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    """How deeply one arm ran one aspect: its distinct cases, its trials, the mean repeats a case got."""
    aspect_id, arm_id, field = values
    depth = (_aspect_by_id(report_input, aspect_id).depth or {})[arm_id]
    if field == "repeats":
        return _number(path, _mean_repeats(depth), "the mean repeats a case got", "count")
    return _number(path, getattr(depth, field), f"the arm's {field} on the aspect", "count")


def _late_set_at(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    when = list(dict.fromkeys(late.set_at for late in report_input.caveats.late_criteria))
    if not when:
        raise _refuse(path, "names when the late criteria were set, and no criterion was set late")
    if len(when) > 1:
        per = ", ".join(f"`late_criteria.{late}.set_at`" for late in _late_names(report_input, ()))
        raise _unstated(
            path,
            f"names one date, which the input does not state: the late criteria were set at {len(when)} different "
            f"times; name one: {per}",
        )
    return Figure(path=path, kind="text", value=when[0])


def _late_name(late: LateCriterion) -> str:
    """How a key names one late criterion: what was set, then which one where it names one (``floor:<floor>``), as
    its caveat's id names it after ``late:``."""
    return f"{late.criterion}:{late.name}" if late.name else late.criterion


def _late_names(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(_late_name(late) for late in report_input.caveats.late_criteria))


def _late_criterion_set_at(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (named,) = values
    held = [late for late in report_input.caveats.late_criteria if _late_name(late) == named]
    when = list(dict.fromkeys(late.set_at for late in held))
    if len(when) > 1:
        raise _refuse(path, f"names when {named} was set, which the input records {len(when)} different times")
    return _text(path, when[0], f"when {named} was set")


def _setting_arms(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [arm.id for arm in report_input.arms if arm.settings]


def _setting_names(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(_arm(report_input, prior[0]).settings)


def _setting(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    """One of an arm's settings as the input holds it: a number where it holds a number, else its text."""
    arm_id, name = values
    value = _arm(report_input, arm_id).settings[name]
    if isinstance(value, str) or value is None:
        return _text(path, value, f"the arm's {name} setting")
    return _number(path, value, f"the arm's {name} setting", "plain")


_FAMILY_FIELDS = ("disclosure", "correction", "alpha", "family_size", "level")


def _family_key(report_input: ReportInput, index: int, field: str) -> str:
    """The key that reads one field of the family at ``index``: by its aspect where it alone sits on one, else by its
    place in the input's families."""
    aspect = report_input.contrast_families[index].aspect
    alone = aspect is not None and sum(family.aspect == aspect for family in report_input.contrast_families) == 1
    return f"aspects.{aspect}.family.{field}" if alone else f"families.{index}.{field}"


def _family(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (field,) = values
    families = report_input.contrast_families
    if len(families) > 1:
        keys = ", ".join(f"`{_family_key(report_input, index, field)}`" for index in range(len(families)))
        raise _refuse(
            path, f"names the one contrast family, which the input holds {len(families)} times; name one: {keys}"
        )
    if not families:
        raise _refuse(path, "reads the one contrast family, and the input holds 0")
    return _family_figure(families[0], field, path)


def _aspect_family(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, field = values
    held = [index for index, family in enumerate(report_input.contrast_families) if family.aspect == aspect_id]
    if not held:
        raise _unstated(path, f"names aspect {aspect_id}'s contrast family, which the input does not state")
    if len(held) > 1:
        keys = ", ".join(f"`families.{index}.{field}`" for index in held)
        raise _refuse(
            path, f"names aspect {aspect_id}'s contrast family, and the input holds {len(held)} on it; name one: {keys}"
        )
    return _family_figure(report_input.contrast_families[held[0]], field, path)


def _family_places(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [str(index) for index in range(len(report_input.contrast_families))]


def _indexed_family(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    index, field = values
    return _family_figure(report_input.contrast_families[int(index)], field, path)


def _family_figure(family: ContrastFamily, field: str, path: str) -> Figure:
    if field == "alpha":
        return _number(path, family.alpha, "the family's alpha", "plain")
    if field == "family_size":
        return _number(path, family.family_size, "the family's size", "count")
    return _text(path, getattr(family, field), f"the family's {field}")


def _latency(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    return _text(path, report_input.caveats.latency, "why latency is not compared")


def _held_values(report_input: ReportInput, key: str) -> list[float | None]:
    """Every mean the input holds for a measure or judged dimension: overall and on each other aspect, in each task,
    each stratum and each cell."""
    stats: list[Stat] = []
    for table in (report_input.measures, report_input.judged):
        if key in table:
            stats.extend(table[key].per_arm.values())
            stats.extend(stat for arms in table[key].by_aspect.values() for stat in arms.values())
    slices = [
        *(task.per_arm for task in report_input.tasks),
        *(stratum.per_arm for stratum in report_input.strata),
        *(cell.per_arm for cell in report_input.cells.values()),
    ]
    for per_arm in slices:
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
    per_arm = _task(report_input, task_id).per_arm
    return _slice_figure(report_input, per_arm, (arm_id, key, statistic), path, f"task {task_id}")


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


_CRITERION_FIGURE_FIELDS = ("value", "low", "high", "gap")
#: The criterion figure's field computed here: how far the figure clears the threshold.
_CRITERION_DERIVED = ("gap",)


def _cell_spelling(cell: Cell) -> str:
    """How a key names a target-matrix cell: its row and column level ids joined by a dot, since a cell's id joins
    them with ``|``, which a key cannot hold."""
    return f"{cell.row}.{cell.column}"


def _place(report_input: ReportInput, figure: CriterionFigure) -> str | None:
    """Where a criterion's figure sits, as a key names it: a task's id, a cell's spelling, None for the arm's overall
    value."""
    if figure.cell is not None:
        return _cell_spelling(report_input.cells[figure.cell])
    return figure.task


def _criterion_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    criterion = _criterion(report_input, prior[0])
    return [arm.id for arm in report_input.arms if criterion_figures(report_input, criterion, arm.id)]


def _criterion_places(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    criterion_id, arm_id = prior
    figures = criterion_figures(report_input, _criterion(report_input, criterion_id), arm_id)
    return [place for figure in figures if (place := _place(report_input, figure)) is not None]


def _gap_figure(report_input: ReportInput, path: str, measure: str, line: float, gap: float, what: str) -> Figure:
    """A gap from a threshold on a measure, signed toward passing: in points for a rate, else in the measure's units."""
    scale, unit = _scale_of(report_input, measure, (line,))
    if scale in ("share", "percent"):
        return _number(path, gap * 100 if scale == "share" else gap, what, "points")
    return _number(path, gap, what, scale, unit=unit)


def _criterion_figure_field(
    report_input: ReportInput, criterion: Criterion, figure: CriterionFigure, field: str, path: str
) -> Figure:
    threshold = float(criterion.threshold)
    scale, unit = _scale_of(report_input, criterion.measure, (threshold,))
    if field == "value":
        return _number(path, figure.value, "the criterion's figure", scale, unit=unit)
    if field in ("low", "high"):
        end = _end(path, figure.ci, field, "the figure's interval")
        return _number(path, end, f"the figure's interval's {field} end", scale, unit=unit)
    gap = figure.value - threshold if criterion.direction == "higher_is_better" else threshold - figure.value
    return _gap_figure(report_input, path, criterion.measure, threshold, gap, "the figure's gap from the threshold")


def _criterion_arm(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion_id, arm_id, field = values
    criterion = _criterion(report_input, criterion_id)
    figures = criterion_figures(report_input, criterion, arm_id)
    if len(figures) > 1:
        places = [f"criteria.{criterion_id}.{arm_id}.{_place(report_input, figure)}.{field}" for figure in figures]
        raise _refuse(
            path, f"reads one figure, and the criterion reads {len(figures)}, one per place: {_listed(places)}"
        )
    return _criterion_figure_field(report_input, criterion, figures[0], field, path)


def _criterion_arm_at(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion_id, arm_id, place, field = values
    criterion = _criterion(report_input, criterion_id)
    figure = next(
        figure for figure in criterion_figures(report_input, criterion, arm_id) if _place(report_input, figure) == place
    )
    return _criterion_figure_field(report_input, criterion, figure, field, path)


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

#: A strata family's resolver: what one stratum, and the slots after its id, read.
_StratumRead = Callable[[ReportInput, Stratum, tuple[str, ...], str], Figure]


def _stratum(report_input: ReportInput, stratum_id: str) -> Stratum:
    return next(stratum for stratum in report_input.strata if stratum.id == stratum_id)


def _scoped_strata(report_input: ReportInput, aspect_id: str | None) -> list[Stratum]:
    """The strata a key reads from: every one, or, in an aspect's own strata keys, the ones slicing its cases."""
    return [stratum for stratum in report_input.strata if aspect_id is None or stratum.aspect == aspect_id]


def _strata(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(stratum.id for stratum in report_input.strata))


def _aspect_strata(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(stratum.id for stratum in _scoped_strata(report_input, prior[0])))


def _named_strata(
    report_input: ReportInput, prior: tuple[str, ...], qualified: bool
) -> tuple[list[Stratum], tuple[str, ...]]:
    """The strata a key's stratum id names (in its aspect, where the key names one), and the slots after the id."""
    aspect_id, rest = (prior[0], prior[1:]) if qualified else (None, prior)
    return [stratum for stratum in _scoped_strata(report_input, aspect_id) if stratum.id == rest[0]], rest[1:]


def _one_stratum(report_input: ReportInput, aspect_id: str | None, stratum_id: str, path: str) -> Stratum:
    """The one stratum a key names; a stratum id held on more than one aspect is refused naming each aspect's key."""
    held = [stratum for stratum in _scoped_strata(report_input, aspect_id) if stratum.id == stratum_id]
    if len(held) == 1:
        return held[0]
    if aspect_id is not None:
        raise _refuse(path, f"names stratum {stratum_id}, which aspect {aspect_id} holds {len(held)} times")
    tail = path.removeprefix(f"strata.{stratum_id}.")
    raise _held_apart(
        path,
        f"stratum {stratum_id}",
        [stratum.aspect for stratum in held],
        lambda aspect: f"aspects.{aspect}.strata.{stratum_id}.{tail}",
    )


def _strata_resolver(read: _StratumRead, qualified: bool) -> Callable[[ReportInput, tuple[str, ...], str], Figure]:
    """A strata family's resolver, reading the one stratum its key names, in the key's aspect when it names one."""

    def resolve(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
        aspect_id, rest = (values[0], values[1:]) if qualified else (None, values)
        return read(report_input, _one_stratum(report_input, aspect_id, rest[0], path), rest[1:], path)

    return resolve


def _stratum_arms(qualified: bool) -> Callable[[ReportInput, tuple[str, ...]], list[str]]:
    def options(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
        held, _rest = _named_strata(report_input, prior, qualified)
        return list(dict.fromkeys(arm for stratum in held for arm in stratum.per_arm))

    return options


def _stratum_keys(qualified: bool) -> Callable[[ReportInput, tuple[str, ...]], list[str]]:
    def options(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
        held, (arm_id,) = _named_strata(report_input, prior, qualified)
        return list(dict.fromkeys(key for stratum in held for key in stratum.per_arm.get(arm_id, {})))

    return options


def _loss_keys(qualified: bool) -> Callable[[ReportInput, tuple[str, ...]], list[str]]:
    """The measures and dimensions a candidate's loss may be read on in a stratum: those it and the control hold."""

    def options(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
        held, (arm_id,) = _named_strata(report_input, prior, qualified)
        control = _control(report_input)
        if control is None:
            return []
        return list(
            dict.fromkeys(
                key
                for stratum in held
                for key in stratum.per_arm.get(arm_id, {})
                if key in stratum.per_arm.get(control.id, {})
            )
        )

    return options


def _stratum_field(report_input: ReportInput, stratum: Stratum, rest: tuple[str, ...], path: str) -> Figure:
    (field,) = rest
    if field == "name":
        return _text(path, stratum.name or humanize("dimension", stratum.id), "the stratum's name")
    if field == "n_cases":
        return _number(path, stratum.n_cases, "the stratum's cases", "count")
    return _number(path, stratum.share, "the stratum's share of traffic", "share", whole=True)


def _stratum_figure(report_input: ReportInput, stratum: Stratum, rest: tuple[str, ...], path: str) -> Figure:
    arm_id, key, statistic = rest
    return _slice_figure(report_input, stratum.per_arm, (arm_id, key, statistic), path, f"stratum {stratum.id}")


def _better_sign(report_input: ReportInput, key: str, path: str) -> float:
    """Which way a loss runs on a measure or judged dimension: 1 where higher is better (the control's figure minus the
    arm's), -1 where lower is. A judged dimension reads higher as better."""
    if key in report_input.judged:
        return 1.0
    higher = report_input.measures[key].higher_is_better if key in report_input.measures else None
    if higher is None:
        raise _refuse(path, f"reads a loss on {key}, whose better direction the input does not state")
    return 1.0 if higher else -1.0


def _loss_share_of(report_input: ReportInput, target: Stratum, arm_id: str, key: str | None, path: str) -> float:
    """The share of ``arm_id``'s loss against the control, on ``key`` (else the primary judged dimension), that one
    stratum carries.

    Loss in a stratum is the control's figure minus the arm's where higher is better (the arm's minus the control's
    where lower is), weighted by the stratum's cases (by its traffic share where cases are not stated), over the strata
    of the same stratification on the same aspect.
    """
    dimension = key if key is not None else report_input.primary_judged
    control = _control(report_input)
    if dimension is None or control is None:
        raise _refuse(path, "reads loss on the primary judged dimension against the control; the input lacks one")
    sign = 1.0 if key is None else _better_sign(report_input, key, path)
    group = [s for s in report_input.strata if (s.aspect, s.dimension) == (target.aspect, target.dimension)]
    weights: list[float | None] = [s.n_cases for s in group]
    if any(weight is None for weight in weights):
        weights = [s.share for s in group]
    if any(weight is None for weight in weights):
        raise _refuse(path, "weights each stratum by its cases or its share, and a stratum states neither")
    losses: list[float] = []
    for stratum, weight in zip(group, weights, strict=True):
        figures = stratum.per_arm
        held = figures.get(control.id, {}).get(dimension), figures.get(arm_id, {}).get(dimension)
        if held[0] is None or held[1] is None or held[0].mean is None or held[1].mean is None:
            raise _refuse(path, f"needs both arms' {dimension} in stratum {stratum.id}, which the input lacks")
        losses.append(float(weight or 0) * sign * (held[0].mean - held[1].mean))
    total = sum(losses)
    if total <= 0:
        raise _refuse(path, f"divides by {arm_id}'s loss on {dimension}, and it lost nothing overall")
    return losses[next(at for at, stratum in enumerate(group) if stratum is target)] / total


def _loss_share(report_input: ReportInput, stratum: Stratum, rest: tuple[str, ...], path: str) -> Figure:
    arms = rest[:1] or tuple(_candidates(report_input))
    key = rest[1] if len(rest) > 1 else None
    if not arms:
        raise _refuse(path, "reads each candidate's loss, and the campaign has no candidate")
    share = min(_loss_share_of(report_input, stratum, arm, key, path) for arm in arms)
    return Figure(path=path, kind="number", value=share, scale="share", whole=True)


def _flip_margin(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    return _flip_figure(_reading(_asked_or_refuse(report_input, path), values[0]), path)


def _aspect_flip_margin(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, arm_id = values
    return _flip_figure(_reading(_aspect_by_id(report_input, aspect_id), arm_id), path)


def _flip_figure(reading: Reading, path: str) -> Figure:
    flip = flip_margin(reading)
    if flip is None:
        raise _unstated(path, "names the interval a flip is read on, which the input does not state")
    return Figure(path=path, kind="number", value=flip, scale="points")


# --- prices: one option priced in another, on one gain axis -------------------------------------------------------

#: What a price key reads for a move from one option to another, on one gain axis. ``gain`` is how much better the
#: second option reads, toward the axis's better side; ``gain_interval`` the interval on it, where a contrast states
#: one (paired); ``gain_interval_unpaired`` the gain plus or minus the two options' own half-widths on the axis
#: combined as independent (descriptive, uncorrected), where both state one; ``cost_gap`` what the second costs more
#: per 1,000 replies; ``per_point`` the cost gap per point gained (per unit, on an axis not in points), and its
#: ``low`` and ``high`` ends read off the gain's interval, paired or (``_unpaired``) unpaired; the ``monthly_`` fields
#: the same at the declared monthly volume.
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


def _gain_axes(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    """Every measure or judged dimension a gain can be read on: a judged dimension, or a measure with a direction."""
    measures = [key for key, measure in report_input.measures.items() if measure.higher_is_better is not None]
    return list(dict.fromkeys([*report_input.judged, *measures]))


def _gain_table(report_input: ReportInput, key: str) -> Measure | Judged:
    """The axis's own table: a judged dimension's, where the key names one, else the measure's."""
    return report_input.judged[key] if key in report_input.judged else report_input.measures[key]


def _gain_contrasts(report_input: ReportInput, key: str, control: str) -> list[Contrast]:
    """The contrasts that state a candidate against the control on the axis, with their difference."""
    return [
        contrast
        for contrast in report_input.contrasts
        if contrast.measure == key
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
    holding a figure on the axis holds exactly one contrast on it that states its difference: the paired estimate,
    which the contrast's interval belongs to. Otherwise it reads each arm's **overall mean**, which carries no paired
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
    """What turns a figure per 1,000 replies into one per month at the declared volume."""
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
    """What a price reads: the gain axis, each option's cost per 1,000 replies, the slice it is read within (its key
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
    field = field.removeprefix(_MONTHLY_PREFIX)
    if field == "cost_gap":
        return Figure(path=path, kind="number", value=gap * factor, scale=money, change=True)
    points = 100.0 if scale == "share" else 1.0
    if gain == 0 or gap * gain <= 0:
        raise _refuse(
            path,
            f"is no price: {second} reads no better than {first} on {axis.key!r} for no less cost, or better for no "
            "more, so one is at least as good on both and the move has no price per point",
        )
    if field == "per_point":
        return Figure(path=path, kind="number", value=gap / (gain * points) * factor, scale=money)
    if interval is None:
        raise _unstated(path, f"names an end of the price's interval, which the input does not state: {unstated}")
    bound = _price_bounds(path, gap, gain * points, (interval[0] * points, interval[1] * points), field)
    return Figure(path=path, kind="number", value=bound * factor, scale=money)


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


def _cost_per_month(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    factor = _monthly_factor(report_input, path)
    return Figure(path=path, kind="number", value=_costs_per_thousand(report_input)[values[0]] * factor, scale="usd")


# --- the price frontier: the options best at some price per point --------------------------------------------------

#: Why an option is not priced: a gate or guardrail removes it, the cost rule states no cost for it, or the gain axis
#: places it nowhere.
Unpriced = Literal["gated", "no_cost", "no_gain_figure"]


@dataclass(frozen=True)
class PriceFrontier:
    """The options a trade prices on one gain axis, overall or within one slice, and which are best at some price.

    :ivar gain_key: The gain axis, by key.
    :ivar within: The slice it is read within, as a key's prefix (``task.<task>``); empty overall.
    :ivar axis: The gain axis, as the basis places each arm.
    :ivar costs: Each option's cost per 1,000 replies, as the basis reads it.
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
    surviving_arms`) or is admitted as a what-if, holds a cost per 1,000 replies, and is placed on the axis (overall,
    :func:`gain_axis`; within a slice, its mean there, at the slice's own cost where the slice states one for every
    option it places). An option is dominated when another costs no more and reads no worse, ahead on one of the two;
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
    admitted = tuple(arm.id for arm in report_input.arms if arm.id in admit and arm.id not in survivors)
    unpriced: dict[str, Unpriced] = {}
    options: dict[str, tuple[float, float]] = {}
    for arm in report_input.arms:
        if arm.id not in survivors and arm.id not in admitted:
            unpriced[arm.id] = "gated"
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


# --- counts: what a sentence counts, keyed so prose never types one ------------------------------------------------

#: What ``count.<what>`` counts in the campaign: its arms (every option, the control among them), aspects, tasks,
#: strata, target-matrix cells, configs of a factorial grid, gates and guardrails, every declared criterion, and the
#: options surviving every gate and guardrail.
_COUNTED = ("arms", "aspects", "tasks", "strata", "cells", "configs", "gates", "criteria", "surviving")

#: The criterion kinds that remove an option that misses them, as the gates count them.
_GATE_KINDS = frozenset({"gate", "guardrail"})


def _count(path: str, value: int) -> Figure:
    return Figure(path=path, kind="number", value=float(value), scale="count")


def _gates(report_input: ReportInput) -> list[Criterion]:
    return [criterion for criterion in report_input.criteria if criterion.kind in _GATE_KINDS]


def _campaign_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (what,) = values
    counted: dict[str, int] = {
        "arms": len(report_input.arms),
        "aspects": len(report_input.aspects),
        "tasks": len(report_input.tasks),
        "strata": len(report_input.strata),
        "cells": len(report_input.cells),
        "configs": len(report_input.configs),
        "gates": len(_gates(report_input)),
        "criteria": len(report_input.criteria),
    }
    if what == "surviving":
        return _count(path, len(surviving_arms(report_input)))
    return _count(path, counted[what])


def _gate_ids(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return [criterion.id for criterion in _gates(report_input)]


def _criterion_ids(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return [criterion.id for criterion in report_input.criteria]


def _passing_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion = _criterion(report_input, values[0])
    return _count(path, sum(criterion_passes(report_input, criterion, arm.id) for arm in report_input.arms))


def _places_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    return _count(path, len(criterion_places(report_input, _criterion(report_input, values[0]))))


def _failing_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion_id, arm_id = values
    return _count(path, criterion_misses(report_input, _criterion(report_input, criterion_id), arm_id))


def _failed_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (arm_id,) = values
    missed = [criterion for criterion in _gates(report_input) if not criterion_passes(report_input, criterion, arm_id)]
    return _count(path, len(missed))


def _factor_ids(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return [factor.id for factor in report_input.factors]


def _levels_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    factor = next(factor for factor in report_input.factors if factor.id == values[0])
    return _count(path, len(factor.levels))


# --- an aspect's pair readings, floors and bars ----------------------------------------------------------------------

_PAIR_STATISTICS = ("change", "interval", "low", "high", "confidence", "p_holm", "ahead", "why")


def _pair_firsts(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(pair.arms[0] for pair in _aspect_by_id(report_input, prior[0]).pairs))


def _pair_seconds(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect_id, first = prior
    pairs = _aspect_by_id(report_input, aspect_id).pairs
    return list(dict.fromkeys(pair.arms[1] for pair in pairs if pair.arms[0] == first))


def _pair(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, first, second, statistic = values
    held = [pair for pair in _aspect_by_id(report_input, aspect_id).pairs if pair.arms == (first, second)]
    if len(held) > 1:
        raise _refuse(path, f"names the pair {first} and {second}, which aspect {aspect_id} holds {len(held)} times")
    pair = held[0]
    if statistic == "change":
        return _number(path, pair.change, "the pair's change", "points", change=True)
    if statistic == "interval":
        return _interval(path, pair.interval, "the pair's interval", "points", change=True)
    if statistic in ("low", "high"):
        end = _end(path, pair.interval, statistic, "the pair's interval")
        return _number(path, end, f"the interval's {statistic} end", "points", change=True)
    if statistic == "confidence":
        return _number(path, pair.confidence, "the interval's level", "share", whole=True)
    if statistic == "p_holm":
        return _number(path, pair.p_holm, "the Holm-adjusted p", "plain")
    if statistic == "ahead":
        if pair.ahead is None:
            raise _unstated(path, "names the arm shown ahead, which the input does not state: neither is shown ahead")
        return Figure(path=path, kind="arm", value=pair.ahead)
    return _text(path, pair.why, "why the pair reads as it does")


_FLOOR_FIELDS = (
    "status",
    "reading",
    "speaks_for",
    "interval",
    "low",
    "high",
    "worst_drop",
    "margin",
    "short_by",
    "reason",
    "held_above",
    "broken_at",
)
#: The floor fields computed here: the margin above which the floor holds (``-low``), and at or below which its
#: interval's best end breaks it (``-high``).
_FLOOR_DERIVED = ("held_above", "broken_at")


def _floor_aspects(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    """The aspects floors are read on, as each read names its own: an aspect the input holds a record for or not."""
    return list(dict.fromkeys(read.aspect for read in report_input.floors))


def _floor_keys(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(read.floor for read in report_input.floors if read.aspect == prior[0]))


def _floor_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect_id, floor = prior
    return [read.arm for read in report_input.floors if (read.aspect, read.floor) == (aspect_id, floor)]


def _floor(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, floor, arm_id, field = values
    held = [read for read in report_input.floors if (read.aspect, read.floor, read.arm) == (aspect_id, floor, arm_id)]
    if len(held) > 1:
        raise _refuse(path, f"names floor {floor} on {arm_id}, which aspect {aspect_id} holds {len(held)} times")
    read = held[0]
    if field == "interval":
        return _interval(path, read.interval, "the floor's interval", "points", change=True)
    if field in ("low", "high"):
        end = _end(path, read.interval, field, "the floor's interval")
        return _number(path, end, f"the interval's {field} end", "points", change=True)
    if field in ("held_above", "broken_at"):
        end = _end(path, read.interval, "low" if field == "held_above" else "high", "the floor's interval")
        return _number(path, -float(end), f"the floor's {field}", "points")
    if field in ("worst_drop", "margin", "short_by"):
        return _number(path, getattr(read, field), f"the floor's {field}", "points")
    return _text(path, getattr(read, field), f"the floor's {field}")


_BAR_FIELDS = ("status", "value", "bar", "interval", "low", "high", "why")


def _bar_aspects(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    """The aspects bars are read on, as each read names its own: an aspect the input holds a record for or not."""
    return list(dict.fromkeys(read.aspect for read in report_input.bars))


def _bar_measures(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(read.measure for read in report_input.bars if read.aspect == prior[0]))


def _bar_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect_id, measure = prior
    return [read.arm for read in report_input.bars if (read.aspect, read.measure) == (aspect_id, measure)]


def _bar(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, measure, arm_id, field = values
    held = [read for read in report_input.bars if (read.aspect, read.measure, read.arm) == (aspect_id, measure, arm_id)]
    if len(held) > 1:
        raise _refuse(
            path, f"names the bar on {measure} for {arm_id}, which aspect {aspect_id} holds {len(held)} times"
        )
    read = held[0]
    scale, unit = _scale_of(report_input, measure, (float(read.bar),))
    if field in ("value", "bar"):
        return _number(path, getattr(read, field), f"the bar's {field}", scale, unit=unit)
    if field == "interval":
        return _interval(path, read.interval, "the bar's interval", scale, unit=unit)
    if field in ("low", "high"):
        end = _end(path, read.interval, field, "the bar's interval")
        return _number(path, end, f"the interval's {field} end", scale, unit=unit)
    return _text(path, getattr(read, field), f"the bar's {field}")


# --- target-matrix cells -------------------------------------------------------------------------------------------

_CELL_FIELDS = ("name", "n_cases")


def _cell_spellings(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [_cell_spelling(cell) for cell in report_input.cells.values()]


def _spelled_cell(report_input: ReportInput, spelled: str) -> str:
    return next(cell_id for cell_id, cell in report_input.cells.items() if _cell_spelling(cell) == spelled)


def _cell_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(report_input.cells[_spelled_cell(report_input, prior[0])].per_arm)


def _cell_keys(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    spelled, arm_id = prior
    return list(report_input.cells[_spelled_cell(report_input, spelled)].per_arm[arm_id])


def _cell_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    spelled, field = values
    cell_id = _spelled_cell(report_input, spelled)
    if field == "name":
        return _text(path, _cell_name(report_input, cell_id), "the cell's name")
    return _number(path, report_input.cells[cell_id].n_cases, "the cell's cases", "count")


def _cell_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    spelled, arm_id, key, statistic = values
    per_arm = report_input.cells[_spelled_cell(report_input, spelled)].per_arm
    return _slice_figure(report_input, per_arm, (arm_id, key, statistic), path, f"cell {spelled}")


# --- reader names: the one name a reader is shown for each id -----------------------------------------------------

#: The kinds of id a reader is given a name for (GENERATOR §6.3: reader names, not identifiers).
NamedKind = Literal["aspect", "measure", "judged", "stratum", "task", "criterion"]


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


def _margin_name(margin: Margin, report_input: ReportInput) -> str:
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
        return _text(path, _margin_name(found, report_input), "what the weakest margin is read against")
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
    return _gap_figure(report_input, path, found.measure or "", found.line, found.gap, "the weakest margin")


# --- the families --------------------------------------------------------------------------------------------------


def _arms(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [arm.id for arm in report_input.arms]


def _aspects(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [aspect.id for aspect in report_input.aspects]


def _asked_reading_arms(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return _reading_arms(_asked(report_input))


def _named_reading_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return _reading_arms(_aspect_by_id(report_input, prior[0]))


# --- separation: an arm's spread, two arms' difference, the cases that would settle a reading ------------------------
#
# ``spread.*``, ``compare.*`` and ``settle.*`` read a figure's spread and what it says (:mod:`~threetears.evals.analysis
# .separation` holds the arithmetic and its design note). Each names where it reads with one ``<where>`` slot:
# ``overall`` (the arm's own figure), ``task.<task>``, ``cells.<row>.<column>`` or ``strata.<stratum>`` (a stratum whose
# id the input holds once). Every figure is derived: computed here, read off the figures' stated spreads (a normal
# approximation), never a test the pipeline ran.

#: Where a separation key reads an arm's own figure, rather than one within a slice.
_OVERALL = "overall"

#: What an arm's spread on a figure reads: its mean, its standard error, the spread of one case, the count it rests on,
#: the level a result read from it is stated at, and what the standard error was read from.
_SPREAD_FIELDS = ("mean", "se", "per_case", "n", "level", "basis")

#: What two arms' difference reads: the first minus the second, its interval, the interval's ends and half-width, its
#: level, and whether it is the paired estimate a contrast states or two independent arms' spreads combined.
_COMPARE_FIELDS = ("difference", "interval", "low", "high", "half_width", "level", "basis")

#: What settling two arms apart reads: their difference, the cases each needs for an interval centred on it to clear
#: zero (``cases_needed``: half such designs clear; ``cases_needed_80``: four in five), how often a fresh run of today's
#: size reads the other arm ahead, and the two-sided level.
_SETTLE_PAIR_FIELDS = ("difference", "cases_needed", "cases_needed_80", "wrong_side", "level")

#: What settling an arm against a line reads: its distance from the line toward passing, the cases needed (as for two
#: arms, one-sided), how often a fresh run of today's size lands on the other side, and the one-sided level.
_SETTLE_LINE_FIELDS = ("distance", "cases_needed", "cases_needed_80", "wrong_side", "level")

#: A hypothetical line's fields: the line itself, then what settling an arm against it reads.
_HYPOTHETICAL_FIELDS = ("line", *_SETTLE_LINE_FIELDS)

#: How each basis of a spread reads, in words.
_BASIS_WORDS: dict[str, str] = {
    "interval": "its stated interval",
    "half_width": "its stated half-width",
    "sem": "its standard error",
    "binomial": "a pass/fail rate's binomial spread, from its count",
}


@dataclass(frozen=True)
class _Place:
    """One slice a separation key reads within: how ``<where>`` spells it, each arm's figures there, and its cases."""

    where: str
    per_arm: Mapping[str, Mapping[str, Stat]]
    n_cases: int | None


def _separation_places(report_input: ReportInput) -> list[_Place]:
    """Every slice a separation key can read within: each task, each cell its spelling names alone, and each stratum
    whose id the input holds once."""
    spellings = [_cell_spelling(cell) for cell in report_input.cells.values()]
    ids = [stratum.id for stratum in report_input.strata]
    return [
        *(_Place(f"task.{task.id}", task.per_arm, task.n_cases) for task in report_input.tasks),
        *(
            _Place(f"cells.{spelled}", cell.per_arm, cell.n_cases)
            for cell, spelled in zip(report_input.cells.values(), spellings, strict=True)
            if spellings.count(spelled) == 1
        ),
        *(
            _Place(f"strata.{stratum.id}", stratum.per_arm, stratum.n_cases)
            for stratum in report_input.strata
            if ids.count(stratum.id) == 1
        ),
    ]


def _stat_where(report_input: ReportInput, arm_id: str, key: str, where: str) -> tuple[Stat, float | None] | None:
    """An arm's statistic on a measure or judged dimension at ``where``, with the count it rests on (its own ``n``,
    else its ``n_cases``, else the place's: the slice's cases, or overall the arm's usable trials, else its cases);
    None where the input states no mean there."""
    if where == _OVERALL:
        table = report_input.measures.get(key) or report_input.judged.get(key)
        stat = table.per_arm.get(arm_id) if table is not None else None
        board = report_input.scoreboard.get(arm_id)
        fallback = None if board is None else board.usable if board.usable is not None else board.cases
    else:
        place = next((held for held in _separation_places(report_input) if held.where == where), None)
        stat = place.per_arm.get(arm_id, {}).get(key) if place is not None else None
        fallback = None if place is None else place.n_cases
    if stat is None or stat.mean is None:
        return None
    count = stat.n if stat.n is not None else stat.n_cases if stat.n_cases is not None else fallback
    return stat, None if count is None else float(count)


def _wheres(report_input: ReportInput, arms: Sequence[str], key: str) -> list[str]:
    """Where every one of ``arms`` holds a mean on ``key``: overall first, then each slice in the input's order."""
    candidates = [_OVERALL, *(place.where for place in _separation_places(report_input))]
    return [where for where in candidates if all(_stat_where(report_input, arm, key, where) for arm in arms)]


def _separated_keys(report_input: ReportInput, arms: Sequence[str]) -> list[str]:
    """The measures and judged dimensions every one of ``arms`` holds a mean on somewhere."""
    keys = dict.fromkeys([*report_input.measures, *report_input.judged])
    return [key for key in keys if _wheres(report_input, arms, key)]


def _rate_of(report_input: ReportInput, key: str) -> Literal["share", "percent"] | None:
    """The scale a pass/fail rate is on (a measure graded pass/fail or by class label, a pass/fail judged dimension),
    which a binomial spread reads; None for any other figure."""
    measure = report_input.measures.get(key)
    judged = report_input.judged.get(key)
    graded = (measure is not None and measure.grade in ("pass_fail", "class_label")) or (
        judged is not None and judged.scale == "pass_fail"
    )
    scale, _unit = _scale_of(report_input, key)
    if graded and scale in ("share", "percent"):
        return "share" if scale == "share" else "percent"
    return None


def _stated_level(report_input: ReportInput, key: str) -> float | None:
    """The level a measure's or judged dimension's stated intervals are at, as a share; None where it states none."""
    table = report_input.measures.get(key) or report_input.judged.get(key)
    return None if table is None else level_share(table.level)


def _where_words(where: str) -> str:
    return "overall" if where == _OVERALL else f"in {where}"


def _spread(report_input: ReportInput, arm_id: str, key: str, where: str, path: str) -> Spread:
    """An arm's spread on a figure at ``where`` (:func:`~threetears.evals.analysis.separation.stat_spread`), refused as
    unstated, with why, where the input gives none."""
    held = _stat_where(report_input, arm_id, key, where)
    if held is None:
        raise _unstated(path, f"names {arm_id}'s mean on {key!r} {_where_words(where)}, which the input does not state")
    stat, count = held
    try:
        return stat_spread(
            float(stat.mean) if stat.mean is not None else None,
            ci=None if stat.ci is None else (float(stat.ci[0]), float(stat.ci[1])),
            half_width=None if stat.half_width is None else float(stat.half_width),
            sem=None if stat.sem is None else float(stat.sem),
            n=count,
            stated_level=_stated_level(report_input, key),
            rate=_rate_of(report_input, key),
        )
    except NoSpread as none:
        raise _unstated(
            path, f"reads {arm_id}'s spread on {key!r} {_where_words(where)}, and the figure has none: {none}"
        ) from none


def _apart(report_input: ReportInput, path: str, key: str, value: float, what: str, *, change: bool) -> Figure:
    """A difference, distance or spread on a figure: in points for a rate, else in the figure's units."""
    scale, unit = _scale_of(report_input, key)
    if scale in ("share", "percent"):
        return _number(path, value * 100 if scale == "share" else value, what, "points", change=change)
    return _number(path, value, what, scale, unit=unit, change=change)


def _settled(path: str, settle: Callable[[], float]) -> float:
    """A settle computation's number, refused as unstated, with why, where the arithmetic has none."""
    try:
        return settle()
    except NoSpread as none:
        raise _unstated(path, f"has no figure: {none}") from none


def _spread_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, key, where, field = values
    spread = _spread(report_input, arm_id, key, where, path)
    if field == "mean":
        scale, unit = _scale_of(report_input, key)
        return _number(path, spread.mean, "the arm's mean", scale, unit=unit)
    if field in ("se", "per_case"):
        value = spread.se if field == "se" else spread.per_case
        if value is None:
            raise _unstated(path, "names the spread of one case, which needs the count the figure rests on")
        return _apart(report_input, path, key, value, f"the figure's {field}", change=False)
    if field == "n":
        return _number(path, spread.n, "the count the figure's spread rests on", "count")
    if field == "level":
        return _number(path, spread.level, "the level a result read from it is stated at", "share")
    return Figure(path=path, kind="text", value=_BASIS_WORDS[spread.basis])


def _family_level(report_input: ReportInput, contrast: Contrast) -> float | None:
    """A contrast's interval's level: its own, else the one family correcting its aspect's."""
    own = level_share(contrast.level)
    if own is not None:
        return own
    families = [family for family in report_input.contrast_families if family.aspect == contrast.aspect]
    return level_share(families[0].level) if len(families) == 1 else None


def _paired(report_input: ReportInput, arm_id: str, other: str, key: str) -> Contrast | None:
    """The one contrast stating one of the two arms against the other (the control) on ``key``, with its difference
    and interval: the paired estimate, which the compare family reads overall in place of the arms' own spreads."""
    control = _control(report_input)
    if control is None or control.id not in (arm_id, other):
        return None
    candidate = other if arm_id == control.id else arm_id
    held = [
        contrast
        for contrast in report_input.contrasts
        if contrast.arm == candidate
        and contrast.measure == key
        and contrast.control in (None, control.id)
        and contrast.delta is not None
        and contrast.interval is not None
    ]
    return held[0] if len(held) == 1 else None


def _compare_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, other, key, where, field = values
    if arm_id == other:
        raise _refuse(path, "compares an arm with itself; name two arms")
    contrast = _paired(report_input, arm_id, other, key) if where == _OVERALL else None
    if field == "difference" and contrast is None:
        means = [_stat_where(report_input, arm, key, where) for arm in (arm_id, other)]
        if means[0] is None or means[1] is None or means[0][0].mean is None or means[1][0].mean is None:
            raise _unstated(path, f"names the two arms' means on {key!r} {_where_words(where)}, which one lacks")
        difference = float(means[0][0].mean) - float(means[1][0].mean)
        return _apart(report_input, path, key, difference, "the two arms' difference", change=True)
    level: float | None
    if contrast is not None and contrast.delta is not None and contrast.interval is not None:
        sign = 1.0 if contrast.arm == arm_id else -1.0
        difference = sign * float(contrast.delta)
        low, high = sorted((sign * float(contrast.interval[0]), sign * float(contrast.interval[1])))
        level = _family_level(report_input, contrast)
        basis = (
            f"paired: the contrast of {contrast.arm} against the control on {key!r}, its own interval, corrected "
            "within its family"
        )
    else:
        first = _spread(report_input, arm_id, key, where, path)
        second = _spread(report_input, other, key, where, path)
        level = first.level
        difference, half = independent_difference(first, second, level)
        low, high = difference - half, difference + half
        basis = (
            f"independent: each arm's own spread ({_BASIS_WORDS[first.basis]}; {_BASIS_WORDS[second.basis]}), "
            "combined as if the arms ran different cases, which reads wider than a paired interval where they ran the "
            "same"
        )
    if field == "basis":
        return Figure(path=path, kind="text", value=basis)
    if field == "level":
        if level is None:
            raise _unstated(path, "names the level of the contrast's interval, which the input does not state")
        return _number(path, level, "the interval's level", "share")
    if field == "interval":
        scale, unit = _scale_of(report_input, key)
        factor = 100.0 if scale == "share" else 1.0
        shown: Scale = "points" if scale in ("share", "percent") else scale
        return Figure(
            path=path,
            kind="interval",
            value=(low * factor, high * factor),
            scale=shown,
            unit="" if shown == "points" else unit,
            change=True,
        )
    value = {"difference": difference, "low": low, "high": high, "half_width": (high - low) / 2}[field]
    return _apart(report_input, path, key, value, f"the difference's {field}", change=field != "half_width")


def _settle_pair_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, other, key, where, field = values
    if arm_id == other:
        raise _refuse(path, "settles an arm against itself; name two arms")
    first = _spread(report_input, arm_id, key, where, path)
    second = _spread(report_input, other, key, where, path)
    difference = first.mean - second.mean
    if field == "difference":
        return _apart(report_input, path, key, difference, "the two arms' difference", change=True)
    if field == "level":
        return _number(path, first.level, "the two-sided level", "share")
    if field == "wrong_side":
        share = _settled(path, lambda: wrong_side_share(difference, math.hypot(first.se, second.se)))
        return _number(path, share, "how often a fresh run of today's size reads the other arm ahead", "share")
    if first.per_case is None or second.per_case is None:
        raise _unstated(path, "names the cases a design needs, which needs the count each arm's figure rests on")
    per_case = math.hypot(first.per_case, second.per_case)
    power = POWER if field == "cases_needed_80" else 0.5
    cases = _settled(path, lambda: cases_to_separate(per_case, difference, first.level, sided=2, power=power))
    return _number(path, cases, "the cases each arm needs", "count")


def _settle_line(
    report_input: ReportInput,
    path: str,
    spread: Spread,
    key: str,
    line: float,
    higher_is_better: bool,
    field: str,
) -> Figure:
    """What settling an arm's figure against a line reads, at ``field``: one-sided, toward passing."""
    distance = spread.mean - line if higher_is_better else line - spread.mean
    if field == "distance":
        return _apart(report_input, path, key, distance, "the figure's distance from the line", change=True)
    if field == "level":
        return _number(path, spread.level, "the one-sided level", "share")
    if field == "wrong_side":
        share = _settled(path, lambda: wrong_side_share(distance, spread.se))
        return _number(path, share, "how often a fresh run of today's size lands on the other side", "share")
    per_case = spread.per_case
    if per_case is None:
        raise _unstated(path, "names the cases a design needs, which needs the count the figure rests on")
    power = POWER if field == "cases_needed_80" else 0.5
    cases = _settled(path, lambda: cases_to_separate(per_case, distance, spread.level, sided=1, power=power))
    return _number(path, cases, "the cases a design needs", "count")


def _criterion_wheres(report_input: ReportInput, criterion: Criterion, arm_id: str) -> list[str]:
    """Where a criterion reads an arm, as ``<where>`` spells it."""
    found: list[str] = []
    for figure in criterion_figures(report_input, criterion, arm_id):
        if figure.cell is not None:
            found.append(f"cells.{_cell_spelling(report_input.cells[figure.cell])}")
        else:
            found.append(_OVERALL if figure.task is None else f"task.{figure.task}")
    return found


def _settle_criterion_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion_id, arm_id, where, field = values
    criterion = _criterion(report_input, criterion_id)
    if criterion.statistic != "mean":
        raise _unstated(
            path, f"reads a {criterion.statistic}, which states no spread to size a design by; settle reads a mean"
        )
    spread = _spread(report_input, arm_id, criterion.measure, where, path)
    higher = criterion.direction == "higher_is_better"
    return _settle_line(report_input, path, spread, criterion.measure, float(criterion.threshold), higher, field)


def _directed(report_input: ReportInput, key: str) -> bool | None:
    """Which way a measure or judged dimension is better: a judged dimension higher; None where a measure states none."""
    if key in report_input.judged:
        return True
    return report_input.measures[key].higher_is_better


def _settle_hypothetical_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    line_text, arm_id, key, where, field = values
    line = float(line_text)
    if field == "line":
        scale, unit = _scale_of(report_input, key, (line,))
        return _number(path, line, "the hypothetical line", scale, unit=unit)
    higher = _directed(report_input, key)
    if higher is None:
        raise _refuse(path, f"reads a line on {key!r}, which states no direction (higher or lower is better)")
    spread = _spread(report_input, arm_id, key, where, path)
    return _settle_line(report_input, path, spread, key, line, higher, field)


def separation_wheres(report_input: ReportInput, arms: Sequence[str], key: str) -> list[str]:
    """Where a ``spread.``, ``compare.`` or ``settle.`` key can read every one of ``arms`` on a measure or judged
    dimension, as its ``<where>`` slot spells it: ``overall`` first, then ``task.<task>``, ``cells.<row>.<column>`` and
    ``strata.<stratum>``, each where every arm states a mean.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arms: The arms, by id.
    :ptype arms: Sequence[str]
    :param key: The measure or judged dimension, by key.
    :ptype key: str
    :return: The places, in the input's order; empty where the arms share none.
    :rtype: list[str]
    """
    return _wheres(report_input, arms, key)


def criterion_wheres(report_input: ReportInput, criterion_id: str, arm: str) -> list[str]:
    """Where a ``settle.criteria.`` key reads an arm against a declared criterion, as its ``<where>`` slot spells it:
    ``overall``, ``task.<task>`` or ``cells.<row>.<column>``, each place the criterion reads a figure for the arm.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param criterion_id: The criterion, by id.
    :ptype criterion_id: str
    :param arm: The arm, by id.
    :ptype arm: str
    :return: The places, in the criterion's order; empty where it reads no figure for the arm.
    :rtype: list[str]
    :raises StopIteration: The input declares no criterion of that id.
    """
    return _criterion_wheres(report_input, _criterion(report_input, criterion_id), arm)


def _other_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return [arm.id for arm in report_input.arms if arm.id != prior[-1]]


def _pair_keys(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return _separated_keys(report_input, prior[-2:])


def _pair_wheres(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return _wheres(report_input, prior[-3:-1], prior[-1])


def _one_arm_keys(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return _separated_keys(report_input, prior[-1:])


def _one_arm_wheres(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return _wheres(report_input, prior[-2:-1], prior[-1])


def _criterion_figure_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    criterion = _criterion(report_input, prior[0])
    return [arm.id for arm in report_input.arms if criterion_figures(report_input, criterion, arm.id)]


def _criterion_figure_wheres(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    criterion_id, arm_id = prior
    return _criterion_wheres(report_input, _criterion(report_input, criterion_id), arm_id)


def _keyed(name: str, read: Callable[[ReportInput], Iterable[str]]) -> _Ids:
    return _Ids(name, lambda report_input, _prior: read(report_input))


_ARM = _Ids("<arm>", _arms)
_CANDIDATE = _Ids("<candidate>", _costed_candidates)
#: The slots every key on one aspect begins with.
_ON_ASPECT = (_Word("aspects"), _Ids("<aspect>", _aspects))
#: The slots a strata key begins with: unqualified, and on one aspect.
_STRATUM = (_Word("strata"), _Ids("<stratum>", _strata))
_ASPECT_STRATUM = (*_ON_ASPECT, _Word("strata"), _Ids("<stratum>", _aspect_strata))
#: How a cell is named, and a criterion's place.
_CELL = _Ids("<cell>", _cell_spellings)


def _family_of(
    *slots: _Slot,
    what: str,
    resolve: Callable[[ReportInput, tuple[str, ...], str], Figure],
    derived: bool = False,
    derived_statistics: tuple[str, ...] = (),
    example: str = "",
    derived_note: str = "",
) -> _Family:
    return _Family(
        slots=slots,
        what=what,
        resolve=resolve,
        derived=derived,
        derived_statistics=derived_statistics,
        example=example,
        derived_note=derived_note,
    )


#: What the grammar says of a separation figure, in place of "descriptive, no interval".
_SEPARATION_NOTE = (
    "Derived: computed here from the figures' stated spreads (normal approximation), never a test the pipeline ran."
)

#: The separation families (an arm's spread, two arms' difference, the cases that would settle a reading), read by
#: ``arms_compare`` and ``settle_cases``; every figure derived.
_SEPARATION_FAMILIES: tuple[_Family, ...] = (
    _family_of(
        _Word("spread"),
        _ARM,
        _Ids("<measure>", _one_arm_keys),
        _Ids("<where>", _one_arm_wheres),
        _Statistic(_SPREAD_FIELDS, "<field>"),
        what="an arm's spread on a measure or judged dimension, `overall` or within a slice (`task.<task>`, "
        "`cells.<row>.<column>`, `strata.<stratum>`): `se` the standard error of its mean and `per_case` one case's "
        "spread, in points for a rate; `n` the count it rests on; `level` the level a result read from it is stated "
        "at; `basis` what the standard error was read from (a stated interval or half-width at a stated level, a "
        "standard error, or a pass/fail rate's binomial spread from its count)",
        resolve=_spread_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="spread.fast.v2.tone.kept.overall.se",
    ),
    _family_of(
        _Word("compare"),
        _ARM,
        _Ids("<arm>", _other_arms),
        _Ids("<measure>", _pair_keys),
        _Ids("<where>", _pair_wheres),
        _Statistic(_COMPARE_FIELDS, "<field>"),
        what="the first arm minus the second on a measure or judged dimension, `overall` or within a slice, in points "
        "for a rate, with its interval: paired where a contrast states one against the other (the control) "
        "overall, else the two arms' own spreads combined as independent; `basis` says which",
        resolve=_compare_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="compare.fast.v2.base.tone.kept.overall.interval|pts",
    ),
    _family_of(
        _Word("settle"),
        _ARM,
        _Ids("<arm>", _other_arms),
        _Ids("<measure>", _pair_keys),
        _Ids("<where>", _pair_wheres),
        _Statistic(_SETTLE_PAIR_FIELDS, "<field>"),
        what="the cases each of two arms needs for an interval on their difference, centred on today's, to clear "
        "zero at the two-sided level: `cases_needed` (half such designs clear if the truth equals today's "
        "difference), `cases_needed_80` (four in five do); `wrong_side` how often a fresh run of today's size reads "
        "the other arm ahead",
        resolve=_settle_pair_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="settle.fast.v2.base.tone.kept.overall.cases_needed",
    ),
    _family_of(
        _Word("settle"),
        _Word("criteria"),
        _keyed("<criterion>", lambda ri: [criterion.id for criterion in ri.criteria]),
        _Ids("<arm>", _criterion_figure_arms),
        _Ids("<where>", _criterion_figure_wheres),
        _Statistic(_SETTLE_LINE_FIELDS, "<field>"),
        what="an arm's mean against a declared criterion's threshold, where it reads it: `distance` toward passing, "
        "the cases a design needs for a one-sided interval centred on today's figure to clear the line "
        "(`cases_needed`: half such designs; `cases_needed_80`: four in five), and `wrong_side`, how often a fresh "
        "run of today's size lands on the other side",
        resolve=_settle_criterion_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="settle.criteria.cell_floor.fast.v2.cells.short.en.cases_needed",
    ),
    _family_of(
        _Word("settle"),
        _Word("hypothetical"),
        _Number("<line>"),
        _ARM,
        _Ids("<measure>", _one_arm_keys),
        _Ids("<where>", _one_arm_wheres),
        _Statistic(_HYPOTHETICAL_FIELDS, "<field>"),
        what="an arm's mean against a hypothetical line the snapshot does not declare, written in the figure's own "
        "units: what a declared criterion's settle reads, the line itself as `line`; every figure hypothetical",
        resolve=_settle_hypothetical_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="settle.hypothetical.0.9.fast.v2.tone.kept.overall.wrong_side",
    ),
)

# --- lever effects: one setting moved between two arms that differ in it alone ------------------------------------

#: A lever effect's fields: the second arm's mean on the measure minus the first's (``effect``), and the interval on it
#: from the two arms' own intervals.
_LEVER_FIELDS = ("effect", "interval", "low", "high")

#: An interaction's fields: the second move's effect minus the first's (``difference``), and the interval on it.
_INTERACTION_FIELDS = ("difference", "interval", "low", "high")


def arm_settings(report_input: ReportInput) -> dict[str, dict[str, SettingValue]]:
    """Every arm's settings, each setting any arm states: a setting an arm leaves out has the control's value, as the
    input's own rule says; with no control it is unstated (None), which agrees only with another unstated one.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Each arm's settings, by arm id then setting, in the input's orders.
    :rtype: dict[str, dict[str, SettingValue]]
    """
    control = _control(report_input)
    base: Mapping[str, SettingValue] = control.settings if control is not None else {}
    names = list(dict.fromkeys(name for arm in report_input.arms for name in arm.settings))
    return {arm.id: {name: arm.settings.get(name, base.get(name)) for name in names} for arm in report_input.arms}


def lever_of(settings: Mapping[str, Mapping[str, SettingValue]], first: str, second: str) -> str | None:
    """The one setting two arms differ in, by :func:`arm_settings`; None where they differ in none or in several.

    :param settings: Every arm's settings, as :func:`arm_settings` reads them.
    :ptype settings: Mapping[str, Mapping[str, SettingValue]]
    :param first: One arm's id.
    :ptype first: str
    :param second: The other's.
    :ptype second: str
    :return: The setting, or None.
    :rtype: str | None
    """
    differ = [name for name, value in settings[first].items() if settings[second].get(name) != value]
    return differ[0] if len(differ) == 1 else None


def _lever_settings(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    """Every setting some two arms differ in alone, in the order the arms state settings."""
    settings = arm_settings(report_input)
    arms = list(settings)
    held = {lever_of(settings, first, second) for first in arms for second in arms if first != second}
    return [name for name in dict.fromkeys(n for values in settings.values() for n in values) if name in held]


def _lever_partners(settings: Mapping[str, Mapping[str, SettingValue]], setting: str, arm: str) -> list[str]:
    """The arms whose settings differ from one arm's in ``setting`` alone."""
    return [other for other in settings if other != arm and lever_of(settings, arm, other) == setting]


def _lever_froms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    settings = arm_settings(report_input)
    return [arm for arm in settings if _lever_partners(settings, prior[0], arm)]


def _lever_tos(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return _lever_partners(arm_settings(report_input), prior[0], prior[1])


def _lever_measures(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys([*report_input.measures, *report_input.judged]))


def _lever_places(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    """Every place a lever's effect can be read within: a task, a stratum the input holds on one aspect, a cell."""
    strata = [stratum.id for stratum in report_input.strata]
    return list(
        dict.fromkeys(
            [
                *(task.id for task in report_input.tasks),
                *(held for held in strata if strata.count(held) == 1),
                *(_cell_spelling(cell) for cell in report_input.cells.values()),
            ]
        )
    )


def _place_figures(report_input: ReportInput, place: str, path: str) -> Mapping[str, Mapping[str, Stat]]:
    """Each arm's figures within one place, by arm then measure."""
    held = [
        *(task.per_arm for task in report_input.tasks if task.id == place),
        *(stratum.per_arm for stratum in report_input.strata if stratum.id == place),
        *(cell.per_arm for cell in report_input.cells.values() if _cell_spelling(cell) == place),
    ]
    if len(held) > 1:
        raise _refuse(path, f"names place {place!r}, which the input holds {len(held)} times (a task, stratum or cell)")
    return held[0]


def _lever_mean(
    report_input: ReportInput, key: str, arm: str, place: str | None, path: str
) -> tuple[float, tuple[float, float] | None]:
    """One arm's mean on a measure, overall or within a place, and the interval its figure states on it (``ci``, or
    the mean give or take ``half_width``); None for no interval."""
    if place is None:
        table = report_input.measures.get(key) or report_input.judged.get(key)
        stat = table.per_arm.get(arm) if table is not None else None
    else:
        stat = _place_figures(report_input, place, path).get(arm, {}).get(key)
    where = "" if place is None else f" in {place}"
    if stat is None or stat.mean is None:
        raise _unstated(path, f"names {arm}'s mean on {key}{where}, which the input does not state")
    mean = float(stat.mean)
    if stat.ci is not None:
        return mean, (float(stat.ci[0]), float(stat.ci[1]))
    if stat.half_width is not None:
        return mean, (mean - float(stat.half_width), mean + float(stat.half_width))
    return mean, None


def _recovered(first: tuple[float, float, float], second: tuple[float, float, float]) -> tuple[float, float]:
    """The interval on ``second``'s estimate minus ``first``'s, each ``(estimate, low, high)``, from their own
    intervals by the method of variance estimates recovery, the two read as independent."""
    (one, one_low, one_high), (two, two_low, two_high) = first, second
    change = two - one
    return change - math.hypot(two - two_low, one_high - one), change + math.hypot(two_high - two, one - one_low)


def _lever_effect(
    report_input: ReportInput, values: tuple[str, ...], path: str
) -> tuple[float, tuple[float, float] | None, str]:
    """A move's effect, the interval on it, and why there is none: from ``values`` = (from, to, measure, place)."""
    first, second, key, place = values
    one, one_interval = _lever_mean(report_input, key, first, place or None, path)
    two, two_interval = _lever_mean(report_input, key, second, place or None, path)
    if one_interval is None or two_interval is None:
        missing = first if one_interval is None else second
        return two - one, None, f"{missing}'s figure on {key} states no interval"
    return two - one, _recovered((one, *one_interval), (two, *two_interval)), ""


def _lever_figure(
    report_input: ReportInput,
    key: str,
    change: float,
    interval: tuple[float, float] | None,
    why: str,
    field: str,
    path: str,
) -> Figure:
    scale, unit = _scale_of(report_input, key)
    if field in ("effect", "difference"):
        return Figure(path=path, kind="number", value=change, scale=scale, unit=unit, change=True)
    if interval is None:
        raise _unstated(path, f"names an interval, which the input does not state: {why}")
    if field == "interval":
        return Figure(path=path, kind="interval", value=interval, scale=scale, unit=unit, change=True)
    end = interval[0] if field == "low" else interval[1]
    return Figure(path=path, kind="number", value=end, scale=scale, unit=unit, change=True)


def _lever(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    """``lever.<setting>.<from>.<to>.<measure>[.<place>].<field>``."""
    _setting_id, first, second, key, *place, field = values
    change, interval, why = _lever_effect(report_input, (first, second, key, place[0] if place else ""), path)
    return _lever_figure(report_input, key, change, interval, why, field, path)


def _interaction_froms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    """The arms a second move can start from: the first's setting value, in a context one other setting away."""
    setting, first, second = prior[:3]
    settings = arm_settings(report_input)
    return [
        arm
        for arm in settings
        if arm not in (first, second)
        and settings[arm][setting] == settings[first][setting]
        and lever_of(settings, first, arm) not in (None, setting)
        and _lever_partners(settings, setting, arm)
    ]


def _interaction_tos(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    """The arms a second move can end at: the first move's value, its context the second start's."""
    setting, first, second = prior[:3]
    start = prior[-1]
    settings = arm_settings(report_input)
    across = lever_of(settings, first, start)
    return [
        arm
        for arm in _lever_partners(settings, setting, start)
        if settings[arm][setting] == settings[second][setting] and lever_of(settings, second, arm) == across
    ]


def _interaction(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    """``lever.<setting>.<from>.<to>.<measure>[.<place>].vs.<from>.<to>.<field>``: the second move's effect minus the
    first's, and the interval on it from theirs."""
    _setting_id, first, second, key, *place, start, end, field = values
    where = place[0] if place else ""
    one, one_interval, one_why = _lever_effect(report_input, (first, second, key, where), path)
    two, two_interval, two_why = _lever_effect(report_input, (start, end, key, where), path)
    if one_interval is None or two_interval is None:
        return _lever_figure(report_input, key, two - one, None, one_why or two_why, field, path)
    interval = _recovered((one, *one_interval), (two, *two_interval))
    return _lever_figure(report_input, key, two - one, interval, "", field, path)


#: The slots every lever key begins with: the setting moved, the arm it moves from and the one it moves to (differing
#: in it alone), and the measure read.
_LEVER = (
    _Word("lever"),
    _Ids("<setting>", _lever_settings),
    _Ids("<from>", _lever_froms),
    _Ids("<to>", _lever_tos),
    _Ids("<measure>", _lever_measures),
)
#: The slots an interaction's second move is named by.
_SECOND_MOVE = (_Word("vs"), _Ids("<from>", _interaction_froms), _Ids("<to>", _interaction_tos))
#: Where a lever's effect is read within.
_PLACE = _Ids("<place>", _lever_places)

#: What the grammar says of a lever figure, in place of "descriptive, no interval".
_LEVER_NOTE = "Derived: computed here from two arms' stated figures and their intervals, never a test the pipeline ran."

#: What every lever key's figures are: computed here, from two arms' figures the input states.
_LEVER_BASIS = (
    "Computed here: `interval` (its ends `low`, `high`) is recovered from the two figures' own intervals as if "
    "independent (MOVER), so it is conservative where they share cases; with either figure stating none there is "
    "no interval. levers_effects lists them"
)


#: What the grammar says of a priced move, in place of "descriptive, no interval".
_PRICE_NOTE = (
    "Derived: computed here from the options' stated costs and gains; an interval carried through is labelled paired "
    "(a contrast's) or unpaired (two options' own, as if independent), never a test the pipeline ran."
)
#: What the grammar says of a pooled figure, in place of "descriptive, no interval".
_POOLED_NOTE = (
    "Derived: pooled here from the slices' stated figures with the weights `weights` names; its interval combines "
    "theirs as if independent, never a test the pipeline ran."
)

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
        _Statistic(_SLICE_STATISTICS),
        what="an arm's figure on a measure or judged dimension within one task; `vs_control` is its mean minus the "
        "control's there",
        resolve=_task_figure,
        derived_statistics=_SLICE_DERIVED,
    ),
    _family_of(
        _Word("criteria"),
        _keyed("<criterion>", lambda ri: [criterion.id for criterion in ri.criteria]),
        _Statistic(_CRITERION_FIELDS, "<field>"),
        what="a declared criterion: its threshold in its measure's units, the measure, statistic and direction",
        resolve=_criterion_field,
    ),
    _family_of(
        *_STRATUM,
        _Statistic(_STRATUM_FIELDS, "<field>"),
        what="a stratum",
        resolve=_strata_resolver(_stratum_field, qualified=False),
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
        *_STRATUM,
        _Word("loss_share"),
        what="the least share of any candidate's loss on the primary judged dimension that the stratum carries",
        resolve=_strata_resolver(_loss_share, qualified=False),
        derived=True,
    ),
    _family_of(
        *_STRATUM,
        _Word("loss_share"),
        _Ids("<candidate>", _candidates),
        what="the share of one candidate's loss the stratum carries",
        resolve=_strata_resolver(_loss_share, qualified=False),
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
    _family_of(
        *_ON_ASPECT,
        _Word("contrast"),
        _Ids("<arm>", _aspect_contrast_arms),
        _Ids("<measure>", _aspect_contrasted),
        _Statistic(_CONTRAST_STATISTICS),
        what="a corrected contrast on one aspect, where `contrast.` reads an arm and measure the input holds on more "
        "than one",
        resolve=_aspect_contrast,
        example="aspects.answers.contrast.fast.v2.tone.kept.delta|pts",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("measure"),
        _Ids("<measure>", _aspect_measures),
        _Ids("<arm>", _aspect_measure_arms),
        _Statistic(_MEASURE_STATISTICS),
        what="an arm's figure on a measure on one aspect",
        resolve=_aspect_measure,
        example="aspects.speed.measure.latency_ms.fast.v2.mean",
    ),
    _family_of(
        _Word("judged"),
        _keyed("<dimension>", lambda ri: ri.judged),
        _Ids("<arm>", _judged_arms),
        _Statistic(_MEASURE_STATISTICS),
        what="one statistic of an arm's figure on a judged dimension",
        resolve=_judged_stat,
        example="judged.tone.kept.base.n",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("judged"),
        _Ids("<dimension>", _aspect_judged_dimensions),
        _Ids("<arm>", _aspect_judged_arms),
        _Statistic(_MEASURE_STATISTICS),
        what="an arm's figure on a judged dimension on one aspect",
        resolve=_aspect_judged,
        example="aspects.answers.judged.tone.kept.fast.v2.mean|pct",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("scoreboard"),
        _Ids("<arm>", _aspect_board_arms),
        _Statistic(_SCOREBOARD_STATISTICS),
        what="how an arm's trials came out on one aspect",
        resolve=_aspect_scoreboard,
        example="aspects.speed.scoreboard.fast.v2.passed",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("pair"),
        _Ids("<arm>", _pair_firsts),
        _Ids("<arm>", _pair_seconds),
        _Statistic(_PAIR_STATISTICS),
        what="a pair reading on one aspect: the first arm minus the second, in points, as the input orders them; "
        "`ahead` is the arm shown ahead",
        resolve=_pair,
        example="aspects.speed.pair.base.fast.v2.change|pts",
    ),
    _family_of(
        _Word("aspects"),
        _Ids("<aspect>", _floor_aspects),
        _Word("floor"),
        _Ids("<floor>", _floor_keys),
        _Ids("<arm>", _floor_arms),
        _Statistic(_FLOOR_FIELDS, "<field>"),
        what="a floor read on one candidate, on the aspect the read names (one the input may hold no aspect record "
        "for): `held_above` is the margin above which it holds (minus the interval's low end), `broken_at` the margin "
        "at or below which it breaks (minus the high end)",
        resolve=_floor,
        derived_statistics=_FLOOR_DERIVED,
        example="aspects.answers.floor.answers/rate_drop/*.fast.v2.held_above",
    ),
    _family_of(
        _Word("aspects"),
        _Ids("<aspect>", _bar_aspects),
        _Word("bar"),
        _Ids("<measure>", _bar_measures),
        _Ids("<arm>", _bar_arms),
        _Statistic(_BAR_FIELDS, "<field>"),
        what="an arm against an absolute bar, in the measure's units, on the aspect the read names (one the input may "
        "hold no aspect record for)",
        resolve=_bar,
        example="aspects.speed.bar.latency_ms.fast.v2.value",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("family"),
        _Statistic(_FAMILY_FIELDS, "<field>"),
        what="the contrast family correcting one aspect's contrasts, where the input holds more than one",
        resolve=_aspect_family,
        example="aspects.answers.family.correction",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("flip_margin"),
        _Ids("<candidate>", _named_reading_arms),
        what="the margin above which a candidate's reading on one aspect holds: its interval's worst drop",
        resolve=_aspect_flip_margin,
        derived=True,
        example="aspects.answers.flip_margin.fast.v2",
    ),
    _family_of(
        *_STRATUM,
        _Ids("<arm>", _stratum_arms(qualified=False)),
        _Ids("<measure>", _stratum_keys(qualified=False)),
        _Statistic(_SLICE_STATISTICS),
        what="an arm's figure on a measure or judged dimension within one stratum; `vs_control` is its mean minus the "
        "control's there",
        resolve=_strata_resolver(_stratum_figure, qualified=False),
        derived_statistics=_SLICE_DERIVED,
        example="strata.off_topic.fast.v2.tone.kept.vs_control|pts",
    ),
    _family_of(
        *_STRATUM,
        _Word("loss_share"),
        _Ids("<candidate>", _candidates),
        _Ids("<measure>", _loss_keys(qualified=False)),
        what="the share of one candidate's loss on any measure or judged dimension the stratum carries, a loss "
        "running the way the measure is worse",
        resolve=_strata_resolver(_loss_share, qualified=False),
        derived=True,
        example="strata.off_topic.loss_share.fast.v2.tone.kept|pct",
    ),
    _family_of(
        *_ASPECT_STRATUM,
        _Statistic(_STRATUM_FIELDS, "<field>"),
        what="a stratum of one aspect's cases, where the input holds its id on more than one aspect",
        resolve=_strata_resolver(_stratum_field, qualified=True),
        example="aspects.answers.strata.off_topic.n_cases",
    ),
    _family_of(
        *_ASPECT_STRATUM,
        _Ids("<arm>", _stratum_arms(qualified=True)),
        _Ids("<measure>", _stratum_keys(qualified=True)),
        _Statistic(_SLICE_STATISTICS),
        what="an arm's figure within a stratum of one aspect's cases",
        resolve=_strata_resolver(_stratum_figure, qualified=True),
        derived_statistics=_SLICE_DERIVED,
        example="aspects.answers.strata.off_topic.fast.v2.tone.kept.mean|pct",
    ),
    _family_of(
        *_ASPECT_STRATUM,
        _Word("loss_share"),
        what="the least share of any candidate's loss on the primary judged dimension that a stratum of one aspect "
        "carries",
        resolve=_strata_resolver(_loss_share, qualified=True),
        derived=True,
    ),
    _family_of(
        *_ASPECT_STRATUM,
        _Word("loss_share"),
        _Ids("<candidate>", _candidates),
        what="the share of one candidate's loss a stratum of one aspect carries",
        resolve=_strata_resolver(_loss_share, qualified=True),
        derived=True,
    ),
    _family_of(
        *_ASPECT_STRATUM,
        _Word("loss_share"),
        _Ids("<candidate>", _candidates),
        _Ids("<measure>", _loss_keys(qualified=True)),
        what="the share of one candidate's loss on any measure that a stratum of one aspect carries",
        resolve=_strata_resolver(_loss_share, qualified=True),
        derived=True,
    ),
    _family_of(
        _Word("cells"),
        _CELL,
        _Statistic(_CELL_FIELDS, "<field>"),
        what="a target-matrix cell, written by its row and column level ids joined by a dot (`<row>.<column>`)",
        resolve=_cell_field,
        example="cells.short.en.name",
    ),
    _family_of(
        _Word("cells"),
        _CELL,
        _Ids("<arm>", _cell_arms),
        _Ids("<measure>", _cell_keys),
        _Statistic(_SLICE_STATISTICS),
        what="an arm's figure on a measure or judged dimension within one cell; `vs_control` is its mean minus the "
        "control's there",
        resolve=_cell_figure,
        derived_statistics=_SLICE_DERIVED,
        example="cells.short.en.fast.v2.tone.kept.mean|pct",
    ),
    _family_of(
        _Word("criteria"),
        _keyed("<criterion>", lambda ri: [criterion.id for criterion in ri.criteria]),
        _Ids("<arm>", _criterion_arms),
        _Statistic(_CRITERION_FIGURE_FIELDS, "<field>"),
        what="the one figure a criterion reads for an arm, in its measure's units; `gap` is how far it clears the "
        "threshold, signed toward passing, in points for a rate",
        resolve=_criterion_arm,
        derived_statistics=_CRITERION_DERIVED,
        example="criteria.p95.fast.v2.gap",
    ),
    _family_of(
        _Word("criteria"),
        _keyed("<criterion>", lambda ri: [criterion.id for criterion in ri.criteria]),
        _Ids("<arm>", _criterion_arms),
        _Ids("<place>", _criterion_places),
        _Statistic(_CRITERION_FIGURE_FIELDS, "<field>"),
        what="a criterion's figure for an arm at one place it reads: a task by id, a cell as `<row>.<column>`",
        resolve=_criterion_arm_at,
        derived_statistics=_CRITERION_DERIVED,
        example="criteria.cell_floor.fast.v2.short.en.value|pct",
    ),
    _family_of(
        _Word("settings"),
        _Ids("<arm>", _setting_arms),
        _Ids("<setting>", _setting_names),
        what="one of an arm's settings as the input holds it: a number where it holds a number, else its text",
        resolve=_setting,
        example="settings.base.retries",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("cases"),
        what="the cases one aspect ran: the control's, else the count every arm agrees on",
        resolve=lambda report_input, values, path: _aspect_depth(report_input, (values[0], "cases"), path),
        example="aspects.answers.cases",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("repeats"),
        what="the mean repeats each case of one aspect ran: the control's, else the figure every arm agrees on",
        resolve=lambda report_input, values, path: _aspect_depth(report_input, (values[0], "repeats"), path),
        example="aspects.answers.repeats",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("depth"),
        _Ids("<arm>", _depth_arms),
        _Statistic(_DEPTH_FIELDS, "<field>"),
        what="how deeply one arm ran one aspect: its distinct cases, its trials, the mean repeats a case got",
        resolve=_arm_depth,
        example="aspects.answers.depth.fast.v2.repeats",
    ),
    _family_of(
        _Word("late_criteria"),
        _Ids("<late>", _late_names),
        _Word("set_at"),
        what="when one criterion was set after the results were seen, named as its caveat id names it after `late:` "
        "(what was set, then `:` and which one where it names one)",
        resolve=_late_criterion_set_at,
        example="late_criteria.floor:answers/rate_drop/*.set_at",
    ),
    _family_of(
        _Word("families"),
        _Ids("<family>", _family_places),
        _Statistic(_FAMILY_FIELDS, "<field>"),
        what="any contrast family by its place in the input's families, counting from 0: the key for one that names "
        "no aspect, or shares its aspect with another",
        resolve=_indexed_family,
        example="families.1.correction",
    ),
    _family_of(
        _Word("price"),
        _keyed("<from>", _costs_per_thousand),
        _Ids("<to>", _gain_arms),
        _Ids("<measure>", _gain_axes),
        _Statistic(_PRICE_FIELDS, "<field>"),
        what="a move from one option to another priced on one gain axis (a measure with a direction, or a judged "
        "dimension): `gain` is how much better the second reads, toward better; `cost_gap` what it costs more per "
        "1,000 replies; `per_point` the cost gap per point gained (per unit off a points scale), its `_low` and "
        "`_high` ends read off `gain_interval`, the contrast's interval where the move is between the control and a "
        "candidate; an end the interval leaves open is refused as unbounded; `monthly_` fields at the declared "
        "monthly volume. Cost is read as exact",
        resolve=_price,
        derived=True,
        derived_note=_PRICE_NOTE,
        example="price.base.fast.v2.latency_ms.monthly_per_point",
    ),
    _family_of(
        _Word("task"),
        _Ids("<task>", _tasks),
        _Word("price"),
        _Ids("<from>", _slice_arms("task")),
        _Ids("<to>", _slice_seconds("task")),
        _Ids("<measure>", _gain_axes),
        _Statistic(_SLICE_PRICE_FIELDS, "<field>"),
        what="a move priced within one task, on the arms' figures there: the task's own cost where it states one for "
        "every option it places, else each option's overall cost, as `cost_basis` says; its interval is unpaired, the "
        "two options' own half-widths there combined as independent",
        resolve=_slice_price("task"),
        derived=True,
        derived_note=_PRICE_NOTE,
    ),
    _family_of(
        *_STRATUM,
        _Word("price"),
        _Ids("<from>", _slice_arms("strata")),
        _Ids("<to>", _slice_seconds("strata")),
        _Ids("<measure>", _gain_axes),
        _Statistic(_SLICE_PRICE_FIELDS, "<field>"),
        what="a move priced within one stratum, as within a task",
        resolve=_slice_price("strata"),
        derived=True,
        derived_note=_PRICE_NOTE,
    ),
    _family_of(
        _Word("cells"),
        _CELL,
        _Word("price"),
        _Ids("<from>", _slice_arms("cells")),
        _Ids("<to>", _slice_seconds("cells")),
        _Ids("<measure>", _gain_axes),
        _Statistic(_SLICE_PRICE_FIELDS, "<field>"),
        what="a move priced within one target-matrix cell, as within a task",
        resolve=_slice_price("cells"),
        derived=True,
        derived_note=_PRICE_NOTE,
    ),
    _family_of(
        _Word("pooled"),
        _Ids("<measure>", _pooled_keys),
        _Ids("<arm>", _pooled_arms),
        _Statistic(_POOLED_FIELDS, "<field>"),
        what="an arm's figure pooled over the slices holding a measure or judged dimension (its tasks, else its "
        "strata of one stratification, else its cells): their means weighted by traffic share where the input states "
        "one for every slice, else equally, as `weights` says; `half_width` combines the slices' as independent; "
        "`slices` counts them. A price reads it where no overall figure is stated",
        resolve=_pooled,
        derived=True,
        derived_note=_POOLED_NOTE,
        example="pooled.tone.kept.fast.v2.weights",
    ),
    _family_of(
        _Word("cost_per_month"),
        _keyed("<arm>", _costs_per_thousand),
        what="an arm's cost per month at the declared monthly volume",
        resolve=_cost_per_month,
        derived=True,
        example="cost_per_month.base",
    ),
    *_SEPARATION_FAMILIES,
    _family_of(
        *_LEVER,
        _Statistic(_LEVER_FIELDS, "<field>"),
        what="a lever's effect: one setting moved between two arms whose settings differ in it alone, `<to>`'s mean "
        "on a measure or judged dimension minus `<from>`'s (`effect`), overall. " + _LEVER_BASIS,
        resolve=_lever,
        derived=True,
        derived_note=_LEVER_NOTE,
    ),
    _family_of(
        *_LEVER,
        _PLACE,
        _Statistic(_LEVER_FIELDS, "<field>"),
        what="a lever's effect within one task, stratum or cell (`<row>.<column>`). " + _LEVER_BASIS,
        resolve=_lever,
        derived=True,
        derived_note=_LEVER_NOTE,
    ),
    _family_of(
        *_LEVER,
        *_SECOND_MOVE,
        _Statistic(_INTERACTION_FIELDS, "<field>"),
        what="an interaction: the same move made where one other setting differs, its effect minus the first move's "
        "(`difference`), overall. " + _LEVER_BASIS,
        resolve=_interaction,
        derived=True,
        derived_note=_LEVER_NOTE,
    ),
    _family_of(
        *_LEVER,
        _PLACE,
        *_SECOND_MOVE,
        _Statistic(_INTERACTION_FIELDS, "<field>"),
        what="an interaction within one task, stratum or cell. " + _LEVER_BASIS,
        resolve=_interaction,
        derived=True,
        derived_note=_LEVER_NOTE,
    ),
    _family_of(
        _Word("count"),
        _Statistic(_COUNTED, "<what>"),
        what="how many the campaign holds: arms (every option, the control among them), aspects, tasks, strata, "
        "cells, configs, gates (gates and guardrails), criteria; `surviving` is how many options survive every gate "
        "and guardrail, borderline counting as passing",
        resolve=_campaign_count,
        derived_statistics=("surviving",),
        example="count.arms|words",
    ),
    _family_of(
        _Word("count.passing"),
        _keyed("<criterion>", _gate_ids),
        what="how many options clear one gate or guardrail at every place it reads (a borderline figure clears it)",
        resolve=_passing_count,
        derived=True,
        example="count.passing.p95",
    ),
    _family_of(
        _Word("count.places"),
        _keyed("<criterion>", _criterion_ids),
        what="how many places a criterion reads an arm's figure at: one overall, else each task or cell it reads in",
        resolve=_places_count,
        example="count.places.cell_floor",
    ),
    _family_of(
        _Word("count.failing"),
        _keyed("<criterion>", _criterion_ids),
        _ARM,
        what="how many of a criterion's places an arm misses: a figure on the wrong side of the threshold, or none",
        resolve=_failing_count,
        derived=True,
        example="count.failing.cell_floor.fast.v2|words",
    ),
    _family_of(
        _Word("count.failed"),
        _ARM,
        what="how many gates and guardrails an arm misses somewhere they read",
        resolve=_failed_count,
        derived=True,
        example="count.failed.fast.v2",
    ),
    _family_of(
        _Word("count.frontier"),
        _Ids("<measure>", _gain_axes),
        what="how many options are best at some price per point on a gain axis: the price frontier's options",
        resolve=_frontier_count,
        derived=True,
        example="count.frontier.tone.kept|words",
    ),
    _family_of(
        _Word("count.steps"),
        _Ids("<measure>", _gain_axes),
        what="how many steps the price frontier on a gain axis takes: the prices at which the best option moves",
        resolve=_steps_count,
        derived=True,
        example="count.steps.tone.kept",
    ),
    _family_of(
        _Word("count.levels"),
        _keyed("<factor>", _factor_ids),
        what="how many levels one factor of a factorial grid holds",
        resolve=_levels_count,
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
        options: Sequence[str] = (slot.spelling,)
    elif isinstance(slot, _Ids):
        options = list(slot.options(report_input, values))
    elif isinstance(slot, _Number):
        options = _number_options(path, at)
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
        empty = "a number" if isinstance(slot, _Number) else "none in this input"
        expected = tuple(options) if options else (f"{_slot_name(slot)} ({empty})",)
        if at > stop.at:
            stop.at, stop.expected = at, expected
        elif at == stop.at:
            stop.expected = tuple(dict.fromkeys((*stop.expected, *expected)))


def _slot_name(slot: _Slot) -> str:
    return slot.spelling if isinstance(slot, _Word) else slot.name


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
        parts.append(f"`{slot.spelling}`" if isinstance(slot, _Word) else f"{_slot_name(slot)} `{next(values)}`")
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
    reading = _one_reading(key, report_input)
    figure = reading.family.resolve(report_input, reading.values, key)
    return replace(figure, derived=_is_derived(reading))


def _one_reading(key: str, report_input: ReportInput) -> _Reading:
    """The one reading of a path; a path reading two ways, or none, is refused naming the readings or what could
    come next where it stopped matching."""
    found, stop = _readings(key, report_input)
    if len(found) > 1:
        readings = "; ".join(_named(reading) for reading in found)
        raise _refuse(key, f"reads {len(found)} ways: {readings}. Choose ids that leave one reading.")
    if not found:
        if stop.at <= 0:
            raise _refuse(key, f"names no figure; a key begins with {_listed(_beginnings())}")
        raise _refuse(key, f"names no figure: after `{key[: stop.at]}` comes {_listed(stop.expected)}")
    return found[0]


def _is_derived(reading: _Reading) -> bool:
    """Whether a reading's figure is computed here: its family is derived, or the statistic it names is."""
    if reading.family.derived:
        return True
    values = iter(reading.values)
    for slot in reading.family.slots:
        if isinstance(slot, _Word):
            continue
        value = next(values)
        if isinstance(slot, _Statistic) and value in reading.family.derived_statistics:
            return True
    return False


@dataclass(frozen=True)
class KeySubjects:
    """What one key is about: the arms and aspects its path names, and whether it reads an arm against the control.

    ``arms`` are the ids the path's arm or candidate slots matched (or the arm of ``{arm:<id>}``); ``aspects`` the
    aspect ids it named, or the asked aspect for a family that reads it without naming it (``reading``, ``aspect``,
    ``cases``, ``repeats``, ``flip_margin``). ``against_control`` is True for a figure that is a difference from the control (a
    reading, a contrast, a cost uplift, a flip margin, a stratum's loss share, a slice's ``vs_control``), whose
    evidence is the control's too.
    """

    arms: tuple[str, ...]
    aspects: tuple[str, ...]
    against_control: bool


#: The slots an arm id fills.
_ARM_SLOTS = frozenset({"<arm>", "<candidate>", "<from>", "<to>"})

#: The families that read the asked aspect without naming it.
_ASKED_FAMILIES = frozenset({"aspect", "reading", "cases", "repeats", "flip_margin"})

#: The words that make a figure a difference from the control.
_AGAINST_CONTROL = frozenset(
    {"reading", "contrast", "cost.uplift", "cost.uplift_range", "flip_margin", "loss_share", *_SLICE_DERIVED}
)

#: The statistics that make a figure a difference from the control.
_AGAINST_CONTROL_STATISTICS = frozenset(_SLICE_DERIVED)


def key_subjects(key: str, report_input: ReportInput) -> KeySubjects:
    """The arms and aspects one key's path names, read off its one reading of the input.

    What a durable claim rests on is read from its keys: the runs behind each arm the key names on each aspect it
    names (each insight a planned analysis mints,
    :func:`~threetears.evals.analysis.planned_analysis.generate_planned_analysis`).

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
            words.append(slot.spelling)
            continue
        value = next(values)
        if isinstance(slot, _Statistic) and value in _AGAINST_CONTROL_STATISTICS:
            words.append(value)
        elif isinstance(slot, _Ids) and slot.name in _ARM_SLOTS:
            arms.append(value)
        elif isinstance(slot, _Ids) and slot.name == "<aspect>":
            aspects.append(value)
    if not aspects and words[0] in _ASKED_FAMILIES and (asked := _asked(report_input)) is not None:
        aspects.append(asked.id)
    return KeySubjects(
        arms=tuple(arms), aspects=tuple(aspects), against_control=any(word in _AGAINST_CONTROL for word in words)
    )


@dataclass(frozen=True)
class KeyForm:
    """What one key's path is, as the grammar teaches it: the form of the family it reads, what that family reads,
    and the help page the form is listed on.

    :ivar form: The family's form, its ids as slot names (``price.<from>.<to>.<measure>.<field>``); ``arm:<id>`` for
        an arm.
    :ivar what: What the family reads, as the grammar says it.
    :ivar group: The group of the grammar's paths the form is listed in (``figure_keys.<group>``).
    """

    form: str
    what: str
    group: str


#: An arm key's form, as a key check names it.
_ARM_FORM = KeyForm(form="arm:<id>", what="an arm, printed by its short name and letter tag", group="campaign")


def key_form(key: str, report_input: ReportInput) -> KeyForm:
    """The form one key's path reads by: its family, what the family reads and the grammar's page listing it.

    A path is matched as :func:`resolve` matches it, so a path that names nothing, or reads two ways, is refused
    alike; one naming a figure the input leaves unstated still has a form.

    :param key: The path, without braces or filters.
    :ptype key: str
    :param report_input: The report input the key is read against.
    :ptype report_input: ReportInput
    :return: The form.
    :rtype: KeyForm
    :raises UnresolvableReference: The path names nothing the input holds, or reads two ways.
    """
    if key.startswith(_ARM_PREFIX):
        return _ARM_FORM
    family = _one_reading(key, report_input).family
    group = next(name for name, families in _grouped().items() if family in families)
    return KeyForm(form=family.form(), what=family.what, group=group)


def figure_filters(figure: Figure) -> tuple[str, ...]:
    """The filters a figure takes: each filter :func:`format_figure` spells it with, alone, in the grammar's order.

    :param figure: What a key resolved to.
    :ptype figure: Figure
    :return: The filters; none for an arm or a piece of text.
    :rtype: tuple[str, ...]
    """
    if figure.kind in ("arm", "text"):
        return ()
    taken: list[str] = []
    for name in _FILTERS:
        try:
            format_figure(figure, (name,))
        except UnresolvableReference:
            # NOSILENT: a filter the figure does not suit is refused by format_figure; that refusal is the answer
            # sought here, and the filter is left off the list it returns.
            continue
        taken.append(name)
    return tuple(taken)


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
    if spelling == "words":
        whole = round(value)
        if figure.scale != "count" or whole != value or whole < 0:
            why = "it is not a whole count" if figure.scale == "count" else f"it is on the {figure.scale} scale"
            raise _refuse(figure.path, f"cannot read as a word: {why}")
        if whole < len(COUNT_WORDS):
            return _Spelled(COUNT_WORDS[whole])
        return _Spelled(group_thousands(format_number(float(whole))))
    if spelling == "int":
        # A price per 1,000,000 replies also reads as a whole number, as stored plans spell it.
        if figure.scale not in ("count", "plain", "points", "usd_per_1m"):
            raise _refuse(figure.path, f"cannot read as a whole number: it is on the {figure.scale} scale")
        text = group_thousands(format_number(float(round(value))))
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
    """A small made-up input the grammar's examples are rendered on, with a dotted arm id and a dotted dimension: one
    asked aspect and one other, with a measure on both, a pair reading, a floor, a bar, strata, a cell, criteria, a
    numeric setting, each arm's depth, a criterion set late, a second contrast family naming no aspect, and a
    monthly volume."""
    rate = {"base": (0.912, 897), "fast.v2": (0.874, 895)}
    return ReportInput.model_validate(
        {
            "schema": "report-input/1",
            "meta": {
                "campaign": "Example",
                "kind": "change",
                "question": "Is fast.v2 as good as base?",
                "volume_per_month": 1_000_000,
            },
            "arms": [
                {
                    "id": "base",
                    "name": "Baseline",
                    "role": "control",
                    "served_model": "vendor/base-7",
                    "settings": {"model": "vendor/base-7", "max_output": "64k", "retries": 2},
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
                    "depth": {
                        "base": {"cases": 300, "trials": 900, "repeats_per_case": [3, 3.0, 3]},
                        "fast.v2": {"cases": 300, "trials": 897, "repeats_per_case": [2, 2.99, 3]},
                    },
                },
                {
                    "id": "speed",
                    "readings": [
                        {"arm": "fast.v2", "reading": "improved", "change": 7.4, "within_interval": [3.6, 11.2]}
                    ],
                    "pairs": [
                        {
                            "arms": ["base", "fast.v2"],
                            "change": -7.4,
                            "interval": [-11.2, -3.6],
                            "confidence": 0.95,
                            "ahead": "fast.v2",
                        }
                    ],
                    "scoreboard": {
                        "base": {"ran": 300, "usable": 298, "passed": 240},
                        "fast.v2": {"ran": 300, "usable": 299, "passed": 262},
                    },
                },
            ],
            "floors": [
                {
                    "floor": "answers/rate_drop/*",
                    "aspect": "answers",
                    "arm": "fast.v2",
                    "status": "held",
                    "interval": [-7.1, -1.4],
                    "worst_drop": 7.1,
                    "margin": 10,
                }
            ],
            "bars": [
                {
                    "aspect": "speed",
                    "arm": "fast.v2",
                    "measure": "latency_ms",
                    "bar": 900,
                    "direction": "lower_is_better",
                    "status": "met",
                    "value": 640,
                    "interval": [610, 672],
                }
            ],
            "criteria": [
                {
                    "id": "p95",
                    "kind": "gate",
                    "measure": "latency_ms",
                    "statistic": "p95",
                    "direction": "lower_is_better",
                    "threshold": 1500,
                },
                {
                    "id": "cell_floor",
                    "kind": "floor",
                    "measure": "tone.kept",
                    "direction": "higher_is_better",
                    "scope": "each_cell",
                    "threshold": 0.85,
                },
            ],
            "scoreboard": {
                "base": {"passed": 765, "usable": 897, "reply_cost_usd": 0.001892},
                "fast.v2": {"passed": 731, "usable": 895, "reply_cost_usd": 0.002427},
            },
            "measures": {
                "latency_ms": {
                    "unit": "ms",
                    "higher_is_better": False,
                    "per_arm": {"base": {"mean": 820, "p95": 1310}, "fast.v2": {"mean": 655, "p95": 980}},
                    "by_aspect": {"speed": {"base": {"mean": 790}, "fast.v2": {"mean": 640, "ci": [610, 672]}}},
                }
            },
            "judged": {
                "tone.kept": {
                    "scale": "pass_fail",
                    "per_arm": {arm: {"mean": mean, "n": n} for arm, (mean, n) in rate.items()},
                }
            },
            "primary_judged": "tone.kept",
            "contrasts": [
                {
                    "arm": "fast.v2",
                    "measure": "tone.kept",
                    "kind": "judged",
                    "aspect": "answers",
                    "verdict": "regressed",
                    "delta": -0.038,
                    "interval": [-0.061, -0.015],
                },
                {
                    "arm": "fast.v2",
                    "measure": "latency_ms",
                    "kind": "measure",
                    "aspect": "speed",
                    "verdict": "improved",
                    "delta": -165,
                    "interval": [-190, -140],
                },
            ],
            "contrast_families": [
                {"aspect": "answers", "correction": "holm", "alpha": 0.05, "family_size": 1},
                {"correction": "none", "family_size": 1},
            ],
            "strata": [
                {
                    "id": "off_topic",
                    "aspect": "answers",
                    "n_cases": 40,
                    "per_arm": {"base": {"tone.kept": {"mean": 0.70}}, "fast.v2": {"tone.kept": {"mean": 0.55}}},
                },
                {
                    "id": "greeting",
                    "aspect": "answers",
                    "n_cases": 60,
                    "per_arm": {"base": {"tone.kept": {"mean": 0.95}}, "fast.v2": {"tone.kept": {"mean": 0.85}}},
                },
            ],
            "target_matrix": {
                "rows": {"id": "length", "levels": [{"id": "short", "name": "Short"}]},
                "columns": {"id": "lang", "levels": [{"id": "en", "name": "English"}]},
            },
            "cells": {
                "short|en": {
                    "row": "short",
                    "column": "en",
                    "n_cases": 50,
                    "per_arm": {"base": {"tone.kept": {"mean": 0.94}}, "fast.v2": {"tone.kept": {"mean": 0.9}}},
                }
            },
            "caveats": {
                "late_criteria": [{"criterion": "floor", "name": "answers/rate_drop/*", "set_at": "25 Oct 2026"}]
            },
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
    "words": "count.arms|words",
}


def _family_line(family: _Family, example: ReportInput, names: Mapping[str, str]) -> str:
    """One family's line in the grammar: its form, what it reads, its statistics, what of it is derived, and its
    example rendered on the made-up input."""
    statistic = family.statistic()
    listed = f" ({statistic.name}: {', '.join(statistic.options)})" if statistic else ""
    tag = (
        (f" {family.derived_note}" if family.derived_note else " Derived: descriptive, no interval.")
        if family.derived
        else ""
    )
    if family.derived_statistics:
        derived = ", ".join(f"`{name}`" for name in family.derived_statistics)
        tag += f" {derived} derived: descriptive, no interval."
    if family.example:
        shown = render(f"{{{family.example}}}", example, names)
        tag += f' `{{{family.example}}}` reads "{shown}".'
    return f"- `{family.form()}`: {family.what}{listed}.{tag}"  # one line: module constants


def _grammar_rules() -> list[str]:
    """The grammar's rules, after its heading and before its paths: how a key is written and how a path is matched."""
    return [
        "This section is written by the code that reads your keys, so it is current: where anything above describes "
        "figure keys differently, follow this section.",
        "Never type a number into prose. Where a sentence needs one, write a figure key: `{path|filter|filter}`, "
        "in single braces. (`{{…}}` is a different grammar and is refused here.) Write `{arm:<id>}` where a sentence "
        "names an arm: it prints the arm's short name and letter tag.",
        "A path is matched against the ids this input holds, never split at dots: write an id exactly as the input "
        "spells it, dots and spaces included (`reading.fast.v2.change` is arm `fast.v2`). A path that reads two ways, "
        "or names nothing the input holds, is refused and sent back naming what could come next. Where the input "
        "holds a figure on more than one aspect (a contrast, a stratum, a contrast family), the form without "
        "`aspects.<aspect>.` is refused and names the `aspects.<aspect>.` keys to write instead.",
    ]


def _grammar_filters(example: ReportInput, names: Mapping[str, str]) -> list[str]:
    """The grammar's filters, each with its example rendered, and what a key prints: everything after its paths."""
    lines = ["Filters (one spelling at most, plus `abs` and `interval`):"]
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
        "refused unless it is part of a name the input holds: an arm's id or name, a served model, a setting held as "
        "text (such as `64k`; one held as a number is a key), an aspect, measure, dimension, stratum or task id or name, the campaign's name, "
        "a trial or case id. An id made only of digits is never such a name: write `{arm:1017}`, not `1017`. "
        f"Percentile names ({', '.join(f'`{name}`' for name in PERCENTILE_NAMES)}) and the package's unit phrases "
        f"({', '.join(f'`{phrase}`' for phrase in _UNIT_PHRASES)}, whole) are written as they are; any other number "
        "is a key."
    )
    return lines


#: The names the grammar's examples print each arm by.
_EXAMPLE_NAMES: Mapping[str, str] = {"base": "Baseline", "fast.v2": "fast.v2"}


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
    lines = [f"## {GRAMMAR_HEADING}", *_grammar_rules(), "Paths:"]
    lines += [_family_line(family, example, _EXAMPLE_NAMES) for family in _FAMILIES]
    lines += _grammar_filters(example, _EXAMPLE_NAMES)
    return "\n".join(lines)


# --- the grammar, as a reference read a page at a time -------------------------------------------------------------

#: The groups the grammar's path families are paged in, in the order the index lists them: each group's name, what its
#: figures are of, and the subjects of the families it holds (:func:`_subject`). Every family's subject is in one.
_FAMILY_GROUPS: tuple[tuple[str, str, frozenset[str]], ...] = (
    (
        "campaign",
        "the campaign, its arms, aspects and settings, and how deeply each arm ran",
        frozenset(
            {
                "meta",
                "control",
                "n_candidates",
                "aspect",
                "aspects",
                "settings",
                "cases",
                "repeats",
                "depth",
                "late_criteria",
                "caveats",
            }
        ),
    ),
    (
        "readings",
        "an arm's reading against the control on an aspect, pair readings and flip margins",
        frozenset({"reading", "pair", "flip_margin"}),
    ),
    (
        "contrasts",
        "corrected contrasts against the control, and the families correcting them",
        frozenset({"contrast", "family", "families"}),
    ),
    (
        "measures",
        "an arm's figures on a measure or judged dimension, its pass rate and scoreboard, and the judge",
        frozenset({"measure", "judged", "judge", "pass_rate", "scoreboard", "reasoning_ratio", "pooled"}),
    ),
    (
        "slices",
        "figures within one task, stratum or target-matrix cell, and the share of a loss each carries",
        frozenset({"task", "strata", "cells"}),
    ),
    (
        "gates",
        "declared criteria, floors and bars, and the weakest margin",
        frozenset({"criteria", "floor", "bar", "weakest"}),
    ),
    (
        "cost",
        "cost per reply and per month, cost uplift, and a move between options priced per point",
        frozenset({"cost_per_1k", "cost", "price", "cost_per_month"}),
    ),
    (
        "separation",
        "an arm's spread on a figure, two arms' difference with its interval, and the cases that would settle a "
        "reading against a line or two arms apart",
        frozenset({"spread", "compare", "settle"}),
    ),
    (
        "levers",
        "the effect of one setting moved between arms that differ in it alone, and how it changes with another setting",
        frozenset({"lever"}),
    ),
    (
        "counts",
        "how many of a thing a sentence counts: arms, slices, gates, the options passing each, the places one misses",
        frozenset({"count"}),
    ),
)


def _subject(family: _Family) -> str:
    """What a family's figures are of: its first word, or the word after ``aspects.<aspect>.`` in a qualified form, up
    to any dot a word is written with (``cost.uplift`` is of ``cost``)."""
    (first, *rest) = family.slots
    qualified = (
        isinstance(first, _Word) and first.spelling == "aspects" and len(rest) > 1 and isinstance(rest[1], _Word)
    )
    word = rest[1] if qualified else first
    return word.spelling.split(".")[0] if isinstance(word, _Word) else word.name


def _grouped() -> dict[str, list[_Family]]:
    """Every family, by the group it is paged in, groups and families each in the grammar's order.

    :raises ValueError: A family's subject is in no group: a family was added without a page to be read on.
    """
    grouped: dict[str, list[_Family]] = {name: [] for name, _, _ in _FAMILY_GROUPS}
    for family in _FAMILIES:
        group = next((name for name, _, subjects in _FAMILY_GROUPS if _subject(family) in subjects), None)
        if group is None:
            raise ValueError(f"figure key family `{family.form()}` is in no group of _FAMILY_GROUPS: place it in one")
        grouped[group].append(family)
    return grouped


def figure_key_index(topic: str) -> str:
    """The grammar as a reference's index: its rules and filters whole, and one line per group of path families, naming
    the page each group is read on and its forms.

    The grammar (:func:`figure_key_grammar`) runs past what one tool result may take of a reader's context, so a
    reference serves it in pages (:func:`figure_key_pages`); the index and the pages together hold every line of it.

    :param topic: What the reference is served as: each group's page is ``<topic>.<group>``.
    :ptype topic: str
    :return: The index, heading first.
    :rtype: str
    :raises UnresolvableReference: An example no longer resolves, as :func:`figure_key_grammar` raises it.
    :raises ValueError: A family is in no group.
    """
    lines = [f"## {GRAMMAR_HEADING}", *_grammar_rules()]
    lines.append(
        f"Paths, by group: each group's forms, their statistics and an example are its page, topic='{topic}.<group>'."
    )
    summaries = {name: what for name, what, _ in _FAMILY_GROUPS}
    for group, families in _grouped().items():
        forms = ", ".join(f"`{family.form()}`" for family in families)
        lines.append(f"- {topic}.{group}: {summaries[group]}. {forms}.")
    lines += _grammar_filters(_example_input(), _EXAMPLE_NAMES)
    return "\n".join(lines)


def figure_key_pages() -> dict[str, str]:
    """Each group of the grammar's path families as a page: its forms, what each reads, its statistics and an example,
    written as the grammar writes them (:func:`figure_key_grammar`).

    :return: Each page, by its group's name, in the order the index lists them (:func:`figure_key_index`).
    :rtype: dict[str, str]
    :raises UnresolvableReference: An example no longer resolves, as :func:`figure_key_grammar` raises it.
    :raises ValueError: A family is in no group.
    """
    example = _example_input()
    pages: dict[str, str] = {}
    for (group, families), (_, what, _) in zip(_grouped().items(), _FAMILY_GROUPS, strict=True):
        lines = [f"## {GRAMMAR_HEADING}: {group}", f"Paths of {what}:"]
        lines += [_family_line(family, example, _EXAMPLE_NAMES) for family in families]
        pages[group] = "\n".join(lines)
    return pages


__all__ = [
    "COUNT_WORDS",
    "FILTER_SPELLINGS",
    "PERCENTILE_NAMES",
    "WEAKEST_PREFIX",
    "Figure",
    "FigureKey",
    "GainAxis",
    "PooledFigure",
    "PriceFrontier",
    "KeyForm",
    "KeySubjects",
    "NamedKind",
    "arm_settings",
    "criterion_wheres",
    "POOL_WEIGHT_WORDS",
    "SLICE_COST_WORDS",
    "SliceKind",
    "figure_key_grammar",
    "figure_key_index",
    "figure_filters",
    "figure_key_pages",
    "flip_margin",
    "format_figure",
    "gain_axis",
    "key_form",
    "key_subjects",
    "keys_in",
    "lever_of",
    "literal_numbers",
    "parse_key",
    "pooled_figures",
    "price_frontier",
    "reader_name",
    "reader_names",
    "render",
    "resolve",
    "separation_wheres",
]
