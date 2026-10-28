"""Figure keys: the report plan's prose names a figure by its place in the report input, and code writes the number.

The planner never types a figure into a sentence (GENERATOR.md §6.2). It writes a **figure key** —
``{<path>|<filter>|<filter>}`` — and :func:`render` replaces each with the value read from the
:class:`~threetears.evals.analysis.report_input.ReportInput`, spelled by the package's number rule
(:mod:`threetears.evals.kernel.numbers`, :mod:`threetears.evals.analysis.quantities`). ``{arm:<id>}`` prints an
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
axis (:func:`gain_axis`): the gain, signed toward better, the cost gap per 1,000 of what the cost counts
(:func:`cost_basis`), and the price per point
gained, whose low and high ends are read off the gain's interval where a contrast states one (an end the interval
leaves open is refused as unbounded, never written as a number). ``cost_per_month.<arm>`` and the ``monthly_`` price
fields read the same at the declared ``meta.volume_per_month``. All are derived and marked so.

**A cost some trials did not report.** ``cost.unreported.<arm>.<field>`` reads the page's cost
(``cost_per_1k.<arm>``) where some of the arm's trials report no cost: ``missing`` counts them (the trials the arm's
scoreboard says ran, less the cost measure's observations), ``free`` is the cost per 1,000 had each of them cost
nothing (the mean times observations over trials run: the least the arm's mean over every trial can be), and
``at_p95`` had each cost the arm's own 95th percentile, a what-if, not a ceiling: the snapshot holds no highest
cost, so nothing caps what a missing trial could have cost. Its inputs are the measure the page's cost is read from,
overall on the arm (its mean, observations ``n`` and ``p95``), and the arm's ``scoreboard.<arm>.ran``. It reads each
observation as one trial's, so it refuses, as unstated, a cost the page reads from the scoreboard or blends over
tasks (no observations are counted), an arm whose measure or scoreboard leaves ``n`` or ``ran`` unstated, and a
measure with more observations than trials run, whose observations are not one per trial. All derived.

**Two arms apart, and the cases that would settle a reading.** ``spread.<arm>.<measure>.<where>.*`` reads an arm's
spread on a figure, ``compare.<arm>.<arm>.<measure>.<where>.*`` the difference between two arms with its interval (a
contrast's own where one pairs them, else independent), and ``settle.*`` the cases a design needs to separate an arm
from a declared criterion's threshold, a hypothetical line (``settle.hypothetical.<line>.``, the line a number in the
key), or another arm (:mod:`~threetears.evals.analysis.separation`): one-sided against a line, with the two-sided
size beside it, and two-sided between arms, each ``level`` a text that says which. A size counts cases, each run as
often as today, with the trials it takes beside it. ``settle.margin.`` and ``settle.floor.`` settle a candidate's
change from the control against minus an aspect's margin or a floor's, on the reading's own interval. Derived, read
off stated spreads: a normal approximation, never a test the pipeline ran. ``assumed.<cases>.`` (or
``assumed.share.``) before a spread, compare or settle key reads it as if each slice whose figure states no count held
that many cases (or the arm's own distinct cases times the slice's traffic share): a what-if the key names, refused overall and
where the figures state their count.

**An overall that is the plain mean of its cells.** ``overall_interval.<measure>.<arm>.*`` reads the interval on an
arm's overall figure that states none, where that figure is, by the numbers the input states, the unweighted mean of
one measure's target-matrix cells and every cell states its own spread
(:mod:`~threetears.evals.analysis.cell_mean`): its standard error the root of the cells' squared standard errors over
their count, at the cells' stated level. A criterion figure's ``low`` and ``high`` read the same interval there. All
derived, and refused, as unstated with why, wherever the figure does not qualify.

**What else a sentence counts or compares.** ``count.examples`` and ``count.failures`` (per aspect and arm) say how
many failed trials the snapshot froze as examples and how many the run had; ``overlap.<aspect>.<aspect>.`` whether
two aspects ran the same cases; ``confusion.<axis>.<arm>.<expected>.<predicted>`` one cell of a classifier's
confusion; ``cost.gap.<arm>.<arm>.`` two arms' costs apart, per 1,000, per month and as a multiple. A count the input
does not hold (a planner's input counts no failures, a bare one holds no case ids) is refused as unstated, never read
as none.

**Calls that went unanswered, beside the cost.** ``calls.unanswered.<arm>`` (and ``aspects.<aspect>.`` before it)
reads how many of an arm's calls went unanswered, retried and lost together, summed over every usage role that counted
them, with ``.retried``, ``.lost``, ``.calls`` (the answered ones), ``.share`` (of every call sent), ``.reasons`` and
``.uncounted`` (trials whose calls counted none) beside it. Copied where one role counted them; a share over several
roles is computed here and marked derived. An arm no role counted for is no key at all: not counted is never none.

**A figure held on more than one aspect is named by its aspect.** Every figure that sits on an aspect has an
``aspects.<aspect>.`` form. Where the input holds a contrast's arm and measure, a stratum id or a contrast family on
more than one aspect, the unqualified form is refused, naming the qualified key for each aspect, rather than reading
one of them.

**Digits only inside keys (D3).** :func:`literal_numbers` finds every digit that plan prose types outside a key,
outside a name the input holds, outside a percentile's name (``p95``) and outside one of the package's own unit
phrases (``per 1,000 replies``, and ``per 1,000 conversations`` where the input's costs count conversations). A
number a reader sees is code's.

**The grammar is taught by this module.** :func:`figure_key_grammar` writes the planner prompt's section from the
module's own families, statistics and filters, with examples rendered on a synthetic input, so the prompt can never
teach a form this parser refuses. A reader that takes the grammar a page at a time reads the same lines:
:func:`figure_key_index` (the rules and filters, and a line per group of families) and :func:`figure_key_pages` (each
group's forms).
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, get_args

from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.cell_mean import CellMeanInterval, NoCellMean, cell_mean_interval
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.kernel.numbers import MoneyForm, format_number, group_thousands, money_form, spell_multiple
from threetears.evals.analysis.quantities import PERCENT, POINTS, number_in
from threetears.evals.analysis.report.cost_unit import REPLIES, CostBasis, per_count, per_noun
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
    FailedTrials,
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
    Unanswered,
)
from threetears.evals.analysis.report_margins import (
    CriterionFigure,
    Margin,
    criterion_aspect,
    criterion_figures,
    criterion_misses,
    criterion_passes,
    criterion_places,
    criterion_unknowns,
    flip_margin,
    surviving_arms,
    undecided_arms,
    weakest_margin,
)
from threetears.evals.analysis.separation import (
    POWER,
    NoSpread,
    Spread,
    cases_to_separate,
    difference_spread,
    independent_difference,
    level_share,
    spread_interval,
    stat_spread,
    wrong_side_share,
)
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.analysis.style import humanize, measure_name
from threetears.evals.kernel.metrics import ACCURACY_MEASURE, MATCH_MEASURE, goal_check_of

#: What a figure is: a number, an interval (low, high), a piece of text, or an arm (rendered by name and tag).
FigureKind = Literal["number", "interval", "text", "arm"]

#: The scale a figure's number is on, which decides the filters it takes. ``share``: 0 to 1. ``percent``: 0 to 100.
#: ``points``: a difference already in percentage points. ``usd``: dollars per reply or per unit. ``usd_per_1k``:
#: dollars per 1,000 of what the cost counts (``Figure.per``). ``usd_per_1m``: dollars per 1,000,000 of it. ``count``:
#: a count. ``plain``: any other number, in ``Figure.unit`` when it has one.
Scale = Literal["share", "percent", "points", "usd", "usd_per_1k", "usd_per_1m", "count", "plain"]

#: The dollar scales, each with the factor that states its figure per 1,000 of what it counts.
_PER_THOUSAND_FACTORS: dict[str, float] = {"usd": 1000.0, "usd_per_1k": 1.0, "usd_per_1m": 0.001}

#: The filters a key may carry. One spelling at most (``pct``, ``pts``, ``usd``, ``usd_per_1k``, ``int``,
#: ``pct10``), plus ``abs`` and ``interval``, in any order.
Filter = Literal["pct", "pts", "abs", "usd", "int", "pct10", "interval", "usd_per_1k", "usd_whole", "words"]
_FILTERS: tuple[str, ...] = get_args(Filter)
_SPELLINGS: frozenset[str] = frozenset({"pct", "pts", "usd", "usd_per_1k", "int", "pct10", "usd_whole", "words"})
#: The filters that choose a key's spelling, at most one to a key, in the grammar's order.
FILTER_SPELLINGS: tuple[str, ...] = tuple(name for name in _FILTERS if name in _SPELLINGS)

#: How each filter reads, as the grammar teaches it.
_FILTER_MEANS: dict[str, str] = {
    "pct": "a rate as a percent",
    "pts": "a change in points, signed, one decimal at most",
    "abs": "the magnitude, without its sign",
    "usd": "dollars, in one form for the key's whole family: cents, with more decimals where its least amount needs "
    "them, and per 1,000 (`$8/1000`) where any of its amounts is under a cent",
    "int": "the nearest whole number, its thousands separated; a multiple keeps its unit (`21 times`), every other "
    "unit is left for the prose to write",
    "pct10": "a rate rounded to the nearest 10%",
    "interval": "an interval's two ends, joined by 'to'",
    "usd_per_1k": "dollars per 1,000 of what the cost counts, with the words (`per 1,000 conversations` where the "
    "measure's label counts conversations)",
    "usd_whole": "dollars to the whole dollar, thousands grouped, for a monthly or other large amount whose cents are "
    "noise; refused under $10, which it would round away",
    "words": "a count spelled as a word up to twenty, in grouped digits above",
}

#: The least amount :data:`Filter` ``usd_whole`` reads: under it, whole dollars round away too much of the figure.
_WHOLE_DOLLARS_FROM = 10.0

#: The unit phrases a rendered figure is joined with, one template each, so a translation replaces them here; the
#: phrase a cost's count reads in is :func:`~threetears.evals.analysis.report.cost_unit.per_count`'s.
INTERVAL_TEMPLATE = "{low} to {high}"
RANGE_TEMPLATE = "{low}–{high}"
ARM_TEMPLATE = "{name} ({tag})"
#: How a count from zero to twenty reads as a word (the ``words`` filter), in order; a translation replaces them here.
_COUNT_WORDS: tuple[str, ...] = (
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
#: An input adds the phrases of what its own costs count (:func:`_input_unit_phrases`: ``per 1,000 conversations``).
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
    ``held_above``, a criterion figure's ``gap``): descriptive, with no interval and no test behind it. ``per`` is
    what a money figure counts, as a plural noun (``conversations``), where the input states it
    (:mod:`~threetears.evals.analysis.report.cost_unit`); empty where it does not, or the figure is no money.
    ``level`` marks an interval's confidence level, read in percent as the input states it, to every digit the number
    rule keeps (``97.5%``, ``98.33%``), never rounded to a whole percent or to a rate's one decimal: two levels that
    differ never print alike.

    A money figure also carries the set it is spelled with (#993): ``amounts``, its statistic's amount for every arm of its
    family, on its scale, or, for a change, the amounts it compares, from which
    :func:`~threetears.evals.kernel.numbers.money_form` chooses one form for the whole family, so ``cost_per_1k.*`` never
    reads ``$0.008`` beside ``$0.11``; and ``spread``, the width of the interval a change is read with, which caps its
    decimals. :func:`resolve` fills both; a figure made without them is spelled as a set of its own value. Neither takes
    part in a figure's equality: they say how it reads, not what it is.
    """

    path: str
    kind: FigureKind
    value: float | tuple[float, float] | str
    scale: Scale = "plain"
    unit: str = ""
    change: bool = False
    whole: bool = False
    derived: bool = False
    per: str = ""
    level: bool = False
    amounts: tuple[float, ...] = field(default=(), compare=False, repr=False)
    spread: float | None = field(default=None, compare=False, repr=False)


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
    ``compares`` gives, for a family whose money figures are changes, the amounts a change compares (each arm's cost per
    1,000 for a cost gap, the measure's means for a contrast), whose form the change reads in; ``None`` from it, or no
    ``compares``, reads the family's own amounts (:func:`_money_amounts`).
    """

    slots: tuple[_Slot, ...]
    what: str
    resolve: Callable[[ReportInput, tuple[str, ...], str], Figure]
    derived: bool = False
    derived_statistics: tuple[str, ...] = ()
    example: str = ""
    derived_note: str = ""
    compares: Callable[[ReportInput, tuple[str, ...]], Iterable[float] | None] | None = None

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


def on_lead_aspect(report_input: ReportInput, contrast: Contrast) -> bool:
    """Whether a contrast is read on the lead aspect, the one an arm's overall figures (the by-arm maps) belong to: it
    names that aspect, or none. A contrast on another aspect pairs another aspect's figures, never the overall ones.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param contrast: One of its contrasts.
    :ptype contrast: Contrast
    :return: Whether the contrast pairs the overall figures.
    :rtype: bool
    """
    return contrast.aspect is None or _is_lead(report_input, contrast.aspect)


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


def _level(path: str, value: float | None, what: str) -> Figure:
    """An interval's confidence level, a share, read in percent as stated (:class:`Figure`'s ``level``)."""
    return replace(_number(path, value, what, "share"), level=True)


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
        # The level of the interval `interval`, `low` and `high` read (within_interval): the nominal one, with no
        # correction. gain_confidence is the gain interval's, which no reading key reads.
        stated = INTERVAL_LEVEL if reading.within_interval is not None else None
        return _level(path, stated, "the interval's level")
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

    First the measures read before the scoreboard: the one stated per 1,000 (``usd_per_1k``), per 1,000,000,
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


#: The cost source that is the scoreboard's cost per reply, as :class:`CostBasis` names it.
_SCOREBOARD_COST = "scoreboard"


def _cost_source(report_input: ReportInput) -> tuple[str, dict[str, float]]:
    """The first cost source that holds a figure, by name (a measure's key, or :data:`_SCOREBOARD_COST`), and each
    arm's cost per 1,000 on it; ``("", {})`` where none does (:func:`_costs_per_thousand`'s order)."""
    first, lower = _cost_measures(report_input)
    sources = [(key, _overall_cost(report_input, key)) for key in first]
    sources.append(
        (
            _SCOREBOARD_COST,
            {
                a: row.reply_cost_usd * 1000
                for a, row in report_input.scoreboard.items()
                if row.reply_cost_usd is not None
            },
        )
    )
    sources += [(key, _overall_cost(report_input, key)) for key in lower]
    sources += [(key, _blended_cost(report_input, key)) for key in dict.fromkeys([*first, *lower])]
    return next(((name, source) for name, source in sources if source), ("", {}))


def _costs_per_thousand(report_input: ReportInput) -> dict[str, float]:
    """Each arm's cost per 1,000 of what it counts, from the first source that holds one: the one rule a page and the
    decision read cost by (``cost_per_1k.<arm>``).

    The arms' overall figures on a measure stated per 1,000; else on one stated per 1,000,000, divided by 1,000; else
    on the one dollar measure on the cost merit axis, times 1,000; else the scoreboard's cost per reply, times 1,000;
    else the one dollar measure where lower is better. Where none of those states a figure, each measure in the same
    order blended over the tasks by traffic share (:func:`_blended_cost`). What the count counts is
    :func:`cost_basis`'s.
    """
    return _cost_source(report_input)[1]


def cost_basis(report_input: ReportInput) -> CostBasis:
    """What the page's cost (``cost_per_1k.<arm>``) is per, read from the source it is read from.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The noun (``replies`` from the scoreboard or a dollar-per-reply measure; a per-count measure's label's
        noun; empty where unstated, or where the input states no cost) and the source.
    :rtype: CostBasis
    """
    source, _costs = _cost_source(report_input)
    if source == _SCOREBOARD_COST:
        return CostBasis(noun=REPLIES, source=source)
    measure = report_input.measures.get(source)
    return CostBasis(noun=per_noun(measure) if measure is not None else "", source=source)


def _money_per(report_input: ReportInput, values: tuple[str, ...]) -> str:
    """What a money figure read by a path counts: the money measure the path names, else the page's cost basis."""
    for value in values:
        measure = report_input.measures.get(value)
        if measure is not None and measure.unit in _PER_THOUSAND_FACTORS:
            return per_noun(measure)
    return cost_basis(report_input).noun


def _input_unit_phrases(report_input: ReportInput) -> set[str]:
    """The phrases this input's costs are counted in, which prose may write whole: ``per 1,000 conversations`` and
    ``per 1,000,000 conversations`` for each noun a money measure or the cost basis states."""
    nouns = {per_noun(measure) for measure in report_input.measures.values()} | {cost_basis(report_input).noun}
    return {per_count(noun, count) for noun in nouns if noun for count in (1_000, 1_000_000)}


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

#: How the examples count a failure, which ``count.failures`` counts by, and why it is not the scoreboard's ``failed``.
_FAILURES_RULE = (
    "A failure here is a nought on the primary judged dimension where the snapshot names one, else a trial that "
    "failed: it is not `scoreboard.<arm>.failed`, which counts every trial that failed, for any cause, so the two "
    "differ by design where trials fail on something else"
)

#: What the scoreboard's ``failed`` counts, beside ``count.failures``.
_SCOREBOARD_FAILED = (
    "`failed` counts every trial that failed, for any cause; `count.failures.<aspect>.<arm>` counts failures as the "
    "examples do (a nought on the primary judged dimension, where there is one), so the two can differ"
)


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
        return replace(_number(path, getattr(row, statistic), f"the {statistic}", "usd"), per=REPLIES)
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


_CRITERION_FIELDS = ("threshold", "measure", "statistic", "direction", "aspect")


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
    if field == "aspect":
        aspect = criterion_aspect(report_input, criterion)
        if aspect is None or aspect not in {held.id for held in report_input.aspects}:
            raise _unstated(path, "names the aspect the criterion reads, and the input holds no such aspect")
        return _text(path, reader_name("aspect", aspect, report_input), "the aspect whose figures the criterion reads")
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
        read = _number(path, end, f"the figure's interval's {field} end", scale, unit=unit)
        return replace(read, derived=True) if figure.derived else read
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
        return Figure(path=path, kind="number", value=gap * factor, scale=money, change=True, per=per)
    points = 100.0 if scale == "share" else 1.0
    if gain == 0 or gap * gain <= 0:
        raise _refuse(
            path,
            f"is no price: {second} reads no better than {first} on {axis.key!r} for no less cost, or better for no "
            "more, so one is at least as good on both and the move has no price per point",
        )
    if field == "per_point":
        return Figure(path=path, kind="number", value=gap / (gain * points) * factor, scale=money, per=per)
    if interval is None:
        raise _unstated(path, f"names an end of the price's interval, which the input does not state: {unstated}")
    bound = _price_bounds(path, gap, gain * points, (interval[0] * points, interval[1] * points), field)
    return Figure(path=path, kind="number", value=bound * factor, scale=money, per=per)


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
    return Figure(path=path, kind="number", value=_costs_per_thousand(report_input)[values[0]] * factor, scale="usd")


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


# --- counts: what a sentence counts, keyed so prose never types one ------------------------------------------------

#: What ``count.<what>`` counts in the campaign: its arms (every option, the control among them), aspects, tasks,
#: strata, target-matrix cells, configs of a factorial grid, gates and guardrails, every declared criterion, the
#: options surviving every gate and guardrail, the options undecided (no gate figure fails them, and a gate has no
#: figure for them somewhere), the pairs of arms its pair readings read apart and the pairs of arms there are, the
#: failed trials frozen as examples and the failed trials the run holds (:func:`_failures_held`).
_COUNTED = (
    "arms",
    "aspects",
    "tasks",
    "strata",
    "cells",
    "configs",
    "gates",
    "criteria",
    "surviving",
    "undecided",
    "pairs",
    "arm_pairs",
    "examples",
    "failures",
)

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
    if what == "undecided":
        return _count(path, len(undecided_arms(report_input)))
    if what == "pairs":
        read = {frozenset(pair.arms) for aspect in report_input.aspects for pair in aspect.pairs}
        return _count(path, len(read))
    if what == "arm_pairs":
        return _count(path, math.comb(len(report_input.arms), 2))
    if what == "examples":
        return _count(path, len(report_input.examples))
    if what == "failures":
        held = _failures_held(report_input, [aspect.id for aspect in report_input.aspects], path)
        return _count(path, sum(row.failed for row in held))
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


def _unknown_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion_id, arm_id = values
    return _count(path, criterion_unknowns(report_input, _criterion(report_input, criterion_id), arm_id))


def _failed_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (arm_id,) = values
    missed = [criterion for criterion in _gates(report_input) if criterion_misses(report_input, criterion, arm_id)]
    return _count(path, len(missed))


def _factor_ids(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return [factor.id for factor in report_input.factors]


def _levels_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    factor = next(factor for factor in report_input.factors if factor.id == values[0])
    return _count(path, len(factor.levels))


# --- the examples a snapshot froze, and the failed trials they were kept from -------------------------------------


def _failures_held(report_input: ReportInput, aspects: Sequence[str], path: str) -> list[FailedTrials]:
    """The input's failed-trial counts on ``aspects``, refused as unstated where it counts none on one of them: an input
    built without them (a planner's, a bare one), or an aspect whose bundle left failures out, reads as not known,
    never as none."""
    held = report_input.failed_trials
    counted = {row.aspect for row in held} if held is not None else set()
    missing = [aspect for aspect in aspects if aspect not in counted]
    if held is None or missing:
        where = "" if held is None else f" on {', '.join(missing)}"
        raise _unstated(
            path,
            f"counts the failed trials the run holds{where}, which the snapshot does not count: not known, so a share "
            "of them is not either",
        )
    return [row for row in held if row.aspect in aspects]


def _example_aspects(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    """The aspects an example or a failed-trial count names, in the input's aspect order, then any other."""
    named = [example.aspect for example in report_input.examples if example.aspect is not None]
    named += [row.aspect for row in report_input.failed_trials or ()]
    order = [aspect.id for aspect in report_input.aspects]
    return [aspect for aspect in order if aspect in named] + [a for a in dict.fromkeys(named) if a not in order]


def _example_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    """The arms an example or a failed-trial count names on one aspect, in the input's arm order."""
    aspect = prior[-1]
    named = {example.arm for example in report_input.examples if example.aspect == aspect}
    named |= {row.arm for row in report_input.failed_trials or () if row.aspect == aspect}
    return [arm.id for arm in report_input.arms if arm.id in named]


def _example_strata(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    """The strata or tasks an example or a failed-trial count names for one aspect and arm."""
    aspect, arm = prior[-2:]
    named = [e.stratum for e in report_input.examples if (e.aspect, e.arm) == (aspect, arm)]
    named += [row.stratum for row in report_input.failed_trials or () if (row.aspect, row.arm) == (aspect, arm)]
    return [stratum for stratum in dict.fromkeys(named) if stratum is not None]


def _kept_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect, arm, *stratum = values
    kept = [
        example
        for example in report_input.examples
        if (example.aspect, example.arm) == (aspect, arm) and (not stratum or example.stratum == stratum[0])
    ]
    return _count(path, len(kept))


def _failed_trials_count(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect, arm, *stratum = values
    held = _failures_held(report_input, [aspect], path)
    rows = [row for row in held if row.arm == arm and (not stratum or row.stratum == stratum[0])]
    return _count(path, sum(row.failed for row in rows))


# --- whether two aspects ran the same cases -------------------------------------------------------------------------

#: What two aspects' cases are, as ``overlap.<aspect>.<aspect>.same`` reads it: every case in both, some, none, or not
#: known.
OVERLAP_WORDS: dict[str, str] = {
    "same": "the same cases",
    "partly": "partly the same cases",
    "apart": "different cases",
    "unknown": "not known: the snapshot does not hold the two aspects' case ids",
}

#: What an overlap key reads: whether the two aspects ran the same cases, in words, and the counts behind it.
_OVERLAP_FIELDS = ("same", "shared", "only_first", "only_second")


def _other_aspects(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return [aspect.id for aspect in report_input.aspects if aspect.id != prior[-1]]


def _overlap(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    first, second, field = values
    held = report_input.case_overlap or ()
    found = next((row for row in held if row.aspects in ((first, second), (second, first))), None)
    if found is None:
        if field == "same":
            return Figure(path=path, kind="text", value=OVERLAP_WORDS["unknown"])
        raise _unstated(
            path, f"counts the cases {first} and {second} share, which the snapshot does not hold: not known"
        )
    flipped = found.aspects != (first, second)
    only_first, only_second = (
        (found.only_second, found.only_first) if flipped else (found.only_first, found.only_second)
    )
    if field == "same":
        word = "same" if not only_first and not only_second else "apart" if not found.shared else "partly"
        return Figure(path=path, kind="text", value=OVERLAP_WORDS[word])
    counts = {"shared": found.shared, "only_first": only_first, "only_second": only_second}
    return _count(path, counts[field])


# --- a classifier's confusion: where each expected label's predictions went ------------------------------------------


def _confusion_axes(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return list(report_input.confusion)


def _confusion_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(report_input.confusion[prior[-1]].per_arm)


def _confusion_labels(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    axis = next(value for value in prior if value in report_input.confusion)
    return list(report_input.confusion[axis].labels)


def _confusion(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    axis, arm, expected, predicted = values
    table = report_input.confusion[axis]
    matrix = table.per_arm[arm]
    row, column = table.labels.index(expected), table.labels.index(predicted)
    if row >= len(matrix) or column >= len(matrix[row]):
        raise _unstated(path, f"names a cell of {arm}'s {axis} confusion its matrix does not hold")
    scale: Scale = "percent" if table.unit == "row_percent" else "count"
    return Figure(path=path, kind="number", value=float(matrix[row][column]), scale=scale)


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
    board = report_input.scoreboard if aspect_id is None else _aspect_board(report_input, aspect_id)
    row = board.get(arm_id)
    return list((row.unanswered or {}).values()) if row is not None else []


def _unanswered_arms(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return [arm for arm, row in report_input.scoreboard.items() if row.unanswered]


def _aspect_unanswered_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return [arm for arm, row in _aspect_board(report_input, prior[0]).items() if row.unanswered]


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
        ordered = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
        return Figure(
            path=path, kind="text", value=", ".join(f"{word} {group_thousands(str(count))}" for word, count in ordered)
        )
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
            path=path, kind="number", value=gap * _monthly_factor(report_input, path), scale="usd", change=True
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
        return _level(path, pair.confidence, "the interval's level")
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
        undecided = undecided_arms(report_input)
        why = (
            f"; undecided: {_listed(undecided)}. A gate has no figure for them somewhere and none they have fails, so "
            "no figure of theirs could change the call alone (`count.unknown.<criterion>.<arm>` counts the places)"
            if undecided
            else ""
        )
        raise _refuse(
            path,
            "names the weakest margin, and the input reads no figure against a criterion, margin or floor that could "
            f"change the call alone{why}",
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

#: What an arm's spread on a figure reads: its mean, its standard error, the spread of one case, the count it rests on
#: and what that count counts (cases, trials or observations), the distinct cases behind a count of trials and the
#: trials each ran today, the level a result read from it is stated at, what the standard error was read from, and the
#: interval it gives at that level and its two ends (the mean plus and minus the normal quantile times the standard
#: error, a rate's ends held to none to all: :func:`~threetears.evals.analysis.separation.spread_interval`).
_SPREAD_FIELDS = (
    "mean",
    "se",
    "per_case",
    "n",
    "counts",
    "cases",
    "repeats",
    "level",
    "basis",
    "interval",
    "low",
    "high",
)

#: What two arms' difference reads: the first minus the second, its interval, the interval's ends and half-width, its
#: level, and whether it is the paired estimate a contrast states or two independent arms' spreads combined.
_COMPARE_FIELDS = ("difference", "interval", "low", "high", "half_width", "level", "basis")

#: What settling two arms apart reads: their difference, the cases each needs for an interval centred on it to clear
#: zero (``cases_needed``: half such designs clear; ``cases_needed_80``: four in five) and the trials those take at
#: today's repeats, how often a fresh run of today's size reads the other arm ahead, and the level, its sidedness
#: (two-sided) in its words.
_SETTLE_PAIR_FIELDS = (
    "difference",
    "cases_needed",
    "cases_needed_80",
    "trials_needed",
    "trials_needed_80",
    "wrong_side",
    "level",
)

#: What settling an arm against a line reads: its distance from the line toward passing (negative where today's figure
#: misses), the cases a one-sided bound needs to sit wholly on today's side of the line, the cases the two-sided
#: interval a page prints needs for the same, the trials each size takes at today's repeats, how often a fresh run of
#: today's size lands on the other side, and the level, its sidedness (one-sided) in its words.
_SETTLE_LINE_FIELDS = (
    "distance",
    "cases_needed",
    "cases_needed_80",
    "cases_needed_two_sided",
    "cases_needed_two_sided_80",
    "trials_needed",
    "trials_needed_80",
    "trials_needed_two_sided",
    "trials_needed_two_sided_80",
    "wrong_side",
    "level",
)

#: A hypothetical line's fields: the line itself, then what settling an arm against it reads.
_HYPOTHETICAL_FIELDS = ("line", *_SETTLE_LINE_FIELDS)

#: A margin's or floor's fields: its line (minus the margin, in points), the candidate's change from the control it reads
#: (in points), then what settling that change against the line reads.
_MARGIN_FIELDS = ("line", "change", *_SETTLE_LINE_FIELDS)

#: How each basis of a spread reads, in words.
_BASIS_WORDS: dict[str, str] = {
    "interval": "its stated interval",
    "half_width": "its stated half-width",
    "sem": "its standard error",
    "binomial": "a pass/fail rate's binomial spread, from its count",
    "cells": (
        "the standard error derived from its target-matrix cells, the figure being their plain mean: the root of "
        "the cells' squared standard errors over their count, the cells read as independent, at the cells' stated "
        "level; a size read from it counts cases at today's mix of cells"
    ),
}

#: A cell-derived spread's basis where a case-weighted mean of the cells fits the figure as well as the plain one.
_CELLS_BOTH_WORDS = (
    "the standard error derived from its target-matrix cells, the figure fitting both their plain and their "
    "case-weighted mean: the wider of the two readings', the cells read as independent, at the cells' stated level; "
    "a size read from it counts cases at today's mix of cells"
)


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


#: What a count a spread rests on counts: distinct cases, trials (a case run again counts again), or observations (a
#: statistic's own ``n``, where nothing says whether it counts cases or trials).
_Counted = Literal["cases", "trials", "observations"]


def _separation_place(report_input: ReportInput, where: str) -> _Place | None:
    """The one slice a separation key's ``<where>`` names (:func:`_separation_places`), found by its kind and id alone,
    since every key reads it once per arm and place; None where it names none, or a cell or stratum held twice."""
    kind, _dot, named = where.partition(".")
    found: list[_Place] = []
    if kind == "task":
        found = [_Place(where, task.per_arm, task.n_cases) for task in report_input.tasks if task.id == named]
    elif kind == "cells":
        cells = report_input.cells.values()
        found = [_Place(where, cell.per_arm, cell.n_cases) for cell in cells if _cell_spelling(cell) == named]
    elif kind == "strata":
        strata = report_input.strata
        found = [_Place(where, stratum.per_arm, stratum.n_cases) for stratum in strata if stratum.id == named]
    return found[0] if len(found) == 1 else None


@dataclass(frozen=True)
class _Held:
    """An arm's statistic at one place, and what its spread's count is.

    :ivar stat: The statistic.
    :ivar count: The count its spread rests on; None where none is stated.
    :ivar counted: What that count counts.
    :ivar cases: The distinct cases behind a count of trials, where stated and fewer; None otherwise.
    :ivar source: Where the count was read, in words.
    """

    stat: Stat
    count: float | None
    counted: _Counted | None
    cases: float | None
    source: str


#: Where a spread's count was read, in words: the figure's own, the slice's stated cases, or the arm's scoreboard.
_COUNT_SOURCES: dict[str, str] = {
    "own": "its own count",
    "task": "the task's stated cases",
    "cells": "the cell's stated cases",
    "strata": "the stratum's stated cases",
    "trials": "the arm's usable trials",
    "cases": "the arm's cases",
}


def _held_where(
    report_input: ReportInput, arm_id: str, key: str, where: str, aspect: str | None = None
) -> _Held | None:
    """An arm's statistic on a measure or judged dimension at ``where``, with the count it rests on, what that count
    counts and where it was read: its own ``n`` (observations; cases where its ``n_cases`` equals it, trials where its
    ``n_cases`` is fewer), else its ``n_cases`` (cases), else the place's: the slice's cases, or overall the arm's
    usable trials, else its cases. A count of trials carries the distinct cases behind it where the figure, or overall
    the arm's scoreboard, states them. Overall, ``aspect`` names an aspect other than the lead whose figures
    (``by_aspect``) and scoreboard are read instead (a criterion declared on it); a slice holds no other aspect's
    figures. None where the input states no mean there."""
    fallback: tuple[int | None, _Counted | None, int | None, str] = (None, None, None, "")
    if aspect is not None and where != _OVERALL:
        return None
    if where == _OVERALL:
        table = report_input.measures.get(key) or report_input.judged.get(key)
        per_arm = None if table is None else table.per_arm if aspect is None else table.by_aspect.get(aspect, {})
        stat = per_arm.get(arm_id) if per_arm is not None else None
        held = {candidate.id for candidate in report_input.aspects}
        boards = (
            report_input.scoreboard if aspect is None else _aspect_board(report_input, aspect) if aspect in held else {}
        )
        board = boards.get(arm_id)
        if board is not None and board.usable is not None:
            fallback = (board.usable, "trials", board.cases, _COUNT_SOURCES["trials"])
        elif board is not None:
            fallback = (board.cases, "cases", None, _COUNT_SOURCES["cases"])
    else:
        place = _separation_place(report_input, where)
        stat = place.per_arm.get(arm_id, {}).get(key) if place is not None else None
        if place is not None:
            fallback = (place.n_cases, "cases", None, _COUNT_SOURCES[where.partition(".")[0]])
    if stat is None or stat.mean is None:
        return None
    count: int | None
    counted: _Counted | None
    cases: int | None = None
    source = _COUNT_SOURCES["own"]
    if stat.n is not None:
        count = stat.n
        counted = "observations" if stat.n_cases is None else "cases" if stat.n <= stat.n_cases else "trials"
        cases = stat.n_cases if counted == "trials" else None
    elif stat.n_cases is not None:
        count, counted = stat.n_cases, "cases"
    else:
        count, counted, cases, source = fallback
    if count is None:
        return _Held(stat, None, None, None, "")
    distinct = float(cases) if cases is not None and 0 < cases < count else None
    return _Held(stat, float(count), counted, distinct, source)


def _stat_where(
    report_input: ReportInput, arm_id: str, key: str, where: str, aspect: str | None = None
) -> tuple[Stat, float | None, _Counted | None] | None:
    """An arm's statistic at ``where`` (on another aspect, where ``aspect`` names one), the count it rests on and what
    that count counts (:func:`_held_where`)."""
    held = _held_where(report_input, arm_id, key, where, aspect)
    return None if held is None else (held.stat, held.count, held.counted)


def _wheres(report_input: ReportInput, arms: Sequence[str], key: str) -> list[str]:
    """Where every one of ``arms`` holds a mean on ``key``: overall first, then each slice in the input's order."""
    candidates = [_OVERALL, *(place.where for place in _separation_places(report_input))]
    return [where for where in candidates if all(_stat_where(report_input, arm, key, where) for arm in arms)]


def _separated_keys(report_input: ReportInput, arms: Sequence[str]) -> list[str]:
    """The measures and judged dimensions every one of ``arms`` holds a mean on somewhere."""
    keys = dict.fromkeys([*report_input.measures, *report_input.judged])
    return [key for key in keys if _wheres(report_input, arms, key)]


#: The measures graded by class label whose figure is a share of counted trials: the ones the report input's build grades
#: ``class_label`` (``report_input_build``'s grade rule). A figure graded so by another name may be a derived mean of
#: shares (a balanced accuracy: the mean of each class's recall), which no binomial spread describes.
_TRIAL_SHARE_LABELS = frozenset({MATCH_MEASURE, ACCURACY_MEASURE})


def _rate_of(report_input: ReportInput, key: str) -> Literal["share", "percent"] | None:
    """The scale a figure that is a share of counted trials is on, which a binomial spread reads: a measure graded
    pass/fail (each observation a pass or a fail, the figure the share that passed), one graded by class label that the
    build's grade rule names a share of trials (:data:`_TRIAL_SHARE_LABELS`), a pass/fail judged dimension. None for any
    other figure: a derived mean of shares graded by class label under another name reads no binomial spread."""
    measure = report_input.measures.get(key)
    judged = report_input.judged.get(key)
    trial_share = measure is not None and (
        measure.grade == "pass_fail" or (measure.grade == "class_label" and key in _TRIAL_SHARE_LABELS)
    )
    graded = trial_share or (judged is not None and judged.scale == "pass_fail")
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


def _read_spread(report_input: ReportInput, arm_id: str, key: str, where: str, aspect: str | None = None) -> Spread:
    """An arm's spread on a figure at ``where`` (:func:`~threetears.evals.analysis.separation.stat_spread`), on another
    aspect's figure where ``aspect`` names one (:func:`_stat_where`).

    :raises NoSpread: The input gives none; its words are what a refusal says after the key.
    """
    held = _held_where(report_input, arm_id, key, where, aspect)
    if held is None:
        raise NoSpread(f"names {arm_id}'s mean on {key!r} {_where_words(where)}, which the input does not state")
    stat = held.stat
    if where == _OVERALL and aspect is None and stat.ci is None and stat.half_width is None and stat.sem is None:
        cells = _cells_spread(report_input, arm_id, key, held)
        if cells is not None:
            return cells
    try:
        return stat_spread(
            float(stat.mean) if stat.mean is not None else None,
            ci=None if stat.ci is None else (float(stat.ci[0]), float(stat.ci[1])),
            half_width=None if stat.half_width is None else float(stat.half_width),
            sem=None if stat.sem is None else float(stat.sem),
            n=held.count,
            stated_level=_stated_level(report_input, key),
            rate=_rate_of(report_input, key),
            cases=held.cases,
        )
    except NoSpread as none:
        raise NoSpread(
            f"reads {arm_id}'s spread on {key!r} {_where_words(where)}, and the figure has none: {none}"
        ) from none


def _weighted_too(report_input: ReportInput, key: str, arm_id: str) -> bool:
    """Whether a cell-derived spread's figure fits its cells' case-weighted mean as well as their plain one."""
    try:
        return cell_mean_interval(report_input, key, arm_id).weighted_too
    except NoCellMean:
        # NOSILENT: read only for a spread whose basis is the cells', which qualified; an overall that does not is
        # no weighted reading, which False says.
        return False


def _cells_spread(report_input: ReportInput, arm_id: str, key: str, held: _Held) -> Spread | None:
    """An arm's overall spread read off its target-matrix cells, where the overall states none of its own and is the
    plain mean of its cells (:func:`~threetears.evals.analysis.cell_mean.cell_mean_interval`, the interval a gate on it
    reads): its standard error the cells', combined; one case's spread that times the root of the overall's count. None
    where the overall does not qualify, so its spread is read as before."""
    try:
        derived = cell_mean_interval(report_input, key, arm_id)
    except NoCellMean:
        # NOSILENT: an overall that is no plain mean of its cells is the input's shape, not a fault: its spread is
        # read from what it states, as before, which is what None tells the caller.
        return None
    count, distinct = held.count, held.cases
    per_case = derived.se * math.sqrt(distinct or count) if count is not None else None
    return Spread(derived.mean, derived.se, count, per_case, derived.level, "cells", distinct)


def _spread(
    report_input: ReportInput, arm_id: str, key: str, where: str, path: str, aspect: str | None = None
) -> Spread:
    """An arm's spread on a figure at ``where`` (on another aspect's, where ``aspect`` names one), refused as unstated,
    with why, where the input gives none."""
    try:
        return _read_spread(report_input, arm_id, key, where, aspect)
    except NoSpread as none:
        raise _unstated(path, str(none)) from none


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
    if field == "counts":
        held = _stat_where(report_input, arm_id, key, where)
        counted = None if held is None else held[2]
        if counted is None:
            raise _unstated(path, "names what the figure's count counts, and it rests on no count")
        return Figure(path=path, kind="text", value=counted)
    if field == "cases":
        return _number(path, spread.cases, "the distinct cases behind a count of trials", "count")
    if field == "repeats":
        return _number(path, spread.repeats, "the trials each case ran, where the count is trials", "plain")
    if field == "level":
        return _level(path, spread.level, "the level a result read from it is stated at")
    if field in ("interval", "low", "high"):
        try:
            low, high = spread_interval(spread, rate=_rate_of(report_input, key))
        except NoSpread as none:
            raise _unstated(path, str(none)) from none
        scale, unit = _scale_of(report_input, key)
        if field == "interval":
            return _interval(path, (low, high), "the interval", scale, unit=unit)
        return _number(path, low if field == "low" else high, f"the interval's {field} end", scale, unit=unit)
    return Figure(path=path, kind="text", value=_basis_words(report_input, arm_id, key, where, spread))


def _basis_words(
    report_input: ReportInput, arm_id: str, key: str, where: str, spread: Spread, aspect: str | None = None
) -> str:
    """What a spread was read from, in words; a binomial one says where its count was read (its own, the slice's
    stated cases, or the arm's scoreboard), on another aspect's figures where ``aspect`` names one."""
    if spread.basis == "cells" and _weighted_too(report_input, key, arm_id):
        return _CELLS_BOTH_WORDS
    if spread.basis != "binomial":
        return _BASIS_WORDS[spread.basis]
    held = _held_where(report_input, arm_id, key, where, aspect)
    source = held.source if held is not None and held.source else _COUNT_SOURCES["own"]
    words = f"a pass/fail rate's binomial spread, from {source}"
    if spread.cases is None:
        return words
    return f"{words}, read on its distinct cases, each counted once: {_REPEATS_BOUND}"


#: What a binomial spread on a repeated design is, as its basis says it: the wide end of what its repeats allow.
_REPEATS_BOUND = (
    "the widest spread its repeats allow (repeats read as independent would read it narrower by the root of the "
    "repeats), so an interval or a size read from it is never narrower than the cases support"
)


def _family_level(report_input: ReportInput, contrast: Contrast) -> float | None:
    """A contrast's interval's level: its own, else the one family correcting its aspect's."""
    own = level_share(contrast.level)
    if own is not None:
        return own
    families = [family for family in report_input.contrast_families if family.aspect == contrast.aspect]
    return level_share(families[0].level) if len(families) == 1 else None


def correction_words(report_input: ReportInput, contrast: Contrast) -> str:
    """Whether a contrast's interval is corrected, as the one family on its aspect states it (``correction``), never
    asserted: corrected by the method it names, not corrected where it names ``none``, and unstated where it names none
    or no one family is on the contrast's aspect. A ``compare.*.basis`` key and a compare block's words say it alike.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param contrast: The contrast.
    :ptype contrast: Contrast
    :return: The words, after the contrast's "its own interval, ".
    :rtype: str
    """
    families = [family for family in report_input.contrast_families if family.aspect == contrast.aspect]
    if len(families) != 1:
        return "and no one contrast family on its aspect says whether it is corrected"
    correction = (families[0].correction or "").strip()
    if not correction:
        return "and its family states no correction, so whether it is corrected is unstated"
    if correction.lower() == "none":
        return "not corrected across its family (the family's correction is none)"
    return f"corrected within its family ({correction})"


def _paired(report_input: ReportInput, arm_id: str, other: str, key: str) -> Contrast | None:
    """The one contrast stating one of the two arms against the other (the control) on ``key``, with its difference
    and interval, on the lead aspect whose figures ``overall`` reads (:func:`on_lead_aspect`): the paired estimate, which the
    compare family reads overall in place of the arms' own spreads."""
    control = _control(report_input)
    if control is None or control.id not in (arm_id, other):
        return None
    candidate = other if arm_id == control.id else arm_id
    held = [
        contrast
        for contrast in report_input.contrasts
        if contrast.arm == candidate
        and contrast.measure == key
        and on_lead_aspect(report_input, contrast)
        and contrast.control in (None, control.id)
        and contrast.delta is not None
        and contrast.interval is not None
    ]
    return held[0] if len(held) == 1 else None


def _both_bases(
    report_input: ReportInput, arm_id: str, other: str, key: str, where: str, first: Spread, second: Spread
) -> str:
    """What two arms' spreads were read from, in words: once where both read alike, else each, the first's first."""
    one = _basis_words(report_input, arm_id, key, where, first)
    two = _basis_words(report_input, other, key, where, second)
    return one if one == two else f"{one}; {two}"


def pairing_contrast(report_input: ReportInput, arm_id: str, other: str, key: str, where: str) -> Contrast | None:
    """The contrast a ``compare.<arm>.<other>.<key>.<where>`` key reads as its paired estimate: overall, the one
    contrast stating one of the two arms against the other (the control) on the lead aspect, with its difference and
    interval; None where the compare family reads the two arms' own spreads as independent.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm_id: The first arm.
    :ptype arm_id: str
    :param other: The second.
    :ptype other: str
    :param key: The measure or judged dimension.
    :ptype key: str
    :param where: The place, as a separation key spells it.
    :ptype where: str
    :return: The contrast; None for none.
    :rtype: Contrast | None
    """
    return _paired(report_input, arm_id, other, key) if where == _OVERALL else None


@dataclass(frozen=True)
class _Difference:
    """One arm minus another on a figure at one place, with its interval: the compare family's reading, which the lever
    family reads too, so the two never disagree on whether an interval exists or what it is.

    :ivar difference: The first minus the second, in the figure's own units: a pairing contrast's difference where one
        pairs them, else the two means'.
    :ivar low: The interval's low end, in the same units.
    :ivar high: Its high end.
    :ivar level: Its level, as a share; None where a contrast's interval states none.
    :ivar basis: Paired or independent, and from what, in words.
    """

    difference: float
    low: float
    high: float
    level: float | None
    basis: str


def _mean_difference(report_input: ReportInput, arm_id: str, other: str, key: str, where: str) -> float | None:
    """The first arm's mean minus the second's at ``where``; None where either states none."""
    means = [_stat_where(report_input, arm, key, where) for arm in (arm_id, other)]
    if means[0] is None or means[1] is None or means[0][0].mean is None or means[1][0].mean is None:
        return None
    return float(means[0][0].mean) - float(means[1][0].mean)


def _read_difference(report_input: ReportInput, arm_id: str, other: str, key: str, where: str) -> _Difference:
    """The first arm minus the second on a figure at ``where``, with its interval: overall, the contrast pairing them
    where one does (:func:`_paired`), else the two arms' own spreads combined as independent
    (:func:`~threetears.evals.analysis.separation.independent_difference`).

    :raises NoSpread: Neither a pairing contrast nor both arms' spreads give an interval; its words say why.
    """
    contrast = pairing_contrast(report_input, arm_id, other, key, where)
    if contrast is not None and contrast.delta is not None and contrast.interval is not None:
        sign = 1.0 if contrast.arm == arm_id else -1.0
        low, high = sorted((sign * float(contrast.interval[0]), sign * float(contrast.interval[1])))
        return _Difference(
            difference=sign * float(contrast.delta),
            low=low,
            high=high,
            level=_family_level(report_input, contrast),
            basis=(
                f"paired: the contrast of {contrast.arm} against the control on {key!r}, its own interval, "
                f"{correction_words(report_input, contrast)}"
            ),
        )
    first = _read_spread(report_input, arm_id, key, where)
    second = _read_spread(report_input, other, key, where)
    difference, half = independent_difference(first, second, first.level)
    return _Difference(
        difference=difference,
        low=difference - half,
        high=difference + half,
        level=first.level,
        basis=(
            f"independent: each arm's own spread ({_both_bases(report_input, arm_id, other, key, where, first, second)}), "
            "combined as if the arms ran different cases, which reads wider than a paired interval where they ran the "
            "same"
        ),
    )


def _compare_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, other, key, where, field = values
    if arm_id == other:
        raise _refuse(path, "compares an arm with itself; name two arms")
    contrast = pairing_contrast(report_input, arm_id, other, key, where)
    if field == "difference" and contrast is None:
        difference = _mean_difference(report_input, arm_id, other, key, where)
        if difference is None:
            raise _unstated(path, f"names the two arms' means on {key!r} {_where_words(where)}, which one lacks")
        return _apart(report_input, path, key, difference, "the two arms' difference", change=True)
    try:
        read = _read_difference(report_input, arm_id, other, key, where)
    except NoSpread as none:
        raise _unstated(path, str(none)) from none
    difference, low, high, level, basis = read.difference, read.low, read.high, read.level, read.basis
    if field == "basis":
        return Figure(path=path, kind="text", value=basis)
    if field == "level":
        if level is None:
            raise _unstated(path, "names the level of the contrast's interval, which the input does not state")
        return _level(path, level, "the interval's level")
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
        return _sided_level(path, first.level, 2)
    if field == "wrong_side":
        share = _settled(path, lambda: wrong_side_share(difference, math.hypot(first.se, second.se)))
        return _number(path, share, "how often a fresh run of today's size reads the other arm ahead", "share")
    if first.per_case is None or second.per_case is None:
        raise _unstated(path, "names the cases a design needs, which needs the count each arm's figure rests on")
    per_case = math.hypot(first.per_case, second.per_case)
    power = POWER if field.endswith("_80") else 0.5
    cases = _settled(path, lambda: cases_to_separate(per_case, difference, first.level, sided=2, power=power))
    if field.startswith("trials_"):
        if first.repeats != second.repeats:
            raise _unstated(
                path,
                "names the trials each arm needs, and the two arms ran their cases a different number of times: "
                "read each arm's repeats (`spread.<arm>.<measure>.<where>.repeats`)",
            )
        return _trials(path, cases, first)
    return _number(path, cases, "the cases each arm needs", "count")


def _trials(path: str, cases: float, spread: Spread) -> Figure:
    """The trials a size in cases takes, each case run as often as today's design runs one."""
    if spread.repeats is None:
        raise _unstated(
            path,
            "names the trials a size takes, and the figure states no count of trials over fewer distinct cases: its "
            "size counts what its count counts (`spread.<arm>.<measure>.<where>.counts`)",
        )
    return _number(path, math.ceil(cases * spread.repeats - 1e-9), "the trials the cases take", "count")


def _sided_level(path: str, level: float, sided: Literal[1, 2]) -> Figure:
    """A settle figure's level as words that carry its sidedness (``95% one-sided``), so no citation of it reads as the
    two-sided intervals a page prints when it is not one."""
    percent = format_number(level * 100)
    return Figure(path=path, kind="text", value=f"{percent}% {'one-sided' if sided == 1 else 'two-sided'}")


def _settle_line(
    report_input: ReportInput,
    path: str,
    spread: Spread,
    key: str,
    line: float,
    higher_is_better: bool,
    field: str,
    *,
    in_points: bool = False,
) -> Figure:
    """What settling an arm's figure against a line reads, at ``field``: its distance toward passing, and the sizes at
    which an interval centred on it sits wholly on today's side of the line (clearing it where today's figure passes,
    confirming the miss where it fails), one-sided and, beside it, two-sided as a page prints its intervals; each size
    in cases, and the trials it takes at today's repeats. ``in_points``: the figure is a change already in points (a
    reading against a margin or a floor), not one on ``key``'s scale."""
    distance = spread.mean - line if higher_is_better else line - spread.mean
    if field == "distance" and in_points:
        return _number(path, distance, "the figure's distance from the line", "points", change=True)
    if field == "distance":
        return _apart(report_input, path, key, distance, "the figure's distance from the line", change=True)
    if field == "level":
        return _sided_level(path, spread.level, 1)
    if field == "wrong_side":
        share = _settled(path, lambda: wrong_side_share(distance, spread.se))
        return _number(path, share, "how often a fresh run of today's size lands on the other side", "share")
    per_case = spread.per_case
    if per_case is None:
        raise _unstated(path, "names the cases a design needs, which needs the count the figure rests on")
    power = POWER if field.endswith("_80") else 0.5
    sided: Literal[1, 2] = 2 if "two_sided" in field else 1
    cases = _settled(path, lambda: cases_to_separate(per_case, distance, spread.level, sided=sided, power=power))
    if field.startswith("trials_"):
        return _trials(path, cases, spread)
    return _number(path, cases, "the cases a design needs", "count")


#: What a criterion's settle reads beside its sizes: the count the arm's spread rests on, what that count counts, and
#: what the spread was read from, each on the criterion's own aspect (the ``spread.`` key's, on the lead aspect).
_SETTLE_COUNT_FIELDS = ("n", "counts", "cases", "repeats", "basis")


def _criterion_count(
    report_input: ReportInput, path: str, spread: Spread, at: tuple[str, str, str, str | None], field: str
) -> Figure:
    """A criterion settle's ``n``, ``counts``, ``repeats`` or ``basis``: what ``spread.<arm>.<measure>.<where>``
    reads, on the criterion's aspect."""
    if field == "n":
        return _number(path, spread.n, "the count the figure's spread rests on", "count")
    if field == "repeats":
        return _number(path, spread.repeats, "the trials each case ran, where the count is trials", "plain")
    if field == "cases":
        return _number(path, spread.cases, "the distinct cases behind a count of trials", "count")
    if field == "counts":
        arm_id, key, where, aspect = at
        held = _stat_where(report_input, arm_id, key, where, aspect)
        counted = None if held is None else held[2]
        if counted is None:
            raise _unstated(path, "names what the figure's count counts, and it rests on no count")
        return Figure(path=path, kind="text", value=counted)
    arm_id, key, where, aspect = at
    return Figure(path=path, kind="text", value=_basis_words(report_input, arm_id, key, where, spread, aspect))


def _criterion_wheres(report_input: ReportInput, criterion: Criterion, arm_id: str) -> list[str]:
    """Where a criterion reads an arm, as ``<where>`` spells it."""
    found: list[str] = []
    for figure in criterion_figures(report_input, criterion, arm_id):
        if figure.cell is not None:
            found.append(f"cells.{_cell_spelling(report_input.cells[figure.cell])}")
        else:
            found.append(_OVERALL if figure.task is None else f"task.{figure.task}")
    return found


def _criterion_other_aspect(report_input: ReportInput, criterion: Criterion) -> str | None:
    """The aspect a criterion reads where it is not the lead one (whose figures sit in ``by_aspect``); None for the
    lead's."""
    return None if criterion.aspect is None or _is_lead(report_input, criterion.aspect) else criterion.aspect


def _settle_criterion_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion_id, arm_id, where, field = values
    criterion = _criterion(report_input, criterion_id)
    if criterion.statistic != "mean":
        raise _unstated(
            path, f"reads a {criterion.statistic}, which states no spread to size a design by; settle reads a mean"
        )
    aspect = _criterion_other_aspect(report_input, criterion)
    spread = _spread(report_input, arm_id, criterion.measure, where, path, aspect)
    if field in _SETTLE_COUNT_FIELDS:
        return _criterion_count(report_input, path, spread, (arm_id, criterion.measure, where, aspect), field)
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


# --- settling a candidate's reading against its aspect's margin, or a floor read on it --------------------------------


def _margined_aspects(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    """The aspects some reading is read against a margin on: the reading's own margin, else the aspect's."""
    return [
        aspect.id
        for aspect in report_input.aspects
        if any(_reading_margin(aspect, reading) is not None for reading in aspect.readings)
    ]


def _reading_margin(aspect: Aspect, reading: Reading) -> float | None:
    margin = reading.margin if reading.margin is not None else aspect.margin
    return None if margin is None else float(margin)


def _margined_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect = _aspect_by_id(report_input, prior[-1])
    return [reading.arm for reading in aspect.readings if _reading_margin(aspect, reading) is not None]


def _reading_spread(aspect: Aspect, arm_id: str, path: str) -> Spread:
    """A candidate's reading on an aspect as a spread on its change, in points: its gain interval at its stated level
    (:func:`~threetears.evals.analysis.separation.difference_spread`), with the cases each arm ran."""
    reading = _reading(aspect, arm_id)
    if reading.gain_interval is None:
        raise _unstated(
            path, f"reads {arm_id}'s reading on {aspect.id} by its interval, which the reading does not state"
        )
    try:
        return difference_spread(
            float(reading.change),
            (float(reading.gain_interval[0]), float(reading.gain_interval[1])),
            level_share(reading.gain_confidence),
            tuple(float(count) for count in reading.cases),
        )
    except NoSpread as none:
        raise _unstated(path, f"reads {arm_id}'s reading on {aspect.id}, which has no spread: {none}") from none


def _settle_margin_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, arm_id, field = values
    aspect = _aspect_by_id(report_input, aspect_id)
    margin = _reading_margin(aspect, _reading(aspect, arm_id))
    if margin is None:
        raise _unstated(path, f"reads {arm_id}'s reading on {aspect_id} against its margin, and none is declared")
    if field == "line":
        return _number(path, -margin, "the margin's line: the drop the reading may show", "points", change=True)
    spread = _reading_spread(aspect, arm_id, path)
    if field == "change":
        return _number(path, spread.mean, "the reading's change from the control", "points", change=True)
    return _settle_line(report_input, path, spread, "", -margin, True, field, in_points=True)


def _floored_aspects(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return list(dict.fromkeys(read.aspect for read in report_input.floors if read.margin is not None))


def _floored_keys(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    reads = report_input.floors
    return list(dict.fromkeys(read.floor for read in reads if read.aspect == prior[-1] and read.margin is not None))


def _floored_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    """The arms a floor is read on with a margin stated, which a ``settle.floor.`` key settles against; an arm whose
    read states no margin has no line to settle against."""
    aspect_id, floor = prior[-2:]
    return [
        read.arm
        for read in report_input.floors
        if (read.aspect, read.floor) == (aspect_id, floor) and read.margin is not None
    ]


def _floor_measure(report_input: ReportInput, rate: str, path: str) -> str:
    """The one measure a floor's rate names: the measure of that key, else the check measure minted for it."""
    named = [key for key in report_input.measures if key == rate or goal_check_of(key) == rate]
    if len(named) != 1:
        found = "no measure" if not named else f"{len(named)} measures"
        raise _unstated(path, f"reads the floor's rate {rate!r}, which names {found} the input holds")
    return named[0]


def _aspect_spread(report_input: ReportInput, key: str, aspect_id: str, arm_id: str, path: str) -> Spread:
    """An arm's spread on a measure on one aspect, from its own statistic there: its interval, half-width, standard
    error, or a pass/fail rate's binomial spread from its own count."""
    table = report_input.measures[key]
    stat = _on_aspect(table, aspect_id, report_input).get(arm_id)
    if stat is None:
        raise _unstated(path, f"reads {arm_id}'s figure on {key!r} on {aspect_id}, which the input does not state")
    count = stat.n if stat.n is not None else stat.n_cases
    distinct = stat.n_cases if stat.n is not None and stat.n_cases is not None and stat.n_cases < stat.n else None
    try:
        return stat_spread(
            None if stat.mean is None else float(stat.mean),
            ci=None if stat.ci is None else (float(stat.ci[0]), float(stat.ci[1])),
            half_width=None if stat.half_width is None else float(stat.half_width),
            sem=None if stat.sem is None else float(stat.sem),
            n=None if count is None else float(count),
            stated_level=_stated_level(report_input, key),
            rate=_rate_of(report_input, key),
            cases=None if distinct is None else float(distinct),
        )
    except NoSpread as none:
        raise _unstated(path, f"reads {arm_id}'s spread on {key!r} on {aspect_id}: {none}") from none


def _floor_spread(report_input: ReportInput, aspect_id: str, floor: str, arm_id: str, path: str) -> Spread:
    """A floor's drop on a candidate as a spread, in points: the aspect's reading where the floor speaks for its pass
    rate, else the candidate's rate on the floor's own measure minus the control's, their spreads combined as
    independent."""
    read = next(
        held for held in report_input.floors if (held.aspect, held.floor, held.arm) == (aspect_id, floor, arm_id)
    )
    if read.reading is None:
        aspect = next((held for held in report_input.aspects if held.id == aspect_id), None)
        if aspect is None or arm_id not in _reading_arms(aspect):
            raise _unstated(path, f"reads the pass rate's drop on {aspect_id}, which holds no reading of {arm_id}")
        return _reading_spread(aspect, arm_id, path)
    control = _control(report_input)
    if control is None:
        raise _unstated(path, "reads a drop from the control, and the input names no control")
    key = _floor_measure(report_input, read.reading, path)
    mine = _aspect_spread(report_input, key, aspect_id, arm_id, path)
    theirs = _aspect_spread(report_input, key, aspect_id, control.id, path)
    factor = 100.0 if _scale_of(report_input, key)[0] == "share" else 1.0
    per_case = (
        None
        if mine.per_case is None or theirs.per_case is None
        else factor * math.hypot(mine.per_case, theirs.per_case)
    )
    count = None if mine.n is None or theirs.n is None else min(mine.n, theirs.n)
    return Spread(
        factor * (mine.mean - theirs.mean),
        factor * math.hypot(mine.se, theirs.se),
        count,
        per_case,
        mine.level,
        mine.basis,
        mine.cases if mine.cases == theirs.cases else None,
    )


def _settle_floor_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, floor, arm_id, field = values
    read = next(
        held for held in report_input.floors if (held.aspect, held.floor, held.arm) == (aspect_id, floor, arm_id)
    )
    if read.margin is None:
        raise _unstated(
            path, f"reads {arm_id}'s read of floor {floor} against its margin, which the read does not state"
        )
    if field == "line":
        return _number(path, -float(read.margin), "the floor's line: the drop it allows", "points", change=True)
    spread = _floor_spread(report_input, aspect_id, floor, arm_id, path)
    if field == "change":
        return _number(
            path, spread.mean, "the candidate's change from the control on the floor's rate", "points", change=True
        )
    return _settle_line(report_input, path, spread, "", -float(read.margin), True, field, in_points=True)


# --- settling two arms apart on an aspect's pass rate: trials passed over usable, on its scoreboard -------------------

#: What settling two arms apart on an aspect's pass rate reads: each arm's rate, their difference, the sizes and trials
#: (as two arms apart on a measure), how often a fresh run reads the other arm ahead, the level, the trials each case ran
#: today, and what the spread was read from (its count basis).
_PASSED_FIELDS = (
    "first",
    "second",
    "difference",
    "cases_needed",
    "cases_needed_80",
    "trials_needed",
    "trials_needed_80",
    "wrong_side",
    "level",
    "repeats",
    "basis",
)


def _passing_board(report_input: ReportInput, aspect_id: str) -> dict[str, ScoreboardRow]:
    """The arms an aspect's scoreboard states a pass rate for: trials passed, and usable trials above none."""
    return {
        arm: row
        for arm, row in _aspect_board(report_input, aspect_id).items()
        if row.passed is not None and row.usable is not None and row.usable > 0
    }


def _passed_aspects(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return [aspect.id for aspect in report_input.aspects if len(_passing_board(report_input, aspect.id)) >= 2]


def _passed_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(_passing_board(report_input, prior[-1]))


def _passed_others(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return [arm for arm in _passing_board(report_input, prior[-2]) if arm != prior[-1]]


@dataclass(frozen=True)
class _PassedApart:
    """Two arms apart on an aspect's pass rate, in points: the difference, its spread, the level, today's repeats where
    both arms ran each case alike, and what the spread was read from."""

    difference: float
    se: float
    per_case: float | None
    level: float
    repeats: float | None
    basis: str


def _board_repeats(rows: Sequence[ScoreboardRow]) -> float | None:
    """The trials each case ran today, where every arm states its cases and ran each as often; None otherwise."""
    repeats = {
        float(row.usable) / float(row.cases)
        for row in rows
        if row.usable is not None and row.cases is not None and row.cases > 0
    }
    return repeats.pop() if len(repeats) == 1 and all(row.cases for row in rows) else None


def _passed_reading(
    report_input: ReportInput, aspect_id: str, arm_id: str, other: str
) -> tuple[Reading, tuple[float, float], float] | None:
    """The reading that pairs the two arms on the aspect, its interval, and the sign that makes it the first minus the
    second: one of them is the control, and the other's reading states an interval at a level; None otherwise."""
    control = _control(report_input)
    if control is None or control.id not in (arm_id, other):
        return None
    candidate = other if arm_id == control.id else arm_id
    aspect = _aspect_by_id(report_input, aspect_id)
    held = [
        reading
        for reading in aspect.readings
        if reading.arm == candidate
        and reading.gain_interval is not None
        and level_share(reading.gain_confidence) is not None
    ]
    if len(held) != 1 or held[0].gain_interval is None:
        return None
    low, high = held[0].gain_interval
    return held[0], (float(low), float(high)), 1.0 if candidate == arm_id else -1.0


#: Why an arm's spread read on its trials may be too narrow.
_TRIALS_NARROWER = "which reads narrower than its cases would where a case ran more than once"

#: What a paired row's two estimates are, beside its difference: not the same figures, so said once in its basis.
_PAIRED_ESTIMATES = (
    "first and second are each arm's pooled pass rate on the aspect's scoreboard (trials passed over usable), while "
    "the difference is the reading's own change, the mean of per-case shares over the cases it pairs (under one test "
    "setup where several ran), so it need not equal first minus second: it differs where cases ran unequal usable "
    "trials, the reading pairs fewer cases than the scoreboard counts, or one setup was read; cite the difference "
    "with the sizes, never first minus second"
)


def _passed_apart(report_input: ReportInput, aspect_id: str, arm_id: str, other: str, path: str) -> _PassedApart:
    """Two arms apart on an aspect's pass rate (its scoreboard's trials passed over usable), in points: paired where the
    aspect's reading of a candidate against the control states its interval (its own change, interval and level, the
    cases each arm ran), else each arm's binomial spread from its own count, read on its distinct cases where its
    scoreboard states them, combined as independent."""
    board = _passing_board(report_input, aspect_id)
    rows = (board[arm_id], board[other])
    repeats = _board_repeats(rows)
    paired = _passed_reading(report_input, aspect_id, arm_id, other)
    if paired is not None:
        reading, interval, sign = paired
        try:
            spread = difference_spread(
                float(reading.change),
                interval,
                level_share(reading.gain_confidence),
                tuple(float(count) for count in reading.cases),
            )
        except NoSpread as none:
            raise _unstated(
                path, f"reads {reading.arm}'s reading on {aspect_id}, which has no spread: {none}"
            ) from none
        return _PassedApart(
            difference=sign * spread.mean,
            se=spread.se,
            per_case=spread.per_case,
            level=spread.level,
            repeats=repeats,
            basis=(
                f"paired: {reading.arm}'s reading on {aspect_id} against the control, its own change and interval at "
                "its stated level, over the cases each arm ran (the reading's cases); a size counts cases each arm "
                f"runs, each run as often as today. {_PAIRED_ESTIMATES}"
            ),
        )
    spreads = []
    for arm, row in zip((arm_id, other), rows, strict=True):
        try:
            spreads.append(
                stat_spread(
                    float(row.passed or 0) / float(row.usable or 1),
                    n=float(row.usable or 0),
                    rate="share",
                    cases=None if row.cases is None else float(row.cases),
                )
            )
        except NoSpread as none:
            raise _unstated(path, f"reads {arm}'s pass rate on {aspect_id}, which has no spread: {none}") from none
    first, second = spreads
    per_case = (
        None if first.per_case is None or second.per_case is None else 100 * math.hypot(first.per_case, second.per_case)
    )
    on_cases = [spread.cases is not None for spread in spreads]
    if all(on_cases):
        counted = f"read on its distinct cases, each counted once: {_REPEATS_BOUND}"
    elif not any(on_cases):
        counted = f"read on its usable trials as if each were independent, {_TRIALS_NARROWER}"
    else:
        cased, trialled = (arm_id, other) if on_cases[0] else (other, arm_id)
        counted = (
            f"read for {cased} on its distinct cases, each counted once ({_REPEATS_BOUND}), and for {trialled}, whose "
            f"scoreboard states no cases, on its usable trials as if each were independent, {_TRIALS_NARROWER}"
        )
    return _PassedApart(
        difference=100 * (first.mean - second.mean),
        se=100 * math.hypot(first.se, second.se),
        per_case=per_case,
        level=first.level,
        repeats=repeats,
        basis=(
            f"independent: each arm's trials passed over usable on {aspect_id}'s scoreboard, a pass/fail rate's "
            f"binomial spread {counted}; combined as if the arms ran different cases"
        ),
    )


def _settle_passed_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, arm_id, other, field = values
    if arm_id == other:
        raise _refuse(path, "settles an arm against itself; name two arms")
    board = _passing_board(report_input, aspect_id)
    if field in ("first", "second"):
        row = board[arm_id if field == "first" else other]
        rate = float(row.passed or 0) / float(row.usable or 1)
        return _number(path, rate, "the arm's pass rate on the aspect: trials passed over usable", "share")
    apart = _passed_apart(report_input, aspect_id, arm_id, other, path)
    if field == "difference":
        return _number(path, apart.difference, "the two arms' difference in pass rate", "points", change=True)
    if field == "basis":
        return Figure(path=path, kind="text", value=apart.basis)
    if field == "level":
        return _sided_level(path, apart.level, 2)
    if field == "repeats":
        return _number(path, apart.repeats, "the trials each case ran today, where both arms ran each alike", "plain")
    if field == "wrong_side":
        share = _settled(path, lambda: wrong_side_share(apart.difference, apart.se))
        return _number(path, share, "how often a fresh run of today's size reads the other arm ahead", "share")
    per_case = apart.per_case
    if per_case is None:
        raise _unstated(path, "names the cases a design needs, which needs the count each arm's rate rests on")
    power = POWER if field.endswith("_80") else 0.5
    cases = _settled(path, lambda: cases_to_separate(per_case, apart.difference, apart.level, sided=2, power=power))
    if field.startswith("trials_"):
        if apart.repeats is None:
            raise _unstated(
                path,
                "names the trials a size takes, and the two arms' scoreboards state no cases, or ran them a different "
                "number of times",
            )
        return _number(path, math.ceil(cases * apart.repeats - 1e-9), "the trials the cases take", "count")
    return _number(path, cases, "the cases each arm needs", "count")


def passed_aspects(report_input: ReportInput, arms: Sequence[str]) -> list[str]:
    """The aspects whose scoreboard states a pass rate for every one of ``arms`` (trials passed over usable), which a
    ``settle.passed.<aspect>.`` key settles them apart on, in the input's aspect order.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arms: The arms, by id.
    :ptype arms: Sequence[str]
    :return: The aspects, by id; empty where none holds every arm's pass rate.
    :rtype: list[str]
    """
    return [
        aspect for aspect in _passed_aspects(report_input) if set(arms) <= set(_passing_board(report_input, aspect))
    ]


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
    compares: Callable[[ReportInput, tuple[str, ...]], Iterable[float] | None] | None = None,
) -> _Family:
    return _Family(
        slots=slots,
        what=what,
        resolve=resolve,
        derived=derived,
        derived_statistics=derived_statistics,
        example=example,
        derived_note=derived_note,
        compares=compares,
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
        "spread (its mean's over today's repeats, where the count is trials), in points for a rate; `n` the count it "
        "rests on and `counts` what that count counts (`cases`; `trials`, where a repeated case counts again; or "
        "`observations`, a statistic's own n that says neither); `cases` the distinct cases behind a count of trials "
        "and `repeats` the trials each ran (a binomial spread there is read on the cases, each counted once: the "
        "widest its repeats allow); `level` the level a result read from it is stated at; `basis` what the "
        "standard error was read from (a stated interval or half-width at a stated level, a standard error, or a "
        "pass/fail rate's binomial spread from a count, naming where the count was read: its own, the slice's stated "
        "cases, or the arm's scoreboard); `interval`, `low` and `high` the interval it gives at that level (the mean "
        "plus and minus the normal quantile times `se`, a rate's ends held to none to all; none for a rate of none or "
        "all), which a chart of a rate draws where the input states no interval",
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
        compares=lambda ri, values: _means_of(ri, values[2]),
        example="compare.fast.v2.base.tone.kept.overall.interval|pts",
    ),
    _family_of(
        _Word("settle"),
        _ARM,
        _Ids("<arm>", _other_arms),
        _Ids("<measure>", _pair_keys),
        _Ids("<where>", _pair_wheres),
        _Statistic(_SETTLE_PAIR_FIELDS, "<field>"),
        what="the cases each of two arms needs for a two-sided interval on their difference, centred on today's, to "
        "clear zero: `cases_needed` (half such designs clear if the truth equals today's difference), "
        "`cases_needed_80` (four in five do), in cases where the figures state their cases (each run as often as "
        "today), else in what their count counts (`spread.<arm>.<measure>.<where>.counts`); `trials_needed` and "
        "`trials_needed_80` the trials those take at today's repeats; `wrong_side` how often a fresh run of today's "
        "size reads the other arm ahead; `level` the level, with its sidedness (`95% two-sided`)",
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
        _Statistic((*_SETTLE_LINE_FIELDS, *_SETTLE_COUNT_FIELDS), "<field>"),
        what="an arm's mean against a declared criterion's threshold, where it reads it, on the aspect the criterion "
        "reads: `distance` toward passing "
        "(negative where today's figure misses); the cases a design needs for a one-sided bound centred on today's "
        "figure to sit wholly on today's side of the line, clearing it where the figure passes and confirming the "
        "miss where it misses (`cases_needed`: half such designs; `cases_needed_80`: four in five), and the same for "
        "the two-sided interval a page prints (`cases_needed_two_sided`, `cases_needed_two_sided_80`), each in cases "
        "where the figure states its cases (each run as often as today), else in what its count counts "
        "(`spread.<arm>.<measure>.<where>.counts`), and each with the trials it takes at today's repeats "
        "(`trials_needed`, `trials_needed_80`, `trials_needed_two_sided`, `trials_needed_two_sided_80`); on a "
        "repeated design a binomial spread is read on the cases, each counted once, so a size is the most cases the "
        "repeats could need; `wrong_side`, how often a fresh run of today's size lands on the other side; `level` the "
        "level, with its sidedness (`95% one-sided`); `n`, `counts`, `cases`, `repeats` and `basis` the count the "
        "arm's spread rests on, what it counts, the distinct cases and trials each ran, and what the spread was read "
        "from, as `spread.` reads them, on the criterion's aspect",
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
    _family_of(
        _Word("settle"),
        _Word("margin"),
        _Ids("<aspect>", _margined_aspects),
        _Ids("<candidate>", _margined_arms),
        _Statistic(_MARGIN_FIELDS, "<field>"),
        what="a candidate's reading on an aspect against the aspect's margin: `line` minus the margin, `change` the "
        "reading's change from the control (in points), then what a criterion's settle reads, the spread read off "
        "the reading's own interval at its stated level, each arm running the size in cases",
        resolve=_settle_margin_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="settle.margin.answers.fast.v2.cases_needed",
    ),
    _family_of(
        _Word("settle"),
        _Word("floor"),
        _Ids("<aspect>", _floored_aspects),
        _Ids("<floor>", _floored_keys),
        _Ids("<candidate>", _floored_arms),
        _Statistic(_MARGIN_FIELDS, "<field>"),
        what="a candidate's change from the control against a floor read on it: `line` minus the floor's margin, "
        "`change` the change (in points), then what a criterion's settle reads; the change is the aspect's reading "
        "where the floor speaks for its pass rate, else the candidate's rate on the floor's own measure minus the "
        "control's, their spreads combined as independent",
        resolve=_settle_floor_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="settle.floor.answers.answers/rate_drop/*.fast.v2.wrong_side",
    ),
    _family_of(
        _Word("settle"),
        _Word(TRIAL_PASS_NAME),
        _Ids("<aspect>", _passed_aspects),
        _Ids("<arm>", _passed_arms),
        _Ids("<arm>", _passed_others),
        _Statistic(_PASSED_FIELDS, "<field>"),
        what="two arms apart on an aspect's pass rate (its scoreboard's trials passed over usable), the first minus "
        "the second: `first` and `second` each arm's pooled rate, `difference` in points (on a paired row the "
        "reading's own change, which need not equal first minus second; `basis` says so), then what two arms' "
        "settle reads "
        "(`cases_needed`, `cases_needed_80`, `trials_needed`, `trials_needed_80`, `wrong_side`, `level` two-sided); "
        "`repeats` the trials each case ran today; `basis` the count basis: paired where the aspect's reading of a "
        "candidate against the control states its interval (its own change and interval, over the cases each arm "
        "ran), else each arm's binomial spread from its own trials, read on its distinct cases where its scoreboard "
        "states them, combined as independent",
        resolve=_settle_passed_figure,
        derived=True,
        derived_note=_SEPARATION_NOTE,
        example="settle.passed.answers.fast.v2.base.cases_needed",
    ),
)

# --- an assumed count: the separation families read as if each slice stating no count held one ---------------------

#: How ``assumed.share.`` reads a slice's count: the arm's own distinct cases on the measure, times the slice's
#: traffic share (and its trials times the share, on a repeated design).
ASSUMED_SHARE = "share"

#: What the grammar says of a figure read under an assumed count, in place of "descriptive, no interval".
_ASSUMED_NOTE = (
    "Hypothetical: read as if each slice stating no count of its own held the assumed cases, never a figure the input "
    "states; computed as its plain key's is."
)


def _assumed_count(path: str, spelled: str) -> int:
    """The cases an ``assumed.<cases>.`` key spells, refused unless a whole number above none."""
    value = float(spelled)
    if not value.is_integer() or value < 1:
        raise _refuse(
            path, f"assumes {spelled} cases a slice, and an assumed count is a whole number of cases, one or more"
        )
    return int(value)


def _slice_share(report_input: ReportInput, where: str) -> float | None:
    """A slice's stated share of traffic: a task's or a stratum's; None for a cell, or one that states none."""
    kind, _dot, named = where.partition(".")
    if kind == "task":
        shares = [task.share for task in report_input.tasks if task.id == named]
    elif kind == "strata":
        shares = [stratum.share for stratum in report_input.strata if stratum.id == named]
    else:
        shares = []
    return float(shares[0]) if len(shares) == 1 and shares[0] is not None else None


def _assumed_cases(
    report_input: ReportInput, assumed: int | None, arm_id: str, key: str, where: str, path: str
) -> dict[str, int] | None:
    """The count an assumption gives one arm's figure at a slice, where its count is unstated, as the statistic's own
    fields (``n_cases``, and ``n`` on a repeated design); None where the figure states its count (the assumption leaves
    it as it is). ``assumed`` None reads the arm's own distinct cases times the slice's traffic share, and on a
    repeated design its trials times the share as ``n``, so the slice keeps the arm's repeats."""
    held = _held_where(report_input, arm_id, key, where)
    if held is None or held.count is not None:
        return None
    if assumed is not None:
        return {"n_cases": assumed}
    share = _slice_share(report_input, where)
    overall = _held_where(report_input, arm_id, key, _OVERALL)
    cases = None
    if overall is not None and overall.count is not None:
        cases = overall.cases if overall.cases is not None else overall.count if overall.counted == "cases" else None
    if share is None or overall is None or cases is None:
        lacking = "a traffic share" if share is None else f"{arm_id}'s own distinct cases on {key!r}"
        raise _unstated(
            path, f"reads {where}'s cases as {arm_id}'s cases times its traffic share, and it states no {lacking}"
        )
    filled = {"n_cases": max(1, round(cases * share))}
    if overall.cases is not None and overall.count is not None:
        filled["n"] = max(filled["n_cases"], round(overall.count * share))
    return filled


def _assumed_input(
    report_input: ReportInput, assumed: int | None, arms: Sequence[str], key: str, where: str, path: str
) -> ReportInput:
    """The input as if each of ``arms``' figures on ``key`` at slice ``where`` that states no count held the assumed
    cases; refused where ``where`` is no slice, or every one of them states its count (the plain key reads it)."""
    if where == _OVERALL:
        raise _refuse(
            path, "assumes a count overall, and an assumed count fills only a slice: cite the key without `assumed.`"
        )
    filled = {arm: _assumed_cases(report_input, assumed, arm, key, where, path) for arm in arms}
    filled = {arm: cases for arm, cases in filled.items() if cases is not None}
    if not filled:
        raise _refuse(
            path,
            f"assumes a count at {where}, where every figure it reads states its own: cite the key without `assumed.`",
        )
    kind, _dot, named = where.partition(".")

    def counted(per_arm: Mapping[str, Mapping[str, Stat]]) -> dict[str, dict[str, Stat]]:
        return {
            arm: {
                measure: stat.model_copy(update=filled[arm]) if arm in filled and measure == key else stat
                for measure, stat in figures.items()
            }
            for arm, figures in per_arm.items()
        }

    if kind == "task":
        tasks = [
            task.model_copy(update={"per_arm": counted(task.per_arm)}) if task.id == named else task
            for task in report_input.tasks
        ]
        return report_input.model_copy(update={"tasks": tasks})
    if kind == "strata":
        strata = [
            stratum.model_copy(update={"per_arm": counted(stratum.per_arm)}) if stratum.id == named else stratum
            for stratum in report_input.strata
        ]
        return report_input.model_copy(update={"strata": strata})
    cells = {
        cell_id: cell.model_copy(update={"per_arm": counted(cell.per_arm)}) if _cell_spelling(cell) == named else cell
        for cell_id, cell in report_input.cells.items()
    }
    return report_input.model_copy(update={"cells": cells})


#: What one separation family's values name, for an assumption to fill: the arms, the measure, and the place.
_Reads = Callable[[ReportInput, tuple[str, ...]], tuple[tuple[str, ...], str, str]]


def _assumed_resolver(
    inner: _Family, reads: _Reads, *, counted: bool
) -> Callable[[ReportInput, tuple[str, ...], str], Figure]:
    """A separation family read under an assumed count: ``assumed.<cases>.`` (``counted``) or ``assumed.share.``."""

    def resolve_assumed(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
        assumed = _assumed_count(path, values[0]) if counted else None
        rest = values[1:] if counted else values
        arms, key, where = reads(report_input, rest)
        figure = inner.resolve(_assumed_input(report_input, assumed, arms, key, where, path), rest, path)
        if isinstance(figure.value, str) and figure.kind == "text" and path.endswith(".basis"):
            how = (
                f"{assumed} cases"
                if assumed is not None
                else "the arm's own distinct cases times the slice's traffic share"
            )
            own = f"from {_COUNT_SOURCES['own']}"
            said = figure.value.replace(own, f"from an assumed count ({how})")
            return replace(figure, value=said if said != figure.value else f"{figure.value}; the count assumed: {how}")
        return figure

    return resolve_assumed


def _stripped(slot: _Slot, prefix: int) -> _Slot:
    """An inner family's slot, its options read without the assumption's own values before it."""
    if not isinstance(slot, _Ids) or not prefix:
        return slot
    options = slot.options
    return _Ids(slot.name, lambda report_input, prior: options(report_input, prior[prefix:]))


def _criterion_reads(report_input: ReportInput, values: tuple[str, ...]) -> tuple[tuple[str, ...], str, str]:
    criterion_id, arm_id, where, _field = values
    return (arm_id,), _criterion(report_input, criterion_id).measure, where


#: Each separation family an assumption reads, with what its values name.
_ASSUMABLE: tuple[tuple[_Family, _Reads], ...] = (
    (_SEPARATION_FAMILIES[0], lambda _ri, values: ((values[0],), values[1], values[2])),
    (_SEPARATION_FAMILIES[1], lambda _ri, values: ((values[0], values[1]), values[2], values[3])),
    (_SEPARATION_FAMILIES[2], lambda _ri, values: ((values[0], values[1]), values[2], values[3])),
    (_SEPARATION_FAMILIES[3], _criterion_reads),
    (_SEPARATION_FAMILIES[4], lambda _ri, values: ((values[1],), values[2], values[3])),
)


def _assumed_family(inner: _Family, reads: _Reads, *, counted: bool) -> _Family:
    head: tuple[_Slot, ...] = (_Word("assumed"), _Number("<cases>")) if counted else (_Word("assumed.share"),)
    how = (
        "as if each slice stating no count of its own held `<cases>` cases"
        if counted
        else "as if each slice stating no count of its own held the arm's own distinct cases times the slice's traffic share"
    )
    return _family_of(
        *head,
        *(_stripped(slot, 1 if counted else 0) for slot in inner.slots),
        what=f"{inner.form()} read {how}; refused overall, and at a slice where every figure states its count",
        resolve=_assumed_resolver(inner, reads, counted=counted),
        derived=True,
        derived_note=_ASSUMED_NOTE,
    )


#: The separation families read under an assumed count, each by a number of cases and by traffic share.
_ASSUMED_FAMILIES: tuple[_Family, ...] = tuple(
    _assumed_family(inner, reads, counted=counted) for counted in (True, False) for inner, reads in _ASSUMABLE
)


def assumable(report_input: ReportInput, arms: Sequence[str], key: str, where: str) -> bool:
    """Whether an assumed count would fill something at a separation key's place: it is a slice, and one of ``arms``
    states a mean on ``key`` there and no count.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arms: The arms, by id.
    :ptype arms: Sequence[str]
    :param key: The measure or judged dimension, by key.
    :ptype key: str
    :param where: The place, as ``<where>`` spells it.
    :ptype where: str
    :return: Whether an ``assumed.`` key reads there.
    :rtype: bool
    """
    if where == _OVERALL:
        return False
    held = [_held_where(report_input, arm, key, where) for arm in arms]
    return any(one is not None and one.count is None for one in held)


# --- lever effects: one setting moved between two arms that differ in it alone ------------------------------------

#: A lever effect's fields: the second arm on the measure minus the first (``effect``), the interval on it as the compare
#: family reads the same two arms (its ends, its level), and whether that interval is paired or independent.
_LEVER_FIELDS = ("effect", "interval", "low", "high", "level", "basis")

#: An interaction's fields: the second move's effect minus the first's (``difference``), the interval on it, its level.
_INTERACTION_FIELDS = ("difference", "interval", "low", "high", "level")


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


def _lever_where(report_input: ReportInput, place: str, path: str) -> str:
    """A lever key's ``<place>`` as a separation key's ``<where>`` spells it: ``overall`` for none, else the task, the
    stratum or the cell it names."""
    if not place:
        return _OVERALL
    _place_figures(report_input, place, path)
    if any(task.id == place for task in report_input.tasks):
        return f"task.{place}"
    if any(stratum.id == place for stratum in report_input.strata):
        return f"strata.{place}"
    return f"cells.{place}"


@dataclass(frozen=True)
class _Effect:
    """A move's effect and, where one reads, the interval on it (:class:`_Difference`), or why none does."""

    change: float
    read: _Difference | None
    why: str


def _lever_effect(report_input: ReportInput, values: tuple[str, ...], path: str) -> _Effect:
    """A move's effect, read as the compare family reads the arm it ends at against the arm it starts from (one spread
    rule for both): ``values`` = (from, to, measure, place)."""
    first, second, key, place = values
    where = _lever_where(report_input, place, path)
    try:
        read = _read_difference(report_input, second, first, key, where)
    except NoSpread as none:
        change = _mean_difference(report_input, second, first, key, where)
        if change is None:
            words = "" if where == _OVERALL else f" in {place}"
            raise _unstated(path, f"names {first}'s and {second}'s means on {key}{words}, which one lacks") from none
        return _Effect(change, None, str(none))
    return _Effect(read.difference, read, "")


def _recovered(first: tuple[float, float, float], second: tuple[float, float, float]) -> tuple[float, float]:
    """The interval on ``second``'s estimate minus ``first``'s, each ``(estimate, low, high)``, from their own
    intervals by the method of variance estimates recovery, the two read as independent."""
    (one, one_low, one_high), (two, two_low, two_high) = first, second
    change = two - one
    return change - math.hypot(two - two_low, one_high - one), change + math.hypot(two_high - two, one - one_low)


def _lever_figure(report_input: ReportInput, key: str, effect: _Effect, field: str, path: str) -> Figure:
    scale, unit = _scale_of(report_input, key)
    if field in ("effect", "difference"):
        return Figure(path=path, kind="number", value=effect.change, scale=scale, unit=unit, change=True)
    read = effect.read
    if read is None:
        raise _unstated(path, f"names an interval, which none reads: {effect.why}")
    if field == "basis":
        return Figure(path=path, kind="text", value=read.basis)
    if field == "level":
        if read.level is None:
            raise _unstated(path, "names the level of the contrast's interval, which the input does not state")
        return _level(path, read.level, "the interval's level")
    if field == "interval":
        return Figure(path=path, kind="interval", value=(read.low, read.high), scale=scale, unit=unit, change=True)
    end = read.low if field == "low" else read.high
    return Figure(path=path, kind="number", value=end, scale=scale, unit=unit, change=True)


def _lever(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    """``lever.<setting>.<from>.<to>.<measure>[.<place>].<field>``."""
    _setting_id, first, second, key, *place, field = values
    effect = _lever_effect(report_input, (first, second, key, place[0] if place else ""), path)
    return _lever_figure(report_input, key, effect, field, path)


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
    one = _lever_effect(report_input, (first, second, key, where), path)
    two = _lever_effect(report_input, (start, end, key, where), path)
    change = two.change - one.change
    if one.read is None or two.read is None:
        return _lever_figure(report_input, key, _Effect(change, None, one.why or two.why), field, path)
    if one.read.level != two.read.level:
        why = "its two effects' intervals are at different levels, which do not combine"
        return _lever_figure(report_input, key, _Effect(change, None, why), field, path)
    first_read, second_read = one.read, two.read
    low, high = _recovered(
        (first_read.difference, first_read.low, first_read.high),
        (second_read.difference, second_read.low, second_read.high),
    )
    read = _Difference(change, low, high, first_read.level, "")
    return _lever_figure(report_input, key, _Effect(change, read, ""), field, path)


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
_LEVER_NOTE = "Derived: computed here from two arms' stated figures and their spreads, never a test the pipeline ran."

#: What every lever key's figures are: computed here, from two arms' figures the input states.
_LEVER_BASIS = (
    "Computed here, as `compare.<to>.<from>.<measure>.<where>` reads the same two arms: an effect's `interval` (its "
    "ends `low`, `high`, its `level`) is a contrast's own where one pairs the two arms overall, else the two arms' own "
    "spreads combined as independent (a stated interval, half-width or standard error, or a pass/fail rate's binomial "
    "spread from its count), and its `basis` says which; an interaction's is recovered from its two effects' intervals "
    "as if independent (MOVER), at their common level; with no spread to read there is no interval. levers_effects "
    "lists them"
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

#: What the grammar says of an overall interval derived from its cells, in place of "descriptive, no interval".
_OVERALL_INTERVAL_NOTE = (
    "Derived: combined here from the cells' stated spreads, the cells read as independent, never a test the pipeline "
    "ran."
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
        compares=lambda ri, values: _means_of(ri, values[1]),
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
        what="an arm's cost per 1,000 of what the cost counts (`per 1,000 conversations` where the cost measure's "
        "label counts conversations)",
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
        what="how an arm's trials came out. " + _SCOREBOARD_FAILED,
        resolve=_scoreboard,
    ),
    _family_of(
        _Word("calls.unanswered"),
        _keyed("<arm>", _unanswered_arms),
        what="how many of an arm's calls went unanswered, retried and lost together, summed over every role that "
        "counted them: beside its cost, never in it (the cost counts the answered calls); not counted, and refused, "
        "where a role counted only one kind. " + _UNANSWERED_RULE,
        resolve=_unanswered,
        example="calls.unanswered.fast.v2",
    ),
    _family_of(
        _Word("calls.unanswered"),
        _keyed("<arm>", _unanswered_arms),
        _Statistic(_UNANSWERED_FIELDS, "<field>"),
        what="the same, by field: `retried` (a forced tool choice refused, then answered unforced in the same call: "
        "attempts and latency, never a failure), `lost` (sent and never answered, so the turn ended in error: failures "
        "the arm's trials already count), `calls` the answered calls beside them, `share` retried and lost as a share "
        "of every call sent, `reasons` the reasons counted, in words, `uncounted` the trials whose calls counted none",
        resolve=_unanswered,
        example="calls.unanswered.fast.v2.share|pct",
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
        what="a declared criterion: its threshold in its measure's units, the measure, statistic and direction, and "
        "the aspect whose figures it reads (its own, else the lead aspect)",
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
        what="how much more a candidate costs than the control, as a share of the control's cost",
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
        compares=lambda ri, values: _means_of(ri, values[2]),
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
        what="how an arm's trials came out on one aspect. " + _SCOREBOARD_FAILED,
        resolve=_aspect_scoreboard,
        example="aspects.speed.scoreboard.fast.v2.passed",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("calls.unanswered"),
        _Ids("<arm>", _aspect_unanswered_arms),
        what="how many of an arm's calls went unanswered on one aspect, retried and lost together",
        resolve=_aspect_unanswered,
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("calls.unanswered"),
        _Ids("<arm>", _aspect_unanswered_arms),
        _Statistic(_UNANSWERED_FIELDS, "<field>"),
        what="the same, by field, as `calls.unanswered.<arm>.<field>` reads it",
        resolve=_aspect_unanswered,
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
        "threshold, signed toward passing, in points for a rate; `low` and `high` are derived where the input states "
        "no interval and `overall_interval` derives one",
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
        "1,000 of what the cost counts; `per_point` the cost gap per point gained (per unit off a points scale), its "
        "`_low` and `_high` ends read off `gain_interval`, the contrast's interval where the move is between the "
        "control and a candidate; an end the interval leaves open is refused as unbounded; `monthly_` fields at the "
        "declared monthly volume. Cost is read as exact",
        resolve=_price,
        derived=True,
        derived_note=_PRICE_NOTE,
        example="price.base.fast.v2.latency_ms.monthly_per_point",
        compares=lambda ri, values: _gap_compares(ri, _costs_per_thousand(ri), values[-1]),
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
        compares=_slice_compares("task"),
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
        compares=_slice_compares("strata"),
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
        compares=_slice_compares("cells"),
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
        _Word("overall_interval"),
        _Ids("<measure>", _unspread_keys),
        _Ids("<arm>", _unspread_arms),
        _Statistic(_OVERALL_INTERVAL_FIELDS, "<field>"),
        what="the interval on an arm's overall figure that states none, where that figure is the plain mean of one "
        "measure's target-matrix cells and every cell states a spread: `se` is the root of the cells' squared "
        "standard errors over their count, `interval` the figure plus and minus it at the cells' stated `level`, `of` "
        "the measure averaged, `cells` how many, `weights` whether a case-weighted mean of them fits it too (the "
        "interval then reads the wider standard error), `basis` how it was built. Refused where the figure is no plain mean "
        "of its cells, a cell states no spread, or the levels differ. A gate on the figure reads this interval, and "
        "`criteria.<criterion>.<arm>.low` and `.high` read its ends",
        resolve=_overall_interval,
        derived=True,
        derived_note=_OVERALL_INTERVAL_NOTE,
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
    *_ASSUMED_FAMILIES,
    _family_of(
        *_LEVER,
        _Statistic(_LEVER_FIELDS, "<field>"),
        what="a lever's effect: one setting moved between two arms whose settings differ in it alone, `<to>` minus "
        "`<from>` on a measure or judged dimension (`effect`: a pairing contrast's difference, else the two means'), "
        "overall. " + _LEVER_BASIS,
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
        "and guardrail, borderline counting as passing; `undecided` how many no gate figure fails while a gate has no "
        "figure for them somewhere: neither passed nor removed; `pairs` how many pairs of arms its pair readings read "
        "apart, `arm_pairs` how many pairs of arms there are; `examples` the failed trials the snapshot froze as "
        "examples, `failures` the failed trials the run holds, which they were kept from, a failure counted as the "
        "examples count one, not as the scoreboard's `failed` (refused where the snapshot does not count them: not "
        "known)",
        resolve=_campaign_count,
        derived_statistics=("surviving", "undecided"),
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
        what="how many of a criterion's places an arm misses: a figure on the wrong side of the threshold (a place "
        "with no figure is unknown, counted by `count.unknown`)",
        resolve=_failing_count,
        derived=True,
        example="count.failing.cell_floor.fast.v2|words",
    ),
    _family_of(
        _Word("count.unknown"),
        _keyed("<criterion>", _criterion_ids),
        _ARM,
        what="how many of a criterion's places hold no figure for an arm: unknown there, neither passed nor missed",
        resolve=_unknown_count,
        derived=True,
        example="count.unknown.p95.fast.v2|words",
    ),
    _family_of(
        _Word("count.failed"),
        _ARM,
        what="how many gates and guardrails an arm misses on a figure somewhere they read",
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
    _family_of(
        _Word("count.examples"),
        _keyed("<aspect>", _example_aspects),
        _Ids("<arm>", _example_arms),
        what="how many of one arm's failed trials on one aspect the snapshot froze as examples (`count.examples` "
        "counts every one)",
        resolve=_kept_count,
        example="count.examples.answers.fast.v2",
    ),
    _family_of(
        _Word("count.examples"),
        _keyed("<aspect>", _example_aspects),
        _Ids("<arm>", _example_arms),
        _Ids("<stratum>", _example_strata),
        what="the same within one stratum or task",
        resolve=_kept_count,
    ),
    _family_of(
        _Word("count.failures"),
        _keyed("<aspect>", _example_aspects),
        _Ids("<arm>", _example_arms),
        what="how many of one arm's trials on one aspect failed, as the examples count a failure: the pool its examples "
        "were kept from (`count.failures` counts every one); not known, and refused, where the snapshot does not "
        "count them. " + _FAILURES_RULE,
        resolve=_failed_trials_count,
        example="count.failures.answers.fast.v2",
    ),
    _family_of(
        _Word("count.failures"),
        _keyed("<aspect>", _example_aspects),
        _Ids("<arm>", _example_arms),
        _Ids("<stratum>", _example_strata),
        what="the same within one stratum or task",
        resolve=_failed_trials_count,
    ),
    _family_of(
        _Word("overlap"),
        _keyed("<aspect>", lambda ri: [aspect.id for aspect in ri.aspects]),
        _Ids("<aspect>", _other_aspects),
        _Statistic(_OVERLAP_FIELDS, "<field>"),
        what="whether two aspects ran the same cases, by the cases' stored ids: `same` in words (the same cases, "
        "partly, different, or not known where the snapshot does not hold their case ids), `shared` the cases both "
        "ran, `only_first` and `only_second` the cases only one ran; the counts are refused where not known",
        resolve=_overlap,
        example="overlap.answers.speed.same",
    ),
    _family_of(
        _Word("confusion"),
        _keyed("<axis>", _confusion_axes),
        _Ids("<arm>", _confusion_arms),
        _Ids("<expected>", _confusion_labels),
        _Ids("<predicted>", _confusion_labels),
        what="one cell of an arm's confusion on a label axis: the share of the expected label's cases predicted as the "
        "predicted label, a percent of its row (or a count, where the axis counts observations)",
        resolve=_confusion,
    ),
    _family_of(
        _Word("cost.gap"),
        _keyed("<arm>", _costs_per_thousand),
        _Ids("<arm>", _cost_gap_seconds),
        _Statistic(_COST_GAP_FIELDS, "<field>"),
        what="two arms' costs apart, the first's minus the second's: `per_1k` per 1,000 of what the cost counts, "
        "`monthly` per month at the declared monthly volume, `multiple` the first's cost over the second's, read with its "
        "unit under every filter (`21 times`, `2.4 times`: whole from ten up, one decimal below)",
        resolve=_cost_gap,
        derived=True,
        example="cost.gap.fast.v2.base.multiple",
        compares=lambda ri, values: _gap_compares(ri, _costs_per_thousand(ri), values[-1]),
    ),
    _family_of(
        _Word("cost.unreported"),
        _keyed("<arm>", _costs_per_thousand),
        _Statistic(_UNREPORTED_FIELDS, "<field>"),
        what="an arm's cost where some of its trials report none: `missing` how many (trials run less the cost's "
        "observations), `free` the cost per 1,000 had each cost nothing (the least the mean over every trial can be), "
        "`at_p95` had each cost the arm's own 95th percentile (a what-if, not a ceiling: the snapshot holds no "
        "highest cost); refused where the cost is not read off one measure's observations, one per trial",
        resolve=_unreported,
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


@dataclass
class _Memo:
    """What one report input has read as (:func:`_memo`): each key's figure or the refusal it raised (kept with no frame,
    :func:`_frameless`), and the ids each slot offered after the slots matched before it."""

    figures: dict[str, Figure | UnresolvableReference]
    offered: dict[tuple[Callable[[ReportInput, tuple[str, ...]], Iterable[str]], tuple[str, ...]], list[str]]
    amounts: dict[tuple[int, tuple[str, ...], str, bool], tuple[float, ...]]


@dataclass(frozen=True, eq=False)
class _ByIdentity:
    """A report input by its identity, held weakly: equal only to a handle on the same live input, so a memo never
    outlives its input's identity (an input made later at the same address is another input) and never keeps it alive.
    """

    ident: int
    ref: weakref.ref[ReportInput]

    def __hash__(self) -> int:
        return hash(self.ident)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ByIdentity) or other.ident != self.ident:
            return False
        held = self.ref()
        return held is not None and held is other.ref()


#: The most report inputs whose memos are kept at once, the least recently read let go first: a sweep reads a copy of
#: its input per value, each with a memo of a few keys, while the input it sweeps is read throughout.
_MEMO_INPUTS = 256


@functools.lru_cache(maxsize=_MEMO_INPUTS)
def _memo_of(held: _ByIdentity) -> _Memo:
    """A fresh memo for one input (:func:`_memo`)."""
    return _Memo(figures={}, offered={}, amounts={})


def _memo(report_input: ReportInput) -> _Memo:
    """The memo of one report input, made empty on its first read.

    A key's figure and a slot's ids are pure in the input, and a report input is frozen and never changed in place (an
    input with a moved line is a copy, :meth:`~pydantic.BaseModel.model_copy`, with its own memo). So a key read twice
    on one input, which a cited figure's text, a page sized by halving and a listing ordered by its figures all do,
    walks the grammar once, and each family sharing a slot reads the slot's ids once (four lever families read a
    setting's arm pairs, an O(arms squared) scan, alike).
    """
    return _memo_of(_ByIdentity(id(report_input), weakref.ref(report_input)))


def _offered(slot: _Ids, report_input: ReportInput, values: tuple[str, ...]) -> list[str]:
    """The ids an id slot offers on an input after the slots matched before it, read once per input."""
    offered = _memo(report_input).offered
    key = (slot.options, values)
    found = offered.get(key)
    if found is None:
        found = offered[key] = list(slot.options(report_input, values))
    return found


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
        options = _offered(slot, report_input, values)
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
        first = family.slots[0]
        if isinstance(first, _Word) and not path.startswith(first.spelling):
            # A family whose first word the path does not begin with stops at its start, where no refusal reads what
            # it expected (:func:`_one_reading` names every beginning there), so it is passed over unwalked.
            stop.at = max(stop.at, 0)
            continue
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
    figures = _memo(report_input).figures
    held = figures.get(key)
    if isinstance(held, UnresolvableReference):
        raise _frameless(held)
    if held is not None:
        return held
    try:
        figure = _resolved(key, report_input)
    except UnresolvableReference as refused:
        figures[key] = _frameless(refused)
        raise
    figures[key] = figure
    return figure


def _frameless(refused: UnresolvableReference) -> UnresolvableReference:
    """A copy of a refusal that holds no frame: its type and message, its cause copied the same way, and whether a
    traceback shows its context, as a first raise has it.

    A raised refusal's traceback, and its cause's, hold the frames it unwound through, each with the report input among
    its locals, so the memo keeps this copy and raises a fresh one on each read: a refused key never keeps its input
    alive, and no two reads share an exception.
    """
    twin = type(refused)(*refused.args)
    cause = refused.__cause__
    if cause is not None:
        twin.__cause__ = type(cause)(*cause.args)
    twin.__suppress_context__ = refused.__suppress_context__
    return twin


def _resolved(key: str, report_input: ReportInput) -> Figure:
    """One key's figure, read from the input (:func:`resolve`, which keeps it)."""
    if key.startswith(_ARM_PREFIX):
        arm_id = key.removeprefix(_ARM_PREFIX)
        if arm_id not in {arm.id for arm in report_input.arms}:
            raise _refuse(key, f"names no arm; the arms are {_listed(_arms(report_input, ()))}")
        return Figure(path=key, kind="arm", value=arm_id)
    reading = _one_reading(key, report_input)
    figure = reading.family.resolve(report_input, reading.values, key)
    if figure.scale in _PER_THOUSAND_FACTORS and not figure.per:
        figure = replace(figure, per=_money_per(report_input, reading.values))
    if figure.scale in _PER_THOUSAND_FACTORS and figure.kind in ("number", "interval"):
        figure = replace(
            figure,
            amounts=_money_amounts(reading, figure, report_input),
            spread=_change_spread(reading, figure, report_input),
        )
    return replace(figure, derived=figure.derived or _is_derived(reading))


def _sibling_values(reading: _Reading, report_input: ReportInput) -> list[tuple[str, ...]]:
    """The readings of a reading's family that share its words, its statistic and its ids but its arms: every arm its
    arm slots offer (each slot's ids as offered after the ones before it). A statistic is a figure of its own type (a
    scoreboard's cost per reply is not its trials' total), so it is never varied."""
    combos: list[tuple[str, ...]] = [()]
    values = iter(reading.values)
    for slot in reading.family.slots:
        if isinstance(slot, _Word):
            continue
        value = next(values)
        if isinstance(slot, _Ids) and slot.name in _ARM_SLOTS:
            combos = [(*prefix, option) for prefix in combos for option in _offered(slot, report_input, prefix)]
        else:
            combos = [(*prefix, value) for prefix in combos]
    return combos


def _sibling_figures(reading: _Reading, report_input: ReportInput) -> list[Figure]:
    """The figures of :func:`_sibling_values`' readings that the input states; one it leaves unstated is no member."""
    figures: list[Figure] = []
    for values in _sibling_values(reading, report_input):
        try:
            figures.append(reading.family.resolve(report_input, values, ".".join(values)))
        except UnresolvableReference:
            # NOSILENT: a sibling the input leaves unstated, or one its resolver refuses, has no amount to add to the
            # set; the key being read raises its own refusal where it is the one unstated.
            continue
    return figures


def _fixed_values(reading: _Reading) -> tuple[str, ...]:
    """A reading's values with its arm slots left blank: what every member of its set shares."""
    fixed: list[str] = []
    values = iter(reading.values)
    for slot in reading.family.slots:
        if isinstance(slot, _Word):
            continue
        value = next(values)
        fixed.append("" if isinstance(slot, _Ids) and slot.name in _ARM_SLOTS else value)
    return tuple(fixed)


def _numbers_of(figures: Iterable[Figure]) -> list[float]:
    """Every number the figures hold: a number's value, an interval's two ends."""
    held: list[float] = []
    for figure in figures:
        if isinstance(figure.value, float):
            held.append(figure.value)
        elif isinstance(figure.value, tuple):
            held.extend(float(end) for end in figure.value)
    return held


def _money_amounts(reading: _Reading, figure: Figure, report_input: ReportInput) -> tuple[float, ...]:
    """The set a money figure is spelled with (:class:`Figure`'s ``amounts``): a change's compared amounts where its
    family says what a change compares; else every amount its family states of the same statistic on its scale across
    the arms.

    The set is the family's, never the figure's: every key of ``cost_per_1k.*`` reads the same set, so each is spelled
    in one form. Read once per family and fixed ids on an input."""
    memo = _memo(report_input).amounts
    key = (id(reading.family), _fixed_values(reading), figure.scale, figure.change)
    held = memo.get(key)
    if held is not None:
        return held
    compared = reading.family.compares(report_input, reading.values) if reading.family.compares is not None else None
    if figure.change and compared is not None:
        found = tuple(compared)
    else:
        siblings = [one for one in _sibling_figures(reading, report_input) if one.scale == figure.scale]
        amounts = _numbers_of(one for one in siblings if not one.change)
        found = tuple(amounts or _numbers_of(siblings))
    memo[key] = found
    return found


def _change_spread(reading: _Reading, figure: Figure, report_input: ReportInput) -> float | None:
    """The width of the interval a money change is read with: its own, for an interval; for a number, its family's
    interval at the same arms (a contrast's delta beside its ``interval``), where it states one."""
    if not figure.change:
        return None
    if isinstance(figure.value, tuple):
        return abs(figure.value[1] - figure.value[0])
    statistic = reading.family.statistic()
    if statistic is None:
        return None
    at = [slot for slot in reading.family.slots if not isinstance(slot, _Word)].index(statistic)
    for option in ("interval", "ci", "gain_interval"):
        if option in statistic.options:
            values = list(reading.values)
            values[at] = option
            try:
                held = reading.family.resolve(report_input, tuple(values), ".".join(values))
            except UnresolvableReference:
                # NOSILENT: a change whose family states no interval at these arms reads its set's own decimals.
                return None
            return abs(held.value[1] - held.value[0]) if isinstance(held.value, tuple) else None
    return None


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
    """Whether a reading's figure is computed here: its family is derived, or the statistic it names is. A resolver
    may also mark one figure derived where its family copies the rest (a criterion figure's ends, read off an interval
    derived from the cells)."""
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
    """``text`` with the ``+`` a positive change carries, as :func:`~threetears.evals.kernel.numbers.format_signed`
    places it: never on a bound, never on a zero."""
    return f"+{text}" if value > 0 and not text.startswith(("<", "+")) and text != "0" else text


def _money_form(figure: Figure, factor: float) -> MoneyForm:
    """The one form a money figure's family is spelled in (#993), at ``factor`` times its scale: chosen from the
    family's whole set (:class:`Figure`'s ``amounts``), never from the figure alone, and for a change capped at the
    decimals its interval's width supports. A figure made without its set is a set of its own numbers."""
    own = (
        figure.value if isinstance(figure.value, tuple) else (figure.value,) if isinstance(figure.value, float) else ()
    )
    form = money_form(amount * factor for amount in figure.amounts or own)
    return form.for_change(figure.spread * factor) if figure.change and figure.spread is not None else form


def _money(value: float, signed: bool, form: MoneyForm) -> str:
    """Dollars as a key writes them: in its family's one form (:func:`_money_form`), ``$8/1000`` for a set too small
    for cents, signed where the figure is a change."""
    return form.spell(value, signed=signed)


#: The spelling a figure on the ``usd_per_1m`` scale reads in without a filter: dollars with the words of what it
#: counts, ``$180 per 1,000,000 turns``. No filter names it: a key's own spelling, never chosen.
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
        if figure.level:
            return _Spelled(format_number(hundred), PERCENT)
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
            return _Spelled(_money(value, signed, _money_form(figure, 1.0)))
        factor = _PER_THOUSAND_FACTORS[figure.scale]
        money = _money(value * factor, signed, _money_form(figure, factor))
        return _Spelled(f"{money} {per_count(figure.per, 1_000)}")
    if spelling == "usd_whole":
        if figure.scale not in _PER_THOUSAND_FACTORS:
            raise _refuse(figure.path, f"cannot read as dollars: it is on the {figure.scale} scale")
        if abs(value) < _WHOLE_DOLLARS_FROM:
            raise _refuse(
                figure.path,
                f"is {_money(value, signed, _money_form(figure, 1.0))}, under ${format_number(_WHOLE_DOLLARS_FROM)}, "
                "which whole dollars would "
                "round away: write it with `usd`",
            )
        dollars = "$" + group_thousands(format_number(float(round(abs(value)))))
        return _Spelled(f"-{dollars}" if value < 0 else _plus(dollars, value) if signed else dollars)
    if spelling == _PER_MILLION:
        # In its family's one form, as every other money key: a price per million of anything is whole dollars or near
        # it, so the form's cents are its precision.
        return _Spelled(f"{_money(value, signed, _money_form(figure, 1.0))} {per_count(figure.per, 1_000_000)}")
    if spelling == "words":
        whole = round(value)
        if figure.scale != "count" or whole != value or whole < 0:
            why = "it is not a whole count" if figure.scale == "count" else f"it is on the {figure.scale} scale"
            raise _refuse(figure.path, f"cannot read as a word: {why}")
        if whole < len(_COUNT_WORDS):
            return _Spelled(_COUNT_WORDS[whole])
        return _Spelled(group_thousands(format_number(float(whole))))
    if spelling == "int":
        # A price per 1,000,000 (of what it counts) also reads as a whole number, as stored plans spell it.
        if figure.scale not in ("count", "plain", "points", "usd_per_1m"):
            raise _refuse(figure.path, f"cannot read as a whole number: it is on the {figure.scale} scale")
        text = group_thousands(format_number(float(round(value))))
        text = _plus(text, value) if signed else text
        return _Spelled(f"{text} {figure.unit}" if figure.unit in _KEPT_UNITS else text)
    if figure.unit in _KEPT_UNITS:
        return _Spelled(f"{spell_multiple(value)} {figure.unit}")
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
        if not isinstance(figure.value, float):
            raise TypeError(f"{figure.path} is a number figure holding {figure.value!r}, not a float")
        spelled = _spell(figure.value, figure, spelling, absolute)
        return spelled.number + spelled.suffix
    if not isinstance(figure.value, tuple):
        raise TypeError(f"{figure.path} is an interval figure holding {figure.value!r}, not a pair")
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
        if key.span is None:
            raise TypeError(f"{key.path} was found in the text with no span")
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
    allowed = (
        _allowed_names(report_input) | set(PERCENTILE_NAMES) | set(_UNIT_PHRASES) | _input_unit_phrases(report_input)
    )
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
    numeric setting, each arm's depth, a criterion set late, a second contrast family naming no aspect, a monthly
    volume, a frozen example with the failed trials it was kept from, and the cases the two aspects share."""
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
                        {
                            "arm": "fast.v2",
                            "reading": "within_margin",
                            "change": -4.26,
                            "within_interval": [-7.1, -1.4],
                            "gain_interval": [-7.6, -0.9],
                            "gain_confidence": 0.975,
                            "cases": [300, 300],
                        }
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
                "fast.v2": {
                    "passed": 731,
                    "usable": 895,
                    "reply_cost_usd": 0.002427,
                    "unanswered": {
                        "candidate": {
                            "retried": 9,
                            "lost": 3,
                            "reasons": {"timeout": 2, "tool_choice_refused": 9},
                            "calls": 2685,
                            "share": 12 / 2697,
                        }
                    },
                },
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
            "examples": [{"trial": "T7", "case": "c-12", "aspect": "answers", "arm": "fast.v2", "outcome": "judged"}],
            "failed_trials": [{"aspect": "answers", "arm": "fast.v2", "failed": 61}],
            "case_overlap": [{"aspects": ["answers", "speed"], "shared": 300, "only_first": 0, "only_second": 0}],
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
    "usd_whole": "cost_per_month.base|usd_whole",
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
        "A key prints its own unit (`%`, `$`, `ms`, `s`, or per 1,000 of what the input's cost counts, such as "
        "`per 1,000 conversations`); never write a unit after it. A figure "
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
        "(`per 1,000` or `per 1,000,000` of what the input's costs count, such as `per 1,000 conversations`, and, "
        "for the scoreboard's money per reply (`reply_cost_usd`), "
        f"{', '.join(f'`{phrase}`' for phrase in _UNIT_PHRASES)}; each whole) are written as they are; any other "
        "number is a key."
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
        "the campaign, its arms, aspects and settings, how deeply each arm ran, and whether two aspects ran the same "
        "cases",
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
                "overlap",
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
        frozenset(
            {"measure", "judged", "judge", "pass_rate", "scoreboard", "reasoning_ratio", "pooled", "overall_interval"}
        ),
    ),
    (
        "slices",
        "figures within one task, stratum or target-matrix cell, the share of a loss each carries, and where a "
        "classifier's predictions of each label went",
        frozenset({"task", "strata", "cells", "confusion"}),
    ),
    (
        "gates",
        "declared criteria, floors and bars, and the weakest margin",
        frozenset({"criteria", "floor", "bar", "weakest"}),
    ),
    (
        "cost",
        "cost per 1,000 of what the cost counts and per month, cost uplift, a cost some trials did not report, the calls "
        "that went unanswered beside it, and a move between options priced per point",
        frozenset({"cost_per_1k", "cost", "price", "cost_per_month", "calls"}),
    ),
    (
        "separation",
        "an arm's spread on a figure, two arms' difference with its interval, and the cases that would settle a "
        "reading against a line or two arms apart",
        frozenset({"spread", "compare", "settle"}),
    ),
    (
        "assumed",
        "a spread, comparison or settle key read as if each slice whose figure states no count held an assumed count: "
        "a what-if the key names",
        frozenset({"assumed"}),
    ),
    (
        "levers",
        "the effect of one setting moved between arms that differ in it alone, and how it changes with another setting",
        frozenset({"lever"}),
    ),
    (
        "counts",
        "how many of a thing a sentence counts: arms, slices, gates, the options passing each, the places one misses, "
        "the examples kept and the failures they were kept from",
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
    "FILTER_SPELLINGS",
    "PERCENTILE_NAMES",
    "WEAKEST_PREFIX",
    "Figure",
    "FigureKey",
    "PriceFrontier",
    "KeySubjects",
    "NamedKind",
    "arm_settings",
    "assumable",
    "criterion_wheres",
    "POOL_WEIGHT_WORDS",
    "SLICE_COST_WORDS",
    "SliceKind",
    "figure_key_grammar",
    "figure_key_index",
    "figure_filters",
    "figure_key_pages",
    "cost_basis",
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
    "on_lead_aspect",
    "pairing_contrast",
    "passed_aspects",
    "correction_words",
]
