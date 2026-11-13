"""A figure key and the figure it resolves to: the key as prose writes it, what a figure is, the scales and filters
it reads at, and the words and patterns a key is found and spelled with.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_args

from threetears.evals.analysis.report.figures import PER_MILLION_REPLIES, PER_THOUSAND_LABEL, PER_THOUSAND_REPLIES
from threetears.evals.analysis.report.words import TOLD_APART

if TYPE_CHECKING:
    # A page's forms decide over the figures this module reads, so the module that decides them reads this one: every
    # caller hands its page's forms in, and the type is named for the checker alone.
    from threetears.evals.analysis.report.forms import Role


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
Filter = Literal["pct", "pts", "abs", "signed", "usd", "int", "pct10", "interval", "usd_per_1k", "usd_whole", "words"]
_FILTERS: tuple[str, ...] = get_args(Filter)
_SPELLINGS: frozenset[str] = frozenset({"pct", "pts", "usd", "usd_per_1k", "int", "pct10", "usd_whole", "words"})
#: The filters that choose a key's spelling, at most one to a key, in the grammar's order.
FILTER_SPELLINGS: tuple[str, ...] = tuple(name for name in _FILTERS if name in _SPELLINGS)

#: How each filter reads, as the grammar teaches it.
_FILTER_MEANS: dict[str, str] = {
    "pct": "a rate as a percent",
    "pts": "a change in points, signed, one decimal at most",
    "abs": "the magnitude, without its sign",
    "signed": "a change with its sign, `+` on a rise: a criterion's gap or the weakest margin, which reads without "
    "one as a distance the prose's own words place (`inside the margin`), in any unit",
    "usd": "dollars, in one form for the key's whole family: each amount to two significant figures, cents at least, "
    "and the family moved per 1,000 more of what it counts where any of its amounts is under a cent",
    "int": "the nearest whole number, its thousands separated; a multiple keeps its unit (`21 times`), every other "
    "unit is left for the prose to write",
    "pct10": "a rate rounded to the nearest 10%",
    "interval": "an interval's two ends, joined by 'to'",
    "usd_per_1k": "dollars per 1,000 of what the cost counts, with the words (`per 1,000 conversations` where the "
    "measure's label counts conversations); per 1,000,000 where the family's least cost is under a cent per 1,000",
    "usd_whole": "dollars to the whole dollar, thousands grouped, for a monthly or other large amount whose cents are "
    "noise; refused under $10, which it would round away",
    "words": "a count spelled as a word up to twenty, in grouped digits above",
}

#: The least amount :data:`Filter` ``usd_whole`` reads: under it, whole dollars round away too much of the figure.
_WHOLE_DOLLARS_FROM = 10.0

#: The phrases a rendered figure is joined with, one template each, so a translation replaces them here; an interval's
#: two ends are joined by :func:`~threetears.evals.analysis.report.unit_once.interval_beside`, and the phrase a cost's
#: count reads in is :func:`~threetears.evals.analysis.report.cost_unit.per_count`'s.
ARM_TEMPLATE = "{name} ({tag})"


def written_arm(name: str, tag: str) -> str:
    """An arm as prose writes it: its name, then its letter tag (:data:`ARM_TEMPLATE`, ``v2 (A)``). A name told apart
    by its tag already (``Extractor instructions (B)``,
    :data:`~threetears.evals.analysis.report.words.TOLD_APART`) is not tagged twice.

    :param name: The arm's name on the page.
    :ptype name: str
    :param tag: Its letter tag.
    :ptype tag: str
    :return: The arm, written.
    :rtype: str
    """
    told = TOLD_APART.format(name="", tag=tag)
    return name if name.endswith(told) else ARM_TEMPLATE.format(name=name, tag=tag)


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

#: A key, as report prose holds it: one pair of single braces around anything but braces.
_KEY = re.compile(r"\{([^{}]*)\}")

#: The arm form's prefix, ``{arm:<id>}``.
_ARM_PREFIX = "arm:"

#: A run of digits report prose typed, with any decimal or thousands separators inside it. ``\d`` is every Unicode
#: decimal digit (category Nd), so an Arabic-Indic or Devanagari figure is caught as surely as an ASCII one.
_DIGITS = re.compile(r"\d(?:[.,]?\d)*")

#: How many valid continuations a refusal lists before it stops counting.
_CONTINUATIONS_SHOWN = 24


@dataclass(frozen=True)
class FigureKey:
    """One key as report prose wrote it: its path, its filters, and where it sits in the text (None when parsed
    alone)."""

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
    decimals. :func:`resolve` fills both; a figure made without them is spelled as a set of its own value. ``total``
    marks money that counts nothing (a month's cost, a run's), which reads to cents whatever its size, where money per
    something reads in whole dollars from $10; ``reconciles`` marks the page's cost per 1,000 where a monthly volume is
    declared, so its figures take a third where two would leave it off the month printed beside it
    (:class:`~threetears.evals.kernel.numbers.MoneyForm`). None of these takes part in a figure's equality: they say
    how it reads, not what it is. Nor does ``priced_in_part``: beside a cost of an
    arm whose cost counts only part of its calls, the words its pricing gives (``850 of 897 calls priced``), which
    :func:`format_figure` prints after the amount, so no key states a smaller total bare. Nor do ``statistic`` and
    ``set_of``: the statistic a plain figure's path names last (None where it names none), and the measure whose set
    the page's tables print the figure in, by key and kind (``judged`` for a 1–5 score), from which the page's forms
    read the decimals a key prints the figure at (:meth:`~threetears.evals.analysis.report.forms.PageForms.key_number`),
    so a key reads its figure as the table beside it does; None for a figure of neither, which keeps the number rule.
    Nor does ``measure``: the measure or judged dimension a figure is of where its path names it by another id (a
    criterion's figure, named by the criterion), so the figure reads at that measure's set as its table prints it; empty
    where the path names the measure itself, or names none. Nor does ``ledger_as``: where the page's tables print the
    figure under a measure its path does not name, or in a role its statistic does not say (an arm's cost per reply is
    the page's cost, a reading's change the pass rate's, a percentile a level), that measure, its kind and the role,
    which the page's ledger records the figure under in place of ``set_of``'s; None where ``set_of`` says it. It is the
    ledger's alone: the decimals a key prints at still read ``set_of``, so naming where a figure is held moves no digit.

    A figure said in words keeps its numbers as numbers, so the page's forms print each, as they print every other
    figure, and no number reads one way in a sentence and another beside it. ``said_between`` is the words a number
    figure is said between (``95% one-sided``, ``the 95% interval``); ``counts``, a text figure's
    counts by word, each printed after its word (``timeout 1,204, refused 3``), the figure's ``value`` holding the same
    listing, as a tool serves it.

    ``distance`` marks a change read as a distance from a line (a criterion's ``gap``, the weakest margin): it is a
    change (``change``), so the page's forms hold and decide it as its measure's change, but it prints no ``+``, since
    the prose's own words say which side of the line it is on ("inside the margin", "over the line"); a miss keeps its
    ``-``. The ``signed`` filter prints its ``+``, in any unit, and ``pts`` does in points.

    ``joint`` marks a level said in the package's own words as the guarantee a correcting family holds together: the
    figure's value is that joint level (one minus the family's alpha, ``95%``), and ``joint`` holds it with the
    family's size, which the words print where they hold ``{comparisons}`` (``the 95% interval, adjusted for
    16 comparisons``, owner ruling); None for every other figure. The per-interval level a correction sets
    (``99.69%``) is no such figure: its own keys state it as the input does.
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
    priced_in_part: str = field(default="", compare=False, repr=False)
    total: bool = field(default=False, compare=False, repr=False)
    reconciles: bool = field(default=False, compare=False, repr=False)
    statistic: str | None = field(default=None, compare=False, repr=False)
    set_of: tuple[str, str] | None = field(default=None, compare=False, repr=False)
    measure: str = field(default="", compare=False, repr=False)
    ledger_as: tuple[str, str, Role] | None = field(default=None, compare=False, repr=False)
    said_between: tuple[str, str] = ("", "")
    counts: tuple[tuple[str, int], ...] = ()
    distance: bool = field(default=False, compare=False, repr=False)
    joint: tuple[float, int] | None = field(default=None, compare=False, repr=False)


def _count(path: str, value: int) -> Figure:
    return Figure(path=path, kind="number", value=float(value), scale="count")


def _listed(options: Sequence[str]) -> str:
    shown = [f"`{option}`" for option in options[:_CONTINUATIONS_SHOWN]]
    more = len(options) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more > 0 else "")


#: How many of what it counts a figure on each dollar scale is per, as the one owner of money's scale reads it.
_SCALE_COUNTS: dict[str, int] = {"usd": 1, "usd_per_1k": 1000, "usd_per_1m": 1_000_000}

__all__ = [
    "FILTER_SPELLINGS",
    "PERCENTILE_NAMES",
    "Figure",
    "FigureKey",
    "written_arm",
]
