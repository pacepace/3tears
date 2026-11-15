"""The number policy's primitives: the decimals a measure reads at, read from its uncertainty; the one form an
interval prints in; and the fixed forms of the figures that are no measurement.
"""

from __future__ import annotations

import decimal
import math
from collections.abc import Collection, Iterable, Sequence

from threetears.evals.analysis.quantities import PERCENT
from threetears.evals.analysis.report.unit_once import interval_alone, unit_spaced
from threetears.evals.kernel.numbers import (
    WRITTEN_FROM,
    format_number,
    round_half_even,
    set_places,
    spell_at,
    spell_effect,
    spell_p,
)


#: What an absent figure reads as.
_ABSENT_TEXT = format_number(None)


# === The number policy's primitives ===================================================================================
#
# The number policy in parts small enough to hold each to its own cases: the decimals a measure reads at, read from its
# uncertainty; the one form an interval prints in; and the fixed forms of the figures that are no measurement. A page's
# forms decide every figure through them. Each is public so that a call from outside a page's forms (``forms`` and its
# parts) is a site the ratchet counts (``tests/test_one_owner_of_number_forms.py``): a surface reaches them only
# through a page's forms, asked by measure and role.

#: The most decimals a figure under the policy takes: :data:`~threetears.evals.kernel.numbers.WRITTEN_FROM`'s six, past
#: which a figure reads as the bound it is under.
_POLICY_MOST_PLACES = 6

#: The significant figures an interval's width reads to, which sets the decimals its value and its ends read at
#: (owner ruling, 2026-11-14): two, so a value and its interval print to the precision the interval justifies
#: (``-7.6`` with ``-10.8 to -3.4``, never ``-7.581`` with ``-10.79 to -3.381``, nor ``-8`` with ``-11 to -3``).
_WIDTH_FIGURES = 2

#: The most decimals a rate or a change in points reads at: a tenth of a point is finer than any interval a campaign's
#: cases support, short of an end the boundary exception keeps off a boundary.
_POINT_MOST_PLACES = 1

#: The boundary no interval's end prints onto, whatever its measure: an end that reads ``0`` says the interval touches
#: none.
_ZERO_BOUND = (0.0,)

#: A rate's floor and ceiling, in percent: a rate under 100 that reads ``100%`` says every case passed.
_RATE_BOUNDS = (0.0, 100.0)

#: A judge's preference share's boundaries, in percent: none, all, and the even point, where neither arm is preferred.
#: A share or an end of ``49.7`` that read ``50`` would hide that it falls short of even (the owner's "important
#: boundaries").
_PREFERENCE_BOUNDS = (*_RATE_BOUNDS, 50.0)


def policy_places(
    half_widths: Iterable[float | None], figures: Iterable[float | None] = (), *, capped: bool = False
) -> int:
    """The decimals every figure of one measure reads at, read from its uncertainty: those that print the narrowest of
    its headline intervals (each arm's level, each comparison) to two significant figures of its width, twice its 95%
    half-width (owner ruling, 2026-11-14). The width is placed by what it rounds to, so a half-width of 0.0498 (a width
    of 0.0996, two figures: 0.10) reads at two decimals. A measure with no interval reads at two significant figures of
    its least figure but zero (:func:`~threetears.evals.kernel.numbers.set_places`). Never fewer than none, so a width
    of 74 ms reads whole rather than to the ten.

    :param half_widths: The half-widths, as printed (a rate's in points); ``None``, zero and non-finite ones say nothing
        of the precision.
    :ptype half_widths: Iterable[float | None]
    :param figures: The measure's figures, as printed, read only where no half-width is.
    :ptype figures: Iterable[float | None]
    :param capped: Whether the measure is a rate or a change in points, which reads to at most one decimal.
    :ptype capped: bool
    :return: The decimals.
    :rtype: int
    """
    widths = [2 * abs(width) for width in half_widths if width is not None and math.isfinite(width) and width != 0]
    if widths:
        exponent = int(f"{min(widths):.{_WIDTH_FIGURES - 1}e}".split("e")[1])
        places = _WIDTH_FIGURES - 1 - exponent
    else:
        places = set_places(figures)
    places = min(max(0, places), _POLICY_MOST_PLACES)
    return min(places, _POINT_MOST_PLACES) if capped else places


def policy_interval_places(low: float, high: float, *, capped: bool = False) -> int:
    """The decimals a value and its interval read at where no measure's set holds them (a lone figure: a glyph drawn
    with no table, a chart's own interval): two significant figures of this interval's own width, as
    :func:`policy_places` reads a measure's narrowest. An interval with no width reads its ends at two significant
    figures of the lesser.

    :param low: The low end, as printed.
    :ptype low: float
    :param high: The high end, as printed.
    :ptype high: float
    :param capped: Whether the figure is a rate or a change in points, which reads to at most one decimal.
    :ptype capped: bool
    :return: The decimals.
    :rtype: int
    """
    return policy_places([(high - low) / 2], [low, high], capped=capped)


def policy_lone_interval(
    low: float, high: float, *, rate: bool = False, change: bool = False, value: float | None = None
) -> tuple[str, str]:
    """A figure and its interval where no measure's set holds them (a chart a host built with no decided texts, a glyph
    drawn with no table), read at one precision, two significant figures of the interval's own width
    (:func:`policy_interval_places`): ``-7.6`` with ``-10.8 to -3.4``, never ``-7.581`` with ``-10.79 to -3.381``. The
    one function such a figure and its interval go through, so the two never read at two precisions.

    :param low: The low end, as printed.
    :ptype low: float
    :param high: The high end, as printed.
    :ptype high: float
    :param rate: Whether the figure is a percent or a change in points, which reads to at most one decimal and, as a
        level, is kept off 0% and 100%.
    :ptype rate: bool
    :param change: Whether the figure is a change, its value and ends signed.
    :ptype change: bool
    :param value: The figure the interval is on; None for an interval alone.
    :ptype value: float | None
    :return: The figure's digits (empty for none), and the interval's text, ``low to high``, with no unit.
    :rtype: tuple[str, str]
    """
    places = policy_interval_places(low, high, capped=rate)
    bounds = _RATE_BOUNDS if rate and not change else ()
    figure = "" if value is None else policy_figure(value, places, bounds=bounds, signed=change)
    return figure, policy_interval(low, high, places, change=change, bounds=bounds)[2]


def policy_interval_set(
    rows: Sequence[tuple[float | None, float, float]], *, rate: bool = False
) -> list[tuple[str, str]]:
    """Figures of one measure and their intervals where no measure's set holds them but one column does (a chart a
    host built with no decided texts, read as its table): every row at one precision, two significant figures of the
    narrowest interval's width (:func:`policy_places`), as a measure reads across its campaign (owner ruling,
    2026-11-14), never each row at its own interval's (``12.35`` above ``46``).

    :param rows: Each row's figure (None for an interval alone) and its interval's ends, as printed.
    :ptype rows: Sequence[tuple[float | None, float, float]]
    :param rate: Whether the figures are percents or changes in points, which read to at most one decimal and, as a
        level, are kept off 0% and 100%.
    :ptype rate: bool
    :return: Each row's figure's digits (empty for none) and its interval's text, ``low to high``, with no unit.
    :rtype: list[tuple[str, str]]
    """
    places = policy_places(
        [(high - low) / 2 for _, low, high in rows], [end for _, low, high in rows for end in (low, high)], capped=rate
    )
    bounds = _RATE_BOUNDS if rate else ()
    return [
        (
            "" if value is None else policy_figure(value, places, bounds=bounds),
            policy_interval(low, high, places, change=False, bounds=bounds)[2],
        )
        for value, low, high in rows
    ]


def policy_level(percent: float) -> str:
    """An interval's confidence level as text, in percent, rounded as a value is by its interval
    (:func:`policy_places`): its decimals are those that print its distance from 100%, the chance its interval
    misses, to two significant figures, trailing zeros dropped, since a level is no member of a set (``95%``,
    ``97.5%``, ``99.69%`` for 99.6875%, ``98.3%`` for 98.33%). A chance under 50% (a miss rate, ``2.5%``) is that
    distance itself, so it reads to two significant figures of its own. Owner ruling, 2026-11-14: a level printed as
    text rounds, so two levels that differ past those figures print alike; the figure keeps the level exact
    (:attr:`~threetears.evals.analysis.report.keys.figure.Figure.level`).

    :param percent: The level, in percent (``99.6875``).
    :ptype percent: float
    :return: Its text with its ``%``; an em dash for a non-finite one.
    :rtype: str
    """
    if not math.isfinite(percent):
        return _ABSENT_TEXT
    miss = min(100.0 - percent, percent)
    places = set_places([miss], figures=_WIDTH_FIGURES) if abs(miss) >= WRITTEN_FROM else 0
    digits = round_half_even(percent, places)
    if "." in digits:
        digits = digits.rstrip("0").rstrip(".")
    return digits + PERCENT


def policy_bounded_places(value: float, places: int, *, bounds: Collection[float]) -> int:
    """The decimals one figure reads at: its measure's, or just enough more that a figure off a boundary never prints
    on it. The exception holds for a rate or a change in points, whose cap of one decimal would otherwise print
    ``-0.03`` as ``-0.0`` and ``99.7%`` as ``100%``, and for an interval's ends; any other figure passes no bounds and
    reads at its measure's decimals, as rule 2 sets them. A figure on a boundary, or within float residue of one (under
    :data:`~threetears.evals.kernel.numbers.WRITTEN_FROM` from it), reads at its measure's.

    :param value: The figure, as printed.
    :ptype value: float
    :param places: Its measure's decimals (:func:`policy_places`).
    :ptype places: int
    :param bounds: The boundaries it may not print onto: zero for a change in points, a rate's floor and ceiling for
        its level; none for a figure the exception does not cover.
    :ptype bounds: Collection[float]
    :return: The decimals, at most six.
    :rtype: int
    """
    at = max(0, places)
    if not math.isfinite(value) or _on_a_bound(value, bounds):
        return at
    while at < _POLICY_MOST_PLACES and float(round_half_even(value, at)) in bounds:
        at += 1
    return at


def policy_figure(value: float | None, places: int, *, bounds: Collection[float], signed: bool = False) -> str:
    """One figure's digits, rounded half to even at its measure's decimals, kept off its boundaries
    (:func:`policy_bounded_places`). An exact zero, or float residue of one, reads whole, ``0``, with no sign: it
    states no precision.

    :param value: The figure, as printed.
    :ptype value: float | None
    :param places: Its measure's decimals.
    :ptype places: int
    :param bounds: The boundaries it may not print onto; none for a figure the boundary exception does not cover (an
        uncapped measure's level or change: ``+0.4 ms`` on a whole-millisecond measure reads ``0``).
    :ptype bounds: Collection[float]
    :param signed: Whether a positive figure carries its ``+`` (a change).
    :ptype signed: bool
    :return: Its digits; an em dash for an absent or non-finite one.
    :rtype: str
    """
    if value is None or not math.isfinite(value):
        return _ABSENT_TEXT
    if abs(value) < WRITTEN_FROM:
        return "0"
    return spell_at(value, policy_bounded_places(value, places, bounds=bounds), signed=signed)


def policy_interval(
    low: float, high: float, places: int, *, change: bool, bounds: Collection[float], unit: str = ""
) -> tuple[str, str, str]:
    """An interval in its one form, ``low to high``, at its figure's decimals. A change's ends carry their signs
    (``-13 to +3``), a level's do not. The unit is written once, after the high end (``73 to 92%``, ``-13 to -2 pts``),
    and not at all where a head names it (an empty ``unit``). Each end is kept off zero and off ``bounds`` on its own
    (:func:`policy_bounded_places`: ``-15.1 to -0.03``), whatever its measure: an end that reads ``0`` claims the
    interval touches none. Two ends that differ but would print alike take more decimals together, one at a time until
    they print apart, in this interval alone. An interval with an absent end has no ends to state, so it reads as one
    absent figure, an em dash with no unit.

    :param low: The low end, as printed.
    :ptype low: float
    :param high: The high end, as printed.
    :ptype high: float
    :param places: The figure's measure's decimals.
    :ptype places: int
    :param change: Whether the interval is a change's, whose ends carry their signs.
    :ptype change: bool
    :param bounds: The boundaries beside zero neither end may print onto: a rate level's floor and ceiling; none for
        any other interval. Required, so a rate's interval never drops its ceiling by default.
    :ptype bounds: Collection[float]
    :param unit: The unit written once after it; empty under a head that names it.
    :ptype unit: str
    :return: The low end's digits, the high end's digits, and the interval's text.
    :rtype: tuple[str, str, str]
    """
    if not (math.isfinite(low) and math.isfinite(high)):
        return _ABSENT_TEXT, _ABSENT_TEXT, _ABSENT_TEXT
    ends = (low, high)
    kept_off = (*_ZERO_BOUND, *bounds)
    texts = [policy_figure(end, places, signed=change, bounds=kept_off) for end in ends]
    finer = max(policy_bounded_places(end, places, bounds=kept_off) for end in ends)
    while abs(high - low) >= WRITTEN_FROM and texts[0] == texts[1] and finer < _POLICY_MOST_PLACES:
        finer += 1
        texts = [policy_figure(end, finer, signed=change, bounds=kept_off) for end in ends]
    return texts[0], texts[1], interval_alone(texts[0], texts[1], unit)


def _on_a_bound(value: float, bounds: Collection[float]) -> bool:
    """Whether a figure sits on one of its boundaries, or within float residue of one (a rate of
    ``99.99999999999999``), which no digits should tell from it."""
    return any(abs(value - bound) < WRITTEN_FROM for bound in bounds)


def policy_percent(value: float | None, places: int) -> str:
    """A percentage at ``places`` decimals, rounded half to even as every other figure is (``62.5`` reads ``62%``), its
    sign kept; an exact zero reads ``0%``, which states no precision.

    :param value: The percentage, in percent.
    :ptype value: float | None
    :param places: The decimals.
    :ptype places: int
    :return: ``63%``, ``62.9%``, ``0%``; an em dash for an absent or non-finite one.
    :rtype: str
    """
    if value is None or not math.isfinite(value):
        return _ABSENT_TEXT
    if value == 0:
        return f"0{PERCENT}"
    return round_half_even(value, places) + PERCENT


def policy_rate(value: float | None, places: int) -> str:
    """A rate's level, always with its ``%`` (a lost-call rate of none reads ``0%``, never a bare ``0``), at its
    measure's decimals and kept off its floor and its ceiling: ``99.7%`` at a measure's whole percents, never ``100%``,
    and ``0.3%``, never ``0%``. A rate on its floor or ceiling, or within float residue of one, reads at its
    measure's decimals (``100%``); one under its floor, which no count gives, reads as given, its sign kept.

    :param value: The rate, in percent.
    :ptype value: float | None
    :param places: Its measure's decimals (:func:`policy_places`, capped).
    :ptype places: int
    :return: Its text.
    :rtype: str
    """
    if value is None or not math.isfinite(value):
        return _ABSENT_TEXT
    return policy_percent(value, policy_bounded_places(value, places, bounds=_RATE_BOUNDS))


def policy_p(p: float | None) -> str:
    """A p-value as APA writes it (:func:`~threetears.evals.kernel.numbers.spell_p`): ``< 0.001`` under a thousandth,
    three decimals below 0.1, two from it up. It is no measurement, so no measure's decimals reach it.

    :param p: The p-value.
    :ptype p: float | None
    :return: ``< 0.001``, ``0.003``, ``0.50``, ``1.00``.
    :rtype: str
    """
    return spell_p(p)


def policy_effect(value: float | None) -> str:
    """An effect size or a correlation at two decimals, its sign kept
    (:func:`~threetears.evals.kernel.numbers.spell_effect`): ``0.08``, ``-0.19``.

    :param value: The effect.
    :ptype value: float | None
    :return: Its text.
    :rtype: str
    """
    return spell_effect(value)


def policy_declared(value: float | None, unit: str = "") -> str:
    """A value the input declares (a margin, a floor, a setting) exactly as it was declared, in its unit: the float's
    own shortest digits, never rounded and never in an exponent (``1234.5``, ``0.1``, ``0.00001``), and a whole one
    whole (``-15 pts``, never ``-15.0 pts``). It was chosen, not measured, so no measure's decimals reach it.

    :param value: The value, in its unit.
    :ptype value: float | None
    :param unit: Its unit; empty for none.
    :ptype unit: str
    :return: Its text; an em dash, without the unit, for an absent or non-finite one.
    :rtype: str
    """
    if value is None or not math.isfinite(value):
        return _ABSENT_TEXT
    number = float(value)
    digits = str(int(number)) if number.is_integer() else format(decimal.Decimal(repr(number)), "f")
    return unit_spaced(digits, unit)


# === End of the number policy's primitives ============================================================================


__all__: list[str] = []
