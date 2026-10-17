"""How a report states its quantities: the unit each is stated in, and how a value reads beside it.

The number rule, said once for every surface: that 100174 ms reads 100 s, that a share on 0–1 reads ``98.9%`` and a
change in it ``-1.1 pts``, that a missing value is a dash rather than a zero, that ``42 ms`` takes a space and
``42%`` does not. The report's tables, its charts' text and the run summary all state their figures through it
(:mod:`threetears.evals.analysis.report.figures` chooses each measure's one ruler with :func:`display_scale`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from threetears.evals.analysis.numbers import format_number, format_signed

#: Restatement ladder for durations, largest unit first: ``(unit, in terms of, factor)``.
#:
#: A quantity is stated in the unit a reader thinks in: it goes
#: in the largest unit that keeps two significant figures, so 100174 ms is written
#: 100 s and never both ways in one
#: report. Applied per CHART rather than per value — a column mixing 900 ms with
#: 1.2 s is two rulers read as one, which is the same rule's other half.
_DURATION_LADDER: tuple[tuple[str, str, float], ...] = (("ms", "s", 1000.0),)

#: Units written against the number with no space (``42%``), where every other unit takes one (``42 ms``).
#: Public so a figure stated in prose (:mod:`threetears.evals.analysis.prose_refs`) spaces its unit the same way.
UNSPACED_UNITS: tuple[str, ...] = ("%", "°")

#: The unit a share on 0–1 (a pass rate, an accuracy, a precision) is stated in, and the unit a CHANGE in one is
#: stated in. A rate reads as a percent; the difference of two percents is in points, never a percent of a percent:
#: ``98.9%`` against ``100%`` is ``-1.1 pts``, and a reader who saw ``-1.1%`` would take it for a relative change.
PERCENT = "%"
POINTS = "pts"

#: The units read to one decimal rather than four significant figures. A rate is estimated on tens of cases, so
#: ``98.89%`` states a figure the run never measured; ``98.9%`` is what the cases support, and its interval and its
#: spread read on the same scale beside it.
_ONE_DECIMAL_UNITS: frozenset[str] = frozenset({PERCENT, POINTS})

#: Money's unit as a measure declares it. A figure in it is written with the currency sign on the number
#: (``$0.0031``), as the scoreboard writes its cost: one money format on every surface of a page.
MONEY = "usd"


def display_scale(values: Sequence[float], unit: str | None, *, share: bool = False) -> tuple[float, str]:
    """Choose the unit a chart's or a table column's quantity is stated in, and the factor to reach it.

    Public because a table column follows the same rule as a chart axis: the decision surface's
    table (:mod:`threetears.evals.analysis.report.figures`) chooses each measure's unit here, once,
    for every surface that renders it.

    :param values: Every value the chart will draw in this unit.
    :ptype values: Sequence[float]
    :param unit: The payload's declared unit, or ``None`` when it carried none.
    :ptype unit: str | None
    :param share: Whether a unitless quantity is a share on 0–1 — a rate — which is stated as a percent.
    :ptype share: bool
    :return: ``(factor, unit)`` — multiply each value by the factor and label the axis
        with the unit. ``(100.0, "%")`` for a share. ``(1.0, "")`` when no unit was declared: a quantity whose
        unit is unknown is drawn as it was given and labelled by its measure's
        name alone, because restating an unknown unit would be inventing one.
    :rtype: tuple[float, str]
    """
    if not unit:
        return (100.0, PERCENT) if share else (1.0, "")
    largest = max((abs(value) for value in values if math.isfinite(value)), default=0.0)
    for source, target, factor in _DURATION_LADDER:
        if unit == source and largest >= factor:
            return 1.0 / factor, target
    return 1.0, unit


def delta_unit(unit: str) -> str:
    """The unit a change in a quantity is stated in: points for a percent, the quantity's own unit otherwise."""
    return POINTS if unit == PERCENT else unit


def is_money(unit: str) -> bool:
    """Whether ``unit`` is money, which is written with its currency sign rather than its name."""
    return unit.casefold() == MONEY


def unit_label(unit: str) -> str:
    """A unit as a header names it beside a column of bare numbers: money by its sign, any other as itself."""
    return "$" if is_money(unit) else unit


def number_in(value: float | None, unit: str) -> str:
    """A number spelled as it reads in ``unit`` — at the unit's precision — the unit itself left for a header.

    A percent or a point is read to one decimal; every other unit by the one number rule. The unit is not
    written: a column whose header names it, and an interval's second end, state the number alone.
    """
    return format_number(_at_precision(value, unit))


def render_cell(value: Any) -> str:
    """Render one values-table cell, honestly — an absent value is never a zero."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float):
        return format_number(value)
    return str(value)


def with_unit(value: float | None, unit: str) -> str:
    """Join a number to its unit, spaced the way the unit is actually written.

    ``42 ms`` takes a space and ``42%`` does not; getting it wrong is small but it
    is the kind of small that makes a generated report read as machine output.
    """
    return _joined(number_in(value, unit), value, unit)


def signed_with_unit(value: float | None, unit: str) -> str:
    """A delta in its own unit, carrying its sign — the direction is half the fact.

    Signed by :func:`~threetears.evals.analysis.numbers.format_signed`, the one signed spelling, so a change below
    :data:`~threetears.evals.analysis.numbers.WRITTEN_FROM` reads as its bound (``< 0.000001 ms``), which carries its
    direction already, and never as a signed bound (``+< 0.000001 ms``).
    """
    return _joined(format_signed(_at_precision(value, unit)), value, unit)


def _at_precision(value: float | None, unit: str) -> float | None:
    """``value`` rounded to the one decimal a percent or a point is read to; any other unit's value untouched.

    A non-zero value under a twentieth of a point is left to the number rule rather than rounded: ``0 pts`` says
    nothing happened, and the rule spells what did (``0.04 pts``, ``< 0.000001%``).
    """
    if unit not in _ONE_DECIMAL_UNITS or value is None or not math.isfinite(value):
        return value
    rounded = round(value, 1)
    return value if rounded == 0 and value != 0 else rounded


#: A spelled number split at its sign, so the currency sign goes on the number: after the sign, inside a bound.
_DOLLARS = re.compile(r"^(?P<sign>[<>]?\s?[+\-−]?)(?P<number>.*)$")


def _joined(text: str, value: float | None, unit: str) -> str:
    """A number's spelled ``text`` joined to its unit; an absent or non-finite value (a dash) states none."""
    if not unit or value is None or not math.isfinite(value):
        return text
    if is_money(unit) and (money := _DOLLARS.match(text)) is not None:
        return f"{money['sign']}${money['number']}"
    return f"{text}{unit}" if unit in UNSPACED_UNITS else f"{text} {unit}"


__all__ = [
    "MONEY",
    "PERCENT",
    "POINTS",
    "UNSPACED_UNITS",
    "delta_unit",
    "display_scale",
    "is_money",
    "number_in",
    "render_cell",
    "signed_with_unit",
    "unit_label",
    "with_unit",
]
