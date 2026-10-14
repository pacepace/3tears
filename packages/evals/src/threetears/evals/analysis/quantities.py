"""How a report states its quantities: the unit each is stated in, and how a value reads beside it.

The number rule, said once for every surface: that 100174 ms reads 100 s, that a missing value is a dash rather
than a zero, that ``42 ms`` takes a space and ``42%`` does not. The report's tables, its charts' text and the
run summary all state their figures through it (:mod:`threetears.evals.analysis.report.figures` chooses each
measure's one ruler with :func:`display_scale`).
"""

from __future__ import annotations

import math
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


def display_scale(values: Sequence[float], unit: str | None) -> tuple[float, str]:
    """Choose the unit a chart's or a table column's quantity is stated in, and the factor to reach it.

    Public because a table column follows the same rule as a chart axis: the decision surface's
    table (:mod:`threetears.evals.analysis.report.figures`) chooses each measure's unit here, once,
    for every surface that renders it.

    Args:
        values: Every value the chart will draw in this unit.
        unit: The payload's declared unit, or ``None`` when it carried none.

    Returns:
        ``(factor, unit)`` — multiply each value by the factor and label the axis
        with the unit. ``(1.0, "")`` when no unit was declared: a quantity whose
        unit is unknown is drawn as it was given and labelled by its measure's
        name alone, because restating an unknown unit would be inventing one.
    """
    if not unit:
        return 1.0, ""
    largest = max((abs(value) for value in values if math.isfinite(value)), default=0.0)
    for source, target, factor in _DURATION_LADDER:
        if unit == source and largest >= factor:
            return 1.0 / factor, target
    return 1.0, unit


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
    return _joined(format_number(value), value, unit)


def signed_with_unit(value: float | None, unit: str) -> str:
    """A delta in its own unit, carrying its sign — the direction is half the fact.

    Signed by :func:`~threetears.evals.analysis.numbers.format_signed`, the one signed spelling, so a change below
    :data:`~threetears.evals.analysis.numbers.WRITTEN_FROM` reads as its bound (``< 0.000001 ms``), which carries its
    direction already, and never as a signed bound (``+< 0.000001 ms``).
    """
    return _joined(format_signed(value), value, unit)


def _joined(text: str, value: float | None, unit: str) -> str:
    """A number's spelled ``text`` joined to its unit; an absent or non-finite value (a dash) states none."""
    if not unit or value is None or not math.isfinite(value):
        return text
    return f"{text}{unit}" if unit in UNSPACED_UNITS else f"{text} {unit}"


__all__ = [
    "UNSPACED_UNITS",
    "display_scale",
    "render_cell",
    "signed_with_unit",
    "with_unit",
]
