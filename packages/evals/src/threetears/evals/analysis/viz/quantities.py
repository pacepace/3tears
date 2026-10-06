"""How a chart states its quantities: the unit each is drawn in, how a value reads beside it, what an interval is.

Renderer-independent, and that is why it is its own module. These rules decide what a reader is
TOLD — that 100174 ms reads 100 s, that a missing value is a dash rather than a zero, that a band is a
95% interval over runs — and every surface states them the same way whatever draws the picture: the
chart intent (:mod:`threetears.evals.analysis.viz.intent`) is built from them, the values-as-drawn
tables carry their output, and the report's decision-surface table
(:mod:`threetears.evals.analysis.surface_table`) chooses its column units with the same ladder. A
renderer reads what they decided; it never decides it again.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.viz.payloads import ConfidenceInterval

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
    served table (``threetears.evals.analysis.surface_table``) chooses each column's unit here, once,
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


def axis_title(quantity: str, unit: str) -> str:
    """Name a quantity and, where one is known, the unit it is stated in."""
    return f"{quantity} ({unit})" if unit else quantity


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
    rendered = format_number(value)
    if not unit or value is None:
        return rendered
    return f"{rendered}{unit}" if unit in UNSPACED_UNITS else f"{rendered} {unit}"


def signed_with_unit(value: float | None, unit: str) -> str:
    """A delta in its own unit, carrying its sign — the direction is half the fact."""
    if value is None:
        return "—"
    return f"{'+' if value > 0 else ''}{with_unit(value, unit)}"


def relative_change_text(change: float | None) -> str:
    """A relative change — a signed FRACTION of the baseline — as the percent a reader is told.

    The one spelling for a unitless change from a baseline (``+72.7%`` for 0.727). A quantity already
    measured in percentage points is not this: it carries the unit ``%`` and reads through
    :func:`with_unit` (``-12%`` for -12).
    """
    if change is None or not math.isfinite(change):
        return "—"
    return f"{change:+.1%}"


def table_spellings(value: float, unit: str) -> frozenset[str]:
    """Every text a values table writes for a drawn ``value`` stated in ``unit``.

    The table's builders spell a drawn number through :func:`with_unit`, :func:`signed_with_unit` and,
    for a unitless relative change, :func:`relative_change_text`; this returns what each of them writes
    for ``value``, so a check that a cell states a mark's value formats the mark through the same
    functions and compares the text, rather than reading the cell back as a number. Which spellings are
    candidates is decided by ``unit``, never by a glyph in the cell: under ``%`` the value is already in
    percentage points, and only a unitless value may be a fraction spelled as a percent.

    Args:
        value: The drawn value, in the unit drawn.
        unit: The unit it is drawn in; empty when it has none.

    Returns:
        The texts that state ``value``.
    """
    spelled = {with_unit(value, unit), signed_with_unit(value, unit)}
    if not unit:
        spelled.add(relative_change_text(value))
    return frozenset(spelled)


def interval_disclosures(intervals: Iterable[ConfidenceInterval | None]) -> list[str]:
    """Qualify a chart's intervals: what they cover, and what they vary over.

    Both are needed to read a drawn band, and neither is recoverable from the
    picture — a wide band is a wide band whether it is a 95% interval over five
    runs or a 50% one over twelve cases. Two lines rather than one sentence,
    because they are two facts: a reader checking the coverage level should not
    have to find it inside a statement about the variability source.

    Args:
        intervals: The chart's intervals; ``None`` entries (a group with no
            interval) are ignored.

    Returns:
        The coverage line then the variability line, each only where it has
        something to say — empty when there is no interval to qualify.
    """
    present = [interval for interval in intervals if interval is not None]
    return [line for line in (_coverage_caption(present), _variability_caption(present)) if line]


def interval_statement(intervals: Iterable[ConfidenceInterval | None]) -> str:
    """The interval qualification as one string — what a chart intent says its intervals are.

    One string because it travels as one fact beside the chart (a renderer that writes a figure
    description puts it there); the lines :func:`interval_disclosures` returns are space-joined for it.

    Args:
        intervals: The chart's intervals; ``None`` entries are ignored.

    Returns:
        The sentences, space-joined, or ``""`` when there is no interval to qualify.
    """
    return " ".join(interval_disclosures(intervals))


def interval_sources(intervals: Iterable[ConfidenceInterval | None]) -> str:
    """What a chart's intervals vary over, as an interval encoding states it.

    Args:
        intervals: The chart's intervals; ``None`` entries are ignored.

    Returns:
        The variability sources, sorted and ``; ``-joined, or ``""`` when there is no interval.
    """
    return "; ".join(sorted({interval.variability for interval in intervals if interval is not None}))


def _coverage_caption(intervals: Sequence[ConfidenceInterval]) -> str:
    """Name the coverage a set of intervals was computed at, where they share one.

    A 50% band and a 95% band over the same values are different widths, so an
    interval drawn without its level lets the reader read coverage as spread. Every
    interval states its level (:class:`ConfidenceInterval` requires it).

    Args:
        intervals: The chart's intervals, already filtered of absences.

    Returns:
        The sentence, or ``""`` when there is no interval to qualify.
    """
    if not intervals:
        return ""
    stated = sorted({interval.level for interval in intervals})
    if len(stated) == 1:
        coverage = with_unit(stated[0] * 100, "%")
        return f"Interval is a {coverage} CI." if len(intervals) == 1 else f"Intervals are {coverage} CIs."
    # Naming one level would extend it to intervals that stated another, and the widths on this
    # shared axis are then read against each other as if they covered the same thing.
    joined = "; ".join(with_unit(level * 100, "%") for level in stated)
    return f"Widths are not comparable: the intervals are not all the same coverage level ({joined})."


def _variability_caption(intervals: Sequence[ConfidenceInterval]) -> str:
    """State what a set of intervals varies over.

    Uncertainty is drawn from values, never inferred, so every interval has to say what
    variability it captures — the same [0.77, 0.85] means different things spanning runs of one
    case and spanning cases within one run, so every interval states its source
    (:class:`ConfidenceInterval` requires it). Where the arms of one chart disagree
    about their source it says so rather than picking the first: differing
    variability is exactly the case where a reader must not assume the arms are
    comparable.

    Args:
        intervals: The chart's intervals, already filtered of absences.

    Returns:
        The sentence, or ``""`` when there is no interval to qualify.
    """
    if not intervals:
        return ""
    sources = sorted({interval.variability for interval in intervals})
    if len(sources) == 1:
        return f"Interval spans {sources[0]}." if len(intervals) == 1 else f"Intervals span {sources[0]}."
    joined = "; ".join(sources)
    return f"The intervals span different things ({joined}), so their widths are not comparable."


def strip_common_prefix(labels: Sequence[str]) -> dict[str, str]:
    """Map each label to what it is drawn as, stripping the prefix they all share.

    A ``/``-delimited prefix common to *every* series carries no information
    **within one chart**: if all three are ``anthropic/`` the word distinguishes
    nothing, and if they differ nothing is stripped. Per figure rather than from a
    global alias table, so there are no naming decisions to maintain and no way for
    a rename in one report to change what another draws. The full identity stays on
    the row, which is what the tooltip and the values table read.

    Two things are refused rather than allowed to happen quietly. **At least one
    segment always survives**, so a set like ``anthropic/`` and ``anthropic/claude``
    cannot strip one label to nothing. And **two labels may never become one** —
    unreachable as the arithmetic stands, since a suffix collision after a shared
    prefix means the labels were identical to begin with, but the whole value of
    stripping over aliasing is that it cannot merge two series, and a property that
    load-bearing is worth holding by construction rather than by argument.

    Args:
        labels: Every category the figure draws, in any order.

    Returns:
        Full label → drawn label, for each distinct label.
    """
    unique = list(dict.fromkeys(labels))
    segmented = [label.split("/") for label in unique]
    identity = {label: label for label in unique}
    # No categories at all is a real input — a comparison whose every row turns out
    # unplottable still compiles a frame — and it has no shared prefix rather than an
    # undefined one. Stated here because the arithmetic below takes a `min` over the
    # segments and would raise on the empty case instead.
    if not unique or any(len(parts) < 2 for parts in segmented):
        return identity

    shared = 0
    keep_one = min(len(parts) for parts in segmented) - 1
    while shared < keep_one and len({parts[shared] for parts in segmented}) == 1:
        shared += 1
    if not shared:
        return identity

    stripped = {label: "/".join(parts[shared:]) for label, parts in zip(unique, segmented, strict=True)}
    if any(not drawn for drawn in stripped.values()) or len(set(stripped.values())) < len(stripped):
        return identity
    return stripped


__all__ = [
    "UNSPACED_UNITS",
    "axis_title",
    "display_scale",
    "interval_disclosures",
    "interval_sources",
    "interval_statement",
    "relative_change_text",
    "render_cell",
    "signed_with_unit",
    "strip_common_prefix",
    "table_spellings",
    "with_unit",
]
