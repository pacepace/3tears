"""The one rule for where an interval's unit is written (number rule 3), which the page's forms, its tables and its
charts' values tables all join an interval by.

An interval beside its figure carries no unit: the figure does (``85% (80 to 89)``). An interval standing alone writes
its unit once, after its high end (``80 to 89%``, ``-13 to -2 pts``). Money is the one exception: ``$`` is a prefix,
so each end carries it wherever the interval stands (``$20 ($16 to $25)``, ``$16 to $25``), since ``$16 to 25`` would
read as a bare 25.

It sits below both :mod:`~threetears.evals.analysis.report.forms` and
:mod:`~threetears.evals.analysis.report.charts`, which import it and neither of which it imports. It joins texts
already decided; it spells no number.
"""

from __future__ import annotations

from threetears.evals.analysis.quantities import UNSPACED_UNITS, is_money

__all__ = ["ends_beside", "interval_alone", "interval_beside", "unit_after", "unit_spaced", "unit_written"]


def unit_written(unit: str) -> str:
    """The unit a figure or an interval standing alone writes after it: its own, and none for money, whose ``$`` each
    figure carries already.

    :param unit: The unit; empty for none.
    :ptype unit: str
    :return: The unit to write after a figure; empty for none.
    :rtype: str
    """
    return "" if is_money(unit) else unit


def unit_spaced(digits: str, unit: str) -> str:
    """Digits and their unit, spaced as the package writes the unit (``85%``, ``12 pts``), the unit written as given;
    the digits alone with no unit. :func:`unit_after` is the guarded form, which writes no unit twice and none on money.

    :param digits: The figure's digits.
    :ptype digits: str
    :param unit: The unit; empty for none.
    :ptype unit: str
    :return: The digits with their unit.
    :rtype: str
    """
    if not unit:
        return digits
    return f"{digits}{unit}" if unit in UNSPACED_UNITS else f"{digits} {unit}"


def unit_after(text: str, unit: str) -> str:
    """A figure's text with its unit after it, spaced as the package writes the unit, where it does not already end in
    it; as it is with no unit, for money (:func:`unit_written`), or for a cell that holds no figure.

    :param text: The figure's text.
    :ptype text: str
    :param unit: Its unit; empty for none.
    :ptype unit: str
    :return: The text with its unit.
    :rtype: str
    """
    unit = unit_written(unit)
    if not unit or not text or text.endswith(unit) or not any(char.isdigit() for char in text):
        return text
    return unit_spaced(text, unit)


def interval_alone(low: str, high: str, unit: str) -> str:
    """An interval standing alone, ``low to high``, its unit written once after its high end (``80 to 89%``); money's
    ends each with their ``$`` (``$16 to $25``).

    :param low: The low end's text, without the unit (money's with its ``$``).
    :ptype low: str
    :param high: The high end's text, likewise.
    :ptype high: str
    :param unit: The interval's unit; empty for none.
    :ptype unit: str
    :return: The interval.
    :rtype: str
    """
    return interval_beside(low, unit_after(high, unit))


def interval_beside(low: str, high: str) -> str:
    """An interval written beside its figure, which carries the unit: ``low to high`` with no unit (``80 to 89`` of
    ``85% (80 to 89)``); money's ends each with their ``$``, which they carry already.

    :param low: The low end's text, without the unit (money's with its ``$``).
    :ptype low: str
    :param high: The high end's text, likewise.
    :ptype high: str
    :return: The interval.
    :rtype: str
    """
    return f"{low} to {high}"


def ends_beside(ends: tuple[str, str], unit: str) -> tuple[str, str]:
    """An interval's two end texts as written beside the figure that carries the unit: the unit taken off both
    (``80`` and ``89`` of ``80%`` and ``89%``); money's each keeping its ``$``. Ends that do not both carry the unit are
    as they were.

    :param ends: The two ends' texts.
    :ptype ends: tuple[str, str]
    :param unit: Their unit; empty for none.
    :ptype unit: str
    :return: The two ends beside a figure.
    :rtype: tuple[str, str]
    """
    unit = unit_written(unit)
    low, high = ends
    if unit and low.endswith(unit) and high.endswith(unit):
        return low.removesuffix(unit).rstrip(), high.removesuffix(unit).rstrip()
    return ends
