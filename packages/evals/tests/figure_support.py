"""Reading a resolved figure's number in a test, its kind checked first."""

from __future__ import annotations

from threetears.evals.analysis.analyst.cited import CitedFigure
from threetears.evals.analysis.report.keys.figure import Figure


def figure_number(figure: Figure | CitedFigure) -> float:
    """A figure's number, resolved or cited; a figure holding anything else (an interval, a word) fails here, naming it.

    :param figure: The resolved figure.
    :ptype figure: Figure | CitedFigure
    :return: Its value, as a float.
    :rtype: float
    """
    assert isinstance(figure.value, int | float), figure
    return float(figure.value)
