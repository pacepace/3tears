"""How a report states a figure: each measure's one ruler, and the words a table uses where a figure is absent.

A measure appears in several tables of one report — its contrasts and guardrails against the control, its rows by
stratum, a finding's evidence. Each table choosing its own unit over its own figures is how one latency read
``1.42 s`` in one table and ``1420`` in the next: the same figure on two rulers, one of them unnamed. So the unit is
chosen once per measure (:func:`measure_rulers`) and every table restates onto it.
"""

from __future__ import annotations

from typing import NamedTuple

from threetears.evals.analysis.quantities import display_scale
from threetears.evals.kernel.surface import DecisionSurface

#: What a cell where no result took a turn reads in a cost or latency column. Not a blank: a blank reads as "not
#: measured", and this arm was measured — every result failed, which its rates count. Never a number: a refused
#: call's round trip and its empty spend are what read as a fast, free arm.
NO_SUCCESSFUL_RESULTS = "no successful results"

#: What marks the control's row: it is the reference every other row is read against, never the pick.
REFERENCE_MARK = "(control: the reference)"

#: The row order, stated on a table laid out by :func:`~threetears.evals.analysis.arms.surface_order`, so no reader
#: takes the top row for a recommendation.
SURFACE_ORDER = (
    "The control first, as the reference every other arm is read against; then the other arms in alphabetical "
    "order of their names, each arm's rigs by id. Row order is not a ranking."
)

#: :data:`SURFACE_ORDER` for a table with no control row, which therefore has no reference row to lead with.
SURFACE_ORDER_NO_CONTROL = (
    "No row is the control, so none is a reference; the arms are in alphabetical order of their names, each "
    "arm's rigs by id. Row order is not a ranking."
)


def surface_order_sentence(*, has_control_row: bool) -> str:
    """The row-order sentence for a table laid out by :func:`~threetears.evals.analysis.arms.surface_order`.

    Args:
        has_control_row: Whether the table holds a row of the declared control.

    Returns:
        :data:`SURFACE_ORDER` or :data:`SURFACE_ORDER_NO_CONTROL`.
    """
    return SURFACE_ORDER if has_control_row else SURFACE_ORDER_NO_CONTROL


class MeasureRuler(NamedTuple):
    """The one unit a measure's figures are stated in across a report, and the factor that restates a stored value.

    Attributes:
        factor: Multiply a value in the measure's own unit by this to state it in :attr:`unit`.
        unit: The unit stated, after restatement (``s`` for a latency stored in ``ms`` that reaches a second);
            empty when the measure declares none, since restating an unknown unit would be inventing one.
    """

    factor: float
    unit: str


#: The ruler of a measure the surface knows no unit for: its values as stored, with no unit named.
UNITLESS = MeasureRuler(1.0, "")


def measure_rulers(surface: DecisionSurface) -> dict[str, MeasureRuler]:
    """Each measure's one ruler for a whole report, chosen over every figure the decision surface states of it.

    Chosen over every cell's mean of the measure and every threshold and verdict value of a bar on it, by
    :func:`~threetears.evals.analysis.quantities.display_scale`'s rule.

    Args:
        surface: The decision surface.

    Returns:
        Every measure the surface describes, by name, to its ruler; a measure absent here has none (:data:`UNITLESS`).
    """
    values: dict[str, list[float]] = {name: [] for name in surface.measures}
    for cell in surface.cells:
        for summary in cell.measures.measures:
            if summary.name in values and summary.mean is not None:
                values[summary.name].append(summary.mean)
    for bar in surface.bars:
        if bar.measure_id in values:
            values[bar.measure_id] += [bar.threshold] + [v.value for v in bar.verdicts if v.value is not None]
    return {name: MeasureRuler(*display_scale(values[name], facts.unit)) for name, facts in surface.measures.items()}


__all__ = [
    "NO_SUCCESSFUL_RESULTS",
    "REFERENCE_MARK",
    "SURFACE_ORDER",
    "SURFACE_ORDER_NO_CONTROL",
    "UNITLESS",
    "MeasureRuler",
    "measure_rulers",
    "surface_order_sentence",
]
