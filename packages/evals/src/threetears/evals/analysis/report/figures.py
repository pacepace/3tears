"""How a report states a figure: each measure's one ruler, and the words a table uses where a figure is absent.

A measure appears in several tables of one report — its contrasts and guardrails against the control, its rows by
stratum, a finding's evidence. Each table choosing its own unit over its own figures is how one latency read
``1.42 s`` in one table and ``1420`` in the next: the same figure on two rulers, one of them unnamed. So the unit is
chosen once per measure (:func:`measure_rulers`) and every table restates onto it.

Three rulers are chosen here rather than read off a unit. A share on 0–1 — a pass rate, an accuracy, a precision, a
pass/fail grader's score — is stated as a percent, its change in points (:data:`SHARE`): ``0.9889`` on 30 cases is
a precision the run never measured, and ``98.9%`` is what a reader compares at a glance. A cost per reply under
:data:`REPLY_COST_FLOOR` on every arm is stated per 1,000 replies, and the heading says so: ``$0.0000459`` beside
``$0.00007036`` is two figures nobody can compare, ``$0.0459`` beside ``$0.0704`` is one glance. A judged
dimension on a pass/fail scale is a share like any other.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

from threetears.evals.analysis.quantities import MONEY, PERCENT, display_scale, is_money
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.metrics import METRIC_DESCRIPTORS, MetricDescriptor, classifier_label_of, goal_check_of
from threetears.evals.kernel.surface import DecisionSurface, MeasureFacts

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

#: The cost per reply below which, on every arm, a report states cost per 1,000 replies instead.
REPLY_COST_FLOOR = 0.01

#: The words a heading adds when a per-reply cost is stated per 1,000 replies, and the label a column's header or a
#: reading's name carries for it. Replies, as the scoreboard counts them: one trial is one reply.
PER_THOUSAND_REPLIES = "per 1,000 replies"
PER_THOUSAND_LABEL = "($/1,000 replies)"


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
        unit: The unit stated, after restatement (``s`` for a latency stored in ``ms`` that reaches a second, ``%``
            for a share on 0–1); empty when the measure declares none, since restating an unknown unit would be
            inventing one.
        per: The words a heading adds when the factor restates a per-reply figure onto a larger count of replies
            (``per 1,000 replies``); empty when the figure is stated as measured.
    """

    factor: float
    unit: str
    per: str = ""

    def scaled(self, heading: str) -> str:
        """``heading`` with the scale a reader needs to read its figures, where the figures do not carry it.

        A cost stated per 1,000 replies says so beside its name, since ``$0.0459`` alone reads as a cost per reply;
        every other ruler's figures carry their own unit, so the heading stands alone.
        """
        return f"{heading} {PER_THOUSAND_LABEL}" if self.per else heading

    def labelled(self, heading: str) -> str:
        """``heading`` with its unit, for a row whose figures are stated as bare numbers beside it.

        A percent and money carry their unit on the figure (``98.9%``, ``$0.0459``), so only the scale is added;
        any other unit is named once in the heading (``Turn time (ms)``).
        """
        if self.unit == PERCENT or is_money(self.unit):
            return self.scaled(heading)
        return f"{heading} ({self.unit})" if self.unit else heading


#: The ruler of a measure the surface knows no unit for: its values as stored, with no unit named.
UNITLESS = MeasureRuler(1.0, "")

#: The ruler every share on 0–1 is stated on: a percent, its change in points.
SHARE = MeasureRuler(100.0, PERCENT)


def measure_rulers(
    surface: DecisionSurface, catalog: Mapping[str, MetricDescriptor] | None = None
) -> dict[str, MeasureRuler]:
    """Each measure's one ruler for a whole report, chosen over every figure the decision surface states of it.

    Chosen over every cell's mean of the measure and every threshold and verdict value of a bar on it, by
    :func:`~threetears.evals.analysis.quantities.display_scale`'s rule; a share on 0–1 onto :data:`SHARE`, and a
    cost per reply under :data:`REPLY_COST_FLOOR` on every arm onto its per-1,000 ruler.

    Which measures are shares is read off the bundle's catalog where the caller has it — a host's pass/fail grader
    declares its range there and nowhere on the surface, whose frozen facts carry no range — and off the engine's
    own descriptions otherwise: a core measure's declared range, a classifier's precision, recall and F1, a goal
    check's pass rate.

    Args:
        surface: The decision surface.
        catalog: The bundle's ``measure_catalog``, when the caller has the bundle; None reads the engine's
            descriptions alone.

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
    rulers = {}
    for name, facts in surface.measures.items():
        described = catalog.get(name) if catalog is not None else None
        factor, unit = display_scale(values[name], facts.unit, share=_is_share(name, facts, described))
        rulers[name] = _money_ruler(values[name]) if is_money(unit) else MeasureRuler(factor, unit)
    return rulers


def _is_share(name: str, facts: MeasureFacts, described: MetricDescriptor | None) -> bool:
    """Whether a measure is a share on 0–1: a rate, a 0–1 score, or a boolean's share held."""
    if facts.unit:
        return False
    descriptor = described if described is not None else METRIC_DESCRIPTORS.get(name)
    if descriptor is not None:
        return descriptor.data_type == "boolean" or descriptor.value_range == (0.0, 1.0)
    return classifier_label_of(name) is not None or goal_check_of(name) is not None


def _money_ruler(amounts: Iterable[float | None]) -> MeasureRuler:
    """Money's ruler over the per-reply amounts it will state: per 1,000 replies when every one is under the floor."""
    stated = [amount for amount in amounts if amount is not None]
    if stated and all(abs(amount) < REPLY_COST_FLOOR for amount in stated):
        return MeasureRuler(1000.0, MONEY, PER_THOUSAND_REPLIES)
    return MeasureRuler(1.0, MONEY)


def reply_cost_ruler(costs: Iterable[float | None]) -> MeasureRuler:
    """The ruler the scoreboard's cost per reply is stated on, by the one rule a cost measure's ruler follows.

    Args:
        costs: Each arm's cost per reply; None for an arm whose cost was not measured.

    Returns:
        Money per reply, or per 1,000 replies when every measured cost is under :data:`REPLY_COST_FLOOR`.
    """
    return _money_ruler(costs)


class Rulers(NamedTuple):
    """Every ruler one report states its readings on: its measures', its judged dimensions', and the pass rate's."""

    measures: Mapping[str, MeasureRuler]
    dimensions: Mapping[str, MeasureRuler]

    def of(self, name: str, reading: str) -> MeasureRuler:
        """The ruler of one reading: a measure's, a judged dimension's, or :data:`SHARE` for the trials' pass rate.

        Args:
            name: The measure's key, or the judged dimension's name.
            reading: ``measure``, ``judged`` or ``trial``.

        Returns:
            The ruler; :data:`UNITLESS` for a reading the report knows no unit for.
        """
        if reading == "trial":
            return SHARE
        if reading == "judged":
            return self.dimensions.get(name, UNITLESS)
        return self.measures.get(name, UNITLESS)


def report_rulers(surface: DecisionSurface, catalog: Mapping[str, MetricDescriptor] | None = None) -> Rulers:
    """Every ruler a report states its readings on, chosen once over the decision surface.

    A judged dimension's ruler is read off the surface's own facts for it: a dimension scored on a 0–1 scale is a
    share, and a 1–5 score is stated as scored.

    Args:
        surface: The decision surface.
        catalog: The bundle's ``measure_catalog``, when the caller has the bundle (see :func:`measure_rulers`).

    Returns:
        The rulers.
    """
    dimensions = {
        name: SHARE if facts.value_range == (0.0, 1.0) else UNITLESS for name, facts in surface.dimensions.items()
    }
    return Rulers(measure_rulers(surface, catalog), dimensions)


def analysis_rulers(analysis: EvalAnalysis) -> Rulers:
    """Every ruler a stored analysis's readings are stated on, over the surface and catalog frozen with it.

    Args:
        analysis: The analysis.

    Returns:
        The rulers, chosen as :func:`report_rulers` chooses them over the bundle the analysis was written over.
    """
    frozen: Mapping[str, Any] = analysis.bundle
    entries = frozen.get("measure_catalog") or {}
    catalog = {name: MetricDescriptor.model_validate(entry) for name, entry in entries.items()}
    return report_rulers(analysis.decision_surface, catalog)


__all__ = [
    "NO_SUCCESSFUL_RESULTS",
    "PER_THOUSAND_LABEL",
    "PER_THOUSAND_REPLIES",
    "REFERENCE_MARK",
    "REPLY_COST_FLOOR",
    "SHARE",
    "SURFACE_ORDER",
    "SURFACE_ORDER_NO_CONTROL",
    "UNITLESS",
    "MeasureRuler",
    "Rulers",
    "analysis_rulers",
    "measure_rulers",
    "reply_cost_ruler",
    "report_rulers",
    "surface_order_sentence",
]
