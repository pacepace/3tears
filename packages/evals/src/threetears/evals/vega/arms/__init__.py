"""One Vega-Lite arm per chart type, and the registry that names them.

An arm draws one decided :class:`~threetears.evals.analysis.viz.intent.ChartIntent` as a Vega-Lite
spec. It owns its own shape and nothing else: what the chart says — order, units, values, disclosures —
arrives decided in the intent, and the layout arithmetic, the marks, the value axis and the label
placement come from :mod:`threetears.evals.vega.compiler`, which every arm imports and no arm
imports from a sibling. That is what makes a new type a new file rather than a new branch in a shared
function.

:data:`ARMS` mirrors :data:`~threetears.evals.analysis.viz.payloads.PAYLOAD_MODELS` key for
key, and is held to it by test: a type that gains a payload model without gaining
an arm would decide its intent and then refuse to draw, which is a registration mistake
wearing a data error's clothes.
"""

from __future__ import annotations

from typing import Any, Protocol

from threetears.evals.vega.arms.attribution import compile_attribution
from threetears.evals.vega.arms.breakdown import compile_breakdown
from threetears.evals.vega.arms.delta_table import compile_delta_table
from threetears.evals.vega.arms.distribution import compile_distribution
from threetears.evals.vega.arms.frontier import compile_frontier
from threetears.evals.vega.arms.null_result import compile_null_result
from threetears.evals.vega.arms.sweep_ranking import compile_sweep_ranking
from threetears.evals.vega.arms.timeseries import compile_timeseries
from threetears.evals.analysis.viz.intent import ChartIntent
from threetears.evals.contracts.host import ChartFont


class Arm(Protocol):
    """What draws one chart type: an intent, laid out in a typeface, as a Vega-Lite spec."""

    def __call__(self, intent: ChartIntent, *, font: ChartFont | None = None) -> dict[str, Any]:
        """Draw ``intent``; ``font`` is the face its layout is measured in, ``None`` for the packaged one."""
        ...


def _frontier(intent: ChartIntent, *, font: ChartFont | None = None) -> dict[str, Any]:
    """The frontier arm, which does not take a typeface yet and lays out in the packaged face.

    Its one measured layout decision is where its title wraps. Under a host's own face that title is
    wrapped against the packaged face's widths, which can overrun the figure or wrap early; nothing
    else on the figure is measured. Threading ``font`` into ``compile_frontier`` retires this.
    """
    return compile_frontier(intent)


#: ``Viz.type`` → the arm that draws it.
#:
#: Keyed by the same strings as ``PAYLOAD_MODELS`` so the two registries can be
#: compared directly rather than related by convention.
ARMS: dict[str, Arm] = {
    "attribution": compile_attribution,
    "breakdown": compile_breakdown,
    "delta_table": compile_delta_table,
    "distribution": compile_distribution,
    "frontier": _frontier,
    "null_result": compile_null_result,
    "sweep_ranking": compile_sweep_ranking,
    "timeseries": compile_timeseries,
}


__all__ = [
    "ARMS",
    "Arm",
]
