"""One intent builder per chart type, and the registry that names them.

A builder turns one validated payload into a :class:`~threetears.evals.analysis.viz.intent.ChartIntent`:
the order its rows are drawn in, the unit each quantity is stated in, the values as drawn, what each
field encodes and what the chart must say beside itself. Nothing here knows how a chart is drawn —
no geometry, no fonts, no charting grammar — which is what lets a renderer be swapped without the
chart changing what it claims.

:data:`INTENTS` mirrors :data:`~threetears.evals.analysis.viz.payloads.PAYLOAD_MODELS` key for key, and
is held to it by test: a type with a payload model and no builder would parse and then refuse to be
decided, which is a registration mistake wearing a data error's clothes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from threetears.evals.analysis.viz.intent import ChartIntent
from threetears.evals.analysis.viz.intents.attribution import attribution_intent
from threetears.evals.analysis.viz.intents.breakdown import breakdown_intent
from threetears.evals.analysis.viz.intents.delta_table import delta_table_intent
from threetears.evals.analysis.viz.intents.distribution import distribution_intent
from threetears.evals.analysis.viz.intents.frontier import frontier_intent
from threetears.evals.analysis.viz.intents.null_result import null_result_intent
from threetears.evals.analysis.viz.intents.sweep_ranking import sweep_ranking_intent
from threetears.evals.analysis.viz.intents.timeseries import timeseries_intent

#: ``Viz.type`` → the builder that decides it. Keyed by the same strings as ``PAYLOAD_MODELS``.
INTENTS: dict[str, Callable[[Any], ChartIntent]] = {
    "attribution": attribution_intent,
    "breakdown": breakdown_intent,
    "delta_table": delta_table_intent,
    "distribution": distribution_intent,
    "frontier": frontier_intent,
    "null_result": null_result_intent,
    "sweep_ranking": sweep_ranking_intent,
    "timeseries": timeseries_intent,
}


__all__ = [
    "INTENTS",
]
