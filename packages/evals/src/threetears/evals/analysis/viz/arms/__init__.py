"""One compiler arm per viz type, and the registry that names them.

An arm turns one validated payload into a :class:`CompiledChart`. It owns its own
shape and nothing else: the layout arithmetic, the marks, the value axis and the
label placement all come from :mod:`threetears.evals.analysis.viz.compiler`, which every arm
imports and no arm imports from a sibling. That is what makes a new type a new
file rather than a new branch in a shared function.

:data:`ARMS` mirrors :data:`~threetears.evals.analysis.viz.payloads.PAYLOAD_MODELS` key for
key, and is held to it by test: a type that gains a payload model without gaining
an arm would parse and then refuse to draw, which is a registration mistake
wearing a data error's clothes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from threetears.evals.analysis.viz.arms.attribution import compile_attribution
from threetears.evals.analysis.viz.arms.breakdown import compile_breakdown
from threetears.evals.analysis.viz.arms.delta_table import compile_delta_table
from threetears.evals.analysis.viz.arms.distribution import compile_distribution
from threetears.evals.analysis.viz.arms.frontier import compile_frontier
from threetears.evals.analysis.viz.arms.null_result import compile_null_result
from threetears.evals.analysis.viz.arms.sweep_ranking import compile_sweep_ranking
from threetears.evals.analysis.viz.arms.timeseries import compile_timeseries
from threetears.evals.analysis.viz.compiler import CompiledChart

#: ``Viz.type`` → the arm that draws it.
#:
#: Keyed by the same strings as ``PAYLOAD_MODELS`` so the two registries can be
#: compared directly rather than related by convention.
ARMS: dict[str, Callable[[Any], CompiledChart]] = {
    "attribution": compile_attribution,
    "breakdown": compile_breakdown,
    "delta_table": compile_delta_table,
    "distribution": compile_distribution,
    "frontier": compile_frontier,
    "null_result": compile_null_result,
    "sweep_ranking": compile_sweep_ranking,
    "timeseries": compile_timeseries,
}


__all__ = [
    "ARMS",
]
