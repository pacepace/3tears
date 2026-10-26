"""The analyst's analyses as operations: each reads one snapshot's frozen report input and computes over it.

Every operation reads the input by the snapshot's id (:func:`~threetears.evals.ops.analyst.snapshot_input`), so the
analyst, the editor and a replay compute over the same figures, and runs its computation
(:mod:`threetears.evals.analysis.analyst.analyses`) on the host's blocking executor: a sweep reads code's decision at
every flip. A computation that refuses its arguments (a parameter naming nothing the snapshot holds, a breakdown with
no control) is refused as the call, with the remedy it names.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from threetears.evals.analysis.analyst.analyses import (
    BREAKDOWN_PAGE,
    SENSITIVITY_PAGE,
    Breakdown,
    Dominance,
    Sensitivity,
    SliceBy,
    breakdown,
    dominance,
    sensitivity,
)
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.errors import RefusedInput, ValidationFailedError
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops.analyst import snapshot_input
from threetears.evals.ops.host import OpsHost


async def _computed[T](host: OpsHost, scope_id: str, snapshot_id: str, compute: Callable[[ReportInput], T]) -> T:
    """One analysis over a snapshot's input, on the blocking executor; a refusal of its arguments is the call's."""
    held = await snapshot_input(host, scope_id, snapshot_id)
    try:
        return await run_blocking(host.eval_host.blocking_executor, compute, held)
    except RefusedInput as refused:
        raise ValidationFailedError(refused.fault, remedy=refused.remedy) from refused


async def options_dominance(host: OpsHost, scope_id: str, *, snapshot_id: str, axes: Sequence[str]) -> Dominance:
    """Which options survive the gates, which dominates on the decision's axes, and what extra axes say beside them.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :param axes: Extra measures or judged dimensions to compare the survivors on, descriptively.
    :ptype axes: Sequence[str]
    :return: The survivors, the options removed, the dominant ones, the extra axes and code's call.
    :rtype: Dominance
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read, or an axis names nothing it holds.
    """
    return await _computed(host, scope_id, snapshot_id, lambda held: dominance(held, axes))


async def decision_sensitivity(
    host: OpsHost,
    scope_id: str,
    *,
    snapshot_id: str,
    parameter: str,
    range_low: float | None = None,
    range_high: float | None = None,
    limit: int = SENSITIVITY_PAGE,
    offset: int = 0,
) -> Sensitivity:
    """How code's call moves as one declared criterion moves: where each arm's status flips, and the call between.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :param parameter: ``margin:<aspect>``, ``floor:<aspect>/<floor>`` or ``threshold:<criterion>``.
    :ptype parameter: str
    :param range_low: Where the sweep starts; omitted, zero for a margin or floor, no lower end for a threshold.
    :ptype range_low: float | None
    :param range_high: Where it ends; omitted, no upper end.
    :ptype range_high: float | None
    :param limit: The most flips one page lists.
    :ptype limit: int
    :param offset: The first flip the page lists.
    :ptype offset: int
    :return: The segments, a page of flips, and whether the call holds across the range.
    :rtype: Sensitivity
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read, the parameter names nothing it holds, or the range is
        not one.
    """
    return await _computed(
        host,
        scope_id,
        snapshot_id,
        lambda held: sensitivity(
            held, parameter, range_low=range_low, range_high=range_high, limit=limit, offset=offset
        ),
    )


async def measure_breakdown(
    host: OpsHost,
    scope_id: str,
    *,
    snapshot_id: str,
    measure_key: str,
    arm: str | None = None,
    slice_by: SliceBy = "stratum",
    aspect: str | None = None,
    limit: int = BREAKDOWN_PAGE,
    offset: int = 0,
) -> Breakdown:
    """A measure slice by slice: each candidate against the control in each stratum, task or cell. Exploratory.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :param measure_key: The measure or judged dimension, by key.
    :ptype measure_key: str
    :param arm: The one candidate to show; None for every candidate.
    :ptype arm: str | None
    :param slice_by: ``stratum``, ``task`` or ``cell``.
    :ptype slice_by: SliceBy
    :param aspect: For strata, only those slicing this aspect's cases.
    :ptype aspect: str | None
    :param limit: The most slices one page holds.
    :ptype limit: int
    :param offset: The first slice the page holds.
    :ptype offset: int
    :return: A page of slices, and where each candidate's loss sits most.
    :rtype: Breakdown
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read or has no control, or the arguments name nothing it
        holds.
    """
    return await _computed(
        host,
        scope_id,
        snapshot_id,
        lambda held: breakdown(
            held, measure_key, arm=arm, slice_by=slice_by, aspect=aspect, limit=limit, offset=offset
        ),
    )


__all__ = [
    "BREAKDOWN_PAGE",
    "SENSITIVITY_PAGE",
    "Breakdown",
    "Dominance",
    "Sensitivity",
    "SliceBy",
    "decision_sensitivity",
    "measure_breakdown",
    "options_dominance",
]
