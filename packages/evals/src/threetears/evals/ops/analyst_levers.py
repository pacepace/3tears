"""A campaign's lever effects as an operation: it reads one snapshot's frozen report input and reads what each setting
the arms vary does to a measure.

The operation reads the input by the snapshot's id (:func:`~threetears.evals.ops.analyst.snapshot_input`), so the
analyst, the editor and a replay read the same figures, and runs the computation
(:func:`~threetears.evals.analysis.analyst.levers.lever_effects`) on the host's blocking executor. A computation that
refuses its arguments (a measure, place or setting naming nothing the snapshot holds, arms that isolate no setting) is
refused as the call, with the remedy it names.
"""

from __future__ import annotations

from collections.abc import Callable

from threetears.evals.analysis.analyst.levers import LEVERS_LIMIT, LeverEffects
from threetears.evals.analysis.analyst.levers import lever_effects as read_levers
from threetears.evals.kernel.errors import RefusedInput, ValidationFailedError
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops.analyst import sized_page, snapshot_input
from threetears.evals.ops.host import OpsHost


async def levers_effects(
    host: OpsHost,
    scope_id: str,
    *,
    snapshot_id: str,
    measure_key: str,
    setting: str | None = None,
    place: str | None = None,
    limit: int = LEVERS_LIMIT,
    offset: int = 0,
    fits: Callable[[LeverEffects], bool] | None = None,
) -> LeverEffects:
    """What each setting the arms vary does to one measure, between arms that differ in it alone, and whether that
    changes with another setting.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :param measure_key: The measure or judged dimension, by key.
    :ptype measure_key: str
    :param setting: Only this setting's effects and interactions; None for every setting the arms vary.
    :ptype setting: str | None
    :param place: The task, stratum or cell to read the measure within; None for overall.
    :ptype place: str | None
    :param limit: The most rows (effects, then interactions) one page holds.
    :ptype limit: int
    :param offset: The first of them the page holds.
    :ptype offset: int
    :param fits: Whether a page's text is short enough; the page is the largest within ``limit`` that passes, and
        ``None`` takes ``limit`` as it is.
    :ptype fits: Callable[[LeverEffects], bool] | None
    :return: The settings the arms vary, and a page of effects and interactions.
    :rtype: LeverEffects
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read; the measure, place or setting names nothing it holds;
        no setting varies, or no two arms differ in exactly one; or no such pair holds the measure there.
    """
    held = await snapshot_input(host, scope_id, snapshot_id)
    try:
        return await run_blocking(
            host.eval_host.blocking_executor,
            sized_page,
            lambda rows: read_levers(
                held, measure_key, setting_filter=setting, place_filter=place, limit=rows, offset=offset
            ),
            limit,
            fits,
        )
    except RefusedInput as refused:
        raise ValidationFailedError.refusing(refused) from refused


__all__ = ["LEVERS_LIMIT", "LeverEffects", "levers_effects"]
