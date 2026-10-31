"""The price of a trade as an operation: it reads one snapshot's frozen report input and prices it.

The operation reads the input by the snapshot's id (:func:`~threetears.evals.ops.analyst.snapshot_input`), so the
analyst, the editor and a replay price the same figures, and runs the computation
(:func:`~threetears.evals.analysis.analyst.price.price_tradeoff`) on the host's blocking executor. A computation that
refuses its arguments (a gain key naming nothing the snapshot holds, a snapshot with no cost) is refused as the call,
with the remedy it names.
"""

from __future__ import annotations

from collections.abc import Callable

from threetears.evals.analysis.analyst.price import PRICE_PAGE, PriceTradeoff
from threetears.evals.analysis.analyst.price import price_tradeoff as priced
from threetears.evals.kernel.errors import RefusedInput, ValidationFailedError
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops.analyst import sized_page, snapshot_input
from threetears.evals.ops.host import OpsHost


async def price_tradeoff(
    host: OpsHost,
    scope_id: str,
    *,
    snapshot_id: str,
    gain_key: str,
    arm: str | None = None,
    slice_filter: str | None = None,
    what_if: str | None = None,
    limit: int = PRICE_PAGE,
    offset: int = 0,
    fits: Callable[[PriceTradeoff], bool] | None = None,
) -> PriceTradeoff:
    """The trade between a snapshot's surviving options, priced on one gain axis.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :param gain_key: The gain axis: a measure with a direction, or a judged dimension, by key.
    :ptype gain_key: str
    :param arm: The option every other is priced against; None for the control.
    :ptype arm: str | None
    :param slice_filter: The one task, stratum or cell to price within, by id; None prices overall.
    :ptype slice_filter: str | None
    :param what_if: An option the gates remove or leave undecided, to price as a labelled what-if; None for none.
    :ptype what_if: str | None
    :param limit: The most options one page covers.
    :ptype limit: int
    :param offset: The first of them the page holds.
    :ptype offset: int
    :param fits: Whether a page's text is short enough; the page is the largest within ``limit`` that passes, and
        ``None`` takes ``limit`` as it is.
    :ptype fits: Callable[[PriceTradeoff], bool] | None
    :return: The frontier, the price per point at each step along it, every option priced against one, why each other
        option is off the frontier, and the what-if asked for.
    :rtype: PriceTradeoff
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read; the gain key names no measure or judged dimension with
        a direction; no arm has a cost; ``arm`` names no priced option; ``slice_filter`` names no one slice; or
        ``what_if`` names no option the gates remove or leave undecided.
    """
    held = await snapshot_input(host, scope_id, snapshot_id)
    try:
        return await run_blocking(
            host.eval_host.blocking_executor,
            sized_page,
            lambda rows: priced(
                held, gain_key, arm=arm, slice_filter=slice_filter, what_if=what_if, limit=rows, offset=offset
            ),
            limit,
            fits,
        )
    except RefusedInput as refused:
        raise ValidationFailedError.refusing(refused) from refused


__all__ = ["PRICE_PAGE", "PriceTradeoff", "price_tradeoff"]
