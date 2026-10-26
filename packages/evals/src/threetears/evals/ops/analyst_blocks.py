"""The analyst's block operations: the block a claim would show, drawn on a snapshot, and the blocks there are.

Each operation reads the snapshot's input (:func:`~threetears.evals.ops.analyst.snapshot_input`, refusing one whose
input moved or no longer reads) and runs its computation (:mod:`threetears.evals.analysis.analyst.blocks`) on the
host's blocking executor. A block the page cannot draw on the snapshot reaches the caller as a
:class:`~threetears.evals.kernel.errors.ValidationFailedError` naming every fault, with its remedy. Every operation is
a pure read: the same snapshot and parameters always return the same result.

The result models are the computation's, and the figure-key grammar the catalog's key forms are read from
(:func:`figure_key_grammar`), with its index and pages (:func:`figure_key_index`, :func:`figure_key_pages`), which the
analyst tool's help serves, re-exported here for the actions layer, which reads no analysis module.
"""

from __future__ import annotations

from collections.abc import Callable

from threetears.evals.analysis.analyst import blocks
from threetears.evals.analysis.analyst.blocks import (
    BLOCK_ROWS,
    BlockCatalog,
    RenderedBlock,
)
from threetears.evals.analysis.report.figure_keys import figure_key_grammar, figure_key_index, figure_key_pages
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.errors import RefusedInput, ValidationFailedError
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.report_plan import ChartQuery
from threetears.evals.ops.analyst import snapshot_input
from threetears.evals.ops.host import OpsHost

__all__ = [
    "BLOCK_ROWS",
    "BlockCatalog",
    "RenderedBlock",
    "block_catalog",
    "block_render",
    "figure_key_grammar",
    "figure_key_index",
    "figure_key_pages",
]


def _refused[T](compute: Callable[[], T]) -> T:
    """Run one computation, its refusal of what the caller asked turned into the service's. Blocking.

    :raises ValidationFailedError: The computation refused the block, with its remedy.
    """
    try:
        return compute()
    except RefusedInput as refused:
        raise ValidationFailedError.refusing(refused) from refused


async def _read[T](host: OpsHost, scope_id: str, snapshot_id: str, compute: Callable[[ReportInput], T]) -> T:
    """One snapshot's input, and a computation run on it on the host's blocking executor."""
    held = await snapshot_input(host, scope_id, snapshot_id)
    return await run_blocking(host.eval_host.blocking_executor, _refused, lambda: compute(held))


async def block_render(
    host: OpsHost,
    scope_id: str,
    *,
    snapshot_id: str,
    block_kind: str,
    query: ChartQuery,
    reference: str | None = None,
    emphasis: str = "",
    title: str = "",
    limit: int = BLOCK_ROWS,
    offset: int = 0,
) -> RenderedBlock:
    """The block a page would draw of a query on the snapshot, as its values table.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :param block_kind: The kind of block.
    :ptype block_kind: str
    :param query: What it reads.
    :ptype query: ChartQuery
    :param reference: The reference line to draw; None for none.
    :ptype reference: str | None
    :param emphasis: The one arm, stratum or task it stresses; empty for none.
    :ptype emphasis: str
    :param title: Its title, figure keys allowed; empty for none.
    :ptype title: str
    :param limit: The most values-table rows on the page.
    :ptype limit: int
    :param offset: The rows to skip.
    :ptype offset: int
    :return: The block.
    :rtype: RenderedBlock
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read, or the block cannot be drawn on it, naming every fault.
    """
    return await _read(
        host,
        scope_id,
        snapshot_id,
        lambda held: blocks.block_render(
            held,
            block_kind=block_kind,
            query=query,
            reference=reference,
            emphasis=emphasis,
            title=title,
            limit=limit,
            offset=offset,
        ),
    )


async def block_catalog(host: OpsHost, scope_id: str, *, snapshot_id: str) -> BlockCatalog:
    """Every block the snapshot can be shown by: the kinds, the sources, the query's options, the figure key forms.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :return: The catalog.
    :rtype: BlockCatalog
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read; the remedy is to freeze again.
    """
    return await _read(host, scope_id, snapshot_id, lambda held: blocks.block_catalog(held, snapshot_id))
