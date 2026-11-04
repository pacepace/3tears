"""The kernel's names the in-cell cost cap left, kept for one minor after the cap moved between cells.

The run's cost cap used to refuse a cell's next paid call from inside the cell, raised as :class:`CostCapReached`. It is
now read between cells, against what the recorded cells landed (:func:`~threetears.evals.run.runner.execute_run`), so
nothing the engine runs raises it and nothing it runs catches it. A host that imports it, or catches it, keeps
importing and compiling: it stays importable from :mod:`threetears.evals.kernel` and warns on every read
(:func:`~threetears.evals.kernel.retired.served_retired`). The next minor removes it.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from threetears.evals.kernel.retired import COST_CAP_INSIDE_A_CELL_RETIRED, Retired


class CostCapReached(Exception):
    """Retired: the engine no longer raises it, because no ceiling is asked inside a cell.

    The run's cost cap used to refuse a cell's next paid call from inside the cell, raised as this from
    :meth:`~threetears.evals.kernel.candidate_kind.CellSink.reserve`. A cell already running now finishes whole, and a
    cell that loops is bounded by the cell timeout. Drop the ``except`` clause that catches it.
    """

    def __init__(self, detail: str = "the run's cost cap has no room for another paid call") -> None:
        """Say what the cap refused.

        :param detail: What was refused, for a log line; the cell's record names the cap's own numbers.
        :ptype detail: str
        """
        super().__init__(detail)
        self.detail = detail


#: Each retired name the kernel still serves, with why it retired.
RETIRED: Mapping[str, Retired] = MappingProxyType(
    {
        "CostCapReached": Retired(
            CostCapReached, f"{COST_CAP_INSIDE_A_CELL_RETIRED}; nothing raises it, so drop the except clause"
        ),
    }
)

__all__ = ["RETIRED", "CostCapReached"]
