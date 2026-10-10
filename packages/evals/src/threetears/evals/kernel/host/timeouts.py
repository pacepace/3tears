"""The cell-timeout port: what bounds one cell's wall clock, supplied by the host.

A port rather than a fixed ``asyncio.timeout`` for the reason :mod:`~threetears.evals.schema.traces`
is one: a host with a timeout layer of its own (named budgets, telemetry, nested attribution) runs every
long operation inside it, and a cell that ran outside it would vanish from that host's operation
registry. :func:`default_cell_timeout` is the engine's own answer for a host with no such layer, and a
host that takes it names it on its :class:`~threetears.evals.kernel.host.eval_host.EvalHost` rather
than inheriting it, so the choice is visible where the host is built.

Here, beside the host contract, because :class:`~threetears.evals.kernel.host.eval_host.EvalHost`
carries the factory and the contracts package imports nothing of the run package. The exception lives
with the factory because the default raises it and the run loop catches exactly it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol


class EvalCellTimeout(Exception):
    """A cell outlived the wall-clock budget its timeout context was enforcing.

    Engine-owned for the reason the job manager's own timeout type is: the loop needs one type to
    branch on — a breached cell budget records a ``cell_timeout`` result and the run goes on to the
    next cell, where anything else leaves the loop — while the context manager enforcing the budget
    comes from the host. A type the host owned would make the engine name a module it cannot be
    installed without.

    Attributes:
        budget_s: The budget that was breached, in seconds.
        elapsed_s: Wall-clock seconds spent before it fired.
        attribution: What the host names as innermost when the deadline hit, for a host whose
            timeouts carry names; ``None`` when it has none to give, which is the case for
            :func:`default_cell_timeout`.
    """

    def __init__(self, budget_s: float, elapsed_s: float, attribution: str | None = None) -> None:
        """Record the breached budget, the time spent under it, and the host's attribution.

        Args:
            budget_s: The budget that was breached, in seconds.
            elapsed_s: Wall-clock seconds spent before it fired.
            attribution: The host's name for what was innermost, or ``None``.
        """
        super().__init__(f"cell exceeded its {budget_s:.0f}s budget after {elapsed_s:.0f}s")
        self.budget_s = budget_s
        self.elapsed_s = elapsed_s
        self.attribution = attribution


class CellTimeoutFactory(Protocol):
    """Builds the context manager one cell runs inside, from that cell's budget.

    The cell-level twin of the job manager's timeout factory, with the same three obligations, each
    of which the loop relies on:

    * Reject a non-positive budget **on entry, before the body runs**.
    * Raise :class:`EvalCellTimeout` when the budget is breached. The loop catches that type and
      nothing else, so a host's own timeout type escaping the cell ends the run rather than
      recording one cell.
    * Let everything else through untouched, cancellation included — an operator cancelling the
      run is not a cell timing out.

    The loop binds nothing from the yield, so a host owes this port three behaviours and no value.
    """

    def __call__(self, budget_s: float) -> AbstractAsyncContextManager[Any]:
        """Return the context manager that enforces ``budget_s`` for one cell."""
        ...  # pragma: no cover — protocol


@asynccontextmanager
async def default_cell_timeout(budget_s: float) -> AsyncIterator[float]:
    """Enforce a cell budget with :func:`asyncio.timeout` and nothing else.

    What the engine ships to a host that supplies no timeout of its own: no named operation, no
    nesting and no attribution, because those are a host's concerns.

    Args:
        budget_s: Wall-clock ceiling for the cell, in seconds.

    Yields:
        The budget.

    Raises:
        ValueError: If ``budget_s`` is not positive — on entry, before the body runs.
        EvalCellTimeout: If the body outlives the budget.
    """
    if budget_s <= 0:
        raise ValueError(f"cell timeout budget must be positive, got {budget_s}")
    started = time.monotonic()
    deadline = asyncio.timeout(budget_s)
    try:
        async with deadline:
            yield budget_s
    except TimeoutError as exc:
        # A body that raises TimeoutError of its own is not this budget firing, and converting it
        # would record a cell as having outlived a budget it was still inside.
        if not deadline.expired():
            raise
        raise EvalCellTimeout(budget_s=budget_s, elapsed_s=time.monotonic() - started) from exc


__all__ = ["CellTimeoutFactory", "EvalCellTimeout", "default_cell_timeout"]
