"""A job's wall-clock budget: the context a job's work runs inside, and what it raises when the budget runs out.

Kernel-owned because every job kind reads it the same way — a run (``budget_stopped``), a launch, a judge pass, an
action, a sweep and an analysis generation (``timed_out``) — and the analysis side, which imports nothing of
:mod:`threetears.evals.run`, records a generation's timeout as any other job does. The context manager itself is the
host's (:class:`JobTimeoutFactory`); :func:`default_job_timeout` is the engine's for a host that has none.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol


class EvalJobTimeout(Exception):
    """A job outlived the wall-clock budget its timeout context was enforcing.

    Engine-owned on purpose. The manager needs one type to branch on, so that a
    breached budget records the designed ``budget_stopped`` with :attr:`stop_reason`
    instead of falling through to the boundary that catches everything and
    records ``failed`` — and the context manager enforcing the budget comes from
    the host. A type the host owned would make the engine name a module it cannot
    be installed without.

    Attributes:
        budget_s: The budget that was breached, in seconds.
        elapsed_s: Wall-clock seconds spent before it fired.
        attribution: What the host names as innermost when the deadline hit — the
            operation itself when nothing was nested inside it, for a host whose
            timeouts carry names. ``None`` only when the implementation has no
            name to give, which is the case for :func:`default_job_timeout`:
            plain ``asyncio.timeout`` has no vocabulary of operations at all.
    """

    def __init__(self, budget_s: float, elapsed_s: float, attribution: str | None = None) -> None:
        """Record the breached budget, the time spent under it, and the host's attribution."""
        super().__init__(f"job exceeded its {budget_s:.0f}s budget after {elapsed_s:.0f}s")
        self.budget_s = budget_s
        self.elapsed_s = elapsed_s
        self.attribution = attribution

    @property
    def stop_reason(self) -> str:
        """Why the run stopped, as ``EvalRun.budget_stop_reason`` records it.

        Opens with ``wall-clock budget`` so a reader tells it from the cost cap's reason, which names
        dollars, without a second field: both are a budget the run was launched under doing its job.
        """
        named = f"; the host named {self.attribution!r} as running when it fired" if self.attribution else ""
        return (
            f"wall-clock budget reached — the run's {self.budget_s:.0f}s time budget ran out after "
            f"{self.elapsed_s:.0f}s{named}; the cells it delivered are kept and its completeness record counts them"
        )


class JobTimeoutFactory(Protocol):
    """Builds the context manager a single job runs inside.

    An async timeout is generic machinery rather than eval domain knowledge, and
    a host that already owns one — a registry of named operation budgets, say,
    with its own telemetry and its own nesting rules — wants the job to run under
    that one rather than beside it. So the engine takes the context manager as a
    value and ships a default (:func:`default_job_timeout`) for a host that has
    none.

    **Called once per job, with that job's own resolved budget**, never once per
    manager: two runs in one process size their matrices differently
    (:func:`adaptive_job_timeout_s`) and must not share a deadline.

    Three obligations, and each is something the manager relies on:

    * Reject a non-positive budget **on entry, before the body runs**. The
      manager tells "this run never got going" from "this run ran" by whether
      its work function started, and a budget rejected after the body had begun
      would make that derivation lie.
    * Raise :class:`EvalJobTimeout` when the budget is breached, so the manager
      records the designed ``budget_stopped`` rather than a generic failure.
    * Let everything else through untouched, cancellation included — an operator
      cancelling a job is not a timeout and must reach its own branch.

    **The manager binds nothing from the yield** — it enters the context with no
    ``as`` — so the type is ``Any`` rather than the budget. An implementation may
    yield whatever its own layer yields (``operation_timeout`` yields the computed
    budget, and its other callers do read it); the point of saying so here is that
    a host owes this port three behaviours and no value.
    """

    def __call__(self, budget_s: float) -> AbstractAsyncContextManager[Any]:
        """Return the context manager that enforces ``budget_s`` for one job."""
        ...


@asynccontextmanager
async def default_job_timeout(budget_s: float) -> AsyncIterator[float]:
    """Enforce a job budget with :func:`asyncio.timeout` and nothing else.

    What the engine ships to a host that supplies no timeout of its own.
    Deliberately thinner than what a host with a timeout registry injects — no
    named operation, no nesting, no attribution — because those are the host's
    concerns, and an engine that assumed them would not install without one.

    Args:
        budget_s: Wall-clock ceiling for the body, in seconds.

    Yields:
        The budget, so a caller can honour it at an inner layer.

    Raises:
        ValueError: If ``budget_s`` is not positive — on entry, before the body
            runs. See :class:`JobTimeoutFactory` for why the ordering is part of
            the contract rather than an implementation detail.
        EvalJobTimeout: If the body outlives the budget.
    """
    if budget_s <= 0:
        raise ValueError(f"job timeout budget must be positive, got {budget_s}")
    started = time.monotonic()
    deadline = asyncio.timeout(budget_s)
    try:
        async with deadline:
            yield budget_s
    except TimeoutError as exc:
        # A body that raises TimeoutError of its own is not this budget firing, and
        # converting it would report a job as having outlived a budget it was inside.
        if not deadline.expired():
            raise
        raise EvalJobTimeout(budget_s=budget_s, elapsed_s=time.monotonic() - started) from exc


def timed_out_detail(timed_out: EvalJobTimeout) -> str:
    """Why a job's work was stopped by its time budget, in a poll's words — one wording for every job kind.

    Args:
        timed_out: The budget's firing.

    Returns:
        The record's ``detail`` (an analysis attempt's ``error``).
    """
    named = f"; the host named {timed_out.attribution!r} as running when it fired" if timed_out.attribution else ""
    return (
        f"timed out: its {timed_out.budget_s:.0f}s wall-clock budget ran out after {timed_out.elapsed_s:.0f}s"
        f"{named}; what it recorded before then is kept"
    )


__all__ = ["EvalJobTimeout", "JobTimeoutFactory", "default_job_timeout", "timed_out_detail"]
