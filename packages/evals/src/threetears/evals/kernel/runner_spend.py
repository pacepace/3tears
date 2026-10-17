"""What a runner's own model calls spent: a ledger open around one runner call, which a client adds each completion to.

A runner an eval calls may call a model itself (``await model.text(...)``) and return a plain value, reporting nothing
of what that cost. So every runner call is made inside :func:`cell_spend`, a ledger for that one call, and a client
that reports its completions (the shipped :class:`~threetears.evals.providers.Model` does) adds each one while it is
open (:func:`note_spend`); :func:`spent` folds the ledger's rows into the call's telemetry, as the ``candidate``
role's usage, unpriced where a completion was. Outside a runner call, :func:`note_spend` keeps nothing.

The ledger is a context variable, so the cells of one run, each its own task, never share one, and a synchronous
runner run in a worker thread (which copies the context) writes to its own cell's.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from threetears.evals.kernel.candidate_kind import CandidateTelemetry
from threetears.evals.kernel.usage_capture import RoleUsageLedger
from threetears.evals.schema.completion import CompletionResult

#: The ledger of the runner call in progress in this context, or ``None`` outside one.
_SPEND: ContextVar[RoleUsageLedger | None] = ContextVar("threetears_evals_runner_spend", default=None)


@contextmanager
def cell_spend() -> Iterator[RoleUsageLedger]:
    """A ledger for one runner call: every completion :func:`note_spend` is handed while it is open is added to it.

    :yield: The ledger, the ``candidate`` role's, whose rows are the call's spend.
    :rtype: Iterator[RoleUsageLedger]
    """
    ledger = RoleUsageLedger(role="candidate")
    token = _SPEND.set(ledger)
    try:
        yield ledger
    finally:
        _SPEND.reset(token)


def note_spend(completion: CompletionResult) -> None:
    """Add one completion to the ledger of the runner call in progress; outside a runner call, nothing is kept.

    :param completion: What the call returned: its model, tokens and cost, unpriced when ``cost_usd`` is ``None``.
    :ptype completion: CompletionResult
    """
    ledger = _SPEND.get()
    if ledger is not None:
        ledger.add_llm_result(completion)


def spent(ledger: RoleUsageLedger, telemetry: CandidateTelemetry) -> CandidateTelemetry:
    """``telemetry`` with ``ledger``'s rows added: what the runner's model calls cost beside what it reported.

    :param ledger: The runner call's ledger (:func:`cell_spend`).
    :ptype ledger: RoleUsageLedger
    :param telemetry: What the runner's return reported (an ``Answer``'s usage, or none).
    :ptype telemetry: CandidateTelemetry
    :return: The telemetry carrying both; ``telemetry`` itself when no completion was noted.
    :rtype: CandidateTelemetry
    """
    rows = ledger.rows()
    if not rows:
        return telemetry
    return telemetry.model_copy(update={"usage": [*telemetry.usage, *rows]})


__all__ = ["cell_spend", "note_spend", "spent"]
