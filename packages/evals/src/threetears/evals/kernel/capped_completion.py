"""A completion client held to the run's cost cap, one call at a time, through one cell's sink.

A cell's paid calls are what a run's cost cap exists to bound, and the cap counts a cell's spend only
once its result lands unless the cell reports it as it goes. :class:`CappedCompletion` is that report
for a completion client, written once: before each call it reserves the call's ceiling on the cell's
sink (:meth:`~threetears.evals.kernel.candidate_kind.CellSink.reserve`) — or, for a client that cannot
price the call, asks whether the cap is reached — and when the call returns it settles the reservation
to what the call cost. A call the cap has no room for is never made: the reservation raises
:class:`~threetears.evals.kernel.candidate_kind.CostCapReached` instead.

The engine wraps its own cell-side clients in it — the simulated user's
(:func:`~threetears.evals.run.conversation.drive_conversation`) and the reporter kind's generator — and a
host's kind wraps its candidate's client the same way. :func:`call_ceiling` is the pricing it does,
for a caller that reserves a call it makes through something other than ``generate`` (a judge request,
whose service sends it).

**What a ceiling is comes from the client** (:meth:`~threetears.evals.schema.completion.PricedCompletion.price_ceiling`):
the engine knows neither a model's rates nor the output cap the client was built with. A client with no
``price_ceiling``, or one answering ``None``, is unpriced: its calls hold nothing and are admitted only
while the cap is not yet reached.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from threetears.evals.kernel.candidate_kind import CellPending, CellSink
from threetears.evals.kernel.usage_capture import CallUsage
from threetears.evals.schema.completion import CompletionGenerator

__all__ = ["CappedCompletion", "call_ceiling"]


@runtime_checkable
class _PricesItsCalls(Protocol):
    """A client that can bound what one call costs before it is made."""

    def price_ceiling(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> float | None: ...  # pragma: no cover — protocol


def call_ceiling(
    client: object,
    *,
    system: str,
    user: str,
    response_format: dict[str, Any] | None = None,
    attempts: int = 1,
) -> float | None:
    """The most ``attempts`` requests of one prompt pair on ``client`` can cost, or ``None`` when it cannot say.

    Args:
        client: The client the call will be made on.
        system: The system prompt the call sends.
        user: The user message the call sends.
        response_format: The provider directive the call sends, or ``None``.
        attempts: How many requests the call can become in the engine's own hands — a judge's parse retry
            (:data:`~threetears.evals.run.judge.JUDGE_CALL_ATTEMPTS`) re-sends the same pair. The client's
            own re-sends are already inside its ceiling.

    Returns:
        The ceiling in dollars; ``None`` for a client with no ``price_ceiling``, or one that answered ``None``.

    Raises:
        ValueError: ``attempts`` is below 1, or the client answered an amount that is negative or not finite.
    """
    if attempts < 1:
        raise ValueError(f"a call is at least one request; got attempts={attempts}")
    if not isinstance(client, _PricesItsCalls):
        return None
    ceiling = client.price_ceiling(system=system, user=user, response_format=response_format)
    if ceiling is None:
        return None
    if isinstance(ceiling, bool) or not isinstance(ceiling, int | float) or not math.isfinite(ceiling) or ceiling < 0:
        raise ValueError(
            f"{type(client).__name__} priced a call at {ceiling!r}; a ceiling is a finite amount, 0 or more"
        )
    return float(ceiling) * attempts


class CappedCompletion:
    """One cell's view of a completion client: every call reserved against the run's cap, and settled to its cost.

    Per cell, since the sink is. It owns nothing — the client it forwards to is whoever built it — so it has no
    lifecycle of its own. What a call returns is handed back unchanged, so a caller reading its usage reads it
    as it would off the client.
    """

    def __init__(self, inner: CompletionGenerator, *, sink: CellSink, source: CellPending) -> None:
        """Bind the client, the cell's sink, and whose spend its calls are.

        Args:
            inner: The client every call is forwarded to.
            sink: The cell's sink, which holds each call against the cap.
            source: Whose spend the calls are — ``simulator`` for the simulated user's, ``candidate`` for a
                candidate's own.
        """
        self._inner = inner
        self._sink = sink
        self._source: CellPending = source

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        """Reserve the call's ceiling, make it, and settle the reservation to what it cost.

        Args:
            system: The system prompt.
            user: The user message.
            response_format: The provider directive, or ``None``.

        Returns:
            The client's completion, unchanged.

        Raises:
            CostCapReached: The cap has no room for the call, which is not made.
        """
        held = self._sink.reserve(
            self._source, call_ceiling(self._inner, system=system, user=user, response_format=response_format)
        )
        try:
            result = await self._inner.generate(system=system, user=user, response_format=response_format)
        except BaseException:
            # A call that raised — or was cancelled — returned no cost to settle to; whatever it was billed is
            # unknown here, and the cell's own record reports what its caller learned of it.
            held.release()
            raise
        held.settle(CallUsage.of(result).cost_usd)
        return result
