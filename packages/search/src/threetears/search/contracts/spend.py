"""Spend -- every resource a call consumed, not only money (SR-E1).

Modelling spend as currency alone is the documented trap: the two
constraints that bite hardest are unpriced -- a self-hosted SearXNG costs
nothing and fails by ban rather than by bill (SR-D6), and wall-clock burned
against an already-spent budget is pure latency. Money is one dimension
among five.

Rules this type carries:

- money is :class:`~decimal.Decimal`, never float (billing arithmetic);
- the count a cap enforces and the count a bill prices are the same
  ``calls`` number (SR-E2) -- there is deliberately no second tally field;
- weighted provider units are their own dimension (SR-E4: Tavily
  ``advanced`` = 2 credits; the unit is provider-defined);
- per-request (not per-result) pricing is representable because spend
  attaches per call, carrying no result count (SR-E5);
- spend survives the failure path: every typed error in
  :mod:`threetears.search.contracts.errors` carries one (SR-E3).
"""

from __future__ import annotations

from decimal import Decimal

from threetears.search.contracts._base import ContractModel

__all__ = ["Spend"]


class Spend(ContractModel):
    """Resources one call (or one aggregation of calls) consumed.

    A zero spend is ``Spend()`` -- every dimension defaults to zero, so a
    free call is representable without inventing synthetic pricing (D6).
    """

    #: money actually billed, in ``currency``. Follows the bill (D4): a
    #: retried attempt that never billed contributes nothing here.
    money: Decimal = Decimal("0")
    #: ISO 4217 code for ``money``.
    currency: str = "USD"
    #: elapsed wall-clock attributable to the call, in seconds.
    wall_clock_seconds: float = 0.0
    #: provider calls made. The single number both caps and bills read
    #: (SR-E2).
    calls: int = 0
    #: weighted provider-defined units (credits) consumed (SR-E4).
    provider_units: Decimal = Decimal("0")
    #: bytes moved (request + response payloads, extraction downloads).
    bytes_transferred: int = 0

    def __add__(self, other: Spend) -> Spend:
        """Combine two spends dimension-wise.

        :param other: the spend to add to this one
        :ptype other: Spend
        :return: a new spend with every dimension summed
        :rtype: Spend
        :raises ValueError: when both spends carry nonzero money in
            different currencies -- silently summing across currencies
            would fabricate a bill
        """
        if self.money and other.money and self.currency != other.currency:
            raise ValueError(f"cannot sum spend across currencies: {self.currency!r} + {other.currency!r}")
        currency = self.currency if self.money or not other.money else other.currency
        return Spend(
            money=self.money + other.money,
            currency=currency,
            wall_clock_seconds=self.wall_clock_seconds + other.wall_clock_seconds,
            calls=self.calls + other.calls,
            provider_units=self.provider_units + other.provider_units,
            bytes_transferred=self.bytes_transferred + other.bytes_transferred,
        )
