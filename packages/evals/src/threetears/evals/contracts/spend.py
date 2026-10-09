"""What eval prices a reported external spend at.

The eval half of the metering seam. :class:`~threetears.evals.contracts.host.spend.ExternalSpend` is the
report — shared with the host's tools, which is why it lives on the contracts leaf — and
this module holds what only eval does with it: apply an operator-declared rate, and name
where the resulting figure came from. A tool never prices, so nothing here is shared.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from threetears.evals.contracts.host.spend import ExternalSpend

__all__ = ["ExternalRateTable", "reported_price_source"]


def reported_price_source(provider: str, *, from_rate_table: bool) -> str:
    """Name where a dollar figure came from, so it stays re-derivable.

    Distinguishes the two provenances a non-LLM charge can have, because a reader deciding
    whether to re-derive at current rates needs to tell them apart: a provider-reported
    charge is an observation and will not move, while an operator-declared rate applied to
    a counted volume moves the moment the operator edits it.

    Args:
        provider: The provider the call was made to.
        from_rate_table: Whether the figure came from an operator-declared rate rather than
            from the provider's own report.

    Returns:
        A provenance string, e.g. ``"search_api:configured_rate"`` or ``"search_api:reported"``.
    """
    return f"{provider}:configured_rate" if from_rate_table else f"{provider}:reported"


@dataclass(frozen=True)
class ExternalRateTable:
    """Operator-declared money per provider unit, resolved once for a whole run.

    Resolved at launch and carried to every cell, which is the property the single rate
    card already had and the reason the rate is not looked up at read time: an operator
    editing a hot-reloadable rate mid-run would otherwise leave one run's cells measured
    against two different rates. What widens here is only the **arity** — one entry per
    ``(provider, unit)`` instead of one card for one provider.

    **A provider with no declared rate stays counted and unpriced.** That arm is not a gap
    to close in the widening; it is the honest state for a real cost nobody declared a rate
    for, and every cost surface already discloses it.

    **Rates are USD**, as is every figure this seam carries — see
    :mod:`threetears.evals.contracts.host.spend`. A currency field here would be a knob nothing honours:
    ``cost_usd`` is a USD field and every cost surface renders it with ``$``, so a second
    currency is a change to the stored schema, the budget cap and every renderer, not a
    value to declare.

    Attributes:
        rates: USD per single unit, keyed by ``(provider, bare unit name)``.
    """

    rates: Mapping[tuple[str, str], float]

    def money_for(self, spend: ExternalSpend) -> float | None:
        """What ``spend`` cost, or ``None`` when nothing can honestly say.

        A charge the provider itself reported wins over any declared rate: it is an
        observation, and re-deriving it from a rate would replace a measurement with an
        estimate. Otherwise the units are priced at this table's rate for that exact
        ``(provider, unit)`` pair — never at another provider's rate, and never at a
        default standing in for a rate nobody set.

        Args:
            spend: The contribution to price.

        Returns:
            The dollar figure, or ``None`` when the provider reported none and no rate was
            declared for its unit.
        """
        if spend.money is not None:
            return spend.money
        if spend.provider is None or spend.provider_units is None or spend.provider_unit is None:
            return None
        rate = self.rates.get((spend.provider, spend.provider_unit))
        if rate is None:
            return None
        return spend.provider_units * rate
