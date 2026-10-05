"""What one external call consumed, as the caller that made it reports it.

The value type both sides of the metering seam name. A tool describes its own spend and
eval prices it, so the description has to be sayable without either side importing the
other's internals — which is why it lives on this leaf beside ``apparatus.py`` rather than
being declared twice and adapted by hand in the middle. A second declaration is not a
smaller cost than a shared one: it is a pair that must be edited together, with nothing to
say so, and a field added to one silently reaches neither the other nor the code reading
across.

Two rules shape it, and both are reactions to what the single-provider version could not
say:

**The caller reports; eval prices.** The thing that made the call is the only thing that
knows what the call actually was. When eval instead reached into the host's tools for a
credit table and re-derived the volume from a ``search_depth`` it also had to understand, a
``web_search`` at ``basic`` got billed at another tool's ``advanced`` rate — the code
doing the arithmetic was not the code that knew the depth. Reporting outward removes the
whole class of defect: a caller cannot mis-bill a call it is describing.

**Units are qualified by provider, or they are not units.** Two providers may both call
their unit "credits" without those credits being one fungible quantity, and a total that
adds them is not a smaller truth but a fabricated one. Every contribution therefore STATES
its provider — naming one, or saying explicitly that it could not — and
:attr:`ExternalSpend.qualified_unit` is the one place the ``"<provider>:<unit>"`` form is
composed, so a second spelling cannot appear beside the first and compare unequal.

This is deliberately eval's own vocabulary and not an import of
``threetears.search.contracts.Spend``, which is the same shape for search alone. Eval
accounts for **every** metered external call — image generation, a computation API, a video-API quota —
so search's spend is one compatible instance of this, not its parent. Unifying them later
is a shared-contracts decision, never a dependency edge from eval to search.

**Every figure here is USD**, and that is a property of the system rather than of this type.
``EvalResult.cost_usd`` is a USD field and every cost surface renders it with ``$``, so a
second currency is not a field to add here — it is a change to the stored schema, the budget
cap, and every renderer, and until that happens a currency field would be a knob promising
what nothing honours. The one-currency-per-run property R5 asks for is held by there being
exactly one currency to be measured in.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ExternalSpend"]


@dataclass(frozen=True)
class ExternalSpend:
    """One caller's report of what a provider call (or a batch of them) consumed.

    Every dimension except the provider defaults to nothing-observed, so a caller reports
    only what it can actually say. That is the whole point of the type: a provider that
    publishes no weighted unit contributes calls and leaves :attr:`provider_units` ``None``,
    which is a different fact from reporting zero.

    Attributes:
        provider: Who was called — ``"search_api"``, ``"video_api"``. ``None`` only when
            the caller genuinely could not say, and it carries no default, so every
            construction states one or the other rather than inheriting an answer. A unit
            count that cannot name whose it is may not be summed with one that can and
            cannot be priced from a per-provider table; both of those follow from the value
            being ``None`` rather than from the field being mandatory, which is why the
            absent case is expressible instead of being spelled ``""`` by whichever caller
            met it first.
        calls: How many provider calls this covers. A tool describing ONE of its own calls
            reports 1; a delivery reporting a batch reports the batch.
        provider_units: The provider's own weighted metering, where the caller can state
            it. ``None`` means no countable unit applies or none was observed — never 0,
            which would claim the calls consumed nothing.
        provider_unit: The **bare** name of that unit, as the provider calls it —
            ``"credits"``, ``"quota_units"``. ``None`` whenever ``provider_units`` is.
            Bare here and qualified by :attr:`qualified_unit`, mirroring the split between
            what a provider declares and what is safe to compare across providers.
        money: What the provider actually charged, in USD, where it reports a charge.
            ``None`` means unreported — which is NOT the same as ``0.0``. A self-hosted
            backend costs a real, observed zero; an unpriced paid provider is a real cost
            nobody declared a rate for. One bucket cannot say which is which, so this field
            has three states and all of them mean something. **Only the delivery seam sets
            it**: background work reports a provider's charge on
            ``AsyncExternalSpend.money``, which reaches here through
            ``AsyncExternalSpend.as_external_spend``. The action seam is dollar-free by rule
            (a replay would report spend it never incurred), and drops it.
    """

    provider: str | None
    calls: int = 0
    provider_units: int | None = None
    provider_unit: str | None = None
    money: float | None = None

    @property
    def qualified_unit(self) -> str | None:
        """This spend's unit, qualified by its provider, or ``None`` if it meters none.

        The single place ``"<provider>:<unit>"`` is composed. A second spelling appearing
        somewhere else would compare unequal to this one, and two spends from the same
        provider would then refuse to combine — the failure the qualification exists to
        prevent, reintroduced by the fix for it.

        Returns:
            ``"<provider>:<unit>"``, or ``None`` when no weighted unit was consumed or no
            provider could be named. An unattributed unit is not comparable to anything, so
            there is no safe qualified form of it to hand out.
        """
        if self.provider_unit is None or self.provider is None:
            return None
        return f"{self.provider}:{self.provider_unit}"
