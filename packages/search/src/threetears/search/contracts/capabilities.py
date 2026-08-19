"""Declared, queryable provider capabilities (SR-B4).

A consumer must be able to branch *before* sending rather than after
failing: SearXNG has no domain allow-list and no search depth, Tavily has
both and no engine list, and a caller that learns this from a 400 has
already paid for the round trip. So every provider declares what it can
express, and the declaration is a queryable value rather than prose in a
docstring.

The pattern is ``3tears-models``' capability metadata, deliberately: one
wide model with optional fields sectioned by concern, plus a module-level
registry each provider populates at import time so a consumer can query
without constructing a provider. What differs is what the fields *are* --
this registry answers "which criteria can you honour, and how", because
that is the question the criteria vocabulary (SR-B1) forces every layer
above to ask.

The three criterion tuples are the load-bearing part. Together they are the
provider's own statement of the dispositions (SR-B2) its results will
carry, which is what lets Call guarantee an answer for every criterion the
caller sent -- even one the adapter forgot to answer for.

Named for what it is, not for the layer that owns it: ``Adapter`` is module
vocabulary and never a type name (search-spec.md §2).
"""

from __future__ import annotations

import threading
from typing import Final

from pydantic import model_validator

from threetears.search.contracts._base import ContractModel
from threetears.search.contracts.criteria import Disposition

__all__ = [
    "PRICING_FREE_SELF_HOSTED",
    "PRICING_PER_REQUEST",
    "PRICING_PER_WEIGHTED_UNIT",
    "ProviderCapabilities",
    "get_capabilities",
    "list_capabilities",
    "register_capabilities",
]

#: pricing model: the provider is self-hosted and bills nothing, so the real
#: constraint is rate and ban rather than money (D6, SR-D6).
PRICING_FREE_SELF_HOSTED: Final[str] = "free-self-hosted"

#: pricing model: one charge per request, whatever the result count (SR-E5).
PRICING_PER_REQUEST: Final[str] = "per-request"

#: pricing model: charged in provider-defined units whose weight varies by
#: request shape (Tavily's ``advanced`` = 2 credits -- SR-E4).
PRICING_PER_WEIGHTED_UNIT: Final[str] = "per-weighted-unit"


class ProviderCapabilities(ContractModel):
    """What one provider can express, as the provider itself declares it.

    Every field beyond ``provider`` is optional, and ``None`` means "this
    concern does not apply to me" rather than "no": a provider with no
    engine list leaves ``engines`` unset, and a reader must not infer that
    its engine list is empty.
    """

    #: product name -- ``searxng``, ``tavily``. Not an instance name: two
    #: SearXNG deployments are two instances of one set of capabilities.
    provider: str

    # --- criteria negotiation (SR-B4, SR-B2) -------------------------------
    #: well-known criterion keys the provider's own API expresses, so the
    #: constraint is pushed down rather than applied here.
    pushdown_criteria: tuple[str, ...] = ()
    #: well-known criterion keys the adapter honours by filtering what came
    #: back. Declaring one is a promise that the filtering happens.
    local_criteria: tuple[str, ...] = ()
    #: well-known criterion keys this provider cannot honour at all. Named,
    #: never silently dropped (SR-B3) -- a caller reads this and decides.
    unsatisfiable_criteria: tuple[str, ...] = ()
    #: ``<namespace>:<name>`` criterion keys the adapter accepts for
    #: provider-specific parameters this vocabulary does not own.
    namespaced_parameters: tuple[str, ...] = ()

    # --- result shape ------------------------------------------------------
    #: whether the provider pages, so a caller wanting depth knows whether
    #: asking for it is possible.
    supports_paging: bool | None = None
    #: how many candidates one request can yield, when the provider bounds
    #: it. ``None`` means unbounded or undeclared.
    max_results_per_page: int | None = None

    # --- provider vocabularies (open; a caller builds namespaced params
    #     from these rather than guessing at accepted values) --------------
    #: result categories the provider recognises (SearXNG's ``categories``).
    categories: tuple[str, ...] | None = None
    #: named engines the provider can be restricted to (SearXNG).
    engines: tuple[str, ...] | None = None
    #: accepted safesearch levels (SearXNG's ``0``/``1``/``2``).
    safesearch_levels: tuple[int, ...] | None = None
    #: relative publication windows the provider accepts. Distinct from the
    #: ``time-range`` criterion, which is *absolute*: a provider offering
    #: only these declares ``time-range`` unsatisfiable and accepts the
    #: relative window as a namespaced parameter.
    relative_time_ranges: tuple[str, ...] | None = None
    #: retrieval depths, where the provider sells more than one (Tavily).
    search_depths: tuple[str, ...] | None = None
    #: result topics, where the provider has them (Tavily).
    topics: tuple[str, ...] | None = None

    # --- spend shape -------------------------------------------------------
    #: how this provider charges: one of :data:`PRICING_FREE_SELF_HOSTED`,
    #: :data:`PRICING_PER_REQUEST`, :data:`PRICING_PER_WEIGHTED_UNIT`. Open
    #: vocabulary with named well-known values.
    pricing_model: str | None = None
    #: the name of the unit a :data:`PRICING_PER_WEIGHTED_UNIT` provider
    #: meters in -- Tavily's ``credits``. Bare here, because this declares
    #: what the provider calls its own unit; :attr:`Spend.provider_unit`
    #: qualifies it with the provider name, since two providers both saying
    #: "credits" do not have one fungible unit between them.
    #:
    #: ``None`` on a provider that meters no weighted unit, which is every
    #: :data:`PRICING_FREE_SELF_HOSTED` and :data:`PRICING_PER_REQUEST`
    #: provider. Declaring a unit while charging per request would promise a
    #: count nothing produces.
    metered_unit: str | None = None

    @property
    def qualified_unit(self) -> str | None:
        """This provider's metered unit, qualified for :attr:`Spend.provider_unit`.

        The one place ``"<provider>:<unit>"`` is composed, so a second
        spelling cannot appear beside the first and compare unequal --
        which would make two spends from the same provider refuse to sum.

        :return: ``"<provider>:<metered_unit>"``, or ``None`` when this
            provider meters no weighted unit
        :rtype: str | None
        """
        if self.metered_unit is None:
            return None
        return f"{self.provider}:{self.metered_unit}"

    @model_validator(mode="after")
    def _criteria_declarations_do_not_contradict(self) -> ProviderCapabilities:
        """Refuse a key claimed under two different dispositions.

        A criterion cannot be both pushed down and unsatisfiable. A
        contradictory declaration would make :meth:`disposition_for`
        answer by tuple order, which is not an answer.

        :return: the validated capabilities
        :rtype: ProviderCapabilities
        :raises ValueError: when a key appears in more than one of the
            three well-known criterion tuples
        """
        buckets = (self.pushdown_criteria, self.local_criteria, self.unsatisfiable_criteria)
        seen: dict[str, int] = {}
        clashes: list[str] = []
        for index, bucket in enumerate(buckets):
            for key in bucket:
                if key in seen and seen[key] != index:
                    clashes.append(key)
                seen[key] = index
        if clashes:
            raise ValueError(
                f"provider {self.provider!r} declares {sorted(set(clashes))} under more than one "
                f"disposition; a criterion has exactly one"
            )
        return self

    def disposition_for(self, criterion_key: str) -> Disposition:
        """Answer how this provider would handle ``criterion_key``.

        This is the branch-before-sending query of SR-B4, and the fallback
        Call uses to guarantee SR-B2's one-answer-per-criterion when an
        adapter's own reporting is incomplete.

        :param criterion_key: a well-known or ``<namespace>:<name>`` key
        :ptype criterion_key: str
        :return: the disposition this provider declares for the key;
            ``ignored-unknown`` for anything it has not declared
        :rtype: Disposition
        """
        if criterion_key in self.pushdown_criteria or criterion_key in self.namespaced_parameters:
            return "pushdown"
        if criterion_key in self.local_criteria:
            return "local"
        if criterion_key in self.unsatisfiable_criteria:
            return "unsatisfied"
        return "ignored-unknown"


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: dict[str, ProviderCapabilities] = {}


def register_capabilities(capabilities: ProviderCapabilities) -> None:
    """Record ``capabilities`` under its own provider name.

    Adapters call this at import time, following the ``3tears-models``
    precedent, so a consumer can query a provider's shape without
    constructing one (which would need a base URL and a transport).
    Re-registering a provider replaces the previous entry wholesale.

    :param capabilities: the provider's declaration
    :ptype capabilities: ProviderCapabilities
    """
    with _REGISTRY_LOCK:
        _REGISTRY[capabilities.provider] = capabilities


def get_capabilities(provider: str) -> ProviderCapabilities | None:
    """Return the registered declaration for ``provider``, if any.

    ``None`` means no adapter for that provider has been imported -- which
    is a different fact from "the provider can do nothing", so it is not
    flattened into an empty declaration.

    :param provider: product name (``searxng``, ``tavily``)
    :ptype provider: str
    :return: the declaration, or ``None`` when nothing registered it
    :rtype: ProviderCapabilities | None
    """
    with _REGISTRY_LOCK:
        return _REGISTRY.get(provider)


def list_capabilities() -> dict[str, ProviderCapabilities]:
    """Return every registered declaration, keyed by provider name.

    :return: a copy of the registry -- mutating it changes nothing
    :rtype: dict[str, ProviderCapabilities]
    """
    with _REGISTRY_LOCK:
        return dict(_REGISTRY)
