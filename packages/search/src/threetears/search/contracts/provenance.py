"""Provenance -- where a candidate came from, as recorded fact (SR-A3, P2).

Every candidate carries one. The grounding question a consumer eventually
asks -- "does this claim appear on the page it was cited from" -- is
per-result, and no aggregate can answer it after the fact.

``egress`` follows D20/SR-N2: which exit the request left by is provenance
on every result, and ``direct`` is a *named value*, never an absence --
rate/ban budgets key on ``(provider instance, egress)`` (D8) and replay
comparability depends on it.

``producer`` follows D3: model-mediated candidate production arrives later
at Aggregate, and provenance keeps the producer classes distinct from day
one so a model-mediated candidate can never impersonate an API provider.
The vocabulary is open (a string with named well-known values), matching
the criteria discipline.
"""

from __future__ import annotations

from typing import Final

from pydantic import AwareDatetime, Field

from threetears.search.contracts._base import ContractModel

__all__ = [
    "EGRESS_DIRECT",
    "PRODUCER_API_PROVIDER",
    "PRODUCER_MODEL_MEDIATED",
    "Provenance",
]

#: the named egress value for a request that left by the default route --
#: an exit like any other, never an empty field (D20).
EGRESS_DIRECT: Final[str] = "direct"

#: producer class: a search provider's API returned this candidate.
PRODUCER_API_PROVIDER: Final[str] = "api-provider"

#: producer class: a model-mediated retrieval path produced this candidate
#: (D3 -- enters at Aggregate, never at Adapter or Call).
PRODUCER_MODEL_MEDIATED: Final[str] = "model-mediated"


class Provenance(ContractModel):
    """The recorded origin of one candidate.

    Timestamps are timezone-aware by construction; a naive retrieval time
    is rejected at the border rather than misread later.
    """

    #: the query text as sent for this retrieval. Queries are user content
    #: (D11): available here for the consumer's redaction policy, which
    #: stays with the consumer.
    query: str
    #: which configured provider instance answered -- an *instance* name
    #: (two SearXNG deployments are two instances), not a product name.
    provider_instance: str
    #: the provider's own identifiers for this result, keyed by the
    #: provider's field name (e.g. engine attribution, native result ids).
    provider_ids: dict[str, str] = Field(default_factory=dict)
    #: when the result was retrieved (timezone-aware).
    retrieved_at: AwareDatetime
    #: which egress the retrieving request left by (D20). ``direct`` is
    #: :data:`EGRESS_DIRECT`, a value like any other.
    egress: str = EGRESS_DIRECT
    #: producer class -- :data:`PRODUCER_API_PROVIDER` for adapter-made
    #: candidates; open vocabulary with named well-known values (D3).
    producer: str = PRODUCER_API_PROVIDER
