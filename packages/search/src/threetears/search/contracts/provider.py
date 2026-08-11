"""SearchProvider -- the seam between one provider's API and everything above.

Structural, injected, and named for what it is rather than for the layer
that implements it (search-spec.md §2 forbids layer names as type names).
An implementation wraps exactly one provider's API, reached through an
injected :class:`~threetears.search.contracts.transport.SearchTransport`
and nothing else.

Two consumers depend on this shape and neither imports a concrete provider:

- Call (``threetears.search.call``) turns a request into one candidate set
  through whichever provider it was handed;
- the conformance suite (``threetears.search.testing``) runs the same five
  pins against every provider that claims to satisfy it (SR-O5).

What an implementation owes:

- **capabilities before contact** -- :attr:`SearchProvider.capabilities`
  answers what the provider can express without a request being made
  (SR-B4);
- **dispositions with the results** -- one honest answer per criterion the
  request carried (SR-B2, SR-B3), matching what the capabilities declared;
- **spend on every outcome** -- attached to the returned set and to every
  typed failure it raises (SR-E1, SR-E3);
- **the typed taxonomy** -- provider and transport failures mapped onto
  :mod:`threetears.search.contracts.errors`, because the transport raises
  whatever it likes;
- **zero results as a success** -- an empty
  :class:`~threetears.search.contracts.candidate.CandidateSet`, never an
  exception (SR-J2);
- **host-supplied configuration** -- base URL and credentials come from
  the deployment, never from the environment (D21, SR-K1).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from threetears.search.contracts.candidate import CandidateSet
from threetears.search.contracts.capabilities import ProviderCapabilities
from threetears.search.contracts.request import SearchRequest

__all__ = ["SearchProvider"]


@runtime_checkable
class SearchProvider(Protocol):
    """One provider's API, behind one query method."""

    @property
    def provider(self) -> str:
        """Product name of the provider this implementation speaks to.

        :return: the product name (``searxng``, ``tavily``) -- the key its
            capabilities are registered under
        :rtype: str
        """
        ...

    @property
    def provider_instance(self) -> str:
        """Name of the configured deployment this implementation reaches.

        An *instance* name: two SearXNG deployments are two instances of
        one product, they get banned and rate-limited separately (D8,
        SR-N4), and provenance records which one answered (SR-A3).

        :return: the instance name
        :rtype: str
        """
        ...

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Declare what this provider can express, before anything is sent.

        :return: the provider's own capability declaration (SR-B4)
        :rtype: ProviderCapabilities
        """
        ...

    async def search(self, request: SearchRequest, *, timeout_seconds: float | None = None) -> CandidateSet:
        """Retrieve one candidate set for ``request``.

        :param request: what the caller asked for
        :ptype request: SearchRequest
        :param timeout_seconds: bound for this call, derived from the
            caller's remaining deadline where there is one (SR-G2); None
            leaves the transport's configured timeout in force (SR-G1)
        :ptype timeout_seconds: float | None
        :return: the candidates, one disposition per criterion, and the
            spend the call consumed. Zero candidates is a success (SR-J2)
        :rtype: CandidateSet
        :raises threetears.search.contracts.errors.SearchFailure: one of
            the typed classes, carrying the spend consumed before the
            failure (SR-E3, SR-J1)
        """
        ...
