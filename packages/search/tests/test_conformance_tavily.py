"""Tavily runs the shared provider-conformance suite (SR-O5, §3.11).

The point of the shared suite is the same as SearXNG's: nothing provider-
specific is re-proven here. This module supplies only what only Tavily's own
API can supply -- its JSON shapes and the two criteria it treats two
different ways.

One difference from the SearXNG case is structural rather than incidental:
Tavily declares no ``local`` criterion at all (it either pushes a criterion
to its API or reports it unsatisfied -- SR-B3), so ``local_criterion`` is
left unset. That is a fact about Tavily's own capability declaration, not a
gap in the suite -- see :class:`~threetears.search.testing.conformance.ProviderConformanceCase`.
"""

from __future__ import annotations

from threetears.search.adapters.tavily import TavilyAdapter
from threetears.search.contracts import Criterion, SearchProvider, SearchTransport
from threetears.search.testing import ProviderConformanceCase, ProviderConformanceSuite
from _tavily_payloads import MALFORMED_BODY, TWO_RESULTS_BODY, ZERO_RESULTS_BODY

_API_KEY = "tvly-conformance-key"


def _tavily(transport: SearchTransport) -> SearchProvider:
    """Build the adapter under conformance over an injected transport.

    :param transport: the transport the suite drives the provider through
    :ptype transport: SearchTransport
    :return: the provider under test
    :rtype: SearchProvider
    """
    return TavilyAdapter(api_key=_API_KEY, transport=transport, provider_instance="tavily-conf")


class TestTavilyConformance(ProviderConformanceSuite):
    """Tavily against the shared suite."""

    case = ProviderConformanceCase(
        provider_factory=_tavily,
        success_body=TWO_RESULTS_BODY,
        zero_results_body=ZERO_RESULTS_BODY,
        malformed_body=MALFORMED_BODY,
        # domains-include: Tavily's API takes it, so it is pushed down (as
        # JSON body rather than SearXNG's query string -- the suite's wire
        # check reads either).
        pushdown_criterion=Criterion.domains_include(["example.org"]),
        pushdown_parameter="include_domains",
        # Tavily filters nothing locally: every criterion it can act on goes
        # to the wire, and everything else is reported unsatisfied.
        local_criterion=None,
        # language: Tavily exposes no language filter and reports no
        # language to filter on locally either.
        unsatisfiable_criterion=Criterion.language("en"),
    )
