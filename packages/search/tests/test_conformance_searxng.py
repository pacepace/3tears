"""SearXNG runs the shared provider-conformance suite (SR-O5, §3.11).

The whole value of the suite is that it is not written here: the same five
pins will run against Tavily, and against any adapter a consumer writes, so
a consumer swapping providers keeps the guarantees it was relying on. All this
module supplies is what only SearXNG's own API can supply -- its JSON shapes
and three criteria it treats three different ways.

That the criteria are only *named* here and their dispositions are checked
against the adapter's own capability declaration is deliberate: a case that
restated the expected dispositions could agree with itself while disagreeing
with the adapter.
"""

from __future__ import annotations

from datetime import UTC, datetime

from threetears.search.adapters.searxng import SearxngAdapter
from threetears.search.contracts import Criterion, SearchProvider, SearchTransport
from threetears.search.testing import ProviderConformanceCase, ProviderConformanceSuite
from _searxng_payloads import MALFORMED_BODY, TWO_RESULTS_BODY, ZERO_RESULTS_BODY


def _searxng(transport: SearchTransport) -> SearchProvider:
    """Build the adapter under conformance over an injected transport.

    :param transport: the transport the suite drives the provider through
    :ptype transport: SearchTransport
    :return: the provider under test
    :rtype: SearchProvider
    """
    return SearxngAdapter(base_url="https://searx.example.org", transport=transport, provider_instance="searxng-conf")


class TestSearxngConformance(ProviderConformanceSuite):
    """SearXNG against the shared suite."""

    case = ProviderConformanceCase(
        provider_factory=_searxng,
        success_body=TWO_RESULTS_BODY,
        zero_results_body=ZERO_RESULTS_BODY,
        malformed_body=MALFORMED_BODY,
        # language: SearXNG's API takes it, so it is pushed down.
        pushdown_criterion=Criterion.language("en"),
        pushdown_parameter="language",
        # max-results: SearXNG has no such parameter, so the adapter caps
        # after parsing -- and having declared 'local', it must actually do it.
        local_criterion=Criterion.max_results(5),
        # time-range: absolute, and SearXNG expresses only relative windows.
        unsatisfiable_criterion=Criterion.time_range(start=datetime(2026, 1, 1, tzinfo=UTC)),
    )
