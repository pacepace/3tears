"""Check 12: two providers, two exits, one process -- and every result says which it left by.

``search-requirements.md`` §3 check 12 is *"a deployment routes search egress
independently of the rest of its traffic, and a result says which exit it left
by (SR-N2)"*, and it names its own hard case: **"the self-hosted SearXNG and the
Tavily API can take different exits in the same process."**

Every egress pin that existed before this module drives **one** provider through
**one** exit -- ``test_call_wiring.py`` asserts the pacing key and the refusal
record carry what Call was handed, and the conformance suite asserts a candidate
names an exit at all. Each of those asserts a value was *carried*. None of them
asserts the property the check is actually about, which is **independence**: that
two exits configured side by side stay apart, in the pacing buckets and on the
results, rather than the second quietly inheriting the first. That is the same
"a value was passed rather than behaviour held" gap the spec has now recorded
three times (§7 Phase 2 items 5 and 6), and it is why this file drives the two
*real* adapters rather than a wiring stub.

**Where a result's exit comes from, which is worth stating because it is not
where a reader would guess.** Call takes ``egress=`` and uses it for the pacing
key (D8's ``(provider instance, egress)``) and for the attribution on a refusal;
it does **not** stamp it onto a candidate. ``Provenance.egress`` is stamped by
the *adapter*, from ``TransportResponse.egress``, which the transport reports for
the exchange it actually made. So the tests below assert against both authorities
deliberately: the transport's, because that is what a result's grounding rests
on, and Call's, because that is what pacing separates on. A deployment wires them
to agree; nothing in the contracts makes them agree, which is the carried gap
§7 Phase 1 item 2 records against a later contracts change.
"""

from __future__ import annotations

import pytest

from threetears.search.adapters.searxng import SearxngAdapter
from threetears.search.adapters.tavily import TavilyAdapter
from threetears.search.aggregate import aggregate
from threetears.search.call import search
from threetears.search.contracts import EGRESS_DIRECT, CandidateSet, SearchRequest
from threetears.search.testing import ScriptedTransport, TransportScript
from threetears.search.testing.fakes import FakeRateLimiterPort
from _searxng_payloads import TWO_RESULTS_BODY as SEARXNG_BODY
from _tavily_payloads import TWO_RESULTS_BODY as TAVILY_BODY

#: The two exits, named so neither is the default. A test where one side is
#: ``direct`` cannot tell "the second exit was carried" from "the second exit
#: fell back to the default", which is the confusion this check exists to catch.
#:
#: These two names are the module's fixture, and every pin below is written to
#: hold *behaviour* against them rather than to restate them -- assertions read
#: ``a != b`` and ``disjoint`` rather than ``== _TAVILY_EGRESS``. The first draft
#: of this file did restate them, and collapsing the two constants to one value
#: left all four tests green: with the expectation derived from the same constant
#: as the configuration, a deployment that routed both providers through one exit
#: would have been reported as passing check 12. ``test_the_fixture_configures
#: _two_genuinely_different_exits`` is the guard that now fails first if anyone
#: collapses them again.
_SEARXNG_EGRESS = "lan-direct"
_TAVILY_EGRESS = "corp-proxy"

_QUERY = "capybara habitat range"


def _searxng_over(egress: str) -> SearxngAdapter:
    """Build the self-hosted adapter behind a transport leaving by ``egress``.

    :param egress: the exit this adapter's transport reports leaving by
    :ptype egress: str
    :return: the adapter, ready to be driven by Call
    :rtype: SearxngAdapter
    """
    return SearxngAdapter(
        base_url="https://searx.internal.example.org",
        transport=ScriptedTransport([TransportScript(body=SEARXNG_BODY)], egress_name=egress),
        provider_instance="searxng-main",
    )


def _tavily_over(egress: str) -> TavilyAdapter:
    """Build the hosted-API adapter behind a transport leaving by ``egress``.

    :param egress: the exit this adapter's transport reports leaving by
    :ptype egress: str
    :return: the adapter, ready to be driven by Call
    :rtype: TavilyAdapter
    """
    return TavilyAdapter(
        api_key="tvly-egress-key",
        transport=ScriptedTransport([TransportScript(body=TAVILY_BODY)], egress_name=egress),
        provider_instance="tavily-main",
    )


async def _both_sides(limiter: FakeRateLimiterPort | None = None) -> tuple[CandidateSet, CandidateSet]:
    """Run one query through both providers, each by its own exit, in this process.

    :param limiter: the pacing seam both calls share, or None to leave them unpaced
    :ptype limiter: FakeRateLimiterPort | None
    :return: the self-hosted set and the hosted-API set, in that order
    :rtype: tuple[CandidateSet, CandidateSet]
    """
    request = SearchRequest(query=_QUERY)
    self_hosted = await search(
        request, provider=_searxng_over(_SEARXNG_EGRESS), limiter=limiter, egress=_SEARXNG_EGRESS
    )
    hosted_api = await search(request, provider=_tavily_over(_TAVILY_EGRESS), limiter=limiter, egress=_TAVILY_EGRESS)
    return self_hosted, hosted_api


def test_the_fixture_configures_two_genuinely_different_exits() -> None:
    """Guard on this module's own fixture, and the reason the rest of it can be trusted.

    Every pin below compares the two sides *to each other*. That is only
    evidence of independence while the two sides were configured differently
    to begin with, so the precondition is asserted rather than assumed --
    this is the test that fails first if the constants are ever collapsed.
    """
    assert _SEARXNG_EGRESS != _TAVILY_EGRESS, "the two sides share an exit, so no pin below proves independence"
    assert EGRESS_DIRECT not in {_SEARXNG_EGRESS, _TAVILY_EGRESS}, (
        "an exit is the default, so a dropped value would be indistinguishable from a carried one"
    )


@pytest.mark.asyncio
async def test_each_provider_leaves_by_its_own_exit_in_one_process() -> None:
    """The check that bites: two exits configured side by side stay apart on the results."""
    self_hosted, hosted_api = await _both_sides()

    assert self_hosted.candidates, "the self-hosted side returned nothing to attribute"
    assert hosted_api.candidates, "the hosted-API side returned nothing to attribute"
    self_hosted_exits = {c.provenance.egress for c in self_hosted.candidates}
    hosted_api_exits = {c.provenance.egress for c in hosted_api.candidates}

    # Disjointness rather than equality against the constants: what check 12
    # asks is that the two sides stayed apart, and a pin restating its own
    # configuration answers a different question (see the module note).
    assert self_hosted_exits.isdisjoint(hosted_api_exits), (
        f"the two providers report overlapping exits ({self_hosted_exits} / {hosted_api_exits}); "
        "one side inherited the other's routing"
    )
    assert len(self_hosted_exits) == 1, "one provider over one transport reported more than one exit"
    assert len(hosted_api_exits) == 1, "one provider over one transport reported more than one exit"


@pytest.mark.asyncio
async def test_each_result_names_the_exit_its_own_transport_served_it_by() -> None:
    """A result's exit is the one the transport actually reported, not one Call was told about.

    Read off each side's transport rather than off a module constant, because
    the transport is the authority the adapter stamps from -- an assertion
    against the constant would agree with the configuration even if the
    stamping were dropped.
    """
    searxng_transport = ScriptedTransport([TransportScript(body=SEARXNG_BODY)], egress_name=_SEARXNG_EGRESS)
    tavily_transport = ScriptedTransport([TransportScript(body=TAVILY_BODY)], egress_name=_TAVILY_EGRESS)
    request = SearchRequest(query=_QUERY)

    self_hosted = await search(
        request,
        provider=SearxngAdapter(
            base_url="https://searx.internal.example.org",
            transport=searxng_transport,
            provider_instance="searxng-main",
        ),
        egress=_SEARXNG_EGRESS,
    )
    hosted_api = await search(
        request,
        provider=TavilyAdapter(api_key="tvly-egress-key", transport=tavily_transport, provider_instance="tavily-main"),
        egress=_TAVILY_EGRESS,
    )

    for candidate in self_hosted.candidates:
        assert candidate.provenance.egress == searxng_transport.egress_name
    for candidate in hosted_api.candidates:
        assert candidate.provenance.egress == tavily_transport.egress_name


@pytest.mark.asyncio
async def test_the_two_exits_pace_on_separate_keys() -> None:
    """D8's ``(provider instance, egress)`` keeps two exits in two buckets, not one.

    A shared limiter is the realistic wiring -- one deployment, one pacing
    authority -- and it is where a collapsed key would actually hurt: the
    self-hosted instance would be throttled by the hosted API's quota. The
    assertion is on the *egress* component specifically, because the two
    provider instances already differ -- a key that carried only the instance
    would separate these two calls while collapsing every pair of exits a
    single provider is reached by, which is the case SR-N2 is really about.
    """
    limiter = FakeRateLimiterPort()

    await _both_sides(limiter)

    keys = [(instance, egress) for instance, egress, _tokens, _budget in limiter.acquisitions]
    assert len(keys) == 2, "both calls should have taken a pacing slot"
    assert keys[0][1] != keys[1][1], f"two exits collapsed onto one pacing bucket: {keys}"
    assert len(set(keys)) == 2


@pytest.mark.asyncio
async def test_both_exits_survive_into_the_corpus() -> None:
    """Aggregate keeps each contribution whole (R1), so the exits stay distinguishable after merging.

    Attribution that only survives as far as Call is attribution a consumer
    cannot act on -- what it holds is the corpus. R1 keeps contributions
    unmerged precisely so a singular ``Provenance`` never has to discard one
    origin, and an exit is one of the facts that would go with it.
    """
    self_hosted, hosted_api = await _both_sides()

    corpus = aggregate([self_hosted, hosted_api])

    exits = {contribution.provenance.egress for entry in corpus.entries for contribution in entry.contributions}
    assert len(exits) == 2, f"the corpus collapsed two exits into {exits}"
