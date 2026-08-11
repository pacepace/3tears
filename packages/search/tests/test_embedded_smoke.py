"""The embedded smoke: a whole search from a cold one-shot ``asyncio.run()``.

Two tiers, and both matter for different reasons.

**The offline tier** is the one that runs everywhere, and it is the shape
check that no unit test can make: a full search -- standalone transport,
SearXNG adapter, Call, Bind -- from a single ``asyncio.run()`` with no ambient
loop, no broker, no long-lived client, and nothing left open when it returns
(SR-L5, checks 5 and 9). It is a *composition* pin: every piece already has
unit tests, and the thing that breaks is the assembly acquiring a dependency
on a runtime nobody noticed it needed.

**The live tier** is env-gated and skips with a reason. It exists because the
offline tier cannot settle SR-A4's open assumption about what SearXNG's
``score`` actually means, nor prove that a real instance's JSON matches the
fixtures. Gate B closes that against a real instance; this is the test it
closes it with.

**The package never reads the environment** (SR-K1). A *test* may, and this
one does -- the base URL travels from ``SEARXNG_BASE_URL`` into the adapter's
constructor argument, which is exactly the path a host takes.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from threetears.search.adapters.searxng import SEARXNG_PROVIDER, SearxngAdapter
from threetears.search.bind import bind_search
from threetears.search.contracts import (
    SEARCH_RESULTS_METADATA_KEY,
    Criterion,
    SearchRequest,
    SearchResultsMetadata,
    get_capabilities,
)
from threetears.search.standalone import StandaloneTransport
from _http_server import LocalHttpServer, Reply
from _searxng_payloads import TWO_RESULTS_BODY

#: the host-supplied base URL for the live tier. Read by the TEST, never by
#: the package: SR-K1 forbids the package reading ambient config, and the
#: value travels from here into a constructor argument like any other.
LIVE_BASE_URL_VAR = "SEARXNG_BASE_URL"

#: set to ``1`` when the live instance is on the host's own network, which a
#: self-hosted SearXNG usually is (D21 makes that deployment config).
LIVE_ALLOW_PRIVATE_VAR = "SEARXNG_ALLOW_PRIVATE"


def test_a_whole_search_runs_from_one_shot_asyncio_run() -> None:
    """SR-L5: no ambient loop, no broker, no lifecycle, nothing left open."""

    async def _one_search() -> tuple[str, dict[str, object]]:
        async with LocalHttpServer((Reply(body=TWO_RESULTS_BODY),)) as server:
            adapter = SearxngAdapter(
                base_url=server.base_url,
                transport=StandaloneTransport(allow_private_addresses=True, timeout_seconds=5.0),
                provider_instance="searxng-smoke",
            )
            rendered = await bind_search(SearchRequest(query="capybara"), provider=adapter)
            return rendered.content, rendered.metadata

    prose, metadata = asyncio.run(_one_search())

    assert prose.startswith("1. Capybara")
    projection = SearchResultsMetadata.from_metadata(metadata[SEARCH_RESULTS_METADATA_KEY])  # type: ignore[arg-type]
    assert len(projection.candidates) == 2
    assert projection.spend.calls == 1


def test_capabilities_are_queryable_without_constructing_anything() -> None:
    """SR-B4: a consumer branches before it has a base URL or a transport."""
    declared = get_capabilities(SEARXNG_PROVIDER)

    assert declared is not None
    assert declared.disposition_for("language") == "pushdown"


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get(LIVE_BASE_URL_VAR),
    reason=f"live SearXNG tier needs {LIVE_BASE_URL_VAR} set to a reachable instance",
)
def test_a_live_searxng_instance_returns_typed_candidates() -> None:
    """Gate B's pin: real JSON, real scores, from one ``asyncio.run()``.

    Asserts only what a healthy instance must be true of, so it is a signal
    rather than a flake: the call succeeds, the dispositions answer, and every
    candidate that came back is well-formed. It deliberately does NOT require
    a non-empty result set -- an instance whose engines are all rate-limited
    right now answers zero results, and that is a success (SR-J2).
    """
    base_url = os.environ[LIVE_BASE_URL_VAR]
    allow_private = os.environ.get(LIVE_ALLOW_PRIVATE_VAR, "") == "1"

    async def _live() -> tuple[str, SearchResultsMetadata]:
        adapter = SearxngAdapter(
            base_url=base_url,
            transport=StandaloneTransport(timeout_seconds=20.0, allow_private_addresses=allow_private),
            provider_instance=f"live:{base_url}",
        )
        rendered = await bind_search(
            SearchRequest(query="capybara habitat range", criteria=(Criterion.max_results(5),)),
            provider=adapter,
        )
        payload = rendered.metadata[SEARCH_RESULTS_METADATA_KEY]
        assert isinstance(payload, dict)
        return rendered.content, SearchResultsMetadata.from_metadata(payload)

    prose, projection = asyncio.run(_live())

    assert projection.failure is None, f"live search failed: {prose}"
    assert projection.spend.calls == 1
    assert projection.spend.wall_clock_seconds > 0
    assert len(projection.candidates) <= 5, "the max-results cap is applied locally, and it must hold"

    answers = {disposition.criterion_key: disposition.disposition for disposition in projection.dispositions}
    assert answers["max-results"] == "local"

    for candidate in projection.candidates:
        assert candidate.identity.startswith("http")
        assert candidate.locators
        assert candidate.provenance.provider_instance == f"live:{base_url}"
        assert candidate.provenance.retrieved_at.tzinfo is not None
        for score in candidate.scores:
            # SR-A4's open assumption is about what these values MEAN, and a
            # live run is where that gets settled. What is pinned here is that
            # whatever they mean, they arrive named, scaled and marked
            # non-comparable rather than as a bare number.
            assert score.name and score.scale
            assert score.comparable is False
    assert prose
