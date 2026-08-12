"""The embedded smoke: a whole search from a cold one-shot ``asyncio.run()``.

Two tiers, and both matter for different reasons.

**The offline tier** is the one that runs everywhere, and it is the shape
check that no unit test can make: a full search -- standalone transport,
SearXNG adapter, Call, Bind -- from a single ``asyncio.run()`` with no ambient
loop, no broker, no long-lived client, and nothing left open when it returns
(SR-L5, checks 5 and 9). It is a *composition* pin: every piece already has
unit tests, and the thing that breaks is the assembly acquiring a dependency
on a runtime nobody noticed it needed.

**The live tier** is env-gated and skips with a reason, one per provider
(§3.11). The SearXNG live test exists because the offline tier cannot prove
that a real instance's JSON matches the fixtures, nor observe what its
``score`` field actually holds. SR-A4's score semantics were settled that way
on 2026-08-12 -- against a live instance plus ``calculate_score`` itself -- and
one residue remains: the unbounded claim rests on the formula rather than on a
capture, because the instance that answered had a single healthy engine. This
test surfaces the weights it sees so a run against healthy engines discharges
that, and ``SEARXNG_REQUIRE_RESULTS`` makes an empty run fail rather than pass
having checked nothing. The Tavily live test is the same shape behind
explicit credentials instead of a self-hosted base URL -- it proves a real
Tavily response still decodes into typed candidates and bills real credits,
without pinning any content a live provider could change query to query.

**The package never reads the environment** (SR-K1). A *test* may, and both
of these do -- the SearXNG base URL travels from ``SEARXNG_BASE_URL`` and the
Tavily API key from ``TAVILY_API_KEY``, each into the adapter's own
constructor argument, which is exactly the path a host takes.
"""

from __future__ import annotations

import asyncio
import os
import warnings

import pytest

from threetears.search.adapters.searxng import SEARXNG_PROVIDER, SearxngAdapter
from threetears.search.adapters.tavily import TavilyAdapter
from threetears.search.bind import bind_search
from threetears.search.contracts import (
    SCALE_RANK,
    SCALE_UNBOUNDED,
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

#: set when the live instance is on the host's own network, which a
#: self-hosted SearXNG usually is (D21 makes that deployment config).
LIVE_ALLOW_PRIVATE_VAR = "SEARXNG_ALLOW_PRIVATE"

#: set to require the live instance to actually return candidates.
#:
#: Zero results is a legitimate success (SR-J2), so this test tolerates it --
#: but the tolerance means every per-candidate assertion below sits inside a
#: loop that runs zero times, and the test passes having checked nothing about
#: a score. That is how it passed on 2026-08-12 against a real instance whose
#: four general engines had all been CAPTCHA'd: the run was green and settled
#: nothing. Set this where the instance is known healthy -- which is what
#: discharging Gate B's SR-A4 line requires -- and the empty run becomes a
#: failure instead of a silent pass.
LIVE_REQUIRE_RESULTS_VAR = "SEARXNG_REQUIRE_RESULTS"

#: the spellings :func:`_flag` accepts, beyond the documented ``1``.
FLAG_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _flag(name: str) -> bool:
    """Read a boolean test flag from the environment, tolerantly.

    ``1`` is the documented spelling, but an operator who exports ``true``
    means the same thing, and for :data:`LIVE_REQUIRE_RESULTS_VAR` an ignored
    value is the exact failure the flag exists to prevent: the run is green
    either way, so a strict ``== "1"`` turns a typo into a Gate B discharge
    that settled nothing, with no signal that anything was ignored. A typo'd
    :data:`LIVE_ALLOW_PRIVATE_VAR` at least fails loudly at connect time.

    :param name: the environment variable to read
    :ptype name: str
    :return: whether the flag is set to any recognised true value
    :rtype: bool
    """
    return os.environ.get(name, "").strip().lower() in FLAG_TRUE_VALUES


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
    candidate that came back is well-formed. By default it does NOT require a
    non-empty result set -- an instance whose engines are all rate-limited
    right now answers zero results, and that is a success (SR-J2).

    Set :data:`LIVE_REQUIRE_RESULTS_VAR` to flip that: against an instance
    known to be healthy, an empty answer means the score assertions below
    checked nothing, and silence there is what let a green run be mistaken for
    evidence once already. That is the mode Gate B's SR-A4 line is discharged
    in, and the engine-fusion weights this run observed are surfaced as a
    warning so the run leaves the capture behind rather than just a pass.
    """
    base_url = os.environ[LIVE_BASE_URL_VAR]
    allow_private = _flag(LIVE_ALLOW_PRIVATE_VAR)
    require_results = _flag(LIVE_REQUIRE_RESULTS_VAR)

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

    if require_results:
        assert projection.candidates, (
            "the instance answered zero results, so every score assertion below was skipped; "
            f"unset {LIVE_REQUIRE_RESULTS_VAR} to tolerate that, or check the instance's engines"
        )

    observed_weights: list[float] = []
    for candidate in projection.candidates:
        assert candidate.identity.startswith("http")
        assert candidate.locators
        assert candidate.provenance.provider_instance == f"live:{base_url}"
        assert candidate.provenance.retrieved_at.tzinfo is not None
        for score in candidate.scores:
            # SR-A4 is settled (2026-08-12): the engine-fusion weight is
            # ``sum(weight/position)`` with ``weight`` scaled by the number of
            # agreeing engines, so it is unbounded above and deployment-
            # dependent -- see SearxngAdapter._scores. The scales are constants
            # in the adapter and are pinned offline, so re-asserting them here
            # settles nothing; what only a live run can supply is the values.
            assert score.name and score.scale
            assert score.comparable is False
            if score.name == "engine-fusion-weight":
                assert score.scale == SCALE_UNBOUNDED
                # every term of the sum is positive, and the floor is the 0 a
                # ``priority: low`` engine assigns -- which is a judgment, not
                # an absence, so it must arrive rather than being dropped.
                assert score.value >= 0.0, f"a fused weight cannot be negative, got {score.value}"
                observed_weights.append(score.value)
            elif score.name == "best-engine-position":
                assert score.scale == SCALE_RANK
                assert score.value >= 1.0, f"positions are 1-based ranks, got {score.value}"

    if observed_weights:
        # the residue SR-A4 still owes is a *capture*: the unbounded claim
        # (two agreeing engines score 4.0, three 9.0) rests on the formula,
        # because the one live instance available had a single healthy engine
        # and single-engine data tops out at 1.0. A weight past 1.0 here is
        # the observation §12 is waiting for, so the run reports what it saw
        # instead of asserting a bound nobody can promise.
        warnings.warn(
            f"[SR-A4] live engine-fusion weights, high to low: {sorted(observed_weights, reverse=True)}",
            stacklevel=1,
        )
    if require_results:
        assert observed_weights, (
            "the instance returned candidates but not one engine-fusion weight, so the live "
            "SR-A4 pin observed no score at all; check the instance's engines"
        )
    assert prose


#: the host-supplied API key for the live Tavily tier. Read by the TEST,
#: never by the package: same SR-K1 boundary as LIVE_BASE_URL_VAR above --
#: TavilyAdapter refuses to construct with no key rather than reach for one.
LIVE_TAVILY_API_KEY_VAR = "TAVILY_API_KEY"


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get(LIVE_TAVILY_API_KEY_VAR),
    reason=f"live Tavily tier needs {LIVE_TAVILY_API_KEY_VAR} set to a valid API key",
)
def test_a_live_tavily_instance_returns_typed_candidates() -> None:
    """Gate B's Tavily pin: real JSON, real credits, from one ``asyncio.run()``.

    Behind explicit credentials (§3.11), the way the SearXNG live tier sits
    behind a self-hosted instance. Asserts only what a healthy call must be
    true of -- typed candidates, credits actually spent, dispositions that
    answer -- and deliberately not provider content or a non-empty result
    set, either of which would make this a flake tied to what Tavily's index
    happens to hold today rather than a signal about the adapter.
    """
    api_key = os.environ[LIVE_TAVILY_API_KEY_VAR]

    async def _live() -> tuple[str, SearchResultsMetadata]:
        adapter = TavilyAdapter(
            api_key=api_key,
            transport=StandaloneTransport(timeout_seconds=20.0),
            provider_instance="live:tavily",
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
    assert projection.spend.provider_units > 0, "a successful Tavily call always bills at least one credit (SR-E4)"
    assert len(projection.candidates) <= 5, "the max-results cap is honored, whether pushed down or applied locally"

    answers = {disposition.criterion_key: disposition.disposition for disposition in projection.dispositions}
    assert answers["max-results"] == "pushdown", "Tavily takes max_results as a request parameter (§3.11)"

    for candidate in projection.candidates:
        assert candidate.identity
        assert candidate.locators
        assert candidate.provenance.provider_instance == "live:tavily"
        assert candidate.provenance.retrieved_at.tzinfo is not None
    assert prose
