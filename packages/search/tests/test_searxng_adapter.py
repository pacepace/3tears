"""The SearXNG adapter: pushdown, honesty, typed shape, typed failures.

Everything here runs through the injected transport, which is the point of
the seam: no network, no fixture server, and the wire parameters are
inspectable, so a disposition claiming ``pushdown`` can be checked against
what actually went out.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import JsonValue

from threetears.search.adapters.searxng import (
    SEARXNG_403_REMEDIATION,
    SEARXNG_CAPABILITIES,
    SEARXNG_PARAM_CATEGORIES,
    SEARXNG_PARAM_ENGINES,
    SEARXNG_PARAM_PAGE,
    SEARXNG_PARAM_SAFESEARCH,
    SEARXNG_PARAM_TIME_RANGE,
    SearxngAdapter,
)
from threetears.search.contracts import (
    CRITERION_CARRIER,
    CRITERION_DOMAINS_INCLUDE,
    CRITERION_LANGUAGE,
    CRITERION_MAX_RESULTS,
    WELL_KNOWN_CRITERIA,
    FACET_HAS_DOWNLOADABLE_DATA,
    FACET_HEIGHT,
    FACET_LOCATOR_KIND,
    FACET_MEDIA_CATEGORY,
    FACET_RIGHTS_STATUS,
    FACET_WIDTH,
    FIDELITY_BYTES,
    FIDELITY_SNIPPET,
    PRODUCER_API_PROVIDER,
    SCALE_RANK,
    SCALE_UNBOUNDED,
    AuthFailed,
    Criterion,
    LocalCapExceeded,
    MalformedResponse,
    QuotaExhausted,
    RateLimited,
    SearchProvider,
    SearchRequest,
    SearchTransport,
    Spend,
    TimedOut,
    TransportFailed,
)
from threetears.search.testing import ScriptedTransport, TransportScript
from _searxng_payloads import IMAGE_RESULT, MALFORMED_BODY, TWO_RESULTS_BODY, WEB_RESULT, ZERO_RESULTS_BODY, body

BASE_URL = "https://searx.example.org"


def _adapter(transport: SearchTransport, **kwargs: object) -> SearxngAdapter:
    """Build an adapter over ``transport`` against a fixed base URL.

    :param transport: the injected transport
    :ptype transport: SearchTransport
    :param kwargs: constructor overrides
    :ptype kwargs: object
    :return: the adapter under test
    :rtype: SearxngAdapter
    """
    return SearxngAdapter(base_url=BASE_URL, transport=transport, **kwargs)  # type: ignore[arg-type]


def _scripted(*steps: TransportScript) -> tuple[SearxngAdapter, ScriptedTransport]:
    """Build an adapter over a scripted transport.

    :param steps: the exchanges the transport answers with
    :ptype steps: TransportScript
    :return: the adapter and its transport
    :rtype: tuple[SearxngAdapter, ScriptedTransport]
    """
    transport = ScriptedTransport(steps)
    return _adapter(transport), transport


def test_the_adapter_satisfies_the_provider_seam_by_shape() -> None:
    """P9: the seam is structural -- no base class, no registration."""
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    assert isinstance(adapter, SearchProvider)


def test_a_caller_supplied_base_url_of_the_wrong_shape_is_refused() -> None:
    """D21: base URLs come from deployment config, and are validated here."""
    transport = ScriptedTransport((TransportScript(),))
    for bad in ("file:///etc/passwd", "/search", "searx.example.org", "gopher://searx.example.org"):
        with pytest.raises(ValueError, match="absolute http"):
            SearxngAdapter(base_url=bad, transport=transport)


def test_the_instance_name_defaults_to_the_configured_host() -> None:
    """Two deployments are two instances without anyone naming them."""
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    assert adapter.provider_instance == "searx.example.org"
    named = _adapter(ScriptedTransport((TransportScript(),)), provider_instance="searxng-main")
    assert named.provider_instance == "searxng-main"


async def test_a_bare_query_asks_for_json_and_nothing_else() -> None:
    """The request carries what the query needs and no invented defaults."""
    adapter, transport = _scripted(TransportScript(body=TWO_RESULTS_BODY))
    await adapter.search(SearchRequest(query="capybara"))

    call = transport.calls[-1]
    assert call["url"] == f"{BASE_URL}/search"
    assert call["method"] == "GET"
    assert call["params"] == {"q": "capybara", "format": "json"}


async def test_host_supplied_credentials_ride_the_request_headers() -> None:
    """Credentials come from the host, already resolved (SR-K1)."""
    transport = ScriptedTransport((TransportScript(body=ZERO_RESULTS_BODY),))
    adapter = _adapter(transport, credentials={"Authorization": "Bearer opaque"})
    await adapter.search(SearchRequest(query="capybara"))

    headers = transport.calls[-1]["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer opaque"
    assert headers["Accept"] == "application/json"


async def test_deployment_defaults_are_sent_when_the_request_states_none() -> None:
    """Instance-wide configuration is the host's, and it reaches the wire."""
    transport = ScriptedTransport((TransportScript(body=ZERO_RESULTS_BODY),))
    adapter = _adapter(
        transport, default_categories=("general", "news"), default_engines=("duckduckgo",), default_safesearch=1
    )
    await adapter.search(SearchRequest(query="capybara"))

    params = transport.calls[-1]["params"]
    assert isinstance(params, dict)
    assert params["categories"] == "general,news"
    assert params["engines"] == "duckduckgo"
    assert params["safesearch"] == "1"


def test_an_undeclared_safesearch_level_is_refused_at_construction() -> None:
    """A misconfiguration fails where it is written, not on the first query."""
    with pytest.raises(ValueError, match="safesearch"):
        _adapter(ScriptedTransport((TransportScript(),)), default_safesearch=7)


# --- the typed shape ------------------------------------------------------


async def test_a_web_result_becomes_a_fully_typed_candidate() -> None:
    """Everything the provider returned survives in typed form (§3.2)."""
    adapter, _ = _scripted(TransportScript(body=body((WEB_RESULT,))))
    result = await adapter.search(SearchRequest(query="capybara"))

    candidate = result.candidates[0]
    assert candidate.identity == WEB_RESULT["url"]
    assert candidate.title == "Capybara"
    assert candidate.snippet == WEB_RESULT["content"]
    assert [locator.rel for locator in candidate.locators] == ["canonical"]
    assert candidate.fidelity_available == FIDELITY_SNIPPET
    assert candidate.fidelity_achieved == FIDELITY_SNIPPET
    assert candidate.content is None, "SearXNG supplies no page content, so the slot stays empty (SR-A2)"
    assert candidate.provenance.producer == PRODUCER_API_PROVIDER
    assert candidate.provenance.provider_ids["engine"] == "duckduckgo"
    assert candidate.provenance.provider_ids["engines"] == "duckduckgo,brave"
    assert candidate.provenance.provider_ids["positions"] == "1,3"


async def test_a_naive_published_date_is_read_as_utc_and_kept_raw() -> None:
    """The zone assumption is stated and the provider's own string survives."""
    adapter, _ = _scripted(TransportScript(body=body((WEB_RESULT,))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    assert candidate.published_at == datetime(2026, 2, 1, tzinfo=UTC)
    assert candidate.provenance.provider_ids["published_date_raw"] == "2026-02-01T00:00:00"


async def test_an_unparseable_published_date_invents_nothing() -> None:
    """A date nobody can read leaves the field unset rather than guessed."""
    adapter, _ = _scripted(TransportScript(body=body(({**WEB_RESULT, "publishedDate": "last tuesday"},))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    assert candidate.published_at is None
    assert candidate.provenance.provider_ids["published_date_raw"] == "last tuesday"


async def test_scores_are_named_scaled_and_non_comparable() -> None:
    """D1/SR-A4: a set of named judgments, never one 'score' field."""
    adapter, _ = _scripted(TransportScript(body=body((WEB_RESULT,))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    by_name = {score.name: score for score in candidate.scores}
    # read off the fixture rather than restated: the adapter passes the weight
    # through, so a literal here is a second place for the value to drift from
    # what SearXNG actually computes (it held 2.5 against positions [1, 3],
    # which the real formula puts at 2.667 -- see _searxng_payloads).
    assert by_name["engine-fusion-weight"].value == WEB_RESULT["score"]
    assert by_name["engine-fusion-weight"].scale == SCALE_UNBOUNDED
    assert by_name["best-engine-position"].value == min(WEB_RESULT["positions"])
    assert by_name["best-engine-position"].scale == SCALE_RANK
    assert all(score.comparable is False for score in candidate.scores)
    assert all(score.source == "searx.example.org" for score in candidate.scores)


async def test_a_zero_score_is_reported_rather_than_dropped() -> None:
    """SR-A4: 0 is a judgment, not an absence -- a ``priority: low`` engine scores every result 0.

    So a consumer culling on ``score > 0`` would drop results the deployment
    deliberately included, and it can only know that if the zero survives to
    the border. Pinned because the natural truthiness check here would eat it.
    """
    adapter, _ = _scripted(TransportScript(body=body(({**WEB_RESULT, "score": 0.0},))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    by_name = {score.name: score for score in candidate.scores}
    assert "engine-fusion-weight" in by_name
    assert by_name["engine-fusion-weight"].value == 0.0


async def test_a_missing_score_is_an_absence_not_a_zero() -> None:
    """The other side of it: no reported score means no entry, never a fabricated 0."""
    without_score = {key: value for key, value in WEB_RESULT.items() if key != "score"}
    adapter, _ = _scripted(TransportScript(body=body((without_score,))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    assert "engine-fusion-weight" not in {score.name for score in candidate.scores}


async def test_an_image_result_carries_the_media_facets() -> None:
    """SR-C3: carrier detail rides facets keyed by the media vocabulary."""
    adapter, _ = _scripted(TransportScript(body=body((IMAGE_RESULT,))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    rels = {locator.rel: locator.url for locator in candidate.locators}
    assert rels["canonical"] == IMAGE_RESULT["url"]
    assert rels["direct-file"] == IMAGE_RESULT["img_src"]
    assert rels["thumbnail"] == IMAGE_RESULT["thumbnail_src"]
    assert candidate.fidelity_available == FIDELITY_BYTES
    assert candidate.facets[FACET_MEDIA_CATEGORY] == "image"
    assert candidate.facets[FACET_HAS_DOWNLOADABLE_DATA] is True
    assert candidate.facets[FACET_LOCATOR_KIND] == "direct-file"
    assert candidate.facets[FACET_WIDTH] == 1920
    assert candidate.facets[FACET_HEIGHT] == 1080
    assert candidate.facets[FACET_RIGHTS_STATUS] == "CC BY 2.0"


async def test_a_page_result_says_its_locator_is_a_containing_page() -> None:
    """The direct-file / containing-page distinction is always answered."""
    adapter, _ = _scripted(TransportScript(body=body((WEB_RESULT,))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    assert candidate.facets[FACET_LOCATOR_KIND] == "containing-page"
    assert candidate.facets[FACET_HAS_DOWNLOADABLE_DATA] is False


async def test_the_egress_the_request_left_by_is_on_every_candidate() -> None:
    """D20: egress is provenance, and a named value rather than an absence."""
    transport = ScriptedTransport((TransportScript(body=TWO_RESULTS_BODY),), egress_name="warp")
    result = await _adapter(transport).search(SearchRequest(query="capybara"))

    assert {candidate.provenance.egress for candidate in result.candidates} == {"warp"}


# --- spend ----------------------------------------------------------------


async def test_spend_is_attached_to_a_successful_call() -> None:
    """SR-E1/D6: five dimensions, and a self-hosted instance bills nothing."""
    adapter, _ = _scripted(TransportScript(body=TWO_RESULTS_BODY, elapsed_seconds=0.25))
    result = await adapter.search(SearchRequest(query="capybara"))

    assert result.spend.money == Decimal("0")
    assert result.spend.provider_units == Decimal("0")
    assert result.spend.calls == 1
    assert result.spend.wall_clock_seconds == 0.25
    assert result.spend.bytes_transferred == len(TWO_RESULTS_BODY)


async def test_a_failure_that_never_reached_the_provider_counts_no_calls() -> None:
    """D4: budget follows the bill, and nothing billed here."""
    adapter, _ = _scripted(TransportScript(failure=OSError("connection refused")))
    with pytest.raises(TransportFailed) as raised:
        await adapter.search(SearchRequest(query="capybara"))

    assert raised.value.spend.calls == 0


async def test_a_served_failure_counts_the_call_it_consumed() -> None:
    """SR-E2/SR-E3: the provider answered, so the exchange is accounted."""
    adapter, _ = _scripted(TransportScript(status_code=429, body=b"slow down", elapsed_seconds=0.1))
    with pytest.raises(RateLimited) as raised:
        await adapter.search(SearchRequest(query="capybara"))

    assert raised.value.spend.calls == 1
    assert raised.value.spend.bytes_transferred == len(b"slow down")


# --- the taxonomy ---------------------------------------------------------


async def test_a_403_teaches_the_json_formats_fix() -> None:
    """The single most common SearXNG setup failure ships its remediation."""
    adapter, _ = _scripted(TransportScript(status_code=403, body=b""))
    with pytest.raises(MalformedResponse) as raised:
        await adapter.search(SearchRequest(query="capybara"))

    assert raised.value.remediation == SEARXNG_403_REMEDIATION
    assert "search.formats" in (raised.value.remediation or "")


async def test_a_429_carries_the_providers_stated_backoff() -> None:
    """A pacing refusal is only actionable with the interval it named."""
    adapter, _ = _scripted(TransportScript(status_code=429, body=b"", headers={"retry-after": "30"}))
    with pytest.raises(RateLimited) as raised:
        await adapter.search(SearchRequest(query="capybara"))

    assert raised.value.retry_after_seconds == 30.0


async def test_a_failure_is_fully_attributed_for_pacing_keys() -> None:
    """D8/D20: rate and ban budgets key on (provider instance, egress), and
    pod-resident the failure record on ToolResult.metadata is the only fact
    a consumer-side tracker can rebuild that key from -- so every failure
    leaves the adapter carrying instance, egress and occurrence time."""
    adapter, transport = _scripted(TransportScript(status_code=429, body=b"", headers={"retry-after": "30"}))
    with pytest.raises(RateLimited) as raised:
        await adapter.search(SearchRequest(query="capybara"))

    assert raised.value.provider_instance == "searx.example.org"
    assert raised.value.egress == transport.egress_name
    assert raised.value.occurred_at is not None
    assert raised.value.occurred_at.tzinfo is not None
    record = raised.value.to_record()
    assert record.egress == transport.egress_name
    assert record.occurred_at == raised.value.occurred_at


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, AuthFailed),
        (407, AuthFailed),
        (402, QuotaExhausted),
        (408, TimedOut),
        (504, TimedOut),
        (500, TransportFailed),
        (400, TransportFailed),
    ],
)
async def test_statuses_map_onto_distinguishable_classes(status: int, expected: type[Exception]) -> None:
    """SR-J1: the correct response differs per class, so they stay distinct."""
    adapter, _ = _scripted(TransportScript(status_code=status, body=b""))
    with pytest.raises(expected):
        await adapter.search(SearchRequest(query="capybara"))


async def test_a_body_that_is_not_json_is_a_malformed_response() -> None:
    """An HTML error page is the shape a misconfigured instance answers with."""
    adapter, _ = _scripted(TransportScript(body=b"<html>403 Forbidden</html>"))
    with pytest.raises(MalformedResponse, match="not JSON"):
        await adapter.search(SearchRequest(query="capybara"))


async def test_json_without_a_results_list_is_a_malformed_response() -> None:
    """Well-formed JSON in the wrong shape is still the wrong shape."""
    adapter, _ = _scripted(TransportScript(body=MALFORMED_BODY))
    with pytest.raises(MalformedResponse, match="'results' list"):
        await adapter.search(SearchRequest(query="capybara"))


async def test_a_result_without_a_url_fails_loudly_rather_than_vanishing() -> None:
    """Skipping it would report a narrower set as complete."""
    adapter, _ = _scripted(TransportScript(body=body(({**WEB_RESULT, "url": ""},))))
    with pytest.raises(MalformedResponse, match="no url"):
        await adapter.search(SearchRequest(query="capybara"))


async def test_a_transport_that_speaks_the_taxonomy_keeps_its_own_record() -> None:
    """The transport knows attempts and bytes; the adapter adds the instance."""
    refused = LocalCapExceeded(
        "response is 99999999 bytes, past this transport's cap",
        spend=Spend(wall_clock_seconds=0.4, calls=0, bytes_transferred=99999999),
        scope="response-bytes",
    )
    adapter, _ = _scripted(TransportScript(failure=refused))
    with pytest.raises(LocalCapExceeded) as raised:
        await adapter.search(SearchRequest(query="capybara"))

    assert raised.value.scope == "response-bytes"
    assert raised.value.spend.bytes_transferred == 99999999
    assert raised.value.provider_instance == "searx.example.org"


async def test_a_timeout_from_the_transport_stays_a_timeout() -> None:
    """A retryable failure must not be flattened into an unretryable one."""
    adapter, _ = _scripted(TransportScript(failure=TimeoutError("read timed out")))
    with pytest.raises(TimedOut):
        await adapter.search(SearchRequest(query="capybara"))


# --- criteria -------------------------------------------------------------


async def test_language_is_pushed_down() -> None:
    """SearXNG expresses it, so it goes on the wire."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=(Criterion.language("pt-BR"),)))

    params = transport.calls[-1]["params"]
    assert isinstance(params, dict)
    assert params["language"] == "pt-BR"
    assert result.dispositions[0].disposition == "pushdown"


async def test_a_mappable_carrier_becomes_a_category() -> None:
    """Carrier scoping is a criterion, not a second tool (D17)."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=(Criterion.carrier("image"),)))

    params = transport.calls[-1]["params"]
    assert isinstance(params, dict)
    assert params["categories"] == "images"
    assert result.dispositions[0].disposition == "pushdown"


async def test_an_unmappable_carrier_says_so_and_names_what_works() -> None:
    """SR-B3: unsatisfiable is reported with the reason, never dropped."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=(Criterion.carrier("hologram"),)))

    params = transport.calls[-1]["params"]
    assert isinstance(params, dict)
    assert "categories" not in params
    assert result.dispositions[0].disposition == "unsatisfied"
    assert "image" in (result.dispositions[0].detail or "")


async def test_an_absolute_time_range_is_named_unsatisfiable_not_approximated() -> None:
    """Widening it to a relative window would answer a different question."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.time_range(start=datetime(2026, 1, 1, tzinfo=UTC)),))
    )

    params = transport.calls[-1]["params"]
    assert isinstance(params, dict)
    assert "time_range" not in params
    assert result.dispositions[0].disposition == "unsatisfied"
    assert SEARXNG_PARAM_TIME_RANGE in (result.dispositions[0].detail or "")


async def test_the_relative_window_rides_a_namespaced_criterion() -> None:
    """Provider-specific vocabulary is namespaced, never a new plain key."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.namespaced("searxng", "time-range", "week"),))
    )

    params = transport.calls[-1]["params"]
    assert isinstance(params, dict)
    assert params["time_range"] == "week"
    assert result.dispositions[0].disposition == "pushdown"


async def test_an_invalid_relative_window_is_refused_with_the_accepted_set() -> None:
    """A typo gets the vocabulary back, not a 400 from the provider."""
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.namespaced("searxng", "time-range", "fortnight"),))
    )

    assert result.dispositions[0].disposition == "unsatisfied"
    assert "day" in (result.dispositions[0].detail or "")


@pytest.mark.parametrize(
    ("criterion", "parameter", "expected"),
    [
        (Criterion.namespaced("searxng", "engines", ["duckduckgo", "brave"]), "engines", "duckduckgo,brave"),
        (Criterion.namespaced("searxng", "categories", "news"), "categories", "news"),
        (Criterion.namespaced("searxng", "safesearch", 2), "safesearch", "2"),
        (Criterion.namespaced("searxng", "page", 3), "pageno", "3"),
    ],
)
async def test_every_declared_namespaced_parameter_reaches_the_wire(
    criterion: Criterion, parameter: str, expected: str
) -> None:
    """A declaration that cannot be exercised is a declaration that drifts."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=(criterion,)))

    params = transport.calls[-1]["params"]
    assert isinstance(params, dict)
    assert params[parameter] == expected
    assert result.dispositions[0].disposition == "pushdown"


async def test_a_foreign_namespace_is_ignored_but_reported() -> None:
    """Another provider's parameter is unknown here, and says so."""
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.namespaced("tavily", "search-depth", "advanced"),))
    )

    assert result.dispositions[0].disposition == "ignored-unknown"


async def test_domain_scoping_is_applied_locally_because_searxng_cannot() -> None:
    """Declaring 'local' is a promise, and this is where it is kept."""
    adapter, transport = _scripted(TransportScript(body=TWO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.domains_include(["example.org"]),))
    )

    params = transport.calls[-1]["params"]
    assert isinstance(params, dict)
    assert "domains" not in str(params)
    assert [candidate.identity for candidate in result.candidates] == [WEB_RESULT["url"]]
    assert result.dispositions[0].disposition == "local"


async def test_domain_exclusion_drops_only_the_named_domain() -> None:
    """The other half of local domain scoping."""
    adapter, _ = _scripted(TransportScript(body=TWO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.domains_exclude(["images.example.net"]),))
    )

    assert [candidate.identity for candidate in result.candidates] == [WEB_RESULT["url"]]


async def test_domain_scoping_matches_subdomains() -> None:
    """A domain criterion means the domain and what sits under it."""
    adapter, _ = _scripted(TransportScript(body=TWO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.domains_include(["example.net"]),))
    )

    assert [candidate.identity for candidate in result.candidates] == [IMAGE_RESULT["url"]]


async def test_the_result_cap_is_applied_after_filtering() -> None:
    """A cap of one with a filter yields one in-scope result, not one of any."""
    adapter, _ = _scripted(TransportScript(body=TWO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(
            query="capybara",
            criteria=(Criterion.domains_include(["images.example.net"]), Criterion.max_results(1)),
        )
    )

    assert [candidate.identity for candidate in result.candidates] == [IMAGE_RESULT["url"]]


async def test_min_resolution_and_rights_are_named_unsatisfiable() -> None:
    """SearXNG has neither filter, and pretending otherwise would be a lie."""
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(
            query="capybara",
            criteria=(Criterion.min_resolution(width=800, height=600), Criterion.rights_class("public-domain")),
        )
    )

    assert {disposition.disposition for disposition in result.dispositions} == {"unsatisfied"}
    assert all(disposition.detail for disposition in result.dispositions)


async def test_every_criterion_gets_exactly_one_answer() -> None:
    """SR-B2, over all four dispositions at once."""
    criteria = (
        Criterion.language("en"),
        Criterion.max_results(5),
        Criterion.rights_class("public-domain"),
        Criterion.namespaced("tavily", "topic", "news"),
    )
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=criteria))

    answers = {disposition.criterion_key: disposition.disposition for disposition in result.dispositions}
    assert len(result.dispositions) == len(criteria)
    assert answers == {
        "language": "pushdown",
        "max-results": "local",
        "rights-class": "unsatisfied",
        "tavily:topic": "ignored-unknown",
    }


# --- one parameter, one owner ---------------------------------------------


def _categories_pair(explicit: JsonValue = "news") -> tuple[Criterion, Criterion]:
    """The two criteria that both reach for SearXNG's 'categories' slot.

    :param explicit: value for the namespaced criterion
    :ptype explicit: JsonValue
    :return: the carrier criterion and the namespaced one, in that order
    :rtype: tuple[Criterion, Criterion]
    """
    return Criterion.carrier("image"), Criterion.namespaced("searxng", "categories", explicit)


@pytest.mark.parametrize("reversed_order", [False, True])
async def test_only_the_criterion_that_wins_the_categories_slot_reports_pushdown(reversed_order: bool) -> None:
    """Two criteria, one wire parameter: the loser is named, not overwritten.

    Both wrote params['categories'] and both reported pushdown, so whichever
    came last silently won while the caller was told both scopings applied.
    """
    carrier, explicit = _categories_pair()
    criteria = (explicit, carrier) if reversed_order else (carrier, explicit)
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=criteria))

    params = transport.calls[-1]["params"]
    assert isinstance(params, dict)
    assert params["categories"] == "news", "the explicit SearXNG vocabulary takes the slot, whatever the order"
    answers = {disposition.criterion_key: disposition for disposition in result.dispositions}
    assert answers[SEARXNG_PARAM_CATEGORIES].disposition == "pushdown"
    assert answers[CRITERION_CARRIER].disposition == "unsatisfied"


@pytest.mark.parametrize("reversed_order", [False, True])
async def test_the_losing_carrier_names_the_clash_and_why_the_two_are_not_merged(reversed_order: bool) -> None:
    """SR-B3: a suppressed criterion teaches, and the teaching is order-free."""
    carrier, explicit = _categories_pair()
    criteria = (explicit, carrier) if reversed_order else (carrier, explicit)
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=criteria))

    detail = next(d.detail or "" for d in result.dispositions if d.criterion_key == CRITERION_CARRIER)
    assert SEARXNG_PARAM_CATEGORIES in detail, "the winner is named"
    assert "union" in detail, "and why merging would widen rather than intersect the two scopings"
    assert "images" in detail, "with the category value that would express both intents in one"


async def test_a_carrier_alone_still_takes_the_categories_slot() -> None:
    """Precedence only bites when something else is actually asking for it."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=(Criterion.carrier("video"),)))

    params = transport.calls[-1]["params"]
    assert isinstance(params, dict)
    assert params["categories"] == "videos"
    assert result.dispositions[0].disposition == "pushdown"


async def test_an_unusable_categories_criterion_suppresses_nothing() -> None:
    """Gating on validity, not presence: one typo must not lose both scopings."""
    carrier, explicit = _categories_pair(explicit=42)
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=(explicit, carrier)))

    params = transport.calls[-1]["params"]
    assert isinstance(params, dict)
    assert params["categories"] == "images", "the carrier claims the slot the refused criterion never took"
    answers = {disposition.criterion_key: disposition.disposition for disposition in result.dispositions}
    assert answers == {SEARXNG_PARAM_CATEGORIES: "unsatisfied", CRITERION_CARRIER: "pushdown"}


# --- a recognised key with a value the adapter cannot use ------------------


@pytest.mark.parametrize("value", [-1, 0, True, "10", 2.5, None])
async def test_a_malformed_result_cap_is_refused_rather_than_mis_sliced(value: JsonValue) -> None:
    """The slice reads -1 as 'drop the last' and 0 as 'return nothing'.

    Both came back as a clean success under a disposition claiming the cap
    had been honoured -- 0 in particular reported an empty result set as one
    the caller had asked for.
    """
    adapter, _ = _scripted(TransportScript(body=TWO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion(key=CRITERION_MAX_RESULTS, value=value),))
    )

    assert len(result.candidates) == 2, "an uncapped search returns what the page returned"
    assert result.dispositions[0].disposition == "unsatisfied"
    assert "positive integer" in (result.dispositions[0].detail or "")
    assert type(value).__name__ in (result.dispositions[0].detail or ""), "the detail names what arrived"


async def test_a_well_formed_result_cap_is_still_applied_locally() -> None:
    """The guard refuses bad values without refusing the criterion itself."""
    adapter, _ = _scripted(TransportScript(body=TWO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=(Criterion.max_results(1),)))

    assert len(result.candidates) == 1
    assert result.dispositions[0].disposition == "local"


@pytest.mark.parametrize(
    ("criterion", "parameter"),
    [
        (Criterion(key=CRITERION_LANGUAGE, value=["en", "fr"]), "language"),
        (Criterion(key=CRITERION_LANGUAGE, value=""), "language"),
        (Criterion(key=CRITERION_CARRIER, value={"kind": "image"}), "categories"),
        (Criterion.namespaced("searxng", "safesearch", "high"), "safesearch"),
        (Criterion.namespaced("searxng", "safesearch", 7), "safesearch"),
        (Criterion.namespaced("searxng", "safesearch", True), "safesearch"),
        (Criterion.namespaced("searxng", "page", 0), "pageno"),
        (Criterion.namespaced("searxng", "page", {"n": 2}), "pageno"),
        (Criterion.namespaced("searxng", "page", "2"), "pageno"),
        (Criterion.namespaced("searxng", "time-range", ["week"]), "time_range"),
        (Criterion.namespaced("searxng", "categories", 42), "categories"),
        (Criterion.namespaced("searxng", "engines", []), "engines"),
        (Criterion(key=CRITERION_DOMAINS_INCLUDE, value=["example.org", 7]), "domains"),
    ],
)
async def test_a_recognised_key_with_a_bad_value_is_unsatisfied_and_never_reaches_the_wire(
    criterion: Criterion, parameter: str
) -> None:
    """One disease, one answer shape: unsatisfied, with what was expected."""
    adapter, transport = _scripted(TransportScript(body=TWO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=(criterion,)))

    params = transport.calls[-1]["params"]
    assert isinstance(params, dict)
    assert parameter not in params, "a value the adapter refused must not have gone out anyway"
    answer = result.dispositions[0]
    assert answer.disposition == "unsatisfied", f"{criterion.key} was answered {answer.disposition}"
    assert answer.detail and "takes" in answer.detail, "the detail says what the criterion takes"
    assert repr(criterion.value) in answer.detail, "and what arrived instead"


async def test_a_domain_list_carrying_a_non_domain_is_refused_whole() -> None:
    """Scoping by three of the four domains a caller named is the quiet lie."""
    adapter, _ = _scripted(TransportScript(body=TWO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion(key=CRITERION_DOMAINS_INCLUDE, value=["example.org", 7]),))
    )

    assert len(result.candidates) == 2, "no partial filter was applied"
    assert result.dispositions[0].disposition == "unsatisfied"


@pytest.mark.parametrize("key", sorted(WELL_KNOWN_CRITERIA | set(SEARXNG_CAPABILITIES.namespaced_parameters)))
async def test_no_key_this_adapter_declares_can_ever_be_answered_ignored_unknown(key: str) -> None:
    """Recognition is decided by key, before any value is looked at.

    The bug this pins: a well-known key with a wrong-typed value fell out of
    the elif chain into the unknown branch and was told "'language' is not a
    criterion this adapter recognises" -- contradicting the adapter's own
    capability declaration in the same breath.
    """
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion(key=key, value={"nonsense": [1, 2]}),))
    )

    assert len(result.dispositions) == 1
    assert result.dispositions[0].disposition != "ignored-unknown"
    assert result.dispositions[0].detail, "and it says what went wrong (SR-B3)"


async def test_a_key_this_adapter_really_does_not_know_is_still_ignored_unknown() -> None:
    """'ignored-unknown' keeps its meaning: a key from someone else's vocabulary."""
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.namespaced("acme", "vibe", ["loud"]),))
    )

    assert result.dispositions[0].disposition == "ignored-unknown"


async def test_a_valid_safesearch_level_still_reaches_the_wire_as_an_integer() -> None:
    """The vocabulary check refuses the wrong values, not the right ones."""
    for level in SEARXNG_CAPABILITIES.safesearch_levels or ():
        adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
        result = await adapter.search(
            SearchRequest(query="capybara", criteria=(Criterion.namespaced("searxng", "safesearch", level),))
        )
        params = transport.calls[-1]["params"]
        assert isinstance(params, dict)
        assert params["safesearch"] == str(level)
        assert result.dispositions[0].disposition == "pushdown"


async def test_a_refused_namespaced_value_leaves_the_deployment_default_alone() -> None:
    """A local refusal must not clear configuration it was never asked about."""
    transport = ScriptedTransport((TransportScript(body=ZERO_RESULTS_BODY),))
    adapter = _adapter(transport, default_safesearch=1, default_engines=("duckduckgo",))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.namespaced("searxng", "safesearch", "high"),))
    )

    params = transport.calls[-1]["params"]
    assert isinstance(params, dict)
    assert params["safesearch"] == "1", "the instance's configured level stayed in force, as the detail says"
    assert params["engines"] == "duckduckgo"
    assert result.dispositions[0].disposition == "unsatisfied"
    assert "configured level stayed in force" in (result.dispositions[0].detail or "")


async def test_every_declared_namespaced_parameter_has_a_mapping() -> None:
    """A declaration with no branch would answer nothing, or the wrong thing."""
    assert set(SEARXNG_CAPABILITIES.namespaced_parameters) == {
        SEARXNG_PARAM_CATEGORIES,
        SEARXNG_PARAM_ENGINES,
        SEARXNG_PARAM_PAGE,
        SEARXNG_PARAM_SAFESEARCH,
        SEARXNG_PARAM_TIME_RANGE,
    }


# --- notices --------------------------------------------------------------


async def test_unresponsive_engines_become_a_notice() -> None:
    """A narrower answer that reads as complete is the defect P8 prevents."""
    adapter, _ = _scripted(TransportScript(body=body(unresponsive=(["wikidata", "timeout"], ["startpage", "CAPTCHA"]))))
    result = await adapter.search(SearchRequest(query="capybara"))

    assert len(result.notices) == 1
    assert "startpage" in result.notices[0]
    assert "wikidata" in result.notices[0]


async def test_a_healthy_fan_in_reports_no_notices() -> None:
    """Nothing wrong means nothing said."""
    adapter, _ = _scripted(TransportScript(body=TWO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara"))

    assert result.notices == ()


async def test_zero_results_is_a_success_with_spend() -> None:
    """SR-J2: an empty set is a value, and it still cost something."""
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY, elapsed_seconds=0.05))
    result = await adapter.search(SearchRequest(query="nothing at all"))

    assert result.candidates == ()
    assert result.spend.calls == 1
    assert result.spend.wall_clock_seconds == 0.05
