"""The Tavily adapter: the ported semantics, and the SR-E4 defect's grave.

Everything here runs through the injected transport, which is the point of the
seam: no network, no fixture server, and the wire body is inspectable, so a
disposition claiming ``pushdown`` can be checked against what actually went
out -- and the credits a call bills can be checked against the depth it sent.

The pins that matter most are the ported ones (search-spec.md §3.2: extract,
don't invent): the depth/credit coupling SR-E4 names as a live defect, the
domain scoping, the score coercion that reads a missing score as unknown
rather than zero, and the absolute-dates-beat-``time_range`` precedence
RES-T4M9 established.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from decimal import Decimal

import pytest

import threetears.search.adapters.tavily as tavily_module
from threetears.search.adapters.tavily import (
    TAVILY_400_REMEDIATION,
    TAVILY_API_BASE_URL,
    TAVILY_CAPABILITIES,
    TAVILY_CREDITS_BY_DEPTH,
    TAVILY_MAX_QUERY_CHARACTERS,
    TAVILY_MAX_RESULTS_CEILING,
    TAVILY_PARAM_RAW_CONTENT,
    TAVILY_PARAM_SEARCH_DEPTH,
    TAVILY_PARAM_TIME_RANGE,
    TAVILY_PROVIDER,
    TavilyAdapter,
)
from threetears.search.contracts import (
    CRITERION_DOMAINS_INCLUDE,
    CRITERION_MAX_RESULTS,
    CRITERION_TIME_RANGE,
    FACET_HAS_DOWNLOADABLE_DATA,
    FACET_LOCATOR_KIND,
    FIDELITY_CONTENT,
    FIDELITY_SNIPPET,
    PRICING_PER_WEIGHTED_UNIT,
    PRODUCER_API_PROVIDER,
    SCALE_UNIT_INTERVAL,
    AuthFailed,
    Criterion,
    LocalCapExceeded,
    MalformedResponse,
    QuotaExhausted,
    RateLimited,
    SearchFailure,
    SearchProvider,
    SearchRequest,
    SearchTransport,
    Spend,
    TimedOut,
    TransportFailed,
    get_capabilities,
)
from threetears.search.testing import ScriptedTransport, TransportScript
from _tavily_payloads import (
    CONTENT_RESULT,
    MALFORMED_BODY,
    NEWS_RESULT,
    REQUEST_ID,
    TWO_RESULTS_BODY,
    WEB_RESULT,
    ZERO_RESULTS_BODY,
    body,
)

API_KEY = "tvly-test-key"


def _adapter(transport: SearchTransport, **kwargs: object) -> TavilyAdapter:
    """Build an adapter over ``transport`` with a host-supplied key.

    :param transport: the injected transport
    :ptype transport: SearchTransport
    :param kwargs: constructor overrides
    :ptype kwargs: object
    :return: the adapter under test
    :rtype: TavilyAdapter
    """
    return TavilyAdapter(api_key=API_KEY, transport=transport, **kwargs)  # type: ignore[arg-type]


def _scripted(*steps: TransportScript) -> tuple[TavilyAdapter, ScriptedTransport]:
    """Build an adapter over a scripted transport.

    :param steps: the exchanges the transport answers with
    :ptype steps: TransportScript
    :return: the adapter and its transport
    :rtype: tuple[TavilyAdapter, ScriptedTransport]
    """
    transport = ScriptedTransport(steps)
    return _adapter(transport), transport


def _sent(transport: ScriptedTransport) -> dict[str, object]:
    """The JSON body of the last request the transport was asked to make.

    :param transport: the scripted transport under test
    :ptype transport: ScriptedTransport
    :return: the request body
    :rtype: dict[str, object]
    """
    sent = transport.calls[-1]["json_body"]
    assert isinstance(sent, dict)
    return sent


def _answers(result: object) -> dict[str, str]:
    """Dispositions of a candidate set, keyed by criterion.

    :param result: the candidate set under test
    :ptype result: object
    :return: disposition per criterion key
    :rtype: dict[str, str]
    """
    dispositions = getattr(result, "dispositions", ())
    return {entry.criterion_key: entry.disposition for entry in dispositions}


def _detail(result: object, key: str) -> str:
    """Disposition detail for one criterion key.

    :param result: the candidate set under test
    :ptype result: object
    :param key: the criterion key
    :ptype key: str
    :return: the detail text, empty when there is none
    :rtype: str
    """
    for entry in getattr(result, "dispositions", ()):
        if entry.criterion_key == key:
            return entry.detail or ""
    raise AssertionError(f"no disposition answered for {key!r}")


# --- construction, capabilities, configuration ----------------------------


def test_the_adapter_satisfies_the_provider_seam_by_shape() -> None:
    """P9: the seam is structural -- no base class, no registration."""
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    assert isinstance(adapter, SearchProvider)
    assert adapter.provider == TAVILY_PROVIDER


def test_importing_the_adapter_registers_what_tavily_can_express() -> None:
    """SR-B4: depth, domains, topic and dates, queryable before contact."""
    declared = get_capabilities(TAVILY_PROVIDER)

    assert declared == TAVILY_CAPABILITIES
    assert declared is not None
    assert declared.search_depths == ("basic", "advanced")
    assert declared.topics == ("general", "news")
    assert declared.relative_time_ranges == ("day", "week", "month", "year")
    assert declared.max_results_per_page == TAVILY_MAX_RESULTS_CEILING
    assert declared.pricing_model == PRICING_PER_WEIGHTED_UNIT


def test_the_declaration_answers_the_criteria_tavily_actually_takes() -> None:
    """The declaration is the provider's API, not an aspiration.

    Tavily's domain allow-list and absolute date scoping are exactly the two
    things SearXNG cannot do, and both are load-bearing for a consumer
    choosing between the two.
    """
    assert TAVILY_CAPABILITIES.disposition_for(CRITERION_DOMAINS_INCLUDE) == "pushdown"
    assert TAVILY_CAPABILITIES.disposition_for(CRITERION_TIME_RANGE) == "pushdown"
    assert TAVILY_CAPABILITIES.disposition_for(CRITERION_MAX_RESULTS) == "pushdown"
    assert TAVILY_CAPABILITIES.disposition_for("language") == "unsatisfied"
    assert TAVILY_CAPABILITIES.disposition_for(TAVILY_PARAM_SEARCH_DEPTH) == "pushdown"
    assert TAVILY_CAPABILITIES.disposition_for("searxng:engines") == "ignored-unknown"


def test_a_missing_credential_is_refused_at_construction() -> None:
    """D21/SR-K1: the key comes from the host, and there is no other source."""
    with pytest.raises(ValueError, match="host-supplied api_key"):
        TavilyAdapter(api_key="", transport=ScriptedTransport((TransportScript(),)))


def test_construction_reads_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """SR-K1: an env-provisioned key is not this package's business.

    Two halves, because either alone passes for the wrong reason: the adapter
    sends the key it was handed rather than the one in the environment, and
    the module contains no environment read at all -- which is the half that
    keeps holding when a future edit adds a "convenient" fallback.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "env-key")
    monkeypatch.setenv("TAVILY_BASE_URL", "https://evil.example.net")

    source = inspect.getsource(tavily_module)
    assert "os.environ" not in source
    assert "getenv" not in source
    assert "\nimport os" not in source

    with pytest.raises(TypeError):
        TavilyAdapter(transport=ScriptedTransport((TransportScript(),)))  # type: ignore[call-arg]


async def test_the_host_supplied_key_rides_the_authorization_header() -> None:
    """Credentials come from the host, already resolved (SR-K1)."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    await adapter.search(SearchRequest(query="capybara"))

    headers = transport.calls[-1]["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == f"Bearer {API_KEY}"
    assert headers["Accept"] == "application/json"


def test_a_base_url_of_the_wrong_shape_is_refused() -> None:
    """D21: base URLs come from deployment config, and are validated here."""
    transport = ScriptedTransport((TransportScript(),))
    for bad in ("file:///etc/passwd", "/search", "api.tavily.com", "gopher://api.tavily.com"):
        with pytest.raises(ValueError, match="absolute http"):
            TavilyAdapter(api_key=API_KEY, transport=transport, base_url=bad)


async def test_the_published_endpoint_is_the_default_and_a_gateway_can_replace_it() -> None:
    """A compiled-in product constant an auditor can read -- not an env default."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    await adapter.search(SearchRequest(query="capybara"))
    assert transport.calls[-1]["url"] == f"{TAVILY_API_BASE_URL}/search"
    assert transport.calls[-1]["method"] == "POST"

    fronted = ScriptedTransport((TransportScript(body=ZERO_RESULTS_BODY),))
    await _adapter(fronted, base_url="https://gateway.example.org/tavily/").search(SearchRequest(query="capybara"))
    assert fronted.calls[-1]["url"] == "https://gateway.example.org/tavily/search"


def test_two_keys_are_two_instances() -> None:
    """EVL-TQ7K: separate keys are separate quotas, so they pace separately."""
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    assert adapter.provider_instance == "api.tavily.com"

    named = _adapter(ScriptedTransport((TransportScript(),)), provider_instance="tavily-eval")
    assert named.provider_instance == "tavily-eval"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("default_search_depth", "deep", "default_search_depth"),
        ("default_topic", "gossip", "default_topic"),
        ("default_include_raw_content", "html", "default_include_raw_content"),
        ("default_max_results", 50, "default_max_results"),
        ("usd_per_credit", Decimal("-1"), "usd_per_credit"),
    ],
)
def test_a_misconfigured_default_fails_where_it_is_written(field: str, value: object, message: str) -> None:
    """Not on the first query, and never by silently coercing to a default."""
    with pytest.raises(ValueError, match=message):
        _adapter(ScriptedTransport((TransportScript(),)), **{field: value})


async def test_deployment_defaults_reach_the_wire() -> None:
    """Instance-wide configuration is the host's, and it is what goes out."""
    transport = ScriptedTransport((TransportScript(body=ZERO_RESULTS_BODY),))
    adapter = _adapter(
        transport,
        default_max_results=8,
        default_topic="news",
        default_include_raw_content="text",
        default_include_domains=("example.org",),
    )
    await adapter.search(SearchRequest(query="capybara"))

    sent = _sent(transport)
    assert sent["query"] == "capybara"
    assert sent["max_results"] == 8
    assert sent["topic"] == "news"
    assert sent["include_raw_content"] == "text"
    assert sent["include_domains"] == ["example.org"]


async def test_a_bare_query_asks_for_no_page_text() -> None:
    """Content a consumer will not read is tokens and latency for nothing."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    await adapter.search(SearchRequest(query="capybara"))

    sent = _sent(transport)
    assert "include_raw_content" not in sent
    assert "include_domains" not in sent
    assert sent["search_depth"] == "basic"


# --- the typed shape ------------------------------------------------------


async def test_a_result_becomes_a_fully_typed_candidate() -> None:
    """Everything the provider returned survives in typed form (§3.2)."""
    adapter, _ = _scripted(TransportScript(body=body((WEB_RESULT,))))
    result = await adapter.search(SearchRequest(query="capybara"))

    candidate = result.candidates[0]
    assert candidate.identity == WEB_RESULT["url"]
    assert candidate.title == "Capybara"
    assert candidate.snippet == WEB_RESULT["content"]
    assert [locator.rel for locator in candidate.locators] == ["canonical"]
    assert candidate.facets[FACET_LOCATOR_KIND] == "containing-page"
    assert candidate.facets[FACET_HAS_DOWNLOADABLE_DATA] is False
    assert candidate.provenance.producer == PRODUCER_API_PROVIDER
    assert candidate.provenance.provider_instance == "api.tavily.com"
    assert candidate.provenance.provider_ids["request_id"] == REQUEST_ID
    assert candidate.provenance.provider_ids["search_depth"] == "basic"


async def test_the_depth_a_candidate_was_bought_at_is_provenance() -> None:
    """Two sets bought at two depths are not the same evidence (SR-A3)."""
    transport = ScriptedTransport((TransportScript(body=body((WEB_RESULT,))),))
    adapter = _adapter(transport, default_search_depth="advanced")
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    assert candidate.provenance.provider_ids["search_depth"] == "advanced"


async def test_the_egress_the_request_left_by_is_on_every_candidate() -> None:
    """D20: egress is provenance, and a named value rather than an absence."""
    transport = ScriptedTransport((TransportScript(body=TWO_RESULTS_BODY),), egress_name="warp")
    result = await _adapter(transport).search(SearchRequest(query="capybara"))

    assert {candidate.provenance.egress for candidate in result.candidates} == {"warp"}


async def test_page_text_arrives_as_content_the_search_already_bought() -> None:
    """SR-A2: the Tavily case -- Extract must be able to see it has nothing to do."""
    transport = ScriptedTransport((TransportScript(body=body((CONTENT_RESULT,))),))
    adapter = _adapter(transport, default_include_raw_content="markdown")
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    assert candidate.content is not None
    assert candidate.content.text == CONTENT_RESULT["raw_content"]
    assert candidate.content.origin == "provider-response"
    assert candidate.content.mime_type == "text/markdown"
    assert candidate.content.size_bytes is None, "no fetch happened, so no fetch size is claimed"
    assert candidate.fidelity_achieved == FIDELITY_CONTENT


async def test_a_snippet_only_result_says_content_was_available_and_not_taken() -> None:
    """The two fidelity fields exist so a partial answer cannot read as complete."""
    adapter, _ = _scripted(TransportScript(body=body((WEB_RESULT,))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    assert candidate.content is None
    assert candidate.fidelity_achieved == FIDELITY_SNIPPET
    assert candidate.fidelity_available == FIDELITY_CONTENT


async def test_a_content_fidelity_request_asks_tavily_for_the_page_text() -> None:
    """SR-B6: the consumer states the fidelity; the adapter knows the parameter."""
    adapter, transport = _scripted(TransportScript(body=body((CONTENT_RESULT,))))
    result = await adapter.search(SearchRequest(query="capybara", fidelity=FIDELITY_CONTENT))

    assert _sent(transport)["include_raw_content"] == "text"
    assert result.candidates[0].content is not None


async def test_an_rfc_2822_published_date_is_read_and_kept_raw() -> None:
    """Tavily's news topic reports one shape and its general results another."""
    adapter, _ = _scripted(TransportScript(body=body((NEWS_RESULT,))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    assert candidate.published_at == datetime(2024, 8, 21, 7, 0, tzinfo=UTC)
    assert candidate.provenance.provider_ids["published_date_raw"] == NEWS_RESULT["published_date"]


async def test_a_naive_iso_published_date_is_read_as_utc_and_kept_raw() -> None:
    """The zone assumption is stated and the provider's own string survives."""
    adapter, _ = _scripted(TransportScript(body=body((CONTENT_RESULT,))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    assert candidate.published_at == datetime(2026, 2, 1, tzinfo=UTC)
    assert candidate.provenance.provider_ids["published_date_raw"] == "2026-02-01T00:00:00"


async def test_an_unparseable_published_date_invents_nothing() -> None:
    """A date nobody can read leaves the field unset rather than guessed."""
    adapter, _ = _scripted(TransportScript(body=body(({**WEB_RESULT, "published_date": "last tuesday"},))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    assert candidate.published_at is None
    assert candidate.provenance.provider_ids["published_date_raw"] == "last tuesday"


async def test_relevance_is_named_scaled_and_non_comparable() -> None:
    """D1/SR-A4: a set of named judgments, never one bare 'score' field."""
    adapter, _ = _scripted(TransportScript(body=body((WEB_RESULT,))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    assert len(candidate.scores) == 1
    score = candidate.scores[0]
    assert score.name == "relevance"
    assert score.value == 0.94
    assert score.scale == SCALE_UNIT_INTERVAL
    assert score.comparable is False
    assert score.source == "api.tavily.com"


async def test_a_numeric_string_score_is_coerced() -> None:
    """The ported coercion: Tavily has reported its score as text."""
    adapter, _ = _scripted(TransportScript(body=body((NEWS_RESULT,))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    assert candidate.scores[0].value == 0.5


@pytest.mark.parametrize("reported", [None, "very relevant", True, {"value": 1}])
async def test_an_unusable_score_is_absent_rather_than_zero(reported: object) -> None:
    """Unknown and 'judged irrelevant' are different claims, and only one culls.

    The ported reason: a relevance cull that reads a missing score as zero
    drops the result instead of keeping it, so the contract's spelling for
    unknown -- no score entry at all -- is what has to come out of here.
    """
    adapter, _ = _scripted(TransportScript(body=body(({**WEB_RESULT, "score": reported},))))
    candidate = (await adapter.search(SearchRequest(query="capybara"))).candidates[0]

    assert candidate.scores == ()


# --- spend ----------------------------------------------------------------


async def test_a_basic_search_bills_one_credit() -> None:
    """SR-E1: five dimensions, and the one Tavily actually charges in."""
    adapter, _ = _scripted(TransportScript(body=TWO_RESULTS_BODY, elapsed_seconds=0.25))
    result = await adapter.search(SearchRequest(query="capybara"))

    assert result.spend.provider_units == Decimal("1")
    assert result.spend.calls == 1
    assert result.spend.wall_clock_seconds == 0.25
    assert result.spend.bytes_transferred == len(TWO_RESULTS_BODY)


async def test_an_advanced_search_bills_two_credits() -> None:
    """**SR-E4, the regression test.**

    Discodon counted every search as one unit against a budget whose stated
    purpose was managing shared API credits, while ``advanced`` spends two --
    so an operator who turned depth up under-billed by 2x with nothing to
    notice. Configured here, and again through a criterion below, because the
    defect was that the two numbers could disagree at all.
    """
    transport = ScriptedTransport((TransportScript(body=TWO_RESULTS_BODY),))
    adapter = _adapter(transport, default_search_depth="advanced")
    result = await adapter.search(SearchRequest(query="capybara"))

    assert _sent(transport)["search_depth"] == "advanced"
    assert result.spend.provider_units == Decimal("2")
    assert TAVILY_CREDITS_BY_DEPTH["advanced"] == Decimal("2")


async def test_asking_for_advanced_depth_moves_the_bill_with_it() -> None:
    """SR-E4's other half: the depth on the wire and the billed weight are one act."""
    adapter, transport = _scripted(TransportScript(body=TWO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.namespaced("tavily", "search-depth", "advanced"),))
    )

    assert _sent(transport)["search_depth"] == "advanced"
    assert result.spend.provider_units == Decimal("2")
    assert "2 credit" in _detail(result, TAVILY_PARAM_SEARCH_DEPTH)


async def test_a_configured_rate_prices_the_credits() -> None:
    """Money is Decimal and derived from the rate the host actually pays."""
    transport = ScriptedTransport((TransportScript(body=TWO_RESULTS_BODY),))
    adapter = _adapter(transport, default_search_depth="advanced", usd_per_credit=Decimal("0.008"))
    result = await adapter.search(SearchRequest(query="capybara"))

    assert result.spend.money == Decimal("0.016")
    assert result.spend.currency == "USD"


async def test_without_a_rate_the_credits_still_count() -> None:
    """Unpriced is not free: the rate is plan-dependent, so none is invented (D6)."""
    adapter, _ = _scripted(TransportScript(body=TWO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara"))

    assert result.spend.money == Decimal("0")
    assert result.spend.provider_units == Decimal("1")


async def test_a_failure_that_never_reached_the_provider_counts_no_calls() -> None:
    """D4: budget follows the bill, and nothing billed here."""
    adapter, _ = _scripted(TransportScript(failure=OSError("connection refused")))
    with pytest.raises(TransportFailed) as raised:
        await adapter.search(SearchRequest(query="capybara"))

    assert raised.value.spend.calls == 0
    assert raised.value.spend.provider_units == Decimal("0")


async def test_a_refused_call_is_counted_but_buys_no_credits() -> None:
    """SR-E2/SR-E3: the exchange happened; the search Tavily refused did not."""
    transport = ScriptedTransport((TransportScript(status_code=429, body=b"slow down", elapsed_seconds=0.1),))
    adapter = _adapter(transport, default_search_depth="advanced", usd_per_credit=Decimal("0.008"))
    with pytest.raises(RateLimited) as raised:
        await adapter.search(SearchRequest(query="capybara"))

    assert raised.value.spend.calls == 1
    assert raised.value.spend.bytes_transferred == len(b"slow down")
    assert raised.value.spend.provider_units == Decimal("0")
    assert raised.value.spend.money == Decimal("0")


async def test_zero_results_is_a_success_with_spend() -> None:
    """SR-J2: an empty set is a value, and it still cost a credit."""
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY, elapsed_seconds=0.05))
    result = await adapter.search(SearchRequest(query="nothing at all"))

    assert result.candidates == ()
    assert result.spend.calls == 1
    assert result.spend.provider_units == Decimal("1")


# --- the taxonomy ---------------------------------------------------------


@pytest.mark.parametrize("status", [432, 433, 402])
async def test_quota_exhaustion_is_its_own_class(status: int) -> None:
    """SR-D3: 432 is the plan limit, 433 pay-as-you-go, and neither is pacing."""
    adapter, _ = _scripted(TransportScript(status_code=status, body=b""))
    with pytest.raises(QuotaExhausted) as raised:
        await adapter.search(SearchRequest(query="capybara"))

    assert not isinstance(raised.value, RateLimited)
    assert "top up billing" in (raised.value.remediation or "")


async def test_the_provider_quota_and_a_local_cap_are_different_refusals() -> None:
    """D5: one bounds money, the other bounds a run's shape (SR-D3).

    Merging them would hide which authority said no -- and the two want
    opposite responses: stop searching on this key, versus fix the call.
    """
    adapter, _ = _scripted(TransportScript(status_code=432, body=b""))
    with pytest.raises(QuotaExhausted) as provider_refusal:
        await adapter.search(SearchRequest(query="capybara"))

    capped, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    with pytest.raises(LocalCapExceeded) as local_refusal:
        await capped.search(SearchRequest(query="x" * (TAVILY_MAX_QUERY_CHARACTERS + 1)))

    assert not isinstance(provider_refusal.value, LocalCapExceeded)
    assert not isinstance(local_refusal.value, QuotaExhausted)
    assert isinstance(local_refusal.value, SearchFailure)
    assert local_refusal.value.scope == "query-length"
    assert transport.calls == [], "a query Tavily would refuse must not cost an exchange"
    assert local_refusal.value.spend == Spend()


async def test_an_empty_query_is_refused_before_an_exchange_is_spent() -> None:
    """The other ported local guard; whitespace is not a query."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    with pytest.raises(LocalCapExceeded) as raised:
        await adapter.search(SearchRequest(query="   "))

    assert raised.value.scope == "query-empty"
    assert transport.calls == []


async def test_a_429_carries_the_providers_stated_backoff() -> None:
    """A pacing refusal is only actionable with the interval it named."""
    adapter, _ = _scripted(TransportScript(status_code=429, body=b"", headers={"retry-after": "30"}))
    with pytest.raises(RateLimited) as raised:
        await adapter.search(SearchRequest(query="capybara"))

    assert raised.value.retry_after_seconds == 30.0


async def test_a_400_teaches_the_precedence_defect_it_usually_means() -> None:
    """RES-T4M9 shipped as remediation rather than living in a runbook."""
    adapter, _ = _scripted(TransportScript(status_code=400, body=b""))
    with pytest.raises(TransportFailed) as raised:
        await adapter.search(SearchRequest(query="capybara"))

    assert raised.value.remediation == TAVILY_400_REMEDIATION
    assert "RES-T4M9" in (raised.value.remediation or "")


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, AuthFailed),
        (403, AuthFailed),
        (407, AuthFailed),
        (408, TimedOut),
        (504, TimedOut),
        (500, TransportFailed),
        (422, TransportFailed),
    ],
)
async def test_statuses_map_onto_distinguishable_classes(status: int, expected: type[Exception]) -> None:
    """SR-J1: the correct response differs per class, so they stay distinct."""
    adapter, _ = _scripted(TransportScript(status_code=status, body=b""))
    with pytest.raises(expected):
        await adapter.search(SearchRequest(query="capybara"))


async def test_a_failure_is_fully_attributed_for_pacing_keys() -> None:
    """D8/D20: rate and quota tracking key on (provider instance, egress), and
    pod-resident the failure record on ToolResult.metadata is the only fact a
    consumer-side tracker can rebuild that key from."""
    transport = ScriptedTransport((TransportScript(status_code=433, body=b""),), egress_name="warp")
    adapter = _adapter(transport, provider_instance="tavily-eval")
    with pytest.raises(QuotaExhausted) as raised:
        await adapter.search(SearchRequest(query="capybara"))

    assert raised.value.provider_instance == "tavily-eval"
    assert raised.value.egress == "warp"
    assert raised.value.occurred_at is not None
    assert raised.value.occurred_at.tzinfo is not None
    record = raised.value.to_record()
    assert record.failure_class == "quota-exhausted"
    assert record.spend.calls == 1


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
    assert raised.value.provider_instance == "api.tavily.com"


async def test_a_timeout_from_the_transport_stays_a_timeout() -> None:
    """A retryable failure must not be flattened into an unretryable one."""
    adapter, _ = _scripted(TransportScript(failure=TimeoutError("read timed out")))
    with pytest.raises(TimedOut):
        await adapter.search(SearchRequest(query="capybara"))


async def test_a_body_that_is_not_json_is_a_malformed_response() -> None:
    """An HTML error page is the shape a gateway in front of Tavily answers with."""
    adapter, _ = _scripted(TransportScript(body=b"<html>502 Bad Gateway</html>"))
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


async def test_a_malformed_response_still_carries_what_the_call_cost() -> None:
    """SR-E3: the search was served and billed, whatever the body turned out to be."""
    adapter, _ = _scripted(TransportScript(body=b"not json at all"))
    with pytest.raises(MalformedResponse) as raised:
        await adapter.search(SearchRequest(query="capybara"))

    assert raised.value.spend.calls == 1
    assert raised.value.spend.provider_units == Decimal("1")


# --- criteria -------------------------------------------------------------


async def test_domain_scoping_is_pushed_down_because_tavily_expresses_it() -> None:
    """The ported allow-list, and the thing SearXNG has to do locally."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(
            query="capybara",
            criteria=(
                Criterion.domains_include(["Example.ORG", "docs.example.org"]),
                Criterion.domains_exclude(["spam.example.net"]),
            ),
        )
    )

    sent = _sent(transport)
    assert sent["include_domains"] == ["example.org", "docs.example.org"]
    assert sent["exclude_domains"] == ["spam.example.net"]
    assert set(_answers(result).values()) == {"pushdown"}


async def test_an_empty_domain_list_scopes_nothing_and_says_so() -> None:
    """An unsatisfiable criterion is named, never quietly dropped (SR-B3)."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=(Criterion.domains_include([]),)))

    assert "include_domains" not in _sent(transport)
    assert _answers(result)[CRITERION_DOMAINS_INCLUDE] == "unsatisfied"


async def test_a_result_cap_is_pushed_down_and_clamped_to_tavilys_ceiling() -> None:
    """Twenty results do not exceed a cap of fifty, so the cap is honoured."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=(Criterion.max_results(50),)))

    assert _sent(transport)["max_results"] == TAVILY_MAX_RESULTS_CEILING
    assert _answers(result)[CRITERION_MAX_RESULTS] == "pushdown"
    assert "SR-E5" in _detail(result, CRITERION_MAX_RESULTS)


async def test_a_cap_inside_the_ceiling_goes_down_unchanged() -> None:
    """The ordinary case, so the clamp cannot be the only path that works."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    await adapter.search(SearchRequest(query="capybara", criteria=(Criterion.max_results(3),)))

    assert _sent(transport)["max_results"] == 3


async def test_an_absolute_window_becomes_tavilys_dates() -> None:
    """Tavily takes absolute publication scoping, so it is pushed down."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(
            query="capybara",
            criteria=(
                Criterion.time_range(
                    start=datetime(2026, 1, 1, 12, tzinfo=UTC), end=datetime(2026, 2, 1, 12, tzinfo=UTC)
                ),
            ),
        )
    )

    sent = _sent(transport)
    assert sent["start_date"] == "2026-01-01"
    assert sent["end_date"] == "2026-02-01"
    assert _answers(result)[CRITERION_TIME_RANGE] == "pushdown"


async def test_a_relative_window_rides_a_namespaced_criterion() -> None:
    """Provider-specific vocabulary is namespaced, never a new plain key."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.namespaced("tavily", "time-range", "week"),))
    )

    assert _sent(transport)["time_range"] == "week"
    assert _answers(result)[TAVILY_PARAM_TIME_RANGE] == "pushdown"


async def test_absolute_dates_beat_the_relative_window_and_the_caller_is_told() -> None:
    """**RES-T4M9, the ported precedence.**

    Tavily answers 400 when the two forms arrive together. The fix was stated
    precedence rather than silent suppression of either -- so the absolute
    range goes out, the relative window is reported unsatisfied, and the
    detail says which rule applied and how to get the other behaviour.
    """
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(
            query="capybara",
            criteria=(
                Criterion.namespaced("tavily", "time-range", "week"),
                Criterion.time_range(start=datetime(2026, 1, 1, tzinfo=UTC)),
            ),
        )
    )

    sent = _sent(transport)
    assert sent["start_date"] == "2026-01-01"
    assert "time_range" not in sent, "the combination is exactly what Tavily 400s on"
    answers = _answers(result)
    assert answers[CRITERION_TIME_RANGE] == "pushdown"
    assert answers[TAVILY_PARAM_TIME_RANGE] == "unsatisfied"
    assert "RES-T4M9" in _detail(result, TAVILY_PARAM_TIME_RANGE)


async def test_a_malformed_absolute_date_degrades_to_the_relative_window() -> None:
    """The gate is *validity*, not presence -- the other half of the port.

    Gating on presence would let a date nobody can send silently suppress a
    perfectly good relative window, which is a worse answer than either form
    alone.
    """
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(
            query="capybara",
            criteria=(
                Criterion(key=CRITERION_TIME_RANGE, value={"start": "last tuesday"}),
                Criterion.namespaced("tavily", "time-range", "month"),
            ),
        )
    )

    sent = _sent(transport)
    assert "start_date" not in sent, "a malformed date would earn a 400 and spend an exchange"
    assert sent["time_range"] == "month"
    answers = _answers(result)
    assert answers[CRITERION_TIME_RANGE] == "unsatisfied"
    assert answers[TAVILY_PARAM_TIME_RANGE] == "pushdown"


@pytest.mark.parametrize(
    ("criterion", "parameter", "expected"),
    [
        (Criterion.namespaced("tavily", "topic", "news"), "topic", "news"),
        (Criterion.namespaced("tavily", "search-depth", "advanced"), "search_depth", "advanced"),
        (Criterion.namespaced("tavily", "include-raw-content", "markdown"), "include_raw_content", "markdown"),
        (Criterion.namespaced("tavily", "time-range", "day"), "time_range", "day"),
    ],
)
async def test_every_declared_namespaced_parameter_reaches_the_wire(
    criterion: Criterion, parameter: str, expected: str
) -> None:
    """A declaration that cannot be exercised is a declaration that drifts."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=(criterion,)))

    assert _sent(transport)[parameter] == expected
    assert _answers(result)[criterion.key] == "pushdown"


async def test_asking_for_no_raw_content_takes_the_parameter_back_off() -> None:
    """A request must be able to decline what the deployment turned on."""
    transport = ScriptedTransport((TransportScript(body=ZERO_RESULTS_BODY),))
    adapter = _adapter(transport, default_include_raw_content="text")
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.namespaced("tavily", "include-raw-content", "none"),))
    )

    assert "include_raw_content" not in _sent(transport)
    assert _answers(result)[TAVILY_PARAM_RAW_CONTENT] == "pushdown"


@pytest.mark.parametrize(
    ("criterion", "expected_in_detail"),
    [
        (Criterion.namespaced("tavily", "search-depth", "deep"), "basic, advanced"),
        (Criterion.namespaced("tavily", "topic", "gossip"), "general, news"),
        (Criterion.namespaced("tavily", "time-range", "fortnight"), "day, week, month, year"),
        (Criterion.namespaced("tavily", "include-raw-content", "html"), "text, markdown, none"),
    ],
)
async def test_a_typo_gets_the_vocabulary_back_rather_than_a_400(criterion: Criterion, expected_in_detail: str) -> None:
    """A refusal that costs an exchange to say less is a worse refusal."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=(criterion,)))

    assert _answers(result)[criterion.key] == "unsatisfied"
    assert expected_in_detail in _detail(result, criterion.key)
    assert str(criterion.value) not in str(_sent(transport))


async def test_a_bad_depth_criterion_leaves_the_bill_where_it_was() -> None:
    """The SR-E4 coupling holds through the refusal path too."""
    transport = ScriptedTransport((TransportScript(body=TWO_RESULTS_BODY),))
    adapter = _adapter(transport, default_search_depth="basic")
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.namespaced("tavily", "search-depth", "deepest"),))
    )

    assert _sent(transport)["search_depth"] == "basic"
    assert result.spend.provider_units == Decimal("1")


async def test_what_tavily_cannot_express_is_named_rather_than_dropped() -> None:
    """SR-B3: no language, no carrier scoping, no resolution, no rights."""
    adapter, transport = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(
            query="capybara",
            criteria=(
                Criterion.language("pt-BR"),
                Criterion.carrier("image"),
                Criterion.min_resolution(width=800, height=600),
                Criterion.rights_class("public-domain"),
            ),
        )
    )

    assert set(_answers(result).values()) == {"unsatisfied"}
    assert all(disposition.detail for disposition in result.dispositions)
    sent = _sent(transport)
    assert set(sent) == {"query", "max_results", "topic", "search_depth"}


async def test_a_foreign_namespace_is_ignored_but_reported() -> None:
    """Another provider's parameter is unknown here, and says so."""
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(
        SearchRequest(query="capybara", criteria=(Criterion.namespaced("searxng", "engines", "brave"),))
    )

    assert _answers(result)["searxng:engines"] == "ignored-unknown"


async def test_every_criterion_gets_exactly_one_answer() -> None:
    """SR-B2, over all four dispositions at once."""
    criteria = (
        Criterion.domains_include(["example.org"]),
        Criterion.namespaced("tavily", "topic", "news"),
        Criterion.language("en"),
        Criterion.namespaced("searxng", "safesearch", 2),
    )
    adapter, _ = _scripted(TransportScript(body=ZERO_RESULTS_BODY))
    result = await adapter.search(SearchRequest(query="capybara", criteria=criteria))

    assert len(result.dispositions) == len(criteria)
    assert _answers(result) == {
        "domains-include": "pushdown",
        "tavily:topic": "pushdown",
        "language": "unsatisfied",
        "searxng:safesearch": "ignored-unknown",
    }
