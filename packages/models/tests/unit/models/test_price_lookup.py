"""Unit tests for the pluggable price lookup (sources + matching + conversion, no network).

A captured-shape catalogue fixture drives the OpenRouter matcher; a trimmed copy of the real
Voyage pricing page (``_voyage_pricing_fixture.html``) drives the scraper. Source-registry routing
and ``register_price_source`` insertion are covered without any network. An opt-in live test
(network, gated by ``THREETEARS_LIVE_PRICE_LOOKUP=1``) proves the real OpenRouter endpoint still has
the shape we map.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import threetears.models.price_lookup as price_lookup_module
from threetears.models import (
    OPENROUTER_MODELS_URL,
    VOYAGE_PRICING_URL,
    OpenRouterPriceSource,
    PriceLookup,
    VoyagePriceSource,
    fetch_provider_models,
    list_provider_models,
    lookup_price,
    match_price,
    register_price_source,
)
from threetears.models.price_lookup import (
    _SourceRegistry,
    parse_voyage_pricing,
)

_VOYAGE_FIXTURE = (Path(__file__).parent / "_voyage_pricing_fixture.html").read_text(encoding="utf-8")

# A trimmed catalogue with OpenRouter's real shape (ids, per-token pricing strings, context).
_CATALOGUE: list[dict[str, Any]] = [
    {
        "id": "anthropic/claude-3.5-sonnet",
        "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        "context_length": 200000,
    },
    {
        "id": "anthropic/claude-sonnet-4.5",
        "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        "context_length": 1000000,
    },
    {"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025", "completion": "0.00001"}, "context_length": 128000},
    {
        "id": "openai/gpt-4o-mini",
        "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
        "context_length": 128000,
    },
    {"id": "meta-llama/llama-3-70b", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 8192},
]


def test_openrouter_slug_matches_exactly() -> None:
    # an openrouter-provider model: the api_name IS the slug
    result = match_price(_CATALOGUE, provider="openrouter", api_name="anthropic/claude-3.5-sonnet")
    assert result.matched_id == "anthropic/claude-3.5-sonnet"
    assert result.cost_per_1m_input == Decimal("3.000000")
    assert result.cost_per_1m_output == Decimal("15.000000")
    assert result.context_window_tokens == 200000


def test_openai_direct_matches_via_provider_prefix() -> None:
    # api_name 'gpt-4o' → 'openai/gpt-4o'
    result = match_price(_CATALOGUE, provider="openai", api_name="gpt-4o")
    assert result.matched_id == "openai/gpt-4o"
    assert result.cost_per_1m_input == Decimal("2.5")


def test_anthropic_dated_id_matches_via_normalisation() -> None:
    # 'claude-3-5-sonnet-20241022' (anthropic api id) ↔ 'anthropic/claude-3.5-sonnet' (slug)
    result = match_price(_CATALOGUE, provider="anthropic", api_name="claude-3-5-sonnet-20241022")
    assert result.matched_id == "anthropic/claude-3.5-sonnet"
    assert result.cost_per_1m_output == Decimal("15.000000")


def test_no_near_match_returns_empty_rather_than_a_wrong_price() -> None:
    # 'claude-sonnet-4-20250514' normalises to 'claudesonnet4'; the catalogue only has '4.5'
    # ('claudesonnet45') — conservative matcher returns nothing rather than the 4.5 price.
    result = match_price(_CATALOGUE, provider="anthropic", api_name="claude-sonnet-4-20250514")
    assert result.matched_id is None
    assert result.cost_per_1m_input is None
    assert result.cost_per_1m_output is None
    assert result.context_window_tokens is None


def test_cross_provider_id_is_not_matched() -> None:
    # I2: 'gpt-4o' under provider 'anthropic' must NOT match 'openai/gpt-4o' (a near, WRONG price).
    result = match_price(_CATALOGUE, provider="anthropic", api_name="gpt-4o")
    assert result.matched_id is None
    # but the SAME id under its real provider matches
    assert match_price(_CATALOGUE, provider="openai", api_name="gpt-4o").matched_id == "openai/gpt-4o"


def test_zero_price_becomes_none() -> None:
    result = match_price(_CATALOGUE, provider="openrouter", api_name="meta-llama/llama-3-70b")
    assert result.matched_id == "meta-llama/llama-3-70b"
    assert result.cost_per_1m_input is None  # a "0" price is treated as unpriced
    assert result.context_window_tokens == 8192


def test_malformed_price_string_becomes_none_not_a_crash() -> None:
    # a non-numeric price must not raise (the InvalidOperation/ValueError guard) — it's unpriced.
    catalogue = [
        {"id": "openrouter/broken", "pricing": {"prompt": "not-a-number", "completion": "??"}},
    ]
    result = match_price(catalogue, provider="openrouter", api_name="openrouter/broken")
    assert result.matched_id == "openrouter/broken"
    assert result.cost_per_1m_input is None
    assert result.cost_per_1m_output is None


def test_unknown_model_returns_empty() -> None:
    result = match_price(_CATALOGUE, provider="openai", api_name="gpt-9-ultra")
    assert result == type(result)(
        provider="openai",
        api_name="gpt-9-ultra",
        matched_id=None,
        cost_per_1m_input=None,
        cost_per_1m_output=None,
        context_window_tokens=None,
    )


def test_discovery_openrouter_lists_every_slug_as_api_name() -> None:
    opts = list_provider_models(_CATALOGUE, provider="openrouter")
    names = {o.api_name for o in opts}
    assert "anthropic/claude-3.5-sonnet" in names  # the slug IS the openrouter api_name
    assert "openai/gpt-4o" in names
    assert "meta-llama/llama-3-70b" in names
    # prices come along for the pre-fill (per-1M); a zero price → None
    gpt = next(o for o in opts if o.api_name == "openai/gpt-4o")
    assert gpt.cost_per_1m_input == Decimal("2.5") and gpt.context_window_tokens == 128000


def test_discovery_anthropic_filters_to_provider_and_strips_prefix() -> None:
    opts = list_provider_models(_CATALOGUE, provider="anthropic")
    names = {o.api_name for o in opts}
    # only anthropic entries, with the provider/ prefix stripped to the model segment
    assert names == {"claude-3.5-sonnet", "claude-sonnet-4.5"}
    assert all("/" not in n for n in names)


def test_discovery_sorted_by_api_name() -> None:
    opts = list_provider_models(_CATALOGUE, provider="openrouter")
    assert [o.api_name for o in opts] == sorted(o.api_name for o in opts)


# parity-exempt: minimal httpx.Response stub (raise_for_status + json/text); mirrors no 3tears protocol
class _FakeResponse:
    def __init__(self, payload: dict[str, Any] | None = None, *, text: str = "") -> None:
        self._payload = payload
        self.text = text

    def raise_for_status(self) -> None: ...

    def json(self) -> dict[str, Any]:
        assert self._payload is not None
        return self._payload


# parity-with: threetears.models.price_lookup._HttpClient
class _FakeClient:
    def __init__(self, payload: dict[str, Any] | None = None, *, text: str = "") -> None:
        self._payload = payload
        self._text = text
        self.requested: str | None = None

    async def get(self, url: str) -> _FakeResponse:
        self.requested = url
        return _FakeResponse(self._payload, text=self._text)


async def test_fetch_provider_models_lists_options() -> None:
    client = _FakeClient({"data": _CATALOGUE})
    opts = await fetch_provider_models(client, provider="openai")
    assert client.requested == OPENROUTER_MODELS_URL
    assert {o.api_name for o in opts} == {"gpt-4o", "gpt-4o-mini"}


async def test_lookup_price_fetches_then_matches() -> None:
    client = _FakeClient({"data": _CATALOGUE})
    result = await lookup_price(client, provider="openai", api_name="gpt-4o-mini")
    assert client.requested == OPENROUTER_MODELS_URL
    assert result.matched_id == "openai/gpt-4o-mini"
    assert result.cost_per_1m_input == Decimal("0.15")


async def test_lookup_price_handles_a_shapeless_payload() -> None:
    client = _FakeClient({"unexpected": True})
    result = await lookup_price(client, provider="openai", api_name="gpt-4o")
    assert result.matched_id is None


@pytest.mark.skipif(
    os.environ.get("THREETEARS_LIVE_PRICE_LOOKUP") != "1",
    reason="live network lookup — set THREETEARS_LIVE_PRICE_LOOKUP=1 to run",
)
async def test_live_openrouter_prices_a_known_model() -> None:
    import httpx

    async with httpx.AsyncClient(timeout=20.0) as client:
        result = await lookup_price(client, provider="openai", api_name="gpt-4o")
    assert result.matched_id is not None
    assert result.cost_per_1m_input is not None and result.cost_per_1m_input > 0


# ===========================================================================
# Voyage scraper — parses the trimmed live-page fixture.
# ===========================================================================


def test_voyage_parse_extracts_per_million_prices() -> None:
    prices = parse_voyage_pricing(_VOYAGE_FIXTURE)
    # the per-MILLION column (not per-thousand / per-request), as Decimals
    assert prices["voyage-4"] == Decimal("0.06")
    assert prices["voyage-4-large"] == Decimal("0.12")
    assert prices["voyage-4-lite"] == Decimal("0.02")
    assert prices["voyage-context-3"] == Decimal("0.18")
    # a reranker from the second table
    assert prices["rerank-2.5"] == Decimal("0.05")


def test_voyage_parse_splits_multi_model_cell() -> None:
    # one cell lists voyage-finance-2 / voyage-law-2 / voyage-code-2 sharing one price.
    prices = parse_voyage_pricing(_VOYAGE_FIXTURE)
    assert prices["voyage-finance-2"] == Decimal("0.12")
    assert prices["voyage-law-2"] == Decimal("0.12")
    assert prices["voyage-code-2"] == Decimal("0.12")


def test_voyage_parse_malformed_body_returns_empty() -> None:
    # no "body" hydration field at all → no guessing, empty map.
    assert parse_voyage_pricing("<html><body>no hydration blob here</body></html>") == {}
    # a "body" field whose JSON string doesn't decode → empty (fail-safe).
    assert parse_voyage_pricing('{"body":"\\uXXXX broken escape"}') == {}
    # a well-formed body with NO per-million-tokens table → empty (never a wrong column).
    no_table = '{"body":"<table><thead><tr><th>Model</th><th>Price per pixel</th></tr></thead><tbody><tr><td><code>voyage-x</code></td><td>$9</td></tr></tbody></table>"}'
    assert parse_voyage_pricing(no_table) == {}


async def test_voyage_source_lookup_matches_fixture_model() -> None:
    client = _FakeClient(text=_VOYAGE_FIXTURE)
    result = await VoyagePriceSource().lookup(client, provider="voyage", api_name="voyage-4")
    assert client.requested == VOYAGE_PRICING_URL
    assert result.matched_id == "voyage-4"
    assert result.cost_per_1m_input == Decimal("0.06")
    assert result.cost_per_1m_output is None  # embeddings: input-priced only
    assert result.context_window_tokens is None
    assert result.source == "voyage"
    assert result.source_url == VOYAGE_PRICING_URL


async def test_voyage_source_normalises_dotted_id() -> None:
    # request 'voyage-3-5' should normalise onto the scraped 'rerank-2.5' style dotted id…
    client = _FakeClient(text=_VOYAGE_FIXTURE)
    result = await VoyagePriceSource().lookup(client, provider="voyageai", api_name="rerank-2-5")
    assert result.matched_id == "rerank-2.5"
    assert result.cost_per_1m_input == Decimal("0.05")


async def test_voyage_source_no_match_returns_all_none() -> None:
    client = _FakeClient(text=_VOYAGE_FIXTURE)
    result = await VoyagePriceSource().lookup(client, provider="voyage", api_name="voyage-99-ultra")
    assert result.matched_id is None
    assert result.cost_per_1m_input is None
    assert result.cost_per_1m_output is None
    assert result.source == "voyage"  # provenance still attached for the admin
    assert result.source_url == VOYAGE_PRICING_URL


async def test_voyage_source_malformed_page_fails_safe_not_raises() -> None:
    client = _FakeClient(text="<html>not the pricing page</html>")
    result = await VoyagePriceSource().lookup(client, provider="voyage", api_name="voyage-4")
    assert result.matched_id is None
    assert result.cost_per_1m_input is None


async def test_voyage_source_transport_error_propagates() -> None:
    # a fetch (transport) error must propagate, like the OpenRouter path — only PARSE misses are
    # swallowed into all-None.
    class _BoomClient:
        async def get(self, url: str) -> Any:
            raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError):
        await VoyagePriceSource().lookup(_BoomClient(), provider="voyage", api_name="voyage-4")


# ===========================================================================
# The source registry — routing + register_price_source insertion.
# ===========================================================================


def test_voyage_source_covers_voyage_providers() -> None:
    src = VoyagePriceSource()
    assert src.covers("voyage")
    assert src.covers("voyageai")
    assert not src.covers("anthropic")
    assert not src.covers("openai")


def test_openrouter_source_is_the_catch_all() -> None:
    src = OpenRouterPriceSource()
    assert src.covers("anthropic")
    assert src.covers("openai")
    assert src.covers("voyage")  # covers EVERYTHING — it's the default backstop
    assert src.covers("some-future-provider")


def test_registry_routes_voyage_to_voyage_and_rest_to_openrouter() -> None:
    reg = _SourceRegistry()
    assert isinstance(reg.resolve("voyage"), VoyagePriceSource)
    assert isinstance(reg.resolve("voyageai"), VoyagePriceSource)
    assert isinstance(reg.resolve("anthropic"), OpenRouterPriceSource)
    assert isinstance(reg.resolve("openai"), OpenRouterPriceSource)
    assert isinstance(reg.resolve("anything-else"), OpenRouterPriceSource)


def test_registry_keeps_openrouter_catch_all_last() -> None:
    reg = _SourceRegistry()
    assert isinstance(reg.sources[-1], OpenRouterPriceSource)


def test_register_inserts_before_the_catch_all() -> None:
    reg = _SourceRegistry()

    class _AcmeSource:
        def covers(self, provider: str) -> bool:
            return provider == "acme"

        async def lookup(self, client: Any, *, provider: str, api_name: str) -> PriceLookup:
            return PriceLookup(provider, api_name, None, None, None, None, source="acme")

    acme = _AcmeSource()
    reg.register(acme)
    # inserted BEFORE the catch-all, which remains last…
    assert reg.sources[-1].__class__ is OpenRouterPriceSource
    assert acme in reg.sources[:-1]
    # …and it now wins for its provider.
    assert reg.resolve("acme") is acme


async def test_module_register_price_source_routes_through_lookup_price() -> None:
    # register a stub source on the MODULE-LEVEL registry, then prove lookup_price dispatches to it.
    sentinel = PriceLookup("acme", "widget-1", "acme/widget-1", Decimal("1.5"), None, None, source="acme")

    class _AcmeSource:
        def covers(self, provider: str) -> bool:
            return provider == "acme"

        async def lookup(self, client: Any, *, provider: str, api_name: str) -> PriceLookup:
            return sentinel

    original = list(price_lookup_module._REGISTRY.sources)  # noqa: SLF001 -- save/restore module state for test isolation
    try:
        register_price_source(_AcmeSource())
        result = await lookup_price(_FakeClient(text=""), provider="acme", api_name="widget-1")
        assert result is sentinel
        # a non-acme provider still falls through to the OpenRouter catch-all.
        catalogue_client = _FakeClient({"data": _CATALOGUE})
        fallthrough = await lookup_price(catalogue_client, provider="openai", api_name="gpt-4o")
        assert fallthrough.matched_id == "openai/gpt-4o"
    finally:
        price_lookup_module._REGISTRY.sources = original  # noqa: SLF001 -- restore module state


async def test_lookup_price_routes_voyage_to_the_scraper() -> None:
    # end-to-end through the module-level registry: a voyage provider hits the Voyage scraper.
    client = _FakeClient(text=_VOYAGE_FIXTURE)
    result = await lookup_price(client, provider="voyage", api_name="voyage-4-large")
    assert client.requested == VOYAGE_PRICING_URL
    assert result.matched_id == "voyage-4-large"
    assert result.cost_per_1m_input == Decimal("0.12")
    assert result.source == "voyage"
