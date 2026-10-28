"""The pricer a model is handed, and the default one: Anthropic's published rates as the package ships them.

The engine cannot know what a host pays, so a :class:`~threetears.evals.providers.Model` is priced by the pricer it is
handed, and by OpenRouter's catalogue for any call the pricer leaves to it. The default pricer,
:class:`~threetears.evals.providers.AnthropicPublishedRates`, is for standalone and development use: a model handed
neither a pricer nor a catalogue gets it. Its rates are pinned here as Anthropic's pricing page states them (read
2026-10-28, https://platform.claude.com/docs/en/about-claude/pricing). Every call goes through
``httpx.MockTransport``, and a catalogue that must not be read fails the test if it is fetched.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Final

import httpx
import pytest

from threetears.evals.providers import (
    AnthropicPublishedRates,
    CallPrice,
    Catalogue,
    Clients,
    HostRates,
    Model,
    ModelRates,
    Provider,
)
from threetears.evals.providers.wire import usage_counts
from threetears.evals.schema.turns import TurnMessage

#: Per model, dollars a million tokens: base input, five-minute cache write, one-hour cache write, cache read, output.
PAGE: Final[dict[str, tuple[str, str, str, str, str]]] = {
    "claude-fable-5-1": ("10", "12.50", "20", "0.25", "50"),
    "claude-mythos-5-1": ("10", "12.50", "20", "0.25", "50"),
    "claude-fable-5": ("10", "12.50", "20", "1", "50"),
    "claude-mythos-5": ("10", "12.50", "20", "1", "50"),
    "claude-opus-5-5": ("4", "5", "8", "0.20", "20"),
    "claude-opus-5": ("5", "6.25", "10", "0.50", "25"),
    "claude-opus-4-8": ("5", "6.25", "10", "0.50", "25"),
    "claude-opus-4-7": ("5", "6.25", "10", "0.50", "25"),
    "claude-opus-4-6": ("5", "6.25", "10", "0.50", "25"),
    "claude-opus-4-5": ("5", "6.25", "10", "0.50", "25"),
    "claude-sonnet-5-5": ("2", "2.50", "4", "0.10", "10"),
    "claude-sonnet-5": ("2", "2.50", "4", "0.20", "10"),
    "claude-sonnet-4-6": ("3", "3.75", "6", "0.30", "15"),
    "claude-sonnet-4-5": ("3", "3.75", "6", "0.30", "15"),
    "claude-haiku-5-5": ("0.10", "0.125", "0.20", "0.01", "0.50"),
    "claude-haiku-4-5": ("1", "1.25", "2", "0.10", "5"),
}

#: Haiku 5.5 for prompts over 100,000 tokens.
HAIKU_LONG: Final = ("0.50", "0.625", "1", "0.05", "2.50")

SOURCE: Final = "anthropic-published:2026-10-28"


def never_read() -> Catalogue:
    """OpenRouter's catalogue, every fetch of which fails the test."""

    def fetched(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"a call its pricer prices read OpenRouter's catalogue at {request.url}")

    return Catalogue(http=httpx.AsyncClient(transport=httpx.MockTransport(fetched)))


def anthropic_reply(model_id: str, usage: dict[str, Any]) -> httpx.AsyncClient:
    reply = {
        "model": f"{model_id}-20261001",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": usage,
    }
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=reply)))


def published(model_id: str, usage: dict[str, Any]) -> tuple[Model, Catalogue]:
    """``model_id`` on Anthropic, priced by the shipped rates, its catalogue one that must not be read."""
    catalogue = never_read()
    model = Model(
        model_id,
        provider="anthropic",
        api_key="test-key",
        pricer=AnthropicPublishedRates(),
        catalogue=catalogue,
        http=anthropic_reply(model_id, usage),
    )
    return model, catalogue


def as_page(rates: ModelRates) -> tuple[str, str, str, str, str]:
    return (
        str(rates.base_input),
        str(rates.cache_write_5m),
        str(rates.cache_write_1h),
        str(rates.cache_read),
        str(rates.output),
    )


def test_the_default_rates_are_the_pricing_page_as_read_on_their_date() -> None:
    default = AnthropicPublishedRates()
    for model_id, page in PAGE.items():
        answered = default.rates_for(model_id, provider="anthropic")
        assert answered is not None, model_id
        assert answered.guaranteed, "Anthropic's published rates are what it bills a direct call"
        found = answered.rates
        assert as_page(found) == page, model_id
        if model_id == "claude-haiku-5-5":
            assert found.long_prompt is not None and as_page(found.long_prompt) == HAIKU_LONG
            assert found.long_prompt_tokens == 100_000
        else:
            assert found.long_prompt is None, model_id


@pytest.mark.parametrize("provider", ["openrouter", "openai"])
def test_the_default_leaves_every_route_but_anthropic_s_to_the_catalogue(provider: Provider) -> None:
    assert AnthropicPublishedRates().rates_for("claude-opus-5-5", provider=provider) is None


def test_a_dated_or_dotted_id_names_its_row_and_an_unlisted_one_none() -> None:
    default = AnthropicPublishedRates()
    dated = default.rates_for("claude-haiku-4-5-20251001", provider="anthropic")
    dotted = default.rates_for("Claude-Opus-4.8", provider="anthropic")
    assert dated is not None and as_page(dated.rates) == PAGE["claude-haiku-4-5"]
    assert dotted is not None and as_page(dotted.rates) == PAGE["claude-opus-4-8"]
    assert default.rates_for("claude-opus-9", provider="anthropic") is None


@pytest.mark.parametrize("model_id", sorted(PAGE))
async def test_every_listed_model_prices_a_call_and_bounds_its_turns_with_no_catalogue_read(model_id: str) -> None:
    model, catalogue = published(model_id, {"input_tokens": 1000, "output_tokens": 100})
    base_input, _, _, _, output = (Decimal(rate) for rate in PAGE[model_id])

    assert await model.turn_pricing_refusal() is None
    assert model.price_ceiling(system="", user="x") is not None
    turn = [TurnMessage("user", "x")]
    assert model.turn_price_ceiling(system="", messages=turn, tools=(), max_tokens=1000) is not None
    completion = await model.generate(system="", user="x")

    assert completion.cost_usd == pytest.approx(float((1000 * base_input + 100 * output) / 1_000_000))
    assert completion.price_source == SOURCE
    assert completion.pricing_failure is None and completion.pricing_stale_since is None
    assert not catalogue.loaded


async def test_a_reply_with_cache_tokens_is_priced_share_by_share_as_by_hand() -> None:
    usage = {
        "input_tokens": 1_200,
        "output_tokens": 300,
        "cache_read_input_tokens": 40_000,
        "cache_creation_input_tokens": 6_000,
        "cache_creation": {"ephemeral_5m_input_tokens": 5_000, "ephemeral_1h_input_tokens": 1_000},
        "output_tokens_details": {"thinking_tokens": 120},
    }
    model, catalogue = published("claude-opus-5-5", usage)

    completion = await model.generate(system="", user="x")

    # Opus 5.5: 1,200 fresh x $4 + 40,000 read x $0.20 + 5,000 five-minute writes x $5 + 1,000 one-hour writes x $8
    # + 300 output (thinking included) x $20, a million tokens: 4,800 + 8,000 + 25,000 + 8,000 + 6,000 = 51,800.
    assert completion.input_tokens == 47_200
    assert completion.cost_usd == pytest.approx(0.0518)
    assert not catalogue.loaded


async def test_cache_writes_not_split_by_lifetime_are_priced_at_the_five_minute_rate() -> None:
    model, _ = published(
        "claude-opus-5-5", {"input_tokens": 100, "output_tokens": 4, "cache_creation_input_tokens": 500}
    )
    completion = await model.generate(system="", user="x")
    assert completion.cost_usd == pytest.approx((100 * 4 + 500 * 5 + 4 * 20) / 1e6), "the only lifetime asked for"
    assert completion.price_source == SOURCE


async def test_haiku_5_5_bills_a_prompt_over_100000_tokens_wholly_at_its_long_prompt_rates() -> None:
    at_bound, _ = published("claude-haiku-5-5", {"input_tokens": 100_000, "output_tokens": 10})
    over, _ = published(
        "claude-haiku-5-5", {"input_tokens": 1, "output_tokens": 10, "cache_read_input_tokens": 100_000}
    )

    assert (await at_bound.generate(system="", user="x")).cost_usd == pytest.approx((100_000 * 0.10 + 10 * 0.50) / 1e6)
    assert (await over.generate(system="", user="x")).cost_usd == pytest.approx(
        (1 * 0.50 + 100_000 * 0.05 + 10 * 2.50) / 1e6
    ), "cache reads count toward the prompt's size"


def test_a_ceiling_bounds_every_prompt_token_at_the_five_minute_write_and_reaches_the_long_tier() -> None:
    default = AnthropicPublishedRates()
    opus = default.rates_for("claude-opus-5-5", provider="anthropic")
    haiku = default.rates_for("claude-haiku-5-5", provider="anthropic")
    assert opus is not None and haiku is not None
    assert opus.rates.ceiling(prompt_tokens=10_000, output_tokens=1_000) == Decimal("0.07")  # 10,000 x $5 + 1,000 x $20
    assert haiku.rates.ceiling(prompt_tokens=200_000, output_tokens=0) == Decimal("0.125")  # the long tier's $0.625 write


def test_counts_that_contradict_each_other_are_unpriced_never_guessed() -> None:
    answered = AnthropicPublishedRates().rates_for("claude-opus-5-5", provider="anthropic")
    assert answered is not None
    opus = answered.rates
    cost, why = opus.cost(input_tokens=10, output_tokens=1, cache_read_tokens=20)
    assert cost is None and why is not None and "20 cached prompt tokens within 10" in why
    cost, why = opus.cost(input_tokens=100, output_tokens=1, cache_write_tokens=5, cache_write_1h_tokens=6)
    assert cost is None and why is not None and "6 one-hour cache writes within 5" in why


def catalogue_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve OpenRouter's catalogue offline, down: every client opened from here on answers 503."""
    real = httpx.AsyncClient

    def opened(**options: Any) -> httpx.AsyncClient:
        options.pop("transport", None)
        return real(transport=httpx.MockTransport(lambda _: httpx.Response(503, text="unavailable")), **options)

    monkeypatch.setattr(httpx, "AsyncClient", opened)


async def test_a_model_the_default_rates_do_not_list_is_refused_before_a_run_starts_saying_what_to_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http = anthropic_reply("claude-opus-9", {"input_tokens": 1, "output_tokens": 1})
    catalogue_down(monkeypatch)
    model = Model("claude-opus-9", provider="anthropic", api_key="test-key", http=http)  # the default pricer

    why = await model.turn_pricing_refusal()

    assert why is not None
    assert why.startswith(
        "no turn on 'claude-opus-9' can be bounded before it is sent: the default pricer answers no rates for it, "
        "and the price catalogue it leaves the call to cannot bound it ("
    )
    assert "could not be read" in why, "the catalogue the default left it to was asked, and could not say"
    assert "read 2026-10-28 from https://platform.claude.com/docs/en/about-claude/pricing" in why
    assert "do not list 'claude-opus-9': use a model they list (claude-fable-5-1, " in why
    assert "hand the model a pricer with the host's own rates" in why


def _down_catalogue() -> Catalogue:
    return Catalogue(
        http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(503, text="unavailable")))
    )


async def test_a_host_s_catalogue_that_cannot_be_read_is_named_and_another_model_is_not_suggested() -> None:
    # The Clients and Discodon shape: a catalogue handed in, no pricer. Opus 5.5 is in the shipped rates, and they are
    # not what prices this model, so they are not offered.
    model = Model(
        "claude-opus-5-5",
        provider="anthropic",
        api_key="test-key",
        catalogue=_down_catalogue(),
        http=anthropic_reply("claude-opus-5-5", {"input_tokens": 1, "output_tokens": 1}),
    )

    why = await model.turn_pricing_refusal()

    assert why is not None
    assert why.startswith(
        "no turn on 'claude-opus-5-5' can be bounded before it is sent: it is priced by the price catalogue it was "
        "handed, which cannot bound it (the price catalogue could not be read (HTTPStatusError"
    )
    assert "the model is not at fault, so naming another will not help: retry once the catalogue reads" in why
    assert "published rates" not in why and "use a model" not in why


async def test_a_host_s_catalogue_with_no_row_for_the_model_says_another_model_could_be_priced() -> None:
    rows = {
        "data": [{"id": "anthropic/claude-haiku-5.5", "pricing": {"prompt": "0.0000001", "completion": "0.0000005"}}]
    }
    catalogue = Catalogue(
        http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=rows)))
    )
    model = Model("claude-opus-5-5", provider="anthropic", api_key="test-key", catalogue=catalogue)

    why = await model.turn_pricing_refusal()

    assert why is not None and "no single priced row of OpenRouter's catalogue matches 'claude-opus-5-5'" in why
    assert "use a model the catalogue prices, or hand the model a pricer" in why
    assert "published rates" not in why


async def test_a_host_s_pricer_that_answers_no_rates_is_named_and_the_shipped_rates_are_not_offered() -> None:
    model = Model(
        "claude-sonnet-5-5",  # in the shipped rates, and not in the host's: the host's pricer is what prices it
        provider="anthropic",
        api_key="test-key",
        pricer=FakeHostPricer(),
        catalogue=_down_catalogue(),
        http=anthropic_reply("claude-sonnet-5-5", {"input_tokens": 1, "output_tokens": 1}),
    )

    why = await model.turn_pricing_refusal()

    assert why is not None
    assert why.startswith(
        "no turn on 'claude-sonnet-5-5' can be bounded before it is sent: the pricer it was handed answers no rates "
        "for it, and the price catalogue it leaves the call to cannot bound it ("
    )
    assert why.endswith("have the host's pricer answer rates for 'claude-sonnet-5-5'")
    assert "published rates" not in why


# --- a host's own prices -------------------------------------------------------------------------------------------


#: A host's deal on Opus 5.5: half the list price.
DEAL: Final = ModelRates(
    base_input=Decimal(2),
    cache_write_5m=Decimal("2.5"),
    cache_write_1h=Decimal(4),
    cache_read=Decimal("0.1"),
    output=Decimal(10),
)


# parity-with: threetears.evals.providers.Pricer
class FakeHostPricer:
    """A host's deal: Opus 5.5 at half the list price, named as the host names it; anything else to the default."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, str]] = []

    def rates_for(self, model: str, *, provider: Provider) -> HostRates | None:
        self.asked.append((model, provider))
        return HostRates(DEAL, guaranteed=True) if model == "claude-opus-5-5" else None

    def price(
        self, model: str, *, provider: Provider, usage: Mapping[str, Any], served_provider: str | None
    ) -> CallPrice:
        counts = usage_counts(provider, usage)
        assert counts.input_tokens is not None and counts.output_tokens is not None
        dollars, _ = DEAL.cost(input_tokens=counts.input_tokens, output_tokens=counts.output_tokens)
        assert dollars is not None
        return CallPrice(usd=float(dollars), source="host-deal:2026")


async def test_a_host_s_pricer_replaces_the_default_and_reads_no_catalogue() -> None:
    host, catalogue = FakeHostPricer(), never_read()
    model = Model(
        "claude-opus-5-5",
        provider="anthropic",
        api_key="test-key",
        pricer=host,
        catalogue=catalogue,
        http=anthropic_reply("claude-opus-5-5", {"input_tokens": 1000, "output_tokens": 100}),
    )

    await model.refresh_prices()
    completion = await model.generate(system="", user="x")

    assert completion.cost_usd == pytest.approx((1000 * 2 + 100 * 10) / 1e6)
    assert completion.price_source == "host-deal:2026"
    assert ("claude-opus-5-5", "anthropic") in host.asked
    assert not catalogue.loaded


async def test_a_derived_model_and_a_factory_s_roles_keep_the_host_s_pricer() -> None:
    template = Model(
        "claude-opus-5-5",
        provider="anthropic",
        api_key="test-key",
        pricer=FakeHostPricer(),
        http=anthropic_reply("claude-opus-5-5", {"input_tokens": 1000, "output_tokens": 100}),
    )
    clients = Clients(default=template, catalogue=never_read())

    assert await clients.load_prices() == {}, "priced by the host's rates, so the catalogue is not asked for it"
    completion = await clients("judge", None).generate(system="", user="x")
    assert completion.price_source == "host-deal:2026"


async def test_a_model_handed_a_catalogue_and_no_pricer_is_priced_from_the_catalogue_as_before() -> None:
    rows = {"data": [{"id": "anthropic/claude-opus-5.5", "pricing": {"prompt": "0.000004", "completion": "0.00002"}}]}
    catalogue = Catalogue(
        http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=rows)))
    )
    model = Model(
        "claude-opus-5-5",
        provider="anthropic",
        api_key="test-key",
        catalogue=catalogue,
        http=anthropic_reply("claude-opus-5-5", {"input_tokens": 1000, "output_tokens": 100}),
    )

    completion = await model.generate(system="", user="x")

    assert completion.price_source == "openrouter-catalogue:anthropic/claude-opus-5.5", "a host's catalogue, unchanged"
    assert catalogue.loaded
