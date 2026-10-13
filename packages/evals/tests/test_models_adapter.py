"""The models adapter (``models_client``) against mocked provider transports: no provider is called.

What each test holds, by the port's terms: a completion meets :class:`~threetears.evals.schema.CompletionResult`
on every provider; a schema's reply is the raw JSON text the model wrote; reasoning settings reach the wire as
each provider takes them, and every one a provider cannot take is refused by name before anything is sent;
retries are off (on OpenRouter too, whose SDK has a default backoff of its own) and the client's breaker is its
own; a call is priced from the catalogue or OpenRouter's own figure, and an unpriced one says why; the served
model is the reply's, never a requested alias echoed back; the stop reason is the engine's word, with a refusal as
the filter and an unnamed stop cut short; and ``aclose`` releases what the client owns and nothing shared. The
factory's tests are ``test_models_clients.py``, the describer's ``test_models_failure_describer.py``.

Every request goes to an ``httpx.MockTransport`` (``models_adapter_support``); every catalogue read to a fake.
Recorded replies replace the canned ones in a later release; the assertions stay.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from typing import Any

import anthropic
import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableBinding

from threetears.evals.kernel import OutOfRunBudget, PlannedCall, ValidationFailedError
from threetears.evals.providers import ModelsClientRefused, ModelsCompletion, ModelsCompletionClient, models_client
from threetears.evals.schema import OutOfRunSpend
from threetears.evals.testing import check_completion_conformance
from threetears.models import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL, CircuitOpenError, create_chat_model

from packages.evals.tests.models_adapter_support import (
    ANTHROPIC_HOST,
    HOSTS,
    MODELS,
    OPENAI_HOST,
    OPENROUTER_HOST,
    FakeCatalogue,
    ProviderTransport,
    anthropic_reply,
    build,
    mocked_providers,
    ok,
    openai_reply,
    openai_responses_reply,
    plan_reply,
    status,
)

#: The logger the adapter writes to.
ADAPTER_LOGGER = "threetears.evals.providers.adapter"

PROVIDERS = ("anthropic", "openai", "openrouter")

#: The strict directive every engine call that reads a typed reply sends.
SCHEMA_DIRECTIVE: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_reply",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"score": {"type": "integer"}},
            "required": ["score"],
            "additionalProperties": False,
        },
    },
}


@pytest.fixture
def net(monkeypatch: pytest.MonkeyPatch) -> Iterator[ProviderTransport]:
    """The mocked network for one test."""
    yield from mocked_providers(monkeypatch)


async def _generate(client: Any, **kwargs: Any) -> ModelsCompletion:
    result = await client.generate(system="You grade.", user="Grade this.", **kwargs)
    assert isinstance(result, ModelsCompletion)
    return result


# --- the port ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_every_provider_s_completion_meets_the_port(net: ProviderTransport, provider: str) -> None:
    plan_reply(net, provider)
    result = await _generate(build(net, provider, temperature=0.2))
    check_completion_conformance(result)


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_concurrent_calls_on_one_client_each_send_their_own_prompt(net: ProviderTransport, provider: str) -> None:
    plan_reply(net, provider)
    client = build(net, provider)
    await asyncio.gather(*(client.generate(system="s", user=f"prompt {n}") for n in range(4)))
    users = sorted(
        next(m["content"] for m in body["messages"] if m["role"] == "user") for body in net.sent(HOSTS[provider])
    )
    flattened = [u if isinstance(u, str) else u[0]["text"] for u in users]
    assert sorted(flattened) == [f"prompt {n}" for n in range(4)]


# --- response_format ----------------------------------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_schema_s_reply_is_the_raw_json_text_the_model_wrote(net: ProviderTransport, provider: str) -> None:
    """Not a parsed object and not re-serialised: the spacing the model wrote survives."""
    written = '{ "score" :4 }'
    if provider == "openrouter":
        plan_reply(net, provider, text=('{ "score" ', ":4 }"))
    else:
        plan_reply(net, provider, text=written)
    result = await _generate(build(net, provider), response_format=SCHEMA_DIRECTIVE)
    assert result.content == written


async def test_a_schema_goes_as_each_provider_takes_it(net: ProviderTransport) -> None:
    schema = SCHEMA_DIRECTIVE["json_schema"]["schema"]
    for provider in PROVIDERS:
        plan_reply(net, provider)
        await _generate(build(net, provider), response_format=SCHEMA_DIRECTIVE)
    anthropic_body = net.last(ANTHROPIC_HOST)
    assert anthropic_body["output_config"]["format"]["type"] == "json_schema"
    assert anthropic_body["output_config"]["format"]["schema"]["properties"] == schema["properties"]
    openai_format = net.last(OPENAI_HOST)["response_format"]
    assert openai_format["type"] == "json_schema" and openai_format["json_schema"]["schema"] == schema
    openrouter_body = net.last(OPENROUTER_HOST)
    assert openrouter_body["response_format"]["json_schema"]["schema"] == schema
    assert openrouter_body["provider"]["require_parameters"] is True, "an upstream that cannot honour it is refused"


async def test_json_mode_goes_where_a_provider_has_one_and_nowhere_on_anthropic(net: ProviderTransport) -> None:
    for provider in PROVIDERS:
        plan_reply(net, provider)
        await _generate(build(net, provider), response_format={"type": "json_object"})
    assert net.last(OPENAI_HOST)["response_format"] == {"type": "json_object"}
    assert net.last(OPENROUTER_HOST)["response_format"] == {"type": "json_object"}
    assert "output_config" not in net.last(ANTHROPIC_HOST), "Anthropic has no JSON mode; an empty schema forces {}"


async def test_an_unknown_directive_is_refused_before_anything_is_sent(net: ProviderTransport) -> None:
    client = build(net, "openai")
    with pytest.raises(ModelsClientRefused, match="response_format"):
        await client.generate(system="s", user="u", response_format={"type": "xml"})
    with pytest.raises(ModelsClientRefused, match="json_schema"):
        await client.generate(system="s", user="u", response_format={"type": "json_schema", "json_schema": {}})
    assert net.requests == []


# --- reasoning --------------------------------------------------------------------------------------------


async def test_anthropic_s_effort_rides_one_output_config_beside_the_schema(net: ProviderTransport) -> None:
    """A second bind would replace the first, losing the schema; the adapter merges them."""
    plan_reply(net, "anthropic")
    client = build(net, "anthropic", reasoning_effort="low", temperature=0.2)
    await _generate(client, response_format=SCHEMA_DIRECTIVE)
    body = net.last(ANTHROPIC_HOST)
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["temperature"] == 0.2
    await _generate(client)
    assert net.last(ANTHROPIC_HOST)["output_config"] == {"effort": "low"}


async def test_anthropic_s_none_turns_thinking_off_and_a_budget_turns_it_on_without_a_temperature(
    net: ProviderTransport,
) -> None:
    plan_reply(net, "anthropic")
    await _generate(build(net, "anthropic", reasoning_effort="none"))
    assert net.last(ANTHROPIC_HOST)["thinking"] == {"type": "disabled"}
    budgeted = build(net, "anthropic", reasoning_max_tokens=2048, max_tokens=8192, temperature=0.2)
    result = await _generate(budgeted)
    body = net.last(ANTHROPIC_HOST)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in body, "the API refuses a temperature beside extended thinking"
    assert result.temperature is None and budgeted.temperature is None


async def test_openai_and_openrouter_take_the_effort_word(net: ProviderTransport) -> None:
    plan_reply(net, "openai")
    await _generate(build(net, "openai", model="gpt-5-mini", reasoning_effort="minimal"))
    assert net.last(OPENAI_HOST)["reasoning_effort"] == "minimal"
    plan_reply(net, "openrouter")
    for word in ("max", "minimal", "none"):
        await _generate(build(net, "openrouter", reasoning_effort=word))
        assert net.last(OPENROUTER_HOST)["reasoning"] == {"effort": word}


@pytest.mark.parametrize(
    ("provider", "settings", "setting", "says"),
    [
        ("anthropic", {"reasoning_effort": "minimal"}, "reasoning_effort", "ask for 'low'"),
        ("openai", {"reasoning_effort": "max"}, "reasoning_effort", "ask for 'xhigh'"),
        ("openai", {"reasoning_max_tokens": 4096}, "reasoning_max_tokens", "no reasoning token budget"),
        ("openrouter", {"reasoning_max_tokens": 4096}, "reasoning_max_tokens", "strips reasoning.max_tokens"),
        ("anthropic", {"reasoning_effort": "low", "reasoning_max_tokens": 2048}, "reasoning_effort", "never both"),
        ("anthropic", {"reasoning_max_tokens": 512}, "reasoning_max_tokens", "below 1024"),
        ("anthropic", {"reasoning_max_tokens": 4096, "max_tokens": 4096}, "reasoning_max_tokens", "not below"),
        ("anthropic", {"reasoning_max_tokens": 70_000}, "reasoning_max_tokens", "default output cap"),
    ],
)
def test_a_reasoning_setting_a_provider_cannot_carry_is_refused_by_name(
    net: ProviderTransport, provider: str, settings: dict[str, Any], setting: str, says: str
) -> None:
    with pytest.raises(ModelsClientRefused, match=says) as refused:
        build(net, provider, **settings)
    assert refused.value.setting == setting and refused.value.provider == provider
    assert net.requests == []


@pytest.mark.parametrize(
    "keyword",
    ["max_retries", "breaker", "disable_streaming", "streaming", "thinking", "effort", "reasoning", "output_config"],
)
def test_a_provider_keyword_that_would_bypass_the_adapter_is_refused(net: ProviderTransport, keyword: str) -> None:
    with pytest.raises(ModelsClientRefused, match=keyword):
        build(net, "anthropic", **{keyword: object()})


def test_a_model_the_registry_cannot_place_needs_a_provider() -> None:
    with pytest.raises(ModelsClientRefused, match="pass provider=") as refused:
        models_client("vendor/unheard-of-model", api_key="k", prices=FakeCatalogue())
    assert refused.value.setting == "model_id"


def test_a_provider_the_adapter_does_not_build_for_is_refused() -> None:
    with pytest.raises(ModelsClientRefused, match="bring a completion client of your own"):
        models_client(DEFAULT_EMBEDDING_MODEL, api_key="k", provider="voyageai", prices=FakeCatalogue())


def test_the_claude_cli_route_is_refused_with_its_reason() -> None:
    """A subscription token routes to the CLI, whose completions read as finished whatever stopped them."""
    with pytest.raises(ModelsClientRefused, match="read as finished") as refused:
        models_client(DEFAULT_CHAT_MODEL, api_key="sk-ant-oat01-subscription", prices=FakeCatalogue())
    assert refused.value.setting == "api_key"


# --- what reaches the built model ---------------------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
def test_retries_and_streaming_are_off_on_the_built_model(net: ProviderTransport, provider: str) -> None:
    bound = build(net, provider).chat_model.bound
    assert bound.max_retries == 0
    assert bound.disable_streaming is True


@pytest.mark.parametrize(
    ("provider", "failure"),
    [
        ("anthropic", status(529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})),
        ("openai", status(500, {"error": {"message": "boom", "type": "server_error"}})),
        ("openrouter", status(502, {"error": {"code": 502, "message": "bad gateway"}})),
    ],
)
async def test_a_retryable_failure_is_sent_once(net: ProviderTransport, provider: str, failure: Any) -> None:
    """The SDKs would retry this (OpenRouter's, given no retry setting, for up to an hour); the adapter sends it once."""
    net.plan(HOSTS[provider], failure)
    with pytest.raises(Exception):  # noqa: B017 -- each SDK's own error type; the count is what is held
        await _generate(build(net, provider))
    assert len(net.sent(HOSTS[provider])) == 1


def test_openrouter_s_calls_carry_the_sdk_s_no_retry(net: ProviderTransport) -> None:
    """``max_retries=0`` alone leaves the OpenRouter SDK on its own default backoff; ``retries=None`` turns it off."""
    assert build(net, "openrouter").chat_model.bound.model_kwargs == {"retries": None}
    kept = build(net, "openrouter", model_kwargs={"top_a": 0.5}).chat_model.bound
    assert kept.model_kwargs == {"top_a": 0.5, "retries": None}
    with pytest.raises(ModelsClientRefused, match="model_kwargs.retries"):
        build(net, "openrouter", model_kwargs={"retries": object()})


async def test_the_client_s_breaker_never_opens(net: ProviderTransport) -> None:
    """Six failures in a row, past models' default threshold of five, and the seventh call still reaches the provider."""
    net.plan(ANTHROPIC_HOST, status(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}))
    client = build(net, "anthropic")
    for _ in range(7):
        with pytest.raises(anthropic.BadRequestError):
            await _generate(client)
    assert len(net.sent(ANTHROPIC_HOST)) == 7


async def test_the_process_wide_breaker_neither_trips_the_client_nor_is_tripped_by_it(net: ProviderTransport) -> None:
    """models' default breaker is keyed by provider and key; the adapter's client is on neither side of it."""
    bad = status(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "x"}})
    net.plan(ANTHROPIC_HOST, bad)
    tripped_key, untouched_key = "sk-ant-api03-tripped", "sk-ant-api03-untouched"
    shared_url = net.base_url(ANTHROPIC_HOST)
    plain = create_chat_model(DEFAULT_CHAT_MODEL, api_key=tripped_key, max_retries=0, base_url=shared_url)
    for _ in range(5):
        with pytest.raises(anthropic.BadRequestError):
            await plain.ainvoke("x")
    with pytest.raises(CircuitOpenError):
        await plain.ainvoke("x")
    before = len(net.sent(ANTHROPIC_HOST))
    with pytest.raises(anthropic.BadRequestError):
        await _generate(build(net, "anthropic", api_key=tripped_key, base_url=shared_url))
    assert len(net.sent(ANTHROPIC_HOST)) == before + 1, "the open process-wide breaker did not stop the client"

    adapter = build(net, "anthropic", api_key=untouched_key, base_url=shared_url)
    for _ in range(6):
        with pytest.raises(anthropic.BadRequestError):
            await _generate(adapter)
    net.plan(ANTHROPIC_HOST, ok(anthropic_reply()))
    other = create_chat_model(DEFAULT_CHAT_MODEL, api_key=untouched_key, max_retries=0, base_url=shared_url)
    await other.ainvoke("x")  # the adapter's six failures left the process-wide breaker for this key closed


# --- temperature ----------------------------------------------------------------------------------------


async def test_the_temperature_reported_is_the_one_sent(net: ProviderTransport) -> None:
    """LangChain drops a temperature gpt-5 refuses at build; the completion says none was sent."""
    plan_reply(net, "openai")
    kept = await _generate(build(net, "openai", model="gpt-4o", temperature=0.2))
    assert kept.temperature == 0.2 and net.last(OPENAI_HOST)["temperature"] == 0.2
    dropped = await _generate(build(net, "openai", model="gpt-5-mini", temperature=0.2))
    assert dropped.temperature is None and "temperature" not in net.last(OPENAI_HOST)
    plan_reply(net, "openrouter")
    routed = await _generate(build(net, "openrouter", temperature=0.0))
    assert routed.temperature == 0.0 and net.last(OPENROUTER_HOST)["temperature"] == 0.0
    plan_reply(net, "anthropic")
    unset = await _generate(build(net, "anthropic"))
    assert unset.temperature is None and "temperature" not in net.last(ANTHROPIC_HOST)


# --- pricing --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "model", "per_input", "per_output", "row"),
    [
        ("anthropic", DEFAULT_CHAT_MODEL, 3.0, 15.0, f"anthropic/{DEFAULT_CHAT_MODEL}"),
        ("openai", "gpt-4o", 2.5, 10.0, "openai/gpt-4o"),
    ],
)
async def test_a_call_is_priced_from_the_catalogue_row_for_the_requested_model(
    net: ProviderTransport, provider: str, model: str, per_input: float, per_output: float, row: str
) -> None:
    plan_reply(net, provider, **({"cache_read": 0} if provider == "anthropic" else {}))
    result = await _generate(build(net, provider, model=model))
    assert result.input_tokens is not None and result.output_tokens is not None
    expected = (result.input_tokens * per_input + result.output_tokens * per_output) / 1_000_000
    assert result.cost_usd == pytest.approx(expected)
    assert result.price_source == f"openrouter:{row}"
    assert result.pricing_failure is None


async def test_the_catalogue_is_read_once_per_client(net: ProviderTransport) -> None:
    plan_reply(net, "anthropic")
    catalogue = FakeCatalogue()
    client = build(net, "anthropic", prices=catalogue)
    for _ in range(3):
        await _generate(client)
    assert len(catalogue.fetches) == 1


@pytest.mark.parametrize(
    ("model", "catalogue", "reason"),
    [
        ("nowhere-1", FakeCatalogue(), "no price catalogue row matches"),
        ("unpriced", FakeCatalogue(), "carries no input or output rate"),
        (DEFAULT_CHAT_MODEL, FakeCatalogue(failure=httpx.ConnectError("offline")), "could not be fetched"),
    ],
)
async def test_an_unpriced_call_is_none_with_its_reason(
    net: ProviderTransport, model: str, catalogue: FakeCatalogue, reason: str
) -> None:
    plan_reply(net, "anthropic")
    client = build(net, "anthropic", model=model, prices=catalogue)
    result = await _generate(client)
    assert result.cost_usd is None and result.price_source is None
    assert result.pricing_failure is not None and reason in result.pricing_failure
    assert client.last_pricing_failure == result.pricing_failure
    assert client.price_ceiling(system="s", user="u") is None


async def test_a_reply_without_token_counts_is_unpriced_not_free(net: ProviderTransport) -> None:
    plan_reply(net, "openai", usage=False)
    result = await _generate(build(net, "openai"))
    assert (result.input_tokens, result.output_tokens, result.cost_usd) == (None, None, None)
    assert result.pricing_failure is not None and "token count" in result.pricing_failure


async def test_openrouter_s_own_figure_prices_its_calls(net: ProviderTransport) -> None:
    catalogue = FakeCatalogue()
    plan_reply(net, "openrouter", cost=0.00123)
    reported = await _generate(build(net, "openrouter", prices=catalogue))
    assert (reported.cost_usd, reported.price_source) == (0.00123, "openrouter:reported")
    plan_reply(net, "openrouter", cost=0.0001, upstream_cost=0.002)
    byok = await _generate(build(net, "openrouter", prices=catalogue))
    assert (byok.cost_usd, byok.price_source) == (0.0001, "openrouter:reported"), "reported as OpenRouter reported it"
    plan_reply(net, "openrouter", usage=False)
    client = build(net, "openrouter", prices=catalogue)
    silent = await _generate(client)
    assert silent.cost_usd is None and silent.pricing_failure is not None
    assert client.last_pricing_failure == silent.pricing_failure
    assert catalogue.fetches == [], "OpenRouter's calls are priced by its own figure; no catalogue is read"


async def test_the_price_ceiling_needs_the_rates_and_an_output_cap(net: ProviderTransport) -> None:
    plan_reply(net, "anthropic")
    client = build(net, "anthropic", max_tokens=1000)
    assert client.price_ceiling(system="s", user="u") is None, "no rates are known before the first priced call"
    result = await _generate(client)
    ceiling = client.price_ceiling(system="s", user="u")
    assert ceiling is not None and result.cost_usd is not None and ceiling > result.cost_usd
    assert ceiling >= 1000 * 15.0 / 1_000_000
    uncapped = build(net, "openai")
    plan_reply(net, "openai")
    await _generate(uncapped)
    assert uncapped.price_ceiling(system="s", user="u") is None, "OpenAI sends no cap unless one is built in"


# --- served model --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "reply_model", "served"),
    [
        ("~anthropic/claude-sonnet-latest", "", None),
        ("anthropic/claude-sonnet-latest", "", None),
        ("openrouter/auto", "", None),
        (
            "~anthropic/claude-sonnet-latest",
            "anthropic/claude-4.6-sonnet-20260201",
            "anthropic/claude-4.6-sonnet-20260201",
        ),
        ("anthropic/claude-sonnet-4.6", "", "anthropic/claude-sonnet-4.6"),
    ],
)
async def test_openrouter_s_echo_of_a_requested_alias_is_not_a_served_model(
    net: ProviderTransport, requested: str, reply_model: str, served: str | None
) -> None:
    """LangChain fills an empty reply model with the request; for an alias that names the pointer, not the model."""
    plan_reply(net, "openrouter", model=reply_model)
    result = await _generate(build(net, "openrouter", model=requested))
    assert result.model == requested
    assert result.served_model == served


async def test_the_served_model_is_the_reply_s_on_the_direct_providers(net: ProviderTransport) -> None:
    plan_reply(net, "anthropic", model=f"{DEFAULT_CHAT_MODEL}-20260201")
    assert (await _generate(build(net, "anthropic"))).served_model == f"{DEFAULT_CHAT_MODEL}-20260201"
    plan_reply(net, "openai", model=None)
    unnamed = await _generate(build(net, "openai"))
    assert unnamed.served_model is None and unnamed.model == MODELS["openai"]


# --- stop reasons ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "raw", "engine"),
    [
        ("anthropic", "end_turn", "end_turn"),
        ("anthropic", "stop_sequence", "end_turn"),
        ("anthropic", "tool_use", "end_turn"),
        ("anthropic", "max_tokens", "max_tokens"),
        ("anthropic", "model_context_window_exceeded", "truncated"),
        ("anthropic", "refusal", "content_filter"),
        ("anthropic", "pause_turn", "truncated"),
        ("anthropic", None, "truncated"),
        ("anthropic", "a_future_reason", "truncated"),
        ("openai", "stop", "end_turn"),
        ("openai", "tool_calls", "end_turn"),
        ("openai", "length", "max_tokens"),
        ("openai", "content_filter", "content_filter"),
        ("openai", "a_future_reason", "truncated"),
        ("openai", None, "truncated"),
        ("openrouter", "stop", "end_turn"),
        ("openrouter", "length", "max_tokens"),
        ("openrouter", "error", "error"),
        ("openrouter", None, "truncated"),
        ("openrouter", "a_future_reason", "truncated"),
    ],
)
async def test_the_stop_reason_is_the_engine_s_word(
    net: ProviderTransport, provider: str, raw: Any, engine: str
) -> None:
    """A truncation is never read as finished, and output cut short is never read as a failed call.

    An absent or unknown reason, and Anthropic's ``pause_turn``, had output and stopped early for a cause the provider
    did not name: ``truncated``. Only OpenRouter's own ``error`` finish reason, the provider reporting a failure, is
    ``error``.
    """
    if provider == "anthropic":
        plan_reply(net, provider, stop_reason=raw)
    else:
        plan_reply(net, provider, finish_reason=raw)
    result = await _generate(build(net, provider))
    assert result.stop_reason == engine
    # The provider's own word rides beside the engine's, so a `truncated` names what, if anything, was sent.
    assert result.provider_stop_reason == raw


async def test_an_openai_refusal_beside_a_finished_reply_is_the_content_filter(net: ProviderTransport) -> None:
    plan_reply(net, "openai", text=None, finish_reason="stop", refusal="I can't help with that.")
    result = await _generate(build(net, "openai"))
    assert result.stop_reason == "content_filter" and result.content == ""


# --- lifecycle ------------------------------------------------------------------------------------------


async def test_aclose_closes_openrouter_s_own_clients(net: ProviderTransport) -> None:
    plan_reply(net, "openrouter")
    client = build(net, "openrouter")
    await _generate(client)
    configuration = client.chat_model.bound.client.sdk_configuration
    owned_async, owned_sync = configuration.async_client, configuration.client
    async with client:
        pass
    assert owned_async.is_closed and owned_sync.is_closed
    await client.aclose()  # idempotent


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
async def test_aclose_leaves_the_shared_client_open(net: ProviderTransport, provider: str) -> None:
    """LangChain shares one httpx client per base URL across every instance; closing one would break the rest."""
    plan_reply(net, provider)
    url = net.base_url(HOSTS[provider])
    first, second = build(net, provider, base_url=url), build(net, provider, base_url=url)
    await _generate(first)
    await first.aclose()
    await _generate(second)
    await _generate(first)


# --- the price load at boot, and pricing failures -------------------------------------------------------


# parity-with: threetears.evals.schema.out_of_run_spend.OutOfRunSpendStore
class _FakeLedger:
    """Keeps every out-of-run row written: admission writes none, so the tests here read it only to build a budget."""

    def __init__(self) -> None:
        self.rows: list[OutOfRunSpend] = []

    def save_out_of_run_spend(self, spend: OutOfRunSpend, /) -> None:
        self.rows.append(spend)


@pytest.mark.parametrize(
    ("provider", "model", "per_output"),
    [
        ("anthropic", DEFAULT_CHAT_MODEL, 15.0),
        ("openai", "gpt-4o", 10.0),
        ("openrouter", "anthropic/claude-sonnet-4.6", 15.0),
        ("openrouter", "~anthropic/claude-sonnet-latest", 15.0),
    ],
)
async def test_a_loaded_client_answers_a_ceiling_on_every_provider(
    net: ProviderTransport, provider: str, model: str, per_output: float
) -> None:
    """The ceiling is synchronous, so it answers from rates ``load_prices`` loaded, and from nothing before."""
    client = build(net, provider, model=model, max_tokens=1000)
    assert client.price_ceiling(system="s", user="u") is None, "nothing is loaded at construction"
    assert await client.load_prices() is None
    ceiling = client.price_ceiling(system="s", user="u")
    assert ceiling is not None and ceiling >= 1000 * per_output / 1_000_000
    assert net.requests == [], "loading the prices calls no provider"


async def test_an_out_of_run_call_under_a_cap_is_admitted_once_the_prices_are_loaded(net: ProviderTransport) -> None:
    """The engine refuses a call it cannot price under an enforced cap; a fresh client is priceable after the load."""
    client = build(net, "openrouter", model="anthropic/claude-sonnet-4.6", max_tokens=1000)
    budget = OutOfRunBudget(_FakeLedger(), scope_id="s", cap_usd=1.0, blocking_executor=None)
    call = PlannedCall(system="Write the analysis.", user="the bundle")
    with pytest.raises(ValidationFailedError, match="cannot be priced"):
        budget.admit(client, "analysis", [call])
    await client.load_prices()
    assert len(budget.admit(client, "analysis", [call])) == 1


async def test_openrouter_s_loaded_rates_serve_the_ceiling_and_its_cost_stays_its_own(net: ProviderTransport) -> None:
    plan_reply(net, "openrouter", cost=0.00123)
    client = build(net, "openrouter", model="anthropic/claude-sonnet-4.6", max_tokens=1000)
    assert await client.load_prices() is None
    result = await _generate(client)
    assert (result.cost_usd, result.price_source) == (0.00123, "openrouter:reported")
    assert client.price_ceiling(system="s", user="u") is not None


async def test_a_failed_catalogue_fetch_is_reported_logged_and_tried_again(
    net: ProviderTransport, caplog: pytest.LogCaptureFixture
) -> None:
    catalogue = FakeCatalogue(failure=httpx.ConnectError("offline"))
    client = build(net, "anthropic", max_tokens=1000, prices=catalogue)
    with caplog.at_level(logging.WARNING, logger=ADAPTER_LOGGER):
        reason = await client.load_prices()
    assert reason is not None and "could not be fetched" in reason and "offline" in reason
    assert [r.levelno for r in caplog.records if "offline" in r.getMessage()] == [logging.WARNING]
    assert client.price_ceiling(system="s", user="u") is None
    catalogue.failure = None
    assert await client.load_prices() is None, "a failed fetch is not kept: the next load tries again"
    assert client.price_ceiling(system="s", user="u") is not None


async def test_a_call_after_a_failed_fetch_tries_the_catalogue_again(net: ProviderTransport) -> None:
    plan_reply(net, "anthropic", cache_read=0)
    catalogue = FakeCatalogue(failure=httpx.ConnectError("offline"))
    client = build(net, "anthropic", prices=catalogue)
    first = await _generate(client)
    assert (
        first.cost_usd is None and first.pricing_failure is not None and "could not be fetched" in first.pricing_failure
    )
    catalogue.failure = None
    second = await _generate(client)
    assert second.cost_usd is not None and second.pricing_failure is None


async def test_a_catalogue_that_is_not_json_is_reported_as_unreadable(net: ProviderTransport) -> None:
    client = build(net, "anthropic", prices=FakeCatalogue(body="<html>maintenance</html>"))
    reason = await client.load_prices()
    assert reason is not None and "could not be read" in reason


@pytest.mark.parametrize("failure", [KeyError("acct_SECRET"), ValueError("a bug in the host's price source")])
async def test_a_price_source_s_own_exception_at_load_is_raised_not_read_as_a_bad_catalogue(
    net: ProviderTransport, failure: Exception
) -> None:
    """Only the catalogue's own parse is read as unreadable; a host source's ``ValueError`` is its defect."""
    client = build(net, "anthropic", prices=FakeCatalogue(failure=failure))
    with pytest.raises(type(failure)):
        await client.load_prices()


@pytest.mark.parametrize("failure", [KeyError("acct_SECRET"), ValueError("a bug in the host's price source")])
async def test_a_pricing_exception_after_the_call_returns_the_paid_completion_unpriced(
    net: ProviderTransport, caplog: pytest.LogCaptureFixture, failure: Exception
) -> None:
    plan_reply(net, "anthropic")
    client = build(net, "anthropic", prices=FakeCatalogue(failure=failure))
    with caplog.at_level(logging.ERROR, logger=ADAPTER_LOGGER):
        result = await _generate(client)
    assert result.content == '{"score": 4}', "the paid answer is the caller's"
    assert result.cost_usd is None and result.price_source is None
    assert result.pricing_failure is not None
    assert result.pricing_failure.startswith(f"pricing the call raised {type(failure).__name__}")
    assert client.last_pricing_failure == result.pricing_failure
    [logged] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert logged.exc_info is not None and logged.exc_info[1] is failure, "logged with its traceback"


@pytest.mark.parametrize(
    ("provider", "reply"),
    [
        ("anthropic", {"cache_read": 3, "cache_creation": 0}),
        ("anthropic", {"cache_read": 0, "cache_creation": 5}),
        ("openai", {"cached_tokens": 4}),
    ],
)
async def test_a_cached_prompt_s_cost_is_marked_approximate(
    net: ProviderTransport, provider: str, reply: dict[str, Any]
) -> None:
    """The catalogue has no cache rates: a cache read or write is priced at the input rate, and says so."""
    plan_reply(net, provider, **reply)
    result = await _generate(build(net, provider))
    assert result.cost_usd is not None
    assert result.price_source is not None and result.price_source.endswith(";cache-approximate")


@pytest.mark.parametrize(
    ("provider", "reply"),
    [("anthropic", {"cache_read": 0, "cache_creation": 0}), ("openai", {"cached_tokens": 0}), ("openai", {})],
)
async def test_an_uncached_prompt_s_cost_carries_no_marker(
    net: ProviderTransport, provider: str, reply: dict[str, Any]
) -> None:
    plan_reply(net, provider, **reply)
    result = await _generate(build(net, provider))
    assert result.price_source is not None and not result.price_source.endswith(";cache-approximate")


# --- stop reasons beyond the table ------------------------------------------------------------------------


async def test_openrouter_s_provider_stop_reason_is_its_native_finish_reason_when_langchain_carries_it() -> None:
    """The upstream's own word is the raw value; OpenRouter's normalised ``finish_reason`` still decides the mapping."""
    carried = AIMessage(
        content="{",
        response_metadata={"finish_reason": "length", "native_finish_reason": "MAX_TOKENS", "model_name": "x/y"},
    )
    client = ModelsCompletionClient(
        model_name="x/y",
        provider="openrouter",
        chat_model=RunnableBinding(bound=GenericFakeChatModel(messages=iter([carried])), kwargs={}),
        call_output_config={},
        prices=FakeCatalogue(),
    )
    result = await _generate(client)
    assert (result.stop_reason, result.provider_stop_reason) == ("max_tokens", "MAX_TOKENS")


async def test_openrouter_s_finish_reason_is_the_raw_value_while_the_sdk_drops_the_native_one(
    net: ProviderTransport,
) -> None:
    """Pinned: the OpenRouter SDK's typed chunk declares no ``native_finish_reason``, so LangChain never sees it today.

    When an SDK or LangChain upgrade starts carrying it, this fails and the adapter already reports it (above).
    """
    plan_reply(net, "openrouter", finish_reason="length", native_finish_reason="MAX_TOKENS")
    result = await _generate(build(net, "openrouter"))
    assert (result.stop_reason, result.provider_stop_reason) == ("max_tokens", "length")


async def test_a_truncated_reply_logs_the_provider_s_raw_stop_reason(
    net: ProviderTransport, caplog: pytest.LogCaptureFixture
) -> None:
    """The engine stores only its own word, so the raw value is logged where an operator can find it."""

    def truncations() -> list[str]:
        return [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "truncated" in r.getMessage()
        ]

    with caplog.at_level(logging.WARNING, logger=ADAPTER_LOGGER):
        plan_reply(net, "anthropic", stop_reason="pause_turn")
        await _generate(build(net, "anthropic"))
        plan_reply(net, "openai", finish_reason=None)
        await _generate(build(net, "openai"))
        plan_reply(net, "openrouter")
        await _generate(build(net, "openrouter"))
    assert truncations() == [
        f"eval.providers reply read as truncated provider=anthropic model={DEFAULT_CHAT_MODEL} "
        "provider_stop_reason=pause_turn",
        f"eval.providers reply read as truncated provider=openai model={MODELS['openai']} provider_stop_reason=none",
    ], "one line per truncated reply, and none for a finished one"


# --- OpenAI's Responses API ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status_word", "reason", "engine", "raw"),
    [
        ("completed", None, "end_turn", "completed"),
        ("incomplete", "max_output_tokens", "max_tokens", "max_output_tokens"),
        ("incomplete", "content_filter", "content_filter", "content_filter"),
        ("incomplete", "a_future_reason", "truncated", "a_future_reason"),
        ("incomplete", None, "truncated", "incomplete"),
        ("failed", None, "truncated", "failed"),
    ],
)
async def test_a_responses_api_reply_s_stop_is_read_from_its_status(
    net: ProviderTransport, status_word: str, reason: str | None, engine: str, raw: str
) -> None:
    """LangChain's Responses-API result has no ``finish_reason``; its ``status`` and ``incomplete_details`` say why."""
    net.plan(OPENAI_HOST, ok(openai_responses_reply(status=status_word, incomplete_reason=reason)))
    result = await _generate(build(net, "openai", model="gpt-5-pro"))
    assert net.paths[-1][1].endswith("/responses")
    assert (result.stop_reason, result.provider_stop_reason) == (engine, raw)


async def test_the_temperature_a_responses_api_request_carries_is_the_one_reported(net: ProviderTransport) -> None:
    """LangChain drops the temperature from a Responses-API request for a reasoning gpt-5 model, even a kept 1."""
    net.plan(OPENAI_HOST, ok(openai_responses_reply()))
    dropped = build(net, "openai", model="gpt-5-pro", temperature=1.0)
    result = await _generate(dropped)
    assert "temperature" not in net.last(OPENAI_HOST)
    assert result.temperature is None and dropped.temperature is None
    kept = build(net, "openai", model="gpt-4o", temperature=0.3, use_responses_api=True)
    result = await _generate(kept)
    assert net.last(OPENAI_HOST)["temperature"] == 0.3 and result.temperature == 0.3


@pytest.mark.parametrize(
    ("model", "kwargs", "responses"),
    [
        ("gpt-5-pro", {}, True),
        ("gpt-5.2-pro", {}, True),
        ("gpt-5.4-pro", {}, True),
        ("gpt-5.5-pro", {}, True),
        ("gpt-5-codex", {}, True),
        ("gpt-4o", {"use_responses_api": True}, True),
        ("gpt-4o", {"output_version": "responses/v1"}, True),
        ("gpt-4o", {"truncation": "auto"}, True),
        ("gpt-4o", {"include": ["message.output_text.logprobs"]}, True),
        ("gpt-4o", {}, False),
        ("gpt-5-mini", {}, False),
        ("gpt-5-pro", {"use_responses_api": False}, False),
    ],
)
async def test_the_adapter_knows_which_api_langchain_sends_a_model_to(
    net: ProviderTransport, model: str, kwargs: dict[str, Any], responses: bool
) -> None:
    """The adapter's copy of LangChain's routing rule, held to where LangChain actually sends each request.

    On the Responses API a schema directive is refused before anything is sent: models' structured-output
    translation writes Chat Completions' ``response_format``, which that API does not take.
    """
    net.plan(OPENAI_HOST, ok(openai_responses_reply()) if responses else ok(openai_reply()))
    client = build(net, "openai", model=model, **kwargs)
    await _generate(client)
    assert net.paths[-1][1].endswith("/responses" if responses else "/chat/completions")
    sent = len(net.requests)
    if responses:
        with pytest.raises(ModelsClientRefused, match="Responses API") as refused:
            await client.generate(system="s", user="u", response_format=SCHEMA_DIRECTIVE)
        assert refused.value.setting == "response_format"
        assert len(net.requests) == sent, "nothing was sent"
    else:
        await _generate(client, response_format=SCHEMA_DIRECTIVE)
        assert net.paths[-1][1].endswith("/chat/completions")
