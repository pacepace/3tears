"""The shipped adapter sends the closest setting a model accepts, learns what a provider refuses, and reports it.

Against mocked provider transports (``models_adapter_support``): no provider is called. What each test holds: a
request a provider refuses for a setting it carried is resent once with the closest setting the model then takes,
read per SDK (Anthropic's ``invalid_request_error`` message, OpenAI's ``param`` and ``code``, OpenRouter's 404 under
``require_parameters``), and once served the fact is kept for every client of the factory and logged once; on
OpenRouter the temperature goes first and enforcement last (Discodon's first ``.24`` campaign, 2026-10-13, is
reproduced exactly); what is known before any request comes from 3tears-models' capabilities (read with
``getattr``, including the published layer for a model with no static entry) and OpenRouter's catalogue; and every
completion's ``sent_settings`` and ``temperature`` are what the served request carried. The pure choice of what to
send is ``test_closest_settings.py``; the temperature's own learning, ``test_models_adapter.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

import threetears.models as models_api
from threetears.evals.providers import ModelsCompletion, models_client, models_clients
from threetears.evals.run import JUDGE_REQUEST_SETTINGS
from threetears.evals.schema import DEFAULT_JUDGE_TEMPERATURE, ClientRequestSettings
from threetears.models import ModelCapabilities, ModelTier, ModelType

from packages.evals.tests.models_adapter_support import (
    ANTHROPIC_HOST,
    CATALOGUE,
    KEYS,
    OPENAI_HOST,
    OPENROUTER_HOST,
    FakeCatalogue,
    ProviderTransport,
    anthropic_reply,
    build,
    mocked_providers,
    ok,
    openai_reply,
    openrouter_stream,
    status,
)

#: The logger the adapter writes to.
ADAPTER_LOGGER = "threetears.evals.providers.adapter"

#: The judge's strict reply directive, as every judge call sends it.
JUDGE_DIRECTIVE: dict[str, Any] = {
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

#: A model only the published layer knows, as ``claude-haiku-5-5`` is (no static registry entry).
PUBLISHED_ONLY = "published-only-model"

#: OpenRouter's answer when ``require_parameters`` leaves no endpoint, as Discodon's prod campaign met it.
NO_ENDPOINT = status(
    404, {"error": {"code": 404, "message": "No endpoints found that can handle the requested parameters."}}
)


def anthropic_refusal(message: str) -> Any:
    """Anthropic's 400 for a request it does not take, in its own body shape."""
    return status(400, {"type": "error", "error": {"type": "invalid_request_error", "message": message}})


def openai_refusal(message: str, *, param: str, code: str | None) -> Any:
    """OpenAI's 400 for a request it does not take, ``param`` and ``code`` structured."""
    return status(400, {"error": {"message": message, "type": "invalid_request_error", "param": param, "code": code}})


class _DeclaredCapabilities(ModelCapabilities):
    """A capability entry with the request-setting fields a newer 3tears-models declares (PR #782)."""

    accepts_temperature: bool | None = None
    reasoning_efforts: list[str] | None = None
    reasoning_budget: bool | None = None
    adaptive_reasoning: bool | None = None
    reasoning_disable: bool | None = None
    structured_output: bool | None = None


@pytest.fixture
def net(monkeypatch: pytest.MonkeyPatch) -> Iterator[ProviderTransport]:
    """The mocked network for one test."""
    yield from mocked_providers(monkeypatch)


async def _judge(client: Any) -> ModelsCompletion:
    result = await client.generate(system="You grade.", user="Grade this.", response_format=JUDGE_DIRECTIVE)
    assert isinstance(result, ModelsCompletion)
    return result


def _openrouter_judges(**kwargs: Any) -> Any:
    return models_clients(
        api_keys={"openrouter": KEYS["openrouter"]},
        defaults={"judge": "openai/gpt-6-luna"},
        prices=kwargs.pop("prices", FakeCatalogue()),
        judge={"provider": "openrouter"},
    )


def _declare(monkeypatch: pytest.MonkeyPatch, model: str, **facts: Any) -> None:
    """3tears-models declares ``facts`` for ``model``, as a build with the request-setting fields would."""
    declared = _DeclaredCapabilities(
        model_name=model, model_type=ModelType.CHAT, model_tier=ModelTier.MEDIUM, provider_name="anthropic", **facts
    )
    real = models_api.get_capabilities
    monkeypatch.setattr(models_api, "get_capabilities", lambda asked: declared if asked == model else real(asked))


# --- OpenRouter: Discodon G2's production sequence --------------------------------------------------------------


async def test_an_openrouter_judge_whose_endpoints_take_no_temperature_is_served_without_one(
    net: ProviderTransport, caplog: pytest.LogCaptureFixture
) -> None:
    """Discodon's first ``.24`` campaign (2026-10-13), exactly: every judge call 404'd, and no longer fails.

    The engine asks every judge call for ``DEFAULT_JUDGE_TEMPERATURE`` under the strict directive
    (``require_parameters``), and no endpoint serving ``openai/gpt-6-luna`` takes a temperature. Dropping the
    temperature alone was served live, and enforcement stays: the completion records no temperature, strict
    settings, and every later call of the factory, on any of its clients, goes without the temperature at once.
    """
    net.plan(OPENROUTER_HOST, NO_ENDPOINT, ok(openrouter_stream(model="openai/gpt-6-luna", upstream="OpenAI")))
    clients = _openrouter_judges()
    judge = clients("judge", None, temperature=DEFAULT_JUDGE_TEMPERATURE)
    with caplog.at_level(logging.WARNING, logger=ADAPTER_LOGGER):
        result = await _judge(judge)
        later = await _judge(judge)
        other = await _judge(clients("judge", None, temperature=0.7))
    refused, served, *after = net.sent(OPENROUTER_HOST)
    assert refused["temperature"] == DEFAULT_JUDGE_TEMPERATURE
    assert refused["provider"]["require_parameters"] is True and served["provider"]["require_parameters"] is True
    assert "temperature" not in served and served["reasoning"] == {"effort": "high"}
    assert served["messages"] == refused["messages"], "the same request, less the temperature"
    assert len(after) == 2 and all("temperature" not in body for body in after), "learned for the factory"
    for completion in (result, later, other):
        assert completion.temperature is None
        assert completion.sent_settings == JUDGE_REQUEST_SETTINGS, "strict, effort high, the judge's cap"
    logged = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "refuses" in r.getMessage()]
    assert len(logged) == 1 and "refuses a temperature" in logged[0]
    assert "provider=openrouter model=openai/gpt-6-luna registry=unknown" in logged[0]
    assert "No endpoints found" in logged[0], "with OpenRouter's message"


async def test_openrouter_drops_enforcement_only_when_the_request_without_a_temperature_still_finds_no_endpoint(
    net: ProviderTransport, caplog: pytest.LogCaptureFixture
) -> None:
    """Decision 8: a strict call the router cannot place is resent once without enforcement, and says so."""
    net.plan(OPENROUTER_HOST, NO_ENDPOINT, NO_ENDPOINT, ok(openrouter_stream()))
    judge = _openrouter_judges()("judge", None, temperature=DEFAULT_JUDGE_TEMPERATURE)
    with caplog.at_level(logging.WARNING, logger=ADAPTER_LOGGER):
        result = await _judge(judge)
        later = await _judge(judge)
    refused, without_temperature, unenforced, after = net.sent(OPENROUTER_HOST)
    assert refused["temperature"] == DEFAULT_JUDGE_TEMPERATURE and refused["provider"]["require_parameters"] is True
    assert "temperature" not in without_temperature and without_temperature["provider"]["require_parameters"] is True
    assert "temperature" not in unenforced and unenforced["provider"]["require_parameters"] is False
    assert unenforced["response_format"]["type"] == "json_schema", "the format still goes, unenforced"
    assert after == {**unenforced, "messages": after["messages"]}, "the served request's shape, at once"
    unenforced_settings = JUDGE_REQUEST_SETTINGS.model_copy(update={"strict_output": False})
    assert result.sent_settings == later.sent_settings == unenforced_settings
    assert result.temperature is None
    logged = [r.getMessage() for r in caplog.records if "refuses" in r.getMessage()]
    assert [m.split(" provider=")[0] for m in logged] == [
        "eval.providers model refuses a temperature",
        "eval.providers model refuses require_parameters",
    ], "each step learned, once"


async def test_openrouter_s_catalogue_says_what_a_model_takes_before_any_request(net: ProviderTransport) -> None:
    """Once the catalogue is read (``load_prices``), a parameter it does not list for the model is never sent."""
    listed = {
        "id": "openai/gpt-6-luna",
        "pricing": {"prompt": "0.000001", "completion": "0.000004"},
        "supported_parameters": ["max_tokens", "reasoning", "response_format", "structured_outputs"],
    }
    catalogue = FakeCatalogue(payload={"data": [*CATALOGUE["data"], listed]})
    net.plan(OPENROUTER_HOST, ok(openrouter_stream()))
    judge = _openrouter_judges(prices=catalogue)("judge", None, temperature=DEFAULT_JUDGE_TEMPERATURE)
    assert judge.chat_model.bound.temperature == DEFAULT_JUDGE_TEMPERATURE, "built as asked: nothing read yet"
    assert await judge.load_prices() is None
    result = await _judge(judge)
    (body,) = net.sent(OPENROUTER_HOST)
    assert "temperature" not in body and body["provider"]["require_parameters"] is True
    assert result.temperature is None and result.sent_settings == JUDGE_REQUEST_SETTINGS


async def test_a_404_on_a_call_that_does_not_require_every_parameter_is_raised(net: ProviderTransport) -> None:
    net.plan(OPENROUTER_HOST, NO_ENDPOINT)
    client = build(net, "openrouter", temperature=0.0)
    with pytest.raises(Exception):  # noqa: B017 -- the SDK's own error type; the count is what is held
        await client.generate(system="s", user="u")
    assert len(net.sent(OPENROUTER_HOST)) == 1


# --- Anthropic: thinking and effort, learned ----------------------------------------------------------------------


async def test_an_anthropic_model_that_refuses_a_budget_is_sent_the_effort_word_its_share_maps_to(
    net: ProviderTransport,
) -> None:
    """``thinking.type.enabled`` refused (Opus 4.7+, Sonnet 5): the judge's budget goes as effort ``high``."""
    refusal = anthropic_refusal(
        '"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" and '
        '"output_config.effort" to control thinking behavior.'
    )
    net.plan(ANTHROPIC_HOST, refusal, ok(anthropic_reply()))
    client = build(net, "anthropic", max_tokens=10240, reasoning_max_tokens=8192)
    first = await _judge(client)
    second = await _judge(client)
    refused, resent, later = net.sent(ANTHROPIC_HOST)
    assert refused["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    for body in (resent, later):
        assert body["thinking"] == {"type": "adaptive"} and body["output_config"]["effort"] == "high"
        assert body["output_config"]["format"]["type"] == "json_schema"
    sent = ClientRequestSettings(max_tokens=10240, reasoning_effort="high", strict_output=True)
    assert first.sent_settings == second.sent_settings == sent


async def test_an_anthropic_model_that_cannot_turn_thinking_off_is_sent_its_lowest_effort(
    net: ProviderTransport,
) -> None:
    net.plan(
        ANTHROPIC_HOST,
        anthropic_refusal('"thinking.type.disabled" is not supported for this model.'),
        ok(anthropic_reply()),
    )
    result = await _judge(build(net, "anthropic", max_tokens=4096, reasoning_effort="none"))
    refused, resent = net.sent(ANTHROPIC_HOST)
    assert refused["thinking"] == {"type": "disabled"} and "effort" not in refused.get("output_config", {})
    assert resent["thinking"] == {"type": "adaptive"} and resent["output_config"]["effort"] == "low"
    assert result.sent_settings == ClientRequestSettings(max_tokens=4096, reasoning_effort="low", strict_output=True)


async def test_an_anthropic_model_that_takes_no_effort_is_sent_a_budget_by_the_word_s_share(
    net: ProviderTransport,
) -> None:
    """Haiku 4.5's shape: effort and adaptive thinking are refused, a budget is taken; ``high`` is 0.80 of the cap."""
    net.plan(
        ANTHROPIC_HOST,
        anthropic_refusal('"thinking.type.adaptive" is not supported for this model.'),
        anthropic_refusal("output_config.effort: this model does not support effort."),
        ok(anthropic_reply()),
    )
    result = await _judge(build(net, "anthropic", max_tokens=10240, reasoning_effort="high"))
    adaptive, effort_alone, budgeted = net.sent(ANTHROPIC_HOST)
    assert adaptive["thinking"] == {"type": "adaptive"} and adaptive["output_config"]["effort"] == "high"
    assert "thinking" not in effort_alone and effort_alone["output_config"]["effort"] == "high"
    assert budgeted["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert "effort" not in budgeted["output_config"]
    assert result.sent_settings == ClientRequestSettings(
        max_tokens=10240, reasoning_max_tokens=8192, strict_output=True
    )


async def test_a_refusal_naming_a_setting_the_request_did_not_carry_is_raised(net: ProviderTransport) -> None:
    """The budget refusal, met by a request that carried no budget, is not the adapter's to answer."""
    net.plan(ANTHROPIC_HOST, anthropic_refusal('"thinking.type.enabled" is not supported for this model.'))
    with pytest.raises(Exception):  # noqa: B017 -- the SDK's own error type; the count is what is held
        await _judge(build(net, "anthropic", max_tokens=4096))
    assert len(net.sent(ANTHROPIC_HOST)) == 1


# --- OpenAI: the effort word and the reply format, learned --------------------------------------------------------


async def test_an_openai_model_that_refuses_an_effort_word_is_sent_the_closest_it_lists(
    net: ProviderTransport,
) -> None:
    refusal = openai_refusal(
        "Unsupported value: 'reasoning_effort' does not support 'xhigh' with this model. Supported values are: "
        "'low', 'medium', and 'high'.",
        param="reasoning_effort",
        code="unsupported_value",
    )
    net.plan(OPENAI_HOST, refusal, ok(openai_reply()))
    client = build(net, "openai", model="gpt-5-mini", max_tokens=4096, reasoning_effort="xhigh")
    first = await _judge(client)
    await _judge(client)
    assert [body["reasoning_effort"] for body in net.sent(OPENAI_HOST)] == ["xhigh", "high", "high"]
    assert first.sent_settings == ClientRequestSettings(max_tokens=4096, reasoning_effort="high", strict_output=True)


async def test_an_openai_model_that_takes_no_schema_is_sent_json_mode_unenforced(net: ProviderTransport) -> None:
    refusal = openai_refusal(
        "Invalid parameter: 'response_format' of type 'json_schema' is not supported with this model.",
        param="response_format",
        code=None,
    )
    net.plan(OPENAI_HOST, refusal, ok(openai_reply()))
    with pytest.warns(UserWarning, match="structured output"):  # LangChain's own, as it re-raises the refusal
        result = await _judge(build(net, "openai", model="gpt-4o", max_tokens=4096))
    refused, resent = net.sent(OPENAI_HOST)
    assert refused["response_format"]["type"] == "json_schema"
    assert resent["response_format"] == {"type": "json_object"}
    assert result.sent_settings == ClientRequestSettings(max_tokens=4096, strict_output=False)


async def test_a_completion_built_without_an_output_cap_reports_no_sent_settings(net: ProviderTransport) -> None:
    """A settings value names a cap, and an OpenAI request built with none carried none: unverified, not guessed."""
    net.plan(OPENAI_HOST, ok(openai_reply()))
    result = await _judge(build(net, "openai", model="gpt-4o"))
    assert result.sent_settings is None


# --- what 3tears-models declares ----------------------------------------------------------------------------------


async def test_3tears_models_declared_facts_are_followed_before_any_request(
    net: ProviderTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read with ``getattr``: a models build with the request-setting fields is followed with no refusal at all."""
    _declare(
        monkeypatch,
        "declared-model",
        accepts_temperature=False,
        reasoning_budget=False,
        adaptive_reasoning=True,
        max_output_tokens=4096,
    )
    net.plan(ANTHROPIC_HOST, ok(anthropic_reply()))
    client = build(
        net, "anthropic", model="declared-model", temperature=0.2, max_tokens=8192, reasoning_max_tokens=2048
    )
    result = await _judge(client)
    (body,) = net.sent(ANTHROPIC_HOST)
    assert body["max_tokens"] == 4096 and "temperature" not in body
    assert body["thinking"] == {"type": "adaptive"} and body["output_config"]["effort"] == "low", "2048 of 8192"
    assert result.sent_settings == ClientRequestSettings(max_tokens=4096, reasoning_effort="low", strict_output=True)
    assert result.temperature is None and client.temperature is None


async def test_a_model_with_no_static_entry_is_read_from_what_its_provider_published(
    net: ProviderTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model with no static registry entry (``claude-haiku-5-5``, the examples' model) is read from what its provider
    published: models' published layer, filled from Anthropic's Models API. The id here stands in for it."""
    published = SimpleNamespace(
        accepts_temperature=False,
        reasoning_efforts=["low", "medium", "high", "xhigh", "max"],
        reasoning_budget=False,
        adaptive_reasoning=True,
        reasoning_disable=True,
        structured_output=True,
        max_output_tokens=None,
    )
    assert models_api.get_capabilities(PUBLISHED_ONLY) is None
    monkeypatch.setattr(
        models_api,
        "get_published_capabilities",
        lambda model: published if model == PUBLISHED_ONLY else None,
        raising=False,
    )
    net.plan(ANTHROPIC_HOST, ok(anthropic_reply()))
    client = build(
        net,
        "anthropic",
        model=PUBLISHED_ONLY,
        temperature=DEFAULT_JUDGE_TEMPERATURE,
        max_tokens=JUDGE_REQUEST_SETTINGS.max_tokens,
        reasoning_effort="high",
    )
    result = await _judge(client)
    (body,) = net.sent(ANTHROPIC_HOST)
    assert "temperature" not in body, "published: no temperature, so none is sent and nothing is refused"
    assert body["thinking"] == {"type": "adaptive"} and body["output_config"]["effort"] == "high"
    assert result.temperature is None and result.sent_settings == JUDGE_REQUEST_SETTINGS


async def test_a_refusal_that_contradicts_the_registry_wins_and_is_logged_as_its_error(
    net: ProviderTransport, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _declare(monkeypatch, "misdeclared-model", accepts_temperature=True)
    net.plan(ANTHROPIC_HOST, anthropic_refusal("`temperature` is deprecated for this model."), ok(anthropic_reply()))
    client = build(net, "anthropic", model="misdeclared-model", temperature=0.0)
    with caplog.at_level(logging.WARNING, logger=ADAPTER_LOGGER):
        await client.generate(system="s", user="u")
    assert "registry=accepts (a registry error to correct in 3tears-models)" in caplog.text
    assert client.temperature is None


def test_a_declared_fact_of_the_wrong_type_is_a_models_defect_not_an_unknown(
    net: ProviderTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = SimpleNamespace(accepts_temperature="yes")
    monkeypatch.setattr(models_api, "get_capabilities", lambda model: broken)
    with pytest.raises(TypeError, match="accepts_temperature"):
        models_client(
            "broken-model", api_key=KEYS["anthropic"], provider="anthropic", base_url=net.base_url("anthropic")
        )
