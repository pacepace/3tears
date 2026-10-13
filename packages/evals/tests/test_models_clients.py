"""The role factory (``models_clients``) against mocked provider transports: no provider is called.

What the factory owes the engine (:class:`~threetears.evals.kernel.host.CompletionClients`) and a host: one client
per role, model, temperature and reasoning setting, built on first use and kept, with a lease handed out whose
``aclose`` releases nothing; every client of one factory reading the price catalogue once; a provider's key read
from ``api_keys`` or its SDK's environment variable and refused by name at first use, never at construction; a
role with no model, a role the engine does not have, and a role keyword the factory supplies itself refused by
name; a judge built as a judged run records it (``JUDGE_REQUEST_SETTINGS``) or refused where it cannot be; the
boot-time price load that lets every client's ceiling answer; and ``aclose_all`` closing everything the factory
built even when one close fails.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from threetears.evals.providers import PROVIDER_KEY_VARIABLES, ModelsClientRefused, ModelsCompletion, models_clients
from threetears.evals.run import JUDGE_REQUEST_SETTINGS
from threetears.models import DEFAULT_CHAT_MODEL

from packages.evals.tests.models_adapter_support import (
    ANTHROPIC_HOST,
    KEYS,
    OPENAI_HOST,
    FakeCatalogue,
    ProviderTransport,
    mocked_providers,
    plan_reply,
)


@pytest.fixture
def net(monkeypatch: pytest.MonkeyPatch) -> Iterator[ProviderTransport]:
    """The mocked network for one test."""
    yield from mocked_providers(monkeypatch)


async def _generate(client: Any) -> ModelsCompletion:
    result = await client.generate(system="You grade.", user="Grade this.")
    assert isinstance(result, ModelsCompletion)
    return result


def _factory(net: ProviderTransport, **kwargs: Any) -> Any:
    url = net.base_url(ANTHROPIC_HOST)
    kwargs.setdefault("api_keys", {"anthropic": KEYS["anthropic"]})
    kwargs.setdefault("prices", FakeCatalogue())
    return models_clients(defaults={"judge": DEFAULT_CHAT_MODEL}, judge={"base_url": url}, **kwargs)


async def test_the_factory_builds_once_per_key_and_hands_out_leases(net: ProviderTransport) -> None:
    plan_reply(net, "anthropic")
    clients = _factory(net)
    first = clients("judge", None, temperature=0.0)
    again = clients("judge", DEFAULT_CHAT_MODEL, temperature=0.0)
    warmer = clients("judge", None, temperature=0.7)
    assert first.model_name == DEFAULT_CHAT_MODEL
    assert first.chat_model is again.chat_model and first.chat_model is not warmer.chat_model
    async with first:
        await _generate(first)
    await _generate(again), "closing a lease released nothing"
    await clients.aclose_all()


async def test_every_client_of_a_factory_reads_the_catalogue_once(net: ProviderTransport) -> None:
    plan_reply(net, "anthropic")
    catalogue = FakeCatalogue()
    clients = _factory(net, prices=catalogue)
    for temperature in (0.0, 0.5, 1.0):
        await _generate(clients("judge", None, temperature=temperature))
    assert len(catalogue.fetches) == 1


def test_a_missing_key_is_refused_at_first_use_not_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROVIDER_KEY_VARIABLES["openai"], raising=False)
    clients = models_clients(defaults={"analysis": "gpt-4o"})
    with pytest.raises(ModelsClientRefused, match="OPENAI_API_KEY") as refused:
        clients("analysis", None)
    assert refused.value.setting == "api_key" and refused.value.provider == "openai"


def test_a_key_is_read_from_the_provider_s_environment_variable(
    net: ProviderTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROVIDER_KEY_VARIABLES["openrouter"], KEYS["openrouter"])
    clients = models_clients(
        defaults={"simulator": "~anthropic/claude-sonnet-latest"},
        prices=FakeCatalogue(),
        simulator={"provider": "openrouter", "reasoning_effort": "minimal"},
    )
    assert clients("simulator", None).chat_model.bound.reasoning == {"effort": "minimal"}


def test_a_role_with_no_model_is_refused_by_name(net: ProviderTransport) -> None:
    with pytest.raises(ModelsClientRefused, match="defaults\\['analysis'\\]"):
        _factory(net)("analysis", None)


@pytest.mark.parametrize(
    ("kwargs", "setting"),
    [
        ({"defaults": {"grader": "gpt-4o"}}, "grader"),
        ({"defaults": {}, "judge": {"temperature": 0.2}}, "judge.temperature"),
        ({"defaults": {}, "judge": {"api_key": "k"}}, "judge.api_key"),
    ],
)
def test_a_factory_setting_it_supplies_or_a_role_it_does_not_know_is_refused(
    kwargs: dict[str, Any], setting: str
) -> None:
    with pytest.raises(ModelsClientRefused) as refused:
        models_clients(**kwargs)
    assert refused.value.setting == setting


async def test_aclose_all_closes_what_the_factory_built(net: ProviderTransport) -> None:
    plan_reply(net, "openrouter")
    clients = models_clients(
        api_keys={"openrouter": KEYS["openrouter"]},
        defaults={"simulator": "~anthropic/claude-sonnet-latest"},
        prices=FakeCatalogue(),
        simulator={"provider": "openrouter"},
    )
    lease = clients("simulator", None)
    await _generate(lease)
    owned = lease.chat_model.bound.client.sdk_configuration.async_client
    await lease.aclose()
    assert not owned.is_closed, "a lease releases nothing"
    await clients.aclose_all()
    assert owned.is_closed


# --- the judge, as the run records it -------------------------------------------------------------------


async def test_a_judge_with_no_settings_of_its_own_is_built_from_judge_request_settings(
    net: ProviderTransport,
) -> None:
    """What a judged run records is what the judge sends: the output cap, and the budget as Anthropic's thinking."""
    plan_reply(net, "anthropic")
    judge = _factory(net)("judge", None, temperature=0.0)
    assert judge.chat_model.bound.max_tokens == JUDGE_REQUEST_SETTINGS.max_tokens
    await _generate(judge)
    body = net.last(ANTHROPIC_HOST)
    assert body["max_tokens"] == JUDGE_REQUEST_SETTINGS.max_tokens
    assert body["thinking"] == {"type": "enabled", "budget_tokens": JUDGE_REQUEST_SETTINGS.reasoning_max_tokens}


async def test_a_judge_s_own_settings_are_kept(net: ProviderTransport) -> None:
    plan_reply(net, "anthropic")
    url = net.base_url(ANTHROPIC_HOST)
    clients = models_clients(
        api_keys={"anthropic": KEYS["anthropic"]},
        defaults={"judge": DEFAULT_CHAT_MODEL},
        prices=FakeCatalogue(),
        judge={"base_url": url, "max_tokens": 2048, "reasoning_effort": "low"},
    )
    await _generate(clients("judge", None, temperature=0.0))
    body = net.last(ANTHROPIC_HOST)
    assert body["max_tokens"] == 2048 and "thinking" not in body and body["output_config"] == {"effort": "low"}


@pytest.mark.parametrize(
    ("model", "provider"), [("gpt-4o", "openai"), ("~anthropic/claude-sonnet-latest", "openrouter")]
)
def test_a_default_judge_that_cannot_carry_the_budget_is_refused_at_construction(model: str, provider: str) -> None:
    """The run would record a reasoning budget the request never carried: refused by name, before any key is read."""
    judge = {"provider": provider} if provider == "openrouter" else {}
    with pytest.raises(ModelsClientRefused, match="the record would be false") as refused:
        models_clients(defaults={"judge": model}, prices=FakeCatalogue(), judge=judge)
    assert refused.value.setting == "judge" and refused.value.provider == provider


def test_a_judge_the_engine_names_on_such_a_provider_is_refused_when_asked_for(net: ProviderTransport) -> None:
    clients = _factory(net, api_keys={"anthropic": KEYS["anthropic"], "openai": KEYS["openai"]})
    with pytest.raises(ModelsClientRefused, match="JUDGE_REQUEST_SETTINGS") as refused:
        clients("judge", "gpt-4o", temperature=0.0)
    assert refused.value.setting == "judge" and refused.value.provider == "openai"


def test_a_judge_on_such_a_provider_with_settings_passed_explicitly_is_built(net: ProviderTransport) -> None:
    clients = models_clients(
        api_keys={"openai": KEYS["openai"]},
        defaults={"judge": "gpt-5-mini"},
        prices=FakeCatalogue(),
        judge={
            "provider": "openai",
            "base_url": net.base_url(OPENAI_HOST),
            "max_tokens": 4096,
            "reasoning_effort": "high",
        },
    )
    assert clients("judge", None).chat_model.bound.reasoning_effort == "high"


# --- the boot-time price load ---------------------------------------------------------------------------


async def test_load_prices_lets_every_default_role_s_ceiling_answer(net: ProviderTransport) -> None:
    catalogue = FakeCatalogue()
    clients = models_clients(
        api_keys={"anthropic": KEYS["anthropic"], "openai": KEYS["openai"], "openrouter": KEYS["openrouter"]},
        defaults={"judge": DEFAULT_CHAT_MODEL, "analysis": "gpt-4o", "simulator": "anthropic/claude-sonnet-4.6"},
        prices=catalogue,
        judge={"base_url": net.base_url(ANTHROPIC_HOST)},
        analysis={"base_url": net.base_url(OPENAI_HOST), "max_tokens": 4096},
        simulator={"provider": "openrouter", "max_tokens": 4096},
    )
    roles: tuple[Any, ...] = ("judge", "analysis", "simulator")
    before = [clients(role, None) for role in roles]
    assert [lease.price_ceiling(system="s", user="u") for lease in before] == [None, None, None]
    assert await clients.load_prices() == {}
    assert all(lease.price_ceiling(system="s", user="u") is not None for lease in before)
    later = clients("analysis", None, temperature=0.9)
    assert later.price_ceiling(system="s", user="u") is not None, "a client built after the load reads the same book"
    assert len(catalogue.fetches) == 1, "one catalogue read for the whole factory"
    assert net.requests == []
    await clients.aclose_all()


async def test_load_prices_names_each_default_model_left_unpriced(net: ProviderTransport) -> None:
    clients = _factory(net, prices=FakeCatalogue(payload={"data": []}))
    unpriced = await clients.load_prices()
    assert list(unpriced) == [DEFAULT_CHAT_MODEL] and "no price catalogue row matches" in unpriced[DEFAULT_CHAT_MODEL]


# --- shutdown ---------------------------------------------------------------------------------------------


async def test_aclose_all_closes_every_client_when_one_fails_and_raises_the_failures_together(
    net: ProviderTransport,
) -> None:
    clients = models_clients(
        api_keys={"openrouter": KEYS["openrouter"]},
        defaults={"simulator": "~anthropic/claude-sonnet-latest"},
        prices=FakeCatalogue(),
        simulator={"provider": "openrouter"},
    )
    failing, healthy = clients("simulator", None, temperature=0.0), clients("simulator", None, temperature=0.5)
    sdk = failing.chat_model.bound.client

    def refuse(*_: object) -> None:
        raise RuntimeError("the SDK could not close")

    sdk.__exit__ = refuse
    owned = healthy.chat_model.bound.client.sdk_configuration.async_client
    with pytest.raises(ExceptionGroup) as raised:
        await clients.aclose_all()
    assert [str(failure) for failure in raised.value.exceptions] == ["the SDK could not close"]
    assert owned.is_closed, "one failing close left the rest closed"
