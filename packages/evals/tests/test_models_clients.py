"""The role factory (``models_clients``) against mocked provider transports: no provider is called.

What the factory owes the engine (:class:`~threetears.evals.kernel.host.CompletionClients`) and a host: one client
per role, model, temperature and reasoning setting, built on first use and kept, with a lease handed out whose
``aclose`` releases nothing; every client of one factory reading the price catalogue once; a provider's key read
from ``api_keys`` or its SDK's environment variable and refused by name at first use, never at construction; a
role with no model, a role the engine does not have, and a role keyword the factory supplies itself refused by
name; and ``aclose_all`` closing what the factory built.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from threetears.evals.providers import PROVIDER_KEY_VARIABLES, ModelsClientRefused, ModelsCompletion, models_clients
from threetears.models import DEFAULT_CHAT_MODEL

from packages.evals.tests.models_adapter_support import (
    ANTHROPIC_HOST,
    KEYS,
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
