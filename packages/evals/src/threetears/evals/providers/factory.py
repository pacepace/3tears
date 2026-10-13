"""The host-profile factory: :func:`models_clients`, the engine's role factory over :func:`models_client`.

What a host that brings no client of its own hands :class:`~threetears.evals.kernel.host.EvalHost` as
``clients``. The engine asks it for a client per role, model and temperature
(:class:`~threetears.evals.kernel.host.CompletionClients`), uses it for one unit of work, and closes what it was
handed. Building a chat model per unit of work would rebuild an SDK client each time, so the factory keeps one
:class:`~threetears.evals.providers.ModelsCompletionClient` per role, model, temperature and reasoning setting for
its own life and hands the engine a lease on it, whose ``aclose`` releases nothing; :meth:`ModelsClients.aclose_all`
closes what the factory built, once, when the host shuts down.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Final, Self, get_args

from threetears.evals.kernel.host import CompletionRole
from threetears.evals.schema import BoundCompletionClient, KnownReasoningEffort
from threetears.models.price_lookup import HttpClient

from threetears.evals.providers.adapter import (
    PROVIDER_KEY_VARIABLES,
    CatalogueFetches,
    ModelsClientRefused,
    ModelsCompletion,
    ModelsCompletionClient,
    SupportedProvider,
    models_client,
    resolve_provider,
)

__all__ = ["ModelsClients", "models_clients"]

#: Every role the engine asks a client for (:data:`~threetears.evals.kernel.host.CompletionRole`).
_ROLES: Final[frozenset[str]] = frozenset(get_args(CompletionRole))

#: Settings a role's keywords may not carry, and why: each is the engine's or the factory's to supply.
_NOT_PER_ROLE: Final[Mapping[str, str]] = {
    "temperature": "the engine names the temperature on each request for a client",
    "api_key": "keys come from api_keys or the provider's environment variable",
    "prices": "the factory's prices= serves every role",
    "model_id": "the role's model is defaults[role], or the one the engine names",
}

#: A cached client's key: role, model, temperature, reasoning word and reasoning budget.
_ClientKey = tuple[str, str, float | None, str | None, int | None]


class _Lease:
    """The client the engine is handed: the factory's client, with an ``aclose`` that releases nothing.

    The engine closes every client it is handed when its unit of work ends; the factory reuses the client for the
    next one, so the lease forwards everything else to it and leaves the release to
    :meth:`ModelsClients.aclose_all`.

    Attributes:
        model_name: The model the client calls.
    """

    def __init__(self, client: ModelsCompletionClient) -> None:
        self._client = client
        self.model_name = client.model_name

    @property
    def chat_model(self) -> Any:  # Any: LangChain's binding, typed as ModelsCompletionClient.chat_model types it
        """The factory's client's chat model (:attr:`ModelsCompletionClient.chat_model`)."""
        return self._client.chat_model

    @property
    def last_pricing_failure(self) -> str | None:
        """The factory's client's :attr:`ModelsCompletionClient.last_pricing_failure`."""
        return self._client.last_pricing_failure

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> ModelsCompletion:
        """Send through the factory's client (:meth:`ModelsCompletionClient.generate`)."""
        return await self._client.generate(system=system, user=user, response_format=response_format)

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """The factory's client's ceiling (:meth:`ModelsCompletionClient.price_ceiling`)."""
        return self._client.price_ceiling(system=system, user=user, response_format=response_format)

    async def aclose(self) -> None:
        """Release nothing: the factory owns the client and closes it in :meth:`ModelsClients.aclose_all`."""

    async def __aenter__(self) -> Self:
        """Enter a scope; leaving it releases nothing."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Leave the scope; the factory's client stays open."""


class ModelsClients:
    """The engine's role factory over :func:`~threetears.evals.providers.models_client`: :func:`models_clients` builds one.

    Implements :class:`~threetears.evals.kernel.host.CompletionClients`. A client is built the first time a role,
    model, temperature and reasoning setting is asked for, and kept: later requests for the same key get a lease
    on the same client. Every client of one factory reads the price catalogue through one cache, so it is fetched
    once per factory.
    """

    def __init__(
        self,
        *,
        api_keys: Mapping[str, str] | None,
        defaults: Mapping[CompletionRole, str],
        prices: HttpClient | None,
        role_settings: Mapping[str, Mapping[str, Any]],  # Any: models_client's keywords, per role
    ) -> None:
        """Hold the configuration; nothing is built and no key is read until a client is asked for.

        Args:
            api_keys: Provider -> key, or ``None`` to read every key from the providers' environment variables.
            defaults: Role -> the model a request naming none resolves to.
            prices: The price client every role's catalogue reads go through, or ``None``.
            role_settings: Role -> keywords for :func:`~threetears.evals.providers.models_client`.

        Raises:
            ModelsClientRefused: A role that is not one of the engine's, or a role keyword the factory supplies.
        """
        if unknown := sorted({*defaults, *role_settings} - _ROLES):
            raise ModelsClientRefused(
                unknown[0], f"is not a role the engine asks a client for ({', '.join(sorted(_ROLES))})"
            )
        for role, settings in sorted(role_settings.items()):
            for keyword in sorted(settings):
                if keyword in _NOT_PER_ROLE:
                    raise ModelsClientRefused(f"{role}.{keyword}", _NOT_PER_ROLE[keyword])
        self._api_keys = dict(api_keys) if api_keys is not None else {}
        self._defaults: dict[CompletionRole, str] = dict(defaults)
        self._role_settings = {role: dict(settings) for role, settings in role_settings.items()}
        self._catalogue = CatalogueFetches(prices)
        self._clients: dict[_ClientKey, ModelsCompletionClient] = {}

    def _key_for(self, provider: SupportedProvider) -> str:
        """The key for ``provider``: the mapping's, else its SDK's environment variable, else a refusal by name."""
        if provider in self._api_keys:
            return self._api_keys[provider]
        variable = PROVIDER_KEY_VARIABLES[provider]
        key = os.environ.get(variable)
        if not key:
            raise ModelsClientRefused(
                "api_key",
                f"no key for {provider}: pass api_keys={{'{provider}': ...}} to models_clients, or set {variable}",
                provider=provider,
            )
        return key

    def __call__(
        self, role: CompletionRole, model: str | None, *, temperature: float | None = None
    ) -> BoundCompletionClient:
        """A client for ``role`` bound to ``model``, built on first use and kept.

        Args:
            role: The role the client serves.
            model: The model, or ``None`` for ``defaults[role]``.
            temperature: The temperature to build with, or ``None`` for the provider's default.

        Returns:
            A lease on the factory's client; closing it releases nothing.

        Raises:
            ModelsClientRefused: No model was named and the role has no default; no key for the model's provider is
                in ``api_keys`` or the environment; or a setting the client cannot carry.
        """
        chosen = model if model is not None else self._defaults.get(role)
        if chosen is None:
            raise ModelsClientRefused(
                "model", f"the engine named no model for {role!r} and defaults has none for it; add defaults[{role!r}]"
            )
        settings = self._role_settings.get(role, {})
        effort: KnownReasoningEffort | None = settings.get("reasoning_effort")
        budget: int | None = settings.get("reasoning_max_tokens")
        key: _ClientKey = (role, chosen, temperature, effort, budget)
        client = self._clients.get(key)
        if client is None:
            provider = resolve_provider(chosen, settings.get("provider"))
            client = models_client(
                chosen,
                api_key=self._key_for(provider),
                temperature=temperature,
                prices=self._catalogue,
                **settings,
            )
            self._clients[key] = client
        return _Lease(client)

    async def aclose_all(self) -> None:
        """Close every client the factory built (:meth:`ModelsCompletionClient.aclose`), and forget them.

        A later request builds afresh. Call it once, when the host shuts down.
        """
        clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            await client.aclose()


def models_clients(
    *,
    api_keys: Mapping[str, str] | None = None,
    defaults: Mapping[CompletionRole, str],
    prices: HttpClient | None = None,
    **per_role_kwargs: Mapping[str, Any],  # Any: models_client's keywords, per role
) -> ModelsClients:
    """The host-profile factory: every role the engine calls, on models 3tears-models builds.

    The three lines a host writes::

        clients = models_clients(defaults={"judge": "claude-sonnet-4-6", "analysis": "claude-sonnet-4-6"})
        host = EvalHost(..., clients=clients, failure_describer=ModelsFailureDescriber())
        await clients.aclose_all()  # at shutdown

    Keys come from ``api_keys`` (provider -> key) or, for a provider it does not name, the provider SDK's own
    environment variable (:data:`~threetears.evals.providers.PROVIDER_KEY_VARIABLES`). A missing key is refused
    by name the first time a client for that provider is asked for, never here: a host whose engine work never
    calls a provider needs no key for it.

    Args:
        api_keys: Provider -> key, or ``None`` to read every key from the environment.
        defaults: Role -> the model a request that names none resolves to (the engine asks for a role's default
            by passing ``model=None``).
        prices: The httpx-like client every role's catalogue reads go through, or ``None`` for the adapter to
            open its own per fetch (see :func:`~threetears.evals.providers.models_client`).
        **per_role_kwargs: Role -> keywords for :func:`~threetears.evals.providers.models_client` for every client
            of that role: ``judge={"reasoning_max_tokens": 8192, "max_tokens": 16384}``,
            ``simulator={"reasoning_effort": "minimal", "provider": "openrouter"}``. The temperature, the key and
            the prices are the engine's and the factory's to supply, and are refused here.

    Returns:
        The factory.

    Raises:
        ModelsClientRefused: A role that is not one of the engine's, or a role keyword the factory supplies.
    """
    return ModelsClients(api_keys=api_keys, defaults=defaults, prices=prices, role_settings=per_role_kwargs)
