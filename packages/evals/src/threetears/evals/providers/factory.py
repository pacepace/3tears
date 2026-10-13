"""The host-profile factory: :func:`models_clients`, the engine's role factory over :func:`models_client`.

What a host that brings no client of its own hands :class:`~threetears.evals.kernel.host.EvalHost` as
``clients``. The engine asks it for a client per role, model and temperature
(:class:`~threetears.evals.kernel.host.CompletionClients`), uses it for one unit of work, and closes what it was
handed. Building a chat model per unit of work would rebuild an SDK client each time, so the factory keeps one
:class:`~threetears.evals.providers.ModelsCompletionClient` per role, model, temperature and reasoning setting for
its own life and hands the engine a lease on it, whose ``aclose`` releases nothing; :meth:`ModelsClients.aclose_all`
closes what the factory built, once, when the host shuts down.

**At boot, the host awaits** ``await clients.load_prices()``: it looks up the price catalogue's rates for every
role's default model, in the one price book every client of the factory reads, so each client's synchronous
``price_ceiling`` can answer before the engine makes an out-of-run call (an analysis, a case generation, a second
judge or a rejudge) under an enforced ``out_of_run_cap_usd``. Without it a ceiling is ``None`` until a priced call
loaded the rates (never, on OpenRouter), and the engine refuses every such call. A model the engine names beyond
``defaults`` is loaded the same way through its client: ``await clients(role, model).load_prices()``.

**The judge is built as the run asks it.** A run launched with a judge records
:data:`~threetears.evals.run.JUDGE_REQUEST_SETTINGS` as its ``judge_asked_settings``, so a ``judge`` role given no
request settings of its own (no ``max_tokens``, ``reasoning_max_tokens`` or ``reasoning_effort``) is built from
those values: its output cap and its reasoning effort word, which every provider the adapter supports takes. Their
``strict_output`` is the strict structured directive the judge already sends on every call. A host that passes the
judge's settings explicitly asks for those instead, and the run's asked settings do not describe them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Final, Self, get_args

from threetears.evals.kernel.host import CompletionRole
from threetears.evals.run import JUDGE_REQUEST_SETTINGS
from threetears.evals.schema import BoundCompletionClient, KnownReasoningEffort
from threetears.models.price_lookup import HttpClient

from threetears.evals.providers.adapter import (
    PROVIDER_KEY_VARIABLES,
    ModelsClientRefused,
    ModelsCompletion,
    ModelsCompletionClient,
    PriceBook,
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

#: The role keywords that state a client's request settings. A ``judge`` role naming none of them is built from
#: :data:`~threetears.evals.run.JUDGE_REQUEST_SETTINGS`.
_REQUEST_KEYWORDS: Final = ("max_tokens", "reasoning_max_tokens", "reasoning_effort")


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

    async def load_prices(self) -> str | None:
        """Load the factory's client's rates (:meth:`ModelsCompletionClient.load_prices`), into the factory's book."""
        return await self._client.load_prices()

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
        self._book = PriceBook(prices)
        self._clients: dict[_ClientKey, ModelsCompletionClient] = {}

    def _settings_for(self, role: CompletionRole) -> dict[str, Any]:  # Any: models_client's keywords
        """The keywords ``role``'s client is built with: the role's own, and the judge's defaults.

        Args:
            role: The role.

        Returns:
            The role's keywords; for a ``judge`` that states no request settings of its own, with the output cap and
            the reasoning parameter of :data:`~threetears.evals.run.JUDGE_REQUEST_SETTINGS` added (its effort word,
            or its budget when it asks by one).
        """
        settings = self._role_settings.get(role, {})
        if role != "judge" or any(keyword in settings for keyword in _REQUEST_KEYWORDS):
            return settings
        asked = {
            "max_tokens": JUDGE_REQUEST_SETTINGS.max_tokens,
            "reasoning_effort": JUDGE_REQUEST_SETTINGS.reasoning_effort,
            "reasoning_max_tokens": JUDGE_REQUEST_SETTINGS.reasoning_max_tokens,
        }
        return {**settings, **{keyword: value for keyword, value in asked.items() if value is not None}}

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
        settings = self._settings_for(role)
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
                prices=self._book,
                **settings,
            )
            self._clients[key] = client
        return _Lease(client)

    async def load_prices(self) -> dict[str, str]:
        """Look up the rates of every role's default model, so every client's ``price_ceiling`` can answer: the boot step.

        Await it once, at boot, before the engine makes an out-of-run call (an analysis, a case generation, a second
        judge or a rejudge) under an enforced ``out_of_run_cap_usd``: the ceiling is synchronous and answers only
        from loaded rates, and the engine refuses a call it cannot price. The rates go into the price book every
        client of the factory reads, so a client built later for one of these models is priced at once. A model
        the engine names beyond ``defaults`` is loaded through its own client:
        ``await clients(role, model).load_prices()``. Awaiting it again retries what failed to fetch.

        On OpenRouter the rates serve the ceiling alone (its catalogue prices OpenRouter ids); a call's stored cost
        stays OpenRouter's own reported figure.

        Returns:
            Model id -> why it is unpriced, for each default model the catalogue could not price (each also logged
            at WARNING); empty when every one is loaded.

        Raises:
            ModelsClientRefused: A default model the registry cannot place and whose role names no ``provider``.
            Exception: What a host's registered price source, or a host ``prices=`` client, raises beyond a failed
                fetch: a defect there surfaces at boot.
        """
        unpriced: dict[str, str] = {}
        for role, model in sorted(self._defaults.items()):
            provider = resolve_provider(model, self._role_settings.get(role, {}).get("provider"))
            if (reason := await self._book.load(provider, model)) is not None:
                unpriced[model] = reason
        return unpriced

    async def aclose_all(self) -> None:
        """Close every client the factory built (:meth:`ModelsCompletionClient.aclose`), and forget them.

        Every client is closed even when one fails to: the failures are collected and raised together once the rest
        are closed. A later request builds afresh. Call it once, when the host shuts down.

        Raises:
            ExceptionGroup: One or more clients failed to close; each failure is in the group.
        """
        clients, self._clients = list(self._clients.values()), {}
        failures: list[Exception] = []
        for client in clients:
            try:
                await client.aclose()
            except Exception as failure:  # collected and raised below, once every other client is closed
                failures.append(failure)
        if failures:
            raise ExceptionGroup(f"{len(failures)} of {len(clients)} clients failed to close", failures)


def models_clients(
    *,
    api_keys: Mapping[str, str] | None = None,
    defaults: Mapping[CompletionRole, str],
    prices: HttpClient | None = None,
    **per_role_kwargs: Mapping[str, Any],  # Any: models_client's keywords, per role
) -> ModelsClients:
    """The host-profile factory: every role the engine calls, on models 3tears-models builds.

    The lines a host writes::

        clients = models_clients(defaults={"judge": "claude-sonnet-4-6", "analysis": "claude-sonnet-4-6"})
        unpriced = await clients.load_prices()  # at boot: model -> why, for each default model left unpriced
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
            of that role: ``simulator={"reasoning_effort": "low", "provider": "openrouter"}``. The temperature,
            the key and the prices are the engine's and the factory's to supply, and are refused here. A ``judge``
            that names no ``max_tokens``, ``reasoning_max_tokens`` or ``reasoning_effort`` is built from
            :data:`~threetears.evals.run.JUDGE_REQUEST_SETTINGS` (``max_tokens`` and the reasoning effort word),
            which is what a judged run records it asked.

    Returns:
        The factory.

    Raises:
        ModelsClientRefused: A role that is not one of the engine's, or a role keyword the factory supplies.
    """
    return ModelsClients(api_keys=api_keys, defaults=defaults, prices=prices, role_settings=per_role_kwargs)
