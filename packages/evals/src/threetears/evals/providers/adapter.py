"""One 3tears-models chat model as the evals completion client, and the host-profile factory over it.

:func:`models_client` builds a chat model through :func:`threetears.models.create_chat_model` and wraps it in
:class:`ModelsCompletionClient`, which implements the completion port
(:class:`~threetears.evals.schema.CompletionClient`, and :class:`~threetears.evals.schema.BoundCompletionClient`
since it names its model and prices a call before it is made). :func:`models_clients` is the factory a host hands
the engine as :attr:`~threetears.evals.kernel.host.EvalHost.clients`: one client per role, model, temperature and
reasoning setting, built on first use and kept for the factory's life.

What the adapter reads off a reply, and where each member comes from per provider, is the root's docstring
(:mod:`threetears.evals.providers`). This module is the mechanism: what is built once per client, what is built
per call, and what is refused before anything is sent.

**Built once, per client.** The chat model, with retries off (``max_retries=0``, and on OpenRouter ``retries=None``
on every call, since that SDK otherwise falls back to its own hour-long backoff), streaming off
(``disable_streaming=True``, so a host's ``astream_events`` tap cannot switch the call onto the streamed path,
whose metadata differs) and a circuit breaker of the client's own whose threshold is unreachable, so an eval run
neither trips nor is tripped by the process-wide breaker ``create_chat_model`` would otherwise wire for the
provider and key. The temperature, the output cap and the reasoning setting are built into the instance and the
temperature is read back from it, because LangChain drops a temperature some models refuse at construction and
what the engine records is what was sent.

**Built per call.** The messages and the bind that carries a ``response_format`` (and, on Anthropic, the effort in
the same ``output_config``). Nothing about a call is written to the client, so concurrent calls on one instance
each send exactly their own prompt pair, as the port requires.

**Loaded once, at boot: the prices.** A client's rates come from models' price catalogue, which is read over the
network, and :meth:`ModelsCompletionClient.price_ceiling` is synchronous (the port's signature), so it can only
answer from rates already loaded. A host awaits ``await client.load_prices()`` once after building a client (the
factory's ``await clients.load_prices()`` does it for every role's default model), before the engine makes an
out-of-run call under an enforced ``out_of_run_cap_usd``: until then the ceiling is ``None``, and the engine refuses
such a call rather than make it, since unknown is not $0. On Anthropic and OpenAI the first priced call also loads
the rates; on OpenRouter, whose calls are priced by OpenRouter's own reported figure, only ``load_prices`` does,
from the same catalogue (OpenRouter's models endpoint prices OpenRouter ids), for the ceiling alone.

**A pricing failure never costs the answer.** Pricing runs after the call. A catalogue that cannot be fetched or
read is logged at WARNING and not kept, so the next call (or the next ``load_prices``) tries again; a model the
catalogue does not price is logged once and kept, since asking again changes nothing. Any other exception while
pricing (a bug in a host's registered price source, a host ``prices=`` client raising something other than a
transport error) is logged at ERROR with its traceback, and the paid completion is returned unpriced, its
``pricing_failure`` naming the exception: the call was made and paid for, and the caller is owed its answer.

**Refused, by name, before anything is sent.** A provider the adapter does not support, a model id models'
registry cannot place, the Claude CLI route, a reasoning word or budget the provider cannot carry, and a provider
keyword that would bypass what the adapter guarantees (retries, breaker, streaming, reasoning). Each is a
:class:`ModelsClientRefused` naming the setting and what to do instead. Nothing is dropped and nothing is sent as
its nearest neighbour, because the host records the request settings it asked for
(:class:`~threetears.evals.schema.ClientRequestSettings`), and a request quietly sent otherwise makes that record
false.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final, Literal, Self, get_args

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableBinding

from threetears.evals.schema import KnownReasoningEffort, StopReason
from threetears.evals.schema.logs import get_logger
from threetears.models import CircuitBreaker, PriceLookup, create_chat_model, get_capabilities, lookup_price
from threetears.models.price_lookup import OPENROUTER_MODELS_URL, HttpClient
from threetears.models.providers.claude_cli import OAUTH_TOKEN_PREFIX, is_subscription_token
from threetears.models.providers.structured_output import structured_output_kwargs

__all__ = [
    "PROVIDER_KEY_VARIABLES",
    "SUPPORTED_PROVIDERS",
    "ModelsClientRefused",
    "ModelsCompletion",
    "ModelsCompletionClient",
    "PriceBook",
    "SupportedProvider",
    "models_client",
    "resolve_provider",
]

log = get_logger(__name__)

#: The providers the adapter builds a client for, as models' registry names them. The Claude CLI route (the
#: ``anthropic`` provider given a subscription token) is refused even though models routes it: see
#: :func:`models_client`.
SupportedProvider = Literal["anthropic", "openai", "openrouter"]

#: Every :data:`SupportedProvider`, in the order the docs list them.
SUPPORTED_PROVIDERS: Final[tuple[SupportedProvider, ...]] = get_args(SupportedProvider)

#: The environment variable each provider's own SDK reads its key from, which :func:`models_clients` reads when
#: its ``api_keys`` mapping names no key for a provider. ``create_chat_model`` takes the key as an argument and
#: reads no environment of its own, so these are the SDKs' names (``ChatAnthropic``, ``ChatOpenAI`` and
#: ``ChatOpenRouter`` read the same ones), not a convention of this package.
PROVIDER_KEY_VARIABLES: Final[Mapping[SupportedProvider, str]] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

#: The reasoning words each provider takes, out of the engine's seven
#: (:data:`~threetears.evals.schema.REASONING_EFFORTS`). Anthropic's ``effort`` has five, and ``none`` is sent as
#: thinking disabled; OpenAI's ``reasoning_effort`` has no ``max``; OpenRouter's SDK types all seven.
_EFFORT_WORDS: Final[Mapping[SupportedProvider, frozenset[str]]] = {
    "anthropic": frozenset({"max", "xhigh", "high", "medium", "low", "none"}),
    "openai": frozenset({"xhigh", "high", "medium", "low", "minimal", "none"}),
    "openrouter": frozenset(get_args(KnownReasoningEffort)),
}

#: What to ask for instead of a word a provider does not take, by provider and word.
_EFFORT_INSTEAD: Final[Mapping[tuple[SupportedProvider, str], str]] = {
    ("anthropic", "minimal"): (
        "ask for 'low', Anthropic's lowest effort, or 'none' for no reasoning; 'minimal' is not sent as 'low', which "
        "would record a request that was never made"
    ),
    ("openai", "max"): (
        "ask for 'xhigh', OpenAI's highest reasoning_effort; 'max' is not sent as 'xhigh', which would record a "
        "request that was never made"
    ),
}

#: Why a reasoning token budget cannot be sent, by provider; Anthropic takes one.
_NO_BUDGET: Final[Mapping[SupportedProvider, str]] = {
    "openai": "OpenAI takes a reasoning_effort word and no reasoning token budget; pass reasoning_effort instead",
    "openrouter": (
        "the OpenRouter SDK's typed request declares reasoning.effort and strips reasoning.max_tokens without an "
        "error, so the budget would never be sent; pass reasoning_effort, or keep a client of your own until "
        "3tears-models declares the field"
    ),
}

#: Anthropic's floor on a thinking budget, in tokens.
_ANTHROPIC_MIN_THINKING_BUDGET: Final = 1024

#: Provider keywords the adapter sets itself or maps from its own parameters. Passing one through
#: ``provider_kwargs`` would bypass a guarantee (retries off, the private breaker, streaming off) or the reasoning
#: mapping and its refusals, so each is refused, naming the parameter to use instead.
_RESERVED_KWARGS: Final[Mapping[str, str]] = {
    "max_retries": "the adapter sends every request once (max_retries=0); nothing is retried under the engine",
    "breaker": "the adapter builds a breaker of its own that never opens; compose a breaker outside the engine",
    "disable_streaming": "the adapter always builds with streaming off",
    "streaming": "the adapter always builds with streaming off",
    "thinking": "pass reasoning_max_tokens (a budget) or reasoning_effort='none'",
    "effort": "pass reasoning_effort",
    "reasoning": "pass reasoning_effort",
    "output_config": "pass reasoning_effort; a response_format is the engine's to send per call",
    "response_format": "a response_format is the engine's to send per call",
}

#: The tokens the price ceiling allows for a prompt's framing beyond its text: role markers, the provider's own
#: instructions for a structured reply, the schema's re-encoding. A generous constant rather than a measurement,
#: because the ceiling must not be below the real call.
_PROMPT_FRAMING_ALLOWANCE_TOKENS: Final = 2048

#: How long the adapter's own catalogue fetch may take, in seconds, when the host injected no price client.
_CATALOGUE_TIMEOUT_S: Final = 20.0

#: The prefix of OpenRouter's own routers (``openrouter/auto``), which it resolves to a model on its side, as it
#: does a ``~`` pointer and a ``-latest`` alias.
_OPENROUTER_ROUTERS_PREFIX: Final = "openrouter/"

#: Anthropic's stop reasons, in the engine's words. ``tool_use`` is a finish: the adapter binds no tools.
#: ``model_context_window_exceeded`` is the context window filling, not the output cap: it is ``truncated``, never
#: ``max_tokens``, whose remedy (raise the cap) makes an overflowing context worse. ``pause_turn`` is a server-tool
#: pause: output exists and stopped early for a cause that is neither the cap nor a filter, so it is ``truncated``,
#: never ``error`` (nothing failed) and never ``end_turn`` (the turn is not over). The raw value rides beside either
#: (:attr:`ModelsCompletion.provider_stop_reason`).
_ANTHROPIC_STOPS: Final[Mapping[str, StopReason]] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "end_turn",
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "truncated",
    "refusal": "content_filter",
    "pause_turn": "truncated",
}

#: OpenAI-shaped finish reasons (OpenAI and OpenRouter), in the engine's words. OpenRouter's own ``error`` finish
#: reason is the provider reporting an error in place of a completion, so it stays ``error``.
_OPENAI_STOPS: Final[Mapping[str, StopReason]] = {
    "stop": "end_turn",
    "tool_calls": "end_turn",
    "function_call": "end_turn",
    "length": "max_tokens",
    "content_filter": "content_filter",
    "error": "error",
}

#: Why an OpenAI Responses-API reply stopped short, in the engine's words: its ``incomplete_details.reason``. A reply
#: whose ``status`` is ``completed`` is ``end_turn``; any other status, or a reason not named here, is ``truncated``.
_RESPONSES_INCOMPLETE: Final[Mapping[str, StopReason]] = {
    "max_output_tokens": "max_tokens",
    "content_filter": "content_filter",
}

#: Model id prefixes ``ChatOpenAI`` always sends to OpenAI's Responses API, whatever it is asked: langchain-openai's
#: own list (``chat_models/base.py``), copied because the original is private there, as is the ``codex`` rule beside
#: it. ``tests/test_models_adapter.py`` holds the copy to where LangChain actually sends each id, so an upgrade that
#: moves an id fails there by name.
_RESPONSES_ONLY_PREFIXES: Final = ("gpt-5-pro", "gpt-5.2-pro", "gpt-5.4-pro", "gpt-5.5-pro")

#: ``ChatOpenAI`` fields any non-``None`` value of which routes every call to the Responses API.
_RESPONSES_FIELDS: Final = ("context_management", "include", "reasoning", "truncation")

#: Request keywords that route a ``ChatOpenAI`` call to the Responses API when its payload carries one, as a
#: provider's ``model_kwargs`` would.
_RESPONSES_PAYLOAD_KEYS: Final = frozenset(
    {"context_management", "include", "previous_response_id", "reasoning", "text", "truncation"}
)

#: What ``price_source`` gains when a catalogue-priced call read or wrote the prompt cache: the catalogue carries no
#: cache rates, so those tokens were priced at the plain input rate, and the stored cost is approximate, in either
#: direction (a provider bills a cache read below the input rate and an Anthropic cache write above it).
_CACHE_APPROXIMATE: Final = ";cache-approximate"

#: What an absent or unknown finish reason reads as: cut short, cause unnamed, never finished. A reply with no
#: finish reason (a stream that ended without one) or with one this build does not know (a future provider value)
#: still carried output, so it is not ``error``, which says the call failed; and a truncation read as ``end_turn``
#: is the defect the port exists to prevent. The raw value rides beside it, as
#: :attr:`ModelsCompletion.provider_stop_reason`.
_UNNAMED_STOP: Final[StopReason] = "truncated"


class ModelsClientRefused(ValueError):
    """A setting the adapter cannot carry to the provider, refused before anything is sent.

    Raised at construction for a provider, model, reasoning word or budget, or provider keyword the adapter cannot
    send as asked; by :meth:`ModelsCompletionClient.generate` for a ``response_format`` it cannot translate; and
    by :class:`ModelsClients` for a role with no model or a provider with no key, at the first call that needs
    one. A ``ValueError``, so a caller already catching one keeps catching it. The message names the setting, why
    it cannot be sent, and what to pass instead.

    Attributes:
        setting: The parameter or keyword refused, as the caller wrote it.
        provider: The provider the client was being built for, or ``None`` when none was resolved yet.
    """

    def __init__(self, setting: str, why: str, *, provider: str | None = None) -> None:
        """Name the refused setting and why.

        Args:
            setting: The parameter or keyword refused.
            why: Why it cannot be sent and what to pass instead, as a clause.
            provider: The provider the client was being built for, when one was resolved.
        """
        self.setting = setting
        self.provider = provider
        where = f" on {provider}" if provider else ""
        super().__init__(f"{setting} refused{where}: {why}")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelsCompletion:
    """One completion as the adapter read it off the chat model's reply: the port's ten members and three of its own.

    Implements :class:`~threetears.evals.schema.CompletionResult` structurally, frozen as the protocol's read-only
    members ask. Every count and the cost are ``None`` where the provider reported nothing, never ``0``. Which
    members each provider populates is pinned by ``tests/test_models_adapter_dimensions.py``.

    Attributes:
        content: The reply's text (``AIMessage.text``): the text blocks only, never a thinking block, and under a
            schema the raw JSON text the model wrote, never a parsed object.
        input_tokens: Prompt tokens from ``usage_metadata``, cache reads and writes included on Anthropic.
        output_tokens: Completion tokens from ``usage_metadata``, reasoning included.
        reasoning_tokens: The reasoning share of ``output_tokens``, or ``None`` when the provider reported no split.
        cost_usd: What the call cost, or ``None`` when nothing priced it (``pricing_failure`` says why).
        price_source: Where ``cost_usd`` came from: ``openrouter:reported`` for OpenRouter's own figure, or
            ``<source>:<matched catalogue id>`` for a rate from ``threetears.models.price_lookup``, which ends in
            ``;cache-approximate`` when the reply read or wrote the prompt cache: the catalogue has no cache rates,
            so those tokens were priced at the plain input rate and the cost is approximate, in either direction.
        model: The model id the client was built for, which is what spend is attributed to.
        served_model: The model the provider's reply named, or ``None`` when it named none.
        temperature: The temperature the request carried, or ``None`` when it carried none.
        stop_reason: The provider's stop reason in the engine's words; an absent or unknown one, and Anthropic's
            ``pause_turn`` and ``model_context_window_exceeded``, is ``truncated``, and only a provider-reported
            error (OpenRouter's ``error``) is ``error``.
        provider_stop_reason: The stop reason as the provider named it (Anthropic's ``stop_reason``; OpenAI's
            ``finish_reason``, or on its Responses API the ``incomplete_details.reason``, else the ``status``;
            OpenRouter's ``native_finish_reason`` when LangChain carries it, else its ``finish_reason``), or ``None``
            when it named none. It is what to look at when ``stop_reason`` is ``truncated``: the engine's word says
            the cause went unnamed, this says what, if anything, was sent. The adapter logs it at WARNING whenever a
            reply reads as ``truncated``, since the engine stores only its own word.
        served_provider: The upstream that served an OpenRouter call, as OpenRouter's reply names it; ``None`` on
            a provider called directly, whose reply names no upstream, and on an OpenRouter reply that named none.
        pricing_failure: Why ``cost_usd`` is ``None``, in a sentence; ``None`` when the call was priced.
    """

    content: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None
    price_source: str | None
    model: str
    served_model: str | None
    temperature: float | None
    stop_reason: StopReason
    provider_stop_reason: str | None
    served_provider: str | None
    pricing_failure: str | None


@dataclass(frozen=True, slots=True)
class _Unpriced:
    """A pricing outcome that names no rate: why the client's calls go unpriced."""

    reason: str


class _CatalogueUnreadable(Exception):
    """OpenRouter's catalogue answered, and its body is not JSON: the one parse the adapter reads as a bad catalogue.

    Raised by :meth:`PriceBook.get` rather than left to ``price_lookup``'s own ``response.json()``, so a ``ValueError``
    from anywhere else (a host's registered price source) is not mistaken for a malformed catalogue.
    """


class PriceBook:
    """The price catalogue a client reads, or every client of one :class:`~threetears.evals.providers.ModelsClients`.

    Two caches. ``threetears.models.price_lookup.lookup_price`` GETs the whole catalogue on every call and caches
    nothing, so this sits between it and the HTTP client and keeps each URL's response once it passed
    ``raise_for_status``. And it keeps each model's looked-up rates by provider and model id, which is what lets
    :meth:`ModelsCompletionClient.price_ceiling`, which is synchronous, answer from rates an earlier
    :meth:`ModelsCompletionClient.load_prices` loaded, for every client of the model the book serves.

    What is kept is what asking again would not change: a response that was read, rates that were found, and a
    catalogue that prices no row for the model. A fetch that failed, or a catalogue body that is not JSON, is not
    kept: the next call or load tries again. Every outcome that leaves a model unpriced is logged at WARNING.
    """

    def __init__(self, client: HttpClient | None) -> None:
        """Wrap the host's client, or none, in which case each fetch opens and closes a client of its own.

        Args:
            client: The host's httpx-like client, or ``None``.
        """
        self._client = client
        self._responses: dict[str, Any] = {}
        self._fetch_lock = asyncio.Lock()
        self._rates: dict[tuple[str, str], PriceLookup | _Unpriced] = {}
        self._rates_lock = asyncio.Lock()

    async def get(self, url: str, /) -> Any:  # Any: models' HttpClient protocol types the response as Any
        """The response for ``url``, fetched on the first read and kept: the ``HttpClient`` ``lookup_price`` reads.

        Args:
            url: The URL a price source reads.

        Returns:
            The response, already checked with ``raise_for_status`` (and, for OpenRouter's catalogue, parsed).

        Raises:
            _CatalogueUnreadable: OpenRouter's catalogue answered with a body that is not JSON.
        """
        async with self._fetch_lock:
            if url not in self._responses:
                response = await self._fetch(url)
                response.raise_for_status()
                if url == OPENROUTER_MODELS_URL:
                    try:
                        response.json()
                    except ValueError as unparsed:
                        raise _CatalogueUnreadable(f"{type(unparsed).__name__}: {unparsed}") from unparsed
                self._responses[url] = response
            return self._responses[url]

    async def _fetch(self, url: str) -> Any:  # Any: as above
        if self._client is not None:
            return await self._client.get(url)
        async with httpx.AsyncClient(timeout=_CATALOGUE_TIMEOUT_S, follow_redirects=True) as owned:
            return await owned.get(url)

    def rates(self, provider: str, model: str) -> PriceLookup | _Unpriced | None:
        """The rates already looked up for ``model`` on ``provider``, why it is unpriced, or ``None`` when not looked up.

        Args:
            provider: The provider, as models' registry names it.
            model: The model id.

        Returns:
            What the book holds for the model, without touching the network.
        """
        return self._rates.get((provider, model))

    async def look_up(self, provider: str, model: str) -> PriceLookup | _Unpriced:
        """The model's rates, looked up on the first ask and kept, or why it is unpriced.

        Args:
            provider: The provider, as models' registry names it.
            model: The model id.

        Returns:
            The rates, both of them present, or why there are none.

        Raises:
            Exception: Whatever a host's registered price source, or a host ``prices=`` client, raises beyond a
                failed fetch (``httpx.HTTPError``, ``OSError``) and an unreadable catalogue: a defect there is not a
                price, and is not kept.
        """
        async with self._rates_lock:
            known = self._rates.get((provider, model))
            if known is not None:
                return known
            try:
                found = await lookup_price(self, provider=provider, api_name=model)
            except (httpx.HTTPError, OSError) as exc:
                return self._unpriced(
                    provider, model, f"the price catalogue could not be fetched ({type(exc).__name__}: {exc})"
                )
            except _CatalogueUnreadable as exc:
                return self._unpriced(provider, model, f"the price catalogue could not be read ({exc})")
            if found.matched_id is None:
                outcome: PriceLookup | _Unpriced = self._unpriced(
                    provider, model, f"no price catalogue row matches {provider} model {model!r}"
                )
            elif found.cost_per_1m_input is None or found.cost_per_1m_output is None:
                outcome = self._unpriced(
                    provider, model, f"the price catalogue row {found.matched_id!r} carries no input or output rate"
                )
            else:
                outcome = found
            self._rates[(provider, model)] = outcome
            return outcome

    async def load(self, provider: str, model: str) -> str | None:
        """Look the model's rates up (:meth:`look_up`), and say why it is unpriced, or ``None`` when it is priced.

        Args:
            provider: The provider, as models' registry names it.
            model: The model id.

        Returns:
            ``None`` when the rates are loaded, else why the model is unpriced (already logged at WARNING).
        """
        rates = await self.look_up(provider, model)
        return rates.reason if isinstance(rates, _Unpriced) else None

    @staticmethod
    def _unpriced(provider: str, model: str, reason: str) -> _Unpriced:
        log.warning("eval.providers unpriced provider=%s model=%s: %s", provider, model, reason)
        return _Unpriced(reason)


def resolve_provider(model_id: str, provider: str | None) -> SupportedProvider:
    """The provider a model id is built on: ``provider`` when given, else the one models' registry names.

    Args:
        model_id: The model id as the host names it.
        provider: An explicit provider, which overrides the registry.

    Returns:
        The provider.

    Raises:
        ModelsClientRefused: The registry cannot place ``model_id`` and no ``provider`` was given, or the provider
            is not one the adapter supports.
    """
    if provider is None:
        capabilities = get_capabilities(model_id)
        if capabilities is None or not capabilities.provider_name:
            raise ModelsClientRefused(
                "model_id",
                f"3tears-models' capability registry cannot place {model_id!r}; pass provider= (one of "
                f"{', '.join(SUPPORTED_PROVIDERS)}), or register the model's capabilities "
                "(threetears.models.register_capabilities) first",
            )
        provider = capabilities.provider_name
    for supported in SUPPORTED_PROVIDERS:
        if provider == supported:
            return supported
    raise ModelsClientRefused(
        "provider",
        f"{provider!r} is not a provider this adapter builds a client for ({', '.join(SUPPORTED_PROVIDERS)}); bring "
        "a completion client of your own for it",
    )


def _reasoning_build_kwargs(
    provider: SupportedProvider,
    effort: KnownReasoningEffort | None,
    budget: int | None,
    max_tokens: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The build keywords and the per-call ``output_config`` that carry a reasoning setting, or a refusal.

    Args:
        provider: The provider.
        effort: The reasoning word asked for, or ``None``.
        budget: The reasoning token budget asked for, or ``None``.
        max_tokens: The output cap asked for, or ``None`` for the provider's default.

    Returns:
        The keywords for ``create_chat_model``, and the ``output_config`` entries each call merges into its own.

    Raises:
        ModelsClientRefused: The setting cannot be carried to ``provider`` as asked.
    """
    if effort is not None and budget is not None:
        raise ModelsClientRefused(
            "reasoning_effort",
            "reasoning is asked for one way, by a level or by a token budget, never both "
            "(ClientRequestSettings refuses the pair too); pass reasoning_effort or reasoning_max_tokens",
            provider=provider,
        )
    if effort is not None:
        if effort not in _EFFORT_WORDS[provider]:
            instead = _EFFORT_INSTEAD.get((provider, effort), f"ask for one of {sorted(_EFFORT_WORDS[provider])}")
            raise ModelsClientRefused(
                "reasoning_effort", f"{effort!r} is not a word {provider} takes; {instead}", provider=provider
            )
        if provider == "anthropic":
            if effort == "none":
                return {"thinking": {"type": "disabled"}}, {}
            return {}, {"effort": effort}
        if provider == "openai":
            return {"reasoning_effort": effort}, {}
        return {"reasoning": {"effort": effort}}, {}
    if budget is not None:
        if provider != "anthropic":
            raise ModelsClientRefused("reasoning_max_tokens", _NO_BUDGET[provider], provider=provider)
        if budget < _ANTHROPIC_MIN_THINKING_BUDGET:
            raise ModelsClientRefused(
                "reasoning_max_tokens",
                f"Anthropic refuses a thinking budget below {_ANTHROPIC_MIN_THINKING_BUDGET} tokens; ask for at "
                f"least {_ANTHROPIC_MIN_THINKING_BUDGET}",
                provider=provider,
            )
        if max_tokens is not None and budget >= max_tokens:
            raise ModelsClientRefused(
                "reasoning_max_tokens",
                f"Anthropic refuses a thinking budget ({budget}) that is not below max_tokens ({max_tokens}); raise "
                "max_tokens or lower the budget",
                provider=provider,
            )
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}, {}
    return {}, {}


def _int_or_none(value: object) -> int | None:
    """``value`` when it is an int (and not a bool), else ``None``: a count the provider did not report."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _mapping(value: object) -> Mapping[str, Any]:  # Any: LangChain's metadata dicts are dict[str, Any]
    """``value`` when it is a mapping, else an empty one, so a nested read of absent metadata is ``None``."""
    return value if isinstance(value, Mapping) else {}


def _named(value: object) -> str | None:
    """``value`` when it is a non-empty string, else ``None``: a stop reason the provider named, or none."""
    return value if isinstance(value, str) and value else None


def _routes_to_responses(bound: object) -> bool:
    """Whether ``ChatOpenAI`` sends this model's calls to OpenAI's Responses API rather than Chat Completions.

    LangChain's own rule, read off the built model's public fields: ``use_responses_api`` when it is set; otherwise
    an ``output_version`` of ``responses/v1``, any Responses-only field or ``model_kwargs`` keyword, a previous
    response id, or a model id LangChain always routes there (:data:`_RESPONSES_ONLY_PREFIXES`, and any id
    containing ``codex``).

    Args:
        bound: The built ``ChatOpenAI``.

    Returns:
        Whether every call goes to the Responses API.
    """
    chosen = getattr(bound, "use_responses_api", None)
    if isinstance(chosen, bool):
        return chosen
    if getattr(bound, "output_version", None) == "responses/v1" or getattr(bound, "use_previous_response_id", False):
        return True
    if any(getattr(bound, name, None) is not None for name in _RESPONSES_FIELDS):
        return True
    if _RESPONSES_PAYLOAD_KEYS & set(_mapping(getattr(bound, "model_kwargs", None))):
        return True
    model = getattr(bound, "model_name", None)
    return isinstance(model, str) and (model.startswith(_RESPONSES_ONLY_PREFIXES) or "codex" in model)


def _responses_temperature(bound: object, temperature: float | None) -> float | None:
    """The temperature a Responses-API payload carries: LangChain drops it there for a reasoning ``gpt-5`` model.

    Its payload builder removes the temperature for a model id starting ``gpt-5`` that is not a ``chat`` model,
    unless the reasoning effort is ``none``; construction already dropped any value but 1 for the same models, so
    this is where a ``temperature=1`` goes too.

    Args:
        bound: The built ``ChatOpenAI``.
        temperature: The temperature the built model holds.

    Returns:
        The temperature the request carries, or ``None``.
    """
    model = getattr(bound, "model_name", None)
    effort = getattr(bound, "reasoning_effort", None) or _mapping(getattr(bound, "reasoning", None)).get("effort")
    if isinstance(model, str) and model.startswith("gpt-5") and "chat" not in model and effort != "none":
        return None
    return temperature


class ModelsCompletionClient:
    """The completion port over one chat model 3tears-models built: :func:`models_client` returns one.

    Implements :class:`~threetears.evals.schema.CompletionClient` and
    :class:`~threetears.evals.schema.BoundCompletionClient`. ``generate`` is safe to call concurrently: every call
    builds its own messages and its own bind, and the only state it writes is the price book's (under its locks)
    and :attr:`last_pricing_failure`.

    **Await** :meth:`load_prices` **once after building it**, before the engine makes an out-of-run call on it under
    an enforced cap: :meth:`price_ceiling` answers only from loaded rates.

    Attributes:
        model_name: The model id the client calls, which every completion is attributed to.
        provider: The provider the client calls, as models' registry names it.
    """

    def __init__(
        self,
        *,
        model_name: str,
        provider: SupportedProvider,
        chat_model: RunnableBinding[Any, Any],  # Any: LangChain types a chat model's input and output as Any
        call_output_config: Mapping[str, Any],  # Any: the provider's own output_config values
        prices: HttpClient | None,
    ) -> None:
        """Wrap a built chat model. Build one with :func:`models_client`, which checks every setting first.

        Args:
            model_name: The model id the chat model was built for.
            provider: Its provider.
            chat_model: The binding ``create_chat_model`` returned.
            call_output_config: ``output_config`` entries every call sends (Anthropic's effort), merged with a
                schema's format into one ``output_config``, since a second bind replaces the first.
            prices: The httpx-like client a price book of the client's own reads the catalogue through, or
                ``None`` for one that opens a client of its own per fetch; a factory passes its own
                :class:`PriceBook` (itself such a client), which every client it builds then shares.
        """
        self.model_name = model_name
        self.provider: SupportedProvider = provider
        self._chat = chat_model
        self._call_output_config = dict(call_output_config)
        self._book = prices if isinstance(prices, PriceBook) else PriceBook(prices)
        bound = chat_model.bound
        self._responses_api = provider == "openai" and _routes_to_responses(bound)
        temperature = getattr(bound, "temperature", None)
        built = float(temperature) if isinstance(temperature, int | float) else None
        self._temperature = _responses_temperature(bound, built) if self._responses_api else built
        self._output_cap = _int_or_none(getattr(bound, "max_tokens", None))
        self._closed = False
        self.last_pricing_failure: str | None = None
        """Why the most recent completion went unpriced, or ``None`` when it was priced.

        The same sentence as that completion's ``pricing_failure``, kept on the client so a host that reads only
        the engine's records (``cost_usd`` is ``None``) can still ask why. Under concurrent calls it is the last
        one to finish; read a completion's own ``pricing_failure`` when that matters."""

    @property
    def chat_model(self) -> RunnableBinding[Any, Any]:  # Any: as in __init__
        """The LangChain runnable this client calls: ``create_chat_model``'s binding, its model at ``.bound``.

        For a host that wants to see what every request carries (``chat_model.bound.max_retries``,
        ``.temperature``, ``.max_tokens``) or to make calls of its own on the same model. A call made on it
        directly bypasses everything this client maps: the stop reason, the served model, the pricing.
        """
        return self._chat

    @property
    def temperature(self) -> float | None:
        """The temperature every request is sent with, read back from the built model; ``None`` when none is sent.

        Read after construction, never echoed from the argument: LangChain drops a temperature a model refuses at
        build (OpenAI's ``gpt-5`` family), and again from the request on OpenAI's Responses API for a reasoning
        ``gpt-5`` model, and on Anthropic a thinking budget is built with none, since the API refuses one beside
        extended thinking.
        """
        return self._temperature

    async def load_prices(self) -> str | None:
        """Look the model's rates up in the price catalogue, so :meth:`price_ceiling` can answer: the boot step.

        A host awaits it once after building the client, before the engine makes an out-of-run call on it under an
        enforced cap (``out_of_run_cap_usd``); a client a factory built shares the factory's book, so a model loaded
        once is loaded for every client of it. Rates found, and a catalogue that prices no row for the model, are
        kept; a failed fetch is not, so awaiting it again tries again. On OpenRouter the rates serve the ceiling
        alone: a call's stored cost stays OpenRouter's own reported figure.

        Returns:
            ``None`` when the rates are loaded, else why the model is unpriced (also logged at WARNING), in which
            case :meth:`price_ceiling` stays ``None`` and an out-of-run call under an enforced cap is refused.

        Raises:
            Exception: What a host's registered price source, or a host ``prices=`` client, raises beyond a failed
                fetch: a defect there surfaces at boot rather than as an unpriced model.
        """
        return await self._book.load(self.provider, self.model_name)

    def _request_kwargs(self, response_format: dict[str, Any] | None) -> dict[str, Any]:  # Any: provider kwargs
        """The bind keywords for one call: the response format as the provider takes it, and the effort beside it."""
        kwargs: dict[str, Any] = {}
        output_config: dict[str, Any] = dict(self._call_output_config)
        if response_format is not None and self._responses_api:
            raise ModelsClientRefused(
                "response_format",
                f"{self.model_name!r} is sent to OpenAI's Responses API, which takes a reply format as text.format, "
                "and models' structured-output translation writes Chat Completions' response_format, which that API "
                "refuses; use a model ChatOpenAI sends to Chat Completions (and no Responses-only keyword), or a "
                "client of your own",
                provider=self.provider,
            )
        if response_format is not None:
            kind = response_format.get("type")
            if kind == "json_schema":
                spec = response_format.get("json_schema")
                schema = spec.get("schema") if isinstance(spec, dict) else None
                if not isinstance(spec, dict) or not isinstance(schema, dict):
                    raise ModelsClientRefused(
                        "response_format",
                        "a json_schema directive carries its schema at response_format['json_schema']['schema']",
                        provider=self.provider,
                    )
                built = structured_output_kwargs(
                    self.provider,
                    schema,
                    name=str(spec.get("name") or "response"),
                    strict=bool(spec.get("strict", True)),
                )
                if self.provider == "anthropic":
                    output_config.update(_mapping(built.pop("output_config", None)))
                kwargs.update(built)
            elif kind == "json_object":
                if self.provider == "openai":
                    # Through extra_body: a top-level response_format diverts ChatOpenAI onto its parse path.
                    kwargs["extra_body"] = {"response_format": {"type": "json_object"}}
                elif self.provider == "openrouter":
                    kwargs["response_format"] = {"type": "json_object"}
                # Anthropic has no JSON mode, and an empty-object schema would force `{}`: nothing is sent, and the
                # engine validates the reply on return as it does for every directive.
            else:
                raise ModelsClientRefused(
                    "response_format",
                    f"{kind!r} is not a directive the adapter translates; send a json_schema or a json_object one",
                    provider=self.provider,
                )
        if output_config:
            kwargs["output_config"] = output_config
        return kwargs

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> ModelsCompletion:
        """Send the prompt pair once and read the reply into the port's members.

        Args:
            system: The system message.
            user: The one human message.
            response_format: ``None``; a ``json_schema`` directive, sent through models' structured-output
                translation for the provider (Anthropic's ``output_config.format``, OpenAI's
                ``extra_body.response_format``, OpenRouter's ``response_format`` with ``require_parameters``, so
                an upstream that cannot honour it is refused rather than answering in prose); or a
                ``json_object`` one, sent as OpenAI and OpenRouter take it and not at all on Anthropic, which has
                no JSON mode. Either way the content is the reply's raw text. Refused on a model ``ChatOpenAI``
                sends to OpenAI's Responses API, which takes neither in that shape.

        Returns:
            The completion. Its price never costs it: a pricing failure after the call returns it unpriced.

        Raises:
            ModelsClientRefused: ``response_format`` is not a directive the adapter translates, or the model is on
                OpenAI's Responses API.
            TypeError: The chat model returned something other than an ``AIMessage``.
        """
        kwargs = self._request_kwargs(response_format)
        runnable = self._chat.bind(**kwargs) if kwargs else self._chat
        reply = await runnable.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
        if not isinstance(reply, AIMessage):
            raise TypeError(f"{self.provider} chat model returned {type(reply).__name__}, not an AIMessage")
        return await self._read(reply)

    async def _read(self, reply: AIMessage) -> ModelsCompletion:
        """The reply's members, per provider."""
        meta = _mapping(reply.response_metadata)
        usage = _mapping(reply.usage_metadata)
        input_tokens = _int_or_none(usage.get("input_tokens"))
        output_tokens = _int_or_none(usage.get("output_tokens"))
        if self.provider == "anthropic":
            # LangChain does not copy Anthropic's thinking-token split into usage_metadata; the response dump it
            # merges into response_metadata carries it (on this non-streamed path).
            reasoning = _int_or_none(
                _mapping(_mapping(meta.get("usage")).get("output_tokens_details")).get("thinking_tokens")
            )
        else:
            reasoning = _int_or_none(_mapping(usage.get("output_token_details")).get("reasoning"))
        cost, source, unpriced = await self._price(meta, usage, input_tokens, output_tokens)
        self.last_pricing_failure = unpriced
        stop_reason, provider_stop_reason = self._stop(reply, meta)
        if stop_reason == "truncated":
            log.warning(
                "eval.providers reply read as truncated provider=%s model=%s provider_stop_reason=%s",
                self.provider,
                self.model_name,
                provider_stop_reason or "none",
            )
        served_provider = meta.get("provider") if self.provider == "openrouter" else None
        return ModelsCompletion(
            content=reply.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning,
            cost_usd=cost,
            price_source=source,
            model=self.model_name,
            served_model=self._served_model(meta),
            temperature=self._temperature,
            stop_reason=stop_reason,
            provider_stop_reason=provider_stop_reason,
            served_provider=served_provider if isinstance(served_provider, str) and served_provider else None,
            pricing_failure=unpriced,
        )

    async def _price(
        self,
        meta: Mapping[str, Any],  # Any: LangChain's metadata
        usage: Mapping[str, Any],  # Any: as above
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> tuple[float | None, str | None, str | None]:
        """The call's cost, where it came from, and why it is unpriced: never an exception, since the call is paid.

        Pricing runs after the provider answered. An exception from it (a host's price source, a host ``prices=``
        client raising something other than a transport error) is logged at ERROR with its traceback and recorded on
        the completion as its ``pricing_failure``; the paid completion is still returned.
        """
        try:
            if self.provider == "openrouter":
                return self._reported_cost(meta)
            return await self._catalogue_cost(usage, input_tokens, output_tokens)
        except Exception as exc:  # the paid answer is returned unpriced; the failure is logged and recorded on it
            log.exception(
                "eval.providers pricing raised after a paid call provider=%s model=%s; the completion is returned "
                "unpriced",
                self.provider,
                self.model_name,
            )
            return None, None, f"pricing the call raised {type(exc).__name__}: {exc}"

    def _served_model(self, meta: Mapping[str, Any]) -> str | None:  # Any: LangChain's metadata
        """The model the reply named. On OpenRouter, LangChain's fallback to the requested alias is not evidence."""
        named = meta.get("model") if self.provider == "anthropic" else meta.get("model_name")
        if not isinstance(named, str) or not named:
            return None
        if self.provider == "openrouter" and named == self.model_name and _is_openrouter_alias(self.model_name):
            # LangChain fills an empty chunk model with the requested id, so an alias echoed back names the
            # pointer, not the model behind it: reporting it would make two models served on one alias compare
            # equal, the defect CompletionResult.served_model warns about.
            return None
        return named

    def _stop(self, reply: AIMessage, meta: Mapping[str, Any]) -> tuple[StopReason, str | None]:  # Any: metadata
        """The stop reason in the engine's words, and as the provider named it (``None`` when it named none).

        A refusal is the content filter; an absent or unknown stop is cut short (:data:`_UNNAMED_STOP`).
        """
        if self.provider == "anthropic":
            raw = _named(meta.get("stop_reason"))
            return (_ANTHROPIC_STOPS.get(raw, _UNNAMED_STOP) if raw else _UNNAMED_STOP), raw
        if self.provider == "openai" and "finish_reason" not in meta and "status" in meta:
            return self._responses_stop(reply, meta)
        finish = _named(meta.get("finish_reason"))
        raw = (_named(meta.get("native_finish_reason")) or finish) if self.provider == "openrouter" else finish
        if reply.additional_kwargs.get("refusal"):
            # A structured-mode refusal arrives beside empty content with finish_reason `stop`.
            return "content_filter", raw
        return (_OPENAI_STOPS.get(finish, _UNNAMED_STOP) if finish else _UNNAMED_STOP), raw

    @staticmethod
    def _responses_stop(reply: AIMessage, meta: Mapping[str, Any]) -> tuple[StopReason, str | None]:  # Any: as above
        """An OpenAI Responses-API reply's stop: its ``status``, and when incomplete the ``incomplete_details``."""
        status = _named(meta.get("status"))
        reason = _named(_mapping(meta.get("incomplete_details")).get("reason"))
        raw = reason or status
        if reply.additional_kwargs.get("refusal"):
            return "content_filter", raw
        if status == "completed":
            return "end_turn", raw
        return (_RESPONSES_INCOMPLETE.get(reason, _UNNAMED_STOP) if reason else _UNNAMED_STOP), raw

    def _reported_cost(self, meta: Mapping[str, Any]) -> tuple[float | None, str | None, str | None]:  # Any: as above
        """OpenRouter's own figure, as it reported it, or why there is none.

        Under bring-your-own-key that figure is OpenRouter's fee alone, and the upstream charges the key
        separately (``cost_details.upstream_inference_cost``). LangChain drops the reply's ``is_byok``, so a BYOK
        reply cannot be told from one whose cost details merely itemise the same charge, and adding the two could
        count it twice: the figure is reported as OpenRouter reported it, which understates a BYOK call.
        """
        cost = meta.get("cost")
        if not isinstance(cost, int | float) or isinstance(cost, bool):
            return None, None, "OpenRouter's reply carried no cost (no usage chunk arrived, or it named none)"
        return float(cost), "openrouter:reported", None

    async def _catalogue_cost(
        self,
        usage: Mapping[str, Any],  # Any: LangChain's usage metadata
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> tuple[float | None, str | None, str | None]:
        """Tokens times the catalogue's per-million rates for the requested model, or why there is no price.

        A prompt that read or wrote the cache is priced at the plain input rate (the catalogue has no cache rates),
        so its cost is approximate and its source says so (:data:`_CACHE_APPROXIMATE`).
        """
        rates = await self._book.look_up(self.provider, self.model_name)
        if isinstance(rates, _Unpriced):
            return None, None, rates.reason
        if input_tokens is None or output_tokens is None:
            return None, None, "the reply reported no input or output token count, so there is nothing to price"
        assert rates.cost_per_1m_input is not None and rates.cost_per_1m_output is not None  # held by the book
        dollars = (input_tokens * rates.cost_per_1m_input + output_tokens * rates.cost_per_1m_output) / Decimal(
            1_000_000
        )
        cached = _mapping(usage.get("input_token_details"))
        cache_tokens = (_int_or_none(cached.get("cache_read")) or 0) + (_int_or_none(cached.get("cache_creation")) or 0)
        marker = _CACHE_APPROXIMATE if cache_tokens else ""
        return float(dollars), f"{rates.source}:{rates.matched_id}{marker}", None

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """The most one :meth:`generate` with this prompt pair can cost, or ``None`` when the client cannot say.

        The client's catalogue rates applied to a bound on the prompt's tokens (its UTF-8 bytes, which no
        byte-level tokenizer exceeds, plus an allowance for the framing the provider adds) and to the output cap
        the model was built with, reasoning included. One attempt per request, since the adapter retries nothing.

        ``None`` until the rates are loaded: **await** :meth:`load_prices` **once at boot** (on Anthropic and OpenAI a
        first priced call also loads them; on OpenRouter only ``load_prices`` does, since its calls are priced by
        OpenRouter's own figure). Also ``None`` when no catalogue row prices the model (``load_prices`` returned
        why), and when the model was built with no output cap (pass ``max_tokens``). An out-of-run call under an
        enforced cap is then refused by the engine rather than made, since unknown is not $0.

        Args:
            system: The system message the call would send.
            user: The user message the call would send.
            response_format: The directive the call would send, or ``None``.

        Returns:
            The ceiling in dollars, or ``None``.
        """
        rates = self._book.rates(self.provider, self.model_name)
        if not isinstance(rates, PriceLookup) or self._output_cap is None:
            return None
        assert rates.cost_per_1m_input is not None and rates.cost_per_1m_output is not None  # held by the book
        prompt_bound = len(system.encode()) + len(user.encode()) + _PROMPT_FRAMING_ALLOWANCE_TOKENS
        if response_format is not None:
            prompt_bound += len(json.dumps(response_format).encode())
        dollars = (prompt_bound * rates.cost_per_1m_input + self._output_cap * rates.cost_per_1m_output) / Decimal(
            1_000_000
        )
        return float(dollars)

    async def aclose(self) -> None:
        """Release the SDK client the chat model owns, on OpenRouter; nothing elsewhere.

        ``ChatOpenRouter`` builds an OpenRouter SDK client per instance, which owns its httpx clients, so this
        closes both (through the SDK's public context-manager exits). ``ChatAnthropic`` and ``ChatOpenAI`` share one
        process-wide httpx client per base URL and timeout, cached by LangChain, which every other instance in the
        process is using: closing it would break them, so this is a no-op there, and nothing leaks per instance.
        The catalogue fetch, when the host injected no price client, opens and closes its own client per fetch.
        Idempotent.
        """
        if self._closed:
            return
        self._closed = True
        if self.provider == "openrouter":
            sdk = getattr(self._chat.bound, "client", None)
            if sdk is not None:
                # The SDK's context-manager exits: each closes the httpx client it built, and leaves alone one a
                # caller supplied (a model built with default_headers gets LangChain's, which only GC frees).
                sdk.__exit__(None, None, None)
                await sdk.__aexit__(None, None, None)

    async def __aenter__(self) -> Self:
        """Enter a scope that releases the client on exit."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Release the client, whether or not the body raised."""
        await self.aclose()


def _openrouter_model_kwargs(host: object) -> dict[str, Any]:  # Any: the SDK call's own keyword values
    """The host's ``model_kwargs`` for ``ChatOpenRouter``, with the SDK's per-call retries turned off.

    ``max_retries=0`` stops LangChain building a retry config, and the OpenRouter SDK then falls back to its OWN
    default: backoff on every 5XX and connection error for up to an hour. ``retries=None`` on the call is the SDK's
    "no retry", and ``model_kwargs`` is how ``ChatOpenRouter`` passes a keyword through to every call.

    Args:
        host: The ``model_kwargs`` the host passed, or ``None``.

    Returns:
        The keywords to build with.

    Raises:
        ModelsClientRefused: The host's ``model_kwargs`` is not a mapping, or sets ``retries`` itself.
    """
    if host is None:
        return {"retries": None}
    if not isinstance(host, Mapping):
        raise ModelsClientRefused("model_kwargs", "pass a mapping of keywords for every call", provider="openrouter")
    if "retries" in host:
        raise ModelsClientRefused("model_kwargs.retries", _RESERVED_KWARGS["max_retries"], provider="openrouter")
    return {**host, "retries": None}


def _is_openrouter_alias(model_id: str) -> bool:
    """Whether OpenRouter resolves ``model_id`` on its side: a ``~`` pointer, a ``-latest`` alias, a router."""
    return model_id.startswith("~") or model_id.endswith("-latest") or model_id.startswith(_OPENROUTER_ROUTERS_PREFIX)


def models_client(
    model_id: str,
    *,
    api_key: str,
    provider: str | None = None,
    temperature: float | None = None,
    reasoning_effort: KnownReasoningEffort | None = None,
    reasoning_max_tokens: int | None = None,
    max_tokens: int | None = None,
    prices: HttpClient | None = None,
    **provider_kwargs: Any,  # Any: forwarded to the provider's LangChain constructor, which types them so
) -> ModelsCompletionClient:
    """Build the completion client for one model: ``create_chat_model`` with retries off, wrapped in the port.

    One client per model, temperature and reasoning setting, which is how the engine's role factory asks for
    them; :func:`models_clients` builds and keeps them per role.

    **Reasoning, per provider.** ``reasoning_effort`` is one of the engine's seven words:

    * Anthropic: ``max``, ``xhigh``, ``high``, ``medium`` and ``low`` go as ``output_config.effort`` (merged into
      the same ``output_config`` as a schema's format on each call); ``none`` builds thinking disabled;
      ``minimal`` is refused, since Anthropic has no such level and sending ``low`` in its place would record a
      request that was not made. ``reasoning_max_tokens`` goes as ``thinking={"type": "enabled",
      "budget_tokens": N}``, at least 1024 and below ``max_tokens``, and the model is then built with no
      temperature (the API refuses one beside thinking), so completions report ``temperature=None``.
    * OpenAI: ``reasoning_effort`` as the word; ``max`` is refused (OpenAI's highest is ``xhigh``, which is not
      sent in its place), and so is any ``reasoning_max_tokens``: OpenAI takes no reasoning budget.
    * OpenRouter: ``reasoning={"effort": word}``, all seven words; ``reasoning_max_tokens`` is refused, because
      the OpenRouter SDK strips ``reasoning.max_tokens`` from the typed request without an error.

    Asking by a word and by a budget at once is refused, as
    :class:`~threetears.evals.schema.ClientRequestSettings` refuses it.

    **The Claude CLI route is refused.** Models routes an Anthropic subscription token (``sk-ant-oat…``) to the
    Claude CLI, whose wrapper reports every completion as finished (it synthesises ``stop``), and drops the served
    model and the reasoning tokens. A judge on it cannot see its own truncation, which is a correctness hazard,
    not a missing nicety; the route opens when 3tears-models carries the CLI's real stop reason. Use an API key,
    or keep a client of your own for a subscription.

    Args:
        model_id: The model id, as the provider names it (``claude-sonnet-4-6``, ``gpt-5-mini``,
            ``~anthropic/claude-sonnet-latest``). Spend is attributed to it.
        api_key: The provider's API key.
        provider: The provider, overriding models' capability registry; required for an id the registry does not
            know (every OpenRouter route beyond the few models registers).
        temperature: The sampling temperature to build with, or ``None`` for the provider's default. What a
            completion reports is what the built model sends (:attr:`ModelsCompletionClient.temperature`).
        reasoning_effort: The reasoning level, one of the engine's seven words, or ``None`` to send none.
        reasoning_max_tokens: The reasoning token budget, or ``None`` to send none; Anthropic only.
        max_tokens: The output cap, or ``None`` for the provider's default (Anthropic's is the model profile's;
            OpenAI and OpenRouter send none). The price ceiling needs one.
        prices: An httpx-like client (``threetears.models.price_lookup.HttpClient``: one async ``get``) the
            catalogue is read through, or ``None`` for the adapter to open its own per fetch. Anthropic and OpenAI
            calls are priced from ``threetears.models.price_lookup.lookup_price``, read once per client (a host's
            source registered with ``register_price_source`` is consulted before the OpenRouter catch-all);
            OpenRouter calls by OpenRouter's own figure. Await :meth:`ModelsCompletionClient.load_prices` on the
            client before an out-of-run call under an enforced cap, so its price ceiling can answer.
        **provider_kwargs: Passed to the provider's LangChain constructor through ``create_chat_model``
            (``base_url``, ``timeout``, OpenRouter routing). A keyword the adapter sets or maps itself is refused.

    Returns:
        The client.

    Raises:
        ModelsClientRefused: The provider, the model, the key's route, a reasoning setting or a provider keyword
            cannot be carried as asked.
    """
    resolved = resolve_provider(model_id, provider)
    if resolved == "anthropic" and is_subscription_token(api_key):
        raise ModelsClientRefused(
            "api_key",
            f"a subscription token ({OAUTH_TOKEN_PREFIX}…) routes to the Claude CLI, whose completions read as "
            "finished whatever stopped them and name no served model, so a truncated judge reply would be scored "
            "as complete; use an Anthropic API key, or keep a client of your own for the CLI",
            provider=resolved,
        )
    for keyword in sorted(provider_kwargs):
        if keyword in _RESERVED_KWARGS:
            raise ModelsClientRefused(keyword, _RESERVED_KWARGS[keyword], provider=resolved)
    build, call_output_config = _reasoning_build_kwargs(resolved, reasoning_effort, reasoning_max_tokens, max_tokens)
    thinking_on = _mapping(build.get("thinking")).get("type") == "enabled"
    if temperature is not None and not thinking_on:
        build["temperature"] = temperature
    if max_tokens is not None:
        build["max_tokens"] = max_tokens
    if resolved == "openrouter":
        build["model_kwargs"] = _openrouter_model_kwargs(provider_kwargs.pop("model_kwargs", None))
    chat = create_chat_model(
        model_id,
        api_key=api_key,
        provider=resolved,
        max_retries=0,
        disable_streaming=True,
        breaker=CircuitBreaker(f"evals:{resolved}", failure_threshold=sys.maxsize),
        **build,
        **provider_kwargs,
    )
    if not isinstance(chat, RunnableBinding):
        raise TypeError(f"create_chat_model returned {type(chat).__name__}, not the binding it documents")
    built_cap = _int_or_none(getattr(chat.bound, "max_tokens", None))
    if thinking_on and reasoning_max_tokens is not None and built_cap is not None and reasoning_max_tokens >= built_cap:
        # max_tokens was not given, so the cap is the model profile's default, known only once built.
        raise ModelsClientRefused(
            "reasoning_max_tokens",
            f"Anthropic refuses a thinking budget ({reasoning_max_tokens}) that is not below the model's default "
            f"output cap ({built_cap}); pass a max_tokens above the budget",
            provider=resolved,
        )
    return ModelsCompletionClient(
        model_name=model_id,
        provider=resolved,
        chat_model=chat,
        call_output_config=call_output_config,
        prices=prices,
    )
