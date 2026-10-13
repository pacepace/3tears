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
from threetears.models import CircuitBreaker, PriceLookup, create_chat_model, get_capabilities, lookup_price
from threetears.models.price_lookup import HttpClient
from threetears.models.providers.claude_cli import OAUTH_TOKEN_PREFIX, is_subscription_token
from threetears.models.providers.structured_output import structured_output_kwargs

__all__ = [
    "PROVIDER_KEY_VARIABLES",
    "SUPPORTED_PROVIDERS",
    "CatalogueFetches",
    "ModelsClientRefused",
    "ModelsCompletion",
    "ModelsCompletionClient",
    "SupportedProvider",
    "models_client",
    "resolve_provider",
]

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
#: ``model_context_window_exceeded`` is a truncation at the window, so it reads as the cap. ``pause_turn`` is a
#: server-tool pause: output exists and stopped early for a cause that is neither the cap nor a filter, so it is
#: ``truncated``, never ``error`` (nothing failed) and never ``end_turn`` (the turn is not over).
_ANTHROPIC_STOPS: Final[Mapping[str, StopReason]] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "end_turn",
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "max_tokens",
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
    """One completion as the adapter read it off the chat model's reply: the port's ten members and two of its own.

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
            ``<source>:<matched catalogue id>`` for a rate from ``threetears.models.price_lookup``.
        model: The model id the client was built for, which is what spend is attributed to.
        served_model: The model the provider's reply named, or ``None`` when it named none.
        temperature: The temperature the request carried, or ``None`` when it carried none.
        stop_reason: The provider's stop reason in the engine's words; an absent or unknown one, and Anthropic's
            ``pause_turn``, is ``truncated``, and only a provider-reported error (OpenRouter's ``error``) is
            ``error``.
        provider_stop_reason: The stop reason as the provider named it (Anthropic's ``stop_reason``, the
            ``finish_reason`` elsewhere), or ``None`` when it named none. It is what to look at when ``stop_reason``
            is ``truncated``: the engine's word says the cause went unnamed, this says what, if anything, was sent.
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


class CatalogueFetches:
    """Each URL a price source reads, fetched once and kept: ``price_lookup`` fetches it on every lookup.

    ``threetears.models.price_lookup.lookup_price`` GETs the whole OpenRouter catalogue every time it is called,
    and caches nothing. This sits between it and the HTTP client, so a client (or, under :class:`ModelsClients`,
    every client of one factory) reads each catalogue once. Only a response that passed ``raise_for_status`` is
    kept; a failed fetch propagates, and the client that saw it records why it is unpriced.
    """

    def __init__(self, client: HttpClient | None) -> None:
        """Wrap the host's client, or none, in which case each fetch opens and closes a client of its own.

        Args:
            client: The host's httpx-like client, or ``None``.
        """
        self._client = client
        self._responses: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, url: str, /) -> Any:  # Any: models' HttpClient protocol types the response as Any
        """The response for ``url``, fetched on the first read and kept.

        Args:
            url: The URL a price source reads.

        Returns:
            The response, already checked with ``raise_for_status``.
        """
        async with self._lock:
            if url not in self._responses:
                response = await self._fetch(url)
                response.raise_for_status()
                self._responses[url] = response
            return self._responses[url]

    async def _fetch(self, url: str) -> Any:  # Any: as above
        if self._client is not None:
            return await self._client.get(url)
        async with httpx.AsyncClient(timeout=_CATALOGUE_TIMEOUT_S, follow_redirects=True) as owned:
            return await owned.get(url)


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


class ModelsCompletionClient:
    """The completion port over one chat model 3tears-models built: :func:`models_client` returns one.

    Implements :class:`~threetears.evals.schema.CompletionClient` and
    :class:`~threetears.evals.schema.BoundCompletionClient`. ``generate`` is safe to call concurrently: every call
    builds its own messages and its own bind, and the only state it writes is the pricing outcome (computed once,
    under a lock) and :attr:`last_pricing_failure`.

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
            prices: The httpx-like client the price catalogue is read through, or ``None`` for one of the
                client's own per fetch.
        """
        self.model_name = model_name
        self.provider: SupportedProvider = provider
        self._chat = chat_model
        self._call_output_config = dict(call_output_config)
        self._catalogue = prices if isinstance(prices, CatalogueFetches) else CatalogueFetches(prices)
        bound = chat_model.bound
        temperature = getattr(bound, "temperature", None)
        self._temperature = float(temperature) if isinstance(temperature, int | float) else None
        self._output_cap = _int_or_none(getattr(bound, "max_tokens", None))
        self._rates: PriceLookup | _Unpriced | None = None
        self._rates_lock = asyncio.Lock()
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
        build (OpenAI's ``gpt-5`` family), and on Anthropic a thinking budget is built with none, since the API
        refuses one beside extended thinking.
        """
        return self._temperature

    def _request_kwargs(self, response_format: dict[str, Any] | None) -> dict[str, Any]:  # Any: provider kwargs
        """The bind keywords for one call: the response format as the provider takes it, and the effort beside it."""
        kwargs: dict[str, Any] = {}
        output_config: dict[str, Any] = dict(self._call_output_config)
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
                no JSON mode. Either way the content is the reply's raw text.

        Returns:
            The completion.

        Raises:
            ModelsClientRefused: ``response_format`` is not a directive the adapter translates.
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
        if self.provider == "openrouter":
            cost, source, unpriced = self._reported_cost(meta)
        else:
            cost, source, unpriced = await self._catalogue_cost(input_tokens, output_tokens)
        self.last_pricing_failure = unpriced
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
            stop_reason=self._stop_reason(reply, meta),
            provider_stop_reason=self._provider_stop_reason(meta),
            served_provider=served_provider if isinstance(served_provider, str) and served_provider else None,
            pricing_failure=unpriced,
        )

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

    def _provider_stop_reason(self, meta: Mapping[str, Any]) -> str | None:  # Any: LangChain's metadata
        """The stop reason exactly as the provider named it, or ``None`` when it named none."""
        raw = meta.get("stop_reason") if self.provider == "anthropic" else meta.get("finish_reason")
        return raw if isinstance(raw, str) and raw else None

    def _stop_reason(self, reply: AIMessage, meta: Mapping[str, Any]) -> StopReason:  # Any: LangChain's metadata
        """The provider's stop reason in the engine's words; a refusal is the content filter, an unnamed stop is cut short."""
        if self.provider == "anthropic":
            raw = meta.get("stop_reason")
            return _ANTHROPIC_STOPS.get(raw, _UNNAMED_STOP) if isinstance(raw, str) else _UNNAMED_STOP
        if reply.additional_kwargs.get("refusal"):
            # A structured-mode refusal arrives beside empty content with finish_reason `stop`.
            return "content_filter"
        raw = meta.get("finish_reason")
        return _OPENAI_STOPS.get(raw, _UNNAMED_STOP) if isinstance(raw, str) else _UNNAMED_STOP

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
        self, input_tokens: int | None, output_tokens: int | None
    ) -> tuple[float | None, str | None, str | None]:
        """Tokens times the catalogue's per-million rates for the requested model, or why there is no price."""
        rates = await self._load_rates()
        if isinstance(rates, _Unpriced):
            return None, None, rates.reason
        if input_tokens is None or output_tokens is None:
            return None, None, "the reply reported no input or output token count, so there is nothing to price"
        assert rates.cost_per_1m_input is not None and rates.cost_per_1m_output is not None  # held by _load_rates
        dollars = (input_tokens * rates.cost_per_1m_input + output_tokens * rates.cost_per_1m_output) / Decimal(
            1_000_000
        )
        return float(dollars), f"{rates.source}:{rates.matched_id}", None

    async def _load_rates(self) -> PriceLookup | _Unpriced:
        """The client's rates, looked up once and kept, failure included: a client is priced alike for its life."""
        async with self._rates_lock:
            if self._rates is None:
                self._rates = await self._lookup_rates()
            return self._rates

    async def _lookup_rates(self) -> PriceLookup | _Unpriced:
        try:
            found = await lookup_price(self._catalogue, provider=self.provider, api_name=self.model_name)
        except (httpx.HTTPError, OSError) as exc:
            return _Unpriced(f"the price catalogue could not be fetched ({type(exc).__name__}: {exc})")
        except ValueError as exc:
            return _Unpriced(f"the price catalogue could not be read ({type(exc).__name__}: {exc})")
        if found.matched_id is None:
            return _Unpriced(f"no price catalogue row matches {self.provider} model {self.model_name!r}")
        if found.cost_per_1m_input is None or found.cost_per_1m_output is None:
            return _Unpriced(f"the price catalogue row {found.matched_id!r} carries no input or output rate")
        return found

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """The most one :meth:`generate` with this prompt pair can cost, or ``None`` when the client cannot say.

        The client's catalogue rates applied to a bound on the prompt's tokens (its UTF-8 bytes, which no
        byte-level tokenizer exceeds, plus an allowance for the framing the provider adds) and to the output cap
        the model was built with, reasoning included. One attempt per request, since the adapter retries nothing.

        ``None`` until the client has looked its rates up, which its first priced call does (the lookup is async
        and this is not), on OpenRouter (whose calls are priced by its own figure, so no rates are fetched), when
        no catalogue row prices the model, and when the model was built with no output cap. An out-of-run call
        under an enforced cap is then refused by the engine rather than made, since unknown is not $0.

        Args:
            system: The system message the call would send.
            user: The user message the call would send.
            response_format: The directive the call would send, or ``None``.

        Returns:
            The ceiling in dollars, or ``None``.
        """
        rates = self._rates
        if not isinstance(rates, PriceLookup) or self._output_cap is None:
            return None
        assert rates.cost_per_1m_input is not None and rates.cost_per_1m_output is not None  # held by _load_rates
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
            OpenRouter calls by OpenRouter's own figure.
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
