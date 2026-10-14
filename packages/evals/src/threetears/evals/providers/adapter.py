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

**The closest setting the model accepts is sent, and reported** (owner ruling, 2026-10-13: a model that lacks a
request feature never fails the request). What a client was asked for (its output cap, temperature, reasoning word
or budget, and per call whether the reply format is enforced) goes through :func:`closest_settings` against what
is known of the model (:class:`SettingFacts`), and the request carries the result: a word the model lacks as its
closest word on the same side of the on/off line (ties go lower, so never more than was asked), a budget as the
effort word its share of the cap maps to on a model that takes words only (:data:`BUDGET_SHARES`), a word as a
budget on a model that takes budgets only, Anthropic's effort words over adaptive thinking with no temperature
beside them, a strict reply format the model cannot take as an unenforced one, an output cap above the model's
known one as that cap. What is known comes, highest first, from what the provider refused in this factory's life
(:class:`LearnedSettings`), 3tears-models' capabilities (read with ``getattr``, so a models build that declares
none of these facts leaves every one unknown, and including the facts a provider published for a model the static
registry does not know), and on OpenRouter its cached catalogue's ``supported_parameters``. Unknown is sent as
asked. Each completion's ``sent_settings`` is what its request carried, and its ``temperature`` likewise.

**A setting the provider refuses is learned, once, per factory.** A request refused for a setting it carried, read
off each SDK's error type and structured fields (Anthropic's ``invalid_request_error`` naming the parameter as
deprecated or unsupported; OpenAI's ``unsupported_parameter`` or ``unsupported_value`` naming it in ``param``; on
OpenRouter a 404 for a call under ``require_parameters``, which names nothing), is sent again with the closest
setting the model then accepts, and once that is served the fact is kept for every client of the factory and
logged once at WARNING. On OpenRouter the order is fixed: the temperature first, keeping enforcement, and only if
that still finds no endpoint, enforcement itself (sent ``strict_output=False``). That is not a hidden retry: a
refused request was rejected before anything was generated and is not billed (an assumption no provider documents,
which the price ceiling rests on). At most one resend per setting and four per call; any other refusal, and a
refusal of the resend for a setting already dropped, propagates as it was raised.

**Built per call.** The messages, what to send (:func:`closest_settings` over the facts as they stand, since another
client of the factory may have learned one since), and the bind that carries a ``response_format`` (and, on
Anthropic, the effort in the same ``output_config``). A call whose settings differ from those the chat model was
built with goes through a copy of it at those settings, made once and kept, which shares the original's SDK
client, config and callbacks. Nothing about a call is written to the client, so concurrent calls on one instance
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
registry cannot place, the Claude CLI route, a provider keyword that would bypass what the adapter guarantees
(retries, breaker, streaming, reasoning), and the caller's own errors in a reasoning setting: a word and a budget
together, a word that is not one of the engine's seven, a budget below Anthropic's 1024 floor, a budget not below
the output cap, and a budget with no cap to take its share of. Each is a :class:`ModelsClientRefused` naming the
setting and what to pass instead. A setting the model merely does not take is never refused: it is sent as the
closest one the model accepts, and the completion says so.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from typing import Any, Final, Literal, Self, get_args

import anthropic
import httpx
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableBinding
from openrouter import errors as openrouter_errors

import threetears.models as models_api
from threetears.evals.schema import ClientRequestSettings, KnownReasoningEffort, StopReason
from threetears.evals.schema.logs import get_logger
from threetears.models import CircuitBreaker, PriceLookup, create_chat_model, get_capabilities, lookup_price
from threetears.models.price_lookup import OPENROUTER_MODELS_URL, HttpClient
from threetears.models.providers.claude_cli import OAUTH_TOKEN_PREFIX, is_subscription_token
from threetears.models.providers.structured_output import structured_output_kwargs

__all__ = [
    "BUDGET_SHARES",
    "EFFORT_LADDER",
    "PROVIDER_KEY_VARIABLES",
    "SUPPORTED_PROVIDERS",
    "LearnedSettings",
    "ModelsClientRefused",
    "ModelsCompletion",
    "ModelsCompletionClient",
    "PriceBook",
    "SettingFacts",
    "SupportedProvider",
    "WireSettings",
    "closest_settings",
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

#: The engine's seven reasoning words (:data:`~threetears.evals.schema.REASONING_EFFORTS`) as a ladder, lowest
#: first: what "closest" is measured on. ``none`` is the off side of the line, every other word the on side.
EFFORT_LADDER: Final[tuple[str, ...]] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

#: The share of the output cap each effort word stands for: OpenRouter's documented ratios for turning an effort
#: into a reasoning budget on a model that takes budgets, so a budget sent to OpenRouter as its word comes back as
#: about the budget asked (the judge's 8192 of 10240 is 0.80, ``high``, and ``high`` is 0.80 of the cap again).
#: Read both ways: a budget asked of a model that takes words only is sent as the word whose share is nearest its
#: own (ties go to the lower word, so the boundaries 0.15, 0.35, 0.65 and 0.875 all go lower), and a word asked of a
#: model that takes budgets only is sent as that share of the cap. ``max`` has no share of its own above ``xhigh``'s
#: and is sent as ``xhigh``'s budget; ``none`` is no reasoning, never a budget.
BUDGET_SHARES: Final[Mapping[str, Fraction]] = {
    "minimal": Fraction("0.10"),
    "low": Fraction("0.20"),
    "medium": Fraction("0.50"),
    "high": Fraction("0.80"),
    "xhigh": Fraction("0.95"),
    "max": Fraction("0.95"),
}

#: The effort words each provider's API names, which is what a model of it is sent from when nothing narrower is
#: known of the model. Anthropic's ``none`` is thinking disabled, and its others ride adaptive thinking as
#: ``output_config.effort``; OpenAI's ``reasoning_effort`` has no ``max``; OpenRouter's SDK types all seven.
_PROVIDER_EFFORTS: Final[Mapping[SupportedProvider, tuple[str, ...]]] = {
    "anthropic": ("none", "low", "medium", "high", "xhigh", "max"),
    "openai": ("none", "minimal", "low", "medium", "high", "xhigh"),
    "openrouter": EFFORT_LADDER,
}

#: Anthropic's floor on a thinking budget, in tokens.
_ANTHROPIC_MIN_THINKING_BUDGET: Final = 1024

#: The most times one call is resent for a setting the provider refused: one per setting the adapter can drop or
#: move (the temperature, the reasoning parameter, the reply format, enforcement).
_MAX_RESENDS: Final = 4

#: The chat-model fields a call's settings are built into: what a copy of the client's model is made at, and kept
#: under (:meth:`ModelsCompletionClient._variant`). A field a provider's model does not have is left alone.
_BUILT_FIELDS: Final = ("temperature", "max_tokens", "thinking", "reasoning_effort", "reasoning")

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

#: Words a provider's refusal uses when a parameter is not taken at all, as against a value out of range (Anthropic's
#: range error), which is the caller's to fix and is never resent. Anthropic's refusal names the parameter in its
#: message only ("`temperature` is deprecated for this model."), so the message is read for the parameter's name and
#: one of these words, never matched as a whole sentence.
_NOT_TAKEN_WORDS: Final = ("deprecated", "not supported", "unsupported", "does not support")

#: OpenAI's error codes for a parameter the model does not take (``Unsupported parameter: 'temperature' …``) and for
#: a value of it the model does not take (``Unsupported value: 'temperature' does not support 0.2 …``); either
#: arrives with ``param`` naming the parameter.
_OPENAI_NOT_TAKEN_CODES: Final = frozenset({"unsupported_parameter", "unsupported_value"})

#: What an absent or unknown finish reason reads as: cut short, cause unnamed, never finished. A reply with no
#: finish reason (a stream that ended without one) or with one this build does not know (a future provider value)
#: still carried output, so it is not ``error``, which says the call failed; and a truncation read as ``end_turn``
#: is the defect the port exists to prevent. The raw value rides beside it, as
#: :attr:`ModelsCompletion.provider_stop_reason`.
_UNNAMED_STOP: Final[StopReason] = "truncated"


class ModelsClientRefused(ValueError):
    """A setting the adapter refuses by name before anything is sent: the caller's error, never a model's lack.

    Raised at construction for a provider or model the adapter cannot build for, a provider keyword that would bypass
    it, and a reasoning setting that is the caller's error (never for one the model merely does not take, which is
    sent as the closest setting it accepts); by :meth:`ModelsCompletionClient.generate` for a ``response_format`` it
    cannot translate; and
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
    """One completion as the adapter read it off the chat model's reply: the port's members, and three of its own.

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
        temperature: The temperature the request carried, or ``None`` when it carried none: dropped because the model
            takes none (known, or refused and resent without), because Anthropic takes none beside thinking, or
            because LangChain dropped it.
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
        sent_settings: What the served request carried (the port's optional member): its output cap, its reasoning
            effort word or budget as it went on the wire (after any mapping to the closest setting the model
            accepts, and any resend), and ``strict_output`` exactly when the request enforced its reply format (a
            strict schema on Anthropic and OpenAI, ``require_parameters`` on OpenRouter). ``None`` when the request
            carried no output cap (OpenAI and OpenRouter built without ``max_tokens``), since a settings value
            names one; the temperature is ``temperature``'s.
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
    sent_settings: ClientRequestSettings | None


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

    It is also what those clients share of what they learn: the request settings a provider refused
    (:attr:`learned`), and from OpenRouter's catalogue, once read, which parameters each model takes
    (:meth:`supported_parameters`).

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
        self._parameters: dict[str, frozenset[str] | None] | None = None
        self.learned = LearnedSettings()
        """The request settings the provider refused, per model, for every client that reads this book."""

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

    def supported_parameters(self, model: str) -> frozenset[str] | None:
        """The parameters OpenRouter's catalogue lists for ``model``, once the catalogue was read; else ``None``.

        Never fetches: the catalogue is read by :meth:`get` (a ``load_prices`` on an OpenRouter client reads it), and
        until then, or for a model the catalogue lists without ``supported_parameters`` or not at all, nothing is
        known. Parsed once, on the first ask after the read.

        Args:
            model: The OpenRouter model id, as the catalogue's ``id`` names it.

        Returns:
            The listed parameters, or ``None``.
        """
        if self._parameters is None:
            response = self._responses.get(OPENROUTER_MODELS_URL)
            if response is None:
                return None
            payload = response.json()
            rows = payload.get("data") if isinstance(payload, Mapping) else None
            self._parameters = {
                row["id"]: (
                    frozenset(p for p in row["supported_parameters"] if isinstance(p, str))
                    if isinstance(row.get("supported_parameters"), list)
                    else None
                )
                for row in (rows if isinstance(rows, list) else [])
                if isinstance(row, Mapping) and isinstance(row.get("id"), str)
            }
        return self._parameters.get(model)

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


@dataclass(frozen=True, slots=True, kw_only=True)
class WireSettings:
    """Request settings as a client is asked for them, or as one request carries them.

    The adapter's own shape, beside :class:`~threetears.evals.schema.ClientRequestSettings`, because a client can
    be built with no output cap and with a temperature, which that shape does not hold. :func:`closest_settings`
    takes the asked value and returns the sent one; :meth:`reported` is the sent value as a completion's
    ``sent_settings``.

    Attributes:
        max_tokens: The output cap, or ``None`` for the provider's default (none is sent).
        temperature: The sampling temperature, or ``None`` for none.
        reasoning_effort: An effort word on :data:`EFFORT_LADDER`, or ``None``.
        reasoning_max_tokens: A reasoning budget in tokens, or ``None``; never beside ``reasoning_effort``.
        strict_output: Whether the reply format is enforced: a strict schema a direct provider refuses rather than
            ignores, or OpenRouter's ``require_parameters``.
    """

    max_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    reasoning_max_tokens: int | None = None
    strict_output: bool = False

    def reported(self) -> ClientRequestSettings | None:
        """These settings as a completion's ``sent_settings``, or ``None`` when they name no output cap.

        Returns:
            The settings value; the temperature is not part of it (a completion reports it as ``temperature``).
        """
        if self.max_tokens is None:
            return None
        return ClientRequestSettings(
            max_tokens=self.max_tokens,
            reasoning_max_tokens=self.reasoning_max_tokens,
            reasoning_effort=self.reasoning_effort,
            strict_output=self.strict_output,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SettingFacts:
    """What is known of the request settings one model accepts: each fact ``None`` while nothing says.

    3tears-models' capability fields of the same names fill them (read with ``getattr``, so a models build without
    them leaves them unknown), a refusal the provider answered fills them for the factory's life
    (:class:`LearnedSettings`), and on OpenRouter the catalogue's ``supported_parameters`` does. Unknown is sent as
    asked.

    Attributes:
        accepts_temperature: Whether a temperature is taken.
        reasoning_efforts: The provider-native effort words the model takes; ``()`` when it takes no effort
            parameter at all. Anthropic's off switch is not a word here but ``reasoning_disable``.
        refused_efforts: Words the provider refused for this model, whatever ``reasoning_efforts`` says.
        reasoning_budget: Whether a reasoning token budget is taken (Anthropic's ``thinking.budget_tokens``).
        adaptive_reasoning: Whether Anthropic's adaptive thinking is taken, which its effort words ride on.
        reasoning_disable: Whether reasoning can be switched off (Anthropic's thinking disabled, or a provider's
            ``none``).
        structured_output: Whether a strict ``json_schema`` reply format is taken.
        enforcement: Whether OpenRouter finds an endpoint for the model's requests under ``require_parameters``.
        max_output_tokens: The model's own output cap, or ``None`` when unknown.
    """

    accepts_temperature: bool | None = None
    reasoning_efforts: tuple[str, ...] | None = None
    refused_efforts: frozenset[str] = frozenset()
    reasoning_budget: bool | None = None
    adaptive_reasoning: bool | None = None
    reasoning_disable: bool | None = None
    structured_output: bool | None = None
    enforcement: bool | None = None
    max_output_tokens: int | None = None

    def over(self, lower: SettingFacts) -> SettingFacts:
        """These facts, each unknown one taken from ``lower``; refused words from both.

        Args:
            lower: The facts of a source below this one.

        Returns:
            The combined facts.
        """
        return SettingFacts(
            accepts_temperature=_first(self.accepts_temperature, lower.accepts_temperature),
            reasoning_efforts=self.reasoning_efforts if self.reasoning_efforts is not None else lower.reasoning_efforts,
            refused_efforts=self.refused_efforts | lower.refused_efforts,
            reasoning_budget=_first(self.reasoning_budget, lower.reasoning_budget),
            adaptive_reasoning=_first(self.adaptive_reasoning, lower.adaptive_reasoning),
            reasoning_disable=_first(self.reasoning_disable, lower.reasoning_disable),
            structured_output=_first(self.structured_output, lower.structured_output),
            enforcement=_first(self.enforcement, lower.enforcement),
            max_output_tokens=self.max_output_tokens if self.max_output_tokens is not None else lower.max_output_tokens,
        )


def _first(known: bool | None, fallback: bool | None) -> bool | None:
    """``known`` unless it is unknown, then ``fallback``."""
    return known if known is not None else fallback


def _rung(word: str) -> int:
    """Where ``word`` stands on :data:`EFFORT_LADDER`."""
    return EFFORT_LADDER.index(word)


def _closest_word(word: str, on: Sequence[str], off: bool) -> str | None:
    """The accepted word closest to ``word``, on its own side of the on/off line.

    ``none`` asked of a model that cannot switch reasoning off crosses the line to the lowest word that reasons. A
    reasoning word asked of a model that takes none never crosses to ``none``: no reasoning parameter is sent, which
    is what such a model does by default, rather than an off switch nobody asked for.

    Args:
        word: The word asked for, on :data:`EFFORT_LADDER`.
        on: The accepted words that reason (every one but ``none``).
        off: Whether reasoning can be switched off (``none`` is accepted).

    Returns:
        The word to send, the lower of two equally close; ``None`` for no reasoning parameter.
    """
    if word == "none":
        if off:
            return "none"
        return min(on, key=_rung) if on else None
    if not on:
        return None
    return min(on, key=lambda candidate: (abs(_rung(candidate) - _rung(word)), _rung(candidate)))


def _word_for_share(budget: int, cap: int) -> str:
    """The effort word whose share of the cap (:data:`BUDGET_SHARES`) is nearest ``budget / cap``; ties go lower."""
    share = Fraction(budget, cap)
    return min(
        (word for word in BUDGET_SHARES if word != "max"),
        key=lambda word: (abs(BUDGET_SHARES[word] - share), _rung(word)),
    )


def _budget_for_word(word: str, cap: int) -> int | None:
    """A thinking budget for ``word``: its share of the cap, at least Anthropic's floor and below the cap, if any is."""
    if cap <= _ANTHROPIC_MIN_THINKING_BUDGET:
        return None
    share = BUDGET_SHARES.get(word, BUDGET_SHARES["minimal"])
    return min(max(round(share * cap), _ANTHROPIC_MIN_THINKING_BUDGET), cap - 1)


def _vocabulary(provider: SupportedProvider, facts: SettingFacts) -> tuple[list[str], bool]:
    """The words the model is sent from: those that reason, lowest first, and whether reasoning can be switched off."""
    words = facts.reasoning_efforts if facts.reasoning_efforts is not None else _PROVIDER_EFFORTS[provider]
    on = [word for word in EFFORT_LADDER if word != "none" and word in words and word not in facts.refused_efforts]
    if provider == "anthropic":
        # Anthropic's off switch is thinking disabled, which no effort list names.
        off = facts.reasoning_disable is not False and "none" not in facts.refused_efforts
    else:
        off = "none" in words and "none" not in facts.refused_efforts and facts.reasoning_disable is not False
    return on, off


def _closest_reasoning(
    provider: SupportedProvider,
    asked: WireSettings,
    cap: int | None,
    facts: SettingFacts,
    notes: list[str],
) -> tuple[str | None, int | None]:
    """The reasoning word and budget to send: one, the other, or neither."""
    takes_budget = provider == "anthropic" and facts.reasoning_budget is not False
    effort, budget = asked.reasoning_effort, asked.reasoning_max_tokens
    if budget is not None:
        if takes_budget:
            if cap is None or budget < cap:
                return None, budget
            if cap - 1 >= _ANTHROPIC_MIN_THINKING_BUDGET:
                notes.append(f"reasoning budget {budget} sent as {cap - 1}, below the model's output cap {cap}")
                return None, cap - 1
        if asked.max_tokens is None:
            if takes_budget:
                return None, budget
            notes.append(f"reasoning budget {budget} sent as no reasoning: there is no max_tokens to take its share of")
            return None, None
        effort = _word_for_share(budget, asked.max_tokens)
        why = "the model takes no reasoning budget" if not takes_budget else f"the model's output cap is {cap}"
        notes.append(
            f"reasoning budget {budget} of max_tokens {asked.max_tokens} (share "
            f"{float(Fraction(budget, asked.max_tokens)):.2f}) taken as effort {effort!r}: {why}"
        )
    if effort is None:
        return None, None
    on, off = _vocabulary(provider, facts)
    if provider == "anthropic" and not on and takes_budget and effort != "none":
        if cap is None:
            notes.append(
                f"effort {effort!r} sent as no reasoning: the model takes budgets only, and no max_tokens sizes one"
            )
            return None, None
        sent_budget = _budget_for_word(effort, cap)
        notes.append(
            f"effort {effort!r} sent as a thinking budget of {sent_budget} ({float(BUDGET_SHARES[effort]):.2f} of "
            f"max_tokens {cap}): the model takes budgets, not effort words"
            if sent_budget is not None
            else f"effort {effort!r} sent as no reasoning: the model takes budgets only, and max_tokens {cap} leaves "
            "no room for one"
        )
        return None, sent_budget
    word = _closest_word(effort, on, off)
    if word is None:
        notes.append(f"effort {effort!r} sent as no reasoning parameter: the model takes none")
    elif word != effort:
        notes.append(f"effort {effort!r} sent as {word!r}, the closest word the model takes")
    return word, None


def closest_settings(
    asked: WireSettings, facts: SettingFacts, provider: SupportedProvider
) -> tuple[WireSettings, tuple[str, ...]]:
    """What to send for ``asked``: the closest settings ``facts`` say the model accepts, and why each one moved.

    Pure: the same inputs give the same answer, and anything ``facts`` leave unknown is sent as asked, for the
    provider to refuse and the adapter to learn from. The rules, per setting:

    * **Output cap**: above the model's known cap (``max_output_tokens``), that cap.
    * **Effort word**: the accepted word closest on :data:`EFFORT_LADDER`, on the same side of the on/off line
      (``none`` is off), crossing it only for ``none`` asked of a model that cannot switch reasoning off, which gets
      its lowest word; a tie goes to the lower word, which never spends more than was asked. The words a model is
      sent from are its declared ones, else its provider's (:data:`_PROVIDER_EFFORTS`: ``max`` on OpenAI is
      ``xhigh``, ``minimal`` on Anthropic is ``low``), less any it refused. A model that takes no effort word is
      sent none, except an Anthropic model that takes a budget, which is sent the word's share of the cap
      (:data:`BUDGET_SHARES`), at least 1024 and below the cap.
    * **Budget**: as asked on an Anthropic model that takes one (lowered below a cap the clamp moved under it);
      otherwise the effort word its share of ``max_tokens`` maps to (:data:`BUDGET_SHARES`), then as a word. OpenAI
      takes no budget, and OpenRouter's SDK types none, so there a budget is always a word.
    * **Temperature**: dropped when the model takes none, and on Anthropic beside any thinking (a budget, or an
      effort word over adaptive thinking), which the API refuses.
    * **Strict output**: kept unless the model takes no strict schema, or, on OpenRouter, no endpoint serves its
      requests under ``require_parameters``; the request then carries the format unenforced.

    Args:
        asked: What the client was asked for, ``strict_output`` being the call's own.
        facts: What is known of the model.
        provider: The provider the request goes to.

    Returns:
        The settings the request carries, and one sentence per setting that moved, for the log.
    """
    notes: list[str] = []
    cap = asked.max_tokens
    if cap is not None and facts.max_output_tokens is not None and cap > facts.max_output_tokens:
        notes.append(f"max_tokens {cap} sent as {facts.max_output_tokens}, the model's output cap")
        cap = facts.max_output_tokens
    effort, budget = _closest_reasoning(provider, asked, cap, facts, notes)
    temperature = asked.temperature
    if temperature is not None and facts.accepts_temperature is False:
        notes.append(f"temperature {temperature} sent as none: the model takes no temperature")
        temperature = None
    elif temperature is not None and provider == "anthropic" and _anthropic_thinks(effort, budget, facts):
        notes.append(f"temperature {temperature} sent as none: Anthropic takes no temperature beside thinking")
        temperature = None
    strict = asked.strict_output
    if strict and facts.structured_output is False:
        notes.append("the reply format sent unenforced: the model takes no strict schema")
        strict = False
    elif strict and provider == "openrouter" and facts.enforcement is False:
        notes.append("the reply format sent without require_parameters: no endpoint serves the model under it")
        strict = False
    sent = WireSettings(
        max_tokens=cap,
        temperature=temperature,
        reasoning_effort=effort,
        reasoning_max_tokens=budget,
        strict_output=strict,
    )
    return sent, tuple(notes)


def _anthropic_thinks(effort: str | None, budget: int | None, facts: SettingFacts) -> bool:
    """Whether an Anthropic request with this reasoning carries thinking: a budget, or a word over adaptive thinking."""
    return budget is not None or (effort not in (None, "none") and facts.adaptive_reasoning is not False)


def _anthropic_thinking(sent: WireSettings, facts: SettingFacts) -> dict[str, Any] | None:  # Any: the API's JSON
    """The ``thinking`` an Anthropic request carries for ``sent``, or ``None`` for none."""
    if sent.reasoning_max_tokens is not None:
        return {"type": "enabled", "budget_tokens": sent.reasoning_max_tokens}
    if sent.reasoning_effort == "none":
        return {"type": "disabled"}
    if sent.reasoning_effort is not None and facts.adaptive_reasoning is not False:
        return {"type": "adaptive"}
    return None


def _flag(capabilities: object, name: str) -> bool | None:
    """A tri-state capability field, or ``None`` when this models build does not declare it.

    Raises:
        TypeError: The field holds something other than a bool or ``None``: a models defect, not an unknown.
    """
    value = getattr(capabilities, name, None)
    if value is None or isinstance(value, bool):
        return value
    raise TypeError(f"3tears-models' capability {name!r} holds a {type(value).__name__}, not a bool or None")


def _registry_facts(model_id: str) -> SettingFacts:
    """What 3tears-models says of ``model_id``'s request settings: its capabilities, else what a provider published.

    ``get_capabilities`` applies the published layer and any override over a static entry; a model with no static
    entry (``claude-haiku-5-5``) is read from ``get_published_capabilities`` alone, when this models build has it.
    Each field is read with ``getattr``, so a build that declares none of them leaves every fact unknown.

    Raises:
        TypeError: A field holds a value of the wrong type.
    """
    capabilities: object = models_api.get_capabilities(model_id)
    if capabilities is None:
        published = getattr(models_api, "get_published_capabilities", None)
        capabilities = published(model_id) if callable(published) else None
    if capabilities is None:
        return SettingFacts()
    efforts = getattr(capabilities, "reasoning_efforts", None)
    if efforts is not None and not (isinstance(efforts, list | tuple) and all(isinstance(w, str) for w in efforts)):
        raise TypeError(f"3tears-models' capability 'reasoning_efforts' holds {efforts!r}, not a list of words")
    return SettingFacts(
        accepts_temperature=_flag(capabilities, "accepts_temperature"),
        reasoning_efforts=tuple(efforts) if efforts is not None else None,
        reasoning_budget=_flag(capabilities, "reasoning_budget"),
        adaptive_reasoning=_flag(capabilities, "adaptive_reasoning"),
        reasoning_disable=_flag(capabilities, "reasoning_disable"),
        structured_output=_flag(capabilities, "structured_output"),
        max_output_tokens=_int_or_none(getattr(capabilities, "max_output_tokens", None)),
    )


def _catalogue_facts(parameters: frozenset[str] | None) -> SettingFacts:
    """What OpenRouter's catalogue says a model takes: its ``supported_parameters``, or nothing when it lists none."""
    if parameters is None:
        return SettingFacts()
    return SettingFacts(
        accepts_temperature="temperature" in parameters,
        reasoning_efforts=None if "reasoning" in parameters else (),
        structured_output="structured_outputs" in parameters,
    )


#: What a refusal is of: the fact it teaches.
_Refused = Literal["temperature", "effort_word", "effort", "budget", "adaptive", "disable", "format", "enforcement"]


@dataclass(frozen=True, slots=True)
class _Refusal:
    """A provider refusing a request for one setting it carried, in the provider's own words."""

    setting: _Refused
    message: str
    word: str | None = None
    accepted: tuple[str, ...] | None = None

    def teach(self, facts: SettingFacts) -> SettingFacts:
        """``facts`` with what this refusal says added."""
        if self.setting == "temperature":
            return replace(facts, accepts_temperature=False)
        if self.setting == "effort_word":
            refused = facts.refused_efforts | ({self.word} if self.word else set())
            efforts = self.accepted if self.accepted is not None else facts.reasoning_efforts
            return replace(facts, refused_efforts=frozenset(refused), reasoning_efforts=efforts)
        if self.setting == "effort":
            return replace(facts, reasoning_efforts=())
        if self.setting == "budget":
            return replace(facts, reasoning_budget=False)
        if self.setting == "adaptive":
            return replace(facts, adaptive_reasoning=False)
        if self.setting == "disable":
            return replace(facts, reasoning_disable=False)
        if self.setting == "format":
            return replace(facts, structured_output=False)
        return replace(facts, enforcement=False)

    @property
    def label(self) -> str:
        """The refused setting, as the log names it."""
        return {
            "temperature": "a temperature",
            "effort_word": f"reasoning effort {self.word!r}",
            "effort": "an effort parameter",
            "budget": "a thinking budget",
            "adaptive": "adaptive thinking",
            "disable": "thinking disabled",
            "format": "a strict reply format",
            "enforcement": "require_parameters",
        }[self.setting]


class LearnedSettings:
    """What providers refused, per provider and model, kept for the life of the clients that share it.

    Every client of one :class:`~threetears.evals.providers.ModelsClients` shares one (through the factory's
    :class:`PriceBook`), so a refusal one client met is not met again by another client of the same model; a client
    built alone with :func:`models_client` has its own. Not persisted: a provider may start accepting a setting, and
    each completion already records what its request carried. A refusal wins over what 3tears-models declared, and
    a contradiction is logged as the registry error it is.
    """

    def __init__(self) -> None:
        """Know nothing yet."""
        self._facts: dict[tuple[str, str], SettingFacts] = {}

    def facts(self, provider: str, model: str) -> SettingFacts:
        """What has been learned of ``model`` on ``provider``; every fact unknown until a refusal.

        Args:
            provider: The provider.
            model: The model id.

        Returns:
            The learned facts.
        """
        return self._facts.get((provider, model), SettingFacts())

    def learn(
        self,
        provider: str,
        model: str,
        refusals: Sequence[_Refusal],
        *,
        registry: SettingFacts,
        sent: WireSettings,
    ) -> None:
        """Keep what each refusal of a call says, now that the request without those settings was served.

        Logs each fact not already known once, at WARNING, with the provider's message (never the error's text,
        which on OpenAI and Anthropic repeats the whole body), what the registry said of it, and what the served
        request carried in its place.

        Args:
            provider: The provider.
            model: The model id.
            refusals: The call's refusals, in the order met.
            registry: What 3tears-models said of the model, for the log.
            sent: What the served request carried.
        """
        for refusal in refusals:
            known = self.facts(provider, model)
            taught = refusal.teach(known)
            if taught == known:
                continue
            self._facts[(provider, model)] = taught
            log.warning(
                "eval.providers model refuses %s provider=%s model=%s registry=%s; the request was resent as %s and "
                "is sent so from now on: %s",
                refusal.label,
                provider,
                model,
                _registry_said(refusal, registry),
                _settings_text(sent),
                refusal.message[:500],
            )


def _registry_said(refusal: _Refusal, registry: SettingFacts) -> str:
    """What 3tears-models declared of the refused setting: ``unknown``, or a contradiction to fix there."""
    declared = {
        "temperature": registry.accepts_temperature,
        "budget": registry.reasoning_budget,
        "adaptive": registry.adaptive_reasoning,
        "disable": registry.reasoning_disable,
        "format": registry.structured_output,
        "effort": None if registry.reasoning_efforts is None else bool(registry.reasoning_efforts),
        "effort_word": None if registry.reasoning_efforts is None else refusal.word in registry.reasoning_efforts,
        "enforcement": None,
    }[refusal.setting]
    if declared is None:
        return "unknown"
    return "accepts (a registry error to correct in 3tears-models)" if declared else "refuses"


def _settings_text(sent: WireSettings) -> str:
    """``sent`` as one line for the log."""
    reasoning = (
        f"budget {sent.reasoning_max_tokens}"
        if sent.reasoning_max_tokens is not None
        else f"effort {sent.reasoning_effort!r}"
        if sent.reasoning_effort is not None
        else "no reasoning parameter"
    )
    temperature = "no temperature" if sent.temperature is None else f"temperature {sent.temperature}"
    strict = "enforced" if sent.strict_output else "unenforced"
    return f"max_tokens={sent.max_tokens}, {reasoning}, {temperature}, {strict}"


#: An OpenAI ``unsupported_value`` message's list of what the model does take ("Supported values are: 'low',
#: 'medium', and 'high'."), read only to fill the accepted words of a refusal already read off ``code``.
_SUPPORTED_VALUES = re.compile(r"supported values are:(?P<values>[^.]*)", re.IGNORECASE)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Carried:
    """What one request carried, of the settings a provider can refuse: what a refusal may be read as being of."""

    temperature: bool
    effort: str | None
    thinking: str | None
    strict_format: bool
    enforcement: bool


def _refusal(provider: SupportedProvider, exc: Exception, carried: _Carried, facts: SettingFacts) -> _Refusal | None:
    """Whether ``exc`` is the provider refusing a request for a setting it carried, which, and in what words.

    Read by each SDK's error type and structured fields, never by the exception's text, which on Anthropic and
    OpenAI repeats the whole body; and only as a refusal of a setting the request carried:

    * Anthropic: a 400 ``invalid_request_error`` whose message says a parameter is not taken (:data:`_NOT_TAKEN_WORDS`)
      and names one the request carried: ``temperature``; ``thinking.type.enabled`` or ``budget_tokens`` (a budget);
      ``thinking.type.adaptive``; ``thinking.type.disabled``; ``effort`` (the word, when the message quotes it, else
      the parameter); ``output_config.format`` or ``structured output``. The one named first is the refused one,
      since the message goes on to name what to send instead. A value out of range (a budget below 1024) says
      nothing is unsupported and is the caller's, never resent.
    * OpenAI: a 400 whose ``code`` is ``unsupported_parameter`` or ``unsupported_value`` and whose ``param`` names
      ``temperature`` or ``reasoning_effort`` (an ``unsupported_value`` there is the word, and its message's
      "Supported values are" list fills the words the model takes), or whose ``param`` is ``response_format`` with a
      message saying the format is not supported.
    * OpenRouter: by default it drops a parameter the routed upstream does not take, so no refusal arrives; under
      ``require_parameters`` an upstream that cannot take one is excluded, and when none is left it answers 404,
      naming no parameter. That is read as the temperature's while the request carried one the facts do not say
      is taken, and otherwise as enforcement's: the resend settles which.

    Args:
        provider: The provider the request went to.
        exc: What the call raised.
        carried: What the request carried.
        facts: What was known of the model when the request was built.

    Returns:
        The refusal, or ``None`` when ``exc`` is anything else.
    """
    if provider == "anthropic" and isinstance(exc, anthropic.BadRequestError):
        error = _mapping(_mapping(exc.body).get("error"))
        message = _error_message(exc.body)
        if error.get("type") != "invalid_request_error" or message is None:
            return None
        said = message.lower()
        if not any(word in said for word in _NOT_TAKEN_WORDS):
            return None
        named: list[tuple[int, _Refusal]] = []

        def name(tokens: Sequence[str], refusal: _Refusal) -> None:
            found = [said.find(token) for token in tokens if token in said]
            if found:
                named.append((min(found), refusal))

        if carried.temperature:
            name(("temperature",), _Refusal("temperature", message))
        if carried.thinking == "enabled":
            name(("thinking.type.enabled", "budget_tokens"), _Refusal("budget", message))
        elif carried.thinking == "adaptive":
            name(("thinking.type.adaptive",), _Refusal("adaptive", message))
        elif carried.thinking == "disabled":
            name(("thinking.type.disabled",), _Refusal("disable", message))
        if carried.effort is not None and carried.effort != "none":
            quoted = f"'{carried.effort}'" in said or f'"{carried.effort}"' in said or f"`{carried.effort}`" in said
            name(
                ("effort",),
                _Refusal("effort_word", message, word=carried.effort) if quoted else _Refusal("effort", message),
            )
        if carried.strict_format:
            name(("output_config.format", "structured output"), _Refusal("format", message))
        return min(named, key=lambda found: found[0])[1] if named else None
    if provider == "openai" and isinstance(exc, openai.BadRequestError):
        message = _error_message(exc.body) or f"{exc.code}: {exc.param}"
        if exc.param == "temperature" and carried.temperature and exc.code in _OPENAI_NOT_TAKEN_CODES:
            return _Refusal("temperature", message)
        if exc.param == "reasoning_effort" and carried.effort is not None and exc.code in _OPENAI_NOT_TAKEN_CODES:
            if exc.code == "unsupported_parameter":
                return _Refusal("effort", message)
            listed = _SUPPORTED_VALUES.search(message)
            accepted = (
                tuple(word for word in EFFORT_LADDER if f"'{word}'" in listed.group("values")) if listed else None
            )
            return _Refusal("effort_word", message, word=carried.effort, accepted=accepted or None)
        if (
            exc.param == "response_format"
            and carried.strict_format
            and (exc.code in _OPENAI_NOT_TAKEN_CODES or "not supported" in message.lower())
        ):
            return _Refusal("format", message)
        return None
    if provider == "openrouter" and carried.enforcement and isinstance(exc, openrouter_errors.NotFoundResponseError):
        if carried.temperature and facts.accepts_temperature is not True:
            return _Refusal("temperature", exc.message)
        return _Refusal("enforcement", exc.message)
    return None


def _refuse_caller_errors(
    provider: SupportedProvider,
    effort: str | None,
    budget: int | None,
    max_tokens: int | None,
) -> None:
    """Refuse, by name, a reasoning setting that is the caller's error whatever the model takes.

    Args:
        provider: The provider.
        effort: The reasoning word asked for, or ``None``.
        budget: The reasoning token budget asked for, or ``None``.
        max_tokens: The output cap asked for, or ``None`` for the provider's default.

    Raises:
        ModelsClientRefused: A word and a budget together; a word that is not one of the engine's seven; a budget
            below Anthropic's floor, or not below the output cap; a budget off Anthropic with no cap to take a share
            of.
    """
    if effort is not None and budget is not None:
        raise ModelsClientRefused(
            "reasoning_effort",
            "reasoning is asked for one way, by a level or by a token budget, never both "
            "(ClientRequestSettings refuses the pair too); pass reasoning_effort or reasoning_max_tokens",
            provider=provider,
        )
    if effort is not None and effort not in EFFORT_LADDER:
        raise ModelsClientRefused(
            "reasoning_effort",
            f"{effort!r} is not one of the engine's words ({', '.join(EFFORT_LADDER)})",
            provider=provider,
        )
    if budget is None:
        return
    if provider == "anthropic" and budget < _ANTHROPIC_MIN_THINKING_BUDGET:
        raise ModelsClientRefused(
            "reasoning_max_tokens",
            f"Anthropic refuses a thinking budget below {_ANTHROPIC_MIN_THINKING_BUDGET} tokens; ask for at "
            f"least {_ANTHROPIC_MIN_THINKING_BUDGET}",
            provider=provider,
        )
    if max_tokens is not None and budget >= max_tokens:
        raise ModelsClientRefused(
            "reasoning_max_tokens",
            f"a reasoning budget ({budget}) that is not below max_tokens ({max_tokens}) leaves the answer no room; "
            "raise max_tokens or lower the budget",
            provider=provider,
        )
    if max_tokens is None and provider != "anthropic":
        raise ModelsClientRefused(
            "reasoning_max_tokens",
            f"{provider} takes an effort word and no reasoning budget, so a budget is sent as the word its share of "
            "max_tokens maps to, and there is no max_tokens to take a share of; pass max_tokens",
            provider=provider,
        )


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


def _error_message(body: object) -> str | None:
    """The provider's own message in an SDK error's ``body``: ``error.message``, else ``message``, else ``None``.

    Anthropic's SDK keeps the whole body (``{"type": "error", "error": {...}}``), OpenAI's the inner ``error``
    object; this reads either.
    """
    error = _mapping(body).get("error")
    message = _mapping(error if isinstance(error, Mapping) else body).get("message")
    return message if isinstance(message, str) and message else None


def _requires_parameters(bound: object, call_kwargs: Mapping[str, Any]) -> bool:  # Any: the bind's keyword values
    """Whether an OpenRouter call routes only to an upstream that takes every parameter it carries.

    ``require_parameters`` comes per call from models' structured-output translation (a ``json_schema`` directive),
    or from routing the host built the model with (``openrouter_provider``).

    Args:
        bound: The built ``ChatOpenRouter``.
        call_kwargs: The keywords the call is bound with.

    Returns:
        Whether the call carries ``provider.require_parameters``.
    """
    per_call = _mapping(call_kwargs.get("provider")).get("require_parameters")
    built = _mapping(getattr(bound, "openrouter_provider", None)).get("require_parameters")
    return per_call is True or built is True


@dataclass(frozen=True, slots=True)
class _Directive:
    """A call's reply-format directive, checked before anything is sent: its kind, and a schema's parts."""

    kind: Literal["json_schema", "json_object"] | None
    schema: dict[str, Any] | None = None  # Any: a JSON schema
    name: str = "response"
    strict: bool = False


@dataclass(frozen=True, slots=True)
class _Request:
    """One request as it goes out: the binding it is sent through, the bind keywords, and what it carried."""

    chat: RunnableBinding[Any, Any]  # Any: LangChain's types
    kwargs: dict[str, Any]  # Any: the provider's keyword values
    sent: WireSettings
    carried: _Carried


class ModelsCompletionClient:
    """The completion port over one chat model 3tears-models built: :func:`models_client` returns one.

    Implements :class:`~threetears.evals.schema.CompletionClient` and
    :class:`~threetears.evals.schema.BoundCompletionClient`. ``generate`` is safe to call concurrently: every call
    builds its own messages, its own settings and its own bind, and the only state it writes is the price book's
    and the learned facts' (each in one step), :attr:`last_pricing_failure`, and the copies of the chat model it
    keeps, one per distinct set of built settings.

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
        asked: WireSettings,
        prices: HttpClient | None,
    ) -> None:
        """Wrap a built chat model. Build one with :func:`models_client`, which checks every setting first.

        Args:
            model_name: The model id the chat model was built for.
            provider: Its provider.
            chat_model: The binding ``create_chat_model`` returned.
            asked: What the client was asked for (``strict_output`` aside, which each call's directive sets): what
                every call's settings are chosen from (:func:`closest_settings`). The chat model is copied at the
                settings each call sends, so it is built at the same ones (:func:`models_client` builds both).
            prices: The httpx-like client a price book of the client's own reads the catalogue through, or
                ``None`` for one that opens a client of its own per fetch; a factory passes its own
                :class:`PriceBook` (itself such a client), which every client it builds then shares, with the
                settings its clients learned from refusals.
        """
        self.model_name = model_name
        self.provider: SupportedProvider = provider
        self._chat = chat_model
        self._book = prices if isinstance(prices, PriceBook) else PriceBook(prices)
        bound = chat_model.bound
        self._responses_api = provider == "openai" and _routes_to_responses(bound)
        self._output_cap = _int_or_none(getattr(bound, "max_tokens", None))
        cap = asked.max_tokens if asked.max_tokens is not None else self._output_cap
        self._asked = replace(asked, max_tokens=cap, strict_output=False)
        self._registry = _registry_facts(model_name)
        self._variants: dict[str, RunnableBinding[Any, Any]] = {self._built_key(bound): chat_model}
        self._logged_notes: set[tuple[str, ...]] = set()
        self._closed = False
        self.last_pricing_failure: str | None = None
        """Why the most recent completion went unpriced, or ``None`` when it was priced.

        The same sentence as that completion's ``pricing_failure``, kept on the client so a host that reads only
        the engine's records (``cost_usd`` is ``None``) can still ask why. Under concurrent calls it is the last
        one to finish; read a completion's own ``pricing_failure`` when that matters."""
        self._request(_Directive(None), self._facts())  # logs how a plain call's settings moved, at build

    @property
    def chat_model(self) -> RunnableBinding[Any, Any]:  # Any: as in __init__
        """The LangChain runnable this client was built on: ``create_chat_model``'s binding, its model at ``.bound``.

        For a host that wants to see what the client was built with (``chat_model.bound.max_retries``,
        ``.temperature``, ``.max_tokens``) or to make calls of its own on the same model. A call whose settings
        moved since (a refusal learned, a catalogue read) goes through a copy at those settings; a call made on this
        one directly bypasses everything this client maps: the settings, the stop reason, the served model, the
        pricing.
        """
        return self._chat

    @property
    def temperature(self) -> float | None:
        """The temperature a call with no reply format is sent with now; ``None`` when none is sent.

        Read back from the model the call goes through, never echoed from the argument: LangChain drops a
        temperature a model refuses at build (OpenAI's ``gpt-5`` family), and again from the request on OpenAI's
        Responses API for a reasoning ``gpt-5`` model; on Anthropic none is sent beside thinking; and none is sent
        once the model is known not to take one (declared, or refused and the request without it served, by this
        client or another of its factory).
        """
        return self._carried_temperature(self._request(_Directive(None), self._facts()).chat)

    async def load_prices(self) -> str | None:
        """Look the model's rates up in the price catalogue, so :meth:`price_ceiling` can answer: the boot step.

        A host awaits it once after building the client, before the engine makes an out-of-run call on it under an
        enforced cap (``out_of_run_cap_usd``); a client a factory built shares the factory's book, so a model loaded
        once is loaded for every client of it. Rates found, and a catalogue that prices no row for the model, are
        kept; a failed fetch is not, so awaiting it again tries again. On OpenRouter the rates serve the ceiling
        alone: a call's stored cost stays OpenRouter's own reported figure; and the catalogue it read says which
        parameters the model takes, which every later call's settings follow.

        Returns:
            ``None`` when the rates are loaded, else why the model is unpriced (also logged at WARNING), in which
            case :meth:`price_ceiling` stays ``None`` and an out-of-run call under an enforced cap is refused.

        Raises:
            Exception: What a host's registered price source, or a host ``prices=`` client, raises beyond a failed
                fetch: a defect there surfaces at boot rather than as an unpriced model.
        """
        return await self._book.load(self.provider, self.model_name)

    def _facts(self, refusals: Sequence[_Refusal] = ()) -> SettingFacts:
        """What is known of the model now: learned over registered over catalogued, and this call's own refusals."""
        facts = self._book.learned.facts(self.provider, self.model_name).over(self._registry)
        if self.provider == "openrouter":
            facts = facts.over(_catalogue_facts(self._book.supported_parameters(self.model_name)))
        for refusal in refusals:
            facts = refusal.teach(facts)
        return facts

    def _directive(self, response_format: dict[str, Any] | None) -> _Directive:  # Any: a JSON directive
        """The call's directive, checked: a translatable kind, and a schema where its kind needs one.

        Raises:
            ModelsClientRefused: The directive is not one the adapter translates, or the model is on OpenAI's
                Responses API.
        """
        if response_format is None:
            return _Directive(None)
        if self._responses_api:
            raise ModelsClientRefused(
                "response_format",
                f"{self.model_name!r} is sent to OpenAI's Responses API, which takes a reply format as text.format, "
                "and models' structured-output translation writes Chat Completions' response_format, which that API "
                "refuses; use a model ChatOpenAI sends to Chat Completions (and no Responses-only keyword), or a "
                "client of your own",
                provider=self.provider,
            )
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
            return _Directive(
                "json_schema", schema, str(spec.get("name") or "response"), bool(spec.get("strict", True))
            )
        if kind == "json_object":
            return _Directive("json_object")
        raise ModelsClientRefused(
            "response_format",
            f"{kind!r} is not a directive the adapter translates; send a json_schema or a json_object one",
            provider=self.provider,
        )

    def _asks_strict(self, directive: _Directive) -> bool:
        """Whether the call asks for enforcement: a strict schema, and on OpenRouter any schema or built routing.

        On OpenRouter models' structured-output translation puts ``require_parameters`` on every schema call, and a
        host may have built the model with it (``openrouter_provider``).
        """
        if self.provider == "openrouter":
            built = _mapping(getattr(self._chat.bound, "openrouter_provider", None)).get("require_parameters")
            return directive.kind == "json_schema" or built is True
        return directive.kind == "json_schema" and directive.strict

    def _request(self, directive: _Directive, facts: SettingFacts) -> _Request:
        """The request one call sends under ``facts``: the binding at its settings, and its bind keywords.

        Logs, once per client and distinct outcome at INFO, how the settings moved from what was asked.
        """
        asked = replace(self._asked, strict_output=self._asks_strict(directive))
        sent, notes = closest_settings(asked, facts, self.provider)
        if notes and notes not in self._logged_notes:
            self._logged_notes.add(notes)
            log.info(
                "eval.providers sends the closest settings the model takes provider=%s model=%s: %s",
                self.provider,
                self.model_name,
                "; ".join(notes),
            )
        fields: dict[str, Any] = {"temperature": sent.temperature, "max_tokens": sent.max_tokens}  # Any: field values
        output_config: dict[str, Any] = {}  # Any: the provider's own output_config values
        kwargs: dict[str, Any] = {}  # Any: as above
        thinking = None
        if self.provider == "anthropic":
            thinking = _anthropic_thinking(sent, facts)
            fields["thinking"] = thinking
            if sent.reasoning_effort not in (None, "none"):
                output_config["effort"] = sent.reasoning_effort
        elif self.provider == "openai":
            fields["reasoning_effort"] = sent.reasoning_effort
        else:
            fields["reasoning"] = {"effort": sent.reasoning_effort} if sent.reasoning_effort is not None else None
        strict_format = False
        if directive.kind == "json_schema" and directive.schema is not None:
            if self.provider == "openrouter" or facts.structured_output is not False:
                built = structured_output_kwargs(
                    self.provider, directive.schema, name=directive.name, strict=directive.strict
                )
                if self.provider == "anthropic":
                    output_config.update(_mapping(built.pop("output_config", None)))
                kwargs.update(built)
                strict_format = directive.strict
            elif self.provider == "openai":
                # The model takes no schema: JSON mode, through extra_body as below, and the engine validates.
                kwargs["extra_body"] = {"response_format": {"type": "json_object"}}
            # Anthropic without structured outputs sends no format; the engine validates the reply on return.
        elif directive.kind == "json_object":
            if self.provider == "openai":
                # Through extra_body: a top-level response_format diverts ChatOpenAI onto its parse path.
                kwargs["extra_body"] = {"response_format": {"type": "json_object"}}
            elif self.provider == "openrouter":
                kwargs["response_format"] = {"type": "json_object"}
            # Anthropic has no JSON mode, and an empty-object schema would force `{}`: nothing is sent, and the
            # engine validates the reply on return as it does for every directive.
        if self.provider == "openrouter" and asked.strict_output and not sent.strict_output:
            # Merged into the model's routing by models' ChatOpenRouter, so it overrides a built require_parameters.
            kwargs["provider"] = {**_mapping(kwargs.get("provider")), "require_parameters": False}
        if output_config:
            kwargs["output_config"] = output_config
        chat = self._variant(fields)
        carried = _Carried(
            temperature=self._carried_temperature(chat) is not None,
            effort=sent.reasoning_effort,
            thinking=_mapping(thinking).get("type"),
            strict_format=strict_format and self.provider != "openrouter",
            enforcement=self.provider == "openrouter" and _requires_parameters(chat.bound, kwargs),
        )
        return _Request(chat=chat, kwargs=kwargs, sent=sent, carried=carried)

    @staticmethod
    def _built_key(bound: object) -> str:
        """The settings a built model holds, as the key its copies are kept under."""
        return json.dumps({name: getattr(bound, name, None) for name in _BUILT_FIELDS}, sort_keys=True, default=str)

    def _variant(self, fields: Mapping[str, Any]) -> RunnableBinding[Any, Any]:  # Any: LangChain's types
        """The chat model at ``fields``: the one built, or a copy of it made once and kept.

        The copy is shallow, so it shares the SDK client the original owns (OpenRouter's, which :meth:`aclose`
        closes), and the binding keeps its class and config, so models' tracker and the client's breaker still
        run. A temperature LangChain dropped at build is never put back on OpenAI: its validator dropped it for the
        model, and a copy does not run the validator again.
        """
        built = self._chat.bound
        update = {name: value for name, value in fields.items() if hasattr(built, name)}
        if update.get("max_tokens") is None:
            update.pop("max_tokens", None)  # no cap asked: the built model's default stands
        if self.provider == "openai" and update.get("temperature") is not None:
            update["temperature"] = getattr(built, "temperature", None)
        key = json.dumps(
            {name: update.get(name, getattr(built, name, None)) for name in _BUILT_FIELDS}, sort_keys=True, default=str
        )
        variant = self._variants.get(key)
        if variant is None:
            if not isinstance(built, BaseChatModel):
                raise TypeError(f"create_chat_model's binding holds {type(built).__name__}, not a LangChain chat model")
            variant = self._chat.model_copy(update={"bound": built.model_copy(update=update)})
            self._variants[key] = variant
        return variant

    def _carried_temperature(self, chat: RunnableBinding[Any, Any]) -> float | None:  # Any: as above
        """The temperature a request through ``chat`` carries, read back from its model."""
        bound = chat.bound
        temperature = getattr(bound, "temperature", None)
        built = float(temperature) if isinstance(temperature, int | float) else None
        return _responses_temperature(bound, built) if self._responses_api else built

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> ModelsCompletion:
        """Send the prompt pair with the closest settings the model takes, and read the reply into the port's members.

        **What is sent.** The settings the client was asked for, moved to the closest the model accepts as far as
        is known now (:func:`closest_settings`; the module docstring lists the sources), and the completion's
        ``sent_settings`` and ``temperature`` say what the served request carried.

        **A setting the model refuses is dropped or moved, once, and remembered.** When the provider refuses the
        request for a setting it carried (read by each SDK's error type and fields, as the module docstring lists),
        the request is sent again with the closest setting the model then accepts; once that is served, every
        client of the factory sends so for the rest of its life, logged once at WARNING with the provider's
        message. On OpenRouter, whose refusal (a 404 under ``require_parameters``) names nothing, the temperature
        goes first, and only if the request without it is refused too, enforcement (``strict_output=False``). This
        is not a hidden retry: the provider refused before generating anything, and a refusal is not billed, so
        the call costs what the one served request costs and :meth:`price_ceiling` holds. Any other failure, and a
        refusal of a resend for a setting already dropped, propagates as it was raised, and nothing is learned.

        Args:
            system: The system message.
            user: The one human message.
            response_format: ``None``; a ``json_schema`` directive, sent through models' structured-output
                translation for the provider (Anthropic's ``output_config.format``, OpenAI's
                ``extra_body.response_format``, OpenRouter's ``response_format`` with ``require_parameters``, so
                an upstream that cannot honour it is refused rather than answering in prose), or, on a model that
                takes no strict schema, unenforced (JSON mode on OpenAI, no format on Anthropic, no
                ``require_parameters`` on OpenRouter); or a ``json_object`` one, sent as OpenAI and OpenRouter take
                it and not at all on Anthropic, which has no JSON mode. Either way the content is the reply's raw
                text. Refused on a model ``ChatOpenAI`` sends to OpenAI's Responses API, which takes neither in that
                shape.

        Returns:
            The completion. Its price never costs it: a pricing failure after the call returns it unpriced.

        Raises:
            ModelsClientRefused: ``response_format`` is not a directive the adapter translates, or the model is on
                OpenAI's Responses API.
            TypeError: The chat model returned something other than an ``AIMessage``.
            Exception: The provider SDK's own error for a refused or failed request, as it was raised (for a
                refused setting, the resend's, when that fails too).
        """
        directive = self._directive(response_format)
        messages: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=user)]
        refusals: list[_Refusal] = []
        request = self._request(directive, self._facts())
        while True:
            try:
                reply = await self._send(request.chat, request.kwargs, messages)
                break
            except (anthropic.BadRequestError, openai.BadRequestError, openrouter_errors.NotFoundResponseError) as exc:
                refusal = _refusal(self.provider, exc, request.carried, self._facts(refusals))
                if (
                    refusal is None
                    or len(refusals) >= _MAX_RESENDS
                    or any((r.setting, r.word) == (refusal.setting, refusal.word) for r in refusals)
                ):
                    raise
                resend = self._request(directive, self._facts([*refusals, refusal]))
                if resend.chat is request.chat and resend.kwargs == request.kwargs:
                    raise  # nothing closer to send: the refusal stands
                refusals.append(refusal)
                request = resend
        if refusals:
            self._book.learned.learn(
                self.provider, self.model_name, refusals, registry=self._registry, sent=request.sent
            )
        return await self._read(reply, self._carried_temperature(request.chat), request.sent.reported())

    async def _send(
        self,
        chat: RunnableBinding[Any, Any],  # Any: as in __init__
        kwargs: Mapping[str, Any],  # Any: the bind's keyword values
        messages: list[BaseMessage],
    ) -> AIMessage:
        """One request: ``chat`` bound with the call's keywords, given the prompt pair.

        Args:
            chat: The binding to send through: the client's, or its copy at the call's settings.
            kwargs: The call's bind keywords (:meth:`_request`).
            messages: The system and human messages.

        Returns:
            The reply.

        Raises:
            TypeError: The chat model returned something other than an ``AIMessage``.
        """
        runnable = chat.bind(**kwargs) if kwargs else chat
        reply = await runnable.ainvoke(messages)
        if not isinstance(reply, AIMessage):
            raise TypeError(f"{self.provider} chat model returned {type(reply).__name__}, not an AIMessage")
        return reply

    async def _read(
        self, reply: AIMessage, temperature: float | None, sent: ClientRequestSettings | None
    ) -> ModelsCompletion:
        """The reply's members, per provider; ``temperature`` and ``sent`` are what the served request carried."""
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
            temperature=temperature,
            stop_reason=stop_reason,
            provider_stop_reason=provider_stop_reason,
            served_provider=served_provider if isinstance(served_provider, str) and served_provider else None,
            pricing_failure=unpriced,
            sent_settings=sent,
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
        the model was built with, reasoning included (the cap a call sends is never above it). One served request
        per call, since the adapter retries nothing: a resend it makes with a setting the model refused dropped or
        moved (:meth:`generate`) follows a refusal that was not billed, so the ceiling is the same with or without
        it.

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

    **Reasoning, per provider, as the closest setting the model takes** (:func:`closest_settings`; unknown is sent as
    asked, and a refusal is learned from). ``reasoning_effort`` is one of the engine's seven words, and
    ``reasoning_max_tokens`` a budget in tokens:

    * Anthropic: a word goes as ``output_config.effort`` over adaptive thinking (``thinking={"type": "adaptive"}``;
      the effort alone on a model that takes no adaptive thinking), merged into the same ``output_config`` as a
      schema's format on each call; ``minimal`` is sent as ``low``, its lowest; ``none`` builds thinking disabled
      (``low`` on a model that cannot turn thinking off). A budget goes as ``thinking={"type": "enabled",
      "budget_tokens": N}``, at least 1024 and below ``max_tokens``; on a model that takes no budget, as the effort
      word its share of ``max_tokens`` maps to (:data:`BUDGET_SHARES`), and a word on a model that takes only budgets
      as that share of the cap. No temperature is sent beside any thinking (the API refuses one), so completions
      report ``temperature=None`` there.
    * OpenAI: ``reasoning_effort`` as the word; ``max`` is sent as ``xhigh``, OpenAI's highest, and a budget as the
      word its share of ``max_tokens`` maps to: OpenAI takes no reasoning budget.
    * OpenRouter: ``reasoning={"effort": word}``, all seven words (narrowed only by what is known of the model); a
      budget as the word its share maps to, because the OpenRouter SDK's typed request strips
      ``reasoning.max_tokens`` without an error. OpenRouter maps a word back onto a budget for an upstream that
      takes one by the same shares, so a budget comes back about as asked.

    Asking by a word and by a budget at once is refused, as
    :class:`~threetears.evals.schema.ClientRequestSettings` refuses it; so are a budget below 1024 on Anthropic, a
    budget not below ``max_tokens``, and a budget off Anthropic with no ``max_tokens`` to take its share of. What a
    request carried is each completion's ``sent_settings`` (``None`` when it carried no output cap).

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
            completion reports is what its request carried (:attr:`ModelsCompletionClient.temperature`): ``None``
            beside Anthropic thinking, on a model known to take none, and once the provider refused a request for
            it, which is sent again without it, with none sent after (:meth:`ModelsCompletionClient.generate`).
        reasoning_effort: The reasoning level, one of the engine's seven words, or ``None`` to send none.
        reasoning_max_tokens: The reasoning token budget, or ``None`` to send none; sent as a budget where the
            model takes one, else as the effort word its share of ``max_tokens`` maps to.
        max_tokens: The output cap, or ``None`` for the provider's default (Anthropic's is the model profile's;
            OpenAI and OpenRouter send none, and their completions then report no ``sent_settings``); above the
            model's known output cap, that cap. The price ceiling needs one.
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
        ModelsClientRefused: The provider, the model or the key's route cannot be built for, a provider keyword
            would bypass the adapter, or a reasoning setting is the caller's error (above).
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
    _refuse_caller_errors(resolved, reasoning_effort, reasoning_max_tokens, max_tokens)
    asked = WireSettings(
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        reasoning_max_tokens=reasoning_max_tokens,
    )
    # Built at the settings a call with no reply format sends, as far as is known before the model exists; a call
    # whose settings differ (a cap known only once built, a refusal learned) goes through a copy at its own.
    facts = _registry_facts(model_id)
    if isinstance(prices, PriceBook):
        facts = prices.learned.facts(resolved, model_id).over(facts)
    planned, _ = closest_settings(asked, facts, resolved)
    build: dict[str, Any] = {}  # Any: create_chat_model's keyword values
    if planned.temperature is not None:
        build["temperature"] = planned.temperature
    if planned.max_tokens is not None:
        build["max_tokens"] = planned.max_tokens
    if resolved == "anthropic" and (thinking := _anthropic_thinking(planned, facts)) is not None:
        build["thinking"] = thinking
    elif resolved == "openai" and planned.reasoning_effort is not None:
        build["reasoning_effort"] = planned.reasoning_effort
    elif resolved == "openrouter" and planned.reasoning_effort is not None:
        build["reasoning"] = {"effort": planned.reasoning_effort}
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
    if (
        max_tokens is None
        and reasoning_max_tokens is not None
        and built_cap is not None
        and reasoning_max_tokens >= built_cap
    ):
        # max_tokens was not given, so the cap is the model profile's default, known only once built.
        raise ModelsClientRefused(
            "reasoning_max_tokens",
            f"a reasoning budget ({reasoning_max_tokens}) that is not below the model's default output cap "
            f"({built_cap}) leaves the answer no room; pass a max_tokens above the budget",
            provider=resolved,
        )
    return ModelsCompletionClient(model_name=model_id, provider=resolved, chat_model=chat, asked=asked, prices=prices)
