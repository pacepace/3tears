"""The completion client: :class:`Model` for one model, :class:`Clients` for a host's roles, and :class:`Completion`.

A :class:`Model` is built from a model id and speaks to its provider over HTTP. Each call sends the closest request
settings the model takes and answers a refusal of one with a resend (:func:`~threetears.evals.negotiation.negotiate`),
reads the reply into a :class:`Completion`, and prices it. Given ``recording=``, it records its calls into a JSONL
file and replays them offline through :class:`~threetears.evals.testing.RecordedCompletions`.
"""

from __future__ import annotations

import json
import os
import string
import time
import unicodedata
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Self, get_args

import httpx
from pydantic import BaseModel

from threetears.evals.kernel import beside_eval_file, note_spend, strict_response_format
from threetears.evals.kernel.host import CompletionRole
from threetears.evals.negotiation import (
    ANTHROPIC_MIN_THINKING_BUDGET,
    EFFORT_LADDER,
    SettingFacts,
    WireSettings,
    facts_from_router_catalogue,
    negotiate,
)
from threetears.evals.providers.catalogue import Catalogue, PriceRefresh
from threetears.evals.providers.failures import Provider, ProviderError, error_from_response
from threetears.evals.providers.wire import (
    BASE_URLS,
    FAMILIES,
    KEY_VARIABLES,
    PriceSource,
    Reply,
    declared_facts,
    directive_of,
    read_reply,
    reported_cost,
    request_body,
)
from threetears.evals.run import JUDGE_REQUEST_SETTINGS, SIMULATOR_REQUEST_SETTINGS
from threetears.evals.schema import BoundCompletionClient, ClientRequestSettings, CompletionResult, StopReason
from threetears.evals.schema.logs import get_logger

if TYPE_CHECKING:
    from threetears.evals.testing import RecordedCompletions

__all__ = ["DEFAULT_MAX_TOKENS", "Clients", "Completion", "Model"]

log = get_logger(__name__)

#: The output cap a :class:`Model` sends when given none: Anthropic requires one, and a price ceiling needs one.
DEFAULT_MAX_TOKENS: Final = 8192

#: How long one request may take, in seconds, on a client the model opens itself, unless ``timeout=`` says
#: otherwise. Nothing is retried.
_REQUEST_TIMEOUT: Final = httpx.Timeout(600.0, connect=20.0)

#: The tokens a price ceiling allows for a prompt's framing beyond its text (role markers, a schema's re-encoding).
_FRAMING_TOKENS: Final = 2048

#: What a catalogue-priced cost's source gains when a cache read or write was priced at a rate the catalogue row does
#: not list for it: the plain prompt rate, or the dearer of two write rates the reply does not split between.
_CACHE_APPROXIMATE: Final = ";cache-approximate"

#: When this module was imported. Under ``RECORD=1`` a recording last written before it is from an earlier run and
#: is started afresh; one written since is this run's, and is added to.
_IMPORTED_AT: Final = time.time()

#: The engine's roles (:data:`~threetears.evals.kernel.host.CompletionRole`).
_ROLES: Final = frozenset(get_args(CompletionRole))

#: The request settings the engine records as asked of a role, which :class:`Clients` builds that role from when
#: given a model id alone.
_ASKED: Final[Mapping[str, ClientRequestSettings]] = {
    "judge": JUDGE_REQUEST_SETTINGS,
    "simulator": SIMULATOR_REQUEST_SETTINGS,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class Completion:
    """One reply: the completion port's ten members, and five of the client's own.

    Implements :class:`~threetears.evals.schema.CompletionResult`. Every count and the cost are ``None`` where the
    provider reported nothing, never ``0``.

    Attributes:
        content: The reply's text, never a thinking block; under a schema, the raw JSON the model wrote.
        input_tokens: Prompt tokens, cache reads and writes included.
        output_tokens: Completion tokens, reasoning included.
        reasoning_tokens: The reasoning share of ``output_tokens``, or ``None`` when unsplit.
        cost_usd: What the call cost, or ``None`` when unpriced (``pricing_failure`` says why).
        price_source: Who priced it, in :class:`~threetears.evals.providers.PriceSource`'s words: ``openrouter:reported``
            (OpenRouter's figure), ``openrouter:reported+upstream`` (its fee and a bring-your-own-key upstream's
            charge), or ``openrouter-catalogue:<id>`` (the reply's tokens priced by the catalogue row, ending
            ``;cache-approximate`` when a cache share was priced at a rate the row does not list for it).
        model: The model id the client was built for.
        served_model: The model the reply named; ``None`` when it named none, or named a requested alias back.
        temperature: The temperature the served request carried, or ``None``.
        stop_reason: Why it stopped, in the engine's words: an absent or unknown reason is ``truncated``.
        sent_settings: What the served request carried: its cap, its reasoning word or budget, ``strict_output``
            exactly when it was enforced, and ``unsent_reasoning_max_tokens`` when a budget asked went as a word.
        provider_stop_reason: The stop reason as the provider named it, or ``None``.
        served_provider: The upstream OpenRouter routed the call to; ``None`` on a direct provider.
        pricing_failure: Why ``cost_usd`` is ``None``; ``None`` when priced.
        pricing_stale_since: When the catalogue rates that priced the call expired with every refresh since failing
            (UTC): the cost is the last rates read, kept rather than failing the call. ``None`` when the rates were
            fresh, and for a cost the provider reported (OpenRouter's), which no catalogue priced.
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
    sent_settings: ClientRequestSettings | None
    provider_stop_reason: str | None
    served_provider: str | None
    pricing_failure: str | None
    pricing_stale_since: datetime | None


#: What :meth:`Model.refresh_prices` answers while replaying: no catalogue is read, and none needs to be.
_NOTHING_READ: Final = PriceRefresh(refreshed=False, fetched_at=None, failure=None)


@dataclass(slots=True, eq=False)
class _MadeRead:
    """A catalogue read ``refresh_prices`` made, which the model's next pricing use takes once.

    Mutable on purpose: a task started after the refresh inherits the same cell through the context, so whichever
    task prices first takes it, and every other use, in any task, reads for itself.
    """

    read: PriceRefresh
    taken: bool = False

    def take(self) -> PriceRefresh | None:
        """The read, the first time it is asked for; ``None`` every time after."""
        if self.taken:
            return None
        self.taken = True
        return self.read


def _provider_of(model_id: str, provider: Provider | None, base_url: str | None) -> Provider:
    """The provider a model id goes to: as named; else OpenRouter for a ``vendor/model`` id, an OpenAI-compatible
    endpoint for a ``base_url``; a bare id with neither is refused, never guessed."""
    if provider is not None:
        if provider not in KEY_VARIABLES:
            raise ValueError(f"provider {provider!r} is not one of {', '.join(KEY_VARIABLES)}")
        return provider
    if base_url is not None:
        return "openai"
    if "/" in model_id:
        return "openrouter"
    raise ValueError(
        f"{model_id!r} names no provider: pass provider='anthropic' or provider='openai' (or base_url= for an "
        "OpenAI-compatible endpoint), or a vendor/model id, which goes to OpenRouter"
    )


def _asked(
    provider: Provider, max_tokens: int | None, effort: str | int | None, temperature: float | None
) -> WireSettings:
    """What the model is asked for, checked: the caller's own errors are refused by name."""
    cap = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError(f"max_tokens {max_tokens!r} is not a positive number of tokens")
    if isinstance(effort, bool) or (effort is not None and not isinstance(effort, str | int)):
        raise ValueError(f"effort {effort!r} is neither an effort word nor a token budget")
    if isinstance(effort, str) and effort not in EFFORT_LADDER:
        raise ValueError(f"effort {effort!r} is not one of {', '.join(EFFORT_LADDER)}, nor a token budget")
    if isinstance(effort, int):
        if effort >= cap:
            raise ValueError(f"a reasoning budget ({effort}) not below max_tokens ({cap}) leaves the answer no room")
        if provider == "anthropic" and effort < ANTHROPIC_MIN_THINKING_BUDGET:
            raise ValueError(f"Anthropic refuses a thinking budget below {ANTHROPIC_MIN_THINKING_BUDGET} tokens")
    return WireSettings(
        max_tokens=cap,
        temperature=temperature,
        reasoning_effort=effort if isinstance(effort, str) else None,
        reasoning_max_tokens=effort if isinstance(effort, int) else None,
    )


def _text_bytes(text: str) -> int:
    """The most tokens a tokenizer can read ``text`` as: one per UTF-8 byte, as written or after NFKC normalisation,
    whichever is more (a normalising tokenizer can expand a compatibility character: ``U+FDFA`` is 3 bytes, 33 after).
    """
    return max(
        len(text.encode("utf-8", errors="surrogatepass")),
        len(unicodedata.normalize("NFKC", text).encode("utf-8", errors="surrogatepass")),
    )


def _prompt_bound(system: str, user: str, response_format: Mapping[str, Any] | None) -> int:  # Any: a JSON directive
    """A bound on the prompt tokens a call sends: no token is shorter than a byte, so one per byte of each text (and
    of the directive, written out as JSON either way), and an allowance for the framing around them."""
    bound = _text_bytes(system) + _text_bytes(user) + _FRAMING_TOKENS
    if response_format is not None:
        written = json.dumps(response_format, ensure_ascii=False, default=str)
        bound += max(_text_bytes(written), len(json.dumps(response_format, default=str)))
    return bound


class _Live:
    """The model over HTTP: what :class:`Model` sends through when it is not replaying a recording."""

    def __init__(
        self,
        model_name: str,
        provider: Provider,
        *,
        key: str,
        base_url: str | None,
        asked: WireSettings,
        catalogue: Catalogue,
        http: httpx.AsyncClient | None,
        transport: httpx.AsyncBaseTransport | None,
        timeout: float | httpx.Timeout | None,
        headers: Mapping[str, str] | None,
    ) -> None:
        self.model_name = model_name
        self.provider: Provider = provider
        self._key = key
        self._url = (base_url or BASE_URLS[provider]).rstrip("/")
        self._direct_openai = provider == "openai" and base_url is None
        self._asked = asked
        self._catalogue = catalogue
        self._http = http
        self._transport = transport
        self._timeout = _REQUEST_TIMEOUT if timeout is None else timeout
        self._headers = dict(headers) if headers is not None else None
        self._opened: httpx.AsyncClient | None = None
        self._logged: set[tuple[str, ...]] = set()
        self._unbounded: set[str] = set()
        #: What the last :meth:`refresh_prices` found, or ``None`` before the first.
        self.last_refresh: PriceRefresh | None = None
        #: The catalogue read this model's last ``refresh_prices`` in this context made, which its next pricing use (a
        #: call, or a price ceiling asked without one) takes once, in whichever task comes first; every other use
        #: reads for itself. The model's own, so no other model, and no other host, ever takes it.
        self._made_read: ContextVar[_MadeRead | None] = ContextVar(
            f"threetears_evals_read_for_call_{id(self):x}", default=None
        )

    def _client(self) -> httpx.AsyncClient:
        if self._http is not None:
            return self._http
        if self._opened is None:
            self._opened = httpx.AsyncClient(timeout=self._timeout, headers=self._headers, transport=self._transport)
        return self._opened

    def _take_made_read(self) -> PriceRefresh | None:
        """The read ``refresh_prices`` made, if no pricing use has taken it yet; taking it spends it for every task."""
        made = self._made_read.get()
        return None if made is None else made.take()

    async def _read_for_call(self) -> PriceRefresh:
        """The catalogue read this call makes: the one ``refresh_prices`` made, when nothing took it first, or a read
        of its own (:meth:`Catalogue.read`)."""
        made = self._take_made_read()
        if made is not None:
            return made
        return await self._catalogue.read()

    def _catalogue_id(self) -> str | None:
        """The catalogue's id for this model: its own on OpenRouter, the matching row's elsewhere."""
        if self.provider == "openrouter":
            return self.model_name
        vendor = None if not self._direct_openai and self.provider == "openai" else self.provider
        return self._catalogue.match(self.model_name, vendor=vendor)

    def _declared(self) -> SettingFacts:
        facts = declared_facts(self.model_name)
        if self.provider != "openrouter":
            return facts
        listed = facts_from_router_catalogue(self._catalogue.supported_parameters(self.model_name))
        cap = SettingFacts(max_output_tokens=self._catalogue.max_output_tokens(self.model_name))
        return facts.over(listed).over(cap)

    def _json_mode(self) -> bool:
        listed = self._catalogue.supported_parameters(self.model_name)
        return listed is None or "response_format" in listed

    def _log_moves(self, notes: tuple[str, ...]) -> None:
        if notes and notes not in self._logged:
            self._logged.add(notes)
            log.info(
                "eval.providers sends the closest settings the model takes provider=%s model=%s: %s",
                self.provider,
                self.model_name,
                "; ".join(notes),
            )

    async def _post(self, body: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:  # Any: JSON
        if self.provider == "anthropic":
            path, headers = "/messages", {"x-api-key": self._key, "anthropic-version": "2023-06-01"}
        else:
            path, headers = "/chat/completions", {"authorization": f"Bearer {self._key}"}
        response = await self._client().post(self._url + path, json=body, headers=headers)
        if not response.is_success:
            raise error_from_response(self.provider, response.status_code, response.content)
        try:
            parsed = response.json()
        except ValueError as unreadable:
            raise ProviderError(self.provider, response.status_code, "the reply is not JSON") from unreadable
        if not isinstance(parsed, Mapping):
            raise ProviderError(self.provider, response.status_code, "the reply is not a JSON object")
        return response.status_code, parsed

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> Completion:  # Any: a JSON directive
        directive = directive_of(response_format)
        read = await self._read_for_call()
        declared = self._declared()
        asked = replace(self._asked, strict_output=directive.asks_strict(self.provider))
        refused: list[tuple[dict[str, Any], ProviderError]] = []  # Any: a JSON body

        async def send(wire: WireSettings, facts: SettingFacts) -> tuple[int, Mapping[str, Any]]:  # Any: JSON
            body = request_body(
                self.provider,
                self.model_name,
                system=system,
                user=user,
                sent=wire,
                facts=facts,
                directive=directive,
                direct_openai=self._direct_openai,
                json_mode=self._json_mode(),
            )
            if refused and refused[-1][0] == body:
                raise refused[-1][1]  # nothing closer to send: the refusal stands, as it came
            try:
                return await self._post(body)
            except ProviderError as failure:
                refused.append((body, failure))
                raise

        negotiated = await negotiate(
            asked,
            provider_family=FAMILIES[self.provider],
            provider=self.provider,
            model=self.model_name,
            send=send,
            read_refusal=lambda exc: exc.refusal() if isinstance(exc, ProviderError) else None,
            learned=self._catalogue.learned,
            declared=declared,
        )
        self._log_moves(negotiated.notes)
        status, body = negotiated.reply
        reply = read_reply(self.provider, self.model_name, status, body)
        cost, source, unpriced, stale_since = self._price(reply, read.failure)
        if reply.stop_reason == "truncated":
            log.warning(
                "eval.providers reply read as truncated provider=%s model=%s provider_stop_reason=%s",
                self.provider,
                self.model_name,
                reply.provider_stop_reason or "none",
            )
        return Completion(
            content=reply.content,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            reasoning_tokens=reply.reasoning_tokens,
            cost_usd=cost,
            price_source=source,
            model=self.model_name,
            served_model=reply.served_model,
            temperature=negotiated.temperature,
            stop_reason=reply.stop_reason,
            sent_settings=negotiated.sent_settings,
            provider_stop_reason=reply.provider_stop_reason,
            served_provider=reply.served_provider,
            pricing_failure=unpriced,
            pricing_stale_since=stale_since,
        )

    def _price(self, reply: Reply, unread: str | None) -> tuple[float | None, str | None, str | None, datetime | None]:
        """The call's cost, its source, why it is unpriced, and when the catalogue rates pricing it went stale;
        ``unread`` is why the call's catalogue read failed, which unprices it while the catalogue holds no rows."""
        if self.provider == "openrouter":
            return (*reported_cost(reply.usage), None)
        if not self._catalogue.loaded:
            return None, None, unread or "the price catalogue has not been read", None
        matched = self._catalogue_id()
        rates = self._catalogue.rates(matched) if matched is not None else None
        if matched is None or rates is None:
            return None, None, f"no single priced row of OpenRouter's catalogue matches {self.model_name!r}", None
        if reply.input_tokens is None or reply.output_tokens is None:
            return None, None, "the reply reported no input or output token count, so there is nothing to price", None
        dollars, approximate, why = self._catalogue.cost(
            matched,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            reasoning_tokens=reply.reasoning_tokens,
            cache_read_tokens=reply.cache_read_tokens,
            cache_write_tokens=reply.cache_write_tokens,
            cache_write_1h_tokens=reply.cache_write_1h_tokens,
        )
        if dollars is None:
            return None, None, why, None
        marker = _CACHE_APPROXIMATE if approximate else ""
        source = f"{PriceSource.OPENROUTER_CATALOGUE}:{matched}{marker}"
        return float(dollars), source, None, self._catalogue.stale_since

    async def refresh_prices(self) -> PriceRefresh:
        read = await self._catalogue.read()
        self.last_refresh = read
        self._made_read.set(_MadeRead(read))
        return read

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        self._take_made_read()  # an estimate is a pricing use: the refresh it followed is spent on it
        matched = self._catalogue_id()
        if self._asked.max_tokens is None:
            self._unbounded_because("the request has no output cap")
            return None
        if matched is None:
            if not self._catalogue.loaded:
                self._unbounded_because(
                    "the price catalogue has not been read (await Catalogue.load() or Clients.load_prices())"
                )
                return None
            self._unbounded_because(f"no single row of OpenRouter's catalogue matches {self.model_name!r}")
            return None
        ceiling, why = self._catalogue.ceiling(
            matched,
            prompt_tokens=_prompt_bound(system, user, response_format),
            output_tokens=self._asked.max_tokens,
        )
        if ceiling is None:
            self._unbounded_because(why or "the catalogue gave no reason")
            return None
        return float(ceiling)

    def _unbounded_because(self, why: str) -> None:
        """No ceiling, logged at WARNING once per reason: an enforced cap holds such a call against nothing."""
        if why not in self._unbounded:
            self._unbounded.add(why)
            log.warning(
                "eval.providers price_ceiling cannot bound a call provider=%s model=%s: %s",
                self.provider,
                self.model_name,
                why,
            )
        return None

    async def aclose(self) -> None:
        if self._opened is not None:
            opened, self._opened = self._opened, None
            await opened.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


class Model:
    """One model, as the engine's completion client and as three helpers for a runner of your own.

    ``Model("deepseek/deepseek-v4-flash")`` goes to OpenRouter on ``OPENROUTER_API_KEY``: an id with a ``/`` is
    OpenRouter's. A bare id names its provider, ``provider="anthropic"`` (``ANTHROPIC_API_KEY``) or
    ``provider="openai"`` (``OPENAI_API_KEY``), never guessed; ``base_url=`` takes any OpenAI-compatible endpoint
    (DeepSeek direct, Gemini's compatibility endpoint, vLLM). A missing key is refused when the model is built,
    naming the variable, unless a recording will serve every call.

    Implements :class:`~threetears.evals.schema.CompletionClient` and
    :class:`~threetears.evals.schema.BoundCompletionClient`; ``generate`` is safe to call concurrently. Nothing is
    retried: the only resend answers a provider's refusal of a setting the request carried, which the provider did
    not bill, with the closest setting the model then takes, learned for every model sharing the
    :class:`~threetears.evals.providers.Catalogue`.

    **Recording.** Given ``recording=``, a file that exists is replayed, offline and free, and ``RECORD=1`` in the
    environment records it again live (started afresh by the first model built on it in the run). A file that does
    not exist yet is recorded live. A replay serves only calls recorded under the same model and request settings,
    and refuses any other by name (:class:`~threetears.evals.testing.RecordingRefused`).

    **Spend.** A call made inside an eval's runner call, live or replayed, is added to that call's spend
    (:func:`~threetears.evals.kernel.note_spend`), so a runner that calls a model reports what it cost with no
    ``Answer``.

    **HTTP.** A model sends through a client it opens itself on its first live call, set up by ``transport=``,
    ``timeout=`` and ``headers=``, or through one handed in with ``http=``. :meth:`aclose` closes the one it opened
    (and with it, as ``httpx`` does, the transport it was opened on), and one handed in only with ``own_http=True``.

    Attributes:
        model_name: The model id, which every completion is attributed to.
        provider: The provider it goes to.
        recording: The recording's path, or ``None``.
    """

    def __init__(
        self,
        id: str,
        *,
        provider: Provider | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        effort: str | int | None = None,
        temperature: float | None = None,
        recording: str | os.PathLike[str] | None = None,
        catalogue: Catalogue | None = None,
        http: httpx.AsyncClient | None = None,
        own_http: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float | httpx.Timeout | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Build the client; nothing is sent until the first call.

        Args:
            id: The model id: ``vendor/model`` for OpenRouter, the provider's own id otherwise.
            provider: ``"openrouter"``, ``"anthropic"`` or ``"openai"``; required for a bare id with no ``base_url``.
            api_key: The key, or ``None`` to read the provider's variable.
            base_url: An OpenAI-compatible endpoint's root (``https://api.deepseek.com/v1``), or ``None``.
            max_tokens: The output cap, reasoning included; ``None`` sends :data:`DEFAULT_MAX_TOKENS`. Above the
                model's own cap (as the catalogue or a refusal says), that cap.
            effort: Reasoning, as an effort word (``none`` to ``max``) or a token budget, sent as the family takes
                it: Anthropic's ``output_config.effort`` over adaptive thinking or a thinking budget
                (``thinking.budget_tokens``), OpenAI's ``reasoning_effort``, OpenRouter's ``reasoning.effort`` or
                ``reasoning.max_tokens``. A budget where only words are taken (OpenAI's chat completions, a model
                that refused a budget) goes as the word its share of ``max_tokens`` maps to, and the completion's
                ``sent_settings.unsent_reasoning_max_tokens`` names it. ``None`` sends none.
            temperature: The sampling temperature, or ``None`` for the model's own. Never sent to a Claude 5.5
                model, nor beside Anthropic thinking; each completion says what was sent.
            recording: A JSONL recording to replay, or to record into; a relative path is read beside the eval
                file being loaded (``python -m threetears.evals FILE``), else in the working directory.
            catalogue: The price catalogue to share with other models, or ``None`` for one of its own.
            http: An ``httpx.AsyncClient`` to send through, used as it is built, or ``None`` to open one.
            own_http: Whether the model owns ``http``, so :meth:`aclose` closes it; ``False`` leaves it the caller's.
                Refused without ``http``.
            transport: The ``httpx`` transport of the client the model opens itself (HTTP/2, a proxy, a test's mock),
                or ``None`` for ``httpx``'s own. Closing the model closes it with that client, as ``httpx`` does, so
                a transport shared between models is closed by the first one closed. Refused beside ``http``.
            timeout: The timeout of the client the model opens itself (seconds, or an ``httpx.Timeout``); ``None``
                is 600 seconds a request, 20 to connect. Refused beside ``http``, whose own timeout holds.
            headers: Headers every request on the client the model opens itself carries; ``None`` for none. The
                provider's key and version headers, which each request sets, win over one of the same name. Refused
                beside ``http``, whose own headers hold.

        Raises:
            ValueError: A bare id with no provider, an unknown provider, no key for a live model (naming the
                variable), ``transport``, ``timeout`` or ``headers`` beside ``http``, ``own_http`` without it, or a
                setting that is the caller's error (an effort that is neither a word nor a budget, a budget not below
                ``max_tokens``, an Anthropic budget below 1024).
        """
        if http is not None and (transport is not None or timeout is not None or headers is not None):
            raise ValueError(
                "transport=, timeout= and headers= set up the client the model opens itself, and http= hands it one: "
                "set them on that client instead"
            )
        if own_http and http is None:
            raise ValueError("own_http=True says the model closes the client handed in with http=, and none was")
        self.model_name = id
        self.provider: Provider = _provider_of(id, provider, base_url)
        self.recording = beside_eval_file(recording) if recording is not None else None
        self._options: dict[str, Any] = {  # Any: the constructor's keyword values, for derive()
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "max_tokens": max_tokens,
            "effort": effort,
            "recording": self.recording,
            "http": http,
            "transport": transport,
            "timeout": timeout,
            "headers": headers,
        }
        #: The client handed in that :meth:`aclose` closes, live or replaying: ``http`` when the model owns it.
        self._owned_http = http if own_http else None
        asked = _asked(self.provider, max_tokens, effort, temperature)
        self._settings = ClientRequestSettings(
            max_tokens=asked.max_tokens or DEFAULT_MAX_TOKENS,
            reasoning_effort=asked.reasoning_effort,
            reasoning_max_tokens=asked.reasoning_max_tokens,
        )
        rerecording = os.environ.get("RECORD") == "1"
        replaying = self.recording is not None and self.recording.exists() and not rerecording
        self._client: _Live | RecordedCompletions
        self._live: _Live | None = None
        self._last_price_refresh: PriceRefresh | None = None
        if replaying:
            from threetears.evals.testing import RecordedCompletions, RecordingRefused

            assert self.recording is not None
            replay = RecordedCompletions.replay(self.recording, settings=self._settings)
            # A recording holds one model's calls, so one of another model's is no answer for this one.
            if replay.model_name != id:
                raise RecordingRefused(
                    self.recording,
                    f"it holds calls of model {replay.model_name!r}, and this client is {id!r}",
                    f"record {id!r} into a file of its own",
                )
            self._client = replay
            return
        variable = KEY_VARIABLES[self.provider]
        key = api_key or os.environ.get(variable)
        if not key:
            where = f"; or record {self.recording} once with it set" if self.recording is not None else ""
            raise ValueError(f"no key for {self.provider}: pass api_key= or set {variable}{where}")
        live = _Live(
            id,
            self.provider,
            key=key,
            base_url=base_url,
            asked=asked,
            catalogue=catalogue if catalogue is not None else Catalogue(),
            http=http,
            transport=transport,
            timeout=timeout,
            headers=headers,
        )
        self._live = live
        if self.recording is None:
            self._client = live
            return
        from threetears.evals.testing import RecordedCompletions

        stale = rerecording and self.recording.exists() and self.recording.stat().st_mtime < _IMPORTED_AT
        self._client = RecordedCompletions.record(live, self.recording, settings=self._settings, replace=stale)

    def derive(self, id: str | None = None, *, temperature: float | None, catalogue: Catalogue | None = None) -> Model:
        """This model's settings for another model id or temperature: how :class:`Clients` builds a role's client.

        The new model sends through this one's ``http`` client but never owns it: closing it leaves the client open,
        and the model that owns it (this one, with ``own_http=True``) closes it. Given a ``transport``, it opens a
        client of its own on that same transport, which closing it closes.

        Args:
            id: The model id, or ``None`` for this one.
            temperature: The temperature, or ``None`` for the model's own.
            catalogue: The catalogue to share, or ``None`` for one of its own.

        Returns:
            A new model, built as this one was in every other setting.
        """
        return Model(id or self.model_name, temperature=temperature, catalogue=catalogue, **self._options)

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> CompletionResult:  # Any: a JSON directive
        """Send a prompt pair, or serve it from the recording: the completion port's one call.

        Args:
            system: The system message; an empty one is not sent.
            user: The user message.
            response_format: ``None``, a ``json_object`` directive, or a strict ``json_schema`` one, which goes as
                each provider enforces it (and as JSON mode, or no format on Anthropic, where it cannot be).

        Returns:
            The completion: a :class:`Completion` live, a replayed one from a recording.

        Raises:
            ProviderError: The provider refused or failed the request (a refused setting is resent first).
            httpx.HTTPError: The request could not be sent, or timed out.
            ValueError: ``response_format`` is not a directive the client sends.
            RecordingRefused: The recording holds no answer for the call.
        """
        completion = await self._client.generate(system=system, user=user, response_format=response_format)
        # Inside an eval's runner call, what the call spent is the runner's, reported with no Answer.
        note_spend(completion)
        return completion

    async def text(self, user: str, *, system: str | None = None) -> str:
        """The reply to ``user``, as text.

        Args:
            user: The user message.
            system: A system message, or ``None``.

        Returns:
            The reply's text.

        Raises:
            ValueError: The reply did not finish (cut short, filtered, or an error), named with its stop reason.
        """
        return _finished(await self.generate(system=system or "", user=user)).content

    async def json[T: BaseModel](self, user: str, schema: type[T], *, system: str | None = None) -> T:
        """The reply to ``user`` as ``schema``, sent as a strict ``json_schema`` and validated on return.

        Args:
            user: The user message.
            schema: The pydantic model the reply must be.
            system: A system message, or ``None``.

        Returns:
            The validated reply.

        Raises:
            ValueError: The reply did not finish, named with its stop reason.
            pydantic.ValidationError: The reply is not a ``schema``: pydantic's message says why. Never a default.
        """
        directive = strict_response_format(schema.__name__, schema.model_json_schema())
        completion = await self.generate(system=system or "", user=user, response_format=directive)
        return schema.model_validate_json(_finished(completion).content)

    def prompt[T: BaseModel](
        self, template: str, schema: type[T] | None = None, *, system: str | None = None
    ) -> Callable[[Mapping[str, Any]], Awaitable[str | T]]:  # Any: a case's values
        """A runner: ``template`` filled from each case, answered as text or as ``schema``.

        ``{field}`` is the case's ``field``; ``{{`` and ``}}`` are literal braces. A field the case lacks is refused
        by name when the runner is called, and a placeholder that is not a plain name when the template is read.

        Args:
            template: The user message, with ``{field}`` placeholders.
            schema: The pydantic model each answer must be, or ``None`` for text.
            system: A system message, or ``None``.

        Returns:
            ``async (case) -> answer``, whose docstring is the template's first line.

        Raises:
            ValueError: A placeholder that is not a plain name (``{0}``, ``{a.b}``, ``{a!r}``), or an unbalanced brace.
        """
        fields: list[str] = []
        for _, name, spec, conversion in string.Formatter().parse(template):
            if name is None:
                continue
            if not name.isidentifier() or spec or conversion:
                raise ValueError(f"the prompt's placeholder {{{name}}} is not a plain field name like {{email}}")
            fields.append(name)

        async def run(case: Mapping[str, Any]) -> str | T:  # Any: a case's values
            if missing := [name for name in fields if name not in case]:
                has = ", ".join(map(str, case))
                raise KeyError(f"the prompt names {', '.join(missing)}, which the case lacks (it has {has})")
            user = template.format_map({name: case[name] for name in fields})
            if schema is None:
                return await self.text(user, system=system)
            return await self.json(user, schema, system=system)

        run.__doc__ = next((line.strip() for line in template.splitlines() if line.strip()), None)
        return run

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """The most one :meth:`generate` with this prompt pair can cost, or ``None`` when the client cannot say.

        A bound on the prompt (one token per UTF-8 byte of each text, as written or after NFKC normalisation,
        whichever is more, and an allowance for framing) and the output cap, priced by every price the catalogue row
        lists (:meth:`Catalogue.ceiling`): the highest rate a prompt or output token can bill at, a per-request fee, a
        long-context tier the prompt bound reaches. ``None``, never a lower figure, until the catalogue is loaded
        (await :meth:`Catalogue.load` on the one handed in, or :meth:`Clients.load_prices`), once its rows are
        ``Catalogue.ceiling_stale_after`` old, for a model it does not price, and for a row listing a price the client
        cannot bound; each reason is logged at WARNING once. The engine awaits :meth:`refresh_prices` before it prices
        a call, so rows go that old only when a refresh failed, not because the model sat idle. ``0.0`` while
        replaying.

        Args:
            system: The system message the call would send.
            user: The user message the call would send.
            response_format: The directive the call would send, or ``None``.

        Returns:
            The ceiling in dollars, or ``None``.
        """
        return self._client.price_ceiling(system=system, user=user, response_format=response_format)

    @property
    def last_price_refresh(self) -> PriceRefresh | None:
        """What the last :meth:`refresh_prices` found, or ``None`` before the first; a call's own read leaves it."""
        return self._last_price_refresh

    async def refresh_prices(self) -> PriceRefresh:
        """Read the catalogue, or refresh rows past ``refresh_after``, as a live call does before it is sent.

        What the engine awaits before it prices a call (:func:`~threetears.evals.kernel.capped_completion.refresh_prices`),
        so a model idle past ``Catalogue.ceiling_stale_after`` is priced from fresh rows rather than refused. Loads
        made together share one fetch. The model's next pricing use takes this read as its own, once, whichever task
        makes it: a :meth:`generate` rather than reading again (so a read that failed is not tried again by the call
        it was made for), or a :meth:`price_ceiling`, which spends it. Every other use reads for itself, so calls
        made together after one refresh do not all lean on it, and an estimate's refresh is not left for later
        work. Never raises: a first read that fails leaves :meth:`price_ceiling` ``None``
        and the call unpriced, and a refresh that fails keeps the rows, stale, as a live call does; either is logged.
        Free while replaying.

        Returns:
            What the read found (:class:`~threetears.evals.providers.PriceRefresh`), also kept as
            :attr:`last_price_refresh`: whether it fetched the rows anew, when the rows held were fetched, and why the
            prices are not current. While replaying, nothing read and no failure.
        """
        await self._client.refresh_prices()  # recording, the recorder forwards it to the live client
        read = _NOTHING_READ if self._live is None else self._live.last_refresh
        assert read is not None, "a live client's refresh_prices records what it read"
        self._last_price_refresh = read
        return read

    async def aclose(self) -> None:
        """Close the HTTP client the model opened, and the one handed in when ``own_http=True``; either only once."""
        owned, self._owned_http = self._owned_http, None
        try:
            await self._client.aclose()
        finally:
            if owned is not None:
                await owned.aclose()

    async def __aenter__(self) -> Self:
        """Enter a scope that closes the model on exit."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close the model, whether or not the body raised."""
        await self.aclose()


def _finished(completion: CompletionResult) -> CompletionResult:
    """``completion`` when it finished; else refused naming why it stopped."""
    if completion.stop_reason != "end_turn":
        raw = getattr(completion, "provider_stop_reason", None)
        said = f" ({raw})" if raw else ""
        raise ValueError(f"the reply did not finish: it stopped at {completion.stop_reason}{said}")
    return completion


class _Lease:
    """A role's client as the engine is handed it: the factory's model, with an ``aclose`` that releases nothing."""

    def __init__(self, model: Model) -> None:
        self._model = model
        self.model_name = model.model_name

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> CompletionResult:  # Any: a JSON directive
        return await self._model.generate(system=system, user=user, response_format=response_format)

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        return self._model.price_ceiling(system=system, user=user, response_format=response_format)

    async def refresh_prices(self) -> PriceRefresh:
        return await self._model.refresh_prices()

    async def aclose(self) -> None:
        """Release nothing: the factory closes its models in :meth:`Clients.aclose`."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Leave the scope; the factory's model stays open."""


class Clients:
    """A host's role factory (``EvalHost.clients``): one :class:`Model` per role, model id and temperature.

    ``Clients(default="deepseek/deepseek-v4-flash", judge="google/gemini-3.5-flash-lite")``. Each role is a model id
    or a :class:`Model` whose settings that role's models are built with; a role not named takes ``default``. Every
    model shares one :class:`~threetears.evals.providers.Catalogue`, so the catalogue is fetched once and a refusal
    one model met is known to all. A ``judge`` or ``simulator`` given as a model id is asked as the run records it
    asked (:data:`~threetears.evals.run.JUDGE_REQUEST_SETTINGS`,
    :data:`~threetears.evals.run.SIMULATOR_REQUEST_SETTINGS`): its output cap and its reasoning, the judge's budget
    sent as a budget wherever the provider takes one.

    The engine closes what it is handed after each unit of work; that releases nothing here, and :meth:`aclose`
    closes every model once, at shutdown, and then the ``http`` client when the factory owns it (``own_http=True``).
    A role given as a :class:`Model` stays the caller's: the models built from it send through its ``http`` client
    without closing it (:meth:`Model.derive`), and open their own on its ``transport``. Await :meth:`load_prices` at
    boot so every price ceiling can answer.
    """

    def __init__(
        self,
        default: str | Model | None = None,
        *,
        catalogue: Catalogue | None = None,
        http: httpx.AsyncClient | None = None,
        own_http: bool = False,
        **roles: str | Model,
    ) -> None:
        """Hold the configuration; nothing is built and no key is read until a role is asked for.

        Args:
            default: The model a role named nowhere else gets, or ``None``.
            catalogue: The catalogue every model shares, or ``None`` for one of the factory's own.
            http: The ``httpx.AsyncClient`` every model given by id sends through, never closing it, or ``None``
                for each to open its own.
            own_http: Whether the factory owns ``http``, so :meth:`aclose` closes it after every model; ``False``
                leaves it the caller's. Refused without ``http``.
            **roles: Role (``judge``, ``simulator``, ``analysis``, ``variation``, ``proposer``) -> its model.

        Raises:
            ValueError: A keyword that is not one of the engine's roles, or ``own_http`` without ``http``.
        """
        if own_http and http is None:
            raise ValueError("own_http=True says the factory closes the client handed in with http=, and none was")
        if unknown := sorted(set(roles) - _ROLES):
            roles_text = ", ".join(sorted(_ROLES))
            raise ValueError(f"{unknown[0]!r} is not a role the engine asks a client for ({roles_text})")
        self._default = default
        self._roles = dict(roles)
        self.catalogue = catalogue if catalogue is not None else Catalogue()
        self._http = http
        self._owned_http = http if own_http else None
        self._models: dict[tuple[str, str, float | None], Model] = {}

    def __call__(
        self, role: CompletionRole, model: str | None, *, temperature: float | None = None
    ) -> BoundCompletionClient:
        """A client for ``role`` bound to ``model``, built on first use and kept.

        Args:
            role: The role it serves.
            model: The model id, or ``None`` for the role's own.
            temperature: The temperature, or ``None`` for the model's own.

        Returns:
            A lease on the factory's model; closing it releases nothing.

        Raises:
            ValueError: No model was named and the role has none, or the model cannot be built (no key, no provider).
        """
        template = self._roles.get(role, self._default)
        named = template.model_name if isinstance(template, Model) else template
        chosen = model if model is not None else named
        if chosen is None:
            raise ValueError(f"the engine named no model for {role!r}, and Clients has none for it nor a default")
        key = (role, chosen, temperature)
        built = self._models.get(key)
        if built is None:
            if isinstance(template, Model):
                built = template.derive(chosen, temperature=temperature, catalogue=self.catalogue)
            else:
                asked = _ASKED.get(role)
                effort = None if asked is None else (asked.reasoning_effort or asked.reasoning_max_tokens)
                cap = None if asked is None else asked.max_tokens
                built = Model(
                    chosen,
                    max_tokens=cap,
                    effort=effort,
                    temperature=temperature,
                    catalogue=self.catalogue,
                    http=self._http,
                )
            self._models[key] = built
        return _Lease(built)

    async def refresh_prices(self) -> PriceRefresh:
        """Read the one catalogue every model shares, once, or refresh rows past ``refresh_after``: one outcome for
        every role, since each prices from the same rows. Never raises; a first read that fails is logged.

        Returns:
            What the read found (:class:`~threetears.evals.providers.PriceRefresh`).
        """
        return await self.catalogue.read()

    async def load_prices(self) -> dict[str, str]:
        """Read the catalogue, so every model's price ceiling can answer: the boot step.

        Returns:
            Model id -> why it is unpriced, for each model named in the factory the catalogue cannot price; empty
            when every one is priced.
        """
        named = {
            template if isinstance(template, str) else template.model_name: template
            for template in (self._default, *self._roles.values())
            if template is not None
        }
        read = await self.catalogue.read()
        if not self.catalogue.loaded:
            return dict.fromkeys(sorted(named), read.failure or "the price catalogue has not been read")
        unpriced: dict[str, str] = {}
        for model_id, template in sorted(named.items()):
            provider = template.provider if isinstance(template, Model) else ("openrouter" if "/" in model_id else None)
            if provider == "openrouter":
                matched: str | None = model_id
            else:
                matched = self.catalogue.match(model_id, vendor=provider)
            if matched is None or self.catalogue.rates(matched) is None:
                unpriced[model_id] = f"no single priced row of OpenRouter's catalogue matches {model_id!r}"
        return unpriced

    async def aclose(self) -> None:
        """Close every model the factory built, and forget them, then the ``http`` client it owns; a failure to close
        one is raised once all are tried.

        Raises:
            ExceptionGroup: One or more models, or the client, failed to close.
        """
        models, self._models = list(self._models.values()), {}
        owned, self._owned_http = self._owned_http, None
        failures: list[Exception] = []
        for model in models:
            try:
                await model.aclose()
            except Exception as failure:  # collected, and raised below once every other model is closed
                failures.append(failure)
        if owned is not None:
            try:
                await owned.aclose()
            except Exception as failure:  # collected with the models' and raised below
                failures.append(failure)
        if failures:
            closed = f"{len(models)} models" + (" and the http client" if owned is not None else "")
            raise ExceptionGroup(f"{len(failures)} of {closed} failed to close", failures)
