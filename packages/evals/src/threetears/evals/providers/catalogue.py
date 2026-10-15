"""OpenRouter's public model catalogue, read and refreshed: rates, the parameters each model takes, and what refusals
taught.

:class:`Catalogue` is the one price source the client has. OpenRouter's ``GET /api/v1/models`` needs no key and lists,
per model, its per-token rates, the ``supported_parameters`` some endpoint of it takes, and its output cap. It prices
Anthropic and OpenAI models too (``anthropic/claude-haiku-5.5``), so a call made to either directly is priced by the
same rows. Every :class:`~threetears.evals.providers.Model` handed one shares it, with the request settings a provider
refused, which the catalogue holds for every model that reads it (:attr:`Catalogue.learned`). There is no
process-wide catalogue: a ``Model`` given none builds its own, fetched on its first live call.

**Rates go stale, so a long-lived catalogue refreshes.** A first fetch that fails is not kept, so the next asks again.
Once read, the rows are kept for ``refresh_after`` (:data:`DEFAULT_REFRESH_AFTER` unless the constructor says
otherwise), and the first :meth:`Catalogue.load` after that, which every live call makes, fetches them again. A
refresh that fails keeps the last rows, logs the failure, and marks the catalogue stale
(:attr:`Catalogue.stale_since`, :attr:`Catalogue.refresh_failure`), which every completion it then prices carries
(``Completion.pricing_stale_since``): never silent, and never failing the call. The failed refresh is tried again on
the first load :data:`RETRY_AFTER_FAILURE` later (sooner when ``refresh_after`` is shorter), not on every call.
Callers loading at once share one fetch, and its result or its error. One request reads the catalogue once: a
model's call takes the read its ``refresh_prices()`` made for it in the same task as its own, so a read that failed,
a first read included, is not tried again by the call it was made for.

**A price ceiling is a true upper bound, or none.** :meth:`Catalogue.ceiling` bounds one call from every price the
row lists, not only its base rates: a prompt token at the highest of the plain, cache-read and cache-write rates, an
output token at the highest of the completion, reasoning and output-image or -audio rates (every output token is
within the request's output cap), the per-request fee, and a long-context tier (``overrides``) wherever the prompt
bound reaches its ``min_prompt_tokens``; a time-of-day tier is taken to apply, since a call's billing time is not
known before it is made. Image and audio input prices apply to inputs the client never sends, and a web-search price
to searches it never asks for. A completion is priced from the same reading (:meth:`Catalogue.cost`), at the tier
its prompt fell in: its reasoning tokens at the reasoning rate where the row lists one apart from the completion
rate, its cache reads and writes at their rates, and the request fee; a price that applies and that the reply's
counts cannot price (reasoning tokens the reply does not split out) leaves it unpriced, never priced lower. A price the client cannot bound (one it does not know the unit of, one that is not a
number, a web search on a model that can search unasked) answers no ceiling and says why, never a lower one; so do
rows read longer ago than ``ceiling_stale_after`` (``refresh_after`` unless the constructor says otherwise), so an
enforced cap never holds a call against prices that old. A completion is still priced from them, marked stale.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Final

import httpx

from threetears.evals.negotiation import LearnedSettings
from threetears.evals.schema.logs import get_logger

__all__ = [
    "CATALOGUE_FETCH_TIMEOUT_S",
    "DEFAULT_REFRESH_AFTER",
    "OPENROUTER_MODELS_URL",
    "RETRY_AFTER_FAILURE",
    "Catalogue",
    "PriceRefresh",
]

log = get_logger(__name__)

#: How long a catalogue keeps the rows it read before the next load fetches them again, unless built with another.
DEFAULT_REFRESH_AFTER: Final = timedelta(hours=6)

#: How long after a failed refresh the next load tries again (or ``refresh_after``, when that is shorter): a provider
#: outage costs one failed fetch per this interval, not one per call.
RETRY_AFTER_FAILURE: Final = timedelta(minutes=5)

#: OpenRouter's public model catalogue.
OPENROUTER_MODELS_URL: Final = "https://openrouter.ai/api/v1/models"

#: How long the catalogue fetch may take, in seconds, on a client the catalogue opens itself. A host handing the
#: catalogue its own client (``Catalogue(http=...)``) sets that client's timeout, and matches this one with it.
CATALOGUE_FETCH_TIMEOUT_S: Final = 30.0

#: The per-token prices a prompt token can be billed at: the plain rate, and a cache read or write of it.
_INPUT_PRICES: Final = frozenset({"prompt", "input_cache_read", "input_cache_write", "input_cache_write_1h"})

#: The per-token prices an output token can be billed at. Every one is within the request's output cap: reasoning,
#: and an image or audio the model writes, count as output tokens.
_OUTPUT_PRICES: Final = frozenset({"completion", "internal_reasoning", "image_output", "audio_output"})

#: A flat fee per request.
_REQUEST_FEE: Final = "request"

#: Prices of inputs the client never sends (it sends text): an image, audio, and a cache of audio.
_UNSENT_INPUTS: Final = frozenset({"image", "audio", "input_audio_cache"})

#: The price of one web search: billed only for a search, which the client never asks for (no ``plugins``, no
#: ``web_search_options``). A model that takes ``web_search_options`` may search unasked, any number of times.
_WEB_SEARCH: Final = "web_search"

#: A price tier's conditions, not prices: the prompt size it starts at, and the hours and days it applies on.
_TIER_CONDITIONS: Final = frozenset({"min_prompt_tokens", "utc_start", "utc_end", "utc_days"})

#: The row's price tiers.
_TIERS: Final = "overrides"

#: A date suffix on a direct provider's model id (``claude-haiku-4-5-20251001``), which the catalogue's ids leave off.
_DATE_SUFFIX: Final = re.compile(r"-\d{8}$")


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceRefresh:
    """What one read of the price catalogue found: what ``Model.refresh_prices()`` and ``Clients.refresh_prices()``
    return, and ``Model.last_price_refresh`` keeps.

    A host writes its own pricing-outage line from it: a ``failure`` with no ``fetched_at`` is a first read that
    failed, so no model can be priced; one with a ``fetched_at`` is a refresh that failed, the rows read then kept.

    Attributes:
        refreshed: Whether the read fetched the rows anew (by itself, or by a load in flight it shared); ``False``
            when the rows held were young enough to keep, when the fetch failed, and while replaying.
        fetched_at: When the rows the catalogue now holds were fetched (UTC), or ``None`` while it holds none.
        failure: Why the prices are not current, naming the error's class and message: the first read failed
            (``fetched_at`` is ``None``), or the last refresh did and the rows held are stale
            (:attr:`Catalogue.refresh_failure`). ``None`` when they are current, and while replaying.
    """

    refreshed: bool
    fetched_at: datetime | None
    failure: str | None


@dataclass(frozen=True, slots=True)
class _Row:
    """One catalogue row: the rates per token (``None`` when unpriced), the listed parameters, the output cap, and
    every price as listed."""

    rates: tuple[Decimal, Decimal] | None
    parameters: frozenset[str] | None
    max_output_tokens: int | None
    pricing: Mapping[str, object]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Usage:
    """What a reply reports it was billed for, as :meth:`Catalogue.cost` takes it.

    Attributes:
        input_tokens: Prompt tokens, cache reads and writes included.
        output_tokens: Completion tokens, reasoning included.
        reasoning_tokens: The reasoning share of ``output_tokens``, or ``None`` when the reply does not split it.
        cache_read_tokens: Prompt tokens read from a cache.
        cache_write_tokens: Prompt tokens written to a cache, both lifetimes.
        cache_write_1h_tokens: The one-hour share of ``cache_write_tokens``, or ``None`` when the reply does not
            split it.
    """

    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None = None
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int | None = 0


def _rate(pricing: Mapping[str, object], key: str) -> Decimal | None:
    """A per-token rate the catalogue states as a decimal string, or ``None`` when it states none or no number."""
    value = pricing.get(key)
    if not isinstance(value, str | int | float) or isinstance(value, bool):
        return None
    try:
        rate = Decimal(str(value))
    except InvalidOperation:
        # NOSILENT: a rate that is not a number prices nothing; the row reads as unpriced, which every caller reports.
        return None
    return rate if rate.is_finite() and rate >= 0 else None


def _row(entry: Mapping[str, object]) -> _Row:
    pricing = entry.get("pricing")
    prompt = _rate(pricing, "prompt") if isinstance(pricing, Mapping) else None
    completion = _rate(pricing, "completion") if isinstance(pricing, Mapping) else None
    listed = entry.get("supported_parameters")
    top = entry.get("top_provider")
    cap = top.get("max_completion_tokens") if isinstance(top, Mapping) else None
    return _Row(
        rates=(prompt, completion) if prompt is not None and completion is not None else None,
        parameters=frozenset(p for p in listed if isinstance(p, str)) if isinstance(listed, list) else None,
        max_output_tokens=cap if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0 else None,
        pricing=dict(pricing) if isinstance(pricing, Mapping) else {},
    )


def _tier_applies(tier: Mapping[str, object], prompt_tokens: int) -> bool:
    """Whether a price tier can apply to a call of up to ``prompt_tokens``: always, unless the prompt bound is below a
    ``min_prompt_tokens`` it can read. A time-of-day tier can apply, since when a call is billed is not known."""
    threshold = tier.get("min_prompt_tokens")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float | str):
        return True
    try:
        return prompt_tokens >= Decimal(str(threshold))
    except InvalidOperation:
        # NOSILENT: a threshold that is not a number is taken to apply, which can only raise the ceiling.
        return True


def _prices(row: _Row, prompt_tokens: int) -> tuple[dict[str, Decimal] | None, str | None]:
    """Every nonzero price that can bill a call of ``prompt_tokens`` under ``row``, each at the highest the row and the
    tiers that apply to that prompt list it, or ``None`` and why when a price cannot be read or bounded.

    The one reading of a row both the ceiling and a completion's cost are priced from, so the two cannot disagree on
    what a row charges.
    """
    tiers = row.pricing.get(_TIERS, [])
    if not isinstance(tiers, list) or not all(isinstance(tier, Mapping) for tier in tiers):
        return None, f"its price tiers (`{_TIERS}`) are not a list of objects"
    prices: dict[str, Decimal] = {}
    for tier in (row.pricing, *(tier for tier in tiers if _tier_applies(tier, prompt_tokens))):
        for key, value in tier.items():
            if key in _TIER_CONDITIONS or key == _TIERS:
                continue
            rate = _rate(tier, key)
            if rate is None:
                return None, f"its `{key}` price ({value!r}) is not a price (a number, 0 or more)"
            if rate == 0 or key in _UNSENT_INPUTS:
                continue
            if key == _WEB_SEARCH:
                if row.parameters is not None and "web_search_options" in row.parameters:
                    return None, (
                        f"it can search the web unasked (it takes web_search_options) at {value} a search, and "
                        "nothing says how many searches one call makes"
                    )
                continue
            if key not in _INPUT_PRICES | _OUTPUT_PRICES | {_REQUEST_FEE}:
                return None, f"it lists a `{key}` price ({value}) the client does not know the unit of"
            prices[key] = max(prices.get(key, rate), rate)
    return prices, None


def _ceiling(row: _Row, *, prompt_tokens: int, output_tokens: int) -> tuple[Decimal | None, str | None]:
    """The most one call can be billed under ``row``, or ``None`` and why when a price it lists cannot be bounded."""
    prices, why = _prices(row, prompt_tokens)
    if prices is None:
        return None, why
    input_rate = max((prices.get(key, Decimal(0)) for key in _INPUT_PRICES), default=Decimal(0))
    output_rate = max((prices.get(key, Decimal(0)) for key in _OUTPUT_PRICES), default=Decimal(0))
    return prompt_tokens * input_rate + output_tokens * output_rate + prices.get(_REQUEST_FEE, Decimal(0)), None


def _cost(prices: Mapping[str, Decimal], usage: _Usage) -> tuple[Decimal | None, bool, str | None]:
    """What a call cost under ``prices`` (the tier its prompt fell in), whether a cache share was priced at a rate the
    row does not list for it, or ``None`` and why when a price applies that the reply's counts cannot price."""
    plain = prices.get("prompt", Decimal(0))
    approximate = False

    def cache_rate(*keys: str) -> Decimal:
        nonlocal approximate
        listed = [prices[key] for key in keys if key in prices]
        if len(listed) != 1:  # none listed: the plain rate; more than one, unsplit: the dearest. Either is a guess.
            approximate = True
        return max(listed, default=plain)

    one_hour = usage.cache_write_1h_tokens
    cached = usage.cache_read_tokens + usage.cache_write_tokens
    fresh = usage.input_tokens - cached
    if fresh < 0:
        return (
            None,
            False,
            (f"the reply counts {cached} cached prompt tokens within {usage.input_tokens} prompt tokens in all"),
        )
    dollars = fresh * plain
    if usage.cache_read_tokens:
        dollars += usage.cache_read_tokens * cache_rate("input_cache_read")
    if usage.cache_write_tokens:
        if one_hour is None:
            dollars += usage.cache_write_tokens * cache_rate("input_cache_write", "input_cache_write_1h")
        else:
            if usage.cache_write_tokens - one_hour:
                dollars += (usage.cache_write_tokens - one_hour) * cache_rate("input_cache_write")
            if one_hour:
                dollars += one_hour * cache_rate("input_cache_write_1h")
    completion = prices.get("completion", Decimal(0))
    reasoning = prices.get("internal_reasoning")
    if reasoning is not None and reasoning != completion and usage.output_tokens:
        if usage.reasoning_tokens is None:
            return (
                None,
                False,
                (
                    f"it prices reasoning tokens at {reasoning} and other output at {completion}, and the reply did not "
                    "say how many of its output tokens were reasoning"
                ),
            )
        dollars += usage.reasoning_tokens * reasoning + (usage.output_tokens - usage.reasoning_tokens) * completion
    else:
        dollars += usage.output_tokens * completion
    return dollars + prices.get(_REQUEST_FEE, Decimal(0)), approximate, None


def _plain(model_id: str) -> str:
    """A model id as the catalogue's would match it: lower case, dots as dashes, no date suffix."""
    return _DATE_SUFFIX.sub("", model_id.lower().replace(".", "-"))


class Catalogue:
    """OpenRouter's model catalogue, refreshed every ``refresh_after``, and what providers refused, for every model
    that reads it.

    Hand one catalogue to several :class:`~threetears.evals.providers.Model`\\s (``Clients`` does) and they fetch it
    together and learn from each other's refusals. Await :meth:`load` at boot when a model's
    :meth:`~threetears.evals.providers.Model.price_ceiling` must answer before its first call; a live call loads it
    otherwise, and refreshes it once its rows are older than ``refresh_after``. A price ceiling answers only from rows
    younger than ``ceiling_stale_after``; the engine awaits a model's
    :meth:`~threetears.evals.providers.Model.refresh_prices`, which loads, before it prices a call, so rows reach
    that age only when a refresh failed. ``ceiling_stale_after`` below ``refresh_after`` is refused: rows past it
    would not be due a refresh, and no call could be priced.

    Attributes:
        url: Where the catalogue is read from.
        refresh_after: How long the rows read are kept before the next load fetches them again.
        ceiling_stale_after: How old the rows may be for :meth:`ceiling` to answer from them.
        learned: The request settings providers refused, per provider and model, for every model reading this
            catalogue (:class:`~threetears.evals.negotiation.LearnedSettings`).
    """

    def __init__(
        self,
        *,
        url: str = OPENROUTER_MODELS_URL,
        http: httpx.AsyncClient | None = None,
        refresh_after: timedelta = DEFAULT_REFRESH_AFTER,
        ceiling_stale_after: timedelta | None = None,
    ) -> None:
        """Know nothing yet: nothing is fetched until :meth:`load`.

        Args:
            url: The catalogue's URL.
            http: The client to fetch it with (not closed here), or ``None`` to open one for the fetch, with a
                timeout of :data:`CATALOGUE_FETCH_TIMEOUT_S`. A client handed in keeps its own settings.
            refresh_after: How long the rows read are kept before the next load fetches them again; zero refreshes
                on every load.
            ceiling_stale_after: How old the rows may be for :meth:`ceiling` to answer from them; older, it answers
                no ceiling. ``None`` is ``refresh_after``; below it is refused.

        Raises:
            ValueError: ``refresh_after`` is negative, or ``ceiling_stale_after`` is below it.
        """
        if refresh_after < timedelta(0):
            raise ValueError(
                f"refresh_after {refresh_after} is negative: a catalogue keeps its rows for no time or more"
            )
        if ceiling_stale_after is not None and ceiling_stale_after < refresh_after:
            raise ValueError(
                f"ceiling_stale_after {ceiling_stale_after} is below refresh_after {refresh_after}: rows past it would "
                "not be due a refresh, so no call could be priced from them"
            )
        self.url = url
        self.refresh_after = refresh_after
        self.ceiling_stale_after = refresh_after if ceiling_stale_after is None else ceiling_stale_after
        self._http = http
        self._rows: dict[str, _Row] | None = None
        #: When the kept rows were read (UTC, and as a monotonic instant), and the monotonic instant the next load
        #: fetches again.
        self._read_at: datetime | None = None
        self._read_monotonic = 0.0
        self._due = 0.0
        self._stale_since: datetime | None = None
        self._refresh_failure: str | None = None
        #: The load in flight, which every caller arriving while it runs awaits rather than fetching again.
        self._in_flight: asyncio.Task[None] | None = None
        self.learned = LearnedSettings(
            client="eval.providers",
            declared_by="catalogue",
            contradiction="OpenRouter's catalogue, or a fact the client declares, which the provider contradicts",
        )

    @property
    def loaded(self) -> bool:
        """Whether the catalogue has been read (its rows may since have gone stale: :attr:`stale_since`)."""
        return self._rows is not None

    @property
    def stale_since(self) -> datetime | None:
        """When the kept rows expired with every refresh since failing (UTC), or ``None`` while they are fresh.

        Every completion priced from them carries it as ``pricing_stale_since``.
        """
        return self._stale_since

    @property
    def fetched_at(self) -> datetime | None:
        """When the rows held were fetched (UTC), or ``None`` while the catalogue holds none."""
        return self._read_at

    async def read(self) -> PriceRefresh:
        """:meth:`load`, saying what it found rather than raising: whether it fetched the rows anew, when the rows
        held were fetched, and why the prices are not current. A first read that fails is logged at WARNING.

        Returns:
            The outcome (:class:`PriceRefresh`).
        """
        before = self._read_at
        try:
            await self.load()
        except (httpx.HTTPError, ValueError) as failed:
            failure = f"the price catalogue could not be read ({type(failed).__name__}: {failed})"
            log.warning("eval.providers catalogue unread url=%s: %s", self.url, failure)
            return PriceRefresh(refreshed=False, fetched_at=self._read_at, failure=failure)
        fetched = self._read_at is not None and self._read_at != before
        return PriceRefresh(refreshed=fetched, fetched_at=self._read_at, failure=self._refresh_failure)

    @property
    def refresh_failure(self) -> str | None:
        """Why the last refresh failed, while the rows are stale; ``None`` while they are fresh."""
        return self._refresh_failure

    async def load(self) -> None:
        """Fetch and read the catalogue unless the rows kept are younger than ``refresh_after``.

        A first read that fails raises and keeps nothing, so the next load fetches again. A refresh that fails keeps
        the rows read before, logs the failure and marks them stale (:attr:`stale_since`, :attr:`refresh_failure`)
        rather than raising, and the next load :data:`RETRY_AFTER_FAILURE` later (or ``refresh_after``, when that is
        shorter) tries again; one that succeeds clears the mark. Loads made while one is in flight share it: one
        fetch, whose result or error each of them sees. A caller cancelled while waiting cancels only its own wait.

        Raises:
            httpx.HTTPError: The first fetch failed or answered with a failure status.
            ValueError: The first fetch's body is not JSON, or holds no ``data`` list.
        """
        if self._rows is not None and time.monotonic() < self._due:
            return
        if self._in_flight is None:
            self._in_flight = asyncio.create_task(self._load_once())
        await asyncio.shield(self._in_flight)

    async def _load_once(self) -> None:
        """The one load every caller arriving while it runs awaits; it clears itself when it ends."""
        try:
            if self._rows is not None and time.monotonic() < self._due:
                return
            try:
                rows = await self._fetch()
            except (httpx.HTTPError, ValueError) as failed:
                if self._rows is None or self._read_at is None:
                    raise
                self._refresh_failure = (
                    f"the price catalogue could not be refreshed ({type(failed).__name__}: {failed})"
                )
                if self._stale_since is None:
                    self._stale_since = self._read_at + self.refresh_after
                self._due = time.monotonic() + min(self.refresh_after, RETRY_AFTER_FAILURE).total_seconds()
                log.warning(
                    "eval.providers catalogue refresh failed url=%s stale_since=%s; pricing from the rows read at %s: %s",
                    self.url,
                    self._stale_since.isoformat(),
                    self._read_at.isoformat(),
                    self._refresh_failure,
                )
                return
            self._rows = rows
            self._read_at = datetime.now(UTC)
            self._read_monotonic = time.monotonic()
            self._due = self._read_monotonic + self.refresh_after.total_seconds()
            self._stale_since = None
            self._refresh_failure = None
        finally:
            self._in_flight = None

    async def _fetch(self) -> dict[str, _Row]:
        """One fetch of the catalogue, read into rows.

        Raises:
            httpx.HTTPError: The fetch failed or answered with a failure status.
            ValueError: The body is not JSON, or holds no ``data`` list.
        """
        if self._http is not None:
            response = await self._http.get(self.url)
        else:
            async with httpx.AsyncClient(timeout=CATALOGUE_FETCH_TIMEOUT_S, follow_redirects=True) as owned:
                response = await owned.get(self.url)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            raise ValueError(f"the catalogue at {self.url} holds no `data` list")
        return {
            entry["id"]: _row(entry)
            for entry in rows
            if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)
        }

    def rates(self, model_id: str) -> tuple[Decimal, Decimal] | None:
        """The model's input and output rates in dollars per token, or ``None`` (not loaded, not listed, unpriced).

        Args:
            model_id: The catalogue's id for the model (:meth:`match` finds it for a direct provider's id).

        Returns:
            ``(input, output)`` per token, or ``None``.
        """
        row = self._rows.get(model_id) if self._rows is not None else None
        return row.rates if row is not None else None

    def ceiling(self, model_id: str, *, prompt_tokens: int, output_tokens: int) -> tuple[Decimal | None, str | None]:
        """The most one call on the model can be billed, in dollars, or no ceiling and why.

        Every price the row lists is bounded (the module docstring says how), or there is no ceiling: never a lower
        one. Rows read ``ceiling_stale_after`` ago or longer bound nothing.

        Args:
            model_id: The catalogue's id for the model.
            prompt_tokens: A bound on the call's prompt tokens, which also says which long-context tiers it can reach.
            output_tokens: The call's output cap, reasoning included.

        Returns:
            ``(ceiling, None)``, or ``(None, why)``: not loaded, rows too old, no priced row, or a price that cannot
            be bounded.
        """
        if self._rows is None or self._read_at is None:
            return None, "the price catalogue has not been read (await Catalogue.load() or Clients.load_prices())"
        if time.monotonic() - self._read_monotonic >= self.ceiling_stale_after.total_seconds():
            return None, (
                f"the catalogue's rates were read at {self._read_at.isoformat()}, ceiling_stale_after "
                f"({self.ceiling_stale_after}) ago or more, so they bound no call until a load refreshes them"
            )
        row = self._rows.get(model_id)
        if row is None or row.rates is None:
            return None, f"no priced row of OpenRouter's catalogue is {model_id!r}"
        ceiling, why = _ceiling(row, prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        if ceiling is None:
            return None, f"{model_id!r} cannot be bounded: {why}"
        return ceiling, None

    def cost(
        self,
        model_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int | None = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_write_1h_tokens: int | None = 0,
    ) -> tuple[Decimal | None, bool, str | None]:
        """What a call on the model cost, from what its reply reports, priced as the ceiling bounds it.

        Priced at the tier its prompt fell in, and from the rows held however old (a completion discloses that with
        ``pricing_stale_since``). A cache share priced at a rate the row does not list for it (the plain prompt rate,
        or the dearer of two write rates the reply does not split between) is approximate.

        Args:
            model_id: The catalogue's id for the model.
            input_tokens: Prompt tokens, cache reads and writes included.
            output_tokens: Completion tokens, reasoning included.
            reasoning_tokens: The reasoning share of ``output_tokens``, or ``None`` when the reply does not split it.
            cache_read_tokens: Prompt tokens read from a cache.
            cache_write_tokens: Prompt tokens written to a cache, both lifetimes.
            cache_write_1h_tokens: The one-hour share of ``cache_write_tokens``, or ``None`` when the reply does not
                split it.

        Returns:
            ``(cost, approximate, None)``, or ``(None, False, why)``: not loaded, no priced row, a price that cannot
            be read or bounded, or one that applies and that the reply's counts cannot price.
        """
        if self._rows is None:
            return None, False, "the price catalogue has not been read"
        row = self._rows.get(model_id)
        if row is None or row.rates is None:
            return None, False, f"no priced row of OpenRouter's catalogue is {model_id!r}"
        usage = _Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_write_1h_tokens=cache_write_1h_tokens,
        )
        prices, why = _prices(row, input_tokens)
        if prices is None:
            return None, False, f"{model_id!r} cannot be priced: {why}"
        cost, approximate, unpriced = _cost(prices, usage)
        if cost is None:
            return None, False, f"{model_id!r} cannot be priced: {unpriced}"
        return cost, approximate, None

    def supported_parameters(self, model_id: str) -> frozenset[str] | None:
        """The parameters some endpoint of the model takes, or ``None`` when the catalogue does not say.

        Args:
            model_id: The catalogue's id for the model.

        Returns:
            The listed parameters, or ``None`` (not loaded, not listed, or listed without them).
        """
        row = self._rows.get(model_id) if self._rows is not None else None
        return row.parameters if row is not None else None

    def max_output_tokens(self, model_id: str) -> int | None:
        """The model's output cap as the catalogue states it, or ``None``.

        Args:
            model_id: The catalogue's id for the model.

        Returns:
            The cap, or ``None``.
        """
        row = self._rows.get(model_id) if self._rows is not None else None
        return row.max_output_tokens if row is not None else None

    def match(self, model_id: str, *, vendor: str | None) -> str | None:
        """The catalogue id a direct provider's model id names, or ``None`` when no single row does.

        Ids are compared lower case, with dots as dashes and a date suffix left off, so ``claude-haiku-5-5`` matches
        ``anthropic/claude-haiku-5.5``. Variant rows (``:batch``, ``:free``) and aliases (``~…``) never match.

        Args:
            model_id: The id as the provider names it.
            vendor: The catalogue's vendor prefix (``anthropic``, ``openai``), or ``None`` for an endpoint whose
                vendor is unknown (an OpenAI-compatible ``base_url``): then exactly one row of any vendor must match.

        Returns:
            The catalogue id, or ``None`` (not loaded, no row, or more than one).
        """
        if self._rows is None:
            return None
        wanted = _plain(model_id.rsplit("/", 1)[-1])
        found = [
            row_id
            for row_id in self._rows
            if ":" not in row_id
            and not row_id.startswith("~")
            and "/" in row_id
            and (vendor is None or row_id.split("/", 1)[0] == vendor)
            and _plain(row_id.split("/", 1)[1]) == wanted
        ]
        return found[0] if len(found) == 1 else None
