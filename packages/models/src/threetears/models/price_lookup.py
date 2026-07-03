"""Pluggable model price lookup — a registry of pricing SOURCES (the platform pricing primitive).

3tears does NOT bake vendor prices into its provider modules — prices change, so consuming
products look them up. A **price source** knows how to price one family of providers; the module
keeps an ordered registry of them and dispatches ``lookup_price`` to the first source that
``covers`` the requested provider.

Built-in sources:

* :class:`OpenRouterPriceSource` — the **catch-all**. OpenRouter's public ``/models`` endpoint
  (no key needed) lists not only its own routes but anthropic and openai models too, each with
  per-token ``pricing`` and a ``context_length``. So a single fetch prices anthropic / openai /
  openrouter (and anything else OpenRouter happens to list). It ``covers`` every provider, so it
  sits LAST in the registry as the default.
* :class:`VoyagePriceSource` — Voyage embeddings/rerankers are **not** on OpenRouter, so this
  source scrapes Voyage's live pricing page (no baked numbers) and prices ``voyage`` / ``voyageai``.

Consumers extend the set with :func:`register_price_source` (the new source is inserted BEFORE the
catch-all, so its ``covers`` is consulted first).

Matching is conservative in every source — exact / provider-prefixed / suffix /
date-and-punctuation-normalised equality — so a source never silently attaches a *near* model's
price (a wrong price is worse than none → the admin hand-enters). Prices are returned **per 1M
tokens**.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

__all__ = [
    "OPENROUTER_MODELS_URL",
    "VOYAGE_PRICING_URL",
    "ModelOption",
    "OpenRouterPriceSource",
    "PriceLookup",
    "PriceSource",
    "VoyagePriceSource",
    "default_price_sources",
    "fetch_provider_models",
    "list_provider_models",
    "lookup_price",
    "match_price",
    "register_price_source",
]

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
VOYAGE_PRICING_URL = "https://docs.voyageai.com/docs/pricing"

#: per-token → per-1M-token.
_PER_MILLION = Decimal(1_000_000)
#: a trailing vendor date stamp — an 8-digit ``YYYYMMDD`` only (``-20250514`` / ``-20241022``). We
#: deliberately do NOT strip shorter digit runs: an OpenAI snapshot id like ``gpt-4-0613`` ends in 4
#: digits that are NOT a date, and stripping them would collapse distinct snapshots to one normalised
#: key and attach the WRONG snapshot's price (the matcher's "never a near price" contract).
_DATE_SUFFIX = re.compile(r"[-_]?\d{8}$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class PriceLookup:
    """A lookup result. ``matched_id`` is ``None`` (and prices ``None``) when nothing matched.

    ``source`` / ``source_url`` carry provenance so a consumer can show *where* the price came
    from (e.g. ``"openrouter"`` vs ``"voyage"``). They default to the OpenRouter source so existing
    callers and the all-``None`` "no match" result stay unchanged.
    """

    provider: str
    api_name: str
    matched_id: str | None
    cost_per_1m_input: Decimal | None
    cost_per_1m_output: Decimal | None
    context_window_tokens: int | None
    source: str = "openrouter"
    source_url: str | None = OPENROUTER_MODELS_URL


class _HttpClient(Protocol):
    """The minimal httpx-AsyncClient surface this module needs (one ``get``)."""

    async def get(self, url: str, /) -> Any: ...


class PriceSource(Protocol):
    """A pluggable pricing source for one family of providers.

    A source decides which providers it ``covers`` and knows how to ``lookup`` a single model's
    price for them. Registered sources are consulted in order (see :func:`register_price_source`);
    the first whose ``covers`` is ``True`` handles the request.
    """

    def covers(self, provider: str) -> bool:
        """Report whether this source prices ``provider`` (a lowercase slug).

        :param provider: the provider slug to test (e.g. ``voyage`` / ``anthropic``).
        :ptype provider: str
        :return: ``True`` if this source can price the provider.
        :rtype: bool
        """
        ...

    async def lookup(self, client: _HttpClient, *, provider: str, api_name: str) -> PriceLookup:
        """Price ``(provider, api_name)`` using ``client`` for any network fetch.

        :param client: an async HTTP client exposing ``get(url)`` (e.g. ``httpx.AsyncClient``).
        :ptype client: _HttpClient
        :param provider: the provider slug to price under.
        :ptype provider: str
        :param api_name: the model id to price.
        :ptype api_name: str
        :return: the matched price (or an all-``None`` result when nothing matched).
        :rtype: PriceLookup
        """
        ...


def _normalise(name: str) -> str:
    """Lowercase, drop a trailing vendor date, and strip punctuation — so ``claude-3-5-sonnet``
    (anthropic) and ``claude-3.5-sonnet`` (OpenRouter) collapse to the same token."""
    base = _DATE_SUFFIX.sub("", name.strip().lower())
    return _NON_ALNUM.sub("", base)


def _to_per_million(per_token: object) -> Decimal | None:
    """Convert OpenRouter's per-token price string to a per-1M ``Decimal``; ``None`` if absent/zero/bad."""
    if per_token in (None, "", "0"):
        return None
    try:
        value = Decimal(str(per_token)) * _PER_MILLION
    except InvalidOperation, ValueError:
        return None
    return value if value > 0 else None


def _candidate_ids(provider: str, api_name: str) -> tuple[str, ...]:
    """The exact ids we accept before falling back to normalised matching."""
    return (api_name, f"{provider}/{api_name}")


def match_price(catalogue: Sequence[dict[str, Any]], *, provider: str, api_name: str) -> PriceLookup:
    """Match ``(provider, api_name)`` against an OpenRouter catalogue list; never a *near* price.

    Tiers, first hit wins: (1) exact id; (2) ``provider/api_name``; (3) the id's last path
    segment equals ``api_name``; (4) date-and-punctuation-normalised equality of the last segment.

    :param catalogue: OpenRouter ``/models`` ``data`` entries (each a dict with ``id``/``pricing``).
    :ptype catalogue: Sequence[dict[str, Any]]
    :param provider: the provider slug to confine fuzzy tiers to (e.g. ``anthropic``).
    :ptype provider: str
    :param api_name: the model id to price (e.g. ``claude-3-5-sonnet-20241022``).
    :ptype api_name: str
    :return: the matched price (or an all-``None`` result when nothing matched).
    :rtype: PriceLookup
    """
    exact = set(_candidate_ids(provider, api_name))
    target_norm = _normalise(api_name)
    suffix_hit: dict[str, Any] | None = None
    norm_hit: dict[str, Any] | None = None

    for entry in catalogue:
        entry_id = str(entry.get("id", ""))
        if not entry_id:
            continue
        if entry_id in exact:
            return _result(entry, provider, api_name)
        # The fuzzy tiers (suffix / normalised) compare only the model segment, so they MUST be
        # confined to the requested provider's namespace — else `anthropic` + `gpt-4o` would match
        # `openai/gpt-4o` and attach the wrong provider's price (the "never a near price" contract).
        if "/" in entry_id and entry_id.split("/", 1)[0] != provider:
            continue
        last = entry_id.split("/")[-1]
        if suffix_hit is None and last == api_name:
            suffix_hit = entry
        elif norm_hit is None and _normalise(last) == target_norm:
            norm_hit = entry

    chosen = suffix_hit or norm_hit
    if chosen is not None:
        return _result(chosen, provider, api_name)
    return PriceLookup(provider, api_name, None, None, None, None)


def _result(entry: dict[str, Any], provider: str, api_name: str) -> PriceLookup:
    pricing = entry.get("pricing") or {}
    context = entry.get("context_length")
    return PriceLookup(
        provider=provider,
        api_name=api_name,
        matched_id=str(entry.get("id")),
        cost_per_1m_input=_to_per_million(pricing.get("prompt")),
        cost_per_1m_output=_to_per_million(pricing.get("completion")),
        context_window_tokens=int(context) if isinstance(context, int) else None,
    )


@dataclass(frozen=True)
class ModelOption:
    """One pickable model from the catalogue — the api_name to register + its pre-fill metadata."""

    api_name: str
    display_name: str
    context_window_tokens: int | None
    cost_per_1m_input: Decimal | None
    cost_per_1m_output: Decimal | None


def list_provider_models(catalogue: Sequence[dict[str, Any]], *, provider: str) -> list[ModelOption]:
    """The selectable models for ``provider`` from an OpenRouter catalogue list (for the admin picker).

    For ``openrouter`` the catalogue id IS the api_name (the routed slug, e.g.
    ``deepseek/deepseek-v4-flash``) — every entry is offered. For ``anthropic`` / ``openai`` only the
    entries under that provider's namespace are offered, and the api_name is the slug's model segment
    (exact for openai; best-effort for anthropic's dated ids — the admin can adjust). Sorted by name.

    :param catalogue: OpenRouter ``/models`` ``data`` entries (each a dict with ``id``/``pricing``).
    :ptype catalogue: Sequence[dict[str, Any]]
    :param provider: the provider slug to list models for (e.g. ``openrouter``).
    :ptype provider: str
    :return: the selectable models, sorted by ``api_name``.
    :rtype: list[ModelOption]
    """
    out: list[ModelOption] = []
    for entry in catalogue:
        entry_id = str(entry.get("id", ""))
        if not entry_id:
            continue
        if provider == "openrouter":
            api_name = entry_id
        else:
            if "/" not in entry_id or entry_id.split("/", 1)[0] != provider:
                continue
            api_name = entry_id.split("/", 1)[1]
        pricing = entry.get("pricing") or {}
        context = entry.get("context_length")
        out.append(
            ModelOption(
                api_name=api_name,
                display_name=str(entry.get("name") or api_name),
                context_window_tokens=int(context) if isinstance(context, int) else None,
                cost_per_1m_input=_to_per_million(pricing.get("prompt")),
                cost_per_1m_output=_to_per_million(pricing.get("completion")),
            )
        )
    out.sort(key=lambda m: m.api_name)
    return out


async def _fetch_openrouter_catalogue(client: _HttpClient) -> list[dict[str, Any]]:
    """Fetch OpenRouter's ``/models`` catalogue ``data`` list (a transport error propagates)."""
    response = await client.get(OPENROUTER_MODELS_URL)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", []) if isinstance(payload, dict) else []


class OpenRouterPriceSource:
    """The catch-all price source: OpenRouter's ``/models`` catalogue.

    Prices anthropic / openai / openrouter (and anything else the catalogue lists) from a single
    fetch, per-token → per-1M, conservative matching. ``covers`` returns ``True`` for every provider
    — this source is the registry's default, so it MUST sit last (see :func:`default_price_sources`).
    """

    source_name = "openrouter"

    def covers(self, provider: str) -> bool:  # noqa: ARG002 -- catch-all covers everything
        """Always ``True`` — OpenRouter is the universal/default source.

        :param provider: the provider slug (ignored; the catch-all covers all).
        :ptype provider: str
        :return: ``True``.
        :rtype: bool
        """
        return True

    async def lookup(self, client: _HttpClient, *, provider: str, api_name: str) -> PriceLookup:
        """Fetch the OpenRouter catalogue and conservatively match ``(provider, api_name)``.

        A transport/HTTP error propagates to the caller (a lookup failure just means hand-enter).

        :param client: an async HTTP client exposing ``get(url)`` (e.g. ``httpx.AsyncClient``).
        :ptype client: _HttpClient
        :param provider: the provider slug to price under (e.g. ``openai``).
        :ptype provider: str
        :param api_name: the model id to price (e.g. ``gpt-4o``).
        :ptype api_name: str
        :return: the matched price (or an all-``None`` result when nothing matched).
        :rtype: PriceLookup
        """
        catalogue = await _fetch_openrouter_catalogue(client)
        return match_price(catalogue, provider=provider, api_name=api_name)

    async def list_models(self, client: _HttpClient, *, provider: str) -> list[ModelOption]:
        """Fetch the catalogue and return the selectable models for ``provider`` (the admin picker).

        :param client: an async HTTP client exposing ``get(url)`` (e.g. ``httpx.AsyncClient``).
        :ptype client: _HttpClient
        :param provider: the provider slug to list models for (e.g. ``openrouter``).
        :ptype provider: str
        :return: the selectable models, sorted by ``api_name``.
        :rtype: list[ModelOption]
        """
        catalogue = await _fetch_openrouter_catalogue(client)
        return list_provider_models(catalogue, provider=provider)


# ---------------------------------------------------------------------------
# Voyage — scrapes the live pricing page (no baked numbers).
# ---------------------------------------------------------------------------

#: providers the Voyage source prices.
_VOYAGE_PROVIDERS = frozenset({"voyage", "voyageai"})
#: the readme.io hydration field whose JSON-string value is the rendered page body HTML.
_VOYAGE_BODY_FIELD = re.compile(r'"body":"((?:\\.|[^"\\])*)"')
#: one HTML table.
_TABLE_RE = re.compile(r"<table\b[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
#: one table row.
_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
#: one header cell.
_TH_RE = re.compile(r"<th\b[^>]*>(.*?)</th>", re.IGNORECASE | re.DOTALL)
#: one body cell.
_TD_RE = re.compile(r"<td\b[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
#: a ``<code>voyage-4</code>`` model token inside a cell (one cell may list several via ``<br/>``).
_CODE_RE = re.compile(r"<code\b[^>]*>(.*?)</code>", re.IGNORECASE | re.DOTALL)
#: strip any residual tags from a cell's text.
_TAG_RE = re.compile(r"<[^>]+>")
#: the per-1M column header we key on (NOT the per-thousand or per-pixel columns).
_PER_MILLION_HEADER = "price per million tokens"
#: a ``$0.06`` style price.
_PRICE_RE = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")


def _cell_text(cell_html: str) -> str:
    """The plain text of an HTML cell (tags stripped, entities unescaped, whitespace collapsed)."""
    return " ".join(html.unescape(_TAG_RE.sub(" ", cell_html)).split())


def _parse_voyage_prices(body_html: str) -> dict[str, Decimal]:
    """Parse Voyage's pricing tables into ``{model_id: per_1M_usd}``.

    The Voyage docs (a readme.io site) embed the rendered page body as an HTML string in a JSON
    hydration field; every pricing table has a ``Model`` column of ``<code>`` ids and a
    ``Price per million tokens`` column of ``$X`` values. We locate that column by its header text
    (so the per-*thousand* and per-*pixel* columns are ignored), then map each model id in the row
    to the row's per-1M price. A cell may list several models (``<br/>``-joined ``<code>``) sharing
    one price. Rows/tables that don't fit the shape are skipped — never guessed.

    :param body_html: the decoded page body HTML (the value of the ``"body"`` hydration field).
    :ptype body_html: str
    :return: model id → per-1M USD price (may be empty if no priced rows were found).
    :rtype: dict[str, Decimal]
    """
    prices: dict[str, Decimal] = {}
    for table_html in _TABLE_RE.findall(body_html):
        rows = _ROW_RE.findall(table_html)
        if not rows:
            continue
        # The header row carries the column labels; find the per-1M column index.
        headers = [_cell_text(h).lower() for h in _TH_RE.findall(rows[0])]
        try:
            price_col = headers.index(_PER_MILLION_HEADER)
        except ValueError:
            continue  # not a per-1M-token table (e.g. multimodal per-pixel) — skip it.
        for row_html in rows[1:]:
            cells = _TD_RE.findall(row_html)
            if len(cells) <= price_col:
                continue
            model_ids = [html.unescape(c).strip() for c in _CODE_RE.findall(cells[0])]
            if not model_ids:
                continue
            price_match = _PRICE_RE.search(_cell_text(cells[price_col]))
            if not price_match:
                continue
            try:
                price = Decimal(price_match.group(1))
            except InvalidOperation:
                continue
            if price <= 0:
                continue
            for model_id in model_ids:
                if model_id:
                    prices.setdefault(model_id, price)
    return prices


def _extract_voyage_body(page_html: str) -> str | None:
    """Pull the decoded page-body HTML out of the readme.io hydration blob.

    The page embeds its rendered body as a JSON string under a ``"body":`` key. We grab that string
    and JSON-decode it (handling the ``\\uXXXX`` / ``\\n`` escapes). Returns ``None`` if the field
    isn't present or doesn't decode — the caller then fails safe to all-``None``.

    :param page_html: the raw pricing-page HTML fetched over HTTP.
    :ptype page_html: str
    :return: the decoded body HTML, or ``None`` on any structural surprise.
    :rtype: str | None
    """
    match = _VOYAGE_BODY_FIELD.search(page_html)
    if not match:
        return None
    try:
        decoded = json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError, ValueError:
        return None
    return decoded if isinstance(decoded, str) and decoded else None


def parse_voyage_pricing(page_html: str) -> dict[str, Decimal]:
    """Parse a fetched Voyage pricing page into ``{model_id: per_1M_usd}`` (empty on any surprise).

    Extracts the hydration body then parses its pricing tables (see :func:`_parse_voyage_prices`).
    Never raises on a structural mismatch — a parse miss yields an empty map so the caller hand-enters.

    :param page_html: the raw pricing-page HTML fetched over HTTP.
    :ptype page_html: str
    :return: model id → per-1M USD price (empty if the page couldn't be parsed).
    :rtype: dict[str, Decimal]
    """
    body = _extract_voyage_body(page_html)
    if body is None:
        return {}
    return _parse_voyage_prices(body)


def _match_voyage_model(prices: dict[str, Decimal], api_name: str) -> tuple[str, Decimal] | None:
    """Conservatively match ``api_name`` against scraped Voyage model ids (never a near price).

    Exact id first, then date-and-punctuation-normalised equality (so ``voyage-3.5`` matches a
    request for ``voyage-3-5``). No fuzzy/substring matching — an unmatched name yields ``None``.

    :param prices: scraped model id → per-1M price.
    :ptype prices: dict[str, Decimal]
    :param api_name: the requested model id.
    :ptype api_name: str
    :return: ``(matched_id, price)`` or ``None`` when nothing matched.
    :rtype: tuple[str, Decimal] | None
    """
    if api_name in prices:
        return api_name, prices[api_name]
    target = _normalise(api_name)
    for model_id, price in prices.items():
        if _normalise(model_id) == target:
            return model_id, price
    return None


class VoyagePriceSource:
    """Prices Voyage embeddings/rerankers by scraping the live Voyage pricing page.

    Voyage models are not on OpenRouter, so this source fetches ``VOYAGE_PRICING_URL`` and parses
    the per-1M column out of its pricing tables. Embeddings/rerankers are input-priced, so the
    result sets ``cost_per_1m_input`` and leaves ``cost_per_1m_output`` / ``context_window_tokens``
    ``None``. A fetch transport error propagates (like the OpenRouter path); any *parse* miss
    (page restructured, model not listed) fails safe to an all-``None`` result — never a wrong price.
    """

    source_name = "voyage"

    def covers(self, provider: str) -> bool:
        """``True`` for ``voyage`` / ``voyageai``.

        :param provider: the provider slug to test.
        :ptype provider: str
        :return: ``True`` when this is a Voyage provider.
        :rtype: bool
        """
        return provider in _VOYAGE_PROVIDERS

    async def lookup(self, client: _HttpClient, *, provider: str, api_name: str) -> PriceLookup:
        """Scrape Voyage pricing and conservatively match ``api_name``.

        A transport/HTTP error propagates; a parse miss or no-match returns all-``None`` (with the
        Voyage provenance still attached) so the admin hand-enters.

        :param client: an async HTTP client exposing ``get(url)`` (e.g. ``httpx.AsyncClient``).
        :ptype client: _HttpClient
        :param provider: the provider slug (``voyage`` / ``voyageai``).
        :ptype provider: str
        :param api_name: the model id to price (e.g. ``voyage-4``).
        :ptype api_name: str
        :return: the matched price (or an all-``None`` result on parse miss / no match).
        :rtype: PriceLookup
        """
        empty = PriceLookup(
            provider=provider,
            api_name=api_name,
            matched_id=None,
            cost_per_1m_input=None,
            cost_per_1m_output=None,
            context_window_tokens=None,
            source=self.source_name,
            source_url=VOYAGE_PRICING_URL,
        )
        response = await client.get(VOYAGE_PRICING_URL)
        response.raise_for_status()
        page_html = response.text
        if not isinstance(page_html, str):
            return empty
        prices = parse_voyage_pricing(page_html)
        matched = _match_voyage_model(prices, api_name)
        if matched is None:
            return empty
        matched_id, price = matched
        return PriceLookup(
            provider=provider,
            api_name=api_name,
            matched_id=matched_id,
            cost_per_1m_input=price,
            cost_per_1m_output=None,
            context_window_tokens=None,
            source=self.source_name,
            source_url=VOYAGE_PRICING_URL,
        )


# ---------------------------------------------------------------------------
# The source registry.
# ---------------------------------------------------------------------------


def default_price_sources() -> list[PriceSource]:
    """The built-in source ordering: specific sources first, the OpenRouter catch-all LAST.

    :return: a fresh list of the built-in sources in dispatch order.
    :rtype: list[PriceSource]
    """
    return [VoyagePriceSource(), OpenRouterPriceSource()]


@dataclass
class _SourceRegistry:
    """An ordered list of price sources whose last entry is the catch-all.

    ``register`` inserts a new source BEFORE the catch-all so the default still backstops anything
    the new source doesn't ``cover``.
    """

    sources: list[PriceSource] = field(default_factory=default_price_sources)

    def register(self, source: PriceSource) -> None:
        # Insert before the final (catch-all) entry, so the catch-all always remains last.
        self.sources.insert(len(self.sources) - 1, source)

    def resolve(self, provider: str) -> PriceSource:
        for source in self.sources:
            if source.covers(provider):
                return source
        # The catch-all's covers() is always True, so this is unreachable in practice; fall back
        # to the last source defensively rather than raising.
        return self.sources[-1]


#: the module-level default registry that :func:`lookup_price` dispatches through.
_REGISTRY = _SourceRegistry()


def register_price_source(source: PriceSource) -> None:
    """Add ``source`` to the default registry, BEFORE the OpenRouter catch-all.

    Consumers call this to teach the lookup about a provider OpenRouter doesn't list. The new
    source's ``covers`` is consulted before the catch-all, so it wins for the providers it claims
    while the catch-all still backstops everything else.

    :param source: a :class:`PriceSource` implementation to register.
    :ptype source: PriceSource
    :return: ``None``.
    :rtype: None
    """
    _REGISTRY.register(source)


async def lookup_price(client: _HttpClient, *, provider: str, api_name: str) -> PriceLookup:
    """Price ``(provider, api_name)`` via the first registered source that ``covers`` ``provider``.

    The caller owns the HTTP client lifecycle (an ``httpx.AsyncClient`` with a timeout). A
    transport/HTTP error propagates to the caller, which surfaces it to the admin (lookup is a
    best-effort pre-fill; a failure just means hand-enter). A *parse* miss inside a source yields an
    all-``None`` result rather than raising.

    :param client: an async HTTP client exposing ``get(url)`` (e.g. ``httpx.AsyncClient``).
    :ptype client: _HttpClient
    :param provider: the provider slug to price under (e.g. ``openai`` / ``voyage``).
    :ptype provider: str
    :param api_name: the model id to price (e.g. ``gpt-4o`` / ``voyage-4``).
    :ptype api_name: str
    :return: the matched price (or an all-``None`` result when nothing matched).
    :rtype: PriceLookup
    """
    source = _REGISTRY.resolve(provider)
    return await source.lookup(client, provider=provider, api_name=api_name)


async def fetch_provider_models(client: _HttpClient, *, provider: str) -> list[ModelOption]:
    """Fetch the OpenRouter catalogue and return the selectable models for ``provider`` (the picker).

    The admin model picker is OpenRouter-backed regardless of pricing source (it lists the
    catalogue's chat models); embedding-only vendors are added by the admin directly.

    :param client: an async HTTP client exposing ``get(url)`` (e.g. ``httpx.AsyncClient``).
    :ptype client: _HttpClient
    :param provider: the provider slug to list models for (e.g. ``openrouter``).
    :ptype provider: str
    :return: the selectable models, sorted by ``api_name``.
    :rtype: list[ModelOption]
    """
    catalogue = await _fetch_openrouter_catalogue(client)
    return list_provider_models(catalogue, provider=provider)
