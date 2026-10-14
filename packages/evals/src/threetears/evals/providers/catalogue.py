"""OpenRouter's public model catalogue, read once: rates, the parameters each model takes, and what refusals taught.

:class:`Catalogue` is the one price source the client has. OpenRouter's ``GET /api/v1/models`` needs no key and lists,
per model, its per-token rates, the ``supported_parameters`` some endpoint of it takes, and its output cap. It prices
Anthropic and OpenAI models too (``anthropic/claude-haiku-5.5``), so a call made to either directly is priced by the
same rows. Each instance fetches it at most once (a failed fetch is not kept, so the next asks again) and every
:class:`~threetears.evals.providers.Model` handed one shares it, with the request settings a provider refused, which
the catalogue holds for every model that reads it (:attr:`Catalogue.learned`). There is no process-wide catalogue: a
``Model`` given none builds its own, fetched on its first live call.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Final

import httpx

from threetears.evals.negotiation import LearnedSettings

__all__ = ["OPENROUTER_MODELS_URL", "Catalogue"]

#: OpenRouter's public model catalogue.
OPENROUTER_MODELS_URL: Final = "https://openrouter.ai/api/v1/models"

#: How long the catalogue fetch may take, in seconds, on a client the catalogue opens itself.
_FETCH_TIMEOUT_S: Final = 30.0

#: A date suffix on a direct provider's model id (``claude-haiku-4-5-20251001``), which the catalogue's ids leave off.
_DATE_SUFFIX: Final = re.compile(r"-\d{8}$")


@dataclass(frozen=True, slots=True)
class _Row:
    """One catalogue row: the rates per token (``None`` when unpriced), the listed parameters, the output cap."""

    rates: tuple[Decimal, Decimal] | None
    parameters: frozenset[str] | None
    max_output_tokens: int | None


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
    )


def _plain(model_id: str) -> str:
    """A model id as the catalogue's would match it: lower case, dots as dashes, no date suffix."""
    return _DATE_SUFFIX.sub("", model_id.lower().replace(".", "-"))


class Catalogue:
    """OpenRouter's model catalogue, fetched at most once, and what providers refused, for every model that reads it.

    Hand one catalogue to several :class:`~threetears.evals.providers.Model`\\s (``Clients`` does) and they fetch it
    once and learn from each other's refusals. Await :meth:`load` at boot when a model's
    :meth:`~threetears.evals.providers.Model.price_ceiling` must answer before its first call; a live call loads it
    otherwise.

    Attributes:
        url: Where the catalogue is read from.
        learned: The request settings providers refused, per provider and model, for every model reading this
            catalogue (:class:`~threetears.evals.negotiation.LearnedSettings`).
    """

    def __init__(self, *, url: str = OPENROUTER_MODELS_URL, http: httpx.AsyncClient | None = None) -> None:
        """Know nothing yet: nothing is fetched until :meth:`load`.

        Args:
            url: The catalogue's URL.
            http: The client to fetch it with (not closed here), or ``None`` to open one for the fetch.
        """
        self.url = url
        self._http = http
        self._rows: dict[str, _Row] | None = None
        self._lock = asyncio.Lock()
        self.learned = LearnedSettings(
            client="eval.providers",
            declared_by="catalogue",
            contradiction="OpenRouter's catalogue, or a fact the client declares, which the provider contradicts",
        )

    @property
    def loaded(self) -> bool:
        """Whether the catalogue has been read."""
        return self._rows is not None

    async def load(self) -> None:
        """Fetch and read the catalogue, once; a later call returns at once.

        Raises:
            httpx.HTTPError: The fetch failed or answered with a failure status. Nothing is kept, so a later call
                fetches again.
            ValueError: The body is not JSON, or holds no ``data`` list.
        """
        async with self._lock:
            if self._rows is not None:
                return
            if self._http is not None:
                response = await self._http.get(self.url)
            else:
                async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S, follow_redirects=True) as owned:
                    response = await owned.get(self.url)
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("data") if isinstance(payload, Mapping) else None
            if not isinstance(rows, list):
                raise ValueError(f"the catalogue at {self.url} holds no `data` list")
            self._rows = {
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
