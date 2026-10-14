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

__all__ = ["DEFAULT_REFRESH_AFTER", "OPENROUTER_MODELS_URL", "RETRY_AFTER_FAILURE", "Catalogue"]

log = get_logger(__name__)

#: How long a catalogue keeps the rows it read before the next load fetches them again, unless built with another.
DEFAULT_REFRESH_AFTER: Final = timedelta(hours=6)

#: How long after a failed refresh the next load tries again (or ``refresh_after``, when that is shorter): a provider
#: outage costs one failed fetch per this interval, not one per call.
RETRY_AFTER_FAILURE: Final = timedelta(minutes=5)

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
    """OpenRouter's model catalogue, refreshed every ``refresh_after``, and what providers refused, for every model
    that reads it.

    Hand one catalogue to several :class:`~threetears.evals.providers.Model`\\s (``Clients`` does) and they fetch it
    together and learn from each other's refusals. Await :meth:`load` at boot when a model's
    :meth:`~threetears.evals.providers.Model.price_ceiling` must answer before its first call; a live call loads it
    otherwise, and refreshes it once its rows are older than ``refresh_after``.

    Attributes:
        url: Where the catalogue is read from.
        refresh_after: How long the rows read are kept before the next load fetches them again.
        learned: The request settings providers refused, per provider and model, for every model reading this
            catalogue (:class:`~threetears.evals.negotiation.LearnedSettings`).
    """

    def __init__(
        self,
        *,
        url: str = OPENROUTER_MODELS_URL,
        http: httpx.AsyncClient | None = None,
        refresh_after: timedelta = DEFAULT_REFRESH_AFTER,
    ) -> None:
        """Know nothing yet: nothing is fetched until :meth:`load`.

        Args:
            url: The catalogue's URL.
            http: The client to fetch it with (not closed here), or ``None`` to open one for the fetch.
            refresh_after: How long the rows read are kept before the next load fetches them again; zero refreshes
                on every load.

        Raises:
            ValueError: ``refresh_after`` is negative.
        """
        if refresh_after < timedelta(0):
            raise ValueError(
                f"refresh_after {refresh_after} is negative: a catalogue keeps its rows for no time or more"
            )
        self.url = url
        self.refresh_after = refresh_after
        self._http = http
        self._rows: dict[str, _Row] | None = None
        #: When the kept rows were read, and the monotonic instant the next load fetches again.
        self._read_at: datetime | None = None
        self._due = 0.0
        self._stale_since: datetime | None = None
        self._refresh_failure: str | None = None
        self._lock = asyncio.Lock()
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
    def refresh_failure(self) -> str | None:
        """Why the last refresh failed, while the rows are stale; ``None`` while they are fresh."""
        return self._refresh_failure

    async def load(self) -> None:
        """Fetch and read the catalogue unless the rows kept are younger than ``refresh_after``.

        A first read that fails raises and keeps nothing, so the next load fetches again. A refresh that fails keeps
        the rows read before, logs the failure and marks them stale (:attr:`stale_since`, :attr:`refresh_failure`)
        rather than raising, and the next load :data:`RETRY_AFTER_FAILURE` later (or ``refresh_after``, when that is
        shorter) tries again; one that succeeds clears the mark.

        Raises:
            httpx.HTTPError: The first fetch failed or answered with a failure status.
            ValueError: The first fetch's body is not JSON, or holds no ``data`` list.
        """
        async with self._lock:
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
            self._due = time.monotonic() + self.refresh_after.total_seconds()
            self._stale_since = None
            self._refresh_failure = None

    async def _fetch(self) -> dict[str, _Row]:
        """One fetch of the catalogue, read into rows.

        Raises:
            httpx.HTTPError: The fetch failed or answered with a failure status.
            ValueError: The body is not JSON, or holds no ``data`` list.
        """
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
