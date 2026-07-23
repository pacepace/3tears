"""ListingDetailDriver -- ``ScrapeDriver`` backend for a listing table whose
own rows link to a per-record detail page carrying fields the listing itself
doesn't have.

**Design (scrape-task-07 follow-up, 2026-07-16):** the Geographic Solutions
"JobLink" platform (Rails + Ransack search) backs Arizona/Delaware/Kansas/
Vermont's real WARN Act pages, and (previously mis-configured, see below)
Maine's -- confirmed live, identical across all 5 states: a plain GET-
reachable listing (``<base_url>/search/warn_lookups?q[s]=notice_on+desc``,
no browser, no auth) whose table columns are Employer/City/ZIP/LWIB Area/
Notice Date/WARN Type, each employer name linking to a per-record detail
page at ``<base_url>/search/warn_lookups/<id>`` with a uniform
``<h3 class="definition-list__title">Label</h3><p class="definition-list__definition">Value</p>``
structure carrying Company Name/[Address]/Notice Date/Number of Employees Affected.

**Why not :class:`~threetears.scrape.drivers.multi_document.
MultiDocumentDriver`:** that driver's own data-flow discards the listing's
row data once document links are found, treating each linked document as a
wholly self-contained record (true for West Virginia/Hawaii's independent
PDF letters) -- but a JobLink record's own fields are genuinely split
across BOTH layers, and can even disagree: a real Kansas record's detail
page had a blank "Notice Date" while its listing row had a real one. A
record's true field values have to be resolved by MERGING listing-row data
and detail-page data, not sourced from one exclusively -- which needs each
detail link resolved WITHIN its own row's scope (pairing a row's listing
fields with that SAME row's detail fields), not the flat, page-wide link
list ``MultiDocumentDriver``'s own ``_discover_links_html`` returns. This
driver's own row-scoped resolution is genuinely different from that
helper's shape, not a copy of it.

**Why the driver resolves everything itself instead of feeding a
``per_document``-style LLM extraction pass:** these detail pages are
uniform structured markup, not freeform prose (West Virginia/Hawaii's own
independently-worded letters) -- an LLM read of them would be real,
unnecessary cost for data a deterministic label-match reads for free (the
same "cheap path where it works" call as Mississippi's PDF row-merge fix
vs. Nevada's vision extraction, scrape-task-07). This driver parses both
layers itself and emits ONE flat synthetic ``<table>`` (via
:func:`~threetears.scrape.drivers.api._records_to_synthetic_table`, reused
unchanged) -- the existing, unmodified plain ``"css"`` multi-row strategy
(``generate_row_candidates`` -> ``validate_row_candidate`` -> judge ->
cached recipe) does everything downstream with zero new extraction-side
code, inheriting its own all-or-nothing-per-record contract (a record
missing a required schema field is dropped, never fabricated) for free.

**Merge rule, per field:** ``detail_fields.get(field) or listing_fields.
get(field)`` -- prefer the detail page's own value, fall back to the
listing row's when the detail page's is blank or absent (the real gap the
Kansas record above showed). A field absent on BOTH layers stays absent,
flowing into ``validate_row_candidate``'s own drop-incomplete-rows behavior
untouched -- never fabricated.

**Politeness, on by default:** unlike ``MultiDocumentDriver`` (existing,
proven, no evidenced need to change default behavior for its own
consumers), this is a brand-new driver with no existing traffic pattern to
preserve -- :data:`_DEFAULT_PACE_DELAY_SECONDS` defaults to a real, non-zero
delay between detail-page fetches. These are unprotected GETs against real
state government servers (confirmed live: zero bot protection on any of the
5 states checked); staying unprotected means not behaving like a threat.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup, Tag
from threetears.observe import get_logger

from ..driver import NavStep, RenderedPage, ScrapeDriver
from .api import _records_to_synthetic_table

__all__ = ["ListingDetailDriver", "ListingDetailDriverError"]

log = get_logger(__name__)

#: Same fix, same reason as every other driver's own _DEFAULT_USER_AGENT.
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

#: See this module's own docstring, "Politeness, on by default" -- a real,
#: non-zero pause between detail-page fetches. Live-tuned, not guessed: an
#: initial 0.1s pace against the real Arizona/Delaware/Kansas/Vermont/Maine
#: servers drew intermittent 429 Too Many Requests on detail fetches
#: (live-observed across all 5 during this driver's own eval-loop proof run)
#: -- raised to 0.5s plus explicit 429 backoff (below) in response.
_DEFAULT_PACE_DELAY_SECONDS = 0.5

#: A detail fetch returning 429 is the server's own explicit "slow down"
#: signal (distinct from a generic error status) -- extend the NEXT pause
#: rather than just moving on at the normal pace, so a burst of 429s backs
#: off instead of plowing through at the same rate that triggered them.
_RATE_LIMITED_EXTRA_DELAY_SECONDS = 2.0

#: Mirrors MultiDocumentDriver's own _DEFAULT_MAX_DOCUMENTS -- "cover a
#: typical poll cycle's worth of new notices with margin," not evidenced to
#: need a per-target override yet.
_DEFAULT_MAX_ROWS = 20


class ListingDetailDriverError(Exception):
    """Raised when the listing fetch fails outright (not a single row's own detail-fetch failure).

    Mirrors ``MultiDocumentDriverError``/``ApiDriverError``'s ``code``/``message`` shape.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _extract_listing_row_fields(row: Tag, column_fields: dict[int, str]) -> dict[str, str]:
    """Read *column_fields*' requested ``<td>`` indices out of one listing *row*.

    :param row: one ``<tr>`` element from the listing table
    :ptype row: Tag
    :param column_fields: 0-based ``<td>`` index -> field name
    :ptype column_fields: dict[int, str]
    :return: field_name -> text (present only when that index existed and had text)
    :rtype: dict[str, str]
    """
    cells = row.find_all("td")
    fields: dict[str, str] = {}
    for index, field_name in column_fields.items():
        if index < len(cells):
            text = cells[index].get_text(" ", strip=True)
            if text:
                fields[field_name] = text
    return fields


def _parse_definition_list(html: str, field_labels: dict[str, str]) -> dict[str, str]:
    """Read a detail page's own ``definition-list__title``/``definition-list__definition`` pairs.

    :param html: the detail page's full HTML
    :ptype html: str
    :param field_labels: field_name -> the exact label text to look for (e.g.
        ``{"affected_count": "Number of Employees Affected"}``)
    :ptype field_labels: dict[str, str]
    :return: field_name -> text, present only for a label that was found AND had a non-empty value
    :rtype: dict[str, str]
    """
    soup = BeautifulSoup(html, "html.parser")
    label_to_value: dict[str, str] = {}
    for title in soup.select(".definition-list__title"):
        value_tag = title.find_next_sibling(class_="definition-list__definition")
        if value_tag is None:
            continue
        text = value_tag.get_text(" ", strip=True)
        if text:
            label_to_value[title.get_text(strip=True)] = text
    return {field_name: label_to_value[label] for field_name, label in field_labels.items() if label in label_to_value}


class ListingDetailDriver(ScrapeDriver):
    """``ScrapeDriver`` backed by a listing table plus a per-row detail-page fetch, merged."""

    def __init__(
        self,
        *,
        row_selector: str,
        listing_field_columns: dict[int, str],
        detail_link_column: int,
        detail_field_labels: dict[str, str],
        client: httpx.AsyncClient | None = None,
        max_rows: int = _DEFAULT_MAX_ROWS,
        pace_delay_seconds: float = _DEFAULT_PACE_DELAY_SECONDS,
    ) -> None:
        """
        :param row_selector: CSS selector for each listing row (e.g. ``"table tbody tr"``)
        :ptype row_selector: str
        :param listing_field_columns: 0-based ``<td>`` index -> field name, applied to every row
        :ptype listing_field_columns: dict[int, str]
        :param detail_link_column: 0-based ``<td>`` index containing the ``<a href>`` to that row's detail page
        :ptype detail_link_column: int
        :param detail_field_labels: field_name -> exact label text to read off each detail page
        :ptype detail_field_labels: dict[str, str]
        :param client: an already-constructed httpx client to reuse (test
            injection); a fresh one is created per call when omitted
        :ptype client: httpx.AsyncClient | None
        :param max_rows: how many of the listing's rows to resolve, newest-first
        :ptype max_rows: int
        :param pace_delay_seconds: pause between detail-page fetches (see this
            module's own docstring, "Politeness, on by default")
        :ptype pace_delay_seconds: float
        """
        self._row_selector = row_selector
        self._listing_field_columns = listing_field_columns
        self._detail_link_column = detail_link_column
        self._detail_field_labels = detail_field_labels
        self._client = client
        self._max_rows = max_rows
        self._pace_delay_seconds = pace_delay_seconds

    @property
    def name(self) -> str:
        """Stable string key for this driver."""
        return "listing_detail"

    async def render(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        wait_for: str | None = None,
        capture_network: bool = False,
        nav_steps: list[NavStep] | None = None,
        results_path: str | None = None,
        fragment_field: str | None = None,
        link_selector: str | None = None,
        seen_urls: set[str] | None = None,
    ) -> RenderedPage:
        """Fetch the listing at *url*, resolve each row's own detail page, merge, and return a synthetic table.

        :param url: the listing page
        :ptype url: str
        :param timeout: seconds to wait for the listing fetch (each detail fetch gets its own budget, same value)
        :ptype timeout: float
        :param wait_for: accepted for interface conformance; not applicable (plain HTTP GET, no browser)
        :ptype wait_for: str | None
        :param capture_network: accepted for interface conformance; not applicable
        :ptype capture_network: bool
        :param nav_steps: accepted for interface conformance; not applicable
        :ptype nav_steps: list[NavStep] | None
        :param results_path: accepted for interface conformance; not applicable
        :ptype results_path: str | None
        :param fragment_field: accepted for interface conformance; not applicable
        :ptype fragment_field: str | None
        :param link_selector: accepted for interface conformance; not applicable
            (this driver's own *row_selector*/*detail_link_column* replace it)
        :ptype link_selector: str | None
        :return: one synthetic ``<table>`` page, one row per successfully-resolved record
        :rtype: RenderedPage
        :raises ListingDetailDriverError: the listing fetch itself fails outright
        """
        start = time.monotonic()
        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(
                timeout=timeout, follow_redirects=True, headers={"User-Agent": _DEFAULT_USER_AGENT}
            )
        try:
            try:
                listing_response = await client.get(url)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                log.warning(
                    "listing_detail: listing fetch failed", extra={"extra_data": {"url": url, "error": str(exc)}}
                )
                raise ListingDetailDriverError("transport", str(exc)) from exc
            if listing_response.status_code >= 400:
                raise ListingDetailDriverError(
                    "fetch_failed", f"HTTP {listing_response.status_code} fetching listing {url}"
                )

            soup = BeautifulSoup(listing_response.text, "html.parser")
            rows = soup.select(self._row_selector)[: self._max_rows]
            all_fields = set(self._listing_field_columns.values()) | set(self._detail_field_labels)

            merged_records: list[dict[str, Any]] = []
            for row_index, row in enumerate(rows):
                listing_fields = _extract_listing_row_fields(row, self._listing_field_columns)
                cells = row.find_all("td")
                detail_fields: dict[str, str] = {}
                if self._detail_link_column < len(cells):
                    link_tag = cells[self._detail_link_column].find("a")
                    href = link_tag.get("href") if link_tag else None
                    if href:
                        detail_url = urljoin(str(listing_response.url), str(href))
                        rate_limited = False
                        try:
                            detail_response = await client.get(detail_url)
                            if detail_response.status_code == 429:
                                rate_limited = True
                                log.warning(
                                    "listing_detail: one detail fetch was rate-limited, backing off",
                                    extra={"extra_data": {"url": detail_url}},
                                )
                            elif detail_response.status_code < 400:
                                detail_fields = _parse_definition_list(detail_response.text, self._detail_field_labels)
                            else:
                                log.warning(
                                    "listing_detail: one detail fetch returned an error status, using listing fields only",
                                    extra={"extra_data": {"url": detail_url, "status": detail_response.status_code}},
                                )
                        except (httpx.ConnectError, httpx.TimeoutException) as exc:
                            log.warning(
                                "listing_detail: one detail fetch failed, using listing fields only",
                                extra={"extra_data": {"url": detail_url, "error": str(exc)}},
                            )
                        # no delay after the LAST row -- nothing left to protect a server from
                        if row_index < len(rows) - 1:
                            delay = self._pace_delay_seconds + (
                                _RATE_LIMITED_EXTRA_DELAY_SECONDS if rate_limited else 0.0
                            )
                            if delay:
                                await asyncio.sleep(delay)

                merged: dict[str, Any] = {
                    field_name: (detail_fields.get(field_name) or listing_fields.get(field_name))
                    for field_name in all_fields
                }
                merged_records.append({k: v for k, v in merged.items() if v is not None})

            table_html = _records_to_synthetic_table(merged_records)
        finally:
            if owns_client:
                await client.aclose()

        return RenderedPage(
            html=f"<html><body>{table_html}</body></html>",
            status=listing_response.status_code,
            final_url=str(listing_response.url),
            timing_ms=(time.monotonic() - start) * 1000,
        )
