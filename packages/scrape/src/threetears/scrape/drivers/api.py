"""ApiDriver -- ``ScrapeDriver`` backend for a raw JSON API response.

**Design (2026-07-14, network/API-query capability):** the direct "query the
api directly if needed" follow-through to Chunk 15's network/API-detection
capability, which stopped at *capturing* a page's real backend API call
without ever *querying* it as a first-class fetch mechanism. Michigan's real
WARN listing (a Sitecore XA search endpoint, confirmed live via
``capture_network``) is the concrete driver: its JSON response wraps a real,
already-rendered HTML fragment per record (``Results[].Html``), not raw
field values.

Mirrors ``DocumentDriver``'s own "fetch raw bytes, produce a synthetic page,
feed it through the same unmodified eval loop" shape: this driver fetches
*url* via a plain HTTP GET (the API itself, not a browser-rendered page),
parses the JSON body, walks *results_path* to the list of per-record
objects, and concatenates each record's *fragment_field* value into one
``<html><body>`` document -- the CSS-selector or regex extraction strategy
(whichever the target's own fragment shape needs) then runs against it
completely unmodified.

**Structured-record mode (2026-07-15, Texas's real Socrata open-data WARN
endpoint):** not every JSON API pre-renders an HTML/text fragment per
record -- a plain open-data API returns flat objects with several named
fields (``notice_date``, ``job_site_name``, ``county_name``, ...), the same
shape a CSV row or spreadsheet cell would have. ``fragment_field=None`` (or
``""``, treated identically) switches to this mode: each record's own
key/value pairs become one synthetic ``<table>`` row (header row from the
UNION of every record's own keys, first-seen order, one ``<tr>`` per record
after it) -- the same "turn tabular data into a real ``<table>`` so the
unmodified CSS-selector eval loop can run against it" principle
``DocumentDriver``'s markdown-pipe-table conversion already applies to
PDF/XLSX tables, applied to JSON records instead. ``results_path`` also
accepts ``""`` (empty string, distinct from the required-but-unset ``None``)
to mean "the response root itself is the records list" -- Socrata endpoints
return a bare JSON array, not one wrapped under a named key.
"""

from __future__ import annotations

import html as html_lib
import time
from typing import Any

import httpx
from threetears.observe import get_logger

from ..driver import NavStep, RenderedPage, ScrapeDriver

__all__ = ["ApiDriver", "ApiDriverError"]

log = get_logger(__name__)

#: Live-found (Michigan's real Sitecore XA search API, 2026-07-14): a plain
#: httpx client's default User-Agent gets a flat 403 from the CDN/WAF in
#: front of the endpoint; a genuine browser UA passes cleanly. Only applied
#: to a client this driver constructs itself -- an injected *client* (test
#: injection, or a caller with its own header policy) is used exactly as given.
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class ApiDriverError(Exception):
    """Raised when a JSON API fetch or response-shape resolution fails.

    Mirrors ``DocumentDriverError``/``NodriverSidecarError``'s ``code``/``message`` shape.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _resolve_path(data: Any, path: str) -> list[Any]:
    """Walk a dotted path (e.g. ``"Results"``, ``"data.records"``) to a list.

    Deliberately minimal -- plain dict-key traversal only, no array
    indexing or wildcards. Real JSON APIs vary too much for a generic path
    language to be worth building speculatively; a future target needing
    more than "walk N dict keys to find the records list" is real, separate
    scope, not something to guess at now.

    ``path == ""`` is a special case, not the first step of an empty-string
    key traversal -- it means the response ROOT is already the records list
    (a bare JSON array, e.g. Socrata open-data endpoints), skipping the
    key-walk entirely.

    :raises ApiDriverError: if any path segment is missing, or the final
        value isn't a list
    """
    if path == "":
        if not isinstance(data, list):
            raise ApiDriverError(
                "bad_results_path",
                f"empty results_path expects the response root to be a list (got {type(data).__name__})",
            )
        return data
    current: Any = data
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            raise ApiDriverError("bad_results_path", f"{path!r}: no key {segment!r} in response at this level")
        current = current[segment]
    if not isinstance(current, list):
        raise ApiDriverError("bad_results_path", f"{path!r} does not resolve to a list (got {type(current).__name__})")
    return current


def _records_to_synthetic_table(records: list[Any]) -> str:
    """Turn a list of flat record dicts into one synthetic HTML ``<table>``.

    Header row from the UNION of every record's own keys (first-seen order
    preserved across the whole list, not just the first record) -- Critic
    caught (chunk review) that taking columns from the first record alone
    silently drops a key that only appears on a later record, for
    heterogeneous record shapes a real API can genuinely return. One
    ``<tr>`` per record after the header, values in that same key order;
    a record missing a key gets an empty cell rather than shifting columns.
    Mirrors :func:`~threetears.scrape.drivers.document.document_text_to_html`'s
    "turn tabular data into a real ``<table>`` for the unmodified CSS-selector
    eval loop" principle, applied to JSON records instead of a markdown
    pipe-table.

    :param records: per-record dicts (non-dict entries are skipped)
    :ptype records: list[Any]
    :return: a ``<table>...</table>`` fragment; empty if no dict records
    :rtype: str
    """
    dict_records = [r for r in records if isinstance(r, dict)]
    if not dict_records:
        return ""
    columns: list[str] = []
    seen: set[str] = set()
    for record in dict_records:
        for key in record:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    header = "<tr>" + "".join(f"<th>{html_lib.escape(str(col))}</th>" for col in columns) + "</tr>"
    rows = [
        "<tr>" + "".join(f"<td>{html_lib.escape(str(record.get(col, '')))}</td>" for col in columns) + "</tr>"
        for record in dict_records
    ]
    return "<table>" + header + "".join(rows) + "</table>"


class ApiDriver(ScrapeDriver):
    """``ScrapeDriver`` backed by a raw JSON API response.

    Every real dependency is a plain HTTP GET (no session, no nonce, no
    browser) -- a target needing those first (Georgia's own
    ``admin-ajax.php`` nonce, live-verified during Chunk 19) isn't a fit for
    this driver; ``nodriver``'s real browser session already handles that
    case via ``wait_for``/``nav_steps``.
    """

    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        """
        :param client: an already-constructed httpx client to reuse (test
            injection); a fresh one is created per call when omitted.
        :ptype client: httpx.AsyncClient | None
        """
        self._client = client

    @property
    def name(self) -> str:
        """Stable string key for this driver."""
        return "api"

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
        """Fetch *url*'s JSON response and concatenate per-record fragments into synthetic HTML.

        :param url: the JSON API endpoint
        :ptype url: str
        :param timeout: seconds to wait for the HTTP fetch before failing
        :ptype timeout: float
        :param wait_for: accepted for interface conformance; not applicable
            (a single plain HTTP GET, not a rendered page)
        :ptype wait_for: str | None
        :param capture_network: accepted for interface conformance; not applicable
        :ptype capture_network: bool
        :param nav_steps: accepted for interface conformance; not applicable (no browser to drive)
        :ptype nav_steps: list[NavStep] | None
        :param results_path: dotted JSON path to the list of per-record objects
            (required; pass ``""`` if the response root itself is the list)
        :ptype results_path: str | None
        :param fragment_field: which field within each per-record object holds
            the HTML/text fragment to concatenate; ``None`` or ``""`` (treated
            identically) switches to structured-record mode (each record's
            own keys become a synthetic
            ``<table>`` row instead -- see this module's own docstring)
        :ptype fragment_field: str | None
        :param link_selector: accepted for interface conformance; not
            applicable (only :class:`~threetears.scrape.drivers.multi_document.MultiDocumentDriver` uses it)
        :ptype link_selector: str | None
        :return: the concatenated fragments, or the synthetic table (structured mode), as HTML
        :rtype: RenderedPage
        :raises ApiDriverError: on a transport failure, a non-2xx HTTP
            response, missing *results_path*, a response that isn't valid
            JSON, or *results_path* not resolving to a list
        """
        if results_path is None:
            raise ApiDriverError("missing_config", "results_path is required for ApiDriver.render()")

        start = time.monotonic()
        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(
                timeout=timeout, follow_redirects=True, headers={"User-Agent": _DEFAULT_USER_AGENT}
            )
        try:
            try:
                response = await client.get(url)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                log.warning("api driver transport failure", extra={"extra_data": {"url": url, "error": str(exc)}})
                raise ApiDriverError("transport", str(exc)) from exc
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code >= 400:
            raise ApiDriverError("fetch_failed", f"HTTP {response.status_code} fetching {url}")

        try:
            data = response.json()
        except ValueError as exc:
            raise ApiDriverError("invalid_json", f"response from {url} is not valid JSON: {exc}") from exc

        records = _resolve_path(data, results_path)
        if not fragment_field:
            # Critic-caught (chunk review): an empty string (distinct from None -- e.g. a
            # nullable api_fragment_field TEXT column holding "" rather than NULL) must
            # trigger the same structured-record mode None does, not silently look up a
            # literal "" key and return an empty body.
            body = _records_to_synthetic_table(records)
        else:
            fragments = [
                str(record[fragment_field])
                for record in records
                if isinstance(record, dict) and fragment_field in record
            ]
            body = "\n".join(fragments)
        html = "<html><body>" + body + "</body></html>"
        return RenderedPage(
            html=html,
            status=response.status_code,
            final_url=str(response.url),
            timing_ms=(time.monotonic() - start) * 1000,
        )
