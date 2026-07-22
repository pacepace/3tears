"""NetworkCaptureDriver -- ``ScrapeDriver`` backend for a page whose real data
arrives via an authenticated, in-session XHR/fetch call, not a statelessly
GET-able JSON API.

**Design (2026-07-15, Oklahoma's own live proof case):** Chunk 15's network
detection captures every XHR/fetch call during a real browser render, and
Chunk 20's ``ApiDriver`` queries a JSON API directly -- but Oklahoma's real
WARN listing (a Salesforce Experience Cloud / Aura page) has no such API:
the actual data arrives via ``POST .../sfsites/aura``, authenticated by
session cookies and a CSRF-bearing Aura context only a real browser session
can produce, and the response wraps the record list several JSON levels deep
inside a batched ``actions[]`` envelope (only one of several actions in a
given response actually carries the WARN records; the others are unrelated
menu/config/telemetry payloads). ``ApiDriver``'s plain unauthenticated GET +
fixed dotted ``results_path`` can't reach or parse this.

Mirrors ``ApiDriver``'s "JSON -> synthetic HTML -> same unmodified eval loop"
shape, but sourced differently: this driver renders *url* through a real,
injected browser driver with ``capture_network`` forced on, then searches
every captured call's JSON body for the largest list of same-shaped
records -- no fixed path required, since a real data table reliably
outnumbers incidental dict-lists (nav menus, feature-flag maps) elsewhere in
the same response. Each record's own field names become the synthetic
table's column headers, exactly like ``DocumentDriver``'s XLSX/CSV table
conversion -- the eval loop's real column-name discovery, not a hand-mapped
schema.
"""

from __future__ import annotations

import html as html_lib
import json
import time
from typing import Any

from threetears.observe import get_logger

from ..driver import NavStep, RenderedPage, ScrapeDriver

__all__ = ["NetworkCaptureDriver", "NetworkCaptureDriverError"]

log = get_logger(__name__)

#: A real data table reliably has many more rows than an incidental
#: dict-list elsewhere in the same JSON payload (a 2-3 item nav menu, a
#: handful of feature flags) -- live-found against Oklahoma's own Aura
#: response, which has several small decoy dict-lists alongside the 217-row
#: WARN table. Not a hard requirement on real record count, just a floor
#: below which a match is more likely noise than data.
_MIN_RECORDS = 2


class NetworkCaptureDriverError(Exception):
    """Raised when no captured network call yields a usable record list.

    Mirrors ``ApiDriverError``/``DocumentDriverError``'s ``code``/``message`` shape.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _find_largest_record_list(node: Any) -> list[dict[str, Any]] | None:
    """Recursively search *node* for the largest list of dicts, anywhere in the tree.

    No fixed path -- a real batched API response (Salesforce Aura's
    ``actions[]`` envelope, or any similarly-nested shape) can bury the
    actual record list at an unpredictable depth, alongside smaller
    unrelated dict-lists. Picking the single LARGEST list of dicts found
    anywhere is a simple, general, live-verified-sufficient heuristic: a
    real data table reliably has far more rows than incidental config/menu
    lists in the same payload.

    :param node: a parsed JSON value (dict, list, or scalar)
    :ptype node: Any
    :return: the largest qualifying list found, or ``None`` if none has at
        least :data:`_MIN_RECORDS` dict items
    :rtype: list[dict[str, Any]] | None
    """
    best: list[dict[str, Any]] | None = None

    def _walk(current: Any) -> None:
        nonlocal best
        if isinstance(current, list):
            if len(current) >= _MIN_RECORDS and all(isinstance(item, dict) for item in current):
                if best is None or len(current) > len(best):
                    best = current
            for item in current:
                _walk(item)
        elif isinstance(current, dict):
            for value in current.values():
                _walk(value)

    _walk(node)
    return best


def _records_to_html(records: list[dict[str, Any]]) -> str:
    """Convert a flat list of JSON record dicts into a synthetic HTML table.

    Columns are the union of every key across every record, in first-seen
    order -- records need not share identical keys (Oklahoma's own data:
    some WARN entries have ``OESC_Employer_City__c``, others don't). A
    record missing a given column renders an empty cell, mirroring
    ``DocumentDriver``'s XLSX/CSV pad-to-header-width discipline.
    """
    columns: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    thead = "<tr>" + "".join(f"<th>{html_lib.escape(col)}</th>" for col in columns) + "</tr>"
    body_rows = []
    for record in records:
        cells = "".join(f"<td>{html_lib.escape(str(record.get(col, '')))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return "<html><body><table>" + thead + "".join(body_rows) + "</table></body></html>"


class NetworkCaptureDriver(ScrapeDriver):
    """``ScrapeDriver`` backed by a captured XHR/fetch call from a real browser render.

    Requires a real browser-capable *inner* driver (``NodriverSidecarDriver``
    or ``CamoufoxDriver``) -- unlike ``ApiDriver``/``DocumentDriver``, there
    is no sensible zero-config default here (the whole point is reaching
    data only a real, cookie-bearing session can produce).
    """

    def __init__(self, inner: ScrapeDriver) -> None:
        """
        :param inner: a real browser-capable driver to render *url* through,
            with ``capture_network`` forced on regardless of what the caller passes
        :ptype inner: ScrapeDriver
        """
        self._inner = inner

    @property
    def name(self) -> str:
        """Stable string key for this driver."""
        return "network_capture"

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
        """Render *url* through the inner driver and synthesize HTML from its largest captured record list.

        :param url: the page to fetch
        :ptype url: str
        :param timeout: seconds to wait for the render before failing
        :ptype timeout: float
        :param wait_for: optional CSS selector to wait for before considering
            the page settled, passed straight through to the inner driver
        :ptype wait_for: str | None
        :param capture_network: accepted for interface conformance; this
            driver always captures network calls regardless of the value given
        :ptype capture_network: bool
        :param nav_steps: ordered browser actions, passed straight through
            to the inner driver
        :ptype nav_steps: list[NavStep] | None
        :param results_path: accepted for interface conformance; not
            applicable (this driver auto-detects the record list, see
            :func:`_find_largest_record_list`)
        :ptype results_path: str | None
        :param fragment_field: accepted for interface conformance; not applicable
        :ptype fragment_field: str | None
        :param link_selector: accepted for interface conformance; not applicable
        :ptype link_selector: str | None
        :return: synthetic HTML built from the largest captured JSON record
            list, with the inner render's real status/final_url/timing
        :rtype: RenderedPage
        :raises NetworkCaptureDriverError: if no captured network call's
            body contains a usable record list
        """
        start = time.monotonic()
        page = await self._inner.render(
            url, timeout=timeout, wait_for=wait_for, capture_network=True, nav_steps=nav_steps
        )

        best: list[dict[str, Any]] | None = None
        for call in page.network_calls:
            try:
                data = json.loads(call.body)
            except (json.JSONDecodeError, ValueError) as exc:
                log.debug(
                    "network_capture: skipping non-JSON call body for %s: %s",
                    call.url,
                    exc,
                )
                continue
            candidate = _find_largest_record_list(data)
            if candidate is not None and (best is None or len(candidate) > len(best)):
                best = candidate

        if best is None:
            raise NetworkCaptureDriverError(
                "no_record_list_found",
                f"none of the {len(page.network_calls)} captured network call(s) for {url} "
                f"contained a list of {_MIN_RECORDS}+ same-shaped JSON records",
            )

        html = _records_to_html(best)
        return RenderedPage(
            html=html,
            status=page.status,
            final_url=page.final_url,
            timing_ms=(time.monotonic() - start) * 1000,
        )
