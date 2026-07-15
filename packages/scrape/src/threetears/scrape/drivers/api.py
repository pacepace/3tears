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
"""

from __future__ import annotations

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

    :raises ApiDriverError: if any path segment is missing, or the final
        value isn't a list
    """
    current: Any = data
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            raise ApiDriverError("bad_results_path", f"{path!r}: no key {segment!r} in response at this level")
        current = current[segment]
    if not isinstance(current, list):
        raise ApiDriverError("bad_results_path", f"{path!r} does not resolve to a list (got {type(current).__name__})")
    return current


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
        :param results_path: dotted JSON path to the list of per-record objects (required)
        :ptype results_path: str | None
        :param fragment_field: which field within each per-record object
            holds the HTML/text fragment to concatenate (required)
        :ptype fragment_field: str | None
        :return: the concatenated fragments as synthetic HTML
        :rtype: RenderedPage
        :raises ApiDriverError: on a transport failure, a non-2xx HTTP
            response, missing *results_path*/*fragment_field*, a response
            that isn't valid JSON, or *results_path* not resolving to a list
        """
        if not results_path or not fragment_field:
            raise ApiDriverError(
                "missing_config", "results_path and fragment_field are required for ApiDriver.render()"
            )

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
        fragments = [
            str(record[fragment_field]) for record in records if isinstance(record, dict) and fragment_field in record
        ]
        html = "<html><body>" + "\n".join(fragments) + "</body></html>"
        return RenderedPage(
            html=html,
            status=response.status_code,
            final_url=str(response.url),
            timing_ms=(time.monotonic() - start) * 1000,
        )
