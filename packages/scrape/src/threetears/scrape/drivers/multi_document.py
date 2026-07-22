"""MultiDocumentDriver -- ``ScrapeDriver`` backend for a page that publishes a
LIST of links, each pointing at one document that is itself one complete
record (not a row in a table).

**Design (scrape-task-04, 2026-07-15):** Hawaii and West Virginia's real WARN
Act pages both publish one PDF per individual notice with no consolidated
current-period listing -- a materially different shape from every other
driver in this package, none of which turn "N separate document fetches"
into the one ``RenderedPage`` the eval loop expects.

Two independently pluggable axes, matching how heterogeneous the two real
targets driving this design already are:

- **Discovery strategy** -- how to find the N document URLs. ``link_selector``
  set: HTML mode (fetch the listing as a page, ``BeautifulSoup.select()`` for
  ``<a href>`` elements, e.g. Hawaii's real listing). ``results_path``/
  ``fragment_field`` set: JSON mode (fetch the listing as an API response,
  reuse :func:`~threetears.scrape.drivers.api._resolve_path` -- the exact same
  helper ``ApiDriver`` already uses, not a second JSON-path walker -- then
  read *fragment_field* off each record as the document URL, e.g. West
  Virginia's WordPress ``wp-json/wp/v2/media`` endpoint's own ``source_url``).
- **Fetch strategy** -- how to retrieve one document's bytes, constructor-
  injected. Plain :class:`~threetears.scrape.drivers.document.DocumentDriver`
  for a document reachable by a stateless HTTP GET (Hawaii). A browser-backed
  driver (:class:`~threetears.scrape.drivers.nodriver_download.
  NodriverDownloadDriver`) for one behind a real bot challenge a plain client
  can't pass (West Virginia).

Each document's synthetic HTML is wrapped in a delimiting
``<div class="notice" data-was-ocr="true|false">`` block (see
``extraction.NOTICE_DOCUMENT_CLASS``) and concatenated into one combined page --
``data-was-ocr`` carries the injected *document_driver*'s own
``RenderedPage.was_ocr`` flag through, since the combined-page HTML is the only
channel a per-document caller downstream (``eval_loop._run_per_document_extraction``)
has into what any one sub-document actually was.

**Revision (scrape-task-05, 2026-07-15):** the first version of this design
assumed ``extraction_strategy_type: regex`` (Pennsylvania/Michigan's existing
pattern) would apply unmodified once documents were combined -- live-verified
wrong. Regex/CSS strategies both assume one shared template repeated across
every row on a page, learned once and cached forever; West Virginia's real
documents are independently-worded business letters (a different employer's
own prose per notice), sharing no boilerplate a single pattern could ever
generalize across (verified: candidates each hardcoded one specific letter's
exact wording, matching 1 of 10 real documents). ``extraction_strategy_type:
per_document`` (``eval_loop.StrategyType``) is the correct fit instead: no
cached pattern, a fresh, independent LLM extraction call per document, every
poll -- ``extraction.split_notice_documents`` is this driver's own combined-page
convention's other half.

**Revision (scrape-task-06, 2026-07-16):** per_document's own OCR'd-text path
(``extraction.extract_fields_directly_chunked``) turned out unreliable on real
scanned WARN letters -- full-set live comparison against all 10 of West
Virginia's real documents: 2/10 complete records via OCR'd text vs. 10/10 via a
vision-capable model reading the page images directly (``anthropic/claude-sonnet-5``
via OpenRouter, zero new API key needed). Two senses, picked by document shape,
not one sense with a sad backup: a scanned document (``was_ocr=True``, this
driver's own ``data-was-ocr`` attribute) now routes to vision extraction
(``extraction.extract_fields_from_images``, reading the page images
:func:`~threetears.scrape.drivers.document._embed_ocr_page_images` already
embedded); a born-digital document stays on the fast/cheap OCR-free text path,
unchanged -- vision's real cost/latency is only paid where OCR was needed anyway.

**Revision (2026-07-16): document-level dedup.** Every prior version
re-fetched, re-OCR'd, and re-extracted every document ``render()`` discovered
on EVERY call -- with no memory of documents already processed on a prior
poll. For a real, continuously-polled target (WARN Act's daily cadence,
this driver's one real consumer), that means paying the full fetch/OCR/LLM
cost for the same historical documents indefinitely, not just once. `render`
now accepts an optional ``seen_urls: set[str] | None`` (part of every
driver's shared :meth:`~threetears.scrape.driver.ScrapeDriver.render`
signature, accepted-and-ignored by every other backend, same convention as
``link_selector``/``results_path``) -- a caller-owned, caller-persisted set
this driver only reads from and mutates in place, never stores itself
(keeping this module's own zero-domain-coupling design intact: it has no
opinion on how or where the caller durably persists the set between calls).
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from threetears.observe import get_logger

from ..driver import NavStep, RenderedPage, ScrapeDriver
from ..extraction import NOTICE_DOCUMENT_CLASS
from .api import ApiDriverError, _resolve_path

__all__ = ["MultiDocumentDriver", "MultiDocumentDriverError", "discover_links", "discover_links_labeled"]

log = get_logger(__name__)

#: Same fix, same reason as every other driver's own _DEFAULT_USER_AGENT.
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

#: No concrete evidence yet that a real target needs a different depth than
#: "cover a typical poll cycle's worth of new notices with margin" -- a fixed
#: driver default, not a per-target ScrapeTarget field, per this repo's own
#: "don't design for hypothetical future requirements" discipline. Revisit
#: if a real target ever needs otherwise.
_DEFAULT_MAX_DOCUMENTS = 20


class MultiDocumentDriverError(Exception):
    """Raised when listing discovery fails outright (not a single document's own fetch failure).

    Mirrors ``ApiDriverError``/``DocumentDriverError``'s ``code``/``message`` shape.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def discover_links_labeled(html: str, base_url: str, link_selector: str) -> list[tuple[str, str]]:
    """Resolve every ``link_selector``-matching ``<a href>`` to ``(absolute_url, anchor_text)``.

    The label-carrying sibling of :func:`discover_links`: the identical selector walk, but each entry
    also carries the anchor's **visible text** (whitespace-collapsed to a single line; ``""`` when the
    anchor has no text, e.g. an image-only link). A listing's anchor text is often the only
    human-readable name a linked document has ("November 2024 schedule"), and it is discarded the moment
    the page is re-parsed — so a caller that wants it (per-document provenance, a labeled inventory) must
    capture it on the same walk that resolves the URL. Ordering, de-duplication, host policy, and pattern
    filtering remain the caller's, exactly as for :func:`discover_links`.

    :param html: the listing page's HTML
    :ptype html: str
    :param base_url: the URL to resolve relative hrefs against (usually the listing's own URL)
    :ptype base_url: str
    :param link_selector: a CSS selector for the anchors to follow
    :ptype link_selector: str
    :return: ``(resolved_absolute_url, anchor_text)`` for every matched href, in document order
    :rtype: list[tuple[str, str]]
    """
    soup = BeautifulSoup(html, "html.parser")
    labeled: list[tuple[str, str]] = []
    for tag in soup.select(link_selector):
        href = tag.get("href")
        if href:
            label = " ".join(tag.get_text().split())
            labeled.append((urljoin(base_url, str(href)), label))
    return labeled


def discover_links(html: str, base_url: str, link_selector: str) -> list[str]:
    """Resolve every ``link_selector``-matching ``<a href>`` on a listing page to an absolute URL.

    A listing/index page carries links to the documents (or detail pages) a crawl then fetches.
    ``link_selector`` is a CSS selector for the anchors to follow; each matched ``href`` is resolved
    against ``base_url`` (typically the listing's own URL) so relative and absolute hrefs both come
    back absolute, in document order. Parsing uses ``html.parser`` (the same parser the extraction
    eval loop is authored against, so selectors resolve consistently). Ordering, de-duplication, host
    policy, and any pattern filtering are the caller's to apply — this returns the raw resolved set so
    each consumer can weigh them (a crawl inventory, a multi-document driver) without a baked-in policy.

    The URL-only projection of :func:`discover_links_labeled` (same walk; anchor text dropped) — a caller
    that also needs each anchor's text calls that variant instead.

    :param html: the listing page's HTML
    :ptype html: str
    :param base_url: the URL to resolve relative hrefs against (usually the listing's own URL)
    :ptype base_url: str
    :param link_selector: a CSS selector for the anchors to follow
    :ptype link_selector: str
    :return: every matched href resolved to an absolute URL, in document order
    :rtype: list[str]
    """
    return [url for url, _label in discover_links_labeled(html, base_url, link_selector)]


#: Backwards-compatible private alias — the function was internal (``_discover_links_html``) before it
#: was promoted to public :func:`discover_links`; kept so any in-tree reference keeps resolving.
_discover_links_html = discover_links


def _discover_links_json(data: Any, results_path: str, fragment_field: str) -> list[str]:
    """Resolve every record's *fragment_field* value to a document URL, reusing ``ApiDriver``'s own path walker."""
    try:
        records = _resolve_path(data, results_path)
    except ApiDriverError as exc:
        raise MultiDocumentDriverError(exc.code, exc.message) from exc
    return [str(record[fragment_field]) for record in records if isinstance(record, dict) and fragment_field in record]


class MultiDocumentDriver(ScrapeDriver):
    """``ScrapeDriver`` backed by a listing page/API plus N individual document fetches."""

    def __init__(
        self,
        *,
        document_driver: ScrapeDriver,
        client: httpx.AsyncClient | None = None,
        max_documents: int = _DEFAULT_MAX_DOCUMENTS,
    ) -> None:
        """
        :param document_driver: fetches and parses one document URL into a
            ``RenderedPage`` -- a plain ``DocumentDriver`` for a document
            reachable by a stateless HTTP GET, or a browser-backed driver
            for one behind a real bot challenge
        :ptype document_driver: ScrapeDriver
        :param client: an already-constructed httpx client to reuse for the
            LISTING fetch (test injection); a fresh one is created per call
            when omitted. Does not affect how *document_driver* fetches
            individual documents -- that's entirely its own concern.
        :ptype client: httpx.AsyncClient | None
        :param max_documents: how many of the listing's document URLs to
            fetch, newest-first (the same assumption every dated-listing
            target in this package already relies on)
        :ptype max_documents: int
        """
        self._document_driver = document_driver
        self._client = client
        self._max_documents = max_documents

    @property
    def name(self) -> str:
        """Stable string key for this driver."""
        return "multi_document"

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
        """Fetch the listing at *url*, discover document URLs, fetch and combine up to *max_documents*.

        Exactly one discovery mode must be configured: *link_selector* (HTML
        mode) or both *results_path* and *fragment_field* (JSON mode).

        :param url: the listing page or API endpoint
        :ptype url: str
        :param timeout: seconds to wait for the listing fetch before failing
        :ptype timeout: float
        :param wait_for: accepted for interface conformance; not applicable
            (the listing fetch is a plain HTTP GET, no browser)
        :ptype wait_for: str | None
        :param capture_network: accepted for interface conformance; not applicable
        :ptype capture_network: bool
        :param nav_steps: accepted for interface conformance; not applicable
        :ptype nav_steps: list[NavStep] | None
        :param results_path: JSON mode's dotted path to the records list
            (``""`` for a bare-array response) -- paired with *fragment_field*
        :ptype results_path: str | None
        :param fragment_field: JSON mode's field name holding each record's document URL
        :ptype fragment_field: str | None
        :param link_selector: HTML mode's CSS selector for document ``<a href>`` elements
        :ptype link_selector: str | None
        :param seen_urls: document URLs already fetched and extracted by a
            prior call -- skipped entirely here (no HTTP fetch, no OCR, no
            LLM extraction cost), rather than re-processed every poll
            (document-dedup capability, 2026-07-16). Mutated in place: every
            URL this call successfully fetches is added, whether or not it
            was already present, so the caller's own durable store reflects
            the full up-to-date seen set once this call returns. Applied
            BEFORE the ``max_documents`` cap, not after -- a raw listing
            that (briefly) exceeds ``max_documents`` entries would otherwise
            let already-seen documents crowd out genuinely new ones sitting
            just past the cap. ``None`` disables dedup entirely (every
            discovered document is (re-)fetched, matching this driver's
            pre-2026-07-16 behavior).
        :ptype seen_urls: set[str] | None
        :return: one combined page -- each successfully fetched document's
            content wrapped in its own delimiting block
        :rtype: RenderedPage
        :raises MultiDocumentDriverError: neither or both discovery modes
            configured, the listing fetch/parse fails outright, or JSON-mode
            path resolution fails
        """
        start = time.monotonic()
        html_mode = link_selector is not None
        json_mode = results_path is not None and fragment_field is not None
        if html_mode == json_mode:
            raise MultiDocumentDriverError(
                "missing_config",
                "exactly one of link_selector (HTML mode) or results_path+fragment_field (JSON mode) is required",
            )

        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers={"User-Agent": _DEFAULT_USER_AGENT})
        try:
            try:
                response = await client.get(url)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                log.warning("multi-document listing fetch failed", extra={"extra_data": {"url": url, "error": str(exc)}})
                raise MultiDocumentDriverError("transport", str(exc)) from exc
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code >= 400:
            raise MultiDocumentDriverError("fetch_failed", f"HTTP {response.status_code} fetching {url}")

        if html_mode:
            assert link_selector is not None  # narrowed by html_mode above
            document_urls = discover_links(response.text, str(response.url), link_selector)
        else:
            assert results_path is not None and fragment_field is not None  # narrowed by json_mode above
            try:
                data = response.json()
            except ValueError as exc:
                raise MultiDocumentDriverError("invalid_json", f"response from {url} is not valid JSON: {exc}") from exc
            document_urls = _discover_links_json(data, results_path, fragment_field)

        # Applied before the max_documents cap (see this method's own
        # docstring): filtering to unseen URLs first, then capping, so a
        # raw listing that (briefly) exceeds max_documents entries can't let
        # already-seen documents crowd out genuinely new ones just past the
        # cap. seen_urls=None keeps this driver's pre-2026-07-16 behavior
        # (every discovered document (re-)fetched) byte-for-byte unchanged.
        candidate_urls = (
            [u for u in document_urls if u not in seen_urls][: self._max_documents]
            if seen_urls is not None
            else document_urls[: self._max_documents]
        )
        skipped_seen = len(document_urls[: self._max_documents]) - len(candidate_urls) if seen_urls is not None else 0
        if skipped_seen:
            log.debug(
                "multi-document: skipped %d already-seen document(s)",
                skipped_seen,
                extra={"extra_data": {"listing_url": url}},
            )

        bodies: list[str] = []
        for doc_url in candidate_urls:
            try:
                page = await self._document_driver.render(doc_url, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- one bad document must never sink the others, mirrors _regenerate_row_recipe's own per-candidate resilience
                log.warning(
                    "multi-document: one document fetch failed, skipping",
                    extra={"extra_data": {"url": doc_url, "error": str(exc)}},
                )
                continue
            if seen_urls is not None:
                # Marked only on a successful fetch -- a failed fetch (above)
                # must stay eligible for retry on the next poll, not get
                # silently skipped forever (same invariant every checkpoint
                # cursor in this codebase already holds).
                seen_urls.add(doc_url)
            body_start = page.html.find("<body")
            body_open_end = page.html.find(">", body_start) + 1 if body_start != -1 else -1
            body_end = page.html.rfind("</body>")
            inner = page.html[body_open_end:body_end] if body_start != -1 and body_end != -1 else page.html
            was_ocr = "true" if page.was_ocr else "false"
            bodies.append(f'<div class="{NOTICE_DOCUMENT_CLASS}" data-was-ocr="{was_ocr}">{inner}</div>')

        combined_html = "<html><body>" + "\n".join(bodies) + "</body></html>"
        return RenderedPage(
            html=combined_html,
            status=response.status_code,
            final_url=str(response.url),
            timing_ms=(time.monotonic() - start) * 1000,
        )
