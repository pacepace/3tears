"""ScrapeDriver — pure-Python ABC for pluggable browser-rendering backends.

Zero non-stdlib imports. Lift-ready: move this module (and drivers/) to a
3tears package without any changes to driver code — mirrors
``src/faidh/intake/rate_limit/strategy.py``'s zero-faidh-imports discipline.

Deliberately excludes anything backend-specific (no CDP handles, no
Firefox-specific objects): a driver takes a URL and returns a
``RenderedPage`` carrying only plain data. This genericness is what keeps
the nodriver sidecar boundary "arm's length" under FSF's own aggregation
test (see ``scrape-api-contract.md``), not merely a style preference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["NavStep", "NetworkCall", "RenderedPage", "ScrapeDriver"]

#: The closed set of browser actions a ``NavStep`` can describe. Kept small
#: and generic on purpose -- a per-target sequence of these is enough to
#: drive a search form, click into a result page, or page through a listing,
#: without the core needing to know anything about what's being searched for
#: (multi-step navigation, 2026-07-14).
NavStepAction = Literal["click", "fill", "wait_for", "wait_ms", "scroll_into_view", "scroll_page", "evaluate"]


@dataclass(frozen=True)
class NavStep:
    """One browser action a driver performs before the page is considered ready.

    Multi-step navigation capability (2026-07-14): some real target pages
    (e.g. a state's WARN listing gated behind a search form, or paginated
    into a second page) can't be reached by a single ``render(url)`` call --
    the driver needs to be "driven" through an intermediate interaction
    first. A ``ScrapeTarget``'s ``nav_steps`` (see ``collections.py``) is an
    ordered list of these, executed in sequence after the initial navigation
    to *url* and before the existing ``wait_for``/settle-wait logic -- a
    per-target config knob, the same category as ``wait_for``/``multi_row``,
    not a per-state code hack. The eval loop's own AI-driven extraction still
    runs unmodified on whatever HTML the driven-to page produces.

    - ``click``: wait for *selector* to appear, then click it.
    - ``fill``: wait for *selector* to appear, clear it, then type *value*.
    - ``wait_for``: wait for *selector* to appear (a pause between two other
      steps, e.g. after a click that triggers an async page update).
    - ``wait_ms``: a fixed delay, for a step with no reliable selector to
      wait on instead.
    - ``scroll_into_view``: wait for *selector* to appear, then scroll it
      into the viewport -- some widgets defer their own data fetch until
      they're actually visible on screen (an ``IntersectionObserver``-style
      lazy-render gate), not merely present in the DOM the way ``wait_for``
      already checks. Added because two real capture attempts against Google
      Trends' own TIMESERIES widget, which only waited and never scrolled,
      never observed that widget's data call fire at all (2026-07-17).
    - ``scroll_page``: scroll the whole page down by *value* percent of the
      viewport height (a real synthesized scroll gesture, not a DOM API call
      -- some lazy-render triggers watch real scroll/wheel events, not just
      element visibility), or a driver-chosen default when *value* is
      ``None``. Needs no target-specific selector at all -- the selector-free
      sibling of ``scroll_into_view``, for a target whose real lazy-loading
      container isn't known/guessable up front.
    - ``evaluate``: run the JS expression in *value* in the page's own
      context and record its (JSON-serializable) return value onto
      :attr:`RenderedPage.eval_results`, in step order -- ground truth read
      of a page's own client-side state (a controller method's real return
      value, a computed property) instead of guessing what a request body or
      DOM structure would produce. General-purpose: any target whose real
      answer lives in in-page JS state, not just one target's app framework.
    """

    action: NavStepAction
    selector: str | None = None
    value: str | None = None
    ms: int | None = None


@dataclass
class NetworkCall:
    """One captured XHR/fetch call whose response body looks like JSON.

    Network/API-detection capability (2026-07-14): a JS widget that renders
    its own data client-side (nothing in the static/rendered HTML at all --
    e.g. Michigan's Coveo shadow-DOM widget, Georgia's third-party embed)
    is often calling a plain JSON backend API to get that data. Capturing
    those calls lets a caller (or the eval loop) discover and query that API
    directly instead of needing to scrape rendered HTML that may not even
    contain the data at all.
    """

    url: str
    method: str
    status: int
    content_type: str
    body: str


@dataclass
class RenderedPage:
    """Plain-data result of rendering one URL through a driver backend."""

    html: str
    status: int
    final_url: str
    timing_ms: float
    #: Empty unless the caller passed ``capture_network=True`` to
    #: :meth:`ScrapeDriver.render` -- capturing costs an extra round-trip per
    #: request, so it's opt-in, not collected by default.
    network_calls: list[NetworkCall] = field(default_factory=list)
    #: True when this page's own document (only meaningful for
    #: :class:`~threetears.scrape.drivers.document.DocumentDriver`/
    #: :class:`~threetears.scrape.drivers.nodriver_download.NodriverDownloadDriver`)
    #: needed OCR fallback -- a scanned/image PDF, not born-digital text. Always
    #: ``False`` for every other driver (an HTML page render has no such concept).
    #: Consumed by :class:`~threetears.scrape.drivers.multi_document.
    #: MultiDocumentDriver` (scrape-task-06) to mark which combined-page documents
    #: need vision-based extraction rather than the faster/cheaper text path --
    #: see ``eval_loop._run_per_document_extraction``.
    was_ocr: bool = False
    #: One entry per ``NavStep(action="evaluate", ...)`` in the caller's
    #: ``nav_steps``, in step order -- each entry is that step's JS
    #: expression's own (JSON-serializable) return value. Empty when no
    #: ``evaluate`` step ran.
    eval_results: list[Any] = field(default_factory=list)


class ScrapeDriver(ABC):
    """Abstract base for pluggable browser-rendering backends.

    Implementations render a URL and return the resulting page as plain
    data (:class:`RenderedPage`) — no backend-specific handles leak across
    this boundary, so callers can swap backends without caring which one
    rendered the page.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable string key for this driver (e.g. ``"nodriver"``)."""

    @abstractmethod
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
        """Render *url* and return the resulting page.

        :param url: the page to fetch
        :ptype url: str
        :param timeout: seconds to wait for the render before failing
        :ptype timeout: float
        :param wait_for: optional CSS selector to wait for before
            considering the page rendered; ``None`` means no wait beyond
            normal navigation completion
        :ptype wait_for: str | None
        :param capture_network: when true, capture every XHR/fetch call
            whose response body looks like JSON (see :class:`NetworkCall`)
        :ptype capture_network: bool
        :param nav_steps: ordered browser actions (see :class:`NavStep` for
            the full action set) executed after the initial navigation to
            *url* and before *wait_for*'s settle-wait -- drives the browser
            to a page not reachable by a bare ``render(url)`` call (a search
            form, a second page in a listing), or reads back the page's own
            JS state (``evaluate``); ``None``/empty means no interaction
            beyond plain navigation
        :ptype nav_steps: list[NavStep] | None
        :param results_path: dotted JSON path to the list of per-record
            objects in a JSON API response (e.g. ``"Results"``) -- only
            meaningful to :class:`~threetears.scrape.drivers.api.ApiDriver`
            (network/API-query capability, 2026-07-14) and, in its JSON
            discovery mode, :class:`~threetears.scrape.drivers.
            multi_document.MultiDocumentDriver`; every other backend accepts
            and ignores it, per this contract's own "accept the full
            signature, use what you need" precedent (``wait_for`` on
            ``DocumentDriver``)
        :ptype results_path: str | None
        :param fragment_field: which field within each per-record JSON
            object holds the HTML/text fragment to concatenate into a
            synthetic page (``ApiDriver``), or the document URL to fetch
            (``MultiDocumentDriver``'s JSON discovery mode) -- only
            meaningful to those two backends
        :ptype fragment_field: str | None
        :param link_selector: CSS selector matching document links on a
            listing page -- only meaningful to
            :class:`~threetears.scrape.drivers.multi_document.
            MultiDocumentDriver`'s HTML discovery mode (multi-document
            capability, 2026-07-15); every other backend accepts and
            ignores it
        :ptype link_selector: str | None
        :param seen_urls: document URLs the caller already has real data
            for -- only meaningful to :class:`~threetears.scrape.drivers.
            multi_document.MultiDocumentDriver` (document-dedup capability,
            2026-07-16): a URL present here is skipped entirely (no fetch,
            no OCR, no LLM extraction cost) rather than re-processed on
            every poll. Mutated in place -- every URL this call successfully
            fetches (whether or not it was already present) is added, so the
            caller's own durable store (a growable set has no natural
            "return value" otherwise) reflects the full up-to-date seen set
            after the call returns. ``None`` disables the skip entirely
            (matches every driver's pre-2026-07-16 behavior); every other
            backend accepts and ignores it.
        :ptype seen_urls: set[str] | None
        :return: the rendered page's HTML, status, final URL, timing, (if
            requested) captured network calls, and any ``evaluate`` step
            results
        :rtype: RenderedPage
        :raises Exception: a backend-specific error (its own ``code``/
            ``message`` shape) when a nav step can't be executed -- e.g. a
            selector that never appears
        """
