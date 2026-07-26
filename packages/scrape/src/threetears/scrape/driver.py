"""ScrapeDriver -- pure-Python ABC for pluggable browser-rendering backends.

Zero non-stdlib imports -- the discipline that let this module (and
``drivers/``) move out of the application it was first written in as a plain
directory move, with no changes to driver code at all.

Deliberately excludes anything backend-specific (no CDP handles, no
Firefox-specific objects): a driver takes a URL and returns a
``RenderedPage`` carrying only plain data. This genericness is what keeps
the nodriver sidecar boundary "arm's length" under FSF's own aggregation
test -- the two processes exchange plain data over a documented HTTP
contract and neither is built around the other's internals -- not merely a
style preference. See ``docs/scrape-lift-design.md`` (D4) for why that
isolation is treated as structural rather than maintainer-dependent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

__all__ = ["NavStep", "NetworkCall", "RenderedPage", "ScrapeDriver"]

#: The closed set of browser actions a ``NavStep`` can describe. Kept small
#: and generic on purpose -- a per-target sequence of these is enough to
#: drive a search form, click into a result page, or page through a listing,
#: without the core needing to know anything about what's being searched for
#: (multi-step navigation, 2026-07-14).
NavStepAction = Literal["click", "fill", "wait_for", "wait_ms", "scroll_into_view", "scroll_page", "evaluate"]

#: Default advice when a driver drops a solve. Wrong for the sidecar-backed download driver,
#: which is why :meth:`ScrapeDriver._warn_dropped_session_state` takes an override.
_SIDECAR_REMEDY = "Use the nodriver sidecar driver to reuse a solved session."

#: Ceiling on remembered origins per driver, so the dedupe set cannot grow without bound in a
#: long-lived process. Reached, it clears wholesale, and the cost is one repeated warning per
#: site still being scraped -- not one in total, which an earlier version of this comment
#: claimed. Wholesale rather than evicting the oldest because the alternative is tracking
#: recency for a set whose whole job is to be forgotten occasionally.
_MAX_WARNED_ORIGINS = 512


def _origin_of(url: str) -> str:
    """``scheme://host[:port]`` for *url*, or the url itself when it has no parseable origin.

    Deliberately NOT shared with :func:`threetears.scrape.robots._origin_of`, whose contract is
    ``str | None``. The two answer the same question for opposite purposes, and the difference
    is the return type. Robots must distinguish "no usable origin" so it can decline to apply a
    site's rules to something that is not a site; here the value is only ever a dedupe key, and
    ``None`` would collapse every unparseable url into one bucket -- so a batch of odd urls
    would report the first and silence the rest, which is the failure this dedupe exists to
    avoid. Falling back to the url keeps each one distinct.

    Sharing them would mean one of the two callers handling a case it has no answer for. This
    module also keeps a zero-non-stdlib-import discipline, so importing from ``robots`` would
    cost more than the six lines it saved.
    """
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url
    return f"{parts.scheme}://{parts.netloc}"


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
    #: MultiDocumentDriver` to mark which combined-page documents
    #: need vision-based extraction rather than the faster/cheaper text path --
    #: see ``eval_loop._run_per_document_extraction``.
    was_ocr: bool = False
    #: One entry per ``NavStep(action="evaluate", ...)`` in the caller's
    #: ``nav_steps``, in step order -- each entry is that step's JS
    #: expression's own (JSON-serializable) return value. Empty when no
    #: ``evaluate`` step ran.
    eval_results: list[Any] = field(default_factory=list)
    #: Which exit this page was fetched through, when the backend knows. ``None`` from a
    #: backend that has no concept of one. Reported by the fetcher rather than assumed by the
    #: caller, so a dropped proxy argument surfaces as a mismatch rather than as a result
    #: quietly stamped with an exit that was never used.
    egress: str | None = None


class ScrapeDriver(ABC):
    """Abstract base for pluggable browser-rendering backends.

    Implementations render a URL and return the resulting page as plain
    data (:class:`RenderedPage`) -- no backend-specific handles leak across
    this boundary, so callers can swap backends without caring which one
    rendered the page.
    """

    #: Origins this instance has already reported a dropped solve for. The state is per
    #: instance, but the dedupe KEY is the origin, which is the whole point -- keying on the
    #: instance itself was the bug. `ScrapeTool` builds its driver map once and reuses it for
    #: the life of the process, so deduping per instance meant per PROCESS: the first target
    #: warned and every later one was rendered logged-out, in exactly the silence this exists
    #: to prevent.
    #: Per render was the opposite failure, a warning per document across a whole listing.
    #: An origin is the unit a human's solve actually belongs to, so it is the unit here.
    _warned_dropped_origins: set[str] | None = None

    #: What to tell an operator instead. PUBLIC and a class attribute: subclasses are meant to
    #: override it, which the repo's underscore rule rightly forbids for a private name -- an
    #: underscore attribute is implementation detail of the class that declares it. A class
    #: attribute rather than a call-site keyword so it is a property of the DRIVER: the
    #: download driver's own remedy is different, and a
    #: keyword passed at the point of call made that difference invisible to any test that did
    #: not go through that exact line.
    dropped_solve_remedy: str = _SIDECAR_REMEDY

    def _warn_dropped_session_state(self, url: str, log: Any) -> None:
        """Report ONCE that this driver is discarding a human's exported session.

        Silence is the failure this prevents: a caller hands over a session a person spent real
        time solving, gets a successful render back, and learns nothing until extraction fails
        on a login wall and the target is escalated to a human who already did the work.

        **Once per ORIGIN**, which is the only cardinality that is wrong in neither direction.
        Per render is a storm: :class:`MultiDocumentDriver` forwards a solve to its inner
        document driver once per document, so one listing emitted a warning per document, and a
        warning that repeats that way trains its reader to filter it out. Per driver INSTANCE is
        silence: `ScrapeTool` builds its driver map once and reuses it for the whole process, so
        the first dropped solve would warn and every later target would be rendered logged-out
        with nothing said. An origin is what a human's solve actually belongs to, so it is the
        unit that makes "this site's solve was discarded" true exactly once.

        On the base class rather than a module function so the per-instance set of already-
        reported origins has somewhere to live, and so every backend gets this by inheriting
        rather than by each author remembering the pattern -- the previous version was added to
        whichever driver a review happened to name, and the others stayed silent.

        :param url: the url being rendered without the solve
        :ptype url: str
        :param log: the calling module's own logger, so the record carries its name
        :ptype log: Any
        """
        origin = _origin_of(url)
        if self._warned_dropped_origins is None:
            # Lazily built per instance, so no backend has to remember to initialise it and
            # the class-level default is never mutated into a set shared by every driver.
            self._warned_dropped_origins = set()
        if origin in self._warned_dropped_origins:
            return
        if len(self._warned_dropped_origins) >= _MAX_WARNED_ORIGINS:
            # Bounded rather than unbounded: a long-lived process scraping a wide set of sites
            # would otherwise hold one string per origin forever, which is the same leak the
            # robots gate had to fix. Clearing wholesale re-warns once for each site still in
            # play, which is the honest cost -- noisier than "at most one", and still bounded.
            self._warned_dropped_origins.clear()
        self._warned_dropped_origins.add(origin)
        log.warning(
            "%s driver: session_state was supplied but this driver cannot apply it; rendering %s "
            "unauthenticated. %s (reported once per site)",
            self.name,
            url,
            self.dropped_solve_remedy,
        )

    @property
    def egress(self) -> object | None:
        """The exit this driver's fetches leave by, or ``None`` for the default route.

        Concrete rather than abstract, and returning ``None``, because most backends have no
        concept of an exit and should not be made to declare one. What it buys is that asking
        the question is always valid: ``ScrapeTool`` inspects this to warn about a split
        configuration -- drivers proxied while its own ``robots.txt`` read is not -- and a
        check that had to ``getattr`` its way to an undeclared name would fail silently the
        moment that name changed, which is the worst failure mode a security check can have.

        Typed ``object | None`` rather than ``EgressDriver``: this module keeps a
        zero-non-stdlib-import discipline, which is what lets it be imported from anywhere
        without dragging a dependency along. ``object | None`` needs no import, accepts every
        override, and supports the only operation anyone performs on it -- asking whether it
        is there. ``Any`` would satisfy the same constraint while typing nothing.
        """
        return None

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
        session_state: dict[str, Any] | None = None,
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
        :param session_state: a human's previously cleared browser state -- cookies and origin
            storage exported from a HITL session -- to apply BEFORE navigating, so the request
            that would have been challenged carries the credential that clears it. Raw and
            already opened; sealing is the caller's business, not a driver's. Every non-browser
            backend accepts and ignores it, per this protocol's established convention: a
            driver that cannot restore a browser session has nothing to do with one, and the
            alternative is a capability flag every caller has to branch on
        :ptype session_state: dict[str, Any] | None
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
