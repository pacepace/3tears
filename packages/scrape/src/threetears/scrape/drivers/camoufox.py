"""CamoufoxDriver -- in-process ``ScrapeDriver`` backend using Camoufox (a
stealth-patched Firefox build).

The second driver backend (Chunk 6). Its purpose is explicitly to pressure-
test whether ``ScrapeDriver`` is a genuinely pluggable interface or secretly
shaped around ``NodriverSidecarDriver``'s sidecar-only assumptions (HTTP
transport, JSON error bodies) -- not because a v1 target currently needs
Camoufox's evasion ceiling.

Unlike ``NodriverSidecarDriver``, this launches its own browser process
in-process, no separate container: safe per ``scrape-api-contract.md``'s
license read (verified via GitHub's license API) -- the ``camoufox`` Python
wrapper package is MIT-licensed, and the browser binary it launches is
MPL-2.0 (file-level copyleft only, no network-copyleft clause), unlike
nodriver's AGPL-3.0 boundary that requires the sidecar's separate-process
isolation. This module has zero faidh imports (see ``scrape/__init__.py``).

**Pinned dependency note (2026-07-14):** ``playwright`` is pinned below
1.61 in ``pyproject.toml`` -- 1.61 added an ``isMobile`` viewport field
Camoufox's bundled Firefox patch (Juggler protocol) doesn't recognize yet,
breaking every browser launch. Confirmed live against a real browser launch;
matches a currently-open upstream bug (``daijro/camoufox#653``). Revisit
the pin once camoufox ships a fix.
"""

from __future__ import annotations

import time
from typing import Any

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Response as PlaywrightResponse
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from threetears.observe import get_logger

from ..driver import NavStep, NetworkCall, RenderedPage, ScrapeDriver

__all__ = ["CamoufoxDriver", "CamoufoxDriverError"]

log = get_logger(__name__)

#: Same bound and same resource-type filter as the nodriver sidecar's own
#: capture_network implementation (services/nodriver-sidecar/main.py) --
#: kept in sync deliberately so the two drivers behave identically for a
#: caller that doesn't care which one rendered the page.
_API_RESOURCE_TYPES = frozenset({"xhr", "fetch"})
_MAX_NETWORK_CALLS = 30
_MAX_NETWORK_BODY_BYTES = 500_000

#: Encodings tried, in order, when a captured body is not valid UTF-8. cp1252 is the practical
#: second guess for western-language public-sector APIs (its 0xA9 is the copyright sign that turns
#: up in agency footers and boilerplate). Deliberately does NOT end in a never-failing codec such as
#: latin-1: silently mojibaking a body would hand downstream shape-detection corrupted field values
#: that look valid, which is worse than skipping one unreadable response.
_BODY_FALLBACK_ENCODINGS = ("cp1252",)

#: Matches the nodriver sidecar's own default for a ``scroll_page`` step with
#: no *value* -- a quarter of the viewport height.
_DEFAULT_SCROLL_PAGE_AMOUNT = 25


def _decode_captured_body(raw: bytes) -> str | None:
    """Decode a captured response body that is not valid UTF-8, or ``None`` if no candidate reads it.

    Only reached after ``Response.text()`` -- which assumes UTF-8 -- has already failed, so UTF-8 is
    not retried here.

    :param raw: the response's raw bytes
    :ptype raw: bytes
    :return: the decoded body, or ``None`` when every candidate encoding fails
    :rtype: str | None
    """
    for encoding in _BODY_FALLBACK_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError, LookupError:
            continue
    return None


class CamoufoxDriverError(Exception):
    """Raised when a Camoufox render fails.

    Mirrors ``NodriverSidecarError``'s ``code``/``message`` shape so
    callers that log/handle scrape-driver failures don't need per-backend
    cases, even though the two backends fail for structurally different
    reasons (HTTP transport vs. in-process browser navigation).
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class CamoufoxDriver(ScrapeDriver):
    """``ScrapeDriver`` backed by an in-process Camoufox (Firefox) browser.

    Launches lazily on the first :meth:`render` call and stays alive across
    calls -- a fresh page (tab) is opened and closed per call, never the
    same tab reused across requests (the same "new tab per request" lesson
    Chunk 1's nodriver sidecar work already learned the hard way: reusing a
    single tab across requests severs the browser's control connection
    after the first request).
    """

    def __init__(self, *, headless: bool = True, browser: Any | None = None) -> None:
        """
        :param headless: launch Camoufox headless (default) or with a visible window.
        :ptype headless: bool
        :param browser: an already-launched Playwright-shaped ``Browser`` to reuse
            (test injection); a real Camoufox browser is launched lazily on the
            first :meth:`render` call when omitted.
        :ptype browser: Any | None
        """
        self._headless = headless
        self._browser = browser
        self._owns_browser = browser is None
        self._camoufox: Any | None = None

    @property
    def name(self) -> str:
        """Stable string key for this driver."""
        return "camoufox"

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
        """Render *url* through a fresh, in-process Camoufox page.

        :param url: the page to fetch
        :ptype url: str
        :param timeout: seconds to wait for navigation (and, if given,
            *wait_for*) before failing
        :ptype timeout: float
        :param wait_for: optional CSS selector to wait for before
            considering the page rendered; ``None`` means no wait beyond
            normal navigation completion
        :ptype wait_for: str | None
        :param capture_network: when true, capture every XHR/fetch call
            whose response body looks like JSON (see :class:`NetworkCall`)
        :ptype capture_network: bool
        :param nav_steps: ordered browser actions executed after navigation
            and before *wait_for*'s settle-wait
        :ptype nav_steps: list[NavStep] | None
        :param results_path: accepted for interface conformance; not
            applicable (only :class:`~threetears.scrape.drivers.api.ApiDriver` uses it)
        :ptype results_path: str | None
        :param fragment_field: accepted for interface conformance; not
            applicable (only :class:`~threetears.scrape.drivers.api.ApiDriver` uses it)
        :ptype fragment_field: str | None
        :param link_selector: accepted for interface conformance; not
            applicable (only :class:`~threetears.scrape.drivers.multi_document.MultiDocumentDriver` uses it)
        :ptype link_selector: str | None
        :return: the rendered page's HTML, status, final URL, timing, and
            (if requested) captured network calls
        :rtype: RenderedPage
        :raises CamoufoxDriverError: on navigation timeout/failure, a
            *wait_for* selector that never appears within *timeout*, or a
            nav step that can't be executed (``code="nav_step_failed"``)
        """
        browser = await self._ensure_browser()
        timeout_ms = timeout * 1000
        start = time.monotonic()
        page = await browser.new_page()
        captured_responses: list[PlaywrightResponse] = []
        if capture_network:
            # Only records the Response object here -- fetching bodies happens
            # below, in this function's own awaited control flow, after the
            # page has settled. Mirrors the nodriver sidecar's identical
            # design decision (services/nodriver-sidecar/main.py's _render):
            # an async page.on() handler that awaited response.text() itself
            # would run concurrently with the rest of this function with no
            # guarantee it finishes before we return, a real race.
            page.on(
                "response",
                lambda resp: (
                    captured_responses.append(resp) if resp.request.resource_type in _API_RESOURCE_TYPES else None
                ),
            )
        try:
            try:
                response = await page.goto(url, timeout=timeout_ms, wait_until="load")
            except PlaywrightTimeoutError as exc:
                log.warning("camoufox navigation timeout", extra={"extra_data": {"url": url, "error": str(exc)}})
                raise CamoufoxDriverError("navigation_timeout", str(exc)) from exc
            except PlaywrightError as exc:
                log.warning("camoufox navigation failed", extra={"extra_data": {"url": url, "error": str(exc)}})
                raise CamoufoxDriverError("navigation_failed", str(exc)) from exc

            if nav_steps:
                await self._execute_nav_steps(page, nav_steps, timeout_ms)

            if wait_for is not None:
                try:
                    await page.wait_for_selector(wait_for, timeout=timeout_ms)
                except PlaywrightTimeoutError as exc:
                    log.warning(
                        "camoufox wait_for selector never appeared",
                        extra={"extra_data": {"url": url, "wait_for": wait_for, "error": str(exc)}},
                    )
                    raise CamoufoxDriverError("wait_for_timeout", str(exc)) from exc

            html = await page.content()
            network_calls = []
            for resp in captured_responses[:_MAX_NETWORK_CALLS]:
                try:
                    try:
                        body = await resp.text()
                    except UnicodeDecodeError:
                        # Playwright's Response.text() decodes as UTF-8 unconditionally. JSON is
                        # required to be UTF-8 (RFC 8259), but real government APIs serve cp1252
                        # anyway -- and UnicodeDecodeError is a ValueError, NOT a PlaywrightError,
                        # so it escaped this loop's per-response guard and aborted the ENTIRE render:
                        # one mis-encoded response discarded every other captured call on the page.
                        # Re-read the raw bytes and decode them tolerantly instead of losing the page.
                        body = _decode_captured_body(await resp.body())
                        if body is None:
                            log.debug("camoufox network capture: undecodable body for %s -- skipped", resp.url)
                            continue
                    if len(body) > _MAX_NETWORK_BODY_BYTES:
                        continue
                    stripped = body.lstrip()
                    if not (stripped.startswith("{") or stripped.startswith("[")):
                        continue  # not JSON-shaped -- not a useful "backend API" signal
                    headers = await resp.all_headers()
                except PlaywrightError as exc:
                    log.debug("camoufox network capture: body/header fetch failed for %s: %s", resp.url, exc)
                    continue
                network_calls.append(
                    NetworkCall(
                        url=resp.url,
                        method=resp.request.method,
                        status=resp.status,
                        content_type=headers.get("content-type", ""),
                        body=body,
                    )
                )
            result = RenderedPage(
                html=html,
                status=response.status if response is not None else 0,
                final_url=page.url,
                timing_ms=(time.monotonic() - start) * 1000,
                network_calls=network_calls,
            )
        finally:
            await page.close()
        return result

    async def _execute_nav_steps(self, page: Any, nav_steps: list[NavStep], timeout_ms: float) -> None:
        """Drive *page* through *nav_steps* in order, before *wait_for*'s settle-wait.

        Playwright's own ``click``/``fill``/``wait_for_selector`` already
        auto-wait for the selector to appear (and, for click/fill, become
        actionable) before acting -- the same "wait for a real page to
        respond" semantics the nodriver sidecar's ``tab.select()`` retry loop
        provides, no extra polling needed here.
        """
        for i, step in enumerate(nav_steps):
            try:
                if step.action == "click":
                    await page.click(step.selector, timeout=timeout_ms)
                elif step.action == "fill":
                    await page.fill(step.selector, step.value or "", timeout=timeout_ms)
                elif step.action == "wait_for":
                    await page.wait_for_selector(step.selector, timeout=timeout_ms)
                elif step.action == "wait_ms":
                    await page.wait_for_timeout(step.ms or 0)
                elif step.action == "scroll_into_view":
                    await page.locator(step.selector).scroll_into_view_if_needed(timeout=timeout_ms)
                elif step.action == "scroll_page":
                    try:
                        amount = int(step.value) if step.value else _DEFAULT_SCROLL_PAGE_AMOUNT
                    except ValueError as exc:
                        raise CamoufoxDriverError(
                            "nav_step_failed", f"nav_step[{i}] (scroll_page): value {step.value!r} is not an int percentage"
                        ) from exc
                    viewport = page.viewport_size or {"height": 1080}
                    await page.mouse.wheel(0, viewport["height"] * (amount / 100))
                else:
                    raise CamoufoxDriverError("nav_step_failed", f"nav_step[{i}] unsupported action {step.action!r}")
            except PlaywrightTimeoutError as exc:
                log.warning(
                    "camoufox nav step failed",
                    extra={"extra_data": {"step": i, "action": step.action, "selector": step.selector}},
                )
                raise CamoufoxDriverError("nav_step_failed", f"nav_step[{i}] ({step.action}): {exc}") from exc

    async def close(self) -> None:
        """Release the launched browser process, if this driver launched one.

        A no-op when a *browser* was injected at construction (the caller
        owns its lifecycle then) or when no browser has been launched yet.
        Not required before process exit, but frees resources for a
        long-lived process that constructs many short-lived drivers.
        """
        if self._owns_browser and self._camoufox is not None:
            await self._camoufox.__aexit__(None, None, None)
            self._camoufox = None
            self._browser = None

    async def _ensure_browser(self) -> Any:
        """Lazily launch (once) and return the Playwright-shaped ``Browser``."""
        if self._browser is None:
            # Deferred: camoufox's import chain pulls numpy/lxml/browserforge/etc,
            # a real cost tests that always inject *browser* should never pay.
            from camoufox.async_api import AsyncCamoufox

            self._camoufox = AsyncCamoufox(headless=self._headless)
            self._browser = await self._camoufox.__aenter__()
        return self._browser
