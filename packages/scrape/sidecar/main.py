"""nodriver sidecar -- thin HTTP wrapper around nodriver + Xvfb.

Runs inside its own AGPL-3.0-licensed container (see LICENSE). Never
imported as a Python library from 3tears-scrape's (MIT) tree; consumers
only ever talk to this process over HTTP, via the ``POST /v1/render`` and
``POST /v1/download`` endpoints defined below -- those request/response
models ARE the contract, and ``tests/test_render_contract.py`` is what
holds them to it. ``entrypoint.sh`` starts Xvfb and points
``DISPLAY`` at it before this process starts -- nodriver launches Chromium
with ``headless=False`` against that virtual display, per nodriver's own
documented guidance for headless-machine deployments (real headed Chromium
under Xvfb has better real-world site compatibility than ``headless=True``,
matching the product brief's rationale for choosing nodriver first).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any, NamedTuple

import hitl
import nodriver as uc
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from nodriver.core.connection import ProtocolException
from pydantic import BaseModel

log = logging.getLogger("nodriver_sidecar")
logging.basicConfig(level=logging.INFO)

CHROMIUM_PATH = os.environ.get("CHROMIUM_PATH", "/usr/bin/chromium")

# Egress: this container's DEFAULT exit, as a --proxy-server value (e.g.
# "socks5://tor:9050"). Set by the deployment, because which exits exist is
# deployment knowledge -- this container only needs to be told which one to use.
#
# Applied at browser launch, which Chromium takes process-wide: every render that
# does not ask for something else leaves by this one.
#
# A render CAN ask for something else. `RenderRequest.egress_proxy` renders in its
# own browser context, and `Target.createBrowserContext` accepts its own
# `proxyServer` -- verified against this image's own CDP bindings. An earlier
# version of this comment said a per-request proxy would advertise a capability
# Chromium does not have; that was true of the command-line flag and wrong about
# contexts, which is why per-target egress is a request field rather than a second
# container.
EGRESS_PROXY = os.environ.get("EGRESS_PROXY") or None
# `None` when the deployment configured no exit at all, because that is what the consumers of
# this value mean by it: `ScrapeTargetHealth.last_egress` and `TargetCircuit` both treat `None`
# as "nobody said" and reserve a name for a stated choice. Stamping "direct" here would fill
# every row of an unconfigured deployment with the one value that convention exists to withhold,
# and a reader could no longer tell a deployment that chose the default route from one that
# never considered the question. A deployment that HAS chosen it says so with `EGRESS_NAME`, or
# by passing `DirectEgress` per request.
#
# When a proxy is set without a name the name is unknown rather than absent, and "unnamed" says
# so. A literal placeholder like "direct" would be indistinguishable from a real exit called
# that, and telling exits apart is the whole point of the name.
EGRESS_NAME = os.environ.get("EGRESS_NAME") or ("unnamed" if EGRESS_PROXY else None)

# Browser-forced-download capability (scrape-task-04, 2026-07-15): a fixed profile
# directory (rather than nodriver's own auto-generated temp one) so the Preferences
# file below is written before uc.start() launches Chromium and reliably applies to
# the ONE persistent browser instance this process runs for its whole lifetime.
_USER_DATA_DIR = "/tmp/nodriver-sidecar-profile"

# Live-verified (2026-07-15, real West Virginia Cloudflare-protected PDF): Chrome's
# own built-in PDF viewer intercepts a direct navigation to a PDF response BEFORE
# Browser.setDownloadBehavior's "allow" has any effect, unless this preference is
# set -- and it must be written into the profile's Preferences file before Chromium
# starts; a CLI flag (--disable-extensions) does NOT touch it, since the built-in
# PDF viewer is a Chrome component, not a regular extension. Only affects a direct
# navigation TO a PDF response -- confirmed live this does not affect normal HTML
# page rendering (a page that merely links to a PDF triggers nothing).
_CHROME_PREFERENCES = {"plugins": {"always_open_pdf_externally": True}}

# Download polling (mirrors _render's own settle-wait shape): Chrome writes a
# ".crdownload" extension while a download is still in progress -- only a file
# WITHOUT that suffix is complete. Bounded by the caller's own request timeout,
# not a separate constant, matching /v1/render's own timeout-is-the-caller's-budget
# contract.
_DOWNLOAD_POLL_INTERVAL_SECONDS = 0.5

# Target-list propagation race (live-reproduced, 2026-07-15): browser.create_context()'s
# own internal target lookup (a single 0.5s sleep then one `self.targets` check) is
# NOT reliably enough time for a freshly created target to appear -- reproduced live,
# StopIteration inside nodriver's own create_context. Bounded retry with an explicit
# browser.update_targets() call each attempt, the same "retry the whole find-then-act
# sequence from scratch" shape _select_with_retry already established for a different
# CDP timing race in this same file.
_TAB_LOOKUP_ATTEMPTS = 10
_TAB_LOOKUP_DELAY_SECONDS = 0.2

# Network-capture bounds (2026-07-14, network/API-detection capability): a
# page can fire dozens of XHR/fetch calls -- capped so one render can't blow
# up the response payload or the render's own wall-clock (each captured body
# is one extra CDP round-trip). Tuned generously, not scientifically: enough
# to almost certainly include the real data-bearing call on a normal page,
# small enough that a chatty page doesn't turn a render into a slow-loris.
_MAX_NETWORK_CALLS = 30
_MAX_NETWORK_BODY_BYTES = 500_000

# Cold-start mitigation (2026-07-14): the browser's very first real render can
# race navigation and return the pre-load empty shell even with _render's own
# explicit settle wait (reproduced live) -- a first-request-only phenomenon
# once the browser has completed one real render cycle. A warm-up render at
# startup (see _warm_up) absorbs that cost here, once, so every /v1/render
# consumer never has to retry-tolerate it themselves.
#
# Must be a real network navigation, not "about:blank": the race is between a
# real page load and get_content(), so a target with no network round-trip at
# all wouldn't exercise the same timing path and could report "ready" without
# actually proving the browser survives it (Critic finding, this review).
# example.com is IANA-reserved for exactly this kind of use -- small, fast,
# stable, no rate-limit/availability risk from a single hit at startup.
_WARMUP_URL = "https://example.com"
_WARMUP_ATTEMPTS = 3
_WARMUP_RETRY_DELAY_SECONDS = 2.0
_WARMUP_TIMEOUT_SECONDS = 15.0

_browser: Any = None
#: True only after a real warm-up render completes (or exhausts retries and
#: fails open) -- distinct from "_browser is not None" (the browser process
#: started, but hasn't proven it can render yet). /healthz reports "ok" only
#: once this is True.
_ready: bool = False


class NavStepModel(BaseModel):
    """One browser action, wire shape -- mirrors ``threetears.scrape.driver.NavStep``.

    Multi-step navigation capability (2026-07-14): see that dataclass's
    docstring for the supported actions and why they're driven
    deterministically (per-target config) rather than LLM-decided per fetch.
    """

    action: str
    selector: str | None = None
    value: str | None = None
    ms: int | None = None


class RenderRequest(BaseModel):
    url: str
    timeout: float = 30.0
    wait_for: str | None = None
    #: Network/API-detection capability (2026-07-14): when true, capture every
    #: XHR/fetch call the page makes whose response body looks like JSON, so a
    #: caller can discover a backend API a JS widget calls internally instead
    #: of scraping its rendered (or unrendered -- shadow DOM, client-only
    #: state) HTML. False by default: it's an extra CDP round-trip per
    #: request, real cost for a capability most renders don't need.
    capture_network: bool = False
    #: Multi-step navigation capability (2026-07-14): ordered actions
    #: executed after the initial navigation to *url* and before *wait_for*'s
    #: settle-wait -- drives the browser to a page not reachable by a bare
    #: navigation (a search form, a second page in a listing).
    nav_steps: list[NavStepModel] | None = None
    #: A human's previously cleared browser state, applied before navigating so the request
    #: that would be challenged carries the credential that clears it. Raw, not sealed: this
    #: container holds no key and never has.
    session_state: dict[str, Any] | None = None
    #: Exit for THIS request only, as a proxy url. Renders in its own browser context, so two
    #: targets in one container can leave by two different exits -- which is what makes egress
    #: a per-target choice. Omitted uses the container's own route.
    egress_proxy: str | None = None
    #: Name recorded against the result, so "walled" can be told apart from "walled from this
    #: exit". Free-form because the names are the deployment's.
    #:
    #: Echoed back on the response ONLY when sent alongside :attr:`egress_proxy`. Sent alone it
    #: names an exit this render did not take -- there is no proxy to take it -- so the response
    #: reports the container's own exit instead, and a consumer that assumed an echo would
    #: record a route that was never used. The two travel together or neither is meaningful.
    #:
    #: The echo exists so a caller records what THIS sidecar reported rather than what it sent:
    #: without it, a request whose proxy argument was dropped is indistinguishable from one that
    #: was honoured. Reported, not observed -- see :attr:`RenderResponse.egress`.
    egress_name: str | None = None


class NetworkCall(BaseModel):
    url: str
    method: str
    status: int
    content_type: str
    body: str


class RenderResponse(BaseModel):
    html: str
    status: int
    final_url: str
    timing_ms: float
    network_calls: list[NetworkCall] = []
    #: The exit this render was CONFIGURED to leave by: the request's own name when it selected
    #: one, otherwise the container's. A caller stamps THIS against its result rather than what
    #: it sent, so a dropped proxy argument shows up as a mismatch instead of silently recording
    #: an exit that was never asked for.
    #:
    #: Configured, not observed. This value is derived from the request, so a per-context proxy
    #: Chromium accepted and then ignored is still reported under the name it was asked for.
    #: Nothing in this process can tell the difference; confirming traffic genuinely leaves by
    #: an exit needs an observer outside it. Said here rather than only at the construction
    #: site, because an API consumer reads the model and never sees the handler.
    #:
    #: ``null`` means no exit was configured, which is a different fact from choosing the
    #: default route -- that choice is reported as ``direct``.
    egress: str | None = None
    #: One entry per real ``evaluate`` nav step, in step order -- see
    #: ``threetears.scrape.driver.NavStep``'s own docstring.
    eval_results: list[Any] = []


class DownloadRequest(BaseModel):
    """Browser-forced-download capability (scrape-task-04, 2026-07-15) -- a distinct
    contract from RenderRequest/RenderResponse, not another optional field bolted onto
    them: the response shape is fundamentally different (raw file bytes, not HTML)."""

    url: str
    timeout: float = 30.0


class DownloadResponse(BaseModel):
    status: int
    filename: str
    content_type: str
    content_base64: str
    timing_ms: float


def _is_main_frame_document(event: uc.cdp.network.ResponseReceived, main_frame_id: str) -> bool:
    """True for the top-level navigation's own document response, not a sub-resource.

    ``frame_id`` on ``Network.responseReceived`` is empty for a request
    "fetched from worker" (per the CDP field's own doc comment) and non-empty
    for everything else -- comparing to *main_frame_id* excludes iframes and
    other subframes, ``ResourceType.DOCUMENT`` excludes images/scripts/XHRs
    fired by the page itself.
    """
    return event.type_ == uc.cdp.network.ResourceType.DOCUMENT and str(event.frame_id) == main_frame_id


class _RenderResult(NamedTuple):
    html: str
    final_url: str
    status: int
    network_calls: list[dict[str, Any]]
    eval_results: list[Any]


class NavStepError(Exception):
    """Raised when a nav step can't be executed (selector never appeared).

    Caught by the ``/v1/render`` endpoint and reported as a distinct
    ``nav_step_failed`` error code (422) -- a bad/stale selector in a
    target's config, not a sidecar crash (``driver_crash``, 502) or a plain
    navigation timeout (504).
    """

    def __init__(self, step_index: int, action: str, message: str) -> None:
        self.step_index = step_index
        self.action = action
        super().__init__(f"nav_step[{step_index}] ({action}): {message}")


#: Live-reproduced against Maine's real WARN search form (2026-07-14): an
#: element resolved by tab.select() can go stale by the time click()/
#: send_keys() actually runs on it, OR even inside a later tab.select() call
#: itself ("Could not find node with given id [code: -32000]") -- the page
#: is still settling/re-rendering out from under CDP's node-id bookkeeping.
#: Confirmed non-deterministic (network/render timing): 6 back-to-back
#: renders in one debug run hit it zero times, a near-identical run hit it
#: on the 4th; live against the real container, a plain wait_for selector
#: search (no nav_steps involved at all) hit the identical error -- this is
#: not nav_steps-specific, it's inherent to tab.select() on a still-settling
#: page. _select_with_retry (below) is shared by nav_steps' click/fill/
#: wait_for actions AND _render()'s own final wait_for settle-wait, so both
#: get the same transient-failure tolerance the page's real-world timing
#: variance turns out to need.
_NAV_STEP_RETRY_ATTEMPTS = 3
_NAV_STEP_RETRY_DELAY_SECONDS = 0.5

#: nodriver's own ``Tab.scroll_down`` default (25 == a quarter of the
#: viewport height) -- used when a ``scroll_page`` step doesn't specify
#: *value*.
_DEFAULT_SCROLL_PAGE_AMOUNT = 25


async def _select_with_retry(tab: Any, selector: str, timeout: float, action: str, value: str | None = None) -> Any:
    """Find *selector*, then perform *action* (``"click"``/``"fill"``/
    ``"wait_for"``/``"scroll_into_view"``) on the result -- retrying the
    whole find-then-act sequence from scratch (a fresh ``tab.select()``
    re-queries the live DOM) when a stale CDP node id is hit mid-sequence.
    See the module-level comment above for the live reproduction this is
    built from.

    :raises ProtocolException: if every retry attempt still hits the race
    :return: the found element, or ``None`` if *selector* never appeared
        (a real "not found", not the same failure mode as the race above)
    """
    last_exc: ProtocolException | None = None
    el = None
    for attempt in range(1, _NAV_STEP_RETRY_ATTEMPTS + 1):
        try:
            el = await tab.select(selector, timeout=timeout)
            if el is not None:
                if action == "click":
                    await el.click()
                elif action == "fill":
                    await el.clear_input()
                    await el.send_keys(value or "")
                elif action == "scroll_into_view":
                    await el.scroll_into_view()
                # "wait_for": nothing further to do once the element is found
            last_exc = None
            break
        except ProtocolException as exc:
            last_exc = exc
            if attempt < _NAV_STEP_RETRY_ATTEMPTS:
                await tab.sleep(_NAV_STEP_RETRY_DELAY_SECONDS)
    if last_exc is not None:
        raise last_exc
    return el


async def _execute_nav_steps(tab: Any, nav_steps: list[NavStepModel], timeout: float, eval_results: list[Any]) -> None:
    """Drive *tab* through *nav_steps* in order, before the caller's own settle-wait.

    Each step gets the full outer *timeout* to find its selector, matching
    ``wait_for``'s own per-render (not per-step) timeout budget -- simpler
    than apportioning a shared budget across steps, and a nav step search is
    the same class of "wait for a real page to respond" operation ``wait_for``
    already gets the full timeout for.

    *eval_results* is mutated in place -- one entry appended per real
    ``evaluate`` step, matching ``ScrapeDriver.render``'s own ``seen_urls``
    mutate-in-place precedent for a per-call accumulator that isn't this
    function's own return value.
    """
    for i, step in enumerate(nav_steps):
        if step.action == "wait_ms":
            await tab.sleep((step.ms or 0) / 1000)
            continue
        if step.action == "scroll_page":
            try:
                amount = int(step.value) if step.value is not None else _DEFAULT_SCROLL_PAGE_AMOUNT
            except ValueError as exc:
                raise NavStepError(i, step.action, f"value {step.value!r} is not an int percentage") from exc
            await tab.scroll_down(amount)
            continue
        if step.action == "evaluate":
            try:
                eval_results.append(await tab.evaluate(step.value or "", return_by_value=True))
            except ProtocolException as exc:
                raise NavStepError(i, step.action, str(exc)) from exc
            continue
        if step.action not in ("click", "fill", "wait_for", "scroll_into_view"):
            raise NavStepError(i, step.action, f"unsupported action {step.action!r}")
        try:
            el = await _select_with_retry(tab, step.selector, timeout, step.action, step.value)
        except ProtocolException as exc:
            raise NavStepError(i, step.action, str(exc)) from exc
        if el is None:
            raise NavStepError(i, step.action, f"selector {step.selector!r} not found")


#: XHR/fetch are the resource types a JS widget's own data calls show up as
#: -- excludes images/scripts/stylesheets/fonts/documents, which are never
#: the "backend API" a page is calling for its data.
_API_RESOURCE_TYPES = frozenset({uc.cdp.network.ResourceType.XHR, uc.cdp.network.ResourceType.FETCH})

#: Anti-JSON-hijacking prefixes real APIs prepend before the actual JSON body
#: (the response is deliberately not valid JSON/JS on its own until stripped,
#: a defense against a cross-origin <script> tag executing it) -- e.g. Google's
#: own internal APIs (Trends' explore/widgetdata endpoints, live-verified
#: 2026-07-17). Stripped before the JSON-shape check below so a real API using
#: this standard convention isn't silently dropped as "not JSON-shaped."
_JSON_HIJACK_PREFIXES: tuple[str, ...] = (")]}'",)


async def _render(
    url: str,
    wait_for: str | None,
    *,
    capture_network: bool = False,
    nav_steps: list[NavStepModel] | None = None,
    timeout: float = 30.0,
    session_state: dict[str, Any] | None = None,
    egress_proxy: str | None = None,
) -> _RenderResult:
    """Navigate to *url*, optionally drive it through *nav_steps*, wait for a
    selector, and return the rendered page.

    ``status`` -- fixed 2026-07-14: a plain ``browser.get(url,
    new_tab=True)`` gives no way to observe the real top-level HTTP response
    status -- nodriver's ``Tab`` exposes no ``.status`` attribute (checked
    live against nodriver 0.50.3's ``Tab``/CDP bindings), and the browser
    never raises on a successful 404/500 page load, it just renders the
    error page. Wiring CDP ``Network.responseReceived`` requires the domain
    enabled and the handler registered BEFORE navigation starts to avoid
    missing the event -- ``browser.get(url, new_tab=True)`` bakes the URL
    into ``Target.createTarget`` itself, so navigation begins before we'd
    ever get a `Tab` handle back to enable Network on. Opening a blank tab
    first (near-instant, no network round trip, so it doesn't reintroduce
    the cold-start race ``_warm_up`` already handles separately), enabling
    Network + registering the handler, THEN navigating via
    ``cdp.page.navigate`` closes that race deterministically rather than
    hoping the local CDP round-trip usually wins it.

    ``new_tab=True`` (for the initial blank tab) is still load-bearing:
    ``browser.get(url)`` without it reuses the browser's single default tab,
    and closing that tab after every request (to avoid leaking a tab per
    fetch) severs the CDP connection every subsequent request depends on --
    reproduced live: request 1 succeeds, request 2+ fail with "no close
    frame received or sent". Opening a throwaway tab per request and closing
    only that one avoids it.
    """
    # With a human's session state, the render runs in an ISOLATED browser context rather
    # than the shared profile. Applying one target's cleared cookies to the profile every
    # other target also renders through would hand them to sites that never earned them, and
    # would mix two targets' sessions for the same origin. The context is disposed in the
    # same `finally` that closes the tab, so the state lives exactly as long as the fetch.
    render_context_id: Any = None
    if egress_proxy is not None:
        # An exit for this request alone. Its own context, disposed with the tab, so nothing
        # about this render's route leaks into the next one.
        #
        # `is not None` rather than truthiness, because the caller selecting the DEFAULT route
        # is a selection and has to be honoured like any other. It arrives as `direct://` --
        # Chromium's own no-proxy URI -- so it takes this branch and gets a context whose proxy
        # setting overrides the container-wide `--proxy-server` applied at launch. Falling
        # through to the shared browser instead would route an explicitly-direct request out
        # through the container's proxy while still reporting `direct` back to the caller.
        tab, render_context_id = await _create_isolated_tab(_browser, "about:blank", proxy_server=egress_proxy)
        if session_state:
            await hitl._apply_context_state(_browser, render_context_id, session_state)
    elif session_state:
        # about:blank first so the cookies are in place before the real navigation -- a cookie
        # set after the page loads arrives too late to have been sent with the request that
        # was going to be challenged. Storage is applied after that navigation instead, since
        # localStorage is origin-scoped and about:blank is not the origin.
        tab, render_context_id = await _create_isolated_tab(_browser, "about:blank")
        await hitl._apply_context_state(_browser, render_context_id, session_state)
    else:
        tab = await _browser.get("about:blank", new_tab=True)
    main_frame_id = str(tab.target.target_id)
    last_response: dict[str, Any] = {}
    # Network-capture bookkeeping (only populated when capture_network=True):
    # request_id -> {url, method} from RequestWillBeSent, request_id ->
    # {status, content_type} from ResponseReceived, and the ordered list of
    # request_ids LoadingFinished fired for (bodies are only fetchable once
    # loading has finished -- fetching earlier races the browser and 404s).
    pending_requests: dict[Any, dict[str, Any]] = {}
    pending_responses: dict[Any, dict[str, Any]] = {}
    finished_request_ids: list[Any] = []
    eval_results: list[Any] = []

    def _capture_response(event: uc.cdp.network.ResponseReceived) -> None:
        # Overwrites on every matching event rather than keeping only the
        # first, belt-and-suspenders against any DOCUMENT responseReceived
        # firing more than once for this frame (e.g. a client-side navigation
        # during the settle wait) -- the LAST one observed is what's actually
        # rendered by the time get_content() runs.
        if _is_main_frame_document(event, main_frame_id):
            last_response["status"] = event.response.status
            last_response["url"] = event.response.url
        if capture_network and event.type_ in _API_RESOURCE_TYPES:
            pending_responses[event.request_id] = {
                "status": event.response.status,
                "content_type": event.response.mime_type,
            }

    def _capture_request(event: uc.cdp.network.RequestWillBeSent) -> None:
        if capture_network and event.type_ in _API_RESOURCE_TYPES:
            pending_requests[event.request_id] = {"url": event.request.url, "method": event.request.method}

    def _capture_loading_finished(event: uc.cdp.network.LoadingFinished) -> None:
        if capture_network and event.request_id in pending_requests:
            finished_request_ids.append(event.request_id)

    await tab.send(uc.cdp.network.enable())
    tab.add_handler(uc.cdp.network.ResponseReceived, _capture_response)
    if capture_network:
        tab.add_handler(uc.cdp.network.RequestWillBeSent, _capture_request)
        tab.add_handler(uc.cdp.network.LoadingFinished, _capture_loading_finished)
    try:
        await tab.send(uc.cdp.page.navigate(url))
        if session_state and render_context_id is not None:
            # Storage lands after the navigation and the page is then reloaded, because
            # localStorage is origin-scoped: it can only be written while a page from that
            # origin is loaded, which about:blank was not. The cookies were already in place
            # for the navigation above, which is the part that carries a cleared challenge.
            await hitl._apply_origin_storage(tab, session_state)
            await tab.send(uc.cdp.page.navigate(url))
        if nav_steps:
            # A settle wait before interacting, not just before the final content
            # capture -- live-reproduced against Maine's real WARN search form:
            # calling tab.select() immediately after navigate() (no settle) finds
            # the submit button, but by the time el.click() actually runs, the
            # node has gone stale ("Could not find node with given id [code:
            # -32000]") -- the still-loading page is still mutating/re-rendering
            # the DOM out from under the resolved backend_node_id. The same class
            # of race wait_for's own settle wait already exists to close, just
            # earlier in the sequence (before ANY interaction, not only before
            # get_content()).
            await tab.sleep(1.0)
            await _execute_nav_steps(tab, nav_steps, timeout, eval_results)
        if wait_for:
            # _select_with_retry (not a bare tab.select()): the same stale-CDP-
            # node race nav_steps hit live also reproduced here, against this
            # exact call, with no nav_steps involved at all -- see that
            # function's docstring. Retries internally until the selector
            # appears (or times out via the caller's outer asyncio.wait_for).
            await _select_with_retry(tab, wait_for, timeout, "wait_for")
        else:
            # nodriver has no load-event-based wait in this version (Tab.wait()
            # is a plain sleep under the hood); cdp.page.navigate does not block
            # until the page finishes loading. Reproduced live: without this,
            # get_content() reliably raced navigation and returned the pre-load
            # empty shell ("<html><head></head><body></body></html>", 39 bytes)
            # instead of the real page.
            await tab.sleep(1.0)
        html = await tab.get_content()
        network_calls: list[dict[str, Any]] = []
        if capture_network:
            # Fetched here, in the caller's own awaited control flow, NOT inside
            # an async event handler -- nodriver dispatches async handlers via
            # `asyncio.create_task(...)` (fire-and-forget), so get_response_body
            # calls made from inside a handler would race this function's own
            # return with no way to await their completion first (reproduced
            # live: an early version lost captured calls intermittently this
            # way). Bounded by _MAX_NETWORK_CALLS regardless of how many fired.
            for request_id in finished_request_ids[:_MAX_NETWORK_CALLS]:
                req_meta = pending_requests.get(request_id)
                resp_meta = pending_responses.get(request_id)
                if req_meta is None or resp_meta is None:
                    continue
                try:
                    body, is_base64 = await tab.send(uc.cdp.network.get_response_body(request_id))
                except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- one failed body fetch (e.g. the response was evicted from the browser's cache before we asked) must not drop every other captured call
                    log.debug("network capture: get_response_body failed for %s: %s", req_meta.get("url"), exc)
                    continue
                if is_base64 or len(body) > _MAX_NETWORK_BODY_BYTES:
                    continue
                stripped = body.lstrip()
                for prefix in _JSON_HIJACK_PREFIXES:
                    if stripped.startswith(prefix):
                        stripped = stripped[len(prefix) :].lstrip()
                        break
                if not (stripped.startswith("{") or stripped.startswith("[")):
                    continue  # not JSON-shaped -- not a useful "backend API" signal
                # `body` (the original, un-stripped response) is what's stored below --
                # a caller parsing a known API's real response needs the real bytes it
                # would have received directly, prefix included, not this function's own
                # internal shape-detection view of it.
                network_calls.append(
                    {
                        "url": req_meta["url"],
                        "method": req_meta["method"],
                        "status": resp_meta["status"],
                        "content_type": resp_meta["content_type"],
                        "body": body,
                    }
                )
    finally:
        # tab.close() lives here, not after this block, so a NavStepError (or
        # any other exception raised mid-render) still closes the tab instead
        # of leaking it -- before nav_steps, nothing in this try body could
        # actually raise (tab.select()'s own timeout returns None rather than
        # raising), so this distinction was previously unreachable dead code,
        # not a live bug.
        tab.remove_handler(uc.cdp.network.ResponseReceived, _capture_response)
        if capture_network:
            tab.remove_handler(uc.cdp.network.RequestWillBeSent, _capture_request)
            tab.remove_handler(uc.cdp.network.LoadingFinished, _capture_loading_finished)
        await tab.close()
        if render_context_id is not None:
            try:
                await _browser.send(uc.cdp.target.dispose_browser_context(render_context_id))
            except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- the page has already been rendered and returned; a context that will not dispose is a bounded leak, where raising here would discard a good result. Logged with its traceback below
                log.exception("scrape sidecar: could not dispose the session-state render context")
    # Fails open to 200/the originally requested url rather than raising or
    # blocking on a request whose DOCUMENT response genuinely never fired
    # (e.g. a same-document navigation) -- a render that produced real content
    # is a success either way; both fields are best-effort, not a correctness
    # gate on the fetch itself. final_url sourced from the captured response
    # (not `tab.url`) -- live-verified 2026-07-14: `tab.url` (nodriver's
    # `Tab.__getattr__` forwarding to `self.target.url`) does not reliably
    # reflect the post-navigate URL when navigating via raw `cdp.page.navigate`
    # instead of the higher-level `browser.get()` wrapper this function used
    # before the status fix above -- reproduced live against a real running
    # container (empty string returned for both a 200 and a 404 real fetch).
    # The captured response's own URL is the actual URL that document came
    # from, redirects included, with no dependency on that internal tracking.
    status = last_response.get("status", 200)
    final_url = last_response.get("url") or url
    return _RenderResult(
        html=html, final_url=final_url, status=status, network_calls=network_calls, eval_results=eval_results
    )


class DownloadError(Exception):
    """Raised when a forced download never completes (mirrors NavStepError's role for /v1/render).

    Caught by the ``/v1/download`` endpoint and reported as ``download_timeout`` (504).
    """


class _DownloadResult(NamedTuple):
    status: int
    filename: str
    content_type: str
    data: bytes


async def _create_isolated_tab(
    browser: Any, url: str, *, proxy_server: str | None = None
) -> tuple[Any, uc.cdp.browser.BrowserContextID]:
    """Create a fresh, isolated browser context + one tab within it, navigated to *url*.

    Live-reproduced (2026-07-15): ``browser.create_context()``'s own internal
    target lookup (one 0.5s sleep, one ``self.targets`` check) is not
    reliably enough time for a freshly created target to appear --
    ``StopIteration`` inside nodriver's own implementation. Reimplemented
    here with the same bounded-retry-with-explicit-refresh shape
    ``_select_with_retry`` already established for a different CDP timing
    race in this file, rather than trusting the library's own single-shot
    lookup.

    :return: the new tab, and its isolated browser context's id (needed by
        the caller to scope ``Browser.setDownloadBehavior`` and to dispose
        the context afterward)
    :rtype: tuple[Any, uc.cdp.browser.BrowserContextID]
    :raises RuntimeError: the created target never appeared in ``browser.targets``
    """
    # PER-CONTEXT proxying, which is what makes an exit a per-target choice rather than a
    # per-container one. `--proxy-server` on the browser command line is process-wide and
    # cannot vary; `Target.createBrowserContext` takes its own `proxyServer`, so two targets
    # in one browser can leave by two different exits. Verified against the running image's
    # own CDP bindings rather than assumed -- the earlier claim that Chromium could only do
    # this process-wide was true of the flag and wrong about contexts.
    context_id = await browser.send(uc.cdp.target.create_browser_context(proxy_server=proxy_server))
    target_id = await browser.send(uc.cdp.target.create_target(url, browser_context_id=context_id, new_window=True))
    for attempt in range(_TAB_LOOKUP_ATTEMPTS):
        await browser.update_targets()
        tab = next((t for t in browser.targets if t.target.target_id == target_id), None)
        if tab is not None:
            return tab, context_id
        if attempt < _TAB_LOOKUP_ATTEMPTS - 1:
            await asyncio.sleep(_TAB_LOOKUP_DELAY_SECONDS)
    raise RuntimeError(f"tab for target {target_id} never appeared in browser.targets")


async def _download(url: str, *, timeout: float = 30.0) -> _DownloadResult:
    """Navigate to *url* in an isolated browser context with forced-download behavior,
    and return the downloaded file's own bytes.

    Live-verified (2026-07-15, real West Virginia Cloudflare-protected PDF): a
    genuine browser session passes a real Cloudflare managed challenge on its
    own (no active challenge-solving involved) -- the only reason this needs
    to exist at all is that Chrome's built-in PDF viewer intercepts the
    navigation before any bytes are otherwise reachable, which
    ``_CHROME_PREFERENCES``' ``always_open_pdf_externally`` setting plus
    ``Browser.setDownloadBehavior`` fixes.

    Isolated context per call (not the shared/default one): concurrent
    ``/v1/download`` requests must never race each other's download
    directories -- live-verified with two real concurrent downloads into two
    separate directories, zero cross-contamination.

    :param url: the document URL to download
    :ptype url: str
    :param timeout: seconds to wait for the download to complete
    :ptype timeout: float
    :raises DownloadError: no file appeared in the download directory within *timeout*
    """
    download_dir = tempfile.mkdtemp(prefix="nodriver-download-")
    tab, context_id = await _create_isolated_tab(_browser, "about:blank")
    try:
        await _browser.send(
            uc.cdp.browser.set_download_behavior(
                behavior="allow", browser_context_id=context_id, download_path=download_dir
            )
        )
        await tab.send(uc.cdp.page.navigate(url))
        deadline = time.monotonic() + timeout
        downloaded_path: str | None = None
        while time.monotonic() < deadline:
            # A ".crdownload" suffix means Chrome is still writing the file --
            # only a file WITHOUT it is complete (mirrors _render's own
            # settle-wait shape: poll, don't assume one wait is enough).
            complete = [f for f in os.listdir(download_dir) if not f.endswith(".crdownload")]
            if complete:
                downloaded_path = os.path.join(download_dir, complete[0])
                break
            await asyncio.sleep(_DOWNLOAD_POLL_INTERVAL_SECONDS)
        if downloaded_path is None:
            raise DownloadError(f"no download completed for {url} within {timeout}s")
        with open(downloaded_path, "rb") as f:
            data = f.read()
        filename = os.path.basename(downloaded_path)
        content_type = "application/pdf" if filename.lower().endswith(".pdf") else "application/octet-stream"
        return _DownloadResult(status=200, filename=filename, content_type=content_type, data=data)
    finally:
        await tab.close()
        try:
            await _browser.send(uc.cdp.target.dispose_browser_context(context_id))
        except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- context disposal is best-effort cleanup, must never mask the real download outcome above
            log.debug("download: browser context disposal failed: %s", exc)
        shutil.rmtree(download_dir, ignore_errors=True)


async def _warm_up() -> None:
    """Render one real page before declaring the sidecar ready.

    Mitigates nodriver's cold-start timing gap at the source, in the
    container, instead of every consumer needing retry-tolerance of their
    own (reproduced live in a real consumer, 2026-07-14): a freshly-started
    browser's very first real render can race navigation and return the
    pre-load empty shell even with :func:`_render`'s own explicit settle
    wait -- a first-request-only phenomenon once the browser has completed
    one real render cycle. Retries a bounded number of times, then fails
    open (marks ready anyway, logged loudly) rather than blocking container
    startup forever on a single flaky attempt -- a still-cold browser on the
    first *real* request is the same failure mode this was already tolerant
    of before this mitigation existed, not a new risk.
    """
    global _ready
    for attempt in range(1, _WARMUP_ATTEMPTS + 1):
        try:
            await asyncio.wait_for(_render(_WARMUP_URL, None), timeout=_WARMUP_TIMEOUT_SECONDS)
            log.info("warm-up render succeeded (attempt %d/%d)", attempt, _WARMUP_ATTEMPTS)
            break
        except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- warm-up must degrade (fail open), never block startup forever
            log.warning("warm-up render failed (attempt %d/%d): %s", attempt, _WARMUP_ATTEMPTS, exc)
            if attempt < _WARMUP_ATTEMPTS:
                await asyncio.sleep(_WARMUP_RETRY_DELAY_SECONDS)
    else:
        log.error(
            "warm-up render never succeeded after %d attempts -- marking ready anyway (fail open)", _WARMUP_ATTEMPTS
        )
    _ready = True


#: How much larger to draw everything a human looks at, like the display-scaling setting on a
#: desktop OS. ``1.0`` is unscaled; ``1.5`` makes text half again as large.
#:
#: A requirement raised by an operator during verification, not a fix: the display is a fixed
#: 1920x1080 scaled down to whatever browser window is looking at it, so on a laptop the text
#: arrives small and there was no way to ask for it bigger. Scaling the DESKTOP would undo the
#: fit; scaling the CONTENT leaves the fit alone, which is what was asked for.
#:
#: Deliberately affects only what a person reads. Rendering for extraction never goes through
#: this path -- a scraped page's own layout must not depend on an operator's comfort setting, or
#: two deployments with different values would extract differently from the same site.
UI_SCALE = os.environ.get("UI_SCALE", "1.0")


def _browser_args() -> list[str]:
    """Chromium's launch arguments, including the egress exit when one is configured.

    A function rather than a literal inside ``_lifespan`` so a test can assert on the arguments
    PRODUCTION builds. The first version of that test rebuilt the list itself and asserted on
    its own copy, which stayed green with the production line deleted -- a test of the test.
    """
    args = ["--disable-dev-shm-usage", "--disable-gpu"]
    if UI_SCALE not in ("", "1.0", "1"):
        # Chromium's own display-scaling knob, the same one a desktop OS drives. Applied to the
        # browser rather than the X server so the desktop geometry -- and therefore the fit the
        # operator already has -- is untouched.
        args.append(f"--force-device-scale-factor={UI_SCALE}")
    if EGRESS_PROXY:
        # Process-wide by nature: Chromium applies --proxy-server to the whole browser, so this
        # is the DEFAULT exit rather than the only one. A render that wants a different exit
        # gets its own browser context, which takes its own proxyServer.
        args.append(f"--proxy-server={EGRESS_PROXY}")
    return args


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _browser
    # Pinned user_data_dir (not nodriver's own auto-generated temp one) so this
    # Preferences file is guaranteed to be the ONE the persistent browser instance
    # below actually reads -- see _CHROME_PREFERENCES' own comment for why this is
    # needed at all (forced-download capability, scrape-task-04).
    profile_default_dir = os.path.join(_USER_DATA_DIR, "Default")
    os.makedirs(profile_default_dir, exist_ok=True)
    with open(os.path.join(profile_default_dir, "Preferences"), "w") as f:
        json.dump(_CHROME_PREFERENCES, f)

    _browser = await uc.start(
        headless=False,
        browser_executable_path=CHROMIUM_PATH,
        user_data_dir=_USER_DATA_DIR,
        # sandbox=False is nodriver's own recognized kwarg for "running as
        # root" (the container has no non-root USER); passing --no-sandbox
        # only via browser_args is not sufficient -- nodriver's own
        # connect-back check still refuses to start without this.
        sandbox=False,
        browser_args=_browser_args(),
    )
    await _warm_up()
    yield
    # Before the browser, because the VNC processes are children of this one and a
    # container stopping should not leave an x11vnc holding the RFB port for whatever
    # restarts into the same namespace.
    await _sessions.shutdown()
    if _browser is not None:
        _browser.stop()


app = FastAPI(lifespan=_lifespan)


@app.post("/v1/render", response_model=RenderResponse)
async def render(req: RenderRequest) -> RenderResponse | JSONResponse:
    """Render *req.url* through nodriver and return the page."""
    if _browser is None:
        return JSONResponse(status_code=503, content={"error": {"code": "not_ready", "message": "browser not started"}})

    start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            _render(
                req.url,
                req.wait_for,
                capture_network=req.capture_network,
                nav_steps=req.nav_steps,
                timeout=req.timeout,
                session_state=req.session_state,
                egress_proxy=req.egress_proxy,
            ),
            timeout=req.timeout,
        )
    except TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"error": {"code": "navigation_timeout", "message": f"render timed out after {req.timeout}s"}},
        )
    except NavStepError as exc:
        return JSONResponse(status_code=422, content={"error": {"code": "nav_step_failed", "message": str(exc)}})
    except Exception as exc:  # prawduct:allow prawduct/broad-except -- driver crash surface must not take the sidecar process down; reported to the caller, never swallowed
        return JSONResponse(status_code=502, content={"error": {"code": "driver_crash", "message": str(exc)}})

    timing_ms = (time.monotonic() - start) * 1000
    return RenderResponse(
        html=result.html,
        status=result.status,
        final_url=result.final_url,
        timing_ms=timing_ms,
        network_calls=[NetworkCall(**call) for call in result.network_calls],
        eval_results=result.eval_results,
        # The exit this render was CONFIGURED to use -- the request's own name when it selected
        # one, otherwise the container's. Deliberately not the exit OBSERVED: this value is
        # derived from the request, so a per-context proxy that Chromium accepted and ignored
        # would still be reported as `tor` here, and that string is written through to
        # `ScrapeTargetHealth.last_egress`.
        #
        # Confirming that traffic genuinely leaves by this exit needs an outside observer --
        # something reading the address this container presents to a third party. Nothing in
        # this process can tell the difference.
        #
        # `is not None` matches the routing branch above -- a request that selected the default
        # route sends `direct://` and must report `direct`, not the container's name.
        egress=(req.egress_name or "unnamed") if req.egress_proxy is not None else EGRESS_NAME,
    )


@app.post("/v1/download", response_model=DownloadResponse)
async def download(req: DownloadRequest) -> DownloadResponse | JSONResponse:
    """Download *req.url*'s real file bytes through a real browser session with forced-download
    behavior -- for a document a plain HTTP client can't reach (see :class:`DownloadRequest`).

    Design and live verification: docs/scrape-task-04-multi-document-driver.md's
    "New sidecar capability: browser-forced download" section."""
    if _browser is None:
        return JSONResponse(status_code=503, content={"error": {"code": "not_ready", "message": "browser not started"}})

    start = time.monotonic()
    try:
        result = await asyncio.wait_for(_download(req.url, timeout=req.timeout), timeout=req.timeout)
    except (TimeoutError, DownloadError) as exc:
        return JSONResponse(
            status_code=504,
            content={
                "error": {"code": "download_timeout", "message": str(exc) or f"download timed out after {req.timeout}s"}
            },
        )
    except Exception as exc:  # prawduct:allow prawduct/broad-except -- driver crash surface must not take the sidecar process down; reported to the caller, never swallowed
        return JSONResponse(status_code=502, content={"error": {"code": "driver_crash", "message": str(exc)}})

    timing_ms = (time.monotonic() - start) * 1000
    return DownloadResponse(
        status=result.status,
        filename=result.filename,
        content_type=result.content_type,
        content_base64=base64.b64encode(result.data).decode("ascii"),
        timing_ms=timing_ms,
    )


@app.get("/healthz")
async def healthz() -> dict[str, str | None]:
    """Liveness/readiness probe for docker-compose healthcheck.

    Reports the configured egress so a caller can tell WHICH exit a result came from. A
    deployment running one container per exit otherwise has no way to confirm, from outside,
    that the container it is talking to is the one it thinks it is.

    ``null`` when the deployment configured no exit -- the same "nobody said" this value carries
    everywhere else, rather than a claim that the default route was chosen.
    """
    return {"status": "ok" if _ready else "starting", "egress": EGRESS_NAME}


# --------------------------------------------------------------------------
# HITL: the bare VNC path.
#
# These predate the session API below and remain for the case it does not cover:
# bringing the display up on its own, to look at what the unattended browser is
# doing, with no session and no target. That is a real diagnostic need and it is
# why they were not deleted.
#
# They no longer act while a session is open, which is the part that matters.
# The session owns the display for its whole life, so a bare POST bringing it up
# outside a TTL, or a bare DELETE killing it under a live session that then goes
# on reporting itself open, is two owners of one resource and a state divergence
# waiting to happen. Both refuse with 409 and name the session API instead.
#
# Unauthenticated, like every other endpoint here: the sidecar holds no identity
# and authenticates nobody. It is reachable only from inside the deployment, and
# the token check that fronts it belongs on the MIT side, which is the only side
# that can evaluate a policy.
# --------------------------------------------------------------------------

_sessions = hitl.SessionManager(browser_provider=lambda: _browser)
_vnc = _sessions.vnc


class HitlSessionRequest(BaseModel):
    """Nothing to supply: one display means the session's shape is not negotiable."""


class HitlTabRequest(BaseModel):
    """Pull one target into the session.

    Carries `url` + `nav_steps` rather than any handle to an earlier fetch, because nothing is
    held while a target waits for a human. It is reported and forgotten, and re-driven from
    these two fields when an operator actually arrives -- so waiting costs no container
    resources, and the replay is deterministic because nav-step replay is already how this
    package reaches gated pages.
    """

    target_id: str
    url: str
    nav_steps: list[NavStepModel] | None = None
    #: A previously exported state, applied to the isolated context BEFORE navigating. Raw,
    #: not sealed: this container holds no key. Whoever calls this has already opened it.
    session_state: dict[str, Any] | None = None


# response_model=None: the union of a dict and a JSONResponse is not a Pydantic
# field type, and FastAPI infers a response model from the annotation unless told not
# to -- the same reason the render/download endpoints name theirs explicitly.
@app.post("/v1/hitl/vnc", response_model=None)
async def hitl_vnc_start() -> dict[str, Any] | JSONResponse:
    """Start the VNC path, or return the one already running.

    Idempotent: a caller that retries gets the running session rather than a second
    ``x11vnc`` losing a race for the RFB port.
    """
    if _sessions.owns_display():
        return JSONResponse(
            status_code=409,
            content={
                "error": (
                    "a HITL session owns the display; use POST /v1/hitl/session to open one, "
                    "or DELETE it before driving the display directly"
                )
            },
        )
    try:
        session = await _vnc.start()
    except hitl.VncUnavailable as exc:
        log.warning("hitl: could not start the vnc path: %s", exc)
        return JSONResponse(status_code=503, content={"error": str(exc)})
    return {"display": session.display}


@app.get("/v1/hitl/vnc")
async def hitl_vnc_status() -> dict[str, Any]:
    """Whether the display is being served right now.

    Asks about the process that actually serves it: a readiness signal reporting healthy over a
    blank display is worse than none, because it is believed.
    """
    return {"running": _vnc.health(), "display": _vnc.display}


@app.delete("/v1/hitl/vnc", response_model=None)
async def hitl_vnc_stop() -> dict[str, Any] | JSONResponse:
    """Stop both processes, leaving nothing listening.

    Refuses while a session owns the display: stopping it underneath one would leave a session
    reporting itself open with nothing for its operator to look at.
    """
    # `owns_display`, not `current`: an EXPIRED session is refused by `authorize`, so it cannot
    # be torn down through the session API -- and if it also blocked here, nothing but the
    # reaper could ever release the display. This is the escape hatch for exactly that, so it
    # closes the session too rather than stopping the display out from under a tracked one.
    if _sessions.owns_display():
        return JSONResponse(
            status_code=409,
            content={"error": "a HITL session owns the display; DELETE /v1/hitl/session/{id} instead"},
        )
    if _sessions.current() is not None:
        log.info("hitl: releasing the display held by an expired session")
        await _sessions.close()
        return {"running": _vnc.health(), "released_expired_session": True}
    await _vnc.stop()
    return {"running": _vnc.health()}


# --------------------------------------------------------------------------
# HITL sessions.
#
# The token is returned once, by the create call, and every later call carries
# it. It is checked with a constant-time compare against a session this process
# minted -- which is a weaker claim than authentication and is deliberately all
# the sidecar makes: it holds no identity and cannot evaluate a policy. Who was
# entitled to be handed the token is the MIT side's question.
# --------------------------------------------------------------------------


def _token_from(authorization: str | None, x_hitl_token: str | None) -> str:
    """Read the session token from either header, preferring the standard one."""
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return (x_hitl_token or "").strip()


@app.post("/v1/hitl/session", response_model=None)
async def hitl_session_open(req: HitlSessionRequest | None = None) -> dict[str, Any] | JSONResponse:
    """Open the session and bring up the display.

    409 rather than a queue when one is already open. One display means one operator, and
    queueing would hold an HTTP request open for however long the first operator takes --
    minutes to hours, which is not a thing to do to a caller.
    """
    del req
    try:
        session = await _sessions.open()
    except hitl.SessionUnavailable as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    except hitl.VncUnavailable as exc:
        log.warning("hitl: session refused, no display: %s", exc)
        return JSONResponse(status_code=503, content={"error": str(exc)})
    return {
        "session_id": session.session_id,
        "token": session.token,
        "expires_at": session.expires_at,
        "max_slots": session.max_slots,
    }


@app.get("/v1/hitl/session/{session_id}", response_model=None)
async def hitl_session_get(
    session_id: str,
    authorization: str | None = Header(default=None),
    x_hitl_token: str | None = Header(default=None),
) -> dict[str, Any] | JSONResponse:
    """Session state and the tabs currently open in it."""
    try:
        session = _sessions.authorize(session_id, _token_from(authorization, x_hitl_token))
    except hitl.SessionNotFound as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    return {
        "session_id": session.session_id,
        "expires_at": session.expires_at,
        "max_slots": session.max_slots,
        "free_slots": session.free_slots(),
        "tabs": [
            {"tab_id": t.tab_id, "target_id": t.target_id, "url": t.url, "opened_at": t.opened_at}
            for t in session.tabs.values()
        ],
    }


@app.post("/v1/hitl/session/{session_id}/tab", response_model=None)
async def hitl_tab_open(
    session_id: str,
    req: HitlTabRequest,
    authorization: str | None = Header(default=None),
    x_hitl_token: str | None = Header(default=None),
) -> dict[str, Any] | JSONResponse:
    """Bring one target into the session, in its own isolated context."""
    try:
        session = _sessions.authorize(session_id, _token_from(authorization, x_hitl_token))
    except hitl.SessionNotFound as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    try:
        tab = await _sessions.open_tab(
            session,
            target_id=req.target_id,
            url=req.url,
            nav_steps=req.nav_steps,
            session_state=req.session_state,
        )
    except hitl.SessionUnavailable as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    except hitl.SessionNotFound as exc:
        # The session was closed or reaped while this navigation was in flight. An ordinary
        # race, not a fault: INFO and 409, where the broad handler below would call it a bad
        # gateway and log a traceback for something nobody needs to investigate.
        log.info("hitl: session closed while opening a tab for target %s", req.target_id)
        return JSONResponse(status_code=409, content={"error": str(exc)})
    except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a nav-step replay or a CDP timing failure is this target's problem, not the session's; surfacing it as a ToolResult-shaped error keeps the operator's other tabs alive. Logged with its traceback below
        log.exception("hitl: could not open a tab for target %s", req.target_id)
        return JSONResponse(status_code=502, content={"error": f"could not open the target: {exc}"})
    return {"tab_id": tab.tab_id, "target_id": tab.target_id, "url": tab.url, "free_slots": session.free_slots()}


@app.post("/v1/hitl/session/{session_id}/tab/{tab_id}/complete", response_model=None)
async def hitl_tab_complete(
    session_id: str,
    tab_id: str,
    authorization: str | None = Header(default=None),
    x_hitl_token: str | None = Header(default=None),
) -> dict[str, Any] | JSONResponse:
    """The human says this one is cleared: close the tab and free its slot.

    Exporting the context's cookies happens here because this is the last moment the context
    exists -- once it is disposed the human's work is gone.
    """
    try:
        session = _sessions.authorize(session_id, _token_from(authorization, x_hitl_token))
    except hitl.SessionNotFound as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    try:
        tab = await _sessions.complete_tab(session, tab_id)
    except hitl.SessionNotFound as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    # `session_state` is the human's work, raw and unsealed. It is returned exactly once, to
    # the caller that completed the tab, and never logged: the MIT side seals it before it
    # touches a database. A cookie jar for a cleared challenge is a credential.
    return {
        "tab_id": tab.tab_id,
        "target_id": tab.target_id,
        "free_slots": session.free_slots(),
        "session_state": tab.exported_state,
    }


@app.delete("/v1/hitl/session/{session_id}", response_model=None)
async def hitl_session_close(
    session_id: str,
    authorization: str | None = Header(default=None),
    x_hitl_token: str | None = Header(default=None),
) -> dict[str, Any] | JSONResponse:
    """Tear the session down: drop every context, stop the display."""
    try:
        session = _sessions.authorize(session_id, _token_from(authorization, x_hitl_token))
    except hitl.SessionNotFound as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    await _sessions.close(session)
    return {"closed": True, "session_id": session_id}
