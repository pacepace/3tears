"""nodriver sidecar -- thin HTTP wrapper around nodriver + Xvfb.

Runs inside its own AGPL-3.0-licensed container (see LICENSE). Never
imported as a Python library from 3tears-scrape's (MIT) tree; consumers
only ever talk to this process over HTTP, per scrape-api-contract.md's
``POST /v1/render`` contract. ``entrypoint.sh`` starts Xvfb and points
``DISPLAY`` at it before this process starts -- nodriver launches Chromium
with ``headless=False`` against that virtual display, per nodriver's own
documented guidance for headless-machine deployments (real headed Chromium
under Xvfb has better real-world site compatibility than ``headless=True``,
matching the product brief's rationale for choosing nodriver first).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, NamedTuple

import nodriver as uc
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from nodriver.core.connection import ProtocolException
from pydantic import BaseModel

log = logging.getLogger("nodriver_sidecar")
logging.basicConfig(level=logging.INFO)

CHROMIUM_PATH = os.environ.get("CHROMIUM_PATH", "/usr/bin/chromium")

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
    docstring for the four supported actions and why they're driven
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


async def _select_with_retry(tab: Any, selector: str, timeout: float, action: str, value: str | None = None) -> Any:
    """Find *selector*, then perform *action* (``"click"``/``"fill"``/
    ``"wait_for"``) on the result -- retrying the whole find-then-act
    sequence from scratch (a fresh ``tab.select()`` re-queries the live DOM)
    when a stale CDP node id is hit mid-sequence. See the module-level
    comment above for the live reproduction this is built from.

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


async def _execute_nav_steps(tab: Any, nav_steps: list[NavStepModel], timeout: float) -> None:
    """Drive *tab* through *nav_steps* in order, before the caller's own settle-wait.

    Each step gets the full outer *timeout* to find its selector, matching
    ``wait_for``'s own per-render (not per-step) timeout budget -- simpler
    than apportioning a shared budget across steps, and a nav step search is
    the same class of "wait for a real page to respond" operation ``wait_for``
    already gets the full timeout for.
    """
    for i, step in enumerate(nav_steps):
        if step.action == "wait_ms":
            await tab.sleep((step.ms or 0) / 1000)
            continue
        if step.action not in ("click", "fill", "wait_for"):
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


async def _render(
    url: str,
    wait_for: str | None,
    *,
    capture_network: bool = False,
    nav_steps: list[NavStepModel] | None = None,
    timeout: float = 30.0,
) -> _RenderResult:
    """Navigate to *url*, optionally drive it through *nav_steps*, wait for a
    selector, and return the rendered page.

    ``status`` -- SCR-7L4M fix (2026-07-14): a plain ``browser.get(url,
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
            await _execute_nav_steps(tab, nav_steps, timeout)
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
                if not (stripped.startswith("{") or stripped.startswith("[")):
                    continue  # not JSON-shaped -- not a useful "backend API" signal
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
    # Fails open to 200/the originally requested url rather than raising or
    # blocking on a request whose DOCUMENT response genuinely never fired
    # (e.g. a same-document navigation) -- a render that produced real content
    # is a success either way; both fields are best-effort, not a correctness
    # gate on the fetch itself. final_url sourced from the captured response
    # (not `tab.url`) -- live-verified 2026-07-14: `tab.url` (nodriver's
    # `Tab.__getattr__` forwarding to `self.target.url`) does not reliably
    # reflect the post-navigate URL when navigating via raw `cdp.page.navigate`
    # instead of the higher-level `browser.get()` wrapper this function used
    # before the SCR-7L4M status fix -- reproduced live against a real running
    # container (empty string returned for both a 200 and a 404 real fetch).
    # The captured response's own URL is the actual URL that document came
    # from, redirects included, with no dependency on that internal tracking.
    status = last_response.get("status", 200)
    final_url = last_response.get("url") or url
    return _RenderResult(html=html, final_url=final_url, status=status, network_calls=network_calls)


async def _warm_up() -> None:
    """Render one real page before declaring the sidecar ready.

    Mitigates nodriver's cold-start timing gap at the source, in the
    container, instead of every consumer needing retry-tolerance of their
    own (reproduced live, faidh-side, 2026-07-14): a freshly-started
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


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _browser
    _browser = await uc.start(
        headless=False,
        browser_executable_path=CHROMIUM_PATH,
        # sandbox=False is nodriver's own recognized kwarg for "running as
        # root" (the container has no non-root USER); passing --no-sandbox
        # only via browser_args is not sufficient -- nodriver's own
        # connect-back check still refuses to start without this.
        sandbox=False,
        browser_args=["--disable-dev-shm-usage", "--disable-gpu"],
    )
    await _warm_up()
    yield
    if _browser is not None:
        _browser.stop()


app = FastAPI(lifespan=_lifespan)


@app.post("/v1/render", response_model=RenderResponse)
async def render(req: RenderRequest) -> RenderResponse | JSONResponse:
    """Render *req.url* through nodriver and return the page, per scrape-api-contract.md."""
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
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness/readiness probe for docker-compose healthcheck."""
    return {"status": "ok" if _ready else "starting"}
