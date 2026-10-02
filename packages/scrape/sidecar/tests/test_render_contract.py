"""Unit tests for the sidecar's POST /v1/render contract.

No real Chromium/Xvfb involved -- the container is booted through its own
lifespan with ``uc.start`` handing back a fake shaped like nodriver's
``Browser``/``Tab`` (see the ``boot`` fixture), so these tests stay hermetic
while startup, warm-up and shutdown all run for real. Proving a genuine
nodriver-driven Chromium render works end to end requires this container
actually running, so it is a consuming application's job, exercised against
the image built from this directory via docker compose.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import os
from contextlib import AsyncExitStack
from types import SimpleNamespace

import httpx
import main
import nodriver as uc
import pytest
from nodriver.core.connection import ProtocolException

from tests.conftest import X11vncStub, lifecycle_on_test_port


def _use_timings(monkeypatch: pytest.MonkeyPatch, **overrides: float) -> None:
    """Run the container with some of its wait and retry budgets shortened for this test.

    Replaces the app's own ``SidecarTimings`` with a copy carrying *overrides*, so every other
    budget stays at its production value and the paths read them exactly where production does.

    :param monkeypatch: the test's monkeypatch, which restores the production budgets afterwards
    :ptype monkeypatch: pytest.MonkeyPatch
    :param overrides: ``SidecarTimings`` fields to change
    :ptype overrides: float
    """
    monkeypatch.setattr(main.app.state, "timings", dataclasses.replace(main.app.state.timings, **overrides))


# parity-exempt: hand-rolled subset stub of nodriver's third-party Element (only click/clear_input/send_keys/scroll_into_view, the only surface nav-steps drive); nodriver is AGPL-isolated to this sidecar and never installed in the workspace venv, so a parity-with marker cannot resolve there
class _FakeElement:
    """Nav-steps (2026-07-14): fakes nodriver's ``Element``
    (click/clear_input/send_keys/scroll_into_view)."""

    def __init__(self, selector: str) -> None:
        self.selector = selector
        self.clicked = False
        self.cleared = False
        self.sent_keys: str | None = None
        self.scrolled_into_view = False

    async def click(self) -> None:
        self.clicked = True

    async def clear_input(self) -> None:
        self.cleared = True

    async def send_keys(self, text: str) -> None:
        self.sent_keys = text

    async def scroll_into_view(self) -> None:
        self.scrolled_into_view = True


# parity-exempt: hand-rolled subset stub of nodriver's third-party Tab (only the CDP surface _render uses: .target.target_id/send()/add_handler()/remove_handler()); nodriver is AGPL-isolated to this sidecar and never installed in the workspace venv, so a parity-with marker cannot resolve there
class _FakeTab:
    """Also fakes the CDP surface ``_render`` uses to capture the
    real HTTP status -- ``.target.target_id``, ``.send()``, ``.add_handler()``/
    ``.remove_handler()``. ``fire_response`` lets a test simulate the browser
    emitting ``Network.responseReceived`` for the registered handler, the way
    ``send(cdp.page.navigate(...))`` would trigger it for real.
    """

    def __init__(
        self,
        html: str,
        url: str,
        target_id: str = "main-frame-id",
        response_status: int | None = None,
        network_calls_to_fire: list[dict] | None = None,
        findable_selectors: set[str] | None = None,
        select_protocol_exceptions: int = 0,
        evaluate_returns: list | None = None,
        evaluate_exc: Exception | None = None,
    ) -> None:
        self._html = html
        self.url = url
        self.target = SimpleNamespace(target_id=target_id)
        self.selected_for: str | None = None
        self.slept: float | None = None
        self.closed = False
        self.sent: list[object] = []
        self._handlers: dict[type, list] = {}
        # When set, auto-fires a matching Network.responseReceived the moment
        # `send(cdp.page.navigate(...))` is called -- models the ordinary case
        # (response arrives after navigation starts); tests needing finer
        # control (redirect chains, wrong frame/type) call fire_response directly.
        self._response_status = response_status
        # request_id (str) -> (body, is_base64), configured by fire_network_call --
        # what get_response_body() returns when _render asks for that request's body.
        self._network_bodies: dict[str, tuple[str, bool]] = {}
        # Same auto-fire-on-navigate convenience as response_status, for the
        # common "one or more full XHR/fetch cycles happen during this render"
        # case -- each dict is fire_network_call()'s own kwargs.
        self._network_calls_to_fire = network_calls_to_fire or []
        # Nav-steps (2026-07-14): which selectors select() resolves to a real
        # _FakeElement -- None means "every selector is found" (every
        # pre-nav-steps test's implicit assumption, preserved as the default).
        self._findable_selectors = findable_selectors
        self.select_calls: list[str] = []
        self.sleep_calls: list[float] = []
        self.scroll_down_calls: list[int] = []
        # _select_with_retry (2026-07-14, live-reproduced stale-CDP-node race):
        # how many of the next select() calls raise ProtocolException before
        # succeeding -- simulates a resolved element/document going stale
        # mid-sequence, decremented on every call regardless of selector.
        self._select_protocol_exceptions_remaining = select_protocol_exceptions
        # evaluate() capability: a queue of return values, one per real call
        # (None returned once exhausted); evaluate_exc, if set, is raised on
        # every call instead.
        self._evaluate_returns = list(evaluate_returns) if evaluate_returns is not None else []
        self._evaluate_exc = evaluate_exc
        self.evaluate_calls: list[str] = []

    async def send(self, cmd: object):
        self.sent.append(cmd)
        co_name = getattr(getattr(cmd, "gi_code", None), "co_name", None)
        if co_name == "navigate":
            if self._response_status is not None:
                self.fire_response(self._response_status)
            for call_kwargs in self._network_calls_to_fire:
                self.fire_network_call(**call_kwargs)
        if co_name == "get_response_body":
            request_id = str(cmd.gi_frame.f_locals["request_id"])
            return self._network_bodies.get(request_id, ("", False))
        return None

    def add_handler(self, event_type: type, callback) -> None:
        self._handlers.setdefault(event_type, []).append(callback)

    def remove_handler(self, event_type: type, callback) -> bool:
        callbacks = self._handlers.get(event_type)
        if not callbacks or callback not in callbacks:
            return False
        callbacks.remove(callback)
        return True

    def fire_response(
        self, status: int, *, frame_id: str | None = None, resource_type=None, response_url: str | None = None
    ) -> None:
        """Simulate a ``Network.responseReceived`` event for every registered handler."""
        event = uc.cdp.network.ResponseReceived(
            request_id=uc.cdp.network.RequestId("req-1"),
            loader_id=uc.cdp.network.LoaderId("loader-1"),
            timestamp=uc.cdp.network.MonotonicTime(0.0),
            type_=resource_type or uc.cdp.network.ResourceType.DOCUMENT,
            response=uc.cdp.network.Response(
                url=response_url if response_url is not None else self.url,
                status=status,
                status_text="",
                headers=uc.cdp.network.Headers({}),
                mime_type="text/html",
                charset="utf-8",
                connection_reused=False,
                connection_id=0.0,
                encoded_data_length=0.0,
                security_state=uc.cdp.security.SecurityState.NEUTRAL,
            ),
            has_extra_info=False,
            frame_id=frame_id if frame_id is not None else self.target.target_id,
        )
        for callback in self._handlers.get(uc.cdp.network.ResponseReceived, []):
            callback(event)

    def fire_network_call(
        self,
        request_id: str,
        url: str,
        *,
        method: str = "GET",
        status: int = 200,
        resource_type=None,
        content_type: str = "application/json",
        body: str = "{}",
        is_base64: bool = False,
        frame_id: str | None = None,
        post_data: str | None = None,
    ) -> None:
        """Simulate a full XHR/fetch request/response/loading-finished cycle
        -- RequestWillBeSent -> ResponseReceived -> LoadingFinished, then
        configures what get_response_body(request_id) returns, matching
        exactly the sequence real CDP fires for one network call."""
        resource_type = resource_type or uc.cdp.network.ResourceType.XHR
        resolved_frame_id = frame_id if frame_id is not None else self.target.target_id
        rid = uc.cdp.network.RequestId(request_id)
        request = uc.cdp.network.Request(
            url=url,
            method=method,
            headers=uc.cdp.network.Headers({}),
            initial_priority=uc.cdp.network.ResourcePriority.LOW,
            referrer_policy="strict-origin-when-cross-origin",
            post_data=post_data,
        )
        req_event = uc.cdp.network.RequestWillBeSent(
            request_id=rid,
            loader_id=uc.cdp.network.LoaderId("loader-1"),
            document_url=self.url,
            request=request,
            timestamp=uc.cdp.network.MonotonicTime(0.0),
            wall_time=uc.cdp.network.TimeSinceEpoch(0.0),
            initiator=uc.cdp.network.Initiator(type_="script"),
            redirect_has_extra_info=False,
            type_=resource_type,
            frame_id=resolved_frame_id,
            redirect_response=None,
            has_user_gesture=None,
            render_blocking_behavior=None,
        )
        for callback in self._handlers.get(uc.cdp.network.RequestWillBeSent, []):
            callback(req_event)

        resp_event = uc.cdp.network.ResponseReceived(
            request_id=rid,
            loader_id=uc.cdp.network.LoaderId("loader-1"),
            timestamp=uc.cdp.network.MonotonicTime(0.0),
            type_=resource_type,
            response=uc.cdp.network.Response(
                url=url,
                status=status,
                status_text="",
                headers=uc.cdp.network.Headers({}),
                mime_type=content_type,
                charset="utf-8",
                connection_reused=False,
                connection_id=0.0,
                encoded_data_length=0.0,
                security_state=uc.cdp.security.SecurityState.NEUTRAL,
            ),
            has_extra_info=False,
            frame_id=resolved_frame_id,
        )
        for callback in self._handlers.get(uc.cdp.network.ResponseReceived, []):
            callback(resp_event)

        self._network_bodies[request_id] = (body, is_base64)
        finished_event = uc.cdp.network.LoadingFinished(
            request_id=rid, timestamp=uc.cdp.network.MonotonicTime(0.0), encoded_data_length=float(len(body))
        )
        for callback in self._handlers.get(uc.cdp.network.LoadingFinished, []):
            callback(finished_event)

    async def select(self, selector: str, timeout: float = 10) -> _FakeElement | None:
        self.selected_for = selector
        self.select_calls.append(selector)
        if self._select_protocol_exceptions_remaining > 0:
            self._select_protocol_exceptions_remaining -= 1
            raise ProtocolException("Could not find node with given id [code: -32000]")
        if self._findable_selectors is not None and selector not in self._findable_selectors:
            return None
        return _FakeElement(selector)

    async def sleep(self, t: float) -> None:
        self.slept = t
        self.sleep_calls.append(t)

    async def scroll_down(self, amount: int = 25) -> None:
        self.scroll_down_calls.append(amount)

    async def evaluate(self, expression: str, return_by_value: bool = True):
        self.evaluate_calls.append(expression)
        if self._evaluate_exc is not None:
            raise self._evaluate_exc
        return self._evaluate_returns.pop(0) if self._evaluate_returns else None

    async def get_content(self) -> str:
        return self._html

    async def close(self) -> None:
        self.closed = True


# parity-exempt: hand-rolled subset stub of nodriver's third-party Browser (only get()/stop(), the only surface the render path calls); nodriver is AGPL-isolated to this sidecar and never installed in the workspace venv, so a parity-with marker cannot resolve there
class _FakeBrowser:
    """A browser whose renders a test configures.

    Until it is ``booted`` it answers the startup warm-up from tabs of its own, so the tab a test
    hands it -- and the failures it is told to produce -- belong to the test's requests alone.
    ``including_warm_up=True`` is for the tests that are ABOUT the warm-up: the configured tab
    and failures then apply from the first render startup makes.
    """

    def __init__(
        self,
        tab: _FakeTab | None = None,
        raise_exc: Exception | None = None,
        hang: bool = False,
        fail_times: int = 0,
        including_warm_up: bool = False,
    ) -> None:
        self._tab = tab
        self._raise_exc = raise_exc
        self._hang = hang
        self._fail_times = fail_times
        self.booted = including_warm_up
        self.get_calls = 0
        self.stopped = False
        self.cdp_calls: list[object] = []

    async def send(self, cmd: object) -> object:
        """Stand in for the CDP calls the isolated-context path makes.

        Needed for context DISPOSAL: a render that named its own exit disposes its context in
        `_render`'s finally, on the browser connection. The tab creation itself is
        monkeypatched at `create_isolated_tab`, so `update_targets`/`targets` are not needed
        and are deliberately absent -- unreachable fake surface reads as covered behaviour and
        is worse than none.
        """
        self.cdp_calls.append(cmd)
        return "ctx-fake"

    async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
        if not self.booted:
            return _warm_up_tab()
        self.get_calls += 1
        if self._hang:
            await asyncio.sleep(3600)
        if self.get_calls <= self._fail_times:
            raise RuntimeError(f"cold-start failure (call {self.get_calls})")
        if self._raise_exc is not None:
            raise self._raise_exc
        assert self._tab is not None
        return self._tab

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=main.app)
    return httpx.AsyncClient(transport=transport, base_url="http://sidecar.test")


@pytest.fixture
async def boot(monkeypatch: pytest.MonkeyPatch):
    """Start the container the way production does, against fakes for what ``uc.start`` launches.

    The front door to every browser-dependent route is the app's own lifespan: it is what
    launches Chromium, warms it up, hides its idle window, and tears all of it down again. So a
    test hands this its fake browser and the REAL startup runs against it -- warm-up render
    included, which is why each fake serves the warm-up from tabs of its own until it is
    ``booted`` and the tabs a test inspects are untouched by startup. Each later ``uc.start`` (a
    relaunch) takes the next browser in line.

    Teardown runs the real shutdown, which leaves the container not-started for the next test.

    :return: an async callable taking the browsers ``uc.start`` hands out, in order, and returning
        the keyword arguments every ``uc.start`` call received
    """
    stack = AsyncExitStack()

    async def _boot(*browsers: object) -> list[dict[str, object]]:
        queue = list(browsers)
        launches: list[dict[str, object]] = []

        async def _start(**kwargs: object) -> object:
            launches.append(kwargs)
            return queue.pop(0)

        monkeypatch.setattr(main.uc, "start", _start)
        await stack.enter_async_context(main.app.router.lifespan_context(main.app))
        first = browsers[0] if browsers else None
        if first is not None and hasattr(first, "booted"):
            first.booted = True
        return launches

    yield _boot
    await stack.aclose()


def _warm_up_tab() -> _FakeTab:
    """A throwaway tab for the startup warm-up render, so it never touches a test's own tab."""
    return _FakeTab(html="<html></html>", url="https://example.com")


class TestHealthz:
    async def test_not_ready_before_startup(self, client: httpx.AsyncClient):
        async with client:
            r = await client.get("/healthz")
        # The status field is the contract; the body is an envelope that gains fields.
        # Asserting the whole dict made every additive field a breaking change.
        assert r.json()["status"] == "starting"

    async def test_not_ready_while_browser_started_but_warm_up_incomplete(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ):
        """Browser started =/= ready -- the warm-up render must complete (or
        fail open) first, matching the cold-start mitigation's own contract.

        Startup is held INSIDE the warm-up render, so the browser genuinely exists and the
        warm-up genuinely has not finished when /healthz is asked -- then released, to prove the
        same probe flips to ready once it does.
        """
        warm_up_reached = asyncio.Event()
        release = asyncio.Event()

        class _GatedWarmUpTab(_FakeTab):
            async def get_content(self) -> str:
                warm_up_reached.set()
                await release.wait()
                return await super().get_content()

        browser = _FakeBrowser(
            tab=_GatedWarmUpTab(html="<html></html>", url="https://example.com"), including_warm_up=True
        )

        async def _start(**_kwargs: object) -> _FakeBrowser:
            return browser

        monkeypatch.setattr(main.uc, "start", _start)
        lifespan = main.app.router.lifespan_context(main.app)
        startup = asyncio.create_task(lifespan.__aenter__())
        try:
            await asyncio.wait_for(warm_up_reached.wait(), timeout=5.0)
            async with client:
                during = await client.get("/healthz")
                release.set()
                await startup
                after = await client.get("/healthz")
        finally:
            release.set()
            await startup
            await lifespan.__aexit__(None, None, None)
        assert during.json()["status"] == "starting"
        assert after.json()["status"] == "ok"

    async def test_ready_once_warm_up_completes(self, client: httpx.AsyncClient, boot):
        await boot(_FakeBrowser())
        async with client:
            r = await client.get("/healthz")
        assert r.json()["status"] == "ok"

    async def test_a_stopped_container_does_not_report_ready(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ):
        """After shutdown the browser is gone, and /healthz and the routes have to say so.

        Shutdown used to stop the browser and leave the module still holding it, so a probe
        arriving after teardown was told "ok" by a container with no browser, and a render was
        handed to a stopped process instead of being answered "not_ready".
        """
        browser = _FakeBrowser()

        async def _start(**_kwargs: object) -> _FakeBrowser:
            return browser

        monkeypatch.setattr(main.uc, "start", _start)
        async with main.app.router.lifespan_context(main.app):
            pass

        async with client:
            health = await client.get("/healthz")
            render = await client.post(
                "/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None}
            )
        assert browser.stopped is True
        assert health.json()["status"] == "starting", "a stopped container still reported ready"
        assert render.status_code == 503
        assert render.json()["error"]["code"] == "not_ready"


class TestRenderContract:
    async def test_returns_503_when_browser_not_started(self, client: httpx.AsyncClient):
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "not_ready"

    async def test_success_shape(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html>hi</html>", url="https://example.gov/final", response_status=200)
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": ".content"}
            )
        assert r.status_code == 200
        body = r.json()
        assert body["html"] == "<html>hi</html>"
        assert body["status"] == 200
        assert body["final_url"] == "https://example.gov/final"
        assert isinstance(body["timing_ms"], float)
        assert tab.selected_for == ".content"
        assert tab.slept is None
        assert tab.closed is True

    async def test_navigation_timeout(self, client: httpx.AsyncClient, boot):
        await boot(_FakeBrowser(hang=True))
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 0.05, "wait_for": None})
        assert r.status_code == 504
        assert r.json()["error"]["code"] == "navigation_timeout"

    async def test_driver_crash(self, client: httpx.AsyncClient, boot):
        await boot(_FakeBrowser(raise_exc=RuntimeError("chromium crashed")))
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.status_code == 502
        assert r.json()["error"]["code"] == "driver_crash"
        assert "chromium crashed" in r.json()["error"]["message"]

    async def test_no_wait_for_skips_select_but_settles(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.status_code == 200
        assert tab.selected_for is None
        assert tab.slept == 1.0


class TestRenderRealStatus:
    """The sidecar must surface the real top-level HTTP status
    (a successfully-rendered 404/500 page is not the same as a driver crash)
    instead of always reporting 200."""

    async def test_real_404_page_reports_404_not_200(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html>not found</html>", url="https://example.gov/missing", response_status=404)
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render", json={"url": "https://example.gov/missing", "timeout": 5.0, "wait_for": None}
            )
        assert r.status_code == 200  # the render itself succeeded -- a 404 page is still real content
        assert r.json()["status"] == 404

    async def test_no_response_event_falls_back_to_200_and_requested_url(self, client: httpx.AsyncClient, boot):
        """No CDP event fired at all (e.g. a same-document navigation) -- fails
        open to 200/the originally-requested url rather than raising or
        leaving either field unset."""
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        body = r.json()
        assert body["status"] == 200
        assert body["final_url"] == "https://example.gov"

    async def test_final_url_reflects_redirect_not_originally_requested_url(self, client: httpx.AsyncClient, boot):
        """final_url is sourced from the captured response, not `tab.url` --
        proves a redirect (requested example.gov/start, landed on
        example.gov/final) is reported correctly."""
        tab = _FakeTab(html="<html></html>", url="https://example.gov/final")

        async def _send_with_redirect_url(cmd: object) -> None:
            tab.sent.append(cmd)
            if getattr(getattr(cmd, "gi_code", None), "co_name", None) == "navigate":
                tab.fire_response(200, response_url="https://example.gov/final")

        tab.send = _send_with_redirect_url
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render", json={"url": "https://example.gov/start", "timeout": 5.0, "wait_for": None}
            )
        assert r.json()["final_url"] == "https://example.gov/final"

    async def test_subresource_response_ignored(self, client: httpx.AsyncClient, boot):
        """An image/script response for the same frame must not overwrite the
        document's own status -- only ResourceType.DOCUMENT counts."""
        tab = _FakeTab(html="<html></html>", url="https://example.gov")

        async def _send_with_subresource_noise(cmd: object) -> None:
            tab.sent.append(cmd)
            if getattr(getattr(cmd, "gi_code", None), "co_name", None) == "navigate":
                tab.fire_response(200, resource_type=uc.cdp.network.ResourceType.IMAGE)
                tab.fire_response(200, resource_type=uc.cdp.network.ResourceType.DOCUMENT)
                tab.fire_response(999, resource_type=uc.cdp.network.ResourceType.IMAGE)  # a later sub-resource

        tab.send = _send_with_subresource_noise
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.json()["status"] == 200  # the DOCUMENT status, not the later IMAGE noise

    async def test_iframe_response_for_different_frame_ignored(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")

        async def _send_with_iframe_noise(cmd: object) -> None:
            tab.sent.append(cmd)
            if getattr(getattr(cmd, "gi_code", None), "co_name", None) == "navigate":
                tab.fire_response(500, frame_id="some-other-iframe-id")
                tab.fire_response(200, frame_id=tab.target.target_id)

        tab.send = _send_with_iframe_noise
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.json()["status"] == 200

    async def test_redirect_chain_reports_final_status_not_first(self, client: httpx.AsyncClient, boot):
        """A 301 -> 200 redirect chain fires two DOCUMENT events for the same
        frame; the actually-rendered page's status (the last one) must win."""
        tab = _FakeTab(html="<html></html>", url="https://example.gov/final")

        async def _send_with_redirect(cmd: object) -> None:
            tab.sent.append(cmd)
            if getattr(getattr(cmd, "gi_code", None), "co_name", None) == "navigate":
                tab.fire_response(301)
                tab.fire_response(200)

        tab.send = _send_with_redirect
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.json()["status"] == 200


class TestNetworkCapture:
    """Network/API-detection capability (2026-07-14): capture_network=True
    captures XHR/fetch calls with JSON-shaped bodies."""

    async def test_capture_network_false_returns_no_calls(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            network_calls_to_fire=[
                {"request_id": "r1", "url": "https://example.gov/api/notices", "body": '{"notices": []}'}
            ],
        )
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": False},
            )
        assert r.json()["network_calls"] == []

    async def test_captures_a_real_json_call(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            network_calls_to_fire=[
                {
                    "request_id": "r1",
                    "url": "https://example.gov/api/notices",
                    "method": "GET",
                    "status": 200,
                    "content_type": "application/json",
                    "body": '{"notices": [1, 2]}',
                }
            ],
        )
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": True},
            )
        calls = r.json()["network_calls"]
        assert len(calls) == 1
        assert calls[0] == {
            "url": "https://example.gov/api/notices",
            "method": "GET",
            "status": 200,
            "content_type": "application/json",
            "body": '{"notices": [1, 2]}',
            # None, not "" -- this GET carried no body, which is a different fact
            # from carrying an empty one.
            "request_body": None,
        }

    async def test_captures_a_post_requests_payload(self, client: httpx.AsyncClient, boot):
        """A POST-read API's payload is its query -- without it the call cannot
        be replayed. CDP carries it on the RequestWillBeSent event's request."""
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            network_calls_to_fire=[
                {
                    "request_id": "r1",
                    "url": "https://api.example.gov/api/Grids/GetData",
                    "method": "POST",
                    "body": '{"data": {"items": [{"id": 1}]}}',
                    "post_data": '{"pageNumber": 1, "pageSize": 1000}',
                }
            ],
        )
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": True},
            )
        calls = r.json()["network_calls"]
        assert len(calls) == 1
        assert calls[0]["method"] == "POST"
        assert calls[0]["request_body"] == '{"pageNumber": 1, "pageSize": 1000}'

    async def test_non_json_body_is_not_captured(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            network_calls_to_fire=[
                {"request_id": "r1", "url": "https://example.gov/api/frag", "body": "<div>not json</div>"}
            ],
        )
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": True},
            )
        assert r.json()["network_calls"] == []

    async def test_xssi_prefixed_json_body_is_captured(self, client: httpx.AsyncClient, boot):
        """A real API's anti-JSON-hijacking prefix (e.g. Google's own internal
        APIs, live-verified 2026-07-17) must not cause a genuinely JSON-shaped
        response to be dropped as "not JSON-shaped" -- the prefix is stripped
        only for the shape check; the captured body stays the original,
        unmodified bytes a real caller would need to parse it correctly."""
        prefixed_body = ')]}\'\n{"widgets": [{"id": "TIMESERIES", "token": "abc"}]}'
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            network_calls_to_fire=[
                {"request_id": "r1", "url": "https://trends.google.com/trends/api/explore", "body": prefixed_body}
            ],
        )
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": True},
            )
        calls = r.json()["network_calls"]
        assert len(calls) == 1
        assert calls[0]["body"] == prefixed_body  # original bytes, prefix intact

    async def test_genuinely_non_json_body_still_dropped_after_prefix_stripping(self, client: httpx.AsyncClient, boot):
        """A body that merely starts with the XSSI prefix but isn't actually
        JSON-shaped underneath it must still be dropped -- the fix strips a
        known prefix before the shape check, it does not loosen the check itself."""
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            network_calls_to_fire=[
                {"request_id": "r1", "url": "https://example.gov/api/frag", "body": ")]}'\n<div>not json</div>"}
            ],
        )
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": True},
            )
        assert r.json()["network_calls"] == []

    async def test_non_xhr_fetch_resource_type_is_not_captured(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            network_calls_to_fire=[
                {
                    "request_id": "r1",
                    "url": "https://example.gov/style.css",
                    "resource_type": uc.cdp.network.ResourceType.STYLESHEET,
                    "body": '{"looks": "json but is not an api call"}',
                }
            ],
        )
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": True},
            )
        assert r.json()["network_calls"] == []

    async def test_base64_body_is_not_captured(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            network_calls_to_fire=[
                {"request_id": "r1", "url": "https://example.gov/api/binary", "body": "e30=", "is_base64": True}
            ],
        )
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": True},
            )
        assert r.json()["network_calls"] == []

    async def test_multiple_calls_all_captured(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            network_calls_to_fire=[
                {"request_id": "r1", "url": "https://example.gov/api/one", "body": '{"a": 1}'},
                {"request_id": "r2", "url": "https://example.gov/api/two", "body": '{"b": 2}'},
            ],
        )
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": True},
            )
        urls = {c["url"] for c in r.json()["network_calls"]}
        assert urls == {"https://example.gov/api/one", "https://example.gov/api/two"}


class TestNavSteps:
    """Multi-step navigation capability (2026-07-14)."""

    async def test_no_nav_steps_selects_nothing_extra(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.status_code == 200
        assert tab.select_calls == []

    async def test_click_step_selects_and_clicks(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors={"#search"})
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "click", "selector": "#search"}],
                },
            )
        assert r.status_code == 200
        assert tab.select_calls == ["#search"]

    async def test_fill_step_clears_and_sends_keys(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors={"#q"})
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "fill", "selector": "#q", "value": "Maine"}],
                },
            )
        assert r.status_code == 200
        assert tab.select_calls == ["#q"]

    async def test_wait_for_step_selects_without_clicking_or_typing(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors={".results"})
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "wait_for", "selector": ".results"}],
                },
            )
        assert r.status_code == 200
        assert tab.select_calls == [".results"]

    async def test_scroll_into_view_step_selects_and_scrolls(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors={"#chart"})
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "scroll_into_view", "selector": "#chart"}],
                },
            )
        assert r.status_code == 200
        assert tab.select_calls == ["#chart"]

    async def test_scroll_page_step_scrolls_with_the_given_amount(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "scroll_page", "value": "50"}],
                },
            )
        assert r.status_code == 200
        assert tab.scroll_down_calls == [50]
        assert tab.select_calls == []  # no selector needed at all

    async def test_scroll_page_step_uses_the_default_amount_when_value_omitted(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "scroll_page"}],
                },
            )
        assert r.status_code == 200
        # 25 is a quarter of the viewport, nodriver's own `Tab.scroll_down` default, which the
        # sidecar adopts for a step that names no amount.
        assert tab.scroll_down_calls == [25]

    async def test_scroll_page_step_non_int_value_returns_422_nav_step_failed(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "scroll_page", "value": "not-a-number"}],
                },
            )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "nav_step_failed"

    async def test_evaluate_step_runs_the_expression_and_records_the_result(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", evaluate_returns=[{"foo": "bar"}])
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "evaluate", "value": "({foo: 'bar'})"}],
                },
            )
        assert r.status_code == 200
        assert tab.evaluate_calls == ["({foo: 'bar'})"]
        assert r.json()["eval_results"] == [{"foo": "bar"}]

    async def test_evaluate_step_records_each_step_result_in_order(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", evaluate_returns=[1, 2])
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [
                        {"action": "evaluate", "value": "1"},
                        {"action": "evaluate", "value": "2"},
                    ],
                },
            )
        assert r.status_code == 200
        assert r.json()["eval_results"] == [1, 2]

    async def test_no_evaluate_steps_leaves_eval_results_empty(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.status_code == 200
        assert r.json()["eval_results"] == []

    async def test_evaluate_step_protocol_exception_returns_422_nav_step_failed(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", evaluate_exc=ProtocolException("gone"))
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "evaluate", "value": "1"}],
                },
            )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "nav_step_failed"

    async def test_wait_ms_step_sleeps_the_given_duration(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "wait_ms", "ms": 250}],
                },
            )
        assert r.status_code == 200
        # sleep_calls[0] is the settle sleep before nav_steps begin executing
        # (see the module-level rationale on tab.sleep(1.0) in main.py); the
        # wait_ms step's own sleep is the one after it.
        assert tab.sleep_calls[1] == 0.25

    async def test_steps_execute_in_order_before_the_final_wait_for(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors={"#q", "#submit", ".final"})
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": ".final",
                    "nav_steps": [
                        {"action": "fill", "selector": "#q", "value": "Maine"},
                        {"action": "click", "selector": "#submit"},
                    ],
                },
            )
        assert r.status_code == 200
        assert tab.select_calls == ["#q", "#submit", ".final"]

    async def test_click_step_selector_never_found_returns_422_nav_step_failed(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors=set())
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "click", "selector": "#missing"}],
                },
            )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "nav_step_failed"
        assert "#missing" in r.json()["error"]["message"]

    async def test_a_failed_nav_step_still_closes_the_tab(self, client: httpx.AsyncClient, boot):
        """A nav step failure must not leak the tab -- the same discipline
        driver_crash/navigation_timeout already need, extended to this new
        failure mode (tab.close() moved into the shared finally block)."""
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors=set())
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "click", "selector": "#missing"}],
                },
            )
        assert r.status_code == 422
        assert tab.closed is True

    async def test_a_failing_step_aborts_before_the_final_settle_wait(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors=set())
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": ".final",
                    "nav_steps": [{"action": "click", "selector": "#missing"}],
                },
            )
        assert r.status_code == 422
        # the final settle wait_for's own select() call for ".final" never happened
        assert tab.select_calls == ["#missing"]

    async def test_a_transient_stale_node_error_is_retried_and_succeeds(self, client: httpx.AsyncClient, boot):
        """Live-reproduced (2026-07-14): a resolved element can go stale
        before click() actually runs, raising ProtocolException -- retrying
        the whole find-then-act sequence (a fresh select() re-queries the
        live DOM) resolves it, matching the real observed behavior (0/6 and
        4/6 failure rates across otherwise-identical live runs)."""
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            findable_selectors={"#search"},
            select_protocol_exceptions=2,
        )
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "click", "selector": "#search"}],
                },
            )
        assert r.status_code == 200
        assert tab.select_calls == ["#search", "#search", "#search"]

    async def test_stale_node_error_exhausting_every_retry_returns_422(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            findable_selectors={"#search"},
            select_protocol_exceptions=99,
        )
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "click", "selector": "#search"}],
                },
            )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "nav_step_failed"
        assert tab.closed is True

    async def test_final_wait_for_also_retries_the_same_stale_node_race(self, client: httpx.AsyncClient, boot):
        """Not nav_steps-specific -- the pre-existing wait_for settle-wait
        call hit the identical race live, with zero nav_steps involved."""
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            findable_selectors={"table"},
            select_protocol_exceptions=2,
        )
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": "table"}
            )
        assert r.status_code == 200
        assert tab.select_calls == ["table", "table", "table"]

    async def test_unsupported_action_returns_422_nav_step_failed(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        await boot(_FakeBrowser(tab=tab))
        async with client:
            r = await client.post(
                "/v1/render",
                json={
                    "url": "https://example.gov",
                    "timeout": 5.0,
                    "wait_for": None,
                    "nav_steps": [{"action": "scroll_to", "selector": "#x"}],
                },
            )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "nav_step_failed"
        assert "scroll_to" in r.json()["error"]["message"]


class TestWarmUp:
    """Cold-start mitigation (2026-07-14): a real warm-up render must
    complete -- or fail open -- before /healthz reports "ok".

    Driven through container startup, which is the only thing that runs the warm-up, and read
    back through /healthz, which is the only thing that reports what it decided.
    """

    async def test_succeeds_on_first_attempt_marks_ready(self, client: httpx.AsyncClient, boot):
        tab = _FakeTab(html="<html></html>", url="https://example.com/")
        browser = _FakeBrowser(tab=tab, including_warm_up=True)

        await boot(browser)

        async with client:
            r = await client.get("/healthz")
        assert r.json()["status"] == "ok"
        assert browser.get_calls == 1
        assert tab.closed is True  # warm-up renders through the real _render path

    async def test_retries_then_succeeds(self, client: httpx.AsyncClient, boot, monkeypatch):
        tab = _FakeTab(html="<html></html>", url="https://example.com/")
        browser = _FakeBrowser(tab=tab, fail_times=2, including_warm_up=True)  # fails twice, succeeds on the 3rd
        _use_timings(monkeypatch, warmup_retry_delay_seconds=0.0)

        await boot(browser)

        async with client:
            r = await client.get("/healthz")
        assert r.json()["status"] == "ok"
        assert browser.get_calls == 3

    async def test_fails_open_after_exhausting_retries(self, client: httpx.AsyncClient, boot, monkeypatch):
        """A warm-up that never succeeds must not block startup forever --
        marks ready anyway, logged loudly (the real first request would hit
        the same failure mode this mitigation is tolerant of, not a new one)."""
        browser = _FakeBrowser(raise_exc=RuntimeError("cold-start failure"), including_warm_up=True)
        _use_timings(monkeypatch, warmup_attempts=3, warmup_retry_delay_seconds=0.0)

        await boot(browser)

        async with client:
            r = await client.get("/healthz")
        assert r.json()["status"] == "ok"
        assert browser.get_calls == 3


# ===========================================================================
# POST /v1/download (browser-forced-download capability; design and live
# verification in docs/scrape-task-04-multi-document-driver.md)
# ===========================================================================


# parity-exempt: hand-rolled subset stub of nodriver's third-party Tab for the download path (only .target/send()/close()); nodriver is AGPL-isolated to this sidecar and never installed in the workspace venv, so a parity-with marker cannot resolve there
class _FakeDownloadTab:
    """A tab created within an isolated browser context for a download request."""

    def __init__(self, target_id: str, context_id: str, owner: "_FakeDownloadBrowser") -> None:
        self.target = SimpleNamespace(target_id=target_id, browser_context_id=context_id)
        self._context_id = context_id
        self._owner = owner
        self.closed = False
        self.navigated_to: str | None = None

    async def send(self, cmd: object):
        co_name = getattr(getattr(cmd, "gi_code", None), "co_name", None)
        if co_name == "navigate":
            url = cmd.gi_frame.f_locals["url"]
            self.navigated_to = url
            await self._owner.simulate_navigation(self._context_id, url)
        return None

    async def close(self) -> None:
        self.closed = True


# parity-exempt: hand-rolled subset stub of nodriver's third-party Browser for the download path (the isolated-context surface _download/create_isolated_tab use, plus the get() container startup warms up through); nodriver is AGPL-isolated to this sidecar and never installed in the workspace venv, so a parity-with marker cannot resolve there
class _FakeDownloadBrowser:
    """Fakes the subset of nodriver's ``Browser`` surface ``_download``/``create_isolated_tab`` use.

    Simulates a real download by writing a file into whatever
    ``download_path`` was configured for a context, the moment a tab within
    that context "navigates" -- real arguments extracted via
    ``cmd.gi_frame.f_locals`` (live-verified this works for nodriver's own
    CDP generator commands), not guessed at.
    """

    def __init__(
        self,
        *,
        file_content: bytes | None = b"%PDF-fake-content",
        write_delay_seconds: float = 0.0,
        never_writes: bool = False,
        target_lookup_failures: int = 0,
        context_dispose_raises: bool = False,
    ) -> None:
        self.sent: list[object] = []
        self.targets: list[_FakeDownloadTab] = []
        self._pending_targets: dict[str, _FakeDownloadTab] = {}
        self._context_download_paths: dict[str, str] = {}
        self._file_content = file_content
        self._write_delay_seconds = write_delay_seconds
        self._never_writes = never_writes
        self._target_lookup_failures_remaining = target_lookup_failures
        self._context_dispose_raises = context_dispose_raises
        self.disposed_contexts: list[str] = []
        self._context_counter = 0
        self._target_counter = 0
        self.booted = False

    async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
        """Serve container startup's warm-up render; downloads never come through here."""
        return _warm_up_tab()

    async def send(self, cmd: object):
        self.sent.append(cmd)
        co_name = getattr(getattr(cmd, "gi_code", None), "co_name", None)
        if co_name == "create_browser_context":
            self._context_counter += 1
            return f"ctx-{self._context_counter}"
        if co_name == "create_target":
            context_id = cmd.gi_frame.f_locals["browser_context_id"]
            self._target_counter += 1
            target_id = f"target-{self._target_counter}"
            self._pending_targets[target_id] = _FakeDownloadTab(target_id, context_id, self)
            return target_id
        if co_name == "set_download_behavior":
            context_id = cmd.gi_frame.f_locals["browser_context_id"]
            download_path = cmd.gi_frame.f_locals["download_path"]
            self._context_download_paths[context_id] = download_path
            return None
        if co_name == "dispose_browser_context":
            context_id = cmd.gi_frame.f_locals["browser_context_id"]
            self.disposed_contexts.append(context_id)
            if self._context_dispose_raises:
                raise RuntimeError("simulated context disposal failure")
            return None
        return None

    async def update_targets(self) -> None:
        if self._target_lookup_failures_remaining > 0:
            self._target_lookup_failures_remaining -= 1
            return
        self.targets.extend(self._pending_targets.values())
        self._pending_targets.clear()

    async def simulate_navigation(self, context_id: str, url: str) -> None:
        if self._never_writes:
            return
        if self._write_delay_seconds:
            await asyncio.sleep(self._write_delay_seconds)
        download_path = self._context_download_paths.get(context_id)
        if download_path is None:
            return
        filename = url.rsplit("/", 1)[-1] or "download.pdf"
        with open(os.path.join(download_path, filename), "wb") as f:
            f.write(self._file_content or b"")

    def stop(self) -> None:
        pass


class TestDownloadContract:
    async def test_returns_503_when_browser_not_started(self, client: httpx.AsyncClient):
        async with client:
            r = await client.post("/v1/download", json={"url": "https://example.gov/notice.pdf", "timeout": 5.0})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "not_ready"

    async def test_success_shape(self, client: httpx.AsyncClient, boot):
        browser = _FakeDownloadBrowser(file_content=b"%PDF-1.7 real content")
        await boot(browser)
        async with client:
            r = await client.post("/v1/download", json={"url": "https://example.gov/notice.pdf", "timeout": 5.0})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == 200
        assert body["filename"] == "notice.pdf"
        assert body["content_type"] == "application/pdf"
        assert base64.b64decode(body["content_base64"]) == b"%PDF-1.7 real content"
        assert body["timing_ms"] >= 0

    async def test_isolated_context_is_disposed_after_download(self, client: httpx.AsyncClient, boot):
        browser = _FakeDownloadBrowser()
        await boot(browser)
        async with client:
            await client.post("/v1/download", json={"url": "https://example.gov/notice.pdf", "timeout": 5.0})
        assert len(browser.disposed_contexts) == 1

    async def test_tab_is_closed_after_download(self, client: httpx.AsyncClient, boot):
        browser = _FakeDownloadBrowser()
        await boot(browser)
        async with client:
            await client.post("/v1/download", json={"url": "https://example.gov/notice.pdf", "timeout": 5.0})
        assert len(browser.targets) == 1
        assert browser.targets[0].closed is True

    async def test_download_that_never_completes_returns_504(self, client: httpx.AsyncClient, boot, monkeypatch):
        _use_timings(monkeypatch, download_poll_interval_seconds=0.01)
        browser = _FakeDownloadBrowser(never_writes=True)
        await boot(browser)
        async with client:
            r = await client.post("/v1/download", json={"url": "https://example.gov/notice.pdf", "timeout": 0.1})
        assert r.status_code == 504
        assert r.json()["error"]["code"] == "download_timeout"

    async def test_crdownload_in_progress_file_is_not_treated_as_complete(
        self, client: httpx.AsyncClient, boot, monkeypatch
    ):
        """A file still being written carries a .crdownload suffix -- must not
        be mistaken for a completed download."""
        _use_timings(monkeypatch, download_poll_interval_seconds=0.01)

        class _PartialThenCompleteBrowser(_FakeDownloadBrowser):
            async def simulate_navigation(self, context_id, url):
                download_path = self._context_download_paths.get(context_id)
                with open(os.path.join(download_path, "notice.pdf.crdownload"), "wb") as f:
                    f.write(b"partial")
                await asyncio.sleep(0.05)
                os.remove(os.path.join(download_path, "notice.pdf.crdownload"))
                with open(os.path.join(download_path, "notice.pdf"), "wb") as f:
                    f.write(b"%PDF-complete")

        browser = _PartialThenCompleteBrowser()
        await boot(browser)
        async with client:
            r = await client.post("/v1/download", json={"url": "https://example.gov/notice.pdf", "timeout": 5.0})
        assert r.status_code == 200
        assert base64.b64decode(r.json()["content_base64"]) == b"%PDF-complete"

    async def test_non_pdf_filename_gets_octet_stream_content_type(self, client: httpx.AsyncClient, boot):
        browser = _FakeDownloadBrowser()
        await boot(browser)
        async with client:
            r = await client.post("/v1/download", json={"url": "https://example.gov/notice.docx", "timeout": 5.0})
        assert r.json()["content_type"] == "application/octet-stream"

    async def test_target_lookup_retries_past_transient_propagation_delay(
        self, client: httpx.AsyncClient, boot, monkeypatch
    ):
        """Live-reproduced (2026-07-15): a freshly created target does not
        always appear in browser.targets on the first update_targets() call."""
        _use_timings(monkeypatch, tab_lookup_delay_seconds=0.01)
        browser = _FakeDownloadBrowser(target_lookup_failures=3)
        await boot(browser)
        async with client:
            r = await client.post("/v1/download", json={"url": "https://example.gov/notice.pdf", "timeout": 5.0})
        assert r.status_code == 200

    async def test_target_never_appearing_reports_driver_crash(self, client: httpx.AsyncClient, boot, monkeypatch):
        _use_timings(monkeypatch, tab_lookup_attempts=3, tab_lookup_delay_seconds=0.01)
        browser = _FakeDownloadBrowser(target_lookup_failures=999)
        await boot(browser)
        async with client:
            r = await client.post("/v1/download", json={"url": "https://example.gov/notice.pdf", "timeout": 5.0})
        assert r.status_code == 502
        assert r.json()["error"]["code"] == "driver_crash"

    async def test_context_disposal_failure_does_not_mask_a_successful_download(self, client: httpx.AsyncClient, boot):
        browser = _FakeDownloadBrowser(context_dispose_raises=True)
        await boot(browser)
        async with client:
            r = await client.post("/v1/download", json={"url": "https://example.gov/notice.pdf", "timeout": 5.0})
        assert r.status_code == 200

    async def test_concurrent_downloads_do_not_cross_contaminate(self, client: httpx.AsyncClient, boot):
        """Two concurrent requests must each get their own isolated context/download
        directory -- mirrors the real live-verified concurrency proof recorded in
        docs/scrape-task-04-multi-document-driver.md."""
        browser = _FakeDownloadBrowser()
        await boot(browser)
        async with client:
            r1, r2 = await asyncio.gather(
                client.post("/v1/download", json={"url": "https://example.gov/a.pdf", "timeout": 5.0}),
                client.post("/v1/download", json={"url": "https://example.gov/b.pdf", "timeout": 5.0}),
            )
        assert r1.status_code == 200
        assert r2.status_code == 200
        filenames = {r1.json()["filename"], r2.json()["filename"]}
        assert filenames == {"a.pdf", "b.pdf"}
        assert len(browser.disposed_contexts) == 2
        assert len(set(browser.disposed_contexts)) == 2  # each context disposed exactly once, no reuse/collision


class TestEgressReporting:
    """Which exit this container leaves by, visible from outside it."""

    async def test_healthz_reports_nothing_when_nothing_is_configured(self) -> None:
        """A deployment running one container per exit needs to confirm which one it reached.

        Without this the only way to tell a TOR container from a direct one is to fetch an
        address-echo service through it, which is a network round trip to answer a question
        the container already knows.

        ``null`` rather than ``"direct"``: this value is written through to
        ``ScrapeTargetHealth.last_egress``, whose convention is that a name means somebody chose
        an exit. An unconfigured container reporting "direct" stamped that claim on every row.
        """
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
            body = (await client.get("/healthz")).json()
        assert body["egress"] is None

    async def test_healthz_reports_the_name_a_deployment_chose(self, monkeypatch) -> None:
        """The other half: a stated choice is reported, so ``null`` above means absence not silence.

        Without this the assertion above passes just as well against a ``/healthz`` that never
        reports an egress at all.
        """
        monkeypatch.setattr(main, "EGRESS_NAME", "tor")
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
            body = (await client.get("/healthz")).json()
        assert body["egress"] == "tor"

    async def test_a_configured_proxy_reaches_the_browser_args(self, boot, monkeypatch) -> None:
        """Reads the list PRODUCTION hands the browser launch, which the first version did not.

        That version rebuilt the argument list inside itself and asserted on its own copy, so
        deleting the production line left it green -- a test of the test. The failure being
        guarded is the argument never being built, which looks identical from inside the
        container and only differs at the far end, where nobody is watching. Read at the
        launch itself, so an argument built and then not passed fails too.
        """
        monkeypatch.setattr(main, "EGRESS_PROXY", "socks5://tor:9050")
        (launch,) = await boot(_FakeBrowser())
        assert "--proxy-server=socks5://tor:9050" in launch["browser_args"]

    async def test_no_proxy_configured_adds_no_argument(self, boot, monkeypatch) -> None:
        """The default route stays the default route, with no empty --proxy-server."""
        monkeypatch.setattr(main, "EGRESS_PROXY", None)
        (launch,) = await boot(_FakeBrowser())
        args = launch["browser_args"]
        assert not any(a.startswith("--proxy-server") for a in args)
        assert "--disable-dev-shm-usage" in args, "the other launch arguments were lost"

    def test_a_per_request_exit_gets_its_own_context(self, monkeypatch) -> None:
        """The capability three artifacts said Chromium did not have.

        `--proxy-server` is process-wide, which is what made "one container is one exit" look
        true. `Target.createBrowserContext` takes its own `proxyServer`, so a render can leave
        by a different exit than the container's default without a second container.

        Asserted on the CDP command this builds, because that is the whole mechanism: the
        argument is accepted silently and only differs at the far end.
        """
        import nodriver as uc

        cmd = uc.cdp.target.create_browser_context(proxy_server="socks5://tor:9050")
        payload = next(iter(cmd)) if hasattr(cmd, "__iter__") else cmd
        assert "proxyServer" in str(payload) or "proxy_server" in str(payload), (
            "per-context proxying is not reaching CDP, so a per-request exit is silently the default one"
        )

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ({"egress_proxy": "socks5://tor:9050", "egress_name": "tor"}, "tor"),
            ({"egress_proxy": "socks5://tor:9050"}, "unnamed"),
            # `DirectEgress`. The container here has an exit configured, so this is the case
            # that distinguishes honouring the selection from ignoring it: falling through to
            # the shared browser would report "container-default" AND route the request out
            # through the container's `--proxy-server`, while the caller had asked for neither.
            ({"egress_proxy": "direct://", "egress_name": "direct"}, "direct"),
            # A name with no proxy to take it. `RenderRequest.egress_name` documents this to
            # consumers as reporting the CONTAINER's exit rather than the name sent, because
            # nothing routed the render anywhere -- so a consumer that assumed an echo would
            # record a route never used.
            #
            # No producer among the SHIPPED drivers, which is why it had no test. It is not
            # unreachable: `EgressDriver.browser_proxy_arg()` is documented as returning
            # `None` to express no opinion about the browser's proxy, and the driver forwards
            # that verbatim alongside the name -- so a third-party exit written to the
            # protocol posts exactly this body. A stated contract with no test is how the
            # comment this row pins drifted into being wrong once already.
            ({"egress_name": "tor"}, "container-default"),
            ({}, "container-default"),
        ],
        ids=[
            "named-request-exit",
            "unnamed-request-exit",
            "explicit-direct",
            "name-without-proxy",
            "container-default",
        ],
    )
    async def test_the_response_reports_the_exit_the_render_used(
        self, client, boot, monkeypatch, body: dict, expected: str
    ) -> None:
        """Driven through the real endpoint, because the previous version was a test of itself.

        That version re-typed the production expression inside the test and compared it
        against literals, so deleting `egress=` from the RenderResponse left it green. It sat
        four tests below one whose docstring describes exactly that defect -- "rebuilt the
        argument list inside itself and asserted on its own copy" -- which is how a pattern
        survives being named.

        What matters to a caller is that the RESPONSE carries the exit, so the response is
        what gets read.
        """
        monkeypatch.setattr(main, "EGRESS_NAME", "container-default")
        tab = _FakeTab(html="<html>hi</html>", url="https://example.gov/x", response_status=200)
        await boot(_FakeBrowser(tab=tab))

        # A render that names its own exit goes through the isolated-context path, which needs
        # real CDP target bookkeeping. Substituted at the helper -- a module seam -- so the
        # endpoint, `_render`, and the response construction under test all run for real.
        async def _fake_isolated(_browser, _url, *, proxy_server=None):
            captured["proxy_server"] = proxy_server
            return tab, "ctx-fake"

        captured: dict = {}
        monkeypatch.setattr(main, "create_isolated_tab", _fake_isolated)

        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov/x", "timeout": 5.0, **body})

        assert r.status_code == 200
        assert r.json()["egress"] == expected
        if "egress_proxy" in body:
            # Membership rather than truthiness, mirroring production's `is not None`. A
            # truthiness test here would quietly stop asserting for any exit whose proxy
            # argument is falsy, which is exactly the class of bug the direct case covers.
            assert captured["proxy_server"] == body["egress_proxy"], (
                "the request's exit never reached the browser context, so the echo names an exit that was not used"
            )


# parity-exempt: hand-rolled subset stub of nodriver's third-party Tab for the startup-window path (only .target.url and get()); nodriver is AGPL-isolated to this sidecar and never installed in the workspace venv, so a parity-with marker cannot resolve there
class _FakeStartupTab:
    """A tab that remembers where it was told to go."""

    def __init__(self, url: str) -> None:
        self.target = SimpleNamespace(url=url)
        self.navigated_to: list[str] = []
        self.closed = 0

    async def get(self, url: str) -> None:
        """Record the navigation and reflect it, as a real tab's target would."""
        self.navigated_to.append(url)
        self.target.url = url

    async def close(self) -> None:
        """Record a close, so a test can prove this path does NOT take one."""
        self.closed += 1


# parity-exempt: hand-rolled subset stub of nodriver's third-party Browser for the startup-window path (tabs/update_targets, plus the get()/stop() container startup and shutdown call); nodriver is AGPL-isolated to this sidecar and never installed in the workspace venv, so a parity-with marker cannot resolve there
class _FakeStartupBrowser:
    """The tab list and the refresh the startup-window blanking walks, and enough to boot on."""

    def __init__(self, tabs: list[_FakeStartupTab]) -> None:
        self.tabs = tabs
        self.refreshed = 0

    async def update_targets(self) -> None:
        """Count the refresh; a stale target list is how this finds nothing to blank."""
        self.refreshed += 1

    async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
        """Serve the startup warm-up render from a tab of its own, never one of ``tabs``."""
        return _warm_up_tab()

    def stop(self) -> None:
        """Shutdown stops the browser; nothing to release here."""


# parity-exempt: stands in for asyncio's Process at the window-manager boundary (returncode/communicate/kill/wait, the only surface the window-manager helper uses); a stdlib class, faked rather than spawning xdotool/wmctrl that a developer machine does not have
class _FakeWmProcess:
    """A window-manager call that answers at once with *out*."""

    def __init__(self, out: bytes, returncode: int = 0) -> None:
        self._out = out
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._out, b""


class TestChromiumsIdleWindowIsNotSomethingAnOperatorCanClickOn:
    """Chromium needs one window, so the one it opens at launch lives for the container's life.

    On this image the new-tab page renders the search engine's home page, so an operator summoned
    to clear one challenge arrived at a display holding their target AND a second window that
    looked exactly like a usable browser. Reported by an operator on the real screen.

    Not cosmetic: that window belongs to the DEFAULT browser context, so it is the one place on
    the display where what somebody types is not isolated per target -- which is the promise this
    whole surface makes.

    Every test here boots the container for real, so each one asserts the WIRING as well as the
    mechanism. The two fail independently: an earlier suite drove the hiding helper directly and
    stayed green with the call deleted from startup -- the function worked and nothing invoked it.
    """

    async def test_the_new_tab_page_is_left_on_about_blank(self, boot) -> None:
        """The observed state at boot was `chrome://newtab/`, which is what renders as search."""
        tab = _FakeStartupTab("chrome://newtab/")

        await boot(_FakeStartupBrowser([tab]))

        assert tab.navigated_to == ["about:blank"], (
            "the startup window was left on the new-tab page, so an operator sees a second "
            "browser that looks usable and is not context-isolated"
        )

    async def test_it_is_navigated_rather_than_closed(self, boot) -> None:
        """By this point the warm-up render has disposed of its own tab, so this is the ONLY window.

        Closing the last window exits Chromium, which would take the sidecar down at startup --
        the reason this navigates instead, and the reason that is worth a test rather than a
        comment.
        """
        tab = _FakeStartupTab("chrome://newtab/")

        await boot(_FakeStartupBrowser([tab]))

        assert tab.closed == 0, "the startup window was closed, which exits the browser"

    async def test_a_real_page_is_left_alone(self, boot) -> None:
        """Only the startup page is touched. Navigating a session's tab away would take an
        operator's half-finished challenge with it."""
        target = _FakeStartupTab("https://example.gov/some-walled-target")

        await boot(_FakeStartupBrowser([target]))

        assert target.navigated_to == [], "a real page was blanked, losing whatever was on it"

    async def test_a_browser_that_refuses_does_not_fail_startup(self, client: httpx.AsyncClient, boot) -> None:
        """An operable deployment beats a tidy one: this runs during container startup."""

        class _Refuses(_FakeStartupBrowser):
            async def update_targets(self) -> None:
                raise RuntimeError("CDP is not answering")

        await boot(_Refuses([]))

        async with client:
            r = await client.get("/healthz")
        assert r.json()["status"] == "ok", "a browser that would not list its tabs failed startup"

    async def test_no_browser_at_all_is_not_an_error(self, client: httpx.AsyncClient, boot) -> None:
        """A launch that hands back no browser must still let startup finish.

        The warm-up fails open in that state, so the idle-window step is reached with nothing to
        talk to -- and must return rather than raise.
        """
        await boot(None)

        async with client:
            r = await client.get("/healthz")
        assert r.json()["status"] == "ok"

    async def test_the_idle_window_is_taken_off_the_screen_and_out_of_the_taskbar(self, boot, monkeypatch) -> None:
        """Both, because either alone leaves something to click on.

        A window that skips the taskbar is still sitting on the desktop; a minimised one that the
        taskbar still lists is still one click away. An operator paid by the cleared challenge
        should not have to know which of the two windows is real.
        """
        calls: list[list[str]] = []

        async def _spawn(*argv: str, **_kwargs: object) -> _FakeWmProcess:
            calls.append(list(argv))
            return _FakeWmProcess(b"6291459\n" if argv[0] == "xdotool" and "search" in argv else b"")

        monkeypatch.setattr(main.asyncio, "create_subprocess_exec", _spawn)
        await boot(_FakeStartupBrowser([]))

        joined = [" ".join(c) for c in calls]
        assert any("skip_taskbar" in c and "wmctrl" in c for c in joined), (
            f"the idle window was left in the operator's taskbar: {joined}"
        )
        assert any("windowminimize" in c for c in joined), (
            f"the idle window was left on the operator's screen: {joined}"
        )

    async def test_no_matching_window_is_not_an_error(self, boot, monkeypatch) -> None:
        """The window may legitimately be gone -- a render in flight, a browser mid-restart.

        Startup finishes, and nothing is acted on: the search is the only call made.
        """
        calls: list[list[str]] = []

        async def _nothing(*argv: str, **_kwargs: object) -> _FakeWmProcess:
            calls.append(list(argv))
            return _FakeWmProcess(b"")

        monkeypatch.setattr(main.asyncio, "create_subprocess_exec", _nothing)
        await boot(_FakeStartupBrowser([]))

        assert [c[:2] for c in calls] == [["xdotool", "search"]], f"a window that was not found was acted on: {calls}"

    async def test_a_window_manager_that_will_not_answer_does_not_fail_startup(
        self, client: httpx.AsyncClient, boot, monkeypatch
    ) -> None:
        """This runs during container startup; an operable deployment beats a tidy one.

        A binary that is not there is what a stripped image looks like, and it surfaces as the
        spawn itself failing.
        """

        async def _missing(*argv: str, **_kwargs: object) -> _FakeWmProcess:
            raise FileNotFoundError(argv[0])

        monkeypatch.setattr(main.asyncio, "create_subprocess_exec", _missing)
        await boot(_FakeStartupBrowser([]))

        async with client:
            r = await client.get("/healthz")
        assert r.json()["status"] == "ok", "a missing window manager failed container startup"

    async def test_chromium_is_launched_on_a_blank_page(self, boot) -> None:
        """So no window in the container's life ever showed something worth clicking.

        Asserted against the arguments PRODUCTION hands the browser launch, not a copy: without
        the positional URL Chromium opens its new-tab page, which on this image renders a search
        engine's home page.
        """
        (launch,) = await boot(_FakeStartupBrowser([]))

        assert "about:blank" in launch["browser_args"], (
            "Chromium is launched with no start page, so its idle window shows the new-tab page"
        )

    async def test_a_hung_window_manager_call_is_killed_rather_than_abandoned(self, boot, monkeypatch) -> None:
        """`wait_for` cancels the await, not the CHILD, which is a process leak not a timeout.

        The container is meant to run long and unattended, so a window-manager call that never
        answers would otherwise leave a process behind for its whole life. This was the one
        untested branch in the helper, which is exactly where that kind of thing survives.
        """
        killed: list[bool] = []
        reaped: list[bool] = []

        class _Hangs:
            returncode = None

            async def communicate(self) -> tuple[bytes, bytes]:
                await asyncio.sleep(3600)
                raise AssertionError("unreachable")

            def kill(self) -> None:
                killed.append(True)

            async def wait(self) -> int:
                reaped.append(True)
                return -9

        async def _spawn(*_argv: object, **_kwargs: object) -> _Hangs:
            return _Hangs()

        _use_timings(monkeypatch, wm_call_timeout_seconds=0.05)
        monkeypatch.setattr(main.asyncio, "create_subprocess_exec", _spawn)

        await boot(_FakeStartupBrowser([]))

        # One call reached: the hung search answers nothing, so there is no window to act on.
        assert killed == [True], "the hung child was abandoned rather than killed, leaking a process"
        # Reaped as well as killed. A killed child that is never waited on becomes a zombie, which
        # is a smaller leak than a running process and still a leak in a container that runs for
        # weeks. Asserted because the comment claiming it was the only thing holding it: deleting
        # the `await proc.wait()` left the whole suite green.
        assert reaped == [True], "the killed child was never reaped, so it lingers as a zombie"


# parity-exempt: hand-rolled subset stub of nodriver's third-party Browser for the HITL heal path (the isolated-context surface operator tabs use, plus a render path that can be wedged); nodriver is AGPL-isolated to this sidecar and never installed in the workspace venv, so a parity-with marker cannot resolve there
class _FakeSharedBrowser(_FakeDownloadBrowser):
    """The ONE Chromium both HITL tabs and renders run in, whose render path can be wedged.

    Operator tabs come from isolated contexts (the inherited surface); renders and the healer's
    probe come through ``get``, which is what hangs when the browser is wedged -- the observed
    live failure, where every new render tab timed out while the browser process stayed up.
    """

    def __init__(self, *, wedged: bool = False) -> None:
        super().__init__()
        self.wedged = wedged
        self.stopped = False
        self.render_gets = 0

    async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
        if not self.booted:
            return _warm_up_tab()
        self.render_gets += 1
        if self.wedged:
            await asyncio.sleep(3600)
        return _FakeTab(html="<html>rendered</html>", url="https://example.gov", response_status=200)

    def stop(self) -> None:
        self.stopped = True


async def _open_session(client: httpx.AsyncClient) -> tuple[str, dict[str, str]]:
    """Open a HITL session over HTTP and return its id and the header that authorizes it."""
    r = await client.post("/v1/hitl/session")
    assert r.status_code == 200, r.text
    body = r.json()
    return body["session_id"], {"Authorization": f"Bearer {body['token']}"}


class TestRenderPathHealsAfterHitl:
    """The regression guard for the live wedge (2026-09-05).

    A HITL session's isolated contexts live in the ONE Chromium the render path also drives, and
    disposing one was observed to leave that browser unable to open a fresh ``/v1/render`` tab --
    every render timed out until the container was restarted. The session manager now calls back
    into the container after any dispose; these pin that the callback probes the render path and
    relaunches the browser exactly when that probe is wedged, and not otherwise.

    Driven end to end: a real container startup, a real session opened and closed over HTTP
    against a real (stub) display, and the real session manager doing the dispose. So each test
    also proves the WIRING -- a manager that stopped calling back would leave the healthy test's
    probe count at zero and the wedged test never relaunched.

    No real Chromium: ``uc.start`` hands back the next fake in line, so a relaunch is observable
    as a second launch and the wedged browser being stopped.
    """

    @pytest.fixture(autouse=True)
    def _fast(self, monkeypatch: pytest.MonkeyPatch, x11vnc_stub: X11vncStub):
        # The probe waits this long before calling a wedge a wedge; a real 15s would make the
        # wedged-browser test glacial, and the timeout value is not what is under test.
        _use_timings(monkeypatch, hitl_healthcheck_timeout_seconds=0.05)
        # The manager production builds -- browser provider and healer wired by the same call --
        # over a display on the stub's test port rather than the production one.
        monkeypatch.setattr(main.app.state, "sessions", main.build_session_manager(vnc=lifecycle_on_test_port()))
        del x11vnc_stub

    async def test_a_healthy_render_path_is_left_alone(self, client: httpx.AsyncClient, boot) -> None:
        """A browser that still renders must not be torn down -- the probe is not a reset button."""
        healthy = _FakeSharedBrowser()
        launches = await boot(healthy)

        async with client:
            session_id, auth = await _open_session(client)
            closed = await client.delete(f"/v1/hitl/session/{session_id}", headers=auth)
            health = await client.get("/healthz")

        assert closed.status_code == 200
        assert healthy.render_gets == 1, "closing the session never probed the render path"
        assert len(launches) == 1, "the browser was relaunched despite the render path being healthy"
        assert healthy.stopped is False, "a healthy browser was needlessly replaced"
        assert health.json()["status"] == "ok"

    async def test_a_wedged_render_path_is_relaunched(self, client: httpx.AsyncClient, boot) -> None:
        """The wedge itself: the probe hangs, so the browser is swapped for a fresh, working one."""
        wedged = _FakeSharedBrowser(wedged=True)
        fresh = _FakeSharedBrowser()
        launches = await boot(wedged, fresh)

        async with client:
            session_id, auth = await _open_session(client)
            await client.delete(f"/v1/hitl/session/{session_id}", headers=auth)
            health = await client.get("/healthz")
            render = await client.post(
                "/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None}
            )

        assert len(launches) == 2, "the wedged browser was not relaunched"
        assert wedged.stopped is True, "the wedged browser was left running beside its replacement"
        assert health.json()["status"] == "ok", "the fresh browser was left not-ready after its warm-up"
        assert render.status_code == 200, "renders still go to the wedged browser after the relaunch"

    async def test_a_wedged_browser_is_not_relaunched_under_a_live_session_with_tabs(
        self, client: httpx.AsyncClient, boot
    ) -> None:
        """Withheld when it would drop an operator's open tabs.

        A relaunch drops every context, so completing ONE tab of a multi-tab session must not
        reap the others here. The wedge is real, but those tabs ride the same wedged browser and
        are the session's own close (or the reaper) to clean up once the operator is done -- so
        that close is where the relaunch then happens.
        """
        wedged = _FakeSharedBrowser(wedged=True)
        fresh = _FakeSharedBrowser()
        launches = await boot(wedged, fresh)

        async with client:
            session_id, auth = await _open_session(client)
            first = await client.post(
                f"/v1/hitl/session/{session_id}/tab",
                json={"target_id": "a", "url": "https://a.example"},
                headers=auth,
            )
            second = await client.post(
                f"/v1/hitl/session/{session_id}/tab",
                json={"target_id": "b", "url": "https://b.example"},
                headers=auth,
            )
            assert first.status_code == 200, first.text
            assert second.status_code == 200, second.text
            completed = await client.post(
                f"/v1/hitl/session/{session_id}/tab/{first.json()['tab_id']}/complete", headers=auth
            )
            assert completed.status_code == 200, completed.text

            assert wedged.render_gets == 1, "completing a tab never probed the render path"
            assert len(launches) == 1, "the browser was relaunched while a live session still held tabs"
            assert wedged.stopped is False, "an operator's open tabs were dropped by a relaunch here"

            await client.delete(f"/v1/hitl/session/{session_id}", headers=auth)

        assert len(launches) == 2, "the session's own close did not heal the wedge it had deferred"

    async def test_healing_with_no_browser_is_a_no_op(self, client: httpx.AsyncClient, boot) -> None:
        """With no browser there is nothing to probe, and the healer must not raise trying."""
        launches = await boot(None)

        async with client:
            session_id, auth = await _open_session(client)
            closed = await client.delete(f"/v1/hitl/session/{session_id}", headers=auth)

        assert closed.status_code == 200
        assert len(launches) == 1, "a browser was launched with no browser to heal"
