"""Unit tests for the sidecar's POST /v1/render contract.

No real Chromium/Xvfb involved -- ``main._browser`` is monkeypatched with a
fake object shaped like nodriver's ``Browser``/``Tab`` so these tests stay
hermetic. The real, live proof that a genuine nodriver-driven Chromium
render works end-to-end lives in a consumer repo (e.g. faidh's
tests/integration/test_scrape_nodriver_sidecar_live.py), exercised against
this container via docker compose.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import main
import nodriver as uc
import pytest
from nodriver.core.connection import ProtocolException


class _FakeElement:
    """Nav-steps (2026-07-14): fakes nodriver's ``Element`` (click/clear_input/send_keys)."""

    def __init__(self, selector: str) -> None:
        self.selector = selector
        self.clicked = False
        self.cleared = False
        self.sent_keys: str | None = None

    async def click(self) -> None:
        self.clicked = True

    async def clear_input(self) -> None:
        self.cleared = True

    async def send_keys(self, text: str) -> None:
        self.sent_keys = text


class _FakeTab:
    """SCR-7L4M: also fakes the CDP surface ``_render`` uses to capture the
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
        # _select_with_retry (2026-07-14, live-reproduced stale-CDP-node race):
        # how many of the next select() calls raise ProtocolException before
        # succeeding -- simulates a resolved element/document going stale
        # mid-sequence, decremented on every call regardless of selector.
        self._select_protocol_exceptions_remaining = select_protocol_exceptions

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

    async def get_content(self) -> str:
        return self._html

    async def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    def __init__(
        self,
        tab: _FakeTab | None = None,
        raise_exc: Exception | None = None,
        hang: bool = False,
        fail_times: int = 0,
    ) -> None:
        self._tab = tab
        self._raise_exc = raise_exc
        self._hang = hang
        self._fail_times = fail_times
        self.get_calls = 0

    async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
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
        pass


@pytest.fixture
def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=main.app)
    return httpx.AsyncClient(transport=transport, base_url="http://sidecar.test")


@pytest.fixture(autouse=True)
def _reset_browser():
    main._browser = None
    main._ready = False
    yield
    main._browser = None
    main._ready = False


class TestHealthz:
    async def test_not_ready_before_startup(self, client: httpx.AsyncClient):
        async with client:
            r = await client.get("/healthz")
        assert r.json() == {"status": "starting"}

    async def test_not_ready_while_browser_started_but_warm_up_incomplete(self, client: httpx.AsyncClient):
        """Browser started =/= ready -- the warm-up render must complete (or
        fail open) first, matching the cold-start mitigation's own contract."""
        main._browser = _FakeBrowser()
        async with client:
            r = await client.get("/healthz")
        assert r.json() == {"status": "starting"}

    async def test_ready_once_warm_up_completes(self, client: httpx.AsyncClient):
        main._browser = _FakeBrowser()
        main._ready = True
        async with client:
            r = await client.get("/healthz")
        assert r.json() == {"status": "ok"}


class TestRenderContract:
    async def test_returns_503_when_browser_not_started(self, client: httpx.AsyncClient):
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "not_ready"

    async def test_success_shape(self, client: httpx.AsyncClient):
        tab = _FakeTab(html="<html>hi</html>", url="https://example.gov/final", response_status=200)
        main._browser = _FakeBrowser(tab=tab)
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

    async def test_navigation_timeout(self, client: httpx.AsyncClient):
        main._browser = _FakeBrowser(hang=True)
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 0.05, "wait_for": None})
        assert r.status_code == 504
        assert r.json()["error"]["code"] == "navigation_timeout"

    async def test_driver_crash(self, client: httpx.AsyncClient):
        main._browser = _FakeBrowser(raise_exc=RuntimeError("chromium crashed"))
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.status_code == 502
        assert r.json()["error"]["code"] == "driver_crash"
        assert "chromium crashed" in r.json()["error"]["message"]

    async def test_no_wait_for_skips_select_but_settles(self, client: httpx.AsyncClient):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.status_code == 200
        assert tab.selected_for is None
        assert tab.slept == 1.0


class TestRenderRealStatus:
    """SCR-7L4M: the sidecar must surface the real top-level HTTP status
    (a successfully-rendered 404/500 page is not the same as a driver crash)
    instead of always reporting 200."""

    async def test_real_404_page_reports_404_not_200(self, client: httpx.AsyncClient):
        tab = _FakeTab(html="<html>not found</html>", url="https://example.gov/missing", response_status=404)
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post(
                "/v1/render", json={"url": "https://example.gov/missing", "timeout": 5.0, "wait_for": None}
            )
        assert r.status_code == 200  # the render itself succeeded -- a 404 page is still real content
        assert r.json()["status"] == 404

    async def test_no_response_event_falls_back_to_200_and_requested_url(self, client: httpx.AsyncClient):
        """No CDP event fired at all (e.g. a same-document navigation) -- fails
        open to 200/the originally-requested url rather than raising or
        leaving either field unset."""
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        body = r.json()
        assert body["status"] == 200
        assert body["final_url"] == "https://example.gov"

    async def test_final_url_reflects_redirect_not_originally_requested_url(self, client: httpx.AsyncClient):
        """final_url is sourced from the captured response, not `tab.url` --
        proves a redirect (requested example.gov/start, landed on
        example.gov/final) is reported correctly."""
        tab = _FakeTab(html="<html></html>", url="https://example.gov/final")

        async def _send_with_redirect_url(cmd: object) -> None:
            tab.sent.append(cmd)
            if getattr(getattr(cmd, "gi_code", None), "co_name", None) == "navigate":
                tab.fire_response(200, response_url="https://example.gov/final")

        tab.send = _send_with_redirect_url
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post(
                "/v1/render", json={"url": "https://example.gov/start", "timeout": 5.0, "wait_for": None}
            )
        assert r.json()["final_url"] == "https://example.gov/final"

    async def test_subresource_response_ignored(self, client: httpx.AsyncClient):
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
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.json()["status"] == 200  # the DOCUMENT status, not the later IMAGE noise

    async def test_iframe_response_for_different_frame_ignored(self, client: httpx.AsyncClient):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")

        async def _send_with_iframe_noise(cmd: object) -> None:
            tab.sent.append(cmd)
            if getattr(getattr(cmd, "gi_code", None), "co_name", None) == "navigate":
                tab.fire_response(500, frame_id="some-other-iframe-id")
                tab.fire_response(200, frame_id=tab.target.target_id)

        tab.send = _send_with_iframe_noise
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.json()["status"] == 200

    async def test_redirect_chain_reports_final_status_not_first(self, client: httpx.AsyncClient):
        """A 301 -> 200 redirect chain fires two DOCUMENT events for the same
        frame; the actually-rendered page's status (the last one) must win."""
        tab = _FakeTab(html="<html></html>", url="https://example.gov/final")

        async def _send_with_redirect(cmd: object) -> None:
            tab.sent.append(cmd)
            if getattr(getattr(cmd, "gi_code", None), "co_name", None) == "navigate":
                tab.fire_response(301)
                tab.fire_response(200)

        tab.send = _send_with_redirect
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.json()["status"] == 200


class TestNetworkCapture:
    """Network/API-detection capability (2026-07-14): capture_network=True
    captures XHR/fetch calls with JSON-shaped bodies."""

    async def test_capture_network_false_returns_no_calls(self, client: httpx.AsyncClient):
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            network_calls_to_fire=[
                {"request_id": "r1", "url": "https://example.gov/api/notices", "body": '{"notices": []}'}
            ],
        )
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": False},
            )
        assert r.json()["network_calls"] == []

    async def test_captures_a_real_json_call(self, client: httpx.AsyncClient):
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
        main._browser = _FakeBrowser(tab=tab)
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
        }

    async def test_non_json_body_is_not_captured(self, client: httpx.AsyncClient):
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            network_calls_to_fire=[
                {"request_id": "r1", "url": "https://example.gov/api/frag", "body": "<div>not json</div>"}
            ],
        )
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": True},
            )
        assert r.json()["network_calls"] == []

    async def test_non_xhr_fetch_resource_type_is_not_captured(self, client: httpx.AsyncClient):
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
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": True},
            )
        assert r.json()["network_calls"] == []

    async def test_base64_body_is_not_captured(self, client: httpx.AsyncClient):
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            network_calls_to_fire=[
                {"request_id": "r1", "url": "https://example.gov/api/binary", "body": "e30=", "is_base64": True}
            ],
        )
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": True},
            )
        assert r.json()["network_calls"] == []

    async def test_multiple_calls_all_captured(self, client: httpx.AsyncClient):
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            network_calls_to_fire=[
                {"request_id": "r1", "url": "https://example.gov/api/one", "body": '{"a": 1}'},
                {"request_id": "r2", "url": "https://example.gov/api/two", "body": '{"b": 2}'},
            ],
        )
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post(
                "/v1/render",
                json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None, "capture_network": True},
            )
        urls = {c["url"] for c in r.json()["network_calls"]}
        assert urls == {"https://example.gov/api/one", "https://example.gov/api/two"}


class TestNavSteps:
    """Multi-step navigation capability (2026-07-14)."""

    async def test_no_nav_steps_selects_nothing_extra(self, client: httpx.AsyncClient):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post("/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": None})
        assert r.status_code == 200
        assert tab.select_calls == []

    async def test_click_step_selects_and_clicks(self, client: httpx.AsyncClient):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors={"#search"})
        main._browser = _FakeBrowser(tab=tab)
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

    async def test_fill_step_clears_and_sends_keys(self, client: httpx.AsyncClient):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors={"#q"})
        main._browser = _FakeBrowser(tab=tab)
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

    async def test_wait_for_step_selects_without_clicking_or_typing(self, client: httpx.AsyncClient):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors={".results"})
        main._browser = _FakeBrowser(tab=tab)
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

    async def test_wait_ms_step_sleeps_the_given_duration(self, client: httpx.AsyncClient):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        main._browser = _FakeBrowser(tab=tab)
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

    async def test_steps_execute_in_order_before_the_final_wait_for(self, client: httpx.AsyncClient):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors={"#q", "#submit", ".final"})
        main._browser = _FakeBrowser(tab=tab)
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

    async def test_click_step_selector_never_found_returns_422_nav_step_failed(self, client: httpx.AsyncClient):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors=set())
        main._browser = _FakeBrowser(tab=tab)
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

    async def test_a_failed_nav_step_still_closes_the_tab(self, client: httpx.AsyncClient):
        """A nav step failure must not leak the tab -- the same discipline
        driver_crash/navigation_timeout already need, extended to this new
        failure mode (tab.close() moved into the shared finally block)."""
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors=set())
        main._browser = _FakeBrowser(tab=tab)
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

    async def test_a_failing_step_aborts_before_the_final_settle_wait(self, client: httpx.AsyncClient):
        tab = _FakeTab(html="<html></html>", url="https://example.gov", findable_selectors=set())
        main._browser = _FakeBrowser(tab=tab)
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

    async def test_a_transient_stale_node_error_is_retried_and_succeeds(self, client: httpx.AsyncClient):
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
        main._browser = _FakeBrowser(tab=tab)
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

    async def test_stale_node_error_exhausting_every_retry_returns_422(self, client: httpx.AsyncClient):
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            findable_selectors={"#search"},
            select_protocol_exceptions=99,
        )
        main._browser = _FakeBrowser(tab=tab)
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

    async def test_final_wait_for_also_retries_the_same_stale_node_race(self, client: httpx.AsyncClient):
        """Not nav_steps-specific -- the pre-existing wait_for settle-wait
        call hit the identical race live, with zero nav_steps involved."""
        tab = _FakeTab(
            html="<html></html>",
            url="https://example.gov",
            findable_selectors={"table"},
            select_protocol_exceptions=2,
        )
        main._browser = _FakeBrowser(tab=tab)
        async with client:
            r = await client.post(
                "/v1/render", json={"url": "https://example.gov", "timeout": 5.0, "wait_for": "table"}
            )
        assert r.status_code == 200
        assert tab.select_calls == ["table", "table", "table"]

    async def test_unsupported_action_returns_422_nav_step_failed(self, client: httpx.AsyncClient):
        tab = _FakeTab(html="<html></html>", url="https://example.gov")
        main._browser = _FakeBrowser(tab=tab)
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
    complete -- or fail open -- before /healthz reports "ok"."""

    async def test_succeeds_on_first_attempt_marks_ready(self, monkeypatch):
        tab = _FakeTab(html="<html></html>", url="https://example.com/")
        browser = _FakeBrowser(tab=tab)
        main._browser = browser
        monkeypatch.setattr(main, "_WARMUP_RETRY_DELAY_SECONDS", 0.0)

        await main._warm_up()

        assert main._ready is True
        assert browser.get_calls == 1
        assert tab.closed is True  # warm-up renders through the real _render path

    async def test_retries_then_succeeds(self, monkeypatch):
        tab = _FakeTab(html="<html></html>", url="https://example.com/")
        browser = _FakeBrowser(tab=tab, fail_times=2)  # fails twice, succeeds on the 3rd
        main._browser = browser
        monkeypatch.setattr(main, "_WARMUP_RETRY_DELAY_SECONDS", 0.0)

        await main._warm_up()

        assert main._ready is True
        assert browser.get_calls == 3

    async def test_fails_open_after_exhausting_retries(self, monkeypatch):
        """A warm-up that never succeeds must not block startup forever --
        marks ready anyway, logged loudly (the real first request would hit
        the same failure mode this mitigation is tolerant of, not a new one)."""
        browser = _FakeBrowser(raise_exc=RuntimeError("cold-start failure"))
        main._browser = browser
        monkeypatch.setattr(main, "_WARMUP_ATTEMPTS", 3)
        monkeypatch.setattr(main, "_WARMUP_RETRY_DELAY_SECONDS", 0.0)

        await main._warm_up()

        assert main._ready is True
        assert browser.get_calls == 3
