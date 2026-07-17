"""Unit tests for CamoufoxDriver (Chunk 6 -- second ScrapeDriver backend).

All tests are fully mocked/in-memory -- no real browser launch, no camoufox
binary download. The real, live browser proof lives in
tests/integration/test_scrape_camoufox_live.py. The generic,
backend-agnostic ScrapeDriver contract (shared with NodriverSidecarDriver)
lives in tests/scrape/test_driver_contract.py, not here.
"""

from __future__ import annotations

import pytest
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from threetears.scrape.driver import NavStep, RenderedPage
from threetears.scrape.drivers.camoufox import CamoufoxDriver, CamoufoxDriverError


# parity-exempt: hand-rolled subset stub of Playwright's third-party Locator (only scroll_into_view_if_needed, the only surface CamoufoxDriver calls)
class _FakeCamoufoxLocator:
    def __init__(self, selector: str, *, scroll_into_view_calls: list[dict], scroll_into_view_exc=None) -> None:
        self._selector = selector
        self._scroll_into_view_calls = scroll_into_view_calls
        self._scroll_into_view_exc = scroll_into_view_exc

    async def scroll_into_view_if_needed(self, *, timeout=None):
        self._scroll_into_view_calls.append({"selector": self._selector, "timeout": timeout})
        if self._scroll_into_view_exc is not None:
            raise self._scroll_into_view_exc


# parity-exempt: hand-rolled subset stub of Playwright's third-party Mouse (only wheel, the only surface CamoufoxDriver calls)
class _FakeCamoufoxMouse:
    def __init__(self, *, wheel_calls: list[dict]) -> None:
        self._wheel_calls = wheel_calls

    async def wheel(self, delta_x, delta_y):
        self._wheel_calls.append({"delta_x": delta_x, "delta_y": delta_y})


# parity-exempt: hand-rolled subset stub of Playwright's third-party Page (only goto/wait_for_selector/click/fill/wait_for_timeout/locator/mouse/viewport_size/content/url/close/on, the only surface CamoufoxDriver calls)
class _FakeCamoufoxPage:
    def __init__(
        self,
        *,
        goto_result=None,
        goto_exc=None,
        wait_for_exc=None,
        click_exc=None,
        fill_exc=None,
        scroll_into_view_exc=None,
        html="<html>ok</html>",
        url=None,
        network_responses=None,
        viewport_size=None,
    ):
        self._goto_result = goto_result
        self._goto_exc = goto_exc
        self._wait_for_exc = wait_for_exc
        self._click_exc = click_exc
        self._fill_exc = fill_exc
        self._scroll_into_view_exc = scroll_into_view_exc
        self._html = html
        self.url = url or "https://example.gov/final"
        self.viewport_size = viewport_size or {"width": 1920, "height": 1080}
        self.goto_calls: list[dict] = []
        self.wait_for_calls: list[dict] = []
        self.click_calls: list[dict] = []
        self.fill_calls: list[dict] = []
        self.wait_for_timeout_calls: list[int] = []
        self.scroll_into_view_calls: list[dict] = []
        self.wheel_calls: list[dict] = []
        self.mouse = _FakeCamoufoxMouse(wheel_calls=self.wheel_calls)
        self.closed = False
        # Simulates the responses Playwright would have fired via page.on("response", ...)
        # during navigation -- goto() replays these into the registered handler.
        self._network_responses = network_responses or []
        self._response_handler = None

    async def goto(self, url, *, timeout=None, wait_until=None):
        self.goto_calls.append({"url": url, "timeout": timeout, "wait_until": wait_until})
        if self._goto_exc is not None:
            raise self._goto_exc
        if self._response_handler is not None:
            for resp in self._network_responses:
                self._response_handler(resp)
        return self._goto_result

    async def wait_for_selector(self, selector, *, timeout=None):
        self.wait_for_calls.append({"selector": selector, "timeout": timeout})
        if self._wait_for_exc is not None:
            raise self._wait_for_exc

    async def click(self, selector, *, timeout=None):
        self.click_calls.append({"selector": selector, "timeout": timeout})
        if self._click_exc is not None:
            raise self._click_exc

    async def fill(self, selector, value, *, timeout=None):
        self.fill_calls.append({"selector": selector, "value": value, "timeout": timeout})
        if self._fill_exc is not None:
            raise self._fill_exc

    async def wait_for_timeout(self, ms):
        self.wait_for_timeout_calls.append(ms)

    def locator(self, selector):
        return _FakeCamoufoxLocator(
            selector, scroll_into_view_calls=self.scroll_into_view_calls, scroll_into_view_exc=self._scroll_into_view_exc
        )

    async def content(self):
        return self._html

    async def close(self):
        self.closed = True

    def on(self, event, handler):
        if event == "response":
            self._response_handler = handler


# parity-exempt: hand-rolled subset stub of Playwright's third-party Request (only .resource_type/.method, the only attributes CamoufoxDriver reads)
class _FakeCamoufoxRequest:
    def __init__(self, resource_type: str, method: str = "GET") -> None:
        self.resource_type = resource_type
        self.method = method


# parity-exempt: hand-rolled subset stub of Playwright's third-party Response used for network-capture (only .request/.status/.url/.text()/.all_headers(), the only surface CamoufoxDriver's capture_network path reads)
class _FakeCamoufoxNetworkResponse:
    def __init__(
        self,
        *,
        url: str,
        status: int = 200,
        resource_type: str = "xhr",
        body: str = "{}",
        content_type: str = "application/json",
        text_exc: Exception | None = None,
        headers_exc: Exception | None = None,
    ):
        self.url = url
        self.status = status
        self.request = _FakeCamoufoxRequest(resource_type)
        self._body = body
        self._content_type = content_type
        self._text_exc = text_exc
        self._headers_exc = headers_exc

    async def text(self):
        if self._text_exc is not None:
            raise self._text_exc
        return self._body

    async def all_headers(self):
        if self._headers_exc is not None:
            raise self._headers_exc
        return {"content-type": self._content_type}


# parity-exempt: hand-rolled subset stub of Playwright's third-party Response (only .status, the only attribute CamoufoxDriver reads)
class _FakeCamoufoxResponse:
    def __init__(self, status: int) -> None:
        self.status = status


# parity-exempt: hand-rolled subset stub of Playwright's third-party Browser (only new_page(), the only method CamoufoxDriver calls)
class _FakeCamoufoxBrowser:
    def __init__(self, page: _FakeCamoufoxPage | list[_FakeCamoufoxPage]) -> None:
        self._pages = page if isinstance(page, list) else [page]
        self.new_page_calls = 0

    async def new_page(self):
        # Repeats the last page if new_page() is called more times than pages
        # were supplied -- single-page callers never need to think about this.
        result = self._pages[min(self.new_page_calls, len(self._pages) - 1)]
        self.new_page_calls += 1
        return result


class TestCamoufoxDriverName:
    def test_name(self):
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(_FakeCamoufoxPage()))
        assert driver.name == "camoufox"


class TestCamoufoxDriverRender:
    async def test_render_success_returns_rendered_page(self):
        page = _FakeCamoufoxPage(
            goto_result=_FakeCamoufoxResponse(200), html="<html>real</html>", url="https://example.gov/page"
        )
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov/page")

        assert isinstance(result, RenderedPage)
        assert result.html == "<html>real</html>"
        assert result.status == 200
        assert result.final_url == "https://example.gov/page"
        assert result.timing_ms >= 0
        assert page.closed is True  # new tab closed after use, never reused

    async def test_render_converts_seconds_to_milliseconds(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", timeout=9.5)

        assert page.goto_calls[0]["timeout"] == 9500.0
        assert page.goto_calls[0]["wait_until"] == "load"

    async def test_render_waits_for_selector_when_given(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", timeout=5.0, wait_for=".content")

        assert page.wait_for_calls == [{"selector": ".content", "timeout": 5000.0}]

    async def test_render_skips_wait_for_selector_when_omitted(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov")

        assert page.wait_for_calls == []

    async def test_render_new_page_per_call_never_reused(self):
        """Chunk 1's own lesson: never reuse the same tab across requests."""
        pages = [_FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200)) for _ in range(2)]
        browser = _FakeCamoufoxBrowser(pages)
        driver = CamoufoxDriver(browser=browser)

        await driver.render("https://example.gov/one")
        await driver.render("https://example.gov/two")

        assert browser.new_page_calls == 2
        assert pages[0].closed is True
        assert pages[1].closed is True

    async def test_render_raises_on_navigation_timeout(self):
        page = _FakeCamoufoxPage(goto_exc=PlaywrightTimeoutError("navigation timed out"))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render("https://example.gov")

        assert exc_info.value.code == "navigation_timeout"
        assert page.closed is True  # still closed even on failure

    async def test_render_raises_on_navigation_failure(self):
        page = _FakeCamoufoxPage(goto_exc=PlaywrightError("navigation crashed"))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render("https://example.gov")

        assert exc_info.value.code == "navigation_failed"
        assert page.closed is True

    async def test_render_raises_on_wait_for_timeout(self):
        page = _FakeCamoufoxPage(
            goto_result=_FakeCamoufoxResponse(200), wait_for_exc=PlaywrightTimeoutError("selector never appeared")
        )
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render("https://example.gov", wait_for=".missing")

        assert exc_info.value.code == "wait_for_timeout"
        assert page.closed is True


class TestCamoufoxDriverNetworkCapture:
    async def test_capture_network_false_by_default_returns_no_calls(self):
        page = _FakeCamoufoxPage(
            goto_result=_FakeCamoufoxResponse(200),
            network_responses=[_FakeCamoufoxNetworkResponse(url="https://example.gov/api/notices")],
        )
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov")

        assert result.network_calls == []  # handler never registered when capture_network=False

    async def test_captures_a_real_json_xhr_response(self):
        page = _FakeCamoufoxPage(
            goto_result=_FakeCamoufoxResponse(200),
            network_responses=[
                _FakeCamoufoxNetworkResponse(
                    url="https://example.gov/api/notices",
                    resource_type="xhr",
                    body='{"notices": [1, 2]}',
                )
            ],
        )
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert len(result.network_calls) == 1
        call = result.network_calls[0]
        assert call.url == "https://example.gov/api/notices"
        assert call.method == "GET"
        assert call.status == 200
        assert call.content_type == "application/json"
        assert call.body == '{"notices": [1, 2]}'

    async def test_captures_fetch_resource_type_too(self):
        page = _FakeCamoufoxPage(
            goto_result=_FakeCamoufoxResponse(200),
            network_responses=[_FakeCamoufoxNetworkResponse(url="https://example.gov/api/data", resource_type="fetch")],
        )
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert len(result.network_calls) == 1

    async def test_non_api_resource_types_are_not_captured(self):
        """Images/scripts/stylesheets are never a "backend API" signal."""
        page = _FakeCamoufoxPage(
            goto_result=_FakeCamoufoxResponse(200),
            network_responses=[
                _FakeCamoufoxNetworkResponse(url="https://example.gov/style.css", resource_type="stylesheet"),
                _FakeCamoufoxNetworkResponse(url="https://example.gov/logo.png", resource_type="image"),
                _FakeCamoufoxNetworkResponse(url="https://example.gov/app.js", resource_type="script"),
            ],
        )
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert result.network_calls == []

    async def test_non_json_bodies_are_not_captured(self):
        page = _FakeCamoufoxPage(
            goto_result=_FakeCamoufoxResponse(200),
            network_responses=[
                _FakeCamoufoxNetworkResponse(url="https://example.gov/api/html-fragment", body="<div>not json</div>"),
            ],
        )
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert result.network_calls == []

    async def test_a_failed_body_fetch_does_not_drop_other_captured_calls(self):
        page = _FakeCamoufoxPage(
            goto_result=_FakeCamoufoxResponse(200),
            network_responses=[
                _FakeCamoufoxNetworkResponse(url="https://example.gov/api/broken", text_exc=PlaywrightError("gone")),
                _FakeCamoufoxNetworkResponse(url="https://example.gov/api/good", body='{"ok": true}'),
            ],
        )
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert len(result.network_calls) == 1
        assert result.network_calls[0].url == "https://example.gov/api/good"

    async def test_a_failed_headers_fetch_does_not_drop_other_captured_calls(self):
        page = _FakeCamoufoxPage(
            goto_result=_FakeCamoufoxResponse(200),
            network_responses=[
                _FakeCamoufoxNetworkResponse(
                    url="https://example.gov/api/broken", body='{"ok": true}', headers_exc=PlaywrightError("gone")
                ),
                _FakeCamoufoxNetworkResponse(url="https://example.gov/api/good", body='{"ok": true}'),
            ],
        )
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert len(result.network_calls) == 1
        assert result.network_calls[0].url == "https://example.gov/api/good"

    async def test_capture_bounded_by_max_network_calls(self):
        from threetears.scrape.drivers.camoufox import _MAX_NETWORK_CALLS

        responses = [
            _FakeCamoufoxNetworkResponse(url=f"https://example.gov/api/{i}") for i in range(_MAX_NETWORK_CALLS + 5)
        ]
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200), network_responses=responses)
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert len(result.network_calls) == _MAX_NETWORK_CALLS


class TestCamoufoxDriverNavSteps:
    """Multi-step navigation capability (2026-07-14)."""

    async def test_no_nav_steps_executes_nothing(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov")

        assert page.click_calls == []
        assert page.fill_calls == []
        assert page.wait_for_timeout_calls == []
        assert page.scroll_into_view_calls == []

    async def test_click_step_clicks_the_selector(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="click", selector="#search")])

        assert page.click_calls == [{"selector": "#search", "timeout": 30.0 * 1000}]

    async def test_fill_step_fills_the_selector_with_value(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="fill", selector="#q", value="Maine")])

        assert page.fill_calls == [{"selector": "#q", "value": "Maine", "timeout": 30.0 * 1000}]

    async def test_wait_for_step_waits_for_the_selector(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="wait_for", selector=".results")])

        assert page.wait_for_calls == [{"selector": ".results", "timeout": 30.0 * 1000}]

    async def test_scroll_into_view_step_scrolls_the_selector(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="scroll_into_view", selector="#chart")])

        assert page.scroll_into_view_calls == [{"selector": "#chart", "timeout": 30.0 * 1000}]

    async def test_scroll_page_step_scrolls_by_percent_of_viewport_height(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200), viewport_size={"width": 1920, "height": 1000})
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="scroll_page", value="50")])

        assert page.wheel_calls == [{"delta_x": 0, "delta_y": 500.0}]

    async def test_scroll_page_step_uses_the_default_amount_when_value_omitted(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200), viewport_size={"width": 1920, "height": 1000})
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="scroll_page")])

        assert page.wheel_calls == [{"delta_x": 0, "delta_y": 250.0}]

    async def test_scroll_page_step_non_int_value_raises_nav_step_failed(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render("https://example.gov", nav_steps=[NavStep(action="scroll_page", value="not-a-number")])

        assert exc_info.value.code == "nav_step_failed"
        assert page.wheel_calls == []

    async def test_wait_ms_step_sleeps_for_the_given_duration(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="wait_ms", ms=500)])

        assert page.wait_for_timeout_calls == [500]

    async def test_steps_execute_in_order_before_the_final_wait_for(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        await driver.render(
            "https://example.gov",
            wait_for=".final",
            nav_steps=[
                NavStep(action="fill", selector="#q", value="Maine"),
                NavStep(action="click", selector="#submit"),
            ],
        )

        assert page.fill_calls == [{"selector": "#q", "value": "Maine", "timeout": 30.0 * 1000}]
        assert page.click_calls == [{"selector": "#submit", "timeout": 30.0 * 1000}]
        # the final settle wait_for still runs, after every nav step
        assert page.wait_for_calls == [{"selector": ".final", "timeout": 30.0 * 1000}]

    async def test_click_step_selector_never_appearing_raises_nav_step_failed(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200), click_exc=PlaywrightTimeoutError("gone"))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render("https://example.gov", nav_steps=[NavStep(action="click", selector="#missing")])

        assert exc_info.value.code == "nav_step_failed"

    async def test_fill_step_selector_never_appearing_raises_nav_step_failed(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200), fill_exc=PlaywrightTimeoutError("gone"))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render(
                "https://example.gov", nav_steps=[NavStep(action="fill", selector="#missing", value="x")]
            )

        assert exc_info.value.code == "nav_step_failed"

    async def test_wait_for_step_selector_never_appearing_raises_nav_step_failed(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200), wait_for_exc=PlaywrightTimeoutError("gone"))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render("https://example.gov", nav_steps=[NavStep(action="wait_for", selector="#missing")])

        assert exc_info.value.code == "nav_step_failed"

    async def test_scroll_into_view_step_selector_never_appearing_raises_nav_step_failed(self):
        page = _FakeCamoufoxPage(
            goto_result=_FakeCamoufoxResponse(200), scroll_into_view_exc=PlaywrightTimeoutError("gone")
        )
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render(
                "https://example.gov", nav_steps=[NavStep(action="scroll_into_view", selector="#missing")]
            )

        assert exc_info.value.code == "nav_step_failed"

    async def test_a_failing_step_aborts_before_the_final_settle_wait(self):
        """The final wait_for/settle-wait must not run when an earlier nav
        step already failed -- the page was never successfully driven to
        where that wait_for's selector would even make sense."""
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200), click_exc=PlaywrightTimeoutError("gone"))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError):
            await driver.render(
                "https://example.gov", wait_for=".final", nav_steps=[NavStep(action="click", selector="#missing")]
            )

        assert page.wait_for_calls == []

    async def test_unsupported_action_raises_nav_step_failed(self):
        """NavStep.action's Literal type isn't runtime-enforced by the frozen
        dataclass -- an invalid value can still reach here (e.g. a typo'd
        action decoded from stored config); the driver must reject it
        loudly, not silently no-op or crash with an unrelated error."""
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render(
                "https://example.gov",
                nav_steps=[NavStep(action="scroll_to", selector="#x")],  # type: ignore[arg-type]
            )

        assert exc_info.value.code == "nav_step_failed"
        assert "scroll_to" in exc_info.value.message


class TestCamoufoxDriverLazyLaunch:
    async def test_browser_launched_once_and_reused_across_render_calls(self, monkeypatch):
        launched: list[dict] = []

        # parity-exempt: hand-rolled subset stub of camoufox's third-party AsyncCamoufox (only the async-context-manager surface CamoufoxDriver._ensure_browser calls)
        class _FakeAsyncCamoufox:
            def __init__(self, **kwargs):
                launched.append(kwargs)
                self._browser = _FakeCamoufoxBrowser(_FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200)))

            async def __aenter__(self):
                return self._browser

            async def __aexit__(self, *exc_info):
                return None

        monkeypatch.setattr("camoufox.async_api.AsyncCamoufox", _FakeAsyncCamoufox)

        driver = CamoufoxDriver(headless=True)
        await driver.render("https://example.gov/one")
        await driver.render("https://example.gov/two")

        assert len(launched) == 1  # launched once, reused for the second render()
        assert launched[0] == {"headless": True}

    async def test_close_is_a_noop_when_browser_was_injected(self):
        page = _FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=_FakeCamoufoxBrowser(page))

        await driver.close()  # must not raise; injected browser's lifecycle isn't ours

    async def test_close_tears_down_owned_browser(self, monkeypatch):
        exited: list[bool] = []

        # parity-exempt: hand-rolled subset stub of camoufox's third-party AsyncCamoufox (only the async-context-manager surface CamoufoxDriver._ensure_browser calls)
        class _FakeAsyncCamoufox:
            def __init__(self, **kwargs):
                self._browser = _FakeCamoufoxBrowser(_FakeCamoufoxPage(goto_result=_FakeCamoufoxResponse(200)))

            async def __aenter__(self):
                return self._browser

            async def __aexit__(self, *exc_info):
                exited.append(True)

        monkeypatch.setattr("camoufox.async_api.AsyncCamoufox", _FakeAsyncCamoufox)

        driver = CamoufoxDriver()
        await driver.render("https://example.gov")
        await driver.close()

        assert exited == [True]
