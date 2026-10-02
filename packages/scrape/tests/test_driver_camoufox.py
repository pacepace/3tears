"""Unit tests for CamoufoxDriver, the second ScrapeDriver backend.

All tests are fully mocked/in-memory -- no real browser launch, no camoufox
binary download, and no live-browser suite ships with this package (that
needs a real Camoufox binary on the machine running it). The generic,
backend-agnostic ScrapeDriver contract (shared with NodriverSidecarDriver)
lives in test_driver_contract.py, not here.
"""

from __future__ import annotations

import pytest
from packages.scrape.tests._camoufox_fakes import (
    FakeCamoufoxBrowser,
    FakeCamoufoxNetworkResponse,
    FakeCamoufoxPage,
    FakeCamoufoxResponse,
)
from packages.scrape.tests._driver_log_helpers import driver_warnings
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from threetears.scrape.driver import NavStep, RenderedPage
from threetears.scrape.drivers.camoufox import CamoufoxDriver, CamoufoxDriverError


class TestCamoufoxDriverName:
    def test_name(self):
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(FakeCamoufoxPage()))
        assert driver.name == "camoufox"


class TestCamoufoxDriverRender:
    async def test_render_success_returns_rendered_page(self):
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200), html="<html>real</html>", url="https://example.gov/page"
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov/page")

        assert isinstance(result, RenderedPage)
        assert result.html == "<html>real</html>"
        assert result.status == 200
        assert result.final_url == "https://example.gov/page"
        assert result.timing_ms >= 0
        assert page.closed is True  # new tab closed after use, never reused

    async def test_a_dropped_session_state_is_announced_rather_than_silent(self, caplog):
        """The whole value of the warning is that it is heard, so the assertion is on the log.

        This driver cannot apply a human's exported session and renders unauthenticated. In
        silence, a caller gets a successful render back and learns nothing until extraction
        fails on a login wall and the target is escalated to a person who already solved it.
        Without this assertion the warning was executed by the contract suite and checked by
        nothing, so deleting it left the suite green.
        """
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        with caplog.at_level("WARNING", logger="threetears.scrape.drivers.camoufox"):
            await driver.render("https://example.gov", session_state={"cookies": [{"name": "s"}]})

        # Through the shared helper like every other driver suite, so this asserts the record
        # came from camoufox's own logger rather than from whatever happened to be captured.
        mine = [r for r in driver_warnings(caplog, "camoufox") if "cannot apply it" in r.getMessage()]
        assert mine, (
            f"a dropped session state was not announced; saw {[(r.name, r.getMessage()) for r in caplog.records]}"
        )

    async def test_no_session_state_says_nothing(self, caplog):
        """A warning on every ordinary render would be noise that trains the reader to ignore it."""
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        with caplog.at_level("WARNING", logger="threetears.scrape.drivers.camoufox"):
            await driver.render("https://example.gov")

        assert driver_warnings(caplog, "camoufox") == []

    async def test_render_converts_seconds_to_milliseconds(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", timeout=9.5)

        assert page.goto_calls[0]["timeout"] == 9500.0
        assert page.goto_calls[0]["wait_until"] == "load"

    async def test_render_waits_for_selector_when_given(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", timeout=5.0, wait_for=".content")

        assert page.wait_for_calls == [{"selector": ".content", "timeout": 5000.0}]

    async def test_render_skips_wait_for_selector_when_omitted(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov")

        assert page.wait_for_calls == []

    async def test_render_new_page_per_call_never_reused(self):
        """The sidecar backend's own hard-won lesson: never reuse a tab across requests."""
        pages = [FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200)) for _ in range(2)]
        browser = FakeCamoufoxBrowser(pages)
        driver = CamoufoxDriver(browser=browser)

        await driver.render("https://example.gov/one")
        await driver.render("https://example.gov/two")

        assert browser.new_page_calls == 2
        assert pages[0].closed is True
        assert pages[1].closed is True

    async def test_render_raises_on_navigation_timeout(self):
        page = FakeCamoufoxPage(goto_exc=PlaywrightTimeoutError("navigation timed out"))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render("https://example.gov")

        assert exc_info.value.code == "navigation_timeout"
        assert page.closed is True  # still closed even on failure

    async def test_render_raises_on_navigation_failure(self):
        page = FakeCamoufoxPage(goto_exc=PlaywrightError("navigation crashed"))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render("https://example.gov")

        assert exc_info.value.code == "navigation_failed"
        assert page.closed is True

    async def test_render_raises_on_wait_for_timeout(self):
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200), wait_for_exc=PlaywrightTimeoutError("selector never appeared")
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render("https://example.gov", wait_for=".missing")

        assert exc_info.value.code == "wait_for_timeout"
        assert page.closed is True


class TestCamoufoxDriverNetworkCapture:
    async def test_capture_network_false_by_default_returns_no_calls(self):
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200),
            network_responses=[FakeCamoufoxNetworkResponse(url="https://example.gov/api/notices")],
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov")

        assert result.network_calls == []  # handler never registered when capture_network=False

    async def test_captures_a_real_json_xhr_response(self):
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200),
            network_responses=[
                FakeCamoufoxNetworkResponse(
                    url="https://example.gov/api/notices",
                    resource_type="xhr",
                    body='{"notices": [1, 2]}',
                )
            ],
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert len(result.network_calls) == 1
        call = result.network_calls[0]
        assert call.url == "https://example.gov/api/notices"
        assert call.method == "GET"
        assert call.status == 200
        assert call.content_type == "application/json"
        assert call.body == '{"notices": [1, 2]}'

    async def test_captures_a_post_requests_payload(self):
        """A POST-read API's payload is its query -- without it the call cannot be replayed."""
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200),
            network_responses=[
                FakeCamoufoxNetworkResponse(
                    url="https://api.example.gov/api/Grids/GetData",
                    resource_type="xhr",
                    body='{"data": {"items": [{"id": 1}]}}',
                    method="POST",
                    post_data='{"pageNumber": 1, "pageSize": 1000}',
                )
            ],
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        call = (await driver.render("https://portal.example.gov", capture_network=True)).network_calls[0]

        assert call.method == "POST"
        assert call.request_body == '{"pageNumber": 1, "pageSize": 1000}'

    async def test_a_get_reports_no_request_payload(self):
        """None, not "" -- "had no body" and "had an empty body" are different facts."""
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200),
            network_responses=[FakeCamoufoxNetworkResponse(url="https://example.gov/api/rows", body='{"rows": []}')],
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        call = (await driver.render("https://example.gov", capture_network=True)).network_calls[0]

        assert call.request_body is None

    async def test_captures_fetch_resource_type_too(self):
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200),
            network_responses=[FakeCamoufoxNetworkResponse(url="https://example.gov/api/data", resource_type="fetch")],
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert len(result.network_calls) == 1

    async def test_non_api_resource_types_are_not_captured(self):
        """Images/scripts/stylesheets are never a "backend API" signal."""
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200),
            network_responses=[
                FakeCamoufoxNetworkResponse(url="https://example.gov/style.css", resource_type="stylesheet"),
                FakeCamoufoxNetworkResponse(url="https://example.gov/logo.png", resource_type="image"),
                FakeCamoufoxNetworkResponse(url="https://example.gov/app.js", resource_type="script"),
            ],
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert result.network_calls == []

    async def test_non_json_bodies_are_not_captured(self):
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200),
            network_responses=[
                FakeCamoufoxNetworkResponse(url="https://example.gov/api/html-fragment", body="<div>not json</div>"),
            ],
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert result.network_calls == []

    async def test_a_failed_body_fetch_does_not_drop_other_captured_calls(self):
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200),
            network_responses=[
                FakeCamoufoxNetworkResponse(url="https://example.gov/api/broken", text_exc=PlaywrightError("gone")),
                FakeCamoufoxNetworkResponse(url="https://example.gov/api/good", body='{"ok": true}'),
            ],
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert len(result.network_calls) == 1
        assert result.network_calls[0].url == "https://example.gov/api/good"

    async def test_a_non_utf8_body_is_recovered_rather_than_aborting_the_render(self):
        # Response.text() decodes as UTF-8 unconditionally, and UnicodeDecodeError is a ValueError,
        # not a PlaywrightError -- so a cp1252 body escaped the per-response guard and killed the
        # whole render, discarding every other captured call on the page. Observed live against two
        # state disclosure portals whose JSON carries an 0xA9 copyright sign.
        cp1252_json = '{"agency": "Dept \xa9 2025", "ok": true}'.encode("cp1252")
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200),
            network_responses=[
                FakeCamoufoxNetworkResponse(url="https://example.gov/api/latin", body=cp1252_json),
                FakeCamoufoxNetworkResponse(url="https://example.gov/api/good", body='{"ok": true}'),
            ],
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        # Both calls survive, and the mis-encoded one is decoded rather than dropped.
        assert [c.url for c in result.network_calls] == [
            "https://example.gov/api/latin",
            "https://example.gov/api/good",
        ]
        assert "Dept \xa9 2025" in result.network_calls[0].body

    async def test_an_undecodable_body_is_skipped_without_dropping_other_calls(self):
        # The floor beneath the recovery above: a body no candidate encoding reads must be skipped
        # like any other unusable response, never crash the render and never be silently mojibaked
        # into plausible-looking but corrupted values.
        undecodable = b'{"x": \x81\x8d\x8f}'  # unmapped in cp1252, invalid in UTF-8
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200),
            network_responses=[
                FakeCamoufoxNetworkResponse(url="https://example.gov/api/broken", body=undecodable),
                FakeCamoufoxNetworkResponse(url="https://example.gov/api/good", body='{"ok": true}'),
            ],
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert [c.url for c in result.network_calls] == ["https://example.gov/api/good"]

    async def test_a_failed_headers_fetch_does_not_drop_other_captured_calls(self):
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200),
            network_responses=[
                FakeCamoufoxNetworkResponse(
                    url="https://example.gov/api/broken", body='{"ok": true}', headers_exc=PlaywrightError("gone")
                ),
                FakeCamoufoxNetworkResponse(url="https://example.gov/api/good", body='{"ok": true}'),
            ],
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert len(result.network_calls) == 1
        assert result.network_calls[0].url == "https://example.gov/api/good"

    async def test_capture_bounded_by_max_network_calls(self):
        responses = [FakeCamoufoxNetworkResponse(url=f"https://example.gov/api/{i}") for i in range(8)]
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200), network_responses=responses)
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page), max_network_calls=3)

        result = await driver.render("https://example.gov", capture_network=True)

        assert [call.url for call in result.network_calls] == [f"https://example.gov/api/{i}" for i in range(3)]

    async def test_the_default_bound_matches_the_nodriver_sidecar(self):
        """Thirty, the sidecar's own bound: one more response than that is dropped."""
        responses = [FakeCamoufoxNetworkResponse(url=f"https://example.gov/api/{i}") for i in range(31)]
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200), network_responses=responses)
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov", capture_network=True)

        assert len(result.network_calls) == 30


class TestCamoufoxDriverNavSteps:
    """Multi-step navigation capability (2026-07-14)."""

    async def test_no_nav_steps_executes_nothing(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov")

        assert page.click_calls == []
        assert page.fill_calls == []
        assert page.wait_for_timeout_calls == []
        assert page.scroll_into_view_calls == []

    async def test_click_step_clicks_the_selector(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="click", selector="#search")])

        assert page.click_calls == [{"selector": "#search", "timeout": 30.0 * 1000}]

    async def test_fill_step_fills_the_selector_with_value(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="fill", selector="#q", value="Maine")])

        assert page.fill_calls == [{"selector": "#q", "value": "Maine", "timeout": 30.0 * 1000}]

    async def test_wait_for_step_waits_for_the_selector(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="wait_for", selector=".results")])

        assert page.wait_for_calls == [{"selector": ".results", "timeout": 30.0 * 1000}]

    async def test_scroll_into_view_step_scrolls_the_selector(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="scroll_into_view", selector="#chart")])

        assert page.scroll_into_view_calls == [{"selector": "#chart", "timeout": 30.0 * 1000}]

    async def test_scroll_page_step_scrolls_by_percent_of_viewport_height(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200), viewport_size={"width": 1920, "height": 1000})
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="scroll_page", value="50")])

        assert page.wheel_calls == [{"delta_x": 0, "delta_y": 500.0}]

    async def test_scroll_page_step_uses_the_default_amount_when_value_omitted(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200), viewport_size={"width": 1920, "height": 1000})
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="scroll_page")])

        assert page.wheel_calls == [{"delta_x": 0, "delta_y": 250.0}]

    async def test_scroll_page_step_non_int_value_raises_nav_step_failed(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render("https://example.gov", nav_steps=[NavStep(action="scroll_page", value="not-a-number")])

        assert exc_info.value.code == "nav_step_failed"
        assert page.wheel_calls == []

    async def test_evaluate_step_runs_the_expression_and_records_the_result(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200), evaluate_returns=[{"foo": "bar"}])
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render(
            "https://example.gov", nav_steps=[NavStep(action="evaluate", value="({foo: 'bar'})")]
        )

        assert page.evaluate_calls == ["({foo: 'bar'})"]
        assert result.eval_results == [{"foo": "bar"}]

    async def test_evaluate_step_records_each_step_result_in_order(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200), evaluate_returns=[1, 2])
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render(
            "https://example.gov",
            nav_steps=[NavStep(action="evaluate", value="1"), NavStep(action="evaluate", value="2")],
        )

        assert result.eval_results == [1, 2]

    async def test_no_evaluate_steps_leaves_eval_results_empty(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        result = await driver.render("https://example.gov")

        assert result.eval_results == []

    async def test_evaluate_step_js_exception_raises_nav_step_failed(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200), evaluate_exc=PlaywrightError("boom"))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render("https://example.gov", nav_steps=[NavStep(action="evaluate", value="throw 1")])

        assert exc_info.value.code == "nav_step_failed"

    async def test_wait_ms_step_sleeps_for_the_given_duration(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        await driver.render("https://example.gov", nav_steps=[NavStep(action="wait_ms", ms=500)])

        assert page.wait_for_timeout_calls == [500]

    async def test_steps_execute_in_order_before_the_final_wait_for(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

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
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200), click_exc=PlaywrightTimeoutError("gone"))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render("https://example.gov", nav_steps=[NavStep(action="click", selector="#missing")])

        assert exc_info.value.code == "nav_step_failed"

    async def test_fill_step_selector_never_appearing_raises_nav_step_failed(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200), fill_exc=PlaywrightTimeoutError("gone"))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render(
                "https://example.gov", nav_steps=[NavStep(action="fill", selector="#missing", value="x")]
            )

        assert exc_info.value.code == "nav_step_failed"

    async def test_wait_for_step_selector_never_appearing_raises_nav_step_failed(self):
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200), wait_for_exc=PlaywrightTimeoutError("gone"))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render("https://example.gov", nav_steps=[NavStep(action="wait_for", selector="#missing")])

        assert exc_info.value.code == "nav_step_failed"

    async def test_scroll_into_view_step_selector_never_appearing_raises_nav_step_failed(self):
        page = FakeCamoufoxPage(
            goto_result=FakeCamoufoxResponse(200), scroll_into_view_exc=PlaywrightTimeoutError("gone")
        )
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        with pytest.raises(CamoufoxDriverError) as exc_info:
            await driver.render(
                "https://example.gov", nav_steps=[NavStep(action="scroll_into_view", selector="#missing")]
            )

        assert exc_info.value.code == "nav_step_failed"

    async def test_a_failing_step_aborts_before_the_final_settle_wait(self):
        """The final wait_for/settle-wait must not run when an earlier nav
        step already failed -- the page was never successfully driven to
        where that wait_for's selector would even make sense."""
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200), click_exc=PlaywrightTimeoutError("gone"))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

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
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

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
                self._browser = FakeCamoufoxBrowser(FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200)))

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
        page = FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200))
        driver = CamoufoxDriver(browser=FakeCamoufoxBrowser(page))

        await driver.close()  # must not raise; injected browser's lifecycle isn't ours

    async def test_close_tears_down_owned_browser(self, monkeypatch):
        exited: list[bool] = []

        # parity-exempt: hand-rolled subset stub of camoufox's third-party AsyncCamoufox (only the async-context-manager surface CamoufoxDriver._ensure_browser calls)
        class _FakeAsyncCamoufox:
            def __init__(self, **kwargs):
                self._browser = FakeCamoufoxBrowser(FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200)))

            async def __aenter__(self):
                return self._browser

            async def __aexit__(self, *exc_info):
                exited.append(True)

        monkeypatch.setattr("camoufox.async_api.AsyncCamoufox", _FakeAsyncCamoufox)

        driver = CamoufoxDriver()
        await driver.render("https://example.gov")
        await driver.close()

        assert exited == [True]


async def _launch_options(driver: CamoufoxDriver, monkeypatch) -> dict:
    """Render once through *driver* and return the options its browser was launched with.

    Camoufox is replaced at its import site by a stub that records its constructor
    arguments, so this is what a real launch would have been handed.
    """
    launched: list[dict] = []

    # parity-exempt: hand-rolled subset stub of camoufox's third-party AsyncCamoufox (only the async-context-manager surface CamoufoxDriver._ensure_browser calls)
    class _RecordingAsyncCamoufox:
        def __init__(self, **kwargs):
            launched.append(kwargs)
            self._browser = FakeCamoufoxBrowser(FakeCamoufoxPage(goto_result=FakeCamoufoxResponse(200)))

        async def __aenter__(self):
            return self._browser

        async def __aexit__(self, *exc_info):
            return None

    monkeypatch.setattr("camoufox.async_api.AsyncCamoufox", _RecordingAsyncCamoufox)
    await driver.render("https://example.gov")
    await driver.close()
    assert len(launched) == 1
    return launched[0]


class TestTheExitReachesTheBrowserLaunch:
    """The half the driver contract cannot see: whether the launch actually carries the exit.

    `test_driver_contract.py` pins that an exit given to this driver comes back on the
    `RenderedPage`. That round trip passes against an INJECTED browser, so it says nothing
    about whether a real launch would have been proxied -- which is the whole of what this
    driver was missing. These render through a launch the driver performs itself and read
    the options that launch was given.
    """

    async def test_a_proxy_exit_becomes_a_playwright_proxy_option(self, monkeypatch) -> None:
        """Camoufox is Firefox via Playwright, so the exit is `proxy={"server": ...}`.

        :return: nothing
        :rtype: None
        """
        from threetears.core.egress import ProxyEgress

        driver = CamoufoxDriver(egress=ProxyEgress("tor", "socks5://127.0.0.1:9050"))

        assert await _launch_options(driver, monkeypatch) == {
            "headless": True,
            "proxy": {"server": "socks5://127.0.0.1:9050"},
        }

    async def test_no_exit_expresses_no_opinion(self, monkeypatch) -> None:
        """`None` must leave the launch alone rather than inventing a proxy key.

        :return: nothing
        :rtype: None
        """
        assert await _launch_options(CamoufoxDriver(), monkeypatch) == {"headless": True}

    async def test_a_direct_exit_does_not_become_a_proxy_server(self, monkeypatch) -> None:
        """`direct://` is Chromium's spelling and would be a bogus host to Firefox.

        Forwarding it as a Playwright `server` would make Playwright try to resolve
        `direct://` as a real proxy and fail every navigation -- an exit configured as
        "explicitly no proxy" would break the driver outright rather than send it direct.

        :return: nothing
        :rtype: None
        """
        from threetears.core.egress import DirectEgress

        driver = CamoufoxDriver(egress=DirectEgress())

        assert await _launch_options(driver, monkeypatch) == {"headless": True}

    async def test_the_direct_sentinel_is_the_one_egress_actually_returns(self, monkeypatch) -> None:
        """Pins the two sides together rather than against a literal typed twice.

        If `DirectEgress` ever changed its spelling, a hard-coded `"direct://"` in this
        driver would silently start forwarding it as a real proxy server. Asserted on the
        launch itself: `DirectEgress` must return a non-``None`` argument (so the launch is
        decided by the driver recognising that spelling, not by the no-argument branch),
        and the launch that results must still carry no proxy.

        :return: nothing
        :rtype: None
        """
        from threetears.core.egress import DirectEgress

        egress = DirectEgress()
        assert egress.browser_proxy_arg() is not None

        launch = await _launch_options(CamoufoxDriver(egress=egress), monkeypatch)

        assert "proxy" not in launch
