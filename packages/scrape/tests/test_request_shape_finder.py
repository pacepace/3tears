"""Unit tests for threetears.scrape.request_shape_finder -- the "how" sibling
to test_page_finder.py's "what."

All tests use a fake driver (no real network calls, no real browser) --
mirrors test_driver_network_capture.py's own ``_FakeInnerDriver`` convention.
The real, live proof against Google Trends (the case that motivated this
module) lives in faidh's own repo, not here -- this module carries zero
target-specific logic by design (see its own docstring's "gaming test"), so
its unit coverage is deliberately generic: one synthetic authenticated-API
target (Google-shaped, XSSI-prefixed) plus one DIFFERENT, unrelated synthetic
target (a plain JSON API with no prefix at all) proving this genuinely
generalizes, not just "works for the one real case it was built against."
"""

from __future__ import annotations

import json

from threetears.scrape.driver import NavStep, NetworkCall, RenderedPage
from threetears.scrape.request_shape_finder import (
    CapturedRequestShape,
    RequestShapeResult,
    _parse_body_shape,
    capture_request_shape,
)


# parity-with: threetears.scrape.driver.ScrapeDriver
class _FakeDriver:
    """Mirrors test_driver_network_capture.py's own _FakeInnerDriver."""

    def __init__(self, network_calls: list[NetworkCall], status: int = 200, final_url: str | None = None) -> None:
        self._network_calls = network_calls
        self._status = status
        self._final_url = final_url
        self.render_calls: list[dict] = []

    @property
    def name(self) -> str:
        return "fake-driver"

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
        self.render_calls.append({"url": url, "capture_network": capture_network, "nav_steps": nav_steps})
        return RenderedPage(
            html="<html></html>",
            status=self._status,
            final_url=self._final_url or url,
            timing_ms=1.0,
            network_calls=self._network_calls,
        )


def _call(
    url: str, body: str, *, method: str = "GET", status: int = 200, content_type: str = "application/json"
) -> NetworkCall:
    return NetworkCall(url=url, method=method, status=status, content_type=content_type, body=body)


# ===========================================================================
# _parse_body_shape
# ===========================================================================


class TestParseBodyShape:
    def test_plain_json_object_parses(self) -> None:
        assert _parse_body_shape('{"a": 1}') == {"a": 1}

    def test_plain_json_array_parses(self) -> None:
        assert _parse_body_shape("[1, 2, 3]") == [1, 2, 3]

    def test_xssi_prefixed_json_parses(self) -> None:
        """A generic anti-hijacking-prefix convention, not a Google-specific one --
        this test's prefix is deliberately a different, made-up string from any
        real vendor's, proving the skip-to-first-brace logic doesn't hardcode
        Google's own )]}' string anywhere."""
        assert _parse_body_shape(')]}\'\nSOME_OTHER_PREFIX{"a": 1}') == {"a": 1}

    def test_non_json_body_returns_none(self) -> None:
        assert _parse_body_shape("<html>not json</html>") is None

    def test_empty_body_returns_none(self) -> None:
        assert _parse_body_shape("") is None

    def test_malformed_json_after_brace_returns_none(self) -> None:
        assert _parse_body_shape("{not: valid, json}") is None


# ===========================================================================
# capture_request_shape
# ===========================================================================


class TestCaptureRequestShape:
    async def test_captures_every_real_call_with_parsed_body(self) -> None:
        driver = _FakeDriver(
            [
                _call("https://example.com/api/data", '{"records": [1, 2]}'),
                _call("https://example.com/api/telemetry", '{"beacon": true}'),
            ]
        )
        result = await capture_request_shape("https://example.com/page", driver=driver)
        assert isinstance(result, RequestShapeResult)
        assert result.page_status == 200
        assert len(result.calls) == 2
        assert result.calls[0] == CapturedRequestShape(
            url="https://example.com/api/data",
            method="GET",
            status=200,
            content_type="application/json",
            body='{"records": [1, 2]}',
            body_shape={"records": [1, 2]},
        )

    async def test_empty_network_calls_returns_empty_tuple(self) -> None:
        driver = _FakeDriver([])
        result = await capture_request_shape("https://example.com/page", driver=driver)
        assert result.calls == ()

    async def test_passes_capture_network_true_to_the_driver(self) -> None:
        driver = _FakeDriver([])
        await capture_request_shape("https://example.com/page", driver=driver)
        assert driver.render_calls[0]["capture_network"] is True

    async def test_appends_a_settle_wait_nav_step(self) -> None:
        driver = _FakeDriver([])
        await capture_request_shape("https://example.com/page", driver=driver, settle_wait_ms=5000)
        nav_steps = driver.render_calls[0]["nav_steps"]
        assert nav_steps[-1] == NavStep(action="wait_ms", ms=5000)

    async def test_caller_supplied_nav_steps_run_before_the_settle_wait(self) -> None:
        driver = _FakeDriver([])
        click_step = NavStep(action="click", selector="#accept")
        await capture_request_shape("https://example.com/page", driver=driver, nav_steps=[click_step])
        nav_steps = driver.render_calls[0]["nav_steps"]
        assert nav_steps[0] == click_step
        assert nav_steps[-1].action == "wait_ms"

    async def test_does_not_pick_a_winner_among_multiple_calls(self) -> None:
        """The core design contract: no heuristic picks "the real" call --
        every captured call comes back, in capture order, for the caller to
        judge. A silent "largest wins" pick would have been WRONG for this
        module's own real motivating case (Google Trends' real data call
        was smaller than incidental picker-prefetch calls captured alongside
        it) -- this test pins that this module never does that."""
        driver = _FakeDriver(
            [
                _call("https://example.com/api/huge-unrelated-list", json.dumps({"items": list(range(500))})),
                _call("https://example.com/api/actual-data", '{"widgets": [{"token": "abc"}]}'),
            ]
        )
        result = await capture_request_shape("https://example.com/page", driver=driver)
        assert len(result.calls) == 2
        assert result.calls[0].url == "https://example.com/api/huge-unrelated-list"
        assert result.calls[1].url == "https://example.com/api/actual-data"

    async def test_non_json_call_still_returned_with_none_body_shape(self) -> None:
        driver = _FakeDriver([_call("https://example.com/api/frag", "<div>fragment</div>")])
        result = await capture_request_shape("https://example.com/page", driver=driver)
        assert len(result.calls) == 1
        assert result.calls[0].body_shape is None
        assert result.calls[0].body == "<div>fragment</div>"  # real bytes preserved regardless


# ===========================================================================
# Gaming test: does this generalize to a DIFFERENT, unrelated target class,
# or does it only work for Google-shaped (XSSI-prefixed) responses?
# ===========================================================================


class TestGamingCheckDifferentTargetClass:
    """A completely different, synthetic authenticated-API target -- no XSSI
    prefix, a different method, a different body shape, nothing resembling
    Google Trends -- proving this module's real value is generic capture, not
    a Google-specific fix wearing a general-sounding name."""

    async def test_plain_post_json_api_with_no_hijack_prefix(self) -> None:
        driver = _FakeDriver(
            [
                _call(
                    "https://some-other-vendor.example/internal/v2/query",
                    json.dumps({"resultRows": [{"id": 1, "label": "x"}], "cursor": "abc123"}),
                    method="POST",
                )
            ]
        )
        result = await capture_request_shape("https://some-other-vendor.example/dashboard", driver=driver)
        assert len(result.calls) == 1
        call = result.calls[0]
        assert call.method == "POST"
        assert call.body_shape == {"resultRows": [{"id": 1, "label": "x"}], "cursor": "abc123"}
