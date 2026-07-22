"""Unit tests for NetworkCaptureDriver and its internal helpers.

All tests use a fake *inner* driver (no real network calls, no real browser).
The real, live proof against Oklahoma's genuine Salesforce Aura page lives in
tests/e2e/test_warn_act_eval_loop_live.py (target_id="warn_act_ok").

**Not added to tests/scrape/test_driver_contract.py's shared ``_BACKENDS``
list, on purpose:** same reason as ApiDriver/DocumentDriver -- this driver
transforms a captured network call's JSON body into synthetic HTML rather
than passing already-HTML source through verbatim, and it requires a real
*inner* driver with actual network_calls to have anything to find, which the
contract's own bare fixture doesn't provide.
"""

from __future__ import annotations

import json

import pytest

from threetears.scrape.driver import NavStep, NetworkCall, RenderedPage
from threetears.scrape.drivers.network_capture import (
    NetworkCaptureDriver,
    NetworkCaptureDriverError,
    _find_largest_record_list,
    _records_to_html,
)


# parity-with: threetears.scrape.driver.ScrapeDriver
class _FakeInnerDriver:
    def __init__(self, network_calls: list[NetworkCall], html: str = "<html><body>shell</body></html>"):
        self._network_calls = network_calls
        self._html = html
        self.render_calls: list[str] = []
        self.capture_network_calls: list[bool] = []

    @property
    def name(self) -> str:
        return "fake-inner"

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
    ) -> RenderedPage:
        self.render_calls.append(url)
        self.capture_network_calls.append(capture_network)
        return RenderedPage(
            html=self._html,
            status=200,
            final_url=url,
            timing_ms=1.0,
            network_calls=self._network_calls,
        )


def _call(body: dict, url: str = "https://example.gov/api", content_type: str = "application/json") -> NetworkCall:
    return NetworkCall(url=url, method="POST", status=200, content_type=content_type, body=json.dumps(body))


# ===========================================================================
# _find_largest_record_list
# ===========================================================================


class TestFindLargestRecordList:
    def test_top_level_list_of_dicts(self):
        data = {"Results": [{"a": 1}, {"a": 2}]}
        assert _find_largest_record_list(data) == [{"a": 1}, {"a": 2}]

    def test_deeply_nested_list_is_found(self):
        data = {"actions": [{"returnValue": {"returnValue": [{"a": 1}, {"a": 2}, {"a": 3}]}}]}
        assert _find_largest_record_list(data) == [{"a": 1}, {"a": 2}, {"a": 3}]

    def test_picks_the_largest_among_multiple_candidate_lists(self):
        # A small nav-menu-shaped dict-list alongside the real, much larger data table --
        # exactly Oklahoma's own Aura response shape (several small decoy lists, one real one).
        data = {
            "menu": [{"LinkName": "About"}, {"LinkName": "Contact"}],
            "actions": [{"returnValue": {"returnValue": [{"employer": f"Co{i}"} for i in range(50)]}}],
        }
        result = _find_largest_record_list(data)
        assert result is not None
        assert len(result) == 50

    def test_list_of_scalars_is_not_a_record_list(self):
        data = {"names": ["Employ Oklahoma", "Policies", "Partner Agencies", "Contact Us"]}
        assert _find_largest_record_list(data) is None

    def test_list_below_minimum_size_is_ignored(self):
        data = {"tiny": [{"a": 1}]}
        assert _find_largest_record_list(data) is None

    def test_no_list_anywhere_returns_none(self):
        assert _find_largest_record_list({"a": {"b": {"c": 1}}}) is None

    def test_mixed_list_with_a_non_dict_item_is_not_a_record_list(self):
        data = {"mixed": [{"a": 1}, "not a dict", {"a": 2}]}
        assert _find_largest_record_list(data) is None


# ===========================================================================
# _records_to_html
# ===========================================================================


class TestRecordsToHtml:
    def test_builds_a_real_table_with_union_of_keys_as_columns(self):
        records = [{"employer": "Acme", "county": "Oakland"}, {"employer": "Widgets"}]
        html = _records_to_html(records)
        assert "<th>employer</th><th>county</th>" in html
        assert "<td>Acme</td><td>Oakland</td>" in html
        # missing key on the second record renders an empty cell, not a dropped column
        assert "<td>Widgets</td><td></td>" in html

    def test_values_are_html_escaped(self):
        records = [{"employer": "Macy's & <Co>"}]
        html = _records_to_html(records)
        assert "Macy&#x27;s &amp; &lt;Co&gt;" in html

    def test_empty_records_list_produces_an_empty_table(self):
        html = _records_to_html([])
        assert html == "<html><body><table><tr></tr></table></body></html>"


# ===========================================================================
# NetworkCaptureDriver
# ===========================================================================


class TestNetworkCaptureDriver:
    def test_name(self):
        driver = NetworkCaptureDriver(_FakeInnerDriver([]))
        assert driver.name == "network_capture"

    async def test_render_builds_synthetic_html_from_the_largest_captured_record_list(self):
        calls = [
            _call({"menu": [{"LinkName": "About"}, {"LinkName": "Contact"}]}),
            _call({"actions": [{"returnValue": {"returnValue": [{"employer": f"Co{i}"} for i in range(10)]}}]}),
        ]
        inner = _FakeInnerDriver(calls)
        driver = NetworkCaptureDriver(inner)

        page = await driver.render("https://example.gov/warn")

        assert isinstance(page, RenderedPage)
        assert page.status == 200
        assert "<td>Co0</td>" in page.html
        assert "LinkName" not in page.html  # the smaller decoy list must not win

    async def test_render_forces_capture_network_on_the_inner_driver(self):
        inner = _FakeInnerDriver([_call({"data": [{"a": 1}, {"a": 2}]})])
        driver = NetworkCaptureDriver(inner)

        await driver.render("https://example.gov/warn", capture_network=False)

        assert inner.capture_network_calls == [True]

    async def test_render_forwards_wait_for_and_nav_steps_to_the_inner_driver(self):
        inner = _FakeInnerDriver([_call({"data": [{"a": 1}, {"a": 2}]})])
        driver = NetworkCaptureDriver(inner)
        steps = [NavStep(action="wait_ms", ms=5000)]

        await driver.render("https://example.gov/warn", wait_for="body", nav_steps=steps)

        assert inner.render_calls == ["https://example.gov/warn"]

    async def test_render_raises_when_no_call_has_a_usable_record_list(self):
        # a single-item list falls below _MIN_RECORDS -- not a usable table
        inner = _FakeInnerDriver([_call({"menu": [{"LinkName": "About"}]})])
        driver = NetworkCaptureDriver(inner)

        with pytest.raises(NetworkCaptureDriverError) as exc_info:
            await driver.render("https://example.gov/warn")

        assert exc_info.value.code == "no_record_list_found"

    async def test_render_raises_when_no_network_calls_were_captured_at_all(self):
        inner = _FakeInnerDriver([])
        driver = NetworkCaptureDriver(inner)

        with pytest.raises(NetworkCaptureDriverError):
            await driver.render("https://example.gov/warn")

    async def test_render_skips_non_json_call_bodies_without_crashing(self):
        calls = [
            NetworkCall(
                url="https://example.gov/x",
                method="GET",
                status=200,
                content_type="text/html",
                body="<html>not json</html>",
            ),
            _call({"data": [{"a": 1}, {"a": 2}]}),
        ]
        inner = _FakeInnerDriver(calls)
        driver = NetworkCaptureDriver(inner)

        page = await driver.render("https://example.gov/warn")

        assert "<td>1</td>" in page.html
