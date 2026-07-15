"""Unit tests for ApiDriver and _resolve_path.

All tests are fully mocked -- no real network calls (httpx.MockTransport
throughout). The real, live proof against Michigan's genuine Sitecore XA
search API lives in tests/e2e/test_warn_act_eval_loop_live.py
(target_id="warn_act_mi").

**Not added to tests/scrape/test_driver_contract.py's shared ``_BACKENDS``
list, on purpose:** same reason as DocumentDriver (see
tests/scrape/test_driver_document.py's own module docstring) -- this driver
transforms concatenated JSON fragments into synthetic HTML rather than
passing already-HTML source through verbatim, so the contract's exact-
content-equality assertion doesn't apply the same way. It also requires
``results_path``/``fragment_field`` on every call (the contract's own calls
omit them), which would fail every shared test immediately rather than
exercise anything backend-agnostic.
"""

from __future__ import annotations

import json

import httpx
import pytest

from threetears.scrape.driver import NavStep, RenderedPage
from threetears.scrape.drivers.api import ApiDriver, ApiDriverError, _resolve_path

# ===========================================================================
# _resolve_path
# ===========================================================================


class TestResolvePath:
    def test_single_segment_path(self):
        assert _resolve_path({"Results": [1, 2, 3]}, "Results") == [1, 2, 3]

    def test_dotted_multi_segment_path(self):
        assert _resolve_path({"data": {"records": ["a", "b"]}}, "data.records") == ["a", "b"]

    def test_missing_key_raises(self):
        with pytest.raises(ApiDriverError) as exc_info:
            _resolve_path({"Other": []}, "Results")
        assert exc_info.value.code == "bad_results_path"

    def test_missing_nested_key_raises(self):
        with pytest.raises(ApiDriverError) as exc_info:
            _resolve_path({"data": {}}, "data.records")
        assert exc_info.value.code == "bad_results_path"

    def test_non_list_terminal_value_raises(self):
        with pytest.raises(ApiDriverError) as exc_info:
            _resolve_path({"Results": "not a list"}, "Results")
        assert exc_info.value.code == "bad_results_path"

    def test_non_dict_intermediate_value_raises(self):
        with pytest.raises(ApiDriverError) as exc_info:
            _resolve_path({"data": ["not", "a", "dict"]}, "data.records")
        assert exc_info.value.code == "bad_results_path"


# ===========================================================================
# ApiDriver
# ===========================================================================


def _json_response_handler(body: dict, *, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=json.dumps(body).encode())

    return handler


class TestApiDriver:
    def test_name(self):
        driver = ApiDriver()
        assert driver.name == "api"

    async def test_render_concatenates_fragments_into_synthetic_html(self):
        body = {"Results": [{"Html": "<p>Record One</p>"}, {"Html": "<p>Record Two</p>"}]}
        client = httpx.AsyncClient(transport=httpx.MockTransport(_json_response_handler(body)))
        driver = ApiDriver(client=client)

        page = await driver.render("https://example.gov/api/search", results_path="Results", fragment_field="Html")

        assert isinstance(page, RenderedPage)
        assert page.status == 200
        assert "<p>Record One</p>" in page.html
        assert "<p>Record Two</p>" in page.html
        assert page.html.startswith("<html><body>")
        assert page.html.endswith("</body></html>")
        await client.aclose()

    async def test_render_resolves_a_nested_dotted_results_path(self):
        body = {"data": {"records": [{"Html": "<p>Nested Record</p>"}]}}
        client = httpx.AsyncClient(transport=httpx.MockTransport(_json_response_handler(body)))
        driver = ApiDriver(client=client)

        page = await driver.render("https://example.gov/api/search", results_path="data.records", fragment_field="Html")

        assert "<p>Nested Record</p>" in page.html
        await client.aclose()

    async def test_render_skips_records_missing_the_fragment_field(self):
        body = {"Results": [{"Html": "<p>Has It</p>"}, {"Other": "no fragment field here"}]}
        client = httpx.AsyncClient(transport=httpx.MockTransport(_json_response_handler(body)))
        driver = ApiDriver(client=client)

        page = await driver.render("https://example.gov/api/search", results_path="Results", fragment_field="Html")

        assert "<p>Has It</p>" in page.html
        assert "no fragment field here" not in page.html
        await client.aclose()

    async def test_render_raises_when_results_path_is_missing(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(_json_response_handler({"Results": []})))
        driver = ApiDriver(client=client)

        with pytest.raises(ApiDriverError) as exc_info:
            await driver.render("https://example.gov/api/search", fragment_field="Html")

        assert exc_info.value.code == "missing_config"
        await client.aclose()

    async def test_render_raises_when_fragment_field_is_missing(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(_json_response_handler({"Results": []})))
        driver = ApiDriver(client=client)

        with pytest.raises(ApiDriverError) as exc_info:
            await driver.render("https://example.gov/api/search", results_path="Results")

        assert exc_info.value.code == "missing_config"
        await client.aclose()

    async def test_render_raises_on_transport_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = ApiDriver(client=client)

        with pytest.raises(ApiDriverError) as exc_info:
            await driver.render("https://example.gov/api/search", results_path="Results", fragment_field="Html")

        assert exc_info.value.code == "transport"
        await client.aclose()

    async def test_render_raises_on_http_error_status(self):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(403, content=b"forbidden"))
        )
        driver = ApiDriver(client=client)

        with pytest.raises(ApiDriverError) as exc_info:
            await driver.render("https://example.gov/api/search", results_path="Results", fragment_field="Html")

        assert exc_info.value.code == "fetch_failed"
        await client.aclose()

    async def test_render_raises_on_invalid_json(self):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"not json at all"))
        )
        driver = ApiDriver(client=client)

        with pytest.raises(ApiDriverError) as exc_info:
            await driver.render("https://example.gov/api/search", results_path="Results", fragment_field="Html")

        assert exc_info.value.code == "invalid_json"
        await client.aclose()

    async def test_render_raises_on_bad_results_path(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(_json_response_handler({"Other": []})))
        driver = ApiDriver(client=client)

        with pytest.raises(ApiDriverError) as exc_info:
            await driver.render("https://example.gov/api/search", results_path="Results", fragment_field="Html")

        assert exc_info.value.code == "bad_results_path"
        await client.aclose()

    async def test_render_accepts_and_ignores_wait_for_capture_network_and_nav_steps(self):
        """Interface conformance only -- a plain API GET has no browser to
        wait on or drive, but must still accept the full ScrapeDriver
        signature like every other backend."""
        body = {"Results": [{"Html": "hi"}]}
        client = httpx.AsyncClient(transport=httpx.MockTransport(_json_response_handler(body)))
        driver = ApiDriver(client=client)

        page = await driver.render(
            "https://example.gov/api/search",
            wait_for=".content",
            capture_network=True,
            nav_steps=[NavStep(action="click", selector="#x")],
            results_path="Results",
            fragment_field="Html",
        )

        assert isinstance(page, RenderedPage)
        await client.aclose()

    async def test_an_injected_client_is_used_as_given_no_default_user_agent_override(self):
        """The default browser User-Agent (a live-found WAF workaround) only
        applies to a client this driver constructs itself -- an injected
        client's own header policy (or lack of one) must be left alone."""
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["user_agent"] = request.headers.get("user-agent")
            return httpx.Response(200, content=json.dumps({"Results": [{"Html": "hi"}]}).encode())

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = ApiDriver(client=client)

        await driver.render("https://example.gov/api/search", results_path="Results", fragment_field="Html")

        assert captured["user_agent"] != (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
        await client.aclose()
