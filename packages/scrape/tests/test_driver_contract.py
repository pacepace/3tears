"""Shared, parametrized ``ScrapeDriver`` contract tests -- Chunk 6's own
acceptance criteria: "no ScrapeDriver-contract test is nodriver-specific."

Both backends (NodriverSidecarDriver, CamoufoxDriver) are constructed with a
backend-specific injected fake (httpx.MockTransport / a fake Playwright
Browser) that produces the SAME logical page, then run through the SAME
generic assertions here -- the actual proof that ``ScrapeDriver`` is a real,
backend-agnostic interface and not secretly shaped around one backend's
assumptions. Backend-specific behavior (payload shapes, error codes,
timeout-unit conversions) is tested in each backend's own test file
(tests/scrape/test_driver_nodriver_sidecar.py, tests/scrape/test_driver_camoufox.py).

DocumentDriver (Chunk 17) deliberately does NOT join ``_BACKENDS`` below --
see tests/scrape/test_driver_document.py's own module docstring for why
(it transforms content into synthetic HTML rather than passing through
already-HTML source verbatim, so this file's exact-content-equality
assertion doesn't apply to it the same way).
"""

from __future__ import annotations

import json

import httpx
import pytest

from threetears.scrape.driver import NavStep, RenderedPage, ScrapeDriver
from threetears.scrape.drivers.camoufox import CamoufoxDriver
from threetears.scrape.drivers.nodriver_sidecar import NodriverSidecarDriver

_PAGE_HTML = "<html><body>contract test page</body></html>"
_PAGE_STATUS = 200
_PAGE_FINAL_URL = "https://example.gov/contract-page"

#: A deliberately generic marker value (not Google/Trends-shaped) both fake
#: backends return for an ``evaluate`` step -- the "would this help a
#: different, unrelated target" gaming test's own return value.
_CONTRACT_EVAL_RESULT = {"generic": "capability", "not": "google-specific"}


def _nodriver_backend() -> ScrapeDriver:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        nav_steps = payload.get("nav_steps") or []
        eval_results = [_CONTRACT_EVAL_RESULT for step in nav_steps if step.get("action") == "evaluate"]
        return httpx.Response(
            200,
            json={
                "html": _PAGE_HTML,
                "status": _PAGE_STATUS,
                "final_url": _PAGE_FINAL_URL,
                "timing_ms": 12.3,
                "eval_results": eval_results,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return NodriverSidecarDriver("http://sidecar.test", client=client)


# parity-exempt: hand-rolled subset stub of Playwright's third-party Page (only goto/content/url/close/on, the only surface CamoufoxDriver calls) -- duplicated from test_driver_camoufox.py to keep this file self-contained as the contract's source of truth
class _ContractFakePage:
    def __init__(self) -> None:
        self.url = _PAGE_FINAL_URL

    async def goto(self, url, *, timeout=None, wait_until=None):
        return _ContractFakeResponse()

    async def content(self):
        return _PAGE_HTML

    async def close(self):
        pass

    def on(self, event, handler):
        pass  # no response events ever fire in this minimal contract stub

    async def click(self, selector, *, timeout=None):
        pass

    async def fill(self, selector, value, *, timeout=None):
        pass

    async def wait_for_selector(self, selector, *, timeout=None):
        pass

    async def wait_for_timeout(self, ms):
        pass

    async def evaluate(self, expression):
        return _CONTRACT_EVAL_RESULT


# parity-exempt: hand-rolled subset stub of Playwright's third-party Response (only .status, the only attribute CamoufoxDriver reads)
class _ContractFakeResponse:
    status = _PAGE_STATUS


# parity-exempt: hand-rolled subset stub of Playwright's third-party Browser (only new_page(), the only method CamoufoxDriver calls)
class _ContractFakeBrowser:
    async def new_page(self):
        return _ContractFakePage()


def _camoufox_backend() -> ScrapeDriver:
    return CamoufoxDriver(browser=_ContractFakeBrowser())


_BACKENDS = [
    pytest.param(_nodriver_backend, id="nodriver"),
    pytest.param(_camoufox_backend, id="camoufox"),
]


class TestScrapeDriverContract:
    """Every ``ScrapeDriver`` backend must satisfy this identical contract."""

    @pytest.mark.parametrize("make_driver", _BACKENDS)
    def test_name_is_a_stable_nonempty_string(self, make_driver):
        driver = make_driver()
        assert isinstance(driver.name, str)
        assert driver.name

    @pytest.mark.parametrize("make_driver", _BACKENDS)
    async def test_render_returns_a_rendered_page_with_correct_field_types(self, make_driver):
        driver = make_driver()

        page = await driver.render("https://example.gov/contract-page")

        assert isinstance(page, RenderedPage)
        assert isinstance(page.html, str)
        assert isinstance(page.status, int)
        assert isinstance(page.final_url, str)
        assert isinstance(page.timing_ms, float)
        assert isinstance(page.network_calls, list)
        assert isinstance(page.eval_results, list)

    @pytest.mark.parametrize("make_driver", _BACKENDS)
    async def test_render_returns_the_backend_supplied_content(self, make_driver):
        driver = make_driver()

        page = await driver.render("https://example.gov/contract-page")

        assert page.html == _PAGE_HTML
        assert page.status == _PAGE_STATUS
        assert page.final_url == _PAGE_FINAL_URL

    @pytest.mark.parametrize("make_driver", _BACKENDS)
    async def test_render_accepts_default_and_explicit_timeout_and_wait_for(self, make_driver):
        """Every backend's render() must accept the full ScrapeDriver signature,
        even if a given backend ignores wait_for internally -- the caller-facing
        contract is what's pinned here, not each backend's internal handling."""
        driver = make_driver()

        page_default = await driver.render("https://example.gov/contract-page")
        page_explicit = await driver.render("https://example.gov/contract-page", timeout=5.0, wait_for=None)

        assert page_default.html == page_explicit.html

    @pytest.mark.parametrize("make_driver", _BACKENDS)
    async def test_render_accepts_capture_network(self, make_driver):
        """Every backend's render() must accept capture_network -- real
        capture behavior (what gets filtered in/out) is each backend's own
        test file's responsibility, per this file's own docstring."""
        driver = make_driver()

        page = await driver.render("https://example.gov/contract-page", capture_network=True)

        assert isinstance(page.network_calls, list)

    @pytest.mark.parametrize("make_driver", _BACKENDS)
    async def test_render_accepts_nav_steps(self, make_driver):
        """Every backend's render() must accept nav_steps -- real step
        execution (click/fill/wait_for/wait_ms semantics, failure modes) is
        each backend's own test file's responsibility, per this file's own
        docstring."""
        driver = make_driver()

        page = await driver.render(
            "https://example.gov/contract-page",
            nav_steps=[NavStep(action="click", selector="#search"), NavStep(action="wait_ms", ms=10)],
        )

        assert page.html == _PAGE_HTML

    @pytest.mark.parametrize("make_driver", _BACKENDS)
    async def test_evaluate_step_is_a_general_capability_not_google_specific(self, make_driver):
        """Gaming test: runs a plain JS expression against a synthetic
        contract-test page (https://example.gov/contract-page) wholly
        unrelated to Google/Trends. If ``evaluate`` only worked there, it
        would be a Trends fix wearing a general name, not a real platform
        capability -- see threetears.scrape.driver.NavStep's own docstring."""
        driver = make_driver()

        page = await driver.render(
            "https://example.gov/contract-page",
            nav_steps=[NavStep(action="evaluate", value="1 + 1")],
        )

        assert page.eval_results == [_CONTRACT_EVAL_RESULT]
