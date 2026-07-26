"""Shared, parametrized ``ScrapeDriver`` contract tests, holding to one rule:
no ScrapeDriver-contract test may be nodriver-specific.

Both backends (NodriverSidecarDriver, CamoufoxDriver) are constructed with a
backend-specific injected fake (httpx.MockTransport / a fake Playwright
Browser) that produces the SAME logical page, then run through the SAME
generic assertions here -- the actual proof that ``ScrapeDriver`` is a real,
backend-agnostic interface and not secretly shaped around one backend's
assumptions. Backend-specific behavior (payload shapes, error codes,
timeout-unit conversions) is tested in each backend's own test file
(test_driver_nodriver_sidecar.py, test_driver_camoufox.py).

DocumentDriver deliberately does NOT join ``_BACKENDS`` below --
see test_driver_document.py's own module docstring for why
(it transforms content into synthetic HTML rather than passing through
already-HTML source verbatim, so this file's exact-content-equality
assertion doesn't apply to it the same way).
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from threetears.scrape.driver import NavStep, RenderedPage, ScrapeDriver
from threetears.scrape.drivers.api import ApiDriver
from threetears.scrape.drivers.camoufox import CamoufoxDriver
from threetears.scrape.drivers.document import DocumentDriver
from threetears.scrape.drivers.listing_detail import ListingDetailDriver
from threetears.scrape.drivers.nodriver_download import NodriverDownloadDriver
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

    @pytest.mark.parametrize("make_driver", _BACKENDS)
    async def test_a_parametrized_backend_accepts_and_ignores_session_state(self, make_driver):
        """The "accept the full signature, use what you need" rule, applied to a new parameter.

        A backend that cannot restore a browser session has nothing to do with one, but it
        must still ACCEPT it: the alternative is every caller branching on which driver it
        happens to hold, which is the coupling this protocol exists to prevent. The same rule
        already governs ``link_selector``, ``results_path`` and ``seen_urls``.

        This covers the two backends this suite parametrizes, which is a behavioural check
        against real objects. ``test_every_render_implementation_declares_session_state``
        below covers all nine by signature, because constructing every composite backend here
        would be a different and much heavier test than this file is for.
        """
        driver = make_driver()

        page = await driver.render(
            "https://example.gov/contract-page",
            session_state={"cookies": [{"name": "cf_clearance", "value": "x", "domain": ".example.gov"}]},
        )

        assert isinstance(page, RenderedPage)

    @pytest.mark.parametrize("make_driver", _BACKENDS)
    async def test_session_state_defaults_to_absent(self, make_driver):
        """Every pre-existing caller keeps working without knowing this parameter exists."""
        driver = make_driver()
        page = await driver.render("https://example.gov/contract-page")
        assert isinstance(page, RenderedPage)


def test_every_render_implementation_declares_session_state():
    """All nine ``render`` implementations, by signature rather than by construction.

    The parametrized contract tests above instantiate two representative backends. The other
    seven are composites and wrappers whose construction needs collections, HTTP clients or a
    parent driver, so exercising them here would make this file about fixtures rather than
    about the contract. A signature check is weaker than a call, but it covers the whole set
    and it catches the failure that actually happens: a new parameter added to the protocol
    and to some of its implementers, leaving one that raises ``TypeError`` the first time a
    caller passes it -- at runtime, in whichever deployment happens to use that backend.
    """
    import importlib
    import inspect
    import pkgutil

    import threetears.scrape.drivers as drivers_pkg
    from threetears.scrape.driver import ScrapeDriver

    checked: list[str] = []
    modules = [m.name for m in pkgutil.iter_modules(drivers_pkg.__path__)]
    for mod_name in modules:
        module = importlib.import_module(f"threetears.scrape.drivers.{mod_name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            render = getattr(obj, "render", None)
            if render is None or not callable(render):
                continue
            params = inspect.signature(render).parameters
            if "url" not in params:
                continue
            checked.append(f"{mod_name}.{obj.__name__}")
            assert "session_state" in params, (
                f"{mod_name}.{obj.__name__}.render does not accept session_state, so a caller "
                f"passing it gets a TypeError at runtime rather than a driver that ignores it"
            )

    assert "session_state" in inspect.signature(ScrapeDriver.render).parameters
    assert len(checked) >= 8, f"the sweep only found {len(checked)} render implementations: {checked}"


# ---------------------------------------------------------------------------
# Dropping a human's solve, tested at the level of the BASE CLASS rather than
# per driver. Three consecutive reviews found this defect one driver at a time:
# the behaviour was added to whichever backend a review named, and the others
# kept discarding a person's credential in silence. Asserting it against every
# accept-and-ignore backend at once is what stops the fourth round.
# ---------------------------------------------------------------------------

_DROPS_THE_SOLVE = [
    pytest.param(lambda: ApiDriver(), "api", id="api"),
    pytest.param(lambda: DocumentDriver(), "document", id="document"),
    pytest.param(
        lambda: ListingDetailDriver(
            row_selector="tr",
            listing_field_columns={0: "employer"},
            detail_link_column=0,
            detail_field_labels={"Employer": "employer"},
        ),
        "listing_detail",
        id="listing-detail",
    ),
    pytest.param(lambda: NodriverDownloadDriver("http://sidecar:8088"), "nodriver_download", id="nodriver-download"),
    pytest.param(lambda: CamoufoxDriver(), "camoufox", id="camoufox"),
]


class TestADroppedSolveIsNeverSilent:
    """Every backend that cannot apply a session must say so, not just the reviewed one."""

    @pytest.mark.parametrize(("make_driver", "module"), _DROPS_THE_SOLVE)
    def test_it_warns_when_a_solve_is_dropped(self, caplog, make_driver, module: str) -> None:
        """Asserted on the emitted record, so deleting the call fails this.

        The failure being excluded is silent: a successful render is returned, the page is the
        login wall, extraction fails, and the target is escalated to a person who already
        cleared it.
        """
        driver = make_driver()
        with caplog.at_level("WARNING", logger=f"threetears.scrape.drivers.{module}"):
            driver._warn_dropped_session_state(
                "https://example.gov/x", logging.getLogger(f"threetears.scrape.drivers.{module}")
            )

        assert any("cannot apply it" in r.getMessage() for r in caplog.records), (
            f"{module} dropped a human's solve without saying so; records: {[r.getMessage() for r in caplog.records]}"
        )

    @pytest.mark.parametrize(("make_driver", "module"), _DROPS_THE_SOLVE)
    def test_it_says_so_once_per_instance_not_once_per_render(self, caplog, make_driver, module: str) -> None:
        """A per-render warning is a storm, and a storm trains its reader to filter it.

        `MultiDocumentDriver` forwards a solve to its inner document driver once per document,
        so per-render meant one warning per document up to the cap. The fact reported is a
        property of the driver and does not change between calls.
        """
        driver = make_driver()
        log = logging.getLogger(f"threetears.scrape.drivers.{module}")
        with caplog.at_level("WARNING", logger=f"threetears.scrape.drivers.{module}"):
            for _ in range(5):
                driver._warn_dropped_session_state("https://example.gov/x", log)

        emitted = [r for r in caplog.records if "cannot apply it" in r.getMessage()]
        assert len(emitted) == 1, f"{module} warned {len(emitted)} times for one instance"

    def test_the_download_driver_does_not_tell_you_to_use_the_thing_it_is(self, caplog) -> None:
        """It IS sidecar-backed, so the default advice names what it already is.

        The endpoint it posts to carries no session state, which is the actual reason and the
        actual remedy -- generic advice that happens to be wrong is worse than none, because a
        reader who follows it changes nothing and concludes the warning was noise.
        """
        driver = NodriverDownloadDriver("http://sidecar:8088")
        log = logging.getLogger("threetears.scrape.drivers.nodriver_download")
        with caplog.at_level("WARNING", logger="threetears.scrape.drivers.nodriver_download"):
            driver._warn_dropped_session_state("https://example.gov/f.pdf", log)

        message = caplog.records[0].getMessage()
        assert "/v1/download" in message, f"the remedy was not made specific to this driver: {message}"
        assert "Use the nodriver sidecar driver" not in message
