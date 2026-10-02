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

import httpx
import pytest
from packages.scrape.tests._driver_log_helpers import driver_warnings

from threetears.scrape.driver import NavStep, RenderedPage, ScrapeDriver
from threetears.scrape.drivers.api import ApiDriver, ApiDriverError
from threetears.scrape.drivers.camoufox import CamoufoxDriver
from threetears.scrape.drivers.document import DocumentDriver, DocumentDriverError
from threetears.scrape.drivers.listing_detail import ListingDetailDriver, ListingDetailDriverError
from threetears.scrape.drivers.multi_document import MultiDocumentDriver
from threetears.scrape.drivers.nodriver_download import NodriverDownloadDriver, NodriverDownloadError
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
# Reporting the exit back. A driver that accepts an egress and does not report
# it leaves `ScrapeTargetHealth.last_egress` empty for every target it serves,
# which collapses "walled" and "walled FROM THIS EXIT" -- the distinction that
# column and its migration exist for. ApiDriver honoured an exit and reported
# nothing for exactly as long as nothing asked it to.
# ---------------------------------------------------------------------------


async def _render_api_driver(egress):
    return await ApiDriver(
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=[{"a": 1}]))),
        egress=egress,
    ).render("https://example.gov/api", results_path="")


async def _render_sidecar_driver(egress):
    def _handler(_request: httpx.Request) -> httpx.Response:
        # The sidecar reports the exit itself, so a driver test that invented the value would
        # assert its own arithmetic. This echoes what a sidecar honouring the request would say.
        body = json.loads(_request.content)
        return httpx.Response(
            200,
            json={
                "html": _PAGE_HTML,
                "status": 200,
                "final_url": _PAGE_FINAL_URL,
                "timing_ms": 1.0,
                "egress": body.get("egress_name"),
            },
        )

    return await NodriverSidecarDriver(
        "http://sidecar:8088", client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)), egress=egress
    ).render("https://example.gov/x")


async def _render_document_driver(egress):
    return await DocumentDriver(
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(200, content=b"a plain text document", headers={"content-type": "text/plain"})
            )
        ),
        egress=egress,
    ).render("https://example.gov/notice.txt")


async def _render_listing_detail_driver(egress):
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/listing"):
            body = (
                '<html><body><table><tbody><tr><td><a href="/detail/1">row</a></td></tr></tbody></table></body></html>'
            )
        else:
            body = "<html><body><p>Employer: Acme</p></body></html>"
        return httpx.Response(200, content=body.encode())

    return await ListingDetailDriver(
        row_selector="table tbody tr",
        listing_field_columns={},
        detail_link_column=0,
        detail_field_labels={"employer": "Employer"},
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        pace_delay_seconds=0.0,
        egress=egress,
    ).render("https://example.gov/listing")


async def _render_multi_document_driver(egress):
    # BOTH halves get the same exit. The wrapper reports one only when the listing fetch and the
    # documents agree, so handing it to one side alone would assert the split case here instead
    # of the round trip this contract is about -- `test_driver_multi_document.py` owns the split.
    class _InnerDocumentDriver(DocumentDriver):
        @property
        def egress(self):
            return egress

    return await MultiDocumentDriver(
        document_driver=_InnerDocumentDriver(),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"<html><body></body></html>"))
        ),
        egress=egress,
    ).render("https://example.gov/listing", link_selector="a")


async def _render_camoufox_driver(egress):
    # An INJECTED browser, because launching a real Camoufox here would be a browser download
    # in a unit suite. That is honest for this contract: the round trip under test is
    # "an exit given to the driver comes back on the RenderedPage", which does not depend on
    # the launch. Whether the launch actually carries it is the separate concern that
    # `test_driver_camoufox.py` pins against `_launch_proxy_options`.
    # Reusing the camoufox suite's own browser/page doubles rather than growing a second
    # pair here: two hand-written stand-ins for one Playwright surface drift, and this file
    # already imports a sibling test helper the same way.
    from packages.scrape.tests._camoufox_fakes import FakeCamoufoxBrowser, FakeCamoufoxPage

    return await CamoufoxDriver(browser=FakeCamoufoxBrowser(FakeCamoufoxPage()), egress=egress).render(
        "https://example.gov/x"
    )


_REPORTS_ITS_EXIT = {
    "api": _render_api_driver,
    "camoufox": _render_camoufox_driver,
    "document": _render_document_driver,
    "listing_detail": _render_listing_detail_driver,
    "multi_document": _render_multi_document_driver,
    "nodriver_sidecar": _render_sidecar_driver,
}


@pytest.mark.parametrize("module_name", sorted(_REPORTS_ITS_EXIT), ids=sorted(_REPORTS_ITS_EXIT))
async def test_a_driver_given_an_exit_reports_it_back(module_name: str) -> None:
    """Both halves: a configured exit comes back by name, and an unconfigured one comes back None.

    Asserting only the first would pass against a driver that hard-coded any string; asserting
    only the second would pass against one that reported nothing at all.
    """
    from threetears.core.egress import ProxyEgress

    tor = ProxyEgress("tor", "socks5://127.0.0.1:9050")
    assert (await _REPORTS_ITS_EXIT[module_name](tor)).egress == "tor", (
        f"the {module_name} driver honours an exit but does not say which, so every health row "
        f"it produces records no exit at all"
    )
    assert (await _REPORTS_ITS_EXIT[module_name](None)).egress is None, (
        f"the {module_name} driver claims an exit nobody configured"
    )


def test_every_driver_that_accepts_an_exit_is_covered_above() -> None:
    """The list is checked rather than maintained by hope.

    A driver gaining an `egress` parameter is exactly when this contract starts applying to it,
    and that is the moment nobody thinks to add it to a hand-written list. The sweep fails then,
    naming the driver, instead of the omission surfacing as an empty column months later.
    """
    import importlib
    import inspect
    import pkgutil

    import threetears.scrape.drivers as drivers_pkg

    accepting: set[str] = set()
    for mod_name in (m.name for m in pkgutil.iter_modules(drivers_pkg.__path__)):
        module = importlib.import_module(f"threetears.scrape.drivers.{mod_name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__ or not callable(getattr(obj, "render", None)):
                continue
            if "egress" in inspect.signature(obj.__init__).parameters:
                accepting.add(mod_name)

    assert accepting == set(_REPORTS_ITS_EXIT), (
        "these drivers take an egress but no round-trip test asserts they report it: "
        f"{sorted(accepting - set(_REPORTS_ITS_EXIT))}; and these are covered but no longer take "
        f"one: {sorted(set(_REPORTS_ITS_EXIT) - accepting)}"
    )


# ---------------------------------------------------------------------------
# Dropping a human's solve, tested at the level of the BASE CLASS rather than
# per driver. Three consecutive reviews found this defect one driver at a time:
# the behaviour was added to whichever backend a review named, and the others
# kept discarding a person's credential in silence. Asserting it against every
# accept-and-ignore backend at once is what stops the fourth round.
# ---------------------------------------------------------------------------

_SOLVE = {"cookies": [{"name": "s", "value": "solved"}]}


def _unavailable_client() -> httpx.AsyncClient:
    """A client every request through which is answered 503.

    Each accept-and-ignore backend reports the dropped solve BEFORE it does any I/O, so what
    the fetch returns is irrelevant to the warning; answering 503 keeps every render offline
    and makes each one end in its driver's own documented error, which is asserted.
    """
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(503, text="unavailable")))


#: (driver factory, logger module, the error a 503 ends that render in -- or ``None`` when the
#: backend renders from an injected browser and succeeds).
_DROPS_THE_SOLVE = [
    pytest.param(lambda: ApiDriver(client=_unavailable_client()), "api", ApiDriverError, id="api"),
    pytest.param(lambda: DocumentDriver(client=_unavailable_client()), "document", DocumentDriverError, id="document"),
    pytest.param(
        lambda: ListingDetailDriver(
            row_selector="tr",
            listing_field_columns={0: "employer"},
            detail_link_column=0,
            detail_field_labels={"Employer": "employer"},
            client=_unavailable_client(),
        ),
        "listing_detail",
        ListingDetailDriverError,
        id="listing-detail",
    ),
    pytest.param(
        lambda: NodriverDownloadDriver("http://sidecar:8088", client=_unavailable_client()),
        "nodriver_download",
        NodriverDownloadError,
        id="nodriver-download",
    ),
    pytest.param(lambda: CamoufoxDriver(browser=_ContractFakeBrowser()), "camoufox", None, id="camoufox"),
]


async def _render_with_solve(driver: ScrapeDriver, url: str, error: type[Exception] | None) -> None:
    """Render *url* through *driver* with a human's solve attached, as a caller would.

    :param driver: backend under test
    :ptype driver: ScrapeDriver
    :param url: url to render
    :ptype url: str
    :param error: the error this backend's render ends in offline, or ``None`` if it succeeds
    :ptype error: type[Exception] | None
    :return: nothing
    :rtype: None
    """
    if error is None:
        await driver.render(url, session_state=_SOLVE)
        return
    with pytest.raises(error):
        await driver.render(url, session_state=_SOLVE)


class TestADroppedSolveIsNeverSilent:
    """Every backend that cannot apply a session must say so, not just the reviewed one.

    Driven through ``render(session_state=...)``, the way a caller hands a solve over, so a
    backend that stopped calling the base-class warning from its own render fails here.
    """

    @pytest.mark.parametrize(("make_driver", "module", "error"), _DROPS_THE_SOLVE)
    async def test_it_warns_when_a_solve_is_dropped(self, caplog, make_driver, module: str, error) -> None:
        """Asserted on the emitted record, so deleting the call fails this.

        The failure being excluded is silent: a successful render is returned, the page is the
        login wall, extraction fails, and the target is escalated to a person who already
        cleared it.
        """
        driver = make_driver()
        with caplog.at_level("WARNING", logger=f"threetears.scrape.drivers.{module}"):
            await _render_with_solve(driver, "https://example.gov/x", error)

        assert [r for r in driver_warnings(caplog, module) if "cannot apply it" in r.getMessage()], (
            f"{module} dropped a human's solve without saying so; records: {[(r.name, r.getMessage()) for r in caplog.records]}"
        )

    @pytest.mark.parametrize(("make_driver", "module", "error"), _DROPS_THE_SOLVE)
    async def test_one_origin_is_reported_once_however_many_documents_it_has(
        self, caplog, make_driver, module: str, error
    ) -> None:
        """Per render is a storm: `MultiDocumentDriver` forwards a solve once per document, so
        one listing emitted a warning per document up to the cap, and a warning that repeats
        that way trains its reader to filter it out."""
        driver = make_driver()
        with caplog.at_level("WARNING", logger=f"threetears.scrape.drivers.{module}"):
            for i in range(5):
                await _render_with_solve(driver, f"https://example.gov/doc{i}.pdf", error)

        emitted = [r for r in driver_warnings(caplog, module) if "cannot apply it" in r.getMessage()]
        assert len(emitted) == 1, f"{module} warned {len(emitted)} times for one origin"
        # The TAIL, not just the stem. This is the only operator-visible statement of the
        # cardinality, and it went stale silently once already: every assertion in this branch
        # matched on "cannot apply it" and none on what followed, so the message kept promising
        # "once per driver instance" after the behaviour became once per site -- telling an
        # operator that a second site's silence was expected.
        assert "once per site" in emitted[0].getMessage(), (
            f"the message describes a cardinality the code no longer has: {emitted[0].getMessage()}"
        )

    @pytest.mark.parametrize(("make_driver", "module", "error"), _DROPS_THE_SOLVE)
    async def test_a_second_site_is_still_reported(self, caplog, make_driver, module: str, error) -> None:
        """The opposite failure, and the one that is silent rather than noisy.

        `ScrapeTool` builds its driver map once and reuses it for the life of the process, so
        deduping per driver INSTANCE meant per process: the first target warned and every later
        one was rendered logged-out with nothing said at all. An origin is what a human's solve
        belongs to, so it is the unit that makes this true exactly once per site.
        """
        driver = make_driver()
        with caplog.at_level("WARNING", logger=f"threetears.scrape.drivers.{module}"):
            await _render_with_solve(driver, "https://first.example/a", error)
            await _render_with_solve(driver, "https://second.example/a", error)

        emitted = [r for r in driver_warnings(caplog, module) if "cannot apply it" in r.getMessage()]
        assert len(emitted) == 2, (
            f"{module} reported {len(emitted)} of 2 sites; a driver reused across targets goes "
            "silent after the first, which is the failure this cardinality exists to avoid"
        )

    async def test_the_download_driver_does_not_tell_you_to_use_the_thing_it_is(self, caplog) -> None:
        """It IS sidecar-backed, so the default advice names what it already is.

        The endpoint it posts to carries no session state, which is the actual reason and the
        actual remedy -- generic advice that happens to be wrong is worse than none, because a
        reader who follows it changes nothing and concludes the warning was noise.
        """
        driver = NodriverDownloadDriver("http://sidecar:8088", client=_unavailable_client())
        with caplog.at_level("WARNING", logger="threetears.scrape.drivers.nodriver_download"):
            await _render_with_solve(driver, "https://example.gov/f.pdf", NodriverDownloadError)

        message = driver_warnings(caplog, "nodriver_download")[0].getMessage()
        assert "/v1/download" in message, f"the remedy was not made specific to this driver: {message}"
        assert "Use the nodriver sidecar driver" not in message


#: Far above any sane cap on remembered origins, and still small enough to run in a unit test.
#: The bound is asserted to take effect somewhere below this, not at a particular value.
_ORIGINS_PROBED = 4096


async def test_the_dropped_solve_memory_does_not_grow_without_bound(caplog) -> None:
    """The resource guard, which was the one added branch nothing asserted.

    A long-lived process scraping a wide set of sites would otherwise hold one origin string
    per site it had ever touched, forever -- the same leak the robots gate had to fix. Deleting
    the cap left the suite green while the CHANGELOG claimed the bound, which is the shape where
    a documented guarantee quietly stops being true.

    Asserted on what the bound DOES rather than on the constant: once the remembered set is
    full it is cleared, so the first site ever reported is forgotten and reported again. An
    unbounded set would remember it forever and never re-report it, however many sites passed.
    A bound that is configured and never enforced is exactly the failure worth excluding.
    """
    driver = ApiDriver(client=_unavailable_client())
    first = "https://s0.example/a"

    def first_site_reported() -> bool:
        return any(
            "cannot apply it" in r.getMessage() and first in r.getMessage() for r in driver_warnings(caplog, "api")
        )

    reported_again_after: int | None = None
    with caplog.at_level("WARNING", logger="threetears.scrape.drivers.api"):
        await _render_with_solve(driver, first, ApiDriverError)
        assert first_site_reported()
        for sites_seen in range(1, _ORIGINS_PROBED):
            await _render_with_solve(driver, f"https://s{sites_seen}.example/a", ApiDriverError)
            caplog.clear()
            # Re-rendering a remembered site neither reports nor changes what is remembered, so
            # this probe does not disturb the set it is observing.
            await _render_with_solve(driver, first, ApiDriverError)
            if first_site_reported():
                reported_again_after = sites_seen
                break

    assert reported_again_after is not None, (
        f"the first site was still remembered after {_ORIGINS_PROBED} distinct sites; the set of "
        "reported origins grows with every site the process ever sees"
    )


async def test_urls_with_no_parseable_origin_stay_distinct(caplog) -> None:
    """The fallback branch the docstring's whole design argument rests on, asserted by what is reported.

    `robots._origin_of` returns None for these, deliberately, so it can decline to apply a
    site's rules to something that is not a site. Here the value is only ever a dedupe key, so
    None would collapse every unparseable url into ONE bucket -- the first would be reported
    and the rest silenced. Falling back to the url keeps them distinct, which is what makes the
    two helpers' different return types a decision rather than an accident. The ordinary case
    still keys on the origin rather than the path: two paths on one site report once.
    """
    driver = ApiDriver(client=_unavailable_client())

    with caplog.at_level("WARNING", logger="threetears.scrape.drivers.api"):
        await _render_with_solve(driver, "file.pdf", ApiDriverError)
        await _render_with_solve(driver, "other.pdf", ApiDriverError)
        unparseable = [r for r in driver_warnings(caplog, "api") if "cannot apply it" in r.getMessage()]
        caplog.clear()
        await _render_with_solve(driver, "https://example.gov/a", ApiDriverError)
        await _render_with_solve(driver, "https://example.gov/b", ApiDriverError)
        one_site = [r for r in driver_warnings(caplog, "api") if "cannot apply it" in r.getMessage()]

    assert len(unparseable) == 2, "two unparseable urls collapsed to one dedupe key, so only the first was reported"
    assert len(one_site) == 1, "two paths on one origin were each reported; the key is the path, not the origin"


async def test_two_unparseable_urls_are_each_reported(caplog) -> None:
    """The behaviour that branch exists for, asserted through the warning rather than the helper."""
    driver = ApiDriver(client=_unavailable_client())

    with caplog.at_level("WARNING", logger="threetears.scrape.drivers.api"):
        await _render_with_solve(driver, "garbage-one", ApiDriverError)
        await _render_with_solve(driver, "garbage-two", ApiDriverError)

    emitted = [r for r in driver_warnings(caplog, "api") if "cannot apply it" in r.getMessage()]
    assert len(emitted) == 2, (
        f"reported {len(emitted)} of 2; unparseable urls collapsed into one dedupe key and "
        "silenced everything after the first"
    )
