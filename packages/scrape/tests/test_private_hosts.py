"""The SSRF guard follows the fetch: every request a guarded scrape sends is checked, not just the URL.

``ScrapeTool`` refuses a target whose host resolves to a non-public address. The HTTP clients
behind it follow redirects and the link-following drivers fetch URLs a page chose, so a public
URL answering ``302 Location: http://169.254.169.254/...`` used to be fetched after the check
had passed. These tests drive the real drivers and the real robots gate through ``ScrapeTool``,
the one front door that turns the guard on, over a recording exit, and read what reached the
wire.

Hosts are IP literals throughout. The guard resolves every host it checks, and a literal
resolves without a lookup, so nothing here depends on DNS or reaches the network.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, patch
from urllib.parse import urljoin

import httpx
import pytest
from packages.scrape.tests.egress_fakes import FakeEgress
from packages.scrape.tests.scrape_tool_support import derived_target_id
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.models.circuit_breaker import CircuitBreaker, CircuitState
from threetears.scrape.circuit import TargetCircuit
from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
from threetears.scrape.driver import NavStep, RenderedPage, ScrapeDriver
from threetears.scrape.drivers.api import ApiDriver
from threetears.scrape.drivers.document import DocumentDriver
from threetears.scrape.drivers.listing_detail import ListingDetailDriver
from threetears.scrape.drivers.multi_document import MultiDocumentDriver
from threetears.scrape.health import ScrapeTargetHealthCollection
from threetears.scrape.robots import RobotsGate
from threetears.scrape.tool import ScrapeTool

_START = "https://8.8.8.8/start"
_OTHER_PUBLIC = "https://8.8.4.4/elsewhere"
_LOOPBACK = "http://127.0.0.1/admin"
_METADATA = "http://169.254.169.254/latest/meta-data/"
_SCHEMA = {"employer": "str"}

_LISTING_HTML = """
<html><body><table><tbody>
  <tr><td>Acme Corp</td><td><a href="{first}">detail</a></td></tr>
  <tr><td>Beta LLC</td><td><a href="{second}">detail</a></td></tr>
</tbody></table></body></html>
"""

_DETAIL_HTML = (
    '<html><body><div class="definition-list__title">Affected</div>'
    '<div class="definition-list__definition">42</div></body></html>'
)


def _redirect(location: str) -> httpx.Response:
    return httpx.Response(302, headers={"location": location})


def _routes(table: dict[str, httpx.Response]) -> Any:
    """An exit's answer per URL; anything not in *table* is a 404."""

    def _respond(request: httpx.Request) -> httpx.Response:
        return table.get(str(request.url), httpx.Response(404))

    return _respond


def _requested(egress: FakeEgress) -> list[str]:
    return [str(request.url) for request in egress.requests]


def _hosts(egress: FakeEgress) -> set[str]:
    return {request.url.host for request in egress.requests}


class _BoundDriver(ScrapeDriver):
    """Hands an HTTP driver the per-target configuration ``ScrapeTool`` has no input for.

    ``ApiDriver`` and ``MultiDocumentDriver`` need a ``results_path`` or ``link_selector`` on
    every render, and the tool passes neither, so a deployment that puts one behind the tool
    wraps it like this. The guard has to hold for those drivers there too.
    """

    def __init__(self, inner: ScrapeDriver, **bound: Any) -> None:
        self._inner = inner
        self._bound = bound

    @property
    def name(self) -> str:
        return self._inner.name

    async def render(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        wait_for: str | None = None,
        capture_network: bool = False,
        nav_steps: list[NavStep] | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> RenderedPage:
        return await self._inner.render(url, timeout=timeout, **self._bound)


def _driver(kind: str, egress: FakeEgress) -> ScrapeDriver:
    if kind == "document":
        return DocumentDriver(egress=egress)
    if kind == "listing_detail":
        return ListingDetailDriver(
            row_selector="tbody tr",
            listing_field_columns={0: "employer"},
            detail_link_column=1,
            detail_field_labels={"affected_count": "Affected"},
            pace_delay_seconds=0.0,
            egress=egress,
        )
    if kind == "api":
        return _BoundDriver(ApiDriver(egress=egress), results_path="")
    if kind == "multi_document":
        return _BoundDriver(
            MultiDocumentDriver(document_driver=DocumentDriver(egress=egress), egress=egress), link_selector="a"
        )
    raise AssertionError(f"no such driver kind {kind!r}")


def _tool(driver: ScrapeDriver, **kwargs: Any) -> ScrapeTool:
    registry, config = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    return ScrapeTool(
        recipe_collection=ScrapeRecipeCollection(registry, config, nats_client=None),
        extraction_collection=ScrapeExtractionCollection(registry, config, nats_client=None),
        drivers={"under_test": driver},
        api_key="unused-the-eval-loop-is-not-reached-or-is-patched",
        **{"robots": None, **kwargs},
    )


async def _scrape(tool: ScrapeTool, url: str = _START) -> Any:
    return await tool.execute(url=url, field_schema=_SCHEMA, driver_backend="under_test")


class TestAnInwardRedirectIsRefused:
    """A redirect into private address space is refused like a private target URL would be."""

    @pytest.mark.parametrize("inward", [_LOOPBACK, _METADATA])
    @pytest.mark.parametrize("kind", ["document", "listing_detail", "api", "multi_document"])
    async def test_the_redirect_target_is_never_requested(self, kind: str, inward: str) -> None:
        egress = FakeEgress(respond=_routes({_START: _redirect(inward)}))

        result = await _scrape(_tool(_driver(kind, egress)))

        assert _requested(egress) == [_START], f"the inward hop left the exit: {_requested(egress)}"
        assert result.success is False
        assert (result.error or "").startswith("refused: "), (
            f"a refused redirect must answer as the SSRF refusal it is, not as {result.error!r}"
        )
        assert httpx.URL(inward).host in (result.error or "")

    async def test_the_refusal_is_logged_as_a_security_event(self, caplog: pytest.LogCaptureFixture) -> None:
        egress = FakeEgress(respond=_routes({_START: _redirect(_METADATA)}))

        with caplog.at_level(logging.WARNING):
            await _scrape(_tool(_driver("document", egress)))

        refusals = [
            rec for rec in caplog.records if rec.levelno == logging.WARNING and "SSRF guard refused" in rec.getMessage()
        ]
        assert refusals, "a redirect into the metadata endpoint left no WARNING naming the SSRF refusal"
        assert _METADATA in refusals[0].getMessage()

    async def test_every_hop_is_checked_not_only_the_first_redirect(self) -> None:
        """A public hop first, then the inward one: the check runs per hop, so depth does not help."""
        egress = FakeEgress(respond=_routes({_START: _redirect(_OTHER_PUBLIC), _OTHER_PUBLIC: _redirect(_LOOPBACK)}))

        result = await _scrape(_tool(_driver("document", egress)))

        assert _requested(egress) == [_START, _OTHER_PUBLIC]
        assert (result.error or "").startswith("refused: ")

    async def test_a_refused_redirect_returns_the_circuits_probe(self) -> None:
        """The refusal reports an outcome, so a half-open breaker does not hold its probe forever."""
        target_id = await derived_target_id(_START, _SCHEMA)
        registry, config = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
        health = ScrapeTargetHealthCollection(registry, config, nats_client=None)
        breaker = CircuitBreaker(target_id, failure_threshold=1, recovery_timeout_seconds=0.0)
        breaker.record_failure()
        egress = FakeEgress(respond=_routes({_START: _redirect(_LOOPBACK)}))
        tool = _tool(
            _driver("document", egress),
            health_collection=health,
            circuit=TargetCircuit(health, breaker_for=lambda _target: breaker),
        )

        result = await _scrape(tool)

        assert (result.error or "").startswith("refused: ")
        assert breaker.state is not CircuitState.HALF_OPEN, "the refused redirect stranded the circuit's probe"


class TestALinkThePageChoseIsChecked:
    """``listing_detail`` fetches each row's detail link, and the page decides where it points."""

    async def test_an_inward_detail_link_is_refused_and_its_row_keeps_its_listing_fields(self) -> None:
        second = urljoin(_START, "/detail/2")
        egress = FakeEgress(
            respond=_routes(
                {
                    _START: httpx.Response(200, html=_LISTING_HTML.format(first=_METADATA, second="/detail/2")),
                    second: httpx.Response(200, html=_DETAIL_HTML),
                }
            )
        )
        eval_loop = AsyncMock(return_value=None)

        with patch("threetears.scrape.tool.run_eval_loop", eval_loop):
            result = await _scrape(_tool(_driver("listing_detail", egress)))

        assert "169.254.169.254" not in _hosts(egress), "a detail link into the metadata endpoint was fetched"
        assert _requested(egress) == [_START, second]
        assert not (result.error or "").startswith("refused: "), "one hostile row refused the whole listing"
        page_html = eval_loop.await_args.args[1]
        assert "Acme Corp" in page_html and "Beta LLC" in page_html, "a row was dropped rather than degraded"
        assert "42" in page_html, "the public row lost its detail fields"


class TestTheRobotsReadIsChecked:
    """The robots gate fetches ``/robots.txt`` before the page, and that can redirect too."""

    async def test_a_robots_file_redirecting_inward_is_not_followed(self) -> None:
        robots_url = urljoin(_START, "/robots.txt")
        egress = FakeEgress(respond=_routes({robots_url: _redirect(_LOOPBACK)}))
        tool = _tool(_driver("document", egress), robots=RobotsGate(egress=egress))

        result = await _scrape(tool)

        assert robots_url in _requested(egress), "the robots file was never read, so this proves nothing"
        assert "127.0.0.1" not in _hosts(egress), "the robots read followed a redirect to loopback"
        # An unreadable robots.txt means "the site told us nothing", so the scrape goes ahead.
        assert _START in _requested(egress)
        assert (result.error or "").startswith("fetch failed: "), result.error

    async def test_without_the_guard_the_robots_redirect_is_followed(self) -> None:
        robots_url = urljoin(_START, "/robots.txt")
        egress = FakeEgress(respond=_routes({robots_url: _redirect(_LOOPBACK)}))
        tool = _tool(_driver("document", egress), robots=RobotsGate(egress=egress), block_private_hosts=False)

        await _scrape(tool)

        assert _LOOPBACK in _requested(egress)


class TestWhatTheGuardLeavesAlone:
    """Allowed fetches are unchanged, and so is every caller that did not turn the guard on."""

    async def test_a_redirect_to_another_public_host_is_followed(self) -> None:
        egress = FakeEgress(
            respond=_routes(
                {
                    _START: _redirect(_OTHER_PUBLIC),
                    _OTHER_PUBLIC: httpx.Response(200, html=_LISTING_HTML.format(first="/a", second="/b")),
                }
            )
        )
        eval_loop = AsyncMock(return_value=None)

        with patch("threetears.scrape.tool.run_eval_loop", eval_loop):
            result = await _scrape(_tool(_driver("multi_document", egress)))

        assert _requested(egress)[:2] == [_START, _OTHER_PUBLIC]
        assert not (result.error or "").startswith("refused: "), result.error
        assert eval_loop.await_args.args[2] == _OTHER_PUBLIC, "the page was not the redirect's target"

    @pytest.mark.parametrize("kind", ["document", "listing_detail", "api", "multi_document"])
    async def test_a_tool_built_without_the_guard_follows_an_inward_redirect(self, kind: str) -> None:
        """``block_private_hosts=False`` is a deployment that scrapes internal hosts on purpose."""
        egress = FakeEgress(respond=_routes({_START: _redirect(_LOOPBACK)}))

        result = await _scrape(_tool(_driver(kind, egress), block_private_hosts=False))

        assert _requested(egress)[:2] == [_START, _LOOPBACK]
        assert not (result.error or "").startswith("refused: ")

    async def test_a_driver_rendered_outside_a_tool_is_unchanged(self) -> None:
        """The guard is the tool's setting; a scheduler driving a driver directly never asked for it."""
        egress = FakeEgress(respond=_routes({_START: _redirect(_LOOPBACK)}))

        with pytest.raises(Exception, match="404"):
            await DocumentDriver(egress=egress).render(_START)

        assert _requested(egress) == [_START, _LOOPBACK]
