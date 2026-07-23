"""Unit tests for the ScrapeDriver interface, NodriverSidecarDriver, and the
scrape collections' entity logic.

All tests are fully mocked/in-memory -- no network calls, no sidecar
container. The real, live sidecar proof lives in
tests/integration/test_scrape_nodriver_sidecar_live.py.
"""

from __future__ import annotations

import httpx
import pytest

from threetears.scrape.collections import (
    ScrapeExtraction,
    ScrapeExtractionCollection,
    ScrapeRecipe,
    ScrapeTarget,
    ScrapeTargetCollection,
)
from threetears.scrape.driver import NavStep, NetworkCall, RenderedPage, ScrapeDriver
from threetears.scrape.drivers.nodriver_sidecar import NodriverSidecarDriver, NodriverSidecarError
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

_test_registry = CollectionRegistry()
_test_config = DefaultCoreConfig()


def get_registry() -> CollectionRegistry:
    return _test_registry


def get_config() -> DefaultCoreConfig:
    return _test_config


# ===========================================================================
# RenderedPage / ScrapeDriver
# ===========================================================================


class TestRenderedPage:
    def test_construction(self):
        page = RenderedPage(html="<html></html>", status=200, final_url="https://example.gov", timing_ms=123.4)
        assert page.html == "<html></html>"
        assert page.status == 200
        assert page.final_url == "https://example.gov"
        assert page.timing_ms == 123.4

    def test_network_calls_defaults_to_empty_list(self):
        page = RenderedPage(html="", status=200, final_url="https://example.gov", timing_ms=1.0)
        assert page.network_calls == []

    def test_network_calls_default_is_not_a_shared_mutable(self):
        """A dataclass field default must be a fresh list per instance, not
        one list object shared (and silently accumulated into) across every
        RenderedPage ever constructed."""
        page_a = RenderedPage(html="", status=200, final_url="https://example.gov", timing_ms=1.0)
        page_b = RenderedPage(html="", status=200, final_url="https://example.gov", timing_ms=1.0)
        page_a.network_calls.append(NetworkCall(url="x", method="GET", status=200, content_type="", body=""))
        assert page_b.network_calls == []


class TestScrapeDriverIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ScrapeDriver()  # type: ignore[abstract]


# ===========================================================================
# NodriverSidecarDriver
# ===========================================================================


class TestNodriverSidecarDriver:
    def test_name(self):
        driver = NodriverSidecarDriver("http://localhost:8088")
        assert driver.name == "nodriver"

    async def test_render_success_parses_response(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = request.content
            return httpx.Response(
                200,
                json={
                    "html": "<html>ok</html>",
                    "status": 200,
                    "final_url": "https://example.gov/page",
                    "timing_ms": 456.7,
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverSidecarDriver("http://sidecar.test", client=client)

        page = await driver.render("https://example.gov/page", timeout=15.0, wait_for=".content")

        assert captured["url"] == "http://sidecar.test/v1/render"
        assert page == RenderedPage(
            html="<html>ok</html>", status=200, final_url="https://example.gov/page", timing_ms=456.7
        )
        await client.aclose()

    async def test_render_sends_request_payload_shape(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            captured["payload"] = _json.loads(request.content)
            return httpx.Response(
                200, json={"html": "", "status": 200, "final_url": "https://example.gov", "timing_ms": 1.0}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverSidecarDriver("http://sidecar.test", client=client)

        await driver.render("https://example.gov", timeout=9.5, wait_for=None)

        assert captured["payload"] == {
            "url": "https://example.gov",
            "timeout": 9.5,
            "wait_for": None,
            "capture_network": False,
            "nav_steps": None,
        }
        await client.aclose()

    async def test_render_forwards_capture_network_flag(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            captured["payload"] = _json.loads(request.content)
            return httpx.Response(
                200, json={"html": "", "status": 200, "final_url": "https://example.gov", "timing_ms": 1.0}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverSidecarDriver("http://sidecar.test", client=client)

        await driver.render("https://example.gov", capture_network=True)

        assert captured["payload"]["capture_network"] is True
        await client.aclose()

    async def test_render_forwards_nav_steps_as_plain_dicts(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            captured["payload"] = _json.loads(request.content)
            return httpx.Response(
                200, json={"html": "", "status": 200, "final_url": "https://example.gov", "timing_ms": 1.0}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverSidecarDriver("http://sidecar.test", client=client)
        steps = [NavStep(action="click", selector="#search"), NavStep(action="fill", selector="#q", value="hi")]

        await driver.render("https://example.gov", nav_steps=steps)

        assert captured["payload"]["nav_steps"] == [
            {"action": "click", "selector": "#search", "value": None, "ms": None},
            {"action": "fill", "selector": "#q", "value": "hi", "ms": None},
        ]
        await client.aclose()

    async def test_render_omits_nav_steps_key_when_none(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            captured["payload"] = _json.loads(request.content)
            return httpx.Response(
                200, json={"html": "", "status": 200, "final_url": "https://example.gov", "timing_ms": 1.0}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverSidecarDriver("http://sidecar.test", client=client)

        await driver.render("https://example.gov")

        assert captured["payload"]["nav_steps"] is None
        await client.aclose()

    async def test_render_parses_network_calls_from_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "html": "<html></html>",
                    "status": 200,
                    "final_url": "https://example.gov",
                    "timing_ms": 1.0,
                    "network_calls": [
                        {
                            "url": "https://example.gov/api/notices",
                            "method": "GET",
                            "status": 200,
                            "content_type": "application/json",
                            "body": '{"notices": []}',
                        }
                    ],
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverSidecarDriver("http://sidecar.test", client=client)

        page = await driver.render("https://example.gov", capture_network=True)

        assert page.network_calls == [
            NetworkCall(
                url="https://example.gov/api/notices",
                method="GET",
                status=200,
                content_type="application/json",
                body='{"notices": []}',
            )
        ]
        await client.aclose()

    async def test_render_omits_network_calls_key_defaults_to_empty(self):
        """A sidecar response with no network_calls key at all (older sidecar,
        or capture_network=False) must not crash the parse."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"html": "", "status": 200, "final_url": "https://example.gov", "timing_ms": 1.0}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverSidecarDriver("http://sidecar.test", client=client)

        page = await driver.render("https://example.gov")

        assert page.network_calls == []
        await client.aclose()

    async def test_render_raises_on_sidecar_error_body(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(504, json={"error": {"code": "navigation_timeout", "message": "page did not load"}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverSidecarDriver("http://sidecar.test", client=client)

        with pytest.raises(NodriverSidecarError) as exc_info:
            await driver.render("https://example.gov")

        assert exc_info.value.code == "navigation_timeout"
        assert exc_info.value.message == "page did not load"
        await client.aclose()

    async def test_render_raises_on_transport_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverSidecarDriver("http://sidecar.test", client=client)

        with pytest.raises(NodriverSidecarError) as exc_info:
            await driver.render("https://example.gov")

        assert exc_info.value.code == "transport"
        await client.aclose()


# ===========================================================================
# ScrapeExtraction entity defaults (Chunk 1: pre-eval-loop, pre-enrichment)
# ===========================================================================


class TestScrapeExtractionDefaults:
    def test_pre_eval_loop_fields_default_none(self):
        entity = ScrapeExtraction(
            {
                "target_id": "warn_act_ca",
                "source_url": "https://edd.ca.gov/warn",
                "structured_fields": {"employer": "Acme"},
            }
        )
        assert entity.extraction_recipe_id is None
        assert entity.field_confidences is None
        assert entity.enrichment_notes is None

    def test_validation_status_defaults_needs_review(self):
        entity = ScrapeExtraction({"target_id": "warn_act_ca", "source_url": "https://edd.ca.gov/warn"})
        assert entity.validation_status == "needs_review"

    def test_id_auto_generated_as_uuid7(self):
        entity = ScrapeExtraction({"target_id": "warn_act_ca", "source_url": "https://edd.ca.gov/warn"})
        assert entity.id
        # uuid7's version nibble is 7 (positions per RFC 9562)
        assert entity.id[14] == "7"

    def test_explicit_fields_are_not_overridden_by_defaults(self):
        entity = ScrapeExtraction(
            {
                "target_id": "warn_act_ca",
                "source_url": "https://edd.ca.gov/warn",
                "extraction_recipe_id": "recipe-1",
                "validation_status": "validated",
            }
        )
        assert entity.extraction_recipe_id == "recipe-1"
        assert entity.validation_status == "validated"

    def test_structured_fields_round_trips_dict(self):
        entity = ScrapeExtraction(
            {
                "target_id": "warn_act_ca",
                "source_url": "https://edd.ca.gov/warn",
                "structured_fields": {"employer": "Acme"},
            }
        )
        assert entity.structured_fields == {"employer": "Acme"}


class TestScrapeTargetAndRecipe:
    def test_scrape_target_reads_fields(self):
        target = ScrapeTarget(
            {
                "target_id": "warn_act_ca",
                "url": "https://edd.ca.gov/warn",
                "driver_backend": "nodriver",
                "rate_limit_key": "warn_act_state_sites",
                "cadence": "daily",
            }
        )
        assert target.target_id == "warn_act_ca"
        assert target.driver_backend == "nodriver"
        assert target.rate_limit_key == "warn_act_state_sites"

    def test_scrape_recipe_failure_count_defaults_zero(self):
        recipe = ScrapeRecipe({"target_id": "warn_act_ca", "extraction_strategy": {}})
        assert recipe.consecutive_validation_failures == 0


# ===========================================================================
# ScrapeExtractionCollection persistence (in-memory L3 fallback)
# ===========================================================================


class TestScrapeExtractionCollection:
    async def test_save_and_get_round_trips(self):
        collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
        entity = collection.create(
            {
                "target_id": "warn_act_ca",
                "source_url": "https://edd.ca.gov/warn",
                "structured_fields": {"employer": "Acme Corp"},
            }
        )
        await collection.save_entity(entity)

        fetched = await collection.get(entity.id)
        assert fetched is not None
        assert fetched.target_id == "warn_act_ca"
        assert fetched.structured_fields == {"employer": "Acme Corp"}
        assert fetched.extraction_recipe_id is None
        assert fetched.validation_status == "needs_review"

    async def test_table_name(self):
        collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
        assert collection.table_name == "scrape_extractions"


class TestScrapeTargetCollection:
    async def test_save_and_get_round_trips(self):
        collection = ScrapeTargetCollection(get_registry(), get_config(), nats_client=None)
        entity = collection.create(
            {
                "target_id": "warn_act_ca",
                "url": "https://edd.ca.gov/warn",
                "driver_backend": "nodriver",
                "rate_limit_key": "warn_act_state_sites",
                "cadence": "daily",
            }
        )
        await collection.save_entity(entity)

        fetched = await collection.get("warn_act_ca")
        assert fetched is not None
        assert fetched.url == "https://edd.ca.gov/warn"
