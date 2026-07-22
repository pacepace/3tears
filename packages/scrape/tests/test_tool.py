"""Unit tests for threetears.scrape.tool.ScrapeTool -- the ad-hoc MCP scrape tool.

All LLM/driver calls are mocked/fake; the real sidecar + real LLM proof
lives in a live script exercised manually against the running sidecar
(see build-plan.md Chunk 18's Context note) and in
tests/e2e/test_warn_act_eval_loop_live.py's own live driver+LLM proof for
the eval loop this tool is a thin wrapper over.
"""

from __future__ import annotations

import json

from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
from threetears.scrape.driver import NavStep, RenderedPage
from threetears.scrape.tool import ScrapeTool, _derive_target_id
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

_test_registry = CollectionRegistry()
_test_config = DefaultCoreConfig()


def get_registry() -> CollectionRegistry:
    return _test_registry


def get_config() -> DefaultCoreConfig:
    return _test_config


_ROW_STRATEGY = {
    "row_selector": "tbody tr",
    "field_selectors": {"employer": "td:nth-child(1)", "affected_count": "td:nth-child(2)"},
}
_SINGLE_STRATEGY = {"employer": "td:nth-child(1)", "affected_count": "td:nth-child(2)"}

_ROWS_HTML = """
<html><body><table><tbody>
  <tr><td>Acme Corp</td><td>42</td></tr>
  <tr><td>Beta LLC</td><td>7</td></tr>
</tbody></table></body></html>
"""

_SINGLE_HTML = "<html><body><table><tr><td>Acme Corp</td><td>42</td></tr></table></body></html>"


# parity-with: threetears.scrape.driver.ScrapeDriver
class _FakeDriver:
    def __init__(self, html: str, final_url: str = "https://example.gov/warn", raise_exc: Exception | None = None):
        self._html = html
        self._final_url = final_url
        self._raise_exc = raise_exc
        self.render_calls: list[str] = []
        self.wait_for_calls: list[str | None] = []
        self.nav_steps_calls: list[list[NavStep] | None] = []

    @property
    def name(self) -> str:
        return "fake"

    async def render(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        wait_for: str | None = None,
        capture_network: bool = False,
        nav_steps: list[NavStep] | None = None,
    ) -> RenderedPage:
        self.render_calls.append(url)
        self.wait_for_calls.append(wait_for)
        self.nav_steps_calls.append(nav_steps)
        if self._raise_exc is not None:
            raise self._raise_exc
        return RenderedPage(html=self._html, status=200, final_url=self._final_url, timing_ms=1.0)


def _collections():
    return (
        ScrapeRecipeCollection(get_registry(), get_config(), nats_client=None),
        ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None),
    )


async def _seed_recipe(recipe_collection, target_id: str, strategy: dict) -> None:
    entity = recipe_collection.create(
        {
            "target_id": target_id,
            "extraction_strategy": strategy,
            "won_at": None,
            "last_validated_at": None,
            "consecutive_validation_failures": 0,
        }
    )
    await recipe_collection.save_entity(entity)


class TestDeriveTargetId:
    def test_deterministic_for_the_same_url_and_schema(self):
        assert _derive_target_id("https://x.gov", {"employer": "str"}) == _derive_target_id(
            "https://x.gov", {"employer": "str"}
        )

    def test_field_order_does_not_change_the_id(self):
        assert _derive_target_id("https://x.gov", {"a": "str", "b": "int"}) == _derive_target_id(
            "https://x.gov", {"b": "int", "a": "str"}
        )

    def test_different_url_changes_the_id(self):
        assert _derive_target_id("https://x.gov", {"a": "str"}) != _derive_target_id("https://y.gov", {"a": "str"})

    def test_different_schema_changes_the_id(self):
        assert _derive_target_id("https://x.gov", {"a": "str"}) != _derive_target_id("https://x.gov", {"b": "str"})

    def test_starts_with_adhoc_prefix(self):
        assert _derive_target_id("https://x.gov", {"a": "str"}).startswith("adhoc_")


class TestScrapeToolSchema:
    def test_mcp_name_and_version(self):
        recipe_collection, extraction_collection = _collections()
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={},
            api_key="k",
        )
        assert tool.mcp_name() == "3tears.scrape"
        assert tool.mcp_version()

    def test_schema_requires_url_and_field_schema(self):
        recipe_collection, extraction_collection = _collections()
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={},
            api_key="k",
        )
        schema = tool.mcp_schema()
        assert schema.input_schema["required"] == ["url", "field_schema"]
        assert "nav_steps" in schema.input_schema["properties"]
        assert "driver_backend" in schema.input_schema["properties"]


class TestScrapeToolExecute:
    async def test_missing_url_is_an_error(self):
        recipe_collection, extraction_collection = _collections()
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={"nodriver": _FakeDriver(_SINGLE_HTML)},
            api_key="k",
        )

        result = await tool.execute(field_schema={"employer": "str"})

        assert result.success is False
        assert "url" in (result.error or "")

    async def test_missing_field_schema_is_an_error(self):
        recipe_collection, extraction_collection = _collections()
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={"nodriver": _FakeDriver(_SINGLE_HTML)},
            api_key="k",
        )

        result = await tool.execute(url="https://example.gov")

        assert result.success is False
        assert "field_schema" in (result.error or "")

    async def test_invalid_field_schema_type_name_is_an_error(self):
        recipe_collection, extraction_collection = _collections()
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={"nodriver": _FakeDriver(_SINGLE_HTML)},
            api_key="k",
        )

        result = await tool.execute(url="https://example.gov", field_schema={"employer": "not_a_real_type"})

        assert result.success is False
        assert "employer" in (result.error or "")

    async def test_unsupported_driver_backend_is_an_error(self):
        recipe_collection, extraction_collection = _collections()
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={"nodriver": _FakeDriver(_SINGLE_HTML)},
            api_key="k",
        )

        result = await tool.execute(
            url="https://example.gov", field_schema={"employer": "str"}, driver_backend="not_a_real_backend"
        )

        assert result.success is False
        assert "not_a_real_backend" in (result.error or "")

    async def test_render_failure_is_reported_not_raised(self):
        recipe_collection, extraction_collection = _collections()
        driver = _FakeDriver(_SINGLE_HTML, raise_exc=RuntimeError("connection refused"))
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={"nodriver": driver},
            api_key="k",
        )

        result = await tool.execute(url="https://example.gov", field_schema={"employer": "str"})

        assert result.success is False
        assert "connection refused" in (result.error or "")

    async def test_invalid_nav_steps_is_an_error(self):
        recipe_collection, extraction_collection = _collections()
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={"nodriver": _FakeDriver(_SINGLE_HTML)},
            api_key="k",
        )

        result = await tool.execute(
            url="https://example.gov",
            field_schema={"employer": "str"},
            nav_steps=[{"action": "click", "css_selector": "#x"}],
        )

        assert result.success is False
        assert "nav_steps" in (result.error or "")

    async def test_single_record_extraction_via_seeded_recipe(self):
        recipe_collection, extraction_collection = _collections()
        target_id = _derive_target_id("https://example.gov/warn", {"employer": "str", "affected_count": "int"})
        # single-record recipes wrap their strategy in a {"selectors": ...}
        # envelope (see eval_loop._reuse_recipe/_regenerate_recipe) -- unlike
        # multi-row recipes, which store the strategy dict directly.
        await _seed_recipe(recipe_collection, target_id, {"selectors": _SINGLE_STRATEGY})
        driver = _FakeDriver(_SINGLE_HTML)
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={"nodriver": driver},
            api_key="k",
        )

        result = await tool.execute(
            url="https://example.gov/warn", field_schema={"employer": "str", "affected_count": "int"}
        )

        assert result.success is True
        assert result.metadata["validation_status"] == "validated"
        assert result.metadata["target_id"] == target_id
        records = json.loads(result.content)["records"]
        assert records == [{"employer": "Acme Corp", "affected_count": 42}]

    async def test_multi_row_extraction_via_seeded_recipe(self):
        recipe_collection, extraction_collection = _collections()
        target_id = _derive_target_id("https://example.gov/warn", {"employer": "str", "affected_count": "int"})
        await _seed_recipe(recipe_collection, target_id, _ROW_STRATEGY)
        driver = _FakeDriver(_ROWS_HTML)
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={"nodriver": driver},
            api_key="k",
        )

        result = await tool.execute(
            url="https://example.gov/warn",
            field_schema={"employer": "str", "affected_count": "int"},
            multi_row=True,
        )

        assert result.success is True
        assert result.metadata["record_count"] == 2

    async def test_wait_for_and_nav_steps_are_forwarded_to_the_driver(self):
        recipe_collection, extraction_collection = _collections()
        target_id = _derive_target_id("https://example.gov/warn", {"employer": "str", "affected_count": "int"})
        await _seed_recipe(recipe_collection, target_id, {"selectors": _SINGLE_STRATEGY})
        driver = _FakeDriver(_SINGLE_HTML)
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={"nodriver": driver},
            api_key="k",
        )

        await tool.execute(
            url="https://example.gov/warn",
            field_schema={"employer": "str", "affected_count": "int"},
            wait_for=".content",
            nav_steps=[{"action": "click", "selector": "#search"}],
        )

        assert driver.wait_for_calls == [".content"]
        assert driver.nav_steps_calls == [[NavStep(action="click", selector="#search")]]

    async def test_explicit_target_id_is_used_verbatim(self):
        recipe_collection, extraction_collection = _collections()
        await _seed_recipe(recipe_collection, "my_custom_id", {"selectors": _SINGLE_STRATEGY})
        driver = _FakeDriver(_SINGLE_HTML)
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={"nodriver": driver},
            api_key="k",
        )

        result = await tool.execute(
            url="https://example.gov/warn",
            field_schema={"employer": "str", "affected_count": "int"},
            target_id="my_custom_id",
        )

        assert result.metadata["target_id"] == "my_custom_id"

    async def test_repeated_call_reuses_the_recipe_with_no_new_candidate_generation(self):
        """The self-healing recipe-reuse contract: a second identical call
        against a target with a healthy recipe never regenerates candidates
        -- proven here via the seeded-recipe path producing consistent
        output across two calls with the SAME derived target_id."""
        recipe_collection, extraction_collection = _collections()
        target_id = _derive_target_id("https://example.gov/warn", {"employer": "str", "affected_count": "int"})
        await _seed_recipe(recipe_collection, target_id, {"selectors": _SINGLE_STRATEGY})
        driver = _FakeDriver(_SINGLE_HTML)
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={"nodriver": driver},
            api_key="k",
        )

        first = await tool.execute(
            url="https://example.gov/warn", field_schema={"employer": "str", "affected_count": "int"}
        )
        second = await tool.execute(
            url="https://example.gov/warn", field_schema={"employer": "str", "affected_count": "int"}
        )

        assert first.metadata["target_id"] == second.metadata["target_id"] == target_id
        assert first.content == second.content
        recipe = await recipe_collection.get(target_id)
        assert recipe is not None
        assert recipe.consecutive_validation_failures == 0
