"""Unit tests for threetears.scrape.tool.ScrapeTool -- the ad-hoc MCP scrape tool.

All LLM/driver calls are mocked/fake; the real sidecar + real LLM proof
lives in a live script exercised manually against the running sidecar, and
in the consuming application's own live driver+LLM proof for the eval loop
this tool is a thin wrapper over.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from _pacer_fakes import _FakeDelayPacer
from threetears.models.circuit_breaker import CircuitBreaker, CircuitState
from threetears.scrape.challenge import PageVerdict
from threetears.scrape.circuit import BackoffPolicy, TargetCircuit
from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
from threetears.scrape.health import ScrapeTargetHealthCollection
from threetears.scrape.robots import RobotsGate
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
    def __init__(
        self,
        html: str,
        final_url: str = "https://example.gov/warn",
        # BaseException, not Exception: a cancelled render is the case the driver's own
        # handler deliberately does not catch, so a fake that cannot raise one cannot
        # exercise it.
        raise_exc: BaseException | None = None,
        status: int = 200,
    ):
        self._html = html
        self._final_url = final_url
        self._raise_exc = raise_exc
        self._status = status
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
        # Accepted and ignored, like every other optional on this protocol. Fake-parity
        # enforcement only requires the params production REQUIRES, so an optional added to
        # ScrapeDriver does not fail the gate -- it fails at the first test that passes it.
        # ScrapeDriver's contract is "accept the full signature, use what you need", so a
        # stand-in that does not is not standing in for it.
        session_state: dict[str, object] | None = None,
    ) -> RenderedPage:
        self.render_calls.append(url)
        self.wait_for_calls.append(wait_for)
        self.nav_steps_calls.append(nav_steps)
        if self._raise_exc is not None:
            raise self._raise_exc
        return RenderedPage(html=self._html, status=self._status, final_url=self._final_url, timing_ms=1.0)


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
        # envelope -- unlike multi-row recipes, which store the strategy dict
        # directly. Both shapes declare that wrapping on their own
        # eval_loop._StrategyShape (as_strategy / from_strategy).
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


class TestScrapeToolFetchHealth:
    """The tool is this repo's only in-tree consumer of the eval loop.

    A parameter the entry point accepts but nothing in the repo ever passes is plumbing, not
    a feature, and it rots without anything noticing. These two tests are what make the
    classification path reachable from a real caller rather than only from a test that calls
    the eval loop directly.
    """

    async def test_a_wall_keeps_the_recipe_when_a_health_collection_is_supplied(self):
        recipe_collection, extraction_collection = _collections()
        health_collection = ScrapeTargetHealthCollection(get_registry(), get_config(), nats_client=None)
        target_id = _derive_target_id("https://example.gov/walled", {"employer": "str"})
        await _seed_recipe(recipe_collection, target_id, {"selectors": _SINGLE_STRATEGY})
        driver = _FakeDriver("<html><body><h1>Checking your browser</h1></body></html>", status=503)
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            health_collection=health_collection,
            drivers={"nodriver": driver},
            api_key="k",
        )
        verdict = PageVerdict(
            kind="blocked", evidence="the page asks the visitor to verify a browser", confidence="high"
        )
        seen: list[str] = []

        def _capture(*_args, **_kwargs):
            def _with_structured_output(_schema, **_kw):
                async def _ainvoke(prompt):
                    seen.append(prompt)
                    return verdict

                return SimpleNamespace(ainvoke=_ainvoke)

            return SimpleNamespace(with_structured_output=_with_structured_output)

        with patch("threetears.scrape.llm_retry.create_chat_model", side_effect=_capture):
            result = await tool.execute(url="https://example.gov/walled", field_schema={"employer": "str"})

        assert json.loads(result.content)["validation_status"] == "blocked"
        assert not result.success, "a walled fetch produced no records and must not report success"
        # A caller that cannot tell a wall from a broken extraction will retry forever and
        # count it against the target. `error` is the field a failed ToolResult is read
        # through, so the distinction has to survive there rather than only in metadata.
        assert result.error is not None
        assert "blocked" in result.error
        assert "not implicated" in result.error
        recipe = await recipe_collection.get(target_id)
        assert recipe is not None
        assert recipe.consecutive_validation_failures == 0, "the tool let a wall count against the recipe"
        assert seen, "the tool never reached the classifier"
        assert "HTTP status 503" in seen[0], "the tool held the status and did not pass it to the classifier"

    async def test_without_a_health_collection_the_tool_behaves_exactly_as_before(self):
        """The default, and every pre-existing caller. No classification, no model call at all."""
        recipe_collection, extraction_collection = _collections()
        target_id = _derive_target_id("https://example.gov/unwatched", {"employer": "str"})
        await _seed_recipe(recipe_collection, target_id, {"selectors": _SINGLE_STRATEGY})
        driver = _FakeDriver("<html><body><h1>Checking your browser</h1></body></html>", status=503)
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={"nodriver": driver},
            api_key="k",
        )

        with patch("threetears.scrape.llm_retry.create_chat_model") as create_model:
            result = await tool.execute(url="https://example.gov/unwatched", field_schema={"employer": "str"})

        create_model.assert_not_called()
        assert json.loads(result.content)["validation_status"] == "failed"
        recipe = await recipe_collection.get(target_id)
        assert recipe is not None
        assert recipe.consecutive_validation_failures == 1


# parity-with: threetears.scrape.driver.ScrapeDriver
class _WallDriver:
    """A wall that renders a fresh per-request id every time, like a real interstitial.

    This is the shape that defeats the classifier's verdict cache: the cache keys on a
    digest of the page's visible text, and a per-request id lives in exactly that text, so
    every poll looks like a page nobody has ever classified.
    """

    def __init__(self) -> None:
        self.render_calls = 0

    @property
    def name(self) -> str:
        return "wall"

    async def render(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        wait_for: str | None = None,
        capture_network: bool = False,
        nav_steps: list[NavStep] | None = None,
        session_state: dict[str, object] | None = None,
    ) -> RenderedPage:
        del timeout, wait_for, capture_network, nav_steps
        self.render_calls += 1
        return RenderedPage(
            html=(
                f"<html><body><h1>Checking your browser</h1><p>Ray ID: 8f2c{self.render_calls:08d}</p></body></html>"
            ),
            status=503,
            final_url=url,
            timing_ms=1.0,
        )


class TestScrapeToolFetchCircuit:
    """The acceptance criteria for the backoff: BOTH rates decay, and they are not one rate.

    The fetch rate is the obvious one. The classification rate is the one that does not
    follow from it, because classification has its own cache and that cache provably misses
    on a page carrying a per-request id -- which is what a real interstitial is. Only the
    fetch never happening bounds it, so both are counted here, separately, over many polls.
    """

    @staticmethod
    def _tool(driver, circuit, recipe_collection, extraction_collection, health_collection):
        return ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            health_collection=health_collection,
            circuit=circuit,
            drivers={"nodriver": driver},
            api_key="k",
        )

    async def test_a_repeatedly_blocked_target_stops_being_fetched_and_stops_being_classified(self):
        recipe_collection, extraction_collection = _collections()
        health_collection = ScrapeTargetHealthCollection(get_registry(), get_config(), nats_client=None)
        url = "https://example.gov/decay"
        schema = {"employer": "str"}
        target_id = _derive_target_id(url, schema)
        await _seed_recipe(recipe_collection, target_id, {"selectors": _SINGLE_STRATEGY})

        driver = _WallDriver()
        circuit = TargetCircuit(
            health_collection,
            policy=BackoffPolicy(failure_threshold=3, base_delay_seconds=900.0),
        )
        tool = self._tool(driver, circuit, recipe_collection, extraction_collection, health_collection)

        verdict = PageVerdict(
            kind="blocked", evidence="the page asks the visitor to verify a browser", confidence="high"
        )
        classifications: list[str] = []

        def _capture(*_args, **_kwargs):
            def _with_structured_output(_schema, **_kw):
                async def _ainvoke(prompt):
                    classifications.append(prompt)
                    return verdict

                return SimpleNamespace(ainvoke=_ainvoke)

            return SimpleNamespace(with_structured_output=_with_structured_output)

        with patch("threetears.scrape.llm_retry.create_chat_model", side_effect=_capture):
            results = [await tool.execute(url=url, field_schema=schema) for _ in range(20)]

        assert driver.render_calls == 3, "the circuit did not stop the fetch of a repeatedly walled target"
        assert len(classifications) == 3, (
            "the classification rate did not decay with the fetch rate -- a page carrying a "
            "per-request id misses the verdict cache on every poll, so only the suppressed "
            "fetch can bound it"
        )
        assert all(not r.success for r in results)
        assert "backing off" in (results[-1].error or "")
        assert results[-1].metadata["circuit_state"] == "open"
        assert results[-1].metadata["retry_after_seconds"] > 0

        row = await health_collection.get(target_id)
        assert row is not None
        assert row.circuit_state == "open"
        recipe = await recipe_collection.get(target_id)
        assert recipe is not None
        assert recipe.consecutive_validation_failures == 0, "a wall counted against the recipe"

    async def test_a_suppressed_poll_persists_no_extraction(self):
        """Backing off harder must not write more rows than backing off less.

        A suppressed poll made no observation. Persisting one anyway would mean the emptier
        the result the busier the table, and would give an operator counting extractions a
        row saying "blocked" for a fetch that never happened.
        """
        recipe_collection, extraction_collection = _collections()
        health_collection = ScrapeTargetHealthCollection(get_registry(), get_config(), nats_client=None)
        url = "https://example.gov/quiet"
        schema = {"employer": "str"}
        target_id = _derive_target_id(url, schema)
        circuit = TargetCircuit(health_collection, policy=BackoffPolicy(failure_threshold=1))
        await circuit.record_blocked(target_id)

        driver = _WallDriver()
        tool = self._tool(driver, circuit, recipe_collection, extraction_collection, health_collection)
        with patch("threetears.scrape.llm_retry.create_chat_model") as create_model:
            result = await tool.execute(url=url, field_schema=schema)

        create_model.assert_not_called()
        assert driver.render_calls == 0
        # "backoff" even though this particular circuit was opened by a real wall. The status
        # describes THIS poll, and this poll observed nothing -- it is the same reason no
        # extraction row is written below. Whether the target was last walled is a historical
        # question, and `last_blocked_at` on the health row is where it is answered.
        assert json.loads(result.content)["validation_status"] == "backoff"
        assert await extraction_collection.get(target_id) is None

    async def test_a_target_that_comes_back_clears_its_circuit(self):
        """One good fetch is enough. A recovered target must not stay half-suppressed."""
        recipe_collection, extraction_collection = _collections()
        health_collection = ScrapeTargetHealthCollection(get_registry(), get_config(), nats_client=None)
        url = "https://example.gov/recovers"
        schema = {"employer": "str", "affected_count": "int"}
        target_id = _derive_target_id(url, schema)
        await _seed_recipe(recipe_collection, target_id, {"selectors": _SINGLE_STRATEGY})
        circuit = TargetCircuit(health_collection, policy=BackoffPolicy(failure_threshold=5))
        await circuit.record_blocked(target_id)
        await circuit.record_blocked(target_id)

        driver = _FakeDriver(_SINGLE_HTML)
        tool = self._tool(driver, circuit, recipe_collection, extraction_collection, health_collection)
        result = await tool.execute(url=url, field_schema=schema)

        assert result.success
        row = await health_collection.get(target_id)
        assert row is not None
        assert row.circuit_state == "closed"
        assert row.consecutive_fetch_failures == 0
        assert row.blocked_until is None

    async def test_a_render_that_never_returns_a_page_counts_as_a_fetch_failure(self):
        """A target that stops responding backs off too; it is just not recorded as walled."""
        recipe_collection, extraction_collection = _collections()
        health_collection = ScrapeTargetHealthCollection(get_registry(), get_config(), nats_client=None)
        url = "https://example.gov/gone"
        schema = {"employer": "str"}
        target_id = _derive_target_id(url, schema)
        circuit = TargetCircuit(health_collection, policy=BackoffPolicy(failure_threshold=2))
        driver = _FakeDriver("", raise_exc=RuntimeError("connection refused"))
        tool = self._tool(driver, circuit, recipe_collection, extraction_collection, health_collection)

        await tool.execute(url=url, field_schema=schema)
        await tool.execute(url=url, field_schema=schema)
        third = await tool.execute(url=url, field_schema=schema)

        assert "backing off" in (third.error or "")
        row = await health_collection.get(target_id)
        assert row is not None
        assert row.circuit_state == "open"
        assert row.last_blocked_at is None, "a transport failure was recorded as a bot wall"

        # The machine-readable half of the same claim. `error` is prose for whoever reads it;
        # `content` is the payload a consumer parses, and in this package "blocked" means a
        # bot wall stood there. This host simply stopped answering, and no fetch happened at
        # all, so reporting "blocked" would have a consumer chasing a challenge page that
        # does not exist.
        assert json.loads(third.content)["validation_status"] == "backoff", (
            "a suppressed poll on a circuit opened by transport failures reported the target as walled"
        )
        assert third.metadata["validation_status"] == "backoff"

    async def test_an_eval_loop_that_raises_does_not_strand_the_targets_probe(self):
        """The permitted path's version of the stranding the suppressed path already handles.

        A permitted decision can promote the in-process breaker to HALF_OPEN and mark its
        probe in flight; only an outcome clears that flag. The eval loop raising -- an L3
        write failing inside ``save_entity``, say -- means no outcome is ever reported, so
        the flag is held for the life of the process. Every later ``check`` then fast-fails
        on the in-process branch before the durable row is read and answers "retry in about
        0s", so the tool tells its caller to hammer a target it will never actually fetch.

        Uses the real ``CircuitBreaker`` because the in-flight flag IS the behaviour under
        test, and drives it through ``tool.execute`` because the gap being closed is the
        tool's missing handler, not the circuit's.
        """
        recipe_collection, extraction_collection = _collections()
        health_collection = ScrapeTargetHealthCollection(get_registry(), get_config(), nats_client=None)
        url = "https://example.gov/raises"
        schema = {"employer": "str"}
        target_id = _derive_target_id(url, schema)
        await _seed_recipe(recipe_collection, target_id, {"selectors": _SINGLE_STRATEGY})

        breaker = CircuitBreaker(target_id, failure_threshold=1, recovery_timeout_seconds=0.0)
        breaker.record_failure()
        assert breaker.state is CircuitState.OPEN

        circuit = TargetCircuit(health_collection, breaker_for=lambda _target: breaker)
        tool = self._tool(
            _FakeDriver(_SINGLE_HTML), circuit, recipe_collection, extraction_collection, health_collection
        )

        with (
            patch("threetears.scrape.tool.run_eval_loop", side_effect=RuntimeError("L3 write failed")),
            pytest.raises(RuntimeError, match="L3 write failed"),
        ):
            await tool.execute(url=url, field_schema=schema)

        # The breaker promoted itself on the way in and its probe was abandoned; the tool is
        # the only thing that knows the probe is dead, so the tool has to say so.
        assert breaker.state is not CircuitState.HALF_OPEN, (
            "the in-process breaker was left holding a probe no outcome will ever clear"
        )

        decision = await circuit.check(target_id)
        assert decision.permitted, (
            "a stranded probe fast-failed the target before the durable row was read, so the "
            "caller is told to retry immediately into a circuit that cannot admit it"
        )

    async def test_a_cancelled_render_does_not_strand_the_targets_probe(self):
        """The same strand, in the longest await of the function, via the one exception class
        the driver's own handler does not catch.

        ``render`` is guarded by ``except Exception``, which a ``CancelledError`` is not, so
        a poll cancelled during the fetch propagates with no outcome recorded and no probe
        released. Cancellation is where this most often lands, because that await is where
        the time goes. It must not be recorded as a DURABLE fetch outcome either: a shutdown
        is not evidence about the target, and persisting it as one would back the target off
        for something it did not do, across every pod and past the process that was
        cancelled.

        The in-process breaker does take a failure, because ``CircuitBreakerLike`` has no
        "never mind" and reporting one is the only way to clear an admitted probe. That is
        the cheap half to be wrong about: seconds-scale recovery, process-local, and it dies
        with the process being cancelled. The health row is the expensive half, and is what
        this asserts stays untouched.
        """
        recipe_collection, extraction_collection = _collections()
        health_collection = ScrapeTargetHealthCollection(get_registry(), get_config(), nats_client=None)
        url = "https://example.gov/cancelled"
        schema = {"employer": "str"}
        target_id = _derive_target_id(url, schema)

        breaker = CircuitBreaker(target_id, failure_threshold=1, recovery_timeout_seconds=0.0)
        breaker.record_failure()
        circuit = TargetCircuit(health_collection, breaker_for=lambda _target: breaker)
        driver = _FakeDriver("", raise_exc=asyncio.CancelledError())
        tool = self._tool(driver, circuit, recipe_collection, extraction_collection, health_collection)

        with pytest.raises(asyncio.CancelledError):
            await tool.execute(url=url, field_schema=schema)

        assert breaker.state is not CircuitState.HALF_OPEN, (
            "a cancelled fetch left the in-process breaker holding a probe no outcome clears"
        )
        row = await health_collection.get(target_id)
        assert row is None or row.consecutive_fetch_failures == 0, (
            "a cancellation was persisted as a fetch failure, so a shutdown backs the target "
            "off across every pod and outlives the process that was cancelled"
        )
        assert row is None or row.circuit_state == "closed"

    async def test_a_cancellation_while_recording_the_failure_still_releases_the_probe(self):
        """A cancellation inside the failure REPORT, and the third strand in this family.

        A render that fails with an ordinary exception is handled by reporting the fetch
        failure, and ``record_unreachable`` clears the probe as its first act. But it awaits, so
        a cancellation can land inside it before that happens -- escaping the ``except
        Exception`` handler it is running in. `_render_once` returns its error rather than
        raising it precisely so this path stays inside the single guard: a failure in the
        recovery propagates out of the call, and the one handler releases the probe.
        """
        recipe_collection, extraction_collection = _collections()
        health_collection = ScrapeTargetHealthCollection(get_registry(), get_config(), nats_client=None)
        url = "https://example.gov/cancelled-mid-report"
        schema = {"employer": "str"}
        target_id = _derive_target_id(url, schema)

        breaker = CircuitBreaker(target_id, failure_threshold=1, recovery_timeout_seconds=0.0)
        breaker.record_failure()
        circuit = TargetCircuit(health_collection, breaker_for=lambda _target: breaker)
        driver = _FakeDriver("", raise_exc=RuntimeError("connection refused"))
        tool = self._tool(driver, circuit, recipe_collection, extraction_collection, health_collection)

        with (
            patch.object(TargetCircuit, "record_unreachable", side_effect=asyncio.CancelledError()),
            pytest.raises(asyncio.CancelledError),
        ):
            await tool.execute(url=url, field_schema=schema)

        assert breaker.state is not CircuitState.HALF_OPEN, (
            "a cancellation inside the failure report escaped the guard and left the in-process breaker holding a probe"
        )

    @staticmethod
    def _robots_fetcher(body: str):
        async def _fetch(_url: str) -> tuple[int, str]:
            return 200, body

        return _fetch

    async def test_a_cancellation_during_the_crawl_delay_still_releases_the_probe(self):
        """The fourth strand in this family, and the first one that is the EXPECTED case.

        The circuit admits a probe, then the crawl delay is waited before the render. That
        sleep sat outside every guard: the render's own handler starts after it. So a poll
        cancelled while being polite propagates with the probe still held, and
        ``release_probe``'s docstring says the target is then fast-failed with
        ``retry_after_seconds=0.0`` for the life of the process -- a target that is behaving
        perfectly, punished for the one thing it asked us to do.

        Unlike the three before it this is not a narrow window. The tool advertises a deadline
        of ``default_timeout + 60`` while an honoured ``Crawl-delay`` is capped at 300s, so an
        executor cancelling inside the sleep is the ordinary outcome for any site polite
        enough to ask for a long one.
        """
        recipe_collection, extraction_collection = _collections()
        health_collection = ScrapeTargetHealthCollection(get_registry(), get_config(), nats_client=None)
        url = "https://example.gov/slow-and-polite"
        schema = {"employer": "str"}
        target_id = _derive_target_id(url, schema)

        breaker = CircuitBreaker(target_id, failure_threshold=1, recovery_timeout_seconds=0.0)
        breaker.record_failure()
        circuit = TargetCircuit(health_collection, breaker_for=lambda _target: breaker)
        driver = _FakeDriver(_SINGLE_HTML)
        gate = RobotsGate(fetch=self._robots_fetcher("User-agent: *\nCrawl-delay: 120\n"))
        tool = self._tool(driver, circuit, recipe_collection, extraction_collection, health_collection)
        tool._robots = gate
        gate.note_fetched(url)

        with (
            patch("asyncio.sleep", side_effect=asyncio.CancelledError()),
            pytest.raises(asyncio.CancelledError),
        ):
            await tool.execute(url=url, field_schema=schema)

        assert breaker.state is not CircuitState.HALF_OPEN, (
            "a cancellation during the crawl delay left the in-process breaker holding a "
            "probe, so this target fast-fails with retry_after=0 for the life of the process"
        )
        assert len(driver.render_calls) == 0, "the fetch never happened, so nothing was observed"
        row = await health_collection.get(target_id)
        assert row is None or row.consecutive_fetch_failures == 0, (
            "a cancellation was persisted as a fetch failure for a target that did nothing wrong"
        )

    async def test_without_a_circuit_nothing_is_suppressed(self):
        """The default, and every pre-existing caller: every call fetches, as it always did."""
        recipe_collection, extraction_collection = _collections()
        url = "https://example.gov/ungated"
        schema = {"employer": "str", "affected_count": "int"}
        await _seed_recipe(recipe_collection, _derive_target_id(url, schema), {"selectors": _SINGLE_STRATEGY})
        driver = _FakeDriver(_SINGLE_HTML)
        tool = ScrapeTool(
            recipe_collection=recipe_collection,
            extraction_collection=extraction_collection,
            drivers={"nodriver": driver},
            api_key="k",
        )
        for _ in range(3):
            assert (await tool.execute(url=url, field_schema=schema)).success
        assert len(driver.render_calls) == 3


# ---------------------------------------------------------------------------
# Egress at the tool level: which exit this tool's own requests leave by, and
# what it says when the drivers are proxied and it is not. Here rather than in
# test_robots.py because the subject is ScrapeTool's wiring; robots is only how
# one of the symptoms shows up.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver_kind", ["api", "sidecar"], ids=["api-driver", "sidecar-driver"])
def test_a_proxied_driver_with_an_unproxied_tool_says_so(caplog, driver_kind: str) -> None:
    """One security property, two wiring points, and getting only one right is invisible.

    The page leaves by TOR while the robots.txt read in front of it leaves by the container's
    own address -- both halves work, and the target learns the real address from the request
    nobody was thinking about. A warning, not a refusal: a deployment may want it, but it
    should have to be a decision.
    """
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.core.egress import ProxyEgress
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.drivers.api import ApiDriver
    from threetears.scrape.drivers.nodriver_sidecar import NodriverSidecarDriver
    from threetears.scrape.tool import ScrapeTool

    # Parametrised over both drivers that carry an exit. Covering only ApiDriver left the
    # sidecar's property untested, so deleting it would keep the suite green while the
    # deployment shape the CHANGELOG actually describes -- a browser behind TOR -- silently
    # stopped emitting this warning.
    tor = ProxyEgress("tor", "socks5://127.0.0.1:9050")
    driver = ApiDriver(egress=tor) if driver_kind == "api" else NodriverSidecarDriver("http://s:8088", egress=tor)

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    with caplog.at_level("WARNING", logger="threetears.scrape.tool"):
        ScrapeTool(
            recipe_collection=ScrapeRecipeCollection(reg, cfg, nats_client=None),
            extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
            drivers={driver_kind: driver},
            api_key="k",
        )

    assert any("container's own address" in r.message for r in caplog.records), (
        f"a split egress configuration passed silently for the {driver_kind} driver"
    )


def test_matching_egress_on_both_says_nothing(caplog) -> None:
    """The correct configuration must not be noisy, or the warning stops being read."""
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.core.egress import ProxyEgress
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.drivers.api import ApiDriver
    from threetears.scrape.tool import ScrapeTool

    tor = ProxyEgress("tor", "socks5://127.0.0.1:9050")
    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    with caplog.at_level("WARNING", logger="threetears.scrape.tool"):
        ScrapeTool(
            recipe_collection=ScrapeRecipeCollection(reg, cfg, nats_client=None),
            extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
            drivers={"api": ApiDriver(egress=tor)},
            egress=tor,
            api_key="k",
        )

    assert not any("container's own address" in r.message for r in caplog.records)


def _build_tool(caplog, **kwargs):
    """Construct a ScrapeTool with the collections these warning tests do not care about.

    Returns the WARNING records emitted during construction, which is the only thing under test
    in the four cases below.
    """
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.tool import ScrapeTool

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    with caplog.at_level("WARNING", logger="threetears.scrape.tool"):
        ScrapeTool(
            recipe_collection=ScrapeRecipeCollection(reg, cfg, nats_client=None),
            extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
            api_key="k",
            **kwargs,
        )
    return [r.message for r in caplog.records]


def test_a_proxied_tool_with_an_unproxied_driver_says_so(caplog) -> None:
    """The mirror case, and the more damaging one.

    With the gate proxied and the driver not, what leaves by the container's own address is the
    PAGE FETCH -- the request the exit was configured for -- rather than a robots.txt read. The
    check covered only the other direction, so this configuration passed in silence while the
    noisier and less consequential one was reported.
    """
    from threetears.core.egress import ProxyEgress
    from threetears.scrape.drivers.api import ApiDriver

    messages = _build_tool(
        caplog,
        drivers={"api": ApiDriver()},
        egress=ProxyEgress("tor", "socks5://127.0.0.1:9050"),
    )

    assert any("the page fetch itself goes out on the container's own address" in m for m in messages), (
        "a tool proxied in front of an unproxied driver passed silently"
    )


def test_the_backends_that_cannot_honour_an_exit_are_named(caplog) -> None:
    """The README points an operator at this warning, so the warning has to be worth pointing at.

    Most backends construct their own bare `httpx.AsyncClient` or launch a browser with no proxy
    support, and reach the target on the container's default route no matter what this tool is
    configured with. Nothing in this package can route them; what it CAN do is refuse to let the
    bypass be quiet, and say which drivers are responsible so the message is actionable rather
    than a general disclaimer.

    Named-driver assertions rather than "some warning fired": an operator reading `document,
    camoufox` knows what to change, and a message that only said "some drivers" would pass this
    test while telling them nothing.
    """
    from threetears.core.egress import SocksEgress
    from threetears.scrape.drivers.camoufox import CamoufoxDriver
    from threetears.scrape.drivers.document import DocumentDriver

    messages = _build_tool(
        caplog,
        drivers={"document": DocumentDriver(), "camoufox": CamoufoxDriver()},
        egress=SocksEgress("tor"),
    )

    assert any("camoufox, document" in m for m in messages), (
        f"the drivers that cannot honour the configured exit were not named: {messages}"
    )


def test_a_caller_supplied_proxied_gate_is_not_called_split(caplog) -> None:
    """Reading the constructor argument called a correct configuration wrong.

    Passing a `RobotsGate` built with its own egress is the documented way to control how the
    robots read leaves, and it makes `ScrapeTool(egress=...)` unnecessary. The check branched on
    that unused argument, so this pair -- proxied on both halves -- was reported as leaking on a
    request that was in fact proxied. A security warning that fires on correct configuration is
    one readers learn to filter, which costs the warnings that are true.
    """
    from threetears.core.egress import ProxyEgress
    from threetears.scrape.drivers.api import ApiDriver
    from threetears.scrape.robots import RobotsGate

    tor = ProxyEgress("tor", "socks5://127.0.0.1:9050")
    messages = _build_tool(caplog, drivers={"api": ApiDriver(egress=tor)}, robots=RobotsGate(egress=tor))

    assert not any("container's own address" in m for m in messages), (
        f"a correctly proxied gate and driver were reported as split: {messages}"
    )


def test_a_disabled_gate_has_no_split_to_report(caplog) -> None:
    """With robots off there is no second request, so there is nothing to be split about.

    The old check would warn here, describing a robots.txt read that never happens.
    """
    from threetears.core.egress import ProxyEgress
    from threetears.scrape.drivers.api import ApiDriver

    messages = _build_tool(
        caplog,
        drivers={"api": ApiDriver(egress=ProxyEgress("tor", "socks5://127.0.0.1:9050"))},
        robots=None,
    )

    assert not any("container's own address" in m for m in messages), (
        f"warned about robots.txt reads for a tool with no robots gate: {messages}"
    )


def test_a_wrapper_driver_does_not_hide_its_inner_exit(caplog) -> None:
    """A wrapper reporting the base class's `None` lands on the safe-looking side.

    `NetworkCaptureDriver` wraps a real browser driver and performs no fetch of its own, so the
    exit that matters is the inner one. Inheriting the default made a genuinely proxied sidecar
    read as unconfigured -- and because the reading is used to decide whether a configuration is
    split, the wrapper silently suppressed the warning it should have raised.
    """
    from threetears.core.egress import ProxyEgress
    from threetears.scrape.drivers.network_capture import NetworkCaptureDriver
    from threetears.scrape.drivers.nodriver_sidecar import NodriverSidecarDriver

    tor = ProxyEgress("tor", "socks5://127.0.0.1:9050")
    wrapped = NetworkCaptureDriver(NodriverSidecarDriver("http://s:8088", egress=tor))
    assert wrapped.egress is tor, "the wrapper does not report the exit its fetches actually use"

    messages = _build_tool(caplog, drivers={"capture": wrapped})

    assert any("container's own address" in m for m in messages), "a proxied driver behind a wrapper passed silently"


async def test_the_exit_a_page_came_through_reaches_the_health_row() -> None:
    """The last unasserted link in the egress round trip.

    The driver reports which exit a page came through, and the tool passes it to the circuit --
    but nothing drove a real fetch and read `last_egress` back, so setting both call sites to
    `None` left the whole suite green. Every other assertion about that column calls the
    circuit or the health layer directly, which cannot see whether the tool wires them.
    """
    from unittest.mock import patch as _patch

    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.scrape.circuit import BackoffPolicy, TargetCircuit
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
    from threetears.scrape.driver import RenderedPage
    from threetears.scrape.health import ScrapeTargetHealthCollection
    from threetears.scrape.tool import ScrapeTool, _derive_target_id

    class _ExitReportingDriver:
        """# parity-with: threetears.scrape.driver.ScrapeDriver"""

        @property
        def name(self) -> str:
            return "exit-reporting"

        async def render(self, url: str, **_kw: Any) -> RenderedPage:
            # A page that fails extraction, so the circuit records a blocked observation and
            # the exit is stamped. Extraction succeeding would take the reachable path, which
            # is asserted separately at the circuit layer.
            return RenderedPage(
                html="<html><body>nothing extractable here</body></html>",
                status=200,
                final_url=url,
                timing_ms=1.0,
                egress="tor",
            )

    url = "https://example.gov/list"
    schema = {"employer": "str", "affected_count": "int"}
    target_id = _derive_target_id(url, schema)

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    health = ScrapeTargetHealthCollection(reg, cfg, nats_client=None)
    # Seeded so the eval loop takes the REUSE path: its selectors miss this page, the verdict
    # says blocked, and the circuit records it. Without a recipe it regenerates instead, which
    # reaches candidate generation and spends its full retry budget against a fake key -- 32s
    # measured, for a path this test is not about.
    recipes = ScrapeRecipeCollection(reg, cfg, nats_client=None)
    await recipes.save_entity(
        recipes.create(
            {
                "target_id": target_id,
                "extraction_strategy": {"employer": "td:nth-child(1)", "affected_count": "td:nth-child(2)"},
                "won_at": None,
                "last_validated_at": None,
                "consecutive_validation_failures": 0,
            }
        )
    )
    tool = ScrapeTool(
        recipe_collection=recipes,
        extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
        health_collection=health,
        circuit=TargetCircuit(health, policy=BackoffPolicy(failure_threshold=1), egress_name="container-default"),
        drivers={"nodriver": _ExitReportingDriver()},
        robots=None,
        api_key="k",
    )

    # A BLOCKED verdict, so the circuit records a failure and writes the row. A `None` verdict
    # takes the reachable path, which writes nothing at all on a healthy circuit -- so the
    # first version of this asserted against a row that never existed.
    from threetears.scrape.challenge import PageVerdict

    verdict = PageVerdict(kind="blocked", evidence="a wall stood here", confidence="high")
    with _patch("threetears.scrape.eval_loop.classify_failed_page", return_value=verdict):
        await tool.execute(url=url, field_schema=schema)

    row = await health.get(target_id)
    assert row is not None
    assert row.last_egress == "tor", (
        "the tool recorded the circuit's configured exit rather than the one the page came through"
    )


async def test_a_driver_predating_session_state_still_works() -> None:
    """`ScrapeDriver` ships on PyPI as a pluggable contract, so out-of-tree drivers exist.

    One written against 0.19.x has no `session_state` parameter. Passing it unconditionally
    made EVERY fetch through such a driver raise TypeError -- including the overwhelming
    majority that carry no stored solve at all. The egress half of the same change reasoned
    about precisely this consumer and reads its attribute through a getattr; this half did not,
    which is what makes it an oversight rather than a judgement.
    """
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection

    class _PreSessionStateDriver:
        """A driver as it would have been written before this bundle. Deliberately NOT
        declaring session_state, which is the whole point."""

        @property
        def name(self) -> str:
            return "old"

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
            return RenderedPage(html=_SINGLE_HTML, status=200, final_url=url, timing_ms=1.0)

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    recipes = ScrapeRecipeCollection(reg, cfg, nats_client=None)
    url, schema = "https://example.gov/old-driver", {"employer": "str"}
    await _seed_recipe(recipes, _derive_target_id(url, schema), {"selectors": _SINGLE_STRATEGY})

    tool = ScrapeTool(
        recipe_collection=recipes,
        extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
        drivers={"nodriver": _PreSessionStateDriver()},  # type: ignore[dict-item]
        api_key="k",
        robots=None,
    )

    result = await tool.execute(url=url, field_schema=schema)

    assert result.success, f"a pre-existing driver broke on an ordinary fetch: {result.error}"


async def test_a_driver_that_cannot_take_a_solve_is_not_backed_off() -> None:
    """A configuration error must not be answered with a timer.

    The sibling above covers the common case -- no stored solve, so nothing is passed and the
    old driver works. This is the other one: a solve EXISTS and the driver cannot take it. That
    is a static incompatibility, identical on every poll and unfixable by waiting, and it used
    to reach `record_unreachable` through the broad guard -- so `failure_threshold` polls later
    the durable circuit opened and suppressed the target for fifteen minutes escalating to six
    hours, while also stamping circuit state that `list_walled` and the operator queue then
    reason about.

    Both halves are asserted. The error alone would pass against a version that reported the
    problem AND still backed the target off, which is the actual defect.
    """
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig
    from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection

    class _PreSessionStateDriver:
        """# parity-with: threetears.scrape.driver.ScrapeDriver"""

        @property
        def name(self) -> str:
            return "old"

        async def render(
            self,
            url: str,
            *,
            timeout: float = 30.0,
            wait_for: str | None = None,
            capture_network: bool = False,
            nav_steps: list[NavStep] | None = None,
        ) -> RenderedPage:
            raise AssertionError("the render must not be attempted once the mismatch is known")

    class _SpyCircuit:
        """# parity-with: threetears.scrape.circuit.TargetCircuit"""

        def __init__(self) -> None:
            self.unreachable: list[str] = []

        async def record_unreachable(self, target_id: str) -> None:
            self.unreachable.append(target_id)

    reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
    circuit = _SpyCircuit()
    tool = ScrapeTool(
        recipe_collection=ScrapeRecipeCollection(reg, cfg, nats_client=None),
        extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
        drivers={"old": _PreSessionStateDriver()},  # type: ignore[dict-item]
        circuit=circuit,  # type: ignore[arg-type]
        api_key="k",
        robots=None,
    )

    page, error = await tool._render_once(
        _PreSessionStateDriver(),  # type: ignore[arg-type]
        "https://example.gov/x",
        wait_for=None,
        nav_steps=None,
        solved_state={"cookies": {"session": "abc"}},
        target_id="t1",
        driver_backend="old",
    )

    assert page is None
    assert "does not accept session_state" in (error or ""), (
        f"the incompatibility was reported as something else: {error}"
    )
    assert circuit.unreachable == [], (
        "a driver that cannot take a solve was recorded as an unreachable fetch, so the durable "
        "circuit will back the target off for hours over a wiring mistake time cannot fix"
    )


def test_a_driver_taking_kwargs_is_not_called_incompatible() -> None:
    """A forwarding wrapper can take the solve, and refusing it would break a driver that works.

    Asserted alongside the negative case so the helper is not merely rejecting everything it
    does not recognise.
    """
    from threetears.scrape.tool import _accepts_session_state

    class _Forwards:
        """# parity-with: threetears.scrape.driver.ScrapeDriver"""

        @property
        def name(self) -> str:
            return "forwards"

        async def render(self, url: str, **kwargs: Any) -> RenderedPage:
            raise NotImplementedError

    class _Declines:
        """# parity-with: threetears.scrape.driver.ScrapeDriver"""

        @property
        def name(self) -> str:
            return "declines"

        async def render(self, url: str, *, timeout: float = 30.0) -> RenderedPage:
            raise NotImplementedError

    assert _accepts_session_state(_Forwards())  # type: ignore[arg-type]
    assert not _accepts_session_state(_Declines())  # type: ignore[arg-type]


class TestEgressByName:
    """The registry's reason to exist: configuration names an exit, nothing branches on it."""

    def test_a_name_resolves_through_the_registry(self) -> None:
        from threetears.core.collections.registry import CollectionRegistry
        from threetears.core.config import DefaultCoreConfig
        from threetears.core.egress import EgressRegistry, SocksEgress
        from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection

        reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
        tool = ScrapeTool(
            recipe_collection=ScrapeRecipeCollection(reg, cfg, nats_client=None),
            extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
            drivers={},
            api_key="k",
            egress="tor",
            egress_registry=EgressRegistry({"tor": SocksEgress("tor")}),
            robots=None,
        )

        assert tool._egress is not None
        assert tool._egress.name == "tor"

    def test_an_unknown_name_raises_rather_than_quietly_going_direct(self) -> None:
        """The failure this forbids is silent and total: a deployment that asked for TOR,
        got the container's own address, and was told nothing by anything."""
        from threetears.core.collections.registry import CollectionRegistry
        from threetears.core.config import DefaultCoreConfig
        from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection

        reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
        with pytest.raises(KeyError, match="no egress driver named"):
            ScrapeTool(
                recipe_collection=ScrapeRecipeCollection(reg, cfg, nats_client=None),
                extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
                drivers={},
                api_key="k",
                egress="tor",
                robots=None,
            )


class TestTheFleetAndTheSiteBothBind:
    """The `max(local, fleet)` lives in `ScrapeTool`, so it is asserted here.

    The gate's own tests deliberately assert each half separately. Writing `max(...)` in a test
    body moves the combination into the test -- it then computes what production is supposed to
    compute, and deleting the `max` in the tool leaves the suite green. That is exactly what
    happened, and this class is the fix.
    """

    @staticmethod
    def _pacer(*, claimed: bool, retry_after: float = 0.0) -> _FakeDelayPacer:
        return _FakeDelayPacer(claimed=claimed, retry_after_seconds=retry_after)

    async def _slept_waiting_for(self, pacer) -> list[float]:
        from threetears.core.collections.registry import CollectionRegistry
        from threetears.core.config import DefaultCoreConfig
        from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection

        async def _fetch(_url: str) -> tuple[int, str]:
            return 200, "User-agent: *\nCrawl-delay: 10\n"

        reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
        recipes = ScrapeRecipeCollection(reg, cfg, nats_client=None)
        url, schema = "https://example.gov/paced", {"employer": "str"}
        await _seed_recipe(recipes, _derive_target_id(url, schema), {"selectors": _SINGLE_STRATEGY})

        gate = RobotsGate(fetch=_fetch, delay_pacer=pacer)
        gate.note_fetched(url)  # this pod owes the site ~10s
        tool = ScrapeTool(
            recipe_collection=recipes,
            extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
            drivers={"nodriver": _FakeDriver(_SINGLE_HTML)},
            api_key="k",
            robots=gate,
        )

        slept: list[float] = []

        async def _record(seconds: float) -> None:
            slept.append(seconds)

        with patch("asyncio.sleep", _record):
            await tool.execute(url=url, field_schema=schema)
        return slept

    async def test_the_fleets_longer_wait_wins(self):
        """Deleting the `max` silently discards the fleet's answer."""
        slept = await self._slept_waiting_for(self._pacer(claimed=False, retry_after=25.0))

        assert slept, "nothing was waited"
        assert slept[0] == pytest.approx(25.0, abs=0.5), (
            f"the fleet owed 25s and this pod owed ~10s; the tool waited {slept[0]}s"
        )

    async def test_a_suppressed_poll_takes_no_turn_at_all(self):
        """The `and fetch_will_happen` guard, asserted where it lives.

        The gate-level version of this drives `RobotsGate` directly, and its own docstring says
        the scenario is a `ScrapeTool` one: robots is consulted BEFORE the circuit, so without
        this guard a walled target inside its backoff claims a token on every poll and delays
        every sibling target on the origin. Delete the guard and only a tool-level test that
        builds a real suppressing circuit can notice.
        """
        from threetears.core.collections.registry import CollectionRegistry
        from threetears.core.config import DefaultCoreConfig
        from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection

        async def _fetch(_url: str) -> tuple[int, str]:
            return 200, "User-agent: *\nCrawl-delay: 10\n"

        reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
        health = ScrapeTargetHealthCollection(get_registry(), get_config(), nats_client=None)
        url, schema = "https://example.gov/walled", {"employer": "str"}
        target_id = _derive_target_id(url, schema)

        circuit = TargetCircuit(health, policy=BackoffPolicy(failure_threshold=1))
        await circuit.record_blocked(target_id)

        pacer = self._pacer(claimed=True)
        tool = ScrapeTool(
            recipe_collection=ScrapeRecipeCollection(reg, cfg, nats_client=None),
            extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
            drivers={"nodriver": _FakeDriver(_SINGLE_HTML)},
            api_key="k",
            health_collection=health,
            circuit=circuit,
            robots=RobotsGate(fetch=_fetch, delay_pacer=pacer),
        )

        # A delay is already owed, so the OTHER `fetch_will_happen` guard -- the one on the
        # sleep -- is exercised too. Without it a suppressed poll sleeps out the site's crawl
        # delay before being told it will not fetch: a caller blocked for up to the delay
        # ceiling to be handed a backoff result. The previous version never called
        # `note_fetched`, so the wait was zero either way and that guard was deletable green.
        tool._robots.note_fetched(url)
        slept: list[float] = []

        async def _record(seconds: float) -> None:
            slept.append(seconds)

        with patch("asyncio.sleep", _record):
            result = await tool.execute(url=url, field_schema=schema)

        assert result.success is False, "the circuit should have suppressed this poll"
        assert slept == [], f"a suppressed poll waited {slept} before declining to fetch"
        assert pacer.keys == [], (
            "a suppressed poll spent the origin's shared token, so one walled target delays every sibling on that site"
        )

    async def test_the_sites_longer_wait_wins(self):
        """And a granted fleet turn must not shorten what the site itself asked for."""
        slept = await self._slept_waiting_for(self._pacer(claimed=True))

        assert slept, "nothing was waited"
        assert slept[0] == pytest.approx(10.0, abs=0.5), (
            f"the site asked 10s and the fleet owed nothing; the tool waited {slept[0]}s"
        )


class TestACancelledPollReturnsItsTurn:
    """SCR-6QF2: the leak the single guard in `execute` finally gave a home to.

    `claim_fleet_turn` consumes, and the sleep straight after it is the point this file's own
    comments call the EXPECTED cancellation site -- the tool advertises a deadline an honoured
    Crawl-delay can approach. So the ordinary case held the origin's shared budget down for a
    fetch that never happened.
    """

    async def test_a_cancellation_after_the_claim_gives_the_turn_back(self):
        from threetears.core.collections.registry import CollectionRegistry
        from threetears.core.config import DefaultCoreConfig
        from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection

        async def _fetch(_url: str) -> tuple[int, str]:
            return 200, "User-agent: *\nCrawl-delay: 10\n"

        reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
        recipes = ScrapeRecipeCollection(reg, cfg, nats_client=None)
        url, schema = "https://example.gov/cancelled-turn", {"employer": "str"}
        await _seed_recipe(recipes, _derive_target_id(url, schema), {"selectors": _SINGLE_STRATEGY})

        pacer = _FakeDelayPacer()
        gate = RobotsGate(fetch=_fetch, delay_pacer=pacer)
        gate.note_fetched(url)  # a delay is owed, so the sleep below is real
        tool = ScrapeTool(
            recipe_collection=recipes,
            extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
            drivers={"nodriver": _FakeDriver(_SINGLE_HTML)},
            api_key="k",
            robots=gate,
        )

        with (
            patch("asyncio.sleep", side_effect=asyncio.CancelledError()),
            pytest.raises(asyncio.CancelledError),
        ):
            await tool.execute(url=url, field_schema=schema)

        # `fire_and_forget` schedules the refund as its own task rather than awaiting it, since
        # an await inside a cancellation handler re-raises before reaching the store. Yield once
        # so that task runs.
        await asyncio.sleep(0)

        assert pacer.keys == ["https://example.gov"], "the turn was never taken, so this asserts nothing"
        assert pacer.refunded == ["https://example.gov"], (
            "a cancelled poll kept the origin's shared turn, so a restart loop starves every other target on that site"
        )

    async def test_an_uncancelled_poll_keeps_its_turn(self):
        """The turn is spent on a fetch that happened, so returning it would be wrong."""
        from threetears.core.collections.registry import CollectionRegistry
        from threetears.core.config import DefaultCoreConfig
        from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection

        async def _fetch(_url: str) -> tuple[int, str]:
            return 200, "User-agent: *\nCrawl-delay: 10\n"

        reg, cfg = CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS")
        recipes = ScrapeRecipeCollection(reg, cfg, nats_client=None)
        url, schema = "https://example.gov/kept-turn", {"employer": "str"}
        await _seed_recipe(recipes, _derive_target_id(url, schema), {"selectors": _SINGLE_STRATEGY})

        pacer = _FakeDelayPacer()
        tool = ScrapeTool(
            recipe_collection=recipes,
            extraction_collection=ScrapeExtractionCollection(reg, cfg, nats_client=None),
            drivers={"nodriver": _FakeDriver(_SINGLE_HTML)},
            api_key="k",
            robots=RobotsGate(fetch=_fetch, delay_pacer=pacer),
        )

        with patch("asyncio.sleep", _noop_sleep):
            await tool.execute(url=url, field_schema=schema)
        await asyncio.sleep(0)

        assert pacer.keys == ["https://example.gov"]
        assert pacer.refunded == [], "a completed fetch gave back a turn it had legitimately spent"


async def _noop_sleep(_seconds: float) -> None:
    return None
