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
from unittest.mock import patch

import pytest
from threetears.models.circuit_breaker import CircuitBreaker, CircuitState
from threetears.scrape.challenge import PageVerdict
from threetears.scrape.circuit import BackoffPolicy, TargetCircuit
from threetears.scrape.collections import ScrapeExtractionCollection, ScrapeRecipeCollection
from threetears.scrape.health import ScrapeTargetHealthCollection
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
        """The narrow window between the two handlers, and the third strand in this family.

        A render that fails with an ordinary exception is handled by reporting the fetch
        failure, and ``record_unreachable`` clears the probe as its first act. But it awaits,
        so a cancellation can land inside it before that happens -- escaping the
        ``except Exception`` handler it is running in, and never reaching an
        ``except BaseException`` placed only around the render. Nesting the handlers so the
        outer one covers the recovery path is what closes it.
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
            "a cancellation inside the failure report escaped between the two handlers and "
            "left the in-process breaker holding a probe"
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
