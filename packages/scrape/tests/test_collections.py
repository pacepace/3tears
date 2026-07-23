"""Tests for ScrapeTarget / ScrapeRecipe / ScrapeExtraction collections' L3 wiring.

Covers both storage tiers: the in-memory ``self._rows`` fallback (no L3 pool
configured — the shape every other scrape unit test already exercises) and
the real ``DurableStore`` branch added so scrape's collections are
multi-pod-safe (mirrors ``faidh.db.collection.FaidhCollection``'s existing
pattern). ``FakeDurableStore`` below is a pure in-memory stand-in for the
``threetears.core.backends.protocol.DurableStore`` protocol — no real
database involved; the live, real-Postgres proof lives in
``tests/e2e/test_scrape_collections_persistence_live.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.scrape.collections import (
    ScrapeExtractionCollection,
    ScrapeRecipeCollection,
    ScrapeTarget,
    ScrapeTargetCollection,
    decode_field_schema,
    decode_nav_steps,
    encode_field_schema,
    encode_nav_steps,
)
from threetears.scrape.driver import NavStep


# parity-with: threetears.core.backends.protocol.DurableStore
class FakeDurableStore:
    """In-memory stand-in conforming to ``DurableStore`` (fetch_one/upsert/delete/scan)."""

    def __init__(self) -> None:
        self.tables: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}

    def _table(self, table: str) -> dict[tuple[Any, ...], dict[str, Any]]:
        return self.tables.setdefault(table, {})

    @staticmethod
    def _pk_tuple(pk: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(sorted(pk.items()))

    async def fetch_one(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        return self._table(table).get(self._pk_tuple(pk))

    async def upsert(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: list[str] | None = None,
        on_conflict: str = "update",
        cas: datetime | None = None,
        conn: Any = None,
    ) -> int:
        pk = pk or []
        key = self._pk_tuple({col: row[col] for col in pk})
        self._table(table)[key] = dict(row)
        return 1

    async def delete(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> None:
        self._table(table).pop(self._pk_tuple(pk), None)

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        return list(self._table(table).values())


@pytest.fixture()
def fake_store() -> FakeDurableStore:
    return FakeDurableStore()


@pytest.fixture()
def l3_registry(fake_store: FakeDurableStore) -> CollectionRegistry:
    registry = CollectionRegistry()
    registry.configure(l3_pool=fake_store)
    return registry


@pytest.fixture()
def memory_registry() -> CollectionRegistry:
    """A registry with no L3 pool configured -- forces the in-memory ``_rows`` fallback."""
    return CollectionRegistry()


@pytest.fixture()
def config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS")


# ---------------------------------------------------------------------------
# L3 (DurableStore) branch
# ---------------------------------------------------------------------------


async def test_target_save_and_get_round_trips_through_l3(
    l3_registry: CollectionRegistry, config: DefaultCoreConfig
) -> None:
    coll = ScrapeTargetCollection(l3_registry, config, nats_client=None)
    target = coll.create(
        {
            "target_id": "warn_act_md",
            "url": "https://example.gov/warn",
            "driver_backend": "nodriver",
            "rate_limit_key": "gov_default",
            "cadence": "daily",
        }
    )
    await target.save()

    fetched = await coll.get("warn_act_md")
    assert fetched is not None
    assert fetched.url == "https://example.gov/warn"
    assert fetched.cadence == "daily"
    assert fetched.multi_row is False  # SCR-6P2X: defaults False, not requested above


async def test_target_multi_row_flag_round_trips_through_l3(
    l3_registry: CollectionRegistry, config: DefaultCoreConfig
) -> None:
    coll = ScrapeTargetCollection(l3_registry, config, nats_client=None)
    target = coll.create({"target_id": "warn_act_ny", "url": "https://example.gov/ny", "multi_row": True})
    await target.save()

    fetched = await coll.get("warn_act_ny")
    assert fetched is not None
    assert fetched.multi_row is True


class TestFieldSchemaCodec:
    """Chunk 13's field_schema codec -- a closed, explicit type-name resolver,
    never eval()."""

    def test_encode_then_decode_round_trips_real_types(self) -> None:
        """The real ad-hoc-target flow: a caller building a ScrapeTarget from
        actual Python types (not already-encoded strings) must pre-encode
        before constructing the entity -- this is encode_field_schema's one
        real call site."""
        schema = {"employer": str, "affected_count": int, "ratio": float, "active": bool}

        target = ScrapeTarget(
            {
                "target_id": "adhoc_1",
                "url": "https://example.gov/one-off",
                "field_schema": encode_field_schema(schema),
            }
        )

        assert target.field_schema == schema

    def test_decode_rejects_an_unsupported_type_name(self) -> None:
        """A typo'd/unsupported type name in a target's config must fail
        loudly at load time, not silently resolve to the wrong field type."""
        with pytest.raises(ValueError, match="employer"):
            decode_field_schema({"employer": "not_a_real_type"})

    def test_decode_of_empty_or_missing_schema_is_empty_dict(self) -> None:
        assert decode_field_schema(None) == {}
        assert decode_field_schema({}) == {}


class TestNavStepsCodec:
    """Multi-step navigation capability (2026-07-14) -- unlike field_schema,
    no type-name resolution is needed since every NavStep field is already
    JSON-safe, but the round trip and load-time validation still matter."""

    def test_encode_then_decode_round_trips_real_steps(self) -> None:
        steps = [
            NavStep(action="fill", selector="#q", value="Maine"),
            NavStep(action="click", selector="#submit"),
            NavStep(action="wait_for", selector=".results"),
            NavStep(action="wait_ms", ms=500),
        ]

        target = ScrapeTarget(
            {
                "target_id": "adhoc_1",
                "url": "https://example.gov/one-off",
                "nav_steps": encode_nav_steps(steps),
            }
        )

        assert target.nav_steps == steps

    def test_decode_of_missing_nav_steps_is_none(self) -> None:
        assert decode_nav_steps(None) is None

    def test_decode_rejects_an_unsupported_field_name(self) -> None:
        """A typo'd field in a target's nav_steps config must fail loudly at
        load time, not silently drop or misinterpret the step."""
        with pytest.raises(TypeError):
            decode_nav_steps([{"action": "click", "css_selector": "#x"}])

    def test_target_with_no_nav_steps_configured_defaults_to_none(self) -> None:
        target = ScrapeTarget({"target_id": "adhoc_1", "url": "https://example.gov/one-off"})

        assert target.nav_steps is None


async def test_target_l3_write_is_visible_to_a_second_collection_instance(
    fake_store: FakeDurableStore, config: DefaultCoreConfig
) -> None:
    """Proves the multi-pod scenario: two independent collection instances sharing one L3 store."""
    reg_a = CollectionRegistry()
    reg_a.configure(l3_pool=fake_store)
    reg_b = CollectionRegistry()
    reg_b.configure(l3_pool=fake_store)

    coll_a = ScrapeTargetCollection(reg_a, config, nats_client=None)
    coll_b = ScrapeTargetCollection(reg_b, config, nats_client=None)

    target = coll_a.create({"target_id": "warn_act_ny", "url": "https://example.gov/ny"})
    await target.save()

    fetched = await coll_b.get("warn_act_ny")
    assert fetched is not None
    assert fetched.url == "https://example.gov/ny"


async def test_target_delete_removes_from_l3(l3_registry: CollectionRegistry, config: DefaultCoreConfig) -> None:
    coll = ScrapeTargetCollection(l3_registry, config, nats_client=None)
    target = coll.create({"target_id": "warn_act_wa", "url": "https://example.gov/wa"})
    await target.save()

    await coll.delete("warn_act_wa")

    assert await coll.get("warn_act_wa") is None


async def test_target_list_all_scans_l3(l3_registry: CollectionRegistry, config: DefaultCoreConfig) -> None:
    coll = ScrapeTargetCollection(l3_registry, config, nats_client=None)
    for target_id in ("warn_act_md", "warn_act_ny"):
        target = coll.create({"target_id": target_id, "url": f"https://example.gov/{target_id}"})
        await target.save()

    all_targets = await coll.list_all()

    assert {t.target_id for t in all_targets} == {"warn_act_md", "warn_act_ny"}


async def test_recipe_and_extraction_round_trip_through_l3(
    l3_registry: CollectionRegistry, config: DefaultCoreConfig
) -> None:
    recipe_coll = ScrapeRecipeCollection(l3_registry, config, nats_client=None)
    recipe = recipe_coll.create(
        {
            "target_id": "warn_act_md",
            "extraction_strategy": {"row_selector": "table tr", "field_selectors": {"employer": "td:nth-child(1)"}},
            "consecutive_validation_failures": 0,
        }
    )
    await recipe.save()
    fetched_recipe = await recipe_coll.get("warn_act_md")
    assert fetched_recipe is not None
    assert fetched_recipe.extraction_strategy["row_selector"] == "table tr"

    ext_coll = ScrapeExtractionCollection(l3_registry, config, nats_client=None)
    extraction = ext_coll.create(
        {
            "target_id": "warn_act_md",
            "source_url": "https://example.gov/warn",
            "structured_fields": {"records": [{"employer": "Acme", "employees_affected": 42}]},
        }
    )
    await extraction.save()
    fetched_extraction = await ext_coll.get(extraction.id)
    assert fetched_extraction is not None
    assert fetched_extraction.structured_fields == {"records": [{"employer": "Acme", "employees_affected": 42}]}


# ---------------------------------------------------------------------------
# In-memory fallback branch (no L3 pool configured)
# ---------------------------------------------------------------------------


async def test_target_falls_back_to_in_memory_dict_when_no_l3_configured(
    memory_registry: CollectionRegistry, config: DefaultCoreConfig
) -> None:
    coll = ScrapeTargetCollection(memory_registry, config, nats_client=None)
    target = coll.create({"target_id": "warn_act_ca", "url": "https://example.gov/ca"})
    await target.save()

    assert coll._rows["warn_act_ca"]["url"] == "https://example.gov/ca"
    fetched = await coll.get("warn_act_ca")
    assert fetched is not None
    assert fetched.url == "https://example.gov/ca"


async def test_target_in_memory_fallback_is_not_shared_across_instances(
    memory_registry: CollectionRegistry, config: DefaultCoreConfig
) -> None:
    """Documents the exact gap the L3 branch closes: two collection instances
    with no shared L3 pool do NOT see each other's writes."""
    coll_a = ScrapeTargetCollection(memory_registry, config, nats_client=None)
    coll_b = ScrapeTargetCollection(memory_registry, config, nats_client=None)

    target = coll_a.create({"target_id": "warn_act_tx", "url": "https://example.gov/tx"})
    await target.save()

    assert await coll_b.get("warn_act_tx") is None


async def test_target_list_all_returns_in_memory_rows(
    memory_registry: CollectionRegistry, config: DefaultCoreConfig
) -> None:
    coll = ScrapeTargetCollection(memory_registry, config, nats_client=None)
    target = coll.create({"target_id": "warn_act_or", "url": "https://example.gov/or"})
    await target.save()

    all_targets = await coll.list_all()

    assert [t.target_id for t in all_targets] == ["warn_act_or"]


async def test_target_delete_removes_from_in_memory_dict(
    memory_registry: CollectionRegistry, config: DefaultCoreConfig
) -> None:
    coll = ScrapeTargetCollection(memory_registry, config, nats_client=None)
    target = coll.create({"target_id": "warn_act_pa", "url": "https://example.gov/pa"})
    await target.save()

    await coll.delete("warn_act_pa")

    assert "warn_act_pa" not in coll._rows
