"""A new entity's working row is its own: an L1 eviction while a caller holds it across an await loses nothing.

On the hub (cobalt dev), ``group_member_add`` built a new membership, awaited the audit's before-snapshot, then saved:
the row left L1 in between and the save raised ``L1 cache miss in to_dict()``. Any caller that holds a new entity
across an await can hit it.
"""

from __future__ import annotations

import uuid

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from .test_base_collection import StubCollection

#: any principal scope: a registry with none refuses to key anything
SCOPE = "test-principal"


@pytest.fixture()
def collection() -> StubCollection:
    registry = CollectionRegistry()
    registry.configure(l1_backend=SQLiteBackend(db_name=f"evict_{uuid.uuid4().hex[:8]}"), kv_key_scope=SCOPE)
    return StubCollection(registry, DefaultCoreConfig())


def test_a_new_entity_answers_after_its_working_row_leaves_l1(collection: StubCollection) -> None:
    entity = collection.entity_class({"id": "a", "name": "x", "score": 1}, is_new=True, collection=collection)
    collection.evict_from_cache_sync("a")
    assert entity.to_dict()["name"] == "x"


async def test_a_new_entity_whose_working_row_left_l1_still_saves(collection: StubCollection) -> None:
    entity = collection.entity_class({"id": "b", "name": "y", "score": 2}, is_new=True, collection=collection)
    collection.evict_from_cache_sync("b")  # stands in for any eviction during an await before the save
    entity.score = 3
    await collection.save_entity(entity)
    stored = await collection.fetch_from_store("b")
    assert stored is not None and stored["name"] == "y" and stored["score"] == 3
