"""two seams a git-backed application needed and reached into private members for.

- **A store table apart from the cache's table.** Many collections, each cached under a name of its own
  (``story_1_scenes``), over one store table laid out by a bare name (``scenes``): ``store_table=``.
- **Adopting a row L3 already holds.** A writer that owns its L3 write (a coordinator's file write and stage, for its
  own crash recovery) then makes every cache say what L3 holds: ``adopt_stored_row``, in place of writing L1, L2 and
  the broadcast through private members.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

from threetears.core.backends.git import GitDurableStore, JsonCodec, TableLayout, TemplateLayout
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.durable_store import DurableStoreCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeNatsClient
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

from .conftest import MakeRepo

T0 = datetime(2026, 11, 25, 9, 0, 0, tzinfo=UTC)


class _Scene(BaseEntity):
    primary_key_field = "id"


class _Scenes(DurableStoreCollection[_Scene]):
    """a story's scenes: cached as ``<story>_scenes``, stored in the layout's bare ``scenes``."""

    primary_key_column = "id"

    def __init__(self, story: str, *args: Any, **kwargs: Any) -> None:
        self._story = story
        super().__init__(*args, **kwargs)

    @property
    def table_name(self) -> str:
        return f"{self._story}_scenes"

    @property
    def entity_class(self) -> type[_Scene]:
        return _Scene


def _layout() -> TemplateLayout:
    return TemplateLayout([TableLayout("scenes", "scenes/{id}.json", ("id",), JsonCodec())])


def _l1(*stories: str) -> SQLiteBackend:
    metadata = MetaData()
    for story in stories:
        Table(
            f"{story}_scenes",
            metadata,
            Column("id", String, primary_key=True),
            Column("words", Integer),
            Column("date_updated", DateTime(timezone=True)),
            info={L1_EVICTION_INFO: TEST_L1_TABLE},
        )
    l1 = SQLiteBackend(f"hooks_{'_'.join(stories)}")
    l1.initialize(metadata)
    return l1


@pytest.fixture
async def store(make_repo: MakeRepo) -> GitDurableStore:
    return GitDurableStore(await make_repo("stories", layout=_layout()))


async def test_a_collection_cached_under_its_own_name_reads_and_writes_the_store_table(
    store: GitDurableStore,
) -> None:
    l1 = _l1("s1")
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=store)
    scenes = _Scenes("s1", registry, DefaultCoreConfig(), store, nats_client=None, store_table="scenes")
    try:
        assert (scenes.table_name, scenes.store_table) == ("s1_scenes", "scenes")
        await scenes.save_entity(scenes.create({"id": "a", "words": 10, "date_updated": T0}))
        assert await store.fetch_one("scenes", {"id": "a"}) is not None
        scenes.evict_from_cache_sync("a")
        got = await scenes.get("a")
        assert got is not None
        assert got.words == 10
        assert [scene.id for scene in await scenes.list()] == ["a"]
        await scenes.delete("a")
        assert await store.fetch_one("scenes", {"id": "a"}) is None
    finally:
        l1.reset()


async def test_without_store_table_the_store_is_asked_for_the_collections_own_name(store: GitDurableStore) -> None:
    registry = CollectionRegistry()
    scenes = _Scenes("s1", registry, DefaultCoreConfig(), store, nats_client=None)
    assert scenes.store_table == "s1_scenes"


def test_an_empty_store_table_is_refused(store: GitDurableStore) -> None:
    with pytest.raises(ValueError, match="store_table"):
        _Scenes("s1", CollectionRegistry(), DefaultCoreConfig(), store, nats_client=None, store_table="")


async def test_a_row_l3_already_holds_is_adopted_by_every_cache(store: GitDurableStore) -> None:
    l1 = _l1("s2")
    nats = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(kv_key_scope="test-principal", l1_backend=l1, l3_pool=store)
    scenes = _Scenes("s2", registry, DefaultCoreConfig(), store, nats_client=nats, store_table="scenes")
    row = {"id": "b", "words": 20, "date_updated": T0}
    try:
        # the coordinator's own L3 write
        await store.upsert("scenes", row)
        published = len(nats.published)

        await scenes.adopt_stored_row(row)

        assert scenes.exists_in_cache_sync("b")
        assert len(nats.published) == published + 1
        # L2 holds it: a process with an empty L1 reads it there, without L3
        scenes.evict_from_cache_sync("b")
        await store.delete("scenes", {"id": "b"})
        got = await scenes.get("b")
        assert got is not None
        assert got.words == 20
    finally:
        l1.reset()


async def test_a_row_without_its_key_is_refused(store: GitDurableStore) -> None:
    scenes = _Scenes("s3", CollectionRegistry(), DefaultCoreConfig(), store, nats_client=None, store_table="scenes")
    with pytest.raises(KeyError, match="'id'"):
        await scenes.adopt_stored_row({"words": 1})
