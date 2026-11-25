"""two seams an application that uses the git store needed and reached into private members for.

- **A store table apart from the cache's table.** Many collections, each cached under a name of its own
  (``story_1_scenes``), over one store table laid out by a bare name (``scenes``): ``store_table=``.
- **Adopting a row L3 already holds.** A writer that owns its L3 write (a coordinator's file write and stage, for its
  own crash recovery) then makes every cache say what L3 holds: ``adopt_stored_row`` drops the row from L1, L2 and
  every peer (never puts it, so an older row cannot land over a newer save) and advances a switched-on generation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

from threetears.core.backends.git import GitDurableStore, JsonCodec, TableLayout, TemplateLayout, bind_collections
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import NO_L2, WRITE_GENERATION
from threetears.core.collections.durable_store import DurableStoreCollection
from threetears.core.collections.l2_first import L2_FIRST
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import GenerationUnavailableError
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


def _pod(store: GitDurableStore, story: str, *, cls: type[_Scenes] | None = None, source: Any = None) -> Any:
    l1 = _l1(story)
    nats = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(kv_key_scope="test-principal", l1_backend=l1, l3_pool=store)
    if source is not None:
        registry.set_generation_source(source)
    scenes = (cls or _Scenes)(story, registry, DefaultCoreConfig(), store, nats_client=nats, store_table="scenes")
    return l1, nats, scenes


def _told(nats: FakeNatsClient, scenes: _Scenes) -> list[tuple[str, list[str]]]:
    return [(message.table, list(message.ids)) for message in nats.published if hasattr(message, "ids")]


async def test_an_adopted_row_leaves_every_cache_and_is_read_from_l3(store: GitDurableStore) -> None:
    l1, nats, scenes = _pod(store, "s2")
    try:
        await scenes.save_entity(scenes.create({"id": "b", "words": 10, "date_updated": T0}))
        assert scenes.exists_in_cache_sync("b")
        # the coordinator's own L3 write
        adopted = {"id": "b", "words": 20, "date_updated": T0.replace(minute=1)}
        await store.upsert("scenes", adopted)
        told = len(_told(nats, scenes))

        await scenes.adopt_stored_row(adopted)

        assert not scenes.exists_in_cache_sync("b")
        # the broadcast names this row, so every peer drops its copy
        assert _told(nats, scenes)[told:] == [("s2_scenes", ["b"])]
        # L2 holds no copy either: with L3 emptied, nothing answers the row
        await store.delete("scenes", {"id": "b"})
        assert await scenes.get("b") is None
        await store.upsert("scenes", adopted)
        got = await scenes.get("b")
        assert got is not None
        assert got.words == 20
    finally:
        l1.reset()


async def test_an_adopted_older_row_never_overwrites_a_newer_save_in_any_cache(store: GitDurableStore) -> None:
    l1, _nats, scenes = _pod(store, "s4")
    try:
        older = {"id": "c", "words": 1, "date_updated": T0}
        await store.upsert("scenes", older)
        # a save of the same key commits after the coordinator's write, and caches its row in L1 and L2
        await scenes.save_entity(scenes.create({"id": "c", "words": 2, "date_updated": T0.replace(minute=2)}))

        await scenes.adopt_stored_row(older)

        got = await scenes.get("c")
        assert got is not None
        assert got.words == 2, "the adopted older row was cached over the newer save"
        scenes.evict_from_cache_sync("c")
        got = await scenes.get("c")
        assert got is not None
        assert got.words == 2
    finally:
        l1.reset()


class _SwitchedOnScenes(_Scenes):
    write_generation = WRITE_GENERATION


class _CountingSource:
    """a generation source that counts its advances, and can be made to refuse them."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.failing = False

    async def current(self, table_name: str) -> str:
        return f"inc:{self.counts.get(table_name, 0)}"

    async def advance(self, table_name: str) -> str:
        if self.failing:
            raise GenerationUnavailableError(f"generation for {table_name!r} is unreachable")
        self.counts[table_name] = self.counts.get(table_name, 0) + 1
        return f"inc:{self.counts[table_name]}"

    async def advance_with_change(self, table_name: str, rows: Any, **_kwargs: Any) -> str:
        return await self.advance(table_name)

    async def change(self, table_name: str, token: str) -> None:
        return None


async def test_an_adopted_row_advances_a_switched_on_collections_generation(store: GitDurableStore) -> None:
    source = _CountingSource()
    l1, _nats, scenes = _pod(store, "s5", cls=_SwitchedOnScenes, source=source)
    try:
        row = {"id": "d", "words": 3, "date_updated": T0}
        await store.upsert("scenes", row)
        await scenes.adopt_stored_row(row)
        assert source.counts == {"s5_scenes": 1}
    finally:
        l1.reset()


async def test_a_failed_advance_still_drops_and_tells_then_raises(store: GitDurableStore) -> None:
    source = _CountingSource()
    l1, nats, scenes = _pod(store, "s6", cls=_SwitchedOnScenes, source=source)
    try:
        await scenes.save_entity(scenes.create({"id": "e", "words": 4, "date_updated": T0}))
        assert scenes.exists_in_cache_sync("e")
        row = {"id": "e", "words": 5, "date_updated": T0.replace(minute=3)}
        await store.upsert("scenes", row)
        source.failing = True
        told = len(_told(nats, scenes))
        with pytest.raises(GenerationUnavailableError):
            await scenes.adopt_stored_row(row)
        assert not scenes.exists_in_cache_sync("e")
        assert _told(nats, scenes)[told:] == [("s6_scenes", ["e"])]
    finally:
        l1.reset()


async def test_a_row_without_its_key_is_refused(store: GitDurableStore) -> None:
    scenes = _Scenes("s3", CollectionRegistry(), DefaultCoreConfig(), store, nats_client=None, store_table="scenes")
    with pytest.raises(KeyError, match="'id'"):
        await scenes.adopt_stored_row({"words": 1})


async def test_each_storys_collection_bound_on_its_own_repository_hears_only_its_hand_commits(
    make_repo: MakeRepo, git_cli: Any
) -> None:
    # many collections over one bare layout table: bind_collections maps a table to ONE collection, so each is bound
    # on its own repository's on_rows_changed, never once over GitRepos (whose changes name no repository)
    repos = {story: await make_repo(f"story_{story}", layout=_layout()) for story in ("s7", "s8")}
    l1 = _l1("s7", "s8")
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, kv_key_scope="test-principal")
    scenes = {
        story: _Scenes(
            story, registry, DefaultCoreConfig(), GitDurableStore(repo), nats_client=NO_L2, store_table="scenes"
        )
        for story, repo in repos.items()
    }
    try:
        for story, repo in repos.items():
            bind_collections(repo, {"scenes": scenes[story]})
            await scenes[story].save_entity(scenes[story].create({"id": "x", "words": 1, "date_updated": T0}))
            assert scenes[story].get_row_sync("x") is not None

        path = repos["s8"].config.work_tree / "scenes/x.json"
        path.write_text(path.read_text().replace('"words": 1', '"words": 2'))
        git_cli(repos["s8"].config.work_tree, "commit", "-qam", "edited by hand")
        await repos["s8"].sync()

        assert scenes["s8"].get_row_sync("x") is None, "the hand commit in story 8's repository was not heard"
        assert scenes["s7"].get_row_sync("x") is not None, "story 7's row was dropped by another story's commit"
        fresh = await scenes["s8"].get("x")
        assert fresh is not None
        assert fresh.words == 2
    finally:
        l1.reset()


class _L2FirstScenes(_Scenes):
    l2_first = L2_FIRST


class _L2StoreScenes(_Scenes):
    l2_is_the_store = True


@pytest.mark.parametrize("cls", [_L2FirstScenes, _L2StoreScenes], ids=["l2-first", "l2-is-the-store"])
async def test_a_collection_whose_record_is_l2_refuses_to_adopt(store: GitDurableStore, cls: type[_Scenes]) -> None:
    l1, _nats, scenes = _pod(store, "s9", cls=cls)
    try:
        with pytest.raises(ValueError, match="L2 is this collection's record"):
            await scenes.adopt_stored_row({"id": "f", "words": 1, "date_updated": T0})
    finally:
        l1.reset()
