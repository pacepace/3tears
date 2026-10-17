"""change sets: many rows one commit, refused whole, and collections joining them through CallerTransaction."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import NO_L2, DurableStoreCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError
from threetears.git_store import Change, ChangeConflictError, ChangeSet, GitDurableStore, HandEditPendingError

T0 = datetime(2026, 10, 17, 9, 0, tzinfo=UTC)


def keeper(tile: str, *, when: datetime = T0, **extra: Any) -> dict[str, Any]:
    return {"character": "aria", "batch": "V3", "tile": tile, "date_updated": when, **extra}


def key(tile: str) -> dict[str, str]:
    return {"character": "aria", "batch": "V3", "tile": tile}


def commits(repo: Any) -> int:
    return len(repo.git.log(limit=1000))


async def test_many_rows_across_tables_are_one_commit(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    async with repo.change("aria: bank 2 keepers and the recipe", trailers={"Job": "42"}) as change_set:
        await store.upsert("keepers", keeper("1"), conn=change_set)
        await store.upsert("keepers", keeper("2"), conn=change_set)
        await store.upsert("recipes", {"character": "aria", "look": "x", "date_updated": T0}, conn=change_set)
        assert commits(repo) == 0  # nothing lands before the block ends
        assert await store.fetch_one("keepers", key("1"), conn=change_set) == keeper("1")  # the set reads its own write
    assert change_set.result is not None and change_set.result.sha is not None
    assert len(change_set.result.paths) == 3
    assert commits(repo) == 1
    entry = repo.git.log(limit=1)[0]
    assert entry.subject == "aria: bank 2 keepers and the recipe"
    assert "Job: 42" in entry.body


async def test_a_compare_and_swap_miss_refuses_the_whole_set_and_touches_no_file(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert("keepers", keeper("1", score=1))
    before = (repo.config.work_tree / "aria/keepers/V3/1.json").read_bytes()
    with pytest.raises(ChangeConflictError) as raised:
        await repo.commit(
            [
                Change("keepers", key("2"), keeper("2")),
                Change(
                    "keepers",
                    key("1"),
                    keeper("1", when=T0 + timedelta(minutes=1), score=2),
                    cas=T0 - timedelta(seconds=1),
                ),
            ],
            "would lose a change",
        )
    assert raised.value.rows[0].reason == "changed"
    assert isinstance(raised.value, ConcurrentModificationError)
    assert (repo.config.work_tree / "aria/keepers/V3/1.json").read_bytes() == before
    assert not (repo.config.work_tree / "aria/keepers/V3/2.json").exists()
    assert commits(repo) == 1


async def test_a_set_that_changes_no_bytes_makes_no_commit(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert("keepers", keeper("1"))
    async with repo.change("nothing new") as change_set:
        await store.upsert("keepers", keeper("1"), conn=change_set)
    assert change_set.result is not None and change_set.result.sha is None
    assert commits(repo) == 1


async def test_a_commit_takes_only_its_own_files(make_repo: Any, git_cli: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    root = repo.config.work_tree
    (root / "aria").mkdir()
    (root / "aria/training-log.md").write_text("a note in progress\n")
    git_cli(root, "add", "aria/training-log.md")  # even a staged file stays out
    await store.upsert("keepers", keeper("1"))
    assert repo.git.changed(None, repo.git.head() or "")[0].path == "aria/keepers/V3/1.json"
    assert len(repo.git.changed(None, repo.git.head() or "")) == 1
    assert ("A ", "aria/training-log.md") in repo.git.dirty()


async def test_a_hand_edit_in_the_way_is_never_written_over(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    await store.upsert("keepers", keeper("1", score=1))
    target = repo.config.work_tree / "aria/keepers/V3/1.json"
    target.write_text(target.read_text().replace('"score": 1', '"score": 5'))
    with pytest.raises(HandEditPendingError) as raised:
        await repo.commit([Change("keepers", key("1"), keeper("1", score=2))], "over a hand edit")
    assert raised.value.paths == ("aria/keepers/V3/1.json",)
    assert '"score": 5' in target.read_text()  # the person's edit is untouched
    assert commits(repo) == 1


async def test_a_nested_transaction_that_fails_drops_only_its_own_writes(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    change_set = ChangeSet(repo, "outer")
    async with change_set.transaction():
        await store.upsert("keepers", keeper("1"), conn=change_set)
        with pytest.raises(RuntimeError):
            async with change_set.transaction():
                await store.upsert("keepers", keeper("2"), conn=change_set)
                raise RuntimeError("savepoint rolls back")
    assert change_set.result is not None
    assert change_set.result.paths == ("aria/keepers/V3/1.json",)


async def test_a_set_that_raises_commits_nothing(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    with pytest.raises(RuntimeError):
        async with repo.change("abandoned") as change_set:
            await store.upsert("keepers", keeper("1"), conn=change_set)
            raise RuntimeError("the caller gave up")
    assert commits(repo) == 0
    assert not (repo.config.work_tree / "aria").exists()


# ---------------------------------------------------------------- collections on the store


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "keepers",
        metadata,
        Column("character", String(64), primary_key=True),
        Column("batch", String(64), primary_key=True),
        Column("tile", String(64), primary_key=True),
        Column("score", Integer),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    Table(
        "recipes",
        metadata,
        Column("character", String(64), primary_key=True),
        Column("look", String(255)),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    return metadata


class _Keeper(BaseEntity):
    primary_key_field = "tile"


class _Recipe(BaseEntity):
    primary_key_field = "character"


class _Keepers(DurableStoreCollection[_Keeper]):
    primary_key_column = ("character", "batch", "tile")

    @property
    def table_name(self) -> str:
        return "keepers"

    @property
    def entity_class(self) -> type[_Keeper]:
        return _Keeper


class _Recipes(DurableStoreCollection[_Recipe]):
    primary_key_column = "character"

    @property
    def table_name(self) -> str:
        return "recipes"

    @property
    def entity_class(self) -> type[_Recipe]:
        return _Recipe


@pytest.fixture()
def registry() -> Iterator[CollectionRegistry]:
    backend = SQLiteBackend(db_name=f"git_store_{uuid.uuid4().hex[:8]}")
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=backend, kv_key_scope="test-principal")
    yield registry
    from threetears.core.testing.bridge import drain_and_shutdown_bridge

    drain_and_shutdown_bridge()


async def test_collection_saves_join_one_commit_and_the_caches_settle_after(
    make_repo: Any, registry: CollectionRegistry
) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    keepers = _Keepers(registry, DefaultCoreConfig(), store, nats_client=NO_L2)
    recipes = _Recipes(registry, DefaultCoreConfig(), store, nats_client=NO_L2)

    async with repo.change("aria: bank a keeper with the recipe") as change_set:
        await keepers.save_entity(
            _Keeper({"character": "aria", "batch": "V3", "tile": "1", "score": 3, "date_updated": None}, is_new=True),
            conn=change_set,
        )
        await recipes.save_entity(
            _Recipe({"character": "aria", "look": "x", "date_updated": None}, is_new=True), conn=change_set
        )
    assert commits(repo) == 1

    loaded = await keepers.get(("aria", "V3", "1"))
    assert loaded is not None and loaded.score == 3
    assert isinstance(loaded.original_date_updated, datetime)  # the version a later save swaps against

    # a save from what was loaded lands; the next save from the same stale copy is refused
    stale = await keepers.get(("aria", "V3", "1"))
    assert stale is not None
    async with repo.change("raise the score") as change_set:
        loaded.score = 4
        await keepers.save_entity(loaded, conn=change_set)
    assert commits(repo) == 2
    with pytest.raises(ConcurrentModificationError):
        async with repo.change("from a stale copy") as change_set:
            stale.score = 1
            await keepers.save_entity(stale, conn=change_set)
    assert commits(repo) == 2
    current = await keepers.get(("aria", "V3", "1"))
    assert current is not None and current.score == 4


async def test_a_conflict_found_at_commit_refuses_the_set_and_the_caches_still_settle(
    make_repo: Any, registry: CollectionRegistry
) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    keepers = _Keepers(registry, DefaultCoreConfig(), store, nats_client=NO_L2)
    await keepers.save_entity(
        _Keeper({"character": "aria", "batch": "V3", "tile": "1", "score": 1, "date_updated": None}, is_new=True)
    )
    mine = await keepers.get(("aria", "V3", "1"))
    theirs = await keepers.get(("aria", "V3", "1"))
    assert mine is not None and theirs is not None

    with pytest.raises(ChangeConflictError):
        async with repo.change("mine") as change_set:
            mine.score = 2
            await keepers.save_entity(mine, conn=change_set)  # checks out against the file now
            theirs.score = 3
            await keepers.save_entity(theirs)  # another save of the same row lands first, as its own commit
    assert commits(repo) == 2
    after = await keepers.get(("aria", "V3", "1"))
    assert after is not None and after.score == 3  # theirs stands; L1 holds no trace of mine
