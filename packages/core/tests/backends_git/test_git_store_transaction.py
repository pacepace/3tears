"""a write-buffered flush over the git store is one commit, not one commit per row.

``flush_pending`` persists a batch inside ``backend.transaction()`` when every table's L3 offers one, and degrades
to one durable write per entity otherwise. The git store's transaction is a change set (``GitRepo.change``): every
row the flush writes lands in its one commit, and a row changed since it was read refuses the whole set, which the
flush answers by its per-entity path, as it does for a database transaction that failed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from threetears.core.backends.git import GitDurableStore
from threetears.core.collections.durable_store import DurableStoreCollection
from threetears.core.collections.flush import WriteBuffer, flush_pending
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity

T0 = datetime(2026, 11, 25, 9, 0, 0, tzinfo=UTC)


class _Recipe(BaseEntity):
    primary_key_field = "character"


class _Recipes(DurableStoreCollection[_Recipe]):
    primary_key_column = "character"

    @property
    def table_name(self) -> str:
        return "recipes"

    @property
    def entity_class(self) -> type[_Recipe]:
        return _Recipe


def _commits(repo: Any) -> int:
    return len(repo.git.log(limit=1000))


async def test_a_flush_of_many_rows_is_one_commit(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    registry = CollectionRegistry()
    registry.configure(l3_pool=store)
    _Recipes(registry, DefaultCoreConfig(), store, nats_client=None)
    before = _commits(repo)
    buffer = WriteBuffer()
    for name in ("aria", "bran", "cora"):
        await buffer.add("recipes", name, {"character": name, "steps": 3, "date_updated": T0})
    assert await flush_pending(buffer, registry) == 3
    assert _commits(repo) == before + 1
    for name in ("aria", "bran", "cora"):
        assert await store.fetch_one("recipes", {"character": name}) is not None


async def test_the_transaction_is_a_change_set_committing_its_writes_together(make_repo: Any) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    before = _commits(repo)
    async with store.transaction(message="two recipes") as conn:
        await store.upsert("recipes", {"character": "aria", "steps": 1, "date_updated": T0}, conn=conn)
        await store.upsert("recipes", {"character": "bran", "steps": 2, "date_updated": T0}, conn=conn)
        assert _commits(repo) == before
    assert _commits(repo) == before + 1
    assert repo.git.log(limit=1)[0].subject == "two recipes"


async def test_a_hand_edit_on_one_rows_file_refuses_the_set_and_the_other_rows_land_one_by_one(
    make_repo: Any,
) -> None:
    repo = await make_repo()
    store = GitDurableStore(repo)
    registry = CollectionRegistry()
    registry.configure(l3_pool=store)
    _Recipes(registry, DefaultCoreConfig(), store, nats_client=None)
    await store.upsert("recipes", {"character": "bran", "steps": 1, "date_updated": T0})
    edited = repo.config.work_tree / "bran/recipe.json"
    edited.write_text('{"character": "bran", "steps": 99}\n')  # a person's edit, not yet committed
    buffer = WriteBuffer()
    for name in ("aria", "bran", "cora"):
        await buffer.add("recipes", name, {"character": name, "steps": 3, "date_updated": T0})
    before = _commits(repo)

    assert await flush_pending(buffer, registry) == 2
    # the set was refused whole; the per-entity pass landed the two rows, each its own commit
    assert _commits(repo) == before + 2
    for name in ("aria", "cora"):
        assert await store.fetch_one("recipes", {"character": name}) is not None
    # the hand edit is not written over, and the row waits in the buffer for another try
    assert edited.read_text() == '{"character": "bran", "steps": 99}\n'
    waiting = await buffer.drain()
    assert [(item.table_name, item.entity_id, item.retries) for item in waiting] == [("recipes", "bran", 1)]
