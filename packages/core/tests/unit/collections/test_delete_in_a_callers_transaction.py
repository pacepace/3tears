"""``delete(key, conn=)``: a delete joins a caller's transaction, or is refused when the collection cannot keep what a
direct delete keeps.

Over the SQLite L3 (a real transaction), a schema-backed collection's delete commits with the caller's transaction and
rolls back with it. Each refusal is a collection whose direct delete does something a caller's transaction cannot:
announce its delete cascades after its own commit, advance a cached absence, commit L2 first or defer its L3 write,
delete its own way (an overridden ``delete_from_store``), or keep no L3 at all. A connection no ``CallerTransaction``
opened is refused too.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from threetears.core.backends import SqliteL3Backend
from threetears.core.cache import SQLiteBackend
from threetears.core.collections import NO_L2
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.delete_cascade import DeleteCascade
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.l1_tables import TEST_L1_TABLE

NOTES = TableSchema(
    name="notes",
    primary_key="id",
    columns=[
        Column("id", STRING_TYPE),
        Column("text", STRING_TYPE),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    cas_column="date_updated",
    l1_eviction=TEST_L1_TABLE,
)
Notes: type[SchemaBackedCollection[Any]] = collection_for_schema(NOTES)


class _OwnDeleteNotes(Notes):  # type: ignore[valid-type,misc]
    """deletes its own way: a delete on a caller's transaction cannot know what it does."""

    async def delete_from_store(self, entity_id: Any) -> None:
        await super().delete_from_store(entity_id)


class _AbsenceCachingNotes(Notes):  # type: ignore[valid-type,misc]
    """caches absences: its absence-caching switch is turned on once it is built (building one needs a generation
    source this test has no use for)."""


class _L2FirstNotes(Notes):  # type: ignore[valid-type,misc]
    @property
    def writes_l2_first(self) -> bool:
        return True


class _CascadingNotes(Notes):  # type: ignore[valid-type,misc]
    delete_cascades = (DeleteCascade(child_table="note_links", child_column="note_id", action="CASCADE"),)


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteL3Backend]:
    async with SqliteL3Backend(tmp_path / "state.sqlite") as opened:
        yield opened


@pytest.fixture
def registry(store: SqliteL3Backend) -> CollectionRegistry:
    l1 = SQLiteBackend(f"delete_tx_{uuid.uuid4().hex[:8]}")
    built = CollectionRegistry()
    built.configure(l1_backend=l1, l3_pool=store)
    return built


def _row(key: str) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {"id": key, "text": "kept", "date_created": now, "date_updated": now}


async def _saved(notes: Any, key: str) -> None:
    await notes.save_entity(notes.create(_row(key)))


async def test_a_delete_commits_with_the_callers_transaction(store: SqliteL3Backend, registry: Any) -> None:
    notes = Notes(registry, DefaultCoreConfig(), NO_L2)
    await _saved(notes, "a")
    async with store.acquire() as conn, CallerTransaction(conn):
        await notes.delete("a", conn=conn)
        # the row leaves this process's cache now, and the store when the transaction commits
        assert not notes.exists_in_cache_sync("a")
    assert await store.fetch_one("notes", {"id": "a"}) is None
    assert await notes.get("a") is None


async def test_a_delete_rolls_back_with_the_callers_transaction(store: SqliteL3Backend, registry: Any) -> None:
    notes = Notes(registry, DefaultCoreConfig(), NO_L2)
    await _saved(notes, "a")
    with pytest.raises(RuntimeError, match="abandoned"):
        async with store.acquire() as conn, CallerTransaction(conn):
            await notes.delete("a", conn=conn)
            raise RuntimeError("abandoned")
    assert await store.fetch_one("notes", {"id": "a"}) is not None
    got = await notes.get("a")
    assert got is not None and got.text == "kept"


@pytest.mark.parametrize(
    ("collection", "refusal"),
    [
        pytest.param(_OwnDeleteNotes, "deletes its own way", id="overridden-delete_from_store"),
        pytest.param(_AbsenceCachingNotes, "caches absences", id="absence-caching"),
        pytest.param(_L2FirstNotes, "L2-first", id="l2-first"),
        pytest.param(_CascadingNotes, "delete cascades", id="delete-cascades"),
    ],
)
async def test_a_collection_that_cannot_keep_a_direct_deletes_obligations_refuses_conn(
    store: SqliteL3Backend, registry: Any, collection: Any, refusal: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    notes = collection(registry, DefaultCoreConfig(), NO_L2)
    if collection is _AbsenceCachingNotes:
        monkeypatch.setattr(_AbsenceCachingNotes, "_negative_cache_writes_advance", property(lambda self: True))
    await store.upsert("notes", _row("a"))
    async with store.acquire() as conn, CallerTransaction(conn):
        with pytest.raises(ValueError, match=refusal):
            await notes.delete("a", conn=conn)
    assert await store.fetch_one("notes", {"id": "a"}) is not None


async def test_a_connection_no_caller_transaction_opened_is_refused(store: SqliteL3Backend, registry: Any) -> None:
    notes = Notes(registry, DefaultCoreConfig(), NO_L2)
    await _saved(notes, "a")
    async with store.acquire() as conn:
        with pytest.raises(ValueError, match="CallerTransaction"):
            await notes.delete("a", conn=conn)
    assert await store.fetch_one("notes", {"id": "a"}) is not None


async def test_a_collection_with_no_transactional_delete_refuses_conn(registry: Any) -> None:
    """BaseCollection's own answer: a collection that cannot delete on a caller's connection says so."""
    from threetears.core.collections.base import BaseCollection
    from threetears.core.entities.base import BaseEntity

    class _Plain(BaseCollection[BaseEntity]):
        @property
        def table_name(self) -> str:
            return "plain"

        @property
        def entity_class(self) -> type[BaseEntity]:
            return BaseEntity

        async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
            return None

        async def save_to_store(self, data: dict[str, Any], original_timestamp: Any = None, *, conn: Any = None) -> int:
            return 1

        async def delete_from_store(self, entity_id: Any) -> None:
            raise AssertionError("a refused delete reached the store")

        def serialize(self, data: dict[str, Any]) -> bytes:
            return b"{}"

        def deserialize(self, data: bytes) -> dict[str, Any]:
            return {}

    plain = _Plain(registry, DefaultCoreConfig(), NO_L2)
    with pytest.raises(ValueError, match="cannot delete inside a caller's transaction"):
        await plain.delete("a", conn=object())


async def test_a_durable_store_collection_that_deletes_its_own_way_refuses_conn(store: SqliteL3Backend) -> None:
    from threetears.core.collections.durable_store import DurableStoreCollection
    from threetears.core.entities.base import BaseEntity

    store.register_schema("notes", NOTES)

    class _Docs(DurableStoreCollection[BaseEntity]):
        primary_key_column = "id"

        @property
        def table_name(self) -> str:
            return "notes"

        @property
        def entity_class(self) -> type[BaseEntity]:
            return BaseEntity

        async def delete_from_store(self, entity_id: Any) -> None:
            await super().delete_from_store(entity_id)

    built = CollectionRegistry()
    built.configure(l3_pool=store)
    docs = _Docs(built, DefaultCoreConfig(), store, nats_client=None)
    await store.upsert("notes", _row("a"))
    async with store.acquire() as conn, CallerTransaction(conn):
        with pytest.raises(ValueError, match="deletes its own way"):
            await docs.delete("a", conn=conn)
    assert await store.fetch_one("notes", {"id": "a"}) is not None


async def test_a_durable_store_collection_enrolls_the_row_before_it_goes_for_its_invalidation_columns(
    store: SqliteL3Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """the eviction's broadcast carries the declared columns of the row removed, read on the caller's transaction."""
    from threetears.core.collections.durable_store import DurableStoreCollection
    from threetears.core.entities.base import BaseEntity

    store.register_schema("notes", NOTES)

    class _Docs(DurableStoreCollection[BaseEntity]):
        primary_key_column = "id"
        invalidation_columns = ("text",)

        @property
        def table_name(self) -> str:
            return "notes"

        @property
        def entity_class(self) -> type[BaseEntity]:
            return BaseEntity

    enrolled: list[tuple[Any, dict[str, Any] | None]] = []
    original = CallerTransaction.enroll

    def enroll(self: CallerTransaction, collection: Any, entity_id: Any, *, row: dict[str, Any] | None = None) -> None:
        enrolled.append((entity_id, row))
        original(self, collection, entity_id, row=row)

    monkeypatch.setattr(CallerTransaction, "enroll", enroll)
    built = CollectionRegistry()
    built.configure(l3_pool=store)
    docs = _Docs(built, DefaultCoreConfig(), store, nats_client=None)
    await store.upsert("notes", _row("a"))
    async with store.acquire() as conn, CallerTransaction(conn):
        await docs.delete("a", conn=conn)
    assert [(key, None if row is None else row["text"]) for key, row in enrolled] == [("a", "kept")]
    assert await store.fetch_one("notes", {"id": "a"}) is None
