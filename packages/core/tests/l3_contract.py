"""the behaviour an L3 durable store owes the collections over it, run against each store unchanged.

Two tiers, because the stores are two kinds:

- :class:`DurableStoreContract` -- every L3: ``SqlL3Backend`` over Postgres, ``SqliteL3Backend`` and the git store
  (``GitDurableStore``). What a :class:`~threetears.core.collections.DurableStoreCollection` relies on: a row of JSON
  values and aware instants written whole and read back, the conflict mode the write asks for, the compare-and-swap
  on ``date_updated``, deletes, composite keys, equality scans, many rows at once, and a transaction (the store's own
  kind of one, opened by :meth:`DurableStoreContract.begin`) committing whole or not at all.
- :class:`SqlStoreContract` -- the stores that take a registered :class:`TableSchema` and serve
  ``SchemaBackedCollection``: Postgres and SQLite. On top: typed values (UUID, ``NUMERIC`` as ``Decimal``, bytes, a
  naive instant read as UTC), an update that rewrites the columns it names and keeps the rest, immutable columns,
  the schema's conflict mode, the NULL-safe and the ordered compare-and-swap, a scan refusing a column the table
  lacks, truncate, and raw SQL.

The git store keeps a row as a file of JSON, written whole, with a transaction that is a commit; those are what it
is, so it runs the first tier. A store's own test subclasses the tier it serves and gives it a ``store`` fixture: an
open store holding the contract's tables, empty. Where the stores run it: ``tests/unit/backends/
test_sqlite_l3_contract.py`` (SQLite), ``tests/backends_git/test_git_store_l3_contract.py`` (git) and
``tests/integration/test_sql_l3_contract_live.py`` (Postgres). A behaviour the stores of one tier differ on is a
failure of one of them, never a branch here; per store are only the exception a duplicate insert raises
(:attr:`DurableStoreContract.duplicate_error`, one type for every store of the schema tier: the ``asyncpg`` one, so
a caller's ``except asyncpg.UniqueViolationError`` holds on either) and how a transaction is opened
(:meth:`DurableStoreContract.begin`).
"""

from __future__ import annotations

import contextlib
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from collections.abc import AsyncIterator, Iterator
from typing import Any, ClassVar

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import NO_L2
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.durable_store import DurableStoreCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError
from threetears.core.testing.kv import FakeNatsClient

from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    Index,
    TableSchema,
    UniqueConstraintDef,
    l2_order_columns,
)
from threetears.core.testing.l1_tables import TEST_L1_TABLE

ITEMS = TableSchema(
    name="l3c_items",
    primary_key="id",
    columns=[
        Column("id", UUID_TYPE),
        Column("name", STRING_TYPE),
        Column("payload", JSONB_TYPE, nullable=True),
        Column("amount", NUMERIC_TYPE, nullable=True, precision=12, scale=2),
        Column("flag", BOOL_TYPE, nullable=True),
        Column("count", INT_TYPE, nullable=True),
        Column("blob", BYTES_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    cas_column="date_updated",
    l1_eviction=TEST_L1_TABLE,
)
DOCS = TableSchema(
    name="l3c_docs",
    primary_key="id",
    columns=[
        Column("id", STRING_TYPE),
        Column("name", STRING_TYPE),
        Column("payload", JSONB_TYPE, nullable=True),
        Column("flag", BOOL_TYPE, nullable=True),
        Column("count", INT_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    cas_column="date_updated",
    l1_eviction=TEST_L1_TABLE,
)
IGNORING = TableSchema(
    name="l3c_ignoring",
    primary_key="id",
    columns=[Column("id", STRING_TYPE), Column("value", STRING_TYPE)],
    on_conflict="ignore",
    l1_eviction=TEST_L1_TABLE,
)
RAISING = TableSchema(
    name="l3c_raising",
    primary_key="id",
    columns=[Column("id", STRING_TYPE), Column("value", STRING_TYPE)],
    on_conflict="raise",
    l1_eviction=TEST_L1_TABLE,
)
COMPOSITE = TableSchema(
    name="l3c_composite",
    primary_key=("tenant", "id"),
    columns=[Column("tenant", STRING_TYPE), Column("id", STRING_TYPE), Column("value", STRING_TYPE, nullable=True)],
    l1_eviction=TEST_L1_TABLE,
)
ORDERED = TableSchema(
    name="l3c_ordered",
    primary_key="id",
    columns=[Column("id", STRING_TYPE), Column("value", STRING_TYPE), *l2_order_columns()],
    l1_eviction=TEST_L1_TABLE,
)
NULL_SAFE = TableSchema(
    name="l3c_null_safe",
    primary_key="id",
    columns=[Column("id", STRING_TYPE), Column("value", INT_TYPE), Column("date_updated", DATETIMETZ_TYPE)],
    cas_column="date_updated",
    cas_null_safe=True,
    l1_eviction=TEST_L1_TABLE,
)

UNIQUE = TableSchema(
    name="l3c_unique",
    primary_key="id",
    columns=[
        Column("id", STRING_TYPE),
        Column("email", STRING_TYPE),
        Column("tenant", STRING_TYPE),
        Column("handle", STRING_TYPE, nullable=True),
    ],
    indexes=(Index("l3c_unique_tenant_handle", "tenant", "handle", unique=True, where="handle IS NOT NULL"),),
    unique_constraints=(UniqueConstraintDef("l3c_unique_email", ("email",)),),
    l1_eviction=TEST_L1_TABLE,
)

#: every table the contract writes, registered by the ``store`` fixture over empty tables (a store whose tables are
#: made by its own DDL creates their unique indexes and constraints too)
CONTRACT_SCHEMAS: tuple[TableSchema, ...] = (DOCS, ITEMS, IGNORING, RAISING, COMPOSITE, ORDERED, NULL_SAFE, UNIQUE)

_T0 = datetime(2026, 11, 25, 9, 30, 0, 123456, tzinfo=UTC)


def _item(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": uuid.uuid4(),
        "name": "first",
        "payload": {"phase": "clone", "steps": [1, 2]},
        "amount": Decimal("12.50"),
        "flag": True,
        "count": 7,
        "blob": b"\x00\x01",
        "date_created": _T0,
        "date_updated": _T0,
    }
    row.update(overrides)
    return row


def _doc(**overrides: Any) -> dict[str, Any]:
    """a row every store holds as written: text, numbers, a flag, a JSON object and aware instants."""
    row: dict[str, Any] = {
        "id": f"doc-{uuid.uuid4().hex[:12]}",
        "name": "first",
        "payload": {"phase": "clone", "steps": [1, 2]},
        "flag": True,
        "count": 7,
        "date_created": _T0,
        "date_updated": _T0,
    }
    row.update(overrides)
    return row


class DurableStoreContract:
    """the tier every L3 keeps; a subclass supplies the ``store`` fixture, :attr:`duplicate_error` and,
    when its transaction is not ``store.transaction()``, :meth:`begin`."""

    #: what an insert of a key a ``raise`` table already holds raises
    duplicate_error: ClassVar[type[BaseException] | tuple[type[BaseException], ...]] = Exception

    def begin(self, store: Any) -> contextlib.AbstractAsyncContextManager[Any]:
        """the store's transaction: its writes, handed the connection it yields, commit together when it ends.

        :param store: the store
        :ptype store: Any
        :return: an async context manager yielding the ``conn`` its writes are given
        :rtype: contextlib.AbstractAsyncContextManager[Any]
        """
        transaction: contextlib.AbstractAsyncContextManager[Any] = store.transaction()
        return transaction

    async def test_a_key_nothing_holds_reads_as_none(self, store: Any) -> None:
        assert await store.fetch_one("l3c_docs", {"id": "nobody"}) is None

    async def test_a_row_reads_back_as_written(self, store: Any) -> None:
        row = _doc()
        assert await store.upsert("l3c_docs", row) == 1
        read = await store.fetch_one("l3c_docs", {"id": row["id"]})
        assert read == row
        assert read is not None
        assert read["date_updated"].utcoffset() == timedelta(0)

    async def test_a_null_reads_back_as_none(self, store: Any) -> None:
        row = _doc(payload=None, flag=None, count=None)
        await store.upsert("l3c_docs", row)
        assert await store.fetch_one("l3c_docs", {"id": row["id"]}) == row

    async def test_a_whole_row_written_again_replaces_it(self, store: Any) -> None:
        row = _doc()
        await store.upsert("l3c_docs", row)
        later = {**row, "name": "second", "count": 8, "date_updated": _T0 + timedelta(seconds=1)}
        assert await store.upsert("l3c_docs", later) == 1
        assert await store.fetch_one("l3c_docs", {"id": row["id"]}) == later

    async def test_a_delete_removes_the_row_and_a_missing_row_is_not_an_error(self, store: Any) -> None:
        row = _doc()
        await store.upsert("l3c_docs", row)
        await store.delete("l3c_docs", {"id": row["id"]})
        await store.delete("l3c_docs", {"id": row["id"]})
        assert await store.fetch_one("l3c_docs", {"id": row["id"]}) is None

    async def test_a_composite_key_addresses_one_row(self, store: Any) -> None:
        await store.upsert("l3c_composite", {"tenant": "a", "id": "1", "value": "a1"})
        await store.upsert("l3c_composite", {"tenant": "b", "id": "1", "value": "b1"})
        assert await store.fetch_one("l3c_composite", {"tenant": "b", "id": "1"}) == {
            "tenant": "b",
            "id": "1",
            "value": "b1",
        }
        await store.delete("l3c_composite", {"tenant": "a", "id": "1"})
        assert [r["tenant"] for r in await store.scan("l3c_composite")] == ["b"]

    async def test_an_ignoring_write_keeps_the_first_row_and_reports_no_row(self, store: Any) -> None:
        assert await store.upsert("l3c_ignoring", {"id": "k", "value": "first"}, pk=["id"], on_conflict="ignore") == 1
        assert await store.upsert("l3c_ignoring", {"id": "k", "value": "second"}, pk=["id"], on_conflict="ignore") == 0
        assert await store.fetch_one("l3c_ignoring", {"id": "k"}) == {"id": "k", "value": "first"}

    async def test_a_raising_write_refuses_a_second_insert(self, store: Any) -> None:
        await store.upsert("l3c_raising", {"id": "k", "value": "first"}, pk=["id"], on_conflict="raise")
        with pytest.raises(self.duplicate_error):
            await store.upsert("l3c_raising", {"id": "k", "value": "second"}, pk=["id"], on_conflict="raise")
        assert await store.fetch_one("l3c_raising", {"id": "k"}) == {"id": "k", "value": "first"}

    async def test_a_fenced_write_lands_while_the_row_is_as_read(self, store: Any) -> None:
        row = _doc()
        await store.upsert("l3c_docs", row)
        later = {**row, "name": "second", "date_updated": _T0 + timedelta(seconds=1)}
        assert await store.upsert("l3c_docs", later, cas=_T0) == 1
        assert await store.fetch_one("l3c_docs", {"id": row["id"]}) == later

    async def test_a_fenced_write_over_a_changed_row_affects_none(self, store: Any) -> None:
        row = _doc()
        theirs = {**row, "name": "theirs", "date_updated": _T0 + timedelta(seconds=1)}
        await store.upsert("l3c_docs", row)
        await store.upsert("l3c_docs", theirs)
        mine = {**row, "name": "mine", "date_updated": _T0 + timedelta(seconds=2)}
        assert await store.upsert("l3c_docs", mine, cas=_T0) == 0
        assert await store.fetch_one("l3c_docs", {"id": row["id"]}) == theirs

    async def test_a_fenced_write_of_a_deleted_row_affects_none_and_inserts_nothing(self, store: Any) -> None:
        row = _doc()
        await store.upsert("l3c_docs", row)
        await store.delete("l3c_docs", {"id": row["id"]})
        assert await store.upsert("l3c_docs", {**row, "date_updated": _T0 + timedelta(seconds=1)}, cas=_T0) == 0
        assert await store.fetch_one("l3c_docs", {"id": row["id"]}) is None

    async def test_a_scan_answers_every_row_or_those_its_filters_match(self, store: Any) -> None:
        rows = [_doc(name="a"), _doc(name="b"), _doc(name="a", flag=False)]
        for row in rows:
            await store.upsert("l3c_docs", row)
        assert sorted(r["id"] for r in await store.scan("l3c_docs")) == sorted(r["id"] for r in rows)
        assert sorted(r["id"] for r in await store.scan("l3c_docs", {"name": "a"})) == sorted(
            r["id"] for r in rows if r["name"] == "a"
        )
        assert [r["id"] for r in await store.scan("l3c_docs", {"name": "a", "flag": True})] == [rows[0]["id"]]

    async def test_many_rows_are_written_in_one_call(self, store: Any) -> None:
        rows = [{"tenant": "t", "id": str(i), "value": f"v{i}"} for i in range(50)]
        assert await store.upsert_many("l3c_composite", rows, max_rows=16, max_bytes=1_000_000) == 50
        assert len(await store.scan("l3c_composite", {"tenant": "t"})) == 50

    async def test_writes_in_a_transaction_commit_together(self, store: Any) -> None:
        first, second = _doc(), _doc()
        async with self.begin(store) as conn:
            await store.upsert("l3c_docs", first, conn=conn)
            await store.upsert("l3c_docs", second, conn=conn)
            # the transaction reads its own writes
            assert await store.fetch_one("l3c_docs", {"id": first["id"]}, conn=conn) == first
        assert sorted(r["id"] for r in await store.scan("l3c_docs")) == sorted([first["id"], second["id"]])

    async def test_a_transaction_that_raises_writes_nothing(self, store: Any) -> None:
        with pytest.raises(RuntimeError, match="abandoned"):
            async with self.begin(store) as conn:
                await store.upsert("l3c_docs", _doc(), conn=conn)
                raise RuntimeError("abandoned")
        assert await store.scan("l3c_docs") == []


class _Doc(BaseEntity):
    primary_key_field = "id"


class _DocCollection(DurableStoreCollection[_Doc]):
    """the contract's ``l3c_docs`` table as an application's collection over whichever store it is given."""

    primary_key_column = "id"

    @property
    def table_name(self) -> str:
        return "l3c_docs"

    @property
    def entity_class(self) -> type[_Doc]:
        return _Doc


def _l1_tables() -> MetaData:
    metadata = MetaData()
    DOCS.to_sqlalchemy_table(metadata)
    return metadata


class CollectionOverStoreContract:
    """a collection over the store, as an application builds one: an L1 cache in front, and L2 or none.

    Run for each store with and without L2 (``l2`` is ``NO_L2`` or a fake NATS client); every store serves a
    :class:`~threetears.core.collections.DurableStoreCollection`, the git store's own kind of collection.
    """

    @pytest.fixture(params=["no-l2", "l2"])
    def l2(self, request: pytest.FixtureRequest) -> Any:
        return NO_L2 if request.param == "no-l2" else FakeNatsClient()

    @pytest.fixture
    def docs(self, store: Any, l2: Any) -> Iterator[_DocCollection]:
        l1 = SQLiteBackend(f"l3c_{uuid.uuid4().hex[:10]}")
        l1.initialize(_l1_tables())
        registry = CollectionRegistry()
        registry.configure(kv_key_scope="test-principal", l1_backend=l1)
        try:
            yield _DocCollection(registry, DefaultCoreConfig(), store, nats_client=l2)
        finally:
            l1.reset()

    async def _fresh(self, docs: _DocCollection, key: str) -> _Doc | None:
        """the row as the store holds it: dropped from L1 and L2 first, so the read pulls through."""
        docs.evict_from_cache_sync(key)
        await docs.delete_l2_entry(key)
        return await docs.get(key)

    async def test_a_saved_row_is_read_back_through_the_store(self, docs: _DocCollection) -> None:
        row = _doc()
        await docs.save_entity(docs.create(row))
        got = await self._fresh(docs, row["id"])
        assert got is not None
        assert {key: got.to_dict()[key] for key in ("id", "name", "payload", "flag", "count")} == {
            key: row[key] for key in ("id", "name", "payload", "flag", "count")
        }

    async def test_a_save_from_a_stale_read_is_refused(self, docs: _DocCollection) -> None:
        row = _doc()
        await docs.save_entity(docs.create(row))
        first = await self._fresh(docs, row["id"])
        second = await self._fresh(docs, row["id"])
        assert first is not None and second is not None
        first.name = "first wins"
        await docs.save_entity(first)
        second.name = "second loses"
        with pytest.raises(ConcurrentModificationError):
            await docs.save_entity(second)
        stored = await self._fresh(docs, row["id"])
        assert stored is not None
        assert stored.name == "first wins"

    async def test_a_delete_reaches_the_store(self, docs: _DocCollection) -> None:
        row = _doc()
        await docs.save_entity(docs.create(row))
        await docs.delete(row["id"])
        assert await self._fresh(docs, row["id"]) is None

    @contextlib.asynccontextmanager
    async def caller_transaction(self, store: Any) -> AsyncIterator[Any]:
        """a caller's transaction a collection's write joins (``conn=``): an acquired connection under
        ``CallerTransaction``, as an application opens one over a database store.

        :param store: the store
        :ptype store: Any
        :return: the connection the writes are handed
        :rtype: AsyncIterator[Any]
        """
        async with store.acquire() as conn, CallerTransaction(conn):
            yield conn

    async def test_a_delete_in_a_callers_transaction_commits_with_it(self, docs: _DocCollection, store: Any) -> None:
        row = _doc()
        await docs.save_entity(docs.create(row))
        async with self.caller_transaction(store) as conn:
            await docs.delete(row["id"], conn=conn)
        assert await self._fresh(docs, row["id"]) is None

    async def test_a_delete_in_a_callers_transaction_that_raises_deletes_nothing(
        self, docs: _DocCollection, store: Any
    ) -> None:
        row = _doc()
        await docs.save_entity(docs.create(row))
        with pytest.raises(RuntimeError, match="abandoned"):
            async with self.caller_transaction(store) as conn:
                await docs.delete(row["id"], conn=conn)
                raise RuntimeError("abandoned")
        assert await self._fresh(docs, row["id"]) is not None

    async def test_a_list_reads_the_rows_its_filters_match(self, docs: _DocCollection) -> None:
        rows = [_doc(name="a"), _doc(name="b"), _doc(name="a")]
        for row in rows:
            await docs.save_entity(docs.create(row))
        listed = await docs.list({"name": "a"})
        assert sorted(entity.id for entity in listed) == sorted(r["id"] for r in rows if r["name"] == "a")


class SqlStoreContract(DurableStoreContract):
    """the tier the schema-registering stores keep on top (Postgres, SQLite)."""

    duplicate_error = asyncpg.UniqueViolationError

    # ── what the table admits ──────────────────────────────────────────────────────────────────────────────────────
    async def test_a_unique_constraint_refuses_a_second_row_holding_its_value(self, store: Any) -> None:
        await store.upsert("l3c_unique", {"id": "a", "email": "x@example.org", "tenant": "t", "handle": None})
        with pytest.raises(asyncpg.UniqueViolationError):
            await store.upsert("l3c_unique", {"id": "b", "email": "x@example.org", "tenant": "t", "handle": None})
        assert [row["id"] for row in await store.scan("l3c_unique")] == ["a"]

    async def test_a_partial_unique_index_refuses_only_rows_its_where_covers(self, store: Any) -> None:
        await store.upsert("l3c_unique", {"id": "a", "email": "a@example.org", "tenant": "t", "handle": None})
        # outside the index's WHERE: another row with no handle is admitted
        await store.upsert("l3c_unique", {"id": "b", "email": "b@example.org", "tenant": "t", "handle": None})
        await store.upsert("l3c_unique", {"id": "c", "email": "c@example.org", "tenant": "t", "handle": "h"})
        with pytest.raises(asyncpg.UniqueViolationError):
            await store.upsert("l3c_unique", {"id": "d", "email": "d@example.org", "tenant": "t", "handle": "h"})
        assert sorted(row["id"] for row in await store.scan("l3c_unique")) == ["a", "b", "c"]

    async def test_a_required_column_refuses_null(self, store: Any) -> None:
        with pytest.raises(asyncpg.NotNullViolationError):
            await store.upsert("l3c_raising", {"id": "n", "value": None})
        assert await store.fetch_one("l3c_raising", {"id": "n"}) is None

    # ── reads and plain writes ─────────────────────────────────────────────────────────────────────────────────────
    async def test_a_row_reads_back_with_the_types_it_was_written_with(self, store: Any) -> None:
        row = _item()
        assert await store.upsert("l3c_items", row) == 1
        read = await store.fetch_one("l3c_items", {"id": row["id"]})
        assert read == row
        assert isinstance(read["id"], uuid.UUID)
        assert read["date_updated"].tzinfo is not None
        assert read["date_updated"].utcoffset() == timedelta(0)
        assert isinstance(read["amount"], Decimal)
        assert read["flag"] is True

    async def test_nulls_read_back_as_none(self, store: Any) -> None:
        row = _item(payload=None, amount=None, flag=None, count=None, blob=None)
        await store.upsert("l3c_items", row)
        assert await store.fetch_one("l3c_items", {"id": row["id"]}) == row

    async def test_a_naive_instant_is_taken_as_utc(self, store: Any) -> None:
        row = _item(date_updated=_T0.replace(tzinfo=None))
        await store.upsert("l3c_items", row)
        read = await store.fetch_one("l3c_items", {"id": row["id"]})
        assert read is not None
        assert read["date_updated"] == _T0

    async def test_an_update_rewrites_the_columns_it_names_and_keeps_the_rest(self, store: Any) -> None:
        row = _item()
        await store.upsert("l3c_items", row)
        later = _T0 + timedelta(seconds=1)
        # every required column is named, as a save names them; the nullable ones it leaves out keep their values
        named = {key: value for key, value in row.items() if key not in ("payload", "amount", "blob")}
        assert await store.upsert("l3c_items", {**named, "name": "second", "date_updated": later}) == 1
        read = await store.fetch_one("l3c_items", {"id": row["id"]})
        assert read == {**row, "name": "second", "date_updated": later}

    async def test_an_immutable_column_keeps_its_first_value(self, store: Any) -> None:
        row = _item()
        await store.upsert("l3c_items", row)
        await store.upsert("l3c_items", {**row, "date_created": _T0 + timedelta(days=1)})
        read = await store.fetch_one("l3c_items", {"id": row["id"]})
        assert read is not None
        assert read["date_created"] == _T0

    # ── conflict modes ─────────────────────────────────────────────────────────────────────────────────────────────
    async def test_an_ignoring_table_keeps_the_first_row_and_reports_no_row(self, store: Any) -> None:
        assert await store.upsert("l3c_ignoring", {"id": "k", "value": "first"}) == 1
        assert await store.upsert("l3c_ignoring", {"id": "k", "value": "second"}) == 0
        assert await store.fetch_one("l3c_ignoring", {"id": "k"}) == {"id": "k", "value": "first"}

    async def test_a_raising_table_refuses_a_second_insert(self, store: Any) -> None:
        await store.upsert("l3c_raising", {"id": "k", "value": "first"})
        with pytest.raises(self.duplicate_error):
            await store.upsert("l3c_raising", {"id": "k", "value": "second"})
        assert await store.fetch_one("l3c_raising", {"id": "k"}) == {"id": "k", "value": "first"}

    # ── compare-and-swap ───────────────────────────────────────────────────────────────────────────────────────────
    async def test_a_null_safe_fence_lets_one_first_writer_in(self, store: Any) -> None:
        first = {"id": "c", "value": 1, "date_updated": _T0}
        second = {"id": "c", "value": 2, "date_updated": _T0 + timedelta(seconds=1)}
        assert await store.upsert("l3c_null_safe", first, cas=None) == 1
        # the second first-writer read no row, so it fences on NULL, which the stored row no longer is
        assert await store.upsert("l3c_null_safe", second, cas=None) == 0
        assert await store.upsert("l3c_null_safe", second, cas=_T0) == 1
        assert await store.fetch_one("l3c_null_safe", {"id": "c"}) == second

    async def test_an_ordered_upsert_lands_only_over_an_older_order(self, store: Any) -> None:
        epoch = _T0

        def row(value: str, revision: int) -> dict[str, Any]:
            return {"id": "o", "value": value, "l2_epoch": epoch, "l2_revision": revision}

        assert await store.upsert_ordered("l3c_ordered", row("five", 5)) == 1
        assert await store.upsert_ordered("l3c_ordered", row("four", 4)) == 0
        assert await store.upsert_ordered("l3c_ordered", row("five again", 5)) == 0
        assert await store.upsert_ordered("l3c_ordered", row("six", 6)) == 1
        read = await store.fetch_one("l3c_ordered", {"id": "o"})
        assert read is not None
        assert (read["value"], read["l2_revision"]) == ("six", 6)

    async def test_an_ordered_upsert_lands_over_a_row_with_no_order(self, store: Any) -> None:
        await store.upsert("l3c_ordered", {"id": "o", "value": "plain", "l2_epoch": None, "l2_revision": None})
        assert (
            await store.upsert_ordered("l3c_ordered", {"id": "o", "value": "won", "l2_epoch": _T0, "l2_revision": 1})
            == 1
        )

    # ── many rows ──────────────────────────────────────────────────────────────────────────────────────────────────
    async def test_a_scan_of_typed_rows_matches_typed_filters(self, store: Any) -> None:
        rows = [_item(name="a"), _item(name="b"), _item(name="a", flag=None)]
        for row in rows:
            await store.upsert("l3c_items", row)
        assert sorted(r["id"] for r in await store.scan("l3c_items")) == sorted(r["id"] for r in rows)
        assert sorted(r["id"] for r in await store.scan("l3c_items", {"name": "a"})) == sorted(
            r["id"] for r in rows if r["name"] == "a"
        )
        assert [r["id"] for r in await store.scan("l3c_items", {"name": "a", "flag": True})] == [rows[0]["id"]]

    async def test_a_scan_refuses_a_column_the_table_does_not_hold(self, store: Any) -> None:
        with pytest.raises(ValueError, match="no column"):
            await store.scan("l3c_items", {"colour": "red"})

    async def test_a_truncate_empties_the_table(self, store: Any) -> None:
        await store.upsert("l3c_ignoring", {"id": "k", "value": "v"})
        await store.truncate("l3c_ignoring")
        await store.truncate("l3c_ignoring")
        assert await store.scan("l3c_ignoring") == []

    # ── raw SQL ────────────────────────────────────────────────────────────────────────────────────────────────────
    async def test_a_plain_statement_runs_and_reports_its_rows(self, store: Any) -> None:
        await store.upsert("l3c_ignoring", {"id": "a", "value": "1"})
        tag = await store.execute("UPDATE l3c_ignoring SET value = $1 WHERE id = $2", "9", "a")
        assert tag.split()[-1] == "1"
        assert await store.fetchval("SELECT value FROM l3c_ignoring WHERE id = $1", "a") == "9"
        assert await store.fetchrow("SELECT value FROM l3c_ignoring WHERE id = $1", "none") is None
