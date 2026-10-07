"""Many rows saved as multi-row upserts on a caller's transaction: batched to fit, rolled back whole."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.core.backends.schema_sql import (
    build_bulk_delete_sql,
    build_bulk_insert_sql,
    build_insert_params,
    bulk_batches,
    json_default,
)
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity

SCHEMA = TableSchema(
    name="results",
    primary_key=("office_key", "geo_id"),
    columns=[
        Column("office_key", STRING_TYPE),
        Column("geo_id", STRING_TYPE),
        Column("votes", INT_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)


class _ResultRow(BaseEntity):
    """a result row: its own id is the geography, within the race its key leads with."""

    primary_key_field = "geo_id"


class _Conn:
    """an asyncpg-shaped connection: its transaction keeps statements only when it commits."""

    def __init__(self, *, fail_on: int | None = None) -> None:
        self.fail_on = fail_on
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.committed: list[tuple[str, tuple[Any, ...]]] = []
        self.rolled_back = False

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        try:
            yield
        except BaseException:
            self.rolled_back = True
            raise
        self.committed = list(self.executed)

    def transaction(self, **options: Any) -> Any:
        return self._transaction()

    async def execute(self, sql: str, *params: Any) -> str:
        if self.fail_on == len(self.executed) + 1:
            raise ConnectionError("statement timed out")
        self.executed.append((sql, params))
        return f"INSERT 0 {sql.count('),(') + 1}"


class _Pool:
    """a raw transport the collection wraps as its SQL store; writes go to the caller's connection."""


def _collection(schema: TableSchema = SCHEMA) -> Any:
    registry = CollectionRegistry()
    entity = _ResultRow if len(schema.pk_columns) > 1 else None
    collection = collection_for_schema(schema, entity_class=entity)(registry, DefaultCoreConfig(), None)
    collection.l3_pool = _Pool()
    return collection


def _rows(count: int) -> list[dict[str, Any]]:
    return [{"office_key": "sn_NC1", "geo_id": f"{i:05d}", "votes": i} for i in range(count)]


def test_one_statement_inserts_every_row_and_updates_a_row_already_held() -> None:
    sql = build_bulk_insert_sql(SCHEMA, rows=2)
    assert sql.startswith("INSERT INTO results (office_key, geo_id, votes, date_created, date_updated) VALUES ")
    assert "($1, $2, $3, $4, $5), ($6, $7, $8, $9, $10)" in sql
    assert "ON CONFLICT (office_key, geo_id) DO UPDATE SET" in sql
    assert "votes = EXCLUDED.votes" in sql
    # a row's creation time stays the one it was first written with
    assert "date_created = EXCLUDED" not in sql


async def test_save_rows_writes_them_in_one_statement_on_the_caller_s_transaction() -> None:
    collection, conn = _collection(), _Conn()
    async with CallerTransaction(conn):
        written = await collection.save_rows(_rows(3), conn=conn)
    assert written == 3
    [(sql, params)] = conn.committed
    assert sql == build_bulk_insert_sql(SCHEMA, rows=3)
    assert params[0:3] == ("sn_NC1", "00000", 0)
    assert isinstance(params[3], datetime) and isinstance(params[4], datetime)


async def test_a_batch_that_fails_rolls_the_whole_transaction_back() -> None:
    collection, conn = _collection(), _Conn(fail_on=2)
    with pytest.raises(ConnectionError):
        async with CallerTransaction(conn):
            await collection.save_rows(_rows(5), conn=conn, max_rows=2)
    assert conn.rolled_back
    assert conn.committed == []


async def test_rows_are_split_so_no_statement_passes_the_row_limit() -> None:
    collection, conn = _collection(), _Conn()
    async with CallerTransaction(conn):
        await collection.save_rows(_rows(5), conn=conn, max_rows=2)
    assert [sql.count("), (") + 1 for sql, _ in conn.committed] == [2, 2, 1]


async def test_rows_are_split_so_no_statement_passes_the_byte_limit() -> None:
    collection, conn = _collection(), _Conn()
    stamped = {**_rows(1)[0], "date_created": datetime.now(UTC), "date_updated": datetime.now(UTC)}
    one_row = len(json.dumps(build_insert_params(SCHEMA, stamped), default=json_default))
    async with CallerTransaction(conn):
        await collection.save_rows(_rows(4), conn=conn, max_bytes=one_row * 2 + 10)
    assert [len(params) // 5 for _, params in conn.committed] == [2, 2]


def test_a_batch_never_holds_more_parameters_than_a_statement_may_bind() -> None:
    batches = bulk_batches([[1, 2, 3]] * 10, max_rows=100, max_bytes=10_000, max_params=7)
    assert [len(batch) for batch in batches] == [2, 2, 2, 2, 2]


def test_a_row_larger_than_the_byte_limit_is_a_batch_of_its_own() -> None:
    batches = bulk_batches([["x" * 50], ["y"], ["z"]], max_rows=100, max_bytes=20, max_params=100)
    assert [len(batch) for batch in batches] == [1, 2]


async def test_save_rows_refuses_a_connection_no_caller_transaction_opened() -> None:
    with pytest.raises(ValueError, match="CallerTransaction"):
        await _collection().save_rows(_rows(1), conn=_Conn())


async def test_every_row_written_is_settled_once_when_the_transaction_ends() -> None:
    """The keys are settled together after the transaction, one call per collection, not one per row."""
    collection, conn = _collection(), _Conn()
    settled: list[list[Any]] = []

    async def settle(keys: Any) -> None:
        settled.append(list(keys))

    collection.invalidate_cache_many = settle
    async with CallerTransaction(conn):
        await collection.save_rows(_rows(3), conn=conn)
        assert settled == []
    assert settled == [[("sn_NC1", "00000"), ("sn_NC1", "00001"), ("sn_NC1", "00002")]]


DEFAULTED = TableSchema(
    name="loads",
    primary_key="source",
    columns=[
        Column("source", STRING_TYPE),
        Column("rows", INT_TYPE, nullable=True),
        Column("loaded_at", DATETIMETZ_TYPE, nullable=True, server_default="now()"),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)


async def test_a_column_every_row_leaves_to_its_server_default_is_left_out_of_sql_and_params_alike() -> None:
    collection, conn = _collection(DEFAULTED), _Conn()
    async with CallerTransaction(conn):
        await collection.save_rows([{"source": "a", "rows": 1}, {"source": "b", "rows": 2}], conn=conn)
    [(sql, params)] = conn.committed
    assert "loaded_at" not in sql
    assert sql.startswith("INSERT INTO loads (source, rows, date_created, date_updated) VALUES ($1, $2, $3, $4), ")
    assert len(params) == 8
    assert params[0:2] == ("a", 1) and params[4:6] == ("b", 2)


async def test_rows_that_disagree_on_a_server_default_column_are_refused_naming_the_table() -> None:
    collection, conn = _collection(DEFAULTED), _Conn()
    rows = [{"source": "a"}, {"source": "b", "loaded_at": datetime.now(UTC)}]
    with pytest.raises(ValueError, match="loads"):
        async with CallerTransaction(conn):
            await collection.save_rows(rows, conn=conn)
    assert conn.executed == []


async def test_a_row_missing_a_key_column_names_the_table_and_the_column() -> None:
    collection, conn = _collection(), _Conn()
    with pytest.raises(KeyError, match="results.*geo_id"):
        async with CallerTransaction(conn):
            await collection.save_rows([{"office_key": "sn_NC1", "votes": 1}], conn=conn)


class _BulkStore:
    """a non-SQL store that saves many rows at once."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, Any]], Any]] = []

    async def fetch_one(self, table: str, pk: Any, *, conn: Any = None) -> None:
        return None

    async def upsert(self, table: str, row: Any, **kwargs: Any) -> int:
        raise AssertionError("a store that saves many rows at once is not asked one at a time")

    async def upsert_many(self, table: str, rows: Any, *, max_rows: int, max_bytes: int, conn: Any = None) -> int:
        self.calls.append((table, [dict(r) for r in rows], conn))
        return len(rows)

    async def delete(self, table: str, pk: Any, *, conn: Any = None) -> None:
        return None

    async def scan(self, table: str, filters: Any = None) -> list[dict[str, Any]]:
        return []


class _RowStore(_BulkStore):
    """a non-SQL store with no bulk save: it is written row by row."""

    upsert_many = None  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, dict[str, Any], Any]] = []

    async def upsert(self, table: str, row: Any, **kwargs: Any) -> int:
        self.rows.append((table, dict(row), kwargs.get("conn")))
        return 1


async def test_save_rows_goes_through_the_collection_s_store() -> None:
    collection, conn, store = _collection(), _Conn(), _BulkStore()
    collection.l3_pool = store
    async with CallerTransaction(conn):
        assert await collection.save_rows(_rows(3), conn=conn) == 3
    [(table, rows, used)] = store.calls
    assert table == "results" and used is conn
    assert [r["geo_id"] for r in rows] == ["00000", "00001", "00002"]
    assert conn.executed == []


async def test_a_store_with_no_bulk_save_is_written_a_row_at_a_time() -> None:
    collection, conn, store = _collection(), _Conn(), _RowStore()
    collection.l3_pool = store
    async with CallerTransaction(conn):
        assert await collection.save_rows(_rows(2), conn=conn) == 2
    assert [(t, r["geo_id"], c is conn) for t, r, c in store.rows] == [
        ("results", "00000", True),
        ("results", "00001", True),
    ]


async def test_no_rows_writes_nothing() -> None:
    collection, conn = _collection(), _Conn()
    async with CallerTransaction(conn):
        assert await collection.save_rows([], conn=conn) == 0
    assert conn.committed == []


async def test_two_rows_with_one_key_are_refused_before_anything_is_written() -> None:
    """One statement cannot upsert a key twice; the database would refuse the whole batch."""
    collection, conn = _collection(), _Conn()
    rows = [*_rows(2), _rows(1)[0]]
    with pytest.raises(ValueError, match="share a key"):
        async with CallerTransaction(conn):
            await collection.save_rows(rows, conn=conn)
    assert conn.executed == []


class _Buffer:
    """a write buffer: present, so a write-behind collection really defers."""


async def test_a_collection_with_no_durable_store_is_refused() -> None:
    collection, conn = _collection(), _Conn()
    collection.l3_pool = None
    with pytest.raises(ValueError, match="no durable store"):
        async with CallerTransaction(conn):
            await collection.save_rows(_rows(1), conn=conn)


async def test_a_collection_caching_absences_is_refused() -> None:
    class _Absences(type(_collection())):  # type: ignore[misc]
        negative_cache_max_age = timedelta(seconds=60)

    collection, conn = _Absences(CollectionRegistry(), DefaultCoreConfig(), None), _Conn()
    collection.l3_pool = _Pool()
    with pytest.raises(ValueError, match="caches absences"):
        async with CallerTransaction(conn):
            await collection.save_rows(_rows(1), conn=conn)
    assert conn.executed == []


async def test_a_collection_deferring_its_l3_writes_is_refused() -> None:
    class _Deferred(type(_collection())):  # type: ignore[misc]
        l3_write_policy = "write_behind"

    collection = _Deferred(CollectionRegistry(), DefaultCoreConfig(), None, write_buffer=_Buffer())  # type: ignore[arg-type]
    collection.l3_pool = _Pool()
    conn = _Conn()
    with pytest.raises(ValueError, match="defers its L3 writes"):
        async with CallerTransaction(conn):
            await collection.save_rows(_rows(1), conn=conn)
    assert conn.executed == []


async def test_a_collection_fencing_with_a_null_safe_cas_is_refused() -> None:
    fenced = TableSchema(
        name="counters",
        primary_key="id",
        columns=[
            Column("id", STRING_TYPE),
            Column("count", INT_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        cas_column="date_updated",
        cas_null_safe=True,
    )
    collection, conn = _collection(fenced), _Conn()
    with pytest.raises(ValueError, match="null-safe CAS"):
        async with CallerTransaction(conn):
            await collection.save_rows([{"id": "a", "count": 1}], conn=conn)
    assert conn.executed == []


async def test_a_sql_store_refuses_a_bulk_upsert_to_a_table_whose_schema_it_does_not_hold() -> None:
    from threetears.core.backends.sql import SqlL3Backend

    store = SqlL3Backend(_Pool())
    with pytest.raises(ValueError, match="schema of 'nowhere' registered"):
        await store.upsert_many("nowhere", [{"id": 1}], max_rows=10, max_bytes=1000)


async def test_save_rows_answers_how_many_rows_it_submitted() -> None:
    collection, conn = _collection(), _Conn()
    async with CallerTransaction(conn):
        assert await collection.save_rows(_rows(4), conn=conn, max_rows=3) == 4


# -- delete_rows: the bulk write's other half -------------------------------------------------------


def test_one_statement_deletes_every_key_named() -> None:
    sql = build_bulk_delete_sql(SCHEMA, rows=2)
    assert sql == "DELETE FROM results WHERE (office_key, geo_id) IN (($1, $2), ($3, $4))"


async def test_delete_rows_deletes_them_in_one_statement_on_the_caller_s_transaction() -> None:
    collection, conn = _collection(), _Conn()
    async with CallerTransaction(conn):
        deleted = await collection.delete_rows([("sn_NC1", "00001"), ("sn_NC1", "00007")], conn=conn)
    assert deleted == 2
    [(sql, params)] = conn.committed
    assert sql == build_bulk_delete_sql(SCHEMA, rows=2)
    assert params == ("sn_NC1", "00001", "sn_NC1", "00007")


async def test_keys_to_delete_are_split_so_no_statement_passes_the_row_limit() -> None:
    collection, conn = _collection(), _Conn()
    keys = [("sn_NC1", f"{i:05d}") for i in range(5)]
    async with CallerTransaction(conn):
        await collection.delete_rows(keys, conn=conn, max_rows=2)
    assert [len(params) // 2 for _, params in conn.committed] == [2, 2, 1]


async def test_a_failing_delete_rolls_the_whole_transaction_back() -> None:
    collection, conn = _collection(), _Conn(fail_on=2)
    with pytest.raises(ConnectionError):
        async with CallerTransaction(conn):
            await collection.delete_rows([("sn_NC1", f"{i:05d}") for i in range(4)], conn=conn, max_rows=2)
    assert conn.rolled_back
    assert conn.committed == []


async def test_every_key_deleted_is_settled_once_when_the_transaction_ends() -> None:
    collection, conn = _collection(), _Conn()
    settled: list[list[Any]] = []

    async def settle(keys: Any) -> None:
        settled.append(list(keys))

    collection.invalidate_cache_many = settle
    async with CallerTransaction(conn):
        await collection.delete_rows([("sn_NC1", "00001")], conn=conn)
        assert settled == []
    assert settled == [[("sn_NC1", "00001")]]


async def test_a_key_of_the_wrong_width_is_refused_naming_the_table() -> None:
    collection, conn = _collection(), _Conn()
    with pytest.raises(ValueError, match="results"):
        async with CallerTransaction(conn):
            await collection.delete_rows([("sn_NC1",)], conn=conn)
    assert conn.executed == []


async def test_delete_rows_refuses_a_connection_no_caller_transaction_opened() -> None:
    with pytest.raises(ValueError, match="CallerTransaction"):
        await _collection().delete_rows([("sn_NC1", "00001")], conn=_Conn())


async def test_no_keys_deletes_nothing() -> None:
    collection, conn = _collection(), _Conn()
    async with CallerTransaction(conn):
        assert await collection.delete_rows([], conn=conn) == 0
    assert conn.committed == []


async def test_a_store_that_cannot_delete_many_is_asked_a_key_at_a_time() -> None:
    collection, conn, store = _collection(), _Conn(), _DeletingStore()
    collection.l3_pool = store
    async with CallerTransaction(conn):
        assert await collection.delete_rows([("sn_NC1", "00001"), ("sn_NC1", "00002")], conn=conn) == 2
    assert store.deleted == [
        ("results", {"office_key": "sn_NC1", "geo_id": "00001"}, True),
        ("results", {"office_key": "sn_NC1", "geo_id": "00002"}, True),
    ]


class _DeletingStore(_BulkStore):
    """a non-SQL store with no bulk delete: it deletes a key at a time, on the caller's transaction."""

    def __init__(self) -> None:
        super().__init__()
        self.deleted: list[tuple[str, dict[str, Any], bool]] = []
        self._conn: Any = None

    async def delete(self, table: str, pk: Any, *, conn: Any = None) -> None:
        self.deleted.append((table, dict(pk), conn is not None))
