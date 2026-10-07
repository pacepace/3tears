"""Many rows saved as multi-row upserts on a caller's transaction: batched to fit, rolled back whole."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.core.backends.schema_sql import build_bulk_insert_sql, build_insert_params, bulk_batches, json_default
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


def _collection() -> Any:
    return collection_for_schema(SCHEMA)(CollectionRegistry(), DefaultCoreConfig(), None)


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


async def test_every_row_written_is_settled_when_the_transaction_ends() -> None:
    collection, conn = _collection(), _Conn()
    async with CallerTransaction(conn) as transaction:
        await collection.save_rows(_rows(3), conn=conn)
        enrolled = [key for _, key in transaction.enrolled]
    assert enrolled == [("sn_NC1", "00000"), ("sn_NC1", "00001"), ("sn_NC1", "00002")]


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
