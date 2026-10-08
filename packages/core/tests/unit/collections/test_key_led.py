"""Reads and deletes by many keys, every statement led by the key, so each stays bounded on a hash-sharded table.

On YugabyteDB a statement no key leads reads every row of the table (``ORDER BY`` the key included,
the key being hashed), and the L3 rail cuts every answer at its row cap without saying so. So a read
by many leading-key values goes a batch of values a statement (``lead = ANY($1)``), is read again in
halves when an answer may have been cut, pages one value's rows by the rest of its key when that
value alone fills the cap, and a delete by full keys names one varying key column as ``= ANY($1)``
with the rest of the key fixed.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from threetears.core.backends.schema_sql import build_key_led_delete_sql
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity

LAYER = TableSchema(
    name="layer",
    primary_key=("feature_id", "source_version"),
    columns=[
        Column("feature_id", STRING_TYPE),
        Column("source_version", BIGINT_TYPE),
        Column("name", STRING_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)

ONE_KEY = TableSchema(
    name="widgets",
    primary_key="id",
    columns=[
        Column("id", STRING_TYPE),
        Column("doc", JSONB_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)

#: a statement is key-led when its WHERE opens on the leading key column, as an array or one value
_LED = re.compile(r"WHERE feature_id = (ANY\(\$1::text\[\]\)|\$1)")


class _Shape(BaseEntity):
    primary_key_field = "source_version"


class _Table:
    """an L3 rail over one table's rows: it answers the two key-led reads and cuts every answer at ``cap``."""

    def __init__(self, rows: list[dict[str, Any]], *, cap: int) -> None:
        self.rows = rows
        self.cap = cap
        self.fetched: list[tuple[str, tuple[Any, ...]]] = []

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        yield

    def transaction(self, **options: Any) -> Any:
        return self._transaction()

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.fetched.append((sql, params))
        columns = [c.strip() for c in sql.split("SELECT ", 1)[1].split(" FROM ", 1)[0].split(",")]
        if "= ANY($1" in sql:
            wanted = set(params[0])
            held = [r for r in self.rows if r["feature_id"] in wanted]
        else:
            held = sorted((r for r in self.rows if r["feature_id"] == params[0]), key=lambda r: r["source_version"])
            if len(params) > 1:
                held = [r for r in held if r["source_version"] > params[1]]
            limit = re.search(r"LIMIT (\d+)", sql)
            assert limit is not None, sql
            held = held[: int(limit.group(1))]
        return [{c: r[c] for c in columns} for r in held[: self.cap]]


class _Pool:
    """a raw transport the collection wraps as its SQL store; statements go to the connection given."""


def _layer(cap: int = 1000, **kwargs: Any) -> Any:
    cls = collection_for_schema(LAYER, entity_class=_Shape)
    cls.L3_ROW_CAP = cap
    collection = cls(CollectionRegistry(), DefaultCoreConfig(), None)
    collection.l3_pool = _Pool()
    return collection


def _rows(features: int, generations: int) -> list[dict[str, Any]]:
    return [
        {"feature_id": f"f{f:04d}", "source_version": g, "name": f"n{f}"}
        for f in range(features)
        for g in range(generations)
    ]


def _keys(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    return sorted((r["feature_id"], r["source_version"]) for r in rows)


# -- reads: the keys held for many leading-key values ----------------------------------------------


async def test_the_keys_held_for_many_values_are_read_by_the_leading_key() -> None:
    table = _Table(_rows(5, 2), cap=1000)
    held = await _layer().read_rows_led_by(["f0001", "f0003", "nope"], conn=table)

    assert _keys(held) == [("f0001", 0), ("f0001", 1), ("f0003", 0), ("f0003", 1)]
    [(sql, params)] = table.fetched
    assert sql == "SELECT feature_id, source_version FROM layer WHERE feature_id = ANY($1::text[])"
    assert params == (["f0001", "f0003", "nope"],)


async def test_values_go_a_batch_a_statement_each_named_once() -> None:
    table = _Table(_rows(10, 1), cap=1000)
    values = [f"f{i:04d}" for i in range(10)] + ["f0000"]

    held = await _layer().read_rows_led_by(values, max_values=4, conn=table)

    assert len(held) == 10
    assert [len(params[0]) for _, params in table.fetched] == [4, 4, 2]


async def test_an_answer_that_reaches_the_cap_is_read_again_in_halves() -> None:
    # eight features of three generations: 24 rows, and the rail answers at most 10 a statement
    table = _Table(_rows(8, 3), cap=10)

    held = await _layer(cap=10).read_rows_led_by([f"f{i:04d}" for i in range(8)], conn=table)

    assert _keys(held) == _keys(_rows(8, 3))
    assert len(table.fetched) > 1
    assert all(_LED.search(sql) for sql, _ in table.fetched)


async def test_one_value_holding_more_rows_than_the_cap_is_paged_by_the_rest_of_its_key() -> None:
    table = _Table(_rows(2, 12), cap=5)

    held = await _layer(cap=5).read_rows_led_by(["f0000", "f0001"], conn=table)

    assert _keys(held) == _keys(_rows(2, 12))
    paged = [sql for sql, _ in table.fetched if "= ANY(" not in sql]
    assert paged and all("ORDER BY source_version LIMIT 5" in sql for sql in paged)
    assert all(_LED.search(sql) for sql, _ in table.fetched)


async def test_named_columns_are_read_beside_the_key() -> None:
    table = _Table(_rows(2, 1), cap=1000)

    held = await _layer().read_rows_led_by(["f0001"], columns=["name"], conn=table)

    assert held == [{"feature_id": "f0001", "source_version": 0, "name": "n1"}]
    assert table.fetched[0][0].startswith("SELECT feature_id, source_version, name FROM layer")


async def test_a_column_the_table_does_not_have_is_refused_naming_the_table() -> None:
    with pytest.raises(ValueError, match="layer"):
        await _layer().read_rows_led_by(["f0001"], columns=["nope"], conn=_Table([], cap=1000))


async def test_no_values_reads_nothing() -> None:
    table = _Table(_rows(2, 1), cap=1000)
    assert await _layer().read_rows_led_by([], conn=table) == []
    assert table.fetched == []


async def test_a_store_without_the_key_led_read_is_scanned_a_value_at_a_time() -> None:
    class _ScanStore:
        """a non-SQL store: it answers equality scans only."""

        def __init__(self) -> None:
            self.scans: list[dict[str, Any]] = []

        async def fetch_one(self, table: str, pk: Any, *, conn: Any = None) -> None:
            return None

        async def upsert(self, table: str, row: Any, **kwargs: Any) -> int:
            return 1

        async def delete(self, table: str, pk: Any, *, conn: Any = None) -> None:
            return None

        async def scan(self, table: str, filters: Any = None) -> list[dict[str, Any]]:
            self.scans.append(dict(filters))
            return [r for r in _rows(3, 2) if r["feature_id"] == filters["feature_id"]]

    collection, store = _layer(), _ScanStore()
    collection.l3_pool = store

    held = await collection.read_rows_led_by(["f0002", "f0002", "f0000"])

    assert store.scans == [{"feature_id": "f0002"}, {"feature_id": "f0000"}]
    assert held[0] == {"feature_id": "f0002", "source_version": 0}
    assert _keys(held) == [("f0000", 0), ("f0000", 1), ("f0002", 0), ("f0002", 1)]


# -- deletes: by full keys, one key column varying -------------------------------------------------


class _Conn:
    """an asyncpg-shaped connection recording the statements its transaction commits."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        yield

    def transaction(self, **options: Any) -> Any:
        return self._transaction()

    async def execute(self, sql: str, *params: Any) -> str:
        self.executed.append((sql, params))
        return "DELETE 1"


def test_a_key_led_delete_names_one_key_column_as_an_array_and_fixes_the_rest() -> None:
    assert build_key_led_delete_sql(LAYER, varying="feature_id") == (
        "DELETE FROM layer WHERE feature_id = ANY($1::text[]) AND source_version = $2"
    )
    assert build_key_led_delete_sql(LAYER, varying="source_version") == (
        "DELETE FROM layer WHERE source_version = ANY($1::bigint[]) AND feature_id = $2"
    )
    assert build_key_led_delete_sql(ONE_KEY, varying="id") == "DELETE FROM widgets WHERE id = ANY($1::text[])"


def test_a_key_column_with_no_array_form_is_refused() -> None:
    schema = TableSchema(name="docs", primary_key="doc", columns=[Column("doc", JSONB_TYPE)])
    with pytest.raises(ValueError, match="docs"):
        build_key_led_delete_sql(schema, varying="doc")


async def test_a_generation_is_deleted_by_its_features_a_batch_a_statement() -> None:
    collection, conn = _layer(), _Conn()
    keys = [(f"f{i:04d}", 7) for i in range(5)]

    async with CallerTransaction(conn):
        assert await collection.delete_rows(keys, conn=conn, max_rows=2) == 5

    assert [sql for sql, _ in conn.executed] == [build_key_led_delete_sql(LAYER, varying="feature_id")] * 3
    assert [params for _, params in conn.executed] == [
        (["f0000", "f0001"], 7),
        (["f0002", "f0003"], 7),
        (["f0004"], 7),
    ]


async def test_keys_sharing_their_lead_vary_the_rest_of_the_key() -> None:
    collection, conn = _layer(), _Conn()

    async with CallerTransaction(conn):
        await collection.delete_rows([("f0001", 1), ("f0001", 2), ("f0001", 3)], conn=conn)

    assert conn.executed == [(build_key_led_delete_sql(LAYER, varying="source_version"), ([1, 2, 3], "f0001"))]


async def test_mixed_keys_go_one_group_of_the_fixed_columns_at_a_time() -> None:
    collection, conn = _layer(), _Conn()
    keys = [("a", 1), ("b", 1), ("c", 2), ("a", 2)]

    async with CallerTransaction(conn):
        assert await collection.delete_rows(keys, conn=conn) == 4

    deleted = sorted((value, params[1]) for _, params in conn.executed for value in params[0])
    assert deleted == sorted(keys)
    assert all(sql == build_key_led_delete_sql(LAYER, varying="feature_id") for sql, _ in conn.executed)
    assert len(conn.executed) == 2
