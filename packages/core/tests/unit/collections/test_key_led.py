"""Reads and deletes by many keys, every statement led by the key, so each stays bounded on a hash-sharded table.

On YugabyteDB a statement no key leads reads every row of the table (``ORDER BY`` the key included,
the key being hashed), and the L3 rail cuts every answer at its row cap without saying so. So a read
by many leading-key values goes a batch of values a statement (``lead = ANY($1)``), is read again in
halves when an answer may have been cut, pages one value's rows by the rest of its key when that
value alone fills the cap, and a delete by full keys names one varying key column as ``= ANY($1)``
with the rest of the key fixed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from threetears.core.backends.protocol import L3_RAIL_ROW_CAP
from threetears.core.backends.schema_sql import build_key_led_delete_sql
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE, read_l3_rows
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
    """a raw transport the collection wraps as its SQL store; statements go to the connection given.

    It says how many rows it answers a statement, as the L3 rail's client does.
    """

    def __init__(self, cap: int | None = L3_RAIL_ROW_CAP) -> None:
        self.rows_per_statement = cap


def _layer(cap: int | None = L3_RAIL_ROW_CAP, schema: TableSchema = LAYER) -> Any:
    collection = collection_for_schema(schema, entity_class=_Shape)(CollectionRegistry(), DefaultCoreConfig(), None)
    collection.l3_pool = _Pool(cap)
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
    table = _Table(_rows(5, 2), cap=L3_RAIL_ROW_CAP)
    held = await _layer().read_rows_led_by(["f0001", "f0003", "nope"], conn=table)

    assert _keys(held) == [("f0001", 0), ("f0001", 1), ("f0003", 0), ("f0003", 1)]
    [(sql, params)] = table.fetched
    assert sql == "SELECT feature_id, source_version FROM layer WHERE feature_id = ANY($1::text[])"
    assert params == (["f0001", "f0003", "nope"],)


async def test_values_go_a_batch_a_statement_each_named_once() -> None:
    table = _Table(_rows(10, 1), cap=L3_RAIL_ROW_CAP)
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
    table = _Table(_rows(2, 1), cap=L3_RAIL_ROW_CAP)

    held = await _layer().read_rows_led_by(["f0001"], columns=["name"], conn=table)

    assert held == [{"feature_id": "f0001", "source_version": 0, "name": "n1"}]
    assert table.fetched[0][0].startswith("SELECT feature_id, source_version, name FROM layer")


async def test_a_column_the_table_does_not_have_is_refused_naming_the_table() -> None:
    with pytest.raises(ValueError, match="layer"):
        await _layer().read_rows_led_by(["f0001"], columns=["nope"], conn=_Table([], cap=L3_RAIL_ROW_CAP))


async def test_no_values_reads_nothing() -> None:
    table = _Table(_rows(2, 1), cap=L3_RAIL_ROW_CAP)
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


async def test_mixed_keys_take_whichever_key_led_form_needs_fewer_statements() -> None:
    collection, conn = _layer(), _Conn()
    keys = [("a", 1), ("b", 1), ("c", 2), ("a", 2)]

    async with CallerTransaction(conn):
        assert await collection.delete_rows(keys, conn=conn, max_rows=2) == 4

    # grouped by generation: two groups of two, two statements; whole keys: two statements too
    assert [sql for sql, _ in conn.executed] == [build_key_led_delete_sql(LAYER, varying="feature_id")] * 2

    conn = _Conn()
    async with CallerTransaction(conn):
        await collection.delete_rows(keys, conn=conn)
    # one statement of whole keys beats a statement per generation
    [(sql, params)] = conn.executed
    assert "IN (SELECT * FROM unnest(" in sql
    assert sorted(zip(*params, strict=True)) == sorted(keys)


DOCS = TableSchema(
    name="docs",
    primary_key=("doc_id", "shape"),
    columns=[
        Column("doc_id", STRING_TYPE),
        Column("shape", JSONB_TYPE),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)


async def test_a_jsonb_key_column_held_fixed_is_bound_with_its_cast() -> None:
    collection, conn = _layer(schema=DOCS), _Conn()

    async with CallerTransaction(conn):
        assert await collection.delete_rows([("d1", {"k": 1}), ("d2", {"k": 1})], conn=conn) == 2

    [(sql, params)] = conn.executed
    assert sql == "DELETE FROM docs WHERE doc_id = ANY($1::text[]) AND shape = $2::jsonb"
    assert params[0] == ["d1", "d2"]


async def test_keys_sharing_no_value_go_max_rows_a_statement_led_by_the_key() -> None:
    collection, conn = _layer(), _Conn()
    keys = [(f"f{i:04d}", i) for i in range(5)]

    async with CallerTransaction(conn):
        assert await collection.delete_rows(keys, conn=conn, max_rows=2) == 5

    assert [sql for sql, _ in conn.executed] == [
        "DELETE FROM layer WHERE feature_id = ANY($1::text[]) "
        "AND (feature_id, source_version) IN (SELECT * FROM unnest($1::text[], $2::bigint[]))"
    ] * 3
    assert [params for _, params in conn.executed] == [
        (["f0000", "f0001"], [0, 1]),
        (["f0002", "f0003"], [2, 3]),
        (["f0004"], [4]),
    ]


async def test_a_batch_size_under_one_is_refused() -> None:
    collection = _layer()
    with pytest.raises(ValueError, match="max_values"):
        await collection.read_rows_led_by(["f0001"], max_values=-1, conn=_Table([], cap=L3_RAIL_ROW_CAP))
    with pytest.raises(ValueError, match="max_values"):
        await collection.read_rows_led_by(["f0001"], max_values=0, conn=_Table([], cap=L3_RAIL_ROW_CAP))
    conn = _Conn()
    with pytest.raises(ValueError, match="max_rows"):
        async with CallerTransaction(conn):
            await collection.delete_rows([("f0001", 1)], conn=conn, max_rows=-1)
    assert conn.executed == []


async def test_a_transport_with_a_cap_under_two_is_refused() -> None:
    with pytest.raises(ValueError, match="row cap"):
        await _layer(cap=1).read_rows_led_by(["f0001"], conn=_Table([], cap=1))


async def test_the_rail_s_cap_has_one_owner_the_copies_page_under() -> None:
    assert DEFAULT_PAGE_SIZE == L3_RAIL_ROW_CAP - 1


async def test_a_transport_that_cuts_nothing_is_never_read_again() -> None:
    # past the rail's cap in one answer: a transport that never cuts is believed, not split
    table = _Table(_rows(400, 3), cap=10_000)

    held = await _layer(cap=None).read_rows_led_by([f"f{i:04d}" for i in range(400)], max_values=400, conn=table)

    assert len(held) == 1200 > L3_RAIL_ROW_CAP
    assert len(table.fetched) == 1


async def test_a_transport_that_says_nothing_is_taken_to_cut_at_the_rail_s_cap() -> None:
    collection = collection_for_schema(LAYER, entity_class=_Shape)(CollectionRegistry(), DefaultCoreConfig(), None)

    class _Silent:
        """a raw transport that does not say how many rows it answers."""

    collection.l3_pool = _Silent()
    table = _Table(_rows(400, 3), cap=L3_RAIL_ROW_CAP)

    held = await collection.read_rows_led_by([f"f{i:04d}" for i in range(400)], max_values=400, conn=table)

    assert len(held) == 1200
    assert len(table.fetched) > 1


async def test_a_read_that_had_to_split_or_page_says_so_once(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="threetears.core.backends.sql")
    table = _Table(_rows(2, 12), cap=5)

    await _layer(cap=5).read_rows_led_by(["f0000", "f0001"], conn=table)

    [line] = [r for r in caplog.records if r.name == "threetears.core.backends.sql" and r.levelno == logging.INFO]
    detail = line.extra_data  # type: ignore[attr-defined]
    assert detail["table"] == "layer"
    assert detail["statements"] == len(table.fetched)
    assert detail["splits"] == 1
    assert detail["paged_values"] == 2


async def test_a_read_that_did_not_split_logs_nothing_at_info(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="threetears.core.backends.sql")
    await _layer().read_rows_led_by(["f0000"], conn=_Table(_rows(2, 2), cap=L3_RAIL_ROW_CAP))
    assert not [r for r in caplog.records if r.name == "threetears.core.backends.sql" and r.levelno >= logging.INFO]


async def test_a_store_without_the_key_led_read_refuses_a_connection_it_cannot_read_on() -> None:
    class _ScanOnly:
        """a non-SQL store whose scan takes no connection."""

        async def fetch_one(self, table: str, pk: Any, *, conn: Any = None) -> None:
            return None

        async def upsert(self, table: str, row: Any, **kwargs: Any) -> int:
            return 1

        async def delete(self, table: str, pk: Any, *, conn: Any = None) -> None:
            return None

        async def scan(self, table: str, filters: Any = None) -> list[dict[str, Any]]:
            raise AssertionError("a read that cannot honour the caller's connection is not made")

    collection = _layer()
    collection.l3_pool = _ScanOnly()
    with pytest.raises(ValueError, match="conn"):
        await collection.read_rows_led_by(["f0001"], conn=_Conn())


async def test_a_table_whose_schema_the_store_does_not_hold_is_refused() -> None:
    with pytest.raises(ValueError, match="registered"):
        await SqlL3Backend(_Pool()).fetch_led_by("layer", ["f0001"], columns=["feature_id"], max_values=1)
    with pytest.raises(ValueError, match="registered"):
        await SqlL3Backend(_Pool()).delete_many("layer", [("f0001", 1)], max_rows=1)


async def test_a_whole_table_read_refuses_a_page_of_no_rows() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        await read_l3_rows(_Table([], cap=L3_RAIL_ROW_CAP), "layer", ["feature_id"], ["feature_id"], page_size=0)


async def test_the_store_refuses_a_batch_size_under_one() -> None:
    store = SqlL3Backend(_Pool())
    store.register_schema("layer", LAYER)
    with pytest.raises(ValueError, match="max_values"):
        await store.fetch_led_by("layer", ["f0001"], columns=["feature_id"], max_values=0)
    with pytest.raises(ValueError, match="max_rows"):
        await store.delete_many("layer", [("f0001", 1)], max_rows=0)


JSONB_KEYED = TableSchema(
    name="tiles",
    primary_key=("tile", "doc"),
    columns=[Column("tile", STRING_TYPE), Column("doc", JSONB_TYPE)],
)


async def test_a_paged_cursor_binds_each_key_value_with_its_write_cast() -> None:
    class _Paging:
        """a transport answering the batch read full, then one short page."""

        rows_per_statement = 2

        def __init__(self) -> None:
            self.fetched: list[str] = []

        async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
            self.fetched.append(sql)
            if len(self.fetched) <= 2:
                return [{"tile": "t", "doc": {"n": 1}}, {"tile": "t", "doc": {"n": 2}}]
            return [{"tile": "t", "doc": {"n": 3}}]

    transport = _Paging()
    store = SqlL3Backend(transport)
    store.register_schema("tiles", JSONB_KEYED)
    await store.fetch_led_by("tiles", ["t"], columns=["tile", "doc"], max_values=1)

    # the batch, the first page, then a page past a cursor whose jsonb value keeps its cast
    assert len(transport.fetched) == 3
    assert "WHERE tile = $1 AND (doc) > ($2::jsonb) ORDER BY doc LIMIT 2" in transport.fetched[2]
