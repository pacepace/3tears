"""Many rows saved as multi-row upserts on a caller's transaction: batched to fit, rolled back whole."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from threetears.nats import Subject

from threetears.core.backends.nats_proxy import NatsProxyL3Backend

from threetears.core.backends.schema_sql import (
    build_bulk_insert_sql,
    build_insert_params,
    build_key_led_delete_sql,
    bulk_batches,
    transport_params_bytes,
)
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    INT_TYPE,
    STRING_TYPE,
    Column,
    SchemaBackedCollection,
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


@pytest.mark.parametrize(("slack", "per_statement"), [(0, [2, 2]), (-1, [1, 1, 1, 1])])
async def test_rows_are_split_so_no_statement_passes_the_byte_limit(slack: int, per_statement: list[int]) -> None:
    # with no bus around it a statement is its text and its parameters, both as JSON: the bound is
    # exactly two rows' statement, or a byte short of it
    collection, conn = _collection(), _Conn()
    stamped = {**_rows(1)[0], "date_created": datetime.now(UTC), "date_updated": datetime.now(UTC)}
    two_rows = len(json.dumps(build_bulk_insert_sql(SCHEMA, rows=2))) - 2
    two_rows += transport_params_bytes(build_insert_params(SCHEMA, stamped) * 2)
    async with CallerTransaction(conn):
        await collection.save_rows(_rows(4), conn=conn, max_bytes=two_rows + slack)
    assert [len(params) // 5 for _, params in conn.committed] == per_statement


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


# -- a statement is sized by what the rail sends --------------------------------------------------

#: a NATS deployment's default ``max_payload``, the message :attr:`BULK_MAX_BYTES` is set to fit
NATS_DEFAULT_MAX_PAYLOAD = 1024 * 1024

SHAPES = TableSchema(
    name="shapes",
    primary_key="id",
    columns=[Column("id", STRING_TYPE), Column("shape", BYTES_TYPE)],
)


#: the transaction a recording proxy's broker opens
_TX_ID = "019d9a00-0000-7000-8000-0000000000aa"


def _recording_proxy(*, statements: list[bytes] | None = None) -> tuple[NatsProxyL3Backend, list[bytes]]:
    """the rail's pod-side transport over a bus that keeps every request it is handed.

    :param statements: where to keep only the requests that carry a statement (not a transaction's
        begin or end), when given
    :return: the proxy, and the request payloads it put on the bus, in order
    :rtype: tuple[NatsProxyL3Backend, list[bytes]]
    """
    sent: list[bytes] = []

    async def request_raw(*, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del timeout
        sent.append(payload)
        if statements is not None and "query" in json.loads(payload):
            statements.append(payload)
        del subject
        return json.dumps({"success": True, "row_count": 1, "tx_id": _TX_ID}).encode("utf-8")

    nats = MagicMock()
    nats.request_raw = AsyncMock(side_effect=request_raw)
    proxy = NatsProxyL3Backend(
        nats_client=nats,
        namespace_prefix="test",
        agent_id="agent-123",
        identity_token=lambda: "test-identity-token",
    )
    return proxy, sent


def _params_on_the_wire(request: bytes) -> int:
    """the bytes a request's ``params`` value occupies in the message.

    :param request: one request payload, as sent
    :ptype request: bytes
    :return: the size of its ``params`` value
    :rtype: int
    """
    return len(json.dumps(json.loads(request)["params"]).encode("utf-8"))


async def test_a_page_of_bytes_heavy_rows_fits_the_bus_statement_by_statement() -> None:
    proxy, sent = _recording_proxy()
    store = SqlL3Backend(proxy)
    store.register_schema("shapes", SHAPES)
    # 4 MiB of shape: the page must split, and a split by any size smaller than the wire's
    # (bytes cross the rail as hex, two characters a byte) sends a statement past the bound
    rows = [{"id": f"{i:04d}", "shape": bytes([i % 251]) * (16 * 1024)} for i in range(256)]
    written = await store.upsert_many(
        "shapes",
        rows,
        max_rows=SchemaBackedCollection.BULK_MAX_ROWS,
        max_bytes=SchemaBackedCollection.BULK_MAX_BYTES,
    )
    assert written == 256
    assert len(sent) > 1
    assert max(_params_on_the_wire(request) for request in sent) <= SchemaBackedCollection.BULK_MAX_BYTES
    assert max(len(request) for request in sent) < NATS_DEFAULT_MAX_PAYLOAD
    # sized by the wire and not by something larger: every statement but the last is full to
    # within one row of the bound
    one_row = transport_params_bytes(["0000", rows[0]["shape"]])
    assert min(_params_on_the_wire(request) for request in sent[:-1]) > SchemaBackedCollection.BULK_MAX_BYTES - one_row


#: a table of many jsonb columns: each costs a statement ``$N::jsonb`` in its text, per row, beside
#: its value
WIDE = TableSchema(
    name="wide",
    primary_key="id",
    columns=[Column("id", STRING_TYPE), *(Column(f"j{i:02d}", JSONB_TYPE) for i in range(20))],
)


def _wide_rows(count: int) -> list[dict[str, Any]]:
    return [{"id": f"{i:05d}", **{f"j{c:02d}": {"k": "v" * 24} for c in range(20)}} for i in range(count)]


async def test_a_wide_jsonb_page_fits_the_bus_request_by_request() -> None:
    # parameters alone would fill BULK_MAX_BYTES with the statement's text, a ``$N::jsonb`` a
    # value, on top of it: every whole request is what is bounded
    proxy, sent = _recording_proxy()
    store = SqlL3Backend(proxy)
    store.register_schema("wide", WIDE)
    await store.upsert_many(
        "wide",
        _wide_rows(1000),
        max_rows=SchemaBackedCollection.BULK_MAX_ROWS,
        max_bytes=SchemaBackedCollection.BULK_MAX_BYTES,
    )
    assert len(sent) > 1
    assert max(len(request) for request in sent) <= SchemaBackedCollection.BULK_MAX_BYTES
    assert max(len(request) for request in sent) < NATS_DEFAULT_MAX_PAYLOAD


async def test_a_wide_jsonb_page_fits_the_bus_inside_a_transaction() -> None:
    statements: list[bytes] = []
    proxy, _ = _recording_proxy(statements=statements)
    collection = _collection(WIDE)
    collection.l3_pool = proxy
    async with proxy.acquire() as conn, CallerTransaction(conn):
        await collection.save_rows(_wide_rows(1000), conn=conn)
    assert len(statements) > 1
    assert all(json.loads(request)["tx_id"] == _TX_ID for request in statements)
    assert max(len(request) for request in statements) <= SchemaBackedCollection.BULK_MAX_BYTES


@pytest.mark.parametrize("in_transaction", [False, True])
async def test_a_statement_is_sized_at_exactly_its_whole_request(in_transaction: bool) -> None:
    # the size a batch is cut at is the size of the message the transport sends for it, to the byte:
    # bounded at one statement's request, the rows go in one; a byte under, they do not
    rows = _wide_rows(30)

    async def requests(max_bytes: int) -> list[bytes]:
        statements: list[bytes] = []
        proxy, _ = _recording_proxy(statements=statements)
        store = SqlL3Backend(proxy)
        store.register_schema("wide", WIDE)
        async with proxy.acquire() as conn:
            if in_transaction:
                async with conn.transaction():
                    await store.upsert_many("wide", rows, max_rows=1000, max_bytes=max_bytes, conn=conn)
            else:
                await store.upsert_many("wide", rows, max_rows=1000, max_bytes=max_bytes, conn=conn)
        return statements

    [whole] = await requests(10**9)
    assert len(await requests(len(whole))) == 1
    assert len(await requests(len(whole) - 1)) == 2


@pytest.mark.parametrize("in_transaction", [False, True])
async def test_a_transport_measures_the_request_it_sends(in_transaction: bool) -> None:
    proxy, sent = _recording_proxy()
    sql = "INSERT INTO t (k, v) VALUES ($1, $2::jsonb)"
    params = ["key", {"nested": [1, "two"]}]
    async with proxy.acquire() as conn:
        if in_transaction:
            async with conn.transaction():
                measured = conn.statement_request_bytes(sql, params)
                await conn.execute(sql, *params)
            statement = sent[1]
        else:
            measured = conn.statement_request_bytes(sql, params)
            assert proxy.statement_request_bytes(sql, params) == measured
            await conn.execute(sql, *params)
            statement = sent[0]
    assert measured == len(statement)


@pytest.mark.parametrize(
    "value",
    [
        b"\x00\xff" * 512,
        UUID("019d9a00-0000-7000-8000-000000000000"),
        datetime(2026, 10, 13, 12, 30, tzinfo=UTC),
        Decimal("12345.678900"),
        [UUID("019d9a00-0000-7000-8000-000000000000"), UUID("019d9a00-0000-7000-8000-000000000001")],
        (b"ab", b"cd"),
        {"nested": {"when": datetime(2026, 10, 13, tzinfo=UTC), "blob": b"xyz", "text": "caf\u00e9"}},
        "plain text, with a caf\u00e9",
        17,
        None,
    ],
    ids=["bytes", "uuid", "datetime", "decimal", "uuid-array", "bytes-tuple", "jsonb", "str", "int", "none"],
)
async def test_a_row_is_sized_at_exactly_what_the_rail_sends_for_it(value: Any) -> None:
    # the sizing and the transport share one encoding: a change to either without the other
    # moves one of these two numbers and not the other
    proxy, sent = _recording_proxy()
    row = ["key", value]
    await proxy.execute("INSERT INTO t (k, v) VALUES ($1, $2)", *row)
    [request] = sent
    on_the_wire = _params_on_the_wire(request)
    assert transport_params_bytes(row) == on_the_wire
    other = ["other", 1]
    assert [len(b) for b in bulk_batches([row, other], max_rows=10, max_bytes=on_the_wire + 12)] == [2]
    assert [len(b) for b in bulk_batches([row, other], max_rows=10, max_bytes=on_the_wire + 11)] == [1, 1]


async def test_a_sql_store_refuses_a_bulk_upsert_to_a_table_whose_schema_it_does_not_hold() -> None:
    store = SqlL3Backend(_Pool())
    with pytest.raises(ValueError, match="schema of 'nowhere' registered"):
        await store.upsert_many("nowhere", [{"id": 1}], max_rows=10, max_bytes=1000)


async def test_save_rows_answers_how_many_rows_it_submitted() -> None:
    collection, conn = _collection(), _Conn()
    async with CallerTransaction(conn):
        assert await collection.save_rows(_rows(4), conn=conn, max_rows=3) == 4


# -- delete_rows: the bulk write's other half -------------------------------------------------------


def test_one_statement_deletes_every_key_named_led_by_the_key() -> None:
    # the race's geographies as one array, the race fixed: led by the key, so bounded on a hashed key
    sql = build_key_led_delete_sql(SCHEMA, varying="geo_id")
    assert sql == "DELETE FROM results WHERE geo_id = ANY($1::text[]) AND office_key = $2"


async def test_delete_rows_deletes_them_in_one_statement_on_the_caller_s_transaction() -> None:
    collection, conn = _collection(), _Conn()
    async with CallerTransaction(conn):
        deleted = await collection.delete_rows([("sn_NC1", "00001"), ("sn_NC1", "00007")], conn=conn)
    assert deleted == 2
    [(sql, params)] = conn.committed
    assert sql == build_key_led_delete_sql(SCHEMA, varying="geo_id")
    assert params == (["00001", "00007"], "sn_NC1")


async def test_keys_to_delete_are_split_so_no_statement_passes_the_row_limit() -> None:
    collection, conn = _collection(), _Conn()
    keys = [("sn_NC1", f"{i:05d}") for i in range(5)]
    async with CallerTransaction(conn):
        await collection.delete_rows(keys, conn=conn, max_rows=2)
    assert [len(params[0]) for _, params in conn.committed] == [2, 2, 1]


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
