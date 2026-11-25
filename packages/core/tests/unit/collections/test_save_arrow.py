"""An Arrow table saved or deleted on a caller's transaction as it is, through the hub's Arrow write door.

What the pod sends is the contract the hub's broker implements (``WIRE.md``; the hub's
``test_l3_arrow_writes_real_database`` holds the other half). Pinned here, against a hub on a recording bus:

- to a hub whose ``tx.begin`` announces ``write_arrow``: ``save_arrow`` sends an Arrow ``l3.tx.copy`` whose envelope
  names the upsert (``on_conflict``, the key, the bulk upsert's ``SET`` list) and whose rows are the columns the bulk
  upsert would bind, typed, stamped; ``copy=True`` sends new rows ordered by key; ``delete_arrow`` sends the key's
  columns alone, ``delete: true``. No statement is sent;
- the rows are cut by Arrow slices, every request within the budget, the rows in order;
- keys, settlement and L1 are those of ``save_rows`` / ``delete_rows``; a key twice is refused before anything is sent;
- with no door (a hub without ``write_arrow``, a direct connection) or rows the door cannot carry, the write is
  ``save_rows`` / ``delete_rows`` exactly;
- the Arrow work runs off the event loop.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest
from sqlalchemy import MetaData
from threetears.nats import Subject

from threetears.core.backends import arrow_copy, nats_proxy, sql
from threetears.core.backends.arrow_copy import (
    CONFLICT_COLUMNS_FIELD,
    DELETE_FIELD,
    JSON_COLUMNS_FIELD,
    ON_CONFLICT_FIELD,
    ROWS_DIGEST_FIELD,
    UPDATE_COLUMNS_FIELD,
    WRITE_ARROW_FIELD,
    rows_digest,
    split_copy_request,
)
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.backends.schema_sql import build_bulk_insert_sql
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections import schema_backed
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, UnstampedWriteError, row_versions
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.l1_tables import TEST_L1_TABLE

_TX_ID = "019d9a00-0000-7000-8000-0000000000aa"

SCHEMA = TableSchema(
    name="results",
    primary_key=("office_key", "geo_id"),
    columns=[
        Column("office_key", STRING_TYPE),
        Column("geo_id", STRING_TYPE),
        Column("votes", INT_TYPE, nullable=True),
        Column("share", NUMERIC_TYPE, nullable=True, precision=12, scale=4),
        Column("called_at", DATETIMETZ_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    l1_eviction=TEST_L1_TABLE,
)

#: the columns every request sends for SCHEMA, in the order the bulk upsert binds them
_SENT = ["office_key", "geo_id", "votes", "share", "called_at", "date_created", "date_updated"]


class _ResultRow(BaseEntity):
    """a result row: its own id is the geography, within the race its key leads with."""

    primary_key_field = "geo_id"


class _Hub:
    """the bus a proxy sends on: keeps each request's subject and bytes, answers ``tx.begin`` with ``begin`` and each
    Arrow copy with the rows its stream carries."""

    def __init__(self, **begin: Any) -> None:
        self.raw: list[tuple[str, bytes]] = []
        self.begin = {"success": True, "tx_id": _TX_ID, "copy_arrow": True, **begin}
        self.threads: list[str] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del timeout
        self.raw.append((subject.path, payload))
        reply: dict[str, Any] = {"success": True, "row_count": 1}
        if subject.path.endswith("tx.begin"):
            reply = self.begin
        elif split_copy_request(payload) is not None:
            reply = {"success": True, "row_count": _arrow_request(payload)[1].num_rows}
        return json.dumps(reply).encode("utf-8")

    def proxy(self) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = self.request_raw
        nats.max_payload = 64 * 1024 * 1024
        return NatsProxyL3Backend(
            nats_client=nats, namespace_prefix="test", agent_id="agent-1", identity_token=lambda: "token"
        )

    def arrow_requests(self) -> list[tuple[dict[str, Any], pa.Table, int]]:
        """each Arrow copy request sent: its envelope, its rows and its size in bytes."""
        return [(*_arrow_request(data), len(data)) for subject, data in self.raw if split_copy_request(data)]

    def statements(self) -> list[dict[str, Any]]:
        """each JSON request carrying a statement."""
        return [
            body
            for _subject, data in self.raw
            if split_copy_request(data) is None and "query" in (body := json.loads(data))
        ]


def _arrow_request(data: bytes) -> tuple[dict[str, Any], pa.Table]:
    split = split_copy_request(data)
    assert split is not None
    envelope, offset = split
    ipc = data[offset:]
    body = json.loads(bytes(envelope))
    assert body[ROWS_DIGEST_FIELD] == rows_digest(ipc)
    return body, pa_ipc.open_stream(ipc).read_all()


def _collection(schema: TableSchema = SCHEMA, **attributes: Any) -> Any:
    registry = CollectionRegistry()
    entity = _ResultRow if len(schema.pk_columns) > 1 else None
    cls = collection_for_schema(schema, entity_class=entity)
    for name, value in attributes.items():
        setattr(cls, name, value)
    return cls(registry, DefaultCoreConfig(), None)


def _table(count: int, *, start: int = 0) -> pa.Table:
    """``count`` rows, keys descending so a copy's order by key shows; ``share`` a decimal, ``called_at`` naive (read
    as UTC into the timestamptz column, as a naive datetime is)."""
    numbers = list(range(start + count - 1, start - 1, -1))
    return pa.table(
        {
            "office_key": pa.array(["sn_NC1"] * count, pa.large_string()),
            "geo_id": [f"{n:05d}" for n in numbers],
            "votes": pa.array(numbers, pa.int32()),
            "share": pa.array([Decimal(f"{n}.25") for n in numbers], pa.decimal128(12, 4)),
            "called_at": pa.array(
                [datetime(2026, 11, 4, 4, 41, 7) + timedelta(seconds=n) for n in numbers],  # noqa: DTZ001 -- naive
                pa.timestamp("ms"),
            ),
        }
    )


async def _saved(hub: _Hub, table: pa.Table, *, copy: bool = False, collection: Any = None) -> Any:
    collection = collection or _collection()
    proxy = hub.proxy()
    collection.l3_pool = proxy
    async with proxy.acquire() as conn, CallerTransaction(conn):
        assert await collection.save_arrow(table, conn=conn, copy=copy) == table.num_rows
    return collection


# -- the door: what is sent ------------------------------------------------------------------------------------------


async def test_an_upsert_goes_as_one_arrow_request_naming_the_key_and_the_bulk_upserts_set_list() -> None:
    hub = _Hub(write_arrow=True)
    await _saved(hub, _table(3))
    [(envelope, rows, _size)] = hub.arrow_requests()
    assert hub.statements() == []
    assert envelope["table"] == "results"
    assert envelope["tx_id"] == _TX_ID
    assert envelope["columns"] == _SENT == rows.column_names
    assert envelope[ON_CONFLICT_FIELD] == "update"
    assert envelope[CONFLICT_COLUMNS_FIELD] == ["office_key", "geo_id"]
    # the bulk statement's SET list: every mutable column, never the immutable date_created
    assert envelope[UPDATE_COLUMNS_FIELD] == ["votes", "share", "called_at", "date_updated"]
    sql = build_bulk_insert_sql(SCHEMA, rows=1)
    assert all(f"{name} = EXCLUDED.{name}" in sql for name in envelope[UPDATE_COLUMNS_FIELD])
    assert DELETE_FIELD not in envelope
    assert envelope[JSON_COLUMNS_FIELD] == []
    assert rows.schema.types == [
        pa.string(),
        pa.string(),
        pa.int64(),
        pa.decimal128(12, 4),
        pa.timestamp("us", tz="UTC"),
        pa.timestamp("us", tz="UTC"),
        pa.timestamp("us", tz="UTC"),
    ]
    assert rows.column("geo_id").to_pylist() == ["00002", "00001", "00000"]
    assert rows.column("share").to_pylist() == [Decimal("2.2500"), Decimal("1.2500"), Decimal("0.2500")]
    assert rows.column("called_at").to_pylist()[0] == datetime(2026, 11, 4, 4, 41, 9, tzinfo=UTC)
    created, updated = rows.column("date_created").to_pylist(), rows.column("date_updated").to_pylist()
    assert len(set(created)) == 1 and created == updated


async def test_a_copy_sends_new_rows_ordered_by_key_and_names_no_conflict() -> None:
    hub = _Hub(write_arrow=True)
    await _saved(hub, _table(3), copy=True)
    [(envelope, rows, _size)] = hub.arrow_requests()
    assert not {ON_CONFLICT_FIELD, CONFLICT_COLUMNS_FIELD, UPDATE_COLUMNS_FIELD, DELETE_FIELD} & envelope.keys()
    assert rows.column("geo_id").to_pylist() == ["00000", "00001", "00002"]
    assert rows.column("votes").to_pylist() == [0, 1, 2]


async def test_a_delete_sends_the_keys_columns_alone() -> None:
    hub = _Hub(write_arrow=True)
    collection = _collection()
    proxy = hub.proxy()
    collection.l3_pool = proxy
    async with proxy.acquire() as conn, CallerTransaction(conn):
        assert await collection.delete_arrow(_table(3), conn=conn) == 3
    [(envelope, rows, _size)] = hub.arrow_requests()
    assert hub.statements() == []
    assert envelope[DELETE_FIELD] is True
    assert not {ON_CONFLICT_FIELD, CONFLICT_COLUMNS_FIELD, UPDATE_COLUMNS_FIELD} & envelope.keys()
    assert envelope["columns"] == ["office_key", "geo_id"] == rows.column_names
    assert rows.column("geo_id").to_pylist() == ["00002", "00001", "00000"]


async def test_a_nullable_column_the_rows_leave_out_is_sent_all_null_and_a_required_one_is_refused() -> None:
    hub = _Hub(write_arrow=True)
    await _saved(hub, _table(2).drop_columns(["votes"]))
    [(_envelope, rows, _size)] = hub.arrow_requests()
    assert rows.column_names == _SENT
    assert rows.column("votes").null_count == 2
    with pytest.raises(KeyError, match="geo_id"):
        await _saved(_Hub(write_arrow=True), _table(2).drop_columns(["geo_id"]))


async def test_the_rows_are_cut_by_arrow_slices_every_request_within_the_budget() -> None:
    budget = 64 * 1024 + 4000
    hub = _Hub(write_arrow=True, max_request_bytes=budget)
    await _saved(hub, _table(400))
    requests = hub.arrow_requests()
    assert len(requests) > 1
    assert all(size <= budget for _envelope, _rows, size in requests)
    sent = [geo for _envelope, rows, _size in requests for geo in rows.column("geo_id").to_pylist()]
    assert sent == [f"{n:05d}" for n in range(399, -1, -1)]
    assert all(envelope[CONFLICT_COLUMNS_FIELD] == ["office_key", "geo_id"] for envelope, _r, _s in requests)


def _request_rows(hub: _Hub) -> list[int]:
    return [rows.num_rows for _envelope, rows, _size in hub.arrow_requests()]


async def test_an_upsert_request_carries_no_more_rows_than_a_bulk_statement() -> None:
    # the hub merges a request as one statement under the transaction's statement timeout: an upsert cut by bytes
    # alone was one statement for a whole page (5,000 rows, ~5.8 s on YugabyteDB, cancelled at five)
    hub = _Hub(write_arrow=True)
    await _saved(hub, _table(2500))
    assert _request_rows(hub) == [1000, 1000, 500]
    sent = [geo for _envelope, rows, _size in hub.arrow_requests() for geo in rows.column("geo_id").to_pylist()]
    assert sent == [f"{n:05d}" for n in range(2499, -1, -1)]


async def test_a_delete_request_names_no_more_keys_than_a_bulk_statement() -> None:
    hub = _Hub(write_arrow=True)
    collection = _collection()
    proxy = hub.proxy()
    collection.l3_pool = proxy
    async with proxy.acquire() as conn, CallerTransaction(conn):
        assert await collection.delete_arrow(_table(2500), conn=conn) == 2500
    assert _request_rows(hub) == [1000, 1000, 500]


async def test_a_copy_is_cut_by_its_bytes_alone_unless_the_caller_bounds_its_rows() -> None:
    hub = _Hub(write_arrow=True)
    await _saved(hub, _table(2500), copy=True)
    assert _request_rows(hub) == [2500]
    bounded = _Hub(write_arrow=True)
    collection = _collection()
    proxy = bounded.proxy()
    collection.l3_pool = proxy
    async with proxy.acquire() as conn, CallerTransaction(conn):
        assert await collection.save_arrow(_table(2500), conn=conn, copy=True, max_rows=1200) == 2500
    assert _request_rows(bounded) == [1200, 1200, 100]


async def test_a_bound_under_one_row_is_refused() -> None:
    collection = _collection()
    proxy = _Hub(write_arrow=True).proxy()
    collection.l3_pool = proxy
    async with proxy.acquire() as conn, CallerTransaction(conn):
        with pytest.raises(ValueError, match="max_rows"):
            await collection.save_arrow(_table(2), conn=conn, max_rows=0)
        with pytest.raises(ValueError, match="max_rows"):
            await collection.delete_arrow(_table(2), conn=conn, max_rows=0)


# -- keys, settlement, L1 --------------------------------------------------------------------------------------------


async def test_two_rows_with_one_key_are_refused_before_anything_is_sent() -> None:
    hub = _Hub(write_arrow=True)
    doubled = pa.concat_tables([_table(2), _table(1)])
    with pytest.raises(ValueError, match="share a key"):
        await _saved(hub, doubled)
    assert hub.arrow_requests() == [] and hub.statements() == []


async def test_every_row_is_settled_as_save_rows_settles_it() -> None:
    settled: dict[str, list[Any]] = {}
    for how in ("save_rows", "save_arrow"):
        collection = _collection()
        proxy = _Hub(write_arrow=True).proxy()
        collection.l3_pool = proxy

        async def settle(keys: Any, *, rows: Any = None, how: str = how) -> None:
            settled[how] = [list(keys), list(rows) if rows is not None else None]

        collection.invalidate_cache_many = settle
        async with proxy.acquire() as conn, CallerTransaction(conn):
            if how == "save_rows":
                await collection.save_rows(_table(3).to_pylist(), conn=conn)
            else:
                await collection.save_arrow(_table(3), conn=conn)
    assert settled["save_arrow"][0] == settled["save_rows"][0] == [("sn_NC1", f"{n:05d}") for n in (2, 1, 0)]


async def test_a_collection_with_invalidation_columns_settles_each_row_with_them() -> None:
    changed: dict[str, list[Any]] = {}
    for how in ("save_rows", "save_arrow"):
        collection = _collection(invalidation_columns=("office_key", "votes"))
        proxy = _Hub(write_arrow=True).proxy()
        collection.l3_pool = proxy

        async def settle(keys: Any, *, rows: Any = None, how: str = how, collection: Any = collection) -> None:
            changed[how] = [collection.changed_row(key, row).model_dump() for key, row in zip(keys, rows, strict=True)]

        collection.invalidate_cache_many = settle
        async with proxy.acquire() as conn, CallerTransaction(conn):
            if how == "save_rows":
                await collection.save_rows(_table(3).to_pylist(), conn=conn)
            else:
                await collection.save_arrow(_table(3), conn=conn)
    assert changed["save_arrow"] == changed["save_rows"]
    assert changed["save_arrow"][0]["columns"] == {"office_key": "sn_NC1", "votes": "2"}


def _cached_results(rows: int, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, SQLiteBackend]:
    """a results collection over a real SQLite L1 that holds ``rows`` of its rows."""
    metadata = MetaData()
    SCHEMA.to_sqlalchemy_table(metadata)
    l1 = SQLiteBackend(db_name=f"arrow_evict_{threading.get_ident()}_{rows}")
    l1.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1)
    collection = collection_for_schema(SCHEMA, entity_class=_ResultRow)(registry, DefaultCoreConfig(), None)
    now = datetime.now(UTC)
    for row in _table(rows).to_pylist():
        assert collection.write_to_cache_sync({**row, "date_created": now, "date_updated": now})

    async def settle(keys: Any, *, rows: Any = None) -> None:
        pass

    # the settle evicts again as the transaction ends; these tests pin the write's own eviction
    monkeypatch.setattr(collection, "invalidate_cache_many", settle)
    return collection, l1


@pytest.mark.parametrize("how", ["save_arrow", "delete_arrow"])
async def test_every_key_leaves_l1_in_one_write(monkeypatch: pytest.MonkeyPatch, how: str, l1_statements: Any) -> None:
    collection, l1 = _cached_results(40, monkeypatch)
    proxy = _Hub(write_arrow=True).proxy()
    collection.l3_pool = proxy
    statements = l1_statements(l1)
    async with proxy.acquire() as conn, CallerTransaction(conn):
        await getattr(collection, how)(_table(40), conn=conn)
    assert statements.count("BEGIN IMMEDIATE") == 1
    assert [collection.exists_in_cache_sync(("sn_NC1", f"{n:05d}")) for n in range(40)] == [False] * 40


# -- no door: today's write ------------------------------------------------------------------------------------------


def _statement_shape(hub: _Hub) -> list[tuple[str, list[Any]]]:
    """each statement sent, its stamps left out (they are the write's time)."""
    return [
        (body["query"], [value for value in body["params"] if not str(value).startswith("20")])
        for body in hub.statements()
    ]


async def test_a_hub_without_the_door_is_sent_what_save_rows_sends() -> None:
    hubs = {how: _Hub() for how in ("save_rows", "save_arrow")}
    for how, hub in hubs.items():
        collection = _collection()
        proxy = hub.proxy()
        collection.l3_pool = proxy
        async with proxy.acquire() as conn, CallerTransaction(conn):
            if how == "save_rows":
                await collection.save_rows(_table(3).to_pylist(), conn=conn)
            else:
                await collection.save_arrow(_table(3), conn=conn)
    assert hubs["save_arrow"].arrow_requests() == []
    assert _statement_shape(hubs["save_arrow"]) == _statement_shape(hubs["save_rows"]) != []


async def test_a_hub_without_the_door_is_sent_what_delete_rows_sends() -> None:
    hubs = {how: _Hub() for how in ("delete_rows", "delete_arrow")}
    for how, hub in hubs.items():
        collection = _collection()
        proxy = hub.proxy()
        collection.l3_pool = proxy
        async with proxy.acquire() as conn, CallerTransaction(conn):
            if how == "delete_rows":
                await collection.delete_rows([("sn_NC1", f"{n:05d}") for n in (2, 1, 0)], conn=conn)
            else:
                await collection.delete_arrow(_table(3), conn=conn)
    assert hubs["delete_arrow"].arrow_requests() == []
    assert _statements(hubs["delete_arrow"]) == _statements(hubs["delete_rows"]) != []


def _statements(hub: _Hub) -> list[tuple[str, list[Any]]]:
    return [(body["query"], body["params"]) for body in hub.statements()]


class _Conn:
    """an asyncpg-shaped direct connection: no Arrow door; keeps each statement."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self, **options: Any) -> Any:
        del options
        return _NoTransaction()

    async def execute(self, sql: str, *params: Any) -> str:
        self.executed.append((sql, params))
        return "INSERT 0 1"


class _NoTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _Pool:
    """a raw transport the collection wraps as its SQL store; writes go to the caller's connection."""


async def test_a_direct_connection_is_written_what_save_rows_writes() -> None:
    collection = _collection()
    collection.l3_pool = _Pool()
    conn = _Conn()
    async with CallerTransaction(conn):
        await collection.save_arrow(_table(3), conn=conn)
    [(sql, params)] = conn.executed
    assert sql == build_bulk_insert_sql(SCHEMA, rows=3)
    assert params[0:5] == ("sn_NC1", "00002", 2, Decimal("2.2500"), datetime(2026, 11, 4, 4, 41, 9, tzinfo=UTC))


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("votes", pa.array([1, 2], pa.uint64())),  # past int64
        ("called_at", pa.array([1000, 2000], pa.timestamp("ns"))),  # finer than the door
        ("share", pa.array([b"a", b"b"], pa.binary())),  # no door type
    ],
)
async def test_a_column_the_door_does_not_carry_is_written_as_save_rows_writes_it(column: str, values: Any) -> None:
    hub = _Hub(write_arrow=True)
    table = _table(2)
    table = table.set_column(table.schema.get_field_index(column), column, values)
    collection = _collection()
    proxy = hub.proxy()
    collection.l3_pool = proxy
    async with proxy.acquire() as conn, CallerTransaction(conn):
        await collection.save_arrow(table, conn=conn)
    assert hub.arrow_requests() == []
    [statement] = hub.statements()
    assert statement["query"].startswith("INSERT INTO results")


async def test_a_jsonb_column_is_written_as_save_rows_writes_it() -> None:
    schema = TableSchema(
        name="docs",
        primary_key="id",
        columns=[Column("id", STRING_TYPE), Column("doc", JSONB_TYPE, nullable=True)],
        l1_eviction=TEST_L1_TABLE,
    )
    hub = _Hub(write_arrow=True)
    await _saved(hub, pa.table({"id": ["a"], "doc": ['{"k": 1}']}), collection=_collection(schema))
    assert hub.arrow_requests() == []
    assert len(hub.statements()) == 1


async def test_a_table_whose_conflict_clause_raises_is_upserted_as_save_rows_writes_it() -> None:
    schema = TableSchema(
        name="once",
        primary_key="id",
        columns=[Column("id", STRING_TYPE), Column("n", INT_TYPE, nullable=True)],
        on_conflict="raise",
        l1_eviction=TEST_L1_TABLE,
    )
    hub = _Hub(write_arrow=True)
    await _saved(hub, pa.table({"id": ["a"], "n": [1]}), collection=_collection(schema))
    assert hub.arrow_requests() == []
    hub = _Hub(write_arrow=True)
    await _saved(hub, pa.table({"id": ["a"], "n": [1]}), copy=True, collection=_collection(schema))
    assert len(hub.arrow_requests()) == 1


async def test_an_ignoring_table_s_upsert_names_no_column_to_update() -> None:
    schema = TableSchema(
        name="kept",
        primary_key="id",
        columns=[Column("id", STRING_TYPE), Column("n", INT_TYPE, nullable=True)],
        on_conflict="ignore",
        l1_eviction=TEST_L1_TABLE,
    )
    hub = _Hub(write_arrow=True)
    await _saved(hub, pa.table({"id": ["a"], "n": [1]}), collection=_collection(schema))
    [(envelope, _rows, _size)] = hub.arrow_requests()
    assert envelope[ON_CONFLICT_FIELD] == "update" and envelope[UPDATE_COLUMNS_FIELD] == []


async def test_a_collection_with_invalidation_columns_deletes_as_delete_rows_does() -> None:
    hub = _Hub(write_arrow=True)
    collection = _collection(invalidation_columns=("office_key",))
    proxy = hub.proxy()
    collection.l3_pool = proxy
    async with proxy.acquire() as conn, CallerTransaction(conn):
        await collection.delete_arrow(_table(2), conn=conn)
    assert hub.arrow_requests() == []
    assert hub.statements() != []


async def test_the_announcement_is_read_at_each_begin() -> None:
    hub = _Hub(**{WRITE_ARROW_FIELD: True})
    proxy = hub.proxy()
    async with proxy.acquire() as conn:
        async with conn.transaction():
            assert conn.write_arrow is True
        hub.begin = {"success": True, "tx_id": _TX_ID}
        async with conn.transaction():
            assert conn.write_arrow is False
            with pytest.raises(ValueError, match="write_arrow"):
                await conn.write_arrow_rows("results", _table(1))


# -- off the loop ----------------------------------------------------------------------------------------------------


async def test_the_arrow_work_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    loop_thread = threading.get_ident()
    ran: dict[str, int] = {}

    def recorded(name: str, real: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            ran[name] = threading.get_ident()
            return real(*args, **kwargs)

        return call

    monkeypatch.setattr(nats_proxy, "write_request_rows", recorded("slices", arrow_copy.write_request_rows))
    monkeypatch.setattr(sql, "planned_write_table", recorded("plan", arrow_copy.planned_write_table))
    monkeypatch.setattr(schema_backed, "with_constant_columns", recorded("stamps", arrow_copy.with_constant_columns))
    monkeypatch.setattr(schema_backed, "arrow_python_columns", recorded("keys", arrow_copy.arrow_python_columns))
    await _saved(_Hub(write_arrow=True), _table(3))
    assert set(ran) == {"slices", "plan", "stamps", "keys"}
    assert loop_thread not in ran.values()


# -- the row epoch: every row of an Arrow write carries its write's epoch ---------------------------------------------

_VERSIONED = TableSchema(
    name="versioned_results",
    primary_key=("office_key", "geo_id"),
    columns=[
        Column("office_key", STRING_TYPE),
        Column("geo_id", STRING_TYPE),
        Column("votes", INT_TYPE, nullable=True),
        Column("share", NUMERIC_TYPE, nullable=True, precision=12, scale=4),
        Column("called_at", DATETIMETZ_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    row_epoch=row_versions("office_key"),
    l1_eviction=TEST_L1_TABLE,
)


@pytest.mark.parametrize("copy", [False, True])
async def test_an_arrow_write_of_a_versioned_table_stamps_every_row_with_its_writes_epoch(copy: bool) -> None:
    hub = _Hub(write_arrow=True)
    collection = _collection(_VERSIONED)
    proxy = hub.proxy()
    collection.l3_pool = proxy
    async with proxy.acquire() as conn, CallerTransaction(conn) as transaction:
        transaction.note_write_version(17)
        await collection.save_arrow(_table(3), conn=conn, copy=copy)
        await collection.save_arrow(_table(3), conn=conn, copy=copy)
    first, second = hub.arrow_requests()
    epochs = []
    for envelope, rows, _size in (first, second):
        assert rows.schema.field(EPOCH_COLUMN).type == pa.int64()
        [epoch] = set(rows.column(EPOCH_COLUMN).to_pylist())
        assert _version(epoch) == 17
        epochs.append(epoch)
        if not copy:
            # the conflict is a version of the key, never the key alone
            assert envelope[CONFLICT_COLUMNS_FIELD] == ["office_key", "geo_id", EPOCH_COLUMN]
            assert EPOCH_COLUMN not in envelope[UPDATE_COLUMNS_FIELD]
    # the same keys written twice in one version: the later write sorts newer
    assert epochs[1] > epochs[0]


async def test_an_arrow_write_of_a_versioned_table_on_a_transaction_that_took_no_turn_is_refused() -> None:
    hub = _Hub(write_arrow=True)
    collection = _collection(_VERSIONED)
    proxy = hub.proxy()
    collection.l3_pool = proxy
    with pytest.raises(UnstampedWriteError, match="took no write's turn"):
        async with proxy.acquire() as conn, CallerTransaction(conn):
            await collection.save_arrow(_table(3), conn=conn)
    assert hub.arrow_requests() == []


def _version(epoch: int) -> int:
    """the version an epoch was stamped at, by its documented encoding (``version << 31 | sequence``)."""
    return epoch >> 31
