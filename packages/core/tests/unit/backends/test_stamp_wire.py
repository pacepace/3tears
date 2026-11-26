"""every L3 request says whether its write stamps row epochs, and its lowest stamp (``STAMPS_EPOCH_FIELD``,
``STAMP_S_MIN_FIELD``): only 3tears knows which tables stamp, so the broker reads the request, never the catalog.

Each request type is driven through the store (``SqlL3Backend``) over the proxy as a collection writes, and its body
read as the broker receives it: a write of a table that declares a row epoch says ``true`` with its rows' lowest stamp;
any other request, a table with a column named ``epoch`` that is no row epoch among them (``scope_epochs``), says
``false``. A bulk write's batches are measured inside their stamp, so each is sized exactly as it is sent. The writes
made outside the store say so too: a keyed update of a stamped table (``update_keyed`` with ``stamped_values``) and a
declared table's own statements (``DeclaredTableCollection.write_row``), each with its own stamp as ``s_min``.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pytest
from threetears.core.backends.arrow_copy import WRITE_ARROW_FIELD, split_copy_request
from threetears.core.backends.broker_generation import (
    CLEANUP_DELETE_FIELD,
    CLEANUP_FLOOR_FIELD,
    STAMP_S_MIN_FIELD,
    CleanupMixedWriteError,
    STAMPS_EPOCH_FIELD,
    cleanup_deleting,
    stamp_fields,
    stamping_write,
)
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.backends.protocol import LoadFile
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections import DeclaredTableCollection
from threetears.core.collections.base import NO_L2
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import INT_TYPE, STRING_TYPE, Column, TableSchema, stamped_values
from threetears.core.collections.scope_epochs import scope_epochs_schema
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import (
    EPOCH_COLUMN,
    ROW_STAMPED,
    AppendOnlyWriteError,
    UnstampedWriteError,
    row_versions,
    wall_clock_ms,
)
from threetears.core.entities.base import BaseEntity
from threetears.core.keyed import update_keyed
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.nats import Subject

_TX_ID = str(uuid.UUID("00000000-0000-4000-8000-000000000903"))

#: a stamped table (ROW_STAMPED): one row a key, updated in place
_RUNS = TableSchema(
    name="runs",
    primary_key="run_id",
    columns=[Column("run_id", STRING_TYPE), Column("state", STRING_TYPE, nullable=True)],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)

#: a versioned table (row_versions): its writes append, its deletes are markers
_PROJECTIONS = TableSchema(
    name="projections",
    primary_key=("join_state_code", "run_id", "geo_id"),
    columns=[
        Column("join_state_code", STRING_TYPE),
        Column("run_id", INT_TYPE),
        Column("geo_id", STRING_TYPE),
        Column("value", INT_TYPE),
    ],
    row_epoch=row_versions("join_state_code"),
    l1_eviction=TEST_L1_TABLE,
)

#: a table that stamps nothing, with a BIGINT NOT NULL column named epoch: the catalog cannot tell it from a stamped one
_SCOPE_EPOCHS = scope_epochs_schema()

#: stamps of now, as a transaction's clock draws them: a commit whose lowest stamp is older than W is refused
_LOW = (wall_clock_ms() << 15) + 7
_HIGH = _LOW + 92


class _Broker:
    """answers each request with the next scripted reply, and keeps each request's body (an Arrow copy's envelope)."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        self.replies = list(replies)
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.raw: list[tuple[str, bytes]] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del timeout
        framed = split_copy_request(payload)
        body = json.loads(bytes(framed[0]) if framed is not None else payload)
        self.sent.append((str(subject), body))
        self.raw.append((str(subject), payload))
        return json.dumps(self.replies.pop(0)).encode("utf-8")

    def proxy(self) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = AsyncMock(side_effect=self.request_raw)
        return NatsProxyL3Backend(
            nats_client=nats, namespace_prefix="test", agent_id="agent-123", identity_token=lambda: "token"
        )

    def bodies(self, subject_suffix: str) -> list[dict[str, Any]]:
        return [body for subject, body in self.sent if subject.endswith(subject_suffix)]


def _store(proxy: NatsProxyL3Backend) -> SqlL3Backend:
    store = SqlL3Backend(proxy)
    for schema in (_RUNS, _PROJECTIONS, _SCOPE_EPOCHS):
        store.register_schema(schema.name, schema)
    return store


def _stamp(body: dict[str, Any]) -> tuple[Any, Any]:
    return body.get(STAMPS_EPOCH_FIELD), body.get(STAMP_S_MIN_FIELD)


def _touches(broker: _Broker) -> list[dict[str, Any]]:
    """each scope epochs touch the transaction sent before its COMMIT (a versioned table's write joins a
    CallerTransaction, which touches the scopes its rows name)."""
    return [body for body in broker.bodies("l3.tx.execute") if "SET last_epoch" in json.dumps(body)]


def _writes(broker: _Broker, suffix: str) -> list[dict[str, Any]]:
    """each request on ``suffix`` but the transaction's touches."""
    touches = _touches(broker)
    return [body for body in broker.bodies(suffix) if body not in touches]


_BEGIN = {"success": True, "tx_id": _TX_ID}
_BEGIN_ARROW = {"success": True, "tx_id": _TX_ID, WRITE_ARROW_FIELD: True}
_DONE = {"success": True, "row_count": 1}
_COMMIT = {"success": True}


# -- the transaction's doors: execute, copy, the Arrow write door, delete_prefixes ------------------------------------


async def test_a_stamped_tables_upsert_names_its_stamp_on_tx_execute() -> None:
    broker = _Broker(_BEGIN, _DONE, _COMMIT)
    proxy = broker.proxy()
    async with proxy.transaction() as conn:
        await _store(proxy).upsert("runs", {"run_id": "r1", "state": "NC", EPOCH_COLUMN: _LOW}, conn=conn)
    [body] = broker.bodies("l3.tx.execute")
    assert _stamp(body) == (True, _LOW)


async def test_a_bulk_upsert_names_its_rows_lowest_stamp_on_every_statement() -> None:
    broker = _Broker(_BEGIN, _DONE, _DONE, _COMMIT)
    proxy = broker.proxy()
    rows = [{"run_id": f"r{i}", "state": "NC", EPOCH_COLUMN: epoch} for i, epoch in enumerate((_HIGH, _LOW, _HIGH))]
    async with proxy.transaction() as conn:
        await _store(proxy).upsert_many("runs", rows, max_rows=2, max_bytes=1 << 20, conn=conn)
    bodies = broker.bodies("l3.tx.execute")
    assert len(bodies) == 2
    assert {_stamp(body) for body in bodies} == {(True, _LOW)}


async def test_a_copy_names_its_rows_lowest_stamp_on_tx_copy() -> None:
    broker = _Broker(_BEGIN, _DONE, _DONE, _COMMIT)
    proxy = broker.proxy()
    rows = [
        {"join_state_code": "NC", "run_id": 1, "geo_id": g, "value": 1, EPOCH_COLUMN: e}
        for g, e in (("a", _HIGH), ("b", _LOW))
    ]
    async with proxy.acquire() as conn, CallerTransaction(conn):
        await _store(proxy).copy_many("projections", rows, max_rows=10, max_bytes=1 << 20, conn=conn, append=True)
    [body] = broker.bodies("l3.tx.copy")
    assert _stamp(body) == (True, _LOW)
    [touch] = _touches(broker)
    assert touch["params"] == [_HIGH, ["NC"]]


async def test_an_arrow_write_names_its_rows_lowest_stamp_through_the_door() -> None:
    broker = _Broker(_BEGIN_ARROW, _DONE, _COMMIT)
    proxy = broker.proxy()
    store = _store(proxy)
    rows = pa.table({"run_id": ["r1", "r2"], "state": ["NC", "PA"], EPOCH_COLUMN: pa.array([_HIGH, _LOW], pa.int64())})
    async with proxy.transaction() as conn:
        plan = store.arrow_write_plan("runs", rows.column_names, rows.schema.types, kind="upsert", conn=conn)
        assert plan is not None
        await store.write_arrow("runs", rows, plan, conn=conn)
    [body] = broker.bodies("l3.tx.copy")
    assert _stamp(body) == (True, _LOW)


async def test_scope_epochs_through_the_door_says_it_stamps_nothing() -> None:
    broker = _Broker(_BEGIN_ARROW, _DONE, _COMMIT)
    proxy = broker.proxy()
    store = _store(proxy)
    now = datetime.now(UTC)
    rows = pa.table(
        {
            column.name: (
                pa.array(["NC"], pa.string())
                if column.name == "scope"
                else pa.array([now], pa.timestamp("us", tz="UTC"))
                if column.name.startswith("date_")
                else pa.array([7], pa.int64())
            )
            for column in _SCOPE_EPOCHS.columns
        }
    )
    assert EPOCH_COLUMN in rows.column_names
    async with proxy.transaction() as conn:
        plan = store.arrow_write_plan(
            _SCOPE_EPOCHS.name, rows.column_names, rows.schema.types, kind="upsert", conn=conn
        )
        assert plan is not None
        await store.write_arrow(_SCOPE_EPOCHS.name, rows, plan, conn=conn)
    [body] = broker.bodies("l3.tx.copy")
    assert _stamp(body) == (False, None)


async def test_a_gone_marker_names_the_deletes_stamp() -> None:
    broker = _Broker(_BEGIN, _DONE, _DONE, _COMMIT)
    proxy = broker.proxy()
    async with proxy.acquire() as conn, CallerTransaction(conn):
        await _store(proxy).mark_gone("projections", [("NC", 1, "a")], epoch=_LOW, scope=None, max_rows=10, conn=conn)
    [body] = _writes(broker, "l3.tx.execute")
    assert _stamp(body) == (True, _LOW)
    [touch] = _touches(broker)
    assert touch["params"] == [_LOW, ["NC"]]


async def test_a_run_marker_names_the_deletes_stamp_on_delete_prefixes() -> None:
    broker = _Broker(_BEGIN, _DONE, _DONE, _COMMIT)
    proxy = broker.proxy()
    async with proxy.acquire() as conn, CallerTransaction(conn):
        await _store(proxy).delete_prefixes("projections", [{"join_state_code": "NC"}], conn=conn, epoch=_LOW)
    [body] = broker.bodies("l3.tx.delete_prefixes")
    assert _stamp(body) == (True, _LOW)
    [touch] = _touches(broker)
    assert touch["params"] == [_LOW, ["NC"]]


async def test_a_prefix_delete_of_a_stamped_table_is_refused_before_anything_is_sent() -> None:
    # a stamped table's keys are marked gone in place, never deleted, so a restore hears them (test_stamped_gone)
    broker = _Broker(_BEGIN, _DONE, _COMMIT)
    proxy = broker.proxy()
    async with proxy.transaction() as conn:
        with pytest.raises(AppendOnlyWriteError, match="stamped"):
            await _store(proxy).delete_prefixes("runs", [{"run_id": "r1"}], conn=conn)
    assert broker.bodies("l3.tx.delete_prefixes") == []


async def test_every_other_transaction_statement_says_it_stamps_nothing() -> None:
    broker = _Broker(_BEGIN, _DONE, {"success": True, "rows": []}, {"success": True, "row": None}, _COMMIT)
    async with broker.proxy().transaction() as conn:
        await conn.execute("UPDATE runs SET state = $1", "NC")
        await conn.fetch("SELECT run_id FROM runs")
        await conn.fetchrow("SELECT run_id FROM runs WHERE run_id = $1", "r1")
    bodies = [*broker.bodies("l3.tx.execute"), *broker.bodies("l3.tx.fetch"), *broker.bodies("l3.tx.fetchrow")]
    assert len(bodies) == 3
    assert {_stamp(body) for body in bodies} == {(False, None)}


# -- the doors the broker commits itself: l3.query, l3.batch, l3.load_files -------------------------------------------


async def test_a_load_names_its_stamp_on_load_files() -> None:
    broker = _Broker({"success": True, "row_count": 3, "files": [{"index": 0, "rows_loaded": 3}]})
    proxy = broker.proxy()
    await _store(proxy).load_files(
        "projections",
        [LoadFile(url="https://bucket.s3.amazonaws.com/x.parquet?sig=a", rows=3)],
        columns=["run_id", "geo_id", "value"],
        constants={"join_state_code": "NC", EPOCH_COLUMN: _LOW},
        rows_per_transaction=1000,
        timeout_ms=1000,
    )
    [body] = broker.bodies("l3.load_files")
    assert _stamp(body) == (True, _LOW)


async def test_a_load_through_the_broker_names_its_stamp_as_a_constant() -> None:
    broker = _Broker()
    with pytest.raises(UnstampedWriteError, match="constant"):
        await _store(broker.proxy()).load_files(
            "projections",
            [LoadFile(url="https://bucket.s3.amazonaws.com/x.parquet?sig=a", rows=3)],
            columns=["run_id", "geo_id", "value", EPOCH_COLUMN],
            constants={"join_state_code": "NC"},
            rows_per_transaction=1000,
            timeout_ms=1000,
        )
    assert broker.sent == []


async def test_a_query_and_a_batch_say_what_their_block_names() -> None:
    broker = _Broker(_DONE, {"success": True, "results": [{"success": True}]}, _DONE, {"success": True, "rows": []})
    proxy = broker.proxy()
    with stamping_write(_LOW):
        await proxy.execute("UPDATE runs SET state = $1", "NC")
        await proxy.execute_batch([{"query": "DELETE FROM runs", "params": []}])
    await proxy.execute("UPDATE runs SET state = $1", "NC")
    await proxy.fetch("SELECT run_id FROM runs")
    queries = broker.bodies("l3.query")
    assert [_stamp(body) for body in queries] == [(True, _LOW), (False, None), (False, None)]
    [batch] = broker.bodies("l3.batch")
    assert _stamp(batch) == (True, _LOW)


# -- the writes made outside the store: a keyed update, a declared table's own statements ----------------------------


class _Run(BaseEntity):
    primary_key_field = "run_id"


class _Runs(DeclaredTableCollection[_Run]):
    """a declared collection of the stamped table, writing its own statements (write_row)."""

    schema = _RUNS

    @property
    def entity_class(self) -> type[_Run]:
        return _Run

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        return None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        return await self.write_row(data, conn=conn)

    async def delete_from_store(self, entity_id: Any) -> None:
        return None


def _runs(proxy: NatsProxyL3Backend) -> _Runs:
    collection = _Runs(
        registry=CollectionRegistry(),
        config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
        nats_client=NO_L2,
    )
    collection.l3_pool = cast(Any, proxy)
    return collection


async def test_a_keyed_update_of_a_stamped_table_names_its_stamp_on_the_wire() -> None:
    broker = _Broker(_DONE, _BEGIN, _DONE, _COMMIT)
    proxy = broker.proxy()
    # one statement, its own commit: l3.query
    alone = stamped_values(_RUNS, {"state": "done"})
    await update_keyed(proxy, "runs", values=alone, by={"run_id": "r1"})
    # on a transaction: l3.tx.execute
    async with proxy.transaction() as conn:
        joined = stamped_values(_RUNS, {"state": "done"}, conn=conn)
        await update_keyed(conn, "runs", values=joined, by={"run_id": "r2"})
    [query] = broker.bodies("l3.query")
    assert _stamp(query) == (True, alone[EPOCH_COLUMN].params[0])
    [statement] = broker.bodies("l3.tx.execute")
    assert _stamp(statement) == (True, joined[EPOCH_COLUMN].params[0])


async def test_a_keyed_update_of_an_unstamped_table_stamps_nothing() -> None:
    broker = _Broker(_DONE)
    await update_keyed(broker.proxy(), "runs", values={"state": "done"}, by={"run_id": "r1"})
    [query] = broker.bodies("l3.query")
    assert _stamp(query) == (False, None)


async def test_a_declared_tables_write_row_names_its_stamp_on_the_wire() -> None:
    broker = _Broker(_DONE, _DONE)
    collection = _runs(broker.proxy())
    await collection.write_row({"run_id": "r1", "state": "NC"})
    await collection.write_row({"run_id": "r2"})
    upsert, update = broker.bodies("l3.query")
    for body in (upsert, update):
        # the row's stamp is the one epoch the statement binds: a BIGINT past every other parameter (the gone flag the
        # write also binds is a bool, which Python counts as an int, so it is left out by type)
        [stamp] = [value for value in body["params"] if isinstance(value, int) and not isinstance(value, bool)]
        assert _stamp(body) == (True, stamp)


# -- the measure and the helper --------------------------------------------------------------------------------------


async def test_a_stamped_copy_is_sized_at_exactly_its_whole_request_stamp_and_all() -> None:
    # a bulk write's batches are measured inside its stamp: bounded at one copy's request, stamp fields and all, the
    # rows go in one; a byte under, they do not
    rows = [{"run_id": f"r{i:04d}", "state": "NC", EPOCH_COLUMN: _LOW + i} for i in range(30)]

    async def requests(max_bytes: int) -> _Broker:
        broker = _Broker(_BEGIN, *([_DONE] * 3), _COMMIT)
        proxy = broker.proxy()
        async with proxy.transaction() as conn:
            await _store(proxy).copy_many("runs", rows, max_rows=1000, max_bytes=max_bytes, conn=conn)
        return broker

    whole = await requests(10**9)
    [body] = whole.bodies("l3.tx.copy")
    assert _stamp(body) == (True, _LOW)
    [size] = [len(raw) for subject, raw in whole.raw if subject.endswith("l3.tx.copy")]
    assert len((await requests(size)).bodies("l3.tx.copy")) == 1
    assert len((await requests(size - 1)).bodies("l3.tx.copy")) == 2


def test_a_stamp_is_a_positive_bigint() -> None:
    for bad in (0, -1, 1 << 63):
        with pytest.raises(ValueError, match="positive BIGINT"), stamping_write(bad):
            pass
    assert stamp_fields() == {STAMPS_EPOCH_FIELD: False}
    with stamping_write(_LOW):
        assert stamp_fields() == {STAMPS_EPOCH_FIELD: True, STAMP_S_MIN_FIELD: _LOW}
        with stamping_write(None):
            assert stamp_fields() == {STAMPS_EPOCH_FIELD: False}


# -- quiet-time cleanup's deletes say so, with their floor ------------------------------------------------------------


def test_a_cleanup_delete_names_its_floor_beside_its_stamp() -> None:
    """Flips if a request inside cleanup_deleting does not say it is cleanup's with the floor, if one outside it does,
    or if a floor no row epoch can be is taken."""
    for bad in (0, -1, 1 << 63):
        with pytest.raises(ValueError, match="positive BIGINT"), cleanup_deleting(bad):
            pass
    with cleanup_deleting(_HIGH):
        assert stamp_fields() == {STAMPS_EPOCH_FIELD: False, CLEANUP_DELETE_FIELD: True, CLEANUP_FLOOR_FIELD: _HIGH}
        with stamping_write(_LOW):
            assert stamp_fields() == {
                STAMPS_EPOCH_FIELD: True,
                STAMP_S_MIN_FIELD: _LOW,
                CLEANUP_DELETE_FIELD: True,
                CLEANUP_FLOOR_FIELD: _HIGH,
            }
    assert stamp_fields() == {STAMPS_EPOCH_FIELD: False}


async def test_a_cleanup_delete_on_l3_query_carries_the_marker_and_floor() -> None:
    """Flips if l3.query (execute and fetch, the doors cleanup's point and prefix deletes take) sends a cleanup delete
    without its marker and floor, or marks a request sent outside the block."""
    broker = _Broker(
        {"success": True, "row_count": 1},
        {"success": True, "rows": [], "row_count": 0},
        {"success": True, "row_count": 0},
    )
    proxy = broker.proxy()
    with cleanup_deleting(_HIGH):
        await proxy.execute('DELETE FROM projections WHERE "epoch" < $1', _HIGH)
        await proxy.fetch('DELETE FROM projections WHERE "epoch" < $1 RETURNING "epoch"', _HIGH)
    await proxy.execute("DELETE FROM runs WHERE run_id = $1", "r1")
    marked, fetched, plain = broker.bodies("l3.query")
    for body in (marked, fetched):
        assert (body[CLEANUP_DELETE_FIELD], body[CLEANUP_FLOOR_FIELD]) == (True, _HIGH), body
    assert CLEANUP_DELETE_FIELD not in plain and CLEANUP_FLOOR_FIELD not in plain


# -- a cleanup commit is only cleanup ----------------------------------------------------------------------------------

_CLEANUP_DELETE = 'DELETE FROM projections WHERE "epoch" < $1'
_OTHER_WRITE = "UPDATE runs SET state = $1 WHERE run_id = 'r1'"


async def _cleanup_delete(conn: Any, *, fetch: bool) -> None:
    with cleanup_deleting(_HIGH):
        if fetch:
            await conn.fetch(f'{_CLEANUP_DELETE} RETURNING "epoch"', _HIGH)
        else:
            await conn.execute(_CLEANUP_DELETE, _HIGH)


@pytest.mark.parametrize(
    ("cleanup_first", "fetch"),
    [(True, False), (False, False), (False, True)],
    ids=["write-after-cleanup", "cleanup-after-write", "cleanup-fetch-after-write"],
)
async def test_a_transaction_mixing_a_cleanup_delete_and_another_write_is_refused_before_it_is_sent(
    cleanup_first: bool, fetch: bool
) -> None:
    """Flips if a CallerTransaction sends quiet-time cleanup's delete beside any other write: its commit's records
    would name cleanup's floor, and every copy skips them reading nothing (``catch_up_generation``), the other write
    with them. The second kind is refused before it is sent, and the transaction rolled back."""
    broker = _Broker(_BEGIN, _DONE, _COMMIT)
    proxy = broker.proxy()
    with pytest.raises(CleanupMixedWriteError, match="makes no other write"):
        async with proxy.acquire() as conn, CallerTransaction(conn):
            if cleanup_first:
                await _cleanup_delete(conn, fetch=fetch)
                await conn.execute(_OTHER_WRITE, "PA")
            else:
                await conn.execute(_OTHER_WRITE, "PA")
                await _cleanup_delete(conn, fetch=fetch)
    sent = [body for _, body in broker.sent if "query" in body]
    assert len(sent) == 1, "the refused request was never sent"
    assert bool(sent[0].get(CLEANUP_DELETE_FIELD)) is cleanup_first
    assert broker.bodies("l3.tx.commit") == []
    assert len(broker.bodies("l3.tx.rollback")) == 1


async def test_a_collection_write_beside_a_cleanup_delete_is_refused_as_it_joins() -> None:
    """Flips if a collection write joining the transaction (its enrollment, the rule's door on a connection the L3 rail
    does not carry) is let in inside cleanup_deleting after another write."""
    broker = _Broker(_BEGIN, _DONE, _COMMIT)
    proxy = broker.proxy()
    with pytest.raises(CleanupMixedWriteError):
        async with proxy.acquire() as conn, CallerTransaction(conn) as transaction:
            await conn.execute(_OTHER_WRITE, "PA")
            with cleanup_deleting(_HIGH):
                transaction.enroll(cast(Any, None), "r1")
    assert broker.bodies("l3.tx.commit") == []


async def test_a_transaction_of_only_cleanup_deletes_and_reads_commits() -> None:
    """Flips if the rule refuses what cleanup's own transaction is: its deletes, and reads beside them."""
    broker = _Broker(_BEGIN, _DONE, _DONE, {"success": True, "rows": [], "row_count": 0}, _COMMIT)
    proxy = broker.proxy()
    async with proxy.acquire() as conn, CallerTransaction(conn):
        await _cleanup_delete(conn, fetch=False)
        await _cleanup_delete(conn, fetch=False)
        await conn.fetch("SELECT run_id FROM runs WHERE run_id = $1", "r1")
    assert len(broker.bodies("l3.tx.commit")) == 1


# -- a request marked cleanup's only deletes -------------------------------------------------------------------------

_MARKED_INSERT = "INSERT INTO runs (run_id, state) VALUES ($1, 'NC')"
_MARKED_UPDATE = "UPDATE runs SET state = 'NC' WHERE run_id = $1"


async def _marked_on_door(door: str, query: str) -> _Broker:
    """one request marked cleanup's sent on ``door`` (a statement for the statement doors); the broker, its bodies
    kept. A tx door's request runs in a CallerTransaction, committed or rolled back as its body ends."""
    broker = _Broker(_BEGIN, _DONE, _DONE, {"success": True, "rows": [], "row_count": 0}, _COMMIT)
    proxy = broker.proxy()
    if door == "l3.query":
        broker.replies = [{"success": True, "row_count": 1}]
        with cleanup_deleting(_HIGH):
            await proxy.execute(query, "r1")
        return broker
    if door == "l3.batch":
        broker.replies = [{"success": True, "results": [{"success": True, "row_count": 1}]}]
        with cleanup_deleting(_HIGH):
            await proxy.execute_batch([{"query": query, "params": ["r1"]}])
        return broker
    async with proxy.acquire() as conn, CallerTransaction(conn):
        with cleanup_deleting(_HIGH):
            if door == "l3.tx.execute":
                await conn.execute(query, "r1")
            elif door == "l3.tx.fetch":
                await conn.fetch(f"{query} RETURNING run_id", "r1")
            elif door == "l3.tx.copy":
                await conn.copy_records_to_table("runs", records=[("r1", "NC")], columns=("run_id", "state"))
            else:
                await conn.delete_prefixes("projections", [{"join_state_code": "NC"}], epoch=_LOW, scope="state")
    return broker


@pytest.mark.parametrize(
    ("door", "query"),
    [
        ("l3.query", _MARKED_INSERT),
        ("l3.query", _MARKED_UPDATE),
        ("l3.batch", _MARKED_INSERT),
        ("l3.batch", _MARKED_UPDATE),
        ("l3.tx.execute", _MARKED_INSERT),
        ("l3.tx.execute", _MARKED_UPDATE),
        ("l3.tx.fetch", _MARKED_UPDATE),
        ("l3.tx.copy", ""),
        ("l3.tx.delete_prefixes", ""),
    ],
    ids=[
        "query-insert",
        "query-update",
        "batch-insert",
        "batch-update",
        "tx-insert",
        "tx-update",
        "tx-fetch-update",
        "tx-copy",
        "tx-run-markers",
    ],
)
async def test_a_request_marked_cleanups_that_is_no_delete_is_refused_before_it_is_sent(door: str, query: str) -> None:
    """Flips if a request marked quiet-time cleanup's that writes by anything but a delete (an insert, an update, a
    copy's rows, a versioned table's run markers) is sent, on any door: its commit's records would name cleanup's floor,
    and every copy skips them reading nothing."""
    with pytest.raises(ValueError, match="only deletes"):  # the refusal's class is the module's own
        await _marked_on_door(door, query)


@pytest.mark.parametrize("door", ["l3.query", "l3.batch", "l3.tx.execute", "l3.tx.fetch", "l3.tx.delete_prefixes"])
async def test_a_marked_delete_is_still_sent_on_every_door(door: str) -> None:
    """Flips if the rule refuses what cleanup sends: its deletes, on each door, marked with the floor."""
    query = 'DELETE FROM runs WHERE run_id = $1 AND "epoch" < 7'
    if door == "l3.tx.delete_prefixes":
        broker = _Broker(_BEGIN, _DONE, _COMMIT)
        proxy = broker.proxy()
        async with proxy.acquire() as conn, CallerTransaction(conn):
            with cleanup_deleting(_HIGH):
                await conn.delete_prefixes("runs", [{"run_id": "r1"}])
    else:
        broker = await _marked_on_door(door, query)
    marked = [body for _, body in broker.sent if body.get(CLEANUP_DELETE_FIELD) is True]
    assert len(marked) == 1 and marked[0][CLEANUP_FLOOR_FIELD] == _HIGH, broker.sent
