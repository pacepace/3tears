"""every L3 request says whether its write stamps row epochs, and its lowest stamp (``STAMPS_EPOCH_FIELD``,
``STAMP_S_MIN_FIELD``): only 3tears knows which tables stamp, so the broker reads the request, never the catalog.

Each request type is driven through the store (``SqlL3Backend``) over the proxy as a collection writes, and its body
read as the broker receives it: a write of a table that declares a row epoch says ``true`` with its rows' lowest stamp;
any other request, a table with a column named ``epoch`` that is no row epoch among them (``scope_epochs``), says
``false``. A bulk write's batches are measured inside their stamp, so each is sized exactly as it is sent.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pytest
from threetears.nats import Subject

from threetears.core.backends.arrow_copy import WRITE_ARROW_FIELD, split_copy_request
from threetears.core.backends.broker_generation import (
    STAMP_S_MIN_FIELD,
    STAMPS_EPOCH_FIELD,
    stamp_fields,
    stamping_write,
)
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.backends.protocol import LoadFile
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.schema_backed import INT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.collections.scope_epochs import scope_epochs_schema
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, ROW_STAMPED, UnstampedWriteError, row_versions
from threetears.core.testing.l1_tables import TEST_L1_TABLE

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

_LOW, _HIGH = (1 << 50) + 7, (1 << 50) + 99


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
    broker = _Broker(_BEGIN, _DONE, _COMMIT)
    proxy = broker.proxy()
    rows = [
        {"join_state_code": "NC", "run_id": 1, "geo_id": g, "value": 1, EPOCH_COLUMN: e}
        for g, e in (("a", _HIGH), ("b", _LOW))
    ]
    async with proxy.transaction() as conn:
        await _store(proxy).copy_many("projections", rows, max_rows=10, max_bytes=1 << 20, conn=conn, append=True)
    [body] = broker.bodies("l3.tx.copy")
    assert _stamp(body) == (True, _LOW)


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
    broker = _Broker(_BEGIN, _DONE, _COMMIT)
    proxy = broker.proxy()
    async with proxy.transaction() as conn:
        await _store(proxy).mark_gone("projections", [("NC", 1, "a")], epoch=_LOW, scope=None, max_rows=10, conn=conn)
    [body] = broker.bodies("l3.tx.execute")
    assert _stamp(body) == (True, _LOW)


async def test_a_run_marker_names_the_deletes_stamp_on_delete_prefixes() -> None:
    broker = _Broker(_BEGIN, _DONE, _COMMIT)
    proxy = broker.proxy()
    async with proxy.transaction() as conn:
        await _store(proxy).delete_prefixes("projections", [{"join_state_code": "NC"}], conn=conn, epoch=_LOW)
    [body] = broker.bodies("l3.tx.delete_prefixes")
    assert _stamp(body) == (True, _LOW)


async def test_a_prefix_delete_of_a_stamped_table_stamps_nothing() -> None:
    broker = _Broker(_BEGIN, _DONE, _COMMIT)
    proxy = broker.proxy()
    async with proxy.transaction() as conn:
        await _store(proxy).delete_prefixes("runs", [{"run_id": "r1"}], conn=conn)
    [body] = broker.bodies("l3.tx.delete_prefixes")
    assert _stamp(body) == (False, None)


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
