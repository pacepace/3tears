"""The two bulk L3 operations a table reload uses: ``l3.truncate`` and ``l3.tx.copy``, and the SQL store over them.

What the pod sends is the contract the hub's broker implements, so these pin the request bodies
byte for byte where they matter (the subject, every field, how a row's values cross the rail) and
how a reply is read (a refusal raises as every L3 reply's does; a write's generations are kept).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import asyncpg
import pytest
from threetears.nats import Subject

from threetears.core.backends.broker_generation import BrokerGenerationSource
from threetears.core.backends.nats_proxy import CONSTRAINT_VIOLATION_ERROR_CODE, NatsProxyL3Backend
from threetears.core.backends.schema_sql import build_insert_params
from threetears.core.backends.sql import SqlL3Backend, bound_request_connection
from threetears.core.collections.schema_backed import (
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    VECTOR_TYPE,
    Column,
    TableSchema,
)
from threetears.core.exceptions import DataLayerUnavailableError

#: the transaction a recording broker opens
_TX_ID = "019d9a00-0000-7000-8000-0000000000aa"

SCHEMA = TableSchema(
    name="lsf_rows",
    primary_key=("race", "geo"),
    columns=[
        Column("race", STRING_TYPE),
        Column("geo", STRING_TYPE),
        Column("source_id", UUID_TYPE, nullable=True),
        Column("shape", BYTES_TYPE, nullable=True),
        Column("loaded_at", DATETIMETZ_TYPE, nullable=True),
    ],
)

_WHEN = datetime(2026, 10, 22, 12, 30, tzinfo=UTC)
_SOURCE = UUID("019d9a00-0000-7000-8000-000000000001")


class _Broker:
    """the bus a proxy sends on, keeping every request and answering each with the next reply.

    :ivar requests: ``(subject, body)`` per request, in order
    """

    def __init__(self, *replies: dict[str, Any]) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.raw: list[bytes] = []
        self._replies = list(replies)

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del timeout
        self.raw.append(payload)
        self.requests.append((subject.path, json.loads(payload)))
        reply = self._replies.pop(0) if self._replies else {"success": True, "row_count": 1, "tx_id": _TX_ID}
        return json.dumps(reply).encode("utf-8")

    def proxy(self) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = AsyncMock(side_effect=self.request_raw)
        return NatsProxyL3Backend(
            nats_client=nats,
            namespace_prefix="test",
            agent_id="agent-123",
            identity_token=lambda: "test-identity-token",
        )


def _rows() -> list[dict[str, Any]]:
    # given out of key order: a copy sends them ordered by key
    return [
        {"race": "r2", "geo": "g1", "source_id": _SOURCE, "shape": b"\x00\xff", "loaded_at": _WHEN},
        {"race": "r1", "geo": "g2", "source_id": None, "shape": None, "loaded_at": None},
        {"race": "r1", "geo": "g1", "source_id": None, "shape": None, "loaded_at": _WHEN},
    ]


# -- l3.truncate ----------------------------------------------------------------------------------


async def test_truncate_sends_one_request_naming_the_table_and_the_principal_s_namespace() -> None:
    broker = _Broker()
    await broker.proxy().truncate("lsf_rows")
    [(subject, body)] = broker.requests
    assert subject == "test.l3.truncate"
    assert set(body) == {"correlation_id", "identity_token", "namespace", "table", "timeout_ms"}
    assert body["table"] == "lsf_rows"
    assert body["namespace"] == "agents.agent-123"
    assert body["identity_token"] == "test-identity-token"
    assert body["timeout_ms"] == 5000


async def test_truncate_binds_a_namespace_it_is_given_as_a_query_does() -> None:
    broker = _Broker()
    await broker.proxy().truncate("lsf_rows", namespace="tools.reports")
    [(_, body)] = broker.requests
    assert body["namespace"] == "tools.reports"


async def test_a_refused_truncate_raises_as_every_l3_refusal_does() -> None:
    broker = _Broker({"success": False, "error_code": "FORBIDDEN", "error_message": "not a declared table"})
    with pytest.raises(DataLayerUnavailableError, match="L3 truncate failed: FORBIDDEN"):
        await broker.proxy().truncate("lsf_rows")


async def test_the_generation_a_truncate_s_reply_names_is_handed_to_the_advance_after_it() -> None:
    broker = _Broker({"success": True, "generations": {"lsf_rows": "inc:9"}})
    await broker.proxy().truncate("lsf_rows")
    assert await BrokerGenerationSource().advance("lsf_rows") == "inc:9"


async def test_a_sql_store_over_the_rail_truncates_through_the_broker() -> None:
    broker = _Broker()
    await SqlL3Backend(broker.proxy()).truncate("lsf_rows")
    assert [subject for subject, _ in broker.requests] == ["test.l3.truncate"]


class _LockingConn:
    """one asyncpg-shaped connection that answers the DDL lock's statements and keeps the rest."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append(sql)
        if "current_database" in sql:
            return [{"database_name": "aibots"}]
        if "pg_try_advisory_lock" in sql:
            return [{"acquired": True}]
        return [{"released": True}]

    async def execute(self, sql: str, *params: Any) -> str:
        self.statements.append(sql)
        return "TRUNCATE TABLE"


class _DirectPool:
    """an asyncpg-shaped pool handing out one connection."""

    def __init__(self) -> None:
        self.conn = _LockingConn()

    @asynccontextmanager
    async def _acquire(self) -> AsyncIterator[_LockingConn]:
        yield self.conn

    def acquire(self) -> Any:
        return self._acquire()


async def test_a_direct_pool_truncates_on_one_connection_holding_the_ddl_lock() -> None:
    pool = _DirectPool()
    await SqlL3Backend(pool).truncate("lsf_rows")
    statements = pool.conn.statements
    assert statements.index("TRUNCATE TABLE lsf_rows") == 2
    assert "pg_try_advisory_lock" in statements[1]
    assert "pg_advisory_unlock" in statements[3]


async def test_truncate_is_refused_inside_a_request_s_bound_transaction() -> None:
    pool = _DirectPool()
    async with bound_request_connection(_LockingConn()):
        with pytest.raises(ValueError, match="outside any transaction"):
            await SqlL3Backend(pool).truncate("lsf_rows")
    assert pool.conn.statements == []


@pytest.mark.parametrize("name", ["lsf_rows; DROP TABLE x", 'lsf"rows', "a.b.c", ""])
async def test_a_table_name_that_is_not_a_plain_identifier_is_refused(name: str) -> None:
    pool = _DirectPool()
    with pytest.raises(ValueError, match="illegal table name"):
        await SqlL3Backend(pool).truncate(name)
    assert pool.conn.statements == []


# -- l3.tx.copy -----------------------------------------------------------------------------------


async def test_copy_sends_the_rows_on_the_transaction_as_a_statement_s_parameters_cross() -> None:
    broker = _Broker()
    proxy = broker.proxy()
    records = [("r1", "g1", _SOURCE, b"\x00\xff", {"k": _WHEN}, _WHEN)]
    columns = ["race", "geo", "source_id", "shape", "detail", "loaded_at"]
    async with proxy.acquire() as conn, conn.transaction():
        tag = await conn.copy_records_to_table("lsf_rows", records=records, columns=columns)
    assert tag == "COPY 1"
    subject, body = broker.requests[1]
    assert subject == "test.l3.tx.copy"
    assert set(body) == {"correlation_id", "identity_token", "tx_id", "table", "columns", "rows"}
    assert body["tx_id"] == _TX_ID
    assert body["table"] == "lsf_rows"
    assert body["columns"] == columns
    [row] = body["rows"]
    assert row[:4] == ["r1", "g1", str(_SOURCE), "\\x00ff"]
    assert row[5] == "2026-10-22T12:30:00+00:00"
    # exactly the parameters a statement carrying the same values sends, value by value
    await proxy.execute("INSERT INTO lsf_rows VALUES ($1, $2, $3, $4, $5, $6)", *records[0])
    assert broker.requests[-1][1]["params"] == row


async def test_copy_outside_a_transaction_is_refused_before_anything_is_sent() -> None:
    broker = _Broker()
    async with broker.proxy().acquire() as conn:
        with pytest.raises(ValueError, match="inside an open transaction"):
            await conn.copy_records_to_table("lsf_rows", records=[("r1", "g1")], columns=["race", "geo"])
    assert broker.requests == []


async def test_a_copy_naming_a_key_already_held_raises_the_unique_violation_a_direct_pool_raises() -> None:
    broker = _Broker(
        {"success": True, "tx_id": _TX_ID},
        {
            "success": False,
            "error_code": CONSTRAINT_VIOLATION_ERROR_CODE,
            "error_message": "duplicate key value violates unique constraint",
            "sqlstate": "23505",
            "constraint_name": "lsf_rows_pkey",
            "table_name": "lsf_rows",
        },
    )
    proxy = broker.proxy()
    with pytest.raises(asyncpg.UniqueViolationError):
        async with proxy.acquire() as conn, conn.transaction():
            await conn.copy_records_to_table("lsf_rows", records=[("r1", "g1")], columns=["race", "geo"])
    assert broker.requests[-1][0] == "test.l3.tx.rollback"


async def test_a_copy_measures_the_request_it_sends() -> None:
    broker = _Broker()
    proxy = broker.proxy()
    records = [("r1", "g1", {"nested": [1, "two"]})]
    async with proxy.acquire() as conn, conn.transaction():
        measured = conn.copy_request_bytes("lsf_rows", ["race", "geo", "detail"], records)
        await conn.copy_records_to_table("lsf_rows", records=records, columns=["race", "geo", "detail"])
    assert measured == len(broker.raw[1])


# -- the SQL store's copy -------------------------------------------------------------------------


class _CopyingConn:
    """an asyncpg-shaped connection that keeps each copy it is asked for."""

    def __init__(self) -> None:
        self.copies: list[tuple[str, list[tuple[Any, ...]], list[str]]] = []

    async def copy_records_to_table(self, table_name: str, *, records: Any, columns: Any) -> str:
        self.copies.append((table_name, list(records), list(columns)))
        return f"COPY {len(records)}"


def _store() -> SqlL3Backend:
    store = SqlL3Backend(object())
    store.register_schema("lsf_rows", SCHEMA)
    return store


async def test_copy_many_copies_the_values_an_insert_binds_ordered_by_key() -> None:
    conn = _CopyingConn()
    assert await _store().copy_many("lsf_rows", _rows(), max_rows=1000, max_bytes=10**9, conn=conn) == 3
    [(table, records, columns)] = conn.copies
    assert table == "lsf_rows"
    assert columns == ["race", "geo", "source_id", "shape", "loaded_at"]
    assert [record[:2] for record in records] == [("r1", "g1"), ("r1", "g2"), ("r2", "g1")]
    by_key = {(row["race"], row["geo"]): row for row in _rows()}
    assert [list(record) for record in records] == [
        build_insert_params(SCHEMA, by_key[record[:2]]) for record in records
    ]


async def test_copy_many_splits_at_the_row_limit() -> None:
    conn = _CopyingConn()
    await _store().copy_many("lsf_rows", _rows(), max_rows=2, max_bytes=10**9, conn=conn)
    assert [len(records) for _, records, _ in conn.copies] == [2, 1]


async def test_a_copy_over_the_rail_is_sized_at_exactly_its_whole_request() -> None:
    # the size a batch is cut at is the size of the message the connection sends for it, to the
    # byte: bounded at one copy's request the rows go in one; a byte under, they do not
    rows = [{"race": "r", "geo": f"{i:04d}", "shape": bytes([i]) * 40, "loaded_at": _WHEN} for i in range(30)]

    async def requests(max_bytes: int) -> list[bytes]:
        broker = _Broker()
        proxy = broker.proxy()
        async with proxy.acquire() as conn, conn.transaction():
            await _store().copy_many("lsf_rows", rows, max_rows=1000, max_bytes=max_bytes, conn=conn)
        return [raw for raw in broker.raw if b'"rows"' in raw]

    [whole] = await requests(10**9)
    assert len(await requests(len(whole))) == 1
    assert len(await requests(len(whole) - 1)) == 2


async def test_copy_many_refuses_rows_that_disagree_on_their_columns() -> None:
    defaulted = TableSchema(
        name="loads",
        primary_key="source",
        columns=[
            Column("source", STRING_TYPE),
            Column("loaded_at", DATETIMETZ_TYPE, nullable=True, server_default="now()"),
        ],
    )
    store = SqlL3Backend(object())
    store.register_schema("loads", defaulted)
    conn = _CopyingConn()
    with pytest.raises(ValueError, match="same columns"):
        await store.copy_many(
            "loads", [{"source": "a"}, {"source": "b", "loaded_at": _WHEN}], max_rows=10, max_bytes=10**9, conn=conn
        )
    assert conn.copies == []


@pytest.mark.parametrize(
    ("column", "value"),
    [
        (Column("detail", JSONB_TYPE, nullable=True), {"k": 1}),
        (Column("v", VECTOR_TYPE, nullable=True, vector_dim=3), [1.0, 2.0, 3.0]),
    ],
    ids=["jsonb", "vector"],
)
async def test_copy_many_refuses_a_column_a_binary_copy_cannot_write(column: Column, value: Any) -> None:
    # a jsonb value binds through the pool's text-format codec, a vector through a cast: a binary
    # COPY has neither, so the load is refused before anything is sent
    table = TableSchema(name="docs", primary_key="id", columns=[Column("id", STRING_TYPE), column])
    store = SqlL3Backend(object())
    store.register_schema("docs", table)
    conn = _CopyingConn()
    with pytest.raises(ValueError, match=f"cannot write the {column.column_type} column {column.name!r}"):
        await store.copy_many("docs", [{"id": "a", column.name: value}], max_rows=10, max_bytes=10**9, conn=conn)
    assert conn.copies == []


async def test_copy_many_refuses_a_table_whose_schema_it_does_not_hold() -> None:
    with pytest.raises(ValueError, match="schema of 'nowhere' registered"):
        await SqlL3Backend(object()).copy_many("nowhere", [{"id": 1}], max_rows=10, max_bytes=1000, conn=_CopyingConn())


async def test_no_rows_copies_nothing() -> None:
    conn = _CopyingConn()
    assert await _store().copy_many("lsf_rows", [], max_rows=10, max_bytes=1000, conn=conn) == 0
    assert conn.copies == []
