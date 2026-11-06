"""a bulk write's CPU half runs on a thread, never the event loop; what it sends is unchanged.

A page of a thousand rows is tens of milliseconds of Python: each row's parameters built, each candidate
statement measured as the rail sends it, each value converted and the request passed through JSON several
times. On the loop that held every call the process answers for as long, a page after a page (the reports
tool pod's 60-300 ms stalls in a local LSF pull). These fail if that work runs on the loop's thread again.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from threetears.nats import Subject

import threetears.core.backends.nats_proxy as nats_proxy
import threetears.core.backends.schema_sql as schema_sql
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.schema_backed import DATETIMETZ_TYPE, STRING_TYPE, Column, TableSchema

_TX_ID = "019d9a00-0000-7000-8000-0000000000aa"
_WHEN = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)

SCHEMA = TableSchema(
    name="pages",
    primary_key=("race", "geo"),
    columns=[Column("race", STRING_TYPE), Column("geo", STRING_TYPE), Column("loaded_at", DATETIMETZ_TYPE)],
)


def _rows(n: int) -> list[dict[str, Any]]:
    return [{"race": f"r{i % 7}", "geo": f"g{i:05d}", "loaded_at": _WHEN} for i in range(n)]


class _Threads:
    """the threads a function ran on, recorded by a wrapper patched in its place."""

    def __init__(self) -> None:
        self.seen: set[int] = set()

    def wrap(self, function: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.seen.add(threading.get_ident())
            return function(*args, **kwargs)

        return wrapper


class _Broker:
    """a NATS client stand-in answering every request; keeps the bodies, in order."""

    def __init__(self) -> None:
        self.raw: list[bytes] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del subject, timeout
        self.raw.append(payload)
        return json.dumps({"success": True, "row_count": 1, "tx_id": _TX_ID}).encode("utf-8")

    def proxy(self) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = AsyncMock(side_effect=self.request_raw)
        return NatsProxyL3Backend(
            nats_client=nats, namespace_prefix="test", agent_id="agent-123", identity_token=lambda: "token"
        )


def _store() -> SqlL3Backend:
    store = SqlL3Backend(object())
    store.register_schema("pages", SCHEMA)
    return store


async def test_a_copy_builds_and_sizes_its_rows_off_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _Threads()
    monkeypatch.setattr(schema_sql, "build_insert_params", built.wrap(schema_sql.build_insert_params))
    broker = _Broker()
    async with broker.proxy().acquire() as conn, conn.transaction():
        assert await _store().copy_many("pages", _rows(50), max_rows=20, max_bytes=10**9, conn=conn) == 50
    assert built.seen and threading.get_ident() not in built.seen


async def test_an_upsert_builds_and_measures_its_statements_off_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    built, measured = _Threads(), _Threads()
    monkeypatch.setattr(schema_sql, "build_insert_params", built.wrap(schema_sql.build_insert_params))
    proxy = _Broker().proxy()
    monkeypatch.setattr(proxy, "statement_request_bytes", measured.wrap(proxy.statement_request_bytes))
    store = SqlL3Backend(proxy)
    store.register_schema("pages", SCHEMA)
    assert await store.upsert_many("pages", _rows(50), max_rows=20, max_bytes=10**9) == 50
    loop_thread = threading.get_ident()
    assert built.seen and loop_thread not in built.seen
    assert measured.seen and loop_thread not in measured.seen


async def test_a_copy_page_is_converted_and_encoded_off_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    converted, encoded = _Threads(), _Threads()
    monkeypatch.setattr(nats_proxy, "transport_param", converted.wrap(nats_proxy.transport_param))
    broker = _Broker()
    proxy = broker.proxy()
    monkeypatch.setattr(proxy, "encode_request", encoded.wrap(proxy.encode_request))
    records = [(f"r{i}", f"g{i}", _WHEN) for i in range(10)]
    async with proxy.acquire() as conn, conn.transaction():
        encoded.seen.clear()
        await conn.copy_records_to_table("pages", records=records, columns=["race", "geo", "loaded_at"])
        loop_thread = threading.get_ident()
        assert converted.seen and loop_thread not in converted.seen
        assert encoded.seen and loop_thread not in encoded.seen


async def test_a_statement_binding_a_page_of_parameters_is_encoded_off_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = _Threads()
    proxy = _Broker().proxy()
    monkeypatch.setattr(proxy, "encode_request", encoded.wrap(proxy.encode_request))
    params = [f"v{i}" for i in range(300)]
    await proxy.execute("INSERT INTO pages VALUES " + ", ".join(f"(${i + 1})" for i in range(300)), *params)
    assert encoded.seen and threading.get_ident() not in encoded.seen


async def test_a_small_statement_is_encoded_where_it_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    # a thread hop costs more than a few parameters' encoding; only a page's statement takes one
    encoded = _Threads()
    proxy = _Broker().proxy()
    monkeypatch.setattr(proxy, "encode_request", encoded.wrap(proxy.encode_request))
    await proxy.execute("INSERT INTO pages VALUES ($1, $2)", "a", "b")
    assert encoded.seen == {threading.get_ident()}


async def test_a_page_sent_off_the_loop_carries_the_statement_and_its_parameters_as_before() -> None:
    params = [f"v{i}" for i in range(299)] + [_WHEN]
    query = "INSERT INTO pages VALUES " + ", ".join(f"(${i + 1})" for i in range(300))
    broker = _Broker()
    await broker.proxy().execute(query, *params)
    sent = json.loads(broker.raw[-1])
    assert (sent["query"], sent["operation"]) == (query, "insert")
    assert sent["params"] == [schema_sql.transport_param(value) for value in params]
