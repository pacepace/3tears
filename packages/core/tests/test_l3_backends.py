"""L3 backend protocols: conformance + SqlL3Backend SQL generation + a non-SQL DurableStore.

Proves the L3 tier is genuinely two-layered and storage-agnostic:
- ``NatsProxyL3Backend`` and ``SqlL3Backend`` conform to the raw-SQL ``L3Backend``.
- ``SqlL3Backend`` ALSO conforms to the structured ``DurableStore`` and generates the
  expected parameterized SQL for fetch_one / upsert / delete / scan.
- a **non-SQL** in-memory ``DurableStore`` (the shape a ``GitL3Backend`` takes) conforms
  to ``DurableStore`` and round-trips — no SQL anywhere. This is the seam that makes a
  git working tree a first-class L3 backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.core.backends import DurableStore, L3Backend, NatsProxyL3Backend, SqlL3Backend, parse_rowcount


class _RecordingPool:
    """An asyncpg-pool stand-in that records every (sql, params) and returns canned rows."""

    def __init__(self, fetchrow_result: dict[str, Any] | None = None, rows: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._fetchrow_result = fetchrow_result
        self._rows = rows or []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.calls.append((query, params))
        return self._rows

    async def fetchrow(self, query: str, *params: Any) -> dict[str, Any] | None:
        self.calls.append((query, params))
        return self._fetchrow_result

    async def execute(self, query: str, *params: Any) -> str:
        self.calls.append((query, params))
        return "UPDATE 1"


def test_sql_l3_backend_conforms_to_both_protocols() -> None:
    backend = SqlL3Backend(_RecordingPool())
    assert isinstance(backend, L3Backend)
    assert isinstance(backend, DurableStore)


def test_nats_proxy_conforms_to_l3backend() -> None:
    # structural conformance — the raw-SQL transport surface is present on the class.
    # (isinstance against the runtime_checkable protocol needs a live instance, which
    # NatsProxyL3Backend cannot build without a NATS client; the method surface is the
    # contract the protocol asserts.)
    assert all(
        hasattr(NatsProxyL3Backend, m)
        for m in ("fetch", "fetchrow", "execute", "execute_batch", "acquire", "transaction")
    )


@pytest.mark.asyncio
async def test_sql_durable_store_generates_expected_sql() -> None:
    # fetch_one → SELECT … WHERE pk
    pool = _RecordingPool(fetchrow_result={"id": "e1", "name": "Alice"})
    got = await SqlL3Backend(pool).fetch_one("widgets", {"id": "e1"})
    assert got == {"id": "e1", "name": "Alice"}
    sql, params = pool.calls[-1]
    assert sql == 'SELECT * FROM "widgets" WHERE "id" = $1'
    assert params == ("e1",)

    # upsert (on_conflict=update) → INSERT … ON CONFLICT … DO UPDATE SET mutable=EXCLUDED
    pool = _RecordingPool()
    n = await SqlL3Backend(pool).upsert("widgets", {"id": "e1", "name": "Bob"}, pk=["id"])
    assert n == 1  # parse_rowcount("UPDATE 1")
    sql, params = pool.calls[-1]
    assert (
        sql
        == 'INSERT INTO "widgets" ("id", "name") VALUES ($1, $2) ON CONFLICT ("id") DO UPDATE SET "name" = EXCLUDED."name"'
    )
    assert params == ("e1", "Bob")

    # upsert with a CAS fence → adds WHERE table.date_updated = $n
    pool = _RecordingPool()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    await SqlL3Backend(pool).upsert("widgets", {"id": "e1", "date_updated": ts}, pk=["id"], cas=ts)
    sql, params = pool.calls[-1]
    assert sql.endswith('WHERE "widgets"."date_updated" = $3')
    assert params == ("e1", ts, ts)

    # delete → DELETE … WHERE pk (composite)
    pool = _RecordingPool()
    await SqlL3Backend(pool).delete("links", {"a": 1, "b": 2})
    sql, params = pool.calls[-1]
    assert sql == 'DELETE FROM "links" WHERE "a" = $1 AND "b" = $2'
    assert params == (1, 2)

    # scan with filters → SELECT … WHERE; scan() with none → SELECT *
    pool = _RecordingPool(rows=[{"id": "e1"}])
    await SqlL3Backend(pool).scan("widgets", {"kind": "gadget"})
    assert pool.calls[-1][0] == 'SELECT * FROM "widgets" WHERE "kind" = $1'
    await SqlL3Backend(pool).scan("widgets")
    assert pool.calls[-1][0] == 'SELECT * FROM "widgets"'


def test_sql_l3_backend_rejects_injecting_identifiers() -> None:
    from threetears.core.backends.sql import _quote_ident

    assert _quote_ident("name") == '"name"'
    with pytest.raises(ValueError):  # a quote in a column name must not slip into generated SQL
        _quote_ident('id"; DROP TABLE x; --')


class _InMemoryDurableStore:
    """A NON-SQL ``DurableStore`` — entities live in a plain nested dict, keyed by pk.

    This is the exact shape a ``GitL3Backend`` presents (fetch/upsert/delete/scan a record),
    minus the git I/O. Its mere existence + conformance proves the structured contract is
    satisfiable with zero SQL.
    """

    def __init__(self) -> None:
        self.tables: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}

    @staticmethod
    def _key(pk: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(pk[k] for k in sorted(pk))

    async def fetch_one(self, table: str, pk: Mapping[str, Any]) -> dict[str, Any] | None:
        return self.tables.get(table, {}).get(self._key(pk))

    async def upsert(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: Sequence[str],
        on_conflict: str = "update",
        cas: datetime | None = None,
    ) -> int:
        t = self.tables.setdefault(table, {})
        key = tuple(row[c] for c in sorted(pk))
        if cas is not None and key in t and t[key].get("date_updated") != cas:
            return 0  # optimistic-lock miss
        t[key] = dict(row)
        return 1

    async def delete(self, table: str, pk: Mapping[str, Any]) -> None:
        self.tables.get(table, {}).pop(self._key(pk), None)

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        rows = list(self.tables.get(table, {}).values())
        if not filters:
            return rows
        return [r for r in rows if all(r.get(k) == v for k, v in filters.items())]


@pytest.mark.asyncio
async def test_non_sql_durable_store_conforms_and_round_trips() -> None:
    store = _InMemoryDurableStore()
    assert isinstance(store, DurableStore)  # a non-SQL backend satisfies the contract
    assert not isinstance(store, L3Backend)  # and is NOT required to implement the raw-SQL transport

    assert await store.upsert("scenes", {"id": "scn-1", "text": "alpha"}, pk=["id"]) == 1
    assert await store.fetch_one("scenes", {"id": "scn-1"}) == {"id": "scn-1", "text": "alpha"}
    assert await store.scan("scenes", {"text": "alpha"}) == [{"id": "scn-1", "text": "alpha"}]
    await store.delete("scenes", {"id": "scn-1"})
    assert await store.fetch_one("scenes", {"id": "scn-1"}) is None


def test_parse_rowcount() -> None:
    assert parse_rowcount("INSERT 0 1") == 1
    assert parse_rowcount("UPDATE 3") == 3
    assert parse_rowcount("DELETE 0") == 0
    assert parse_rowcount(None) == 0
    assert parse_rowcount("") == 0
    assert parse_rowcount(42) == 0  # non-string → 0 (mock-pool safety)
