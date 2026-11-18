"""Deleting whole key prefixes: one key-led statement a group, nothing read back, settled by the prefixes.

The contract this pins (reports' gone LSF runs, 2026-11-17: 23 runs dropped row by row after being read back cost
794 s on the 2026-11-02 replay):

- ``delete_prefixes`` deletes a group of prefixes that differ only in their last column in ONE key-led ``DELETE``,
  reads no row, and answers the rows the database deleted; the statements are the same however many rows the
  prefixes hold;
- the delete is marked whole for the change notes (one note naming the prefixes), and the marks it replaced are put
  back right after, so the transaction's other writes are noted as before;
- when the caller's transaction ends, the table is settled by the prefixes: each derived cache that takes a partition
  (a copy follower) is told each prefix, and the generation advances once, its epoch naming the prefixes;
- a prefix that does not lead the key, a write outside a caller's transaction, and a keyed write of the same table in
  the same transaction are refused;
- on the L3 rail the connection's own request carries the table and the prefixes (no SQL, no ``set_config``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections import WRITE_GENERATION, CollectionRegistry
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.epoch_change import ChangedRow, EpochChange
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.change_notes import NOTE_REACH_SETTING, NOTE_WHOLE_SETTING
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.l1_tables import TEST_L1_TABLE

from packages.core.tests.durable_support import ConnOnlyPool

RUNS = TableSchema(
    name="lsf_runs",
    primary_key=("state", "run_id", "geo"),
    columns=[
        Column("state", STRING_TYPE),
        Column("run_id", BIGINT_TYPE),
        Column("geo", STRING_TYPE),
        Column("votes", BIGINT_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    l1_eviction=TEST_L1_TABLE,
)

_GUARD = "threetears.noted_whole_4242"


class _Run(BaseEntity):
    primary_key_field = "geo"


class _Runs(collection_for_schema(RUNS, entity_class=_Run)):  # type: ignore[misc]
    write_generation = WRITE_GENERATION


class _Conn:
    """a caller's connection that records every statement and answers a prefix's DELETE with the rows it holds."""

    def __init__(self, held: Mapping[tuple[str, int], int] | None = None) -> None:
        #: rows each (state, run) holds
        self.held = dict(held or {})
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.fetched: list[tuple[str, tuple[Any, ...]]] = []
        #: what the mark statement found set before it
        self.prior: dict[str, Any] = {"whole": None, "reach": None, "noted": None, "guard": _GUARD}

    def transaction(self, **options: Any) -> Any:
        return _NoTransaction()

    async def fetchrow(self, sql: str, *params: Any) -> dict[str, Any]:
        self.fetched.append((sql, params))
        assert "set_config" in sql, f"a prefix delete reads no row: {sql!r}"
        return dict(self.prior)

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        raise AssertionError(f"a prefix delete reads no row: {sql!r}")

    async def execute(self, sql: str, *params: Any) -> str:
        self.executed.append((sql, params))
        if not sql.startswith("DELETE"):
            return "SELECT 1"
        states = params[0] if isinstance(params[0], list) else [params[0]]
        runs = params[1] if len(params) > 1 else None
        rows = sum(
            count
            for (held_state, run), count in self.held.items()
            if held_state in states and (runs is None or run in (runs if isinstance(runs, list) else [runs]))
        )
        return f"DELETE {rows}"


class _NoTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _Source:
    """a generation source that records each advance's change."""

    def __init__(self) -> None:
        self.changes: list[EpochChange] = []

    async def current(self, table_name: str) -> str:
        return f"inc:{len(self.changes)}"

    async def advance(self, table_name: str) -> str:
        return f"inc:{len(self.changes) + 1}"

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        partition: Mapping[str, str] | None = None,
        prefixes: Sequence[Mapping[str, str]] | None = None,
        reason: str | None = None,
    ) -> str:
        token = f"inc:{len(self.changes) + 1}"
        self.changes.append(
            EpochChange(
                table=table_name,
                token=token,
                whole_table=rows is None and not keys_unknown,
                keys_unknown=keys_unknown,
                rows=list(rows) if rows is not None else [],
                partition=dict(partition) if partition else None,
                prefixes=[dict(prefix) for prefix in prefixes] if prefixes else None,
                reason=reason,
            )
        )
        return token


def _runs(source: _Source | None = None) -> tuple[Any, list[Any]]:
    registry = CollectionRegistry()
    if source is not None:
        registry.set_generation_source(source)
    heard: list[Any] = []
    registry.register_derived_cache(
        RUNS.name,
        on_row=lambda message: heard.append(("row", message)),
        on_table_dropped=lambda: heard.append("dropped"),
        on_partition=lambda prefix: heard.append(dict(prefix)),
    )
    collection = _Runs(registry, DefaultCoreConfig(), None)
    collection.l3_pool = ConnOnlyPool()
    return collection, heard


def _deletes(conn: _Conn) -> list[tuple[str, tuple[Any, ...]]]:
    return [(sql, params) for sql, params in conn.executed if sql.startswith("DELETE")]


async def test_a_group_of_prefixes_is_one_key_led_delete_and_nothing_is_read_back() -> None:
    collection, _ = _runs()
    conn = _Conn({("PA", 1): 10, ("PA", 2): 20, ("PA", 3): 5, ("OH", 7): 3})
    async with CallerTransaction(conn):
        deleted = await collection.delete_prefixes(
            [{"state": "PA", "run_id": 1}, {"state": "PA", "run_id": 2}, {"state": "OH"}], conn=conn
        )
    assert deleted == 33
    assert _deletes(conn) == [
        ("DELETE FROM lsf_runs WHERE state = $1 AND run_id = ANY($2::bigint[])", ("PA", [1, 2])),
        ("DELETE FROM lsf_runs WHERE state = ANY($1::text[])", (["OH"],)),
    ]
    # the one row read is the marks' own answer, never a row of the table
    [(mark, _)] = conn.fetched
    assert "lsf_runs" not in mark


async def test_the_statements_are_the_same_however_many_rows_the_prefixes_hold() -> None:
    statements = []
    for rows in (1_000, 100_000):
        collection, _ = _runs()
        conn = _Conn({("PA", 1): rows})
        async with CallerTransaction(conn):
            assert await collection.delete_prefixes([{"state": "PA", "run_id": 1}], conn=conn) == rows
        statements.append((conn.executed, len(conn.fetched)))
    assert statements[0] == statements[1]


async def test_the_delete_is_noted_whole_by_its_prefixes_and_the_marks_it_replaced_are_put_back() -> None:
    collection, _ = _runs()
    conn = _Conn({("PA", 1): 4})
    conn.prior = {"whole": "1", "reach": None, "noted": None, "guard": _GUARD}
    async with CallerTransaction(conn):
        await collection.delete_prefixes([{"state": "PA", "run_id": 1}], conn=conn)
    [(_, marked)] = conn.fetched
    assert marked == (NOTE_WHOLE_SETTING, NOTE_REACH_SETTING, '{"prefixes":[{"state":"PA","run_id":"1"}]}', "lsf_runs")
    restore, restored = conn.executed[-1]
    assert restore.startswith("SELECT set_config")
    # an earlier whole mark of the transaction stays; the reach and the table's guard go back to unset
    assert restored == (NOTE_WHOLE_SETTING, "1", NOTE_REACH_SETTING, "", _GUARD, "")


async def test_the_table_is_settled_by_its_prefixes_once_the_transaction_ends() -> None:
    source = _Source()
    collection, heard = _runs(source)
    conn = _Conn({("PA", 1): 4})
    async with CallerTransaction(conn):
        await collection.delete_prefixes([{"state": "PA", "run_id": 1}, {"state": "OH"}], conn=conn)
        assert heard == [] and source.changes == []
    assert heard == [{"state": "PA", "run_id": "1"}, {"state": "OH"}]
    [change] = source.changes
    assert change.whole_table and change.rows == []
    assert change.prefixes == [{"state": "PA", "run_id": "1"}, {"state": "OH"}]


async def test_a_rolled_back_delete_is_settled_all_the_same() -> None:
    source = _Source()
    collection, heard = _runs(source)
    conn = _Conn({("PA", 1): 4})
    with pytest.raises(RuntimeError, match="caller failed"):
        async with CallerTransaction(conn):
            await collection.delete_prefixes([{"state": "PA", "run_id": 1}], conn=conn)
            raise RuntimeError("caller failed")
    # a broader invalidation, never a missed one
    assert heard == [{"state": "PA", "run_id": "1"}] and len(source.changes) == 1


async def test_a_prefix_that_does_not_lead_the_key_deletes_nothing() -> None:
    collection, _ = _runs()
    conn = _Conn()
    async with CallerTransaction(conn):
        with pytest.raises(ValueError, match="not a key prefix"):
            await collection.delete_prefixes([{"run_id": 1}], conn=conn)
    assert conn.executed == [] and conn.fetched == []


async def test_a_delete_outside_a_caller_s_transaction_is_refused() -> None:
    collection, _ = _runs()
    with pytest.raises(ValueError, match="CallerTransaction"):
        await collection.delete_prefixes([{"state": "PA"}], conn=_Conn())


async def test_a_table_takes_prefix_deletes_or_keyed_writes_in_one_transaction_never_both() -> None:
    collection, _ = _runs()
    conn = _Conn({("PA", 1): 4})
    async with CallerTransaction(conn):
        await collection.delete_prefixes([{"state": "PA", "run_id": 1}], conn=conn)
        with pytest.raises(ValueError, match="one epoch"):
            await collection.delete_rows([("PA", 2, "g1")], conn=conn)
    conn = _Conn({("PA", 1): 4})
    async with CallerTransaction(conn):
        await collection.delete_rows([("PA", 2, "g1")], conn=conn)
        with pytest.raises(ValueError, match="one epoch"):
            await collection.delete_prefixes([{"state": "PA", "run_id": 1}], conn=conn)


class _RailPool(ConnOnlyPool):
    """the L3 rail's client, as the SQL store sees it."""

    accepts_scoped_reads = True


class _RailConn(_Conn):
    """a rail connection: the broker deletes the prefixes it is named."""

    def __init__(self) -> None:
        super().__init__()
        self.asked: list[tuple[str, list[dict[str, Any]]]] = []

    async def delete_prefixes(self, table: str, prefixes: Sequence[Mapping[str, Any]]) -> int:
        self.asked.append((table, [dict(prefix) for prefix in prefixes]))
        return 12


async def test_on_the_rail_the_broker_is_named_the_table_and_the_prefixes_and_sent_no_sql() -> None:
    store = SqlL3Backend(_RailPool())
    store.register_schema("lsf_runs", RUNS)
    conn = _RailConn()
    assert await store.delete_prefixes("lsf_runs", [{"state": "PA", "run_id": 1}], conn=conn) == 12
    assert conn.asked == [("lsf_runs", [{"state": "PA", "run_id": 1}])]
    assert conn.executed == [] and conn.fetched == []
