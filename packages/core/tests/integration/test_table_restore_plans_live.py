"""Integration: a restore from the tables reads each scope through its (scope, epoch) index, and finds its scopes through
the scope epochs table's last_epoch index; never a whole read of either.

The plan tests the two indexes name (``TableSchema(row_epoch=row_versions(...))``'s ``<table>_scope_epoch``, and
``scope_epochs_schema``'s ``<name>_last_epoch``). A versioned table and its scope epochs table are seeded past where the
planner would rather scan (56 states of versions, most of them older than X; thousands of race scopes beside the
states), and the restore's own stream (:class:`~threetears.core.collections.table_restore.TableChanges`) runs against
them: every statement it sends is planned, counted and timed.

Against the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them there.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import asyncpg
import pytest

from threetears.core.change_note_fold import RestoreFolder
from threetears.core.collections.run_markers import RUN_MARKERS_SCHEMA
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.collections.scope_epochs import scope_epochs_schema
from threetears.core.collections.table_restore import TableChanges, versioned_tables
from threetears.core.ddl.row_epoch import restore_floor, row_versions, scope_epoch_index_name
from threetears.core.eviction import Declared, Policy
from threetears.core.tenancy import TENANT_BY_SCHEMA
from threetears.core.testing.plans import StatementPlan, explain_statement, on_yugabyte, plan_uses_index

pytestmark = pytest.mark.integration

_TABLE = "plan_restore_results"
_EPOCHS = "plan_restore_epochs"
_SCOPE = "join_state_code"
_STATES = 56
_VERSIONS_PER_STATE = 2000
_RACES = 3000
#: the states a stamped write touched since X
_CHANGED = 10
#: of each changed state's versions, how many are stamped at X or later
_RECENT = 200

_RESULTS = TableSchema(
    name=_TABLE,
    primary_key=(_SCOPE, "run_id", "candidate"),
    columns=[
        Column(_SCOPE, STRING_TYPE),
        Column("run_id", STRING_TYPE),
        Column("candidate", STRING_TYPE),
        Column("votes", BIGINT_TYPE),
    ],
    tenancy=TENANT_BY_SCHEMA,
    row_epoch=row_versions(_SCOPE, epochs=_EPOCHS, scope_prefix="state:"),
    l1_eviction=Declared(Policy.EVICTABLE, "a plan test's versioned table"),
)


@dataclass
class _Seeded:
    conn: asyncpg.Connection
    schema: str
    floor: int


@dataclass
class _Recorder:
    """the restore's pool over one connection: every statement recorded with its arguments, then run."""

    conn: asyncpg.Connection
    sent: list[tuple[str, str, tuple[Any, ...]]] = field(default_factory=list)
    #: the seconds the database took, all statements
    seconds: float = 0.0

    async def fetch(self, query: str, *args: Any) -> Any:
        self.sent.append(("fetch", query, args))
        began = time.perf_counter()
        try:
            return await self.conn.fetch(query, *args)
        finally:
            self.seconds += time.perf_counter() - began

    async def copy_from_query(self, query: str, *args: Any, **options: Any) -> Any:
        self.sent.append(("copy", query, args))
        began = time.perf_counter()
        try:
            return await self.conn.copy_from_query(query, *args, **options)
        finally:
            self.seconds += time.perf_counter() - began


def _state(index: int) -> str:
    return f"S{index:02d}"


@pytest.fixture
async def seeded(db_container: str) -> AsyncIterator[_Seeded]:
    """a versioned table of 56 states' versions, its scope epochs table with every state and thousands of races."""
    schema = f"tablerestore_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=600)
    try:
        yugabyte = await on_yugabyte(conn)
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(f"SET search_path TO {schema}")
        for declared in (_RESULTS, scope_epochs_schema(_EPOCHS), RUN_MARKERS_SCHEMA):
            for statement in declared.create_statements(yugabyte=yugabyte):
                await conn.execute(statement)
        floor = restore_floor(int(time.time() * 1000) - 60_000)
        # every state's versions, the oldest far below X; each changed state's last _RECENT at X or later
        await conn.execute(
            f"INSERT INTO {_TABLE} ({_SCOPE}, run_id, candidate, votes, epoch, gone) "  # noqa: S608 - test identifiers
            "SELECT 'S' || lpad(s::text, 2, '0'), 'r1', 'c' || v, v, "
            "CASE WHEN s < $2 AND v > $3 - $4 THEN $1::bigint + v ELSE $1::bigint - (1000000 + v) END, v % 97 = 0 "
            "FROM generate_series(0, $5 - 1) s, generate_series(1, $3) v",
            floor,
            _CHANGED,
            _VERSIONS_PER_STATE,
            _RECENT,
            _STATES,
        )
        await conn.execute(
            f"INSERT INTO {_EPOCHS} (scope, epoch, last_epoch, date_created, date_updated) "  # noqa: S608
            "SELECT 'state:S' || lpad(s::text, 2, '0'), 1, CASE WHEN s < $2 THEN $1::bigint + 1 ELSE $1::bigint - 1000000 END, now(), now() "
            "FROM generate_series(0, $3 - 1) s",
            floor,
            _CHANGED,
            _STATES,
        )
        await conn.execute(
            f"INSERT INTO {_EPOCHS} (scope, epoch, last_epoch, date_created, date_updated) "  # noqa: S608
            "SELECT 'race:r' || g, 1, CASE WHEN g % 30 = 0 THEN $1::bigint + 5 ELSE $1::bigint - 1000000 - g END, now(), now() "
            "FROM generate_series(1, $2) g",
            floor,
            _RACES,
        )
        await conn.execute(
            "INSERT INTO threetears_run_markers (table_name, scope, prefix, epoch) VALUES "
            "($1, 'S00', 'S00' || chr(31) || 'r0', $2), ($1, 'S01', 'S01' || chr(31) || 'r0', $3)",
            _TABLE,
            floor + 7,
            floor - 7,
        )
        await conn.execute(f"ANALYZE {_TABLE}")
        await conn.execute(f"ANALYZE {_EPOCHS}")
        yield _Seeded(conn=conn, schema=schema, floor=floor)
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def _restored(seeded: _Seeded) -> tuple[_Recorder, int, float, int]:
    """the restore's stream run whole: its recorder, the keyed rows its records name, its seconds, its changes."""
    tables = [table for table in await versioned_tables(seeded.conn) if table.schema == seeded.schema]
    assert [table.table for table in tables] == [_TABLE], tables
    recorder = _Recorder(seeded.conn)
    rows = 0
    changes = 0
    async with RestoreFolder() as folder:
        folder.start()
        # the worker's spawn is the restore's fixed cost, not its reads: timed apart
        await folder.fold(b"", final=True, form=(b"",) * 6, chunk_rows=1, max_bytes=1 << 20)
        began = time.perf_counter()
        async for change in TableChanges(recorder, tables, seeded.floor, folder, None):
            changes += 1
            rows += sum(body.count(b'"ids":') for body in change.bodies)
        took = time.perf_counter() - began
    return recorder, rows, took, changes


async def _plans(seeded: _Seeded, recorder: _Recorder, kind: str, relation: str) -> list[StatementPlan]:
    return [
        await explain_statement(seeded.conn, query, *args)
        for sent, query, args in recorder.sent
        if sent == kind and relation in query
    ]


async def test_a_restore_reads_a_scope_through_the_scope_epoch_index(seeded: _Seeded) -> None:
    """Flips if a restore's read of one scope's versions since X stops planning through the (scope, epoch) index, reads
    any scope no stamped write touched since X, or misses a version stamped at X or later."""
    recorder, rows, took, changes = await _restored(seeded)
    copies = [query for sent, query, _ in recorder.sent if sent == "copy"]
    assert len(copies) == _CHANGED, "one statement a scope a stamped write touched since X"
    assert rows == _CHANGED * _RECENT, "every version stamped at X or later, gone markers included"
    plans = await _plans(seeded, recorder, "copy", _TABLE)
    index = scope_epoch_index_name(_TABLE)
    assert plans and all(plan_uses_index(plan, index) for plan in plans), [plan.plan for plan in plans]
    print(  # noqa: T201 -- the measure the brief asks for, read in the run's output
        f"restore from the tables: {len(recorder.sent)} statements ({len(copies)} scope reads), {rows} rows, "
        f"{changes} changes, {took * 1000:.1f} ms, of it the database {recorder.seconds * 1000:.1f} ms"
    )


async def test_a_restore_finds_its_scopes_through_the_last_epoch_index(seeded: _Seeded) -> None:
    """Flips if a restore's read of the scopes a stamped write touched since X stops planning through the scope epochs
    table's last_epoch index (a read of every scope's row for the few that changed)."""
    recorder, _rows, _took, _changes = await _restored(seeded)
    plans = await _plans(seeded, recorder, "fetch", _EPOCHS)
    assert plans and all(plan_uses_index(plan, f"{_EPOCHS}_last_epoch") for plan in plans), [p.plan for p in plans]
    markers = await _plans(seeded, recorder, "fetch", "threetears_run_markers")
    assert markers, "the run markers of the changed scopes are read by their key"
