"""Integration: a write that moves a stamped row between tenants appends its move in the same statement, on a real
database, and a restore reads every move since X (``threetears.core.ddl.tenant_moves``).

What the unit tests cannot prove: that the one statement a moving write becomes (the write in a ``WITH``, its moves
appended from the rows it returned, the caller's answer selected from them) runs on the engine, the upsert's, the keyed
update's and the bulk upsert's alike; that a write that does not move the row appends nothing; that a row moved A to
B, then B to C, comes back from a restore of a lost log for A and B both, through the catalog's own restore
declaration and the moves table's key; and that cleanup's delete of the moves below its floor runs.

Against the session ``db_container``: Postgres in the ``integration`` suite, YugabyteDB in ``integration_yugabyte``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.change_note_fold import RestoreFolder
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.run_markers import RUN_MARKERS_SCHEMA
from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema, stamped_values
from threetears.core.collections.table_restore import TableChanges, restored_tables
from threetears.core.collections.tenant_moves import TENANT_MOVES_SCHEMA, clear_moves
from threetears.core.ddl.epoch_catalog import refuse_moveable_without_moves
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, ROW_STAMPED, next_epoch
from threetears.core.ddl.tenant_moves import TENANT_MOVES_TABLE
from threetears.core.keyed import update_keyed
from threetears.core.tenancy import tenant_column, value_tenant
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.plans import on_yugabyte

pytestmark = pytest.mark.integration

_TABLE = "tm_widgets"

#: a stamped table whose tenant column (and office, a declared invalidation column) a write may change in place
_WIDGETS = TableSchema(
    name=_TABLE,
    primary_key="id",
    columns=[Column("id", STRING_TYPE), Column("customer_id", STRING_TYPE), Column("office", STRING_TYPE)],
    tenancy=tenant_column("customer_id"),
    row_epoch=ROW_STAMPED,
    moveable=("customer_id", "office"),
    invalidation_columns=("office",),
    l1_eviction=TEST_L1_TABLE,
)


@dataclass
class _Held:
    pool: asyncpg.Pool
    schema: str
    store: SqlL3Backend


@pytest.fixture
async def held(db_container: str) -> AsyncIterator[_Held]:
    schema = f"tenant_moves_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(
        db_container, min_size=1, max_size=4, init=init_connection, server_settings={"search_path": schema}
    )
    assert pool is not None
    try:
        async with pool.acquire() as conn:
            yugabyte = await on_yugabyte(conn)
        # the moves table first: a keeper starting beside this test never sees the moveable table without it
        for table in (TENANT_MOVES_SCHEMA, _WIDGETS, RUN_MARKERS_SCHEMA):
            for statement in table.create_statements(yugabyte=yugabyte):
                await pool.execute(statement)
        store = SqlL3Backend(pool)
        store.register_schema(_TABLE, _WIDGETS)
        yield _Held(pool=pool, schema=schema, store=store)
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


def _row(key: str, customer: str, office: str = "gov") -> dict[str, Any]:
    return {"id": key, "customer_id": customer, "office": office, EPOCH_COLUMN: next_epoch()}


async def _moves(held: _Held) -> list[tuple[str, str | None, dict[str, Any]]]:
    rows = await held.pool.fetch(f"SELECT keys, tenant, columns FROM {TENANT_MOVES_TABLE} ORDER BY epoch")
    return [(json.loads(row["keys"])[0], row["tenant"], json.loads(row["columns"])) for row in rows]


async def test_an_upsert_that_moves_a_row_appends_each_move_and_a_restore_hears_every_tenant_it_left(
    held: _Held,
) -> None:
    """Flips if the moving upsert's one statement does not run, writes the row without appending its move, appends one
    for a write that did not move it, or if a restore from X hears a row moved A to B to C for C and B alone."""
    floor = next_epoch()
    assert await held.store.upsert(_TABLE, _row("w1", "ca")) == 1
    assert await held.store.upsert(_TABLE, _row("w1", "ca")) == 1, "a write that does not move the row"
    assert await held.store.upsert(_TABLE, _row("w1", "cb")) == 1
    assert await held.store.upsert(_TABLE, _row("w1", "cc", "sen")) == 1

    assert await _moves(held) == [
        ("w1", value_tenant("ca"), {"office": "gov", "customer_id": "ca"}),
        ("w1", value_tenant("cb"), {"office": "gov", "customer_id": "cb"}),
    ]

    tables = [table for table in await restored_tables(held.pool) if table.schema == held.schema]
    folder = RestoreFolder()
    folder.start()
    try:
        changes = [change async for change in TableChanges(held.pool, tables, floor, folder, None)]
    finally:
        await folder.close()
        await folder.closed()
    heard = sorted({change.tenant for change in changes if change.table == _TABLE}, key=str)
    assert heard == sorted([value_tenant("ca"), value_tenant("cb"), value_tenant("cc")], key=str), changes


async def test_a_keyed_update_and_a_bulk_upsert_append_their_moves_in_the_same_statement(held: _Held) -> None:
    """Flips if a keyed update of ``stamped_values`` or a bulk upsert that moves rows fails on the engine, answers a
    count other than the rows it wrote, or appends a move for a row it did not move."""
    assert await held.store.upsert(_TABLE, _row("w1", "ca")) == 1
    assert await held.store.upsert(_TABLE, _row("w2", "ca")) == 1

    written = await update_keyed(
        held.pool, _TABLE, values=stamped_values(_WIDGETS, {"customer_id": "cb"}), by={"id": "w1"}
    )
    assert written.count == 1
    returned = await update_keyed(
        held.pool,
        _TABLE,
        values=stamped_values(_WIDGETS, {"customer_id": "cb"}),
        by={"id": "w1"},
        returning=("customer_id",),
    )
    assert returned.count == 1 and returned.rows == [{"customer_id": "cb"}], "an update that did not move the row"
    assert await _moves(held) == [("w1", value_tenant("ca"), {"office": "gov", "customer_id": "ca"})]

    written_many = await held.store.upsert_many(
        _TABLE, [_row("w1", "cb", "sen"), _row("w2", "cc"), _row("w3", "cd")], max_rows=100, max_bytes=1 << 20
    )
    assert written_many == 3
    assert (await _moves(held))[1:] == [
        ("w1", value_tenant("cb"), {"office": "gov", "customer_id": "cb"}),
        ("w2", value_tenant("ca"), {"office": "gov", "customer_id": "ca"}),
    ]


async def test_cleanup_deletes_the_moves_below_its_floor(held: _Held) -> None:
    """Flips if cleanup's delete of a table's moves below its floor does not run, or removes one at or above it."""
    await held.store.upsert(_TABLE, _row("w1", "ca"))
    await held.store.upsert(_TABLE, _row("w1", "cb"))
    floor = next_epoch()
    await held.store.upsert(_TABLE, _row("w1", "cc"))
    assert await clear_moves(held.pool, _WIDGETS, floor) == 1
    assert await _moves(held) == [("w1", value_tenant("cb"), {"office": "gov", "customer_id": "cb"})]


async def test_the_start_check_passes_beside_the_moves_table_and_refuses_without_it(held: _Held) -> None:
    """Flips if the catalog read of the start check does not run on the engine, refuses a moveable table whose schema
    holds the moves table, or passes one whose schema lost it."""
    await refuse_moveable_without_moves(held.pool, held.schema)
    await held.pool.execute(f"DROP TABLE {TENANT_MOVES_TABLE}")
    with pytest.raises(RuntimeError, match="REFUSED: a moveable stamped table") as refused:
        await refuse_moveable_without_moves(held.pool, held.schema)
    assert f"  - {held.schema}.{_TABLE} (moveable: customer_id, office)\n" in str(refused.value)
