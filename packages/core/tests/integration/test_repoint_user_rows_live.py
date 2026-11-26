"""Integration: a user merge's repoint (``merge.repoint_user_rows``) of a stamped table raises the rows' epoch, on a real
database, alone on the pool and inside the caller's ``CallerTransaction`` alike.

A stamped table (``ROW_STAMPED``) is read back by its rows' epochs: a restore of a lost change log reads the rows stamped
since X, and a follower the rows its epoch record names at their new epoch. A repoint that rewrote the owner and left the
epoch where it was changed rows neither would ever hear. The repoint finds the table's schema from its collection class
(``generation.schema_of_table``), as a hub merge of a table it holds no collection for does.

Against the session ``db_container``: Postgres in the ``integration`` suite, YugabyteDB in ``integration_yugabyte``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.merge import repoint_user_rows
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, ROW_STAMPED, next_epoch
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.plans import on_yugabyte
from threetears.core.tenancy import tenant_column

pytestmark = pytest.mark.integration

_TABLE = "repoint_drafts"

#: a stamped table a user owns, as the hub's playbooks and concepts are: keyed by id, its owner a plain column
_DRAFTS = TableSchema(
    name=_TABLE,
    primary_key="id",
    columns=[
        Column("id", STRING_TYPE),
        Column("customer_id", STRING_TYPE),
        Column("user_id", UUID_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE, nullable=True),
    ],
    tenancy=tenant_column("customer_id"),
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)


class _DraftEntity(BaseEntity):
    primary_key_field = "id"


class _DraftCollection(SchemaBackedCollection[_DraftEntity]):
    """the table's collection class: what declares its schema to ``schema_of_table``; never built here."""

    primary_key_column: str = "id"
    schema = _DRAFTS

    @property
    def table_name(self) -> str:
        """return table name."""
        return _TABLE

    @property
    def entity_class(self) -> type[_DraftEntity]:
        """return entity class."""
        return _DraftEntity


@dataclass
class _Held:
    pool: asyncpg.Pool
    store: SqlL3Backend


@pytest.fixture
async def held(db_container: str) -> AsyncIterator[_Held]:
    schema = f"repoint_{uuid.uuid4().hex[:8]}"
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
        for statement in _DRAFTS.create_statements(yugabyte=yugabyte):
            await pool.execute(statement)
        store = SqlL3Backend(pool)
        store.register_schema(_TABLE, _DRAFTS)
        yield _Held(pool=pool, store=store)
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


async def _seeded(held: _Held, user: uuid.UUID, *keys: str) -> dict[str, int]:
    for key in keys:
        assert await held.store.upsert(
            _TABLE, {"id": key, "customer_id": "c1", "user_id": user, EPOCH_COLUMN: next_epoch()}
        )
    return await _epochs(held)


async def _epochs(held: _Held) -> dict[str, int]:
    rows = await held.pool.fetch(f"SELECT id, {EPOCH_COLUMN} FROM {_TABLE}")
    return {row["id"]: row[EPOCH_COLUMN] for row in rows}


async def test_a_repoint_on_the_pool_raises_each_moved_rows_epoch_and_no_other(held: _Held) -> None:
    """Flips if the repoint rewrites a stamped table's owner and leaves its rows' epoch where it was (a restore and every
    follower would never hear the move), or raises the epoch of a row it did not move."""
    source, master, other = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    before = await _seeded(held, source, "d1", "d2")
    before |= await _seeded(held, other, "d3")
    floor = next_epoch()

    moved = await repoint_user_rows(
        held.pool, table=_TABLE, user_column="user_id", pk_columns=["id"], from_user_id=source, to_user_id=master
    )

    assert sorted(moved) == [("d1",), ("d2",)]
    after = await _epochs(held)
    assert after["d1"] > floor and after["d2"] > floor, (before, after)
    assert after["d3"] == before["d3"], "a row the repoint did not move keeps its epoch"
    owners = await held.pool.fetch(f"SELECT id, user_id FROM {_TABLE} ORDER BY id")
    assert [(row["id"], row["user_id"]) for row in owners] == [("d1", master), ("d2", master), ("d3", other)]


async def test_a_repoint_inside_the_callers_transaction_stamps_from_it(held: _Held) -> None:
    """Flips if the repoint a merge runs inside its ``CallerTransaction`` leaves the epoch, or stamps it outside the
    transaction's own stamps (its ``s_min``, what the commit's epoch record and the restore scrub read)."""
    source, master = uuid.uuid4(), uuid.uuid4()
    await _seeded(held, source, "d1")
    floor = next_epoch()

    async with held.pool.acquire() as conn:
        async with CallerTransaction(conn) as transaction:
            moved = await repoint_user_rows(
                conn, table=_TABLE, user_column="user_id", pk_columns=["id"], from_user_id=source, to_user_id=master
            )
            s_min = transaction.s_min

    assert moved == [("d1",)]
    assert s_min is not None and s_min > floor, "the repoint drew its stamp from the caller's transaction"
    assert (await _epochs(held))["d1"] >= s_min
