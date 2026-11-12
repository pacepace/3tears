"""integration: langgraph v002 -- a checkpoint row records its customer -- on a real PostgreSQL / YugabyteDB.

pins what a fake cannot:

- the migration fills ``customer_id`` on the rows written before it (the batched backfill), and a replay writes nothing;
- the trigger fills it on every later write, a re-key of ``thread_id`` included, and leaves an unscoped thread ``NULL``;
- the customer purge (``delete_keyed`` by ``customer_id``) deletes exactly the rows the old prefix purge deleted
  (``delete_keyed`` by the ``thread_id`` prefix), on both tables;
- the purge reads its customer's rows through ``idx_checkpoints_customer`` / ``idx_checkpoint_writes_customer``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
import pytest_asyncio
from threetears.core.data.migrations import MigrationRunner
from threetears.core.keyed import delete_keyed
from threetears.core.testing.captured import CapturedL3
from threetears.core.testing.plans import explain_write, plan_uses_index
from threetears.langgraph.checkpoint import ThreeTierCheckpointSaver
from threetears.langgraph.checkpoint_scope import CheckpointScope
from threetears.langgraph.migrations import register as register_langgraph
from threetears.langgraph.migrations.v002_checkpoint_customer_column import (
    CUSTOMER_TABLES,
    backfill_customer_ids,
    customer_of,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

_CUSTOMER_A = uuid.UUID("abcdef01-2345-6789-abcd-ef0123456789")
_CUSTOMER_B = uuid.UUID("12345678-9abc-def0-1234-56789abcdef0")

#: thread ids written before v002, each kind the prefix rule must sort: two customers' (one nested), the customer's
#: UUID with no separator, an upper-cased UUID, a UUID not at the start, and unscoped ones
_THREADS = (
    f"{_CUSTOMER_A}/t-1",
    f"{_CUSTOMER_A}/t-2",
    f"{_CUSTOMER_A}/{_CUSTOMER_B}/t-3",
    f"{_CUSTOMER_B}/t-1",
    f"{_CUSTOMER_A}t-4",
    f"{str(_CUSTOMER_A).upper()}/t-5",
    f"x{_CUSTOMER_A}/t-6",
    "plain/t-7",
    "t-8",
)

#: customers the plan test's table holds, and threads a customer
_PLAN_CUSTOMERS = 400
_PLAN_THREADS = 10


class _MigrationStore:
    """the two calls the migration runner makes, over one asyncpg connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self.conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        result: str = await self.conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in await self.conn.fetch(sql, *params)]


async def _insert(conn: asyncpg.Connection, thread_id: str, n: int) -> None:
    """one checkpoint and one pending write of ``thread_id``."""
    await conn.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_id, checkpoint) VALUES ($1, $2, '\\x00'::bytea)",
        thread_id,
        f"c{n}",
    )
    await conn.execute(
        "INSERT INTO checkpoint_writes (thread_id, checkpoint_id, task_id, idx, channel, blob)"
        " VALUES ($1, $2, 'task', $3, 'ch', '\\x00'::bytea)",
        thread_id,
        f"c{n}",
        n,
    )


@pytest_asyncio.fixture(loop_scope="module")
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """a fresh schema migrated to v001, holding :data:`_THREADS`, then migrated to v002."""
    schema = f"lg_customer_{uuid.uuid4().hex[:12]}"
    held = await asyncpg.connect(db_container)
    await held.execute(f'CREATE SCHEMA "{schema}"')
    try:
        await held.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_langgraph(runner)
        await runner.apply_for_agent_schema(_MigrationStore(held), target=1)
        for n, thread_id in enumerate(_THREADS):
            await _insert(held, thread_id, n)
        assert await runner.apply_for_agent_schema(_MigrationStore(held)) == 1
        yield held
    finally:
        await held.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await held.close()


async def _customers(conn: asyncpg.Connection, table: str) -> dict[str, uuid.UUID | None]:
    rows = await conn.fetch(f"SELECT thread_id, customer_id FROM {table}")  # noqa: S608 - the test's own table
    return {row["thread_id"]: row["customer_id"] for row in rows}


async def test_the_migration_fills_the_rows_written_before_it_and_a_replay_writes_nothing(
    conn: asyncpg.Connection,
) -> None:
    """Flips if a row written before v002 is left without the customer its thread id names, or given one it does not
    (an unscoped thread, a UUID with no separator, an upper-cased one, one not at the start), or a replay writes."""
    for table in CUSTOMER_TABLES:
        assert await _customers(conn, table) == {thread_id: customer_of(thread_id) for thread_id in _THREADS}
    assert customer_of(f"{_CUSTOMER_A}/{_CUSTOMER_B}/t-3") == _CUSTOMER_A
    assert await backfill_customer_ids(conn, page=2) == 0
    runner = MigrationRunner()
    register_langgraph(runner)
    assert await runner.apply_for_agent_schema(_MigrationStore(conn)) == 0


async def test_the_trigger_fills_every_later_write_and_a_rekey(conn: asyncpg.Connection) -> None:
    """Flips if a row written after v002 (or re-keyed onto a customer by the documented UPDATE) is not given the
    customer its thread id names, or an unscoped one is given any."""
    transaction = conn.transaction()
    await transaction.start()
    try:
        await _insert(conn, f"{_CUSTOMER_B}/later", 100)
        await _insert(conn, "unscoped-later", 101)
        for table in CUSTOMER_TABLES:
            await conn.execute(
                f"UPDATE {table} SET thread_id = $1 || '/' || thread_id WHERE thread_id = 'unscoped-later'",  # noqa: S608
                str(_CUSTOMER_A),
            )
        for table in CUSTOMER_TABLES:
            found = await _customers(conn, table)
            assert found[f"{_CUSTOMER_B}/later"] == _CUSTOMER_B
            assert found[f"{_CUSTOMER_A}/unscoped-later"] == _CUSTOMER_A
        await _insert(conn, "still-unscoped", 102)
        for table in CUSTOMER_TABLES:
            assert (await _customers(conn, table))["still-unscoped"] is None
    finally:
        await transaction.rollback()


async def _deleted(conn: asyncpg.Connection, purge: Any) -> dict[str, set[str]]:
    """the thread ids ``purge`` deletes from each table, in a transaction rolled back after it."""
    transaction = conn.transaction()
    await transaction.start()
    try:
        before = {table: set(await _customers(conn, table)) for table in CUSTOMER_TABLES}
        await purge()
        return {table: before[table] - set(await _customers(conn, table)) for table in CUSTOMER_TABLES}
    finally:
        await transaction.rollback()


@pytest.mark.parametrize("customer", [_CUSTOMER_A, _CUSTOMER_B], ids=["a", "b"])
async def test_the_purge_by_customer_deletes_what_the_prefix_purge_deleted(
    conn: asyncpg.Connection, customer: uuid.UUID
) -> None:
    """Flips if the saver's purge (by ``customer_id``) deletes other rows than the prefix purge it replaced."""

    async def by_prefix() -> None:
        for table in ("checkpoint_writes", "checkpoints"):
            await delete_keyed(conn, table, prefix={"thread_id": f"{customer}/"})

    async def by_saver() -> None:
        saver = ThreeTierCheckpointSaver(executor=conn, scope=CheckpointScope.for_customer(customer))
        await saver.adelete_customer_threads()

    old, new = await _deleted(conn, by_prefix), await _deleted(conn, by_saver)
    assert old == new, (old, new)
    assert all(rows for rows in new.values()), new


async def test_the_purge_reads_its_customer_s_rows_by_index(db_container: str) -> None:
    """Flips if the customer purge stops reading idx_checkpoints_customer or idx_checkpoint_writes_customer (it would
    read every checkpoint of the schema)."""
    schema = f"lg_customer_plan_{uuid.uuid4().hex[:12]}"
    conn = await asyncpg.connect(db_container, command_timeout=600)
    await conn.execute(f'CREATE SCHEMA "{schema}"')
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_langgraph(runner)
        await runner.apply_for_agent_schema(_MigrationStore(conn))
        customers = [uuid.uuid4() for _ in range(_PLAN_CUSTOMERS)]
        threads = [f"{customers[n % _PLAN_CUSTOMERS]}/t-{n}" for n in range(_PLAN_CUSTOMERS * _PLAN_THREADS)]
        await conn.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_id, checkpoint)"
            " SELECT t, 'c', '\\x00'::bytea FROM unnest($1::text[]) t",
            threads,
        )
        await conn.execute(
            "INSERT INTO checkpoint_writes (thread_id, checkpoint_id, task_id, idx, channel, blob)"
            " SELECT t, 'c', 'task', 0, 'ch', '\\x00'::bytea FROM unnest($1::text[]) t",
            threads,
        )
        await conn.execute("ANALYZE checkpoints")
        await conn.execute("ANALYZE checkpoint_writes")
        captured = CapturedL3(status="DELETE 0")
        saver = ThreeTierCheckpointSaver(executor=captured, scope=CheckpointScope.for_customer(customers[7]))
        await saver.adelete_customer_threads()
        (writes_sql, writes_args), (checkpoints_sql, checkpoints_args) = captured.asked
        assert 'FROM "checkpoint_writes"' in writes_sql and 'FROM "checkpoints"' in checkpoints_sql, captured.asked
        plan = await explain_write(conn, writes_sql, *writes_args)
        assert plan_uses_index(plan, "idx_checkpoint_writes_customer"), plan.plan
        plan = await explain_write(conn, checkpoints_sql, *checkpoints_args)
        assert plan_uses_index(plan, "idx_checkpoints_customer"), plan.plan
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
