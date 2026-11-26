"""agent-acl v011's backfill is heard: each row it fills carries a fresh epoch, so the migration run's window names it.

A migration writes with no collection and no advance; the run hands its window to its announcer
(``MigrationRunner(announcer=...)``), which advances each table from the rows stamped inside it
(``lost_writes.announce_window``), and a restore reads a stamped table's rows by epoch the same way
(``epoch_reads.range_rows``). A row the backfill wrote but left at its older stamp reaches neither: no follower is told,
and a restore from a mark before the run misses it. Against a real database: the stamped table's ``(bucket, epoch)``
ranges are the reads both use.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
import pytest_asyncio

from threetears.agent.acl.migrations.v011_owner_namespace_by_id import backfill_owner_namespace_ids
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.epoch_reads import epoch_range, range_rows, relations_named
from threetears.core.collections.lost_writes import announce_window

from .access_schema import access_pool, access_schema

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]


class _Source:
    """a recording source: the rows each advance named, by table."""

    def __init__(self) -> None:
        self.rows: dict[str, list[tuple[str, ...]]] = {}

    async def current(self, table_name: str) -> str:
        return "inc:0"

    async def advance(self, table_name: str) -> str:
        raise AssertionError("the window's advance names its rows")

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
        **named: Any,
    ) -> str:
        assert rows is not None and not keys_unknown, "a window of a few rows names them"
        self.rows.setdefault(table_name, []).extend(tuple(row.ids) for row in rows)
        return f"inc:{len(self.rows)}"


@pytest_asyncio.fixture(loop_scope="module")
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """a pool on a schema of the test's own, its tables built by the acl migrations."""
    async with access_schema(db_container, "ownerheard") as schema:
        made = await access_pool(db_container, schema, max_size=2)
        try:
            yield made
        finally:
            await made.close()


async def _insert(conn: Any, *, name: str, owner: str | None, epoch: int) -> uuid.UUID:
    """one platform namespace row naming its owner by name and holding no owner id, as before v011."""
    namespace_id = uuid.uuid4()
    now = datetime.now(UTC)
    await conn.execute(
        "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, owner_namespace,"
        " date_created, date_updated, epoch) VALUES ('platform', $1, $2, 'tool', $3, $4, $4, $5)",
        namespace_id,
        name,
        owner,
        now,
        epoch,
    )
    return namespace_id


async def test_the_backfill_s_rows_are_named_by_the_run_s_window_and_a_restore_by_epoch(pool: asyncpg.Pool) -> None:
    """Flips if a row the backfill fills keeps its older epoch: the migration run's window announcement does not name
    it (no follower hears the backfill), and a restore reading the window's epochs does not find it."""
    # seeded an hour before the run: an epoch the run's window cannot hold
    old = epoch_range((datetime.now(UTC) - timedelta(hours=1), datetime.now(UTC) - timedelta(hours=1)))[0]
    async with pool.acquire() as conn:
        owner = await _insert(conn, name="tools.owner", owner=None, epoch=old)
        children = {await _insert(conn, name=f"tools.child{n}", owner="tools.owner", epoch=old) for n in range(3)}
        bystander = await _insert(conn, name="tools.alone", owner=None, epoch=old)

        since = datetime.now(UTC)
        written = await backfill_owner_namespace_ids(conn)
        until = datetime.now(UTC)

        filled = {
            row["namespace_id"]
            for row in await conn.fetch("SELECT namespace_id FROM namespaces WHERE owner_namespace_id = $1", owner)
        }
        schema = str(await conn.fetchval("SELECT current_schema()"))
    assert written == 3 and filled == children

    expected = {str(child) for child in children}
    # the test's own schema's: the database holds other tests' schemas too
    [relation] = [each for each in await relations_named(pool, "namespaces") if each.schema == schema]
    floor, ceiling = epoch_range((since, until))
    restored_ids = {
        value
        async for page in range_rows(pool, relation, floor, ceiling, page=100)
        for row in page
        for value in row.keys
    }
    assert expected <= restored_ids, "a restore by epoch misses the backfill's rows"
    assert str(owner) not in restored_ids and str(bystander) not in restored_ids, "a row the backfill left was read"

    source = _Source()
    done = await announce_window(
        pool, generation_source=source, notes_schema=schema, since=since, until=until, why="migration", schema=schema
    )
    assert "namespaces" in done.advanced and not done.failed, done
    announced = {value for ids in source.rows["namespaces"] for value in ids}
    assert expected <= announced, "the run's window does not name the backfill's rows"
    assert str(owner) not in announced and str(bystander) not in announced
