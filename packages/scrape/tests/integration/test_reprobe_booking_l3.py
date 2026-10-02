"""Real-Postgres proof that a booked re-probe survives the ``scheduled_jobs`` constraints.

The bind path is where a hand-built row gets judged: the scheduled-jobs upsert writes every
column positionally, so a server-side ``DEFAULT`` never applies and a key this adapter forgets
to set is bound as an explicit NULL. ``save_entity`` stamps ``date_created`` for a new entity
but stamps ``date_updated`` only when the key is already present or the entity is not new, so
the adapter must supply it.

The failure is silent rather than loud. ``TargetCircuit._book_reprobe`` catches and logs, so a
constraint violation does not fail a fetch -- an event-driven deployment just books no
re-probes at all, which is the entire purpose of the ``[reprobe]`` extra, while the health
row's ``blocked_until`` goes on looking correct.

A database is the only honest judge of that. The unit suite once checked it by projecting the
row through the other package's private insert-column list and parsing ``NOT NULL`` out of its
migration's DDL text; both are that package's implementation details, and the second is a
parse of the schema rather than the schema. Here the real migration provisions the table, the
real collection writes the booking, and the database enforces every constraint the migration
declares -- including any ``NOT NULL`` column added after this test was written.

Guarded by ``@pytest.mark.integration``, like ``test_health_l3_roundtrip.py``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.scheduled_jobs.collections import ScheduledJobCollection
from threetears.scheduled_jobs.migrations import register as register_scheduled_jobs

from threetears.scrape.reprobe import ScheduledJobsReprobeScheduler, reprobe_job_id

pytestmark = pytest.mark.integration


# parity-with: threetears.core.data.store.DataStore
class AsyncpgMigrationStore:
    """The ``execute``/``query`` surface the migration runner drives, over one connection.

    :param conn: the connection the migrations run on
    :ptype conn: asyncpg.Connection
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """Execute *sql* on the wrapped connection."""
        result: str = await self._conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """Fetch rows as a list of dicts."""
        return [dict(row) for row in await self._conn.fetch(sql, *params)]


@pytest.fixture
async def jobs_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """A pool on a fresh schema holding the real ``scheduled_jobs`` migration."""
    schema = f"scrape_reprobe_{uuid.uuid4().hex[:12]}"
    setup = await asyncpg.connect(db_container)
    try:
        await setup.execute(f'CREATE SCHEMA "{schema}"')
        await setup.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_scheduled_jobs(runner)
        await runner.apply_for_platform_schema(AsyncpgMigrationStore(setup))  # type: ignore[arg-type]
    finally:
        await setup.close()
    pool = await asyncpg.create_pool(
        db_container,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    assert pool is not None
    try:
        yield pool
    finally:
        await pool.close()
        teardown = await asyncpg.connect(db_container)
        try:
            await teardown.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await teardown.close()


async def test_the_booked_row_binds_a_value_for_every_not_null_column(jobs_pool: asyncpg.Pool) -> None:
    """A booking written through the real collection lands, with no ``NOT NULL`` column empty."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=jobs_pool)
    jobs = ScheduledJobCollection(
        registry=registry,
        config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
        nats_client=None,
    )

    await ScheduledJobsReprobeScheduler(jobs).schedule_reprobe(target_id="warn_oh", delay_seconds=900.0)

    # The database enforces NOT NULL itself, so a row missing a value never lands: the write
    # raises, or (behind a swallowing caller) nothing is there to read back.
    row = await jobs_pool.fetchrow("SELECT * FROM scheduled_jobs WHERE job_id = $1", reprobe_job_id("warn_oh"))
    assert row is not None, "the booking never reached the table"
    assert row["date_updated"] is not None
