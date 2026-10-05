"""Integration test: a durable epoch records the value each move replaced, against real Postgres.

The property is in the SQL -- the update reads the row as it stood -- so only a real database can
witness it. A move may skip (a tile version moved from 3 to 7 after a failed load of 6), and the
previous epoch must then be 3, never ``epoch - 1``.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import asyncpg
import pytest

from threetears.epoch import DurableEpoch, EpochClient
from threetears.epoch.migrations import add_previous_epoch_column, create_config_epochs_table
from threetears.nats.subjects import Subjects

pytestmark = pytest.mark.integration


# parity-with: threetears.core.data.store.DataStore
class _PoolStore:
    """the ``execute`` slice of a DataStore over an asyncpg pool, for running the migrations."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def execute(self, sql: str, *params: Any) -> str:
        """run one statement.

        :param sql: the statement
        :ptype sql: str
        :param params: its parameters
        :ptype params: Any
        :return: the command status
        :rtype: str
        """
        result: str = await self._pool.execute(sql, *params)
        return result


@pytest.fixture
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """a pool bound to a fresh schema, dropped afterwards.

    :param db_container: the shared Postgres URL
    :ptype db_container: str
    :return: the pool
    :rtype: AsyncIterator[asyncpg.Pool]
    """
    schema = f"epoch_prev_{uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    created = await asyncpg.create_pool(db_container, server_settings={"search_path": schema}, min_size=1, max_size=2)
    assert created is not None
    try:
        yield created
    finally:
        await created.close()
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()


def _client(pool: asyncpg.Pool) -> EpochClient:
    nats = MagicMock()
    nats.publish = AsyncMock()
    return EpochClient(pool, nats)


async def test_a_move_that_skips_records_the_epoch_it_replaced(pool: asyncpg.Pool) -> None:
    store = _PoolStore(pool)
    await create_config_epochs_table(store)  # type: ignore[arg-type]
    await add_previous_epoch_column(store)  # type: ignore[arg-type]
    client = _client(pool)
    subject = Subjects.datasource_tile_epoch("geo", "us_county")

    assert await client.advance_to(subject, 3) == DurableEpoch(epoch=3, previous=None)
    # a load of 6 failed and was never reported; the next report is 7
    assert await client.advance_to(subject, 7) == DurableEpoch(epoch=7, previous=3)
    assert await client.versions(subject) == DurableEpoch(epoch=7, previous=3)


async def test_a_move_that_does_not_happen_keeps_the_previous_epoch(pool: asyncpg.Pool) -> None:
    store = _PoolStore(pool)
    await create_config_epochs_table(store)  # type: ignore[arg-type]
    await add_previous_epoch_column(store)  # type: ignore[arg-type]
    client = _client(pool)
    subject = Subjects.datasource_tile_epoch("geo", "us_state")
    await client.advance_to(subject, 3)
    await client.advance_to(subject, 7)

    # a retried report of 7, and a stale one of 5: neither moves, and 3 stays the previous epoch
    assert await client.advance_to(subject, 7) == DurableEpoch(epoch=7, previous=3)
    assert await client.advance_to(subject, 5) == DurableEpoch(epoch=7, previous=3)


async def test_the_column_is_added_to_a_table_that_already_holds_rows(pool: asyncpg.Pool) -> None:
    store = _PoolStore(pool)
    await create_config_epochs_table(store)  # type: ignore[arg-type]
    await pool.execute("INSERT INTO config_epochs (subject_path, epoch) VALUES ('old', 4)")
    await add_previous_epoch_column(store)  # type: ignore[arg-type]
    await add_previous_epoch_column(store)  # type: ignore[arg-type]  # idempotent on replay
    row = await pool.fetchrow("SELECT epoch, previous_epoch FROM config_epochs WHERE subject_path = 'old'")
    assert row is not None
    assert (row["epoch"], row["previous_epoch"]) == (4, None)


async def test_a_subject_that_never_moved_has_no_epochs(pool: asyncpg.Pool) -> None:
    store = _PoolStore(pool)
    await create_config_epochs_table(store)  # type: ignore[arg-type]
    await add_previous_epoch_column(store)  # type: ignore[arg-type]
    assert await _client(pool).versions(Subjects.datasource_tile_epoch("geo", "none")) == DurableEpoch(
        epoch=0, previous=None
    )
