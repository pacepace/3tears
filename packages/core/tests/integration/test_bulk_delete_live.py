"""Integration: rows deleted by key in multi-row statements on the caller's transaction.

What the unit tests' recording connection cannot prove: that Postgres accepts the row-constructor
``IN`` list with typed parameters, deletes exactly the keys named, and keeps them when the
transaction rolls back.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity

pytestmark = pytest.mark.integration

_START = datetime(2026, 11, 4, 1, 0, tzinfo=UTC)

_SCHEMA = TableSchema(
    name="snapshots",
    primary_key=("race", "reported_at"),
    columns=[
        Column("race", STRING_TYPE),
        Column("reported_at", DATETIMETZ_TYPE),
        Column("votes", BIGINT_TYPE, nullable=True),
    ],
    on_conflict="update",
)


class _Snapshot(BaseEntity):
    primary_key_field = "reported_at"


@pytest.fixture
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    schema = f"bulkdel_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    held = await asyncpg.create_pool(db_container, min_size=1, max_size=2, server_settings={"search_path": schema})
    assert held is not None
    try:
        await held.execute(
            "CREATE TABLE snapshots (race TEXT, reported_at TIMESTAMPTZ, votes BIGINT, PRIMARY KEY (race, reported_at))"
        )
        for minute in range(6):
            await held.execute(
                "INSERT INTO snapshots VALUES ($1, $2, $3)",
                f"r{minute % 2}",
                _START + timedelta(minutes=minute),
                minute,
            )
        yield held
    finally:
        await held.close()


def _collection(pool: asyncpg.Pool) -> Any:
    registry = CollectionRegistry()
    registry.configure(l3_pool=SqlL3Backend(pool))
    return collection_for_schema(_SCHEMA, entity_class=_Snapshot)(registry, DefaultCoreConfig(), None)


async def _held(pool: asyncpg.Pool) -> list[int]:
    return [r["votes"] for r in await pool.fetch("SELECT votes FROM snapshots ORDER BY votes")]


async def test_exactly_the_keys_named_are_deleted_across_statements(pool: asyncpg.Pool) -> None:
    collection = _collection(pool)
    keys = [
        ("r1", _START + timedelta(minutes=1)),
        ("r0", _START + timedelta(minutes=4)),
        ("r1", _START + timedelta(minutes=5)),
        ("nope", _START),
    ]

    async with pool.acquire() as conn, CallerTransaction(conn):
        assert await collection.delete_rows(keys, conn=conn, max_rows=2) == 4

    assert await _held(pool) == [0, 2, 3]


async def test_a_rolled_back_delete_keeps_every_row(pool: asyncpg.Pool) -> None:
    collection = _collection(pool)

    with pytest.raises(ConnectionError):
        async with pool.acquire() as conn, CallerTransaction(conn):
            await collection.delete_rows([("r0", _START)], conn=conn)
            raise ConnectionError("the write after it failed")

    assert await _held(pool) == [0, 1, 2, 3, 4, 5]
