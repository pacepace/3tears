"""Integration: the ready signal's fence is a stamped table (``ROW_STAMPED``) on a real database.

An advance rises the row's epoch and a clear marks rows gone in place, never deleting one, so a restore of a lost
change log hears both; the fence reads its live row alone, so a cleared signal accepts its next token as a first.
An install whose fence table predates the stamp is stamped where it stands by 3tears' shared steps
(``stamp_existing_tables`` over ``stamping_table_of(source_tokens_schema())``, the steps the hub takes for a table list
that declares the table stamped), its rows kept and stamped.

On the session's ``db_container``: PostgreSQL, or YugabyteDB in the pool's ``integration_yugabyte`` suite; a schema of
its own a test.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.coordination.source_token import (
    SOURCE_TOKENS_TABLE,
    SourceToken,
    SourceTokenFence,
    clear_signals,
    source_tokens_schema,
)
from threetears.core.data.migrations.stamp_tables import stamp_existing_tables, stamping_table_of
from threetears.core.ddl.row_epoch import bucket_epoch_index_name

pytestmark = pytest.mark.integration

_AT = datetime(2026, 11, 3, 23, 40, tzinfo=UTC)

#: the fence table as 3tears declared it before it was stamped (no epoch, no gone flag, no restore bucket)
_UNSTAMPED_DDL = (
    "CREATE TABLE source_tokens (signal TEXT NOT NULL PRIMARY KEY, run_id BIGINT NOT NULL, "
    "as_of TIMESTAMPTZ NOT NULL, accepted_at TIMESTAMPTZ NOT NULL, date_created TIMESTAMPTZ NOT NULL, "
    "date_updated TIMESTAMPTZ NOT NULL)",
    'CREATE INDEX source_tokens_by_signal_c ON source_tokens (signal COLLATE "C" ASC)',
)


async def _schema_pool(url: str) -> tuple[asyncpg.Pool, str, bool]:
    """a pool on a fresh schema of its own, and whether the server is YugabyteDB."""
    schema = f"stamped_tokens_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(url)
    try:
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        yugabyte = "-YB-" in str(await admin.fetchval("SELECT version()"))
    finally:
        await admin.close()
    created = await asyncpg.create_pool(url, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert created is not None
    return created, schema, yugabyte


async def _drop(url: str, schema: str) -> None:
    admin = await asyncpg.connect(url)
    try:
        await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await admin.close()


@pytest.fixture
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """the fence table created from its declaration, as a table list declaring it creates it."""
    created, schema, yugabyte = await _schema_pool(db_container)
    try:
        for statement in source_tokens_schema().create_statements(yugabyte=yugabyte):
            await created.execute(statement)
        yield created
    finally:
        await created.close()
        await _drop(db_container, schema)


def _transaction(pool: asyncpg.Pool) -> Callable[[], Any]:
    """the transaction an SDK pod's ``provider_transaction`` opens: a :class:`CallerTransaction` on one connection."""

    @asynccontextmanager
    async def opened() -> AsyncIterator[Any]:
        async with pool.acquire() as conn, CallerTransaction(conn):
            yield conn

    return opened


async def _signal(fence: SourceTokenFence, run_id: int) -> bool:
    async with fence.advancing(SourceToken(run_id=run_id, as_of=_AT)) as advance:
        return advance.accepted


async def _stored(pool: asyncpg.Pool, signal: str) -> asyncpg.Record:
    row = await pool.fetchrow("SELECT run_id, epoch, gone FROM source_tokens WHERE signal = $1", signal)
    assert row is not None, f"{signal}'s row is kept"
    return row


async def test_an_advance_stamps_the_row_and_its_epoch_rises(pool: asyncpg.Pool) -> None:
    """Flips if an advance leaves no epoch, or a later advance leaves it where it was (a restore since X would miss
    it), or a refused token moves it."""
    fence = SourceTokenFence(_transaction(pool), table=SOURCE_TOKENS_TABLE, signal="refresh_ready")
    assert await _signal(fence, 9001)
    first = await _stored(pool, "refresh_ready")
    assert first["epoch"] > 0 and first["gone"] is False
    assert not await _signal(fence, 9001), "a repeat"
    assert (await _stored(pool, "refresh_ready"))["epoch"] == first["epoch"], "a refused token changes nothing"
    assert await _signal(fence, 9002)
    assert (await _stored(pool, "refresh_ready"))["epoch"] > first["epoch"]


async def test_a_clear_marks_gone_and_the_next_token_is_accepted_as_a_first(pool: asyncpg.Pool) -> None:
    """Flips if a clear deletes the fence's rows, leaves their epoch where it was, touches a row its prefix does not
    name, or if the fence then reads the cleared token as the last accepted (refusing an earlier run's token), or an
    advance over the cleared row leaves it gone."""
    transaction = _transaction(pool)
    for signal in ("refresh_ready", "refresh_ready/a", "refresh_ready/b", "refresh_ready_other"):
        assert await _signal(SourceTokenFence(transaction, table=SOURCE_TOKENS_TABLE, signal=signal), 9005)
    before = {signal: (await _stored(pool, signal))["epoch"] for signal in ("refresh_ready", "refresh_ready/a")}
    async with transaction() as conn:
        cleared = await clear_signals(conn, SOURCE_TOKENS_TABLE, signal="refresh_ready", prefix="refresh_ready/")
    assert cleared == ["refresh_ready", "refresh_ready/a", "refresh_ready/b"]
    for signal in cleared:
        row = await _stored(pool, signal)
        assert row["gone"] is True, f"{signal} is marked gone, kept"
    for signal, epoch in before.items():
        assert (await _stored(pool, signal))["epoch"] > epoch, f"{signal}'s gone mark rises its epoch"
    assert (await _stored(pool, "refresh_ready_other"))["gone"] is False, "a name that only begins like the prefix"
    async with transaction() as conn:
        assert await clear_signals(conn, SOURCE_TOKENS_TABLE, prefix="refresh_ready/") == [], "a gone row is not again"

    fence = SourceTokenFence(transaction, table=SOURCE_TOKENS_TABLE, signal="refresh_ready")
    assert await fence.last() is None, "a cleared row reads as no token"
    gone_epoch = (await _stored(pool, "refresh_ready"))["epoch"]
    assert await _signal(fence, 9001), "an earlier run after a reset is a first token"
    revived = await _stored(pool, "refresh_ready")
    assert revived["gone"] is False and revived["run_id"] == 9001 and revived["epoch"] > gone_epoch
    assert await fence.last() == SourceToken(run_id=9001, as_of=_AT)


async def test_the_shared_stamp_steps_stamp_an_existing_fence(db_container: str) -> None:
    """Flips if the shared stamping steps, over the fence's declaration, do not turn an unstamped fence table into
    the one the declaration renders (its rows kept, each stamped and live, its restore index there), or the fence
    cannot advance and clear on it after."""
    created, schema, _ = await _schema_pool(db_container)
    try:
        for statement in _UNSTAMPED_DDL:
            await created.execute(statement)
        await created.executemany(
            "INSERT INTO source_tokens (signal, run_id, as_of, accepted_at, date_created, date_updated) "
            "VALUES ($1, $2, $3, now(), now(), now())",
            [("refresh_ready", 9001, _AT), ("refresh_ready/a", 7, _AT)],
        )

        class _Store:
            """the migration runner's store as the shared steps use it: ``query`` answers rows, ``execute`` runs."""

            async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
                return [dict(row) for row in await created.fetch(sql, *params)]

            async def execute(self, sql: str, *params: Any) -> str:
                return str(await created.execute(sql, *params))

        assert await stamp_existing_tables(_Store(), (stamping_table_of(source_tokens_schema()),)) == 2
        rows = await created.fetch("SELECT signal, epoch, gone FROM source_tokens ORDER BY signal")
        assert [row["signal"] for row in rows] == ["refresh_ready", "refresh_ready/a"], "every row is kept"
        assert all(row["epoch"] > 0 and row["gone"] is False for row in rows), rows
        indexes = {row["indexname"] for row in await created.fetch("SELECT indexname FROM pg_indexes")}
        assert bucket_epoch_index_name(SOURCE_TOKENS_TABLE) in indexes, indexes

        transaction = _transaction(created)
        fence = SourceTokenFence(transaction, table=SOURCE_TOKENS_TABLE, signal="refresh_ready")
        assert await fence.last() == SourceToken(run_id=9001, as_of=_AT), "the stamped row reads as it stood"
        assert not await _signal(fence, 9001) and await _signal(fence, 9002)
        async with transaction() as conn:
            assert await clear_signals(conn, SOURCE_TOKENS_TABLE, prefix="refresh_ready/") == ["refresh_ready/a"]
    finally:
        await created.close()
        await _drop(db_container, schema)
