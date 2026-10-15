"""The ready signal's fence on a real Postgres: a repeated or stale token is refused, a newer one accepted once.

The fence's one statement is the whole of its correctness (two replicas given the same token at the
same moment must accept it once), so it is proven here against Postgres itself, in the table the
pod's table list declares, rather than against a fake that would only restate the SQL.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

import asyncpg
from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.coordination.source_token import (
    SOURCE_TOKENS_TABLE,
    SourceToken,
    SourceTokenFence,
    source_tokens_schema,
)

pytestmark = pytest.mark.integration

_AT = datetime(2026, 11, 3, 23, 40, tzinfo=UTC)


@pytest.fixture
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    schema = f"tokens_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    created = await asyncpg.create_pool(db_container, min_size=2, max_size=8, server_settings={"search_path": schema})
    assert created is not None
    table = source_tokens_schema().to_sqlalchemy_table(MetaData())
    await created.execute(str(CreateTable(table).compile(dialect=postgresql.dialect())))
    try:
        yield created
    finally:
        await created.close()


def _transaction(pool: asyncpg.Pool) -> Callable[[], Any]:
    @asynccontextmanager
    async def opened() -> AsyncIterator[Any]:
        async with pool.acquire() as conn, CallerTransaction(conn):
            yield conn

    return opened


def _fence(pool: asyncpg.Pool) -> SourceTokenFence:
    return SourceTokenFence(_transaction(pool), table=SOURCE_TOKENS_TABLE, signal="refresh_ready")


async def _signal(fence: SourceTokenFence, run_id: int, as_of: datetime) -> bool:
    async with fence.advancing(SourceToken(run_id=run_id, as_of=as_of)) as advance:
        return advance.accepted


async def test_the_first_token_is_accepted_and_a_repeat_of_it_is_refused(pool: asyncpg.Pool) -> None:
    fence = _fence(pool)
    assert await _signal(fence, 9001, _AT)
    async with fence.advancing(SourceToken(run_id=9001, as_of=_AT)) as repeat:
        assert not repeat.accepted
        assert repeat.last == SourceToken(run_id=9001, as_of=_AT), "the refusal names the token that stands"
    assert await fence.last() == SourceToken(run_id=9001, as_of=_AT)


async def test_a_stale_token_is_refused_and_a_newer_run_is_accepted(pool: asyncpg.Pool) -> None:
    fence = _fence(pool)
    assert await _signal(fence, 9001, _AT)
    assert not await _signal(fence, 9000, _AT - timedelta(minutes=5)), "an earlier run's late signal"
    assert not await _signal(fence, 9000, _AT + timedelta(minutes=5)), "an earlier run, whatever its data"
    assert not await _signal(fence, 9002, _AT - timedelta(seconds=1)), "a later run with older data"
    assert await _signal(fence, 9002, _AT), "a later run, same newest timestamp: a call or a correction"
    assert await _signal(fence, 9003, _AT + timedelta(minutes=1))
    assert await fence.last() == SourceToken(run_id=9003, as_of=_AT + timedelta(minutes=1))


async def test_a_token_whose_work_fails_to_start_is_not_spent(pool: asyncpg.Pool) -> None:
    """Airflow retries a call that failed; the retry must start the refresh, not be refused as a repeat."""
    fence = _fence(pool)
    with pytest.raises(RuntimeError):
        async with fence.advancing(SourceToken(run_id=9001, as_of=_AT)) as advance:
            assert advance.accepted
            raise RuntimeError("the refresh request could not be recorded")
    assert await fence.last() is None
    assert await _signal(fence, 9001, _AT)


async def test_the_same_token_at_the_same_moment_is_accepted_once(pool: asyncpg.Pool) -> None:
    """two replicas, each answering one of two identical calls: one starts a refresh, the other refuses."""
    fences = [_fence(pool) for _ in range(6)]
    for run_id in (9001, 9002):
        accepted = await asyncio.gather(*(_signal(fence, run_id, _AT) for fence in fences))
        assert sum(accepted) == 1, f"run {run_id}: accepted {sum(accepted)} times"
