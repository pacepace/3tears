"""a pooled connection sends ``SET search_path`` / ``SET statement_timeout`` only when the value changes.

The hub's reset query set ``search_path`` on every release, and its proxy set both settings before every
statement: on cobalt dev, 2026-10-29, most of the YSQL statement mix. These drive a real pool and count the
statements the server is sent, through a query logger on each connection, and read the settings back with
``SHOW``, so a skipped SET is shown both to be skipped and to have changed nothing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import asyncpg
import pytest

from threetears.core.utils import session_settings_connection_class

pytestmark = pytest.mark.integration


class _Sent:
    """every statement the pool's connections sent the server."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def init(self, conn: asyncpg.Connection) -> None:
        conn.add_query_logger(lambda record: self.queries.append(record.query))

    def count(self, prefix: str) -> int:
        return sum(1 for query in self.queries if query.upper().startswith(prefix.upper()))


@pytest.fixture
async def pool_and_sent(db_container: str) -> AsyncIterator[tuple[asyncpg.Pool, _Sent]]:
    """one connection, so every acquire is the same session."""
    sent = _Sent()
    pool = await asyncpg.create_pool(
        db_container,
        min_size=1,
        max_size=1,
        init=sent.init,
        server_settings={"search_path": "public"},
        connection_class=session_settings_connection_class(search_path="public"),
    )
    try:
        yield pool, sent
    finally:
        await pool.close()


async def test_a_release_that_left_the_search_path_alone_sends_no_reset(
    pool_and_sent: tuple[asyncpg.Pool, _Sent],
) -> None:
    pool, sent = pool_and_sent
    for _ in range(10):
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    # a new connection knows neither setting: its first release resets once, and none after
    assert sent.count("SET search_path") <= 1


async def test_a_search_path_a_borrower_moved_is_back_at_the_next_acquire(
    pool_and_sent: tuple[asyncpg.Pool, _Sent],
) -> None:
    pool, _ = pool_and_sent
    async with pool.acquire() as conn:
        await conn.execute("SET search_path TO pg_catalog")
        assert await conn.fetchval("SHOW search_path") == "pg_catalog"
    async with pool.acquire() as conn:
        assert await conn.fetchval("SHOW search_path") == "public"


async def test_setting_what_the_session_holds_is_not_sent_again(pool_and_sent: tuple[asyncpg.Pool, _Sent]) -> None:
    pool, sent = pool_and_sent
    for _ in range(5):
        async with pool.acquire() as conn:
            await conn.execute("SET statement_timeout TO 5000")
            await conn.execute("SET search_path TO pg_catalog")
    assert sent.count("SET statement_timeout") == 1
    async with pool.acquire() as conn:
        assert await conn.fetchval("SHOW statement_timeout") == "5s"


async def test_a_set_a_rollback_undid_is_sent_again(pool_and_sent: tuple[asyncpg.Pool, _Sent]) -> None:
    """a SET inside a transaction is undone by its rollback, so the value is not known after it."""
    pool, _ = pool_and_sent
    async with pool.acquire() as conn:
        await conn.execute("SET statement_timeout TO 5000")
        with pytest.raises(RuntimeError):
            async with conn.transaction():
                await conn.execute("SET statement_timeout TO 7000")
                raise RuntimeError("rolled back")
        assert await conn.fetchval("SHOW statement_timeout") == "5s"
        await conn.execute("SET statement_timeout TO 7000")
        assert await conn.fetchval("SHOW statement_timeout") == "7s"


async def test_a_setting_changed_another_way_is_not_held(pool_and_sent: tuple[asyncpg.Pool, _Sent]) -> None:
    pool, _ = pool_and_sent
    async with pool.acquire() as conn:
        await conn.execute("SET search_path TO pg_catalog")
        await conn.fetchval("SELECT set_config('search_path', 'public', false)")
        await conn.execute("SET search_path TO pg_catalog")
        assert await conn.fetchval("SHOW search_path") == "pg_catalog"
    async with pool.acquire() as conn:
        assert await conn.fetchval("SHOW search_path") == "public"
