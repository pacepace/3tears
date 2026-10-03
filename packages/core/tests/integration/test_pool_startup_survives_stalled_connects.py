"""a pool starts against a real Postgres although some of its connects stall, and leaks nothing when one cannot.

the defect: :func:`create_pool_with_startup_timeout` bounded the whole of ``asyncpg.create_pool`` --
``min_size`` connects -- by one startup budget, and asyncpg's own per-connect timeout (60s) was longer
than that budget. one connect whose backend never answered (a backend stalled under host memory
pressure) consumed the entire budget, nothing retried, and the pool failed to start, although a plain
``asyncpg.connect`` to the same database answered; at pod startup the same stall crash-loops a pod.

Postgres cannot be made to stall one chosen connect, so these tests put a TCP proxy in front of the
session's Postgres container. the proxy numbers the connections it accepts and, per number, either
forwards it, holds it open and never answers (a wedged backend), drops it at once, or forwards it
only after a delay. every pool connection names a per-test ``application_name``, so the server's own
``pg_stat_activity`` says exactly how many backends the pool holds -- the leak check is taken from
the server, not from the client's bookkeeping.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
import uuid
import weakref
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from threetears.core.testing.tcp_proxy import PROXY_DROP, PROXY_STALL, StallingTcpProxy
from threetears.core.utils.pg_pool_kwargs import PoolStartupTimeoutError, create_pool_with_startup_timeout

pytestmark = pytest.mark.integration

_LOGGER = "threetears.core.utils.pg_pool_kwargs"


@pytest.fixture
async def proxy(db_container: str) -> AsyncIterator[StallingTcpProxy]:
    """a stalling proxy in front of the session Postgres; each test sets its plan before connecting."""
    upstream = urlsplit(db_container)
    assert upstream.hostname is not None and upstream.port is not None
    running = StallingTcpProxy(upstream_host=upstream.hostname, upstream_port=upstream.port)
    await running.start()
    try:
        yield running
    finally:
        await running.stop()


@pytest.fixture
def application_name() -> str:
    """a name only this test's pool connections carry, so the server can count them."""
    return f"poolretry-{uuid.uuid4().hex[:12]}"


async def _backends_settle_at(db_container: str, application_name: str, expected: int, *, within: float) -> int:
    """the server's count of backends carrying ``application_name``, once it equals ``expected`` or time runs out.

    a terminated client's backend leaves ``pg_stat_activity`` a moment after the socket closes, so the
    count is polled rather than read once.
    """
    admin = await asyncpg.connect(db_container)
    try:
        deadline = time.monotonic() + within
        while True:
            count: int = await admin.fetchval(
                "SELECT count(*) FROM pg_stat_activity WHERE application_name = $1", application_name
            )
            if count == expected or time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.05)
    finally:
        await admin.close()
    return count


async def _start(
    proxy: StallingTcpProxy,
    dsn: str,
    application_name: str,
    *,
    startup_timeout: float = 6.0,
    size: int = 4,
) -> asyncpg.Pool:
    """start a pool of ``size`` connections through the proxy, each connect bounded at half a second.

    ``ssl=False`` keeps one connect to exactly one TCP connection, so the proxy's numbering is the
    pool's connect order.
    """
    return await create_pool_with_startup_timeout(
        proxy.dsn_through(dsn),
        pool_name="stall_test",
        startup_timeout=startup_timeout,
        connect_timeout=0.5,
        min_size=size,
        max_size=size,
        ssl=False,
        server_settings={"application_name": application_name},
    )


class TestThePoolStartsDespiteStalledConnects:
    """one wedged connect costs one per-connect timeout, then the pool starts with exactly its own backends."""

    async def test_a_stalled_first_connect_is_retried(
        self, proxy: StallingTcpProxy, db_container: str, application_name: str
    ) -> None:
        proxy.plan = {1: PROXY_STALL}
        started = time.monotonic()
        pool = await _start(proxy, db_container, application_name)
        try:
            assert time.monotonic() - started < 6.0
            assert pool.get_size() == 4
            assert await pool.fetchval("SELECT 1") == 1
            assert await _backends_settle_at(db_container, application_name, 4, within=5.0) == 4
        finally:
            await pool.close()
        assert await proxy.all_client_sockets_closed(within=5.0)

    async def test_a_stall_among_the_parallel_connects_closes_the_half_built_pool(
        self, proxy: StallingTcpProxy, db_container: str, application_name: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """attempt 1 opens three connections before its third connect stalls; none of them survive it."""
        proxy.plan = {3: PROXY_STALL}
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            pool = await _start(proxy, db_container, application_name)
        try:
            assert proxy.accepted >= 5, "the first attempt never reached its parallel connects"
            assert pool.get_size() == 4
            assert await _backends_settle_at(db_container, application_name, 4, within=5.0) == 4
        finally:
            await pool.close()
        assert await _backends_settle_at(db_container, application_name, 0, within=5.0) == 0
        # the retry WARNING counts the connections the failed attempt had open, the ones the
        # half-built pool already held among them.
        closed = [r.extra_data["connections_closed"] for r in caplog.records if "attempt 1 failed" in r.getMessage()]
        assert closed == [3]

    async def test_a_started_pool_keeps_no_connection_it_has_since_closed(
        self, db_container: str, application_name: str
    ) -> None:
        """the pool grows past ``min_size`` and then replaces every connection; the closed ones are released.

        asyncpg calls the wrapper's connect hook for each of those connections. a hook still
        recording after the pool started would hold every one of them for the life of the process.
        """
        opened: weakref.WeakSet[asyncpg.Connection] = weakref.WeakSet()

        class _Tracked(asyncpg.Connection):  # type: ignore[misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                opened.add(self)

        pool = await create_pool_with_startup_timeout(
            db_container,
            pool_name="started",
            startup_timeout=6.0,
            min_size=1,
            max_size=3,
            connection_class=_Tracked,
            server_settings={"application_name": application_name},
        )
        try:
            for _ in range(2):
                async with pool.acquire() as first, pool.acquire() as second, pool.acquire() as third:
                    assert await third.fetchval("SELECT 1") == 1
                del first, second, third
                await pool.expire_connections()
            await asyncio.sleep(0.1)
            gc.collect()
            still_held = [connection for connection in opened if connection.is_closed()]
            assert still_held == [], f"{len(still_held)} closed connections are still referenced"
        finally:
            await pool.close()

    async def test_a_connect_finishing_after_its_attempt_failed_is_closed(
        self, proxy: StallingTcpProxy, db_container: str, application_name: str
    ) -> None:
        """connection 2 is slow and connection 3 is dropped: the attempt fails while 2 is still in flight.

        2 is held for 0.3s -- inside its 0.5s connect bound, so it completes -- and 3 fails at once,
        so the attempt is abandoned before 2 arrives and 2 lands during the pause before the next
        attempt. it must be closed, not left in a pool nobody holds.
        """
        proxy.plan = {2: 0.3, 3: PROXY_DROP}
        pool = await _start(proxy, db_container, application_name)
        try:
            assert pool.get_size() == 4
            assert await _backends_settle_at(db_container, application_name, 4, within=5.0) == 4
        finally:
            await pool.close()
        assert await _backends_settle_at(db_container, application_name, 0, within=5.0) == 0


class TestAnErrorFromTheCallersInitKeepsItsOwnText:
    """only asyncpg's connection-parameter handling has its text withheld; a caller's ``init`` is not that."""

    async def test_a_value_error_from_init_raises_as_itself(self, db_container: str, application_name: str) -> None:
        """``init`` runs on a connection that is already open; its ``ValueError`` says nothing about the DSN."""

        async def init(connection: asyncpg.Connection) -> None:
            raise ValueError("init rejected the session")

        with pytest.raises(ValueError, match="init rejected the session"):
            await create_pool_with_startup_timeout(
                db_container,
                startup_timeout=6.0,
                min_size=1,
                max_size=1,
                init=init,
                server_settings={"application_name": application_name},
            )
        assert await _backends_settle_at(db_container, application_name, 0, within=5.0) == 0

    async def test_a_server_error_from_init_keeps_its_text_and_cause(
        self, db_container: str, application_name: str
    ) -> None:
        """a statement ``init`` runs that the server rejects is a server answer: its text and its cause stand."""

        async def init(connection: asyncpg.Connection) -> None:
            await connection.execute("SELECT 1 / 0")

        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                db_container,
                startup_timeout=6.0,
                min_size=1,
                max_size=1,
                init=init,
                server_settings={"application_name": application_name},
            )
        assert "division by zero" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, asyncpg.exceptions.DataError)
        assert await _backends_settle_at(db_container, application_name, 0, within=5.0) == 0


class TestAPoolThatCannotStartLeaksNothing:
    """every connect stalled: the failure arrives inside the budget, names its attempts, and holds no socket."""

    async def test_an_all_stalled_server_fails_within_the_budget(
        self, proxy: StallingTcpProxy, db_container: str, application_name: str
    ) -> None:
        proxy.plan = dict.fromkeys(range(1, 100), PROXY_STALL)
        budget = 2.0
        started = time.monotonic()
        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await _start(proxy, db_container, application_name, startup_timeout=budget)
        elapsed = time.monotonic() - started
        err = exc_info.value
        assert elapsed < budget + 0.5, f"took {elapsed:.2f}s against a {budget}s budget"
        assert err.attempts >= 2
        assert f"{err.attempts} attempts" in str(err)
        assert await proxy.all_client_sockets_closed(within=2.0)
        assert await _backends_settle_at(db_container, application_name, 0, within=2.0) == 0

    async def test_a_server_that_stops_answering_mid_attempt_leaks_no_backend(
        self, proxy: StallingTcpProxy, db_container: str, application_name: str
    ) -> None:
        """the first connection of every attempt reaches the server; a later one always stalls."""
        proxy.plan = {n: PROXY_STALL for n in range(1, 100) if n % 2 == 0}
        with pytest.raises(PoolStartupTimeoutError):
            await _start(proxy, db_container, application_name, startup_timeout=2.5, size=2)
        assert await proxy.all_client_sockets_closed(within=2.0)
        assert await _backends_settle_at(db_container, application_name, 0, within=5.0) == 0

    async def test_a_wrong_password_is_not_retried(
        self, proxy: StallingTcpProxy, db_container: str, application_name: str
    ) -> None:
        """a refusal no retry can clear fails on the first attempt, with the server's error as its cause."""
        parts = urlsplit(db_container)
        wrong = urlunsplit(parts._replace(netloc=f"{parts.username}:not-the-password@{parts.hostname}:{parts.port}"))
        started = time.monotonic()
        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await _start(proxy, wrong, application_name)
        assert time.monotonic() - started < 2.0
        assert exc_info.value.attempts == 1
        assert isinstance(exc_info.value.__cause__, asyncpg.exceptions.InvalidPasswordError)
        assert proxy.accepted == 1
        assert "not-the-password" not in str(exc_info.value)
