""":class:`AsyncpgDriver` starting its pool against a real Postgres whose connects stall.

the driver opens ``pool_min_size`` connections when it first needs its pool. asyncpg opens the
first, then the rest together under ``asyncio.gather``, and a failed connect fails the gather
WITHOUT cancelling its siblings; the failed pool object is then dropped with the siblings' server
connections still open. the driver starts its pool through
:func:`threetears.core.utils.pg_pool_kwargs.create_pool_with_startup_timeout`, its connect guard
running inside the wrapper's hook, so a failed start closes everything it opened.

a :class:`threetears.core.testing.StallingTcpProxy` in front of the session's Postgres container
stalls one chosen connect; the proxy's count of client sockets still open is the leak check.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from urllib.parse import urlsplit

import pytest

from threetears.core.config import DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS, DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS
from threetears.core.testing.tcp_proxy import PROXY_STALL, StallingTcpProxy
from threetears.datasources.config import PostgresConnectionConfig
from threetears.datasources.drivers.asyncpg_driver import AsyncpgDriver
from threetears.datasources.drivers.connect_guard import ConnectGuard
from threetears.datasources.drivers.errors import DriverAuthError, DriverConnectError
from threetears.datasources.entities import DataSourceType

pytestmark = pytest.mark.integration

_PASSWORD_ENV = "ASYNCPG_DRIVER_POOL_START_TEST_PW"


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


def _config_through(
    proxy: StallingTcpProxy, db_container: str, monkeypatch: pytest.MonkeyPatch, *, pool_min_size: int
) -> PostgresConnectionConfig:
    """a driver config for the session Postgres, reached through ``proxy``.

    :param proxy: the running proxy
    :ptype proxy: StallingTcpProxy
    :param db_container: the session Postgres URL
    :ptype db_container: str
    :param monkeypatch: sets the password env var the config reads
    :ptype monkeypatch: pytest.MonkeyPatch
    :param pool_min_size: connections the driver opens when it starts its pool
    :ptype pool_min_size: int
    :return: the config
    :rtype: PostgresConnectionConfig
    """
    upstream = urlsplit(db_container)
    monkeypatch.setenv(_PASSWORD_ENV, upstream.password or "")
    return PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES,
        host="127.0.0.1",
        port=proxy.port,
        database=(upstream.path or "/postgres").lstrip("/"),
        username=upstream.username or "postgres",
        password_ref=f"env://{_PASSWORD_ENV}",
        pool_min_size=pool_min_size,
        pool_max_size=pool_min_size,
        command_timeout_seconds=10,
        allowed_schemas=[],
    )


class _OneLoginAtATime(ConnectGuard):
    """a connect guard that only serializes logins, as every guard does; it never pauses one."""

    def __init__(self) -> None:
        """one slot."""
        self._slot = asyncio.Lock()

    def serialized(self) -> AbstractAsyncContextManager[None]:
        """hold the one login slot.

        :return: the slot's lock, held for the whole login
        :rtype: AbstractAsyncContextManager[None]
        """
        return self._slot

    async def admit(self) -> None:
        """admit every login."""

    async def record_refusal(self, error: DriverAuthError) -> None:
        """nothing is refused in these tests.

        :param error: the refusal
        :ptype error: DriverAuthError
        """
        del error


class TestADriverPoolStartsInABudgetSizedToItsLogins:
    """logins under a connect guard run one at a time, so the start's budget scales with ``pool_min_size``."""

    async def test_four_slow_serialized_logins_start_the_pool(
        self, proxy: StallingTcpProxy, db_container: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """each login is held just under the per-login bound; four in turn need more than the platform's 30s.

        a fixed budget of :data:`DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS` failed this start; each login
        alone always fit its own bound.
        """
        delay = DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS - 1.0
        size = 4
        assert delay * size > DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS
        proxy.plan = dict.fromkeys(range(1, size + 1), delay)
        driver = AsyncpgDriver(
            _config_through(proxy, db_container, monkeypatch, pool_min_size=size),
            connect_guard=_OneLoginAtATime(),
        )
        try:
            await driver.test_connection()
            assert proxy.accepted == size
        finally:
            await driver.close()


class TestADriverPoolThatCannotStartLeaksNothing:
    """one stalled connect among ``pool_min_size`` fails the start, and every connection it opened is closed."""

    async def test_a_stalled_connect_leaves_no_connection_open(
        self, proxy: StallingTcpProxy, db_container: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """connections 1, 2 and 4 reach the server; 3 never answers. the driver's error arrives, and none remain."""
        proxy.plan = {3: PROXY_STALL}
        driver = AsyncpgDriver(_config_through(proxy, db_container, monkeypatch, pool_min_size=4))
        started = time.monotonic()
        try:
            with pytest.raises(DriverConnectError):
                await driver.test_connection()
            # the stalled connect is cut off at the wrapper's per-connect bound, not asyncpg's 60s.
            assert time.monotonic() - started < DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS + 5.0
            assert proxy.accepted == 4
            assert await proxy.all_client_sockets_closed(within=2.0), (
                f"{proxy.open_client_sockets} connections from the failed pool start are still open"
            )
        finally:
            await driver.close()

    async def test_a_pool_that_starts_through_the_proxy_answers(
        self, proxy: StallingTcpProxy, db_container: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the same route with nothing stalled: the driver's connect guard, under the wrapper, connects."""
        driver = AsyncpgDriver(_config_through(proxy, db_container, monkeypatch, pool_min_size=2))
        try:
            await driver.test_connection()
            assert proxy.accepted == 2
        finally:
            await driver.close()
        assert await proxy.all_client_sockets_closed(within=2.0)
