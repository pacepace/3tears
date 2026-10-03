"""a pool starts against a real Postgres although some of its connects stall, and leaks nothing when one cannot.

the defect: :func:`create_pool_with_startup_timeout` bounded the whole of ``asyncpg.create_pool`` --
``min_size`` connects -- by one startup budget, and asyncpg's own per-connect timeout (60s) was longer
than that budget. one connect whose backend never answered (a backend stalled under host memory
pressure) consumed the entire budget, nothing retried, and the pool failed to start. it failed an
identity integration test twice in one day, seconds after a plain ``asyncpg.connect`` had run
migrations against the same database; at pod startup in production the same stall crash-loops a pod.

Postgres cannot be made to stall one chosen connect, so these tests put a TCP proxy in front of the
session's Postgres container. the proxy numbers the connections it accepts and, per number, either
forwards it, holds it open and never answers (a wedged backend), drops it at once, or forwards it
only after a delay. every pool connection names a per-test ``application_name``, so the server's own
``pg_stat_activity`` says exactly how many backends the pool holds -- the leak check is taken from
the server, not from the client's bookkeeping.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from threetears.core.utils.pg_pool_kwargs import PoolStartupTimeoutError, create_pool_with_startup_timeout

pytestmark = pytest.mark.integration

_FORWARD = "forward"
_STALL = "stall"
_DROP = "drop"


@dataclass
class _StallingProxy:
    """a TCP proxy that misbehaves on chosen connections, numbered from 1 in accept order.

    ``plan`` maps a connection number to ``stall`` (accept, read, never answer), ``drop`` (close at
    once) or a delay in seconds (forward after waiting it); any other connection is forwarded.
    """

    upstream_host: str
    upstream_port: int
    plan: dict[int, str | float] = field(default_factory=dict)
    accepted: int = 0
    open_client_sockets: int = 0
    port: int = 0
    _server: asyncio.Server | None = None
    _tasks: set[asyncio.Task[None]] = field(default_factory=set)

    async def start(self) -> None:
        """listen on an ephemeral loopback port."""
        self._server = await asyncio.start_server(self._accept, host="127.0.0.1", port=0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """stop listening and end every connection still being served."""
        if self._server is not None:
            self._server.close()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._server is not None:
            await self._server.wait_closed()

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """serve one client connection according to the plan."""
        self.accepted += 1
        number = self.accepted
        self.open_client_sockets += 1
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        try:
            action = self.plan.get(number, _FORWARD)
            if action == _STALL:
                await _drain_until_eof(reader)
            elif action == _DROP:
                pass
            else:
                if isinstance(action, float):
                    await asyncio.sleep(action)
                await self._forward(reader, writer)
        finally:
            self.open_client_sockets -= 1
            writer.close()
            if task is not None:
                self._tasks.discard(task)

    async def _forward(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """pipe bytes both ways until either side closes."""
        up_reader, up_writer = await asyncio.open_connection(self.upstream_host, self.upstream_port)
        try:
            await asyncio.wait(
                [asyncio.create_task(_pipe(reader, up_writer)), asyncio.create_task(_pipe(up_reader, writer))],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            up_writer.close()

    def dsn_through(self, dsn: str) -> str:
        """``dsn`` with its host and port replaced by this proxy's."""
        parts = urlsplit(dsn)
        userinfo = parts.netloc.rsplit("@", 1)[0]
        return urlunsplit(parts._replace(netloc=f"{userinfo}@127.0.0.1:{self.port}"))

    async def all_client_sockets_closed(self, *, within: float) -> bool:
        """whether the client has closed every connection it opened, waiting up to ``within`` seconds."""
        deadline = time.monotonic() + within
        while self.open_client_sockets > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        return self.open_client_sockets == 0


async def _drain_until_eof(reader: asyncio.StreamReader) -> None:
    """read and discard until the peer closes."""
    try:
        while await reader.read(4096):
            pass
    except ConnectionError:
        pass


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """copy ``reader`` to ``writer`` until EOF or a reset."""
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    except ConnectionError:
        pass
    finally:
        writer.close()


@pytest.fixture
async def proxy(db_container: str) -> AsyncIterator[_StallingProxy]:
    """a stalling proxy in front of the session Postgres; each test sets its plan before connecting."""
    upstream = urlsplit(db_container)
    assert upstream.hostname is not None and upstream.port is not None
    running = _StallingProxy(upstream_host=upstream.hostname, upstream_port=upstream.port)
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
    proxy: _StallingProxy,
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
        self, proxy: _StallingProxy, db_container: str, application_name: str
    ) -> None:
        proxy.plan = {1: _STALL}
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
        self, proxy: _StallingProxy, db_container: str, application_name: str
    ) -> None:
        """attempt 1 opens three connections before its third connect stalls; none of them survive it."""
        proxy.plan = {3: _STALL}
        pool = await _start(proxy, db_container, application_name)
        try:
            assert proxy.accepted >= 5, "the first attempt never reached its parallel connects"
            assert pool.get_size() == 4
            assert await _backends_settle_at(db_container, application_name, 4, within=5.0) == 4
        finally:
            await pool.close()
        assert await _backends_settle_at(db_container, application_name, 0, within=5.0) == 0

    async def test_a_connect_finishing_after_its_attempt_failed_is_closed(
        self, proxy: _StallingProxy, db_container: str, application_name: str
    ) -> None:
        """connection 2 is slow and connection 3 is dropped: the attempt fails while 2 is still in flight.

        2 is held for 0.3s -- inside its 0.5s connect bound, so it completes -- and 3 fails at once,
        so the attempt is abandoned before 2 arrives and 2 lands during the pause before the next
        attempt. it must be closed, not left in a pool nobody holds.
        """
        proxy.plan = {2: 0.3, 3: _DROP}
        pool = await _start(proxy, db_container, application_name)
        try:
            assert pool.get_size() == 4
            assert await _backends_settle_at(db_container, application_name, 4, within=5.0) == 4
        finally:
            await pool.close()
        assert await _backends_settle_at(db_container, application_name, 0, within=5.0) == 0


class TestAPoolThatCannotStartLeaksNothing:
    """every connect stalled: the failure arrives inside the budget, names its attempts, and holds no socket."""

    async def test_an_all_stalled_server_fails_within_the_budget(
        self, proxy: _StallingProxy, db_container: str, application_name: str
    ) -> None:
        proxy.plan = dict.fromkeys(range(1, 100), _STALL)
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
        self, proxy: _StallingProxy, db_container: str, application_name: str
    ) -> None:
        """the first connection of every attempt reaches the server; a later one always stalls."""
        proxy.plan = {n: _STALL for n in range(1, 100) if n % 2 == 0}
        with pytest.raises(PoolStartupTimeoutError):
            await _start(proxy, db_container, application_name, startup_timeout=2.5, size=2)
        assert await proxy.all_client_sockets_closed(within=2.0)
        assert await _backends_settle_at(db_container, application_name, 0, within=5.0) == 0

    async def test_a_wrong_password_is_not_retried(
        self, proxy: _StallingProxy, db_container: str, application_name: str
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
