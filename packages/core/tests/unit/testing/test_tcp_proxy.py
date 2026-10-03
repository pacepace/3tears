""":class:`threetears.core.testing.StallingTcpProxy` against a loopback echo server -- no Docker, so CI runs it.

the proxy is how the pool-start integration tests make a real database stall one chosen connect.
those tests need Docker, which CI does not have; these pin the proxy's own behaviour -- forward,
stall, drop, delay, and the open-socket count a leak check reads -- with nothing but sockets.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from threetears.core.testing import PROXY_DROP, PROXY_STALL, StallingTcpProxy


class _EchoServer:
    """a loopback server that writes back every byte it reads."""

    def __init__(self) -> None:
        """not listening yet."""
        self.port = 0
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        """listen on an ephemeral loopback port."""
        self._server = await asyncio.start_server(self._echo, host="127.0.0.1", port=0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """stop listening."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _echo(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """echo until the client hangs up."""
        try:
            while chunk := await reader.read(4096):
                writer.write(chunk)
                await writer.drain()
        except ConnectionError:
            # NOSILENT: the client resetting its socket ends the echo, as EOF does
            pass
        finally:
            writer.close()


@pytest.fixture
async def proxy() -> AsyncIterator[StallingTcpProxy]:
    """a proxy in front of a running echo server."""
    echo = _EchoServer()
    await echo.start()
    running = StallingTcpProxy(upstream_host="127.0.0.1", upstream_port=echo.port)
    await running.start()
    try:
        yield running
    finally:
        await running.stop()
        await echo.stop()


async def _round_trip(proxy: StallingTcpProxy, payload: bytes, *, within: float) -> bytes:
    """send ``payload`` through the proxy and read what comes back within ``within`` seconds, then hang up."""
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
    try:
        writer.write(payload)
        await writer.drain()
        received = b""
        try:
            async with asyncio.timeout(within):
                while len(received) < len(payload):
                    chunk = await reader.read(4096)
                    if not chunk:
                        break
                    received += chunk
        except TimeoutError:
            # NOSILENT: an answer that never comes is what a stalled connection looks like; the caller asserts on it
            pass
    finally:
        writer.close()
    return received


class TestStallingTcpProxy:
    """each planned behaviour, and the count of client sockets a leak check reads."""

    async def test_an_unplanned_connection_is_forwarded(self, proxy: StallingTcpProxy) -> None:
        assert await _round_trip(proxy, b"hello", within=2.0) == b"hello"
        assert proxy.accepted == 1
        assert await proxy.all_client_sockets_closed(within=2.0)

    async def test_a_stalled_connection_is_accepted_and_never_answered(self, proxy: StallingTcpProxy) -> None:
        proxy.plan = {1: PROXY_STALL}
        assert await _round_trip(proxy, b"anyone there", within=0.3) == b""
        assert proxy.accepted == 1
        # the client hung up; the proxy noticed and counts nothing open.
        assert await proxy.all_client_sockets_closed(within=2.0)

    async def test_a_stalled_connection_counts_as_open_until_the_client_closes_it(
        self, proxy: StallingTcpProxy
    ) -> None:
        proxy.plan = {1: PROXY_STALL}
        _, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
        try:
            # the client's connect returns before the proxy's handler runs; wait for the accept.
            async with asyncio.timeout(2.0):
                while proxy.accepted == 0:
                    await asyncio.sleep(0.01)
            # the open socket is what a leak check must see while the client holds it.
            assert not await proxy.all_client_sockets_closed(within=0.3)
            assert proxy.open_client_sockets == 1
        finally:
            writer.close()
        assert await proxy.all_client_sockets_closed(within=2.0)

    async def test_a_dropped_connection_is_closed_at_once(self, proxy: StallingTcpProxy) -> None:
        proxy.plan = {1: PROXY_DROP}
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
        try:
            async with asyncio.timeout(2.0):
                assert await reader.read(4096) == b""
        finally:
            writer.close()
        assert await proxy.all_client_sockets_closed(within=2.0)

    async def test_a_delayed_connection_is_forwarded_after_the_delay(self, proxy: StallingTcpProxy) -> None:
        proxy.plan = {1: 0.4}
        started = time.monotonic()
        assert await _round_trip(proxy, b"later", within=2.0) == b"later"
        assert time.monotonic() - started >= 0.4

    async def test_the_plan_is_by_accept_order(self, proxy: StallingTcpProxy) -> None:
        proxy.plan = {2: PROXY_STALL}
        assert await _round_trip(proxy, b"one", within=1.0) == b"one"
        assert await _round_trip(proxy, b"two", within=0.3) == b""
        assert await _round_trip(proxy, b"three", within=1.0) == b"three"
        assert proxy.accepted == 3

    def test_dsn_through_points_a_dsn_at_the_proxy_and_keeps_its_credentials(self) -> None:
        running = StallingTcpProxy(upstream_host="db", upstream_port=5432)
        running.port = 6543
        assert running.dsn_through("postgresql://u:p@db:5432/d?sslmode=disable") == (
            "postgresql://u:p@127.0.0.1:6543/d?sslmode=disable"
        )
