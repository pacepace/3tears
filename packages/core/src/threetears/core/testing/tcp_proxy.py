"""a TCP proxy that stalls, drops or delays chosen connections, for testing what a client does about them.

a real database cannot be made to wedge one chosen connect. put this proxy in front of it and point
the client at :attr:`StallingTcpProxy.port` instead: the proxy numbers the connections it accepts
from 1, and per number either forwards it, holds it open and never answers (a backend stalled under
memory pressure), closes it at once, or forwards it only after a delay. it counts the client sockets
still open, so a test can tell a connection the client gave up on from one it leaked.

usage shape::

    proxy = StallingTcpProxy(upstream_host="localhost", upstream_port=5432)
    await proxy.start()
    proxy.plan = {3: PROXY_STALL}
    try:
        ...  # connect to 127.0.0.1:proxy.port, or proxy.dsn_through(dsn)
        assert await proxy.all_client_sockets_closed(within=2.0)
    finally:
        await proxy.stop()
"""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urlsplit, urlunsplit

#: forward the connection to the upstream (the default for any number the plan does not name).
PROXY_FORWARD = "forward"
#: accept the connection, read and discard what the client sends, and never answer.
PROXY_STALL = "stall"
#: close the connection as soon as it is accepted.
PROXY_DROP = "drop"


class StallingTcpProxy:
    """a TCP proxy that misbehaves on chosen connections, numbered from 1 in accept order.

    ``plan`` maps a connection number to :data:`PROXY_STALL`, :data:`PROXY_DROP`, or a delay in
    seconds (a ``float``: forward after waiting it); any other connection is forwarded.

    :param upstream_host: host the proxy forwards to
    :ptype upstream_host: str
    :param upstream_port: port the proxy forwards to
    :ptype upstream_port: int
    """

    def __init__(self, *, upstream_host: str, upstream_port: int) -> None:
        """a proxy that is not listening yet; :meth:`start` opens its port.

        :param upstream_host: host the proxy forwards to
        :ptype upstream_host: str
        :param upstream_port: port the proxy forwards to
        :ptype upstream_port: int
        """
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.plan: dict[int, str | float] = {}
        self.accepted = 0
        self.open_client_sockets = 0
        self.port = 0
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        """listen on an ephemeral loopback port, recorded in :attr:`port`.

        :return: nothing
        :rtype: None
        """
        self._server = await asyncio.start_server(self._accept, host="127.0.0.1", port=0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """stop listening and end every connection still being served.

        :return: nothing
        :rtype: None
        """
        if self._server is not None:
            self._server.close()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._server is not None:
            await self._server.wait_closed()

    def dsn_through(self, dsn: str) -> str:
        """``dsn`` with its host and port replaced by this proxy's.

        :param dsn: a ``postgresql://user:password@host:port/db`` style URL
        :ptype dsn: str
        :return: the same URL pointed at ``127.0.0.1:<port>``
        :rtype: str
        """
        parts = urlsplit(dsn)
        userinfo = parts.netloc.rsplit("@", 1)[0]
        return urlunsplit(parts._replace(netloc=f"{userinfo}@127.0.0.1:{self.port}"))

    async def all_client_sockets_closed(self, *, within: float) -> bool:
        """whether the client has closed every connection it opened, waiting up to ``within`` seconds.

        :param within: how long to wait for the last one, in seconds
        :ptype within: float
        :return: ``True`` once no client socket is open
        :rtype: bool
        """
        deadline = time.monotonic() + within
        while self.open_client_sockets > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        return self.open_client_sockets == 0

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """serve one client connection according to the plan.

        :param reader: the client's stream
        :ptype reader: asyncio.StreamReader
        :param writer: the client's stream
        :ptype writer: asyncio.StreamWriter
        :return: nothing
        :rtype: None
        """
        self.accepted += 1
        number = self.accepted
        self.open_client_sockets += 1
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        try:
            action = self.plan.get(number, PROXY_FORWARD)
            if action == PROXY_STALL:
                await _drain_until_eof(reader)
            elif action != PROXY_DROP:
                if isinstance(action, float):
                    await asyncio.sleep(action)
                await self._forward(reader, writer)
        finally:
            self.open_client_sockets -= 1
            writer.close()
            if task is not None:
                self._tasks.discard(task)

    async def _forward(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """pipe bytes both ways until either side closes.

        :param reader: the client's stream
        :ptype reader: asyncio.StreamReader
        :param writer: the client's stream
        :ptype writer: asyncio.StreamWriter
        :return: nothing
        :rtype: None
        """
        up_reader, up_writer = await asyncio.open_connection(self.upstream_host, self.upstream_port)
        try:
            await asyncio.wait(
                [asyncio.create_task(_pipe(reader, up_writer)), asyncio.create_task(_pipe(up_reader, writer))],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            up_writer.close()


async def _drain_until_eof(reader: asyncio.StreamReader) -> None:
    """read and discard until the peer closes.

    :param reader: stream to drain
    :ptype reader: asyncio.StreamReader
    :return: nothing
    :rtype: None
    """
    try:
        while await reader.read(4096):
            pass
    except ConnectionError:
        # NOSILENT: a peer that resets the socket has closed it -- the end this drain waits for
        pass


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """copy ``reader`` to ``writer`` until EOF or a reset.

    :param reader: source stream
    :ptype reader: asyncio.StreamReader
    :param writer: destination stream
    :ptype writer: asyncio.StreamWriter
    :return: nothing
    :rtype: None
    """
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    except ConnectionError:
        # NOSILENT: either side resetting its socket ends the pipe, as EOF does; the finally closes the other
        pass
    finally:
        writer.close()


__all__ = ["PROXY_DROP", "PROXY_FORWARD", "PROXY_STALL", "StallingTcpProxy"]
