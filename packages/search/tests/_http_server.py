"""A minimal scripted HTTP server, for driving the real httpx path.

The standalone transport's obligations -- retry, byte caps, redirect policy,
timeouts -- are properties of what it does with a socket, so they are pinned
against a real one. A mocked httpx transport would test the mock's idea of
those obligations, and the module exists precisely because httpx's defaults
are not the obligations.

Deliberately hand-rolled rather than a framework: it must be able to answer
*badly* -- close mid-response, stall past a deadline, send a body with no
declared length -- and a well-behaved server framework is built not to.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from types import TracebackType

__all__ = ["LocalHttpServer", "Reply"]


@dataclass(frozen=True, slots=True)
class Reply:
    """One scripted response, including the malformed kinds."""

    #: status line code.
    status: int = 200
    #: response body bytes.
    body: bytes = b'{"results": []}'
    #: extra response headers.
    headers: dict[str, str] = field(default_factory=dict)
    #: seconds to stall before answering, for deadline pins.
    delay: float = 0.0
    #: close the connection without answering at all, for retry pins.
    close_early: bool = False
    #: omit ``Content-Length`` and close after the body, so the client must
    #: read until EOF -- the path where a byte cap has to count chunks
    #: rather than trust a header.
    undeclared_length: bool = False


class LocalHttpServer:
    """An async context manager serving a prepared sequence of replies."""

    def __init__(self, replies: tuple[Reply, ...] = (Reply(),)) -> None:
        """Load the reply sequence.

        :param replies: replies in order; an exhausted sequence repeats its
            last entry
        :ptype replies: tuple[Reply, ...]
        :raises ValueError: when the sequence is empty
        """
        if not replies:
            raise ValueError("LocalHttpServer needs at least one reply")
        self._replies = deque(replies)
        self._last = replies[-1]
        self._server: asyncio.Server | None = None
        self._port = 0
        self.requests: list[str] = []
        #: connections accepted, as distinct from requests served. The two
        #: numbers are the same until something reuses a connection, which
        #: makes this the only observable that can tell a shared client from
        #: a fresh one per call.
        self.connections = 0

    @property
    def base_url(self) -> str:
        """Base URL this server is listening on.

        :return: an ``http://127.0.0.1:<port>`` base URL
        :rtype: str
        :raises RuntimeError: when the server is not running
        """
        if self._server is None:
            raise RuntimeError("server is not running")
        return f"http://127.0.0.1:{self._port}"

    async def __aenter__(self) -> LocalHttpServer:
        """Start listening on an ephemeral loopback port.

        :return: this server
        :rtype: LocalHttpServer
        """
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self._port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop listening and wait for the socket to close.

        :param exc_type: exception type, if the block raised
        :ptype exc_type: type[BaseException] | None
        :param exc: the exception, if the block raised
        :ptype exc: BaseException | None
        :param traceback: the traceback, if the block raised
        :ptype traceback: TracebackType | None
        """
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Serve one connection, answering requests until the client stops.

        Keeps the connection open between requests rather than closing after
        one, so that a client holding a pool can be *observed* holding it:
        connection reuse is invisible to a server that hangs up first, and
        invisible reuse cannot be pinned. The scripted sequence advances per
        request, not per connection.

        :param reader: the connection's reader
        :ptype reader: asyncio.StreamReader
        :param writer: the connection's writer
        :ptype writer: asyncio.StreamWriter
        """
        self.connections += 1
        try:
            while True:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                # NOSILENT: a client that hung up between requests is done
                except asyncio.IncompleteReadError, ConnectionError:
                    return
                self.requests.append(head.decode("latin-1"))
                reply = self._replies.popleft() if self._replies else self._last
                if reply.delay:
                    await asyncio.sleep(reply.delay)
                if reply.close_early:
                    return
                headers = {"Content-Type": "application/json", **reply.headers}
                if reply.undeclared_length:
                    headers["Connection"] = "close"
                else:
                    headers.setdefault("Content-Length", str(len(reply.body)))
                rendered = "".join(f"{name}: {value}\r\n" for name, value in headers.items())
                writer.write(f"HTTP/1.1 {reply.status} SCRIPTED\r\n{rendered}\r\n".encode("latin-1") + reply.body)
                try:
                    await writer.drain()
                # NOSILENT: a byte-cap pin aborts the read by design, closing this socket
                except ConnectionError:
                    return
                if reply.undeclared_length:
                    # the body's end is signalled by the close, so there is
                    # no next request to wait for on this connection.
                    return
        finally:
            writer.close()
