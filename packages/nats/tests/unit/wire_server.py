"""an in-process NATS server speaking just enough of the client protocol for a real nats-py client.

Some behaviour lives in state only nats-py's read loop writes: the queue and pending-byte count of
a subscription that has no callback. A test that wants messages in that queue either reaches into
nats-py's private attributes to put them there, or has a server deliver them. This is the server.

It answers ``CONNECT`` and ``PING``, records every ``SUB`` and ``UNSUB`` it is sent, and delivers
a ``MSG`` to a subscription on request. It routes nothing on its own: a ``PUB`` is read and
dropped, so a test decides exactly what reaches a client and when.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

__all__ = ["WireServer", "wire_server"]

_INFO = {
    "server_id": "threetears-unit-wire-server",
    "version": "2.10.0",
    "proto": 1,
    "max_payload": 1_048_576,
    "headers": True,
}


@dataclass
class WireServer:
    """one listening server; its ``url`` is what a nats-py client connects to.

    :ivar url: ``nats://`` address the server listens on
    :ivar subscribed: every ``SUB`` received, as ``(subject, sid)``, in arrival order
    :ivar unsubscribed: the sid of every ``UNSUB`` received, in arrival order
    """

    url: str = ""
    subscribed: list[tuple[str, int]] = field(default_factory=list)
    unsubscribed: list[int] = field(default_factory=list)
    _writers: dict[int, asyncio.StreamWriter] = field(default_factory=dict)

    def sid_for(self, subject: str) -> int:
        """the sid a client subscribed ``subject`` under.

        :param subject: the subscribed subject
        :ptype subject: str
        :return: the sid of its most recent ``SUB``
        :rtype: int
        :raises LookupError: when nothing subscribed to ``subject``
        """
        sids = [sid for subscribed, sid in self.subscribed if subscribed == subject]
        if not sids:
            raise LookupError(f"nothing subscribed to {subject!r}; subscribed: {self.subscribed}")
        return sids[-1]

    async def deliver(self, subject: str, data: bytes) -> None:
        """send one ``MSG`` for ``subject`` to the connection subscribed to it.

        :param subject: a subject some connection subscribed to
        :ptype subject: str
        :param data: the payload
        :ptype data: bytes
        :return: nothing
        :rtype: None
        """
        sid = self.sid_for(subject)
        writer = self._writers[sid]
        writer.write(f"MSG {subject} {sid} {len(data)}\r\n".encode() + data + b"\r\n")
        await writer.drain()

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """one client connection, until it closes.

        :param reader: the connection's read side
        :ptype reader: asyncio.StreamReader
        :param writer: the connection's write side
        :ptype writer: asyncio.StreamWriter
        :return: nothing
        :rtype: None
        """
        writer.write(f"INFO {json.dumps(_INFO)}\r\n".encode())
        await writer.drain()
        while line := await reader.readline():
            parts = line.decode().split()
            op = parts[0].upper() if parts else ""
            if op == "PING":
                writer.write(b"PONG\r\n")
                await writer.drain()
            elif op == "SUB":
                sid = int(parts[-1])
                self.subscribed.append((parts[1], sid))
                self._writers[sid] = writer
            elif op == "UNSUB":
                self.unsubscribed.append(int(parts[1]))
            elif op in {"PUB", "HPUB"}:
                await reader.readexactly(int(parts[-1]) + 2)
        writer.close()


@asynccontextmanager
async def wire_server() -> AsyncIterator[WireServer]:
    """a :class:`WireServer` listening on a free local port for the duration of the block.

    :return: the running server
    :rtype: AsyncIterator[WireServer]
    """
    state = WireServer()
    server = await asyncio.start_server(state.serve, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    state.url = f"nats://{host}:{port}"
    try:
        yield state
    finally:
        server.close()
        await server.wait_closed()
