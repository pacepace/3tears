"""a NATS server in its own threads that reads every payload whole and answers a request with what it read.

:mod:`.wire_server` runs on the test's own event loop, so a payload of megabytes it reads would itself hold the
loop a test is timing, and its ``readexactly`` builds the payload into one ``bytes``. This one reads on threads of
its own, into one reused buffer, so neither the loop nor the allocator sees the payload on the server's side: what
a test measures there is the client's.

Every ``PUB`` is read in full and digested; one with a reply subject is answered on it with
``b"<size> <sha256 hex>"`` through the subscription that covers the reply (nats-py's request mux subscribes
``_INBOX.<token>.*``), so a request round-trips and proves the bytes arrived whole.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import io

__all__ = ["ByteServer", "byte_server"]

#: big enough for any payload a test sends; nats-py refuses a publish over the server's announced limit
_MAX_PAYLOAD = 256 * 1024 * 1024

#: the reused read buffer: a payload is read through it a slice at a time
_READ_SLICE = 1024 * 1024


@dataclass
class ByteServer:
    """one listening server; its ``url`` is what a nats-py client connects to.

    :ivar url: ``nats://`` address the server listens on
    :ivar received: every ``PUB`` read, as ``(subject, size, sha256 hex)``, in arrival order
    """

    url: str = ""
    received: list[tuple[str, int, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def serve(self, conn: socket.socket) -> None:
        """one client connection, until it closes.

        :param conn: the accepted connection
        :ptype conn: socket.socket
        :return: nothing
        :rtype: None
        """
        info = {"server_id": "threetears-unit-byte-server", "version": "2.10.0", "proto": 1}
        info |= {"max_payload": _MAX_PAYLOAD, "headers": True}
        with conn, io.BufferedReader(socket.SocketIO(conn, "rb"), _READ_SLICE) as stream:
            conn.sendall(f"INFO {json.dumps(info)}\r\n".encode())
            subscriptions: dict[str, int] = {}
            buffer = memoryview(bytearray(_READ_SLICE))
            while line := stream.readline():
                parts = line.decode().split()
                op = parts[0].upper() if parts else ""
                if op == "PING":
                    conn.sendall(b"PONG\r\n")
                elif op == "SUB":
                    subscriptions[parts[1]] = int(parts[-1])
                elif op == "PUB":
                    size = int(parts[-1])
                    digest = _read_digest(stream, size, buffer)
                    if digest is None:
                        return
                    with self._lock:
                        self.received.append((parts[1], size, digest))
                    if len(parts) == 4:
                        _answer(conn, subscriptions, parts[2], f"{size} {digest}".encode())

    @property
    def answered(self) -> list[tuple[str, int, str]]:
        """a copy of :attr:`received`, taken under the server's lock.

        :return: every ``PUB`` read so far
        :rtype: list[tuple[str, int, str]]
        """
        with self._lock:
            return list(self.received)


def _read_digest(stream: io.BufferedReader[socket.SocketIO], size: int, buffer: memoryview) -> str | None:
    """read a ``size``-byte payload and its CRLF through ``buffer``; its sha256, or None if the client left.

    :param stream: the connection's read side
    :ptype stream: io.BufferedReader[socket.SocketIO]
    :param size: the payload's announced size
    :ptype size: int
    :param buffer: the reused read buffer
    :ptype buffer: memoryview
    :return: the payload's sha256 hex digest, or None when the connection closed mid-payload
    :rtype: str | None
    """
    digest = hashlib.sha256()
    left = size
    while left:
        read = stream.readinto(buffer[: min(left, len(buffer))])
        if not read:
            return None
        digest.update(buffer[:read])
        left -= read
    stream.read(2)
    return digest.hexdigest()


def _answer(conn: socket.socket, subscriptions: dict[str, int], reply: str, body: bytes) -> None:
    """send ``body`` to ``reply`` through the subscription covering it (exact, or a trailing ``*``).

    :param conn: the client's connection
    :ptype conn: socket.socket
    :param subscriptions: subject -> sid of every ``SUB`` the client sent
    :ptype subscriptions: dict[str, int]
    :param reply: the reply subject the ``PUB`` named
    :ptype reply: str
    :param body: the answer
    :ptype body: bytes
    :return: nothing
    :rtype: None
    """
    for subject, sid in subscriptions.items():
        if subject == reply or (subject.endswith(".*") and reply.startswith(subject[:-1])):
            conn.sendall(f"MSG {reply} {sid} {len(body)}\r\n".encode() + body + b"\r\n")
            return


@contextmanager
def byte_server() -> Iterator[ByteServer]:
    """a :class:`ByteServer` listening on a free local port, on its own threads, for the duration of the block.

    :return: the running server
    :rtype: Iterator[ByteServer]
    """
    state = ByteServer()
    listener = socket.create_server(("127.0.0.1", 0))
    host, port = listener.getsockname()[:2]
    state.url = f"nats://{host}:{port}"

    listener.settimeout(0.1)
    stop = threading.Event()

    def accept() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            conn.settimeout(None)
            threading.Thread(target=state.serve, args=(conn,), daemon=True).start()

    acceptor = threading.Thread(target=accept, daemon=True)
    acceptor.start()
    try:
        yield state
    finally:
        stop.set()
        acceptor.join(timeout=5)
        listener.close()
