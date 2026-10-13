"""closing a dead Redshift connection leaves every other TLS connection of the process alone.

the incident these pin (a hub, 2026-10-08): a cached Redshift connection had died while idle. the
driver found out on reuse and closed it, and in the same millisecond unrelated asyncpg connections
on the event loop failed with ``BrokenPipeError`` raised out of ``ssl.SSLObject.read`` -- an
in-memory TLS object that owns no socket and so cannot have a broken pipe of its own.

the mechanism: a write to a dead TLS socket fails inside OpenSSL, which records the system error in
the calling THREAD's error queue. CPython raises the ``OSError`` and leaves the record there, and
OpenSSL consults that queue to classify the next TLS call on the thread that returns no data -- so
an idle connection's "nothing to read yet" is reported as the dead socket's broken pipe, on every
TLS connection the thread serves, until something empties the queue. ``redshift_connector``'s
``Connection.close`` writes its Terminate message and swallows the failure, so a close of a dead
connection is exactly such a write.

each test kills a real TLS connection under a ``redshift_connector``-shaped double, drives the
driver down one of its close paths through its public surface, and then uses a TLS connection the
event loop holds to something else entirely.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import gc
import select
import socket
import ssl
import struct
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import redshift_connector
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.entities import DataSourceType

from .helpers.driver_shims import (
    REDSHIFT_TEST_PASSWORD,
    REDSHIFT_TEST_PASSWORD_ENV,
    REDSHIFT_TEST_PASSWORD_REF,
)

# a stream dropped with a flush it could not make reports that through the unraisable hook, in the
# library as in its double here; that report is the dead connection's, not a defect in the test.
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")

_CONNECT = "threetears.datasources.drivers.redshift_driver.redshift_connector.connect"
#: the pgwire Terminate message ``redshift_connector.Connection.close`` writes.
_TERMINATE = b"X\x00\x00\x00\x04"
#: how long a test waits for something that happens at once when the code is right.
_WAIT_S = 5.0


@pytest.fixture(autouse=True)
def _redshift_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """make :data:`REDSHIFT_TEST_PASSWORD_REF` resolvable for every test in this module.

    :param monkeypatch: pytest's environment patcher
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: None
    :rtype: None
    """
    monkeypatch.setenv(REDSHIFT_TEST_PASSWORD_ENV, REDSHIFT_TEST_PASSWORD)


def _config(*, allowed_schemas: list[str] | None = None) -> RedshiftConnectionConfig:
    """a Redshift config for a driver whose ``connect`` the test replaces.

    :param allowed_schemas: schemas for the ``search_path`` the driver re-applies on reuse
    :ptype allowed_schemas: list[str] | None
    :return: the config
    :rtype: RedshiftConnectionConfig
    """
    return RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host="rs.example.com",
        port=5439,
        database="analytics",
        username="rs_user",
        password_ref=REDSHIFT_TEST_PASSWORD_REF,
        allowed_schemas=allowed_schemas or [],
    )


class _Cursor:
    """a cursor that sends each statement over its connection's TLS stream, as the library does."""

    description = (("col", None),)

    def __init__(self, conn: _TlsConnection) -> None:
        """
        :param conn: the connection whose stream statements go over
        :ptype conn: _TlsConnection
        """
        self._conn = conn

    def execute(self, sql: str, *_args: Any) -> None:
        """send ``sql`` to the peer; on a dead connection this is the write that fails.

        :param sql: statement text
        :ptype sql: str
        :return: None
        :rtype: None
        """
        if self._conn.hold_statements.is_set():
            self._conn.statement_held.set()
            self._conn.release_statements.wait(_WAIT_S)
        self._conn.send(sql.encode())

    def fetchall(self) -> list[tuple[Any, ...]]:
        """no rows; the tests read none."""
        return []

    def fetchone(self) -> tuple[int]:
        """a backend pid, so the driver's cancel path has one to terminate."""
        return (4242,)

    def close(self) -> None:
        """nothing to release."""


class _TlsConnection:
    """a ``redshift_connector.Connection`` stand-in over a real TLS socket.

    laid out as redshift_connector 2.1.7 lays itself out -- the socket on ``_usock``, a buffered
    stream over it -- and closed the way ``Connection.close`` closes: write Terminate, flush,
    swallow the socket error, close the socket, drop the stream.
    """

    def __init__(self, sock: ssl.SSLSocket) -> None:
        """
        :param sock: the connected TLS socket
        :ptype sock: ssl.SSLSocket
        """
        self.socket = sock
        # redshift_connector 2.1.7 Connection.__init__ keeps its socket here; the driver reads it there.
        self._usock = sock
        # buffered, as the library's is: a flush that failed is retried with the same bytes by the
        # next flush and again when the stream is dropped, and each retry is another failed write.
        self._stream: Any = sock.makefile(mode="rwb")
        self.close_threads: list[threading.Thread] = []
        self.hold_statements = threading.Event()
        self.statement_held = threading.Event()
        self.release_statements = threading.Event()

    def send(self, payload: bytes) -> None:
        """write ``payload`` to the peer and flush it.

        :param payload: bytes to send
        :ptype payload: bytes
        :return: None
        :rtype: None
        """
        self._stream.write(payload)
        self._stream.flush()

    def cursor(self) -> _Cursor:
        """a cursor on this connection."""
        return _Cursor(self)

    def commit(self) -> None:
        """nothing to commit."""

    def rollback(self) -> None:
        """nothing to roll back."""

    def close(self) -> None:
        """close as ``redshift_connector.Connection.close`` does, recording the calling thread.

        :return: None
        :rtype: None
        :raises redshift_connector.InterfaceError: when already closed, as the library raises
        """
        self.close_threads.append(threading.current_thread())
        try:
            with contextlib.suppress(OSError):
                self.send(_TERMINATE)
                self._stream.close()
        except AttributeError, ValueError:
            raise redshift_connector.InterfaceError("connection is closed") from None
        finally:
            self._usock.close()
            # the library drops its stream here too, so the stream's last retry of a failed
            # flush happens inside this call, on the closing thread.
            self._stream = None

    def wait_until_peer_reset(self) -> None:
        """block until the kernel has seen the peer's reset, without a TLS call on this thread.

        :return: None
        :rtype: None
        """
        readable, _, _ = select.select([self.socket], [], [], _WAIT_S)
        assert readable, "the peer's reset never arrived"


def _self_signed_certificate() -> tuple[bytes, bytes]:
    """a fresh self-signed certificate and its private key, both PEM.

    :return: (certificate, key)
    :rtype: tuple[bytes, bytes]
    """
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .sign(key, hashes.SHA256())
    )
    return (
        certificate.public_bytes(serialization.Encoding.PEM),
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()),
    )


class _Warehouse:
    """a TLS server standing in for Redshift, whose connections a test can kill.

    it completes the handshake and reads nothing. :meth:`kill_connections` resets every accepted
    connection, which is what an idle connection dropped by the network looks like to its client.
    """

    def __init__(self, server_context: ssl.SSLContext, client_context: ssl.SSLContext) -> None:
        """
        :param server_context: TLS context holding the server certificate
        :ptype server_context: ssl.SSLContext
        :param client_context: TLS context the doubles connect with
        :ptype client_context: ssl.SSLContext
        """
        self._server_context = server_context
        self._client_context = client_context
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._accepted: list[ssl.SSLSocket] = []
        self._lock = threading.Lock()
        self.connections: list[_TlsConnection] = []
        self._thread = threading.Thread(target=self._serve, name="fake-warehouse", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        """accept and handshake until the listener closes."""
        while True:
            try:
                raw, _ = self._listener.accept()
            except OSError:
                # NOSILENT: the listener closing is how this thread is told to stop.
                return
            accepted = self._server_context.wrap_socket(raw, server_side=True)
            with self._lock:
                self._accepted.append(accepted)

    def connect(self, **_kwargs: Any) -> _TlsConnection:
        """the ``redshift_connector.connect`` stand-in: a new TLS connection to this server.

        :return: the connection double
        :rtype: _TlsConnection
        """
        raw = socket.create_connection(self._listener.getsockname(), timeout=_WAIT_S)
        conn = _TlsConnection(self._client_context.wrap_socket(raw))
        self.connections.append(conn)
        return conn

    def kill_connections(self) -> None:
        """reset every connection accepted so far, and wait until each client's kernel knows.

        :return: None
        :rtype: None
        """
        with self._lock:
            accepted, self._accepted = self._accepted, []
        for sock in accepted:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            sock.close()
        for conn in self.connections:
            conn.wait_until_peer_reset()

    def stop(self) -> None:
        """stop accepting and release what is still open."""
        self._listener.close()
        self._thread.join(_WAIT_S)
        for conn in self.connections:
            conn.release_statements.set()
            with contextlib.suppress(OSError, ValueError):
                conn.socket.close()


@pytest.fixture
def tls_contexts(tmp_path: Path) -> tuple[ssl.SSLContext, ssl.SSLContext]:
    """a server context with a throwaway certificate, and a client context that accepts it.

    :param tmp_path: pytest's per-test directory
    :ptype tmp_path: Path
    :return: (server context, client context)
    :rtype: tuple[ssl.SSLContext, ssl.SSLContext]
    """
    certificate, key = _self_signed_certificate()
    certfile, keyfile = tmp_path / "cert.pem", tmp_path / "key.pem"
    certfile.write_bytes(certificate)
    keyfile.write_bytes(key)
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(certfile, keyfile)
    client = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client.check_hostname = False
    client.verify_mode = ssl.CERT_NONE
    return server, client


@pytest.fixture
def warehouse(tls_contexts: tuple[ssl.SSLContext, ssl.SSLContext]) -> Iterator[_Warehouse]:
    """the fake Redshift, stopped at teardown.

    :param tls_contexts: (server context, client context)
    :ptype tls_contexts: tuple[ssl.SSLContext, ssl.SSLContext]
    :return: the running server
    :rtype: Iterator[_Warehouse]
    """
    server = _Warehouse(*tls_contexts)
    yield server
    server.stop()


class _Bystander:
    """a TLS connection the event loop holds to something that is not Redshift."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        :param reader: the connection's read side
        :ptype reader: asyncio.StreamReader
        :param writer: the connection's write side
        :ptype writer: asyncio.StreamWriter
        """
        self._reader = reader
        self._writer = writer

    async def assert_usable(self) -> None:
        """send a message and read its echo; fails when the connection was broken under it.

        :return: None
        :rtype: None
        """
        self._writer.write(b"ping")
        await self._writer.drain()
        echoed = await asyncio.wait_for(self._reader.read(4), _WAIT_S)
        assert echoed == b"ping", "an unrelated TLS connection on the event loop was broken"


@pytest.fixture
async def bystander(tls_contexts: tuple[ssl.SSLContext, ssl.SSLContext]) -> AsyncIterator[_Bystander]:
    """an idle TLS connection on the event loop, to an echo server on the same loop.

    :param tls_contexts: (server context, client context)
    :ptype tls_contexts: tuple[ssl.SSLContext, ssl.SSLContext]
    :return: the connection
    :rtype: AsyncIterator[_Bystander]
    """
    server_context, client_context = tls_contexts

    async def _echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while data := await reader.read(64):
            writer.write(data)
            await writer.drain()

    server = await asyncio.start_server(_echo, "127.0.0.1", 0, ssl=server_context)
    port = server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=client_context)
    connection = _Bystander(reader, writer)
    await connection.assert_usable()
    yield connection
    writer.close()
    server.close()


@pytest.mark.asyncio
async def test_a_dead_cached_connection_is_dropped_without_breaking_other_tls_connections(
    warehouse: _Warehouse,
    bystander: _Bystander,
) -> None:
    """the incident: a cached connection found dead on reuse is closed, and nothing else notices."""
    with patch(_CONNECT, side_effect=warehouse.connect):
        driver = RedshiftDriver(_config(allowed_schemas=["reporting_prod"]))
        try:
            await driver.fetch("SELECT 1")
            (dead,) = warehouse.connections
            warehouse.kill_connections()

            await driver.fetch("SELECT 2")

            assert len(warehouse.connections) == 2, "the dead connection was replaced by a fresh login"
            assert dead.close_threads, "the dead connection was closed"
            await bystander.assert_usable()
            assert threading.current_thread() not in dead.close_threads, (
                "a cached connection is never closed on the event loop thread"
            )
        finally:
            await driver.close()
    await bystander.assert_usable()


@pytest.mark.asyncio
async def test_cancelling_a_statement_on_a_dead_connection_does_not_break_other_tls_connections(
    warehouse: _Warehouse,
    bystander: _Bystander,
) -> None:
    """the cancel path closes the connection to abort the statement; a dead one fails that close."""
    with patch(_CONNECT, side_effect=warehouse.connect):
        driver = RedshiftDriver(_config())
        try:
            await driver.fetch("SELECT 1")
            (dead,) = warehouse.connections
            warehouse.kill_connections()
            dead.hold_statements.set()

            running = asyncio.ensure_future(driver.fetch("SELECT pg_sleep(600)"))
            await asyncio.wait_for(asyncio.to_thread(dead.statement_held.wait, _WAIT_S), _WAIT_S)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            dead.release_statements.set()

            assert dead.close_threads, "the cancelled statement's connection was closed"
            await bystander.assert_usable()
        finally:
            await driver.close()
    await bystander.assert_usable()


@pytest.mark.asyncio
async def test_a_collected_driver_closing_a_dead_connection_does_not_break_other_tls_connections(
    warehouse: _Warehouse,
    bystander: _Bystander,
) -> None:
    """the finalizer closes what is cached on whichever thread collects the driver -- here, the loop's."""
    with patch(_CONNECT, side_effect=warehouse.connect):
        driver = RedshiftDriver(_config())
        await driver.fetch("SELECT 1")
        (dead,) = warehouse.connections
        warehouse.kill_connections()

        del driver
        gc.collect()

        assert dead.close_threads == [threading.current_thread()], "the finalizer closed it on this thread"
        await bystander.assert_usable()
