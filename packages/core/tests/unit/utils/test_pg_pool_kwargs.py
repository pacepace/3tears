"""tests for threetears.core.utils.pg_pool_kwargs.

covers default kwargs, env-var override (valid + invalid + zero +
negative), DSN redaction, the per-connect timeout arithmetic, and the
startup-timeout wrapper against local sockets (no database: a server
that accepts and never answers, a port nothing listens on, and a TLS
server whose certificate the client does not trust). the
same wrapper against a real Postgres behind a stalling proxy is
``tests/integration/test_pool_startup_survives_stalled_connects.py``.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import shutil
import socket
import ssl
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import asyncpg
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from threetears.core.config import DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS, DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS
from threetears.core.utils.pg_pool_kwargs import (
    DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS,
    ENV_MAX_INACTIVE_LIFETIME,
    PoolStartupTimeoutError,
    create_pool_with_startup_timeout,
    get_pg_pool_kwargs,
    redact_dsn,
    resolve_pool_connect_timeout,
)

_LOGGER = "threetears.core.utils.pg_pool_kwargs"


class TestGetPgPoolKwargs:
    """resolved kwargs dict carries the documented default + env override."""

    def test_returns_default_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_MAX_INACTIVE_LIFETIME, raising=False)
        kwargs = get_pg_pool_kwargs()
        assert kwargs == {
            "max_inactive_connection_lifetime": DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS,
        }

    def test_respects_valid_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_MAX_INACTIVE_LIFETIME, "60")
        kwargs = get_pg_pool_kwargs()
        assert kwargs["max_inactive_connection_lifetime"] == 60.0

    def test_falls_back_on_garbage_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_MAX_INACTIVE_LIFETIME, "not-a-number")
        kwargs = get_pg_pool_kwargs()
        assert kwargs["max_inactive_connection_lifetime"] == DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS

    def test_rejects_zero_and_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # zero disables the recycler -- the exact bug the helper guards against
        monkeypatch.setenv(ENV_MAX_INACTIVE_LIFETIME, "0")
        assert get_pg_pool_kwargs()["max_inactive_connection_lifetime"] == DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS
        monkeypatch.setenv(ENV_MAX_INACTIVE_LIFETIME, "-1")
        assert get_pg_pool_kwargs()["max_inactive_connection_lifetime"] == DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS


class TestRedactDsn:
    """credential-free identity string for log lines + error messages."""

    def test_strips_password_segment(self) -> None:
        dsn = "postgres://user:secret-pw@db.example.com:5432/analytics"
        out = redact_dsn(dsn)
        assert "secret-pw" not in out
        assert "user@db.example.com:5432/analytics" == out

    def test_handles_password_with_port(self) -> None:
        dsn = "postgres://user:pw@host:5439/db"
        assert redact_dsn(dsn) == "user@host:5439/db"

    def test_handles_no_user(self) -> None:
        dsn = "postgres://host:5432/db"
        assert redact_dsn(dsn) == "host:5432/db"

    def test_handles_empty_string(self) -> None:
        assert redact_dsn("") == "<unparseable>"

    def test_handles_garbage(self) -> None:
        # urlsplit is tolerant; anything that does not give a hostname returns the sentinel
        assert redact_dsn("not-a-dsn") == "<unparseable>"


class TestResolvePoolConnectTimeout:
    """one wedged connect must cost a fraction of the startup budget, never all of it."""

    def test_the_platform_default_budget_takes_the_platform_default_connect_timeout(self) -> None:
        # 30s budget / 3 = 10s, equal to the platform default.
        assert (
            resolve_pool_connect_timeout(DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS) == DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS
        )

    def test_a_large_budget_keeps_the_platform_default(self) -> None:
        assert resolve_pool_connect_timeout(120.0) == DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS

    def test_a_small_budget_takes_a_third_of_it(self) -> None:
        # a budget below three platform-default connects shrinks the connect bound, so one stalled
        # connect still costs no more than a third of it.
        assert resolve_pool_connect_timeout(6.0) == pytest.approx(2.0)
        assert resolve_pool_connect_timeout(0.9) == pytest.approx(0.3)

    def test_an_explicit_connect_timeout_under_the_budget_is_used_as_given(self) -> None:
        assert resolve_pool_connect_timeout(30.0, 4.5) == 4.5
        assert resolve_pool_connect_timeout(30.0, 29.0) == 29.0

    def test_a_connect_timeout_at_or_over_the_budget_is_refused(self) -> None:
        # the defect being fixed: one wedged connect consuming the whole budget, leaving no retry.
        with pytest.raises(ValueError, match="connect_timeout"):
            resolve_pool_connect_timeout(30.0, 30.0)
        with pytest.raises(ValueError, match="connect_timeout"):
            resolve_pool_connect_timeout(30.0, 60.0)

    @pytest.mark.parametrize(
        ("startup_timeout", "connect_timeout"), [(0.0, None), (-1.0, None), (30.0, 0.0), (30.0, -2.0)]
    )
    def test_a_non_positive_bound_is_refused(self, startup_timeout: float, connect_timeout: float | None) -> None:
        with pytest.raises(ValueError):
            resolve_pool_connect_timeout(startup_timeout, connect_timeout)


class _SilentServer:
    """a TCP server that accepts every connection and never sends a byte: a wedged backend.

    counts what it accepted and how many of those the client has since closed, so a test can
    tell a connection the pool gave up on from one it leaked.
    """

    def __init__(self) -> None:
        self.accepted = 0
        self.closed_by_client = 0
        self._server: asyncio.Server | None = None
        self.port = 0

    async def start(self) -> None:
        """listen on an ephemeral loopback port."""
        self._server = await asyncio.start_server(self._handle, host="127.0.0.1", port=0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def start_unix(self, path: str) -> None:
        """listen on a unix socket at ``path``, as a local Postgres does.

        :param path: socket path
        :ptype path: str
        """
        self._server = await asyncio.start_unix_server(self._handle, path=path)

    async def stop(self) -> None:
        """stop listening and wait for every handler to finish."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """read and discard until the client hangs up; never answer."""
        self.accepted += 1
        try:
            while await reader.read(4096):
                pass
        except ConnectionError:
            pass
        self.closed_by_client += 1
        writer.close()

    async def all_closed(self, *, within: float) -> bool:
        """whether every accepted connection has been closed by the client, waiting up to ``within`` seconds."""
        deadline = time.monotonic() + within
        while self.closed_by_client < self.accepted and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        return self.closed_by_client == self.accepted


@pytest.fixture
async def silent_server() -> AsyncIterator[_SilentServer]:
    """a running :class:`_SilentServer`."""
    server = _SilentServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


def _unused_port() -> int:
    """a loopback port nothing is listening on (bound, then released)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


def _self_signed_certificate() -> tuple[bytes, bytes]:
    """a fresh self-signed certificate and its private key, both PEM."""
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
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return (
        certificate.public_bytes(serialization.Encoding.PEM),
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()),
    )


class _UntrustedTlsServer:
    """a server that answers the Postgres TLS request and presents a certificate of its own making."""

    def __init__(self, certfile: Path, keyfile: Path) -> None:
        self.accepted = 0
        self.port = 0
        self._context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._context.load_cert_chain(certfile, keyfile)
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        """listen on an ephemeral loopback port."""
        self._server = await asyncio.start_server(self._handle, host="127.0.0.1", port=0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """stop listening and wait for every handler to finish."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """accept the client's SSLRequest, then offer the handshake it will refuse."""
        self.accepted += 1
        try:
            await reader.readexactly(8)
            writer.write(b"S")
            await writer.drain()
            await writer.start_tls(self._context)
        except OSError, asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()


@pytest.fixture
async def tls_server(tmp_path: Path) -> AsyncIterator[_UntrustedTlsServer]:
    """a running :class:`_UntrustedTlsServer` with a certificate no client root signs."""
    certificate, key = _self_signed_certificate()
    certfile = tmp_path / "server.crt"
    keyfile = tmp_path / "server.key"
    certfile.write_bytes(certificate)
    keyfile.write_bytes(key)
    server = _UntrustedTlsServer(certfile, keyfile)
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


class TestCreatePoolWithStartupTimeout:
    """a wedged or refused connect costs one per-connect timeout and is retried within the budget."""

    async def test_an_all_silent_server_fails_within_the_budget_naming_the_attempts(
        self, silent_server: _SilentServer
    ) -> None:
        budget = 1.5
        started = time.monotonic()
        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                f"postgresql://u:secret@127.0.0.1:{silent_server.port}/d",
                pool_name="silent",
                startup_timeout=budget,
                connect_timeout=0.3,
                min_size=1,
                max_size=1,
                ssl=False,
            )
        elapsed = time.monotonic() - started
        err = exc_info.value
        assert elapsed < budget + 0.5, f"took {elapsed:.2f}s against a {budget}s budget"
        # the old wrapper spent the whole budget on one connect; now each one is cut short and retried.
        assert err.attempts >= 2
        # one TCP connection per attempt (min_size=1); an attempt the budget cut off as it began may
        # not have reached the server yet.
        assert err.attempts - 1 <= silent_server.accepted <= err.attempts
        assert err.connect_timeout_seconds == 0.3
        assert err.startup_timeout_seconds == budget
        assert err.pool_name == "silent"
        message = str(err)
        assert f"{err.attempts} attempts" in message
        assert f"{budget}s" in message
        assert "TimeoutError" in message
        assert "secret" not in message
        # nothing the abandoned attempts opened is left open.
        assert await silent_server.all_closed(within=2.0)

    async def test_each_retry_is_logged_at_warning_with_attempt_elapsed_and_error_class(
        self, silent_server: _SilentServer, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER), pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                f"postgresql://u:p@127.0.0.1:{silent_server.port}/d",
                pool_name="logged",
                startup_timeout=1.2,
                connect_timeout=0.3,
                min_size=1,
                max_size=1,
                ssl=False,
            )
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        retries = [m for m in warnings if "logged" in m and "attempt" in m and "TimeoutError" in m and "elapsed" in m]
        # every failed attempt that is followed by a pause is logged. the last attempt's failure is
        # logged too when the budget runs out during the pause after it, rather than during it.
        attempts = exc_info.value.attempts
        assert attempts - 1 <= len(retries) <= attempts, warnings
        assert [f"attempt {n} failed" in m for n, m in enumerate(retries, start=1)] == [True] * len(retries), retries

    async def test_a_refused_port_is_retried_and_names_the_refusal(self) -> None:
        port = _unused_port()
        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                f"postgresql://u:hidden@127.0.0.1:{port}/d",
                pool_name="refused",
                startup_timeout=1.0,
                connect_timeout=0.3,
                min_size=1,
                max_size=1,
                ssl=False,
            )
        err = exc_info.value
        assert err.attempts >= 2
        assert isinstance(err.__cause__, OSError)
        assert "hidden" not in str(err)

    async def test_the_default_connect_timeout_is_resolved_from_the_budget(self, silent_server: _SilentServer) -> None:
        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                f"postgresql://u:p@127.0.0.1:{silent_server.port}/d",
                pool_name="defaulted",
                startup_timeout=0.9,
                min_size=1,
                max_size=1,
                ssl=False,
            )
        assert exc_info.value.connect_timeout_seconds == pytest.approx(0.3)
        assert exc_info.value.attempts >= 2

    async def test_a_connect_timeout_the_budget_cannot_retry_is_refused_before_connecting(
        self, silent_server: _SilentServer
    ) -> None:
        with pytest.raises(ValueError, match="connect_timeout"):
            await create_pool_with_startup_timeout(
                f"postgresql://u:p@127.0.0.1:{silent_server.port}/d",
                startup_timeout=1.0,
                connect_timeout=1.0,
            )
        assert silent_server.accepted == 0

    async def test_timeout_is_refused_naming_it_and_its_replacement(self, silent_server: _SilentServer) -> None:
        """``timeout`` is the wrapper's: the refusal names the argument passed and ``connect_timeout=``."""
        # matched on the wrapper's own guidance: python's duplicate-keyword TypeError names the
        # argument too, and would pass a match on the name alone with the wrapper's check gone.
        with pytest.raises(TypeError, match=r"sets timeout= itself; pass the per-connect bound as connect_timeout="):
            await create_pool_with_startup_timeout(
                f"postgresql://u:p@127.0.0.1:{silent_server.port}/d",
                startup_timeout=1.0,
                timeout=5,
            )
        assert silent_server.accepted == 0

    async def test_a_callers_connect_hook_runs_inside_the_wrappers(self, silent_server: _SilentServer) -> None:
        """a pool start with its own connect hook keeps it, bounded and cleaned up like any other."""
        calls: list[dict[str, object]] = []

        async def own_hook(*args: object, **kwargs: object) -> asyncpg.Connection:
            calls.append(dict(kwargs))
            return await asyncpg.connect(*args, **kwargs)

        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                f"postgresql://u:p@127.0.0.1:{silent_server.port}/d",
                pool_name="own_hook",
                startup_timeout=1.5,
                connect_timeout=0.3,
                min_size=1,
                max_size=1,
                ssl=False,
                connect=own_hook,
            )
        attempts = exc_info.value.attempts
        assert attempts >= 2
        # the hook made every connect, each under the wrapper's per-connect bound.
        assert attempts - 1 <= len(calls) <= attempts
        assert [call["timeout"] for call in calls] == [0.3] * len(calls)
        assert await silent_server.all_closed(within=2.0)

    async def test_host_keywords_without_a_dsn_name_the_database_without_the_password(
        self, silent_server: _SilentServer
    ) -> None:
        """a caller that passes ``host=``/``port=``/... instead of a DSN still gets a named, credential-free error."""
        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                pool_name="keywords",
                startup_timeout=0.9,
                min_size=1,
                max_size=1,
                ssl=False,
                host="127.0.0.1",
                port=silent_server.port,
                user="u",
                database="d",
                password="keyword-secret",
            )
        err = exc_info.value
        assert err.db_identity == f"u@127.0.0.1:{silent_server.port}/d"
        assert err.db_identity in str(err)
        assert "keyword-secret" not in str(err)

    async def test_a_budget_that_runs_out_mid_connect_says_during_which_attempt(
        self, silent_server: _SilentServer
    ) -> None:
        """attempt 1 times out at 0.6s, the pause runs to 1.1s, and the budget lapses inside attempt 2."""
        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                f"postgresql://u:p@127.0.0.1:{silent_server.port}/d",
                pool_name="mid_connect",
                startup_timeout=1.5,
                connect_timeout=0.6,
                min_size=1,
                max_size=1,
                ssl=False,
            )
        assert exc_info.value.attempts == 2
        assert "the budget ran out during attempt 2" in str(exc_info.value)

    async def test_a_budget_that_runs_out_in_a_pause_says_before_which_attempt(self) -> None:
        """a refused connect fails at once, so the budget lapses in the pause after the last one."""
        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                f"postgresql://u:p@127.0.0.1:{_unused_port()}/d",
                pool_name="in_pause",
                startup_timeout=1.0,
                connect_timeout=0.3,
                min_size=1,
                max_size=1,
                ssl=False,
            )
        attempts = exc_info.value.attempts
        assert f"the budget ran out before attempt {attempts + 1} could start" in str(exc_info.value)

    async def test_a_client_configuration_error_raises_as_itself(self, silent_server: _SilentServer) -> None:
        """an invalid ``sslmode`` is the caller's mistake, not an unreachable database."""
        started = time.monotonic()
        with pytest.raises(asyncpg.exceptions.ClientConfigurationError, match="sslmode"):
            await create_pool_with_startup_timeout(
                f"postgresql://u:p@127.0.0.1:{silent_server.port}/d?sslmode=bogus",
                startup_timeout=5.0,
                min_size=1,
                max_size=1,
            )
        assert time.monotonic() - started < 1.0
        assert silent_server.accepted == 0

    async def test_a_dsn_whose_password_breaks_the_url_logs_no_part_of_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """an unescaped ``@[`` in a password splits the URL inside the password; nothing logged quotes it."""
        with caplog.at_level(logging.DEBUG, logger=_LOGGER), pytest.raises(ValueError) as exc_info:
            await create_pool_with_startup_timeout(
                "postgresql://u:se@[cretTAIL@127.0.0.1:1/d",
                startup_timeout=1.0,
                min_size=1,
                max_size=1,
            )
        assert "cretTAIL" not in str(exc_info.value)
        assert not [r.getMessage() for r in caplog.records if "cretTAIL" in r.getMessage()]

    @pytest.mark.parametrize(
        "raised",
        [
            pytest.param(lambda secret: asyncpg.exceptions.InterfaceError(f"client says {secret}"), id="not-retried"),
            pytest.param(lambda secret: ConnectionResetError(f"socket says {secret}"), id="retried"),
        ],
    )
    async def test_an_error_quoting_the_password_is_logged_without_it(
        self,
        raised: Callable[[str], Exception],
        silent_server: _SilentServer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """whatever asyncpg's own text quotes, the wrapper's log lines and error do not carry the password."""
        secret = "se@[cretTAIL"

        async def leaking_hook(*args: object, **kwargs: object) -> asyncpg.Connection:
            raise raised(secret)

        with caplog.at_level(logging.DEBUG, logger=_LOGGER), pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                pool_name="leaky",
                startup_timeout=1.0,
                connect_timeout=0.3,
                min_size=1,
                max_size=1,
                host="127.0.0.1",
                port=silent_server.port,
                user="u",
                database="d",
                password=secret,
                connect=leaking_hook,
            )
        logged = [r.getMessage() for r in caplog.records]
        assert logged, "the wrapper logged nothing"
        for fragment in (secret, "cretTAIL"):
            assert not [m for m in logged if fragment in m], logged
            assert fragment not in str(exc_info.value)
        assert type(raised(secret)).__name__ in str(exc_info.value)

    async def test_a_unix_socket_that_does_not_exist_yet_is_retried_until_it_appears(self) -> None:
        """a Postgres still starting has not created its socket; the connect is retried until it has.

        the missing socket raises ``FileNotFoundError`` with no filename -- exactly what a missing
        ``sslrootcert`` raises -- so both are retried, and the final error names the class.
        """
        # a short directory: a unix socket path is limited to ~104 bytes, and pytest's tmp_path
        # on macOS is already close to that.
        socket_dir = Path(tempfile.mkdtemp(prefix="pg", dir="/tmp"))
        port = 5432
        server = _SilentServer()

        async def appear_later() -> None:
            await asyncio.sleep(0.4)
            await server.start_unix(str(socket_dir / f".s.PGSQL.{port}"))

        appearing = asyncio.create_task(appear_later())
        try:
            with pytest.raises(PoolStartupTimeoutError) as exc_info:
                await create_pool_with_startup_timeout(
                    f"postgresql://u:p@/d?host={socket_dir}&port={port}",
                    pool_name="unix_socket",
                    startup_timeout=2.0,
                    connect_timeout=0.3,
                    min_size=1,
                    max_size=1,
                    ssl=False,
                )
            await appearing
            # the socket appeared and a retry reached it; that server never answers, so the
            # start still fails, on a timeout rather than on the missing socket.
            assert server.accepted >= 1
            assert exc_info.value.attempts >= 2
            assert "TimeoutError" in str(exc_info.value)
        finally:
            appearing.cancel()
            await server.stop()
            shutil.rmtree(socket_dir, ignore_errors=True)

    async def test_a_missing_certificate_file_is_retried_and_named(self) -> None:
        """a missing root certificate cannot be told from a missing socket; the error names its class."""
        port = _unused_port()
        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                f"postgresql://u:hidden@127.0.0.1:{port}/d?sslmode=verify-full&sslrootcert=/nonexistent/root.crt",
                pool_name="no_cert_file",
                startup_timeout=1.0,
                connect_timeout=0.3,
                min_size=1,
                max_size=1,
            )
        err = exc_info.value
        assert err.attempts >= 2
        assert isinstance(err.__cause__, FileNotFoundError)
        assert "FileNotFoundError" in str(err)
        assert "hidden" not in str(err)

    async def test_a_host_name_that_does_not_resolve_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a name the resolver says does not exist will not exist on the next attempt either.

        the resolver is patched rather than asked: a machine with no DNS answers ``EAI_AGAIN`` for
        any name, which is retried, and the test would fail for a reason that has nothing to do
        with the rule.
        """
        lookups = 0

        async def no_such_name(*args: object, **kwargs: object) -> list[object]:
            nonlocal lookups
            lookups += 1
            raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided, or not known")

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", no_such_name)
        started = time.monotonic()
        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                "postgresql://u:p@no-such-host.invalid:5432/d",
                pool_name="no_such_host",
                startup_timeout=5.0,
                connect_timeout=1.0,
                min_size=1,
                max_size=1,
                ssl=False,
            )
        err = exc_info.value
        assert time.monotonic() - started < 2.0
        assert err.attempts == 1
        assert lookups == 1
        cause = err.__cause__
        assert isinstance(cause, socket.gaierror)
        assert cause.errno == socket.EAI_NONAME
        assert "not retried" in str(err)

    async def test_a_temporary_resolver_failure_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``EAI_AGAIN`` is the resolver saying try again: a cluster DNS that is not ready yet."""
        lookups = 0

        async def resolver_not_ready(*args: object, **kwargs: object) -> list[object]:
            nonlocal lookups
            lookups += 1
            raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolver_not_ready)
        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                "postgresql://u:p@db.not-ready.example:5432/d",
                pool_name="dns_not_ready",
                startup_timeout=1.0,
                connect_timeout=0.3,
                min_size=1,
                max_size=1,
                ssl=False,
            )
        err = exc_info.value
        assert err.attempts >= 2
        assert lookups >= 2
        cause = err.__cause__
        assert isinstance(cause, socket.gaierror)
        assert cause.errno == socket.EAI_AGAIN

    async def test_a_server_certificate_that_does_not_verify_is_not_retried(
        self, tls_server: _UntrustedTlsServer, tmp_path: Path
    ) -> None:
        """a certificate the client's root does not sign fails the same way on every attempt."""
        other_root = tmp_path / "other-root.crt"
        other_root.write_bytes(_self_signed_certificate()[0])
        started = time.monotonic()
        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                f"postgresql://u:p@127.0.0.1:{tls_server.port}/d?sslmode=verify-ca&sslrootcert={other_root}",
                pool_name="untrusted_cert",
                startup_timeout=5.0,
                connect_timeout=1.0,
                min_size=1,
                max_size=1,
            )
        err = exc_info.value
        assert time.monotonic() - started < 2.0
        assert err.attempts == 1
        assert isinstance(err.__cause__, ssl.SSLCertVerificationError)
        assert "not retried" in str(err)
        assert tls_server.accepted == 1

    async def test_a_configuration_error_is_raised_at_once_and_unwrapped(self, silent_server: _SilentServer) -> None:
        """a bad pool shape is a programming error no retry can clear, and not a database being unreachable."""
        started = time.monotonic()
        with pytest.raises(ValueError, match="min_size"):
            await create_pool_with_startup_timeout(
                f"postgresql://u:p@127.0.0.1:{silent_server.port}/d",
                startup_timeout=5.0,
                min_size=3,
                max_size=1,
            )
        assert time.monotonic() - started < 1.0
        assert silent_server.accepted == 0

    async def test_the_caller_cancelling_leaves_nothing_open(self, silent_server: _SilentServer) -> None:
        """a pod shut down mid-startup closes what the in-flight attempt opened."""
        task = asyncio.create_task(
            create_pool_with_startup_timeout(
                f"postgresql://u:p@127.0.0.1:{silent_server.port}/d",
                startup_timeout=10.0,
                connect_timeout=5.0,
                min_size=1,
                max_size=1,
                ssl=False,
            )
        )
        deadline = time.monotonic() + 2.0
        while silent_server.accepted == 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert silent_server.accepted == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await silent_server.all_closed(within=2.0)
