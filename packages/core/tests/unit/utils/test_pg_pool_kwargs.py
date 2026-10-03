"""tests for threetears.core.utils.pg_pool_kwargs.

covers default kwargs, env-var override (valid + invalid + zero +
negative), DSN redaction, the per-connect timeout arithmetic, and the
startup-timeout wrapper against local sockets (no database: a server
that accepts and never answers, and a port nothing listens on). the
same wrapper against a real Postgres behind a stalling proxy is
``tests/integration/test_pool_startup_survives_stalled_connects.py``.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from collections.abc import AsyncIterator

import pytest

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
        # 30s budget / 3 = 10s, equal to the platform default: three attempts fit.
        assert (
            resolve_pool_connect_timeout(DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS) == DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS
        )

    def test_a_large_budget_keeps_the_platform_default(self) -> None:
        assert resolve_pool_connect_timeout(120.0) == DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS

    def test_a_small_budget_takes_a_third_of_it(self) -> None:
        # a budget below three platform-default connects shrinks the connect bound so at least
        # three attempts still fit inside it.
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

    @pytest.mark.parametrize("owned", ["timeout", "connect"])
    async def test_kwargs_the_wrapper_owns_are_refused(self, owned: str, silent_server: _SilentServer) -> None:
        """``timeout`` is ``connect_timeout``'s, and ``connect`` is how the wrapper closes what a failed attempt opened."""
        with pytest.raises(TypeError, match=owned):
            await create_pool_with_startup_timeout(
                f"postgresql://u:p@127.0.0.1:{silent_server.port}/d",
                startup_timeout=1.0,
                **{owned: 5},
            )
        assert silent_server.accepted == 0

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
