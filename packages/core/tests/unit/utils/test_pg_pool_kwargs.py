"""tests for threetears.core.utils.pg_pool_kwargs.

covers default kwargs, env-var override (valid + invalid + zero +
negative), DSN redaction, and the startup-timeout wrapper.
"""

from __future__ import annotations

import asyncio

import pytest

from threetears.core.utils.pg_pool_kwargs import (
    DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS,
    ENV_MAX_INACTIVE_LIFETIME,
    PoolStartupTimeoutError,
    create_pool_with_startup_timeout,
    get_pg_pool_kwargs,
    redact_dsn,
)


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


class TestCreatePoolWithStartupTimeout:
    """startup timeout raises a structured error rather than blocking forever."""

    async def test_returns_pool_on_success(self) -> None:
        async def make() -> str:
            return "ok"

        result = await create_pool_with_startup_timeout(
            make,
            dsn="postgres://u:p@h:5432/d",
            startup_timeout=1.0,
            pool_name="test",
        )
        assert result == "ok"

    async def test_raises_on_timeout(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(10.0)
            return "should not reach"

        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                slow,
                dsn="postgres://u:secret@h:5432/d",
                startup_timeout=0.05,
                pool_name="test_timeout",
            )
        # error message must not leak the password
        assert "secret" not in str(exc_info.value)
        assert exc_info.value.pool_name == "test_timeout"
        assert exc_info.value.startup_timeout_seconds == 0.05

    async def test_wraps_transport_error(self) -> None:
        async def fail() -> str:
            raise ConnectionRefusedError("backend down")

        with pytest.raises(PoolStartupTimeoutError) as exc_info:
            await create_pool_with_startup_timeout(
                fail,
                dsn="postgres://u:hidden@h:5432/d",
                startup_timeout=1.0,
                pool_name="test_transport",
            )
        # ConnectionRefusedError chain preserved through `from exc`, but the
        # outer error message must not contain the password
        assert "hidden" not in str(exc_info.value)
        assert exc_info.value.pool_name == "test_transport"
        assert isinstance(exc_info.value.__cause__, ConnectionRefusedError)
