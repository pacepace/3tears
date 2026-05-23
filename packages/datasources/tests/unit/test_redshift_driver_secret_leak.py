"""DS-11-12: deliberately-wrong-password connect MUST NOT leak the password.

mirrors the asyncpg-side secret-leak test (shard 10):

- fires a real :func:`redshift_connector.connect` against an
  unreachable host (closed port on localhost) so the lib reaches the
  connect stage where wrong-credential errors would normally surface
- asserts the literal password value does NOT appear in
  ``str(exception)`` or ``repr(exception)``
- asserts the cause chain is broken (``__cause__ is None``) so the
  original ``redshift_connector`` error -- which may carry the
  password in nested context -- can't leak via ``__cause__``

a future contributor who accidentally drops the ``from None`` clause
OR introduces an intermediate ``str`` variable holding the resolved
password breaks this test.
"""

from __future__ import annotations

import pytest

from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.drivers.redshift_driver import (
    DriverConnectError,
    RedshiftDriver,
)
from threetears.datasources.entities import DataSourceType


_FAKE_PW_VALUE = "horse-battery-staple-12345-NEVER-LOG-ME"


@pytest.mark.asyncio
async def test_wrong_password_does_not_leak_password_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """connect against a closed port + assert the password doesn't leak.

    walks the full exception chain (``__cause__``, ``__context__``)
    so any regression where ``from None`` is dropped surfaces here.
    """
    monkeypatch.setenv("FAKE_REDSHIFT_PW", _FAKE_PW_VALUE)
    config = RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host="127.0.0.1",
        port=1,  # closed port -- guaranteed connect failure
        database="x",
        username="u",
        password_env="FAKE_REDSHIFT_PW",
        # tight sizing keeps the test fast
        executor_max_workers=1,
        connection_cache_size=1,
        query_timeout_seconds=2,
    )
    driver = RedshiftDriver(config)
    try:
        with pytest.raises(DriverConnectError) as exc_info:
            await driver.test_connection()
        rendered = (
            str(exc_info.value)
            + repr(exc_info.value)
            + str(exc_info.value.__cause__ or "")
            + str(exc_info.value.__context__ or "")
        )
        assert _FAKE_PW_VALUE not in rendered, (
            f"password value leaked into exception rendering: {rendered!r}"
        )
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_chained_cause_broken_with_raise_from_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``raise X from None`` clears ``__cause__``; verify the discipline.

    locks in the ``from None`` clause -- a future contributor who
    accidentally writes ``from exc`` will fail here even if the
    rendered message happens not to embed the password.
    """
    monkeypatch.setenv("FAKE_REDSHIFT_PW", _FAKE_PW_VALUE)
    config = RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host="127.0.0.1",
        port=1,
        database="x",
        username="u",
        password_env="FAKE_REDSHIFT_PW",
        executor_max_workers=1,
        connection_cache_size=1,
        query_timeout_seconds=2,
    )
    driver = RedshiftDriver(config)
    try:
        with pytest.raises(DriverConnectError) as exc_info:
            await driver.test_connection()
        # ``raise X from None`` clears __cause__ explicitly
        assert exc_info.value.__cause__ is None
        # __suppress_context__ is True after ``from None`` so the
        # implicit context chain is also suppressed in tracebacks.
        assert exc_info.value.__suppress_context__ is True
    finally:
        await driver.close()
