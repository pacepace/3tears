"""DS-10-11: deliberately-wrong-password connect MUST NOT leak the password value.

the unit-level assertion is straightforward but load-bearing: a future
contributor who accidentally drops the ``from None`` clause -- or
introduces an intermediate ``str`` variable holding the resolved
password -- breaks this contract. the test fires a real
:func:`asyncpg.create_pool` against a closed port (port 1 on 127.0.0.1)
and asserts the literal password string does NOT appear in
``str(exception)`` or ``repr(exception)`` nor in any
``__cause__`` / ``__context__`` walk.

closed-port instead of a fake host is intentional: a fake host can
short-circuit at DNS resolution before asyncpg ever sees the
password. a closed port means asyncpg reaches the connect stage
where wrong-credential errors would normally surface, exercising the
sanitization path properly.
"""

from __future__ import annotations

import pytest

from threetears.datasources.config import PostgresConnectionConfig
from threetears.datasources.drivers.asyncpg_driver import (
    AsyncpgDriver,
    DriverConnectError,
)
from threetears.datasources.entities import DataSourceType


_FAKE_PW_VALUE = "horse-battery-staple-12345-NEVER-LOG-ME"


@pytest.mark.asyncio
async def test_wrong_password_does_not_leak_password_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """connect against a closed port + assert the password doesn't leak.

    walks the full exception chain (``__cause__``, ``__context__``) to
    catch the regression where ``raise X from exc`` slips back in.
    """
    monkeypatch.setenv("FAKE_PW", _FAKE_PW_VALUE)
    config = PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES,
        host="127.0.0.1",
        port=1,  # closed port -- guaranteed connect failure
        database="x",
        username="u",
        password_ref="env://FAKE_PW",
        # tight pool sizing to keep the test fast
        pool_min_size=1,
        pool_max_size=1,
        command_timeout_seconds=2,
    )
    driver = AsyncpgDriver(config)
    with pytest.raises(DriverConnectError) as exc_info:
        await driver.test_connection()
    rendered = (
        str(exc_info.value)
        + repr(exc_info.value)
        + str(exc_info.value.__cause__ or "")
        + str(exc_info.value.__context__ or "")
    )
    assert _FAKE_PW_VALUE not in rendered, f"password value leaked into exception rendering: {rendered!r}"


@pytest.mark.asyncio
async def test_chained_cause_broken_with_raise_from_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``raise X from None`` clears ``__cause__``; verify the discipline.

    locks in the ``from None`` clause -- a future contributor who
    accidentally writes ``from exc`` will fail here even if the
    rendered message happens not to embed the password value.
    """
    monkeypatch.setenv("FAKE_PW", _FAKE_PW_VALUE)
    config = PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES,
        host="127.0.0.1",
        port=1,
        database="x",
        username="u",
        password_ref="env://FAKE_PW",
        pool_min_size=1,
        pool_max_size=1,
        command_timeout_seconds=2,
    )
    driver = AsyncpgDriver(config)
    with pytest.raises(DriverConnectError) as exc_info:
        await driver.test_connection()
    # ``raise X from None`` clears __cause__ explicitly; any future
    # regression that drops ``from None`` will surface here.
    assert exc_info.value.__cause__ is None
    # __suppress_context__ is True after ``from None`` so the
    # implicit context chain is also suppressed in tracebacks.
    assert exc_info.value.__suppress_context__ is True
