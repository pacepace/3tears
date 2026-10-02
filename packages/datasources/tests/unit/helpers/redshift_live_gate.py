"""the gate every live Redshift test passes through: a password, and ONE login the warehouse accepts.

the live tests log in as the ots agent's production warehouse user, which
Redshift locks after five failed logins and never unlocks by itself. each test
builds its own driver with no connect guard, so without this gate a stale
password is sent once per test -- a run of refusals that locks the user.

the gate runs inside the session-scoped ``redshift_config`` fixture
(``tests/integration/conftest.py``), and pytest caches a session fixture's
setup outcome: the gate runs once per session however many tests request it,
and a refusal is re-raised to every later test without logging in again. the
rest of the run is untouched, so the unit suite beside it still reports.

it takes the config to log in with, so its own tests use a host that cannot
resolve and never name the production user.
"""

from __future__ import annotations

import asyncio
import os
from typing import NoReturn

import pytest

from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.drivers import DriverAuthError, DriverConnectError
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.entities import DataSourceType

__all__ = ["central_reporting", "gated"]


def central_reporting() -> RedshiftConnectionConfig:
    """the central-reporting cluster, as the ots agent's production user.

    :return: the connection the live tests use; derive per-test settings from it
    :rtype: RedshiftConnectionConfig
    """
    return RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host="central.c30hiwrajgjj.us-east-1.redshift.amazonaws.com",
        port=5439,
        database="analytics",
        username="fourteen_eng_ai_bot_agent_ots",
        password_ref="env://OTS_REDSHIFT_PASSWORD",
    )


def gated(config: RedshiftConnectionConfig) -> RedshiftConnectionConfig:
    """``config``, once the warehouse has accepted one login with it.

    :param config: the connection to log in with
    :ptype config: RedshiftConnectionConfig
    :return: the same config
    :rtype: RedshiftConnectionConfig
    :raises pytest.skip.Exception: outside CI, when the password is not set or the
        warehouse cannot be reached -- neither sends a password the server refuses
    :raises pytest.fail.Exception: in CI for either of those, where the proof must run;
        and always when the warehouse refuses the login
    """
    try:
        present = bool(config.resolve_password().get_secret_value())
    except ValueError:
        present = False
    if not present:
        _unavailable(f"{config.password_ref} is not set")
    asyncio.run(_log_in_once(config))
    return config


def _unavailable(reason: str) -> NoReturn:
    """skip locally, fail in CI: the live proof cannot run, and no login was refused.

    :param reason: why the proof cannot run
    :ptype reason: str
    :return: never
    :rtype: NoReturn
    :raises pytest.skip.Exception: outside CI
    :raises pytest.fail.Exception: in CI
    """
    if os.environ.get("CI"):
        pytest.fail(f"{reason}; the live Redshift proof must run in CI")
    pytest.skip(f"{reason}; live Redshift tests skipped locally")


async def _log_in_once(config: RedshiftConnectionConfig) -> None:
    """log in once; stop every live test unless the warehouse accepts.

    :param config: the connection to log in with
    :ptype config: RedshiftConnectionConfig
    :return: nothing
    :rtype: None
    :raises pytest.fail.Exception: when the warehouse refuses the login
    :raises pytest.skip.Exception: outside CI, when the warehouse cannot be reached
    """
    driver = RedshiftDriver(config, datasource_name="live-redshift-gate")
    try:
        await driver.test_connection()
    except DriverAuthError as exc:
        pytest.fail(
            f"the warehouse refused {config.username}'s password (sqlstate {exc.sqlstate}); "
            f"no other live test in this session will send it -- Redshift locks the user after "
            f"five failed logins. fix {config.password_ref} before running these"
        )
    except DriverConnectError as exc:
        _unavailable(f"could not reach {config.host} to log in ({type(exc).__name__})")
    finally:
        await driver.close()
