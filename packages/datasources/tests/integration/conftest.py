"""integration-test scope marker and shared live fixtures for 3tears-datasources.

the canonical session-scoped ``db_container`` fixture comes from the
workspace root conftest's
``pytest_plugins = ["threetears.core.testing.fixtures"]`` declaration
(pytest 8+ requires that registration at the rootdir, not in nested
conftests). ``redshift_config`` lives here, once, because every live
Redshift test must pass the same one-login gate; so does ``live_redshift_config``,
the fetch_within proof's own environment-named cluster, because a live module
that builds a connection of its own is refused by the gate's enforcement test.
Other per-test fixtures unique to this package live in the test modules.
"""

from __future__ import annotations

import os

import pytest

from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.entities import DataSourceType

from ..unit.helpers.redshift_live_gate import central_reporting, gated


@pytest.fixture(scope="session")
def redshift_config() -> RedshiftConnectionConfig:
    """the central-reporting connection, once per session, after one accepted login.

    session scope is what makes the gate send the password once: pytest runs a
    session fixture once and re-raises its failure to every later request.

    :return: the base connection; tests derive their pool and timeout settings from it
    :rtype: RedshiftConnectionConfig
    """
    return gated(central_reporting())


#: the environment naming the fetch_within proof's cluster; every one is required
_ENV = ("THREETEARS_LIVE_REDSHIFT_HOST", "THREETEARS_LIVE_REDSHIFT_DATABASE", "THREETEARS_LIVE_REDSHIFT_USER")


@pytest.fixture(scope="session")
def live_redshift_config() -> RedshiftConnectionConfig:
    """the cluster the environment names, once its one login was accepted.

    :return: the connection, one cached connection so a reuse is visible as the same backend
    :rtype: RedshiftConnectionConfig
    """
    missing = [name for name in _ENV if not os.environ.get(name)]
    if missing:
        if os.environ.get("CI"):
            pytest.fail(f"{', '.join(missing)} not set; the live fetch_within proof must run in CI")
        pytest.skip(f"{', '.join(missing)} not set; live fetch_within proof skipped")
    return gated(
        RedshiftConnectionConfig(
            datasource_type=DataSourceType.REDSHIFT,
            host=os.environ["THREETEARS_LIVE_REDSHIFT_HOST"],
            port=int(os.environ.get("THREETEARS_LIVE_REDSHIFT_PORT", "5439")),
            database=os.environ["THREETEARS_LIVE_REDSHIFT_DATABASE"],
            username=os.environ["THREETEARS_LIVE_REDSHIFT_USER"],
            password_ref="env://THREETEARS_LIVE_REDSHIFT_PASSWORD",
            connection_cache_size=1,
            query_timeout_seconds=60,
        )
    )
