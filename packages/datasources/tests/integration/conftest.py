"""integration-test scope marker and shared live fixtures for 3tears-datasources.

the canonical session-scoped ``db_container`` fixture comes from the
workspace root conftest's
``pytest_plugins = ["threetears.core.testing.fixtures"]`` declaration
(pytest 8+ requires that registration at the rootdir, not in nested
conftests). ``redshift_config`` lives here, once, because every live
Redshift test must pass the same one-login gate; other per-test fixtures
unique to this package live in the test modules.
"""

from __future__ import annotations

import pytest

from threetears.datasources.config import RedshiftConnectionConfig

from ..unit._helpers.redshift_live_gate import central_reporting, gated


@pytest.fixture(scope="session")
def redshift_config() -> RedshiftConnectionConfig:
    """the central-reporting connection, once per session, after one accepted login.

    session scope is what makes the gate send the password once: pytest runs a
    session fixture once and re-raises its failure to every later request.

    :return: the base connection; tests derive their pool and timeout settings from it
    :rtype: RedshiftConnectionConfig
    """
    return gated(central_reporting())
