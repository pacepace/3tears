"""the live Redshift gate sends a password ONCE, and only a refusal is the operator's to fix.

the live tests log in as a production warehouse user that Redshift locks after
five failed logins, and each test builds its own driver with no connect guard,
so the gate is the only thing between a stale password and a run of refusals.
these drive the gate with the warehouse login patched AND pointed at a host
under ``.invalid`` (RFC 2606: never resolves) as a user that does not exist, so
a patch that stopped matching could not send anything to a real warehouse.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import redshift_connector

from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.entities import DataSourceType

from ._helpers.redshift_live_gate import central_reporting, gated

_CONNECT = "threetears.datasources.drivers.redshift_driver.redshift_connector.connect"
_PASSWORD_ENV = "LIVE_GATE_TEST_PASSWORD"


def _unreachable_config() -> RedshiftConnectionConfig:
    """a Redshift config no login can leave the machine with.

    :return: config for a host that cannot resolve and a user that does not exist
    :rtype: RedshiftConnectionConfig
    """
    return RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host="warehouse.invalid",
        port=5439,
        database="d",
        username="not_a_real_user",
        password_ref=f"env://{_PASSWORD_ENV}",
    )


def _refused() -> redshift_connector.InterfaceError:
    """what ``redshift_connector`` raises when the server refuses a login (SQLSTATE 28000).

    :return: the backend exception
    :rtype: redshift_connector.InterfaceError
    """
    return redshift_connector.InterfaceError(
        {"S": "FATAL", "C": "28000", "M": 'password authentication failed for user "not_a_real_user"'}
    )


def _accepted() -> MagicMock:
    """a connection that answers ``SELECT 1``.

    :return: the connection double
    :rtype: MagicMock
    """
    connection = MagicMock()
    connection.cursor.return_value.fetchone.return_value = (1,)
    return connection


@pytest.fixture(autouse=True)
def _outside_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """run each case as a laptop run unless it says otherwise.

    :param monkeypatch: pytest's environment patcher
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: None
    :rtype: None
    """
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv(_PASSWORD_ENV, "a-password")


class TestTheGate:
    """one login decides whether any live test may send the password."""

    def test_a_refused_login_fails_after_one_attempt_and_names_the_lockout(self) -> None:
        with patch(_CONNECT, side_effect=_refused()) as connect, pytest.raises(pytest.fail.Exception) as failed:
            gated(_unreachable_config())

        assert connect.call_count == 1
        assert "five failed logins" in str(failed.value)
        assert "a-password" not in str(failed.value)

    def test_a_refused_login_fails_in_ci_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "1")

        with patch(_CONNECT, side_effect=_refused()), pytest.raises(pytest.fail.Exception):
            gated(_unreachable_config())

    def test_an_unreachable_warehouse_skips_locally_after_one_attempt(self) -> None:
        with patch(_CONNECT, side_effect=OSError("no route to host")) as connect, pytest.raises(pytest.skip.Exception):
            gated(_unreachable_config())

        assert connect.call_count == 1

    def test_an_unreachable_warehouse_fails_in_ci(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "1")

        with patch(_CONNECT, side_effect=OSError("no route to host")), pytest.raises(pytest.fail.Exception):
            gated(_unreachable_config())

    def test_an_accepted_login_lets_the_tests_run(self) -> None:
        config = _unreachable_config()

        with patch(_CONNECT, return_value=_accepted()) as connect:
            assert gated(config) is config

        assert connect.call_count == 1

    def test_no_password_skips_outside_ci_without_a_login(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_PASSWORD_ENV)

        with patch(_CONNECT) as connect, pytest.raises(pytest.skip.Exception):
            gated(_unreachable_config())

        connect.assert_not_called()

    def test_no_password_fails_in_ci_without_a_login(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_PASSWORD_ENV)
        monkeypatch.setenv("CI", "1")

        with patch(_CONNECT) as connect, pytest.raises(pytest.fail.Exception):
            gated(_unreachable_config())

        connect.assert_not_called()

    def test_an_empty_password_is_no_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_PASSWORD_ENV, "")

        with patch(_CONNECT) as connect, pytest.raises(pytest.skip.Exception):
            gated(_unreachable_config())

        connect.assert_not_called()


def test_the_production_connection_reads_the_documented_variable() -> None:
    """the run instructions in the live test module export ``OTS_REDSHIFT_PASSWORD``."""
    assert central_reporting().password_ref == "env://OTS_REDSHIFT_PASSWORD"
