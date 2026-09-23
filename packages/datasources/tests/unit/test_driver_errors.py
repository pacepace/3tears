"""the shared connection-failure types, and the server reason they carry.

A failed connect used to surface as ``connection failed for host:port/db
(InterfaceError)`` and nothing more, because the backend exception is dropped to keep
the password out of the cause chain. That hid a locked production Redshift account for
hours while every retry sent another failing login. These tests pin the three things
that fix it: the server's SQLSTATE and message survive the drop, an auth refusal is its
own type so a caller can stop at the first one, and the password never rides along.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest
import redshift_connector
from pydantic import SecretStr

from threetears.datasources.drivers import (
    DriverAuthError,
    DriverConnectError,
    DriverMissingCredentialError,
)
from threetears.datasources.drivers.errors import (
    AUTH_SQLSTATES,
    connect_error_from,
    optional_password,
    required_password,
)

_PASSWORD = "horse-battery-staple-NEVER-LOG-ME"


def _redshift_server_error(code: str, message: str) -> Exception:
    """the exception ``redshift_connector`` raises for a server ErrorResponse.

    The library decodes the ErrorResponse into a dict keyed by protocol field code and
    raises ``InterfaceError`` for ``28000`` and ``ProgrammingError`` for the rest.

    :param code: the SQLSTATE the server sent
    :ptype code: str
    :param message: the server's message
    :ptype message: str
    :return: the library's exception for that response
    :rtype: Exception
    """
    fields = {"S": "FATAL", "C": code, "M": message}
    error_type = redshift_connector.InterfaceError if code == "28000" else redshift_connector.ProgrammingError
    return error_type(fields)


class TestOneTypeForEveryDriver:
    """a consumer catches a connect failure once, whichever backend raised it."""

    def test_the_auth_types_are_connect_errors(self) -> None:
        assert issubclass(DriverAuthError, DriverConnectError)
        assert issubclass(DriverMissingCredentialError, DriverAuthError)

    def test_the_auth_sqlstates_are_the_login_refusals(self) -> None:
        assert frozenset({"28000", "28P01"}) == AUTH_SQLSTATES


class TestTheServerReasonSurvives:
    """the SQLSTATE and message are read before the backend exception is dropped."""

    def test_a_redshift_login_refusal_is_an_auth_error_carrying_the_reason(self) -> None:
        exc = _redshift_server_error("28000", 'password authentication failed for user "rs_user"')

        error = connect_error_from("connection failed for rs:5439/analytics", exc, password=SecretStr(_PASSWORD))

        assert type(error) is DriverAuthError
        assert error.sqlstate == "28000"
        assert error.server_message == 'password authentication failed for user "rs_user"'
        assert "28000" in str(error)
        assert "password authentication failed" in str(error)
        assert "connection failed for rs:5439/analytics (InterfaceError)" in str(error)

    def test_a_postgres_password_refusal_is_an_auth_error(self) -> None:
        exc = asyncpg.exceptions.InvalidPasswordError('password authentication failed for user "pg"')

        error = connect_error_from("connection failed for pg:5432/app", exc, password=SecretStr(_PASSWORD))

        assert type(error) is DriverAuthError
        assert error.sqlstate == "28P01"
        assert error.server_message == 'password authentication failed for user "pg"'

    def test_a_server_error_that_is_not_a_login_refusal_stays_a_connect_error(self) -> None:
        exc = _redshift_server_error("3D000", 'database "nope" does not exist')

        error = connect_error_from("connection failed for rs:5439/nope", exc, password=SecretStr(_PASSWORD))

        assert type(error) is DriverConnectError
        assert error.sqlstate == "3D000"
        assert 'database "nope" does not exist' in str(error)

    def test_a_failure_with_no_server_answer_names_only_the_type(self) -> None:
        """a refused socket never reached the server, so there is no reason to read."""
        error = connect_error_from(
            "connection failed for rs:5439/analytics",
            ConnectionRefusedError(61, "Connection refused"),
            password=SecretStr(_PASSWORD),
        )

        assert type(error) is DriverConnectError
        assert error.sqlstate is None
        assert error.server_message is None
        assert str(error) == "connection failed for rs:5439/analytics (ConnectionRefusedError)"


class TestThePasswordNeverRides:
    """whatever the server says, the resolved password does not reach the error."""

    def test_a_server_message_that_echoes_the_password_is_masked(self) -> None:
        exc = _redshift_server_error("28000", f"bad password {_PASSWORD} for user u")

        error = connect_error_from("connection failed for rs:5439/analytics", exc, password=SecretStr(_PASSWORD))

        rendered = str(error) + repr(error) + str(error.server_message)
        assert _PASSWORD not in rendered
        assert "bad password *** for user u" in str(error)

    def test_a_long_server_message_is_bounded(self) -> None:
        exc = _redshift_server_error("XX000", "x" * 5000)

        error = connect_error_from("connection failed for rs:5439/analytics", exc, password=None)

        assert error.server_message is not None
        assert len(error.server_message) == 500


class _UnprimedSecretConfig:
    """a config whose reference names a registered scheme that was never primed.

    A consumer's process-local scheme (the aibots hub's ``secret://``) answers a lookup
    it was never primed for with ``KeyError`` rather than ``ValueError``.
    """

    host = "rs.example.com"
    port = 5439
    database = "analytics"
    password_ref: str | None = "secret://0192f0c4-0000-7000-8000-000000000001"

    def resolve_password(self) -> SecretStr:
        """raise the way an unprimed process-local scheme does.

        :return: never returns
        :rtype: SecretStr
        :raises KeyError: always
        """
        raise KeyError(f"{self.password_ref} was not primed")


class TestAPasswordIsResolvedOrRefused:
    """the two resolution rules the drivers share, and the refusal both make."""

    def test_a_required_password_with_no_reference_is_refused(self) -> None:
        config = _UnprimedSecretConfig()
        config.password_ref = None

        with pytest.raises(DriverMissingCredentialError, match="'central-reporting' has no password"):
            required_password(config, datasource_name="central-reporting")

    def test_an_optional_password_with_no_reference_is_none(self) -> None:
        """Postgres and Yugabyte document no reference as trust authentication."""
        config = _UnprimedSecretConfig()
        config.password_ref = None

        assert optional_password(config, datasource_name="local") is None

    @pytest.mark.parametrize("resolve", [required_password, optional_password])
    def test_a_reference_that_resolves_to_nothing_is_refused_either_way(self, resolve: Any) -> None:
        with pytest.raises(DriverMissingCredentialError) as exc_info:
            resolve(_UnprimedSecretConfig(), datasource_name="influencers-build")

        assert "influencers-build" in str(exc_info.value)
        assert "secret://0192f0c4-0000-7000-8000-000000000001" in str(exc_info.value)
        assert "KeyError" in str(exc_info.value)
        assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("password", [None, SecretStr("")])
def test_an_attempt_with_no_password_masks_nothing(password: SecretStr | None) -> None:
    """an empty secret must not turn every character boundary into a mask."""
    exc = _redshift_server_error("28000", "user u is locked")

    error = connect_error_from("connection failed for rs:5439/analytics", exc, password=password)

    assert error.server_message == "user u is locked"
