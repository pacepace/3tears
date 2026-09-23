"""the connection-failure types every datasource driver raises.

one set of types for every backend, so a caller catches "this datasource could not
connect" once rather than once per driver module. two drivers used to define their own
unrelated ``DriverConnectError``, and a consumer holding the asyncpg one could not catch
the Redshift one.

**the server's reason travels with the error.** a failed connect used to surface as
``connection failed for host:port/db (InterfaceError)`` and nothing else, because the
backend exception is dropped with ``from None`` (it can carry the password in nested
context). that hid a locked production account for hours: every retry sent another
failing login, and nothing on the error said the account was locked. the SQLSTATE and
the server's own message are now lifted off the backend exception BEFORE it is dropped
-- they are server-generated text that never contains the password, and any occurrence
of the resolved password is masked anyway -- so the reason survives without the chain.

**auth failures are their own type.** a warehouse counts every failed login against the
account and locks it after a handful (Redshift: five, and it never unlocks by itself),
so a caller that retries an auth failure is the thing that locks the account.
:class:`DriverAuthError` lets it stop at the first one. :class:`DriverMissingCredentialError`
is the case where no password was configured at all: refused before any network attempt,
because an empty password is a failed login the server counts like any other.

:func:`required_password` and :func:`optional_password` are where a driver turns its
password reference into the password it sends, and so where that refusal is made: they
live beside the types they raise so every driver refuses the same way.

imports nothing backend-specific: the backend exception is read by shape, so importing
this module keeps the lazy-import contract of :mod:`threetears.datasources.drivers`.
"""

from __future__ import annotations

from typing import Final, Protocol

from pydantic import SecretStr

__all__ = [
    "AUTH_SQLSTATES",
    "DriverAuthError",
    "DriverConnectError",
    "DriverMissingCredentialError",
    "PasswordConfig",
    "connect_error_from",
    "optional_password",
    "required_password",
]

#: SQLSTATEs that mean "the server refused this login": ``28000`` (invalid authorization
#: specification -- Redshift's code for a bad or locked login) and ``28P01`` (invalid
#: password -- Postgres and Yugabyte).
AUTH_SQLSTATES: Final[frozenset[str]] = frozenset({"28000", "28P01"})

#: a server message longer than this is cut. it is diagnostic text for one log line, and a
#: bound keeps a pathological message from flooding a record.
_SERVER_MESSAGE_LIMIT: Final[int] = 500

_MASK: Final[str] = "***"


class DriverConnectError(Exception):
    """a datasource connection could not be opened or prepared.

    the message carries host / port / database (safe to log), the backend exception's
    TYPE, and -- when the server answered -- its SQLSTATE and message. it NEVER carries
    the resolved password. raise it with ``from None`` so the backend exception, which can
    embed the password in nested context, cannot reach a logger through ``__cause__``.

    :param message: human-readable description; MUST NOT carry the password or any other
        resolved secret
    :ptype message: str
    :param sqlstate: the server's SQLSTATE, when the server answered with one
    :ptype sqlstate: str | None
    :param server_message: the server's own error message, password-masked and bounded,
        when the server answered with one
    :ptype server_message: str | None
    """

    def __init__(
        self,
        message: str,
        *,
        sqlstate: str | None = None,
        server_message: str | None = None,
    ) -> None:
        """record the server's reason alongside the message.

        :param message: human-readable description, free of secrets
        :ptype message: str
        :param sqlstate: the server's SQLSTATE, or ``None``
        :ptype sqlstate: str | None
        :param server_message: the server's message, or ``None``
        :ptype server_message: str | None
        :return: None
        :rtype: None
        """
        super().__init__(message)
        self.sqlstate = sqlstate
        self.server_message = server_message


class DriverAuthError(DriverConnectError):
    """the datasource refused the credential, or none was configured.

    retrying cannot succeed and each retry is another failed login the warehouse counts
    toward locking the account, so a caller stops at the first one rather than retrying.
    """


class DriverMissingCredentialError(DriverAuthError):
    """no password could be resolved for a datasource whose driver requires one.

    raised before any network attempt. sending an empty password instead is a failed
    login the server counts like any other, so a datasource with no credential configured
    would lock its account on its own.
    """


class PasswordConfig(Protocol):
    """the part of a connection config :func:`required_password` and :func:`optional_password` read.

    satisfied by the Postgres, Yugabyte and Redshift configs. the fields are read-only
    here so a frozen model satisfies it.
    """

    @property
    def host(self) -> str:
        """the server host.

        :return: host name or address
        :rtype: str
        """
        ...

    @property
    def port(self) -> int:
        """the server port.

        :return: port number
        :rtype: int
        """
        ...

    @property
    def database(self) -> str:
        """the database connected to.

        :return: database name
        :rtype: str
        """
        ...

    @property
    def password_ref(self) -> str | None:
        """the ``scheme://locator`` reference to the password, or ``None``.

        :return: the reference
        :rtype: str | None
        """
        ...

    def resolve_password(self) -> SecretStr:
        """resolve ``password_ref`` to the password.

        :return: the password
        :rtype: SecretStr
        """
        ...


def _resolve(config: PasswordConfig, *, datasource_name: str) -> SecretStr:
    """resolve a SET ``password_ref``, refusing when it resolves to nothing.

    the refusal names the datasource and the reference (``scheme://locator``, never the
    secret) and drops the resolver's exception, which a custom backend could word with
    anything. ``KeyError`` is caught beside ``ValueError`` because a registered scheme's
    resolver raises whatever it raises -- a process-local scheme that was never primed
    answers ``KeyError``.

    :param config: a connection config whose ``password_ref`` is set
    :ptype config: PasswordConfig
    :param datasource_name: the datasource's name, for the refusal message
    :ptype datasource_name: str
    :return: the password
    :rtype: SecretStr
    :raises DriverMissingCredentialError: if the reference resolves to nothing
    """
    try:
        result = config.resolve_password()
    except (ValueError, KeyError) as exc:
        raise DriverMissingCredentialError(
            f"datasource {datasource_name!r} names a password it cannot resolve for "
            f"{config.host}:{config.port}/{config.database} ({config.password_ref}: "
            f"{type(exc).__name__}); refusing to connect with none"
        ) from None
    return result


def required_password(config: PasswordConfig, *, datasource_name: str) -> SecretStr:
    """the password a connect attempt sends, for a backend that cannot connect without one.

    Redshift documents a config with no ``password_ref`` as a local-fixture shape its
    driver refuses, so its driver calls this: the refusal lands before any network
    attempt, because an empty password is a failed login the warehouse counts toward
    locking the account.

    :param config: the connection config
    :ptype config: PasswordConfig
    :param datasource_name: the datasource's name, for the refusal message
    :ptype datasource_name: str
    :return: the password
    :rtype: SecretStr
    :raises DriverMissingCredentialError: if no reference is configured, or it resolves
        to nothing
    """
    if config.password_ref is None:
        raise DriverMissingCredentialError(
            f"datasource {datasource_name!r} has no password configured for "
            f"{config.host}:{config.port}/{config.database} (password_ref is None); refusing to "
            f"connect, because an empty password is a failed login the warehouse counts toward "
            f"locking the account"
        )
    return _resolve(config, datasource_name=datasource_name)


def optional_password(config: PasswordConfig, *, datasource_name: str) -> SecretStr | None:
    """the password a connect attempt sends, for a backend that may connect without one.

    Postgres and Yugabyte document a config with no ``password_ref`` as trust
    authentication, so a config that names none connects with none. A reference that is
    SET but resolves to nothing is still refused: the datasource names a credential it does
    not have, and sending none in its place is a failed login.

    :param config: the connection config
    :ptype config: PasswordConfig
    :param datasource_name: the datasource's name, for the refusal message
    :ptype datasource_name: str
    :return: the password, or ``None`` when the config names none
    :rtype: SecretStr | None
    :raises DriverMissingCredentialError: if a configured reference resolves to nothing
    """
    result: SecretStr | None = None
    if config.password_ref is not None:
        result = _resolve(config, datasource_name=datasource_name)
    return result


def _server_fields(exc: BaseException) -> tuple[str | None, str | None]:
    """read the server's SQLSTATE and message off a backend exception, by shape.

    two shapes carry a server answer. asyncpg's ``PostgresError`` exposes ``sqlstate``
    (and ``message`` when built from the server's fields; ``str()`` otherwise).
    ``redshift_connector`` raises with the decoded ErrorResponse dict as its first
    argument, keyed by protocol field code (``C`` SQLSTATE, ``M`` message). anything
    else -- a refused socket, a DNS failure, a client-side check -- has no server answer
    and yields ``(None, None)``.

    :param exc: the backend exception, before it is dropped
    :ptype exc: BaseException
    :return: ``(sqlstate, message)``, each ``None`` when absent
    :rtype: tuple[str | None, str | None]
    """
    sqlstate: str | None = None
    message: str | None = None
    code = getattr(exc, "sqlstate", None)
    if isinstance(code, str) and code:
        sqlstate = code
        raw = getattr(exc, "message", None)
        message = raw if isinstance(raw, str) and raw else str(exc)
    elif exc.args and isinstance(exc.args[0], dict):
        fields = exc.args[0]
        field_code = fields.get("C")
        field_message = fields.get("M")
        sqlstate = field_code if isinstance(field_code, str) and field_code else None
        message = field_message if isinstance(field_message, str) and field_message else None
    return sqlstate, message


def _masked(message: str, password: SecretStr | None) -> str:
    """mask any occurrence of the resolved password and bound the length.

    a server has no reason to echo a password, and none of the messages observed does; the
    mask is the guarantee that holds even if one ever did.

    :param message: the server's message
    :ptype message: str
    :param password: the password the connect attempt used, or ``None``
    :ptype password: SecretStr | None
    :return: the message with the password masked, cut to the length bound
    :rtype: str
    """
    result = message
    if password is not None and password.get_secret_value():
        result = result.replace(password.get_secret_value(), _MASK)
    return result[:_SERVER_MESSAGE_LIMIT]


def connect_error_from(
    summary: str,
    exc: BaseException,
    *,
    password: SecretStr | None,
) -> DriverConnectError:
    """build the error to raise, ``from None``, for a failed connect.

    call it inside the ``except`` block, with the backend exception still in hand: the
    server's reason is read here and nowhere else, because the caller then drops the
    exception to keep the password out of the chain.

    :param summary: what failed, with the connection identity (e.g.
        ``"connection failed for host:5439/db"``); must not carry a secret
    :ptype summary: str
    :param exc: the backend exception
    :ptype exc: BaseException
    :param password: the password the attempt used, so an echo of it can be masked;
        ``None`` when the attempt carried none
    :ptype password: SecretStr | None
    :return: a :class:`DriverAuthError` when the server refused the login, else a
        :class:`DriverConnectError`
    :rtype: DriverConnectError
    """
    sqlstate, raw_message = _server_fields(exc)
    server_message = _masked(raw_message, password) if raw_message is not None else None
    text = f"{summary} ({type(exc).__name__})"
    if sqlstate is not None or server_message is not None:
        text = f"{text}: SQLSTATE {sqlstate or 'unknown'}: {server_message or 'no message'}"
    error_type = DriverAuthError if sqlstate in AUTH_SQLSTATES else DriverConnectError
    return error_type(text, sqlstate=sqlstate, server_message=server_message)
