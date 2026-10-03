"""asyncpg pool kwargs helper + pool-creation logging + startup-timeout wrapper.

single source of truth for shared ``asyncpg.create_pool(...)`` kwargs
across every 3tears consumer (hub L3 pool, gateway, ``AsyncpgDriver``).
every ``create_pool`` call site splats :func:`get_pg_pool_kwargs` into
the call so ``max_inactive_connection_lifetime`` is explicit, env-tunable,
and never drifts site-by-site.

Yugabyte-specific context
-------------------------

the bug this module guards against: Yugabyte's pgwire layer keeps
session-scoped prepared-statement state tied to the server-side session
object. after an idle interval (roughly 12 hours in practice), the
server evicts that state but the asyncpg client connection in the pool
continues to look healthy. the next request reuses the stale connection
and fails with a cryptic prepared-statement error, and because the pool
is populated with stale connections the failure cascades across several
requests before good connections reestablish.

the fix is to recycle idle connections on the client side *before* the
server-side eviction window. asyncpg's own default for
``max_inactive_connection_lifetime`` is 300 seconds, which is safely
below every Yugabyte server-side timeout observed in practice. this
module makes that default explicit at every call site and exposes an
operator override via ``THREETEARS_PG_POOL_MAX_INACTIVE_LIFETIME_SECONDS``.

Usage at a call site
--------------------

::

    from threetears.core.utils.pg_pool_kwargs import (
        get_pg_pool_kwargs,
        log_pool_created,
    )

    pool = await asyncpg.create_pool(
        dsn,
        min_size=..., max_size=...,
        server_settings=..., init=..., connection_class=...,
        **get_pg_pool_kwargs(),
    )
    log_pool_created(
        pool_name="l3",
        dsn=dsn,
        pool_kwargs={
            "min_size": ..., "max_size": ...,
            **get_pg_pool_kwargs(),
        },
    )

a call site may pass any kwarg this helper does NOT return alongside the
splat, which is the ordinary case::

    pool = await asyncpg.create_pool(
        dsn,
        **get_pg_pool_kwargs(),
        command_timeout=5,  # not returned by the helper, so no collision
    )

**overriding a value the helper DOES return needs a dict, not a second
kwarg.** ``f(**d, a=2)`` where ``d`` contains ``a`` raises
``TypeError: got multiple values for keyword argument 'a'`` -- it does
not let the later one win. This block previously said the opposite,
with ``command_timeout`` as the worked example, which would have crashed
pod startup the day this helper grew that key. Override by merging::

    kwargs = {**get_pg_pool_kwargs(), "max_inactive_connection_lifetime": 60}
    pool = await asyncpg.create_pool(dsn, **kwargs)

Starting a pool inside a budget
-------------------------------

a service that cannot run without its database starts the pool through
:func:`create_pool_with_startup_timeout`, which calls ``asyncpg.create_pool``
itself so it can bound every connect and close what a failed attempt opened::

    pool = await create_pool_with_startup_timeout(
        dsn,
        pool_name="l3",
        startup_timeout=30.0,
        min_size=..., max_size=...,
        server_settings=..., init=..., connection_class=...,
        **get_pg_pool_kwargs(),
    )

each connect is bounded at :func:`resolve_pool_connect_timeout` (10s, or a
third of a smaller budget), and an attempt that fails on a stalled, refused or
dropped connect is retried until the budget is spent. ``timeout=`` is not
passed; ``connect_timeout=`` is.

a pool start with its own connect hook (``AsyncpgDriver``'s connect guard)
passes it as ``connect=``: the wrapper's hook -- how a failed attempt finds
what it opened -- calls it for every connection, so the bound, the retry and
the cleanup reach it too.

Anti-patterns
-------------

- literal ``max_inactive_connection_lifetime=300`` hardcoded at a call
  site (drifts when the platform default moves)
- ``max_inactive_connection_lifetime=0`` (disables the recycler -- the
  exact bug we are fixing)
- omitting the kwarg entirely and relying on asyncpg's default (the
  default is right today but silent drift if it changes upstream, and
  the operator has no tuning knob)
"""

from __future__ import annotations

import asyncio
import os
import socket
import ssl
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

import asyncpg

from threetears.core.config import (
    DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS,
    POOL_START_RETRY_FIRST_DELAY_SECONDS,
    POOL_START_RETRY_MAX_DELAY_SECONDS,
)
from threetears.observe import get_logger
from threetears.observe.resilience import retry_bounded

log = get_logger(__name__)

#: platform default for asyncpg ``max_inactive_connection_lifetime``.
#:
#: matches asyncpg's own default (300s). the point of pinning this is to
#: make every call site explicit rather than to pick a new value.
DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS: float = 300.0

#: env var operators use to tune ``max_inactive_connection_lifetime``
#: without a code change. verbose on purpose -- it is an operator tuning
#: knob, not a dev convenience.
ENV_MAX_INACTIVE_LIFETIME: str = "THREETEARS_PG_POOL_MAX_INACTIVE_LIFETIME_SECONDS"


def get_pg_pool_kwargs() -> dict[str, Any]:
    """return the shared ``asyncpg.create_pool`` kwargs for the platform.

    current keys:

    - ``max_inactive_connection_lifetime`` -- sourced from
      :data:`ENV_MAX_INACTIVE_LIFETIME` (default
      :data:`DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS`).

    call sites splat the return value into ``create_pool``::

        pool = await asyncpg.create_pool(dsn, ..., **get_pg_pool_kwargs())

    a malformed, non-positive, or zero env value falls back to the
    default and logs a WARNING so misconfigured operators surface
    loudly. zero is explicitly rejected: setting it disables asyncpg's
    recycler, which is the exact production bug this helper exists to
    prevent.

    :return: mapping of pool kwargs safe to splat into ``create_pool``
    :rtype: dict[str, Any]
    """
    lifetime = _resolve_max_inactive_lifetime()
    return {
        "max_inactive_connection_lifetime": lifetime,
    }


def _resolve_max_inactive_lifetime() -> float:
    """read ``max_inactive_connection_lifetime`` from env with fallback.

    invalid, non-positive, or zero values fall back to
    :data:`DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS` and log a WARNING.

    :return: resolved lifetime in seconds
    :rtype: float
    """
    raw = os.environ.get(ENV_MAX_INACTIVE_LIFETIME)
    result = DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS
    if raw is not None:
        parsed: float | None = None
        try:
            parsed = float(raw)
        except ValueError:
            log.warning(
                f"invalid {ENV_MAX_INACTIVE_LIFETIME}={raw!r}, using default {DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS}s",
                extra={
                    "extra_data": {
                        "env_var": ENV_MAX_INACTIVE_LIFETIME,
                        "env_raw": raw,
                        "fallback": DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS,
                    }
                },
            )
        if parsed is not None:
            if parsed <= 0:
                log.warning(
                    f"rejected {ENV_MAX_INACTIVE_LIFETIME}={raw!r}: "
                    f"non-positive lifetime disables connection recycling "
                    f"(the exact Yugabyte-pgwire bug we guard against); "
                    f"using default {DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS}s",
                    extra={
                        "extra_data": {
                            "env_var": ENV_MAX_INACTIVE_LIFETIME,
                            "env_raw": raw,
                            "parsed": parsed,
                            "fallback": DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS,
                        }
                    },
                )
            else:
                result = parsed
    return result


def redact_dsn(dsn: str) -> str:
    """return a credential-free ``user@host:port/dbname`` identity string.

    credentials (the password segment of the userinfo) are stripped.
    an unparseable or empty DSN returns the sentinel ``<unparseable>``
    so operators can see that the helper saw a string but could not
    decode it, which is more actionable than an empty log field.

    the function is deliberately tolerant: pool creation must never
    fail because logging cannot parse the DSN. a DSN whose password
    breaks the URL -- an unescaped ``?`` or ``#`` ends the netloc inside
    the password, leaving a piece of it where the port belongs -- is
    ``<unparseable>`` rather than an identity built from that piece.

    :param dsn: raw asyncpg DSN / libpq connection URL
    :ptype dsn: str
    :return: credential-free connection identity, or ``<unparseable>``
    :rtype: str
    """
    if not dsn:
        return "<unparseable>"
    result = "<unparseable>"
    try:
        parts = urlsplit(dsn)
    except ValueError:
        parts = None
    if parts is not None and "@" in dsn.split("://", 1)[-1] and "@" not in parts.netloc:
        # the userinfo's ``@`` landed outside the netloc: the URL split inside the password.
        parts = None
    if parts is not None:
        host = parts.hostname
        port: int | None = None
        try:
            port = parts.port
        except ValueError:
            host = None
        if host:
            user = parts.username or ""
            path = parts.path.lstrip("/") if parts.path else ""
            userhost = f"{user}@{host}" if user else host
            hostport = f"{userhost}:{port}" if port is not None else userhost
            result = f"{hostport}/{path}" if path else hostport
    return result


def log_pool_created(
    pool_name: str,
    dsn: str,
    pool_kwargs: dict[str, Any],
) -> None:
    """emit the mandatory INFO log at ``asyncpg.create_pool`` success.

    required by the platform logging contract: every connection open
    logs at INFO with structured fields operators can query in Loki.
    specifically, this lets operators confirm from the log alone which
    ``max_inactive_connection_lifetime`` each pod is running with --
    critical for diagnosing the Yugabyte-pgwire stale connection bug
    this helper guards against.

    the raw DSN is never logged (it contains credentials); the redacted
    ``user@host:port/dbname`` form goes into
    ``extra_data.connection_identity`` instead.

    :param pool_name: short identifier for the pool (e.g. ``l3``,
        ``gateway_l3``, ``datasource``) included in the log so multiple
        pools in one pod are distinguishable
    :ptype pool_name: str
    :param dsn: raw DSN the pool was created from (redacted before
        logging)
    :ptype dsn: str
    :param pool_kwargs: kwargs passed to ``create_pool`` (or an
        operator-friendly subset of them); logged verbatim so operators
        can see the resolved configuration
    :ptype pool_kwargs: dict[str, Any]
    :return: nothing
    :rtype: None
    """
    identity = redact_dsn(dsn)
    log.info(
        f"pg pool created: name={pool_name} identity={identity}",
        extra={
            "extra_data": {
                "pool_name": pool_name,
                "connection_identity": identity,
                **pool_kwargs,
            }
        },
    )


class PoolStartupTimeoutError(Exception):
    """raised when ``create_pool_with_startup_timeout`` cannot start the pool.

    either the startup budget was spent on attempts the database never completed (a refused or
    stalled connect, a server starting or out of connections), or the database refused in a way no
    retry can clear (a wrong password, a missing database), which fails on the attempt that met it.

    carries structured context so callers can translate to their own structured-error type (Hub
    maps to its ``ConfigurationError``; other consumers map to whatever their stack uses).
    concrete error-translation belongs at the consumer boundary, which is why the type lives here.

    :param message: human-readable description
    :ptype message: str
    :param pool_name: short identifier for the pool that failed to start
    :ptype pool_name: str
    :param db_identity: redacted db identity (no credentials)
    :ptype db_identity: str
    :param startup_timeout_seconds: the overall budget
    :ptype startup_timeout_seconds: float
    :param elapsed_seconds: wall-clock seconds elapsed before the failure
    :ptype elapsed_seconds: float
    :param attempts: how many pool-creation attempts began, the last of them the one that failed
    :ptype attempts: int
    :param connect_timeout_seconds: the bound each single connect ran under
    :ptype connect_timeout_seconds: float
    """

    def __init__(
        self,
        message: str,
        *,
        pool_name: str,
        db_identity: str,
        startup_timeout_seconds: float,
        elapsed_seconds: float,
        attempts: int,
        connect_timeout_seconds: float,
    ) -> None:
        super().__init__(message)
        self.pool_name = pool_name
        self.db_identity = db_identity
        self.startup_timeout_seconds = startup_timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        self.attempts = attempts
        self.connect_timeout_seconds = connect_timeout_seconds


#: the share of the startup budget one connect may take when the caller names no connect bound: a
#: third, so a connect whose backend never answers costs at most a third of the budget.
_DEFAULT_CONNECT_SHARE_OF_BUDGET = 3.0

#: what a later attempt may not meet. the connect timed out (a backend that never answered), the
#: socket failed or was refused, the server dropped the connection mid-handshake, or it answered
#: that it is starting, shutting down, or out of connections or memory. anything else -- a wrong
#: password, a missing database -- no retry can clear, and fails on the attempt that met it.
_RETRYABLE: tuple[type[Exception], ...] = (
    TimeoutError,
    OSError,
    asyncpg.exceptions.PostgresConnectionError,
    asyncpg.exceptions.OperatorInterventionError,
    asyncpg.exceptions.InsufficientResourcesError,
)

#: a server certificate that does not verify: the caller's trust configuration, not the network. a
#: later attempt meets the same certificate, so it fails on the attempt that met it.
#:
#: ``FileNotFoundError`` is deliberately NOT here. asyncpg raises it, with no filename, both for a
#: ``sslrootcert`` that is not on disk and for a unix socket a still-starting Postgres has not
#: created yet; the two cannot be told apart, the second is exactly what a retry is for, and the
#: startup budget bounds the cost of the first. the final error names ``FileNotFoundError``.
_NEVER_RETRYABLE: tuple[type[Exception], ...] = (ssl.SSLCertVerificationError,)


def _is_retryable(error: Exception) -> bool:
    """whether a later pool-start attempt may not meet ``error``.

    a resolver failure is retried only when the resolver says so (``EAI_AGAIN``: a cluster DNS that
    is not ready yet); a name it says does not exist will not exist on the next attempt.

    :param error: what an attempt failed with
    :ptype error: Exception
    :return: ``True`` when the failure is worth another attempt
    :rtype: bool
    """
    result = isinstance(error, _RETRYABLE) and not isinstance(error, _NEVER_RETRYABLE)
    if isinstance(error, socket.gaierror):
        result = error.errno == socket.EAI_AGAIN
    return result


#: what the database or the network can answer a pool start with; each is reported as a
#: :class:`PoolStartupTimeoutError`. anything else (a bad pool shape, a wrong argument) is a
#: programming error and propagates unchanged. a client-side error met while connecting (see
#: :class:`_ClientSideFailure`) is the caller's mistake too, and raises as its own type.
_DATABASE_FAILURES: tuple[type[Exception], ...] = (
    TimeoutError,
    OSError,
    asyncpg.exceptions.PostgresError,
    asyncpg.exceptions.InterfaceError,
)

#: the message a client-side error is re-raised with, in place of the library's text. a client-side
#: error -- a DSN asyncpg cannot parse, a connect option it refuses -- describes what the client
#: SENT, and with a stray ``@`` or ``?`` in a password, what it sent and quotes is the password.
_WITHHELD_CLIENT_ERROR = (
    "invalid connection configuration{target} (details withheld: they may contain credentials); "
    "check host, port, user, database and sslmode"
)


def resolve_pool_connect_timeout(startup_timeout: float, connect_timeout: float | None = None) -> float:
    """the bound one connect runs under while a pool starts inside ``startup_timeout``.

    with no ``connect_timeout``: the smaller of :data:`DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS` and a
    third of the budget, so one stalled connect costs at most a third of it. an explicit
    ``connect_timeout`` is used as given, but must be shorter than the budget: one equal to it lets
    a single wedged connect spend the whole budget with nothing retried, which is the defect this
    bound exists to remove.

    :func:`create_pool_with_startup_timeout` resolves its bound here; a caller logging the pool's
    configuration (:func:`log_pool_created`) resolves it the same way.

    :param startup_timeout: the overall pool-start budget, in seconds (> 0)
    :ptype startup_timeout: float
    :param connect_timeout: an explicit per-connect bound, in seconds (> 0 and < ``startup_timeout``),
        or ``None`` for the default
    :ptype connect_timeout: float | None
    :return: the per-connect bound in seconds
    :rtype: float
    :raises ValueError: when either bound is not positive, or ``connect_timeout`` is not shorter
        than ``startup_timeout``
    """
    if startup_timeout <= 0:
        raise ValueError(f"startup_timeout must be positive, got {startup_timeout!r}")
    if connect_timeout is not None and connect_timeout <= 0:
        raise ValueError(f"connect_timeout must be positive, got {connect_timeout!r}")
    if connect_timeout is not None and connect_timeout >= startup_timeout:
        raise ValueError(
            f"connect_timeout={connect_timeout}s must be shorter than startup_timeout={startup_timeout}s: "
            f"one connect that never answers would spend the whole startup budget and leave nothing to retry"
        )
    result = connect_timeout
    if result is None:
        result = min(DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS, startup_timeout / _DEFAULT_CONNECT_SHARE_OF_BUDGET)
    return result


class _AttemptAbandonedError(Exception):
    """a connect that completed after its pool attempt had already failed; its connection is closed."""


#: the attempt is still opening its pool: every connect is recorded.
_ATTEMPT_RECORDING = "recording"
#: the attempt failed: what it opened is closed, and a connect completing late is closed too.
_ATTEMPT_ABANDONED = "abandoned"
#: the attempt's pool started: the pool owns its connections and nothing more is recorded.
_ATTEMPT_STARTED = "started"


class _AttemptConnections:
    """every connection one pool-creation attempt opens, so a failed attempt closes all of them.

    passed to ``asyncpg.create_pool`` as its ``connect`` hook; it opens each connection through
    ``opener`` -- ``asyncpg.connect``, or the caller's own ``connect`` hook, which therefore runs
    inside this one. asyncpg opens a pool's first
    connection, then the rest of ``min_size`` together under ``asyncio.gather``, and a failed
    connect fails the gather WITHOUT cancelling its siblings. the pool object that failed is then
    dropped, so without this record a sibling already connected -- or one still connecting that
    completes later -- holds a server backend nothing will ever close.

    asyncpg keeps the hook for the life of the pool and calls it for every connection the pool
    later opens (growth toward ``max_size``, each replacement of a recycled one). the record is the
    attempt's, not the pool's: once the pool has started (:meth:`pool_started`) the hook opens
    connections and records nothing.
    """

    def __init__(self, opener: Callable[..., Awaitable[asyncpg.Connection]]) -> None:
        """start with nothing opened.

        :param opener: what opens one connection, called with asyncpg's connect arguments
        :ptype opener: Callable[..., Awaitable[asyncpg.Connection]]
        """
        self.opener = opener
        self.opened: list[asyncpg.Connection] = []
        self.connecting: set[asyncio.Task[Any]] = set()
        self.state = _ATTEMPT_RECORDING

    async def connect(self, *args: Any, **kwargs: Any) -> asyncpg.Connection:
        """open one connection through the opener, recording it while the attempt is in progress.

        :param args: positional connect arguments asyncpg passes through (the dsn)
        :ptype args: Any
        :param kwargs: keyword connect arguments asyncpg passes through
        :ptype kwargs: Any
        :return: the new connection
        :rtype: asyncpg.Connection
        :raises _AttemptAbandonedError: when the attempt failed while this connect was in flight
        :raises _ClientSideFailure: when asyncpg's connection-parameter handling refused what was
            sent (see :func:`_is_connection_parameter_error`)
        """
        if self.state == _ATTEMPT_STARTED:
            started_pool_connection: asyncpg.Connection = await self.opener(*args, **kwargs)
            return started_pool_connection
        task = asyncio.current_task()
        if task is not None:
            self.connecting.add(task)
        try:
            connection = await self.opener(*args, **kwargs)
        except ValueError as exc:
            if not _is_connection_parameter_error(exc):
                raise
            raise _ClientSideFailure(exc) from None
        finally:
            if task is not None:
                self.connecting.discard(task)
        if self.state == _ATTEMPT_ABANDONED:
            connection.terminate()
            raise _AttemptAbandonedError("pool attempt already failed; its late connection was closed")
        if self.state == _ATTEMPT_RECORDING:
            self.opened.append(connection)
        return connection

    def pool_started(self) -> None:
        """the attempt's pool started: hand its connections to the pool and stop recording."""
        self.state = _ATTEMPT_STARTED
        self.opened.clear()

    def abandon(self, pool: object) -> int:
        """close everything the failed attempt opened and stop what it is still opening.

        synchronous, so it completes even inside a task that is being cancelled. every connection
        the attempt opened went through :meth:`connect` and is closed from the record; the pool
        itself -- what ``asyncpg.create_pool`` returned, already through its failed initialisation
        -- is terminated too when it is an ``asyncpg.Pool``, which also cancels the idle timers its
        connection holders set.

        :param pool: what the attempt's ``asyncpg.create_pool`` call returned
        :ptype pool: object
        :return: how many connections were closed or stopped mid-connect
        :rtype: int
        """
        self.state = _ATTEMPT_ABANDONED
        current = asyncio.current_task()
        stopped = 0
        for task in list(self.connecting):
            if task is not current and not task.done():
                task.cancel()
                stopped += 1
        # counted before ``pool.terminate()``, which closes the ones the pool already holds.
        still_open = [connection for connection in self.opened if not connection.is_closed()]
        stopped += len(still_open)
        if isinstance(pool, asyncpg.Pool):
            pool.terminate()
        for connection in still_open:
            connection.terminate()
        self.opened.clear()
        return stopped


class _PoolStartProgress:
    """what the attempts so far have met, for the retry log and the final error."""

    def __init__(self) -> None:
        """no attempt yet."""
        self.attempts = 0
        self.attempt_in_flight = False
        self.last_failure: Exception | None = None
        self.closed_by_last_failure = 0

    def where_the_budget_ran_out(self) -> str:
        """``during attempt N`` or ``before attempt N could start``, for the final error.

        ``attempt_in_flight`` is set when an attempt begins and cleared only when it succeeds or its
        failure is handed to a retry, so the budget's cancellation unwinding an attempt does not
        clear it before this reads it.

        :return: the phrase
        :rtype: str
        """
        result = f"before attempt {self.attempts + 1} could start"
        if self.attempt_in_flight:
            result = f"during attempt {self.attempts}"
        return result


def _attempts_phrase(attempts: int) -> str:
    """``1 attempt`` / ``3 attempts``.

    :param attempts: attempt count
    :ptype attempts: int
    :return: the count with its noun
    :rtype: str
    """
    return f"{attempts} attempt" if attempts == 1 else f"{attempts} attempts"


def _keyword_target(create_pool_kwargs: dict[str, Any]) -> str | None:
    """``user@host:port/database`` from the ``host`` / ``port`` / ``user`` / ``database`` keywords, if any.

    never from a DSN string: a DSN the URL parser misreads can put a piece of the password where
    the host or port belongs.

    :param create_pool_kwargs: the caller's ``create_pool`` keywords
    :ptype create_pool_kwargs: dict[str, Any]
    :return: the target, or ``None`` when no ``host`` keyword names one
    :rtype: str | None
    """
    result: str | None = None
    if create_pool_kwargs.get("host") is not None:
        user = create_pool_kwargs.get("user")
        port = create_pool_kwargs.get("port")
        database = create_pool_kwargs.get("database")
        result = str(create_pool_kwargs["host"])
        if user is not None:
            result = f"{user}@{result}"
        if port is not None:
            result = f"{result}:{port}"
        if database is not None:
            result = f"{result}/{database}"
    return result


def _pool_identity(dsn: str | None, create_pool_kwargs: dict[str, Any]) -> str:
    """the credential-free ``user@host:port/dbname`` a pool start is reported under.

    from the DSN when there is one (:func:`redact_dsn`), otherwise from the keywords a caller passed
    instead (:func:`_keyword_target`).

    :param dsn: the DSN, or ``None``
    :ptype dsn: str | None
    :param create_pool_kwargs: the caller's ``create_pool`` keywords
    :ptype create_pool_kwargs: dict[str, Any]
    :return: the identity, or ``<unparseable>`` when nothing names a host
    :rtype: str
    """
    result = "<unparseable>"
    if dsn is not None:
        result = redact_dsn(dsn)
    else:
        keyword_target = _keyword_target(create_pool_kwargs)
        if keyword_target is not None:
            result = keyword_target
    return result


def _describe_error(error: BaseException) -> str:
    """an error as the wrapper reports it in its log lines and its :class:`PoolStartupTimeoutError`.

    a server answer (a refused login, a missing database, too many connections, a server starting)
    keeps its text: it comes from the server and never quotes what the client sent. a socket error
    is its class and errno. anything else is its class alone: client-side text can quote what was
    sent, the password included.

    :param error: the failure
    :ptype error: BaseException
    :return: ``ClassName: text``, ``ClassName [errno N]`` or ``ClassName``
    :rtype: str
    """
    result = type(error).__name__
    if isinstance(error, asyncpg.exceptions.PostgresError) and str(error):
        result = f"{result}: {error}"
    elif isinstance(error, OSError) and error.errno is not None:
        result = f"{result} [errno {error.errno}]"
    return result


#: the asyncpg function that turns a DSN and its connect arguments into addresses and parameters,
#: and where every error that can quote what was sent is raised: the DSN's own parse
#: (``urllib.parse``, ``int()`` of a port, the query string), the host list, ``sslmode`` and the
#: other connect options. ``asyncpg.connect_utils._parse_connect_arguments`` calls it AFTER its own
#: checks of ``command_timeout`` and the statement-cache sizes, whose messages quote only those
#: values. read from asyncpg 0.31's ``connect_utils.py``; asyncpg's ``ClientConfigurationError`` is
#: raised nowhere else.
_ASYNCPG_PARAMETER_PARSER = ("asyncpg.connect_utils", "_parse_connect_dsn_and_args")


def _is_connection_parameter_error(error: ValueError) -> bool:
    """whether ``error`` is asyncpg refusing the DSN or connect options it was given.

    a :class:`~asyncpg.exceptions.ClientConfigurationError` always is. a plain ``ValueError`` is when
    it was raised inside :data:`_ASYNCPG_PARAMETER_PARSER` -- read off the traceback's frames, a
    public interpreter surface, not asyncpg's. anything else is not: a bad ``command_timeout``
    (checked before the DSN is parsed), a caller hook's own ``ValueError``, an ``OSError`` that is
    also a ``ValueError`` (a server certificate that does not verify). an error from the pool's
    ``init`` or ``setup`` never reaches the connect hook at all.

    :param error: what a connect raised
    :ptype error: ValueError
    :return: ``True`` when its text describes what was sent and must be withheld
    :rtype: bool
    """
    result = False
    if not isinstance(error, OSError):
        result = isinstance(error, asyncpg.exceptions.ClientConfigurationError)
        frame = error.__traceback__
        while frame is not None and not result:
            code = frame.tb_frame.f_code
            module = frame.tb_frame.f_globals.get("__name__")
            result = (module, code.co_name) == _ASYNCPG_PARAMETER_PARSER
            frame = frame.tb_next
    return result


class _ClientSideFailure(Exception):
    """carries a connection-parameter error out of a pool attempt, so the wrapper can re-raise it without its text.

    raised by the attempt's connect hook for what :func:`_is_connection_parameter_error` accepts:
    asyncpg refusing the DSN or connect options it was given, whose text describes -- and with a
    stray ``@`` or ``?`` in a password, quotes -- what the client sent.

    :param error: the client-side error
    :ptype error: ValueError
    """

    def __init__(self, error: ValueError) -> None:
        """wrap ``error``.

        :param error: the client-side error
        :ptype error: ValueError
        """
        super().__init__(type(error).__name__)
        self.error = error


def _withheld_message(target: str | None) -> str:
    """the fixed message a connection-parameter error is re-raised and logged with.

    :param target: ``user@host:port/database`` from the caller's keywords, or ``None``
    :ptype target: str | None
    :return: :data:`_WITHHELD_CLIENT_ERROR` naming ``target`` when there is one
    :rtype: str
    """
    return _WITHHELD_CLIENT_ERROR.format(target=f" for {target}" if target is not None else "")


def _withheld(error: ValueError, target: str | None) -> ValueError:
    """``error``'s type carrying the fixed :data:`_WITHHELD_CLIENT_ERROR` message instead of its text.

    the type is kept, so a caller that catches ``ClientConfigurationError`` or its own
    ``ValueError`` subclass still does. a type that cannot be built from one message falls back to
    its nearest base the wrapper knows -- ``ClientConfigurationError`` for an asyncpg client error,
    ``ValueError`` otherwise -- and the message then names the original type.

    :param error: the client-side error
    :ptype error: ValueError
    :param target: ``user@host:port/database`` from the caller's keywords, or ``None``
    :ptype target: str | None
    :return: the error to raise in its place
    :rtype: ValueError
    """
    message = _withheld_message(target)
    result: ValueError
    try:
        result = type(error)(message)
    except TypeError:
        base: type[ValueError] = ValueError
        if isinstance(error, asyncpg.exceptions.ClientConfigurationError):
            base = asyncpg.exceptions.ClientConfigurationError
        result = base(f"{type(error).__name__}: {message}")
    return result


async def create_pool_with_startup_timeout(
    dsn: str | None = None,
    *,
    pool_name: str = "db",
    startup_timeout: float = DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS,
    connect_timeout: float | None = None,
    connect: Callable[..., Awaitable[asyncpg.Connection]] | None = None,
    **create_pool_kwargs: Any,
) -> asyncpg.Pool:
    """start an asyncpg pool, retrying stalled or refused connects, inside one startup budget.

    ``asyncpg.create_pool`` opens ``min_size`` connections and fails if any one of them fails. this
    wrapper bounds every single connect at ``connect_timeout`` (see
    :func:`resolve_pool_connect_timeout`), and when an attempt fails in a way a later attempt may
    not meet, closes everything that attempt opened and starts a fresh one after a doubling pause
    (:func:`threetears.observe.retry_bounded`). so one backend that never answers costs one
    per-connect timeout, not the budget. no attempt starts after ``startup_timeout``, and the whole
    call is cut off at it.

    every failed attempt is logged at WARNING with its number, the elapsed time and the error
    class. the DSN is redacted with :func:`redact_dsn` everywhere it is reported.

    **client-side errors withhold the library's text; server answers keep theirs.** a server
    answer (a refused login, a missing database, too many connections, a server starting) is
    reported with its text, which comes from the server and never quotes what the client sent. a
    socket error is reported as its class and errno. an error from asyncpg's connection-parameter
    handling (:func:`_is_connection_parameter_error`: a DSN it cannot parse, a connect option it
    refuses) describes what was sent, which with a stray ``@`` or ``?`` in a password is the
    password: it is raised as its own type with a fixed message naming only the ``host`` /
    ``port`` / ``user`` / ``database`` keywords (never a target read from a DSN), with no cause or
    context, and logged as its class and that message. anything else -- a bad ``command_timeout``,
    a caller hook's own ``ValueError``, an error from ``init`` or ``setup``, a bad pool shape --
    keeps its own text and cause.

    a caller with its own connect hook passes it as ``connect``: the wrapper's hook calls it for
    every connection, so the bound, the record and the cleanup apply to it too. it receives
    asyncpg's connect arguments, ``timeout`` (the per-connect bound) among them, and must pass
    them on to ``asyncpg.connect``. an exception it raises that is not one of asyncpg's or the
    network's (a driver's own error type) is neither retried nor wrapped: it propagates, after the
    attempt's connections are closed.

    :param dsn: the PostgreSQL DSN, or ``None`` when ``host`` / ``port`` / ``user`` / ``database``
        are passed as keywords
    :ptype dsn: str | None
    :param pool_name: short identifier for logs and the error (for example ``hub_l3``)
    :ptype pool_name: str
    :param startup_timeout: max wall time in seconds before declaring the database unreachable
    :ptype startup_timeout: float
    :param connect_timeout: bound on one connect in seconds; ``None`` resolves it from the budget
    :ptype connect_timeout: float | None
    :param connect: the caller's own connect hook, run inside the wrapper's; ``None`` for
        ``asyncpg.connect``
    :ptype connect: Callable[..., Awaitable[asyncpg.Connection]] | None
    :param create_pool_kwargs: everything else ``asyncpg.create_pool`` takes (``min_size``,
        ``max_size``, ``server_settings``, ``init``, ``connection_class``, the
        :func:`get_pg_pool_kwargs` splat, ...), except ``timeout``, which the wrapper sets
    :ptype create_pool_kwargs: Any
    :return: the started pool
    :rtype: asyncpg.Pool
    :raises PoolStartupTimeoutError: when the budget is spent without a started pool, or the
        attempt fails in a way no retry can clear (a refusal from the database, a certificate
        that does not verify, a host name that does not exist); it names the attempts made
    :raises asyncpg.exceptions.ClientConfigurationError: a connect option asyncpg refuses, as its own
        type with the fixed message
    :raises ValueError: when the bounds are inconsistent (see :func:`resolve_pool_connect_timeout`);
        a DSN asyncpg cannot parse, as its own type with the fixed message (a type that cannot be
        built from one message is raised as ``ValueError``); or a ``ValueError`` from anywhere else
        (a bad ``command_timeout``, a pool shape, a hook, ``init``), as itself
    :raises TypeError: when ``create_pool_kwargs`` carries ``timeout``
    """
    if "timeout" in create_pool_kwargs:
        raise TypeError(
            "create_pool_with_startup_timeout sets timeout= itself; pass the per-connect bound as connect_timeout="
        )
    per_connect = resolve_pool_connect_timeout(startup_timeout, connect_timeout)
    opener: Callable[..., Awaitable[asyncpg.Connection]] = connect if connect is not None else asyncpg.connect
    identity = _pool_identity(dsn, create_pool_kwargs)
    client_target = _keyword_target(create_pool_kwargs)
    # passed only when given: a caller naming ``host`` / ``port`` / ... starts the pool exactly as
    # its own ``asyncpg.create_pool`` call did.
    dsn_argument: dict[str, str] = {} if dsn is None else {"dsn": dsn}
    progress = _PoolStartProgress()
    started_at = time.monotonic()

    async def attempt() -> asyncpg.Pool:
        progress.attempts += 1
        connections = _AttemptConnections(opener)
        initialising = asyncpg.create_pool(
            **dsn_argument, connect=connections.connect, timeout=per_connect, **create_pool_kwargs
        )
        started = False
        progress.attempt_in_flight = True
        try:
            pool = await initialising
            connections.pool_started()
            started = True
            progress.attempt_in_flight = False
        finally:
            if not started:
                progress.closed_by_last_failure = connections.abandon(initialising)
        return pool

    def on_retry(error: Exception, attempt_number: int, pause: float) -> None:
        progress.attempt_in_flight = False
        progress.last_failure = error
        elapsed = time.monotonic() - started_at
        log.warning(
            f"pg pool start attempt {attempt_number} failed: name={pool_name} identity={identity} "
            f"error={_describe_error(error)} elapsed={elapsed:.2f}s budget={startup_timeout}s "
            f"connect_timeout={per_connect}s closed={progress.closed_by_last_failure}; retrying in {pause:.2f}s",
            extra={
                "extra_data": {
                    "pool_name": pool_name,
                    "connection_identity": identity,
                    "attempt": attempt_number,
                    "error_class": type(error).__name__,
                    "elapsed_seconds": elapsed,
                    "startup_timeout_seconds": startup_timeout,
                    "connect_timeout_seconds": per_connect,
                    "connections_closed": progress.closed_by_last_failure,
                    "retry_in_seconds": pause,
                }
            },
        )

    def give_up(error: Exception, budget_expired: bool) -> tuple[PoolStartupTimeoutError, Exception | None]:
        """the error a start that met a database or network failure ends with, and its cause; logged at ERROR.

        the cause is the failure itself when it is a server answer, a socket error or a timeout, and
        ``None`` otherwise: a client-side error's text, carried as the cause, would reach every
        traceback the wrapper's own message keeps it out of.

        :param error: what ended the start
        :ptype error: Exception
        :param budget_expired: whether the startup budget ran out
        :ptype budget_expired: bool
        :return: the error to raise, and the failure to raise it from (or ``None``)
        :rtype: tuple[PoolStartupTimeoutError, Exception | None]
        """
        elapsed = time.monotonic() - started_at
        cause: Exception = error
        if budget_expired:
            cause = progress.last_failure if progress.last_failure is not None else error
            message = (
                f"failed to connect to database {identity} within {startup_timeout}s: "
                f"{_attempts_phrase(progress.attempts)}, each connect bounded at {per_connect}s; "
                f"the budget ran out {progress.where_the_budget_ran_out()}, last failure: {type(cause).__name__}"
            )
        elif _is_retryable(error):
            message = (
                f"failed to connect to database {identity} within {startup_timeout}s: "
                f"{_attempts_phrase(progress.attempts)}, each connect bounded at {per_connect}s; "
                f"last failure: {_describe_error(error)}"
            )
        else:
            message = (
                f"failed to create database pool {identity} on attempt {progress.attempts} "
                f"(not retried): {_describe_error(error)}"
            )
        log.error(
            f"pg pool start failed: name={pool_name} {message}",
            extra={
                "extra_data": {
                    "pool_name": pool_name,
                    "connection_identity": identity,
                    "attempts": progress.attempts,
                    "error_class": type(cause).__name__,
                    "elapsed_seconds": elapsed,
                    "startup_timeout_seconds": startup_timeout,
                    "connect_timeout_seconds": per_connect,
                }
            },
        )
        failure = PoolStartupTimeoutError(
            message,
            pool_name=pool_name,
            db_identity=identity,
            startup_timeout_seconds=startup_timeout,
            elapsed_seconds=elapsed,
            attempts=progress.attempts,
            connect_timeout_seconds=per_connect,
        )
        chained = cause if isinstance(cause, (asyncpg.exceptions.PostgresError, OSError)) else None
        return failure, chained

    budget = asyncio.timeout(startup_timeout)
    # each failure is raised OUTSIDE the handler that caught it, so the original is the new error's
    # context only where it is chained on purpose (a server answer, a socket error).
    failure: PoolStartupTimeoutError | None = None
    cause: Exception | None = None
    client_error: ValueError | None = None
    try:
        async with budget:
            pool = await retry_bounded(
                attempt,
                retry_on=_is_retryable,
                first_delay=POOL_START_RETRY_FIRST_DELAY_SECONDS,
                max_delay=POOL_START_RETRY_MAX_DELAY_SECONDS,
                deadline_seconds=startup_timeout,
                on_retry=on_retry,
            )
    except _ClientSideFailure as client_side:
        # the caller's mistake, raised as its own type -- with a fixed message, because the
        # library's text describes what was sent and can quote the password.
        client_error = _withheld(client_side.error, client_target)
        log.error(
            f"pg pool start failed: name={pool_name} identity={identity} "
            f"error={type(client_side.error).__name__}: {_withheld_message(client_target)}",
            extra={
                "extra_data": {
                    "pool_name": pool_name,
                    "connection_identity": identity,
                    "attempts": progress.attempts,
                    "error_class": type(client_side.error).__name__,
                    "elapsed_seconds": time.monotonic() - started_at,
                    "startup_timeout_seconds": startup_timeout,
                    "connect_timeout_seconds": per_connect,
                }
            },
        )
    except _DATABASE_FAILURES as exc:
        failure, cause = give_up(exc, budget.expired())
    if client_error is not None:
        raise client_error
    if failure is not None:
        raise failure from cause
    if progress.attempts > 1:
        log.info(
            f"pg pool started on attempt {progress.attempts}: name={pool_name} identity={identity}",
            extra={
                "extra_data": {
                    "pool_name": pool_name,
                    "connection_identity": identity,
                    "attempts": progress.attempts,
                    "elapsed_seconds": time.monotonic() - started_at,
                }
            },
        )
    return pool


__all__ = [
    "DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS",
    "DEFAULT_POOL_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS",
    "ENV_MAX_INACTIVE_LIFETIME",
    "PoolStartupTimeoutError",
    "create_pool_with_startup_timeout",
    "get_pg_pool_kwargs",
    "log_pool_created",
    "redact_dsn",
    "resolve_pool_connect_timeout",
]
