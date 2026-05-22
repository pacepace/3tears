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
operator override via ``FOURTEENAIBOTS_PG_POOL_MAX_INACTIVE_LIFETIME_SECONDS``.

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

call sites that need to override a helper value (for example a fixture
that wants a shorter ``command_timeout``) may pass the kwarg explicitly
AFTER the splat; Python applies later kwargs over earlier ones so the
call site wins::

    pool = await asyncpg.create_pool(
        dsn,
        **get_pg_pool_kwargs(),
        command_timeout=5,  # overrides helper default for this call site
    )

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
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

from threetears.observe import get_logger

log = get_logger(__name__)


#: default startup timeout in seconds for :func:`create_pool_with_startup_timeout`.
#: mirrors :data:`threetears.nats.DEFAULT_STARTUP_TIMEOUT` so both dependencies are
#: bounded by the same wall clock.
DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS: float = 30.0

#: platform default for asyncpg ``max_inactive_connection_lifetime``.
#:
#: matches asyncpg's own default (300s). the point of pinning this is to
#: make every call site explicit rather than to pick a new value.
DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS: float = 300.0

#: env var operators use to tune ``max_inactive_connection_lifetime``
#: without a code change. verbose on purpose -- it is an operator tuning
#: knob, not a dev convenience.
ENV_MAX_INACTIVE_LIFETIME: str = (
    "FOURTEENAIBOTS_PG_POOL_MAX_INACTIVE_LIFETIME_SECONDS"
)


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
                f"invalid {ENV_MAX_INACTIVE_LIFETIME}={raw!r}, "
                f"using default {DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS}s",
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
    fail because logging cannot parse the DSN.

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
    if parts is not None:
        host = parts.hostname
        if host:
            user = parts.username or ""
            port = parts.port
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
    """raised when ``create_pool_with_startup_timeout`` exceeds its budget.

    carries structured context (``pool_name``, ``db_identity``,
    ``startup_timeout_seconds``, ``elapsed_seconds``) so callers can
    translate to their own structured-error type (Hub maps to
    :class:`aibots.hub.common.errors.ConfigurationError`; other consumers
    map to whatever their stack uses).

    keeping this exception type in 3tears -- not in Hub -- breaks the
    Hub-only coupling the original helper had on
    :class:`ConfigurationError`. concrete error-translation belongs at
    the consumer boundary.

    :param message: human-readable description
    :ptype message: str
    :param pool_name: short identifier for the pool that failed to start
    :ptype pool_name: str
    :param db_identity: redacted db identity (no credentials)
    :ptype db_identity: str
    :param startup_timeout_seconds: budget the call exceeded
    :ptype startup_timeout_seconds: float
    :param elapsed_seconds: wall-clock seconds elapsed before the failure
    :ptype elapsed_seconds: float
    """

    def __init__(
        self,
        message: str,
        *,
        pool_name: str,
        db_identity: str,
        startup_timeout_seconds: float,
        elapsed_seconds: float,
    ) -> None:
        super().__init__(message)
        self.pool_name = pool_name
        self.db_identity = db_identity
        self.startup_timeout_seconds = startup_timeout_seconds
        self.elapsed_seconds = elapsed_seconds


async def create_pool_with_startup_timeout[PoolT](
    create: Callable[[], Awaitable[PoolT]],
    *,
    dsn: str,
    startup_timeout: float = DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS,
    pool_name: str = "db",
) -> PoolT:
    """invoke asyncpg pool factory bounded by an overall startup timeout.

    wraps the caller-supplied ``asyncpg.create_pool(...)`` coroutine in
    :func:`asyncio.wait_for` so an unreachable database produces a
    structured :class:`PoolStartupTimeoutError` inside ``startup_timeout``
    rather than blocking indefinitely on socket-level retries. the DSN
    passed for logging is redacted with :func:`redact_dsn`; raw
    credentials are never emitted.

    :param create: zero-arg callable returning the ``asyncpg.create_pool`` awaitable
    :ptype create: Callable[[], Awaitable[PoolT]]
    :param dsn: DSN the pool is being created from, used for error context (redacted)
    :ptype dsn: str
    :param startup_timeout: max wall time in seconds before declaring the dependency unreachable
    :ptype startup_timeout: float
    :param pool_name: short identifier included in the error context (for example ``hub_l3``)
    :ptype pool_name: str
    :return: the created pool object returned by ``create``
    :rtype: PoolT
    :raises PoolStartupTimeoutError: if pool creation exceeds ``startup_timeout`` or raises a transport error
    """
    identity = redact_dsn(dsn)
    started_at = time.monotonic()
    try:
        pool = await asyncio.wait_for(create(), timeout=startup_timeout)
    except TimeoutError:
        elapsed = time.monotonic() - started_at
        raise PoolStartupTimeoutError(
            f"failed to connect to database {identity} within {startup_timeout}s",
            pool_name=pool_name,
            db_identity=identity,
            startup_timeout_seconds=startup_timeout,
            elapsed_seconds=elapsed,
        ) from None
    except Exception as exc:
        elapsed = time.monotonic() - started_at
        raise PoolStartupTimeoutError(
            f"failed to create database pool {identity}: {exc}",
            pool_name=pool_name,
            db_identity=identity,
            startup_timeout_seconds=startup_timeout,
            elapsed_seconds=elapsed,
        ) from exc
    return pool


__all__ = [
    "DEFAULT_MAX_INACTIVE_LIFETIME_SECONDS",
    "DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS",
    "ENV_MAX_INACTIVE_LIFETIME",
    "PoolStartupTimeoutError",
    "create_pool_with_startup_timeout",
    "get_pg_pool_kwargs",
    "log_pool_created",
    "redact_dsn",
]
