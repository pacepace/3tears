"""asyncpg-backed concrete :class:`Driver` for postgres / yugabyte / agent_internal.

implements the full :class:`threetears.datasources.drivers.base.Driver`
ABC against ``asyncpg.Pool``. covers three backends that all speak
pgwire:

- :data:`DataSourceType.POSTGRES` -- standard PostgreSQL targets
- :data:`DataSourceType.YUGABYTE` -- YugabyteDB (same asyncpg driver,
  port 5433 by default)
- :data:`DataSourceType.AGENT_INTERNAL` -- agent-created tables; pool
  is BORROWED from Hub's L3 pool (constructor ``external_pool=``)
  rather than freshly opened. close() is a no-op for the borrowed-pool
  path -- Hub's L3 lifecycle owns the asyncpg.Pool.

design contract (datasource-task-10):

- DS-10-01: every abstract method is implemented; :meth:`fetch_iter`
  is overridden with a server-side cursor (the ABC default
  materializes via :meth:`fetch` -- defeats streaming for million-row
  results).
- DS-10-02: pool sizing reads from the
  :class:`PostgresConnectionConfig` / :class:`YugabyteConnectionConfig`
  fields (``pool_min_size``, ``pool_max_size``, ``command_timeout_seconds``).
  no inline literals. the enforcement test
  ``tests/enforcement/test_no_hardcoded_pool_params.py`` walks this
  module and fails the build on banned-kwarg literals.
- DS-10-04: callers pass ``$1``-style placeholders; asyncpg uses ``$N``
  natively. :func:`_translate_placeholders` is called with target
  ``"asyncpg"`` (no-op) anyway -- the consistent surface across
  drivers makes the contract enforceable.
- DS-10-08: cancellation uses :meth:`asyncpg.Connection.cancel`,
  NEVER :meth:`asyncpg.Connection.terminate`. ``cancel()`` sends a
  backend cancel request and returns the connection to the pool
  clean; ``terminate()`` force-closes the socket and EVICTS the
  connection from the pool. ``terminate()`` belongs only in
  :meth:`close` (forced teardown).
- DS-10-10: passwords resolve via :meth:`config.resolve_password`
  returning :class:`pydantic.SecretStr`; ``.get_secret_value()`` is
  unwrapped at the LAST moment when handing to
  :func:`asyncpg.create_pool`. no intermediate ``str`` variable.
- DS-10-11: backend exceptions are wrapped in
  :class:`DriverConnectError` / :class:`DriverQueryError` with
  ``from None`` to break the cause chain. raw asyncpg errors
  sometimes carry the password value in nested context; sanitizing
  here keeps logs / tracebacks clean.
- DS-10-12: :func:`_observed` decorator on every query-emitting
  method emits :data:`datasource.driver.query.duration` histogram
  + :data:`datasource.driver.error` counter automatically. the
  manual :data:`datasource.driver.cancellation.fired` counter is
  bumped from the wrapped cancel callback inside
  :meth:`_acquire_and_run`.

close concurrency (DS-09-12 / DS-10-07):

- :meth:`close` is idempotent. second call is a no-op.
- after :meth:`close`, every method raises :class:`RuntimeError`.
- for the borrowed-pool path (``AGENT_INTERNAL``), :meth:`close`
  marks the driver closed but does NOT call ``pool.close()`` -- the
  L3 lifecycle owns the pool. external (non-borrowed) drivers call
  ``await self._pool.close()`` exactly once.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import asyncpg

from threetears.core.utils.pg_pool_kwargs import get_pg_pool_kwargs
from threetears.datasources.config import (
    AgentInternalConnectionConfig,
    PostgresConnectionConfig,
    YugabyteConnectionConfig,
)
from threetears.datasources.drivers._util import (
    _translate_placeholders,
    build_search_path_value,
)
from threetears.datasources.drivers.base import (
    ColumnRow,
    Driver,
    TableRow,
    _check_otel_metrics,
    _instrument_cache,
    _observed,
)
from threetears.observe import get_logger, traced

__all__ = [
    "AsyncpgDriver",
    "DriverCancellationError",
    "DriverConnectError",
    "DriverQueryError",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# SQL constants (DS-10-05)
# ---------------------------------------------------------------------------


#: list tables visible inside the schema allow-list.
#:
#: migrated verbatim from Hub's ``aibots.hub.datasources.schema_introspector``
#: so a future shard-13 rewire flips the introspector to call this driver
#: rather than maintain two copies of the SQL.
_POSTGRES_TABLES_SQL = """
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema = ANY($1)
AND table_type = 'BASE TABLE'
ORDER BY table_schema, table_name
""".strip()


#: list columns for every table in the schema allow-list.
#:
#: ``is_nullable`` is the raw warehouse string (``'YES'`` / ``'NO'``);
#: the :class:`ColumnRow` TypedDict pins it that way so the Tier-2 hash
#: stays byte-equivalent with the warehouse-side MD5 (see
#: :data:`_POSTGRES_TABLE_HASHES_SQL`).
_POSTGRES_COLUMNS_SQL = """
SELECT table_schema, table_name, column_name, data_type, is_nullable, ordinal_position
FROM information_schema.columns
WHERE table_schema = ANY($1)
ORDER BY table_schema, table_name, ordinal_position
""".strip()


#: per-table MD5 over the column shape (Tier-2 change-probe).
#:
#: the formula ``column_name || ':' || data_type || ':' ||
#: COALESCE(is_nullable, '')`` is byte-for-byte the same payload as
#: the python-side helper in ``datasource-task-02`` (see test
#: ``test_table_hash_python_byte_equivalence`` for the cross-language
#: invariant). swapping the COALESCE for an alternative null handling
#: silently breaks the Tier-2 probe -- DO NOT change without updating
#: the python helper in lockstep.
_POSTGRES_TABLE_HASHES_SQL = """
SELECT table_schema, table_name,
       MD5(STRING_AGG(column_name || ':' || data_type || ':' || COALESCE(is_nullable, ''), ',' ORDER BY ordinal_position)) AS column_hash
FROM information_schema.columns
WHERE table_schema = ANY($1)
GROUP BY table_schema, table_name
ORDER BY table_schema, table_name
""".strip()


#: cheapest possible round-trip for :meth:`AsyncpgDriver.test_connection`.
_PING_SQL = "SELECT 1"


# ---------------------------------------------------------------------------
# Exception types (DS-10-11)
# ---------------------------------------------------------------------------


class DriverConnectError(Exception):
    """raised when connect / auth fails.

    the message intentionally carries the host / port / database
    identifiers (safe to log) but NEVER the resolved password value.
    callers should raise the wrapper with ``from None`` to break the
    cause chain -- the original asyncpg exception sometimes embeds the
    password in nested context, which would defeat the sanitization.

    :param message: human-readable description; MUST NOT carry the
        password value or any other resolved secret
    :ptype message: str
    """


class DriverQueryError(Exception):
    """raised when a query fails for non-cancellation reasons.

    cancellation propagates via :class:`asyncio.CancelledError`
    (subclassed by :class:`DriverCancellationError`); all other
    backend failures are wrapped in this type. the message MUST NOT
    carry credentials -- if a future contributor wants to include
    SQL in the message, scrub bind-parameter values first.
    """


class DriverCancellationError(asyncio.CancelledError):
    """asyncpg-specific cancellation marker.

    subclass of :class:`asyncio.CancelledError` so existing
    ``except asyncio.CancelledError`` handlers still catch it. lets
    callers that want to distinguish driver-initiated cancellation
    from generic asyncio cancellation do so via ``isinstance(exc,
    DriverCancellationError)`` without breaking the propagation
    contract.
    """


# pgwire-config carriers: postgres + yugabyte share the same shape, so
# the driver accepts either. agent-internal uses a different shape
# entirely (no host/port/credentials) and is handled separately.
_PgConfig = PostgresConnectionConfig | YugabyteConnectionConfig
_AnyConfig = _PgConfig | AgentInternalConnectionConfig


# ---------------------------------------------------------------------------
# Per-driver-type cancellation-fired counter (DS-10-12)
# ---------------------------------------------------------------------------


def _get_cancellation_fired_counter() -> Any:
    """fetch or create the ``datasource.driver.cancellation.fired`` counter.

    backend-specific counter (not auto-emitted by :func:`_observed`).
    bumped from the wrapped cancel callback so we observe ONLY the
    real fires -- the helper's outer try/except may catch a
    :class:`CancelledError` that fires before the backend call
    actually started; we don't want those bumping the counter.

    :return: OTel Counter instrument (or None if OTel not available)
    :rtype: Any
    """
    result: Any = None
    if _check_otel_metrics():
        key = ("asyncpg", "datasource.driver.cancellation.fired")
        instrument = _instrument_cache.get(key)
        if instrument is None:
            from opentelemetry import metrics

            meter = metrics.get_meter("threetears.datasources.drivers")
            instrument = meter.create_counter(
                name="datasource.driver.cancellation.fired",
                description="datasource driver cancellation fired count",
            )
            _instrument_cache[key] = instrument
        result = instrument
    return result


# ---------------------------------------------------------------------------
# AsyncpgDriver
# ---------------------------------------------------------------------------


class AsyncpgDriver(Driver):
    """concrete :class:`Driver` backed by ``asyncpg.Pool``.

    construct via :func:`threetears.datasources.drivers.create_driver`
    rather than directly -- the factory wires the borrowed-pool kwarg
    for the agent-internal case and enforces the lazy-import contract.

    pool lifecycle:

    - ``__init__`` stores the config + optional ``external_pool``;
      does NOT create a pool eagerly. lazy creation lets the test
      suite construct + immediately :meth:`close` drivers without a
      backend round-trip.
    - first call to any query-emitting method calls
      :meth:`_ensure_pool` which creates the asyncpg pool (and the
      driver owns it). when ``external_pool`` is set, ``_ensure_pool``
      is a no-op and ``self._pool = self._external_pool``.
    - :meth:`close` calls ``self._pool.close()`` only when
      ``self._owns_pool`` is True (the non-borrowed path).

    :param config: per-driver connection config; postgres / yugabyte
        carry connection identity + pool sizing; agent-internal
        carries only the schema name (the pool is borrowed)
    :ptype config: PostgresConnectionConfig | YugabyteConnectionConfig | AgentInternalConnectionConfig
    :param external_pool: pre-existing :class:`asyncpg.Pool` to borrow.
        ONLY supplied for the AGENT_INTERNAL branch by the factory;
        external (postgres / yugabyte) callers pass None and the
        driver creates its own pool on first use
    :ptype external_pool: asyncpg.Pool | None
    :param datasource_name: human-readable name of the datasource this
        driver instance serves (``DatasourceConfig.name`` from
        agent.yaml or :class:`DataSourceEntity.name` from the Hub
        admin row). surfaced as the ``datasource_name`` attribute on
        every OTel metric emitted by :func:`_observed`. defaults to
        ``"unknown"`` when callers can't supply one; the Hub broker /
        tool-pod / introspector (shards 13-14) thread the name through
    :ptype datasource_name: str
    """

    def __init__(
        self,
        config: _AnyConfig,
        *,
        external_pool: asyncpg.Pool[Any] | None = None,
        datasource_name: str = "unknown",
    ) -> None:
        """capture config + optional borrowed pool. no I/O.

        :param config: per-driver connection config
        :ptype config: PostgresConnectionConfig | YugabyteConnectionConfig | AgentInternalConnectionConfig
        :param external_pool: pre-existing pool to borrow (agent-internal)
        :ptype external_pool: asyncpg.Pool | None
        :param datasource_name: name of the datasource this driver serves;
            surfaces on every OTel metric
        :ptype datasource_name: str
        :return: nothing
        :rtype: None
        """
        self._config = config
        self._external_pool = external_pool
        # the pool is None until first use OR a borrowed pool is
        # supplied; ``_owns_pool`` distinguishes the lifecycle paths
        # so :meth:`close` knows whether to call ``pool.close()``.
        self._pool: asyncpg.Pool[Any] | None = external_pool
        self._owns_pool = external_pool is None
        self._closed = False
        # read by :func:`_observed` as the ``datasource_name`` attribute
        # on every emitted metric. the Hub-side caller (shards 13/14)
        # passes the name through ``create_driver``; tests pass an
        # explicit value or accept the ``"unknown"`` default.
        self._datasource_name = datasource_name

    # -------------------------------------------------------------------
    # pool lifecycle helpers
    # -------------------------------------------------------------------

    async def _ensure_pool(self) -> asyncpg.Pool[Any]:
        """lazily create the asyncpg pool on first use; reject calls after close.

        borrowed-pool path (agent-internal): ``self._pool`` is already
        set in ``__init__``; this method is a no-op except for the
        closed-check.

        owned-pool path (postgres / yugabyte): the first call creates
        the asyncpg pool sized from the config's documented defaults
        via :func:`threetears.core.utils.pg_pool_kwargs.get_pg_pool_kwargs`.

        :return: the live :class:`asyncpg.Pool` (owned or borrowed)
        :rtype: asyncpg.Pool
        :raises RuntimeError: if the driver was previously closed
        :raises DriverConnectError: if pool creation fails (auth /
            network / DNS); message carries host/port/database but
            never the password value
        """
        if self._closed:
            raise RuntimeError("AsyncpgDriver is closed")
        pool = self._pool
        if pool is None:
            pool = await self._create_owned_pool()
            self._pool = pool
        return pool

    async def _create_owned_pool(self) -> asyncpg.Pool[Any]:
        """build a new :class:`asyncpg.Pool` from the config.

        passes ``min_size``, ``max_size``, ``command_timeout`` read
        from the config (NOT inline literals; see DS-10-02). splats
        :func:`get_pg_pool_kwargs` for the platform-default
        ``max_inactive_connection_lifetime`` (carries the Yugabyte
        pgwire stale-connection guard).

        the password is resolved through
        :meth:`config.resolve_password` which returns
        :class:`SecretStr`; ``.get_secret_value()`` is unwrapped at
        the LAST moment when handed to ``asyncpg.create_pool``. no
        intermediate ``str`` variable holds the value -- the whole
        point of :class:`SecretStr` is the value never lives in a
        plain string that could leak via logging.

        :return: live owned pool
        :rtype: asyncpg.Pool
        :raises DriverConnectError: on auth / network / DNS failure;
            the wrapper carries host/port/database (safe to log) but
            never the password
        """
        # agent-internal MUST not reach here -- the factory passes
        # external_pool= for that case. defending against a future
        # caller that constructs the driver directly with mismatched
        # args.
        if isinstance(self._config, AgentInternalConnectionConfig):
            raise DriverConnectError(
                "AsyncpgDriver: AGENT_INTERNAL config requires external_pool="
                " (Hub's L3 pool); cannot open a fresh pool from agent_internal"
            )
        cfg: _PgConfig = self._config
        # SecretStr round-trip: resolve only if password_ref is set;
        # local dev / trust-auth setups legitimately have no password.
        # ``.get_secret_value()`` is called inside the ``create_pool``
        # call site to keep the value off any intermediate variable
        # (see DS-10-10).
        #
        # search_path: when ``allowed_schemas`` is non-empty, pass the
        # value through asyncpg's ``server_settings`` connect kwarg.
        # this sends ``search_path=...`` in the pgwire STARTUP packet
        # so the value is the connection's "session default" -- which
        # means ``DISCARD ALL`` / ``RESET ALL`` (asyncpg's default
        # pool-release reset) leaves it intact instead of wiping it
        # back to ``"$user", public``. an ``init=`` cursor.execute
        # would lose the value between acquires; a ``setup=`` would
        # add a round trip on every acquire. STARTUP is the only path
        # that's both correct and zero-cost.
        search_path_value = build_search_path_value(cfg.allowed_schemas)
        server_settings = {"search_path": search_path_value} if search_path_value is not None else None
        try:
            # ``server_settings`` is accepted by asyncpg.connect (and
            # forwarded by create_pool via **connect_kwargs). only set
            # it when non-empty to avoid sending an empty dict that
            # would still trigger the startup-parameter code path.
            connect_kwargs: dict[str, Any] = {}
            if server_settings is not None:
                connect_kwargs["server_settings"] = server_settings
            pool = await asyncpg.create_pool(
                host=cfg.host,
                port=cfg.port,
                database=cfg.database,
                user=cfg.username,
                password=(cfg.resolve_password().get_secret_value() if cfg.password_ref is not None else None),
                min_size=cfg.pool_min_size,
                max_size=cfg.pool_max_size,
                command_timeout=cfg.command_timeout_seconds,
                **connect_kwargs,
                **get_pg_pool_kwargs(),
            )
        except Exception:
            # break the cause chain (``from None``) so the original
            # asyncpg error -- which sometimes embeds the password
            # value in nested context -- does NOT reach loggers /
            # tracebacks via ``__cause__``. the wrapper's message is
            # the only thing callers see.
            raise DriverConnectError(f"connection failed for {cfg.host}:{cfg.port}/{cfg.database}") from None
        # asyncpg.create_pool can return None on edge cases; guard
        # against the typing.
        if pool is None:
            raise DriverConnectError(f"connection returned no pool for {cfg.host}:{cfg.port}/{cfg.database}")
        return pool

    async def _acquire_and_run(
        self,
        coro_fn: Callable[[asyncpg.Connection[Any]], Awaitable[Any]],
    ) -> Any:
        """acquire a connection from the pool + route the call through cancellation.

        canonical wrapper every query-emitting method routes through.
        the connection is acquired BEFORE the cancellation helper so
        ``conn.cancel`` is wired as the cancel callback; this is the
        whole reason :meth:`Driver._with_cancellation` takes a
        callable rather than the awaitable itself.

        the wrapped cancel callback also bumps the
        :data:`datasource.driver.cancellation.fired` counter (manual
        emission per DS-10-12 -- :func:`_observed` doesn't bump it
        for us because the per-driver semantics differ; we only count
        the fires that actually reach the backend).

        :param coro_fn: callable that takes the acquired :class:`asyncpg.Connection`
            and returns the awaitable to run
        :ptype coro_fn: Callable[[asyncpg.Connection], Awaitable[T]]
        :return: whatever ``coro_fn(conn)`` resolved to
        :rtype: T
        :raises asyncio.CancelledError: propagated after best-effort
            backend cancellation via :meth:`asyncpg.Connection.cancel`
        :raises RuntimeError: if the driver was previously closed
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            cancel_counter = _get_cancellation_fired_counter()

            def _on_cancel() -> Any:
                # bump the per-driver-type cancellation counter (DS-10-12)
                # BEFORE forwarding to asyncpg's cancel hook. wrapping
                # this way means the counter only ticks when the
                # cancel callback actually fires (i.e. the awaiting
                # coroutine was cancelled while the backend call was
                # in flight), not on every method invocation.
                if cancel_counter is not None:
                    cancel_counter.add(1, attributes={"driver_type": "asyncpg"})
                return conn.cancel()  # NOT terminate() -- see DS-10-08

            return await self._with_cancellation(
                lambda: coro_fn(conn),
                cancel_callback=_on_cancel,
            )

    # -------------------------------------------------------------------
    # Driver ABC: query surface
    # -------------------------------------------------------------------

    @traced
    @_observed(driver_type="asyncpg")
    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """run a SELECT statement; materialize all rows in memory.

        :param sql: SQL text with ``$1``-style placeholders
        :ptype sql: str
        :param params: positional placeholder values
        :ptype params: Any
        :return: list of column-name -> value dicts in row order
        :rtype: list[dict[str, Any]]
        :raises asyncio.CancelledError: propagated after backend cancel
        :raises RuntimeError: if the driver was previously closed
        :raises DriverConnectError: if the lazy pool creation fails
        """
        if self._closed:
            raise RuntimeError("AsyncpgDriver is closed")
        # placeholder translation is a no-op for asyncpg (it uses $N
        # natively); calling the helper anyway keeps the contract
        # consistent across drivers.
        translated = _translate_placeholders(sql, "asyncpg")
        records = await self._acquire_and_run(
            lambda conn: conn.fetch(translated, *params),
        )
        result: list[dict[str, Any]] = [dict(r) for r in records]
        return result

    @traced
    @_observed(driver_type="asyncpg")
    async def execute(self, sql: str, *params: Any) -> None:
        """run a DML / DDL statement; discard any returned rows.

        :param sql: SQL text with ``$1``-style placeholders
        :ptype sql: str
        :param params: positional placeholder values
        :ptype params: Any
        :return: nothing
        :rtype: None
        :raises asyncio.CancelledError: propagated after backend cancel
        :raises RuntimeError: if the driver was previously closed
        """
        if self._closed:
            raise RuntimeError("AsyncpgDriver is closed")
        translated = _translate_placeholders(sql, "asyncpg")
        await self._acquire_and_run(
            lambda conn: conn.execute(translated, *params),
        )

    @traced
    async def fetch_iter(self, sql: str, *params: Any) -> AsyncIterator[dict[str, Any]]:
        """stream rows via asyncpg's server-side cursor.

        overrides the ABC default (which materializes via :meth:`fetch`
        then yields). server-side cursors require a transaction
        wrapper in asyncpg, hence the ``async with conn.transaction()``
        block.

        cancellation propagates naturally through the generator: if
        the caller cancels the outer task, the ``async for`` raises
        :class:`asyncio.CancelledError` inside this generator, the
        ``async with`` blocks unwind, and asyncpg closes the cursor
        + releases the connection back to the pool.

        :param sql: SQL text with ``$1``-style placeholders
        :ptype sql: str
        :param params: positional placeholder values
        :ptype params: Any
        :return: async iterator over column-name -> value dicts
        :rtype: AsyncIterator[dict[str, Any]]
        :raises RuntimeError: if the driver was previously closed
        """
        if self._closed:
            raise RuntimeError("AsyncpgDriver is closed")
        translated = _translate_placeholders(sql, "asyncpg")
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # ``Connection.cursor`` returns a server-side cursor;
                # this is the WHOLE reason we override the ABC default.
                async for record in conn.cursor(translated, *params):
                    yield dict(record)

    # -------------------------------------------------------------------
    # Driver ABC: introspection surface
    # -------------------------------------------------------------------

    @traced
    @_observed(driver_type="asyncpg")
    async def list_tables(self, schemas: list[str]) -> list[TableRow]:
        """list tables in the schema allow-list using postgres-flavored SQL.

        :param schemas: schema-name allow-list; empty list returns no rows
        :ptype schemas: list[str]
        :return: :class:`TableRow` dicts
        :rtype: list[TableRow]
        :raises RuntimeError: if the driver was previously closed
        """
        if self._closed:
            raise RuntimeError("AsyncpgDriver is closed")
        records = await self._acquire_and_run(
            lambda conn: conn.fetch(_POSTGRES_TABLES_SQL, schemas),
        )
        result: list[TableRow] = [
            TableRow(
                table_schema=r["table_schema"],
                table_name=r["table_name"],
            )
            for r in records
        ]
        return result

    @traced
    @_observed(driver_type="asyncpg")
    async def list_columns(self, schemas: list[str]) -> list[ColumnRow]:
        """list columns for every table in the schema allow-list.

        ``is_nullable`` is preserved as the raw warehouse string
        (``'YES'`` / ``'NO'``) -- never normalized to bool. the
        Tier-2 hash depends on byte-equality with the warehouse-side
        MD5 (see :data:`_POSTGRES_TABLE_HASHES_SQL`).

        :param schemas: schema-name allow-list
        :ptype schemas: list[str]
        :return: :class:`ColumnRow` dicts
        :rtype: list[ColumnRow]
        :raises RuntimeError: if the driver was previously closed
        """
        if self._closed:
            raise RuntimeError("AsyncpgDriver is closed")
        records = await self._acquire_and_run(
            lambda conn: conn.fetch(_POSTGRES_COLUMNS_SQL, schemas),
        )
        result: list[ColumnRow] = [
            ColumnRow(
                table_schema=r["table_schema"],
                table_name=r["table_name"],
                column_name=r["column_name"],
                data_type=r["data_type"],
                is_nullable=r["is_nullable"],
                ordinal_position=r["ordinal_position"],
            )
            for r in records
        ]
        return result

    @traced
    @_observed(driver_type="asyncpg")
    async def table_hashes(self, schemas: list[str]) -> dict[tuple[str, str], str]:
        """compute per-table MD5 over the column shape (Tier-2 change-probe).

        the warehouse-side MD5 formula in :data:`_POSTGRES_TABLE_HASHES_SQL`
        is byte-equivalent to the python-side ``_compute_column_hash``
        helper specified in ``datasource-task-02``. equality is the
        cross-language invariant that makes the Tier-2 probe work --
        see ``tests/integration/test_asyncpg_driver_live.py`` for the
        cross-check.

        :param schemas: schema-name allow-list
        :ptype schemas: list[str]
        :return: mapping of ``(schema, table)`` -> column-shape hex digest
        :rtype: dict[tuple[str, str], str]
        :raises RuntimeError: if the driver was previously closed
        """
        if self._closed:
            raise RuntimeError("AsyncpgDriver is closed")
        records = await self._acquire_and_run(
            lambda conn: conn.fetch(_POSTGRES_TABLE_HASHES_SQL, schemas),
        )
        result: dict[tuple[str, str], str] = {(r["table_schema"], r["table_name"]): r["column_hash"] for r in records}
        return result

    # -------------------------------------------------------------------
    # Driver ABC: lifecycle
    # -------------------------------------------------------------------

    @traced
    @_observed(driver_type="asyncpg")
    async def test_connection(self) -> None:
        """cheapest possible round-trip; verifies credentials + reachability.

        wrapped in :class:`DriverConnectError` on any failure so the
        original asyncpg exception (which sometimes carries the
        password value in nested context) does not reach the caller.

        note: ``except asyncio.CancelledError`` is intentionally absent
        here. :class:`CancelledError` is rooted at :class:`BaseException`
        (Python 3.8+), so the ``except Exception`` below does not catch
        it -- cancellation propagates unchanged, by design.

        :return: nothing; raises on failure
        :rtype: None
        :raises RuntimeError: if the driver was previously closed
        :raises DriverConnectError: on any backend failure (auth /
            network / timeout); message carries host/port/database
            but never the password value
        """
        if self._closed:
            raise RuntimeError("AsyncpgDriver is closed")
        try:
            await self._acquire_and_run(
                lambda conn: conn.fetchval(_PING_SQL),
            )
        except Exception:
            # sanitize: the wrapper carries the connection identity
            # (safe to log) but never the password. ``from None``
            # breaks the cause chain so the original asyncpg
            # exception's message can't surface via ``__cause__``.
            identity = self._connection_identity()
            raise DriverConnectError(f"connection failed for {identity}") from None

    @traced
    async def close(self) -> None:
        """release driver resources; idempotent.

        sets :attr:`_closed` first so any in-flight method that
        re-enters during close raises :class:`RuntimeError`. for the
        owned-pool path, awaits ``self._pool.close()`` once. for the
        borrowed-pool path (agent-internal), the pool is NOT closed
        -- Hub's L3 lifecycle owns it.

        idempotent: a second call returns immediately without
        re-closing the pool.

        :return: nothing
        :rtype: None
        """
        if self._closed:
            return
        self._closed = True
        if self._owns_pool and self._pool is not None:
            await self._pool.close()

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _connection_identity(self) -> str:
        """credential-free identity string for error messages.

        ``host:port/database`` for the postgres / yugabyte variants;
        ``agent_internal://<schema>`` for the borrowed-pool path
        (no host/port to surface). NEVER includes the password.

        :return: safe-to-log identity string
        :rtype: str
        """
        if isinstance(self._config, AgentInternalConnectionConfig):
            return f"agent_internal://{self._config.schema_name}"
        cfg: _PgConfig = self._config
        return f"{cfg.host}:{cfg.port}/{cfg.database}"


# log a module-level marker so operators can confirm from the log
# alone which driver module is loaded. DEBUG so production logging
# configurations stay quiet by default.
log.debug("threetears.datasources.drivers.asyncpg_driver module loaded")
