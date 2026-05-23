"""Snowflake driver stub: contract reference + implementation roadmap.

ships as part of ``datasource-task-12`` to prove the :class:`Driver`
ABC fits a stateful-pooled DB-API backend without requiring the
backend library to be installed. every abstract method raises
:class:`NotImplementedError` with an actionable message; the
implementer reads this module's docstring + the shard doc and knows
exactly what to build toward.

contract reference -- DO NOT REINVENT these helpers in the future
implementation:

Backend library
    `snowflake-connector-python <https://pypi.org/project/snowflake-connector-python/>`_
    (Apache-2.0, Snowflake-maintained). minimum version 3.0. installed
    via the ``[snowflake]`` extras key on this package; the factory
    lazy-imports this module only when a
    :class:`SnowflakeConnectionConfig` actually dispatches, so
    Hub/agent consumers that never use Snowflake never pay the
    import cost.

Connection lifecycle
    stateful + pooled. ``snowflake.connector.connect(account=..,
    warehouse=.., user=.., password=..)`` returns a
    :class:`SnowflakeConnection`; the driver keeps a small pool of
    warm connections (sized by
    :attr:`SnowflakeConnectionConfig.pool_size`). Snowflake auth is
    expensive (~1-2s) so amortizing it across queries is the key
    perf concern -- matches the :class:`RedshiftDriver` shape more
    than the :class:`AsyncpgDriver` shape.

Placeholder style
    DB-API ``%s`` positional placeholders (pyformat). callers pass
    ``$N``-style placeholders per the ABC contract; the driver
    translates via
    :func:`threetears.datasources.drivers._util._translate_placeholders`
    with target ``"pyformat"`` -- the SAME helper the
    :class:`RedshiftDriver` already calls. DO NOT reimplement the
    regex dance.

Cancellation mechanism
    :meth:`SnowflakeConnection.cancel_query` after capturing the
    query id via ``cursor.sfqid``. wire the cancel through the
    shared :meth:`Driver._with_cancellation` helper (NOT a per-
    method try/except block; the shared helper exists so the
    cancel-propagation contract lives in exactly one place).

Sync-to-async bridge
    ``snowflake.connector`` is blocking DB-API; route every call
    through :class:`threetears.datasources.drivers._sync_bridge.AsyncSyncBridge`
    (the SAME bridge :class:`RedshiftDriver` already uses). size from
    :attr:`SnowflakeConnectionConfig.pool_size`-ish or a separate
    ``executor_max_workers`` field if added in shard 08 follow-up.
    NEVER instantiate :class:`concurrent.futures.ThreadPoolExecutor`
    directly -- the enforcement test catches it at compile time.

Row-shape pinning
    :meth:`list_tables` returns :class:`threetears.datasources.drivers.base.TableRow`;
    :meth:`list_columns` returns
    :class:`threetears.datasources.drivers.base.ColumnRow`. the
    ``is_nullable`` field stays the RAW warehouse string
    (``'YES'``/``'NO'``/``''``), NEVER a bool -- Snowflake's
    ``information_schema.columns`` returns the same shape as
    postgres so the byte-equivalence with the python-side
    ``_compute_column_hash`` from ``datasource-task-02`` holds
    without translation.

information_schema source
    Snowflake exposes a postgres-compatible
    ``information_schema.tables`` / ``information_schema.columns``
    that supports aggregates. unlike Redshift, you can use
    ``MD5(LISTAGG(...))`` (Snowflake variant) over
    ``information_schema.columns`` directly. the same SQL template
    shape used by :class:`AsyncpgDriver` adapts cleanly; mind that
    Snowflake reserves the ``$$`` token, so prefer single-quoted
    strings in the SQL constants.

Pool / executor / timeout knobs
    every knob reads from :class:`SnowflakeConnectionConfig`. the
    enforcement test
    ``tests/enforcement/test_no_hardcoded_pool_params.py`` walks
    every concrete driver module on every test run and fails the
    build on banned-kwarg literals (``min_size``, ``max_size``,
    ``timeout``, ``command_timeout``, ``connect_timeout``,
    ``query_timeout``, ``pool_size``, ``cache_size``,
    ``connection_cache_size``, ``executor_max_workers``,
    ``max_workers``). add a new field to
    :class:`SnowflakeConnectionConfig` if a knob needs more
    documentation than a default value provides.

Secret handling
    :meth:`SnowflakeConnectionConfig.resolve_password` returns
    :class:`pydantic.SecretStr`; unwrap via ``.get_secret_value()``
    at the LAST moment inside the ``snowflake.connector.connect(
    password=...)`` call. NEVER an intermediate ``str`` variable.
    wrap backend errors in :class:`DriverConnectError` with
    ``from None`` to break the cause chain so the resolved password
    can't surface via ``__cause__``. mirror the
    :class:`RedshiftDriver` / :class:`AsyncpgDriver` patterns.

Observability
    decorate query-emitting methods with
    :func:`threetears.datasources.drivers.base._observed`
    (``driver_type="snowflake"``). cancellation.fired / .failed,
    cache.{hit,miss}, executor.saturation are manual emissions
    mirroring the :class:`RedshiftDriver` pattern.

Anything that does NOT transfer from postgres/redshift
    - Snowflake's per-warehouse compute model means a "warm
      connection" still costs compute time on the warehouse; the
      driver does NOT need to keep a pool the same size as
      Redshift's connection cache. tune
      :attr:`SnowflakeConnectionConfig.pool_size` against the
      warehouse's typical concurrency.
    - Snowflake's ``ABORT`` is bound to a specific query id; the
      cancel hook MUST capture ``cursor.sfqid`` BEFORE the cancel
      callback fires (the shared cancellation helper takes a
      ``cancel_callback`` -- close over the connection AND the
      live cursor when registering it).

CI-required live test
    when the implementation lands, mirror the
    ``tests/integration/test_redshift_driver_live.py`` shape: env-
    gated on a ``SNOWFLAKE_PASSWORD`` (or analogous) env var; in CI
    when the env var is missing AND ``CI=1`` is set, fail rather
    than skip silently. the live test is the smoking-gun proof the
    driver actually works against a real warehouse.
"""

from __future__ import annotations

from typing import Any

from threetears.datasources.config import SnowflakeConnectionConfig
from threetears.datasources.drivers.base import ColumnRow, Driver, TableRow
from threetears.observe import get_logger

__all__ = ["SnowflakeDriver"]

log = get_logger(__name__)


_NOT_IMPLEMENTED_HINT = (
    "See docs/datasource-task-12-snowflake-bigquery-stubs.md + the module docstring for the implementation roadmap."
)


class SnowflakeDriver(Driver):
    """Snowflake :class:`Driver` -- STUB. raises :class:`NotImplementedError` on every call.

    constructor validates the config so the
    :func:`threetears.datasources.drivers.create_driver` dispatch
    path is exercised end-to-end today; method bodies land when the
    implementation does. read this module's top-of-file docstring
    before writing the implementation -- the helpers to reuse
    (:class:`AsyncSyncBridge`, :meth:`Driver._with_cancellation`,
    :func:`_translate_placeholders`, the :func:`_observed`
    decorator) and the anti-patterns to avoid (raw
    :class:`ThreadPoolExecutor` instantiation, inline pool literals,
    swallowed cancellation) are all documented there.

    :param config: snowflake connection config validated at
        construction; carries account, warehouse, user, password_ref,
        optional role, pool_size, query_timeout_seconds
    :ptype config: SnowflakeConnectionConfig
    :param datasource_type: NOT a kwarg; the discriminator already
        lives on ``config.datasource_type`` via the union
    :param datasource_name: human-readable datasource name surfaced
        as the ``datasource_name`` attribute on every OTel metric
        emitted by :func:`_observed` (when the impl lands). defaults
        to ``"unknown"`` so callers that don't have the name in scope
        still produce valid metric streams
    :ptype datasource_name: str
    """

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        *,
        datasource_name: str = "unknown",
    ) -> None:
        """validate the config + capture the datasource_name. no I/O.

        :param config: snowflake connection config
        :ptype config: SnowflakeConnectionConfig
        :param datasource_name: name surfaced on OTel metric attributes
        :ptype datasource_name: str
        :return: nothing
        :rtype: None
        """
        self._config = config
        self._datasource_name = datasource_name

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """run a SELECT statement -- NOT YET IMPLEMENTED.

        :raises NotImplementedError: stub method; see module docstring
        """
        raise NotImplementedError(f"SnowflakeDriver.fetch is not yet implemented. {_NOT_IMPLEMENTED_HINT}")

    async def execute(self, sql: str, *params: Any) -> None:
        """run a DML / DDL statement -- NOT YET IMPLEMENTED.

        :raises NotImplementedError: stub method; see module docstring
        """
        raise NotImplementedError(f"SnowflakeDriver.execute is not yet implemented. {_NOT_IMPLEMENTED_HINT}")

    async def list_tables(self, schemas: list[str]) -> list[TableRow]:
        """list tables in the schema allow-list -- NOT YET IMPLEMENTED.

        :raises NotImplementedError: stub method; see module docstring
        """
        raise NotImplementedError(f"SnowflakeDriver.list_tables is not yet implemented. {_NOT_IMPLEMENTED_HINT}")

    async def list_columns(self, schemas: list[str]) -> list[ColumnRow]:
        """list columns for every table in the schema allow-list -- NOT YET IMPLEMENTED.

        :raises NotImplementedError: stub method; see module docstring
        """
        raise NotImplementedError(f"SnowflakeDriver.list_columns is not yet implemented. {_NOT_IMPLEMENTED_HINT}")

    async def table_hashes(self, schemas: list[str]) -> dict[tuple[str, str], str]:
        """per-table MD5 over column shape (Tier-2 probe) -- NOT YET IMPLEMENTED.

        :raises NotImplementedError: stub method; see module docstring
        """
        raise NotImplementedError(f"SnowflakeDriver.table_hashes is not yet implemented. {_NOT_IMPLEMENTED_HINT}")

    async def test_connection(self) -> None:
        """cheapest round-trip; verifies credentials -- NOT YET IMPLEMENTED.

        :raises NotImplementedError: stub method; see module docstring
        """
        raise NotImplementedError(f"SnowflakeDriver.test_connection is not yet implemented. {_NOT_IMPLEMENTED_HINT}")

    async def close(self) -> None:
        """release driver resources -- NOT YET IMPLEMENTED.

        :raises NotImplementedError: stub method; see module docstring
        """
        raise NotImplementedError(f"SnowflakeDriver.close is not yet implemented. {_NOT_IMPLEMENTED_HINT}")
