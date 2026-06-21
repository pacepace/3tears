"""Driver ABC + canonical row TypedDicts + shared helpers for datasource drivers.

This module is the contract every concrete datasource driver in 3tears
implements. Read it top-to-bottom before authoring a new driver --
``IMPLEMENTING_DRIVERS.md`` in this package walks the same surface from
the implementer's perspective.

The ABC is intentionally minimal:

- **No pool / no connection on the public surface.** BigQuery is stateless
  HTTPS; Snowflake has its own connection management; asyncpg uses a Pool
  internally but never exposes it. Keeping the surface backend-agnostic
  is how the contract stays honest across four very different backends.
- **Every method is ``async def``** even when the backend library is sync.
  Sync-bridged drivers (Redshift / Snowflake / BigQuery) route through
  :class:`threetears.datasources.drivers._sync_bridge.AsyncSyncBridge`.
- **``$1``-style placeholders** are the contract; concrete drivers
  translate to their dialect via
  :func:`threetears.datasources.drivers._util._translate_placeholders`.
- **Cancellation propagation** is mandatory and routes through
  :meth:`Driver._with_cancellation` so all four drivers share one
  implementation rather than three drifting copies.
- **Row shapes are pinned** via :class:`TableRow` and :class:`ColumnRow`.
  The ``is_nullable`` field is the raw warehouse string
  (``'YES'``/``'NO'``/``''``), NOT a boolean -- the Tier-2 column hash
  from datasource-task-02 depends on byte-equality with the warehouse-
  side MD5, which uses the raw value.

Observability contract (DS-09-11):

- ``datasource.driver.query.duration{driver_type, datasource_name}`` -- histogram
- ``datasource.driver.cancellation.fired{driver_type}`` -- counter
- ``datasource.driver.error{driver_type, error_kind}`` -- counter
- ``datasource.driver.executor.saturation{datasource_name}`` -- gauge
  (sync-bridged drivers only)
- ``datasource.driver.cache.{hit,miss}{datasource_name}`` -- counters
  (sync-bridged drivers only)

The :func:`_observed` decorator wraps method bodies to emit the always-on
duration + error metrics. Cache / saturation / cancellation-fired
metrics are driver-specific and emitted manually by the concrete driver.

``close()`` concurrency semantics (DS-09-12):

Single-shot. Concurrent calls and concurrent in-flight
``fetch``/``execute``/``fetch_iter`` while ``close()`` is running are
undefined behaviour. Drivers SHOULD set a ``_closed: bool`` flag and
reject subsequent operations with :class:`RuntimeError`. Drivers SHOULD
NOT call ``executor.shutdown(wait=True)`` from within an asyncio
coroutine (deadlocks the loop); use ``executor.shutdown(wait=False)``
and let the threads drain naturally.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypedDict, TypeVar

import asyncio

from threetears.observe import get_logger

__all__ = [
    "ColumnCoverage",
    "ColumnRow",
    "Driver",
    "TableRow",
]

log = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


# ---------------------------------------------------------------------------
# Pinned row shapes (DS-09-10)
# ---------------------------------------------------------------------------


class TableRow(TypedDict):
    """canonical row shape returned by :meth:`Driver.list_tables`.

    pinned via TypedDict so every backend driver agrees on the keys
    exposed to schema-introspection callers. matches the columns
    selected from ``information_schema.tables`` by the standard
    driver implementations.

    :key table_schema: schema name (matches the warehouse's
        ``information_schema.tables.table_schema``)
    :key table_name: table name within the schema
    """

    table_schema: str
    table_name: str


class ColumnRow(TypedDict):
    """canonical row shape returned by :meth:`Driver.list_columns`.

    ``is_nullable`` is the RAW warehouse value (``'YES'`` / ``'NO'`` /
    ``''``), NOT a boolean. the Tier-2 column hash in
    datasource-task-02 computes MD5 over a concatenation of the column
    metadata WITH the raw nullable string; converting to bool here
    would make the python-side hash diverge from the warehouse-side
    MD5, breaking the change-probe contract.

    :key table_schema: schema name (matches warehouse
        ``information_schema.columns.table_schema``)
    :key table_name: table name
    :key column_name: column name
    :key data_type: warehouse-reported data type string (varies by
        backend; e.g. asyncpg returns ``'integer'``, Redshift returns
        ``'INT4'``). drivers MUST surface the raw value
    :key is_nullable: raw warehouse nullable indicator -- ``'YES'``,
        ``'NO'``, or ``''``. NEVER a bool
    :key ordinal_position: 1-indexed column position from the
        information schema
    """

    table_schema: str
    table_name: str
    column_name: str
    data_type: str
    is_nullable: str
    ordinal_position: int


class ColumnCoverage(TypedDict):
    """per-column value-coverage counts from :meth:`Driver.column_value_coverage`.

    the raw facts a warehouse scan can decide about one numeric column: how many
    rows the table holds, and in how many of them the column is a non-null,
    non-zero value. ``total_rows > 0 and nonzero_count == 0`` is the unloaded /
    dead-column signal -- a ``0`` that means "not loaded for this table", not a
    measured zero. the driver returns counts only; the loaded-vs-unloaded VERDICT
    is the caller's, never the driver's.

    :key total_rows: row count of the scanned table
    :key nonzero_count: number of rows where the column is NOT NULL and ``<> 0``
    """

    total_rows: int
    nonzero_count: int


def _quote_ident(name: str) -> str:
    """double-quote a SQL identifier, escaping embedded quotes by doubling.

    identifiers (schema / table / column names) cannot be passed as bind
    parameters, so the coverage probe interpolates them -- quoting defends the
    interpolation even though the names come from the warehouse catalog.

    :param name: raw identifier
    :ptype name: str
    :return: a safely double-quoted identifier
    :rtype: str
    """
    return '"' + name.replace('"', '""') + '"'


def _build_coverage_sql(schema: str, table: str, columns: list[str]) -> str:
    """build the single-scan coverage aggregate for ``columns`` in ``schema.table``.

    one pass over the table: ``COUNT(*)`` plus, per column, a
    ``COUNT(NULLIF(col, 0))`` (which counts only non-null, non-zero rows --
    ``NULLIF`` maps both ``0`` and ``NULL`` to ``NULL`` and ``COUNT`` skips
    ``NULL``). columns alias to ``nz_<index>`` so the caller maps results back by
    position without re-quoting. portable standard SQL (Redshift / Postgres /
    Yugabyte / Snowflake); BigQuery's backtick identifiers would need an override.

    :param schema: schema name
    :ptype schema: str
    :param table: table name
    :ptype table: str
    :param columns: numeric columns to probe (non-empty)
    :ptype columns: list[str]
    :return: the coverage SELECT statement
    :rtype: str
    """
    qualified = f"{_quote_ident(schema)}.{_quote_ident(table)}"
    selects = ["COUNT(*) AS total_rows"]
    selects.extend(f"COUNT(NULLIF({_quote_ident(col)}, 0)) AS nz_{i}" for i, col in enumerate(columns))
    return f"SELECT {', '.join(selects)} FROM {qualified}"


def _build_coverage_by_dimension_sql(
    schema: str,
    table: str,
    dimension_column: str,
    columns: list[str],
) -> str:
    """build the single grouped-scan coverage aggregate for ``columns`` per dimension.

    one pass over the table grouped by ``dimension_column``: the dimension value,
    ``COUNT(*)``, and per column a ``COUNT(NULLIF(col, 0))`` (non-null, non-zero
    rows). the dimension aliases to ``dim_value`` and columns to ``nz_<index>`` so
    the caller maps results back by position without re-quoting. portable standard
    SQL (Redshift / Postgres / Yugabyte / Snowflake); BigQuery's backtick
    identifiers would need an override, same as :func:`_build_coverage_sql`.

    :param schema: schema name
    :ptype schema: str
    :param table: table name
    :ptype table: str
    :param dimension_column: column to group coverage by (non-empty)
    :ptype dimension_column: str
    :param columns: numeric columns to probe (non-empty)
    :ptype columns: list[str]
    :return: the grouped coverage SELECT statement
    :rtype: str
    """
    qualified = f"{_quote_ident(schema)}.{_quote_ident(table)}"
    dimension = _quote_ident(dimension_column)
    selects = [f"{dimension} AS dim_value", "COUNT(*) AS total_rows"]
    selects.extend(f"COUNT(NULLIF({_quote_ident(col)}, 0)) AS nz_{i}" for i, col in enumerate(columns))
    return f"SELECT {', '.join(selects)} FROM {qualified} GROUP BY {dimension}"


# ---------------------------------------------------------------------------
# @_observed decorator (DS-09-11)
# ---------------------------------------------------------------------------


# OTel metrics availability is cached after first probe -- zero-cost path
# when ``opentelemetry`` is not installed, matching the ``@traced``
# decorator's discipline in :mod:`threetears.observe.tracing`. drivers
# don't pay an import cost or a per-call attribute lookup beyond a single
# bool check.
_otel_metrics_available: bool | None = None


def _check_otel_metrics() -> bool:
    """probe for ``opentelemetry.metrics``; cache the result.

    :return: True iff ``opentelemetry.metrics`` is importable
    :rtype: bool
    """
    global _otel_metrics_available  # noqa: PLW0603
    if _otel_metrics_available is None:
        try:
            import opentelemetry.metrics  # noqa: F401

            _otel_metrics_available = True
        except ImportError:
            _otel_metrics_available = False
    return _otel_metrics_available


# instrument cache so we don't recreate Histogram / Counter objects on
# every call. keyed by ``(driver_type, metric_name)``. populated lazily
# the first time a metric fires for a given driver type.
_instrument_cache: dict[tuple[str, str], Any] = {}


def _get_query_duration_histogram(driver_type: str) -> Any:
    """fetch or create the per-driver-type query.duration histogram.

    :param driver_type: driver type label (e.g. ``"asyncpg"``, ``"redshift"``)
    :ptype driver_type: str
    :return: OTel Histogram instrument (or None if OTel not available)
    :rtype: Any
    """
    result: Any = None
    if _check_otel_metrics():
        key = (driver_type, "datasource.driver.query.duration")
        instrument = _instrument_cache.get(key)
        if instrument is None:
            from opentelemetry import metrics

            meter = metrics.get_meter("threetears.datasources.drivers")
            instrument = meter.create_histogram(
                name="datasource.driver.query.duration",
                description="datasource driver query duration in seconds",
                unit="s",
            )
            _instrument_cache[key] = instrument
        result = instrument
    return result


def _get_error_counter(driver_type: str) -> Any:
    """fetch or create the per-driver-type error counter.

    :param driver_type: driver type label
    :ptype driver_type: str
    :return: OTel Counter instrument (or None if OTel not available)
    :rtype: Any
    """
    result: Any = None
    if _check_otel_metrics():
        key = (driver_type, "datasource.driver.error")
        instrument = _instrument_cache.get(key)
        if instrument is None:
            from opentelemetry import metrics

            meter = metrics.get_meter("threetears.datasources.drivers")
            instrument = meter.create_counter(
                name="datasource.driver.error",
                description="datasource driver error count by error kind",
            )
            _instrument_cache[key] = instrument
        result = instrument
    return result


def _observed(driver_type: str) -> Callable[[F], F]:
    """decorator factory: wrap an async driver method with standard metric emission.

    emits :data:`datasource.driver.query.duration` (histogram) on every
    completion (success and exception) and
    :data:`datasource.driver.error` (counter) tagged with the exception
    class name on any exception. cancellation is NOT counted as an
    error -- :class:`asyncio.CancelledError` re-raises without bumping
    the error counter (the driver's manual ``cancellation.fired``
    counter covers that). this keeps the always-on metrics zero-effort
    for concrete drivers; backend-specific gauges / counters are
    emitted manually inside the driver.

    when ``opentelemetry.metrics`` isn't installed the decorator is a
    pure passthrough (single bool check per call), matching
    :func:`threetears.observe.traced`.

    :param driver_type: stable label for the driver type (e.g.
        ``"asyncpg"`` / ``"redshift"`` / ``"snowflake"`` / ``"bigquery"``).
        used as the OTel attribute ``driver_type``
    :ptype driver_type: str
    :return: decorator suitable for ``async def`` driver methods
    :rtype: Callable[[F], F]
    """

    def decorator(fn: F) -> F:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@_observed only wraps async functions; {fn.__qualname__} is not async")

        @functools.wraps(fn)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not _check_otel_metrics():
                return await fn(self, *args, **kwargs)
            histogram = _get_query_duration_histogram(driver_type)
            error_counter = _get_error_counter(driver_type)
            # datasource_name attribute is best-effort: drivers expose
            # ``_datasource_name`` if they want to participate, else
            # the metric carries only the driver_type label.
            datasource_name = getattr(self, "_datasource_name", "unknown")
            attrs = {"driver_type": driver_type, "datasource_name": datasource_name}
            start = time.monotonic()
            error_raised: BaseException | None = None
            try:
                result = await fn(self, *args, **kwargs)
            except asyncio.CancelledError:
                # propagation only -- the per-driver
                # ``cancellation.fired`` counter is bumped explicitly
                # by concrete drivers from _with_cancellation hooks
                raise
            except Exception as exc:
                error_raised = exc
                error_counter.add(
                    1,
                    attributes={
                        "driver_type": driver_type,
                        "error_kind": type(exc).__name__,
                    },
                )
                raise
            finally:
                histogram.record(time.monotonic() - start, attributes=attrs)
            return result

        return async_wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Driver ABC (DS-09-01..06)
# ---------------------------------------------------------------------------


class Driver(ABC):
    """abstract base for all datasource drivers.

    every concrete backend driver in
    :mod:`threetears.datasources.drivers` subclasses this ABC and
    implements the documented method surface. instances are created
    via :func:`threetears.datasources.drivers.create_driver` rather
    than direct instantiation -- the factory enforces lazy import of
    the backend library and provides the AGENT_INTERNAL pool-borrow
    plumbing.

    placeholder convention (DS-09-04):
        callers always pass ``$1``-style positional placeholders. each
        concrete driver translates internally via
        :func:`threetears.datasources.drivers._util._translate_placeholders`
        to its backend's expected dialect (``%s`` for pyformat /
        Redshift, ``:1`` for numeric, ``@p1`` for BigQuery named-at).

    cancellation contract (DS-09-05):
        ``fetch`` / ``execute`` / ``fetch_iter`` MUST cooperate with
        :mod:`asyncio` cancellation. if the awaiting coroutine is
        cancelled, the driver MUST attempt to cancel the in-flight
        query at the backend before re-raising
        :class:`asyncio.CancelledError`. concrete drivers route every
        backend call through :meth:`_with_cancellation` so the
        propagation logic lives in one place.

    row-shape contract (DS-09-10):
        :meth:`list_tables` returns :class:`TableRow` dicts and
        :meth:`list_columns` returns :class:`ColumnRow` dicts. the
        ``is_nullable`` field is the RAW warehouse string -- the
        Tier-2 hash depends on byte-equality with the warehouse-side
        MD5 which uses the raw value.

    observability contract (DS-09-11):
        concrete drivers SHOULD decorate query-emitting methods with
        :func:`_observed` to get the standard duration + error metrics
        for free. additional cache / saturation / cancellation-fired
        metrics are emitted manually.

    close concurrency semantics (DS-09-12):
        ``close()`` is single-shot. concurrent calls and concurrent
        in-flight queries while ``close()`` is running are undefined.
        drivers SHOULD set a ``_closed: bool`` flag and reject
        subsequent operations with :class:`RuntimeError`. drivers
        backed by a :class:`concurrent.futures.ThreadPoolExecutor`
        MUST call ``executor.shutdown(wait=False)``;
        ``wait=True`` deadlocks the asyncio event loop.
    """

    @abstractmethod
    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """run an arbitrary SELECT statement; materialize all rows in memory.

        for result sets large enough to risk OOM, prefer
        :meth:`fetch_iter`. ``fetch`` is right for small / bounded
        queries (introspection, single-row lookups, small joins).

        :param sql: SQL text with ``$1``-style placeholders
        :ptype sql: str
        :param params: positional placeholder values
        :ptype params: Any
        :return: list of column-name -> value dicts in row order
        :rtype: list[dict[str, Any]]
        :raises asyncio.CancelledError: propagated after best-effort
            backend cancellation
        :raises RuntimeError: if the driver was previously closed
        """

    @abstractmethod
    async def execute(self, sql: str, *params: Any) -> None:
        """run an arbitrary DML / DDL statement; discard any returned rows.

        :param sql: SQL text with ``$1``-style placeholders
        :ptype sql: str
        :param params: positional placeholder values
        :ptype params: Any
        :return: nothing
        :rtype: None
        :raises asyncio.CancelledError: propagated after best-effort
            backend cancellation
        :raises RuntimeError: if the driver was previously closed
        """

    async def fetch_iter(self, sql: str, *params: Any) -> AsyncIterator[dict[str, Any]]:
        """stream rows for large result sets.

        default implementation calls :meth:`fetch` and yields each row
        -- correct but materializes the full result first, defeating
        the streaming purpose. drivers backed by native server-side
        cursors (``asyncpg``, ``redshift_connector``) override to
        stream incrementally. the default exists so simple stub
        drivers (Snowflake, BigQuery) have a working ``fetch_iter``
        from day one.

        :param sql: SQL text with ``$1``-style placeholders
        :ptype sql: str
        :param params: positional placeholder values
        :ptype params: Any
        :return: async iterator over column-name -> value dicts
        :rtype: AsyncIterator[dict[str, Any]]
        :raises asyncio.CancelledError: propagated after best-effort
            backend cancellation
        :raises RuntimeError: if the driver was previously closed
        """
        rows = await self.fetch(sql, *params)
        for row in rows:
            yield row

    @abstractmethod
    async def list_tables(self, schemas: list[str]) -> list[TableRow]:
        """list tables visible to the connection within the given schemas.

        :param schemas: schema-name allow-list. empty list means "no
            tables" (callers gate against the agent.yaml schema
            whitelist before calling)
        :ptype schemas: list[str]
        :return: list of :class:`TableRow` dicts
        :rtype: list[TableRow]
        :raises RuntimeError: if the driver was previously closed
        """

    @abstractmethod
    async def list_columns(self, schemas: list[str]) -> list[ColumnRow]:
        """list columns for every table in the given schemas.

        :param schemas: schema-name allow-list
        :ptype schemas: list[str]
        :return: list of :class:`ColumnRow` dicts with the raw
            ``is_nullable`` string preserved
        :rtype: list[ColumnRow]
        :raises RuntimeError: if the driver was previously closed
        """

    @abstractmethod
    async def table_hashes(self, schemas: list[str]) -> dict[tuple[str, str], str]:
        """per-table MD5 over the column shape; Tier-2 change-probe.

        MUST byte-equal the python-side ``_compute_column_hash`` from
        datasource-task-02 over identical input. the warehouse-side
        MD5 hashes the raw ``is_nullable`` string (``'YES'`` / ``'NO'``
        / ``''``) -- using a boolean here makes the python-side hash
        diverge and breaks the probe.

        :param schemas: schema-name allow-list
        :ptype schemas: list[str]
        :return: mapping of ``(schema, table)`` -> column-shape hex digest
        :rtype: dict[tuple[str, str], str]
        :raises RuntimeError: if the driver was previously closed
        """

    @abstractmethod
    async def test_connection(self) -> None:
        """cheapest possible round-trip; verifies credentials + reachability.

        intended for health-check use; safe to call repeatedly. concrete
        drivers implement as ``SELECT 1`` (or equivalent). NOT the same
        as opening the connection -- that happens in ``__init__`` or
        lazily on first ``fetch``.

        :return: nothing; raises on failure
        :rtype: None
        :raises Exception: backend-specific error if the round-trip fails
        """

    @abstractmethod
    async def close(self) -> None:
        """release driver resources (connections, executors, clients).

        single-shot per the close-concurrency contract in the class
        docstring. drivers SHOULD set ``self._closed = True`` and
        reject subsequent operations with :class:`RuntimeError`.

        sync-bridged drivers MUST call ``executor.shutdown(wait=False)``
        -- ``wait=True`` deadlocks the asyncio event loop because the
        worker threads may be awaiting a coroutine that can't run
        while the loop is blocked.

        :return: nothing
        :rtype: None
        """

    # -------------------------------------------------------------------
    # Concrete: value-coverage probe (datasource honesty)
    # -------------------------------------------------------------------

    async def column_value_coverage(
        self,
        schema: str,
        table: str,
        columns: list[str],
    ) -> dict[str, ColumnCoverage]:
        """scan ``schema.table`` once and report value-coverage per numeric column.

        the decidable half of the unloaded-zero problem: a numeric column that
        is non-null and non-zero in ZERO of the table's rows is unloaded (the
        Alaska ``agg_dummy_eip`` incident), and a ``0`` read from it is missing
        data, not a measured zero. this method returns the raw counts
        (:class:`ColumnCoverage`); the caller decides the verdict. CONCRETE on the
        ABC (portable SQL routed through :meth:`fetch`) so every backend inherits
        it; it does NOT carry its own ``@_observed`` because the inner
        :meth:`fetch` it delegates to is already instrumented.

        :param schema: schema name of the table to scan
        :ptype schema: str
        :param table: table name to scan
        :ptype table: str
        :param columns: numeric column names to probe; an empty list short-circuits
            with no query
        :ptype columns: list[str]
        :return: mapping of column name -> :class:`ColumnCoverage` counts; empty
            when ``columns`` is empty
        :rtype: dict[str, ColumnCoverage]
        :raises asyncio.CancelledError: propagated from :meth:`fetch`
        :raises RuntimeError: if the driver was previously closed
        """
        result: dict[str, ColumnCoverage] = {}
        if columns:
            sql = _build_coverage_sql(schema, table, columns)
            rows = await self.fetch(sql)
            row = rows[0] if rows else {}
            total_rows = int(row.get("total_rows") or 0)
            for index, column in enumerate(columns):
                nonzero = row.get(f"nz_{index}")
                result[column] = ColumnCoverage(
                    total_rows=total_rows,
                    nonzero_count=int(nonzero or 0),
                )
        return result

    async def column_value_coverage_by_dimension(
        self,
        schema: str,
        table: str,
        dimension_column: str,
        columns: list[str],
    ) -> dict[str, dict[str, ColumnCoverage]]:
        """scan ``schema.table`` once grouped by ``dimension_column`` and report
        value-coverage per numeric column per dimension value.

        the grouped sibling of :meth:`column_value_coverage`: the same decidable
        question, but a single ``GROUP BY`` pass so the caller can decide a column
        is unloaded for SOME dimension values while loaded for others -- the
        partial-coverage case the whole-table probe cannot see (a column loaded for
        49 states but all-zero for every row of one). returns the raw counts per
        ``(column, dimension_value)``; the loaded-vs-unloaded VERDICT is the
        caller's. CONCRETE on the ABC (portable SQL routed through :meth:`fetch`).

        every requested column maps to a (possibly empty) per-dimension dict so
        callers can index ``result[column]`` safely. a NULL dimension value (rows
        with no value for the dimension) keys on the empty string. an empty
        ``columns`` OR empty ``dimension_column`` short-circuits with no query.

        :param schema: schema name of the table to scan
        :ptype schema: str
        :param table: table name to scan
        :ptype table: str
        :param dimension_column: column to group coverage by (e.g. ``state_code``);
            empty short-circuits with no query
        :ptype dimension_column: str
        :param columns: numeric column names to probe; empty short-circuits
        :ptype columns: list[str]
        :return: mapping of column name -> {dimension_value -> :class:`ColumnCoverage`};
            empty when ``columns`` or ``dimension_column`` is empty
        :rtype: dict[str, dict[str, ColumnCoverage]]
        :raises asyncio.CancelledError: propagated from :meth:`fetch`
        :raises RuntimeError: if the driver was previously closed
        """
        result: dict[str, dict[str, ColumnCoverage]] = {}
        if columns and dimension_column:
            result = {column: {} for column in columns}
            sql = _build_coverage_by_dimension_sql(schema, table, dimension_column, columns)
            rows = await self.fetch(sql)
            for row in rows:
                raw_dimension = row.get("dim_value")
                dimension_value = "" if raw_dimension is None else str(raw_dimension)
                total_rows = int(row.get("total_rows") or 0)
                for index, column in enumerate(columns):
                    nonzero = row.get(f"nz_{index}")
                    result[column][dimension_value] = ColumnCoverage(
                        total_rows=total_rows,
                        nonzero_count=int(nonzero or 0),
                    )
        return result

    # -------------------------------------------------------------------
    # Shared helpers (DS-09-05)
    # -------------------------------------------------------------------

    async def _with_cancellation(
        self,
        coro_fn: Callable[[], Awaitable[Any]],
        *,
        cancel_callback: Callable[[], Any],
    ) -> Any:
        """run ``coro_fn`` and propagate cancellation to the backend.

        the canonical pattern every concrete driver routes through.
        without a shared helper four drivers would each implement the
        try/except/cancel-callback dance and three would get it
        slightly wrong (forget to suppress callback errors, forget to
        await an async callback, swallow the original
        :class:`CancelledError`). the shared helper makes the contract
        load-bearing in exactly one place.

        propagation rules:

        - on success: return the coroutine's result.
        - on :class:`asyncio.CancelledError`: invoke ``cancel_callback``,
          ignore any exception it raises (logged at debug), then
          re-raise the cancellation. if the callback returns a
          coroutine, it is awaited.

        :param coro_fn: zero-arg callable returning the awaitable to
            run. taking a callable rather than the awaitable itself
            keeps construction inside the try block, which lets us
            short-circuit on cancellation that fires before the
            backend call starts
        :ptype coro_fn: Callable[[], Awaitable[Any]]
        :param cancel_callback: backend-specific cancellation hook
            (e.g. ``conn.cancel`` for asyncpg, ``stmt.cancel`` for
            redshift_connector). may be sync or async; sync callables
            whose return is a coroutine are awaited
        :ptype cancel_callback: Callable[[], Any]
        :return: the wrapped coroutine's result on success
        :rtype: Any
        :raises asyncio.CancelledError: re-raised after best-effort
            cancellation. the original exception is preserved
        """
        try:
            result = await coro_fn()
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                cancel_result = cancel_callback()
                if asyncio.iscoroutine(cancel_result):
                    await cancel_result
            raise
        return result
