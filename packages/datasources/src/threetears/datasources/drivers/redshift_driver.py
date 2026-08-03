"""``redshift_connector``-backed concrete :class:`Driver` for Amazon Redshift.

datasource-task-11: implements the :class:`Driver` ABC against AWS's
official ``redshift_connector`` (Apache-2.0) lib. THIS DRIVER IS THE
WHOLE REASON the datasource migration exists -- ``asyncpg`` against
``information_schema.columns`` on Redshift never completes (timed out
at 60s / 120s / 300s in production); ``redshift_connector`` handles
the Redshift pg-protocol quirks correctly.

architecture (DS-11-01..15):

- **AsyncSyncBridge for sync->async**. ``redshift_connector`` is a
  DB-API sync library; every blocking call runs through the shared
  :class:`AsyncSyncBridge` from
  :mod:`threetears.datasources.drivers._sync_bridge`. driver does NOT
  instantiate :class:`concurrent.futures.ThreadPoolExecutor` directly
  (enforcement test catches drift).
- **connection cache** (``collections.deque``) of size
  :attr:`RedshiftConnectionConfig.connection_cache_size`. Redshift
  TLS+auth costs 1-3s per fresh connection, so the cache amortizes
  the handshake across queries. mutations guarded by
  :class:`asyncio.Lock` (single-event-loop assumption documented per
  DS-11-15).
- **open-connection semaphore** (:class:`asyncio.Semaphore`) sized to
  ``connection_cache_size``. every connection-holding call acquires it
  BEFORE opening, so simultaneously-open connections never exceed the
  cache size: the (cache_size + 1)th concurrent caller waits instead of
  opening a connection past the warehouse user's CONNECTION LIMIT. the
  executor (``executor_max_workers``) bounds concurrent WORK; this
  bounds concurrent open CONNECTIONS -- a burst of N concurrent
  ``fetch()`` can no longer open N connections. NB across the fleet the
  hard cap is per-driver-instance: total open for a warehouse user is
  ``hub_replicas x connection_cache_size``, so the user's CONNECTION
  LIMIT must be sized to the replica count.
- **DB-API ``$1`` -> ``%s`` placeholder translation** via the shared
  :func:`threetears.datasources.drivers._util._translate_placeholders`
  helper with ``target_style="pyformat"``.
- **server-side streaming** via cursor ``arraysize`` +
  ``fetchmany()`` in :meth:`fetch_iter`, wrapped per-chunk through
  the bridge so the asyncio caller stays responsive while the worker
  thread pulls rows.
- **cancellation via best-effort connection-close** (DS-11-08 with the
  ambiguity called out in the implementation notes below).
  ``redshift_connector.Connection`` does NOT expose a public
  ``cancel()`` method (verified against the v2.1.x source); the only
  cancellation primitive is :meth:`Connection.close`, which sends the
  pgwire ``TERMINATE`` message + drops the socket. on
  :class:`asyncio.CancelledError` the driver runs ``conn.close()`` in
  a separate thread guarded by ``asyncio.wait_for(..., 5.0)`` and
  evicts the connection from the cache (a closed connection cannot
  be reused). on the rare timeout/failure path the
  ``datasource.driver.cancellation.failed`` counter is incremented,
  matching the shared observability contract.
- **secret handling** via :meth:`RedshiftConnectionConfig.resolve_password`
  returning :class:`pydantic.SecretStr`; ``.get_secret_value()`` is
  unwrapped at the LAST moment inside
  :func:`redshift_connector.connect`. backend exceptions wrapped with
  ``raise X from None`` so the cause chain cannot smuggle the
  password value into logs.
- **lazy fill** -- ``__init__`` does NO I/O; connections open on
  first query.
- **pod-crash mitigation** via :func:`weakref.finalize` (DS-11-11):
  best-effort cache drain at GC time. NOT a guarantee against SIGKILL
  pod crashes; document accordingly.

deviation from shard 11 spec (DS-11-08):

the shard text presumes ``redshift_connector.Connection.cancel()``
exists. it does NOT. ``cancel()`` is also absent from
:class:`redshift_connector.Cursor`. the AWS lib v2.1.14 has no
in-flight-query cancel primitive; only socket close. this driver
uses ``conn.close()`` as the cancellation mechanism -- closes the
pgwire socket, which causes Redshift's backend to detect the
disconnect and abort the running query on its WLM slot.
trade-off: the connection becomes unusable after cancel (eviction is
automatic, not an error path); the WLM slot frees within a few
seconds of the FIN; the driver pays the TLS+auth cost on the next
fresh connection.

close concurrency (DS-09-12 / DS-11-10):

- :meth:`close` sets :attr:`_closed` first; subsequent calls early-
  return; subsequent in-flight method calls raise :class:`RuntimeError`.
- every cached connection is closed in a worker thread (NOT on the
  asyncio event loop) via the bridge.
- :meth:`AsyncSyncBridge.close` uses ``shutdown(wait=False)``;
  ``wait=True`` would deadlock the event loop.

transactions, per-statement timeouts, and the read path (dsd-task-01):

the three changes below share one root cause and therefore one code
path -- ``redshift_connector`` runs without autocommit and its
``Cursor.execute`` issues ``begin transaction`` whenever the session is
idle (``redshift_connector/cursor.py`` lines 251-254, verified against
v2.1.7). every statement therefore runs inside a real transaction
block, which is what makes all three of these true at once:

- **a transaction is expressible, and needs no explicit BEGIN.**
  :meth:`RedshiftDriver.begin` pins the CONNECTION -- not merely the
  result -- for the unit of work's whole life; the block opens with the
  first statement. an explicit ``BEGIN`` would nest and warn.
  ``CREATE TABLE AS`` / ``DROP`` / ``ALTER`` / ``GRANT`` are all
  transactional on Redshift, so a correct promote is expressible once
  the statements share a session.
- **the per-statement timeout override is ``SET LOCAL``.** Redshift
  documents ``SET [ SESSION | LOCAL ] parameter { TO | = } value``, and
  ``SET LOCAL`` scopes to the transaction block already open, unwinding
  on the commit or rollback that ends it. that makes the leak
  structurally impossible rather than merely unlikely. because the
  ``SET LOCAL`` semantics cannot be proven from a unit test, the
  release path ALSO re-asserts the datasource ceiling whenever a
  checkout lowered it (see :meth:`RedshiftDriver._reset_session_sync`)
  -- belt and braces, and the requirement admits either.
  ``query_timeout_seconds`` is the connection-level CEILING; the
  per-statement value is the caller's.
- **the read path closes its transaction on release.** a completed
  ``SELECT`` used to return a connection to the cache holding an open
  snapshot and its locks, which is what made a ``DROP`` or an ownership
  change block behind a *finished* query. every acquire site now
  routes its release through :meth:`RedshiftDriver._finish_checkout`,
  from a ``finally``.

one consequence worth knowing: the ``SET`` statements issued at
connection open (``statement_timeout``, ``search_path``) are COMMITTED
there, because a session-level ``SET`` made inside a block that later
rolls back is discarded with it -- and the release path now rolls back
on every checkout.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import functools
import socket
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

try:
    import redshift_connector
except ImportError as exc:  # pragma: no cover -- environments without the extra
    # the factory's lazy-import contract means this module only loads
    # when a REDSHIFT-typed config dispatches here. raising at module
    # load with a clear install hint is more actionable than the bare
    # ImportError ``import redshift_connector`` would otherwise raise.
    raise ImportError(
        "redshift-connector not installed; install via 'pip install 3tears-datasources[redshift]'"
    ) from exc

if TYPE_CHECKING:
    # type-only aliases so the ``: "RedshiftConnection"`` annotations
    # below don't trip mypy's attribute-resolution on the dynamically-
    # typed redshift_connector module (it ships no stubs). runtime
    # references go through ``redshift_connector.connect`` /
    # ``conn.cursor`` and friends; mypy only sees this block.
    RedshiftConnection = Any
    RedshiftCursor = Any

from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.drivers._sync_bridge import AsyncSyncBridge
from threetears.datasources.drivers._util import (
    _translate_placeholders,
    build_set_local_statement_timeout_sql,
    build_set_search_path_sql,
    build_set_statement_timeout_sql,
)
from threetears.datasources.drivers.base import (
    CallbackTransaction,
    ColumnRow,
    Driver,
    TableRow,
    Transaction,
    _check_otel_metrics,
    _instrument_cache,
    _observed,
)
from threetears.observe import get_logger, traced

__all__ = [
    "DriverCancellationError",
    "DriverConnectError",
    "DriverQueryError",
    "RedshiftDriver",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# SQL constants (DS-11-07)
# ---------------------------------------------------------------------------


#: list tables visible inside the schema allow-list.
#:
#: NOTE 1 (placeholder shape): Redshift's ``redshift_connector`` lib
#: does NOT support passing a python ``list[str]`` as a parameter for
#: ``ANY(%s)`` (verified empirically: it raises
#: ``ArrayContentNotSupportedError: oid 25 not supported as array
#: contents`` -- text arrays as bind params are explicitly rejected
#: by the lib's ``make_params`` implementation). the driver builds
#: an ``IN (%s, %s, ...)`` clause with one placeholder per schema at
#: call time. the ``{placeholders}`` token is the only piece the
#: driver substitutes -- schema values are still bound via ``%s`` so
#: the contract stays parameterized (NOT SQL-injection-vulnerable).
#:
#: NOTE 2 (source view): we query ``SVV_TABLES`` rather than
#: ``information_schema.tables`` because ``information_schema.*`` on
#: Redshift is a leader-node-only view, slow under WLM contention
#: (observed up to ~7min for ``information_schema.columns`` against
#: ``reporting_prod``). ``SVV_TABLES`` / ``SVV_COLUMNS`` are
#: Redshift-native system views that surface the same rows but
#: execute in seconds and support arbitrary aggregates (which
#: ``information_schema.columns`` does NOT -- LISTAGG over it raises
#: ``Specified types or functions not supported on Redshift tables``).
_REDSHIFT_TABLES_SQL_TEMPLATE = """
SELECT table_schema, table_name
FROM SVV_TABLES
WHERE table_schema IN ({placeholders})
AND table_type = 'BASE TABLE'
ORDER BY table_schema, table_name
""".strip()


#: list columns for every table in the schema allow-list. ``is_nullable``
#: surfaces as the raw warehouse string -- the Tier-2 hash depends on
#: byte-equality with the warehouse-side MD5 (see
#: :data:`_REDSHIFT_TABLE_HASHES_SQL_TEMPLATE`). ``SVV_COLUMNS`` over
#: ``information_schema.columns`` -- see notes on the tables template.
#:
#: NOTE: SVV_COLUMNS' ``data_type`` strings differ from
#: ``information_schema.columns`` (e.g. ``character varying`` vs
#: ``VARCHAR``). the python-side ``_compute_column_hash`` MUST be
#: applied over the same rows that the warehouse-side MD5 sees, so
#: as long as both sides observe SVV_COLUMNS, byte-equivalence holds.
#: cross-driver hash equivalence (asyncpg vs redshift) is NOT
#: guaranteed; same-driver python-vs-SQL IS guaranteed.
_REDSHIFT_COLUMNS_SQL_TEMPLATE = """
SELECT table_schema, table_name, column_name, data_type, is_nullable, ordinal_position
FROM SVV_COLUMNS
WHERE table_schema IN ({placeholders})
ORDER BY table_schema, table_name, ordinal_position
""".strip()


#: per-table MD5 over the column shape (Tier-2 change-probe). same
#: payload formula as the asyncpg driver. byte-equivalent to the
#: python-side ``column_hash_payload`` helper ON THE SAME ROWS -- i.e.
#: both sides MUST read from SVV_COLUMNS for the equivalence to hold.
#:
#: EACH COLUMN IS HASHED BEFORE THE LISTAGG, AND THAT IS LOAD-BEARING.
#: Redshift refuses a LISTAGG result over 65535 bytes, and the previous
#: formula aggregated the raw ``name:type:nullable`` strings, so the
#: payload grew with name lengths. Introspecting ripple's nine source
#: schemas failed outright -- ``influencers.test_targetsmart`` has 1462
#: columns whose raw payload is 69,210 bytes, and ONE table over the
#: limit failed the query for all 5,615 tables in scope, leaving the
#: datasource with no catalog at all.
#:
#: Pre-hashing makes the payload 33 bytes per column regardless of name
#: length: a ceiling of ~1985 columns against Redshift's own 1600-column
#: table limit, so it can no longer overflow here.
#:
#: CHANGING THIS FORMULA CHANGES EVERY STORED HASH, and the python side
#: in ``threetears.datasources.introspection.column_hash_payload`` must
#: change identically in the same commit or the two stop agreeing and
#: every sweep reports spurious changes forever.
_REDSHIFT_TABLE_HASHES_SQL_TEMPLATE = """
SELECT table_schema, table_name,
       MD5(LISTAGG(MD5(column_name || ':' || data_type || ':' || COALESCE(is_nullable, '')), ',') WITHIN GROUP (ORDER BY ordinal_position)) AS column_hash
FROM SVV_COLUMNS
WHERE table_schema IN ({placeholders})
GROUP BY table_schema, table_name
ORDER BY table_schema, table_name
""".strip()


def _build_in_clause(n: int) -> str:
    """build a ``%s, %s, ...`` placeholder string for n positional bind params.

    used by the introspection SQL methods to construct
    ``WHERE table_schema IN (...)`` with one parameterized placeholder
    per schema. fully parameterized (NOT a SQL-injection vector --
    only the placeholder *count* depends on the schema-list length;
    schema values are still bound via the cursor's parameter machinery).

    :param n: number of placeholders (= number of schemas)
    :ptype n: int
    :return: a comma-separated ``%s`` sequence, e.g. ``"%s, %s, %s"``
    :rtype: str
    """
    return ", ".join(["%s"] * n)


#: cheapest possible round-trip for :meth:`RedshiftDriver.test_connection`.
_PING_SQL = "SELECT 1"


@dataclasses.dataclass
class _Checkout:
    """one connection checked out of the cache, and what release owes it.

    the driver hands a connection out in three places (single-statement
    query, streaming ``fetch_iter``, and a pinned transaction) and every
    one of them owes the same two things on release: close the open
    transaction, and restore the datasource timeout ceiling if a
    per-statement override was applied. carrying that state on one
    object is what keeps the release path from drifting into three
    partial copies -- the two fixes share a code path deliberately.

    :param conn: the checked-out redshift connection
    :ptype conn: RedshiftConnection
    :param timeout_overridden: True once a ``SET LOCAL
        statement_timeout`` has run on this checkout, so the release
        path re-asserts the ceiling
    :ptype timeout_overridden: bool
    :param poisoned: True when the connection must NOT go back in the
        cache (cancelled mid-statement, or its release-path reset
        failed)
    :ptype poisoned: bool
    """

    conn: "RedshiftConnection"
    timeout_overridden: bool = False
    poisoned: bool = False


def _apply_socket_keepalive(conn: RedshiftConnection, cfg: RedshiftConnectionConfig) -> None:
    """apply aggressive OS-level TCP keepalive on a redshift_connector connection.

    redshift_connector's ``connect()`` accepts only the ``tcp_keepalive`` bool, so
    the granular idle / interval / count from :class:`RedshiftConnectionConfig` are
    set here via ``setsockopt`` on the ``SSLSocket`` it opened. best-effort +
    platform-guarded: ``TCP_KEEPIDLE`` / ``TCP_KEEPINTVL`` / ``TCP_KEEPCNT`` are
    Linux socket options (the hub + tool pods run Linux); one absent on the host
    platform is skipped, and a ``setsockopt`` failure is logged, not raised -- the
    connection is usable, only half-dead-socket detection falls back to the system
    default. detection window ~= idle + count * interval.

    :param conn: live redshift_connector connection to tune
    :ptype conn: RedshiftConnection
    :param cfg: datasource config carrying the keepalive knobs
    :ptype cfg: RedshiftConnectionConfig
    :return: nothing
    :rtype: None
    """
    if not cfg.tcp_keepalive:
        return
    # redshift_connector exposes its underlying SSLSocket as ``_usock``; read it via
    # getattr so a future rename degrades to a skip (and does not trip SLF001).
    sock = getattr(conn, "_usock", None)
    if sock is None:
        log.warning("redshift keepalive: connection exposes no _usock; leaving keepalive at the system default")
        return
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for opt_name, value in (
            ("TCP_KEEPIDLE", cfg.tcp_keepalive_idle_seconds),
            ("TCP_KEEPINTVL", cfg.tcp_keepalive_interval_seconds),
            ("TCP_KEEPCNT", cfg.tcp_keepalive_count),
        ):
            opt = getattr(socket, opt_name, None)
            if opt is not None:
                sock.setsockopt(socket.IPPROTO_TCP, opt, value)
    except OSError as exc:
        log.warning(
            "redshift keepalive: setsockopt failed (%s); connection usable, keepalive at system default",
            exc,
        )


#: server-side fetchmany batch size for :meth:`fetch_iter`. tunable in
#: the future via a new :class:`RedshiftConnectionConfig` field; for
#: today the value is local because it's a streaming-chunk constant,
#: NOT a pool/executor/timeout knob (the enforcement test's banned
#: kwarg set excludes ``arraysize``).
_FETCH_ITER_ARRAYSIZE = 1000


#: cancel-callback timeout for the cancellation path (DS-11-08).
#: ``redshift_connector.Connection.close()`` (used as our cancel
#: primitive since the lib has no ``cancel()`` API) opens no
#: secondary socket but can still block on the TERMINATE write if
#: the TCP send buffer is wedged. the wait_for guard makes the
#: failure observable rather than silent. module-level so the
#: enforcement test's ``timeout=Constant`` walker doesn't flag the
#: call site (Name reference, not Constant literal).
_CANCEL_TIMEOUT_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Exception types (DS-11-12)
# ---------------------------------------------------------------------------


class DriverConnectError(Exception):
    """raised when connect / auth fails.

    the message carries host / port / database (safe to log) but
    NEVER the resolved password value. callers raise with ``from None``
    so the original ``redshift_connector`` exception -- which sometimes
    embeds the password in nested context -- cannot reach loggers via
    ``__cause__``.
    """


class DriverQueryError(Exception):
    """raised when a query fails for non-cancellation reasons.

    cancellation propagates via :class:`asyncio.CancelledError`
    (subclassed by :class:`DriverCancellationError`); all other
    backend failures wrap in this type. messages MUST NOT carry
    credentials -- if a future contributor wants to embed SQL in the
    message, scrub bind-parameter values first.
    """


class DriverCancellationError(asyncio.CancelledError):
    """redshift-specific cancellation marker.

    subclass of :class:`asyncio.CancelledError` so existing
    ``except asyncio.CancelledError`` handlers still catch it. lets
    callers that want to distinguish driver-initiated cancellation
    from generic asyncio cancellation do so via
    ``isinstance(exc, DriverCancellationError)`` without breaking the
    propagation contract.
    """


# ---------------------------------------------------------------------------
# Per-driver-type metric helpers (DS-11-13)
# ---------------------------------------------------------------------------


def _get_cancellation_fired_counter() -> Any:
    """fetch-or-create the ``datasource.driver.cancellation.fired`` counter.

    bumped from the cancel callback after a successful (in-time)
    ``conn.close()``. matches the asyncpg-driver pattern so the metric
    surface is uniform across drivers.

    :return: OTel Counter (or None if OTel isn't installed)
    :rtype: Any
    """
    result: Any = None
    if _check_otel_metrics():
        key = ("redshift", "datasource.driver.cancellation.fired")
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


def _get_cancellation_failed_counter() -> Any:
    """fetch-or-create the ``datasource.driver.cancellation.failed`` counter.

    bumped when the wrapped ``wait_for(conn.close(), 5.0)`` times out
    or otherwise raises. matches DS-11-08's observable-failure
    requirement -- the cancel path is NEVER silent.

    :return: OTel Counter (or None if OTel isn't installed)
    :rtype: Any
    """
    result: Any = None
    if _check_otel_metrics():
        key = ("redshift", "datasource.driver.cancellation.failed")
        instrument = _instrument_cache.get(key)
        if instrument is None:
            from opentelemetry import metrics

            meter = metrics.get_meter("threetears.datasources.drivers")
            instrument = meter.create_counter(
                name="datasource.driver.cancellation.failed",
                description="datasource driver cancellation failed count",
            )
            _instrument_cache[key] = instrument
        result = instrument
    return result


def _get_cache_hit_counter() -> Any:
    """fetch-or-create the ``datasource.driver.cache.hit`` counter.

    bumped on every :meth:`RedshiftDriver._acquire_connection` that
    pops an existing connection from the cache.

    :return: OTel Counter (or None if OTel isn't installed)
    :rtype: Any
    """
    result: Any = None
    if _check_otel_metrics():
        key = ("redshift", "datasource.driver.cache.hit")
        instrument = _instrument_cache.get(key)
        if instrument is None:
            from opentelemetry import metrics

            meter = metrics.get_meter("threetears.datasources.drivers")
            instrument = meter.create_counter(
                name="datasource.driver.cache.hit",
                description="datasource driver connection-cache hit count",
            )
            _instrument_cache[key] = instrument
        result = instrument
    return result


def _get_cache_miss_counter() -> Any:
    """fetch-or-create the ``datasource.driver.cache.miss`` counter.

    bumped on every :meth:`RedshiftDriver._acquire_connection` that
    opens a fresh connection (cache empty).

    :return: OTel Counter (or None if OTel isn't installed)
    :rtype: Any
    """
    result: Any = None
    if _check_otel_metrics():
        key = ("redshift", "datasource.driver.cache.miss")
        instrument = _instrument_cache.get(key)
        if instrument is None:
            from opentelemetry import metrics

            meter = metrics.get_meter("threetears.datasources.drivers")
            instrument = meter.create_counter(
                name="datasource.driver.cache.miss",
                description="datasource driver connection-cache miss count",
            )
            _instrument_cache[key] = instrument
        result = instrument
    return result


def _get_executor_saturation_gauge() -> Any:
    """fetch-or-create the ``datasource.driver.executor.saturation`` gauge.

    emitted on each :meth:`RedshiftDriver._acquire_and_run` invocation
    as a running snapshot of bridge-executor worker pressure. when
    OTel isn't installed the gauge is None and the driver skips the
    emission with a single bool check.

    :return: OTel UpDownCounter / Gauge instrument, or None
    :rtype: Any
    """
    result: Any = None
    if _check_otel_metrics():
        key = ("redshift", "datasource.driver.executor.saturation")
        instrument = _instrument_cache.get(key)
        if instrument is None:
            from opentelemetry import metrics

            meter = metrics.get_meter("threetears.datasources.drivers")
            # use UpDownCounter rather than Gauge for portability --
            # OTel's sync Gauge API is comparatively new and the
            # UpDownCounter accepts an arbitrary delta which we
            # compute against the previously-reported value via a
            # running ``set``-style emission pattern (saturate
            # increases/decreases each emit).
            instrument = meter.create_up_down_counter(
                name="datasource.driver.executor.saturation",
                description="bridge-executor active-worker pressure snapshot",
            )
            _instrument_cache[key] = instrument
        result = instrument
    return result


# ---------------------------------------------------------------------------
# Pod-crash mitigation (DS-11-11)
# ---------------------------------------------------------------------------


def _drain_cache_static(
    connections: Iterable["RedshiftConnection"],
) -> None:
    """module-level finalize callback; close any connections still alive at GC.

    invoked by :func:`weakref.finalize` registered in
    :class:`RedshiftDriver.__init__`. best-effort -- pod crashes
    bypass GC entirely (SIGKILL doesn't run finalizers), so this is
    NOT a guarantee against orphaned Redshift sessions; the cluster
    cleans them up on session timeout (~4h default).

    NOTE: this runs at GC time, possibly on the asyncio loop thread
    or a thread without an event loop; it MUST be sync and MUST NOT
    raise. each ``conn.close()`` is wrapped in a try/except for that
    reason.

    :param connections: the live cache iterable (the deque held by
        the driver instance, NOT a snapshot). passing the deque
        itself -- rather than ``list(self._cache)`` at init time --
        ensures the finalize sees whatever connections are cached at
        GC time. lazy-fill means the cache is empty at construction;
        capturing a snapshot then would drain nothing
    :ptype connections: Iterable[redshift_connector.Connection]
    """
    # snapshot iteration here is intentional: callers may mutate the
    # underlying deque while we iterate (e.g. a concurrent close()
    # racing with the finalize), and ``list(...)`` gives us a stable
    # view of "what was in the cache at the moment we started".
    for conn in list(connections):
        try:
            conn.close()
        except Exception as exc:  # noqa: BLE001 -- defensive at finalize
            # finalize must not raise; log + continue so other cached
            # connections still get the close attempt.
            log.debug("redshift finalize close failed: %s", exc)


# ---------------------------------------------------------------------------
# RedshiftDriver
# ---------------------------------------------------------------------------


class RedshiftDriver(Driver):
    """concrete :class:`Driver` for Amazon Redshift via ``redshift_connector``.

    construct via :func:`threetears.datasources.drivers.create_driver`
    rather than directly -- the factory enforces the lazy-import
    contract.

    threading model:

    every backend call routes through :attr:`_bridge` (a per-instance
    :class:`AsyncSyncBridge`). the driver does NOT touch
    :class:`concurrent.futures.ThreadPoolExecutor` directly. cache
    mutations are guarded by :attr:`_cache_lock` (an
    :class:`asyncio.Lock`); the cache assumes a single driver
    instance is consumed from a single asyncio event loop (drivers
    are not shared across loops in our deployment -- documented
    assumption per DS-11-15).

    cancellation:

    ``redshift_connector`` exposes NO cancel API. the driver uses
    :meth:`redshift_connector.Connection.close` (the pgwire
    ``TERMINATE`` message) as the cancellation primitive: closes the
    socket from a worker thread, lets the Redshift backend detect
    the disconnect and abort the WLM slot. the connection is
    automatically evicted from the cache because a closed connection
    is unusable.

    the wrapped close runs inside ``asyncio.wait_for(..., 5.0)`` so a
    hung close doesn't pin the cancellation path. failure increments
    :data:`datasource.driver.cancellation.failed` (the failure path
    is observable, never silent).

    :param config: per-driver connection config carrying host/port/
        database/username/password_ref + executor/cache/timeout sizing
    :ptype config: RedshiftConnectionConfig
    :param datasource_name: human-readable name of the datasource this
        driver serves. surfaces as the ``datasource_name`` attribute
        on every OTel metric emitted by :func:`_observed`. defaults
        to ``"unknown"`` so callers without the name in scope can
        omit; Hub broker / tool-pod (shards 13/14) thread the name
        from :attr:`DatasourceConfig.name`
    :ptype datasource_name: str
    """

    def __init__(
        self,
        config: RedshiftConnectionConfig,
        *,
        datasource_name: str = "unknown",
    ) -> None:
        """capture config; build bridge + cache; register finalize. no I/O.

        :param config: per-driver redshift config
        :ptype config: RedshiftConnectionConfig
        :param datasource_name: name of the datasource the driver
            serves; surfaces on every emitted OTel metric
        :ptype datasource_name: str
        :return: nothing
        :rtype: None
        """
        self._config = config
        # bridge sized from config -- the enforcement test catches
        # inline literals. construction does NOT spawn workers; the
        # executor is started lazily on first submission.
        self._bridge = AsyncSyncBridge(
            max_workers=config.executor_max_workers,
            name=f"rs-{config.host}",
        )
        # connection cache: deque bounded by config.connection_cache_size.
        # lazy-fill -- no eager warmup. each enqueued connection has
        # already had ``SET statement_timeout`` applied (see
        # :meth:`_open_connection`).
        self._cache: collections.deque["RedshiftConnection"] = collections.deque(maxlen=config.connection_cache_size)
        # cache mutations are guarded inside this lock so concurrent
        # acquire/release on the same event loop don't race.
        self._cache_lock = asyncio.Lock()
        # bound simultaneously-open connections to connection_cache_size: every
        # connection-holding call (_acquire_and_run, fetch_iter) acquires this
        # BEFORE opening, so the (cache_size + 1)th concurrent caller waits here
        # rather than opening a connection past the warehouse user's CONNECTION
        # LIMIT. sized to the cache so total open never exceeds what the cache
        # retains -- the executor (executor_max_workers) bounds concurrent WORK,
        # this bounds concurrent open CONNECTIONS.
        self._connection_semaphore = asyncio.Semaphore(config.connection_cache_size)
        # per-connection server-side backend pid captured at open via
        # ``SELECT pg_backend_pid()``. on cancel the driver opens a
        # FRESH connection to issue ``pg_terminate_backend(<pid>)`` --
        # closing the CLIENT socket does NOT kill the SERVER-SIDE
        # query (a real abandoned query ran for 7.4h in production,
        # leaking a pool slot). a WeakKeyDictionary so an evicted /
        # GC'd connection drops its pid entry automatically without a
        # manual cleanup pass.
        self._backend_pids: weakref.WeakKeyDictionary[RedshiftConnection, int] = weakref.WeakKeyDictionary()
        self._closed = False
        # read by :func:`_observed` as the ``datasource_name`` attribute
        # on every metric emission. matches the AsyncpgDriver contract.
        self._datasource_name = datasource_name
        # pod-crash mitigation per DS-11-11: register a finalize
        # callback that drains the cache at GC time. NOT a guarantee
        # against SIGKILL pod crashes; the cluster reaps orphaned
        # sessions on its session timeout (~4h default).
        #
        # pass the LIVE deque (not list(self._cache) at init time --
        # lazy-fill means the cache is empty here; a snapshot would
        # drain nothing). weakref.finalize holds a strong ref to the
        # deque, which is safe because the deque doesn't ref-cycle
        # back to the driver instance.
        self._finalize = weakref.finalize(
            self,
            _drain_cache_static,
            self._cache,
        )

    # -------------------------------------------------------------------
    # Connection lifecycle
    # -------------------------------------------------------------------

    def _open_connection_sync(self) -> RedshiftConnection:
        """open a fresh ``redshift_connector.Connection`` (sync).

        called from a worker thread via the bridge. issues
        ``SET statement_timeout`` once so the server-side cancel
        fires cleanly if a query overruns
        :attr:`RedshiftConnectionConfig.query_timeout_seconds`.

        :return: live connection with statement_timeout configured
        :rtype: RedshiftConnection
        :raises DriverConnectError: on auth/network failure; the
            wrapper carries host/port/database but NEVER the password
        """
        cfg = self._config
        try:
            conn = redshift_connector.connect(
                host=cfg.host,
                port=cfg.port,
                database=cfg.database,
                user=cfg.username,
                password=(cfg.resolve_password().get_secret_value() if cfg.password_ref is not None else None),
                sslmode=cfg.sslmode,
                # redshift_connector.connect (2.1.7) accepts only the tcp_keepalive
                # BOOL. the granular idle / interval / count are applied post-connect
                # via setsockopt (see _apply_socket_keepalive) -- passing them as
                # connect kwargs raises TypeError, which this method used to swallow
                # into a bare "connection failed", masking a total datasource outage.
                tcp_keepalive=cfg.tcp_keepalive,
            )
        except Exception as exc:
            # break the cause chain (``from None``) so the original redshift_connector
            # exception -- which may embed sensitive connection detail in its message
            # or nested context -- cannot reach loggers / tracebacks. surface ONLY the
            # exception TYPE (a class name, never sensitive) so a config / library
            # error (an unsupported connect kwarg, a bad sslmode) is diagnosable
            # instead of masked as a bare "connection failed" -- which is exactly how
            # a total datasource outage hid when connect() rejected a keepalive kwarg.
            raise DriverConnectError(
                f"connection failed for {cfg.host}:{cfg.port}/{cfg.database} ({type(exc).__name__})"
            ) from None
        # Aggressive TCP keepalive so the OS detects a half-dead socket (a silently-
        # dropped Redshift connection while a worker blocks awaiting a query result)
        # in ~1 min and surfaces it as a socket error -- instead of the bridge worker
        # hanging forever in a native SSL read no client-side async / statement
        # timeout can cancel. Applied here (not as connect kwargs) because
        # redshift_connector's connect() has no granular keepalive parameters.
        _apply_socket_keepalive(conn, cfg)
        # apply the server-side statement timeout once per connection.
        # Redshift expects milliseconds AND does not accept bind params
        # in ``SET`` statements (parser rejects ``SET x = $1`` with
        # ``syntax error``). format the int inline -- the value is
        # pydantic-validated as int at config-build time, NEVER from
        # user-controlled SQL, so this is not an injection vector.
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(build_set_statement_timeout_sql(cfg.query_timeout_seconds))
            finally:
                cursor.close()
            # commit so the SET survives the release path's ROLLBACK.
            # redshift_connector opens a transaction block for us --
            # ``Cursor.execute`` issues ``begin transaction`` whenever
            # the session is idle and autocommit is off (cursor.py
            # 251-254) -- and a session-level ``SET`` made inside a
            # block that later ROLLBACKs is DISCARDED. without this
            # commit the ceiling would silently vanish the first time a
            # completed SELECT closed its transaction on release.
            conn.commit()
        except Exception:
            # failure to apply timeout is non-fatal for connect but
            # we still wrap with from None so the original error
            # can't leak the password.
            with self._suppress_close():
                conn.close()
            raise DriverConnectError(
                f"failed to set statement_timeout on {cfg.host}:{cfg.port}/{cfg.database}"
            ) from None
        # apply the configured ``search_path`` once per connection so
        # agents can write unqualified table references in SQL
        # (``SELECT * FROM report_geofacts_joined_data`` resolves
        # through the search_path instead of requiring
        # ``reporting_prod.report_geofacts_joined_data`` everywhere).
        # this must run AFTER statement_timeout so the timeout caps
        # the search_path SET itself. it is ALSO re-applied on every
        # cache-hit acquisition (see :meth:`_acquire_connection`) so a
        # reused session that lost the setting still resolves unqualified
        # names -- the alternative is intermittent "relation does not
        # exist" the moment a connection is recycled.
        try:
            self._apply_search_path_sync(conn)
        except Exception:
            # same failure shape as statement_timeout: wrap so the
            # original redshift_connector error (which can carry the
            # password in nested context) does not leak via the
            # exception cause chain. allowed_schemas is the
            # adversarial surface here -- a malformed identifier
            # would land in the server-side syntax error path; we
            # still don't want to surface it to the agent because
            # the wrapper's message is the only thing callers see.
            with self._suppress_close():
                conn.close()
            raise DriverConnectError(f"failed to set search_path on {cfg.host}:{cfg.port}/{cfg.database}") from None
        # capture the server-side backend pid (best-effort) so the
        # cancel path can issue ``pg_terminate_backend(<pid>)`` from a
        # FRESH connection -- closing the client socket alone does NOT
        # kill the running Redshift query. a failure to read the pid is
        # NON-FATAL: the connection is still usable for queries, only
        # the server-side cancel degrades, so we log + continue rather
        # than close / raise.
        pid = self._read_backend_pid_sync(conn)
        if pid is not None:
            self._backend_pids[conn] = pid
        return conn

    def _read_backend_pid_sync(self, conn: RedshiftConnection) -> int | None:
        """read ``pg_backend_pid()`` on ``conn`` best-effort (sync).

        called from a worker thread at open. the pid lets the cancel
        path terminate the SERVER-SIDE backend via a fresh connection.
        ANY failure is non-fatal -- the connection stays usable for
        queries; only the server-side cancel degrades -- so the failure
        is logged at WARNING and ``None`` is returned rather than
        raised.

        :param conn: live redshift connection to read the pid from
        :ptype conn: RedshiftConnection
        :return: backend pid, or None when the read fails / is empty
        :rtype: int | None
        """
        pid: int | None = None
        try:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT pg_backend_pid()")
                row = cursor.fetchone()
            finally:
                cursor.close()
            if row is not None:
                pid = int(row[0])
        except Exception as exc:  # noqa: BLE001 -- best-effort pid read
            # non-fatal: the connection is still usable; only the
            # server-side cancel degrades. log + continue.
            log.warning("redshift backend pid read failed: %s; server-side cancel will degrade", exc)
            pid = None
        return pid

    def _apply_search_path_sync(self, conn: RedshiftConnection) -> None:
        """issue ``SET search_path`` on ``conn`` from the configured schemas (sync).

        no-op when ``allowed_schemas`` is empty (the build helper returns ``None``),
        so datasources that do not configure a search_path pay nothing. raises on a
        failed SET; callers decide whether to close + evict (cache-hit) or close +
        wrap (fresh open).

        :param conn: a live redshift connection
        :ptype conn: RedshiftConnection
        :return: nothing
        :rtype: None
        """
        search_path_sql = build_set_search_path_sql(self._config.allowed_schemas)
        if search_path_sql is not None:
            cursor = conn.cursor()
            try:
                cursor.execute(search_path_sql)
            finally:
                cursor.close()
            # same reason as the timeout SET at open: the statement ran
            # inside the transaction block redshift_connector opened for
            # it, and the release path's ROLLBACK would otherwise
            # discard the search_path along with the read snapshot.
            conn.commit()

    @staticmethod
    @contextlib.contextmanager
    def _suppress_close() -> Iterator[None]:
        """record-and-swallow a failure on a fallback-close path.

        used in places where a best-effort close is desirable but a
        failure must NOT propagate (finalize, cancel cleanup, double-
        close paths). a single helper centralizes the discipline so a
        future reviewer can see every suppression site at one grep --
        and so every one of them leaves a debug line, since a close
        that keeps failing is how a connection leak starts.

        :return: context-manager that logs and absorbs any exception
        :rtype: Iterator[None]
        """
        try:
            yield
        except Exception as exc:  # noqa: BLE001 -- best-effort close; failure must not propagate
            log.debug(
                "best-effort close failed (ignored)",
                extra={"extra_data": {"error": str(exc), "error_type": type(exc).__name__}},
            )

    async def _acquire_connection(self) -> RedshiftConnection:
        """pop a warm connection from the cache OR open a fresh one.

        cache-hit path: pop the leftmost (LIFO would be marginally
        better for thermal locality but the deque cycles fast enough
        that FIFO is fine). cache-miss path: open a new connection
        through the bridge (TLS+auth blocks for ~1-3s on Redshift).

        bumps :data:`datasource.driver.cache.hit` or ``.miss``
        accordingly. attribute carries ``datasource_name`` so per-
        datasource cache effectiveness is observable.

        :return: a connection ready for use
        :rtype: RedshiftConnection
        :raises RuntimeError: if the driver was previously closed
        :raises DriverConnectError: on auth/network failure
        """
        if self._closed:
            raise RuntimeError("RedshiftDriver is closed")
        conn: RedshiftConnection | None = None
        async with self._cache_lock:
            if self._cache:
                conn = self._cache.popleft()
        if conn is not None:
            # re-apply search_path on the reused session: a recycled / reset
            # redshift connection can lose the SET issued at open, which surfaces
            # as intermittent "relation does not exist" on unqualified names. the
            # SET is cheap and a no-op when no search_path is configured. if it
            # fails the connection is unhealthy -- drop it and fall through to a
            # fresh open rather than hand back a broken connection.
            try:
                await self._bridge.to_thread_with_cancel(
                    lambda: self._apply_search_path_sync(conn),
                    cancel_cb=lambda: None,
                )
            except Exception:
                with self._suppress_close():
                    conn.close()
                conn = None
        if conn is not None:
            hit_counter = _get_cache_hit_counter()
            if hit_counter is not None:
                hit_counter.add(1, attributes={"datasource_name": self._datasource_name})
            return conn
        # cache miss: open a fresh connection through the bridge so
        # the TLS+auth wait doesn't block the asyncio loop.
        miss_counter = _get_cache_miss_counter()
        if miss_counter is not None:
            miss_counter.add(1, attributes={"datasource_name": self._datasource_name})
        # NOT to_thread_with_cancel here -- open is a one-shot
        # ; cancellation during connect would leave a half-open
        # connection which the worker thread closes naturally when
        # the call returns. use the bridge's executor directly via
        # a no-cancel-cb path.
        new_conn = await self._bridge.to_thread_with_cancel(
            self._open_connection_sync,
            cancel_cb=lambda: None,
        )
        return new_conn

    def _reset_session_sync(self, conn: RedshiftConnection, restore_timeout: bool) -> None:
        """close the open transaction and restore the timeout ceiling (sync).

        the ONE release-path reset both fixes share. two things happen,
        in this order and only this order:

        1. ``rollback`` -- ``redshift_connector`` runs without
           autocommit and ``Cursor.execute`` opens a transaction block
           for every statement (cursor.py 251-254), so a completed
           ``SELECT`` returns a connection holding an open snapshot and
           its locks. that is what makes a ``DROP`` or an ownership
           change block behind a *finished* query, and it is a
           prerequisite for reaping and stats sharing a warehouse, not
           a tidy-up. the lib no-ops the call when the session is
           already idle, so the hot path pays nothing.
        2. re-assert the datasource ceiling, but ONLY when this
           checkout lowered it. per-statement overrides use ``SET
           LOCAL``, which unwinds with the transaction and therefore
           cannot leak -- this second pass is defence for the case
           where the engine degrades ``SET LOCAL`` to session scope,
           which a unit test cannot rule out. the commit that follows
           makes the restored ceiling durable against the NEXT
           release-path rollback.

        the ordering matters: rollback first, because a session-level
        ``SET`` issued inside a transaction that later rolls back is
        discarded with it.

        :param conn: connection being released
        :ptype conn: RedshiftConnection
        :param restore_timeout: True when a per-statement override ran
            on this checkout
        :ptype restore_timeout: bool
        :return: nothing
        :rtype: None
        :raises Exception: any redshift_connector failure propagates to
            the async caller, which evicts the connection
        """
        conn.rollback()
        if restore_timeout:
            cursor = conn.cursor()
            try:
                cursor.execute(build_set_statement_timeout_sql(self._config.query_timeout_seconds))
            finally:
                cursor.close()
            conn.commit()

    async def _finish_checkout(self, checkout: _Checkout) -> None:
        """reset the session, then return the connection to the cache.

        ALWAYS called from a ``finally``. a reset skipped on the error
        path IS the leak, and a connection returned to the cache with
        an open snapshot IS the blocked ``DROP``, so both live here and
        every acquire site routes through this one method.

        a reset that itself fails poisons the connection: a session
        whose transaction state we could not establish must never be
        handed to the next caller.

        :param checkout: the checkout being finished
        :ptype checkout: _Checkout
        :return: nothing
        :rtype: None
        """
        if not checkout.poisoned:
            try:
                await self._bridge.to_thread_with_cancel(
                    functools.partial(self._reset_session_sync, checkout.conn, checkout.timeout_overridden),
                    cancel_cb=lambda: None,
                )
            except Exception as exc:  # noqa: BLE001 -- release path must not raise over the caller's result
                log.warning(
                    "redshift release-path rollback / timeout reset failed: %s; evicting connection",
                    exc,
                )
                checkout.poisoned = True
        if checkout.poisoned:
            await self._evict_connection(checkout.conn)
        else:
            await self._release_connection(checkout.conn)

    async def _cancel_checkout(self, checkout: _Checkout) -> None:
        """abort the in-flight statement and poison the checkout.

        the cancel path for every acquire site. terminates the
        SERVER-SIDE backend first -- closing the client socket does NOT
        kill a running Redshift query (one leaked a pool slot for 7.4h
        in production) -- then closes the client socket. both are
        best-effort and neither raises; the failure is counted so it is
        observable rather than silent.

        killing the session is also what rolls back an in-flight
        transaction: Redshift discards the block when the backend dies,
        so a cancellation mid-transaction cannot leave a half-applied
        promote behind.

        :param checkout: the checkout whose statement is being cancelled
        :ptype checkout: _Checkout
        :return: nothing
        :rtype: None
        """
        checkout.poisoned = True
        cancel_fired = _get_cancellation_fired_counter()
        cancel_failed = _get_cancellation_failed_counter()
        pid = self._backend_pids.get(checkout.conn)
        if pid is not None:
            await self._terminate_backend(pid)
        try:
            await asyncio.wait_for(
                asyncio.to_thread(checkout.conn.close),
                timeout=_CANCEL_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
            log.warning(
                "redshift cancel (conn.close) failed: %s; evicting connection",
                exc,
            )
            if cancel_failed is not None:
                cancel_failed.add(1, attributes={"driver_type": "redshift"})
        else:
            if cancel_fired is not None:
                cancel_fired.add(1, attributes={"driver_type": "redshift"})

    async def _release_connection(self, conn: RedshiftConnection) -> None:
        """return a connection to the cache; close it if cache is full.

        the deque's ``maxlen`` guarantees no unbounded growth -- an
        ``append`` on a full deque drops the leftmost element. we
        explicitly close the dropped element so the connection
        doesn't leak.

        :param conn: connection to release
        :ptype conn: RedshiftConnection
        :return: nothing
        :rtype: None
        """
        if self._closed:
            # driver is shutting down; just close the connection.
            await self._bridge.to_thread_with_cancel(
                conn.close,
                cancel_cb=lambda: None,
            )
            return
        evicted: RedshiftConnection | None = None
        async with self._cache_lock:
            if len(self._cache) == self._cache.maxlen:
                # the deque would drop the leftmost on append; pop it
                # explicitly so we can close it cleanly.
                evicted = self._cache.popleft()
            self._cache.append(conn)
        if evicted is not None:
            with self._suppress_close():
                await self._bridge.to_thread_with_cancel(
                    evicted.close,
                    cancel_cb=lambda: None,
                )

    async def _evict_connection(self, conn: RedshiftConnection) -> None:
        """explicitly drop a connection from the cache + close it.

        called from the cancellation path -- a closed connection
        cannot be returned to the cache. also called when the cancel
        callback's ``wait_for(..., 5.0)`` itself fails so the
        connection is treated as poisoned regardless of which side
        of the close timed out.

        :param conn: connection to evict + close
        :ptype conn: RedshiftConnection
        :return: nothing
        :rtype: None
        """
        async with self._cache_lock:
            # remove if present; the connection may already be out of
            # the cache if it's currently checked out for a query.
            with self._suppress_close():
                self._cache.remove(conn)
        with self._suppress_close():
            await self._bridge.to_thread_with_cancel(
                conn.close,
                cancel_cb=lambda: None,
            )

    def _terminate_backend_sync(self, pid: int) -> None:
        """open a fresh connection and ``pg_terminate_backend(<pid>)`` (sync).

        called from a worker thread on the cancel path. closing the
        CLIENT socket of the running connection does NOT kill the
        SERVER-SIDE Redshift query; a fresh connection issuing
        ``pg_terminate_backend`` does (our DB user is not a superuser,
        so ``CANCEL`` does not work but ``pg_terminate_backend`` does).
        the fresh connection uses EXACTLY the five connect kwargs of
        :meth:`_open_connection_sync` (no statement_timeout / search_path
        / extra kwargs) and is always closed in a ``finally``.

        :param pid: server-side backend pid to terminate; captured from
            the server at open via ``pg_backend_pid()``, never user SQL
        :ptype pid: int
        :return: nothing
        :rtype: None
        :raises Exception: any redshift_connector failure propagates to
            the async wrapper, which logs + swallows it (best-effort)
        """
        cfg = self._config
        conn = redshift_connector.connect(
            host=cfg.host,
            port=cfg.port,
            database=cfg.database,
            user=cfg.username,
            password=(cfg.resolve_password().get_secret_value() if cfg.password_ref is not None else None),
        )
        try:
            cursor = conn.cursor()
            try:
                # pid is an int captured from the SERVER at open
                # (``pg_backend_pid()``), never user-controlled SQL --
                # same justification as the inline statement_timeout
                # format. Redshift also rejects bind params in this
                # admin function call, so the int is formatted inline.
                cursor.execute("SELECT pg_terminate_backend(%d)" % pid)
            finally:
                cursor.close()
        finally:
            with self._suppress_close():
                conn.close()

    async def _terminate_backend(self, pid: int) -> None:
        """terminate the server-side backend ``pid`` best-effort via a fresh connection.

        runs :meth:`_terminate_backend_sync` in a worker thread wrapped
        in ``asyncio.wait_for`` so a hung connect / terminate cannot pin
        the cancellation path. NEVER raises: a TimeoutError / Exception
        is logged at WARNING (and the ``cancellation.failed`` counter is
        bumped) so the failure is observable, never silent -- the
        client-socket close + evict path still runs regardless.

        :param pid: server-side backend pid to terminate
        :ptype pid: int
        :return: nothing
        :rtype: None
        """
        cancel_failed = _get_cancellation_failed_counter()
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._terminate_backend_sync, pid),
                timeout=_CANCEL_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001 -- best-effort terminate
            log.warning(
                "redshift server-side terminate (pg_terminate_backend) failed for pid=%s: %s",
                pid,
                exc,
            )
            if cancel_failed is not None:
                cancel_failed.add(1, attributes={"driver_type": "redshift"})
        else:
            log.info("terminated server-side backend pid=%s", pid)

    async def _acquire_and_run(
        self,
        op: Callable[["RedshiftConnection"], Awaitable[Any]],
        *,
        timeout_overridden: bool = False,
    ) -> Any:
        """acquire the open-connection semaphore, then run ``op`` holding one connection.

        the semaphore is sized to ``connection_cache_size``, so at most that many
        connections are open at once across concurrent callers: the
        (cache_size + 1)th concurrent caller WAITS here instead of opening a
        connection past the warehouse user's CONNECTION LIMIT. the cache amortizes
        the TLS+auth handshake; the semaphore bounds the count. the held span is
        exactly "this call holds a connection" -- acquired before the open in
        :meth:`_run_with_connection`, released after that method returns the
        connection to the cache.

        :param op: callable taking the acquired connection, returning the awaitable
        :ptype op: Callable[["RedshiftConnection"], Awaitable[Any]]
        :param timeout_overridden: True when ``op`` issues a per-statement
            ``SET LOCAL statement_timeout``, so the release path re-asserts
            the datasource ceiling
        :ptype timeout_overridden: bool
        :return: whatever ``op(conn)`` resolved to
        :rtype: Any
        :raises RuntimeError: if the driver was previously closed
        """
        if self._closed:
            raise RuntimeError("RedshiftDriver is closed")
        async with self._connection_semaphore:
            result = await self._run_with_connection(op, timeout_overridden=timeout_overridden)
        return result

    async def _run_with_connection(
        self,
        op: Callable[["RedshiftConnection"], Awaitable[Any]],
        *,
        timeout_overridden: bool = False,
    ) -> Any:
        """acquire a connection + route through :meth:`_with_cancellation`.

        canonical wrapper every single-statement method uses. wires
        :meth:`_cancel_checkout` as the cancel callback so an outer
        ``asyncio.CancelledError`` terminates the server-side backend,
        closes the socket, and bumps the ``cancellation.fired`` (or
        ``.failed``) counter so the observability is honest.

        the ``finally`` routes through :meth:`_finish_checkout`, which
        closes the session's transaction and restores the timeout
        ceiling before the connection goes anywhere near the cache.
        that placement is load-bearing on both counts: a reset skipped
        on the error path is the timeout leak, and a connection cached
        mid-snapshot is the ``DROP`` that blocks behind a finished
        ``SELECT``. this method used to roll back only on the error
        path, which is exactly the read-path hole -- the success path
        now closes its transaction too.

        also emits one ``datasource.driver.executor.saturation``
        sample per invocation so the bridge-executor pressure is
        observable. snapshot is best-effort -- we cannot interrogate
        the executor's idle/busy split precisely, so the metric
        reports a +1 / -1 delta around the call window.

        :param op: callable that takes the acquired
            :class:`redshift_connector.Connection` and returns the
            awaitable to run (typically a wrapper that runs the sync
            cursor methods through the bridge)
        :ptype op: Callable[["RedshiftConnection"], Awaitable[Any]]
        :param timeout_overridden: True when ``op`` issues a
            per-statement ``SET LOCAL statement_timeout``
        :ptype timeout_overridden: bool
        :return: whatever ``op(conn)`` resolved to
        :rtype: Any
        :raises asyncio.CancelledError: propagated after best-effort
            backend cancellation via :meth:`Connection.close`
        :raises RuntimeError: if the driver was previously closed
        """
        if self._closed:
            raise RuntimeError("RedshiftDriver is closed")
        conn = await self._acquire_connection()
        checkout = _Checkout(conn=conn, timeout_overridden=timeout_overridden)
        # saturate-gauge +1: the next worker is now busy from the
        # asyncio side. -1 happens in the finally below.
        saturation = _get_executor_saturation_gauge()
        if saturation is not None:
            saturation.add(1, attributes={"datasource_name": self._datasource_name})
        try:
            result = await self._with_cancellation(
                lambda: op(conn),
                cancel_callback=lambda: self._cancel_checkout(checkout),
            )
        finally:
            if saturation is not None:
                saturation.add(-1, attributes={"datasource_name": self._datasource_name})
            await self._finish_checkout(checkout)
        return result

    # -------------------------------------------------------------------
    # Driver ABC: query surface
    # -------------------------------------------------------------------

    def _apply_statement_timeout_sync(
        self,
        cursor: RedshiftCursor,
        timeout_seconds: int | None,
    ) -> None:
        """issue the transaction-local per-statement timeout, when one was asked for.

        ``SET LOCAL`` scopes the value to the transaction block
        ``redshift_connector`` opened for this statement (its
        ``Cursor.execute`` issues ``begin transaction`` whenever the
        session is idle and autocommit is off), so it unwinds on the
        commit or rollback that ends the block. that is what makes the
        leak structurally impossible instead of merely unlikely: a
        bare ``SET`` would persist on the cached connection and hand a
        bounded aggregate's 120s to whatever build borrowed it next.

        :param cursor: live cursor the caller's statement will run on
        :ptype cursor: RedshiftCursor
        :param timeout_seconds: per-statement override, or None
        :ptype timeout_seconds: int | None
        :return: nothing
        :rtype: None
        :raises ValueError: if the override is not a positive int
        """
        if timeout_seconds is not None:
            cursor.execute(build_set_local_statement_timeout_sql(timeout_seconds))

    def _fetch_sync(
        self,
        conn: RedshiftConnection,
        sql: str,
        params: tuple[Any, ...],
        timeout_seconds: int | None,
    ) -> list[dict[str, Any]]:
        """run one SELECT on ``conn`` and materialize its rows (sync).

        shared by :meth:`fetch` and the transaction handle so both
        apply the per-statement timeout the same way.

        :param conn: connection to run on
        :ptype conn: RedshiftConnection
        :param sql: already-translated SQL text
        :ptype sql: str
        :param params: positional bind values
        :ptype params: tuple[Any, ...]
        :param timeout_seconds: per-statement override, or None
        :ptype timeout_seconds: int | None
        :return: list of column-name -> value dicts in row order
        :rtype: list[dict[str, Any]]
        """
        cursor = conn.cursor()
        try:
            self._apply_statement_timeout_sync(cursor, timeout_seconds)
            # redshift_connector accepts a tuple OR None for the
            # parameters argument; pass None when there are no
            # params so the lib's "no params" path runs.
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            rows = cursor.fetchall()
            cols = [c[0] for c in cursor.description]
            result = [dict(zip(cols, row)) for row in rows]
        finally:
            cursor.close()
        return result

    def _execute_sync(
        self,
        conn: RedshiftConnection,
        sql: str,
        params: tuple[Any, ...],
        timeout_seconds: int | None,
        *,
        commit: bool,
    ) -> None:
        """run one DML / DDL statement on ``conn`` (sync).

        shared by :meth:`execute` and the transaction handle. the
        transaction handle passes ``commit=False`` -- committing per
        statement is precisely what makes a multi-statement promote
        inexpressible.

        :param conn: connection to run on
        :ptype conn: RedshiftConnection
        :param sql: already-translated SQL text
        :ptype sql: str
        :param params: positional bind values
        :ptype params: tuple[Any, ...]
        :param timeout_seconds: per-statement override, or None
        :ptype timeout_seconds: int | None
        :param commit: whether to commit after the statement
        :ptype commit: bool
        :return: nothing
        :rtype: None
        """
        cursor = conn.cursor()
        try:
            self._apply_statement_timeout_sync(cursor, timeout_seconds)
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            if commit:
                # DDL/DML doesn't always autocommit in DB-API; commit
                # explicitly so callers see the change.
                conn.commit()
        finally:
            cursor.close()

    @traced
    @_observed(driver_type="redshift")
    async def fetch(self, sql: str, *params: Any, timeout_seconds: int | None = None) -> list[dict[str, Any]]:
        """run a SELECT statement; materialize all rows in memory.

        :param sql: SQL text with ``$1``-style placeholders
        :ptype sql: str
        :param params: positional placeholder values
        :ptype params: Any
        :param timeout_seconds: per-statement timeout override in
            seconds, applied as ``SET LOCAL statement_timeout`` so it
            cannot leak to the next borrower of this cached connection.
            None leaves ``query_timeout_seconds`` -- the connection
            CEILING -- in force
        :ptype timeout_seconds: int | None
        :return: list of column-name -> value dicts in row order
        :rtype: list[dict[str, Any]]
        :raises asyncio.CancelledError: propagated after backend cancel
        :raises RuntimeError: if the driver was previously closed
        :raises ValueError: if ``timeout_seconds`` is not a positive int
        :raises DriverConnectError: if connection acquisition fails
        """
        if self._closed:
            raise RuntimeError("RedshiftDriver is closed")
        # validate BEFORE a connection is taken out of the cache: a bad
        # override is a caller bug, not a reason to burn a checkout.
        if timeout_seconds is not None:
            build_set_local_statement_timeout_sql(timeout_seconds)
        translated = _translate_placeholders(sql, "pyformat")

        async def _op(conn: RedshiftConnection) -> Any:
            return await self._bridge.to_thread_with_cancel(
                functools.partial(self._fetch_sync, conn, translated, params, timeout_seconds),
                cancel_cb=conn.close,
            )

        result: list[dict[str, Any]] = await self._acquire_and_run(
            _op,
            timeout_overridden=timeout_seconds is not None,
        )
        return result

    @traced
    @_observed(driver_type="redshift")
    async def execute(self, sql: str, *params: Any, timeout_seconds: int | None = None) -> None:
        """run a DML / DDL statement; discard any returned rows.

        :param sql: SQL text with ``$1``-style placeholders
        :ptype sql: str
        :param params: positional placeholder values
        :ptype params: Any
        :param timeout_seconds: per-statement timeout override in
            seconds; see :meth:`fetch`
        :ptype timeout_seconds: int | None
        :return: nothing
        :rtype: None
        :raises asyncio.CancelledError: propagated after backend cancel
        :raises RuntimeError: if the driver was previously closed
        :raises ValueError: if ``timeout_seconds`` is not a positive int
        """
        if self._closed:
            raise RuntimeError("RedshiftDriver is closed")
        if timeout_seconds is not None:
            build_set_local_statement_timeout_sql(timeout_seconds)
        translated = _translate_placeholders(sql, "pyformat")

        async def _op(conn: RedshiftConnection) -> Any:
            return await self._bridge.to_thread_with_cancel(
                functools.partial(self._execute_sync, conn, translated, params, timeout_seconds, commit=True),
                cancel_cb=conn.close,
            )

        await self._acquire_and_run(_op, timeout_overridden=timeout_seconds is not None)

    # -------------------------------------------------------------------
    # Driver ABC: transaction surface (DSD-01-01 / DSD-01-02)
    # -------------------------------------------------------------------

    @traced
    async def begin(self) -> Transaction:
        """open a transaction pinned to one Redshift session.

        the connection leaves the cache here and does not return until
        commit or rollback, so a transaction reduces effective pool
        capacity for its duration -- which is why the executor caps
        build concurrency at enqueue rather than letting arbitrarily
        many transactions contend for the cache. the two must not be
        tuned independently.

        NO explicit ``BEGIN`` is issued. ``redshift_connector`` runs
        without autocommit and its ``Cursor.execute`` sends ``begin
        transaction`` itself whenever the session is idle (cursor.py
        251-254), so the block opens with the transaction's first
        statement; an explicit ``BEGIN`` here would nest and warn.
        pinning the CONNECTION -- not merely the result -- is what
        makes the block one unit of work.

        :return: a live transaction handle over the pinned session
        :rtype: Transaction
        :raises RuntimeError: if the driver was previously closed
        :raises DriverConnectError: if connection acquisition fails
        """
        if self._closed:
            raise RuntimeError("RedshiftDriver is closed")
        await self._connection_semaphore.acquire()
        try:
            conn = await self._acquire_connection()
        except BaseException:
            self._connection_semaphore.release()
            raise
        checkout = _Checkout(conn=conn)
        return CallbackTransaction(
            on_fetch=functools.partial(self._transaction_fetch, checkout),
            on_execute=functools.partial(self._transaction_execute, checkout),
            on_finish=functools.partial(self._transaction_finish, checkout),
        )

    async def _transaction_fetch(
        self,
        checkout: _Checkout,
        sql: str,
        params: tuple[Any, ...],
        timeout_seconds: int | None,
    ) -> list[dict[str, Any]]:
        """run a SELECT on the transaction's pinned session.

        :param checkout: the pinned checkout
        :ptype checkout: _Checkout
        :param sql: SQL text with ``$1``-style placeholders
        :ptype sql: str
        :param params: positional bind values
        :ptype params: tuple[Any, ...]
        :param timeout_seconds: per-statement override, or None
        :ptype timeout_seconds: int | None
        :return: list of column-name -> value dicts in row order
        :rtype: list[dict[str, Any]]
        :raises asyncio.CancelledError: propagated after backend cancel
        :raises ValueError: if ``timeout_seconds`` is not a positive int
        """
        translated = _translate_placeholders(sql, "pyformat")
        if timeout_seconds is not None:
            build_set_local_statement_timeout_sql(timeout_seconds)
            checkout.timeout_overridden = True
        result: list[dict[str, Any]] = await self._with_cancellation(
            lambda: self._bridge.to_thread_with_cancel(
                functools.partial(self._fetch_sync, checkout.conn, translated, params, timeout_seconds),
                cancel_cb=checkout.conn.close,
            ),
            cancel_callback=lambda: self._cancel_checkout(checkout),
        )
        return result

    async def _transaction_execute(
        self,
        checkout: _Checkout,
        sql: str,
        params: tuple[Any, ...],
        timeout_seconds: int | None,
    ) -> None:
        """run a DML / DDL statement on the transaction's pinned session.

        does NOT commit -- the whole point of the transaction API is
        that the unit of work ends where the caller says it does.

        :param checkout: the pinned checkout
        :ptype checkout: _Checkout
        :param sql: SQL text with ``$1``-style placeholders
        :ptype sql: str
        :param params: positional bind values
        :ptype params: tuple[Any, ...]
        :param timeout_seconds: per-statement override, or None
        :ptype timeout_seconds: int | None
        :return: nothing
        :rtype: None
        :raises asyncio.CancelledError: propagated after backend cancel
        :raises ValueError: if ``timeout_seconds`` is not a positive int
        """
        translated = _translate_placeholders(sql, "pyformat")
        if timeout_seconds is not None:
            build_set_local_statement_timeout_sql(timeout_seconds)
            checkout.timeout_overridden = True
        await self._with_cancellation(
            lambda: self._bridge.to_thread_with_cancel(
                functools.partial(
                    self._execute_sync,
                    checkout.conn,
                    translated,
                    params,
                    timeout_seconds,
                    commit=False,
                ),
                cancel_cb=checkout.conn.close,
            ),
            cancel_callback=lambda: self._cancel_checkout(checkout),
        )

    async def _transaction_finish(self, checkout: _Checkout, commit: bool) -> None:
        """end the transaction and return its pinned connection.

        runs the caller's disposition first, then the same
        :meth:`_finish_checkout` every other acquire site uses, then
        releases the open-connection permit. the permit release is in
        the outermost ``finally`` so a failed commit cannot strand a
        slot and starve every subsequent caller.

        :param checkout: the pinned checkout
        :ptype checkout: _Checkout
        :param commit: True to commit, False to roll back
        :ptype commit: bool
        :return: nothing
        :rtype: None
        :raises Exception: a failing commit propagates; the connection
            is still released
        """
        try:
            if not checkout.poisoned:
                await self._bridge.to_thread_with_cancel(
                    checkout.conn.commit if commit else checkout.conn.rollback,
                    cancel_cb=lambda: None,
                )
        except Exception:
            # a commit that failed leaves the session in an unknown
            # transaction state; never hand that to the next caller.
            checkout.poisoned = True
            raise
        finally:
            try:
                await self._finish_checkout(checkout)
            finally:
                self._connection_semaphore.release()

    @traced
    async def fetch_iter(self, sql: str, *params: Any) -> AsyncIterator[dict[str, Any]]:
        """stream rows via DB-API ``fetchmany`` (server-side cursor).

        overrides the ABC default. composes the streaming so each
        chunk-pull runs through the bridge (the
        :class:`redshift_connector.Cursor` is sync) -- per-chunk
        bridge hops rather than per-row, so the asyncio loop stays
        responsive without paying executor-submit overhead for each
        row yield.

        cancellation between chunks is best-effort: a cancellation
        between two ``fetchmany`` chunks propagates naturally
        through the generator. cancellation during a ``fetchmany``
        runs ``conn.close`` from the bridge's cancel-cb path,
        aborting the cursor on the Redshift side. the connection is
        NOT returned to the cache after this generator completes --
        the cursor lifecycle is tied to the connection lifecycle and
        a half-consumed cursor leaves the connection in an
        ambiguous state.

        :param sql: SQL text with ``$1``-style placeholders
        :ptype sql: str
        :param params: positional placeholder values
        :ptype params: Any
        :return: async iterator over column-name -> value dicts
        :rtype: AsyncIterator[dict[str, Any]]
        :raises RuntimeError: if the driver was previously closed
        """
        if self._closed:
            raise RuntimeError("RedshiftDriver is closed")
        translated = _translate_placeholders(sql, "pyformat")
        # hold the open-connection semaphore for the whole streaming span -- the
        # connection is checked out until the generator is exhausted or closed;
        # released in the finally after the connection is released/evicted. guard
        # the acquire so a failed open does not leak a permit.
        await self._connection_semaphore.acquire()
        try:
            conn = await self._acquire_connection()
        except BaseException:
            self._connection_semaphore.release()
            raise
        checkout = _Checkout(conn=conn)
        cursor: "RedshiftCursor | None" = None

        def _open_cursor() -> tuple["RedshiftCursor", list[str]]:
            """open the cursor + execute the statement; return cursor + col names."""
            cur = conn.cursor()
            cur.arraysize = _FETCH_ITER_ARRAYSIZE
            if params:
                cur.execute(translated, params)
            else:
                cur.execute(translated)
            cols = [c[0] for c in cur.description]
            return cur, cols

        def _next_chunk(
            cur: "RedshiftCursor",
        ) -> list[tuple[Any, ...]]:
            """pull the next batch from the cursor; sync."""
            return list(cur.fetchmany())

        try:
            cursor, col_names = await self._bridge.to_thread_with_cancel(
                _open_cursor,
                cancel_cb=conn.close,
            )
            # bind cursor into a non-Optional local so the inner
            # lambda's mypy inference doesn't trip on the | None type
            # of the outer ``cursor`` variable.
            live_cursor = cursor

            def _pull_next() -> list[tuple[Any, ...]]:
                return _next_chunk(live_cursor)

            while True:
                chunk = await self._bridge.to_thread_with_cancel(
                    _pull_next,
                    cancel_cb=conn.close,
                )
                if not chunk:
                    break
                for row in chunk:
                    yield dict(zip(col_names, row))
        except asyncio.CancelledError:
            # cancel-cb in to_thread_with_cancel already closed the
            # connection; flag for cleanup.
            checkout.poisoned = True
            raise
        except Exception:
            checkout.poisoned = True
            raise
        finally:
            if cursor is not None:
                with self._suppress_close():
                    await self._bridge.to_thread_with_cancel(
                        cursor.close,
                        cancel_cb=lambda: None,
                    )
            # same release path as every other acquire site: close the
            # streaming transaction before the connection can be cached,
            # then release (clean) or evict (poisoned). a streamed
            # SELECT holds a snapshot exactly like a materialized one.
            await self._finish_checkout(checkout)
            # release the open-connection permit AFTER the connection is back in
            # the cache (clean) or closed (poisoned), so a waiting caller that
            # wakes finds the cached connection to reuse rather than opening anew.
            self._connection_semaphore.release()

    # -------------------------------------------------------------------
    # Driver ABC: introspection surface
    # -------------------------------------------------------------------

    @traced
    @_observed(driver_type="redshift")
    async def list_tables(self, schemas: list[str]) -> list[TableRow]:
        """list tables in the schema allow-list using pg-compatible SQL.

        :param schemas: schema-name allow-list; empty list returns no rows
        :ptype schemas: list[str]
        :return: :class:`TableRow` dicts
        :rtype: list[TableRow]
        :raises RuntimeError: if the driver was previously closed
        """
        if self._closed:
            raise RuntimeError("RedshiftDriver is closed")

        sql = _REDSHIFT_TABLES_SQL_TEMPLATE.format(placeholders=_build_in_clause(len(schemas)))
        params = tuple(schemas)

        def _do_sync(
            conn: RedshiftConnection,
        ) -> list[TableRow]:
            cursor = conn.cursor()
            try:
                # empty allow-list: skip the round-trip + return [].
                # Redshift's ``IN ()`` is a parse error so we MUST
                # guard at the python level.
                if not params:
                    return []
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                cols = [c[0] for c in cursor.description]
                dicts = [dict(zip(cols, row)) for row in rows]
                return [
                    TableRow(
                        table_schema=r["table_schema"],
                        table_name=r["table_name"],
                    )
                    for r in dicts
                ]
            finally:
                cursor.close()

        async def _op(conn: RedshiftConnection) -> Any:
            return await self._bridge.to_thread_with_cancel(
                lambda: _do_sync(conn),
                cancel_cb=conn.close,
            )

        result: list[TableRow] = await self._acquire_and_run(_op)
        return result

    @traced
    @_observed(driver_type="redshift")
    async def list_columns(self, schemas: list[str]) -> list[ColumnRow]:
        """list columns for every table in the schema allow-list.

        ``is_nullable`` is preserved as the raw warehouse string
        (``'YES'`` / ``'NO'``) -- never normalized to bool. the
        Tier-2 hash depends on byte-equality with the warehouse-side
        MD5 in :data:`_REDSHIFT_TABLE_HASHES_SQL`.

        THIS IS THE METHOD whose timeout drove the whole datasource
        migration; on Redshift's ``reporting_prod`` schema this call
        returns ~6000 rows in <60s with ``redshift_connector`` where
        ``asyncpg`` never completes.

        :param schemas: schema-name allow-list
        :ptype schemas: list[str]
        :return: :class:`ColumnRow` dicts
        :rtype: list[ColumnRow]
        :raises RuntimeError: if the driver was previously closed
        """
        if self._closed:
            raise RuntimeError("RedshiftDriver is closed")

        sql = _REDSHIFT_COLUMNS_SQL_TEMPLATE.format(placeholders=_build_in_clause(len(schemas)))
        params = tuple(schemas)

        def _do_sync(
            conn: RedshiftConnection,
        ) -> list[ColumnRow]:
            cursor = conn.cursor()
            try:
                if not params:
                    return []
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                cols = [c[0] for c in cursor.description]
                dicts = [dict(zip(cols, row)) for row in rows]
                return [
                    ColumnRow(
                        table_schema=r["table_schema"],
                        table_name=r["table_name"],
                        column_name=r["column_name"],
                        data_type=r["data_type"],
                        is_nullable=r["is_nullable"],
                        ordinal_position=r["ordinal_position"],
                    )
                    for r in dicts
                ]
            finally:
                cursor.close()

        async def _op(conn: RedshiftConnection) -> Any:
            return await self._bridge.to_thread_with_cancel(
                lambda: _do_sync(conn),
                cancel_cb=conn.close,
            )

        result: list[ColumnRow] = await self._acquire_and_run(_op)
        return result

    @traced
    @_observed(driver_type="redshift")
    async def table_hashes(self, schemas: list[str]) -> dict[tuple[str, str], str]:
        """per-table MD5 over the column shape (Tier-2 change-probe).

        the warehouse-side MD5 in :data:`_REDSHIFT_TABLE_HASHES_SQL`
        is byte-equivalent to the python-side ``_compute_column_hash``
        helper from ``datasource-task-02`` AND to the same SQL on
        :class:`AsyncpgDriver`. equality across drivers is the
        cross-engine invariant that lets the Tier-2 probe live in
        Hub agnostic of which warehouse it's hashing.

        :param schemas: schema-name allow-list
        :ptype schemas: list[str]
        :return: mapping of ``(schema, table)`` -> column-shape hex digest
        :rtype: dict[tuple[str, str], str]
        :raises RuntimeError: if the driver was previously closed
        """
        if self._closed:
            raise RuntimeError("RedshiftDriver is closed")

        sql = _REDSHIFT_TABLE_HASHES_SQL_TEMPLATE.format(placeholders=_build_in_clause(len(schemas)))
        params = tuple(schemas)

        def _do_sync(
            conn: RedshiftConnection,
        ) -> dict[tuple[str, str], str]:
            cursor = conn.cursor()
            try:
                if not params:
                    return {}
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                cols = [c[0] for c in cursor.description]
                dicts = [dict(zip(cols, row)) for row in rows]
                return {(r["table_schema"], r["table_name"]): r["column_hash"] for r in dicts}
            finally:
                cursor.close()

        async def _op(conn: RedshiftConnection) -> Any:
            return await self._bridge.to_thread_with_cancel(
                lambda: _do_sync(conn),
                cancel_cb=conn.close,
            )

        result: dict[tuple[str, str], str] = await self._acquire_and_run(_op)
        return result

    # -------------------------------------------------------------------
    # Driver ABC: lifecycle
    # -------------------------------------------------------------------

    @traced
    @_observed(driver_type="redshift")
    async def test_connection(self) -> None:
        """cheapest possible round-trip; verifies credentials + reachability.

        any failure surfaces as :class:`DriverConnectError`; the
        original ``redshift_connector`` exception is suppressed via
        ``from None`` so its message (which can carry password
        fragments) doesn't reach loggers.

        :return: nothing; raises on failure
        :rtype: None
        :raises RuntimeError: if the driver was previously closed
        :raises DriverConnectError: on any backend failure (auth /
            network / timeout); message carries host/port/database
            but never the password value
        """
        if self._closed:
            raise RuntimeError("RedshiftDriver is closed")

        def _do_sync(conn: RedshiftConnection) -> int:
            cursor = conn.cursor()
            try:
                cursor.execute(_PING_SQL)
                row = cursor.fetchone()
                # row is a tuple; first column is the literal 1
                return int(row[0]) if row else 0
            finally:
                cursor.close()

        async def _op(conn: RedshiftConnection) -> Any:
            return await self._bridge.to_thread_with_cancel(
                lambda: _do_sync(conn),
                cancel_cb=conn.close,
            )

        try:
            await self._acquire_and_run(_op)
        except DriverConnectError:
            # already sanitized; re-raise unchanged.
            raise
        except Exception:
            # sanitize: wrap any backend-side failure with
            # connection identity, break the cause chain.
            identity = self._connection_identity()
            raise DriverConnectError(f"connection failed for {identity}") from None

    @traced
    async def close(self) -> None:
        """release driver resources; idempotent.

        sets :attr:`_closed` first so any in-flight method re-entry
        sees the flag. drains the cache: every cached connection is
        closed inside a worker thread (so an unresponsive Redshift
        endpoint doesn't pin the event loop). finally closes the
        bridge (``shutdown(wait=False)`` per the contract).

        idempotent: second call returns immediately.

        :return: nothing
        :rtype: None
        """
        if self._closed:
            return
        self._closed = True
        # snapshot + drain under the lock so concurrent acquire/
        # release don't race with the drain.
        async with self._cache_lock:
            to_close = list(self._cache)
            self._cache.clear()
        for conn in to_close:
            with self._suppress_close():
                await self._bridge.to_thread_with_cancel(
                    conn.close,
                    cancel_cb=lambda: None,
                )
        # bridge close uses shutdown(wait=False) -- contract.
        await self._bridge.close()
        # the finalize is no longer useful; detach so it doesn't
        # try to close the (now-closed) connections at GC time.
        with self._suppress_close():
            self._finalize.detach()

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _connection_identity(self) -> str:
        """credential-free identity string for error messages.

        :return: ``host:port/database`` -- safe to log
        :rtype: str
        """
        cfg = self._config
        return f"{cfg.host}:{cfg.port}/{cfg.database}"


# log a module-level marker so operators can confirm from the log
# alone which driver module is loaded. DEBUG so production logging
# configurations stay quiet by default.
log.debug("threetears.datasources.drivers.redshift_driver module loaded")
