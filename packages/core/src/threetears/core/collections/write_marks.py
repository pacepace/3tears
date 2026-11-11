"""a committed write leaves a mark in its own transaction, and a reconciler advances its table's generation from it.

A write to a table whose collection follows write generations commits to L3, and its generation is
advanced afterwards, as a separate step outside the transaction (``BaseCollection`` and the hub's L3
broker both do this). If that step fails, is cancelled, or the writer dies between the commit and
the advance, the row is committed and the generation is unmoved: every pod holding a recorded
absence for that key keeps answering "absent" -- for the revocation list, a revoked session still
answering "not revoked". Nothing in the epoch system could see a commit that never moved it.

So the database records the write itself. Each switched-on table carries a row trigger
(:data:`WRITE_MARK_TRIGGER`) that inserts one row into :data:`WRITE_MARKS_TABLE` in the same
transaction as the write, at the transaction's first changed row of the table: the mark commits exactly
when the write does, whoever the writer is, and a statement that changed no row marks nothing. The
:class:`WriteMarkReconciler` drains the marks: it advances each marked table's generation once and
deletes exactly the marks it read. A writer that advanced after its commit, as before, costs a
second advance, which only makes followers re-read the table once more. Every process that owns a
database holding switched-on tables runs one over its own database (the hub, identity-core).

**Insert-only, not a per-table counter row.** A row per table that every write increments is one
row every concurrent writer to that table locks: they queue behind each other under read committed,
and abort (40001) under snapshot isolation. A mark is an insert of a fresh row, so writers never
contend, and the queue is bounded by draining it, not by a lifetime. Its key stays hash-sharded, so
inserts spread over tablets; the drain needs no order (it deletes exactly the ids it read), so no
ordered index adds a hot tail to every write.

**Installing is DDL; draining is not.** :func:`install_write_mark_triggers` checks the catalogue
first and takes the database-wide DDL lock
(:func:`~threetears.core.data.migrations.ddl_lock.database_ddl_lock`) only when a trigger is missing,
with a bounded wait; a busy lock defers the install. :meth:`WriteMarkReconciler.run_pass` issues no
DDL, so the drain never waits behind it. Callers run them as separate tasks. The install is the backstop
for one schema, the one whose declarations it is given: a same-named table in another schema is another owner's,
reported and never switched on by its name; each schema's own migration runs give its tables their triggers.

**Several reconcilers at once are safe.** Two that read the same page both advance the same tables,
which costs one extra advance each, and both delete the same ids, which is idempotent. A
:class:`~threetears.core.coordination.KVLease` keeps it to one at a time when the caller supplies one.

**One failing table never stalls the rest.** A table whose advance fails keeps its marks, is counted,
and is excluded from the rest of the pass's pages, so every other table's marks are still drained.

**Worst-case repair.** A commit whose own advance was lost is seen by a follower within the pass
interval, plus the lease's lifetime when the replica holding it died mid-pass, plus the follower's
own catch-up. ``TRUNCATE`` fires no row trigger and leaves no mark.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Final, Protocol

from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter, gauge

from threetears.core.collections.change_notes import NotedRows, noted_rows_since
from threetears.core.collections.epoch_change import ChangedRow, advance_recording
from threetears.core.collections.generation import GenerationSource, tables_written_l2_first, tenancy_of_table
from threetears.core.coordination.lease import KVLease, LeaseUnavailable
from threetears.core.ddl.ddl_lock import DdlLockPolicy, database_ddl_lock
from threetears.core.ddl.errors import DdlLockTimeoutError
from threetears.core.ddl.session import ConnectionSession
from threetears.core.ddl.change_notes import WHOLE_COPY_ROWS
from threetears.core.ddl.write_marks import (
    MARK_FUNCTION,
    WRITE_MARK_TRIGGER,
    WRITE_MARKS_TABLE,
    delete_marks_sql,
    marks_page_sql,
    write_mark_trigger_sql,
    write_marks_ddl,
)
from threetears.core.sql_fragments import quote_identifier
from threetears.core.tenancy import TENANCY_UNDECLARED, schema_tenant

__all__ = [
    "INSTALL_LOCK_POLICY",
    "MARK_FUNCTION",
    "WRITE_MARKS_TABLE",
    "WRITE_MARK_TRIGGER",
    "MarkPass",
    "QueueState",
    "WriteMarkReconciler",
    "drop_write_mark_triggers",
    "install_write_mark_triggers",
    "observe_write_marks",
    "tables_missing_the_trigger",
    "tables_switched_by_trigger",
    "write_mark_trigger_sql",
    "write_marks_ddl",
]

log = get_logger(__name__)

#: how long an install waits for the DDL lock before deferring to its next run: never long, since a
#: migration or an index build holding the lock may hold it for minutes
INSTALL_LOCK_POLICY: Final[DdlLockPolicy] = DdlLockPolicy(poll_interval=0.5, max_wait=5.0)

#: marks read per page, and pages drained per pass before the pass reports itself behind
_PAGE_SIZE: Final[int] = 500
_MAX_PAGES_PER_PASS: Final[int] = 20

#: the most marks the queue gauges count; above it they report "at least" this many
_OBSERVE_CAP: Final[int] = 10_000

#: the lease key one reconciler holds for a pass, and how long a dead holder's hold lasts
_LEASE_KEY: Final[str] = "collection-write-marks-drain"
_LEASE_TTL_SECONDS: Final[int] = 15

_PASSES = counter(
    "collection_write_marks_passes_total",
    description="reconcile passes by outcome: drained, behind, skipped (another replica held the lease), failed",
    label_names=("outcome",),
)
_ISOLATED = counter(
    "collection_write_marks_advance_failed_total",
    description="tables a pass could not advance; their marks are kept and the rest of the pass drains around them",
    label_names=("table",),
)
_APPLIED = counter(
    "collection_write_marks_applied_total",
    description="write marks applied (their table advanced) and deleted",
)
_INSTALLS_DEFERRED = counter(
    "collection_write_marks_installs_deferred_total",
    description="trigger installs deferred because the database's DDL lock was busy",
)
_MISSING_ELSEWHERE = gauge(
    "collection_write_marks_missing_elsewhere",
    description="switched-on table names found without the write-mark trigger in a schema other than the one the "
    "install backstop serves: not installed there (a same-named table of another owner is not switched on by a name), "
    "so nonzero means a schema its migrations never reached, or another owner's table of that name",
)
_PENDING = gauge(
    "collection_write_marks_pending",
    description="write marks queued, read on every pass by every replica; capped at the observe cap",
)
_OLDEST_AGE = gauge(
    "collection_write_marks_oldest_age_seconds",
    description="age of the oldest queued write mark (of the capped sample), read on every pass; 0 when empty",
)


#: the most rows one reconciled table's epoch names from the change notes; more are advanced as keys unknown. A
#: copy of more rows is noted whole (:data:`~threetears.core.ddl.change_notes.WHOLE_COPY_ROWS`, the same count)
_MAX_NOTED_ROWS: Final = WHOLE_COPY_ROWS

_KEYS_UNKNOWN = counter(
    "collection_write_marks_keys_unknown_total",
    description="reconciled tables whose missed writes the change notes could not name, advanced as keys unknown "
    "(every follower is behind on the table, never told to drop it), by why: not_covered, no_notes, "
    "too_many_rows, notes_unavailable",
    label_names=("table", "reason"),
)


class SqlConnection(Protocol):
    """one database connection (an asyncpg ``Connection``)."""

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """rows of ``query``."""
        ...

    async def execute(self, query: str, *args: Any) -> str:
        """run ``query``; its status."""
        ...


class SqlPool(SqlConnection, Protocol):
    """what the installer and the reconciler need of an L3 pool (an ``asyncpg.Pool`` is one)."""

    def acquire(self) -> AbstractAsyncContextManager[Any]:
        """one connection, for a statement sequence that must share a session (the DDL lock's)."""
        ...


_MISSING_SQL: Final[str] = """
SELECT t.table_schema AS table_schema, t.table_name AS table_name
FROM information_schema.tables t
WHERE t.table_type = 'BASE TABLE' AND t.table_name::text = ANY($1::text[]) AND NOT EXISTS (
    SELECT 1 FROM pg_trigger g
    JOIN pg_class c ON c.oid = g.tgrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE g.tgname = $2 AND c.relname = t.table_name AND n.nspname = t.table_schema
    AND (g.tgtype & 1) = 1
)
ORDER BY t.table_schema, t.table_name
"""

_PRESENT_SQL: Final[str] = """
SELECT n.nspname AS table_schema, c.relname AS table_name
FROM pg_trigger g
JOIN pg_class c ON c.oid = g.tgrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE g.tgname = $1
ORDER BY n.nspname, c.relname
"""


_SWITCHED_SQL: Final[str] = """
SELECT c.relname AS table_name
FROM pg_trigger g
JOIN pg_class c ON c.oid = g.tgrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE g.tgname = $1 AND n.nspname::text = $2 AND c.relname::text = ANY($3::text[])
"""


async def tables_switched_by_trigger(pool: SqlConnection, schema: str, tables: Iterable[str]) -> frozenset[str]:
    """the named tables of ``schema`` that carry the write-mark trigger: switched on by the database itself.

    A table is switched on by its declaration (a collection class with ``write_generation``), and the migration that
    creates it gives it the trigger (:meth:`~threetears.core.data.migrations.registry.MigrationRegistry.mark_write_generation`).
    A writer that imports no class for the table -- the hub's broker, committing a tool pod's write to the pod's own
    tables -- reads the switch here instead: the trigger is installed under the database's DDL lock, so a pod cannot
    claim a table is switched on without it being so. A catalogue read: no DDL, no lock.

    :param pool: the L3 pool, or one connection
    :ptype pool: SqlConnection
    :param schema: the schema the tables live in
    :ptype schema: str
    :param tables: the table names
    :ptype tables: Iterable[str]
    :return: those that carry the trigger
    :rtype: frozenset[str]
    """
    names = sorted(set(tables))
    if not names:
        return frozenset()
    rows = await pool.fetch(_SWITCHED_SQL, WRITE_MARK_TRIGGER, schema, names)
    return frozenset(str(row["table_name"]) for row in rows)


async def tables_missing_the_trigger(
    pool: SqlConnection, tables: Iterable[str], *, schema: str | None = None
) -> list[tuple[str, str]]:
    """every ``(schema, table)`` of the named tables that has no write-mark trigger: in ``schema``, or in any.

    A catalogue read: no DDL, no lock.

    :param pool: the L3 pool, or one connection
    :ptype pool: SqlConnection
    :param tables: the switched-on table names
    :ptype tables: Iterable[str]
    :param schema: the one schema to look in; None for every schema
    :ptype schema: str | None
    :return: the tables without one
    :rtype: list[tuple[str, str]]
    """
    names = sorted(set(tables))
    if not names:
        return []
    rows = await pool.fetch(_MISSING_SQL, names, WRITE_MARK_TRIGGER)
    return [(row["table_schema"], row["table_name"]) for row in rows if schema is None or row["table_schema"] == schema]


async def install_write_mark_triggers(
    pool: SqlPool,
    *,
    marks_schema: str,
    tables: Iterable[str],
    table_schema: str | None = None,
    policy: DdlLockPolicy = INSTALL_LOCK_POLICY,
) -> list[str]:
    """give every switched-on table of the declaring schema its write-mark trigger; say which got one.

    The backstop for the schema whose declarations ``tables`` are (``table_schema``; the queue's own schema when
    None): a table there created before the trigger existed, or whose migration run could not give it one. Only
    that schema: a table is switched on by its owner's declaration, and a table of the same name in another schema
    (a tool pod's own table) is another owner's, switched on or off by that owner's own migrations
    (:func:`~threetears.core.data.migrations.write_marks.give_switched_tables_their_trigger` on every run in its
    schema). A same-named table without the trigger in another schema is reported, never installed: a log line and
    the ``collection_write_marks_missing_elsewhere`` gauge, so a schema its migrations never reached shows.

    Reads the catalogue first and returns at once when nothing is missing. Otherwise takes the
    database-wide DDL lock on one connection, with ``policy``'s bounded wait, reads again under it,
    and creates what is still missing. A lock that stays busy defers the install to the next call,
    counted; nothing is raised. Safe to run from several replicas at once.

    :param pool: the L3 pool, allowed DDL
    :ptype pool: SqlPool
    :param marks_schema: the schema holding the queue and its function
    :ptype marks_schema: str
    :param tables: the switched-on table names, as ``table_schema``'s owner declares them
    :ptype tables: Iterable[str]
    :param table_schema: the schema the tables are declared in; None for ``marks_schema``
    :ptype table_schema: str | None
    :param policy: how long to wait for the DDL lock
    :ptype policy: DdlLockPolicy
    :return: ``schema.table`` for each trigger this call created
    :rtype: list[str]
    """
    # an L2-first table advances before its L3 write, so a mark on it would only ever be covered
    names = sorted(set(tables) - tables_written_l2_first())
    scope = table_schema or marks_schema
    unmarked = await tables_missing_the_trigger(pool, names)
    elsewhere = [f"{schema}.{table}" for schema, table in unmarked if schema != scope]
    _MISSING_ELSEWHERE.set(len(elsewhere))
    if elsewhere:
        log.info(
            "switched-on table names without the write-mark trigger outside the declaring schema; not installed "
            "there: each is another owner's table, or a schema its migrations never reached",
            extra={"extra_data": {"declaring_schema": scope, "tables": elsewhere}},
        )
    if len(elsewhere) == len(unmarked):
        return []
    created: list[str] = []
    async with pool.acquire() as conn:
        try:
            async with database_ddl_lock(ConnectionSession(conn), policy):
                missing = await tables_missing_the_trigger(conn, names, schema=scope)
                if missing:
                    # the function the row trigger calls, replacing the one that marked every statement
                    await conn.execute(write_marks_ddl(marks_schema)[1])
                for schema, table in missing:
                    try:
                        # 0.78.0's statement trigger of the same name goes first, so the two never both mark
                        await conn.execute(
                            f"DROP TRIGGER IF EXISTS {quote_identifier(WRITE_MARK_TRIGGER)} "
                            f"ON {quote_identifier(schema)}.{quote_identifier(table)}"
                        )
                        await conn.execute(
                            write_mark_trigger_sql(marks_schema=marks_schema, table_schema=schema, table=table)
                        )
                    # prawduct:allow prawduct/broad-except -- a table dropped meanwhile, or a missing function; logged, and the next install retries
                    except Exception as exc:
                        contain(
                            exc,
                            log=log,
                            what=(
                                "could not give a switched-on table its write-mark trigger; its writes advance only "
                                "after their commit until it has one"
                            ),
                            extra={
                                "schema": schema,
                                "table": table,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            level="error",
                        )
                        continue
                    created.append(f"{schema}.{table}")
        except DdlLockTimeoutError:
            _INSTALLS_DEFERRED.inc()
            log.info("write-mark trigger install deferred: the database's DDL lock is busy")
            return []
    if created:
        log.info("write-mark triggers installed", extra={"extra_data": {"tables": created}})
    return created


async def drop_write_mark_triggers(pool: SqlPool, *, policy: DdlLockPolicy | None = None) -> list[str]:
    """remove every write-mark trigger in the database, under the DDL lock: the rollback step.

    For a rollback to a release with no reconciler: without it the triggers keep queueing marks that
    nothing drains. The queue and its function stay; a roll forward installs the triggers again.

    :param pool: the L3 pool, allowed DDL
    :ptype pool: SqlPool
    :param policy: how long to wait for the DDL lock; the default waits until it is free
    :ptype policy: DdlLockPolicy | None
    :return: ``schema.table`` for each trigger dropped
    :rtype: list[str]
    """
    dropped: list[str] = []
    async with pool.acquire() as conn, database_ddl_lock(ConnectionSession(conn), policy):
        for row in await conn.fetch(_PRESENT_SQL, WRITE_MARK_TRIGGER):
            schema, table = row["table_schema"], row["table_name"]
            await conn.execute(
                f"DROP TRIGGER IF EXISTS {quote_identifier(WRITE_MARK_TRIGGER)} "
                f"ON {quote_identifier(schema)}.{quote_identifier(table)}"
            )
            dropped.append(f"{schema}.{table}")
    log.warning("write-mark triggers dropped", extra={"extra_data": {"tables": dropped}})
    return dropped


@dataclass(frozen=True)
class QueueState:
    """the queue as one read saw it.

    :ivar pending: marks queued, up to the observe cap
    :ivar at_least: whether the queue holds more than the cap
    :ivar oldest_age_seconds: the age of the oldest mark read, ``0.0`` when empty
    """

    pending: int
    at_least: bool
    oldest_age_seconds: float


async def observe_write_marks(pool: SqlConnection, *, marks_schema: str) -> QueueState:
    """read the queue's size and its oldest mark's age, bounded, and set the gauges.

    Run by every replica on every pass, whether or not it holds the drain lease, so no replica's
    gauge holds a value another replica has since changed.

    :param pool: the L3 pool
    :ptype pool: SqlConnection
    :param marks_schema: the schema holding the queue
    :ptype marks_schema: str
    :return: what it read
    :rtype: QueueState
    """
    marks = f"{quote_identifier(marks_schema)}.{quote_identifier(WRITE_MARKS_TABLE)}"
    rows = await pool.fetch(
        "SELECT count(*) AS n, extract(epoch FROM now() - min(date_created)) AS age "
        f"FROM (SELECT date_created FROM {marks} LIMIT $1) sample",
        _OBSERVE_CAP + 1,
    )
    pending = int(rows[0]["n"]) if rows else 0
    age = float(rows[0]["age"]) if rows and rows[0]["age"] is not None else 0.0
    state = QueueState(pending=min(pending, _OBSERVE_CAP), at_least=pending > _OBSERVE_CAP, oldest_age_seconds=age)
    _PENDING.set(state.pending)
    _OLDEST_AGE.set(state.oldest_age_seconds)
    return state


@dataclass(frozen=True)
class MarkPass:
    """what one reconcile pass did.

    :ivar applied: marks whose table was advanced, and which were deleted
    :ivar advanced: the tables advanced
    :ivar failed: the tables whose advance failed; their marks stay for the next pass
    :ivar behind: whether the pass ran out of its page budget with marks still to read
    :ivar skipped: whether another reconciler held the lease, so this one drained nothing
    :ivar queue: the queue as the pass read it
    """

    applied: int
    advanced: tuple[str, ...]
    failed: tuple[str, ...]
    behind: bool
    skipped: bool = False
    queue: QueueState | None = None


class WriteMarkReconciler:
    """drains the write-mark queue: advances each marked table once, deletes exactly the marks it read.

    :param pool: the L3 pool holding the queue
    :ptype pool: SqlPool
    :param generation_source: advances a table's generation (the epoch bucket's writer)
    :ptype generation_source: GenerationSource
    :param marks_schema: the schema holding the queue
    :ptype marks_schema: str
    :param tables: the switched-on table names, read at each install; the triggers follow them
    :ptype tables: Callable[[], Iterable[str]]
    :param page_size: marks read per page
    :ptype page_size: int
    :param max_pages: pages drained per pass before the pass reports itself behind
    :ptype max_pages: int
    :param lease: one pass at a time across replicas; a replica that finds it held skips its drain.
        ``None`` drains on every pass, which is also safe
    :ptype lease: KVLease | None
    :param notes_schema: the schema holding the change-note ring the marked writes' notes are read from;
        ``None`` is ``marks_schema``
    :ptype notes_schema: str | None
    :param tables_schema: the schema ``tables`` are declared in, the only one the install backstop gives triggers
        in (:func:`install_write_mark_triggers`); ``None`` is ``marks_schema``
    :ptype tables_schema: str | None
    """

    def __init__(
        self,
        pool: SqlPool,
        *,
        generation_source: GenerationSource,
        marks_schema: str,
        tables: Callable[[], Iterable[str]],
        page_size: int = _PAGE_SIZE,
        max_pages: int = _MAX_PAGES_PER_PASS,
        lease: KVLease | None = None,
        notes_schema: str | None = None,
        tables_schema: str | None = None,
    ) -> None:
        if page_size < 1 or max_pages < 1:
            raise ValueError("a reconcile pass reads at least one page of at least one mark")
        self._pool = pool
        self._source = generation_source
        self._marks_schema = marks_schema
        self._notes_schema = notes_schema or marks_schema
        self._tables_schema = tables_schema or marks_schema
        self._tables = tables
        self._page_size = page_size
        self._max_pages = max_pages
        self._lease = lease

    async def install_triggers(self) -> list[str]:
        """give every switched-on table of the declaring schema that lacks one its trigger: the backstop, under the
        DDL lock (:func:`install_write_mark_triggers`).

        :return: ``schema.table`` for each trigger created
        :rtype: list[str]
        """
        return await install_write_mark_triggers(
            self._pool, marks_schema=self._marks_schema, tables=self._tables(), table_schema=self._tables_schema
        )

    async def run_pass(self) -> MarkPass:
        """read the queue's state, then drain under the lease when there is one. Issues no DDL.

        :return: what the pass did
        :rtype: MarkPass
        :raises Exception: whatever the drain raised, after counting the pass as failed
        """
        try:
            queue = await observe_write_marks(self._pool, marks_schema=self._marks_schema)
            if self._lease is None:
                done = await self._drain()
            else:
                try:
                    handle = await self._lease.acquire(_LEASE_KEY, ttl_seconds=_LEASE_TTL_SECONDS, max_wait_seconds=0)
                except LeaseUnavailable:
                    _PASSES.labels(outcome="skipped").inc()
                    return MarkPass(applied=0, advanced=(), failed=(), behind=False, skipped=True, queue=queue)
                try:
                    done = await self._drain()
                finally:
                    await handle.release()
        except Exception:
            _PASSES.labels(outcome="failed").inc()
            raise
        _PASSES.labels(outcome="behind" if done.behind else "drained").inc()
        return MarkPass(
            applied=done.applied,
            advanced=done.advanced,
            failed=done.failed,
            behind=done.behind,
            queue=queue,
        )

    async def _drain(self) -> MarkPass:
        """drain up to the page budget, reading around every table that failed this pass.

        :return: what the drain did
        :rtype: MarkPass
        """
        applied = 0
        advanced: set[str] = set()
        failed: set[str] = set()
        behind = True
        started = time.monotonic()
        after = 0
        for _ in range(self._max_pages):
            # a keyset page of the queue after the last mark read: a table that failed this pass is passed over, its
            # marks left for the next pass
            page = await self._pool.fetch(marks_page_sql(self._marks_schema), self._page_size, after)
            if page:
                after = max(int(row["id"]) for row in page)
                wanted = [row for row in page if row["table_name"] not in failed]
            else:
                wanted = []
            if wanted:
                # every mark read here committed before the read, so an advance after the read covers it;
                # only the marks of a table advanced for THIS page are deleted
                done, page_failed = await self._advance(wanted)
                advanced |= done
                failed |= page_failed
                ids = [int(row["id"]) for row in wanted if row["table_name"] in done]
                if ids:
                    await self._pool.execute(delete_marks_sql(self._marks_schema), ids)
                    applied += len(ids)
            if len(page) < self._page_size:
                behind = False
                break
        _APPLIED.inc(applied)
        if behind or failed:
            log.warning(
                "write marks left queued after a reconcile pass; followers of these tables see the writes only "
                "as the drain catches up",
                extra={
                    "extra_data": {
                        "behind": behind,
                        "failed_tables": sorted(failed),
                        "applied": applied,
                        "seconds": round(time.monotonic() - started, 3),
                    }
                },
            )
        return MarkPass(applied=applied, advanced=tuple(sorted(advanced)), failed=tuple(sorted(failed)), behind=behind)

    async def _changes_of(self, table: str, since: Any) -> list[tuple[list[ChangedRow] | None, str | None, bool]]:
        """what the marked writes changed, from the change notes they left: each advance to record.

        A mark names its table, not its rows, but every committed write to a noted table also left a note of
        its keys, and the notes carry each row's tenant. The epoch names those rows, so followers evict them
        rather than dropping the table. A whole-table note (a truncate, a file load) is a real whole-table
        change, recorded as one under its own tenant; it covers that tenant's rows only, so another schema's
        rows in the same window are still named. When the notes cannot say (not covered, none, too many rows,
        no ring) the advance is keys unknown: every follower is behind on the table, never told to drop it,
        said at WARNING and counted on ``collection_write_marks_keys_unknown_total``. An agent-schema table's
        record goes under each schema it may be in (:meth:`_schema_tenants`), never ``_all``.

        :param table: the marked table
        :ptype table: str
        :param since: the oldest mark's time
        :ptype since: Any
        :return: ``(rows, tenant, keys_unknown)`` per advance: rows (``None`` for a rowless record), the rowless
            record's tenant, and whether its keys are unknown
        :rtype: list[tuple[list[ChangedRow] | None, str | None, bool]]
        """
        noted = await noted_rows_since(
            self._pool, notes_schema=self._notes_schema, table=table, since=since, cap=_MAX_NOTED_ROWS
        )
        if isinstance(noted, NotedRows):
            advances: list[tuple[list[ChangedRow] | None, str | None, bool]] = []
            if noted.rows:
                advances.append((list(noted.rows), None, False))
            advances.extend((None, tenant, False) for tenant in noted.whole_tenants)
            return advances
        _KEYS_UNKNOWN.labels(table=table, reason=noted).inc()
        log.warning(
            "a marked table's missed writes are advanced as keys unknown: the change notes cannot name their rows, "
            "so every follower is behind on the table until its next start",
            extra={"extra_data": {"table": table, "reason": noted}},
        )
        return [(None, tenant, True) for tenant in await self._schema_tenants(table)]

    async def _schema_tenants(self, table: str) -> tuple[str | None, ...]:
        """the tenants a rowless advance of ``table`` the notes could not name goes under.

        For an agent-schema table, every schema holding the table (the mark names no schema); ``_all`` for any
        other table.

        :param table: the table
        :ptype table: str
        :return: the tenants, ``None`` for ``_all``
        :rtype: tuple[str | None, ...]
        """
        kind = tenancy_of_table(table).kind
        if kind not in ("schema", TENANCY_UNDECLARED.kind):
            return (None,)
        rows = await self._pool.fetch(
            "SELECT table_schema::text AS table_schema FROM information_schema.tables WHERE table_name = $1 "
            "ORDER BY table_schema",
            table,
        )
        schemas = [str(row["table_schema"]) for row in rows]
        if kind != "schema":
            # a table no class here declares, switched on by its trigger alone: a tool pod's own table, which lives
            # in the pod's schema and whose pods read it there. One in the platform schema stays on _all
            if not schemas or self._marks_schema in schemas:
                return (None,)
        return tuple(schema_tenant(schema) for schema in schemas) or (None,)

    async def _advance(self, page: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
        """advance, once each, every table the page marks.

        :param page: the marks read
        :ptype page: Sequence[Mapping[str, Any]]
        :return: the tables advanced, and those whose advance failed
        :rtype: tuple[set[str], set[str]]
        """
        done: set[str] = set()
        failed: set[str] = set()
        l2_first = tables_written_l2_first()
        for table in sorted({str(row["table_name"]) for row in page}):
            if table in l2_first:
                # its epoch was advanced, naming its rows, before the row reached L3: the mark is covered,
                # and advancing it again would tell every follower the whole table changed
                done.add(table)
                continue
            try:
                since = min(row["date_created"] for row in page if str(row["table_name"]) == table)
                for rows, tenant, keys_unknown in await self._changes_of(table, since):
                    await advance_recording(self._source, table, rows, keys_unknown=keys_unknown, tenant=tenant)
            # prawduct:allow prawduct/broad-except -- a table that cannot be advanced keeps its marks for the next pass, is counted, and the pass drains around it
            except Exception as exc:
                failed.add(table)
                _ISOLATED.labels(table=table).inc()
                contain(
                    exc,
                    log=log,
                    what=(
                        "a marked table's write generation could not be advanced; its marks are kept and the pass "
                        "drains the other tables"
                    ),
                    extra={"table": table, "error": f"{type(exc).__name__}: {exc}"},
                    level="error",
                )
                continue
            done.add(table)
        return done, failed
