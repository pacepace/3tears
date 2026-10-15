"""a committed write leaves a mark in its own transaction, and a reconciler advances its table's generation from it.

A write to a table whose collection follows write generations commits to L3, and its generation is
advanced afterwards, as a separate step outside the transaction (``BaseCollection`` and the hub's L3
broker both do this). If that step fails, is cancelled, or the writer dies between the commit and
the advance, the row is committed and the generation is unmoved: every pod holding a recorded
absence for that key keeps answering "absent" -- for the revocation list, a revoked session still
answering "not revoked". Nothing in the epoch system could see a commit that never moved it.

So the database records the write itself. Each switched-on table carries a statement-level trigger
(:data:`WRITE_MARK_TRIGGER`) that inserts one row into :data:`WRITE_MARKS_TABLE` in the same
transaction as the write: the mark commits exactly when the write does, whoever the writer is (the
broker, a direct pool, a write buffer, a migration, a release that predates this). The
:class:`WriteMarkReconciler` drains the marks: it advances each marked table's generation once and
deletes exactly the marks it read. A writer that advanced after its commit, as before, costs a
second advance, which only makes followers re-read the table once more.

**Insert-only, not a per-table counter row.** A row per table that every write increments is one
row every concurrent writer to that table locks; on YugabyteDB's snapshot isolation a second
concurrent writer aborts (40001) rather than waits. A mark is an insert of a fresh row, so writers
never contend, and the queue is bounded by draining it, not by a lifetime.

**Several reconcilers at once are safe.** Two replicas that read the same page both advance the same
tables, which costs one extra advance each, and both delete the same ids, which is idempotent. A
:class:`~threetears.core.coordination.KVLease` keeps it to one at a time when the caller supplies one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from threetears.observe import get_logger
from threetears.observe.metrics import counter, gauge

from threetears.core.collections.generation import GenerationSource
from threetears.core.coordination.lease import KVLease, LeaseUnavailable
from threetears.core.sql_fragments import quote_identifier

__all__ = [
    "MARK_FUNCTION",
    "WRITE_MARKS_TABLE",
    "WRITE_MARK_TRIGGER",
    "MarkPass",
    "WriteMarkReconciler",
    "install_write_mark_triggers",
    "tables_missing_the_trigger",
    "write_mark_trigger_sql",
    "write_marks_ddl",
]

log = get_logger(__name__)

#: the queue of committed writes not yet reconciled
WRITE_MARKS_TABLE: Final[str] = "collection_write_marks"

#: the trigger function every switched-on table's trigger calls
MARK_FUNCTION: Final[str] = "threetears_mark_collection_write"

#: the trigger's name on every switched-on table
WRITE_MARK_TRIGGER: Final[str] = "threetears_collection_write_mark"

#: marks read per page, and pages drained per pass before the pass reports itself behind
_PAGE_SIZE: Final[int] = 500
_MAX_PAGES_PER_PASS: Final[int] = 20

_DRAIN_BEHIND = counter(
    "collection_write_marks_drain_behind_total",
    description="reconcile passes that ended with write marks still queued",
)
_ADVANCE_FAILED = counter(
    "collection_write_marks_advance_failed_total",
    description="tables whose generation a reconcile pass could not advance; their marks are kept",
    label_names=("table",),
)
_APPLIED = counter(
    "collection_write_marks_applied_total",
    description="write marks applied (their table advanced) and deleted",
)
_PENDING = gauge(
    "collection_write_marks_pending",
    description="write marks left queued at the end of the last reconcile pass, at least",
)


def write_marks_ddl(schema: str) -> tuple[str, ...]:
    """the statements that create the marks table and the trigger function in ``schema``, idempotently.

    The function names the marks table by its schema, so a trigger on a table in any other schema
    (an agent's) writes the one queue.

    :param schema: the schema that holds the queue (the platform schema)
    :ptype schema: str
    :return: the statements, in order; run them one at a time (YugabyteDB runs no DDL in a transaction)
    :rtype: tuple[str, ...]
    """
    marks = f"{quote_identifier(schema)}.{quote_identifier(WRITE_MARKS_TABLE)}"
    function = f"{quote_identifier(schema)}.{quote_identifier(MARK_FUNCTION)}"
    return (
        f"CREATE TABLE IF NOT EXISTS {marks} ("
        "id BIGSERIAL PRIMARY KEY, "
        "table_name TEXT NOT NULL, "
        "date_created TIMESTAMPTZ NOT NULL DEFAULT now())",
        f"CREATE OR REPLACE FUNCTION {function}() RETURNS trigger LANGUAGE plpgsql AS $mark$\n"
        "BEGIN\n"
        f"  INSERT INTO {marks} (table_name) VALUES (TG_TABLE_NAME);\n"
        "  RETURN NULL;\n"
        "END\n"
        "$mark$",
    )


def write_mark_trigger_sql(*, marks_schema: str, table_schema: str, table: str) -> str:
    """the statement that gives one table its write-mark trigger.

    Statement-level: one mark per write statement, however many rows it touched (a statement that
    touched none marks too, which costs one spurious advance).

    :param marks_schema: the schema holding the queue and its function
    :ptype marks_schema: str
    :param table_schema: the table's schema
    :ptype table_schema: str
    :param table: the table
    :ptype table: str
    :return: the ``CREATE TRIGGER`` statement
    :rtype: str
    """
    return (
        f"CREATE TRIGGER {quote_identifier(WRITE_MARK_TRIGGER)} "
        f"AFTER INSERT OR UPDATE OR DELETE ON {quote_identifier(table_schema)}.{quote_identifier(table)} "
        f"FOR EACH STATEMENT EXECUTE FUNCTION {quote_identifier(marks_schema)}.{quote_identifier(MARK_FUNCTION)}()"
    )


class SqlPool(Protocol):
    """what the installer and the reconciler need of an L3 pool (an ``asyncpg.Pool`` is one)."""

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """rows of ``query``."""
        ...

    async def execute(self, query: str, *args: Any) -> str:
        """run ``query``; its status."""
        ...


_MISSING_SQL: Final[str] = """
SELECT t.table_schema AS table_schema, t.table_name AS table_name
FROM information_schema.tables t
WHERE t.table_type = 'BASE TABLE' AND t.table_name = ANY($1::text[]) AND NOT EXISTS (
    SELECT 1 FROM pg_trigger g
    JOIN pg_class c ON c.oid = g.tgrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE g.tgname = $2 AND c.relname = t.table_name AND n.nspname = t.table_schema
)
ORDER BY t.table_schema, t.table_name
"""


async def tables_missing_the_trigger(pool: SqlPool, tables: Iterable[str]) -> list[tuple[str, str]]:
    """every ``(schema, table)`` of the named tables, in any schema, that has no write-mark trigger.

    :param pool: the L3 pool
    :ptype pool: SqlPool
    :param tables: the switched-on table names
    :ptype tables: Iterable[str]
    :return: the tables without one
    :rtype: list[tuple[str, str]]
    """
    names = sorted(set(tables))
    if not names:
        return []
    rows = await pool.fetch(_MISSING_SQL, names, WRITE_MARK_TRIGGER)
    return [(row["table_schema"], row["table_name"]) for row in rows]


async def install_write_mark_triggers(pool: SqlPool, *, marks_schema: str, tables: Iterable[str]) -> list[str]:
    """give every switched-on table, in every schema, its write-mark trigger; say which got one.

    Safe to run from several replicas at once: a trigger another one created first is taken as
    installed.

    :param pool: the L3 pool, allowed DDL
    :ptype pool: SqlPool
    :param marks_schema: the schema holding the queue and its function
    :ptype marks_schema: str
    :param tables: the switched-on table names
    :ptype tables: Iterable[str]
    :return: ``schema.table`` for each trigger this call created
    :rtype: list[str]
    """
    created: list[str] = []
    for schema, table in await tables_missing_the_trigger(pool, tables):
        try:
            await pool.execute(write_mark_trigger_sql(marks_schema=marks_schema, table_schema=schema, table=table))
        # prawduct:allow prawduct/broad-except -- a concurrent replica's trigger or a table dropped meanwhile; the next pass re-reads what is missing
        except Exception as exc:
            if "already exists" in str(exc):
                continue
            log.error(
                "could not give a switched-on table its write-mark trigger; its writes advance only after their commit",
                extra={"extra_data": {"schema": schema, "table": table, "error": f"{type(exc).__name__}: {exc}"}},
            )
            continue
        created.append(f"{schema}.{table}")
    if created:
        log.info("write-mark triggers installed", extra={"extra_data": {"tables": created}})
    return created


@dataclass(frozen=True)
class MarkPass:
    """what one reconcile pass did.

    :ivar applied: marks whose table was advanced, and which were deleted
    :ivar advanced: the tables advanced
    :ivar failed: the tables whose advance failed; their marks stay for the next pass
    :ivar behind: whether marks were still queued when the pass stopped
    :ivar skipped: whether another reconciler held the lease, so this one did nothing
    """

    applied: int
    advanced: tuple[str, ...]
    failed: tuple[str, ...]
    behind: bool
    skipped: bool = False


_SKIPPED: Final[MarkPass] = MarkPass(applied=0, advanced=(), failed=(), behind=False, skipped=True)

#: the lease key one reconciler holds for a pass, and how long an abandoned hold lasts
_LEASE_KEY: Final[str] = "collection-write-marks-drain"
_LEASE_TTL_SECONDS: Final[int] = 60


class WriteMarkReconciler:
    """drains the write-mark queue: advances each marked table once, deletes exactly the marks it read.

    :param pool: the L3 pool holding the queue
    :ptype pool: SqlPool
    :param generation_source: advances a table's generation (the epoch bucket's writer)
    :ptype generation_source: GenerationSource
    :param marks_schema: the schema holding the queue
    :ptype marks_schema: str
    :param tables: the switched-on table names, read at each pass; the triggers follow them
    :ptype tables: Callable[[], Iterable[str]]
    :param page_size: marks read per page
    :ptype page_size: int
    :param max_pages: pages drained per pass before the pass reports itself behind
    :ptype max_pages: int
    :param lease: one pass at a time across replicas; a replica that finds it held skips its pass.
        ``None`` runs every pass, which is also safe (a page read twice is advanced twice and deleted
        once)
    :ptype lease: KVLease | None
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
    ) -> None:
        if page_size < 1 or max_pages < 1:
            raise ValueError("a reconcile pass reads at least one page of at least one mark")
        self._pool = pool
        self._source = generation_source
        self._marks = f"{quote_identifier(marks_schema)}.{quote_identifier(WRITE_MARKS_TABLE)}"
        self._marks_schema = marks_schema
        self._tables = tables
        self._page_size = page_size
        self._max_pages = max_pages
        self._lease = lease

    async def install_triggers(self) -> list[str]:
        """give every switched-on table that lacks one its trigger.

        :return: ``schema.table`` for each trigger created
        :rtype: list[str]
        """
        return await install_write_mark_triggers(self._pool, marks_schema=self._marks_schema, tables=self._tables())

    async def run_pass(self) -> MarkPass:
        """under the lease when there is one: install any missing trigger, then drain up to the page budget.

        :return: what the pass did
        :rtype: MarkPass
        """
        if self._lease is None:
            return await self._drain()
        try:
            handle = await self._lease.acquire(_LEASE_KEY, ttl_seconds=_LEASE_TTL_SECONDS, max_wait_seconds=0)
        except LeaseUnavailable:
            return _SKIPPED
        try:
            return await self._drain()
        finally:
            await handle.release()

    async def _drain(self) -> MarkPass:
        """install any missing trigger, then drain up to the page budget.

        :return: what the pass did
        :rtype: MarkPass
        """
        await self.install_triggers()
        applied = 0
        advanced: set[str] = set()
        failed: set[str] = set()
        behind = False
        for page_number in range(self._max_pages):
            page = await self._pool.fetch(
                f"SELECT id, table_name FROM {self._marks} ORDER BY id LIMIT $1", self._page_size
            )
            if not page:
                break
            # every mark read here committed before the read, so an advance after the read covers it;
            # only the marks of a table advanced for THIS page are deleted
            done, page_failed = await self._advance(page)
            advanced |= done
            failed |= page_failed
            ids = [int(row["id"]) for row in page if row["table_name"] in done]
            if ids:
                await self._pool.execute(f"DELETE FROM {self._marks} WHERE id = ANY($1::bigint[])", ids)
                applied += len(ids)
            left_in_page = len(page) - len(ids)
            if len(page) < self._page_size:
                behind = left_in_page > 0
                break
            if page_number == self._max_pages - 1 or left_in_page == len(page):
                # out of budget, or a page none of whose tables could be advanced: stop and say so
                behind = True
                break
        _APPLIED.inc(applied)
        if behind:
            _DRAIN_BEHIND.inc()
            remaining = await self._pool.fetch(f"SELECT count(*) AS n FROM {self._marks}")
            pending = int(remaining[0]["n"]) if remaining else 0
            _PENDING.set(pending)
            log.warning(
                "write marks are still queued after a reconcile pass; followers of these tables see the "
                "writes only as the drain catches up",
                extra={"extra_data": {"pending": pending, "failed_tables": sorted(failed), "applied": applied}},
            )
        else:
            _PENDING.set(0)
        return MarkPass(applied=applied, advanced=tuple(sorted(advanced)), failed=tuple(sorted(failed)), behind=behind)

    async def _advance(self, page: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
        """advance, once each, every table the page marks.

        :param page: the marks read
        :ptype page: Sequence[Mapping[str, Any]]
        :return: the tables advanced, and those whose advance failed
        :rtype: tuple[set[str], set[str]]
        """
        done: set[str] = set()
        failed: set[str] = set()
        for table in sorted({str(row["table_name"]) for row in page}):
            try:
                await self._source.advance(table)
            # prawduct:allow prawduct/broad-except -- a table that cannot be advanced keeps its marks for the next pass, and is counted
            except Exception as exc:
                failed.add(table)
                _ADVANCE_FAILED.labels(table=table).inc()
                log.error(
                    "a marked table's write generation could not be advanced; its marks are kept",
                    extra={"extra_data": {"table": table, "error": f"{type(exc).__name__}: {exc}"}},
                )
                continue
            done.add(table)
        return done, failed
