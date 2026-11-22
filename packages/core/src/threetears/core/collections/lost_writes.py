"""a committed write whose epoch advance was lost is announced again, from the change notes, with no L3 write a commit.

A write to a generation-switched table commits to L3 and is advanced after, outside its transaction. Three ways an
advance can be lost, and what announces each one again:

- **The advance failed or was cancelled, the writer alive.** :func:`~threetears.core.collections.epoch_change
  .advance_recording` hands it to the process's ledger (:mod:`threetears.core.utils.write_ledger`) before raising, and
  :meth:`WriteBackstop.repair_pass` makes it again, exactly as it was, until it lands.
- **The writer died between its commit and its advance.** Each writer process publishes its ledger's watermark W (the
  oldest start of a write that may still lack its advance) to a NATS KV key every :data:`_BEAT_SECONDS`
  (:meth:`WriteBackstop.beat`). The replica holding the recovery lease (:meth:`WriteBackstop.recover_pass`) takes a key
  whose beat is older than :data:`_STALE_SECONDS` for a dead writer: every switched table with change notes in
  ``[W - skew, min(beat + grace, now) + skew]`` is advanced from those notes (:func:`announce_window`), and the key is
  deleted. A process stopping cleanly deletes its own key after its repairs land (:meth:`WriteBackstop.stop`).
- **A data migration wrote rows** (no collection, no advance). The migration runner hands its run's window to
  :meth:`WriteBackstop.announce_window` (``MigrationRunner(announcer=...)``).

**How a window is announced** (:func:`_announce_table_window`). Every committed write to a noted table left a note of
each row's key in its own transaction, so the window's notes name every row its lost writes changed (and maybe rows
other writes changed then, which a follower reads again harmlessly). Few enough rows are recorded by name; more are
recorded keys unknown with the window as their notes bracket, which each follower pages by the ring's
``(table_name, noted_at)`` index and evicts exactly. A whole-table note (a truncate, a load, an operator's reset) is
re-recorded whole only when it is stamped inside the window: the ONE whole change this module records, and only for
the window's own writes (the dead writer's, the failed one's, the migration's); one stamped before is an earlier
write's, whose advance was heard (2026-11-22: re-recording an old reset made every follower read its table whole,
three times).

**What a table is switched on by.** Its change-note row trigger (:data:`~threetears.core.ddl.change_notes
.NOTE_TRIGGER`): every table whose writes advance a generation carries it from the migration run that creates it, and
a writer that imports no class for the table (the hub's broker, committing a tool pod's write) reads the switch off
the catalogue (:func:`tables_switched_by_trigger`). Until 0.95 the per-row write-mark trigger was the switch, and each
commit inserted a mark the reconciler drained; it is gone (coordination migration v006), and the backstop takes down
any an older replica puts back mid-roll (:meth:`WriteBackstop.drop_leftover_write_mark_triggers`, until 0.96.0).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol

from uuid import uuid7

from threetears.observe import contain, get_logger
from threetears.observe.periodic import PeriodicTask
from threetears.observe.metrics import counter, gauge

from threetears.core.collections.change_notes import NotedChange, NotedRows, noted_rows_since
from threetears.core.collections.epoch_change import ChangedRow, advance_recording
from threetears.core.collections.generation import GenerationSource, tables_written_l2_first, tenancy_of_table
from threetears.core.coordination.lease import KVLease, LeaseUnavailable
from threetears.core.ddl.change_notes import NOTE_TRIGGER, WHOLE_COPY_ROWS
from threetears.core.ddl.ddl_lock import DdlLockPolicy, database_ddl_lock
from threetears.core.ddl.errors import DdlLockTimeoutError
from threetears.core.ddl.session import ConnectionSession
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.ddl.write_marks import MARK_FUNCTION, WRITE_MARK_TRIGGER, WRITE_MARKS_CONTRACT_RELEASE
from threetears.core.sql_fragments import quote_identifier
from threetears.core.tenancy import TENANCY_UNDECLARED, schema_tenant
from threetears.core.utils.write_ledger import CLOCK_SKEW, WRITE_GRACE, WriteLedger, process_write_ledger

__all__ = [
    "WriteBackstop",
    "announce_window",
    "tables_switched_by_trigger",
]

log = get_logger(__name__)

#: how often a writer publishes its watermark: a dead writer is seen within the stale time after its last beat
_BEAT_SECONDS: Final = 5.0

#: a beat older than this is a dead writer's: six beats missed, never a slow one (a beat is one KV put). A writer only
#: cut off from NATS that long is announced too, which costs one extra advance of the tables it wrote, never a miss
_STALE_SECONDS: Final = 30.0

#: the KV key prefix a writer's beat is kept under, then its database (the ring's schema) and its process
_BEAT_PREFIX: Final = "write-ledger"

#: the lease one recoverer holds for a pass, and how long a dead holder's hold lasts
_LEASE_KEY: Final = "write-ledger-recovery"
_LEASE_TTL_SECONDS: Final = 15

#: the most rows one announced table's epoch names from the change notes; more are advanced as keys unknown with the
#: window as their notes bracket. A copy of more rows is noted whole (the same count)
_MAX_NOTED_ROWS: Final = WHOLE_COPY_ROWS

#: how long the leftover write-mark sweep waits for the DDL lock before trying again on its next run
_SWEEP_LOCK_POLICY: Final = DdlLockPolicy(poll_interval=0.5, max_wait=5.0)

_ANNOUNCED = counter(
    "collection_lost_write_tables_announced_total",
    description="tables advanced again from the change notes of a lost write's window, by why: recovery (a dead "
    "writer), migration (a data migration's run), repair",
    label_names=("why",),
)
_KEYS_UNKNOWN = counter(
    "collection_lost_write_keys_unknown_total",
    description="announced tables whose window's rows no one record names, advanced keys unknown, by why: "
    "too_many_rows (with the window as its notes bracket), not_covered or notes_unavailable (every follower is "
    "behind on the table, never told to drop it)",
    label_names=("table", "reason"),
)
_RECOVERIES = counter(
    "collection_lost_write_recoveries_total",
    description="dead writers' windows announced by the recovery lease holder, by outcome: recovered, failed "
    "(retried next pass)",
    label_names=("outcome",),
)
_WRITERS = gauge(
    "collection_lost_write_writers",
    description="writer beats the recovery holder read on its last pass, live and stale",
    label_names=("state",),
)
_LEFTOVER_MARKS = counter(
    "collection_write_mark_triggers_removed_total",
    description="write-mark triggers an older replica put back mid-roll, taken down again (until 0.96.0)",
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
    """what the backstop needs of an L3 pool (an ``asyncpg.Pool`` is one)."""

    def acquire(self) -> AbstractAsyncContextManager[Any]:
        """one connection, for a statement sequence that must share a session (the DDL lock's)."""
        ...


#: the tables of one schema carrying the change-note row trigger, among those named
_SWITCHED_SQL: Final[str] = """
SELECT c.relname::text AS table_name
FROM pg_trigger g
JOIN pg_class c ON c.oid = g.tgrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE g.tgname = $1 AND NOT g.tgisinternal AND n.nspname::text = $2 AND c.relname::text = ANY($3::text[])
"""

#: every table of the database carrying the change-note row trigger, with its schema
_ALL_SWITCHED_SQL: Final[str] = """
SELECT n.nspname::text AS table_schema, c.relname::text AS table_name
FROM pg_trigger g
JOIN pg_class c ON c.oid = g.tgrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE g.tgname = $1 AND NOT g.tgisinternal
ORDER BY 1, 2
"""

#: every write-mark trigger whose function lives in the given schema, with its table
_MARK_TRIGGERS_SQL: Final[str] = """
SELECT n.nspname::text AS table_schema, c.relname::text AS table_name
FROM pg_trigger g
JOIN pg_class c ON c.oid = g.tgrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_proc p ON p.oid = g.tgfoid
JOIN pg_namespace pn ON pn.oid = p.pronamespace
WHERE g.tgname = $1 AND NOT g.tgisinternal AND pn.nspname::text = $2
ORDER BY 1, 2
"""

_MARK_FUNCTION_SQL: Final[str] = """
SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE p.proname::text = $1 AND n.nspname::text = $2
"""


async def tables_switched_by_trigger(pool: SqlConnection, schema: str, tables: Iterable[str]) -> frozenset[str]:
    """the named tables of ``schema`` that carry the change-note row trigger: switched on by the database itself.

    A table is switched on by its declaration (a collection class with ``write_generation``), and the migration that
    creates it gives it its change-note triggers (:meth:`~threetears.core.data.migrations.registry.PackageMigrations
    .mark_write_generation`). A writer that imports no class for the table -- the hub's broker, committing a tool
    pod's write to the pod's own tables -- reads the switch here instead: the triggers are installed under the
    database's DDL lock, so a pod cannot claim a table is switched on without it being so. A catalogue read: no DDL,
    no lock. (Until 0.95 the write-mark trigger was the switch.)

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
    rows = await pool.fetch(_SWITCHED_SQL, NOTE_TRIGGER, schema, names)
    return frozenset(str(row["table_name"]) for row in rows)


async def _switched_tables(pool: SqlConnection) -> dict[str, tuple[str, ...]]:
    """every table of the database a lost write's window may hold notes of: each one carrying the change-note row
    trigger, but the tables written L2-first (their epoch is advanced, naming their rows, before the row reaches L3).

    A catalogue read: no DDL, no lock.

    :param pool: the L3 pool, or one connection
    :ptype pool: SqlConnection
    :return: table name -> the schemas holding it with the trigger
    :rtype: dict[str, tuple[str, ...]]
    """
    l2_first = tables_written_l2_first()
    found: dict[str, list[str]] = {}
    for row in await pool.fetch(_ALL_SWITCHED_SQL, NOTE_TRIGGER):
        table = str(row["table_name"])
        if table not in l2_first:
            found.setdefault(table, []).append(str(row["table_schema"]))
    return {table: tuple(schemas) for table, schemas in found.items()}


@dataclass(frozen=True)
class _Advance:
    """one advance an announced table's window is recorded as.

    :ivar rows: the rows the notes name; ``None`` for a rowless record
    :ivar tenant: a rowless record's tenant, ``None`` for ``_all``
    :ivar keys_unknown: whether no one record names its rows
    :ivar noted_between: a keys-unknown record's notes bracket, when the notes name its rows (too many for one record)
    :ivar whole: a whole-table record's change, with the reach its notes named
    """

    rows: list[ChangedRow] | None = None
    tenant: str | None = None
    keys_unknown: bool = False
    noted_between: tuple[datetime, datetime] | None = None
    whole: NotedChange | None = None


@dataclass(frozen=True)
class _TableAnnounced:
    """what one table's window was recorded as.

    :ivar table: the table
    :ivar tokens: each advance's token, in the order made; empty when the window holds no note of the table
    :ivar records: what each advance recorded, for the log
    """

    table: str
    tokens: tuple[str | None, ...]
    records: tuple[Mapping[str, Any], ...]


def _tenants(table: str, schemas: Sequence[str], *, platform_schema: str, schema: str | None) -> tuple[str | None, ...]:
    """the tenants a rowless advance of ``table`` goes under.

    For an agent-schema table, each schema holding it (``schema`` alone, when the window is one schema's: a migration
    run's); ``_all`` for any other table. A table no class here declares, switched on by its trigger alone (a tool
    pod's own table), lives in its pod's schema; one in the platform schema stays on ``_all``.

    :param table: the table
    :ptype table: str
    :param schemas: the schemas holding it with its trigger
    :ptype schemas: Sequence[str]
    :param platform_schema: the platform schema (the ring's)
    :ptype platform_schema: str
    :param schema: the one schema the window's writes ran in, or ``None`` for any
    :ptype schema: str | None
    :return: the tenants, ``None`` for ``_all``
    :rtype: tuple[str | None, ...]
    """
    kind = tenancy_of_table(table).kind
    if kind not in ("schema", TENANCY_UNDECLARED.kind):
        return (None,)
    held = [each for each in schemas if schema is None or each == schema]
    if kind != "schema" and (not held or platform_schema in held):
        return (None,)
    return tuple(schema_tenant(each) for each in held) or (None,)


async def _changes_of(
    pool: SqlConnection,
    *,
    notes_schema: str,
    table: str,
    since: datetime,
    until: datetime,
    tenants: tuple[str | None, ...],
) -> list[_Advance]:
    """what the window's writes to ``table`` changed, from the notes they left: each advance to record.

    Few enough rows are named, each under its own tenant; a whole-table note stamped inside the window is recorded
    whole under its own tenant, covering that tenant's rows. More rows than one record names are recorded keys unknown
    with the window as their notes bracket, under each tenant no whole change covers. When the notes cannot say (the
    window is older than the ring's coverage, or no ring), the record is keys unknown with no bracket: every follower is
    behind on the table, never told to drop it. Either is said at WARNING and counted.

    :param pool: the L3 pool
    :ptype pool: SqlConnection
    :param notes_schema: the schema holding the ring
    :ptype notes_schema: str
    :param table: the table
    :ptype table: str
    :param since: the window's start, by the database's clock
    :ptype since: datetime
    :param until: the window's end
    :ptype until: datetime
    :param tenants: the tenants a rowless record goes under
    :ptype tenants: tuple[str | None, ...]
    :return: each advance to record; empty when the window holds no note of the table
    :rtype: list[_Advance]
    """
    noted = await noted_rows_since(
        pool, notes_schema=notes_schema, table=table, since=since, until=until, cap=_MAX_NOTED_ROWS
    )
    if noted == "no_notes":
        return []
    if isinstance(noted, NotedRows) and not noted.too_many_rows:
        advances: list[_Advance] = []
        if noted.rows:
            advances.append(_Advance(rows=list(noted.rows)))
        advances.extend(_Advance(tenant=change.tenant, whole=change) for change in noted.whole)
        return advances
    reason = "too_many_rows" if isinstance(noted, NotedRows) else noted
    _KEYS_UNKNOWN.labels(table=table, reason=reason).inc()
    whole = noted.whole if isinstance(noted, NotedRows) else ()
    bracket = (since, until) if isinstance(noted, NotedRows) else None
    log.warning(
        "a lost write's window is advanced keys unknown: "
        + (
            "more rows than one record names, so each follower reads their keys from the change notes"
            if bracket is not None
            else "the change notes cannot name its rows, so every follower is behind on the table until its next start"
        ),
        extra={"extra_data": {"table": table, "reason": reason, "whole_changes": len(whole)}},
    )
    # a whole-table note in the window is a write of its own whose advance may have been lost too: recorded whole
    # beside the bracket, since a follower applying the bracket's keys never acts on a whole note in it. A plain or
    # reset change covers its tenant's rows (every tenant's, for _all), so no bracket is recorded under a tenant it covers
    covering = {change.tenant for change in whole if not change.prefixes}
    unnamed = [
        _Advance(tenant=tenant, keys_unknown=True, noted_between=bracket)
        for tenant in tenants
        if None not in covering and tenant not in covering
    ]
    return [*unnamed, *(_Advance(tenant=change.tenant, whole=change) for change in whole)]


async def _announce_table_window(
    pool: SqlConnection,
    *,
    generation_source: GenerationSource,
    notes_schema: str,
    table: str,
    since: datetime,
    until: datetime,
    schemas: Sequence[str],
    schema: str | None = None,
) -> _TableAnnounced:
    """advance ``table`` once for each change the window's notes name; nothing when they name none.

    :param pool: the L3 pool holding the ring
    :ptype pool: SqlConnection
    :param generation_source: advances the table (the epoch bucket's writer)
    :ptype generation_source: GenerationSource
    :param notes_schema: the schema holding the ring (the platform schema)
    :ptype notes_schema: str
    :param table: the table
    :ptype table: str
    :param since: the window's start, by the database's clock, already widened by the clocks' skew
    :ptype since: datetime
    :param until: the window's end, likewise
    :ptype until: datetime
    :param schemas: the schemas holding the table with its trigger (:func:`_switched_tables`)
    :ptype schemas: Sequence[str]
    :param schema: the one schema the window's writes ran in (a migration run's); ``None`` for any
    :ptype schema: str | None
    :return: what was recorded
    :rtype: _TableAnnounced
    :raises GenerationUnavailableError: when an advance failed; the advances before it stand
    """
    tenants = _tenants(table, schemas, platform_schema=notes_schema, schema=schema)
    tokens: list[str | None] = []
    records: list[Mapping[str, Any]] = []
    for advance in await _changes_of(
        pool, notes_schema=notes_schema, table=table, since=since, until=until, tenants=tenants
    ):
        whole = advance.whole
        # the caller holds the window itself for a retry: a failed advance here is not handed to the ledger again
        token = await advance_recording(
            generation_source,
            table,
            advance.rows,
            keys_unknown=advance.keys_unknown,
            tenant=advance.tenant,
            noted_between=advance.noted_between,
            prefixes=list(whole.prefixes) or None if whole is not None else None,
            reason=whole.reason if whole is not None else None,
            repair=False,
        )
        tokens.append(token)
        records.append(
            {
                "epoch": token,
                "tenant": advance.tenant,
                "rows": None if advance.rows is None else len(advance.rows),
                "keys_unknown": advance.keys_unknown,
                "bracket": None if advance.noted_between is None else [t.isoformat() for t in advance.noted_between],
                "whole": None if whole is None else {"reason": whole.reason, "prefixes": len(whole.prefixes)},
            }
        )
    return _TableAnnounced(table=table, tokens=tuple(tokens), records=tuple(records))


@dataclass(frozen=True)
class _WindowAnnounced:
    """what announcing one window did.

    :ivar advanced: the tables advanced (the window held notes of them)
    :ivar failed: the tables whose advance failed, with why
    """

    advanced: tuple[str, ...]
    failed: Mapping[str, str]


async def announce_window(
    pool: SqlConnection,
    *,
    generation_source: GenerationSource,
    notes_schema: str,
    since: datetime,
    until: datetime,
    why: str,
    schema: str | None = None,
    tables: Mapping[str, tuple[str, ...]] | None = None,
) -> _WindowAnnounced:
    """advance every switched table the window's change notes name, each from its own notes.

    One keyed read of the ring a table, by its ``(table_name, noted_at)`` index: no table holding no note in the window
    is advanced. One failing table never stops the rest.

    :param pool: the L3 pool holding the ring
    :ptype pool: SqlConnection
    :param generation_source: advances each table
    :ptype generation_source: GenerationSource
    :param notes_schema: the schema holding the ring
    :ptype notes_schema: str
    :param since: the window's start, by the database's clock, already widened by the clocks' skew
    :ptype since: datetime
    :param until: the window's end, likewise
    :ptype until: datetime
    :param why: ``recovery``, ``migration`` or ``repair``, for the log and the count
    :ptype why: str
    :param schema: the one schema the window's writes ran in; ``None`` for any
    :ptype schema: str | None
    :param tables: the switched tables (:func:`_switched_tables`); ``None`` reads them
    :ptype tables: Mapping[str, tuple[str, ...]] | None
    :return: what was done
    :rtype: _WindowAnnounced
    """
    if until < since:
        raise ValueError(f"a window ends after it starts: {since.isoformat()} .. {until.isoformat()}")
    switched = tables if tables is not None else await _switched_tables(pool)
    advanced: list[str] = []
    failed: dict[str, str] = {}
    for table, schemas in sorted(switched.items()):
        if schema is not None and schema not in schemas:
            continue
        try:
            done = await _announce_table_window(
                pool,
                generation_source=generation_source,
                notes_schema=notes_schema,
                table=table,
                since=since,
                until=until,
                schemas=schemas,
                schema=schema,
            )
        # prawduct:allow prawduct/broad-except -- one table that cannot be advanced is reported and retried by the caller; the rest are announced
        except Exception as exc:
            failed[table] = f"{type(exc).__name__}: {exc}"
            contain(
                exc,
                log=log,
                what="a table a lost write's window holds notes of could not be advanced; it is retried",
                extra={"table": table, "why": why, "since": since.isoformat(), "until": until.isoformat()},
                level="error",
            )
            continue
        if done.tokens:
            advanced.append(table)
            _ANNOUNCED.labels(why=why).inc()
            log.info(
                "a table was advanced from the change notes of a lost write's window",
                extra={
                    "extra_data": {
                        "table": table,
                        "why": why,
                        "since": since.isoformat(),
                        "until": until.isoformat(),
                        "recorded": list(done.records),
                    }
                },
            )
    return _WindowAnnounced(advanced=tuple(advanced), failed=failed)


@dataclass(frozen=True)
class _WriterBeat:
    """one writer process's beat, as it publishes it.

    :ivar process: the writer's process id
    :ivar watermark: its ledger's watermark when it beat
    :ivar beat_at: when it beat, by its clock
    :ivar grace_seconds: how long its writes' starts are kept after they end (the bound on commit-to-advance)
    """

    process: str
    watermark: datetime
    beat_at: datetime
    grace_seconds: float

    def encode(self) -> bytes:
        """the beat as the KV value.

        :return: JSON bytes
        :rtype: bytes
        """
        return json.dumps(
            {
                "process": self.process,
                "watermark": self.watermark.isoformat(),
                "beat_at": self.beat_at.isoformat(),
                "grace_seconds": self.grace_seconds,
            },
            separators=(",", ":"),
        ).encode()

    @classmethod
    def decode(cls, raw: bytes) -> _WriterBeat:
        """a beat from its KV value.

        :param raw: the value
        :ptype raw: bytes
        :return: the beat
        :rtype: _WriterBeat
        :raises ValueError: when the value is not a beat
        """
        try:
            value = json.loads(raw)
            return cls(
                process=str(value["process"]),
                watermark=datetime.fromisoformat(value["watermark"]),
                beat_at=datetime.fromisoformat(value["beat_at"]),
                grace_seconds=float(value["grace_seconds"]),
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"not a writer beat: {raw!r}") from exc

    def window(self, now: datetime) -> tuple[datetime, datetime]:
        """the window a dead writer's lost advances lie in: from its watermark to the last time it could have
        committed (its beat plus the grace, or now if sooner), widened by the clocks' skew at both ends.

        :param now: the recoverer's clock
        :ptype now: datetime
        :return: ``(since, until)``
        :rtype: tuple[datetime, datetime]
        """
        last = min(self.beat_at + timedelta(seconds=self.grace_seconds), now)
        return self.watermark - CLOCK_SKEW, max(last, self.watermark) + CLOCK_SKEW


@dataclass(frozen=True)
class _RecoveryPass:
    """what one recovery pass did.

    :ivar skipped: another replica held the recovery lease
    :ivar live: writers whose beat was fresh
    :ivar recovered: dead writers whose window was announced and whose key was deleted
    :ivar failed: dead writers whose window could not be announced in full; kept for the next pass
    """

    skipped: bool = False
    live: int = 0
    recovered: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()


async def _drop_leftover_write_mark_triggers(
    pool: SqlPool, *, marks_schema: str, policy: DdlLockPolicy = _SWEEP_LOCK_POLICY
) -> list[str]:
    """take down every write-mark trigger calling ``marks_schema``'s function, and the function: those an older replica
    put back mid-roll after coordination migration v006 took them down.

    Reads the catalogue first and returns at once when there is none (every run once the roll is done). Otherwise takes
    the database's DDL lock, bounded; a busy lock defers to the next run. Rolling upgrade only: gone by
    :data:`~threetears.core.ddl.write_marks.WRITE_MARKS_CONTRACT_RELEASE`.

    :param pool: the L3 pool, allowed DDL
    :ptype pool: SqlPool
    :param marks_schema: the schema whose mark function the triggers call (the platform schema)
    :ptype marks_schema: str
    :param policy: how long to wait for the DDL lock
    :ptype policy: DdlLockPolicy
    :return: ``schema.table`` of each trigger taken down
    :rtype: list[str]
    """
    found = await pool.fetch(_MARK_TRIGGERS_SQL, WRITE_MARK_TRIGGER, marks_schema)
    function = await pool.fetch(_MARK_FUNCTION_SQL, MARK_FUNCTION, marks_schema)
    if not found and not function:
        return []
    dropped: list[str] = []
    async with pool.acquire() as conn:
        try:
            async with database_ddl_lock(ConnectionSession(conn), policy):
                for row in await conn.fetch(_MARK_TRIGGERS_SQL, WRITE_MARK_TRIGGER, marks_schema):
                    schema, table = str(row["table_schema"]), str(row["table_name"])
                    await conn.execute(
                        f"DROP TRIGGER IF EXISTS {quote_identifier(WRITE_MARK_TRIGGER)} "
                        f"ON {quote_identifier(schema)}.{quote_identifier(table)}"
                    )
                    dropped.append(f"{schema}.{table}")
                await conn.execute(
                    f"DROP FUNCTION IF EXISTS {quote_identifier(marks_schema)}.{quote_identifier(MARK_FUNCTION)}()"
                )
        except DdlLockTimeoutError:
            log.info("leftover write-mark triggers not taken down yet: the database's DDL lock is busy")
            return []
    if dropped:
        _LEFTOVER_MARKS.inc(len(dropped))
        log.warning(
            "write-mark triggers an older replica put back were taken down (rolling upgrade, until %s)",
            WRITE_MARKS_CONTRACT_RELEASE,
            extra={"extra_data": {"tables": dropped}},
        )
    return dropped


class WriteBackstop:
    """one process's backstop for lost epoch advances: its heartbeat, its repairs and, under a lease, recovery.

    Every process writing to an L3 database holding switched tables runs one per database (the hub, the gateway,
    identity-core), over a pool built with :func:`~threetears.core.utils.write_ledger.write_ledger_connection_class`:

    - :meth:`beat` publishes the ledger's watermark to ``write-ledger.<notes_schema>.<process>`` in the lease's bucket
      (memory, replicated: the platform's ``leases``), every :data:`_BEAT_SECONDS`;
    - :meth:`repair_pass` makes each advance this process lost again until it lands;
    - :meth:`recover_pass` (``recover=True``), under the ``write-ledger-recovery`` lease, announces each dead writer's
      window of this database and deletes its key;
    - :meth:`announce_window` is the migration runner's hook (``MigrationRunner(announcer=...)``).

    :meth:`start` runs the three as tasks; :meth:`stop` ends them, drains the repairs and deletes this process's key.
    The passes are public so a caller with its own scheduler runs them itself.

    :param pool: the L3 pool (a ``write_ledger_connection_class`` one for the writes it keeps; any for the reads here)
    :ptype pool: SqlPool
    :param generation_source: the process's source, the epoch bucket's writer
    :ptype generation_source: GenerationSource
    :param notes_schema: the schema holding the change-note ring: names the database in the beat's key
    :ptype notes_schema: str
    :param lease: the KV lease factory whose bucket holds the beats and whose lease one recoverer holds
    :ptype lease: KVLease
    :param process_id: this process's id in its key, one NATS token; ``None`` mints one
    :ptype process_id: str | None
    :param ledger: the ledger; ``None`` for the process's
    :ptype ledger: WriteLedger | None
    :param recover: whether this process takes part in recovery (a replica of the database's owner: the hub,
        identity-core); a process that only writes beats and repairs passes ``False``
    :ptype recover: bool
    :param clock: wall-clock seconds since the epoch
    :ptype clock: Callable[[], float]
    """

    def __init__(
        self,
        pool: SqlPool,
        *,
        generation_source: GenerationSource,
        notes_schema: str,
        lease: KVLease,
        process_id: str | None = None,
        ledger: WriteLedger | None = None,
        recover: bool = True,
        beat_seconds: float = _BEAT_SECONDS,
        stale_seconds: float = _STALE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if stale_seconds <= beat_seconds:
            raise ValueError("a writer is stale only after more than one beat was missed")
        self._pool = pool
        self._source = generation_source
        self._notes_schema = notes_schema
        self._lease = lease
        self._process = process_id or f"w{uuid7().hex}"
        if not self._process.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"a process id is one NATS token: {self._process!r}")
        self._ledger = ledger if ledger is not None else process_write_ledger()
        self._recover = recover
        self._beat_seconds = beat_seconds
        self._stale_seconds = stale_seconds
        self._clock = clock
        # the passes, each its own loop; a failed pass is logged and the next one runs
        self._loops = [
            # the first beat runs at once, in the loop, not in start(): the hub declares the leases bucket, and a
            # service the hub waits on (identity-core, whose health the hub's start depends on) would otherwise fail
            # its own start until the hub is up, which it never is (2026-11-22 replay: identity-core crash-looped on
            # "bucket=aibots-leases does not exist"). A beat that cannot land is retried each interval; the
            # watermark keeps every lost advance's start (WriteLedger.watermark), so the first beat that lands
            # covers them
            PeriodicTask(self.beat, interval=beat_seconds, first_delay=0, name="write-ledger-beat", logger=log),
            PeriodicTask(self.repair_pass, interval=1.0, name="write-ledger-repair", logger=log),
        ]
        if recover:
            self._loops += [
                PeriodicTask(self.recover_pass, interval=beat_seconds, name="write-ledger-recovery", logger=log),
                PeriodicTask(
                    self.drop_leftover_write_mark_triggers,
                    interval=60.0,
                    first_delay=0,
                    name="write-ledger-leftover-marks",
                    logger=log,
                ),
            ]

    @property
    def key(self) -> str:
        """this process's beat key, as stored.

        :return: the key
        :rtype: str
        """
        return self._lease.stored_key(f"{_BEAT_PREFIX}.{self._notes_schema}.{self._process}")

    @property
    def ledger(self) -> WriteLedger:
        """the ledger this backstop beats and repairs.

        :return: it
        :rtype: WriteLedger
        """
        return self._ledger

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._clock(), tz=UTC)

    async def beat(self) -> _WriterBeat:
        """publish the ledger's watermark.

        :return: the beat published
        :rtype: _WriterBeat
        """
        beat = _WriterBeat(
            process=self._process,
            watermark=self._ledger.watermark(),
            beat_at=self._now(),
            grace_seconds=WRITE_GRACE.total_seconds(),
        )
        bucket = await self._lease.bucket()
        await bucket.put(key=self.key, value=beat.encode())
        return beat

    async def repair_pass(self) -> int:
        """make each lost advance of this process again; release those that land.

        :return: how many are still held
        :rtype: int
        """
        return await self._ledger.repair_pending()

    async def announce_window(self, since: datetime, until: datetime, *, schema: str | None = None) -> _WindowAnnounced:
        """announce a migration run's window: every switched table of ``schema`` the window's notes name.

        A table that fails is held in the ledger and retried with the same window until it lands; a window whose
        switched tables cannot even be read is held whole the same way.

        :param since: the run's start, by the database's clock
        :ptype since: datetime
        :param until: the run's end
        :ptype until: datetime
        :param schema: the schema the run wrote in
        :ptype schema: str | None
        :return: what was done
        :rtype: _WindowAnnounced
        """
        start, end = since - CLOCK_SKEW, until + CLOCK_SKEW
        try:
            switched = await _switched_tables(self._pool)
        # prawduct:allow prawduct/broad-except -- the window is held whole in the ledger and announced by a retry
        except Exception as exc:
            contain(
                exc,
                log=log,
                what="a migration run's switched tables could not be read; its window is held and announced by a retry",
                extra={"schema": schema, "since": start.isoformat(), "until": end.isoformat()},
            )
            self._ledger.lost_advance(
                f"migration window of {schema}",
                self._retry_window(start, end, schema),
                error=f"{type(exc).__name__}: {exc}",
            )
            return _WindowAnnounced(advanced=(), failed={"*": f"{type(exc).__name__}: {exc}"})
        done = await announce_window(
            self._pool,
            generation_source=self._source,
            notes_schema=self._notes_schema,
            since=start,
            until=end,
            why="migration",
            schema=schema,
            tables=switched,
        )
        for table, error in done.failed.items():
            self._ledger.lost_advance(table, self._retry_table(table, switched[table], start, end, schema), error=error)
        return done

    def _retry_window(self, since: datetime, until: datetime, schema: str | None) -> Callable[[], Any]:
        async def again() -> _WindowAnnounced:
            done = await announce_window(
                self._pool,
                generation_source=self._source,
                notes_schema=self._notes_schema,
                since=since,
                until=until,
                why="repair",
                schema=schema,
            )
            if done.failed:
                raise GenerationUnavailableError(f"a held window's tables are still not advanced: {dict(done.failed)}")
            return done

        return again

    def _retry_table(
        self, table: str, schemas: tuple[str, ...], since: datetime, until: datetime, schema: str | None
    ) -> Callable[[], Any]:
        async def again() -> _TableAnnounced:
            return await _announce_table_window(
                self._pool,
                generation_source=self._source,
                notes_schema=self._notes_schema,
                table=table,
                since=since,
                until=until,
                schemas=schemas,
                schema=schema,
            )

        return again

    async def recover_pass(self) -> _RecoveryPass:
        """under the recovery lease, announce each dead writer's window of this database and delete its key.

        :return: what the pass did
        :rtype: _RecoveryPass
        """
        try:
            handle = await self._lease.acquire(_LEASE_KEY, ttl_seconds=_LEASE_TTL_SECONDS, max_wait_seconds=0)
        except LeaseUnavailable:
            return _RecoveryPass(skipped=True)
        try:
            return await self._recover_dead()
        finally:
            await handle.release()

    async def _recover_dead(self) -> _RecoveryPass:
        bucket = await self._lease.bucket()
        prefix = self._lease.stored_key(f"{_BEAT_PREFIX}.{self._notes_schema}.")
        now = self._now()
        live = 0
        recovered: list[str] = []
        failed: list[str] = []
        switched: dict[str, tuple[str, ...]] | None = None
        for key in await bucket.list_keys(prefix=prefix):
            entry = await bucket.get_entry(key=key)
            if entry is None:
                continue
            raw, revision = entry
            try:
                beat = _WriterBeat.decode(raw)
            except ValueError as exc:
                contain(exc, log=log, what="a writer beat key holds no beat; left as it is", extra={"key": key})
                continue
            if (now - beat.beat_at).total_seconds() <= self._stale_seconds:
                live += 1
                continue
            since, until = beat.window(now)
            if switched is None:
                switched = await _switched_tables(self._pool)
            done = await announce_window(
                self._pool,
                generation_source=self._source,
                notes_schema=self._notes_schema,
                since=since,
                until=until,
                why="recovery",
                tables=switched,
            )
            if done.failed:
                # kept: the next pass announces the window again (a table advanced twice costs its followers a re-read
                # of the window's keys, never a miss)
                failed.append(beat.process)
                _RECOVERIES.labels(outcome="failed").inc()
                continue
            await bucket.delete(key=key, revision=revision)
            recovered.append(beat.process)
            _RECOVERIES.labels(outcome="recovered").inc()
            log.warning(
                "a dead writer's window was announced from the change notes",
                extra={
                    "extra_data": {
                        "process": beat.process,
                        "since": since.isoformat(),
                        "until": until.isoformat(),
                        "beat_at": beat.beat_at.isoformat(),
                        "tables": list(done.advanced),
                    }
                },
            )
        _WRITERS.labels(state="live").set(live)
        _WRITERS.labels(state="stale").set(len(failed))
        return _RecoveryPass(live=live, recovered=tuple(recovered), failed=tuple(failed))

    async def drop_leftover_write_mark_triggers(self) -> list[str]:
        """take down write-mark triggers an older replica put back mid-roll (:func:`_drop_leftover_write_mark_triggers`).

        :return: ``schema.table`` of each taken down
        :rtype: list[str]
        """
        return await _drop_leftover_write_mark_triggers(self._pool, marks_schema=self._notes_schema)

    async def start(self) -> None:
        """run the beat (its first at once), the repairs and (with ``recover``) recovery and the leftover sweep, each a
        :class:`~threetears.observe.periodic.PeriodicTask`. Never waits on the leases bucket.

        :return: nothing
        :rtype: None
        """
        self._ledger.attach_repairer()
        for loop in self._loops:
            loop.start()

    async def stop(self, *, drain_seconds: float = 30.0) -> int:
        """end the loops, retry the lost advances until none is left or ``drain_seconds`` pass, then delete this
        process's key; with any left, beat once more instead, so recovery announces them.

        :param drain_seconds: how long to keep retrying
        :ptype drain_seconds: float
        :return: how many lost advances are still held
        :rtype: int
        """
        for loop in self._loops:
            await loop.stop()
        deadline = time.monotonic() + drain_seconds
        left = await self.repair_pass()
        while left and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            left = await self.repair_pass()
        self._ledger.detach_repairer()
        try:
            if left:
                await self.beat()
            else:
                bucket = await self._lease.bucket()
                await bucket.delete(key=self.key)
        # prawduct:allow prawduct/broad-except -- a key left behind is a stale beat: recovery announces its window once
        except Exception as exc:
            contain(exc, log=log, what="could not settle this writer's beat key on stop; recovery reads it as stale")
        return left
