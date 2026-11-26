"""a committed write whose epoch advance was lost is announced again, from the tables' own rows, with no L3 write a
commit.

A write to a generation-switched table commits to L3 and is advanced after, outside its transaction. Three ways an
advance can be lost, and what announces each one again:

- **The advance failed or was cancelled, the writer alive.** :func:`~threetears.core.collections.epoch_change
  .advance_recording` hands it to the process's ledger (:mod:`threetears.core.utils.write_ledger`) before raising, and
  :meth:`WriteBackstop.repair_pass` makes it again, exactly as it was, until it lands.
- **The writer died between its commit and its advance.** Each writer process publishes its ledger's watermark W (the
  oldest start of a write that may still lack its advance) to a NATS KV key every :data:`_BEAT_SECONDS`
  (:meth:`WriteBackstop.beat`). The replica holding the recovery lease (:meth:`WriteBackstop.recover_pass`) takes a key
  whose beat is older than :data:`_STALE_SECONDS` for a dead writer: every switched table with rows stamped in
  ``[W - skew, min(beat + grace, now) + skew]`` is advanced from those rows (:func:`announce_window`), and the key is
  deleted. A process stopping cleanly deletes its own key after its repairs land (:meth:`WriteBackstop.stop`).
- **A data migration wrote rows** (no collection, no advance). The migration runner hands its run's window to
  :meth:`WriteBackstop.announce_window` (``MigrationRunner(announcer=...)``).

**How a window is announced** (:func:`_changes_of`; append-only L3, ``design-append-only-l3.md`` decision 3, chunk 7).
Every committed write stamped each row it wrote with an epoch inside the window (a hybrid logical clock stamp, on a
writer's clock the skew already widens the window by), so a table's rows stamped in the window's epoch range
(:func:`~threetears.core.collections.epoch_reads.epoch_range`) name every row its lost writes changed: a versioned
table's versions and gone markers, a scope at a time through its ``(scope, epoch, key...)`` index; a stamped table's
rows, gone ones too, in each of its 16 ``(epoch_bucket, epoch)`` ranges (the restore's own read,
:mod:`~threetears.core.collections.epoch_reads`, narrowed to the window). Maybe rows other writes changed then too,
which a follower reads again harmlessly. Few enough rows are recorded by name; more are recorded keys unknown with the
window as their bracket, which each follower pages from the same read (``l3.noted_keys``,
:func:`~threetears.core.collections.change_notes.noted_keys_page`) and evicts exactly. A run deleted in the window (its
run marker, :mod:`~threetears.core.ddl.run_markers`) is re-recorded whole by its prefixes: the ONE whole change this
module records, and only for the window's own deletes.

**What a table is switched on by.** Its epoch: every table whose writes advance a generation carries ``epoch BIGINT
NOT NULL`` (:func:`tables_switched_by_epoch`), versioned or stamped, its restore index declaring it in the catalog
(:func:`~threetears.core.collections.table_restore.restored_tables`), which a writer that imports no class for the
table (the hub's broker, committing a tool pod's write) reads. Until 0.95 the change-note row trigger was the switch,
and before it the per-row write-mark trigger; the backstop takes down any write-mark trigger an older replica puts back
mid-roll (:meth:`WriteBackstop.drop_leftover_write_mark_triggers`, until 0.96.0).
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

from threetears.nats.subject_permissions import WRITER_BEAT_KEY_SCOPE
from threetears.observe import contain, get_logger
from threetears.observe.periodic import PeriodicTask
from threetears.observe.metrics import counter, gauge

from threetears.core.collections.epoch_change import ChangedRow, advance_recording
from threetears.core.collections.epoch_reads import epoch_range, range_rows, tenant_of_relation
from threetears.core.collections.noted_change import NotedChange
from threetears.core.collections.table_restore import (
    RestoredTable,
    VersionedTable,
    changed_scopes,
    restored_tables,
    run_marker_changes,
)
from threetears.core.collections.generation import GenerationSource, tenancy_of_table, written_l2_first
from threetears.core.coordination.lease import KVLease, LeaseUnavailable
from threetears.core.ddl.change_notes import WHOLE_COPY_ROWS
from threetears.core.ddl.ddl_lock import DdlLockPolicy, database_ddl_lock
from threetears.core.ddl.epoch_catalog import tables_carrying_epoch
from threetears.core.ddl.errors import DdlLockTimeoutError
from threetears.core.ddl.session import ConnectionSession
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.ddl.write_marks import MARK_FUNCTION, WRITE_MARK_TRIGGER, WRITE_MARKS_CONTRACT_RELEASE
from threetears.core.sql_fragments import quote_identifier
from threetears.core.collections.undeclared_tables import is_undeclared_table
from threetears.core.tenancy import TENANCY_UNDECLARED, Tenancy, schema_tenant
from threetears.core.utils.write_ledger import CLOCK_SKEW, WRITE_GRACE, WriteLedger, process_write_ledger

__all__ = [
    "WriteBackstop",
    "WriterBeats",
    "announce_window",
    "tables_switched_by_epoch",
]

log = get_logger(__name__)

#: how often a writer publishes its watermark: a dead writer is seen within the stale time after its last beat
_BEAT_SECONDS: Final = 5.0

#: a beat older than this is a dead writer's: six beats missed, never a slow one (a beat is one KV put). A writer only
#: cut off from NATS that long is announced too, which costs one extra advance of the tables it wrote, never a miss
_STALE_SECONDS: Final = 30.0

#: the KV key prefix a writer's beat is kept under, then its database (the ring's schema) and its process: the key
#: scope a tool pod opted in to delta files is granted to read them by, so the two never drift
_BEAT_PREFIX: Final = WRITER_BEAT_KEY_SCOPE

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


async def tables_switched_by_epoch(pool: SqlConnection, schema: str, tables: Iterable[str]) -> frozenset[str]:
    """the named tables of ``schema`` switched on by the database itself: each carrying an epoch
    (:func:`~threetears.core.ddl.epoch_catalog.tables_carrying_epoch`, ``epoch BIGINT NOT NULL``), versioned or
    stamped. The epoch is the switch append-only L3 keeps (chunk 7): every write stamps it, and every reader of what a
    table changed reads it (:mod:`~threetears.core.collections.epoch_reads`). Until the ring retired the change-note
    row trigger was the switch, and before it the write-mark trigger.

    A writer that imports no class for the table -- the hub's broker, committing a tool pod's write to the pod's own
    tables -- reads the switch here. A catalogue read: no DDL, no lock.

    :param pool: the L3 pool, or one connection
    :ptype pool: SqlConnection
    :param schema: the schema the tables live in
    :ptype schema: str
    :param tables: the table names
    :ptype tables: Iterable[str]
    :return: those that carry an epoch
    :rtype: frozenset[str]
    """
    names = sorted(set(tables))
    if not names:
        return frozenset()
    return await tables_carrying_epoch(pool, schema, names)


async def _switched_tables(pool: SqlConnection) -> dict[str, tuple[RestoredTable, ...]]:
    """every table of the database a lost write's window may have changed: each one carrying an epoch, versioned or
    stamped (its restore index's declaration, :func:`~threetears.core.collections.table_restore.restored_tables`),
    but the tables written L2-first (their epoch is advanced, naming their rows, before the row reaches L3).

    A catalogue read: no DDL, no lock.

    :param pool: the L3 pool, or one connection
    :ptype pool: SqlConnection
    :return: table name -> each relation of that name, in schema order
    :rtype: dict[str, tuple[RestoredTable, ...]]
    """
    found: dict[str, list[RestoredTable]] = {}
    for relation in await restored_tables(pool):
        table = relation.table
        # a table no class declares (a tool pod's own) is never written L2-first: asked of no registry, so no
        # undeclared lookup is reported for it (:data:`_TRIGGER_ONLY`)
        if is_undeclared_table(table) or not written_l2_first(table):
            found.setdefault(table, []).append(relation)
    return {table: tuple(relations) for table, relations in found.items()}


@dataclass(frozen=True)
class _Advance:
    """one advance an announced table's window is recorded as.

    :ivar rows: the rows the notes name; ``None`` for a rowless record
    :ivar tenant: a rowless record's tenant, ``None`` for ``_all``
    :ivar keys_unknown: whether no one record names its rows
    :ivar noted_between: a keys-unknown record's bracket: its rows too many for one record, a follower pages them
        from the table's versions in it (:func:`~threetears.core.collections.change_notes.noted_keys_page`)
    :ivar whole: a whole-table record's change, with the reach its run markers named
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


#: the scope a lost write's recovery gives a table it knows only from its change-note trigger (no class declares it:
#: a tool pod's own table, undeclared on purpose): every tenant it may be read under -- each schema holding it (its
#: pods' key, were their class to declare it by schema) and ``_all`` (every other declaration's followers read it).
#: The conservative scope: a broader invalidation never misses a follower. Passed explicitly in place of a tenancy, so
#: recovery asks no registry for the table and reports no undeclared lookup (``threetears_undeclared_table_total``
#: stays at zero for it, and next release's refusal of an undeclared lookup never breaks recovery)
_TRIGGER_ONLY: Final = "trigger_only"


def _tenancy_of(table: str, declared: Tenancy) -> Tenancy | str:
    """what a lost write's recovery advances ``table`` under: the tenancy its restore declaration names; for one that
    names none, its class's, or :data:`_TRIGGER_ONLY` for a table no class declares (asked quietly,
    :func:`~threetears.core.collections.undeclared_tables.is_undeclared_table`).

    :param table: the table
    :ptype table: str
    :param declared: the tenancy its restore declaration names
    :ptype declared: Tenancy
    :return: the tenancy, or :data:`_TRIGGER_ONLY`
    :rtype: Tenancy | str
    """
    if declared.kind != TENANCY_UNDECLARED.kind:
        return declared
    return _TRIGGER_ONLY if is_undeclared_table(table) else tenancy_of_table(table)


def _tenants(
    tenancy: Tenancy | str, schemas: Sequence[str], *, platform_schema: str, schema: str | None
) -> tuple[str | None, ...]:
    """the tenants a rowless advance of a table of ``tenancy`` goes under.

    For an agent-schema table, each schema holding it (``schema`` alone, when the window is one schema's: a migration
    run's); ``_all`` for any other declared table. A table known only from its trigger (:data:`_TRIGGER_ONLY`) goes
    under each pod schema holding it and ``_all``: every tenant. A table the registry answers undeclared for (the
    startup loader never ran) lives in its pod's schema; one in the platform schema stays on ``_all``.

    :param tenancy: the table's tenancy, or :data:`_TRIGGER_ONLY` (:func:`_tenancy_of`)
    :ptype tenancy: Tenancy | str
    :param schemas: the schemas holding it with its trigger
    :ptype schemas: Sequence[str]
    :param platform_schema: the platform schema (the ring's)
    :ptype platform_schema: str
    :param schema: the one schema the window's writes ran in, or ``None`` for any
    :ptype schema: str | None
    :return: the tenants, ``None`` for ``_all``
    :rtype: tuple[str | None, ...]
    """
    kind = tenancy if isinstance(tenancy, str) else tenancy.kind
    held = [each for each in schemas if schema is None or each == schema]
    pods = tuple(schema_tenant(each) for each in held if each != platform_schema)
    tenants: tuple[str | None, ...] = (None,)
    if kind == _TRIGGER_ONLY:
        tenants = (*pods, None)
    elif kind == "schema":
        tenants = tuple(schema_tenant(each) for each in held) or (None,)
    elif kind == TENANCY_UNDECLARED.kind and held and platform_schema not in held:
        tenants = pods
    return tenants


async def _changes_of(
    pool: SqlConnection,
    *,
    platform_schema: str,
    table: str,
    relations: Sequence[RestoredTable],
    since: datetime,
    until: datetime,
    tenants: tuple[str | None, ...],
) -> list[_Advance]:
    """what the window's writes to ``table`` changed, from the table's own rows stamped in the window: each advance
    to record.

    Every committed write stamped its rows with epochs inside the window (on a writer's clock the skew already widens
    it by), so the versions (a stamped table's rows) in the window's epoch range name every row its lost writes
    changed, gone markers included (and maybe rows other writes changed then, which a follower reads again
    harmlessly). Read as the restore reads them (:func:`~threetears.core.collections.epoch_reads.range_rows`): a
    scope's or a bucket's index range a keyset page at a time, stopped as soon as more distinct rows are held than one
    record names. Few enough rows are named, each under its own tenant; a run deleted in the window (its run marker)
    is recorded whole under its tenant, naming its prefixes. More rows than one record names are recorded keys
    unknown with the window as their bracket, under each tenant, which each follower pages from the table's versions
    in it (:func:`~threetears.core.collections.change_notes.noted_keys_page`), said at WARNING and counted.

    :param pool: the L3 pool
    :ptype pool: SqlConnection
    :param platform_schema: the platform schema: an undeclared table there is ``_all``'s
    :ptype platform_schema: str
    :param table: the table
    :ptype table: str
    :param relations: its relations the window's writes ran in
    :ptype relations: Sequence[RestoredTable]
    :param since: the window's start
    :ptype since: datetime
    :param until: the window's end
    :ptype until: datetime
    :param tenants: the tenants a rowless record goes under
    :ptype tenants: tuple[str | None, ...]
    :return: each advance to record; empty when the window holds no row of the table
    :rtype: list[_Advance]
    """
    floor, ceiling = epoch_range((since, until))
    # function-local: the distinct rows, keyed by tenant and key, at most one record's worth plus one
    named: dict[tuple[str | None, tuple[str, ...]], dict[str, str | None] | None] = {}
    too_many = False
    whole: list[NotedChange] = []
    for relation in relations:
        own = tenant_of_relation(relation, platform_schema)
        by_column = relation.tenancy.kind == "column"
        async for rows in range_rows(pool, relation, floor, ceiling, page=_MAX_NOTED_ROWS + 1):
            for row in rows:
                named.setdefault((row.tenant if by_column else own, row.keys), row.columns)
            if len(named) > _MAX_NOTED_ROWS:
                too_many = True
                break
        if isinstance(relation, VersionedTable) and relation.run_markers:
            values = [
                scope[len(relation.scope_prefix) :]
                for scope in await changed_scopes(pool, relation.schema, relation.epochs, floor)
                if scope.startswith(relation.scope_prefix)
            ]
            whole.extend(await run_marker_changes(pool, relation, values, floor, ceiling))
        if too_many:
            break
    if not too_many:
        advances: list[_Advance] = []
        if named:
            advances.append(
                _Advance(
                    rows=[
                        ChangedRow(ids=list(keys), columns=columns, tenant=tenant)
                        for (tenant, keys), columns in named.items()
                    ]
                )
            )
        advances.extend(_Advance(tenant=change.tenant, whole=change) for change in whole)
        return advances
    _KEYS_UNKNOWN.labels(table=table, reason="too_many_rows").inc()
    log.warning(
        "a lost write's window is advanced keys unknown: more rows than one record names, so each follower pages "
        "their keys from the table's versions in the window",
        extra={"extra_data": {"table": table, "reason": "too_many_rows", "whole_changes": len(whole)}},
    )
    # a run deleted in the window is a write of its own whose advance may have been lost too: recorded whole beside
    # the bracket, since a follower applying the bracket's keys never acts on a run in it. A run names its prefixes,
    # so no tenant is covered whole: the bracket goes under every tenant
    unnamed = [_Advance(tenant=tenant, keys_unknown=True, noted_between=(since, until)) for tenant in tenants]
    return [*unnamed, *(_Advance(tenant=change.tenant, whole=change) for change in whole)]


async def _announce_table_window(
    pool: SqlConnection,
    *,
    generation_source: GenerationSource,
    notes_schema: str,
    table: str,
    since: datetime,
    until: datetime,
    relations: Sequence[RestoredTable],
    schema: str | None = None,
) -> _TableAnnounced:
    """advance ``table`` once for each change its rows stamped in the window name; nothing when there are none.

    :param pool: the L3 pool
    :ptype pool: SqlConnection
    :param generation_source: advances the table (the epoch bucket's writer)
    :ptype generation_source: GenerationSource
    :param notes_schema: the platform schema
    :ptype notes_schema: str
    :param table: the table
    :ptype table: str
    :param since: the window's start, by the database's clock, already widened by the clocks' skew
    :ptype since: datetime
    :param until: the window's end, likewise
    :ptype until: datetime
    :param relations: the table's relations (:func:`_switched_tables`)
    :ptype relations: Sequence[RestoredTable]
    :param schema: the one schema the window's writes ran in (a migration run's); ``None`` for any
    :ptype schema: str | None
    :return: what was recorded
    :rtype: _TableAnnounced
    :raises GenerationUnavailableError: when an advance failed; the advances before it stand
    """
    held = [relation for relation in relations if schema is None or relation.schema == schema]
    if not held:
        return _TableAnnounced(table=table, tokens=(), records=())
    tenants = _tenants(
        _tenancy_of(table, held[0].tenancy),
        [relation.schema for relation in relations],
        platform_schema=notes_schema,
        schema=schema,
    )
    tokens: list[str | None] = []
    records: list[Mapping[str, Any]] = []
    for advance in await _changes_of(
        pool,
        platform_schema=notes_schema,
        table=table,
        relations=held,
        since=since,
        until=until,
        tenants=tenants,
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
    tables: Mapping[str, tuple[RestoredTable, ...]] | None = None,
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
    :ptype tables: Mapping[str, tuple[RestoredTable, ...]] | None
    :return: what was done
    :rtype: _WindowAnnounced
    """
    if until < since:
        raise ValueError(f"a window ends after it starts: {since.isoformat()} .. {until.isoformat()}")
    switched = tables if tables is not None else await _switched_tables(pool)
    advanced: list[str] = []
    failed: dict[str, str] = {}
    for table, relations in sorted(switched.items()):
        if schema is not None and all(relation.schema != schema for relation in relations):
            continue
        try:
            done = await _announce_table_window(
                pool,
                generation_source=generation_source,
                notes_schema=notes_schema,
                table=table,
                since=since,
                until=until,
                relations=relations,
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


class WriterBeats:
    """whether a writer's beat is still kept: the liveness every writer process already publishes (:meth:`WriteBackstop
    .beat`), read by its key, for a reader told which writer to ask after (a load's intent record names its loading
    writer's beat key: ``EpochChange.unconfirmed_writer``, read by ``delta_recreate.DeltaRecreator``).

    A writer is gone once its key is: a writer stopping cleanly deletes its own (:meth:`WriteBackstop.stop`), and the
    replica holding the recovery lease deletes a dead writer's once it has announced its window
    (:meth:`WriteBackstop.recover_pass`). This reader keeps no clock of its own: a key still there is a writer not yet
    gone, however old its beat (the recovery judges that, once, for every reader).

    :param lease: a factory over the bucket the beats are kept in (the platform's ``leases``), opened as it is: a
        reader granted the beat keys alone builds it with ``create_if_missing=False``
    :ptype lease: KVLease
    """

    def __init__(self, lease: KVLease) -> None:
        self._lease = lease

    async def gone(self, key: str) -> bool:
        """whether the writer whose beat is kept under ``key`` is gone: the key holds no beat.

        :param key: the beat's key as stored (:attr:`WriteBackstop.key`)
        :ptype key: str
        :return: ``True`` once the key holds no value
        :rtype: bool
        :raises Exception: what the bucket's read raised (the caller asks again; never read as gone)
        """
        if not key.startswith(f"{_BEAT_PREFIX}."):
            raise ValueError(f"not a writer beat's key: {key!r}")
        bucket = await self._lease.bucket()
        return await bucket.get_entry(key=key) is None


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

    async def repair_pass(self, *, deadline: float | None = None) -> int:
        """make each lost advance of this process again; release those that land.

        :param deadline: a ``time.monotonic()`` past which no retry runs on (:meth:`WriteLedger.repair_pending`)
        :ptype deadline: float | None
        :return: how many are still held
        :rtype: int
        """
        return await self._ledger.repair_pending(deadline=deadline)

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
        self, table: str, relations: tuple[RestoredTable, ...], since: datetime, until: datetime, schema: str | None
    ) -> Callable[[], Any]:
        async def again() -> _TableAnnounced:
            return await _announce_table_window(
                self._pool,
                generation_source=self._source,
                notes_schema=self._notes_schema,
                table=table,
                since=since,
                until=until,
                relations=relations,
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
        switched: dict[str, tuple[RestoredTable, ...]] | None = None
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
        left = await self.repair_pass(deadline=deadline)
        while left and time.monotonic() < deadline:
            await asyncio.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
            left = await self.repair_pass(deadline=deadline)
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
