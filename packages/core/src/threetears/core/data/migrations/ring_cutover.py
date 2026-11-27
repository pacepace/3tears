"""the outage-window cutover's step that takes the change-note ring down: its triggers, functions and tables.

Append-only L3 chunk 7 (``design-append-only-l3.md``, decision 3: the ring is retired entirely). Rollout is one
outage-window cutover (Pace 2026-11-25): every writer is scaled to zero, then this runs once per database that held a
ring (the hub's platform database; identity-core's), then the new release starts. Run while nothing writes: a writer of
an older release still running would find its note function gone and fail its write, which is the outage the window
accepts, never a silent miss.

What :func:`drop_ring_on` takes down, under the database's DDL lock, in this order:

1. every change-note trigger on every table of the database, in every schema (the platform schema's and every agent
   schema's, ``ns_*``): one by the ring's three trigger names or the older truncate trigger's, or one whose function
   is a note function, by ``DROP TRIGGER IF EXISTS``, read from the catalog;
2. then, every trigger calling one gone, every note function of the database (the per-table ``threetears_note_*``
   and the ring's own ``threetears_change_note_*``), by ``DROP FUNCTION IF EXISTS`` with its exact arguments, never
   ``CASCADE``. A table's function lives in the ring's schema while its triggers live on the table, in the table's
   schema: dropped a schema at a time, a function went before another schema's triggers and the drop was refused;
3. the ring's tables (its three note tables, its ring row, the note buffer's guard tables), by one ``DROP TABLE IF
   EXISTS``. WHY ``DROP TABLE`` and never ``TRUNCATE`` or a ``DELETE``: the tables go whole during the outage, and
   ``TRUNCATE`` bumps YugabyteDB's catalog version cluster-wide.

What it keeps: the keeper's marks table (:data:`~threetears.core.ddl.keeper_marks.MARKS_TABLE`). The log's keeper
still marks the log's position, and a restore from the tables reads X from a mark (``table_restore``).

Idempotent: a second run finds nothing and drops nothing. Each statement commits on its own (the run holds no
transaction around a migration body, on PostgreSQL or YugabyteDB), so a run that fails part way leaves what it dropped
dropped, and a rerun reads what is left from the catalog and finishes the job. Each step is said at WARNING with what it dropped, and the
answer (:class:`RingDropped`) names it for the caller's own log.

One way in. :func:`drop_ring_on` runs on a session the caller already holds, for a boot migration (the hub's v193;
identity-core's start): the run holds the DDL lock on its one connection, and a second connection would wait for that
lock forever (the run waiting on itself). On the held session the lock is re-entrant, so the step takes it again at
once and gives back only its own hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from threetears.core.data.store import DataStore
from threetears.core.ddl.ddl_lock import DdlLockPolicy, database_ddl_lock
from threetears.core.ddl.keeper_marks import MARKS_TABLE
from threetears.core.ddl.session import MigrationSession
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import get_logger

__all__ = ["RING_TABLES", "RingDropped", "drop_ring_on"]

log = get_logger(__name__)

#: the ring's one-row table, naming its current note table: a schema holding it held a ring
RING_TABLE: Final = "threetears_change_note_ring"

#: the ring's three note tables, ``threetears_change_notes_0`` to ``_2``
_NOTE_TABLES: Final = tuple(f"threetears_change_notes_{index}" for index in range(3))

#: every ring trigger name a table may carry: the row note, the truncate note, the whole-write note, and the name an
#: older release gave the truncate note. Spelled here, since the ring's own DDL module retired with it
_TRIGGER_NAMES: Final = (
    "threetears_change_note",
    "threetears_change_note_truncate_table",
    "threetears_change_note_whole_write",
    "threetears_change_note_truncate",
)

#: the ring's tables: its note tables, its ring row, and the note buffer's guard tables (a connection's buffer is a
#: temporary table, gone with its session). Never the marks table
RING_TABLES: Final = (
    *_NOTE_TABLES,
    RING_TABLE,
    "threetears_note_buffer_guard",
    "threetears_note_buffer_never",
)

#: the schemas a lookup never reads: the catalog's own, TOAST's, and every session's temporary schema (a writer's note
#: buffer was a temporary table, gone with its session, and another session's temporary objects cannot be dropped)
_SKIPPED_SCHEMAS: Final = (
    "n.nspname NOT IN ('pg_catalog', 'information_schema') "
    r"AND n.nspname NOT LIKE 'pg\_toast%' AND n.nspname NOT LIKE 'pg\_temp\_%'"
)

#: a note function's name: a table's own (``threetears_note_`` and a digest of its schema and table) or the ring's
#: (``threetears_change_note_put``, ``_whole``, ``_buffer_open``, ``_flush``)
_RING_FUNCTION_NAME: Final = (
    r"(p.proname::text LIKE 'threetears\_note\_%' OR p.proname::text LIKE 'threetears\_change\_note%')"
)

#: every ring trigger of the database, in every schema: one by a ring trigger name, or one whose function is a note
#: function whatever its name. A catalog read, its size the triggers.
#:
#: WHY the whole database before any function goes (hub v193 on a database from before 0.95, 2026-11-27): a table's
#: note function lives in the ring's schema (the platform schema), but its three triggers live on the table, in the
#: table's own schema: every agent schema's (``ns_*``) noted tables call the platform schema's functions. Dropped a
#: schema at a time, the platform schema's functions went before a later schema's triggers, and ``DROP FUNCTION`` was
#: refused: ``cannot drop function threetears_note_... because other objects depend on it``
_TRIGGERS_SQL: Final = f"""
SELECT n.nspname::text AS schema, c.relname::text AS table_name, g.tgname::text AS trigger_name
FROM pg_trigger g
JOIN pg_class c ON c.oid = g.tgrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_proc p ON p.oid = g.tgfoid
WHERE NOT g.tgisinternal AND {_SKIPPED_SCHEMAS}
  AND (g.tgname::text = ANY($1::text[]) OR {_RING_FUNCTION_NAME})
ORDER BY 1, 2, 3
"""

#: every note function of the database, in every schema, with its identity arguments
_FUNCTIONS_SQL: Final = f"""
SELECT n.nspname::text AS schema, p.proname::text AS name, pg_get_function_identity_arguments(p.oid) AS args
FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE {_SKIPPED_SCHEMAS} AND {_RING_FUNCTION_NAME}
ORDER BY 1, 2, 3
"""

#: the ring's tables present in the database, in every schema
_TABLES_SQL: Final = f"""
SELECT n.nspname::text AS schema, c.relname::text AS table_name
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE {_SKIPPED_SCHEMAS} AND c.relkind = 'r' AND c.relname::text = ANY($1::text[])
ORDER BY 1, 2
"""


@dataclass(frozen=True)
class RingDropped:
    """what :func:`drop_ring_on` took down.

    :ivar triggers: ``schema.table: trigger`` for each trigger dropped
    :ivar functions: ``schema.function(args)`` for each function dropped
    :ivar tables: ``schema.table`` for each ring table dropped
    """

    triggers: tuple[str, ...] = ()
    functions: tuple[str, ...] = ()
    tables: tuple[str, ...] = field(default=())


async def drop_ring_on(session: MigrationSession, *, policy: DdlLockPolicy | None = None) -> RingDropped:
    """take the change-note ring down on a session the caller holds: its triggers, its functions and its tables (not
    its marks), every statement on that one session, under the database's DDL lock.

    The boot migration's step. A session that already holds the DDL lock takes it again at once (a session-level
    lock is re-entrant) and gives back only this step's hold, so the caller still holds it after. A session that does
    not hold it waits for it under ``policy``.

    :param session: one database session: a :class:`~threetears.core.ddl.session.ConnectionSession` over the held
        connection, or a :class:`~threetears.core.data.store.DataStore` (a migration body's store, which holds the run's
        lock, is used as it is; any other store is bound to one connection for the step by ``DataStore.ddl_session``)
    :ptype session: MigrationSession
    :param policy: how long to wait for the DDL lock when the session does not hold it; the default waits until it is
        free
    :ptype policy: DdlLockPolicy | None
    :return: what was dropped; nothing on a second run
    :rtype: RingDropped
    """
    if isinstance(session, DataStore):
        # a store yields itself when it holds the lock, takes it on its own session when bound to one, and otherwise
        # pins one pooled connection: never a statement per pooled connection, which would drop the session lock
        async with session.ddl_session(policy) as locked:
            dropped = await _drop_under_lock(locked)
    else:
        async with database_ddl_lock(session, policy):
            dropped = await _drop_under_lock(session)
    log.warning(
        "the change-note ring was taken down for the append-only cutover (the keeper's marks are kept)",
        extra={
            "extra_data": {
                "triggers": len(dropped.triggers),
                "functions": len(dropped.functions),
                "tables": list(dropped.tables),
                "marks_kept": MARKS_TABLE,
            }
        },
    )
    return dropped


async def _drop_under_lock(session: MigrationSession) -> RingDropped:
    """the three drops, each across the whole database, on a session holding the DDL lock: every ring trigger in
    every schema, then every note function, then the ring's tables.

    Each statement commits on its own (the migration run holds no transaction around a body, on either engine; on
    YugabyteDB DDL is a statement each anyway), so a run that fails part way leaves what it dropped dropped. Every list
    is read from the catalog as it stands, so a rerun finds only what is left and finishes the job.

    :param session: one session, holding the database's DDL lock
    :ptype session: MigrationSession
    :return: what was dropped
    :rtype: RingDropped
    """
    triggers: list[str] = []
    functions: list[str] = []
    tables: list[str] = []
    for row in await session.query(_TRIGGERS_SQL, list(_TRIGGER_NAMES)):
        schema, table, trigger = str(row["schema"]), str(row["table_name"]), str(row["trigger_name"])
        await session.execute(
            f"DROP TRIGGER IF EXISTS {quote_identifier(trigger)} ON {quote_identifier(schema)}.{quote_identifier(table)}"
        )
        triggers.append(f"{schema}.{table}: {trigger}")
    # every trigger calling a note function is gone, in every schema: a function goes without CASCADE, so anything
    # else depending on one refuses the drop loudly rather than going with it
    for row in await session.query(_FUNCTIONS_SQL):
        schema, name, args = str(row["schema"]), str(row["name"]), str(row["args"])
        await session.execute(f"DROP FUNCTION IF EXISTS {quote_identifier(schema)}.{quote_identifier(name)}({args})")
        functions.append(f"{schema}.{name}({args})")
    present = [
        (str(row["schema"]), str(row["table_name"])) for row in await session.query(_TABLES_SQL, list(RING_TABLES))
    ]
    if present:
        listed = ", ".join(f"{quote_identifier(schema)}.{quote_identifier(table)}" for schema, table in present)
        # one statement: the guard tables' deferred keys reach the note tables, so they go together
        await session.execute(f"DROP TABLE IF EXISTS {listed}")
        tables.extend(f"{schema}.{table}" for schema, table in present)
    return RingDropped(triggers=tuple(triggers), functions=tuple(functions), tables=tuple(tables))
