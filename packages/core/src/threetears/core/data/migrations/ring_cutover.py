"""the outage-window cutover's step that takes the change-note ring down: its triggers, functions and tables.

Append-only L3 chunk 7 (``design-append-only-l3.md``, decision 3: the ring is retired entirely). Rollout is one
outage-window cutover (Pace 2026-11-25): every writer is scaled to zero, then this runs once per database that held a
ring (the hub's platform database; identity-core's), then the new release starts. Run while nothing writes: a writer of
an older release still running would find its note function gone and fail its write, which is the outage the window
accepts, never a silent miss.

What :func:`drop_ring` takes down, under the database's DDL lock, in this order:

1. every change-note trigger on every table of the database (the ring's three trigger names and the older truncate
   trigger's), by ``DROP TRIGGER IF EXISTS``, a table at a time from the catalog;
2. every note function of the ring's schemas (the per-table ``threetears_note_*`` and the ring's own
   ``threetears_change_note_*``), by ``DROP FUNCTION IF EXISTS`` with its exact arguments, never ``CASCADE``;
3. the ring's tables (its three note tables, its ring row, the note buffer's guard tables), by one ``DROP TABLE IF
   EXISTS``. WHY ``DROP TABLE`` and never ``TRUNCATE`` or a ``DELETE``: the tables go whole during the outage, and
   ``TRUNCATE`` bumps YugabyteDB's catalog version cluster-wide.

What it keeps: the keeper's marks table (:data:`~threetears.core.ddl.keeper_marks.MARKS_TABLE`). The log's keeper
still marks the log's position, and a restore from the tables reads X from a mark (``table_restore``).

Idempotent: a second run finds nothing and drops nothing. Each step is said at WARNING with what it dropped, and the
answer (:class:`RingDropped`) names it for the cutover script's own log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from threetears.core.ddl.ddl_lock import DdlLockPolicy, database_ddl_lock
from threetears.core.ddl.keeper_marks import MARKS_TABLE
from threetears.core.ddl.session import ConnectionSession
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import get_logger

__all__ = ["RING_TABLES", "RingDropped", "drop_ring"]

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

#: every table carrying a ring trigger, with the trigger: a catalog read, its size the triggers
_TRIGGERS_SQL: Final = """
SELECT n.nspname::text AS table_schema, c.relname::text AS table_name, g.tgname::text AS trigger_name
FROM pg_trigger g
JOIN pg_class c ON c.oid = g.tgrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE NOT g.tgisinternal AND g.tgname::text = ANY($1::text[])
ORDER BY 1, 2, 3
"""

#: the schemas holding a ring: those with its ring row's table
_RING_SCHEMAS_SQL: Final = """
SELECT n.nspname::text AS schema FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relname::text = $1 AND c.relkind = 'r'
ORDER BY 1
"""

#: every note function of a schema, with its identity arguments (the per-table ``threetears_note_*`` and the ring's
#: ``threetears_change_note_*``)
_FUNCTIONS_SQL: Final = r"""
SELECT p.proname::text AS name, pg_get_function_identity_arguments(p.oid) AS args
FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname::text = $1 AND (p.proname::text LIKE 'threetears\_note\_%' OR p.proname::text LIKE 'threetears\_change\_note%')
ORDER BY 1, 2
"""

#: the ring's tables present in a schema
_TABLES_SQL: Final = """
SELECT c.relname::text AS table_name FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname::text = $1 AND c.relkind = 'r' AND c.relname::text = ANY($2::text[])
ORDER BY 1
"""


@dataclass(frozen=True)
class RingDropped:
    """what :func:`drop_ring` took down.

    :ivar triggers: ``schema.table: trigger`` for each trigger dropped
    :ivar functions: ``schema.function(args)`` for each function dropped
    :ivar tables: ``schema.table`` for each ring table dropped
    """

    triggers: tuple[str, ...] = ()
    functions: tuple[str, ...] = ()
    tables: tuple[str, ...] = field(default=())


async def drop_ring(pool: Any, *, policy: DdlLockPolicy | None = None) -> RingDropped:
    """take the change-note ring down in the pool's database: its triggers, its functions and its tables (not its
    marks), under the database's DDL lock. The outage-window cutover's step; run while nothing writes.

    :param pool: the database's pool (an ``asyncpg.Pool``), allowed DDL
    :ptype pool: Any
    :param policy: how long to wait for the DDL lock; the default waits until it is free
    :ptype policy: DdlLockPolicy | None
    :return: what was dropped; nothing on a second run
    :rtype: RingDropped
    """
    triggers: list[str] = []
    functions: list[str] = []
    tables: list[str] = []
    async with pool.acquire() as conn:
        async with database_ddl_lock(ConnectionSession(conn), policy):
            for row in await conn.fetch(_TRIGGERS_SQL, list(_TRIGGER_NAMES)):
                schema, table, trigger = str(row["table_schema"]), str(row["table_name"]), str(row["trigger_name"])
                await conn.execute(
                    f"DROP TRIGGER IF EXISTS {quote_identifier(trigger)} "
                    f"ON {quote_identifier(schema)}.{quote_identifier(table)}"
                )
                triggers.append(f"{schema}.{table}: {trigger}")
            schemas = [str(row["schema"]) for row in await conn.fetch(_RING_SCHEMAS_SQL, RING_TABLE)]
            for schema in schemas:
                for row in await conn.fetch(_FUNCTIONS_SQL, schema):
                    name, args = str(row["name"]), str(row["args"])
                    await conn.execute(
                        f"DROP FUNCTION IF EXISTS {quote_identifier(schema)}.{quote_identifier(name)}({args})"
                    )
                    functions.append(f"{schema}.{name}({args})")
                present = [str(row["table_name"]) for row in await conn.fetch(_TABLES_SQL, schema, list(RING_TABLES))]
                if present:
                    listed = ", ".join(f"{quote_identifier(schema)}.{quote_identifier(table)}" for table in present)
                    # one statement: the guard tables' deferred keys reach the note tables, so they go together
                    await conn.execute(f"DROP TABLE IF EXISTS {listed}")
                    tables.extend(f"{schema}.{table}" for table in present)
    dropped = RingDropped(triggers=tuple(triggers), functions=tuple(functions), tables=tuple(tables))
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
