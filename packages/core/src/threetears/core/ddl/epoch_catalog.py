"""what the catalog says of a table's epoch: whether it carries one, whether it is versioned, and its per-row triggers.

Append-only L3 (``design-append-only-l3.md``, decisions 3 and 4; build plan chunk 7). A page-written table is versioned
(:func:`~threetears.core.ddl.row_epoch.row_versions`): it is its own change record, read back through its
``(scope, epoch)`` index, and needs no change note. A per-row trigger on it is the slow path the design removes: on
YugabyteDB a table with a row trigger loses its writes' read batching (YugabyteDB's docs: batching is "always disabled
for ... relations that have row triggers"; 2026-11-25, 1.58 ms a row with the change-note trigger against 0.43 without,
5,000 existing keys of ``latest_results_candidates_by_id``). So 3tears gives a versioned table no per-row trigger, and
refuses one found on it (:func:`refuse_row_triggers`), at each migration run's end and at the keeper's install, so the
slow path cannot come back by a migration, a hand-written statement or an older replica.

**Which tables are versioned, from the catalog.** A versioned table's ``(scope, epoch)`` index carries its restore
declaration as its comment (:func:`~threetears.core.ddl.row_epoch.restore_declaration`); a stamped table's
``(bucket, epoch)`` index carries a stamped one (``"kind":"stamped"``). The catalog read here takes the first and not
the second, so it needs no class of the table: the hub's process, which imports no consumer's tables, judges them too.

**Whether a table carries an epoch.** Its ``epoch`` column (:data:`~threetears.core.ddl.row_epoch.EPOCH_COLUMN`),
``BIGINT NOT NULL``: a write that carries none is refused by the database itself. Every versioned and stamped table
carries it (:func:`tables_carrying_epoch`).

A leaf module of catalog statements: it imports no collection and no Arrow, so the migration runner reaches it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final, Protocol

from threetears.core.ddl.row_epoch import EPOCH_COLUMN, RESTORE_DECLARATION_MARK

__all__ = [
    "REFUSE_TABLES_WITHOUT_EPOCH",
    "PerRowTriggerError",
    "TableWithoutEpochError",
    "refuse_tables_without_epoch",
    "refuse_row_triggers",
    "row_triggers_on_versioned",
    "tables_carrying_epoch",
    "versioned_among",
]


class _Fetching(Protocol):
    """anything with an asyncpg-shaped ``fetch`` (a pool, a connection)."""

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """rows of ``query``."""
        ...


#: what a restore declaration's comment starts with, and what a stamped one carries besides (both written with no
#: spaces, ``separators=(",", ":")``)
_DECLARED: Final = f'{{"{RESTORE_DECLARATION_MARK}":%'
_STAMPED: Final = '%"kind":"stamped"%'

#: the relations whose index carries a versioned table's restore declaration (not a stamped one's), as a subquery of
#: ``(table_schema, table_name, relid)``: a catalog read, its size the declared indexes
_VERSIONED_RELATIONS: Final = """
SELECT DISTINCT n.nspname::text AS table_schema, c.relname::text AS table_name, c.oid AS relid
FROM pg_description d
JOIN pg_index i ON i.indexrelid = d.objoid
JOIN pg_class c ON c.oid = i.indrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE d.classoid = 'pg_class'::regclass AND d.objsubid = 0 AND d.description LIKE $1 AND d.description NOT LIKE $2
"""

#: the versioned tables of one schema among those named
_VERSIONED_AMONG_SQL: Final = f"""
SELECT v.table_name FROM ({_VERSIONED_RELATIONS}) v
WHERE v.table_schema = $3 AND v.table_name = ANY($4::text[])
ORDER BY 1
"""

#: every per-row trigger a user made (not a constraint's internal one) on a versioned table, of one schema or of every
#: schema (``$3`` null). ``tgtype & 1`` is ``TRIGGER_TYPE_ROW`` (``pg_trigger.h``; the same on YugabyteDB)
_ROW_TRIGGERS_SQL: Final = f"""
SELECT v.table_schema, v.table_name, g.tgname::text AS trigger_name
FROM ({_VERSIONED_RELATIONS}) v
JOIN pg_trigger g ON g.tgrelid = v.relid
WHERE NOT g.tgisinternal AND (g.tgtype::int & 1) = 1 AND ($3::text IS NULL OR v.table_schema = $3::text)
ORDER BY 1, 2, 3
"""


#: the tables of one schema among those named that carry the epoch column, ``BIGINT NOT NULL``
_CARRYING_EPOCH_SQL: Final = """
SELECT c.relname::text AS table_name
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname::text = $1 AND c.relname::text = ANY($2::text[]) AND a.attname::text = $3
  AND a.attnotnull AND NOT a.attisdropped AND a.atttypid = 'int8'::regtype
ORDER BY 1
"""


class PerRowTriggerError(RuntimeError):
    """a versioned (page-written) table carries a per-row trigger: every write to it would pay the slow path.

    :ivar found: ``schema.table: trigger`` for each one found
    """

    def __init__(self, found: tuple[str, ...]) -> None:
        self.found = found
        listed = "\n".join(f"  - {each}" for each in found)
        super().__init__(
            "\n\n"
            "=====================================================================================================\n"
            "REFUSED: a versioned (page-written) table carries a PER-ROW TRIGGER\n"
            "=====================================================================================================\n"
            f"{listed}\n\n"
            "A versioned table (TableSchema(row_epoch=row_versions(...))) is append-only and is its own change\n"
            "record: a restore reads it back through its (scope, epoch) index, so it needs no change note and no\n"
            "trigger. On YugabyteDB a row trigger switches off the writes' read batching, and each row then pays its\n"
            "own read and flush (1.58 ms a row against 0.43 without, 2026-11-25). 3tears refuses to run beside one.\n\n"
            "How to fix:\n"
            "  1. Drop the trigger in a migration: DROP TRIGGER IF EXISTS <trigger> ON <schema>.<table>;\n"
            "     A change-note trigger (threetears_change_note*) left by an older release goes with\n"
            "     threetears.core.data.migrations.ring_cutover.drop_ring (the cutover's step).\n"
            "  2. Whatever the trigger computed, compute in the write itself (the broker stamps every row's epoch;\n"
            "     a derived column is a GENERATED column, never a trigger).\n"
            "  3. Run the migrations again; this check runs at the end of every migration run and at the keeper's\n"
            "     install.\n"
            "=====================================================================================================\n"
        )


async def versioned_among(conn: _Fetching, schema: str, tables: Iterable[str]) -> frozenset[str]:
    """the versioned tables of ``schema`` among ``tables``, from the restore declaration their index carries.

    :param conn: a pool or a connection
    :ptype conn: _Fetching
    :param schema: the schema
    :ptype schema: str
    :param tables: the table names
    :ptype tables: Iterable[str]
    :return: those that are versioned
    :rtype: frozenset[str]
    """
    names = sorted(set(tables))
    if not names:
        return frozenset()
    rows = await conn.fetch(_VERSIONED_AMONG_SQL, _DECLARED, _STAMPED, schema, names)
    return frozenset(str(row["table_name"]) for row in rows)


async def row_triggers_on_versioned(conn: _Fetching, schema: str | None = None) -> tuple[str, ...]:
    """every per-row trigger on a versioned table, of ``schema`` or (None) of every schema, as ``schema.table: name``.

    :param conn: a pool or a connection
    :ptype conn: _Fetching
    :param schema: the one schema judged; None judges every schema
    :ptype schema: str | None
    :return: the triggers found, in schema, table and trigger order
    :rtype: tuple[str, ...]
    """
    rows = await conn.fetch(_ROW_TRIGGERS_SQL, _DECLARED, _STAMPED, schema)
    return tuple(f"{row['table_schema']}.{row['table_name']}: {row['trigger_name']}" for row in rows)


async def refuse_row_triggers(conn: _Fetching, schema: str | None = None) -> None:
    """refuse a per-row trigger on any versioned table of ``schema`` (None: of every schema).

    :param conn: a pool or a connection
    :ptype conn: _Fetching
    :param schema: the one schema judged; None judges every schema
    :ptype schema: str | None
    :return: nothing
    :rtype: None
    :raises PerRowTriggerError: when one is found, naming each
    """
    found = await row_triggers_on_versioned(conn, schema)
    if found:
        raise PerRowTriggerError(found)


async def tables_carrying_epoch(conn: _Fetching, schema: str, tables: Iterable[str]) -> frozenset[str]:
    """the tables of ``schema`` among ``tables`` that carry the epoch column (``BIGINT NOT NULL``).

    :param conn: a pool or a connection
    :ptype conn: _Fetching
    :param schema: the schema
    :ptype schema: str
    :param tables: the table names
    :ptype tables: Iterable[str]
    :return: those that carry it
    :rtype: frozenset[str]
    """
    names = sorted(set(tables))
    if not names:
        return frozenset()
    rows = await conn.fetch(_CARRYING_EPOCH_SQL, schema, names, EPOCH_COLUMN)
    return frozenset(str(row["table_name"]) for row in rows)


#: TODO(work/0950-stamp-own-tables): set True once that branch has merged. A switched-on table (one whose writes
#: advance a generation) that carries no epoch is refused at the migration run that declares it
#: (:func:`refuse_tables_without_epoch`): with the change-note ring retired (append-only L3 chunk 7) nothing records
#: its writes for a restore, the writers' backstop or a keys-unknown page, so a lost advance of it could never be
#: announced. Held off only because 3tears' own ten tables do not declare ``ROW_STAMPED`` yet (that branch does it);
#: ``tests/unit/ddl/test_tables_without_epoch_refused.py`` fails while this is False, so it cannot be forgotten
REFUSE_TABLES_WITHOUT_EPOCH: Final = False


class TableWithoutEpochError(RuntimeError):
    """switched-on tables that carry no epoch: their writes would reach no reader of what changed.

    :ivar tables: ``schema.table`` of each
    """

    def __init__(self, tables: Iterable[str]) -> None:
        self.tables = tuple(sorted(tables))
        super().__init__(
            "switched-on tables carry no epoch, so a restore, the writers' backstop and a keys-unknown page would "
            f"never hear their writes: {list(self.tables)}. How to fix: declare each table's row epoch in its "
            "TableSchema, row_versions(scope, ...) for a page-written table or ROW_STAMPED for one row a key updated "
            "in place, and migrate it (epoch BIGINT NOT NULL, its restore index); or stop declaring a write "
            "generation for it"
        )


async def refuse_tables_without_epoch(conn: _Fetching, schema: str, tables: Iterable[str]) -> frozenset[str]:
    """the switched-on ``tables`` of ``schema`` that carry no epoch; refused (:class:`TableWithoutEpochError`) once
    :data:`REFUSE_TABLES_WITHOUT_EPOCH` is on. One catalogue read.

    :param conn: a pool or a connection
    :ptype conn: _Fetching
    :param schema: the schema
    :ptype schema: str
    :param tables: the switched-on tables a run declares
    :ptype tables: Iterable[str]
    :return: those without an epoch (empty once refused, since a refusal raises)
    :rtype: frozenset[str]
    :raises TableWithoutEpochError: when one carries none and the refusal is on
    """
    names = frozenset(tables)
    missing = names - await tables_carrying_epoch(conn, schema, names)
    if missing and REFUSE_TABLES_WITHOUT_EPOCH:
        raise TableWithoutEpochError(f"{schema}.{name}" for name in missing)
    return missing
