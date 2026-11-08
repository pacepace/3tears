"""the migration runner gives each noted table its change-note triggers, inside the run that creates it.

A noted table (:func:`~threetears.core.collections.generation.noted_tables`) must carry its change-note
triggers from its first write (:mod:`threetears.core.collections.change_notes`), or the writes made before
the keeper's next ``install_triggers`` leave no note and a restore after a NATS loss would not put them back.
At the end of every run the runner calls :func:`give_noted_tables_their_note_trigger` on the run's own
session, which holds the database-wide DDL lock, with each table the run's packages declare switched on
(:meth:`PackageMigrations.mark_write_generation`) that this process's classes note, with their keys and
invalidation columns. A noted table no package declares gets its triggers from the keeper's install.

The triggers call the ring's functions where the keeper installed them: the run's own schema when it holds
the ring, otherwise the one schema of the database that does. With neither (the ring not yet installed, as
on a database's first start), nothing is created and the keeper's install gives the triggers later.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from threetears.core.collections.generation import NotedTable
from threetears.core.ddl.change_notes import (
    NOTE_PUT_FUNCTION,
    NOTE_TRIGGER,
    NOTE_TRUNCATE_TRIGGER,
    inexact_note_columns,
    note_function_name,
    note_trigger_sql,
    trigger_function_body,
    trigger_function_sql,
)
from threetears.observe import get_logger

__all__ = ["give_noted_tables_their_note_trigger"]

log = get_logger(__name__)

_RING_SCHEMAS = """
SELECT n.nspname AS schema FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE p.proname = $1 ORDER BY 1
"""

_TABLES_HERE = """
SELECT t.table_name AS table_name FROM information_schema.tables t
WHERE t.table_type = 'BASE TABLE' AND t.table_schema = current_schema() AND t.table_name::text = ANY($1::text[])
ORDER BY 1
"""

_TRIGGERS_HERE = """
SELECT c.relname AS table_name, g.tgname AS trigger_name
FROM pg_trigger g JOIN pg_class c ON c.oid = g.tgrelid JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = current_schema() AND g.tgname::text = ANY($1::text[])
"""

_COLUMN_TYPES_HERE = """
SELECT c.table_name::text AS table_name, c.column_name::text AS column_name, c.data_type::text AS data_type
FROM information_schema.columns c WHERE c.table_schema = current_schema() AND c.table_name::text = ANY($1::text[])
"""

_FUNCTION_BODIES = """
SELECT p.proname AS name, p.prosrc AS body FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = $1 AND p.proname::text = ANY($2::text[])
"""


async def give_noted_tables_their_note_trigger(store: Any, tables: Sequence[NotedTable]) -> list[str]:
    """create the note function and triggers of each noted table of the run's schema that lacks them.

    Runs on the migration run's own store, inside its DDL lock; it takes no lock of its own. A function
    whose key or columns changed is replaced, as the keeper's install does.

    :param store: the run's locked ``DataStore``, bound to the target schema
    :ptype store: Any
    :param tables: the noted tables this process declares
    :ptype tables: Sequence[NotedTable]
    :return: ``schema.table`` for each table given anything
    :rtype: list[str]
    """
    noted = {each.table: each for each in tables}
    if not noted:
        return []
    here = str((await store.query("SELECT current_schema() AS schema"))[0]["schema"])
    holders = [str(row["schema"]) for row in await store.query(_RING_SCHEMAS, NOTE_PUT_FUNCTION)]
    if here in holders:
        notes_schema = here
    elif len(holders) == 1:
        notes_schema = holders[0]
    else:
        if holders:
            log.warning(
                "noted tables were not given their change-note triggers by this run: more than one schema holds "
                "the ring and none is this one; the keeper's install gives them",
                extra={"extra_data": {"schema": here, "holders": holders}},
            )
        return []
    present = [str(row["table_name"]) for row in await store.query(_TABLES_HERE, sorted(noted))]
    if not present:
        return []
    triggers = {
        (str(row["table_name"]), str(row["trigger_name"]))
        for row in await store.query(_TRIGGERS_HERE, [NOTE_TRIGGER, NOTE_TRUNCATE_TRIGGER])
    }
    types: dict[str, dict[str, str]] = {}
    for row in await store.query(_COLUMN_TYPES_HERE, present):
        types.setdefault(str(row["table_name"]), {})[str(row["column_name"])] = str(row["data_type"])
    names = [note_function_name(here, table) for table in present]
    bodies = {str(row["name"]): str(row["body"]) for row in await store.query(_FUNCTION_BODIES, notes_schema, names)}
    given: list[str] = []
    for table in present:
        each = noted[table]
        refused = inexact_note_columns(types.get(table, {}), (*each.keys, *each.columns))
        if refused:
            log.error(
                "a noted table gets no change-note trigger: a key or invalidation column's text form does not "
                "name its value exactly (only uuid, text, varchar and integers do)",
                extra={"extra_data": {"table": f"{here}.{table}", "columns": [f"{c} ({k})" for c, k in refused]}},
            )
            continue
        wanted = trigger_function_body(notes_schema=notes_schema, keys=each.keys, columns=each.columns)
        stale = bodies.get(note_function_name(here, table)) != wanted
        missing = [name for name in (NOTE_TRIGGER, NOTE_TRUNCATE_TRIGGER) if (table, name) not in triggers]
        if not stale and not missing:
            continue
        if stale:
            await store.execute(
                trigger_function_sql(
                    notes_schema=notes_schema, table_schema=here, table=table, keys=each.keys, columns=each.columns
                )
            )
        row, truncate = note_trigger_sql(notes_schema=notes_schema, table_schema=here, table=table)
        for name, statement in ((NOTE_TRIGGER, row), (NOTE_TRUNCATE_TRIGGER, truncate)):
            if name in missing:
                await store.execute(statement)
        given.append(f"{here}.{table}")
    if given:
        log.info("noted tables given their change-note triggers", extra={"extra_data": {"tables": given}})
    return given
