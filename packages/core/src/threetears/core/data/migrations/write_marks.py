"""the migration runner gives each switched-on table its write-mark trigger, inside the run that creates it.

A table whose writes advance a write generation must carry the write-mark trigger from its first write
(:mod:`threetears.core.collections.write_marks`), or a commit whose advance is lost before the trigger
exists is caught by nothing. The package that creates such a table declares it
(:meth:`PackageMigrations.mark_write_generation`), from the table's ``TableSchema``; at the end of every
run the runner calls :func:`give_switched_tables_their_trigger` on the run's own session, which holds the
database-wide DDL lock. A table created before the trigger existed gets it on the first run after.

The trigger calls the queue's function where coordination migration v003 created it: the run's own
schema when it holds the function (the platform schema, or identity-core's), otherwise the one schema of
the database that does (an agent schema's tables, whose queue is the platform schema's). With neither,
nothing is created and the owner's install backstop gives the trigger later.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from threetears.observe import get_logger

__all__ = ["give_switched_tables_their_trigger"]

log = get_logger(__name__)

_FUNCTION_SCHEMAS = """
SELECT n.nspname AS schema FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE p.proname = $1 ORDER BY 1
"""

_MISSING_HERE = """
SELECT t.table_name AS table_name FROM information_schema.tables t
WHERE t.table_type = 'BASE TABLE' AND t.table_schema = current_schema() AND t.table_name::text = ANY($1::text[])
AND NOT EXISTS (
    SELECT 1 FROM pg_trigger g JOIN pg_class c ON c.oid = g.tgrelid JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE g.tgname = $2 AND c.relname = t.table_name AND n.nspname = t.table_schema
)
ORDER BY 1
"""


async def give_switched_tables_their_trigger(store: Any, tables: Sequence[str]) -> list[str]:
    """create the write-mark trigger on each named table of the run's schema that lacks it.

    Runs on the migration run's own store, inside its DDL lock; it takes no lock of its own.

    :param store: the run's locked ``DataStore``, bound to the target schema
    :ptype store: Any
    :param tables: the switched-on tables the run's packages declare
    :ptype tables: Sequence[str]
    :return: ``schema.table`` for each trigger created
    :rtype: list[str]
    """
    from threetears.core.collections.write_marks import (  # noqa: PLC0415 - the collections module imports this package
        MARK_FUNCTION,
        WRITE_MARK_TRIGGER,
        write_mark_trigger_sql,
    )

    here = str((await store.query("SELECT current_schema() AS schema"))[0]["schema"])
    holders = [str(row["schema"]) for row in await store.query(_FUNCTION_SCHEMAS, MARK_FUNCTION)]
    if here in holders:
        marks_schema = here
    elif len(holders) == 1:
        marks_schema = holders[0]
    else:
        if holders:
            log.warning(
                "switched-on tables were not given their write-mark trigger by this run: more than one schema "
                "holds the queue's function and none is this one; the install backstop gives it",
                extra={"extra_data": {"schema": here, "holders": holders}},
            )
        return []
    created: list[str] = []
    for row in await store.query(_MISSING_HERE, list(tables), WRITE_MARK_TRIGGER):
        table = str(row["table_name"])
        await store.execute(write_mark_trigger_sql(marks_schema=marks_schema, table_schema=here, table=table))
        created.append(f"{here}.{table}")
    if created:
        log.info("switched-on tables given their write-mark trigger", extra={"extra_data": {"tables": created}})
    return created
