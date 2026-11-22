"""what is left of the write-mark queue: its names, its table, and the release the table is dropped by.

Until 0.95 every write to a switched-on table ran a per-row trigger that inserted a mark into
``collection_write_marks`` in its own transaction, and a reconciler advanced the table from each mark. It cost each
write on YugabyteDB about three times an upsert (a per-row trigger defeats the batched existence check) plus a row
inserted and later deleted, and its reconciler raced writers into re-announcing their commits (2026-11-22). Its job --
announcing an advance lost after a commit -- is now the writer's own, with no L3 write a commit
(:mod:`threetears.core.collections.lost_writes`). Coordination migration v006 takes every mark trigger and the
function down.

**Rolling upgrade (expand/contract).** The table stays, empty, until :data:`WRITE_MARKS_CONTRACT_RELEASE`: an older
replica mid-roll still reads it (its writers read and clear their own marks, its reconciler counts and drains it), and
its install backstop may put triggers back, which the new backstop takes down again
(:meth:`~threetears.core.collections.lost_writes.WriteBackstop.drop_leftover_write_mark_triggers`). The contract migration drops the
table in that release; ``tests/unit/coordination/test_write_marks_contract.py`` fails the build there until it does.
"""

from __future__ import annotations

from typing import Final

from threetears.core.sql_fragments import quote_identifier

__all__ = [
    "MARK_FUNCTION",
    "WRITE_MARKS_CONTRACT_RELEASE",
    "WRITE_MARKS_TABLE",
    "WRITE_MARK_TRIGGER",
    "write_marks_ddl",
]

#: the release whose contract migration drops :data:`WRITE_MARKS_TABLE`, and by which the leftover-trigger sweep is gone
WRITE_MARKS_CONTRACT_RELEASE: Final[str] = "0.96.0"

#: the queue older replicas still read mid-roll, empty from 0.95 on
WRITE_MARKS_TABLE: Final[str] = "collection_write_marks"

#: the queue's table, ``{schema}`` its schema (quoted): written out whole, so it declares the queue's key where it is read
_MARKS_TABLE_DDL: Final[str] = (
    'CREATE TABLE IF NOT EXISTS {schema}."collection_write_marks" ('
    "id BIGSERIAL PRIMARY KEY, "
    "table_name TEXT NOT NULL, "
    "date_created TIMESTAMPTZ NOT NULL DEFAULT now())"
)

#: the trigger function the mark triggers called; dropped by coordination v006
MARK_FUNCTION: Final[str] = "threetears_mark_collection_write"

#: the mark trigger's name on every switched-on table; dropped by coordination v006
WRITE_MARK_TRIGGER: Final[str] = "threetears_collection_write_mark"


def write_marks_ddl(schema: str) -> tuple[str, ...]:
    """the statement that creates the (now empty) marks table in ``schema``, idempotently: coordination v003's body,
    kept so a new database's ledger holds the same versions as an old one's. Its ``(table_name, id)`` index (v004) is
    created no more: an older replica reads it only on a database that already has it.

    :param schema: the schema that holds the queue (the platform schema)
    :ptype schema: str
    :return: the statements, in order; run them one at a time (YugabyteDB runs no DDL in a transaction)
    :rtype: tuple[str, ...]
    """
    return (_MARKS_TABLE_DDL.format(schema=quote_identifier(schema)),)
