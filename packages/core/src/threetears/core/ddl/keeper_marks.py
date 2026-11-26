"""the log keeper's marks table: the change log's position, by the database's clock, every so often.

A mark (P, T) says the change log stood at position P when the database's clock read T. A restore of a lost log reads
the versioned and stamped tables from X = (T - M) << 15 (:func:`~threetears.core.ddl.row_epoch.restore_floor`) and
labels the new log as covering every record after P (:mod:`threetears.core.collections.table_restore`). The newest mark
also names the log the keeper last knew (:mod:`threetears.core.collections.log_keeper`).

The table keeps its name from the change-note ring it outlived (the cutover's ``ring_cutover.drop_ring`` keeps it), so
a database cut over keeps every mark its window needs.
"""

from __future__ import annotations

from typing import Final

from threetears.core.sql_fragments import quote_identifier

__all__ = ["MARKS_TABLE", "keeper_marks_ddl"]

#: the keeper's marks
MARKS_TABLE: Final = "threetears_change_note_marks"

#: the marks table, ``{schema}`` its schema (quoted): written out whole, so it declares its key where it is read
_MARKS_TABLE_DDL: Final = (
    'CREATE TABLE IF NOT EXISTS {schema}."threetears_change_note_marks" ('
    "id UUID PRIMARY KEY DEFAULT gen_random_uuid(), log_incarnation TEXT NOT NULL, seq BIGINT NOT NULL, "
    "at TIMESTAMPTZ NOT NULL)"
)

#: the marks table's indexes, ``{schema}`` its schema (quoted): a restore reads one lost log's marks in position order;
#: the window's sweep deletes the marks older than its anchor, a page at a time by their time, and every pass reads the
#: newest mark (the log the keeper knows) from the same time index, backwards
_MARKS_INDEX_DDL: Final = (
    "-- purpose: packages/core/src/threetears/core/collections/log_keeper.py::LogKeeper.marks_past\n"
    "-- plan_test: packages/core/tests/integration/test_keeper_marks_plans.py::test_threetears_change_note_marks_position_reads_a_restores_marks\n"
    'CREATE INDEX IF NOT EXISTS "threetears_change_note_marks_position" ON {schema}."threetears_change_note_marks" '
    "(log_incarnation, seq ASC, id ASC)",
    "-- purpose: packages/core/src/threetears/core/collections/log_keeper.py::LogKeeper.drop_marks_before; "
    "packages/core/src/threetears/core/collections/log_keeper.py::LogKeeper.newest_mark\n"
    "-- plan_test: packages/core/tests/integration/test_keeper_marks_plans.py::test_threetears_change_note_marks_at_reads_the_windows_sweep\n"
    'CREATE INDEX IF NOT EXISTS "threetears_change_note_marks_at" ON {schema}."threetears_change_note_marks" (at ASC, id ASC)',
)


def keeper_marks_ddl(schema: str) -> tuple[str, ...]:
    """the statements that create the marks table and its indexes where missing, in order.

    :param schema: the schema holding the marks, a TRUSTED identifier
    :ptype schema: str
    :return: the statements
    :rtype: tuple[str, ...]
    """
    owner = quote_identifier(schema)
    return (
        _MARKS_TABLE_DDL.format(schema=owner),
        *(statement.format(schema=owner) for statement in _MARKS_INDEX_DDL),
    )
