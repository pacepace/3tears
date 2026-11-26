"""run markers: a whole run of a versioned table deleted as ONE row, "run X gone at epoch E".

Append-only L3 (``design-append-only-l3.md``, decision 5, Pace 2026-11-25). A versioned table
(:func:`~threetears.core.ddl.row_epoch.row_versions`) is never deleted from: a key's delete appends a "gone" marker for
the key. A whole run deleted (a load's gone LSF run, the key prefix ``(join_state_code, run_id)``) would be a marker a
row, ~1.4 M rows for one night's 23 runs, so it is ONE row of this table instead: the table, the prefix and the epoch
of the delete. A read skips every row of the table under the prefix whose epoch is older than the newest marker of the
prefix; cleanup clears the run's rows and its marker together, by the same lifetime rule as a key's marker.

**The row.** ``table_name`` the versioned table; ``scope`` the prefix's value of the table's scope column (the column
a restore reads "changed since epoch X" by, :class:`~threetears.core.ddl.row_epoch.RowEpoch`), so a restore of one
scope finds the scope's run markers by the same key; ``prefix`` the prefix's values as text in key order joined by
:data:`_PREFIX_SEPARATOR` (the form a change note's ``key_text`` takes, so a read compares it with the row's own key
columns cast to text the same way); ``epoch`` the delete's epoch, stamped as every write's is. The prefix must name
the scope column, which makes the scope one of its values.

**Who writes it.** The store that runs the delete: on the L3 rail the hub's broker (``l3.tx.delete_prefixes``, the
request carrying the epoch), on a direct connection 3tears' ``SqlL3Backend.delete_prefixes``; both through
:func:`run_marker_insert_sql`, so the text form of every value is the database's own. Its declaration, the table's DDL
for Postgres and YugabyteDB, is :data:`threetears.core.collections.run_markers.RUN_MARKERS_SCHEMA`.

**A table reset is one row too** (:func:`table_reset_marker_sql`). A ``TRUNCATE`` (an operator's reset, a
super-admin's) leaves no row for a restore or the writers' backstop to read ``epoch >= X`` from, and since the
change-note ring retired (0.95) no note says so. So whatever resets a table that carries an epoch writes, before its
``TRUNCATE`` and on the same connection, the table's reset marker: ``scope`` :data:`TABLE_RESET_SCOPE` and an empty
``prefix``, the whole table, at the reset's epoch. A restore, the backstop and a keys-unknown page read it by the key
``(table_name, scope)`` (``table_restore.table_resets``) and hear the whole table cleared, as a reset. No run's marker
takes that key: a run's ``scope`` is a value of the table's scope column, which a scope epochs table names, and
:data:`TABLE_RESET_SCOPE` is no such value; nor does a read of a row probe it, since every row's prefix names values.

A leaf module: the backends and the hub import it, and it imports nothing of 3tears but the dependency-free
interfaces' spelling of a prefix value's text and the restore declaration's mark.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from threetears.core.ddl.row_epoch import RESTORE_DECLARATION_MARK
from threetears.interfaces.keyed.keys import prefix_text

__all__ = [
    "RUN_MARKERS_TABLE",
    "RUN_MARKER_COLUMNS",
    "TABLE_RESET_SCOPE",
    "TableResetUnrecordedError",
    "record_table_reset",
    "run_marker_insert_sql",
    "table_reset_marker_sql",
]

#: the run-marker table, in the same schema as the versioned tables it marks
RUN_MARKERS_TABLE: Final[str] = "threetears_run_markers"

#: its columns, in order: the table marked, the prefix's scope value, the prefix and the delete's epoch
RUN_MARKER_COLUMNS: Final[tuple[str, ...]] = ("table_name", "scope", "prefix", "epoch")

#: what a marker's ``prefix`` joins the prefix's values with: chr(31), as a change note's ``key_text`` joins a key's
_PREFIX_SEPARATOR: Final[str] = "\x1f"

#: the ``scope`` of a table's reset marker (its ``prefix`` empty): the whole table, emptied at the marker's epoch.
#: chr(30), the record separator: no scope column's value a scope epochs table names, and no run's prefix
TABLE_RESET_SCOPE: Final[str] = "\x1e"


def _text_term(placeholder: str, declared: str) -> str:
    """a bound prefix value as the marker's text holds it, cast by the database from its column's own type: the
    spelling a read's probe of the marker uses on the row's own column
    (:func:`~threetears.interfaces.keyed.keys.prefix_text`), so the two texts are the same.

    :param placeholder: the parameter (``$3``)
    :ptype placeholder: str
    :param declared: its column's type as declared (``format_type``), a TRUSTED string read from the catalog or a
        schema
    :ptype declared: str
    :return: the SQL term
    :rtype: str
    """
    return prefix_text(placeholder, declared)


def run_marker_insert_sql(declared: Sequence[str], scope_position: int, *, table: str = RUN_MARKERS_TABLE) -> str:
    """the statement that writes one run marker: ``$1`` the table marked, ``$2`` the delete's epoch, ``$3``... the
    prefix's values in key order, each bound as its column's type and cast to text by the database.

    :param declared: each prefix column's type as declared (``format_type``), in key order; TRUSTED (the catalog's or a
        schema's)
    :ptype declared: Sequence[str]
    :param scope_position: the scope column's place among the prefix's columns
    :ptype scope_position: int
    :param table: the run-marker table as the statement names it (bare, resolved by the session's ``search_path``)
    :ptype table: str
    :return: the ``INSERT``
    :rtype: str
    :raises ValueError: when the prefix is empty or the scope is not one of its columns
    """
    if not declared:
        raise ValueError("a run marker's prefix names at least one key column")
    if not 0 <= scope_position < len(declared):
        raise ValueError(f"a run marker's scope is one of its prefix's {len(declared)} columns, got {scope_position}")
    terms = [_text_term(f"${index + 3}", kind) for index, kind in enumerate(declared)]
    return (
        f"INSERT INTO {table} ({', '.join(RUN_MARKER_COLUMNS)}) "
        f"VALUES ($1, {terms[scope_position]}, "
        f"array_to_string(ARRAY[{', '.join(terms)}], chr({ord(_PREFIX_SEPARATOR)})), $2)"
    )


def table_reset_marker_sql(*, table: str = RUN_MARKERS_TABLE) -> str:
    """the statement that writes one table's reset marker: ``$1`` the table reset, ``$2`` the reset's epoch; its scope
    :data:`TABLE_RESET_SCOPE`, its prefix empty (the whole table).

    Written before the reset's ``TRUNCATE``, on its connection (``SqlL3Backend.truncate``; the hub's broker and its
    data-space truncate the same way): a marker whose ``TRUNCATE`` then fails only has followers read the table again.

    :param table: the run-marker table as the statement names it: bare (the session's ``search_path``), or
        ``schema.table`` quoted by the caller, TRUSTED
    :ptype table: str
    :return: the ``INSERT``
    :rtype: str
    """
    return f"INSERT INTO {table} ({', '.join(RUN_MARKER_COLUMNS)}) VALUES ($1, chr({ord(TABLE_RESET_SCOPE)}), '', $2)"


class TableResetUnrecordedError(RuntimeError):
    """a reset of a table that carries an epoch, refused before its ``TRUNCATE``: its schema holds no run-marker table,
    so nothing could say the table was emptied, and a restore or the writers' backstop would never hear it."""


#: the table a reset names, as the catalog holds it: its schema, whether one of its indexes carries a restore
#: declaration (a versioned or stamped table, which a restore reads back), and whether its schema holds the run-marker
#: table. ``$1`` the table as the statement names it, ``$2`` the declaration's ``LIKE`` pattern. A catalog read by oid
_RESET_TARGET_SQL: Final[str] = f"""
SELECT n.nspname::text AS table_schema, c.relname::text AS table_name,
       EXISTS (
           SELECT 1 FROM pg_index i
           JOIN pg_description d ON d.objoid = i.indexrelid AND d.classoid = 'pg_class'::regclass AND d.objsubid = 0
           WHERE i.indrelid = c.oid AND d.description LIKE $2
       ) AS declared,
       to_regclass(quote_ident(n.nspname) || '.{RUN_MARKERS_TABLE}') IS NOT NULL AS run_markers
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.oid = to_regclass($1)
"""


async def record_table_reset(conn: Any, table: str, epoch: int) -> bool:
    """write ``table``'s reset marker (:func:`table_reset_marker_sql`) at ``epoch``, before its ``TRUNCATE``, on the
    connection that runs it: a restore, the writers' backstop and a keys-unknown page then hear the whole table cleared.

    A table that carries no restore declaration is read back by no restore, so nothing is written for it. A table that
    carries one, in a schema with no run-marker table, is refused: the reset would be lost with the change log.

    WHY before the ``TRUNCATE`` and committed on its own, not in one transaction with it: on YugabyteDB a ``TRUNCATE``
    takes effect apart from the transaction around it, so a marker committed with that transaction could be lost after
    the rows were. Written first, the marker is never missing; a ``TRUNCATE`` that then fails leaves a marker that only
    has followers read the table again.

    :param conn: the connection that runs the reset (asyncpg-shaped: ``fetch``, ``execute``), no transaction open
    :ptype conn: Any
    :param table: the table as the ``TRUNCATE`` names it (``table`` or ``schema.table``), TRUSTED (a plain identifier)
    :ptype table: str
    :param epoch: the reset's epoch (``row_epoch.next_epoch()``)
    :ptype epoch: int
    :return: whether a marker was written
    :rtype: bool
    :raises TableResetUnrecordedError: when the table carries an epoch and its schema holds no run-marker table
    """
    rows = await conn.fetch(_RESET_TARGET_SQL, table, f'{{"{RESTORE_DECLARATION_MARK}":%')
    if not rows or not rows[0]["declared"]:
        return False
    schema = str(rows[0]["table_schema"])
    if not rows[0]["run_markers"]:
        raise TableResetUnrecordedError(
            f"refusing to reset {table}: it carries an epoch (a restore reads it back from its rows), and its schema "
            f"{schema!r} holds no {RUN_MARKERS_TABLE} table to record the reset in, so a restore after a NATS loss and "
            f"the writers' backstop would never hear the table emptied. Fix: create the run-marker table in "
            f"{schema!r} (threetears.core.collections.run_markers.RUN_MARKERS_SCHEMA.create_statements) in the "
            f"migration that creates the table, then reset again"
        )
    quoted = '"' + schema.replace('"', '""') + '".' + RUN_MARKERS_TABLE
    await conn.execute(table_reset_marker_sql(table=quoted), str(rows[0]["table_name"]), epoch)
    return True
