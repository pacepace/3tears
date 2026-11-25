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

A leaf module: the backends and the hub import it, and it imports nothing of 3tears but the dependency-free
interfaces' spelling of a prefix value's text.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from threetears.interfaces.keyed.keys import prefix_text

__all__ = [
    "RUN_MARKERS_TABLE",
    "RUN_MARKER_COLUMNS",
    "run_marker_insert_sql",
]

#: the run-marker table, in the same schema as the versioned tables it marks
RUN_MARKERS_TABLE: Final[str] = "threetears_run_markers"

#: its columns, in order: the table marked, the prefix's scope value, the prefix and the delete's epoch
RUN_MARKER_COLUMNS: Final[tuple[str, ...]] = ("table_name", "scope", "prefix", "epoch")

#: what a marker's ``prefix`` joins the prefix's values with: chr(31), as a change note's ``key_text`` joins a key's
_PREFIX_SEPARATOR: Final[str] = "\x1f"


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
