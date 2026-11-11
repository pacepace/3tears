"""keyed statements' vocabulary and pure builders: the half of 3tears' keyed helpers that needs no connection.

Every keyed read and write the family makes is one of a few shapes, each written once. Their vocabulary (the set
filter :class:`AnyOf`, the trusted :class:`Fragment` and :class:`Selected`, the row :class:`Lock`, a batch of keys
:class:`KeyBatch`, a page's :class:`Sort`, a step's :class:`Match`, and the value types the runners answer) and the
builders writing each shape's SQL text and binds are pure, so they live here, in the package that depends on nothing:

- a point read: :func:`read_by_statement`, :func:`read_row_statement`;
- a keyed batch: :func:`read_keyed_statements` (:func:`keyed_read_statement`, :func:`key_chunks`);
- a listing's page, and a listing read to its end: :func:`keyset_page_statement` (:meth:`KeyPage.cut`),
  :func:`keyset_rows_statement` (:func:`next_past`); a visibility probe :func:`exists_keyed`;
- a keyed write: :func:`update_keyed_statements`, :func:`delete_keyed_statements`;
- an expiry sweep, one page a statement: :func:`delete_before_statement`, :func:`update_before_statement`.

The runners (anything that executes on a connection: the transport's row cap, the epoch-aware sweeps, the
YugabyteDB check, the Arrow reply) are ``threetears.core.keyed``'s, which re-exports every name here, so a consumer
of core changes nothing. A package that may not depend on core (``3tears-evals``, ``3tears-iam``) calls a builder
and runs its statement on its own :class:`KeyedConnection` (or SQLite, through :func:`sqlite_numbered`), writing no
SQL by hand. The no-whole-reads rule judges a builder's call as it judges its runner's.
"""

from __future__ import annotations

from threetears.interfaces.keyed.filters import (
    IN_SUBQUERY,
    PLACEHOLDER,
    SQL_TYPE,
    AnyOf,
    Derived,
    Fragment,
    Lock,
    Pin,
    Selected,
    batched_join_hint,
    by_conditions,
    narrowing,
    prefix_conditions,
    quote_identifier,
    quoted_table,
    relation,
    renumbered,
    selected_column,
    sqlite_numbered,
)
from threetears.interfaces.keyed.connection import (
    KeyedConnection,
)
from threetears.interfaces.keyed.point import (
    read_by_statement,
    read_row_statement,
)
from threetears.interfaces.keyed.rows import (
    ANY_ROWS_CAP,
    DEFAULT_KEY_BATCH,
    DIRECT_KEY_BATCH,
    KeyBatch,
    KeyTooWide,
    key_chunks,
    keyed_read_statement,
    read_keyed_statements,
)
from threetears.interfaces.keyed.page import (
    CursorError,
    KeyPage,
    Sort,
    check_page,
    cursor_values,
    exists_keyed,
    keyset_cursor,
    keyset_page_statement,
    keyset_rows_statement,
    next_past,
    page_statement,
    seeks_by_row,
)
from threetears.interfaces.keyed.writes import (
    Increment,
    KeyedWrite,
    StageMode,
    StageWrite,
    delete_keyed_statements,
    update_keyed_statements,
)
from threetears.interfaces.keyed.sweeps import (
    SweptRows,
    delete_before_statement,
    update_before_statement,
)
from threetears.interfaces.keyed.counts import (
    NO_MARK_FLOOR,
    SQL_DIALECT_ATTRIBUTE,
    BoundedCount,
    Dialect,
    MarkedCount,
    rows_affected,
)
from threetears.interfaces.keyed.marks import (
    MarkedRead,
    PageAfter,
)
from threetears.interfaces.keyed.steps import (
    Match,
    Quoted,
    Rel,
    Step,
    change_relation,
)
from threetears.interfaces.keyed.changed import (
    ROW_HASH_COLUMN,
    ChangedRows,
    GroupHash,
    HashColumn,
    HashKind,
)

__all__ = [
    "ANY_ROWS_CAP",
    "AnyOf",
    "BoundedCount",
    "ChangedRows",
    "CursorError",
    "DEFAULT_KEY_BATCH",
    "DIRECT_KEY_BATCH",
    "Derived",
    "Dialect",
    "Fragment",
    "GroupHash",
    "HashColumn",
    "HashKind",
    "IN_SUBQUERY",
    "Increment",
    "KeyBatch",
    "KeyPage",
    "KeyTooWide",
    "KeyedConnection",
    "KeyedWrite",
    "Lock",
    "MarkedCount",
    "MarkedRead",
    "Match",
    "NO_MARK_FLOOR",
    "PLACEHOLDER",
    "PageAfter",
    "Pin",
    "Quoted",
    "ROW_HASH_COLUMN",
    "Rel",
    "SQL_DIALECT_ATTRIBUTE",
    "SQL_TYPE",
    "Selected",
    "Sort",
    "StageMode",
    "StageWrite",
    "Step",
    "SweptRows",
    "batched_join_hint",
    "by_conditions",
    "change_relation",
    "check_page",
    "cursor_values",
    "delete_before_statement",
    "delete_keyed_statements",
    "exists_keyed",
    "key_chunks",
    "keyed_read_statement",
    "keyset_cursor",
    "keyset_page_statement",
    "keyset_rows_statement",
    "narrowing",
    "next_past",
    "page_statement",
    "prefix_conditions",
    "quote_identifier",
    "quoted_table",
    "read_by_statement",
    "read_keyed_statements",
    "read_row_statement",
    "relation",
    "renumbered",
    "rows_affected",
    "seeks_by_row",
    "selected_column",
    "sqlite_numbered",
    "update_before_statement",
    "update_keyed_statements",
]
