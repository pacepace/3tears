"""keyed reads: one helper for each shape a load, a cache or a listing reads by, never a table whole.

Every fix of a whole read takes one of a few shapes. Each is written once, here, so a fix calls it rather than writing
its SQL again (the no-whole-reads rule, :mod:`threetears.enforcement.no_whole_reads`, reads a call to one of these
as keyed by construction, judged by the key it names):

- **a keyed read** by a batch of keys or key prefixes, chunked: :func:`read_keyed`, :func:`read_keyed_arrow`
  (:class:`KeyBatch`, ``by`` pins a tenant or scope; :class:`AnyOf` is a set filter);
- **a listing's page**: a bound prefix, a closed sort, an opaque cursor, a ``LIMIT``, visibility as a keyed
  ``EXISTS``, filters on a partial index's predicate or a range (``where=``), a row lock (:class:`Lock`):
  :func:`keyset_page` (:class:`Sort`, :func:`exists_keyed`, :class:`Fragment`); its joined rows by a keyed read of
  the values the page holds: :func:`read_joined`. An internal drain of
  every row a prefix names, a page a statement, from a saved place: :func:`threetears.core.keyset.keyset_pages`
  (``after=``);
- **a bounded look**, any rows up to a cap in no order (an operator command's "any 500"): :func:`any_rows`;
- **counts** from the statement's own rows (:func:`rows_affected`), the planner's estimate (:func:`estimated_rows`),
  a bounded probe (:func:`holds_any`, :func:`empty_tables`, :func:`count_at_most`, keyed or of a table's first rows)
  or a keyed ``GROUP BY`` (:func:`counts_by`; several relations in one statement: :func:`counts_by_relations`);
- a relation is a table (``schema.table`` quoted a segment at a time) or a TRUSTED derived ``SELECT``
  (:class:`Derived`), judged on its own text where it is written;
- **a source's change record past a saved mark**: :func:`read_past_marks` (stamps), :func:`read_after` (a growing
  id), :func:`read_keyed` (the ids a runs record named);
- **a write by key** (a status flip, a stamp, a fence-checked save, a delete of a key's rows): :func:`update_keyed`,
  :func:`delete_keyed` (:class:`KeyedWrite`, by ``by`` or a :class:`KeyBatch`; :class:`Increment`), a table's columns
  from a keyed derived select (:func:`update_from`), a stage relation written by its key (:func:`apply_stage`:
  replace, upsert or the difference); a listing's every row, page after page: :func:`keyset_rows`;
- **an expiry sweep or a stale-row reap** a page at a time through its index: :func:`delete_before`,
  :func:`update_before`; a cache's trim to its cap by its key's order (SQLite's rowid): :func:`delete_oldest`;
- **a cache derived from rows**, kept by key: :class:`threetears.epoch.RowFollower`;
- **the whole reads a component may still make**, named and counted: :class:`WholeReads`.

Every helper takes anything with an asyncpg-shaped ``fetch`` (and ``execute`` for a write): an L3 backend, the rail's
proxy, a connection, DuckDB through :class:`ThreadedDuckDB`, or SQLite through :class:`ThreadedSQLite`; synchronous
code on its own DuckDB connection, in its own transaction, through :class:`InlineDuckDB` and :func:`run_inline`. The
SQL is one spelling for PostgreSQL, YugabyteDB and DuckDB; a database with no arrays (Redshift, SQLite) binds sets as a list
(``AnyOf(..., as_list=True)``, ``KeyBatch(..., as_list=True)``). Each is IO only, awaited on the caller's connection; none touches a row in Python beyond its answer.
"""

from __future__ import annotations

from threetears.core.keyed._filters import AnyOf, Derived, Fragment, Lock, Selected
from threetears.core.keyed.counts import (
    BoundedCount,
    Dialect,
    count_at_most,
    counts_by,
    counts_by_relations,
    empty_tables,
    estimated_rows,
    holds_any,
    rows_affected,
)
from threetears.core.keyed.deletes import SweptRows, delete_before, delete_oldest, update_before
from threetears.core.keyed.inline import InlineDuckDB, InlineTransaction, run_inline
from threetears.core.keyed.threaded import ThreadedDuckDB, ThreadedSQLite
from threetears.core.keyed.marks import MarkedRead, PageAfter, read_after, read_past_marks
from threetears.core.keyed.page import KeyPage, Sort, exists_keyed, keyset_page, keyset_rows
from threetears.core.keyed.rows import (
    ANY_ROWS_CAP,
    DEFAULT_KEY_BATCH,
    KeyBatch,
    KeyTooWide,
    any_rows,
    read_joined,
    read_keyed,
    read_keyed_arrow,
)
from threetears.core.keyed.whole_reads import WholeReads
from threetears.core.keyed.writes import (
    Increment,
    KeyedWrite,
    StageMode,
    StageWrite,
    apply_stage,
    delete_keyed,
    update_from,
    update_keyed,
)

__all__ = [
    "ANY_ROWS_CAP",
    "DEFAULT_KEY_BATCH",
    "AnyOf",
    "BoundedCount",
    "Derived",
    "Dialect",
    "Fragment",
    "Increment",
    "InlineDuckDB",
    "InlineTransaction",
    "KeyBatch",
    "KeyPage",
    "KeyTooWide",
    "KeyedWrite",
    "Lock",
    "MarkedRead",
    "PageAfter",
    "Selected",
    "Sort",
    "StageMode",
    "StageWrite",
    "SweptRows",
    "ThreadedDuckDB",
    "ThreadedSQLite",
    "WholeReads",
    "any_rows",
    "apply_stage",
    "count_at_most",
    "counts_by",
    "counts_by_relations",
    "delete_before",
    "delete_keyed",
    "delete_oldest",
    "empty_tables",
    "estimated_rows",
    "exists_keyed",
    "holds_any",
    "keyset_page",
    "keyset_rows",
    "read_after",
    "read_joined",
    "read_keyed",
    "read_keyed_arrow",
    "read_past_marks",
    "rows_affected",
    "run_inline",
    "update_before",
    "update_from",
    "update_keyed",
]
