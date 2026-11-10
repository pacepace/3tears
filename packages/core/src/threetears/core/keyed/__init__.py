"""keyed reads: one helper for each shape a load, a cache or a listing reads by, never a table whole.

Every fix of a whole read takes one of a few shapes. Each is written once, here, so a fix calls it rather than writing
its SQL again (the no-whole-reads rule, :mod:`threetears.enforcement.no_whole_reads`, reads a call to one of these
as keyed by construction, judged by the key it names):

- **a keyed read** by a batch of keys or key prefixes, chunked: :func:`read_keyed`, :func:`read_keyed_arrow`
  (:class:`KeyBatch`, ``by`` pins a tenant or scope; :class:`AnyOf` is a set filter);
- **a listing's page**: a bound prefix, a closed sort, an opaque cursor, a ``LIMIT``, visibility as a keyed
  ``EXISTS``: :func:`keyset_page` (:class:`Sort`, :func:`exists_keyed`, :class:`Fragment`). An internal drain of
  every row a prefix names, a page a statement, from a saved place: :func:`threetears.core.keyset.keyset_pages`
  (``after=``);
- **counts** from the statement's own rows (:func:`rows_affected`), the planner's estimate (:func:`estimated_rows`),
  a bounded probe (:func:`holds_any`, :func:`empty_tables`, :func:`count_at_most`) or a keyed ``GROUP BY``
  (:func:`counts_by`);
- **a source's change record past a saved mark**: :func:`read_past_marks` (stamps), :func:`read_after` (a growing
  id), :func:`read_keyed` (the ids a runs record named);
- **an expiry sweep or a stale-row reap** a page at a time through its index: :func:`delete_before`,
  :func:`update_before`;
- **a cache derived from rows**, kept by key: :class:`threetears.epoch.RowFollower`;
- **the whole reads a component may still make**, named and counted: :class:`WholeReads`.

Every helper takes anything with an asyncpg-shaped ``fetch`` (and ``execute`` for a write): an L3 backend, the rail's
proxy, a connection, DuckDB through :class:`ThreadedDuckDB`, or SQLite through :class:`ThreadedSQLite`. The SQL is
one spelling for PostgreSQL, YugabyteDB and DuckDB; a database with no arrays (Redshift, SQLite) binds sets as a list
(``AnyOf(..., as_list=True)``, ``KeyBatch(..., as_list=True)``). Each is IO only, awaited on the caller's connection; none touches a row in Python beyond its answer.
"""

from __future__ import annotations

from threetears.core.keyed._filters import AnyOf, Fragment
from threetears.core.keyed.counts import (
    BoundedCount,
    Dialect,
    count_at_most,
    counts_by,
    empty_tables,
    estimated_rows,
    holds_any,
    rows_affected,
)
from threetears.core.keyed.deletes import SweptRows, delete_before, update_before
from threetears.core.keyed.threaded import ThreadedDuckDB, ThreadedSQLite
from threetears.core.keyed.marks import MarkedRead, PageAfter, read_after, read_past_marks
from threetears.core.keyed.page import KeyPage, Sort, exists_keyed, keyset_page
from threetears.core.keyed.rows import DEFAULT_KEY_BATCH, KeyBatch, KeyTooWide, read_keyed, read_keyed_arrow
from threetears.core.keyed.whole_reads import WholeReads

__all__ = [
    "DEFAULT_KEY_BATCH",
    "AnyOf",
    "BoundedCount",
    "Dialect",
    "Fragment",
    "KeyBatch",
    "KeyPage",
    "KeyTooWide",
    "MarkedRead",
    "PageAfter",
    "Sort",
    "SweptRows",
    "ThreadedDuckDB",
    "ThreadedSQLite",
    "WholeReads",
    "count_at_most",
    "counts_by",
    "delete_before",
    "empty_tables",
    "estimated_rows",
    "exists_keyed",
    "holds_any",
    "keyset_page",
    "read_after",
    "read_keyed",
    "read_keyed_arrow",
    "read_past_marks",
    "rows_affected",
    "update_before",
]
