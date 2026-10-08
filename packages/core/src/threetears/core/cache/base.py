"""L1 cache backend protocol, and the row-age policy every backend shares.

Holds four things, not one:

- :class:`L1Backend`, the protocol every L1 backend implements.
- :data:`MISSING`, the cache-miss sentinel (distinct from a cached ``None``).
- The cached-at stamp: :data:`CACHED_AT_COLUMN`, the tables exempt from it, and
  :func:`entry_is_fresh`, which is the single copy of the max-age predicate so
  two backends cannot disagree about what "expired" means.
- :func:`build_select_clause`, shared SQL construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from threetears.core.sql_fragments import quote_identifier

__all__ = [
    "CACHED_AT_COLUMN",
    "L1Backend",
    "MISSING",
    "TABLES_WITHOUT_CACHE_STAMP",
    "build_select_clause",
    "bulk_columns",
    "entry_is_fresh",
    # released here (v0.66.0) before it moved to threetears.core.sql_fragments; still importable
    "quote_identifier",
]

MISSING = object()
"""Sentinel for cache miss. Distinct from None (which is a valid cached value)."""

CACHED_AT_COLUMN = "_3t_cached_at"
"""Reserved L1 column holding the monotonic reading at which a row was pulled through.

Injected into generated entity tables by the backend, never declared by a
caller's SQLAlchemy metadata, and stripped from every row a read returns --
so it is invisible above the cache tier and a table declaring it is a
collision, not a contribution.

The leading underscore and the ``3t`` prefix are the collision guard.
``collection_scan_cache`` already carries its own ``stored_at_monotonic``
(``collections/scan_cache.py``), which is why this is not simply named for
what it holds: a blanket injection under that name would emit a duplicate
column, and a blanket strip under it would break the scan cache's own read.
"""

TABLES_WITHOUT_CACHE_STAMP: frozenset[str] = frozenset(
    {
        # Not entity caches. They ride the same L1 backend but are internal
        # bookkeeping with their own lifetimes: the scan cache already has
        # its own monotonic stamp, and the write buffer holds pending L3
        # writes whose age means something entirely different.
        #
        # These literals are owned by ``collections/scan_cache.py`` and
        # ``collections/flush.py``, which declare the tables. They are repeated
        # rather than imported because ``cache/`` must not import ``collections/``
        # -- the dependency runs the other way, and inverting it to save two
        # strings would be the worse trade. ``test_exempt_tables_match_their_
        # declarations`` fails if the two ever disagree, so the duplication
        # cannot drift silently.
        "collection_scan_cache",
        "write_buffer",
    }
)
"""Tables the cache stamp is never injected into."""


def entry_is_fresh(
    stored_at_monotonic: float | None,
    *,
    now_monotonic: float,
    max_age_seconds: float,
) -> bool:
    """Whether a cached entry stamped at ``stored_at_monotonic`` is still within its max age.

    Shared by the age-bounded cache tiers so the rule cannot drift
    between them, the same reason :func:`build_select_clause` is shared.

    Public because more than one module calls it -- ``cache/sqlite.py`` and
    ``collections/scan_cache.py`` -- and a leading underscore is a stability
    contract that a sibling module of the package may not bind to.

    Both readings come from :func:`time.monotonic` in the *same*
    process. That is what makes this safe where a wall-clock comparison
    would not be: no clock is shared with another host, so there is no
    skew to be wrong about, and a monotonic reading cannot step backwards
    under an NTP correction. The corollary is a constraint on the
    caller, not on this function -- an L1 tier whose storage outlives the
    process cannot use it, because a reading taken by one process means
    nothing to another.

    Callers supply the clock reading rather than this function taking
    one, so a test can exercise an hour-long window without sleeping.

    :param stored_at_monotonic: the reading taken when the entry was
        cached, or ``None`` when the entry carries no stamp
    :ptype stored_at_monotonic: float | None
    :param now_monotonic: caller-supplied monotonic clock reading
    :ptype now_monotonic: float
    :param max_age_seconds: how long an entry stays fresh
    :ptype max_age_seconds: float
    :return: ``True`` when the entry may still be served
    :rtype: bool
    """
    # An unstamped entry has never been obtained from a lower tier: it
    # holds a value this process authored and nothing else knows yet.
    # Expiring it would discard a local write in favour of the older
    # value a pull-through would return, so it is fresh by definition.
    #
    # This covers a DEFERRED flush only for a row this process AUTHORED. A
    # row that was pulled through earlier keeps its stamp across a local
    # save (``upsert`` preserves a stamp the caller does not supply), so it
    # can age out while its write is still sitting in the write buffer. The
    # pull-through that follows reads L2, which the same save already wrote,
    # so the new value comes back and nothing reverts. Without L2 wired it
    # would read L3 and serve the pre-write value. Stated because an earlier
    # version of this comment claimed the no-stamp rule covered the deferred
    # case outright, and it does not.
    if stored_at_monotonic is None:
        return True
    return now_monotonic - stored_at_monotonic <= max_age_seconds


def build_select_clause(
    schema: dict[str, str] | None,
    table: str,
    columns: Sequence[str] | None,
) -> str:
    """Build a validated SELECT column list, ``*`` when unprojected.

    Shared by every backend so projection validation cannot drift
    between them.

    :param schema: the table's registered column-to-type mapping, or
        ``None``/empty when the table is not registered; validation is
        skipped then and the engine reports unknown columns itself
    :ptype schema: dict[str, str] | None
    :param table: target table name, used in error messages
    :ptype table: str
    :param columns: requested projection, or ``None`` for all columns;
        duplicates collapse, first occurrence wins
    :ptype columns: Sequence[str] | None
    :return: the SELECT clause column list
    :rtype: str
    :raises ValueError: if ``columns`` is empty, or names a column the
        registered schema does not have
    """
    if columns is None:
        return "*"
    deduped = list(dict.fromkeys(columns))
    if not deduped:
        raise ValueError("columns must be None or a non-empty sequence")
    if schema:
        unknown = [c for c in deduped if c not in schema]
        if unknown:
            raise ValueError(f"unknown columns for table {table}: {unknown}")
    return ", ".join(quote_identifier(c) for c in deduped)


def bulk_columns(rows: Sequence[Mapping[str, Any]], schema: Mapping[str, str]) -> list[str]:
    """the columns a bulk write writes: those the rows name, in the table's order, all rows alike.

    :param rows: the rows to write
    :ptype rows: Sequence[Mapping[str, Any]]
    :param schema: the table's declared columns (empty when unknown: every named column is written)
    :ptype schema: Mapping[str, str]
    :return: the columns, filtered to the table's as ``upsert`` filters them
    :rtype: list[str]
    :raises ValueError: when the rows do not all name the same columns
    """
    named = set(rows[0]) if rows else set()
    ragged = next((i for i, row in enumerate(rows) if set(row) != named), None)
    if ragged is not None:
        raise ValueError(f"row {ragged} names different columns from row 0; a bulk write needs every row alike")
    return [c for c in schema if c in named] if schema else sorted(named)


@runtime_checkable
class L1Backend(Protocol):
    """Protocol defining the interface for L1 cache backends.

    All methods are synchronous — L1 cache is local in-memory,
    so async adds overhead for no benefit.
    """

    def initialize(self, sa_metadata: Any) -> None:
        """Initialize the backend with schema derived from SQLAlchemy metadata."""
        ...

    def get_connection(self) -> Any:
        """Return a connection (or connection proxy) for the current thread."""
        ...

    def upsert(self, table: str, data: dict[str, Any], primary_key: str | tuple[str, ...] = "id") -> None:
        """insert or update row atomically.

        :param table: destination table name
        :ptype table: str
        :param data: row data keyed by column name
        :ptype data: dict[str, Any]
        :param primary_key: pk column name (single-PK) or tuple of pk
            column names in declared order (composite-PK). all pk
            columns named here MUST be present in ``data``.
        :ptype primary_key: str | tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        ...

    def upsert_many(
        self, table: str, rows: Sequence[Mapping[str, Any]], primary_key: str | tuple[str, ...] = "id"
    ) -> int:
        """insert or update many rows in one statement, as ``upsert`` would one by one.

        every row must name the same columns; a bulk write with ragged rows has
        no single meaning for the columns some rows leave out.

        :param table: destination table name
        :ptype table: str
        :param rows: the rows, each keyed by column name, every pk column present
        :ptype rows: Sequence[Mapping[str, Any]]
        :param primary_key: pk column name or tuple of pk column names
        :ptype primary_key: str | tuple[str, ...]
        :return: how many rows were written
        :rtype: int
        :raises ValueError: when the rows do not all name the same columns
        """
        ...

    def column_types(self, table: str) -> Mapping[str, str]:
        """the type codes this backend reads and writes a table's columns by, by column name.

        the backend's serialization codes (``TEXT_UUID``, ``VARCHAR_JSON``), not SQL
        types; a backend's own bookkeeping columns are not included.

        :param table: the table
        :ptype table: str
        :return: each column's type code (empty for a table it does not know)
        :rtype: Mapping[str, str]
        """
        ...

    def select_by_id(
        self,
        table: str,
        entity_id: Any,
        primary_key: str | tuple[str, ...] = "id",
        columns: Sequence[str] | None = None,
        *,
        max_age_seconds: float | None = None,
        now_monotonic: float | None = None,
    ) -> dict[str, Any] | None:
        """select single row by primary key, returning None on miss.

        ``max_age_seconds`` bounds how long a row cached from a lower tier may
        be served: past it the row is deleted and the read reports a miss, so
        the caller pulls through. ``None`` (the default) disables expiry, which
        is what every caller that has no lower tier to pull from must use.
        ``now_monotonic`` lets a test drive the window without sleeping.

        :param table: target table name
        :ptype table: str
        :param entity_id: pk value (single-PK) or tuple of pk values in
            declared column order (composite-PK). length of tuple MUST
            equal length of ``primary_key`` tuple.
        :ptype entity_id: Any
        :param primary_key: pk column name (single-PK) or tuple of pk
            column names in declared order (composite-PK)
        :ptype primary_key: str | tuple[str, ...]
        :param columns: columns to select and deserialize; ``None``
            selects every column. Exactly the named columns come back --
            pk columns are NOT implicitly added. Projection skips
            deserialization of every unselected column, which is the
            point: a wide row with one large JSON column costs its full
            parse on every unprojected read.
        :ptype columns: Sequence[str] | None
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        :raises ValueError: if ``columns`` is empty, or names a column
            the table's registered schema does not have
        """
        ...

    def select_batch(
        self,
        table: str,
        entity_ids: list[Any],
        primary_key: str | tuple[str, ...] = "id",
        columns: Sequence[str] | None = None,
        *,
        max_age_seconds: float | None = None,
        now_monotonic: float | None = None,
    ) -> list[dict[str, Any]]:
        """select multiple rows by primary key.

        :param table: target table name
        :ptype table: str
        :param entity_ids: list of pk values (single-PK) or list of
            tuples of pk values (composite-PK). every tuple MUST match
            the length of ``primary_key``.
        :ptype entity_ids: list[Any]
        :param primary_key: pk column name (single-PK) or tuple of pk
            column names in declared order (composite-PK)
        :ptype primary_key: str | tuple[str, ...]
        :param columns: columns to select and deserialize; ``None``
            selects every column. Exactly the named columns come back --
            pk columns are NOT implicitly added.
        :ptype columns: Sequence[str] | None
        :return: list of matching row dicts; empty list when ``entity_ids`` is empty
        :rtype: list[dict[str, Any]]
        :raises ValueError: if ``columns`` is empty, or names a column
            the table's registered schema does not have
        """
        ...

    def delete_by_id(
        self,
        table: str,
        entity_id: Any,
        primary_key: str | tuple[str, ...] = "id",
    ) -> None:
        """delete single row by primary key.

        :param table: target table name
        :ptype table: str
        :param entity_id: pk value (single-PK) or tuple of pk values in
            declared column order (composite-PK)
        :ptype entity_id: Any
        :param primary_key: pk column name (single-PK) or tuple of pk
            column names in declared order (composite-PK)
        :ptype primary_key: str | tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        ...

    def execute_query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a generic SELECT query, returning list of row dicts."""
        ...

    def serialize_value(self, value: Any, col_type: str) -> Any:
        """Serialize a Python value for storage based on column type hint."""
        ...

    def deserialize_field(self, value: Any, col_type: str) -> Any:
        """Deserialize a stored value back to the correct Python type."""
        ...

    def reset(self) -> None:
        """Close all connections and clear state."""
        ...

    def is_initialized(self) -> bool:
        """Return True if the backend has been initialized."""
        ...

    def has_table(self, table: str) -> bool:
        """Return True if ``table`` was registered via ``initialize()``.

        A pod's L1 backend is only ever initialized with the tables its OWN
        collections were created for (``collection_factory.create_dynamic_collection``
        calls ``initialize()`` per-table, lazily, the first time a Collection for
        that table is instantiated) -- a pod that never touches a given table's
        Collection locally never has it in its L1 cache at all, which is expected,
        not an error: a cross-pod cache-invalidation broadcast (``threetears.
        cache.invalidate``) is heard by EVERY pod regardless of which tables each
        one actually caches. Callers use this to skip a table their L1 backend was
        never told about, the same "unknown receipts are expected" treatment
        already given to an unrecognized ``Collection`` entirely.
        """
        ...
