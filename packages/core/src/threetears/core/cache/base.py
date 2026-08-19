"""L1 cache backend protocol and sentinel value."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "L1Backend",
    "MISSING",
    "build_select_clause",
]

MISSING = object()
"""Sentinel for cache miss. Distinct from None (which is a valid cached value)."""


def _entry_is_fresh(
    stored_at_monotonic: float | None,
    *,
    now_monotonic: float,
    max_age_seconds: float,
) -> bool:
    """Whether a cached entry stamped at ``stored_at_monotonic`` is still within its max age.

    Shared by the age-bounded cache tiers so the rule cannot drift
    between them, the same reason :func:`build_select_clause` is shared.

    Underscored, and therefore private to ``threetears.core``, on
    purpose. A name is only as internal as its spelling makes it: absence
    from ``__all__`` restricts no import, so a sibling package could
    couple to it with nothing in the intra-family bounds recording that
    it had. Nothing outside this package needs it, and declaring it
    public would oblige the whole family to a minor bump to add a helper
    that changes no consumer's API.

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
    return ", ".join(deduped)


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

    def select_by_id(
        self,
        table: str,
        entity_id: Any,
        primary_key: str | tuple[str, ...] = "id",
        columns: Sequence[str] | None = None,
    ) -> dict[str, Any] | None:
        """select single row by primary key, returning None on miss.

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
