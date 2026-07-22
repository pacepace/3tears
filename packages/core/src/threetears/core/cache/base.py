"""L1 cache backend protocol and sentinel value."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "L1Backend",
    "MISSING",
]

MISSING = object()
"""Sentinel for cache miss. Distinct from None (which is a valid cached value)."""


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
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        ...

    def select_batch(
        self,
        table: str,
        entity_ids: list[Any],
        primary_key: str | tuple[str, ...] = "id",
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
        :return: list of matching row dicts; empty list when ``entity_ids`` is empty
        :rtype: list[dict[str, Any]]
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
