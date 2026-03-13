"""L1 cache backend protocol and sentinel value."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

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

    def upsert(self, table: str, data: dict[str, Any], primary_key: str = "id") -> None:
        """Insert or update a row atomically."""
        ...

    def select_by_id(self, table: str, entity_id: str, primary_key: str = "id") -> dict[str, Any] | None:
        """Select a single row by primary key, returning None on miss."""
        ...

    def select_batch(self, table: str, entity_ids: list[str], primary_key: str = "id") -> list[dict[str, Any]]:
        """Select multiple rows by primary key."""
        ...

    def delete_by_id(self, table: str, entity_id: str, primary_key: str = "id") -> None:
        """Delete a single row by primary key."""
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
