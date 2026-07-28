"""Data layer exceptions."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ConcurrentModificationError",
    "CorruptCacheEntry",
    "DataLayerUnavailableError",
]


class ConcurrentModificationError(Exception):
    """Raised when optimistic locking detects a concurrent modification."""

    def __init__(self, table_name: str, entity_id: Any, expected_timestamp: Any) -> None:
        self.table_name = table_name
        self.entity_id = entity_id
        self.expected_timestamp = expected_timestamp
        super().__init__(
            f"Concurrent modification on {table_name}:{entity_id} (expected date_updated={expected_timestamp})"
        )


class DataLayerUnavailableError(Exception):
    """Raised when persistence layer is unavailable."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class CorruptCacheEntry(Exception):
    """Raised when an L2 value cannot be decoded back into the types it claims to hold.

    Deliberately NOT a data-layer outage and not a caller error. L2 is a cache: an entry that
    will not decode is a corrupt cache entry, and the correct response to one is to stop
    serving it, not to fail the read. Every read path in :class:`BaseCollection` catches this
    and falls through to L3, which is authoritative.

    That is the whole reason it exists as its own type. The alternatives both looked reasonable
    and were both worse: letting the underlying ``ValueError`` propagate turns one poisoned key
    into a failed read that L1 or L3 could have served, and swallowing it to return the raw
    undecoded value hands the caller a string where it declared a ``datetime``, which fails far
    from here and usually at the database border.
    """

    def __init__(self, table_name: str, column: str, value: Any) -> None:
        self.table_name = table_name
        self.column = column
        self.value = value
        super().__init__(f"{table_name}.{column} holds a value that will not decode: {value!r}")
