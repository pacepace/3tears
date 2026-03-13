"""Data layer exceptions."""

from __future__ import annotations

from typing import Any


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
