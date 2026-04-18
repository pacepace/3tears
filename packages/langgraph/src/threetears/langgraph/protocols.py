"""Cache protocols for the three-tier checkpoint saver.

L1 and L2 are optional cache layers that sit in front of PostgreSQL.
Host applications provide implementations matching these protocols.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AsyncQueryExecutor",
    "CheckpointL1Cache",
    "CheckpointL2Cache",
    "FlushCallback",
]


@runtime_checkable
class CheckpointL1Cache(Protocol):
    """Protocol for L1 (local, fast) checkpoint cache.

    Implementations should be fast and synchronous-friendly (e.g. SQLite).
    All methods must be async to allow asyncio.to_thread() wrapping.
    Failures should raise exceptions — the checkpointer catches and degrades.
    """

    async def get(self, thread_id: str, checkpoint_ns: str) -> bytes | None:
        """Read a cached checkpoint blob. Returns None on miss."""
        ...

    async def put(self, thread_id: str, checkpoint_ns: str, data: bytes) -> None:
        """Write a checkpoint blob to cache."""
        ...

    async def delete(self, thread_id: str) -> None:
        """Delete all cached checkpoints for a thread."""
        ...


@runtime_checkable
class CheckpointL2Cache(Protocol):
    """Protocol for L2 (distributed, shared) checkpoint cache.

    Implementations should be network-backed (e.g. NATS KV, Redis).
    Uses simple key-value semantics with string keys and bytes values.
    Failures should raise exceptions — the checkpointer catches and degrades.
    """

    async def get(self, bucket: str, key: str) -> bytes | None:
        """Read a value from the cache. Returns None on miss."""
        ...

    async def put(self, bucket: str, key: str, value: bytes) -> None:
        """Write a value to the cache."""
        ...

    async def delete(self, bucket: str, key: str) -> None:
        """Delete a key from the cache."""
        ...


@runtime_checkable
class AsyncQueryExecutor(Protocol):
    """Protocol for async SQL query execution.

    abstracts the database access layer so checkpoint savers can work
    with any backend: direct asyncpg, NATS L3 proxy, or other transports.
    implementations return dict-like rows with string keys. row values
    are typed ``Any`` because SQL column values are dynamically typed at
    the database boundary; callers cast or validate at use sites.
    """

    async def fetch(self, query: str, *args: object) -> list[dict[str, Any]]:
        """execute query and return all matching rows as dicts."""
        ...

    async def fetchrow(self, query: str, *args: object) -> dict[str, Any] | None:
        """execute query and return first row as dict, or None if empty."""
        ...

    async def execute(self, query: str, *args: object) -> str:
        """execute statement and return status string."""
        ...


@runtime_checkable
class FlushCallback(Protocol):
    """Protocol for an optional write-buffer flush on checkpoint.

    Called after each checkpoint write to drain pending writes.
    Returns the number of items flushed.
    """

    async def __call__(self) -> int:
        """Flush pending writes. Returns count of items flushed."""
        ...
