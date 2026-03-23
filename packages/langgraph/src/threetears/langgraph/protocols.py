"""Cache protocols for the three-tier checkpoint saver.

L1 and L2 are optional cache layers that sit in front of PostgreSQL.
Host applications provide implementations matching these protocols.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


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
class FlushCallback(Protocol):
    """Protocol for an optional write-buffer flush on checkpoint.

    Called after each checkpoint write to drain pending writes.
    Returns the number of items flushed.
    """

    async def __call__(self) -> int:
        """Flush pending writes. Returns count of items flushed."""
        ...
